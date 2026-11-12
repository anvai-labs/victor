"""FormatterRenderer - OutputFormatter-based stream renderer.

This renderer delegates to OutputFormatter for oneshot mode rendering,
suitable for non-interactive CLI usage.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

from rich.console import Console

from victor.ui.rendering.metrics import StreamingMetrics

logger = logging.getLogger(__name__)

from victor.ui.rendering.utils import (
    expand_tool_output,
    format_tool_display_name,
    render_edit_preview,
    render_file_preview,
    render_thinking_indicator,
    render_thinking_text,
)

if TYPE_CHECKING:
    from victor.ui.rendering.utils import Severity
    from victor.ui.output_formatter import OutputFormatter


class FormatterRenderer:
    """Renderer using OutputFormatter for oneshot mode.

    This renderer delegates to OutputFormatter methods which handle
    the visual presentation with proper formatting.

    Uses shared utility functions for common rendering operations
    (file preview, edit preview, thinking display).

    Attributes:
        formatter: OutputFormatter instance for formatting output
        console: Rich Console for direct output
    """

    def __init__(self, formatter: OutputFormatter, console: Console):
        """Initialize FormatterRenderer.

        Args:
            formatter: OutputFormatter to delegate to
            console: Rich Console for direct output
        """
        self.formatter = formatter
        self.console = console
        self._content_buffer = ""
        self._pause_count = 0  # Depth counter for nested pause/resume
        self._last_tool_result: dict | None = None
        self._metrics = StreamingMetrics()
        self._pause_start_ms: float | None = None  # Wall time when outermost pause began

    def start(self) -> None:
        """Start streaming mode in the formatter."""
        self.formatter.start_streaming()

    def get_metrics(self) -> StreamingMetrics:
        """Return accumulated streaming metrics for this session."""
        return self._metrics

    def pause(self) -> None:
        """End streaming to allow status output.

        Supports nested calls via depth counting — the underlying formatter
        is only paused on the first call, preventing redundant end/start cycles.
        """
        self._pause_count += 1
        if self._pause_count == 1:
            self._pause_start_ms = time.monotonic() * 1000
            logger.debug("FormatterRenderer: paused (depth=1)")

    def resume(self) -> None:
        """Resume streaming mode in the formatter.

        Only resumes the underlying formatter when the depth counter reaches zero,
        allowing nested pause/resume callers to unwind correctly.
        """
        if self._pause_count <= 0:
            logger.warning("FormatterRenderer: resume() called with no matching pause — ignoring")
            return
        self._pause_count -= 1
        if self._pause_count == 0:
            if self._pause_start_ms is not None:
                duration_ms = time.monotonic() * 1000 - self._pause_start_ms
                self._metrics.record_pause(duration_ms)
                self._pause_start_ms = None
            self._metrics.record_resume()
            self.formatter.start_streaming(preserve_buffer=True)
            logger.debug("FormatterRenderer: resumed (depth=0)")

    def on_tool_start(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
        batch_index: int | None = None,
        batch_total: int | None = None,
        execution_mode: str | None = None,
    ) -> None:
        """Handle tool execution start - store info for result display.

        Args:
            name: Tool name
            arguments: Tool arguments
        """
        # Store tool info without printing "running" status to avoid duplicate output
        self._last_tool_start = {"name": name, "arguments": arguments}
        self.formatter.tool_start(name, arguments)

    def on_tool_progress(
        self,
        name: str,
        stdout: str = "",
        stderr: str = "",
        progress: float = 0.0,
        is_final: bool = False,
    ) -> None:
        """No-op: formatter mode renders consolidated output, not live progress."""

    def on_tool_result(
        self,
        name: str,
        success: bool,
        elapsed: float,
        arguments: dict[str, Any],
        error: str | None = None,
        follow_up_suggestions: list[dict[str, Any]] | None = None,
        was_pruned: bool = False,
        original_result: Any = None,
        result: Any = None,  # Tool output for preview
        tool_call_id: str | None = None,
    ) -> None:
        """Handle tool execution result.

        Args:
            name: Tool name
            success: Whether tool succeeded
            elapsed: Execution time in seconds
            arguments: Tool arguments
            error: Error message if failed
            follow_up_suggestions: Optional follow-up suggestions
            result: Tool output (for preview display)
        """
        preview_output = str(result) if result is not None else None
        full_output = original_result or preview_output
        self._last_tool_result = {
            "name": name,
            "success": success,
            "result": full_output or "",
            "arguments": arguments,
            "elapsed": elapsed,
        }
        self._metrics.record_tool_result()
        self.pause()
        formatter_kwargs = {
            "tool_name": name,
            "success": success,
            "error": error,
            "follow_up_suggestions": follow_up_suggestions,
        }
        if full_output is not None:
            formatter_kwargs["original_result"] = full_output
        if preview_output is not None:
            formatter_kwargs["result"] = preview_output
        if was_pruned:
            formatter_kwargs["was_pruned"] = True
        self.formatter.tool_result(**formatter_kwargs)
        self.resume()

    def on_status(self, message: str, severity: Optional["Severity"] = None) -> None:
        """Handle status message.

        Args:
            message: Status message to display
        """
        self.pause()
        self.formatter.status(message, severity)
        self.resume()

    def on_file_preview(self, path: str, content: str) -> None:
        """Handle file content preview.

        Args:
            path: File path
            content: File content to preview
        """
        self.pause()
        render_file_preview(self.console, path, content)
        self.resume()

    def on_edit_preview(self, path: str, diff: str) -> None:
        """Handle edit diff preview.

        Args:
            path: File path being edited
            diff: Diff content to display
        """
        self.pause()
        render_edit_preview(self.console, path, diff)
        self.resume()

    def on_content(self, text: str) -> None:
        """Handle content chunk.

        Args:
            text: Content text to append
        """
        t0 = time.monotonic() * 1000
        self._content_buffer += text
        self.formatter.stream_chunk(text)
        duration_ms = time.monotonic() * 1000 - t0
        self._metrics.record_content_chunk(duration_ms)
        if duration_ms > 100:
            logger.debug("FormatterRenderer: slow render %.1fms", duration_ms)

    def on_thinking_content(self, text: str) -> None:
        """Render thinking content as dimmed/italic text.

        Args:
            text: Thinking text to display
        """
        # Don't add to content buffer (thinking is ephemeral)
        render_thinking_text(self.console, text)

    def on_thinking_start(self) -> None:
        """Show thinking indicator."""
        self.pause()
        render_thinking_indicator(self.console)

    def on_thinking_end(self) -> None:
        """Show end of thinking."""
        self.console.print()  # Newline after thinking
        self.resume()

    def had_tool_calls(self) -> bool:
        """Return True if at least one tool call was processed this turn."""
        return self._last_tool_result is not None

    def finalize(self) -> str:
        """Finalize response and return accumulated content.

        Returns:
            Accumulated response content
        """
        self.formatter.response(content=self._content_buffer)
        return self._content_buffer

    def expand_last_output(self) -> None:
        """Expand the last tool output to show full content."""
        if not self._last_tool_result:
            self.console.print("[dim]No tool output to expand[/]")
            return

        data = self._last_tool_result
        if not data["success"] or not data["result"]:
            return

        expand_tool_output(self.console, data["name"], data["result"])

    def cleanup(self) -> None:
        """Clean up resources."""
        self._last_tool_result = None
