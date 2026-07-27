"""Buffered renderer for non-streaming mode.

Collects streaming events and renders them at completion.
Used for --no-stream mode to capture tool calls, reasoning,
and final content that would otherwise be swallowed.
"""

from __future__ import annotations

from typing import Any

from victor.ui.rendering.utils import (
    format_duration,
    format_tool_args,
    format_tool_display_name,
    render_tool_preview,
)


class BufferedRenderer:
    """Collects streaming events and renders them at completion.

    Implements the StreamRenderer protocol but buffers all output
    until flush() or finalize() is called.
    """

    def __init__(
        self,
        show_reasoning: bool = False,
        plain: bool = False,
        user_message: str = "",
    ):
        self._show_reasoning = show_reasoning
        self._plain = plain
        self._user_message = user_message
        self._tool_calls: list[dict[str, Any]] = []
        self._reasoning_chunks: list[str] = []
        self._content_chunks: list[str] = []
        self._statuses: list[str] = []

    def start(self) -> None:
        """Start the buffered display."""
        pass

    def pause(self) -> None:
        """Pause (no-op for buffered)."""
        pass

    def resume(self) -> None:
        """Resume (no-op for buffered)."""
        pass

    def on_tool_start(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
        batch_index: int | None = None,
        batch_total: int | None = None,
        execution_mode: str | None = None,
    ) -> None:
        """Record tool execution start."""
        self._tool_calls.append({"name": name, "arguments": arguments, "result": None})

    def on_tool_progress(
        self,
        name: str,
        stdout: str = "",
        stderr: str = "",
        progress: float = 0.0,
        is_final: bool = False,
    ) -> None:
        """No-op: buffered mode shows only final output, not live progress."""

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
        result: Any = None,
        tool_call_id: str | None = None,
    ) -> None:
        """Record tool execution result."""
        tool_output = str(result) if result is not None else ""
        full_output = str(original_result) if original_result is not None else tool_output
        result_data = {
            "success": success,
            "elapsed": elapsed,
            "error": error,
            "was_pruned": was_pruned,
            "output": tool_output,
            "full_output": full_output,
            "follow_up_suggestions": follow_up_suggestions or [],
        }

        # Update last matching tool call with result
        for tc in reversed(self._tool_calls):
            if tc["name"] == name and tc["result"] is None:
                tc["result"] = result_data
                break
        else:
            self._tool_calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "result": result_data,
                }
            )

    def on_status(self, message: str) -> None:
        """Record status message."""
        self._statuses.append(message)

    def on_file_preview(self, path: str, content: str) -> None:
        """Record file preview (included in content)."""
        pass

    def on_edit_preview(self, path: str, diff: str) -> None:
        """Record edit preview (included in content)."""
        pass

    def on_content(self, text: str) -> None:
        """Buffer content chunk."""
        self._content_chunks.append(text)

    def on_thinking_content(self, text: str) -> None:
        """Buffer thinking content if show_reasoning is enabled."""
        if self._show_reasoning:
            self._reasoning_chunks.append(text)

    def on_thinking_start(self) -> None:
        """Handle transition into thinking state."""
        pass

    def on_thinking_end(self) -> None:
        """Handle transition out of thinking state."""
        pass

    def had_tool_calls(self) -> bool:
        """Return True if at least one tool call was processed this turn."""
        return bool(self._tool_calls)

    def statuses(self) -> list[str]:
        """Return buffered status messages (includes surfaced provider errors)."""
        return list(self._statuses)

    def finalize(self) -> str:
        """Return accumulated content."""
        content = "".join(self._content_chunks)
        if self._user_message:
            from victor.framework.task.direct_response import (
                normalize_direct_response_output,
            )

            # Compatibility fallback for callers that still bypass framework-owned
            # stream normalization and only hand the renderer raw content.
            content = normalize_direct_response_output(self._user_message, content)
        return content

    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def flush(self, console: Any) -> None:
        """Print collected output to console.

        Args:
            console: Rich Console instance for output
        """
        from rich.markdown import Markdown

        # Print tool calls summary
        if self._tool_calls:
            for tc in self._tool_calls:
                result = tc.get("result", {})
                display_name = format_tool_display_name(tc["name"])
                if result:
                    icon = "[green]✓[/]" if result.get("success") else "[red]✗[/]"
                    elapsed = result.get("elapsed", 0)
                    args_display = format_tool_args(tc.get("arguments", {}))
                    status_line = f"{icon} [bold]{display_name}[/]"
                    if args_display:
                        status_line += f" [dim]{args_display}[/]"
                    status_line += f" [dim]• {format_duration(elapsed)}[/]"
                    if result.get("error"):
                        status_line += f" [red]{str(result['error'])[:80]}[/]"
                    console.print(status_line)

                    output = result.get("output")
                    if output and result.get("success"):
                        preview_lines = 3
                        preview_text = "\n".join(output.splitlines()[:preview_lines])
                        if preview_text:
                            render_tool_preview(
                                console,
                                preview_text,
                                total_lines=len(output.splitlines()),
                                preview_lines=preview_lines,
                                hotkey="^O",
                            )
                else:
                    console.print(f"[blue]•[/] [bold]{display_name}[/] [dim]pending[/]")

        # Print statuses — this is where surfaced provider errors land
        # (EventDispatcher routes ERROR events to on_status), so skipping them
        # hides the real failure from non-interactive output.
        for status in self._statuses:
            console.print(status)

        # Print reasoning if --show-reasoning
        if self._reasoning_chunks:
            reasoning_text = "".join(self._reasoning_chunks)
            console.print(f"\n[dim italic]{reasoning_text}[/]\n")

        # Print final content
        content = self.finalize()
        if content.strip():
            if self._plain:
                console.print(content)
            else:
                console.print(Markdown(content))
