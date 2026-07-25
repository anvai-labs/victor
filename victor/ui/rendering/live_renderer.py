"""LiveDisplayRenderer - Rich Live display-based stream renderer.

This renderer uses Rich's Live display for real-time markdown rendering,
suitable for interactive CLI usage.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.markup import escape as _markup_escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from victor.ui.rendering.markdown import find_safe_split, render_markdown_with_hooks

from victor.ui.rendering.metrics import StreamingMetrics

logger = logging.getLogger(__name__)

from victor.ui.rendering.utils import (
    expand_tool_output,
    format_bash_command_invocation,
    format_duration,
    format_access_mode_badge,
    format_tool_args,
    format_tool_display_name,
    get_tool_metadata_for_display,
    render_content_badge,
    render_edit_preview,
    render_file_preview,
    render_status_message,
    render_thinking_indicator,
    render_thinking_text,
    render_tool_preview,
)


class LiveDisplayRenderer:
    """Renderer using Rich Live display for interactive mode.

    This renderer manages a Live display that updates in real-time
    with markdown rendering.

    Uses shared utility functions for common rendering operations
    (file preview, edit preview, thinking display, arg formatting).

    Attributes:
        console: Rich Console for output
    """

    # Class-level constants for buffer management (Fix 3 & 5)
    _MAX_CONTENT_BUFFER_SIZE = 50_000  # 50K chars — prevents unbounded memory growth
    _THINKING_BUFFER_LIMIT = 10_000  # Discard thinking content beyond this

    def __init__(self, console: Console):
        """Initialize LiveDisplayRenderer.

        Args:
            console: Rich Console for output
        """
        self.console = console
        self._live: Live | None = None
        self._content_buffer = ""  # Single source of truth for all content
        self._is_paused = False
        self._pause_count = 0  # Depth counter for nested pause/resume
        self._pause_start_ms: float | None = None
        self._metrics = StreamingMetrics()
        self._last_tool_result: dict | None = None
        self._thinking_indicator_shown = False
        self._in_thinking_mode = False
        self._content_shown_before_pause = ""
        self._tool_section_shown = False  # Track if tool section separator shown
        # In-flight tools keyed by tool_call_id (or a synthetic key), so N
        # concurrent tools in a parallel batch each get their own row instead
        # of clobbering a single scalar slot.
        self._active_tools: dict[str, dict[str, Any]] = {}
        self._tool_seq = 0
        # During-turn status footer: cached tool-call budget. Mirrors the
        # between-turns ``bottom_toolbar`` (chat.py::_build_cli_runtime_segment)
        # so the user can see Tools used/budget WHILE a turn runs, not only
        # between turns. ``_tool_budget_resolved`` distinguishes "not yet read"
        # from "read and unset".
        self._tool_budget: int | None = None
        self._tool_budget_resolved = False
        # Live tool-output streaming (progressive terminal block)
        self._tool_progress_lines: deque[str] = deque(maxlen=12)
        self._tool_progress_active = False
        self._tool_progress_name = ""
        self._last_progress_render_ms = 0.0
        # Incremental streaming render: cached HEAD (complete markdown blocks) so
        # only the active TAIL (in-progress last block) is re-rendered each tick.
        self._rendered_head: Any = None  # cached renderable for the HEAD
        self._rendered_head_source: str = ""  # source string the HEAD was rendered from

    def start(self) -> None:
        """Start the Live display."""
        from victor.ui.rendering.log_handler import register_live_console

        self._live = Live(
            render_markdown_with_hooks(""), console=self.console, refresh_per_second=10
        )
        self._live.start()
        self._is_paused = False
        self._pause_count = 0
        self._invalidate_head_cache()
        # Route console log records through this console while the display is
        # live (they print above the region instead of tearing it).
        register_live_console(self.console)

    def get_metrics(self) -> StreamingMetrics:
        """Return accumulated streaming metrics for this session."""
        return self._metrics

    def pause(self) -> None:
        """Pause the Live display to show status output.

        Supports nested calls via depth counting — the Live display is only
        stopped on the first call, preventing unnecessary stop/restart cycles.
        """
        self._pause_count += 1
        if self._pause_count == 1 and self._live and not self._is_paused:
            self._pause_start_ms = time.monotonic() * 1000
            self._live.stop()
            self._is_paused = True
            self._content_shown_before_pause = self._content_buffer
            logger.debug("LiveDisplayRenderer: paused (depth=1)")

    def resume(self) -> None:
        """Resume the Live display with only NEW content since the outermost pause.

        Only restarts the Live display when the depth counter reaches zero,
        preventing duplication from nested pause/resume callers (e.g., thinking
        blocks that contain tool calls).
        """
        if self._pause_count <= 0:
            logger.debug("LiveDisplayRenderer: resume() called with no matching pause — ignoring")
            return
        self._pause_count -= 1
        if self._pause_count == 0 and self._is_paused:
            if self._pause_start_ms is not None:
                duration_ms = time.monotonic() * 1000 - self._pause_start_ms
                self._metrics.record_pause(duration_ms)
                self._pause_start_ms = None
            self._metrics.record_resume()
            new_content = self._content_buffer[len(self._content_shown_before_pause) :]
            self._live = Live(
                Markdown(new_content),
                console=self.console,
                refresh_per_second=10,
            )
            self._live.start()
            self._is_paused = False
            # New visible slice begins — discard the stale HEAD cache.
            self._invalidate_head_cache()
            logger.debug("LiveDisplayRenderer: resumed (depth=0)")

    # Tools that frequently take >2s — show a starting hint so users don't
    # perceive a stall during long operations. Fast tools (read, list, grep)
    # are excluded to avoid output noise.
    _SLOW_TOOL_PREFIXES = (
        "shell",
        "bash",
        "code_exec",
        "code_search",
        "web_",
        "browser",
        "docker",
        "graph_",
        "embedding_",
        "vector_",
    )

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
            tool_call_id: Provider tool_calls[].id for correlating concurrent
                start/result pairs (parallel batches)
            batch_index: 1-based position within the batch
            batch_total: Number of calls dispatched together
            execution_mode: "single" or "parallel_batch"
        """
        self._tool_seq += 1
        key = tool_call_id or f"{name}#{self._tool_seq}"
        self._active_tools[key] = {"name": name, "started": time.monotonic()}

        # Tool chrome (the section separator + the bash-style invocation line) is
        # styled with Rich markup, so it must go through the Rich-markup console
        # path — the same one the [RUNNING]/[DONE] lines use — NOT the markdown
        # content buffer, which renders Rich [tags] literally. Use the single shared
        # formatter (format_bash_command_invocation) rather than an inline copy.
        if self._live:
            self.pause()
            if not self._tool_section_shown:
                self._print_section_separator("Tool Execution")
                self._tool_section_shown = True
            self.console.print(format_bash_command_invocation(name, arguments))
            self.resume()

        # Always activate progress panel to show the spinner with [RUNNING] status
        if self._live and not self._is_paused:
            self._tool_progress_name = name
            self._tool_progress_active = True
            if len(self._active_tools) == 1:
                self._tool_progress_lines.clear()
            self._render_tool_progress()

    def _visible_content(self) -> str:
        """Post-pause slice of the content buffer (what the live panel shows)."""
        return self._content_buffer[len(self._content_shown_before_pause) :]

    def on_tool_progress(
        self,
        name: str,
        stdout: str = "",
        stderr: str = "",
        progress: float = 0.0,
        is_final: bool = False,
    ) -> None:
        """Render a live, updating terminal block from streamed tool output."""
        try:
            if not self._live or self._is_paused:
                return

            combined = (stdout or "") + (stderr or "")
            if combined:
                for line in combined.splitlines():
                    self._tool_progress_lines.append(line)
                self._tool_progress_name = name
                self._tool_progress_active = True

            now_ms = time.monotonic() * 1000
            if not is_final and (now_ms - self._last_progress_render_ms) < 100.0:
                return  # throttle to ~10 fps
            self._last_progress_render_ms = now_ms

            self._render_tool_progress()
            if is_final:
                self._clear_tool_progress_panel()
        except Exception:  # pragma: no cover - defensive; never break the stream
            logger.debug("on_tool_progress render failed", exc_info=True)

    def _get_tool_budget(self) -> int | None:
        """Resolve and cache the configured tool-call budget.

        Read once (lazily, behind the first render) from settings so repeated
        Live ticks don't re-load config. Returns None when unset/unavailable.
        """
        if not self._tool_budget_resolved:
            self._tool_budget_resolved = True
            try:
                from victor.config.settings import load_settings

                tools = getattr(load_settings(), "tools", None)
                budget = getattr(tools, "tool_call_budget", None)
                self._tool_budget = int(budget) if budget else None
            except Exception:  # pragma: no cover - never break the stream over UI
                logger.debug("tool budget resolution failed", exc_info=True)
                self._tool_budget = None
        return self._tool_budget

    def _status_widget(self) -> Text | None:
        """Persistent during-turn footer: ``Tools used/budget``.

        Mirrors the between-turns ``bottom_toolbar`` so progress is visible
        while a turn is running (the toolbar is suspended during a turn).
        Returns None before any tool has run.
        """
        if self._tool_seq <= 0:
            return None
        budget = self._get_tool_budget()
        if budget and budget > 0:
            return Text.from_markup(f"[dim]Tools {self._tool_seq}/{budget}[/]")
        return Text.from_markup(f"[dim]Tools {self._tool_seq}[/]")

    def _render_tool_progress(self) -> None:
        """Update the Live renderable to content + a live tool-output panel.

        One row per in-flight tool (parallel batches render as a small table
        of spinners), plus any streamed output lines from the most recently
        emitting tool.
        """
        if not self._live or not self._tool_progress_active:
            return

        from rich.spinner import Spinner

        now = time.monotonic()
        rows: list[Any] = []
        for info in self._active_tools.values():
            tool_label = format_tool_display_name(str(info.get("name", "tool")))
            running_text = f"[dim]• [/][bold yellow][RUNNING][/] [bold cyan]{tool_label}[/]"
            elapsed = now - float(info.get("started", now))
            if elapsed >= 3.0:
                running_text += f" [dim]{elapsed:.0f}s[/]"
            rows.append(Spinner("dots", text=running_text))
        if not rows:
            # Result arrived before any start event — fall back to one row.
            tool_label = format_tool_display_name(self._tool_progress_name)
            rows.append(
                Spinner(
                    "dots", text=f"[dim]• [/][bold yellow][RUNNING][/] [bold cyan]{tool_label}[/]"
                )
            )

        body_text = "\n".join(self._tool_progress_lines)
        if body_text:
            rows.append(Text(body_text, style="dim"))
        content = Group(*rows) if len(rows) > 1 else rows[0]

        panel = Panel(
            content,
            border_style="dim",
            box=box.MINIMAL,
            expand=False,
            padding=(0, 1),
        )
        # Persistent during-turn status footer (Tools used/budget) so progress is
        # visible while the turn runs — the between-turns toolbar is suspended here.
        parts: list[Any] = [render_markdown_with_hooks(self._visible_content()), panel]
        status = self._status_widget()
        if status is not None:
            parts.append(status)
        self._live.update(Group(*parts))

    def _clear_tool_progress_panel(self) -> None:
        """Drop the live panel and reset progress state.

        Called before the result line prints so the frozen final frame is clean.
        """
        self._tool_progress_lines.clear()
        self._tool_progress_active = False
        if self._live and not self._is_paused:
            self._live.update(render_markdown_with_hooks(self._visible_content()))

    def on_tool_result(
        self,
        name: str,
        success: bool,
        elapsed: float,
        arguments: dict[str, Any],
        error: str | None = None,
        follow_up_suggestions: list[dict[str, Any]] | None = None,
        # New parameters for preview
        original_result: str | None = None,
        preview_lines: int = 3,
        was_pruned: bool = False,
        result: Any = None,  # Alias for original_result (for compatibility)
        tool_call_id: str | None = None,
    ) -> None:
        """Handle tool execution result - print consolidated single line with preview.

        Args:
            name: Tool name
            success: Whether tool succeeded
            elapsed: Execution time in seconds
            arguments: Tool arguments
            error: Error message if failed
            original_result: Unmodified tool output (before pruning)
            preview_lines: Number of lines to show in preview
            was_pruned: Whether output was pruned before sending to LLM
            result: Tool output (alias for original_result)
        """
        from victor.config.tool_settings import get_tool_settings

        # Resolve this result's in-flight entry (by id, else first name match)
        # so concurrent tools retire their own row rather than clobbering a
        # shared slot.
        entry_key: str | None = None
        if tool_call_id and tool_call_id in self._active_tools:
            entry_key = tool_call_id
        else:
            for key, info in self._active_tools.items():
                if info.get("name") == name:
                    entry_key = key
                    break
        entry = self._active_tools.pop(entry_key, None) if entry_key else None

        # Tear down any live progress panel before the result line is committed
        # to scrollback, so the streamed block does not freeze permanently.
        if self._tool_progress_active:
            self._clear_tool_progress_panel()

        # Show tool section separator before first tool call
        if not self._tool_section_shown:
            self._print_section_separator("Tool Execution")
            self._tool_section_shown = True

        # Check if tool took a long time and show progress (if applicable)
        if entry is not None:
            tool_elapsed = time.monotonic() - float(entry.get("started", time.monotonic()))
            if tool_elapsed > 3.0:
                self._update_tool_progress(name, tool_elapsed)

        tool_settings = get_tool_settings()
        show_preview = tool_settings.tool_output_preview_enabled

        preview_output = str(result) if result is not None else original_result
        full_output = original_result or (str(result) if result is not None else None)

        self.pause()

        # Compact one-line status: ✓ tool_name [WRITE] · summary · duration.
        # One colored access badge, and only when this invocation isn't a pure
        # read — success/failure is the ✓/✗ color, not a token.
        color = "success" if success else "error"
        duration_str = format_duration(elapsed)
        icon = "✓" if success else "✗"

        # Access mode narrowed to this invocation (a code grep shows no write
        # badge; a real write still does).
        metadata = get_tool_metadata_for_display(name, arguments)
        access_mode = str(metadata.get("access_mode", ""))
        access_badge = ""
        if access_mode and access_mode != "readonly":
            access_badge = f" {format_access_mode_badge(access_mode)}"

        status_line = (
            f"[bold {color}]{icon}[/] [bold cyan]{format_tool_display_name(name)}[/]{access_badge}"
        )

        # Add result summary (file count, match count, etc.)
        if success:
            result_summary = self._extract_result_summary(name, preview_output)
            if result_summary:
                status_line += f" [dim]·[/] [green]{result_summary}[/]"

        status_line += f" [dim]·[/] [tool.time]{duration_str}[/]"

        if error:
            # Show more context for errors - up to 150 chars with better formatting
            error_text = error[:150] + "..." if len(error) > 150 else error
            status_line += f"\n[{color}]  Error: {error_text}[/]"
        self.console.print(status_line)

        # Show preview if enabled
        if show_preview and success and preview_output:
            # Calculate adaptive preview lines if enabled
            if tool_settings.tool_output_preview_adaptive:
                adaptive_lines = self._calculate_adaptive_preview_lines(
                    preview_output, error, preview_lines, tool_settings
                )
            else:
                adaptive_lines = preview_lines

            from victor.ui.rendering.tool_preview import (
                renderer as _tool_preview_renderer,
            )

            preview = _tool_preview_renderer.render(
                name, arguments, preview_output, max_lines=adaptive_lines
            )
            if preview.header or preview.lines:
                if preview.header:
                    self.console.print(f"[dim]│ {_markup_escape(preview.header)}[/]")
                if preview.lines:
                    # Check if preview contains Rich markup (e.g., formatted diffs)
                    if preview.contains_rich_markup:
                        # Don't escape markup - the lines already contain Rich formatting
                        preview_text = "\n".join(preview.lines)
                    else:
                        # Escape markup to prevent rendering issues
                        preview_text = "\n".join(
                            _markup_escape(line_text) for line_text in preview.lines
                        )

                    render_tool_preview(
                        self.console,
                        preview_text,
                        total_lines=preview.total_line_count,
                        preview_lines=adaptive_lines,
                        hotkey=tool_settings.tool_output_expand_hotkey,
                        contains_rich_markup=preview.contains_rich_markup,
                    )

        # Show pruning transparency
        if was_pruned and tool_settings.tool_output_show_transparency:
            self.console.print("[dim yellow]! Preview truncated (full output sent to model)[/]")

        # Store result for potential expansion
        self._last_tool_result = {
            "name": name,
            "success": success,
            "result": full_output or "",
            "arguments": arguments,
            "elapsed": elapsed,
        }

        # Show follow-up suggestions
        if follow_up_suggestions:
            for suggestion in follow_up_suggestions[:2]:
                if not isinstance(suggestion, dict):
                    continue
                command = suggestion.get("command")
                if not isinstance(command, str) or not command.strip():
                    continue
                self.console.print(f"[dim]  next: {command}[/]")

        self._metrics.record_tool_result()
        self.resume()

        # Other tools from the batch are still running — bring their rows back.
        if self._active_tools and self._live and not self._is_paused:
            self._tool_progress_active = True
            self._render_tool_progress()

    def on_status(self, message: str) -> None:
        """Handle status message.

        Args:
            message: Status message to display
        """
        self.pause()
        render_status_message(self.console, message)
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
        """Handle content chunk - route to appropriate handler based on state.

        All content goes into _content_buffer regardless of mode (single source
        of truth).  During thinking mode the text is also printed directly to
        console and _content_shown_before_pause is advanced so that resume()
        does not re-display thinking content in the Live panel.

        Args:
            text: Content text to append
        """
        # Always buffer — cap to prevent unbounded memory growth
        if len(self._content_buffer) + len(text) > self._MAX_CONTENT_BUFFER_SIZE:
            excess = len(self._content_buffer) + len(text) - self._MAX_CONTENT_BUFFER_SIZE
            self._content_buffer = self._content_buffer[excess:]
        self._content_buffer += text

        if self._in_thinking_mode:
            # Advance the "shown" pointer so resume() starts from here, not
            # from before thinking started — prevents thinking text appearing
            # a second time in the Live panel when normal content resumes.
            self._content_shown_before_pause = self._content_buffer
            self._handle_thinking_content(text)
        else:
            self._handle_normal_content(text)

    def _handle_thinking_content(self, text: str) -> None:
        """Handle content during thinking mode - print immediately.

        Key design decisions:
        - Print directly to console (NOT Live display which is paused)
        - Don't buffer to _thinking_buffer (eliminates duplication)
        - Don't call _live.update() (wasted cycles, display is paused)

        Args:
            text: Content text to display
        """
        render_thinking_text(self.console, text)

    def _handle_normal_content(self, text: str) -> None:
        """Handle content during normal mode - stream to Live display.

        Key design decisions:
        - Show only the post-pause slice of the buffer. Pre-pause content
          was already committed to the terminal by the previous Live session
          stop; re-rendering it makes old iterations' text reappear after
          each tool execution.
        - Track metrics for performance monitoring

        Args:
            text: Content text to append
        """
        t0 = time.monotonic() * 1000
        if self._live:
            try:
                self._update_live_incremental()
            except Exception:
                # Incremental render must never break the stream — fall back to the
                # original full-slice re-render, then reset the HEAD cache.
                logger.debug("LiveDisplayRenderer: incremental render failed; full-render fallback")
                self._invalidate_head_cache()
                visible = self._content_buffer[len(self._content_shown_before_pause) :]
                self._live.update(render_markdown_with_hooks(visible))
        duration_ms = time.monotonic() * 1000 - t0
        self._metrics.record_content_chunk(duration_ms)
        if duration_ms > 100:
            logger.debug("LiveDisplayRenderer: slow render %.1fms", duration_ms)

    @staticmethod
    def _incremental_render_enabled() -> bool:
        """Incremental (head/tail) streaming render — default on, env-gated."""
        raw = os.getenv("VICTOR_INCREMENTAL_RENDER", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _invalidate_head_cache(self) -> None:
        """Drop the cached HEAD render (new slice / fallback / reset)."""
        self._rendered_head = None
        self._rendered_head_source = ""

    def _update_live_incremental(self) -> None:
        """Render the visible slice incrementally: cache the stable HEAD (complete
        markdown blocks) and re-render only the active TAIL (in-progress last
        block) each tick. Falls back to a full re-render when disabled/empty.

        This turns per-tick cost from O(turn size) into O(one block) — the McGugan
        incremental-streaming technique applied to the existing Rich pipeline.
        """
        visible = self._content_buffer[len(self._content_shown_before_pause) :]
        if not self._incremental_render_enabled() or not visible:
            self._live.update(render_markdown_with_hooks(visible))
            return

        split = find_safe_split(visible)
        head_source = visible[:split]
        # Re-render HEAD only when its source changed (grew a complete block, or the
        # slice shifted). Keying on content (not offset) is robust to buffer trims.
        if head_source != self._rendered_head_source:
            self._rendered_head = render_markdown_with_hooks(head_source) if head_source else None
            self._rendered_head_source = head_source

        tail_render = render_markdown_with_hooks(visible[split:])
        if self._rendered_head is not None:
            self._live.update(Group(self._rendered_head, tail_render))
        else:
            self._live.update(tail_render)

    def on_thinking_content(self, text: str) -> None:
        """Display thinking content immediately during streaming.

        Design principle:
        - Thinking content from API (DeepSeek, Z.AI) is different from inline markers
        - API reasoning: print immediately, don't buffer
        - Inline markers: handled by StreamingContentFilter
        - Single responsibility: just display, don't accumulate

        Note: Delta normalization is handled in stream_response() handler,
        so this method receives only the new portion to display.

        Args:
            text: Thinking text to display (already normalized/delta-extracted)
        """
        if not text or not text.strip():
            return

        # Pause live display to show thinking content
        self.pause()
        render_thinking_text(self.console, text)
        # Resume live display
        self.resume()

    def on_thinking_start(self) -> None:
        """Show thinking indicator and pause Live display."""
        self.pause()
        # Only show indicator once per response (reset in cleanup)
        if not self._thinking_indicator_shown:
            # Add section separator for visual hierarchy
            self._print_section_separator("Thinking")
            # Show content type badge
            render_content_badge(self.console, "thinking")
            render_thinking_indicator(self.console)
            self._thinking_indicator_shown = True
        # Mark that we're in thinking mode - content will be separate
        self._in_thinking_mode = True

    def on_thinking_end(self) -> None:
        """Exit thinking state and resume Live display."""
        self._in_thinking_mode = False
        self.resume()

    def _print_section_separator(self, title: str = "") -> None:
        """Print a subtle section separator for visual hierarchy.

        Args:
            title: Optional title to display in the separator
        """
        from rich.rule import Rule

        if title:
            # Styled separator with title
            self.console.print(Rule(f"[dim]{title}[/]", style="dim"))
        else:
            # Simple separator line
            self.console.print(Rule(style="dim"))

    def _update_tool_progress(self, tool_name: str, elapsed: float) -> None:
        """Show progress indicator for long-running tools.

        Note: Caller (on_tool_result) handles pause/resume, so this method
        only prints the progress message without touching the Live display state.

        Args:
            tool_name: Name of the tool being executed
            elapsed: Time elapsed since tool started (in seconds)
        """
        if elapsed > 3.0 and int(elapsed) % 2 == 0:  # Every 2 seconds after 3s
            dots = "." * (int(elapsed) % 3 + 1)
            display_name = format_tool_display_name(tool_name)
            self.console.print(f"[dim]  {display_name} still running{dots} ({elapsed:.1f}s)[/]")

    def _extract_result_summary(self, tool_name: str, output: str | None) -> str | None:
        """Extract a meaningful summary from tool output for display.

        Args:
            tool_name: Name of the tool that was executed
            output: Tool output string

        Returns:
            Summary string like "3 matches", "file written", etc. or None
        """
        if not output:
            return None

        output_lower = output.lower()

        # Search tools - show match/file counts
        if tool_name in ("code_search", "grep", "search"):
            if "match" in output_lower or "result" in output_lower:
                # Try to extract count from output
                count_match = re.search(r"(\d+)\s+(matches?|results?|files?)", output_lower)
                if count_match:
                    return f"{count_match.group(1)} {count_match.group(2)}"
            # Check for "found X" pattern
            found_match = re.search(r"found\s+(\d+)", output_lower)
            if found_match:
                return f"{found_match.group(1)} items"
            return "completed"

        # File tools - show action taken
        if tool_name in ("write", "edit", "create_file"):
            if "written" in output_lower or "saved" in output_lower:
                return "file saved"
            return "file modified"

        # Read tools
        if tool_name in ("read", "file_read"):
            lines = len(output.splitlines())
            return f"{lines} lines"

        # Shell tools - show exit code if available
        if tool_name == "shell":
            if "exit code" in output_lower or "exited with" in output_lower:
                exit_match = re.search(r"exit\s+code:\s*(\d+)", output_lower)
                if exit_match:
                    code = exit_match.group(1)
                    if code == "0":
                        return "success"
                    return f"exit {code}"
            return "completed"

        # Git tools
        if tool_name in ("git_commit", "git_push"):
            if "committed" in output_lower or "pushed" in output_lower:
                return "committed"
            return "git operation"

        return None

    def had_tool_calls(self) -> bool:
        """Return True if at least one tool call was processed this turn.

        Tracks actual tool events (``_tool_seq``), not just whether the tool
        section chrome was rendered — in non-TTY contexts tools can run
        without the section ever being shown, and ``stream_response`` uses
        this to judge whether an empty final content is a bug.
        """
        return self._tool_section_shown or self._tool_seq > 0

    def finalize(self) -> str:
        """Finalize response and return accumulated content.

        Returns:
            Accumulated response content
        """
        from rich.markdown import Markdown
        import time

        # REMOVED: Flush thinking buffer (no longer exists - caused duplication)

        # FAIL-SAFE: Ensure content buffer is displayed to user
        # Only print the portion not already shown directly (e.g., via thinking-mode print)
        unshown = self._content_buffer[len(self._content_shown_before_pause) :]
        if unshown and not self._live:
            # Wrap final text in nice padding
            self.console.print(Padding(render_markdown_with_hooks(unshown), (1, 2)))
        elif self._live and self._content_buffer:
            # Live display exists - ensure final update
            final_content = self._content_buffer[len(self._content_shown_before_pause) :]
            if final_content.strip():
                self._live.update(Padding(render_markdown_with_hooks(final_content), (1, 2)))
                # Small delay to ensure user sees final content
                time.sleep(0.1)

        # Add section separator before final response if we have content
        if self._content_buffer.strip() and (
            self._thinking_indicator_shown or self._tool_section_shown
        ):
            self._print_section_separator("Response")
            # Show content type badge
            render_content_badge(self.console, "response")

        # Log for debugging
        logger.debug(
            f"LiveDisplayRenderer.finalize(): "
            f"content_buffer_len={len(self._content_buffer)}, "
            f"in_thinking_mode={self._in_thinking_mode}"
        )

        self.cleanup()
        return self._content_buffer

    def _generate_preview(self, text: str, num_lines: int = 3) -> str:
        """Generate a preview of the first few lines of output.

        Args:
            text: Full text to generate preview from
            num_lines: Number of lines to include in preview

        Returns:
            Preview text with ellipsis if truncated
        """
        if not text:
            return ""
        lines = text.split("\n")
        preview = "\n".join(lines[:num_lines])
        if len(lines) > num_lines:
            preview += "..."
        # Truncate long lines
        max_line_length = 120
        preview = "\n".join(
            line[:max_line_length] + "..." if len(line) > max_line_length else line
            for line in preview.split("\n")
        )
        return preview

    def _calculate_adaptive_preview_lines(
        self,
        tool_output: str,
        error: str | None,
        default_lines: int,
        tool_settings: Any,
    ) -> int:
        """Calculate adaptive preview lines based on content characteristics.

        Args:
            tool_output: The tool output text
            error: Error message if any
            default_lines: Default preview lines from caller
            tool_settings: ToolSettings instance with adaptive config

        Returns:
            Number of lines to show in preview
        """
        # If there's an error, show all output (no preview limit)
        if error:
            return len(tool_output.split("\n"))

        total_lines = len(tool_output.split("\n"))
        min_lines = tool_settings.tool_output_preview_lines_min
        max_lines = tool_settings.tool_output_preview_lines_max

        # Small outputs: show everything
        if total_lines <= 5:
            return total_lines

        # Medium outputs (5-50 lines): show moderate preview
        if 5 < total_lines <= 50:
            # Use 3-5 lines for medium outputs, bounded by min/max
            adaptive = min(5, max_lines)
            return max(min_lines, adaptive)

        # Large outputs (>50 lines): show minimal preview
        # Use 1-2 lines for large outputs, bounded by min/max
        adaptive = min(2, max_lines)
        return max(min_lines, adaptive)

    def expand_last_output(self) -> None:
        """Expand the last tool output to show full content."""
        if not self._last_tool_result:
            self.console.print("[dim]No tool output to expand[/]")
            return

        data = self._last_tool_result
        if not data["success"] or not data["result"]:
            return

        expand_tool_output(
            self.console,
            data["name"],
            data["result"],
            pause_fn=self.pause,
            resume_fn=lambda: self.resume(),
        )

    def cleanup(self) -> None:
        """Clean up the Live display."""
        from victor.ui.rendering.log_handler import unregister_live_console

        unregister_live_console()
        if self._live:
            self._live.stop()
            self._live = None
        self._is_paused = False
        self._pause_count = 0
        self._active_tools.clear()
        self._thinking_indicator_shown = False
        self._in_thinking_mode = False
        self._content_shown_before_pause = ""
        self._last_tool_result = None
        self._tool_section_shown = False  # Reset tool section flag
