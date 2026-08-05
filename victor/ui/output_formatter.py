from __future__ import annotations

# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified output formatting for CLI automation and interactive use.

This module provides a consistent output interface that works seamlessly for:
- Human-interactive use (Rich formatted output)
- Automation/scripting (JSON or plain text output)
- Benchmark harness integration (structured machine-readable output)

Usage:
    # For automation
    formatter = OutputFormatter(mode=OutputMode.JSON)
    formatter.response(content="Hello", tool_calls=[...])

    # For interactive
    formatter = OutputFormatter(mode=OutputMode.RICH)
    formatter.response(content="Hello")

    # For code extraction
    formatter = OutputFormatter(mode=OutputMode.CODE_ONLY)
    formatter.response(content="Here's the code:\n```python\ndef foo(): pass\n```")
"""

import json
import re
import sys
import time
from rich.markup import escape as _markup_escape


def _colorize_diff_line(line: str) -> str:
    """Return Rich markup for a unified-diff line (colors + and - lines)."""
    escaped = _markup_escape(line)
    if line.startswith("+"):
        return f"[green]{escaped}[/]"
    if line.startswith("-"):
        return f"[red]{escaped}[/]"
    if line.startswith("@@"):
        return f"[cyan]{escaped}[/]"
    return f"[dim]{escaped}[/]"


from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TextIO, Tuple

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from victor.ui.rendering.markdown import render_markdown_with_hooks
from victor.ui.rendering.utils import (
    format_duration,
    format_tool_args,
    format_tool_display_name,
    render_status_message,
    render_tool_preview,
)
from victor.ui.rendering.tool_preview import renderer as _tool_preview_renderer
from victor.ui.theme import victor_theme


class OutputMode(Enum):
    """Output formatting modes."""

    RICH = "rich"  # Full Rich formatting (boxes, colors, emoji)
    PLAIN = "plain"  # Clean text without formatting
    JSON = "json"  # Single JSON object at end
    JSONL = "jsonl"  # Streaming JSON Lines
    CODE_ONLY = "code_only"  # Extract and return only code blocks


@dataclass
class OutputConfig:
    """Configuration for output formatting."""

    mode: OutputMode = OutputMode.RICH
    quiet: bool = False  # Suppress status messages
    show_tools: bool = True  # Show tool execution info
    show_thinking: bool = True  # Show thinking/reasoning
    show_metrics: bool = True  # Show performance metrics
    stream: bool = True  # Stream output as it arrives
    stdout: TextIO = field(default_factory=lambda: sys.stdout)
    stderr: TextIO = field(default_factory=lambda: sys.stderr)


class OutputFormatter:
    """Unified output formatter for CLI.

    Provides consistent output across different use cases:
    - Interactive human use (Rich formatting)
    - Automation/scripting (JSON/plain text)
    - Benchmark integration (code extraction)
    """

    def __init__(self, config: Optional[OutputConfig] = None):
        """Initialize formatter with configuration.

        Args:
            config: Output configuration. Defaults to Rich mode.
        """
        self.config = config or OutputConfig()
        self._console = Console(
            file=self.config.stdout,
            force_terminal=False,
            theme=victor_theme,
        )
        self._content_buffer: List[str] = []
        self._tool_calls: List[Dict[str, Any]] = []
        self._metrics: Dict[str, Any] = {}
        # Live markdown rendering for streaming in RICH mode
        self._live: Optional[Live] = None
        self._stream_buffer: str = ""
        # Tool timing tracking for compact output
        self._pending_tool: Optional[Tuple[str, Dict[str, Any], float]] = None
        # Tool result storage for expansion
        self._last_tool_result: Optional[Dict[str, Any]] = None

    @property
    def mode(self) -> OutputMode:
        """Get current output mode."""
        return self.config.mode

    def status(self, message: str) -> None:
        """Output a status message (non-content).

        Status messages are suppressed in quiet mode and JSON modes.

        Args:
            message: Status message to display
        """
        if self.config.quiet:
            return

        if self.config.mode == OutputMode.RICH:
            render_status_message(self._console, message)
        elif self.config.mode == OutputMode.PLAIN:
            print(f"# {message}", file=self.config.stderr)
            # Flush stderr immediately to ensure status appears before next content
            self.config.stderr.flush()
        # JSON modes: skip status messages

    def info(self, message: str) -> None:
        """Output an informational message.

        Info messages are displayed in all non-JSON modes and not suppressed in quiet mode.

        Args:
            message: Info message to display
        """
        if self.config.mode == OutputMode.RICH:
            self._console.print(f"[cyan]{message}[/]")
        elif self.config.mode == OutputMode.PLAIN:
            print(message, file=self.config.stderr)
            # Flush stderr immediately to ensure info appears before next content
            self.config.stderr.flush()
        # JSON modes: skip info messages

    def error(self, message: str, details: Optional[str] = None) -> None:
        """Output an error message.

        Args:
            message: Error message
            details: Optional detailed error info
        """
        if self.config.mode == OutputMode.JSON:
            error_obj = {"error": message}
            if details:
                error_obj["details"] = details
            print(json.dumps(error_obj), file=self.config.stdout)
        elif self.config.mode == OutputMode.JSONL:
            error_obj = {"type": "error", "message": message}
            if details:
                error_obj["details"] = details
            print(json.dumps(error_obj), file=self.config.stdout)
        elif self.config.mode == OutputMode.RICH:
            self._console.print(f"[bold red]Error:[/] {message}")
            if details:
                self._console.print(f"[dim]{details}[/]")
        else:  # PLAIN or CODE_ONLY
            print(f"Error: {message}", file=self.config.stderr)
            if details:
                print(details, file=self.config.stderr)
            self.config.stderr.flush()

    def _format_args(self, arguments: Dict[str, Any], max_width: int = 80) -> str:
        """Format tool arguments for display.

        Args:
            arguments: Tool arguments
            max_width: Maximum total width for args string (default 80 to use more line width)

        Returns:
            Formatted args string like "path='/file.py', limit=100"
        """
        return format_tool_args(arguments, max_width=max_width)

    def tool_start(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Indicate tool execution has started.

        Stores timing info for compact output in tool_result.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
        """
        # Store timing info for later
        self._pending_tool = (tool_name, arguments, time.time())

        if not self.config.show_tools:
            return

        if self.config.mode == OutputMode.JSONL:
            print(
                json.dumps(
                    {
                        "type": "tool_start",
                        "tool": tool_name,
                        "arguments": arguments,
                    }
                ),
                file=self.config.stdout,
            )
        # RICH and PLAIN modes defer output to tool_result for single-line format

    def tool_result(
        self,
        tool_name: str,
        success: bool,
        result: Optional[str] = None,
        error: Optional[str] = None,
        follow_up_suggestions: Optional[List[Dict[str, Any]]] = None,
        # New parameters for preview
        original_result: Optional[str] = None,
        preview_lines: int = 3,
        was_pruned: bool = False,
        show_preview: bool = True,
    ) -> None:
        """Record tool execution result and output compact single-line format with preview.

        Args:
            tool_name: Name of the tool
            success: Whether tool succeeded
            result: Tool result (if success)
            error: Error message (if failed)
            original_result: Unmodified tool output (before pruning)
            preview_lines: Number of lines to show in preview
            was_pruned: Whether output was pruned before sending to LLM
            show_preview: Whether to show preview (controlled by settings)
        """
        preview_result = result or original_result
        full_result = original_result or result

        tool_record = {
            "tool": tool_name,
            "success": success,
        }
        if preview_result:
            tool_record["result"] = preview_result[:500]  # Truncate for JSON
        if error:
            tool_record["error"] = error
        if follow_up_suggestions:
            tool_record["follow_up_suggestions"] = [
                suggestion.get("command")
                for suggestion in follow_up_suggestions[:3]
                if isinstance(suggestion, dict) and suggestion.get("command")
            ]

        self._tool_calls.append(tool_record)

        # Calculate duration from pending tool; retain arguments for preview
        duration: Optional[float] = None
        duration_str = ""
        args_str = ""
        _tool_arguments: Dict[str, Any] = {}
        if self._pending_tool and self._pending_tool[0] == tool_name:
            _, _tool_arguments, start_time = self._pending_tool
            duration = time.time() - start_time
            duration_str = f"({duration:.1f}s)"
            args_str = self._format_args(_tool_arguments)
            self._pending_tool = None

        if not self.config.show_tools:
            return

        if self.config.mode == OutputMode.JSONL:
            print(
                json.dumps(
                    {
                        "type": "tool_result",
                        **tool_record,
                    }
                ),
                file=self.config.stdout,
            )
        elif self.config.mode == OutputMode.RICH:
            status_icon = "[green]✓[/]" if success else "[red]✗[/]"
            display_name = format_tool_display_name(tool_name)
            status_line = f"{status_icon} [bold]{display_name}[/]"
            if args_str:
                status_line += f" [dim]{args_str}[/]"
            if duration_str:
                status_line += f" [dim]• {format_duration(duration)}[/]"
            if error:
                status_line += f" [red]{error[:80]}[/]"
            self._console.print(status_line)

            # Show per-tool preview using the strategy renderer.
            # Note: condition does NOT require original_result — some strategies
            # (e.g. DiffPreviewStrategy) generate output purely from arguments.
            if show_preview and success:
                try:
                    from victor.config.tool_settings import get_tool_settings

                    hotkey = get_tool_settings().tool_output_expand_hotkey
                except Exception:
                    hotkey = "^O"

                preview = _tool_preview_renderer.render(
                    tool_name,
                    _tool_arguments,
                    preview_result or "",
                    max_lines=preview_lines,
                )
                if preview.header or preview.lines:
                    if preview.header:
                        self._console.print(f"[dim]│ {_markup_escape(preview.header)}[/]")
                    if preview.lines:
                        if preview.syntax_hint == "diff":
                            render_tool_preview(
                                self._console,
                                "\n".join(_colorize_diff_line(line) for line in preview.lines),
                                total_lines=preview.total_line_count,
                                preview_lines=preview_lines,
                                hotkey=hotkey,
                            )
                        else:
                            render_tool_preview(
                                self._console,
                                "\n".join(_markup_escape(line) for line in preview.lines),
                                total_lines=preview.total_line_count,
                                preview_lines=preview_lines,
                                hotkey=hotkey,
                            )

            # Show pruning transparency if enabled
            if was_pruned and self.config.show_tools:
                try:
                    from victor.config.tool_settings import get_tool_settings

                    tool_settings = get_tool_settings()
                    if tool_settings.tool_output_show_transparency:
                        self._console.print(
                            "[dim yellow]! Preview truncated (full output sent to model)[/]"
                        )
                except Exception:
                    # Fallback if settings not available
                    self._console.print(
                        "[dim yellow]! Preview truncated (full output sent to model)[/]"
                    )

            # Store full result for potential expansion
            self._last_tool_result = {
                "tool_name": tool_name,
                "arguments": _tool_arguments,
                "success": success,
                "result": full_result,
                "pruned_result": preview_result,
                "was_pruned": was_pruned,
                "error": error,
                "follow_up_suggestions": follow_up_suggestions,
            }

            if success and follow_up_suggestions:
                for suggestion in follow_up_suggestions[:2]:
                    if not isinstance(suggestion, dict):
                        continue
                    command = suggestion.get("command")
                    if isinstance(command, str) and command.strip():
                        self._console.print(f"[dim]  next: {command}[/]")
        elif self.config.mode == OutputMode.PLAIN:
            # Compact single-line format: ✓ tool_name(args) (X.XXs)
            status_icon = "✓" if success else "✗"
            display_name = format_tool_display_name(tool_name)
            base = f"{display_name}({args_str})" if args_str else display_name
            time_display = f" ({duration_str.strip('()')})" if duration_str else ""
            error_display = f" {error[:40]}" if error else ""
            print(
                f"{status_icon} {base}{time_display}{error_display}",
                file=self.config.stderr,
            )
            if success and follow_up_suggestions:
                for suggestion in follow_up_suggestions[:2]:
                    if not isinstance(suggestion, dict):
                        continue
                    command = suggestion.get("command")
                    if isinstance(command, str) and command.strip():
                        print(f"  next: {command}", file=self.config.stderr)
            # Flush stderr immediately to ensure tool output appears before next content
            self.config.stderr.flush()

    def _generate_preview(self, text: str, num_lines: int = 3) -> str:
        """Generate a preview of the first few lines of output.

        Args:
            text: Full tool output
            num_lines: Number of lines to include in preview

        Returns:
            Preview string with first N lines
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

    def expand_last_tool_output(self) -> None:
        """Expand the last tool output to show full content.

        Displays the full tool output in a Rich panel with syntax highlighting.
        """
        if not hasattr(self, "_last_tool_result") or not self._last_tool_result:
            self._console.print("[dim]No tool output to expand[/]")
            return

        data = self._last_tool_result
        if not data["success"]:
            return

        from rich.panel import Panel
        from rich.syntax import Syntax

        content = data["result"]
        tool_name = data["tool_name"]
        display_name = format_tool_display_name(tool_name)

        # Try syntax highlighting for code-like content
        if self.config.mode == OutputMode.RICH:
            try:
                # Extract file extension from tool name if possible
                ext = tool_name.split("_")[-1] if "_" in tool_name else "txt"
                # Map common tool names to extensions
                ext_map = {
                    "read": "py",
                    "grep": "txt",
                    "code_search": "py",
                    "cat": "txt",
                }
                ext = ext_map.get(tool_name, ext)

                syntax = Syntax(content, ext, theme="monokai", line_numbers=True, word_wrap=True)
                self._console.print(
                    Panel(
                        syntax,
                        title=f"[bold]{display_name}[/] - Full Output",
                        border_style="blue",
                    )
                )
            except Exception:
                # Fallback to plain panel
                self._console.print(
                    Panel(
                        content,
                        title=f"[bold]{display_name}[/] - Full Output",
                        border_style="blue",
                    )
                )
        else:
            # Plain mode
            print(f"\n--- Full Output: {display_name} ---")
            print(content)
            print("--- End of Output ---\n")

    def thinking(self, content: str) -> None:
        """Output thinking/reasoning content.

        Args:
            content: Thinking content
        """
        if not self.config.show_thinking:
            return

        if self.config.mode == OutputMode.JSONL:
            print(
                json.dumps(
                    {
                        "type": "thinking",
                        "content": content,
                    }
                ),
                file=self.config.stdout,
            )
        elif self.config.mode == OutputMode.RICH:
            self._console.print(
                Panel(
                    content,
                    title="Thinking",
                    border_style="dim",
                )
            )
        elif self.config.mode == OutputMode.PLAIN:
            print(f"# Thinking: {content[:200]}...", file=self.config.stderr)
            self.config.stderr.flush()

    def start_streaming(self, preserve_buffer: bool = False) -> None:
        """Start streaming mode with live markdown rendering (RICH mode only).

        Call this before the first stream_chunk() to enable live markdown rendering.

        Args:
            preserve_buffer: If True, don't reset the stream buffer (for resuming)
        """
        if self.config.mode == OutputMode.RICH and self.config.stream:
            if not preserve_buffer:
                self._stream_buffer = ""
            # Don't start a new Live if one is already active
            if self._live is None:
                self._live = Live(
                    Markdown(self._stream_buffer),
                    console=self._console,
                    refresh_per_second=10,
                )
                self._live.start()

    def end_streaming(self, finalize: bool = True) -> None:
        """End streaming mode and finalize live markdown rendering.

        Call this after the last stream_chunk() if start_streaming() was called.

        Args:
            finalize: If True, stop the Live and clear buffer. If False, just pause.
        """
        if self._live is not None and finalize:
            self._live.stop()
            self._live = None
            self._stream_buffer = ""

    def stream_chunk(self, content: str) -> None:
        """Output a streaming content chunk.

        Args:
            content: Content chunk
        """
        self._content_buffer.append(content)

        if not self.config.stream:
            return

        if self.config.mode == OutputMode.JSONL:
            print(
                json.dumps(
                    {
                        "type": "chunk",
                        "content": content,
                    }
                ),
                file=self.config.stdout,
            )
            self.config.stdout.flush()
        elif self.config.mode == OutputMode.RICH:
            # Use live markdown rendering if active
            if self._live is not None:
                self._stream_buffer += content
                self._live.update(Markdown(self._stream_buffer))
            else:
                # Fallback to raw print if live not started
                print(content, end="", file=self.config.stdout, flush=True)
        elif self.config.mode == OutputMode.PLAIN:
            print(content, end="", file=self.config.stdout, flush=True)
        # CODE_ONLY and JSON: buffer only, output at end

    def response(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        usage: Optional[Dict[str, int]] = None,
        model: Optional[str] = None,
    ) -> str:
        """Output final response.

        This is called at the end of a response. For streaming, content
        may already have been output via stream_chunk().

        Args:
            content: Full response content (or None if streamed)
            tool_calls: Tool calls made during response
            usage: Token usage statistics
            model: Model that generated the response

        Returns:
            The output that was written (useful for testing)
        """
        # Use buffered content if not provided
        if content is None:
            content = "".join(self._content_buffer)

        # Merge tool calls
        if tool_calls:
            self._tool_calls.extend(tool_calls)

        # Store metrics
        if usage:
            self._metrics["usage"] = usage
        if model:
            self._metrics["model"] = model

        output = ""

        if self.config.mode == OutputMode.JSON:
            response_obj = {
                "content": content,
                "tool_calls": self._tool_calls if self._tool_calls else None,
                "metrics": self._metrics if self._metrics else None,
            }
            # Remove None values
            response_obj = {k: v for k, v in response_obj.items() if v is not None}
            output = json.dumps(response_obj, indent=2)
            print(output, file=self.config.stdout)

        elif self.config.mode == OutputMode.JSONL:
            response_obj = {
                "type": "response",
                "content": content,
            }
            if self._tool_calls:
                response_obj["tool_calls"] = self._tool_calls
            if self._metrics:
                response_obj["metrics"] = self._metrics
            output = json.dumps(response_obj)
            print(output, file=self.config.stdout)

        elif self.config.mode == OutputMode.CODE_ONLY:
            output = self._extract_code(content)
            print(output, file=self.config.stdout)

        elif self.config.mode == OutputMode.PLAIN:
            # For plain mode, content was already streamed
            if not self.config.stream:
                output = content
                print(output, file=self.config.stdout)
            else:
                output = content
                print("", file=self.config.stdout)  # Final newline

        else:  # RICH
            # End live markdown streaming if active
            self.end_streaming()

            # For Rich mode, content was already streamed with markdown rendering
            if not self.config.stream:
                output = content
                self._console.print(render_markdown_with_hooks(content))
            else:
                output = content
                # Final newline already handled by live display

            # Show metrics if enabled
            if self.config.show_metrics and self._metrics:
                self._show_metrics()

        # Reset buffers
        self._content_buffer = []
        self._tool_calls = []
        self._metrics = {}

        return output

    def _show_metrics(self) -> None:
        """Show performance metrics in Rich mode."""
        if not self._metrics:
            return

        parts = []
        if "usage" in self._metrics:
            usage = self._metrics["usage"]
            if "prompt_tokens" in usage:
                parts.append(f"in: {usage['prompt_tokens']}")
            if "completion_tokens" in usage:
                parts.append(f"out: {usage['completion_tokens']}")
        if "model" in self._metrics:
            parts.append(f"model: {self._metrics['model']}")

        if parts:
            self._console.print(f"[dim]{' | '.join(parts)}[/]")

    def _extract_code(self, content: str) -> str:
        """Extract code blocks from content.

        Handles:
        - Markdown code blocks (```python ... ```)
        - Tool call format (<parameter=content>...</tool_call>)
        - Indented code blocks
        - Raw function definitions

        Args:
            content: Full response content

        Returns:
            Extracted code, or empty string if none found
        """
        # Try tool call format first (model hallucinated tool calls)
        # Format: <parameter=content> ... </tool_call> or <parameter=content>...</parameter>
        tool_content_pattern = r"<parameter=content>\s*(.*?)\s*(?:</tool_call>|</parameter>|\Z)"
        tool_matches = re.findall(tool_content_pattern, content, re.DOTALL | re.IGNORECASE)
        if tool_matches:
            # Extract the code from tool call content
            extracted = tool_matches[0].strip()
            # The content might still have function definitions, return as-is if it looks like code
            if extracted and (
                "def " in extracted or "class " in extracted or "import " in extracted
            ):
                return extracted

        # Try markdown code blocks
        code_block_pattern = r"```(?:python)?\n?(.*?)```"
        matches = re.findall(code_block_pattern, content, re.DOTALL)

        if matches:
            # Prefer the first code block containing a function definition
            for match in matches:
                if "def " in match:
                    return match.strip()
            # Otherwise return the first code block (more likely to be the function)
            return matches[0].strip()

        # Try to find function definitions directly
        func_pattern = r"(def\s+\w+\s*\([^)]*\).*?)(?=\ndef\s|\nclass\s|\Z)"
        matches = re.findall(func_pattern, content, re.DOTALL)

        if matches:
            # Return the first function definition (primary implementation)
            return matches[0].strip()

        # Try indented code blocks (4 spaces or 1 tab)
        lines = content.split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if line.startswith("    ") or line.startswith("\t"):
                code_lines.append(line)
                in_code = True
            elif in_code and line.strip() == "":
                code_lines.append("")
            elif in_code:
                break

        if code_lines:
            # Remove common leading whitespace
            return "\n".join(
                line[4:] if line.startswith("    ") else line for line in code_lines
            ).strip()

        return ""


class InputReader:
    """Unified input reader for CLI.

    Handles different input sources:
    - Command line argument
    - Stdin (piped or interactive)
    - File input
    """

    @staticmethod
    def read_message(
        argument: Optional[str] = None,
        from_stdin: bool = False,
        input_file: Optional[str] = None,
    ) -> Optional[str]:
        """Read input message from available sources.

        Priority: argument > input_file > stdin (if piped)

        Args:
            argument: Message passed as CLI argument
            from_stdin: Explicitly read from stdin
            input_file: Path to file containing input

        Returns:
            Input message, or None if no input available
        """
        # CLI argument takes priority
        if argument:
            return argument

        # File input
        if input_file:
            try:
                with open(input_file, "r") as f:
                    return f.read().strip()
            except Exception as e:
                raise ValueError(f"Failed to read input file: {e}")

        # Stdin - read all at once for multi-line support
        if from_stdin or not sys.stdin.isatty():
            # Stdin is piped or --stdin flag set
            return sys.stdin.read().strip()

        return None

    @staticmethod
    def is_piped() -> bool:
        """Check if stdin is piped (not interactive)."""
        return not sys.stdin.isatty()


def create_formatter(
    json_mode: bool = False,
    plain: bool = False,
    code_only: bool = False,
    jsonl: bool = False,
    quiet: bool = False,
    stream: bool = True,
) -> OutputFormatter:
    """Factory function to create OutputFormatter from CLI flags.

    Args:
        json_mode: Output as JSON object
        plain: Output as plain text (no Rich formatting)
        code_only: Extract and output only code
        jsonl: Output as streaming JSON Lines
        quiet: Suppress status messages
        stream: Stream output as it arrives

    Returns:
        Configured OutputFormatter
    """
    # Determine mode (priority: json > jsonl > code_only > plain > rich)
    if json_mode:
        mode = OutputMode.JSON
    elif jsonl:
        mode = OutputMode.JSONL
    elif code_only:
        mode = OutputMode.CODE_ONLY
    elif plain:
        mode = OutputMode.PLAIN
    else:
        mode = OutputMode.RICH

    config = OutputConfig(
        mode=mode,
        quiet=quiet,
        stream=stream,
        show_tools=not quiet,
        show_thinking=not quiet,
        show_metrics=mode == OutputMode.RICH,
    )

    return OutputFormatter(config)
