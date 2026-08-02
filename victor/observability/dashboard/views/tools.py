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

"""Tool execution dashboard views.

Provides widgets for viewing tool execution statistics:
- ToolStatsWidget: Summary statistics per tool
- ToolHistoryWidget: Detailed execution history
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import DataTable, RichLog, Static
from textual.reactive import reactive

from victor.core.events import MessagingEvent


@dataclass
class ToolStats:
    """Statistics for a single tool."""

    name: str
    call_count: int = 0
    success_count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: Optional[float] = None
    max_duration_ms: Optional[float] = None
    last_called: Optional[datetime] = None
    last_result: Optional[str] = None
    error_messages: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.call_count == 0:
            return 0.0
        return (self.success_count / self.call_count) * 100

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration in milliseconds."""
        if self.call_count == 0:
            return 0.0
        return self.total_duration_ms / self.call_count

    def record_call(
        self,
        success: bool,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record a tool execution.

        Args:
            success: Whether the call succeeded
            duration_ms: Execution duration in milliseconds
            error: Optional error message if failed
        """
        self.call_count += 1
        if success:
            self.success_count += 1
        self.last_called = datetime.now()

        if duration_ms is not None:
            self.total_duration_ms += duration_ms
            if self.min_duration_ms is None or duration_ms < self.min_duration_ms:
                self.min_duration_ms = duration_ms
            if self.max_duration_ms is None or duration_ms > self.max_duration_ms:
                self.max_duration_ms = duration_ms

        if error:
            self.error_messages.append(error)
            # Keep only last 10 errors
            if len(self.error_messages) > 10:
                self.error_messages = self.error_messages[-10:]


class ToolStatsWidget(DataTable):
    """Widget displaying tool execution statistics.

    Features:
    - Per-tool call counts
    - Success rates with color coding
    - Average/min/max execution times
    - Sortable columns
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stats: Dict[str, ToolStats] = {}

    def on_mount(self) -> None:
        """Set up table columns."""
        self.add_columns(
            "Tool",
            "Calls",
            "Success Rate",
            "Avg Time",
            "Min/Max",
            "Last Called",
        )
        self.cursor_type = "row"
        self.zebra_stripes = True

    def add_tool_event(self, event: MessagingEvent) -> None:
        """Process a tool event and update statistics.

        Args:
            event: Tool event to process
        """
        if not event.topic.startswith("tool."):
            return

        data = event.data or {}
        tool_name = data.get("tool_name", "unknown")

        # Initialize stats if needed
        if tool_name not in self._stats:
            self._stats[tool_name] = ToolStats(name=tool_name)

        stats = self._stats[tool_name]

        # Process end events for statistics
        if event.topic.endswith(".end"):
            success = data.get("success", True)
            duration_ms = data.get("duration_ms")
            error = data.get("error") or data.get("message")

            stats.record_call(
                success=success,
                duration_ms=duration_ms,
                error=error if not success else None,
            )

            # Store last result preview
            result = data.get("result")
            if result:
                result_str = str(result)
                stats.last_result = result_str[:100] if len(result_str) > 100 else result_str

            self._refresh_table()

    def _refresh_table(self) -> None:
        """Refresh the table with current statistics."""
        self.clear()

        for tool_name in sorted(self._stats.keys()):
            stats = self._stats[tool_name]

            # Format success rate with color
            rate = stats.success_rate
            if rate >= 90:
                rate_str = f"[green]{rate:.0f}%[/]"
            elif rate >= 70:
                rate_str = f"[yellow]{rate:.0f}%[/]"
            else:
                rate_str = f"[red]{rate:.0f}%[/]"

            # Format duration
            avg_time = f"{stats.avg_duration_ms:.0f}ms" if stats.call_count > 0 else "-"

            # Format min/max
            if stats.min_duration_ms is not None and stats.max_duration_ms is not None:
                min_max = f"{stats.min_duration_ms:.0f}/{stats.max_duration_ms:.0f}ms"
            else:
                min_max = "-"

            # Format last called
            last_called = stats.last_called.strftime("%H:%M:%S") if stats.last_called else "-"

            self.add_row(
                tool_name,
                str(stats.call_count),
                rate_str,
                avg_time,
                min_max,
                last_called,
            )

    def get_stats(self, tool_name: str) -> Optional[ToolStats]:
        """Get statistics for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolStats or None if not found
        """
        return self._stats.get(tool_name)

    def get_all_stats(self) -> Dict[str, ToolStats]:
        """Get all tool statistics.

        Returns:
            Dictionary of tool name to ToolStats
        """
        return self._stats.copy()


class ToolHistoryWidget(RichLog):
    """Widget displaying detailed tool execution history.

    Features:
    - Chronological execution log
    - Argument and result preview
    - Error highlighting
    - Duration tracking
    """

    def __init__(self, *args, max_entries: int = 200, **kwargs):
        """Initialize the tool history widget.

        Args:
            max_entries: Maximum history entries to keep
        """
        super().__init__(*args, highlight=True, markup=True, **kwargs)
        self._max_entries = max_entries
        self._entry_count = 0

    def add_tool_event(self, event: MessagingEvent) -> None:
        """Add a tool event to the history.

        Args:
            event: Tool event to add
        """
        if not event.topic.startswith("tool."):
            return

        data = event.data or {}
        tool_name = data.get("tool_name", "unknown")
        timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]

        if event.topic.endswith(".start"):
            # Tool start
            self.write(f"[dim]{timestamp}[/] [cyan]{tool_name}[/] [yellow]STARTED[/]")

            # Show arguments preview
            args = data.get("arguments", {})
            if args:
                args_preview = self._format_arguments(args)
                self.write(f"           [dim]args: {args_preview}[/]")

        elif event.topic.endswith(".end"):
            # Tool end
            success = data.get("success", True)
            duration_ms = data.get("duration_ms")

            status = "[green]SUCCESS[/]" if success else "[red]FAILED[/]"
            duration_str = f" ({duration_ms:.0f}ms)" if duration_ms else ""

            self.write(f"[dim]{timestamp}[/] [cyan]{tool_name}[/] {status}{duration_str}")

            # Show result preview or error
            if not success:
                error = data.get("error") or data.get("message")
                if error:
                    error_preview = str(error)[:100]
                    self.write(f"           [red]error: {error_preview}[/]")
            else:
                result = data.get("result")
                if result:
                    result_preview = str(result)[:80]
                    self.write(f"           [dim]result: {result_preview}[/]")

        self._entry_count += 1

    def _format_arguments(self, args: Dict[str, Any]) -> str:
        """Format tool arguments for display.

        Args:
            args: Arguments dictionary

        Returns:
            Formatted string preview
        """
        parts = []
        for key, value in list(args.items())[:3]:  # Limit to 3 args
            if isinstance(value, str):
                if len(value) > 30:
                    value = value[:30] + "..."
            elif isinstance(value, (list, dict)):
                value = f"[{len(value)} items]" if isinstance(value, list) else "{...}"
            parts.append(f"{key}={value}")

        result = ", ".join(parts)
        if len(args) > 3:
            result += f", ... (+{len(args) - 3} more)"

        return result if len(result) <= 80 else result[:77] + "..."


class ToolDetailPanel(Container):
    """Panel showing detailed information about a selected tool.

    Features:
    - Complete statistics
    - Recent execution history
    - Error log
    - Argument patterns
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_tool: Optional[str] = None
        self._stats: Optional[ToolStats] = None

    def compose(self) -> ComposeResult:
        yield Static("[bold]Tool Details[/]", id="detail-title")
        yield Static("[dim]Select a tool to view details[/]", id="detail-content")

    def show_tool(self, tool_name: str, stats: ToolStats) -> None:
        """Display details for a specific tool.

        Args:
            tool_name: Name of the tool
            stats: Statistics for the tool
        """
        self._current_tool = tool_name
        self._stats = stats

        content = self.query_one("#detail-content", Static)

        # Build detailed view
        lines = [
            f"[bold cyan]{tool_name}[/]",
            "",
            f"[bold]Calls:[/] {stats.call_count}",
            f"[bold]Success Rate:[/] {stats.success_rate:.1f}%",
            "",
            "[bold]Timing:[/]",
            f"  Average: {stats.avg_duration_ms:.0f}ms",
        ]

        if stats.min_duration_ms is not None:
            lines.append(f"  Min: {stats.min_duration_ms:.0f}ms")
        if stats.max_duration_ms is not None:
            lines.append(f"  Max: {stats.max_duration_ms:.0f}ms")

        if stats.last_called:
            lines.extend(
                [
                    "",
                    f"[bold]Last Called:[/] {stats.last_called.strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            )

        if stats.last_result:
            lines.extend(
                [
                    "",
                    "[bold]Last Result:[/]",
                    f"  [dim]{stats.last_result}[/]",
                ]
            )

        if stats.error_messages:
            lines.extend(
                [
                    "",
                    f"[bold red]Recent Errors ({len(stats.error_messages)}):[/]",
                ]
            )
            for error in stats.error_messages[-3:]:
                lines.append(f"  [red]- {error[:60]}[/]")

        content.update("\n".join(lines))

    def clear_selection(self) -> None:
        """Clear the current tool selection."""
        self._current_tool = None
        self._stats = None
        content = self.query_one("#detail-content", Static)
        content.update("[dim]Select a tool to view details[/]")
