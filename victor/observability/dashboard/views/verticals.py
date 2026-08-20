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

"""Vertical integration dashboard views.

Provides widgets for viewing vertical integration traces:
- VerticalTraceWidget: Trace of vertical configuration and actions
- IntegrationResultWidget: Browse integration results from JSONL
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import DataTable, RichLog, Static, Tree
from textual.widgets.tree import TreeNode

from victor.core.events import MessagingEvent


class VerticalTraceWidget(RichLog):
    """Widget displaying vertical integration traces.

    Features:
    - Configuration change tracking
    - Tool enablement/disablement
    - Mode transitions
    - Workflow execution traces
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, highlight=True, markup=True, **kwargs)
        self._trace_count = 0
        self._active_verticals: Dict[str, Dict[str, Any]] = {}

    def add_vertical_event(self, event: MessagingEvent) -> None:
        """Add a vertical integration event to the trace.

        Args:
            event: Vertical event to add
        """
        if not event.topic.startswith("vertical."):
            return

        self._trace_count += 1
        data = event.data or {}
        timestamp = datetime.fromtimestamp(event.timestamp, tz=None).strftime("%H:%M:%S.%f")[:-3]

        # Extract key information
        vertical_name = data.get("vertical", "unknown")
        action = data.get("action", event.topic.split(".")[-1])

        # Color code by action type
        action_colors = {
            "applied": "green",
            "configured": "cyan",
            "enabled": "green",
            "disabled": "yellow",
            "error": "red",
            "workflow_start": "blue",
            "workflow_end": "blue",
        }

        action_key = action.lower().split(".")[-1] if "." in action else action.lower()
        color = action_colors.get(action_key, "white")

        # Main trace line
        self.write(f"[dim]{timestamp}[/] [magenta]{vertical_name}[/] [{color}]{action}[/]")

        # Show configuration details
        if "config" in data:
            config = data["config"]
            if isinstance(config, dict):
                self._write_config_preview(config)

        # Show tool changes
        if "tools_enabled" in data:
            tools = data["tools_enabled"]
            if tools:
                self.write(f"           [dim]tools: {', '.join(tools[:5])}")
                if len(tools) > 5:
                    self.write(f"           [dim]       ... and {len(tools) - 5} more[/]")

        # Show workflow information
        if "workflow" in data:
            workflow = data["workflow"]
            self.write(f"           [dim]workflow: {workflow}[/]")

        # Track active verticals
        if action.lower() in ("applied", "enabled"):
            self._active_verticals[vertical_name] = {
                "enabled_at": event.timestamp,
                "config": data.get("config", {}),
            }
        elif action.lower() == "disabled" and vertical_name in self._active_verticals:
            del self._active_verticals[vertical_name]

    def _write_config_preview(self, config: Dict[str, Any]) -> None:
        """Write a preview of configuration.

        Args:
            config: Configuration dictionary
        """
        # Show key configuration values
        important_keys = ["mode", "tool_budget", "max_iterations", "enabled_tools"]
        shown = []

        for key in important_keys:
            if key in config:
                value = config[key]
                if isinstance(value, list):
                    value = f"[{len(value)} items]"
                shown.append(f"{key}={value}")

        if shown:
            self.write(f"           [dim]config: {', '.join(shown)}[/]")

    @property
    def trace_count(self) -> int:
        """Get the number of traces recorded."""
        return self._trace_count

    @property
    def active_verticals(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active verticals."""
        return self._active_verticals.copy()


class IntegrationResultWidget(Container):
    """Widget for browsing integration results from JSONL files.

    Features:
    - Load and parse JSONL result files
    - Filter by vertical type
    - Browse result details
    - Success/failure statistics
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._results: List[Dict[str, Any]] = []
        self._file_path: Optional[Path] = None

    def compose(self) -> ComposeResult:
        yield Static("[bold]Integration Results[/]", id="results-title")
        yield Static(
            "[dim]Load a JSONL file to browse integration results...[/]",
            id="results-status",
        )
        yield DataTable(id="results-table")
        yield ScrollableContainer(
            Static("", id="result-detail"),
            id="result-detail-container",
        )

    def on_mount(self) -> None:
        """Set up the results table."""
        table = self.query_one("#results-table", DataTable)
        table.add_columns(
            "Time",
            "Vertical",
            "Action",
            "Status",
            "Duration",
        )
        table.cursor_type = "row"
        table.zebra_stripes = True

    def load_file(self, path: Path) -> int:
        """Load integration results from a JSONL file.

        Args:
            path: Path to the JSONL file

        Returns:
            Number of results loaded
        """
        self._results.clear()
        self._file_path = path

        status = self.query_one("#results-status", Static)
        table = self.query_one("#results-table", DataTable)

        if not path.exists():
            status.update(f"[red]File not found: {path}[/]")
            return 0

        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        # Filter for vertical/integration events
                        category = data.get("category", "")
                        if category in ("vertical", "integration"):
                            self._results.append(data)
                    except json.JSONDecodeError:
                        continue

            # Update display
            status.update(f"[green]Loaded {len(self._results)} results from {path.name}[/]")

            table.clear()
            # Show last 100 results in descending order (newest first)
            for result in reversed(self._results[-100:]):
                self._add_result_row(result)

            return len(self._results)

        except Exception as e:
            status.update(f"[red]Error loading file: {e}[/]")
            return 0

    def _add_result_row(self, result: Dict[str, Any]) -> None:
        """Add a result row to the table."""
        table = self.query_one("#results-table", DataTable)

        # Parse timestamp
        timestamp_str = result.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            time_display = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            time_display = timestamp_str[:19] if timestamp_str else "-"

        # Extract fields
        data = result.get("data", {})
        vertical = data.get("vertical", "-")
        action = result.get("name", "-")

        # Determine status
        success = data.get("success", True)
        status = "[green]OK[/]" if success else "[red]FAIL[/]"

        # Duration
        duration = data.get("duration_ms", data.get("duration_seconds"))
        if duration is not None:
            if isinstance(duration, float) and duration < 1:
                # Probably in seconds, convert to ms
                duration_str = f"{duration * 1000:.0f}ms"
            else:
                duration_str = f"{duration:.0f}ms"
        else:
            duration_str = "-"

        table.add_row(time_display, vertical, action, status, duration_str)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection to show details."""
        if event.cursor_row < len(self._results):
            result = self._results[-(len(self._results) - event.cursor_row)]
            self._show_result_detail(result)

    def _show_result_detail(self, result: Dict[str, Any]) -> None:
        """Show detailed view of a result."""
        detail = self.query_one("#result-detail", Static)

        lines = [
            "[bold]Result Details[/]",
            "",
        ]

        # Basic info
        lines.append(f"[bold]Name:[/] {result.get('name', '-')}")
        lines.append(f"[bold]Category:[/] {result.get('category', '-')}")
        lines.append(f"[bold]Timestamp:[/] {result.get('timestamp', '-')}")

        if result.get("session_id"):
            lines.append(f"[bold]Session:[/] {result['session_id']}")

        # Data section
        data = result.get("data", {})
        if data:
            lines.extend(["", "[bold]Data:[/]"])
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"  {key}:")
                    for k, v in list(value.items())[:5]:
                        lines.append(f"    {k}: {v}")
                    if len(value) > 5:
                        lines.append(f"    ... (+{len(value) - 5} more)")
                elif isinstance(value, list):
                    lines.append(f"  {key}: [{len(value)} items]")
                else:
                    str_value = str(value)
                    if len(str_value) > 60:
                        str_value = str_value[:60] + "..."
                    lines.append(f"  {key}: {str_value}")

        detail.update("\n".join(lines))


class VerticalConfigTree(Tree):
    """Tree view of vertical configuration hierarchy.

    Features:
    - Expandable configuration sections
    - Tool groupings
    - Mode settings
    - Workflow definitions
    """

    def __init__(self, *args, **kwargs):
        super().__init__("Verticals", *args, **kwargs)
        self._verticals: Dict[str, Dict[str, Any]] = {}

    def update_vertical(self, name: str, config: Dict[str, Any]) -> None:
        """Update or add a vertical configuration.

        Args:
            name: Vertical name
            config: Configuration dictionary
        """
        self._verticals[name] = config
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        """Rebuild the tree from current verticals."""
        self.clear()

        for name, config in sorted(self._verticals.items()):
            vertical_node = self.root.add(f"[magenta]{name}[/]", expand=True)

            # Add configuration sections
            if "mode" in config:
                vertical_node.add_leaf(f"Mode: {config['mode']}")

            if "tool_budget" in config:
                vertical_node.add_leaf(f"Tool Budget: {config['tool_budget']}")

            if "enabled_tools" in config:
                tools = config["enabled_tools"]
                tools_node = vertical_node.add("Tools", expand=False)
                for tool in tools[:10]:
                    tools_node.add_leaf(f"[cyan]{tool}[/]")
                if len(tools) > 10:
                    tools_node.add_leaf(f"[dim]... +{len(tools) - 10} more[/]")

            if "workflows" in config:
                workflows = config["workflows"]
                wf_node = vertical_node.add("Workflows", expand=False)
                for wf in workflows[:5]:
                    wf_node.add_leaf(f"[blue]{wf}[/]")

    def remove_vertical(self, name: str) -> None:
        """Remove a vertical from the tree.

        Args:
            name: Vertical name to remove
        """
        if name in self._verticals:
            del self._verticals[name]
            self._rebuild_tree()
