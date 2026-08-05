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

"""Event-related dashboard views.

Provides widgets for viewing and filtering events:
- EventLogWidget: Scrollable log of events
- EventTableWidget: Tabular view with sorting
- EventFilterWidget: Filter controls for categories/names
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import DataTable, RichLog, Select, Static, Switch
from textual.reactive import reactive

from victor.core.events import MessagingEvent, ObservabilityBus, get_observability_bus

# Color mapping for event categories (topic prefixes)
CATEGORY_COLORS: Dict[str, str] = {
    "tool.": "cyan",
    "state.": "yellow",
    "model.": "green",
    "error.": "red",
    "audit.": "magenta",
    "metric.": "blue",
    "lifecycle.": "white",
    "vertical.": "bright_magenta",
    "custom": "dim",
}


class EventLogWidget(RichLog):
    """Rich log widget for displaying events in real-time.

    Features:
    - Color-coded categories
    - Timestamp display
    - Expandable event details
    - Auto-scroll with pause option
    """

    auto_scroll: reactive[bool] = reactive(True)

    def __init__(
        self,
        *args,
        show_data: bool = True,
        max_data_length: int = 100,
        **kwargs,
    ):
        """Initialize the event log widget.

        Args:
            show_data: Whether to show event data details
            max_data_length: Maximum length of data preview
        """
        super().__init__(*args, highlight=True, markup=True, **kwargs)
        self._show_data = show_data
        self._max_data_length = max_data_length
        self._event_count = 0

    def add_event(self, event: MessagingEvent) -> None:
        """Add an event to the log.

        Args:
            event: The VictorEvent to display
        """
        self._event_count += 1
        timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        color = CATEGORY_COLORS.get(event.topic.split(".")[0], "white")
        category_name = (
            event.topic.split(".")[0].upper() if event.topic.split(".")[0] else "UNKNOWN"
        )

        # Main event line
        self.write(f"[dim]{timestamp}[/] [{color}]{category_name:10}[/] [bold]{event.topic}[/]")

        # Optional data preview
        if self._show_data and event.data:
            data_str = self._format_data(event.data)
            if data_str:
                self.write(f"           [dim]{data_str}[/]")

    def _format_data(self, data: Dict[str, Any]) -> str:
        """Format event data for display.

        Args:
            data: Event data dictionary

        Returns:
            Formatted string preview
        """
        # Prioritize interesting fields
        priority_fields = [
            "tool_name",
            "old_stage",
            "new_stage",
            "message",
            "error",
            "success",
        ]

        parts = []
        for field in priority_fields:
            if field in data:
                value = data[field]
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                parts.append(f"{field}={value}")

        # Add remaining fields up to limit
        for key, value in data.items():
            if key not in priority_fields and len(parts) < 4:
                if isinstance(value, str) and len(value) > 30:
                    value = value[:30] + "..."
                elif isinstance(value, (list, dict)):
                    value = f"[{len(value)} items]" if isinstance(value, list) else "{...}"
                parts.append(f"{key}={value}")

        result = ", ".join(parts)
        if len(result) > self._max_data_length:
            result = result[: self._max_data_length] + "..."

        return result

    @property
    def event_count(self) -> int:
        """Get the number of events displayed."""
        return self._event_count


class EventTableWidget(DataTable):
    """Tabular view of events with sorting and selection.

    Features:
    - Sortable columns
    - Row selection for details
    - Category filtering
    - Timestamp ordering
    """

    def __init__(self, *args, max_rows: int = 500, **kwargs):
        """Initialize the event table widget.

        Args:
            max_rows: Maximum rows to display (older events removed)
        """
        super().__init__(*args, **kwargs)
        self._max_rows = max_rows
        self._events: List[MessagingEvent] = []

    def on_mount(self) -> None:
        """Set up table columns."""
        self.add_columns("Time", "Category", "Name", "Session", "Details")
        self.cursor_type = "row"
        self.zebra_stripes = True

    def add_event(self, event: MessagingEvent) -> None:
        """Add an event to the table.

        Events are displayed in descending order (newest first).

        Args:
            event: The VictorEvent to add
        """
        # Insert at beginning to maintain descending order
        self._events.insert(0, event)

        # Remove old events if over limit
        if len(self._events) > self._max_rows:
            self._events = self._events[: self._max_rows]

        # Rebuild table
        self._rebuild_table()

    def _add_row(self, event: MessagingEvent) -> None:
        """Add a single row for an event."""
        timestamp = event.timestamp.strftime("%H:%M:%S")
        category = event.topic.split(".")[0] if event.topic.split(".")[0] else "unknown"
        session = (event.session_id or "")[:8] if event.session_id else "-"

        # Build details
        details = self._get_event_details(event)

        self.add_row(timestamp, category, event.topic, session, details)

    def _get_event_details(self, event: MessagingEvent) -> str:
        """Extract key details from event data."""
        if not event.data:
            return ""

        data = event.data

        # Tool events
        if "tool_name" in data:
            tool = data["tool_name"]
            if "duration_ms" in data:
                return f"{tool} ({data['duration_ms']:.0f}ms)"
            return tool

        # State events
        if "old_stage" in data and "new_stage" in data:
            return f"{data['old_stage']} -> {data['new_stage']}"

        # Error events
        if "message" in data:
            msg = str(data["message"])
            return msg[:60] + "..." if len(msg) > 60 else msg

        # Generic
        return str(data)[:50]

    def _rebuild_table(self) -> None:
        """Rebuild the entire table from events list.

        Displays events in descending order (newest first).
        Events list is already in descending order.
        """
        self.clear()
        # Events are already in descending order (newest first)
        for event in self._events:
            self._add_row(event)

    def get_selected_event(self) -> Optional[MessagingEvent]:
        """Get the currently selected event.

        Events list and table are both in descending order (newest first),
        so cursor_row maps directly to list index.
        """
        if self.cursor_row < len(self._events):
            return self._events[self.cursor_row]
        return None


class EventFilterWidget(Container):
    """Widget for filtering events by category and other criteria.

    Features:
    - Category toggles
    - Name pattern filter
    - Session filter
    - Time range filter
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default enabled topic prefixes (all categories enabled)
        self._enabled_categories: Set[str] = {
            "tool",
            "state",
            "model",
            "error",
            "audit",
            "metric",
            "lifecycle",
            "vertical",
        }

    def compose(self) -> ComposeResult:
        yield Static("[bold]Event Filters[/]", classes="filter-title")
        with Horizontal(classes="filter-row"):
            yield Static("Categories:", classes="filter-label")
            # Iterate over topic prefixes (was EventCategory enum)
            for category in [
                "tool",
                "state",
                "model",
                "error",
                "audit",
                "metric",
                "lifecycle",
                "vertical",
            ]:
                color = CATEGORY_COLORS.get(category, "white")
                yield Switch(
                    value=True,
                    id=f"filter-{category}",
                    classes="category-switch",
                )
                yield Static(
                    f"[{color}]{category}[/]",
                    classes="category-label",
                )

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle category filter toggle."""
        switch_id = event.switch.id or ""
        if switch_id.startswith("filter-"):
            category_name = switch_id[7:]  # Remove "filter-" prefix
            try:
                # EventCategory removed - use topic-based filtering
                category = category_name  # Just use the string directly
                if event.value:
                    self._enabled_categories.add(category)
                else:
                    self._enabled_categories.discard(category)
            except ValueError:
                pass

    def should_show_event(self, event: MessagingEvent) -> bool:
        """Check if an event should be displayed based on current filters.

        Args:
            event: The event to check

        Returns:
            True if the event should be shown
        """
        return event.topic.split(".")[0] in self._enabled_categories

    @property
    def enabled_categories(self) -> Set[str]:
        """Get the set of enabled categories (topic prefixes)."""
        return self._enabled_categories.copy()
