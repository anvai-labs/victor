# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Command discoverability for the TUI: `/help` overlay + command palette (ADR-021).

Both surfaces reuse the surface-agnostic slash registry
(:func:`victor.ui.slash.registry.get_command_registry`) so the TUI exposes the
same commands as the REPL with no duplication. :class:`HelpScreen` lists them;
:class:`CommandPalette` offers a filterable picker that returns the chosen
command string for the app to dispatch through its ``SlashCommandHandler``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from rich.markup import escape
from rich.table import Table
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static

#: (command name without slash, description) pairs.
CommandRow = Tuple[str, str]


def load_commands() -> List[CommandRow]:
    """Return ``(name, description)`` for every registered slash command, sorted."""
    try:
        from victor.ui.slash.registry import get_command_registry

        registry = get_command_registry()
        rows: List[CommandRow] = []
        for name, metadata in registry.list_commands():
            description = getattr(metadata, "description", "") or ""
            rows.append((name, description))
        return sorted(rows, key=lambda row: row[0])
    except Exception:  # noqa: BLE001 - discoverability must never crash the app
        return []


class HelpScreen(ModalScreen[None]):
    """Scrollable overlay listing all slash commands."""

    BINDINGS = [("escape", "close", "Close"), ("q", "close", "Close")]

    def __init__(self, commands: Optional[List[CommandRow]] = None) -> None:
        super().__init__()
        self._commands = commands if commands is not None else load_commands()

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static(self._table(), id="help-table")
            yield Static("[dim]esc to close[/]", id="help-footer")

    def _table(self) -> Table:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="default")
        if not self._commands:
            table.add_row("/help", "no commands registered")
            return table
        for name, description in self._commands:
            table.add_row(f"/{name}", escape(description))
        return table

    def action_close(self) -> None:
        self.dismiss(None)


class CommandPalette(ModalScreen[Optional[str]]):
    """Filterable command picker; dismisses with the chosen ``/command`` or ``None``."""

    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(self, commands: Optional[List[CommandRow]] = None) -> None:
        super().__init__()
        self._commands = commands if commands is not None else load_commands()

    def compose(self) -> ComposeResult:
        with Vertical(id="palette-dialog"):
            yield Input(placeholder="Type a command…", id="palette-input")
            yield ListView(id="palette-list")

    def on_mount(self) -> None:
        self._populate("")
        self.query_one("#palette-input", Input).focus()

    def _populate(self, query: str) -> None:
        listview = self.query_one("#palette-list", ListView)
        listview.clear()
        needle = query.strip().lower().lstrip("/")
        for name, description in self._commands:
            if needle and needle not in name.lower():
                continue
            listview.append(ListItem(Label(f"/{name}  —  {escape(description)}"), name=name))

    def on_input_changed(self, event: Input.Changed) -> None:
        self._populate(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        listview = self.query_one("#palette-list", ListView)
        item = listview.highlighted_child
        if item is not None and item.name:
            self.dismiss(f"/{item.name}")
            return
        text = event.value.strip()
        if not text:
            self.dismiss(None)
        else:
            self.dismiss(text if text.startswith("/") else f"/{text}")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is not None and event.item.name:
            self.dismiss(f"/{event.item.name}")

    def action_cancel(self) -> None:
        self.dismiss(None)
