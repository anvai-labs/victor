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

"""Agent-state sidebar for the interactive TUI (left pane).

A read-only projection of session state — id, mode, model/provider, cwd, tool
budget, and context-window usage — refreshed on turn boundaries. Purely
presentational; it holds no session logic and reaches into no runtime internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rich.table import Table
from textual.widgets import Static


@dataclass
class AgentState:
    """Snapshot of the session fields shown in the sidebar."""

    session_id: str = "—"
    mode: str = "—"
    model: str = "—"
    provider: str = "—"
    cwd: str = "—"
    budget_used: int = 0
    budget_total: Optional[int] = None
    context_pct: Optional[float] = None
    context_used: Optional[int] = None
    context_total: Optional[int] = None


class AgentStatePanel(Static):
    """Left-pane widget rendering an :class:`AgentState` as a compact table."""

    def set_state(self, state: AgentState) -> None:
        """Replace the displayed state and re-render."""
        self.update(self._build_table(state))

    @staticmethod
    def _build_table(state: AgentState) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="dim", no_wrap=True)
        table.add_column(justify="left", no_wrap=True)

        def row(label: str, value: str) -> None:
            table.add_row(label, value)

        row("session", state.session_id)
        row("mode", state.mode)
        row("model", state.model)
        row("provider", state.provider)
        row("cwd", state.cwd)
        if state.budget_total:
            row("budget", f"{state.budget_used}/{state.budget_total}")
        else:
            row("budget", str(state.budget_used))
        if state.context_used is not None and state.context_total is not None:
            # Show context usage as "used/total (pct%)" for full visibility
            used_str = _format_tokens(state.context_used)
            total_str = _format_tokens(state.context_total)
            pct_str = f"{state.context_pct:.0f}%" if state.context_pct is not None else ""
            context_value = f"{used_str}/{total_str}"
            if pct_str:
                context_value += f" ({pct_str})"
            row("context", context_value)
        elif state.context_pct is not None:
            row("context", f"{state.context_pct:.0f}%")
        return table


def _format_tokens(tokens: int) -> str:
    """Format token count with K suffix for thousands."""
    if tokens >= 1000:
        return f"{tokens / 1000:.1f}K"
    return str(tokens)
