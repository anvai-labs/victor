# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Status footer for the interactive TUI.

One line projecting the inferred loop phase, a live tool counter, and the most
recent per-turn token/cost figures (``VictorClient.get_last_turn_cost``). When
the stall watchdog fires, the phase is replaced by a visible "waiting on model"
state so a quiet agent never reads as a frozen terminal (ADR-021).
"""

from __future__ import annotations

from typing import Optional

from rich.markup import escape
from textual.widgets import Static


class StatusBar(Static):
    """Footer widget rendering phase · tools · tokens · cost."""

    def set_status(
        self,
        *,
        phase_label: str,
        tool_count: int = 0,
        total_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        waiting_seconds: Optional[int] = None,
    ) -> None:
        """Re-render the footer from the current turn state.

        Args:
            phase_label: Inferred phase text (e.g. ``"acting · read"``).
            tool_count: Tools started so far this turn.
            total_tokens: Total tokens for the last completed turn, if known.
            cost_usd: USD cost for the last completed turn, if known.
            waiting_seconds: When set, the watchdog is active; overrides the
                phase with a "waiting on model (Ns)…" indicator.
        """
        self.update(
            self._compose_line(
                phase_label=phase_label,
                tool_count=tool_count,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                waiting_seconds=waiting_seconds,
            )
        )

    @staticmethod
    def _compose_line(
        *,
        phase_label: str,
        tool_count: int,
        total_tokens: Optional[int],
        cost_usd: Optional[float],
        waiting_seconds: Optional[int],
    ) -> str:
        if waiting_seconds is not None:
            head = f"[yellow]▸ waiting on model ({waiting_seconds}s)…[/]"
        else:
            head = f"[bold]▸ {escape(phase_label)}[/]"
        parts = [head]
        if tool_count:
            parts.append(f"{tool_count} tool{'s' if tool_count != 1 else ''}")
        if total_tokens is not None:
            parts.append(f"{_human_tokens(total_tokens)} tok")
        if cost_usd is not None:
            parts.append(f"${cost_usd:.4f}")
        return " · ".join(parts)


def _human_tokens(n: int) -> str:
    """Render a token count compactly (e.g. ``12.4k``)."""
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"
