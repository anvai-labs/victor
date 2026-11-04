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

"""Token usage display bar for the TUI.

Shows token usage, cost, and context percentage above the prompt input.
"""

from __future__ import annotations

from typing import Optional

from textual.widgets import Static


def _human_tokens(n: int) -> str:
    """Render a token count compactly (e.g. ``12.4k``)."""
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"


class TokenUsageBar(Static):
    """Widget displaying token usage, cost, and context percentage."""

    DEFAULT_CSS = """
    TokenUsageBar {
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $foreground;
        text-style: italic;
    }
    """

    def set_usage(
        self,
        *,
        total_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        context_used_pct: Optional[float] = None,
    ) -> None:
        """Update the token usage display.

        Args:
            total_tokens: Total tokens used in the last turn.
            cost_usd: Cost in USD for the last turn.
            context_used_pct: Percentage of context window used.
        """
        parts: list[str] = []

        if total_tokens is not None:
            parts.append(f"[dim]Tokens:[/] [bold]{_human_tokens(total_tokens)}[/]")

        if cost_usd is not None:
            parts.append(f"[dim]Cost:[/] [bold]${cost_usd:.4f}[/]")

        if context_used_pct is not None:
            color = (
                "green" if context_used_pct < 60 else ("yellow" if context_used_pct < 85 else "red")
            )
            parts.append(f"[dim]Context:[/] [{color}]{context_used_pct:.0f}%[/]")

        if not parts:
            self.update("[dim italic]No usage data yet[/]")
        else:
            self.update("  ·  ".join(parts))
