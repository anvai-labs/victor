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

"""Horizontal session info bar for the interactive TUI.

Displays session metadata (id, mode, model/provider, budget, context usage)
in a compact horizontal layout above the prompt input. Replaces the left
sidebar for better space utilization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rich.markup import escape
from rich.text import Text
from textual.widgets import Static


@dataclass
class TokenUsage:
    """Detailed token usage breakdown by type."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt_tokens,
            "completion": self.completion_tokens,
            "reasoning": self.reasoning_tokens,
            "cached": self.cached_tokens,
            "total": self.total_tokens,
        }


@dataclass
class SessionInfo:
    """Snapshot of session metadata for the horizontal bar."""

    session_id: str = "—"
    mode: str = "—"
    model: str = "—"
    provider: str = "—"
    profile: str = "—"
    budget_used: int = 0
    budget_total: Optional[int] = None
    context_used: Optional[int] = None
    context_total: Optional[int] = None
    context_pct: Optional[float] = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: Optional[float] = None


class SessionBar(Static):
    """Horizontal widget rendering session metadata above the prompt."""

    DEFAULT_CSS = """
    SessionBar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    """

    def set_session_info(self, info: SessionInfo) -> None:
        """Replace the displayed session info and re-render."""
        self.update(self._build_line(info))

    @staticmethod
    def _build_line(info: SessionInfo) -> str:
        """Build a compact horizontal representation of session state.

        Static info (session, mode, profile, model, provider) uses neutral colors.
        Dynamic info (context, tokens, cost) uses bright accent colors to stand out.
        """
        parts: list[str] = []

        # STATIC INFO - neutral colors
        # Session ID (truncated)
        session_short = info.session_id[:8] if len(info.session_id) > 8 else info.session_id
        parts.append(f"[dim]session:[/] [white]{escape(session_short)}[/]")

        # Mode
        parts.append(f"[dim]mode:[/] [white]{escape(info.mode)}[/]")

        # Profile and Model/Provider
        if info.profile and info.profile != "—":
            parts.append(f"[dim]profile:[/] [white]{escape(info.profile)}[/]")
        model_provider = f"{info.model}"
        if info.provider and info.provider != "—":
            model_provider += f" [dim]via[/] [white]{escape(info.provider)}[/]"
        parts.append(f"[dim]model:[/] [white]{model_provider}[/]")

        # Tool budget (only show if set)
        if info.budget_total and info.budget_total > 0:
            budget_str = f"{info.budget_used}/{info.budget_total}"
            parts.append(f"[dim]budget:[/] [white]{budget_str}[/]")

        # DYNAMIC INFO - bright accent colors
        # Context usage with percentage
        if info.context_used is not None and info.context_total is not None:
            used_str = _format_tokens(info.context_used)
            total_str = _format_tokens(info.context_total)
            if info.context_pct is not None:
                pct_str = f"{info.context_pct:.1f}%"
                # Color-code based on usage percentage
                if info.context_pct >= 90:
                    color = "red"
                elif info.context_pct >= 70:
                    color = "yellow"
                else:
                    color = "green"
                context_str = f"[{color}]{used_str}/{total_str} ({pct_str})[/]"
            else:
                context_str = f"[bright_cyan]{used_str}/{total_str}[/]"
            parts.append(f"[dim]context:[/] {context_str}")
        elif info.context_pct is not None:
            pct_str = f"{info.context_pct:.1f}%"
            parts.append(f"[dim]context:[/] [bright_cyan]{pct_str}[/]")

        # Token breakdown (if available)
        usage = info.token_usage
        if usage.total_tokens > 0:
            token_parts = []
            if usage.prompt_tokens > 0:
                token_parts.append(
                    f"prompt=[bright_magenta]{_format_tokens(usage.prompt_tokens)}[/]"
                )
            if usage.completion_tokens > 0:
                token_parts.append(
                    f"completion=[bright_magenta]{_format_tokens(usage.completion_tokens)}[/]"
                )
            if usage.reasoning_tokens > 0:
                token_parts.append(
                    f"reasoning=[bright_magenta]{_format_tokens(usage.reasoning_tokens)}[/]"
                )
            if usage.cached_tokens > 0:
                token_parts.append(
                    f"cached=[bright_magenta]{_format_tokens(usage.cached_tokens)}[/]"
                )
            if token_parts:
                tokens_str = " ".join(token_parts)
                parts.append(f"[dim]tokens:[/] {tokens_str}")

        # Cost
        if info.cost_usd is not None:
            parts.append(f"[dim]cost:[/] [bright_yellow]${info.cost_usd:.4f}[/]")

        return " [dim]·[/] ".join(parts)


def _format_tokens(tokens: int) -> str:
    """Format token count with K suffix for thousands."""
    if tokens >= 1000:
        return f"{tokens / 1000:.1f}K"
    return str(tokens)
