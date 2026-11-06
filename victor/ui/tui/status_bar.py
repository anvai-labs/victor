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

"""Status bar for the interactive TUI.

One line projecting runtime state: inferred loop phase, live tool counter,
and session id. Model/provider/profile live in the header;
token/cost/context telemetry lives in
``TokenUsageBar`` to avoid duplication. When the stall watchdog fires, the
phase is replaced by a visible "waiting on model" state so a quiet agent
never reads as a frozen terminal (ADR-021).
"""

from __future__ import annotations

from typing import Optional

from rich.markup import escape
from textual.widgets import Static


class StatusBar(Static):
    """Runtime state bar: phase · tools · session."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $foreground;
    }
    """

    def set_status(
        self,
        *,
        phase_label: str,
        tool_count: int = 0,
        waiting_seconds: Optional[int] = None,
        queued: int = 0,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        profile: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> None:
        """Re-render the status bar from current runtime state.

        Args:
            phase_label: Inferred phase text (e.g. ``"acting · read"``).
            tool_count: Tools started so far this turn.
            waiting_seconds: When set, the watchdog is active; overrides the
                phase with a "waiting on model (Ns)…" indicator.
            queued: Prompts waiting in the mid-turn queue (hidden when 0).
            session_id: Short session identifier.
            model: Current model name.
            provider: Current provider name.
            profile: Active profile name.
            cwd: Current working directory.
        """
        self.update(
            self._compose_line(
                phase_label=phase_label,
                tool_count=tool_count,
                waiting_seconds=waiting_seconds,
                queued=queued,
                session_id=session_id,
                model=model,
                provider=provider,
                profile=profile,
                cwd=cwd,
            )
        )

    @staticmethod
    def _compose_line(
        *,
        phase_label: str,
        tool_count: int,
        waiting_seconds: Optional[int],
        queued: int = 0,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        profile: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> str:
        """Compose a single-line status: phase · tools · queued · session · cwd.

        Model/provider/profile are intentionally omitted — the model appears
        in the header subtitle and the TokenUsageBar's context line. Session
        is kept here as the minimal runtime identifier.
        """
        if waiting_seconds is not None:
            head = f"[yellow]▸ waiting ({waiting_seconds}s)…[/]"
        else:
            head = f"[bold cyan]▸ {escape(phase_label)}[/]"
        parts = [head]
        if tool_count:
            parts.append(f"[dim]{tool_count} tool{'s' if tool_count != 1 else ''}[/]")
        if queued:
            parts.append(f"[dim]{queued} queued[/]")
        if session_id:
            parts.append(f"[dim]s:[/] [white]{escape(session_id)}[/]")
        if cwd:
            # Show last 2 path components for compactness
            cwd_parts = cwd.rsplit("/", 2)
            cwd_short = (
                "/".join(cwd_parts) if len(cwd_parts) <= 2 else "…/" + "/".join(cwd_parts[-2:])
            )
            parts.append(f"[dim]cwd:[/] [white]{escape(cwd_short)}[/]")
        return " [dim]·[/] ".join(parts)
