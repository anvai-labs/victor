# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
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

"""``/rate`` — manual RL feedback for the FEP-0012 reward loop.

Records whether the current session resolved the user's task. The yes/no
answer is the reward label: it flows (via :func:`record_session_outcome` →
``decision_outcome``) into classifier training, joining the session's logged
decisions by ``session_id``. Works regardless of the auto-prompt
(``enable_rl_feedback_prompt``) — use it to label a turn the auto-prompt
skipped or to correct an earlier rating.
"""

from __future__ import annotations

import logging

from rich.prompt import Confirm

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class RateCommand(BaseSlashCommand):
    """Record RL task-completion feedback for the current session."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="rate",
            description="Record whether this session resolved your task (RL feedback)",
            usage="/rate",
            category="metrics",
            requires_agent=True,
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        orch = self._get_orchestrator(ctx)
        sid = self._get_session_id(ctx, orch)
        if not sid:
            ctx.console.print("[yellow]No active session to rate.[/]")
            return

        success = Confirm.ask("Did this resolve your task?", default=True)
        try:
            # Log a task_completion decision so the key head gets a row even when
            # the auto-prompt was off/skipped. result reflects the user's verdict.
            from victor.agent.decisions.chain import log_decision

            log_decision(
                "task_completion",
                {"source": "manual_rate"},
                result="fulfilled" if success else "not_fulfilled",
                source="manual_rate",
                session_id_override=sid,
            )
            from victor.agent.decisions.outcome import record_session_outcome

            count = record_session_outcome(
                sid, success=success, quality_score=1.0 if success else 0.0
            )
            ctx.console.print(
                f"[green]✓[/] Recorded {'success' if success else 'failure'} "
                f"for {count} decision(s) in session {sid[:8]}."
            )
        except Exception as exc:
            ctx.console.print(f"[red]Failed to record outcome:[/] {exc}")
            logger.exception("rate command failed")

    def _get_orchestrator(self, ctx: CommandContext):
        """Reach the orchestrator defensively (agent may or may not wrap it)."""
        agent = ctx.agent
        if agent is not None and hasattr(agent, "get_orchestrator"):
            try:
                return agent.get_orchestrator()
            except Exception:
                return None
        return agent

    def _get_session_id(self, ctx: CommandContext, orch) -> str:
        """Return the session_id the session's decisions were logged under."""
        for obj in (orch, ctx.agent):
            sid = getattr(obj, "active_session_id", None)
            if sid:
                return str(sid)
        return ""
