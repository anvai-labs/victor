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

"""Status command for showing plan execution status."""

from __future__ import annotations

import logging

from victor.agent.planning.base import StepStatus
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class StatusCommand(BaseSlashCommand):
    """Show current plan execution status."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="status",
            description="Show plan execution status",
            usage="/status",
            category="planning",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        # Try to get current plan from agent
        conversation_controller = getattr(ctx.agent, "conversation_controller", None)
        if not conversation_controller:
            ctx.console.print("[yellow]No conversation controller available.[/]")
            return

        current_plan = getattr(conversation_controller, "current_plan", None)
        if not current_plan:
            ctx.console.print(
                "[yellow]No active plan.[/]\n"
                "[dim]Create a plan with /plan <task> or load with /plan load <id>[/]"
            )
            return

        # Show plan status
        completed = len([s for s in current_plan.steps if s.status == StepStatus.COMPLETED])
        total = len(current_plan.steps)
        failed = len([s for s in current_plan.steps if s.status == StepStatus.FAILED])
        blocked = len([s for s in current_plan.steps if s.status == StepStatus.BLOCKED])
        pending = len([s for s in current_plan.steps if s.status == StepStatus.PENDING])
        in_progress = len([s for s in current_plan.steps if s.status == StepStatus.IN_PROGRESS])

        from rich.panel import Panel

        status_lines = [
            f"[bold]Goal:[/] {current_plan.goal[:100]}{'...' if len(current_plan.goal) > 100 else ''}",
            f"[bold]Status:[/] {current_plan.status.value if hasattr(current_plan.status, 'value') else current_plan.status}",
            f"[bold]Progress:[/] [cyan]{completed}/{total}[/] steps completed",
        ]

        if failed > 0:
            status_lines.append(f"[red]{failed}[/] steps failed")
        if blocked > 0:
            status_lines.append(f"[yellow]{blocked}[/] steps blocked")
        if pending > 0:
            status_lines.append(f"[dim]{pending}[/] steps pending")
        if in_progress > 0:
            status_lines.append(f"[blue]{in_progress}[/] steps in progress")

        ctx.console.print(Panel("\n".join(status_lines), title="Plan Status", border_style="cyan"))

        # Show incomplete steps
        incomplete = [
            s
            for s in current_plan.steps
            if s.status in {StepStatus.PENDING, StepStatus.IN_PROGRESS}
        ]
        if incomplete:
            ctx.console.print(f"\n[bold]Next Steps:[/]")
            for step in incomplete[:5]:  # Show next 5
                status_icon = {
                    StepStatus.PENDING: "⏳",
                    StepStatus.IN_PROGRESS: "🔄",
                }.get(step.status, "❓")
                status_color = {
                    StepStatus.PENDING: "dim",
                    StepStatus.IN_PROGRESS: "cyan",
                }.get(step.status, "white")

                step_desc = (
                    step.description[:70] + "..."
                    if len(step.description) > 70
                    else step.description
                )
                ctx.console.print(
                    f"  {status_icon} [{status_color}]{step.desc}[/]"
                    if hasattr(step, "desc")
                    else f"  {status_icon} [{status_color}]{step_desc}[/]"
                )

            if len(incomplete) > 5:
                ctx.console.print(f"  ... and [dim]{len(incomplete) - 5}[/] more")

        # Show failed steps
        failed_steps = [s for s in current_plan.steps if s.status == StepStatus.FAILED]
        if failed_steps:
            ctx.console.print(f"\n[bold]Failed Steps:[/]")
            for step in failed_steps[:3]:
                error_msg = (
                    step.result.error
                    if step.result and hasattr(step.result, "error")
                    else "Unknown error"
                )
                error_msg = error_msg[:60] + "..." if len(error_msg) > 60 else error_msg
                step_desc = (
                    step.description[:50] + "..."
                    if len(step.description) > 50
                    else step.description
                )
                ctx.console.print(f"  ❌ [red]{step_desc}[/]")
                ctx.console.print(f"     [dim]Error: {error_msg}[/]")

            if len(failed_steps) > 3:
                ctx.console.print(f"  ... and [dim]{len(failed_steps) - 3}[/] more failed steps")

        # Show blocked steps
        blocked_steps = [s for s in current_plan.steps if s.status == StepStatus.BLOCKED]
        if blocked_steps:
            ctx.console.print(f"\n[bold]Blocked Steps:[/]")
            for step in blocked_steps[:3]:
                step_desc = (
                    step.description[:50] + "..."
                    if len(step.description) > 50
                    else step.description
                )
                ctx.console.print(f"  🚫 [yellow]{step_desc}[/]")

            if len(blocked_steps) > 3:
                ctx.console.print(f"  ... and [dim]{len(blocked_steps) - 3}[/] more blocked steps")
            ctx.console.print("\n[dim]Blocked steps require approval. Use /resume to resume.[/]")
