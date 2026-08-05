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

"""Resume command for resuming paused plan execution."""

from __future__ import annotations

import asyncio
import logging

from rich.panel import Panel

from victor.agent.planning.base import ExecutionPlan, StepStatus
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class ResumeCommand(BaseSlashCommand):
    """Resume plan execution from where it left off."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="resume",
            description="Resume executing the current plan",
            usage="/resume [step_number]",
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

        current_plan: ExecutionPlan = getattr(conversation_controller, "current_plan", None)
        if not current_plan:
            ctx.console.print(
                "[red]No plan found to resume.[/]\n"
                "[dim]Create a plan with /plan <task> or load with /plan load <id>[/]"
            )
            return

        # Check if plan has incomplete steps
        incomplete = [
            s
            for s in current_plan.steps
            if s.status in {StepStatus.PENDING, StepStatus.IN_PROGRESS}
        ]
        if not incomplete:
            ctx.console.print(
                "[green]All plan steps completed![/]\n"
                "[dim]Use /plan show to see final results.[/]"
            )
            return

        # Resume execution
        ctx.console.print(
            f"[cyan]Continuing plan:[/] {current_plan.goal[:80]}{'...' if len(current_plan.goal) > 80 else ''}"
        )
        ctx.console.print(f"[dim]Incomplete steps: {len(incomplete)}/{len(current_plan.steps)}[/]")

        # Parse optional step number
        start_from = None
        if ctx.args:
            try:
                start_from = int(ctx.args[0])
                if start_from < 1 or start_from > len(current_plan.steps):
                    ctx.console.print(
                        f"[red]Invalid step number:[/] {start_from} (must be 1-{len(current_plan.steps)})"
                    )
                    return
                start_from -= 1  # Convert to 0-indexed
            except ValueError:
                ctx.console.print(f"[red]Invalid step number:[/] {ctx.args[0]}")
                return

        # Execute the plan asynchronously
        asyncio.create_task(self._execute_plan_async(ctx, current_plan, start_from))

        ctx.console.print("[dim]Plan execution resumed in background...[/]")

    async def _execute_plan_async(
        self,
        ctx: CommandContext,
        plan: ExecutionPlan,
        start_from: int | None = None,
    ):
        """Execute plan asynchronously."""
        try:
            # Get autonomous planner from agent
            orchestrator = getattr(ctx.agent, "orchestrator", None) or ctx.agent
            planner = getattr(orchestrator, "autonomous_planner", None)

            if not planner:
                ctx.console.print(
                    "[yellow]Autonomous planner not available.[/]\n"
                    "[dim]Plan execution requires planner support.[/]"
                )
                return

            # Reset failed/blocked steps if starting from a specific step
            if start_from is not None:
                for i, step in enumerate(plan.steps):
                    if i >= start_from and step.status in {
                        StepStatus.FAILED,
                        StepStatus.BLOCKED,
                    }:
                        step.status = StepStatus.PENDING

            # Execute with auto-approve for research/planning steps
            result = await planner.execute_plan(
                plan,
                auto_approve=False,  # Use smart defaults from _default_approval
                progress_callback=lambda step, status: self._on_progress(ctx, step, status),
            )

            # Show final results
            if result.success:
                ctx.console.print(
                    Panel(
                        f"[green]✓ Plan completed successfully![/]\n\n"
                        f"[bold]Steps:[/] {result.steps_completed}/{result.total_steps}\n"
                        f"[bold]Duration:[/] {result.total_duration:.1f}s\n"
                        f"[bold]Tool calls:[/] {result.total_tool_calls}",
                        title="Plan Results",
                        border_style="green",
                    )
                )
            else:
                ctx.console.print(
                    Panel(
                        f"[yellow]⚠ Plan execution incomplete[/]\n\n"
                        f"[bold]Steps:[/] {result.steps_completed}/{result.total_steps}\n"
                        f"[bold]Failed:[/] {result.steps_failed}\n"
                        f"[bold]Duration:[/] {result.total_duration:.1f}s\n"
                        f"[bold]Tool calls:[/] {result.total_tool_calls}",
                        title="Plan Results",
                        border_style="yellow",
                    )
                )

                if result.final_output:
                    ctx.console.print(f"\n[dim]Final output:[/]\n{result.final_output}")

        except Exception as e:
            logger.error(f"Plan execution failed: {e}", exc_info=True)
            ctx.console.print(f"[red]Plan execution failed:[/] {e}")

    def _on_progress(self, ctx: CommandContext, step, status: StepStatus):
        """Progress callback for plan execution."""
        if status == StepStatus.IN_PROGRESS:
            ctx.console.print(f"[dim]→ Executing:[/] {step.description[:60]}...")
        elif status == StepStatus.COMPLETED:
            ctx.console.print(f"[green]✓[/] {step.description[:60]}...")
        elif status == StepStatus.FAILED:
            error_msg = (
                step.result.error
                if step.result and hasattr(step.result, "error")
                else "Unknown error"
            )
            ctx.console.print(f"[red]✗[/] {step.description[:60]}...")
            ctx.console.print(f"  [dim]Error: {error_msg[:80]}...[/]")
        elif status == StepStatus.BLOCKED:
            ctx.console.print(f"[yellow]⏸[/] {step.description[:60]}... [dim](blocked)[/]")
