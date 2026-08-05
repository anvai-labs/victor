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

"""Agent mode slash commands: mode, build, plan, review, delegate, explore."""

from __future__ import annotations

import logging

from rich.panel import Panel

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class ModeCommand(BaseSlashCommand):
    """Switch agent mode."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="mode",
            description="Switch agent mode (build/plan/review/delegate/explore)",
            usage="/mode [build|plan|review|delegate|explore]",
            aliases=["m"],
            category="mode",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        from victor.agent.mode_controller import AgentMode, get_mode_controller

        # Safely get mode controller - try multiple methods
        mode_controller = None

        # Method 1: Direct attribute (if agent is AgentOrchestrator with ModeAwareMixin)
        if hasattr(ctx.agent, "mode_controller"):
            try:
                mode_controller = ctx.agent.mode_controller
            except Exception as e:
                logger.debug(f"Failed to access mode_controller attribute: {e}")

        # Method 2: Get from DI container/singleton
        if mode_controller is None:
            try:
                mode_controller = get_mode_controller()
            except Exception as e:
                logger.debug(f"Failed to get mode controller from singleton: {e}")

        if not ctx.args:
            # Show current mode using public interface
            current_mode = (
                mode_controller.current_mode
                if mode_controller
                else AgentMode.BUILD  # Default fallback
            )

            ctx.console.print(
                Panel(
                    f"[bold]Current Mode:[/] [cyan]{current_mode.value}[/]\n\n"
                    "[bold]Available Modes:[/]\n"
                    "  [cyan]build[/]    - Implementation mode (default)\n"
                    "  [cyan]plan[/]     - Planning and research mode\n"
                    "  [cyan]review[/]   - Findings-first review and validation mode\n"
                    "  [cyan]delegate[/] - Parallel-work delegation and merge planning mode\n"
                    "  [cyan]explore[/]  - Advanced code navigation and analysis mode\n\n"
                    "[dim]Switch with: /mode <mode_name>[/]",
                    title="Agent Mode",
                    border_style="cyan",
                )
            )
            return

        mode_name = ctx.args[0].lower()
        valid_modes = {"build", "plan", "review", "delegate", "explore"}

        if mode_name not in valid_modes:
            ctx.console.print(f"[red]Invalid mode:[/] {mode_name}")
            ctx.console.print(f"[dim]Available modes: {', '.join(valid_modes)}[/]")
            return

        try:
            new_mode = AgentMode(mode_name)

            if mode_controller:
                mode_controller.switch_mode(new_mode)
            else:
                ctx.console.print("[yellow]Mode controller not available, mode not switched[/]")
                return

            if hasattr(ctx.agent, "refresh_system_prompt"):
                ctx.agent.refresh_system_prompt()

            ctx.console.print(f"[green]Switched to mode:[/] [cyan]{mode_name}[/]")

            # Show mode-specific hints
            hints = {
                "build": "Implementation mode: Focused on writing and modifying code",
                "plan": "Planning mode: Research and design before implementation",
                "review": "Review mode: Diagnose issues and report findings before changes",
                "delegate": "Delegate mode: Break work into scoped parallel tasks and merge plans",
                "explore": "Explore mode: Code navigation and analysis without changes",
            }
            ctx.console.print(f"[dim]{hints.get(mode_name, '')}[/]")

        except Exception as e:
            ctx.console.print(f"[red]Failed to switch mode:[/] {e}")


@register_command
class BuildCommand(BaseSlashCommand):
    """Switch to build mode for implementation."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="build",
            description="Switch to build mode for implementation",
            usage="/build",
            category="mode",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        # Delegate to mode command
        ctx.args = ["build"]
        ModeCommand().execute(ctx)


@register_command
class ExploreCommand(BaseSlashCommand):
    """Switch to explore mode for code navigation."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="explore",
            description="Switch to explore mode for code navigation",
            usage="/explore",
            category="mode",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        # Delegate to mode command
        ctx.args = ["explore"]
        ModeCommand().execute(ctx)


@register_command
class ReviewCommand(BaseSlashCommand):
    """Switch to review mode for diagnostics and findings-first feedback."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="review",
            description="Switch to review mode for diagnostics and code review",
            usage="/review",
            category="mode",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        ctx.args = ["review"]
        ModeCommand().execute(ctx)


@register_command
class DelegateCommand(BaseSlashCommand):
    """Switch to delegate mode for work decomposition and merge planning."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="delegate",
            description="Switch to delegate mode for parallel worker planning",
            usage="/delegate",
            category="mode",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        ctx.args = ["delegate"]
        ModeCommand().execute(ctx)


@register_command
class PlanCommand(BaseSlashCommand):
    """Enter planning mode and manage plans.

    Subcommands:
        /plan              - Show current mode and plan status
        /plan <task>       - Start planning a task
        /plan save [name]  - Save current plan to disk
        /plan load <id>    - Load a plan by ID or filename
        /plan list         - List saved plans
        /plan show         - Show current plan details
    """

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="plan",
            description="Enter planning mode and manage plans",
            usage="/plan [save|load|list|show|<task>]",
            category="mode",
            requires_agent=True,
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        # Check for subcommands
        if ctx.args:
            subcommand = ctx.args[0].lower()
            if subcommand == "save":
                await self._save_plan(ctx)
                return
            elif subcommand == "load":
                await self._load_plan(ctx)
                return
            elif subcommand == "list":
                self._list_plans(ctx)
                return
            elif subcommand == "show":
                self._show_plan(ctx)
                return

        from victor.agent.mode_controller import AgentMode, get_mode_controller

        # Safely get mode controller - try multiple methods
        mode_controller = None

        # Method 1: Direct attribute (if agent is AgentOrchestrator with ModeAwareMixin)
        if hasattr(ctx.agent, "mode_controller"):
            try:
                mode_controller = ctx.agent.mode_controller
            except Exception as e:
                logger.debug(f"Failed to access mode_controller attribute: {e}")

        # Method 2: Get from DI container/singleton
        if mode_controller is None:
            try:
                mode_controller = get_mode_controller()
            except Exception as e:
                logger.debug(f"Failed to get mode controller from singleton: {e}")

        if mode_controller:
            mode_controller.switch_mode(AgentMode.PLAN)
        else:
            ctx.console.print("[yellow]Mode controller not available[/]")
            return

        if hasattr(ctx.agent, "refresh_system_prompt"):
            ctx.agent.refresh_system_prompt()

        ctx.console.print("[green]Switched to planning mode[/]")
        ctx.console.print("[dim]Sandbox edits enabled in .victor/sandbox/ directory[/]")

        # If task description provided, start planning
        if ctx.args:
            task = " ".join(ctx.args)
            ctx.console.print(f"[dim]Planning task: {task}[/]")

            planning_prompt = (
                f"I need to plan the following task before implementation:\n\n"
                f"{task}\n\n"
                f"Please analyze this task and provide:\n"
                f"1. Understanding of requirements\n"
                f"2. Relevant files and code sections to examine\n"
                f"3. Proposed approach and implementation steps\n"
                f"4. Potential challenges and considerations\n"
            )

            try:
                response = await ctx.agent.chat(planning_prompt)
                from rich.markdown import Markdown

                ctx.console.print(
                    Panel(
                        Markdown(response.content),
                        title="Planning Analysis",
                        border_style="blue",
                    )
                )
            except Exception as e:
                ctx.console.print(f"[red]Planning failed:[/] {e}")
        else:
            ctx.console.print(
                "\n[bold]Plan Commands:[/]\n"
                "  /plan <task>       - Start planning a task\n"
                "  /plan save [name]  - Save current plan\n"
                "  /plan load <id>    - Load a saved plan\n"
                "  /plan list         - List saved plans\n"
                "  /plan show         - Show current plan\n"
            )

    async def _save_plan(self, ctx: CommandContext) -> None:
        """Save the current plan to disk."""
        from pathlib import Path

        from victor.agent.planning.store import get_plan_store

        # Get current plan from agent using public interface
        conversation_controller = getattr(ctx.agent, "conversation_controller", None)
        current_plan = (
            getattr(conversation_controller, "current_plan", None)
            if conversation_controller
            else None
        )

        if not current_plan:
            # Check if there's a plan in conversation context using public interface
            conv_controller = ctx.agent.conversation_controller
            if conv_controller:
                current_plan = conv_controller.current_plan

        if not current_plan:
            ctx.console.print(
                "[yellow]No active plan to save.[/]\n"
                "[dim]Start planning with /plan <task> first.[/]"
            )
            return

        try:
            store = get_plan_store(Path.cwd())

            # Optional custom filename
            filename = ctx.args[1] if len(ctx.args) > 1 else None
            filepath = store.save(current_plan, filename)

            ctx.console.print(
                Panel(
                    f"[green]Plan saved successfully![/]\n\n"
                    f"[bold]ID:[/] {current_plan.id}\n"
                    f"[bold]Goal:[/] {current_plan.goal[:60]}{'...' if len(current_plan.goal) > 60 else ''}\n"
                    f"[bold]File:[/] {filepath}\n\n"
                    f"[dim]Load with: /plan load {current_plan.id}[/]",
                    title="Plan Saved",
                    border_style="green",
                )
            )
        except Exception as e:
            ctx.console.print(f"[red]Failed to save plan:[/] {e}")

    async def _load_plan(self, ctx: CommandContext) -> None:
        """Load a plan from disk."""
        from pathlib import Path

        from victor.agent.planning.store import get_plan_store

        if len(ctx.args) < 2:
            ctx.console.print(
                "[yellow]Usage:[/] /plan load <id_or_filename>\n"
                "[dim]Use /plan list to see available plans.[/]"
            )
            return

        plan_id = ctx.args[1]

        try:
            store = get_plan_store(Path.cwd())
            plan = store.load(plan_id)

            if not plan:
                ctx.console.print(f"[red]Plan not found:[/] {plan_id}")
                ctx.console.print("[dim]Use /plan list to see available plans.[/]")
                return

            # Attach plan to agent using public interface
            conversation_controller = getattr(ctx.agent, "conversation_controller", None)
            if conversation_controller:
                conversation_controller.set_current_plan(plan)

            from rich.markdown import Markdown

            ctx.console.print(
                Panel(
                    Markdown(plan.to_markdown()),
                    title=f"Loaded Plan: {plan.id}",
                    border_style="blue",
                )
            )
            ctx.console.print(
                "\n[green]Plan loaded.[/] Use [cyan]/mode build[/] to start implementation."
            )
        except Exception as e:
            ctx.console.print(f"[red]Failed to load plan:[/] {e}")

    def _list_plans(self, ctx: CommandContext) -> None:
        """List saved plans with enhanced table styling."""
        from pathlib import Path

        from victor.agent.planning.store import get_plan_store
        from victor.ui.rendering.table_builder import (
            create_plan_list_table,
            format_plan_status,
        )

        try:
            store = get_plan_store(Path.cwd())
            plans = store.list_plans(limit=20)

            if not plans:
                ctx.console.print(
                    "[dim]No saved plans found.[/]\n"
                    "[dim]Create a plan with /plan <task> then save with /plan save[/]"
                )
                return

            table = create_plan_list_table(title="Saved Plans")

            for p in plans:
                # Format status with icon and color
                status_formatted = format_plan_status(p.get("status", "unknown"))
                # Count tasks if available
                task_count = p.get("task_count", 0)
                tasks_str = f"{task_count}" if task_count else "—"

                table.add_row(
                    p["id"][:8] + "...",
                    p["goal"],
                    p["created_at"][:16],
                    status_formatted,
                    tasks_str,
                )

            ctx.console.print(table)
            ctx.console.print("\n[dim]Load a plan with: /plan load <id>[/]")
        except Exception as e:
            ctx.console.print(f"[red]Failed to list plans:[/] {e}")

    def _show_plan(self, ctx: CommandContext) -> None:
        """Show the current plan with enhanced table display."""
        conversation_controller = getattr(ctx.agent, "conversation_controller", None)
        current_plan = (
            getattr(conversation_controller, "current_plan", None)
            if conversation_controller
            else None
        )

        if not current_plan:
            # Use public interface for conversation controller and current_plan
            conv_controller = ctx.agent.conversation_controller
            if conv_controller:
                current_plan = conv_controller.current_plan

        if not current_plan:
            ctx.console.print(
                "[dim]No active plan.[/]\n" "[dim]Start planning with /plan <task>[/]"
            )
            return

        from rich.markdown import Markdown
        from rich.panel import Panel

        from victor.ui.rendering.table_builder import (
            create_plan_task_table,
            format_task_status,
        )

        # Show plan header
        progress = current_plan.progress_percentage()
        total_steps = len(current_plan.steps)
        completed_steps = len(current_plan.get_completed_steps())

        ctx.console.print(
            Panel(
                f"[bold]{current_plan.goal}[/]\n\n"
                f"Progress: [cyan]{completed_steps}/{total_steps}[/] "
                f"([cyan]{progress:.0f}%[/])",
                title=f"Plan: {current_plan.id[:12]}...",
                border_style="blue",
            )
        )

        # Show plan steps in a table if there are steps
        if current_plan.steps:
            table = create_plan_task_table(title="Tasks")

            for idx, step in enumerate(current_plan.steps, 1):
                # Format status with icon and color
                status_str = (
                    step.status.value if hasattr(step.status, "value") else str(step.status)
                )
                status_formatted = format_task_status(status_str)

                # Build details string (dependencies, estimated calls)
                details_parts = []
                if hasattr(step, "depends_on") and step.depends_on:
                    deps = ", ".join(step.depends_on[:2])
                    if len(step.depends_on) > 2:
                        deps += f" +{len(step.depends_on) - 2}"
                    details_parts.append(f"after: {deps}")

                details = " • ".join(details_parts) if details_parts else ""

                # Truncate description for table
                description = (
                    step.description[:60] + "..."
                    if len(step.description) > 60
                    else step.description
                )

                table.add_row(
                    str(idx),
                    description,
                    status_formatted,
                    details,
                )

            ctx.console.print(table)
        else:
            # Fallback to markdown display if no steps
            ctx.console.print(
                Panel(
                    Markdown(current_plan.to_markdown()),
                    title=f"Plan: {current_plan.id}",
                    border_style="blue",
                )
            )
