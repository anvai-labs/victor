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

"""Debug slash commands for workflow breakpoints and debugging.

Commands:
- /debug break: Set a breakpoint on a workflow node
- /debug clear: Clear a breakpoint by ID
- /debug list: List all active breakpoints
- /debug enable: Enable a disabled breakpoint
- /debug disable: Disable a breakpoint
- /debug state: Show current workflow state
- /debug continue: Continue from a breakpoint
- /debug step: Step to next node

Example:
    /debug break analyze_code          # Break before 'analyze_code' node
    /debug break analyze_code --after  # Break after node execution
    /debug clear bp_abc123             # Clear specific breakpoint
    /debug list                        # List all breakpoints
"""

from __future__ import annotations

import logging
from typing import Optional

from rich.panel import Panel
from rich.table import Table

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


def _get_breakpoint_manager(ctx: CommandContext):
    """Get breakpoint manager from agent if available."""
    # Try to get from orchestrator or agent
    if hasattr(ctx.agent, "breakpoint_manager"):
        return ctx.agent.breakpoint_manager
    if hasattr(ctx.agent, "_breakpoint_manager"):
        return ctx.agent._breakpoint_manager
    # Try through workflow engine
    if hasattr(ctx.agent, "workflow_engine"):
        engine = ctx.agent.workflow_engine
        if hasattr(engine, "breakpoint_manager"):
            return engine.breakpoint_manager
    return None


@register_command
class DebugCommand(BaseSlashCommand):
    """Manage workflow debugging and breakpoints."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="debug",
            description="Manage workflow debugging and breakpoints",
            usage="/debug <break|clear|list|enable|disable|state|continue|step> [args]",
            category="debug",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        if not ctx.args:
            self._show_help(ctx)
            return

        subcommand = ctx.args[0].lower()
        subargs = ctx.args[1:]

        if subcommand == "break":
            self._handle_break(ctx, subargs)
        elif subcommand == "clear":
            self._handle_clear(ctx, subargs)
        elif subcommand == "list":
            self._handle_list(ctx)
        elif subcommand == "enable":
            self._handle_enable(ctx, subargs)
        elif subcommand == "disable":
            self._handle_disable(ctx, subargs)
        elif subcommand == "state":
            self._handle_state(ctx)
        elif subcommand == "continue":
            self._handle_continue(ctx)
        elif subcommand == "step":
            self._handle_step(ctx)
        else:
            ctx.console.print(f"[yellow]Unknown subcommand:[/] {subcommand}")
            self._show_help(ctx)

    def _show_help(self, ctx: CommandContext) -> None:
        """Show debug command help."""
        help_text = """
[bold]Debug Commands[/]

[cyan]/debug break[/] <node_id> [--after] [--condition <expr>] [--ignore <count>]
  Set a breakpoint on a workflow node
  Options:
    --after      Break after node executes (default: before)
    --condition  Python expression to evaluate (e.g., "error_count > 5")
    --ignore     Skip first N hits before breaking
  Examples:
    /debug break analyze_code
    /debug break process_data --after
    /debug break validate --condition "len(errors) > 0"

[cyan]/debug clear[/] <breakpoint_id|all>
  Clear a breakpoint by ID or clear all breakpoints
  Example: /debug clear bp_abc123
           /debug clear all

[cyan]/debug list[/]
  List all active breakpoints with their status

[cyan]/debug enable[/] <breakpoint_id>
  Enable a disabled breakpoint

[cyan]/debug disable[/] <breakpoint_id>
  Disable a breakpoint without removing it

[cyan]/debug state[/]
  Show current workflow execution state and variables

[cyan]/debug continue[/]
  Continue execution from current breakpoint

[cyan]/debug step[/]
  Step to the next workflow node
"""
        ctx.console.print(Panel(help_text, title="Debug Help", border_style="blue"))

    def _handle_break(self, ctx: CommandContext, args: list[str]) -> None:
        """Handle setting a breakpoint."""
        if not args:
            ctx.console.print("[yellow]Usage:[/] /debug break <node_id> [options]")
            return

        manager = _get_breakpoint_manager(ctx)
        if not manager:
            ctx.console.print(
                "[yellow]Breakpoint manager not available.[/] "
                "Ensure workflow debugging is enabled."
            )
            return

        # Parse arguments
        node_id = args[0]
        position = "before"
        condition_expr: Optional[str] = None
        ignore_count = 0

        i = 1
        while i < len(args):
            arg = args[i]
            if arg == "--after":
                position = "after"
            elif arg == "--condition" and i + 1 < len(args):
                i += 1
                condition_expr = args[i]
            elif arg == "--ignore" and i + 1 < len(args):
                i += 1
                try:
                    ignore_count = int(args[i])
                except ValueError:
                    ctx.console.print(f"[yellow]Invalid ignore count:[/] {args[i]}")
                    return
            i += 1

        try:
            from victor.framework.debugging.breakpoints import (
                BreakpointPosition,
                BreakpointType,
            )

            # Parse position
            bp_position = (
                BreakpointPosition.AFTER if position == "after" else BreakpointPosition.BEFORE
            )

            # Create condition function if expression provided
            condition_fn = None
            if condition_expr:
                # Create a safe condition function
                def make_condition(expr: str):
                    def condition(state):
                        try:
                            # Limited evaluation context
                            return eval(expr, {"__builtins__": {}}, state)
                        except Exception:
                            return False

                    return condition

                condition_fn = make_condition(condition_expr)

            # Set the breakpoint
            bp = manager.set_breakpoint(
                node_id=node_id,
                position=bp_position,
                condition=condition_fn,
                bp_type=(BreakpointType.CONDITIONAL if condition_fn else BreakpointType.NODE),
                ignore_count=ignore_count,
            )

            ctx.console.print(
                Panel(
                    f"Breakpoint set successfully!\n\n"
                    f"[bold]ID:[/] {bp.id}\n"
                    f"[bold]Node:[/] {node_id}\n"
                    f"[bold]Position:[/] {position}\n"
                    f"[bold]Condition:[/] {condition_expr or '(none)'}\n"
                    f"[bold]Ignore Count:[/] {ignore_count}\n\n"
                    f"[dim]Use '/debug clear {bp.id}' to remove[/]",
                    title="Breakpoint Set",
                    border_style="green",
                )
            )

        except Exception as e:
            ctx.console.print(f"[red]Error setting breakpoint:[/] {e}")
            logger.exception("Breakpoint set error")

    def _handle_clear(self, ctx: CommandContext, args: list[str]) -> None:
        """Handle clearing breakpoints."""
        if not args:
            ctx.console.print("[yellow]Usage:[/] /debug clear <breakpoint_id|all>")
            return

        manager = _get_breakpoint_manager(ctx)
        if not manager:
            ctx.console.print("[yellow]Breakpoint manager not available.[/]")
            return

        target = args[0]

        try:
            if target.lower() == "all":
                # Clear all breakpoints
                breakpoints = manager.list_breakpoints()
                count = len(breakpoints)
                for bp in breakpoints:
                    manager.clear_breakpoint(bp.id)
                ctx.console.print(f"[green]Cleared {count} breakpoint(s)[/]")
            else:
                # Clear specific breakpoint
                if manager.clear_breakpoint(target):
                    ctx.console.print(f"[green]Breakpoint {target} cleared[/]")
                else:
                    ctx.console.print(f"[yellow]Breakpoint not found:[/] {target}")

        except Exception as e:
            ctx.console.print(f"[red]Error clearing breakpoint:[/] {e}")
            logger.exception("Breakpoint clear error")

    def _handle_list(self, ctx: CommandContext) -> None:
        """Handle listing breakpoints."""
        manager = _get_breakpoint_manager(ctx)
        if not manager:
            ctx.console.print("[yellow]Breakpoint manager not available.[/]")
            return

        try:
            breakpoints = manager.list_breakpoints()

            if not breakpoints:
                ctx.console.print("[dim]No breakpoints set.[/]")
                return

            table = Table(title="Active Breakpoints")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Node", style="green")
            table.add_column("Position", style="yellow")
            table.add_column("Type")
            table.add_column("Enabled", justify="center")
            table.add_column("Hits", justify="right")
            table.add_column("Ignore", justify="right")

            for bp in breakpoints:
                table.add_row(
                    bp.id[:12] + "...",
                    bp.node_id or "(any)",
                    bp.position.value,
                    bp.type.value,
                    "✓" if bp.enabled else "✗",
                    str(bp.hit_count),
                    str(bp.ignore_count),
                )

            ctx.console.print(table)

        except Exception as e:
            ctx.console.print(f"[red]Error listing breakpoints:[/] {e}")
            logger.exception("Breakpoint list error")

    def _handle_enable(self, ctx: CommandContext, args: list[str]) -> None:
        """Handle enabling a breakpoint."""
        if not args:
            ctx.console.print("[yellow]Usage:[/] /debug enable <breakpoint_id>")
            return

        manager = _get_breakpoint_manager(ctx)
        if not manager:
            ctx.console.print("[yellow]Breakpoint manager not available.[/]")
            return

        bp_id = args[0]

        try:
            if manager.enable_breakpoint(bp_id):
                ctx.console.print(f"[green]Breakpoint {bp_id} enabled[/]")
            else:
                ctx.console.print(f"[yellow]Breakpoint not found:[/] {bp_id}")

        except Exception as e:
            ctx.console.print(f"[red]Error enabling breakpoint:[/] {e}")
            logger.exception("Breakpoint enable error")

    def _handle_disable(self, ctx: CommandContext, args: list[str]) -> None:
        """Handle disabling a breakpoint."""
        if not args:
            ctx.console.print("[yellow]Usage:[/] /debug disable <breakpoint_id>")
            return

        manager = _get_breakpoint_manager(ctx)
        if not manager:
            ctx.console.print("[yellow]Breakpoint manager not available.[/]")
            return

        bp_id = args[0]

        try:
            if manager.disable_breakpoint(bp_id):
                ctx.console.print(f"[green]Breakpoint {bp_id} disabled[/]")
            else:
                ctx.console.print(f"[yellow]Breakpoint not found:[/] {bp_id}")

        except Exception as e:
            ctx.console.print(f"[red]Error disabling breakpoint:[/] {e}")
            logger.exception("Breakpoint disable error")

    def _handle_state(self, ctx: CommandContext) -> None:
        """Handle showing current workflow state."""
        # Try to get workflow state from various sources
        state = None

        if hasattr(ctx.agent, "workflow_state"):
            state = ctx.agent.workflow_state
        elif hasattr(ctx.agent, "_current_workflow_state"):
            state = ctx.agent._current_workflow_state
        elif hasattr(ctx.agent, "workflow_engine"):
            engine = ctx.agent.workflow_engine
            if hasattr(engine, "current_state"):
                state = engine.current_state

        if not state:
            ctx.console.print("[dim]No active workflow state available.[/]")
            return

        try:
            # Format state as a table
            table = Table(title="Workflow State")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Type", style="dim")

            if isinstance(state, dict):
                for key, value in state.items():
                    # Truncate long values
                    value_str = str(value)
                    if len(value_str) > 80:
                        value_str = value_str[:77] + "..."
                    table.add_row(
                        str(key),
                        value_str,
                        type(value).__name__,
                    )
            else:
                table.add_row("(state)", str(state)[:100], type(state).__name__)

            ctx.console.print(table)

        except Exception as e:
            ctx.console.print(f"[red]Error displaying state:[/] {e}")
            logger.exception("State display error")

    def _handle_continue(self, ctx: CommandContext) -> None:
        """Handle continuing from a breakpoint."""
        # Signal continue to workflow engine
        if hasattr(ctx.agent, "workflow_engine"):
            engine = ctx.agent.workflow_engine
            if hasattr(engine, "debug_continue"):
                try:
                    engine.debug_continue()
                    ctx.console.print("[green]Continuing workflow execution...[/]")
                    return
                except Exception as e:
                    ctx.console.print(f"[red]Error continuing:[/] {e}")
                    return

        # Try alternative signal mechanism
        if hasattr(ctx.agent, "_debug_signal"):
            ctx.agent._debug_signal = "continue"
            ctx.console.print("[green]Continue signal sent[/]")
            return

        ctx.console.print("[yellow]No active debug session to continue.[/]")

    def _handle_step(self, ctx: CommandContext) -> None:
        """Handle stepping to next node."""
        # Signal step to workflow engine
        if hasattr(ctx.agent, "workflow_engine"):
            engine = ctx.agent.workflow_engine
            if hasattr(engine, "debug_step"):
                try:
                    engine.debug_step()
                    ctx.console.print("[green]Stepping to next node...[/]")
                    return
                except Exception as e:
                    ctx.console.print(f"[red]Error stepping:[/] {e}")
                    return

        # Try alternative signal mechanism
        if hasattr(ctx.agent, "_debug_signal"):
            ctx.agent._debug_signal = "step"
            ctx.console.print("[green]Step signal sent[/]")
            return

        ctx.console.print("[yellow]No active debug session to step.[/]")
