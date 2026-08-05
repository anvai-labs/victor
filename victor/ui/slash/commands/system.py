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

"""System slash commands: help, config, status, exit, clear, theme."""

from __future__ import annotations

from typing import Optional, Tuple

from rich.panel import Panel
from rich.table import Table

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import get_command_registry, register_command


def get_builtin_tool_help(name: str) -> Optional[Tuple[str, str]]:
    """Return embedded help text for common tools exposed via prompts.

    Slash help is command-oriented, but some high-value tools are easier to
    discover if `/help` can explain their primary usage patterns directly.
    """
    normalized = name.strip().lower().lstrip("/")
    if normalized != "graph":
        return None

    content = (
        "[bold]graph[/]\n\n"
        "Use this tool for call-graph and execution-flow questions.\n\n"
        "[dim]Common modes:[/]\n"
        'graph(mode="callers", node="parse_json", depth=2) - who calls a function\n'
        'graph(mode="callees", node="main", depth=2) - what a function calls\n'
        'graph(mode="trace", node="main", depth=3) - trace execution from an entry point\n\n'
        "[dim]Useful prompts:[/]\n"
        '"Who calls parse_json?"\n'
        '"What does main call?"\n'
        '"Trace execution from main"\n\n'
        "[dim]Tip:[/] Add [bold]file[/] when the symbol name is ambiguous."
    )
    return ("graph", content)


def print_builtin_tool_help(ctx: CommandContext, name: str) -> bool:
    """Print embedded help for a supported tool name."""
    tool_help = get_builtin_tool_help(name)
    if tool_help is None:
        return False

    title, content = tool_help
    ctx.console.print(Panel(content, title=f"Help: {title}", border_style="blue"))
    return True


@register_command
class HelpCommand(BaseSlashCommand):
    """Show available commands or help for a specific command."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="help",
            description="Show available commands",
            usage="/help [command]",
            aliases=["?"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        registry = get_command_registry()

        if ctx.args:
            # Help for specific command
            cmd_name = ctx.args[0].lstrip("/")
            command = registry.get(cmd_name)
            if command:
                meta = command.metadata
                aliases = ", ".join(f"/{a}" for a in meta.aliases) if meta.aliases else "none"
                ctx.console.print(
                    Panel(
                        f"[bold]/{meta.name}[/]\n\n"
                        f"{meta.description}\n\n"
                        f"[dim]Usage:[/] {meta.usage}\n"
                        f"[dim]Aliases:[/] {aliases}",
                        title=f"Help: /{meta.name}",
                        border_style="blue",
                    )
                )
            elif print_builtin_tool_help(ctx, cmd_name):
                return
            else:
                ctx.console.print(f"[yellow]Unknown command:[/] /{cmd_name}")
            return

        # Show all commands grouped by category
        table = Table(title="Available Commands", show_header=True)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description")
        table.add_column("Aliases", style="dim")

        for name, meta in registry.list_commands():
            aliases = ", ".join(f"/{a}" for a in meta.aliases) if meta.aliases else ""
            table.add_row(f"/{name}", meta.description, aliases)

        ctx.console.print(table)
        ctx.console.print("\n[dim]Type /help <command> for more details[/]")


@register_command
class ConfigCommand(BaseSlashCommand):
    """Show current configuration."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="config",
            description="Show current configuration",
            usage="/config",
            aliases=["settings"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        ctx.console.print(
            Panel(
                f"[bold]Provider:[/] {ctx.settings.provider.default_provider}\n"
                f"[bold]Model:[/] {ctx.settings.provider.default_model}\n"
                f"[bold]Ollama URL:[/] {ctx.settings.provider.ollama_base_url}\n"
                f"[bold]Air-gapped:[/] {ctx.settings.security.airgapped_mode}\n"
                f"[bold]Semantic Selection:[/] {ctx.settings.tools.use_semantic_tool_selection}\n"
                f"[bold]Embedding Model:[/] {ctx.settings.search.unified_embedding_model}\n"
                f"[bold]Session Tool Cap:[/] {ctx.settings.tools.tool_call_budget}\n"
                f"[bold]Budget Calibration:[/] "
                f"{'on' if ctx.settings.tools.tool_budget_calibration_enabled else 'off'}"
                f" (min_confidence={ctx.settings.tools.tool_budget_calibration_min_confidence})\n"
                f"[bold]Config Dir:[/] {ctx.settings.get_config_dir()}",
                title="Configuration",
                border_style="blue",
            )
        )


@register_command
class StatusCommand(BaseSlashCommand):
    """Show current session status."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="status",
            description="Show current session status",
            usage="/status",
            aliases=["info"],
            category="system",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        agent = ctx.agent
        history = agent.conversation
        tool_calls = getattr(agent, "tool_calls_used", 0)
        # Pair the counter with the budget it is enforced against. Using
        # settings.tools.tool_call_budget here reported a session cap (2000) as the
        # denominator of a per-turn count.
        tool_budget = getattr(agent, "tool_budget", None)

        tool_calls_line = (
            f"{tool_calls} / {tool_budget}" if tool_budget is not None else f"{tool_calls}"
        )
        content = (
            f"[bold]Provider:[/] {agent.provider_name}\n"
            f"[bold]Model:[/] {agent.model}\n"
            f"[bold]Messages:[/] {history.message_count()}\n"
            f"[bold]Tool Calls:[/] {tool_calls_line}\n"
        )

        # Add conversation state if available
        state_machine = getattr(agent, "conversation_state", None)
        if state_machine:
            stage = state_machine.get_stage()
            content += f"[bold]Stage:[/] {stage.name}\n"

        # Add mode controller info
        mode_controller = getattr(agent, "mode_controller", None)
        if mode_controller:
            current_mode = mode_controller.get_current_mode()
            content += f"[bold]Mode:[/] {current_mode.value}\n"

        # Add RL recommendation if available
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner:
                rec = learner.recommend(agent.provider_name, "coding")
                if rec and rec.confidence > 0:
                    content += f"\n[dim]RL: Using optimal provider (Q={rec.confidence:.2f})[/]"
        except Exception:
            pass

        ctx.console.print(Panel(content, title="Session Status", border_style="blue"))


@register_command
class ClearCommand(BaseSlashCommand):
    """Clear conversation history."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="clear",
            description="Clear conversation history",
            usage="/clear",
            aliases=["reset"],
            category="system",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        ctx.agent.reset_conversation()
        ctx.console.print("[green]Conversation cleared[/]")


@register_command
class ExitCommand(BaseSlashCommand):
    """Exit Victor."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="exit",
            description="Exit Victor",
            usage="/exit",
            aliases=["quit"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        ctx.console.print("[dim]Goodbye![/]")
        raise SystemExit(0)


@register_command
class ThemeCommand(BaseSlashCommand):
    """Toggle between dark and light theme."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="theme",
            description="Toggle between dark and light theme",
            usage="/theme [dark|light]",
            aliases=["dark", "light"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        current_theme = ctx.settings.ui.theme if ctx.settings.ui else "dark"

        if ctx.args:
            new_theme = ctx.args[0].lower()
            if new_theme not in ("dark", "light"):
                ctx.console.print(f"[red]Invalid theme:[/] {new_theme}")
                ctx.console.print("[dim]Available: dark, light[/]")
                return
        else:
            # Toggle
            new_theme = "light" if current_theme == "dark" else "dark"

        ctx.settings.ui.theme = new_theme
        ctx.console.print(f"[green]Theme set to:[/] {new_theme}")


@register_command
class BugCommand(BaseSlashCommand):
    """Report an issue or bug."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="bug",
            description="Report an issue or bug",
            usage="/bug",
            aliases=["issue", "feedback"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        ctx.console.print(
            Panel(
                "[bold]Report Issues[/]\n\n"
                "For bugs or feature requests, please visit:\n\n"
                "[link=https://github.com/vijayksingh/victor/issues]"
                "https://github.com/vijayksingh/victor/issues[/link]\n\n"
                "When reporting, please include:\n"
                "- Victor version ([cyan]victor --version[/])\n"
                "- Provider and model being used\n"
                "- Steps to reproduce the issue\n"
                "- Any error messages",
                title="Report an Issue",
                border_style="yellow",
            )
        )


@register_command
class ApprovalsCommand(BaseSlashCommand):
    """Configure what actions require user approval."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="approvals",
            description="Configure what actions require user approval",
            usage="/approvals [suggest|auto|full-auto]",
            aliases=["safety"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        current_mode = getattr(ctx.settings, "approval_mode", "suggest")

        if not ctx.args:
            # Show current mode
            ctx.console.print(
                Panel(
                    f"[bold]Current Mode:[/] {current_mode}\n\n"
                    "[bold]Available Modes:[/]\n"
                    "  [cyan]suggest[/]   - Ask before any write operation (safest)\n"
                    "  [cyan]auto[/]      - Auto-approve safe operations, ask for others\n"
                    "  [cyan]full-auto[/] - Auto-approve everything (use with caution)",
                    title="Approval Settings",
                    border_style="yellow",
                )
            )
            return

        mode = ctx.args[0].lower()
        if mode not in ("suggest", "auto", "full-auto"):
            ctx.console.print(f"[red]Invalid mode:[/] {mode}")
            ctx.console.print("[dim]Available: suggest, auto, full-auto[/]")
            return

        ctx.settings.approval_mode = mode
        ctx.console.print(f"[green]Approval mode set to:[/] {mode}")

        if mode == "full-auto":
            ctx.console.print(
                "[yellow]Warning: full-auto mode will auto-approve all actions. "
                "Use with caution![/]"
            )


@register_command
class CleanupHistoryCommand(BaseSlashCommand):
    """Clean up CLI chat history to improve typing performance."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="cleanup-history",
            description="Clean up CLI chat history to improve typing performance",
            usage="/cleanup-history",
            aliases=["prune-history"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        from victor.config.settings import get_project_paths

        history_file = get_project_paths().project_victor_dir / "chat_history"

        if not history_file.exists():
            ctx.console.print("[green]✓[/] No history file to clean up")
            return

        # Import the prune function from chat.py
        from victor.ui.commands.chat import _prune_history_file

        with open(history_file) as f:
            original_lines = sum(1 for _ in f)
        _prune_history_file(history_file, max_entries=250)

        with open(history_file) as f:
            new_lines = sum(1 for _ in f)
        removed = original_lines - new_lines

        ctx.console.print("[green]✓[/] Cleaned up history file:")
        ctx.console.print(f"  Entries: {original_lines:,} → {new_lines:,} (removed {removed:,})")
        ctx.console.print(f"  File size: {history_file.stat().st_size / 1024:.1f} KB")
        ctx.console.print()
        ctx.console.print("[dim]Typing should feel snappier now![/]")
