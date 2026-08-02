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

"""Slash command handler for orchestrating command execution.

This module provides the main SlashCommandHandler class that:
- Parses command input
- Routes to appropriate command handlers
- Manages command execution context
- Provides backward compatibility with legacy code
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    # ✅ PROPER: Only import Settings for type hints (no agent layer imports)
    from victor.config.settings import Settings

from victor.core.async_utils import run_sync
from victor.ui.slash.protocol import CommandContext, CommandMetadata
from victor.ui.slash.registry import CommandRegistry, get_command_registry

logger = logging.getLogger(__name__)


class SlashCommandHandler:
    """Handles slash command parsing and execution.

    This is the main entry point for slash command processing.
    It delegates actual command logic to registered command classes.
    """

    def __init__(
        self,
        console: Console,
        settings: "Settings",
        agent: Optional[Any] = None,  # ✅ PROPER: Use Any instead of AgentOrchestrator
        registry: Optional[CommandRegistry] = None,
        auto_discover: bool = True,
    ) -> None:
        """Initialize the command handler.

        Args:
            console: Rich console for output.
            settings: Application settings.
            agent: Optional agent instance (any type - duck typing used).
            registry: Optional custom registry (defaults to global).
            auto_discover: Whether to auto-discover commands on init.
        """
        self.console = console
        self.settings = settings
        self.agent = agent
        self._registry = registry or get_command_registry()
        self.exit_callback: Optional[Callable[[], None]] = None

        # Auto-discover commands if registry is empty
        if auto_discover and not any(self._registry.iter_commands()):
            self._discover_and_register()

    def _discover_and_register(self) -> None:
        """Discover and register all commands."""
        # First, discover commands from the commands package
        self._registry.discover_commands()

        # If still empty, register built-in fallback help
        if not any(self._registry.iter_commands()):
            logger.warning("No commands discovered, registering minimal fallback")
            from victor.ui.slash.commands.system import HelpCommand

            self._registry.register_class(HelpCommand)

    def set_agent(self, agent: Any) -> None:
        """Set the agent reference (for commands that need it).

        Args:
            agent: Agent instance (any type - duck typing used).
        """
        self.agent = agent

    def set_exit_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set a custom exit callback for interactive CLI mode.

        When set, commands like /exit will call this instead of sys.exit().

        Args:
            callback: Callable to invoke on exit, or None to clear.
        """
        self.exit_callback = callback

    @property
    def registry(self) -> CommandRegistry:
        """Access the command registry."""
        return self._registry

    def is_command(self, text: str) -> bool:
        """Check if text is a slash command."""
        return text.strip().startswith("/")

    def parse_command(self, text: str) -> Tuple[str, List[str]]:
        """Parse command name and arguments from text.

        Args:
            text: Raw command text (e.g., "/model qwen2.5:7b")

        Returns:
            Tuple of (command_name, [args])
        """
        parts = text.strip().split()
        if not parts:
            return "", []

        cmd_name = parts[0].lstrip("/").lower()
        args = parts[1:] if len(parts) > 1 else []
        return cmd_name, args

    async def execute(self, text: str) -> bool:
        """Execute a slash command.

        Args:
            text: Raw command text.

        Returns:
            True if command was handled, False otherwise.
        """
        cmd_name, args = self.parse_command(text)

        if not cmd_name:
            return False

        command = self._registry.get(cmd_name)
        if not command:
            self.console.print(f"[red]Unknown command:[/] /{cmd_name}")
            self.console.print("Type [bold]/help[/] for available commands")
            return True

        # Check if command requires agent
        if command.metadata.requires_agent and self.agent is None:
            self.console.print(f"[yellow]Command /{cmd_name} requires an active session[/]")
            return True

        # Build execution context
        ctx = CommandContext(
            console=self.console,
            settings=self.settings,
            agent=self.agent,
            args=args,
        )

        try:
            result = command.execute(ctx)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            self.console.print(f"[red]Command error:[/] {e}")
            logger.exception(f"Error executing command /{cmd_name}")

        return True

    def execute_sync(self, text: str) -> bool:
        """Synchronously execute a command (blocks if async).

        For use in contexts where async is not available.
        """
        return run_sync(self.execute(text))

    def get_help(self, command_name: Optional[str] = None) -> str:
        """Get help text for a command or all commands.

        Args:
            command_name: Optional specific command to get help for.

        Returns:
            Formatted help text.
        """
        if command_name:
            command = self._registry.get(command_name)
            if command:
                meta = command.metadata
                aliases = ", ".join(f"/{a}" for a in meta.aliases) if meta.aliases else "none"
                return (
                    f"[bold]/{meta.name}[/]\n\n"
                    f"{meta.description}\n\n"
                    f"[dim]Usage:[/] {meta.usage}\n"
                    f"[dim]Aliases:[/] {aliases}\n"
                    f"[dim]Category:[/] {meta.category}"
                )
            from victor.ui.slash.commands.system import get_builtin_tool_help

            tool_help = get_builtin_tool_help(command_name)
            if tool_help:
                _, content = tool_help
                return content
            return f"Unknown command: /{command_name}"

        # Build table of all commands
        lines = ["[bold]Available Commands[/]\n"]
        for category in self._registry.categories():
            commands = self._registry.list_by_category(category)
            if commands:
                lines.append(f"\n[cyan]{category.title()}[/]")
                for cmd in commands:
                    meta = cmd.metadata
                    aliases = (
                        f" ({', '.join('/' + a for a in meta.aliases)})" if meta.aliases else ""
                    )
                    lines.append(f"  /{meta.name}{aliases} - {meta.description}")

        lines.append("\n[dim]Type /help <command> for more details[/]")
        return "\n".join(lines)

    def print_help(self, command_name: Optional[str] = None) -> None:
        """Print help for a command or all commands."""
        if command_name:
            command = self._registry.get(command_name)
            if command:
                meta = command.metadata
                aliases = ", ".join(f"/{a}" for a in meta.aliases) if meta.aliases else "none"
                self.console.print(
                    Panel(
                        f"[bold]/{meta.name}[/]\n\n"
                        f"{meta.description}\n\n"
                        f"[dim]Usage:[/] {meta.usage}\n"
                        f"[dim]Aliases:[/] {aliases}",
                        title=f"Help: /{meta.name}",
                        border_style="blue",
                    )
                )
            else:
                from victor.ui.slash.commands.system import get_builtin_tool_help

                tool_help = get_builtin_tool_help(command_name)
                if tool_help:
                    title, content = tool_help
                    self.console.print(Panel(content, title=f"Help: {title}", border_style="blue"))
                    return
                self.console.print(f"[yellow]Unknown command:[/] /{command_name}")
            return

        # Build table of all commands
        table = Table(title="Available Commands", show_header=True)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description")
        table.add_column("Aliases", style="dim")

        for name, meta in self._registry.list_commands():
            aliases = ", ".join(f"/{a}" for a in meta.aliases) if meta.aliases else ""
            table.add_row(f"/{name}", meta.description, aliases)

        self.console.print(table)
        self.console.print("\n[dim]Type /help <command> for more details[/]")

    def get_completions(self, partial: str) -> List[str]:
        """Get command completions for partial input.

        Args:
            partial: Partial command text (with or without /).

        Returns:
            List of matching command names.
        """
        if partial.startswith("/"):
            partial = partial[1:]
        partial = partial.lower()

        completions = []
        for name, _ in self._registry.list_commands():
            if name.startswith(partial):
                completions.append(f"/{name}")

        return sorted(completions)
