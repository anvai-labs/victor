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

"""Protocol definitions for slash commands.

This module defines the SlashCommandProtocol interface that all
slash commands must implement. Following SOLID principles:

- Single Responsibility: Each command class handles one command
- Open-Closed: New commands can be added without modifying existing code
- Liskov Substitution: All commands implement the same protocol
- Interface Segregation: Minimal protocol interface
- Dependency Inversion: Depend on abstractions (protocol)
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rich.console import Console

    # ✅ PROPER: Only import Settings for type hints (no agent layer imports)
    from victor.config.settings import Settings


@dataclass
class CommandMetadata:
    """Metadata for a slash command."""

    name: str
    description: str
    usage: str
    aliases: List[str] = field(default_factory=list)
    category: str = "general"
    requires_agent: bool = False  # True if command needs an active agent
    is_async: bool = False  # True if execute() returns a coroutine


@dataclass
class CommandContext:
    """Context passed to command execution."""

    console: Console
    settings: Settings
    agent: Optional[Any] = None  # ✅ PROPER: Use Any instead of AgentOrchestrator
    args: List[str] = field(default_factory=list)


@runtime_checkable
class SlashCommandProtocol(Protocol):
    """Protocol interface for all slash commands.

    Each slash command must implement this protocol to be
    discoverable and executable by the command handler.

    Example usage:
        class MyCommand:
            @property
            def metadata(self) -> CommandMetadata:
                return CommandMetadata(
                    name="mycommand",
                    description="Does something useful",
                    usage="/mycommand [options]",
                    aliases=["mc"],
                    category="tools",
                )

            def execute(self, ctx: CommandContext) -> None:
                ctx.console.print("Hello from mycommand!")
    """

    @property
    @abstractmethod
    def metadata(self) -> CommandMetadata:
        """Return command metadata."""
        ...

    @abstractmethod
    def execute(self, ctx: CommandContext) -> Any:
        """Execute the command.

        Args:
            ctx: Command execution context containing console,
                 settings, agent, and arguments.

        Returns:
            None for sync commands, or a coroutine for async commands.
        """
        ...


class BaseSlashCommand:
    """Base class for slash commands with common functionality.

    Provides helper methods for common patterns like checking
    for agent availability, parsing flags, etc.
    """

    def _require_agent(self, ctx: CommandContext) -> bool:
        """Check if agent is available, print warning if not.

        Returns:
            True if agent is available, False otherwise.
        """
        if ctx.agent is None:
            ctx.console.print("[yellow]No active session[/]")
            return False
        return True

    def _has_flag(self, ctx: CommandContext, *flags: str) -> bool:
        """Check if any of the specified flags are present in args."""
        return any(flag in ctx.args for flag in flags)

    def _get_arg(
        self, ctx: CommandContext, index: int, default: Optional[str] = None
    ) -> Optional[str]:
        """Get argument at index, or default if not present."""
        if index < len(ctx.args):
            return ctx.args[index]
        return default

    def _parse_int_arg(self, ctx: CommandContext, index: int, default: int = 0) -> int:
        """Parse integer argument at index, or return default."""
        val = self._get_arg(ctx, index)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default

    def _get_flag_value(
        self, ctx: CommandContext, flag: str, default: Optional[str] = None
    ) -> Optional[str]:
        """Get value following a flag (e.g., --keep 5 returns '5' for flag='--keep')."""
        for i, arg in enumerate(ctx.args):
            if arg == flag and i + 1 < len(ctx.args):
                return ctx.args[i + 1]
        return default


# Type alias for command factories
CommandFactory = type[SlashCommandProtocol]
