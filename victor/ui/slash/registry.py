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

"""Command registry for slash commands.

Provides dynamic registration and discovery of slash commands.
Supports both explicit registration and decorator-based auto-registration.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Type

from victor.ui.slash.protocol import CommandMetadata, SlashCommandProtocol

logger = logging.getLogger(__name__)


class CommandRegistry:
    """Registry for slash commands.

    Provides:
    - Explicit command registration
    - Decorator-based auto-registration
    - Dynamic module discovery
    - Command lookup by name or alias
    - Category-based filtering
    """

    def __init__(self) -> None:
        self._commands: Dict[str, SlashCommandProtocol] = {}
        self._aliases: Dict[str, str] = {}  # alias -> primary name
        self._categories: Dict[str, List[str]] = {}  # category -> command names
        self._auto_discover_attempted = False

    def _ensure_discovered(self) -> None:
        """Lazily discover commands on first read (co-design review item 21).

        Discovery previously ran unconditionally at package-import time
        (``victor.ui.slash.__init__`` eagerly importing ``commands``),
        pulling in the framework/RL/agent import chain (~0.9s measured)
        even for ``victor --help``. Read paths now trigger discovery
        exactly once, memoized, so callers that bypass
        SlashCommandHandler's own auto_discover logic (e.g. the TUI
        command palette) still see a populated registry.
        """
        if self._auto_discover_attempted or self._commands:
            return
        self._auto_discover_attempted = True
        self.discover_commands()

    def register(self, command: SlashCommandProtocol) -> None:
        """Register a command instance.

        Args:
            command: Command implementing SlashCommandProtocol.
        """
        meta = command.metadata
        name = meta.name.lower()

        if name in self._commands:
            logger.debug("Command '%s' already registered, overwriting", name)

        self._commands[name] = command

        # Register aliases
        for alias in meta.aliases:
            alias_lower = alias.lower()
            self._aliases[alias_lower] = name

        # Track by category
        category = meta.category.lower()
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

        logger.debug(f"Registered command: /{name} (category: {category})")

    def register_class(self, cls: Type[SlashCommandProtocol]) -> None:
        """Register a command class (creates instance).

        Args:
            cls: Command class implementing SlashCommandProtocol.
        """
        try:
            instance = cls()
            self.register(instance)
        except Exception as e:
            logger.error(f"Failed to instantiate command class {cls.__name__}: {e}")

    def unregister(self, name: str) -> bool:
        """Unregister a command by name.

        Args:
            name: Command name to unregister.

        Returns:
            True if command was unregistered, False if not found.
        """
        name = name.lower()
        if name not in self._commands:
            return False

        command = self._commands.pop(name)

        # Remove aliases
        for alias in command.metadata.aliases:
            self._aliases.pop(alias.lower(), None)

        # Remove from category
        category = command.metadata.category.lower()
        if category in self._categories:
            self._categories[category] = [n for n in self._categories[category] if n != name]

        return True

    def get(self, name: str) -> Optional[SlashCommandProtocol]:
        """Get command by name or alias.

        Args:
            name: Command name or alias.

        Returns:
            Command instance if found, None otherwise.
        """
        self._ensure_discovered()
        name_lower = name.lower()

        # Direct lookup
        if name_lower in self._commands:
            return self._commands[name_lower]

        # Alias lookup
        if name_lower in self._aliases:
            primary = self._aliases[name_lower]
            return self._commands.get(primary)

        return None

    def has(self, name: str) -> bool:
        """Check if command exists."""
        self._ensure_discovered()
        name_lower = name.lower()
        return name_lower in self._commands or name_lower in self._aliases

    def list_commands(self) -> List[Tuple[str, CommandMetadata]]:
        """List all registered commands with metadata.

        Returns:
            List of (name, metadata) tuples, sorted by name.
        """
        self._ensure_discovered()
        return sorted(
            [(name, cmd.metadata) for name, cmd in self._commands.items()],
            key=lambda x: x[0],
        )

    def list_by_category(self, category: str) -> List[SlashCommandProtocol]:
        """Get all commands in a category.

        Args:
            category: Category name.

        Returns:
            List of command instances in the category.
        """
        self._ensure_discovered()
        category = category.lower()
        names = self._categories.get(category, [])
        return [self._commands[name] for name in names if name in self._commands]

    def categories(self) -> List[str]:
        """Get all registered categories."""
        self._ensure_discovered()
        return sorted(self._categories.keys())

    def iter_commands(self) -> Iterator[SlashCommandProtocol]:
        """Iterate over all registered commands."""
        self._ensure_discovered()
        return iter(self._commands.values())

    def discover_commands(self, package_path: str = "victor.ui.slash.commands") -> int:
        """Auto-discover and register commands from a package.

        Scans the specified package for modules containing command classes.
        Commands are identified by having a `metadata` property returning
        CommandMetadata.

        Args:
            package_path: Dotted path to the commands package.

        Returns:
            Number of commands registered.
        """
        count = 0
        try:
            package = importlib.import_module(package_path)
            package_dir = Path(package.__file__).parent

            for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
                if module_name.startswith("_"):
                    continue

                try:
                    module = importlib.import_module(f"{package_path}.{module_name}")

                    # Look for command classes in the module
                    for attr_name in dir(module):
                        if attr_name.startswith("_"):
                            continue

                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and attr is not SlashCommandProtocol
                            and hasattr(attr, "metadata")
                            and hasattr(attr, "execute")
                        ):
                            try:
                                self.register_class(attr)
                                count += 1
                            except Exception as e:
                                logger.warning(f"Failed to register {attr_name}: {e}")

                except Exception as e:
                    logger.warning(f"Failed to import command module {module_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to discover commands from {package_path}: {e}")

        logger.info(f"Discovered {count} slash commands from {package_path}")
        return count


# Global registry instance
_global_registry: Optional[CommandRegistry] = None


def get_command_registry() -> CommandRegistry:
    """Get the global command registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CommandRegistry()
    return _global_registry


def register_command(cls: Type[SlashCommandProtocol]) -> Type[SlashCommandProtocol]:
    """Decorator to auto-register a command class.

    Example:
        @register_command
        class HelpCommand:
            @property
            def metadata(self) -> CommandMetadata:
                return CommandMetadata(name="help", ...)

            def execute(self, ctx: CommandContext) -> None:
                ...
    """
    get_command_registry().register_class(cls)
    return cls


def command(
    name: str,
    description: str,
    usage: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    category: str = "general",
    requires_agent: bool = False,
    is_async: bool = False,
) -> Callable[[type], type]:
    """Decorator factory for creating and registering command classes.

    This decorator converts a simple class with just an `execute` method
    into a full command by adding the metadata property.

    Example:
        @command(
            name="hello",
            description="Say hello",
            aliases=["hi"],
            category="greeting",
        )
        class HelloCommand:
            def execute(self, ctx: CommandContext) -> None:
                ctx.console.print("Hello!")
    """

    def decorator(cls: type) -> type:
        # Create metadata
        meta = CommandMetadata(
            name=name,
            description=description,
            usage=usage or f"/{name}",
            aliases=aliases or [],
            category=category,
            requires_agent=requires_agent,
            is_async=is_async,
        )

        # Add metadata property if not present
        if not hasattr(cls, "metadata") or not isinstance(getattr(cls, "metadata", None), property):
            cls.metadata = property(lambda self: meta)

        # Register the command
        get_command_registry().register_class(cls)
        return cls

    return decorator
