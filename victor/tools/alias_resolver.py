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

"""Tool alias resolution for shell variants and alternatives."""

from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolAlias:
    """Configuration for a tool alias."""

    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    resolver: Optional[Callable[[str], str]] = None


class ToolAliasResolver:
    """Resolves tool aliases to their enabled variants.

    This class follows the Open/Closed Principle by allowing new tool aliases
    to be registered without modifying existing code. Shell variants (bash, zsh, sh)
    and alternative tools (grep/ripgrep, find/fd) can be registered dynamically.

    Example:
        resolver = ToolAliasResolver.get_instance()
        resolver.register("shell", ["bash", "zsh", "sh"])
        resolver.register("grep", ["ripgrep", "rg"])

        # Resolve to an enabled variant
        actual_tool = resolver.resolve("shell", enabled_tools=["zsh"])  # Returns "zsh"
    """

    _instance: Optional["ToolAliasResolver"] = None

    def __init__(self):
        self._aliases: Dict[str, ToolAlias] = {}
        self._reverse_map: Dict[str, str] = {}

    @classmethod
    def get_instance(cls) -> "ToolAliasResolver":
        """Get the singleton instance of ToolAliasResolver.

        Returns:
            The singleton ToolAliasResolver instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance.

        Useful for testing to ensure a clean state between tests.
        """
        cls._instance = None

    def register(
        self,
        canonical_name: str,
        aliases: List[str],
        resolver: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Register a tool with its aliases.

        Args:
            canonical_name: The canonical/primary name for the tool.
            aliases: List of alternative names that should resolve to this tool.
            resolver: Optional custom resolver function that takes the requested
                     name and returns the actual tool name to use.
        """
        self._aliases[canonical_name] = ToolAlias(canonical_name, aliases, resolver)
        # Map canonical name to itself
        self._reverse_map[canonical_name] = canonical_name
        # Map all aliases to the canonical name
        for alias in aliases:
            self._reverse_map[alias] = canonical_name

    def resolve(self, name: str, enabled_tools: List[str]) -> str:
        """Resolve a tool name to an enabled variant.

        Looks up the canonical name for the requested tool, then finds
        an enabled variant from the canonical name or its aliases.

        Args:
            name: The requested tool name.
            enabled_tools: List of currently enabled tool names.

        Returns:
            The name of an enabled tool variant, or the original name
            if no variant is found or enabled.
        """
        canonical = self._reverse_map.get(name, name)
        config = self._aliases.get(canonical)
        if not config:
            return name

        # Use custom resolver if provided
        if config.resolver:
            return config.resolver(name)

        # Check if canonical name is enabled
        if canonical in enabled_tools:
            return canonical

        # Check aliases in order
        for alias in config.aliases:
            if alias in enabled_tools:
                return alias

        # No enabled variant found, return original name
        return name

    def get_canonical(self, name: str) -> str:
        """Get the canonical name for a tool.

        Args:
            name: A tool name or alias.

        Returns:
            The canonical name for the tool, or the original name
            if it's not a registered alias.
        """
        return self._reverse_map.get(name, name)

    def is_registered(self, name: str) -> bool:
        """Check if a name is registered (as canonical or alias).

        Args:
            name: The tool name to check.

        Returns:
            True if the name is registered, False otherwise.
        """
        return name in self._reverse_map

    def get_aliases(self, canonical_name: str) -> List[str]:
        """Get all aliases for a canonical tool name.

        Args:
            canonical_name: The canonical tool name.

        Returns:
            List of aliases, or empty list if not registered.
        """
        config = self._aliases.get(canonical_name)
        if config:
            return config.aliases.copy()
        return []


def get_alias_resolver() -> ToolAliasResolver:
    """Get the global ToolAliasResolver instance.

    Convenience function for accessing the singleton.

    Returns:
        The singleton ToolAliasResolver instance.
    """
    return ToolAliasResolver.get_instance()
