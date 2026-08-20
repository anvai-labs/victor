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

"""Registry for tools with progressive parameters.

This module provides a registry for tools that support progressive parameter
escalation - starting with conservative values and increasing limits on retry.

Refactored in Phase 4 to inherit from SingletonRegistry base class,
eliminating duplicate singleton boilerplate code.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from victor.core.registry import SingletonRegistry


@dataclass
class ProgressiveToolConfig:
    """Configuration for progressive tool parameters.

    Attributes:
        tool_name: Name of the tool
        progressive_params: Parameters that can be escalated
        initial_values: Starting values for progressive params
        max_values: Maximum values for progressive params
    """

    tool_name: str
    progressive_params: Dict[str, Any] = field(default_factory=dict)
    initial_values: Dict[str, Any] = field(default_factory=dict)
    max_values: Dict[str, Any] = field(default_factory=dict)


class ProgressiveToolsRegistry(SingletonRegistry["ProgressiveToolsRegistry"]):
    """Registry for tools that support progressive parameter escalation.

    Removes hardcoded PROGRESSIVE_TOOLS from orchestrator by providing
    a centralized, configurable registry.

    Inherits thread-safe singleton pattern from SingletonRegistry.

    Example:
        registry = ProgressiveToolsRegistry.get_instance()
        registry.register(
            "read",
            progressive_params={"limit": [100, 500, 1000]},
            initial_values={"limit": 100},
            max_values={"limit": 1000},
        )

        if registry.is_progressive("read"):
            config = registry.get_config("read")
    """

    def __init__(self) -> None:
        """Initialize the progressive tools registry."""
        super().__init__()
        self._tools: Dict[str, ProgressiveToolConfig] = {}

    def register(
        self,
        tool_name: str,
        progressive_params: Dict[str, Any],
        initial_values: Optional[Dict[str, Any]] = None,
        max_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a tool with progressive parameter configuration.

        Args:
            tool_name: Name of the tool to register
            progressive_params: Parameters that can be escalated
            initial_values: Starting values for progressive params
            max_values: Maximum values for progressive params
        """
        self._tools[tool_name] = ProgressiveToolConfig(
            tool_name=tool_name,
            progressive_params=progressive_params,
            initial_values=initial_values or {},
            max_values=max_values or {},
        )

    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool.

        Args:
            tool_name: Name of the tool to unregister

        Returns:
            True if tool was found and removed
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False

    def is_progressive(self, tool_name: str) -> bool:
        """Check if a tool has progressive parameters.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is registered as progressive
        """
        return tool_name in self._tools

    def get_config(self, tool_name: str) -> Optional[ProgressiveToolConfig]:
        """Get progressive configuration for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            ProgressiveToolConfig or None if not found
        """
        return self._tools.get(tool_name)

    def list_progressive_tools(self) -> Set[str]:
        """Get set of all registered progressive tool names.

        Returns:
            Set of tool names
        """
        return set(self._tools.keys())

    def clear(self) -> int:
        """Clear all registered tools.

        Returns:
            Number of tools that were cleared
        """
        count = len(self._tools)
        self._tools.clear()
        return count


def get_progressive_registry() -> ProgressiveToolsRegistry:
    """Get the global progressive tools registry instance.

    Convenience function for accessing the singleton.

    Returns:
        ProgressiveToolsRegistry singleton instance
    """
    return ProgressiveToolsRegistry.get_instance()
