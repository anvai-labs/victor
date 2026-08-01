# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tool registration strategy registry.

Manages the selection and registration of tool registration strategies.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.tools.registration.strategies import ToolRegistrationStrategy

logger = logging.getLogger(__name__)


class ToolRegistrationStrategyRegistry:
    """Registry for tool registration strategies.

    Manages available strategies and selects the appropriate
    strategy for a given tool type.

    Strategies are checked in priority order (highest first).
    The first strategy that can_handle() the tool is used.

    Example:
        registry = ToolRegistrationStrategyRegistry()

        # Register custom strategy
        registry.register_strategy(MyCustomStrategy())

        # Get strategy for a tool
        strategy = registry.get_strategy_for(tool)
        if strategy:
            strategy.register(tool_registry, tool)
    """

    _instance: Optional["ToolRegistrationStrategyRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the registry with default strategies."""
        self._strategies: List["ToolRegistrationStrategy"] = []
        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Register default strategies in priority order."""
        from victor.tools.registration.strategies import (
            FunctionDecoratorStrategy,
            BaseToolSubclassStrategy,
            DuckTypedToolStrategy,
            MCPDictStrategy,
        )

        self.register_strategy(FunctionDecoratorStrategy())
        self.register_strategy(BaseToolSubclassStrategy())
        self.register_strategy(DuckTypedToolStrategy())
        self.register_strategy(MCPDictStrategy())

        logger.debug("Registered default tool registration strategies")

    @classmethod
    def get_instance(cls) -> "ToolRegistrationStrategyRegistry":
        """Get singleton instance of the registry.

        Returns:
            Global strategy registry instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_strategy(
        self,
        strategy: "ToolRegistrationStrategy",
    ) -> None:
        """Register a new strategy.

        Strategies are automatically sorted by priority after registration.

        Args:
            strategy: Strategy to register

        Example:
            registry = ToolRegistrationStrategyRegistry()
            registry.register_strategy(MyCustomStrategy())
        """
        self._strategies.append(strategy)
        # Sort by priority (highest first)
        self._strategies.sort(key=lambda s: s.priority, reverse=True)
        logger.debug(f"Registered strategy with priority {strategy.priority}")

    def get_strategy_for(
        self,
        tool: Any,
    ) -> Optional["ToolRegistrationStrategy"]:
        """Get the first strategy that can handle the tool.

        Checks strategies in priority order and returns the first
        one where can_handle() returns True.

        Args:
            tool: Tool object to find strategy for

        Returns:
            Matching strategy, or None if no strategy found

        Example:
            strategy = registry.get_strategy_for(my_tool)
            if strategy:
                strategy.register(tool_registry, my_tool)
            else:
                raise TypeError(f"No strategy for tool type: {type(my_tool)}")
        """
        for strategy in self._strategies:
            if strategy.can_handle(tool):
                return strategy

        return None

    def list_strategies(self) -> List[str]:
        """List all registered strategies.

        Returns:
            List of strategy class names
        """
        return [type(s).__name__ for s in self._strategies]

    def clear(self) -> None:
        """Clear all strategies (useful for testing).

        Note: Default strategies will NOT be re-registered automatically.
        """
        self._strategies.clear()
        logger.debug("Cleared all strategies")


def get_tool_registration_strategy_registry() -> ToolRegistrationStrategyRegistry:
    """Get the global tool registration strategy registry.

    Returns:
        Global strategy registry instance

    Example:
        registry = get_tool_registration_strategy_registry()
        registry.register_strategy(MyCustomStrategy())
    """
    return ToolRegistrationStrategyRegistry.get_instance()
