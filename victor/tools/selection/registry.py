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

"""Tool Selection Strategy Registry.

Central registry for tool selection strategies, enabling pluggable
selection algorithms across verticals.

Example:
    from victor.tools.selection import (
        ToolSelectionStrategyRegistry,
        ToolSelectionContext,
        get_strategy,
        get_best_strategy,
    )

    # Register a custom strategy
    registry = ToolSelectionStrategyRegistry.get_instance()
    registry.register("custom", my_custom_selector)

    # Get a specific strategy
    semantic = get_strategy("semantic")
    tools = await semantic.select_tools(context)

    # Auto-select best strategy for context
    best = get_best_strategy(context)
    tools = await best.select_tools(context)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Type

from victor.tools.selection.protocol import (
    PerformanceProfile,
    ToolSelectionContext,
    ToolSelectionStrategy,
    ToolSelectorFeatures,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ToolSelectionStrategyRegistry:
    """Registry of tool selection strategies.

    Singleton registry that manages available tool selection strategies.
    Supports both instance-based and class-based registration.

    Built-in strategies:
    - keyword: Fast registry-based matching (<1ms)
    - semantic: ML-based embedding similarity (10-50ms)
    - hybrid: Blends both approaches (best of both worlds)

    Example:
        # Get registry
        registry = ToolSelectionStrategyRegistry.get_instance()

        # Register a strategy
        registry.register("my_strategy", MyStrategy())

        # Get a strategy
        strategy = registry.get("semantic")

        # Auto-select best strategy
        best = registry.get_best_strategy(context)
    """

    _instance: Optional["ToolSelectionStrategyRegistry"] = None
    _strategies: Dict[str, ToolSelectionStrategy]
    _strategy_classes: Dict[str, Type[ToolSelectionStrategy]]

    def __init__(self) -> None:
        """Initialize registry."""
        self._strategies = {}
        self._strategy_classes = {}

    @classmethod
    def get_instance(cls) -> "ToolSelectionStrategyRegistry":
        """Get the singleton registry instance.

        Returns:
            The global registry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def register(
        self,
        name: str,
        strategy: ToolSelectionStrategy,
        *,
        replace: bool = False,
    ) -> None:
        """Register a strategy instance.

        Args:
            name: Strategy name (e.g., "keyword", "semantic")
            strategy: Strategy instance
            replace: Whether to replace existing strategy

        Raises:
            ValueError: If name exists and replace=False
        """
        if name in self._strategies and not replace:
            raise ValueError(f"Strategy '{name}' already registered. Use replace=True to override.")

        self._strategies[name] = strategy
        logger.debug(f"Registered strategy: {name}")

    def register_class(
        self,
        name: str,
        strategy_class: Type[ToolSelectionStrategy],
        *,
        replace: bool = False,
    ) -> None:
        """Register a strategy class (lazy instantiation).

        The class will be instantiated on first use.

        Args:
            name: Strategy name
            strategy_class: Strategy class (not instance)
            replace: Whether to replace existing

        Raises:
            ValueError: If name exists and replace=False
        """
        if name in self._strategy_classes and not replace:
            raise ValueError(
                f"Strategy class '{name}' already registered. Use replace=True to override."
            )

        self._strategy_classes[name] = strategy_class
        logger.debug(f"Registered strategy class: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a strategy.

        Args:
            name: Strategy name to remove

        Returns:
            True if strategy was removed, False if not found
        """
        removed = False
        if name in self._strategies:
            del self._strategies[name]
            removed = True
        if name in self._strategy_classes:
            del self._strategy_classes[name]
            removed = True
        return removed

    def get(self, name: str) -> Optional[ToolSelectionStrategy]:
        """Get a strategy by name.

        If a class is registered but not instantiated, instantiates it.

        Args:
            name: Strategy name

        Returns:
            Strategy instance or None if not found
        """
        # Check instance registry first
        if name in self._strategies:
            return self._strategies[name]

        # Check class registry and instantiate
        if name in self._strategy_classes:
            try:
                strategy = self._strategy_classes[name]()
                self._strategies[name] = strategy
                return strategy
            except Exception as e:
                logger.error(f"Failed to instantiate strategy '{name}': {e}")
                return None

        return None

    def get_best_strategy(
        self,
        context: ToolSelectionContext,
        *,
        prefer_fast: bool = False,
        require_embeddings: bool = False,
    ) -> Optional[ToolSelectionStrategy]:
        """Get the best strategy for the given context.

        Selection priority:
        1. If require_embeddings: semantic > hybrid
        2. If prefer_fast: keyword > hybrid > semantic
        3. Default: hybrid > semantic > keyword

        Args:
            context: Selection context
            prefer_fast: Prefer faster strategies over quality
            require_embeddings: Require embedding-based strategy

        Returns:
            Best available strategy or None
        """
        # Define preference orders
        if require_embeddings:
            preferred = ["semantic", "hybrid"]
        elif prefer_fast:
            preferred = ["keyword", "hybrid", "semantic"]
        else:
            preferred = ["hybrid", "semantic", "keyword"]

        # Find first available that supports context
        for name in preferred:
            strategy = self.get(name)
            if strategy and strategy.supports_context(context):
                return strategy

        # Fallback: any strategy that supports context
        for name in self.list_strategies():
            strategy = self.get(name)
            if strategy and strategy.supports_context(context):
                return strategy

        return None

    def list_strategies(self) -> List[str]:
        """List all registered strategy names.

        Returns:
            List of strategy names
        """
        all_names = set(self._strategies.keys()) | set(self._strategy_classes.keys())
        return sorted(all_names)

    def get_strategy_info(self, name: str) -> Optional[Dict]:
        """Get information about a strategy.

        Args:
            name: Strategy name

        Returns:
            Dict with strategy info or None if not found
        """
        strategy = self.get(name)
        if not strategy:
            return None

        profile = strategy.get_performance_profile()
        features = strategy.get_supported_features()

        return {
            "name": strategy.get_strategy_name(),
            "performance": {
                "avg_latency_ms": profile.avg_latency_ms,
                "requires_embeddings": profile.requires_embeddings,
                "requires_model_inference": profile.requires_model_inference,
                "memory_usage_mb": profile.memory_usage_mb,
            },
            "features": {
                "semantic_matching": features.supports_semantic_matching,
                "context_awareness": features.supports_context_awareness,
                "cost_optimization": features.supports_cost_optimization,
                "usage_learning": features.supports_usage_learning,
                "workflow_patterns": features.supports_workflow_patterns,
            },
        }

    async def close_all(self) -> None:
        """Close all registered strategies.

        Should be called during shutdown to cleanup resources.
        """
        for name, strategy in self._strategies.items():
            try:
                await strategy.close()
            except Exception as e:
                logger.error(f"Error closing strategy '{name}': {e}")

        self._strategies.clear()


# =============================================================================
# Convenience Functions
# =============================================================================


def get_strategy_registry() -> ToolSelectionStrategyRegistry:
    """Get the global strategy registry.

    Returns:
        Global ToolSelectionStrategyRegistry instance
    """
    return ToolSelectionStrategyRegistry.get_instance()


def register_strategy(
    name: str,
    strategy: ToolSelectionStrategy,
    *,
    replace: bool = False,
) -> None:
    """Register a strategy in the global registry.

    Args:
        name: Strategy name
        strategy: Strategy instance
        replace: Whether to replace existing
    """
    get_strategy_registry().register(name, strategy, replace=replace)


def get_strategy(name: str) -> Optional[ToolSelectionStrategy]:
    """Get a strategy from the global registry.

    Args:
        name: Strategy name

    Returns:
        Strategy instance or None
    """
    return get_strategy_registry().get(name)


def get_best_strategy(
    context: ToolSelectionContext,
    *,
    prefer_fast: bool = False,
    require_embeddings: bool = False,
) -> Optional[ToolSelectionStrategy]:
    """Get the best strategy for context from global registry.

    Args:
        context: Selection context
        prefer_fast: Prefer faster strategies
        require_embeddings: Require embedding-based

    Returns:
        Best available strategy or None
    """
    return get_strategy_registry().get_best_strategy(
        context,
        prefer_fast=prefer_fast,
        require_embeddings=require_embeddings,
    )


def list_strategies() -> List[str]:
    """List all strategies in global registry.

    Returns:
        List of strategy names
    """
    return get_strategy_registry().list_strategies()


__all__ = [
    # Registry
    "ToolSelectionStrategyRegistry",
    # Convenience functions
    "get_strategy_registry",
    "register_strategy",
    "get_strategy",
    "get_best_strategy",
    "list_strategies",
]
