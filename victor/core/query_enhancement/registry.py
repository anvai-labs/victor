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

"""Query Enhancement Strategy Registry.

Provides registration and lookup of query enhancement strategies.
Supports singleton pattern for global access while allowing
per-pipeline customization.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Type

from victor.protocols.query_enhancement import (
    EnhancementTechnique,
    IQueryEnhancementStrategy,
)

logger = logging.getLogger(__name__)

# Global registry instance
_default_registry: Optional["QueryEnhancementRegistry"] = None


class QueryEnhancementRegistry:
    """Registry for query enhancement strategies.

    Maps EnhancementTechnique enum values to strategy classes.
    Supports lazy initialization of strategy instances.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._strategies: Dict[EnhancementTechnique, Type[IQueryEnhancementStrategy]] = {}
        self._instances: Dict[EnhancementTechnique, IQueryEnhancementStrategy] = {}

    def register(
        self,
        technique: EnhancementTechnique,
        strategy_class: Type[IQueryEnhancementStrategy],
    ) -> None:
        """Register a strategy class for a technique.

        Args:
            technique: Enhancement technique enum
            strategy_class: Strategy class to register
        """
        self._strategies[technique] = strategy_class
        # Clear any cached instance
        self._instances.pop(technique, None)
        logger.debug(f"Registered {strategy_class.__name__} for {technique.value}")

    def get(
        self,
        technique: EnhancementTechnique,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[IQueryEnhancementStrategy]:
        """Get a strategy instance for a technique.

        Uses cached instance if available with same config,
        otherwise creates a new instance.

        Args:
            technique: Enhancement technique enum
            provider: Optional LLM provider name
            model: Optional model name

        Returns:
            Strategy instance or None if not registered
        """
        if technique not in self._strategies:
            logger.debug("No strategy registered for %s", technique.value)
            return None

        # For simplicity, always create new instance if provider/model specified
        # Otherwise use cached instance
        if provider or model:
            strategy_class = self._strategies[technique]
            return strategy_class(provider=provider, model=model)

        # Use cached instance
        if technique not in self._instances:
            strategy_class = self._strategies[technique]
            self._instances[technique] = strategy_class()

        return self._instances[technique]

    def list_techniques(self) -> list[EnhancementTechnique]:
        """List all registered techniques.

        Returns:
            List of registered technique enums
        """
        return list(self._strategies.keys())

    def clear(self) -> None:
        """Clear all registrations and cached instances."""
        self._strategies.clear()
        self._instances.clear()


def get_default_registry() -> QueryEnhancementRegistry:
    """Get the default global registry.

    Creates and populates with default strategies on first call.

    Returns:
        Global QueryEnhancementRegistry instance
    """
    global _default_registry

    if _default_registry is None:
        _default_registry = QueryEnhancementRegistry()
        _register_default_strategies(_default_registry)

    return _default_registry


def _register_default_strategies(registry: QueryEnhancementRegistry) -> None:
    """Register default strategies in registry.

    Args:
        registry: Registry to populate
    """
    from victor.core.query_enhancement.strategies.rewrite import RewriteStrategy
    from victor.core.query_enhancement.strategies.decomposition import (
        DecompositionStrategy,
    )
    from victor.core.query_enhancement.strategies.entity_expand import (
        EntityExpandStrategy,
    )

    registry.register(EnhancementTechnique.REWRITE, RewriteStrategy)
    registry.register(EnhancementTechnique.DECOMPOSITION, DecompositionStrategy)
    registry.register(EnhancementTechnique.ENTITY_EXPAND, EntityExpandStrategy)

    logger.debug("Registered default query enhancement strategies")
