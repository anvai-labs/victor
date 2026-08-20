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

"""Handler Registry for workflow compute handlers.

This module provides centralized management for compute handlers across verticals,
eliminating duplication and providing consistent handler discovery and registration.

Design Pattern: Registry + Namespace Isolation
====================================================
- Handlers stored by vertical name for namespace isolation
- Global handlers for cross-vertical functionality
- Lazy loading of vertical handlers
- Integration with workflow executor for handler resolution

Phase 1 Gap #2: Handler Registry
===================================
This addresses the handler duplication gap identified in the architecture analysis.
Verticals no longer need to manage HANDLERS dicts directly.

Usage:
    from victor.framework.handler_registry import get_handler_registry

    # Register handlers from a vertical
    registry = get_handler_registry()
    registry.register_vertical("coding", {
        "code_validation": CodeValidationHandler(),
        "test_runner": TestRunnerHandler(),
    })

    # Get a handler
    handler = registry.get_handler("coding", "code_validation")

    # List all handlers
    all_handlers = registry.list_handlers()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Handler Metadata
# =============================================================================


@dataclass
class HandlerSpec:
    """Metadata for a registered handler.

    Attributes:
        name: Handler name (unique within vertical)
        description: What this handler does
        category: Handler category (validation, execution, analysis, etc.)
        handler_class: The handler class (for inspection)
        version: Handler version string
    """

    name: str
    description: str
    category: str
    handler_class: type
    version: str = "1.0"

    def __post_init__(self) -> None:
        """Validate handler spec."""
        if not self.name:
            raise ValueError("Handler name cannot be empty")
        if not self.category:
            raise ValueError("Handler category cannot be empty")


# =============================================================================
# Handler Registry
# =============================================================================


class HandlerRegistry:
    """Centralized registry for compute handlers from verticals.

    Provides namespace isolation between verticals while supporting
    global handlers for cross-cutting concerns.

    Attributes:
        _vertical_handlers: Handlers stored by vertical name
        _global_handlers: Global handlers accessible to all verticals
        _specs: Handler metadata by vertical:handler
    """

    _instance: Optional["HandlerRegistry"] = None

    def __init__(self) -> None:
        """Initialize the handler registry."""
        self._vertical_handlers: Dict[str, Dict[str, Any]] = {}
        self._global_handlers: Dict[str, Any] = {}
        self._specs: Dict[str, HandlerSpec] = {}
        logger.debug("HandlerRegistry initialized")

    @classmethod
    def get_instance(cls) -> "HandlerRegistry":
        """Get singleton instance of the registry.

        Returns:
            HandlerRegistry singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for testing).

        This clears all registered handlers and creates a fresh instance
        on the next call to get_instance().
        """
        cls._instance = None
        logger.debug("HandlerRegistry instance reset")

    def register_vertical(
        self,
        vertical_name: str,
        handlers: Dict[str, Any],
        category: str = "general",
        description: str = "",
    ) -> None:
        """Register all handlers from a vertical.

        Args:
            vertical_name: Name of the vertical (e.g., "coding", "research")
            handlers: Dict mapping handler names to handler instances
            category: Default category for these handlers
            description: Optional description for the vertical's handlers
        """
        if not vertical_name:
            raise ValueError("vertical_name cannot be empty")

        for handler_name, handler_instance in handlers.items():
            # Create spec
            spec = HandlerSpec(
                name=handler_name,
                description=description or f"{vertical_name} handler",
                category=category,
                handler_class=type(handler_instance),
            )

            # Store handler and spec
            key = f"{vertical_name}.{handler_name}"
            self._specs[key] = spec

            # Store in vertical namespace
            if vertical_name not in self._vertical_handlers:
                self._vertical_handlers[vertical_name] = {}
            self._vertical_handlers[vertical_name][handler_name] = handler_instance

        logger.debug(f"Registered {len(handlers)} handlers from vertical '{vertical_name}'")

    def register_global(self, name: str, handler: Any, category: str = "global") -> None:
        """Register a global handler accessible to all verticals.

        Args:
            name: Global handler name
            handler: Handler instance
            category: Handler category
        """
        if not name:
            raise ValueError("Global handler name cannot be empty")

        spec = HandlerSpec(
            name=name,
            description=f"Global handler: {name}",
            category=category,
            handler_class=type(handler),
        )

        self._global_handlers[name] = handler
        self._specs[f"global.{name}"] = spec
        logger.debug(f"Registered global handler: {name}")

    def get_handler(self, vertical_name: str, handler_name: str) -> Optional[Any]:
        """Get a handler by vertical and name.

        Args:
            vertical_name: Name of the vertical
            handler_name: Name of the handler within the vertical

        Returns:
            Handler instance or None if not found
        """
        # Check vertical-specific handlers first
        if vertical_name in self._vertical_handlers:
            handler = self._vertical_handlers[vertical_name].get(handler_name)
            if handler:
                return handler

        # Fall back to global handlers
        return self._global_handlers.get(handler_name)

    def get_vertical_handlers(self, vertical_name: str) -> Dict[str, Any]:
        """Get all handlers for a specific vertical.

        Args:
            vertical_name: Name of the vertical

        Returns:
            Dict mapping handler names to handler instances
        """
        return self._vertical_handlers.get(vertical_name, {}).copy()

    def list_handlers(self, vertical_name: Optional[str] = None) -> Dict[str, List[str]]:
        """List available handlers.

        Args:
            vertical_name: Optional vertical name to filter by

        Returns:
            Dict mapping vertical names to lists of handler names
        """
        if vertical_name:
            return {vertical_name: list(self._vertical_handlers.get(vertical_name, {}).keys())}

        return {
            vertical: list(handlers.keys())
            for vertical, handlers in self._vertical_handlers.items()
        }

    def list_specs(self, category: Optional[str] = None) -> List[HandlerSpec]:
        """List handler specifications.

        Args:
            category: Optional category filter

        Returns:
            List of HandlerSpec objects
        """
        specs = list(self._specs.values())
        if category:
            specs = [s for s in specs if s.category == category]
        return specs

    def get_spec(self, vertical_name: str, handler_name: str) -> Optional[HandlerSpec]:
        """Get handler specification.

        Args:
            vertical_name: Name of the vertical
            handler_name: Name of the handler

        Returns:
            HandlerSpec or None if not found
        """
        key = f"{vertical_name}.{handler_name}"
        return self._specs.get(key)

    def clear_vertical(self, vertical_name: str) -> None:
        """Clear all handlers for a vertical.

        Args:
            vertical_name: Name of the vertical to clear
        """
        if vertical_name in self._vertical_handlers:
            # Remove specs
            for handler_name in self._vertical_handlers[vertical_name]:
                key = f"{vertical_name}.{handler_name}"
                self._specs.pop(key, None)

            # Remove handlers
            del self._vertical_handlers[vertical_name]
            logger.debug(f"Cleared handlers for vertical: {vertical_name}")

    def clear_global(self, name: str) -> None:
        """Clear a global handler.

        Args:
            name: Name of the global handler to clear
        """
        if name in self._global_handlers:
            self._specs.pop(f"global.{name}", None)
            del self._global_handlers[name]
            logger.debug(f"Cleared global handler: {name}")


# =============================================================================
# Convenience Functions
# =============================================================================


def get_handler_registry() -> HandlerRegistry:
    """Get the singleton HandlerRegistry instance.

    Returns:
        HandlerRegistry singleton
    """
    return HandlerRegistry.get_instance()


def register_vertical_handlers(
    vertical_name: str,
    handlers: Dict[str, Any],
    category: str = "general",
    description: str = "",
) -> None:
    """Register handlers from a vertical with the registry.

    Convenience function for registering vertical handlers.

    Args:
        vertical_name: Name of the vertical
        handlers: Dict of handler name to handler instance
        category: Default category for handlers
        description: Optional description
    """
    registry = get_handler_registry()
    registry.register_vertical(vertical_name, handlers, category, description)


def register_global_handler(
    name: str,
    handler: Any,
    category: str = "global",
) -> None:
    """Register a global handler.

    Convenience function for registering global handlers.

    Args:
        name: Global handler name
        handler: Handler instance
        category: Handler category
    """
    registry = get_handler_registry()
    registry.register_global(name, handler, category)


__all__ = [
    "HandlerSpec",
    "HandlerRegistry",
    "get_handler_registry",
    "register_vertical_handlers",
    "register_global_handler",
]
