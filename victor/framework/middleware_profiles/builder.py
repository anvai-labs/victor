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

"""Middleware profile builder for custom middleware configurations.

This module provides a builder pattern for creating custom middleware profiles.

Design Pattern: Builder
- Fluent interface for building custom profiles
- Extensible for vertical-specific middleware
- Type-safe middleware composition

Example:
    from victor.framework.middleware_profiles import MiddlewareProfileBuilder

    profile = (
        MiddlewareProfileBuilder()
        .add_middleware(GitSafetyMiddleware(block_dangerous=True))
        .add_middleware(SecretMaskingMiddleware())
        .set_priority(25)
        .build()
    )
"""

from __future__ import annotations

from typing import Any, List, Optional

from victor.framework.middleware_profiles.profiles import MiddlewareProfile

# =============================================================================
# Middleware Profile Builder
# =============================================================================


class MiddlewareProfileBuilder:
    """Builder for creating custom middleware profiles.

    Provides a fluent interface for composing custom middleware profiles
    from individual middleware components.

    Example:
        builder = MiddlewareProfileBuilder()
        profile = (
            builder
            .set_name("custom_profile")
            .set_description("My custom middleware profile")
            .add_middleware(GitSafetyMiddleware())
            .add_middleware(SecretMaskingMiddleware())
            .set_priority(50)
            .build()
        )
    """

    def __init__(self) -> None:
        """Initialize the builder with empty state."""
        self._name: str = "custom"
        self._description: str = "Custom middleware profile"
        self._middlewares: List[Any] = []
        self._priority: int = 50

    def set_name(self, name: str) -> "MiddlewareProfileBuilder":
        """Set the profile name.

        Args:
            name: Profile name

        Returns:
            Self for method chaining
        """
        self._name = name
        return self

    def set_description(self, description: str) -> "MiddlewareProfileBuilder":
        """Set the profile description.

        Args:
            description: Profile description

        Returns:
            Self for method chaining
        """
        self._description = description
        return self

    def add_middleware(self, middleware: Any) -> "MiddlewareProfileBuilder":
        """Add a middleware to the profile.

        Args:
            middleware: Middleware instance

        Returns:
            Self for method chaining
        """
        self._middlewares.append(middleware)
        return self

    def add_middlewares(self, middlewares: List[Any]) -> "MiddlewareProfileBuilder":
        """Add multiple middlewares to the profile.

        Args:
            middlewares: List of middleware instances

        Returns:
            Self for method chaining
        """
        self._middlewares.extend(middlewares)
        return self

    def remove_middleware(self, middleware_type: type) -> "MiddlewareProfileBuilder":
        """Remove middleware of a specific type.

        Args:
            middleware_type: Type of middleware to remove

        Returns:
            Self for method chaining
        """
        self._middlewares = [m for m in self._middlewares if not isinstance(m, middleware_type)]
        return self

    def set_priority(self, priority: int) -> "MiddlewareProfileBuilder":
        """Set the execution priority.

        Args:
            priority: Priority value (lower = higher priority)

        Returns:
            Self for method chaining
        """
        self._priority = priority
        return self

    def clear_middlewares(self) -> "MiddlewareProfileBuilder":
        """Clear all middlewares from the profile.

        Returns:
            Self for method chaining
        """
        self._middlewares = []
        return self

    def build(self) -> MiddlewareProfile:
        """Build the middleware profile.

        Returns:
            MiddlewareProfile with configured middleware
        """
        return MiddlewareProfile(
            name=self._name,
            description=self._description,
            middlewares=self._middlewares.copy(),
            priority=self._priority,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_profile(
    name: str = "custom",
    description: str = "Custom middleware profile",
    middlewares: Optional[List[Any]] = None,
    priority: int = 50,
) -> MiddlewareProfile:
    """Create a middleware profile directly.

    Args:
        name: Profile name
        description: Profile description
        middlewares: Optional list of middleware instances
        priority: Execution priority

    Returns:
        MiddlewareProfile
    """
    return MiddlewareProfile(
        name=name,
        description=description,
        middlewares=middlewares or [],
        priority=priority,
    )


__all__ = [
    "MiddlewareProfileBuilder",
    "create_profile",
]
