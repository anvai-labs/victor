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

"""Focused Tool Extensions Protocol (ISP: Interface Segregation Principle).

This module contains a focused protocol specifically for tool-related extensions.
Following ISP, this protocol extracts only tool-related fields from the larger
VerticalExtensions interface.

Key fields extracted:
- middleware: List of middleware implementations for tool call interception
- tool_dependency_provider: Tool dependency information for intelligent selection

Usage:
    from victor.core.verticals.protocols.focused.tool_extensions import (
        ToolExtensionsProtocol,
    )

    class MyVerticalToolExtensions(ToolExtensionsProtocol):
        # Implement only tool-related functionality
        ...
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from victor.core.tool_types import ToolDependencyProviderProtocol

# =============================================================================
# Focused Tool Extensions Protocol
# =============================================================================


@runtime_checkable
class ToolExtensionsProtocol(Protocol):
    """Focused protocol for tool-related vertical extensions.

    This protocol extracts only the tool-related fields from the larger
    VerticalExtensions interface, following the Interface Segregation Principle (ISP).

    Verticals that only need tool-related capabilities can implement this
    focused protocol instead of the full VerticalExtensions interface.

    Example:
        class CodingToolExtensions(ToolExtensionsProtocol):
            @property
            def middleware(self) -> List[MiddlewareProtocol]:
                return [self._validation_middleware, self._logging_middleware]

            @property
            def tool_dependency_provider(self) -> Optional[ToolDependencyProviderProtocol]:
                return self._dependency_provider
    """

    @property
    def middleware(self) -> List[MiddlewareProtocol]:
        """List of middleware implementations for tool call interception.

        Middleware can intercept and modify tool calls before and after execution.
        Use for validation, transformation, logging, or domain-specific processing.

        Returns:
            List of middleware implementations
        """
        ...

    @property
    def tool_dependency_provider(self) -> Optional[ToolDependencyProviderProtocol]:
        """Tool dependency provider for intelligent tool selection.

        Enables verticals to define tool execution patterns and transition
        probabilities for intelligent tool selection.

        Returns:
            Tool dependency provider, or None if not supported
        """
        ...


# =============================================================================
# Middleware Protocol (re-exported from the canonical module)
# =============================================================================

# The class body below was a verbatim re-declaration of
# victor.core.verticals.protocols.middleware.MiddlewareProtocol (jscpd
# exact-duplicate: identical modulo @abstractmethod and blank lines). Import
# the one definition; __all__ keeps the re-export for existing importers.
from victor.core.verticals.protocols.middleware import MiddlewareProtocol  # noqa: F401,E402

__all__ = [
    "ToolExtensionsProtocol",
    "MiddlewareProtocol",
]
