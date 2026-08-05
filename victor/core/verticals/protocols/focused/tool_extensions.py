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

from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from victor.core.tool_types import ToolDependency, ToolDependencyProviderProtocol
from victor.core.vertical_types import MiddlewarePriority, MiddlewareResult

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
# Middleware Protocol (Re-exported for convenience)
# =============================================================================


@runtime_checkable
class MiddlewareProtocol(Protocol):
    """Protocol for tool execution middleware.

    Middleware can intercept and modify tool calls before and after execution.
    Use for validation, transformation, logging, or domain-specific processing.

    Example:
        class CodeValidationMiddleware(MiddlewareProtocol):
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                if tool_name == "write_file" and "content" in arguments:
                    # Validate code syntax before writing
                    is_valid, error = self._validate_syntax(arguments["content"])
                    if not is_valid:
                        return MiddlewareResult(
                            proceed=False,
                            error_message=f"Syntax error: {error}"
                        )
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.HIGH
    """

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Called before a tool is executed.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult indicating whether to proceed
        """
        ...

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Called after a tool is executed.

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed
            result: Result from the tool execution
            success: Whether the tool execution succeeded

        Returns:
            Modified result (or None to keep original)
        """
        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware.

        Returns:
            Priority level for execution ordering
        """
        return MiddlewarePriority.NORMAL

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            Set of tool names, or None for all tools
        """
        return None


__all__ = [
    "ToolExtensionsProtocol",
    "MiddlewareProtocol",
]
