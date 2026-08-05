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

"""Middleware Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for tool execution middleware.
Following ISP, these protocols are focused on a single responsibility:
intercepting and modifying tool calls.

Usage:
    from victor.core.verticals.protocols.middleware import (
        MiddlewareProtocol,
    )

    class CodeValidationMiddleware(MiddlewareProtocol):
        async def before_tool_call(
            self, tool_name: str, arguments: Dict[str, Any]
        ) -> MiddlewareResult:
            if tool_name == "write_file":
                # Validate code before writing
                ...
            return MiddlewareResult()
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Protocol, Set, runtime_checkable

from victor.core.vertical_types import MiddlewarePriority, MiddlewareResult

# =============================================================================
# Middleware Protocol
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

    @abstractmethod
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
    "MiddlewareProtocol",
    "MiddlewarePriority",
    "MiddlewareResult",
]
