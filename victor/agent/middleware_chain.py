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

"""Middleware Chain for tool execution pipeline.

This module provides a middleware chain pattern for intercepting and
processing tool calls before and after execution.

Design:
- Middleware is sorted by priority for consistent execution order
- Before: Lower priority values execute first (validation before processing)
- After: Higher priority values execute first (cleanup after logging)
- Each middleware can modify arguments, abort execution, or transform results

Usage:
    from victor.agent.middleware_chain import MiddlewareChain
    from victor.core.verticals.protocols import MiddlewareProtocol

    # Create chain
    chain = MiddlewareChain()

    # Add middleware
    chain.add(ValidationMiddleware())
    chain.add(LoggingMiddleware())

    # Process tool call
    result = await chain.process_before("write_file", {"path": "test.py", "content": "..."})
    if result.proceed:
        tool_result = await tool.execute(**result.modified_arguments or arguments)
        final_result = await chain.process_after("write_file", arguments, tool_result, True)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from victor.core.verticals.protocols import (
    MiddlewarePriority,
    MiddlewareProtocol,
    MiddlewareResult,
)

if TYPE_CHECKING:
    from victor.agent.vertical_context import VerticalContext

logger = logging.getLogger(__name__)


class MiddlewareChain:
    """Chain of middleware for tool execution processing.

    Manages a sorted collection of middleware and provides methods
    to process tool calls through the entire chain.

    Attributes:
        _middleware: List of middleware instances, sorted by priority
        _enabled: Whether the chain is enabled
    """

    def __init__(self, vertical_context: Optional["VerticalContext"] = None) -> None:
        """Initialize an empty middleware chain.

        Args:
            vertical_context: Optional vertical context for DIP-compliant
                vertical-specific middleware behavior
        """
        self._middleware: List[MiddlewareProtocol] = []
        self._enabled: bool = True
        self._sorted: bool = True
        self._vertical_context: Optional["VerticalContext"] = vertical_context

    @property
    def enabled(self) -> bool:
        """Whether the middleware chain is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the middleware chain."""
        self._enabled = value

    @property
    def vertical_context(self) -> Optional["VerticalContext"]:
        """Get the current vertical context."""
        return self._vertical_context

    def set_vertical_context(self, context: Optional["VerticalContext"]) -> None:
        """Set vertical context for vertical-aware middleware processing.

        This enables middleware to access vertical-specific configuration
        and behavior (DIP compliance).

        Args:
            context: VerticalContext instance or None to clear
        """
        self._vertical_context = context
        if context:
            logger.debug(
                "MiddlewareChain vertical context set: %s",
                context.vertical_name,
            )

    def add(self, middleware: MiddlewareProtocol) -> None:
        """Add middleware to the chain.

        Args:
            middleware: Middleware instance to add
        """
        self._middleware.append(middleware)
        self._sorted = False
        logger.debug(
            "Added middleware %s with priority %s",
            type(middleware).__name__,
            middleware.get_priority().name,
        )

    def remove(self, middleware: MiddlewareProtocol) -> bool:
        """Remove middleware from the chain.

        Args:
            middleware: Middleware instance to remove

        Returns:
            True if middleware was found and removed
        """
        try:
            self._middleware.remove(middleware)
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        """Remove all middleware from the chain."""
        self._middleware.clear()
        self._sorted = True

    def _ensure_sorted(self) -> None:
        """Ensure middleware is sorted by priority."""
        if not self._sorted:
            self._middleware.sort(key=lambda m: m.get_priority().value)
            self._sorted = True

    def _get_applicable_middleware(self, tool_name: str) -> List[MiddlewareProtocol]:
        """Get middleware applicable to a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            List of applicable middleware
        """
        self._ensure_sorted()
        applicable = []
        for mw in self._middleware:
            tools = mw.get_applicable_tools()
            if tools is None or tool_name in tools:
                applicable.append(mw)
        return applicable

    async def process_before(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MiddlewareResult:
        """Process a tool call through all before_tool_call middleware.

        Middleware is processed in priority order (low values first).
        If any middleware returns proceed=False, processing stops.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            Aggregated MiddlewareResult with final decision
        """
        if not self._enabled:
            return MiddlewareResult()

        applicable = self._get_applicable_middleware(tool_name)
        current_args = arguments.copy()

        # Initialize metadata with vertical context info (DIP - provides vertical
        # awareness to middleware without tight coupling)
        aggregated_metadata: Dict[str, Any] = {}
        if self._vertical_context:
            aggregated_metadata["vertical_name"] = self._vertical_context.vertical_name
            aggregated_metadata["vertical_mode"] = getattr(self._vertical_context, "mode", None)

        for middleware in applicable:
            try:
                result = await middleware.before_tool_call(tool_name, current_args)

                # Aggregate metadata
                aggregated_metadata.update(result.metadata)

                # Check if we should abort
                if not result.proceed:
                    logger.info(
                        "Middleware %s blocked tool %s: %s",
                        type(middleware).__name__,
                        tool_name,
                        result.error_message,
                    )
                    return MiddlewareResult(
                        proceed=False,
                        error_message=result.error_message,
                        metadata=aggregated_metadata,
                    )

                # Apply argument modifications
                if result.modified_arguments:
                    current_args = result.modified_arguments

            except Exception as e:
                logger.error(
                    "Middleware %s failed in before_tool_call: %s",
                    type(middleware).__name__,
                    e,
                )
                # Continue with other middleware on error
                continue

        return MiddlewareResult(
            proceed=True,
            modified_arguments=current_args if current_args != arguments else None,
            metadata=aggregated_metadata,
        )

    async def process_after(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Any:
        """Process a tool result through all after_tool_call middleware.

        Middleware is processed in reverse priority order (high values first).
        Each middleware can transform the result.

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed
            result: Result from the tool execution
            success: Whether the tool execution succeeded

        Returns:
            Potentially modified result
        """
        if not self._enabled:
            return result

        applicable = self._get_applicable_middleware(tool_name)
        current_result = result

        # Process in reverse order for after calls
        for middleware in reversed(applicable):
            try:
                modified = await middleware.after_tool_call(
                    tool_name, arguments, current_result, success
                )
                if modified is not None:
                    current_result = modified
            except Exception as e:
                logger.error(
                    "Middleware %s failed in after_tool_call: %s",
                    type(middleware).__name__,
                    e,
                )
                # Continue with other middleware on error
                continue

        return current_result

    async def process_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        executor: Any,
    ) -> Any:
        """Process a complete tool call through the middleware chain.

        Convenience method that handles the full before -> execute -> after flow.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            executor: Async callable that executes the tool

        Returns:
            Tool result (potentially modified by after middleware)

        Raises:
            MiddlewareAbortError: If middleware blocked the call
        """
        # Process before
        before_result = await self.process_before(tool_name, arguments)

        if not before_result.proceed:
            raise MiddlewareAbortError(
                tool_name=tool_name,
                message=before_result.error_message or "Blocked by middleware",
            )

        # Execute tool with potentially modified arguments
        exec_args = before_result.modified_arguments or arguments
        success = True
        result = None

        try:
            result = await executor(**exec_args)
        except Exception as e:
            success = False
            result = str(e)
            raise
        finally:
            # Always process after, even on error
            result = await self.process_after(tool_name, exec_args, result, success)

        return result

    def get_middleware_info(self) -> List[Dict[str, Any]]:
        """Get information about registered middleware.

        Returns:
            List of dicts with middleware info
        """
        self._ensure_sorted()
        return [
            {
                "name": type(mw).__name__,
                "priority": mw.get_priority().name,
                "applicable_tools": mw.get_applicable_tools(),
            }
            for mw in self._middleware
        ]

    def __len__(self) -> int:
        """Get number of middleware in chain."""
        return len(self._middleware)

    def __bool__(self) -> bool:
        """Check if chain has any middleware."""
        return len(self._middleware) > 0


class MiddlewareAbortError(Exception):
    """Error raised when middleware aborts a tool call."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"Tool '{tool_name}' blocked: {message}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_middleware_chain(
    middleware_list: Optional[List[MiddlewareProtocol]] = None,
    vertical_context: Optional["VerticalContext"] = None,
) -> MiddlewareChain:
    """Create a middleware chain with optional initial middleware.

    Args:
        middleware_list: Optional list of middleware to add
        vertical_context: Optional vertical context for DIP-compliant
            vertical-specific middleware behavior

    Returns:
        Configured MiddlewareChain
    """
    chain = MiddlewareChain(vertical_context=vertical_context)
    if middleware_list:
        for mw in middleware_list:
            chain.add(mw)
    return chain


__all__ = [
    "MiddlewareChain",
    "MiddlewareAbortError",
    "create_middleware_chain",
]
