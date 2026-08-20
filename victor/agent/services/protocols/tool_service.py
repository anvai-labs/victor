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

"""Tool service protocol.

Defines the interface for tool operations, including selection,
execution, and budgeting.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.services.tool_service import ToolResultContext
    from victor.tools.base import ToolResult


@runtime_checkable
class ToolSelectionContext(Protocol):
    """Context for tool selection decisions.

    Provides information about the current state to help
    make intelligent tool selection decisions.
    """

    @property
    def user_message(self) -> str:
        """The current user message."""
        ...

    @property
    def conversation_stage(self) -> str:
        """The current conversation stage."""
        ...

    @property
    def available_tools(self) -> List[str]:
        """List of available tool names."""
        ...

    @property
    def recent_tools(self) -> List[str]:
        """Recently used tools."""
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Additional metadata for selection."""
        ...


@runtime_checkable
class ToolServiceProtocol(Protocol):
    """[CANONICAL] Protocol for tool operations service.

    This protocol represents the target architecture for tool operations,
    replacing the facade-driven Coordinator pattern with a state-passed
    Service pattern.

    This protocol follows the Interface Segregation Principle (ISP)
    by focusing only on tool-related operations.

    Methods:
        select_tools: Select tools based on context and constraints
        execute_tool: Execute a single tool with validation
        execute_tools_parallel: Execute multiple tools in parallel
        get_tool_budget: Get remaining tool budget
        get_tool_usage_stats: Get tool usage statistics

    Example:
        class MyToolService(ToolServiceProtocol):
            def __init__(self, selector, executor, budget_manager):
                self._selector = selector
                self._executor = executor
                self._budget = budget_manager

            async def select_tools(self, context, max_tools=10):
                return await self._selector.select(context, max_tools)

            async def execute_tool(self, tool_name, arguments):
                if self._budget.is_exhausted():
                    raise ToolBudgetExceededError()
                result = await self._executor.execute(tool_name, arguments)
                self._budget.record_usage()
                return result
    """

    async def select_tools(
        self,
        context: "ToolSelectionContext",
        max_tools: int = 10,
    ) -> List[str]:
        """Select tools for execution based on context.

        Uses semantic analysis and heuristics to select the most
        relevant tools for the current task.

        Args:
            context: Tool selection context with message, stage, available tools
            max_tools: Maximum number of tools to select

        Returns:
            List of selected tool names, ordered by relevance

        Raises:
            ToolSelectionError: If tool selection fails
        """
        ...

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> "ToolResult":
        """Execute a single tool with validation and error handling.

        Validates arguments, checks budget, executes the tool,
        and tracks usage for analytics.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            ToolResult with execution outcome

        Raises:
            ToolNotFoundError: If tool is not registered
            ToolValidationError: If arguments are invalid
            ToolBudgetExceededError: If tool budget is exhausted
            ToolExecutionError: If tool execution fails
        """
        ...

    async def execute_tools_parallel(
        self,
        tool_calls: List[tuple[str, Dict[str, Any]]],
        max_parallel: int = 5,
    ) -> AsyncIterator["ToolResult"]:
        """Execute multiple tools in parallel.

        Executes independent tools concurrently for improved performance.

        Args:
            tool_calls: List of (tool_name, arguments) tuples
            max_parallel: Maximum number of concurrent executions

        Yields:
            ToolResult objects as they complete

        Raises:
            ToolExecutionError: If any critical tool execution fails
        """
        ...

    def get_tool_budget(self) -> int:
        """Get the remaining executable tool budget for the current turn.

        The executable budget limits the number of tool calls inside the
        current prompt/turn to prevent loops and excessive API usage.

        Returns:
            Number of remaining tool calls allowed

        Example:
            budget = tool_service.get_tool_budget()
            if budget < 5:
                logger.warning(f"Low tool budget: {budget} remaining")
        """
        ...

    def set_tool_budget(self, budget: int) -> None:
        """Set the tool budget limit.

        Args:
            budget: Maximum number of tool calls allowed

        Raises:
            ValueError: If budget is negative
        """
        ...

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics.

        Returns statistics about tool usage including:
        - Total tool calls
        - Calls by tool name
        - Success/failure rates
        - Average execution time

        Returns:
            Dictionary with usage statistics

        Example:
            stats = tool_service.get_tool_usage_stats()
            print(f"Total calls: {stats['total_calls']}")
            print(f"Success rate: {stats['success_rate']:.1%}")
        """
        ...

    def reset_tool_budget(self) -> None:
        """Fully reset the tool budget to its initial limit.

        Useful for:
        - Starting new sessions
        - Testing and development
        - Explicit recovery from budget exhaustion
        """
        ...

    def process_tool_results(
        self,
        pipeline_result: Any,
        ctx: "ToolResultContext",
    ) -> List[Dict[str, Any]]:
        """Process tool execution results.

        Handles state mutations, analytics, failure detection,
        conversation injection, and error display.

        Args:
            pipeline_result: Result from ToolPipeline.execute_tool_calls
            ctx: Context carrying mutable state and callbacks

        Returns:
            List of result dicts with name, success, elapsed, error, etc.
        """
        ...

    def get_available_tools(self) -> Set[str]:
        """Get all registered tool names."""
        ...

    def get_enabled_tools(self) -> Set[str]:
        """Get currently enabled tool names for the session."""
        ...

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set which tools are enabled for the session."""
        ...

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check whether a tool is enabled for the current session."""
        ...

    def resolve_tool_alias(self, tool_name: str) -> str:
        """Resolve a tool alias to its canonical runtime name."""
        ...

    def parse_and_validate_tool_calls(
        self,
        tool_calls: Optional[List[Dict[str, Any]]],
        full_content: str,
        tool_adapter: Any,
    ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        """Parse, validate, normalize, and filter tool calls from model output."""
        ...

    async def execute_tool_with_retry(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any],
        tool_executor: Optional[Callable[..., Awaitable[Any]]] = None,
        cache: Optional[Any] = None,
        on_success: Optional[Callable[[str, Dict[str, Any], Any], None]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Any], bool, Optional[str]]:
        """Execute a tool with retry logic and exponential backoff."""
        ...

    def normalize_tool_arguments(
        self,
        tool_args: Dict[str, Any],
        tool_name: str,
    ) -> Tuple[Dict[str, Any], Any]:
        """Normalize raw tool arguments before execution."""
        ...

    def build_tool_access_context(self) -> Any:
        """Build access-control context for tool gating decisions."""
        ...

    def validate_tool_call(
        self,
        tool_call: Any,
        sanitizer: Any,
        is_tool_enabled_fn: Optional[Callable[[str], bool]] = None,
    ) -> Any:
        """Validate one tool call and return a compatibility validation result."""
        ...

    def normalize_arguments_full(
        self,
        tool_name: str,
        original_name: str,
        raw_args: Any,
        argument_normalizer: Any,
        tool_adapter: Any,
        failed_signatures: Optional[Set[Tuple[str, str]]] = None,
    ) -> Any:
        """Normalize arguments using parser repair, adapter defaults, and dedup metadata."""
        ...

    def on_tool_complete(
        self,
        result: Any,
        metrics_collector: Optional[Any] = None,
        *,
        read_files_session: Optional[Set[str]] = None,
        required_files: Optional[List[str]] = None,
        required_outputs: Optional[List[str]] = None,
        nudge_sent_flag: Optional[List[bool]] = None,
        add_message: Optional[Callable[[str, str], None]] = None,
        observability: Optional[Any] = None,
        iteration_count: int = 0,
        tool_name: Optional[str] = None,
        elapsed: float = 0.0,
        session_id: Optional[str] = None,
    ) -> None:
        """Notify the service that a tool execution has completed.

        Used by CallbackCoordinator and legacy coordinator shims to record
        tool completion metrics, nudges, and observability events without
        requiring ToolCoordinator to remain the canonical owner.

        Args:
            result: The tool execution result
            metrics_collector: Optional metrics collector for the full completion flow
            read_files_session: Mutable set of read files for synthesis nudges
            required_files: Files required to complete the current task
            required_outputs: Outputs still required from the current task
            nudge_sent_flag: Mutable one-item flag tracking synthesis nudges
            add_message: Optional message injection callback
            observability: Optional observability integration
            iteration_count: Current iteration/tool-call count
            tool_name: Optional explicit tool name for narrow observability hooks
            elapsed: Elapsed wall-clock seconds for the execution
            session_id: Optional session identifier for metrics
        """
        ...

    def is_healthy(self) -> bool:
        """Check if the tool service is healthy.

        A healthy tool service should:
        - Have tools registered
        - Have budget available
        - Have selector and executor available

        Returns:
            True if the service is healthy, False otherwise
        """
        ...


@runtime_checkable
class ToolCachingProtocol(Protocol):
    """Protocol for tool result caching.

    Enables intelligent caching of tool results to avoid
    redundant executions and improve performance.
    """

    async def get_cached_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional["ToolResult"]:
        """Get cached tool result if available.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Cached ToolResult if available and valid, None otherwise
        """
        ...

    async def cache_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: "ToolResult",
        ttl: int = 600,
    ) -> None:
        """Cache a tool result.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments (used as cache key)
            result: Result to cache
            ttl: Time to live in seconds (default: 10 minutes)
        """
        ...

    def invalidate_cache(self, tool_name: Optional[str] = None) -> None:
        """Invalidate cached results.

        Args:
            tool_name: Specific tool to invalidate, or None for all tools
        """
        ...
