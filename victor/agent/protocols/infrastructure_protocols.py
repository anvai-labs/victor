"""Protocol definitions for infrastructure protocols."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

__all__ = [
    "ObservabilityProtocol",
    "MetricsCollectorProtocol",
    # Recovery (ISP-split)
    "FailureDetectorProtocol",
    "RecoveryExecutorProtocol",
    "RecoveryDiagnosticsProtocol",
    "RecoveryHandlerProtocol",
    "ResponseSanitizerProtocol",
    "ArgumentNormalizerProtocol",
    "ProjectContextProtocol",
    "CodeExecutionManagerProtocol",
    "WorkflowRegistryProtocol",
    "UsageAnalyticsProtocol",
    "ContextCompactorProtocol",
    "DebugLoggerProtocol",
    "ReminderManagerProtocol",
    "RLCoordinatorProtocol",
    "SafetyCheckerProtocol",
    "AutoCommitterProtocol",
    "MCPBridgeProtocol",
    "UsageLoggerProtocol",
    "SystemPromptBuilderProtocol",
    "ParallelExecutorProtocol",
    "ResponseCompleterProtocol",
    # Vertical storage (ISP-split)
    "VerticalMiddlewareStorageProtocol",
    "SafetyPatternStorageProtocol",
    "TeamSpecStorageProtocol",
    "VerticalStorageProtocol",
    "TaskTrackerProtocol",
    "CompactionSummarizerProtocol",
    "HierarchicalCompactionProtocol",
    "SessionContextLinkerProtocol",
]


@runtime_checkable
class ObservabilityProtocol(Protocol):
    """Protocol for observability integration.

    Provides event emission for monitoring and tracing.
    """

    def on_tool_start(self, tool_name: str, arguments: Dict[str, Any], tool_id: str) -> None:
        """Called when tool execution starts."""
        ...

    def on_tool_end(
        self,
        tool_name: str,
        result: Any,
        success: bool,
        tool_id: str,
        error: Optional[str] = None,
    ) -> None:
        """Called when tool execution ends."""
        ...

    def wire_state_machine(self, state_machine: Any) -> None:
        """Wire state machine for automatic state change events."""
        ...

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Called when an error occurs."""
        ...


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection."""

    def on_tool_start(self, tool_name: str, arguments: Dict[str, Any], iteration: int) -> None:
        """Record tool start metrics."""
        ...

    def on_tool_complete(self, result: Any) -> None:
        """Record tool completion metrics."""
        ...

    def on_streaming_session_complete(self, session: Any) -> None:
        """Record session completion metrics."""
        ...


@runtime_checkable
class FailureDetectorProtocol(Protocol):
    """Protocol for detecting model failures (ISP: read-only detection)."""

    def detect_failure(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        mentioned_tools: Optional[List[str]],
        elapsed_time: float,
        session_idle_timeout: float,
        quality_score: float,
        consecutive_failures: int,
        recent_responses: Optional[List[str]],
        context_utilization: Optional[float],
    ) -> Optional[Any]:
        """Detect failure type from response characteristics.

        Returns:
            FailureType if failure detected, None otherwise
        """
        ...


@runtime_checkable
class RecoveryExecutorProtocol(Protocol):
    """Protocol for executing recovery actions (ISP: mutating recovery)."""

    async def recover(
        self,
        failure_type: Any,
        provider: str,
        model: str,
        content: str,
        tool_calls_made: int,
        tool_budget: int,
        iteration_count: int,
        max_iterations: int,
        elapsed_time: float,
        session_idle_timeout: float,
        current_temperature: float,
        consecutive_failures: int,
        mentioned_tools: Optional[List[str]],
        recent_responses: Optional[List[str]],
        quality_score: float,
        task_type: str,
        is_analysis_task: bool,
        is_action_task: bool,
        session_id: Optional[str],
    ) -> Any:
        """Attempt recovery using appropriate strategy.

        Returns:
            RecoveryOutcome with action to take
        """
        ...

    def reset_session(self, session_id: str) -> None:
        """Reset recovery state for a new session."""
        ...


@runtime_checkable
class RecoveryDiagnosticsProtocol(Protocol):
    """Protocol for recovery metrics and diagnostics (ISP: observability)."""

    def record_outcome(self, success: bool, quality_improvement: float) -> None:
        """Record recovery outcome for Q-learning."""
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about recovery system."""
        ...


@runtime_checkable
class RecoveryHandlerProtocol(
    FailureDetectorProtocol,
    RecoveryExecutorProtocol,
    RecoveryDiagnosticsProtocol,
    Protocol,
):
    """Composed protocol for full recovery handler (backward compatible).

    Clients that need only failure detection should depend on
    FailureDetectorProtocol. Clients that need only recovery execution
    should depend on RecoveryExecutorProtocol. This composed protocol
    is for the orchestrator which needs all capabilities.
    """

    ...


# NOTE: ResponseSanitizerProtocol and ReminderManagerProtocol are canonical
# service-owned runtime protocols in
# victor.agent.services.protocols.infrastructure_runtime. This module re-exports
# them at the bottom as deprecated compatibility names.


@runtime_checkable
class ArgumentNormalizerProtocol(Protocol):
    """Protocol for argument normalization."""

    def normalize(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tool arguments.

        Handles malformed arguments, type coercion, etc.
        """
        ...


@runtime_checkable
class ProjectContextProtocol(Protocol):
    """Protocol for project context loading."""

    @property
    def content(self) -> Optional[str]:
        """Get loaded project context content."""
        ...

    def load(self) -> None:
        """Load project context from file."""
        ...

    def get_system_prompt_addition(self) -> str:
        """Get context as system prompt addition."""
        ...


@runtime_checkable
class CodeExecutionManagerProtocol(Protocol):
    """Protocol for code execution management."""

    def start(self) -> None:
        """Start the execution manager."""
        ...

    def stop(self) -> None:
        """Stop the execution manager."""
        ...


@runtime_checkable
class WorkflowRegistryProtocol(Protocol):
    """Protocol for workflow registry."""

    def register(self, workflow: Any) -> None:
        """Register a workflow."""
        ...

    def get(self, name: str) -> Optional[Any]:
        """Get a workflow by name."""
        ...


@runtime_checkable
class UsageAnalyticsProtocol(Protocol):
    """Protocol for usage analytics."""

    def record_tool_selection(
        self,
        tool_name: str,
        score: float,
        selected: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a tool selection decision."""
        ...

    def end_session(self) -> None:
        """End the current analytics session."""
        ...

    @classmethod
    def get_instance(cls, config: Any) -> "UsageAnalyticsProtocol":
        """Get singleton instance."""
        ...


@runtime_checkable
class ContextCompactorProtocol(Protocol):
    """Protocol for context compaction."""

    def maybe_compact_proactively(self) -> bool:
        """Attempt proactive compaction if threshold reached.

        Returns:
            True if compaction occurred
        """
        ...

    def truncate_tool_result(
        self, tool_name: str, result: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Truncate tool result if needed."""
        ...


@runtime_checkable
class DebugLoggerProtocol(Protocol):
    """Protocol for debug logging service.

    Provides clean, scannable debug output focused on meaningful events.
    """

    def reset(self) -> None:
        """Reset state for new conversation."""
        ...

    def log_iteration_start(self, iteration: int, **context: Any) -> None:
        """Log iteration start."""
        ...

    def log_iteration_end(
        self, iteration: int, has_tool_calls: bool = False, **context: Any
    ) -> None:
        """Log iteration end summary."""
        ...

    def log_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        iteration: int,
    ) -> None:
        """Log tool call."""
        ...

    def log_tool_result(
        self,
        tool_name: str,
        success: bool,
        output: Any,
        elapsed_ms: float,
    ) -> None:
        """Log tool result."""
        ...


# NOTE: RLCoordinatorProtocol is a canonical service-owned runtime protocol in
# victor.agent.services.protocols.runtime_support. This module re-exports it at
# the bottom as a deprecated compatibility name.


@runtime_checkable
class SafetyCheckerProtocol(Protocol):
    """Protocol for safety checking service.

    Detects dangerous operations and requests confirmation.
    """

    def is_write_tool(self, tool_name: str) -> bool:
        """Check if a tool is a write/modify operation."""
        ...

    async def check_and_confirm(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Check operation safety and request confirmation if needed.

        Returns:
            Tuple of (should_proceed, optional_rejection_reason)
        """
        ...

    def add_custom_pattern(
        self,
        pattern: str,
        description: str,
        risk_level: str = "HIGH",
        category: str = "custom",
    ) -> None:
        """Add a custom safety pattern from vertical extensions."""
        ...


@runtime_checkable
class AutoCommitterProtocol(Protocol):
    """Protocol for automatic git commits service.

    Handles automatic git commits for AI-assisted changes.
    """

    def is_git_repo(self) -> bool:
        """Check if workspace is a git repository."""
        ...

    def has_changes(self, files: Optional[List[str]] = None) -> bool:
        """Check if there are uncommitted changes."""
        ...

    def commit_changes(
        self,
        files: Optional[List[str]] = None,
        description: str = "AI-assisted changes",
        change_type: Optional[str] = None,
        scope: Optional[str] = None,
        auto_stage: bool = True,
    ) -> Any:
        """Commit changes to git."""
        ...


@runtime_checkable
class MCPBridgeProtocol(Protocol):
    """Protocol for Model Context Protocol bridge.

    Provides access to MCP tools as Victor tools.
    """

    def configure_client(self, client: Any, prefix: str = "mcp") -> None:
        """Configure the MCP client.

        Args:
            client: MCPClient instance
            prefix: Prefix for tool names
        """
        ...

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return MCP tools as Victor tool definitions with a name prefix."""
        ...


@runtime_checkable
class UsageLoggerProtocol(Protocol):
    """Protocol for usage logging service.

    Logs tool and provider usage for analytics.
    """

    def log_tool_call(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        **metadata: Any,
    ) -> None:
        """Log a tool call.

        Args:
            tool_name: Name of the tool
            success: Whether the call succeeded
            duration_ms: Duration in milliseconds
            **metadata: Additional metadata
        """
        ...

    def log_provider_call(
        self,
        provider: str,
        model: str,
        tokens_used: int,
        duration_ms: float,
        **metadata: Any,
    ) -> None:
        """Log a provider API call.

        Args:
            provider: Provider name
            model: Model identifier
            tokens_used: Number of tokens consumed
            duration_ms: Duration in milliseconds
            **metadata: Additional metadata
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary of usage statistics
        """
        ...


@runtime_checkable
class SystemPromptBuilderProtocol(Protocol):
    """Protocol for system prompt building service.

    Constructs system prompts from various components.
    """

    def build(
        self,
        base_prompt: str,
        tool_descriptions: Optional[str] = None,
        project_context: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Build system prompt from components.

        Args:
            base_prompt: Base system prompt
            tool_descriptions: Tool descriptions to include
            project_context: Project-specific context
            **kwargs: Additional prompt components

        Returns:
            Complete system prompt
        """
        ...


@runtime_checkable
class ParallelExecutorProtocol(Protocol):
    """Protocol for parallel tool execution service.

    Executes multiple tools in parallel.
    """

    async def execute_parallel(
        self,
        tool_calls: List[Any],
        **kwargs: Any,
    ) -> List[Any]:
        """Execute multiple tool calls in parallel.

        Args:
            tool_calls: List of tool calls to execute
            **kwargs: Additional execution parameters

        Returns:
            List of tool results
        """
        ...


@runtime_checkable
class ResponseCompleterProtocol(Protocol):
    """Protocol for response completion service.

    Completes partial responses and handles tool failures.
    """

    async def complete_response(
        self,
        partial_response: str,
        context: Any,
        **kwargs: Any,
    ) -> str:
        """Complete a partial response.

        Args:
            partial_response: Partial response text
            context: Completion context
            **kwargs: Additional completion parameters

        Returns:
            Completed response
        """
        ...


@runtime_checkable
class VerticalMiddlewareStorageProtocol(Protocol):
    """Protocol for vertical middleware storage (ISP: middleware only)."""

    def set_middleware(self, middleware: List[Any]) -> None:
        """Store middleware configuration."""
        ...

    def get_middleware(self) -> List[Any]:
        """Retrieve middleware configuration."""
        ...


@runtime_checkable
class SafetyPatternStorageProtocol(Protocol):
    """Protocol for vertical safety pattern storage (ISP: safety only)."""

    def set_safety_patterns(self, patterns: List[Any]) -> None:
        """Store safety patterns."""
        ...

    def get_safety_patterns(self) -> List[Any]:
        """Retrieve safety patterns."""
        ...


@runtime_checkable
class TeamSpecStorageProtocol(Protocol):
    """Protocol for vertical team spec storage (ISP: teams only)."""

    def set_team_specs(self, specs: Dict[str, Any]) -> None:
        """Store team specifications."""
        ...

    def get_team_specs(self) -> Dict[str, Any]:
        """Retrieve team specifications."""
        ...


@runtime_checkable
class VerticalStorageProtocol(
    VerticalMiddlewareStorageProtocol,
    SafetyPatternStorageProtocol,
    TeamSpecStorageProtocol,
    Protocol,
):
    """Composed protocol for full vertical storage (backward compatible).

    Clients that need only middleware should depend on
    VerticalMiddlewareStorageProtocol. This composed protocol is for
    the orchestrator which needs all three capabilities.
    """

    ...


@runtime_checkable
class TaskTrackerProtocol(Protocol):
    """Protocol for task tracking."""

    def start_task(self, task_id: str, description: str) -> None:
        """Start tracking a task."""
        ...

    def complete_task(self, task_id: str) -> None:
        """Mark task as complete."""
        ...

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all active tasks."""
        ...

    def is_loop_detected(self) -> bool:
        """Check if execution loop is detected."""
        ...


@runtime_checkable
class CompactionSummarizerProtocol(Protocol):
    """Protocol for compaction summarization strategies."""

    def summarize(self, removed_messages: List[Any], ledger: Optional[object] = None) -> str:
        """Summarize removed messages into a compact context string."""
        ...


@runtime_checkable
class HierarchicalCompactionProtocol(Protocol):
    """Protocol for hierarchical compaction management."""

    def add_summary(self, summary: str, turn_index: int) -> None:
        """Add a compaction summary; may trigger epoch creation."""
        ...

    def get_active_context(self, max_chars: int = 2000) -> str:
        """Get epoch summaries + recent individual summaries within budget."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        ...


@runtime_checkable
class SessionContextLinkerProtocol(Protocol):
    """Protocol for cross-session context linking."""

    def build_resume_context(self, session_id: str) -> Any:
        """Build rich resume context from persisted session data."""
        ...

    def find_related_sessions(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Semantic search across sessions for cross-session linking."""
        ...


from victor.agent.services.protocols.infrastructure_runtime import (
    ReminderManagerProtocol,
    ResponseSanitizerProtocol,
)
from victor.agent.services.protocols.runtime_support import (
    RLLearningRuntimeProtocol as RLCoordinatorProtocol,
)
