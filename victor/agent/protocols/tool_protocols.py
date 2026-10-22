"""Protocol definitions for tool protocols."""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    TYPE_CHECKING,
    Tuple,
    runtime_checkable,
)

from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from victor.agent.conversation.state_machine import ConversationStage
    from victor.agent.tool_pipeline import ToolCallResult
    from victor.tools.base import CostTier


__all__ = [
    "AgentToolSelectionContext",
    "ToolSelectorFeatures",
    "ToolRegistryProtocol",
    "ToolSelectorProtocol",
    "ToolExecutorProtocol",
    "ToolCacheProtocol",
    "ToolRegistrarProtocol",
    "ToolSequenceTrackerProtocol",
    "ToolOutputFormatterProtocol",
    "ToolCallTrackerProtocol",
    "ToolDeduplicationTrackerProtocol",
    "ToolDependencyGraphProtocol",
    "ToolPluginRegistryProtocol",
    "SemanticToolSelectorProtocol",
    "IToolAdapterCoordinator",
    "AccessPrecedence",
    "ToolAccessDecision",
    "ToolAccessContext",
    "IToolAccessController",
]


@dataclass
class AgentToolSelectionContext:
    """Agent-level context for tool selection decisions.

    Renamed from ToolSelectionContext to be semantically distinct:
    - AgentToolSelectionContext (here): Basic agent-level context
    - VerticalToolSelectionContext (victor.core.verticals.protocols.tool_provider): Vertical-specific
    - CrossVerticalToolSelectionContext (victor.tools.selection.protocol): Extended cross-vertical

    Provides conversation history, stage info, and task metadata
    to tool selectors for intelligent tool filtering.
    """

    # Conversation stage (e.g., PLANNING, EXECUTING, REVIEWING)
    stage: Optional[str] = None

    # Conversation stage object (typed version)
    conversation_stage: Optional[Any] = None

    # Task type from analysis (e.g., "analysis", "action", "create")
    task_type: str = "default"

    # Recent tool execution history
    recent_tools: List[str] = field(default_factory=list)

    # Current conversation turn number
    turn_number: int = 0

    # Whether this is a continuation of previous work
    is_continuation: bool = False

    # Maximum tools to select
    max_tools: int = 15

    # Planned tools from dependency planning (tool names pre-selected)
    planned_tools: Optional[List[str]] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# Backward compatibility alias
ToolSelectionContext = AgentToolSelectionContext


@dataclass
class ToolSelectorFeatures:
    """Features supported by a tool selector implementation.

    Used to advertise capabilities and for feature detection.
    """

    # ML-based semantic matching (requires embeddings)
    supports_semantic_matching: bool = False

    # Context-aware selection (conversation stage, history)
    supports_context_awareness: bool = False

    # Cost-based tool optimization
    supports_cost_optimization: bool = False

    # Learning from usage patterns
    supports_usage_learning: bool = False

    # Workflow pattern detection
    supports_workflow_patterns: bool = False

    # Whether embeddings are required
    requires_embeddings: bool = False


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Protocol for tool registry.

    Manages tool registration, lookup, and cost tiers.
    """

    def register(self, tool: Any) -> None:
        """Register a tool with the registry."""
        ...

    def get(self, name: str) -> Optional[Any]:
        """Get a tool by name."""
        ...

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        ...

    def get_tool_cost(self, name: str) -> "CostTier":
        """Get the cost tier for a tool."""
        ...

    def register_before_hook(self, hook: Callable[..., Any]) -> None:
        """Register a hook to run before tool execution."""
        ...


@runtime_checkable
class ToolSelectorProtocol(Protocol):
    """Protocol for tool selection.

    Selects appropriate tools based on user prompt and context.
    """

    def select_tools(
        self,
        prompt: str,
        max_tools: int,
        stage: Optional["ConversationStage"] = None,
    ) -> List[str]:
        """Select tools for a given prompt.

        Args:
            prompt: User prompt
            max_tools: Maximum tools to return
            stage: Current conversation stage

        Returns:
            List of selected tool names
        """
        ...

    def get_tools_for_stage(self, stage: "ConversationStage") -> Set[str]:
        """Get recommended tools for a conversation stage."""
        ...


@runtime_checkable
class ToolExecutorProtocol(Protocol):
    """Protocol for tool execution - DIP compliant.

    Handles individual tool execution with validation and context support.
    This protocol enables dependency inversion by allowing consumers to
    depend on the abstraction rather than concrete implementations.

    The protocol provides:
    - Synchronous and asynchronous execution methods
    - Argument validation before execution
    - Optional context passing for execution environment

    Usage:
        from victor.agent.protocols import ToolExecutorProtocol

        def run_tool(executor: ToolExecutorProtocol, tool: str, args: dict) -> Any:
            if executor.validate_arguments(tool, args):
                return await executor.aexecute(tool, args)
            raise ValueError(f"Invalid arguments for {tool}")

        # Mock in tests
        mock_executor = MagicMock(spec=ToolExecutorProtocol)
        mock_executor.validate_arguments.return_value = True
    """

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Any] = None,
    ) -> Any:
        """Execute a tool synchronously.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments as dictionary
            context: Optional execution context (e.g., workspace, session info)

        Returns:
            Tool execution result
        """
        ...

    async def aexecute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Any] = None,
    ) -> Any:
        """Execute a tool asynchronously.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments as dictionary
            context: Optional execution context (e.g., workspace, session info)

        Returns:
            Tool execution result
        """
        ...

    def validate_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> bool:
        """Validate tool arguments before execution.

        Checks that the provided arguments match the tool's expected schema.
        Should be called before execute() or aexecute() to ensure valid input.

        Args:
            tool_name: Name of tool to validate against
            arguments: Arguments to validate

        Returns:
            True if arguments are valid for the tool, False otherwise
        """
        ...


@runtime_checkable
class ToolCacheProtocol(Protocol):
    """Protocol for tool result caching."""

    def get(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for a tool call."""
        ...

    def set(self, tool_name: str, arguments: Dict[str, Any], result: Any) -> None:
        """Cache a tool result."""
        ...

    def invalidate(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Invalidate a cached result."""
        ...


@runtime_checkable
class ToolRegistrarProtocol(Protocol):
    """Protocol for tool registration and plugin management."""

    def set_background_task_callback(self, callback: Callable[[Any], Any]) -> None:
        """Set callback for background task creation."""
        ...


@runtime_checkable
class ToolSequenceTrackerProtocol(Protocol):
    """Protocol for tool sequence tracking."""

    def record_transition(self, from_tool: str, to_tool: str) -> None:
        """Record a tool-to-tool transition."""
        ...

    def get_next_tool_suggestions(
        self, current_tool: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Get suggested next tools based on patterns."""
        ...


@runtime_checkable
class ToolOutputFormatterProtocol(Protocol):
    """Protocol for tool output formatting."""

    def format(
        self,
        tool_name: str,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format tool output for LLM consumption.

        Args:
            tool_name: Name of the tool
            result: Raw tool result
            context: Optional formatting context

        Returns:
            Formatted output string
        """
        ...


@runtime_checkable
class ToolCallTrackerProtocol(Protocol):
    """Protocol for tool call tracking.

    Tracks recent tool calls to detect and prevent redundant operations.

    This is DISTINCT from victor.tools.deduplication/, which handles tool
    DEFINITION deduplication (resolving naming conflicts between native,
    LangChain, and MCP tools). This protocol tracks RUNTIME tool CALLS.
    """

    def add_call(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Add a tool call to the tracker.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
        """
        ...

    def is_redundant(self, tool_name: str, args: Dict[str, Any], explain: bool = False) -> bool:
        """Check if a tool call is redundant given recent history.

        Args:
            tool_name: Name of the tool to call
            args: Tool arguments
            explain: If True, log explanation for why call is redundant

        Returns:
            True if the call is likely redundant, False otherwise
        """
        ...

    def clear(self) -> None:
        """Clear all tracked tool calls."""
        ...

    def get_recent_calls(self, limit: Optional[int] = None) -> List[Any]:
        """Get recent tool calls.

        Args:
            limit: Maximum number of calls to return

        Returns:
            List of recent tool calls (most recent first)
        """
        ...


# Backward compatibility alias
ToolDeduplicationTrackerProtocol = ToolCallTrackerProtocol


@runtime_checkable
class ToolDependencyGraphProtocol(Protocol):
    """Protocol for tool dependency graph.

    Manages tool dependencies for intelligent execution ordering.
    """

    def add_dependency(self, tool: str, depends_on: str) -> None:
        """Add a dependency relationship between tools.

        Args:
            tool: Tool name that has a dependency
            depends_on: Tool that must be executed first
        """
        ...

    def get_dependencies(self, tool: str) -> List[str]:
        """Get dependencies for a tool.

        Args:
            tool: Tool name

        Returns:
            List of tool names that this tool depends on
        """
        ...

    def get_execution_order(self, tools: List[str]) -> List[str]:
        """Get optimal execution order for a list of tools.

        Args:
            tools: List of tool names

        Returns:
            Ordered list respecting dependencies
        """
        ...

    def record_transition(self, from_tool: str, to_tool: str, task_type: str) -> None:
        """Record a tool→tool transition for trajectory learning.

        Args:
            from_tool: Tool that just executed
            to_tool: Tool that executed next
            task_type: Task type context for the transition
        """
        ...

    def predict_next(self, current_tool: str, task_type: str, top_k: int = 3) -> List[tuple]:
        """Predict the most likely next tools from transition history.

        Args:
            current_tool: Tool that just ran
            task_type: Current task type
            top_k: Maximum number of predictions to return

        Returns:
            List of (tool_name, probability) tuples sorted by probability desc
        """
        ...


@runtime_checkable
class ToolPluginRegistryProtocol(Protocol):
    """Protocol for tool plugin registry.

    Manages dynamic tool loading from plugins.
    """

    def register_plugin(self, plugin_path: str) -> None:
        """Register a plugin directory.

        Args:
            plugin_path: Path to plugin directory
        """
        ...

    def discover_tools(self) -> List[Any]:
        """Discover and load tools from registered plugins.

        Returns:
            List of discovered tool instances
        """
        ...

    def reload_plugins(self) -> None:
        """Reload all registered plugins."""
        ...


@runtime_checkable
class SemanticToolSelectorProtocol(Protocol):
    """Protocol for semantic tool selection.

    Uses embeddings to select relevant tools for a task.
    """

    def select_tools(
        self,
        query: str,
        available_tools: List[Any],
        max_tools: int = 10,
        threshold: float = 0.3,
    ) -> List[Any]:
        """Select relevant tools using semantic similarity.

        Args:
            query: User query or task description
            available_tools: List of available tools
            max_tools: Maximum number of tools to select
            threshold: Minimum similarity threshold

        Returns:
            List of selected tools
        """
        ...

    def compute_similarity(self, query: str, tool_description: str) -> float:
        """Compute semantic similarity between query and tool.

        Args:
            query: User query
            tool_description: Tool description

        Returns:
            Similarity score (0-1)
        """
        ...


class IToolAdapterCoordinator(Protocol):
    """Protocol for tool adapter coordination.

    Defines interface for initializing and managing tool adapters.
    Separated to allow independent testing and mocking.
    """

    def initialize_adapter(self) -> Any:
        """Initialize tool adapter for current provider.

        Returns:
            ToolCallingCapabilities instance

        Raises:
            ValueError: If no provider is configured
        """
        ...

    def get_capabilities(self) -> Any:
        """Get tool calling capabilities.

        Returns:
            ToolCallingCapabilities instance

        Raises:
            ValueError: If adapter not initialized
        """
        ...

    def get_adapter(self) -> Any:
        """Get current tool adapter instance.

        Returns:
            Tool adapter instance

        Raises:
            ValueError: If adapter not initialized
        """
        ...

    def is_initialized(self) -> bool:
        """Check if adapter has been initialized.

        Returns:
            True if adapter is initialized, False otherwise
        """
        ...


class AccessPrecedence(int, Enum):
    """Precedence levels for tool access control.

    Lower numbers = higher precedence.
    Safety (L0) > Mode (L1) > Session (L2) > Vertical (L3) > Stage (L4) > Intent (L5)
    """

    SAFETY = 0  # DangerLevel checks - highest precedence
    MODE = 1  # Mode restrictions (BUILD/PLAN/EXPLORE)
    SESSION = 2  # Session-enabled tools set
    VERTICAL = 3  # TieredToolConfig from vertical
    STAGE = 4  # Conversation stage filtering
    INTENT = 5  # DISPLAY_ONLY/READ_ONLY blocking


@dataclass
class ToolAccessDecision:
    """Result of a tool access check.

    Provides detailed information about why a tool was allowed or denied,
    and which layer made the decision.

    Attributes:
        allowed: Whether the tool is allowed
        tool_name: Name of the tool checked
        reason: Human-readable explanation
        source: Which layer made the decision
        precedence_level: Numeric precedence (lower = higher priority)
        checked_layers: Names of layers that were checked
        layer_results: Results from each checked layer
    """

    allowed: bool
    tool_name: str
    reason: str
    source: str  # Which layer decided (e.g., "mode", "safety")
    precedence_level: int = 0
    checked_layers: List[str] = field(default_factory=list)
    layer_results: Dict[str, bool] = field(default_factory=dict)

    def explain(self) -> str:
        """Get detailed explanation of the decision."""
        status = "ALLOWED" if self.allowed else "DENIED"
        layers = ", ".join(self.checked_layers) if self.checked_layers else "none"
        return (
            f"Tool '{self.tool_name}' is {status}\n"
            f"  Decision by: {self.source} (precedence {self.precedence_level})\n"
            f"  Reason: {self.reason}\n"
            f"  Layers checked: {layers}"
        )


@dataclass
class ToolAccessContext:
    """Context for tool access decisions.

    Provides all information needed by access layers to make decisions.

    Attributes:
        user_message: Current user message (for intent detection)
        conversation_stage: Current conversation stage
        intent: Detected user intent
        current_mode: Current agent mode (BUILD/PLAN/EXPLORE)
        session_enabled_tools: Tools enabled for this session
        vertical_name: Active vertical (if any)
        metadata: Additional context data
    """

    user_message: Optional[str] = None
    conversation_stage: Optional["ConversationStage"] = None
    intent: Optional[Any] = None  # ActionIntent
    current_mode: Optional[str] = None
    session_enabled_tools: Optional[Set[str]] = None
    vertical_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class IToolAccessController(Protocol):
    """Protocol for unified tool access control.

    Provides a single point of control for all tool access decisions,
    replacing scattered checks throughout the codebase.

    The controller applies layers in precedence order:
    Safety (L0) > Mode (L1) > Session (L2) > Vertical (L3) > Stage (L4) > Intent (L5)

    A tool is blocked if ANY layer denies it. The first layer to deny
    becomes the authoritative source of the decision.
    """

    def check_access(
        self, tool_name: str, context: Optional[ToolAccessContext] = None
    ) -> ToolAccessDecision:
        """Check if a tool is allowed in the given context.

        Args:
            tool_name: Name of the tool to check
            context: Access context (mode, stage, intent, etc.)

        Returns:
            ToolAccessDecision with result and explanation
        """
        ...

    def filter_tools(
        self, tools: List[str], context: Optional[ToolAccessContext] = None
    ) -> Tuple[List[str], List[ToolAccessDecision]]:
        """Filter a list of tools to only allowed ones.

        Args:
            tools: List of tool names to filter
            context: Access context

        Returns:
            Tuple of (allowed_tools, denial_decisions)
        """
        ...

    def get_allowed_tools(self, context: Optional[ToolAccessContext] = None) -> Set[str]:
        """Get all tools allowed in the given context.

        Args:
            context: Access context

        Returns:
            Set of allowed tool names
        """
        ...

    def explain_decision(self, tool_name: str, context: Optional[ToolAccessContext] = None) -> str:
        """Get detailed explanation for a tool access decision.

        Args:
            tool_name: Name of the tool
            context: Access context

        Returns:
            Human-readable explanation
        """
        ...
