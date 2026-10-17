"""Stable protocols for framework-orchestrator integration.

These protocols define the contract that any orchestrator implementation
must satisfy. Framework code uses these protocols, never direct attribute access.

Design Pattern: Protocol-First Architecture
- Eliminates duck-typing (hasattr/getattr calls)
- Provides type safety via Protocol structural subtyping
- Enables clean mocking for tests
- Documents the exact interface contract

Usage:
    # Type hint with protocol instead of concrete class
    def process(orchestrator: OrchestratorProtocol) -> None:
        stage = orchestrator.get_stage()  # Type-safe method call
        tools = orchestrator.get_available_tools()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
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
    from victor.framework.events import EventType
    from victor.framework.state import Stage


# =============================================================================
# Exceptions
# =============================================================================


class IncompatibleVersionError(Exception):
    """Raised when a capability version is incompatible with requirements.

    This exception is raised when invoking a capability that does not meet
    the minimum version requirement specified by the caller.

    Attributes:
        capability_name: Name of the capability
        required_version: Minimum version required
        actual_version: Actual version of the capability
    """

    def __init__(
        self,
        capability_name: str,
        required_version: str,
        actual_version: str,
    ) -> None:
        self.capability_name = capability_name
        self.required_version = required_version
        self.actual_version = actual_version
        super().__init__(
            f"Capability '{capability_name}' version {actual_version} "
            f"is incompatible with required version {required_version}"
        )


# =============================================================================
# Streaming Types
# =============================================================================


class ChunkType(str, Enum):
    """Types of streaming chunks from orchestrator.

    Maps to EventType but represents raw orchestrator output
    before conversion to framework events.
    """

    CONTENT = "content"
    """Text content from model response."""

    THINKING = "thinking"
    """Extended thinking content (reasoning mode)."""

    TOOL_CALL = "tool_call"
    """Tool invocation starting."""

    TOOL_RESULT = "tool_result"
    """Tool execution completed."""

    TOOL_ERROR = "tool_error"
    """Tool execution failed."""

    STAGE_CHANGE = "stage_change"
    """Conversation stage transition."""

    ERROR = "error"
    """General error occurred."""

    STREAM_START = "stream_start"
    """Streaming session started."""

    STREAM_END = "stream_end"
    """Streaming session ended."""


@dataclass
class OrchestratorStreamChunk:
    """Standardized streaming chunk format from orchestrator.

    This is the canonical format returned by OrchestratorProtocol.stream_chat().
    Framework code converts these to Event instances for user consumption.

    Renamed from StreamChunk to be semantically distinct from other streaming types:
    - StreamChunk (victor.providers.base): Provider-level raw streaming
    - OrchestratorStreamChunk: Orchestrator protocol with typed ChunkType
    - TypedStreamChunk: Safe typed accessor with nested StreamDelta
    - ClientStreamChunk: Protocol interface for clients (CLI/VS Code)

    Attributes:
        chunk_type: Type of this chunk (see ChunkType enum)
        content: Text content for CONTENT/THINKING chunks
        tool_name: Tool name for tool-related chunks
        tool_id: Unique identifier for tool call correlation
        tool_arguments: Arguments passed to tool (for TOOL_CALL)
        tool_result: Result from tool (for TOOL_RESULT)
        error: Error message (for ERROR/TOOL_ERROR chunks)
        old_stage: Previous stage (for STAGE_CHANGE)
        new_stage: New stage (for STAGE_CHANGE)
        metadata: Additional context-specific data
        is_final: True if this is the last chunk in the stream
    """

    chunk_type: ChunkType = ChunkType.CONTENT
    content: str = ""
    tool_name: Optional[str] = None
    tool_id: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    error: Optional[str] = None
    old_stage: Optional[str] = None
    new_stage: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = False


# =============================================================================
# Component Protocols
# =============================================================================


@runtime_checkable
class ConversationStateProtocol(Protocol):
    """Protocol for conversation state access.

    Provides read-only access to conversation state, including
    stage, tool usage, and file tracking.
    """

    def get_stage(self) -> "Stage":
        """Get current conversation stage.

        Returns:
            Current Stage enum value

        Raises:
            RuntimeError: If conversation state is not initialized
        """
        ...

    def get_tool_calls_count(self) -> int:
        """Get total tool calls made in this conversation.

        Returns:
            Non-negative count of tool calls
        """
        ...

    def get_tool_budget(self) -> int:
        """Get maximum allowed tool calls.

        Returns:
            Tool budget limit
        """
        ...

    def get_observed_files(self) -> Set[str]:
        """Get set of files observed (read) during conversation.

        Returns:
            Set of absolute file paths
        """
        ...

    def get_modified_files(self) -> Set[str]:
        """Get set of files modified (written/edited) during conversation.

        Returns:
            Set of absolute file paths
        """
        ...

    def get_iteration_count(self) -> int:
        """Get current agent loop iteration count.

        Returns:
            Non-negative iteration count
        """
        ...

    def get_max_iterations(self) -> int:
        """Get maximum allowed iterations.

        Returns:
            Max iteration limit
        """
        ...


@runtime_checkable
class ProviderProtocol(Protocol):
    """Protocol for LLM provider management.

    Provides access to current provider/model and switching capability.
    """

    @property
    def current_provider(self) -> str:
        """Get current provider name.

        Returns:
            Provider identifier (e.g., "anthropic", "openai", "ollama")
        """
        ...

    @property
    def current_model(self) -> str:
        """Get current model name.

        Returns:
            Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4")
        """
        ...

    async def switch_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        on_switch: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Switch to a different provider and/or model.

        Args:
            provider: Target provider name
            model: Optional specific model (uses provider default if None)
            on_switch: Optional callback called with (provider, model) after switch

        Raises:
            ProviderError: If provider/model is not available
        """
        ...


@runtime_checkable
class ToolsProtocol(Protocol):
    """Protocol for tool management.

    Provides access to available tools and ability to enable/disable them.
    """

    def get_available_tools(self) -> Set[str]:
        """Get names of all registered tools.

        Returns:
            Set of tool names available in the registry
        """
        ...

    def get_enabled_tools(self) -> Set[str]:
        """Get names of currently enabled tools.

        Returns:
            Set of tool names that can be used in this session
        """
        ...

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set which tools are enabled for this session.

        Args:
            tools: Set of tool names to enable

        Raises:
            ValueError: If any tool name is not in available tools
        """
        ...

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is enabled
        """
        ...


@runtime_checkable
class SystemPromptProtocol(Protocol):
    """Protocol for system prompt management.

    Provides access to and modification of the system prompt.
    """

    def get_system_prompt(self) -> str:
        """Get current system prompt.

        Returns:
            Complete system prompt string
        """
        ...

    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt (replaces existing).

        Args:
            prompt: New system prompt
        """
        ...

    def append_to_system_prompt(self, content: str) -> None:
        """Append content to existing system prompt.

        Args:
            content: Content to append (will be separated by newline)
        """
        ...


@runtime_checkable
class MessagesProtocol(Protocol):
    """Protocol for conversation messages access.

    Provides read access to conversation message history.
    """

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get conversation message history.

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        ...

    def get_message_count(self) -> int:
        """Get number of messages in conversation.

        Returns:
            Non-negative message count
        """
        ...


@runtime_checkable
class StreamingProtocol(Protocol):
    """Protocol for streaming state.

    Provides access to streaming status.
    """

    def is_streaming(self) -> bool:
        """Check if agent is currently streaming.

        Returns:
            True if a streaming operation is in progress
        """
        ...


# =============================================================================
# Composite Protocol
# =============================================================================


@runtime_checkable
class FrameworkOrchestratorProtocol(
    ConversationStateProtocol,
    ProviderProtocol,
    ToolsProtocol,
    SystemPromptProtocol,
    MessagesProtocol,
    StreamingProtocol,
    Protocol,
):
    """Complete orchestrator protocol combining all capabilities.

    Any orchestrator implementation must satisfy this protocol to work
    with the framework layer. This is the primary interface contract.

    Design Pattern: Composite Protocol (ISP-compliant)
    ================================================
    This protocol inherits from 6 focused sub-protocols. Consumers that only
    need a subset should depend on the specific sub-protocol instead:
    - ConversationStateProtocol: Stage, tool usage, file tracking
    - ProviderProtocol: Provider/model management
    - ToolsProtocol: Tool access and management
    - SystemPromptProtocol: Prompt customization
    - MessagesProtocol: Message history
    - StreamingProtocol: Streaming status

    When to use:
    - Framework code needing full access: Use this protocol
    - Framework code needing partial access: Use specific sub-protocols
    - Core modules: Use victor.core.protocols.OrchestratorProtocol for minimal interface
    - SubAgents: Use SubAgentContext for ISP-compliant minimal interface

    Note: Renamed from OrchestratorProtocol to avoid name collision with
    victor.core.protocols.OrchestratorProtocol. Use FrameworkOrchestratorProtocol
    when importing from victor.framework.protocols.

    Usage:
        async def run_agent(orch: FrameworkOrchestratorProtocol, prompt: str):
            print(f"Provider: {orch.current_provider}")
            print(f"Stage: {orch.get_stage()}")

            async for chunk in orch.stream_chat(prompt):
                if chunk.chunk_type == ChunkType.CONTENT:
                    print(chunk.content, end="")
    """

    # --- Core Chat Operations ---
    async def stream_chat(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[OrchestratorStreamChunk]:
        """Stream a chat response with standardized chunks.

        This is the primary method for interactive chat. Returns an
        async iterator yielding OrchestratorStreamChunk instances.

        Args:
            message: User message to process
            context: Optional additional context

        Yields:
            OrchestratorStreamChunk instances representing response fragments

        Raises:
            ProviderError: If LLM call fails
            ToolError: If tool execution fails critically
        """
        ...
        yield OrchestratorStreamChunk()  # Preserve the async-iterator protocol shape.

    async def chat(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Non-streaming chat that returns complete response.

        Convenience method that collects all content chunks
        into a single response string.

        Args:
            message: User message to process
            context: Optional additional context

        Returns:
            Complete response content as string

        Raises:
            ProviderError: If LLM call fails
            ToolError: If tool execution fails critically
        """
        ...

    # --- Lifecycle Methods ---
    def cancel(self) -> None:
        """Cancel any in-progress operation.

        Sets cancellation flag that streaming operations check.
        Safe to call multiple times.
        """
        ...

    def reset(self) -> None:
        """Reset conversation state.

        Clears message history, resets stage to INITIAL,
        and resets tool call counters.
        """
        ...


# =============================================================================
# Infrastructure Port Protocols (DIP Boundaries)
# =============================================================================


@runtime_checkable
class ServiceContainerPortProtocol(Protocol):
    """Protocol for orchestrators exposing their DI container safely."""

    def get_service_container(self) -> Any:
        """Get service container used by the orchestrator."""
        ...


@runtime_checkable
class CapabilityConfigScopePortProtocol(Protocol):
    """Protocol for orchestrators exposing capability-config scope identity."""

    def get_capability_config_scope_key(self) -> Optional[str]:
        """Return stable capability-config scope key for this orchestrator/session."""
        ...


@runtime_checkable
class CapabilityLoaderPortProtocol(Protocol):
    """Protocol for orchestrators exposing capability loader lifecycle."""

    def get_capability_loader(self) -> Optional[Any]:
        """Get configured capability loader if available."""
        ...

    def set_capability_loader(self, loader: Any) -> None:
        """Set capability loader instance."""
        ...

    def get_or_create_capability_loader(self) -> Any:
        """Get or lazily create capability loader instance."""
        ...


@runtime_checkable
class ObservabilityPortProtocol(Protocol):
    """Protocol for observability wiring without private attribute access."""

    def set_observability(self, value: Optional[Any]) -> None:
        """Set observability integration on orchestrator."""
        ...


# =============================================================================
# Capability Discovery Protocol
# =============================================================================


class CapabilityType(str, Enum):
    """Types of orchestrator capabilities.

    Categorizes capabilities for discovery and documentation.
    """

    TOOL = "tool"
    """Tool-related capabilities (enable, disable, budget)."""

    PROMPT = "prompt"
    """Prompt building capabilities (system prompt, hints)."""

    MODE = "mode"
    """Mode/configuration capabilities (adaptive mode, budgets)."""

    SAFETY = "safety"
    """Safety-related capabilities (patterns, middleware)."""

    RL = "rl"
    """Reinforcement learning capabilities (hooks, learners)."""

    TEAM = "team"
    """Team/multi-agent capabilities (specs, coordination)."""

    WORKFLOW = "workflow"
    """Workflow capabilities (sequences, dependencies)."""

    VERTICAL = "vertical"
    """Vertical integration capabilities (context, extensions)."""


@dataclass
class OrchestratorCapability:
    """Explicit capability declaration for orchestrator features.

    Replaces hasattr/getattr duck-typing with explicit contracts.
    Each capability declares how to interact with it.

    Versioning:
        Capabilities support semantic versioning for backward compatibility.
        Version format: "MAJOR.MINOR" (e.g., "1.0", "2.1")
        - MAJOR: Breaking changes (incompatible signature changes)
        - MINOR: Backward-compatible additions

        When invoking a capability, callers can specify minimum required version.
        Default version is "1.0" for backward compatibility.

    Attributes:
        name: Unique capability identifier
        capability_type: Category of capability
        version: Semantic version string (default "1.0")
        setter: Method name to set/configure (if settable)
        getter: Method name to get current value (if gettable)
        attribute: Direct attribute name (if attribute access)
        description: Human-readable description
        required: Whether this capability is mandatory
        deprecated: Whether this capability is deprecated
        deprecated_message: Message explaining deprecation and migration path
    """

    name: str
    capability_type: CapabilityType
    version: str = "1.0"
    setter: Optional[str] = None
    getter: Optional[str] = None
    attribute: Optional[str] = None
    description: str = ""
    required: bool = False
    deprecated: bool = False
    deprecated_message: str = ""

    def __post_init__(self) -> None:
        """Validate capability declaration."""
        # Validate access method
        if not any([self.setter, self.getter, self.attribute]):
            raise ValueError(
                f"Capability '{self.name}' must specify at least one of: "
                "setter, getter, or attribute"
            )
        # Validate version format
        if not self._is_valid_version(self.version):
            raise ValueError(
                f"Capability '{self.name}' has invalid version '{self.version}'. "
                "Expected format: 'MAJOR.MINOR' (e.g., '1.0', '2.1')"
            )

    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Validate version string format.

        Args:
            version: Version string to validate

        Returns:
            True if version is valid MAJOR.MINOR format
        """
        try:
            parts = version.split(".")
            if len(parts) != 2:
                return False
            major, minor = int(parts[0]), int(parts[1])
            return major >= 0 and minor >= 0
        except (ValueError, AttributeError):
            return False

    def is_compatible_with(self, required_version: str) -> bool:
        """Check if this capability's version is compatible with required version.

        A capability is compatible if its version >= required version.
        Comparison is done semantically (1.10 > 1.9).

        Args:
            required_version: Minimum required version

        Returns:
            True if this capability meets the version requirement
        """
        try:
            req_parts = required_version.split(".")
            cap_parts = self.version.split(".")
            req_major, req_minor = int(req_parts[0]), int(req_parts[1])
            cap_major, cap_minor = int(cap_parts[0]), int(cap_parts[1])

            # Major version must match or be greater
            if cap_major > req_major:
                return True
            if cap_major < req_major:
                return False
            # Same major, check minor
            return cap_minor >= req_minor
        except (ValueError, IndexError, AttributeError):
            return False


@runtime_checkable
class CapabilityRegistryProtocol(Protocol):
    """Protocol for capability discovery and invocation.

    Enables explicit capability checking instead of hasattr duck-typing.
    Implementations should register all capabilities at initialization.

    Versioning Support:
        All methods support optional version requirements:
        - has_capability(name, min_version="1.0") - check version compatibility
        - invoke_capability(name, *args, min_version="1.0") - version-safe invoke

    Example:
        # Instead of:
        if hasattr(orch, "set_enabled_tools") and callable(orch.set_enabled_tools):
            orch.set_enabled_tools(tools)

        # Use:
        if orch.has_capability("enabled_tools"):
            orch.invoke_capability("enabled_tools", tools)

        # With version checking:
        if orch.has_capability("enabled_tools", min_version="1.1"):
            orch.invoke_capability("enabled_tools", tools, min_version="1.1")
    """

    def get_capabilities(self) -> Dict[str, OrchestratorCapability]:
        """Get all registered capabilities.

        Returns:
            Dict mapping capability names to their declarations
        """
        ...

    def has_capability(
        self,
        name: str,
        min_version: Optional[str] = None,
    ) -> bool:
        """Check if a capability is available and meets version requirements.

        Args:
            name: Capability name to check
            min_version: Minimum required version (default: None = any version)

        Returns:
            True if capability is registered, functional, and meets version requirement
        """
        ...

    def get_capability(self, name: str) -> Optional[OrchestratorCapability]:
        """Get a specific capability declaration.

        Args:
            name: Capability name

        Returns:
            Capability declaration or None if not found
        """
        ...

    def get_capability_version(self, name: str) -> Optional[str]:
        """Get the version of a registered capability.

        Args:
            name: Capability name

        Returns:
            Version string or None if capability not found
        """
        ...

    def invoke_capability(
        self,
        name: str,
        *args: Any,
        min_version: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke a capability's setter method with optional version check.

        Args:
            name: Capability name
            *args: Positional arguments for setter
            min_version: Minimum required version (default: None = no check)
            **kwargs: Keyword arguments for setter

        Returns:
            Result from setter method

        Raises:
            KeyError: If capability not found
            TypeError: If capability has no setter
            IncompatibleVersionError: If capability version is incompatible
        """
        ...

    def get_capability_value(self, name: str) -> Any:
        """Get a capability's current value via getter or attribute.

        Args:
            name: Capability name

        Returns:
            Current value

        Raises:
            KeyError: If capability not found
            TypeError: If capability has no getter/attribute
        """
        ...

    def get_capabilities_by_type(
        self, capability_type: CapabilityType
    ) -> Dict[str, OrchestratorCapability]:
        """Get all capabilities of a specific type.

        Args:
            capability_type: Type to filter by

        Returns:
            Dict of matching capabilities
        """
        ...


# =============================================================================
# Protocol Utilities
# =============================================================================


def verify_protocol_conformance(obj: Any, protocol: type) -> Tuple[bool, List[str]]:
    """Verify that an object conforms to a protocol.

    Checks that all required methods/properties are present.
    Useful for debugging protocol conformance issues.

    Args:
        obj: Object to check
        protocol: Protocol class to check against

    Returns:
        Tuple of (conforms: bool, missing: List[str])

    Example:
        conforms, missing = verify_protocol_conformance(orch, FrameworkOrchestratorProtocol)
        if not conforms:
            raise TypeError(f"Missing protocol methods: {missing}")
    """
    missing = []

    # Get protocol's __protocol_attrs__ if available (runtime_checkable)
    if hasattr(protocol, "__protocol_attrs__"):
        for attr in protocol.__protocol_attrs__:
            if not hasattr(obj, attr):
                missing.append(attr)
    else:
        # Fall back to checking annotations
        hints = getattr(protocol, "__annotations__", {})
        for attr in hints:
            if not hasattr(obj, attr):
                missing.append(attr)

    return len(missing) == 0, missing


# =============================================================================
# Capability Registry Role Protocols (ISP decomposition)
# =============================================================================
# CapabilityRegistryProtocol is preserved unchanged for backward compatibility.
# These three focused protocols let callers declare only the surface they need.


@runtime_checkable
class CapabilityDiscoveryProtocol(Protocol):
    """Read-only enumeration of registered capabilities.

    Use this protocol when code only needs to check whether a capability
    exists or enumerate the capability registry — no value reads or mutations.
    """

    def get_capabilities(self) -> Dict[str, OrchestratorCapability]: ...

    def has_capability(
        self,
        name: str,
        min_version: Optional[str] = None,
    ) -> bool: ...

    def get_capability(self, name: str) -> Optional[OrchestratorCapability]: ...

    def get_capability_version(self, name: str) -> Optional[str]: ...

    def get_capabilities_by_type(
        self, capability_type: CapabilityType
    ) -> Dict[str, OrchestratorCapability]: ...


@runtime_checkable
class CapabilityReadProtocol(Protocol):
    """Reads the current value associated with a named capability.

    Use this protocol when code only needs to retrieve the value behind a
    capability (e.g., reading the middleware chain or safety checker object).
    """

    def get_capability_value(self, name: str) -> Any: ...


@runtime_checkable
class CapabilityMutationProtocol(Protocol):
    """Invokes (writes/triggers) a named capability.

    Use this protocol when code only needs to invoke a capability setter,
    with optional minimum-version enforcement.
    """

    def invoke_capability(
        self,
        name: str,
        *args: Any,
        min_version: Optional[str] = None,
        **kwargs: Any,
    ) -> Any: ...


# =============================================================================
# Domain Keywords Provider Protocol (OCP extension point)
# =============================================================================


@runtime_checkable
class DomainKeywordsProviderProtocol(Protocol):
    """Contributes domain-specific keyword lists for topic-switch detection.

    Verticals or plugins can implement this protocol and register it via
    ``PluginContext.register_domain_keyword_provider(provider)`` to extend
    the adaptive compaction system with domain-specific vocabulary without
    modifying core framework code.

    The provider's keywords are merged with the built-in defaults at
    ``AdaptiveCompactionThreshold`` construction time. If a domain key
    collides with a built-in, the provider's keywords win.

    Example::

        class SwiftKeywordsProvider:
            def get_domain_keywords(self) -> Dict[str, List[str]]:
                return {"swift": ["swift", "swiftui", "xcode", "cocoa", "objc"]}

            def get_provider_name(self) -> str:
                return "swift-keywords"

        # In VictorPlugin.register():
        context.register_domain_keyword_provider(SwiftKeywordsProvider())
    """

    def get_domain_keywords(self) -> Dict[str, List[str]]:
        """Return a dict mapping domain names to keyword lists."""
        ...

    def get_provider_name(self) -> str:
        """Return a short identifier for logging and debug purposes."""
        ...


# =============================================================================
# LSP & Language Registry Protocols
# =============================================================================


@runtime_checkable
class LSPPoolProtocol(Protocol):
    """Protocol for LSP connection pool.

    Framework-level abstraction so tools don't depend on the coding vertical's
    concrete LSPConnectionPool. Implementations live in verticals.
    """

    def _path_to_uri(self, path: str) -> str: ...
    def open_document(self, uri: str, content: str) -> None: ...
    def get_diagnostics(self, path: str) -> List[Dict[str, Any]]: ...
    def close_document(self, path: str) -> None: ...


@runtime_checkable
class LanguageRegistryProtocol(Protocol):
    """Protocol for language registry.

    Framework-level abstraction for language detection and plugin lookup.
    Implementations live in verticals.
    """

    def detect_language(self, path: Any) -> Optional[str]: ...
    def get(self, language: str) -> Any: ...
