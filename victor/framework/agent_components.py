"""Agent Component Decomposition - Builder/Session/Bridge pattern.

This module decomposes the Agent class into focused components following
the Single Responsibility Principle:

1. **AgentBuilder** - Factory for creating Agent instances with fluent API
2. **AgentSession** - Multi-turn conversation management
3. **AgentBridge** - CQRS and observability integration layer

Design Patterns:
- Builder Pattern: AgentBuilder for flexible agent configuration
- Strategy Pattern: Pluggable session management strategies
- Adapter Pattern: AgentBridge adapts between framework and CQRS/observability
- Observer Pattern: State change notifications

Phase 7.4: Agent class decomposition for better maintainability and testability.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field, replace
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
    Type,
    runtime_checkable,
)
from contextlib import asynccontextmanager
import uuid

from victor.agent.config import (
    FrameworkCompatibleAgentConfig,
    UnifiedAgentConfig,
    normalize_agent_config,
)
from victor.core.errors import AgentError, ConfigurationError
from victor.framework.events import AgentExecutionEvent, EventType
from victor.framework.state import State
from victor.framework.task import TaskResult
from victor.framework.tools import ToolSet, ToolsInput

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator
    from victor.core.container import ServiceContainer
    from victor.framework.service_provider import (
        EventRegistryService,
        ToolConfiguratorService,
    )
    from victor.runtime.context import ResolvedRuntimeServices
    from victor.core.verticals.base import VerticalBase


logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol defining the core Agent interface."""

    async def run(self, prompt: str, *, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        """Run a task and return the result."""
        ...

    async def stream(
        self, prompt: str, *, context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[AgentExecutionEvent]:
        """Stream events as the agent processes a task."""
        ...

    @property
    def state(self) -> State:
        """Get current agent state."""
        ...

    def get_orchestrator(self) -> "AgentOrchestrator":
        """Get the underlying orchestrator."""
        ...


@runtime_checkable
class SessionProtocol(Protocol):
    """Protocol for conversation session management."""

    async def send(self, message: str) -> TaskResult:
        """Send a message and get a response."""
        ...

    async def stream(self, message: str) -> AsyncIterator[AgentExecutionEvent]:
        """Stream a response."""
        ...

    @property
    def turn_count(self) -> int:
        """Get the number of conversation turns."""
        ...

    @property
    def history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        ...


# =============================================================================
# Builder Configuration
# =============================================================================


class BuilderPreset(str, Enum):
    """Pre-defined builder configurations.

    Note: Domain-specific presets (CODING, RESEARCH, etc.) have been removed.
    Users should install the relevant vertical package and use it directly:

        # For coding tasks:
        pip install victor-coding
        agent = await Agent.create(vertical="coding")

        # For research tasks:
        pip install victor-research
        agent = await Agent.create(vertical="research")
    """

    DEFAULT = "default"  # Balanced configuration
    MINIMAL = "minimal"  # Minimal resources
    HIGH_BUDGET = "high_budget"  # Extended resources
    AIRGAPPED = "airgapped"  # No network access


@dataclass
class AgentBuildOptions:
    """Options for building an Agent instance."""

    provider: str = "anthropic"
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: ToolsInput = None
    thinking: bool = False
    airgapped: bool = False
    profile: Optional[str] = None
    workspace: Optional[str] = None
    config: Optional[FrameworkCompatibleAgentConfig] = None
    vertical: Optional[Type["VerticalBase"]] = None
    enable_observability: bool = True
    session_id: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    state_hooks: Optional[Dict[str, Callable]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize deprecated framework config inputs to the canonical form."""
        self.config = normalize_agent_config(self.config)


# =============================================================================
# AgentBuilder
# =============================================================================


class AgentBuilder:
    """Builder for creating Agent instances with a fluent API.

    The AgentBuilder provides a flexible, chainable interface for configuring
    agents. It supports presets for common configurations and allows fine-grained
    control over all agent options.

    Phase 8.2: Enhanced with ServiceContainer integration for dependency injection.
    When a container is provided, the builder uses container-managed services for:
    - Tool configuration (ToolConfiguratorService)
    - Event handling (EventRegistryService)
    - Session management (AgentSessionService)

    Example:
        # Basic usage
        agent = await AgentBuilder().provider("anthropic").build()

        # With preset
        agent = await AgentBuilder().preset(BuilderPreset.CODING).build()

        # Fluent configuration
        agent = await (
            AgentBuilder()
            .provider("openai")
            .model("gpt-4-turbo")
            .tools(["filesystem", "git"])
            .thinking(True)
            .build()
        )

        # From existing options
        options = AgentBuildOptions(provider="anthropic", thinking=True)
        agent = await AgentBuilder.from_options(options).build()

        # With DI container (Phase 8.2)
        from victor.framework.service_provider import configure_framework_services
        container = configure_framework_services()
        agent = await AgentBuilder().with_container(container).build()
    """

    def __init__(self, container: Optional["ServiceContainer"] = None) -> None:
        """Initialize builder with default options.

        Args:
            container: Optional ServiceContainer for dependency injection.
                       When provided, services are resolved from the container.
        """
        self._options = AgentBuildOptions()
        self._presets_applied: List[BuilderPreset] = []
        self._container: Optional["ServiceContainer"] = container
        self._tool_filters: List[Any] = []

    @classmethod
    def from_options(
        cls,
        options: AgentBuildOptions,
        container: Optional["ServiceContainer"] = None,
    ) -> "AgentBuilder":
        """Create builder from existing options.

        Args:
            options: Pre-configured options
            container: Optional ServiceContainer for dependency injection

        Returns:
            Builder initialized with options
        """
        builder = cls(container=container)
        builder._options = options
        return builder

    @classmethod
    def from_container(cls, container: "ServiceContainer") -> "AgentBuilder":
        """Create builder from a ServiceContainer.

        This is the recommended way to create builders when using DI.
        The container is used to resolve all framework services.

        Args:
            container: ServiceContainer with framework services registered

        Returns:
            Builder with container integration

        Example:
            container = configure_framework_services()
            builder = AgentBuilder.from_container(container)
            agent = await builder.provider("anthropic").build()
        """
        return cls(container=container)

    # -------------------------------------------------------------------------
    # Preset Configurations
    # -------------------------------------------------------------------------

    def preset(self, preset: BuilderPreset) -> "AgentBuilder":
        """Apply a pre-defined configuration preset.

        Presets can be combined - later presets override earlier settings.

        Args:
            preset: Configuration preset to apply

        Returns:
            Self for chaining
        """
        self._presets_applied.append(preset)

        if preset == BuilderPreset.DEFAULT:
            self._options.tools = ToolSet.default()
            self._options.config = UnifiedAgentConfig.foreground()

        elif preset == BuilderPreset.MINIMAL:
            self._options.tools = ToolSet.minimal()
            self._options.config = UnifiedAgentConfig.minimal()
            self._options.enable_observability = False

        elif preset == BuilderPreset.HIGH_BUDGET:
            self._options.config = UnifiedAgentConfig.high_budget()

        elif preset == BuilderPreset.AIRGAPPED:
            self._options.airgapped = True
            self._options.tools = ToolSet.airgapped()
            self._options.config = UnifiedAgentConfig.airgapped()

        else:
            raise ValueError(f"Unknown preset: {preset}")

        return self

    # -------------------------------------------------------------------------
    # Provider Configuration
    # -------------------------------------------------------------------------

    def provider(self, name: str) -> "AgentBuilder":
        """Set the LLM provider.

        Args:
            name: Provider name (anthropic, openai, ollama, google, etc.)

        Returns:
            Self for chaining
        """
        self._options.provider = name
        return self

    def model(self, name: str) -> "AgentBuilder":
        """Set the model identifier.

        Args:
            name: Model identifier

        Returns:
            Self for chaining
        """
        self._options.model = name
        return self

    def temperature(self, value: float) -> "AgentBuilder":
        """Set sampling temperature.

        Args:
            value: Temperature (0.0 to 1.0)

        Returns:
            Self for chaining
        """
        if not 0.0 <= value <= 2.0:
            raise ConfigurationError(f"Temperature must be between 0.0 and 2.0, got {value}")
        self._options.temperature = value
        return self

    def max_tokens(self, value: int) -> "AgentBuilder":
        """Set maximum tokens to generate.

        Args:
            value: Maximum tokens

        Returns:
            Self for chaining
        """
        if value < 1:
            raise ConfigurationError(f"max_tokens must be positive, got {value}")
        self._options.max_tokens = value
        return self

    # -------------------------------------------------------------------------
    # Tool Configuration
    # -------------------------------------------------------------------------

    def tools(self, tools: ToolsInput) -> "AgentBuilder":
        """Set available tools.

        Args:
            tools: ToolSet, list of categories, or None

        Returns:
            Self for chaining
        """
        self._options.tools = tools
        return self

    def default_tools(self) -> "AgentBuilder":
        """Use default tool set.

        Returns:
            Self for chaining
        """
        self._options.tools = ToolSet.default()
        return self

    def minimal_tools(self) -> "AgentBuilder":
        """Use minimal tool set.

        Returns:
            Self for chaining
        """
        self._options.tools = ToolSet.minimal()
        return self

    def full_tools(self) -> "AgentBuilder":
        """Use full tool set.

        Returns:
            Self for chaining
        """
        self._options.tools = ToolSet.full()
        return self

    def airgapped_tools(self) -> "AgentBuilder":
        """Use airgapped tool set (no network).

        Returns:
            Self for chaining
        """
        self._options.tools = ToolSet.airgapped()
        return self

    def add_tool_filter(self, tool_filter: Any) -> "AgentBuilder":
        """Add a tool filter for filtering tools during build.

        Tool filters are applied when configuring tools on the orchestrator.
        This enables runtime tool filtering based on security, cost, or custom criteria.

        Args:
            tool_filter: A filter implementing ToolFilterProtocol

        Returns:
            Self for chaining

        Example:
            from victor.framework.tool_config import AirgappedFilter, CostTierFilter
            builder = (
                AgentBuilder()
                .add_tool_filter(AirgappedFilter())
                .add_tool_filter(CostTierFilter(max_tier="MEDIUM"))
                .build()
            )
        """
        self._tool_filters.append(tool_filter)
        return self

    # -------------------------------------------------------------------------
    # DI Container Configuration (Phase 8.2)
    # -------------------------------------------------------------------------

    def with_container(self, container: "ServiceContainer") -> "AgentBuilder":
        """Set the ServiceContainer for dependency injection.

        When a container is set, the builder uses container-managed services
        for tool configuration, event handling, and session management.

        Args:
            container: ServiceContainer with framework services registered

        Returns:
            Self for chaining

        Example:
            from victor.framework.service_provider import configure_framework_services
            container = configure_framework_services()
            agent = await AgentBuilder().with_container(container).build()
        """
        self._container = container
        return self

    @property
    def has_container(self) -> bool:
        """Check if a container is configured.

        Returns:
            True if a ServiceContainer is set
        """
        return self._container is not None

    def _get_tool_configurator(self) -> Optional["ToolConfiguratorService"]:
        """Get ToolConfigurator from container if available.

        Returns:
            ToolConfigurator service or None
        """
        if self._container is None:
            return None
        try:
            from victor.framework.service_provider import ToolConfiguratorService

            return self._container.get(ToolConfiguratorService)
        except Exception as e:
            logger.debug(f"Could not get ToolConfigurator from container: {e}")
            return None

    def _get_event_registry(self) -> Optional["EventRegistryService"]:
        """Get EventRegistry from container if available.

        Returns:
            EventRegistry service or None
        """
        if self._container is None:
            return None
        try:
            from victor.framework.service_provider import EventRegistryService

            return self._container.get(EventRegistryService)
        except Exception as e:
            logger.debug(f"Could not get EventRegistry from container: {e}")
            return None

    # -------------------------------------------------------------------------
    # Feature Configuration
    # -------------------------------------------------------------------------

    def thinking(self, enabled: bool = True) -> "AgentBuilder":
        """Enable extended thinking mode.

        Args:
            enabled: Whether to enable thinking

        Returns:
            Self for chaining
        """
        self._options.thinking = enabled
        return self

    def airgapped(self, enabled: bool = True) -> "AgentBuilder":
        """Enable airgapped mode (no network).

        Args:
            enabled: Whether to enable airgapped mode

        Returns:
            Self for chaining
        """
        self._options.airgapped = enabled
        return self

    def profile(self, name: str) -> "AgentBuilder":
        """Use a configuration profile.

        Args:
            name: Profile name from ~/.victor/profiles.yaml

        Returns:
            Self for chaining
        """
        self._options.profile = name
        return self

    def workspace(self, path: str) -> "AgentBuilder":
        """Set working directory for file operations.

        Args:
            path: Working directory path

        Returns:
            Self for chaining
        """
        self._options.workspace = path
        return self

    def config(self, config: FrameworkCompatibleAgentConfig) -> "AgentBuilder":
        """Set advanced configuration.

        Args:
            config: AgentConfig (deprecated) or UnifiedAgentConfig instance.

        Returns:
            Self for chaining
        """
        self._options.config = normalize_agent_config(config)
        return self

    def vertical(self, vertical_class: Type["VerticalBase"]) -> "AgentBuilder":
        """Use a domain-specific vertical.

        Args:
            vertical_class: Vertical class or name string

        Returns:
            Self for chaining
        """
        self._options.vertical = vertical_class
        return self

    def system_prompt(self, prompt: str) -> "AgentBuilder":
        """Set custom system prompt.

        Args:
            prompt: Custom system prompt

        Returns:
            Self for chaining
        """
        self._options.custom_system_prompt = prompt
        return self

    # -------------------------------------------------------------------------
    # Observability Configuration
    # -------------------------------------------------------------------------

    def with_observability(self, enabled: bool = True) -> "AgentBuilder":
        """Enable observability integration.

        Args:
            enabled: Whether to enable

        Returns:
            Self for chaining
        """
        self._options.enable_observability = enabled
        return self

    def session_id(self, session_id: str) -> "AgentBuilder":
        """Set session ID for event correlation.

        Args:
            session_id: Session identifier

        Returns:
            Self for chaining
        """
        self._options.session_id = session_id
        return self

    # -------------------------------------------------------------------------
    # State Hooks
    # -------------------------------------------------------------------------

    def on_enter_stage(self, callback: Callable[[str, Dict], None]) -> "AgentBuilder":
        """Register callback for stage entry.

        Args:
            callback: Function called on stage entry

        Returns:
            Self for chaining
        """
        if self._options.state_hooks is None:
            self._options.state_hooks = {}
        self._options.state_hooks["on_enter"] = callback
        return self

    def on_exit_stage(self, callback: Callable[[str, Dict], None]) -> "AgentBuilder":
        """Register callback for stage exit.

        Args:
            callback: Function called on stage exit

        Returns:
            Self for chaining
        """
        if self._options.state_hooks is None:
            self._options.state_hooks = {}
        self._options.state_hooks["on_exit"] = callback
        return self

    def on_transition(self, callback: Callable[[str, str, Dict], None]) -> "AgentBuilder":
        """Register callback for state transitions.

        Args:
            callback: Function called on transition

        Returns:
            Self for chaining
        """
        if self._options.state_hooks is None:
            self._options.state_hooks = {}
        self._options.state_hooks["on_transition"] = callback
        return self

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def metadata(self, key: str, value: Any) -> "AgentBuilder":
        """Add metadata to the agent.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for chaining
        """
        self._options.metadata[key] = value
        return self

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def get_options(self) -> AgentBuildOptions:
        """Get current build options.

        Returns:
            Copy of build options
        """
        return AgentBuildOptions(
            provider=self._options.provider,
            model=self._options.model,
            temperature=self._options.temperature,
            max_tokens=self._options.max_tokens,
            tools=self._options.tools,
            thinking=self._options.thinking,
            airgapped=self._options.airgapped,
            profile=self._options.profile,
            workspace=self._options.workspace,
            config=self._options.config,
            vertical=self._options.vertical,
            enable_observability=self._options.enable_observability,
            session_id=self._options.session_id,
            custom_system_prompt=self._options.custom_system_prompt,
            state_hooks=self._options.state_hooks,
            metadata=dict(self._options.metadata),
        )

    async def build(self) -> "Agent":
        """Build and return the configured Agent.

        When a container is available (via with_container() or from_container()),
        the build process uses container-managed services for:
        - Tool configuration (ToolConfiguratorService with filters)
        - Event handling (EventRegistryService for event conversion)

        Returns:
            Configured Agent instance

        Raises:
            AgentError: If build fails
        """
        # Import here to avoid circular imports
        from victor.framework.agent import Agent

        # Create agent using existing create() method with our options
        agent = await Agent.create(
            provider=self._options.provider,
            model=self._options.model,
            temperature=self._options.temperature,
            max_tokens=self._options.max_tokens,
            tools=self._options.tools,
            thinking=self._options.thinking,
            airgapped=self._options.airgapped,
            profile=self._options.profile,
            workspace=self._options.workspace,
            config=self._options.config,
            vertical=self._options.vertical,
            enable_observability=self._options.enable_observability,
            session_id=self._options.session_id,
        )

        # Apply custom system prompt if set
        if self._options.custom_system_prompt:
            orchestrator = agent.get_orchestrator()
            if hasattr(orchestrator, "prompt_builder"):
                if hasattr(orchestrator.prompt_builder, "set_custom_prompt"):
                    orchestrator.prompt_builder.set_custom_prompt(
                        self._options.custom_system_prompt
                    )

        # Apply tool filters via container-managed ToolConfigurator (Phase 8.2)
        if self._tool_filters:
            await self._apply_tool_filters(agent)

        # Store metadata and container reference
        agent._builder_metadata = self._options.metadata
        agent._presets_applied = self._presets_applied.copy()
        if self._container is not None:
            agent._container = self._container

        return agent

    async def _apply_tool_filters(self, agent: "Agent") -> None:
        """Apply tool filters to agent using container services if available.

        This method uses the ToolConfiguratorService from the container when
        available, falling back to direct filter application otherwise.

        Args:
            agent: The agent to configure
        """
        orchestrator = agent.get_orchestrator()

        # Try container-managed configurator first
        configurator = self._get_tool_configurator()
        if configurator is not None:
            # Use container-managed configurator
            for tool_filter in self._tool_filters:
                configurator.add_filter(tool_filter)

            # Re-configure tools with filters applied
            if self._options.tools is not None:
                if isinstance(self._options.tools, ToolSet):
                    configurator.configure_from_toolset(orchestrator, self._options.tools)
                else:
                    from victor.framework.tool_config import ToolConfigMode

                    configurator.configure(
                        orchestrator, set(self._options.tools), ToolConfigMode.REPLACE
                    )
            logger.debug("Applied %d tool filters via container", len(self._tool_filters))
        else:
            # Fallback: Direct filter application
            from victor.framework.tool_config import get_tool_configurator

            configurator = get_tool_configurator()
            for tool_filter in self._tool_filters:
                configurator.add_filter(tool_filter)

            if self._options.tools is not None:
                if isinstance(self._options.tools, ToolSet):
                    configurator.configure_from_toolset(orchestrator, self._options.tools)
                else:
                    from victor.framework.tool_config import ToolConfigMode

                    configurator.configure(
                        orchestrator, set(self._options.tools), ToolConfigMode.REPLACE
                    )
            logger.debug("Applied %d tool filters via fallback", len(self._tool_filters))


# =============================================================================
# AgentSession
# =============================================================================


class SessionState(str, Enum):
    """State of an AgentSession."""

    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


@dataclass
class SessionContext:
    """Context information for a session."""

    session_id: str
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionMetrics:
    """Metrics collected during a session.

    Phase 8.3: Enhanced lifecycle management with metrics tracking.
    """

    total_turns: int = 0
    total_duration: float = 0.0
    total_tool_calls: int = 0
    successful_turns: int = 0
    failed_turns: int = 0
    average_turn_duration: float = 0.0

    def update(self, turn_data: Dict[str, Any]) -> None:
        """Update metrics with turn data."""
        self.total_turns += 1
        self.total_duration += turn_data.get("duration", 0.0)
        self.total_tool_calls += turn_data.get("tool_count", 0)

        if turn_data.get("success", True):
            self.successful_turns += 1
        else:
            self.failed_turns += 1

        if self.total_turns > 0:
            self.average_turn_duration = self.total_duration / self.total_turns

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_turns": self.total_turns,
            "total_duration": self.total_duration,
            "total_tool_calls": self.total_tool_calls,
            "successful_turns": self.successful_turns,
            "failed_turns": self.failed_turns,
            "average_turn_duration": self.average_turn_duration,
        }


@dataclass
class SessionLifecycleHooks:
    """Lifecycle hooks for session events.

    Phase 8.3: Enables external observation of session lifecycle.
    """

    on_start: Optional[Callable[["AgentSession"], None]] = None
    on_turn_start: Optional[Callable[["AgentSession", str], None]] = None
    on_turn_end: Optional[Callable[["AgentSession", TaskResult], None]] = None
    on_error: Optional[Callable[["AgentSession", Exception], None]] = None
    on_pause: Optional[Callable[["AgentSession"], None]] = None
    on_resume: Optional[Callable[["AgentSession"], None]] = None
    on_close: Optional[Callable[["AgentSession", SessionMetrics], Any]] = None


class AgentSession:
    """Enhanced multi-turn conversation session.

    AgentSession provides sophisticated conversation management with:
    - Turn tracking and history
    - Context injection
    - Pause/resume support
    - Session state management
    - CQRS event correlation
    - Lifecycle hooks (Phase 8.3)
    - Metrics tracking (Phase 8.3)
    - Container integration for scoped services (Phase 8.3)

    Example:
        session = AgentSession(agent, "Let's analyze the code")

        # Basic conversation
        response = await session.send("What patterns do you see?")
        print(response.content)

        # With context
        response = await session.send_with_context(
            "Fix this bug",
            context={"error": "IndexError at line 42"}
        )

        # Streaming
        async for event in session.stream("Refactor the function"):
            print(event.content, end="")

        # Access history
        print(f"Turns: {session.turn_count}")
        for msg in session.history:
            print(f"{msg['role']}: {msg['content'][:50]}...")

        # With lifecycle hooks (Phase 8.3)
        hooks = SessionLifecycleHooks(
            on_start=lambda s: print(f"Session {s.session_id} started"),
            on_close=lambda s, m: print(f"Completed {m.total_turns} turns"),
        )
        session = AgentSession(agent, "Analyze code", hooks=hooks)

        # Access metrics (Phase 8.3)
        print(session.metrics.to_dict())
    """

    def __init__(
        self,
        agent: "Agent",
        initial_prompt: Optional[str] = None,
        *,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        hooks: Optional[SessionLifecycleHooks] = None,
        container: Optional["ServiceContainer"] = None,
    ) -> None:
        """Initialize session.

        Args:
            agent: Agent instance
            initial_prompt: Optional first message in conversation. When omitted,
                the first ``send()`` or ``stream()`` message becomes the opening turn.
            session_id: Optional session ID
            metadata: Optional session metadata
            hooks: Optional lifecycle hooks (Phase 8.3)
            container: Optional ServiceContainer for scoped services (Phase 8.3)
        """
        import time

        self._agent = agent
        self._initial_prompt = initial_prompt
        self._turn_count = 0
        self._initialized = False
        self._state = SessionState.IDLE

        self._context = SessionContext(
            session_id=session_id or str(uuid.uuid4()),
            created_at=time.time(),
            metadata=metadata or {},
        )

        self._history: List[Dict[str, str]] = []
        # Turn history for this session
        self._turns: List[Dict[str, Any]] = []
        self._runtime_session_started = False
        self._runtime_session_closed = False
        self._runtime_services: Optional["ResolvedRuntimeServices"] = None

        # Phase 8.3: Lifecycle management
        self._hooks = hooks or SessionLifecycleHooks()
        self._metrics = SessionMetrics()
        self._container = container
        self._scope: Optional[Any] = None  # Scoped container for session

        # Create scoped container if container provided
        if self._container is not None:
            try:
                self._scope = self._container.create_scope()
            except Exception as e:
                logger.debug(f"Could not create scoped container: {e}")

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._context.session_id

    @property
    def state(self) -> SessionState:
        """Get session state."""
        return self._state

    @property
    def turn_count(self) -> int:
        """Get number of conversation turns."""
        return self._turn_count

    @property
    def history(self) -> List[Dict[str, str]]:
        """Get conversation history.

        Returns:
            List of message dicts with role and content
        """
        if self._history:
            return [dict(item) for item in self._history]

        messages = getattr(self._agent._orchestrator, "messages", [])
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    @property
    def turns(self) -> List[Dict[str, Any]]:
        """Get detailed turn history.

        Returns:
            List of turn records with prompt, response, and metadata
        """
        return self._turns.copy()

    @property
    def agent(self) -> "Agent":
        """Get the underlying Agent."""
        return self._agent

    @property
    def metrics(self) -> SessionMetrics:
        """Get session metrics (Phase 8.3).

        Returns:
            SessionMetrics with turn statistics
        """
        return self._metrics

    @property
    def hooks(self) -> SessionLifecycleHooks:
        """Get lifecycle hooks (Phase 8.3).

        Returns:
            SessionLifecycleHooks instance
        """
        return self._hooks

    @property
    def scope(self) -> Optional[Any]:
        """Get scoped container (Phase 8.3).

        Returns:
            Scoped container if container was provided, None otherwise
        """
        return self._scope

    async def initialize(self) -> "AgentSession":
        """Bind canonical runtime session services before the first turn."""
        await self._ensure_runtime_session()
        return self

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    async def send(self, message: str) -> TaskResult:
        """Send a message and get a response.

        Args:
            message: User message

        Returns:
            TaskResult with response
        """
        return await self.send_with_context(message)

    async def send_with_context(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """Send a message with optional context.

        Args:
            message: User message
            context: Optional context dict

        Returns:
            TaskResult with response
        """
        import time

        if self._state == SessionState.CLOSED:
            raise AgentError("Session is closed")

        await self._ensure_runtime_session()

        # First turn uses initial prompt and triggers on_start hook
        if not self._initialized:
            self._initialized = True
            prompt = self._initial_prompt if self._initial_prompt is not None else message
            self._state = SessionState.ACTIVE

            # Phase 8.3: Invoke on_start lifecycle hook
            if self._hooks.on_start:
                try:
                    self._hooks.on_start(self)
                except Exception as e:
                    logger.debug(f"on_start hook error: {e}")
        else:
            prompt = message

        self._turn_count += 1
        turn_start = time.time()

        # Phase 8.3: Invoke on_turn_start lifecycle hook
        if self._hooks.on_turn_start:
            try:
                self._hooks.on_turn_start(self, prompt)
            except Exception as e:
                logger.debug(f"on_turn_start hook error: {e}")

        try:
            result = await self._agent.run(prompt, context=context)
        except Exception as e:
            # Phase 8.3: Invoke on_error lifecycle hook
            if self._hooks.on_error:
                try:
                    self._hooks.on_error(self, e)
                except Exception as hook_error:
                    logger.debug(f"on_error hook error: {hook_error}")
            raise

        # Record turn
        turn_data = {
            "turn": self._turn_count,
            "prompt": prompt,
            "response": result.content,
            "success": result.success,
            "tool_count": result.tool_count,
            "duration": time.time() - turn_start,
            "context": context,
        }
        self._turns.append(turn_data)
        self._record_history_turn(prompt, result.content)

        # Phase 8.3: Update metrics
        self._metrics.update(turn_data)

        # Phase 8.3: Invoke on_turn_end lifecycle hook
        if self._hooks.on_turn_end:
            try:
                self._hooks.on_turn_end(self, result)
            except Exception as e:
                logger.debug(f"on_turn_end hook error: {e}")

        return result

    async def stream(self, message: str) -> AsyncIterator[AgentExecutionEvent]:
        """Stream a response.

        Args:
            message: User message

        Yields:
            AgentExecutionEvent objects
        """
        async for event in self.stream_with_context(message):
            yield event

    async def stream_with_context(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[AgentExecutionEvent]:
        """Stream a response with optional context.

        Args:
            message: User message
            context: Optional context dict

        Yields:
            AgentExecutionEvent objects
        """
        import time

        if self._state == SessionState.CLOSED:
            raise AgentError("Session is closed")

        await self._ensure_runtime_session()

        # First turn uses initial prompt and triggers on_start hook
        if not self._initialized:
            self._initialized = True
            prompt = self._initial_prompt if self._initial_prompt is not None else message
            self._state = SessionState.ACTIVE

            # Phase 8.3: Invoke on_start lifecycle hook
            if self._hooks.on_start:
                try:
                    self._hooks.on_start(self)
                except Exception as e:
                    logger.debug(f"on_start hook error: {e}")
        else:
            prompt = message

        self._turn_count += 1
        turn_start = time.time()

        # Phase 8.3: Invoke on_turn_start lifecycle hook
        if self._hooks.on_turn_start:
            try:
                self._hooks.on_turn_start(self, prompt)
            except Exception as e:
                logger.debug(f"on_turn_start hook error: {e}")

        content_parts: List[str] = []
        tool_count = 0
        success = True

        try:
            async for event in self._agent.stream(prompt, context=context):
                if event.type == EventType.CONTENT:
                    content_parts.append(event.content)
                elif event.type == EventType.TOOL_CALL:
                    tool_count += 1
                elif event.type == EventType.ERROR:
                    success = False

                yield event
        except Exception as e:
            success = False
            # Phase 8.3: Invoke on_error lifecycle hook
            if self._hooks.on_error:
                try:
                    self._hooks.on_error(self, e)
                except Exception as hook_error:
                    logger.debug(f"on_error hook error: {hook_error}")
            raise

        # Record turn
        turn_data = {
            "turn": self._turn_count,
            "prompt": prompt,
            "response": "".join(content_parts),
            "success": success,
            "tool_count": tool_count,
            "duration": time.time() - turn_start,
            "context": context,
        }
        self._turns.append(turn_data)
        self._record_history_turn(prompt, "".join(content_parts))

        # Phase 8.3: Update metrics
        self._metrics.update(turn_data)

        # Phase 8.3: Invoke on_turn_end lifecycle hook
        if self._hooks.on_turn_end:
            try:
                # Create a TaskResult for the hook
                # Use tool_calls list to track tool count (tool_count is a property)
                result = TaskResult(
                    content="".join(content_parts),
                    success=success,
                    tool_calls=[{"name": f"tool_{i}"} for i in range(tool_count)],
                )
                self._hooks.on_turn_end(self, result)
            except Exception as e:
                logger.debug(f"on_turn_end hook error: {e}")

    # -------------------------------------------------------------------------
    # Session Control
    # -------------------------------------------------------------------------

    def pause(self) -> None:
        """Pause the session.

        Pausing a session suspends activity and invokes the on_pause
        lifecycle hook if registered.
        """
        if self._state == SessionState.ACTIVE:
            self._state = SessionState.PAUSED

            # Phase 8.3: Invoke on_pause lifecycle hook
            if self._hooks.on_pause:
                try:
                    self._hooks.on_pause(self)
                except Exception as e:
                    logger.debug(f"on_pause hook error: {e}")

    def resume(self) -> None:
        """Resume a paused session.

        Resuming a session reactivates it and invokes the on_resume
        lifecycle hook if registered.
        """
        if self._state == SessionState.PAUSED:
            self._state = SessionState.ACTIVE

            # Phase 8.3: Invoke on_resume lifecycle hook
            if self._hooks.on_resume:
                try:
                    self._hooks.on_resume(self)
                except Exception as e:
                    logger.debug(f"on_resume hook error: {e}")

    async def close(self) -> None:
        """Close the session.

        Closing a session:
        1. Invokes the on_close lifecycle hook with final metrics
        2. Disposes the scoped container if present
        3. Sets state to CLOSED
        """
        if self._state == SessionState.CLOSED:
            return  # Already closed

        self._state = SessionState.CLOSED

        if (
            self._runtime_session_started
            and not self._runtime_session_closed
            and self._runtime_services is not None
            and self._runtime_services.session is not None
        ):
            try:
                await self._runtime_services.session.close_session(self._context.session_id)
                self._runtime_session_closed = True
            except Exception as e:
                logger.debug(f"Error closing session service runtime: {e}")

        # Phase 8.3: Invoke on_close lifecycle hook with final metrics
        if self._hooks.on_close:
            try:
                close_result = self._hooks.on_close(self, self._metrics)
                if inspect.isawaitable(close_result):
                    await close_result
            except Exception as e:
                logger.debug(f"on_close hook error: {e}")

        # Phase 8.3: Dispose scoped container
        if self._scope is not None:
            try:
                self._scope.dispose()
            except Exception as e:
                logger.debug(f"Error disposing scoped container: {e}")
            self._scope = None

    async def reset(self) -> None:
        """Reset the session to initial state.

        Resetting a session:
        1. Closes the session (invoking on_close hook)
        2. Resets the agent
        3. Resets all session state including metrics
        4. Creates a new scoped container if container provided
        """
        # First close the session to invoke on_close and cleanup
        if self._state != SessionState.CLOSED:
            await self.close()

        # Reset agent state
        await self._agent.reset()

        # Reset session state
        self._turn_count = 0
        self._initialized = False
        self._history.clear()
        self._turns.clear()
        self._metrics = SessionMetrics()  # Reset metrics
        self._state = SessionState.IDLE
        self._runtime_session_started = False
        self._runtime_session_closed = False
        self._runtime_services = None

        # Phase 8.3: Recreate scoped container if container provided
        if self._container is not None:
            try:
                self._scope = self._container.create_scope()
            except Exception as e:
                logger.debug(f"Could not create scoped container on reset: {e}")

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "AgentSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"AgentSession(id={self._context.session_id[:8]}..., "
            f"turns={self._turn_count}, state={self._state.value})"
        )

    def _record_history_turn(self, prompt: str, response: str) -> None:
        """Append the executed turn to the explicit session history."""
        self._history.append({"role": "user", "content": prompt})
        self._history.append({"role": "assistant", "content": response})

    def _resolve_runtime_services(self) -> "ResolvedRuntimeServices":
        """Resolve the canonical runtime service bundle through the shared helper."""
        from victor.runtime.context import resolve_runtime_services
        from victor.runtime.context import ResolvedRuntimeServices

        orchestrator = getattr(self._agent, "_orchestrator", None)
        if orchestrator is None:
            return ResolvedRuntimeServices()

        return resolve_runtime_services(
            orchestrator,
            getattr(self._agent, "execution_context", None),
        )

    async def _ensure_runtime_session(self) -> None:
        """Create the canonical runtime session once and reset conversation state."""
        if self._runtime_session_started:
            return

        self._runtime_services = self._resolve_runtime_services()
        session_service = self._runtime_services.session

        if session_service is not None:
            metadata = dict(self._context.metadata)
            try:
                self._context.session_id = await session_service.create_session(metadata=metadata)
            except Exception as e:
                logger.debug(f"Could not create session service runtime: {e}")
                self._runtime_services = replace(self._runtime_services, session=None)

        reset_target = self._runtime_services.chat
        if reset_target is not None and hasattr(reset_target, "reset_conversation"):
            try:
                reset_target.reset_conversation()
            except Exception as e:
                logger.debug(f"Could not reset chat service conversation: {e}")
        else:
            orchestrator = getattr(self._agent, "_orchestrator", None)
            if orchestrator is not None and hasattr(orchestrator, "reset_conversation"):
                try:
                    orchestrator.reset_conversation()
                except Exception as e:
                    logger.debug(f"Could not reset orchestrator conversation: {e}")

        self._runtime_session_started = True
        self._runtime_session_closed = False


# =============================================================================
# AgentBridge
# =============================================================================


@dataclass
class BridgeConfiguration:
    """Configuration for AgentBridge."""

    enable_observability: bool = True
    enable_metrics: bool = True
    auto_forward_events: bool = True


class AgentBridge:
    """Bridge between Agent and external systems (Observability).

    The AgentBridge provides a clean integration layer that:
    - Forwards events to observability infrastructure
    - Manages session correlation
    - Handles cleanup on disconnect

    Example:
        # Create and connect
        bridge = AgentBridge(agent, BridgeConfiguration())
        await bridge.connect()

        # Disconnect when done
        await bridge.disconnect()

        # Or use as context manager
        async with AgentBridge(agent, config) as bridge:
            result = await agent.run("Do something")
    """

    def __init__(
        self,
        agent: "Agent",
        config: Optional[BridgeConfiguration] = None,
    ) -> None:
        """Initialize bridge.

        Args:
            agent: Agent to bridge
            config: Bridge configuration
        """
        self._agent = agent
        self._config = config or BridgeConfiguration()
        self._connected = False
        self._session_id: Optional[str] = None

    @property
    def connected(self) -> bool:
        """Check if bridge is connected."""
        return self._connected

    @property
    def session_id(self) -> Optional[str]:
        """Get session ID."""
        return self._session_id

    @property
    def agent(self) -> "Agent":
        """Get bridged agent."""
        return self._agent

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self, session_id: Optional[str] = None) -> str:
        """Connect the bridge.

        Args:
            session_id: Optional session ID for correlation

        Returns:
            Session ID

        Raises:
            AgentError: If already connected
        """
        if self._connected:
            raise AgentError("Bridge already connected")

        self._session_id = session_id or str(uuid.uuid4())
        self._connected = True
        return self._session_id

    async def disconnect(self) -> None:
        """Disconnect the bridge."""
        if not self._connected:
            return

        self._session_id = None
        self._connected = False

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "AgentBridge":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"AgentBridge(agent={self._agent}, status={status})"


# =============================================================================
# Factory Functions
# =============================================================================


def create_builder(container: Optional["ServiceContainer"] = None) -> AgentBuilder:
    """Create a new AgentBuilder.

    Args:
        container: Optional ServiceContainer for dependency injection.
                   When provided, the builder uses container-managed services.

    Returns:
        Fresh AgentBuilder instance

    Example:
        # Without container
        builder = create_builder()
        agent = await builder.provider("anthropic").build()

        # With container (recommended for DI)
        from victor.framework.service_provider import configure_framework_services
        container = configure_framework_services()
        builder = create_builder(container)
        agent = await builder.provider("anthropic").build()
    """
    return AgentBuilder(container=container)


@asynccontextmanager
async def create_session(
    agent: "Agent",
    initial_prompt: Optional[str] = None,
    **kwargs: Any,
) -> AsyncIterator[AgentSession]:
    """Create and manage an AgentSession as a context manager.

    Args:
        agent: Agent instance
        initial_prompt: First message
        **kwargs: Additional session options

    Yields:
        AgentSession instance

    Example:
        async with create_session(agent, "Let's analyze") as session:
            response = await session.send("What do you see?")
    """
    session = AgentSession(agent, initial_prompt, **kwargs)
    try:
        yield session
    finally:
        await session.close()


@asynccontextmanager
async def create_bridge(
    agent: "Agent",
    config: Optional[BridgeConfiguration] = None,
    session_id: Optional[str] = None,
) -> AsyncIterator[AgentBridge]:
    """Create and manage an AgentBridge as a context manager.

    Args:
        agent: Agent to bridge
        config: Bridge configuration
        session_id: Optional session ID

    Yields:
        Connected AgentBridge instance

    Example:
        async with create_bridge(agent, BridgeConfiguration()) as bridge:
            result = await agent.run("Do something")
            metrics = await bridge.get_metrics()
    """
    bridge = AgentBridge(agent, config)
    try:
        await bridge.connect(session_id=session_id)
        yield bridge
    finally:
        await bridge.disconnect()


# =============================================================================
# Type alias for Agent (for forward reference resolution)
# =============================================================================

# Import Agent at runtime to avoid circular imports
# This is resolved when the module is fully loaded
Agent = "Agent"  # type: ignore[misc]


def _resolve_agent_type() -> None:
    """Resolve Agent type at module load time."""
    global Agent
    from victor.framework.agent import Agent as RealAgent

    Agent = RealAgent  # type: ignore[misc]


# Resolve on import
try:
    _resolve_agent_type()
except ImportError:
    pass  # Will be resolved later
