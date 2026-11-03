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

"""Agent orchestrator for managing conversations and tool execution.

Architecture: Facade Pattern
============================
AgentOrchestrator acts as a facade coordinating several extracted components:

Extracted Components (separate modules):
- ConversationController: Message history, context tracking, stage management
- ToolPipeline: Tool validation, execution coordination, budget enforcement
- StreamingController: Session lifecycle, metrics collection, cancellation
- StreamingCoordinator: Response processing, chunk aggregation, event dispatch (NEW)
- LifecycleManager: Session lifecycle and resource cleanup coordination (NEW)
- TaskAnalyzer: Unified facade for complexity/task/intent classification
- ToolSelector: Semantic and keyword-based tool selection
- ToolRegistrar: Tool registration, plugins, MCP integration (NEW)

Remaining Orchestrator Responsibilities:
- High-level chat flow coordination
- Configuration loading and validation
- Post-switch hooks (prompt rebuilding, tracker updates)

Recently Integrated:
- ProviderManager: Provider initialization, switching, health checks (NEW)
- StreamingCoordinator: Simple response processing for streaming use cases (NEW)
- LifecycleManager: Session reset, recovery, graceful shutdown, resource cleanup (NEW)

Note: Keep orchestrator as a thin facade. New logic should go into
appropriate extracted components, not added here.

Recent Refactoring (December 2025 - January 2025):
- Extracted ToolRegistrar to own tool registration, plugin bootstrap,
    _setup_mcp_integration, _plan_tools, and _goal_hints_for_message
- Added ProviderHealthChecker for proactive health monitoring
- Added ResilienceMetricsExporter for dashboard integration
- Added classification-aware tool selection in SemanticToolSelector
- Added StreamingCoordinator for response processing (January 2025)
- Added LifecycleManager for lifecycle management (January 2025)
"""

import asyncio
import logging
import threading
from typing import Any, AsyncIterator, Dict, List, Optional, Set, TYPE_CHECKING

from rich.console import Console

# Coordinators (Phase 2 refactoring - being integrated)
# NOTE: These are runtime imports, not type-checking only
from victor.agent.runtime.context import RuntimeIdentity, require_idle_turn
from victor.core.async_utils import aclosing_if_supported as _aclosing
from victor.agent.services.metrics_service import AgentMetricsService  # noqa: F401

if TYPE_CHECKING:
    # Type-only imports (created by factory, only used for type hints)
    from victor.agent.orchestrator_integration import OrchestratorIntegration
    from victor.agent.protocols import ToolAccessContext

    # Factory-created components (type hints only)
    from victor.runtime.context import RuntimeExecutionContext
    from victor.agent.safety import SafetyChecker
    from victor.agent.auto_commit import AutoCommitter

# Runtime imports - used for instantiation, enums, constants, or function calls
from victor.agent.tool_supply_policy import (
    classify_tool_supply,
    demote_tools_to_fit,
    should_skip_tools_via_edge,
)
from victor.agent.control_plane import envelope_if_internal
from victor.agent.system_nudges import log_dropped_system_nudge

# DI container bootstrap
from victor.core.bootstrap import ensure_bootstrapped

# Service protocols for DI resolution

# Mixins (used at class definition time)
from victor.protocols.mode_aware import ModeAwareMixin
from victor.agent.capability_registry import OrchestratorCapabilityMixin

# Config and enums (used at runtime)
from victor.agent.search_router import SearchRoute, SearchType
from victor.agent.unified_task_tracker import TrackerTaskType

# Decomposed components - configs, strategies, functions
from victor.agent.conversation.controller import (
    ContextMetrics,
)
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import (
    create_session_state_manager,
)

# Recovery - enums and functions used at runtime
from victor.agent.orchestrator_recovery import (
    RecoveryAction as OrchestratorRecoveryAction,
)

# Pipeline - configs and results used at runtime
from victor.agent.tool_pipeline import ToolCallResult
from victor.agent.streaming_controller import (
    StreamingSession,
)
from victor.agent.task_analyzer import get_task_analyzer

# Observability
# from victor.observability.event_bus import EventBus, EventCategory, VictorEvent  # DELETED
from victor.observability.integration import ObservabilityIntegration

# Tool execution - functions and enums
from victor.agent.tool_calling import ToolCallParseResult
from victor.agent.orchestrator_utils import (
    calculate_max_context_chars,
)
from victor.agent.orchestrator_factory import OrchestratorFactory
from victor.config.settings import Settings
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
)
from victor.providers.registry import ProviderRegistry

# NOTE: no module-level import of victor.tools.semantic_selector here — it
# would complete the cycle semantic_selector -> victor.agent (package
# __init__ pulls this module) -> back to semantic_selector, which breaks
# cold-start import orders where numpy is absent (import-guard test,
# frozen binary). The selector is constructed via the registry instead.
from victor.workflows.discovery import register_builtin_workflows

# Streaming submodule - extracted for testability
from victor.agent.streaming import (
    StreamingChatContext,
)

# Progressive-tools support (moved verbatim to its own module; re-exported
# via __all__ for the historical module surface). Importing it here keeps the
# registry registration at orchestrator import time, as before.
from victor.agent.orchestrator_context import OrchestratorContextMixin
from victor.agent.orchestrator_init_runtime import OrchestratorInitRuntimeMixin
from victor.agent.orchestrator_metrics import OrchestratorMetricsMixin
from victor.agent.orchestrator_prompting import OrchestratorPromptingMixin
from victor.agent.orchestrator_vertical import OrchestratorVerticalMixin
from victor.agent.orchestrator_progressive import (
    PROGRESSIVE_TOOLS,
    _detect_mentioned_tools,
)
from victor.agent.orchestrator_protocol_conformance import (
    OrchestratorProtocolConformanceMixin,
)
from victor.agent.orchestrator_runtime_intelligence import (
    OrchestratorRuntimeIntelligenceMixin,
)

logger = logging.getLogger(__name__)

# Historical module surface preserved by the facade decomposition. Names are
# re-exported (some are patch targets, e.g. ProviderRegistry.create and
# asyncio.get_running_loop via this module's asyncio import; _detect_mentioned_tools
# is imported by tests).
__all__ = [
    "AgentOrchestrator",
    "PROGRESSIVE_TOOLS",
    "Settings",
    "Message",
    "BaseProvider",
    "RuntimeIdentity",
    "ensure_bootstrapped",
    "ProviderRegistry",
    "StreamChunk",
    "CompletionResponse",
    "_detect_mentioned_tools",
]


class AgentOrchestrator(
    OrchestratorInitRuntimeMixin,
    OrchestratorProtocolConformanceMixin,
    OrchestratorRuntimeIntelligenceMixin,
    OrchestratorContextMixin,
    OrchestratorMetricsMixin,
    OrchestratorVerticalMixin,
    OrchestratorPromptingMixin,
    ModeAwareMixin,
    OrchestratorCapabilityMixin,
):
    """Orchestrates agent interactions, tool execution, and provider communication.

    Uses ModeAwareMixin for consistent mode controller access (via self.is_build_mode,
    self.mode_controller, self.exploration_multiplier, etc.).

    Uses OrchestratorCapabilityMixin for explicit capability discovery, replacing hasattr
    duck-typing with type-safe protocol conformance. See capability_registry.py.
    """

    @staticmethod
    def _calculate_max_context_chars(
        settings: "Settings",
        provider: "BaseProvider",
        model: str,
    ) -> int:
        """Calculate maximum context size in characters for a model.

        Delegates to orchestrator_utils.calculate_max_context_chars.
        """
        return calculate_max_context_chars(settings, provider, model)

    def _initialize_services(self) -> None:
        """Initialize service layer by registering coordinators and bootstrapping services.

        Services are mandatory — all orchestrator delegation methods call services
        directly without coordinator fallback.

        Steps:
        1. Register coordinators in the container (for service dependencies)
        2. Bootstrap services using the registered coordinators
        3. Resolve service instances from the container
        """
        if not hasattr(self, "_container"):
            raise RuntimeError("ServiceContainer required for orchestrator operation")

        from victor.agent.services.protocols import (
            ChatServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
            SessionServiceProtocol,
            ToolServiceProtocol,
        )
        from victor.runtime.context import (
            ResolvedRuntimeServices,
            register_runtime_services,
        )

        register_runtime_services(
            self._container,
            ResolvedRuntimeServices(
                chat=getattr(self, "_chat_service", None),
                tool=getattr(self, "_tool_service", None),
                session=getattr(self, "_session_service", None),
                context=getattr(self, "_context_service", None),
                provider=getattr(self, "_provider_service", None),
                recovery=getattr(self, "_recovery_service", None),
            ),
        )

        # Register coordinators in container for service dependencies
        self._register_coordinators_for_services()

        # Bootstrap services using the registered coordinators
        self._bootstrap_service_layer()

        # Resolve service instances from container
        self._chat_service = self._container.get_optional(ChatServiceProtocol)
        self._tool_service = self._container.get_optional(ToolServiceProtocol)
        self._session_service = self._container.get_optional(SessionServiceProtocol)
        self._context_service = self._container.get_optional(ContextServiceProtocol)
        self._provider_service = self._container.get_optional(ProviderServiceProtocol)
        self._recovery_service = self._container.get_optional(RecoveryServiceProtocol)

        if self._tool_service is None:
            from victor.agent.services.tool_service import ToolService, ToolServiceConfig

            self._tool_service = ToolService(
                config=ToolServiceConfig(default_tool_budget=getattr(self, "tool_budget", 100)),
                tool_selector=(self.tool_selector if hasattr(self, "tool_selector") else None),
                tool_executor=getattr(self, "tool_executor", None),
                tool_registrar=getattr(self, "tools", None),
                settings=self.settings,
            )

        if self._tool_service and hasattr(self._tool_service, "bind_runtime_components"):
            self._tool_service.bind_runtime_components(
                tool_registry=getattr(self, "tools", None),
                tool_pipeline=getattr(self, "_tool_pipeline", None),
                tool_cache=getattr(self, "tool_cache", None),
                mode_controller=(
                    self.mode_controller if hasattr(self, "mode_controller") else None
                ),
                tool_planner=getattr(self, "_tool_planner", None),
                argument_normalizer=getattr(self, "argument_normalizer", None),
            )

        if self._chat_service and hasattr(self._chat_service, "bind_runtime_components"):
            from victor.agent.runtime.provider_runtime import LazyRuntimeProxy
            from victor.agent.factory.chat_runtime_bindings import bind_chat_turn_runtime

            chat_stream_adapter = self._get_chat_stream_adapter()
            self._chat_service.bind_runtime_components(
                turn_executor=LazyRuntimeProxy(
                    factory=lambda: self.turn_executor,
                    name="turn_executor",
                ),
                planning_handler=self._get_planning_chat_runtime().run,
                stream_chat_handler=chat_stream_adapter.stream_chat_under_turn_lock,
                context_limit_handler=self._get_context_limit_runtime().handle_limits,
                turn_runtime=bind_chat_turn_runtime(self),
                stream_turn_lock=chat_stream_adapter.services.stream_turn_lock,
            )
        if self._provider_service is not None and hasattr(
            self._provider_service, "bind_runtime_components"
        ):
            self._provider_service.bind_runtime_components(
                provider_manager=getattr(self, "_provider_manager", None),
            )
        elif self._provider_service is not None and hasattr(self, "provider"):
            self._provider_service._current_provider = self.provider

        # RecoveryService is now created with native streaming runtime enabled in service_provider.
        # No additional binding is required. The _recovery_coordinator attribute is now
        # the RecoveryService instance created by factory.create_recovery_coordinator().

        logger.debug(
            "Service layer initialized: chat=%s, tool=%s, session=%s, "
            "context=%s, provider=%s, recovery=%s",
            self._chat_service is not None,
            self._tool_service is not None,
            self._session_service is not None,
            self._context_service is not None,
            self._provider_service is not None,
            self._recovery_service is not None,
        )

    def _register_coordinators_for_services(self) -> None:
        """Register coordinators in the container for service layer dependencies.

        Services need access to coordinators through the container. This method
        registers the coordinator protocols so services can resolve them.
        """
        from victor.agent.protocols import (
            ConversationControllerProtocol,
            StreamingCoordinatorProtocol,
        )
        from victor.core.container import ServiceLifetime

        # Register conversation controller if not already registered
        if not self._container.is_registered(ConversationControllerProtocol):
            self._container.register(
                ConversationControllerProtocol,
                lambda c: self._conversation_controller,
                ServiceLifetime.SINGLETON,
            )

        # Register streaming coordinator if not already registered
        if not self._container.is_registered(StreamingCoordinatorProtocol):
            self._container.register(
                StreamingCoordinatorProtocol,
                lambda c: self._streaming_controller,
                ServiceLifetime.SINGLETON,
            )

        logger.debug("Registered coordinators in container for service layer")

    def _bootstrap_service_layer(self) -> None:
        """Bootstrap the service layer using registered coordinators.

        This creates and registers all services (ChatService, ToolService, etc.)
        using the coordinators that were registered in the container.
        """
        from victor.core.bootstrap_services import bootstrap_new_services

        # Bootstrap services with the registered coordinators
        bootstrap_new_services(
            self._container,
            conversation_controller=self._conversation_controller,
            streaming_coordinator=self._streaming_controller,
        )

        logger.debug("Bootstrapped service layer")

    def _create_execution_context(self) -> "RuntimeExecutionContext":
        """Create a RuntimeExecutionContext carrying all runtime dependencies.

        SEAM BOUNDARY: Agent-level execution context (not workflow or tool-level)
        - Use RuntimeExecutionContext for: Agent orchestrator, session management, DI services
        - Do NOT use for: Workflow nodes (use WorkflowNodeContext), tools (use ToolExecutionContext)

        Replaces scattered get_global_manager() / get_instance() calls
        with explicit context passing. Created after services are
        bootstrapped so all dependencies are available.

        Returns:
            RuntimeExecutionContext with settings, container, state manager,
            and lazy service accessor.
        """
        from victor.runtime.context import RuntimeExecutionContext

        session_id = getattr(self, "_memory_session_id", "") or ""
        return RuntimeExecutionContext.create(
            settings=self.settings,
            container=self._container,
            session_id=session_id,
        )

    def __init__(
        self,
        settings: Settings,
        provider: BaseProvider,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        console: Optional[Console] = None,
        tool_selection: Optional[Dict[str, Any]] = None,
        thinking: bool = False,
        provider_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        system_prompt_override: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """Initialize orchestrator.

        Args:
            settings: The application settings.
            provider: LLM provider instance
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            console: Rich console for output
            tool_selection: Optional tool selection configuration (base_threshold, base_max_tools)
            thinking: Enable extended thinking mode (Claude models only)
            provider_name: Optional provider label from profile (e.g., lmstudio, vllm) to disambiguate OpenAI-compatible providers
            profile_name: Optional profile name (e.g., "groq-fast", "claude-sonnet") for session tracking
            system_prompt_override: Optional complete system prompt for isolated child runtimes
            reasoning_effort: Optional reasoning effort ("low"/"medium"/"high") for
                reasoning-capable models. None inherits/omits. Forwarded to the
                provider only when it reports support (see
                ``BaseProvider.supports_reasoning_effort``); ignored otherwise.
        """
        # Store profile name for session tracking
        self._profile_name = profile_name
        # Track active session ID for parallel session support
        self.active_session_id: Optional[str] = None
        self._runtime_identity = RuntimeIdentity(self)
        # Bootstrap DI container - ensures all services are available
        # This is idempotent and will only bootstrap if not already done
        self._container = ensure_bootstrapped(settings)

        # Create factory for component initialization (CRITICAL-001)
        # This enables DI-aware creation with fallback and reduces __init__ complexity
        self._factory = OrchestratorFactory(
            settings=settings,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider_name=provider_name,
            profile_name=profile_name,
            tool_selection=tool_selection,
            thinking=thinking,
        )
        self._factory._container = self._container  # Share container

        self.settings = settings
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.console = console or Console()
        self.tool_selection = tool_selection or {}
        self.thinking = thinking

        # Tool calling matrix for managing provider capabilities (via factory)
        self.tool_calling_models, self.tool_capabilities = (
            self._factory.create_tool_calling_matrix()
        )

        # Initialize ProviderManager with tool adapter (via factory)
        (
            self._provider_manager,
            self.provider,
            self.model,
            self.provider_name,
            self.tool_adapter,
            self.tool_calling_caps,
        ) = self._factory.create_provider_manager_with_adapter(provider, model, provider_name)

        # FEP-0016: InitializationPhaseManager drives the 9 interleaved init phases in place.
        from victor.agent.runtime.initialization_manager import InitializationPhaseManager

        self._init_manager = InitializationPhaseManager()
        self._init_manager.run_phase(self, "provider_runtime")

        # Response sanitizer for cleaning model output (via factory - DI with fallback)
        self.sanitizer = self._factory.create_sanitizer()

        # System prompt builder with vertical prompt contributors (via factory)
        self.prompt_builder = self._factory.create_system_prompt_builder(
            provider_name=self.provider_name,
            model=model,
            tool_adapter=self.tool_adapter,
            tool_calling_caps=self.tool_calling_caps,
        )

        # UnifiedPromptPipeline: consolidated prompt assembly (replaces
        # PromptComposer + the older frozen-prompt glue layers).
        try:
            from victor.agent.content_registry import create_default_registry
            from victor.agent.optimization_injector import OptimizationInjector
            from victor.agent.prompt_pipeline import UnifiedPromptPipeline
            from victor.agent.services.runtime_intelligence import (
                RuntimeIntelligenceService,
            )

            self._optimization_injector = OptimizationInjector()
            self._runtime_intelligence = RuntimeIntelligenceService.from_orchestrator(
                self,
                task_analyzer=get_task_analyzer(),
                optimization_injector=self._optimization_injector,
            )
            # Fetch optional services for prompt pipeline
            from victor.agent.tool_pipeline import ToolPipeline
            from victor.framework.rl.credit_tracking_service import (
                CreditTrackingService,
            )

            credit_tracking_service = self._container.get_optional(CreditTrackingService)
            tool_pipeline = self._container.get_optional(ToolPipeline)

            self._prompt_pipeline = UnifiedPromptPipeline(
                provider=self.provider,
                builder=self.prompt_builder,
                registry=create_default_registry(),
                optimizer=self._optimization_injector,
                runtime_intelligence=self._runtime_intelligence,
                get_context_window=self._get_model_context_window,
                credit_tracking_service=credit_tracking_service,
                tool_pipeline=tool_pipeline,
            )
            # Backward compat alias — old code referencing _prompt_composer
            self._prompt_composer = self._prompt_pipeline
        except Exception as e:
            logger.debug("UnifiedPromptPipeline unavailable: %s", e)
            self._optimization_injector = None
            self._runtime_intelligence = None
            self._prompt_pipeline = None
            self._prompt_composer = None

        # Load project context from .victor/init.md (via factory - DI with fallback)
        self.project_context = self._factory.create_project_context()

        # Build system prompt using the canonical prompt runtime surface
        base_system_prompt = self.build_system_prompt()

        if system_prompt_override:
            self._system_prompt = system_prompt_override
        else:
            self._system_prompt = base_system_prompt

        self._system_added = False

        # Streaming loop tracking counters (previously injected via hasattr in pipeline.py)
        self._continuation_prompts: int = 0
        self._asking_input_prompts: int = 0
        self._consecutive_blocked_attempts: int = 0
        self._cumulative_prompt_interventions: int = 0

        # Initialize tool call budget (via factory) - uses adapter recommendations with settings override
        self.tool_budget = self._factory.initialize_tool_budget(self.tool_calling_caps)

        # Initialize SessionStateManager for consolidated execution state tracking (TD-002)
        # Replaces scattered state variables: tool_calls_used, observed_files, executed_tools,
        # failed_tool_signatures, _read_files_session, _required_files, _required_outputs, etc.
        self._session_state = create_session_state_manager(tool_budget=self.tool_budget)
        self._session_accessor = SessionStateAccessor(self._session_state)
        self._runtime_identity.bind_turn_lock(self._session_accessor.stream_turn_lock)

        # Gap implementations: Complexity classifier, action authorizer, search router (via factory)
        self.task_classifier = self._factory.create_complexity_classifier()
        self.intent_detector = self._factory.create_action_authorizer()
        self.search_router = self._factory.create_search_router()

        # Presentation adapter for icon/emoji rendering (via factory, DI)
        # Decouples agent layer from direct UI dependencies
        self._presentation = self._factory.create_presentation_adapter()

        # Task Completion Detection: signal-based completion detection
        # Uses explicit rare completion markers for deterministic completion
        from victor.agent.task_completion import TaskCompletionDetector

        self._task_completion_detector = TaskCompletionDetector(
            runtime_intelligence=self._runtime_intelligence
        )
        logger.debug("TaskCompletionDetector initialized (signal-based completion)")

        # Context reminder manager for intelligent system message injection (via factory, DI)
        # Reduces token waste by consolidating reminders and only injecting when context changes
        self.reminder_manager = self._factory.create_reminder_manager(
            provider=self.provider_name,
            task_complexity="medium",  # Will be updated per-task
            tool_budget=self.tool_budget,
        )

        # Debug logger for incremental output and conversation tracking (via factory)
        self.debug_logger = self._factory.create_debug_logger_configured()

        self._init_manager.run_phase(self, "metrics_runtime")

        # CallbackCoordinator: centralized callback delegation for tool/streaming events.
        self._callback_coordinator = self._build_callback_coordinator()

        # Cancellation support for streaming
        self._cancel_event: Optional[asyncio.Event] = None
        self._is_streaming = False

        # Background task tracking for graceful shutdown
        self._background_tasks: set[asyncio.Task] = set()
        self._bg_task_lock = threading.Lock()

        # Tool error dedup (bounded to prevent memory leak in long sessions)
        self._shown_tool_errors: set = set()
        self._tool_context_cache: Optional[dict] = None

        self._init_manager.run_phase(self, "workflow_runtime")

        # Constraint activation service for unified constraint management
        from victor.agent.constraint_activation_service import get_constraint_activator

        self._constraint_activator = get_constraint_activator()
        logger.debug("ConstraintActivationService initialized")

        # Conversation history (via factory) - MessageHistory for better encapsulation
        self.conversation = self._factory.create_message_history(self._system_prompt)

        self._init_manager.run_phase(self, "memory_runtime")

        # Conversation state machine for intelligent stage detection
        self.conversation_state = self._factory.create_conversation_state_machine()

        # Intent classifier for semantic continuation/completion detection
        self.intent_classifier = self._factory.create_intent_classifier()

        # Runtime-intelligence integration (lazy initialization)
        self._runtime_intelligence_integration: Optional["OrchestratorIntegration"] = None
        self._runtime_intelligence_integration_config = self._factory.create_integration_config()
        from victor.config.pipeline_settings import resolve_runtime_intelligence_enabled

        self._runtime_intelligence_enabled = resolve_runtime_intelligence_enabled(settings)

        # Sub-agent orchestration (lazy initialization)
        self._subagent_orchestrator, self._subagent_orchestration_enabled = (
            self._factory.setup_subagent_orchestration()
        )

        # Component assembly phases (tools → conversation → intelligence)
        from victor.agent.runtime.component_assembler import ComponentAssembler

        ComponentAssembler.assemble_tools(self, provider, model)
        ComponentAssembler.assemble_conversation(self, provider, model)
        ComponentAssembler.assemble_intelligence(self)

        # Post-factory component preparation and facade assembly
        from victor.agent.runtime.bootstrapper import AgentRuntimeBootstrapper

        AgentRuntimeBootstrapper.prepare_components(self, settings)
        AgentRuntimeBootstrapper.finalize(self)

        # Create RuntimeExecutionContext — explicit context object replacing global singletons.
        # Available after services are bootstrapped; passed to coordinators and workflows.
        self._execution_context = self._create_execution_context()

        # Cache optimization flags once (provider is fully initialized)
        self._kv_opt_cached: Optional[bool] = None
        self._cache_opt_cached: Optional[bool] = None
        self._last_sorted_tool_names: Optional[frozenset] = None
        self._last_sorted_tools: Optional[list] = None
        self._session_semantic_tools: Optional[list] = None
        self._compute_cache_flags()

    # =====================================================================
    # Callbacks for decomposed components — delegated to CallbackCoordinator
    # =====================================================================

    def _build_callback_coordinator(self) -> Any:
        """Lazily construct the CallbackCoordinator."""
        from victor.agent.callback_coordinator import CallbackCoordinator

        return CallbackCoordinator(
            metrics_coordinator=self._metrics_coordinator,
            get_observability=lambda: getattr(self, "_observability", None),
            get_iteration_count=lambda: (
                self._tool_pipeline.calls_used if hasattr(self, "_tool_pipeline") else 0
            ),
            get_usage_analytics=lambda: (
                self._usage_analytics
                if hasattr(self, "_usage_analytics") and self._usage_analytics
                else None
            ),
            get_rl_coordinator=lambda: self._rl_coordinator,
            get_vertical_context=lambda: self._vertical_context,
            get_tool_service=lambda: getattr(self, "_tool_service", None),
        )

    def _on_tool_start_callback(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Callback when tool execution starts (from ToolPipeline)."""
        self._callback_coordinator.on_tool_start(tool_name, arguments)

    def _on_tool_complete_callback(self, result: ToolCallResult) -> None:
        """Callback when tool execution completes (from ToolPipeline)."""
        nudge_flag = [getattr(self, "_all_files_read_nudge_sent", False)]
        self._callback_coordinator.on_tool_complete(
            result,
            read_files_session=self._read_files_session,
            required_files=self._required_files,
            required_outputs=self._required_outputs,
            nudge_sent_flag=nudge_flag,
            add_message=self.add_message,
        )
        self._all_files_read_nudge_sent = nudge_flag[0]

    def _on_streaming_session_complete(self, session: StreamingSession) -> None:
        """Callback when streaming session completes (from StreamingController)."""
        self._callback_coordinator.on_streaming_session_complete(session)

    @property
    def workflow_registry(self) -> Any:
        """Get workflow registry, materializing lazy runtime on first access."""
        registry = getattr(self, "_workflow_registry", None)
        if hasattr(registry, "get_instance"):
            resolved = registry.get_instance()
            self._workflow_registry = resolved
            return resolved
        return registry

    @workflow_registry.setter
    def workflow_registry(self, value: Any) -> None:
        """Set workflow registry (supports test overrides)."""
        self._workflow_registry = value

    @property
    def observability(self) -> Optional[ObservabilityIntegration]:
        """Get the observability integration component.

        Returns:
            ObservabilityIntegration instance for event bus access, or None if disabled
        """
        return getattr(self, "_observability", None)

    @observability.setter
    def observability(self, value: Optional[ObservabilityIntegration]) -> None:
        """Set the observability integration component.

        This allows FrameworkShim to inject an externally-configured
        ObservabilityIntegration instance for unified event handling.

        Args:
            value: ObservabilityIntegration instance or None to disable
        """
        self._observability = value

    def set_observability(self, value: Optional[ObservabilityIntegration]) -> None:
        """Set observability integration via explicit port method.

        Args:
            value: ObservabilityIntegration instance or None.
        """
        self.observability = value

    @property
    def container(self) -> Any:
        """Get the service container via a public property."""
        return self._container

    def get_service_container(self) -> Any:
        """Get the DI service container via explicit port method."""
        return self._container

    def get_capability_config_scope_key(self) -> str:
        """Get stable scope key for framework capability-config service storage.

        Always returns a value (falls back to a per-instance key), matching the
        SDK port's ``-> str`` contract.
        """
        if self.active_session_id:
            normalized_active = str(self.active_session_id).strip()
            if normalized_active:
                return normalized_active

        memory_session_id = getattr(self, "_memory_session_id", None)
        if memory_session_id:
            normalized_memory = str(memory_session_id).strip()
            if normalized_memory:
                return normalized_memory

        return f"orchestrator:{id(self)}"

    def get_capability_loader(self) -> Optional[Any]:
        """Get the cached framework CapabilityLoader if available."""
        return getattr(self, "_capability_loader", None)

    def set_capability_loader(self, loader: Any) -> None:
        """Set the framework CapabilityLoader instance."""
        self._capability_loader = loader

    def get_or_create_capability_loader(self) -> Any:
        """Get or lazily create the framework CapabilityLoader."""
        loader = self.get_capability_loader()
        if loader is None:
            from victor.framework.capability_loader import CapabilityLoader

            loader = CapabilityLoader()
            self.set_capability_loader(loader)
        return loader

    # =========================================================================
    # Vertical Protocol Methods
    # These implement OrchestratorVerticalProtocol for proper vertical integration
    # =========================================================================

    async def save_checkpoint(
        self,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Save a manual checkpoint of the current conversation state.

        Delegates to SessionService.

        Args:
            description: Human-readable description for the checkpoint
            tags: Optional tags for categorization

        Returns:
            Checkpoint ID if saved, None if checkpointing is disabled
        """
        return await self._session_service.save_checkpoint(description, tags)

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore conversation state from a checkpoint.

        Delegates to SessionService.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            True if restored successfully, False otherwise
        """
        return await self._session_service.restore_checkpoint(checkpoint_id)

    async def maybe_auto_checkpoint(self) -> Optional[str]:
        """Trigger auto-checkpoint if interval threshold is met.

        Delegates to SessionService.

        Returns:
            Checkpoint ID if auto-checkpoint was created, None otherwise
        """
        return await self._session_service.maybe_auto_checkpoint()

    @property
    def safety_checker(self) -> "SafetyChecker":
        """Get the safety checker for dangerous operation detection.

        UI layers can use this to set confirmation callbacks:
            orchestrator.safety_checker.confirmation_callback = my_callback

        Returns:
            SafetyChecker instance for dangerous operation detection
        """
        return self._safety_checker

    @property
    def auto_committer(self) -> Optional["AutoCommitter"]:
        """Get the auto-committer for AI-assisted code change commits.

        Usage:
            result = orchestrator.auto_committer.commit_changes(
                files=["src/api.py"],
                description="Add input validation",
                change_type="feat"
            )

        Returns:
            AutoCommitter instance or None if not enabled
        """
        return self._auto_committer

    # =========================================================================
    # Vertical Extension Support
    # =========================================================================

    # =========================================================================
    # Internal Storage Setters (DIP Compliance)
    # These methods provide controlled access for adapter implementations,
    # replacing direct private attribute writes. Only called by
    # VerticalIntegrationAdapter - not for general use.
    # =========================================================================

    # =========================================================================
    # VerticalStorageProtocol Implementation (DIP Compliance)
    # These public methods implement VerticalStorageProtocol, providing a clean
    # interface for vertical data storage and retrieval. This replaces direct
    # private attribute access with protocol-compliant methods.
    # =========================================================================

    def get_context_metrics(self) -> ContextMetrics:
        """Get detailed context metrics.

        Returns:
            ContextMetrics with size and overflow information
        """
        return self._context_service.get_context_metrics()

    def _create_background_task(
        self, coro: Any, name: str = "background_task"
    ) -> Optional[asyncio.Task]:
        """Create and track a background task for graceful shutdown."""
        from victor.agent.services.session_service import SessionService

        return SessionService.create_background_task(
            coro, name, self._background_tasks, self._bg_task_lock
        )

    def route_search_query(self, query: str) -> Dict[str, Any]:
        """Route a search query to the optimal search tool using SearchRouter.

        Analyzes the query to determine whether keyword search (code_search)
        semantic search (semantic_code_search), or graph traversal (graph)
        would yield better results.

        Args:
            query: The search query

        Returns:
            Dictionary with routing recommendation:
                - recommended_tool: "code_search" or "semantic_code_search" or "graph" or "both"
                - recommended_args: Suggested tool arguments (for example {"mode": "bugs"})
                - confidence: Confidence in the recommendation (0.0-1.0)
                - reason: Human-readable explanation
                - search_type: SearchType enum value

        Example:
            route = orchestrator.route_search_query("class BaseTool")
            # Returns: {"recommended_tool": "code_search", "confidence": 1.0, ...}

            route = orchestrator.route_search_query("how does error handling work")
            # Returns: {"recommended_tool": "semantic_code_search", "confidence": 0.9, ...}
        """
        route: SearchRoute = self.search_router.route(query)

        # Map SearchType to tool name
        tool_map = {
            SearchType.KEYWORD: "code_search",
            SearchType.SEMANTIC: "semantic_code_search",
            SearchType.HYBRID: "both",
        }

        recommended_tool = route.tool_name or tool_map.get(route.search_type, "code_search")

        return {
            "recommended_tool": recommended_tool,
            "recommended_args": route.tool_arguments,
            "confidence": route.confidence,
            "reason": route.reason,
            "search_type": route.search_type.value,
            "matched_patterns": route.matched_patterns,
            "transformed_query": route.transformed_query,
        }

    def get_recommended_search_tool(self, query: str) -> str:
        """Get the recommended search tool name for a query.

        Convenience method that returns just the tool name.

        Args:
            query: The search query

        Returns:
            Tool name: "code_search", "semantic_code_search", "graph", or "both"
        """
        return self.route_search_query(query)["recommended_tool"]

    def _get_provider_management_runtime(self) -> Any:
        """Get the canonical service-owned provider management runtime helper."""
        runtime = getattr(self, "_provider_management_runtime", None)
        if runtime is None:
            from victor.agent.services.provider_management_runtime import (
                ProviderManagementRuntime,
            )

            runtime = ProviderManagementRuntime(self.protocol_adapter)
            self._provider_management_runtime = runtime
        return runtime

    async def start_health_monitoring(self) -> bool:
        """[CANONICAL] Start background health monitoring via ProviderService.

        Returns:
            True if monitoring started, False if already running or unavailable
        """
        return await self._get_provider_management_runtime().start_health_monitoring()

    async def stop_health_monitoring(self) -> bool:
        """[CANONICAL] Stop background health monitoring via ProviderService.

        Returns:
            True if monitoring stopped, False if not running or error
        """
        return await self._get_provider_management_runtime().stop_health_monitoring()

    async def get_provider_health(self) -> Dict[str, Any]:
        """[CANONICAL] Get health status via ProviderService.

        Delegates to ProviderService for diagnostics, maintaining backward
        compatibility by returning the expected health information dictionary.
        """
        return await self._get_provider_management_runtime().get_provider_health()

    async def graceful_shutdown(self) -> Dict[str, bool]:
        """Perform graceful shutdown of all orchestrator components.

        Delegates to LifecycleManager for core shutdown logic.

        Flushes analytics, stops health monitoring, and cleans up resources.
        Call this before application exit.

        Returns:
            Dictionary with shutdown status for each component
        """
        # Delegate to LifecycleManager for graceful shutdown
        return await self._lifecycle_manager.graceful_shutdown()

    # =========================================================================
    # Provider/Model Hot-Swap Methods
    # =========================================================================

    def get_current_provider_info(self) -> Dict[str, Any]:
        """[CANONICAL] Get information about the current provider via ProviderService.

        Combines provider metadata from the service with orchestrator-specific
        runtime state (tool budget, usage).

        Returns:
            Dictionary with provider/model info and capabilities.
        """
        return self._get_provider_management_runtime().get_current_provider_info()

    def _parse_tool_calls_with_adapter(
        self, content: str, raw_tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ToolCallParseResult:
        """Parse tool calls using the tool calling adapter.

        This is the unified method for parsing tool calls that handles:
        1. Native tool calls from provider
        2. JSON fallback parsing
        3. XML fallback parsing
        4. Tool name validation

        Args:
            content: Response content text
            raw_tool_calls: Native tool_calls from provider (if any)

        Returns:
            ToolCallParseResult with parsed tool calls and metadata
        """
        result = self.tool_adapter.parse_tool_calls(content, raw_tool_calls)

        # Log any warnings
        for warning in result.warnings:
            logger.warning(f"Tool call parse warning: {warning}")

        # Log parse method for debugging
        if result.tool_calls:
            logger.debug(
                f"Parsed {len(result.tool_calls)} tool calls via {result.parse_method} "
                f"(confidence={result.confidence})"
            )

        return result

    def _log_tool_call(self, name: str, kwargs: dict) -> None:
        """A hook that logs information before a tool is called."""
        # Move verbose argument logging to debug level - not user-facing
        logger.debug(f"Tool call: {name} with args: {kwargs}")

    def _classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task type based on keywords in the user message.

        Delegates to the service-owned task-guidance runtime.

        Args:
            user_message: The user's input message

        Returns:
            Dictionary with classification results
        """
        return self._get_task_guidance_runtime().classify_task_keywords(user_message)

    def _classify_task_with_context(
        self, user_message: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Classify task with conversation context for improved accuracy.

        Delegates to UnifiedPromptPipeline or TaskAnalyzer directly.

        Args:
            user_message: The user's input message
            history: Optional conversation history for context boosting

        Returns:
            Dictionary with classification results
        """
        pipeline = getattr(self, "_prompt_pipeline", None)
        if pipeline is not None:
            return pipeline.classify_task_with_context(user_message, history)

        task_analyzer = getattr(self, "_task_analyzer", None)
        if task_analyzer is not None:
            try:
                method = getattr(
                    task_analyzer,
                    "classify_task_with_context",
                    getattr(task_analyzer, "classify_with_context", None),
                )
                if method is not None:
                    return method(user_message, history)
            except Exception as exc:
                logger.debug("Context-aware task classification fallback failed: %s", exc)

        return {"task_type": "default", "confidence": 0.0}

    def _format_tool_output(self, tool_name: str, args: Dict[str, Any], output: Any) -> str:
        """Format tool output via the canonical tool execution runtime."""
        return self._get_tool_execution_runtime().format_tool_output(tool_name, args, output)

    def _register_default_workflows(self) -> None:
        """Register default workflows via dynamic discovery.

        Uses DIP-compliant workflow discovery to avoid hardcoded imports.
        Workflows are discovered from victor.workflows package automatically.
        """
        count = register_builtin_workflows(self.workflow_registry)
        logger.debug(f"Dynamically registered {count} workflows")

    def _should_use_tools(self) -> bool:
        """Always return True - tool selection is handled by _select_relevant_tools()."""
        return True

    def _model_supports_tool_calls(self) -> bool:
        """Check provider/model tool-call support via the canonical provider runtime."""
        return self._get_provider_management_runtime().model_supports_tool_calls()

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to conversation history with smart system handling.

        When role="system":
        - If cache_optimization_enabled=True and a root system prompt exists,
          intercepts the message and passes it to reminder_manager for per-turn
          injection into the user prefix instead of appending a new system message.
        - This preserves byte-identical system prompts for provider KV caching.

        Delegates count retention to MessageHistory, preserving the active task
        and current tool exchange even when their retention floor exceeds the
        configured target. Persistence and usage logging are delegated to
        ``ChatService.persist_message``.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            **kwargs: Additional fields (name, tool_call_id, tool_calls)
        """
        persist_synchronously = bool(kwargs.pop("persist_synchronously", False))
        require_persistence = bool(kwargs.pop("require_persistence", False))
        preview_keys = {
            "preview_body",
            "preview_kind",
            "preview_language",
            "preview_path",
        }
        preview_metadata = {key: kwargs[key] for key in preview_keys if key in kwargs}
        if preview_metadata and hasattr(self.conversation, "add_preview_message"):
            self.conversation.add_preview_message(role, content, preview_metadata)
            return

        _nonce = getattr(getattr(self, "_prompt_pipeline", None), "_channel_nonce", "")
        content = envelope_if_internal(role, content, kwargs.get("metadata"), _nonce)
        # Intercept dynamic system nudges for cache-friendly injection
        if role == "system" and getattr(self, "_cache_optimization_enabled", False):
            # Check if history already contains a system message (the root prompt)
            has_root_system = any(m.role == "system" for m in self.conversation._messages)
            if has_root_system and hasattr(self, "reminder_manager") and self.reminder_manager:
                if content.startswith("[") and content.endswith("]"):
                    log_dropped_system_nudge(content)
                    return

        from victor.agent.services.chat_service import ChatService

        ChatService.publish_message(
            self,
            role,
            content,
            persist_synchronously=persist_synchronously,
            require_persistence=require_persistence,
            **kwargs,
        )

    async def chat(
        self,
        user_message: str,
        use_planning: Optional[bool] = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send a chat message through the canonical ChatService runtime.

        .. deprecated::
            Direct orchestrator access is deprecated. Use ChatService instead:
            - From Agent: Use Agent.run() or Agent.stream() on the framework surface
            - Direct access: Use ChatService.chat() from the service layer
            This method will be removed in v2.0.

        Args:
            user_message: User's message
            use_planning: Whether to use planning. None = auto-detect,
                True = always, False = never.
            **kwargs: Additional arguments, including optional 'constraints' for constraint activation

        Returns:
            CompletionResponse from the model with complete response
        """
        import warnings

        warnings.warn(
            "Direct orchestrator.chat() access is deprecated. "
            "Use ChatService.chat() from the service layer instead. "
            "This method will be removed in v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )

        return await self._chat_service.chat(
            user_message,
            use_planning=use_planning,
            **kwargs,
        )

    async def chat_with_planning(
        self,
        user_message: str,
        use_planning: Optional[bool] = None,
    ) -> CompletionResponse:
        """Chat with automatic planning for complex multi-step tasks.

        This method extends regular chat by automatically detecting when a task
        is complex enough to benefit from structured planning. It then:

        1. Generates a structured plan using ReadableTaskPlan (token-efficient)
        2. Executes the plan step-by-step with context-aware tools
        3. Provides a comprehensive summary of results

        For simple tasks, it falls back to regular direct chat.

        Args:
            user_message: User's message
            use_planning: Force planning on/off. None = auto-detect

        Returns:
            CompletionResponse from the model

        Example:
            # Auto-detect if planning is needed (recommended)
            response = await agent.chat_with_planning(
                "Analyze the codebase architecture and provide SOLID evaluation"
            )

            # Force planning mode
            response = await agent.chat_with_planning(
                "Implement user auth",
                use_planning=True
            )

            # Disable planning (use direct chat)
            response = await agent.chat_with_planning(
                "Quick question",
                use_planning=False
            )
        """
        return await self._chat_service.chat_with_planning(user_message, use_planning)

    def _get_planning_chat_runtime(self) -> Any:
        """Get the canonical service-owned planning chat runtime helper."""
        runtime = getattr(self, "_planning_chat_runtime", None)
        if runtime is None:
            from victor.agent.services.planning_chat_runtime import PlanningChatRuntime

            runtime = PlanningChatRuntime(self)
            self._planning_chat_runtime = runtime
        return runtime

    def _get_context_limit_runtime(self) -> Any:
        """Get the canonical service-owned context/iteration limit runtime helper."""
        runtime = getattr(self, "_context_limit_runtime", None)
        if runtime is None:
            from victor.agent.services.context_limit_runtime import ContextLimitRuntime

            runtime = ContextLimitRuntime(self)
            self._context_limit_runtime = runtime
        return runtime

    def _get_chat_stream_adapter(self) -> Any:
        """Get the canonical service-owned chat-stream adapter."""
        adapter = getattr(self, "_chat_stream_adapter", None)
        if adapter is None:
            factory = getattr(self, "_factory", None)
            if factory is not None and hasattr(factory, "create_streaming_chat_adapter"):
                adapter = factory.create_streaming_chat_adapter(self.protocol_adapter)
            else:
                from victor.agent.services.chat_stream_runtime import (
                    ServiceStreamingRuntime,
                )

                adapter = ServiceStreamingRuntime(self.protocol_adapter)
            self._chat_stream_adapter = adapter
        return adapter

    def _get_task_guidance_runtime(self) -> Any:
        """Get the canonical service-owned task guidance runtime helper."""
        runtime = getattr(self, "_task_guidance_runtime", None)
        if runtime is None:
            from victor.agent.services.task_guidance_runtime import TaskGuidanceRuntime

            runtime = TaskGuidanceRuntime(self.protocol_adapter)
            self._task_guidance_runtime = runtime
        return runtime

    def _get_tool_selection_runtime(self) -> Any:
        """Get the canonical service-owned tool selection runtime helper."""
        runtime = getattr(self, "_tool_selection_runtime", None)
        if runtime is None:
            from victor.agent.services.tool_selection_runtime import (
                ToolSelectionRuntime,
            )

            runtime = ToolSelectionRuntime(self.protocol_adapter)
            self._tool_selection_runtime = runtime
        return runtime

    def _prepare_task(
        self, user_message: str, unified_task_type: TrackerTaskType
    ) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments.

        Delegates to TaskCoordinator for centralized task preparation.

        Extracted from CRITICAL-001 Phase 2D.
        """
        return self._get_task_guidance_runtime().prepare_task(user_message, unified_task_type)

    def _apply_intent_guard(self, user_message: str) -> None:
        """Detect intent and inject prompt guards for read-only tasks.

        Delegates to TaskCoordinator for centralized intent detection.

        Extracted from CRITICAL-001 Phase 2D.
        """
        self._get_task_guidance_runtime().apply_intent_guard(user_message)

    def _apply_task_guidance(
        self,
        user_message: str,
        unified_task_type: TrackerTaskType,
        is_analysis_task: bool,
        is_action_task: bool,
        needs_execution: bool,
        max_exploration_iterations: int,
    ) -> None:
        """Apply guidance and budget tweaks for analysis/action tasks.

        Delegates to TaskCoordinator for centralized task guidance.

        Extracted from CRITICAL-001 Phase 2D.
        """
        self._get_task_guidance_runtime().apply_task_guidance(
            user_message=user_message,
            unified_task_type=unified_task_type,
            is_analysis_task=is_analysis_task,
            is_action_task=is_action_task,
            needs_execution=needs_execution,
            max_exploration_iterations=max_exploration_iterations,
        )

    async def _select_tools_for_turn(
        self,
        context_msg: str,
        goals: Any,
        planned_tools: Any = None,
    ) -> Any:
        """Select and prioritize tools for the current turn.

        Applies TOOL_NECESSITY check first: if the edge model (or heuristic)
        determines the request is purely conversational (Q&A, greeting,
        clarification), tool selection and broadcasting are skipped entirely.
        This saves ~2-4K tokens per conversational turn.
        """
        return await self._get_tool_selection_runtime().select_tools_for_turn(
            context_msg,
            goals,
            planned_tools=planned_tools,
        )

    # ------------------------------------------------------------------
    # Tool Necessity Gate (HDPO-inspired Q&A bypass)
    # ------------------------------------------------------------------

    # Tool-supply policy constants + heuristic moved to
    # victor/agent/tool_supply_policy.py (ADR-019 increment 2).

    def _tool_skip_mode(self, context_msg: str) -> str:
        """Decide tool supply for a turn — delegates to the pure tool-supply policy.

        Returns ``"skip"`` / ``"read_core"`` / ``"tools"`` (ADR-019 increment 2;
        heuristic lives in ``victor/agent/tool_supply_policy.py``).
        """
        return classify_tool_supply(context_msg, edge_check=self._check_tool_necessity_via_edge)

    def _should_skip_tools_for_turn(self, context_msg: str) -> bool:
        """Back-compat shim: True when full tool selection is not needed this turn.

        Prefer :meth:`_tool_skip_mode`, which distinguishes a hard ``"skip"`` (greeting)
        from ``"read_core"`` (borderline Q&A still gets read tools).
        """
        return self._tool_skip_mode(context_msg) != "tools"

    def _check_tool_necessity_via_edge(self, context_msg: str, heuristic_conf: float) -> bool:
        """Consult the service-backed tool-supply policy's edge decision."""
        return should_skip_tools_via_edge(
            context_msg,
            heuristic_conf,
            container=getattr(self, "_container", None),
        )

    def _sort_tools_for_kv_stability(self, tools):
        """Sort tools for KV-cache stability; session-local cache is managed here."""
        from victor.agent.services.tool_strategy_runtime import ToolStrategyRuntime

        return ToolStrategyRuntime(self).sort_tools_for_kv_stability(tools)

    def _should_stabilize_tool_order(self) -> bool:
        """Return whether this turn should use byte-stable tool ordering."""
        from victor.agent.services.tool_strategy_runtime import ToolStrategyRuntime

        return ToolStrategyRuntime(self).should_stabilize_tool_order()

    def _resolve_kv_strategy_setting(self) -> str:
        """Read kv_tool_strategy from settings; default to 'context_aware'."""
        from victor.agent.services.tool_strategy_runtime import ToolStrategyRuntime

        return ToolStrategyRuntime(self).resolve_kv_strategy_setting()

    def _apply_kv_tool_strategy(self, tools):
        """Delegate configured KV strategy and session caching to the strategy runtime."""
        from victor.agent.services.tool_strategy_runtime import ToolStrategyRuntime

        return ToolStrategyRuntime(self).apply_kv_tool_strategy(tools)

    def _context_aware_profile_session_locked(self) -> bool:
        """Whether the provider-economics profile wants session-stable tools."""
        from victor.agent.services.tool_strategy_runtime import ToolStrategyRuntime

        return ToolStrategyRuntime(self).context_aware_profile_session_locked()

    def _is_tool_strategy_v2_enabled(self) -> bool:
        """Check if the context-aware tool strategy (v2) is enabled.

        Gated solely by the ``tool_strategy_v2`` feature flag. The former
        ``tool_strategy_v2_enabled`` settings field was an unreachable,
        exception-only fallback (defaulted False) and has been removed.
        """
        from victor.agent.services.tool_strategy_runtime import ToolStrategyRuntime

        return ToolStrategyRuntime(self).is_tool_strategy_v2_enabled()

    def _apply_context_aware_strategy(self, tools):
        """Delegate context-aware selection and its telemetry to the strategy runtime."""
        from victor.agent.services.tool_strategy_runtime import ToolStrategyRuntime

        return ToolStrategyRuntime(self).apply_context_aware_strategy(tools)

    def _get_context_window(self, provider, model: str) -> int:
        """Get context window size for provider/model.

        Args:
            provider: Provider instance
            model: Model identifier

        Returns:
            Context window in tokens
        """
        from victor.agent.services.tool_strategy_runtime import ToolStrategyRuntime

        return ToolStrategyRuntime(self).get_context_window(provider, model)

    def _estimate_tool_tokens(
        self, tool, provider_category: str = None, _use_cache: bool = True
    ) -> int:
        """Delegate token estimation to ToolService."""
        from victor.agent.services.tool_strategy_runtime import ToolStrategyRuntime

        return ToolStrategyRuntime(self).estimate_tool_tokens(
            tool, provider_category, _use_cache=_use_cache
        )

    def _demote_tools_to_fit(
        self, tools, max_tokens: int, context_window: int, provider_category: str = None
    ) -> list:
        """Demote/drop low-priority tools to fit budget — delegates to the pure policy.

        See ``victor/agent/tool_supply_policy.py`` (ADR-019 increment 2).
        """
        return demote_tools_to_fit(
            tools,
            max_tokens,
            context_window,
            estimate_tokens=self._estimate_tool_tokens,
            provider_category=provider_category,
            stub_estimate_tokens=lambda t, pc=None: self._estimate_tool_tokens(
                t, pc, _use_cache=False
            ),
        )

    def _semantic_select_tools(self, tools, max_tokens: int, provider_category: str = None) -> list:
        """Delegate semantic tool selection to ToolService."""
        from victor.agent.services.tool_strategy_runtime import ToolStrategyRuntime

        return ToolStrategyRuntime(self).semantic_select_tools(tools, max_tokens, provider_category)

    def _emit_tool_strategy_event(
        self,
        strategy: str,
        tool_count: int,
        tool_tokens: int,
        context_window: int,
        provider: Any,
        reason: str,
        tools=None,
    ):
        """Delegate tool-strategy telemetry to the strategy runtime."""
        from victor.agent.services.tool_strategy_runtime import ToolStrategyRuntime

        ToolStrategyRuntime(self).emit_tool_strategy_event(
            strategy,
            tool_count,
            tool_tokens,
            context_window,
            provider,
            reason,
            tools,
        )

    # =====================================================================
    # Recovery Coordination Helper
    # =====================================================================

    def _get_recovery_runtime(self) -> Any:
        """Get the canonical service-owned recovery runtime helper."""
        runtime = getattr(self, "_recovery_runtime", None)
        if runtime is None:
            from victor.agent.services.recovery_runtime import RecoveryRuntime

            runtime = RecoveryRuntime(self.protocol_adapter)
            self._recovery_runtime = runtime
        return runtime

    def _get_tool_execution_runtime(self) -> Any:
        """Get the canonical service-owned tool execution runtime helper."""
        runtime = getattr(self, "_tool_execution_runtime", None)
        if runtime is None:
            from victor.agent.services.tool_execution_runtime import (
                ToolExecutionRuntime,
            )

            runtime = ToolExecutionRuntime(
                self.protocol_adapter,
                execution=self.protocol_adapter,
                publication=self.protocol_adapter,
                runtime_context=self._runtime_identity.resolve,
            )
            self._tool_execution_runtime = runtime
        return runtime

    def create_recovery_context(
        self,
        stream_ctx: "StreamingChatContext",
    ) -> Any:
        """Create RecoveryContext from current orchestrator state.

        Helper method to construct RecoveryContext for all recovery-related
        method calls. Centralizes context creation to avoid duplication.

        Args:
            stream_ctx: Streaming context

        Returns:
            StreamingRecoveryContext with all necessary state
        """
        return self._get_recovery_runtime().create_recovery_context(stream_ctx)

    # =====================================================================
    # Recovery Facade Methods
    # These methods create RecoveryContext from StreamingChatContext and
    # delegate to the appropriate coordinator. They centralize context
    # creation and provide a clean API for stream_chat.
    # =====================================================================

    async def _handle_recovery_with_integration(
        self,
        stream_ctx: StreamingChatContext,
        full_content: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        mentioned_tools: Optional[List[str]] = None,
    ) -> "OrchestratorRecoveryAction":
        """Handle response using the recovery coordinator.

        Creates RecoveryContext and delegates to recovery_coordinator viahandle_recovery_with_integration() directly.


        Args:
            stream_ctx: The streaming context
            full_content: Full response content
            tool_calls: Tool calls made (if any)
            mentioned_tools: Tools mentioned but not called

        Returns:
            RecoveryAction with action to take (continue, retry, abort, force_summary)
        """
        return await self._get_recovery_runtime().handle_recovery_with_integration(
            stream_ctx,
            full_content,
            tool_calls,
            mentioned_tools,
        )

    def _apply_recovery_action(
        self,
        recovery_action: "OrchestratorRecoveryAction",
        stream_ctx: StreamingChatContext,
    ) -> Optional[StreamChunk]:
        """Apply a recovery action using the recovery coordinator.

        Creates RecoveryContext and delegates to recovery_coordinator viaapply_recovery_action() directly.


        Args:
            recovery_action: The recovery action to apply
            stream_ctx: The streaming context

        Returns:
            StreamChunk if action requires immediate yield, None otherwise
        """
        return self._get_recovery_runtime().apply_recovery_action(
            recovery_action,
            stream_ctx,
        )

    def _parse_and_validate_tool_calls(
        self,
        tool_calls: Optional[List[Dict[str, Any]]],
        full_content: str,
    ) -> tuple[Optional[List[Dict[str, Any]]], str]:
        """Parse, validate, and normalize tool calls from provider response.

        Delegates to ToolService which consolidates all tool-call processing.
        """
        return self._tool_service.parse_and_validate_tool_calls(
            tool_calls, full_content, self.tool_adapter
        )

    async def stream_chat(self, user_message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Stream a chat response through the canonical ChatService runtime.

        .. deprecated::
            Direct orchestrator access is deprecated. Use ChatService instead:
            - From Agent: Use Agent.stream() on the framework surface
            - Direct access: Use ChatService.stream_chat() from the service layer
            This method will be removed in v2.0.

        Args:
            user_message: User's input message

        Returns:
            AsyncIterator yielding StreamChunk objects with incremental response
        """
        import warnings

        warnings.warn(
            "Direct orchestrator.stream_chat() access is deprecated. "
            "Use ChatService.stream_chat() from the service layer instead. "
            "This method will be removed in v2.0.",
            DeprecationWarning,
            stacklevel=2,
        )

        async with _aclosing(self._chat_service.stream_chat(user_message, **kwargs)) as stream:
            async for chunk in stream:
                yield chunk

    async def execute_tool_with_retry(
        self, tool_name: str, tool_args: Dict[str, Any], context: Dict[str, Any]
    ) -> tuple[Any, bool, Optional[str]]:
        """Execute a tool with retry logic via the canonical tool runtime surface.

        Delegates to ToolService.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            context: Execution context

        Returns:
            Tuple of (result, success, error_message or None)
        """
        return await self._tool_service.execute_tool_with_retry(tool_name, tool_args, context)

    async def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls through the canonical tool runtime path.

        Delegates execution to ToolPipeline, then post-processes results via
        the canonical tool service path.

        Args:
            tool_calls: List of tool call requests

        Returns:
            List of result dicts with keys: name, success, elapsed, args,
            error, follow_up_suggestions.
        """
        return await self._get_tool_execution_runtime().execute_tool_calls(tool_calls)

    def _get_tool_context(self) -> dict:
        """Get cached tool execution context dict.

        Reuses the same dict across tool calls within a conversation turn
        to avoid repeated allocation in the hot path.
        """
        if self._tool_context_cache is None:
            self._tool_context_cache = {
                "code_manager": self.code_manager,
                "provider": self.provider,
                "model": self.model,
                "tool_registry": self.tools,
                "workflow_registry": self.workflow_registry,
                "settings": self.settings,
                "current_intent": getattr(self, "_current_intent", None),
                "user_message": getattr(self, "_current_user_message", None),
            }
        runtime_overrides = getattr(self, "_runtime_tool_context_overrides", None)
        if not isinstance(runtime_overrides, dict) or not runtime_overrides:
            return self._tool_context_cache

        merged_context = dict(self._tool_context_cache)
        merged_context.update(runtime_overrides)
        return merged_context

    def reset_conversation(self) -> None:
        """Clear conversation history and session state.

        Delegates to LifecycleManager for core reset logic.

        Resets:
        - Conversation history
        - Tool call counter
        - Failed tool signatures cache
        - Observed files list
        - Executed tools list
        - Conversation state machine
        - Context reminder manager
        - Metrics collector stats
        - Context compactor statistics
        - Sequence tracker history (preserves learned patterns)
        - Usage analytics session (ends current, starts fresh)
        """
        require_idle_turn(self._session_accessor.stream_turn_lock)
        # Delegate to LifecycleManager for core reset
        self._lifecycle_manager.reset_conversation()

        # Reset orchestrator-specific state
        self._system_added = False
        self.tool_calls_used = 0
        self.failed_tool_signatures.clear()
        self.observed_files.clear()
        self.executed_tools.clear()
        self._consecutive_blocked_attempts = 0
        self._total_blocked_attempts = 0
        self._shown_tool_errors.clear()
        self._tool_context_cache = None

        logger.debug("Conversation and session state reset (via LifecycleManager)")

    def request_cancellation(self) -> None:
        """Request cancellation of the current streaming operation.

        Safe to call even if not streaming. The cancellation will take
        effect at the next check point in the stream loop.
        """
        if self._cancel_event:
            self._cancel_event.set()
            logger.info("Cancellation requested for streaming operation")

    def is_streaming(self) -> bool:
        """Check if a streaming operation is currently in progress.

        Returns:
            True if streaming, False otherwise.
        """
        return self._is_streaming

    def _check_cancellation(self) -> bool:
        """Check if cancellation has been requested.

        Returns:
            True if cancelled, False otherwise.
        """
        if self._cancel_event and self._cancel_event.is_set():
            return True
        return False

    # =========================================================================
    # Conversation Memory Management
    # =========================================================================

    def get_memory_session_id(self) -> Optional[str]:
        """Get the current memory session ID.

        Returns:
            Session ID or None if memory manager not enabled.
        """
        return self._memory_session_id

    def _sync_session_service_runtime_state(self) -> None:
        """Keep SessionService bound to the current live runtime state."""
        self._get_session_runtime().sync_runtime_state()

    def _get_session_runtime(self) -> Any:
        """Get the canonical service-owned session runtime helper."""
        runtime = getattr(self, "_session_runtime", None)
        if runtime is None:
            from victor.agent.services.session_runtime import SessionRuntime

            runtime = SessionRuntime(
                self.protocol_adapter,
                validate_session=self._runtime_identity.validate_session_transition,
            )
            self._session_runtime = runtime
        return runtime

    def _record_prompt_optimization_metadata(self, turn_context: Any) -> None:
        """Persist the active live prompt-optimization identity for the session."""
        self._get_session_runtime().record_prompt_optimization_metadata(turn_context)

    def get_active_prompt_optimization_metadata(self) -> Dict[str, Any]:
        """Return the canonical prompt-optimization metadata for the live session."""
        return self._get_session_runtime().get_active_prompt_optimization_metadata()

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation sessions for recovery.

        Delegates to SessionService.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata dictionaries
        """
        return self._get_session_runtime().get_recent_sessions(limit)

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous conversation session.

        Delegates to SessionService.

        Args:
            session_id: ID of the session to recover

        Returns:
            True if session was recovered successfully
        """
        return self._get_session_runtime().recover_session(session_id)

    def get_memory_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get token-aware context messages from memory manager.

        Delegates to SessionService.

        Args:
            max_tokens: Override max tokens for this retrieval.

        Returns:
            List of messages in provider format.
        """
        return self._get_session_runtime().get_memory_context(max_tokens=max_tokens)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current memory session.

        Delegates to SessionService.

        Returns:
            Dictionary with session statistics.
        """
        return self._get_session_runtime().get_session_stats()

    async def shutdown(self) -> None:
        """Clean up resources and shutdown gracefully.

        Delegates to LifecycleManager for core shutdown logic.

        Should be called when the orchestrator is no longer needed.
        Cleans up:
        - Background async tasks
        - Provider connections
        - Code execution manager (Docker containers)
        - Semantic selector resources
        - HTTP clients
        """
        # Delegate to LifecycleManager for shutdown
        await self._lifecycle_manager.shutdown()

        observability = getattr(self, "observability", None)
        event_bus = getattr(observability, "event_bus", None)
        if event_bus is not None and hasattr(event_bus, "disconnect"):
            try:
                await event_bus.disconnect()
            except Exception as e:
                logger.debug("Observability event bus shutdown skipped: %s", e)
        logger.debug("AgentOrchestrator shutdown complete")

    async def close(self) -> None:
        """Compatibility alias for shutdown."""
        await self.shutdown()

    # =========================================================================
    # Protocol Conformance Methods (OrchestratorProtocol)
    # =========================================================================
    # These methods implement the stable interface contract defined in
    # victor/framework/protocols.py, enabling the framework layer to
    # interact with the orchestrator without duck-typing.

    # --- ConversationStateProtocol ---

    # --- ToolsProtocol ---

    def get_available_tools(self) -> Set[str]:
        """Get all registered tool names (protocol method).

        Delegates to ToolService.

        Returns:
            Set of tool names available in registry
        """
        return self._tool_service.get_available_tools()

    def _build_tool_access_context(self) -> "ToolAccessContext":
        """Build ToolAccessContext for unified access control checks.

        Delegates to ToolService.

        Returns:
            ToolAccessContext with session tools and current mode
        """
        return self._tool_service.build_tool_access_context()

    def get_enabled_tools(self) -> Set[str]:
        """Get currently enabled tool names (protocol method).

        Delegates to ToolService.

        Returns:
            Set of enabled tool names for this session
        """
        return self._tool_service.get_enabled_tools()

    def set_enabled_tools(self, tools: Set[str], tiered_config: Any = None) -> None:
        """Set which tools are enabled for this session (protocol method).

        Delegates core logic to ToolService with orchestrator-specific propagation.

        Args:
            tools: Set of tool names to enable
            tiered_config: Optional TieredToolConfig to propagate for stage filtering.
        """
        self._enabled_tools = tools
        self._tool_service.set_enabled_tools(tools)

        # Apply to vertical context and tool access controller
        self._apply_vertical_tools(tools)

        # Propagate TieredToolConfig for stage-aware filtering
        if hasattr(self, "tool_selector") and self.tool_selector:
            if tiered_config is None:
                tiered_config = self._get_vertical_tiered_config()
            if tiered_config is not None:
                self.tool_selector.set_tiered_config(tiered_config)
                logger.debug(
                    f"Tiered config propagated to selector: "
                    f"mandatory={sorted(tiered_config.mandatory)}, "
                    f"vertical_core={sorted(tiered_config.vertical_core)}"
                )

    def _get_vertical_tiered_config(self) -> Any:
        """Get TieredToolConfig from active vertical canonical API."""
        from victor.agent.vertical_context import VerticalContext

        return VerticalContext.get_vertical_tiered_config()

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled (protocol method).

        Delegates to ToolService.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is enabled
        """
        return self._tool_service.is_tool_enabled(tool_name)

    # --- SystemPromptProtocol ---

    # --- Lifecycle Methods ---
    # Note: is_streaming() already exists at line ~5285
    # Note: reset() is provided via reset_conversation() at line ~5221

    def reset(self) -> None:
        """Reset conversation state (protocol method).

        Alias for reset_conversation() for protocol conformance.
        """
        self.reset_conversation()

    async def __aenter__(self) -> "AgentOrchestrator":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.shutdown()

    @classmethod
    async def from_settings(
        cls,
        settings: Settings,
        profile_name: str = "default",
        thinking: bool = False,
    ) -> "AgentOrchestrator":
        """Create orchestrator from settings. Delegates to orchestrator_creation module."""
        from victor.agent.orchestrator_creation import create_orchestrator_from_settings

        return await create_orchestrator_from_settings(
            cls, settings, profile_name=profile_name, thinking=thinking
        )


# Install extracted property descriptors onto the class.
# This runs once at import time and keeps full mock.patch.object compatibility.
from victor.agent.orchestrator_properties import (
    install_properties as _install_properties,
)

_install_properties(AgentOrchestrator)
del _install_properties
