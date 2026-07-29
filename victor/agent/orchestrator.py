# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

import ast
import asyncio
import inspect
import logging
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from rich.console import Console

# Coordinators (Phase 2 refactoring - being integrated)
# NOTE: These are runtime imports, not type-checking only
from victor.agent.services.metrics_service import (
    AgentMetricsService,
)  # noqa: F401  # imported for runtime use

if TYPE_CHECKING:
    # Type-only imports (created by factory, only used for type hints)
    from victor.agent.orchestrator_integration import OrchestratorIntegration
    from victor.agent.services.recovery_service import RecoveryService
    from victor.agent.services.chunk_runtime import ChunkGenerator
    from victor.agent.services.tool_planning_runtime import ToolPlanner
    from victor.agent.services.task_runtime import TaskCoordinator
    from victor.agent.protocols import ToolAccessContext
    from victor.evaluation.protocol import TokenUsage

    # Factory-created components (type hints only)
    from victor.agent.response_sanitizer import ResponseSanitizer
    from victor.agent.search_router import SearchRouter
    from victor.framework.task import TaskComplexityService as ComplexityClassifier
    from victor.agent.metrics_collector import MetricsCollector
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.tool_sequence_tracker import ToolSequenceTracker
    from victor.agent.recovery import RecoveryHandler
    from victor.agent.orchestrator_recovery import OrchestratorRecoveryIntegration
    from victor.agent.tool_output_formatter import ToolOutputFormatter
    from victor.agent.tool_pipeline import ToolPipeline
    from victor.runtime.context import RuntimeExecutionContext
    from victor.agent.streaming_controller import StreamingController
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.agent.tool_registrar import ToolRegistrar
    from victor.agent.provider_manager import ProviderManager
    from victor.agent.tool_selection import ToolSelector
    from victor.agent.tool_executor import ToolExecutor
    from victor.agent.safety import SafetyChecker
    from victor.agent.auto_commit import AutoCommitter
    from victor.agent.conversation_manager import (
        ConversationManager,
        create_conversation_manager,
    )

# Runtime imports - used for instantiation, enums, constants, or function calls
from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.message_history import MessageHistory
from victor.agent.conversation.store import ConversationStore
from victor.agent.prompt_prefix import apply_turn_prefix
from victor.agent.system_nudges import log_dropped_system_nudge

# DI container bootstrap
from victor.core.bootstrap import ensure_bootstrapped, get_service_optional
from victor.core.container import MetricsServiceProtocol, LoggerServiceProtocol

# Service protocols for DI resolution
from victor.agent.protocols import (
    ComplexityClassifierProtocol,
    ActionAuthorizerProtocol,
    SearchRouterProtocol,
    ProjectContextProtocol,
    ArgumentNormalizerProtocol,
    ConversationStateMachineProtocol,
    TaskTrackerProtocol,
    CodeExecutionManagerProtocol,
    WorkflowRegistryProtocol,
    UsageAnalyticsProtocol,
    ToolSequenceTrackerProtocol,
    ContextCompactorProtocol,
)
from victor.agent.services.protocols import ResponseSanitizerProtocol

# Mixins (used at class definition time)
from victor.protocols.mode_aware import ModeAwareMixin
from victor.agent.capability_registry import OrchestratorCapabilityMixin

# Config and enums (used at runtime)
from victor.config.config_loaders import get_provider_limits
from victor.agent.conversation_embedding_store import ConversationEmbeddingStore
from victor.agent.conversation.state_machine import (
    ConversationStateMachine,
    ConversationStage,
)
from victor.agent.action_authorizer import ActionIntent, INTENT_BLOCKED_TOOLS
from victor.agent.prompt_builder import get_task_type_hint, SystemPromptBuilder
from victor.agent.search_router import SearchRoute, SearchType
from victor.framework.task import TaskComplexity, DEFAULT_BUDGETS
from victor.agent.stream_handler import StreamMetrics
from victor.agent.metrics_collector import MetricsCollectorConfig
from victor.agent.unified_task_tracker import TrackerTaskType, UnifiedTaskTracker
from victor.agent.prompt_requirement_extractor import extract_prompt_requirements

# Decomposed components - configs, strategies, functions
from victor.core.context import session_id as ctx_session_id, set_session_id
from victor.agent.conversation.controller import (
    ConversationConfig,
    ContextMetrics,
    CompactionStrategy,
)
from victor.agent.context_compactor import (
    TruncationStrategy,
    create_context_compactor,
    calculate_parallel_read_budget,
)
from victor.agent.context_manager import (
    ContextManager,
    ContextManagerConfig,
    create_context_manager,
)
from victor.agent.continuation_strategy import ContinuationStrategy
from victor.agent.tool_call_extractor import ExtractedToolCall
from victor.agent.services.rl_runtime import get_rl_coordinator
from victor.agent.usage_analytics import AnalyticsConfig
from victor.agent.tool_sequence_tracker import create_sequence_tracker
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import (
    SessionStateManager,
    create_session_state_manager,
)

# Recovery - enums and functions used at runtime
from victor.agent.recovery import RecoveryOutcome, FailureType, RecoveryAction
from victor.agent.vertical_context import VerticalContext, create_vertical_context
from victor.agent.vertical_integration_adapter import VerticalIntegrationAdapter
from victor.agent.protocols import RecoveryHandlerProtocol
from victor.agent.orchestrator_recovery import (
    create_recovery_integration,
    RecoveryAction as OrchestratorRecoveryAction,
)
from victor.agent.tool_output_formatter import (
    ToolOutputFormatterConfig,
    create_tool_output_formatter,
)

# Pipeline - configs and results used at runtime
from victor.agent.tool_pipeline import ToolPipelineConfig, ToolCallResult
from victor.agent.streaming_controller import (
    StreamingControllerConfig,
    StreamingSession,
)
from victor.agent.task_analyzer import get_task_analyzer
from victor.agent.tool_registrar import ToolRegistrarConfig
from victor.agent.provider_manager import ProviderManagerConfig, ProviderState

# Observability
# from victor.observability.event_bus import EventBus, EventCategory, VictorEvent  # DELETED
from victor.observability.integration import ObservabilityIntegration
from victor.agent.orchestrator_integration import IntegrationConfig

# Tool execution - functions and enums
from victor.agent.tool_selection import get_critical_tools
from victor.agent.tool_calling import ToolCallParseResult
from victor.agent.tool_executor import ValidationMode
from victor.agent.orchestrator_utils import (
    calculate_max_context_chars,
    infer_git_operation,
    get_tool_status_message,
)
from victor.agent.orchestrator_factory import OrchestratorFactory
from victor.agent.parallel_executor import create_parallel_executor
from victor.agent.response_completer import (
    ToolFailureContext,
    create_response_completer,
)
from victor.observability.analytics.logger import UsageLogger
from victor.observability.analytics.streaming_metrics import StreamingMetricsCollector
from victor.storage.cache.tool_cache import ToolCache
from victor.config.model_capabilities import ToolCallingMatrix
from victor.config.settings import Settings
from victor.context.project_context import ProjectContext
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.registry import ProviderRegistry
from victor.core.errors import (
    ToolNotFoundError,
    ToolValidationError,
)
from victor.tools.enums import CostTier
from victor.tools.registry import ToolRegistry
from victor.tools.mcp_bridge_tool import get_mcp_tool_definitions
from victor.tools.plugin_registry import ToolPluginRegistry
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.tool_names import ToolNames, TOOL_ALIASES
from victor.tools.alias_resolver import get_alias_resolver
from victor.tools.progressive_registry import get_progressive_registry
from victor.workflows.base import WorkflowRegistry
from victor.workflows.discovery import register_builtin_workflows

# Streaming submodule - extracted for testability
from victor.agent.streaming import (
    CoordinatorConfig,
    ContinuationHandler,
    ContinuationResult,
    IntentClassificationHandler,
    IntentClassificationResult,
    IterationCoordinator,
    StreamingChatContext,
    StreamingChatHandler,
    ToolExecutionHandler,
    ToolExecutionResult,
    TrackingState,
    apply_tracking_state_updates,
    create_continuation_handler,
    create_coordinator,
    create_intent_classification_handler,
    create_stream_context,
    create_tool_execution_handler,
    create_tracking_state,
)

logger = logging.getLogger(__name__)

# Tools with progressive parameters - different params = progress, not a loop
# Format: tool_name -> list of param names that indicate progress
# NOTE: Includes both canonical short names and legacy names for LLM compatibility
# NOTE: These values are registered with ProgressiveToolsRegistry for extensibility
_PROGRESSIVE_TOOLS_CONFIG = {
    # Canonical short names
    "read": ["path", "offset", "limit"],
    "grep": ["query", "directory"],
    "search": ["query", "directory"],
    "ls": ["path", "recursive"],
    "shell": ["cmd"],
    "git": ["operation", "files", "branch"],
    "http": ["url", "method"],
    "web": ["query"],
    "summarize": ["query"],
    "fetch": ["url"],
    "code_search": ["query", "directory"],
    "semantic_code_search": ["query", "directory"],
    "http_request": ["url", "method"],
    "web_search": ["query"],
    "web_summarize": ["query"],
    "web_fetch": ["url"],
}


def _register_progressive_tools() -> None:
    """Register all progressive tools with the ProgressiveToolsRegistry.

    This function is idempotent and can be called multiple times safely.
    Tools are only registered if not already present in the registry.
    """
    registry = get_progressive_registry()
    for tool_name, params in _PROGRESSIVE_TOOLS_CONFIG.items():
        if not registry.is_progressive(tool_name):
            # Convert list of param names to dict format expected by registry
            progressive_params = dict.fromkeys(params, "any")
            registry.register(tool_name, progressive_params)


def _ensure_progressive_tools_registered() -> None:
    """Ensure progressive tools are registered in the registry.

    This is called lazily on first access to handle cases where
    the registry singleton was reset (e.g., during testing).
    """
    registry = get_progressive_registry()
    # Quick check if already registered by checking for one known tool
    if not registry.is_progressive("read"):
        _register_progressive_tools()


# Register tools at module load time
_register_progressive_tools()


class _ProgressiveToolsProxy:
    """Proxy class that delegates to ProgressiveToolsRegistry for backward compatibility.

    This allows existing code using PROGRESSIVE_TOOLS dict-like access to continue working
    while the actual data is managed by the registry.

    The proxy ensures tools are re-registered if the registry was reset (e.g., during tests).
    """

    def __getitem__(self, key: str) -> List[str]:
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        config = registry.get_config(key)
        if config is None:
            raise KeyError(key)
        return list(config.progressive_params.keys())

    def __contains__(self, key: str) -> bool:
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        return registry.is_progressive(key)

    def get(self, key: str, default: Any = None) -> Any:
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        config = registry.get_config(key)
        if config is None:
            return default
        return list(config.progressive_params.keys())

    def keys(self) -> Set[str]:
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        return registry.list_progressive_tools()

    def items(self):
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        for tool_name in registry.list_progressive_tools():
            config = registry.get_config(tool_name)
            if config:
                yield tool_name, list(config.progressive_params.keys())

    def __iter__(self):
        _ensure_progressive_tools_registered()
        registry = get_progressive_registry()
        return iter(registry.list_progressive_tools())


# Backward-compatible constant that delegates to the registry
PROGRESSIVE_TOOLS = _ProgressiveToolsProxy()

# Build set of all known tool names (canonical + aliases) for detection
_ALL_TOOL_NAMES: Set[str] = set()
for attr in dir(ToolNames):
    if not attr.startswith("_"):
        val = getattr(ToolNames, attr)
        if isinstance(val, str):
            _ALL_TOOL_NAMES.add(val)
_ALL_TOOL_NAMES.update(TOOL_ALIASES.keys())


def _detect_mentioned_tools(text: str) -> List[str]:
    """Detect tool names mentioned in text that model said it would call.

    Looks for patterns like:
    - "let me call read()"
    - "I'll use web_search to"
    - "calling the ls tool"
    - "execute grep"

    Args:
        text: Model response text

    Returns:
        List of mentioned tool names (canonical form)
    """
    import re

    mentioned: List[str] = []
    text_lower = text.lower()

    # Look for tool names followed by common patterns
    for tool_name in _ALL_TOOL_NAMES:
        # Match patterns like: call read, use read, execute read, run read
        # Also: read() or read( with args
        patterns = [
            rf"\b(?:call|use|execute|run|invoke|perform)\s+{re.escape(tool_name)}\b",
            rf"\b{re.escape(tool_name)}\s*\(",  # tool_name( or tool_name (
            rf"\bthe\s+{re.escape(tool_name)}\s+tool\b",  # "the read tool"
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower):
                # Resolve to canonical name
                canonical = TOOL_ALIASES.get(tool_name, tool_name)
                if canonical not in mentioned:
                    mentioned.append(canonical)
                break

    return mentioned


class AgentOrchestrator(ModeAwareMixin, OrchestratorCapabilityMixin):
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

    def _initialize_provider_runtime(self) -> None:
        """Initialize provider runtime boundaries with canonical service.

        Migration Note (2026-05-01):
        - ProviderCoordinator removed from runtime components (unused internally)
        - ProviderService is the canonical owner for provider operations
        """
        from victor.agent.runtime.provider_runtime import (
            create_provider_runtime_components,
        )
        from victor.agent.services.provider_service import ProviderService

        # [CANONICAL] ProviderService is the single authority for provider operations
        self._provider_service = ProviderService(registry=ProviderRegistry)

        self._provider_runtime = create_provider_runtime_components(
            settings=self.settings,
            provider_manager=self._provider_manager,
        )

    def _initialize_memory_runtime(self) -> None:
        """Initialize memory/session runtime boundaries with canonical service."""
        from victor.agent.runtime.memory_runtime import create_memory_runtime_components
        from victor.agent.services.session_service import SessionService

        self._memory_runtime = create_memory_runtime_components(
            factory=self._factory,
            provider_name=self.provider_name,
            native_tool_calls=self.tool_calling_caps.native_tool_calls,
        )
        self.memory_manager = self._memory_runtime.memory_manager
        self._memory_session_id = self._memory_runtime.memory_session_id

        # [CANONICAL] Initialize state-passed session service
        # Note: lifecycle_manager is None initially - will be set by component_assembler
        self._session_service = SessionService(
            session_state_manager=self._session_state,
            lifecycle_manager=None,  # Will be set later by component_assembler
            memory_manager=self.memory_manager,
            checkpoint_manager=getattr(self, "_checkpoint_manager", None),
            cost_tracker=getattr(self, "_cost_tracker", None),
        )

        # Initialize LanceDB embedding store for efficient semantic retrieval if memory enabled.
        if self.memory_manager and getattr(self.settings, "conversation_embeddings_enabled", True):
            try:
                self._init_conversation_embedding_store()
            except ImportError as embed_err:
                logger.debug(f"ConversationEmbeddingStore dependencies not available: {embed_err}")
            except (OSError, IOError) as embed_err:
                logger.warning(
                    f"Failed to initialize ConversationEmbeddingStore (I/O error): {embed_err}"
                )
            except (ValueError, TypeError) as embed_err:
                logger.warning(
                    f"Failed to initialize ConversationEmbeddingStore (config error): {embed_err}"
                )

    def _initialize_metrics_runtime(self) -> None:
        """Initialize metrics/analytics runtime boundaries."""
        from victor.agent.runtime.metrics_runtime import (
            create_metrics_runtime_components,
        )

        self._metrics_runtime = create_metrics_runtime_components(
            factory=self._factory,
            provider=self.provider,
            model=self.model,
            debug_logger=self.debug_logger,
            cumulative_token_usage=self._cumulative_token_usage,
            tool_cost_lookup=lambda name: (
                self.tools.get_tool_cost(name) if hasattr(self, "tools") else CostTier.FREE
            ),
            settings=self.settings,
        )
        self.usage_logger = self._metrics_runtime.usage_logger

        # Wrap with trace enricher for GEPA ASI when prompt optimization enabled
        try:
            from victor.observability.analytics.trace_enrichment import (
                create_trace_enricher,
            )

            po_cfg = getattr(self.settings, "prompt_optimization", None)
            gepa_cfg = po_cfg.gepa if po_cfg and po_cfg.enabled else None
            self.usage_logger = create_trace_enricher(self.usage_logger, gepa_settings=gepa_cfg)
        except Exception as exc:
            logger.debug("Trace enrichment unavailable: %s", exc)

        self.streaming_metrics_collector = self._metrics_runtime.streaming_metrics_collector
        self._metrics_collector = self._metrics_runtime.metrics_collector
        self._session_cost_tracker = self._metrics_runtime.session_cost_tracker
        self._metrics_coordinator = self._metrics_runtime.metrics_coordinator

        self.usage_logger.log_event(
            "session_start",
            {"model": self.model, "provider": self.provider.__class__.__name__},
        )
        if self.streaming_metrics_collector:
            logger.debug("StreamingMetricsCollector initialized via runtime boundary")

    def _initialize_workflow_runtime(self) -> None:
        """Initialize workflow runtime boundaries with lazy registry loading."""
        from victor.agent.runtime.workflow_runtime import (
            create_workflow_runtime_components,
        )

        self._workflow_runtime = create_workflow_runtime_components(factory=self._factory)
        self._workflow_registry = self._workflow_runtime.workflow_registry

    def _initialize_coordination_runtime(self) -> None:
        """Initialize coordination runtime boundaries with lazy components."""
        from victor.agent.runtime.coordination_runtime import (
            create_coordination_runtime_components,
        )

        self._coordination_runtime = create_coordination_runtime_components(
            factory=self._factory,
        )
        self._recovery_coordinator = self._coordination_runtime.recovery_coordinator
        self._chunk_generator = self._coordination_runtime.chunk_generator
        self._tool_planner = self._coordination_runtime.tool_planner
        self._task_coordinator = self._coordination_runtime.task_coordinator
        self._coordination_advisor_runtime = self._coordination_runtime.coordination_advisor_runtime
        if hasattr(self, "_task_analyzer") and hasattr(
            self._task_analyzer, "set_coordination_runtime"
        ):
            self._task_analyzer.set_coordination_runtime(self._coordination_advisor_runtime)

    def _initialize_resilience_runtime(self, *, context_compactor: Any) -> None:
        """Initialize resilience runtime boundaries with lazy recovery components."""
        from victor.agent.runtime.resilience_runtime import (
            create_resilience_runtime_components,
        )

        self._resilience_runtime = create_resilience_runtime_components(
            factory=self._factory,
            context_compactor=context_compactor,
        )
        self._recovery_handler = self._resilience_runtime.recovery_handler
        self._recovery_integration = self._resilience_runtime.recovery_integration

    def _initialize_interaction_runtime(self) -> None:
        """Initialize interaction runtime boundaries with canonical services first."""
        from victor.agent.runtime.interaction_runtime import (
            create_interaction_runtime_components,
        )
        from victor.runtime.context import resolve_runtime_services

        self._interaction_runtime = create_interaction_runtime_components(
            enabled_tools=getattr(self, "_enabled_tools", None),
            factory=self._factory,
            context_compactor=getattr(self, "_context_compactor", None),
            tool_pipeline=self._tool_pipeline,
            tool_registry=self.tools,
            tool_executor=getattr(self, "tool_executor", None),
            tool_cache=getattr(self, "tool_cache", None),
            tool_budget=self.tool_budget,
            tool_selector=(self.tool_selector if hasattr(self, "tool_selector") else None),
            tool_access_controller=getattr(self, "_tool_access_controller", None),
            mode_controller=(self.mode_controller if hasattr(self, "mode_controller") else None),
            argument_normalizer=getattr(self, "argument_normalizer", None),
            session_state_manager=self._session_state,
            lifecycle_manager=self._lifecycle_manager,
            memory_manager=self.memory_manager,
            memory_session_id=self._memory_session_id,
            checkpoint_manager=self._checkpoint_manager,
            cost_tracker=self._session_cost_tracker,
            conversation_controller=self._conversation_controller,
            streaming_coordinator=self._streaming_controller,
            runtime_services=resolve_runtime_services(self),
        )
        self._chat_service = self._interaction_runtime.chat_service
        self._tool_service = self._interaction_runtime.tool_service
        self._session_service = self._interaction_runtime.session_service
        self._context_service = self._interaction_runtime.context_service
        self._recovery_service = self._interaction_runtime.recovery_service

    def _initialize_credit_runtime(self) -> None:
        """Initialize credit assignment runtime (FEP-0001 Phase 3).

        Creates CreditTrackingService and attaches it to the ToolPipeline.
        Gated by settings.credit_assignment.enabled (default: False).
        """
        self._credit_tracking_service = None
        ca_settings = getattr(self.settings, "credit_assignment", None)
        if ca_settings is None or not getattr(ca_settings, "enabled", False):
            return

        try:
            from victor.framework.rl.credit_tracking_service import (
                CreditTrackingService,
            )
            from victor.core.container import ServiceLifetime

            obs_bus = getattr(self, "_observability_bus", None)
            self._credit_tracking_service = CreditTrackingService.from_settings(
                self.settings, observability_bus=obs_bus
            )

            # Attach to ToolPipeline
            if hasattr(self, "_tool_pipeline") and self._tool_pipeline is not None:
                self._tool_pipeline._credit_tracking_service = self._credit_tracking_service

            # Register the runtime service in DI so prompt-time and learner-time
            # consumers see the same live credit history instance.
            self._container.register_or_replace(
                CreditTrackingService,
                lambda c: self._credit_tracking_service,
                ServiceLifetime.SINGLETON,
            )

            logger.debug("Credit tracking service initialized")
        except Exception as e:
            logger.warning("Failed to initialize credit tracking: %s", e)

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
            from victor.agent.services.tool_service import (
                ToolService,
                ToolServiceConfig,
            )

            self._tool_service = ToolService(
                config=ToolServiceConfig(default_tool_budget=getattr(self, "tool_budget", 100)),
                tool_selector=(self.tool_selector if hasattr(self, "tool_selector") else None),
                tool_executor=getattr(self, "tool_executor", None),
                tool_registrar=getattr(self, "tools", None),
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

            chat_stream_adapter = self._get_chat_stream_adapter()
            self._chat_service.bind_runtime_components(
                turn_executor=LazyRuntimeProxy(
                    factory=lambda: self.turn_executor,
                    name="turn_executor",
                ),
                planning_handler=self._get_planning_chat_runtime().run,
                stream_chat_handler=chat_stream_adapter.stream_chat,
                context_limit_handler=self._get_context_limit_runtime().handle_limits,
                task_report_start_handler=self._start_task_report,
                task_report_finish_handler=self._finish_task_report,
                turn_setup_handler=self._prepare_chat_service_turn_runtime,
                turn_teardown_handler=self._teardown_chat_service_turn_runtime,
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
        elif self.project_context.content:
            self._system_prompt = self._get_prompt_builder_runtime().compose_system_prompt(
                base_system_prompt
            )
            logger.debug(f"Loaded project context from {self.project_context.context_file}")
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

    def get_capability_config_scope_key(self) -> Optional[str]:
        """Get stable scope key for framework capability-config service storage."""
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

    def get_vertical_context(self) -> VerticalContext:
        """Get vertical context via explicit protocol getter."""
        return self._vertical_context

    @property
    def lsp(self) -> Optional[Any]:
        """Get the LSP capability for code intelligence.

        Returns:
            LSPCapability instance or None if not configured.
        """
        return getattr(self, "_lsp", None)

    def set_lsp(self, lsp_capability: Any) -> None:
        """Set the LSP capability (LSPServiceProtocol/LSPPoolProtocol).

        When a capability is provided, the LSP diagnostics middleware
        (FEP-0019) auto-activates on the tool middleware chain.
        """
        self._lsp = lsp_capability
        if lsp_capability is not None and getattr(self, "_middleware_chain", None):
            from victor.framework.lsp_middleware import register_lsp_on_chain

            register_lsp_on_chain(self, self._middleware_chain)
        logger.debug("LSP capability registered with orchestrator")

    def get_team_suggestions(
        self,
        task_type: str,
        complexity: str,
    ) -> Any:
        """Deprecated compatibility alias for ``get_coordination_suggestion()``.

        Returns coordination suggestions for a task while preserving the older
        team-centric method name used by legacy callers.
        """
        import warnings

        warnings.warn(
            "AgentOrchestrator.get_team_suggestions(...) is deprecated. "
            "Use AgentOrchestrator.get_coordination_suggestion(...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_coordination_suggestion(task_type, complexity)

    def get_coordination_suggestion(
        self,
        task_type: str,
        complexity: str,
        *,
        mode: Optional[str] = None,
    ) -> Any:
        """Get canonical coordination suggestions for a task.

        Prefers the service-owned coordination runtime, which delegates to the
        shared framework recommendation engine.

        Args:
            task_type: Classified task type (e.g., "feature", "bugfix", "refactor")
            complexity: Complexity level (e.g., "low", "medium", "high", "extreme")
            mode: Optional explicit mode override. Defaults to runtime mode.

        Returns:
            CoordinationSuggestion with team and workflow recommendations
        """
        coordination_runtime = getattr(self, "_coordination_advisor_runtime", None)
        if coordination_runtime is None:
            factory = getattr(self, "_factory", None)
            if factory is not None and hasattr(factory, "create_coordination_advisor_runtime"):
                coordination_runtime = factory.create_coordination_advisor_runtime()
            else:
                from victor.agent.services.coordination_advisor_runtime import (
                    CoordinationAdvisorRuntime,
                )

                coordination_runtime = CoordinationAdvisorRuntime()

            self._coordination_advisor_runtime = coordination_runtime

        mode_controller = getattr(self, "mode_controller", None)
        current_mode = getattr(mode_controller, "current_mode", None)
        resolved_mode = mode or getattr(current_mode, "value", None) or "build"

        return coordination_runtime.suggest_for_task(
            runtime_subject=self,
            task_type=task_type,
            complexity=complexity,
            mode=resolved_mode,
        )

    # =========================================================================
    # Vertical Protocol Methods
    # These implement OrchestratorVerticalProtocol for proper vertical integration
    # =========================================================================

    def set_vertical_context(self, context: VerticalContext) -> None:
        """Set the vertical context (OrchestratorVerticalProtocol).

        This replaces direct assignment to _vertical_context and provides
        a proper API for framework integration.

        Args:
            context: VerticalContext to set
        """
        self._vertical_context = context

        # Sync coordination advisor with new vertical context (if already initialized)
        if getattr(self, "_coordination_advisor", None) is not None:
            self._coordination_advisor.set_vertical_context(context)
            logger.debug(
                "Coordination advisor synced with vertical context: %s",
                context.vertical_name,
            )

        # Sync tool selector with vertical context for vertical-specific tool selection (DIP)
        if hasattr(self, "tool_selector") and self.tool_selector is not None:
            self.tool_selector.set_vertical_context(context)
            if context.has_tool_selection_strategy:
                logger.debug(
                    f"Tool selector synced with vertical tool selection strategy: "
                    f"{context.vertical_name}"
                )

        # Sync middleware chain with vertical context for vertical-aware middleware (DIP)
        if hasattr(self, "_middleware_chain") and self._middleware_chain is not None:
            self._middleware_chain.set_vertical_context(context)
            logger.debug(f"Middleware chain synced with vertical context: {context.vertical_name}")

        logger.debug(f"Vertical context set: {context.vertical_name}")

    def set_tiered_tool_config(self, config: Any) -> None:
        """Set tiered tool configuration (Phase 1: Gap fix).

        Applies tiered tool config from vertical to:
        1. VerticalContext for storage
        2. ToolAccessController.VerticalLayer for access filtering

        Args:
            config: TieredToolConfig from the active vertical
        """
        # Store in vertical context
        if self._vertical_context is not None:
            self._vertical_context.apply_tiered_config(config)

        # Apply to tool access controller
        if self._tool_access_controller is not None:
            self._tool_access_controller.set_tiered_config(config)
            logger.debug("Tiered config applied to ToolAccessController")

        logger.debug("Tiered tool config set")

    def set_workspace(self, workspace_dir: Path) -> None:
        """Set the workspace directory for task execution.

        Updates the global project root and orchestrator's project context
        to work in the specified directory. This is essential for benchmark
        evaluations where each task operates in a different workspace.

        Args:
            workspace_dir: Path to the workspace directory
        """
        from victor.config.settings import set_project_root
        from victor.context.project_context import ProjectContext

        # Update global project root
        set_project_root(workspace_dir)
        logger.debug(f"Project root set to: {workspace_dir}")

        # Create new project context for this workspace
        self.project_context = ProjectContext(root_path=str(workspace_dir))
        self.project_context.load()

        # Set RL repo context for per-repo isolation
        try:
            from victor.agent.services.rl_runtime import get_rl_coordinator

            coordinator = get_rl_coordinator()
            coordinator.set_repo_context(workspace_dir.name)
            # Push the session's actual provider/model into GEPA so prompt
            # optimization uses the right provider (e.g., zai/glm-5.2 from
            # -p zai-coding), not the global settings default (ollama).
            coordinator.set_session_provider(
                getattr(self, "provider_name", None) or getattr(self.provider, "name", ""),
                self.model,
            )
        except Exception as exc:
            logger.debug("RL coordinator repo context unavailable: %s", exc)

        # Rebuild system prompt with new workspace context (replaces old init.md)
        # Workspace switch is a session reset — unfreeze and re-sample GEPA sections
        if getattr(self, "_prompt_pipeline", None):
            self._prompt_pipeline.unfreeze()
        self._system_prompt_frozen = False

        base_prompt = self.build_system_prompt()
        if self.project_context.content:
            self._system_prompt = self._get_prompt_builder_runtime().compose_system_prompt(
                base_prompt
            )
            logger.debug(f"Loaded project context from {self.project_context.context_file}")
        else:
            self._system_prompt = base_prompt

        if self._kv_optimization_enabled:
            self._system_prompt_frozen = True
        if self._cache_optimization_enabled:
            self._session_tools = None  # Re-lock tools for new workspace

        # Clear conversation history to remove old workspace context
        # This ensures the model doesn't see init.md from the previous project
        if hasattr(self, "reset_conversation"):
            self.reset_conversation()

    def _apply_vertical_tools(self, tools: Set[str]) -> None:
        """Apply enabled tools to vertical context and access controller.

        Internal helper called by set_enabled_tools. Separated to avoid
        duplication and maintain single method for protocol compliance.

        Args:
            tools: Set of tool names to enable
        """
        self._vertical_context.apply_enabled_tools(tools)
        logger.debug(f"Applied {len(tools)} tools to vertical context")

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

    async def _prepare_runtime_intelligence_request(
        self, task: str, task_type: str
    ) -> Optional[Dict[str, Any]]:
        """Pre-request hook for runtime-intelligence integration.

        Delegates to OrchestratorIntegration.prepare_runtime_intelligence_request().

        Args:
            task: The user's task/query
            task_type: Detected task type (analysis, edit, etc.)

        Returns:
            Dictionary with recommendations, or None if pipeline disabled
        """
        integration = self.runtime_intelligence_integration
        if not integration:
            return None

        return await integration.prepare_runtime_intelligence_request(
            task=task,
            task_type=task_type,
            conversation_state=self.conversation_state,
            unified_tracker=self.unified_tracker,
        )

    async def _validate_runtime_intelligence_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Post-response hook for runtime-intelligence integration.

        Delegates to OrchestratorIntegration.validate_runtime_intelligence_response().

        Args:
            response: The model's response content
            query: Original user query
            tool_calls: Number of tool calls made so far
            task_type: Task type for context

        Returns:
            Dictionary with quality/grounding scores, or None if pipeline disabled
        """
        integration = self.runtime_intelligence_integration
        if not integration:
            return None

        return await integration.validate_runtime_intelligence_response(
            response=response,
            query=query,
            tool_calls=tool_calls,
            task_type=task_type,
        )

    def _record_runtime_intelligence_outcome(
        self,
        success: bool,
        quality_score: float = 0.5,
        user_satisfied: bool = True,
        completed: bool = True,
    ) -> None:
        """Record outcome for Q-learning feedback.

        Delegates to OrchestratorIntegration.record_runtime_intelligence_outcome().

        Args:
            success: Whether the task was completed successfully
            quality_score: Final quality score (0.0-1.0)
            user_satisfied: Whether user seemed satisfied
            completed: Whether task reached completion
        """
        integration = self.runtime_intelligence_integration
        if not integration:
            return

        # Get orchestrator state to pass to integration
        stream_context = getattr(self, "_current_stream_context", None)
        continuation_prompts = getattr(self, "_continuation_prompts", 0)
        max_continuation_prompts_used = getattr(self, "_max_continuation_prompts_used", 6)
        stuck_loop_detected = getattr(self, "_stuck_loop_detected", False)

        try:
            integration.record_runtime_intelligence_outcome(
                success=success,
                quality_score=quality_score,
                user_satisfied=user_satisfied,
                completed=completed,
                rl_coordinator=self._rl_coordinator,
                stream_context=stream_context,
                vertical_context=self._vertical_context,
                provider_name=self.provider.name,
                model=self.model,
                tool_calls_used=self.tool_calls_used,
                continuation_prompts=continuation_prompts,
                max_continuation_prompts_used=max_continuation_prompts_used,
                stuck_loop_detected=stuck_loop_detected,
            )
        except Exception as e:
            logger.debug(f"RuntimeIntelligencePipeline record_outcome failed: {e}")

        # EvoTest: trigger session-end GEPA evolution (E12)
        if self.tool_calls_used >= 3:
            try:
                rl_coord = getattr(self, "_rl_coordinator", None)
                if rl_coord and hasattr(rl_coord, "try_evolve_on_session_end"):
                    result = rl_coord.try_evolve_on_session_end(
                        getattr(self.provider, "name", "unknown"), self.model
                    )
                    if result and isinstance(result, dict):
                        # Display evolution report via debug logger
                        debug_log = getattr(self, "debug_logger", None)
                        if debug_log and hasattr(debug_log, "log_evolution_report"):
                            debug_log.log_evolution_report(result)
            except Exception as e:
                logger.debug(f"[evotest] Session-end evolution failed: {e}")

    def _should_continue_runtime_intelligence(self) -> tuple[bool, str]:
        """Check if processing should continue using learned behaviors.

        Delegates to OrchestratorIntegration.should_continue_runtime_intelligence().

        Returns:
            Tuple of (should_continue, reason)
        """
        integration = self.runtime_intelligence_integration
        if not integration:
            return True, "Pipeline disabled"

        return integration.should_continue_runtime_intelligence()

    # Backward-compatible aliases while runtime-intelligence naming propagates.
    async def _prepare_intelligent_request(
        self, task: str, task_type: str
    ) -> Optional[Dict[str, Any]]:
        return await self._prepare_runtime_intelligence_request(task, task_type)

    async def _validate_intelligent_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        return await self._validate_runtime_intelligence_response(
            response=response,
            query=query,
            tool_calls=tool_calls,
            task_type=task_type,
        )

    def _record_intelligent_outcome(
        self,
        success: bool,
        quality_score: float = 0.5,
        user_satisfied: bool = True,
        completed: bool = True,
    ) -> None:
        self._record_runtime_intelligence_outcome(
            success=success,
            quality_score=quality_score,
            user_satisfied=user_satisfied,
            completed=completed,
        )

    def _should_continue_intelligent(self) -> tuple[bool, str]:
        return self._should_continue_runtime_intelligence()

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

    def apply_vertical_middleware(self, middleware_list: List[Any]) -> None:
        """Apply middleware from vertical extensions.

        Called by FrameworkShim after orchestrator creation to inject
        vertical-specific middleware. This enables the middleware chain
        pattern for tool execution.

        Delegates to VerticalIntegrationAdapter for single-source implementation.

        Args:
            middleware_list: List of MiddlewareProtocol implementations.
        """
        self._vertical_integration_adapter.apply_middleware(middleware_list)

    def apply_vertical_safety_patterns(self, patterns: List[Any]) -> None:
        """Apply safety patterns from vertical extensions.

        Called by FrameworkShim to inject vertical-specific danger patterns
        into the safety checker.

        Delegates to VerticalIntegrationAdapter for single-source implementation.

        Args:
            patterns: List of SafetyPattern objects.
        """
        self._vertical_integration_adapter.apply_safety_patterns(patterns)

    def get_middleware_chain(self) -> Optional[Any]:
        """Get the middleware chain for tool execution.

        Returns:
            MiddlewareChain instance or None if not initialized.
        """
        return getattr(self, "_middleware_chain", None)

    def set_middleware_chain(self, chain: Any) -> None:
        """Store middleware chain via public runtime port."""
        self._middleware_chain = chain
        # If LSP was set before the chain existed, register its middleware now.
        if getattr(self, "_lsp", None) is not None:
            from victor.framework.lsp_middleware import register_lsp_on_chain

            register_lsp_on_chain(self, chain)

    # =========================================================================
    # Internal Storage Setters (DIP Compliance)
    # These methods provide controlled access for adapter implementations,
    # replacing direct private attribute writes. Only called by
    # VerticalIntegrationAdapter - not for general use.
    # =========================================================================

    def _set_vertical_middleware_storage(self, middleware: List[Any]) -> None:
        """Internal: Set vertical middleware storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            middleware: List of middleware instances
        """
        self.set_middleware(middleware)

    def _set_middleware_chain_storage(self, chain: Any) -> None:
        """Internal: Set middleware chain storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            chain: MiddlewareChain instance
        """
        self.set_middleware_chain(chain)

    def _set_safety_patterns_storage(self, patterns: List[Any]) -> None:
        """Internal: Set safety patterns storage.

        DIP Compliance: Provides controlled setter instead of direct
        private attribute access. Called by VerticalIntegrationAdapter.

        Args:
            patterns: List of safety pattern instances
        """
        self.set_safety_patterns(patterns)

    # =========================================================================
    # VerticalStorageProtocol Implementation (DIP Compliance)
    # These public methods implement VerticalStorageProtocol, providing a clean
    # interface for vertical data storage and retrieval. This replaces direct
    # private attribute access with protocol-compliant methods.
    # =========================================================================

    def set_middleware(self, middleware: List[Any]) -> None:
        """Store middleware configuration.

        Implements VerticalStorageProtocol.set_middleware().
        Provides a clean public interface for setting vertical middleware,
        replacing direct private attribute access.

        Args:
            middleware: List of MiddlewareProtocol implementations
        """
        self._vertical_middleware = middleware

    def get_middleware(self) -> List[Any]:
        """Retrieve middleware configuration.

        Implements VerticalStorageProtocol.get_middleware().
        Returns the list of middleware instances configured by vertical integration.

        Returns:
            List of middleware instances, or empty list if not set
        """
        return getattr(self, "_vertical_middleware", [])

    def set_safety_patterns(self, patterns: List[Any]) -> None:
        """Store safety patterns.

        Implements VerticalStorageProtocol.set_safety_patterns().
        Provides a clean public interface for setting vertical safety patterns,
        replacing direct private attribute access.

        Args:
            patterns: List of SafetyPattern instances from vertical extensions
        """
        self._vertical_safety_patterns = patterns

    def get_safety_patterns(self) -> List[Any]:
        """Retrieve safety patterns.

        Implements VerticalStorageProtocol.get_safety_patterns().
        Returns the list of safety patterns configured by vertical integration.

        Returns:
            List of safety pattern instances, or empty list if not set
        """
        return getattr(self, "_vertical_safety_patterns", [])

    def set_team_specs(self, specs: Dict[str, Any]) -> None:
        """Store team specifications.

        Implements VerticalStorageProtocol.set_team_specs().
        Provides a clean public interface for setting team specs,
        replacing direct private attribute access.

        Args:
            specs: Dictionary mapping team names to TeamSpec instances
        """
        self._team_specs = specs

    def get_team_specs(self) -> Dict[str, Any]:
        """Retrieve team specifications.

        Implements VerticalStorageProtocol.get_team_specs().
        Returns the dictionary of team specs configured by vertical integration.

        Returns:
            Dictionary of team specs, or empty dict if not set
        """
        return getattr(self, "_team_specs", {})

    @property
    def messages(self) -> List[Message]:
        """Get conversation messages (backward compatibility property).

        Returns:
            List of messages in conversation history
        """
        return self.conversation.messages

    def _get_conversation_message_count(self) -> int:
        """Return the current conversation length across history implementations."""
        conversation = getattr(self, "conversation", None)
        if conversation is None:
            return 0

        messages = getattr(conversation, "messages", None)
        if messages is not None:
            try:
                return len(messages)
            except TypeError:
                pass

        message_count = getattr(conversation, "message_count", None)
        if callable(message_count):
            try:
                return int(message_count())
            except TypeError:
                pass

        return 0

    def _get_model_context_window(self) -> int:
        """Get context window size for the current model.

        Delegates to ContextManager when available (TD-002 refactoring).
        Falls back to direct implementation during __init__ before
        ContextManager is created.

        Returns:
            Context window size in tokens
        """
        # Delegate to ContextManager if available
        if hasattr(self, "_context_manager") and self._context_manager is not None:
            return self._context_manager.get_model_context_window()

        # Fallback for calls during __init__ before ContextManager is created
        try:
            from victor.config.config_loaders import get_provider_limits

            limits = get_provider_limits(self.provider_name, self.model)
            return limits.context_window
        except Exception as e:
            logger.warning(f"Could not load provider limits from config: {e}")
            return 128000  # Default safe value

    def _get_max_context_chars(self) -> int:
        """Get maximum context size in characters.

        Delegates to ContextManager (TD-002 refactoring).

        Returns:
            Maximum context size in characters
        """
        return self._context_manager.get_max_context_chars()

    def _check_cache_setting_enabled(self) -> bool:
        """Check if cache optimization is enabled in settings."""
        return self._get_prompt_builder_runtime().check_cache_setting_enabled()

    @property
    def _cache_optimization_enabled(self) -> bool:
        """Full API-level cache optimization — session-lock all tools + freeze prompt.

        Uses cached flag from _compute_cache_flags() when available.
        """
        return self._get_prompt_builder_runtime().is_cache_optimization_enabled()

    def _compute_cache_flags(self) -> None:
        """Compute and cache optimization flags once (called after provider init).

        Caches _kv_opt_cached and _cache_opt_cached so the @property accessors
        don't repeat try/except/getattr/hasattr chains on every access.
        """
        self._get_prompt_builder_runtime().compute_cache_flags()

    @property
    def _kv_optimization_enabled(self) -> bool:
        """KV prefix cache optimization — freeze prompt for prefix reuse.

        Uses cached flag from _compute_cache_flags() when available.
        """
        return self._get_prompt_builder_runtime().is_kv_optimization_enabled()

    async def warm_up_kv_cache(self) -> None:
        """Prime the KV cache by sending a minimal request with the system prompt.

        For KV prefix caching providers (Ollama, LMStudio), the first API call
        is always cold — full prefill computation. This method sends a 1-token
        completion to prime the KV cache so subsequent calls reuse the prefix.

        No-op when KV optimization is disabled.
        """
        await self._get_prompt_builder_runtime().warm_up_kv_cache()

    def _kv_prefix_fingerprint(self) -> str:
        """Compute a short fingerprint of the system prompt for KV cache observability.

        Returns a hex hash of the first 500 chars of the system prompt. Identical
        hashes across turns indicate the KV prefix is stable (likely cache hit).
        """
        return self._get_prompt_builder_runtime().kv_prefix_fingerprint()

    def get_session_tools(self) -> Optional[list]:
        """Get session-locked tools for cache-friendly API calls.

        Returns the currently enabled tool set (frozen at session start) so the
        tool prefix remains byte-identical across API calls in the same session.
        When cache optimization is disabled, returns None (use per-turn).
        """
        if not self._cache_optimization_enabled:
            return None
        session_tools = getattr(self, "_session_tools", None)
        if session_tools is None:
            try:
                enabled_tools = self.get_enabled_tools()
                locked_tools = [
                    tool
                    for tool in self.tools.list_tools(only_enabled=True)
                    if getattr(tool, "name", None) in enabled_tools
                ]
                if locked_tools:
                    self._session_tools = locked_tools
                    logger.debug(
                        "[cache] Session tools locked: %d tools (prefix-stable)",
                        len(locked_tools),
                    )
                    return locked_tools
            except Exception:
                return None
        return session_tools

    def get_assembled_messages(
        self,
        current_query: Optional[str] = None,
        *,
        selected_tools: Optional[list[Any]] = None,
        planned_tools: Optional[list[Any]] = None,
        goals: Optional[list[str]] = None,
    ) -> List["Message"]:
        """Get context-assembled messages for provider calls.

        When context assembler is available: keeps system prompt + ledger +
        last N full turns + score-selected older messages within budget.
        Falls back to raw self.messages when assembler unavailable.

        When the canonical prompt pipeline is available, dynamic prompt guidance
        (skills, reminders, GEPA/MiPRO/CoT optimizations, credit guidance) is
        prepended to the last user message instead of being injected as system
        messages. For KV-capable providers this also keeps the system prompt
        byte-identical across turns for provider prefix caching.
        """
        self._get_prompt_builder_runtime().ensure_system_prompt_current()

        assembler = getattr(self, "_context_assembler", None)
        if assembler is None:
            messages = list(self.messages)
        else:
            max_chars = self._get_max_context_chars()
            messages = assembler.assemble(self.messages, max_chars, current_query=current_query)

            if len(messages) < len(self.messages):
                total_original = sum(len(m.content) for m in self.messages)
                total_assembled = sum(len(m.content) for m in messages)
                logger.debug(
                    "[context] Assembled %d/%d messages (%dK/%dK chars, budget=%dK)",
                    len(messages),
                    len(self.messages),
                    total_assembled // 1024,
                    total_original // 1024,
                    max_chars // 1024,
                )

        # Boundary guard: ensure all items are Message objects before sending
        # to the provider. Converts dicts/strings that may have leaked in from
        # session restore, compaction, or serialization round-trips.
        from victor.providers.base import Message as _Msg

        normalized = []
        for m in messages:
            if isinstance(m, _Msg):
                normalized.append(m)
            elif isinstance(m, dict) and "role" in m:
                normalized.append(_Msg(**{k: v for k, v in m.items() if k in _Msg.model_fields}))
            elif isinstance(m, str):
                normalized.append(_Msg(role="assistant", content=m))
            else:
                logger.debug("Dropping non-Message item from assembled messages: %s", type(m))
        if len(normalized) != len(messages):
            logger.debug(
                "[context] Normalized %d/%d non-Message items in assembled messages",
                len(messages) - len(normalized),
                len(messages),
            )
        messages = normalized

        # Log KV prefix fingerprint for observability
        if self._kv_optimization_enabled:
            logger.debug(
                "[kv-cache] prefix=%s frozen=%s tools_cached=%s",
                self._kv_prefix_fingerprint(),
                getattr(self, "_system_prompt_frozen", False),
                getattr(self, "_last_sorted_tool_names", None) is not None,
            )

        # Canonical per-turn prompt optimizations live on the prompt pipeline.
        # KV-capable providers additionally benefit from stable system-prompt
        # caching, but Tier C providers still need the same dynamic guidance.
        prefix = ""
        pipeline = getattr(self, "_prompt_pipeline", None)
        if messages and pipeline:
            # Use UnifiedPromptPipeline for per-turn prefix composition
            from victor.agent.prompt_pipeline import TurnContext, coerce_prompt_overlays

            # Build turn context for the pipeline
            injector = getattr(self, "_optimization_injector", None)
            turn_ctx = TurnContext(
                provider_name=self.provider_name or "",
                model=getattr(self, "_model", "") or "",
                task_type=getattr(self, "_current_task_type", "default"),
                active_skill_prompt=(
                    self.get_skill_user_prefix() if hasattr(self, "get_skill_user_prefix") else None
                ),
                last_turn_failed=(
                    injector._last_failure_category is not None if injector else False
                ),
                last_failure_category=(injector._last_failure_category if injector else None),
                last_failure_error=(injector._last_failure_error if injector else None),
                task_guidance_text=getattr(self, "_dynamic_task_guidance", None),
            )
            runtime_context_overrides = getattr(self, "_runtime_tool_context_overrides", None)
            if isinstance(runtime_context_overrides, dict):
                turn_ctx.prompt_overlays = coerce_prompt_overlays(
                    runtime_context_overrides.get("prompt_overlays")
                )

            # Get context reminders
            reminder_mgr = getattr(self, "_reminder_manager", None)
            if reminder_mgr:
                turn_ctx.reminder_text = reminder_mgr.get_user_message_prefix()

            # Get last user message text for KNN few-shot matching
            last_user_msg = ""
            for m in reversed(messages):
                if m.role == "user":
                    last_user_msg = m.content[:200]
                    break

            prompt_runtime = self._get_prompt_builder_runtime()
            turn_ctx.dynamic_tool_guidance = prompt_runtime.build_dynamic_tool_guidance(
                current_query or last_user_msg,
                goals=goals,
                planned_tools=planned_tools,
                selected_tools=selected_tools,
            )

            prefix = pipeline.compose_turn_prefix(last_user_msg, turn_ctx)
            self._record_prompt_optimization_metadata(turn_ctx)

            # Clear failure state after injecting hint
            if injector and turn_ctx.last_turn_failed:
                injector._last_failure_category = None
                injector._last_failure_error = None

        return apply_turn_prefix(messages, prefix)

    def _check_context_overflow(self, max_context_chars: int = 200000) -> bool:
        """Check if context is at risk of overflow.

        Args:
            max_context_chars: Maximum allowed context size in chars

        Returns:
            True if context is dangerously large
        """
        # Use the sync context manager for context overflow check.
        # ContextService is async and may not accept max_context_chars,
        # so delegate to the conversation controller's sync check.
        return self._context_manager.check_context_overflow(max_context_chars)

    def get_context_metrics(self) -> ContextMetrics:
        """Get detailed context metrics.

        Returns:
            ContextMetrics with size and overflow information
        """
        return self._context_service.get_context_metrics()

    def _init_conversation_embedding_store(self) -> None:
        """Initialize embedding store via the memory runtime."""
        from victor.agent.runtime.memory_runtime import (
            initialize_conversation_embedding_store,
        )

        store, cache = initialize_conversation_embedding_store(
            memory_manager=self.memory_manager,
        )
        self._conversation_embedding_store = store
        if cache is not None:
            self._pending_semantic_cache = cache

    def _finalize_stream_metrics(
        self, usage_data: Optional[Dict[str, int]] = None
    ) -> Optional[StreamMetrics]:
        """Finalize stream metrics at end of streaming session.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Args:
            usage_data: Optional cumulative token usage from provider API.
                       When provided, enables accurate token counts.
        """
        return self._metrics_coordinator.finalize_stream_metrics(usage_data)

    def get_last_stream_metrics(self) -> Optional[StreamMetrics]:
        """Get metrics from the last streaming session.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Returns:
            StreamMetrics from the last session or None if no metrics available
        """
        return self._metrics_coordinator.get_last_stream_metrics()

    def get_streaming_metrics_summary(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive streaming metrics summary.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Returns:
            Dictionary with aggregated metrics or None if metrics disabled.
        """
        return self._metrics_coordinator.get_streaming_metrics_summary()

    def get_streaming_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent streaming metrics history.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Args:
            limit: Maximum number of recent metrics to return

        Returns:
            List of recent metrics dictionaries
        """
        return self._metrics_coordinator.get_streaming_metrics_history(limit)

    def get_session_cost_summary(self) -> Dict[str, Any]:
        """Get session cost summary.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Returns:
            Dictionary with session cost statistics
        """
        return self._metrics_coordinator.get_session_cost_summary()

    def get_session_cost_formatted(self) -> str:
        """Get formatted session cost string.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Returns:
            Cost string like "$0.0123" or "cost n/a"
        """
        return self._metrics_coordinator.get_session_cost_formatted()

    def get_last_task_report(self) -> Optional[Dict[str, Any]]:
        """Get the most recent canonical per-task metrics report."""
        return self._metrics_coordinator.get_last_task_report()

    def get_task_report_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent canonical per-task metrics reports."""
        return self._metrics_coordinator.get_task_report_history(limit)

    def export_session_costs(self, path: str, format: str = "json") -> None:
        """Export session costs to file.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Args:
            path: Output file path
            format: Export format ("json" or "csv")
        """
        self._metrics_coordinator.export_session_costs(path, format)

    def _start_task_report(
        self,
        user_message: str,
        *,
        stream: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Start a first-class task report on the canonical chat runtime path."""
        task_type = self._resolve_task_report_task_type()
        report_metadata = {
            "stream": stream,
            "provider": getattr(self.provider, "name", getattr(self, "provider_name", "unknown")),
            "model": getattr(self, "model", "unknown"),
        }
        if metadata:
            report_metadata.update(metadata)
        return self._metrics_coordinator.start_task_report(
            description=user_message,
            task_type=task_type,
            metadata=report_metadata,
        )

    def _finish_task_report(
        self,
        success: bool,
        *,
        user_message: str,
        stream: bool = False,
        response: Optional[Any] = None,
        error: Optional[BaseException] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Finish the active task report with compaction and tool-strategy metadata."""
        report_metadata = {
            "stream": stream,
            "provider": getattr(self.provider, "name", getattr(self, "provider_name", "unknown")),
            "model": getattr(self, "model", "unknown"),
            "task_preview": user_message[:200],
        }
        if response is not None and hasattr(response, "stop_reason"):
            report_metadata["stop_reason"] = getattr(response, "stop_reason", None)
        report_metadata.update(self._build_task_report_continuation_metadata())
        if metadata:
            report_metadata.update(metadata)

        last_tool_event = self._metrics_coordinator.get_last_tool_strategy_event() or {}
        return self._metrics_coordinator.finish_task_report(
            success,
            task_type=self._resolve_task_report_task_type(),
            metadata=report_metadata,
            error=str(error) if error else None,
            tool_schema_tokens=int(last_tool_event.get("tool_tokens", 0) or 0),
            compaction=self._build_task_report_compaction_metadata(),
        )

    def _resolve_task_report_task_type(self) -> str:
        """Resolve the most specific available task type for task-level metrics."""
        stream_ctx = getattr(self, "_current_stream_context", None)
        if stream_ctx is not None:
            unified_task_type = getattr(
                getattr(stream_ctx, "unified_task_type", None), "value", None
            )
            if unified_task_type:
                return str(unified_task_type)
            coarse_task_type = getattr(stream_ctx, "coarse_task_type", None)
            if coarse_task_type:
                return str(coarse_task_type)

        unified_tracker = getattr(self, "unified_tracker", None)
        tracker_task_type = getattr(unified_tracker, "task_type", None)
        if tracker_task_type:
            return str(tracker_task_type)

        for attr_name in ("_current_task_type", "_task_type"):
            value = getattr(self, attr_name, None)
            if value:
                return str(value)

        return "default"

    def _build_task_report_compaction_metadata(self) -> Dict[str, Any]:
        """Collect compaction continuity signals for the current task report."""
        stream_ctx = getattr(self, "_current_stream_context", None)
        perf_metrics: Dict[str, Any] = {}
        context_service = getattr(self, "_context_service", None)
        if context_service is not None and hasattr(context_service, "get_performance_metrics"):
            try:
                perf_metrics = context_service.get_performance_metrics() or {}
            except Exception as exc:
                logger.debug(
                    "Failed to read context performance metrics for task report: %s",
                    exc,
                )

        summary = ""
        occurred = False
        messages_removed = 0
        if stream_ctx is not None:
            summary = str(getattr(stream_ctx, "compaction_summary", "") or "")
            occurred = bool(
                getattr(stream_ctx, "compaction_occurred", False)
                or getattr(stream_ctx, "last_compaction_turn", -1) >= 0
            )
            messages_removed = int(getattr(stream_ctx, "compaction_message_removed_count", 0) or 0)

        saved_tokens = int(perf_metrics.get("last_compaction_saved_tokens", 0) or 0)
        return {
            "occurred": occurred or bool(summary) or messages_removed > 0 or saved_tokens > 0,
            "summary": summary,
            "messages_removed": messages_removed,
            "saved_tokens": saved_tokens,
            "strategy": str(getattr(stream_ctx, "last_compaction_strategy", "") or ""),
            "reason": str(getattr(stream_ctx, "last_compaction_reason", "") or ""),
            "policy_reason": str(getattr(stream_ctx, "last_compaction_policy_reason", "") or ""),
        }

    def _build_task_report_continuation_metadata(self) -> Dict[str, Any]:
        """Collect bounded continuation-ledger state for reporting/export paths."""
        stream_ctx = getattr(self, "_current_stream_context", None)
        if stream_ctx is None:
            return {}

        metadata: Dict[str, Any] = {}
        task_intent = str(getattr(stream_ctx, "task_intent", "") or "").strip()
        if task_intent:
            metadata["task_intent"] = task_intent

        plan_steps = [
            str(item).strip()
            for item in (getattr(stream_ctx, "plan_steps", []) or [])
            if str(item).strip()
        ][:6]
        if plan_steps:
            metadata["plan_steps"] = plan_steps

        intent_log = [
            dict(item)
            for item in (getattr(stream_ctx, "intent_log", []) or [])
            if isinstance(item, dict)
        ][-6:]
        if intent_log:
            metadata["intent_log"] = intent_log

        resume_summary = str(getattr(stream_ctx, "resume_summary", "") or "").strip()
        if resume_summary:
            metadata["resume_summary"] = resume_summary

        if bool(getattr(stream_ctx, "degraded_resume_state", False)):
            metadata["degraded_resume_state"] = True

        build_ledger = getattr(stream_ctx, "build_continuation_ledger", None)
        if callable(build_ledger):
            try:
                continuation_ledger = str(
                    build_ledger(max_events=4, max_plan_steps=4, max_chars=500) or ""
                ).strip()
            except TypeError:
                continuation_ledger = str(build_ledger() or "").strip()
            if continuation_ledger:
                metadata["continuation_ledger"] = continuation_ledger

        return metadata

    async def _preload_embeddings(self) -> None:
        """Preload tool embeddings in background to avoid blocking first query.

        This is called asynchronously during initialization if semantic tool
        selection is enabled. Errors are logged but don't crash the app.

        Note: The ToolSelector owns the _embeddings_initialized state to avoid
        DRY violations and consistency issues.
        """
        if not self.semantic_selector:
            return
        # ToolSelector owns the initialization state
        if self.tool_selector._embeddings_initialized:
            return

        try:
            logger.debug("Starting background embedding preload...")
            await self.semantic_selector.initialize_tool_embeddings(self.tools)
            # Mark initialization complete in ToolSelector (single source of truth)
            self.tool_selector._embeddings_initialized = True
            logger.debug(
                f"{self._presentation.icon('success')} Tool embeddings preloaded successfully in background"
            )
        except Exception as e:
            logger.warning(
                f"Failed to preload embeddings in background: {e}. "
                "Will load on first query (may add ~5s latency)."
            )

    def _create_background_task(
        self, coro: Any, name: str = "background_task"
    ) -> Optional[asyncio.Task]:
        """Create and track a background task for graceful shutdown."""
        from victor.agent.services.session_service import SessionService

        return SessionService.create_background_task(
            coro, name, self._background_tasks, self._bg_task_lock
        )

    async def _run_runtime_preload(self) -> None:
        """Run feature-flagged runtime preload tasks via PreloadManager."""
        from victor.framework.preload import (
            PreloadManager,
            PreloadPriority,
            preload_configuration,
        )

        preload_manager = PreloadManager(
            enable_parallel=bool(getattr(self.settings, "framework_preload_parallel", True))
        )
        preload_manager.add_task(
            "configuration",
            preload_configuration,
            priority=PreloadPriority.CRITICAL.value,
            required=False,
        )

        if self.use_semantic_selection:
            preload_manager.add_task(
                "tool_embeddings",
                self._preload_embeddings,
                priority=PreloadPriority.HIGH.value,
                required=False,
            )

        if getattr(self.settings, "http_connection_pool_enabled", False):

            async def _preload_http_pool() -> None:
                from victor.tools.http_pool import ConnectionPoolConfig, get_http_pool

                await get_http_pool(
                    ConnectionPoolConfig(
                        max_connections=int(
                            getattr(
                                self.settings,
                                "http_connection_pool_max_connections",
                                100,
                            )
                        ),
                        max_connections_per_host=int(
                            getattr(
                                self.settings,
                                "http_connection_pool_max_connections_per_host",
                                10,
                            )
                        ),
                        connection_timeout=int(
                            getattr(
                                self.settings,
                                "http_connection_pool_connection_timeout",
                                30,
                            )
                        ),
                        total_timeout=int(
                            getattr(self.settings, "http_connection_pool_total_timeout", 60)
                        ),
                    )
                )

            preload_manager.add_task(
                "http_pool",
                _preload_http_pool,
                priority=PreloadPriority.HIGH.value,
                required=False,
            )

        if not preload_manager.list_tasks():
            return

        if getattr(self.settings, "framework_preload_parallel", True):
            stats = await preload_manager.preload_parallel()
        else:
            stats = await preload_manager.preload_all()
        logger.debug(
            "Runtime preload completed: %s/%s tasks in %.2fs",
            stats.completed_tasks,
            stats.total_tasks,
            stats.duration,
        )

    def start_embedding_preload(self) -> None:
        """Start background embedding preload task.

        Should be called after orchestrator initialization to avoid blocking
        the main thread. Safe to call multiple times (no-op if already started).

        Embedding preload is deferred by default to avoid +694MB memory spike
        on first task. Set `preload_embeddings=True` in settings to eagerly load.
        The embedding model will still lazy-load on first semantic query via
        EmbeddingService._ensure_model_loaded().
        """
        if getattr(self.settings, "framework_preload_enabled", False):
            if self._runtime_preload_task is not None:
                return
            task = self._create_background_task(self._run_runtime_preload(), name="runtime_preload")
            if task:
                self._runtime_preload_task = task
                logger.debug("Started runtime preload task")
            return

        # Only preload embeddings if explicitly requested (default: deferred)
        if not getattr(self.settings, "preload_embeddings", False):
            return

        if not self.use_semantic_selection or self._embedding_preload_task:
            return

        task = self._create_background_task(self._preload_embeddings(), name="embedding_preload")
        if task:
            self._embedding_preload_task = task
            logger.debug("Started background task for embedding preload")

    def _record_tool_selection(self, method: str, num_tools: int) -> None:
        """Record tool selection statistics.

        Args:
            method: Selection method used ('semantic', 'keyword', 'fallback')
            num_tools: Number of tools selected
        """
        self._metrics_coordinator.record_tool_selection(method, num_tools)

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

    def _record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        elapsed_ms: float,
        error_type: Optional[str] = None,
    ) -> None:
        """Record tool execution statistics. Delegates to AgentMetricsService."""
        self._metrics_coordinator.record_tool_execution_full(
            tool_name=tool_name,
            success=success,
            elapsed_ms=elapsed_ms,
            error_type=error_type,
            usage_analytics=getattr(self, "_usage_analytics", None),
            sequence_tracker=getattr(self, "_sequence_tracker", None),
            tool_selector=getattr(self, "tool_selector", None),
            rl_coordinator=self._rl_coordinator,
            conversation_controller=getattr(self, "_conversation_controller", None),
            provider_name=getattr(self.current_provider, "name", "unknown"),
            model_name=getattr(self, "_current_model", "unknown"),
            task_type=getattr(self, "_task_type", "general"),
            vertical_name=getattr(self, "_vertical_name", None),
        )

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive tool usage statistics.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Returns:
            Dictionary with usage analytics including:
            - Selection stats (semantic/keyword/fallback counts)
            - Per-tool execution stats (calls, success rate, timing)
            - Cost tracking (by tier and total)
            - Overall metrics
        """
        return self._metrics_coordinator.get_tool_usage_stats(
            conversation_state_summary=self.conversation_state.get_state_summary()
        )

    def get_token_usage(self) -> "TokenUsage":
        """Get cumulative token usage for evaluation tracking.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Returns cumulative tokens used across all stream_chat calls.
        Used by VictorAgentAdapter for benchmark token tracking.

        Returns:
            TokenUsage dataclass with input/output/total token counts
        """
        return self._metrics_coordinator.get_token_usage()

    def reset_token_usage(self) -> None:
        """Reset cumulative token usage tracking.

        Delegates to AgentMetricsService (Phase 2 refactoring).

        Call this at the start of a new evaluation task to get fresh counts.
        """
        # Reset through coordinator (which updates the cumulative dict)
        for key in self._cumulative_token_usage:
            self._cumulative_token_usage[key] = 0

    def get_conversation_stage(self) -> ConversationStage:
        """Get the current conversation stage.

        Returns:
            Current ConversationStage enum value
        """
        return self.conversation_state.get_stage()

    def get_stage_recommended_tools(self) -> Set[str]:
        """Get tools recommended for the current conversation stage.

        Returns:
            Set of tool names recommended for current stage
        """
        return self.conversation_state.get_stage_tools()

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all integrated optimization components.

        Delegates to AgentMetricsService which owns the reporting logic.
        """
        return self._metrics_coordinator.get_optimization_status(
            context_compactor=self._context_compactor,
            usage_analytics=self._usage_analytics,
            sequence_tracker=self._sequence_tracker,
            code_correction_middleware=self._code_correction_middleware,
            safety_checker=self._safety_checker,
            auto_committer=self._auto_committer,
            search_router=self.search_router,
        )

    def flush_analytics(self) -> Dict[str, bool]:
        """Flush all analytics and cached data. Delegates to AgentMetricsService."""
        return self._metrics_coordinator.flush_analytics(
            usage_analytics=getattr(self, "_usage_analytics", None),
            sequence_tracker=getattr(self, "_sequence_tracker", None),
            tool_cache=getattr(self, "tool_cache", None),
        )

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

    def _get_skill_runtime(self) -> Any:
        """Get the canonical service-owned skill runtime helper."""
        runtime = getattr(self, "_skill_runtime", None)
        if runtime is None:
            from victor.agent.services.skill_runtime import SkillRuntime

            runtime = SkillRuntime(self.protocol_adapter)
            self._skill_runtime = runtime
        return runtime

    def _apply_skill_for_turn(self, user_message: str) -> None:
        """Apply skill auto-selection for a turn (used by both sync and streaming paths).

        Clears previous skill, matches new skill(s), injects into prompt,
        and records analytics.
        """
        self._get_skill_runtime().apply_skill_for_turn(user_message)

    def get_last_skill_match_info(self) -> Optional[Dict[str, Any]]:
        """Return metadata about the last skill match for response attachment."""
        return self._get_skill_runtime().get_last_skill_match_info()

    def _get_prompt_builder_runtime(self) -> Any:
        """Get the canonical service-owned prompt-builder runtime helper."""
        runtime = getattr(self, "_prompt_builder_runtime", None)
        if runtime is None:
            from victor.agent.services.prompt_builder_runtime import (
                PromptBuilderRuntime,
            )

            runtime = PromptBuilderRuntime(self.protocol_adapter)
            self._prompt_builder_runtime = runtime
        return runtime

    def clear_active_skills(self) -> None:
        """Remove any active skill injection.

        Called at the start of each turn to ensure skills don't accumulate.
        When cache_optimization enabled: just clears _active_skill_prompt
        (system prompt was never touched). Otherwise: restores base prompt.
        """
        self._get_skill_runtime().clear_active_skills()

    def get_skill_user_prefix(self) -> str:
        """Get active skill prompt as user message prefix (cache-friendly).

        When cache_optimization is enabled, skills are injected into the
        user message instead of mutating the system prompt. This keeps
        the system prompt byte-identical across turns for prefix caching.

        Returns:
            Skill prompt prefix string, or empty string if no active skill.
        """
        return self._get_skill_runtime().get_skill_user_prefix()

    def inject_skill(self, skill: Any) -> None:
        """Inject a skill's prompt fragment.

        When cache_optimization is enabled: stores skill text for user
        message injection (via get_skill_user_prefix). System prompt
        stays frozen.

        When disabled: legacy behavior — prepends to system prompt.

        Args:
            skill: A SkillDefinition with name, description, prompt_fragment.
        """
        self._get_skill_runtime().inject_skill(skill)

    def inject_skills(self, skills: List[Any]) -> None:
        """Inject multiple skills' prompt fragments.

        Cache-friendly: when cache_optimization enabled, stores for user
        message injection. Otherwise, legacy system prompt mutation.

        Args:
            skills: List of (SkillDefinition, score) tuples.
        """
        self._get_skill_runtime().inject_skills(skills)

    def update_system_prompt_for_query(self, query_classification=None) -> None:
        """Rebuild system prompt with query-specific classification context.

        Called by coordinators when a query classification is available,
        injecting task-aware guidance and tool constraints into the prompt.
        Updates both the orchestrator's cached prompt AND the conversation's
        live system message so the provider receives the updated prompt.

        When cache_optimization_enabled is True and the system prompt has
        already been built, this is a no-op to preserve prefix cache stability.
        Query-specific guidance should be injected via user messages instead.

        Args:
            query_classification: Optional QueryClassification for task-aware prompting
        """
        self._get_prompt_builder_runtime().update_system_prompt_for_query(
            query_classification=query_classification
        )

    def refresh_system_prompt(
        self,
        query_classification=None,
        *,
        preserve_existing_classification: bool = True,
    ) -> None:
        """Force a system prompt rebuild after an explicit runtime change."""
        self._get_prompt_builder_runtime().refresh_system_prompt(
            query_classification=query_classification,
            preserve_existing_classification=preserve_existing_classification,
        )

    def _sync_prompt_builder_runtime_state(self) -> None:
        """Align prompt-builder state with current mode and enabled tools."""
        self._get_prompt_builder_runtime().sync_prompt_builder_runtime_state()

    def build_system_prompt(self) -> str:
        """Build the system prompt via the canonical prompt runtime surface.

        Delegates to UnifiedPromptPipeline.build_system_prompt() when
        available. Falls back to the prompt-builder runtime helper during
        early initialization or compatibility-only pipeline failures.
        """
        self._sync_prompt_builder_runtime_state()

        pipeline = getattr(self, "_prompt_pipeline", None)
        if pipeline is not None:
            return pipeline.build_system_prompt()

        return self._get_prompt_builder_runtime().build_system_prompt_fallback()

    def _emit_prompt_used_event(self, prompt: str) -> None:
        """Emit PROMPT_USED event for RL prompt template learner.

        Delegates to UnifiedPromptPipeline.
        """
        pipeline = getattr(self, "_prompt_pipeline", None)
        if pipeline is not None:
            pipeline._emit_prompt_used_event(prompt)

    def _resolve_shell_variant(self, tool_name: str) -> str:
        """Resolve shell aliases to the appropriate enabled shell variant.

        Delegates to UnifiedPromptPipeline, then falls back to direct runtime
        dependency resolution.
        """
        pipeline = getattr(self, "_prompt_pipeline", None)
        if pipeline is not None:
            return pipeline.resolve_shell_variant(tool_name)

        try:
            from victor.agent.shell_resolver import resolve_shell_variant

            return resolve_shell_variant(
                tool_name,
                getattr(self, "tools", None),
                getattr(self, "mode_controller", None),
            )
        except Exception as exc:
            logger.debug("Shell resolver unavailable during runtime fallback: %s", exc)
            return tool_name

    def _get_thinking_disabled_prompt(self, base_prompt: str) -> str:
        """Prefix a prompt with the thinking disable prefix if supported.

        IMPORTANT: This should ONLY be used in RECOVERY scenarios where:
        - Model returned empty response (stuck in thinking)
        - Context overflow forced completion
        - Iteration limit forced completion

        Normal model calls should NOT use this - thinking mode produces
        better quality results. This is a last-resort recovery mechanism.

        For models with thinking mode (e.g., Qwen3), prepends the configured
        disable prefix (e.g., "/no_think") to get direct responses without
        internal reasoning overhead.

        Args:
            base_prompt: The base prompt text

        Returns:
            Prompt with thinking disable prefix if available, otherwise base_prompt
        """
        prefix = getattr(self.tool_calling_caps, "thinking_disable_prefix", None)
        if prefix:
            return f"{prefix}\n{base_prompt}"
        return base_prompt

    def _log_tool_call(self, name: str, kwargs: dict) -> None:
        """A hook that logs information before a tool is called."""
        # Move verbose argument logging to debug level - not user-facing
        logger.debug(f"Tool call: {name} with args: {kwargs}")

    def _classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task type based on keywords in the user message.

        Delegates to UnifiedPromptPipeline or TaskAnalyzer directly.

        Args:
            user_message: The user's input message

        Returns:
            Dictionary with classification results
        """
        pipeline = getattr(self, "_prompt_pipeline", None)
        if pipeline is not None:
            return pipeline.classify_task_keywords(user_message)

        task_analyzer = getattr(self, "_task_analyzer", None)
        if task_analyzer is not None:
            try:
                method = getattr(
                    task_analyzer,
                    "classify_task_keywords",
                    getattr(task_analyzer, "classify_keywords", None),
                )
                if method is not None:
                    return method(user_message)
            except Exception as exc:
                logger.debug("Task keyword classification fallback failed: %s", exc)

        return {"task_type": "default", "confidence": 0.0}

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

        Enforces max_conversation_history ceiling by removing oldest
        non-system messages when the limit is reached. Persistence
        and usage logging are delegated to ``ChatService.persist_message``.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            **kwargs: Additional fields (name, tool_call_id, tool_calls)
        """
        persist_synchronously = bool(kwargs.pop("persist_synchronously", False))
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

        # Intercept dynamic system nudges for cache-friendly injection
        if role == "system" and getattr(self, "_cache_optimization_enabled", False):
            # Check if history already contains a system message (the root prompt)
            has_root_system = any(m.role == "system" for m in self.conversation._messages)
            if has_root_system and hasattr(self, "reminder_manager") and self.reminder_manager:
                # If content looks like a nudge (e.g. "[FILES: ...]" or budget warning),
                # update reminder_manager state instead of appending.
                if content.startswith("[") and content.endswith("]"):
                    log_dropped_system_nudge(content)
                    return

        max_history = getattr(self.settings, "max_conversation_history", 100)
        # Trim oldest non-system message when history exceeds limit.
        # Note: Only protect the root system message (index 0).
        if len(self.conversation._messages) >= max_history:
            for i, msg in enumerate(self.conversation._messages):
                # Don't pop the root system prompt (index 0)
                if i == 0 and getattr(msg, "role", None) == "system":
                    continue
                self.conversation._messages.pop(i)
                break

        if role == "tool":
            logger.debug(
                "add_message(role=tool): name=%s tool_call_id=%s content_len=%d",
                kwargs.get("name"),
                kwargs.get("tool_call_id"),
                len(content),
            )
        self.conversation.add_message(role, content, **kwargs)

        from victor.agent.services.chat_service import ChatService

        ChatService.persist_message(
            role=role,
            content=content,
            memory_manager=self.memory_manager,
            memory_session_id=self._memory_session_id,
            usage_logger=self.usage_logger,
            tool_name=kwargs.get("name"),
            tool_call_id=kwargs.get("tool_call_id"),
            tool_calls=kwargs.get("tool_calls"),
            metadata=kwargs.get("metadata"),
            persist_synchronously=persist_synchronously,
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

        # Re-stamp the session_id contextvar from the persistent attr. The
        # contextvar can be lost when the caller (e.g. the benchmark adapter)
        # set it in a different async context, or when decisions cross a
        # thread/loop boundary (decide_sync). Setting it here, in the chat
        # call's context, ensures every decision the agentic loop logs carries
        # the task's session_id (FEP-0012 decision_outcome spine integrity).
        if self.active_session_id:
            try:
                set_session_id(self.active_session_id)
            except Exception:
                pass

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

    def _prepare_chat_service_turn_runtime(
        self,
        user_message: str,
        *,
        stream: bool,
        constraints: Any = None,
        vertical: Optional[str] = None,
    ) -> None:
        """Prepare the canonical chat-service turn scope for a live task."""
        if stream:
            self._apply_skill_for_turn(user_message)

        if constraints:
            self._constraint_activator.activate_constraints(
                constraints=constraints,
                vertical=vertical or getattr(self, "vertical", "coding"),
            )

    def _teardown_chat_service_turn_runtime(
        self,
        _user_message: str,
        *,
        stream: bool,
        constraints: Any = None,
        vertical: Optional[str] = None,
    ) -> None:
        """Tear down any temporary chat-service turn scope state."""
        del stream, vertical
        if constraints:
            self._constraint_activator.deactivate_constraints()

        # Close the credit-assignment feedback loop (universal per-turn teardown).
        if getattr(self, "_credit_tracking_service", None) is not None:
            self._credit_tracking_service.assign_turn_credit_at_boundary()

    async def _handle_context_and_iteration_limits(
        self,
        user_message: str,
        max_total_iterations: int,
        max_context: int,
        total_iterations: int,
        last_quality_score: float,
    ) -> tuple[bool, Optional[StreamChunk]]:
        """Handle context overflow and hard iteration limits.

        Delegates to ChatService.
        """
        return await self._chat_service.handle_context_and_iteration_limits(
            user_message,
            max_total_iterations,
            max_context,
            total_iterations,
            last_quality_score,
        )

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

    # Keywords that strongly suggest tool usage is required
    _TOOL_SIGNAL_KEYWORDS = frozenset(
        {
            "file",
            "code",
            "search",
            "edit",
            "write",
            "read",
            "create",
            "delete",
            "run",
            "test",
            "debug",
            "fix",
            "build",
            "deploy",
            "docker",
            "git",
            "commit",
            "branch",
            "install",
            "refactor",
            "implement",
            "function",
            "class",
            "import",
            "variable",
            "error",
            "bug",
            "compile",
            "execute",
            "database",
            "query",
            "api",
            "endpoint",
            "config",
            "log",
            "trace",
            "mkdir",
            "move",
            "rename",
            "grep",
            "find",
            "ls",
            "bash",
            "shell",
            "pip",
            "npm",
            "make",
            "curl",
            "fetch",
            "download",
            "upload",
        }
    )

    # Patterns that indicate pure Q&A (no tools needed)
    _QA_SIGNAL_PATTERNS = (
        "what is",
        "what are",
        "what does",
        "what do",
        "how does",
        "how do",
        "how is",
        "how are",
        "why does",
        "why do",
        "why is",
        "why are",
        "can you explain",
        "explain",
        "describe",
        "tell me about",
        "what's the difference",
        "thanks",
        "thank you",
        "hello",
        "hi ",
        "good morning",
        "good evening",
    )

    # Bare continuation/affirmation user turns (< 15 chars, no tool keyword) would trip
    # the length gate below and drop the working tool set; keep the read-only core so
    # the model can still reason over in-progress work (_tool_skip_mode: read_core).
    _CONTINUATION_TOKENS = frozenset(
        {
            "continue",
            "proceed",
            "go",
            "go on",
            "keep going",
            "next",
            "more",
            "again",
            "yes",
            "y",
            "ok",
            "okay",
            "sure",
            "do it",
            "apply",
            "apply it",
        }
    )

    def _tool_skip_mode(self, context_msg: str) -> str:
        """Decide tool supply for a (possibly conversational) turn (tool-supply P3).

        Uses a fast heuristic first (keyword scan), then optionally consults the edge
        model via DecisionService for borderline Q&A. Returns one of:

        - ``"skip"``      — trivially-safe conversational turn (short greeting): no tools.
        - ``"read_core"`` — borderline Q&A: provide ONLY a minimal read-only core so the
          model can look something up if it turns out it needs to, rather than removing
          the entire tool set (the old behavior left "how does X work?" unable to read X).
        - ``"tools"``     — proceed with normal tool selection.
        """
        msg_lower = context_msg.lower().strip()

        # Bare continuation/affirmation of an in-progress task ("continue",
        # "proceed", "go", "yes", "apply it", ...). These are short and carry no
        # tool-signal keyword, so the length gate below would drop the tool set.
        # Preserve the read-only core instead so the model can keep working the
        # active task. Checked BEFORE the length short-circuit on purpose.
        if msg_lower in self._CONTINUATION_TOKENS:
            return "read_core"

        # Very short messages are almost always greetings/Q&A — the only hard no-tools path.
        if len(msg_lower) < 15:
            # Unless they look like commands: "fix it", "run tests", etc.
            if any(kw in msg_lower for kw in ("fix", "run", "edit", "create", "delete")):
                return "tools"
            return "skip"

        # Heuristic: count tool-signal keywords vs Q&A patterns
        words = set(msg_lower.split())
        tool_signals = len(words & self._TOOL_SIGNAL_KEYWORDS)
        qa_match = any(msg_lower.startswith(pat) for pat in self._QA_SIGNAL_PATTERNS)

        # High-confidence heuristic paths
        if tool_signals >= 2:
            return "tools"  # Clearly needs tools
        if qa_match and tool_signals == 0:
            # Consult edge model for borderline Q&A (might still need tools). Even when it
            # says "skip", give the read-only core rather than nothing.
            skip = self._check_tool_necessity_via_edge(context_msg, heuristic_conf=0.85)
            return "read_core" if skip else "tools"

        # Ambiguous: 0-1 tool signals — default to providing tools
        if tool_signals == 0 and qa_match:
            skip = self._check_tool_necessity_via_edge(context_msg, heuristic_conf=0.6)
            return "read_core" if skip else "tools"

        return "tools"  # Default: provide tools

    def _should_skip_tools_for_turn(self, context_msg: str) -> bool:
        """Back-compat shim: True when full tool selection is not needed this turn.

        Prefer :meth:`_tool_skip_mode`, which distinguishes a hard ``"skip"`` (greeting)
        from ``"read_core"`` (borderline Q&A still gets read tools).
        """
        return self._tool_skip_mode(context_msg) != "tools"

    def _check_tool_necessity_via_edge(self, context_msg: str, heuristic_conf: float) -> bool:
        """Consult the decision service for tool necessity.

        Falls back to the heuristic when no decision service is registered or it
        times out. The service is registered per the ``decision_backend`` enum
        (FEP-0012); the legacy ``USE_LLM_DECISION_SERVICE`` gate was removed so a
        registered local-classifier is honored here regardless of that flag.
        """
        try:
            from victor.agent.services.protocols.decision_service import (
                get_decision_service,
            )
            from victor.agent.decisions.schemas import DecisionType

            service = get_decision_service(getattr(self, "_container", None))
            if service is None:
                return heuristic_conf >= 0.7

            decision = service.decide_sync(
                DecisionType.TOOL_NECESSITY,
                context={"message_excerpt": context_msg[:300]},
                heuristic_result={"requires_tools": heuristic_conf < 0.7},
                heuristic_confidence=heuristic_conf,
            )

            if decision.source == "heuristic" or decision.result is None:
                return heuristic_conf >= 0.7

            # Edge model returned a ToolNecessityDecision
            requires = decision.result.get("requires_tools", True)
            conf = decision.result.get("confidence", 0.5)
            if not requires and conf >= 0.6:
                logger.debug(
                    "TOOL_NECESSITY: skipping tools for Q&A turn " "(confidence=%.2f, source=%s)",
                    conf,
                    decision.source,
                )
                return True
            return False

        except Exception:
            logger.debug("Tool necessity check failed, defaulting to provide tools")
            return False

    def _sort_tools_for_kv_stability(self, tools):
        """Sort tools for KV-cache stability; session-local cache is managed here."""
        if tools is None:
            return None
        if self._kv_optimization_enabled:
            stabilize_order = True
        else:
            stabilize_order = AgentOrchestrator._should_stabilize_tool_order(self)
        if not stabilize_order:
            return tools
        current_names = frozenset(t.name for t in tools)
        last_names = getattr(self, "_last_sorted_tool_names", None)
        last_tools = getattr(self, "_last_sorted_tools", None)
        if last_names == current_names and last_tools is not None:
            return last_tools
        sorted_tools = self._tool_service.sort_tools_for_kv_stability(
            tools, kv_optimization_enabled=stabilize_order
        )
        self._last_sorted_tool_names = current_names
        self._last_sorted_tools = sorted_tools
        return sorted_tools

    def _should_stabilize_tool_order(self) -> bool:
        """Return whether this turn should use byte-stable tool ordering."""
        if self._kv_optimization_enabled:
            return True

        strategy = self._resolve_kv_strategy_setting()
        if strategy in ("session_stable", "additive"):
            return True
        if strategy != "context_aware":
            return False
        return self._context_aware_profile_session_locked()

    def _resolve_kv_strategy_setting(self) -> str:
        """Read kv_tool_strategy from settings; default to 'context_aware'."""
        try:
            ctx = getattr(self, "settings", None)
            if ctx is not None:
                context = getattr(ctx, "context", None)
                if context is not None:
                    return getattr(context, "kv_tool_strategy", "context_aware")
        except Exception:
            pass
        return "context_aware"

    def _apply_kv_tool_strategy(self, tools):
        """Delegate KV tool strategy to ToolService; manage session-stable cache here."""
        if self._is_tool_strategy_v2_enabled():
            result = self._tool_service.apply_context_aware_strategy(
                tools,
                provider=self.provider,
                model=self.model,
                session_semantic_tools=getattr(self, "_session_semantic_tools", None),
            )
            if self._context_aware_profile_session_locked():
                self._session_semantic_tools = result
            return result
        strategy = self._resolve_kv_strategy_setting()
        result = self._tool_service.apply_kv_tool_strategy(
            tools,
            kv_optimization_enabled=self._kv_optimization_enabled,
            provider=self.provider,
            model=self.model,
            session_semantic_tools=getattr(self, "_session_semantic_tools", None),
            kv_tool_strategy=strategy,
        )
        # Persist the (grow-only) session-stable selection at orchestrator level
        # (ToolService is a singleton). P4: persist EVERY turn — not just the first —
        # so newly-added tools join the cached set and stay available next turn.
        if strategy in ("session_stable", "additive") or (
            strategy == "context_aware" and self._context_aware_profile_session_locked()
        ):
            self._session_semantic_tools = result
        return result

    def _context_aware_profile_session_locked(self) -> bool:
        """Whether the provider-economics profile wants session-stable tools."""
        try:
            from victor.config.tool_tiers import resolve_tool_supply_profile

            context_window = self._get_context_window(self.provider, self.model)
            fallback_max_tools = None
            try:
                fallback_max_tools = getattr(getattr(self.settings, "tools", None), "budget", None)
            except Exception:
                fallback_max_tools = None
            profile = resolve_tool_supply_profile(
                self.provider,
                context_window,
                fallback_max_tools=int(fallback_max_tools or 8),
            )
            return profile.session_lock != "off"
        except Exception:
            return False

    def _is_tool_strategy_v2_enabled(self) -> bool:
        """Check if the context-aware tool strategy (v2) is enabled.

        Gated solely by the ``tool_strategy_v2`` feature flag. The former
        ``tool_strategy_v2_enabled`` settings field was an unreachable,
        exception-only fallback (defaulted False) and has been removed.
        """
        try:
            from victor.core.feature_flags import is_enabled

            return is_enabled("tool_strategy_v2")
        except Exception:
            return False

    def _apply_context_aware_strategy(self, tools):
        """Delegate context-aware selection to ToolService; emit strategy event here."""
        result = self._tool_service.apply_context_aware_strategy(
            tools, provider=self.provider, model=self.model
        )
        try:
            from victor.config.tool_tiers import get_provider_category

            context_window = self._get_context_window(self.provider, self.model)
            provider_category = get_provider_category(context_window)
            self._emit_tool_strategy_event(
                strategy="context_aware",
                tool_count=len(result),
                tool_tokens=sum(self._estimate_tool_tokens(t, provider_category) for t in result),
                context_window=context_window,
                provider=self.provider,
                reason="kv_context_aware",
                tools=result,
            )
        except Exception:
            pass
        return result

    def _get_context_window(self, provider, model: str) -> int:
        """Get context window size for provider/model.

        Args:
            provider: Provider instance
            model: Model identifier

        Returns:
            Context window in tokens
        """
        if hasattr(provider, "context_window"):
            return provider.context_window(model)

        # Fallback: safe default
        logger.warning(
            f"Provider {provider.name} does not support context_window(), using default 8192"
        )
        return 8192

    def _estimate_tool_tokens(self, tool, provider_category: str = None) -> int:
        """Delegate token estimation to ToolService."""
        return self._tool_service.estimate_tool_tokens(tool, provider_category=provider_category)

    def _should_session_lock_all_tools(
        self, provider, context_window: int, tool_tokens: int
    ) -> bool:
        """Determine if all tools should be session-locked.

        Session-locking is economy-optimal when:
        1. Provider supports API-level caching (90% discount)
        2. Context window is large (≥32K) for KV efficiency
        3. Tool token count is reasonable

        Args:
            provider: Provider instance
            context_window: Context window in tokens
            tool_tokens: Current tool token count

        Returns:
            True if session-locking is recommended
        """
        # Cloud providers with caching: Always session-lock
        if hasattr(provider, "supports_prompt_caching") and provider.supports_prompt_caching():
            return True

        # Large local models (≥32K): Session-lock for KV efficiency
        if context_window >= 32000:
            return True

        # Otherwise: Don't session-lock (use semantic selection)
        return False

    def _is_gemini_provider(self, provider) -> bool:
        """Check if provider is Gemini.

        Args:
            provider: Provider instance

        Returns:
            True if this is a Gemini provider
        """
        provider_name = getattr(provider, "name", "").lower()
        return "gemini" in provider_name or "google" in provider_name

    def _demote_tools_to_fit(
        self, tools, max_tokens: int, context_window: int, provider_category: str = None
    ) -> list:
        """Demote or drop low-priority tools until within budget.

        Args:
            tools: List of tools to filter
            max_tokens: Maximum tool tokens allowed (25% of context window)
            context_window: Context window size
            provider_category: Provider category for tier selection

        Returns:
            Filtered list of tools that fit within budget
        """
        from victor.tools.enums import Priority, SchemaLevel

        # Sort by priority (CRITICAL first)
        sorted_tools = sorted(
            tools,
            key=lambda t: (t.priority.value if hasattr(t, "priority") else 99, t.name),
        )

        result = []
        current_tokens = 0

        for tool in sorted_tools:
            # Estimate current token cost using provider-specific tiers
            tool_cost = self._estimate_tool_tokens(tool, provider_category)

            if current_tokens + tool_cost <= max_tokens:
                # Tool fits within budget
                result.append(tool)
                current_tokens += tool_cost
            elif hasattr(tool, "priority") and tool.priority == Priority.CRITICAL:
                # Critical tools MUST fit - demote to STUB
                try:
                    # Temporarily override schema level to STUB
                    original_schema = getattr(tool, "_schema_level", None)
                    tool._schema_level = SchemaLevel.STUB
                    stub_cost = self._estimate_tool_tokens(tool)
                    tool._schema_level = original_schema  # Restore

                    if current_tokens + stub_cost <= max_tokens:
                        result.append(tool)
                        current_tokens += stub_cost
                        logger.debug(f"Demoted critical tool {tool.name} to STUB to fit budget")
                    else:
                        logger.warning(
                            f"Critical tool {tool.name} ({stub_cost} tokens) exceeds budget "
                            f"even as STUB. Dropping tool."
                        )
                except Exception as e:
                    logger.warning(f"Error demoting tool {tool.name}: {e}")
            else:
                # Skip non-critical tool
                logger.debug(
                    f"Skipping {tool.name} (priority: {tool.priority if hasattr(tool, 'priority') else 'unknown'}) "
                    f"to fit within {max_tokens} token budget"
                )

        logger.info(
            f"Demoted tools to fit context window: {len(tools)} → {len(result)} tools, "
            f"{current_tokens} tokens (budget: {max_tokens}, context: {context_window})"
        )

        return result

    def _semantic_select_tools(self, tools, max_tokens: int, provider_category: str = None) -> list:
        """Delegate semantic tool selection to ToolService."""
        return self._tool_service.semantic_select_tools(
            tools, max_tokens, provider_category=provider_category
        )

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
        """Emit tool strategy selection event via AgentMetricsService.

        Args:
            strategy: Strategy name (session_lock, semantic_selection, etc.)
            tool_count: Number of tools selected
            tool_tokens: Total tool tokens
            context_window: Context window size
            provider: Provider instance or provider name
            reason: Reason for strategy selection
            tools: Optional list of tools for detailed metrics
        """
        try:
            self._metrics_coordinator.emit_tool_strategy_event(
                strategy=strategy,
                tool_count=tool_count,
                tool_tokens=tool_tokens,
                context_window=context_window,
                provider=provider,
                model=self.model,
                reason=reason,
                tools=tools,
                v2_enabled=self._is_tool_strategy_v2_enabled(),
            )
        except Exception as e:
            logger.debug(f"Failed to emit tool strategy event: {e}")

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

            runtime = ToolExecutionRuntime(self.protocol_adapter)
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

        # Re-stamp the session_id contextvar (see chat() for rationale).
        if self.active_session_id:
            try:
                set_session_id(self.active_session_id)
            except Exception:
                pass

        async for chunk in self._chat_service.stream_chat(user_message, **kwargs):
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

            runtime = SessionRuntime(self.protocol_adapter)
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

    def get_stage(self) -> "ConversationStage":
        """Get current conversation stage (protocol method).

        Returns:
            Current ConversationStage enum value

        Note:
            Framework layer converts this to framework.state.Stage
        """
        if self.conversation_state:
            return self.conversation_state.get_stage()
        return ConversationStage.INITIAL

    def get_tool_calls_count(self) -> int:
        """Get total tool calls made (protocol method), matching get_tool_budget().

        Returns:
            Non-negative count of tool calls in this session
        """
        return int(getattr(self, "tool_calls_used", 0) or 0)

    def get_tool_budget(self) -> int:
        """Get the *enforcing* tool budget (protocol method).

        ``self.tool_budget`` seeds ``StreamingChatContext`` and halts the loop;
        ``UnifiedTaskTracker`` is advisory by design and must not be quoted here.

        Returns:
            Maximum allowed tool calls
        """
        return int(getattr(self, "tool_budget", 50) or 50)

    def get_observed_files(self) -> Set[str]:
        """Get files observed/read during conversation (protocol method).

        Returns:
            Set of absolute file paths
        """
        return set(getattr(self, "observed_files", []))

    def get_modified_files(self) -> Set[str]:
        """Get files modified during conversation (protocol method).

        Returns:
            Set of absolute file paths
        """
        if self.conversation_state and hasattr(self.conversation_state, "state"):
            return set(getattr(self.conversation_state.state, "modified_files", []))
        return set()

    def get_iteration_count(self) -> int:
        """Get current agent loop iteration count (protocol method).

        Returns:
            Non-negative iteration count
        """
        if self.unified_tracker:
            return self.unified_tracker.iteration_count
        return 0

    def get_max_iterations(self) -> int:
        """Get maximum allowed iterations (protocol method).

        Returns:
            Max iteration limit
        """
        if self.unified_tracker:
            return self.unified_tracker.max_iterations
        return 25

    # --- ProviderProtocol ---

    @property
    def current_provider(self) -> str:
        """Get current provider name (protocol property).

        Returns:
            Provider identifier (e.g., "anthropic", "openai")
        """
        return self.provider_name

    @property
    def current_model(self) -> str:
        """Get current model name (protocol property).

        Returns:
            Model identifier
        """
        return self.model

    async def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None,
        on_switch: Optional[Any] = None,
    ) -> bool:
        """[CANONICAL] Switch provider via ProviderService.

        Delegates to ProviderService for orchestration, maintaining the
        legacy coordinator in sync for backward compatibility.

        Args:
            provider_name: Target provider name
            model: Optional specific model
            on_switch: Optional callback after switch
        """
        return await self._get_provider_management_runtime().switch_provider(
            provider_name,
            model,
            on_switch,
        )

    async def switch_model(self, model: str) -> bool:
        """[CANONICAL] Switch model via ProviderService."""
        return await self._get_provider_management_runtime().switch_model(model)

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

    def get_system_prompt(self) -> str:
        """Get current system prompt (protocol method).

        Returns:
            Complete system prompt string
        """
        return self._get_prompt_builder_runtime().get_system_prompt()

    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt (protocol method).

        Args:
            prompt: New system prompt (replaces existing)
        """
        self._get_prompt_builder_runtime().set_system_prompt(prompt)

    def append_to_system_prompt(self, content: str) -> None:
        """Append content to system prompt (protocol method).

        Args:
            content: Content to append
        """
        self._get_prompt_builder_runtime().append_to_system_prompt(content)

    # --- MessagesProtocol ---

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get conversation messages (protocol method).

        Returns:
            List of message dictionaries
        """
        return [{"role": m.role, "content": m.content} for m in self.conversation.messages]

    def get_message_count(self) -> int:
        """Get message count (protocol method).

        Returns:
            Number of messages in conversation
        """
        return len(self.conversation.messages)

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
