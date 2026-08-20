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

"""Service provider for orchestrator dependencies.

Registers all services required by AgentOrchestrator with the DI container.
This enables:
- Centralized service configuration
- Consistent lifecycle management
- Easy testing via override_services
- Type-safe service resolution

Design Pattern: Service Provider
- Groups related service registrations
- Separates singleton vs scoped lifetimes
- Provides factory functions for complex service creation

Usage:
    from victor.core.container import ServiceContainer
    from victor.agent.service_provider import (
        OrchestratorServiceProvider,
        configure_orchestrator_services,
    )

    # Option 1: Full registration
    container = ServiceContainer()
    configure_orchestrator_services(container, settings)

    # Option 2: Selective registration
    provider = OrchestratorServiceProvider(settings)
    provider.register_singleton_services(container)  # Only singletons
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Optional

from victor.config.tool_selection_access import is_semantic_tool_selection_enabled
from victor.core.container import ServiceContainer, ServiceLifetime
from victor.core.service_registration import (
    ServiceRegistrationSpec,
    register_services_from_specs,
)
from victor.core.service_specs import AGENT_SINGLETON_SPECS

if TYPE_CHECKING:
    from victor.config.settings import Settings
    from victor.agent.protocols import (
        ActionAuthorizerProtocol,
        ArgumentNormalizerProtocol,
        AutoCommitterProtocol,
        CodeExecutionManagerProtocol,
        ComplexityClassifierProtocol,
        ContextCompactorProtocol,
        ContextTemperatureClassifierProtocol,
        ConversationEmbeddingStoreProtocol,
        ConversationStateMachineProtocol,
        DebugLoggerProtocol,
        MCPBridgeProtocol,
        MessageHistoryProtocol,
        MetricsCollectorProtocol,
        ModeControllerProtocol,
        ObservabilityProtocol,
        ParallelExecutorProtocol,
        ProjectContextProtocol,
        ProviderRegistryProtocol,
        RecoveryHandlerProtocol,
        ResponseCompleterProtocol,
        SafetyCheckerProtocol,
        SearchRouterProtocol,
        SemanticToolSelectorProtocol,
        SystemPromptBuilderProtocol,
        TaskAnalyzerProtocol,
        TaskTrackerProtocol,
        TaskTypeHinterProtocol,
        ToolCacheProtocol,
        ToolCallTrackerProtocol,
        ToolDeduplicationTrackerProtocol,
        ToolDependencyGraphProtocol,
        ToolExecutorProtocol,
        ToolOutputFormatterProtocol,
        ToolPluginRegistryProtocol,
        ToolRegistrarProtocol,
        ToolSelectorProtocol,
        ToolSequenceTrackerProtocol,
        UsageAnalyticsProtocol,
        UsageLoggerProtocol,
        WorkflowRegistryProtocol,
    )
    from victor.agent.services.protocols import (
        ChunkRuntimeProtocol,
        CoordinationAdvisorRuntimeProtocol,
        IntentClassifierProtocol,
        PromptRuntimeProtocol,
        RLLearningRuntimeProtocol,
        ReminderManagerProtocol,
        ResponseSanitizerProtocol,
        StateRuntimeProtocol,
        StreamingConfidenceMonitorProtocol,
        StreamingHandlerProtocol,
        StreamingMetricsCollectorProtocol,
        StreamingRecoveryRuntimeProtocol,
        TaskRuntimeProtocol,
        ToolPlanningRuntimeProtocol,
    )

logger = logging.getLogger(__name__)


class OrchestratorServiceProvider:
    """Service provider for orchestrator dependencies.

    Manages registration of all services required by AgentOrchestrator.
    Services are categorized by lifetime:

    Singleton Services (application lifetime):
        - ToolRegistry: Shared tool definitions
        - ObservabilityIntegration: Event bus
        - TaskAnalyzer: Shared analysis
        - IntentClassifier: Semantic classification
        - ComplexityClassifier: Task complexity
        - ActionAuthorizer: Action authorization
        - SearchRouter: Search routing
        - ResponseSanitizer: Response cleanup
        - ArgumentNormalizer: Argument normalization
        - ProjectContext: Project-specific instructions

    Scoped Services (per-session):
        - ConversationStateMachine: Per-session state
        - UnifiedTaskTracker: Per-session tracking
        - MessageHistory: Per-session messages

    Attributes:
        _settings: Application settings for service configuration

    Example:
        container = ServiceContainer()
        provider = OrchestratorServiceProvider(settings)
        provider.register_services(container)

        # Resolve singletons directly
        sanitizer = container.get(ResponseSanitizerProtocol)

        # Resolve scoped services within a scope
        with container.create_scope() as scope:
            state_machine = scope.get(ConversationStateMachineProtocol)
    """

    def __init__(self, settings: "Settings"):
        """Initialize the service provider.

        Args:
            settings: Application settings for service configuration
        """
        self._settings = settings

    def register_services(self, container: ServiceContainer) -> None:
        """Register all orchestrator services.

        Registers both singleton and scoped services. Call this method
        during application bootstrap to set up all orchestrator dependencies.

        Args:
            container: DI container to register services in
        """
        # Store container reference for factory methods
        self.container = container
        self.register_singleton_services(container)
        self.register_scoped_services(container)
        logger.info("Registered all orchestrator services")

    def register_singleton_services(self, container: ServiceContainer) -> None:
        """Register singleton (application-lifetime) services.

        These services are created once and shared across all sessions.
        Use for stateless services or those with expensive initialization.

        Args:
            container: DI container to register services in
        """
        spec_entries = []
        for spec in AGENT_SINGLETON_SPECS:
            factory = getattr(self, spec.factory_attr)
            if spec.pass_container:

                def make_factory(c, method=factory):
                    return method(c)

                factory_fn = make_factory
            else:

                def make_factory(c, method=factory):
                    return method()

                factory_fn = make_factory
            spec_entries.append(
                ServiceRegistrationSpec(
                    spec.protocol,
                    factory_fn,
                    lifetime=spec.lifetime,
                )
            )

        register_services_from_specs(container, spec_entries)

        # ToolRegistry and ToolRegistrar - core tool management
        self._register_tool_registry(container)
        # RecoveryHandler - model failure recovery with Q-learning
        self._register_recovery_handler(container)
        # ToolCacheManager - singleton for internal tool state (indexes, connections)
        self._register_tool_cache_manager(container)
        # PathResolver - singleton for centralized path resolution
        self._register_path_resolver(container)
        # UnifiedMemoryCoordinator - singleton for federated memory search
        self._register_unified_memory_coordinator(container)
        # PresentationAdapter - singleton for icon/formatting concerns
        self._register_presentation_adapter(container)
        # TaskAnalyzer - singleton for task analysis and classification
        self._register_task_analyzer(container)

        logger.debug("Registered singleton orchestrator services")

    def register_scoped_services(self, container: ServiceContainer) -> None:
        """Register scoped (per-session) services.

        These services are created fresh for each orchestrator session.
        Use for stateful services that need isolation between sessions.

        Args:
            container: DI container to register services in
        """
        from victor.agent.protocols import (
            ConversationStateMachineProtocol,
            TaskTrackerProtocol,
            MessageHistoryProtocol,
        )

        # ConversationStateMachine - per-session state
        container.register(
            ConversationStateMachineProtocol,
            lambda c: self._create_conversation_state_machine(c),
            ServiceLifetime.SCOPED,
        )

        # UnifiedTaskTracker - per-session tracking
        container.register(
            TaskTrackerProtocol,
            lambda c: self._create_unified_task_tracker(),
            ServiceLifetime.SCOPED,
        )

        # MessageHistory - per-session messages
        container.register(
            MessageHistoryProtocol,
            lambda c: self._create_message_history(),
            ServiceLifetime.SCOPED,
        )

        # ToolAccessController - per-session tool access control
        self._register_tool_access_controller(container)

        # BudgetManager - per-session budget management
        self._register_budget_manager(container)

        logger.debug("Registered scoped orchestrator services")

    # =========================================================================
    # Factory methods for singleton services
    # =========================================================================

    def _register_tool_registry(self, container: ServiceContainer) -> None:
        """Register ToolRegistry as singleton."""
        from victor.agent.protocols import ToolRegistryProtocol, ToolRegistrarProtocol
        from victor.tools.registry import ToolRegistry

        container.register(
            ToolRegistryProtocol,
            lambda c: ToolRegistry(),
            ServiceLifetime.SINGLETON,
        )

        # ToolRegistrar - manages tool registration, plugins, and MCP integration
        container.register(
            ToolRegistrarProtocol,
            lambda c: self._create_tool_registrar(c),
            ServiceLifetime.SINGLETON,
        )

    def _create_tool_registrar(self, container: ServiceContainer) -> "ToolRegistrarProtocol":
        """Create ToolRegistrar instance.

        ToolRegistrar manages:
        - Dynamic tool discovery from victor/tools directory
        - Plugin system management
        - MCP (Model Context Protocol) integration
        - Tool dependency graph setup

        Args:
            container: DI container for resolving dependencies

        Returns:
            ToolRegistrar instance
        """
        from victor.agent.tool_registrar import ToolRegistrar, ToolRegistrarConfig
        from victor.agent.protocols import (
            ToolRegistryProtocol,
            ToolDependencyGraphProtocol,
        )

        # Get ToolRegistry from container
        tool_registry = container.get(ToolRegistryProtocol)

        # Get ToolDependencyGraph from container for tool planning
        tool_graph = None
        if getattr(self._settings, "enable_tool_graph", True):
            try:
                tool_graph = container.get(ToolDependencyGraphProtocol)
            except Exception:
                pass  # Tool graph is optional

        # Build config from settings
        config = ToolRegistrarConfig(
            enable_plugins=getattr(self._settings, "enable_plugins", True),
            enable_mcp=getattr(self._settings, "use_mcp_tools", False),
            enable_tool_graph=getattr(self._settings, "enable_tool_graph", True),
            airgapped_mode=getattr(self._settings, "airgapped_mode", False),
        )

        return ToolRegistrar(
            tools=tool_registry,
            settings=self._settings,
            provider=None,  # Will be set later by orchestrator
            model=getattr(self._settings, "model", None),
            tool_graph=tool_graph,  # Pass tool graph for tool planning
            config=config,
        )

    def _register_observability(self, container: ServiceContainer) -> None:
        """Register ObservabilityIntegration as singleton."""
        from victor.agent.protocols import ObservabilityProtocol

        def create_observability(_: ServiceContainer) -> "ObservabilityProtocol":
            enable = getattr(self._settings, "enable_observability", True)
            if not enable:
                return _NullObservability()

            try:
                from victor.observability.integration import ObservabilityIntegration

                return ObservabilityIntegration()
            except ImportError:
                logger.warning("ObservabilityIntegration not available")
                return _NullObservability()

        container.register(
            ObservabilityProtocol,
            create_observability,
            ServiceLifetime.SINGLETON,
        )

    def _register_task_analyzer(self, container: ServiceContainer) -> None:
        """Register TaskAnalyzer as singleton."""
        from victor.agent.protocols import TaskAnalyzerProtocol

        def create_task_analyzer(_: ServiceContainer) -> "TaskAnalyzerProtocol":
            try:
                from victor.agent.task_analyzer import get_task_analyzer

                return get_task_analyzer()
            except ImportError:
                logger.warning("TaskAnalyzer not available")
                return _NullTaskAnalyzer()

        container.register(
            TaskAnalyzerProtocol,
            create_task_analyzer,
            ServiceLifetime.SINGLETON,
        )

    def _register_intent_classifier(self, container: ServiceContainer) -> None:
        """Register IntentClassifier as singleton."""
        from victor.storage.embeddings.intent_classifier import IntentClassifier

        container.register(
            IntentClassifier,
            lambda c: IntentClassifier.get_instance(),
            ServiceLifetime.SINGLETON,
        )

    def _create_complexity_classifier(self) -> "ComplexityClassifierProtocol":
        """Create ComplexityClassifier instance."""
        from victor.framework.task import TaskComplexityService as ComplexityClassifier

        return ComplexityClassifier()

    def _create_action_authorizer(self) -> "ActionAuthorizerProtocol":
        """Create ActionAuthorizer instance."""
        from victor.agent.action_authorizer import ActionAuthorizer

        return ActionAuthorizer()

    def _create_search_router(self) -> "SearchRouterProtocol":
        """Create SearchRouter instance."""
        from victor.agent.search_router import SearchRouter

        return SearchRouter()

    def _create_response_sanitizer(self) -> "ResponseSanitizerProtocol":
        """Create ResponseSanitizer instance."""
        from victor.agent.response_sanitizer import ResponseSanitizer

        return ResponseSanitizer()

    def _create_argument_normalizer(self) -> "ArgumentNormalizerProtocol":
        """Create ArgumentNormalizer instance."""
        from victor.agent.argument_normalizer import ArgumentNormalizer

        # Provider name will be updated when orchestrator is created
        return ArgumentNormalizer(provider_name="unknown")

    def _create_project_context(self) -> "ProjectContextProtocol":
        """Create ProjectContext instance."""
        from victor.context.project_context import ProjectContext

        context = ProjectContext()
        context.load()
        return context

    def _register_recovery_handler(self, container: ServiceContainer) -> None:
        """Register RecoveryHandler as singleton.

        The RecoveryHandler integrates with:
        - Q-learning for adaptive recovery strategy selection
        - UsageAnalytics for telemetry
        - ContextCompactor for proactive context management

        Note: Session-specific state (recent_responses, consecutive_failures)
        is reset via set_session_id() when orchestrator creates a new session.
        """
        from victor.agent.protocols import RecoveryHandlerProtocol

        def create_recovery_handler(_: ServiceContainer) -> "RecoveryHandlerProtocol":
            enabled = getattr(self._settings, "enable_recovery_system", True)
            if not enabled:
                return _NullRecoveryHandler()

            try:
                from victor.agent.recovery import RecoveryHandler

                return RecoveryHandler.create(settings=self._settings)
            except ImportError as e:
                logger.warning(f"RecoveryHandler not available: {e}")
                return _NullRecoveryHandler()
            except Exception as e:
                logger.warning(f"RecoveryHandler creation failed: {e}")
                return _NullRecoveryHandler()

        container.register(
            RecoveryHandlerProtocol,
            create_recovery_handler,
            ServiceLifetime.SINGLETON,
        )

    # =========================================================================
    # Factory methods for scoped services
    # =========================================================================

    def _create_conversation_state_machine(
        self,
        container: Optional[ServiceContainer] = None,
    ) -> "ConversationStateMachineProtocol":
        """Create ConversationStateMachine instance."""
        from victor.agent.conversation.state_machine import ConversationStateMachine
        from victor.agent.services.runtime_intelligence import (
            RuntimeIntelligenceService,
        )

        runtime_container = container or getattr(self, "container", None)
        runtime_intelligence = None
        if runtime_container is not None:
            runtime_intelligence = RuntimeIntelligenceService.from_container(runtime_container)

        return ConversationStateMachine(runtime_intelligence=runtime_intelligence)

    def _create_unified_task_tracker(self) -> "TaskTrackerProtocol":
        """Create UnifiedTaskTracker instance."""
        from victor.agent.unified_task_tracker import UnifiedTaskTracker

        return UnifiedTaskTracker()

    def _create_message_history(self) -> "MessageHistoryProtocol":
        """Create MessageHistory instance."""
        from victor.agent.message_history import MessageHistory

        return MessageHistory(
            system_prompt="",  # Will be set by orchestrator
            max_history_messages=getattr(self._settings, "max_conversation_history", 100000),
        )

    # =========================================================================
    # Factory methods for new framework services
    # =========================================================================

    def _register_path_resolver(self, container: ServiceContainer) -> None:
        """Register PathResolver as singleton."""
        from victor.protocols.path_resolver import PathResolver, IPathResolver

        container.register(
            IPathResolver,
            lambda c: PathResolver(),
            ServiceLifetime.SINGLETON,
        )

    def _register_unified_memory_coordinator(self, container: ServiceContainer) -> None:
        """Register UnifiedMemoryCoordinator as singleton.

        The UnifiedMemoryCoordinator provides federated search across all
        memory backends (entity, conversation, graph, embeddings) with
        pluggable ranking strategies.
        """
        from victor.agent.protocols import UnifiedMemoryCoordinatorProtocol

        def create_memory_coordinator(
            _: ServiceContainer,
        ) -> "UnifiedMemoryCoordinatorProtocol":
            try:
                from victor.storage.memory.unified import get_memory_coordinator

                return get_memory_coordinator()
            except ImportError as e:
                logger.warning(f"UnifiedMemoryCoordinator not available: {e}")
                return _NullMemoryCoordinator()

        container.register(
            UnifiedMemoryCoordinatorProtocol,
            create_memory_coordinator,
            ServiceLifetime.SINGLETON,
        )

    def _register_tool_access_controller(self, container: ServiceContainer) -> None:
        """Register ToolAccessController as scoped service."""
        from victor.agent.protocols import IToolAccessController
        from victor.agent.tool_access_controller import (
            create_tool_access_controller,
        )

        container.register(
            IToolAccessController,
            lambda c: create_tool_access_controller(
                registry=c.get_or_none("ToolRegistryProtocol"),
            ),
            ServiceLifetime.SCOPED,
        )

    def _register_tool_cache_manager(self, container: ServiceContainer) -> None:
        """Register ToolCacheManager as singleton.

        The ToolCacheManager provides centralized cache management for tools,
        replacing module-level caches like _INDEX_CACHE in code_search_tool.py
        and _connections in database_tool.py.

        This enables:
        - Proper test isolation (caches can be cleared between tests)
        - DI-based tool configuration
        - Unified cache statistics and monitoring
        """
        from victor.tools.cache_manager import ToolCacheManager

        container.register(
            ToolCacheManager,
            lambda c: ToolCacheManager(),
            ServiceLifetime.SINGLETON,
        )

    def _register_budget_manager(self, container: ServiceContainer) -> None:
        """Register BudgetManager as scoped service."""
        from victor.agent.protocols import IBudgetManager, BudgetConfig
        from victor.agent.budget_manager import create_budget_manager

        # Get budget settings from settings
        base_tool_calls = getattr(self._settings, "tool_budget", 30)
        base_iterations = getattr(self._settings, "max_iterations", 50)
        base_exploration = getattr(self._settings, "max_exploration_iterations", 8)
        base_action = getattr(self._settings, "max_action_iterations", 12)

        config = BudgetConfig(
            base_tool_calls=base_tool_calls,
            base_iterations=base_iterations,
            base_exploration=base_exploration,
            base_action=base_action,
        )

        container.register(
            IBudgetManager,
            lambda c: create_budget_manager(config=config),
            ServiceLifetime.SCOPED,
        )

    # =========================================================================
    # Infrastructure service factory methods
    # =========================================================================

    def _create_code_execution_manager(self) -> "CodeExecutionManagerProtocol":
        """Create CodeSandbox instance via dynamic vertical loading."""
        try:
            from victor.core.utils.capability_loader import load_code_executor_module

            module = load_code_executor_module()
            manager = module.CodeSandbox()
            manager.start()
            return manager
        except ImportError:
            # Provide a dummy implementation if victor-coding is not installed
            class DummyExecutionManager:
                def start(self):
                    pass

                def stop(self):
                    pass

            return DummyExecutionManager()

    def _create_workflow_registry(self) -> "WorkflowRegistryProtocol":
        """Create WorkflowRegistry instance."""
        from victor.workflows.base import WorkflowRegistry

        return WorkflowRegistry()

    def _create_usage_analytics(self) -> "UsageAnalyticsProtocol":
        """Create UsageAnalytics singleton instance."""
        from victor.agent.usage_analytics import UsageAnalytics, AnalyticsConfig
        from pathlib import Path

        # Use settings cache_dir if available
        analytics_cache_dir = (
            Path(self._settings.cache_dir)
            if hasattr(self._settings, "cache_dir") and self._settings.cache_dir
            else None
        )

        return UsageAnalytics.get_instance(
            AnalyticsConfig(
                cache_dir=analytics_cache_dir,
                enable_prometheus_export=getattr(self._settings, "enable_prometheus_export", True),
            )
        )

    def _create_tool_sequence_tracker(self) -> "ToolSequenceTrackerProtocol":
        """Create ToolSequenceTracker instance."""
        from victor.agent.tool_sequence_tracker import create_sequence_tracker

        return create_sequence_tracker(
            use_predefined=getattr(self._settings, "use_predefined_patterns", True),
            learning_rate=getattr(self._settings, "sequence_learning_rate", 0.3),
        )

    def _create_compaction_summarizer(self) -> Any:
        """Create CompactionSummarizer instance.

        Note: Returns None because the real instance needs a provider dependency
        which is not available at container initialization time.
        """
        return None

    def _create_hierarchical_compaction_manager(self) -> Any:
        """Create HierarchicalCompactionManager instance.

        Note: Returns None because the real instance needs a summarizer dependency
        which is created by the orchestrator at runtime.
        """
        return None

    def _create_session_context_linker(self) -> Any:
        """Create SessionContextLinker instance.

        Note: Returns None because the real instance needs a persistence dependency
        which is not available at container initialization time.
        """
        return None

    def _create_context_compactor(self) -> "ContextCompactorProtocol | None":
        """Create ContextCompactor instance.

        Note: This is a placeholder that will be replaced when the orchestrator
        creates the actual compactor with a ConversationController instance.
        The DI container registration is primarily for protocol compliance.
        """
        # ContextCompactor requires a ConversationController which is not available
        # at container initialization time. Return None and let the orchestrator
        # create it via the factory method when the controller is available.
        return None

    def _create_mode_controller(self) -> "ModeControllerProtocol":
        """Create AgentModeController instance."""
        from victor.agent.mode_controller import AgentModeController, AgentMode

        initial_mode_str = getattr(self._settings, "default_agent_mode", "build")
        try:
            initial_mode = AgentMode(initial_mode_str.lower())
        except ValueError:
            initial_mode = AgentMode.BUILD

        return AgentModeController(initial_mode=initial_mode)

    def _create_tool_deduplication_tracker(self) -> "ToolDeduplicationTrackerProtocol":
        """Create ToolDeduplicationTracker instance.

        .. deprecated::
            Use ToolCallTracker instead. This method now imports from
            tool_call_tracker module for backward compatibility.
        """
        from victor.agent.tool_call_tracker import ToolCallTracker

        window_size = getattr(self._settings, "dedup_window_size", 10)
        similarity_threshold = getattr(self._settings, "dedup_similarity_threshold", 0.7)

        return ToolCallTracker(
            window_size=window_size,
            similarity_threshold=similarity_threshold,
        )

    # =========================================================================
    # Utility service factory methods
    # =========================================================================

    def _create_debug_logger(self) -> "DebugLoggerProtocol":
        """Create DebugLogger instance."""
        from victor.agent.debug_logger import get_debug_logger

        return get_debug_logger()

    def _create_task_type_hinter(self) -> "TaskTypeHinterProtocol":
        """Create TaskTypeHinter wrapper."""
        from victor.core.capability_registry import CapabilityRegistry
        from victor.framework.vertical_protocols import TaskTypeHintProtocol

        hinter = CapabilityRegistry.get_instance().get(TaskTypeHintProtocol)
        if hinter is not None:
            return hinter

        # Fallback: return a no-op hinter
        from victor.contrib.prompts.task_hints import NullTaskTypeHinter

        return NullTaskTypeHinter()

    def _create_reminder_manager(self) -> "ReminderManagerProtocol":
        """Create ContextReminderManager instance."""
        from victor.agent.context_reminder import create_reminder_manager

        provider = getattr(self._settings, "provider", "unknown")
        task_complexity = "medium"

        # No tool_budget here on purpose. `Settings` has no top-level `tool_budget`,
        # so the previous `getattr(self._settings, "tool_budget", 10)` always yielded
        # the literal 10 — which the reminder then counted down against and stated to
        # the model as fact. The enforced budget arrives later via update_state().
        return create_reminder_manager(
            provider=provider,
            task_complexity=task_complexity,
        )

    def _create_rl_coordinator(self) -> "RLLearningRuntimeProtocol":
        """Create RLCoordinator instance."""
        from victor.agent.services.rl_runtime import get_rl_coordinator

        return get_rl_coordinator()

    def _create_safety_checker(self) -> "SafetyCheckerProtocol":
        """Create SafetyChecker instance."""
        from victor.agent.safety import get_safety_checker

        return get_safety_checker()

    def _create_auto_committer(self) -> "AutoCommitterProtocol":
        """Create AutoCommitter instance."""
        from victor.agent.auto_commit import get_auto_committer

        return get_auto_committer()

    def _create_mcp_bridge(self) -> "MCPBridgeProtocol":
        """Create MCP bridge wrapper."""
        from victor.tools.mcp_bridge_tool import get_mcp_tool_definitions

        class MCPBridge:
            """Wrapper for MCP bridge functionality."""

            def get_tool_definitions(self) -> list:
                """Return MCP tools as Victor tool definitions."""
                return get_mcp_tool_definitions()

        return MCPBridge()

    # =========================================================================
    # Infrastructure service factory methods
    # =========================================================================

    def _create_tool_dependency_graph(self) -> "ToolDependencyGraphProtocol":
        """Create ToolDependencyGraph instance."""
        from victor.tools.dependency_graph import ToolDependencyGraph

        return ToolDependencyGraph()

    def _create_tool_plugin_registry(self) -> "ToolPluginRegistryProtocol":
        """Create ToolPluginRegistry instance."""
        from victor.tools.plugin_registry import ToolPluginRegistry

        return ToolPluginRegistry()

    def _create_context_temperature_classifier(
        self,
    ) -> "ContextTemperatureClassifierProtocol":
        """Create the ContextTemperatureClassifier service."""
        from victor.agent.context_temperature import ContextTemperatureClassifier

        return ContextTemperatureClassifier()

    def _create_semantic_tool_selector(self) -> "SemanticToolSelectorProtocol":
        """Create SemanticToolSelector instance."""
        from victor.tools.semantic_selector import SemanticToolSelector

        # Get embedding model from settings
        embedding_model = getattr(self._settings, "embedding_model", "BAAI/bge-small-en-v1.5")
        use_semantic_selection = is_semantic_tool_selection_enabled(self._settings, default=True)

        if not use_semantic_selection:
            # Return a no-op selector that just returns all tools
            class NoOpSelector:
                def select_tools(
                    self,
                    query: str,
                    available_tools: list,
                    max_tools: int = 10,
                    threshold: float = 0.3,
                ) -> list:
                    return available_tools[:max_tools]

                def compute_similarity(self, query: str, tool_description: str) -> float:
                    return 1.0

            return NoOpSelector()

        return SemanticToolSelector(embedding_model=embedding_model)

    def _create_provider_registry(self) -> "ProviderRegistryProtocol":
        """Create ProviderRegistry wrapper.

        Note: ProviderRegistry is a class with class methods,
        so we return the class itself, not an instance.
        """
        from victor.providers.registry import ProviderRegistry

        # Return the class itself - it's used as a class with class methods
        return ProviderRegistry

    # =========================================================================
    # Analytics & observability service factory methods
    # =========================================================================

    def _create_conversation_embedding_store(
        self,
    ) -> "ConversationEmbeddingStoreProtocol":
        """Create ConversationEmbeddingStore instance."""
        from victor.agent.conversation_embedding_store import ConversationEmbeddingStore
        from victor.storage.embeddings.service import EmbeddingService
        from pathlib import Path

        embedding_model = getattr(self._settings, "embedding_model", "BAAI/bge-small-en-v1.5")
        cache_dir = Path(getattr(self._settings, "cache_dir", ".victor/cache"))

        # Create embedding service
        embedding_service = EmbeddingService(model_name=embedding_model)

        # Create embedding store with SQLite backend
        # Note: Uses separate embeddings database, not project.db or victor.db
        sqlite_path = cache_dir / "embeddings" / "conversation.db"

        return ConversationEmbeddingStore(
            embedding_service=embedding_service,
            sqlite_db_path=sqlite_path,
        )

    def _create_metrics_collector(self) -> "MetricsCollectorProtocol":
        """Create MetricsCollector instance."""
        from victor.agent.metrics_collector import (
            MetricsCollector,
            MetricsCollectorConfig,
        )
        from victor.observability.analytics.logger import UsageLogger
        from pathlib import Path

        # Create config with provider/model info
        model = getattr(self._settings, "model", "unknown")
        provider = getattr(self._settings, "provider", "unknown")
        analytics_enabled = getattr(self._settings, "enable_analytics", True)

        config = MetricsCollectorConfig(
            model=model,
            provider=provider,
            analytics_enabled=analytics_enabled,
        )

        # Create usage logger
        log_file = Path(getattr(self._settings, "usage_log_path", ".victor/logs/usage.log"))
        usage_logger = UsageLogger(log_file=log_file)

        return MetricsCollector(
            config=config,
            usage_logger=usage_logger,
        )

    def _create_tool_cache(self) -> "ToolCacheProtocol":
        """Create ToolCache instance."""
        from victor.storage.cache.tool_cache import ToolCache

        enabled = getattr(self._settings, "enable_tool_cache", True)
        ttl = getattr(self._settings, "tool_cache_ttl", 300)

        if not enabled:
            # Return a no-op cache
            class NoOpCache:
                def get(self, key: str) -> Any:
                    return None

                def set(self, key: str, value: Any, ttl: int = None) -> None:
                    pass

                def invalidate(self, key: str) -> None:
                    pass

                def clear(self) -> None:
                    pass

            return NoOpCache()

        return ToolCache(ttl=ttl)

    def _create_usage_logger(self) -> "UsageLoggerProtocol":
        """Create UsageLogger instance."""
        from victor.observability.analytics.logger import UsageLogger
        from pathlib import Path

        log_file = Path(getattr(self._settings, "usage_log_path", ".victor/logs/usage.log"))
        enabled = getattr(self._settings, "enable_usage_logging", True)

        return UsageLogger(log_file=log_file, enabled=enabled)

    def _create_streaming_metrics_collector(
        self,
    ) -> "StreamingMetricsCollectorProtocol":
        """Create StreamingMetricsCollector instance."""
        from victor.observability.analytics.streaming_metrics import (
            StreamingMetricsCollector,
        )
        from pathlib import Path

        enabled = getattr(self._settings, "streaming_metrics_enabled", True)
        max_history = getattr(self._settings, "streaming_metrics_history_size", 100)

        if not enabled:
            # Return a no-op collector
            class NoOpStreamingMetrics:
                def record_chunk(self, chunk_size: int, timestamp: float, **metadata: Any) -> None:
                    pass

                def get_metrics(self) -> dict:
                    return {}

                def reset(self) -> None:
                    pass

            return NoOpStreamingMetrics()

        export_path = (
            Path(getattr(self._settings, "cache_dir", ".victor/cache")) / "streaming_metrics.json"
        )

        return StreamingMetricsCollector(max_history=max_history, export_path=export_path)

    def _create_intent_classifier(self) -> "IntentClassifierProtocol":
        """Create IntentClassifier instance."""
        from victor.storage.embeddings.intent_classifier import IntentClassifier
        from victor.storage.embeddings.service import EmbeddingService
        from pathlib import Path

        embedding_model = getattr(self._settings, "embedding_model", "BAAI/bge-small-en-v1.5")
        cache_dir = (
            Path(getattr(self._settings, "cache_dir", ".victor/cache")) / "intent_classifier"
        )

        # Create embedding service
        embedding_service = EmbeddingService(model_name=embedding_model)

        return IntentClassifier(
            cache_dir=cache_dir,
            embedding_service=embedding_service,
        )

    # =========================================================================
    # Helper/adapter service factory methods
    # =========================================================================

    def _create_system_prompt_builder(self) -> "SystemPromptBuilderProtocol":
        """Create SystemPromptBuilder instance."""
        from victor.agent.prompt_builder import SystemPromptBuilder

        provider_name = getattr(self._settings, "provider", "anthropic")
        model = getattr(self._settings, "model", "claude-opus-4")

        return SystemPromptBuilder(
            provider_name=provider_name,
            model=model,
            concise_mode=True,  # Enable concise mode for more focused responses
        )

    def _create_tool_selector(self) -> "ToolSelectorProtocol":
        """Create ToolSelector instance."""
        from victor.agent.services.runtime_intelligence import (
            RuntimeIntelligenceService,
        )
        from victor.agent.tool_selection import ToolSelector
        from victor.tools.registry import ToolRegistry

        # Create a ToolRegistry instance for the ToolSelector
        tools = ToolRegistry()
        model = getattr(self._settings, "model", "claude-opus-4")
        provider_name = getattr(self._settings, "provider", "anthropic")
        fallback_max_tools = getattr(self._settings, "fallback_max_tools", 8)
        runtime_container = getattr(self, "container", None)
        runtime_intelligence = None
        if runtime_container is not None:
            runtime_intelligence = RuntimeIntelligenceService.from_container(runtime_container)

        return ToolSelector(
            tools=tools,
            model=model,
            provider_name=provider_name,
            fallback_max_tools=fallback_max_tools,
            runtime_intelligence=runtime_intelligence,
        )

    def _create_tool_executor(self) -> "ToolExecutorProtocol":
        """Create ToolExecutor instance."""
        from victor.agent.tool_executor import ToolExecutor
        from victor.tools.registry import ToolRegistry

        # Create a ToolRegistry instance for the ToolExecutor
        tool_registry = ToolRegistry()

        return ToolExecutor(tool_registry=tool_registry)

    def _create_tool_output_formatter(self) -> "ToolOutputFormatterProtocol":
        """Create ToolOutputFormatter instance."""
        from victor.agent.tool_output_formatter import create_tool_output_formatter

        # Factory function handles all defaults
        return create_tool_output_formatter()

    def _create_parallel_executor(self) -> "ParallelExecutorProtocol":
        """Create ParallelExecutor instance."""
        from victor.agent.parallel_executor import create_parallel_executor
        from victor.agent.tool_executor import ToolExecutor
        from victor.tools.registry import ToolRegistry

        # Create ToolExecutor instance needed by parallel executor
        tool_registry = ToolRegistry()
        tool_executor = ToolExecutor(tool_registry=tool_registry)

        max_concurrent = getattr(self._settings, "max_concurrent_tools", 5)
        enable = getattr(self._settings, "enable_parallel_execution", True)

        return create_parallel_executor(
            tool_executor=tool_executor,
            max_concurrent=max_concurrent,
            enable=enable,
        )

    def _create_response_completer(self) -> "ResponseCompleterProtocol":
        """Create ResponseCompleter instance.

        Note: This returns a no-op completer since we don't have access to the provider
        instance here. The actual ResponseCompleter should be created when the provider
        is available.
        """
        from victor.agent.response_completer import create_response_completer

        # Create a mock provider for now - in practice, this would need the actual provider
        class MockProvider:
            """Mock provider for ResponseCompleter."""

            async def chat(self, *args, **kwargs):
                return {"content": "", "model": "mock"}

        mock_provider = MockProvider()
        max_retries = getattr(self._settings, "max_response_retries", 3)
        force_response = getattr(self._settings, "force_response", True)

        return create_response_completer(
            provider=mock_provider,
            max_retries=max_retries,
            force_response=force_response,
        )

    def _create_streaming_handler(self) -> "StreamingHandlerProtocol":
        """Create StreamingHandler instance."""
        from victor.agent.streaming import StreamingChatHandler

        # Create a mock message adder for now
        class MockMessageAdder:
            """Mock message adder for StreamingChatHandler."""

            def add_message(self, role: str, content: str, **metadata):
                pass

        message_adder = MockMessageAdder()
        session_idle_timeout = getattr(self._settings, "session_idle_timeout", 180.0)

        return StreamingChatHandler(
            settings=self._settings,
            message_adder=message_adder,
            session_idle_timeout=session_idle_timeout,
        )

    def _create_confidence_monitor(self) -> "StreamingConfidenceMonitorProtocol":
        """Create the StreamingConfidenceMonitor service."""
        from victor.agent.streaming.confidence_monitor import StreamingConfidenceMonitor

        return StreamingConfidenceMonitor()

    def _create_recovery_coordinator(self) -> "StreamingRecoveryRuntimeProtocol":
        """Create RecoveryService instance (canonical replacement for StreamingRecoveryCoordinator).

        The RecoveryService centralizes all recovery and error handling logic
        for streaming chat sessions, including condition checking, action handling,
        and recovery integration.

        Returns:
            RecoveryService instance with native streaming runtime enabled
        """
        from victor.agent.services.recovery_service import RecoveryService
        from victor.agent.protocols import (
            RecoveryHandlerProtocol,
            ContextCompactorProtocol,
            TaskTrackerProtocol,
        )
        from victor.agent.services.protocols import StreamingHandlerProtocol

        # Create RecoveryService with native streaming runtime
        recovery_service = RecoveryService()

        # Get streaming handler from DI container
        streaming_handler = self.container.get(StreamingHandlerProtocol)

        # Get context compactor from DI container (optional)
        context_compactor = self.container.get_optional(ContextCompactorProtocol)

        # Get unified tracker from DI container
        unified_tracker = self.container.get(TaskTrackerProtocol)

        # Get recovery handler from DI container (optional)
        recovery_handler = self.container.get_optional(RecoveryHandlerProtocol)

        # Get recovery integration (optional)
        recovery_integration = None

        # Get presentation adapter (optional)
        presentation = None
        try:
            from victor.agent.presentation import create_presentation_adapter

            presentation = create_presentation_adapter()
        except Exception:
            pass

        # Bind runtime components to enable native streaming
        recovery_service.bind_runtime_components(
            streaming_handler=streaming_handler,
            context_compactor=context_compactor,
            unified_tracker=unified_tracker,
            recovery_handler=recovery_handler,
            recovery_integration=recovery_integration,
            settings=self._settings,
            presentation=presentation,
        )

        return recovery_service

    def _create_chunk_generator(self) -> "ChunkRuntimeProtocol":
        """Create ChunkGenerator instance.

        The ChunkGenerator provides a centralized interface for generating streaming
        chunks for various purposes (tool execution, status updates, metrics, content).

        Returns:
            ChunkGenerator instance
        """
        from victor.agent.services.chunk_runtime import ChunkGenerator
        from victor.agent.services.protocols import StreamingHandlerProtocol

        # Get streaming handler from DI container
        streaming_handler = self.container.get(StreamingHandlerProtocol)

        return ChunkGenerator(
            streaming_handler=streaming_handler,
            settings=self._settings,
        )

    def _create_tool_planner(self) -> "ToolPlanningRuntimeProtocol":
        """Create ToolPlanner instance.

        The ToolPlanner provides a centralized interface for tool planning operations,
        including goal inference, tool sequence planning, and intent-based filtering.

        Returns:
            ToolPlanner instance
        """
        from victor.agent.services.tool_planning_runtime import ToolPlanner
        from victor.agent.protocols import ToolRegistrarProtocol

        # Get tool registrar from DI container
        tool_registrar = self.container.get(ToolRegistrarProtocol)

        return ToolPlanner(
            tool_registrar=tool_registrar,
            settings=self._settings,
        )

    def _create_task_coordinator(self) -> "TaskRuntimeProtocol":
        """Create TaskCoordinator instance.

        The TaskCoordinator provides a centralized interface for task coordination,
        including task preparation, intent detection, and task-specific guidance.

        Note: ConversationController is not injected as it hasn't been migrated
        to DI yet. It's passed as a parameter when calling TaskCoordinator methods.

        Returns:
            TaskCoordinator instance
        """
        from victor.agent.services.task_runtime import TaskCoordinator
        from victor.agent.protocols import (
            TaskAnalyzerProtocol,
            TaskTrackerProtocol,
            SystemPromptBuilderProtocol,
        )

        # Get dependencies from DI container
        task_analyzer = self.container.get(TaskAnalyzerProtocol)
        unified_tracker = self.container.get(TaskTrackerProtocol)
        prompt_builder = self.container.get(SystemPromptBuilderProtocol)

        return TaskCoordinator(
            task_analyzer=task_analyzer,
            unified_tracker=unified_tracker,
            prompt_builder=prompt_builder,
            settings=self._settings,
        )

    def _create_coordination_advisor_runtime(
        self,
    ) -> "CoordinationAdvisorRuntimeProtocol":
        """Create the service-owned coordination runtime adapter."""
        from victor.agent.services.coordination_advisor_runtime import (
            CoordinationAdvisorRuntime,
        )

        return CoordinationAdvisorRuntime()

    # =========================================================================
    # New Coordinator Factory Methods (WS-D: Orchestrator SOLID Fixes)
    # =========================================================================

    def _create_state_runtime(
        self,
        container: ServiceContainer,
    ) -> "StateRuntimeProtocol | None":
        """Create the canonical live state runtime adapter."""
        from victor.agent.services.state_runtime import StateRuntimeAdapter
        from victor.agent.protocols import (
            ConversationControllerProtocol,
            ConversationStateMachineProtocol,
        )

        # Get dependencies from DI container
        conversation_controller = container.get_optional(ConversationControllerProtocol)
        state_machine = container.get_optional(ConversationStateMachineProtocol)

        if conversation_controller is None:
            logger.debug("ConversationController not available for StateRuntimeAdapter")
            return None

        return StateRuntimeAdapter(
            conversation_controller=conversation_controller,
            state_machine=state_machine,
        )

    def _create_prompt_runtime(self) -> "PromptRuntimeProtocol":
        """Create the canonical prompt runtime adapter."""
        from victor.agent.services.prompt_runtime import (
            PromptRuntimeAdapter,
            PromptRuntimeConfig,
        )
        from victor.agent.system_prompt_policy import create_policy_from_settings
        from victor.framework.prompt_builder import PromptBuilder

        config = PromptRuntimeConfig(
            default_grounding_mode=getattr(self._settings, "grounding_mode", "minimal"),
            enable_task_hints=getattr(self._settings, "enable_task_hints", True),
            enable_vertical_sections=True,
            enable_safety_rules=True,
            max_context_tokens=getattr(self._settings, "max_context_tokens", 2000),
        )

        base_identity = getattr(self._settings, "base_identity", None)
        policy = create_policy_from_settings(self._settings)

        return PromptRuntimeAdapter(
            prompt_builder=PromptBuilder(),
            config=config,
            base_identity=base_identity,
            policy=policy,
        )

    # =========================================================================
    # Presentation Abstraction Layer Factory Methods
    # =========================================================================

    def _register_presentation_adapter(self, container: ServiceContainer) -> None:
        """Register PresentationAdapter as singleton.

        The PresentationAdapter provides a clean abstraction for icon/emoji
        rendering, decoupling the agent layer from direct UI dependencies.

        Uses EmojiPresentationAdapter by default, which respects the
        `use_emojis` setting from configuration.
        """
        from victor.agent.presentation import (
            PresentationProtocol,
            EmojiPresentationAdapter,
        )

        container.register(
            PresentationProtocol,
            lambda c: EmojiPresentationAdapter(),
            ServiceLifetime.SINGLETON,
        )


# =============================================================================
# Null implementations for graceful degradation
# =============================================================================


class _NullObservability:
    """No-op observability implementation."""

    def on_tool_start(self, tool_name: str, arguments: dict, tool_id: str) -> None:
        pass

    def on_tool_end(
        self,
        tool_name: str,
        result: Any,
        success: bool,
        tool_id: str,
        error: Optional[str] = None,
    ) -> None:
        pass

    def wire_state_machine(self, state_machine: Any) -> None:
        pass

    def on_error(self, error: Exception, context: dict) -> None:
        pass


class _NullTaskAnalyzer:
    """No-op task analyzer implementation."""

    def analyze(self, prompt: str) -> dict:
        return {"complexity": "unknown", "intent": "unknown"}

    def classify_complexity(self, prompt: str) -> Any:
        return None

    def detect_intent(self, prompt: str) -> Any:
        return None


class _NullRecoveryHandler:
    """No-op recovery handler implementation for disabled mode."""

    @property
    def enabled(self) -> bool:
        return False

    @property
    def consecutive_failures(self) -> int:
        return 0

    def detect_failure(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def recover(self, *args: Any, **kwargs: Any) -> Any:
        # Return a no-op outcome
        from dataclasses import dataclass, field
        from enum import Enum, auto

        class _RecoveryAction(Enum):
            CONTINUE = auto()

        @dataclass
        class _RecoveryResult:
            action: _RecoveryAction = _RecoveryAction.CONTINUE
            success: bool = True
            strategy_name: str = "disabled"
            reason: str = "Recovery system disabled"

        @dataclass
        class _RecoveryOutcome:
            result: _RecoveryResult = field(default_factory=_RecoveryResult)

        return _RecoveryOutcome()

    def record_outcome(self, success: bool, quality_improvement: float = 0.0) -> None:
        pass

    def track_response(self, content: str) -> None:
        pass

    def reset_session(self, session_id: str) -> None:
        pass

    def set_context_compactor(self, compactor: Any) -> None:
        pass

    def set_session_id(self, session_id: str) -> None:
        pass

    def get_diagnostics(self) -> dict:
        return {"enabled": False}


class _NullMemoryCoordinator:
    """No-op memory coordinator implementation for disabled mode."""

    async def search_all(
        self,
        query: str,
        limit: int = 20,
        memory_types: Optional[list] = None,
        session_id: Optional[str] = None,
        filters: Optional[dict] = None,
        min_relevance: float = 0.0,
    ) -> list:
        return []

    async def search_type(
        self,
        memory_type: Any,
        query: str,
        limit: int = 20,
        **kwargs: Any,
    ) -> list:
        return []

    async def store(
        self,
        memory_type: Any,
        key: str,
        value: Any,
        metadata: Optional[dict] = None,
    ) -> bool:
        return False

    async def get(
        self,
        memory_type: Any,
        key: str,
    ) -> Optional[Any]:
        return None

    def register_provider(self, provider: Any) -> None:
        pass

    def unregister_provider(self, memory_type: Any) -> bool:
        return False

    def get_registered_types(self) -> list:
        return []

    def get_stats(self) -> dict:
        return {"enabled": False, "providers": []}


# =============================================================================
# Convenience function
# =============================================================================


def configure_orchestrator_services(
    container: ServiceContainer,
    settings: "Settings",
) -> None:
    """Configure all orchestrator services in one call.

    Convenience function for application bootstrap. Creates a service
    provider and registers all services.

    Args:
        container: DI container to register services in
        settings: Application settings

    Example:
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services

        container = ServiceContainer()
        configure_orchestrator_services(container, settings)
    """
    provider = OrchestratorServiceProvider(settings)
    provider.register_services(container)


__all__ = [
    "OrchestratorServiceProvider",
    "configure_orchestrator_services",
]
