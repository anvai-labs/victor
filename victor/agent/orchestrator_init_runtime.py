"""Init-phase runtime wiring for the orchestrator.

Moved verbatim from ``victor/agent/orchestrator.py`` (facade decomposition):
the eight _initialize_*_runtime helpers bound by InitializationPhaseManager.
_initialize_services stays in the class body (service-delegation floors
and the six ServiceProtocol import guard).
"""

import logging
from typing import Any

from victor.providers.registry import ProviderRegistry
from victor.tools.enums import CostTier

logger = logging.getLogger(__name__)


class OrchestratorInitRuntimeMixin:
    """Moved verbatim from orchestrator.py; see module docstring."""

    def _initialize_provider_runtime(self) -> None:
        """Initialize provider runtime boundaries with canonical service.

        Migration Note (2026-05-01):
        - ProviderCoordinator removed from runtime components (unused internally)
        - ProviderService is the canonical owner for provider operations
        """
        from victor.agent.runtime.provider_runtime import create_provider_runtime_components
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
        from victor.agent.runtime.metrics_runtime import create_metrics_runtime_components

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
            from victor.observability.analytics.trace_enrichment import create_trace_enricher

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
        from victor.agent.runtime.workflow_runtime import create_workflow_runtime_components

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
            settings=self.settings,
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
