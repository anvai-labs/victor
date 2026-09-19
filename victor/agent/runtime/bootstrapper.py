"""Post-initialization bootstrapper for AgentOrchestrator.

Extracts domain facade creation, lifecycle wiring, protocol conformance
assertions, and session context setup from the orchestrator's __init__.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from victor.agent.runtime.provider_runtime import LazyRuntimeProxy

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


class AgentRuntimeBootstrapper:
    """Assembles domain facades and wires post-initialization dependencies.

    All methods are static and operate on a fully-initialized orchestrator
    instance. Called at the end of AgentOrchestrator.__init__ after all
    components have been created by the factory and runtime boundaries.
    """

    @staticmethod
    def create_facades(orchestrator: AgentOrchestrator) -> None:
        """Create the live OrchestrationFacade and set it on the orchestrator.

        The seven per-domain facades (Chat/Tool/Session/Provider/Resilience/
        Workflow/Metrics) were removed as dead parallel views (zero production
        readers). OrchestrationFacade is the one runtime-facing boundary.
        """
        from victor.agent.facades import OrchestrationFacade

        orchestrator._orchestration_facade = LazyRuntimeProxy(
            factory=lambda: OrchestrationFacade(
                interaction_runtime=orchestrator._interaction_runtime,
                chat_service=getattr(orchestrator, "_chat_service", None),
                get_chat_stream_adapter=orchestrator._get_chat_stream_adapter,
                tool_service=getattr(orchestrator, "_tool_service", None),
                session_service=getattr(orchestrator, "_session_service", None),
                context_service=getattr(orchestrator, "_context_service", None),
                provider_service=getattr(orchestrator, "_provider_service", None),
                recovery_service=getattr(orchestrator, "_recovery_service", None),
                turn_executor=orchestrator._turn_executor,
                protocol_adapter=orchestrator._protocol_adapter,
                streaming_handler=orchestrator._streaming_handler,
                streaming_controller=orchestrator._streaming_controller,
                streaming_coordinator=orchestrator._streaming_coordinator,
                iteration_coordinator=getattr(orchestrator, "_iteration_coordinator", None),
                task_analyzer=orchestrator._task_analyzer,
                exploration_state_passed=(
                    orchestrator._factory.create_exploration_state_passed_coordinator()
                ),
                system_prompt_state_passed=(
                    orchestrator._factory.create_system_prompt_state_passed_coordinator(
                        task_analyzer=orchestrator._task_analyzer,
                    )
                ),
                safety_state_passed=(
                    orchestrator._factory.create_safety_state_passed_coordinator()
                ),
                coordination_state_passed=(
                    orchestrator._factory.create_coordination_state_passed_coordinator(
                        coordination_runtime=orchestrator._coordination_advisor_runtime,
                        coordination_advisor=getattr(orchestrator, "_coordination_advisor", None),
                        vertical_context=orchestrator._vertical_context,
                    )
                ),
                presentation=orchestrator._presentation,
                vertical_integration_adapter=(orchestrator._vertical_integration_adapter),
                vertical_context=orchestrator._vertical_context,
                observability=orchestrator._observability,
                execution_tracer=getattr(orchestrator, "_execution_tracer", None),
                tool_call_tracer=getattr(orchestrator, "_tool_call_tracer", None),
                runtime_state_host=orchestrator,
                get_runtime_intelligence_integration=(
                    lambda: getattr(orchestrator, "runtime_intelligence_integration", None)
                ),
                get_subagent_orchestrator=(
                    lambda: getattr(orchestrator, "subagent_orchestrator", None)
                ),
            ),
            name="orchestration_facade",
        )

        logger.debug("OrchestrationFacade created")

    @staticmethod
    def wire_lifecycle(orchestrator: AgentOrchestrator) -> None:
        """Wire LifecycleManager with component dependencies for shutdown."""
        lm = orchestrator._lifecycle_manager
        lm.set_provider(orchestrator.provider)
        lm.set_code_manager(
            orchestrator.code_manager if hasattr(orchestrator, "code_manager") else None
        )
        lm.set_semantic_selector(
            orchestrator.semantic_selector if hasattr(orchestrator, "semantic_selector") else None
        )
        lm.set_usage_logger(
            orchestrator.usage_logger if hasattr(orchestrator, "usage_logger") else None
        )
        lm.set_background_tasks(list(orchestrator._background_tasks))
        lm.set_flush_analytics_callback(orchestrator.flush_analytics)
        lm.set_stop_health_monitoring_callback(orchestrator.stop_health_monitoring)

    @staticmethod
    def assert_protocol_conformance(orchestrator: AgentOrchestrator) -> None:
        """Debug-mode assertion for the legacy ChatOrchestratorProtocol shim."""
        if __debug__:
            from victor.agent.services.protocols.chat_runtime import (
                ChatOrchestratorProtocol,
            )
            from victor.framework.protocols import (
                verify_protocol_conformance,
            )

            _conforms, _missing = verify_protocol_conformance(
                orchestrator, ChatOrchestratorProtocol
            )
            assert _conforms, (
                "AgentOrchestrator missing " f"ChatOrchestratorProtocol members: {_missing}"
            )

    @staticmethod
    def setup_session_context(orchestrator: AgentOrchestrator) -> None:
        """Initialize session context for tracing/propagation."""
        from victor.core.context import set_session_id

        set_session_id(orchestrator.active_session_id or "")

    @staticmethod
    def prepare_components(orchestrator: AgentOrchestrator, settings: Any) -> None:
        """Create post-factory components and wire dependencies.

        Handles checkpoint manager, workflow optimization, vertical context,
        runtime boundaries, and coordinator placeholders. Called after factory
        component creation but before facade assembly.
        """
        from victor.agent.vertical_context import (
            VerticalContext,
            create_vertical_context,
        )
        from victor.agent.vertical_integration_adapter import (
            VerticalIntegrationAdapter,
        )

        # Checkpoint manager for time-travel debugging
        orchestrator._checkpoint_manager = orchestrator._factory.create_checkpoint_manager()

        # Wire component dependencies
        orchestrator._factory.wire_component_dependencies(
            recovery_handler=None,
            context_compactor=orchestrator._context_compactor,
            observability=orchestrator._observability,
            conversation_state=orchestrator.conversation_state,
        )

        # Vertical context and integration adapter
        orchestrator._vertical_context: VerticalContext = create_vertical_context()
        orchestrator._vertical_integration_adapter = VerticalIntegrationAdapter(orchestrator)

        # Lazy coordinator placeholders must exist before any runtime wiring that
        # may resolve protocol-based lazy properties.
        orchestrator._coordination_advisor = None
        orchestrator._coordination_advisor_runtime = None
        orchestrator._turn_executor = None
        orchestrator._protocol_adapter = None
        # Interaction / service-layer / credit runtime boundaries (FEP-0016: driven by
        # the init manager at their existing sites). Credit is opt-in
        # (settings.credit_assignment.enabled) and a no-op when disabled.
        orchestrator._init_manager.run_phase(orchestrator, "interaction_runtime")
        orchestrator._init_manager.run_phase(orchestrator, "services")
        orchestrator._init_manager.run_phase(orchestrator, "credit_runtime")

        # Capability registry
        orchestrator.__init_capability_registry__()

    @classmethod
    def finalize(cls, orchestrator: AgentOrchestrator) -> None:
        """Run all post-initialization steps in order.

        Called at the end of AgentOrchestrator.__init__ after all
        components and runtime boundaries are initialized.
        """
        cls.create_facades(orchestrator)
        cls.wire_lifecycle(orchestrator)
        cls.assert_protocol_conformance(orchestrator)
        cls.setup_session_context(orchestrator)

        logger.info(
            "Orchestrator initialized with decomposed components: "
            "ConversationController, ToolPipeline, StreamingController, "
            "StreamingChatHandler, TaskAnalyzer, ContextCompactor, "
            "UsageAnalytics, ToolSequenceTracker, ToolOutputFormatter, "
            "RecoveryCoordinator, ChunkGenerator, ToolPlanner, "
            "TaskCoordinator, ObservabilityIntegration, "
            "WorkflowOptimization, VerticalContext, "
            "CoordinationAdvisorRuntime, CapabilityRegistry"
        )
