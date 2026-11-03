"""Mirror tests for victor.agent.orchestrator_init_runtime.

Pins the eight _initialize_*_runtime helpers moved verbatim out of
victor/agent/orchestrator.py (facade decomposition): component wiring,
the canonical service constructions, phase-manager binding, and the
credit-tracking enable/disable branches.
"""

import inspect
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.providers.registry import ProviderRegistry


def make_orch():
    return object.__new__(AgentOrchestrator)


def patched_state(orch, **values):
    stack = ExitStack()
    for name, value in values.items():
        owner = next((b for b in type(orch).__mro__ if name in vars(b)), None)
        if owner is not None:
            stack.enter_context(patch.object(owner, name, property(lambda self, v=value: v)))
        else:
            setattr(orch, name, value)
    return stack


class TestProviderRuntime:
    def test_provider_service_is_canonical_with_module_registry(self):
        orch = make_orch()
        orch.settings = MagicMock()
        orch._provider_manager = MagicMock()
        svc = MagicMock()
        comps = MagicMock()
        with ExitStack() as stack:
            ctor = stack.enter_context(
                patch(
                    "victor.agent.services.provider_service.ProviderService",
                    return_value=svc,
                )
            )
            create = stack.enter_context(
                patch(
                    "victor.agent.runtime.provider_runtime" ".create_provider_runtime_components",
                    return_value=comps,
                )
            )
            AgentOrchestrator._initialize_provider_runtime(orch)
        assert orch._provider_service is svc
        assert orch._provider_runtime is comps
        # Pin: the service is built with the module-level ProviderRegistry —
        # the same object exposed as victor.agent.orchestrator.ProviderRegistry
        # (a live patch target in tests/test_thinking_mode.py).
        import victor.agent.orchestrator as orch_mod

        assert ctor.call_args[1]["registry"] is ProviderRegistry
        assert orch_mod.ProviderRegistry is ProviderRegistry
        _, kwargs = create.call_args
        assert kwargs["settings"] is orch.settings
        assert kwargs["provider_manager"] is orch._provider_manager


class TestMemoryRuntime:
    def test_session_service_construction_and_embeddings_skipped(self):
        orch = make_orch()
        orch.settings = MagicMock()
        orch.settings.conversation_embeddings_enabled = False
        orch._factory = MagicMock()
        orch.provider_name = "p"
        orch.tool_calling_caps = MagicMock()
        orch._session_state = MagicMock()
        orch._checkpoint_manager = None
        orch._cost_tracker = None
        runtime = MagicMock()
        svc = MagicMock()
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "victor.agent.runtime.memory_runtime" ".create_memory_runtime_components",
                    return_value=runtime,
                )
            )
            ctor = stack.enter_context(
                patch(
                    "victor.agent.services.session_service.SessionService",
                    return_value=svc,
                )
            )
            AgentOrchestrator._initialize_memory_runtime(orch)
        assert orch.memory_manager is runtime.memory_manager
        assert orch._memory_session_id is runtime.memory_session_id
        assert orch._session_service is svc
        _, kwargs = ctor.call_args
        assert kwargs["session_state_manager"] is orch._session_state
        assert kwargs["lifecycle_manager"] is None


class TestMetricsRuntime:
    def test_components_wired_and_session_start_logged(self):
        orch = make_orch()
        orch.settings = MagicMock()
        orch._factory = MagicMock()
        orch.provider = MagicMock()
        orch.model = "m"
        orch.debug_logger = MagicMock()
        orch.tools = MagicMock()
        runtime = MagicMock()
        with patched_state(orch, _cumulative_token_usage={"input": 0}):
            with ExitStack() as stack:
                stack.enter_context(
                    patch(
                        "victor.agent.runtime.metrics_runtime" ".create_metrics_runtime_components",
                        return_value=runtime,
                    )
                )
                stack.enter_context(
                    patch(
                        "victor.observability.analytics.trace_enrichment" ".create_trace_enricher",
                        side_effect=ImportError("no enricher"),
                    )
                )
                AgentOrchestrator._initialize_metrics_runtime(orch)
        assert orch._metrics_coordinator is runtime.metrics_coordinator
        assert orch.streaming_metrics_collector is runtime.streaming_metrics_collector
        # Trace-enricher unavailable -> raw usage_logger stays and logs start.
        assert orch.usage_logger is runtime.usage_logger
        orch.usage_logger.log_event.assert_called_once_with(
            "session_start", {"model": "m", "provider": orch.provider.__class__.__name__}
        )


class TestWorkflowCoordinationResilience:
    def test_workflow_runtime(self):
        orch = make_orch()
        orch._factory = MagicMock()
        runtime = MagicMock()
        with patch(
            "victor.agent.runtime.workflow_runtime" ".create_workflow_runtime_components",
            return_value=runtime,
        ):
            AgentOrchestrator._initialize_workflow_runtime(orch)
        assert orch._workflow_registry is runtime.workflow_registry

    def test_coordination_runtime_fanout(self):
        orch = make_orch()
        orch._factory = MagicMock()
        orch._task_analyzer = MagicMock()
        runtime = MagicMock()
        with patch(
            "victor.agent.runtime.coordination_runtime" ".create_coordination_runtime_components",
            return_value=runtime,
        ):
            AgentOrchestrator._initialize_coordination_runtime(orch)
        assert orch._recovery_coordinator is runtime.recovery_coordinator
        assert orch._task_coordinator is runtime.task_coordinator
        orch._task_analyzer.set_coordination_runtime.assert_called_once_with(
            runtime.coordination_advisor_runtime
        )

    def test_resilience_runtime(self):
        orch = make_orch()
        orch._factory = MagicMock()
        compactor = MagicMock()
        runtime = MagicMock()
        with patch(
            "victor.agent.runtime.resilience_runtime" ".create_resilience_runtime_components",
            return_value=runtime,
        ) as create:
            AgentOrchestrator._initialize_resilience_runtime(orch, context_compactor=compactor)
        assert create.call_args[1]["context_compactor"] is compactor
        assert orch._recovery_handler is runtime.recovery_handler


class TestInteractionRuntime:
    def test_six_services_assigned_from_runtime(self):
        orch = make_orch()
        orch.settings = MagicMock()
        orch._factory = MagicMock()
        orch._tool_pipeline = MagicMock()
        orch.tools = MagicMock()
        orch.tool_budget = 30
        orch._session_state = MagicMock()
        orch._lifecycle_manager = MagicMock()
        orch.memory_manager = MagicMock()
        orch._memory_session_id = "s1"
        orch._checkpoint_manager = MagicMock()
        orch._session_cost_tracker = MagicMock()
        orch._conversation_controller = MagicMock()
        orch._streaming_controller = MagicMock()
        orch.tool_selector = MagicMock()
        orch.mode_controller = MagicMock()
        runtime = MagicMock()
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "victor.agent.runtime.interaction_runtime"
                    ".create_interaction_runtime_components",
                    return_value=runtime,
                )
            )
            stack.enter_context(patch("victor.runtime.context.resolve_runtime_services"))
            AgentOrchestrator._initialize_interaction_runtime(orch)
        assert orch._chat_service is runtime.chat_service
        assert orch._tool_service is runtime.tool_service
        assert orch._session_service is runtime.session_service
        assert orch._context_service is runtime.context_service
        assert orch._recovery_service is runtime.recovery_service


class TestCreditRuntime:
    def test_disabled_by_default(self):
        # Negative: no settings.credit_assignment -> service stays None.
        orch = make_orch()
        orch.settings = MagicMock()
        orch.settings.credit_assignment = None
        AgentOrchestrator._initialize_credit_runtime(orch)
        assert orch._credit_tracking_service is None

    def test_enabled_path_registers_service(self):
        orch = make_orch()
        orch.settings = MagicMock()
        orch.settings.credit_assignment = MagicMock(enabled=True)
        orch._observability_bus = MagicMock()
        orch._tool_pipeline = MagicMock()
        orch._container = MagicMock()
        svc = MagicMock()
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "victor.framework.rl.credit_tracking_service"
                    ".CreditTrackingService.from_settings",
                    return_value=svc,
                )
            )
            stack.enter_context(patch("victor.core.container.ServiceLifetime", create=True))
            AgentOrchestrator._initialize_credit_runtime(orch)
        assert orch._credit_tracking_service is svc
        assert orch._tool_pipeline._credit_tracking_service is svc
        orch._container.register_or_replace.assert_called_once()


class TestPhaseBindingAndResidency:
    def test_initialization_phase_manager_finds_methods(self):
        # The phase manager binds phases by attribute access on the instance;
        # MRO resolution must find them on the mixin.
        from victor.agent.runtime.initialization_manager import (
            InitializationPhaseManager,
        )

        manager = InitializationPhaseManager()
        orch = make_orch()
        for method in (
            "_initialize_provider_runtime",
            "_initialize_memory_runtime",
            "_initialize_metrics_runtime",
            "_initialize_workflow_runtime",
            "_initialize_coordination_runtime",
            "_initialize_resilience_runtime",
            "_initialize_interaction_runtime",
            "_initialize_credit_runtime",
        ):
            assert callable(getattr(orch, method, None)), method

    def test_methods_live_on_mixin_and_services_stay(self):
        # Negative pin: helpers moved; _initialize_services must stay in the
        # class body (six-protocol import guard + delegation floor tests).
        class_src = inspect.getsource(AgentOrchestrator)
        for name in ("_initialize_provider_runtime", "_initialize_credit_runtime"):
            assert f"def {name}(" not in class_src, name
            path = getattr(AgentOrchestrator, name).__code__.co_filename
            assert path.endswith("orchestrator_init_runtime.py"), (name, path)
        assert "def _initialize_services(" in class_src
