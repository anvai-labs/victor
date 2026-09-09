import importlib
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.services.orchestrator_protocol_adapter import (
    OrchestratorProtocolAdapter,
)
from victor.agent.services.chat_stream_runtime import ServiceStreamingRuntime
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import SessionStateManager
from victor.agent.streaming.context import StreamingChatContext
from victor.agent.unified_task_tracker import TrackerTaskType
from victor.agent.topology_contract import (
    TopologyAction,
    TopologyDecision,
    TopologyDecisionInput,
    TopologyGroundingRequirements,
    TopologyKind,
)
from victor.agent.topology_grounder import GroundedTopologyPlan
from victor.framework.routing_policy import StructuredRoutingPolicy
from victor.framework.task import TaskComplexity
from victor.framework.team_runtime import ResolvedTeamExecutionPlan
from victor.providers.base import StreamChunk
from victor.teams.types import TeamFormation


def _make_orchestrator_stub():
    orch = MagicMock()
    orch._session_accessor = SessionStateAccessor(SessionStateManager())
    orch.has_capability.return_value = False
    orch.get_capability_value.return_value = None
    orch._cumulative_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    orch._conversation_controller = MagicMock()
    orch.messages = []
    return orch


def test_streaming_runtime_bindings_resolve_adapter_state_and_capabilities():
    orch = _make_orchestrator_stub()
    adapter = OrchestratorProtocolAdapter(orch)
    orch._perception_integration = "perception"

    binding = ServiceStreamingRuntime(adapter)._get_runtime_bindings(adapter)

    assert binding.state_host is orch
    assert binding.state_dict is orch.__dict__
    assert binding.get_capability_value("perception_integration") == "perception"
    assert binding.has_capability("perception_integration") is True


def test_service_streaming_runtime_caches_executor(monkeypatch):
    orch = _make_orchestrator_stub()
    runtime = ServiceStreamingRuntime(orch)
    created = []

    class DummyExecutor:
        pass

    def fake_factory(owner, **kwargs):
        created.append((owner, kwargs))
        return DummyExecutor()

    service_module = importlib.import_module("victor.agent.services.chat_stream_executor")

    monkeypatch.setattr(service_module, "create_streaming_chat_executor", fake_factory)

    first = runtime.get_executor()
    second = runtime.get_executor()

    assert first is second
    assert len(created) == 1
    owner, kwargs = created[0]
    assert owner is runtime
    assert kwargs["perception"] is None
    assert kwargs["fulfillment"] is None
    assert kwargs["runtime_intelligence"] is orch._runtime_intelligence


def test_service_streaming_runtime_exposes_only_executor_interface(monkeypatch):
    orch = _make_orchestrator_stub()
    runtime = ServiceStreamingRuntime(orch)

    class DummyExecutor:
        pass

    service_module = importlib.import_module("victor.agent.services.chat_stream_executor")
    monkeypatch.setattr(
        service_module,
        "create_streaming_chat_executor",
        lambda owner, **kwargs: DummyExecutor(),
    )

    assert not hasattr(runtime, "get_pipeline")
    assert isinstance(runtime.get_executor(), DummyExecutor)


def test_service_streaming_runtime_executor_initialization_is_warning_free(monkeypatch):
    orch = _make_orchestrator_stub()
    runtime = ServiceStreamingRuntime(orch)

    class DummyExecutor:
        pass

    service_module = importlib.import_module("victor.agent.services.chat_stream_executor")
    monkeypatch.setattr(
        service_module,
        "create_streaming_chat_executor",
        lambda owner, **kwargs: DummyExecutor(),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        executor = runtime.get_executor()

    assert isinstance(executor, DummyExecutor)
    assert caught == []


@pytest.mark.asyncio
async def test_service_streaming_runtime_supports_protocol_adapter_host(monkeypatch):
    orch = _make_orchestrator_stub()
    adapter = OrchestratorProtocolAdapter(orch)
    runtime = ServiceStreamingRuntime(adapter)
    chunk = StreamChunk(content="service", is_final=True)

    class DummyExecutor:
        async def run_unified(self, user_message: str, **kwargs):
            assert user_message == "hello"
            assert kwargs == {"mode": "test"}
            yield chunk

    def fake_factory(owner, **kwargs):
        assert owner is runtime
        return DummyExecutor()

    service_module = importlib.import_module("victor.agent.services.chat_stream_executor")
    monkeypatch.setattr(service_module, "create_streaming_chat_executor", fake_factory)

    ctx = SimpleNamespace(
        cumulative_usage={
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
        runtime_override_snapshot=None,
    )
    orch._current_stream_context = ctx

    chunks = [item async for item in runtime.stream_chat("hello", mode="test")]

    assert chunks == [chunk]
    assert orch._cumulative_token_usage["prompt_tokens"] == 2
    assert orch._cumulative_token_usage["completion_tokens"] == 3
    assert orch._cumulative_token_usage["total_tokens"] == 5
    assert orch._current_stream_context is None


@pytest.mark.asyncio
async def test_service_streaming_runtime_stream_chat_uses_executor(monkeypatch):
    orch = _make_orchestrator_stub()
    runtime = ServiceStreamingRuntime(orch)
    chunk = StreamChunk(content="service", is_final=True)

    class DummyExecutor:
        def __init__(self):
            self.calls = []

        async def run_unified(self, user_message: str, **kwargs):
            self.calls.append((user_message, kwargs))
            yield chunk

    executor = DummyExecutor()

    def fake_factory(owner, **kwargs):
        return executor

    service_module = importlib.import_module("victor.agent.services.chat_stream_executor")
    monkeypatch.setattr(service_module, "create_streaming_chat_executor", fake_factory)

    chunks = [item async for item in runtime.stream_chat("hello", mode="test")]

    assert chunks == [chunk]
    assert executor.calls == [("hello", {"mode": "test"})]


@pytest.mark.asyncio
async def test_service_streaming_runtime_prefers_chat_service_for_context_limit_handling():
    orch = _make_orchestrator_stub()
    orch._chat_service = SimpleNamespace(
        handle_context_and_iteration_limits=AsyncMock(
            return_value=(True, StreamChunk(content="service-stop", is_final=True))
        )
    )
    orch._handle_context_and_iteration_limits_runtime = AsyncMock(
        side_effect=AssertionError("legacy runtime helper should not be used")
    )
    runtime = ServiceStreamingRuntime(OrchestratorProtocolAdapter(orch))

    handled, chunk = await runtime._handle_context_and_iteration_limits(
        "hello",
        5,
        1000,
        1,
        0.8,
    )

    assert handled is True
    assert chunk is not None
    assert chunk.content == "service-stop"
    orch._chat_service.handle_context_and_iteration_limits.assert_awaited_once_with(
        "hello",
        5,
        1000,
        1,
        0.8,
    )


@pytest.mark.asyncio
async def test_service_streaming_runtime_context_limit_uses_canonical_helper_when_service_absent():
    orch = _make_orchestrator_stub()
    orch._chat_service = None
    expected_chunk = StreamChunk(content="legacy-stop", is_final=True)
    context_limit_helper = MagicMock()
    context_limit_helper.handle_limits = AsyncMock(return_value=(True, expected_chunk))
    orch._get_context_limit_runtime = MagicMock(return_value=context_limit_helper)
    orch._handle_context_and_iteration_limits_runtime = AsyncMock(
        side_effect=AssertionError("legacy runtime wrapper should not be used")
    )
    runtime = ServiceStreamingRuntime(OrchestratorProtocolAdapter(orch))

    handled, chunk = await runtime._handle_context_and_iteration_limits(
        "hello",
        5,
        1000,
        1,
        0.8,
    )

    assert handled is True
    assert chunk is expected_chunk
    orch._get_context_limit_runtime.assert_called_once_with()
    context_limit_helper.handle_limits.assert_awaited_once_with(
        "hello",
        5,
        1000,
        1,
        0.8,
    )


@pytest.mark.asyncio
async def test_service_streaming_runtime_context_limit_does_not_use_name_resolver_hook():
    orch = _make_orchestrator_stub()
    orch._chat_service = None
    expected_chunk = StreamChunk(content="legacy-stop", is_final=True)
    context_limit_helper = MagicMock()
    context_limit_helper.handle_limits = AsyncMock(return_value=(True, expected_chunk))
    orch._get_context_limit_runtime = MagicMock(return_value=context_limit_helper)
    orch._handle_context_and_iteration_limits_runtime = AsyncMock(
        side_effect=AssertionError("legacy runtime wrapper should not be used")
    )
    runtime = ServiceStreamingRuntime(OrchestratorProtocolAdapter(orch))
    runtime._get_orchestrator_runtime_helper = MagicMock(
        side_effect=AssertionError("name-based runtime helper resolver should not be used")
    )

    handled, chunk = await runtime._handle_context_and_iteration_limits(
        "hello",
        5,
        1000,
        1,
        0.8,
    )

    assert handled is True
    assert chunk is expected_chunk
    orch._get_context_limit_runtime.assert_called_once_with()
    context_limit_helper.handle_limits.assert_awaited_once_with(
        "hello",
        5,
        1000,
        1,
        0.8,
    )


@pytest.mark.asyncio
async def test_service_streaming_runtime_context_limit_bypasses_adapter_wrapper_method():
    orch = _make_orchestrator_stub()
    orch._chat_service = None
    expected_chunk = StreamChunk(content="service-stop", is_final=True)
    context_limit_helper = MagicMock()
    context_limit_helper.handle_limits = AsyncMock(return_value=(True, expected_chunk))
    orch._get_context_limit_runtime = MagicMock(return_value=context_limit_helper)
    adapter = OrchestratorProtocolAdapter(orch)
    adapter._handle_context_and_iteration_limits_runtime = AsyncMock(
        side_effect=AssertionError("adapter wrapper method should not be used")
    )
    runtime = ServiceStreamingRuntime(adapter)

    handled, chunk = await runtime._handle_context_and_iteration_limits(
        "hello",
        5,
        1000,
        1,
        0.8,
    )

    assert handled is True
    assert chunk is expected_chunk
    orch._get_context_limit_runtime.assert_called_once_with()
    context_limit_helper.handle_limits.assert_awaited_once_with(
        "hello",
        5,
        1000,
        1,
        0.8,
    )


@pytest.mark.asyncio
async def test_service_streaming_runtime_context_limit_does_not_fall_back_to_host_wrapper():
    orch = _make_orchestrator_stub()
    orch._chat_service = None
    orch._get_context_limit_runtime = None
    orch._handle_context_and_iteration_limits_runtime = AsyncMock(
        side_effect=AssertionError("host wrapper should not be used")
    )
    runtime = ServiceStreamingRuntime(orch)

    handled, chunk = await runtime._handle_context_and_iteration_limits(
        "hello",
        5,
        1000,
        1,
        0.8,
    )

    assert handled is False
    assert chunk is None


@pytest.mark.asyncio
async def test_service_streaming_runtime_create_stream_context_uses_blocked_threshold_setting():
    orch = _make_orchestrator_stub()
    orch.settings = SimpleNamespace(recovery_blocked_consecutive_threshold=7)
    orch._classify_task_keywords.return_value = {}
    orch._tool_planner = SimpleNamespace(infer_goals_from_message=lambda _: [])
    orch.tool_budget = 200
    orch.tool_calls_used = 0
    orch._task_completion_detector = None

    runtime = ServiceStreamingRuntime(orch)
    runtime._prepare_stream = AsyncMock(
        return_value=(
            SimpleNamespace(),
            0.0,
            0.0,
            {},
            30,
            10,
            0,
            False,
            SimpleNamespace(value="default"),
            None,
            None,
        )
    )

    ctx = await runtime._create_stream_context("hello")

    assert ctx.max_blocked_before_force == 7


@pytest.mark.asyncio
async def test_service_streaming_runtime_create_stream_context_applies_topology_overrides(
    monkeypatch,
):
    orch = _make_orchestrator_stub()
    orch.settings = SimpleNamespace(
        recovery_blocked_consecutive_threshold=7,
        enable_topology_routing=True,
    )
    orch._classify_task_keywords.return_value = {
        "coarse_task_type": "analysis",
        "is_analysis_task": True,
    }
    orch._tool_planner = SimpleNamespace(infer_goals_from_message=lambda _: ["inspect"])
    orch.tool_budget = 200
    orch.tool_calls_used = 0
    orch._task_completion_detector = None
    orch.messages = []
    orch.model = "test-model"
    orch.provider = SimpleNamespace()
    orch.task_coordinator = SimpleNamespace(tool_budget=200)
    orch._runtime_intelligence = SimpleNamespace(
        get_structured_routing_policy=MagicMock(
            return_value=StructuredRoutingPolicy(
                scope_context={"task_type": "design", "provider_hint": "smart-router"},
                topology_hints={
                    "learned_topology_action": "team_plan",
                    "learned_provider_hint": "anthropic",
                    "learned_formation_hint": "parallel",
                    "learned_topology_support": 0.75,
                },
            )
        )
    )
    orch._tool_service = SimpleNamespace(
        get_tool_budget=lambda: 200,
        set_tool_budget=MagicMock(),
    )
    orch._tool_pipeline = SimpleNamespace(config=SimpleNamespace(tool_budget=200))

    runtime = ServiceStreamingRuntime(orch)
    runtime._prepare_stream = AsyncMock(
        return_value=(
            SimpleNamespace(),
            0.0,
            0.0,
            {},
            30,
            10,
            0,
            False,
            SimpleNamespace(value="analysis"),
            SimpleNamespace(complexity=TaskComplexity.COMPLEX, task_type="design"),
            6,
        )
    )
    runtime._get_stream_topology_provider_hints = AsyncMock(
        return_value={
            "provider_hint": "smart-router",
            "fallback_chain": ["smart-router"],
        }
    )
    runtime._paradigm_router = SimpleNamespace(
        build_topology_input=MagicMock(
            return_value=TopologyDecisionInput(
                query="hello",
                task_type="design",
                task_complexity="high",
                tool_budget=6,
                iteration_budget=3,
                available_team_formations=["parallel", "hierarchical"],
                provider_candidates=["smart-router"],
            )
        )
    )
    runtime._topology_selector = SimpleNamespace(
        select=MagicMock(
            return_value=TopologyDecision(
                action=TopologyAction.TEAM_PLAN,
                topology=TopologyKind.TEAM,
                confidence=0.8,
                rationale="Deep task favors a team plan.",
                grounding_requirements=TopologyGroundingRequirements(
                    provider="smart-router",
                    formation="parallel",
                    max_workers=3,
                    tool_budget=4,
                    iteration_budget=2,
                ),
                provider="smart-router",
                formation="parallel",
            )
        )
    )
    runtime._topology_grounder = SimpleNamespace(
        ground=MagicMock(
            return_value=GroundedTopologyPlan(
                action=TopologyAction.TEAM_PLAN,
                topology=TopologyKind.TEAM,
                execution_mode="team_execution",
                provider="smart-router",
                formation="parallel",
                max_workers=3,
                tool_budget=4,
                iteration_budget=2,
            )
        )
    )

    helpers_module = importlib.import_module("victor.agent.services.chat_stream_helpers")
    emit_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(helpers_module, "emit_topology_telemetry_event", emit_mock)

    with patch(
        "victor.framework.topology_runtime.resolve_configured_team",
        return_value=ResolvedTeamExecutionPlan(
            team_name="feature_team",
            display_name="Feature Team",
            formation=TeamFormation.PARALLEL,
            member_count=2,
            total_tool_budget=4,
            max_iterations=20,
            max_workers=3,
        ),
    ):
        ctx = await runtime._create_stream_context("hello")

    assert ctx.topology_plan["execution_mode"] == "team_execution"
    assert ctx.topology_plan["team_name"] == "feature_team"
    assert ctx.topology_preparation["action"] == "team_plan"
    assert ctx.topology_preparation["prepared"] is True
    assert ctx.provider_kwargs["provider_hint"] == "smart-router"
    assert ctx.runtime_context_overrides["formation_hint"] == "parallel"
    assert ctx.runtime_context_overrides["team_name"] == "feature_team"
    assert ctx.tool_budget == 4
    assert ctx.max_total_iterations == 2
    assert orch.tool_budget == 4
    assert orch._runtime_tool_context_overrides["max_workers"] == 3
    assert len(ctx.topology_events) == 1
    assert ctx.structured_routing_policy is not None
    topology_context = runtime._paradigm_router.build_topology_input.call_args.kwargs["context"]
    assert topology_context["learned_topology_action"] == "team_plan"
    assert topology_context["learned_provider_hint"] == "anthropic"
    learned_scope_context = (
        orch._runtime_intelligence.get_structured_routing_policy.call_args.kwargs["scope_context"]
    )
    assert orch._runtime_intelligence.get_structured_routing_policy.call_args.kwargs["query"] == (
        "hello"
    )
    assert learned_scope_context["task_type"] == "design"
    assert learned_scope_context["provider_hint"] == "smart-router"
    emit_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_streaming_runtime_stream_chat_restores_runtime_overrides(
    monkeypatch,
):
    class FakeToolService:
        def __init__(self, budget: int, used: int = 0):
            self.budget = budget
            self.used = used
            self.history = []

        def get_tool_budget(self) -> int:
            return max(0, self.budget - self.used)

        def get_remaining_budget(self) -> int:
            return self.get_tool_budget()

        def set_tool_budget(self, budget: int) -> None:
            self.history.append(budget)
            self.budget = budget

    orch = _make_orchestrator_stub()
    orch.tool_budget = 9
    orch.task_coordinator = SimpleNamespace(tool_budget=9)
    orch._tool_service = FakeToolService(budget=9, used=3)
    orch._tool_pipeline = SimpleNamespace(config=SimpleNamespace(tool_budget=9))
    orch._conversation_controller.messages = [SimpleNamespace(content="abc")]
    orch._runtime_intelligence = SimpleNamespace(record_topology_outcome=MagicMock())
    orch.has_capability.side_effect = lambda name: name == "current_stream_context"

    runtime = ServiceStreamingRuntime(orch)
    ctx = SimpleNamespace(
        cumulative_usage={
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
        unified_task_type=TrackerTaskType.ANALYZE,
        task_classification=SimpleNamespace(complexity=TaskComplexity.ANALYSIS),
        complexity_tool_budget=9,
        coarse_task_type="analysis",
        is_analysis_task=True,
        is_action_task=False,
        needs_execution=False,
        last_quality_score=0.81,
        topology_events=[
            {
                "action": "team_plan",
                "topology": "team",
                "execution_mode": "team_execution",
                "provider": "smart-router",
                "formation": "parallel",
                "confidence": 0.82,
                "outcome": {"status": "planned"},
            }
        ],
        total_iterations=2,
        tool_calls_used=4,
        unique_resources={
            "victor/framework/graph.py",
            "victor/framework/graph_cache.py",
        },
        executed_tool_names={"read", "edit"},
        provider_status_events=[
            {
                "kind": "tool_history_repaired",
                "stripped_assistant_tool_calls": 1,
                "removed_orphaned_tool_responses": 1,
            }
        ],
        degradation_events=[],
        recovery_events=[],
        force_completion=False,
        has_substantial_content=lambda: True,
        last_compaction_policy_reason="high_utilization_large_tool_output",
        runtime_override_snapshot=None,
    )
    ctx.runtime_override_snapshot = runtime._apply_stream_runtime_overrides(
        {"tool_budget": 4, "provider_hint": "smart-router"}
    )
    orch.get_capability_value.side_effect = lambda name: (
        ctx if name == "current_stream_context" else None
    )
    orch._current_stream_context = ctx

    class DummyExecutor:
        async def run_unified(self, user_message: str, **kwargs):
            yield StreamChunk(content="service", is_final=True)

    runtime._streaming_executor = DummyExecutor()

    chunks = [item async for item in runtime.stream_chat("hello")]

    assert chunks == [StreamChunk(content="service", is_final=True)]
    assert orch.tool_budget == 9
    assert orch.task_coordinator.tool_budget == 9
    assert orch._tool_service.budget == 9
    assert orch._tool_service.history == [4, 9]
    assert "_runtime_tool_context_overrides" not in orch.__dict__
    orch._runtime_intelligence.record_topology_outcome.assert_called_once()
    feedback_payload = orch._runtime_intelligence.record_topology_outcome.call_args.args[0]
    assert feedback_payload["status"] == "completed"
    assert feedback_payload["completion_score"] == pytest.approx(0.81)
    assert ctx.topology_events[-1]["outcome"]["runtime"] == "streaming"
    assert ctx.runtime_override_snapshot is None
    assert orch._current_stream_context is None
    assert orch._last_stream_task_context["unified_task_type"] == TrackerTaskType.ANALYZE
    assert orch._last_stream_task_context["coarse_task_type"] == "analysis"
    assert orch._last_stream_task_context["degraded_resume_state"] is True
    assert orch._last_stream_task_context["resume_recent_resources"] == [
        "victor/framework/graph.py",
        "victor/framework/graph_cache.py",
    ]
    assert (
        orch._last_stream_task_context["last_compaction_policy_reason"]
        == "high_utilization_large_tool_output"
    )
    assert "tool-history repair" in orch._last_stream_task_context["resume_summary"]


@pytest.mark.asyncio
async def test_service_streaming_runtime_create_stream_context_applies_pending_continuation_shape():
    orch = _make_orchestrator_stub()
    orch.settings = SimpleNamespace(recovery_blocked_consecutive_threshold=5)
    orch._classify_task_keywords.return_value = {}
    orch._tool_planner = SimpleNamespace(infer_goals_from_message=lambda _: [])
    orch.tool_budget = 42
    orch.tool_calls_used = 0
    orch._task_completion_detector = None
    orch._pending_continuation_task_context = {
        "carry_forward_task_shape": True,
        "coarse_task_type": "analysis",
        "is_analysis_task": True,
        "is_action_task": False,
        "needs_execution": False,
        "degraded_resume_state": True,
        "resume_summary": "2 tool calls used; previous provider request required tool-history repair",
        "resume_recent_resources": ["victor/framework/graph.py"],
        "resume_recent_tools": ["read", "edit"],
        "task_intent": "Investigate the graph runtime path",
        "plan_steps": ["Read graph runtime", "Check compaction flow"],
        "intent_log": [{"kind": "tool_intent", "summary": "planned read"}],
        "last_compaction_policy_reason": "tool_output_exceeds_remaining_budget",
    }

    runtime = ServiceStreamingRuntime(orch)
    runtime._prepare_stream = AsyncMock(
        return_value=(
            SimpleNamespace(),
            0.0,
            0.0,
            {},
            30,
            10,
            0,
            False,
            TrackerTaskType.GENERAL,
            SimpleNamespace(complexity=TaskComplexity.ANALYSIS),
            42,
        )
    )

    ctx = await runtime._create_stream_context("continue")

    assert ctx.is_analysis_task is True
    assert ctx.coarse_task_type == "analysis"
    assert ctx.degraded_resume_state is True
    assert "tool-history repair" in ctx.resume_summary
    assert ctx.resume_recent_resources == ["victor/framework/graph.py"]
    assert ctx.resume_recent_tools == ["read", "edit"]
    assert ctx.task_intent == "Investigate the graph runtime path"
    assert ctx.plan_steps == ["Read graph runtime", "Check compaction flow"]
    assert ctx.intent_log[-1]["summary"] == "planned read"
    assert ctx.last_compaction_policy_reason == "tool_output_exceeds_remaining_budget"
    assert orch._pending_continuation_task_context is None


@pytest.mark.asyncio
async def test_service_streaming_runtime_promotes_write_followup_to_action_shape():
    orch = _make_orchestrator_stub()
    orch.settings = SimpleNamespace(recovery_blocked_consecutive_threshold=5)
    orch._classify_task_keywords.return_value = {
        "coarse_task_type": "default",
        "is_action_task": False,
        "needs_execution": False,
    }
    orch._tool_planner = SimpleNamespace(infer_goals_from_message=lambda _: [])
    orch.tool_budget = 60
    orch.tool_calls_used = 0
    orch._task_completion_detector = None

    runtime = ServiceStreamingRuntime(orch)
    runtime._prepare_stream = AsyncMock(
        return_value=(
            SimpleNamespace(),
            0.0,
            0.0,
            {},
            30,
            10,
            0,
            False,
            TrackerTaskType.EDIT,
            SimpleNamespace(complexity=TaskComplexity.ACTION),
            60,
        )
    )

    ctx = await runtime._create_stream_context(
        "Are you able to address them. if yes please address them comprehensively."
    )

    assert ctx.is_action_task is True
    assert ctx.coarse_task_type == "action"


@pytest.mark.asyncio
async def test_service_streaming_runtime_honors_action_complexity_without_action_task_type():
    orch = _make_orchestrator_stub()
    orch.settings = SimpleNamespace(recovery_blocked_consecutive_threshold=5)
    orch._classify_task_keywords.return_value = {
        "coarse_task_type": "design",
        "is_action_task": False,
        "needs_execution": False,
    }
    orch._tool_planner = SimpleNamespace(infer_goals_from_message=lambda _: [])
    orch.tool_budget = 60
    orch.tool_calls_used = 0
    orch._task_completion_detector = None

    runtime = ServiceStreamingRuntime(orch)
    runtime._prepare_stream = AsyncMock(
        return_value=(
            SimpleNamespace(),
            0.0,
            0.0,
            {},
            30,
            10,
            0,
            False,
            TrackerTaskType.GENERAL,
            SimpleNamespace(complexity=TaskComplexity.ACTION),
            60,
        )
    )

    ctx = await runtime._create_stream_context(
        "continue to address the remaining findings and suggestions"
    )

    assert ctx.is_action_task is True
    assert ctx.needs_execution is True
    assert ctx.coarse_task_type == "design"


@pytest.mark.asyncio
async def test_service_streaming_runtime_reuses_resume_context_for_remediation_continuation_payload():
    orch = _make_orchestrator_stub()
    orch.settings = SimpleNamespace(recovery_blocked_consecutive_threshold=5)
    orch._classify_task_keywords.return_value = {
        "coarse_task_type": "default",
        "is_action_task": False,
        "needs_execution": False,
    }
    orch._tool_planner = SimpleNamespace(infer_goals_from_message=lambda _: [])
    orch.tool_budget = 60
    orch.tool_calls_used = 0
    orch._task_completion_detector = None
    orch._pending_continuation_task_context = {
        "carry_forward_task_shape": False,
        "carry_forward_resume_context": True,
        "degraded_resume_state": True,
        "resume_summary": "previous turn ended before completion",
        "resume_recent_resources": ["victor/framework/graph.py"],
        "resume_recent_tools": ["read", "edit"],
    }

    runtime = ServiceStreamingRuntime(orch)
    runtime._prepare_stream = AsyncMock(
        return_value=(
            SimpleNamespace(),
            0.0,
            0.0,
            {},
            30,
            10,
            0,
            False,
            TrackerTaskType.DESIGN,
            SimpleNamespace(complexity=TaskComplexity.ACTION),
            60,
        )
    )

    ctx = await runtime._create_stream_context(
        "continue to address the remaining findings and suggestions"
    )

    assert ctx.is_action_task is True
    assert ctx.needs_execution is True
    assert ctx.coarse_task_type == "action"
    assert ctx.degraded_resume_state is True
    assert ctx.resume_summary == "previous turn ended before completion"
    assert ctx.resume_recent_resources == ["victor/framework/graph.py"]
    assert ctx.resume_recent_tools == ["read", "edit"]
    assert orch._pending_continuation_task_context is None


@pytest.mark.asyncio
async def test_service_streaming_runtime_preserves_empty_provider_response_for_recovery():
    orch = _make_orchestrator_stub()
    orch.model = "glm-5.1"
    orch.temperature = 0.1
    orch.max_tokens = 512
    orch.settings = SimpleNamespace(
        stream_provider_wait_heartbeat_seconds=15.0,
        stream_provider_stall_timeout_seconds=30.0,
    )
    orch.get_assembled_messages = MagicMock(return_value=[])
    orch.sanitizer = SimpleNamespace(is_garbage_content=lambda _content: False)

    async def _empty_stream(**_kwargs):
        if False:
            yield None

    orch.provider = SimpleNamespace(stream=_empty_stream)

    runtime = ServiceStreamingRuntime(orch)
    ctx = StreamingChatContext(user_message="address the findings")

    full_content, tool_calls, total_tokens, garbage_detected = (
        await runtime._stream_provider_response(
            tools=None,
            provider_kwargs={},
            stream_ctx=ctx,
        )
    )

    assert full_content == ""
    assert tool_calls is None
    assert total_tokens == 0
    assert garbage_detected is False
    assert [event["kind"] for event in ctx.provider_status_events] == [
        "completion_detected",
        "empty_stream_completed",
    ]


@pytest.mark.asyncio
async def test_service_streaming_runtime_does_not_charge_event_loop_stall_to_provider(
    monkeypatch,
):
    helpers_module = importlib.import_module("victor.agent.services.chat_stream_helpers")

    orch = _make_orchestrator_stub()
    orch.model = "glm-5.1"
    orch.temperature = 0.1
    orch.max_tokens = 512
    orch.settings = SimpleNamespace(
        stream_provider_wait_heartbeat_seconds=0.01,
        stream_provider_stall_timeout_seconds=0.02,
        stream_provider_loop_stall_grace_seconds=0.0,
    )
    orch.get_assembled_messages = MagicMock(return_value=[])
    orch.sanitizer = SimpleNamespace(is_garbage_content=lambda _content: False)

    async def _stream(**_kwargs):
        yield StreamChunk(content="done")

    orch.provider = SimpleNamespace(stream=_stream)

    original_wait_for = helpers_module.asyncio.wait_for
    calls = {"count": 0}

    async def _wait_for_once_stalled(awaitable, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise helpers_module.asyncio.TimeoutError()
        return await original_wait_for(awaitable, timeout=timeout)

    monotonic_values = iter([0.0, 0.0, 590.0, 590.0, 590.0, 590.0])

    monkeypatch.setattr(helpers_module.asyncio, "wait_for", _wait_for_once_stalled)
    monkeypatch.setattr(
        helpers_module.time,
        "monotonic",
        lambda: next(monotonic_values, 590.0),
    )

    runtime = ServiceStreamingRuntime(orch)
    ctx = StreamingChatContext(user_message="address the findings")

    full_content, tool_calls, total_tokens, garbage_detected = (
        await runtime._stream_provider_response(
            tools=None,
            provider_kwargs={},
            stream_ctx=ctx,
        )
    )

    assert full_content == "done"
    assert tool_calls is None
    assert total_tokens == 1
    assert garbage_detected is False
    assert "local_runtime_stall" in [event["kind"] for event in ctx.provider_status_events]
    # Restore global time.monotonic (and asyncio.wait_for) BEFORE the event loop
    # teardown. monkeypatch.setattr(helpers_module.time, "monotonic", ...) patches
    # the GLOBAL time module (helpers_module.time IS time), and the skewed value
    # (590.0) propagates into asyncio's loop.time() -> the loop's timer scheduling
    # breaks -> selector.select stalls indefinitely at teardown (no per-test timeout
    # on CI -> the whole shard hangs at this test). monkeypatch.undo() here restores
    # the real monotonic before pytest-asyncio finalizes the loop.
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_service_streaming_runtime_stream_chat_normalizes_recovery_events():
    orch = _make_orchestrator_stub()
    orch._runtime_intelligence = SimpleNamespace(record_topology_outcome=MagicMock())
    orch.has_capability.side_effect = lambda name: name == "current_stream_context"

    runtime = ServiceStreamingRuntime(orch)
    ctx = SimpleNamespace(
        cumulative_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        last_quality_score=0.78,
        topology_events=[],
        degradation_events=[
            {
                "source": "streaming_confidence",
                "kind": "confidence_early_stop",
                "post_degraded": False,
                "recovered": False,
            }
        ],
        recovery_events=[
            {
                "action": "retry",
                "failure_type": "PROVIDER_ERROR",
                "strategy_name": "retry_with_hint",
                "reason": "empty response loop",
                "confidence": 0.61,
                "iteration": 2,
            }
        ],
        total_iterations=2,
        tool_calls_used=0,
        force_completion=False,
        has_substantial_content=lambda: True,
        runtime_override_snapshot=None,
    )
    orch.get_capability_value.side_effect = lambda name: (
        ctx if name == "current_stream_context" else None
    )
    orch._current_stream_context = ctx

    class DummyExecutor:
        async def run_unified(self, user_message: str, **kwargs):
            yield StreamChunk(content="service", is_final=True)

    runtime._streaming_executor = DummyExecutor()

    chunks = [item async for item in runtime.stream_chat("hello")]

    assert chunks == [StreamChunk(content="service", is_final=True)]
    assert len(ctx.degradation_events) == 2
    recovery_event = ctx.degradation_events[-1]
    assert recovery_event["source"] == "streaming_recovery"
    assert recovery_event["failure_type"] == "PROVIDER_ERROR"
    assert recovery_event["recovered"] is True
    assert recovery_event["post_degraded"] is False
    assert recovery_event["degradation_reasons"] == [
        "retry",
        "retry_with_hint",
        "empty response loop",
    ]
