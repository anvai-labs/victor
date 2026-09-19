"""Focused coverage for the chat-runtime composition boundary."""

import gc
from types import SimpleNamespace
import weakref
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.factory.chat_runtime_bindings import (
    bind_chat_runtime_services,
    bind_chat_turn_runtime,
)
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import SessionStateManager
from victor.agent.unified_task_tracker import TrackerTaskType


class _Owner:
    def on_tool_start(self, *args, **kwargs):
        pass

    def on_tool_complete(self, *args, **kwargs):
        pass


def _owner() -> _Owner:
    owner = _Owner()
    owner._session_accessor = SessionStateAccessor(SessionStateManager())
    owner._chunk_generator = None
    owner.sanitizer = None
    owner._recovery_service = None
    owner._recovery_coordinator = None
    owner._tool_planner = None
    owner._message_policy_gate = None
    owner._task_completion_detector = None
    owner._conversation_controller = None
    owner._tool_service = None
    owner.tool_adapter = None
    owner._tool_pipeline = None
    owner._metrics_collector = MagicMock()
    owner._metrics_coordinator = MagicMock()
    owner._session_state = MagicMock()
    owner.reminder_manager = MagicMock()
    owner.unified_tracker = MagicMock()
    owner.unified_tracker.config = {"max_total_iterations": 50}
    owner.unified_tracker.progress = SimpleNamespace(
        has_prompt_requirements=False,
        tool_budget=10,
    )
    owner.unified_tracker._task_config = SimpleNamespace(max_exploration_iterations=8)
    owner.unified_tracker.max_exploration_iterations = 8
    owner._context_manager = None
    owner._context_lifecycle_service = None
    owner._context_service = None
    owner._context_compactor = None
    owner._runtime_intelligence = None
    owner._perception_integration = None
    owner._optimization_injector = None
    owner.runtime_intelligence_integration = None
    owner.conversation_state = {}
    owner.settings = SimpleNamespace(context_compaction_strategy="tiered")
    owner.tool_calls_used = 0
    return owner


def test_binding_reuses_each_session_owners_stable_turn_lock() -> None:
    first = _owner()
    second = _owner()

    first_view = bind_chat_runtime_services(first)
    rebound_first_view = bind_chat_runtime_services(first)
    second_view = bind_chat_runtime_services(second)

    assert first_view.stream_turn_lock is first._session_accessor.stream_turn_lock
    assert rebound_first_view.stream_turn_lock is first_view.stream_turn_lock
    assert second_view.stream_turn_lock is not first_view.stream_turn_lock


def test_binding_coordinates_stream_lifecycle_without_retaining_owner() -> None:
    owner = _owner()
    owner._cancel_event = None
    owner._is_streaming = False
    owner._current_stream_context = None
    view = bind_chat_runtime_services(owner)

    assert view.stream_lifecycle is not None
    view.stream_lifecycle.begin()
    assert owner._is_streaming is True
    assert owner._cancel_event is not None
    assert view.stream_lifecycle.is_cancelled() is False

    AgentOrchestrator.request_cancellation(owner)
    assert view.stream_lifecycle.is_cancelled() is True

    context = object()
    view.stream_lifecycle.bind_context(context)
    assert view.stream_lifecycle.current_context() is context

    view.stream_lifecycle.finish()
    assert owner._is_streaming is False
    assert owner._cancel_event is None
    view.stream_lifecycle.clear_context(context)
    assert owner._current_stream_context is None


def test_binding_stream_context_prefers_capability_and_clears_only_bound_context() -> None:
    owner = _owner()
    field_context = object()
    capability_context = object()
    replacement_context = object()
    owner._current_stream_context = field_context
    owner.get_capability_value = MagicMock(return_value=capability_context)
    lifecycle = bind_chat_runtime_services(owner).stream_lifecycle

    assert lifecycle.current_context() is capability_context

    owner._current_stream_context = replacement_context
    lifecycle.clear_context(field_context)
    assert owner._current_stream_context is replacement_context


def test_binding_stream_context_uses_public_then_private_instance_state() -> None:
    owner = _owner()
    public_context = object()
    private_context = object()
    owner.current_stream_context = public_context
    owner._current_stream_context = private_context
    lifecycle = bind_chat_runtime_services(owner).stream_lifecycle

    assert lifecycle.current_context() is public_context
    lifecycle.clear_context(public_context)
    assert owner.current_stream_context is None
    assert lifecycle.current_context() is private_context


def test_binding_stream_context_falls_back_after_capability_failure() -> None:
    owner = _owner()
    context = object()
    owner._current_stream_context = context
    owner.get_capability_value = MagicMock(side_effect=RuntimeError("registry unavailable"))

    assert bind_chat_runtime_services(owner).stream_lifecycle.current_context() is context


def test_binding_routes_stream_metrics_through_enumerated_components() -> None:
    owner = _owner()
    stream_metrics = object()
    finalized = object()
    owner._metrics_collector.init_stream_metrics.return_value = stream_metrics
    owner._metrics_coordinator.finalize_stream_metrics.return_value = finalized
    view = bind_chat_runtime_services(owner)

    assert view.metrics.begin() is stream_metrics
    view.metrics.record_first_token()
    assert (
        view.metrics.finalize(
            {"prompt_tokens": 7},
            provider_diagnostics={"attempts": 2},
        )
        is finalized
    )

    owner._metrics_collector.init_stream_metrics.assert_called_once_with()
    owner._metrics_collector.record_first_token.assert_called_once_with()
    owner._metrics_coordinator.finalize_stream_metrics.assert_called_once_with(
        {"prompt_tokens": 7},
        provider_diagnostics={"attempts": 2},
    )


def test_binding_routes_task_state_without_exposing_tracker_internals() -> None:
    owner = _owner()
    owner.unified_tracker.detect_task_type.return_value = TrackerTaskType.EDIT
    view = bind_chat_runtime_services(owner)

    view.task_state.reset_turn()
    assert view.task_state.max_total_iterations() == 50
    assert view.task_state.detect_task_type("change app.py") == TrackerTaskType.EDIT
    view.task_state.set_continuation_context({"resume": True})
    assert view.task_state.continuation_context() == {"resume": True}
    view.task_state.publish_task_type(TrackerTaskType.EDIT)
    assert view.task_state.apply_prompt_requirements(
        tool_budget=25,
        iteration_budget=12,
    ) == (True, True)
    assert view.task_state.max_exploration_iterations() == 8
    view.task_state.set_tool_budget(30)
    view.task_state.set_continuation_context(None)

    owner._session_state.reset_for_new_turn.assert_called_once_with()
    owner.unified_tracker.reset.assert_called_once_with()
    owner.reminder_manager.reset.assert_called_once_with()
    owner.unified_tracker.set_task_type.assert_called_once_with(TrackerTaskType.EDIT)
    assert owner._current_task_type == "edit"
    assert owner._pending_continuation_task_context is None
    assert owner.unified_tracker.progress.has_prompt_requirements is True
    assert owner.unified_tracker.set_tool_budget.call_args_list == [
        ((25,), {}),
        ((30,), {}),
    ]
    owner.unified_tracker.set_max_iterations.assert_called_once_with(12)


@pytest.mark.asyncio
async def test_binding_starts_background_compaction_through_context_capability() -> None:
    owner = _owner()
    owner._context_manager = SimpleNamespace(start_background_compaction=AsyncMock())
    view = bind_chat_runtime_services(owner)

    await view.context_lifecycle.start_background_compaction()

    owner._context_manager.start_background_compaction.assert_awaited_once_with(
        interval_seconds=15.0
    )


@pytest.mark.asyncio
async def test_binding_context_lifecycle_has_priority_and_blocks_fallbacks() -> None:
    owner = _owner()
    owner.get_messages = MagicMock(return_value=[{"role": "user", "content": "hello"}])
    owner.active_session_id = "session-1"
    owner.agent_id = "root"
    owner.display_name = "Root"
    owner._context_lifecycle_service = SimpleNamespace(
        after_agent_turn=AsyncMock(
            return_value={
                "compacted": True,
                "messages_removed": 2,
                "tokens_freed": 80,
                "strategy": "semantic",
                "summary": "kept the plan",
            }
        )
    )
    owner._context_service = MagicMock()
    owner._context_compactor = MagicMock()
    view = bind_chat_runtime_services(owner)

    event = await view.context_lifecycle.compact_before_iteration("continue")

    assert event is not None
    assert event.messages_removed == 2
    assert event.tokens_freed == 80
    assert event.strategy == "semantic"
    assert event.summary == "kept the plan"
    assert event.policy_reason == "context_lifecycle"
    call = owner._context_lifecycle_service.after_agent_turn.await_args
    assert call.args[0].agent_id == "root"
    assert call.args[0].session_id == "session-1"
    assert call.kwargs["messages"] == [{"role": "user", "content": "hello"}]
    owner._context_service.get_compaction_recommendation.assert_not_called()
    owner._context_compactor.check_and_compact.assert_not_called()


@pytest.mark.asyncio
async def test_binding_context_lifecycle_noop_still_blocks_older_fallbacks() -> None:
    owner = _owner()
    owner._context_lifecycle_service = SimpleNamespace(
        after_agent_turn=AsyncMock(return_value={"compacted": False})
    )
    owner._context_service = MagicMock()
    owner._context_compactor = MagicMock()
    view = bind_chat_runtime_services(owner)

    assert await view.context_lifecycle.compact_before_iteration("continue") is None

    owner._context_service.get_compaction_recommendation.assert_not_called()
    owner._context_compactor.check_and_compact.assert_not_called()


@pytest.mark.asyncio
async def test_binding_context_lifecycle_treats_unavailable_history_as_empty() -> None:
    class BrokenConversation:
        @property
        def messages(self):
            raise RuntimeError("history unavailable")

    owner = _owner()
    owner.get_messages = MagicMock(side_effect=RuntimeError("root history unavailable"))
    owner._conversation_controller = BrokenConversation()
    owner._context_lifecycle_service = SimpleNamespace(
        after_agent_turn=AsyncMock(return_value={"compacted": False})
    )
    view = bind_chat_runtime_services(owner)

    assert await view.context_lifecycle.compact_before_iteration("continue") is None

    owner._context_lifecycle_service.after_agent_turn.assert_awaited_once()
    assert owner._context_lifecycle_service.after_agent_turn.await_args.kwargs["messages"] == []


@pytest.mark.asyncio
async def test_binding_context_service_precedes_legacy_compactor() -> None:
    owner = _owner()
    owner.settings.context_compaction_strategy = "semantic"
    owner._context_service = SimpleNamespace(
        get_compaction_recommendation=MagicMock(return_value={"should_compact": True}),
        compact_context=AsyncMock(return_value=3),
    )
    owner._context_compactor = MagicMock()
    view = bind_chat_runtime_services(owner)

    event = await view.context_lifecycle.compact_before_iteration("continue")

    assert event is not None
    assert event.messages_removed == 3
    assert event.strategy == "semantic"
    assert event.policy_reason == "context_service"
    owner._context_compactor.check_and_compact.assert_not_called()


@pytest.mark.asyncio
async def test_binding_uses_legacy_compactor_only_as_final_fallback() -> None:
    owner = _owner()
    owner._context_compactor = MagicMock()
    owner._context_compactor.check_and_compact.return_value = SimpleNamespace(
        action_taken=True,
        messages_removed=4,
        tokens_freed=120,
    )
    owner.conversation_controller = SimpleNamespace(
        get_compaction_summaries=MagicMock(return_value=["legacy summary"])
    )
    view = bind_chat_runtime_services(owner)

    event = await view.context_lifecycle.compact_before_iteration("continue")

    assert event is not None
    assert event.messages_removed == 4
    assert event.tokens_freed == 120
    assert event.summary == "legacy summary"
    assert event.policy_reason == ""
    owner._context_compactor.check_and_compact.assert_called_once_with(
        current_query="continue",
        force=False,
        tool_call_count=0,
        task_complexity="complex",
    )


def test_binding_reuses_or_creates_one_executor_runtime(monkeypatch) -> None:
    from victor.agent.services.runtime_intelligence import RuntimeIntelligenceService

    owner = _owner()
    owner._perception_integration = object()
    owner._optimization_injector = object()
    runtime = object()
    factory = MagicMock(return_value=runtime)
    monkeypatch.setattr(RuntimeIntelligenceService, "from_orchestrator", factory)
    view = bind_chat_runtime_services(owner)

    assert view.intelligence.executor_runtime() is runtime
    assert view.intelligence.executor_runtime() is runtime
    assert owner._runtime_intelligence is runtime
    factory.assert_called_once_with(
        owner,
        perception_integration=owner._perception_integration,
        optimization_injector=owner._optimization_injector,
    )


def test_binding_executor_runtime_prefers_declared_capabilities(monkeypatch) -> None:
    from victor.agent.services.runtime_intelligence import RuntimeIntelligenceService

    owner = _owner()
    perception = object()
    optimization = object()
    owner.get_capability_value = lambda name: {
        "perception_integration": perception,
        "optimization_injector": optimization,
    }.get(name)
    runtime = object()
    factory = MagicMock(return_value=runtime)
    monkeypatch.setattr(RuntimeIntelligenceService, "from_orchestrator", factory)

    assert bind_chat_runtime_services(owner).intelligence.executor_runtime() is runtime

    factory.assert_called_once_with(
        owner,
        perception_integration=perception,
        optimization_injector=optimization,
    )


def test_binding_executor_runtime_falls_back_when_capability_read_fails(monkeypatch) -> None:
    from victor.agent.services.runtime_intelligence import RuntimeIntelligenceService

    owner = _owner()
    owner._perception_integration = object()
    owner._optimization_injector = object()
    owner.get_capability_value = MagicMock(side_effect=RuntimeError("capability unavailable"))
    runtime = object()
    factory = MagicMock(return_value=runtime)
    monkeypatch.setattr(RuntimeIntelligenceService, "from_orchestrator", factory)

    assert bind_chat_runtime_services(owner).intelligence.executor_runtime() is runtime

    factory.assert_called_once_with(
        owner,
        perception_integration=owner._perception_integration,
        optimization_injector=owner._optimization_injector,
    )


@pytest.mark.asyncio
async def test_binding_prepares_runtime_intelligence_with_live_turn_state() -> None:
    owner = _owner()
    owner.conversation_state = {"turn": 3}
    integration = SimpleNamespace(
        prepare_runtime_intelligence_request=AsyncMock(return_value={"hint": "parallel"})
    )
    owner.runtime_intelligence_integration = integration
    view = bind_chat_runtime_services(owner)

    assert await view.intelligence.prepare_request(task="inspect", task_type="analysis") == {
        "hint": "parallel"
    }
    integration.prepare_runtime_intelligence_request.assert_awaited_once_with(
        task="inspect",
        task_type="analysis",
        conversation_state=owner.conversation_state,
        unified_tracker=owner.unified_tracker,
    )


def test_binding_prefers_structured_runtime_routing_policy() -> None:
    owner = _owner()
    policy = SimpleNamespace(
        to_dict=MagicMock(return_value={"policy": "learned"}),
        selector_context=MagicMock(return_value={"formation": "parallel"}),
    )
    legacy = MagicMock(return_value={"formation": "sequential"})
    owner._runtime_intelligence = SimpleNamespace(
        get_structured_routing_policy=MagicMock(return_value=policy),
        get_topology_routing_context=legacy,
    )
    view = bind_chat_runtime_services(owner)

    result = view.intelligence.routing_context(
        query="inspect",
        scope_context={"task_type": "analysis"},
    )

    assert result.context == {"formation": "parallel"}
    assert result.structured_policy == {"policy": "learned"}
    legacy.assert_not_called()


def test_binding_uses_legacy_runtime_routing_when_structured_api_is_absent() -> None:
    owner = _owner()
    legacy = MagicMock(return_value={"formation": "sequential"})
    owner._runtime_intelligence = SimpleNamespace(get_topology_routing_context=legacy)
    view = bind_chat_runtime_services(owner)

    result = view.intelligence.routing_context(
        query="inspect",
        scope_context={"task_type": "analysis"},
    )

    assert result.context == {"formation": "sequential"}
    assert result.structured_policy is None
    legacy.assert_called_once_with(
        query="inspect",
        scope_context={"task_type": "analysis"},
    )


def test_binding_structured_routing_failure_does_not_run_legacy_policy() -> None:
    owner = _owner()
    legacy = MagicMock(return_value={"formation": "sequential"})
    owner._runtime_intelligence = SimpleNamespace(
        get_structured_routing_policy=MagicMock(side_effect=RuntimeError("policy unavailable")),
        get_topology_routing_context=legacy,
    )
    view = bind_chat_runtime_services(owner)

    result = view.intelligence.routing_context(
        query="inspect",
        scope_context={"task_type": "analysis"},
    )

    assert result.context == {}
    assert result.structured_policy is None
    legacy.assert_not_called()


def test_binding_runtime_intelligence_records_topology_best_effort() -> None:
    owner = _owner()
    record = MagicMock(side_effect=RuntimeError("telemetry unavailable"))
    owner._runtime_intelligence = SimpleNamespace(record_topology_outcome=record)
    view = bind_chat_runtime_services(owner)

    view.intelligence.record_topology_outcome({"success": False})

    record.assert_called_once_with({"success": False})


def test_binding_rejects_owner_that_cannot_honor_non_retention_contract() -> None:
    owner = SimpleNamespace(_session_accessor=SessionStateAccessor(SessionStateManager()))

    with pytest.raises(TypeError, match="must support weak references"):
        bind_chat_runtime_services(owner)


def test_binding_enumerates_stream_execution_collaborators_without_owner_retention() -> None:
    owner = _owner()
    gate = object()
    detector = object()
    conversation = SimpleNamespace(
        messages=[SimpleNamespace(content="abc"), SimpleNamespace(content="de")],
        record_actual_usage=MagicMock(),
        persist_compaction_summary=MagicMock(),
        inject_compaction_context=MagicMock(return_value=True),
    )
    parser = SimpleNamespace(
        parse_and_validate_tool_calls=MagicMock(return_value=([{"x": 1}], "ok"))
    )
    adapter = object()
    pipeline = SimpleNamespace(
        reset=MagicMock(),
        on_tool_start=owner.on_tool_start,
        on_tool_complete=owner.on_tool_complete,
    )
    owner._message_policy_gate = gate
    owner._task_completion_detector = detector
    owner._conversation_controller = conversation
    owner._tool_service = parser
    owner.tool_adapter = adapter
    owner._tool_pipeline = pipeline
    owner._record_runtime_intelligence_outcome = MagicMock()

    view = bind_chat_runtime_services(owner)

    assert view.governance.gate is gate
    assert view.completion.detector is detector
    assert view.conversation.messages() == conversation.messages
    view.conversation.record_actual_usage(7)
    conversation.record_actual_usage.assert_called_once_with(7, 5)
    view.conversation.persist_terminal_summary("done")
    conversation.persist_compaction_summary.assert_called_once_with("done", [])
    conversation.inject_compaction_context.assert_called_once_with()
    assert view.tool_calls.parse_and_validate([{"x": 1}], "raw") == ([{"x": 1}], "ok")
    parser.parse_and_validate_tool_calls.assert_called_once_with([{"x": 1}], "raw", adapter)
    view.tool_calls.reset()
    pipeline.reset.assert_called_once_with()
    view.intelligence.record_outcome(
        success=False,
        quality_score=0.3,
        user_satisfied=False,
        completed=False,
    )
    owner._record_runtime_intelligence_outcome.assert_called_once_with(
        success=False,
        quality_score=0.3,
        user_satisfied=False,
        completed=False,
    )

    owner_ref = weakref.ref(owner)
    # Only the capability view remains. The pipeline itself intentionally has
    # callbacks bound to the owner, mirroring the production ToolPipeline.
    del pipeline
    del owner
    gc.collect()

    assert owner_ref() is None
    with pytest.raises(RuntimeError, match="no longer available"):
        view.tool_calls.reset()
    assert view.conversation.messages() == []
    with pytest.raises(RuntimeError, match="no longer available"):
        view.intelligence.record_outcome(
            success=False,
            quality_score=0.3,
            user_satisfied=False,
            completed=False,
        )


def test_turn_runtime_uses_enumerated_live_state_without_retaining_owner() -> None:
    owner = _owner()
    owner.provider = SimpleNamespace(name="anthropic")
    owner.model = "claude-sonnet"
    owner.vertical = "coding"
    owner._current_stream_context = None
    owner.unified_tracker = None
    owner._current_task_type = "analysis"
    owner._task_type = None
    owner._context_service = None
    owner._metrics_coordinator = MagicMock()
    owner._metrics_coordinator.start_task_report.return_value = "task-1"
    owner._metrics_coordinator.get_last_tool_strategy_event.return_value = None
    owner._metrics_coordinator.finish_task_report.return_value = {"task_id": "task-1"}
    owner._constraint_activator = MagicMock()
    owner._credit_tracking_service = MagicMock()
    skill_runtime = MagicMock()
    owner._get_skill_runtime = MagicMock(return_value=skill_runtime)
    runtime = bind_chat_turn_runtime(owner)
    assert not hasattr(runtime.state, "__dict__")

    constraints = object()
    runtime.enter("inspect app.py", stream=True, constraints=constraints)
    assert runtime.start_task_report("inspect app.py") == "task-1"
    assert runtime.finish_task_report(True, user_message="inspect app.py") == {"task_id": "task-1"}
    runtime.exit("inspect app.py", stream=True, constraints=constraints)

    skill_runtime.apply_skill_for_turn.assert_called_once_with("inspect app.py")
    owner._constraint_activator.activate_constraints.assert_called_once_with(
        constraints=constraints,
        vertical="coding",
    )
    owner._constraint_activator.deactivate_constraints.assert_called_once_with()
    owner._credit_tracking_service.assign_turn_credit_at_boundary.assert_called_once_with()
    assert owner._metrics_coordinator.start_task_report.call_args.kwargs["task_type"] == "analysis"

    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()

    assert owner_ref() is None
    with pytest.raises(RuntimeError, match="no longer available"):
        _ = runtime.state.model


def test_turn_runtime_rejects_nonweakrefable_owner() -> None:
    with pytest.raises(TypeError, match="must support weak references"):
        bind_chat_turn_runtime(SimpleNamespace())
