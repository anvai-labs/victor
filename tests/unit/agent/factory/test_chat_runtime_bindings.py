"""Focused coverage for the chat-runtime composition boundary."""

import gc
from types import SimpleNamespace
import weakref
from unittest.mock import MagicMock

import pytest

from victor.agent.factory.chat_runtime_bindings import (
    bind_chat_runtime_services,
    bind_chat_turn_runtime,
)
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import SessionStateManager


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
    view.feedback.record_outcome(
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
        view.feedback.record_outcome(
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
