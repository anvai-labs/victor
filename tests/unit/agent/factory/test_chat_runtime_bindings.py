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
    pass


def _owner() -> _Owner:
    owner = _Owner()
    owner._session_accessor = SessionStateAccessor(SessionStateManager())
    owner._chunk_generator = None
    owner.sanitizer = None
    owner._recovery_service = None
    owner._recovery_coordinator = None
    owner._tool_planner = None
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
