"""Focused coverage for the chat-runtime composition boundary."""

from types import SimpleNamespace

from victor.agent.factory.chat_runtime_bindings import bind_chat_runtime_services
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import SessionStateManager


def _owner() -> SimpleNamespace:
    return SimpleNamespace(
        _session_accessor=SessionStateAccessor(SessionStateManager()),
        _chunk_generator=None,
        sanitizer=None,
        _recovery_service=None,
        _recovery_coordinator=None,
        _tool_planner=None,
    )


def test_binding_reuses_each_session_owners_stable_turn_lock() -> None:
    first = _owner()
    second = _owner()

    first_view = bind_chat_runtime_services(first)
    rebound_first_view = bind_chat_runtime_services(first)
    second_view = bind_chat_runtime_services(second)

    assert first_view.stream_turn_lock is first._session_accessor.stream_turn_lock
    assert rebound_first_view.stream_turn_lock is first_view.stream_turn_lock
    assert second_view.stream_turn_lock is not first_view.stream_turn_lock
