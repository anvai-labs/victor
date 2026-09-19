"""Focused coverage for the chat-runtime composition boundary."""

from types import SimpleNamespace

import pytest

from victor.agent.factory.chat_runtime_bindings import bind_chat_runtime_services
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
