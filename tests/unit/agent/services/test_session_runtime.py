from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.services.orchestrator_protocol_adapter import (
    OrchestratorProtocolAdapter,
)
from victor.agent.services.session_runtime import SessionRuntime


def _make_runtime_host(**overrides):
    values = {
        "_session_service": MagicMock(),
        "_lifecycle_manager": MagicMock(),
        "memory_manager": MagicMock(),
        "_checkpoint_manager": MagicMock(),
        "_session_cost_tracker": MagicMock(),
        "_memory_session_id": "session-123",
        "messages": [{"role": "user", "content": "hello"}],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_session_runtime_sync_runtime_state_binds_current_components():
    host = _make_runtime_host()
    runtime = SessionRuntime(OrchestratorProtocolAdapter(host))

    runtime.sync_runtime_state()

    host._session_service.bind_runtime_components.assert_called_once_with(
        lifecycle_manager=host._lifecycle_manager,
        memory_manager=host.memory_manager,
        checkpoint_manager=host._checkpoint_manager,
        cost_tracker=host._session_cost_tracker,
        memory_session_id="session-123",
    )


def test_session_runtime_get_recent_sessions_delegates_after_sync():
    host = _make_runtime_host()
    host._session_service.get_recent_sessions.return_value = [{"session_id": "session-1"}]
    runtime = SessionRuntime(OrchestratorProtocolAdapter(host))

    result = runtime.get_recent_sessions(limit=5)

    assert result == [{"session_id": "session-1"}]
    host._session_service.get_recent_sessions.assert_called_once_with(5)


def test_session_runtime_recover_session_syncs_memory_session_id_on_success():
    host = _make_runtime_host(_memory_session_id=None, active_session_id="old-active")
    host._session_service.recover_session.return_value = True
    runtime = SessionRuntime(OrchestratorProtocolAdapter(host))

    result = runtime.recover_session("session-9")

    assert result is True
    assert host._memory_session_id == "session-9"
    assert host.active_session_id == "session-9"
    host._session_service.recover_session.assert_called_once_with("session-9")


def test_session_runtime_get_memory_context_passes_messages():
    host = _make_runtime_host(messages=[{"role": "assistant", "content": "cached"}])
    host._session_service.get_memory_context.return_value = [
        {"role": "assistant", "content": "cached"}
    ]
    runtime = SessionRuntime(OrchestratorProtocolAdapter(host))

    result = runtime.get_memory_context(max_tokens=200)

    assert result == [{"role": "assistant", "content": "cached"}]
    host._session_service.get_memory_context.assert_called_once_with(
        max_tokens=200,
        messages=[{"role": "assistant", "content": "cached"}],
    )


def test_session_runtime_get_session_stats_delegates_after_sync():
    host = _make_runtime_host()
    host._session_service.get_session_stats.return_value = {"enabled": True}
    runtime = SessionRuntime(OrchestratorProtocolAdapter(host))

    result = runtime.get_session_stats()

    assert result == {"enabled": True}
    host._session_service.get_session_stats.assert_called_once_with()


def test_session_runtime_recovery_preflight_blocks_member_crossing_before_mutation():
    import pytest
    from victor.agent.runtime.context import RuntimeIdentity

    state = SimpleNamespace(active_session_id="root", _memory_session_id=None)
    identity = RuntimeIdentity(state)
    child = identity.resolve().derive_child(
        agent_id="a", display_name="Member", role="researcher", member_id="m"
    )
    identity.bind_member(child)
    host = _make_runtime_host(active_session_id=child.session_id)
    runtime = SessionRuntime(
        OrchestratorProtocolAdapter(host), validate_session=identity.validate_session_transition
    )
    with pytest.raises(ValueError, match="member session"):
        runtime.recover_session("other-member")
    host._session_service.bind_runtime_components.assert_not_called()
    host._session_service.recover_session.assert_not_called()
    assert host.active_session_id == child.session_id
    assert host._memory_session_id == "session-123"


def test_failed_session_recovery_preserves_both_session_ids():
    host = _make_runtime_host(active_session_id="active")
    host._session_service.recover_session.return_value = False
    runtime = SessionRuntime(OrchestratorProtocolAdapter(host))
    assert runtime.recover_session("unavailable") is False
    assert host.active_session_id == "active"
    assert host._memory_session_id == "session-123"


async def test_recovery_rejects_active_turn_before_sync_or_hydration():
    import asyncio
    import pytest
    from victor.agent.runtime.context import RuntimeIdentity

    host = _make_runtime_host(active_session_id="active")
    identity = RuntimeIdentity(host)
    lock = asyncio.Lock()
    identity.bind_turn_lock(lock)
    runtime = SessionRuntime(host, validate_session=identity.validate_session_transition)
    async with lock:
        with pytest.raises(RuntimeError, match="active turn"):
            runtime.recover_session("resumed")
        host._session_service.bind_runtime_components.assert_not_called()
        host._session_service.recover_session.assert_not_called()
        assert host.active_session_id == "active"
    host._session_service.recover_session.return_value = True
    assert runtime.recover_session("resumed")
    assert identity.resolve().session_id == "resumed"


async def test_member_binding_registers_actual_session_before_compaction(tmp_path, monkeypatch):
    import sqlite3
    from victor.agent.conversation.store import ConversationStore
    from victor.agent.runtime.context import RuntimeIdentity
    from victor.agent.services.context_lifecycle_service import ContextLifecycleService

    store = ConversationStore(db_path=tmp_path / "sessions.db")
    initial = store.create_session(session_id="assembly-session")
    host = _make_runtime_host(
        memory_manager=store,
        active_session_id=None,
        _memory_session_id=initial.session_id,
        _conversation_controller=MagicMock(),
        provider_name="test",
        model="test-model",
    )
    host._runtime_identity = RuntimeIdentity(host)
    member = host._runtime_identity.resolve().derive_child(
        agent_id="member",
        display_name="Reviewer",
        role="reviewer",
        member_id="review",
    )
    runtime = SessionRuntime(host)
    import pytest

    with monkeypatch.context() as patcher:
        patcher.setattr(store, "_persist_session", MagicMock(side_effect=OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            runtime.bind_member(member)
    assert store.get_session(member.session_id) is None
    assert host.active_session_id is None
    assert host._memory_session_id == initial.session_id
    host._conversation_controller.set_conversation_store.assert_not_called()
    runtime.bind_member(member)
    lifecycle = ContextLifecycleService.with_defaults(
        max_tokens=10,
        min_messages_to_keep=2,
        conversation_store=store,
    )
    result = await lifecycle.after_agent_turn(
        member,
        messages=[{"role": "user", "content": "x" * 40} for _ in range(4)],
    )
    assert result["compaction_event_id"]
    runtime.bind_member(member)  # Existing durable events must survive rebinding.
    assert host.active_session_id == host._memory_session_id == member.session_id
    host._conversation_controller.set_conversation_store.assert_called_with(
        store, member.session_id
    )
    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute(
            "SELECT e.session_id, e.agent_id FROM compaction_events e "
            "JOIN sessions s ON s.session_id=e.session_id"
        ).fetchone()
        assert row == (member.session_id, "member")
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
