"""The first chat-runtime capability mutates its existing session owner."""

from dataclasses import fields, FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.agent.factory.chat_runtime_bindings import bind_chat_runtime_services
from victor.agent.factory.runtime_builders import RuntimeBuildersMixin
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.services.chat_runtime_services import ChatRuntimeServices
from victor.agent.services.chat_stream_executor import StreamingChatExecutor
from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.streaming_act_adapter import StreamingActAdapter
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import SessionStateManager


def owner():
    # Exercise the real facade's existing session properties without provider setup.
    result = object.__new__(AgentOrchestrator)
    result._session_accessor = SessionStateAccessor(SessionStateManager())
    return result


@pytest.mark.parametrize("through_protocol_adapter", [False, True])
async def test_factory_view_updates_existing_requirement_owner(
    through_protocol_adapter, monkeypatch
):
    orchestrator = owner()
    accessor = orchestrator._session_accessor
    orchestrator._required_files = ["old.py"]
    orchestrator._required_outputs = ["old output"]
    orchestrator._read_files_session.add("old.py")
    orchestrator._all_files_read_nudge_sent = True
    original_read_set = orchestrator._read_files_session
    supplied = (
        OrchestratorProtocolAdapter(orchestrator) if through_protocol_adapter else orchestrator
    )
    runtime = RuntimeBuildersMixin().create_streaming_chat_adapter(supplied)
    executor = StreamingChatExecutor(runtime)
    assert executor.services is runtime.services
    session = executor.services.session
    assert session.required_files is accessor.required_files
    assert session.read_files is original_read_set
    bus = SimpleNamespace(emit=AsyncMock())
    monkeypatch.setattr("victor.core.events.get_observability_bus", lambda: bus)

    await executor._extract_task_requirements(session, "Read ./src/main.py and ./src/main.py")

    assert orchestrator._required_files == ["./src/main.py"]
    assert session.required_files is orchestrator._required_files
    assert orchestrator._required_outputs == []
    assert original_read_set == set()
    assert orchestrator._all_files_read_nudge_sent is False
    bus.emit.assert_awaited_once_with(
        topic="state.task.requirements_extracted",
        data={
            "required_files": ["./src/main.py"],
            "required_outputs": [],
            "file_count": 1,
            "output_count": 0,
            "category": "state",
        },
    )


async def test_view_tracks_restored_and_reset_state_without_sharing_owners(monkeypatch):
    first, second = owner(), owner()
    view = bind_chat_runtime_services(first)
    other = bind_chat_runtime_services(second)
    manager = first._session_accessor.session_state
    other.session.required_files = ["unrelated.py"]
    other.session.read_files.add("unrelated.py")
    other.session.all_files_read_nudge_sent = True
    first._required_files = ["saved.py"]
    first._read_files_session.add("saved.py")
    first._all_files_read_nudge_sent = True
    checkpoint = manager.get_checkpoint_state()
    old_files = view.session.required_files
    old_reads = view.session.read_files
    manager.reset_for_new_turn()
    manager.apply_checkpoint_state(checkpoint)
    assert view.session.required_files == ["saved.py"]
    assert view.session.required_files is first._required_files
    assert view.session.required_files is not old_files
    assert view.session.read_files is first._read_files_session
    assert view.session.read_files is not old_reads
    assert view.session.all_files_read_nudge_sent is True
    bus = SimpleNamespace(emit=AsyncMock())
    monkeypatch.setattr("victor.core.events.get_observability_bus", lambda: bus)

    await StreamingChatExecutor._extract_task_requirements(view.session, "Read ../next.py")
    assert first._required_files == ["../next.py"]
    assert first._read_files_session == set()
    assert first._all_files_read_nudge_sent is False
    await StreamingChatExecutor._extract_task_requirements(view.session, "plain next turn")
    assert first._required_files == []
    assert first._required_outputs == []
    assert bus.emit.await_count == 1  # no event for empty requirements
    assert other.session.required_files == ["unrelated.py"]
    assert other.session.read_files == {"unrelated.py"}
    assert other.session.all_files_read_nudge_sent is True


async def test_act_prepare_uses_bound_session_capability(monkeypatch):
    orchestrator = owner()
    orchestrator._recovery_service = None
    orchestrator._recovery_coordinator = object()
    runtime = RuntimeBuildersMixin().create_streaming_chat_adapter(orchestrator)
    executor = StreamingChatExecutor(runtime)
    ctx = SimpleNamespace(max_exploration_iterations=4)
    monkeypatch.setattr(runtime, "_create_stream_context", AsyncMock(return_value=ctx))
    monkeypatch.setattr(executor, "_reset_streaming_turn_state", lambda _: None)
    monkeypatch.setattr(executor, "_apply_run_guidance", lambda *args: None)
    monkeypatch.setattr(executor, "_initialize_task_intent", lambda *args: [])
    bus = SimpleNamespace(emit=AsyncMock())
    monkeypatch.setattr("victor.core.events.get_observability_bus", lambda: bus)

    adapter = await StreamingActAdapter.prepare(executor, "Inspect ./live.py")

    assert adapter.session.stream_ctx is ctx
    assert orchestrator._required_files == ["./live.py"]
    assert executor.services.session.required_files is orchestrator._required_files
    bus.emit.assert_awaited_once()


def test_view_is_enumerated_and_does_not_retain_or_forward_facade():
    orchestrator = owner()
    view = bind_chat_runtime_services(orchestrator)
    assert [field.name for field in fields(view)] == ["session"]
    assert not hasattr(view, "__dict__")
    assert not hasattr(view.session, "__dict__")
    for name in ("orchestrator", "_orchestrator", "runtime_owner", "state_host", "get", "settings"):
        assert not hasattr(view, name)
        assert not hasattr(view.session, name)
    assert view.session._accessor is orchestrator._session_accessor
    with pytest.raises(FrozenInstanceError):
        view.session = object()
    with pytest.raises(AttributeError):
        view.session._orchestrator = orchestrator


def test_missing_session_owner_requires_explicit_binding():
    with pytest.raises(TypeError, match="SessionStateAccessor"):
        bind_chat_runtime_services(SimpleNamespace())
