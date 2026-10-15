"""Tests for VictorClient conversation management methods."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from victor.agent.services import protocols as service_protocols
from victor.framework.session_config import SessionConfig
from victor.framework.client import VictorClient


def _make_reset_client(binding="context"):
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.services.adapters.context_adapter import ContextServiceAdapter
    from victor.agent.services.chat_service import ChatService, ChatServiceConfig

    controller = ConversationController()
    controller.set_system_prompt("system")
    controller.ensure_system_message()
    controller.add_message("user", "preserve until reset")
    service = ChatService(
        ChatServiceConfig(), None, None, ContextServiceAdapter(controller), None, controller, None
    )
    client = VictorClient(SessionConfig(), container=object())
    container = MagicMock()
    container.get_optional.side_effect = lambda protocol: {
        service_protocols.ChatServiceProtocol: service,
    }.get(protocol)
    client._context = SimpleNamespace(
        services=SimpleNamespace(chat=service if binding == "context" else None),
        container=container,
    )
    client._initialized = True
    return client, service, controller


@pytest.mark.parametrize("binding", ["context", "container"])
async def test_victor_client_reset_conversation_uses_synchronous_service(binding):
    """Real service composition catches a sync/async mismatch that AsyncMock hid."""
    client, service, controller = _make_reset_client(binding)
    service._last_task_report = {"description": "previous"}
    controller.reset = MagicMock(wraps=controller.reset)

    await client.reset_conversation()

    controller.reset.assert_called_once()
    assert [(message.role, message.content) for message in controller.messages] == [
        ("system", "system")
    ]
    assert service.get_last_task_report() is None


@pytest.mark.asyncio
async def test_victor_client_reset_conversation_raises_when_not_initialized() -> None:
    """Test reset_conversation raises RuntimeError when not initialized."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    with pytest.raises(RuntimeError, match="not initialized"):
        await client.reset_conversation()


@pytest.mark.parametrize("binding", ["context", "container"])
async def test_victor_client_get_messages_uses_real_context_adapter(binding):
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

    controller = ConversationController()
    controller.add_message("user", "Atlas")
    controller.add_message("assistant", "History before Monaco")
    service = ContextServiceAdapter(controller)
    client = VictorClient(SessionConfig(), container=object())
    container = MagicMock()
    container.get_optional.side_effect = lambda protocol: {
        service_protocols.ContextServiceProtocol: service,
    }.get(protocol)
    client._context = SimpleNamespace(
        services=SimpleNamespace(context=service if binding == "context" else None),
        container=container,
    )
    client._initialized = True
    messages = await client.get_messages(limit=10, role="user")
    assert [(m.role, m.content) for m in messages] == [("user", "Atlas")]
    assert [m.content for m in await client.get_messages(limit=1)] == ["History before Monaco"]


@pytest.mark.asyncio
async def test_victor_client_get_messages_raises_when_not_initialized() -> None:
    """Test get_messages raises RuntimeError when not initialized."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    with pytest.raises(RuntimeError, match="not initialized"):
        await client.get_messages()


@pytest.mark.asyncio
async def test_victor_client_get_messages_returns_empty_when_service_unavailable() -> None:
    """Test get_messages returns empty list when ContextService unavailable."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    execution_context = SimpleNamespace(services=SimpleNamespace(context=None))

    client._context = execution_context
    client._initialized = True

    messages = await client.get_messages()

    assert messages == []


@pytest.mark.asyncio
async def test_victor_client_reset_conversation_handles_missing_service() -> None:
    """Test reset_conversation handles missing ChatService gracefully."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    execution_context = SimpleNamespace(services=SimpleNamespace(chat=None))

    client._context = execution_context
    client._initialized = True

    # Should not raise, just log warning
    await client.reset_conversation()


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [True, False])
async def test_victor_client_resume_session_validates_before_hydration(
    monkeypatch, allowed
) -> None:
    """resume_session loads the store, hydrates both stores, stamps the id."""
    session_data = {"metadata": {"title": "arithmetic"}, "conversation": {}}
    monkeypatch.setattr(
        "victor.agent.conversation.store.ConversationStore",
        lambda *a, **k: SimpleNamespace(load_session=lambda sid: session_data),
    )
    # Capture the hydration call — the helper's own two-store correctness is
    # covered by test_session_resume; here we assert VictorClient wires it.
    captured = {}

    def _fake_hydrate(context_service, controller, data):
        captured["context"] = context_service
        captured["controller"] = controller
        captured["data"] = data
        return data.get("metadata", {})

    monkeypatch.setattr("victor.agent.conversation.session_resume.hydrate_session", _fake_hydrate)

    client = VictorClient(SessionConfig(), container=object())
    client._context = object()
    client._initialized = True

    context_service = object()
    monkeypatch.setattr(
        client,
        "_resolve_runtime_services",
        lambda: SimpleNamespace(chat=None, context=context_service, recovery=None),
    )
    controller = object()
    orchestrator = SimpleNamespace(
        active_session_id=None,
        conversation_state=None,
        _conversation_controller=controller,
        _runtime_identity=SimpleNamespace(validate_session_transition=MagicMock()),
    )
    binding = MagicMock(
        side_effect=lambda store, sid: setattr(orchestrator, "active_session_id", sid)
    )
    orchestrator._get_session_runtime = lambda: SimpleNamespace(bind_restored_session=binding)
    client._agent = SimpleNamespace(_orchestrator=orchestrator)

    if not allowed:
        orchestrator._runtime_identity.validate_session_transition.side_effect = ValueError(
            "member session"
        )
        with pytest.raises(ValueError, match="member session"):
            await client.resume_session("s42")
        assert captured == {}
        binding.assert_not_called()
        assert orchestrator.active_session_id is None
        return

    metadata = await client.resume_session("s42")
    orchestrator._runtime_identity.validate_session_transition.assert_called_once_with("s42")

    assert metadata == {"title": "arithmetic"}
    assert captured["data"] == session_data
    assert captured["context"] is context_service
    assert captured["controller"] is controller
    assert orchestrator.active_session_id == "s42"


@pytest.mark.asyncio
async def test_victor_client_resume_session_returns_none_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "victor.agent.conversation.store.ConversationStore",
        lambda *a, **k: SimpleNamespace(load_session=lambda sid: None),
    )
    client = VictorClient(SessionConfig(), container=object())
    client._context = object()
    client._initialized = True

    assert await client.resume_session("gone") is None


@pytest.mark.asyncio
async def test_victor_client_resume_session_raises_when_not_initialized() -> None:
    client = VictorClient(SessionConfig(), container=object())
    with pytest.raises(RuntimeError):
        await client.resume_session("s1")


def test_victor_client_list_recent_sessions(monkeypatch, tmp_path) -> None:
    from victor.agent.conversation.store import ConversationStore
    from victor.agent.conversation.types import MessageRole

    store = ConversationStore(db_path=tmp_path / "history.db")
    session = store.create_session(provider="openai", model="gpt-4.1")
    store.add_message(session.session_id, MessageRole.USER, "Atlas conversation history")
    store.add_message(session.session_id, MessageRole.ASSISTANT, "History before Monaco")
    reopened = ConversationStore(db_path=store.db_path)
    monkeypatch.setattr("victor.agent.conversation.store.ConversationStore", lambda: reopened)
    client = VictorClient(SessionConfig(), container=object())
    summaries = client.list_recent_sessions(limit=5)
    assert len(summaries) == 1
    assert summaries[0]["session_id"] == session.session_id
    assert summaries[0]["message_count"] == 2
    assert summaries[0]["title"] == "Atlas conversation history"
    assert summaries[0]["preview_count"] == 0
    assert summaries[0]["tags"] == []


def test_victor_client_list_recent_sessions_survives_failure(monkeypatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr("victor.agent.conversation.store.ConversationStore", _boom)
    client = VictorClient(SessionConfig(), container=object())
    assert client.list_recent_sessions() == []


@pytest.mark.parametrize("entry", ["chat", "planning", "stream", "aggregate"])
@pytest.mark.parametrize("target", ["active", "resumed"])
async def test_resume_rejects_active_turn_before_hydration(monkeypatch, entry, target):
    import asyncio
    from victor.agent.runtime.context import RuntimeIdentity
    from victor.agent.services.chat_service import ChatService, ChatServiceConfig
    from victor.providers.base import CompletionResponse, StreamChunk

    owner = SimpleNamespace(active_session_id="active", _memory_session_id=None)
    identity = owner._runtime_identity = RuntimeIdentity(owner)
    owner._get_session_runtime = lambda: SimpleNamespace(
        bind_restored_session=lambda store, sid: setattr(owner, "active_session_id", sid)
    )
    service = ChatService(ChatServiceConfig(), None, None, None, None, None, None)
    identity.bind_turn_lock(service._turn_lock)
    started, finish = asyncio.Event(), asyncio.Event()

    async def execute(message):
        started.set()
        await finish.wait()
        return CompletionResponse(content=message, role="assistant")

    async def stream(message, **kwargs):
        started.set()
        yield StreamChunk(content=message)
        await finish.wait()
        yield StreamChunk(content="done", is_final=True)

    service.bind_runtime_components(
        turn_executor=SimpleNamespace(execute_agentic_loop=execute),
        planning_handler=execute,
        stream_chat_handler=stream,
        turn_runtime=SimpleNamespace(
            resolve_runtime_context=identity.resolve,
            enter=lambda *a, **k: None,
            exit=lambda *a, **k: None,
            start_task_report=lambda *a, **k: None,
            finish_task_report=lambda *a, **k: None,
        ),
    )
    client = VictorClient(SessionConfig(), container=object())
    client._context, client._initialized = object(), True
    client._agent = SimpleNamespace(_orchestrator=owner)
    monkeypatch.setattr(
        client, "_resolve_runtime_services", lambda: SimpleNamespace(context=object())
    )
    monkeypatch.setattr(
        "victor.agent.conversation.store.ConversationStore",
        lambda: SimpleNamespace(load_session=lambda _: {"metadata": {"title": "saved"}}),
    )
    hydrate = MagicMock(return_value={"title": "saved"})
    monkeypatch.setattr("victor.agent.conversation.session_resume.hydrate_session", hydrate)

    active_stream = None
    task = None
    if entry == "stream":
        active_stream = service.stream_chat("hello", use_planning=False)
        await anext(active_stream)
    else:
        task = asyncio.create_task(
            service.chat_with_planning("hello")
            if entry == "planning"
            else service.chat("hello", stream=entry == "aggregate", use_planning=False)
        )
        await asyncio.wait_for(started.wait(), 2)
    try:
        with pytest.raises(RuntimeError, match="active turn"):
            await client.resume_session(target)
        hydrate.assert_not_called()
        assert owner.active_session_id == identity.resolve().session_id == "active"
    finally:
        finish.set()
        if active_stream is not None:
            await active_stream.aclose()
        if task is not None:
            await asyncio.wait_for(task, 2)
    assert await client.resume_session(target) == {"title": "saved"}
    hydrate.assert_called_once()
    assert identity.resolve().session_id == target


@pytest.mark.parametrize("entry", ["chat", "planning", "stream", "aggregate"])
async def test_client_reset_rejects_active_turn_without_mutation(entry):
    import asyncio
    from victor.agent.runtime.context import AgentRuntimeContext
    from victor.providers.base import CompletionResponse, StreamChunk

    client, service, controller = _make_reset_client()
    entered, release = asyncio.Event(), asyncio.Event()

    async def execute(message):
        entered.set()
        await release.wait()
        return CompletionResponse(content=message, role="assistant")

    async def stream(message, **kwargs):
        entered.set()
        yield StreamChunk(content=message)
        await release.wait()
        yield StreamChunk(content="done", is_final=True)

    service.bind_runtime_components(
        turn_executor=SimpleNamespace(execute_agentic_loop=execute),
        planning_handler=execute,
        stream_chat_handler=stream,
        turn_runtime=SimpleNamespace(
            resolve_runtime_context=lambda: AgentRuntimeContext(
                "root", "Root", "supervisor", "active"
            ),
            enter=lambda *a, **k: None,
            exit=lambda *a, **k: None,
            start_task_report=lambda *a, **k: None,
            finish_task_report=lambda *a, **k: None,
        ),
    )
    active_stream, task = None, None
    if entry == "stream":
        active_stream = service.stream_chat("hello", use_planning=False)
        await anext(active_stream)
    else:
        task = asyncio.create_task(
            service.chat_with_planning("hello")
            if entry == "planning"
            else service.chat("hello", stream=entry == "aggregate", use_planning=False)
        )
        await asyncio.wait_for(entered.wait(), 2)
    report = service._last_task_report = {"description": "previous"}
    messages = list(controller.messages)
    controller.reset = MagicMock(wraps=controller.reset)
    try:
        with pytest.raises(RuntimeError, match="active turn"):
            await client.reset_conversation()
        assert controller.messages == messages
        assert service.get_last_task_report() == report
        controller.reset.assert_not_called()
        assert service._turn_lock.locked()
    finally:
        release.set()
        if active_stream is not None:
            await active_stream.aclose()
        if task is not None:
            await asyncio.wait_for(task, 2)
    await client.reset_conversation()
    controller.reset.assert_called_once()
    assert [message.role for message in controller.messages] == ["system"]


@pytest.mark.parametrize("memory_enabled", [True, False])
async def test_resume_continuation_persists_to_selected_session(
    tmp_path, monkeypatch, memory_enabled
):
    from victor.agent.conversation.store import ConversationStore
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.conversation.types import MessageRole
    from victor.agent.runtime.context import RuntimeIdentity
    from victor.agent.services.adapters.context_adapter import ContextServiceAdapter
    from victor.agent.services.session_runtime import SessionRuntime

    store = ConversationStore(db_path=tmp_path / "sessions.db")
    store.create_session(session_id="assembly")
    store.create_session(session_id="atlas")
    store.add_message("atlas", MessageRole.USER, "Atlas before Monaco")
    controller = ConversationController()
    if memory_enabled:
        controller.set_conversation_store(store, "assembly")
    host = SimpleNamespace(
        active_session_id=None,
        _memory_session_id="assembly" if memory_enabled else None,
        _conversation_controller=controller,
        memory_manager=store if memory_enabled else None,
        _session_service=MagicMock(),
        _lifecycle_manager=None,
        _checkpoint_manager=None,
        _session_cost_tracker=None,
        conversation_state=None,
    )
    host._runtime_identity = RuntimeIdentity(host)
    runtime = SessionRuntime(
        host, validate_session=host._runtime_identity.validate_session_transition
    )
    host._get_session_runtime = lambda: runtime
    client = VictorClient(SessionConfig(), container=object())
    client._agent = SimpleNamespace(_orchestrator=host)
    client._context, client._initialized = object(), True
    monkeypatch.setattr(
        client,
        "_resolve_runtime_services",
        lambda: SimpleNamespace(context=ContextServiceAdapter(controller)),
    )
    monkeypatch.setattr("victor.agent.conversation.store.ConversationStore", lambda: store)
    await client.resume_session("atlas")
    assert host.active_session_id == "atlas"
    assert host._memory_session_id == ("atlas" if memory_enabled else None)
    assert controller._session_id == ("atlas" if memory_enabled else None)
    assert controller._conversation_store is (store if memory_enabled else None)
    assert [m.content for m in controller.messages if m.role == "user"] == ["Atlas before Monaco"]
    host._session_service.bind_runtime_components.assert_called_once()
    from victor.agent.services.chat_persistence import ChatPersistenceMixin

    host.usage_logger = None
    host.conversation = controller
    ChatPersistenceMixin.publish_message(
        host,
        "assistant",
        "Continued",
        persist_synchronously=True,
        require_persistence=memory_enabled,
    )
    reopened = type(store)(db_path=store.db_path)
    assert {s.session_id: s.message_count for s in reopened.list_session_summaries()} == {
        "atlas": 2 if memory_enabled else 1,
        "assembly": 0,
    }


def test_picker_uses_runtime_store_instead_of_default(tmp_path, monkeypatch):
    from victor.agent.conversation.store import ConversationStore

    runtime_store = ConversationStore(db_path=tmp_path / "runtime.db")
    runtime_store.create_session(session_id="runtime")
    default_store = ConversationStore(db_path=tmp_path / "default.db")
    default_store.create_session(session_id="default")
    monkeypatch.setattr("victor.agent.conversation.store.ConversationStore", lambda: default_store)
    client = VictorClient(SessionConfig(), container=object())
    assert [s["session_id"] for s in client.list_recent_sessions()] == ["default"]
    client._agent = SimpleNamespace(_orchestrator=SimpleNamespace(memory_manager=runtime_store))
    assert [s["session_id"] for s in client.list_recent_sessions()] == ["runtime"]
    assert client._stored_conversations() is runtime_store
