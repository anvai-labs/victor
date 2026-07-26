"""Tests for VictorClient conversation management methods."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from victor.agent.services import protocols as service_protocols
from victor.framework.session_config import SessionConfig
from victor.framework.client import VictorClient


@pytest.mark.asyncio
async def test_victor_client_reset_conversation_delegates_to_chat_service() -> None:
    """Test reset_conversation delegates to ChatService."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    mock_chat_service = AsyncMock()
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=mock_chat_service))

    client._context = execution_context
    client._initialized = True

    await client.reset_conversation()

    mock_chat_service.reset_conversation.assert_awaited_once()


@pytest.mark.asyncio
async def test_victor_client_reset_conversation_raises_when_not_initialized() -> None:
    """Test reset_conversation raises RuntimeError when not initialized."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    with pytest.raises(RuntimeError, match="not initialized"):
        await client.reset_conversation()


@pytest.mark.asyncio
async def test_victor_client_get_messages_delegates_to_context_service() -> None:
    """Test get_messages delegates to ContextService."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    fake_messages = [MagicMock(role="user", content="hello")]
    mock_context_service = AsyncMock()
    mock_context_service.get_messages.return_value = fake_messages

    execution_context = SimpleNamespace(services=SimpleNamespace(context=mock_context_service))

    client._context = execution_context
    client._initialized = True

    messages = await client.get_messages(limit=10, role="user")

    mock_context_service.get_messages.assert_called_once_with(limit=10, role="user")
    assert messages == fake_messages


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
async def test_victor_client_reset_conversation_resolves_chat_service_from_context_container() -> (
    None
):
    """Test reset_conversation uses canonical runtime service resolution."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    mock_chat_service = AsyncMock()
    container = MagicMock()
    container.get_optional.side_effect = lambda protocol: {
        service_protocols.ChatServiceProtocol: mock_chat_service,
    }.get(protocol)
    execution_context = SimpleNamespace(
        services=SimpleNamespace(chat=None),
        container=container,
    )

    client._context = execution_context
    client._initialized = True

    await client.reset_conversation()

    mock_chat_service.reset_conversation.assert_awaited_once()


@pytest.mark.asyncio
async def test_victor_client_get_messages_resolves_context_service_from_context_container() -> None:
    """Test get_messages uses canonical runtime service resolution."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    fake_messages = [MagicMock(role="assistant", content="resolved")]
    mock_context_service = AsyncMock()
    mock_context_service.get_messages.return_value = fake_messages
    container = MagicMock()
    container.get_optional.side_effect = lambda protocol: {
        service_protocols.ContextServiceProtocol: mock_context_service,
    }.get(protocol)
    execution_context = SimpleNamespace(
        services=SimpleNamespace(context=None),
        container=container,
    )

    client._context = execution_context
    client._initialized = True

    messages = await client.get_messages(limit=5)

    mock_context_service.get_messages.assert_awaited_once_with(limit=5, role=None)
    assert messages == fake_messages


@pytest.mark.asyncio
async def test_victor_client_resume_session_hydrates_and_stamps(monkeypatch) -> None:
    """resume_session loads the store, hydrates both stores, stamps the id."""
    import victor.agent.sqlite_session_persistence as persistence_mod
    import victor.framework.client as client_mod

    session_data = {"metadata": {"title": "arithmetic"}, "conversation": {}}
    monkeypatch.setattr(
        persistence_mod,
        "get_sqlite_session_persistence",
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

    monkeypatch.setattr(
        "victor.agent.conversation.session_resume.hydrate_session", _fake_hydrate
    )

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
        active_session_id=None, conversation_state=None, _conversation_controller=controller
    )
    client._agent = SimpleNamespace(_orchestrator=orchestrator)

    metadata = await client.resume_session("s42")

    assert metadata == {"title": "arithmetic"}
    assert captured["data"] == session_data
    assert captured["context"] is context_service
    assert captured["controller"] is controller
    assert orchestrator.active_session_id == "s42"


@pytest.mark.asyncio
async def test_victor_client_resume_session_returns_none_when_missing(monkeypatch) -> None:
    import victor.agent.sqlite_session_persistence as persistence_mod

    monkeypatch.setattr(
        persistence_mod,
        "get_sqlite_session_persistence",
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


def test_victor_client_list_recent_sessions(monkeypatch) -> None:
    import victor.agent.sqlite_session_persistence as persistence_mod

    rows = [{"session_id": "s1", "title": "a"}, {"session_id": "s2", "title": "b"}]
    monkeypatch.setattr(
        persistence_mod,
        "get_sqlite_session_persistence",
        lambda *a, **k: SimpleNamespace(list_sessions=lambda limit: rows[:limit]),
    )
    client = VictorClient(SessionConfig(), container=object())
    assert client.list_recent_sessions(limit=5) == rows


def test_victor_client_list_recent_sessions_survives_failure(monkeypatch) -> None:
    import victor.agent.sqlite_session_persistence as persistence_mod

    def _boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(persistence_mod, "get_sqlite_session_persistence", _boom)
    client = VictorClient(SessionConfig(), container=object())
    assert client.list_recent_sessions() == []
