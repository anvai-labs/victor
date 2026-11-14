from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.agent import Agent
from victor.framework.events import EventType
from victor.framework.client import VictorClient
from victor.framework.session_config import SessionConfig
from victor.framework.task import TaskResult
from victor.providers.base import CompletionResponse, StreamChunk


@pytest.mark.asyncio
async def test_victor_client_ensure_initialized_captures_execution_context() -> None:
    config = SessionConfig()
    client = VictorClient(config, container=object())
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=None))
    orchestrator = SimpleNamespace(
        _execution_context=execution_context, settings=SimpleNamespace(sandbox=None)
    )

    class _FakeAgent:
        def get_orchestrator(self):
            return orchestrator

    settings = SimpleNamespace(
        provider=SimpleNamespace(default_provider="ollama", default_model="test-model")
    )

    with (
        patch("victor.config.settings.load_settings", return_value=settings),
        patch("victor.framework.Agent.create", new=AsyncMock(return_value=_FakeAgent())),
    ):
        await client._ensure_initialized()

    assert client._context is execution_context


@pytest.mark.asyncio
async def test_victor_client_ensure_initialized_prefers_agent_execution_context_surface() -> None:
    config = SessionConfig()
    client = VictorClient(config, container=object())
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=None))

    class _FakeAgent:
        def __init__(self, ctx):
            self.execution_context = ctx

        def get_orchestrator(self):
            raise AssertionError("execution_context surface should be preferred over orchestrator")

    settings = SimpleNamespace(
        provider=SimpleNamespace(default_provider="ollama", default_model="test-model")
    )

    with (
        patch("victor.config.settings.load_settings", return_value=settings),
        patch(
            "victor.framework.Agent.create",
            new=AsyncMock(return_value=_FakeAgent(execution_context)),
        ),
    ):
        await client._ensure_initialized()

    assert client._context is execution_context


@pytest.mark.asyncio
async def test_victor_client_chat_prefers_execution_context_chat_service() -> None:
    config = SessionConfig.from_cli_flags(tool_budget=4)
    client = VictorClient(config, container=object())
    chat_service = SimpleNamespace(
        chat=AsyncMock(
            return_value=CompletionResponse(
                content="Service response",
                role="assistant",
                tool_calls=[{"name": "read"}],
                model="test-model",
            )
        )
    )
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service))
    orchestrator = SimpleNamespace(
        _execution_context=execution_context, settings=SimpleNamespace(sandbox=None)
    )

    class _FakeAgent:
        async def run(self, _message: str):
            raise AssertionError("VictorClient.chat() should prefer execution-context chat service")

        def get_orchestrator(self):
            return orchestrator

    client._agent = _FakeAgent()
    client._context = execution_context

    result = await client.chat("ping")

    chat_service.chat.assert_awaited_once_with("ping", stream=False)
    assert isinstance(result, TaskResult)
    assert result.content == "Service response"
    assert result.tool_count == 1


@pytest.mark.asyncio
async def test_victor_client_stream_prefers_execution_context_chat_service() -> None:
    config = SessionConfig()
    client = VictorClient(config, container=object())

    async def _stream_chat(_message: str):
        yield StreamChunk(content="Service ")
        yield StreamChunk(content="stream")
        yield StreamChunk(content="", is_final=True)

    chat_service = SimpleNamespace(
        stream_chat=MagicMock(side_effect=_stream_chat),
    )
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service, session=None))
    orchestrator = SimpleNamespace(
        _execution_context=execution_context, settings=SimpleNamespace(sandbox=None)
    )

    class _FakeAgent:
        async def stream(self, _message: str):
            raise AssertionError(
                "VictorClient.stream() should prefer execution-context chat service"
            )

        def get_orchestrator(self):
            return orchestrator

    client._agent = _FakeAgent()
    client._context = execution_context

    events = [event async for event in client.stream("ping")]

    chat_service.stream_chat.assert_called_once_with("ping")
    content_events = [event for event in events if event.event_type == "content"]
    assert [event.content for event in content_events] == ["Service ", "stream"]


@pytest.mark.asyncio
async def test_victor_client_session_prefers_execution_context_services() -> None:
    config = SessionConfig.from_cli_flags(tool_budget=4, enable_smart_routing=True)
    client = VictorClient(config, container=object())
    session_service = SimpleNamespace(
        create_session=AsyncMock(return_value="session-123"),
        close_session=AsyncMock(return_value=True),
    )
    chat_service = SimpleNamespace(
        chat=AsyncMock(
            return_value=CompletionResponse(
                content="Session response",
                role="assistant",
                model="test-model",
            )
        ),
        reset_conversation=MagicMock(),
    )
    execution_context = SimpleNamespace(
        services=SimpleNamespace(chat=chat_service, session=session_service)
    )
    mock_orchestrator = MagicMock()
    mock_orchestrator.__class__.__name__ = "AgentOrchestrator"
    mock_orchestrator.provider = MagicMock()
    mock_orchestrator.provider.name = "test-provider"
    mock_orchestrator.model = "test-model"
    mock_orchestrator.messages = []
    mock_orchestrator.settings = SimpleNamespace(sandbox=None)
    mock_orchestrator.get_stage = MagicMock(return_value=MagicMock(value="INITIAL"))
    mock_orchestrator.get_tool_calls_count = MagicMock(return_value=0)
    mock_orchestrator.get_tool_budget = MagicMock(return_value=50)
    mock_orchestrator.get_observed_files = MagicMock(return_value=set())
    mock_orchestrator.get_modified_files = MagicMock(return_value=set())
    mock_orchestrator.get_message_count = MagicMock(return_value=0)
    mock_orchestrator.is_streaming = MagicMock(return_value=False)
    mock_orchestrator.get_iteration_count = MagicMock(return_value=0)
    mock_orchestrator.get_max_iterations = MagicMock(return_value=25)
    mock_orchestrator.reset = MagicMock()
    mock_orchestrator.close = AsyncMock()
    mock_orchestrator.chat = AsyncMock(
        side_effect=AssertionError(
            "AgentSession.send() should prefer execution-context chat service"
        )
    )

    agent = Agent.from_orchestrator(mock_orchestrator)
    agent._context = execution_context

    client._agent = agent
    client._context = execution_context

    session = await client.create_session()
    result = await session.send("ping")
    await session.close()

    session_service.create_session.assert_awaited_once_with(
        metadata={"tool_budget": 4, "smart_routing": True}
    )
    chat_service.reset_conversation.assert_called_once_with()
    chat_service.chat.assert_awaited_once_with("ping")
    mock_orchestrator.chat.assert_not_awaited()
    session_service.close_session.assert_awaited_once_with("session-123")
    assert result.content == "Session response"
    assert session.turns[0]["prompt"] == "ping"
    assert session.turns[0]["response"] == "Session response"


@pytest.mark.asyncio
async def test_chat_session_stream_prefers_execution_context_chat_service() -> None:
    config = SessionConfig()
    client = VictorClient(config, container=object())

    async def _stream_chat(_message: str):
        yield StreamChunk(content="Session ")
        yield StreamChunk(content="stream")
        yield StreamChunk(content="", is_final=True)

    chat_service = SimpleNamespace(
        stream_chat=MagicMock(side_effect=_stream_chat),
        reset_conversation=MagicMock(),
    )
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service, session=None))
    mock_orchestrator = MagicMock()
    mock_orchestrator.__class__.__name__ = "AgentOrchestrator"
    mock_orchestrator.provider = MagicMock()
    mock_orchestrator.provider.name = "test-provider"
    mock_orchestrator.model = "test-model"
    mock_orchestrator.messages = []
    mock_orchestrator.settings = SimpleNamespace(sandbox=None)
    mock_orchestrator.get_stage = MagicMock(return_value=MagicMock(value="INITIAL"))
    mock_orchestrator.get_tool_calls_count = MagicMock(return_value=0)
    mock_orchestrator.get_tool_budget = MagicMock(return_value=50)
    mock_orchestrator.get_observed_files = MagicMock(return_value=set())
    mock_orchestrator.get_modified_files = MagicMock(return_value=set())
    mock_orchestrator.get_message_count = MagicMock(return_value=0)
    mock_orchestrator.is_streaming = MagicMock(return_value=False)
    mock_orchestrator.get_iteration_count = MagicMock(return_value=0)
    mock_orchestrator.get_max_iterations = MagicMock(return_value=25)
    mock_orchestrator.reset = MagicMock()
    mock_orchestrator.close = AsyncMock()
    mock_orchestrator.stream_chat = AsyncMock(
        side_effect=AssertionError(
            "AgentSession.stream() should prefer execution-context chat service"
        )
    )

    agent = Agent.from_orchestrator(mock_orchestrator)
    agent._context = execution_context

    client._agent = agent
    client._context = execution_context

    session = await client.create_session()
    events = [event async for event in session.stream("ping")]

    chat_service.stream_chat.assert_called_once_with("ping")
    mock_orchestrator.stream_chat.assert_not_called()
    assert session.turns[0]["prompt"] == "ping"
    assert session.turns[0]["response"] == "Session stream"
    content_events = [event for event in events if event.type == EventType.CONTENT]
    assert [event.content for event in content_events] == ["Session ", "stream"]
