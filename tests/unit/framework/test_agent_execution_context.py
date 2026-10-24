from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.core.shared_types import ConversationStage
from victor.framework.agent import Agent
from victor.framework.agent_components import AgentSession
from victor.providers.base import CompletionResponse
from victor.runtime.context import ResolvedRuntimeServices


class AgentOrchestratorMock:
    """Lightweight orchestrator double that passes Agent type validation."""


def _make_orchestrator(
    *,
    execution_context=None,
    chat_service=None,
):
    orchestrator = AgentOrchestratorMock()
    orchestrator.provider = SimpleNamespace(name="test_provider")
    orchestrator.model = "test-model"
    orchestrator._execution_context = execution_context
    orchestrator._chat_service = chat_service
    orchestrator._container = None
    orchestrator.chat = AsyncMock(
        return_value=CompletionResponse(
            content="orchestrator response",
            role="assistant",
            model="test-model",
        )
    )
    orchestrator.reset_conversation = MagicMock()
    orchestrator.get_stage = MagicMock(return_value=ConversationStage.INITIAL)
    return orchestrator


def test_agent_exposes_cached_execution_context():
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=object()))
    orchestrator = _make_orchestrator(execution_context=execution_context)

    agent = Agent.from_orchestrator(orchestrator)

    assert agent.execution_context is execution_context


def test_agent_execution_context_refreshes_from_orchestrator():
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=object()))
    orchestrator = _make_orchestrator(execution_context=execution_context)
    agent = Agent.from_orchestrator(orchestrator)
    agent._context = None

    assert agent.execution_context is execution_context


@pytest.mark.asyncio
async def test_agent_run_prefers_execution_context_chat_service():
    runtime_chat_service = SimpleNamespace(
        chat=AsyncMock(
            return_value=CompletionResponse(
                content="runtime service response",
                role="assistant",
                model="runtime-model",
            )
        )
    )
    legacy_chat_service = SimpleNamespace(
        chat=AsyncMock(
            side_effect=AssertionError("legacy orchestrator-bound service should not be used")
        )
    )
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=runtime_chat_service))
    orchestrator = _make_orchestrator(
        execution_context=execution_context,
        chat_service=legacy_chat_service,
    )

    agent = Agent.from_orchestrator(orchestrator)
    result = await agent.run("test prompt")

    runtime_chat_service.chat.assert_awaited_once_with("test prompt")
    assert result.content == "runtime service response"


@pytest.mark.asyncio
async def test_agent_session_initialize_prefers_execution_context_services():
    runtime_chat_service = SimpleNamespace(
        chat=AsyncMock(
            return_value=CompletionResponse(
                content="runtime service response",
                role="assistant",
                model="runtime-model",
            )
        ),
        reset_conversation=MagicMock(),
    )
    session_service = SimpleNamespace(
        create_session=AsyncMock(return_value="session-123"),
        close_session=AsyncMock(return_value=True),
    )
    execution_context = SimpleNamespace(
        services=SimpleNamespace(chat=runtime_chat_service, session=session_service)
    )
    orchestrator = _make_orchestrator(execution_context=execution_context)

    agent = Agent.from_orchestrator(orchestrator)
    session = AgentSession(agent, metadata={"origin": "framework"})

    await session.initialize()
    result = await session.send("test prompt")
    await session.close()

    session_service.create_session.assert_awaited_once_with(metadata={"origin": "framework"})
    runtime_chat_service.reset_conversation.assert_called_once_with()
    runtime_chat_service.chat.assert_awaited_once_with("test prompt")
    session_service.close_session.assert_awaited_once_with("session-123")
    assert result.content == "runtime service response"
    assert session.history == [
        {"role": "user", "content": "test prompt"},
        {"role": "assistant", "content": "runtime service response"},
    ]
    assert session._runtime_services == ResolvedRuntimeServices(
        chat=runtime_chat_service,
        session=session_service,
    )


@pytest.mark.asyncio
async def test_agent_session_stream_tracks_explicit_history():
    async def _stream_chat(_message: str):
        from victor.providers.base import StreamChunk

        yield StreamChunk(content="runtime ")
        yield StreamChunk(content="stream")
        yield StreamChunk(content="", is_final=True)

    runtime_chat_service = SimpleNamespace(
        stream_chat=_stream_chat,
        reset_conversation=MagicMock(),
    )
    execution_context = SimpleNamespace(
        services=SimpleNamespace(chat=runtime_chat_service, session=None)
    )
    orchestrator = _make_orchestrator(execution_context=execution_context)

    agent = Agent.from_orchestrator(orchestrator)
    session = AgentSession(agent)

    events = [event async for event in session.stream("stream prompt")]

    runtime_chat_service.reset_conversation.assert_called_once_with()
    assert [event.content for event in events if event.content] == [
        "runtime ",
        "stream",
    ]
    assert session.history == [
        {"role": "user", "content": "stream prompt"},
        {"role": "assistant", "content": "runtime stream"},
    ]


@pytest.mark.asyncio
async def test_agent_session_reset_clears_runtime_service_bundle():
    runtime_chat_service = SimpleNamespace(
        chat=AsyncMock(
            return_value=CompletionResponse(
                content="runtime service response",
                role="assistant",
                model="runtime-model",
            )
        ),
        reset_conversation=MagicMock(),
    )
    session_service = SimpleNamespace(
        create_session=AsyncMock(return_value="session-123"),
        close_session=AsyncMock(return_value=True),
    )
    execution_context = SimpleNamespace(
        services=SimpleNamespace(chat=runtime_chat_service, session=session_service)
    )
    orchestrator = _make_orchestrator(execution_context=execution_context)

    agent = Agent.from_orchestrator(orchestrator)
    session = AgentSession(agent)

    await session.initialize()
    assert session._runtime_services == ResolvedRuntimeServices(
        chat=runtime_chat_service,
        session=session_service,
    )

    await session.reset()

    assert session._runtime_services is None
    session_service.close_session.assert_awaited_once_with("session-123")
