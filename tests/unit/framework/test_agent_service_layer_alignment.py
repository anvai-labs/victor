# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for framework service-first runtime alignment."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.framework.agent import Agent
from victor.framework.events import EventType
from victor.providers.base import CompletionResponse, StreamChunk


class AgentOrchestrator(MagicMock):
    """MagicMock subclass whose type name satisfies Agent.__init__ validation."""


@pytest.fixture
def mock_orchestrator() -> AgentOrchestrator:
    """Create a mock AgentOrchestrator for testing."""
    orchestrator = AgentOrchestrator()
    orchestrator.provider = MagicMock()
    orchestrator.provider.name = "test_provider"
    orchestrator.model = "test-model"
    orchestrator.messages = []

    def mock_chat_response(content="Hello World", tool_calls=None):
        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="test-model",
        )

    orchestrator.chat = AsyncMock(return_value=mock_chat_response())

    async def mock_stream_chat(message, **kwargs):
        _ = message, kwargs
        chunks = [
            StreamChunk(content="Hello ", finish_reason=None, is_complete=False),
            StreamChunk(content="World", finish_reason=None, is_complete=False),
            StreamChunk(content="", finish_reason="stop", is_complete=True),
        ]
        for chunk in chunks:
            yield chunk

    orchestrator.stream_chat = mock_stream_chat
    orchestrator.get_stage = MagicMock(return_value=MagicMock(value="INITIAL"))
    orchestrator.get_tool_calls_count = MagicMock(return_value=0)
    orchestrator.get_tool_budget = MagicMock(return_value=50)
    orchestrator.get_observed_files = MagicMock(return_value=set())
    orchestrator.get_modified_files = MagicMock(return_value=set())
    orchestrator.get_message_count = MagicMock(return_value=0)
    orchestrator.is_streaming = MagicMock(return_value=False)
    orchestrator.current_provider = "test_provider"
    orchestrator.current_model = "test-model"
    orchestrator.get_iteration_count = MagicMock(return_value=0)
    orchestrator.get_max_iterations = MagicMock(return_value=25)
    orchestrator.reset = MagicMock()
    orchestrator.close = AsyncMock()
    orchestrator._container = MagicMock()
    return orchestrator


@pytest.fixture
def mock_chat_service():
    """Create a mock ChatService."""

    def mock_chat_response(content="Hello from ChatService", tool_calls=None):
        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="test-model",
        )

    chat_service = SimpleNamespace(
        chat=AsyncMock(return_value=mock_chat_response()),
    )

    async def mock_stream_chat(message, **kwargs):
        _ = message, kwargs
        chunks = [
            StreamChunk(content="Hello ", finish_reason=None, is_complete=False),
            StreamChunk(content="from ", finish_reason=None, is_complete=False),
            StreamChunk(content="ChatService", finish_reason=None, is_complete=False),
            StreamChunk(content="", finish_reason="stop", is_complete=True),
        ]
        for chunk in chunks:
            yield chunk

    chat_service.stream_chat = mock_stream_chat
    return chat_service


class TestAgentRunServiceFirst:
    """Tests for Agent.run() using the shared service-first runtime."""

    @pytest.mark.asyncio
    async def test_run_prefers_execution_context_chat_service(
        self, mock_orchestrator, mock_chat_service
    ):
        agent = Agent(orchestrator=mock_orchestrator)
        agent._context = SimpleNamespace(services=SimpleNamespace(chat=mock_chat_service))

        result = await agent.run("Hello")

        mock_chat_service.chat.assert_awaited_once_with("Hello")
        mock_orchestrator.chat.assert_not_called()
        assert result.content == "Hello from ChatService"

    @pytest.mark.asyncio
    async def test_run_fallback_is_internal_and_suppresses_legacy_warning(self, mock_orchestrator):
        agent = Agent(orchestrator=mock_orchestrator)
        mock_orchestrator._container = None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            result = await agent.run("Hello")

        mock_orchestrator.chat.assert_awaited_once_with("Hello")
        assert result.content == "Hello World"
        assert not caught


class TestAgentStreamServiceFirst:
    """Tests for Agent.stream() using the shared service-first runtime."""

    @pytest.mark.asyncio
    async def test_stream_prefers_execution_context_chat_service(
        self, mock_orchestrator, mock_chat_service
    ):
        agent = Agent(orchestrator=mock_orchestrator)
        agent._context = SimpleNamespace(services=SimpleNamespace(chat=mock_chat_service))

        events = [event async for event in agent.stream("Hello")]

        content_events = [event for event in events if event.type == EventType.CONTENT]
        assert content_events
        contents = "".join(event.content or "" for event in content_events)
        assert "Hello" in contents
        assert "ChatService" in contents

    @pytest.mark.asyncio
    async def test_stream_fallback_is_internal_and_suppresses_legacy_warning(
        self, mock_orchestrator
    ):
        agent = Agent(orchestrator=mock_orchestrator)
        mock_orchestrator._container = None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            events = [event async for event in agent.stream("Hello")]

        content_events = [event for event in events if event.type == EventType.CONTENT]
        assert "".join(event.content or "" for event in content_events) == "Hello World"
        assert not caught
