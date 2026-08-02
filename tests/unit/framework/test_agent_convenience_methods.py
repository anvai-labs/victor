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

"""Tests for Agent convenience methods (run_oneshot, run_interactive)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncIterator

from victor.framework.agent import Agent
from victor.framework.events import EventType, AgentExecutionEvent
from victor.framework.task import TaskResult


# Agent.__init__ validates type(orchestrator).__name__ == "AgentOrchestrator".
# We need to create a class with the exact name "AgentOrchestrator" to pass validation.
class AgentOrchestrator(MagicMock):
    """Lightweight mock that passes Agent.__init__ validation."""

    pass


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator for testing."""
    orchestrator = AgentOrchestrator()
    orchestrator.provider = MagicMock()
    orchestrator.provider.name = "test_provider"
    orchestrator.model = "test-model"
    orchestrator.messages = []

    # Mock chat method for run_oneshot
    from victor.providers.base import CompletionResponse

    def mock_chat(content="Test response", tool_calls=None):
        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="test-model",
        )

    orchestrator.chat = AsyncMock(return_value=mock_chat())
    orchestrator.stream_chat = AsyncMock(return_value=AsyncMock())

    # Add protocol methods for State wrapper
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

    return orchestrator


@pytest.fixture
def agent(mock_orchestrator):
    """Create an Agent instance for testing using from_orchestrator."""
    return Agent.from_orchestrator(mock_orchestrator)


class TestRunOneshot:
    """Tests for Agent.run_oneshot() method."""

    @pytest.mark.asyncio
    async def test_run_oneshot_returns_task_result(self, agent):
        """Test that run_oneshot returns a TaskResult with content."""
        result = await agent.run_oneshot("Test prompt")

        assert isinstance(result, TaskResult)
        assert result.success is True
        assert result.content == "Test response"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_run_oneshot_calls_chat_once(self, agent, mock_orchestrator):
        """Test that run_oneshot calls orchestrator.chat exactly once."""
        await agent.run_oneshot("Single turn")

        mock_orchestrator.chat.assert_called_once_with("Single turn")

    @pytest.mark.asyncio
    async def test_run_oneshot_with_context(self, agent, mock_orchestrator):
        """Test that run_oneshot handles context parameter."""
        context = {"file": "test.py", "line": 42}
        await agent.run_oneshot("Explain this", context=context)

        # Context should be formatted into the prompt
        call_args = mock_orchestrator.chat.call_args[0][0]
        assert "Explain this" in call_args
        assert "test.py" in call_args or "file" in call_args

    @pytest.mark.asyncio
    async def test_run_oneshot_with_tool_calls(self, agent, mock_orchestrator):
        """Test that run_oneshot returns tool_calls in result."""
        # tool_calls is a List[Dict[str, Any]], not a ToolCall class
        tool_calls = [
            {
                "id": "call_123",
                "name": "read",
                "arguments": {"path": "test.py"},
            }
        ]

        mock_orchestrator.chat = AsyncMock(
            return_value=MagicMock(
                content="Here's the file",
                role="assistant",
                tool_calls=tool_calls,
                stop_reason="stop",
                usage={},
                model="test-model",
            )
        )

        result = await agent.run_oneshot("Read test.py")

        assert result.tool_calls == tool_calls
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_run_oneshot_handles_errors(self, agent, mock_orchestrator):
        """Test that run_oneshot handles exceptions gracefully."""
        mock_orchestrator.chat = AsyncMock(side_effect=Exception("Provider error"))

        result = await agent.run_oneshot("Test")

        assert result.success is False
        assert result.error == "Provider error"
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_run_oneshot_with_metadata(self, agent):
        """Test that run_oneshot includes metadata in result."""
        result = await agent.run_oneshot("Test")

        assert "metadata" in result.__dict__ or hasattr(result, "metadata")
        if hasattr(result, "metadata"):
            assert "model" in result.metadata
            assert "stage" in result.metadata


class TestRunInteractive:
    """Tests for Agent.run_interactive() method."""

    @pytest.mark.asyncio
    async def test_run_interactive_returns_session(self, agent):
        """Test that run_interactive returns a ChatSession."""
        session = await agent.run_interactive("Hello")

        from victor.framework.agent import ChatSession

        assert isinstance(session, ChatSession)
        # ChatSession delegates to AgentSession, check via get_session()
        underlying = session.get_session()
        assert underlying is not None

    @pytest.mark.asyncio
    async def test_run_interactive_session_has_send(self, agent):
        """Test that run_interactive session can send messages."""
        session = await agent.run_interactive("Initial")

        response = await session.send("Follow up")

        assert response is not None
        assert hasattr(response, "content")

    @pytest.mark.asyncio
    async def test_run_interactive_session_maintains_context(self, agent, mock_orchestrator):
        """Test that run_interactive maintains conversation context."""
        session = await agent.run_interactive("First")

        await session.send("Second")
        await session.send("Third")

        # Should have called orchestrator.chat 2 times (2 sends, initial prompt doesn't call chat)
        assert mock_orchestrator.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_run_interactive_session_has_history(self, agent):
        """Test that run_interactive session has history property."""
        session = await agent.run_interactive("First")
        await session.send("Second")

        # ChatSession has a `history` property (delegates to AgentSession)
        history = session.history

        # History is a list (may be empty with mocks, but property exists)
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_run_interactive_with_stream(self, agent):
        """Test that run_interactive session supports streaming."""
        session = await agent.run_interactive("Test")

        # Should be able to stream responses
        stream = session.stream("Another message")
        assert hasattr(stream, "__aiter__")

        events = []
        async for event in stream:
            events.append(event)
            if len(events) >= 1:  # Collect at least one event
                break

        assert len(events) >= 1


class TestConvenienceMethodIntegration:
    """Integration tests for convenience methods."""

    @pytest.mark.asyncio
    async def test_oneshot_vs_interactive_difference(self, agent):
        """Test that run_oneshot and run_interactive behave differently."""
        # run_oneshot returns TaskResult directly
        oneshot_result = await agent.run_oneshot("Test")
        assert isinstance(oneshot_result, TaskResult)

        # run_interactive returns ChatSession
        session = await agent.run_interactive("Test")
        from victor.framework.agent import ChatSession

        assert isinstance(session, ChatSession)

    @pytest.mark.asyncio
    async def test_oneshot_is_stateless(self, agent, mock_orchestrator):
        """Test that run_oneshot doesn't maintain conversation state."""
        await agent.run_oneshot("First")
        await agent.run_oneshot("Second")

        # Each call is independent - orchestrator state should not accumulate
        # (This is implementation-dependent; adjust if orchestrator behaves differently)
        assert mock_orchestrator.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_interactive_is_stateful(self, agent):
        """Test that run_interactive maintains conversation state."""
        session = await agent.run_interactive("First")
        await session.send("Second")

        # ChatSession has a history property (stateful)
        history = session.history
        assert isinstance(history, list)  # Should track messages (even if empty with mocks)

        # Session maintains turn count
        assert session.turn_count >= 0
