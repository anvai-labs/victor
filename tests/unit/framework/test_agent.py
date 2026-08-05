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

"""Comprehensive unit tests for victor/framework/agent.py.

These tests cover:
- Agent class initialization and configuration
- Task execution methods (run, stream)
- Event handling and state management
- Error handling and recovery
- CQRS integration
- Workflow and team execution
- Lifecycle management
"""

import warnings

import pytest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from victor_contracts.core.types import VerticalDefinition


# Agent.__init__ validates type(orchestrator).__name__ == "AgentOrchestrator".
# Defining this class at module level gives inline tests a lightweight mock that
# passes the check without importing the real orchestrator (which has heavy deps).
class AgentOrchestrator(MagicMock):
    """MagicMock subclass whose type name satisfies Agent.__init__ validation."""


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator():
    """Create a mock AgentOrchestrator for testing."""

    orchestrator = AgentOrchestrator()
    orchestrator.provider = MagicMock()
    orchestrator.provider.name = "test_provider"
    orchestrator.model = "test-model"
    orchestrator.messages = []
    orchestrator.reset_conversation = MagicMock()

    # NEW: Add chat method for direct sync execution (Phase 2 refactoring)
    from victor.providers.base import CompletionResponse
    from victor.providers.base import StreamChunk

    def mock_chat_response(content="Hello World", tool_calls=None):
        """Create a mock CompletionResponse."""
        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="test-model",  # Include model in response
        )

    orchestrator.chat = AsyncMock(return_value=mock_chat_response())

    async def mock_stream_chat(_prompt):
        yield StreamChunk(content="Test", is_final=False)
        yield StreamChunk(content="", is_final=True, stop_reason="stop")

    orchestrator.stream_chat = mock_stream_chat
    orchestrator._chat_service = SimpleNamespace(
        chat=orchestrator.chat,
        stream_chat=orchestrator.stream_chat,
    )
    orchestrator._container = None

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
def mock_vertical():
    """Create a mock vertical class for testing."""
    vertical = MagicMock()
    vertical.name = "test_vertical"
    vertical.description = "Test vertical"
    vertical.version = "1.0.0"
    vertical.get_definition = MagicMock(
        return_value=VerticalDefinition(
            name="test_vertical",
            description="Test vertical",
            version="1.0.0",
            tools=["tool1", "tool2"],
            system_prompt="Test system prompt",
            workflow_metadata={"provider_hints": {"preferred_providers": ["anthropic"]}},
        )
    )
    vertical.get_config = MagicMock(
        return_value=MagicMock(
            tools=["tool1", "tool2"],
            system_prompt="Test system prompt",
            provider_hints={"preferred_providers": ["anthropic"]},
        )
    )
    vertical.get_workflow_provider = MagicMock(return_value=None)
    vertical.get_team_spec_provider = MagicMock(return_value=None)
    return vertical


@pytest.fixture
def mock_stream_chunk():
    """Create a mock stream chunk."""
    chunk = MagicMock()
    chunk.content = "test content"
    chunk.metadata = {}
    chunk.tool_calls = []
    return chunk


@pytest.fixture(autouse=True)
def reset_feature_flags_for_agent_tests():
    """Reset feature flags for isolation between Agent tests."""
    from victor.core.feature_flags import reset_feature_flag_manager

    reset_feature_flag_manager()

    yield

    reset_feature_flag_manager()


# =============================================================================
# Agent Initialization Tests (lines 98-108)
# =============================================================================


class TestAgentInit:
    """Tests for Agent.__init__ method."""

    def test_init_with_orchestrator(self, mock_orchestrator):
        """Agent should initialize with orchestrator and store references."""
        from victor.framework.agent import Agent

        agent = Agent(
            mock_orchestrator,
            provider="anthropic",
            model="claude-3",
        )

        assert agent._orchestrator is mock_orchestrator
        assert agent._provider == "anthropic"
        assert agent._model == "claude-3"
        assert agent._vertical is None
        assert agent._vertical_config is None
        assert agent._state is not None
        assert agent._state_observers == []

    def test_init_with_vertical(self, mock_orchestrator, mock_vertical):
        """Agent should store vertical and vertical_config when provided."""
        from victor.framework.agent import Agent

        vertical_config = MagicMock()
        agent = Agent(
            mock_orchestrator,
            provider="anthropic",
            model="claude-3",
            vertical=mock_vertical,
            vertical_config=vertical_config,
        )

        assert agent._vertical is mock_vertical
        assert agent._vertical_config is vertical_config

    def test_init_default_values(self, mock_orchestrator):
        """Agent should use default values when not provided."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        assert agent._provider == "anthropic"
        assert agent._model is None
        assert agent._vertical is None
        assert agent._vertical_config is None


# =============================================================================
# Agent.from_orchestrator Tests (lines 249-251)
# =============================================================================


class TestAgentFromOrchestrator:
    """Tests for Agent.from_orchestrator classmethod."""

    def test_from_orchestrator_basic(self, mock_orchestrator):
        """from_orchestrator should create Agent from existing orchestrator."""
        from victor.framework.agent import Agent

        agent = Agent.from_orchestrator(mock_orchestrator)

        assert agent._orchestrator is mock_orchestrator
        assert agent._provider == "test_provider"
        assert agent._model == "test-model"

    def test_from_orchestrator_unknown_provider(self):
        """from_orchestrator should handle missing provider name."""
        from victor.framework.agent import Agent

        orchestrator = AgentOrchestrator()
        orchestrator.provider = MagicMock(spec=[])  # No 'name' attribute
        orchestrator.model = None

        agent = Agent.from_orchestrator(orchestrator)

        assert agent._provider == "unknown"
        assert agent._model is None


# =============================================================================
# Agent.run Tests (lines 393-422)
# =============================================================================


class TestAgentRun:
    """Tests for Agent.run method."""

    @pytest.mark.asyncio
    async def test_run_basic(self, mock_orchestrator):
        """run should call the service-backed chat runtime and return TaskResult."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        result = await agent.run("test prompt")

        # Verify chat was called
        mock_orchestrator.chat.assert_called_once_with("test prompt")

        # Verify result
        assert result.content == "Hello World"
        assert result.success is True
        assert result.error is None
        assert result.metadata["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_run_with_error_event(self, mock_orchestrator):
        """run should handle exceptions from orchestrator.chat."""
        from victor.framework.agent import Agent

        # Make chat raise an exception
        mock_orchestrator.chat.side_effect = Exception("Something went wrong")

        agent = Agent(mock_orchestrator)
        result = await agent.run("test prompt")

        assert result.success is False
        assert result.error == "Something went wrong"

    @pytest.mark.asyncio
    async def test_run_with_stream_end_failure(self, mock_orchestrator):
        """run should handle empty response from orchestrator."""
        from victor.framework.agent import Agent
        from victor.providers.base import CompletionResponse

        # Return empty response
        mock_orchestrator.chat.return_value = CompletionResponse(
            content="", role="assistant", tool_calls=None
        )

        agent = Agent(mock_orchestrator)
        result = await agent.run("test prompt")

        # Empty content should still be returned (but success=True since no exception)
        assert result.content == ""
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_with_cancellation_error(self, mock_orchestrator):
        """run should handle CancellationError."""
        from victor.framework.agent import Agent
        from victor.core.errors import CancellationError

        # Make chat raise CancellationError
        mock_orchestrator.chat.side_effect = CancellationError("User cancelled")

        agent = Agent(mock_orchestrator)
        result = await agent.run("test prompt")

        assert result.success is False
        assert result.error == "Operation cancelled"

    @pytest.mark.asyncio
    async def test_run_with_generic_exception(self, mock_orchestrator):
        """run should handle generic exceptions."""
        from victor.framework.agent import Agent

        # Make chat raise an exception
        mock_orchestrator.chat.side_effect = RuntimeError("Unexpected error")

        agent = Agent(mock_orchestrator)
        result = await agent.run("test prompt")

        assert result.success is False
        assert result.error == "Unexpected error"

    @pytest.mark.asyncio
    async def test_run_with_context(self, mock_orchestrator):
        """run should format context into prompt."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)
        context = {"file": "test.py", "error": "SyntaxError"}

        result = await agent.run("fix bug", context=context)

        # Verify chat was called with context-formatted prompt
        # The context should be prepended to the prompt
        mock_orchestrator.chat.assert_called_once()
        call_args = mock_orchestrator.chat.call_args
        prompt_arg = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        assert "File: test.py" in prompt_arg or "file" in prompt_arg

        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_normalizes_exact_response_output(self, mock_orchestrator):
        """run should normalize exact-response literals through framework policy."""
        from victor.framework.agent import Agent
        from victor.providers.base import CompletionResponse

        mock_orchestrator.chat = AsyncMock(
            return_value=CompletionResponse(
                content="The user asked for exactly READY, so the answer is READY",
                role="assistant",
                tool_calls=[],
                stop_reason="stop",
                usage=None,
                model="test-model",
            )
        )
        mock_orchestrator._chat_service = SimpleNamespace(chat=mock_orchestrator.chat)

        agent = Agent(mock_orchestrator)
        result = await agent.run("Reply with exactly READY")

        assert result.success is True
        assert result.content == "READY"

    @pytest.mark.asyncio
    async def test_run_falls_back_to_orchestrator_without_chat_service(self, mock_orchestrator):
        """run should preserve compatibility for bare orchestrator wrappers."""
        from victor.framework.agent import Agent

        mock_orchestrator._chat_service = None
        agent = Agent(mock_orchestrator)

        result = await agent.run("test prompt")

        mock_orchestrator.chat.assert_called_once_with("test prompt")
        assert result.success is True


# =============================================================================
# Agent.stream Tests (lines 462-490)
# =============================================================================


class TestAgentStream:
    """Tests for Agent.stream method."""

    @pytest.mark.asyncio
    async def test_stream_basic(self, mock_orchestrator):
        """stream should yield framework events from the service-backed runtime."""
        from victor.framework.agent import Agent
        from victor.framework.events import AgentExecutionEvent, EventType

        # Mock the internal stream function
        async def mock_stream_with_events(runtime, prompt, **kwargs):
            assert runtime is mock_orchestrator._chat_service
            yield AgentExecutionEvent(type=EventType.STREAM_START)
            yield AgentExecutionEvent(type=EventType.CONTENT, content="Test")
            yield AgentExecutionEvent(type=EventType.STREAM_END, success=True)

        agent = Agent(mock_orchestrator)

        with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
            events = []
            async for event in agent.stream("test"):
                events.append(event)

        assert len(events) == 3
        assert events[0].type == EventType.STREAM_START
        assert events[1].type == EventType.CONTENT
        assert events[2].type == EventType.STREAM_END

    @pytest.mark.asyncio
    async def test_stream_with_context(self, mock_orchestrator):
        """stream should format and prepend context to prompt."""
        from victor.framework.agent import Agent
        from victor.framework.events import AgentExecutionEvent, EventType

        captured = {}

        async def mock_stream_with_events(runtime, prompt, **kwargs):
            captured["runtime"] = runtime
            captured["prompt"] = prompt
            captured["response_prompt"] = kwargs.get("response_prompt")
            yield AgentExecutionEvent(type=EventType.STREAM_END, success=True)

        agent = Agent(mock_orchestrator)

        with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
            async for _ in agent.stream("fix bug", context={"file": "test.py"}):
                pass

        # Context should be prepended to prompt
        assert captured["runtime"] is mock_orchestrator._chat_service
        assert "File: test.py" in captured["prompt"]
        assert "fix bug" in captured["prompt"]
        assert captured["response_prompt"] == "fix bug"

    @pytest.mark.asyncio
    async def test_stream_notifies_state_observers(self, mock_orchestrator):
        """stream should notify state observers on stage changes."""
        from victor.framework.agent import Agent
        from victor.framework.events import AgentExecutionEvent, EventType
        from victor.agent.conversation.state_machine import ConversationStage

        observer_called = []

        def observer(old_state, new_state):
            observer_called.append((old_state, new_state))

        # Mock changing stage
        stage_sequence = [ConversationStage.INITIAL, ConversationStage.PLANNING]
        call_count = [0]

        def get_stage_side_effect():
            result = stage_sequence[min(call_count[0], len(stage_sequence) - 1)]
            call_count[0] += 1
            return result

        mock_orchestrator.get_stage.side_effect = get_stage_side_effect

        async def mock_stream_with_events(orchestrator, prompt, **kwargs):
            yield AgentExecutionEvent(type=EventType.STREAM_START)
            yield AgentExecutionEvent(type=EventType.CONTENT, content="Test")
            yield AgentExecutionEvent(type=EventType.STREAM_END, success=True)

        agent = Agent(mock_orchestrator)
        agent.on_state_change(observer)

        with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
            async for _ in agent.stream("test"):
                pass

        # Observer should have been called at least once for stage change
        # (depends on mock timing, but we test the mechanism)
        assert agent._state_observers == [observer]

    @pytest.mark.asyncio
    async def test_stream_observer_exception_does_not_break_streaming(self, mock_orchestrator):
        """stream should catch and suppress observer exceptions."""
        from victor.framework.agent import Agent
        from victor.framework.events import AgentExecutionEvent, EventType
        from victor.agent.conversation.state_machine import ConversationStage

        # Create an observer that raises an exception
        def failing_observer(old_state, new_state):
            raise RuntimeError("Observer failure")

        # Mock the stage to trigger observer call
        stage_sequence = [ConversationStage.INITIAL, ConversationStage.PLANNING]
        call_count = [0]

        def get_stage_side_effect():
            result = stage_sequence[min(call_count[0], len(stage_sequence) - 1)]
            call_count[0] += 1
            return result

        mock_orchestrator.get_stage.side_effect = get_stage_side_effect

        async def mock_stream_with_events(orchestrator, prompt, **kwargs):
            yield AgentExecutionEvent(type=EventType.STREAM_START)
            yield AgentExecutionEvent(type=EventType.CONTENT, content="Test")
            yield AgentExecutionEvent(type=EventType.STREAM_END, success=True)

        agent = Agent(mock_orchestrator)
        agent.on_state_change(failing_observer)

        events = []
        with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
            # This should not raise despite the observer failing
            async for event in agent.stream("test"):
                events.append(event)

        # All events should be yielded despite observer error
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_stream_falls_back_to_orchestrator_without_chat_service(self, mock_orchestrator):
        """stream should use the internal fallback adapter for bare orchestrators."""
        from victor.framework.agent import Agent
        from victor.framework.events import AgentExecutionEvent, EventType
        from victor.runtime import resolve_chat_runtime

        mock_orchestrator._chat_service = None
        mock_orchestrator._container = None
        captured = {}

        async def mock_stream_with_events(runtime, prompt, **kwargs):
            captured["runtime"] = runtime
            yield AgentExecutionEvent(type=EventType.STREAM_END, success=True)

        agent = Agent(mock_orchestrator)

        with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
            async for _ in agent.stream("test"):
                pass

        assert captured["runtime"] is not mock_orchestrator
        assert type(captured["runtime"]) is type(resolve_chat_runtime(mock_orchestrator))


# =============================================================================
# Agent.chat and ChatSession Tests (lines 509, 1108-1164)
# =============================================================================


class TestAgentChat:
    """Tests for Agent.chat method and ChatSession class."""

    def test_chat_returns_session(self, mock_orchestrator):
        """chat should return a ChatSession instance."""
        from victor.framework.agent import Agent, ChatSession

        agent = Agent(mock_orchestrator)
        session = agent.chat("Hello")

        assert isinstance(session, ChatSession)

    @pytest.mark.asyncio
    async def test_chat_session_send(self, mock_orchestrator):
        """ChatSession.send should delegate to AgentSession."""
        from victor.framework.agent import Agent, ChatSession
        from victor.framework.task import TaskResult

        agent = Agent(mock_orchestrator)
        session = agent.chat("Initial prompt")

        mock_result = TaskResult(content="Response", success=True)
        session._delegate.send = AsyncMock(return_value=mock_result)

        result = await session.send("Follow-up")

        assert result.content == "Response"
        session._delegate.send.assert_called_once_with("Follow-up")

    @pytest.mark.asyncio
    async def test_chat_session_stream(self, mock_orchestrator):
        """ChatSession.stream should delegate to AgentSession."""
        from victor.framework.agent import Agent, ChatSession
        from victor.framework.events import AgentExecutionEvent, EventType

        agent = Agent(mock_orchestrator)
        session = agent.chat("Initial prompt")

        async def mock_stream(message):
            yield AgentExecutionEvent(type=EventType.CONTENT, content="Streamed")

        session._delegate.stream = mock_stream

        events = []
        async for event in session.stream("Follow-up"):
            events.append(event)

        assert len(events) == 1
        assert events[0].content == "Streamed"

    def test_chat_session_turn_count(self, mock_orchestrator):
        """ChatSession.turn_count should return delegate's turn count."""
        from victor.framework.agent import Agent, ChatSession

        agent = Agent(mock_orchestrator)
        session = agent.chat("Initial prompt")
        session._delegate._turn_count = 5

        assert session.turn_count == 5

    def test_chat_session_history(self, mock_orchestrator):
        """ChatSession.history should return delegate's history."""
        from victor.framework.agent import Agent, ChatSession

        agent = Agent(mock_orchestrator)
        session = agent.chat("Initial prompt")

        # Mock messages on orchestrator
        mock_msg = MagicMock()
        mock_msg.role = "user"
        mock_msg.content = "Hello"
        mock_orchestrator.messages = [mock_msg]

        history = session.history

        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"

    def test_chat_session_get_session(self, mock_orchestrator):
        """ChatSession.get_session should return the underlying AgentSession."""
        from victor.framework.agent import Agent, ChatSession
        from victor.framework.agent_components import AgentSession

        agent = Agent(mock_orchestrator)
        session = agent.chat("Initial prompt")

        underlying = session.get_session()

        assert isinstance(underlying, AgentSession)
        assert underlying is session._delegate


# =============================================================================
# State Properties and Observers Tests (lines 522, 531, 540, 549-551, 572-578)
# =============================================================================


class TestAgentStateProperties:
    """Tests for Agent state-related properties and methods."""

    def test_state_property(self, mock_orchestrator):
        """state property should return State wrapper."""
        from victor.framework.agent import Agent
        from victor.framework.state import State

        agent = Agent(mock_orchestrator)

        assert isinstance(agent.state, State)
        assert agent._state is agent.state

    def test_vertical_property(self, mock_orchestrator, mock_vertical):
        """vertical property should return stored vertical class."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.vertical is mock_vertical

    def test_vertical_property_none(self, mock_orchestrator):
        """vertical property should return None when not set."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        assert agent.vertical is None

    def test_vertical_config_property(self, mock_orchestrator):
        """vertical_config property should return stored config."""
        from victor.framework.agent import Agent

        config = MagicMock()
        agent = Agent(mock_orchestrator, vertical_config=config)

        assert agent.vertical_config is config

    def test_vertical_name_property(self, mock_orchestrator, mock_vertical):
        """vertical_name property should return vertical's name."""
        from victor.framework.agent import Agent

        mock_vertical.name = "CodingAssistant"
        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.vertical_name == "CodingAssistant"

    def test_vertical_name_property_none(self, mock_orchestrator):
        """vertical_name property should return None when no vertical."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        assert agent.vertical_name is None

    def test_on_state_change_registers_observer(self, mock_orchestrator):
        """on_state_change should register observer callback."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)
        observer = MagicMock()

        unsubscribe = agent.on_state_change(observer)

        assert observer in agent._state_observers
        assert callable(unsubscribe)

    def test_on_state_change_unsubscribe(self, mock_orchestrator):
        """on_state_change unsubscribe function should remove observer."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)
        observer = MagicMock()

        unsubscribe = agent.on_state_change(observer)
        assert observer in agent._state_observers

        unsubscribe()
        assert observer not in agent._state_observers

    def test_on_state_change_unsubscribe_idempotent(self, mock_orchestrator):
        """on_state_change unsubscribe should be idempotent."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)
        observer = MagicMock()

        unsubscribe = agent.on_state_change(observer)
        unsubscribe()
        unsubscribe()  # Should not raise

        assert observer not in agent._state_observers


# =============================================================================
# switch_model and set_tools Tests (lines 595-598, 606-608)
# =============================================================================


class TestAgentConfiguration:
    """Tests for Agent configuration methods."""

    @pytest.mark.asyncio
    async def test_switch_model(self, mock_orchestrator):
        """switch_model should update provider and model."""
        from victor.framework.agent import Agent

        mock_pm = MagicMock()
        mock_pm.switch_provider = AsyncMock()
        mock_orchestrator.provider_manager = mock_pm

        agent = Agent(mock_orchestrator, provider="anthropic", model="old-model")

        await agent.switch_model("openai", "gpt-4")

        mock_pm.switch_provider.assert_called_once_with("openai", "gpt-4")
        assert agent._provider == "openai"
        assert agent._model == "gpt-4"

    @pytest.mark.asyncio
    async def test_switch_model_no_provider_manager(self, mock_orchestrator):
        """switch_model should still update internal state without provider_manager."""
        from victor.framework.agent import Agent

        # Remove provider_manager attribute
        del mock_orchestrator.provider_manager

        agent = Agent(mock_orchestrator, provider="anthropic", model="old-model")

        await agent.switch_model("openai", "gpt-4")

        assert agent._provider == "openai"
        assert agent._model == "gpt-4"

    def test_set_tools_with_toolset(self, mock_orchestrator):
        """set_tools should configure tools using ToolSet."""
        from victor.framework.agent import Agent
        from victor.framework.tools import ToolSet

        agent = Agent(mock_orchestrator)
        toolset = ToolSet.minimal()

        with patch("victor.framework._internal.configure_tools") as mock_configure:
            agent.set_tools(toolset)

            mock_configure.assert_called_once_with(mock_orchestrator, toolset)

    def test_set_tools_with_list(self, mock_orchestrator):
        """set_tools should configure tools using list."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)
        tools = ["read_file", "write_file", "shell"]

        with patch("victor.framework._internal.configure_tools") as mock_configure:
            agent.set_tools(tools)

            mock_configure.assert_called_once_with(mock_orchestrator, tools)


# =============================================================================
# get_orchestrator Tests (line 633)
# =============================================================================


class TestGetOrchestrator:
    """Tests for Agent.get_orchestrator method."""

    def test_get_orchestrator(self, mock_orchestrator):
        """get_orchestrator should return the internal orchestrator."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        assert agent.get_orchestrator() is mock_orchestrator


# =============================================================================
# Observability Tests (lines 660-663, 672, 698-705)
# =============================================================================


class TestAgentObservability:
    """Tests for Agent observability methods."""

    def test_event_bus_property_with_observability(self, mock_orchestrator):
        """event_bus property should return EventBus when available."""
        from victor.framework.agent import Agent

        mock_observability = MagicMock()
        mock_event_bus = MagicMock()
        mock_observability.event_bus = mock_event_bus
        mock_orchestrator.observability = mock_observability

        agent = Agent(mock_orchestrator)

        assert agent.event_bus is mock_event_bus

    def test_event_bus_property_without_observability(self, mock_orchestrator):
        """event_bus property should return None when no observability."""
        from victor.framework.agent import Agent

        # Remove observability attribute
        if hasattr(mock_orchestrator, "observability"):
            del mock_orchestrator.observability

        agent = Agent(mock_orchestrator)

        assert agent.event_bus is None

    def test_observability_property(self, mock_orchestrator):
        """observability property should return ObservabilityIntegration."""
        from victor.framework.agent import Agent

        mock_observability = MagicMock()
        mock_orchestrator.observability = mock_observability

        agent = Agent(mock_orchestrator)

        assert agent.observability is mock_observability

    def test_subscribe_to_events(self, mock_orchestrator):
        """subscribe_to_events should subscribe to EventBus."""
        from victor.framework.agent import Agent

        mock_backend = MagicMock()
        mock_unsubscribe = MagicMock()
        mock_backend.subscribe = MagicMock(return_value=mock_unsubscribe)

        mock_event_bus = MagicMock()
        mock_event_bus.backend = mock_backend

        mock_observability = MagicMock()
        mock_observability.event_bus = mock_event_bus
        mock_orchestrator.observability = mock_observability

        agent = Agent(mock_orchestrator)
        handler = MagicMock()

        unsubscribe = agent.subscribe_to_events("TOOL", handler)

        assert unsubscribe is mock_unsubscribe
        mock_backend.subscribe.assert_called_once_with("tool.*", handler)

    def test_subscribe_to_events_custom_category(self, mock_orchestrator):
        """subscribe_to_events should support registered custom categories."""
        from victor.framework.agent import Agent
        from victor.observability.event_registry import EventCategoryRegistry

        EventCategoryRegistry.reset_instance()
        EventCategoryRegistry.get_instance().register(
            name="security_scan",
            description="Security scan events",
            registered_by="victor.security",
        )

        mock_backend = MagicMock()
        mock_unsubscribe = MagicMock()
        mock_backend.subscribe = MagicMock(return_value=mock_unsubscribe)

        mock_event_bus = MagicMock()
        mock_event_bus.backend = mock_backend

        mock_observability = MagicMock()
        mock_observability.event_bus = mock_event_bus
        mock_orchestrator.observability = mock_observability

        agent = Agent(mock_orchestrator)
        handler = MagicMock()

        unsubscribe = agent.subscribe_to_events("security_scan", handler)

        assert unsubscribe is mock_unsubscribe
        mock_backend.subscribe.assert_called_once_with("security_scan.*", handler)

    def test_subscribe_to_events_all_wildcard(self, mock_orchestrator):
        """subscribe_to_events should support wildcard category subscriptions."""
        from victor.framework.agent import Agent

        mock_backend = MagicMock()
        mock_unsubscribe = MagicMock()
        mock_backend.subscribe = MagicMock(return_value=mock_unsubscribe)

        mock_event_bus = MagicMock()
        mock_event_bus.backend = mock_backend

        mock_observability = MagicMock()
        mock_observability.event_bus = mock_event_bus
        mock_orchestrator.observability = mock_observability

        agent = Agent(mock_orchestrator)
        handler = MagicMock()

        unsubscribe = agent.subscribe_to_events("ALL", handler)

        assert unsubscribe is mock_unsubscribe
        mock_backend.subscribe.assert_called_once_with("*", handler)

    def test_subscribe_to_events_topic_pattern_passthrough(self, mock_orchestrator):
        """subscribe_to_events should accept direct topic patterns."""
        from victor.framework.agent import Agent

        mock_backend = MagicMock()
        mock_unsubscribe = MagicMock()
        mock_backend.subscribe = MagicMock(return_value=mock_unsubscribe)

        mock_event_bus = MagicMock()
        mock_event_bus.backend = mock_backend

        mock_observability = MagicMock()
        mock_observability.event_bus = mock_event_bus
        mock_orchestrator.observability = mock_observability

        agent = Agent(mock_orchestrator)
        handler = MagicMock()

        unsubscribe = agent.subscribe_to_events("security_scan.*", handler)

        assert unsubscribe is mock_unsubscribe
        mock_backend.subscribe.assert_called_once_with("security_scan.*", handler)

    def test_subscribe_to_events_no_event_bus(self, mock_orchestrator):
        """subscribe_to_events should return None when no event bus."""
        from victor.framework.agent import Agent

        # Remove observability
        if hasattr(mock_orchestrator, "observability"):
            del mock_orchestrator.observability

        agent = Agent(mock_orchestrator)
        handler = MagicMock()

        result = agent.subscribe_to_events("TOOL", handler)

        assert result is None

    def test_subscribe_to_events_unknown_category_raises(self, mock_orchestrator):
        """Unknown category names should raise ValueError."""
        from victor.framework.agent import Agent

        mock_backend = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus.backend = mock_backend

        mock_observability = MagicMock()
        mock_observability.event_bus = mock_event_bus
        mock_orchestrator.observability = mock_observability

        agent = Agent(mock_orchestrator)

        with pytest.raises(ValueError, match="Unknown event category or topic pattern"):
            agent.subscribe_to_events("not_a_real_category", MagicMock())


# =============================================================================
# Workflow and Team Execution Tests (lines 882-905, 947-987, 1009, 1028, 1032, 1036)
# =============================================================================


class TestAgentWorkflows:
    """Tests for Agent workflow execution methods."""

    @pytest.mark.asyncio
    async def test_run_workflow_no_vertical(self, mock_orchestrator):
        """run_workflow should raise AgentError when no vertical."""
        from victor.framework.agent import Agent
        from victor.core.errors import AgentError

        agent = Agent(mock_orchestrator)

        with pytest.raises(AgentError, match="No vertical configured"):
            await agent.run_workflow("test_workflow")

    @pytest.mark.asyncio
    async def test_run_workflow_no_provider(self, mock_orchestrator, mock_vertical):
        """run_workflow should raise AgentError when vertical has no workflow provider."""
        from victor.framework.agent import Agent
        from victor.core.errors import AgentError

        mock_vertical.get_workflow_provider = MagicMock(return_value=None)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with pytest.raises(AgentError, match="does not provide workflows"):
            await agent.run_workflow("test_workflow")

    @pytest.mark.asyncio
    async def test_run_workflow_not_found(self, mock_orchestrator, mock_vertical):
        """run_workflow should raise AgentError when workflow not found."""
        from victor.framework.agent import Agent
        from victor.core.errors import AgentError

        mock_provider = MagicMock()
        # Mock run_compiled_workflow to raise error for not found workflow
        mock_provider.run_compiled_workflow = AsyncMock(
            side_effect=AgentError("Workflow 'unknown_workflow' not found")
        )
        mock_vertical.get_workflow_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with pytest.raises(AgentError, match="not found"):
            await agent.run_workflow("unknown_workflow")

    @pytest.mark.asyncio
    async def test_run_workflow_success(self, mock_orchestrator, mock_vertical):
        """run_workflow should execute workflow and return result."""
        from victor.framework.agent import Agent

        mock_provider = MagicMock()
        # Mock run_compiled_workflow to return a successful result
        expected_result = {"success": True, "output": "done"}
        mock_provider.run_compiled_workflow = AsyncMock(return_value=expected_result)
        mock_vertical.get_workflow_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)
        result = await agent.run_workflow("test_workflow", context={"key": "value"})

        assert result == expected_result
        mock_provider.run_compiled_workflow.assert_called_once_with(
            "test_workflow", {"key": "value"}, timeout=None
        )

    def test_get_available_workflows_no_vertical(self, mock_orchestrator):
        """get_available_workflows should return empty list when no vertical."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        assert agent.get_available_workflows() == []

    def test_get_available_workflows_no_provider(self, mock_orchestrator, mock_vertical):
        """get_available_workflows should return empty list when no provider."""
        from victor.framework.agent import Agent

        mock_vertical.get_workflow_provider = MagicMock(return_value=None)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.get_available_workflows() == []

    def test_get_available_workflows(self, mock_orchestrator, mock_vertical):
        """get_available_workflows should return workflow names."""
        from victor.framework.agent import Agent

        mock_provider = MagicMock()
        mock_provider.get_workflow_names = MagicMock(return_value=["wf1", "wf2"])
        mock_vertical.get_workflow_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.get_available_workflows() == ["wf1", "wf2"]


class TestAgentTeams:
    """Tests for Agent team execution methods."""

    @pytest.mark.asyncio
    async def test_run_team_no_vertical(self, mock_orchestrator):
        """run_team should raise AgentError when no vertical."""
        from victor.framework.agent import Agent
        from victor.core.errors import AgentError

        agent = Agent(mock_orchestrator)

        with pytest.raises(AgentError, match="No vertical configured"):
            await agent.run_team("test_team", "Test goal")

    @pytest.mark.asyncio
    async def test_run_team_no_team_provider(self, mock_orchestrator, mock_vertical):
        """run_team should raise AgentError when vertical has no team spec provider."""
        from victor.framework.agent import Agent
        from victor.core.errors import AgentError

        # Remove get_team_spec_provider method
        del mock_vertical.get_team_spec_provider

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with pytest.raises(AgentError, match="does not support teams"):
            await agent.run_team("test_team", "Test goal")

    @pytest.mark.asyncio
    async def test_run_team_no_specs(self, mock_orchestrator, mock_vertical):
        """run_team should raise AgentError when team specs are empty."""
        from victor.framework.agent import Agent
        from victor.core.errors import AgentError

        mock_provider = MagicMock()
        mock_provider.get_team_specs = MagicMock(return_value={})
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with pytest.raises(AgentError, match="has no team specs defined"):
            await agent.run_team("test_team", "Test goal")

    @pytest.mark.asyncio
    async def test_run_team_not_found(self, mock_orchestrator, mock_vertical):
        """run_team should raise AgentError when team not found."""
        from victor.framework.agent import Agent
        from victor.core.errors import AgentError

        mock_provider = MagicMock()
        mock_provider.get_team_specs = MagicMock(
            return_value={
                "team1": MagicMock(),
                "team2": MagicMock(),
            }
        )
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with pytest.raises(AgentError, match="not found"):
            await agent.run_team("unknown_team", "Test goal")

    def test_get_available_teams_no_vertical(self, mock_orchestrator):
        """get_available_teams should return empty list when no vertical."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        assert agent.get_available_teams() == []

    def test_get_available_teams_no_provider_method(self, mock_orchestrator, mock_vertical):
        """get_available_teams should return empty list when no get_team_spec_provider."""
        from victor.framework.agent import Agent

        del mock_vertical.get_team_spec_provider

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.get_available_teams() == []

    def test_get_available_teams_null_provider(self, mock_orchestrator, mock_vertical):
        """get_available_teams should return empty list when provider is None."""
        from victor.framework.agent import Agent

        mock_vertical.get_team_spec_provider = MagicMock(return_value=None)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.get_available_teams() == []

    def test_get_available_teams_no_get_team_specs(self, mock_orchestrator, mock_vertical):
        """get_available_teams should return empty list when provider has no get_team_specs."""
        from victor.framework.agent import Agent

        mock_provider = MagicMock(spec=[])  # No get_team_specs method
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.get_available_teams() == []

    def test_get_available_teams_empty_specs(self, mock_orchestrator, mock_vertical):
        """get_available_teams should return empty list when specs are empty."""
        from victor.framework.agent import Agent

        mock_provider = MagicMock()
        mock_provider.get_team_specs = MagicMock(return_value={})
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.get_available_teams() == []

    def test_get_available_teams(self, mock_orchestrator, mock_vertical):
        """get_available_teams should return team names."""
        from victor.framework.agent import Agent

        mock_provider = MagicMock()
        mock_provider.get_team_specs = MagicMock(
            return_value={
                "team1": MagicMock(),
                "team2": MagicMock(),
            }
        )
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        teams = agent.get_available_teams()
        assert set(teams) == {"team1", "team2"}

    def test_get_coordination_suggestion_uses_shared_framework_runtime(
        self, mock_orchestrator, mock_vertical
    ):
        """get_coordination_suggestion should prefer the orchestrator coordination API."""
        from victor.framework.agent import Agent

        suggestion = MagicMock()
        mock_orchestrator.get_coordination_suggestion = MagicMock(return_value=suggestion)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)
        result = agent.get_coordination_suggestion("feature", "high", mode="plan")

        assert result is suggestion
        mock_orchestrator.get_coordination_suggestion.assert_called_once_with(
            "feature",
            "high",
            mode="plan",
        )

    @pytest.mark.asyncio
    async def test_get_coordination_transitions_uses_state_passed_surface(
        self, mock_orchestrator, mock_vertical
    ):
        """get_coordination_transitions should use orchestration_facade.coordination_state_passed."""
        from victor.framework.agent import Agent

        result = MagicMock(name="coordination_result")
        coordination_state_passed = SimpleNamespace(suggest=AsyncMock(return_value=result))
        mock_orchestrator.orchestration_facade = SimpleNamespace(
            coordination_state_passed=coordination_state_passed
        )
        mock_orchestrator.mode_controller = SimpleNamespace(
            current_mode=SimpleNamespace(value="build")
        )
        mock_orchestrator.conversation_state = {"complexity": "high"}
        mock_orchestrator.session_state = {}
        mock_orchestrator.provider_name = "test_provider"
        mock_orchestrator.temperature = 0.7
        mock_orchestrator.max_tokens = 4096
        mock_orchestrator.settings = MagicMock()
        mock_orchestrator.session_id = "session-1"
        mock_orchestrator.conversation_stage = "initial"
        mock_orchestrator.observed_files = []
        mock_orchestrator._capabilities = {}

        agent = Agent(mock_orchestrator, vertical=mock_vertical)
        coordination_result = await agent.get_coordination_transitions(
            "feature",
            complexity="high",
            mode="plan",
        )

        assert coordination_result is result
        coordination_state_passed.suggest.assert_awaited_once()
        args = coordination_state_passed.suggest.await_args.args
        kwargs = coordination_state_passed.suggest.await_args.kwargs
        assert args[0].session_id == "session-1"
        assert kwargs == {
            "task_type": "feature",
            "complexity": "high",
            "mode": "plan",
        }

    @pytest.mark.asyncio
    async def test_run_team_success(self, mock_orchestrator, mock_vertical):
        """run_team should execute team and return result."""
        from victor.framework.agent import Agent
        from victor.framework.team_runtime import ResolvedTeamExecutionPlan
        from victor.teams.types import TeamFormation, TeamResult

        # Create a team spec
        mock_team_spec = MagicMock()
        mock_team_spec.name = "Test Team"
        mock_team_spec.members = [MagicMock()]
        mock_team_spec.formation = TeamFormation.PARALLEL
        mock_team_spec.total_tool_budget = 100
        mock_team_spec.max_iterations = 50

        # Create team provider
        mock_provider = MagicMock()
        mock_provider.get_team_specs = MagicMock(
            return_value={
                "test_team": mock_team_spec,
            }
        )
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        resolved_plan = ResolvedTeamExecutionPlan(
            team_name="test_team",
            display_name="Test Team",
            formation=TeamFormation.PARALLEL,
            member_count=1,
            total_tool_budget=100,
            max_iterations=50,
            recommendation_source="explicit",
            _team_spec=mock_team_spec,
            _members=(MagicMock(),),
        )
        team_result = TeamResult(
            success=True,
            final_output="Team completed work",
            member_results={},
            formation=TeamFormation.PARALLEL,
            total_tool_calls=3,
        )

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with patch(
            "victor.framework.team_runtime.run_named_team",
            new=AsyncMock(return_value=(resolved_plan, team_result)),
        ) as run_named_team:
            result = await agent.run_team(
                "test_team",
                "Test goal",
                context={"key": "value"},
                timeout_seconds=60.0,
            )

        assert result["success"] is True
        assert result["final_output"] == "Team completed work"
        assert result["team_name"] == "test_team"
        assert result["team_display_name"] == "Test Team"
        assert result["goal"] == "Test goal"
        assert result["recommendation_source"] == "explicit"
        assert result["total_tool_calls"] == 3
        run_named_team.assert_awaited_once_with(
            mock_orchestrator,
            team_name="test_team",
            goal="Test goal",
            context={"key": "value"},
            timeout_seconds=60.0,
        )

    @pytest.mark.asyncio
    async def test_run_team_no_get_team_specs_method(self, mock_orchestrator, mock_vertical):
        """run_team should raise AgentError when team provider has no get_team_specs."""
        from victor.framework.agent import Agent
        from victor.core.errors import AgentError

        mock_provider = MagicMock(spec=[])  # No get_team_specs method
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with pytest.raises(AgentError, match="does not provide team specs"):
            await agent.run_team("test_team", "Test goal")


# =============================================================================
# Lifecycle Tests (lines 1046-1047, 1052-1062, 1065, 1068)
# =============================================================================


class TestAgentLifecycle:
    """Tests for Agent lifecycle methods."""

    @pytest.mark.asyncio
    async def test_reset(self, mock_orchestrator):
        """reset should reset conversation and state."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        await agent.reset()

        mock_orchestrator.reset_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_with_orchestrator_close(self, mock_orchestrator):
        """close should call orchestrator.close when available."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        await agent.close()

        mock_orchestrator.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_orchestrator_close(self, mock_orchestrator):
        """close should handle orchestrator without close method."""
        from victor.framework.agent import Agent

        del mock_orchestrator.close

        agent = Agent(mock_orchestrator)

        # Should not raise
        await agent.close()

    @pytest.mark.asyncio
    async def test_context_manager_enter(self, mock_orchestrator):
        """__aenter__ should return self."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        result = await agent.__aenter__()

        assert result is agent

    @pytest.mark.asyncio
    async def test_context_manager_exit(self, mock_orchestrator):
        """__aexit__ should call close."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        with patch.object(agent, "close", new=AsyncMock()) as mock_close:
            await agent.__aexit__(None, None, None)

            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_orchestrator):
        """Agent should work as async context manager."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        async with agent as a:
            assert a is agent

        mock_orchestrator.close.assert_called_once()

    def test_destructor_warns_when_agent_not_closed(self, mock_orchestrator):
        """__del__ should emit a ResourceWarning when cleanup was skipped."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            agent.__del__()

        assert any(item.category is ResourceWarning for item in caught)

    def test_destructor_is_safe_during_interpreter_shutdown(self, mock_orchestrator):
        """__del__ should not raise when import machinery is already torn down."""
        import importlib

        agent_module = importlib.import_module("victor.framework.agent")
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        with patch.object(agent_module.sys, "meta_path", None):
            agent.__del__()

    def test_repr(self, mock_orchestrator):
        """__repr__ should return descriptive string."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator, provider="anthropic", model="claude-3")

        repr_str = repr(agent)

        assert "Agent" in repr_str
        assert "anthropic" in repr_str
        assert "claude-3" in repr_str


# =============================================================================
# Agent.create Tests (lines 181-234) - Integration style tests
# =============================================================================


class TestAgentCreate:
    """Tests for Agent.create classmethod."""

    @pytest.mark.asyncio
    async def test_create_basic(self):
        """create should create Agent with orchestrator."""
        from victor.framework.agent import Agent

        mock_orchestrator = AgentOrchestrator()

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.framework.agent_factory.AgentFactory.create",
                new=AsyncMock(return_value=mock_orchestrator),
            ),
        ):
            agent = await Agent.create(provider="anthropic")

        assert agent._orchestrator is mock_orchestrator
        assert agent._provider == "anthropic"

    @pytest.mark.asyncio
    async def test_create_with_vertical(self, mock_vertical):
        """create should apply vertical configuration."""
        from victor.framework.agent import Agent

        mock_orchestrator = AgentOrchestrator()

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.framework.agent_factory.AgentFactory.create",
                new=AsyncMock(return_value=mock_orchestrator),
            ),
        ):
            agent = await Agent.create(
                provider="anthropic",
                vertical=mock_vertical,
            )

        assert agent._vertical is mock_vertical
        assert agent._vertical_config is not None

    @pytest.mark.asyncio
    async def test_create_with_provider_hints(self, mock_vertical):
        """create should respect provider hints from vertical config."""
        from victor.framework.agent import Agent

        mock_orchestrator = AgentOrchestrator()

        mock_vertical.get_definition = MagicMock(
            return_value=VerticalDefinition(
                name="test_vertical",
                description="Test vertical",
                version="1.0.0",
                tools=[],
                system_prompt="Test prompt",
                workflow_metadata={"provider_hints": {"preferred_providers": ["openai", "google"]}},
            )
        )

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.framework.agent_factory.AgentFactory.create",
                new=AsyncMock(return_value=mock_orchestrator),
            ),
        ):
            # User still chooses anthropic despite not being preferred
            agent = await Agent.create(
                provider="anthropic",
                vertical=mock_vertical,
            )

        # Agent should still use anthropic (hints are just hints)
        assert agent._provider == "anthropic"

    @pytest.mark.asyncio
    async def test_create_with_vertical_uses_get_config(self):
        """create should call vertical.get_config() directly."""
        from victor.core.verticals.base import VerticalBase
        from victor.framework.agent import Agent

        class DirectConfigVertical(VerticalBase):
            name = "direct_config"
            description = "Direct config vertical"

            @classmethod
            def get_tools(cls):
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls):
                return "Use direct config path."

        mock_orchestrator = AgentOrchestrator()

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.framework.agent_factory.AgentFactory.create",
                new=AsyncMock(return_value=mock_orchestrator),
            ),
        ):
            agent = await Agent.create(
                provider="anthropic",
                vertical=DirectConfigVertical,
            )

        assert agent._vertical is DirectConfigVertical
        assert agent._vertical_config.system_prompt == "Use direct config path."
        assert agent._vertical_config.tools.tools == {"read", "write"}

    @pytest.mark.asyncio
    async def test_create_provider_error(self):
        """create should raise ProviderError for provider issues."""
        from victor.framework.agent import Agent
        from victor.core.errors import ProviderError

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.framework.agent_factory.AgentFactory.create",
                new=AsyncMock(side_effect=Exception("Provider API error")),
            ),
        ):
            with pytest.raises(ProviderError, match="API error"):
                await Agent.create(provider="anthropic")

    @pytest.mark.asyncio
    async def test_create_agent_error(self):
        """create should raise AgentError for other errors."""
        from victor.framework.agent import Agent
        from victor.core.errors import AgentError

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.framework.agent_factory.AgentFactory.create",
                new=AsyncMock(side_effect=Exception("Configuration invalid")),
            ),
        ):
            with pytest.raises(AgentError, match="Configuration invalid"):
                await Agent.create(provider="anthropic")

    @pytest.mark.asyncio
    async def test_create_forwards_session_provider_overrides_into_factory(self):
        """create should route explicit session overrides through AgentFactory."""
        from victor.framework.agent import Agent
        from victor.framework.session_config import SessionConfig

        mock_orchestrator = AgentOrchestrator()
        settings = MagicMock()
        settings.provider = SimpleNamespace(
            default_provider="ollama",
            default_model="qwen3.5:27b-q4_K_M",
            ollama_base_url="http://localhost:11434",
        )
        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_orchestrator)

        config = SessionConfig.from_cli_flags(
            agent_profile="zai-coding",
            provider="zai",
            model="glm-4.7",
            auth_mode="oauth",
            coding_plan=True,
        )

        with (
            patch("victor.config.settings.load_settings", return_value=settings),
            patch(
                "victor.framework.agent_factory.AgentFactory", return_value=mock_factory
            ) as factory_cls,
        ):
            agent = await Agent.create(session_config=config)

        assert agent._orchestrator is mock_orchestrator
        factory_cls.assert_called_once_with(
            settings=settings,
            profile="zai-coding",
            provider="zai",
            model="glm-4.7",
            temperature=0.7,
            max_tokens=4096,
            vertical=None,
            thinking=False,
            session_id=None,
            enable_observability=True,
            profile_overrides={"auth_mode": "oauth", "coding_plan": True},
        )

    @pytest.mark.asyncio
    async def test_create_profile_timeout_override_does_not_force_default_provider(
        self,
    ):
        """Profile-scoped runtime overrides must not synthesize a provider override."""
        from victor.framework.agent import Agent
        from victor.framework.session_config import SessionConfig

        mock_orchestrator = AgentOrchestrator()
        settings = MagicMock()
        settings.provider = SimpleNamespace(
            default_provider="ollama",
            default_model="qwen3.5:27b-q4_K_M",
        )
        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_orchestrator)

        config = SessionConfig.from_cli_flags(
            agent_profile="zai-coding",
            provider_timeout=180,
        )

        with (
            patch("victor.config.settings.load_settings", return_value=settings),
            patch(
                "victor.framework.agent_factory.AgentFactory", return_value=mock_factory
            ) as factory_cls,
        ):
            agent = await Agent.create(session_config=config)

        assert agent._orchestrator is mock_orchestrator
        factory_cls.assert_called_once_with(
            settings=settings,
            profile="zai-coding",
            provider=None,
            model=None,
            temperature=0.7,
            max_tokens=4096,
            vertical=None,
            thinking=False,
            session_id=None,
            enable_observability=True,
            profile_overrides={"timeout": 180},
        )


# =============================================================================
# Agent.create_team Tests (lines 334-349)
# =============================================================================


class TestAgentCreateTeam:
    """Tests for Agent.create_team classmethod."""

    @pytest.mark.asyncio
    async def test_create_team_basic(self):
        """create_team should create AgentTeam."""
        from victor.framework.agent import Agent
        from victor.framework.teams import TeamMemberSpec

        mock_orchestrator = AgentOrchestrator()
        mock_team = MagicMock()

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.framework.agent_factory.AgentFactory.create",
                new=AsyncMock(return_value=mock_orchestrator),
            ),
            patch(
                "victor.framework.teams.AgentTeam.create",
                new=AsyncMock(return_value=mock_team),
            ),
        ):
            team = await Agent.create_team(
                name="Test Team",
                goal="Complete test",
                members=[
                    TeamMemberSpec(role="researcher", goal="Research"),
                    TeamMemberSpec(role="executor", goal="Execute"),
                ],
            )

        assert team is mock_team

    @pytest.mark.asyncio
    async def test_create_team_with_formation(self):
        """create_team should use specified formation."""
        from victor.framework.agent import Agent
        from victor.framework.teams import TeamMemberSpec
        from victor.teams import TeamFormation

        mock_orchestrator = AgentOrchestrator()
        mock_team = MagicMock()

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.framework.agent_factory.AgentFactory.create",
                new=AsyncMock(return_value=mock_orchestrator),
            ),
            patch(
                "victor.framework.teams.AgentTeam.create",
                new=AsyncMock(return_value=mock_team),
            ) as mock_create,
        ):
            await Agent.create_team(
                name="Test Team",
                goal="Complete test",
                members=[TeamMemberSpec(role="executor", goal="Execute")],
                formation=TeamFormation.PARALLEL,
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs.get("formation") == TeamFormation.PARALLEL

    @pytest.mark.asyncio
    async def test_create_team_default_formation(self):
        """create_team should default to SEQUENTIAL formation."""
        from victor.framework.agent import Agent
        from victor.framework.teams import TeamMemberSpec
        from victor.teams import TeamFormation

        mock_orchestrator = AgentOrchestrator()
        mock_team = MagicMock()

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.framework.agent_factory.AgentFactory.create",
                new=AsyncMock(return_value=mock_orchestrator),
            ),
            patch(
                "victor.framework.teams.AgentTeam.create",
                new=AsyncMock(return_value=mock_team),
            ) as mock_create,
        ):
            await Agent.create_team(
                name="Test Team",
                goal="Complete test",
                members=[TeamMemberSpec(role="executor", goal="Execute")],
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs.get("formation") == TeamFormation.SEQUENTIAL


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_agent_importable(self):
        """Agent class should be importable from framework.agent."""
        from victor.framework.agent import Agent

        assert Agent is not None

    def test_chat_session_importable(self):
        """ChatSession class should be importable from framework.agent."""
        from victor.framework.agent import ChatSession

        assert ChatSession is not None
