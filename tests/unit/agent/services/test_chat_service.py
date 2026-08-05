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

"""Tests for ChatService implementation."""

import asyncio
import inspect
from unittest import mock

import pytest

from victor.agent.services.chat_service import ChatService, ChatServiceConfig
from victor.providers.base import CompletionResponse, StreamChunk

# =============================================================================
# Mock Dependencies
# =============================================================================


class MockProviderService:
    """Mock provider service."""

    def __init__(self):
        self.healthy = True
        self.current_provider = self

    def is_healthy(self):
        return self.healthy

    def get_current_provider(self):
        return self

    async def chat_completion(self, messages, **kwargs):
        from victor.providers.base import CompletionResponse

        return CompletionResponse(
            content="Mock response",
            stop_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

    async def stream_completion(self, messages, **kwargs):
        from victor.providers.base import StreamChunk

        yield StreamChunk(content="Mock chunk")

    async def check_provider_health(self, provider=None):
        return True


class MockToolService:
    """Mock tool service."""

    def __init__(self):
        self.healthy = True
        self.tools_executed = []
        self.turn_resets = 0

    def is_healthy(self):
        return self.healthy

    def start_new_turn(self):
        self.turn_resets += 1

    def reset_tool_budget(self):
        self.turn_resets += 1

    async def execute_tool(self, tool_name, arguments):
        from victor.tools.base import ToolResult

        self.tools_executed.append((tool_name, arguments))
        return ToolResult(success=True, output=f"Executed {tool_name}")


class MockContextService:
    """Mock context service."""

    def __init__(self):
        self.healthy = True
        self.messages = []
        self.overflow = False

    def is_healthy(self):
        return self.healthy

    async def check_context_overflow(self):
        return self.overflow

    async def compact_context(self):
        self.messages = self.messages[-6:]

    def get_messages(self):
        return self.messages

    def add_message(self, message):
        self.messages.append(message)

    def clear_messages(self, retain_system=True):
        self.messages.clear()


class MockRecoveryService:
    """Mock recovery service."""

    def __init__(self):
        self.recovery_attempts = 0

    def is_healthy(self):
        return True

    async def execute_recovery(self, context):
        self.recovery_attempts += 1
        return False  # Recovery fails by default


class MockConversationController:
    """Mock conversation controller."""

    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class MockStreamingCoordinator:
    """Mock streaming coordinator."""

    pass


class FailingMemoryManager:
    def add_message(self, **kwargs):
        raise RuntimeError("database is locked")


@pytest.mark.asyncio
async def test_persist_message_background_failure_is_consumed():
    """Background persistence failures should not surface as event-loop errors."""
    loop = asyncio.get_running_loop()
    exceptions = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: exceptions.append(context))

    try:
        ChatService.persist_message(
            "assistant",
            "hello",
            FailingMemoryManager(),
            "session-1",
            usage_logger=None,
        )
        await asyncio.sleep(0.05)
    finally:
        loop.set_exception_handler(previous_handler)

    assert exceptions == []


# =============================================================================
# Base Test Class with Helper Methods
# =============================================================================


class BaseChatServiceTest:
    """Base class for ChatService tests with helper methods."""

    def _create_test_service(self, config=None):
        """Create a test ChatService with mocked dependencies."""
        if config is None:
            config = ChatServiceConfig()

        provider = MockProviderService()
        tools = MockToolService()
        context = MockContextService()
        recovery = MockRecoveryService()
        conversation = MockConversationController()
        streaming = MockStreamingCoordinator()

        return ChatService(
            config=config,
            provider_service=provider,
            tool_service=tools,
            context_service=context,
            recovery_service=recovery,
            conversation_controller=conversation,
            streaming_coordinator=streaming,
        )


# =============================================================================
# Tests
# =============================================================================


class TestChatServiceConfig(BaseChatServiceTest):
    """Tests for ChatServiceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChatServiceConfig()

        assert config.max_iterations == 200
        assert config.max_continuation_prompts == 3
        assert config.stream_chunk_size == 100
        assert config.enable_response_caching is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChatServiceConfig(
            max_iterations=100,
            max_continuation_prompts=5,
            stream_chunk_size=200,
            enable_response_caching=False,
        )

        assert config.max_iterations == 100
        assert config.max_continuation_prompts == 5
        assert config.stream_chunk_size == 200
        assert config.enable_response_caching is False


class TestChatServiceInit(BaseChatServiceTest):
    """Tests for ChatService initialization."""

    def test_initialization(self):
        """Test service initialization with all dependencies."""
        config = ChatServiceConfig()
        provider = MockProviderService()
        tools = MockToolService()
        context = MockContextService()
        recovery = MockRecoveryService()
        conversation = MockConversationController()
        streaming = MockStreamingCoordinator()

        service = ChatService(
            config=config,
            provider_service=provider,
            tool_service=tools,
            context_service=context,
            recovery_service=recovery,
            conversation_controller=conversation,
            streaming_coordinator=streaming,
        )

        assert service._config is config
        assert service._provider is provider
        assert service._tools is tools
        assert service._context is context
        assert service._recovery is recovery
        assert service._conversation is conversation
        assert service._streaming is streaming


class TestChatServiceHealth(BaseChatServiceTest):
    """Tests for ChatService health checks."""

    def test_is_healthy_all_services_healthy(self):
        """Test health check when all services are healthy."""
        service = self._create_test_service()
        assert service.is_healthy() is True

    def test_is_healthy_provider_unhealthy(self):
        """Test health check when provider is unhealthy."""
        service = self._create_test_service()
        service._provider.healthy = False
        assert service.is_healthy() is False

    def test_is_healthy_tools_unhealthy(self):
        """Test health check when tools are unhealthy."""
        service = self._create_test_service()
        service._tools.healthy = False
        assert service.is_healthy() is False

    def test_is_healthy_context_unhealthy(self):
        """Test health check when context is unhealthy."""
        service = self._create_test_service()
        service._context.healthy = False
        assert service.is_healthy() is False


class TestChatServiceReset(BaseChatServiceTest):
    """Tests for ChatService.reset_conversation method."""

    def test_reset_conversation(self):
        """Test conversation reset."""
        service = self._create_test_service()

        # Add some context
        service._context.messages = [{"role": "user", "content": "test"}]

        service.reset_conversation()

        assert service._context.messages == []
        assert service._conversation.reset_count == 1


class TestChatServiceDictBasedContext(BaseChatServiceTest):
    """Bound runtimes should support dict-based ContextServiceProtocol writes."""

    @pytest.mark.asyncio
    async def test_chat_records_user_and_assistant_messages_in_dict_based_context(self):
        service = self._create_test_service()
        response = CompletionResponse(content="done", role="assistant", stop_reason="stop")

        class _TurnExecutor:
            async def execute_agentic_loop(self, user_message):
                service._context.add_message({"role": "user", "content": user_message})
                service._context.add_message({"role": "assistant", "content": response.content})
                return response

        service.bind_runtime_components(turn_executor=_TurnExecutor())

        result = await service.chat("hello")

        assert result is response
        assert [message["role"] for message in service._context.messages] == [
            "user",
            "assistant",
        ]
        assert [message["content"] for message in service._context.messages] == [
            "hello",
            "done",
        ]

    @pytest.mark.asyncio
    async def test_chat_passes_runtime_context_overrides_to_bound_executor(self):
        service = self._create_test_service()
        response = CompletionResponse(content="done", role="assistant", stop_reason="stop")
        observed = {}

        class _TurnExecutor:
            async def execute_agentic_loop(self, user_message, runtime_context_overrides=None):
                observed["user_message"] = user_message
                observed["runtime_context_overrides"] = runtime_context_overrides
                return response

        service.bind_runtime_components(turn_executor=_TurnExecutor())
        runtime_context_overrides = {
            "prompt_overlays": [
                {
                    "name": "test.scoped_prompt",
                    "content": "Scoped prompt",
                    "placement": "turn_prefix",
                }
            ]
        }

        result = await service.chat(
            "hello",
            runtime_context_overrides=runtime_context_overrides,
        )

        assert result is response
        assert observed == {
            "user_message": "hello",
            "runtime_context_overrides": runtime_context_overrides,
        }

    def test_add_tool_result_supports_dict_based_context_protocol(self):
        service = self._create_test_service()
        result = mock.MagicMock(output="tool output", error=None)

        service._add_tool_result_to_context(
            "read",
            result,
            tool_call_id="call-1",
            formatted_content="formatted output",
        )

        tool_message = service._context.messages[-1]
        assert tool_message["role"] == "tool"
        assert tool_message["content"] == "formatted output"
        assert tool_message["tool_call_id"] == "call-1"
        assert tool_message["name"] == "read"


class TestChatServiceControllerBackedContext(BaseChatServiceTest):
    """Bound runtimes should share conversation state with controller-backed context."""

    def _create_controller_backed_service(self):
        from victor.agent.conversation.controller import ConversationController
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        config = ChatServiceConfig()
        provider = MockProviderService()
        tools = MockToolService()
        controller = ConversationController()
        context = ContextServiceAdapter(controller)
        recovery = MockRecoveryService()
        streaming = MockStreamingCoordinator()

        service = ChatService(
            config=config,
            provider_service=provider,
            tool_service=tools,
            context_service=context,
            recovery_service=recovery,
            conversation_controller=controller,
            streaming_coordinator=streaming,
        )
        return service, controller

    @pytest.mark.asyncio
    async def test_chat_records_user_and_assistant_messages_in_controller_backed_context(
        self,
    ):
        service, controller = self._create_controller_backed_service()
        response = CompletionResponse(content="done", role="assistant", stop_reason="stop")

        class _TurnExecutor:
            async def execute_agentic_loop(self, user_message):
                controller.add_message("user", user_message)
                controller.add_message("assistant", response.content)
                return response

        service.bind_runtime_components(turn_executor=_TurnExecutor())

        result = await service.chat("hello")

        assert result is response
        assert [message.role for message in controller.messages] == [
            "user",
            "assistant",
        ]
        assert [message.content for message in controller.messages] == ["hello", "done"]

    def test_add_tool_result_supports_controller_backed_context_adapter(self):
        service, controller = self._create_controller_backed_service()
        result = mock.MagicMock(output="tool output", error=None)

        service._add_tool_result_to_context(
            "read",
            result,
            tool_call_id="call-1",
            formatted_content="formatted output",
        )

        tool_message = controller.messages[-1]
        assert tool_message.role == "tool"
        assert tool_message.content == "formatted output"
        assert tool_message.tool_call_id == "call-1"
        assert tool_message.name == "read"

    def test_reset_conversation_preserves_system_prompt_without_double_reset(self):
        service, controller = self._create_controller_backed_service()
        controller.set_system_prompt("system prompt")
        controller.ensure_system_message()
        controller.add_message("user", "hello")
        controller.reset = mock.MagicMock(wraps=controller.reset)

        service.reset_conversation()

        assert controller.reset.call_count == 1
        assert [message.role for message in controller.messages] == ["system"]
        assert [message.content for message in controller.messages] == ["system prompt"]

    @pytest.mark.asyncio
    async def test_chat_replenishes_tool_budget_per_prompt_and_preserves_usage_stats(
        self,
    ):
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig
        from victor.tools.base import ToolResult

        tool_executor = mock.Mock()
        tool_executor.execute = mock.AsyncMock(return_value=ToolResult(success=True, output="ok"))
        tool_service = ToolService(
            config=ToolServiceConfig(default_tool_budget=1, enable_caching=False),
            tool_selector=mock.Mock(),
            tool_executor=tool_executor,
            tool_registrar=mock.Mock(),
        )

        service = ChatService(
            config=ChatServiceConfig(),
            provider_service=MockProviderService(),
            tool_service=tool_service,
            context_service=MockContextService(),
            recovery_service=MockRecoveryService(),
            conversation_controller=MockConversationController(),
            streaming_coordinator=MockStreamingCoordinator(),
        )

        class _TurnExecutor:
            async def execute_agentic_loop(self, user_message):
                await tool_service.execute_tool("read", {"prompt": user_message})
                return CompletionResponse(
                    content=f"done:{user_message}",
                    role="assistant",
                    stop_reason="stop",
                )

        service.bind_runtime_components(turn_executor=_TurnExecutor())

        first = await service.chat("first prompt")
        second = await service.chat("second prompt")

        assert first.content == "done:first prompt"
        assert second.content == "done:second prompt"
        assert tool_service.budget_used == 1
        assert tool_service.get_tool_usage_stats()["total_calls"] == 2
        assert tool_executor.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_stream_chat_replenishes_tool_budget_per_prompt_and_preserves_usage_stats(
        self,
    ):
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig
        from victor.tools.base import ToolResult

        tool_executor = mock.Mock()
        tool_executor.execute = mock.AsyncMock(return_value=ToolResult(success=True, output="ok"))
        tool_service = ToolService(
            config=ToolServiceConfig(default_tool_budget=1, enable_caching=False),
            tool_selector=mock.Mock(),
            tool_executor=tool_executor,
            tool_registrar=mock.Mock(),
        )

        service = ChatService(
            config=ChatServiceConfig(),
            provider_service=MockProviderService(),
            tool_service=tool_service,
            context_service=MockContextService(),
            recovery_service=MockRecoveryService(),
            conversation_controller=MockConversationController(),
            streaming_coordinator=MockStreamingCoordinator(),
        )

        async def _stream_handler(user_message, **kwargs):
            await tool_service.execute_tool("read", {"prompt": user_message})
            yield StreamChunk(content=f"done:{user_message}", is_final=True)

        service.bind_runtime_components(stream_chat_handler=_stream_handler)

        first = [chunk async for chunk in service.stream_chat("first prompt")]
        second = [chunk async for chunk in service.stream_chat("second prompt")]

        assert [chunk.content for chunk in first] == ["done:first prompt"]
        assert [chunk.content for chunk in second] == ["done:second prompt"]
        assert tool_service.budget_used == 1
        assert tool_service.get_tool_usage_stats()["total_calls"] == 2
        assert tool_executor.execute.await_count == 2

    def test_chat_service_declares_single_keyword_based_context_helpers(self):
        source = inspect.getsource(ChatService)

        assert source.count("def _add_user_message_to_context(") == 1
        assert source.count("def _add_assistant_message_to_context(") == 1
        assert "self._context.add_message(msg)" not in source


class TestChatServicePlanningStreaming(BaseChatServiceTest):
    """Tests for planning-aware streaming behavior."""

    @pytest.mark.asyncio
    async def test_stream_chat_auto_detects_plan_mode_requests(self):
        service = self._create_test_service()
        planning_response = CompletionResponse(content="planned", role="assistant")
        planning_handler = mock.AsyncMock(return_value=planning_response)
        stream_handler_called = False

        async def _stream_handler(user_message, **kwargs):
            nonlocal stream_handler_called
            stream_handler_called = True
            yield StreamChunk(content=f"stream:{user_message}", is_final=True)

        service.bind_runtime_components(
            planning_handler=planning_handler,
            stream_chat_handler=_stream_handler,
        )

        chunks = [
            chunk
            async for chunk in service.stream_chat("continue addressing the findings in plan mode")
        ]

        assert [chunk.content for chunk in chunks] == ["planned"]
        assert chunks[0].is_final is True
        planning_handler.assert_awaited_once_with("continue addressing the findings in plan mode")
        assert stream_handler_called is False

    @pytest.mark.asyncio
    async def test_stream_chat_preserves_live_streaming_for_auto_complexity_requests(
        self,
    ):
        service = self._create_test_service()
        planning_response = CompletionResponse(content="planned", role="assistant")
        planning_handler = mock.AsyncMock(return_value=planning_response)
        streamed_messages = []

        async def _stream_handler(user_message, **kwargs):
            streamed_messages.append(user_message)
            yield StreamChunk(content=f"stream:{user_message}", is_final=True)

        service.bind_runtime_components(
            planning_handler=planning_handler,
            stream_chat_handler=_stream_handler,
        )

        chunks = [
            chunk
            async for chunk in service.stream_chat(
                "Review the graph related codebase and identify structural design weaknesses."
            )
        ]

        assert [chunk.content for chunk in chunks] == [
            "stream:Review the graph related codebase and identify structural design weaknesses."
        ]
        planning_handler.assert_not_awaited()
        assert streamed_messages == [
            "Review the graph related codebase and identify structural design weaknesses."
        ]

    @pytest.mark.asyncio
    async def test_stream_chat_auto_detects_checklist_first_requests(self):
        service = self._create_test_service()
        planning_response = CompletionResponse(content="checklist", role="assistant")
        planning_handler = mock.AsyncMock(return_value=planning_response)
        stream_handler_called = False

        async def _stream_handler(user_message, **kwargs):
            nonlocal stream_handler_called
            stream_handler_called = True
            yield StreamChunk(content=f"stream:{user_message}", is_final=True)

        service.bind_runtime_components(
            planning_handler=planning_handler,
            stream_chat_handler=_stream_handler,
        )

        prompt = (
            "Review this Rust workspace by creating a checklist first and showing me, "
            "then go module by module."
        )
        chunks = [chunk async for chunk in service.stream_chat(prompt)]

        assert [chunk.content for chunk in chunks] == ["checklist"]
        assert chunks[0].is_final is True
        planning_handler.assert_awaited_once_with(prompt)
        assert stream_handler_called is False

    @pytest.mark.asyncio
    async def test_chat_stream_forwards_use_planning_to_stream_runtime(self):
        service = self._create_test_service()
        planning_response = CompletionResponse(content="planned", role="assistant")
        planning_handler = mock.AsyncMock(return_value=planning_response)
        stream_handler_called = False

        async def _stream_handler(user_message, **kwargs):
            nonlocal stream_handler_called
            stream_handler_called = True
            yield StreamChunk(content=f"stream:{user_message}", is_final=True)

        service.bind_runtime_components(
            planning_handler=planning_handler,
            stream_chat_handler=_stream_handler,
        )

        response = await service.chat(
            "quick prompt",
            stream=True,
            use_planning=True,
        )

        assert response.content == "planned"
        planning_handler.assert_awaited_once_with("quick prompt")
        assert stream_handler_called is False


class TestChatServiceTaskReporting(BaseChatServiceTest):
    """Tests for canonical per-task reporting hooks."""

    @pytest.mark.asyncio
    async def test_chat_invokes_task_report_handlers_on_success(self):
        service = self._create_test_service()
        response = CompletionResponse(content="done", role="assistant", stop_reason="stop")
        started = []
        finished = []

        class _TurnExecutor:
            async def execute_agentic_loop(self, user_message):
                return response

        service.bind_runtime_components(
            turn_executor=_TurnExecutor(),
            task_report_start_handler=lambda user_message, **kwargs: started.append(
                (user_message, kwargs)
            ),
            task_report_finish_handler=lambda success, **kwargs: finished.append((success, kwargs)),
        )

        result = await service.chat("hello")

        assert result is response
        assert started == [("hello", {"stream": False, "metadata": {"use_planning": False}})]
        assert finished[0][0] is True
        assert finished[0][1]["user_message"] == "hello"
        assert finished[0][1]["response"] is response

    @pytest.mark.asyncio
    async def test_chat_invokes_task_report_finish_on_failure(self):
        service = self._create_test_service()
        started = []
        finished = []

        class _TurnExecutor:
            async def execute_agentic_loop(self, user_message):
                raise RuntimeError("boom")

        service.bind_runtime_components(
            turn_executor=_TurnExecutor(),
            task_report_start_handler=lambda user_message, **kwargs: started.append(
                (user_message, kwargs)
            ),
            task_report_finish_handler=lambda success, **kwargs: finished.append((success, kwargs)),
        )

        with pytest.raises(RuntimeError, match="boom"):
            await service.chat("explode")

        assert started == [("explode", {"stream": False, "metadata": {"use_planning": False}})]
        assert finished[0][0] is False
        assert str(finished[0][1]["error"]) == "boom"

    @pytest.mark.asyncio
    async def test_chat_marks_agentic_loop_metadata_failure_as_task_failure(self):
        service = self._create_test_service()
        response = CompletionResponse(content="stopped", role="assistant")
        response.metadata = {
            "agentic_loop_success": False,
            "agentic_loop_error": "Insufficient progress",
        }
        finished = []

        class _TurnExecutor:
            async def execute_agentic_loop(self, user_message):
                return response

        service.bind_runtime_components(
            turn_executor=_TurnExecutor(),
            task_report_finish_handler=lambda success, **kwargs: finished.append((success, kwargs)),
        )

        result = await service.chat("inventory")

        assert result is response
        assert finished[0][0] is False
        assert finished[0][1]["response"] is response

    @pytest.mark.asyncio
    async def test_stream_chat_invokes_task_report_handlers(self):
        service = self._create_test_service()
        started = []
        finished = []

        async def _stream_handler(user_message, **kwargs):
            yield StreamChunk(content=f"done:{user_message}", is_final=True)

        service.bind_runtime_components(
            stream_chat_handler=_stream_handler,
            task_report_start_handler=lambda user_message, **kwargs: started.append(
                (user_message, kwargs)
            ),
            task_report_finish_handler=lambda success, **kwargs: finished.append((success, kwargs)),
        )

        chunks = [chunk async for chunk in service.stream_chat("stream me")]

        assert [chunk.content for chunk in chunks] == ["done:stream me"]
        assert started == [("stream me", {"stream": True, "metadata": {"use_planning": None}})]
        assert finished[0][0] is True
        assert finished[0][1]["stream"] is True


class TestChatServiceTurnScope(BaseChatServiceTest):
    """Tests for canonical per-turn setup/teardown hooks."""

    @pytest.mark.asyncio
    async def test_chat_invokes_turn_scope_handlers_with_constraints(self):
        service = self._create_test_service()
        response = CompletionResponse(content="done", role="assistant", stop_reason="stop")
        events = []

        class _TurnExecutor:
            async def execute_agentic_loop(self, user_message):
                assert user_message == "hello"
                events.append(("execute", user_message))
                return response

        async def _turn_setup(user_message, **kwargs):
            events.append(("setup", user_message, kwargs))

        async def _turn_teardown(user_message, **kwargs):
            events.append(("teardown", user_message, kwargs))

        service.bind_runtime_components(
            turn_executor=_TurnExecutor(),
            turn_setup_handler=_turn_setup,
            turn_teardown_handler=_turn_teardown,
        )

        constraints = object()
        result = await service.chat("hello", constraints=constraints, vertical="review")

        assert result is response
        assert events == [
            (
                "setup",
                "hello",
                {"stream": False, "constraints": constraints, "vertical": "review"},
            ),
            ("execute", "hello"),
            (
                "teardown",
                "hello",
                {"stream": False, "constraints": constraints, "vertical": "review"},
            ),
        ]

    @pytest.mark.asyncio
    async def test_stream_chat_invokes_turn_scope_handlers_and_passes_kwargs(self):
        service = self._create_test_service()
        events = []

        async def _stream_handler(user_message, **kwargs):
            events.append(("execute", user_message, kwargs))
            yield StreamChunk(content="chunk", is_final=True)

        async def _turn_setup(user_message, **kwargs):
            events.append(("setup", user_message, kwargs))

        async def _turn_teardown(user_message, **kwargs):
            events.append(("teardown", user_message, kwargs))

        service.bind_runtime_components(
            stream_chat_handler=_stream_handler,
            turn_setup_handler=_turn_setup,
            turn_teardown_handler=_turn_teardown,
        )

        constraints = object()
        chunks = [
            chunk
            async for chunk in service.stream_chat(
                "stream me",
                constraints=constraints,
                vertical="coding",
                mode="compat",
            )
        ]

        assert [chunk.content for chunk in chunks] == ["chunk"]
        assert events == [
            (
                "setup",
                "stream me",
                {"stream": True, "constraints": constraints, "vertical": "coding"},
            ),
            ("execute", "stream me", {"mode": "compat"}),
            (
                "teardown",
                "stream me",
                {"stream": True, "constraints": constraints, "vertical": "coding"},
            ),
        ]

    @pytest.mark.asyncio
    async def test_chat_turn_teardown_runs_after_execution_failure(self):
        service = self._create_test_service()
        events = []

        class _TurnExecutor:
            async def execute_agentic_loop(self, user_message):
                events.append(("execute", user_message))
                raise RuntimeError("boom")

        async def _turn_setup(user_message, **kwargs):
            events.append(("setup", user_message, kwargs))

        async def _turn_teardown(user_message, **kwargs):
            events.append(("teardown", user_message, kwargs))

        service.bind_runtime_components(
            turn_executor=_TurnExecutor(),
            turn_setup_handler=_turn_setup,
            turn_teardown_handler=_turn_teardown,
        )

        constraints = object()
        with pytest.raises(RuntimeError, match="boom"):
            await service.chat("hello", constraints=constraints)

        assert events == [
            (
                "setup",
                "hello",
                {"stream": False, "constraints": constraints, "vertical": None},
            ),
            ("execute", "hello"),
            (
                "teardown",
                "hello",
                {"stream": False, "constraints": constraints, "vertical": None},
            ),
        ]


class TestStreamingExecutorIntegration:
    """Tests verifying the canonical streaming executor owns loop behavior."""

    def test_pipeline_accepts_perception(self):
        """StreamingChatExecutor constructor accepts perception parameter."""
        from unittest.mock import MagicMock
        from victor.agent.services.chat_stream_executor import StreamingChatExecutor

        mock_coord = MagicMock()
        mock_perception = MagicMock()
        pipeline = StreamingChatExecutor(mock_coord, perception=mock_perception)
        assert pipeline._perception is mock_perception

    def test_pipeline_accepts_fulfillment(self):
        """StreamingChatExecutor constructor accepts fulfillment parameter."""
        from unittest.mock import MagicMock
        from victor.agent.services.chat_stream_executor import StreamingChatExecutor

        mock_coord = MagicMock()
        mock_fulfillment = MagicMock()
        pipeline = StreamingChatExecutor(mock_coord, fulfillment=mock_fulfillment)
        assert pipeline._fulfillment is mock_fulfillment

    def test_pipeline_has_progress_tracking(self):
        """Progress/plateau tracking is the shared PlateauDetector (consolidated from the
        per-executor _progress_scores list into victor.agent.turn_policy)."""
        from victor.agent.turn_policy import PlateauDetector

        detector = PlateauDetector()
        detector.record(0, 100)
        detector.record(0, 100)
        # Three unproductive low-score turns -> productivity-weighted plateau detected.
        assert detector.record(0, 100).is_plateau

    def test_factory_passes_components(self):
        """create_streaming_chat_executor passes perception and fulfillment."""
        from unittest.mock import MagicMock
        from victor.agent.services.chat_stream_executor import (
            create_streaming_chat_executor,
        )

        mock_perception = MagicMock()
        mock_fulfillment = MagicMock()
        pipeline = create_streaming_chat_executor(
            MagicMock(), perception=mock_perception, fulfillment=mock_fulfillment
        )
        assert pipeline._perception is mock_perception
        assert pipeline._fulfillment is mock_fulfillment


# =============================================================================
# Test Helpers
# =============================================================================
class TestToolService:
    """Tests for ToolService implementation."""

    def test_initialization(self):
        """Test service initialization."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        config = ToolServiceConfig()
        selector = mock.Mock()
        executor = mock.Mock()
        registrar = mock.Mock()

        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        assert service._config is config
        assert service._selector is selector
        assert service._executor is executor
        assert service._registrar is registrar

    @pytest.mark.asyncio
    async def test_select_tools(self):
        """Test tool selection."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        config = ToolServiceConfig()
        selector = mock.Mock()
        selector.select = mock.AsyncMock(return_value=["tool1", "tool2"])
        executor = mock.Mock()
        registrar = mock.Mock()

        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        context = mock.Mock()
        tools = await service.select_tools(context, max_tools=10)

        assert tools == ["tool1", "tool2"]
        selector.select.assert_called_once_with(context, 10)

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig
        from victor.tools.base import ToolResult

        config = ToolServiceConfig()
        selector = mock.Mock()
        executor = mock.Mock()
        executor.execute = mock.AsyncMock(return_value=ToolResult(success=True, output="Success"))
        registrar = mock.Mock()

        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        result = await service.execute_tool("test_tool", {"arg": "value"})

        assert result.success is True
        assert result.output == "Success"
        executor.execute.assert_called_once_with("test_tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_tool_budget(self):
        """Test tool budget management."""
        from victor.agent.services.tool_service import (
            ToolService,
            ToolServiceConfig,
            ToolBudgetExceededError,
        )
        from victor.tools.base import ToolResult

        config = ToolServiceConfig(default_tool_budget=2)
        selector = mock.Mock()
        executor = mock.Mock()
        executor.execute = mock.AsyncMock(return_value=ToolResult(success=True, output="Success"))
        registrar = mock.Mock()

        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        # Initial budget
        assert service.get_tool_budget() == 2

        # Execute tool
        await service.execute_tool("test", {})

        # Budget decreased
        assert service.get_tool_budget() == 1

        # Execute another tool
        await service.execute_tool("test", {})

        # Budget exhausted
        assert service.get_tool_budget() == 0

        # Next execution should fail
        with pytest.raises(ToolBudgetExceededError):
            await service.execute_tool("test", {})

    def test_get_tool_usage_stats(self):
        """Test getting tool usage statistics."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        config = ToolServiceConfig()
        selector = mock.Mock()
        executor = mock.Mock()
        registrar = mock.Mock()

        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        # Track some usage
        service._track_tool_usage("read", True)
        service._track_tool_usage("read", True)
        service._track_tool_usage("write", True)
        service._track_tool_usage("error:grep", False)

        stats = service.get_tool_usage_stats()

        assert stats["total_calls"] == 4
        assert stats["successful_calls"] == 3
        assert stats["failed_calls"] == 1
        assert stats["success_rate"] == 0.75
