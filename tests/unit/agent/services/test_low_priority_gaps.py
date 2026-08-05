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

"""Tests for LOW priority feature gap implementations.

Tests the 10 LOW priority methods added to achieve 100% feature parity:
- ToolService: normalize_tool_arguments, set_argument_normalizer (2)
- ChatService: 5 context/error methods (5)
- SessionService: 3 checkpoint/token methods (3)
- ProviderService: test_provider (1)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Dict, Any

# =============================================================================
# ToolService Tests (2 methods)
# =============================================================================


class TestToolServiceArgumentNormalization:
    """Test ToolService argument normalization methods."""

    def test_set_argument_normalizer(self):
        """Test setting argument normalizer."""
        from victor.agent.services.tool_service import ToolService

        # Create service
        service = _make_tool_service()

        # Create mock normalizer
        normalizer = Mock()
        normalizer.__class__.__name__ = "TestNormalizer"

        # Set normalizer
        service.set_argument_normalizer(normalizer)

        # Verify normalizer was set
        assert service._argument_normalizer == normalizer

    def test_normalize_tool_arguments_with_normalizer(self):
        """Test argument normalization with normalizer set."""
        from victor.agent.services.tool_service import ToolService

        # Create service
        service = _make_tool_service()

        # Create mock normalizer that returns normalized args
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(return_value=({"foo": "bar"}, "normalized"))
        service.set_argument_normalizer(normalizer)

        # Normalize arguments
        tool_args = {"malformed": "json"}
        result, method = service.normalize_tool_arguments(tool_args, "test_tool")

        # Verify normalization was called
        normalizer.normalize_arguments.assert_called_once_with(tool_args, "test_tool")
        assert result == {"foo": "bar"}
        assert method == "normalized"

    def test_normalize_tool_arguments_without_normalizer(self):
        """Test argument normalization without normalizer (direct path)."""
        from victor.agent.services.tool_service import ToolService

        # Create service
        service = _make_tool_service()

        # Normalize arguments (no normalizer set)
        tool_args = {"foo": "bar"}
        result, method = service.normalize_tool_arguments(tool_args, "test_tool")

        # Verify direct return
        assert result == tool_args
        assert method == "direct"

    def test_normalize_tool_arguments_with_error(self):
        """Test argument normalization when normalizer raises error."""
        from victor.agent.services.tool_service import ToolService

        # Create service
        service = _make_tool_service()

        # Create mock normalizer that raises error
        normalizer = Mock()
        normalizer.normalize_arguments = Mock(side_effect=Exception("Normalization failed"))
        service.set_argument_normalizer(normalizer)

        # Normalize arguments (should fall back to direct)
        tool_args = {"foo": "bar"}
        result, method = service.normalize_tool_arguments(tool_args, "test_tool")

        # Verify fallback to direct
        assert result == tool_args
        assert method == "direct"


# =============================================================================
# ChatService Tests (5 methods)
# =============================================================================

_UNSET = object()


def _make_chat_service(provider=None, recovery=None, context=None, tools=_UNSET):
    """Helper to build a ChatService with minimal mocks for unit tests."""
    from victor.agent.services.chat_service import ChatService, ChatServiceConfig

    tool_service = MagicMock() if tools is _UNSET else tools
    return ChatService(
        config=ChatServiceConfig(),
        provider_service=provider or MagicMock(),
        tool_service=tool_service,
        context_service=context,
        recovery_service=recovery,
        conversation_controller=MagicMock(),
        streaming_coordinator=MagicMock(),
    )


def _make_tool_service():
    """Helper to build a ToolService with minimal mocks for unit tests."""
    from victor.agent.services.tool_service import ToolService, ToolServiceConfig

    return ToolService(
        config=ToolServiceConfig(),
        tool_selector=MagicMock(),
        tool_executor=MagicMock(),
        tool_registrar=MagicMock(),
    )


class TestChatServiceContextMethods:
    """Test ChatService context management methods."""

    def test_add_user_message_to_context(self):
        """Test adding user message to context."""
        from victor.agent.services.chat_service import ChatService

        # Create service
        provider = Mock()
        recovery = Mock()
        context = Mock()
        context.add_message = Mock()
        service = _make_chat_service(provider=provider, recovery=recovery, context=context)

        # Add user message
        service._add_user_message_to_context("Hello, world!", {"test": "metadata"})

        # Verify context was called
        context.add_message.assert_called_once_with(
            role="user",
            content="Hello, world!",
            metadata={"test": "metadata"},
        )

    def test_add_user_message_to_context_without_context(self):
        """Test adding user message when context service unavailable."""
        from victor.agent.services.chat_service import ChatService

        # Create service without context
        provider = Mock()
        service = _make_chat_service(provider=provider, recovery=None, context=None)

        # Should not raise error
        service._add_user_message_to_context("Hello, world!")

    def test_add_user_message_to_context_with_error(self):
        """Test adding user message when context service raises error."""
        from victor.agent.services.chat_service import ChatService

        # Create service with failing context
        provider = Mock()
        context = Mock()
        context.add_message = Mock(side_effect=Exception("Context error"))
        service = _make_chat_service(provider=provider, recovery=None, context=context)

        # Should not raise error (caught and logged)
        service._add_user_message_to_context("Hello, world!")

    def test_add_assistant_message_to_context(self):
        """Test adding assistant message to context."""
        from victor.agent.services.chat_service import ChatService

        # Create service
        provider = Mock()
        context = Mock()
        context.add_message = Mock()
        service = _make_chat_service(provider=provider, recovery=None, context=context)

        # Add assistant message
        tool_calls = [{"name": "test_tool", "arguments": "{}"}]
        service._add_assistant_message_to_context(
            "Response",
            tool_calls=tool_calls,
            metadata={"test": "metadata"},
        )

        # Verify context was called
        context.add_message.assert_called_once_with(
            role="assistant",
            content="Response",
            tool_calls=tool_calls,
            metadata={"test": "metadata"},
        )

    def test_add_assistant_message_to_context_without_tool_calls(self):
        """Test adding assistant message without tool calls."""
        from victor.agent.services.chat_service import ChatService

        # Create service
        provider = Mock()
        context = Mock()
        context.add_message = Mock()
        service = _make_chat_service(provider=provider, recovery=None, context=context)

        # Add assistant message without tool calls
        service._add_assistant_message_to_context("Response")

        # Verify context was called
        context.add_message.assert_called_once_with(
            role="assistant",
            content="Response",
            tool_calls=None,
            metadata={},
        )

    def test_add_tool_result_to_context(self):
        """Test adding tool result to context."""
        from victor.agent.services.chat_service import ChatService
        from victor.tools.base import ToolResult

        # Create service
        provider = Mock()
        context = Mock()
        context.add_message = Mock()
        service = _make_chat_service(provider=provider, recovery=None, context=context)

        # Add tool result
        result = ToolResult(output="Result content", error=None, success=True)
        service._add_tool_result_to_context(
            "test_tool",
            result,
            tool_call_id="call_test_123",
        )

        # Verify context was called
        context.add_message.assert_called_once()
        call_args = context.add_message.call_args
        message = call_args.kwargs if call_args.kwargs else call_args[0][0]

        assert message["role"] == "tool"
        assert message["content"] == "Result content"
        assert message["tool_call_id"] == "call_test_123"
        assert message["name"] == "test_tool"

    def test_add_tool_result_to_context_with_error(self):
        """Test adding tool result with error."""
        from victor.agent.services.chat_service import ChatService
        from victor.tools.base import ToolResult

        # Create service
        provider = Mock()
        context = Mock()
        context.add_message = Mock()
        service = _make_chat_service(provider=provider, recovery=None, context=context)

        # Add tool result with error
        result = ToolResult(output="Original result", error="Error occurred", success=False)
        service._add_tool_result_to_context(
            "test_tool",
            result,
            tool_call_id="call_error_123",
        )

        # Verify context was called with error message
        context.add_message.assert_called_once()
        call_args = context.add_message.call_args
        message = call_args.kwargs if call_args.kwargs else call_args[0][0]

        assert message["role"] == "tool"
        # When output is not None, it's used even if there's an error
        assert message["content"] == "Original result"
        assert message["tool_call_id"] == "call_error_123"
        assert message["name"] == "test_tool"

    def test_handle_chat_error_with_recovery(self):
        """Test handling chat error with recovery service."""
        from victor.agent.services.chat_service import ChatService

        # Create service with recovery
        provider = Mock()
        recovery = Mock()
        recovery.should_attempt_recovery = Mock(return_value=True)
        service = _make_chat_service(provider=provider, recovery=recovery, context=None)

        # Handle error
        error = Exception("Test error")
        result = service.handle_chat_error(error, {"context": "info"})

        # Verify recovery was attempted with error type name (string) and consecutive_failures kwarg
        recovery.should_attempt_recovery.assert_called_once_with(
            "Exception",
            consecutive_failures=0,
        )
        assert result["handled"] is True
        assert result["action"] == "retry"
        assert "message" in result

    def test_handle_chat_error_without_recovery(self):
        """Test handling chat error without recovery service."""
        from victor.agent.services.chat_service import ChatService

        # Create service without recovery
        provider = Mock()
        service = _make_chat_service(provider=provider, recovery=None, context=None)

        # Handle error
        error = Exception("Test error")
        result = service.handle_chat_error(error)

        # Verify error was not handled
        assert result["handled"] is False
        assert result["action"] == "abort"
        assert result["message"] == "Test error"

    def test_normalize_tool_arguments_delegates_to_tool_service(self):
        """Test ChatService normalizes arguments via ToolService."""
        from victor.agent.services.chat_service import ChatService

        # Create service with tool service
        provider = Mock()
        tools = Mock()
        tools.normalize_tool_arguments = Mock(return_value=({"normalized": "args"}, "normalized"))
        service = _make_chat_service(provider=provider, recovery=None, context=None, tools=tools)

        # Normalize arguments
        tool_args = {"malformed": "json"}
        result, method = service.normalize_tool_arguments(tool_args, "test_tool")

        # Verify delegation to tool service
        tools.normalize_tool_arguments.assert_called_once_with(tool_args, "test_tool")
        assert result == {"normalized": "args"}
        assert method == "normalized"

    def test_normalize_tool_arguments_without_tool_service(self):
        """Test ChatService normalizes arguments without tool service (direct)."""
        from victor.agent.services.chat_service import ChatService

        # Create service without tool service
        provider = Mock()
        service = _make_chat_service(provider=provider, recovery=None, context=None, tools=None)

        # Normalize arguments
        tool_args = {"foo": "bar"}
        result, method = service.normalize_tool_arguments(tool_args, "test_tool")

        # Verify direct return
        assert result == tool_args
        assert method == "direct"


# =============================================================================
# SessionService Tests (3 methods)
# =============================================================================


class TestSessionServiceCheckpointAndTokenMethods:
    """Test SessionService checkpoint and token usage methods."""

    def test_apply_checkpoint_state(self):
        """Test applying checkpoint state to session."""
        from victor.agent.services.session_service import SessionService

        # Create service
        memory_mgr = Mock()
        service = SessionService(session_state_manager=Mock(), memory_manager=memory_mgr)

        # Create checkpoint state
        checkpoint_state = {
            "session_id": "test-session",
            "tool_calls_used": 5,
            "token_usage": {"input": 100, "output": 50},
            "observed_files": ["/path/to/file.py"],
        }

        # Apply checkpoint
        result = service._apply_checkpoint_state(checkpoint_state)

        # Verify state was applied
        assert result is True
        assert service._session_state.tool_calls_used == 5

    def test_apply_checkpoint_state_with_empty_state(self):
        """Test applying empty checkpoint state."""
        from victor.agent.services.session_service import SessionService

        # Create service
        memory_mgr = Mock()
        service = SessionService(session_state_manager=Mock(), memory_manager=memory_mgr)

        # Apply empty checkpoint
        result = service._apply_checkpoint_state({})

        # Verify returns False
        assert result is False

    def test_apply_checkpoint_state_with_token_usage(self):
        """Test applying checkpoint state with token usage."""
        from victor.agent.services.session_service import SessionService

        # Create service
        memory_mgr = Mock()
        service = SessionService(session_state_manager=Mock(), memory_manager=memory_mgr)

        # Mock update_token_usage
        service._session_state.update_token_usage = Mock()

        # Create checkpoint with token usage
        checkpoint_state = {
            "session_id": "test-session",
            "token_usage": {"input": 200, "output": 100},
        }

        # Apply checkpoint
        result = service._apply_checkpoint_state(checkpoint_state)

        # Verify token usage was updated
        assert result is True
        service._session_state.update_token_usage.assert_called_once_with(200, 100)

    def test_update_token_usage(self):
        """Test updating token usage statistics."""
        from victor.agent.services.session_service import SessionService

        # Create service
        memory_mgr = Mock()
        service = SessionService(session_state_manager=Mock(), memory_manager=memory_mgr)

        # Mock update_token_usage
        service._session_state.update_token_usage = Mock()

        # Update token usage
        service.update_token_usage(150, 75)

        # Verify update was called
        service._session_state.update_token_usage.assert_called_once_with(150, 75)

    def test_update_token_usage_with_setter_fallback(self):
        """Test updating token usage with set_token_usage fallback."""
        from types import SimpleNamespace
        from victor.agent.services.session_service import SessionService

        # Use SimpleNamespace with only set_token_usage so the setter path triggers.
        memory_mgr = Mock()
        service = SessionService(session_state_manager=Mock(), memory_manager=memory_mgr)
        mock_setter = Mock()
        service._session_state = SimpleNamespace(set_token_usage=mock_setter)

        # Update token usage
        service.update_token_usage(150, 75)

        # Verify setter was called
        mock_setter.assert_called_once_with(150, 75)

    def test_update_token_usage_with_dict_fallback(self):
        """Test updating token usage with dict fallback."""
        from types import SimpleNamespace
        from victor.agent.services.session_service import SessionService

        # Use a SimpleNamespace with no token methods so the dict fallback triggers.
        # (Mock objects always return True for hasattr, defeating delattr-based tests.)
        memory_mgr = Mock()
        service = SessionService(session_state_manager=Mock(), memory_manager=memory_mgr)
        service._session_state = SimpleNamespace()  # no update_token_usage or set_token_usage

        # Update token usage (multiple times)
        service.update_token_usage(100, 50)
        service.update_token_usage(50, 25)

        # Verify dict tracking
        assert service._session_state._token_usage["input"] == 150
        assert service._session_state._token_usage["output"] == 75

    def test_reset_token_usage(self):
        """Test resetting token usage statistics."""
        from victor.agent.services.session_service import SessionService

        # Create service
        memory_mgr = Mock()
        service = SessionService(session_state_manager=Mock(), memory_manager=memory_mgr)

        # Mock reset_token_usage
        service._session_state.reset_token_usage = Mock()

        # Reset token usage
        service.reset_token_usage()

        # Verify reset was called
        service._session_state.reset_token_usage.assert_called_once()

    def test_reset_token_usage_with_dict_reset(self):
        """Test resetting token usage with dict reset."""
        from types import SimpleNamespace
        from victor.agent.services.session_service import SessionService

        # Use SimpleNamespace with no reset_token_usage so the dict reset path triggers.
        memory_mgr = Mock()
        service = SessionService(session_state_manager=Mock(), memory_manager=memory_mgr)
        state = SimpleNamespace(_token_usage={"input": 100, "output": 50})
        service._session_state = state

        # Reset token usage
        service.reset_token_usage()

        # Verify dict was reset
        assert state._token_usage == {"input": 0, "output": 0}


# =============================================================================
# ProviderService Tests (1 method)
# =============================================================================


class TestProviderServiceTestMethod:
    """Test ProviderService test_provider method."""

    @pytest.mark.asyncio
    async def test_test_provider_success(self):
        """Test testing a provider successfully."""
        from victor.agent.services.provider_service import ProviderService

        # Create service
        registry = Mock()
        provider = Mock()
        provider.get_model = Mock(return_value="test-model")
        registry.get_provider = Mock(return_value=provider)
        service = ProviderService(registry=registry)

        # Test provider (no health checker, so uses default check)
        result = await service.test_provider("test-provider")

        # Verify success (get_provider called twice: once in test_provider, once in check_provider_health)
        assert result is True
        registry.get_provider.assert_called_with("test-provider")

    @pytest.mark.asyncio
    async def test_test_provider_with_model(self):
        """Test testing a provider with specific model."""
        from victor.agent.services.provider_service import ProviderService

        # Create service
        registry = Mock()
        provider = Mock()
        provider.get_model = Mock(return_value="specific-model")
        provider.set_model = Mock()
        registry.get_provider = Mock(return_value=provider)
        service = ProviderService(registry=registry)

        # Test provider with model
        result = await service.test_provider("test-provider", model="gpt-4")

        # Verify model was set
        assert result is True
        provider.set_model.assert_called_once_with("gpt-4")

    @pytest.mark.asyncio
    async def test_test_provider_not_found(self):
        """Test testing a provider that doesn't exist."""
        from victor.agent.services.provider_service import ProviderService

        # Create service
        registry = Mock()
        registry.get_provider = Mock(return_value=None)
        service = ProviderService(registry=registry)

        # Test non-existent provider
        result = await service.test_provider("unknown-provider")

        # Verify failure
        assert result is False

    @pytest.mark.asyncio
    async def test_test_provider_with_exception(self):
        """Test testing a provider that raises exception."""
        from victor.agent.services.provider_service import ProviderService

        # Create service
        registry = Mock()
        provider = Mock()
        provider.get_model = Mock(side_effect=Exception("Provider error"))
        registry.get_provider = Mock(return_value=provider)
        service = ProviderService(registry=registry)

        # Test provider that raises exception
        result = await service.test_provider("failing-provider")

        # Verify failure (exception caught)
        assert result is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestLowPriorityGapsIntegration:
    """Integration tests for LOW priority gap implementations."""

    def test_all_10_methods_exist(self):
        """Test that all 10 LOW priority methods exist."""
        from victor.agent.services.tool_service import ToolService
        from victor.agent.services.chat_service import ChatService
        from victor.agent.services.session_service import SessionService
        from victor.agent.services.provider_service import ProviderService

        # ToolService methods (2)
        assert hasattr(ToolService, "set_argument_normalizer")
        assert hasattr(ToolService, "normalize_tool_arguments")

        # ChatService methods (5)
        assert hasattr(ChatService, "_add_user_message_to_context")
        assert hasattr(ChatService, "_add_assistant_message_to_context")
        assert hasattr(ChatService, "_add_tool_result_to_context")
        assert hasattr(ChatService, "handle_chat_error")
        assert hasattr(ChatService, "normalize_tool_arguments")

        # SessionService methods (3)
        assert hasattr(SessionService, "_apply_checkpoint_state")
        assert hasattr(SessionService, "update_token_usage")
        assert hasattr(SessionService, "reset_token_usage")

        # ProviderService methods (1)
        assert hasattr(ProviderService, "test_provider")

    def test_method_signatures(self):
        """Test that all methods have correct signatures."""
        import inspect

        from victor.agent.services.tool_service import ToolService
        from victor.agent.services.chat_service import ChatService
        from victor.agent.services.session_service import SessionService
        from victor.agent.services.provider_service import ProviderService

        # ToolService.set_argument_normalizer(normalizer) -> None
        sig = inspect.signature(ToolService.set_argument_normalizer)
        assert len(sig.parameters) == 2  # self, normalizer

        # ToolService.normalize_tool_arguments(tool_args, tool_name) -> tuple
        sig = inspect.signature(ToolService.normalize_tool_arguments)
        assert len(sig.parameters) == 3  # self, tool_args, tool_name

        # ChatService._add_user_message_to_context(content, metadata) -> None
        sig = inspect.signature(ChatService._add_user_message_to_context)
        assert len(sig.parameters) == 3  # self, content, metadata

        # ChatService.handle_chat_error(error, context) -> Dict
        sig = inspect.signature(ChatService.handle_chat_error)
        assert len(sig.parameters) == 3  # self, error, context

        # SessionService.update_token_usage(input, output) -> None
        sig = inspect.signature(SessionService.update_token_usage)
        assert len(sig.parameters) == 3  # self, input_tokens, output_tokens

        # ProviderService.test_provider(provider, model) -> bool (async)
        sig = inspect.signature(ProviderService.test_provider)
        assert len(sig.parameters) == 3  # self, provider, model
        assert inspect.iscoroutinefunction(ProviderService.test_provider)
