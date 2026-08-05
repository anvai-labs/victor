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

"""Integration tests for all services working together.

Tests the complete service-based architecture with all services
interacting correctly.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from typing import Dict, Any


class TestServiceIntegration:
    """Integration tests for all services."""

    @pytest.mark.asyncio
    async def test_complete_chat_flow_with_all_services(self):
        """Test complete chat flow using all services."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig
        from victor.agent.services.chat_service import ChatService, ChatServiceConfig
        from victor.agent.services.session_service import SessionService
        from victor.agent.services.provider_service import ProviderService
        from victor.agent.services.recovery_service import RecoveryService

        # Create all services
        tool_config = ToolServiceConfig()
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        tool_service = ToolService(
            config=tool_config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        provider = Mock()
        recovery = Mock()
        context = Mock()
        conversation_controller = Mock()
        streaming_coordinator = Mock()
        chat_service = ChatService(
            config=ChatServiceConfig(),
            provider_service=provider,
            recovery_service=recovery,
            context_service=context,
            tool_service=tool_service,
            conversation_controller=conversation_controller,
            streaming_coordinator=streaming_coordinator,
        )

        state_manager = Mock()
        lifecycle_manager = Mock()
        memory_manager = Mock()
        session_service = SessionService(
            session_state_manager=state_manager,
            lifecycle_manager=lifecycle_manager,
            memory_manager=memory_manager,
        )

        registry = Mock()
        provider_service = ProviderService(
            registry=registry,
        )

        recovery_service = RecoveryService()
        assert provider_service is not None
        assert recovery_service is not None

        # Test: Create session
        session_id = await session_service.create_session()
        assert session_id is not None
        assert session_service.is_active

        # Test: Add user message to chat
        chat_service._add_user_message_to_context("Hello, world!")

        # Test: Validate tool calls
        tool_service.get_available_tools = Mock(return_value={"code_search", "file_read"})
        valid, invalid = tool_service.validate_tool_calls(
            [
                {"name": "code_search", "arguments": {"query": "test"}},
                {"name": "invalid_tool", "arguments": {}},
            ]
        )
        assert len(valid) == 1
        assert len(invalid) == 1

        # Test: Get session info
        session_info = session_service.get_session_info()
        assert session_info is not None
        assert session_info["session_id"] == session_id

        # Test: End session
        await session_service.end_session()
        assert not session_service.is_active

    @pytest.mark.asyncio
    async def test_session_service_lifecycle_complete(self):
        """Test complete SessionService lifecycle."""
        from victor.agent.services.session_service import SessionService

        # Create service
        state_manager = Mock()
        state_manager.session_id = None
        state_manager.is_active = False
        state_manager.tool_calls_used = 0

        lifecycle_manager = Mock()
        lifecycle_manager.recover_session = Mock(return_value=True)

        memory_manager = Mock()
        memory_manager.create_session = Mock(return_value="mem_session_123")
        memory_manager.end_session = Mock()

        service = SessionService(
            session_state_manager=state_manager,
            lifecycle_manager=lifecycle_manager,
            memory_manager=memory_manager,
        )

        # Test: Create session
        session_id = await service.create_session(metadata={"project": "test"})
        assert session_id is not None
        assert service.is_active is True

        # Test: Update token usage
        service.update_token_usage(100, 50)

        # Test: Get session stats
        stats = service.get_session_stats()
        assert stats is not None
        assert stats["session_id"] == session_id

        # Test: Reset session
        await service.reset_session()
        assert service.is_active is True  # Still active after reset

        # Test: End session
        await service.end_session()
        assert service.is_active is False

    @pytest.mark.asyncio
    async def test_provider_service_switching_and_health(self):
        """Test ProviderService switching and health checks."""
        from victor.agent.services.provider_service import ProviderService

        # Create service
        registry = Mock()
        provider1 = Mock()
        provider1.name = "anthropic"
        provider1.model = "claude-sonnet-4-5"
        provider1.api_key = "key123"
        provider1.supports_streaming = True
        provider1.supports_tools = True
        provider1.max_tokens = 100000

        provider2 = Mock()
        provider2.name = "openai"
        provider2.model = "gpt-4"
        provider2.api_key = "key456"
        provider2.supports_streaming = True
        provider2.supports_tools = True
        provider2.max_tokens = 8000

        registry.get_provider = Mock(
            side_effect=lambda p: provider1 if p == "anthropic" else provider2
        )

        service = ProviderService(
            registry=registry,
        )

        # Test: Switch to first provider
        await service.switch_provider("anthropic", "claude-sonnet-4-5")
        assert service.provider_name == "anthropic"
        assert service.switch_count == 1

        # Test: Get provider info
        info = service.get_current_provider_info()
        assert info.provider_name == "anthropic"
        assert info.model_name == "claude-sonnet-4-5"
        assert info.api_key_configured is True

        # Test: Switch to second provider
        await service.switch_provider("openai", "gpt-4")
        assert service.provider_name == "openai"
        assert service.switch_count == 2

        # Test: Health check
        is_healthy = await service.check_provider_health()
        assert is_healthy is True

        # Test: Test provider
        can_test = await service.test_provider("anthropic")
        assert can_test is True

    def test_recovery_service_strategies_and_fallback(self):
        """Test RecoveryService strategies and fallback."""
        from victor.agent.services.recovery_service import RecoveryService

        # Create service
        config = Mock()
        config.retry_config = Mock()
        config.retry_config.max_attempts = 3
        config.retry_config.base_delay = 1.0
        config.retry_config.max_delay = 60.0
        config.retry_config.strategy = "exponential"

        service = RecoveryService(
            max_retry_attempts=config.retry_config.max_attempts,
            base_retry_delay=config.retry_config.base_delay,
            max_retry_delay=config.retry_config.max_delay,
        )

        # Test: Can retry
        error = Exception("Temporary error")
        assert service.can_retry(error, attempt_count=1) is True
        assert service.can_retry(error, attempt_count=3) is False

        # Test: Calculate backoff delay
        delay = service.calculate_backoff_delay(attempt=1, strategy="exponential")
        assert delay >= 1.0

        # Test: Configure provider chain
        service.configure_provider_chain(
            primary_provider="anthropic", fallback_providers=["openai", "google"]
        )

        assert service.get_primary_provider() == "anthropic"
        assert service.get_fallback_providers() == ["openai", "google"]

        # Test: Should switch provider
        should_switch = service.should_switch_provider(
            current_provider="anthropic",
            error_type="rate_limit",
            consecutive_failures=3,
        )
        assert should_switch is True

        # Test: Get next provider
        next_provider = service.get_next_provider("anthropic")
        assert next_provider == "openai"

        # Test: Loop detection
        content = "This is repeated content"
        recent_responses = ["content1", "content2", content, content]
        is_loop = service.detect_stuck_loop(content, recent_responses)
        # Should detect loop if content repeats

    def test_chat_service_context_and_error_handling(self):
        """Test ChatService context management and error handling."""
        from victor.agent.services.chat_service import ChatService, ChatServiceConfig

        # Create service
        provider = Mock()
        recovery = Mock()
        recovery.should_attempt_recovery = Mock(return_value=False)

        context = Mock()
        context.add_message = Mock()
        conversation_controller = Mock()
        streaming_coordinator = Mock()

        tool_service = Mock()
        tool_service.normalize_tool_arguments = Mock(
            return_value=({"args": "normalized"}, "normalized")
        )

        service = ChatService(
            config=ChatServiceConfig(),
            provider_service=provider,
            recovery_service=recovery,
            context_service=context,
            tool_service=tool_service,
            conversation_controller=conversation_controller,
            streaming_coordinator=streaming_coordinator,
        )

        # Test: Add user message
        service._add_user_message_to_context("Hello", {"test": "meta"})
        context.add_message.assert_called()

        # Test: Add assistant message
        service._add_assistant_message_to_context(
            "Response", tool_calls=[{"name": "test"}], metadata={"test": "meta"}
        )
        assert context.add_message.call_count == 2

        # Test: Add tool result
        service._add_tool_result_to_context(
            "test_tool",
            MagicMock(output="Result", error=None),
            formatted_content="Result",
        )
        assert context.add_message.call_count == 3

        # Test: Handle chat error
        error = Exception("Test error")
        result = service.handle_chat_error(error, {"context": "info"})
        assert result["handled"] is True
        assert result["action"] == "abort"
        assert "message" in result

        # Test: Normalize tool arguments
        args, method = service.normalize_tool_arguments({"args": "test"}, "test_tool")
        assert args == {"args": "normalized"}
        assert method == "normalized"

    def test_tool_service_budget_and_validation_integration(self):
        """Test ToolService budget management and validation integration."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        # Create service
        config = ToolServiceConfig(default_tool_budget=10)
        tool_selector = Mock()
        tool_executor = Mock()
        tool_registrar = Mock()
        service = ToolService(
            config=config,
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registrar,
        )

        # Test: Budget management
        assert service.get_tool_budget() == 10
        assert service.get_remaining_budget() == 10
        assert service.is_budget_exhausted() is False

        # Consume budget
        service.consume_budget(3)
        assert service.get_remaining_budget() == 7
        assert service.budget_used == 3

        # Reset budget
        service.reset_tool_budget()
        assert service.get_remaining_budget() == 10
        assert service.budget_used == 0

        # Test: Tool configuration
        service.get_available_tools = Mock(return_value={"code_search", "file_read"})

        service.set_enabled_tools({"code_search"})
        assert service.is_tool_enabled("code_search") is True
        assert service.is_tool_enabled("file_read") is False

        # Test: Get enabled tools
        enabled = service.get_enabled_tools()
        assert "code_search" in enabled
        assert "file_read" not in enabled

        # Test: Validation integration
        valid, invalid = service.validate_tool_calls(
            [
                {"name": "code_search", "arguments": {"query": "test"}},
                {"name": "file_read", "arguments": {"path": "/tmp/test"}},
            ]
        )
        assert len(valid) == 1
        assert valid[0]["name"] == "code_search"
        assert len(invalid) == 1
        assert invalid[0]["name"] == "file_read"

    @pytest.mark.asyncio
    async def test_service_error_recovery_workflow(self):
        """Test complete error recovery workflow across services."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig
        from victor.agent.services.chat_service import ChatService, ChatServiceConfig
        from victor.agent.services.provider_service import ProviderService
        from victor.agent.services.recovery_service import RecoveryService

        # Create services
        tool_config = ToolServiceConfig()
        tool_service = ToolService(
            config=tool_config,
            tool_selector=Mock(),
            tool_executor=Mock(),
            tool_registrar=Mock(),
        )

        provider = Mock()
        recovery = Mock()
        recovery.should_attempt_recovery = Mock(return_value=True)
        context = Mock()
        conversation_controller = Mock()
        streaming_coordinator = Mock()
        chat_service = ChatService(
            config=ChatServiceConfig(),
            provider_service=provider,
            recovery_service=recovery,
            context_service=context,
            tool_service=tool_service,
            conversation_controller=conversation_controller,
            streaming_coordinator=streaming_coordinator,
        )

        registry = Mock()
        provider_service = ProviderService(
            registry=registry,
        )

        recovery_service = RecoveryService()

        # Simulate error scenario
        error = Exception("Rate limit exceeded")

        # Test: Chat service handles error
        result = chat_service.handle_chat_error(error)
        assert result["action"] == "retry"

        # Test: Recovery service suggests provider switch
        recovery_service.configure_provider_chain(
            primary_provider="anthropic", fallback_providers=["openai"]
        )

        should_switch = recovery_service.should_switch_provider("anthropic", "rate_limit", 3)
        assert should_switch is True

        # Test: Provider service switches
        next_provider = recovery_service.get_next_provider("anthropic")
        assert next_provider == "openai"

    @pytest.mark.asyncio
    async def test_session_persistence_and_serialization(self):
        """Test session persistence and serialization."""
        from victor.agent.services.session_service import SessionService

        # Create service
        state_manager = Mock()
        state_manager.reset = Mock()
        state_manager.tool_calls_used = 5
        state_manager.get_token_usage = Mock(return_value={"input": 100, "output": 50})

        lifecycle_manager = Mock()
        memory_manager = Mock()
        memory_manager.create_session = Mock(return_value="memory_test_session")

        service = SessionService(
            session_state_manager=state_manager,
            lifecycle_manager=lifecycle_manager,
            memory_manager=memory_manager,
        )
        await service.create_session(session_id="test_session")

        # Test: Serialize to dict
        data = service.to_dict()
        assert data["session_id"] == "test_session"
        assert data["is_active"] is True
        assert data["tool_calls_used"] == 5
        assert data["token_usage"]["input"] == 100

        # Test: Deserialize from dict
        restored_service = SessionService.from_dict(data)
        assert restored_service.session_id == "test_session"

    def test_all_services_health_checks(self):
        """Test health checks across all services."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig
        from victor.agent.services.chat_service import ChatService, ChatServiceConfig
        from victor.agent.services.session_service import SessionService
        from victor.agent.services.provider_service import ProviderService
        from victor.agent.services.recovery_service import RecoveryService

        # Create services
        tool_service = ToolService(
            config=ToolServiceConfig(),
            tool_selector=Mock(),
            tool_executor=Mock(),
            tool_registrar=Mock(),
        )
        tool_service._executor = Mock()  # Make it healthy

        chat_service = ChatService(
            config=ChatServiceConfig(),
            provider_service=Mock(),
            recovery_service=Mock(),
            context_service=Mock(),
            tool_service=Mock(),
            conversation_controller=Mock(),
            streaming_coordinator=Mock(),
        )

        state_manager = Mock()
        session_service = SessionService(
            session_state_manager=state_manager,
            lifecycle_manager=Mock(),
            memory_manager=Mock(),
        )

        provider = Mock()
        registry = Mock()
        registry.get_provider = Mock(return_value=provider)
        provider_service = ProviderService(
            registry=registry,
        )

        recovery_service = RecoveryService()

        # Test: All services are healthy
        assert tool_service.is_healthy() is True
        assert chat_service.is_healthy() is True
        assert session_service.is_healthy() is True
        assert provider_service.is_healthy() is False  # No current provider
        assert recovery_service.is_healthy() is True

    @pytest.mark.asyncio
    async def test_service_interaction_patterns(self):
        """Test common interaction patterns between services."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig
        from victor.agent.services.chat_service import ChatService, ChatServiceConfig

        # Create services
        tool_config = ToolServiceConfig()
        tool_service = ToolService(
            config=tool_config,
            tool_selector=Mock(),
            tool_executor=AsyncMock(return_value={"result": "success"}),
            tool_registrar=Mock(),
        )

        chat_service = ChatService(
            config=ChatServiceConfig(),
            provider_service=Mock(),
            recovery_service=Mock(),
            context_service=Mock(),
            tool_service=tool_service,
            conversation_controller=Mock(),
            streaming_coordinator=Mock(),
        )

        # Pattern 1: ChatService delegates to ToolService
        args, method = chat_service.normalize_tool_arguments({"query": "test"}, "code_search")
        # Should delegate to tool_service

        # Pattern 2: Tool execution with budget
        tool_service.consume_budget(1)
        assert tool_service.budget_used == 1

        # Pattern 3: Error handling
        try:
            raise Exception("Test error")
        except Exception as e:
            result = chat_service.handle_chat_error(e)
            assert result["handled"] is True


class TestServiceErrorScenarios:
    """Test error scenarios across services."""

    def test_tool_service_budget_exhausted(self):
        """Test ToolService behavior when budget exhausted."""
        from victor.agent.services.tool_service import ToolService, ToolServiceConfig

        config = ToolServiceConfig(default_tool_budget=5)
        service = ToolService(
            config=config,
            tool_selector=Mock(),
            tool_executor=Mock(),
            tool_registrar=Mock(),
        )

        # Exhaust budget
        service.consume_budget(5)
        assert service.is_budget_exhausted() is True

        # Try to consume more
        from victor.core.errors import BudgetExhaustedError

        with pytest.raises(BudgetExhaustedError):
            service.consume_budget(1)
        assert service.budget_used == 5

    @pytest.mark.asyncio
    async def test_provider_service_switch_failure(self):
        """Test ProviderService when switch fails."""
        from victor.agent.services.provider_service import ProviderService

        registry = Mock()
        registry.get_provider = Mock(return_value=None)  # Provider not found

        service = ProviderService(
            registry=registry,
        )

        # Test: Switch to non-existent provider
        with pytest.raises(ValueError):
            await service.switch_provider("unknown_provider")

    def test_session_service_recovery_failure(self):
        """Test SessionService when recovery fails."""
        from victor.agent.services.session_service import SessionService

        state_manager = Mock()
        lifecycle_manager = Mock()
        lifecycle_manager.recover_session = Mock(return_value=False)  # Recovery fails
        memory_manager = Mock()

        service = SessionService(
            session_state_manager=state_manager,
            lifecycle_manager=lifecycle_manager,
            memory_manager=memory_manager,
        )

        # Test: Recover session that doesn't exist
        success = service.recover_session("non_existent_session")
        assert success is False

    def test_recovery_service_max_retries_exceeded(self):
        """Test RecoveryService when max retries exceeded."""
        from victor.agent.services.recovery_service import RecoveryService

        config = Mock()
        config.retry_config = Mock()
        config.retry_config.max_attempts = 3

        service = RecoveryService(max_retry_attempts=config.retry_config.max_attempts)

        # Test: Cannot retry after max attempts
        error = Exception("Persistent error")
        assert service.can_retry(error, attempt_count=3) is False
        assert service.can_retry(error, attempt_count=4) is False

    def test_chat_service_error_without_recovery(self):
        """Test ChatService when recovery service not available."""
        from victor.agent.services.chat_service import ChatService, ChatServiceConfig

        service = ChatService(
            config=ChatServiceConfig(),
            provider_service=Mock(),
            recovery_service=None,  # No recovery service
            context_service=Mock(),
            tool_service=Mock(),
            conversation_controller=Mock(),
            streaming_coordinator=Mock(),
        )

        # Test: Handle error without recovery
        error = Exception("Test error")
        result = service.handle_chat_error(error)
        assert result["handled"] is False
        assert result["action"] == "abort"
