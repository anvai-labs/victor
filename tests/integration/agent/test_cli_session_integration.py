"""Integration tests for CLI session management.

Tests the unified session management for CLI mode including:
- Interactive CLI session initialization
- Message processing with streaming
- Session lifecycle (initialize → process → cleanup)
- Mode switching and overrides
- Error handling and graceful degradation

Test Approach:
- Uses real session handlers and creation strategies but mocks AgentFactory
- Tests integration between BaseSessionHandler, InteractiveSessionHandler, and CLI
- Verifies end-to-end session flow from user input to agent response
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from victor.agent.session_manager_base import (
    SessionMode,
    SessionConfig,
    SessionMetrics,
    InteractiveSessionHandler,
    OneshotSessionHandler,
)
from victor.core.verticals.import_resolver import import_module_with_fallback


def _load_vertical_attr(module_path: str, attr_name: str):
    """Resolve a vertical attribute and skip test when unavailable."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None or not hasattr(module, attr_name):
        pytest.skip(f"Vertical module or attribute unavailable: {module_path}:{attr_name}")
    return getattr(module, attr_name)


class TestCLISessionInitialization:
    """Test CLI session initialization flow."""

    @pytest.fixture
    def session_config(self):
        """Return a standard CLI session configuration."""
        return SessionConfig(
            mode=SessionMode.INTERACTIVE,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
            thinking=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_session_initialization(self, session_config):
        """Test complete session initialization from config to agent."""
        handler = InteractiveSessionHandler()

        # Mock the canonical agent creation path
        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            agent = await handler.initialize(session_config)

        # Verify agent was created
        assert agent is mock_agent
        mock_factory.create.assert_called_once_with()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_session_with_custom_budget_and_iterations(self, session_config):
        """Test session with custom budget and iteration overrides."""
        handler = InteractiveSessionHandler()

        session_config.tool_budget = 100
        session_config.max_iterations = 50

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch(
            "victor.framework.agent_factory.AgentFactory", return_value=mock_factory
        ) as mock_agent_factory:
            await handler.initialize(session_config)

        # Verify overrides were forwarded to the canonical factory
        assert mock_agent_factory.call_args.kwargs["tool_budget"] == 100
        assert mock_agent_factory.call_args.kwargs["max_iterations"] == 50

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_session_with_vertical_integration(self, session_config):
        """Test session with vertical integration."""
        CodingAssistant = _load_vertical_attr("victor.coding.assistant", "CodingAssistant")

        session_config.vertical = "coding"

        handler = InteractiveSessionHandler()

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            with patch("victor.core.verticals.get_vertical") as mock_get_vertical:
                mock_get_vertical.return_value = CodingAssistant

                await handler.initialize(session_config)

                # Verify vertical was loaded
                mock_get_vertical.assert_called_once_with("coding")


class TestCLIMessageProcessing:
    """Test CLI message processing flow."""

    @pytest.fixture
    def handler(self):
        """Return InteractiveSessionHandler."""
        return InteractiveSessionHandler()

    @pytest.fixture
    def mock_agent(self):
        """Return mock agent with streaming support."""
        agent = MagicMock()
        agent.provider.supports_streaming.return_value = True

        # Mock streaming chat
        async def mock_stream(message):
            chunks = [
                MagicMock(content="Hello ", type="content"),
                MagicMock(content="from ", type="content"),
                MagicMock(content="Victor!", type="content"),
            ]
            for chunk in chunks:
                yield chunk

        agent.stream_chat = mock_stream
        return agent

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_streaming_message_processing(self, handler, mock_agent):
        """Test streaming message processing."""
        response = await handler.process_message(mock_agent, "Hello", stream=True)

        assert response == "Hello from Victor!"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_non_streaming_message_processing(self, handler):
        """Test non-streaming message processing."""
        mock_agent = MagicMock()
        mock_agent.provider.supports_streaming.return_value = False

        mock_response = MagicMock()
        mock_response.content = "Direct response"
        mock_agent.chat = AsyncMock(return_value=mock_response)

        response = await handler.process_message(mock_agent, "Test", stream=False)

        assert response == "Direct response"
        mock_agent.chat.assert_called_once_with("Test")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_message_processing(self, handler):
        """Test concurrent message processing (simulating rapid user input)."""
        import asyncio

        mock_agent = MagicMock()
        mock_agent.provider.supports_streaming.return_value = True

        # Mock streaming chat
        async def mock_stream(message):
            await asyncio.sleep(0.01)  # Simulate processing time
            yield MagicMock(content=f"Response to {message}", type="content")

        mock_agent.stream_chat = mock_stream

        # Process multiple messages concurrently
        messages = ["Hello", "World", "Test"]
        tasks = [handler.process_message(mock_agent, msg, stream=True) for msg in messages]

        responses = await asyncio.gather(*tasks)

        assert len(responses) == 3
        assert all("Response to" in r for r in responses)


class TestCLISessionLifecycle:
    """Test complete CLI session lifecycle."""

    @pytest.fixture
    def handler(self):
        """Return InteractiveSessionHandler."""
        return InteractiveSessionHandler()

    @pytest.fixture
    def session_config(self):
        """Return session configuration."""
        return SessionConfig(
            mode=SessionMode.INTERACTIVE,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_session_lifecycle(self, handler, session_config):
        """Test complete lifecycle: initialize → process → cleanup."""
        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()
        mock_agent.get_session_metrics.return_value = {
            "tool_calls": 10,
            "tokens": 500,
            "iterations": 5,
        }

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            # Initialize
            agent = await handler.initialize(session_config)
            assert agent is mock_agent

            # Process message
            mock_agent.provider.supports_streaming.return_value = True

            async def mock_stream(msg):
                yield MagicMock(content="Response", type="content")

            mock_agent.stream_chat = mock_stream

            response = await handler.process_message(agent, "Test", stream=True)
            assert response == "Response"

            # Cleanup
            metrics = SessionMetrics()
            await handler.cleanup(agent, metrics)

            assert metrics.end_time is not None
            assert metrics.tool_calls == 10
            assert metrics.tokens_used == 500
            assert metrics.iterations == 5

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_session_cleanup_without_metrics(self, handler, session_config):
        """Test session cleanup when agent doesn't provide metrics."""
        mock_agent = MagicMock()
        mock_agent.get_session_metrics.return_value = None

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            agent = await handler.initialize(session_config)

        metrics = SessionMetrics()

        # Should not crash
        await handler.cleanup(agent, metrics)

        assert metrics.end_time is not None
        assert metrics.tool_calls == 0  # Should remain at default


class TestOneShotIntegration:
    """Test one-shot session integration."""

    @pytest.fixture
    def handler(self):
        """Return OneshotSessionHandler."""
        return OneshotSessionHandler()

    @pytest.fixture
    def session_config(self):
        """Return one-shot session configuration."""
        return SessionConfig(
            mode=SessionMode.ONESHOT,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_oneshot_execute_full_flow(self, handler, session_config):
        """Test complete one-shot execution flow."""
        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()
        mock_agent.get_session_metrics.return_value = {
            "tool_calls": 3,
            "tokens": 100,
            "iterations": 1,
        }

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):

            async def mock_stream(msg):
                yield MagicMock(content="One-shot response", type="content")

            with patch.object(handler, "process_message", return_value="One-shot response"):
                metrics = await handler.execute(session_config, "Test message")

        assert metrics.success is True
        assert metrics.end_time is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_oneshot_error_handling(self, handler, session_config):
        """Test one-shot error handling and cleanup."""
        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(side_effect=Exception("Initialization failed"))

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            with pytest.raises(Exception, match="Initialization failed"):
                await handler.execute(session_config, "Test message")


class TestCLIModeSwitching:
    """Test mode switching integration."""

    @pytest.fixture
    def session_config(self):
        """Return session configuration with mode override."""
        return SessionConfig(
            mode=SessionMode.INTERACTIVE,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
            mode_name="plan",  # Switch to plan mode
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mode_switch_during_initialization(self, session_config):
        """Test mode switching during session initialization."""
        handler = InteractiveSessionHandler()

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            with patch("victor.agent.mode_controller.get_mode_controller") as mock_get_controller:
                mock_controller = MagicMock()
                mock_get_controller.return_value = mock_controller

                await handler.initialize(session_config)

                # Verify mode was switched
                mock_controller.switch_mode.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_invalid_mode_switch_handling(self, session_config):
        """Test graceful handling of invalid mode switch."""

        handler = InteractiveSessionHandler()

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            with patch("victor.agent.mode_controller.get_mode_controller") as mock_get_controller:
                mock_controller = MagicMock()
                # Simulate mode switch failure
                mock_controller.switch_mode.side_effect = ValueError("Invalid mode")
                mock_get_controller.return_value = mock_controller

                # Should not crash despite mode switch failure
                agent = await handler.initialize(session_config)

                assert agent is mock_agent


class TestCLISessionPersistence:
    """Test session persistence and state management."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_session_metrics_tracking(self):
        """Test that session metrics are tracked across the lifecycle."""
        handler = InteractiveSessionHandler()

        config = SessionConfig(
            mode=SessionMode.INTERACTIVE,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

        mock_agent = MagicMock()
        mock_agent.unified_tracker = MagicMock()
        mock_agent.get_session_metrics.return_value = {
            "tool_calls": 15,
            "tokens": 750,
            "iterations": 8,
        }

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            agent = await handler.initialize(config)

        # Process some messages
        async def mock_stream(msg):
            yield MagicMock(content="Response", type="content")

        mock_agent.stream_chat = mock_stream
        mock_agent.provider.supports_streaming.return_value = True

        await handler.process_message(agent, "Message 1", stream=True)
        await handler.process_message(agent, "Message 2", stream=True)

        # Cleanup and collect metrics
        metrics = SessionMetrics()
        await handler.cleanup(agent, metrics)

        # Verify metrics were collected
        assert metrics.tool_calls == 15
        assert metrics.tokens_used == 750
        assert metrics.iterations == 8
        assert metrics.success is True
        assert metrics.duration_seconds > 0


class TestCLIErrorRecovery:
    """Test error recovery in CLI sessions."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_recovery_from_streaming_error(self):
        """Test recovery from streaming error."""
        handler = InteractiveSessionHandler()

        config = SessionConfig(
            mode=SessionMode.INTERACTIVE,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

        mock_agent = MagicMock()

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(return_value=mock_agent)

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            agent = await handler.initialize(config)

        # Simulate streaming failure
        mock_agent.provider.supports_streaming.return_value = True

        async def mock_stream_error(msg):
            yield MagicMock(content="Start", type="content")
            raise Exception("Connection lost")

        mock_agent.stream_chat = mock_stream_error

        # Should raise the error
        with pytest.raises(Exception, match="Connection lost"):
            await handler.process_message(agent, "Test", stream=True)

        # But cleanup should still work
        metrics = SessionMetrics()
        await handler.cleanup(agent, metrics)

        assert metrics.end_time is not None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_recovery_from_initialization_error(self):
        """Test recovery from initialization error."""
        handler = InteractiveSessionHandler()

        config = SessionConfig(
            mode=SessionMode.INTERACTIVE,
            provider="anthropic",
            model="claude-3-5-sonnet",
            profile="default",
        )

        mock_factory = MagicMock()
        mock_factory.create = AsyncMock(side_effect=RuntimeError("Provider unavailable"))

        with patch("victor.framework.agent_factory.AgentFactory", return_value=mock_factory):
            # Should raise initialization error
            with pytest.raises(RuntimeError, match="Provider unavailable"):
                await handler.initialize(config)
