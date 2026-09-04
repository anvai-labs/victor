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

"""Tests for LifecycleManager."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, call
from typing import Any, Dict, Optional

from victor.agent.lifecycle_manager import LifecycleManager


class TestLifecycleManager:
    """Tests for LifecycleManager class."""

    @pytest.fixture
    def conversation_controller(self):
        """Create a mock conversation controller."""
        controller = MagicMock()
        controller.clear = MagicMock()
        controller.reset = MagicMock()
        return controller

    @pytest.fixture
    def metrics_collector(self):
        """Create a mock metrics collector."""
        collector = MagicMock()
        collector.reset_stats = MagicMock()
        return collector

    @pytest.fixture
    def context_compactor(self):
        """Create a mock context compactor."""
        compactor = MagicMock()
        compactor.reset_statistics = MagicMock()
        return compactor

    @pytest.fixture
    def sequence_tracker(self):
        """Create a mock sequence tracker."""
        tracker = MagicMock()
        tracker.clear_history = MagicMock()
        return tracker

    @pytest.fixture
    def usage_analytics(self):
        """Create a mock usage analytics."""
        analytics = MagicMock()
        analytics._current_session = MagicMock()
        analytics.end_session = MagicMock()
        analytics.start_session = MagicMock()
        return analytics

    @pytest.fixture
    def reminder_manager(self):
        """Create a mock reminder manager."""
        manager = MagicMock()
        manager.reset = MagicMock()
        return manager

    @pytest.fixture
    def provider(self):
        """Create a mock provider."""
        provider = AsyncMock()
        provider.close = AsyncMock()
        return provider

    @pytest.fixture
    def code_manager(self):
        """Create a mock code execution manager."""
        manager = MagicMock()
        manager.stop = MagicMock()
        return manager

    @pytest.fixture
    def semantic_selector(self):
        """Create a mock semantic selector."""
        selector = AsyncMock()
        selector.close = AsyncMock()
        return selector

    @pytest.fixture
    def background_tasks(self):
        """Create mock background tasks."""
        # Create simple mock tasks
        task1 = MagicMock()
        task1.done = MagicMock(return_value=False)
        task1.cancel = MagicMock()
        task1.result = MagicMock(return_value=None)

        task2 = MagicMock()
        task2.done = MagicMock(return_value=True)
        task2.cancel = MagicMock()

        return [task1, task2]

    @pytest.fixture
    def lifecycle_manager(
        self,
        conversation_controller,
        metrics_collector,
        context_compactor,
        sequence_tracker,
        usage_analytics,
        reminder_manager,
    ):
        """Create LifecycleManager with mocks."""
        from victor.agent.lifecycle_manager import LifecycleManager

        return LifecycleManager(
            conversation_controller=conversation_controller,
            metrics_collector=metrics_collector,
            context_compactor=context_compactor,
            sequence_tracker=sequence_tracker,
            usage_analytics=usage_analytics,
            reminder_manager=reminder_manager,
        )

    def test_init(self, lifecycle_manager, conversation_controller):
        """Test lifecycle manager initialization."""
        assert lifecycle_manager._conversation_controller is conversation_controller

    def test_reset_conversation(self, lifecycle_manager, conversation_controller):
        """Test conversation reset."""
        # Reset conversation
        lifecycle_manager.reset_conversation()

        # Verify controller was reset
        conversation_controller.reset.assert_called_once()

    def test_reset_conversation_with_all_components(
        self,
        lifecycle_manager,
        conversation_controller,
        metrics_collector,
        context_compactor,
        sequence_tracker,
        usage_analytics,
        reminder_manager,
    ):
        """Test conversation reset with all components."""
        # Reset
        lifecycle_manager.reset_conversation()

        # Verify all components were reset
        conversation_controller.reset.assert_called_once()
        reminder_manager.reset.assert_called_once()
        metrics_collector.reset_stats.assert_called_once()
        context_compactor.reset_statistics.assert_called_once()
        sequence_tracker.clear_history.assert_called_once()

        # Verify usage analytics session was restarted
        usage_analytics.end_session.assert_called_once()
        usage_analytics.start_session.assert_called_once()

    def test_reset_conversation_without_usage_analytics(
        self, lifecycle_manager, conversation_controller, usage_analytics
    ):
        """Test reset when usage analytics has no active session."""
        # Setup analytics with no active session
        usage_analytics._current_session = None

        # Reset
        lifecycle_manager.reset_conversation()

        # Verify no error, graceful handling
        conversation_controller.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_success(self, lifecycle_manager, usage_analytics):
        """Test successful graceful shutdown."""
        # Mock flush_analytics
        lifecycle_manager._flush_analytics = MagicMock(return_value={"analytics": True})

        # Mock stop_health_monitoring
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        # Shutdown
        result = await lifecycle_manager.graceful_shutdown()

        # Verify result
        assert result["analytics_flushed"] is True
        assert result["health_monitoring_stopped"] is True
        assert result["session_ended"] is True

        # Verify usage analytics session ended
        usage_analytics.end_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_analytics_failure(
        self, lifecycle_manager, usage_analytics
    ):
        """Test graceful shutdown when analytics flush fails."""
        # Mock flush_analytics that fails
        lifecycle_manager._flush_analytics = MagicMock(return_value={"analytics": False})

        # Mock stop_health_monitoring
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        # Shutdown
        result = await lifecycle_manager.graceful_shutdown()

        # Verify partial success
        assert result["analytics_flushed"] is False
        assert result["health_monitoring_stopped"] is True

    @pytest.mark.asyncio
    async def test_graceful_shutdown_without_usage_analytics(self, lifecycle_manager):
        """Test graceful shutdown when usage analytics not configured."""
        # Remove usage analytics
        lifecycle_manager._usage_analytics = None

        # Mock dependencies
        lifecycle_manager._flush_analytics = MagicMock(return_value={})
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        # Shutdown - should not fail
        result = await lifecycle_manager.graceful_shutdown()

        # Verify success despite no analytics
        assert result["session_ended"] is True

    @pytest.mark.asyncio
    async def test_graceful_shutdown_closes_usage_logger(self, lifecycle_manager):
        """Co-design review item 14: graceful_shutdown must drain any
        buffered usage-logger events before process exit."""
        usage_logger = MagicMock()
        lifecycle_manager._usage_logger = usage_logger

        lifecycle_manager._flush_analytics = MagicMock(return_value={})
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        result = await lifecycle_manager.graceful_shutdown()

        usage_logger.close.assert_called_once()
        assert result["usage_logger_closed"] is True

    @pytest.mark.asyncio
    async def test_graceful_shutdown_without_usage_logger(self, lifecycle_manager):
        """No usage logger configured: shutdown must not fail and must
        report the step as trivially successful."""
        lifecycle_manager._usage_logger = None

        lifecycle_manager._flush_analytics = MagicMock(return_value={})
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        result = await lifecycle_manager.graceful_shutdown()

        assert result["usage_logger_closed"] is True

    @pytest.mark.asyncio
    async def test_graceful_shutdown_usage_logger_without_close(self, lifecycle_manager):
        """A usage logger lacking a close() method must not crash shutdown
        (hasattr guard) and must be reported as a no-op success."""
        usage_logger = MagicMock(spec=[])  # no attributes at all, including close
        lifecycle_manager._usage_logger = usage_logger

        lifecycle_manager._flush_analytics = MagicMock(return_value={})
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        result = await lifecycle_manager.graceful_shutdown()

        assert result["usage_logger_closed"] is True

    @pytest.mark.asyncio
    async def test_graceful_shutdown_usage_logger_close_raises(self, lifecycle_manager):
        """A usage logger whose close() raises must not abort the rest of
        shutdown, and must be reported as a failed (not successful) step."""
        usage_logger = MagicMock()
        usage_logger.close.side_effect = RuntimeError("disk full")
        lifecycle_manager._usage_logger = usage_logger

        lifecycle_manager._flush_analytics = MagicMock(return_value={})
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        result = await lifecycle_manager.graceful_shutdown()

        usage_logger.close.assert_called_once()
        assert result["usage_logger_closed"] is False
        # The rest of shutdown still completed.
        assert result["session_ended"] is True

    @pytest.mark.asyncio
    async def test_shutdown(
        self,
        lifecycle_manager,
        provider,
        code_manager,
        semantic_selector,
        background_tasks,
    ):
        """Test full shutdown."""
        # Setup dependencies
        lifecycle_manager._provider = provider
        lifecycle_manager._code_manager = code_manager
        lifecycle_manager._semantic_selector = semantic_selector
        lifecycle_manager._background_tasks = background_tasks
        lifecycle_manager._usage_logger = MagicMock()

        # Shutdown
        await lifecycle_manager.shutdown()

        # Verify background tasks cancelled (if any)
        if background_tasks:
            background_tasks[0].cancel.assert_called_once()

        # Verify provider closed
        provider.close.assert_called_once()

        # Verify code manager stopped
        code_manager.stop.assert_called_once()

        # Verify semantic selector closed
        semantic_selector.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_provider_error(self, lifecycle_manager, provider):
        """Test shutdown when provider close fails."""
        # Setup provider that raises error
        provider.close = AsyncMock(side_effect=RuntimeError("Provider error"))

        lifecycle_manager._provider = provider
        lifecycle_manager._code_manager = None
        lifecycle_manager._semantic_selector = None
        lifecycle_manager._background_tasks = []
        lifecycle_manager._usage_logger = MagicMock()

        # Shutdown - should not raise
        await lifecycle_manager.shutdown()

        # Verify close was attempted
        provider.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_without_background_tasks(self, lifecycle_manager, provider):
        """Test shutdown when no background tasks."""
        # Setup with no background tasks
        lifecycle_manager._background_tasks = []
        lifecycle_manager._provider = provider
        lifecycle_manager._code_manager = None
        lifecycle_manager._semantic_selector = None
        lifecycle_manager._usage_logger = MagicMock()

        # Shutdown - should not fail
        await lifecycle_manager.shutdown()

        # Verify provider closed
        provider.close.assert_called_once()

    def test_get_session_stats(self, lifecycle_manager):
        """Test getting session statistics."""
        # Setup conversation controller
        lifecycle_manager._conversation_controller.message_count = MagicMock(return_value=42)

        # Get stats
        stats = lifecycle_manager.get_session_stats()

        # Verify stats
        assert "message_count" in stats
        assert stats["message_count"] == 42

    def test_recover_session_success(self, lifecycle_manager):
        """Test successful session recovery."""
        # Mock memory manager
        memory_manager = MagicMock()
        session = MagicMock()
        session.session_id = "test-session-123"
        session.messages = [MagicMock(role="user", content="Hello", to_provider_format=MagicMock())]

        memory_manager.get_session = MagicMock(return_value=session)

        # Recover
        result = lifecycle_manager.recover_session(
            session_id="test-session-123",
            memory_manager=memory_manager,
        )

        # Verify recovery
        assert result is True
        memory_manager.get_session.assert_called_once_with("test-session-123")

    def test_recover_session_not_found(self, lifecycle_manager):
        """Test session recovery when session not found."""
        # Mock memory manager that returns None
        memory_manager = MagicMock()
        memory_manager.get_session = MagicMock(return_value=None)

        # Recover
        result = lifecycle_manager.recover_session(
            session_id="nonexistent-session",
            memory_manager=memory_manager,
        )

        # Verify failure
        assert result is False

    def test_recover_session_without_memory_manager(self, lifecycle_manager):
        """Test session recovery without memory manager."""
        # Recover without memory manager
        result = lifecycle_manager.recover_session(
            session_id="test-session",
            memory_manager=None,
        )

        # Verify failure
        assert result is False


class TestLifecycleManagerIntegration:
    """Integration tests for LifecycleManager."""

    @pytest.fixture
    def lifecycle_manager(self):
        """Create lifecycle manager with real dependencies."""
        from victor.agent.lifecycle_manager import LifecycleManager

        # Use mocks for dependencies
        return LifecycleManager(
            conversation_controller=MagicMock(),
            metrics_collector=MagicMock(),
            context_compactor=MagicMock(),
            sequence_tracker=MagicMock(),
            usage_analytics=MagicMock(),
            reminder_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, lifecycle_manager):
        """Test complete lifecycle: reset -> shutdown."""
        # Reset
        lifecycle_manager.reset_conversation()

        # Setup for shutdown
        lifecycle_manager._provider = AsyncMock()
        lifecycle_manager._provider.close = AsyncMock()
        lifecycle_manager._code_manager = None
        lifecycle_manager._semantic_selector = None
        lifecycle_manager._background_tasks = []
        lifecycle_manager._usage_logger = MagicMock()
        lifecycle_manager._flush_analytics = MagicMock(return_value={})
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        # Shutdown
        await lifecycle_manager.graceful_shutdown()
        await lifecycle_manager.shutdown()

        # Verify all components reset
        lifecycle_manager._conversation_controller.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_recover_shutdown_cycle(self, lifecycle_manager):
        """Test cycle: reset -> recover -> shutdown."""
        # Reset
        lifecycle_manager.reset_conversation()

        # Mock recovery
        memory_manager = MagicMock()
        session = MagicMock()
        session.session_id = "test-123"
        session.messages = []
        memory_manager.get_session = MagicMock(return_value=session)

        # Recover
        result = lifecycle_manager.recover_session(
            session_id="test-123",
            memory_manager=memory_manager,
        )
        assert result is True

        # Setup for shutdown
        lifecycle_manager._provider = AsyncMock()
        lifecycle_manager._provider.close = AsyncMock()
        lifecycle_manager._code_manager = None
        lifecycle_manager._semantic_selector = None
        lifecycle_manager._background_tasks = []
        lifecycle_manager._usage_logger = MagicMock()
        lifecycle_manager._flush_analytics = MagicMock(return_value={})
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        # Shutdown
        await lifecycle_manager.shutdown()


class TestLifecycleManagerErrorHandling:
    """Error handling tests for LifecycleManager."""

    @pytest.fixture
    def lifecycle_manager(self):
        """Create lifecycle manager."""
        from victor.agent.lifecycle_manager import LifecycleManager

        return LifecycleManager(
            conversation_controller=MagicMock(),
            metrics_collector=MagicMock(),
            context_compactor=MagicMock(),
            sequence_tracker=MagicMock(),
            usage_analytics=MagicMock(),
            reminder_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_graceful_shutdown_handles_exception(self, lifecycle_manager):
        """Test that graceful shutdown exceptions are handled gracefully."""
        # Mock flush that raises exception
        lifecycle_manager._flush_analytics = MagicMock(side_effect=RuntimeError("Flush failed"))
        lifecycle_manager._stop_health_monitoring = AsyncMock(return_value=True)

        # Shutdown should handle exception
        result = await lifecycle_manager.graceful_shutdown()

        # Verify partial success despite error
        assert "analytics_flushed" in result
        assert result["analytics_flushed"] is False

    @pytest.mark.asyncio
    async def test_shutdown_handles_all_component_errors(self, lifecycle_manager):
        """Test shutdown when multiple components fail."""
        # Setup components that fail
        provider = AsyncMock()
        provider.close = AsyncMock(side_effect=RuntimeError("Provider error"))

        code_manager = MagicMock()
        code_manager.stop = MagicMock(side_effect=RuntimeError("Code manager error"))

        semantic_selector = AsyncMock()
        semantic_selector.close = AsyncMock(side_effect=RuntimeError("Selector error"))

        lifecycle_manager._provider = provider
        lifecycle_manager._code_manager = code_manager
        lifecycle_manager._semantic_selector = semantic_selector
        lifecycle_manager._background_tasks = []
        lifecycle_manager._usage_logger = MagicMock()

        # Shutdown should handle all errors
        await lifecycle_manager.shutdown()

        # Verify all cleanup was attempted
        provider.close.assert_called_once()
        code_manager.stop.assert_called_once()
        semantic_selector.close.assert_called_once()
