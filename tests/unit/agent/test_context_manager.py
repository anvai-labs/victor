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

"""Tests for ContextManager.

Tests the context management module including:
- ContextManagerConfig configuration
- ContextManager initialization and dependency injection
- Context window queries
- Context overflow detection
- Proactive compaction handling
- Context metrics retrieval
- Factory function create_context_manager
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from victor.agent.context_manager import (
    ContextManager,
    ContextManagerConfig,
    create_context_manager,
)
from victor.agent.conversation.controller import ContextMetrics


class TestContextManagerConfig:
    """Tests for ContextManagerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ContextManagerConfig()

        assert config.max_context_chars is None
        assert config.chars_per_token == 3.5
        assert config.safety_margin == 0.8
        assert config.default_context_window == 128000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ContextManagerConfig(
            max_context_chars=100000,
            chars_per_token=4.0,
            safety_margin=0.9,
            default_context_window=200000,
        )

        assert config.max_context_chars == 100000
        assert config.chars_per_token == 4.0
        assert config.safety_margin == 0.9
        assert config.default_context_window == 200000

    def test_partial_config(self):
        """Test partial custom configuration values."""
        config = ContextManagerConfig(
            max_context_chars=50000,
        )

        assert config.max_context_chars == 50000
        # Other fields should remain at defaults
        assert config.chars_per_token == 3.5
        assert config.safety_margin == 0.8
        assert config.default_context_window == 128000


class TestContextManagerInit:
    """Tests for ContextManager initialization."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock ConversationController."""
        controller = MagicMock()
        controller.get_context_metrics.return_value = ContextMetrics(
            char_count=50000,
            estimated_tokens=17000,
            message_count=10,
            is_overflow_risk=False,
            max_context_chars=200000,
        )
        return controller

    @pytest.fixture
    def config(self):
        """Create a default ContextManagerConfig."""
        return ContextManagerConfig()

    def test_init_with_required_params(self, config, mock_controller):
        """Test initialization with required parameters only."""
        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
        )

        assert manager._config is config
        assert manager._provider_name == "anthropic"
        assert manager._model == "claude-sonnet-4-20250514"
        assert manager._conversation_controller is mock_controller
        assert manager._context_compactor is None
        assert manager._debug_logger is None
        assert manager._settings is None

    def test_init_with_all_params(self, config, mock_controller):
        """Test initialization with all optional parameters."""
        mock_compactor = MagicMock()
        mock_debug_logger = MagicMock()
        mock_settings = MagicMock()

        manager = ContextManager(
            config=config,
            provider_name="openai",
            model="gpt-4",
            conversation_controller=mock_controller,
            context_compactor=mock_compactor,
            debug_logger=mock_debug_logger,
            settings=mock_settings,
        )

        assert manager._context_compactor is mock_compactor
        assert manager._debug_logger is mock_debug_logger
        assert manager._settings is mock_settings

    def test_config_property(self, config, mock_controller):
        """Test config property."""
        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
        )

        assert manager.config is config

    def test_provider_name_property(self, config, mock_controller):
        """Test provider_name property."""
        manager = ContextManager(
            config=config,
            provider_name="google",
            model="gemini-pro",
            conversation_controller=mock_controller,
        )

        assert manager.provider_name == "google"

    def test_model_property(self, config, mock_controller):
        """Test model property."""
        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-opus-4",
            conversation_controller=mock_controller,
        )

        assert manager.model == "claude-opus-4"


class TestContextManagerContextWindow:
    """Tests for context window queries."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock ConversationController."""
        return MagicMock()

    @pytest.fixture
    def config(self):
        """Create a default ContextManagerConfig."""
        return ContextManagerConfig()

    def test_get_model_context_window_success(self, config, mock_controller):
        """Test successful context window retrieval from provider limits."""
        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
        )

        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 200000
            mock_get_limits.return_value = mock_limits

            result = manager.get_model_context_window()

            assert result == 200000
            mock_get_limits.assert_called_once_with("anthropic", "claude-sonnet-4-20250514")

    def test_get_model_context_window_fallback_on_exception(self, config, mock_controller):
        """Test fallback to default on provider limits exception."""
        manager = ContextManager(
            config=config,
            provider_name="unknown",
            model="some-model",
            conversation_controller=mock_controller,
        )

        with patch(
            "victor.config.config_loaders.get_provider_limits",
            side_effect=Exception("Config not found"),
        ):
            result = manager.get_model_context_window()

            # Should fall back to default_context_window
            assert result == 128000

    def test_get_model_context_window_custom_default(self, mock_controller):
        """Test fallback to custom default context window."""
        config = ContextManagerConfig(default_context_window=256000)
        manager = ContextManager(
            config=config,
            provider_name="unknown",
            model="some-model",
            conversation_controller=mock_controller,
        )

        with patch(
            "victor.config.config_loaders.get_provider_limits",
            side_effect=ValueError("Not found"),
        ):
            result = manager.get_model_context_window()

            assert result == 256000


class TestContextManagerMaxContextChars:
    """Tests for max context chars calculation."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock ConversationController."""
        return MagicMock()

    def test_get_max_context_chars_from_settings(self, mock_controller):
        """Test max_context_chars from settings override."""
        config = ContextManagerConfig()
        mock_settings = MagicMock()
        mock_settings.max_context_chars = 150000

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            settings=mock_settings,
        )

        result = manager.get_max_context_chars()

        assert result == 150000

    def test_get_max_context_chars_settings_zero_ignored(self, mock_controller):
        """Test that zero value in settings is ignored."""
        config = ContextManagerConfig(max_context_chars=100000)
        mock_settings = MagicMock()
        mock_settings.max_context_chars = 0

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            settings=mock_settings,
        )

        result = manager.get_max_context_chars()

        # Should fall back to config.max_context_chars
        assert result == 100000

    def test_get_max_context_chars_settings_none_ignored(self, mock_controller):
        """Test that None value in settings is ignored."""
        config = ContextManagerConfig(max_context_chars=80000)
        mock_settings = MagicMock()
        mock_settings.max_context_chars = None

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            settings=mock_settings,
        )

        result = manager.get_max_context_chars()

        # Should fall back to config.max_context_chars
        assert result == 80000

    def test_get_max_context_chars_from_config(self, mock_controller):
        """Test max_context_chars from config override."""
        config = ContextManagerConfig(max_context_chars=120000)

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
        )

        result = manager.get_max_context_chars()

        assert result == 120000

    def test_get_max_context_chars_calculated_from_model(self, mock_controller):
        """Test max_context_chars calculated from model context window."""
        config = ContextManagerConfig(
            max_context_chars=None,  # No override
            chars_per_token=4.0,
            safety_margin=0.75,
        )

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
        )

        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 100000
            mock_get_limits.return_value = mock_limits

            result = manager.get_max_context_chars()

            # 100000 tokens * 4.0 chars/token * 0.75 safety = 300000
            assert result == 300000

    def test_get_max_context_chars_default_calculation(self, mock_controller):
        """Test default calculation with default config values."""
        config = ContextManagerConfig()  # All defaults

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
        )

        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = MagicMock()
            mock_limits.context_window = 128000
            mock_get_limits.return_value = mock_limits

            result = manager.get_max_context_chars()

            # 128000 tokens * 3.5 chars/token * 0.8 safety = 358400
            assert result == 358400

    def test_get_max_context_chars_no_settings_attribute(self, mock_controller):
        """Test handling when settings object lacks max_context_chars attribute."""
        config = ContextManagerConfig(max_context_chars=75000)
        mock_settings = MagicMock(spec=[])  # Empty spec, no attributes

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            settings=mock_settings,
        )

        result = manager.get_max_context_chars()

        # Should fall back to config.max_context_chars
        assert result == 75000


class TestContextManagerOverflowDetection:
    """Tests for context overflow detection."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock ConversationController."""
        return MagicMock()

    @pytest.fixture
    def config(self):
        """Create a default ContextManagerConfig."""
        return ContextManagerConfig(max_context_chars=200000)

    def test_check_context_overflow_no_risk(self, config, mock_controller):
        """Test overflow check when context is safe."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=50000,
            estimated_tokens=17000,
            message_count=10,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
        )

        result = manager.check_context_overflow()

        assert result is False
        mock_controller.get_context_metrics.assert_called_once()

    def test_check_context_overflow_at_risk(self, config, mock_controller):
        """Test overflow check when context is at risk."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=190000,
            estimated_tokens=65000,
            message_count=50,
            is_overflow_risk=True,
            max_context_chars=200000,
        )

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
        )

        result = manager.check_context_overflow()

        assert result is True

    def test_check_context_overflow_with_custom_max(self, config, mock_controller):
        """Test overflow check with custom max_context_chars."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=50000,
            estimated_tokens=17000,
            message_count=10,
            is_overflow_risk=False,
            max_context_chars=100000,
        )

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
        )

        result = manager.check_context_overflow(max_context_chars=100000)

        assert result is False

    def test_check_context_overflow_logs_to_debug_logger(self, config, mock_controller):
        """Test that overflow check logs to debug logger."""
        mock_debug_logger = MagicMock()
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=75000,
            estimated_tokens=25000,
            message_count=20,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            debug_logger=mock_debug_logger,
        )

        manager.check_context_overflow()

        mock_debug_logger.log_context_size.assert_called_once_with(75000, 25000)

    def test_check_context_overflow_no_debug_logger(self, config, mock_controller):
        """Test overflow check without debug logger (should not error)."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=50000,
            estimated_tokens=17000,
            message_count=10,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            debug_logger=None,
        )

        # Should not raise
        result = manager.check_context_overflow()
        assert result is False

    def test_check_context_overflow_logs_warning_on_risk(self, config, mock_controller, caplog):
        """Test that overflow risk logs a warning."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=195000,
            estimated_tokens=67000,
            message_count=55,
            is_overflow_risk=True,
            max_context_chars=200000,
        )

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
        )

        import logging

        with caplog.at_level(logging.WARNING):
            result = manager.check_context_overflow()

        assert result is True
        assert "Context overflow risk" in caplog.text


class TestContextManagerCompaction:
    """Tests for proactive compaction handling."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock ConversationController."""
        controller = MagicMock()
        controller.inject_compaction_context.return_value = None
        return controller

    @pytest.fixture
    def config(self):
        """Create a default ContextManagerConfig."""
        return ContextManagerConfig()

    def test_handle_compaction_no_compactor(self, config, mock_controller):
        """Test handle_compaction returns None when no compactor."""
        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            context_compactor=None,
        )

        result = manager.handle_compaction("test query")

        assert result is None

    def test_handle_compaction_no_action_taken(self, config, mock_controller):
        """Test handle_compaction when no compaction is needed."""
        mock_compactor = MagicMock()
        mock_action = MagicMock()
        mock_action.action_taken = False
        mock_compactor.check_and_compact.return_value = mock_action

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            context_compactor=mock_compactor,
        )

        result = manager.handle_compaction("test query")

        assert result is None
        mock_compactor.check_and_compact.assert_called_once_with(current_query="test query")

    def test_handle_compaction_action_taken_no_messages_removed(self, config, mock_controller):
        """Test handle_compaction when action taken but no messages removed."""
        mock_compactor = MagicMock()
        mock_action = MagicMock()
        mock_action.action_taken = True
        mock_action.messages_removed = 0
        mock_action.chars_freed = 1000
        mock_action.trigger.value = "threshold"
        mock_compactor.check_and_compact.return_value = mock_action

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            context_compactor=mock_compactor,
        )

        result = manager.handle_compaction("test query")

        # No messages removed, so no StreamChunk returned
        assert result is None
        # inject_compaction_context should NOT be called
        mock_controller.inject_compaction_context.assert_not_called()

    def test_handle_compaction_messages_removed(self, config, mock_controller):
        """Test handle_compaction when messages are removed."""
        mock_compactor = MagicMock()
        mock_action = MagicMock()
        mock_action.action_taken = True
        mock_action.messages_removed = 5
        mock_action.chars_freed = 15000
        mock_action.trigger.value = "threshold"
        mock_compactor.check_and_compact.return_value = mock_action

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            context_compactor=mock_compactor,
        )

        result = manager.handle_compaction("test query")

        # Should return a StreamChunk with compaction notification
        assert result is not None
        assert "Proactively compacted history" in result.content
        assert "5 messages" in result.content
        assert "15,000 chars freed" in result.content

        # inject_compaction_context should be called
        mock_controller.inject_compaction_context.assert_called_once()

    def test_handle_compaction_overflow_trigger(self, config, mock_controller):
        """Test handle_compaction with overflow trigger."""
        mock_compactor = MagicMock()
        mock_action = MagicMock()
        mock_action.action_taken = True
        mock_action.messages_removed = 10
        mock_action.chars_freed = 50000
        mock_action.trigger.value = "overflow"
        mock_compactor.check_and_compact.return_value = mock_action

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            context_compactor=mock_compactor,
        )

        result = manager.handle_compaction("emergency query")

        assert result is not None
        assert "10 messages" in result.content
        assert "50,000 chars freed" in result.content

    def test_handle_compaction_prefers_context_service_before_legacy_compactor(
        self, config, mock_controller
    ):
        """Sync compaction should use ContextService before legacy compactor."""
        context_service = MagicMock()
        context_service.get_compaction_recommendation.return_value = {"should_compact": True}
        context_service.compact_context.return_value = 4
        mock_compactor = MagicMock()

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            context_compactor=mock_compactor,
            context_service=context_service,
        )

        result = manager.handle_compaction("test query")

        assert result is not None
        assert "4 messages" in result.content
        assert "ContextService" in result.content
        context_service.get_compaction_recommendation.assert_called_once()
        context_service.compact_context.assert_called_once_with(
            strategy="tiered",
            min_messages=6,
        )
        mock_compactor.check_and_compact.assert_not_called()
        mock_controller.inject_compaction_context.assert_called_once()

    def test_handle_compaction_uses_context_service_noop_decision(self, config, mock_controller):
        """ContextService no-op decisions should not fall through to legacy compactor."""
        context_service = MagicMock()
        context_service.get_compaction_recommendation.return_value = {"should_compact": False}
        context_service.compact_context.return_value = 0
        mock_compactor = MagicMock()

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            context_compactor=mock_compactor,
            context_service=context_service,
        )

        result = manager.handle_compaction("test query")

        assert result is None
        context_service.get_compaction_recommendation.assert_called_once()
        context_service.compact_context.assert_not_called()
        mock_compactor.check_and_compact.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_compaction_async_prefers_context_service_before_legacy_compactor(
        self, config, mock_controller
    ):
        """Async compaction should use ContextService before legacy compactor."""
        context_service = MagicMock()
        context_service.get_compaction_recommendation.return_value = {"should_compact": True}
        context_service.compact_context = AsyncMock(return_value=4)
        mock_compactor = MagicMock()
        mock_compactor.check_and_compact_async = AsyncMock()

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            context_compactor=mock_compactor,
            context_service=context_service,
        )

        result = await manager.handle_compaction_async("test query")

        assert result is not None
        assert "4 messages" in result.content
        assert "ContextService" in result.content
        context_service.get_compaction_recommendation.assert_called_once()
        context_service.compact_context.assert_awaited_once_with(
            strategy="tiered",
            min_messages=6,
        )
        mock_compactor.check_and_compact_async.assert_not_called()
        mock_controller.inject_compaction_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_compaction_async_uses_context_service_noop_decision(
        self, config, mock_controller
    ):
        """ContextService no-op decisions should not fall through to legacy compactor."""
        context_service = MagicMock()
        context_service.get_compaction_recommendation.return_value = {"should_compact": False}
        context_service.compact_context = AsyncMock(return_value=0)
        mock_compactor = MagicMock()
        mock_compactor.check_and_compact_async = AsyncMock()

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            context_compactor=mock_compactor,
            context_service=context_service,
        )

        result = await manager.handle_compaction_async("test query")

        assert result is None
        context_service.get_compaction_recommendation.assert_called_once()
        context_service.compact_context.assert_not_called()
        mock_compactor.check_and_compact_async.assert_not_called()


class TestContextManagerMetrics:
    """Tests for context metrics retrieval."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock ConversationController."""
        return MagicMock()

    @pytest.fixture
    def config(self):
        """Create a default ContextManagerConfig."""
        return ContextManagerConfig()

    def test_get_context_metrics(self, config, mock_controller):
        """Test get_context_metrics delegates to controller."""
        expected_metrics = ContextMetrics(
            char_count=100000,
            estimated_tokens=35000,
            message_count=25,
            is_overflow_risk=False,
            max_context_chars=200000,
        )
        mock_controller.get_context_metrics.return_value = expected_metrics

        manager = ContextManager(
            config=config,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
        )

        result = manager.get_context_metrics()

        assert result is expected_metrics
        mock_controller.get_context_metrics.assert_called_once()


class TestCreateContextManager:
    """Tests for create_context_manager factory function."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock ConversationController."""
        return MagicMock()

    def test_create_with_defaults(self, mock_controller):
        """Test factory with default config."""
        manager = create_context_manager(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
        )

        assert isinstance(manager, ContextManager)
        assert manager.provider_name == "anthropic"
        assert manager.model == "claude-sonnet-4-20250514"
        assert manager.config.max_context_chars is None
        assert manager.config.chars_per_token == 3.5

    def test_create_with_custom_config(self, mock_controller):
        """Test factory with custom config."""
        custom_config = ContextManagerConfig(
            max_context_chars=150000,
            chars_per_token=4.0,
        )

        manager = create_context_manager(
            provider_name="openai",
            model="gpt-4",
            conversation_controller=mock_controller,
            config=custom_config,
        )

        assert manager.config is custom_config
        assert manager.config.max_context_chars == 150000
        assert manager.config.chars_per_token == 4.0

    def test_create_with_all_dependencies(self, mock_controller):
        """Test factory with all optional dependencies."""
        mock_compactor = MagicMock()
        mock_debug_logger = MagicMock()
        mock_settings = MagicMock()

        manager = create_context_manager(
            provider_name="google",
            model="gemini-pro",
            conversation_controller=mock_controller,
            context_compactor=mock_compactor,
            debug_logger=mock_debug_logger,
            settings=mock_settings,
        )

        assert manager._context_compactor is mock_compactor
        assert manager._debug_logger is mock_debug_logger
        assert manager._settings is mock_settings

    def test_create_without_optional_dependencies(self, mock_controller):
        """Test factory without optional dependencies."""
        manager = create_context_manager(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            conversation_controller=mock_controller,
            context_compactor=None,
            debug_logger=None,
            settings=None,
        )

        assert manager._context_compactor is None
        assert manager._debug_logger is None
        assert manager._settings is None
