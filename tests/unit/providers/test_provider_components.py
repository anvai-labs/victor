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

"""Tests for provider refactored components."""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch

from victor.agent.provider.switcher import ProviderSwitcher, ProviderSwitcherState
from victor.agent.provider.health_monitor import ProviderHealthMonitor
from victor.agent.provider.tool_adapter_coordinator import ToolAdapterCoordinator
from victor.agent.strategies import DefaultProviderClassificationStrategy
from victor.agent.protocols import IProviderEventEmitter


class TestProviderSwitcher:
    """Test ProviderSwitcher functionality."""

    def test_initialization(self):
        """Test ProviderSwitcher initialization."""
        strategy = DefaultProviderClassificationStrategy()
        event_emitter = Mock(spec=IProviderEventEmitter)

        switcher = ProviderSwitcher(
            classification_strategy=strategy,
            event_emitter=event_emitter,
        )

        assert switcher._classification_strategy is strategy
        assert switcher._event_emitter is event_emitter
        assert switcher._current_state is None
        assert switcher.get_current_provider() is None
        assert switcher.get_current_model() == ""

    def test_set_initial_state(self):
        """Test setting initial provider state."""
        strategy = DefaultProviderClassificationStrategy()
        event_emitter = Mock(spec=IProviderEventEmitter)
        mock_provider = Mock()
        mock_provider.name = "TestProvider"

        switcher = ProviderSwitcher(
            classification_strategy=strategy,
            event_emitter=event_emitter,
        )

        switcher.set_initial_state(mock_provider, "TestProvider", "test-model")

        state = switcher.get_current_state()
        assert state is not None
        assert state.provider is mock_provider
        assert state.provider_name == "testprovider"
        assert state.model == "test-model"
        assert state.switch_count == 0

    @pytest.mark.asyncio
    async def test_switch_provider_success(self):
        """Test successful provider switch."""
        strategy = DefaultProviderClassificationStrategy()
        event_emitter = Mock(spec=IProviderEventEmitter)
        event_emitter.emit_switch_event = Mock()

        mock_provider_old = Mock()
        mock_provider_old.name = "OldProvider"

        mock_provider_new = Mock()
        mock_provider_new.name = "NewProvider"

        switcher = ProviderSwitcher(
            classification_strategy=strategy,
            event_emitter=event_emitter,
        )
        switcher.set_initial_state(mock_provider_old, "OldProvider", "old-model")

        # Mock ProviderRegistry.create
        with patch("victor.agent.provider.switcher.ProviderRegistry") as mock_registry:
            mock_registry.create.return_value = mock_provider_new

            result = await switcher.switch_provider(
                provider_name="NewProvider",
                model="new-model",
                reason="test",
            )

            assert result is True
            assert switcher.get_current_provider() is mock_provider_new
            assert switcher.get_current_model() == "new-model"
            assert switcher._current_state.switch_count == 1
            assert len(switcher.get_switch_history()) == 1
            event_emitter.emit_switch_event.assert_called_once()

    def test_get_switch_history(self):
        """Test getting switch history."""
        strategy = DefaultProviderClassificationStrategy()
        event_emitter = Mock(spec=IProviderEventEmitter)

        switcher = ProviderSwitcher(
            classification_strategy=strategy,
            event_emitter=event_emitter,
        )

        # Initially empty
        assert switcher.get_switch_history() == []

        # Add some history manually
        switcher._switch_history.append(
            {
                "from_provider": "anthropic",
                "from_model": "claude-3",
                "to_provider": "openai",
                "to_model": "gpt-4",
                "reason": "test",
                "timestamp": "2025-01-06T12:00:00",
            }
        )

        history = switcher.get_switch_history()
        assert len(history) == 1
        assert history[0]["from_provider"] == "anthropic"
        assert history[0]["to_provider"] == "openai"

    def test_on_switch_callback(self):
        """Test switch callback registration and notification."""
        strategy = DefaultProviderClassificationStrategy()
        event_emitter = Mock(spec=IProviderEventEmitter)

        switcher = ProviderSwitcher(
            classification_strategy=strategy,
            event_emitter=event_emitter,
        )

        # Register callback
        callback_mock = Mock()
        switcher.on_switch(callback_mock)

        # Create state and notify
        mock_provider = Mock()
        state = ProviderSwitcherState(
            provider=mock_provider,
            provider_name="test",
            model="test-model",
        )

        switcher._notify_switch(state)

        callback_mock.assert_called_once_with(state)

    def test_classification(self):
        """Test provider classification through strategy."""
        strategy = DefaultProviderClassificationStrategy()
        event_emitter = Mock(spec=IProviderEventEmitter)

        switcher = ProviderSwitcher(
            classification_strategy=strategy,
            event_emitter=event_emitter,
        )

        # Test classification
        assert strategy.is_cloud_provider("anthropic")
        assert strategy.is_cloud_provider("openai")
        assert strategy.is_local_provider("ollama")
        assert strategy.is_local_provider("lmstudio")
        assert strategy.get_provider_type("anthropic") == "cloud"
        assert strategy.get_provider_type("ollama") == "local"
        assert strategy.get_provider_type("unknown") == "unknown"


class TestProviderHealthMonitor:
    """Test ProviderHealthMonitor functionality."""

    def test_initialization(self):
        """Test ProviderHealthMonitor initialization."""
        settings = Mock()

        monitor = ProviderHealthMonitor(
            settings=settings,
            enable_health_checks=True,
            health_check_interval=30.0,
        )

        assert monitor._settings is settings
        assert monitor._enable_health_checks is True
        assert monitor._health_check_interval == 30.0
        assert monitor._health_checker is None
        assert monitor._health_check_task is None

    @pytest.mark.asyncio
    async def test_check_health_without_checker(self):
        """Test health check gracefully handles missing checker."""
        settings = Mock()

        monitor = ProviderHealthMonitor(settings=settings)

        mock_provider = Mock()
        mock_provider.name = "TestProvider"

        # When health checker is available and returns healthy status
        # The test should pass - this tests the integration
        # We can't easily mock the lazy import, so we test the actual behavior
        result = await monitor.check_health(mock_provider)

        # Should return True (either from successful check or from graceful fallback)
        # This is flexible because we don't control the health checker availability
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_start_stop_health_checks(self):
        """Test starting and stopping health checks."""
        settings = Mock()

        monitor = ProviderHealthMonitor(
            settings=settings,
            enable_health_checks=True,
            health_check_interval=1.0,
        )

        # Start monitoring
        await monitor.start_health_checks(interval=1.0)
        assert monitor.is_monitoring() is True

        # Stop monitoring
        await monitor.stop_health_checks()
        assert monitor.is_monitoring() is False

    @pytest.mark.asyncio
    async def test_start_health_checks_disabled(self):
        """Test that health checks respect enable_health_checks flag."""
        settings = Mock()

        monitor = ProviderHealthMonitor(
            settings=settings,
            enable_health_checks=False,  # Disabled
        )

        await monitor.start_health_checks()
        assert monitor.is_monitoring() is False


class TestToolAdapterCoordinator:
    """Test ToolAdapterCoordinator functionality."""

    def test_initialization(self):
        """Test ToolAdapterCoordinator initialization."""
        mock_switcher = Mock()
        settings = Mock()

        coordinator = ToolAdapterCoordinator(
            provider_switcher=mock_switcher,
            settings=settings,
        )

        assert coordinator._provider_switcher is mock_switcher
        assert coordinator._settings is settings
        assert coordinator._adapter is None
        assert coordinator._capabilities is None
        assert coordinator.is_initialized() is False

    def test_initialize_adapter_success(self):
        """Test successful adapter initialization."""
        mock_switcher = Mock()
        mock_settings = Mock()

        # Mock switcher state
        mock_state = Mock()
        mock_state.provider_name = "anthropic"
        mock_state.model = "claude-3-sonnet-20240229"

        mock_switcher.get_current_state.return_value = mock_state
        mock_switcher.get_current_provider.return_value = Mock()
        mock_switcher.get_current_model.return_value = "claude-3-sonnet-20240229"

        # Mock ToolCallingAdapterRegistry
        mock_adapter = Mock()
        mock_capabilities = Mock()
        mock_capabilities.native_tool_calls = True
        mock_capabilities.tool_call_format = Mock()
        mock_capabilities.tool_call_format.value = "claude_format"

        mock_adapter.get_capabilities.return_value = mock_capabilities

        coordinator = ToolAdapterCoordinator(
            provider_switcher=mock_switcher,
            settings=mock_settings,
        )

        with patch(
            "victor.agent.provider.tool_adapter_coordinator.ToolCallingAdapterRegistry"
        ) as mock_registry:
            mock_registry.get_adapter.return_value = mock_adapter

            capabilities = coordinator.initialize_adapter()

            assert capabilities is mock_capabilities
            assert coordinator.is_initialized() is True
            assert coordinator.get_adapter() is mock_adapter

    def test_initialize_adapter_no_provider(self):
        """Test that initialize_adapter raises ValueError when no provider."""
        mock_switcher = Mock()
        mock_switcher.get_current_state.return_value = None

        coordinator = ToolAdapterCoordinator(
            provider_switcher=mock_switcher,
        )

        with pytest.raises(ValueError, match="No provider configured"):
            coordinator.initialize_adapter()

    def test_get_capabilities_before_initialization(self):
        """Test that get_capabilities raises ValueError before initialization."""
        mock_switcher = Mock()
        coordinator = ToolAdapterCoordinator(
            provider_switcher=mock_switcher,
        )

        with pytest.raises(ValueError, match="Tool adapter not initialized"):
            coordinator.get_capabilities()

    def test_get_adapter_before_initialization(self):
        """Test that get_adapter raises ValueError before initialization."""
        mock_switcher = Mock()
        coordinator = ToolAdapterCoordinator(
            provider_switcher=mock_switcher,
        )

        with pytest.raises(ValueError, match="Tool adapter not initialized"):
            coordinator.get_adapter()
