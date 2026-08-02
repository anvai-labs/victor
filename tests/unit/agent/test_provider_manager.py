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

"""Tests for ProviderManager component.

Tests cover:
- Configuration
- Provider switching
- Model switching
- Health monitoring
- Fallback handling
- Switch history tracking
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from victor.agent.provider_manager import (
    ProviderManager,
    ProviderManagerConfig,
    ProviderState,
)


class TestProviderManagerConfig:
    """Tests for ProviderManagerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProviderManagerConfig()

        assert config.enable_health_checks is True
        assert config.health_check_interval == 60.0
        assert config.auto_fallback is True
        assert config.fallback_providers == []
        assert config.max_fallback_attempts == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProviderManagerConfig(
            enable_health_checks=False,
            health_check_interval=30.0,
            auto_fallback=False,
            fallback_providers=["anthropic", "openai"],
            max_fallback_attempts=5,
        )

        assert config.enable_health_checks is False
        assert config.health_check_interval == 30.0
        assert config.auto_fallback is False
        assert "anthropic" in config.fallback_providers
        assert config.max_fallback_attempts == 5


class TestProviderState:
    """Tests for ProviderState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        provider = MagicMock()
        state = ProviderState(
            provider=provider,
            provider_name="test",
            model="test-model",
        )

        assert state.provider is provider
        assert state.provider_name == "test"
        assert state.model == "test-model"
        assert state.is_healthy is True
        assert state.last_error is None


class TestProviderManagerInit:
    """Tests for ProviderManager initialization."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.get_provider_settings.return_value = {}
        return settings

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        provider = MagicMock()
        provider.name = "test_provider"
        provider.supports_tools.return_value = True
        provider.close = AsyncMock()
        return provider

    def test_initialization_without_provider(self, mock_settings):
        """Test initialization without initial provider."""
        manager = ProviderManager(settings=mock_settings)

        assert manager.provider is None
        assert manager.provider_name == ""
        assert manager.model == ""

    def test_initialization_with_provider(self, mock_settings, mock_provider):
        """Test initialization with provider."""
        manager = ProviderManager(
            settings=mock_settings,
            initial_provider=mock_provider,
            initial_model="test-model",
            provider_name="anthropic",
        )

        assert manager.provider is mock_provider
        assert manager.provider_name == "anthropic"
        assert manager.model == "test-model"

    def test_initialization_with_config(self, mock_settings, mock_provider):
        """Test initialization with custom config."""
        config = ProviderManagerConfig(enable_health_checks=False)

        manager = ProviderManager(
            settings=mock_settings,
            initial_provider=mock_provider,
            initial_model="test-model",
            config=config,
        )

        assert manager.config.enable_health_checks is False


class TestProviderTypeChecks:
    """Tests for provider type detection."""

    @pytest.fixture
    def manager(self):
        """Create manager with mock provider."""
        settings = MagicMock()
        provider = MagicMock()

        return ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="anthropic",
        )

    def test_is_cloud_provider_true(self, manager):
        """Test cloud provider detection."""
        assert manager.is_cloud_provider() is True

    def test_is_cloud_provider_false(self):
        """Test non-cloud provider detection."""
        settings = MagicMock()
        provider = MagicMock()

        manager = ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="ollama",
        )

        assert manager.is_cloud_provider() is False

    def test_is_local_provider_true(self):
        """Test local provider detection."""
        settings = MagicMock()
        provider = MagicMock()

        manager = ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="ollama",
        )

        assert manager.is_local_provider() is True

    def test_is_local_provider_false(self, manager):
        """Test non-local provider detection."""
        assert manager.is_local_provider() is False


class TestContextWindow:
    """Tests for context window detection."""

    @pytest.fixture
    def manager(self):
        """Create manager with mock provider."""
        settings = MagicMock()
        provider = MagicMock()

        return ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="anthropic",
        )

    def test_context_window_from_provider(self, manager):
        """Test context window from provider method."""
        manager.provider.get_context_window = MagicMock(return_value=150000)

        window = manager.get_context_window()

        assert window == 150000

    def test_context_window_fallback(self, manager):
        """Test context window fallback."""
        # Remove the method
        if hasattr(manager.provider, "get_context_window"):
            del manager.provider.get_context_window

        window = manager.get_context_window()

        # Should return anthropic default
        assert window == 200000

    def test_context_window_unknown_provider(self):
        """Test context window for unknown provider."""
        settings = MagicMock()
        provider = MagicMock()
        # Configure mock to raise exception (simulating no get_context_window method)
        provider.get_context_window.side_effect = AttributeError("no method")

        manager = ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="unknown_provider",
        )

        window = manager.get_context_window()

        # Should return default from ProviderLimits (128000)
        assert window == 128000


class TestToolAdapter:
    """Tests for tool adapter initialization."""

    @pytest.fixture
    def manager(self):
        """Create manager with mock provider."""
        settings = MagicMock()
        provider = MagicMock()

        return ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="anthropic",
        )

    @patch("victor.agent.provider.tool_adapter_coordinator.ToolCallingAdapterRegistry")
    def test_initialize_tool_adapter(self, mock_registry, manager):
        """Test tool adapter initialization."""
        mock_adapter = MagicMock()
        mock_caps = MagicMock()
        mock_caps.native_tool_calls = True
        mock_caps.tool_call_format = MagicMock()
        mock_caps.tool_call_format.value = "native"
        mock_adapter.get_capabilities.return_value = mock_caps
        mock_registry.get_adapter.return_value = mock_adapter

        caps = manager.initialize_tool_adapter()

        mock_registry.get_adapter.assert_called_once()
        assert caps is mock_caps
        assert manager.capabilities is mock_caps
        assert manager.tool_adapter is mock_adapter


class TestProviderSwitching:
    """Tests for provider switching."""

    @pytest.fixture
    def manager(self):
        """Create manager with mock provider."""
        settings = MagicMock()
        settings.get_provider_settings.return_value = {}
        provider = MagicMock()
        provider.supports_tools.return_value = True

        return ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="anthropic",
            config=ProviderManagerConfig(enable_health_checks=False),
        )

    @pytest.mark.asyncio
    @patch("victor.agent.provider.switcher.ProviderRegistry")
    @patch("victor.agent.provider_manager.ToolCallingAdapterRegistry")
    async def test_switch_provider_success(
        self, mock_adapter_registry, mock_provider_registry, manager
    ):
        """Test successful provider switch."""
        # Mock new provider with async discover_capabilities
        new_provider = MagicMock()
        new_provider.supports_tools.return_value = True
        new_provider.supports_streaming.return_value = True
        new_provider.discover_capabilities = AsyncMock(
            return_value=MagicMock(
                provider="openai",
                model="gpt-4-turbo",
                context_window=128000,
                supports_tools=True,
                supports_streaming=True,
                source="runtime",
            )
        )
        mock_provider_registry.create.return_value = new_provider

        # Mock adapter
        mock_adapter = MagicMock()
        mock_caps = MagicMock()
        mock_caps.native_tool_calls = True
        mock_caps.tool_call_format = MagicMock()
        mock_caps.tool_call_format.value = "native"
        mock_adapter.get_capabilities.return_value = mock_caps
        mock_adapter_registry.get_adapter.return_value = mock_adapter

        # Mock health check to pass
        with patch.object(
            manager._provider_switcher._health_monitor,
            "check_health",
            return_value=True,
        ):
            result = await manager.switch_provider("openai", "gpt-4-turbo")

            assert result is True
            assert manager.provider_name == "openai"
            assert manager.model == "gpt-4-turbo"

    @pytest.mark.asyncio
    @patch("victor.agent.provider.switcher.ProviderRegistry")
    async def test_switch_provider_failure(self, mock_registry, manager):
        """Test failed provider switch."""
        mock_registry.create.side_effect = Exception("Connection failed")

        result = await manager.switch_provider("openai")

        assert result is False
        # Should remain on original provider
        assert manager.provider_name == "anthropic"

    @pytest.mark.asyncio
    @patch("victor.agent.provider.switcher.ProviderRegistry")
    @patch("victor.agent.provider_manager.ToolCallingAdapterRegistry")
    async def test_switch_provider_with_fallback(
        self, mock_adapter_registry, mock_provider_registry
    ):
        """Test provider switch with fallback on health check failure."""
        settings = MagicMock()
        settings.get_provider_settings.return_value = {}
        provider = MagicMock()
        provider.discover_capabilities = AsyncMock(return_value=MagicMock())

        config = ProviderManagerConfig(
            enable_health_checks=True,
            auto_fallback=True,
            fallback_providers=["openai"],
        )

        manager = ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="anthropic",
            config=config,
        )

        # Mock providers with async discover_capabilities
        new_provider = MagicMock()
        new_provider.supports_tools.return_value = True
        new_provider.supports_streaming.return_value = True
        new_provider.discover_capabilities = AsyncMock(return_value=MagicMock())
        mock_provider_registry.create.return_value = new_provider

        # Mock adapter
        mock_adapter = MagicMock()
        mock_caps = MagicMock()
        mock_caps.native_tool_calls = True
        mock_caps.tool_call_format = MagicMock()
        mock_caps.tool_call_format.value = "native"
        mock_adapter.get_capabilities.return_value = mock_caps
        mock_adapter_registry.get_adapter.return_value = mock_adapter

        # Mock health check to fail then succeed (for fallback)
        health_check_results = [False, True]  # First fails, second succeeds

        async def mock_check_health(provider):
            return health_check_results.pop(0)

        with patch.object(
            manager._provider_switcher._health_monitor,
            "check_health",
            side_effect=mock_check_health,
        ):
            result = await manager.switch_provider("google")

        # Should have fallen back
        assert result is True


class TestModelSwitching:
    """Tests for model switching."""

    @pytest.fixture
    def manager(self):
        """Create manager with mock provider."""
        settings = MagicMock()
        provider = MagicMock()
        provider.supports_tools.return_value = True
        provider.supports_streaming.return_value = True
        provider.discover_capabilities = AsyncMock(return_value=MagicMock())

        return ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="anthropic",
        )

    @pytest.mark.asyncio
    @patch("victor.agent.provider_manager.ToolCallingAdapterRegistry")
    async def test_switch_model_success(self, mock_registry, manager):
        """Test successful model switch."""
        mock_adapter = MagicMock()
        mock_caps = MagicMock()
        mock_caps.native_tool_calls = True
        mock_caps.tool_call_format = MagicMock()
        mock_caps.tool_call_format.value = "native"
        mock_adapter.get_capabilities.return_value = mock_caps
        mock_registry.get_adapter.return_value = mock_adapter

        result = await manager.switch_model("claude-opus-4-5-20251101")

        assert result is True
        assert manager.model == "claude-opus-4-5-20251101"
        assert manager.provider_name == "anthropic"  # Same provider

    @pytest.mark.asyncio
    async def test_switch_model_no_provider(self):
        """Test model switch without provider."""
        settings = MagicMock()
        manager = ProviderManager(settings=settings)

        result = await manager.switch_model("test-model")

        assert result is False


class TestGetInfo:
    """Tests for getting provider info."""

    @pytest.fixture
    def manager(self):
        """Create manager with mock provider."""
        settings = MagicMock()
        provider = MagicMock()
        provider.supports_tools.return_value = True

        return ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="anthropic",
        )

    def test_get_info_with_provider(self, manager):
        """Test getting info with provider."""
        info = manager.get_info()

        assert info["provider"] == "anthropic"
        assert info["model"] == "test-model"
        assert info["supports_tools"] is True
        assert info["is_cloud"] is True
        assert info["is_local"] is False
        assert info["is_healthy"] is True

    def test_get_info_without_provider(self):
        """Test getting info without provider."""
        settings = MagicMock()
        manager = ProviderManager(settings=settings)

        info = manager.get_info()

        assert info["provider"] is None
        assert info["model"] is None


class TestSwitchHistory:
    """Tests for switch history tracking."""

    @pytest.fixture
    def manager(self):
        """Create manager with mock provider."""
        settings = MagicMock()
        settings.get_provider_settings.return_value = {}
        provider = MagicMock()
        provider.supports_tools.return_value = True
        provider.supports_streaming.return_value = True
        provider.discover_capabilities = AsyncMock(return_value=MagicMock())

        return ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="anthropic",
            config=ProviderManagerConfig(enable_health_checks=False),
        )

    @pytest.mark.asyncio
    @patch("victor.agent.provider.switcher.ProviderRegistry")
    @patch("victor.agent.provider_manager.ToolCallingAdapterRegistry")
    async def test_switch_history_recorded(
        self, mock_adapter_registry, mock_provider_registry, manager
    ):
        """Test that switch history is recorded."""
        new_provider = MagicMock()
        new_provider.supports_tools.return_value = True
        new_provider.supports_streaming.return_value = True
        new_provider.discover_capabilities = AsyncMock(return_value=MagicMock())
        mock_provider_registry.create.return_value = new_provider

        mock_adapter = MagicMock()
        mock_caps = MagicMock()
        mock_caps.native_tool_calls = True
        mock_caps.tool_call_format = MagicMock()
        mock_caps.tool_call_format.value = "native"
        mock_adapter.get_capabilities.return_value = mock_caps
        mock_adapter_registry.get_adapter.return_value = mock_adapter

        # Mock health check to pass
        with patch.object(
            manager._provider_switcher._health_monitor,
            "check_health",
            return_value=True,
        ):
            await manager.switch_provider("openai", "gpt-4")

        history = manager.get_switch_history()
        assert len(history) >= 1


class TestSwitchCallbacks:
    """Tests for switch callbacks."""

    @pytest.fixture
    def manager(self):
        """Create manager with mock provider."""
        settings = MagicMock()
        settings.get_provider_settings.return_value = {}
        provider = MagicMock()
        provider.supports_tools.return_value = True
        provider.supports_streaming.return_value = True
        provider.discover_capabilities = AsyncMock(return_value=MagicMock())

        return ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="anthropic",
            config=ProviderManagerConfig(enable_health_checks=False),
        )

    @pytest.mark.asyncio
    @patch("victor.agent.provider.switcher.ProviderRegistry")
    @patch("victor.agent.provider.tool_adapter_coordinator.ToolCallingAdapterRegistry")
    async def test_callback_called_on_switch(
        self, mock_adapter_registry, mock_provider_registry, manager
    ):
        """Test that callbacks are called on switch."""
        new_provider = MagicMock()
        new_provider.supports_tools.return_value = True
        new_provider.supports_streaming.return_value = True
        new_provider.discover_capabilities = AsyncMock(return_value=MagicMock())
        mock_provider_registry.create.return_value = new_provider

        mock_adapter = MagicMock()
        mock_caps = MagicMock()
        mock_caps.native_tool_calls = True
        mock_caps.tool_call_format = MagicMock()
        mock_caps.tool_call_format.value = "native"
        mock_adapter.get_capabilities.return_value = mock_caps
        mock_adapter_registry.get_adapter.return_value = mock_adapter

        callback_calls = []

        def callback(state):
            callback_calls.append(state)

        manager.add_switch_callback(callback)

        # Mock health check to pass
        with patch.object(
            manager._provider_switcher._health_monitor,
            "check_health",
            return_value=True,
        ):
            await manager.switch_provider("openai", "gpt-4")

        assert len(callback_calls) == 1
        assert callback_calls[0].provider_name == "openai"


class TestHealthMonitoring:
    """Tests for health monitoring."""

    @pytest.fixture
    def manager(self):
        """Create manager with mock provider."""
        settings = MagicMock()
        provider = MagicMock()
        provider.health_check = AsyncMock(return_value=True)

        return ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="anthropic",
            config=ProviderManagerConfig(
                enable_health_checks=True,
                health_check_interval=0.1,
            ),
        )

    @pytest.mark.asyncio
    async def test_start_health_monitoring(self, manager):
        """Test starting health monitoring."""
        await manager.start_health_monitoring()

        assert manager._health_check_task is not None

        await manager.stop_health_monitoring()

    @pytest.mark.asyncio
    async def test_stop_health_monitoring(self, manager):
        """Test stopping health monitoring."""
        await manager.start_health_monitoring()
        await manager.stop_health_monitoring()

        assert manager._health_check_task is None


class TestCleanup:
    """Tests for cleanup."""

    @pytest.fixture
    def manager(self):
        """Create manager with mock provider."""
        settings = MagicMock()
        provider = MagicMock()
        provider.close = AsyncMock()

        return ProviderManager(
            settings=settings,
            initial_provider=provider,
            initial_model="test-model",
            provider_name="anthropic",
        )

    @pytest.mark.asyncio
    async def test_close(self, manager):
        """Test closing provider manager."""
        await manager.close()

        manager.provider.close.assert_called_once()
