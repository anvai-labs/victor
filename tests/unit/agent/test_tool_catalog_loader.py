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

"""Unit tests for ToolCatalogLoader component.

Tests the SRP-compliant tool catalog loading functionality.
"""

import pytest
from unittest.mock import MagicMock, patch

from victor.agent.tool_catalog_loader import (
    ToolCatalogLoader,
    ToolCatalogConfig,
    CatalogLoadResult,
)


class TestToolCatalogConfig:
    """Tests for ToolCatalogConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ToolCatalogConfig()

        assert config.airgapped_mode is False
        assert config.enabled_tools == []
        assert config.disabled_tools == []

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ToolCatalogConfig(
            airgapped_mode=True,
            enabled_tools=["tool1", "tool2"],
            disabled_tools=["tool3"],
        )

        assert config.airgapped_mode is True
        assert config.enabled_tools == ["tool1", "tool2"]
        assert config.disabled_tools == ["tool3"]


class TestCatalogLoadResult:
    """Tests for CatalogLoadResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = CatalogLoadResult()

        assert result.tools_loaded == 0
        assert result.tools_disabled == 0
        assert result.errors == []

    def test_custom_values(self):
        """Test custom result values."""
        result = CatalogLoadResult(
            tools_loaded=10,
            tools_disabled=2,
            errors=["error1"],
        )

        assert result.tools_loaded == 10
        assert result.tools_disabled == 2
        assert result.errors == ["error1"]


class TestToolCatalogLoaderInit:
    """Tests for ToolCatalogLoader initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default config."""
        registry = MagicMock()
        settings = MagicMock()

        loader = ToolCatalogLoader(registry=registry, settings=settings)

        assert loader._registry is registry
        assert loader._settings is settings
        assert loader._config.airgapped_mode is False
        assert loader.is_loaded is False

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config."""
        registry = MagicMock()
        settings = MagicMock()
        config = ToolCatalogConfig(airgapped_mode=True)

        loader = ToolCatalogLoader(registry=registry, settings=settings, config=config)

        assert loader._config.airgapped_mode is True


class TestToolCatalogLoaderLoad:
    """Tests for ToolCatalogLoader.load() method."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock tool registry."""
        registry = MagicMock()
        registry.list_tools.return_value = []
        return registry

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        return settings

    def test_load_registers_tools_from_shared_registry(self, mock_registry, mock_settings):
        """Test that load() registers tools from SharedToolRegistry."""
        loader = ToolCatalogLoader(registry=mock_registry, settings=mock_settings)

        # Mock SharedToolRegistry
        mock_shared = MagicMock()
        mock_tools = [MagicMock(name=f"tool{i}") for i in range(5)]
        mock_shared.get_all_tools_for_registration.return_value = mock_tools

        with patch(
            "victor.agent.shared_tool_registry.SharedToolRegistry.get_instance",
            return_value=mock_shared,
        ):
            result = loader.load()

        assert result.tools_loaded == 5
        assert loader.is_loaded is True
        mock_shared.get_all_tools_for_registration.assert_called_once_with(airgapped_mode=False)

    def test_load_respects_airgapped_mode(self, mock_registry, mock_settings):
        """Test that load() passes airgapped_mode to SharedToolRegistry."""
        config = ToolCatalogConfig(airgapped_mode=True)
        loader = ToolCatalogLoader(registry=mock_registry, settings=mock_settings, config=config)

        mock_shared = MagicMock()
        mock_shared.get_all_tools_for_registration.return_value = []

        with patch(
            "victor.agent.shared_tool_registry.SharedToolRegistry.get_instance",
            return_value=mock_shared,
        ):
            loader.load()

        mock_shared.get_all_tools_for_registration.assert_called_once_with(airgapped_mode=True)

    def test_load_handles_registration_errors(self, mock_registry, mock_settings):
        """Test that load() handles tool registration errors gracefully."""
        loader = ToolCatalogLoader(registry=mock_registry, settings=mock_settings)

        # Make registry.register raise an exception for some tools
        mock_registry.register.side_effect = [
            None,
            Exception("Registration failed"),
            None,
        ]

        mock_shared = MagicMock()
        mock_tools = [MagicMock() for _ in range(3)]
        mock_shared.get_all_tools_for_registration.return_value = mock_tools

        with patch(
            "victor.agent.shared_tool_registry.SharedToolRegistry.get_instance",
            return_value=mock_shared,
        ):
            result = loader.load()

        # Should still register 2 tools (skipping the failed one)
        assert result.tools_loaded == 2

    def test_load_is_idempotent(self, mock_registry, mock_settings):
        """Test that calling load() multiple times only loads once."""
        loader = ToolCatalogLoader(registry=mock_registry, settings=mock_settings)

        mock_shared = MagicMock()
        mock_shared.get_all_tools_for_registration.return_value = [MagicMock() for _ in range(3)]

        with patch(
            "victor.agent.shared_tool_registry.SharedToolRegistry.get_instance",
            return_value=mock_shared,
        ):
            result1 = loader.load()
            # Manually reset to test idempotency logic
            assert loader.is_loaded is True

        assert result1.tools_loaded == 3


class TestToolCatalogLoaderConfiguration:
    """Tests for ToolCatalogLoader configuration application."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock tool registry."""
        registry = MagicMock()
        tools = [MagicMock(name=f"tool{i}") for i in range(5)]
        for i, tool in enumerate(tools):
            tool.name = f"tool{i}"
        registry.list_tools.return_value = tools
        return registry

    @pytest.fixture
    def loader_with_tools(self, mock_registry):
        """Create loader with mocked SharedToolRegistry."""
        settings = MagicMock()
        settings.load_tool_config.return_value = {}

        loader = ToolCatalogLoader(registry=mock_registry, settings=settings)

        mock_shared = MagicMock()
        mock_shared.get_all_tools_for_registration.return_value = []

        return loader, mock_shared

    def test_apply_enabled_list_disables_others(self, mock_registry):
        """Test that enabled list disables tools not in the list."""
        settings = MagicMock()
        settings.load_tool_config.return_value = {"enabled": ["tool0", "tool1"]}

        loader = ToolCatalogLoader(registry=mock_registry, settings=settings)

        mock_shared = MagicMock()
        mock_shared.get_all_tools_for_registration.return_value = []

        with patch(
            "victor.agent.shared_tool_registry.SharedToolRegistry.get_instance",
            return_value=mock_shared,
        ):
            result = loader.load()

        # Should disable tool2, tool3, tool4 (3 tools)
        assert result.tools_disabled == 3

    def test_apply_disabled_list(self, mock_registry):
        """Test that disabled list disables specified tools."""
        settings = MagicMock()
        settings.load_tool_config.return_value = {"disabled": ["tool0", "tool1"]}

        loader = ToolCatalogLoader(registry=mock_registry, settings=settings)

        mock_shared = MagicMock()
        mock_shared.get_all_tools_for_registration.return_value = []

        with patch(
            "victor.agent.shared_tool_registry.SharedToolRegistry.get_instance",
            return_value=mock_shared,
        ):
            result = loader.load()

        assert result.tools_disabled == 2
        mock_registry.disable_tool.assert_any_call("tool0")
        mock_registry.disable_tool.assert_any_call("tool1")

    def test_apply_per_tool_config(self, mock_registry):
        """Test per-tool configuration with enabled=False."""
        settings = MagicMock()
        settings.load_tool_config.return_value = {
            "tool0": {"enabled": False},
            "tool1": {"enabled": True},
        }

        loader = ToolCatalogLoader(registry=mock_registry, settings=settings)

        mock_shared = MagicMock()
        mock_shared.get_all_tools_for_registration.return_value = []

        with patch(
            "victor.agent.shared_tool_registry.SharedToolRegistry.get_instance",
            return_value=mock_shared,
        ):
            result = loader.load()

        assert result.tools_disabled == 1
        mock_registry.disable_tool.assert_called_with("tool0")

    def test_handles_config_load_error(self, mock_registry):
        """Test graceful handling of config load errors."""
        settings = MagicMock()
        settings.load_tool_config.side_effect = Exception("Config error")

        loader = ToolCatalogLoader(registry=mock_registry, settings=settings)

        mock_shared = MagicMock()
        mock_shared.get_all_tools_for_registration.return_value = []

        with patch(
            "victor.agent.shared_tool_registry.SharedToolRegistry.get_instance",
            return_value=mock_shared,
        ):
            # Should not raise, just log warning
            result = loader.load()

        assert result.tools_disabled == 0


class TestToolCatalogLoaderGetToolConfig:
    """Tests for ToolCatalogLoader.get_tool_config() method."""

    def test_get_tool_config_empty_when_airgapped(self):
        """Test get_tool_config returns empty in airgapped mode."""
        registry = MagicMock()
        settings = MagicMock()
        config = ToolCatalogConfig(airgapped_mode=True)

        loader = ToolCatalogLoader(registry=registry, settings=settings, config=config)

        result = loader.get_tool_config()

        assert result == {}

    def test_get_tool_config_loads_web_config(self):
        """Test get_tool_config loads web tool configuration."""
        registry = MagicMock()
        settings = MagicMock()
        settings.load_tool_config.return_value = {
            "web_tools": {
                "summarize_fetch_top": 5,
                "summarize_fetch_pool": 10,
                "summarize_max_content_length": 50000,
            }
        }

        loader = ToolCatalogLoader(registry=registry, settings=settings)

        result = loader.get_tool_config()

        assert result["web_fetch_top"] == 5
        assert result["web_fetch_pool"] == 10
        assert result["max_content_length"] == 50000

    def test_get_tool_config_handles_load_error(self):
        """Test get_tool_config handles load errors gracefully."""
        registry = MagicMock()
        settings = MagicMock()
        settings.load_tool_config.side_effect = Exception("Load error")

        loader = ToolCatalogLoader(registry=registry, settings=settings)

        result = loader.get_tool_config()

        assert result == {}
