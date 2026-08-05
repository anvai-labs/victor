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

"""Unit tests for PluginLoader component.

Tests the SRP-compliant plugin loading functionality.
"""

import pytest
from unittest.mock import MagicMock, patch

from victor.agent.plugin_loader import (
    PluginLoader,
    PluginLoaderConfig,
    PluginLoadResult,
    PluginInfo,
)


class TestPluginLoaderConfig:
    """Tests for PluginLoaderConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PluginLoaderConfig()

        assert config.enabled is True
        assert config.plugin_dirs == []
        assert config.disabled_plugins == set()
        assert config.plugin_packages == []

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PluginLoaderConfig(
            enabled=False,
            plugin_dirs=["/path/to/plugins"],
            disabled_plugins={"plugin1", "plugin2"},
            plugin_packages=["my.plugin.package"],
        )

        assert config.enabled is False
        assert config.plugin_dirs == ["/path/to/plugins"]
        assert config.disabled_plugins == {"plugin1", "plugin2"}
        assert config.plugin_packages == ["my.plugin.package"]


class TestPluginLoadResult:
    """Tests for PluginLoadResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = PluginLoadResult()

        assert result.plugins_loaded == 0
        assert result.tools_registered == 0
        assert result.errors == []

    def test_custom_values(self):
        """Test custom result values."""
        result = PluginLoadResult(
            plugins_loaded=3,
            tools_registered=10,
            errors=["error1"],
        )

        assert result.plugins_loaded == 3
        assert result.tools_registered == 10
        assert result.errors == ["error1"]


class TestPluginInfo:
    """Tests for PluginInfo dataclass."""

    def test_default_values(self):
        """Test default PluginInfo values."""
        info = PluginInfo(name="test_plugin")

        assert info.name == "test_plugin"
        assert info.version == "unknown"
        assert info.tool_count == 0

    def test_custom_values(self):
        """Test custom PluginInfo values."""
        info = PluginInfo(
            name="my_plugin",
            version="1.2.3",
            tool_count=5,
        )

        assert info.name == "my_plugin"
        assert info.version == "1.2.3"
        assert info.tool_count == 5


class TestPluginLoaderInit:
    """Tests for PluginLoader initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default config."""
        registry = MagicMock()
        settings = MagicMock()

        loader = PluginLoader(registry=registry, settings=settings)

        assert loader._registry is registry
        assert loader._settings is settings
        assert loader._config.enabled is True
        assert loader.is_loaded is False
        assert loader.plugin_manager is None

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config."""
        registry = MagicMock()
        settings = MagicMock()
        config = PluginLoaderConfig(enabled=False)

        loader = PluginLoader(registry=registry, settings=settings, config=config)

        assert loader._config.enabled is False


class TestPluginLoaderLoad:
    """Tests for PluginLoader.load() method."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock tool registry."""
        return MagicMock()

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.plugin_config = {}
        return settings

    def test_load_disabled_returns_empty_result(self, mock_registry, mock_settings):
        """Test that load() returns empty result when disabled."""
        config = PluginLoaderConfig(enabled=False)
        loader = PluginLoader(registry=mock_registry, settings=mock_settings, config=config)

        result = loader.load()

        assert result.plugins_loaded == 0
        assert result.tools_registered == 0
        assert loader.is_loaded is True

    def test_load_initializes_plugin_manager(self, mock_registry, mock_settings):
        """Test that load() initializes plugin manager."""
        loader = PluginLoader(registry=mock_registry, settings=mock_settings)

        mock_plugin_registry = MagicMock()
        mock_plugin_registry.loaded_plugins = {"plugin1": MagicMock()}
        mock_plugin_registry.discover_and_load.return_value = 1
        mock_plugin_registry.register_tools.return_value = 5

        with patch(
            "victor.tools.plugin_registry.ToolPluginRegistry",
            return_value=mock_plugin_registry,
        ):
            with patch("victor.config.settings.get_project_paths") as mock_paths:
                mock_paths.return_value.global_plugins_dir = "/global/plugins"
                result = loader.load()

        assert result.plugins_loaded == 1
        assert result.tools_registered == 5
        assert loader.plugin_manager is mock_plugin_registry

    def test_load_disables_specified_plugins(self, mock_registry, mock_settings):
        """Test that load() disables specified plugins."""
        config = PluginLoaderConfig(disabled_plugins={"bad_plugin"})
        loader = PluginLoader(registry=mock_registry, settings=mock_settings, config=config)

        mock_plugin_registry = MagicMock()
        mock_plugin_registry.loaded_plugins = {}
        mock_plugin_registry.discover_and_load.return_value = 0
        mock_plugin_registry.register_tools.return_value = 0

        with patch(
            "victor.tools.plugin_registry.ToolPluginRegistry",
            return_value=mock_plugin_registry,
        ):
            with patch("victor.config.settings.get_project_paths") as mock_paths:
                mock_paths.return_value.global_plugins_dir = "/global/plugins"
                loader.load()

        mock_plugin_registry.disable_plugin.assert_called_once_with("bad_plugin")

    def test_load_loads_package_plugins(self, mock_registry, mock_settings):
        """Test that load() loads plugins from packages."""
        config = PluginLoaderConfig(plugin_packages=["my.plugin"])
        loader = PluginLoader(registry=mock_registry, settings=mock_settings, config=config)

        mock_plugin = MagicMock()
        mock_plugin_registry = MagicMock()
        mock_plugin_registry.loaded_plugins = {}
        mock_plugin_registry.discover_and_load.return_value = 0
        mock_plugin_registry.load_plugin_from_package.return_value = mock_plugin
        mock_plugin_registry.register_tools.return_value = 0

        with patch(
            "victor.tools.plugin_registry.ToolPluginRegistry",
            return_value=mock_plugin_registry,
        ):
            with patch("victor.config.settings.get_project_paths") as mock_paths:
                mock_paths.return_value.global_plugins_dir = "/global/plugins"
                loader.load()

        mock_plugin_registry.load_plugin_from_package.assert_called_once_with("my.plugin")
        mock_plugin_registry.register_plugin.assert_called_once_with(mock_plugin)

    def test_load_handles_errors_gracefully(self, mock_registry, mock_settings):
        """Test that load() handles initialization errors gracefully."""
        loader = PluginLoader(registry=mock_registry, settings=mock_settings)

        with patch(
            "victor.tools.plugin_registry.ToolPluginRegistry",
            side_effect=Exception("Plugin system error"),
        ):
            with patch("victor.config.settings.get_project_paths") as mock_paths:
                mock_paths.return_value.global_plugins_dir = "/global/plugins"
                result = loader.load()

        assert result.plugins_loaded == 0
        assert result.tools_registered == 0
        assert "Plugin system error" in result.errors[0]
        assert loader.plugin_manager is None


class TestPluginLoaderGetPluginInfo:
    """Tests for PluginLoader.get_plugin_info() method."""

    def test_get_plugin_info_empty_when_no_manager(self):
        """Test get_plugin_info returns empty list when no manager."""
        registry = MagicMock()
        settings = MagicMock()

        loader = PluginLoader(registry=registry, settings=settings)

        result = loader.get_plugin_info()

        assert result == []

    def test_get_plugin_info_returns_plugin_details(self):
        """Test get_plugin_info returns correct plugin details."""
        registry = MagicMock()
        settings = MagicMock()

        loader = PluginLoader(registry=registry, settings=settings)

        # Mock plugin manager with plugins
        mock_plugin1 = MagicMock()
        mock_plugin1.name = "plugin1"
        mock_plugin1.version = "1.0.0"
        mock_plugin1.get_tools.return_value = [MagicMock(), MagicMock()]

        mock_plugin2 = MagicMock()
        mock_plugin2.name = "plugin2"
        mock_plugin2.version = "2.0.0"
        mock_plugin2.get_tools.return_value = [MagicMock()]

        loader._plugin_manager = MagicMock()
        loader._plugin_manager.loaded_plugins = {
            "plugin1": mock_plugin1,
            "plugin2": mock_plugin2,
        }

        result = loader.get_plugin_info()

        assert len(result) == 2
        plugin_names = {p.name for p in result}
        assert plugin_names == {"plugin1", "plugin2"}


class TestPluginLoaderGetSummary:
    """Tests for PluginLoader.get_summary() method."""

    def test_get_summary_empty_when_no_plugins(self):
        """Test get_summary returns empty when no plugins."""
        registry = MagicMock()
        settings = MagicMock()

        loader = PluginLoader(registry=registry, settings=settings)

        result = loader.get_summary()

        assert result == {"plugins": [], "total": 0}

    def test_get_summary_returns_correct_structure(self):
        """Test get_summary returns correct structure."""
        registry = MagicMock()
        settings = MagicMock()

        loader = PluginLoader(registry=registry, settings=settings)

        # Mock plugin manager
        mock_plugin = MagicMock()
        mock_plugin.name = "test_plugin"
        mock_plugin.version = "1.0.0"
        mock_plugin.get_tools.return_value = [MagicMock(), MagicMock(), MagicMock()]

        loader._plugin_manager = MagicMock()
        loader._plugin_manager.loaded_plugins = {"test_plugin": mock_plugin}

        result = loader.get_summary()

        assert result["total"] == 1
        assert len(result["plugins"]) == 1
        assert result["plugins"][0]["name"] == "test_plugin"
        assert result["plugins"][0]["version"] == "1.0.0"
        assert result["plugins"][0]["tools"] == 3
