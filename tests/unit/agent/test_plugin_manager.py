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

"""Tests for plugin_registry module."""

import pytest
import tempfile
from pathlib import Path

from victor.tools.plugin_registry import ToolPluginRegistry, PluginLoadError


class TestToolPluginRegistry:
    """Tests for ToolPluginRegistry class."""

    def test_init_no_dirs(self):
        """Test initialization without directories."""
        manager = ToolPluginRegistry()
        assert manager.plugin_dirs == []
        assert manager.loaded_plugins == {}
        assert manager.config == {}

    def test_init_with_nonexistent_dir(self):
        """Test initialization with nonexistent directory."""
        manager = ToolPluginRegistry(plugin_dirs=[Path("/nonexistent/path")])
        assert len(manager.plugin_dirs) == 0

    def test_init_with_existing_dir(self):
        """Test initialization with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ToolPluginRegistry(plugin_dirs=[Path(tmpdir)])
            assert len(manager.plugin_dirs) == 1

    def test_init_with_config(self):
        """Test initialization with config."""
        config = {"test_plugin": {"api_key": "test_key"}}
        manager = ToolPluginRegistry(config=config)
        assert manager.config == config

    def test_config_access(self):
        """Test accessing plugin config via config attribute."""
        config = {"my_plugin": {"setting": "value"}}
        manager = ToolPluginRegistry(config=config)
        assert manager.config.get("my_plugin") == {"setting": "value"}
        assert manager.config.get("nonexistent", {}) == {}

    def test_get_all_tools_empty(self):
        """Test getting tools when no plugins loaded."""
        manager = ToolPluginRegistry()
        tools = manager.get_all_tools()
        assert tools == []

    def test_list_plugins_empty(self):
        """Test listing plugins when none loaded."""
        manager = ToolPluginRegistry()
        result = manager.list_plugins()
        # Returns empty dict when no plugins loaded
        assert result == {}

    def test_discover_plugins_empty_dirs(self):
        """Test discovering plugins in empty directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ToolPluginRegistry(plugin_dirs=[Path(tmpdir)])
            discovered = manager.discover_plugins()
            assert discovered == []

    def test_context_manager(self):
        """Test context manager support."""
        with ToolPluginRegistry() as manager:
            assert isinstance(manager, ToolPluginRegistry)


class TestPluginLoadError:
    """Tests for PluginLoadError exception."""

    def test_plugin_load_error(self):
        """Test PluginLoadError can be raised."""
        with pytest.raises(PluginLoadError):
            raise PluginLoadError("Test error")

    def test_plugin_load_error_message(self):
        """Test PluginLoadError message."""
        try:
            raise PluginLoadError("Plugin failed to load")
        except PluginLoadError as e:
            assert "Plugin failed to load" in str(e)


class TestPluginDiscovery:
    """Tests for plugin discovery functionality."""

    def test_discover_plugins_subdirectory(self):
        """Test discovering plugins in subdirectories (covers lines 121-126)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a plugin subdirectory with plugin.py
            plugin_dir = Path(tmpdir) / "my_plugin"
            plugin_dir.mkdir()
            plugin_file = plugin_dir / "plugin.py"
            plugin_file.write_text("# Plugin file")

            manager = ToolPluginRegistry(plugin_dirs=[Path(tmpdir)])
            discovered = manager.discover_plugins()

            assert len(discovered) == 1
            assert discovered[0] == plugin_dir

    def test_discover_plugins_direct_file(self):
        """Test discovering plugin.py directly in directory (covers lines 129-131)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create plugin.py directly in the directory
            plugin_file = Path(tmpdir) / "plugin.py"
            plugin_file.write_text("# Plugin file")

            manager = ToolPluginRegistry(plugin_dirs=[Path(tmpdir)])
            discovered = manager.discover_plugins()

            assert len(discovered) == 1
            assert discovered[0] == Path(tmpdir)

    def test_discover_plugins_not_a_directory(self):
        """Test discover_plugins skips non-directories (covers line 117-118)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file instead of directory
            file_path = Path(tmpdir) / "not_a_dir.txt"
            file_path.write_text("Not a directory")

            # The manager filters non-existent dirs in __init__
            # but discover_plugins also checks is_dir
            manager = ToolPluginRegistry()
            manager.plugin_dirs = [file_path]  # Force add a file

            discovered = manager.discover_plugins()
            assert discovered == []


class TestPluginLoading:
    """Tests for plugin loading functionality."""

    def test_load_plugin_no_plugin_file(self):
        """Test load_plugin_from_path with missing plugin.py (covers lines 145-148)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ToolPluginRegistry()
            result = manager.load_plugin_from_path(Path(tmpdir))
            assert result is None

    def test_load_plugin_invalid_spec(self):
        """Test load_plugin_from_path with invalid spec returns None (covers lines 155-157, 181-183)."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_file = Path(tmpdir) / "plugin.py"
            plugin_file.write_text("# Invalid plugin")

            manager = ToolPluginRegistry()

            # Mock spec_from_file_location to return None
            with patch("importlib.util.spec_from_file_location", return_value=None):
                # Method catches exception and returns None
                result = manager.load_plugin_from_path(Path(tmpdir))
                assert result is None

    def test_load_plugin_no_plugin_class(self):
        """Test load_plugin_from_path with missing Plugin class returns None (covers lines 164-165)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_file = Path(tmpdir) / "plugin.py"
            plugin_file.write_text("# No Plugin class here\nclass NotAPlugin: pass")

            manager = ToolPluginRegistry()
            # Method catches exception and returns None
            result = manager.load_plugin_from_path(Path(tmpdir))
            assert result is None

    def test_load_plugin_not_toolplugin_subclass(self):
        """Test load_plugin_from_path with non-ToolPlugin class returns None (covers lines 168-169)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_file = Path(tmpdir) / "plugin.py"
            # Create a Plugin class that doesn't extend ToolPlugin
            plugin_file.write_text("class Plugin: pass")

            manager = ToolPluginRegistry()
            # Method catches exception and returns None
            result = manager.load_plugin_from_path(Path(tmpdir))
            assert result is None


class TestDisabledPlugins:
    """Tests for disabled plugin functionality."""

    def test_init_creates_disabled_set(self):
        """Test that __init__ creates _disabled_plugins set (covers line 100)."""
        manager = ToolPluginRegistry()
        assert hasattr(manager, "_disabled_plugins")
        assert isinstance(manager._disabled_plugins, set)

    def test_auto_load_parameter(self):
        """Test auto_load parameter triggers discover_and_load (covers lines 102-103)."""
        from unittest.mock import patch

        with patch.object(ToolPluginRegistry, "discover_and_load") as mock_discover:
            ToolPluginRegistry(auto_load=True)
            mock_discover.assert_called_once()

    def test_no_auto_load_by_default(self):
        """Test auto_load is False by default."""
        from unittest.mock import patch

        with patch.object(ToolPluginRegistry, "discover_and_load") as mock_discover:
            ToolPluginRegistry()
            mock_discover.assert_not_called()


class TestPluginDirectory:
    """Tests for plugin directory handling."""

    def test_expanduser_on_plugin_dirs(self):
        """Test that plugin dirs are expanded (covers line 92)."""
        # Create a real temp dir since we can't reliably test ~
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ToolPluginRegistry(plugin_dirs=[Path(tmpdir)])
            assert len(manager.plugin_dirs) == 1
            # Path should be expanded (absolute)
            assert manager.plugin_dirs[0].is_absolute()

    def test_nonexistent_dir_logged(self):
        """Test that nonexistent dirs are logged (covers line 96)."""
        # This should not add to plugin_dirs
        manager = ToolPluginRegistry(plugin_dirs=[Path("/nonexistent/path/123456")])
        assert len(manager.plugin_dirs) == 0


class TestMockPlugin:
    """Helper for creating mock plugins."""

    @pytest.fixture
    def mock_plugin(self):
        """Create a mock plugin for testing."""
        from victor.tools.plugin import ToolPlugin

        class TestPlugin(ToolPlugin):
            name = "test_plugin"
            version = "1.0.0"
            description = "Test plugin"
            author = "Test"

            def initialize(self):
                pass

            def cleanup(self):
                pass

            def get_tools(self):
                return []

        return TestPlugin()


class TestRegisterPlugin(TestMockPlugin):
    """Tests for register_plugin method."""

    def test_register_plugin_success(self, mock_plugin):
        """Test registering a plugin (covers lines 244-249)."""
        manager = ToolPluginRegistry()
        result = manager.register_plugin(mock_plugin)

        assert result is True
        assert "test_plugin" in manager.loaded_plugins

    def test_register_plugin_already_registered(self, mock_plugin):
        """Test registering already registered plugin (covers lines 236-238)."""
        manager = ToolPluginRegistry()
        manager.register_plugin(mock_plugin)

        # Try to register again
        result = manager.register_plugin(mock_plugin)

        assert result is False

    def test_register_plugin_disabled(self, mock_plugin):
        """Test registering disabled plugin (covers lines 240-242)."""
        manager = ToolPluginRegistry()
        manager._disabled_plugins.add("test_plugin")

        result = manager.register_plugin(mock_plugin)

        assert result is False

    def test_register_plugin_init_fails(self, mock_plugin):
        """Test register when plugin init fails (covers lines 251-253)."""
        manager = ToolPluginRegistry()
        mock_plugin._do_initialize = lambda: (_ for _ in ()).throw(Exception("Init failed"))

        result = manager.register_plugin(mock_plugin)

        assert result is False


class TestUnloadPlugin(TestMockPlugin):
    """Tests for unload_plugin method."""

    def test_unload_plugin_success(self, mock_plugin):
        """Test unloading a plugin (covers lines 270-274)."""
        manager = ToolPluginRegistry()
        manager.register_plugin(mock_plugin)

        result = manager.unload_plugin("test_plugin")

        assert result is True
        assert "test_plugin" not in manager.loaded_plugins

    def test_unload_plugin_not_loaded(self):
        """Test unloading non-loaded plugin (covers lines 264-266)."""
        manager = ToolPluginRegistry()

        result = manager.unload_plugin("nonexistent")

        assert result is False

    def test_unload_plugin_cleanup_fails(self, mock_plugin):
        """Test unload when cleanup fails (covers lines 276-278)."""
        manager = ToolPluginRegistry()
        manager.register_plugin(mock_plugin)
        mock_plugin._do_cleanup = lambda: (_ for _ in ()).throw(Exception("Cleanup failed"))

        result = manager.unload_plugin("test_plugin")

        assert result is False


class TestDiscoverAndLoad(TestMockPlugin):
    """Tests for discover_and_load method."""

    def test_discover_and_load_empty(self):
        """Test discover_and_load with no plugins (covers lines 286-295)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ToolPluginRegistry(plugin_dirs=[Path(tmpdir)])
            count = manager.discover_and_load()
            assert count == 0


class TestGetAllTools(TestMockPlugin):
    """Tests for get_all_tools method."""

    def test_get_all_tools_with_plugins(self, mock_plugin):
        """Test get_all_tools with loaded plugins (covers lines 303-306)."""
        manager = ToolPluginRegistry()
        manager.register_plugin(mock_plugin)

        tools = manager.get_all_tools()

        # mock_plugin returns empty list
        assert tools == []


class TestRegisterTools(TestMockPlugin):
    """Tests for register_tools method."""

    def test_register_tools_empty(self, mock_plugin):
        """Test register_tools with no tools (covers lines 317-330)."""
        from victor.tools.registry import ToolRegistry

        manager = ToolPluginRegistry()
        manager.register_plugin(mock_plugin)

        registry = ToolRegistry()
        count = manager.register_tools(registry)

        assert count == 0


class TestGetPluginInfo(TestMockPlugin):
    """Tests for get_plugin_info method."""

    def test_get_plugin_info(self, mock_plugin):
        """Test get_plugin_info (covers line 338)."""
        manager = ToolPluginRegistry()
        manager.register_plugin(mock_plugin)

        info = manager.get_plugin_info()

        assert len(info) == 1
        assert info[0].name == "test_plugin"


class TestDisableEnablePlugin(TestMockPlugin):
    """Tests for disable/enable plugin methods."""

    def test_disable_plugin(self):
        """Test disable_plugin (covers line 346)."""
        manager = ToolPluginRegistry()
        manager.disable_plugin("some_plugin")

        assert "some_plugin" in manager._disabled_plugins

    def test_disable_loaded_plugin(self, mock_plugin):
        """Test disabling a loaded plugin (covers lines 347-348)."""
        manager = ToolPluginRegistry()
        manager.register_plugin(mock_plugin)

        manager.disable_plugin("test_plugin")

        assert "test_plugin" in manager._disabled_plugins
        assert "test_plugin" not in manager.loaded_plugins

    def test_enable_plugin(self):
        """Test enable_plugin (covers line 356)."""
        manager = ToolPluginRegistry()
        manager._disabled_plugins.add("some_plugin")

        manager.enable_plugin("some_plugin")

        assert "some_plugin" not in manager._disabled_plugins


class TestReloadPlugin(TestMockPlugin):
    """Tests for reload_plugin method."""

    def test_reload_plugin_not_loaded(self):
        """Test reloading non-loaded plugin (covers lines 369-371)."""
        manager = ToolPluginRegistry()

        result = manager.reload_plugin("nonexistent")

        assert result is False

    def test_reload_plugin_no_path(self, mock_plugin):
        """Test reloading plugin without path (covers lines 376-378)."""
        manager = ToolPluginRegistry()
        manager.register_plugin(mock_plugin)
        # Ensure metadata has no path
        mock_plugin.get_metadata().path = None

        result = manager.reload_plugin("test_plugin")

        assert result is False


class TestCleanupAll(TestMockPlugin):
    """Tests for cleanup_all method."""

    def test_cleanup_all(self, mock_plugin):
        """Test cleanup_all (covers lines 394-395)."""
        manager = ToolPluginRegistry()
        manager.register_plugin(mock_plugin)

        manager.cleanup_all()

        assert len(manager.loaded_plugins) == 0


class TestLoadUnload(TestMockPlugin):
    """Tests for load and unload alias methods."""

    def test_load(self, mock_plugin):
        """Test load method (covers line 408)."""
        manager = ToolPluginRegistry()

        result = manager.load(mock_plugin)

        assert result is True
        assert "test_plugin" in manager.loaded_plugins

    def test_unload(self, mock_plugin):
        """Test unload method (covers line 421)."""
        manager = ToolPluginRegistry()
        manager.load(mock_plugin)

        result = manager.unload("test_plugin")

        assert result is True
        assert "test_plugin" not in manager.loaded_plugins


class TestListPlugins(TestMockPlugin):
    """Tests for list_plugins method."""

    def test_list_plugins_with_plugin(self, mock_plugin):
        """Test list_plugins with loaded plugin (covers lines 430-437)."""
        manager = ToolPluginRegistry()
        manager.register_plugin(mock_plugin)

        result = manager.list_plugins()

        assert "test_plugin" in result
        assert result["test_plugin"]["version"] == "1.0.0"
        assert result["test_plugin"]["enabled"] is True


class TestLoadPluginFromPackage:
    """Tests for load_plugin_from_package method."""

    def test_load_plugin_from_package_not_found(self):
        """Test load_plugin_from_package with non-existent package (covers lines 220-222)."""
        manager = ToolPluginRegistry()

        result = manager.load_plugin_from_package("nonexistent_package_12345")

        assert result is None

    def test_load_plugin_from_package_no_plugin_class(self):
        """Test load_plugin_from_package with no Plugin class (covers lines 199-205)."""
        from unittest.mock import patch, MagicMock

        manager = ToolPluginRegistry()

        # Mock module without Plugin class
        mock_module = MagicMock()
        del mock_module.Plugin  # Ensure no Plugin attribute

        # Also mock the submodule import to fail
        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = [mock_module, ImportError("No submodule")]

            result = manager.load_plugin_from_package("some_package")

            assert result is None
