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

"""Tests for the plugin module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from victor.tools.plugin import PluginMetadata, ToolPlugin, FunctionToolPlugin
from victor.tools.base import BaseTool, ToolResult


class MockTool(BaseTool):
    """Mock tool for testing."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"input": {"type": "string", "description": "Test input"}},
            "required": ["input"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="Mock result")


class TestPluginMetadata:
    """Tests for PluginMetadata dataclass."""

    def test_metadata_creation_minimal(self):
        """Test creating metadata with minimal fields."""
        metadata = PluginMetadata(name="test", version="1.0.0")
        assert metadata.name == "test"
        assert metadata.version == "1.0.0"
        assert metadata.description == ""
        assert metadata.author == ""
        assert metadata.homepage == ""
        assert metadata.dependencies == []
        assert metadata.path is None
        assert metadata.enabled is True

    def test_metadata_creation_full(self):
        """Test creating metadata with all fields."""
        metadata = PluginMetadata(
            name="full_plugin",
            version="2.1.0",
            description="A full plugin",
            author="Test Author",
            homepage="https://example.com",
            dependencies=["dep1", "dep2"],
            path=Path("/plugins/full"),
            enabled=False,
        )
        assert metadata.name == "full_plugin"
        assert metadata.version == "2.1.0"
        assert metadata.description == "A full plugin"
        assert metadata.author == "Test Author"
        assert metadata.homepage == "https://example.com"
        assert metadata.dependencies == ["dep1", "dep2"]
        assert metadata.path == Path("/plugins/full")
        assert metadata.enabled is False

    def test_metadata_dependencies_isolation(self):
        """Test that dependencies list is properly isolated."""
        metadata1 = PluginMetadata(name="p1", version="1.0")
        metadata2 = PluginMetadata(name="p2", version="1.0")

        metadata1.dependencies.append("new_dep")
        assert "new_dep" not in metadata2.dependencies


class ConcretePlugin(ToolPlugin):
    """Concrete plugin implementation for testing."""

    name = "test_plugin"
    version = "1.0.0"
    description = "Test plugin"
    author = "Test Author"
    homepage = "https://test.com"
    dependencies = ["dep1"]

    def __init__(self, config=None):
        super().__init__(config)
        self.initialize_called = False
        self.cleanup_called = False
        self.tool_registered_calls = []
        self.tool_executed_calls = []

    def get_tools(self):
        return [MockTool()]

    def initialize(self):
        self.initialize_called = True

    def cleanup(self):
        self.cleanup_called = True

    def on_tool_registered(self, tool):
        self.tool_registered_calls.append(tool)

    def on_tool_executed(self, tool_name, success, result):
        self.tool_executed_calls.append((tool_name, success, result))


class TestToolPlugin:
    """Tests for ToolPlugin abstract base class."""

    def test_plugin_init_no_config(self):
        """Test plugin initialization without config."""
        plugin = ConcretePlugin()
        assert plugin.config == {}
        assert plugin._initialized is False
        assert plugin._tools == []

    def test_plugin_init_with_config(self):
        """Test plugin initialization with config."""
        config = {"key": "value", "number": 42}
        plugin = ConcretePlugin(config)
        assert plugin.config == config
        assert plugin.config["key"] == "value"
        assert plugin.config["number"] == 42

    def test_get_metadata(self):
        """Test getting plugin metadata."""
        plugin = ConcretePlugin()
        metadata = plugin.get_metadata()

        assert isinstance(metadata, PluginMetadata)
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test plugin"
        assert metadata.author == "Test Author"
        assert metadata.homepage == "https://test.com"
        assert metadata.dependencies == ["dep1"]
        assert metadata.enabled is True

    def test_validate_config_default(self):
        """Test default config validation returns empty list."""
        plugin = ConcretePlugin()
        errors = plugin.validate_config()
        assert errors == []

    def test_do_initialize(self):
        """Test internal initialization wrapper."""
        plugin = ConcretePlugin()
        assert plugin._initialized is False

        plugin._do_initialize()

        assert plugin._initialized is True
        assert plugin.initialize_called is True
        assert len(plugin._tools) == 1
        assert isinstance(plugin._tools[0], MockTool)

    def test_do_initialize_idempotent(self):
        """Test that initialize is only called once."""
        plugin = ConcretePlugin()
        plugin._do_initialize()
        plugin.initialize_called = False  # Reset flag

        plugin._do_initialize()  # Call again

        assert plugin.initialize_called is False  # Should not be called again

    def test_do_initialize_with_validation_errors(self):
        """Test initialization fails with config validation errors."""

        class ValidatingPlugin(ConcretePlugin):
            def validate_config(self):
                return ["Missing required field: api_key"]

        plugin = ValidatingPlugin()

        with pytest.raises(ValueError) as exc_info:
            plugin._do_initialize()

        assert "config validation failed" in str(exc_info.value)
        assert "api_key" in str(exc_info.value)

    def test_do_cleanup(self):
        """Test internal cleanup wrapper."""
        plugin = ConcretePlugin()
        plugin._do_initialize()
        assert plugin._initialized is True

        plugin._do_cleanup()

        assert plugin._initialized is False
        assert plugin.cleanup_called is True
        assert plugin._tools == []

    def test_do_cleanup_not_initialized(self):
        """Test cleanup does nothing if not initialized."""
        plugin = ConcretePlugin()
        plugin._do_cleanup()

        assert plugin.cleanup_called is False

    def test_do_cleanup_handles_exception(self):
        """Test cleanup handles exceptions gracefully."""

        class FailingCleanupPlugin(ConcretePlugin):
            def cleanup(self):
                raise RuntimeError("Cleanup failed")

        plugin = FailingCleanupPlugin()
        plugin._do_initialize()

        # Should not raise
        plugin._do_cleanup()

        assert plugin._initialized is False
        assert plugin._tools == []

    def test_on_tool_registered_callback(self):
        """Test on_tool_registered callback."""
        plugin = ConcretePlugin()
        tool = MockTool()

        plugin.on_tool_registered(tool)

        assert len(plugin.tool_registered_calls) == 1
        assert plugin.tool_registered_calls[0] is tool

    def test_on_tool_executed_callback(self):
        """Test on_tool_executed callback."""
        plugin = ConcretePlugin()

        plugin.on_tool_executed("mock_tool", True, {"output": "test"})

        assert len(plugin.tool_executed_calls) == 1
        assert plugin.tool_executed_calls[0] == ("mock_tool", True, {"output": "test"})

    def test_on_tool_executed_failure(self):
        """Test on_tool_executed callback for failed execution."""
        plugin = ConcretePlugin()

        plugin.on_tool_executed("mock_tool", False, {"error": "Failed"})

        assert len(plugin.tool_executed_calls) == 1
        name, success, result = plugin.tool_executed_calls[0]
        assert name == "mock_tool"
        assert success is False
        assert result["error"] == "Failed"


class TestFunctionToolPlugin:
    """Tests for FunctionToolPlugin class."""

    def test_function_plugin_init(self):
        """Test FunctionToolPlugin initialization."""
        plugin = FunctionToolPlugin(
            name="func_plugin",
            version="1.0.0",
            tool_functions=[],
            description="Function plugin",
        )

        assert plugin.name == "func_plugin"
        assert plugin.version == "1.0.0"
        assert plugin.description == "Function plugin"
        assert plugin._tool_functions == []

    def test_function_plugin_with_config(self):
        """Test FunctionToolPlugin with config."""
        config = {"api_key": "test123"}
        plugin = FunctionToolPlugin(
            name="configured", version="1.0.0", tool_functions=[], config=config
        )

        assert plugin.config == config

    def test_get_tools_with_tool_attribute(self):
        """Test get_tools with @tool decorated functions."""
        # Create mock function with .Tool attribute
        mock_func = MagicMock()
        mock_func.Tool = MockTool()
        mock_func.__name__ = "mock_tool_func"

        plugin = FunctionToolPlugin(name="test", version="1.0.0", tool_functions=[mock_func])

        tools = plugin.get_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], MockTool)

    def test_get_tools_with_is_tool_attribute(self):
        """Test get_tools with legacy _is_tool attribute."""
        mock_func = MagicMock()
        mock_func._is_tool = True
        mock_func.Tool = MockTool()
        mock_func.__name__ = "legacy_tool"

        plugin = FunctionToolPlugin(name="legacy", version="1.0.0", tool_functions=[mock_func])

        tools = plugin.get_tools()
        assert len(tools) == 1

    def test_get_tools_warns_for_non_tool_function(self):
        """Test get_tools warns for functions without @tool decorator."""

        def regular_function():
            pass

        plugin = FunctionToolPlugin(name="test", version="1.0.0", tool_functions=[regular_function])

        tools = plugin.get_tools()
        assert len(tools) == 0  # Non-decorated functions are skipped

    def test_get_tools_mixed_functions(self):
        """Test get_tools with mix of decorated and non-decorated functions."""
        mock_tool_func = MagicMock()
        mock_tool_func.Tool = MockTool()
        mock_tool_func.__name__ = "tool_func"

        def regular_func():
            pass

        plugin = FunctionToolPlugin(
            name="mixed", version="1.0.0", tool_functions=[mock_tool_func, regular_func]
        )

        tools = plugin.get_tools()
        assert len(tools) == 1  # Only the decorated one

    def test_function_plugin_inherits_lifecycle(self):
        """Test FunctionToolPlugin inherits lifecycle methods."""
        plugin = FunctionToolPlugin(name="lifecycle", version="1.0.0", tool_functions=[])

        # Should have inherited methods
        assert hasattr(plugin, "initialize")
        assert hasattr(plugin, "cleanup")
        assert hasattr(plugin, "_do_initialize")
        assert hasattr(plugin, "_do_cleanup")
        assert hasattr(plugin, "get_metadata")

    def test_function_plugin_metadata(self):
        """Test FunctionToolPlugin metadata generation."""
        plugin = FunctionToolPlugin(
            name="meta_test",
            version="2.0.0",
            tool_functions=[],
            description="Metadata test plugin",
        )

        metadata = plugin.get_metadata()

        assert metadata.name == "meta_test"
        assert metadata.version == "2.0.0"
        assert metadata.description == "Metadata test plugin"


class TestPluginDefaultValues:
    """Tests for default values in ToolPlugin class attributes."""

    def test_unnamed_plugin_defaults(self):
        """Test default class attribute values."""

        class MinimalPlugin(ToolPlugin):
            def get_tools(self):
                return []

        plugin = MinimalPlugin()

        assert plugin.name == "unnamed_plugin"
        assert plugin.version == "0.0.0"
        assert plugin.description == ""
        assert plugin.author == ""
        assert plugin.homepage == ""
        assert plugin.dependencies == []

    def test_partial_override_defaults(self):
        """Test partial override of class attributes."""

        class PartialPlugin(ToolPlugin):
            name = "partial"
            version = "0.5.0"

            def get_tools(self):
                return []

        plugin = PartialPlugin()

        assert plugin.name == "partial"
        assert plugin.version == "0.5.0"
        assert plugin.description == ""  # Default
        assert plugin.author == ""  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
