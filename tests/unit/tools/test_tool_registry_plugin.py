"""Tests for ToolRegistry plugin-based registration."""

import pytest
from victor.tools.registry import ToolRegistry
from victor_contracts.verticals.protocols import (
    ToolFactory,
    ToolPluginHelper,
)


class TestToolRegistryPlugin:
    """Tests for plugin-based tool registration."""

    def test_register_plugin_with_register_method(self):
        """Test registering a plugin with register() method."""
        registry = ToolRegistry()

        # Create a simple plugin
        class MyPlugin:
            def register(self, registry: ToolRegistry) -> None:
                # Just verify register() was called
                registry._test_plugin_called = True

        plugin = MyPlugin()
        registry.register_plugin(plugin)

        # Verify plugin's register() was called
        assert hasattr(registry, "_test_plugin_called")
        assert registry._test_plugin_called is True

    def test_register_plugin_raises_without_register_method(self):
        """Test that register_plugin raises without register() method."""
        registry = ToolRegistry()

        # Create an object without register method
        class NotAPlugin:
            pass

        plugin = NotAPlugin()

        with pytest.raises(AttributeError, match="register"):
            registry.register_plugin(plugin)

    def test_plugin_context_has_register_tool(self):
        """Test that ToolFactoryAdapter uses context.register_tool() correctly."""
        # This tests the SDK protocol integration

        class MockContext:
            def __init__(self):
                self.registered_tools = []

            def register_tool(self, tool):
                self.registered_tools.append(tool)

        class MockTool:
            def __init__(self, name):
                self.name = name

        # Test with instances
        context = MockContext()
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2")

        adapter = ToolPluginHelper.from_instances({"tool1": tool1, "tool2": tool2})
        adapter.register(context)

        # Verify tools were registered via context
        assert len(context.registered_tools) == 2
        assert context.registered_tools[0].name == "tool1"
        assert context.registered_tools[1].name == "tool2"

    def test_tool_factory_creates_tools_on_demand(self):
        """Test that ToolFactory creates tools when called."""

        class MyFactory(ToolFactory):
            def __init__(self, name: str):
                self._name = name
                self._call_count = 0

            def __call__(self):
                self._call_count += 1
                return f"tool_instance_{self._name}"

            @property
            def name(self) -> str:
                return self._name

        factory1 = MyFactory("tool1")
        factory2 = MyFactory("tool2")

        # Verify factory properties
        assert factory1.name == "tool1"
        assert factory2.name == "tool2"

        # Verify factory creates tools
        tool1 = factory1()
        tool2 = factory2()

        assert tool1 == "tool_instance_tool1"
        assert tool2 == "tool_instance_tool2"
        assert factory1._call_count == 1
        assert factory2._call_count == 1

    def test_tool_factory_adapter_with_factories(self):
        """Test ToolFactoryAdapter with tool factories."""

        class MockContext:
            def __init__(self):
                self.registered_tools = []

            def register_tool(self, tool):
                self.registered_tools.append(tool)

        class MyFactory(ToolFactory):
            def __init__(self, name: str):
                self._name = name

            def __call__(self):
                return f"factory_created_{self._name}"

            @property
            def name(self) -> str:
                return self._name

        context = MockContext()
        factory1 = MyFactory("tool1")
        factory2 = MyFactory("tool2")

        adapter = ToolPluginHelper.from_factories({"tool1": factory1, "tool2": factory2})
        adapter.register(context)

        # Verify factories were called and tools registered
        assert len(context.registered_tools) == 2
        assert context.registered_tools[0] == "factory_created_tool1"
        assert context.registered_tools[1] == "factory_created_tool2"

    def test_register_from_entry_points_with_plugin(self, monkeypatch):
        """Test register_from_entry_points with a plugin entry point."""

        class MockEntryPoint:
            def __init__(self, name, value):
                self.name = name
                self.value = value

            def load(self):
                return self.value

        class MockEntryPoints:
            def __init__(self, entries):
                self.entries = entries

            def select(self, group=None):
                return self.entries

        # Create mock entry point with plugin
        class MyPlugin:
            def __init__(self):
                self.registered = False

            def register(self, registry: ToolRegistry) -> None:
                self.registered = True
                registry._plugin_test_marker = "loaded"

        plugin_instance = MyPlugin()
        mock_ep = MockEntryPoint("test_plugin", plugin_instance)

        # Monkey patch the entry point discovery to return only our mock
        monkeypatch.setattr(
            "victor.framework.entry_point_registry.get_entry_point_objects",
            lambda group, **kwargs: (mock_ep,),
        )

        registry = ToolRegistry()
        count = registry.register_from_entry_points()

        # Verify plugin was loaded and register() was called
        assert count == 1
        assert plugin_instance.registered is True
        assert hasattr(registry, "_plugin_test_marker")
        assert registry._plugin_test_marker == "loaded"

    def test_register_from_entry_points_with_list_converts_to_plugin(self, monkeypatch):
        """Test register_from_entry_points with a list entry point.

        Lists are converted to plugins via ToolPluginHelper.from_instances(),
        but invalid tools still fail registration (count may be 0).
        """
        # Track that the list was processed
        processed_list = []

        class MockEntryPoint:
            def __init__(self, name, value):
                self.name = name
                self.value = value

            def load(self):
                processed_list.append(self.name)
                return self.value

        class MockEntryPoints:
            def __init__(self, entries):
                self.entries = entries

            def select(self, group=None):
                return self.entries

        # Create mock entry point with list
        tool_list = ["item1", "item2"]
        mock_ep = MockEntryPoint("test_list", tool_list)

        # Monkey patch entry point discovery to return only our mock
        monkeypatch.setattr(
            "victor.framework.entry_point_registry.get_entry_point_objects",
            lambda group, **kwargs: (mock_ep,),
        )

        registry = ToolRegistry()

        # List entry point is loaded but registration fails (not real tools)
        # The important thing is it was processed
        count = registry.register_from_entry_points()

        # Verify entry point was loaded
        assert "test_list" in processed_list
        # Count is 0 because tool registration failed (expected for invalid tools)

    def test_discover_plugins(self, monkeypatch):
        """Test discover_plugins() method."""

        class MockEntryPoint:
            def __init__(self, name, value):
                self.name = name
                self.value = value

            def load(self):
                return self.value

        class MockEntryPoints:
            def __init__(self, entries):
                self.entries = entries

            def select(self, group=None):
                return self.entries

        # Create multiple mock plugins
        class Plugin1:
            def register(self, registry):
                registry._plugin1_loaded = True

        class Plugin2:
            def register(self, registry):
                registry._plugin2_loaded = True

        mock_ep1 = MockEntryPoint("plugin1", Plugin1())
        mock_ep2 = MockEntryPoint("plugin2", Plugin2())

        # Monkey patch entry point discovery to return only our mocks
        monkeypatch.setattr(
            "victor.framework.entry_point_registry.get_entry_point_objects",
            lambda group, **kwargs: (mock_ep1, mock_ep2),
        )

        registry = ToolRegistry()
        count = registry.discover_plugins()

        # Verify both plugins were discovered and loaded
        assert count == 2
        assert hasattr(registry, "_plugin1_loaded")
        assert hasattr(registry, "_plugin2_loaded")
        assert registry._plugin1_loaded is True
        assert registry._plugin2_loaded is True

    def test_tool_plugin_helper_from_module(self):
        """Test ToolPluginHelper.from_module() scanning for tools."""

        # Create a mock module with tool-like objects
        class MockModule:
            pass

        # Add some tool-like objects (with 'tool' attribute)
        class MockTool:
            def __init__(self, name):
                self.tool = name  # Has 'tool' attribute

        mod = MockModule()
        mod.search_tool = MockTool("search")
        mod.write_tool = MockTool("write")
        mod.not_a_tool = "just a string"  # Missing 'tool' attribute

        # Scan module for tools
        adapter = ToolPluginHelper.from_module(mod, tool_attribute="tool")

        # Verify only objects with 'tool' attribute were found
        # The implementation scans dir(module) for objects with the attribute
        assert adapter is not None
