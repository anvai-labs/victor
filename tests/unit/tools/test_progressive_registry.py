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

"""Tests for victor.tools.progressive_registry module."""

import pytest

from victor.tools.progressive_registry import (
    ProgressiveToolConfig,
    ProgressiveToolsRegistry,
    get_progressive_registry,
)


class TestProgressiveToolConfig:
    """Tests for ProgressiveToolConfig dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal fields."""
        config = ProgressiveToolConfig(tool_name="test_tool")
        assert config.tool_name == "test_tool"
        assert config.progressive_params == {}
        assert config.initial_values == {}
        assert config.max_values == {}

    def test_create_full(self):
        """Test creating with all fields."""
        config = ProgressiveToolConfig(
            tool_name="search_tool",
            progressive_params={"max_results": "int"},
            initial_values={"max_results": 10},
            max_values={"max_results": 100},
        )
        assert config.tool_name == "search_tool"
        assert config.progressive_params == {"max_results": "int"}
        assert config.initial_values == {"max_results": 10}
        assert config.max_values == {"max_results": 100}


class TestProgressiveToolsRegistry:
    """Tests for ProgressiveToolsRegistry singleton."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the singleton before each test."""
        ProgressiveToolsRegistry.reset_instance()
        yield
        ProgressiveToolsRegistry.reset_instance()

    def test_singleton_get_instance(self):
        """Test that get_instance returns the same instance."""
        instance1 = ProgressiveToolsRegistry.get_instance()
        instance2 = ProgressiveToolsRegistry.get_instance()
        assert instance1 is instance2

    def test_singleton_reset_instance(self):
        """Test that reset_instance clears the singleton."""
        instance1 = ProgressiveToolsRegistry.get_instance()
        ProgressiveToolsRegistry.reset_instance()
        instance2 = ProgressiveToolsRegistry.get_instance()
        assert instance1 is not instance2

    def test_register_tool(self):
        """Test registering a tool with progressive parameters."""
        registry = ProgressiveToolsRegistry.get_instance()
        registry.register(
            tool_name="code_search",
            progressive_params={"max_files": "int", "depth": "int"},
            initial_values={"max_files": 5, "depth": 2},
            max_values={"max_files": 50, "depth": 10},
        )

        config = registry.get_config("code_search")
        assert config is not None
        assert config.tool_name == "code_search"
        assert config.progressive_params == {"max_files": "int", "depth": "int"}
        assert config.initial_values == {"max_files": 5, "depth": 2}
        assert config.max_values == {"max_files": 50, "depth": 10}

    def test_register_tool_minimal(self):
        """Test registering a tool with only required fields."""
        registry = ProgressiveToolsRegistry.get_instance()
        registry.register(
            tool_name="simple_tool",
            progressive_params={"limit": "int"},
        )

        config = registry.get_config("simple_tool")
        assert config is not None
        assert config.tool_name == "simple_tool"
        assert config.progressive_params == {"limit": "int"}
        assert config.initial_values == {}
        assert config.max_values == {}

    def test_is_progressive_true(self):
        """Test is_progressive returns True for registered tools."""
        registry = ProgressiveToolsRegistry.get_instance()
        registry.register(
            tool_name="progressive_tool",
            progressive_params={"count": "int"},
        )
        assert registry.is_progressive("progressive_tool") is True

    def test_is_progressive_false(self):
        """Test is_progressive returns False for unregistered tools."""
        registry = ProgressiveToolsRegistry.get_instance()
        assert registry.is_progressive("nonexistent_tool") is False

    def test_get_config_existing(self):
        """Test get_config returns config for registered tool."""
        registry = ProgressiveToolsRegistry.get_instance()
        registry.register(
            tool_name="existing_tool",
            progressive_params={"size": "int"},
            initial_values={"size": 100},
        )

        config = registry.get_config("existing_tool")
        assert config is not None
        assert isinstance(config, ProgressiveToolConfig)
        assert config.tool_name == "existing_tool"

    def test_get_config_nonexistent(self):
        """Test get_config returns None for unregistered tool."""
        registry = ProgressiveToolsRegistry.get_instance()
        config = registry.get_config("nonexistent_tool")
        assert config is None

    def test_list_progressive_tools_empty(self):
        """Test list_progressive_tools returns empty set when no tools registered."""
        registry = ProgressiveToolsRegistry.get_instance()
        tools = registry.list_progressive_tools()
        assert tools == set()

    def test_list_progressive_tools_with_tools(self):
        """Test list_progressive_tools returns all registered tool names."""
        registry = ProgressiveToolsRegistry.get_instance()
        registry.register("tool_a", {"param": "int"})
        registry.register("tool_b", {"param": "int"})
        registry.register("tool_c", {"param": "int"})

        tools = registry.list_progressive_tools()
        assert tools == {"tool_a", "tool_b", "tool_c"}

    def test_register_overwrites_existing(self):
        """Test that registering the same tool overwrites previous config."""
        registry = ProgressiveToolsRegistry.get_instance()

        registry.register(
            tool_name="overwrite_tool",
            progressive_params={"old_param": "str"},
            initial_values={"old_param": "old_value"},
        )

        registry.register(
            tool_name="overwrite_tool",
            progressive_params={"new_param": "int"},
            initial_values={"new_param": 42},
        )

        config = registry.get_config("overwrite_tool")
        assert config is not None
        assert config.progressive_params == {"new_param": "int"}
        assert config.initial_values == {"new_param": 42}


class TestGetProgressiveRegistry:
    """Tests for the get_progressive_registry convenience function."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the singleton before each test."""
        ProgressiveToolsRegistry.reset_instance()
        yield
        ProgressiveToolsRegistry.reset_instance()

    def test_returns_singleton(self):
        """Test that get_progressive_registry returns the singleton instance."""
        registry = get_progressive_registry()
        assert registry is ProgressiveToolsRegistry.get_instance()

    def test_returns_same_instance(self):
        """Test that get_progressive_registry returns same instance on multiple calls."""
        registry1 = get_progressive_registry()
        registry2 = get_progressive_registry()
        assert registry1 is registry2

    def test_modifications_persist(self):
        """Test that modifications via get_progressive_registry persist."""
        registry1 = get_progressive_registry()
        registry1.register("persist_tool", {"param": "int"})

        registry2 = get_progressive_registry()
        assert registry2.is_progressive("persist_tool") is True
