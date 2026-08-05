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

"""Integration tests for progressive tools registry.

Tests ProgressiveToolsRegistry integration with tool configurations
and progressive parameter escalation behavior.
"""

import pytest
from typing import Any, Dict

from victor.tools.progressive_registry import (
    ProgressiveToolsRegistry,
    ProgressiveToolConfig,
    get_progressive_registry,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clean_registry():
    """Provide a clean registry instance for testing."""
    ProgressiveToolsRegistry.reset_instance()
    registry = ProgressiveToolsRegistry.get_instance()
    yield registry
    ProgressiveToolsRegistry.reset_instance()


@pytest.fixture
def populated_registry(clean_registry):
    """Provide a registry with pre-registered progressive tools."""
    registry = clean_registry

    # Register code search with progressive parameters
    registry.register(
        tool_name="code_search",
        progressive_params={
            "max_results": {"min": 10, "max": 100, "step": 10},
            "search_depth": {"min": 1, "max": 5, "step": 1},
        },
        initial_values={
            "max_results": 10,
            "search_depth": 1,
        },
        max_values={
            "max_results": 100,
            "search_depth": 5,
        },
    )

    # Register file read with line limits
    registry.register(
        tool_name="read_file",
        progressive_params={
            "max_lines": {"min": 100, "max": 1000, "step": 100},
        },
        initial_values={
            "max_lines": 100,
        },
        max_values={
            "max_lines": 1000,
        },
    )

    # Register web search with result limits
    registry.register(
        tool_name="web_search",
        progressive_params={
            "num_results": {"min": 5, "max": 50, "step": 5},
            "include_snippets": {"default": False, "escalated": True},
        },
        initial_values={
            "num_results": 5,
            "include_snippets": False,
        },
        max_values={
            "num_results": 50,
            "include_snippets": True,
        },
    )

    return registry


# =============================================================================
# Test Class: ProgressiveToolsRegistry Singleton
# =============================================================================


@pytest.mark.integration
class TestProgressiveToolsRegistrySingleton:
    """Tests for singleton behavior of ProgressiveToolsRegistry."""

    def test_get_instance_returns_singleton(self, clean_registry):
        """Test that get_instance returns the same instance."""
        instance1 = ProgressiveToolsRegistry.get_instance()
        instance2 = ProgressiveToolsRegistry.get_instance()

        assert instance1 is instance2

    def test_reset_instance_clears_singleton(self, clean_registry):
        """Test that reset_instance creates a new singleton."""
        instance1 = ProgressiveToolsRegistry.get_instance()
        instance1.register("test_tool", {"param": {"min": 1}})

        ProgressiveToolsRegistry.reset_instance()

        instance2 = ProgressiveToolsRegistry.get_instance()

        assert instance1 is not instance2
        assert not instance2.is_progressive("test_tool")

    def test_convenience_function(self):
        """Test get_progressive_registry convenience function."""
        ProgressiveToolsRegistry.reset_instance()

        registry = get_progressive_registry()

        assert isinstance(registry, ProgressiveToolsRegistry)
        assert registry is ProgressiveToolsRegistry.get_instance()


# =============================================================================
# Test Class: Tool Registration
# =============================================================================


@pytest.mark.integration
class TestProgressiveToolRegistration:
    """Tests for registering tools with progressive parameters."""

    def test_register_tool_with_params(self, clean_registry):
        """Test registering a tool with progressive parameters."""
        registry = clean_registry

        registry.register(
            tool_name="search",
            progressive_params={
                "limit": {"min": 10, "max": 100, "step": 10},
            },
            initial_values={"limit": 10},
            max_values={"limit": 100},
        )

        assert registry.is_progressive("search")

    def test_register_tool_minimal(self, clean_registry):
        """Test registering with minimal configuration."""
        registry = clean_registry

        registry.register(
            tool_name="simple_tool",
            progressive_params={"depth": {"escalate": True}},
        )

        assert registry.is_progressive("simple_tool")

        config = registry.get_config("simple_tool")
        assert config is not None
        assert config.tool_name == "simple_tool"

    def test_is_progressive_false_for_unregistered(self, clean_registry):
        """Test is_progressive returns False for unregistered tools."""
        registry = clean_registry

        assert not registry.is_progressive("nonexistent_tool")
        assert not registry.is_progressive("")

    def test_get_config_returns_none_for_unregistered(self, clean_registry):
        """Test get_config returns None for unregistered tools."""
        registry = clean_registry

        config = registry.get_config("nonexistent")
        assert config is None

    def test_list_progressive_tools(self, populated_registry):
        """Test listing all progressive tools."""
        tools = populated_registry.list_progressive_tools()

        assert isinstance(tools, set)
        assert "code_search" in tools
        assert "read_file" in tools
        assert "web_search" in tools
        assert len(tools) == 3


# =============================================================================
# Test Class: ProgressiveToolConfig
# =============================================================================


@pytest.mark.integration
class TestProgressiveToolConfig:
    """Tests for ProgressiveToolConfig data class."""

    def test_config_attributes(self, populated_registry):
        """Test that config has correct attributes."""
        config = populated_registry.get_config("code_search")

        assert config is not None
        assert config.tool_name == "code_search"
        assert "max_results" in config.progressive_params
        assert "search_depth" in config.progressive_params

    def test_config_initial_values(self, populated_registry):
        """Test initial values are stored correctly."""
        config = populated_registry.get_config("code_search")

        assert config.initial_values["max_results"] == 10
        assert config.initial_values["search_depth"] == 1

    def test_config_max_values(self, populated_registry):
        """Test max values are stored correctly."""
        config = populated_registry.get_config("code_search")

        assert config.max_values["max_results"] == 100
        assert config.max_values["search_depth"] == 5

    def test_config_with_boolean_params(self, populated_registry):
        """Test config handles boolean parameters."""
        config = populated_registry.get_config("web_search")

        assert config.initial_values["include_snippets"] is False
        assert config.max_values["include_snippets"] is True


# =============================================================================
# Test Class: Progressive Parameter Escalation Logic
# =============================================================================


@pytest.mark.integration
class TestProgressiveParameterEscalation:
    """Tests for progressive parameter escalation behavior."""

    def test_escalation_calculation(self, populated_registry):
        """Test calculating escalated parameter values."""
        config = populated_registry.get_config("code_search")

        # Simulate escalation logic
        initial = config.initial_values["max_results"]
        max_val = config.max_values["max_results"]
        step = config.progressive_params["max_results"].get("step", 10)

        # First escalation
        escalated_1 = min(initial + step, max_val)
        assert escalated_1 == 20

        # Multiple escalations
        escalated_5 = min(initial + (step * 5), max_val)
        assert escalated_5 == 60

        # Cap at max
        escalated_20 = min(initial + (step * 20), max_val)
        assert escalated_20 == 100

    def test_escalation_respects_max(self, populated_registry):
        """Test that escalation doesn't exceed max values."""
        config = populated_registry.get_config("read_file")

        initial = config.initial_values["max_lines"]
        max_val = config.max_values["max_lines"]
        step = config.progressive_params["max_lines"].get("step", 100)

        # Try to escalate beyond max
        over_max = initial + (step * 100)
        capped = min(over_max, max_val)

        assert capped == max_val
        assert capped == 1000

    def test_multiple_params_escalate_independently(self, populated_registry):
        """Test that multiple parameters escalate independently."""
        config = populated_registry.get_config("code_search")

        # Start with initial values
        current = config.initial_values.copy()
        assert current["max_results"] == 10
        assert current["search_depth"] == 1

        # Escalate max_results only
        current["max_results"] = min(
            current["max_results"] + 10,
            config.max_values["max_results"],
        )

        assert current["max_results"] == 20
        assert current["search_depth"] == 1  # Unchanged


# =============================================================================
# Test Class: Integration with Tool Execution Context
# =============================================================================


@pytest.mark.integration
class TestProgressiveToolsExecutionContext:
    """Tests simulating progressive tools in execution context."""

    def test_progressive_tool_workflow(self, populated_registry):
        """Test complete workflow with progressive parameters."""
        # Simulate tool invocation with progressive escalation
        tool_name = "code_search"
        config = populated_registry.get_config(tool_name)

        # Iteration 1: Use initial values
        iteration_1_params = config.initial_values.copy()
        assert iteration_1_params["max_results"] == 10
        assert iteration_1_params["search_depth"] == 1

        # Simulate insufficient results - escalate for iteration 2
        step_results = config.progressive_params["max_results"].get("step", 10)
        iteration_2_params = iteration_1_params.copy()
        iteration_2_params["max_results"] = min(
            iteration_2_params["max_results"] + step_results,
            config.max_values["max_results"],
        )
        assert iteration_2_params["max_results"] == 20

        # Still insufficient - also escalate search_depth
        step_depth = config.progressive_params["search_depth"].get("step", 1)
        iteration_3_params = iteration_2_params.copy()
        iteration_3_params["search_depth"] = min(
            iteration_3_params["search_depth"] + step_depth,
            config.max_values["search_depth"],
        )
        assert iteration_3_params["max_results"] == 20
        assert iteration_3_params["search_depth"] == 2

    def test_non_progressive_tool_check(self, populated_registry):
        """Test that non-progressive tools are handled correctly."""
        tool_name = "regular_tool"

        is_progressive = populated_registry.is_progressive(tool_name)
        assert not is_progressive

        config = populated_registry.get_config(tool_name)
        assert config is None

    def test_registry_isolation_between_tests(self):
        """Test that registry state is properly isolated."""
        ProgressiveToolsRegistry.reset_instance()
        registry = ProgressiveToolsRegistry.get_instance()

        # Should start empty
        assert len(registry.list_progressive_tools()) == 0

        # Add a tool
        registry.register("isolated_tool", {"param": {}})

        assert registry.is_progressive("isolated_tool")
        assert len(registry.list_progressive_tools()) == 1


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


@pytest.mark.integration
class TestProgressiveToolsEdgeCases:
    """Tests for edge cases in progressive tools registry."""

    def test_register_same_tool_twice(self, clean_registry):
        """Test registering the same tool overwrites previous config."""
        registry = clean_registry

        registry.register(
            tool_name="duplicate",
            progressive_params={"v1": {"min": 1}},
            initial_values={"v1": 1},
        )

        registry.register(
            tool_name="duplicate",
            progressive_params={"v2": {"min": 2}},
            initial_values={"v2": 2},
        )

        config = registry.get_config("duplicate")
        assert "v2" in config.progressive_params
        assert "v1" not in config.progressive_params

    def test_empty_progressive_params(self, clean_registry):
        """Test registering with empty progressive params."""
        registry = clean_registry

        registry.register(
            tool_name="empty_params",
            progressive_params={},
        )

        assert registry.is_progressive("empty_params")
        config = registry.get_config("empty_params")
        assert config.progressive_params == {}

    def test_none_initial_and_max_values(self, clean_registry):
        """Test registering without initial and max values."""
        registry = clean_registry

        registry.register(
            tool_name="no_values",
            progressive_params={"param": {"escalate": True}},
        )

        config = registry.get_config("no_values")
        assert config.initial_values == {}
        assert config.max_values == {}

    def test_special_characters_in_tool_name(self, clean_registry):
        """Test tool names with special characters."""
        registry = clean_registry

        registry.register(
            tool_name="tool-with-dashes",
            progressive_params={"param": {}},
        )
        registry.register(
            tool_name="tool_with_underscores",
            progressive_params={"param": {}},
        )

        assert registry.is_progressive("tool-with-dashes")
        assert registry.is_progressive("tool_with_underscores")
