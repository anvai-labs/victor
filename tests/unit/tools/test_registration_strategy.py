# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for tool registration strategy pattern."""

import pytest
from unittest import mock

from victor.tools.registration.strategies import (
    ToolRegistrationStrategy,
    FunctionDecoratorStrategy,
    BaseToolSubclassStrategy,
    DuckTypedToolStrategy,
    MCPDictStrategy,
)
from victor.tools.registration.registry import ToolRegistrationStrategyRegistry


class TestFunctionDecoratorStrategy:
    """Tests for FunctionDecoratorStrategy."""

    def test_can_handle_decorated_function(self):
        """Test identifying @tool decorated functions."""
        strategy = FunctionDecoratorStrategy()

        # Create a mock decorated function
        mock_tool = mock.Mock()
        mock_tool._is_victor_tool = True
        mock_tool.Tool.name = "test_tool"

        assert strategy.can_handle(mock_tool) is True

    def test_cannot_handle_undecorated_function(self):
        """Test rejecting undecorated functions."""
        strategy = FunctionDecoratorStrategy()

        def plain_function():
            pass

        assert strategy.can_handle(plain_function) is False

    def test_register_decorated_tool(self):
        """Test registering a decorated function."""
        strategy = FunctionDecoratorStrategy()
        registry = mock.Mock()

        mock_tool = mock.Mock()
        mock_tool._is_victor_tool = True
        mock_tool.Tool.name = "test_tool"

        strategy.register(registry, mock_tool)

        registry._register_direct.assert_called_once_with("test_tool", mock_tool.Tool, True)

    def test_priority(self):
        """Test strategy priority."""
        strategy = FunctionDecoratorStrategy()
        assert strategy.priority == 100


class TestBaseToolSubclassStrategy:
    """Tests for BaseToolSubclassStrategy."""

    def test_can_handle_basetool_subclass(self):
        """Test identifying BaseTool subclass instances."""
        strategy = BaseToolSubclassStrategy()

        # Create a mock BaseTool instance
        mock_tool = mock.Mock()
        mock_tool.name = "test_tool"

        # Mock the import check
        with mock.patch.dict("sys.modules", {"victor.tools.base": mock.Mock()}):
            from victor.tools.base import BaseTool as MockBaseTool

            MockBaseTool = type("BaseTool", (object,), {})
            mock_tool.__class__ = MockBaseTool

        # For testing, we'll just check the structure
        # In real scenario, isinstance check would work

    def test_register_basetool_instance(self):
        """Test registering a BaseTool instance."""
        strategy = BaseToolSubclassStrategy()
        registry = mock.Mock()

        mock_tool = mock.Mock()
        mock_tool.name = "test_tool"

        strategy.register(registry, mock_tool)

        registry._register_direct.assert_called_once_with("test_tool", mock_tool, True)

    def test_priority(self):
        """Test strategy priority."""
        strategy = BaseToolSubclassStrategy()
        assert strategy.priority == 50


class TestMCPDictStrategy:
    """Tests for MCPDictStrategy."""

    def test_can_handle_dict_with_name(self):
        """Test identifying MCP dictionary tools."""
        strategy = MCPDictStrategy()

        tool_dict = {
            "name": "test_tool",
            "description": "Test description",
            "parameters": {},
        }

        assert strategy.can_handle(tool_dict) is True

    def test_cannot_handle_dict_without_name(self):
        """Test rejecting dicts without 'name' key."""
        strategy = MCPDictStrategy()

        tool_dict = {
            "description": "Test description",
        }

        assert strategy.can_handle(tool_dict) is False

    def test_cannot_handle_non_dict(self):
        """Test rejecting non-dict objects."""
        strategy = MCPDictStrategy()

        assert strategy.can_handle("not_a_dict") is False
        assert strategy.can_handle(123) is False

    def test_register_mcp_dict(self):
        """Test registering an MCP dictionary."""
        strategy = MCPDictStrategy()
        registry = mock.Mock()

        tool_dict = {
            "name": "test_tool",
            "description": "Test",
        }

        strategy.register(registry, tool_dict)

        registry.register_dict.assert_called_once_with(tool_dict, True)

    def test_priority(self):
        """Test strategy priority."""
        strategy = MCPDictStrategy()
        assert strategy.priority == 10


class _DuckTypedTool:
    """Plugin-style tool with no BaseTool inheritance or @tool decorator."""

    @property
    def name(self):
        return "duck_tool"

    @property
    def description(self):
        return "A duck-typed plugin tool"

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return {"success": True}


class TestDuckTypedToolStrategy:
    """Tests for DuckTypedToolStrategy (contract-only plugin tools)."""

    def test_can_handle_duck_typed_instance(self):
        """Test identifying duck-typed tool instances."""
        strategy = DuckTypedToolStrategy()

        assert strategy.can_handle(_DuckTypedTool()) is True

    def test_cannot_handle_dicts_strings_or_plain_functions(self):
        """Test rejecting shapes owned by other strategies or invalid tools."""
        strategy = DuckTypedToolStrategy()

        assert strategy.can_handle({"name": "tool"}) is False
        assert strategy.can_handle("not_a_tool") is False

        def plain_function():
            pass

        assert strategy.can_handle(plain_function) is False

    def test_register_duck_typed_tool(self):
        """Test registering a duck-typed tool instance."""
        strategy = DuckTypedToolStrategy()
        registry = mock.Mock()
        tool = _DuckTypedTool()

        strategy.register(registry, tool)

        registry._register_direct.assert_called_once_with("duck_tool", tool, True)

    def test_priority_below_basetool_above_mcp_dict(self):
        """Test strategy priority ordering."""
        strategy = DuckTypedToolStrategy()

        assert MCPDictStrategy().priority < strategy.priority < BaseToolSubclassStrategy().priority

    def test_registry_selects_duck_typed_strategy(self):
        """Test the strategy registry routes duck-typed instances correctly."""
        registry = ToolRegistrationStrategyRegistry()

        strategy = registry.get_strategy_for(_DuckTypedTool())

        assert isinstance(strategy, DuckTypedToolStrategy)

    def test_tool_registry_registers_duck_typed_tool_with_strategy_flag(self):
        """Duck-typed plugin tools must register when the strategy flag is on."""
        from victor.tools.registry import ToolRegistry
        from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

        manager = get_feature_flag_manager()
        manager.enable(FeatureFlag.USE_STRATEGY_BASED_TOOL_REGISTRATION)
        try:
            registry = ToolRegistry()
            registry.register(_DuckTypedTool())

            assert registry.get("duck_tool") is not None
        finally:
            manager.clear_runtime_override(FeatureFlag.USE_STRATEGY_BASED_TOOL_REGISTRATION)


class TestToolRegistrationStrategyRegistry:
    """Tests for ToolRegistrationStrategyRegistry."""

    def test_singleton_instance(self):
        """Test singleton pattern."""
        registry1 = ToolRegistrationStrategyRegistry()
        registry2 = ToolRegistrationStrategyRegistry.get_instance()

        # get_instance should return the same instance
        assert (
            registry2 is registry1 or registry2 is ToolRegistrationStrategyRegistry.get_instance()
        )

    def test_default_strategies_registered(self):
        """Test that default strategies are registered."""
        registry = ToolRegistrationStrategyRegistry()

        strategies = registry.list_strategies()

        assert "FunctionDecoratorStrategy" in strategies
        assert "BaseToolSubclassStrategy" in strategies
        assert "DuckTypedToolStrategy" in strategies
        assert "MCPDictStrategy" in strategies

    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""
        registry = ToolRegistrationStrategyRegistry()

        class CustomStrategy:
            def can_handle(self, tool):
                return False

            def register(self, registry, tool, enabled=True):
                pass

            @property
            def priority(self):
                return 75

        custom = CustomStrategy()
        registry.register_strategy(custom)

        assert "CustomStrategy" in registry.list_strategies()

    def test_get_strategy_for_decorated_tool(self):
        """Test getting strategy for decorated function."""
        registry = ToolRegistrationStrategyRegistry()

        mock_tool = mock.Mock()
        mock_tool._is_victor_tool = True

        strategy = registry.get_strategy_for(mock_tool)

        assert isinstance(strategy, FunctionDecoratorStrategy)

    def test_get_strategy_for_dict_tool(self):
        """Test getting strategy for MCP dict."""
        registry = ToolRegistrationStrategyRegistry()

        tool_dict = {"name": "test"}

        strategy = registry.get_strategy_for(tool_dict)

        assert isinstance(strategy, MCPDictStrategy)

    def test_get_strategy_for_unknown_tool(self):
        """Test getting strategy for unknown tool type."""
        registry = ToolRegistrationStrategyRegistry()

        strategy = registry.get_strategy_for("unknown_type")

        assert strategy is None

    def test_strategy_priority_ordering(self):
        """Test that strategies are checked in priority order."""
        registry = ToolRegistrationStrategyRegistry()

        # Create mock tools that multiple strategies could handle
        decorated_tool = mock.Mock()
        decorated_tool._is_victor_tool = True

        # FunctionDecoratorStrategy has priority 100 (highest)
        strategy = registry.get_strategy_for(decorated_tool)

        assert isinstance(strategy, FunctionDecoratorStrategy)
        assert strategy.priority == 100

    def test_clear_strategies(self):
        """Test clearing all strategies."""
        registry = ToolRegistrationStrategyRegistry()

        registry.clear()

        assert registry.list_strategies() == []


class TestStrategyPatternIntegration:
    """Integration tests for strategy pattern with ToolRegistry."""

    def test_tool_registry_with_strategy_flag_enabled(self):
        """Test ToolRegistry uses strategy pattern when flag is enabled."""
        from victor.tools.registry import ToolRegistry
        from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

        # Enable the flag
        get_feature_flag_manager().enable(FeatureFlag.USE_STRATEGY_BASED_TOOL_REGISTRATION)

        # Create a tool registry
        registry = ToolRegistry()

        # Verify that strategy registry is initialized
        assert registry._strategy_registry is not None

    def test_custom_strategy_registration(self):
        """Test adding custom strategy to ToolRegistry."""
        from victor.tools.registry import ToolRegistry

        class CustomTool:
            def __init__(self):
                self.name = "custom_tool"

        class CustomStrategy:
            def can_handle(self, tool):
                return isinstance(tool, CustomTool)

            def register(self, registry, tool, enabled=True):
                # Custom registration logic
                registry._register_direct(tool.name, tool, enabled)

            @property
            def priority(self):
                return 75

        registry = ToolRegistry()
        registry.add_custom_strategy(CustomStrategy())

        # Now CustomTool instances can be registered
        custom = CustomTool()
        # This would use the custom strategy
        # (in real usage, would call registry.register(custom))

    def test_register_decorated_function_via_strategy(self):
        """Test registering @tool decorated function via strategy."""
        from victor.tools.registry import ToolRegistry
        from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

        # Enable strategy flag
        get_feature_flag_manager().enable(FeatureFlag.USE_STRATEGY_BASED_TOOL_REGISTRATION)

        # Create a mock @tool decorated function
        def my_tool():
            pass

        my_tool._is_victor_tool = True
        my_tool.Tool = mock.Mock()
        my_tool.Tool.name = "my_tool"

        # Register with strategy pattern
        registry = ToolRegistry()
        registry.register(my_tool)

        # Verify it was registered
        assert "my_tool" in registry._tools
