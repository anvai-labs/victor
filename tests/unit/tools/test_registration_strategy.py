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

"""Tests for tool registration strategy pattern."""

import subprocess
import sys
from typing import Any

import pytest
from unittest import mock

from victor.tools.registration.strategies.base import (
    ToolRegistrationStrategy,
    FunctionDecoratorStrategy,
    BaseToolSubclassStrategy,
    MCPDictStrategy,
)
from victor.tools.registration.strategies import DuckTypedToolStrategy
from victor.tools.registration.registry import ToolRegistrationStrategyRegistry


@pytest.fixture(autouse=True)
def isolate_strategy_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore both shared strategy and feature-flag state after every case."""
    from victor.core import feature_flags

    monkeypatch.setattr(ToolRegistrationStrategyRegistry, "_instance", None)
    monkeypatch.setattr(feature_flags, "_global_manager", None)


def test_legacy_strategy_exports_resolve_in_a_fresh_interpreter() -> None:
    """The old import path resolves first and preserves canonical class identities."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """import victor.tools.registration.strategies.base as legacy
from victor.tools.registration.strategies import strategies as canonical
names = ('ToolRegistrationStrategy', 'FunctionDecoratorStrategy',
         'BaseToolSubclassStrategy', 'MCPDictStrategy')
assert set(legacy.__all__) == set(names)
for name in names:
    assert getattr(legacy, name) is getattr(canonical, name), name
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
        """Distinguish a real BaseTool instance from other plugin/tool shapes."""
        from victor.tools.base import BaseTool, ToolResult

        class ConcreteTool(BaseTool):
            name = "test_tool"
            description = "Test registration"
            parameters: dict[str, Any] = {}

            async def execute(self, _exec_ctx: dict[str, Any], **kwargs: Any) -> ToolResult:
                return ToolResult(success=True, output="ok")

        strategy = BaseToolSubclassStrategy()
        assert strategy.can_handle(ConcreteTool()) is True
        assert strategy.can_handle(_DuckTypedTool()) is False
        assert strategy.can_handle(object()) is False

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
        """get_instance shares one registry; explicit construction remains independent."""
        registry = ToolRegistrationStrategyRegistry.get_instance()
        assert ToolRegistrationStrategyRegistry.get_instance() is registry
        assert ToolRegistrationStrategyRegistry() is not registry

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

        from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

        get_feature_flag_manager().enable(FeatureFlag.USE_STRATEGY_BASED_TOOL_REGISTRATION)
        registry = ToolRegistry()
        strategy = CustomStrategy()
        registry.add_custom_strategy(strategy)
        custom = CustomTool()
        assert registry._strategy_registry.get_strategy_for(custom) is strategy
        registry.register(custom)
        assert registry.get("custom_tool") is custom
        with pytest.raises(TypeError, match="No registration strategy found"):
            registry.register(object())

    @pytest.mark.parametrize("use_strategy", [False, True])
    def test_register_decorated_function_via_strategy(self, use_strategy: bool):
        """Legacy and opt-in strategy dispatch register the same tool instance."""
        from victor.tools.registry import ToolRegistry
        from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

        manager = get_feature_flag_manager()
        flag = FeatureFlag.USE_STRATEGY_BASED_TOOL_REGISTRATION
        if use_strategy:
            manager.enable(flag)
        else:
            manager.disable(flag)

        # Create a mock @tool decorated function
        def my_tool():
            pass

        my_tool._is_victor_tool = True
        my_tool.Tool = mock.Mock()
        my_tool.Tool.name = "my_tool"

        # Register with strategy pattern
        registry = ToolRegistry()
        registry.register(my_tool)

        assert (registry._strategy_registry is not None) is use_strategy
        assert registry._tools["my_tool"] is registry.get("my_tool") is my_tool.Tool
