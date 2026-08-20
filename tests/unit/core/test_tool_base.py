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

"""Tests for tools/base.py module."""

import pytest
from typing import Dict, Any

from victor.tools.base import (
    ToolParameter,
    ToolResult,
    BaseTool,
    ToolRegistry,
)


class TestToolParameter:
    """Tests for ToolParameter model."""

    def test_tool_parameter_creation(self):
        """Test creating a ToolParameter."""
        param = ToolParameter(name="test_param", type="string", description="A test parameter")

        assert param.name == "test_param"
        assert param.type == "string"
        assert param.description == "A test parameter"
        assert param.required is True
        assert param.enum is None

    def test_tool_parameter_with_enum(self):
        """Test ToolParameter with enum values."""
        param = ToolParameter(
            name="choice",
            type="string",
            description="A choice parameter",
            enum=["option1", "option2", "option3"],
            required=False,
        )

        assert param.enum == ["option1", "option2", "option3"]
        assert param.required is False


class TestToolResult:
    """Tests for ToolResult model."""

    def test_tool_result_success(self):
        """Test successful ToolResult."""
        result = ToolResult(success=True, output={"data": "test"})

        assert result.success is True
        assert result.output == {"data": "test"}
        assert result.error is None
        assert result.metadata is None

    def test_tool_result_failure(self):
        """Test failed ToolResult."""
        result = ToolResult(
            success=False,
            output=None,
            error="Test error",
            metadata={"exception": "ValueError"},
        )

        assert result.success is False
        assert result.output is None
        assert result.error == "Test error"
        assert result.metadata == {"exception": "ValueError"}


class ConcreteTool(BaseTool):
    """Concrete implementation of BaseTool for testing."""

    def __init__(self):
        """Initialize the concrete tool."""
        self._name = "test_tool"
        self._description = "A test tool"
        self._parameters = {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First param"},
                "param2": {"type": "number", "description": "Second param"},
            },
            "required": ["param1"],
        }

    @property
    def name(self) -> str:
        """Return tool name."""
        return self._name

    @property
    def description(self) -> str:
        """Return tool description."""
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        """Return tool parameters."""
        return self._parameters

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute the tool."""
        return ToolResult(success=True, output=kwargs)


class TestBaseTool:
    """Tests for BaseTool abstract class."""

    def test_convert_parameters_to_schema_basic(self):
        """Test converting parameters to JSON schema."""
        params = [
            ToolParameter(name="param1", type="string", description="First", required=True),
            ToolParameter(name="param2", type="number", description="Second", required=False),
        ]

        schema = BaseTool.convert_parameters_to_schema(params)

        assert schema["type"] == "object"
        assert "param1" in schema["properties"]
        assert "param2" in schema["properties"]
        assert schema["properties"]["param1"]["type"] == "string"
        assert schema["properties"]["param2"]["type"] == "number"
        assert schema["required"] == ["param1"]

    def test_convert_parameters_to_schema_with_enum(self):
        """Test converting parameters with enum to schema."""
        params = [
            ToolParameter(
                name="choice",
                type="string",
                description="A choice",
                enum=["a", "b", "c"],
                required=True,
            )
        ]

        schema = BaseTool.convert_parameters_to_schema(params)

        assert schema["properties"]["choice"]["enum"] == ["a", "b", "c"]

    def test_convert_parameters_to_schema_no_required(self):
        """Test schema with no required parameters."""
        params = [ToolParameter(name="opt1", type="string", description="Optional", required=False)]

        schema = BaseTool.convert_parameters_to_schema(params)

        assert "required" not in schema

    def test_to_json_schema(self):
        """Test converting tool to JSON schema."""
        tool = ConcreteTool()
        schema = tool.to_json_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool"
        assert schema["function"]["parameters"] == tool.parameters

    def test_validate_parameters_valid(self):
        """Test validating valid parameters."""
        tool = ConcreteTool()

        assert tool.validate_parameters(param1="test", param2=42) is True
        assert tool.validate_parameters(param1="test") is True  # param2 is optional

    def test_validate_parameters_missing_required(self):
        """Test validation fails with missing required parameter."""
        tool = ConcreteTool()

        assert tool.validate_parameters(param2=42) is False

    def test_validate_parameters_wrong_type(self):
        """Test validation fails with wrong parameter type."""
        tool = ConcreteTool()

        # param2 should be number, not string
        assert tool.validate_parameters(param1="test", param2="not a number") is False

    def test_check_type_string(self):
        """Test type checking for strings."""
        tool = ConcreteTool()

        assert tool._check_type("test", "string") is True
        assert tool._check_type(123, "string") is False

    def test_check_type_number(self):
        """Test type checking for numbers."""
        tool = ConcreteTool()

        assert tool._check_type(42, "number") is True
        assert tool._check_type(3.14, "number") is True
        assert tool._check_type("not a number", "number") is False

    def test_check_type_integer(self):
        """Test type checking for integers."""
        tool = ConcreteTool()

        assert tool._check_type(42, "integer") is True
        assert tool._check_type(3.14, "integer") is False

    def test_check_type_boolean(self):
        """Test type checking for booleans."""
        tool = ConcreteTool()

        assert tool._check_type(True, "boolean") is True
        assert tool._check_type(False, "boolean") is True
        assert tool._check_type(1, "boolean") is False

    def test_check_type_array(self):
        """Test type checking for arrays."""
        tool = ConcreteTool()

        assert tool._check_type([1, 2, 3], "array") is True
        assert tool._check_type({"key": "value"}, "array") is False

    def test_check_type_object(self):
        """Test type checking for objects."""
        tool = ConcreteTool()

        assert tool._check_type({"key": "value"}, "object") is True
        assert tool._check_type([1, 2, 3], "object") is False

    def test_check_type_unknown(self):
        """Test type checking with unknown type."""
        tool = ConcreteTool()

        # Unknown types should be allowed
        assert tool._check_type("anything", "unknown_type") is True


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_registry_initialization(self):
        """Test creating a ToolRegistry."""
        registry = ToolRegistry()

        assert len(registry._tools) == 0
        assert len(registry._before_hooks) == 0
        assert len(registry._after_hooks) == 0

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = ConcreteTool()

        registry.register(tool)

        assert "test_tool" in registry._tools
        assert registry._tools["test_tool"] == tool

    def test_register_invalid_type(self):
        """Test registering invalid type raises error."""
        registry = ToolRegistry()

        with pytest.raises(
            TypeError,
            match="Can only register BaseTool|No registration strategy found",
        ):
            registry.register("not a tool")

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = ConcreteTool()

        registry.register(tool)
        assert "test_tool" in registry._tools

        registry.unregister("test_tool")
        assert "test_tool" not in registry._tools

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent tool doesn't error."""
        registry = ToolRegistry()

        # Should not raise error
        registry.unregister("nonexistent")

    def test_higher_priority_registration_replaces_lower_priority_conflict(self):
        """Native tools should replace lower-priority conflicting tools."""
        registry = ToolRegistry()

        class MCPStyleTool(ConcreteTool):
            def __init__(self):
                super().__init__()
                self._name = "mcp_test_tool"
                self._tool_source = "mcp"

        low_priority_tool = MCPStyleTool()
        high_priority_tool = ConcreteTool()

        registry.register(low_priority_tool)
        registry.register(high_priority_tool)

        assert registry.get("test_tool") is high_priority_tool
        assert registry.get("mcp_test_tool") is None

    def test_clear_resets_auxiliary_registry_state(self):
        """Clearing the registry should reset caches and state-tracking indexes."""
        registry = ToolRegistry()

        class ConfigurableTool(ConcreteTool):
            @property
            def requires_configuration(self) -> bool:
                return True

        registry.register(ConfigurableTool())
        registry.get_tool_schemas()

        registry.clear()

        assert registry.list_tools() == []
        assert registry.get_tool_schemas() == []
        assert registry._dynamic_availability_tools == set()

    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        tool = ConcreteTool()
        registry.register(tool)

        retrieved = registry.get("test_tool")

        assert retrieved == tool

    def test_get_nonexistent_tool(self):
        """Test getting non-existent tool returns None."""
        registry = ToolRegistry()

        assert registry.get("nonexistent") is None

    def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()
        tool1 = ConcreteTool()

        # Create a second tool
        class AnotherTool(ConcreteTool):
            def __init__(self):
                super().__init__()
                self._name = "another_tool"

        tool2 = AnotherTool()

        registry.register(tool1)
        registry.register(tool2)

        tools = registry.list_tools()

        assert len(tools) == 2
        assert tool1 in tools
        assert tool2 in tools

    def test_get_tool_schemas(self):
        """Test getting JSON schemas for all tools."""
        registry = ToolRegistry()
        tool = ConcreteTool()
        registry.register(tool)

        schemas = registry.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful tool execution."""
        registry = ToolRegistry()
        tool = ConcreteTool()
        registry.register(tool)

        result = await registry.execute("test_tool", {}, param1="test", param2=42)

        assert result.success is True
        assert result.output == {"param1": "test", "param2": 42}

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test executing non-existent tool."""
        registry = ToolRegistry()

        result = await registry.execute("nonexistent", {})

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_invalid_parameters(self):
        """Test executing with invalid parameters."""
        registry = ToolRegistry()
        tool = ConcreteTool()
        registry.register(tool)

        # Missing required param1
        result = await registry.execute("test_tool", {}, param2=42)

        assert result.success is False
        assert "Invalid parameters" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_exception(self):
        """Test tool execution that raises exception."""
        registry = ToolRegistry()

        class FailingTool(ConcreteTool):
            async def execute(self, context, **kwargs):
                raise ValueError("Test error")

        tool = FailingTool()
        registry.register(tool)

        # Need to provide required param1 to pass validation
        result = await registry.execute("test_tool", {}, param1="test")

        assert result.success is False
        assert "Tool execution failed" in result.error
        assert result.metadata["exception"] == "ValueError"

    def test_register_before_hook(self):
        """Test registering before-execution hook."""
        registry = ToolRegistry()

        def before_hook(name: str, kwargs: Dict[str, Any]) -> None:
            pass

        registry.register_before_hook(before_hook)

        assert len(registry._before_hooks) == 1
        # Hook wraps the callback - check the callback attribute
        assert registry._before_hooks[0].callback == before_hook

    def test_register_after_hook(self):
        """Test registering after-execution hook."""
        registry = ToolRegistry()

        def after_hook(result: ToolResult) -> None:
            pass

        registry.register_after_hook(after_hook)

        assert len(registry._after_hooks) == 1
        # Hook wraps the callback - check the callback attribute
        assert registry._after_hooks[0].callback == after_hook

    @pytest.mark.asyncio
    async def test_execute_with_hooks(self):
        """Test that hooks are called during execution."""
        registry = ToolRegistry()
        tool = ConcreteTool()
        registry.register(tool)

        before_called = []
        after_called = []

        def before_hook(name: str, kwargs: Dict[str, Any]) -> None:
            before_called.append((name, kwargs))

        def after_hook(result: ToolResult) -> None:
            after_called.append(result)

        registry.register_before_hook(before_hook)
        registry.register_after_hook(after_hook)

        result = await registry.execute("test_tool", {}, param1="test")

        assert len(before_called) == 1
        assert before_called[0][0] == "test_tool"
        assert before_called[0][1] == {"param1": "test"}

        assert len(after_called) == 1
        assert after_called[0] == result
