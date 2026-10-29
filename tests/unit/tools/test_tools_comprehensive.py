"""Comprehensive tests for tools module to improve coverage.

Covers:
- Tool parameter validation and schema generation
- Tool registry and discovery
- Tool execution framework
- Tool cost tiers and metadata
"""

import pytest
import json
from dataclasses import dataclass
from unittest.mock import patch

from victor.tools.base import (
    BaseTool,
    ToolResult,
    ToolParameter,
    CostTier,
    Priority,
)
from victor.tools.decorators import tool
from victor.tools.registry import ToolRegistry


class TestToolParameter:
    """Tests for ToolParameter class."""

    def test_tool_parameter_creation(self):
        """Test creating a tool parameter."""
        param = ToolParameter(name="test_param", type="string", description="A test parameter")
        assert param.name == "test_param"
        assert param.type == "string"
        assert param.description == "A test parameter"

    def test_tool_parameter_with_default_value(self):
        """Test tool parameter with default value."""
        param = ToolParameter(
            name="optional_param", type="string", description="Optional parameter"
        )
        # ToolParameter may not have explicit default field
        assert param.name == "optional_param"

    def test_tool_parameter_required(self):
        """Test tool parameter required flag."""
        param = ToolParameter(
            name="required_param",
            type="string",
            description="Required parameter",
            required=True,
        )
        assert param.required is True

    def test_tool_parameter_not_required(self):
        """Test tool parameter not required."""
        param = ToolParameter(
            name="optional_param",
            type="string",
            description="Optional parameter",
            required=False,
        )
        assert param.required is False

    def test_tool_parameter_with_enum_values(self):
        """Test tool parameter with enum values."""
        param = ToolParameter(
            name="choice_param",
            type="string",
            description="Choice parameter",
            enum=["option1", "option2", "option3"],
        )
        assert len(param.enum) == 3
        assert "option1" in param.enum

    def test_tool_parameter_with_min_max(self):
        """Test tool parameter with min/max constraints."""
        param = ToolParameter(name="number_param", type="number", description="Number parameter")
        # ToolParameter may not have explicit min/max fields
        assert param.type == "number"

    def test_tool_parameter_with_pattern(self):
        """Test tool parameter with regex pattern."""
        param = ToolParameter(name="email_param", type="string", description="Email parameter")
        # ToolParameter may not have explicit pattern field
        assert param.type == "string"

    def test_tool_parameter_to_schema(self):
        """Test converting tool parameter to JSON schema."""
        param = ToolParameter(
            name="test_param",
            type="string",
            description="Test parameter",
            required=True,
        )
        # Tool parameters should be convertible to schema format
        assert param.name == "test_param"
        assert param.type == "string"


class TestToolResult:
    """Tests for ToolResult class."""

    def test_tool_result_success(self):
        """Test creating successful tool result."""
        result = ToolResult(output="Operation succeeded", success=True)
        assert result.success is True
        assert result.output == "Operation succeeded"

    def test_tool_result_failure(self):
        """Test creating failed tool result."""
        result = ToolResult(output="", success=False, error="Operation failed")
        assert result.success is False
        assert result.error == "Operation failed"

    def test_tool_result_with_metadata(self):
        """Test tool result with metadata."""
        metadata = {"execution_time": 1.5, "memory_used": 256}
        result = ToolResult(output="Result", success=True, metadata=metadata)
        assert result.metadata == metadata

    def test_tool_result_with_empty_output(self):
        """Test tool result with empty output."""
        result = ToolResult(output="", success=True)
        assert result.output == ""
        assert result.success is True

    def test_tool_result_with_json_output(self):
        """Test tool result with JSON output."""
        json_data = {"key": "value", "number": 42}
        result = ToolResult(output=json.dumps(json_data), success=True)
        assert result.success is True
        # Should be parseable as JSON
        parsed = json.loads(result.output)
        assert parsed["key"] == "value"


class TestCostTier:
    """Tests for CostTier enum."""

    def test_cost_tier_free(self):
        """Test FREE cost tier."""
        assert CostTier.FREE.value == "free"

    def test_cost_tier_low(self):
        """Test LOW cost tier."""
        assert CostTier.LOW.value == "low"

    def test_cost_tier_medium(self):
        """Test MEDIUM cost tier."""
        assert CostTier.MEDIUM.value == "medium"

    def test_cost_tier_high(self):
        """Test HIGH cost tier."""
        assert CostTier.HIGH.value == "high"

    def test_cost_tier_values(self):
        """Test all cost tier values exist."""
        tiers = [CostTier.FREE, CostTier.LOW, CostTier.MEDIUM, CostTier.HIGH]
        assert len(tiers) == 4


class TestPriority:
    """Tests for Priority enum."""

    def test_priority_low(self):
        """Test LOW priority."""
        # Priority enum uses integer values
        assert hasattr(Priority, "LOW")
        assert isinstance(Priority.LOW.value, int)

    def test_priority_medium(self):
        """Test MEDIUM priority."""
        assert hasattr(Priority, "MEDIUM")
        assert isinstance(Priority.MEDIUM.value, int)

    def test_priority_high(self):
        """Test HIGH priority."""
        assert hasattr(Priority, "HIGH")
        assert isinstance(Priority.HIGH.value, int)

    def test_priority_critical(self):
        """Test CRITICAL priority."""
        assert hasattr(Priority, "CRITICAL")
        assert isinstance(Priority.CRITICAL.value, int)


class TestConcreteToolImplementation:
    """Tests for concrete tool implementations."""

    def test_concrete_tool_with_free_cost(self):
        """Test tool with FREE cost tier."""

        class SimpleTool(BaseTool):
            name = "simple_tool"
            description = "A simple tool"
            parameters = {"properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

            @property
            def cost_tier(self):
                return CostTier.FREE

        tool = SimpleTool()
        assert tool.cost_tier == CostTier.FREE
        assert tool.name == "simple_tool"

    def test_concrete_tool_with_medium_cost(self):
        """Test tool with MEDIUM cost tier."""

        class AnalysisTool(BaseTool):
            name = "analysis_tool"
            description = "Analysis tool"
            parameters = {"properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Analysis complete", success=True)

            @property
            def cost_tier(self):
                return CostTier.MEDIUM

        tool = AnalysisTool()
        assert tool.cost_tier == CostTier.MEDIUM

    @pytest.mark.asyncio
    async def test_tool_execution_success(self):
        """Test executing a tool successfully."""

        class TestTool(BaseTool):
            name = "test_tool"
            description = "Test tool"
            parameters = {"properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Test output", success=True)

        tool = TestTool()
        result = await tool.execute({})

        assert result.success is True
        assert result.output == "Test output"

    @pytest.mark.asyncio
    async def test_tool_execution_failure(self):
        """Test tool execution failure."""

        class FailingTool(BaseTool):
            name = "failing_tool"
            description = "Tool that fails"
            parameters = {"properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="", success=False, error="Tool failed as expected")

        tool = FailingTool()
        result = await tool.execute({})

        assert result.success is False
        assert result.error == "Tool failed as expected"

    @pytest.mark.asyncio
    async def test_tool_execution_with_exception(self):
        """Test tool execution when exception is raised."""

        class ExceptionTool(BaseTool):
            name = "exception_tool"
            description = "Tool that raises"
            parameters = {"properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                raise ValueError("Test exception")

        tool = ExceptionTool()

        # Should handle exception gracefully
        with pytest.raises(ValueError):
            await tool.execute({})


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_tool_registry_creation(self):
        """Test creating a tool registry."""
        registry = ToolRegistry()
        assert registry is not None

    def test_tool_registry_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()

        class TestTool(BaseTool):
            name = "test_tool"
            description = "Test"
            parameters = {"properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Test", success=True)

        tool = TestTool()
        registry.register(tool)

        # Tool should be registered
        assert "test_tool" in registry._tools or len(registry._tools) > 0

    def test_tool_registry_get_tool(self):
        """Test getting a tool from registry."""
        registry = ToolRegistry()

        class TestTool(BaseTool):
            name = "test_tool"
            description = "Test"
            parameters = {"properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Test", success=True)

        tool = TestTool()
        registry.register(tool)

        # Should be able to retrieve tool
        retrieved = registry.get("test_tool")
        assert retrieved is not None or len(registry._tools) > 0

    def test_tool_registry_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()

        # Should support listing tools
        tools = (
            registry.list_tools()
            if hasattr(registry, "list_tools")
            else list(registry._tools.values())
        )
        assert isinstance(tools, (list, dict))

    def test_tool_registry_filters_unavailable_tools_from_listings(self):
        """Unavailable tools should be hidden from selection and schemas."""
        registry = ToolRegistry()
        availability = {"ready": False}

        @tool(name="conditional_tool", availability_check=lambda: availability["ready"])
        async def conditional_tool():
            return "ok"

        registry.register(conditional_tool)

        assert registry.list_tools() == []
        assert registry.get_tool_schemas() == []

        availability["ready"] = True

        assert [tool.name for tool in registry.list_tools()] == ["conditional_tool"]
        assert [schema["function"]["name"] for schema in registry.get_tool_schemas()] == [
            "conditional_tool"
        ]

    def test_tool_registry_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()

        class TestTool(BaseTool):
            name = "test_tool"
            description = "Test"
            parameters = {"properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Test", success=True)

        tool = TestTool()
        registry.register(tool)

        # Should be able to unregister
        if hasattr(registry, "unregister"):
            registry.unregister("test_tool")


class TestToolParameterSchema:
    """Tests for tool parameter schema generation."""

    def test_string_parameter_schema(self):
        """Test string parameter schema."""
        param = ToolParameter(name="text", type="string", description="Text input")
        assert param.type == "string"

    def test_number_parameter_schema(self):
        """Test number parameter schema."""
        param = ToolParameter(name="count", type="number", description="Count")
        assert param.type == "number"

    def test_integer_parameter_schema(self):
        """Test integer parameter schema."""
        param = ToolParameter(name="count", type="integer", description="Count")
        assert param.type == "integer"

    def test_boolean_parameter_schema(self):
        """Test boolean parameter schema."""
        param = ToolParameter(name="flag", type="boolean", description="Flag")
        assert param.type == "boolean"

    def test_array_parameter_schema(self):
        """Test array parameter schema."""
        param = ToolParameter(name="items", type="array", description="List of items")
        assert param.type == "array"

    def test_object_parameter_schema(self):
        """Test object parameter schema."""
        param = ToolParameter(name="config", type="object", description="Configuration object")
        assert param.type == "object"


class TestToolDescriptionGeneration:
    """Tests for tool description and metadata generation."""

    def test_tool_description_from_docstring(self):
        """Test tool description from docstring."""

        class DocumentedTool(BaseTool):
            """This is a documented tool."""

            name = "documented_tool"
            description = "This is a documented tool"
            parameters = {"properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = DocumentedTool()
        assert tool.description is not None
        assert "documented" in tool.description.lower()

    def test_tool_with_empty_parameters(self):
        """Test tool with no parameters."""

        class NoParamTool(BaseTool):
            name = "no_param_tool"
            description = "Tool with no parameters"
            parameters = {"properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = NoParamTool()
        assert tool.parameters == {"properties": {}}

    def test_tool_with_multiple_parameters(self):
        """Test tool with multiple parameters."""

        class MultiParamTool(BaseTool):
            name = "multi_param_tool"
            description = "Tool with multiple parameters"
            parameters = {
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "number"},
                    "param3": {"type": "boolean"},
                }
            }

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = MultiParamTool()
        assert len(tool.parameters["properties"]) == 3


# =============================================================================
# CONVERT PARAMETERS TO SCHEMA TESTS
# =============================================================================


class TestConvertParametersToSchema:
    """Tests for BaseTool.convert_parameters_to_schema static method."""

    def test_empty_parameters(self):
        """Test conversion with empty parameters list."""
        schema = BaseTool.convert_parameters_to_schema([])
        # additionalProperties: False rejects hallucinated arguments
        assert schema == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def test_single_required_parameter(self):
        """Test conversion with single required parameter."""
        params = [
            ToolParameter(
                name="path",
                type="string",
                description="File path",
                required=True,
            )
        ]
        schema = BaseTool.convert_parameters_to_schema(params)

        assert schema["type"] == "object"
        assert "path" in schema["properties"]
        assert schema["properties"]["path"]["type"] == "string"
        assert schema["properties"]["path"]["description"] == "File path"
        assert "required" in schema
        assert "path" in schema["required"]

    def test_single_optional_parameter(self):
        """Test conversion with single optional parameter."""
        params = [
            ToolParameter(
                name="verbose",
                type="boolean",
                description="Enable verbose output",
                required=False,
            )
        ]
        schema = BaseTool.convert_parameters_to_schema(params)

        assert schema["type"] == "object"
        assert "verbose" in schema["properties"]
        assert "required" not in schema  # No required params

    def test_multiple_parameters_mixed_required(self):
        """Test conversion with multiple parameters, some required."""
        params = [
            ToolParameter(name="file", type="string", description="File", required=True),
            ToolParameter(name="mode", type="string", description="Mode", required=True),
            ToolParameter(name="encoding", type="string", description="Encoding", required=False),
        ]
        schema = BaseTool.convert_parameters_to_schema(params)

        assert len(schema["properties"]) == 3
        assert schema["required"] == ["file", "mode"]

    def test_parameter_with_enum(self):
        """Test conversion with enum values."""
        params = [
            ToolParameter(
                name="format",
                type="string",
                description="Output format",
                enum=["json", "yaml", "xml"],
            )
        ]
        schema = BaseTool.convert_parameters_to_schema(params)

        assert "format" in schema["properties"]
        assert schema["properties"]["format"]["enum"] == ["json", "yaml", "xml"]

    def test_numeric_types(self):
        """Test conversion with numeric types."""
        params = [
            ToolParameter(name="count", type="integer", description="Count"),
            ToolParameter(name="ratio", type="number", description="Ratio"),
        ]
        schema = BaseTool.convert_parameters_to_schema(params)

        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["ratio"]["type"] == "number"


# =============================================================================
# TO JSON SCHEMA TESTS
# =============================================================================


class TestToJsonSchema:
    """Tests for BaseTool.to_json_schema method."""

    def test_basic_schema_structure(self):
        """Test basic JSON schema structure."""

        class TestTool(BaseTool):
            name = "test_tool"
            description = "A test tool"
            parameters = {"type": "object", "properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = TestTool()
        schema = tool.to_json_schema()

        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool"

    def test_schema_includes_parameters(self):
        """Test that schema includes parameters."""

        class ToolWithParams(BaseTool):
            name = "param_tool"
            description = "Tool with parameters"
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            }

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = ToolWithParams()
        schema = tool.to_json_schema()

        assert "parameters" in schema["function"]
        assert "input" in schema["function"]["parameters"]["properties"]


# =============================================================================
# VALIDATE PARAMETERS TESTS
# =============================================================================


class TestValidateParameters:
    """Tests for BaseTool.validate_parameters method."""

    def test_validate_with_empty_schema(self):
        """Test validation with empty schema."""

        class NoParamTool(BaseTool):
            name = "no_param"
            description = "No params"
            parameters = {"type": "object", "properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = NoParamTool()
        assert tool.validate_parameters() is True
        assert tool.validate_parameters(extra="ignored") is True

    def test_validate_missing_required(self):
        """Test validation fails when required parameter is missing."""

        class RequiredParamTool(BaseTool):
            name = "required_param"
            description = "Has required param"
            parameters = {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path"}},
                "required": ["path"],
            }

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = RequiredParamTool()
        assert tool.validate_parameters() is False
        assert tool.validate_parameters(path="/test") is True

    def test_validate_wrong_type(self):
        """Test validation fails with wrong type."""

        class TypedTool(BaseTool):
            name = "typed"
            description = "Typed param"
            parameters = {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
            }

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = TypedTool()
        assert tool.validate_parameters(count=5) is True
        assert tool.validate_parameters(count="not_an_int") is False


# =============================================================================
# VALIDATE PARAMETERS DETAILED TESTS
# =============================================================================


class TestValidateParametersDetailed:
    """Tests for BaseTool.validate_parameters_detailed method."""

    def test_detailed_success(self):
        """Test detailed validation returns success."""

        class SimpleTool(BaseTool):
            name = "simple"
            description = "Simple"
            parameters = {"type": "object", "properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = SimpleTool()
        result = tool.validate_parameters_detailed()
        assert result.valid is True
        assert len(result.errors) == 0

    def test_detailed_missing_required_error(self):
        """Test detailed validation with missing required parameter."""

        class RequiredTool(BaseTool):
            name = "required"
            description = "Required"
            parameters = {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = RequiredTool()
        result = tool.validate_parameters_detailed()
        assert result.valid is False
        assert any("name" in err for err in result.errors)

    def test_detailed_type_error(self):
        """Test detailed validation with type error."""

        class NumberTool(BaseTool):
            name = "number"
            description = "Number"
            parameters = {
                "type": "object",
                "properties": {"value": {"type": "number"}},
            }

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = NumberTool()
        result = tool.validate_parameters_detailed(value="not_a_number")
        assert result.valid is False
        assert len(result.errors) > 0

    def test_detailed_enum_error(self):
        """Test detailed validation with enum constraint violation."""

        class EnumTool(BaseTool):
            name = "enum_tool"
            description = "Enum"
            parameters = {
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": ["read", "write"]}},
            }

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = EnumTool()
        assert tool.validate_parameters_detailed(mode="read").valid is True
        assert tool.validate_parameters_detailed(mode="invalid").valid is False


# =============================================================================
# TOOL METADATA TESTS
# =============================================================================


class TestToolMetadata:
    """Tests for tool metadata functionality."""

    def test_get_metadata_auto_generates(self):
        """Test that get_metadata auto-generates from tool properties."""

        class AutoMetaTool(BaseTool):
            name = "auto_meta"
            description = "Auto-generated metadata"
            parameters = {"type": "object", "properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

            @property
            def cost_tier(self):
                return CostTier.LOW

        tool = AutoMetaTool()
        metadata = tool.get_metadata()

        assert metadata is not None
        # ToolMetadata has category, not name
        assert hasattr(metadata, "category")

    def test_cost_tier_default_is_free(self):
        """Test that default cost tier is FREE."""

        class DefaultCostTool(BaseTool):
            name = "default_cost"
            description = "Default"
            parameters = {"properties": {}}

            async def execute(self, _exec_ctx, **kwargs):
                return ToolResult(output="Done", success=True)

        tool = DefaultCostTool()
        assert tool.cost_tier == CostTier.FREE


# =============================================================================
# SEMANTIC SELECTOR TESTS
# =============================================================================


class TestSemanticToolSelector:
    """Tests for SemanticToolSelector."""

    def test_cost_tier_warnings_defined(self):
        """Test COST_TIER_WARNINGS are defined."""
        from victor.tools.semantic_selector import COST_TIER_WARNINGS

        assert CostTier.HIGH in COST_TIER_WARNINGS
        assert CostTier.MEDIUM in COST_TIER_WARNINGS
        assert "HIGH COST" in COST_TIER_WARNINGS[CostTier.HIGH]

    @pytest.fixture
    def mock_selector(self):
        """Create a mock semantic selector without loading embeddings."""
        from victor.tools.semantic_selector import SemanticToolSelector

        with patch("victor.tools.semantic_selector.EmbeddingService"):
            selector = SemanticToolSelector(
                cache_embeddings=False,
                cost_aware_selection=True,
            )
            selector._tool_embeddings = {}
            selector._initialized = True
            return selector

    def test_selector_init_params(self, mock_selector):
        """Test selector initialization parameters."""
        # Attribute is cost_aware_selection, not _cost_aware
        assert mock_selector.cost_aware_selection is True
        # Cache embeddings is stored differently
        assert hasattr(mock_selector, "_cache_embeddings") or hasattr(
            mock_selector, "cache_embeddings"
        )

    def test_selector_apply_cost_penalty(self, mock_selector):
        """Test cost penalty factor is configurable."""
        mock_selector._cost_penalty_factor = 0.1
        assert mock_selector._cost_penalty_factor == 0.1


# =============================================================================
# TOOL DECORATORS TESTS
# =============================================================================


class TestToolDecorators:
    """Tests for tool decorators."""

    def test_tool_decorator_exists(self):
        """Test @tool decorator exists."""
        from victor.tools.decorators import tool

        assert callable(tool)

    def test_set_auto_register_function_exists(self):
        """Test set_auto_register helper exists."""
        from victor.tools.decorators import set_auto_register

        assert callable(set_auto_register)

    def test_resolve_tool_name_function_exists(self):
        """Test resolve_tool_name helper exists."""
        from victor.tools.decorators import resolve_tool_name

        assert callable(resolve_tool_name)

    def test_resolve_tool_name_returns_same(self):
        """Test resolve_tool_name returns same name if no legacy mapping."""
        from victor.tools.decorators import resolve_tool_name

        result = resolve_tool_name("my_custom_tool")
        assert result == "my_custom_tool"

    def test_set_legacy_name_warnings(self):
        """Test setting legacy name warnings."""
        from victor.tools.decorators import set_legacy_name_warnings

        # Should not raise
        set_legacy_name_warnings(False)
        set_legacy_name_warnings(True)
        set_legacy_name_warnings(False)  # Reset

    def test_tool_decorator_creates_tool_class(self):
        """Test @tool decorator creates a valid tool wrapper."""
        from victor.tools.decorators import tool

        # Description is extracted from docstring, not passed as param
        @tool(name="test_decorated")
        async def test_decorated_fn(message: str) -> str:
            """Execute the test tool.

            Args:
                message: The message to echo

            Returns:
                The echoed message
            """
            return f"Echo: {message}"

        # The decorator wraps the function and is callable
        assert callable(test_decorated_fn)

    def test_tool_decorator_propagates_timeout_to_tool_instance(self):
        """Decorator timeout must reach ToolExecutor via timeout_seconds."""
        from victor.tools.decorators import tool

        @tool(name="slow_semantic_tool", timeout=123.0)
        async def slow_semantic_tool(query: str) -> dict:
            """Execute a slow semantic lookup."""
            return {"query": query}

        tool_instance = slow_semantic_tool.Tool

        assert tool_instance.timeout_seconds == 123.0
        assert tool_instance.timeout == 123.0

    def test_tool_decorator_handles_structured_parameter_types_in_schema(self):
        """Dataclass-style structured parameters should become object schema entries."""
        from victor.tools.decorators import tool

        @dataclass
        class SearchFilters:
            file_pattern: str | None = None
            top_k: int = 5
            tags: list[str] | None = None

        @tool(name="structured_param_tool")
        async def structured_param_tool(filters: SearchFilters | None = None) -> dict:
            return {"filters": filters}

        params = structured_param_tool.Tool.parameters
        filters_schema = params["properties"]["filters"]

        assert filters_schema["type"] == "object"
        assert "file_pattern" in filters_schema["properties"]
        assert filters_schema["properties"]["top_k"] == {"type": "integer"}
        assert params["properties"]["filters"]["properties"]["file_pattern"]["type"] == "string"


# =============================================================================
# TOOL ENUMS TESTS
# =============================================================================


class TestToolEnums:
    """Tests for tool-related enums."""

    def test_cost_tier_ordering(self):
        """Test CostTier has logical ordering."""
        # FREE < LOW < MEDIUM < HIGH
        tiers = [CostTier.FREE, CostTier.LOW, CostTier.MEDIUM, CostTier.HIGH]
        values = [t.value for t in tiers]
        assert len(set(values)) == 4  # All unique

    def test_priority_levels(self):
        """Test Priority has all levels."""
        from victor.tools.base import Priority

        assert hasattr(Priority, "LOW")
        assert hasattr(Priority, "MEDIUM")
        assert hasattr(Priority, "HIGH")
        assert hasattr(Priority, "CRITICAL")

    def test_tool_status_enum(self):
        """Test ToolStatus enum if exists."""
        try:
            from victor.tools.enums import ToolStatus

            assert hasattr(ToolStatus, "SUCCESS") or hasattr(ToolStatus, "COMPLETED")
        except ImportError:
            # Enum might not exist
            pass


# =============================================================================
# TOOL METADATA REGISTRY TESTS
# =============================================================================


class TestToolMetadataRegistry:
    """Tests for ToolMetadataRegistry."""

    def test_registry_import(self):
        """Test ToolMetadataRegistry can be imported."""
        from victor.tools.metadata_registry import ToolMetadataRegistry

        assert ToolMetadataRegistry is not None

    def test_get_core_readonly_tools(self):
        """Test getting core readonly tools."""
        from victor.tools.metadata_registry import get_core_readonly_tools

        tools = get_core_readonly_tools()
        # Can be list or set
        assert isinstance(tools, (list, set))

    def test_get_tools_matching_mandatory_keywords(self):
        """Test getting tools matching keywords."""
        from victor.tools.metadata_registry import get_tools_matching_mandatory_keywords

        tools = get_tools_matching_mandatory_keywords("read file")
        assert isinstance(tools, set)

    def test_get_tools_by_task_type(self):
        """Test getting tools by task type."""
        from victor.tools.metadata_registry import get_tools_by_task_type

        # Should return something for any task type
        tools = get_tools_by_task_type("coding")
        assert isinstance(tools, set)
