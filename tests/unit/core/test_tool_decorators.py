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

"""Tests for decorators module."""

from enum import Enum
import pytest
from typing import Dict, Any, Optional
from unittest.mock import patch
import logging

from victor.tools.decorators import (
    tool,
    _create_tool_class,
    resolve_tool_name,
    set_legacy_name_warnings,
    _WARN_ON_LEGACY_NAMES,
)
from victor.tools.base import BaseTool, CostTier, ToolResult


class _SchemaMode(str, Enum):
    OVERVIEW = "overview"
    STATS = "stats"


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_tool_decorator_basic_function(self):
        """Test @tool decorator on a basic function."""

        @tool
        def simple_function(param1: str, param2: int = 5):
            """A simple test function.

            Args:
                param1: First parameter.
                param2: Second parameter.
            """
            return f"{param1}-{param2}"

        # Check that decorator creates Tool attribute (which is an instance)
        assert hasattr(simple_function, "Tool")
        assert isinstance(simple_function.Tool, BaseTool)

        # Function should still be callable
        result = simple_function("test", 10)
        assert result == "test-10"

    def test_tool_decorator_with_context(self):
        """Test @tool decorator with context parameter."""

        @tool
        async def function_with_context(context: Dict[str, Any], value: str):
            """Function that uses context.

            Args:
                context: The tool context.
                value: A value parameter.
            """
            return {"context_used": context is not None, "value": value}

        assert hasattr(function_with_context, "Tool")

    def test_tool_decorator_preserves_metadata(self):
        """Test that decorator preserves function metadata."""

        @tool
        def metadata_function(param: str):
            """Test function for metadata.

            Args:
                param: A parameter.
            """
            return param

        # Wrapper should preserve function name
        assert metadata_function.__name__ == "metadata_function"

    @pytest.mark.asyncio
    async def test_tool_execution_success(self):
        """Test successful tool execution."""

        @tool
        async def async_tool(value: int):
            """Async tool function.

            Args:
                value: An integer value.
            """
            return value * 2

        # Tool is already an instance
        tool_obj = async_tool.Tool

        # Execute tool
        result = await tool_obj.execute({}, value=5)

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == 10
        assert result.error is None

    @pytest.mark.asyncio
    async def test_tool_execution_with_exception(self):
        """Test tool execution with exception."""

        @tool
        async def failing_tool(value: int):
            """Tool that raises exception.

            Args:
                value: An integer value.
            """
            raise ValueError("Test error")

        tool_obj = failing_tool.Tool
        result = await tool_obj.execute({}, value=5)

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.output is None
        assert "Test error" in result.error
        assert result.metadata["exception"] == "ValueError"

    @pytest.mark.asyncio
    async def test_tool_execution_sync_function(self):
        """Test tool execution with synchronous function."""

        @tool
        def sync_tool(value: str):
            """Synchronous tool function.

            Args:
                value: A string value.
            """
            return value.upper()

        tool_obj = sync_tool.Tool
        result = await tool_obj.execute({}, value="hello")

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "HELLO"

    @pytest.mark.asyncio
    async def test_tool_execution_with_context_injection(self):
        """Test that _exec_ctx is injected when function has _exec_ctx parameter.

        Note: We use _exec_ctx (not context) to avoid collision with tool parameters
        named 'context' that LLMs commonly generate.
        """

        @tool
        async def context_aware_tool(_exec_ctx: Dict[str, Any], value: str):
            """Tool that uses execution context.

            Args:
                _exec_ctx: The framework execution context (reserved name).
                value: A value.
            """
            return {"has_context": _exec_ctx is not None, "value": value}

        tool_obj = context_aware_tool.Tool
        test_context = {"test_key": "test_value"}
        result = await tool_obj.execute(test_context, value="data")

        assert result.success is True
        assert result.output["has_context"] is True
        assert result.output["value"] == "data"

    def test_tool_properties(self):
        """Test that tool properties are accessible."""

        @tool
        def property_test_tool(param1: str, param2: int = 10):
            """A tool for testing properties.

            This is the long description.

            Args:
                param1: First parameter.
                param2: Second parameter with default.
            """
            return None

        tool_obj = property_test_tool.Tool

        # Test name property
        assert tool_obj.name == "property_test_tool"

        # Test description property
        assert "A tool for testing properties" in tool_obj.description
        assert "long description" in tool_obj.description

        # Test parameters property
        params = tool_obj.parameters
        assert params["type"] == "object"
        assert "param1" in params["properties"]
        assert "param2" in params["properties"]
        assert "param1" in params["required"]
        assert "param2" not in params["required"]  # Has default


class TestCreateToolClass:
    """Tests for _create_tool_class function."""

    def test_create_tool_class_with_typed_parameters(self):
        """Test creating tool class with type annotations."""

        def typed_function(str_param: str, int_param: int, float_param: float, bool_param: bool):
            """Function with typed parameters.

            Args:
                str_param: A string.
                int_param: An integer.
                float_param: A float.
                bool_param: A boolean.
            """
            return None

        tool_obj = _create_tool_class(typed_function)

        params = tool_obj.parameters
        assert params["properties"]["str_param"]["type"] == "string"
        # int maps to "integer" per JSON Schema spec (more precise than "number")
        assert params["properties"]["int_param"]["type"] == "integer"
        assert params["properties"]["float_param"]["type"] == "number"
        assert params["properties"]["bool_param"]["type"] == "boolean"

    def test_create_tool_class_with_runtime_enum_parameter(self):
        """Runtime Enum annotations should become JSON schema enums."""

        def enum_function(mode: _SchemaMode):
            """Function with enum parameter.

            Args:
                mode: Execution mode.
            """
            return mode

        tool_obj = _create_tool_class(enum_function)

        params = tool_obj.parameters
        assert params["properties"]["mode"]["type"] == "string"
        assert params["properties"]["mode"]["enum"] == ["overview", "stats"]

    def test_create_tool_class_with_string_enum_annotation(self):
        """Stringified enum annotations should resolve through function globals."""

        def enum_function(mode: "_SchemaMode"):
            """Function with string enum annotation.

            Args:
                mode: Execution mode.
            """
            return mode

        tool_obj = _create_tool_class(enum_function)

        params = tool_obj.parameters
        assert params["properties"]["mode"]["type"] == "string"
        assert params["properties"]["mode"]["enum"] == ["overview", "stats"]

    def test_create_tool_class_sanitizes_enum_default_values(self):
        """Enum defaults should be exposed as raw values, not Python reprs."""

        def enum_function(mode: _SchemaMode = _SchemaMode.OVERVIEW):
            """Function with enum default.

            Args:
                mode: Execution mode.
            """
            return mode

        tool_obj = _create_tool_class(enum_function)

        params = tool_obj.parameters
        assert params["properties"]["mode"]["default"] == "overview"

    def test_create_tool_class_with_no_docstring(self):
        """Test creating tool class with no docstring."""

        def no_docstring_function(param):
            return param

        tool_obj = _create_tool_class(no_docstring_function)

        assert "No description provided" in tool_obj.description

    def test_create_tool_class_with_var_args(self):
        """Test creating tool class with *args and **kwargs parameters."""

        def var_args_function(*args, **kwargs):
            """Function with variable arguments.

            Args:
                args: Variable positional arguments.
                kwargs: Variable keyword arguments.
            """
            return None

        tool_obj = _create_tool_class(var_args_function)

        # *args and **kwargs should be skipped
        params = tool_obj.parameters
        # Should not include args/kwargs in properties
        assert len(params["properties"]) == 0

    def test_create_tool_class_with_keyword_only_params(self):
        """Test creating tool class with keyword-only parameters."""

        def keyword_only_function(normal_param: str, *, keyword_only: int):
            """Function with keyword-only parameter.

            Args:
                normal_param: A normal parameter.
                keyword_only: A keyword-only parameter.
            """
            return None

        tool_obj = _create_tool_class(keyword_only_function)

        params = tool_obj.parameters
        assert "normal_param" in params["properties"]
        assert "keyword_only" in params["properties"]

    def test_create_tool_class_inherits_base_tool(self):
        """Test that created tool instance is a BaseTool."""

        def test_function(param: str):
            """Test function.

            Args:
                param: A parameter.
            """
            return param

        tool_obj = _create_tool_class(test_function)

        assert isinstance(tool_obj, BaseTool)

    def test_create_tool_class_to_json_schema(self):
        """Test that tool can be converted to JSON schema."""

        def schema_test_function(param: str):
            """Schema test function.

            Args:
                param: A parameter.
            """
            return param

        tool_obj = _create_tool_class(schema_test_function)

        schema = tool_obj.to_json_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "schema_test_function"
        assert "Schema test function" in schema["function"]["description"]
        assert "param" in schema["function"]["parameters"]["properties"]

    def test_create_tool_class_cost_tier_property(self):
        """Test that cost_tier property is correctly set (covers line 153)."""

        def cost_tier_function(param: str):
            """Function with cost tier.

            Args:
                param: A parameter.
            """
            return param

        # Test with non-default cost tier
        tool_obj = _create_tool_class(cost_tier_function, cost_tier=CostTier.HIGH)
        assert tool_obj.cost_tier == CostTier.HIGH

        # Test default cost tier
        tool_obj_default = _create_tool_class(cost_tier_function)
        assert tool_obj_default.cost_tier == CostTier.FREE


class TestToolDecoratorCostTier:
    """Tests for @tool decorator with cost_tier parameter."""

    def test_tool_decorator_with_cost_tier(self):
        """Test @tool decorator with explicit cost_tier."""

        @tool(cost_tier=CostTier.MEDIUM)
        def medium_cost_tool(param: str):
            """A medium cost tool.

            Args:
                param: A parameter.
            """
            return param

        tool_obj = medium_cost_tool.Tool
        assert tool_obj.cost_tier == CostTier.MEDIUM

    def test_tool_decorator_with_high_cost_tier(self):
        """Test @tool decorator with HIGH cost_tier."""

        @tool(cost_tier=CostTier.HIGH)
        def high_cost_tool(param: str):
            """A high cost tool.

            Args:
                param: A parameter.
            """
            return param

        tool_obj = high_cost_tool.Tool
        assert tool_obj.cost_tier == CostTier.HIGH

    def test_tool_decorator_default_cost_tier(self):
        """Test @tool decorator has FREE as default cost_tier."""

        @tool
        def default_cost_tool(param: str):
            """A default cost tool.

            Args:
                param: A parameter.
            """
            return param

        tool_obj = default_cost_tool.Tool
        assert tool_obj.cost_tier == CostTier.FREE


class TestToolNameResolution:
    """Tests for tool name resolution and legacy warning functions."""

    def test_resolve_tool_name_legacy_to_canonical(self):
        """Test resolving legacy tool name to canonical."""
        # Legacy names should resolve to canonical short names
        assert resolve_tool_name("execute_bash") == "shell"
        assert resolve_tool_name("read_file") == "read"
        assert resolve_tool_name("write_file") == "write"
        assert resolve_tool_name("code_search") == "code_search"
        assert resolve_tool_name("list_directory") == "ls"

    def test_resolve_tool_name_canonical_unchanged(self):
        """Test that canonical names remain unchanged."""
        # Canonical names should resolve to themselves
        assert resolve_tool_name("shell") == "shell"
        assert resolve_tool_name("read") == "read"
        assert resolve_tool_name("write") == "write"
        assert resolve_tool_name("grep") == "grep"
        assert resolve_tool_name("ls") == "ls"

    def test_resolve_tool_name_unknown_unchanged(self):
        """Test that unknown names remain unchanged."""
        # Names not in registry should pass through unchanged
        assert resolve_tool_name("unknown_tool") == "unknown_tool"
        assert resolve_tool_name("my_custom_tool") == "my_custom_tool"

    def test_set_legacy_name_warnings_toggle(self):
        """Test enabling and disabling legacy name warnings."""
        import victor.tools.decorators as decorators_module

        # Initially should be False
        original_value = decorators_module._WARN_ON_LEGACY_NAMES

        try:
            # Enable warnings
            set_legacy_name_warnings(True)
            assert decorators_module._WARN_ON_LEGACY_NAMES is True

            # Disable warnings
            set_legacy_name_warnings(False)
            assert decorators_module._WARN_ON_LEGACY_NAMES is False
        finally:
            # Restore original value
            decorators_module._WARN_ON_LEGACY_NAMES = original_value

    def test_resolve_tool_name_with_warn_on_legacy(self, caplog):
        """Test that warnings are logged when legacy names are used."""
        import victor.tools.decorators as decorators_module

        original_value = decorators_module._WARN_ON_LEGACY_NAMES

        try:
            set_legacy_name_warnings(True)

            with caplog.at_level(logging.WARNING, logger="victor.tools.decorators"):
                result = resolve_tool_name("execute_bash")

            assert result == "shell"
            assert any(
                "Legacy tool name 'execute_bash' used" in record.message
                for record in caplog.records
            )
            assert any("'shell'" in record.message for record in caplog.records)
        finally:
            decorators_module._WARN_ON_LEGACY_NAMES = original_value

    def test_resolve_tool_name_explicit_warn_param(self, caplog):
        """Test that warn_on_legacy param triggers warning for single call."""
        import victor.tools.decorators as decorators_module

        original_value = decorators_module._WARN_ON_LEGACY_NAMES

        try:
            # Make sure global warning is OFF
            set_legacy_name_warnings(False)

            with caplog.at_level(logging.WARNING, logger="victor.tools.decorators"):
                # Use explicit warn_on_legacy=True for this call only
                result = resolve_tool_name("read_file", warn_on_legacy=True)

            assert result == "read"
            assert any(
                "Legacy tool name 'read_file' used" in record.message for record in caplog.records
            )
        finally:
            decorators_module._WARN_ON_LEGACY_NAMES = original_value

    def test_resolve_tool_name_no_warn_for_canonical(self, caplog):
        """Test that no warning is logged for canonical names even when warnings enabled."""
        import victor.tools.decorators as decorators_module

        original_value = decorators_module._WARN_ON_LEGACY_NAMES

        try:
            set_legacy_name_warnings(True)

            with caplog.at_level(logging.WARNING, logger="victor.tools.decorators"):
                result = resolve_tool_name("shell")

            assert result == "shell"
            # Should not have any legacy name warnings
            assert not any("Legacy tool name" in record.message for record in caplog.records)
        finally:
            decorators_module._WARN_ON_LEGACY_NAMES = original_value


class TestToolDecoratorWithAliases:
    """Tests for @tool decorator with name and aliases parameters."""

    def test_tool_decorator_with_explicit_name(self):
        """Test @tool decorator with explicit name parameter."""

        @tool(name="short_name")
        def long_function_name(param: str):
            """A tool with explicit short name.

            Args:
                param: A parameter.
            """
            return param

        tool_obj = long_function_name.Tool
        assert tool_obj.name == "short_name"
        assert tool_obj.original_name == "long_function_name"

    def test_tool_decorator_with_aliases(self):
        """Test @tool decorator with aliases parameter."""

        @tool(name="canonical", aliases=["legacy_name", "old_name"])
        def canonical_function(param: str):
            """A tool with aliases.

            Args:
                param: A parameter.
            """
            return param

        tool_obj = canonical_function.Tool
        assert tool_obj.name == "canonical"
        assert "legacy_name" in tool_obj.aliases
        assert "old_name" in tool_obj.aliases
        # Original function name should also be in aliases if different from canonical
        assert "canonical_function" in tool_obj.aliases

    def test_tool_matches_name_canonical(self):
        """Test that matches_name works with canonical name."""

        @tool(name="short", aliases=["long_name"])
        def my_tool(param: str):
            """Test tool.

            Args:
                param: A parameter.
            """
            return param

        tool_obj = my_tool.Tool
        assert tool_obj.matches_name("short") is True

    def test_tool_matches_name_alias(self):
        """Test that matches_name works with alias."""

        @tool(name="short", aliases=["long_name"])
        def my_tool(param: str):
            """Test tool.

            Args:
                param: A parameter.
            """
            return param

        tool_obj = my_tool.Tool
        assert tool_obj.matches_name("long_name") is True
        assert tool_obj.matches_name("my_tool") is True  # Original function name

    def test_tool_all_names(self):
        """Test that all_names returns canonical plus aliases."""

        @tool(name="canonical", aliases=["alias1", "alias2"])
        def function_name(param: str):
            """Test tool.

            Args:
                param: A parameter.
            """
            return param

        tool_obj = function_name.Tool
        all_names = tool_obj.all_names()

        assert "canonical" in all_names
        assert "alias1" in all_names
        assert "alias2" in all_names
        assert "function_name" in all_names  # Original name auto-added

    def test_tool_auto_resolve_from_registry(self):
        """Test that function names in TOOL_ALIASES auto-resolve."""
        # This tests the auto-resolution feature where function names
        # that match entries in TOOL_ALIASES get resolved automatically

        # We can't easily test this without mocking the registry,
        # but we can verify the mechanism works with explicit name

        @tool(name="shell")
        def execute_bash(command: str):
            """Execute shell command.

            Args:
                command: The command to run.
            """
            return command

        tool_obj = execute_bash.Tool
        assert tool_obj.name == "shell"
        assert "execute_bash" in tool_obj.aliases


class TestDecoratorDrivenSemanticSelection:
    """Tests for new decorator-driven semantic selection attributes."""

    def test_mandatory_keywords_decorator(self):
        """Test mandatory_keywords attribute via @tool decorator."""

        @tool(
            mandatory_keywords=["show diff", "compare files"],
            keywords=["diff", "compare"],
        )
        def diff_tool(file1: str, file2: str):
            """Compare two files.

            Args:
                file1: First file path.
                file2: Second file path.
            """
            return f"{file1} vs {file2}"

        tool_obj = diff_tool.Tool
        assert hasattr(tool_obj, "mandatory_keywords")
        assert "show diff" in tool_obj.mandatory_keywords
        assert "compare files" in tool_obj.mandatory_keywords

    def test_task_types_decorator(self):
        """Test task_types attribute via @tool decorator."""

        @tool(
            task_types=["analysis", "search"],
            keywords=["analyze", "find"],
        )
        def analysis_tool(query: str):
            """Analyze code patterns.

            Args:
                query: Search query.
            """
            return query

        tool_obj = analysis_tool.Tool
        assert hasattr(tool_obj, "task_types")
        assert "analysis" in tool_obj.task_types
        assert "search" in tool_obj.task_types

    def test_progress_params_decorator(self):
        """Test progress_params attribute via @tool decorator."""

        @tool(progress_params=["path", "offset", "limit"])
        def read_tool(path: str, offset: int = 0, limit: int = 100):
            """Read file with pagination.

            Args:
                path: File path to read.
                offset: Start offset.
                limit: Number of lines.
            """
            return f"Reading {path} from {offset}"

        tool_obj = read_tool.Tool
        assert hasattr(tool_obj, "progress_params")
        assert "path" in tool_obj.progress_params
        assert "offset" in tool_obj.progress_params
        assert "limit" in tool_obj.progress_params

    def test_execution_category_decorator(self):
        """Test execution_category attribute via @tool decorator."""
        from victor.tools.base import ExecutionCategory

        @tool(execution_category="read_only")
        def readonly_tool(path: str):
            """Read-only operation.

            Args:
                path: File path.
            """
            return path

        tool_obj = readonly_tool.Tool
        assert hasattr(tool_obj, "execution_category")
        assert tool_obj.execution_category == ExecutionCategory.READ_ONLY

    def test_execution_category_write(self):
        """Test execution_category with write value."""
        from victor.tools.base import ExecutionCategory

        @tool(execution_category="write")
        def write_tool(path: str, content: str):
            """Write operation.

            Args:
                path: File path.
                content: Content to write.
            """
            return f"Writing to {path}"

        tool_obj = write_tool.Tool
        assert tool_obj.execution_category == ExecutionCategory.WRITE

    def test_execution_category_network(self):
        """Test execution_category with network value."""
        from victor.tools.base import ExecutionCategory

        @tool(execution_category="network")
        def fetch_tool(url: str):
            """Fetch from network.

            Args:
                url: URL to fetch.
            """
            return url

        tool_obj = fetch_tool.Tool
        assert tool_obj.execution_category == ExecutionCategory.NETWORK

    def test_all_semantic_attributes_together(self):
        """Test all new decorator attributes work together."""
        from victor.tools.base import ExecutionCategory

        @tool(
            mandatory_keywords=["run tests", "execute tests"],
            task_types=["action", "verification"],
            progress_params=["test_file", "pattern"],
            execution_category="execute",
            keywords=["test", "pytest", "unittest"],
            stages=["verification", "execution"],
        )
        def test_runner(test_file: str, pattern: str = "test_*"):
            """Run tests.

            Args:
                test_file: Test file or directory.
                pattern: Test pattern.
            """
            return f"Running {pattern} in {test_file}"

        tool_obj = test_runner.Tool
        # Mandatory keywords
        assert "run tests" in tool_obj.mandatory_keywords
        assert "execute tests" in tool_obj.mandatory_keywords
        # Task types
        assert "action" in tool_obj.task_types
        assert "verification" in tool_obj.task_types
        # Progress params
        assert "test_file" in tool_obj.progress_params
        assert "pattern" in tool_obj.progress_params
        # Execution category
        assert tool_obj.execution_category == ExecutionCategory.EXECUTE
        # Regular keywords still work
        assert "test" in tool_obj.keywords
        assert "pytest" in tool_obj.keywords
        # Stages still work
        assert "verification" in tool_obj.stages
        assert "execution" in tool_obj.stages

    def test_execution_category_invalid_defaults_to_readonly(self, caplog):
        """Test invalid execution_category value defaults to READ_ONLY with warning."""
        from victor.tools.base import ExecutionCategory

        with caplog.at_level(logging.WARNING):

            @tool(execution_category="invalid_category")
            def bad_category_tool(path: str):
                """Tool with invalid category.

                Args:
                    path: Path.
                """
                return path

            tool_obj = bad_category_tool.Tool
            # Should default to READ_ONLY
            assert tool_obj.execution_category == ExecutionCategory.READ_ONLY
            # Should log a warning
            assert "Invalid execution_category" in caplog.text

    def test_empty_semantic_attributes(self):
        """Test tools without semantic attributes have empty defaults."""
        from victor.tools.base import ExecutionCategory

        @tool
        def simple_tool(value: str):
            """Simple tool.

            Args:
                value: A value.
            """
            return value

        tool_obj = simple_tool.Tool
        # Empty lists for optional attributes
        assert tool_obj.mandatory_keywords == []
        assert tool_obj.task_types == []
        assert tool_obj.progress_params == []
        # Default execution category
        assert tool_obj.execution_category == ExecutionCategory.READ_ONLY


class TestExecCtxSchemaExclusion:
    """Tests for _exec_ctx being excluded from tool JSON schema."""

    def test_exec_ctx_excluded_from_schema(self):
        """_exec_ctx should not appear in tool JSON schema sent to LLM."""

        @tool(name="test_tool_ctx_excl")
        async def my_tool(query: str, _exec_ctx: Optional[Dict] = None):
            """Test tool.

            Args:
                query: A query.
                _exec_ctx: Execution context.
            """
            return {"result": query}

        tool_obj = my_tool.Tool
        assert "_exec_ctx" not in tool_obj.parameters.get("properties", {})
        assert "_exec_ctx" not in tool_obj.parameters.get("required", [])
        # query should still be present
        assert "query" in tool_obj.parameters["properties"]

    def test_exec_ctx_excluded_even_when_required(self):
        """_exec_ctx should be excluded even if it has no default."""

        @tool(name="test_tool_ctx_req")
        async def my_tool(query: str, _exec_ctx: Dict):
            """Test tool.

            Args:
                query: A query.
                _exec_ctx: Execution context.
            """
            return {"result": query}

        tool_obj = my_tool.Tool
        assert "_exec_ctx" not in tool_obj.parameters.get("properties", {})
        assert "_exec_ctx" not in tool_obj.parameters.get("required", [])

    @pytest.mark.asyncio
    async def test_exec_ctx_still_passed_to_function(self):
        """_exec_ctx should still be forwarded by execute() even though not in schema."""

        @tool(name="test_tool_ctx_pass")
        async def my_tool(query: str, _exec_ctx: Optional[Dict] = None):
            """Test tool.

            Args:
                query: A query.
                _exec_ctx: Execution context.
            """
            return {"result": query, "had_ctx": _exec_ctx is not None}

        tool_obj = my_tool.Tool
        result = await tool_obj.execute({"settings": {}}, query="test")
        assert result.success is True
        assert result.output["had_ctx"] is True
