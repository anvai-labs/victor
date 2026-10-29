"""
Tests for Parameter Enforcement Decorator.

This module tests the parameter enforcement system that validates and
fills missing required parameters in tool calls before execution.

Addresses GAP-9 from Grok/DeepSeek provider testing.
"""

import pytest

from victor.agent.parameter_enforcer import (
    ParameterEnforcer,
    ParameterSpec,
    ParameterType,
    InferenceStrategy,
    ParameterValidationError,
    ParameterInferenceError,
    enforce_parameters,
    create_enforcer_for_tool,
)


class TestParameterSpec:
    """Tests for ParameterSpec dataclass."""

    def test_required_parameter_creation(self):
        """Test creating a required parameter spec."""
        spec = ParameterSpec(
            name="file_path",
            param_type=ParameterType.STRING,
            required=True,
            description="Path to file",
        )
        assert spec.name == "file_path"
        assert spec.param_type == ParameterType.STRING
        assert spec.required is True
        assert spec.default is None

    def test_optional_parameter_with_default(self):
        """Test creating optional parameter with default."""
        spec = ParameterSpec(
            name="limit",
            param_type=ParameterType.INTEGER,
            required=False,
            default=100,
        )
        assert spec.required is False
        assert spec.default == 100

    def test_parameter_with_inference_strategy(self):
        """Test parameter with inference strategy."""
        spec = ParameterSpec(
            name="path",
            param_type=ParameterType.STRING,
            required=True,
            inference_strategy=InferenceStrategy.FROM_CONTEXT,
        )
        assert spec.inference_strategy == InferenceStrategy.FROM_CONTEXT


class TestParameterType:
    """Tests for ParameterType enum."""

    def test_all_types_defined(self):
        """Test all expected types exist."""
        assert ParameterType.STRING is not None
        assert ParameterType.INTEGER is not None
        assert ParameterType.BOOLEAN is not None
        assert ParameterType.FLOAT is not None
        assert ParameterType.ARRAY is not None
        assert ParameterType.OBJECT is not None


class TestInferenceStrategy:
    """Tests for InferenceStrategy enum."""

    def test_all_strategies_defined(self):
        """Test all expected strategies exist."""
        assert InferenceStrategy.NONE is not None
        assert InferenceStrategy.FROM_CONTEXT is not None
        assert InferenceStrategy.FROM_PREVIOUS_ARGS is not None
        assert InferenceStrategy.FROM_DEFAULT is not None
        assert InferenceStrategy.FROM_WORKING_DIR is not None


class TestParameterEnforcer:
    """Tests for the main ParameterEnforcer class."""

    @pytest.fixture
    def symbol_enforcer(self):
        """Create enforcer for symbol tool."""
        specs = [
            ParameterSpec(
                name="file_path",
                param_type=ParameterType.STRING,
                required=True,
                description="Path to the file to analyze",
                inference_strategy=InferenceStrategy.FROM_PREVIOUS_ARGS,
                inference_keys=[
                    "path",
                    "file_path",
                    "file",
                ],  # Keys to search in previous args
            ),
            ParameterSpec(
                name="symbol_name",
                param_type=ParameterType.STRING,
                required=True,
                description="Name of the symbol to find",
            ),
        ]
        return ParameterEnforcer(tool_name="symbol", parameter_specs=specs)

    @pytest.fixture
    def read_enforcer(self):
        """Create enforcer for read tool."""
        specs = [
            ParameterSpec(
                name="path",
                param_type=ParameterType.STRING,
                required=True,
                description="Path to the file to read",
            ),
            ParameterSpec(
                name="offset",
                param_type=ParameterType.INTEGER,
                required=False,
                default=0,
            ),
            ParameterSpec(
                name="limit",
                param_type=ParameterType.INTEGER,
                required=False,
                default=2000,
            ),
        ]
        return ParameterEnforcer(tool_name="read", parameter_specs=specs)

    def test_validate_complete_args(self, symbol_enforcer):
        """Test validation passes for complete arguments."""
        args = {"file_path": "/path/to/file.py", "symbol_name": "MyClass"}
        result = symbol_enforcer.validate(args)
        assert result.is_valid is True
        assert len(result.missing_required) == 0

    def test_validate_missing_required(self, symbol_enforcer):
        """Test validation detects missing required parameter."""
        args = {"symbol_name": "MyClass"}  # Missing file_path
        result = symbol_enforcer.validate(args)
        assert result.is_valid is False
        assert "file_path" in result.missing_required

    def test_validate_multiple_missing(self, symbol_enforcer):
        """Test validation detects multiple missing parameters."""
        args = {}  # Missing both required
        result = symbol_enforcer.validate(args)
        assert result.is_valid is False
        assert "file_path" in result.missing_required
        assert "symbol_name" in result.missing_required

    def test_validate_optional_not_required(self, read_enforcer):
        """Test optional parameters don't cause validation failure."""
        args = {"path": "/path/to/file.py"}  # No offset or limit
        result = read_enforcer.validate(args)
        assert result.is_valid is True

    def test_enforce_fills_defaults(self, read_enforcer):
        """Test enforce fills in default values."""
        args = {"path": "/path/to/file.py"}
        enforced = read_enforcer.enforce(args)
        assert enforced["path"] == "/path/to/file.py"
        assert enforced["offset"] == 0
        assert enforced["limit"] == 2000

    def test_enforce_preserves_existing(self, read_enforcer):
        """Test enforce doesn't overwrite existing values."""
        args = {"path": "/path/to/file.py", "offset": 100, "limit": 50}
        enforced = read_enforcer.enforce(args)
        assert enforced["offset"] == 100
        assert enforced["limit"] == 50

    def test_enforce_with_context(self, symbol_enforcer):
        """Test enforce uses context for inference."""
        args = {"symbol_name": "MyClass"}  # Missing file_path
        context = {
            "previous_tool_args": {
                "read": {"path": "/path/to/previous.py"},
            }
        }
        enforced = symbol_enforcer.enforce(args, context=context)
        assert enforced["file_path"] == "/path/to/previous.py"
        assert enforced["symbol_name"] == "MyClass"

    def test_enforce_raises_on_uninferable(self, symbol_enforcer):
        """Test enforce raises when required param cannot be inferred."""
        args = {"symbol_name": "MyClass"}  # Missing file_path, no context
        with pytest.raises(ParameterInferenceError) as exc_info:
            symbol_enforcer.enforce(args, context={})
        assert "file_path" in str(exc_info.value)

    def test_type_coercion_string_to_int(self, read_enforcer):
        """Test type coercion from string to int."""
        args = {"path": "/file.py", "offset": "100", "limit": "50"}
        enforced = read_enforcer.enforce(args)
        assert enforced["offset"] == 100
        assert isinstance(enforced["offset"], int)
        assert enforced["limit"] == 50
        assert isinstance(enforced["limit"], int)

    def test_type_coercion_string_to_bool(self):
        """Test type coercion from string to bool."""
        specs = [
            ParameterSpec(
                name="recursive",
                param_type=ParameterType.BOOLEAN,
                required=False,
                default=False,
            ),
        ]
        enforcer = ParameterEnforcer(tool_name="glob", parameter_specs=specs)

        # Test various boolean string representations
        assert enforcer.enforce({"recursive": "true"})["recursive"] is True
        assert enforcer.enforce({"recursive": "false"})["recursive"] is False
        assert enforcer.enforce({"recursive": "True"})["recursive"] is True
        assert enforcer.enforce({"recursive": "1"})["recursive"] is True
        assert enforcer.enforce({"recursive": "0"})["recursive"] is False

    def test_type_validation_error(self, read_enforcer):
        """Test type validation error for invalid values."""
        args = {"path": "/file.py", "offset": "not_a_number"}
        with pytest.raises(ParameterValidationError) as exc_info:
            read_enforcer.enforce(args)
        assert "offset" in str(exc_info.value)


class TestInferenceStrategies:
    """Tests for different inference strategies."""

    def test_from_working_dir_strategy(self):
        """Test FROM_WORKING_DIR inference strategy."""
        specs = [
            ParameterSpec(
                name="path",
                param_type=ParameterType.STRING,
                required=True,
                inference_strategy=InferenceStrategy.FROM_WORKING_DIR,
            ),
        ]
        enforcer = ParameterEnforcer(tool_name="ls", parameter_specs=specs)

        context = {"working_directory": "/home/user/project"}
        enforced = enforcer.enforce({}, context=context)
        assert enforced["path"] == "/home/user/project"

    def test_from_previous_args_strategy(self):
        """Test FROM_PREVIOUS_ARGS inference strategy."""
        specs = [
            ParameterSpec(
                name="file_path",
                param_type=ParameterType.STRING,
                required=True,
                inference_strategy=InferenceStrategy.FROM_PREVIOUS_ARGS,
                inference_key="path",  # Look for 'path' in previous args
            ),
        ]
        enforcer = ParameterEnforcer(tool_name="symbol", parameter_specs=specs)

        context = {
            "previous_tool_args": {
                "read": {"path": "/src/main.py"},
            }
        }
        enforced = enforcer.enforce({}, context=context)
        assert enforced["file_path"] == "/src/main.py"

    def test_from_context_strategy(self):
        """Test FROM_CONTEXT inference strategy."""
        specs = [
            ParameterSpec(
                name="language",
                param_type=ParameterType.STRING,
                required=False,
                inference_strategy=InferenceStrategy.FROM_CONTEXT,
                inference_key="detected_language",
            ),
        ]
        enforcer = ParameterEnforcer(tool_name="format", parameter_specs=specs)

        context = {"detected_language": "python"}
        enforced = enforcer.enforce({}, context=context)
        assert enforced["language"] == "python"

    def test_fallback_chain(self):
        """Test inference fallback chain."""
        specs = [
            ParameterSpec(
                name="path",
                param_type=ParameterType.STRING,
                required=True,
                inference_strategy=InferenceStrategy.FROM_PREVIOUS_ARGS,
                default=".",  # Fallback to current directory
            ),
        ]
        enforcer = ParameterEnforcer(tool_name="ls", parameter_specs=specs)

        # No previous args, should fall back to default
        context = {"previous_tool_args": {}}
        enforced = enforcer.enforce({}, context=context)
        assert enforced["path"] == "."


class TestEnforceParametersDecorator:
    """Tests for the @enforce_parameters decorator."""

    def test_decorator_validates_args(self):
        """Test decorator validates arguments before execution."""
        call_count = 0

        @enforce_parameters(
            tool_name="test_tool",
            specs=[
                ParameterSpec(
                    name="required_param",
                    param_type=ParameterType.STRING,
                    required=True,
                ),
            ],
        )
        def test_tool_execute(**kwargs):
            nonlocal call_count
            call_count += 1
            return kwargs

        # Should raise because required_param is missing
        with pytest.raises(ParameterInferenceError):
            test_tool_execute()

        assert call_count == 0  # Function should not have been called

    def test_decorator_passes_valid_args(self):
        """Test decorator passes when args are valid."""

        @enforce_parameters(
            tool_name="test_tool",
            specs=[
                ParameterSpec(
                    name="message",
                    param_type=ParameterType.STRING,
                    required=True,
                ),
            ],
        )
        def test_tool_execute(**kwargs):
            return kwargs

        result = test_tool_execute(message="hello")
        assert result["message"] == "hello"

    def test_decorator_fills_defaults(self):
        """Test decorator fills in defaults."""

        @enforce_parameters(
            tool_name="test_tool",
            specs=[
                ParameterSpec(
                    name="count",
                    param_type=ParameterType.INTEGER,
                    required=False,
                    default=10,
                ),
            ],
        )
        def test_tool_execute(**kwargs):
            return kwargs

        result = test_tool_execute()
        assert result["count"] == 10


class TestCreateEnforcerForTool:
    """Tests for the factory function that creates enforcers from tool schemas."""

    def test_create_from_json_schema(self):
        """Test creating enforcer from JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file",
                },
                "line_number": {
                    "type": "integer",
                    "description": "Line number",
                },
            },
            "required": ["file_path"],
        }

        enforcer = create_enforcer_for_tool("read_file", schema)

        # Should pass with required param
        result = enforcer.validate({"file_path": "/path/to/file.py"})
        assert result.is_valid is True

        # Should fail without required param
        result = enforcer.validate({"line_number": 10})
        assert result.is_valid is False
        assert "file_path" in result.missing_required

    def test_create_from_schema_with_defaults(self):
        """Test creating enforcer from schema with defaults."""
        schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to list",
                    "default": ".",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Recursive listing",
                    "default": False,
                },
            },
            "required": [],
        }

        enforcer = create_enforcer_for_tool("list_directory", schema)
        enforced = enforcer.enforce({})

        assert enforced["path"] == "."
        assert enforced["recursive"] is False


class TestDeepSeekScenarios:
    """Tests simulating DeepSeek's specific failure patterns."""

    @pytest.fixture
    def symbol_enforcer_with_inference(self):
        """Create symbol enforcer with inference configured."""
        specs = [
            ParameterSpec(
                name="file_path",
                param_type=ParameterType.STRING,
                required=True,
                description="Path to the file to analyze",
                inference_strategy=InferenceStrategy.FROM_PREVIOUS_ARGS,
                inference_keys=["path", "file_path", "file"],
            ),
            ParameterSpec(
                name="symbol_name",
                param_type=ParameterType.STRING,
                required=True,
                description="Name of the symbol to find",
            ),
        ]
        return ParameterEnforcer(tool_name="symbol", parameter_specs=specs)

    def test_deepseek_missing_file_path_in_symbol(self, symbol_enforcer_with_inference):
        """Simulate DeepSeek's error: symbol() missing file_path."""
        # DeepSeek's malformed call
        args = {"symbol_name": "SearchResult"}  # Missing file_path

        # Context from previous read call
        context = {
            "previous_tool_args": {
                "read": {"path": "investor_homelab/utils/web_search_client.py"},
            }
        }

        # Enforcer should infer file_path from previous read
        enforced = symbol_enforcer_with_inference.enforce(args, context=context)

        assert enforced["file_path"] == "investor_homelab/utils/web_search_client.py"
        assert enforced["symbol_name"] == "SearchResult"

    def test_deepseek_repeated_tool_calls(self, symbol_enforcer_with_inference):
        """Test handling repeated calls with same inferred path."""
        context = {
            "previous_tool_args": {
                "read": {"path": "/src/main.py"},
            }
        }

        # First call - infers path
        args1 = {"symbol_name": "ClassA"}
        enforced1 = symbol_enforcer_with_inference.enforce(args1, context=context)
        assert enforced1["file_path"] == "/src/main.py"

        # Second call - same context, different symbol
        args2 = {"symbol_name": "ClassB"}
        enforced2 = symbol_enforcer_with_inference.enforce(args2, context=context)
        assert enforced2["file_path"] == "/src/main.py"
        assert enforced2["symbol_name"] == "ClassB"

    def test_multiple_previous_reads_uses_most_recent(self, symbol_enforcer_with_inference):
        """Test that most recent read path is used for inference."""
        context = {
            "previous_tool_args": {
                "read": {"path": "/src/latest.py"},  # Most recent
            },
            "previous_tool_calls": [
                {"tool": "read", "args": {"path": "/src/older.py"}},
                {"tool": "read", "args": {"path": "/src/latest.py"}},
            ],
        }

        args = {"symbol_name": "TestClass"}
        enforced = symbol_enforcer_with_inference.enforce(args, context=context)

        assert enforced["file_path"] == "/src/latest.py"


class TestParameterValidationResult:
    """Tests for ParameterValidationResult object."""

    def test_validation_result_is_valid(self):
        """Test ParameterValidationResult.is_valid property."""
        specs = [
            ParameterSpec(
                name="required",
                param_type=ParameterType.STRING,
                required=True,
            ),
        ]
        enforcer = ParameterEnforcer(tool_name="test", parameter_specs=specs)

        valid_result = enforcer.validate({"required": "value"})
        assert valid_result.is_valid is True

        invalid_result = enforcer.validate({})
        assert invalid_result.is_valid is False

    def test_validation_result_details(self):
        """Test ParameterValidationResult contains error details."""
        specs = [
            ParameterSpec(
                name="file_path",
                param_type=ParameterType.STRING,
                required=True,
            ),
            ParameterSpec(
                name="count",
                param_type=ParameterType.INTEGER,
                required=True,
            ),
        ]
        enforcer = ParameterEnforcer(tool_name="test", parameter_specs=specs)

        result = enforcer.validate({})

        assert "file_path" in result.missing_required
        assert "count" in result.missing_required
        assert len(result.missing_required) == 2


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_specs(self):
        """Test enforcer with no parameter specs."""
        enforcer = ParameterEnforcer(tool_name="simple", parameter_specs=[])

        result = enforcer.validate({"any": "args"})
        assert result.is_valid is True

        enforced = enforcer.enforce({"any": "args"})
        assert enforced == {"any": "args"}

    def test_none_value_handling(self):
        """Test handling of None values."""
        specs = [
            ParameterSpec(
                name="optional",
                param_type=ParameterType.STRING,
                required=False,
                default="default_value",
            ),
        ]
        enforcer = ParameterEnforcer(tool_name="test", parameter_specs=specs)

        # None should be replaced with default
        enforced = enforcer.enforce({"optional": None})
        assert enforced["optional"] == "default_value"

    def test_extra_params_preserved(self):
        """Test that extra parameters not in spec are preserved."""
        specs = [
            ParameterSpec(
                name="known",
                param_type=ParameterType.STRING,
                required=True,
            ),
        ]
        enforcer = ParameterEnforcer(tool_name="test", parameter_specs=specs)

        enforced = enforcer.enforce({"known": "value", "extra": "preserved"})
        assert enforced["known"] == "value"
        assert enforced["extra"] == "preserved"

    def test_array_type_handling(self):
        """Test array type parameter handling."""
        specs = [
            ParameterSpec(
                name="items",
                param_type=ParameterType.ARRAY,
                required=False,
                default=[],
            ),
        ]
        enforcer = ParameterEnforcer(tool_name="test", parameter_specs=specs)

        # Empty args should get default empty list
        enforced = enforcer.enforce({})
        assert enforced["items"] == []
        assert isinstance(enforced["items"], list)

        # Provided list should be preserved
        enforced = enforcer.enforce({"items": [1, 2, 3]})
        assert enforced["items"] == [1, 2, 3]

    def test_object_type_handling(self):
        """Test object type parameter handling."""
        specs = [
            ParameterSpec(
                name="config",
                param_type=ParameterType.OBJECT,
                required=False,
                default={},
            ),
        ]
        enforcer = ParameterEnforcer(tool_name="test", parameter_specs=specs)

        # Empty args should get default empty dict
        enforced = enforcer.enforce({})
        assert enforced["config"] == {}
        assert isinstance(enforced["config"], dict)
