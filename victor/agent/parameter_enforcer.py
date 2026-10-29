"""
Parameter Enforcement Decorator for Tool Execution.

This module implements a Decorator pattern to validate and enforce tool parameters,
providing automatic inference for missing required parameters when possible.

SOLID Principles Applied:
- Single Responsibility: Each class has one purpose (validation, inference, enforcement)
- Open/Closed: New inference strategies can be added without modifying existing code
- Liskov Substitution: All inference strategies are interchangeable
- Interface Segregation: ParameterSpec defines minimal interface
- Dependency Inversion: Enforcer depends on abstractions (specs), not concrete tools

Addresses GAP-9 from Grok/DeepSeek provider testing.
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Supported parameter types for validation and coercion."""

    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    FLOAT = "float"
    ARRAY = "array"
    OBJECT = "object"


class InferenceStrategy(Enum):
    """Strategies for inferring missing parameter values."""

    NONE = "none"  # No inference, must be provided
    FROM_CONTEXT = "from_context"  # Infer from context dict
    FROM_PREVIOUS_ARGS = "from_previous_args"  # Infer from previous tool args
    FROM_DEFAULT = "from_default"  # Use default value
    FROM_WORKING_DIR = "from_working_dir"  # Use current working directory


class ParameterValidationError(Exception):
    """Raised when parameter validation fails."""

    def __init__(self, param_name: str, message: str):
        self.param_name = param_name
        super().__init__(f"Validation error for '{param_name}': {message}")


class ParameterInferenceError(Exception):
    """Raised when a required parameter cannot be inferred."""

    def __init__(self, param_name: str, tool_name: str):
        self.param_name = param_name
        self.tool_name = tool_name
        super().__init__(f"Cannot infer required parameter '{param_name}' for tool '{tool_name}'")


@dataclass
class ParameterSpec:
    """Specification for a tool parameter."""

    name: str
    param_type: ParameterType
    required: bool = True
    default: Any = None
    description: str = ""
    inference_strategy: InferenceStrategy = InferenceStrategy.NONE
    inference_key: Optional[str] = None  # Key to look up in context/previous args
    inference_keys: Optional[List[str]] = None  # Multiple keys to try


@dataclass
class ParameterValidationResult:
    """Result of parameter enforcement validation.

    Renamed from ValidationResult to be semantically distinct:
    - ToolValidationResult (victor.tools.base): Tool parameter validation
    - ConfigValidationResult (victor.core.validation): Configuration validation
    - ContentValidationResult (victor.framework.middleware): Content validation
    - ParameterValidationResult (here): Parameter enforcement with missing_required
    - CodeValidationResult (victor.evaluation.correction.types): Code validation
    """

    is_valid: bool
    missing_required: List[str] = field(default_factory=list)
    type_errors: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class ParameterEnforcer:
    """
    Enforces parameter requirements for tool execution.

    Validates that required parameters are present, provides defaults for
    optional parameters, and attempts to infer missing required parameters
    from context when possible.
    """

    def __init__(
        self,
        tool_name: str,
        parameter_specs: List[ParameterSpec],
    ) -> None:
        self.tool_name = tool_name
        self.specs = {spec.name: spec for spec in parameter_specs}
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate(self, args: Dict[str, Any]) -> ParameterValidationResult:
        """
        Validate arguments against parameter specifications.

        Args:
            args: The arguments to validate

        Returns:
            ParameterValidationResult with validation status and any errors
        """
        missing_required = []
        type_errors = {}

        for name, spec in self.specs.items():
            if spec.required:
                value = args.get(name)
                if value is None:
                    missing_required.append(name)

        is_valid = len(missing_required) == 0 and len(type_errors) == 0

        return ParameterValidationResult(
            is_valid=is_valid,
            missing_required=missing_required,
            type_errors=type_errors,
        )

    def enforce(
        self,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Enforce parameter requirements, filling in missing values where possible.

        Args:
            args: The arguments to enforce
            context: Optional context for inference

        Returns:
            Enforced arguments with missing values filled in

        Raises:
            ParameterInferenceError: If a required parameter cannot be inferred
            ParameterValidationError: If a parameter has an invalid value
        """
        context = context or {}
        enforced = dict(args)

        for name, spec in self.specs.items():
            current_value = enforced.get(name)

            # Handle None values
            if current_value is None:
                # Try to infer the value regardless of required status
                if spec.inference_strategy != InferenceStrategy.NONE:
                    inferred = self._infer_value(spec, context)
                    if inferred is not None:
                        enforced[name] = inferred
                        self._logger.debug(f"Inferred {name}={inferred} for {self.tool_name}")
                        continue  # Move to next parameter

                # If we couldn't infer, try defaults or raise error
                if spec.required:
                    if spec.default is not None:
                        enforced[name] = spec.default
                    else:
                        raise ParameterInferenceError(name, self.tool_name)
                elif spec.default is not None:
                    enforced[name] = spec.default
            else:
                # Coerce type if needed
                try:
                    enforced[name] = self._coerce_type(current_value, spec)
                except (ValueError, TypeError) as e:
                    raise ParameterValidationError(name, str(e))

        return enforced

    def _infer_value(
        self,
        spec: ParameterSpec,
        context: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Attempt to infer a parameter value based on its inference strategy.

        Args:
            spec: The parameter specification
            context: Context for inference

        Returns:
            Inferred value or None if inference failed
        """
        if spec.inference_strategy == InferenceStrategy.NONE:
            return None

        if spec.inference_strategy == InferenceStrategy.FROM_WORKING_DIR:
            return context.get("working_directory")

        if spec.inference_strategy == InferenceStrategy.FROM_CONTEXT:
            key = spec.inference_key or spec.name
            return context.get(key)

        if spec.inference_strategy == InferenceStrategy.FROM_PREVIOUS_ARGS:
            return self._infer_from_previous_args(spec, context)

        if spec.inference_strategy == InferenceStrategy.FROM_DEFAULT:
            return spec.default

        return None

    def _infer_from_previous_args(
        self,
        spec: ParameterSpec,
        context: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Infer parameter value from previous tool arguments.

        Args:
            spec: The parameter specification
            context: Context containing previous_tool_args

        Returns:
            Inferred value or None
        """
        previous_args = context.get("previous_tool_args", {})

        # Get keys to search for
        keys_to_try = []
        if spec.inference_keys:
            keys_to_try = spec.inference_keys
        elif spec.inference_key:
            keys_to_try = [spec.inference_key]
        else:
            keys_to_try = [spec.name]

        # Search through all previous tool args
        for tool_args in previous_args.values():
            if isinstance(tool_args, dict):
                for key in keys_to_try:
                    if key in tool_args:
                        return tool_args[key]

        return None

    def _coerce_type(self, value: Any, spec: ParameterSpec) -> Any:
        """
        Coerce a value to the expected type.

        Args:
            value: The value to coerce
            spec: The parameter specification

        Returns:
            Coerced value

        Raises:
            ValueError: If coercion fails
        """
        if spec.param_type == ParameterType.STRING:
            return str(value) if value is not None else value

        if spec.param_type == ParameterType.INTEGER:
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                return int(value)
            if isinstance(value, float):
                return int(value)
            raise ValueError(f"Cannot convert {type(value).__name__} to integer")

        if spec.param_type == ParameterType.FLOAT:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value)
            raise ValueError(f"Cannot convert {type(value).__name__} to float")

        if spec.param_type == ParameterType.BOOLEAN:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lower = value.lower()
                if lower in ("true", "1", "yes", "on"):
                    return True
                if lower in ("false", "0", "no", "off"):
                    return False
                raise ValueError(f"Cannot convert '{value}' to boolean")
            if isinstance(value, (int, float)):
                return bool(value)
            raise ValueError(f"Cannot convert {type(value).__name__} to boolean")

        if spec.param_type == ParameterType.ARRAY:
            if isinstance(value, list):
                return value
            if isinstance(value, (tuple, set)):
                return list(value)
            raise ValueError(f"Cannot convert {type(value).__name__} to array")

        if spec.param_type == ParameterType.OBJECT:
            if isinstance(value, dict):
                return value
            raise ValueError(f"Cannot convert {type(value).__name__} to object")

        return value


def enforce_parameters(
    tool_name: str,
    specs: List[ParameterSpec],
) -> Callable:
    """
    Decorator to enforce parameters on a tool execution function.

    Usage:
        @enforce_parameters(
            tool_name="read",
            specs=[
                ParameterSpec(name="path", param_type=ParameterType.STRING, required=True),
                ParameterSpec(name="offset", param_type=ParameterType.INTEGER, default=0),
            ]
        )
        def execute(**kwargs):
            ...

    Args:
        tool_name: Name of the tool
        specs: List of parameter specifications

    Returns:
        Decorator function
    """
    enforcer = ParameterEnforcer(tool_name=tool_name, parameter_specs=specs)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, context: Optional[Dict[str, Any]] = None, **kwargs):
            # Enforce parameters
            enforced_kwargs = enforcer.enforce(kwargs, context=context)
            return func(*args, **enforced_kwargs)

        # Attach enforcer for introspection
        wrapper._parameter_enforcer = enforcer
        return wrapper

    return decorator


def create_enforcer_for_tool(
    tool_name: str,
    schema: Dict[str, Any],
) -> ParameterEnforcer:
    """
    Factory function to create a ParameterEnforcer from a JSON schema.

    Args:
        tool_name: Name of the tool
        schema: JSON schema for the tool parameters

    Returns:
        Configured ParameterEnforcer
    """
    specs = []
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Type mapping from JSON schema to ParameterType
    type_mapping = {
        "string": ParameterType.STRING,
        "integer": ParameterType.INTEGER,
        "number": ParameterType.FLOAT,
        "boolean": ParameterType.BOOLEAN,
        "array": ParameterType.ARRAY,
        "object": ParameterType.OBJECT,
    }

    for name, prop in properties.items():
        json_type = prop.get("type", "string")
        param_type = type_mapping.get(json_type, ParameterType.STRING)

        spec = ParameterSpec(
            name=name,
            param_type=param_type,
            required=name in required,
            default=prop.get("default"),
            description=prop.get("description", ""),
        )
        specs.append(spec)

    return ParameterEnforcer(tool_name=tool_name, parameter_specs=specs)


# Pre-configured enforcers for common tools
TOOL_ENFORCERS: Dict[str, ParameterEnforcer] = {}


def get_enforcer_for_tool(tool_name: str) -> Optional[ParameterEnforcer]:
    """Get the pre-configured enforcer for a tool, if available."""
    return TOOL_ENFORCERS.get(tool_name)


def register_tool_enforcer(
    tool_name: str,
    specs: List[ParameterSpec],
) -> None:
    """Register a parameter enforcer for a tool."""
    TOOL_ENFORCERS[tool_name] = ParameterEnforcer(
        tool_name=tool_name,
        parameter_specs=specs,
    )


# Register enforcers for tools that commonly have missing parameters
register_tool_enforcer(
    "symbol",
    [
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
    ],
)

register_tool_enforcer(
    "read",
    [
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
            description="Line offset to start reading from",
        ),
        ParameterSpec(
            name="limit",
            param_type=ParameterType.INTEGER,
            required=False,
            default=2000,
            description="Maximum number of lines to read",
        ),
    ],
)

register_tool_enforcer(
    "grep",
    [
        ParameterSpec(
            name="pattern",
            param_type=ParameterType.STRING,
            required=True,
            description="Pattern to search for",
        ),
        ParameterSpec(
            name="path",
            param_type=ParameterType.STRING,
            required=False,
            default=".",
            description="Path to search in",
            inference_strategy=InferenceStrategy.FROM_WORKING_DIR,
        ),
    ],
)

register_tool_enforcer(
    "ls",
    [
        ParameterSpec(
            name="path",
            param_type=ParameterType.STRING,
            required=False,
            default=".",
            description="Path to list",
            inference_strategy=InferenceStrategy.FROM_WORKING_DIR,
        ),
    ],
)

register_tool_enforcer(
    "glob",
    [
        ParameterSpec(
            name="pattern",
            param_type=ParameterType.STRING,
            required=True,
            description="Glob pattern to match",
        ),
        ParameterSpec(
            name="path",
            param_type=ParameterType.STRING,
            required=False,
            default=".",
            description="Base path for glob",
            inference_strategy=InferenceStrategy.FROM_WORKING_DIR,
        ),
    ],
)
