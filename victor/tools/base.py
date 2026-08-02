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

"""Base tool framework for CodingAgent."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from pydantic import BaseModel, Field

# NOTE: ``jsonschema`` is imported lazily inside ``validate_parameters_detailed``.
# Importing it at module scope pulls ``rfc3987`` (and its format-checking machinery)
# into the CLI cold-start path via ``import victor.tools``; deferring it keeps
# ``victor --help`` fast. See tests/unit/runtime/test_cli_import_weight_guard.py.

# Import enums from separate module
from victor.tools.enums import (
    AccessMode,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    Priority,
    SchemaLevel,
)

# Import metadata classes from separate module
from victor.tools.metadata import ToolMetadata

# NOTE: Enums (CostTier, Priority, AccessMode, ExecutionCategory, DangerLevel)
# have been moved to victor/tools/enums.py and are imported above.
# ToolMetadata lives in victor/tools/metadata.py.
# ToolMetadataRegistry lives in victor/tools/metadata_registry.py and is exposed
# here only through a deprecated lazy re-export for backward compatibility.
# ToolRegistry, Hook, and HookError have been moved to victor/tools/registry.py
# Import them with: from victor.tools.registry import ToolRegistry, Hook, HookError

# Import registry classes from canonical location (for backward compatibility with deprecation)
import warnings as _warnings

# Direct imports for Hook and HookError (no deprecation needed)
from victor.tools.registry import Hook, HookError  # noqa: F401

# Lazy import with deprecation warning for ToolRegistry
_ToolRegistry = None
_ToolMetadataRegistry = None


def __getattr__(name: str):
    """Lazy imports with deprecation warnings for moved tool classes.

    This provides backward compatibility while warning users to import
    from the canonical location.
    """
    global _ToolMetadataRegistry
    if name == "ToolRegistry":
        global _ToolRegistry
        if _ToolRegistry is None:
            from victor.tools.registry import ToolRegistry as _TR

            _ToolRegistry = _TR

        _warnings.warn(
            "Direct import of ToolRegistry from victor.tools.base is deprecated. "
            "Use 'from victor.tools.registry import ToolRegistry' instead. "
            "This re-export will be removed in version 0.10.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ToolRegistry
    if name == "ToolMetadataRegistry":
        if _ToolMetadataRegistry is None:
            from victor.tools.metadata_registry import ToolMetadataRegistry as _TMR

            _ToolMetadataRegistry = _TMR

        _warnings.warn(
            "Direct import of ToolMetadataRegistry from victor.tools.base is deprecated. "
            "Use 'from victor.tools.metadata_registry import ToolMetadataRegistry' instead. "
            "This re-export will be removed in version 0.10.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _ToolMetadataRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class ToolValidationResult:
    """Result of tool parameter validation.

    Renamed from ValidationResult to be semantically distinct:
    - ToolValidationResult (here): Tool parameter validation with invalid_params
    - ConfigValidationResult (victor.core.validation): Configuration validation with ValidationIssue list
    - ContentValidationResult (victor.framework.middleware): Content validation with fixed_content
    - ParameterValidationResult (victor.agent.parameter_enforcer): Parameter enforcement with missing_required
    - CodeValidationResult (victor.evaluation.correction.types): Code validation with syntax/imports

    Provides detailed information about validation failures including
    which parameters failed and why.
    """

    valid: bool
    errors: List[str] = field(default_factory=list)
    invalid_params: Dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Allow using ToolValidationResult in boolean context."""
        return self.valid

    @classmethod
    def success(cls) -> "ToolValidationResult":
        """Create a successful validation result."""
        return cls(valid=True)

    @classmethod
    def failure(
        cls, errors: List[str], invalid_params: Optional[Dict[str, str]] = None
    ) -> "ToolValidationResult":
        """Create a failed validation result."""
        return cls(valid=False, errors=errors, invalid_params=invalid_params or {})


class ToolParameter(BaseModel):
    """Tool parameter definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, number, boolean, etc.)")
    description: str = Field(..., description="Parameter description")
    enum: Optional[list[str]] = Field(default=None, description="Allowed values for enum types")
    required: bool = Field(default=True, description="Whether parameter is required")


class ToolResult(BaseModel):
    """Result from tool execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    output: Any = Field(..., description="Tool output data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ToolConfig:
    """Configuration container for tools.

    This replaces global state by providing a centralized configuration
    object that can be passed through context. Tools should access config
    from context['tool_config'] instead of using module-level globals.

    Example usage in tools:
        ```python
        async def git(operation: str, context: Dict[str, Any], **kwargs):
            config = context.get('tool_config')
            if config and config.provider:
                # Use provider for AI features
                pass
        ```

    Example setup in orchestrator:
        ```python
        config = ToolConfig(provider=my_provider, model="gpt-4")
        executor.update_context(tool_config=config)
        ```
    """

    def __init__(
        self,
        provider: Optional[Any] = None,
        model: Optional[str] = None,
        max_complexity: int = 10,
        web_fetch_top: Optional[int] = None,
        web_fetch_pool: Optional[int] = None,
        max_content_length: int = 5000,
        batch_concurrency: int = 5,
        batch_max_files: int = 100,
        generic_result_cache_enabled: bool = False,
        generic_result_cache_ttl: int = 300,
        http_connection_pool_enabled: bool = False,
        http_connection_pool_max_connections: int = 100,
        http_connection_pool_max_connections_per_host: int = 10,
        http_connection_pool_connection_timeout: int = 30,
        http_connection_pool_total_timeout: int = 60,
    ):
        """Initialize tool configuration.

        Args:
            provider: LLM provider for AI-powered features (commit messages, summaries)
            model: Model name to use with the provider
            max_complexity: Maximum cyclomatic complexity threshold for code review
            web_fetch_top: Number of top results to fetch for web search
            web_fetch_pool: Pool size for concurrent web fetches
            max_content_length: Maximum content length for web scraping
            batch_concurrency: Concurrent operations for batch processing
            batch_max_files: Maximum files for batch operations
            generic_result_cache_enabled: Enable GenericResultCache for tool runtime paths
            generic_result_cache_ttl: TTL for generic result cache entries in seconds
            http_connection_pool_enabled: Enable shared HTTP connection pool for web tools
            http_connection_pool_max_connections: Max total pooled HTTP connections
            http_connection_pool_max_connections_per_host: Max pooled connections per host
            http_connection_pool_connection_timeout: HTTP connect timeout in seconds
            http_connection_pool_total_timeout: Total HTTP timeout in seconds
        """
        self.provider = provider
        self.model = model
        self.max_complexity = max_complexity
        self.web_fetch_top = web_fetch_top
        self.web_fetch_pool = web_fetch_pool
        self.max_content_length = max_content_length
        self.batch_concurrency = batch_concurrency
        self.batch_max_files = batch_max_files
        self.generic_result_cache_enabled = generic_result_cache_enabled
        self.generic_result_cache_ttl = generic_result_cache_ttl
        self.http_connection_pool_enabled = http_connection_pool_enabled
        self.http_connection_pool_max_connections = http_connection_pool_max_connections
        self.http_connection_pool_max_connections_per_host = (
            http_connection_pool_max_connections_per_host
        )
        self.http_connection_pool_connection_timeout = http_connection_pool_connection_timeout
        self.http_connection_pool_total_timeout = http_connection_pool_total_timeout

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> Optional["ToolConfig"]:
        """Extract ToolConfig from context dictionary.

        Args:
            context: Context dictionary passed to tools

        Returns:
            ToolConfig if present in context, None otherwise
        """
        return context.get("tool_config") if context else None


# ToolMetadata class has been moved to victor/tools/metadata.py and is imported above


@runtime_checkable
class ToolMetadataProvider(Protocol):
    """Protocol for tools that provide semantic metadata.

    All tools should implement this protocol to enable dynamic discovery
    and semantic tool selection. The protocol enforces that every tool
    can provide metadata either explicitly or through auto-generation.

    Example:
        class MyTool(BaseTool):
            # Explicit metadata (recommended for precise control)
            @property
            def metadata(self) -> ToolMetadata:
                return ToolMetadata(
                    category="my_category",
                    keywords=["keyword1", "keyword2"],
                    use_cases=["use case 1", "use case 2"],
                )

        # Or rely on auto-generation (good for simple tools):
        class SimpleTool(BaseTool):
            # metadata property returns None, get_metadata() auto-generates
            pass
    """

    @property
    def name(self) -> str:
        """Tool name."""
        ...

    @property
    def description(self) -> str:
        """Tool description."""
        ...

    @property
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters JSON schema."""
        ...

    @property
    def cost_tier(self) -> "CostTier":
        """Cost tier for the tool."""
        ...

    def get_metadata(self) -> ToolMetadata:
        """Get semantic metadata for tool selection.

        This method MUST return valid ToolMetadata. Implementations should:
        1. Return explicit metadata if defined (via metadata property)
        2. Auto-generate metadata from tool properties if not explicit

        Returns:
            ToolMetadata with semantic information for tool selection
        """
        ...


class BaseTool(ABC):
    """Abstract base class for all tools.

    Tools should implement name, description, parameters, and execute().
    Optionally, tools can override the metadata property to provide
    semantic information for dynamic tool selection.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass

    @property
    def metadata(self) -> Optional[ToolMetadata]:
        """Semantic metadata for tool selection.

        Override this property to provide category, keywords, use_cases,
        and examples for dynamic tool selection. If None, metadata will
        be auto-generated from tool properties.

        Returns:
            ToolMetadata with semantic information, or None for auto-generation
        """
        return None

    def get_metadata(self) -> ToolMetadata:
        """Get semantic metadata for tool selection (ToolMetadataProvider contract).

        This method fulfills the ToolMetadataProvider protocol and ALWAYS
        returns valid ToolMetadata. It follows a two-tier strategy:

        1. If explicit metadata is defined (via metadata property), use it
        2. Otherwise, auto-generate metadata from tool properties

        This ensures ALL tools can participate in semantic tool selection
        without requiring manual configuration files.

        Returns:
            ToolMetadata with semantic information for tool selection
        """
        # Delegate to the single resolution authority (tool-supply P6). The two-tier
        # precedence (explicit metadata -> auto-generate) now lives in one place so every
        # consumer resolves a tool's metadata identically and the autogen result is cached.
        from victor.tools.contract import resolve_contract

        return resolve_contract(self)

    @property
    def cost_tier(self) -> CostTier:
        """Cost tier for the tool.

        Override this property in subclasses to specify the appropriate tier.

        Tiers:
            FREE: Local operations (filesystem, bash, git) - default
            LOW: Compute-only operations (code review, refactoring)
            MEDIUM: External API calls (web search, fetch)
            HIGH: Resource-intensive (batch processing 100+ files)

        Returns:
            CostTier enum value
        """
        return CostTier.FREE  # Default: local operations are free

    @property
    def is_idempotent(self) -> bool:
        """Whether the tool execution is idempotent.

        An idempotent tool produces the same result for the same input and has
        no side effects that would change subsequent executions. This property
        enables optimizations such as:
        - Result caching (memoization)
        - Safe retries on transient failures
        - Parallel execution without coordination
        - Deduplication of redundant calls

        Examples of idempotent tools:
        - File read operations
        - Search/query operations
        - Git status/log/diff (read-only)
        - Web fetch (GET requests)

        Examples of non-idempotent tools:
        - File write/edit operations
        - Git commit/push
        - API mutations (POST, PUT, DELETE)
        - Docker container operations

        Override this property in subclasses for idempotent operations.

        Returns:
            True if tool execution is idempotent, False otherwise (default)
        """
        return False  # Default: assume side effects

    @staticmethod
    def convert_parameters_to_schema(parameters: List[ToolParameter]) -> Dict[str, Any]:
        """Convert list of ToolParameter objects to JSON Schema format.

        Args:
            parameters: List of ToolParameter objects

        Returns:
            JSON Schema dictionary
        """
        properties = {}
        required = []

        for param in parameters:
            param_schema: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }

            if param.enum:
                param_schema["enum"] = param.enum

            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        schema = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,  # Reject unknown/hallucinated arguments
        }

        if required:
            schema["required"] = required

        return schema

    @abstractmethod
    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            _exec_ctx: Framework execution context (reserved name to avoid collision
                      with tool parameters). Contains shared resources like code_manager.
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        pass

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON Schema format.

        Returns:
            JSON Schema representation
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_schema(self, level: "SchemaLevel" = None) -> Dict[str, Any]:
        """Generate JSON schema at specified verbosity level.

        Args:
            level: Schema verbosity level (FULL, COMPACT, or STUB). Defaults to FULL.

        Returns:
            JSON Schema with appropriate detail level.

        Example:
            # FULL: Complete schema (~100-150 tokens)
            tool.to_schema(SchemaLevel.FULL)

            # COMPACT: All params, shorter descriptions (~60-80 tokens, ~20% reduction)
            tool.to_schema(SchemaLevel.COMPACT)

            # STUB: Minimal schema, required params only (~25-40 tokens)
            tool.to_schema(SchemaLevel.STUB)
        """
        from victor.tools.enums import SchemaLevel

        if level is None:
            level = SchemaLevel.FULL

        if level == SchemaLevel.FULL:
            return self.to_json_schema()

        # Get limits from schema level
        max_desc = level.max_description_chars
        max_param_desc = level.max_param_description_chars
        include_optional = level.include_optional_params

        # Build parameters based on level
        params = {}
        required_list = self.parameters.get("required", [])

        for name, schema in self.parameters.get("properties", {}).items():
            # STUB: only required params; COMPACT: all params
            if not include_optional and name not in required_list:
                continue

            # Truncate description
            desc = schema.get("description", "")
            if len(desc) > max_param_desc:
                desc = desc[: max_param_desc - 3] + "..."

            params[name] = {
                "type": schema.get("type", "string"),
                "description": desc,
            }

            # Preserve enum if present
            if "enum" in schema:
                params[name]["enum"] = schema["enum"]

        # Truncate tool description
        desc = self.description
        # For COMPACT/STUB: use first 1-2 sentences
        first_sentence_end = desc.find(".")
        if first_sentence_end > 0 and first_sentence_end < max_desc:
            # Try to get second sentence for COMPACT
            if level == SchemaLevel.COMPACT:
                second_end = desc.find(".", first_sentence_end + 1)
                if second_end > 0 and second_end < max_desc:
                    desc = desc[: second_end + 1]
                else:
                    desc = desc[: first_sentence_end + 1]
            else:
                desc = desc[: first_sentence_end + 1]

        if len(desc) > max_desc:
            desc = desc[: max_desc - 3] + "..."

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": [r for r in required_list if r in params],
                },
            },
        }

    def validate_parameters(self, **kwargs: Any) -> bool:
        """Validate provided parameters against schema.

        Simple boolean validation - use validate_parameters_detailed() for
        detailed error information.

        Args:
            **kwargs: Parameters to validate

        Returns:
            True if valid, False otherwise
        """
        return self.validate_parameters_detailed(**kwargs).valid

    def validate_parameters_detailed(self, **kwargs: Any) -> ToolValidationResult:
        """Validate provided parameters against JSON Schema with detailed errors.

        Uses JSON Schema Draft 7 validation for comprehensive type checking,
        required field validation, enum constraints, and nested object validation.

        Args:
            **kwargs: Parameters to validate

        Returns:
            ToolValidationResult with detailed error information
        """
        schema = self.parameters

        # Handle empty/minimal schemas gracefully
        if not schema or schema == {"type": "object", "properties": {}}:
            return ToolValidationResult.success()

        try:
            # Ensure schema has proper structure for validation
            if "type" not in schema:
                schema = {"type": "object", **schema}

            # Deferred import (kept out of module scope to avoid pulling
            # rfc3987 into the CLI cold-start path).
            import jsonschema
            from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError

            # Create validator with format checking disabled for flexibility
            validator = Draft7Validator(schema)

            # Collect all validation errors
            errors: List[str] = []
            invalid_params: Dict[str, str] = {}

            for error in validator.iter_errors(kwargs):
                # Format error message based on error type
                if error.validator == "required":
                    # Extract missing field from error message
                    missing_fields = error.validator_value
                    for field_name in missing_fields:
                        if field_name not in kwargs:
                            msg = f"Required parameter '{field_name}' is missing"
                            errors.append(msg)
                            invalid_params[field_name] = "required"
                elif error.validator == "type":
                    # Type mismatch error
                    path = ".".join(str(p) for p in error.path) or "root"
                    expected = error.validator_value
                    actual = type(error.instance).__name__
                    msg = f"Parameter '{path}' has wrong type: expected {expected}, got {actual}"
                    errors.append(msg)
                    if error.path:
                        invalid_params[str(error.path[0])] = f"type: expected {expected}"
                elif error.validator == "enum":
                    # Enum constraint violation
                    path = ".".join(str(p) for p in error.path) or "root"
                    allowed = error.validator_value
                    msg = f"Parameter '{path}' must be one of: {allowed}"
                    errors.append(msg)
                    if error.path:
                        invalid_params[str(error.path[0])] = "invalid enum value"
                elif error.validator == "additionalProperties":
                    # Extra properties that aren't allowed
                    path = ".".join(str(p) for p in error.path) or "root"
                    msg = f"Unknown parameter in '{path}': {error.message}"
                    errors.append(msg)
                else:
                    # Generic error handling
                    path = ".".join(str(p) for p in error.path) or "root"
                    msg = f"Validation error at '{path}': {error.message}"
                    errors.append(msg)
                    if error.path:
                        invalid_params[str(error.path[0])] = error.message

            if errors:
                return ToolValidationResult.failure(errors, invalid_params)

            return ToolValidationResult.success()

        except JsonSchemaValidationError as e:
            # Single validation error (shouldn't happen with iter_errors but handle it)
            return ToolValidationResult.failure(
                [str(e.message)],
                {str(e.path[0]): e.message} if e.path else {},
            )
        except jsonschema.SchemaError as e:
            # Invalid schema - this is a programming error
            return ToolValidationResult.failure(
                [f"Invalid tool schema: {e.message}"],
                {},
            )
        except Exception:
            # Unexpected error - fall back to basic validation
            return self._fallback_validate(**kwargs)

    def _fallback_validate(self, **kwargs: Any) -> ToolValidationResult:
        """Basic validation fallback when JSON Schema validation fails.

        Args:
            **kwargs: Parameters to validate

        Returns:
            ToolValidationResult with basic validation
        """
        errors: List[str] = []
        invalid_params: Dict[str, str] = {}

        required_params = self.parameters.get("required", [])
        properties = self.parameters.get("properties", {})

        # Check required parameters
        for param in required_params:
            if param not in kwargs:
                errors.append(f"Required parameter '{param}' is missing")
                invalid_params[param] = "required"

        # Check parameter types (basic check)
        for param, value in kwargs.items():
            if param in properties:
                expected_type = properties[param].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    actual_type = type(value).__name__
                    errors.append(
                        f"Parameter '{param}' has wrong type: expected {expected_type}, got {actual_type}"
                    )
                    invalid_params[param] = f"type: expected {expected_type}"

        if errors:
            return ToolValidationResult.failure(errors, invalid_params)
        return ToolValidationResult.success()

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type.

        Args:
            value: Value to check
            expected_type: Expected JSON Schema type

        Returns:
            True if types match
        """
        type_mapping: Dict[str, Union[type[Any], tuple[type[Any], ...]]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, allow it

        return isinstance(value, expected_python_type)


# ToolRegistry class has been moved to victor/tools/registry.py
# Import it with: from victor.tools.registry import ToolRegistry


# ToolMetadataRegistry class has moved to victor/tools/metadata_registry.py
