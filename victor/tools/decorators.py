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


import asyncio
from enum import Enum
import dataclasses
import inspect
import logging
import types
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Union, get_origin, get_args

from docstring_parser import parse

from victor.core.errors import ToolExecutionError, ToolValidationError
from victor.tools.base import (
    AccessMode,
    BaseTool,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    Priority,
    ToolMetadata,
    ToolResult,
)

# Module-level flag to control auto-registration (can be disabled for testing)
_AUTO_REGISTER_TOOLS = True


def set_auto_register(enabled: bool) -> None:
    """Enable or disable automatic tool registration with metadata registry.

    By default, tools are automatically registered with the global
    ToolMetadataRegistry when decorated with @tool. This enables
    registry-driven tool discovery without explicit registration calls.

    Args:
        enabled: If True, tools are auto-registered at decoration time.
                 Set to False for unit tests that need isolated tool instances.
    """
    global _AUTO_REGISTER_TOOLS
    _AUTO_REGISTER_TOOLS = enabled


def _auto_register_tool(tool_instance: BaseTool) -> None:
    """Register a tool instance with the global metadata registry.

    This is called automatically when a tool is decorated with @tool,
    unless auto-registration is disabled via set_auto_register(False).

    Args:
        tool_instance: The tool instance to register
    """
    if not _AUTO_REGISTER_TOOLS:
        return

    try:
        from victor.tools.metadata_registry import register_tool_metadata

        register_tool_metadata(tool_instance)
        logger.debug(f"Auto-registered tool '{tool_instance.name}' with metadata registry")
    except ImportError:
        # metadata_registry not available (e.g., during early import stages)
        logger.debug(f"Could not auto-register tool '{tool_instance.name}': registry not available")


logger = logging.getLogger(__name__)

# Global flag to control deprecation warnings for legacy tool names
_WARN_ON_LEGACY_NAMES = False


def set_legacy_name_warnings(enabled: bool) -> None:
    """Enable or disable warnings when legacy tool names are used.

    Args:
        enabled: If True, log warnings when tools are called with legacy names.
                 Useful for identifying code that needs migration.
    """
    global _WARN_ON_LEGACY_NAMES
    _WARN_ON_LEGACY_NAMES = enabled


def resolve_tool_name(name: str, warn_on_legacy: bool = False) -> str:
    """Resolve a tool name to its canonical form.

    This provides centralized name resolution that can be used across the codebase.
    Uses the tool_names registry if available, otherwise returns the name unchanged.

    Args:
        name: Tool name (canonical or legacy)
        warn_on_legacy: If True, emit a warning when a legacy name is used

    Returns:
        Canonical tool name

    Example:
        >>> resolve_tool_name("execute_bash")
        "shell"
        >>> resolve_tool_name("shell")
        "shell"
    """
    try:
        from victor.tools.tool_names import get_canonical_name, TOOL_ALIASES

        canonical = get_canonical_name(name)

        # Check if this was a legacy name
        if (warn_on_legacy or _WARN_ON_LEGACY_NAMES) and name in TOOL_ALIASES:
            logger.warning(
                f"Legacy tool name '{name}' used. Consider using canonical name '{canonical}' instead."
            )

        return canonical
    except ImportError:
        # tool_names module not available, return unchanged
        return name


def _enum_values_to_schema(enum_cls: type[Enum]) -> Dict[str, Any]:
    """Convert an Enum class into a JSON Schema enum definition."""
    values = [member.value for member in enum_cls]
    if not values:
        return {"type": "string"}

    if all(isinstance(value, bool) for value in values):
        schema_type = "boolean"
    elif all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        schema_type = "integer"
    elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        schema_type = "number"
    else:
        schema_type = "string"

    return {"type": schema_type, "enum": values}


def _resolve_annotation_string(annotation: str, globalns: Optional[Dict[str, Any]] = None) -> Any:
    """Resolve a string annotation against the function globals when possible."""
    namespace: Dict[str, Any] = {"__builtins__": __builtins__}
    if globalns:
        namespace.update(globalns)
    try:
        return eval(annotation, namespace, {})
    except Exception:
        return None


def _sanitize_default_for_llm(value: Any) -> Any:
    """Convert Python-specific default values into JSON-safe forms."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _sanitize_default_for_llm(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_default_for_llm(item) for item in value]
    pathlike = getattr(value, "__fspath__", None)
    if callable(pathlike):
        try:
            return pathlike()
        except Exception:
            return str(value)
    return value


def _build_object_schema_from_annotations(
    annotation: Any,
    *,
    globalns: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a JSON schema for classes with typed attributes."""

    class_annotations = getattr(annotation, "__annotations__", None)
    if not class_annotations or not isinstance(class_annotations, dict):
        return None

    properties: Dict[str, Any] = {}
    required: List[str] = []
    for field_name, field_type in class_annotations.items():
        if field_name.startswith("_"):
            continue
        properties[field_name] = _get_json_schema_type(field_type, globalns=globalns)
        try:
            default = getattr(annotation, field_name)
        except Exception:
            default = inspect.Parameter.empty
        if default is inspect.Parameter.empty:
            required.append(field_name)

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def _build_object_schema_for_class(annotation: Any) -> Optional[Dict[str, Any]]:
    """Build schema for user-defined class-like annotations."""

    if not inspect.isclass(annotation):
        return None

    if dataclasses.is_dataclass(annotation):
        try:
            fields = dataclasses.fields(annotation)
        except TypeError:
            fields = []
        if fields:
            properties: Dict[str, Any] = {}
            required: List[str] = []
            for field in fields:
                properties[field.name] = _get_json_schema_type(field.type, globalns=None)
                if (
                    field.default is dataclasses.MISSING
                    and field.default_factory is dataclasses.MISSING
                ):
                    required.append(field.name)
            schema = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required
            return schema

    if hasattr(annotation, "model_json_schema"):
        try:
            model_schema = annotation.model_json_schema()
            if isinstance(model_schema, dict) and model_schema.get("type") == "object":
                properties = model_schema.get("properties")
                if isinstance(properties, dict):
                    return {
                        "type": "object",
                        "properties": properties,
                        "required": model_schema.get("required", []),
                    }
        except Exception:
            pass

    return _build_object_schema_from_annotations(annotation, globalns=None)


def _get_json_schema_type(
    annotation: Any,
    *,
    globalns: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert Python type annotation to JSON Schema type.

    Handles both runtime types (e.g., bool) and string annotations
    from `from __future__ import annotations` (PEP 563).

    Args:
        annotation: Python type annotation (type or string)

    Returns:
        JSON Schema type definition
    """
    if annotation == inspect.Parameter.empty:
        return {"type": "string"}

    # Handle string annotations from `from __future__ import annotations` (PEP 563)
    # These are stored as strings like "bool", "int", "Optional[str]", etc.
    if isinstance(annotation, str):
        annotation_str = annotation.strip()

        resolved_annotation = _resolve_annotation_string(annotation_str, globalns)
        if resolved_annotation is not None:
            return _get_json_schema_type(resolved_annotation, globalns=globalns)

        # Handle Optional[X] as string
        if annotation_str.startswith("Optional[") and annotation_str.endswith("]"):
            inner = annotation_str[9:-1]  # Extract inner type
            return _get_json_schema_type(inner, globalns=globalns)

        # Handle List[X] as string
        if annotation_str.startswith("List[") and annotation_str.endswith("]"):
            inner = annotation_str[5:-1]
            return {
                "type": "array",
                "items": _get_json_schema_type(inner, globalns=globalns),
            }

        # Handle Dict as string
        if annotation_str.startswith("Dict[") or annotation_str == "dict":
            return {"type": "object"}

        # Handle Literal types as enums
        if annotation_str.startswith("Literal[") and annotation_str.endswith("]"):
            # Extract literal values: Literal["a", "b"] -> ["a", "b"]
            inner = annotation_str[8:-1]
            # Simple parsing for string literals
            values = [v.strip().strip("'\"") for v in inner.split(",")]
            return {"type": "string", "enum": values}

        # Basic types as strings
        if annotation_str == "bool":
            return {"type": "boolean"}
        if annotation_str == "int":
            return {"type": "integer"}
        if annotation_str == "float":
            return {"type": "number"}
        if annotation_str == "str":
            return {"type": "string"}

        # Default to string for unknown string annotations
        return {"type": "string"}

    # Handle Optional[X] -> X with nullable (runtime types)
    origin = get_origin(annotation)
    args = get_args(annotation)

    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return _enum_values_to_schema(annotation)

    if origin in (Union, types.UnionType):
        # Check if it's Optional (Union with None)
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            # It's Optional[X]
            inner_schema = _get_json_schema_type(non_none_args[0], globalns=globalns)
            return inner_schema  # JSON Schema handles nullable differently
        else:
            # Complex Union - default to string
            return {"type": "string"}

    if origin is list or annotation is list:
        if args:
            items_schema = _get_json_schema_type(args[0], globalns=globalns)
            return {"type": "array", "items": items_schema}
        return {"type": "array", "items": {"type": "string"}}

    if origin is dict or annotation is dict:
        return {"type": "object"}

    # Support structured user-defined types.
    if inspect.isclass(annotation):
        custom_schema = _build_object_schema_for_class(annotation)
        if custom_schema:
            return custom_schema

    # Basic types (runtime)
    if annotation in (int,):
        return {"type": "integer"}
    if annotation in (float,):
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is str:
        return {"type": "string"}

    # Default to string for unknown types
    return {"type": "string"}


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    cost_tier: CostTier = CostTier.FREE,
    category: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    use_cases: Optional[List[str]] = None,
    examples: Optional[List[str]] = None,
    priority_hints: Optional[List[str]] = None,
    # Selection/approval metadata parameters
    priority: Priority = Priority.MEDIUM,
    access_mode: AccessMode = AccessMode.READONLY,
    danger_level: DangerLevel = DangerLevel.SAFE,
    # Stage affinity for conversation state machine
    stages: Optional[List[str]] = None,
    # NEW: Mandatory keywords that force tool inclusion
    mandatory_keywords: Optional[List[str]] = None,
    # NEW: Task types for classification-aware selection
    task_types: Optional[List[str]] = None,
    # LEGACY/COMPAT: Parameters that indicate forward progress in repeated tool calls
    progress_params: Optional[List[str]] = None,
    # NEW: Signature parameters for loop detection
    signature_params: Optional[List[str]] = None,
    # NEW: Execution category for parallel execution
    execution_category: Optional[str] = None,
    # NEW: Availability check for optional tools requiring configuration
    availability_check: Optional[Callable[[], bool]] = None,
    # Per-tool timeout override (seconds). Overrides pipeline's per_tool_timeout_seconds.
    # None = use pipeline default (30s). Set higher for slow tools (code_search, web_search).
    timeout: Optional[float] = None,
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    A decorator that converts a Python function into a Victor tool.

    This decorator automatically extracts the tool's name, description,
    and parameters from the function's signature and docstring. It also
    supports optional semantic metadata for dynamic tool selection.

    Tool Naming Strategy:
    - If `name` is provided, use it as the canonical name
    - If function name is in TOOL_ALIASES, auto-resolve to canonical name
    - Otherwise, use function name as-is

    Category Semantics:
    - category="filesystem": File operations (read, write, ls)
    - category="git": Version control operations
    - category="web": Web search/fetch operations
    - category="analysis": Code analysis and review tools
    - category="execution": Shell and command execution
    - etc.

    Priority Semantics (decoupled from category):
    - CRITICAL: Always available (read, ls, grep, shell)
    - HIGH: Most tasks (write, edit, git)
    - MEDIUM: Task-specific (docker, db, test)
    - LOW: Specialized (batch, scaffold)
    - CONTEXTUAL: Based on task classification

    Args:
        func: The function to wrap (optional, for @tool without parentheses)
        name: Explicit tool name (overrides function name and auto-resolution)
        aliases: Additional names that resolve to this tool (backward compat)
        cost_tier: Cost tier for the tool (FREE, LOW, MEDIUM, HIGH)
        category: Tool category for semantic grouping.
        keywords: Keywords that trigger this tool in user requests
        use_cases: High-level use cases for semantic matching
        examples: Example requests that should match this tool
        priority_hints: Usage hints for tool selection
        priority: Tool priority for selection (CRITICAL, HIGH, MEDIUM, LOW, CONTEXTUAL)
        access_mode: Access mode for approval tracking (READONLY, WRITE, EXECUTE, NETWORK, MIXED)
        danger_level: Danger level for warnings (SAFE, LOW, MEDIUM, HIGH, CRITICAL)
        stages: Conversation stages where this tool is most relevant. Valid stages:
                "initial", "planning", "reading", "analysis", "execution",
                "verification", "completion". Tools with stages defined will
                be prioritized when the conversation is in matching stages.
        mandatory_keywords: Keywords that FORCE this tool to be included when matched.
                           Unlike regular keywords, these guarantee tool inclusion.
                           E.g., ["show diff", "compare"] for shell tool.
        task_types: Task types this tool is relevant for. Valid types:
                   "analysis", "action", "generation", "search", "edit", "default".
                   Used for classification-aware tool selection.
        progress_params: Backward-compatible alias for signature_params.
                        Historically used to describe which arguments represent
                        forward progress for repeated tool invocations.
        signature_params: Parameters to track in loop detection signatures.
                         Only these parameters are included in the signature used for
                         loop detection. Pagination parameters (offset, limit, k, etc.)
                         should be excluded to allow reading results in chunks.
                         E.g., ["path"] for read tool (excludes offset/limit for pagination).
        execution_category: Category for parallel execution. Valid values:
                           "read_only", "write", "compute", "network", "execute", "mixed".
                           Determines which tools can safely run concurrently.
        availability_check: Optional callable that returns True if the tool is available.
                           Use this for tools that require external configuration
                           (e.g., Slack, Teams, Jira). When provided, the tool's
                           is_available() method will call this function. Unavailable
                           tools are excluded from tool selection and system prompts.
                           Example: availability_check=is_slack_configured

    The docstring should follow the Google Python Style Guide format.

    Example with auto-generated metadata:
        @tool
        def my_tool(param1: str, param2: int = 5):
            '''This is the tool's description.

            Args:
                param1: Description of the first parameter.
                param2: Description of the second parameter.
            '''
            # ... tool logic ...

    Example with core tool (always available):
        @tool(name="shell", category="core")
        def execute_bash(cmd: str):
            '''Run shell command. Platform-agnostic.'''
            # ... tool logic ...

    Example with explicit metadata:
        @tool(
            name="web",
            cost_tier=CostTier.MEDIUM,
            category="web",
            keywords=["search", "google", "find online"],
        )
        def web_search_tool(query: str):
            '''Search the web - makes external API calls.'''
            # ... tool logic ...
    """
    # Capture semantic metadata parameters
    metadata_params = {
        "category": category,
        "keywords": keywords,
        "use_cases": use_cases,
        "examples": examples,
        "priority_hints": priority_hints,
        "stages": stages,
        "mandatory_keywords": mandatory_keywords,
        "task_types": task_types,
        "progress_params": progress_params,
        "signature_params": signature_params,
        "execution_category": execution_category,
        "availability_check": availability_check,
        "timeout": timeout,
    }

    # Capture selection/approval metadata parameters
    selection_params = {
        "priority": priority,
        "access_mode": access_mode,
        "danger_level": danger_level,
    }

    # Capture naming parameters
    _explicit_name = name
    _explicit_aliases = aliases or []

    def decorator(fn: Callable) -> Callable:
        # CRITICAL FIX: Check if function is async and create appropriate wrapper
        # If the wrapped function is async, the wrapper MUST be async to properly await it
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args, **kwargs) -> Any:  # type: ignore[misc]
                # This async wrapper properly awaits the underlying async function
                return await fn(*args, **kwargs)

            # Mark as tool for dynamic discovery
            async_wrapper._is_tool = True  # type: ignore[attr-defined]
            async_wrapper._tool_timeout = timeout  # type: ignore[attr-defined]
            async_wrapper.Tool = _create_tool_class(
                fn,
                cost_tier=cost_tier,
                metadata_params=metadata_params,
                selection_params=selection_params,
                explicit_name=_explicit_name,
                explicit_aliases=_explicit_aliases,
            )

            return async_wrapper
        else:

            @wraps(fn)
            def sync_wrapper(*args, **kwargs) -> Any:
                # Sync wrapper for sync functions
                return fn(*args, **kwargs)

            # Mark as tool for dynamic discovery
            sync_wrapper._is_tool = True  # type: ignore[attr-defined]
            sync_wrapper._tool_timeout = timeout  # type: ignore[attr-defined]
            sync_wrapper.Tool = _create_tool_class(
                fn,
                cost_tier=cost_tier,
                metadata_params=metadata_params,
                selection_params=selection_params,
                explicit_name=_explicit_name,
                explicit_aliases=_explicit_aliases,
            )

            return sync_wrapper

    # Support both @tool and @tool(cost_tier=...) syntax
    if func is not None:
        # Called without parentheses: @tool
        return decorator(func)
    else:
        # Called with parentheses: @tool(cost_tier=...)
        return decorator


def _resolve_tool_name(func_name: str, explicit_name: Optional[str] = None) -> str:
    """Resolve the canonical tool name.

    Resolution order:
    1. Start with explicit name if provided, otherwise function name
    2. Auto-resolve to canonical form from TOOL_ALIASES registry
    3. Use name as-is if not in registry (fallback)

    CRITICAL: We ALWAYS resolve through the canonical name registry to ensure
    the tool name in the schema matches what the LLM expects. This prevents
    mismatches where a tool is registered as "search" but the canonical name
    is "code_search", causing "Unknown or disabled tool" errors.

    Args:
        func_name: The original function name
        explicit_name: Explicitly provided name (used as starting point for resolution)

    Returns:
        The resolved canonical tool name

    Examples:
        >>> _resolve_tool_name("search_tool", "search")
        "code_search"  # Resolved through registry
        >>> _resolve_tool_name("execute_bash", None)
        "shell"  # Auto-resolved through registry
        >>> _resolve_tool_name("custom_tool", None)
        "custom_tool"  # Not in registry, used as-is
    """
    # Start with explicit name or function name
    base_name = explicit_name if explicit_name else func_name

    # Always try to resolve to canonical form through registry
    # This ensures the tool name matches the canonical system
    try:
        from victor.tools.tool_names import get_canonical_name

        canonical = get_canonical_name(base_name)
        if canonical != base_name:
            logger.debug(f"Tool name resolved: '{base_name}' -> '{canonical}' (canonical form)")
        return canonical
    except ImportError:
        # Registry not available, use base name as-is
        logger.debug(f"Tool name registry unavailable, using '{base_name}' as-is")
        return base_name


def _create_tool_class(
    func: Callable,
    cost_tier: CostTier = CostTier.FREE,
    metadata_params: Optional[Dict[str, Any]] = None,
    selection_params: Optional[Dict[str, Any]] = None,
    explicit_name: Optional[str] = None,
    explicit_aliases: Optional[List[str]] = None,
) -> type:
    """Dynamically creates a class that wraps the given function to act as a BaseTool.

    Args:
        func: The function to wrap
        cost_tier: The cost tier for this tool
        metadata_params: Optional dict with category, keywords, use_cases, examples, priority_hints
        selection_params: Optional dict with priority, access_mode, danger_level
        explicit_name: Optional explicit tool name (overrides function name)
        explicit_aliases: Optional list of alias names for backward compatibility
    """
    metadata_params = metadata_params or {}
    selection_params = selection_params or {}
    explicit_aliases = explicit_aliases or []

    # Resolve the canonical tool name
    resolved_name = _resolve_tool_name(func.__name__, explicit_name)

    docstring = parse(func.__doc__ or "")
    tool_description = docstring.short_description or "No description provided."
    if docstring.long_description:
        tool_description += "\n\n" + docstring.long_description

    # NOTE: folded-tool-guidance hints (dependency:, extract:, etc.) were
    # removed from the LLM-facing description — they're internal routing hints
    # that bloat the schema with no actionable value for the model.

    # Create the JSON schema for the parameters
    sig = inspect.signature(func)
    param_docs = {p.arg_name: p.description for p in docstring.params}

    properties = {}
    required = []

    def sanitize_schema_for_llm(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Remove Python-specific annotations from JSON schema before sending to LLM.

        LLMs get confused by Python-specific types like Path, tuple[str, ...], etc.
        Convert these to standard JSON Schema types.

        Args:
            schema: Raw JSON schema with potential Python annotations

        Returns:
            Sanitized schema using only standard JSON Schema types
        """
        import hashlib

        sanitized = schema.copy()

        # Sanitize properties
        if "properties" in sanitized:
            for prop_name, prop_schema in sanitized["properties"].items():
                # Remove example/default if they contain Python types
                prop_schema.pop("example", None)
                prop_schema.pop("default", None)

                # Convert Python-specific types to JSON Schema types
                prop_type = prop_schema.get("type", "string")

                # Path -> string
                if "Path" in str(prop_type) or (
                    hasattr(prop_type, "__name__") and prop_type.__name__ == "Path"
                ):
                    prop_schema["type"] = "string"
                    prop_schema["description"] = prop_schema.get("description", "") + " (file path)"

                # tuple[T, ...] or list[T] -> array
                if "tuple" in str(prop_type) or "list" in str(prop_type):
                    prop_schema["type"] = "array"
                    # Extract item type if possible
                    if "items" in prop_schema:
                        items_schema = prop_schema["items"]
                        # Recursively sanitize items
                        if isinstance(items_schema, dict):
                            prop_schema["items"] = sanitize_schema_for_llm(
                                {"type": "object", "properties": items_schema}
                            )

                # Remove Python-specific annotations
                for key in list(prop_schema.keys()):
                    if key.startswith("_") or key in ["python_type", "annotation"]:
                        del prop_schema[key]

        # Clean up required list
        if "required" in sanitized:
            # Remove 'self' from required parameters (LLM sometimes includes it)
            sanitized["required"] = [r for r in sanitized["required"] if r != "self"]

        return sanitized

    for name, param in sig.parameters.items():
        if name == "_exec_ctx":
            continue  # Framework-internal, not exposed to LLM

        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue

        # Use the enhanced type handler for proper JSON Schema generation
        type_schema = _get_json_schema_type(param.annotation, globalns=func.__globals__)

        # Merge type schema with description, then sanitize for LLM
        raw_schema = {
            **type_schema,
            "description": param_docs.get(name, "No description."),
        }

        # Sanitize to remove Python-specific annotations that confuse LLMs
        properties[name] = sanitize_schema_for_llm(raw_schema)

        # Add explicit call syntax hint for required parameters to prevent hallucination
        # This helps LLMs understand the correct parameter format without examples in schema
        if param.default == inspect.Parameter.empty:
            call_syntax_hint = f"\n\nExpected call format: {resolved_name}({name}=...)"
            properties[name]["description"] += call_syntax_hint

        # Add default value if present (helps LLMs understand optional params)
        if param.default != inspect.Parameter.empty and param.default is not None:
            properties[name]["default"] = _sanitize_default_for_llm(param.default)

        if param.default == inspect.Parameter.empty:
            required.append(name)

    tool_params_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,  # Reject unknown/hallucinated arguments
    }

    # Capture closures
    _cost_tier = cost_tier
    _metadata_params = metadata_params
    _selection_params = selection_params
    _resolved_name = resolved_name
    _original_func_name = func.__name__

    # Build alias set: explicit aliases + original function name (if different from resolved)
    _aliases = set(explicit_aliases)
    if _original_func_name != _resolved_name:
        _aliases.add(_original_func_name)  # Auto-add original name for backward compat

    # Extract selection params with defaults
    _priority = _selection_params.get("priority", Priority.MEDIUM)
    _access_mode = _selection_params.get("access_mode", AccessMode.READONLY)
    _danger_level = _selection_params.get("danger_level", DangerLevel.SAFE)

    # Extract stages for stage-based tool selection
    _stages = _metadata_params.get("stages") or []

    # Extract new decorator-driven fields for semantic selection
    _mandatory_keywords = _metadata_params.get("mandatory_keywords") or []
    _task_types = _metadata_params.get("task_types") or []
    legacy_progress_params = _metadata_params.get("progress_params") or []
    explicit_signature_params = _metadata_params.get("signature_params") or []
    _signature_params = list(dict.fromkeys(legacy_progress_params + explicit_signature_params))
    _availability_check = _metadata_params.get("availability_check")

    # Parse execution_category from string to enum
    _exec_cat_str = _metadata_params.get("execution_category")
    _execution_category: Optional[ExecutionCategory] = None
    if _exec_cat_str:
        try:
            _execution_category = ExecutionCategory(_exec_cat_str)
        except ValueError:
            logger.warning(
                f"Invalid execution_category '{_exec_cat_str}' for tool '{resolved_name}', "
                f"using default READ_ONLY"
            )
            _execution_category = ExecutionCategory.READ_ONLY

    # Build explicit metadata if any params were provided
    _explicit_metadata: Optional[ToolMetadata] = None
    has_explicit = any(v is not None for k, v in _metadata_params.items())
    if has_explicit:
        _explicit_metadata = ToolMetadata(
            category=_metadata_params.get("category") or "",
            keywords=_metadata_params.get("keywords") or [],
            use_cases=_metadata_params.get("use_cases") or [],
            examples=_metadata_params.get("examples") or [],
            priority_hints=_metadata_params.get("priority_hints") or [],
            priority=_priority,
            access_mode=_access_mode,
            danger_level=_danger_level,
            stages=_stages,
            mandatory_keywords=_mandatory_keywords,
            task_types=_task_types,
            signature_params=_signature_params,
            execution_category=_execution_category,
        )

    # Dynamically create the tool class
    class FunctionTool(BaseTool):
        def __init__(self, fn: Callable):
            self._fn = fn
            self._name = _resolved_name  # Use resolved canonical name
            self._original_name = _original_func_name  # Keep original for debugging
            self._aliases = set(_aliases)  # Store aliases for backward compatibility
            self._description = tool_description
            self._parameters = tool_params_schema
            self._cost_tier = _cost_tier
            self._explicit_metadata = _explicit_metadata
            self._category = _metadata_params.get("category")  # Category for grouping
            self._stages = _stages  # Stage affinity for conversation state machine
            # New selection/approval metadata
            self._priority = _priority
            self._access_mode = _access_mode
            self._danger_level = _danger_level
            # New decorator-driven semantic selection fields
            self._mandatory_keywords = _mandatory_keywords
            self._task_types = _task_types
            self._signature_params = _signature_params
            self._execution_category = _execution_category or ExecutionCategory.READ_ONLY
            self._timeout_seconds = _metadata_params.get("timeout")
            # Availability check for optional tools requiring configuration
            self._availability_check = _availability_check

        @property
        def name(self) -> str:
            return self._name

        @property
        def description(self) -> str:
            return self._description

        @property
        def parameters(self) -> Dict[str, Any]:
            return self._parameters

        @property
        def cost_tier(self) -> CostTier:
            return self._cost_tier

        @property
        def aliases(self) -> Set[str]:
            """Return alias names for backward compatibility."""
            return self._aliases

        @property
        def original_name(self) -> str:
            """Return the original function name (before renaming)."""
            return self._original_name

        @property
        def is_critical(self) -> bool:
            """Return True if this tool has CRITICAL priority.

            Critical tools are always available for selection regardless of task type.
            This replaces the legacy 'is_core' property which checked category='core'.
            """
            return self._priority == Priority.CRITICAL

        @property
        def category(self) -> Optional[str]:
            """Return the tool category for semantic grouping."""
            return self._category

        @property
        def keywords(self) -> List[str]:
            """Return keywords for semantic matching from @tool decorator."""
            if self._explicit_metadata and self._explicit_metadata.keywords:
                return self._explicit_metadata.keywords
            return []

        @property
        def stages(self) -> List[str]:
            """Return stages where this tool is relevant from @tool decorator.

            Stages indicate conversation phases where this tool is most useful:
            - "initial": First interaction, exploring the request
            - "planning": Understanding scope, searching for files
            - "reading": Examining files, gathering context
            - "analysis": Reviewing code, analyzing structure
            - "execution": Making changes, running commands
            - "verification": Testing, validating changes
            - "completion": Summarizing, wrapping up
            """
            return self._stages

        @property
        def mandatory_keywords(self) -> List[str]:
            """Return mandatory keywords that force tool inclusion.

            Unlike regular keywords, mandatory keywords guarantee tool inclusion
            when matched in user requests. Useful for tools that handle
            specific operations like "show diff" or "compare files".
            """
            return self._mandatory_keywords

        @property
        def task_types(self) -> List[str]:
            """Return task types this tool is relevant for.

            Valid types: "analysis", "action", "generation", "search", "edit", "default".
            Used for classification-aware tool selection to match tools with task intent.
            """
            return self._task_types

        @property
        def signature_params(self) -> List[str]:
            """Return signature parameters for loop detection.

            Only these parameters are included in the loop detection signature.
            Pagination parameters (offset, limit, k, etc.) should be excluded
            to allow reading results in chunks without triggering false loop detection.

            E.g., ["path"] for read tool excludes offset/limit for pagination.
            """
            return self._signature_params

        @property
        def progress_params(self) -> List[str]:
            """Return legacy progress parameters used by older selection policies.

            This remains a compatibility alias for signature_params so existing
            tool metadata contracts continue to work while loop detection uses
            the canonical signature_params name internally.
            """

            return self._signature_params

        @property
        def execution_category(self) -> ExecutionCategory:
            """Return execution category for parallel execution planning.

            Categories determine which tools can safely run concurrently:
            - READ_ONLY: Pure reads, safe to parallelize
            - WRITE: File modifications, may conflict
            - COMPUTE: CPU-intensive but isolated
            - NETWORK: External calls, rate-limited
            - EXECUTE: Shell commands, may have side effects
            - MIXED: Multiple categories, careful dependency analysis needed
            """
            return self._execution_category

        @property
        def timeout_seconds(self) -> Optional[float]:
            """Return per-tool execution timeout from the decorator, if configured."""
            return self._timeout_seconds

        @property
        def timeout(self) -> Optional[float]:
            """Compatibility alias for registry/pipeline timeout lookups."""
            return self._timeout_seconds

        @property
        def priority(self) -> Priority:
            """Return tool priority for selection availability."""
            return self._priority

        @property
        def access_mode(self) -> AccessMode:
            """Return tool access mode for approval tracking."""
            return self._access_mode

        @property
        def danger_level(self) -> DangerLevel:
            """Return danger level for warning/confirmation logic."""
            return self._danger_level

        @property
        def requires_approval(self) -> bool:
            """Check if this tool requires user approval before execution."""
            return self._access_mode.requires_approval or self._danger_level.requires_confirmation

        @property
        def is_safe(self) -> bool:
            """Check if this tool is safe (readonly, no danger)."""
            return self._access_mode.is_safe and self._danger_level == DangerLevel.SAFE

        @property
        def requires_configuration(self) -> bool:
            """Check if this tool requires external configuration.

            Returns True if an availability_check was provided in the decorator,
            indicating this tool needs configuration (API keys, credentials, etc.)
            before it can be used. Examples: Slack, Teams, Jira.
            """
            return self._availability_check is not None

        def is_available(self) -> bool:
            """Check if this tool is currently available for use.

            For tools without an availability_check, always returns True.
            For tools with an availability_check (e.g., Slack, Teams), calls
            the provided function to check if the tool is properly configured.

            Returns:
                True if the tool is available, False if it requires configuration
                that hasn't been completed.
            """
            if self._availability_check is None:
                return True
            try:
                return self._availability_check()
            except Exception as e:
                logger.warning(f"Availability check for tool '{self._name}' raised exception: {e}")
                return False

        def get_warning_message(self) -> str:
            """Get appropriate warning message for this tool's danger level."""
            return self._danger_level.warning_message

        def should_include_for_task(self, task_type: str) -> bool:
            """Check if this tool should be included for the given task type."""
            return self._priority.should_include_for_task(task_type)

        def all_names(self) -> Set[str]:
            """Return all valid names (canonical + aliases)."""
            return {self._name} | self._aliases

        def matches_name(self, query_name: str) -> bool:
            """Check if this tool matches the given name (canonical or alias)."""
            return query_name == self._name or query_name in self._aliases

        @property
        def metadata(self) -> Optional[ToolMetadata]:
            """Return explicit metadata if provided via decorator."""
            return self._explicit_metadata

        def get_metadata(self) -> ToolMetadata:
            """Get semantic metadata (ToolMetadataProvider contract).

            Returns explicit metadata if provided via @tool decorator,
            otherwise auto-generates from tool properties.
            """
            if self._explicit_metadata is not None:
                return self._explicit_metadata

            # Auto-generate metadata from tool properties
            return ToolMetadata.generate_from_tool(
                name=self._name,
                description=self._description,
                parameters=self._parameters,
                cost_tier=self._cost_tier,
            )

        async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
            try:
                # DEBUG: Log tool execution start
                import logging as _logging

                _logger = _logging.getLogger(__name__)
                _logger.debug(f"[TOOL_EXEC] 🔧 Executing {self._fn.__name__}")
                _logger.debug(f"[TOOL_EXEC] Is coroutine: {inspect.iscoroutinefunction(self._fn)}")
                _logger.debug(f"[TOOL_EXEC] Tool: {self.name}, Args: {list(kwargs.keys())}")

                # Check if the target function wants the framework execution context
                # Note: We use _exec_ctx to avoid collision with tool parameters named 'context'
                sig = inspect.signature(self._fn)
                if "_exec_ctx" in sig.parameters and "_exec_ctx" not in kwargs:
                    kwargs["_exec_ctx"] = _exec_ctx

                if inspect.iscoroutinefunction(self._fn):
                    _logger.debug("[TOOL_EXEC] ⏳ Awaiting coroutine...")
                    result = await self._fn(**kwargs)
                    _logger.debug(f"[TOOL_EXEC] ✓ Coroutine awaited, result type: {type(result)}")
                else:
                    # Offload sync functions to a thread to avoid blocking
                    # the event loop during CPU-bound or I/O-bound operations.
                    _logger.debug("[TOOL_EXEC] 🔄 Running in thread...")
                    result = await asyncio.to_thread(self._fn, **kwargs)
                    _logger.debug(f"[TOOL_EXEC] ✓ Thread completed, result type: {type(result)}")

                # CRITICAL: Check if result is a coroutine (should never happen)
                if inspect.iscoroutine(result):
                    _logger.error(f"[TOOL_EXEC] ❌ RESULT IS COROUTINE: {result}")
                    _logger.error(f"[TOOL_EXEC] Tool: {self._fn.__name__}, Result: {repr(result)}")
                    raise RuntimeError(
                        f"Tool '{self.name}' returned coroutine object instead of result. "
                        f"This indicates the tool function was not properly awaited."
                    )

                # Handle dict-based error returns for backwards compatibility
                # Tools returning {"success": False, "error": "..."} should be converted
                # to proper ToolResult failures. This bridges legacy patterns.
                if isinstance(result, dict):
                    if result.get("success") is False and "error" in result:
                        metadata = result.get("metadata")
                        if not isinstance(metadata, dict):
                            metadata = {}
                        else:
                            metadata = dict(metadata)
                        for key, value in result.items():
                            if key in {"success", "error", "output", "metadata"}:
                                continue
                            metadata.setdefault(key, value)
                        return ToolResult(
                            success=False,
                            output=result,
                            error=result.get("error"),
                            metadata=metadata or None,
                        )
                    # Some tools return {"error": "..."} without success field
                    if "error" in result and "success" not in result and len(result) <= 2:
                        return ToolResult(
                            success=False,
                            output=result,
                            error=result.get("error"),
                        )

                return ToolResult(success=True, output=result)
            except (ToolValidationError, ToolExecutionError) as e:
                # Handle structured tool errors with recovery hints
                return ToolResult(
                    success=False,
                    output=None,
                    error=e.message,
                    metadata={
                        "exception": type(e).__name__,
                        "category": e.category.value,
                        "recovery_hint": e.recovery_hint,
                        "tool_name": e.tool_name,
                    },
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    output=None,
                    error=str(e),
                    metadata={"exception": type(e).__name__},
                )

    # Create the tool instance and auto-register with global metadata registry
    tool_instance = FunctionTool(func)
    _auto_register_tool(tool_instance)
    return tool_instance
