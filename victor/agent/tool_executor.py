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

"""Tool execution with retry logic, caching, metrics, and schema validation.

This module provides robust tool execution with:
- Pre-execution JSON schema validation (configurable strictness)
- Argument normalization and code correction
- Retry logic with exponential backoff
- Result caching for idempotent operations
- Execution metrics and statistics
"""

import asyncio
from victor.core.json_utils import json_loads
import logging
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from typing import TYPE_CHECKING

from victor.agent.debug_logger import TRACE
from victor.core.constants import DEFAULT_VERTICAL

from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.error_recovery import is_confident_path_suggestion, recover_from_error
from victor.agent.safety import SafetyChecker, get_safety_checker
from victor.storage.cache.tool_cache import ToolCache
from victor.core.errors import (
    ErrorCategory,
    ErrorHandler,
    ErrorInfo,
    ToolExecutionError,
    ToolTimeoutError,
    get_error_handler,
)
from victor.core.retry import (
    BaseRetryStrategy,
    RetryContext,
    RetryExecutor,
    tool_retry_strategy,
)
from victor.tools.base import BaseTool, ToolResult, ToolValidationResult
from victor.tools.enums import AccessMode, ExecutionCategory
from victor.tools.registry import Hook, HookError, ToolRegistry
from victor.tools.tool_names import get_canonical_name

# RL hook integration (lazy import to avoid circular dependencies)
_rl_hooks = None


def _get_rl_hooks():
    """Get RL hooks registry (lazy initialization)."""
    global _rl_hooks
    if _rl_hooks is None:
        try:
            from victor.framework.rl.hooks import get_rl_hooks

            _rl_hooks = get_rl_hooks()
        except ImportError:
            _rl_hooks = None
    return _rl_hooks


from victor.tools.metadata_registry import (
    get_idempotent_tools as registry_get_idempotent_tools,
    get_cache_invalidating_tools as registry_get_cache_invalidating_tools,
)

_DEFAULT_CACHEABLE_TOOLS = {"read", "ls", "grep", "overview", "diff"}
_READ_ONLY_PATH_RECOVERY_TOOLS = frozenset(
    {"find", "glob", "grep", "ls", "overview", "read", "read_ro"}
)
_DEFAULT_CACHE_INVALIDATING_TOOLS = {"write", "edit", "shell", "patch"}


class ValidationMode(Enum):
    """Mode for pre-execution argument validation.

    Controls how strictly the executor enforces JSON Schema validation
    before executing tools.

    Modes:
        STRICT: Validation errors block execution and return failure
        LENIENT: Validation errors are logged as warnings but execution proceeds
        OFF: No pre-execution validation (relies on tool's own validation)
    """

    STRICT = "strict"
    LENIENT = "lenient"
    OFF = "off"


if TYPE_CHECKING:
    from victor.agent.code_correction_middleware import CodeCorrectionMiddleware
    from victor.security.auth.rbac import RBACManager

logger = logging.getLogger(__name__)


class ToolExecutionResult:
    """Result of a tool execution attempt.

    Provides structured information about tool execution including success/failure,
    timing metrics, retry counts, and error details with correlation IDs for
    distributed tracing and debugging.
    """

    def __init__(
        self,
        tool_name: str,
        success: bool,
        result: Any,
        error: Optional[str] = None,
        execution_time: float = 0.0,
        cached: bool = False,
        retries: int = 0,
        normalization_strategy: Optional[NormalizationStrategy] = None,
        correlation_id: Optional[str] = None,
        error_info: Optional[ErrorInfo] = None,
        # New fields for preview support
        original_result: Optional[Any] = None,
        preview_lines: int = 3,
        was_pruned: bool = False,
        pruning_info: Any = None,
    ):
        """Initialize tool execution result.

        Args:
            tool_name: Name of the executed tool
            success: Whether execution succeeded
            result: The tool's return value
            error: Error message if execution failed
            execution_time: Time taken in seconds
            cached: Whether result was from cache
            retries: Number of retry attempts made
            normalization_strategy: Strategy used for argument normalization
            correlation_id: Unique ID for tracking this execution across logs
            error_info: Structured error information if execution failed
            original_result: Unmodified tool output (before truncation)
            preview_lines: Number of lines to show in user preview
            was_pruned: Whether user preview was truncated (LLM always receives full output)
            pruning_info: Pruning metadata for user preview (PruningInfo object)
        """
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.cached = cached
        self.retries = retries
        self.normalization_strategy = normalization_strategy
        self.correlation_id = correlation_id
        self.error_info = error_info
        self.truncation_info: Any = None
        # New fields for preview support
        self.original_result = original_result if original_result is not None else result
        self.preview_lines = preview_lines
        self.was_pruned = was_pruned
        self.pruning_info = pruning_info

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/logging.

        Returns:
            Dictionary with all execution result fields, including
            error details if present.
        """
        result = {
            "tool_name": self.tool_name,
            "success": self.success,
            "execution_time": self.execution_time,
            "cached": self.cached,
            "retries": self.retries,
            "correlation_id": self.correlation_id,
        }
        if self.error:
            result["error"] = self.error
        if self.error_info:
            result["error_details"] = self.error_info.to_dict()
        return result


class ToolExecutor:
    """Executes tools with retry logic, caching, and metrics.

    Responsibilities:
    - Execute tools with proper error handling
    - Implement retry logic with exponential backoff
    - Cache idempotent tool results
    - Normalize malformed arguments
    - Track execution metrics
    """

    @classmethod
    def is_cacheable_tool(cls, tool_name: str) -> bool:
        """Check if a tool is safe to cache (idempotent, read-only).

        Tools declare cacheability via @tool(access_mode=AccessMode.READONLY).

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool results can be cached
        """
        from victor.tools.decorators import resolve_tool_name

        canonical_name = resolve_tool_name(tool_name)
        return canonical_name in (registry_get_idempotent_tools() | _DEFAULT_CACHEABLE_TOOLS)

    @classmethod
    def is_cache_invalidating_tool(cls, tool_name: str) -> bool:
        """Check if a tool modifies state and should invalidate cache.

        Tools declare state modification via @tool(access_mode=AccessMode.WRITE/EXECUTE/MIXED).

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool modifies state
        """
        from victor.tools.decorators import resolve_tool_name

        canonical_name = resolve_tool_name(tool_name)
        return canonical_name in (
            registry_get_cache_invalidating_tools() | _DEFAULT_CACHE_INVALIDATING_TOOLS
        )

    def __init__(
        self,
        tool_registry: ToolRegistry,
        argument_normalizer: Optional[ArgumentNormalizer] = None,
        tool_cache: Optional[ToolCache] = None,
        safety_checker: Optional[SafetyChecker] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_strategy: Optional[BaseRetryStrategy] = None,
        context: Optional[Dict[str, Any]] = None,
        code_correction_middleware: Optional["CodeCorrectionMiddleware"] = None,
        enable_code_correction: bool = False,
        validation_mode: ValidationMode = ValidationMode.LENIENT,
        error_handler: Optional[ErrorHandler] = None,
        rbac_manager: Optional["RBACManager"] = None,
        current_user: Optional[str] = None,
        default_timeout_seconds: float = 60.0,
        tool_call_tracer: Optional[Any] = None,  # ToolCallTracer for debugging
    ):
        """Initialize tool executor.

        Args:
            tool_registry: Registry of available tools
            argument_normalizer: Normalizer for fixing malformed arguments
            tool_cache: Cache for idempotent tool results
            safety_checker: Checker for dangerous operations (uses global if None)
            max_retries: Maximum retry attempts for failed tools
            retry_delay: Initial delay between retries (exponential backoff)
            retry_strategy: Unified retry strategy (overrides max_retries/retry_delay)
            context: Shared context passed to all tools
            code_correction_middleware: Optional middleware for code validation/fixing
            enable_code_correction: Enable code correction for code-generating tools
            validation_mode: Pre-execution validation strictness (STRICT, LENIENT, OFF)
            error_handler: Centralized error handler for structured logging (uses global if None)
            rbac_manager: Optional RBAC manager for permission checks (disabled if None)
            current_user: Current user for RBAC checks (defaults to 'default_user')
            default_timeout_seconds: Default timeout when no tool override is set
            tool_call_tracer: Optional ToolCallTracer for debugging tool calls
        """
        self.tools = tool_registry
        self.normalizer = argument_normalizer or ArgumentNormalizer()
        self.cache = tool_cache
        self.safety_checker = safety_checker or get_safety_checker()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Use provided strategy or create one from legacy params
        self.retry_strategy = retry_strategy or tool_retry_strategy(
            max_retries=max_retries, base_delay=retry_delay
        )
        self.retry_executor = RetryExecutor(self.retry_strategy)
        self.context = context or {}
        self.code_correction_middleware = code_correction_middleware
        self.enable_code_correction = enable_code_correction
        self.validation_mode = validation_mode
        self.error_handler = error_handler or get_error_handler()
        self.rbac_manager = rbac_manager
        self.current_user = current_user or "default_user"
        self.default_timeout_seconds = float(default_timeout_seconds)
        self._tool_call_tracer = tool_call_tracer  # Tool call tracer for debugging

        # Execution statistics
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._failed_signatures: set[Tuple[str, str]] = set()
        self._max_failed_signatures = 1000
        self._validation_failures: int = 0  # Track validation failures for metrics
        self._errors_by_category: Dict[str, int] = {}  # Track errors by category

        # Session-scoped circuit breaker: tools that returned unavailable=True are
        # never retried. Populated on first unavailable result; checked before dispatch.
        self._session_disabled_tools: set[str] = set()

        # Session-scoped path redirect cache: maps a hallucinated path to the
        # PathResolver's first suggestion. Applied before dispatch to correct
        # the model when it repeats a bad path across turns.
        self._failed_path_redirects: Dict[str, str] = {}

    @property
    def session_disabled_tools(self) -> set[str]:
        """Read-only view of tools disabled for this session (returned unavailable=True)."""
        return frozenset(self._session_disabled_tools)  # type: ignore[return-value]

    def update_context(self, **kwargs: Any) -> None:
        """Update the shared context passed to tools."""
        self.context.update(kwargs)

    @staticmethod
    def _coerce_timeout_seconds(value: Any) -> Optional[float]:
        """Return a positive timeout value as float or None."""
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            return None
        if timeout > 0:
            return timeout
        return None

    def _get_tool_timeout(self, tool: BaseTool) -> float:
        """Resolve tool timeout with fallback to executor default."""
        # Priority mirrors ToolPipeline._get_tool_timeout:
        # 1) @tool(timeout=...) on function wrappers
        # 2) ToolDefinition.timeout / timeout_seconds
        for attr in ("_tool_timeout", "timeout", "timeout_seconds"):
            try:
                value = self._coerce_timeout_seconds(getattr(tool, attr, None))
            except Exception:
                continue
            if value is not None:
                return value
        return self.default_timeout_seconds

    def get_tool_function(self, tool_name: str) -> Optional[Any]:
        """Get the registered tool object for inspection and metadata lookup.

        Returns the canonicalized tool definition or tool wrapper currently
        registered under the resolved tool name, or ``None`` when unknown.
        """
        from victor.tools.decorators import resolve_tool_name

        canonical_name = resolve_tool_name(tool_name)
        return self.tools.get(canonical_name)

    def set_validation_mode(self, mode: ValidationMode) -> None:
        """Change the validation mode at runtime.

        Args:
            mode: New validation mode to use
        """
        self.validation_mode = mode
        logger.info("Validation mode changed to: %s", mode.value)

    def set_current_user(self, username: str) -> None:
        """Change the current user for RBAC checks.

        Args:
            username: New username for permission checks
        """
        self.current_user = username
        logger.info("Current user changed to: %s", username)

    def _check_rbac(
        self,
        tool: BaseTool,
        tool_name: str,
    ) -> Tuple[bool, Optional[str]]:
        """Check RBAC permissions for tool execution.

        Uses the tool's declared access_mode and category to determine
        if the current user has permission to execute it.

        Args:
            tool: The tool to check permissions for
            tool_name: Name of the tool

        Returns:
            Tuple of (allowed, denial_reason)
            - allowed: True if execution should proceed
            - denial_reason: Explanation if denied, None if allowed
        """
        # Skip RBAC if no manager configured (disabled by default)
        if self.rbac_manager is None:
            return True, None

        # Get tool metadata for permission check
        access_mode = getattr(tool, "access_mode", AccessMode.READONLY)
        category = getattr(tool, "category", "general")

        # Check access via RBAC manager
        if self.rbac_manager.check_tool_access(
            username=self.current_user,
            tool_name=tool_name,
            category=category,
            access_mode=access_mode,
        ):
            return True, None
        else:
            # Build informative denial message
            from victor.security.auth.rbac import Permission

            required_permission = Permission.from_access_mode(access_mode)
            return (
                False,
                f"RBAC denied: User '{self.current_user}' lacks '{required_permission.value}' "
                f"permission for tool '{tool_name}' (category: {category})",
            )

    def _check_missing_required_args(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
    ) -> List[str]:
        """Check for missing required arguments before execution.

        Uses the tool's JSON Schema to identify required parameters
        that were not provided, preventing TypeError crashes.

        Args:
            tool: The tool to check arguments for
            arguments: Arguments provided by the LLM

        Returns:
            List of missing required argument names (empty if all present)
        """
        schema = tool.parameters
        if not schema:
            return []

        required = schema.get("required", [])
        if not required:
            return []

        return [r for r in required if r not in arguments]

    @staticmethod
    def _coerce_arg_types(tool: BaseTool, arguments: Dict[str, Any]) -> None:
        """Best-effort type coercion based on JSON Schema property types.

        Prevents runtime TypeErrors when the LLM sends a value with the
        wrong type (e.g. string "5" for an integer parameter). Modifies
        ``arguments`` in place.

        Only handles safe, lossless conversions:
        - str → int/float (numeric strings)
        - int → float
        - str → bool ("true"/"false")
        - str → array/object (JSON-serialized): smaller / local models frequently send an
          array or object parameter as its JSON string (e.g. the ``edit`` tool's ``ops``
          arg arrived as ``'[{"type":"replace",...}]'`` in real-agent calibration, failing
          strict validation). If the string parses to the expected type, use the parsed value.
        """
        schema = tool.parameters
        if not schema:
            return

        properties = schema.get("properties", {})
        for key, value in list(arguments.items()):
            prop_schema = properties.get(key)
            if not prop_schema:
                continue

            expected = prop_schema.get("type")
            if not expected or not isinstance(expected, str):
                continue

            try:
                if expected == "integer" and isinstance(value, str):
                    arguments[key] = int(value)
                elif expected == "number" and isinstance(value, str):
                    arguments[key] = float(value)
                elif expected == "number" and isinstance(value, int):
                    arguments[key] = float(value)
                elif expected == "boolean" and isinstance(value, str):
                    if value.lower() in ("true", "1", "yes"):
                        arguments[key] = True
                    elif value.lower() in ("false", "0", "no"):
                        arguments[key] = False
                elif expected == "string" and not isinstance(value, str):
                    arguments[key] = str(value)
                elif expected in ("array", "object") and isinstance(value, str):
                    parsed = json_loads(value)
                    want_list = expected == "array"
                    if isinstance(parsed, (list, dict)) and isinstance(parsed, list) == want_list:
                        arguments[key] = parsed
            except (ValueError, TypeError):
                pass  # Leave original value; tool will handle the error

    def _check_unknown_arguments(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Check for unknown/hallucinated arguments before execution.

        This provides clear, actionable error messages when models invent
        arguments that don't exist in the tool schema.

        Args:
            tool: The tool to check arguments for
            arguments: Arguments to check

        Returns:
            Tuple of (valid, unknown_args)
            - valid: True if no unknown arguments found
            - unknown_args: List of argument names that don't exist in schema
        """
        schema = tool.parameters
        if not schema:
            return True, []

        # Get valid parameter names from schema
        properties = schema.get("properties", {})
        valid_params = set(properties.keys())

        # Find unknown arguments
        provided_params = set(arguments.keys())
        unknown = provided_params - valid_params

        if unknown:
            return False, list(unknown)
        return True, []

    def _validate_arguments(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Optional[ToolValidationResult]]:
        """Validate tool arguments against JSON Schema before execution.

        Performs pre-execution validation based on the configured validation_mode:
        - STRICT: Returns (False, result) on validation errors
        - LENIENT: Logs warnings and returns (True, result)
        - OFF: Returns (True, None) without validation

        Args:
            tool: The tool to validate arguments for
            arguments: Arguments to validate

        Returns:
            Tuple of (should_proceed, validation_result)
            - should_proceed: True if execution should continue
            - validation_result: ToolValidationResult or None if validation was skipped
        """
        effective_mode = self._get_effective_validation_mode(tool)

        if effective_mode == ValidationMode.OFF:
            return True, None

        # Coerce safe, lossless serializations (numeric strings, JSON-string arrays/objects)
        # BEFORE validating — in all modes. A model that JSON-serializes an array arg should
        # not be rejected by STRICT for a recoverable serialization (real-agent calibration
        # saw the edit tool's `ops` sent as a JSON string). ``arguments`` is the same dict
        # that flows to execution, so the coerced values are used downstream too.
        self._coerce_arg_types(tool, arguments)

        # First check for unknown/hallucinated arguments (provides clearer errors)
        valid, unknown_args = self._check_unknown_arguments(tool, arguments)
        if not valid:
            # Get valid parameters for helpful error message
            schema = tool.parameters
            valid_params = list(schema.get("properties", {}).keys()) if schema else []
            error_msg = (
                f"Unknown argument(s): {', '.join(sorted(unknown_args))}. "
                f"Valid parameters for '{tool.name}': {', '.join(sorted(valid_params)) or 'none'}"
            )
            self._validation_failures += 1
            if effective_mode == ValidationMode.STRICT:
                logger.error("Unknown arguments for '%s': %s", tool.name, unknown_args)
                return False, ToolValidationResult.failure([error_msg])
            else:
                # Strip unknown arguments so the tool function doesn't crash
                for arg in unknown_args:
                    arguments.pop(arg, None)
                logger.debug(
                    "Stripped unknown arguments for '%s': %s (lenient mode)",
                    tool.name,
                    sorted(unknown_args),
                )

        try:
            validation = tool.validate_parameters_detailed(**arguments)

            if not validation.valid:
                self._validation_failures += 1
                error_summary = "; ".join(validation.errors[:3])  # Limit to first 3 errors
                if len(validation.errors) > 3:
                    error_summary += f" (+{len(validation.errors) - 3} more)"

                if effective_mode == ValidationMode.STRICT:
                    logger.error(
                        "STRICT validation failed for '%s': %s",
                        tool.name,
                        error_summary,
                    )
                    return False, validation
                else:  # LENIENT
                    logger.warning(
                        "Validation issues for '%s' (proceeding anyway): %s",
                        tool.name,
                        error_summary,
                    )
                    return True, validation

            return True, validation

        except Exception as e:
            # Catch any exception during validation (RuntimeError, ValueError, etc)
            logger.warning("Validation error for '%s': %s", tool.name, str(e))
            # On validation system error, proceed in lenient mode, block in strict
            if effective_mode == ValidationMode.STRICT:
                return False, ToolValidationResult.failure([f"Validation system error: {e}"])
            return True, None

    def _get_effective_validation_mode(self, tool: BaseTool) -> ValidationMode:
        """Elevate validation for stateful tools even when global mode is lenient."""
        if self.validation_mode != ValidationMode.LENIENT:
            return self.validation_mode
        access_mode = getattr(tool, "access_mode", AccessMode.READONLY)
        if access_mode in {AccessMode.WRITE, AccessMode.EXECUTE, AccessMode.MIXED}:
            return ValidationMode.STRICT
        return self.validation_mode

    def _complete_tool_call(
        self,
        call_id: Optional[str],
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """Complete or fail a tool call in the tracer.

        Args:
            call_id: Tool call ID from tracer
            success: Whether the call succeeded
            result: Result data if successful
            error: Error message if failed
        """
        if not call_id or not self._tool_call_tracer:
            return

        if success:
            self._tool_call_tracer.complete_call(call_id, result=result)
        else:
            self._tool_call_tracer.fail_call(call_id, error=error)

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        skip_cache: bool = False,
        context: Optional[Dict[str, Any]] = None,
        skip_normalization: bool = False,
        parent_span_id: Optional[str] = None,  # For linking to execution spans
    ) -> ToolExecutionResult:
        """Execute a tool with retry logic and caching.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            skip_cache: Skip cache lookup even for cacheable tools
            context: Optional context to pass to the tool (merged with default context)
            skip_normalization: Skip argument normalization (use when already normalized)
            parent_span_id: Optional parent execution span ID for debugging

        Returns:
            ToolExecutionResult with execution outcome
        """
        # Resolve aliases (e.g., "search" → "code_search") before lookup
        from victor.tools.decorators import resolve_tool_name

        tool_name = resolve_tool_name(tool_name)

        # Issue 3: Circuit breaker — skip tools that are permanently unavailable.
        if tool_name in self._session_disabled_tools:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=(
                    f"Tool '{tool_name}' is unavailable this session "
                    "(a required dependency is not installed). "
                    "Check package installation or use an alternative tool."
                ),
            )

        # Record tool call start for debugging
        call_id = None
        if self._tool_call_tracer:
            call_id = self._tool_call_tracer.record_call(
                tool_name=tool_name,
                arguments=arguments,
                parent_span_id=parent_span_id or "unknown",
            )
        # Merge default context with call-specific context
        exec_context = {**self.context}
        if context:
            exec_context.update(context)
        start_time = time.time()

        # Initialize stats for this tool
        if tool_name not in self._stats:
            self._stats[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "cache_hits": 0,
                "total_time": 0.0,
                "retries": 0,
            }

        self._stats[tool_name]["calls"] += 1

        # Normalize arguments (skip if already normalized by caller)
        if skip_normalization:
            normalized_args = arguments
            strategy = None
        else:
            normalized_args, strategy = self.normalizer.normalize_arguments(arguments, tool_name)

        # Issue 1: Path redirect — correct paths the model hallucinated in a prior
        # turn and that PathResolver suggested a confident alternative for. The
        # substitution is announced in the result so the model can tell the
        # requested path from the served one.
        path_redirect_note: Optional[str] = None
        if self._failed_path_redirects:
            for _path_key in ("path", "file_path", "filename", "root"):
                _bad = str(normalized_args.get(_path_key, ""))
                if _bad and _bad in self._failed_path_redirects:
                    if os.path.exists(_bad):
                        # The path exists now (created/branch-switched since the
                        # redirect was recorded) — the redirect is stale.
                        del self._failed_path_redirects[_bad]
                        break
                    _fixed = self._failed_path_redirects[_bad]
                    normalized_args = {**normalized_args, _path_key: _fixed}
                    path_redirect_note = (
                        f"[Note: requested path '{_bad}' does not exist; serving closest "
                        f"match '{_fixed}' instead. If this is not the intended file, the "
                        f"requested file is absent from the working tree.]"
                    )
                    logger.info("Path redirect applied: '%s' → '%s'", _bad, _fixed)
                    break

        # Code correction middleware - validate and fix executable-code arguments.
        # Gated on the tool's access_mode contract (executable code only); file content is
        # never auto-corrected. Single ``process()`` entry shared with ToolPipeline.
        if self.enable_code_correction and self.code_correction_middleware is not None:
            try:
                tool_obj = self.get_tool_function(tool_name)
                normalized_args, correction_result = self.code_correction_middleware.process(
                    tool_name, normalized_args, tool=tool_obj
                )
                if correction_result is not None and not correction_result.validation.valid:
                    # Log validation errors but proceed - tool may still work
                    logger.warning(
                        "Code validation errors for tool '%s': %s",
                        tool_name,
                        list(correction_result.validation.errors),
                    )
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                logger.warning("Code correction middleware failed: %s", str(e))

        # Check cache first
        if not skip_cache and self.cache:
            cached_result = self.cache.get(tool_name, normalized_args)
            if cached_result is not None:
                self._stats[tool_name]["cache_hits"] += 1
                if path_redirect_note and isinstance(cached_result, str):
                    cached_result = f"{path_redirect_note}\n{cached_result}"
                logger.log(TRACE, "Cache hit for %s", tool_name)
                result = ToolExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    result=cached_result,
                    cached=True,
                    normalization_strategy=strategy,
                )
                self._complete_tool_call(call_id, True, result=cached_result)
                return result

        # Get the tool
        tool = self.tools.get(tool_name)
        if not tool:
            result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found",
            )
            self._complete_tool_call(call_id, False, error=result.error)
            return result

        # Check if tool is enabled
        if not self.tools.is_tool_enabled(tool_name):
            result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' is disabled",
            )
            self._complete_tool_call(call_id, False, error=result.error)
            return result

        # Check for missing required arguments before schema validation
        missing = self._check_missing_required_args(tool, normalized_args)
        if missing:
            error_msg = f"Error: Missing required arguments: {', '.join(missing)}"
            result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=error_msg,
            )
            self._complete_tool_call(call_id, False, error=result.error)
            return result

        # Pre-execution schema validation
        should_proceed, validation_result = self._validate_arguments(tool, normalized_args)
        if not should_proceed:
            error_msg = "Argument validation failed"
            if validation_result and validation_result.errors:
                error_msg = f"Invalid arguments: {'; '.join(validation_result.errors[:3])}"
            result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=error_msg,
            )
            self._complete_tool_call(call_id, False, error=result.error)
            return result

        # (Argument type coercion now runs pre-validation in _validate_arguments for all
        # modes; normalized_args is already coerced by the time we get here.)

        # Safety check for dangerous operations
        should_proceed, rejection_reason = await self.safety_checker.check_and_confirm(
            tool_name, normalized_args
        )
        if not should_proceed:
            logger.info("Tool execution blocked by safety check: %s", tool_name)
            result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=rejection_reason or "Operation cancelled by safety check",
            )
            self._complete_tool_call(call_id, False, error=result.error)
            return result

        # RBAC permission check (runs after safety check)
        rbac_allowed, rbac_denial = self._check_rbac(tool, tool_name)
        if not rbac_allowed:
            logger.warning(
                "Tool execution blocked by RBAC: %s for user %s",
                tool_name,
                self.current_user,
            )
            result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=rbac_denial or "Permission denied by RBAC",
            )
            self._complete_tool_call(call_id, False, error=result.error)
            return result

        # Execute with retry
        result, success, error, retries, error_info = await self._execute_with_retry(
            tool, normalized_args, exec_context
        )
        recovered_execution = None
        if not success:
            recovered_execution = await self._retry_with_recovered_path_if_safe(
                tool,
                normalized_args,
                exec_context,
                error,
            )
        if recovered_execution is not None:
            normalized_args, result, success, error, recovered_retries, error_info = (
                recovered_execution
            )
            retries += recovered_retries

        execution_time = time.time() - start_time
        self._stats[tool_name]["total_time"] += execution_time
        self._stats[tool_name]["retries"] += retries

        # Extract correlation ID from error_info if available
        correlation_id = error_info.correlation_id if error_info else None

        is_unavailable_result = isinstance(result, dict) and result.get("unavailable")
        if is_unavailable_result:
            self._session_disabled_tools.add(tool_name)
            logger.info("Tool '%s' session-disabled: returned unavailable=True", tool_name)

        if success:
            self._stats[tool_name]["successes"] += 1

            # Cache successful results for cacheable tools (registry + fallback)
            if self.cache and self.is_cacheable_tool(tool_name):
                self.cache.set(tool_name, normalized_args, result)

            # Invalidate cache for tools that modify state (registry + fallback)
            if self.cache and self.is_cache_invalidating_tool(tool_name):
                self._invalidate_cache_for_write_tool(tool_name, normalized_args)

            # Announce a path substitution after caching so the cache holds the
            # clean result keyed by the served path.
            if path_redirect_note and isinstance(result, str):
                result = f"{path_redirect_note}\n{result}"

        else:
            self._stats[tool_name]["failures"] += 1
            # Track failed signature to avoid retrying same failure
            sig = (tool_name, str(sorted(normalized_args.items())))
            if len(self._failed_signatures) < self._max_failed_signatures:
                self._failed_signatures.add(sig)

            # Issue 1: Record PathResolver suggestion for future path rewriting.
            # Filesystem tools embed "Did you mean...\n  - /path" in FileNotFoundError.
            # Only typo-class (high basename similarity) suggestions are recorded;
            # a weak fuzzy match must surface as an honest not-found error, not a
            # silent substitute (session proximaDB-5b2726a3 spiral).
            if self._is_safe_read_only_path_retry(tool, normalized_args, error):
                import re as _re

                for _path_key in ("path", "file_path", "filename", "root"):
                    _bad = str(normalized_args.get(_path_key, ""))
                    if _bad:
                        _bad_norm = _bad.rstrip("/").replace("\\", "/")
                        for _suggested in _re.findall(r"-\s+(\S+)", error):
                            if _suggested.rstrip("/").replace(
                                "\\", "/"
                            ) != _bad_norm and is_confident_path_suggestion(_bad, _suggested):
                                self._failed_path_redirects[_bad] = _suggested
                                logger.info(
                                    "Path suggestion recorded: '%s' → '%s'",
                                    _bad,
                                    _suggested,
                                )
                                break
                        break

        # Emit RL event for tool execution (for learner activation)
        self._emit_rl_tool_event(tool_name, success, execution_time, exec_context, result=result)

        # Complete tool call in tracer
        self._complete_tool_call(call_id, success, result=result, error=error)

        # Hard safety cap — only fires for extreme outliers (binary data, runaway output).
        # Normal tool outputs (code files, grep, shell) stay well below 500 KB.
        # LLM accuracy depends on receiving complete output; context budget management is
        # handled upstream by TurnBoundaryContextAssembler, not here.
        _HARD_OUTPUT_BYTE_LIMIT = 500_000
        truncation_info = None
        if success and isinstance(result, str) and len(result) > _HARD_OUTPUT_BYTE_LIMIT:
            from victor.tools.output_utils import truncate_by_lines

            result, truncation_info = truncate_by_lines(result)
            if truncation_info.was_truncated:
                logger.warning(
                    "Tool %s output exceeded hard safety limit (%d bytes): %d→%d lines (%s)",
                    tool_name,
                    _HARD_OUTPUT_BYTE_LIMIT,
                    truncation_info.total_lines,
                    truncation_info.lines_returned,
                    truncation_info.truncation_reason,
                )
                remaining = truncation_info.total_lines - truncation_info.lines_returned
                result = (
                    result
                    + f"\n[OUTPUT TRUNCATED: {remaining} lines omitted (output exceeded 500 KB hard limit). "
                    f"Use offset={truncation_info.lines_returned} to read remaining content.]"
                )

        # Store original_result for user preview regardless of later formatting-stage pruning.
        # Pruning now happens in the tool-result post-processing path after structured outputs
        # have been serialized for LLM injection, which avoids collapsing dict/list tool results
        # into strings too early.
        original_result = result if isinstance(result, str) else str(result)

        exec_result = ToolExecutionResult(
            tool_name=tool_name,
            success=success,
            result=result,
            error=error,
            execution_time=execution_time,
            retries=retries,
            normalization_strategy=strategy,
            correlation_id=correlation_id,
            error_info=error_info,
            # New fields for preview support
            original_result=original_result,
            preview_lines=3,  # Will be overridden by formatter settings
            was_pruned=False,
            pruning_info=None,
        )
        exec_result.truncation_info = truncation_info
        return exec_result

    def _run_before_hooks(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Run before hooks for a tool execution.

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments

        Raises:
            HookError: If a critical hook fails
        """
        hooks = getattr(self.tools, "_before_hooks", [])
        for before_hook in hooks:
            hook_obj = before_hook if isinstance(before_hook, Hook) else None
            hook_name = hook_obj.name if hook_obj else getattr(before_hook, "__name__", "hook")
            is_critical = hook_obj.critical if hook_obj else False
            try:
                before_hook(tool_name, arguments)
            except Exception as e:
                if is_critical:
                    logger.error("Critical before hook '%s' failed: %s", hook_name, str(e))
                    raise HookError(hook_name, e, tool_name) from e
                else:
                    logger.warning("Before hook '%s' failed (non-critical): %s", hook_name, str(e))

    def _run_after_hooks(self, tool_name: str, result: Any) -> None:
        """Run after hooks for a tool execution.

        Args:
            tool_name: Name of the tool that was executed
            result: Tool execution result

        Raises:
            HookError: If a critical hook fails
        """
        hooks = getattr(self.tools, "_after_hooks", [])
        for after_hook in hooks:
            hook_obj = after_hook if isinstance(after_hook, Hook) else None
            hook_name = hook_obj.name if hook_obj else getattr(after_hook, "__name__", "hook")
            is_critical = hook_obj.critical if hook_obj else False
            try:
                after_hook(result)
            except Exception as e:
                if is_critical:
                    logger.error("Critical after hook '%s' failed: %s", hook_name, str(e))
                    raise HookError(hook_name, e, tool_name) from e
                else:
                    logger.warning("After hook '%s' failed (non-critical): %s", hook_name, str(e))

    def _emit_rl_tool_event(
        self,
        tool_name: str,
        success: bool,
        execution_time: float,
        context: Dict[str, Any],
        result: Any = None,
    ) -> None:
        """Emit RL event for tool execution to activate tool_selector learner.

        This is called after every tool execution to provide feedback to the
        RL system for learning optimal tool selection.

        Args:
            tool_name: Name of the tool that was executed
            success: Whether the tool execution succeeded
            execution_time: Time taken to execute the tool (seconds)
            context: Execution context with provider, model, task info
        """
        try:
            rl_hooks = _get_rl_hooks()
            if rl_hooks is None:
                return

            from victor.framework.rl.hooks import RLEvent, RLEventType

            # Truthful RL signal (P7): unified tools report failures as
            # '### ❌' markdown strings with invocation-level success=True —
            # recording those as successes hid the real error rate from
            # the Q-values.
            from victor.tools.unified.parser import classify_result_marker

            if success and classify_result_marker(result) == "tool_error":
                success = False

            # Calculate quality score based on success and execution time
            # Fast successful executions get higher scores
            base_quality = 0.9 if success else 0.3
            time_penalty = min(execution_time / 30.0, 0.3)  # Penalize slow executions
            quality_score = base_quality - time_penalty

            # Extract provider name - handle both string and Provider object
            provider = context.get("provider")
            if provider is not None and not isinstance(provider, str):
                # Provider is an object - get its name attribute or class name
                provider = (
                    getattr(provider, "name", None)
                    or provider.__class__.__name__.replace("Provider", "").lower()
                )

            # Extract model name - handle both string and object
            model = context.get("model")
            if model is not None and not isinstance(model, str):
                model = str(model)

            event = RLEvent(
                type=RLEventType.TOOL_EXECUTED,
                tool_name=tool_name,
                success=success,
                quality_score=quality_score,
                provider=provider,
                model=model,
                task_type=context.get("task_type", "general"),
                vertical=context.get("vertical", DEFAULT_VERTICAL),
                metadata={
                    "execution_time": execution_time,
                },
            )

            rl_hooks.emit(event)

        except Exception as e:
            # RL hook failure should never block tool execution
            logger.debug("RL tool event emission failed: %s", str(e))

    async def _execute_with_retry(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Any, bool, Optional[str], int, Optional[ErrorInfo]]:
        """Execute a tool with retry logic from unified BaseRetryStrategy.

        Uses the configured retry strategy for exponential backoff and
        retry decisions. Hooks are run before/after each attempt. Errors
        are tracked via the centralized ErrorHandler for structured logging.

        Args:
            tool: Tool to execute
            arguments: Tool arguments
            context: Context to pass to the tool

        Returns:
            Tuple of (result, success, error_message, retry_count, error_info)
            error_info is only populated on failure for structured error tracking
        """
        retry_context = RetryContext(
            max_attempts=getattr(self.retry_strategy, "max_attempts", self.max_retries)
        )
        last_error_info: Optional[ErrorInfo] = None

        while True:
            retry_context.attempt += 1

            try:
                # Run before hooks - critical hooks can block execution
                self._run_before_hooks(tool.name, arguments)

                # Execute the tool — strip _exec_ctx if LLM hallucinated it in arguments
                arguments.pop("_exec_ctx", None)
                # Per-tool timeout: tool-level override then executor default
                per_attempt_timeout = self._get_tool_timeout(tool)
                result = await asyncio.wait_for(
                    tool.execute(_exec_ctx=context, **arguments),
                    timeout=per_attempt_timeout,
                )

                # Run after hooks - critical hooks can raise errors
                self._run_after_hooks(tool.name, result)

                # Handle ToolResult
                if isinstance(result, ToolResult):
                    if result.success:
                        self.retry_strategy.on_success(retry_context)
                        return (
                            result.output,
                            True,
                            None,
                            retry_context.attempt - 1,
                            None,
                        )
                    else:
                        # Don't retry if tool explicitly returned failure
                        error = result.error or "Tool returned failure"
                        # Create structured error info for tool failures
                        error_info = self.error_handler.handle(
                            ToolExecutionError(
                                error,
                                tool_name=tool.name,
                                recovery_hint="Check tool arguments and try again.",
                            ),
                            context={"tool": tool.name, "arguments": arguments},
                        )
                        self._track_error_category(error_info.category)
                        return (
                            result.output,
                            False,
                            error,
                            retry_context.attempt - 1,
                            error_info,
                        )
                else:
                    # Raw result (for tools that don't return ToolResult)
                    self.retry_strategy.on_success(retry_context)
                    return result, True, None, retry_context.attempt - 1, None

            except asyncio.TimeoutError as e:
                # Handle timeout specifically
                timeout_error = ToolTimeoutError(tool_name=tool.name)
                last_error_info = self.error_handler.handle(
                    timeout_error,
                    context={
                        "tool": tool.name,
                        "attempt": retry_context.attempt,
                        "arguments": arguments,
                    },
                )
                self._track_error_category(last_error_info.category)
                retry_context.record_exception(e)

                if self.retry_strategy.should_retry(retry_context):
                    self.retry_strategy.on_retry(retry_context)
                    delay = self.retry_strategy.get_delay(retry_context)
                    retry_context.record_delay(delay)
                    if delay > 0:
                        await asyncio.sleep(delay)
                else:
                    self.retry_strategy.on_failure(retry_context)
                    return (
                        None,
                        False,
                        str(timeout_error),
                        retry_context.attempt - 1,
                        last_error_info,
                    )

            except (TimeoutError, asyncio.TimeoutError) as timeout_error:
                retry_context.record_exception(timeout_error)

                # Use centralized error handler for structured logging
                last_error_info = self.error_handler.handle(
                    timeout_error,
                    context={
                        "tool": tool.name,
                        "attempt": retry_context.attempt,
                        "max_attempts": retry_context.max_attempts,
                        "arguments": arguments,
                    },
                )
                self._track_error_category(last_error_info.category)

                if self.retry_strategy.should_retry(retry_context):
                    self.retry_strategy.on_retry(retry_context)
                    delay = self.retry_strategy.get_delay(retry_context)
                    logger.warning(
                        "[%s] Tool %s timeout - retrying in %.2fs " "(attempt %d/%d): %s",
                        last_error_info.correlation_id,
                        tool.name,
                        delay,
                        retry_context.attempt,
                        retry_context.max_attempts,
                        str(timeout_error),
                    )
                    await asyncio.sleep(delay)
                    continue

                return (
                    None,
                    False,
                    str(timeout_error),
                    retry_context.attempt - 1,
                    last_error_info,
                )

            except (
                ToolExecutionError,
                ValueError,
                TypeError,
                KeyError,
                FileNotFoundError,
                OSError,
            ) as e:
                retry_context.record_exception(e)

                # Use centralized error handler for structured logging
                last_error_info = self.error_handler.handle(
                    e,
                    context={
                        "tool": tool.name,
                        "attempt": retry_context.attempt,
                        "max_attempts": retry_context.max_attempts,
                        "arguments": arguments,
                    },
                )
                self._track_error_category(last_error_info.category)

                logger.warning(
                    "[%s] Tool %s failed (attempt %d/%d): %s",
                    last_error_info.correlation_id,
                    tool.name,
                    retry_context.attempt,
                    retry_context.max_attempts,
                    str(e),
                )

                if self.retry_strategy.should_retry(retry_context):
                    self.retry_strategy.on_retry(retry_context)
                    delay = self.retry_strategy.get_delay(retry_context)
                    retry_context.record_delay(delay)

                    if delay > 0:
                        await asyncio.sleep(delay)
                else:
                    self.retry_strategy.on_failure(retry_context)
                    # Include recovery hint in error message
                    error_msg = str(e)
                    if last_error_info.recovery_hint:
                        error_msg = f"{e}\nRecovery hint: {last_error_info.recovery_hint}"
                    return (
                        None,
                        False,
                        error_msg,
                        retry_context.attempt - 1,
                        last_error_info,
                    )

    def _track_error_category(self, category: ErrorCategory) -> None:
        """Track error occurrences by category for metrics.

        Args:
            category: The error category to track
        """
        key = category.value
        self._errors_by_category[key] = self._errors_by_category.get(key, 0) + 1

    def _is_safe_read_only_path_retry(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        error: Optional[str],
    ) -> bool:
        """Check whether a failed call can be safely retried with a suggested path."""
        if not error:
            return False

        if not any(key in arguments for key in ("path", "file_path", "file", "root")):
            return False

        error_text = error.lower()
        if not any(
            token in error_text
            for token in (
                "did you mean",
                "directory not found",
                "file not found",
                "no such file",
            )
        ):
            return False

        canonical_name = get_canonical_name(tool.name)
        if canonical_name in _READ_ONLY_PATH_RECOVERY_TOOLS:
            return True

        execution_category = getattr(tool, "execution_category", None)
        if execution_category is None:
            execution_category = getattr(
                getattr(tool, "metadata", None), "execution_category", None
            )
        return execution_category == ExecutionCategory.READ_ONLY

    async def _retry_with_recovered_path_if_safe(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        context: Dict[str, Any],
        error: Optional[str],
    ) -> Optional[Tuple[Dict[str, Any], Any, bool, Optional[str], int, Optional[ErrorInfo]]]:
        """Retry a read-only path tool once with a recovered path suggestion."""
        if not self._is_safe_read_only_path_retry(tool, arguments, error):
            return None

        recovery = recover_from_error(
            FileNotFoundError(error or "Path not found"),
            tool.name,
            arguments,
            context,
        )
        if not recovery.should_retry or not recovery.modified_args:
            return None

        recovered_args = dict(recovery.modified_args)
        path_key = next(
            (
                key
                for key in ("path", "file_path", "file", "root")
                if key in arguments and key in recovered_args
            ),
            None,
        )
        if path_key is None:
            return None

        original_path = str(arguments.get(path_key, ""))
        recovered_path = str(recovered_args.get(path_key, ""))
        if not recovered_path or recovered_path == original_path:
            return None

        self._failed_path_redirects[original_path] = recovered_path
        logger.info(
            "Retrying read-only tool '%s' with recovered path '%s' -> '%s'",
            tool.name,
            original_path,
            recovered_path,
        )

        result, success, recovered_error, retries, error_info = await self._execute_with_retry(
            tool,
            recovered_args,
            context,
        )
        if success and isinstance(result, str):
            result = (
                f"[Note: requested path '{original_path}' does not exist; serving closest "
                f"match '{recovered_path}' instead. If this is not the intended file, the "
                f"requested file is absent from the working tree.]\n{result}"
            )
        return recovered_args, result, success, recovered_error, retries + 1, error_info

    def has_failed_before(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Check if this exact tool call has failed before."""
        sig = (tool_name, str(sorted(arguments.items())))
        return sig in self._failed_signatures

    def clear_failed_signatures(self) -> None:
        """Clear the record of failed tool calls."""
        self._failed_signatures.clear()

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics for all tools.

        Returns:
            Dictionary with per-tool stats and global metrics including
            validation failures and error category breakdown.
        """
        stats = self._stats.copy()
        stats["_global"] = {
            "validation_failures": self._validation_failures,
            "validation_mode": self.validation_mode.value,
            "errors_by_category": self._errors_by_category.copy(),
            "recent_errors": [e.to_dict() for e in self.error_handler.get_recent_errors(5)],
        }
        return stats

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of recent errors for debugging.

        Returns:
            Dictionary with error counts by category and recent error details.
        """
        return {
            "errors_by_category": self._errors_by_category.copy(),
            "total_errors": sum(self._errors_by_category.values()),
            "recent_errors": [
                {
                    "correlation_id": e.correlation_id,
                    "category": e.category.value,
                    "message": e.message,
                    "recovery_hint": e.recovery_hint,
                }
                for e in self.error_handler.get_recent_errors(10)
            ],
        }

    def get_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        """Get execution statistics for a specific tool."""
        return self._stats.get(tool_name, {}).copy()

    def reset_stats(self) -> None:
        """Reset all execution statistics."""
        self._stats.clear()

    def invalidate_cache_for_paths(self, paths: List[str]) -> None:
        """Invalidate cache entries for modified paths.

        Called when files are modified to ensure stale cache is cleared.

        Args:
            paths: List of file paths that were modified
        """
        if self.cache:
            self.cache.invalidate_paths(paths)

    def clear_cache(self) -> None:
        """Clear the entire tool cache."""
        if self.cache:
            self.cache.clear_all()

    def _invalidate_cache_for_write_tool(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Invalidate cache entries affected by write operations.

        Args:
            tool_name: Name of the write tool that was executed
            arguments: Arguments passed to the tool
        """
        if not self.cache:
            return

        canonical_tool_name = get_canonical_name(tool_name)

        # Extract paths from arguments based on tool type
        paths_to_invalidate: List[str] = []

        if canonical_tool_name == "write":
            file_path = arguments.get("path") or arguments.get("file_path")
            if file_path:
                paths_to_invalidate.append(file_path)
        elif canonical_tool_name == "edit":
            ops = arguments.get("ops") or arguments.get("edits") or arguments.get("files")
            if isinstance(ops, list):
                for op in ops:
                    if not isinstance(op, dict):
                        continue
                    path = op.get("path")
                    new_path = op.get("new_path")
                    if path:
                        paths_to_invalidate.append(path)
                    if new_path:
                        paths_to_invalidate.append(new_path)
            elif "path" in arguments:
                paths_to_invalidate.append(arguments["path"])
        elif canonical_tool_name == "shell":
            # Bash commands can modify anything - invalidate all file-related caches
            self.cache.invalidate_by_tool("read")
            self.cache.invalidate_by_tool("ls")
            return
        elif canonical_tool_name in ("git", "docker"):
            # Git and docker operations can have wide-reaching effects
            self.cache.invalidate_by_tool("read")
            self.cache.invalidate_by_tool("ls")
            self.cache.invalidate_by_tool("code_search")
            return

        if paths_to_invalidate:
            unique_paths = list(dict.fromkeys(paths_to_invalidate))
            self.cache.invalidate_paths(unique_paths)
