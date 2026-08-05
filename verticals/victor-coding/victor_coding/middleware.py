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

"""Coding-specific middleware for tool execution.

This module provides middleware implementations for the coding vertical,
including code validation, syntax checking, and auto-correction.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Set

from victor_contracts.verticals import (
    MiddlewarePriority,
    MiddlewareProtocol,
    MiddlewareResult,
)
from victor_contracts.middleware_runtime import (
    GitSafetyMiddleware as _FrameworkGitSafety,
)

logger = logging.getLogger(__name__)


# Tools that write or execute code
CODE_TOOLS: frozenset[str] = frozenset(
    {
        "code_executor",
        "execute_code",
        "run_code",
        "write_file",
        "file_editor",
        "edit_file",
        "edit_files",
        "create_file",
    }
)

# Argument names that contain code
CODE_ARGUMENT_NAMES: frozenset[str] = frozenset(
    {
        "code",
        "python_code",
        "content",
        "source",
        "script",
        "new_content",
        "file_content",
    }
)


class CodingMiddleware(MiddlewareProtocol):
    """Base middleware for coding-related tool calls.

    Provides common functionality for code validation and processing.
    Can be extended for specific coding operations.
    """

    def __init__(
        self,
        enabled: bool = True,
        applicable_tools: Optional[Set[str]] = None,
    ):
        """Initialize the middleware.

        Args:
            enabled: Whether the middleware is enabled
            applicable_tools: Set of tool names this applies to (None for all)
        """
        self._enabled = enabled
        self._applicable_tools = applicable_tools

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Called before a tool is executed.

        Base implementation does nothing. Override in subclasses.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult indicating whether to proceed
        """
        return MiddlewareResult()

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Called after a tool is executed.

        Base implementation returns None (no modification).

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed
            result: Result from the tool execution
            success: Whether the tool execution succeeded

        Returns:
            Modified result (or None to keep original)
        """
        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware.

        Returns:
            Priority level for execution ordering
        """
        return MiddlewarePriority.NORMAL

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            Set of tool names, or None for all tools
        """
        return self._applicable_tools


class CodeCorrectionMiddleware(MiddlewareProtocol):
    """Middleware for validating and correcting code in tool arguments.

    Integrates with the existing code correction system to provide
    automatic validation and fixing of code before execution.

    This middleware is specific to the coding vertical and wraps
    the framework's CodeCorrectionMiddleware for protocol compliance.

    Also integrates with RL system to record tool success/failure
    for adaptive learning.
    """

    def __init__(
        self,
        enabled: bool = True,
        auto_fix: bool = True,
        max_iterations: int = 1,
        enable_rl: bool = True,
    ):
        """Initialize the middleware.

        Args:
            enabled: Whether correction is enabled
            auto_fix: Whether to automatically fix detected issues
            max_iterations: Maximum correction iterations
            enable_rl: Whether to record RL training data
        """
        self._enabled = enabled
        self._auto_fix = auto_fix
        self._max_iterations = max_iterations
        self._enable_rl = enable_rl
        self._inner_middleware = None
        self._rl_hooks = None
        # Track timing for RL
        self._tool_start_times: Dict[str, float] = {}

    def _get_inner(self):
        """Lazy-load the inner middleware."""
        if self._inner_middleware is None:
            try:
                from victor_contracts.middleware_runtime import (
                    CodeCorrectionConfig,
                    CodeCorrectionMiddleware as InnerMiddleware,
                )

                config = CodeCorrectionConfig(
                    enabled=self._enabled,
                    auto_fix=self._auto_fix,
                    max_iterations=self._max_iterations,
                )
                self._inner_middleware = InnerMiddleware(config=config)
            except ImportError:
                logger.warning("Code correction middleware not available")
                return None
        return self._inner_middleware

    def _get_rl_hooks(self):
        """Lazy-load the RL hooks."""
        if self._rl_hooks is None and self._enable_rl:
            try:
                from victor_coding.rl import CodingRLHooks

                self._rl_hooks = CodingRLHooks()
            except ImportError:
                logger.debug("RL hooks not available")
                return None
        return self._rl_hooks

    def should_validate(self, tool_name: str) -> bool:
        """Check if this tool should have its code validated.

        Delegates to the inner middleware.

        Args:
            tool_name: Name of the tool

        Returns:
            True if code validation should be applied
        """
        inner = self._get_inner()
        if inner is None:
            return False
        return inner.should_validate(tool_name)

    def validate_and_fix(self, tool_name: str, arguments: Dict[str, Any]):
        """Validate and optionally fix code in arguments.

        Delegates to the inner middleware.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            CorrectionResult from inner middleware
        """
        inner = self._get_inner()
        if inner is None:
            # Return a minimal "no issues" result
            from victor_contracts.middleware_runtime import CorrectionResult
            from victor_contracts.middleware_runtime import (
                CodeValidationResult,
                Language,
            )

            return CorrectionResult(
                original_code="",
                corrected_code="",
                validation=CodeValidationResult(
                    valid=True,
                    language=Language.UNKNOWN,
                    syntax_valid=True,
                    imports_valid=True,
                    errors=(),
                    warnings=(),
                ),
                was_corrected=False,
            )
        return inner.validate_and_fix(tool_name, arguments)

    def apply_correction(self, arguments: Dict[str, Any], correction_result):
        """Apply a correction to the arguments.

        Delegates to the inner middleware.

        Args:
            arguments: Original arguments
            correction_result: Result from validate_and_fix

        Returns:
            Modified arguments with corrections applied
        """
        inner = self._get_inner()
        if inner is None:
            return arguments
        return inner.apply_correction(arguments, correction_result)

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Validate and optionally fix code before tool execution.

        Also tracks timing for RL recording.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult with validation status and any corrections
        """
        # Track start time for RL
        if self._enable_rl:
            self._tool_start_times[tool_name] = time.time()

        if not self._enabled:
            return MiddlewareResult()

        inner = self._get_inner()
        if inner is None or not inner.should_validate(tool_name):
            return MiddlewareResult()

        try:
            result = inner.validate_and_fix(tool_name, arguments)

            if result.was_corrected:
                # Apply the correction
                modified = inner.apply_correction(arguments, result)
                logger.debug("Code corrected for tool %s", tool_name)
                return MiddlewareResult(
                    proceed=True,
                    modified_arguments=modified,
                    metadata={"code_corrected": True},
                )

            if not result.validation.valid:
                # Code is invalid and couldn't be fixed
                error_msg = inner.format_validation_error(result)
                logger.warning("Code validation failed for %s: %s", tool_name, error_msg)
                return MiddlewareResult(
                    proceed=True,  # Still proceed, but with metadata
                    metadata={
                        "code_validation_failed": True,
                        "validation_error": error_msg,
                    },
                )

        except Exception as e:
            logger.error("Code correction middleware error: %s", e)
            # Don't block on middleware errors

        return MiddlewareResult()

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Called after tool execution.

        Records RL outcomes for tool success/failure to enable
        adaptive tool selection learning.

        Args:
            tool_name: Name of the tool
            arguments: Arguments passed
            result: Tool result
            success: Whether execution succeeded
            context: Optional execution context with provider/model info

        Returns:
            None (no modification)
        """
        # Record RL outcome
        if self._enable_rl:
            rl_hooks = self._get_rl_hooks()
            if rl_hooks is not None:
                try:
                    # Calculate duration
                    start_time = self._tool_start_times.pop(tool_name, None)
                    duration_ms = (time.time() - start_time) * 1000 if start_time else 0

                    # Get context info
                    ctx = context or {}
                    task_type = ctx.get("task_type", "general")
                    provider = ctx.get("provider", "unknown")
                    model = ctx.get("model", "unknown")

                    if success:
                        rl_hooks.on_tool_success(
                            tool_name=tool_name,
                            task_type=task_type,
                            provider=provider,
                            model=model,
                            duration_ms=duration_ms,
                            context=ctx,
                        )
                    else:
                        error_msg = str(result) if result else "Unknown error"
                        rl_hooks.on_tool_failure(
                            tool_name=tool_name,
                            task_type=task_type,
                            provider=provider,
                            model=model,
                            error=error_msg,
                        )
                except Exception as e:
                    logger.debug("Error recording RL outcome: %s", e)

        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority - HIGH for code validation.

        Returns:
            HIGH priority to run before most other middleware
        """
        return MiddlewarePriority.HIGH

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            Set of code-related tools
        """
        return set(CODE_TOOLS)


class GitSafetyMiddleware(_FrameworkGitSafety):
    """Coding-specific defaults: block_dangerous=False.

    Thin subclass of the framework GitSafetyMiddleware that preserves
    the coding vertical's default of block_dangerous=False (the framework
    default is True).
    """

    def __init__(self, block_dangerous: bool = False, warn_on_risky: bool = True, **kwargs):
        super().__init__(block_dangerous=block_dangerous, warn_on_risky=warn_on_risky, **kwargs)


__all__ = [
    "CodingMiddleware",
    "CodeCorrectionMiddleware",
    "GitSafetyMiddleware",
    "CODE_TOOLS",
    "CODE_ARGUMENT_NAMES",
]
