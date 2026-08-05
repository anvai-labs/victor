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

"""Tool execution middleware implementing ``MiddlewareProtocol``.

This module provides ready-made tool-pipeline middlewares for verticals:

1. LoggingMiddleware - Audit log for every tool call
2. SecretMaskingMiddleware - Mask secrets in tool results
3. MetricsMiddleware - Per-tool execution metrics + Prometheus export
4. GitSafetyMiddleware - Block dangerous git operations
5. OutputValidationMiddleware - Validate and optionally fix tool outputs

All classes implement ``MiddlewareProtocol`` (from
``victor.core.verticals.protocols.middleware``) and are processed by
``victor.agent.middleware_chain.MiddlewareChain`` during tool execution.

**Do NOT inherit from** ``victor.core.middleware.Middleware`` here. That ABC is
for HTTP/provider pipelines. Use ``MiddlewareProtocol`` structural typing only.

Example usage:
    from victor.framework.middleware import (
        LoggingMiddleware,
        SecretMaskingMiddleware,
        MetricsMiddleware,
        GitSafetyMiddleware,
        OutputValidationMiddleware,
    )

    # Add to vertical's middleware list
    middleware = [
        LoggingMiddleware(log_level=logging.DEBUG),
        SecretMaskingMiddleware(replacement="[REDACTED]"),
        MetricsMiddleware(enable_timing=True),
        GitSafetyMiddleware(block_dangerous=True),
    ]

    # Create custom validation middleware
    class JsonValidator:
        def validate(self, content, context=None):
            try:
                json_loads(content)
                return ContentValidationResult(is_valid=True)
            except JSONDecodeError as e:
                return ContentValidationResult(
                    is_valid=False,
                    issues=[ValidationIssue(message=str(e))]
                )

    json_middleware = OutputValidationMiddleware(
        validator=JsonValidator(),
        applicable_tools={"generate_json", "create_config"},
        argument_names={"content", "json_data"},
    )
"""

from __future__ import annotations

from victor.core.json_utils import json_loads
from json import JSONDecodeError
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, runtime_checkable

# FEP-0014 Phase 2a: re-export the canonical severity instead of redefining it.
from victor.core.validation import ValidationSeverity

from victor.core.vertical_types import MiddlewarePriority, MiddlewareResult
from victor.core.verticals.protocols import MiddlewareProtocol
from victor.framework.tool_naming import ToolNames, get_canonical_name

logger = logging.getLogger(__name__)


# =============================================================================
# Output Validation Types
# =============================================================================


@dataclass
class ValidationIssue:
    """A single validation issue found during content validation.

    Attributes:
        message: Human-readable description of the issue
        severity: How serious the issue is
        location: Optional location info (line, column, path, etc.)
        suggestion: Optional suggested fix
        code: Optional error/warning code for programmatic handling
    """

    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    location: Optional[str] = None
    suggestion: Optional[str] = None
    code: Optional[str] = None


@dataclass
class ContentValidationResult:
    """Result of content validation.

    Renamed from ValidationResult to be semantically distinct:
    - ToolValidationResult (victor.tools.base): Tool parameter validation
    - ConfigValidationResult (victor.core.validation): Configuration validation
    - ContentValidationResult (here): Content validation with fixed_content
    - ParameterValidationResult (victor.agent.parameter_enforcer): Parameter enforcement
    - CodeValidationResult (victor.evaluation.correction.types): Code validation

    Attributes:
        is_valid: Whether the content passed validation
        issues: List of issues found (empty if valid)
        fixed_content: Optional auto-fixed content (if validator supports fixing)
        metadata: Additional validation metadata
    """

    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    fixed_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        """Count of ERROR and CRITICAL severity issues."""
        return sum(
            1
            for issue in self.issues
            if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        )

    @property
    def warning_count(self) -> int:
        """Count of WARNING severity issues."""
        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)

    @property
    def has_fix(self) -> bool:
        """Whether a fix is available."""
        return self.fixed_content is not None


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol for content validators.

    Validators check content (code, data, configuration) for issues
    and optionally provide fixes.

    Example implementations:
    - Python linter (using ast, pylint, ruff)
    - JSON schema validator
    - YAML syntax checker
    - SQL syntax validator
    - Pydantic model validator
    """

    def validate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ContentValidationResult:
        """Validate content and return results.

        Args:
            content: The content to validate
            context: Optional context (tool_name, file_path, etc.)

        Returns:
            ContentValidationResult with validation status and any issues
        """
        ...


class FixableValidatorProtocol(ValidatorProtocol, Protocol):
    """Extended protocol for validators that can auto-fix issues.

    Inherits from ValidatorProtocol and adds the ability to
    automatically fix detected issues.
    """

    def fix(
        self,
        content: str,
        issues: List[ValidationIssue],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Attempt to fix issues in content.

        Args:
            content: Original content with issues
            issues: Issues to fix
            context: Optional context

        Returns:
            Fixed content (or original if unfixable)
        """
        ...


# =============================================================================
# Logging Middleware
# =============================================================================


class LoggingMiddleware(MiddlewareProtocol):
    """Log all tool calls for audit/debugging.

    This middleware provides comprehensive logging of tool calls including:
    - Tool name and arguments (before execution)
    - Execution result and timing (after execution)
    - Optional argument sanitization for sensitive values

    Example:
        middleware = LoggingMiddleware(
            log_level=logging.DEBUG,
            include_arguments=True,
            exclude_tools={"read"},  # Skip verbose tools
        )
    """

    def __init__(
        self,
        log_level: int = logging.DEBUG,
        include_arguments: bool = True,
        include_results: bool = False,
        sanitize_arguments: bool = True,
        exclude_tools: Optional[Set[str]] = None,
        logger_name: Optional[str] = None,
    ):
        """Initialize the logging middleware.

        Args:
            log_level: Log level to use (default: DEBUG)
            include_arguments: Include arguments in log messages
            include_results: Include results in log messages (can be verbose)
            sanitize_arguments: Sanitize sensitive argument values
            exclude_tools: Tools to exclude from logging
            logger_name: Custom logger name (default: module logger)
        """
        self._log_level = log_level
        self._include_arguments = include_arguments
        self._include_results = include_results
        self._sanitize_arguments = sanitize_arguments
        self._exclude_tools = {get_canonical_name(tool) for tool in (exclude_tools or set())}
        self._logger = logging.getLogger(logger_name) if logger_name else logger
        self._start_times: Dict[str, float] = {}

    def _sanitize_value(self, key: str, value: Any) -> Any:
        """Sanitize sensitive argument values.

        Args:
            key: Argument key name
            value: Argument value

        Returns:
            Sanitized value (or original if not sensitive)
        """
        if not self._sanitize_arguments:
            return value

        # List of sensitive keys to redact
        sensitive_keys = {
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "key",
            "credential",
            "auth",
        }

        key_lower = key.lower()
        for sensitive in sensitive_keys:
            if sensitive in key_lower:
                return "[REDACTED]"

        # Truncate very long string values
        if isinstance(value, str) and len(value) > 500:
            return f"{value[:200]}... (truncated, {len(value)} chars total)"

        return value

    def _format_arguments(self, arguments: Dict[str, Any]) -> str:
        """Format arguments for logging.

        Args:
            arguments: Tool arguments

        Returns:
            Formatted string representation
        """
        if not self._include_arguments:
            return ""

        sanitized = {k: self._sanitize_value(k, v) for k, v in arguments.items()}
        return f" args={sanitized}"

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Log tool call before execution.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult (always proceeds)
        """
        if get_canonical_name(tool_name) in self._exclude_tools:
            return MiddlewareResult()

        # Track start time for timing
        call_id = f"{tool_name}:{id(arguments)}"
        self._start_times[call_id] = time.time()

        args_str = self._format_arguments(arguments)
        self._logger.log(self._log_level, f"Tool call: {tool_name}{args_str}")

        return MiddlewareResult()

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Log tool result after execution.

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed
            result: Result from the tool execution
            success: Whether the tool execution succeeded

        Returns:
            None (no modification)
        """
        if get_canonical_name(tool_name) in self._exclude_tools:
            return None

        # Calculate duration
        call_id = f"{tool_name}:{id(arguments)}"
        start_time = self._start_times.pop(call_id, None)
        duration_ms = (time.time() - start_time) * 1000 if start_time else 0

        status = "success" if success else "failed"
        result_str = ""
        if self._include_results and result:
            result_preview = str(result)[:200]
            if len(str(result)) > 200:
                result_preview += "..."
            result_str = f" result={result_preview}"

        self._logger.log(
            self._log_level,
            f"Tool {status}: {tool_name} ({duration_ms:.1f}ms){result_str}",
        )

        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware.

        Logging runs last (after other processing).

        Returns:
            DEFERRED priority
        """
        return MiddlewarePriority.DEFERRED

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            None (applies to all tools)
        """
        return None


# =============================================================================
# Secret Masking Middleware
# =============================================================================


class SecretMaskingMiddleware(MiddlewareProtocol):
    """Mask secrets in tool results.

    This middleware automatically detects and redacts secrets, credentials,
    and sensitive data in tool output using the core safety/secrets module.

    Example:
        middleware = SecretMaskingMiddleware(
            replacement="[REDACTED]",
            mask_in_arguments=True,  # Also mask secrets in inputs
        )
    """

    def __init__(
        self,
        replacement: str = "[REDACTED]",
        mask_in_arguments: bool = False,
        include_low_severity: bool = False,
    ):
        """Initialize the secret masking middleware.

        Args:
            replacement: Text to replace secrets with
            mask_in_arguments: Also mask secrets in input arguments
            include_low_severity: Include low-severity matches
        """
        self._replacement = replacement
        self._mask_in_arguments = mask_in_arguments
        self._include_low_severity = include_low_severity

    def _mask_content(self, content: str) -> str:
        """Mask secrets in content.

        Args:
            content: Text content to scan and mask

        Returns:
            Content with secrets replaced
        """
        from victor.security.safety.secrets import mask_secrets

        return mask_secrets(content, replacement=self._replacement)

    def _mask_dict_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask secrets in dictionary values.

        Args:
            data: Dictionary to process

        Returns:
            Dictionary with masked values
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self._mask_content(value)
            elif isinstance(value, dict):
                result[key] = self._mask_dict_values(value)
            elif isinstance(value, list):
                result[key] = [self._mask_content(v) if isinstance(v, str) else v for v in value]
            else:
                result[key] = value
        return result

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Optionally mask secrets in input arguments.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult with potentially masked arguments
        """
        if not self._mask_in_arguments:
            return MiddlewareResult()

        # Mask secrets in string arguments
        masked_args = self._mask_dict_values(arguments)

        # Only return modified if there were changes
        if masked_args != arguments:
            return MiddlewareResult(
                proceed=True,
                modified_arguments=masked_args,
                metadata={"secrets_masked_in_input": True},
            )

        return MiddlewareResult()

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Mask secrets in tool results.

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed
            result: Result from the tool execution
            success: Whether the tool execution succeeded

        Returns:
            Masked result (or None if no secrets found)
        """
        if result is None:
            return None

        # Handle string results
        if isinstance(result, str):
            masked = self._mask_content(result)
            if masked != result:
                return masked
            return None

        # Handle dict results
        if isinstance(result, dict):
            masked = self._mask_dict_values(result)
            if masked != result:
                return masked
            return None

        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware.

        Secret masking runs with HIGH priority to mask secrets early.

        Returns:
            HIGH priority
        """
        return MiddlewarePriority.HIGH

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            None (applies to all tools)
        """
        return None


# =============================================================================
# Metrics Middleware
# =============================================================================


@dataclass
class ToolMetrics:
    """Metrics for a tool execution.

    Attributes:
        tool_name: Name of the tool
        call_count: Number of calls
        success_count: Number of successful calls
        failure_count: Number of failed calls
        total_duration_ms: Total execution time in milliseconds
        min_duration_ms: Minimum execution time
        max_duration_ms: Maximum execution time
    """

    tool_name: str
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration."""
        if self.call_count == 0:
            return 0.0
        return self.total_duration_ms / self.call_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count


class MetricsMiddleware(MiddlewareProtocol):
    """Record tool execution metrics.

    This middleware tracks:
    - Call counts (total, success, failure)
    - Execution timing (min, max, avg)
    - Success rates

    Metrics can be exported to various backends (Prometheus, JSON, etc.)

    Example:
        middleware = MetricsMiddleware(enable_timing=True)

        # ... execute tools ...

        # Get metrics summary
        summary = middleware.get_summary()
        for tool, metrics in summary.items():
            print(f"{tool}: {metrics.call_count} calls, "
                  f"{metrics.avg_duration_ms:.1f}ms avg")
    """

    def __init__(
        self,
        enable_timing: bool = True,
        callback: Optional[Callable[[str, ToolMetrics], None]] = None,
    ):
        """Initialize the metrics middleware.

        Args:
            enable_timing: Track execution timing
            callback: Optional callback for each recorded metric
        """
        self._enable_timing = enable_timing
        self._callback = callback
        self._metrics: Dict[str, ToolMetrics] = {}
        self._start_times: Dict[str, float] = {}

    def _get_or_create_metrics(self, tool_name: str) -> ToolMetrics:
        """Get or create metrics for a tool.

        Args:
            tool_name: Tool name

        Returns:
            ToolMetrics instance
        """
        if tool_name not in self._metrics:
            self._metrics[tool_name] = ToolMetrics(tool_name=tool_name)
        return self._metrics[tool_name]

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Record call start time.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult (always proceeds)
        """
        if self._enable_timing:
            call_id = f"{tool_name}:{id(arguments)}"
            self._start_times[call_id] = time.time()

        return MiddlewareResult()

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Record execution metrics.

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed
            result: Result from the tool execution
            success: Whether the tool execution succeeded

        Returns:
            None (no modification)
        """
        metrics = self._get_or_create_metrics(tool_name)
        metrics.call_count += 1

        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1

        # Calculate duration if timing is enabled
        if self._enable_timing:
            call_id = f"{tool_name}:{id(arguments)}"
            start_time = self._start_times.pop(call_id, None)
            if start_time:
                duration_ms = (time.time() - start_time) * 1000
                metrics.total_duration_ms += duration_ms
                metrics.min_duration_ms = min(metrics.min_duration_ms, duration_ms)
                metrics.max_duration_ms = max(metrics.max_duration_ms, duration_ms)

        # Invoke callback if provided
        if self._callback:
            try:
                self._callback(tool_name, metrics)
            except Exception:
                pass  # Don't fail on callback errors

        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware.

        Metrics runs with LOW priority (after main processing).

        Returns:
            LOW priority
        """
        return MiddlewarePriority.LOW

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            None (applies to all tools)
        """
        return None

    def get_metrics(self, tool_name: str) -> Optional[ToolMetrics]:
        """Get metrics for a specific tool.

        Args:
            tool_name: Tool name

        Returns:
            ToolMetrics or None if not recorded
        """
        return self._metrics.get(tool_name)

    def get_summary(self) -> Dict[str, ToolMetrics]:
        """Get summary of all metrics.

        Returns:
            Dict mapping tool names to their metrics
        """
        return dict(self._metrics)

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._start_times.clear()

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-compatible metrics text
        """
        lines = []
        lines.append("# HELP tool_calls_total Total number of tool calls")
        lines.append("# TYPE tool_calls_total counter")
        for name, m in self._metrics.items():
            lines.append(f'tool_calls_total{{tool="{name}"}} {m.call_count}')

        lines.append("# HELP tool_calls_success_total Successful tool calls")
        lines.append("# TYPE tool_calls_success_total counter")
        for name, m in self._metrics.items():
            lines.append(f'tool_calls_success_total{{tool="{name}"}} {m.success_count}')

        lines.append("# HELP tool_duration_ms_avg Average tool duration in ms")
        lines.append("# TYPE tool_duration_ms_avg gauge")
        for name, m in self._metrics.items():
            lines.append(f'tool_duration_ms_avg{{tool="{name}"}} {m.avg_duration_ms:.2f}')

        return "\n".join(lines)


# =============================================================================
# Git Safety Middleware
# =============================================================================


class GitSafetyMiddleware(MiddlewareProtocol):
    """Block dangerous git operations.

    This middleware validates git operations before execution and can block
    dangerous operations like force push, hard reset, or cleaning untracked files.

    Designed to be used by any vertical that works with git (Coding, DevOps, etc.)

    Example:
        middleware = GitSafetyMiddleware(
            block_dangerous=True,   # Block force push, hard reset
            warn_on_risky=True,     # Add warnings for risky operations
            allowed_force_branches={"feature/*"},  # Allow force push on feature branches
        )
    """

    # Dangerous operations that should be blocked by default
    BLOCKED_OPERATIONS: frozenset = frozenset(
        {
            "push --force",
            "push -f",
            "push --force-with-lease",  # Can still lose commits
            "reset --hard HEAD~",
            "reset --hard origin",
            "clean -fd",
            "clean -fdx",
            "reflog expire --expire=now --all",
            "gc --prune=now",
        }
    )

    # Operations that should generate warnings
    WARNED_OPERATIONS: frozenset = frozenset(
        {
            "reset --hard",
            "checkout --",
            "stash drop",
            "stash clear",
            "branch -D",
            "branch --delete --force",
            "rebase",
            "cherry-pick",
            "filter-branch",
        }
    )

    # Protected branches that should never be force-pushed to
    PROTECTED_BRANCHES: frozenset = frozenset(
        {
            "main",
            "master",
            "develop",
            "release",
            "production",
            "staging",
        }
    )

    APPLICABLE_TOOLS: frozenset[str] = frozenset(
        {
            ToolNames.GIT,
            ToolNames.SHELL,
            "execute_bash",
            "bash",
            "run_command",
        }
    )

    def __init__(
        self,
        block_dangerous: bool = True,
        warn_on_risky: bool = True,
        protected_branches: Optional[Set[str]] = None,
        allowed_force_branches: Optional[Set[str]] = None,
        custom_blocked: Optional[Set[str]] = None,
        custom_warned: Optional[Set[str]] = None,
    ):
        """Initialize git safety middleware.

        Args:
            block_dangerous: Whether to block dangerous operations
            warn_on_risky: Whether to add warnings for risky operations
            protected_branches: Additional branches to protect
            allowed_force_branches: Branch patterns where force push is allowed
            custom_blocked: Additional operations to block
            custom_warned: Additional operations to warn about
        """
        self._block_dangerous = block_dangerous
        self._warn_on_risky = warn_on_risky
        self._protected_branches = self.PROTECTED_BRANCHES | (protected_branches or set())
        self._allowed_force_branches = allowed_force_branches or set()
        self._blocked = self.BLOCKED_OPERATIONS | (custom_blocked or set())
        self._warned = self.WARNED_OPERATIONS | (custom_warned or set())

    def _is_protected_branch_operation(self, command: str) -> Optional[str]:
        """Check if command targets a protected branch.

        Args:
            command: Git command string

        Returns:
            Protected branch name if found, None otherwise
        """
        for branch in self._protected_branches:
            # Check for patterns like "push origin main --force"
            if branch in command and any(force in command for force in ["--force", "-f"]):
                return branch
        return None

    def _is_force_allowed(self, command: str) -> bool:
        """Check if force push is allowed for this command.

        Args:
            command: Git command string

        Returns:
            True if force is allowed on this branch
        """
        import fnmatch

        for pattern in self._allowed_force_branches:
            # Extract branch name from command
            parts = command.split()
            for part in parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
        return False

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Check git operations for safety.

        Args:
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            MiddlewareResult with safety check results
        """
        canonical_tool_name = get_canonical_name(tool_name)
        # Only check git-related tools
        if (
            canonical_tool_name not in {ToolNames.GIT, ToolNames.SHELL}
            and tool_name != "run_command"
        ):
            return MiddlewareResult()

        command = arguments.get("command", "") or arguments.get("args", "")
        if not command:
            return MiddlewareResult()

        # Skip if not a git command
        if "git " not in command and not command.startswith("git"):
            return MiddlewareResult()

        # Check for blocked operations
        if self._block_dangerous:
            for blocked in self._blocked:
                if blocked in command:
                    # Check if force is allowed on this branch
                    if "force" in blocked and self._is_force_allowed(command):
                        continue
                    return MiddlewareResult(
                        proceed=False,
                        error_message=f"Blocked dangerous git operation: {blocked}. "
                        "This operation can cause data loss.",
                    )

            # Check for protected branch force push
            protected = self._is_protected_branch_operation(command)
            if protected:
                return MiddlewareResult(
                    proceed=False,
                    error_message=f"Blocked force push to protected branch: {protected}. "
                    "Force pushing to main/master/develop branches is not allowed.",
                )

        # Add warnings for risky operations
        if self._warn_on_risky:
            for warned in self._warned:
                if warned in command:
                    return MiddlewareResult(
                        proceed=True,
                        metadata={
                            "git_warning": f"Risky operation: {warned}. "
                            "Ensure you have a backup or know what you're doing.",
                        },
                    )

        return MiddlewareResult()

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """No-op after tool call.

        Returns:
            None (no modification)
        """
        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get priority - CRITICAL for safety checks.

        Returns:
            CRITICAL priority (runs first)
        """
        return MiddlewarePriority.CRITICAL

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get applicable tools.

        Returns:
            Set of git-related tools
        """
        return set(self.APPLICABLE_TOOLS)


# =============================================================================
# Output Validation Middleware
# =============================================================================


class OutputValidationMiddleware(MiddlewareProtocol):
    """Generic middleware for validating and optionally fixing tool outputs.

    This middleware provides a configurable validation framework that can:
    - Validate tool arguments before execution using custom validators
    - Optionally auto-fix detected issues
    - Block execution on critical validation failures
    - Report validation metadata for downstream processing

    The middleware is domain-agnostic - validators can be implemented for:
    - Code linting (Python, JavaScript, SQL)
    - Schema validation (JSON, YAML, XML)
    - Data validation (Pydantic models, JSON Schema)
    - Configuration validation (Docker, Kubernetes, Terraform)

    Example:
        # Create a JSON validator
        class JsonValidator:
            def validate(self, content, context=None):
                import json
                try:
                    json_loads(content)
                    return ContentValidationResult(is_valid=True)
                except JSONDecodeError as e:
                    return ContentValidationResult(
                        is_valid=False,
                        issues=[ValidationIssue(
                            message=f"Invalid JSON: {e.msg}",
                            location=f"line {e.lineno}, column {e.colno}",
                        )]
                    )

        # Use in middleware
        middleware = OutputValidationMiddleware(
            validator=JsonValidator(),
            applicable_tools={"write_config", "create_json"},
            argument_names={"content", "data"},
            auto_fix=True,
            block_on_error=False,
        )
    """

    def __init__(
        self,
        validator: ValidatorProtocol,
        applicable_tools: Optional[Set[str]] = None,
        argument_names: Optional[Set[str]] = None,
        auto_fix: bool = True,
        block_on_error: bool = False,
        max_fix_iterations: int = 3,
        priority: MiddlewarePriority = MiddlewarePriority.HIGH,
    ):
        """Initialize the validation middleware.

        Args:
            validator: Validator instance implementing ValidatorProtocol
            applicable_tools: Set of tool names this middleware applies to.
                             None means apply to all tools.
            argument_names: Set of argument names to validate.
                           Default: {"content", "code", "data", "text", "body"}
            auto_fix: Whether to automatically fix issues if validator supports it
            block_on_error: Whether to block execution on validation errors
            max_fix_iterations: Maximum fix attempts before giving up
            priority: Middleware execution priority
        """
        self.validator = validator
        self._applicable_tools = applicable_tools
        self._argument_names = argument_names or {
            "content",
            "code",
            "data",
            "text",
            "body",
        }
        self.auto_fix = auto_fix
        self.block_on_error = block_on_error
        self.max_fix_iterations = max_fix_iterations
        self._priority = priority

    async def before_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MiddlewareResult:
        """Validate arguments before tool execution.

        Validates content in specified arguments and optionally applies fixes.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments

        Returns:
            MiddlewareResult with validation status and optionally modified arguments
        """
        # Find arguments to validate
        content_to_validate: Dict[str, str] = {}
        for arg_name in self._argument_names:
            if arg_name in arguments and isinstance(arguments[arg_name], str):
                content_to_validate[arg_name] = arguments[arg_name]

        if not content_to_validate:
            return MiddlewareResult()

        # Build context for validator
        context = {
            "tool_name": tool_name,
            "argument_names": list(content_to_validate.keys()),
        }

        # Validate each content field
        all_issues: List[ValidationIssue] = []
        modified_arguments: Dict[str, Any] = {}
        any_fixed = False

        for arg_name, content in content_to_validate.items():
            result = self._validate_content(content, context)

            if result.issues:
                all_issues.extend(result.issues)

            # Attempt fix if enabled and validator supports it
            if not result.is_valid and self.auto_fix and result.has_fix:
                modified_arguments[arg_name] = result.fixed_content
                any_fixed = True
                logger.debug(
                    f"OutputValidationMiddleware: Fixed issues in '{arg_name}' for {tool_name}"
                )
            elif not result.is_valid and self.auto_fix:
                # Try to fix using the FixableValidatorProtocol
                fixed = self._try_fix(content, result.issues, context)
                if fixed != content:
                    modified_arguments[arg_name] = fixed
                    any_fixed = True
                    logger.debug(
                        f"OutputValidationMiddleware: Applied fixes to '{arg_name}' for {tool_name}"
                    )

        # Build metadata
        metadata = {
            "validation_performed": True,
            "validation_passed": len(all_issues) == 0,
            "issue_count": len(all_issues),
            "error_count": sum(
                1
                for i in all_issues
                if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            ),
            "warning_count": sum(1 for i in all_issues if i.severity == ValidationSeverity.WARNING),
            "content_fixed": any_fixed,
        }

        if all_issues:
            metadata["validation_issues"] = [
                {
                    "message": i.message,
                    "severity": i.severity.value,
                    "location": i.location,
                }
                for i in all_issues[:10]  # Limit to first 10 issues
            ]

        # Determine if we should block
        has_critical = any(i.severity == ValidationSeverity.CRITICAL for i in all_issues)
        should_block = self.block_on_error and (has_critical or metadata["error_count"] > 0)

        if should_block and not any_fixed:
            error_msg = f"Validation failed: {metadata['error_count']} errors"
            if all_issues:
                error_msg += f". First issue: {all_issues[0].message}"
            return MiddlewareResult(
                proceed=False,
                error_message=error_msg,
                metadata=metadata,
            )

        # Merge modified arguments with original
        final_arguments = None
        if modified_arguments:
            final_arguments = {**arguments, **modified_arguments}

        return MiddlewareResult(
            proceed=True,
            modified_arguments=final_arguments,
            metadata=metadata,
        )

    def _validate_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ContentValidationResult:
        """Validate content using the configured validator.

        Args:
            content: Content to validate
            context: Validation context

        Returns:
            ContentValidationResult from the validator
        """
        try:
            return self.validator.validate(content, context)
        except Exception as e:
            logger.warning(f"OutputValidationMiddleware: Validator error: {e}")
            return ContentValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        message=f"Validation error: {e}",
                        severity=ValidationSeverity.ERROR,
                    )
                ],
            )

    def _try_fix(
        self,
        content: str,
        issues: List[ValidationIssue],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Attempt to fix content if validator supports fixing.

        Args:
            content: Original content
            issues: Issues to fix
            context: Fix context

        Returns:
            Fixed content (or original if unfixable)
        """
        # Check if validator implements FixableValidatorProtocol
        if not hasattr(self.validator, "fix"):
            return content

        try:
            fixed = content
            for iteration in range(self.max_fix_iterations):
                fixed = self.validator.fix(fixed, issues, context)  # type: ignore
                # Re-validate to check if fix worked
                result = self._validate_content(fixed, context)
                if result.is_valid:
                    return fixed
                issues = result.issues
            return fixed
        except Exception as e:
            logger.warning(f"OutputValidationMiddleware: Fix error: {e}")
            return content

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Post-execution hook (no-op for validation middleware).

        Returns:
            None (no modification)
        """
        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get middleware priority.

        Returns:
            Configured priority (default HIGH)
        """
        return self._priority

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get applicable tools.

        Returns:
            Set of tool names or None for all tools
        """
        return self._applicable_tools


__all__ = [
    # Middleware
    "LoggingMiddleware",
    "SecretMaskingMiddleware",
    "MetricsMiddleware",
    "GitSafetyMiddleware",
    "OutputValidationMiddleware",
    # Middleware Composer
    "MiddlewareComposer",
    # Validation Types
    "ValidationSeverity",
    "ValidationIssue",
    "ContentValidationResult",
    "ValidatorProtocol",
    "FixableValidatorProtocol",
    # Types
    "ToolMetrics",
]


# =============================================================================
# Middleware Composer
# =============================================================================


class MiddlewareComposer:
    """Helper for composing middleware lists with common patterns.

    This class provides a fluent API for building middleware lists with
    pre-configured patterns for common use cases. It eliminates
    duplication in vertical middleware configuration and ensures
    consistent middleware ordering.

    Example:
        # Using preset patterns
        middleware = (
            MiddlewareComposer()
            .git_safety(block_dangerous=False, warn_on_risky=True)
            .secret_masking(replacement="[REDACTED]")
            .logging(include_arguments=True)
            .build()
        )

        # Using custom middleware
        middleware = (
            MiddlewareComposer()
            .add(CustomMiddleware())
            .git_safety()
            .build()
        )

        # For DevOps (stricter git safety)
        middleware = (
            MiddlewareComposer()
            .git_safety(
                block_dangerous=True,
                protected_branches={"production", "staging"},
            )
            .secret_masking()
            .logging()
            .build()
        )
    """

    def __init__(self) -> None:
        """Initialize an empty composer."""
        self._middleware: List[MiddlewareProtocol] = []

    def add(self, middleware: MiddlewareProtocol) -> "MiddlewareComposer":
        """Add custom middleware to the composition.

        Args:
            middleware: Middleware instance to add

        Returns:
            Self for method chaining
        """
        self._middleware.append(middleware)
        return self

    def git_safety(
        self,
        block_dangerous: bool = False,
        warn_on_risky: bool = True,
        protected_branches: Optional[Set[str]] = None,
        allowed_force_branches: Optional[Set[str]] = None,
    ) -> "MiddlewareComposer":
        """Add GitSafetyMiddleware with configuration.

        Args:
            block_dangerous: Block dangerous git operations (force push, hard reset)
            warn_on_risky: Warn about risky operations
            protected_branches: Branches to protect from force push
            allowed_force_branches: Branch patterns where force is allowed

        Returns:
            Self for method chaining
        """
        from victor.core.vertical_types import MiddlewarePriority

        # Import here to avoid circular import
        git_middleware = GitSafetyMiddleware(
            block_dangerous=block_dangerous,
            warn_on_risky=warn_on_risky,
            protected_branches=protected_branches,
            allowed_force_branches=allowed_force_branches,
        )
        self._middleware.append(git_middleware)
        return self

    def secret_masking(
        self,
        replacement: str = "[REDACTED]",
        mask_in_arguments: bool = False,
    ) -> "MiddlewareComposer":
        """Add SecretMaskingMiddleware with configuration.

        Args:
            replacement: Text to replace secrets with
            mask_in_arguments: Also mask secrets in input arguments

        Returns:
            Self for method chaining
        """
        secret_middleware = SecretMaskingMiddleware(
            replacement=replacement,
            mask_in_arguments=mask_in_arguments,
        )
        self._middleware.append(secret_middleware)
        return self

    def logging(
        self,
        log_level: int = logging.DEBUG,
        include_arguments: bool = True,
        include_results: bool = False,
        exclude_tools: Optional[Set[str]] = None,
    ) -> "MiddlewareComposer":
        """Add LoggingMiddleware with configuration.

        Args:
            log_level: Logging level (default: DEBUG)
            include_arguments: Include arguments in logs
            include_results: Include results in logs
            exclude_tools: Tools to exclude from logging

        Returns:
            Self for method chaining
        """
        logging_middleware = LoggingMiddleware(
            log_level=log_level,
            include_arguments=include_arguments,
            include_results=include_results,
            exclude_tools=exclude_tools,
        )
        self._middleware.append(logging_middleware)
        return self

    def metrics(
        self,
        enable_timing: bool = True,
    ) -> "MiddlewareComposer":
        """Add MetricsMiddleware with configuration.

        Args:
            enable_timing: Track execution timing

        Returns:
            Self for method chaining
        """
        metrics_middleware = MetricsMiddleware(enable_timing=enable_timing)
        self._middleware.append(metrics_middleware)
        return self

    def output_validation(
        self,
        validator: "ValidatorProtocol",
        applicable_tools: Optional[Set[str]] = None,
        argument_names: Optional[Set[str]] = None,
        auto_fix: bool = True,
        block_on_error: bool = False,
    ) -> "MiddlewareComposer":
        """Add OutputValidationMiddleware with configuration.

        Args:
            validator: Validator instance
            applicable_tools: Tools this middleware applies to
            argument_names: Argument names to validate
            auto_fix: Automatically fix issues if possible
            block_on_error: Block execution on validation errors

        Returns:
            Self for method chaining
        """
        validation_middleware = OutputValidationMiddleware(
            validator=validator,
            applicable_tools=applicable_tools,
            argument_names=argument_names,
            auto_fix=auto_fix,
            block_on_error=block_on_error,
        )
        self._middleware.append(validation_middleware)
        return self

    def standard_devops(self) -> "MiddlewareComposer":
        """Add standard DevOps middleware setup.

        Includes:
        - Git safety (strict for infrastructure)
        - Secret masking (with input masking)
        - Audit logging

        Returns:
            Self for method chaining
        """
        self.git_safety(
            block_dangerous=True,
            protected_branches={"production", "staging"},
        )
        self.secret_masking(mask_in_arguments=True)
        self.logging(include_arguments=True, include_results=True)
        return self

    def standard_production(self) -> "MiddlewareComposer":
        """Add standard production middleware setup.

        Includes:
        - Secret masking
        - Metrics tracking
        - Audit logging

        Returns:
            Self for method chaining
        """
        self.secret_masking()
        self.metrics()
        self.logging(include_arguments=True, include_results=True)
        return self

    def build(self) -> List[MiddlewareProtocol]:
        """Build and return the middleware list.

        Returns:
            List of middleware instances in execution order
        """
        return list(self._middleware)

    def clear(self) -> "MiddlewareComposer":
        """Clear all middleware from the composer.

        Returns:
            Self for method chaining
        """
        self._middleware.clear()
        return self
