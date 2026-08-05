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

"""Tests for victor.framework.middleware module.

These tests verify the framework-level middleware implementations:
- LoggingMiddleware
- SecretMaskingMiddleware
- MetricsMiddleware
- GitSafetyMiddleware
"""

import logging
import pytest
from unittest.mock import MagicMock, patch

from victor.core.verticals.import_resolver import import_module_with_fallback
from victor.core.vertical_types import MiddlewarePriority
from victor.framework.middleware import (
    GitSafetyMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    OutputValidationMiddleware,
    SecretMaskingMiddleware,
    ToolMetrics,
    ValidationIssue,
    ContentValidationResult,
    ValidationSeverity,
    ValidatorProtocol,
)


def _load_vertical_attr(module_path: str, attr_name: str):
    """Load a vertical attribute using external-first import fallbacks."""
    module, _resolved = import_module_with_fallback(module_path)
    if module is None or not hasattr(module, attr_name):
        raise ImportError(f"Unable to resolve {module_path}:{attr_name}")
    return getattr(module, attr_name)


# =============================================================================
# LoggingMiddleware Tests
# =============================================================================


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create a logging middleware instance."""
        return LoggingMiddleware(log_level=logging.DEBUG)

    @pytest.mark.asyncio
    async def test_before_tool_call_logs_message(self, middleware, caplog):
        """LoggingMiddleware should log before tool call."""
        with caplog.at_level(logging.DEBUG):
            result = await middleware.before_tool_call("test_tool", {"arg1": "value1"})

        assert result.proceed is True
        assert "Tool call: test_tool" in caplog.text

    @pytest.mark.asyncio
    async def test_after_tool_call_logs_success(self, middleware, caplog):
        """LoggingMiddleware should log after successful tool call."""
        # Use the same arguments dict so call_id matches between before/after
        args = {"arg1": "value1"}
        await middleware.before_tool_call("test_tool", args)

        with caplog.at_level(logging.DEBUG, logger="victor.framework.middleware"):
            result = await middleware.after_tool_call("test_tool", args, "result", success=True)

        assert result is None  # No modification
        assert "Tool success: test_tool" in caplog.text

    @pytest.mark.asyncio
    async def test_after_tool_call_logs_failure(self, middleware, caplog):
        """LoggingMiddleware should log after failed tool call."""
        await middleware.before_tool_call("test_tool", {})

        with caplog.at_level(logging.DEBUG):
            await middleware.after_tool_call("test_tool", {}, "error", success=False)

        assert "Tool failed: test_tool" in caplog.text

    @pytest.mark.asyncio
    async def test_excluded_tools_not_logged(self):
        """LoggingMiddleware should not log excluded tools."""
        middleware = LoggingMiddleware(exclude_tools={"read"})

        result = await middleware.before_tool_call("read_file", {})
        assert result.proceed is True

        result = await middleware.after_tool_call("read_file", {}, "data", True)
        assert result is None

    @pytest.mark.asyncio
    async def test_excluded_tools_canonicalize_aliases(self):
        """LoggingMiddleware should treat legacy aliases as the same excluded tool."""
        middleware = LoggingMiddleware(exclude_tools={"read"})

        result = await middleware.before_tool_call("read", {})
        assert result.proceed is True

        result = await middleware.after_tool_call("read", {}, "data", True)
        assert result is None

    @pytest.mark.asyncio
    async def test_sanitizes_sensitive_arguments(self, middleware, caplog):
        """LoggingMiddleware should sanitize sensitive arguments."""
        with caplog.at_level(logging.DEBUG):
            await middleware.before_tool_call(
                "api_call", {"password": "secret123", "api_key": "key123"}
            )

        assert "secret123" not in caplog.text
        assert "key123" not in caplog.text
        assert "[REDACTED]" in caplog.text

    def test_priority_is_deferred(self, middleware):
        """LoggingMiddleware should have DEFERRED priority."""
        assert middleware.get_priority() == MiddlewarePriority.DEFERRED

    def test_applicable_tools_is_none(self, middleware):
        """LoggingMiddleware should apply to all tools."""
        assert middleware.get_applicable_tools() is None


# =============================================================================
# SecretMaskingMiddleware Tests
# =============================================================================


class TestSecretMaskingMiddleware:
    """Tests for SecretMaskingMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create a secret masking middleware instance."""
        return SecretMaskingMiddleware(replacement="[MASKED]")

    @pytest.mark.asyncio
    async def test_before_tool_call_no_masking_by_default(self, middleware):
        """SecretMaskingMiddleware should not mask arguments by default."""
        result = await middleware.before_tool_call(
            "test_tool", {"data": "ghp_1234567890abcdefghijklmnopqrstuvwxyz"}
        )
        assert result.proceed is True
        assert result.modified_arguments is None

    @pytest.mark.asyncio
    async def test_before_tool_call_masks_arguments_when_enabled(self):
        """SecretMaskingMiddleware should mask arguments when enabled."""
        middleware = SecretMaskingMiddleware(replacement="[MASKED]", mask_in_arguments=True)

        result = await middleware.before_tool_call(
            "test_tool", {"token": "ghp_1234567890abcdefghijklmnopqrstuvwxyz"}
        )

        assert result.proceed is True
        assert result.modified_arguments is not None
        assert "[MASKED]" in result.modified_arguments["token"]

    @pytest.mark.asyncio
    async def test_after_tool_call_masks_secrets_in_result(self, middleware):
        """SecretMaskingMiddleware should mask secrets in result."""
        result_with_secret = "Found token: ghp_1234567890abcdefghijklmnopqrstuvwxyz"

        masked = await middleware.after_tool_call("test_tool", {}, result_with_secret, success=True)

        assert masked is not None
        assert "[MASKED]" in masked
        assert "ghp_" not in masked

    @pytest.mark.asyncio
    async def test_after_tool_call_masks_aws_keys(self, middleware):
        """SecretMaskingMiddleware should mask AWS keys."""
        result_with_key = "AWS key: AKIAIOSFODNN7EXAMPLE"

        masked = await middleware.after_tool_call("test_tool", {}, result_with_key, success=True)

        assert masked is not None
        assert "[MASKED]" in masked
        assert "AKIAIOSFODNN7" not in masked

    @pytest.mark.asyncio
    async def test_after_tool_call_no_change_when_no_secrets(self, middleware):
        """SecretMaskingMiddleware should return None when no secrets found."""
        clean_result = "This is a normal response without secrets."

        masked = await middleware.after_tool_call("test_tool", {}, clean_result, success=True)

        assert masked is None  # No modification needed

    @pytest.mark.asyncio
    async def test_after_tool_call_handles_dict_result(self, middleware):
        """SecretMaskingMiddleware should handle dict results."""
        result_dict = {
            "output": "Token: ghp_1234567890abcdefghijklmnopqrstuvwxyz",
            "status": "ok",
        }

        masked = await middleware.after_tool_call("test_tool", {}, result_dict, success=True)

        assert masked is not None
        assert "[MASKED]" in masked["output"]
        assert masked["status"] == "ok"

    @pytest.mark.asyncio
    async def test_after_tool_call_handles_none_result(self, middleware):
        """SecretMaskingMiddleware should handle None result."""
        masked = await middleware.after_tool_call("test_tool", {}, None, success=True)
        assert masked is None

    def test_priority_is_high(self, middleware):
        """SecretMaskingMiddleware should have HIGH priority."""
        assert middleware.get_priority() == MiddlewarePriority.HIGH


# =============================================================================
# MetricsMiddleware Tests
# =============================================================================


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create a metrics middleware instance."""
        return MetricsMiddleware(enable_timing=True)

    @pytest.mark.asyncio
    async def test_records_call_count(self, middleware):
        """MetricsMiddleware should record call counts."""
        await middleware.before_tool_call("test_tool", {})
        await middleware.after_tool_call("test_tool", {}, "result", success=True)

        metrics = middleware.get_metrics("test_tool")
        assert metrics is not None
        assert metrics.call_count == 1
        assert metrics.success_count == 1
        assert metrics.failure_count == 0

    @pytest.mark.asyncio
    async def test_records_success_and_failure(self, middleware):
        """MetricsMiddleware should track success and failure separately."""
        # Record successes
        for _ in range(3):
            await middleware.before_tool_call("test_tool", {})
            await middleware.after_tool_call("test_tool", {}, "ok", success=True)

        # Record failures
        for _ in range(2):
            await middleware.before_tool_call("test_tool", {})
            await middleware.after_tool_call("test_tool", {}, "error", success=False)

        metrics = middleware.get_metrics("test_tool")
        assert metrics.call_count == 5
        assert metrics.success_count == 3
        assert metrics.failure_count == 2
        assert metrics.success_rate == 0.6

    @pytest.mark.asyncio
    async def test_records_timing(self, middleware):
        """MetricsMiddleware should record execution timing."""
        await middleware.before_tool_call("test_tool", {})
        # Simulate some delay (would happen naturally in real execution)
        await middleware.after_tool_call("test_tool", {}, "result", success=True)

        metrics = middleware.get_metrics("test_tool")
        assert metrics.total_duration_ms >= 0
        assert metrics.min_duration_ms >= 0
        assert metrics.max_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_callback_invoked(self, middleware):
        """MetricsMiddleware should invoke callback after recording."""
        callback_data = []
        middleware._callback = lambda name, m: callback_data.append((name, m))

        await middleware.before_tool_call("test_tool", {})
        await middleware.after_tool_call("test_tool", {}, "result", success=True)

        assert len(callback_data) == 1
        assert callback_data[0][0] == "test_tool"
        assert callback_data[0][1].call_count == 1

    def test_get_summary_returns_all_metrics(self, middleware):
        """MetricsMiddleware should return summary of all metrics."""
        # Manually add metrics
        middleware._metrics["tool1"] = ToolMetrics(tool_name="tool1", call_count=5)
        middleware._metrics["tool2"] = ToolMetrics(tool_name="tool2", call_count=3)

        summary = middleware.get_summary()
        assert len(summary) == 2
        assert "tool1" in summary
        assert "tool2" in summary

    def test_reset_clears_metrics(self, middleware):
        """MetricsMiddleware.reset() should clear all metrics."""
        middleware._metrics["tool1"] = ToolMetrics(tool_name="tool1", call_count=5)
        middleware.reset()

        assert len(middleware._metrics) == 0
        assert middleware.get_metrics("tool1") is None

    def test_export_prometheus(self, middleware):
        """MetricsMiddleware should export Prometheus format."""
        middleware._metrics["test_tool"] = ToolMetrics(
            tool_name="test_tool",
            call_count=10,
            success_count=8,
            total_duration_ms=500.0,
        )

        output = middleware.export_prometheus()
        assert 'tool_calls_total{tool="test_tool"} 10' in output
        assert 'tool_calls_success_total{tool="test_tool"} 8' in output
        assert "tool_duration_ms_avg" in output

    def test_priority_is_low(self, middleware):
        """MetricsMiddleware should have LOW priority."""
        assert middleware.get_priority() == MiddlewarePriority.LOW


class TestToolMetrics:
    """Tests for ToolMetrics dataclass."""

    def test_avg_duration_zero_when_no_calls(self):
        """avg_duration_ms should be 0 when no calls made."""
        metrics = ToolMetrics(tool_name="test")
        assert metrics.avg_duration_ms == 0.0

    def test_avg_duration_calculated(self):
        """avg_duration_ms should be calculated correctly."""
        metrics = ToolMetrics(tool_name="test", call_count=4, total_duration_ms=200.0)
        assert metrics.avg_duration_ms == 50.0

    def test_success_rate_zero_when_no_calls(self):
        """success_rate should be 0 when no calls made."""
        metrics = ToolMetrics(tool_name="test")
        assert metrics.success_rate == 0.0

    def test_success_rate_calculated(self):
        """success_rate should be calculated correctly."""
        metrics = ToolMetrics(tool_name="test", call_count=10, success_count=7)
        assert metrics.success_rate == 0.7


# =============================================================================
# GitSafetyMiddleware Tests
# =============================================================================


class TestGitSafetyMiddleware:
    """Tests for GitSafetyMiddleware."""

    @pytest.fixture
    def middleware(self):
        """Create a git safety middleware instance."""
        return GitSafetyMiddleware(block_dangerous=True, warn_on_risky=True)

    @pytest.mark.asyncio
    async def test_blocks_force_push(self, middleware):
        """GitSafetyMiddleware should block force push."""
        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git push --force origin main"}
        )

        assert result.proceed is False
        assert "force" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_blocks_force_push_short_flag(self, middleware):
        """GitSafetyMiddleware should block force push with -f flag."""
        result = await middleware.before_tool_call("git", {"command": "git push -f origin main"})

        assert result.proceed is False
        # Message may say "push -f" or "force" depending on which pattern matched
        assert "push -f" in result.error_message or "force" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_blocks_hard_reset(self, middleware):
        """GitSafetyMiddleware should block hard reset."""
        result = await middleware.before_tool_call("bash", {"command": "git reset --hard HEAD~3"})

        assert result.proceed is False

    @pytest.mark.asyncio
    async def test_blocks_clean_fd(self, middleware):
        """GitSafetyMiddleware should block git clean -fd."""
        result = await middleware.before_tool_call("shell", {"command": "git clean -fd"})

        assert result.proceed is False

    @pytest.mark.asyncio
    async def test_warns_on_risky_operations(self, middleware):
        """GitSafetyMiddleware should warn on risky operations."""
        result = await middleware.before_tool_call("execute_bash", {"command": "git rebase main"})

        assert result.proceed is True
        assert "git_warning" in result.metadata

    @pytest.mark.asyncio
    async def test_warns_on_stash_drop(self, middleware):
        """GitSafetyMiddleware should warn on stash drop."""
        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git stash drop stash@{0}"}
        )

        assert result.proceed is True
        assert "git_warning" in result.metadata

    @pytest.mark.asyncio
    async def test_allows_safe_operations(self, middleware):
        """GitSafetyMiddleware should allow safe git operations."""
        result = await middleware.before_tool_call("execute_bash", {"command": "git status"})

        assert result.proceed is True
        assert not result.error_message

    @pytest.mark.asyncio
    async def test_allows_normal_push(self, middleware):
        """GitSafetyMiddleware should allow normal push."""
        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git push origin feature-branch"}
        )

        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_ignores_non_git_commands(self, middleware):
        """GitSafetyMiddleware should ignore non-git commands."""
        result = await middleware.before_tool_call("execute_bash", {"command": "ls -la"})

        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_ignores_non_applicable_tools(self, middleware):
        """GitSafetyMiddleware should ignore non-applicable tools."""
        result = await middleware.before_tool_call("read_file", {"path": "/etc/passwd"})

        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_handles_bash_alias_via_canonicalization(self, middleware):
        """GitSafetyMiddleware should handle shell aliases through canonical matching."""
        result = await middleware.before_tool_call("bash", {"command": "git status"})

        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_blocks_force_push_to_protected_branch(self, middleware):
        """GitSafetyMiddleware should block force push to protected branches."""
        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git push --force origin main"}
        )

        assert result.proceed is False
        assert "main" in result.error_message or "force" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_custom_protected_branches(self):
        """GitSafetyMiddleware should respect custom protected branches."""
        middleware = GitSafetyMiddleware(
            block_dangerous=True,
            protected_branches={"deploy", "release"},
        )

        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git push --force origin deploy"}
        )

        # The force push itself is blocked, regardless of branch
        assert result.proceed is False

    @pytest.mark.asyncio
    async def test_allowed_force_branches(self):
        """GitSafetyMiddleware should allow force push on configured branches."""
        middleware = GitSafetyMiddleware(
            block_dangerous=True,
            allowed_force_branches={"feature/*"},
        )

        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git push --force origin feature/test"}
        )

        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_disabled_blocking(self):
        """GitSafetyMiddleware should not block when disabled."""
        middleware = GitSafetyMiddleware(block_dangerous=False, warn_on_risky=False)

        result = await middleware.before_tool_call(
            "execute_bash", {"command": "git push --force origin main"}
        )

        assert result.proceed is True

    def test_priority_is_critical(self, middleware):
        """GitSafetyMiddleware should have CRITICAL priority."""
        assert middleware.get_priority() == MiddlewarePriority.CRITICAL

    def test_applicable_tools(self, middleware):
        """GitSafetyMiddleware should only apply to shell tools."""
        tools = middleware.get_applicable_tools()
        assert "git" in tools
        assert "execute_bash" in tools
        assert "bash" in tools
        assert "shell" in tools


# =============================================================================
# OutputValidationMiddleware Tests
# =============================================================================


class SimpleJsonValidator:
    """Simple JSON validator for testing."""

    def validate(self, content: str, context=None) -> ContentValidationResult:
        """Validate JSON content."""
        import json

        try:
            json.loads(content)
            return ContentValidationResult(is_valid=True)
        except json.JSONDecodeError as e:
            return ContentValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        message=f"Invalid JSON: {e.msg}",
                        severity=ValidationSeverity.ERROR,
                        location=f"line {e.lineno}",
                    )
                ],
            )


class FixableValidator:
    """Validator that can auto-fix issues for testing."""

    def validate(self, content: str, context=None) -> ContentValidationResult:
        """Validate content - fails if 'BAD' is in content."""
        if "BAD" in content:
            return ContentValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        message="Content contains BAD",
                        severity=ValidationSeverity.ERROR,
                    )
                ],
            )
        return ContentValidationResult(is_valid=True)

    def fix(self, content: str, issues, context=None) -> str:
        """Fix issues by replacing BAD with GOOD."""
        return content.replace("BAD", "GOOD")


class TestOutputValidationMiddleware:
    """Tests for OutputValidationMiddleware."""

    @pytest.fixture
    def json_validator(self):
        """Create a JSON validator."""
        return SimpleJsonValidator()

    @pytest.fixture
    def fixable_validator(self):
        """Create a fixable validator."""
        return FixableValidator()

    @pytest.fixture
    def middleware(self, json_validator):
        """Create validation middleware with JSON validator."""
        return OutputValidationMiddleware(
            validator=json_validator,
            applicable_tools={"write_file", "create_config"},
            argument_names={"content"},
        )

    @pytest.mark.asyncio
    async def test_valid_content_passes(self, middleware):
        """Valid content should pass validation."""
        result = await middleware.before_tool_call(
            "write_file", {"content": '{"valid": true}', "path": "/test.json"}
        )

        assert result.proceed is True
        assert result.metadata.get("validation_passed") is True
        assert result.metadata.get("issue_count") == 0

    @pytest.mark.asyncio
    async def test_invalid_content_detected(self, middleware):
        """Invalid content should be detected."""
        result = await middleware.before_tool_call(
            "write_file", {"content": '{"invalid": }', "path": "/test.json"}
        )

        assert result.proceed is True  # Default: don't block
        assert result.metadata.get("validation_passed") is False
        assert result.metadata.get("issue_count") > 0
        assert "validation_issues" in result.metadata

    @pytest.mark.asyncio
    async def test_block_on_error(self, json_validator):
        """Middleware should block when block_on_error is True."""
        middleware = OutputValidationMiddleware(
            validator=json_validator,
            argument_names={"content"},
            block_on_error=True,
        )

        result = await middleware.before_tool_call(
            "write_file", {"content": "not valid json", "path": "/test.json"}
        )

        assert result.proceed is False
        assert "Validation failed" in result.error_message

    @pytest.mark.asyncio
    async def test_auto_fix_applied(self, fixable_validator):
        """Middleware should auto-fix when enabled and validator supports it."""
        middleware = OutputValidationMiddleware(
            validator=fixable_validator,
            argument_names={"content"},
            auto_fix=True,
        )

        result = await middleware.before_tool_call(
            "write_file", {"content": "This is BAD content", "path": "/test.txt"}
        )

        assert result.proceed is True
        assert result.modified_arguments is not None
        assert "GOOD" in result.modified_arguments["content"]
        assert result.metadata.get("content_fixed") is True

    @pytest.mark.asyncio
    async def test_auto_fix_disabled(self, fixable_validator):
        """Middleware should not auto-fix when disabled."""
        middleware = OutputValidationMiddleware(
            validator=fixable_validator,
            argument_names={"content"},
            auto_fix=False,
        )

        result = await middleware.before_tool_call(
            "write_file", {"content": "This is BAD content", "path": "/test.txt"}
        )

        assert result.proceed is True
        assert result.modified_arguments is None
        assert result.metadata.get("content_fixed") is False

    @pytest.mark.asyncio
    async def test_applicable_tools_filter(self, json_validator):
        """Middleware should only validate applicable tools."""
        middleware = OutputValidationMiddleware(
            validator=json_validator,
            applicable_tools={"write_json"},
            argument_names={"content"},
        )

        # Applicable tool - should validate
        result = await middleware.before_tool_call("write_json", {"content": '{"valid": true}'})
        assert result.metadata.get("validation_performed") is True

        # Non-applicable tool - should skip
        result = await middleware.before_tool_call("other_tool", {"content": "not json"})
        assert middleware.get_applicable_tools() == {"write_json"}

    @pytest.mark.asyncio
    async def test_argument_names_filter(self, json_validator):
        """Middleware should only validate specified argument names."""
        middleware = OutputValidationMiddleware(
            validator=json_validator,
            argument_names={"json_data"},  # Only validate json_data
        )

        # Specified argument - should validate
        result = await middleware.before_tool_call(
            "write_file", {"json_data": '{"valid": true}', "other": "ignored"}
        )
        assert result.metadata.get("validation_performed") is True

        # No matching argument - should skip
        result = await middleware.before_tool_call(
            "write_file", {"content": "will not be validated"}
        )
        # Empty result since no arguments matched
        assert result.proceed is True

    @pytest.mark.asyncio
    async def test_severity_levels(self, json_validator):
        """Middleware should track severity levels correctly."""
        result = await OutputValidationMiddleware(
            validator=json_validator,
            argument_names={"content"},
        ).before_tool_call("write_file", {"content": "not json"})

        assert result.metadata.get("error_count") > 0

    @pytest.mark.asyncio
    async def test_max_fix_iterations(self):
        """Middleware should respect max_fix_iterations."""
        call_count = 0

        class InfiniteFixValidator:
            def validate(self, content, context=None):
                return ContentValidationResult(
                    is_valid=False,
                    issues=[ValidationIssue(message="Always fails")],
                )

            def fix(self, content, issues, context=None):
                nonlocal call_count
                call_count += 1
                return content + "_fixed"

        middleware = OutputValidationMiddleware(
            validator=InfiniteFixValidator(),
            argument_names={"content"},
            auto_fix=True,
            max_fix_iterations=3,
        )

        await middleware.before_tool_call("tool", {"content": "test"})

        # Should stop after max_fix_iterations
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_after_tool_call_no_op(self, middleware):
        """after_tool_call should be a no-op."""
        result = await middleware.after_tool_call(
            "write_file", {"content": "{}"}, "result", success=True
        )
        assert result is None

    def test_priority_is_high(self, middleware):
        """Default priority should be HIGH."""
        assert middleware.get_priority() == MiddlewarePriority.HIGH

    def test_custom_priority(self, json_validator):
        """Custom priority should be respected."""
        middleware = OutputValidationMiddleware(
            validator=json_validator,
            priority=MiddlewarePriority.CRITICAL,
        )
        assert middleware.get_priority() == MiddlewarePriority.CRITICAL

    @pytest.mark.asyncio
    async def test_validator_exception_handled(self):
        """Middleware should handle validator exceptions gracefully."""

        class BrokenValidator:
            def validate(self, content, context=None):
                raise RuntimeError("Validator exploded")

        middleware = OutputValidationMiddleware(
            validator=BrokenValidator(),
            argument_names={"content"},
        )

        result = await middleware.before_tool_call("tool", {"content": "test"})

        assert result.proceed is True  # Should not block on exception
        assert result.metadata.get("validation_passed") is False


class TestValidationTypes:
    """Tests for validation type dataclasses."""

    def test_validation_result_defaults(self):
        """ContentValidationResult should have sensible defaults."""
        result = ContentValidationResult()
        assert result.is_valid is True
        assert result.issues == []
        assert result.fixed_content is None
        assert result.metadata == {}

    def test_validation_result_error_count(self):
        """ContentValidationResult.error_count should count errors and criticals."""
        result = ContentValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(message="Error 1", severity=ValidationSeverity.ERROR),
                ValidationIssue(message="Warning", severity=ValidationSeverity.WARNING),
                ValidationIssue(message="Critical", severity=ValidationSeverity.CRITICAL),
                ValidationIssue(message="Info", severity=ValidationSeverity.INFO),
            ],
        )
        assert result.error_count == 2  # ERROR + CRITICAL
        assert result.warning_count == 1

    def test_validation_result_has_fix(self):
        """ContentValidationResult.has_fix should check for fixed_content."""
        result1 = ContentValidationResult()
        assert result1.has_fix is False

        result2 = ContentValidationResult(fixed_content="fixed")
        assert result2.has_fix is True

    def test_validation_issue_defaults(self):
        """ValidationIssue should have sensible defaults."""
        issue = ValidationIssue(message="test")
        assert issue.message == "test"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.location is None
        assert issue.suggestion is None
        assert issue.code is None

    def test_validation_severity_values(self):
        """ValidationSeverity should have correct values."""
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.CRITICAL.value == "critical"


# =============================================================================
# Integration Tests
# =============================================================================


class TestMiddlewareIntegration:
    """Integration tests for middleware working together."""

    @pytest.mark.asyncio
    async def test_middleware_chain_ordering(self):
        """Middleware should execute in priority order."""
        execution_order = []

        class TrackingMiddleware:
            def __init__(self, name, priority):
                self._name = name
                self._priority = priority

            async def before_tool_call(self, tool_name, arguments):
                from victor.core.vertical_types import MiddlewareResult

                execution_order.append(f"{self._name}_before")
                return MiddlewareResult()

            async def after_tool_call(self, tool_name, arguments, result, success):
                execution_order.append(f"{self._name}_after")
                return None

            def get_priority(self):
                return self._priority

            def get_applicable_tools(self):
                return None

        # Create middleware with different priorities
        m1 = TrackingMiddleware("critical", MiddlewarePriority.CRITICAL)
        m2 = TrackingMiddleware("high", MiddlewarePriority.HIGH)
        m3 = TrackingMiddleware("low", MiddlewarePriority.LOW)

        # Simulate execution in priority order
        middlewares = sorted([m1, m2, m3], key=lambda m: m.get_priority().value)

        for m in middlewares:
            await m.before_tool_call("test", {})

        assert execution_order == ["critical_before", "high_before", "low_before"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_devops_middleware_configuration(self):
        """DevOps vertical should configure middleware correctly."""
        try:
            DevOpsAssistant = _load_vertical_attr("victor.devops.assistant", "DevOpsAssistant")
        except ImportError:
            pytest.skip(
                "DevOps vertical not installed (contrib removed, external package required)"
            )

        middleware_list = DevOpsAssistant.get_middleware()

        # Should have at least GitSafetyMiddleware
        middleware_types = [type(m).__name__ for m in middleware_list]
        assert "GitSafetyMiddleware" in middleware_types
        assert "SecretMaskingMiddleware" in middleware_types
        assert "LoggingMiddleware" in middleware_types

        # Git safety should block dangerous operations for DevOps
        git_middleware = next(
            m for m in middleware_list if type(m).__name__ == "GitSafetyMiddleware"
        )
        result = await git_middleware.before_tool_call(
            "execute_bash", {"command": "git push --force origin main"}
        )
        assert result.proceed is False


__all__ = [
    "TestLoggingMiddleware",
    "TestSecretMaskingMiddleware",
    "TestMetricsMiddleware",
    "TestGitSafetyMiddleware",
    "TestToolMetrics",
    "TestOutputValidationMiddleware",
    "TestValidationTypes",
    "TestMiddlewareIntegration",
]
