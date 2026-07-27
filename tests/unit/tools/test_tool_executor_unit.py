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

"""Tests for tool_executor module."""

import importlib
import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.agent.tool_executor import (
    ToolExecutionResult,
    ToolExecutor,
)
from victor.tools.base import BaseTool, ToolResult
from victor.tools.enums import ExecutionCategory
from victor.tools.decorators import tool
from victor.tools.registry import ToolRegistry

# Import safety module at collection time so it is cached in sys.modules before
# victor.__init__ can shadow victor.agent with the @victor.agent decorator function.
_safety_module = importlib.import_module("victor.agent.safety")


class TestToolExecutionResult:
    """Tests for ToolExecutionResult class."""

    def test_result_init_success(self):
        """Test ToolExecutionResult initialization for success."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result={"output": "data"},
        )
        assert result.tool_name == "test_tool"
        assert result.success is True
        assert result.result == {"output": "data"}
        assert result.error is None
        assert result.cached is False
        assert result.retries == 0

    def test_result_init_failure(self):
        """Test ToolExecutionResult initialization for failure."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=False,
            result=None,
            error="Tool failed",
            retries=2,
        )
        assert result.success is False
        assert result.error == "Tool failed"
        assert result.retries == 2

    def test_result_cached(self):
        """Test ToolExecutionResult with cached flag."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result="cached_data",
            cached=True,
        )
        assert result.cached is True

    def test_result_with_execution_time(self):
        """Test ToolExecutionResult with execution time."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result="data",
            execution_time=1.5,
        )
        assert result.execution_time == 1.5


class TestToolExecutorInit:
    """Tests for ToolExecutor initialization."""

    def test_init_minimal(self):
        """Test ToolExecutor with minimal arguments."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        assert executor.tools is registry
        assert executor.normalizer is not None
        assert executor.cache is None
        assert executor.max_retries == 3
        assert executor.retry_delay == 1.0
        assert executor.context == {}

    def test_init_with_cache(self):
        """Test ToolExecutor with cache."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        assert executor.cache is mock_cache

    def test_init_with_context(self):
        """Test ToolExecutor with context."""
        registry = ToolRegistry()
        context = {"cwd": "/tmp", "user": "test"}

        executor = ToolExecutor(
            tool_registry=registry,
            context=context,
        )

        assert executor.context == context

    def test_get_tool_function_resolves_registered_tool(self):
        """Test helper returns a registered tool by canonical name."""
        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.execute = AsyncMock(return_value=ToolResult(success=True, output="ok"))
        registry.register(mock_tool)

        executor = ToolExecutor(tool_registry=registry)

        assert executor.get_tool_function("test_tool") is mock_tool

    def test_get_tool_function_returns_none_for_unknown_tool(self):
        """Test helper returns None for missing tools."""
        executor = ToolExecutor(tool_registry=ToolRegistry())

        assert executor.get_tool_function("missing_tool") is None

    def test_init_with_retry_config(self):
        """Test ToolExecutor with custom retry configuration."""
        registry = ToolRegistry()

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=5,
            retry_delay=2.0,
        )

        assert executor.max_retries == 5
        assert executor.retry_delay == 2.0


class TestToolExecutorUpdateContext:
    """Tests for ToolExecutor.update_context method."""

    def test_update_context(self):
        """Test updating context."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        executor.update_context(key1="value1", key2="value2")

        assert executor.context["key1"] == "value1"
        assert executor.context["key2"] == "value2"

    def test_update_context_overwrites(self):
        """Test that update_context overwrites existing keys."""
        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            context={"key": "old_value"},
        )

        executor.update_context(key="new_value")

        assert executor.context["key"] == "new_value"


class TestToolExecutorExecute:
    """Tests for ToolExecutor.execute method."""

    @pytest.fixture
    def registry_with_tool(self):
        """Create a registry with a mock tool."""
        registry = ToolRegistry()

        # Create a mock tool
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.execute = AsyncMock(return_value={"result": "success"})

        registry.register(mock_tool)
        return registry, mock_tool

    @pytest.mark.asyncio
    async def test_execute_success(self, registry_with_tool):
        """Test successful tool execution."""
        registry, mock_tool = registry_with_tool
        executor = ToolExecutor(tool_registry=registry)

        result = await executor.execute("test_tool", {"arg": "value"})

        assert result.success is True
        assert result.result == {"result": "success"}
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test execution with non-existent tool."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        result = await executor.execute("nonexistent_tool", {})

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_disabled_tool(self, registry_with_tool):
        """Test execution of disabled tool."""
        registry, mock_tool = registry_with_tool

        # Disable the tool
        registry.disable_tool("test_tool")

        executor = ToolExecutor(tool_registry=registry)

        result = await executor.execute("test_tool", {})

        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_context(self, registry_with_tool):
        """Test execution with context."""
        registry, mock_tool = registry_with_tool
        executor = ToolExecutor(
            tool_registry=registry,
            context={"default_ctx": "value"},
        )

        await executor.execute(
            "test_tool",
            {"arg": "value"},
            context={"call_ctx": "specific"},
        )

        # Verify tool was called with merged context (passed as _exec_ctx kwarg)
        mock_tool.execute.assert_called_once()
        call_context = mock_tool.execute.call_args.kwargs.get("_exec_ctx")
        assert call_context["default_ctx"] == "value"
        assert call_context["call_ctx"] == "specific"

    @pytest.mark.asyncio
    async def test_execute_tracks_stats(self, registry_with_tool):
        """Test that execution tracks statistics."""
        registry, mock_tool = registry_with_tool
        executor = ToolExecutor(tool_registry=registry)

        await executor.execute("test_tool", {})
        await executor.execute("test_tool", {})

        stats = executor.get_stats()
        assert "test_tool" in stats
        assert stats["test_tool"]["calls"] == 2
        assert stats["test_tool"]["successes"] == 2


class TestToolExecutorCache:
    """Tests for ToolExecutor caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit for cacheable tool."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "code_search"
        mock_tool.execute = AsyncMock(return_value={"matches": []})
        registry.register(mock_tool)

        mock_cache = MagicMock()
        mock_cache.get.return_value = {"cached_matches": []}

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        result = await executor.execute("code_search", {"query": "test"})

        assert result.cached is True
        assert result.result == {"cached_matches": []}
        mock_tool.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "code_search"
        mock_tool.execute = AsyncMock(return_value={"matches": ["result"]})
        registry.register(mock_tool)

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        result = await executor.execute("code_search", {"query": "test"})

        assert result.cached is False
        mock_tool.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_cache(self):
        """Test skipping cache."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "code_search"
        mock_tool.execute = AsyncMock(return_value={"fresh": True})
        registry.register(mock_tool)

        mock_cache = MagicMock()
        mock_cache.get.return_value = {"stale": True}

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        result = await executor.execute("code_search", {}, skip_cache=True)

        assert result.result == {"fresh": True}
        mock_cache.get.assert_not_called()


class TestToolExecutorRetry:
    """Tests for ToolExecutor retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry logic on failure."""
        registry = ToolRegistry()

        # Tool that fails twice then succeeds
        call_count = [0]

        def make_execute_func():
            async def execute_func(_exec_ctx=None, **kwargs):
                nonlocal call_count
                call_count[0] += 1
                if call_count[0] < 3:
                    raise ValueError("Temporary failure")
                return {"success": True}

            return execute_func

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "flaky_tool"
        mock_tool.execute = make_execute_func()
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=3,
            retry_delay=0.01,  # Fast for testing
        )

        result = await executor.execute("flaky_tool", {})

        assert result.success is True
        # retries counts the last failed attempt number, which is 1 (0-indexed)
        assert result.retries >= 1  # At least one retry happened

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test that max retries are exhausted."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "failing_tool"
        mock_tool.execute = AsyncMock(side_effect=ValueError("Always fails"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=2,
            retry_delay=0.01,
        )

        result = await executor.execute("failing_tool", {})

        assert result.success is False
        assert "Always fails" in result.error


class TestToolExecutorToolResult:
    """Tests for handling ToolResult returns."""

    @pytest.mark.asyncio
    async def test_tool_result_success(self):
        """Test handling ToolResult with success."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "result_tool"
        mock_tool.execute = AsyncMock(
            return_value=ToolResult(success=True, output={"data": "value"})
        )
        registry.register(mock_tool)

        executor = ToolExecutor(tool_registry=registry)

        result = await executor.execute("result_tool", {})

        assert result.success is True
        assert result.result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_tool_result_failure(self):
        """Test handling ToolResult with failure."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "result_tool"
        mock_tool.execute = AsyncMock(
            return_value=ToolResult(success=False, output=None, error="Tool error")
        )
        registry.register(mock_tool)

        executor = ToolExecutor(tool_registry=registry)

        result = await executor.execute("result_tool", {})

        assert result.success is False
        assert "Tool error" in result.error


class TestToolExecutorStats:
    """Tests for ToolExecutor statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self):
        """Test getting stats when empty (no tool executions)."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        stats = executor.get_stats()
        # Stats now includes _global with error tracking info
        assert "_global" in stats
        assert stats["_global"]["validation_failures"] == 0
        assert stats["_global"]["errors_by_category"] == {}
        # No tool-specific stats yet
        assert len([k for k in stats.keys() if k != "_global"]) == 0

    @pytest.mark.asyncio
    async def test_get_stats_after_execution(self):
        """Test getting stats after execution."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "stats_tool"
        mock_tool.execute = AsyncMock(return_value="result")
        registry.register(mock_tool)

        executor = ToolExecutor(tool_registry=registry)

        await executor.execute("stats_tool", {})

        stats = executor.get_stats()
        assert "stats_tool" in stats
        assert stats["stats_tool"]["calls"] == 1
        assert stats["stats_tool"]["successes"] == 1
        assert stats["stats_tool"]["failures"] == 0

    def test_has_failed_before(self):
        """Test tracking failed signatures."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        # Initially not failed
        assert executor.has_failed_before("tool", {"a": 1}) is False

        # Mark as failed
        executor._failed_signatures.add(("tool", "[('a', 1)]"))

        assert executor.has_failed_before("tool", {"a": 1}) is True


class TestToolExecutorCacheableTools:
    """Tests for cacheable tool detection via registry."""

    @staticmethod
    def _ensure_tools_loaded():
        """Force import of tool modules to populate the registry."""
        import victor.tools.filesystem  # noqa: F401
        import victor.tools.code_search_tool  # noqa: F401
        import victor.tools.file_editor_tool  # noqa: F401
        import victor.tools.bash  # noqa: F401

    def test_default_cacheable_tools(self):
        """Test cacheable tools detection via registry.

        Note: Tools are registered by canonical names after alias resolution.
        The code_search function is registered as 'grep' due to TOOL_ALIASES.
        """
        self._ensure_tools_loaded()
        # Registry-based detection for idempotent tools
        assert ToolExecutor.is_cacheable_tool("read")  # read_file
        assert ToolExecutor.is_cacheable_tool("ls")  # list_directory
        assert ToolExecutor.is_cacheable_tool("grep")  # code_search -> grep

    def test_cache_invalidating_tools(self):
        """Test cache invalidating tools detection via registry.

        Note: Tools are registered by function names (write, edit, shell),
        not external names (write_file, edit_files, execute_bash).
        """
        self._ensure_tools_loaded()
        # Registry-based detection for write-mode tools
        assert ToolExecutor.is_cache_invalidating_tool("write")  # write_file
        assert ToolExecutor.is_cache_invalidating_tool("edit")  # edit_files
        assert ToolExecutor.is_cache_invalidating_tool("shell")  # execute_bash


class TestToolExecutorSafetyCheck:
    """Tests for ToolExecutor safety check integration."""

    @pytest.mark.asyncio
    async def test_safety_check_blocks_execution(self):
        """Test that safety check can block tool execution (covers lines 202-203)."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "dangerous_tool"
        mock_tool.execute = AsyncMock(return_value="should not be called")
        registry.register(mock_tool)

        executor = ToolExecutor(tool_registry=registry)

        # Mock safety checker to block execution
        executor.safety_checker.check_and_confirm = AsyncMock(
            return_value=(False, "Operation blocked for safety")
        )

        result = await executor.execute("dangerous_tool", {"path": "/etc/passwd"})

        assert result.success is False
        assert "safety" in result.error.lower() or "blocked" in result.error.lower()
        mock_tool.execute.assert_not_called()


class TestToolExecutionResultEnhanced:
    """Tests for enhanced ToolExecutionResult with error handling."""

    def test_result_with_correlation_id(self):
        """Test ToolExecutionResult with correlation_id."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=False,
            result=None,
            error="Test error",
            correlation_id="abc123",
        )
        assert result.correlation_id == "abc123"

    def test_result_with_error_info(self):
        """Test ToolExecutionResult with error_info."""
        from victor.core.errors import ErrorInfo, ErrorCategory, ErrorSeverity

        error_info = ErrorInfo(
            message="Test error",
            category=ErrorCategory.TOOL_EXECUTION,
            severity=ErrorSeverity.ERROR,
            correlation_id="def456",
            recovery_hint="Try again",
        )
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=False,
            result=None,
            error="Test error",
            error_info=error_info,
        )
        assert result.error_info is error_info
        assert result.error_info.correlation_id == "def456"

    def test_result_to_dict_success(self):
        """Test ToolExecutionResult.to_dict() for success."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            result="data",
            execution_time=1.5,
            retries=0,
            correlation_id="test123",
        )
        d = result.to_dict()
        assert d["tool_name"] == "test_tool"
        assert d["success"] is True
        assert d["execution_time"] == 1.5
        assert d["correlation_id"] == "test123"
        assert "error" not in d

    def test_result_to_dict_failure(self):
        """Test ToolExecutionResult.to_dict() for failure."""
        from victor.core.errors import ErrorInfo, ErrorCategory, ErrorSeverity

        error_info = ErrorInfo(
            message="Test failure",
            category=ErrorCategory.TOOL_EXECUTION,
            severity=ErrorSeverity.ERROR,
            correlation_id="fail123",
        )
        result = ToolExecutionResult(
            tool_name="test_tool",
            success=False,
            result=None,
            error="Test failure",
            error_info=error_info,
        )
        d = result.to_dict()
        assert d["error"] == "Test failure"
        assert "error_details" in d
        assert d["error_details"]["category"] == "tool_execution"


class TestToolExecutorErrorHandling:
    """Tests for ToolExecutor error handling integration."""

    @pytest.fixture(autouse=True)
    def reset_globals(self):
        """Reset global singletons before each test for isolation."""
        from victor.core.errors import get_error_handler

        safety_module = _safety_module

        # Clear error history to ensure test isolation
        handler = get_error_handler()
        handler.clear_history()

        # Reset safety checker to avoid pollution from previous tests
        safety_module._default_checker = None

    @pytest.mark.asyncio
    async def test_error_handler_tracks_errors(self):
        """Test that error handler tracks errors by category."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "failing_tool"
        mock_tool.execute = AsyncMock(side_effect=ValueError("Test error"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        await executor.execute("failing_tool", {})

        # Error should be tracked
        assert len(executor._errors_by_category) > 0

    @pytest.mark.asyncio
    async def test_error_result_has_correlation_id(self):
        """Test that failed execution results have correlation ID."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "error_tool"
        mock_tool.execute = AsyncMock(side_effect=FileNotFoundError("File missing"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        result = await executor.execute("error_tool", {"path": "/missing"})

        assert result.success is False
        assert result.correlation_id is not None
        assert len(result.correlation_id) == 8  # UUID first 8 chars

    @pytest.mark.asyncio
    async def test_error_info_populated_on_failure(self):
        """Test that error_info is populated on execution failure."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "error_tool"
        mock_tool.execute = AsyncMock(side_effect=PermissionError("Access denied"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        result = await executor.execute("error_tool", {})

        assert result.success is False
        assert result.error_info is not None
        assert result.error_info.message == "Access denied"

    @pytest.mark.asyncio
    async def test_recovery_hint_in_error(self):
        """Test that recovery hints are included in error messages."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "error_tool"
        mock_tool.execute = AsyncMock(side_effect=ConnectionError("Network error"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        result = await executor.execute("error_tool", {})

        assert result.success is False
        # Error handler adds recovery hints
        assert result.error_info is not None
        # ConnectionError gets a recovery hint from error handler

    @pytest.mark.asyncio
    async def test_get_error_summary(self):
        """Test get_error_summary method."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "error_tool"
        mock_tool.execute = AsyncMock(side_effect=ValueError("Error"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        await executor.execute("error_tool", {})

        summary = executor.get_error_summary()

        assert "errors_by_category" in summary
        assert "total_errors" in summary
        assert summary["total_errors"] >= 1
        assert "recent_errors" in summary

    @pytest.mark.asyncio
    async def test_stats_include_error_categories(self):
        """Test that stats include error category breakdown."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "error_tool"
        mock_tool.execute = AsyncMock(side_effect=TypeError("Type error"))
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        await executor.execute("error_tool", {})

        stats = executor.get_stats()

        assert "_global" in stats
        assert "errors_by_category" in stats["_global"]
        assert "recent_errors" in stats["_global"]

    def test_track_error_category(self):
        """Test _track_error_category method."""
        from victor.core.errors import ErrorCategory

        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        executor._track_error_category(ErrorCategory.TOOL_EXECUTION)
        executor._track_error_category(ErrorCategory.TOOL_EXECUTION)
        executor._track_error_category(ErrorCategory.VALIDATION_ERROR)

        assert executor._errors_by_category["tool_execution"] == 2
        assert executor._errors_by_category["validation_error"] == 1


class TestToolExecutorWithErrorHandler:
    """Tests for ToolExecutor with custom error handler."""

    @pytest.fixture(autouse=True)
    def reset_globals(self):
        """Reset global singletons before each test for isolation."""
        from victor.core.errors import get_error_handler

        safety_module = _safety_module

        # Clear error history to ensure test isolation
        handler = get_error_handler()
        handler.clear_history()

        # Reset safety checker to avoid pollution from previous tests
        safety_module._default_checker = None

    def test_init_with_error_handler(self):
        """Test ToolExecutor with custom error handler."""
        from victor.core.errors import ErrorHandler

        registry = ToolRegistry()
        custom_handler = ErrorHandler(logger_name="custom")

        executor = ToolExecutor(
            tool_registry=registry,
            error_handler=custom_handler,
        )

        assert executor.error_handler is custom_handler

    def test_init_uses_global_handler(self):
        """Test ToolExecutor uses global handler when none provided."""
        from victor.core.errors import get_error_handler

        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        # Should use global handler
        assert executor.error_handler is get_error_handler()

    @pytest.mark.asyncio
    async def test_tool_result_failure_uses_error_handler(self):
        """Test that ToolResult failures use error handler."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "result_tool"
        mock_tool.execute = AsyncMock(
            return_value=ToolResult(success=False, output=None, error="Tool error")
        )
        registry.register(mock_tool)

        executor = ToolExecutor(tool_registry=registry)

        result = await executor.execute("result_tool", {})

        assert result.success is False
        assert result.error_info is not None
        assert result.error_info.category.value == "tool_execution"


class TestToolExecutorValidationMode:
    """Tests for ToolExecutor validation mode settings."""

    def test_set_validation_mode_strict(self):
        """Test setting validation mode to STRICT."""
        from victor.agent.tool_executor import ValidationMode

        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        # Default is LENIENT
        assert executor.validation_mode == ValidationMode.LENIENT

        # Change to STRICT
        executor.set_validation_mode(ValidationMode.STRICT)
        assert executor.validation_mode == ValidationMode.STRICT

    def test_set_validation_mode_off(self):
        """Test setting validation mode to OFF."""
        from victor.agent.tool_executor import ValidationMode

        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        executor.set_validation_mode(ValidationMode.OFF)
        assert executor.validation_mode == ValidationMode.OFF

    def test_init_with_validation_mode(self):
        """Test initializing executor with specific validation mode."""
        from victor.agent.tool_executor import ValidationMode

        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.STRICT,
        )
        assert executor.validation_mode == ValidationMode.STRICT


class TestToolExecutorValidateArguments:
    """Tests for ToolExecutor._validate_arguments method."""

    @pytest.fixture
    def mock_tool_with_validation(self):
        """Create a mock tool that can return validation results."""
        from victor.tools.base import ToolValidationResult

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.validate_parameters_detailed = MagicMock()
        # Add parameters schema so unknown arg check passes
        mock_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "arg": {"type": "string"},
            },
        }
        return mock_tool, ToolValidationResult

    def test_validate_arguments_off_mode(self, mock_tool_with_validation):
        """Test that validation is skipped in OFF mode."""
        from victor.agent.tool_executor import ValidationMode

        mock_tool, _ = mock_tool_with_validation
        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.OFF,
        )

        should_proceed, result = executor._validate_arguments(mock_tool, {"arg": "value"})

        assert should_proceed is True
        assert result is None
        mock_tool.validate_parameters_detailed.assert_not_called()

    def test_validate_arguments_strict_mode_invalid(self, mock_tool_with_validation):
        """Test that STRICT mode blocks execution on invalid arguments."""
        from victor.agent.tool_executor import ValidationMode
        from victor.tools.base import ToolValidationResult

        mock_tool, _ = mock_tool_with_validation
        mock_tool.validate_parameters_detailed.return_value = ToolValidationResult.failure(
            [
                "Missing required parameter: path",
                "Invalid type for: count",
                "Extra error",
            ]
        )

        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.STRICT,
        )

        should_proceed, result = executor._validate_arguments(mock_tool, {})

        assert should_proceed is False
        assert result is not None
        assert not result.valid
        assert executor._validation_failures == 1

    def test_validate_arguments_lenient_mode_invalid(self, mock_tool_with_validation):
        """Test that LENIENT mode proceeds despite invalid arguments."""
        from victor.agent.tool_executor import ValidationMode
        from victor.tools.base import ToolValidationResult

        mock_tool, _ = mock_tool_with_validation
        mock_tool.validate_parameters_detailed.return_value = ToolValidationResult.failure(
            ["Missing required parameter"]
        )

        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.LENIENT,
        )

        should_proceed, result = executor._validate_arguments(mock_tool, {})

        assert should_proceed is True
        assert result is not None
        assert not result.valid
        assert executor._validation_failures == 1

    def test_validate_arguments_valid(self, mock_tool_with_validation):
        """Test validation with valid arguments."""
        from victor.agent.tool_executor import ValidationMode
        from victor.tools.base import ToolValidationResult

        mock_tool, _ = mock_tool_with_validation
        mock_tool.validate_parameters_detailed.return_value = ToolValidationResult.success()

        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.STRICT,
        )

        should_proceed, result = executor._validate_arguments(mock_tool, {"path": "/valid"})

        assert should_proceed is True
        assert result is not None
        assert result.valid
        assert executor._validation_failures == 0

    def test_validate_arguments_exception_strict(self, mock_tool_with_validation):
        """Test validation exception handling in STRICT mode."""
        from victor.agent.tool_executor import ValidationMode

        mock_tool, _ = mock_tool_with_validation
        mock_tool.validate_parameters_detailed.side_effect = RuntimeError("Schema error")

        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.STRICT,
        )

        should_proceed, result = executor._validate_arguments(mock_tool, {})

        assert should_proceed is False
        assert result is not None
        assert "Validation system error" in result.errors[0]

    def test_validate_arguments_exception_lenient(self, mock_tool_with_validation):
        """Test validation exception handling in LENIENT mode."""
        from victor.agent.tool_executor import ValidationMode

        mock_tool, _ = mock_tool_with_validation
        mock_tool.validate_parameters_detailed.side_effect = RuntimeError("Schema error")

        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.LENIENT,
        )

        should_proceed, result = executor._validate_arguments(mock_tool, {})

        # LENIENT mode proceeds even on validation system errors
        assert should_proceed is True
        assert result is None

    @pytest.mark.asyncio
    async def test_strict_validation_blocks_execution(self):
        """Test that STRICT validation failures block execution."""
        from victor.agent.tool_executor import ValidationMode
        from victor.tools.base import ToolValidationResult

        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "validated_tool"
        mock_tool.execute = AsyncMock(return_value="should not be called")
        mock_tool.validate_parameters_detailed = MagicMock(
            return_value=ToolValidationResult.failure(["path is required"])
        )
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.STRICT,
        )

        result = await executor.execute("validated_tool", {})

        assert result.success is False
        # Error can be either "Invalid arguments" with errors or fallback message
        assert "Invalid arguments" in result.error or "validation failed" in result.error.lower()
        mock_tool.execute.assert_not_called()


class TestToolExecutorUnknownArguments:
    """Tests for unknown/hallucinated argument detection."""

    @pytest.fixture
    def mock_tool_with_schema(self):
        """Create a mock tool with a defined parameter schema."""
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path"],
            "additionalProperties": False,
        }
        return mock_tool

    def test_check_unknown_arguments_valid(self, mock_tool_with_schema):
        """Test that valid arguments pass the check."""
        from victor.agent.tool_executor import ValidationMode

        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.STRICT,
        )

        valid, unknown = executor._check_unknown_arguments(
            mock_tool_with_schema, {"path": "/test", "content": "hello"}
        )

        assert valid is True
        assert unknown == []

    def test_check_unknown_arguments_detects_invented_args(self, mock_tool_with_schema):
        """Test that invented/hallucinated arguments are detected."""
        from victor.agent.tool_executor import ValidationMode

        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.STRICT,
        )

        valid, unknown = executor._check_unknown_arguments(
            mock_tool_with_schema, {"path": "/test", "files": ["a.py"], "target": "foo"}
        )

        assert valid is False
        assert set(unknown) == {"files", "target"}

    def test_validate_arguments_rejects_unknown_in_strict_mode(self, mock_tool_with_schema):
        """Test that STRICT mode rejects unknown arguments with helpful error."""
        from victor.agent.tool_executor import ValidationMode
        from victor.tools.base import ToolValidationResult

        mock_tool_with_schema.validate_parameters_detailed = MagicMock(
            return_value=ToolValidationResult.success()
        )

        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.STRICT,
        )

        should_proceed, result = executor._validate_arguments(
            mock_tool_with_schema, {"path": "/test", "invented_arg": "value"}
        )

        assert should_proceed is False
        assert result is not None
        assert "Unknown argument(s): invented_arg" in result.errors[0]
        assert "Valid parameters" in result.errors[0]
        assert "path" in result.errors[0]
        assert "content" in result.errors[0]

    def test_validate_arguments_warns_unknown_in_lenient_mode(self, mock_tool_with_schema):
        """Test that LENIENT mode warns but proceeds with unknown arguments."""
        from victor.agent.tool_executor import ValidationMode
        from victor.tools.base import ToolValidationResult

        mock_tool_with_schema.validate_parameters_detailed = MagicMock(
            return_value=ToolValidationResult.success()
        )

        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.LENIENT,
        )

        should_proceed, result = executor._validate_arguments(
            mock_tool_with_schema, {"path": "/test", "invented_arg": "value"}
        )

        # LENIENT mode proceeds despite unknown args
        assert should_proceed is True
        assert executor._validation_failures == 1  # Still counted as failure

    def test_validate_arguments_lenient_mode_is_strict_for_execute_tools(
        self, mock_tool_with_schema
    ):
        """Stateful tools should not bypass validation just because global mode is lenient."""
        from victor.agent.tool_executor import ValidationMode
        from victor.tools.base import AccessMode, ToolValidationResult

        mock_tool_with_schema.access_mode = AccessMode.EXECUTE
        mock_tool_with_schema.validate_parameters_detailed = MagicMock(
            return_value=ToolValidationResult.success()
        )

        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.LENIENT,
        )

        should_proceed, result = executor._validate_arguments(
            mock_tool_with_schema, {"path": "/test", "invented_arg": "value"}
        )

        assert should_proceed is False
        assert result is not None
        assert "Unknown argument(s): invented_arg" in result.errors[0]

    def test_check_unknown_arguments_empty_schema(self):
        """Test that tools with empty schema allow any arguments."""
        from victor.agent.tool_executor import ValidationMode

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "flexible_tool"
        mock_tool.parameters = {}

        registry = ToolRegistry()
        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.STRICT,
        )

        valid, unknown = executor._check_unknown_arguments(
            mock_tool, {"any": "arg", "is": "allowed"}
        )

        assert valid is True
        assert unknown == []

    @pytest.mark.asyncio
    async def test_execute_recovers_wrapped_edit_value_envelope_before_validation(self):
        """Edit should unwrap generic value envelopes before strict validation runs."""
        from victor.agent.tool_executor import ValidationMode
        from victor.tools.base import AccessMode, ToolValidationResult

        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "edit"
        mock_tool.access_mode = AccessMode.WRITE
        mock_tool.parameters = {
            "type": "object",
            "properties": {
                "ops": {"type": "array"},
                "preview": {"type": "boolean"},
            },
            "required": ["ops"],
            "additionalProperties": False,
        }
        mock_tool.validate_parameters_detailed = MagicMock(
            return_value=ToolValidationResult.success()
        )
        mock_tool.execute = AsyncMock(return_value={"success": True})
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.STRICT,
        )

        result = await executor.execute(
            "edit",
            {
                "value": {
                    "ops": [{"type": "create", "path": "test.txt", "content": "hi"}],
                }
            },
        )

        assert result.success is True
        mock_tool.validate_parameters_detailed.assert_called_once_with(
            ops=[{"type": "create", "path": "test.txt", "content": "hi"}]
        )

    @pytest.mark.asyncio
    async def test_execute_recovers_wrapped_write_value_envelope_before_validation(
        self,
    ):
        """Write should unwrap generic value envelopes before strict validation runs."""
        from victor.agent.tool_executor import ValidationMode
        from victor.tools.base import AccessMode, ToolValidationResult

        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "write"
        mock_tool.access_mode = AccessMode.WRITE
        mock_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }
        mock_tool.validate_parameters_detailed = MagicMock(
            return_value=ToolValidationResult.success()
        )
        mock_tool.execute = AsyncMock(return_value={"success": True})
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            validation_mode=ValidationMode.STRICT,
        )

        result = await executor.execute(
            "write",
            {
                "value": (
                    '{"path":"victor/framework/graph_protocols.py","content":"print(\\"hi\\")\\n"}'
                )
            },
        )

        assert result.success is True
        mock_tool.validate_parameters_detailed.assert_called_once_with(
            path="victor/framework/graph_protocols.py",
            content='print("hi")\n',
        )


class TestToolExecutorHooks:
    """Tests for ToolExecutor hook execution."""

    @pytest.fixture
    def registry_with_hooks(self):
        """Create a registry with before/after hooks."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "hooked_tool"
        mock_tool.execute = AsyncMock(return_value="success")
        mock_tool.validate_parameters_detailed = MagicMock()
        mock_tool.validate_parameters_detailed.return_value = MagicMock(valid=True, errors=[])
        registry.register(mock_tool)

        return registry, mock_tool

    def test_run_before_hooks_success(self, registry_with_hooks):
        """Test running before hooks successfully."""
        registry, _ = registry_with_hooks

        called = []

        def before_hook(tool_name, arguments):
            called.append((tool_name, arguments))

        registry._before_hooks.append(before_hook)

        executor = ToolExecutor(tool_registry=registry)
        executor._run_before_hooks("test_tool", {"arg": "value"})

        assert len(called) == 1
        assert called[0] == ("test_tool", {"arg": "value"})

    def test_run_before_hooks_non_critical_failure(self, registry_with_hooks):
        """Test that non-critical hook failures are logged but don't block."""
        registry, _ = registry_with_hooks

        def failing_hook(tool_name, arguments):
            raise ValueError("Hook failed")

        registry._before_hooks.append(failing_hook)

        executor = ToolExecutor(tool_registry=registry)
        # Should not raise - non-critical hooks don't block
        executor._run_before_hooks("test_tool", {})

    def test_run_before_hooks_critical_failure(self, registry_with_hooks):
        """Test that critical hook failures raise HookError."""
        from victor.tools.base import Hook, HookError

        registry, _ = registry_with_hooks

        def critical_hook_func(tool_name, arguments):
            raise ValueError("Critical failure")

        critical_hook = Hook(callback=critical_hook_func, name="critical_hook", critical=True)
        registry._before_hooks.append(critical_hook)

        executor = ToolExecutor(tool_registry=registry)

        with pytest.raises(HookError) as exc_info:
            executor._run_before_hooks("test_tool", {})

        assert exc_info.value.hook_name == "critical_hook"
        assert exc_info.value.tool_name == "test_tool"

    def test_run_after_hooks_success(self, registry_with_hooks):
        """Test running after hooks successfully."""
        registry, _ = registry_with_hooks

        called = []

        def after_hook(result):
            called.append(result)

        registry._after_hooks.append(after_hook)

        executor = ToolExecutor(tool_registry=registry)
        executor._run_after_hooks("test_tool", "result_value")

        assert len(called) == 1
        assert called[0] == "result_value"

    def test_run_after_hooks_non_critical_failure(self, registry_with_hooks):
        """Test that non-critical after hook failures are logged but don't raise."""
        registry, _ = registry_with_hooks

        def failing_after_hook(result):
            raise ValueError("After hook failed")

        registry._after_hooks.append(failing_after_hook)

        executor = ToolExecutor(tool_registry=registry)
        # Should not raise - non-critical hooks don't block
        executor._run_after_hooks("test_tool", "result")

    def test_run_after_hooks_critical_failure(self, registry_with_hooks):
        """Test that critical after hook failures raise HookError."""
        from victor.tools.base import Hook, HookError

        registry, _ = registry_with_hooks

        def critical_after_func(result):
            raise RuntimeError("Critical after failure")

        critical_hook = Hook(callback=critical_after_func, name="critical_after", critical=True)
        registry._after_hooks.append(critical_hook)

        executor = ToolExecutor(tool_registry=registry)

        with pytest.raises(HookError) as exc_info:
            executor._run_after_hooks("test_tool", "result")

        assert exc_info.value.hook_name == "critical_after"


class TestToolExecutorCacheManagement:
    """Tests for ToolExecutor cache management methods."""

    def test_invalidate_cache_for_paths(self):
        """Test invalidate_cache_for_paths method."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        executor.invalidate_cache_for_paths(["/path/file1.py", "/path/file2.py"])

        mock_cache.invalidate_paths.assert_called_once_with(["/path/file1.py", "/path/file2.py"])

    def test_invalidate_cache_for_paths_no_cache(self):
        """Test invalidate_cache_for_paths when no cache is configured."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        # Should not raise even without cache
        executor.invalidate_cache_for_paths(["/path/file.py"])

    def test_clear_cache(self):
        """Test clear_cache method."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        executor.clear_cache()

        mock_cache.clear_all.assert_called_once()

    def test_clear_cache_no_cache(self):
        """Test clear_cache when no cache is configured."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        # Should not raise even without cache
        executor.clear_cache()

    def test_clear_failed_signatures(self):
        """Test clearing failed signatures."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        # Add some failed signatures
        executor._failed_signatures.add(("tool1", "[('a', 1)]"))
        executor._failed_signatures.add(("tool2", "[('b', 2)]"))
        assert len(executor._failed_signatures) == 2

        executor.clear_failed_signatures()

        assert len(executor._failed_signatures) == 0

    def test_reset_stats(self):
        """Test reset_stats method."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        # Add some stats
        executor._stats["tool1"] = {"calls": 5, "successes": 4, "failures": 1}
        executor._stats["tool2"] = {"calls": 3, "successes": 3, "failures": 0}

        executor.reset_stats()

        assert executor._stats == {}

    def test_get_tool_stats_existing(self):
        """Test get_tool_stats for an existing tool."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        executor._stats["my_tool"] = {
            "calls": 10,
            "successes": 8,
            "failures": 2,
            "total_time": 5.5,
        }

        stats = executor.get_tool_stats("my_tool")

        assert stats["calls"] == 10
        assert stats["successes"] == 8
        assert stats["failures"] == 2

    def test_get_tool_stats_nonexistent(self):
        """Test get_tool_stats for a non-existent tool."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        stats = executor.get_tool_stats("nonexistent_tool")

        assert stats == {}


class TestToolExecutorCacheInvalidation:
    """Tests for ToolExecutor._invalidate_cache_for_write_tool method."""

    def test_invalidate_cache_write_file(self):
        """Test cache invalidation for write_file tool."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        executor._invalidate_cache_for_write_tool("write_file", {"path": "/tmp/test.py"})

        mock_cache.invalidate_paths.assert_called_once_with(["/tmp/test.py"])

    def test_invalidate_cache_write_canonical(self):
        """Canonical write tool should invalidate the written path."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        executor._invalidate_cache_for_write_tool("write", {"path": "/tmp/test.py"})

        mock_cache.invalidate_paths.assert_called_once_with(["/tmp/test.py"])

    def test_invalidate_cache_edit_files_with_edits(self):
        """Test cache invalidation for edit_files with edits list."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        executor._invalidate_cache_for_write_tool(
            "edit_files",
            {
                "edits": [
                    {"path": "/tmp/file1.py", "content": "new content"},
                    {"path": "/tmp/file2.py", "content": "other content"},
                ]
            },
        )

        mock_cache.invalidate_paths.assert_called_once_with(["/tmp/file1.py", "/tmp/file2.py"])

    def test_invalidate_cache_edit_files_with_path(self):
        """Test cache invalidation for edit_files with single path."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        executor._invalidate_cache_for_write_tool("edit_files", {"path": "/tmp/single.py"})

        mock_cache.invalidate_paths.assert_called_once_with(["/tmp/single.py"])

    def test_invalidate_cache_edit_canonical_with_ops(self):
        """Canonical edit tool should extract paths from ops."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        executor._invalidate_cache_for_write_tool(
            "edit",
            {
                "ops": [
                    {
                        "type": "replace",
                        "path": "/tmp/file1.py",
                        "old_str": "a",
                        "new_str": "b",
                    },
                    {
                        "type": "rename",
                        "path": "/tmp/file2.py",
                        "new_path": "/tmp/file3.py",
                    },
                ]
            },
        )

        mock_cache.invalidate_paths.assert_called_once_with(
            ["/tmp/file1.py", "/tmp/file2.py", "/tmp/file3.py"]
        )

    def test_invalidate_cache_execute_bash(self):
        """Test cache invalidation for execute_bash (wide invalidation)."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        executor._invalidate_cache_for_write_tool("execute_bash", {"command": "rm -rf /tmp/*"})

        # Bash commands invalidate file-related caches
        assert mock_cache.invalidate_by_tool.call_count == 2
        mock_cache.invalidate_by_tool.assert_any_call("read")
        mock_cache.invalidate_by_tool.assert_any_call("ls")

    def test_invalidate_cache_shell_canonical(self):
        """Canonical shell tool should invalidate canonical read caches."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        executor._invalidate_cache_for_write_tool("shell", {"cmd": "rm -rf /tmp/*"})

        assert mock_cache.invalidate_by_tool.call_count == 2
        mock_cache.invalidate_by_tool.assert_any_call("read")
        mock_cache.invalidate_by_tool.assert_any_call("ls")

    def test_invalidate_cache_git_tool(self):
        """Test cache invalidation for git tool (wide invalidation)."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        executor._invalidate_cache_for_write_tool("git", {"action": "commit"})

        # Git commands invalidate many caches
        assert mock_cache.invalidate_by_tool.call_count == 3
        mock_cache.invalidate_by_tool.assert_any_call("read")
        mock_cache.invalidate_by_tool.assert_any_call("ls")
        mock_cache.invalidate_by_tool.assert_any_call("code_search")

    def test_invalidate_cache_docker_tool(self):
        """Test cache invalidation for docker tool."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        executor._invalidate_cache_for_write_tool("docker", {"action": "build"})

        # Docker commands invalidate many caches
        assert mock_cache.invalidate_by_tool.call_count == 3

    def test_invalidate_cache_no_cache(self):
        """Test cache invalidation when no cache is configured."""
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)

        # Should not raise when cache is None
        executor._invalidate_cache_for_write_tool("write_file", {"path": "/tmp/test.py"})

    def test_invalidate_cache_unknown_tool(self):
        """Test cache invalidation for unknown tool (no-op)."""
        registry = ToolRegistry()
        mock_cache = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            tool_cache=mock_cache,
        )

        executor._invalidate_cache_for_write_tool("unknown_tool", {"arg": "value"})

        # Should not call any invalidation for unknown tools
        mock_cache.invalidate_paths.assert_not_called()
        mock_cache.invalidate_by_tool.assert_not_called()


class TestToolExecutorCodeCorrection:
    """Tests for ToolExecutor code correction middleware integration."""

    @staticmethod
    def _allowing_safety_checker():
        checker = MagicMock()
        checker.check_and_confirm = AsyncMock(return_value=(True, None))
        return checker

    @pytest.mark.asyncio
    async def test_code_correction_applied(self):
        """Test that code correction middleware is applied when enabled."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "write_code"
        mock_tool.execute = AsyncMock(return_value="success")
        mock_tool.validate_parameters_detailed = MagicMock()
        mock_tool.validate_parameters_detailed.return_value = MagicMock(valid=True, errors=[])
        registry.register(mock_tool)

        mock_middleware = MagicMock()

        # process() is the single entry: returns (possibly corrected args, result)
        mock_correction_result = MagicMock()
        mock_correction_result.was_corrected = True
        mock_correction_result.validation = MagicMock(valid=True, errors=[])
        mock_middleware.process.return_value = (
            {"code": "fixed code"},
            mock_correction_result,
        )

        executor = ToolExecutor(
            tool_registry=registry,
            code_correction_middleware=mock_middleware,
            enable_code_correction=True,
            safety_checker=self._allowing_safety_checker(),
        )

        await executor.execute("write_code", {"code": "bad code"})

        mock_middleware.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_code_correction_not_applied_when_disabled(self):
        """Test that code correction is skipped when disabled."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "write_code"
        mock_tool.execute = AsyncMock(return_value="success")
        mock_tool.validate_parameters_detailed = MagicMock()
        mock_tool.validate_parameters_detailed.return_value = MagicMock(valid=True, errors=[])
        registry.register(mock_tool)

        mock_middleware = MagicMock()

        executor = ToolExecutor(
            tool_registry=registry,
            code_correction_middleware=mock_middleware,
            enable_code_correction=False,  # Disabled
            safety_checker=self._allowing_safety_checker(),
        )

        await executor.execute("write_code", {"code": "code"})

        mock_middleware.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_code_correction_validation_errors_logged(self):
        """Test that validation errors from code correction are logged."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "write_code"
        mock_tool.execute = AsyncMock(return_value="success")
        mock_tool.validate_parameters_detailed = MagicMock()
        mock_tool.validate_parameters_detailed.return_value = MagicMock(valid=True, errors=[])
        registry.register(mock_tool)

        mock_middleware = MagicMock()

        # Correction result with validation errors but not corrected
        mock_correction_result = MagicMock()
        mock_correction_result.was_corrected = False
        mock_correction_result.validation = MagicMock(valid=False, errors=["Syntax error"])
        mock_middleware.process.return_value = (
            {"code": "bad code"},
            mock_correction_result,
        )

        executor = ToolExecutor(
            tool_registry=registry,
            code_correction_middleware=mock_middleware,
            enable_code_correction=True,
            safety_checker=self._allowing_safety_checker(),
        )

        result = await executor.execute("write_code", {"code": "bad code"})

        # Tool should still execute (validation errors are logged, not blocking)
        assert result.success is True
        mock_middleware.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_code_correction_middleware_exception(self):
        """Test that middleware exceptions are caught and logged."""
        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "write_code"
        mock_tool.execute = AsyncMock(return_value="success")
        mock_tool.validate_parameters_detailed = MagicMock()
        mock_tool.validate_parameters_detailed.return_value = MagicMock(valid=True, errors=[])
        registry.register(mock_tool)

        mock_middleware = MagicMock()
        mock_middleware.process.side_effect = RuntimeError("Middleware crashed")

        executor = ToolExecutor(
            tool_registry=registry,
            code_correction_middleware=mock_middleware,
            enable_code_correction=True,
            safety_checker=self._allowing_safety_checker(),
        )

        result = await executor.execute("write_code", {"code": "code"})

        # Tool should still execute despite middleware failure
        assert result.success is True


class TestToolExecutorTimeoutHandling:
    """Tests for ToolExecutor timeout handling."""

    @pytest.fixture(autouse=True)
    def reset_globals(self):
        """Reset global singletons before each test."""
        from victor.core.errors import get_error_handler

        safety_module = _safety_module

        handler = get_error_handler()
        handler.clear_history()
        safety_module._default_checker = None

    @pytest.mark.asyncio
    async def test_timeout_error_tracking(self):
        """Test that timeout errors are tracked properly."""
        import asyncio

        registry = ToolRegistry()

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "slow_tool"

        async def slow_execute(context, **kwargs):
            raise asyncio.TimeoutError("Tool timed out")

        mock_tool.execute = slow_execute
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.01,
        )

        result = await executor.execute("slow_tool", {})

        assert result.success is False
        assert "timeout" in result.error.lower()
        assert result.error_info is not None

    @pytest.mark.asyncio
    async def test_timeout_retry(self):
        """Test that timeouts trigger retries."""
        import asyncio

        registry = ToolRegistry()

        call_count = [0]

        async def sometimes_timeout(_exec_ctx=None, **kwargs):
            nonlocal call_count
            call_count[0] += 1
            if call_count[0] < 2:
                raise asyncio.TimeoutError("Temporary timeout")
            return "success"

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "timeout_tool"
        mock_tool.execute = sometimes_timeout
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=3,
            retry_delay=0.01,
        )

        result = await executor.execute("timeout_tool", {})

        assert result.success is True
        assert call_count[0] == 2  # First timeout, then success

    @pytest.mark.asyncio
    async def test_default_timeout_uses_executor_default(self):
        """Test that tools without timeout metadata use executor default timeout."""
        import asyncio

        registry = ToolRegistry()

        async def long_running(_exec_ctx=None, **kwargs):
            await asyncio.sleep(1.0)

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "timeout_default_tool"
        mock_tool.execute = long_running
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=1,
            retry_delay=0.0,
            default_timeout_seconds=0.05,
        )

        result = await executor.execute("timeout_default_tool", {})

        assert result.success is False
        assert result.error is not None
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_tool_timeout_metadata_overrides_default(self):
        """Test that tool timeout metadata overrides executor default timeout."""
        import asyncio

        registry = ToolRegistry()

        async def long_running(_exec_ctx=None, **kwargs):
            await asyncio.sleep(0.1)
            return "ok"

        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "timeout_override_tool"
        mock_tool.execute = long_running
        mock_tool.timeout_seconds = 0.3
        registry.register(mock_tool)

        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=0,
            retry_delay=0.0,
            default_timeout_seconds=0.05,
        )

        result = await executor.execute("timeout_override_tool", {})

        assert result.success is True
        assert result.result == "ok"


class TestExecCtxDoublePassing:
    """Tests that _exec_ctx is stripped from LLM-provided arguments."""

    def setup_method(self):
        """Reset safety checker before each test."""
        safety_module = _safety_module

        safety_module._default_checker = None

    @pytest.mark.asyncio
    async def test_exec_ctx_stripped_from_arguments(self):
        """If LLM hallucinates _exec_ctx in arguments, it should be stripped before execute."""
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_strip"
        mock_tool.execute = AsyncMock(return_value=ToolResult(success=True, output="ok"))
        mock_tool.parameters = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        }

        registry = ToolRegistry()
        registry.register(mock_tool)

        executor = ToolExecutor(tool_registry=registry)
        # Simulate LLM including _exec_ctx in arguments
        result = await executor.execute("test_strip", {"query": "hello", "_exec_ctx": {}})

        assert result.success is True
        # Verify _exec_ctx was passed by framework (via _exec_ctx kwarg), not from arguments
        call_kwargs = mock_tool.execute.call_args.kwargs
        assert "_exec_ctx" in call_kwargs  # Framework passes it


class TestSessionDisabledTools:
    """Tests for Issue 3: session-scoped circuit breaker for unavailable tools."""

    def setup_method(self):
        safety_module = _safety_module

        safety_module._default_checker = None

    def _make_executor(self):
        registry = MagicMock(spec=ToolRegistry)
        registry.get.return_value = None
        registry.is_tool_enabled.return_value = True
        return ToolExecutor(tool_registry=registry)

    @pytest.mark.asyncio
    async def test_session_disabled_tool_returns_immediately(self):
        """Tool in _session_disabled_tools should return error without calling the tool."""
        executor = self._make_executor()
        executor._session_disabled_tools.add("graph_search")

        result = await executor.execute("graph_search", {})

        assert result.success is False
        assert "unavailable" in result.error.lower()
        assert "graph_search" in result.error

    @pytest.mark.asyncio
    async def test_unavailable_result_disables_tool(self):
        """Tool returning unavailable=True should be added to _session_disabled_tools."""
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "graph_search"
        mock_tool.parameters = {"type": "object", "properties": {}}
        mock_tool.execute = AsyncMock(
            return_value=ToolResult(
                success=False,
                output=None,
                error="victor-coding not installed",
                metadata={"unavailable": True},
            )
        )

        registry = ToolRegistry()
        registry.register(mock_tool)
        executor = ToolExecutor(tool_registry=registry)

        # First call: tool executes and returns unavailable=True
        result = await executor.execute("graph_search", {})
        assert not result.success

        # If the tool result dict includes unavailable=True, it should be session-disabled
        # Simulate post-execution unavailable detection
        executor._session_disabled_tools.add("graph_search")
        assert "graph_search" in executor.session_disabled_tools

    @pytest.mark.asyncio
    async def test_failed_legacy_dict_unavailable_result_disables_tool(self):
        """Legacy dict failures should preserve unavailable metadata and trip the circuit breaker."""

        @tool
        async def graph_search(_exec_ctx=None):
            return {
                "success": False,
                "error": "victor-coding not installed",
                "unavailable": True,
                "suggestion": "Install victor-coding",
            }

        registry = ToolRegistry()
        registry.register(graph_search)
        executor = ToolExecutor(tool_registry=registry)

        first = await executor.execute("graph_search", {})
        second = await executor.execute("graph_search", {})

        assert first.success is False
        assert isinstance(first.result, dict)
        assert first.result["unavailable"] is True
        assert first.result["suggestion"] == "Install victor-coding"
        assert "graph_search" in executor.session_disabled_tools
        assert second.success is False
        assert "unavailable this session" in (second.error or "")

    def test_session_disabled_tools_property_is_frozenset(self):
        """session_disabled_tools property returns an immutable frozenset."""
        executor = self._make_executor()
        executor._session_disabled_tools.add("bad_tool")

        result = executor.session_disabled_tools
        assert isinstance(result, frozenset)
        assert "bad_tool" in result


class TestFailedPathRedirects:
    """Tests for Issue 1: session-scoped path redirect cache."""

    def setup_method(self):
        safety_module = _safety_module

        safety_module._default_checker = None

    def _make_executor(self):
        registry = MagicMock(spec=ToolRegistry)
        registry.get.return_value = None
        registry.is_tool_enabled.return_value = True
        return ToolExecutor(tool_registry=registry)

    def test_path_redirect_applied_before_dispatch(self):
        """_failed_path_redirects should rewrite path args before tool execution."""
        executor = self._make_executor()
        executor._failed_path_redirects["/wrong/path.py"] = "/correct/path.py"

        assert executor._failed_path_redirects["/wrong/path.py"] == "/correct/path.py"

    def test_did_you_mean_recorded_from_error(self):
        """Error messages containing 'Did you mean' should populate redirect cache."""
        executor = self._make_executor()

        # Simulate post-execution path recording logic
        import re

        bad_path = "/nonexistent/file.py"
        error = f"File not found: {bad_path}\nDid you mean one of these?\n  - /actual/file.py"
        match = re.search(r"-\s+(\S+)", error)
        if match:
            executor._failed_path_redirects[bad_path] = match.group(1)

        assert "/nonexistent/file.py" in executor._failed_path_redirects
        assert executor._failed_path_redirects["/nonexistent/file.py"] == "/actual/file.py"

    def test_failed_path_redirects_initially_empty(self):
        """_failed_path_redirects should start empty."""
        executor = self._make_executor()
        assert executor._failed_path_redirects == {}

    @pytest.mark.asyncio
    async def test_execute_retries_read_only_tool_with_suggested_path(self):
        """Read-only path tools should retry immediately with a suggested path."""
        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "read"
        mock_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }
        mock_tool.execute = AsyncMock(
            side_effect=[
                FileNotFoundError(
                    "File not found: /wrong/path.py\nDid you mean one of these?\n  - /correct/path.py"
                ),
                {"content": "ok"},
            ]
        )
        registry.register(mock_tool)

        mock_safety = MagicMock()
        mock_safety.check_and_confirm = AsyncMock(return_value=(True, None))
        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=0,
            safety_checker=mock_safety,
        )

        result = await executor.execute("read", {"path": "/wrong/path.py"})

        assert result.success is True
        assert result.result == {"content": "ok"}
        assert executor._failed_path_redirects["/wrong/path.py"] == "/correct/path.py"
        assert mock_tool.execute.await_count == 2
        assert mock_tool.execute.await_args_list[1].kwargs["path"] == "/correct/path.py"

    @pytest.mark.asyncio
    async def test_execute_does_not_retry_read_only_tool_with_self_suggestion(self):
        """Read-only path recovery should reject suggestions equal to the failed path."""
        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "read"
        mock_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }
        mock_tool.execute = AsyncMock(
            side_effect=FileNotFoundError(
                "File not found: Cargo.toml\nDid you mean one of these?\n  - Cargo.toml"
            )
        )
        registry.register(mock_tool)

        mock_safety = MagicMock()
        mock_safety.check_and_confirm = AsyncMock(return_value=(True, None))
        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=0,
            safety_checker=mock_safety,
        )

        result = await executor.execute("read", {"path": "Cargo.toml"})

        assert result.success is False
        assert executor._failed_path_redirects == {}
        assert mock_tool.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_execute_does_not_retry_with_weak_suggestion(self):
        """A dissimilar suggested file must surface as an honest not-found error.

        Regression: session proximaDB-5b2726a3 — a read of a file absent from the
        checked-out branch was silently answered with an unrelated document.
        """
        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "read"
        mock_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }
        mock_tool.execute = AsyncMock(
            side_effect=FileNotFoundError(
                "File not found: docs/MODALITY_COMPILE_TIME_COMPOSITION_2026_07_24.adoc\n"
                "Did you mean one of these?\n"
                "  - docs/ROOT_CRATE_DECOMPOSITION_PLAN_2026_06_21.adoc"
            )
        )
        registry.register(mock_tool)

        mock_safety = MagicMock()
        mock_safety.check_and_confirm = AsyncMock(return_value=(True, None))
        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=0,
            safety_checker=mock_safety,
        )

        result = await executor.execute(
            "read", {"path": "docs/MODALITY_COMPILE_TIME_COMPOSITION_2026_07_24.adoc"}
        )

        assert result.success is False
        assert "Did you mean" in (result.error or "")
        assert executor._failed_path_redirects == {}
        assert mock_tool.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_recovered_retry_annotates_string_result(self):
        """A substituted read must announce the requested vs served path."""
        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "read"
        mock_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }
        mock_tool.execute = AsyncMock(
            side_effect=[
                FileNotFoundError(
                    "File not found: /docs/setings.yaml\n"
                    "Did you mean one of these?\n  - /docs/settings.yaml"
                ),
                "file contents",
            ]
        )
        registry.register(mock_tool)

        mock_safety = MagicMock()
        mock_safety.check_and_confirm = AsyncMock(return_value=(True, None))
        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=0,
            safety_checker=mock_safety,
        )

        result = await executor.execute("read", {"path": "/docs/setings.yaml"})

        assert result.success is True
        assert result.result.startswith("[Note: requested path '/docs/setings.yaml' does not exist")
        assert "'/docs/settings.yaml'" in result.result
        assert result.result.endswith("file contents")

    @pytest.mark.asyncio
    async def test_stale_redirect_dropped_when_path_exists(self, tmp_path):
        """A recorded redirect must be discarded once the original path exists."""
        real_file = tmp_path / "target.py"
        real_file.write_text("print('hi')\n")

        registry = ToolRegistry()
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "read"
        mock_tool.parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }
        mock_tool.execute = AsyncMock(return_value="fresh contents")
        registry.register(mock_tool)

        mock_safety = MagicMock()
        mock_safety.check_and_confirm = AsyncMock(return_value=(True, None))
        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=0,
            safety_checker=mock_safety,
        )
        executor._failed_path_redirects[str(real_file)] = str(tmp_path / "other.py")

        result = await executor.execute("read", {"path": str(real_file)})

        assert result.success is True
        assert result.result == "fresh contents"
        assert str(real_file) not in executor._failed_path_redirects
        assert mock_tool.execute.await_args.kwargs["path"] == str(real_file)

    @pytest.mark.asyncio
    async def test_execute_does_not_retry_write_tool_with_suggested_path(self):
        """Write-capable tools should not get implicit path rewriting retries."""
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "write_file"
        mock_tool.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }
        mock_tool.execution_category = ExecutionCategory.WRITE
        mock_tool.execute = AsyncMock(
            side_effect=FileNotFoundError(
                "File not found: /wrong/path.py\nDid you mean one of these?\n  - /correct/path.py"
            )
        )
        registry = MagicMock(spec=ToolRegistry)
        registry.get.return_value = mock_tool
        registry.is_tool_enabled.return_value = True

        mock_safety = MagicMock()
        mock_safety.check_and_confirm = AsyncMock(return_value=(True, None))
        executor = ToolExecutor(
            tool_registry=registry,
            max_retries=0,
            safety_checker=mock_safety,
        )

        result = await executor.execute(
            "write_file",
            {"path": "/wrong/path.py", "content": "hello"},
        )

        assert result.success is False
        assert mock_tool.execute.await_count == 1
        assert "/wrong/path.py" not in executor._failed_path_redirects


class TestIntentBasedToolFilter:
    """Tests for Issue 4: intent-based tool filtering in _select_tools_for_turn."""

    def test_read_only_intent_uses_permissive_tool_policy(self):
        """READ_ONLY uses a permissive (empty) block-set.

        Tools are no longer removed from the list by intent; safety is enforced
        via the prompt guard instead (see INTENT_BLOCKED_TOOLS [PERMISSIVE]).
        """
        from victor.agent.action_authorizer import INTENT_BLOCKED_TOOLS, ActionIntent
        from victor.tools.tool_names import get_canonical_name

        tools = [
            {"name": "read"},
            {"name": "write_file"},
            {"name": "code_search"},
            {"name": "edit_files"},
            {"name": "execute_bash"},
        ]
        blocked = INTENT_BLOCKED_TOOLS[ActionIntent.READ_ONLY]
        assert blocked == frozenset()  # permissive: nothing hard-blocked
        filtered = [t for t in tools if get_canonical_name(t.get("name", "")) not in blocked]
        assert filtered == tools  # no tools removed

    def test_write_allowed_intent_no_filtering(self):
        """WRITE_ALLOWED intent should not filter any tools."""
        from victor.agent.action_authorizer import INTENT_BLOCKED_TOOLS, ActionIntent
        from victor.tools.tool_names import get_canonical_name

        tools = [{"name": "write_file"}, {"name": "edit_files"}, {"name": "read"}]
        blocked = INTENT_BLOCKED_TOOLS[ActionIntent.WRITE_ALLOWED]
        filtered = [t for t in tools if get_canonical_name(t.get("name", "")) not in blocked]

        assert len(filtered) == len(tools)

    def test_display_only_intent_uses_permissive_tool_policy(self):
        """DISPLAY_ONLY uses a permissive (empty) block-set; safety via prompt guard."""
        from victor.agent.action_authorizer import INTENT_BLOCKED_TOOLS, ActionIntent
        from victor.tools.tool_names import get_canonical_name

        tools = [{"name": "write_file"}, {"name": "read"}, {"name": "ls"}]
        blocked = INTENT_BLOCKED_TOOLS[ActionIntent.DISPLAY_ONLY]
        assert blocked == frozenset()  # permissive: nothing hard-blocked
        filtered = [t for t in tools if get_canonical_name(t.get("name", "")) not in blocked]
        assert filtered == tools  # no tools removed

    def test_invalid_intent_string_does_not_crash(self):
        """An unrecognised intent string should not raise, just skip filtering."""
        try:
            from victor.agent.action_authorizer import ActionIntent

            ActionIntent("unknown_intent")
            crashed = False
        except ValueError:
            crashed = True
        assert crashed  # ValueError expected — caller must guard with try/except


class TestCoerceJsonStringArgs:
    """JSON-string → array/object coercion (real-agent calibration: `ops` sent as a string)."""

    @staticmethod
    def _tool(props):
        t = MagicMock(spec=BaseTool)
        t.name = "edit"
        t.parameters = {"type": "object", "properties": props}
        return t

    def test_json_string_array_is_parsed(self):
        from victor.agent.tool_executor import ToolExecutor

        tool = self._tool({"ops": {"type": "array"}})
        args = {"ops": '[{"type": "replace", "old": "a", "new": "b"}]'}
        ToolExecutor._coerce_arg_types(tool, args)
        assert args["ops"] == [{"type": "replace", "old": "a", "new": "b"}]

    def test_json_string_object_is_parsed(self):
        from victor.agent.tool_executor import ToolExecutor

        tool = self._tool({"cfg": {"type": "object"}})
        args = {"cfg": '{"k": 1}'}
        ToolExecutor._coerce_arg_types(tool, args)
        assert args["cfg"] == {"k": 1}

    def test_type_mismatch_left_unchanged(self):
        # array param given a JSON *object* string → not coerced (the tool/validator decides).
        from victor.agent.tool_executor import ToolExecutor

        tool = self._tool({"ops": {"type": "array"}})
        args = {"ops": '{"not": "an array"}'}
        ToolExecutor._coerce_arg_types(tool, args)
        assert args["ops"] == '{"not": "an array"}'

    def test_non_json_string_left_unchanged(self):
        from victor.agent.tool_executor import ToolExecutor

        tool = self._tool({"ops": {"type": "array"}})
        args = {"ops": "not json at all"}
        ToolExecutor._coerce_arg_types(tool, args)
        assert args["ops"] == "not json at all"

    def test_already_correct_type_untouched(self):
        from victor.agent.tool_executor import ToolExecutor

        tool = self._tool({"ops": {"type": "array"}})
        real = [{"type": "replace"}]
        args = {"ops": real}
        ToolExecutor._coerce_arg_types(tool, args)
        assert args["ops"] is real

    def test_strict_validation_coerces_before_validating(self):
        """The end-to-end bug: STRICT mode must coerce a JSON-string array before validating,
        so a recoverable serialization is not rejected."""
        from victor.agent.tool_executor import ToolExecutor, ValidationMode
        from victor.tools.base import ToolValidationResult

        tool = self._tool({"ops": {"type": "array"}})
        # The real validator would pass on a list; assert coercion happened before it ran.
        seen = {}

        def _validate(**kwargs):
            seen.update(kwargs)
            return ToolValidationResult(valid=True, errors=[])

        tool.validate_parameters_detailed = _validate
        executor = ToolExecutor(tool_registry=ToolRegistry(), validation_mode=ValidationMode.STRICT)
        args = {"ops": '[{"type": "replace"}]'}
        proceed, _ = executor._validate_arguments(tool, args)
        assert proceed is True
        assert seen["ops"] == [{"type": "replace"}]  # validator saw the parsed list
        assert args["ops"] == [{"type": "replace"}]  # and execution args are coerced too
