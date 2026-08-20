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

"""Tests for centralized error handling module."""

import threading

import pytest
from datetime import datetime, timezone

from victor.core.errors import (
    ErrorCategory,
    ErrorSeverity,
    VictorError,
    ProviderError,
    ProviderConnectionError,
    ProviderAuthError,
    ProviderRateLimitError,
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolValidationError,
    ToolTimeoutError,
    ConfigurationError,
    ValidationError,
    FileError,
    FileNotFoundError as VictorFileNotFoundError,
    NetworkError,
    ErrorInfo,
    ErrorHandler,
    handle_errors,
    handle_errors_async,
    get_error_handler,
    handle_exception,
)

# =============================================================================
# ERROR CATEGORY TESTS
# =============================================================================


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_provider_categories(self):
        """Test provider error categories exist."""
        assert ErrorCategory.PROVIDER_CONNECTION.value == "provider_connection"
        assert ErrorCategory.PROVIDER_AUTH.value == "provider_auth"
        assert ErrorCategory.PROVIDER_RATE_LIMIT.value == "provider_rate_limit"
        assert ErrorCategory.PROVIDER_INVALID_RESPONSE.value == "provider_invalid_response"

    def test_tool_categories(self):
        """Test tool error categories exist."""
        assert ErrorCategory.TOOL_NOT_FOUND.value == "tool_not_found"
        assert ErrorCategory.TOOL_EXECUTION.value == "tool_execution"
        assert ErrorCategory.TOOL_VALIDATION.value == "tool_validation"
        assert ErrorCategory.TOOL_TIMEOUT.value == "tool_timeout"

    def test_config_categories(self):
        """Test configuration error categories exist."""
        assert ErrorCategory.CONFIG_INVALID.value == "config_invalid"
        assert ErrorCategory.CONFIG_MISSING.value == "config_missing"

    def test_resource_categories(self):
        """Test resource error categories exist."""
        assert ErrorCategory.FILE_NOT_FOUND.value == "file_not_found"
        assert ErrorCategory.FILE_PERMISSION.value == "file_permission"
        assert ErrorCategory.NETWORK_ERROR.value == "network_error"
        assert ErrorCategory.RESOURCE_EXHAUSTED.value == "resource_exhausted"


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_all_severities(self):
        """Test all severity levels exist."""
        assert ErrorSeverity.DEBUG.value == "debug"
        assert ErrorSeverity.INFO.value == "info"
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.ERROR.value == "error"
        assert ErrorSeverity.CRITICAL.value == "critical"


# =============================================================================
# VICTOR ERROR TESTS
# =============================================================================


class TestVictorError:
    """Tests for VictorError base class."""

    def test_basic_creation(self):
        """Test basic error creation."""
        error = VictorError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.category == ErrorCategory.UNKNOWN
        assert error.severity == ErrorSeverity.ERROR
        assert error.correlation_id is not None
        assert len(error.correlation_id) == 8

    def test_with_all_parameters(self):
        """Test error with all parameters."""
        error = VictorError(
            message="Custom error",
            category=ErrorCategory.TOOL_EXECUTION,
            severity=ErrorSeverity.WARNING,
            details={"key": "value"},
            recovery_hint="Try again",
            correlation_id="test123",
            cause=ValueError("Original error"),
        )
        assert error.message == "Custom error"
        assert error.category == ErrorCategory.TOOL_EXECUTION
        assert error.severity == ErrorSeverity.WARNING
        assert error.details == {"key": "value"}
        assert error.recovery_hint == "Try again"
        assert error.correlation_id == "test123"
        assert isinstance(error.cause, ValueError)
        assert isinstance(error.timestamp, datetime)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        error = VictorError(
            message="Test error",
            category=ErrorCategory.VALIDATION_ERROR,
            recovery_hint="Check inputs",
        )
        d = error.to_dict()

        assert d["error"] == "Test error"
        assert d["category"] == "validation_error"
        assert d["severity"] == "error"
        assert d["recovery_hint"] == "Check inputs"
        assert "correlation_id" in d
        assert "timestamp" in d

    def test_str_representation(self):
        """Test string representation."""
        error = VictorError("Test", correlation_id="abc12345")
        assert "[abc12345] Test" in str(error)

    def test_str_with_recovery_hint(self):
        """Test string representation with recovery hint."""
        error = VictorError("Test error", recovery_hint="Fix it", correlation_id="xyz")
        result = str(error)
        assert "[xyz] Test error" in result
        assert "Recovery hint: Fix it" in result


# =============================================================================
# PROVIDER ERROR TESTS
# =============================================================================


class TestProviderError:
    """Tests for ProviderError class."""

    def test_basic_creation(self):
        """Test basic provider error."""
        error = ProviderError("Provider failed", provider="openai", model="gpt-4")
        assert error.message == "Provider failed"
        assert error.provider == "openai"
        assert error.model == "gpt-4"
        assert error.details["provider"] == "openai"
        assert error.details["model"] == "gpt-4"


class TestProviderConnectionError:
    """Tests for ProviderConnectionError class."""

    def test_creation(self):
        """Test connection error creation."""
        error = ProviderConnectionError("Connection failed", provider="anthropic")
        assert error.category == ErrorCategory.PROVIDER_CONNECTION
        assert error.provider == "anthropic"
        assert "network connection" in error.recovery_hint.lower()


class TestProviderAuthError:
    """Tests for ProviderAuthError class."""

    def test_creation(self):
        """Test auth error creation."""
        error = ProviderAuthError("Invalid API key", provider="openai")
        assert error.category == ErrorCategory.PROVIDER_AUTH
        assert error.provider == "openai"
        assert "api key" in error.recovery_hint.lower()


class TestProviderRateLimitError:
    """Tests for ProviderRateLimitError class."""

    def test_with_retry_after(self):
        """Test rate limit error with retry time."""
        error = ProviderRateLimitError("Rate limited", provider="anthropic", retry_after=60)
        assert error.category == ErrorCategory.PROVIDER_RATE_LIMIT
        assert error.retry_after == 60
        assert "60 seconds" in error.recovery_hint

    def test_without_retry_after(self):
        """Test rate limit error without retry time."""
        error = ProviderRateLimitError("Rate limited", provider="openai")
        assert error.retry_after is None
        assert "retry later" in error.recovery_hint.lower()


# =============================================================================
# TOOL ERROR TESTS
# =============================================================================


class TestToolError:
    """Tests for ToolError class."""

    def test_basic_creation(self):
        """Test basic tool error."""
        error = ToolError("Tool failed", tool_name="read_file")
        assert error.message == "Tool failed"
        assert error.tool_name == "read_file"
        assert error.details["tool_name"] == "read_file"


class TestToolNotFoundError:
    """Tests for ToolNotFoundError class."""

    def test_creation(self):
        """Test tool not found error."""
        error = ToolNotFoundError("unknown_tool")
        assert error.category == ErrorCategory.TOOL_NOT_FOUND
        assert "unknown_tool" in error.message
        assert error.tool_name == "unknown_tool"
        assert "list_tools" in error.recovery_hint


class TestToolExecutionError:
    """Tests for ToolExecutionError class."""

    def test_creation(self):
        """Test execution error creation."""
        error = ToolExecutionError("Execution failed", tool_name="bash")
        assert error.category == ErrorCategory.TOOL_EXECUTION
        assert error.tool_name == "bash"


class TestToolValidationError:
    """Tests for ToolValidationError class."""

    def test_with_invalid_args(self):
        """Test validation error with invalid args."""
        error = ToolValidationError(
            "Invalid arguments",
            tool_name="read_file",
            invalid_args=["path", "encoding"],
        )
        assert error.category == ErrorCategory.TOOL_VALIDATION
        assert error.invalid_args == ["path", "encoding"]
        assert error.details["invalid_args"] == ["path", "encoding"]


class TestToolTimeoutError:
    """Tests for ToolTimeoutError class."""

    def test_with_timeout(self):
        """Test timeout error with duration."""
        error = ToolTimeoutError(tool_name="long_operation", timeout=30)
        assert error.category == ErrorCategory.TOOL_TIMEOUT
        assert error.timeout == 30
        assert "30 seconds" in error.message

    def test_without_timeout(self):
        """Test timeout error without duration."""
        error = ToolTimeoutError(tool_name="operation")
        assert "timed out" in error.message.lower()


# =============================================================================
# CONFIG AND VALIDATION ERROR TESTS
# =============================================================================


class TestConfigurationError:
    """Tests for ConfigurationError class."""

    def test_creation(self):
        """Test configuration error."""
        error = ConfigurationError("Invalid config", config_key="api_key")
        assert error.category == ErrorCategory.CONFIG_INVALID
        assert error.config_key == "api_key"
        assert error.details["config_key"] == "api_key"


class TestValidationError:
    """Tests for ValidationError class."""

    def test_creation(self):
        """Test validation error."""
        error = ValidationError("Invalid value", field="temperature", value=2.5)
        assert error.category == ErrorCategory.VALIDATION_ERROR
        assert error.field == "temperature"
        assert error.value == 2.5
        assert error.details["value"] == "2.5"


# =============================================================================
# FILE AND NETWORK ERROR TESTS
# =============================================================================


class TestFileError:
    """Tests for FileError class."""

    def test_creation(self):
        """Test file error."""
        error = FileError("File operation failed", path="/test/file.txt")
        assert error.path == "/test/file.txt"
        assert error.details["path"] == "/test/file.txt"


class TestVictorFileNotFoundError:
    """Tests for FileNotFoundError class."""

    def test_creation(self):
        """Test file not found error."""
        error = VictorFileNotFoundError("/missing/file.txt")
        assert error.category == ErrorCategory.FILE_NOT_FOUND
        assert error.path == "/missing/file.txt"
        assert "File not found" in error.message


class TestNetworkError:
    """Tests for NetworkError class."""

    def test_creation(self):
        """Test network error."""
        error = NetworkError("Connection refused", url="http://api.example.com")
        assert error.category == ErrorCategory.NETWORK_ERROR
        assert error.url == "http://api.example.com"
        assert "network connection" in error.recovery_hint.lower()


# =============================================================================
# ERROR INFO TESTS
# =============================================================================


class TestErrorInfo:
    """Tests for ErrorInfo dataclass."""

    def test_creation(self):
        """Test error info creation."""
        info = ErrorInfo(
            message="Test error",
            category=ErrorCategory.TOOL_EXECUTION,
            severity=ErrorSeverity.WARNING,
            correlation_id="test123",
        )
        assert info.message == "Test error"
        assert info.category == ErrorCategory.TOOL_EXECUTION
        assert info.severity == ErrorSeverity.WARNING
        assert info.correlation_id == "test123"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = ErrorInfo(
            message="Test",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.ERROR,
            correlation_id="abc",
            recovery_hint="Fix it",
        )
        d = info.to_dict()

        assert d["message"] == "Test"
        assert d["category"] == "unknown"
        assert d["severity"] == "error"
        assert d["correlation_id"] == "abc"
        assert d["recovery_hint"] == "Fix it"

    def test_to_user_message(self):
        """Test user message generation."""
        info = ErrorInfo(
            message="Operation failed",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.ERROR,
            correlation_id="xyz",
            recovery_hint="Try again later",
        )
        msg = info.to_user_message()

        assert "Operation failed" in msg
        assert "Try again later" in msg

    def test_to_user_message_without_hint(self):
        """Test user message without recovery hint."""
        info = ErrorInfo(
            message="Error occurred",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.ERROR,
            correlation_id="test",
        )
        msg = info.to_user_message()

        assert msg == "Error occurred"


# =============================================================================
# ERROR HANDLER TESTS
# =============================================================================


class TestErrorHandler:
    """Tests for ErrorHandler class."""

    def test_basic_creation(self):
        """Test handler creation."""
        handler = ErrorHandler()
        assert handler.include_traceback is True
        assert len(handler.get_recent_errors()) == 0

    def test_handle_victor_error(self):
        """Test handling VictorError."""
        handler = ErrorHandler()
        error = ToolNotFoundError("missing_tool")

        info = handler.handle(error)

        assert info.message == "Tool not found: missing_tool"
        assert info.category == ErrorCategory.TOOL_NOT_FOUND
        assert len(handler.get_recent_errors()) == 1

    def test_handle_standard_exception(self):
        """Test handling standard exceptions."""
        handler = ErrorHandler()
        error = ValueError("Invalid value")

        info = handler.handle(error, context={"operation": "test"})

        assert info.category == ErrorCategory.VALIDATION_ERROR
        assert info.details["operation"] == "test"

    def test_handle_file_not_found(self):
        """Test handling FileNotFoundError."""
        handler = ErrorHandler()
        error = FileNotFoundError("File missing")

        info = handler.handle(error)

        assert info.category == ErrorCategory.FILE_NOT_FOUND

    def test_handle_permission_error(self):
        """Test handling PermissionError."""
        handler = ErrorHandler()
        error = PermissionError("Access denied")

        info = handler.handle(error)

        assert info.category == ErrorCategory.FILE_PERMISSION

    def test_handle_connection_error(self):
        """Test handling ConnectionError."""
        handler = ErrorHandler()
        error = ConnectionError("Connection refused")

        info = handler.handle(error)

        assert info.category == ErrorCategory.NETWORK_ERROR

    def test_handle_key_error(self):
        """Test handling KeyError."""
        handler = ErrorHandler()
        error = KeyError("missing_key")

        info = handler.handle(error)

        assert info.category == ErrorCategory.CONFIG_MISSING

    def test_history_limit(self):
        """Test error history limit."""
        handler = ErrorHandler()
        handler._max_history = 5

        for i in range(10):
            handler.handle(ValueError(f"Error {i}"))

        assert len(handler.get_recent_errors()) == 5

    def test_clear_history(self):
        """Test clearing error history."""
        handler = ErrorHandler()
        handler.handle(ValueError("Test"))

        handler.clear_history()

        assert len(handler.get_recent_errors()) == 0

    def test_get_recent_errors(self):
        """Test getting recent errors."""
        handler = ErrorHandler()
        handler.handle(ValueError("Error 1"))
        handler.handle(ValueError("Error 2"))
        handler.handle(ValueError("Error 3"))

        recent = handler.get_recent_errors(2)

        assert len(recent) == 2
        assert recent[-1].message == "Error 3"


# =============================================================================
# ERROR HANDLER THREAD SAFETY TESTS
# =============================================================================


class TestErrorHandlerThreadSafety:
    """Tests for ErrorHandler thread safety."""

    def test_concurrent_error_handling(self):
        """Test that concurrent error handling does not corrupt history."""
        handler = ErrorHandler()
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            for _ in range(100):
                handler.handle(Exception("test error"))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(handler.get_recent_errors(100)) <= 100


# =============================================================================
# DECORATOR TESTS
# =============================================================================


class TestHandleErrorsDecorator:
    """Tests for handle_errors decorator."""

    def test_successful_function(self):
        """Test decorator on successful function."""

        @handle_errors()
        def success_func():
            return "success"

        result = success_func()
        assert result == "success"

    def test_function_with_error_default_return(self):
        """Test decorator returns default on error."""

        @handle_errors(default_return="default")
        def error_func():
            raise ValueError("Test error")

        result = error_func()
        assert result == "default"

    def test_function_with_error_reraise(self):
        """Test decorator reraises as VictorError."""

        @handle_errors(category=ErrorCategory.TOOL_EXECUTION, reraise=True)
        def error_func():
            raise ValueError("Original error")

        with pytest.raises(VictorError) as exc_info:
            error_func()

        assert exc_info.value.category == ErrorCategory.TOOL_EXECUTION

    def test_victor_error_passes_through(self):
        """Test VictorError is not wrapped."""

        @handle_errors(reraise=True)
        def error_func():
            raise ToolNotFoundError("test_tool")

        with pytest.raises(ToolNotFoundError):
            error_func()


class TestHandleErrorsAsyncDecorator:
    """Tests for handle_errors_async decorator."""

    @pytest.mark.asyncio
    async def test_successful_async_function(self):
        """Test decorator on successful async function."""

        @handle_errors_async()
        async def success_func():
            return "async_success"

        result = await success_func()
        assert result == "async_success"

    @pytest.mark.asyncio
    async def test_async_function_with_error_default(self):
        """Test decorator returns default on async error."""

        @handle_errors_async(default_return="async_default")
        async def error_func():
            raise ValueError("Async error")

        result = await error_func()
        assert result == "async_default"

    @pytest.mark.asyncio
    async def test_async_function_with_reraise(self):
        """Test decorator reraises async error."""

        @handle_errors_async(category=ErrorCategory.NETWORK_ERROR, reraise=True)
        async def error_func():
            raise ConnectionError("Connection lost")

        with pytest.raises(VictorError) as exc_info:
            await error_func()

        assert exc_info.value.category == ErrorCategory.NETWORK_ERROR


# =============================================================================
# GLOBAL HANDLER TESTS
# =============================================================================


class TestGlobalHandler:
    """Tests for global error handler functions."""

    def test_get_error_handler_singleton(self):
        """Test global handler is singleton."""
        handler1 = get_error_handler()
        handler2 = get_error_handler()

        # Both should return an ErrorHandler instance
        assert isinstance(handler1, ErrorHandler)
        assert isinstance(handler2, ErrorHandler)

    def test_handle_exception_function(self):
        """Test handle_exception convenience function."""
        error = ValueError("Test error")

        info = handle_exception(error, context={"test": True})

        assert info.message == "Test error"
        assert info.details.get("test") is True
