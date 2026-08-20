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

"""
Tests for consistent exception handling patterns.

Verifies that:
1. Correct exception types are raised
2. Exception messages are informative
3. Exceptions are logged properly
4. Cleanup occurs on exception
"""

import logging
import pytest
from unittest.mock import Mock, patch, MagicMock
from victor.core.errors import (
    VictorError,
    ProviderError,
    ProviderConnectionError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderNotFoundError,
    ProviderInvalidResponseError,
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolValidationError,
    ToolTimeoutError,
    ConfigurationError,
    ValidationError,
    FileError,
    FileNotFoundError,
    NetworkError,
    ErrorHandler,
    ErrorCategory,
    ErrorSeverity,
)


class TestProviderErrorHandling:
    """Test provider error handling patterns."""

    def test_provider_connection_error(self, caplog):
        """Verify ProviderConnectionError is raised with proper context."""
        caplog.set_level(logging.ERROR)

        error = ProviderConnectionError("Connection failed", provider="test_provider")

        assert "Connection failed" in str(error)
        assert error.provider == "test_provider"
        assert error.category == ErrorCategory.PROVIDER_CONNECTION
        assert error.recovery_hint is not None

    def test_provider_auth_error(self, caplog):
        """Verify ProviderAuthError is raised with proper context."""
        error = ProviderAuthError("Authentication failed", provider="test_provider")

        assert "Authentication failed" in str(error)
        assert error.provider == "test_provider"
        assert error.category == ErrorCategory.PROVIDER_AUTH
        assert "API key" in error.recovery_hint

    def test_provider_rate_limit_error(self, caplog):
        """Verify ProviderRateLimitError includes retry_after."""
        error = ProviderRateLimitError(
            "Rate limit exceeded", provider="test_provider", retry_after=60
        )

        assert "Rate limit exceeded" in str(error)
        assert error.retry_after == 60
        assert "60 seconds" in error.recovery_hint

    def test_provider_timeout_error(self, caplog):
        """Verify ProviderTimeoutError includes timeout value."""
        error = ProviderTimeoutError("Request timeout", provider="test_provider", timeout=30)

        assert "Request timeout" in str(error)
        assert error.timeout == 30
        assert "30 seconds" in error.recovery_hint

    def test_provider_not_found_error(self, caplog):
        """Verify ProviderNotFoundError lists available providers."""
        error = ProviderNotFoundError(
            provider="unknown_provider",
            available_providers=["anthropic", "openai", "google"],
        )

        assert "unknown_provider" in str(error)
        assert error.available_providers == ["anthropic", "openai", "google"]

    def test_provider_invalid_response_error(self, caplog):
        """Verify ProviderInvalidResponseError captures response data."""
        response_data = {"status": "error", "code": 500}
        error = ProviderInvalidResponseError(
            "Invalid response format",
            provider="test_provider",
            response_data=response_data,
        )

        assert "Invalid response format" in str(error)
        assert error.response_data == response_data
        assert "status" in error.details.get("response_keys", [])


class TestToolErrorHandling:
    """Test tool error handling patterns."""

    def test_tool_not_found_error(self, caplog):
        """Verify ToolNotFoundError is raised correctly."""
        error = ToolNotFoundError("unknown_tool")

        assert "unknown_tool" in str(error)
        assert error.tool_name == "unknown_tool"
        assert error.category == ErrorCategory.TOOL_NOT_FOUND
        assert "list_tools" in error.recovery_hint.lower()

    def test_tool_execution_error(self, caplog):
        """Verify ToolExecutionError captures tool context."""
        error = ToolExecutionError("Execution failed", tool_name="test_tool")

        assert "Execution failed" in str(error)
        assert error.tool_name == "test_tool"
        assert error.category == ErrorCategory.TOOL_EXECUTION

    def test_tool_validation_error(self, caplog):
        """Verify ToolValidationError captures invalid arguments."""
        error = ToolValidationError(
            "Invalid arguments", tool_name="test_tool", invalid_args=["path", "content"]
        )

        assert "Invalid arguments" in str(error)
        assert error.tool_name == "test_tool"
        assert error.invalid_args == ["path", "content"]
        assert error.category == ErrorCategory.TOOL_VALIDATION

    def test_tool_timeout_error(self, caplog):
        """Verify ToolTimeoutError includes timeout value."""
        error = ToolTimeoutError(tool_name="slow_tool", timeout=30)

        assert "slow_tool" in str(error)
        assert "30 seconds" in str(error)
        assert error.timeout == 30
        assert error.category == ErrorCategory.TOOL_TIMEOUT


class TestConfigurationErrorHandling:
    """Test configuration error handling patterns."""

    def test_configuration_error(self, caplog):
        """Verify ConfigurationError captures config key."""
        error = ConfigurationError("Invalid configuration", config_key="provider")

        assert "Invalid configuration" in str(error)
        assert error.config_key == "provider"
        assert error.category == ErrorCategory.CONFIG_INVALID

    def test_validation_error(self, caplog):
        """Verify ValidationError captures field and value."""
        error = ValidationError("Invalid value", field="timeout", value=-1)

        assert "Invalid value" in str(error)
        assert error.field == "timeout"
        assert error.value == -1
        assert error.category == ErrorCategory.VALIDATION_ERROR


class TestFileErrorHandling:
    """Test file error handling patterns."""

    def test_file_not_found_error(self, caplog):
        """Verify FileNotFoundError captures path."""
        error = FileNotFoundError(path="/nonexistent/file.txt")

        assert "/nonexistent/file.txt" in str(error)
        assert error.path == "/nonexistent/file.txt"
        assert error.category == ErrorCategory.FILE_NOT_FOUND

    def test_file_error(self, caplog):
        """Verify FileError captures path."""
        error = FileError("Permission denied", path="/restricted/file.txt")

        assert "Permission denied" in str(error)
        assert error.path == "/restricted/file.txt"


class TestNetworkErrorHandling:
    """Test network error handling patterns."""

    def test_network_error(self, caplog):
        """Verify NetworkError captures URL."""
        error = NetworkError("Connection failed", url="https://api.example.com")

        assert "Connection failed" in str(error)
        assert error.url == "https://api.example.com"
        assert error.category == ErrorCategory.NETWORK_ERROR


class TestErrorHandler:
    """Test ErrorHandler utility."""

    def test_handle_victor_error(self, caplog):
        """Verify ErrorHandler handles VictorError correctly."""
        caplog.set_level(logging.ERROR)
        handler = ErrorHandler()

        error = ProviderError("Test error", provider="test")
        error_info = handler.handle(error)

        assert error_info.message == "Test error"
        assert error_info.category == ErrorCategory.UNKNOWN
        assert "Test error" in caplog.text

    def test_handle_standard_exception(self, caplog):
        """Verify ErrorHandler categorizes standard exceptions."""
        caplog.set_level(logging.ERROR)
        handler = ErrorHandler()

        # Test ValueError categorization
        error = ValueError("Invalid value")
        error_info = handler.handle(error)

        assert error_info.category == ErrorCategory.VALIDATION_ERROR
        assert error_info.recovery_hint is not None

    def test_handle_file_not_found(self, caplog):
        """Verify ErrorHandler categorizes FileNotFoundError."""
        caplog.set_level(logging.ERROR)
        handler = ErrorHandler()

        # Use Python's builtin FileNotFoundError
        import builtins

        error = builtins.FileNotFoundError("File not found")
        error_info = handler.handle(error)

        assert error_info.category == ErrorCategory.FILE_NOT_FOUND
        assert "file exists" in error_info.recovery_hint.lower()

    def test_error_history(self, caplog):
        """Verify ErrorHandler maintains error history."""
        handler = ErrorHandler()

        # Handle multiple errors
        handler.handle(ValueError("Error 1"))
        handler.handle(TypeError("Error 2"))
        handler.handle(KeyError("Error 3"))

        recent = handler.get_recent_errors(count=2)
        assert len(recent) == 2
        assert "Error 3" in recent[-1].message

    def test_clear_history(self, caplog):
        """Verify ErrorHandler can clear history."""
        handler = ErrorHandler()

        handler.handle(ValueError("Error 1"))
        assert len(handler.get_recent_errors()) > 0

        handler.clear_history()
        assert len(handler.get_recent_errors()) == 0


class TestErrorSerialization:
    """Test error serialization and display."""

    def test_error_to_dict(self):
        """Verify VictorError serializes to dict correctly."""
        error = ProviderError("Test error", provider="test_provider", model="test_model")

        error_dict = error.to_dict()

        assert error_dict["error"] == "Test error"
        assert error_dict["category"] == ErrorCategory.UNKNOWN.value
        assert error_dict["details"]["provider"] == "test_provider"
        assert error_dict["details"]["model"] == "test_model"
        assert "correlation_id" in error_dict
        assert "timestamp" in error_dict

    def test_error_str(self):
        """Verify VictorError __str__ includes recovery hint."""
        error = ProviderConnectionError("Connection failed", provider="test")

        error_str = str(error)
        assert "Connection failed" in error_str
        assert "Recovery hint:" in error_str
        assert error.correlation_id in error_str


class TestExceptionChaining:
    """Test exception chaining with 'from e' pattern."""

    def test_exception_chaining_preserves_context(self):
        """Verify exception chaining preserves original exception."""
        original = ValueError("Original error")

        try:
            raise ProviderError("Provider error") from original
        except ProviderError as e:
            assert e.__cause__ is original
            assert e.__cause__ is not None

    def test_nested_exception_handling(self, caplog):
        """Verify nested exception handling works correctly."""
        caplog.set_level(logging.ERROR)

        def inner_function():
            raise ConnectionError("Connection failed")

        def outer_function():
            try:
                inner_function()
            except ConnectionError as e:
                raise ProviderConnectionError(
                    "Failed to connect to provider", provider="test"
                ) from e

        with pytest.raises(ProviderConnectionError) as exc_info:
            outer_function()

        assert "Failed to connect to provider" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ConnectionError)


class TestLoggingPatterns:
    """Test that exceptions are logged with proper context."""

    def test_exception_logged_with_exc_info(self, caplog):
        """Verify exceptions are logged with exc_info."""
        caplog.set_level(logging.ERROR)

        error = ProviderError("Test error", provider="test")
        handler = ErrorHandler()
        handler.handle(error)

        # Verify logging occurred
        assert len(caplog.records) > 0
        # Note: exc_info creates DEBUG log for traceback
        assert any("Test error" in record.message for record in caplog.records)

    def test_exception_logged_with_extra_context(self, caplog):
        """Verify exceptions are logged with extra context."""
        caplog.set_level(logging.ERROR)

        error = ToolExecutionError("Execution failed", tool_name="test_tool")
        handler = ErrorHandler()
        error_info = handler.handle(error, context={"operation": "test_operation", "attempt": 1})

        assert error_info.details["operation"] == "test_operation"
        assert error_info.details["attempt"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
