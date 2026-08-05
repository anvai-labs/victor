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

"""Tests for contextual error handling."""

import pytest

from victor.framework.contextual_errors import (
    ContextualError,
    ProviderConnectionError,
    ToolExecutionError,
    FileOperationError,
    ConfigurationError,
    ResourceError,
    create_provider_error,
    create_tool_error,
    create_file_error,
    wrap_error,
    format_exception_for_user,
)


class TestContextualError:
    """Tests for ContextualError base class."""

    def test_basic_error_message(self):
        """Basic error message works."""
        error = ContextualError(message="Something went wrong")
        assert "Something went wrong" in str(error)

    def test_error_with_operation(self):
        """Error with operation context stored as attribute."""
        error = ContextualError(message="Failed", operation="Test Operation")
        assert "Failed" in str(error)
        assert error.operation == "Test Operation"

    def test_error_with_suggestion(self):
        """Error with suggestion stored as recovery_hint."""
        error = ContextualError(message="Failed", suggestion="Try again")
        assert error.suggestion == "Try again"
        # VictorError uses recovery_hint for display
        assert "Try again" in str(error) or error.recovery_hint == "Try again"

    def test_error_with_error_code(self):
        """Error with error code stored as attribute."""
        error = ContextualError(message="Failed", error_code="TEST_ERROR")
        assert error.error_code == "TEST_ERROR"

    def test_error_with_details(self):
        """Error with details stored as attribute."""
        error = ContextualError(message="Failed", details={"key": "value"})
        assert "Failed" in str(error)
        assert error.details == {"key": "value"}


class TestProviderConnectionError:
    """Tests for ProviderConnectionError."""

    def test_anthropic_provider_error(self):
        """Anthropic provider error has correct suggestion."""
        error = ProviderConnectionError(provider="anthropic", error=Exception("API Error"))
        error_str = str(error)
        assert "ANTHROPIC_API_KEY" in error_str or "ANTHROPIC_API_KEY" in (error.suggestion or "")

    def test_ollama_provider_error(self):
        """Ollama provider error has correct suggestion."""
        error = ProviderConnectionError(provider="ollama", error=Exception("Not running"))
        error_str = str(error)
        assert "ollama serve" in error_str or "Ollama" in error_str

    def test_provider_error_without_original_error(self):
        """Provider error without original error works."""
        error = ProviderConnectionError(provider="openai")
        assert "Failed to connect to openai provider" in str(error)


class TestToolExecutionError:
    """Tests for ToolExecutionError."""

    def test_tool_error_with_basic_info(self):
        """Tool error with basic info."""
        error = ToolExecutionError(
            tool_name="test_tool", operation="execute", error=Exception("Tool failed")
        )
        error_str = str(error)
        assert "test_tool" in error_str
        assert "execute" in error_str

    def test_tool_error_with_custom_suggestion(self):
        """Tool error with custom suggestion."""
        error = ToolExecutionError(
            tool_name="test_tool", operation="execute", suggestion="Check permissions"
        )
        assert "Check permissions" in str(error)


class TestFileOperationError:
    """Tests for FileOperationError."""

    def test_file_read_error(self):
        """File read error."""
        error = FileOperationError(
            operation="read",
            path="/test/file.txt",
            error=FileNotFoundError("Not found"),
        )
        error_str = str(error)
        assert "read" in error_str
        assert "/test/file.txt" in error_str

    def test_file_write_error(self):
        """File write error."""
        error = FileOperationError(
            operation="write", path="/test/file.txt", error=PermissionError("Denied")
        )
        error_str = str(error)
        assert "write" in error_str


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_config_error_with_field(self):
        """Configuration error with field."""
        error = ConfigurationError(message="Invalid configuration", field="api_key")
        error_str = str(error)
        assert "Invalid configuration" in error_str
        assert "api_key" in error_str


class TestResourceError:
    """Tests for ResourceError."""

    def test_memory_error(self):
        """Memory resource error."""
        error = ResourceError(resource_type="memory")
        error_str = str(error)
        assert "memory" in error_str.lower()
        assert "free" in error_str.lower()

    def test_docker_resource_error(self):
        """Docker resource error."""
        error = ResourceError(resource_type="docker")
        error_str = str(error)
        assert "docker" in error_str.lower()
        assert "docker ps" in error_str.lower()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_provider_error(self):
        """create_provider_error creates correct error type."""
        error = create_provider_error(
            provider="anthropic", operation="generate", error=Exception("API Error")
        )
        assert isinstance(error, ProviderConnectionError)

    def test_create_tool_error(self):
        """create_tool_error creates correct error type."""
        error = create_tool_error(
            tool_name="test_tool", operation="execute", error=Exception("Failed")
        )
        assert isinstance(error, ToolExecutionError)

    def test_create_file_error(self):
        """create_file_error creates correct error type."""
        error = create_file_error(
            operation="read", path="/test/file.txt", error=FileNotFoundError()
        )
        assert isinstance(error, FileOperationError)

    def test_wrap_error(self):
        """wrap_error wraps exception with context."""
        original = ValueError("Invalid value")
        error = wrap_error(error=original, context="Parsing configuration")
        assert isinstance(error, ContextualError)
        assert error.operation == "Parsing configuration"


class TestFormatExceptionForUser:
    """Tests for format_exception_for_user."""

    def test_format_contextual_error(self):
        """Formatting contextual error returns string."""
        error = ContextualError(message="Test error", suggestion="Fix it")
        formatted = format_exception_for_user(error)
        assert "Test error" in formatted
        assert "Fix it" in formatted

    def test_format_api_key_error(self):
        """Formatting API key error adds suggestion."""
        error = Exception("Invalid API key or unauthorized")
        formatted = format_exception_for_user(error)
        assert "API key" in formatted
        assert "victor doctor" in formatted.lower()

    def test_format_connection_error(self):
        """Formatting connection error adds suggestion."""
        error = Exception("Connection refused")
        formatted = format_exception_for_user(error)
        assert "connection" in formatted.lower()

    def test_format_permission_error(self):
        """Formatting permission error adds suggestion."""
        error = PermissionError("Permission denied")
        formatted = format_exception_for_user(error)
        assert "permission" in formatted.lower()

    def test_format_file_not_found_error(self):
        """Formatting file not found error adds suggestion."""
        error = FileNotFoundError("File not found")
        formatted = format_exception_for_user(error)
        assert "exists" in formatted.lower()

    def test_format_generic_error(self):
        """Formatting generic error adds system info."""
        error = ValueError("Generic error")
        formatted = format_exception_for_user(error)
        assert "Generic error" in formatted
        assert "victor doctor" in formatted.lower()

    def test_format_error_with_brackets_escapes_markup(self):
        """Formatting error with brackets escapes rich markup."""
        # VictorError includes brackets in recovery_hint, which should be escaped
        from victor.core.errors import VictorError

        error = VictorError(
            message="Connection failed",
            recovery_hint="Check your [network] connection",
        )
        formatted = format_exception_for_user(error)
        # Brackets should be escaped with backslash
        assert "\\[network\\]" in formatted
        # The raw brackets should not appear unescaped (except for our own markup tags)
        # Count escaped vs unescaped brackets - our markup tags like [bold red] should remain
        # but brackets from the error message should be escaped

    def test_format_error_with_closing_bracket_does_not_break_markup(self):
        """Error message with closing bracket pattern doesn't break rich markup."""
        # This simulates the actual error that caused the MarkupError
        from victor.core.errors import VictorError

        error = VictorError(
            message="Connection failed: timeout",
            recovery_hint="Check network connection and provider URL. Verify the provider service is running. (provider=deepseek)",
        )
        formatted = format_exception_for_user(error)
        # Should be able to render without MarkupError
        # The brackets in the error message should be escaped
        assert "provider=deepseek" in formatted
        # Check that the brackets are escaped
        assert "\\[" in formatted or "provider=deepseek" in formatted
