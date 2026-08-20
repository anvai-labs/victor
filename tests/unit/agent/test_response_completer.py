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

"""Tests for ResponseCompleter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.response_completer import (
    CompletionConfig,
    CompletionResult,
    CompletionStatus,
    ResponseCompleter,
    ToolFailureContext,
    create_response_completer,
)
from victor.providers.base import CompletionResponse, Message


class TestCompletionConfig:
    """Tests for CompletionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CompletionConfig()
        assert config.max_retries == 3
        assert config.retry_temperature_increment == 0.1
        assert config.min_response_length == 10
        assert config.force_response_on_error is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = CompletionConfig(
            max_retries=5,
            min_response_length=50,
            force_response_on_error=False,
        )
        assert config.max_retries == 5
        assert config.min_response_length == 50
        assert config.force_response_on_error is False


class TestToolFailureContext:
    """Tests for ToolFailureContext."""

    def test_empty_context(self):
        """Test empty failure context."""
        ctx = ToolFailureContext()
        assert len(ctx.failed_tools) == 0
        assert len(ctx.successful_tools) == 0
        assert ctx.last_error is None

    def test_context_with_failures(self):
        """Test context with failed tools."""
        ctx = ToolFailureContext(
            failed_tools=[{"name": "read_file", "error": "File not found"}],
            last_error="File not found",
        )
        assert len(ctx.failed_tools) == 1
        assert ctx.last_error == "File not found"


class TestCompletionResult:
    """Tests for CompletionResult."""

    def test_complete_result(self):
        """Test complete result."""
        result = CompletionResult(
            status=CompletionStatus.SUCCESS,
            content="This is a complete response with enough content.",
        )
        assert result.is_complete is True

    def test_incomplete_result(self):
        """Test incomplete result."""
        result = CompletionResult(
            status=CompletionStatus.SUCCESS,
            content="Short",
        )
        assert result.is_complete is False

    def test_empty_result(self):
        """Test empty result."""
        result = CompletionResult(
            status=CompletionStatus.EMPTY,
            content="",
        )
        assert result.is_complete is False


class TestResponseCompleter:
    """Tests for ResponseCompleter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_provider = MagicMock()
        self.mock_provider.chat = AsyncMock(
            return_value=CompletionResponse(
                content="This is a test response from the model.",
                role="assistant",
                tool_calls=None,
            )
        )

    @pytest.mark.asyncio
    async def test_ensure_response_with_existing_content(self):
        """Test that existing content is returned as-is."""
        completer = ResponseCompleter(self.mock_provider)

        result = await completer.ensure_response(
            messages=[],
            model="test-model",
            temperature=0.7,
            max_tokens=1000,
            current_content="This is already a complete response.",
        )

        assert result.status == CompletionStatus.SUCCESS
        assert result.content == "This is already a complete response."
        # Provider should not be called
        self.mock_provider.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_response_generates_new(self):
        """Test generating new response when content is empty."""
        completer = ResponseCompleter(self.mock_provider)

        result = await completer.ensure_response(
            messages=[Message(role="user", content="Hello")],
            model="test-model",
            temperature=0.7,
            max_tokens=1000,
            current_content="",
        )

        assert result.status == CompletionStatus.SUCCESS
        assert len(result.content) > 10
        self.mock_provider.chat.assert_called()

    @pytest.mark.asyncio
    async def test_handle_tool_failures(self):
        """Test handling tool failures generates helpful message."""
        completer = ResponseCompleter(self.mock_provider)
        failure_context = ToolFailureContext(
            failed_tools=[{"name": "read_file", "error": "File not found"}],
            successful_tools=[{"name": "list_directory"}],
            last_error="File not found",
        )

        result = await completer.ensure_response(
            messages=[Message(role="user", content="Read file.py")],
            model="test-model",
            temperature=0.7,
            max_tokens=1000,
            failure_context=failure_context,
        )

        assert result.status == CompletionStatus.SUCCESS
        self.mock_provider.chat.assert_called()

    @pytest.mark.asyncio
    async def test_retry_on_empty_response(self):
        """Test that empty responses trigger retries."""
        # First call returns empty, second returns content
        self.mock_provider.chat = AsyncMock(
            side_effect=[
                CompletionResponse(content="", role="assistant", tool_calls=None),
                CompletionResponse(
                    content="Retry succeeded with content.",
                    role="assistant",
                    tool_calls=None,
                ),
            ]
        )

        completer = ResponseCompleter(self.mock_provider)
        result = await completer.ensure_response(
            messages=[Message(role="user", content="Hello")],
            model="test-model",
            temperature=0.7,
            max_tokens=1000,
        )

        assert result.status == CompletionStatus.SUCCESS
        assert "Retry succeeded" in result.content
        assert result.retries_used >= 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Test behavior when all retries fail."""
        self.mock_provider.chat = AsyncMock(
            return_value=CompletionResponse(content="", role="assistant", tool_calls=None)
        )

        config = CompletionConfig(max_recovery_attempts=2)
        completer = ResponseCompleter(self.mock_provider, config=config)

        result = await completer.ensure_response(
            messages=[Message(role="user", content="Hello")],
            model="test-model",
            temperature=0.7,
            max_tokens=1000,
        )

        assert result.status == CompletionStatus.EMPTY
        assert result.retries_used == 2

    def test_format_tool_failure_message(self):
        """Test formatting of tool failure messages."""
        completer = ResponseCompleter(self.mock_provider)
        failure_context = ToolFailureContext(
            failed_tools=[
                {"name": "read_file", "error": "File not found"},
                {"name": "write_file", "error": "Permission denied"},
            ],
            successful_tools=[{"name": "list_directory"}],
            files_examined=["/src/main.py", "/src/utils.py"],
        )

        message = completer.format_tool_failure_message(failure_context)

        assert "read_file" in message
        assert "File not found" in message
        assert "write_file" in message
        assert "list_directory" in message


class TestCreateResponseCompleter:
    """Tests for the factory function."""

    def test_create_response_completer(self):
        """Test factory creates completer with correct config."""
        mock_provider = MagicMock()
        completer = create_response_completer(
            provider=mock_provider,
            max_retries=5,
            force_response=False,
        )

        assert isinstance(completer, ResponseCompleter)
        assert completer.config.max_retries == 5
        assert completer.config.force_response_on_error is False

    def test_create_response_completer_defaults(self):
        """Test factory uses defaults."""
        mock_provider = MagicMock()
        completer = create_response_completer(provider=mock_provider)

        assert completer.config.max_retries == 3
        assert completer.config.force_response_on_error is True
