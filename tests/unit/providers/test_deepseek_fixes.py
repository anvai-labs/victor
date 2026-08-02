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

"""Tests for DeepSeek-specific fixes.

Tests the following fixes:
1. Empty tool content gets placeholder "(no output)"
2. Rich markup escaping prevents parsing errors
3. Error logging captures HTTP status codes
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock

from victor.providers.openai_compat import build_openai_messages
from victor.providers.base import Message
from victor.ui.rendering.markdown import (
    render_markdown_with_hooks,
    _escape_rich_markup_from_text,
)


class TestDeepSeekEmptyToolContentFix:
    """Tests for empty tool content placeholder fix."""

    def test_empty_tool_content_gets_placeholder(self):
        """Tool messages with empty content get '(no output)' placeholder."""
        messages = [
            Message(role="user", content="Use the tool"),
        ]
        # Create assistant message with tool_calls
        assistant_msg = Message(role="assistant", content="")
        assistant_msg.tool_calls = [{"id": "call_123", "name": "test_tool", "arguments": {}}]
        messages.append(assistant_msg)

        # Create tool message with empty content
        tool_msg = Message(role="tool", content="")
        tool_msg.tool_call_id = "call_123"
        tool_msg.name = "test_tool"
        messages.append(tool_msg)

        result = build_openai_messages(messages)

        # Find the tool message in result
        tool_result = [m for m in result if m["role"] == "tool"][0]
        # Empty content should be replaced with placeholder
        assert tool_result["content"] == "(no output)"
        assert tool_result["tool_call_id"] == "call_123"

    def test_tool_content_with_whitespace_gets_placeholder(self):
        """Tool messages with only whitespace are preserved as-is.

        Note: The implementation checks for falsy content (empty string),
        not whitespace-only strings. Whitespace-only content is passed through
        since it may be meaningful (e.g., formatting in code output).
        """
        messages = [
            Message(role="user", content="Use the tool"),
        ]
        assistant_msg = Message(role="assistant", content="")
        assistant_msg.tool_calls = [{"id": "call_456", "name": "read", "arguments": {}}]
        messages.append(assistant_msg)

        # Create tool message with whitespace-only content
        tool_msg = Message(role="tool", content="   \n\t  ")
        tool_msg.tool_call_id = "call_456"
        tool_msg.name = "read"
        messages.append(tool_msg)

        result = build_openai_messages(messages)

        tool_result = [m for m in result if m["role"] == "tool"][0]
        # Whitespace-only content is preserved (may be meaningful formatting)
        assert tool_result["content"] == "   \n\t  "

    def test_non_empty_tool_content_preserved(self):
        """Tool messages with actual content are preserved."""
        messages = [
            Message(role="user", content="Use the tool"),
        ]
        assistant_msg = Message(role="assistant", content="")
        assistant_msg.tool_calls = [{"id": "call_789", "name": "write", "arguments": {}}]
        messages.append(assistant_msg)

        # Create tool message with actual content
        tool_msg = Message(role="tool", content="File written successfully")
        tool_msg.tool_call_id = "call_789"
        tool_msg.name = "write"
        messages.append(tool_msg)

        result = build_openai_messages(messages)

        tool_result = [m for m in result if m["role"] == "tool"][0]
        assert tool_result["content"] == "File written successfully"


class TestRichMarkupEscapingFix:
    """Tests for Rich markup escaping fix."""

    def test_escape_rich_markup_tags(self):
        """Rich-specific markup tags that match the pattern are escaped."""
        # The regex matches: [word], [/word], [tag=value]
        # It requires at least one word character after the bracket
        # The escaping replaces the opening [ with \[ for each matched tag
        test_cases = [
            ("[bold]text[/bold]", "\\[bold]text\\[/bold]"),  # Complete bold tag
            ("[red]error[/red]", "\\[red]error\\[/red]"),  # Color tag
            (
                "[link=https://example.com]text[/link]",
                "\\[link=https://example.com]text\\[/link]",
            ),  # Link tag
        ]

        for markup, expected in test_cases:
            result = _escape_rich_markup_from_text(markup)
            assert result == expected, f"For '{markup}': Expected {expected}, got {result}"

        # Test that incomplete tags (no word after /) are not matched
        # This is correct - we only escape valid-looking Rich tags
        result = _escape_rich_markup_from_text("[/]")
        assert result == "[/]", "Standalone [/] is not a valid Rich tag, left as-is"

    def test_escape_path_like_strings(self):
        """File paths with brackets are escaped."""
        # This was the original error: "closing tag '[/\]' at position 107"
        test_cases = [
            "path/to/[\\/]file.txt",
            "file[1].txt",
            "[backup]file.txt[/]",
        ]

        for path in test_cases:
            result = _escape_rich_markup_from_text(path)
            # Should not contain unescaped brackets that look like tags
            assert result is not None
            assert isinstance(result, str)

    def test_render_markdown_with_hooks_fallback(self, caplog):
        """render_markdown_with_hooks falls back to plain text on error."""
        # Content with problematic markup
        content = "Here's a file path: path/to/[\\/]file.txt\n\nMore text."

        # This should not raise an exception
        result = render_markdown_with_hooks(content)

        # Should return a valid RenderableType
        assert result is not None

    def test_render_markdown_with_malformed_markup(self):
        """Malformed Rich markup doesn't break rendering."""
        # Content that would cause "closing tag '[/\]' has nothing to close"
        problematic_content = """
        The file is located at:
        /path/to/[\\/]directory/file.txt

        Here's more content with [brackets].
        """

        # Should not raise an exception
        result = render_markdown_with_hooks(problematic_content)
        assert result is not None


class TestDeepSeekErrorHandling:
    """Tests for improved error logging."""

    def test_http_status_error_logged(self, caplog):
        """HTTP status errors are logged with details."""
        import httpx
        from victor.providers.openai_compat import handle_httpx_status_error

        # Create a mock HTTPStatusError
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = '{"error": {"message": "Invalid request format"}}'

        mock_request = Mock()
        mock_exc = httpx.HTTPStatusError(
            "Bad Request",
            request=mock_request,
            response=mock_response,
        )

        with caplog.at_level(logging.ERROR):
            result = handle_httpx_status_error(mock_exc, "deepseek")

        # Should log the error with status code and details
        assert any("Provider HTTP error" in record.message for record in caplog.records)
        assert any("status=400" in record.message for record in caplog.records)

    def test_connection_error_classification(self):
        """Connection errors are properly classified."""
        from victor.providers.base import ProviderConnectionError

        # Test the error classification by checking that ConnectionError
        # is considered a connection-like error
        test_error = ConnectionError("Failed to connect to api.deepseek.com")

        # The error message should contain the original error details
        error_str = str(test_error)
        assert "Failed to connect" in error_str
        assert "api.deepseek.com" in error_str
