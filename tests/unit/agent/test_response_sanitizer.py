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

"""Tests for ResponseSanitizer module."""

from unittest.mock import patch

import pytest

from victor.agent.response_sanitizer import (
    ResponseSanitizer,
    create_streaming_filter,
    sanitize_response,
    is_garbage_content,
    is_valid_tool_name,
    strip_markup,
)


class TestResponseSanitizer:
    """Tests for ResponseSanitizer class."""

    @pytest.fixture
    def sanitizer(self):
        """Create a ResponseSanitizer instance."""
        return ResponseSanitizer()

    def test_strip_markup_removes_html_tags(self, sanitizer):
        """Test that HTML/XML tags are removed."""
        text = "<p>Hello <b>World</b></p>"
        result = sanitizer.strip_markup(text)
        assert result == "Hello World"

    def test_strip_markup_handles_empty_string(self, sanitizer):
        """Test empty string handling."""
        assert sanitizer.strip_markup("") == ""
        assert sanitizer.strip_markup(None) is None

    def test_sanitize_removes_repeated_closing_tags(self, sanitizer):
        """Test removal of repeated closing tags."""
        text = "Hello</function></function></function></function>"
        result = sanitizer.sanitize(text)
        assert "</function>" not in result or result.count("</function>") < 3

    def test_sanitize_removes_orphaned_tags(self, sanitizer):
        """Test removal of orphaned XML-like tags."""
        text = "Hello <function>test</function> World"
        result = sanitizer.sanitize(text)
        assert "<function>" not in result
        assert "</function>" not in result

    def test_sanitize_removes_parameter_tags(self, sanitizer):
        """Test removal of parameter tags."""
        text = "Result <parameter>value</parameter> here"
        result = sanitizer.sanitize(text)
        assert "<parameter>" not in result

    def test_sanitize_removes_json_tool_calls(self, sanitizer):
        """Test removal of embedded JSON tool calls."""
        text = 'Here is the result {"name": "read_file", "arguments": {"path": "test.py"}} done'
        result = sanitizer.sanitize(text)
        assert '{"name":' not in result

    def test_sanitize_handles_empty_string(self, sanitizer):
        """Test empty string handling."""
        assert sanitizer.sanitize("") == ""
        assert sanitizer.sanitize(None) is None

    def test_sanitize_preserves_clean_text(self, sanitizer):
        """Test that clean text is preserved."""
        text = "This is a normal response with no issues."
        result = sanitizer.sanitize(text)
        assert result == text

    def test_is_garbage_content_detects_function_call_syntax(self, sanitizer):
        """Test garbage detection for raw function call syntax."""
        assert sanitizer.is_garbage_content("FUNCTION_CALL {")
        assert sanitizer.is_garbage_content("FUNCTION_CALL{name}")

    def test_is_garbage_content_detects_repeated_tags(self, sanitizer):
        """Test garbage detection for repeated closing tags."""
        assert sanitizer.is_garbage_content("</function></function>")

    def test_is_garbage_content_detects_instruction_leakage(self, sanitizer):
        """Test garbage detection for instruction leakage."""
        assert sanitizer.is_garbage_content("<IMPORTANT>Do not...")
        assert sanitizer.is_garbage_content("Do NOT mention...")
        assert sanitizer.is_garbage_content("NEVER use this")

    def test_is_garbage_content_detects_tool_request_format(self, sanitizer):
        """Test garbage detection for LMStudio format."""
        assert sanitizer.is_garbage_content("[TOOL_REQUEST]")
        assert sanitizer.is_garbage_content("[END_TOOL_REQUEST]")

    def test_is_garbage_content_returns_false_for_clean_text(self, sanitizer):
        """Test that clean text is not flagged as garbage."""
        assert not sanitizer.is_garbage_content("This is a normal response.")
        assert not sanitizer.is_garbage_content("Here are the file contents...")

    def test_is_garbage_content_handles_empty(self, sanitizer):
        """Test empty string handling."""
        assert not sanitizer.is_garbage_content("")
        assert not sanitizer.is_garbage_content(None)

    def test_is_valid_tool_name_accepts_valid_names(self, sanitizer):
        """Test valid tool names are accepted."""
        assert sanitizer.is_valid_tool_name("read_file")
        assert sanitizer.is_valid_tool_name("execute_bash")
        assert sanitizer.is_valid_tool_name("listDirectory")
        assert sanitizer.is_valid_tool_name("tool123")

    def test_is_valid_tool_name_rejects_example_names(self, sanitizer):
        """Test that example/hallucinated names are rejected."""
        assert not sanitizer.is_valid_tool_name("example_tool")
        assert not sanitizer.is_valid_tool_name("func_test")
        assert not sanitizer.is_valid_tool_name("function_name")
        assert not sanitizer.is_valid_tool_name("tool_name")
        assert not sanitizer.is_valid_tool_name("my_custom_tool")
        assert not sanitizer.is_valid_tool_name("test_tool")
        assert not sanitizer.is_valid_tool_name("sample_tool")

    def test_is_valid_tool_name_rejects_malformed_names(self, sanitizer):
        """Test that malformed names are rejected."""
        assert not sanitizer.is_valid_tool_name("<function>")
        assert not sanitizer.is_valid_tool_name("tool/path")
        assert not sanitizer.is_valid_tool_name("tool with space")
        assert not sanitizer.is_valid_tool_name("123tool")
        assert not sanitizer.is_valid_tool_name("")
        assert not sanitizer.is_valid_tool_name(None)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_sanitize_response_function(self):
        """Test the sanitize_response convenience function."""
        text = "Hello <function>test</function> World"
        result = sanitize_response(text)
        assert "<function>" not in result

    def test_is_garbage_content_function(self):
        """Test the is_garbage_content convenience function."""
        assert is_garbage_content("FUNCTION_CALL {")
        assert not is_garbage_content("Normal text")

    def test_is_valid_tool_name_function(self):
        """Test the is_valid_tool_name convenience function."""
        assert is_valid_tool_name("read_file")
        assert not is_valid_tool_name("example_tool")

    def test_strip_markup_function(self):
        """Test the strip_markup convenience function."""
        assert strip_markup("<p>Hello</p>") == "Hello"


class TestResponseSanitizerEdgeCases:
    """Edge case tests for ResponseSanitizer."""

    @pytest.fixture
    def sanitizer(self):
        """Create a ResponseSanitizer instance."""
        return ResponseSanitizer()

    def test_sanitize_removes_json_name_lines(self, sanitizer):
        """Test removal of lines starting with {"name": (covers line 154)."""
        text = 'Some text\n{"name": "tool", "arguments": {}}\nMore text'
        result = sanitizer.sanitize(text)
        assert '{"name":' not in result
        assert "Some text" in result
        assert "More text" in result

    def test_sanitize_removes_parameter_equals_lines(self, sanitizer):
        """Test removal of lines with parameter= syntax (covers line 157)."""
        text = "Output\nparameter=value\nResult"
        result = sanitizer.sanitize(text)
        assert "parameter=" not in result
        assert "Output" in result
        assert "Result" in result

    def test_sanitize_removes_parameter_tag_lines(self, sanitizer):
        """Test removal of lines with <parameter syntax (covers line 157)."""
        text = "Data\n<parameter=x>value</parameter>\nEnd"
        result = sanitizer.sanitize(text)
        assert "<parameter" not in result
        assert "Data" in result
        assert "End" in result

    def test_sanitize_removes_plaintext_shell_like_command_lines(self, sanitizer):
        text = (
            "I'll analyze the codebase structure.\n"
            "ls rust\n"
            "scripts ls\n"
            "victor/core ls\n"
            "Then I will summarize the findings."
        )
        result = sanitizer.sanitize(text)
        assert "ls rust" not in result
        assert "scripts ls" not in result
        assert "victor/core ls" not in result
        assert "I'll analyze the codebase structure." in result
        assert "Then I will summarize the findings." in result

    def test_has_tool_format_confusion_detects_plaintext_shell_commands(self, sanitizer):
        text = "I'll inspect the repository first.\n" "ls rust\n" "scripts ls\n" "victor/tools ls\n"
        assert sanitizer.has_tool_format_confusion(text) is True

    def test_sanitize_logs_warning_on_significant_removal(self, sanitizer, caplog):
        """Test warning logged when >50% content removed (covers line 167)."""
        import logging

        caplog.set_level(logging.WARNING)
        # Create text > 100 chars that will be mostly garbage
        garbage = "<function>test</function>" * 10 + '{"name": "tool", "arguments": {"x": "test"}}'
        assert len(garbage) > 100
        result = sanitizer.sanitize(garbage)
        # Result should be shorter after removing garbage
        assert len(result) < len(garbage)

    def test_sanitize_removes_lines_starting_with_closing_tag(self, sanitizer):
        """Test removal of lines starting with </."""
        text = "Result\n</function>\n</param>\nDone"
        result = sanitizer.sanitize(text)
        assert "</function>" not in result
        assert "</param>" not in result
        assert "Result" in result
        assert "Done" in result

    def test_sanitize_removes_important_tags_lowercase(self, sanitizer):
        """Test removal of lowercase <important> tags."""
        text = "Text <important>note</important> more"
        result = sanitizer.sanitize(text)
        # IMPORTANT tags should be removed
        assert "Text" in result or result == ""


def test_create_streaming_filter_enforces_patched_thinking_limit() -> None:
    """Factory-created filters should honor the active thinking cap on all backends."""
    with patch("victor.agent.response_sanitizer.StreamingContentFilter.MAX_THINKING_CONTENT", 5):
        content_filter = create_streaming_filter(suppress_thinking=False)

        content_filter.process_chunk("<think>")
        result = content_filter.process_chunk("123456")

    assert result.content == ""
    assert result.is_thinking is True
    assert content_filter.should_abort() is True
    assert content_filter.abort_reason == (
        "Thinking content exceeded 5 chars. Model may be stuck in a reasoning loop."
    )
