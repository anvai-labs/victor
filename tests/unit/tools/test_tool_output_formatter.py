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

"""Tests for tool output formatter - achieving 70%+ coverage."""

import pytest
from unittest.mock import MagicMock, patch

from victor.agent.tool_output_formatter import (
    FormattingContext,
    ToolOutputFormatterConfig,
    TruncationResult,
    ToolOutputFormatter,
    create_tool_output_formatter,
    format_tool_output,
)


class TestFormattingContext:
    """Tests for FormattingContext dataclass."""

    def test_default_values(self):
        """Test default values."""
        ctx = FormattingContext()
        assert ctx.provider_name is None
        assert ctx.model is None
        assert ctx.remaining_tokens == 50000
        assert ctx.max_tokens == 100000
        assert ctx.response_token_reserve == 4096

    def test_custom_values(self):
        """Test custom values."""
        ctx = FormattingContext(
            provider_name="anthropic",
            model="claude-3-sonnet",
            remaining_tokens=30000,
            max_tokens=100000,
            response_token_reserve=8000,
        )
        assert ctx.provider_name == "anthropic"
        assert ctx.model == "claude-3-sonnet"
        assert ctx.remaining_tokens == 30000

    def test_token_pressure_low(self):
        """Test token_pressure when usage is low."""
        ctx = FormattingContext(remaining_tokens=90000, max_tokens=100000)
        assert ctx.token_pressure == pytest.approx(0.1, abs=0.01)

    def test_token_pressure_high(self):
        """Test token_pressure when usage is high."""
        ctx = FormattingContext(remaining_tokens=10000, max_tokens=100000)
        assert ctx.token_pressure == pytest.approx(0.9, abs=0.01)

    def test_token_pressure_full(self):
        """Test token_pressure when context is full."""
        ctx = FormattingContext(remaining_tokens=0, max_tokens=100000)
        assert ctx.token_pressure == 1.0

    def test_token_pressure_zero_max(self):
        """Test token_pressure when max_tokens is zero."""
        ctx = FormattingContext(remaining_tokens=0, max_tokens=0)
        assert ctx.token_pressure == 0.0

    def test_token_pressure_negative_remaining(self):
        """Test token_pressure handles negative remaining."""
        ctx = FormattingContext(remaining_tokens=-1000, max_tokens=100000)
        # Clamped to 1.0
        assert ctx.token_pressure == 1.0


class TestToolOutputFormatterConfig:
    """Tests for ToolOutputFormatterConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values (optimized for token efficiency)."""
        config = ToolOutputFormatterConfig()
        # These values were reduced for token efficiency
        assert config.max_output_chars == 8000  # Reduced from 15000
        assert config.file_structure_threshold == 30000  # Reduced from 50000
        assert config.min_savings_threshold == 0.15
        assert config.max_classes_shown == 15  # Reduced from 20
        assert config.max_functions_shown == 20  # Reduced from 30
        assert config.sample_lines_start == 20  # Reduced from 30
        assert config.sample_lines_end == 15  # Reduced from 20

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ToolOutputFormatterConfig(
            max_output_chars=10000,
            file_structure_threshold=30000,
            min_savings_threshold=0.2,
        )
        assert config.max_output_chars == 10000
        assert config.file_structure_threshold == 30000
        assert config.min_savings_threshold == 0.2


class TestTruncationResult:
    """Tests for TruncationResult dataclass."""

    def test_default_values(self):
        """Test default truncation result values."""
        result = TruncationResult(content="test content")
        assert result.content == "test content"
        assert result.truncated is False
        assert result.truncated_chars == 0

    def test_truncated_result(self):
        """Test truncated result values."""
        result = TruncationResult(
            content="truncated",
            truncated=True,
            truncated_chars=5000,
        )
        assert result.truncated is True
        assert result.truncated_chars == 5000


class TestToolOutputFormatter:
    """Tests for ToolOutputFormatter class."""

    def test_init_default_config(self):
        """Test initialization with default config (optimized for token efficiency)."""
        formatter = ToolOutputFormatter()
        assert formatter.config is not None
        assert formatter.config.max_output_chars == 8000  # Reduced from 15000
        assert formatter._truncator is None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ToolOutputFormatterConfig(max_output_chars=5000)
        formatter = ToolOutputFormatter(config=config)
        assert formatter.config.max_output_chars == 5000

    def test_init_with_truncator(self):
        """Test initialization with truncator."""
        truncator = MagicMock()
        formatter = ToolOutputFormatter(truncator=truncator)
        assert formatter._truncator is truncator

    def test_format_generic_tool(self):
        """Test formatting generic tool output with xml style."""
        formatter = ToolOutputFormatter(
            config=ToolOutputFormatterConfig(default_format_style="xml")
        )
        result = formatter.format_tool_output(
            tool_name="test_tool",
            args={"param": "value"},
            output="Test output content",
        )
        assert '<TOOL_OUTPUT tool="test_tool"' in result
        assert "Test output content" in result
        assert "</TOOL_OUTPUT>" in result

    def test_format_read_file(self):
        """Test formatting read_file tool output with xml strategy."""
        formatter = ToolOutputFormatter(
            config=ToolOutputFormatterConfig(default_format_style="xml")
        )
        result = formatter.format_tool_output(
            tool_name="read_file",
            args={"path": "/path/to/file.py"},
            output="def hello():\n    pass",
        )
        assert '<TOOL_OUTPUT tool="read_file"' in result
        assert 'path="/path/to/file.py"' in result
        assert "def hello():" in result
        assert "</TOOL_OUTPUT>" in result

    def test_format_read_file_truncated(self):
        """Test formatting truncated read_file output with xml strategy."""
        config = ToolOutputFormatterConfig(max_output_chars=100, default_format_style="xml")
        formatter = ToolOutputFormatter(config=config)

        long_content = "x" * 500
        result = formatter.format_tool_output(
            tool_name="read_file",
            args={"path": "/path/to/file.py"},
            output=long_content,
        )
        assert "TRUNCATED" in result
        assert "</TOOL_OUTPUT>" in result

    def test_format_read_file_very_large(self):
        """Test formatting very large file with structure summary."""
        config = ToolOutputFormatterConfig(file_structure_threshold=100)
        formatter = ToolOutputFormatter(config=config)

        large_content = """class MyClass:
    def method1(self):
        pass
    def method2(self):
        pass

def helper_function():
    pass
""" * 50
        result = formatter.format_tool_output(
            tool_name="read_file",
            args={"path": "/path/to/file.py"},
            output=large_content,
        )
        # Check for large file indicator (new format uses "LARGE FILE:")
        assert "LARGE FILE:" in result or "FILE IS VERY LARGE" in result
        assert "FILE STRUCTURE" in result
        # Check for pagination hint (new format uses "HOW TO READ THIS FILE")
        assert "HOW TO READ THIS FILE" in result or "To see specific sections" in result

    def test_format_read_canonical_very_large(self):
        """Canonical read should use the same large-file path as legacy aliases."""
        config = ToolOutputFormatterConfig(file_structure_threshold=100)
        formatter = ToolOutputFormatter(config=config)

        large_content = "def helper():\n    pass\n" * 80
        result = formatter.format_tool_output(
            tool_name="read",
            args={"path": "/path/to/file.py"},
            output=large_content,
        )

        assert "FILE STRUCTURE" in result

    def test_format_list_directory(self):
        """Test formatting list_directory tool output with xml strategy."""
        formatter = ToolOutputFormatter(
            config=ToolOutputFormatterConfig(default_format_style="xml")
        )
        result = formatter.format_tool_output(
            tool_name="list_directory",
            args={"path": "/home/user"},
            output="file1.py\nfile2.py\nsubdir/",
        )
        assert '<TOOL_OUTPUT tool="list_directory"' in result
        assert 'path="/home/user"' in result
        assert "file1.py" in result
        assert "</TOOL_OUTPUT>" in result

    def test_format_code_search(self):
        """Test formatting code_search tool output with xml strategy."""
        formatter = ToolOutputFormatter(
            config=ToolOutputFormatterConfig(default_format_style="xml")
        )
        result = formatter.format_tool_output(
            tool_name="code_search",
            args={"query": "def main"},
            output="file1.py:10: def main():\nfile2.py:25: def main():",
        )
        assert '<TOOL_OUTPUT tool="code_search"' in result
        assert 'query="def main"' in result
        assert "def main" in result

    def test_format_code_search_includes_follow_up_suggestions(self):
        """code_search structured output is serialized inside the xml tag."""
        formatter = ToolOutputFormatter(
            config=ToolOutputFormatterConfig(default_format_style="xml")
        )
        result = formatter.format_tool_output(
            tool_name="code_search",
            args={"query": "entry point for request processing"},
            output={
                "success": True,
                "results": [{"file_path": "src/main.py", "name": "main"}],
                "metadata": {
                    "follow_up_suggestions": [
                        {
                            "command": 'graph(mode="trace", node="main", depth=3)',
                            "reason": "Trace execution starting from main.",
                        }
                    ]
                },
            },
        )

        # Structured output is serialized as JSON inside the XML wrapper
        assert '<TOOL_OUTPUT tool="code_search"' in result
        assert "src/main.py" in result
        assert "follow_up_suggestions" in result

    def test_format_semantic_code_search(self):
        """Test formatting semantic_code_search tool output with xml style."""
        formatter = ToolOutputFormatter(
            config=ToolOutputFormatterConfig(default_format_style="xml")
        )
        result = formatter.format_tool_output(
            tool_name="semantic_code_search",
            args={"query": "find authentication handler"},
            output="auth_handler.py: handles user auth",
        )
        assert '<TOOL_OUTPUT tool="semantic_code_search"' in result
        assert "find authentication handler" in result

    def test_format_execute_bash(self):
        """Test formatting execute_bash tool output with xml strategy."""
        formatter = ToolOutputFormatter(
            config=ToolOutputFormatterConfig(default_format_style="xml")
        )
        result = formatter.format_tool_output(
            tool_name="execute_bash",
            args={"command": "ls -la"},
            output="total 48\ndrwxr-xr-x 5 user",
        )
        assert '<TOOL_OUTPUT tool="execute_bash"' in result
        assert 'command="ls -la"' in result
        assert "total 48" in result

    def test_format_with_truncation_note(self):
        """Test truncation note in generic formatting."""
        config = ToolOutputFormatterConfig(max_output_chars=50)
        formatter = ToolOutputFormatter(config=config)

        result = formatter.format_tool_output(
            tool_name="some_tool",
            args={},
            output="x" * 200,
        )
        assert "[OUTPUT TRUNCATED]" in result

    def test_format_with_smart_truncator(self):
        """Test formatting with smart truncator."""
        truncator = MagicMock()
        truncator.truncate_tool_result.return_value = TruncationResult(
            content="truncated content",
            truncated=True,
            truncated_chars=100,
        )

        formatter = ToolOutputFormatter(truncator=truncator)
        result = formatter.format_tool_output(
            tool_name="test",
            args={},
            output="x" * 500,
        )
        assert "truncated content" in result
        truncator.truncate_tool_result.assert_called_once()

    def test_format_with_truncator_exception(self):
        """Test fallback when truncator raises exception."""
        truncator = MagicMock()
        truncator.truncate_tool_result.side_effect = Exception("Truncation failed")

        config = ToolOutputFormatterConfig(max_output_chars=50)
        formatter = ToolOutputFormatter(config=config, truncator=truncator)

        # Should fall back to simple truncation
        result = formatter.format_tool_output(
            tool_name="test",
            args={},
            output="x" * 200,
        )
        assert "[OUTPUT TRUNCATED]" in result

    def test_format_with_context(self):
        """Test formatting with FormattingContext."""
        formatter = ToolOutputFormatter()
        ctx = FormattingContext(
            provider_name="openai",
            model="gpt-4",
            remaining_tokens=10000,
        )
        result = formatter.format_tool_output(
            tool_name="test",
            args={},
            output="content",
            context=ctx,
        )
        assert "content" in result


class TestExtractFileStructure:
    """Tests for extract_file_structure method."""

    def test_extract_python_structure(self):
        """Test extracting Python file structure."""
        formatter = ToolOutputFormatter()
        content = """class MyClass:
    def method(self):
        pass

class AnotherClass:
    pass

def top_level_function():
    pass

def another_function():
    return 42
"""
        result = formatter.extract_file_structure(content, "test.py")
        assert "FILE STRUCTURE" in result
        assert "Classes" in result
        assert "MyClass" in result
        assert "AnotherClass" in result
        assert "Functions" in result
        assert "top_level_function" in result

    def test_extract_js_structure(self):
        """Test extracting JavaScript file structure."""
        formatter = ToolOutputFormatter()
        content = """export function handleAuth() {
    return true;
}

function privateHelper() {
    return false;
}

export const API_URL = 'https://api.example.com';
"""
        result = formatter.extract_file_structure(content, "test.js")
        assert "FILE STRUCTURE" in result
        assert "Exports" in result
        assert "Functions" in result

    def test_extract_typescript_structure(self):
        """Test extracting TypeScript file structure."""
        formatter = ToolOutputFormatter()
        content = """export interface User {
    name: string;
}

export function getUser() {
    return null;
}
"""
        result = formatter.extract_file_structure(content, "test.ts")
        assert "FILE STRUCTURE" in result

    def test_extract_generic_structure(self):
        """Test extracting structure from non-Python/JS file."""
        formatter = ToolOutputFormatter()
        content = "line1\nline2\nline3\n" * 10
        result = formatter.extract_file_structure(content, "test.txt")
        assert "FILE STRUCTURE" in result
        assert "Total lines" in result
        assert "FIRST" in result
        assert "LAST" in result

    def test_extract_structure_many_classes(self):
        """Test structure extraction with many classes."""
        config = ToolOutputFormatterConfig(max_classes_shown=3)
        formatter = ToolOutputFormatter(config=config)

        classes = "\n".join([f"class Class{i}:\n    pass\n" for i in range(10)])
        result = formatter.extract_file_structure(classes, "many_classes.py")
        assert "Classes" in result
        assert "... and" in result  # Shows truncation

    def test_extract_structure_many_functions(self):
        """Test structure extraction with many functions."""
        config = ToolOutputFormatterConfig(max_functions_shown=3)
        formatter = ToolOutputFormatter(config=config)

        functions = "\n".join([f"def func{i}():\n    pass\n" for i in range(10)])
        result = formatter.extract_file_structure(functions, "many_funcs.py")
        assert "Functions" in result
        assert "... and" in result


class TestSerializeStructuredOutput:
    """Tests for _serialize_structured_output method."""

    def test_serialize_string_output(self):
        """Test serializing string output."""
        formatter = ToolOutputFormatter()
        result, hint = formatter._serialize_structured_output(
            "test_tool", "string output", {}, FormattingContext()
        )
        assert result == "string output"
        assert hint is None

    def test_serialize_none_output(self):
        """Test serializing None output."""
        formatter = ToolOutputFormatter()
        result, hint = formatter._serialize_structured_output(
            "test_tool", None, {}, FormattingContext()
        )
        assert result == ""
        assert hint is None

    def test_serialize_small_list(self):
        """Test serializing small list (not worth optimizing)."""
        formatter = ToolOutputFormatter()
        result, hint = formatter._serialize_structured_output(
            "test_tool", [1, 2], {}, FormattingContext()
        )
        assert result == "[1, 2]"
        assert hint is None

    def test_serialize_small_dict(self):
        """Test serializing small dict (not worth optimizing)."""
        formatter = ToolOutputFormatter()
        result, hint = formatter._serialize_structured_output(
            "test_tool", {"a": 1}, {}, FormattingContext()
        )
        assert "a" in result
        assert hint is None

    def test_serialize_large_list_fallback(self):
        """Test serializing large list falls back to str when serializer not available."""
        formatter = ToolOutputFormatter()
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
        result, hint = formatter._serialize_structured_output(
            "test_tool",
            data,
            {"operation": "list"},
            FormattingContext(),
        )
        # Falls back to str() when serializer is not available or returns low savings
        assert "a" in result or str(data) in result
        # hint may or may not be set depending on serializer

    def test_serialize_large_dict(self):
        """Test serializing large dict."""
        formatter = ToolOutputFormatter()
        data = {"key1": "value1", "key2": "value2", "key3": "value3", "key4": "value4"}
        result, hint = formatter._serialize_structured_output(
            "test_tool", data, {}, FormattingContext()
        )
        # Should contain some representation of the data
        assert "key1" in result

    def test_serialize_with_subcommand_args(self):
        """Test serializing with subcommand in args."""
        formatter = ToolOutputFormatter()
        data = [{"x": 1}, {"x": 2}, {"x": 3}]
        result, hint = formatter._serialize_structured_output(
            "test_tool",
            data,
            {"subcommand": "status"},
            FormattingContext(),
        )
        # Should contain data representation
        assert "x" in result or "[" in result


class TestGetStatusMessage:
    """Tests for get_status_message method."""

    def test_status_execute_bash(self):
        """Test status message for execute_bash."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message("execute_bash", {"command": "ls -la /home"})
        assert "Running execute_bash" in result
        assert "ls -la /home" in result

    def test_status_shell_canonical(self):
        """Test status message for canonical shell tool."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message("shell", {"cmd": "ls -la /home"})
        assert "Running shell" in result
        assert "ls -la /home" in result
        # Accept both emoji (🔧) and text (*) versions
        assert "🔧" in result or "*" in result

    def test_status_execute_bash_long_command(self):
        """Test status message for execute_bash with long command."""
        formatter = ToolOutputFormatter()
        long_cmd = "x" * 200
        result = formatter.get_status_message("execute_bash", {"command": long_cmd})
        assert "..." in result
        assert len(result) < 200

    def test_status_list_directory(self):
        """Test status message for list_directory."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message("list_directory", {"path": "/home"})
        assert "Listing directory" in result
        assert "/home" in result

    def test_status_list_directory_default(self):
        """Test status message for list_directory with default path."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message("list_directory", {})
        assert "." in result

    def test_status_read(self):
        """Test status message for read."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message("read", {"path": "/etc/config"})
        assert "Reading file" in result
        assert "/etc/config" in result

    def test_status_edit_files_single(self):
        """Test status message for edit_files with single file."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message("edit_files", {"files": [{"path": "file1.py"}]})
        assert "Editing" in result
        assert "file1.py" in result

    def test_status_edit_files_multiple(self):
        """Test status message for edit_files with multiple files."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message(
            "edit_files",
            {
                "files": [
                    {"path": "a.py"},
                    {"path": "b.py"},
                    {"path": "c.py"},
                    {"path": "d.py"},
                    {"path": "e.py"},
                ]
            },
        )
        assert "Editing" in result
        assert "+2 more" in result  # Shows overflow

    def test_status_edit_files_empty(self):
        """Test status message for edit_files with no files."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message("edit_files", {"files": []})
        assert "Running edit_files" in result

    def test_status_edit_canonical_ops(self):
        """Test status message for canonical edit tool using ops."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message(
            "edit",
            {
                "ops": [
                    {
                        "type": "replace",
                        "path": "file1.py",
                        "old_str": "a",
                        "new_str": "b",
                    }
                ]
            },
        )
        assert "Editing: file1.py" in result

    def test_status_edit_files_not_list(self):
        """Test status message for edit_files with invalid files type."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message("edit_files", {"files": "not a list"})
        assert "Running edit_files" in result

    def test_status_write(self):
        """Test status message for write."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message("write", {"path": "output.txt"})
        assert "Writing file" in result
        assert "output.txt" in result

    def test_status_code_search(self):
        """Test status message for code_search."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message("code_search", {"query": "def main"})
        assert "Searching" in result
        assert "def main" in result

    def test_status_code_search_long_query(self):
        """Test status message for code_search with long query."""
        formatter = ToolOutputFormatter()
        long_query = "x" * 100
        result = formatter.get_status_message("code_search", {"query": long_query})
        assert "..." in result

    def test_status_generic_tool(self):
        """Test status message for unknown tool."""
        formatter = ToolOutputFormatter()
        result = formatter.get_status_message("unknown_tool", {"arg": "value"})
        assert "Running unknown_tool" in result


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_tool_output_formatter_default(self):
        """Test factory function with defaults (optimized for token efficiency)."""
        formatter = create_tool_output_formatter()
        assert isinstance(formatter, ToolOutputFormatter)
        assert formatter.config.max_output_chars == 8000  # Reduced from 15000

    def test_create_tool_output_formatter_with_config(self):
        """Test factory function with config."""
        config = ToolOutputFormatterConfig(max_output_chars=5000)
        formatter = create_tool_output_formatter(config=config)
        assert formatter.config.max_output_chars == 5000

    def test_create_tool_output_formatter_with_truncator(self):
        """Test factory function with truncator."""
        truncator = MagicMock()
        formatter = create_tool_output_formatter(truncator=truncator)
        assert formatter._truncator is truncator

    def test_format_tool_output_convenience(self):
        """Test convenience function — plain style returns raw content."""
        result = format_tool_output(
            tool_name="test",
            args={"a": 1},
            output="output content",
        )
        # Plain style (default) returns raw content without XML wrapping
        assert "output content" in result

    def test_format_tool_output_with_context(self):
        """Test convenience function with context."""
        ctx = FormattingContext(provider_name="anthropic")
        result = format_tool_output(
            tool_name="test",
            args={},
            output="content",
            context=ctx,
        )
        assert "content" in result

    def test_format_tool_output_with_config(self):
        """Test convenience function with config."""
        config = ToolOutputFormatterConfig(max_output_chars=50)
        result = format_tool_output(
            tool_name="test",
            args={},
            output="x" * 200,
            config=config,
        )
        assert "[OUTPUT TRUNCATED]" in result


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_output(self):
        """Test formatting empty output does not raise and returns a string."""
        formatter = ToolOutputFormatter()
        result = formatter.format_tool_output("test", {}, "")
        assert isinstance(result, str)

    def test_newlines_in_output(self):
        """Test output with many newlines."""
        formatter = ToolOutputFormatter()
        result = formatter.format_tool_output("test", {}, "line1\nline2\nline3\n")
        assert "line1" in result
        assert "line3" in result

    def test_unicode_output(self):
        """Test output with unicode characters."""
        formatter = ToolOutputFormatter()
        result = formatter.format_tool_output("test", {}, "Hello 世界 🌍 café")
        assert "世界" in result
        assert "🌍" in result

    def test_special_xml_chars_in_output(self):
        """Test output with XML special characters."""
        formatter = ToolOutputFormatter()
        result = formatter.format_tool_output("test", {}, "<div>test</div>")
        assert "<div>test</div>" in result

    def test_exact_max_chars_boundary(self):
        """Test output exactly at max_output_chars boundary."""
        config = ToolOutputFormatterConfig(max_output_chars=100)
        formatter = ToolOutputFormatter(config=config)

        # Exactly 100 chars - should not truncate
        result = formatter.format_tool_output("test", {}, "x" * 100)
        assert "[OUTPUT TRUNCATED]" not in result

        # 101 chars - should truncate
        result = formatter.format_tool_output("test", {}, "x" * 101)
        assert "[OUTPUT TRUNCATED]" in result

    def test_extract_python_class_with_inheritance(self):
        """Test extracting Python class with complex inheritance."""
        formatter = ToolOutputFormatter()
        content = """class MyClass(BaseClass, MixinA, MixinB):
    pass

class Simple:
    pass
"""
        result = formatter.extract_file_structure(content, "test.py")
        assert "MyClass" in result
        assert "Simple" in result

    def test_extract_python_async_function(self):
        """Test extracting async functions."""
        formatter = ToolOutputFormatter()
        content = """async def async_handler():
    pass

def sync_handler():
    pass
"""
        result = formatter.extract_file_structure(content, "test.py")
        # async def should be treated same as def for structure
        assert "async_handler" in result or "sync_handler" in result

    def test_pyi_file_structure(self):
        """Test .pyi stub file structure extraction."""
        formatter = ToolOutputFormatter()
        content = """class MyProtocol:
    def method(self) -> None: ...
"""
        result = formatter.extract_file_structure(content, "test.pyi")
        assert "FILE STRUCTURE" in result

    def test_tsx_file_structure(self):
        """Test .tsx file structure extraction."""
        formatter = ToolOutputFormatter()
        content = """export function MyComponent() {
    return <div>Hello</div>;
}
"""
        result = formatter.extract_file_structure(content, "test.tsx")
        assert "FILE STRUCTURE" in result
        assert "Exports" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
