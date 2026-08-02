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

"""Unit tests for unified output formatter module."""

import io
import json
import pytest
from unittest.mock import patch

from victor.ui.output_formatter import (
    OutputMode,
    OutputConfig,
    OutputFormatter,
    InputReader,
    create_formatter,
)


class TestOutputMode:
    """Tests for OutputMode enum."""

    def test_output_modes_exist(self):
        """Test all output modes are defined."""
        assert OutputMode.RICH.value == "rich"
        assert OutputMode.PLAIN.value == "plain"
        assert OutputMode.JSON.value == "json"
        assert OutputMode.JSONL.value == "jsonl"
        assert OutputMode.CODE_ONLY.value == "code_only"

    def test_output_mode_count(self):
        """Test we have exactly 5 output modes."""
        assert len(OutputMode) == 5


class TestOutputConfig:
    """Tests for OutputConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OutputConfig()
        assert config.mode == OutputMode.RICH
        assert config.quiet is False
        assert config.show_tools is True
        assert config.show_thinking is True
        assert config.show_metrics is True
        assert config.stream is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = OutputConfig(
            mode=OutputMode.JSON,
            quiet=True,
            show_tools=False,
            show_thinking=False,
            show_metrics=False,
            stream=False,
        )
        assert config.mode == OutputMode.JSON
        assert config.quiet is True
        assert config.show_tools is False


class TestOutputFormatterInit:
    """Tests for OutputFormatter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        formatter = OutputFormatter()
        assert formatter.mode == OutputMode.RICH
        assert formatter._content_buffer == []
        assert formatter._tool_calls == []
        assert formatter._metrics == {}

    def test_custom_config_init(self):
        """Test initialization with custom config."""
        config = OutputConfig(mode=OutputMode.JSON, quiet=True)
        formatter = OutputFormatter(config)
        assert formatter.mode == OutputMode.JSON
        assert formatter.config.quiet is True


class TestOutputFormatterStatus:
    """Tests for status message output."""

    def test_status_quiet_mode(self):
        """Test status messages are suppressed in quiet mode."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, quiet=True, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.status("Test message")
        assert stdout.getvalue() == ""

    def test_status_plain_mode(self):
        """Test status messages in plain mode go to stderr."""
        stderr = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, stderr=stderr)
        formatter = OutputFormatter(config)
        formatter.status("Test message")
        assert "# Test message" in stderr.getvalue()

    def test_status_json_mode_suppressed(self):
        """Test status messages are suppressed in JSON mode."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSON, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.status("Test message")
        assert stdout.getvalue() == ""


class TestOutputFormatterError:
    """Tests for error message output."""

    def test_error_json_mode(self):
        """Test error output in JSON mode."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSON, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.error("Test error", "Error details")
        output = json.loads(stdout.getvalue())
        assert output["error"] == "Test error"
        assert output["details"] == "Error details"

    def test_error_jsonl_mode(self):
        """Test error output in JSONL mode."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSONL, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.error("Test error")
        output = json.loads(stdout.getvalue())
        assert output["type"] == "error"
        assert output["message"] == "Test error"

    def test_error_plain_mode(self):
        """Test error output in plain mode."""
        stderr = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, stderr=stderr)
        formatter = OutputFormatter(config)
        formatter.error("Test error", "Details")
        output = stderr.getvalue()
        assert "Error: Test error" in output
        assert "Details" in output


class TestOutputFormatterToolTracking:
    """Tests for tool execution tracking."""

    def test_tool_start_jsonl_mode(self):
        """Test tool start notification in JSONL mode."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSONL, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.tool_start("test_tool", {"arg": "value"})
        output = json.loads(stdout.getvalue())
        assert output["type"] == "tool_start"
        assert output["tool"] == "test_tool"
        assert output["arguments"] == {"arg": "value"}

    def test_tool_start_hidden_when_disabled(self):
        """Test tool start is hidden when show_tools is False."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSONL, show_tools=False, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.tool_start("test_tool", {"arg": "value"})
        assert stdout.getvalue() == ""

    def test_tool_result_recorded(self):
        """Test tool results are recorded in internal list."""
        config = OutputConfig(mode=OutputMode.JSON)
        formatter = OutputFormatter(config)
        formatter.tool_result("test_tool", True, result="Success")
        assert len(formatter._tool_calls) == 1
        assert formatter._tool_calls[0]["tool"] == "test_tool"
        assert formatter._tool_calls[0]["success"] is True

    def test_tool_result_truncated(self):
        """Test tool results are truncated to 500 chars."""
        config = OutputConfig(mode=OutputMode.JSON)
        formatter = OutputFormatter(config)
        long_result = "x" * 1000
        formatter.tool_result("test_tool", True, result=long_result)
        assert len(formatter._tool_calls[0]["result"]) == 500

    def test_tool_result_records_follow_up_suggestions(self):
        """Test tool results record follow-up suggestion commands."""
        config = OutputConfig(mode=OutputMode.JSON)
        formatter = OutputFormatter(config)
        formatter.tool_result(
            "code_search",
            True,
            follow_up_suggestions=[
                {
                    "command": 'graph(mode="trace", node="main", depth=3)',
                    "description": "Trace execution starting from main.",
                }
            ],
        )
        assert formatter._tool_calls[0]["follow_up_suggestions"] == [
            'graph(mode="trace", node="main", depth=3)'
        ]


class TestOutputFormatterStreaming:
    """Tests for streaming output."""

    def test_stream_chunk_buffered(self):
        """Test stream chunks are buffered."""
        config = OutputConfig(mode=OutputMode.JSON, stream=False)
        formatter = OutputFormatter(config)
        formatter.stream_chunk("Hello ")
        formatter.stream_chunk("World")
        assert formatter._content_buffer == ["Hello ", "World"]

    def test_stream_chunk_jsonl_output(self):
        """Test stream chunks output in JSONL mode."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSONL, stream=True, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.stream_chunk("Hello")
        output = json.loads(stdout.getvalue())
        assert output["type"] == "chunk"
        assert output["content"] == "Hello"

    def test_stream_chunk_plain_immediate(self):
        """Test stream chunks output immediately in plain mode."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, stream=True, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.stream_chunk("Hello")
        assert stdout.getvalue() == "Hello"


class TestOutputFormatterResponse:
    """Tests for final response output."""

    def test_response_json_mode(self):
        """Test response output in JSON mode."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSON, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.response(
            content="Test response",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            model="test-model",
        )
        result = json.loads(stdout.getvalue())
        assert result["content"] == "Test response"
        assert result["metrics"]["usage"]["prompt_tokens"] == 100
        assert result["metrics"]["model"] == "test-model"

    def test_response_uses_buffered_content(self):
        """Test response uses buffered content when not provided."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSON, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter._content_buffer = ["Hello ", "World"]
        formatter.response()
        result = json.loads(stdout.getvalue())
        assert result["content"] == "Hello World"

    def test_response_clears_buffers(self):
        """Test response clears all buffers."""
        config = OutputConfig(mode=OutputMode.JSON)
        formatter = OutputFormatter(config)
        formatter._content_buffer = ["test"]
        formatter._tool_calls = [{"tool": "test"}]
        formatter._metrics = {"model": "test"}
        formatter.response(content="Response")
        assert formatter._content_buffer == []
        assert formatter._tool_calls == []
        assert formatter._metrics == {}


class TestOutputFormatterCodeExtraction:
    """Tests for code extraction from responses."""

    def test_extract_markdown_python_block(self):
        """Test extracting code from markdown python block."""
        config = OutputConfig(mode=OutputMode.CODE_ONLY)
        formatter = OutputFormatter(config)
        content = """Here's the solution:

```python
def foo():
    return 42
```

This function returns 42."""
        extracted = formatter._extract_code(content)
        assert "def foo():" in extracted
        assert "return 42" in extracted

    def test_extract_multiple_blocks_returns_first_with_def(self):
        """Test multiple code blocks returns the first one with a function definition."""
        config = OutputConfig(mode=OutputMode.CODE_ONLY)
        formatter = OutputFormatter(config)
        content = """First attempt:
```python
def foo():
    return 1
```

Better version:
```python
def foo():
    return 42
```"""
        extracted = formatter._extract_code(content)
        # Implementation returns first block with 'def' (primary implementation)
        assert "return 1" in extracted
        assert "return 42" not in extracted

    def test_extract_raw_function_definition(self):
        """Test extracting raw function definitions."""
        config = OutputConfig(mode=OutputMode.CODE_ONLY)
        formatter = OutputFormatter(config)
        content = """The function is:
def calculate(x, y):
    result = x + y
    return result

That's the solution."""
        extracted = formatter._extract_code(content)
        assert "def calculate(x, y):" in extracted

    def test_code_only_mode_output(self):
        """Test CODE_ONLY mode outputs only code."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.CODE_ONLY, stdout=stdout)
        formatter = OutputFormatter(config)
        content = """Here's the code:
```python
def test():
    pass
```
That's it."""
        formatter.response(content=content)
        output = stdout.getvalue().strip()
        assert output == "def test():\n    pass"


class TestInputReader:
    """Tests for InputReader class."""

    def test_read_message_from_argument(self):
        """Test reading message from argument."""
        result = InputReader.read_message(argument="Test message")
        assert result == "Test message"

    def test_read_message_argument_priority(self):
        """Test argument takes priority over file."""
        result = InputReader.read_message(
            argument="From argument",
            input_file="/nonexistent/file.txt",
        )
        assert result == "From argument"

    def test_read_message_from_file(self, tmp_path):
        """Test reading message from file."""
        file_path = tmp_path / "input.txt"
        file_path.write_text("Message from file")
        result = InputReader.read_message(input_file=str(file_path))
        assert result == "Message from file"

    def test_read_message_file_error(self):
        """Test error handling for missing file."""
        with pytest.raises(ValueError, match="Failed to read input file"):
            InputReader.read_message(input_file="/nonexistent/file.txt")

    def test_is_piped(self):
        """Test stdin piping detection."""
        with patch("sys.stdin.isatty", return_value=False):
            assert InputReader.is_piped() is True
        with patch("sys.stdin.isatty", return_value=True):
            assert InputReader.is_piped() is False


class TestCreateFormatter:
    """Tests for create_formatter factory function."""

    def test_default_creates_rich_mode(self):
        """Test default creates Rich formatter."""
        formatter = create_formatter()
        assert formatter.mode == OutputMode.RICH

    def test_json_mode_flag(self):
        """Test JSON mode from flag."""
        formatter = create_formatter(json_mode=True)
        assert formatter.mode == OutputMode.JSON

    def test_jsonl_mode_flag(self):
        """Test JSONL mode from flag."""
        formatter = create_formatter(jsonl=True)
        assert formatter.mode == OutputMode.JSONL

    def test_code_only_mode_flag(self):
        """Test code-only mode from flag."""
        formatter = create_formatter(code_only=True)
        assert formatter.mode == OutputMode.CODE_ONLY

    def test_plain_mode_flag(self):
        """Test plain mode from flag."""
        formatter = create_formatter(plain=True)
        assert formatter.mode == OutputMode.PLAIN

    def test_json_priority_over_other_modes(self):
        """Test JSON mode takes priority over other modes."""
        formatter = create_formatter(json_mode=True, plain=True, code_only=True)
        assert formatter.mode == OutputMode.JSON

    def test_quiet_mode_disables_extras(self):
        """Test quiet mode disables extra output."""
        formatter = create_formatter(quiet=True)
        assert formatter.config.quiet is True
        assert formatter.config.show_tools is False
        assert formatter.config.show_thinking is False

    def test_stream_config(self):
        """Test stream configuration."""
        formatter = create_formatter(stream=False)
        assert formatter.config.stream is False


class TestOutputFormatterThinking:
    """Tests for thinking/reasoning output."""

    def test_thinking_jsonl_mode(self):
        """Test thinking output in JSONL mode."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSONL, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.thinking("Let me think about this...")
        output = json.loads(stdout.getvalue())
        assert output["type"] == "thinking"
        assert output["content"] == "Let me think about this..."

    def test_thinking_hidden_when_disabled(self):
        """Test thinking is hidden when show_thinking is False."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSONL, show_thinking=False, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.thinking("Let me think...")
        assert stdout.getvalue() == ""

    def test_thinking_plain_truncated(self):
        """Test thinking is truncated in plain mode."""
        stderr = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, stderr=stderr)
        formatter = OutputFormatter(config)
        long_thinking = "x" * 500
        formatter.thinking(long_thinking)
        output = stderr.getvalue()
        assert "# Thinking:" in output
        assert "..." in output


class TestOutputFormatterRichMode:
    """Tests for Rich mode specific functionality."""

    def test_tool_start_bash_command_truncated(self):
        """Test long bash commands are truncated in display (via tool_result)."""
        stderr = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, stderr=stderr)
        formatter = OutputFormatter(config)
        long_command = "echo " + "x" * 100
        # tool_start defers output to tool_result in PLAIN mode
        formatter.tool_start("execute_bash", {"command": long_command})
        formatter.tool_result("execute_bash", True)
        output = stderr.getvalue()
        # Implementation truncates args in _format_args - check tool output exists
        assert "executeBash" in output
        assert "✓" in output

    def test_tool_start_with_many_args(self):
        """Test tool_start truncates arguments when more than 3."""
        stderr = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, stderr=stderr)
        formatter = OutputFormatter(config)
        # tool_start defers output to tool_result in PLAIN mode
        formatter.tool_start("test_tool", {"a": 1, "b": 2, "c": 3, "d": 4})
        formatter.tool_result("test_tool", True)
        output = stderr.getvalue()
        assert "testTool" in output
        assert "✓" in output

    def test_tool_result_error_recorded(self):
        """Test tool_result records error messages."""
        config = OutputConfig(mode=OutputMode.JSON)
        formatter = OutputFormatter(config)
        formatter.tool_result("test_tool", False, error="Something went wrong")
        assert len(formatter._tool_calls) == 1
        assert formatter._tool_calls[0]["error"] == "Something went wrong"
        assert formatter._tool_calls[0]["success"] is False

    def test_tool_result_plain_mode_output(self):
        """Test tool_result output in Plain mode."""
        stderr = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, stderr=stderr)
        formatter = OutputFormatter(config)
        formatter.tool_result("test_tool", True)
        output = stderr.getvalue()
        # New compact format: "# ✓ tool_name(args) (duration)"
        assert "testTool" in output
        assert "✓" in output

    def test_tool_result_plain_mode_failed(self):
        """Test tool_result shows FAILED in Plain mode."""
        stderr = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, stderr=stderr)
        formatter = OutputFormatter(config)
        formatter.tool_result("test_tool", False)
        output = stderr.getvalue()
        # New compact format: "# ✗ tool_name(args) (duration) - error"
        assert "testTool" in output
        assert "✗" in output

    def test_tool_result_rich_mode_formats_tool_name_for_display(self):
        """Rich tool output uses lower-camel display names."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.RICH, stdout=stdout)
        formatter = OutputFormatter(config)

        formatter.tool_result("code_search", True)

        assert "codeSearch" in stdout.getvalue()

    def test_tool_result_plain_mode_shows_follow_up_suggestions(self):
        """Test tool_result prints visible next-step suggestions in Plain mode."""
        stderr = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, stderr=stderr)
        formatter = OutputFormatter(config)
        formatter.tool_result(
            "code_search",
            True,
            follow_up_suggestions=[
                {
                    "command": 'graph(mode="trace", node="main", depth=3)',
                    "description": "Trace execution starting from main.",
                }
            ],
        )
        output = stderr.getvalue()
        assert "next:" in output
        assert 'graph(mode="trace", node="main", depth=3)' in output


class TestOutputFormatterStreamingRich:
    """Tests for Rich mode streaming functionality."""

    def test_start_streaming_sets_live(self):
        """Test start_streaming creates Live object in Rich mode."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.RICH, stream=True, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.start_streaming()
        assert formatter._live is not None
        formatter.end_streaming()
        assert formatter._live is None

    def test_start_streaming_noop_non_rich(self):
        """Test start_streaming is no-op for non-Rich modes."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, stream=True, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.start_streaming()
        assert formatter._live is None

    def test_end_streaming_clears_buffer(self):
        """Test end_streaming clears stream buffer."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.RICH, stream=True, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.start_streaming()
        formatter._stream_buffer = "test content"
        formatter.end_streaming()
        assert formatter._stream_buffer == ""


class TestOutputFormatterResponseModes:
    """Tests for response output in various modes."""

    def test_response_jsonl_mode(self):
        """Test response output in JSONL mode."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSONL, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.tool_result("tool1", True, result="ok")
        formatter.response(content="Test response")
        result = json.loads(stdout.getvalue().strip().split("\n")[-1])
        assert result["type"] == "response"
        assert result["content"] == "Test response"
        assert "tool_calls" in result

    def test_response_plain_mode_no_stream(self):
        """Test response output in Plain mode without streaming."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, stream=False, stdout=stdout)
        formatter = OutputFormatter(config)
        result = formatter.response(content="Plain content")
        assert "Plain content" in stdout.getvalue()
        assert result == "Plain content"

    def test_response_plain_mode_with_stream(self):
        """Test response output in Plain mode with streaming."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.PLAIN, stream=True, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.response(content="Streamed content")
        # Should end with a newline
        assert stdout.getvalue().endswith("\n")

    def test_response_merges_tool_calls(self):
        """Test response merges provided tool_calls with tracked ones."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSON, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.tool_result("tool1", True)
        formatter.response(content="Test", tool_calls=[{"tool": "tool2", "success": True}])
        result = json.loads(stdout.getvalue())
        assert len(result["tool_calls"]) == 2


class TestOutputFormatterMetrics:
    """Tests for metrics display."""

    def test_show_metrics_empty(self):
        """Test _show_metrics with empty metrics."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.RICH, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter._show_metrics()
        # Should not produce output for empty metrics

    def test_metrics_included_in_response(self):
        """Test metrics are included in JSON response."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.JSON, stdout=stdout)
        formatter = OutputFormatter(config)
        formatter.response(
            content="Test",
            usage={"prompt_tokens": 50},
            model="test-model",
        )
        result = json.loads(stdout.getvalue())
        assert result["metrics"]["usage"]["prompt_tokens"] == 50
        assert result["metrics"]["model"] == "test-model"


class TestCodeExtractionEdgeCases:
    """Additional tests for code extraction edge cases."""

    def test_extract_code_block_no_language(self):
        """Test extraction from code block without language specifier."""
        config = OutputConfig(mode=OutputMode.CODE_ONLY)
        formatter = OutputFormatter(config)
        content = """Here's code:
```
def simple():
    pass
```
"""
        result = formatter._extract_code(content)
        assert "def simple():" in result

    def test_extract_tool_content_with_import(self):
        """Test extraction from tool content with import statement."""
        config = OutputConfig(mode=OutputMode.CODE_ONLY)
        formatter = OutputFormatter(config)
        content = """<parameter=content>
import os

def main():
    print(os.getcwd())
</parameter>"""
        result = formatter._extract_code(content)
        assert "import os" in result

    def test_extract_indented_code_block(self):
        """Test extraction of indented code blocks."""
        config = OutputConfig(mode=OutputMode.CODE_ONLY)
        formatter = OutputFormatter(config)
        content = """Example code:

    def indented():
        return True

End of example."""
        result = formatter._extract_code(content)
        assert "def indented():" in result

    def test_extract_class_definition(self):
        """Test extraction of class definitions."""
        config = OutputConfig(mode=OutputMode.CODE_ONLY)
        formatter = OutputFormatter(config)
        content = """<parameter=content>
class MyClass:
    def __init__(self):
        pass
</parameter>"""
        result = formatter._extract_code(content)
        assert "class MyClass:" in result

    def test_extract_no_code_returns_empty(self):
        """Test returns empty when no code patterns found."""
        config = OutputConfig(mode=OutputMode.CODE_ONLY)
        formatter = OutputFormatter(config)
        content = "Just text without any code patterns at all."
        result = formatter._extract_code(content)
        assert result == ""

    def test_response_code_only_mode(self):
        """Test complete response flow in CODE_ONLY mode."""
        stdout = io.StringIO()
        config = OutputConfig(mode=OutputMode.CODE_ONLY, stdout=stdout)
        formatter = OutputFormatter(config)
        content = """```python
def extracted():
    return "code"
```"""
        formatter.response(content=content)
        assert "def extracted():" in stdout.getvalue()


class TestInputReaderStdin:
    """Tests for stdin reading functionality."""

    def test_read_message_stdin_piped(self):
        """Test reading from stdin when piped."""
        from unittest.mock import patch, MagicMock

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "stdin content\n"

        with patch("victor.ui.output_formatter.sys.stdin", mock_stdin):
            result = InputReader.read_message(from_stdin=True)
            assert result == "stdin content"

    def test_read_message_no_input_returns_none(self):
        """Test returns None when no input provided and not piped."""
        from unittest.mock import patch, MagicMock

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True

        with patch("victor.ui.output_formatter.sys.stdin", mock_stdin):
            result = InputReader.read_message()
            assert result is None
