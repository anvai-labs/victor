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

"""Tests for Shell formatter."""

import pytest

from victor.tools.formatters.shell import ShellFormatter
from victor.tools.formatters.base import FormattedOutput


class TestShellFormatter:
    """Test ShellFormatter for shell command execution."""

    def test_validate_input_valid_with_success(self):
        """Test validation with success flag."""
        formatter = ShellFormatter()
        data = {"success": True, "exit_code": 0}
        assert formatter.validate_input(data) is True

    def test_validate_input_valid_with_stdout(self):
        """Test validation with stdout."""
        formatter = ShellFormatter()
        data = {"stdout": "command output"}
        assert formatter.validate_input(data) is True

    def test_validate_input_valid_with_command(self):
        """Test validation with command."""
        formatter = ShellFormatter()
        data = {"command": "ls -la"}
        assert formatter.validate_input(data) is True

    def test_validate_input_invalid(self):
        """Test validation with invalid data."""
        formatter = ShellFormatter()
        data = {"invalid": "data"}
        assert formatter.validate_input(data) is False

    def test_format_successful_command(self):
        """Test formatting successful command execution."""
        formatter = ShellFormatter()
        data = {
            "command": "ls -la",
            "success": True,
            "exit_code": 0,
            "stdout": "file1.py\nfile2.py",
            "duration_ms": 45,
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert result.format_type == "rich"
        assert "[green]✓ Command succeeded[/]" in result.content
        assert "45ms" in result.content
        assert "ls -la" in result.content

    def test_format_failed_command(self):
        """Test formatting failed command execution."""
        formatter = ShellFormatter()
        data = {
            "command": "false",
            "success": False,
            "exit_code": 1,
            "stderr": "Command failed",
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "[red]✗ Command failed" in result.content
        assert "exit code: 1" in result.content

    def test_format_with_stderr(self):
        """Test formatting command with stderr output."""
        formatter = ShellFormatter()
        data = {
            "command": "python script.py",
            "success": False,
            "exit_code": 2,
            "stderr": "SyntaxError: invalid syntax",
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "[red bold]Error Output:[/]" in result.content
        assert "SyntaxError" in result.content

    def test_format_with_long_output(self):
        """Test formatting command with long output."""
        formatter = ShellFormatter()
        long_output = "\n".join([f"line {i}" for i in range(200)])
        data = {"command": "cat large_file.txt", "success": True, "stdout": long_output}

        result = formatter.format(data, max_output_lines=50)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "more lines" in result.content

    def test_format_without_command_display(self):
        """Test formatting without showing command."""
        formatter = ShellFormatter()
        data = {"command": "secret command", "success": True, "stdout": "output"}

        result = formatter.format(data, show_command=False)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "$ secret command" not in result.content

    def test_format_with_error_message(self):
        """Test formatting with error message."""
        formatter = ShellFormatter()
        data = {
            "command": "rm file.txt",
            "success": False,
            "error": "No such file or directory",
        }

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "[red]Error: No such file or directory[/]" in result.content

    def test_summary_extraction_success(self):
        """Test summary extraction for successful command."""
        formatter = ShellFormatter()
        data = {"command": "ls", "success": True, "exit_code": 0}

        result = formatter.format(data)

        assert "✓" in result.summary
        assert "ls" in result.summary

    def test_summary_extraction_failure(self):
        """Test summary extraction for failed command."""
        formatter = ShellFormatter()
        data = {"command": "false", "success": False, "exit_code": 1}

        result = formatter.format(data)

        assert "✗" in result.summary
        assert "false" in result.summary  # Command name is in summary

    def test_fallback_formatter(self):
        """Test fallback formatter is returned."""
        formatter = ShellFormatter()
        fallback = formatter.get_fallback()

        assert fallback is not None
        assert hasattr(fallback, "format")
