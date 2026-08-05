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

"""Unit tests for bash-style tool argument formatting.

These tests verify the Phase 3 console transcript improvements that add
bash CLI-like syntax for tool execution display.
"""

import pytest

from victor.ui.rendering.utils import (
    format_tool_args_bash_style,
    format_bash_command_invocation,
)


class TestBashStyleFormatting:
    """Tests for bash-style tool argument formatting (Phase 3)."""

    def test_empty_args_returns_empty_string(self):
        """Empty arguments dict should return empty string."""
        result = format_tool_args_bash_style({})
        assert result == ""

    def test_single_string_arg(self):
        """Single string argument formatted as --key='value'."""
        result = format_tool_args_bash_style({"path": "main.py"})
        assert result == "--path='main.py'"

    def test_multiple_string_args(self):
        """Multiple string arguments formatted with space separator."""
        result = format_tool_args_bash_style({"path": "main.py", "query": "auth"})
        assert "--path='main.py'" in result
        assert "--query='auth'" in result

    def test_integer_arg(self):
        """Integer argument formatted as --key=value."""
        result = format_tool_args_bash_style({"limit": 100})
        assert result == "--limit=100"

    def test_float_arg(self):
        """Float argument formatted as --key=value."""
        result = format_tool_args_bash_style({"threshold": 0.85})
        assert result == "--threshold=0.85"

    def test_boolean_true_shows_flag(self):
        """Boolean True shows flag, False omits it."""
        result = format_tool_args_bash_style({"recursive": True, "verbose": False})
        assert "--recursive" in result
        assert "verbose" not in result

    def test_none_value_omitted(self):
        """None values should be omitted."""
        result = format_tool_args_bash_style({"path": "file.py", "optional": None})
        assert result == "--path='file.py'"

    def test_list_arg_shows_count(self):
        """List argument shows count in brackets."""
        result = format_tool_args_bash_style({"files": ["a.py", "b.py", "c.py"]})
        assert result == "--files=[3]"

    def test_list_of_dicts_shows_summary(self):
        """List of dicts (edit ops) shows type:path summary."""
        result = format_tool_args_bash_style({"ops": [{"type": "edit", "path": "main.py"}]})
        assert result == "--ops=[edit:main.py]"

    def test_multiple_edit_ops_shows_count(self):
        """Multiple edit ops show first + count."""
        result = format_tool_args_bash_style(
            {
                "ops": [
                    {"type": "edit", "path": "main.py"},
                    {"type": "insert", "path": "utils.py"},
                ]
            }
        )
        assert "+1" in result  # Shows +1 for additional ops

    def test_long_string_truncated(self):
        """Long strings are truncated to 60 chars."""
        long_string = "a" * 100
        result = format_tool_args_bash_style({"path": long_string})
        assert len(result) < len(long_string)
        assert "..." in result

    def test_string_with_single_quote_escaped(self):
        """Single quotes in string values are escaped."""
        result = format_tool_args_bash_style({"pattern": "it's"})
        assert "\\'" in result  # Backslash escape

    def test_max_args_limits_output(self):
        """max_args parameter limits number of arguments shown."""
        args = {f"arg{i}": f"value{i}" for i in range(10)}
        result = format_tool_args_bash_style(args, max_args=3)
        # Should show 3 args plus ellipsis
        assert "..." in result
        parts = result.split(" ")
        assert len(parts) <= 4  # 3 args + "..."

    def test_unknown_type_shows_placeholder(self):
        """Unknown types show ... placeholder."""
        result = format_tool_args_bash_style({"object": object()})
        assert "--object=..." in result


class TestBashCommandInvocation:
    """Tests for complete bash command invocation formatting."""

    def test_command_with_no_args(self):
        """Command with no args shows just tool name."""
        result = format_bash_command_invocation("version", {})
        assert "$" in result
        assert "version" in result
        assert "--" not in result  # No flags

    def test_command_with_args(self):
        """Command with args shows flags."""
        result = format_bash_command_invocation("code_search", {"query": "auth"})
        assert "$" in result
        assert "code_search" in result
        assert "--query='auth'" in result

    def test_rich_markup_in_result(self):
        """Result contains Rich markup for styling."""
        result = format_bash_command_invocation("test", {})
        assert "[dim]$[/]" in result
        assert "[bold cyan]" in result
        assert "[/]" in result  # Closing tag

    def test_multiple_args_formatted_correctly(self):
        """Multiple arguments all formatted."""
        result = format_bash_command_invocation(
            "grep", {"pattern": "TODO", "path": "src/", "recursive": True}
        )
        assert "$" in result
        assert "grep" in result
        assert "--pattern='TODO'" in result
        assert "--path='src/'" in result
        assert "--recursive" in result

    def test_complex_edit_command(self):
        """Complex edit command with ops list."""
        result = format_bash_command_invocation(
            "edit_files",
            {"ops": [{"type": "replace", "path": "main.py", "old": "foo", "new": "bar"}]},
        )
        assert "$" in result
        assert "edit_files" in result
        assert "--ops=" in result  # Should show ops summary


class TestBashFormattingIntegration:
    """Integration tests for bash-style formatting across scenarios."""

    def test_code_search_command(self):
        """Real code_search command scenario."""
        result = format_bash_command_invocation(
            "code_search", {"query": "authentication", "path": "src/auth/", "limit": 10}
        )
        assert "$" in result
        assert "code_search" in result
        assert "--query='authentication'" in result
        assert "--path='src/auth/'" in result
        assert "--limit=10" in result

    def test_grep_command(self):
        """Real grep command scenario."""
        result = format_bash_command_invocation(
            "grep", {"pattern": "TODO", "path": ".", "recursive": True}
        )
        assert "$" in result
        assert "grep" in result
        assert "--pattern='TODO'" in result
        assert "--path='.'" in result
        assert "--recursive" in result

    def test_edit_command(self):
        """Real edit command scenario."""
        result = format_bash_command_invocation(
            "edit", {"path": "main.py", "old_string": "foo", "new_string": "bar"}
        )
        assert "$" in result
        assert "edit" in result
        assert "--path='main.py'" in result
        # old_string and new_string should be present (may be truncated if long)

    def test_execute_command(self):
        """Real execute_bash command scenario."""
        result = format_bash_command_invocation(
            "execute_bash", {"command": "pytest tests/unit/ui/ -v"}
        )
        assert "$" in result
        assert "execute_bash" in result
        assert "--command='pytest tests/unit/ui/ -v'" in result

    def test_file_read_command(self):
        """Real file read scenario."""
        result = format_bash_command_invocation("read_file", {"path": "README.md"})
        assert "$" in result
        assert "read_file" in result
        assert "--path='README.md'" in result

    def test_list_directory_command(self):
        """Real list_directory scenario."""
        result = format_bash_command_invocation(
            "list_directory", {"path": "src/", "recursive": True}
        )
        assert "$" in result
        assert "list_directory" in result
        assert "--path='src/'" in result
        assert "--recursive" in result
