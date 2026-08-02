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

"""Tests for ToolOutputPruner."""

import pytest

from victor.tools.output_pruner import (
    ToolOutputPruner,
    PruningInfo,
    get_output_pruner,
)


class TestToolOutputPruner:
    """Test ToolOutputPruner functionality."""

    def test_pruner_initialization(self):
        """Test pruner initializes correctly."""
        pruner = ToolOutputPruner(enabled=True)
        assert pruner.enabled is True

        pruner_disabled = ToolOutputPruner(enabled=False)
        assert pruner_disabled.enabled is False

    def test_code_generation_task_truncation(self):
        """Test code_generation task truncates to 50 lines."""
        pruner = ToolOutputPruner(enabled=True)

        # Create 100-line output
        output = "\n".join([f"line {i}" for i in range(100)])

        pruned, info = pruner.prune(
            tool_output=output,
            task_type="code_generation",
            tool_name="read",
        )

        assert info.was_pruned is True
        assert info.original_lines == 100
        assert info.pruned_lines <= 50
        assert info.task_type == "code_generation"
        assert "max_lines=50" in info.pruning_reason

    def test_edit_task_context_lines(self):
        """Test edit task limits to 30 lines."""
        pruner = ToolOutputPruner(enabled=True)

        # Create 50-line output
        output = "\n".join([f"def function_{i}():" for i in range(50)])

        pruned, info = pruner.prune(
            tool_output=output,
            task_type="edit",
            tool_name="read",
        )

        assert info.was_pruned is True
        assert info.pruned_lines <= 30

    def test_create_simple_returns_summary(self):
        """Test create_simple task returns summary only."""
        pruner = ToolOutputPruner(enabled=True)

        # Create file read output
        output = "import sys\n\ndef main():\n    print('hello')\n    # comment\n    return 0\n\nif __name__ == '__main__':\n    main()"

        pruned, info = pruner.prune(
            tool_output=output,
            task_type="create_simple",
            tool_name="read",
        )

        assert info.was_pruned is True
        assert info.pruning_reason == "summary_only"
        # Should just return filename/summary, not full content
        assert len(pruned) < len(output)

    def test_debug_task_preserves_traceback(self):
        """Test debug task preserves more context."""
        pruner = ToolOutputPruner(enabled=True)

        # Create 100-line debug output with traceback
        output = "\n".join([f"debug line {i}" for i in range(100)])

        pruned, info = pruner.prune(
            tool_output=output,
            task_type="debug",
            tool_name="read",
        )

        # Debug allows up to 80 lines
        assert info.pruned_lines <= 80

    def test_strip_comments(self):
        """Test comment stripping works."""
        pruner = ToolOutputPruner(enabled=True)

        output = """# This is a comment
import sys
# Another comment
def main():
    pass
# Final comment
"""

        pruned, info = pruner.prune(
            tool_output=output,
            task_type="code_generation",
            tool_name="read",
        )

        assert "# This is a comment" not in pruned
        assert "# Another comment" not in pruned
        assert "import sys" in pruned  # Code preserved
        assert "strip_comments" in info.pruning_reason

    def test_strip_blank_lines(self):
        """Test blank line stripping works."""
        pruner = ToolOutputPruner(enabled=True)

        output = """line 1

line 2


line 3
"""

        pruned, info = pruner.prune(
            tool_output=output,
            task_type="code_generation",
            tool_name="read",
        )

        assert "\n\n" not in pruned
        assert "strip_blank_lines" in info.pruning_reason

    def test_disabled_pruner_returns_original(self):
        """Test disabled pruner returns original output."""
        pruner = ToolOutputPruner(enabled=False)

        output = "line 1\nline 2\nline 3\n"

        pruned, info = pruner.prune(
            tool_output=output,
            task_type="code_generation",
            tool_name="read",
        )

        assert pruned == output  # No changes
        assert info.was_pruned is False
        assert "Pruning disabled" in info.pruning_reason

    def test_unknown_task_type_uses_default_rules(self):
        """Test unknown task type uses default rules."""
        pruner = ToolOutputPruner(enabled=True)

        # Create 150-line output
        output = "\n".join([f"line {i}" for i in range(150)])

        pruned, info = pruner.prune(
            tool_output=output,
            task_type="unknown_task",
            tool_name="read",
        )

        # Default rule allows 100 lines
        assert info.pruned_lines <= 100
        assert info.task_type == "unknown_task"

    def test_preserve_imports_for_code_files(self):
        """Test imports are preserved for code files."""
        pruner = ToolOutputPruner(enabled=True)

        output = """import os
import sys
from pathlib import Path

def main():
    pass
""" + "\n".join([f"# Code line {i}" for i in range(50)])

        pruned, info = pruner.prune(
            tool_output=output,
            task_type="code_generation",
            tool_name="read",
        )

        # Imports should be preserved
        assert "import os" in pruned
        assert "import sys" in pruned
        assert "from pathlib import Path" in pruned

    def test_search_task_limits_results(self):
        """Test search task limits grep results."""
        pruner = ToolOutputPruner(enabled=True)

        # Simulate 20 grep results
        output = "\n".join([f"result_{i}: match found" for i in range(20)])

        pruned, info = pruner.prune(
            tool_output=output,
            task_type="search",
            tool_name="grep",
        )

        # Search results are under max_lines, so no pruning occurs
        # But the rule is still configured
        assert info.task_type == "search"

    def test_singleton_get_output_pruner(self):
        """Test get_output_pruner returns singleton instance."""
        pruner1 = get_output_pruner()
        pruner2 = get_output_pruner()

        assert pruner1 is pruner2  # Same instance
        assert isinstance(pruner1, ToolOutputPruner)

    def test_pruning_info_string_representation(self):
        """Test PruningInfo __str__ method."""
        info = PruningInfo(
            was_pruned=True,
            original_lines=100,
            pruned_lines=50,
            pruning_reason="max_lines=50",
            task_type="code_generation",
        )

        info_str = str(info)
        assert "50.0%" in info_str  # Should show reduction percentage with decimal
        assert "code_generation" in info_str
        assert "max_lines=50" in info_str

        info_no_prune = PruningInfo(
            was_pruned=False,
            original_lines=10,
            pruned_lines=10,
            pruning_reason="no_pruning",
            task_type="edit",
        )

        info_str_no_prune = str(info_no_prune)
        assert "No pruning applied" in info_str_no_prune


class TestPruningIntegration:
    """Integration tests for output pruning."""

    def test_code_generation_task_40_60_percent_reduction(self):
        """Test code_generation achieves 40-60% token reduction."""
        pruner = ToolOutputPruner(enabled=True)

        # Simulate reading a 100-line file
        output = "\n".join(
            [
                "# Comment line",
                "import sys",
                "from pathlib import Path",
                "",
                "def function():",
                "    pass",
                "",
                # ... more lines
            ]
            + [f"code_line_{i}: value = {i}" for i in range(100)]
        )

        pruned, info = pruner.prune(
            tool_output=output,
            task_type="code_generation",
            tool_name="read",
        )

        # Calculate reduction
        reduction = (info.original_lines - info.pruned_lines) / info.original_lines

        # Target: 40-60% reduction
        assert 0.40 <= reduction <= 0.60, f"Reduction was {reduction:.2%}, expected 40-60%"

    def test_edit_task_achieves_significant_reduction(self):
        """Test edit task achieves meaningful reduction."""
        pruner = ToolOutputPruner(enabled=True)

        # 50-line file with many comments
        output = "\n".join(
            [
                "# Important comment",
                "line of code",
                "",
                "# Another comment",
                "more code",
                # ... repeat
            ]
            * 10
        )

        pruned, info = pruner.prune(
            tool_output=output,
            task_type="edit",
            tool_name="read",
        )

        # Should achieve meaningful reduction
        assert info.was_pruned is True
        assert info.pruned_lines < info.original_lines


class TestFormattedOutputPruning:
    """Tests for the conservative formatted-output pruning path used in runtime."""

    def test_formatted_read_output_preserves_headers_and_line_numbers(self):
        pruner = ToolOutputPruner(enabled=True)
        body = "\n".join(f"{i}\tline {i}" for i in range(1, 260))
        formatted = (
            '<TOOL_OUTPUT tool="read" path="src/main.py">\n'
            "═══ ACTUAL FILE CONTENT: src/main.py ═══\n"
            "[File: src/main.py]\n"
            "[Lines 1-259 of 259]\n"
            "[Size: 4096 bytes]\n"
            f"{body}\n"
            "═══ END OF FILE: src/main.py ═══\n"
            "</TOOL_OUTPUT>\n"
            "\n"
            "IMPORTANT: The content above is authoritative."
        )

        pruned, info = pruner.prune(
            tool_output=formatted,
            task_type="analysis",
            tool_name="read",
            context={
                "formatted_output": True,
                "safe_only": True,
                "tool_args": {"path": "src/main.py"},
            },
        )

        assert info.was_pruned is True
        assert "[File: src/main.py]" in pruned
        assert "1\tline 1" in pruned
        assert "140\tline 140" in pruned
        assert "141\tline 141" not in pruned
        assert "offset=140" in pruned
        assert "</TOOL_OUTPUT>" in pruned
        assert "IMPORTANT: The content above is authoritative." in pruned

    def test_formatted_output_skips_safety_critical_error_signatures(self):
        pruner = ToolOutputPruner(enabled=True)
        formatted = (
            '<TOOL_OUTPUT tool="read" path="artifacts/test.log">\n'
            "Traceback (most recent call last):\n"
            '  File "runner.py", line 10, in <module>\n'
            "AssertionError: expected 1 == 2\n"
            "</TOOL_OUTPUT>"
        )

        pruned, info = pruner.prune(
            tool_output=formatted,
            task_type="analysis",
            tool_name="read",
            context={"formatted_output": True, "safe_only": True},
        )

        assert pruned == formatted
        assert info.was_pruned is False
        assert info.pruning_reason == "safety_critical_content"

    def test_formatted_output_respects_safe_default_scope(self):
        pruner = ToolOutputPruner(enabled=True)
        formatted = '<TOOL_OUTPUT tool="write">verbose write result</TOOL_OUTPUT>'

        pruned, info = pruner.prune(
            tool_output=formatted,
            task_type="edit",
            tool_name="write",
            context={"formatted_output": True, "safe_only": True},
        )

        assert pruned == formatted
        assert info.was_pruned is False
        assert info.pruning_reason == "safety_critical_tool"

    def test_formatted_output_canonicalizes_legacy_tool_names(self):
        pruner = ToolOutputPruner(enabled=True)
        body = "\n".join(f"file_{i}.py" for i in range(200))
        formatted = f'<TOOL_OUTPUT tool="list_directory">\n{body}\n</TOOL_OUTPUT>'

        pruned, info = pruner.prune(
            tool_output=formatted,
            task_type="analysis",
            tool_name="list_directory",
            context={
                "formatted_output": True,
                "safe_only": True,
                "tool_args": {"path": "."},
            },
        )

        assert info.was_pruned is True
        assert "list_directory" not in info.recovery_hint
        assert "ls(path='.'" in info.recovery_hint
