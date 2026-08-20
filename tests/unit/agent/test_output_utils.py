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

"""Tests for output_utils module."""

from victor.tools.output_utils import (
    grep_lines,
    filter_paths,
    truncate_output,
    format_diff,
    summarize_changes,
    compact_file_list,
)


class TestGrepLines:
    """Tests for grep_lines function."""

    def test_grep_basic_match(self):
        """Test basic string matching."""
        content = "line one\nline two\nline three\nline four"
        result = grep_lines(content, "two")

        assert result.match_count == 1
        assert result.matches[0].line_number == 2
        assert result.matches[0].content == "line two"

    def test_grep_multiple_matches(self):
        """Test multiple matches."""
        content = "def foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass"
        result = grep_lines(content, "def")

        assert result.match_count == 3
        assert result.matches[0].line_number == 1
        assert result.matches[1].line_number == 3
        assert result.matches[2].line_number == 5

    def test_grep_with_context(self):
        """Test context lines."""
        content = "line 1\nline 2\nMATCH\nline 4\nline 5"
        result = grep_lines(content, "MATCH", context_before=1, context_after=1)

        assert result.match_count == 1
        assert len(result.matches[0].context_before) == 1
        assert len(result.matches[0].context_after) == 1
        assert "line 2" in result.matches[0].context_before[0]
        assert "line 4" in result.matches[0].context_after[0]

    def test_grep_case_insensitive(self):
        """Test case-insensitive matching."""
        content = "Hello World\nhello world\nHELLO WORLD"
        result = grep_lines(content, "hello", case_sensitive=False)

        assert result.match_count == 3

    def test_grep_regex(self):
        """Test regex pattern matching."""
        content = "def foo():\ndef bar(x):\ndef baz(x, y):"
        result = grep_lines(content, r"def \w+\(.*\):", is_regex=True)

        assert result.match_count == 3

    def test_grep_no_matches(self):
        """Test no matches found."""
        content = "line one\nline two\nline three"
        result = grep_lines(content, "nonexistent")

        assert result.match_count == 0
        assert "No matches" in result.to_string()

    def test_grep_max_matches(self):
        """Test max_matches limit."""
        content = "\n".join([f"match {i}" for i in range(100)])
        result = grep_lines(content, "match", max_matches=10)

        assert result.match_count == 10

    def test_grep_result_to_string(self):
        """Test GrepResult.to_string formatting."""
        content = "def foo():\n    return 1"
        result = grep_lines(content, "foo", file_path="test.py")
        output = result.to_string()

        assert "[1 matches in test.py" in output
        assert "foo" in output


class TestFilterPaths:
    """Tests for filter_paths function."""

    def test_filter_include_pattern(self):
        """Test include pattern filtering."""
        paths = ["foo.py", "bar.py", "baz.txt", "qux.js"]
        result = filter_paths(paths, include_pattern="*.py")

        assert result == ["foo.py", "bar.py"]

    def test_filter_exclude_pattern(self):
        """Test exclude pattern filtering."""
        paths = ["foo.py", "bar.py", "test_foo.py", "test_bar.py"]
        result = filter_paths(paths, exclude_pattern="test_*")

        assert result == ["foo.py", "bar.py"]

    def test_filter_extensions(self):
        """Test extension filtering."""
        paths = ["foo.py", "bar.ts", "baz.js", "qux.txt"]
        result = filter_paths(paths, extensions=["py", "ts"])

        assert result == ["foo.py", "bar.ts"]

    def test_filter_combined(self):
        """Test combined filters."""
        paths = ["src/foo.py", "src/bar.py", "test/test_foo.py", "src/baz.txt"]
        result = filter_paths(
            paths,
            include_pattern="src/*",
            extensions=["py"],
        )

        assert result == ["src/foo.py", "src/bar.py"]


class TestTruncateOutput:
    """Tests for truncate_output function."""

    def test_no_truncation_needed(self):
        """Test content below limit."""
        content = "short content"
        result = truncate_output(content, max_tokens=100)

        assert result == content

    def test_truncation(self):
        """Test content truncation."""
        content = "x" * 10000
        result = truncate_output(content, max_tokens=100)

        assert len(result) < len(content)
        assert "truncated" in result

    def test_truncation_at_line_boundary(self):
        """Test truncation at line boundary."""
        content = "line 1\nline 2\nline 3\n" + "x" * 10000
        result = truncate_output(content, max_tokens=50)

        # Should end at a line boundary, not mid-line
        assert result.count("\n") >= 1


class TestFormatDiff:
    """Tests for format_diff function."""

    def test_basic_diff(self):
        """Test basic diff generation."""
        original = "line 1\nline 2\nline 3"
        modified = "line 1\nline 2 modified\nline 3"
        result = format_diff(original, modified)

        assert "---" in result  # Diff header
        assert "+++" in result
        assert "-line 2" in result
        assert "+line 2 modified" in result


class TestSummarizeChanges:
    """Tests for summarize_changes function."""

    def test_summary_added(self):
        """Test counting added lines."""
        original = "line 1\nline 2"
        modified = "line 1\nline 2\nline 3\nline 4"
        result = summarize_changes(original, modified)

        assert result["lines_added"] == 2
        assert result["lines_removed"] == 0

    def test_summary_removed(self):
        """Test counting removed lines."""
        original = "line 1\nline 2\nline 3"
        modified = "line 1"
        result = summarize_changes(original, modified)

        assert result["lines_removed"] == 2


class TestCompactFileList:
    """Tests for compact_file_list function."""

    def test_basic_compact(self):
        """Test basic compaction."""
        files = [
            {"name": "foo.py", "type": "file"},
            {"name": "bar.py", "type": "file"},
            {"name": "src", "type": "directory"},
        ]
        result = compact_file_list(files)

        assert result == ["foo.py", "bar.py", "src/"]

    def test_group_by_extension(self):
        """Test grouping by extension."""
        files = [
            {"name": "foo.py", "type": "file"},
            {"name": "bar.py", "type": "file"},
            {"name": "baz.ts", "type": "file"},
        ]
        result = compact_file_list(files, group_by_extension=True)

        assert ".py" in result
        assert ".ts" in result
        assert len(result[".py"]) == 2
        assert len(result[".ts"]) == 1

    def test_max_items(self):
        """Test max_items limit."""
        files = [{"name": f"file{i}.py", "type": "file"} for i in range(100)]
        result = compact_file_list(files, max_items=10)

        assert len(result) == 10
