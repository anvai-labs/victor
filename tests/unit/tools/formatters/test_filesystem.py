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

"""Tests for FileSystem formatter."""

import pytest

from victor.tools.formatters.filesystem import FileSystemFormatter
from victor.tools.formatters.base import FormattedOutput


class TestFileSystemFormatter:
    """Test FileSystemFormatter for file system operations."""

    def test_validate_input_valid_with_entries(self):
        """Test validation with entries data."""
        formatter = FileSystemFormatter()
        data = {"entries": ["file1.py", "file2.txt"]}
        assert formatter.validate_input(data) is True

    def test_validate_input_valid_with_content(self):
        """Test validation with content data."""
        formatter = FileSystemFormatter()
        data = {"content": "file content", "path": "/path/to/file"}
        assert formatter.validate_input(data) is True

    def test_validate_input_valid_with_matches(self):
        """Test validation with matches data."""
        formatter = FileSystemFormatter()
        data = {"matches": ["/path/to/file1.py", "/path/to/file2.py"]}
        assert formatter.validate_input(data) is True

    def test_validate_input_invalid(self):
        """Test validation with invalid data."""
        formatter = FileSystemFormatter()
        data = {"invalid": "data"}
        assert formatter.validate_input(data) is False

    def test_format_directory_listing(self):
        """Test formatting directory listing."""
        formatter = FileSystemFormatter()
        data = {"directories": ["src", "tests"], "files": ["main.py", "config.yaml"]}

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert result.format_type == "rich"
        assert "Directories" in result.content
        assert "Files" in result.content
        assert "[bold blue]" in result.content

    def test_format_file_content(self):
        """Test formatting file content."""
        formatter = FileSystemFormatter()
        data = {"path": "/path/to/file.py", "content": "print('hello world')"}

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "/path/to/file.py" in result.content
        assert "print('hello world')" in result.content

    def test_format_search_results(self):
        """Test formatting search/find results."""
        formatter = FileSystemFormatter()
        data = {"matches": ["/src/file1.py", "/tests/file2.py"], "path": "."}

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True
        assert "Search results" in result.content
        assert "/src/file1.py" in result.content

    def test_format_with_max_items_limit(self):
        """Test formatting with max_items limit."""
        formatter = FileSystemFormatter()
        data = {"files": [f"file{i}.py" for i in range(100)]}

        result = formatter.format(data, max_items=10)

        assert isinstance(result, FormattedOutput)
        assert "more files" in result.content

    def test_format_with_max_lines_limit(self):
        """Test formatting with max_lines limit for content."""
        formatter = FileSystemFormatter()
        long_content = "\n".join([f"line {i}" for i in range(200)])
        data = {"path": "/path/to/file.txt", "content": long_content}

        result = formatter.format(data, max_lines=50)

        assert isinstance(result, FormattedOutput)
        assert "more lines" in result.content

    def test_format_empty_directory(self):
        """Test formatting empty directory."""
        formatter = FileSystemFormatter()
        data = {"entries": [], "directories": []}

        result = formatter.format(data)

        assert isinstance(result, FormattedOutput)
        assert result.contains_markup is True

    def test_summary_extraction_directory(self):
        """Test summary extraction for directory listing."""
        formatter = FileSystemFormatter()
        data = {"entries": ["file1.py", "file2.py", "file3.py"]}

        result = formatter.format(data)

        assert "3" in result.summary

    def test_summary_extraction_content(self):
        """Test summary extraction for file content."""
        formatter = FileSystemFormatter()
        data = {"content": "test content", "path": "/path/to/file"}

        result = formatter.format(data)

        assert "/path/to/file" in result.summary

    def test_fallback_formatter(self):
        """Test fallback formatter is returned."""
        formatter = FileSystemFormatter()
        fallback = formatter.get_fallback()

        assert fallback is not None
        assert hasattr(fallback, "format")
