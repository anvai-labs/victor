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

"""Tests for batch_processor_tool module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from victor.tools.batch_processor_tool import (
    batch,
    _parallel_search,
    _parallel_replace,
    _parallel_analyze,
    _DEFAULT_MAX_WORKERS,
)


class TestBatchConstants:
    """Tests for batch processor constants."""

    def test_default_max_workers(self):
        """Test default max workers constant."""
        assert _DEFAULT_MAX_WORKERS == 4


class TestBatchList:
    """Tests for batch list operation."""

    @pytest.mark.asyncio
    async def test_batch_list_success(self):
        """Test listing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "file1.txt").write_text("content1")
            (Path(tmpdir) / "file2.txt").write_text("content2")
            (Path(tmpdir) / "file3.py").write_text("content3")

            result = await batch(
                operation="list",
                path=tmpdir,
                file_pattern="*.txt",
            )

            assert result["success"] is True
            assert result["total_files"] == 2

    @pytest.mark.asyncio
    async def test_batch_list_empty(self):
        """Test listing with no matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await batch(
                operation="list",
                path=tmpdir,
                file_pattern="*.nonexistent",
            )

            assert result["success"] is True
            assert result["results"] == []

    @pytest.mark.asyncio
    async def test_batch_invalid_path(self):
        """Test batch with invalid path."""
        result = await batch(
            operation="list",
            path="/nonexistent/path/12345",
            file_pattern="*.*",
        )

        assert result["success"] is False
        assert "error" in result


class TestBatchSearch:
    """Tests for batch search operation."""

    @pytest.mark.asyncio
    async def test_batch_search_success(self):
        """Test searching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "file1.txt").write_text("hello world\ngoodbye world")
            (Path(tmpdir) / "file2.txt").write_text("no match here")
            (Path(tmpdir) / "file3.txt").write_text("hello again")

            result = await batch(
                operation="search",
                path=tmpdir,
                file_pattern="*.txt",
                pattern="hello",
            )

            assert result["success"] is True
            # Should find matches in file1 and file3

    @pytest.mark.asyncio
    async def test_batch_search_regex(self):
        """Test searching with regex."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("hello123\nworld456")
            (Path(tmpdir) / "file2.txt").write_text("no digits")

            result = await batch(
                operation="search",
                path=tmpdir,
                file_pattern="*.txt",
                pattern=r"hello\d+",
                regex=True,
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_batch_search_no_pattern(self):
        """Test search without pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("content")

            result = await batch(
                operation="search",
                path=tmpdir,
                file_pattern="*.txt",
            )

            assert result["success"] is False
            assert "pattern" in result["error"].lower()


class TestBatchReplace:
    """Tests for batch replace operation."""

    @pytest.mark.asyncio
    async def test_batch_replace_dry_run(self):
        """Test replace in dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.txt"
            file1.write_text("hello world")

            result = await batch(
                operation="replace",
                path=tmpdir,
                file_pattern="*.txt",
                find="hello",
                replace="hi",
                dry_run=True,
            )

            assert result["success"] is True
            # Original file should be unchanged
            assert file1.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_batch_replace_actual(self):
        """Test actual replacement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.txt"
            file1.write_text("hello world")

            result = await batch(
                operation="replace",
                path=tmpdir,
                file_pattern="*.txt",
                find="hello",
                replace="hi",
                dry_run=False,
            )

            assert result["success"] is True
            # File should be modified
            assert file1.read_text() == "hi world"

    @pytest.mark.asyncio
    async def test_batch_replace_regex(self):
        """Test replacement with regex."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.txt"
            file1.write_text("hello123 hello456")

            result = await batch(
                operation="replace",
                path=tmpdir,
                file_pattern="*.txt",
                find=r"hello\d+",
                replace="hi",
                regex=True,
                dry_run=False,
            )

            assert result["success"] is True
            assert file1.read_text() == "hi hi"

    @pytest.mark.asyncio
    async def test_batch_replace_no_find(self):
        """Test replace without find text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("content")

            result = await batch(
                operation="replace",
                path=tmpdir,
                file_pattern="*.txt",
            )

            assert result["success"] is False
            assert "find" in result["error"].lower()


class TestBatchAnalyze:
    """Tests for batch analyze operation."""

    @pytest.mark.asyncio
    async def test_batch_analyze_success(self):
        """Test analyzing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("line1\nline2\nline3")
            (Path(tmpdir) / "file2.txt").write_text("single line")

            result = await batch(
                operation="analyze",
                path=tmpdir,
                file_pattern="*.txt",
            )

            assert result["success"] is True
            assert "results" in result
            # Should have analysis for 2 files

    @pytest.mark.asyncio
    async def test_batch_analyze_empty(self):
        """Test analyzing with no files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await batch(
                operation="analyze",
                path=tmpdir,
                file_pattern="*.nonexistent",
            )

            assert result["success"] is True


class TestBatchTransform:
    """Tests for batch transform operation."""

    @pytest.mark.asyncio
    async def test_batch_transform_not_implemented(self):
        """Test transform operation is not yet implemented."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.txt"
            file1.write_text("hello world")

            result = await batch(
                operation="transform",
                path=tmpdir,
                file_pattern="*.txt",
                options={"transform": "uppercase"},
                dry_run=False,
            )

            # Transform is not implemented
            assert result["success"] is False
            assert "not yet implemented" in result["error"].lower()


class TestBatchInvalidOperation:
    """Tests for batch with invalid operations."""

    @pytest.mark.asyncio
    async def test_batch_invalid_operation(self):
        """Test batch with invalid operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("content")

            result = await batch(
                operation="invalid_operation",
                path=tmpdir,
                file_pattern="*.*",
            )

            assert result["success"] is False
            assert "unknown operation" in result["error"].lower()


class TestBatchMaxFiles:
    """Tests for batch max_files limit."""

    @pytest.mark.asyncio
    async def test_batch_max_files_limit(self):
        """Test max_files limits processed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many files
            for i in range(10):
                (Path(tmpdir) / f"file{i}.txt").write_text(f"content{i}")

            result = await batch(
                operation="list",
                path=tmpdir,
                file_pattern="*.txt",
                max_files=5,
            )

            assert result["success"] is True
            assert result["total_files"] <= 5


class TestParallelHelpers:
    """Tests for parallel helper functions."""

    @pytest.mark.asyncio
    async def test_parallel_search_basic(self):
        """Test _parallel_search function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.txt"
            file1.write_text("hello world")
            file2 = Path(tmpdir) / "file2.txt"
            file2.write_text("goodbye world")

            # Mock Progress to avoid terminal output in tests
            with patch("victor.tools.batch_processor_tool.Progress"):
                results = await _parallel_search([file1, file2], "hello", False)

            assert len(results) == 1
            assert "hello" in results[0]["matches"][0]["text"]

    @pytest.mark.asyncio
    async def test_parallel_search_regex(self):
        """Test _parallel_search with regex."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.txt"
            file1.write_text("hello123")

            with patch("victor.tools.batch_processor_tool.Progress"):
                results = await _parallel_search([file1], r"hello\d+", True)

            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_parallel_replace_basic(self):
        """Test _parallel_replace function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.txt"
            file1.write_text("hello world")

            with patch("victor.tools.batch_processor_tool.Progress"):
                results = await _parallel_replace([file1], "hello", "hi", False, False)

            assert len(results) == 1
            assert file1.read_text() == "hi world"

    @pytest.mark.asyncio
    async def test_parallel_replace_dry_run(self):
        """Test _parallel_replace dry run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.txt"
            file1.write_text("hello world")

            with patch("victor.tools.batch_processor_tool.Progress"):
                await _parallel_replace([file1], "hello", "hi", False, True)

            # File should be unchanged
            assert file1.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_parallel_analyze_basic(self):
        """Test _parallel_analyze function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.txt"
            file1.write_text("line1\nline2\nline3")

            with patch("victor.tools.batch_processor_tool.Progress"):
                results = await _parallel_analyze([file1])

            assert len(results) == 1
            assert results[0]["lines"] == 3
            assert "size" in results[0]
            assert results[0]["extension"] == ".txt"

    @pytest.mark.asyncio
    async def test_parallel_search_file_error(self):
        """Test _parallel_search handles file errors."""
        # Non-existent file
        files = [Path("/nonexistent/file.txt")]

        with patch("victor.tools.batch_processor_tool.Progress"):
            results = await _parallel_search(files, "test", False)

        # Should return empty list (error logged but not raised)
        assert results == []

    @pytest.mark.asyncio
    async def test_parallel_replace_file_error(self):
        """Test _parallel_replace handles file errors."""
        files = [Path("/nonexistent/file.txt")]

        with patch("victor.tools.batch_processor_tool.Progress"):
            results = await _parallel_replace(files, "a", "b", False, False)

        # Should return result with error
        assert len(results) == 1
        assert "error" in results[0]

    @pytest.mark.asyncio
    async def test_parallel_analyze_file_error(self):
        """Test _parallel_analyze handles file errors."""
        files = [Path("/nonexistent/file.txt")]

        with patch("victor.tools.batch_processor_tool.Progress"):
            results = await _parallel_analyze(files)

        # Should return result with error
        assert len(results) == 1
        assert "error" in results[0]
