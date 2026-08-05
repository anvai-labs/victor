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

"""Tests for codebase/indexer module."""

import pytest

pytest.importorskip("victor_coding.codebase.indexer")

import tempfile
from pathlib import Path

from victor_coding.codebase.indexer import Symbol, FileMetadata, CodebaseIndex


class TestSymbol:
    """Tests for Symbol model."""

    def test_symbol_creation(self):
        """Test creating a Symbol."""
        symbol = Symbol(
            name="my_function",
            type="function",
            file_path="/test/file.py",
            line_number=10,
        )
        assert symbol.name == "my_function"
        assert symbol.type == "function"
        assert symbol.file_path == "/test/file.py"
        assert symbol.line_number == 10

    def test_symbol_with_optional_fields(self):
        """Test Symbol with optional fields."""
        symbol = Symbol(
            name="MyClass",
            type="class",
            file_path="/test/file.py",
            line_number=20,
            docstring="A test class",
            signature="class MyClass(BaseClass)",
        )
        assert symbol.docstring == "A test class"
        assert symbol.signature == "class MyClass(BaseClass)"

    def test_symbol_references(self):
        """Test Symbol references."""
        symbol = Symbol(
            name="helper",
            type="function",
            file_path="/test/utils.py",
            line_number=5,
            references=["/test/main.py", "/test/app.py"],
        )
        assert len(symbol.references) == 2


class TestFileMetadata:
    """Tests for FileMetadata model."""

    def test_file_metadata_creation(self):
        """Test creating FileMetadata."""
        metadata = FileMetadata(
            path="/test/file.py",
            language="python",
            last_modified=1234567890.0,
            size=1024,
            lines=50,
        )
        assert metadata.path == "/test/file.py"
        assert metadata.language == "python"
        assert metadata.lines == 50

    def test_file_metadata_with_symbols(self):
        """Test FileMetadata with symbols."""
        symbol = Symbol(name="func", type="function", file_path="/test.py", line_number=1)
        metadata = FileMetadata(
            path="/test.py",
            language="python",
            symbols=[symbol],
            last_modified=0.0,
            size=100,
            lines=10,
        )
        assert len(metadata.symbols) == 1

    def test_file_metadata_with_imports(self):
        """Test FileMetadata with imports."""
        metadata = FileMetadata(
            path="/test.py",
            language="python",
            imports=["os", "sys", "pathlib"],
            last_modified=0.0,
            size=100,
            lines=10,
        )
        assert "os" in metadata.imports


class TestCodebaseIndex:
    """Tests for CodebaseIndex class."""

    def test_codebase_index_init(self):
        """Test CodebaseIndex initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = CodebaseIndex(root_path=tmpdir)
            assert index.root == Path(tmpdir).resolve()
            assert len(index.ignore_patterns) > 0

    def test_codebase_index_custom_ignore_patterns(self):
        """Test CodebaseIndex with custom ignore patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            patterns = ["*.log", "temp/"]
            index = CodebaseIndex(root_path=tmpdir, ignore_patterns=patterns)
            assert index.ignore_patterns == patterns

    def test_codebase_index_default_ignore(self):
        """Test CodebaseIndex has default ignore patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = CodebaseIndex(root_path=tmpdir)
            # Should have common patterns like venv, .git, etc.
            assert "venv/" in index.ignore_patterns or ".git/" in index.ignore_patterns
