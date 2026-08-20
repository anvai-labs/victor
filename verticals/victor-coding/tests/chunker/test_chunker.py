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

"""Tests for the robust code chunking module."""

import pytest

# Skip all tests in this module if victor-coding package is not installed
pytest.importorskip("victor_coding.codebase.chunker")

import tempfile
from pathlib import Path

from victor_coding.codebase.chunker import (
    CodeChunker,
    ChunkConfig,
    ChunkingStrategy,
    ChunkType,
    CodeChunk,
    chunk_codebase,
)


class TestChunkConfig:
    """Tests for ChunkConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkConfig()
        assert config.strategy == ChunkingStrategy.BODY_AWARE
        assert config.max_chunk_tokens == 512
        assert config.overlap_tokens == 64
        assert config.large_symbol_threshold == 30
        assert config.include_file_summary is True
        assert config.include_class_summary is True

    def test_max_chunk_chars(self):
        """Test character calculation from tokens."""
        config = ChunkConfig(max_chunk_tokens=512, chars_per_token=4)
        assert config.max_chunk_chars == 2048

    def test_overlap_chars(self):
        """Test overlap character calculation."""
        config = ChunkConfig(overlap_tokens=64, chars_per_token=4)
        assert config.overlap_chars == 256


class TestCodeChunk:
    """Tests for CodeChunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a CodeChunk."""
        chunk = CodeChunk(
            id="test.py:my_func",
            content="def my_func(): pass",
            chunk_type=ChunkType.METHOD_HEADER,
            file_path="test.py",
            symbol_name="my_func",
            symbol_type="function",
            line_start=1,
            line_end=1,
        )
        assert chunk.id == "test.py:my_func"
        assert chunk.chunk_type == ChunkType.METHOD_HEADER

    def test_to_document(self):
        """Test conversion to document format."""
        chunk = CodeChunk(
            id="test.py:MyClass",
            content="class MyClass: pass",
            chunk_type=ChunkType.CLASS_SUMMARY,
            file_path="test.py",
            symbol_name="MyClass",
            symbol_type="class",
            line_start=1,
            line_end=5,
            parent_id=None,
            metadata={"method_count": 3},
        )
        doc = chunk.to_document()
        assert doc["id"] == "test.py:MyClass"
        assert doc["content"] == "class MyClass: pass"
        assert doc["metadata"]["chunk_type"] == "class_summary"
        assert doc["metadata"]["method_count"] == 3


class TestCodeChunker:
    """Tests for CodeChunker."""

    @pytest.fixture
    def simple_file_content(self):
        """Simple Python file for testing."""
        return '''"""Module docstring."""

def simple_func():
    """Simple function."""
    return 42

class MyClass:
    """A test class."""

    def method_one(self):
        """Method one."""
        return 1

    def method_two(self, x: int) -> int:
        """Method two with args."""
        return x * 2
'''

    @pytest.fixture
    def large_function_content(self):
        """Python file with a large function for body chunking."""
        lines = ['"""Module with large function."""', "", "def large_function():"]
        lines.append('    """A very large function."""')
        # Add 50 lines of code
        for i in range(50):
            lines.append(f"    x_{i} = {i}  # Line {i}")
        lines.append("    return x_0")
        return "\n".join(lines)

    def test_chunk_simple_file(self, simple_file_content):
        """Test chunking a simple file."""
        chunker = CodeChunker()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(simple_file_content)
            f.flush()
            chunks = chunker.chunk_file(Path(f.name), "test.py")

        # Should have file summary, function header, class summary, and method headers
        assert len(chunks) >= 5

        chunk_types = [c.chunk_type for c in chunks]
        assert ChunkType.FILE_SUMMARY in chunk_types
        assert ChunkType.METHOD_HEADER in chunk_types
        assert ChunkType.CLASS_SUMMARY in chunk_types

    def test_chunk_with_file_summary_disabled(self, simple_file_content):
        """Test chunking with file summary disabled."""
        config = ChunkConfig(include_file_summary=False)
        chunker = CodeChunker(config)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(simple_file_content)
            f.flush()
            chunks = chunker.chunk_file(Path(f.name), "test.py")

        chunk_types = [c.chunk_type for c in chunks]
        assert ChunkType.FILE_SUMMARY not in chunk_types

    def test_chunk_large_function_creates_body_chunks(self, large_function_content):
        """Test that large functions get body chunks."""
        config = ChunkConfig(
            strategy=ChunkingStrategy.BODY_AWARE,
            large_symbol_threshold=30,  # Function has 50+ lines
            max_chunk_tokens=128,  # Smaller chunks (~512 chars) to trigger body chunking
        )
        chunker = CodeChunker(config)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(large_function_content)
            f.flush()
            chunks = chunker.chunk_file(Path(f.name), "test.py")

        chunk_types = [c.chunk_type for c in chunks]
        assert ChunkType.METHOD_HEADER in chunk_types
        # Large function should have body chunks
        assert ChunkType.METHOD_BODY in chunk_types

    def test_symbol_only_strategy_no_body_chunks(self, large_function_content):
        """Test SYMBOL_ONLY strategy doesn't create body chunks."""
        config = ChunkConfig(strategy=ChunkingStrategy.SYMBOL_ONLY)
        chunker = CodeChunker(config)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(large_function_content)
            f.flush()
            chunks = chunker.chunk_file(Path(f.name), "test.py")

        chunk_types = [c.chunk_type for c in chunks]
        assert ChunkType.METHOD_BODY not in chunk_types

    def test_import_block_extraction(self):
        """Test import block extraction."""
        content = """
import os
import sys
from pathlib import Path
from typing import List, Dict

def my_func():
    pass
"""
        chunker = CodeChunker()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(content)
            f.flush()
            chunks = chunker.chunk_file(Path(f.name), "test.py")

        import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT_BLOCK]
        assert len(import_chunks) == 1
        assert "os" in import_chunks[0].content
        assert "pathlib" in import_chunks[0].content

    def test_parent_id_hierarchy(self, simple_file_content):
        """Test that methods have correct parent_id."""
        chunker = CodeChunker()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(simple_file_content)
            f.flush()
            chunks = chunker.chunk_file(Path(f.name), "test.py")

        # Find method chunks
        method_chunks = [c for c in chunks if c.symbol_name == "method_one"]
        assert len(method_chunks) >= 1

        # Method should have class as parent
        for chunk in method_chunks:
            if chunk.parent_id:
                assert "MyClass" in chunk.parent_id

    def test_async_function_handling(self):
        """Test handling of async functions."""
        content = '''
async def async_func(x: int) -> str:
    """An async function."""
    await something()
    return str(x)
'''
        chunker = CodeChunker()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(content)
            f.flush()
            chunks = chunker.chunk_file(Path(f.name), "test.py")

        func_chunks = [c for c in chunks if c.symbol_name == "async_func"]
        assert len(func_chunks) >= 1
        assert func_chunks[0].metadata.get("is_async") is True

    def test_decorator_extraction(self):
        """Test decorator information is captured."""
        content = '''
@staticmethod
@cache
def decorated_func():
    """A decorated function."""
    pass
'''
        chunker = CodeChunker()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(content)
            f.flush()
            chunks = chunker.chunk_file(Path(f.name), "test.py")

        func_chunks = [c for c in chunks if c.symbol_name == "decorated_func"]
        assert len(func_chunks) >= 1
        assert "Decorators:" in func_chunks[0].content

    def test_syntax_error_handling(self):
        """Test graceful handling of syntax errors."""
        content = "def broken(: pass"  # Invalid syntax
        chunker = CodeChunker()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(content)
            f.flush()
            chunks = chunker.chunk_file(Path(f.name), "test.py")

        # Should return empty list, not crash
        assert chunks == []

    def test_class_inheritance_extraction(self):
        """Test class inheritance is captured."""
        content = '''
class Child(Parent, Mixin):
    """A child class."""
    pass
'''
        chunker = CodeChunker()

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(content)
            f.flush()
            chunks = chunker.chunk_file(Path(f.name), "test.py")

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS_SUMMARY]
        assert len(class_chunks) == 1
        assert "Inherits:" in class_chunks[0].content
        assert "Parent" in class_chunks[0].content


class TestChunkCodebase:
    """Tests for chunk_codebase helper."""

    def test_chunk_codebase_basic(self):
        """Test chunking an entire codebase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            root = Path(tmpdir)
            (root / "main.py").write_text("def main(): pass")
            (root / "utils.py").write_text("def util(): pass")
            (root / "venv").mkdir()
            (root / "venv" / "ignore.py").write_text("def ignored(): pass")

            chunks = chunk_codebase(root, ignore_patterns=["venv/"])

            # Should have chunks from main.py and utils.py, not venv
            file_paths = {c.file_path for c in chunks}
            assert "main.py" in file_paths
            assert "utils.py" in file_paths
            assert "venv/ignore.py" not in file_paths
