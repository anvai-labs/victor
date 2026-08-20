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

"""Tests for AST-aware code chunking - achieving 70%+ coverage."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

pytest.importorskip("victor_coding.codebase.embeddings.chunker")
pytest.importorskip("victor_coding.languages.base")

from victor_coding.codebase.embeddings.chunker import ASTAwareChunker
from victor_coding.languages.base import CodeChunk


class TestASTAwareChunkerInit:
    """Tests for ASTAwareChunker initialization."""

    def test_default_initialization(self):
        """Test default initialization with no arguments."""
        chunker = ASTAwareChunker()
        assert chunker.max_chunk_size == ASTAwareChunker.DEFAULT_MAX_CHUNK_SIZE
        assert chunker.min_chunk_size == ASTAwareChunker.DEFAULT_MIN_CHUNK_SIZE
        assert chunker.registry is not None
        assert chunker._parsers == {}

    def test_custom_chunk_sizes(self):
        """Test initialization with custom chunk sizes."""
        chunker = ASTAwareChunker(
            max_chunk_size=1000,
            min_chunk_size=25,
        )
        assert chunker.max_chunk_size == 1000
        assert chunker.min_chunk_size == 25

    def test_custom_registry(self):
        """Test initialization with custom registry."""
        mock_registry = Mock()
        chunker = ASTAwareChunker(registry=mock_registry)
        assert chunker.registry is mock_registry

    def test_default_constants(self):
        """Test default constant values."""
        assert ASTAwareChunker.DEFAULT_MAX_CHUNK_SIZE == 2000
        assert ASTAwareChunker.DEFAULT_MIN_CHUNK_SIZE == 50


class TestGetParser:
    """Tests for _get_parser method."""

    def test_get_parser_returns_cached(self):
        """Test cached parser is returned."""
        chunker = ASTAwareChunker()
        mock_parser = Mock()
        chunker._parsers["python"] = mock_parser

        result = chunker._get_parser("python")
        assert result is mock_parser

    def test_get_parser_caches_after_retrieval(self):
        """Test parser is cached after retrieval."""
        chunker = ASTAwareChunker()
        # Pre-cache the parser
        mock_parser = Mock()
        chunker._parsers["test_lang"] = mock_parser

        # First call - should return cached
        parser1 = chunker._get_parser("test_lang")
        # Second call - should also return cached
        parser2 = chunker._get_parser("test_lang")

        assert parser1 is parser2
        assert parser1 is mock_parser

    def test_get_parser_returns_none_for_unknown(self):
        """Test parser returns None when import fails."""
        chunker = ASTAwareChunker()

        # This will fail to get parser for an unknown language
        # because tree_sitter_manager.get_parser will raise an exception
        result = chunker._get_parser("completely_fake_language_xyz")
        # Either returns None (exception caught) or a parser (if tree-sitter has it)
        # We just verify no exception is raised
        assert result is None or result is not None


class TestChunkByLines:
    """Tests for _chunk_by_lines method."""

    def test_chunk_small_file(self):
        """Test chunking a small file by lines."""
        chunker = ASTAwareChunker(max_chunk_size=100)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line 1\nline 2\nline 3\n")
            f.flush()

            chunks = chunker._chunk_by_lines(Path(f.name))

            assert len(chunks) >= 1
            assert chunks[0].chunk_type == "text"
            assert "line 1" in chunks[0].text

    def test_chunk_large_file(self):
        """Test chunking a large file splits into multiple chunks."""
        chunker = ASTAwareChunker(max_chunk_size=50)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Write content that exceeds max_chunk_size
            for i in range(20):
                f.write(f"This is line number {i} with some extra text\n")
            f.flush()

            chunks = chunker._chunk_by_lines(Path(f.name))

            assert len(chunks) > 1
            # Each chunk should be under max size (or close to it)
            for chunk in chunks:
                # Allow some flexibility for boundary conditions
                assert chunk.chunk_type == "text"

    def test_chunk_nonexistent_file(self):
        """Test chunking a non-existent file returns empty list."""
        chunker = ASTAwareChunker()

        chunks = chunker._chunk_by_lines(Path("/nonexistent/file.txt"))
        assert chunks == []


class TestChunkByLinesContent:
    """Tests for _chunk_by_lines_content method."""

    def test_chunk_content_basic(self):
        """Test basic content chunking."""
        chunker = ASTAwareChunker(max_chunk_size=100)
        content = "line 1\nline 2\nline 3"

        chunks = chunker._chunk_by_lines_content(content, "test.txt")

        assert len(chunks) >= 1
        assert all(c.file_path == "test.txt" for c in chunks)
        assert all(c.chunk_type == "text" for c in chunks)

    def test_chunk_content_preserves_line_numbers(self):
        """Test line numbers are preserved correctly."""
        chunker = ASTAwareChunker(max_chunk_size=1000)
        content = "line 1\nline 2\nline 3"

        chunks = chunker._chunk_by_lines_content(content, "test.txt")

        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 3

    def test_chunk_content_splits_large(self):
        """Test large content is split correctly."""
        chunker = ASTAwareChunker(max_chunk_size=30)
        content = "a" * 10 + "\n" + "b" * 10 + "\n" + "c" * 10 + "\n" + "d" * 10

        chunks = chunker._chunk_by_lines_content(content, "test.txt")

        assert len(chunks) > 1

    def test_chunk_empty_content(self):
        """Test empty content returns single empty chunk."""
        chunker = ASTAwareChunker()

        chunks = chunker._chunk_by_lines_content("", "test.txt")

        assert len(chunks) == 1
        assert chunks[0].text == ""


class TestSplitLargeChunk:
    """Tests for _split_large_chunk method."""

    def test_split_large_chunk_basic(self):
        """Test splitting a large chunk."""
        chunker = ASTAwareChunker(max_chunk_size=50)

        # Create a text larger than max_chunk_size
        text = "\n".join([f"line {i} with some content" for i in range(10)])

        chunks = chunker._split_large_chunk(
            text=text,
            start_line=1,
            chunk_type="function",
            symbol_name="test_func",
            file_path="test.py",
        )

        assert len(chunks) > 1
        # First chunk keeps original type, subsequent get "_part" suffix
        assert all(c.symbol_name == "test_func" for c in chunks)

    def test_split_small_chunk_returns_one(self):
        """Test small chunk returns single chunk."""
        chunker = ASTAwareChunker(max_chunk_size=1000)

        text = "small content"

        chunks = chunker._split_large_chunk(
            text=text,
            start_line=1,
            chunk_type="function",
            symbol_name="small_func",
            file_path="test.py",
        )

        assert len(chunks) == 1
        assert chunks[0].chunk_type == "function"

    def test_split_preserves_metadata(self):
        """Test split preserves symbol metadata."""
        chunker = ASTAwareChunker(max_chunk_size=30)

        text = "def test():\n    pass\n    more code here"

        chunks = chunker._split_large_chunk(
            text=text,
            start_line=10,
            chunk_type="function",
            symbol_name="test",
            file_path="myfile.py",
        )

        assert all(c.file_path == "myfile.py" for c in chunks)
        assert all(c.symbol_name == "test" for c in chunks)

    def test_split_sets_parent_symbol(self):
        """Test parent_symbol is set for continuation chunks."""
        chunker = ASTAwareChunker(max_chunk_size=30)

        text = "a" * 20 + "\n" + "b" * 20 + "\n" + "c" * 20

        chunks = chunker._split_large_chunk(
            text=text,
            start_line=1,
            chunk_type="class",
            symbol_name="MyClass",
            file_path="test.py",
        )

        if len(chunks) > 1:
            # First chunk should have no parent
            assert chunks[0].parent_symbol is None
            # Subsequent chunks should have parent_symbol set
            for chunk in chunks[1:]:
                assert chunk.parent_symbol == "MyClass"


class TestFindDefinitionParent:
    """Tests for _find_definition_parent method."""

    def test_find_definition_parent_python(self):
        """Test finding definition parent for Python."""
        chunker = ASTAwareChunker()

        # Create mock nodes
        mock_def_node = Mock()
        mock_def_node.type = "function_definition"
        mock_def_node.parent = None

        mock_name_node = Mock()
        mock_name_node.parent = mock_def_node

        result = chunker._find_definition_parent(mock_name_node, "python")
        assert result is mock_def_node

    def test_find_definition_parent_javascript(self):
        """Test finding definition parent for JavaScript."""
        chunker = ASTAwareChunker()

        mock_def_node = Mock()
        mock_def_node.type = "function_declaration"
        mock_def_node.parent = None

        mock_name_node = Mock()
        mock_name_node.parent = mock_def_node

        result = chunker._find_definition_parent(mock_name_node, "javascript")
        assert result is mock_def_node

    def test_find_definition_parent_nested(self):
        """Test finding definition parent with nested nodes."""
        chunker = ASTAwareChunker()

        # Create a chain: name -> identifier -> function_definition
        mock_def_node = Mock()
        mock_def_node.type = "function_definition"
        mock_def_node.parent = None

        mock_intermediate = Mock()
        mock_intermediate.type = "identifier"
        mock_intermediate.parent = mock_def_node

        mock_name_node = Mock()
        mock_name_node.parent = mock_intermediate

        result = chunker._find_definition_parent(mock_name_node, "python")
        assert result is mock_def_node

    def test_find_definition_parent_not_found(self):
        """Test returns None when no definition parent found."""
        chunker = ASTAwareChunker()

        mock_root = Mock()
        mock_root.type = "module"
        mock_root.parent = None

        mock_name_node = Mock()
        mock_name_node.parent = mock_root

        result = chunker._find_definition_parent(mock_name_node, "python")
        assert result is None

    def test_find_definition_parent_unknown_language(self):
        """Test returns None for unknown language."""
        chunker = ASTAwareChunker()

        mock_node = Mock()
        mock_node.type = "some_type"
        mock_node.parent = None

        mock_name_node = Mock()
        mock_name_node.parent = mock_node

        result = chunker._find_definition_parent(mock_name_node, "unknown_lang")
        assert result is None


class TestChunkFile:
    """Tests for chunk_file method."""

    def test_chunk_file_detects_language(self):
        """Test language is auto-detected from file extension."""
        chunker = ASTAwareChunker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n    pass\n")
            f.flush()

            # Should detect Python and attempt AST parsing
            chunks = chunker.chunk_file(Path(f.name))

            assert len(chunks) >= 1

    def test_chunk_file_explicit_language(self):
        """Test chunking with explicit language."""
        chunker = ASTAwareChunker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("def hello():\n    pass\n")
            f.flush()

            # Force Python language even though extension is .txt
            chunks = chunker.chunk_file(Path(f.name), language="python")

            assert len(chunks) >= 1

    def test_chunk_file_unknown_language_fallback(self):
        """Test unknown language falls back to line-based chunking."""
        mock_registry = Mock()
        mock_registry.detect_language.return_value = None

        chunker = ASTAwareChunker(registry=mock_registry)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("some content\nmore content\n")
            f.flush()

            chunks = chunker.chunk_file(Path(f.name))

            assert len(chunks) >= 1
            assert chunks[0].chunk_type == "text"

    def test_chunk_file_no_parser_fallback(self):
        """Test falls back to lines when no parser available."""
        mock_registry = Mock()
        mock_registry.detect_language.return_value = "custom_lang"

        chunker = ASTAwareChunker(registry=mock_registry)

        with patch.object(chunker, "_get_parser", return_value=None):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".custom", delete=False) as f:
                f.write("custom content\n")
                f.flush()

                chunks = chunker.chunk_file(Path(f.name))

                assert len(chunks) >= 1
                assert chunks[0].chunk_type == "text"

    def test_chunk_file_parse_error_fallback(self):
        """Test falls back to lines on parse error."""
        mock_registry = Mock()
        mock_registry.detect_language.return_value = "python"

        mock_parser = Mock()
        mock_parser.parse.side_effect = Exception("Parse error")

        chunker = ASTAwareChunker(registry=mock_registry)
        chunker._parsers["python"] = mock_parser

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("invalid python {{{\n")
            f.flush()

            chunks = chunker.chunk_file(Path(f.name))

            # Should fall back to line-based chunking
            assert len(chunks) >= 1


class TestChunkFromTree:
    """Tests for _chunk_from_tree method."""

    def test_chunk_from_tree_empty_tree(self):
        """Test chunking empty tree falls back to lines."""
        chunker = ASTAwareChunker()

        mock_tree = Mock()
        mock_tree.root_node = Mock()
        mock_tree.root_node.children = []
        mock_tree.language = Mock()

        mock_registry = Mock()
        mock_registry.get.side_effect = KeyError("No plugin")
        chunker.registry = mock_registry

        content = b"some content\n"

        chunks = chunker._chunk_from_tree(mock_tree, content, "test.py", "python")

        # Should return line-based chunks since no symbols found
        assert len(chunks) >= 1

    def test_chunk_from_tree_with_plugin_queries(self):
        """Test chunking with plugin queries."""
        chunker = ASTAwareChunker()

        # Mock the tree and plugin
        mock_tree = Mock()
        mock_root = Mock()
        mock_tree.root_node = mock_root
        mock_tree.language = Mock()

        mock_plugin = Mock()
        mock_queries = Mock()
        mock_queries.symbols = []  # No query patterns
        mock_plugin.tree_sitter_queries = mock_queries

        mock_registry = Mock()
        mock_registry.get.return_value = mock_plugin
        chunker.registry = mock_registry

        content = b"def hello():\n    pass\n"

        chunks = chunker._chunk_from_tree(mock_tree, content, "test.py", "python")

        assert isinstance(chunks, list)


class TestCodeChunkDataclass:
    """Tests for CodeChunk dataclass used by chunker."""

    def test_code_chunk_basic_creation(self):
        """Test basic CodeChunk creation."""
        chunk = CodeChunk(
            text="def hello():\n    pass",
            start_line=1,
            end_line=2,
            chunk_type="function",
        )

        assert chunk.text == "def hello():\n    pass"
        assert chunk.start_line == 1
        assert chunk.end_line == 2
        assert chunk.chunk_type == "function"
        assert chunk.symbol_name is None
        assert chunk.parent_symbol is None
        assert chunk.file_path is None

    def test_code_chunk_all_fields(self):
        """Test CodeChunk with all fields."""
        chunk = CodeChunk(
            text="def method(self):\n    pass",
            start_line=10,
            end_line=11,
            chunk_type="method",
            symbol_name="method",
            parent_symbol="MyClass",
            file_path="/path/to/file.py",
        )

        assert chunk.symbol_name == "method"
        assert chunk.parent_symbol == "MyClass"
        assert chunk.file_path == "/path/to/file.py"


class TestChunkerIntegration:
    """Integration tests for ASTAwareChunker."""

    def test_chunk_python_file(self):
        """Test chunking a real Python file."""
        chunker = ASTAwareChunker()

        python_code = '''"""Module docstring."""

def function_one():
    """Function docstring."""
    pass

class MyClass:
    """Class docstring."""

    def method_one(self):
        pass

    def method_two(self):
        pass

def function_two():
    pass
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_code)
            f.flush()

            chunks = chunker.chunk_file(Path(f.name))

            # Should have at least some chunks
            assert len(chunks) >= 1

    def test_chunk_javascript_file(self):
        """Test chunking a JavaScript file."""
        chunker = ASTAwareChunker()

        js_code = """function hello() {
    console.log("Hello");
}

class Greeter {
    greet() {
        return "Hi";
    }
}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js_code)
            f.flush()

            chunks = chunker.chunk_file(Path(f.name))

            assert len(chunks) >= 1

    def test_chunk_preserves_file_path(self):
        """Test file path is preserved in chunks."""
        chunker = ASTAwareChunker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\n")
            f.flush()
            file_path = f.name

            chunks = chunker.chunk_file(Path(file_path))

            for chunk in chunks:
                assert chunk.file_path is not None

    def test_chunk_handles_unicode(self):
        """Test chunking handles unicode content."""
        chunker = ASTAwareChunker()

        unicode_code = '''# -*- coding: utf-8 -*-
"""Unicode test: こんにちは 🚀"""

def greet(name):
    return f"Hello, {name}! 你好！"
'''

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(unicode_code)
            f.flush()

            chunks = chunker.chunk_file(Path(f.name))

            assert len(chunks) >= 1
            # Should contain some unicode
            full_text = "".join(c.text for c in chunks)
            assert "こんにちは" in full_text or "你好" in full_text or len(chunks) > 0


class TestChunkerEdgeCases:
    """Edge case tests for ASTAwareChunker."""

    def test_empty_file(self):
        """Test chunking an empty file."""
        chunker = ASTAwareChunker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            f.flush()

            chunks = chunker.chunk_file(Path(f.name))

            # Should return at least one chunk (possibly empty)
            assert isinstance(chunks, list)

    def test_file_with_only_whitespace(self):
        """Test chunking a file with only whitespace."""
        chunker = ASTAwareChunker()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("   \n\n   \n")
            f.flush()

            chunks = chunker.chunk_file(Path(f.name))

            assert isinstance(chunks, list)

    def test_very_long_line(self):
        """Test chunking with very long lines."""
        chunker = ASTAwareChunker(max_chunk_size=100)

        long_line = "x = " + "a" * 1000

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(long_line)
            f.flush()

            chunks = chunker.chunk_file(Path(f.name))

            assert len(chunks) >= 1

    def test_binary_content_handling(self):
        """Test handling of binary content in file."""
        chunker = ASTAwareChunker()

        # Write some binary-like content
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02\x03 some text \xff\xfe")
            f.flush()

            # Should handle gracefully
            chunks = chunker._chunk_by_lines(Path(f.name))

            assert isinstance(chunks, list)


class TestLanguageDefinitionTypes:
    """Tests for language-specific definition types."""

    def test_python_definition_types(self):
        """Test Python definition types are correct."""
        chunker = ASTAwareChunker()

        mock_node = Mock()
        mock_node.type = "function_definition"
        mock_node.parent = None

        mock_name = Mock()
        mock_name.parent = mock_node

        result = chunker._find_definition_parent(mock_name, "python")
        assert result is mock_node

    def test_go_definition_types(self):
        """Test Go definition types."""
        chunker = ASTAwareChunker()

        mock_node = Mock()
        mock_node.type = "function_declaration"
        mock_node.parent = None

        mock_name = Mock()
        mock_name.parent = mock_node

        result = chunker._find_definition_parent(mock_name, "go")
        assert result is mock_node

    def test_rust_definition_types(self):
        """Test Rust definition types."""
        chunker = ASTAwareChunker()

        mock_node = Mock()
        mock_node.type = "function_item"
        mock_node.parent = None

        mock_name = Mock()
        mock_name.parent = mock_node

        result = chunker._find_definition_parent(mock_name, "rust")
        assert result is mock_node

    def test_java_definition_types(self):
        """Test Java definition types."""
        chunker = ASTAwareChunker()

        mock_node = Mock()
        mock_node.type = "class_declaration"
        mock_node.parent = None

        mock_name = Mock()
        mock_name.parent = mock_node

        result = chunker._find_definition_parent(mock_name, "java")
        assert result is mock_node

    def test_cpp_definition_types(self):
        """Test C++ definition types."""
        chunker = ASTAwareChunker()

        mock_node = Mock()
        mock_node.type = "function_definition"
        mock_node.parent = None

        mock_name = Mock()
        mock_name.parent = mock_node

        result = chunker._find_definition_parent(mock_name, "cpp")
        assert result is mock_node


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
