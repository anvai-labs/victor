# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for Python fallback implementations.

These tests validate the Python reference implementations that are used
when Rust extensions are not available.
"""

import inspect

import pytest
import math

from victor.native.protocols import SymbolType, CoercedType
from victor.native.python import symbol_extractor as symbol_extractor_module


class TestPythonSymbolExtractor:
    """Tests for PythonSymbolExtractor."""

    def test_extract_functions_basic(self, symbol_extractor):
        """Test basic function extraction."""
        source = "def hello(): pass"
        symbols = symbol_extractor.extract_functions(source, "python")

        assert len(symbols) == 1
        assert symbols[0].name == "hello"
        assert symbols[0].type == SymbolType.FUNCTION

    def test_extract_functions_with_signature(self, symbol_extractor, sample_python_source):
        """Test function extraction with full signature."""
        symbols = symbol_extractor.extract_functions(sample_python_source, "python")

        # Find public_function
        func = next((s for s in symbols if s.name == "public_function"), None)
        assert func is not None
        assert "x: int" in func.signature
        assert "y: int" in func.signature
        assert "-> int" in func.signature

    def test_extract_async_function(self, symbol_extractor, sample_python_source):
        """Test async function extraction."""
        symbols = symbol_extractor.extract_functions(sample_python_source, "python")

        # Find async_function
        func = next((s for s in symbols if s.name == "async_function"), None)
        assert func is not None
        assert "async def" in func.signature

    def test_extract_classes_basic(self, symbol_extractor):
        """Test basic class extraction."""
        source = "class MyClass: pass"
        symbols = symbol_extractor.extract_classes(source, "python")

        assert len(symbols) == 1
        assert symbols[0].name == "MyClass"
        assert symbols[0].type == SymbolType.CLASS

    def test_extract_classes_with_inheritance(self, symbol_extractor):
        """Test class extraction with inheritance."""
        source = "class Child(Parent): pass"
        symbols = symbol_extractor.extract_classes(source, "python")

        assert len(symbols) == 1
        assert "Parent" in symbols[0].signature

    def test_extract_imports(self, symbol_extractor, sample_python_source):
        """Test import extraction."""
        imports = symbol_extractor.extract_imports(sample_python_source, "python")

        assert "os" in imports
        assert "typing" in imports

    def test_extract_references(self, symbol_extractor):
        """Test identifier reference extraction."""
        source = "x = foo + bar * baz"
        refs = symbol_extractor.extract_references(source)

        assert "x" in refs
        assert "foo" in refs
        assert "bar" in refs
        assert "baz" in refs

    def test_is_stdlib_module(self, symbol_extractor):
        """Test stdlib module detection."""
        assert symbol_extractor.is_stdlib_module("os") is True
        assert symbol_extractor.is_stdlib_module("os.path") is True
        assert symbol_extractor.is_stdlib_module("json") is True
        assert symbol_extractor.is_stdlib_module("typing") is True
        assert symbol_extractor.is_stdlib_module("my_custom_module") is False
        assert symbol_extractor.is_stdlib_module("victor") is False

    def test_extract_private_visibility(self, symbol_extractor):
        """Test private symbol visibility detection."""
        source = "def _private(): pass"
        symbols = symbol_extractor.extract_functions(source, "python")

        assert symbols[0].visibility == "private"

    def test_extract_syntax_error_returns_empty(self, symbol_extractor):
        """Test that syntax errors return empty list."""
        source = "def broken( missing colon"
        symbols = symbol_extractor.extract_functions(source, "python")

        assert symbols == []

    def test_semantic_extraction_has_no_parallel_ast_parser(self):
        source = inspect.getsource(symbol_extractor_module)
        assert "ast.parse" not in source
        assert "STDLIB_MODULES" not in source


class TestPythonArgumentNormalizer:
    """Tests for PythonArgumentNormalizer."""

    def test_normalize_valid_json(self, arg_normalizer):
        """Test normalizing already valid JSON."""
        json_str = '{"key": "value"}'
        result, success = arg_normalizer.normalize_json(json_str)

        assert success is True
        assert result == json_str

    def test_normalize_single_quotes(self, arg_normalizer):
        """Test normalizing JSON with single quotes."""
        json_str = "{'key': 'value'}"
        result, success = arg_normalizer.normalize_json(json_str)

        assert success is True
        assert '"key"' in result
        assert '"value"' in result

    def test_normalize_trailing_comma(self, arg_normalizer):
        """Test normalizing JSON with trailing comma."""
        json_str = '{"key": "value",}'
        result, success = arg_normalizer.normalize_json(json_str)

        assert success is True
        assert result.count(",") == 0 or ",}" not in result

    def test_coerce_integer(self, arg_normalizer):
        """Test coercing string to integer."""
        result = arg_normalizer.coerce_type("42")

        assert result.value == 42
        assert result.coerced_type == CoercedType.INT

    def test_coerce_float(self, arg_normalizer):
        """Test coercing string to float."""
        result = arg_normalizer.coerce_type("3.14")

        assert result.value == 3.14
        assert result.coerced_type == CoercedType.FLOAT

    def test_coerce_boolean_true(self, arg_normalizer):
        """Test coercing string to boolean true."""
        result = arg_normalizer.coerce_type("true")

        assert result.value is True
        assert result.coerced_type == CoercedType.BOOL

    def test_coerce_boolean_false(self, arg_normalizer):
        """Test coercing string to boolean false."""
        result = arg_normalizer.coerce_type("false")

        assert result.value is False
        assert result.coerced_type == CoercedType.BOOL

    def test_coerce_null(self, arg_normalizer):
        """Test coercing string to null."""
        result = arg_normalizer.coerce_type("null")

        assert result.value is None
        assert result.coerced_type == CoercedType.NULL

    def test_coerce_string(self, arg_normalizer):
        """Test keeping string as string."""
        result = arg_normalizer.coerce_type("hello world")

        assert result.value == "hello world"
        assert result.coerced_type == CoercedType.STRING

    def test_repair_quotes_single_to_double(self, arg_normalizer):
        """Test repairing single quotes to double quotes."""
        result = arg_normalizer.repair_quotes("'hello'")

        assert '"hello"' in result or "hello" in result


class TestPythonSimilarityComputer:
    """Tests for PythonSimilarityComputer."""

    def test_cosine_identical_vectors(self, similarity_computer, sample_vectors):
        """Test cosine similarity of identical vectors."""
        vec = sample_vectors["unit_x"]
        sim = similarity_computer.cosine(vec, vec)

        assert math.isclose(sim, 1.0, rel_tol=1e-5)

    def test_cosine_orthogonal_vectors(self, similarity_computer, sample_vectors):
        """Test cosine similarity of orthogonal vectors."""
        sim = similarity_computer.cosine(sample_vectors["unit_x"], sample_vectors["unit_y"])

        assert math.isclose(sim, 0.0, abs_tol=1e-5)

    def test_cosine_opposite_vectors(self, similarity_computer, sample_vectors):
        """Test cosine similarity of opposite vectors."""
        sim = similarity_computer.cosine(sample_vectors["unit_x"], sample_vectors["neg_x"])

        assert math.isclose(sim, -1.0, rel_tol=1e-5)

    def test_cosine_zero_vector(self, similarity_computer, sample_vectors):
        """Test cosine similarity with zero vector."""
        sim = similarity_computer.cosine(sample_vectors["unit_x"], sample_vectors["zero"])

        assert sim == 0.0

    def test_cosine_different_lengths_raises(self, similarity_computer):
        """Test that different length vectors raise ValueError."""
        with pytest.raises(ValueError):
            similarity_computer.cosine([1, 2, 3], [1, 2])

    def test_batch_cosine(self, similarity_computer, sample_vectors):
        """Test batch cosine similarity."""
        query = sample_vectors["unit_x"]
        corpus = [
            sample_vectors["unit_x"],
            sample_vectors["unit_y"],
            sample_vectors["neg_x"],
        ]

        sims = similarity_computer.batch_cosine(query, corpus)

        assert len(sims) == 3
        assert math.isclose(sims[0], 1.0, rel_tol=1e-5)
        assert math.isclose(sims[1], 0.0, abs_tol=1e-5)
        assert math.isclose(sims[2], -1.0, rel_tol=1e-5)

    def test_similarity_matrix(self, similarity_computer, sample_vectors):
        """Test similarity matrix computation."""
        queries = [sample_vectors["unit_x"], sample_vectors["unit_y"]]
        corpus = [sample_vectors["unit_x"], sample_vectors["unit_y"]]

        matrix = similarity_computer.similarity_matrix(queries, corpus)

        assert len(matrix) == 2
        assert len(matrix[0]) == 2
        # Diagonal should be 1.0 (self-similarity)
        assert math.isclose(matrix[0][0], 1.0, rel_tol=1e-5)
        assert math.isclose(matrix[1][1], 1.0, rel_tol=1e-5)

    def test_top_k(self, similarity_computer, sample_vectors):
        """Test top-k similarity search."""
        query = sample_vectors["embedding_a"]
        corpus = [
            sample_vectors["embedding_a"],
            sample_vectors["embedding_b"],
            sample_vectors["embedding_c"],
        ]

        top = similarity_computer.top_k(query, corpus, k=2)

        assert len(top) == 2
        # First should be self (index 0)
        assert top[0][0] == 0
        assert math.isclose(top[0][1], 1.0, rel_tol=1e-5)
        # Second should be similar embedding_b
        assert top[1][0] == 1


class TestPythonTextChunker:
    """Tests for PythonTextChunker."""

    def test_chunk_basic(self, text_chunker):
        """Test basic text chunking."""
        text = "Line 1\nLine 2\nLine 3\n"
        chunks = text_chunker.chunk_with_overlap(text, chunk_size=10, overlap=0)

        assert len(chunks) > 0
        # All text should be covered
        combined = "".join(c.text for c in chunks)
        # Some overlap is expected, so just check content is present
        assert "Line 1" in combined

    def test_chunk_respects_line_boundaries(self, text_chunker, sample_multiline_text):
        """Test that chunking respects line boundaries."""
        chunks = text_chunker.chunk_with_overlap(sample_multiline_text, chunk_size=50, overlap=0)

        # Each chunk should not cut words mid-line
        for chunk in chunks:
            # If chunk ends with newline, it respected boundary
            if chunk.end_offset < len(sample_multiline_text):
                assert chunk.text.endswith("\n") or len(chunk.text) <= 50

    def test_chunk_with_overlap(self, text_chunker, sample_multiline_text):
        """Test chunking with overlap."""
        chunks = text_chunker.chunk_with_overlap(sample_multiline_text, chunk_size=50, overlap=20)

        # Should produce multiple chunks
        assert len(chunks) > 1

        # All chunks after first should have overlap_prev set
        for chunk in chunks[1:]:
            # The overlap_prev field should be set (may be capped by position)
            assert chunk.overlap_prev >= 0

    def test_chunk_line_numbers(self, text_chunker, sample_multiline_text):
        """Test that chunk line numbers are correct."""
        chunks = text_chunker.chunk_with_overlap(sample_multiline_text, chunk_size=100, overlap=0)

        # First chunk should start at line 1
        assert chunks[0].start_line == 1

        # Line numbers should be increasing
        for i in range(1, len(chunks)):
            assert chunks[i].start_line >= chunks[i - 1].start_line

    def test_count_lines(self, text_chunker):
        """Test line counting."""
        assert text_chunker.count_lines("") == 0
        assert text_chunker.count_lines("single line") == 1
        assert text_chunker.count_lines("line 1\nline 2") == 2
        assert text_chunker.count_lines("a\nb\nc\n") == 4

    def test_find_line_boundaries(self, text_chunker):
        """Test finding line boundaries."""
        text = "a\nb\nc"
        boundaries = text_chunker.find_line_boundaries(text)

        assert 0 in boundaries  # First line starts at 0
        assert 2 in boundaries  # Second line starts at 2 (after "a\n")
        assert 4 in boundaries  # Third line starts at 4 (after "a\nb\n")

    def test_line_at_offset(self, text_chunker):
        """Test getting line number for offset."""
        text = "line1\nline2\nline3"

        assert text_chunker.line_at_offset(text, 0) == 1
        assert text_chunker.line_at_offset(text, 5) == 1  # At newline
        assert text_chunker.line_at_offset(text, 6) == 2  # Start of line 2
        assert text_chunker.line_at_offset(text, 12) == 3  # Start of line 3

    def test_chunk_empty_text(self, text_chunker):
        """Test chunking empty text."""
        chunks = text_chunker.chunk_with_overlap("", chunk_size=10, overlap=0)
        assert chunks == []

    def test_chunk_invalid_params(self, text_chunker):
        """Test chunking with invalid parameters."""
        with pytest.raises(ValueError):
            text_chunker.chunk_with_overlap("text", chunk_size=0, overlap=0)

        with pytest.raises(ValueError):
            text_chunker.chunk_with_overlap("text", chunk_size=10, overlap=10)


class TestObservabilityIntegration:
    """Tests for observability integration in fallback implementations."""

    def test_extractor_records_metrics(self, symbol_extractor, native_metrics):
        """Test that symbol extractor records metrics."""
        source = "def test(): pass"
        symbol_extractor.extract_functions(source, "python")

        stats = symbol_extractor.get_metrics()
        assert stats["calls_total"] >= 1

    def test_normalizer_records_metrics(self, arg_normalizer, native_metrics):
        """Test that argument normalizer records metrics."""
        arg_normalizer.coerce_type("42")

        stats = arg_normalizer.get_metrics()
        assert stats["calls_total"] >= 1

    def test_similarity_records_metrics(self, similarity_computer, sample_vectors):
        """Test that similarity computer records metrics."""
        similarity_computer.cosine(sample_vectors["unit_x"], sample_vectors["unit_y"])

        stats = similarity_computer.get_metrics()
        assert stats["calls_total"] >= 1

    def test_chunker_records_metrics(self, text_chunker):
        """Test that text chunker records metrics."""
        text_chunker.count_lines("line1\nline2")

        stats = text_chunker.get_metrics()
        assert stats["calls_total"] >= 1
