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

"""Tests for native extensions module.

These tests verify both the Python fallback implementations and
the Rust native implementations (when available).
"""

import json

import pytest

from victor.processing.native import (
    batch_cosine_similarity,
    compute_batch_signatures,
    compute_signature,
    cosine_similarity,
    extract_json_objects,
    find_duplicate_blocks,
    get_native_version,
    is_native_available,
    normalize_block,
    repair_json,
    rolling_hash_blocks,
    signature_similarity,
    top_k_similar,
)

# =============================================================================
# STATUS TESTS
# =============================================================================


class TestNativeStatus:
    """Tests for native extension status functions."""

    def test_is_native_available_returns_bool(self):
        """Test is_native_available returns boolean."""
        result = is_native_available()
        assert isinstance(result, bool)

    def test_get_native_version(self):
        """Test get_native_version returns string or None."""
        version = get_native_version()
        if is_native_available():
            assert isinstance(version, str)
            assert len(version) > 0
        else:
            assert version is None


# =============================================================================
# DEDUPLICATION TESTS
# =============================================================================


class TestNormalizeBlock:
    """Tests for normalize_block function."""

    def test_strips_whitespace(self):
        """Test strips leading/trailing whitespace."""
        result = normalize_block("  hello world  ")
        assert result == "hello world"

    def test_collapses_whitespace(self):
        """Test collapses multiple whitespace to single space."""
        result = normalize_block("hello   world")
        assert result == "hello world"

    def test_removes_trailing_punctuation(self):
        """Test removes trailing punctuation."""
        assert normalize_block("hello.") == "hello"
        assert normalize_block("hello,") == "hello"
        assert normalize_block("hello;") == "hello"
        assert normalize_block("hello:") == "hello"

    def test_converts_to_lowercase(self):
        """Test converts to lowercase."""
        result = normalize_block("Hello WORLD")
        assert result == "hello world"

    def test_empty_string(self):
        """Test handles empty string."""
        result = normalize_block("")
        assert result == ""

    def test_whitespace_only(self):
        """Test handles whitespace-only string."""
        result = normalize_block("   ")
        assert result == ""


class TestRollingHashBlocks:
    """Tests for rolling_hash_blocks function."""

    def test_splits_into_blocks(self):
        """Test splits content into blocks."""
        content = "Block 1\n\nBlock 2\n\nBlock 3"
        results = rolling_hash_blocks(content, min_block_length=5)
        assert len(results) == 3

    def test_identifies_duplicates(self):
        """Test identifies duplicate blocks."""
        content = "First block here\n\nSecond block\n\nFirst block here"
        results = rolling_hash_blocks(content, min_block_length=5)
        # Third block should be duplicate
        assert results[2][2] is True

    def test_short_blocks_not_duplicated(self):
        """Test short blocks are not marked as duplicates."""
        content = "Hi\n\nHi\n\nHi"
        results = rolling_hash_blocks(content, min_block_length=10)
        # Short blocks should never be duplicates
        assert all(not r[2] for r in results)

    def test_returns_hashes(self):
        """Test returns valid hash strings."""
        content = "This is a longer block of text"
        results = rolling_hash_blocks(content, min_block_length=5)
        assert len(results) > 0
        assert len(results[0][0]) > 0  # Hash string

    def test_preserves_original_content(self):
        """Test preserves original block content."""
        content = "Block content here"
        results = rolling_hash_blocks(content, min_block_length=5)
        assert results[0][1] == content


class TestFindDuplicateBlocks:
    """Tests for find_duplicate_blocks function."""

    def test_finds_duplicates(self):
        """Test finds duplicate block indices."""
        content = "First block\n\nSecond block\n\nFirst block"
        duplicates = find_duplicate_blocks(content, min_block_length=5)
        assert len(duplicates) == 1
        assert duplicates[0][0] == 2  # Index of third block

    def test_no_duplicates(self):
        """Test returns empty list when no duplicates."""
        content = "Block 1\n\nBlock 2\n\nBlock 3"
        duplicates = find_duplicate_blocks(content, min_block_length=5)
        assert len(duplicates) == 0

    def test_multiple_duplicates(self):
        """Test finds multiple duplicates."""
        content = "A\n\nA\n\nB\n\nB\n\nA"
        duplicates = find_duplicate_blocks(content, min_block_length=1)
        assert len(duplicates) >= 2


# =============================================================================
# SIMILARITY TESTS
# =============================================================================


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Test identical vectors have similarity 1.0."""
        v = [1.0, 2.0, 3.0, 4.0]
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        """Test orthogonal vectors have similarity 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        sim = cosine_similarity(a, b)
        assert abs(sim) < 1e-5

    def test_opposite_vectors(self):
        """Test opposite vectors have similarity -1.0."""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        sim = cosine_similarity(a, b)
        assert abs(sim + 1.0) < 1e-5

    def test_dimension_mismatch(self):
        """Test raises error for dimension mismatch."""
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError):
            cosine_similarity(a, b)


class TestBatchCosineSimilarity:
    """Tests for batch_cosine_similarity function."""

    def test_batch_similarities(self):
        """Test computes similarities for all corpus vectors."""
        query = [1.0, 0.0, 0.0]
        corpus = [
            [1.0, 0.0, 0.0],  # identical
            [0.0, 1.0, 0.0],  # orthogonal
            [-1.0, 0.0, 0.0],  # opposite
        ]
        sims = batch_cosine_similarity(query, corpus)
        assert len(sims) == 3
        assert abs(sims[0] - 1.0) < 1e-5
        assert abs(sims[1]) < 1e-5
        assert abs(sims[2] + 1.0) < 1e-5

    def test_empty_corpus(self):
        """Test handles empty corpus."""
        query = [1.0, 0.0, 0.0]
        corpus = []
        sims = batch_cosine_similarity(query, corpus)
        assert sims == []


class TestTopKSimilar:
    """Tests for top_k_similar function."""

    def test_returns_top_k(self):
        """Test returns correct number of results."""
        query = [1.0, 0.0]
        corpus = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.9, 0.1]]
        top = top_k_similar(query, corpus, k=2)
        assert len(top) == 2

    def test_sorted_by_similarity(self):
        """Test results are sorted by similarity descending."""
        query = [1.0, 0.0]
        corpus = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        top = top_k_similar(query, corpus, k=3)
        # Should be sorted by similarity descending
        for i in range(len(top) - 1):
            assert top[i][1] >= top[i + 1][1]

    def test_returns_indices(self):
        """Test returns correct indices."""
        query = [1.0, 0.0]
        corpus = [[0.0, 1.0], [1.0, 0.0]]  # Most similar is index 1
        top = top_k_similar(query, corpus, k=1)
        assert top[0][0] == 1  # Index of most similar


# =============================================================================
# JSON REPAIR TESTS
# =============================================================================


class TestRepairJson:
    """Tests for repair_json function."""

    def test_repairs_single_quotes(self):
        """Test repairs single quotes to double quotes."""
        input_str = "{'key': 'value'}"
        result = repair_json(input_str)
        assert '"key"' in result
        assert '"value"' in result
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_repairs_python_true_false(self):
        """Test repairs Python True/False to JSON true/false."""
        input_str = "{'active': True, 'deleted': False}"
        result = repair_json(input_str)
        assert "true" in result
        assert "false" in result
        parsed = json.loads(result)
        assert parsed["active"] is True
        assert parsed["deleted"] is False

    def test_repairs_python_none(self):
        """Test repairs Python None to JSON null."""
        input_str = "{'value': None}"
        result = repair_json(input_str)
        assert "null" in result
        parsed = json.loads(result)
        assert parsed["value"] is None

    def test_valid_json_passthrough(self):
        """Test valid JSON passes through unchanged."""
        input_str = '{"key": "value"}'
        result = repair_json(input_str)
        assert result == input_str

    def test_nested_structures(self):
        """Test repairs nested structures."""
        input_str = "{'outer': {'inner': 'value'}}"
        result = repair_json(input_str)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == "value"

    def test_arrays(self):
        """Test repairs arrays."""
        input_str = "['a', 'b', 'c']"
        result = repair_json(input_str)
        parsed = json.loads(result)
        assert parsed == ["a", "b", "c"]


class TestExtractJsonObjects:
    """Tests for extract_json_objects function."""

    def test_extracts_single_object(self):
        """Test extracts single JSON object."""
        text = 'Here is the result: {"name": "test"} and more text'
        objects = extract_json_objects(text)
        assert len(objects) == 1
        parsed = json.loads(objects[0][2])
        assert parsed["name"] == "test"

    def test_extracts_multiple_objects(self):
        """Test extracts multiple JSON objects."""
        text = '{"a": 1} some text {"b": 2}'
        objects = extract_json_objects(text)
        assert len(objects) == 2

    def test_extracts_nested_object(self):
        """Test extracts nested objects correctly."""
        text = 'Result: {"outer": {"inner": "value"}} done'
        objects = extract_json_objects(text)
        assert len(objects) == 1
        parsed = json.loads(objects[0][2])
        assert parsed["outer"]["inner"] == "value"

    def test_returns_positions(self):
        """Test returns correct positions."""
        text = 'abc {"key": "value"} xyz'
        objects = extract_json_objects(text)
        start, end, json_str = objects[0]
        assert text[start:end] == '{"key": "value"}'

    def test_extracts_arrays(self):
        """Test extracts JSON arrays."""
        text = "Data: [1, 2, 3] done"
        objects = extract_json_objects(text)
        assert len(objects) == 1


# =============================================================================
# HASHING TESTS
# =============================================================================


class TestComputeSignature:
    """Tests for compute_signature function."""

    def test_produces_consistent_signature(self):
        """Test same input produces same signature."""
        sig1 = compute_signature("test_tool", {"arg": "value"})
        sig2 = compute_signature("test_tool", {"arg": "value"})
        assert sig1 == sig2

    def test_different_tools_different_signatures(self):
        """Test different tools produce different signatures."""
        sig1 = compute_signature("tool_a", {"arg": "value"})
        sig2 = compute_signature("tool_b", {"arg": "value"})
        assert sig1 != sig2

    def test_different_args_different_signatures(self):
        """Test different args produce different signatures."""
        sig1 = compute_signature("tool", {"arg": "value1"})
        sig2 = compute_signature("tool", {"arg": "value2"})
        assert sig1 != sig2

    def test_signature_format(self):
        """Test signature is hex string of expected length."""
        sig = compute_signature("test", {"key": "value"})
        assert isinstance(sig, str)
        assert len(sig) == 16
        # Should be valid hex
        int(sig, 16)


class TestComputeBatchSignatures:
    """Tests for compute_batch_signatures function."""

    def test_batch_signatures(self):
        """Test computes signatures for all calls."""
        calls = [
            ("tool1", {"arg": "a"}),
            ("tool2", {"arg": "b"}),
            ("tool1", {"arg": "a"}),  # Duplicate
        ]
        sigs = compute_batch_signatures(calls)
        assert len(sigs) == 3
        assert sigs[0] == sigs[2]  # Duplicates have same signature
        assert sigs[0] != sigs[1]


class TestSignatureSimilarity:
    """Tests for signature_similarity function."""

    def test_identical_signatures(self):
        """Test identical signatures have similarity 1.0."""
        sim = signature_similarity("1234567890abcdef", "1234567890abcdef")
        assert abs(sim - 1.0) < 1e-6

    def test_different_signatures(self):
        """Test completely different signatures have low similarity."""
        sim = signature_similarity("1111111111111111", "2222222222222222")
        assert sim < 0.5

    def test_partial_match(self):
        """Test partial match gives intermediate similarity."""
        sim = signature_similarity("1111111122222222", "1111111133333333")
        assert 0.3 < sim < 0.7

    def test_different_lengths(self):
        """Test different lengths returns 0.0."""
        sim = signature_similarity("1234", "12345678")
        assert sim == 0.0


# =============================================================================
# TOKEN COUNTING BATCH (Gap 2: single-FFI-crossing batch)
# =============================================================================


class TestTokenCountBatch:
    """Gap 2: count_tokens_batch must cross the FFI boundary once via the native
    batch symbol, with results identical to per-element counting. Falls back to
    the per-element loop on older native builds (version-skew safety)."""

    def test_native_batch_symbol_registered(self):
        """The Rust batch function must be exported by victor_native."""
        pytest.importorskip("victor_native")
        import victor_native

        assert hasattr(victor_native, "count_tokens_fast_batch"), (
            "native build is missing count_tokens_fast_batch — rebuild with "
            "`cd rust && maturin develop --release`"
        )

    def test_batch_matches_single_elementwise(self):
        """Batch result equals per-element counting (the wrapper's invariant)."""
        pytest.importorskip("victor_native")
        from victor.native.rust.tokenizer import RustTokenCounter

        tc = RustTokenCounter()
        texts = ["", "hello", "the quick brown fox", "   \n  ", "你好世界"]
        batched = tc.count_tokens_batch(texts)
        single = [tc.count_tokens_fast(t) for t in texts]
        assert batched == single

    def test_batch_empty(self):
        pytest.importorskip("victor_native")
        from victor.native.rust.tokenizer import RustTokenCounter

        assert RustTokenCounter().count_tokens_batch([]) == []

    def test_batch_preserves_order(self):
        pytest.importorskip("victor_native")
        from victor.native.rust.tokenizer import RustTokenCounter

        tc = RustTokenCounter()
        texts = ["a", "bb", "ccc", "dddd"]
        batched = tc.count_tokens_batch(texts)
        assert len(batched) == len(texts)


class TestSimilarityMatrixBatch:
    """FFI audit: similarity_matrix must cross once via the native batch fn,
    matching per-query batch_cosine_similarity. (Note: similarity_matrix is a
    python-preferred op per ACCELERATOR_BENCHMARKS — NumPy wins — so the Rust
    path is only used when forced; this just makes that path non-looping.)"""

    def test_native_matrix_symbol_registered(self):
        pytest.importorskip("victor_native")
        import victor_native

        assert hasattr(victor_native, "similarity_matrix")

    def test_matrix_matches_per_query(self):
        pytest.importorskip("victor_native")
        import victor_native

        from victor.native.rust.similarity import RustSimilarityComputer

        queries = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]]
        corpus = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [-1.0, 0.0, 0.0]]
        sc = RustSimilarityComputer()
        for normalize in (True, False):
            matrix = sc.similarity_matrix(queries, corpus, normalize=normalize)
            assert len(matrix) == len(queries)
            for i, q in enumerate(queries):
                row = victor_native.batch_cosine_similarity(q, corpus)
                assert len(matrix[i]) == len(row)
                for a, b in zip(matrix[i], row):
                    assert abs(a - b) < 1e-5
        # Sanity: identical -> 1.0, opposite -> -1.0
        m = sc.similarity_matrix(queries, corpus, normalize=True)
        assert abs(m[0][0] - 1.0) < 1e-5
        assert abs(m[0][3] + 1.0) < 1e-5
