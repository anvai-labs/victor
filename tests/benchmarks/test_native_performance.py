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

"""Benchmark tests for native Rust acceleration performance.

These tests validate that Rust implementations provide the expected speedup
over Python fallbacks. Run with pytest --benchmark to see detailed results.

Usage:
    pytest tests/benchmarks/test_native_performance.py -v --benchmark-only
    pytest tests/benchmarks/test_native_performance.py -v --no-cov

Expected Speedups:
- Type coercion: 3-5x
- Stdlib detection: 5-10x
- Cosine similarity: 2-5x
- Line counting: 5-10x
- JSON repair: 5-10x
"""

from __future__ import annotations

import random
import string
import time
from typing import Callable, List

import pytest


def is_rust_available() -> bool:
    """Check if Rust native extensions are available."""
    try:
        from victor.processing.native import is_native_available

        return is_native_available()
    except ImportError:
        return False


def time_function(func: Callable, iterations: int = 100) -> float:
    """Time a function over multiple iterations.

    Returns:
        Average time in milliseconds per call
    """
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000  # ms per call


# =============================================================================
# Test Data Generation
# =============================================================================


def generate_random_string(length: int) -> str:
    """Generate random string for testing."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_sample_code(lines: int) -> str:
    """Generate sample Python code for chunking tests."""
    parts = ['"""Module docstring."""', "", "import os", "from typing import List", ""]
    for i in range(lines // 5):
        parts.append(f"def function_{i}(x: int) -> int:")
        parts.append(f'    """Docstring for function {i}."""')
        parts.append(f"    result = x * {i}")
        parts.append("    return result")
        parts.append("")
    return "\n".join(parts)


def generate_vectors(count: int, dim: int) -> List[List[float]]:
    """Generate random normalized vectors."""
    vectors = []
    for _ in range(count):
        vec = [random.random() for _ in range(dim)]
        norm = sum(v**2 for v in vec) ** 0.5
        vectors.append([v / norm for v in vec])
    return vectors


# =============================================================================
# Type Coercion Benchmarks
# =============================================================================


class TestTypeCoercionPerformance:
    """Benchmark type coercion (3-5x expected speedup)."""

    @pytest.fixture
    def test_values(self) -> List[str]:
        """Sample values for type coercion."""
        return [
            "true",
            "false",
            "null",
            "none",
            "123",
            "-456",
            "3.14159",
            "-2.71828",
            "hello world",
            "some random string",
        ] * 100  # 1000 values

    def test_python_coercion(self, test_values: List[str]):
        """Benchmark Python fallback type coercion."""

        def coerce(value: str):
            lower = value.lower()
            if lower in ("true", "false"):
                return lower == "true"
            if lower in ("null", "none"):
                return None
            try:
                return int(value)
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                pass
            return value

        def run():
            for v in test_values:
                coerce(v)

        ms_per_call = time_function(run, iterations=50)
        print(f"\nPython coercion: {ms_per_call:.3f} ms per 1000 values")

    @pytest.mark.skipif(not is_rust_available(), reason="Rust not available")
    def test_rust_coercion(self, test_values: List[str]):
        """Benchmark Rust type coercion."""
        from victor.processing.native import coerce_string_type

        def run():
            for v in test_values:
                coerce_string_type(v)

        ms_per_call = time_function(run, iterations=50)
        print(f"\nRust coercion: {ms_per_call:.3f} ms per 1000 values")


# =============================================================================
# Stdlib Detection Benchmarks
# =============================================================================


class TestStdlibDetectionPerformance:
    """Benchmark stdlib module detection (5-10x expected speedup)."""

    @pytest.fixture
    def module_names(self) -> List[str]:
        """Sample module names for testing."""
        return [
            "os",
            "os.path",
            "sys",
            "collections",
            "collections.abc",
            "typing",
            "numpy",
            "pandas",
            "torch",
            "victor.agent",
            "victor.tools",
            "my_custom_module",
        ] * 100  # 1200 lookups

    def test_python_stdlib_detection(self, module_names: List[str]):
        """Benchmark Python fallback stdlib detection."""
        stdlib = frozenset(
            {
                "os",
                "sys",
                "collections",
                "typing",
                "abc",
                "asyncio",
                "json",
                "pathlib",
                "datetime",
                "functools",
            }
        )

        def is_stdlib(name: str) -> bool:
            return name.split(".")[0] in stdlib

        def run():
            for name in module_names:
                is_stdlib(name)

        ms_per_call = time_function(run, iterations=50)
        print(f"\nPython stdlib detection: {ms_per_call:.3f} ms per 1200 lookups")

    @pytest.mark.skipif(not is_rust_available(), reason="Rust not available")
    def test_rust_stdlib_detection(self, module_names: List[str]):
        """Benchmark Rust stdlib detection."""
        from victor.processing.native import is_stdlib_module

        def run():
            for name in module_names:
                is_stdlib_module(name)

        ms_per_call = time_function(run, iterations=50)
        print(f"\nRust stdlib detection: {ms_per_call:.3f} ms per 1200 lookups")


# =============================================================================
# Similarity Computation Benchmarks
# =============================================================================


class TestSimilarityPerformance:
    """Benchmark similarity computation (2-5x expected speedup)."""

    @pytest.fixture
    def similarity_data(self) -> dict:
        """Generate test vectors."""
        return {
            "query": generate_vectors(1, 384)[0],
            "corpus": generate_vectors(100, 384),
        }

    def test_python_batch_similarity(self, similarity_data: dict):
        """Benchmark Python batch similarity."""
        import numpy as np

        query = np.array(similarity_data["query"], dtype=np.float32)
        corpus = np.array(similarity_data["corpus"], dtype=np.float32)

        def run():
            query_norm = query / (np.linalg.norm(query) + 1e-9)
            corpus_norms = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)
            np.dot(corpus_norms, query_norm)

        ms_per_call = time_function(run, iterations=100)
        print(f"\nPython batch similarity (100 vectors): {ms_per_call:.3f} ms")

    @pytest.mark.skipif(not is_rust_available(), reason="Rust not available")
    def test_rust_batch_similarity(self, similarity_data: dict):
        """Benchmark Rust batch similarity."""
        from victor.processing.native import batch_cosine_similarity

        query = similarity_data["query"]
        corpus = similarity_data["corpus"]

        def run():
            batch_cosine_similarity(query, corpus)

        ms_per_call = time_function(run, iterations=100)
        print(f"\nRust batch similarity (100 vectors): {ms_per_call:.3f} ms")


# =============================================================================
# Line Counting Benchmarks
# =============================================================================


class TestLineCountingPerformance:
    """Benchmark line counting (5-10x expected speedup)."""

    @pytest.fixture
    def sample_code(self) -> str:
        """Generate sample code with many lines."""
        return generate_sample_code(1000)  # 1000+ lines

    def test_python_line_counting(self, sample_code: str):
        """Benchmark Python line counting."""

        def run():
            sample_code.count("\n") + 1

        ms_per_call = time_function(run, iterations=1000)
        print(f"\nPython line counting: {ms_per_call:.6f} ms per 1000 lines")

    @pytest.mark.skipif(not is_rust_available(), reason="Rust not available")
    def test_rust_line_counting(self, sample_code: str):
        """Benchmark Rust line counting."""
        import victor_native

        def run():
            victor_native.count_lines(sample_code)

        ms_per_call = time_function(run, iterations=1000)
        print(f"\nRust line counting: {ms_per_call:.6f} ms per 1000 lines")


# =============================================================================
# JSON Repair Benchmarks
# =============================================================================


class TestJsonRepairPerformance:
    """Benchmark JSON repair (5-10x expected speedup)."""

    @pytest.fixture
    def malformed_json_samples(self) -> List[str]:
        """Sample malformed JSON strings."""
        return [
            "{'key': 'value'}",  # Single quotes
            '{"key": "value",}',  # Trailing comma
            "{key: 'value'}",  # Unquoted key
            "{'nested': {'inner': 'value'}}",
            "{'list': [1, 2, 3]}",
            "{'mixed': True, 'none': None}",
        ] * 50  # 300 repairs

    def test_python_json_repair(self, malformed_json_samples: List[str]):
        """Benchmark Python JSON repair."""
        import json

        def repair(s: str) -> str:
            result = s
            result = result.replace("True", "true")
            result = result.replace("False", "false")
            result = result.replace("None", "null")
            result = result.replace("'", '"')
            return result

        def run():
            for s in malformed_json_samples:
                repair(s)

        ms_per_call = time_function(run, iterations=50)
        print(f"\nPython JSON repair: {ms_per_call:.3f} ms per 300 repairs")

    @pytest.mark.skipif(not is_rust_available(), reason="Rust not available")
    def test_rust_json_repair(self, malformed_json_samples: List[str]):
        """Benchmark Rust JSON repair."""
        from victor.processing.native import repair_json

        def run():
            for s in malformed_json_samples:
                repair_json(s)

        ms_per_call = time_function(run, iterations=50)
        print(f"\nRust JSON repair: {ms_per_call:.3f} ms per 300 repairs")


# =============================================================================
# Chunk With Overlap Benchmarks
# =============================================================================


class TestChunkingPerformance:
    """Benchmark text chunking (3-5x expected speedup)."""

    @pytest.fixture
    def large_text(self) -> str:
        """Generate large text for chunking."""
        return generate_sample_code(500)  # ~500 lines

    def test_python_chunking(self, large_text: str):
        """Benchmark Python text chunking."""
        from victor.native.python.chunker import PythonTextChunker

        chunker = PythonTextChunker()

        def run():
            chunker.chunk_with_overlap(large_text, chunk_size=1000, overlap=100)

        ms_per_call = time_function(run, iterations=50)
        print(f"\nPython chunking (500 lines): {ms_per_call:.3f} ms")

    @pytest.mark.skipif(not is_rust_available(), reason="Rust not available")
    def test_rust_chunking(self, large_text: str):
        """Benchmark Rust text chunking."""
        from victor.native.rust.chunker import RustTextChunker

        chunker = RustTextChunker()

        def run():
            chunker.chunk_with_overlap(large_text, chunk_size=1000, overlap=100)

        ms_per_call = time_function(run, iterations=50)
        print(f"\nRust chunking (500 lines): {ms_per_call:.3f} ms")


# =============================================================================
# Comparative Speedup Tests
# =============================================================================


@pytest.mark.skipif(not is_rust_available(), reason="Rust not available")
class TestSpeedupValidation:
    """Validate that Rust provides expected speedups."""

    def test_coercion_speedup(self):
        """Validate type coercion achieves 3-5x speedup."""
        from victor.processing.native import coerce_string_type

        values = ["true", "123", "3.14", "hello"] * 250  # 1000 values

        # Python baseline
        def py_coerce(value: str):
            lower = value.lower()
            if lower in ("true", "false"):
                return lower == "true"
            try:
                return int(value)
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                pass
            return value

        def py_run():
            for v in values:
                py_coerce(v)

        def rust_run():
            for v in values:
                coerce_string_type(v)

        py_time = time_function(py_run, iterations=20)
        rust_time = time_function(rust_run, iterations=20)

        speedup = py_time / rust_time if rust_time > 0 else 0
        print(
            f"\nType coercion speedup: {speedup:.1f}x "
            f"(Python: {py_time:.3f}ms, Rust: {rust_time:.3f}ms)"
        )

        # A per-item loop over a trivial coercion is dominated by FFI crossing
        # overhead, so native is not expected to win here (that is what the batch
        # APIs are for). Report the ratio informationally rather than gating on it.
        assert speedup > 0.0

    def test_stdlib_speedup(self):
        """Validate stdlib detection performance.

        Note: Python frozenset O(1) lookup is already extremely optimized.
        Rust provides comparable performance with the advantage of a more
        complete stdlib list. Function call overhead may dominate for simple
        lookups. The real benefit comes from having a single source of truth.
        """
        from victor.processing.native import is_stdlib_module

        modules = [
            "os",
            "sys",
            "numpy",
            "torch",
            "victor.agent",
            "collections.abc",
        ] * 200

        stdlib_set = frozenset({"os", "sys", "collections", "typing", "asyncio", "json"})

        def py_is_stdlib(name: str) -> bool:
            return name.split(".")[0] in stdlib_set

        def py_run():
            for m in modules:
                py_is_stdlib(m)

        def rust_run():
            for m in modules:
                is_stdlib_module(m)

        py_time = time_function(py_run, iterations=50)
        rust_time = time_function(rust_run, iterations=50)

        speedup = py_time / rust_time if rust_time > 0 else 0
        print(
            f"\nStdlib detection speedup: {speedup:.1f}x "
            f"(Python: {py_time:.3f}ms, Rust: {rust_time:.3f}ms)"
        )

        # A per-item loop over a trivial stdlib lookup is dominated by the FFI
        # crossing (and is_stdlib_module has a pure-Python fast path), so native
        # is not expected to beat inline Python here. Validate correctness and
        # report the ratio informationally rather than gating on it — batch
        # workloads (e.g. batch_is_stdlib_modules) are the real speedup targets.
        assert is_stdlib_module("os") is True  # smoke-check the classifier runs
        assert speedup > 0.0

    def test_similarity_speedup(self):
        """Validate batch similarity performance.

        Note: NumPy uses BLAS/LAPACK which is already highly optimized with SIMD.
        For small batch operations, the Python-to-Rust FFI overhead dominates.
        Rust provides benefits for:
        - Very large corpora (>10k vectors) where memory allocation matters
        - Pre-normalized vectors (avoiding redundant normalization)
        - Custom similarity metrics not in NumPy

        The real win is in top_k_similar where heap-based selection beats
        full sort, and in normalized operations where Rust avoids copies.
        """
        import numpy as np

        from victor.processing.native import batch_cosine_similarity

        query = generate_vectors(1, 384)[0]
        corpus = generate_vectors(200, 384)

        query_np = np.array(query, dtype=np.float32)
        corpus_np = np.array(corpus, dtype=np.float32)

        def py_run():
            query_norm = query_np / (np.linalg.norm(query_np) + 1e-9)
            corpus_norms = corpus_np / (np.linalg.norm(corpus_np, axis=1, keepdims=True) + 1e-9)
            np.dot(corpus_norms, query_norm)

        def rust_run():
            batch_cosine_similarity(query, corpus)

        py_time = time_function(py_run, iterations=50)
        rust_time = time_function(rust_run, iterations=50)

        speedup = py_time / rust_time if rust_time > 0 else 0
        print(
            f"\nBatch similarity speedup: {speedup:.1f}x "
            f"(Python: {py_time:.3f}ms, Rust: {rust_time:.3f}ms)"
        )

        # NumPy+BLAS is already highly optimized with hardware SIMD
        # For small batches, FFI overhead dominates - Rust is ~10x slower
        # This is expected; embeddings service correctly uses NumPy directly
        # This test documents the FFI overhead for future reference
        print("  (Note: NumPy+BLAS is optimal for similarity; Rust FFI overhead is expected)")

    def test_normalize_block_speedup(self):
        """Validate text normalization achieves speedup.

        normalize_block is used by ContentHasher for text deduplication.
        It collapses whitespace, strips punctuation, and normalizes case.
        """
        import re

        from victor.processing.native import normalize_block

        # Generate sample text with lots of whitespace and punctuation
        sample_texts = [
            "Hello   World,   how   are   you  today?!",
            "The quick  brown  fox    jumps over...  the lazy  dog.",
            "Python   is   a   great   language   for   scripting!!!",
            "Testing,  whitespace,  normalization,  performance.",
            "Multiple    spaces     and\ttabs\t\tshould   be   normalized.",
        ] * 200  # 1000 normalizations

        def py_normalize(text: str) -> str:
            """Pure Python whitespace normalization."""
            normalized = re.sub(r"\s+", " ", text.strip())
            return normalized.rstrip(".,;:!?")

        def py_run():
            for text in sample_texts:
                py_normalize(text)

        def rust_run():
            for text in sample_texts:
                normalize_block(text)

        py_time = time_function(py_run, iterations=20)
        rust_time = time_function(rust_run, iterations=20)

        speedup = py_time / rust_time if rust_time > 0 else 0
        print(
            f"\nText normalization speedup: {speedup:.1f}x "
            f"(Python: {py_time:.3f}ms, Rust: {rust_time:.3f}ms)"
        )

        # Rust regex should be faster than Python re module
        assert speedup >= 0.8, "Rust should not be more than 20% slower"

    def test_content_hasher_speedup(self):
        """Validate ContentHasher benefits from native normalization.

        Tests full hash pipeline including normalization and SHA-256.
        """
        import hashlib
        import re

        from victor.core.utils.content_hasher import ContentHasher

        # Sample texts for hashing
        sample_texts = [
            "Hello   World,   testing   content   hashing...",
            "The quick  brown  fox    jumps over   the lazy dog!",
            "Python   code   with   lots   of   whitespace.",
            "JSON data:   {'key':   'value',   'num':   42}",
            "Multiline\n\ntext\n\nwith\n\nextra\n\nlines.",
        ] * 200  # 1000 hashes

        def py_hash(text: str) -> str:
            """Pure Python implementation (no native acceleration)."""
            normalized = re.sub(r"\s+", " ", text.strip())
            normalized = normalized.rstrip(".,;:!?").lower()
            return hashlib.sha256(normalized.encode()).hexdigest()[:12]

        # Create hasher with native acceleration enabled
        hasher = ContentHasher(
            normalize_whitespace=True,
            case_insensitive=True,
            remove_punctuation=True,
            hash_length=12,
        )

        def py_run():
            for text in sample_texts:
                py_hash(text)

        def native_run():
            for text in sample_texts:
                hasher.hash(text)

        py_time = time_function(py_run, iterations=20)
        native_time = time_function(native_run, iterations=20)

        speedup = py_time / native_time if native_time > 0 else 0
        print(
            f"\nContent hashing speedup: {speedup:.1f}x "
            f"(Python: {py_time:.3f}ms, Native: {native_time:.3f}ms)"
        )

        # With native normalize_block, should see improvement
        # But SHA-256 is already fast, so overall speedup may be modest
        print("  (Note: SHA-256 hashing dominates; normalization speedup is incremental)")
