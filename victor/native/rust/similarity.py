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

"""Rust similarity computer wrapper.

Provides a protocol-compliant wrapper around the Rust SIMD-optimized
similarity functions. The wrapper delegates to victor_native functions
while maintaining the SimilarityComputerProtocol interface.

Performance characteristics:
- cosine: 2-3x faster (SIMD dot product)
- batch_cosine: 3-5x faster (parallel + SIMD)
- top_k: 2-3x faster (partial sort optimization)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import victor_native

from victor.native.observability import InstrumentedAccelerator


class RustSimilarityComputer(InstrumentedAccelerator):
    """Rust implementation of SimilarityComputerProtocol.

    Wraps the high-performance Rust SIMD-optimized similarity functions
    with protocol-compliant interface.

    Performance characteristics:
    - cosine: 2-3x faster (SIMD dot product with wide crate)
    - batch_cosine: 3-5x faster (rayon parallelization + SIMD)
    - top_k: 2-3x faster (partial sort for efficiency)
    - similarity_matrix: 3-5x faster (parallel matrix computation)
    """

    def __init__(self) -> None:
        super().__init__(backend="rust")
        self._version = victor_native.__version__

    def get_version(self) -> Optional[str]:
        return self._version

    def cosine(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Delegates to Rust SIMD-optimized implementation.

        Args:
            a: First vector
            b: Second vector (must have same length)

        Returns:
            Cosine similarity in range [-1, 1]

        Raises:
            ValueError: If vectors have different lengths
        """
        with self._timed_call("similarity_compute"):
            # Convert to f32 for Rust (native uses f32 for SIMD)
            a_f32 = [float(x) for x in a]
            b_f32 = [float(x) for x in b]
            return victor_native.cosine_similarity(a_f32, b_f32)

    def batch_cosine(self, query: List[float], corpus: List[List[float]]) -> List[float]:
        """Compute cosine similarity of query against corpus.

        Delegates to Rust parallel + SIMD implementation.

        Args:
            query: Query vector
            corpus: List of corpus vectors

        Returns:
            List of similarity scores (same order as corpus)
        """
        with self._timed_call("batch_similarity"):
            if not corpus:
                return []
            # Convert to f32 for Rust
            query_f32 = [float(x) for x in query]
            corpus_f32 = [[float(x) for x in vec] for vec in corpus]
            return victor_native.batch_cosine_similarity(query_f32, corpus_f32)

    def similarity_matrix(
        self,
        queries: List[List[float]],
        corpus: List[List[float]],
        normalize: bool = True,
    ) -> List[List[float]]:
        """Compute pairwise similarity matrix.

        Uses Rust batch_cosine for each query row.

        Args:
            queries: List of query vectors
            corpus: List of corpus vectors
            normalize: Whether to L2-normalize vectors first

        Returns:
            Matrix of shape (len(queries), len(corpus))
        """
        with self._timed_call("similarity_matrix"):
            if not queries or not corpus:
                return []

            # Convert to f32
            queries_f32 = [[float(x) for x in q] for q in queries]
            corpus_f32 = [[float(x) for x in c] for c in corpus]

            # Prefer the native similarity_matrix (one FFI crossing for the whole
            # matrix); fall back to the per-query loop on older native builds.
            matrix_fn = getattr(victor_native, "similarity_matrix", None)
            if matrix_fn is not None:
                return [list(row) for row in matrix_fn(queries_f32, corpus_f32, normalize)]

            if normalize:
                # Use pre-normalized path for efficiency
                corpus_normalized = victor_native.batch_normalize_vectors(corpus_f32)
                result = []
                for query in queries_f32:
                    row = victor_native.batch_cosine_similarity_normalized(query, corpus_normalized)
                    result.append(row)
                return result
            else:
                # Direct computation without normalization
                result = []
                for query in queries_f32:
                    row = victor_native.batch_cosine_similarity(query, corpus_f32)
                    result.append(row)
                return result

    def top_k(
        self, query: List[float], corpus: List[List[float]], k: int
    ) -> List[Tuple[int, float]]:
        """Find top-k most similar vectors.

        Delegates to Rust implementation with partial sort optimization.

        Args:
            query: Query vector
            corpus: List of corpus vectors
            k: Number of top results

        Returns:
            List of (index, similarity) tuples, sorted by similarity desc
        """
        with self._timed_call("top_k_similarity"):
            if not corpus:
                return []
            # Convert to f32 for Rust
            query_f32 = [float(x) for x in query]
            corpus_f32 = [[float(x) for x in vec] for vec in corpus]
            return victor_native.top_k_similar(query_f32, corpus_f32, k)
