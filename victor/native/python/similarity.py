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

"""Pure Python similarity computer implementation.

Provides vector similarity computation using NumPy.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from victor.native.observability import InstrumentedAccelerator

# Try to use NumPy for better performance
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class PythonSimilarityComputer(InstrumentedAccelerator):
    """Pure Python implementation of SimilarityComputerProtocol.

    Uses NumPy when available for better performance.
    Falls back to pure Python otherwise.
    """

    def __init__(self) -> None:
        super().__init__(backend="python")
        self._version = "1.0.0"
        self._use_numpy = NUMPY_AVAILABLE

    def get_version(self) -> Optional[str]:
        return self._version

    def cosine(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector (must have same length)

        Returns:
            Cosine similarity in range [-1, 1]

        Raises:
            ValueError: If vectors have different lengths
        """
        with self._timed_call("similarity_compute"):
            if len(a) != len(b):
                raise ValueError(f"Vectors must have same length: {len(a)} vs {len(b)}")

            if len(a) == 0:
                return 0.0

            if self._use_numpy:
                return self._cosine_numpy(a, b)
            return self._cosine_pure(a, b)

    def batch_cosine(self, query: List[float], corpus: List[List[float]]) -> List[float]:
        """Compute cosine similarity of query against corpus.

        Args:
            query: Query vector
            corpus: List of corpus vectors

        Returns:
            List of similarity scores (same order as corpus)
        """
        with self._timed_call("batch_similarity"):
            if not corpus:
                return []

            if self._use_numpy:
                return self._batch_cosine_numpy(query, corpus)
            return [self._cosine_pure(query, vec) for vec in corpus]

    def similarity_matrix(
        self,
        queries: List[List[float]],
        corpus: List[List[float]],
        normalize: bool = True,
    ) -> List[List[float]]:
        """Compute pairwise similarity matrix.

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

            if self._use_numpy:
                return self._similarity_matrix_numpy(queries, corpus, normalize)

            # Pure Python fallback
            result = []
            for query in queries:
                row = [self._cosine_pure(query, vec) for vec in corpus]
                result.append(row)
            return result

    def top_k(
        self, query: List[float], corpus: List[List[float]], k: int
    ) -> List[Tuple[int, float]]:
        """Find top-k most similar vectors.

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

            scores = self.batch_cosine(query, corpus)
            indexed = list(enumerate(scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            return indexed[:k]

    def _cosine_pure(self, a: List[float], b: List[float]) -> float:
        """Pure Python cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _cosine_numpy(self, a: List[float], b: List[float]) -> float:
        """NumPy cosine similarity."""
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)

        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    def _batch_cosine_numpy(self, query: List[float], corpus: List[List[float]]) -> List[float]:
        """NumPy batch cosine similarity."""
        query_arr = np.array(query, dtype=np.float32)
        corpus_arr = np.array(corpus, dtype=np.float32)

        # Normalize query
        query_norm = np.linalg.norm(query_arr)
        if query_norm == 0:
            return [0.0] * len(corpus)
        query_normalized = query_arr / query_norm

        # Normalize corpus
        corpus_norms = np.linalg.norm(corpus_arr, axis=1, keepdims=True)
        # Avoid division by zero
        corpus_norms = np.where(corpus_norms == 0, 1, corpus_norms)
        corpus_normalized = corpus_arr / corpus_norms

        # Compute similarities
        similarities = np.dot(corpus_normalized, query_normalized)
        return similarities.tolist()

    def _similarity_matrix_numpy(
        self,
        queries: List[List[float]],
        corpus: List[List[float]],
        normalize: bool,
    ) -> List[List[float]]:
        """NumPy similarity matrix computation."""
        queries_arr = np.array(queries, dtype=np.float32)
        corpus_arr = np.array(corpus, dtype=np.float32)

        if normalize:
            # Normalize queries
            query_norms = np.linalg.norm(queries_arr, axis=1, keepdims=True)
            query_norms = np.where(query_norms == 0, 1, query_norms)
            queries_arr = queries_arr / query_norms

            # Normalize corpus
            corpus_norms = np.linalg.norm(corpus_arr, axis=1, keepdims=True)
            corpus_norms = np.where(corpus_norms == 0, 1, corpus_norms)
            corpus_arr = corpus_arr / corpus_norms

        # Matrix multiply: (n_queries, dim) @ (dim, n_corpus) = (n_queries, n_corpus)
        similarities = np.dot(queries_arr, corpus_arr.T)
        return similarities.tolist()
