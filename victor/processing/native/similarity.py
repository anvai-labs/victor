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

"""Similarity computation functions with native acceleration."""

from typing import List, Tuple

import numpy as np

from victor.processing.native._base import (
    _NATIVE_AVAILABLE,
    _native,
    should_use_native_for_operation,
)
from victor.processing.native.observability import dispatch_with_observability


@dispatch_with_observability("cosine_similarity")
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity value between -1 and 1

    Raises:
        ValueError: If vectors have different lengths
    """
    if _NATIVE_AVAILABLE:
        return _native.cosine_similarity(a, b)

    # Pure Python fallback using NumPy
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)

    if len(a_arr) != len(b_arr):
        raise ValueError(f"Vectors must have same length: {len(a_arr)} vs {len(b_arr)}")

    if len(a_arr) == 0:
        return 0.0

    norm_a = np.linalg.norm(a_arr) + 1e-9
    norm_b = np.linalg.norm(b_arr) + 1e-9
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


@dispatch_with_observability("batch_cosine_similarity")
def batch_cosine_similarity(query: List[float], corpus: List[List[float]]) -> List[float]:
    """Compute cosine similarity between a query vector and multiple corpus vectors.

    Uses NumPy+BLAS for batch operations (benchmark shows ~6x faster than Rust
    for this operation due to hardware-optimized matrix multiplication).

    Args:
        query: Query embedding vector
        corpus: List of corpus embedding vectors

    Returns:
        List of similarity scores, one per corpus vector

    Raises:
        ValueError: If query dimension doesn't match corpus dimensions
    """
    # Benchmark-aware dispatch: prefer Python (NumPy+BLAS) for batch operations
    if should_use_native_for_operation("batch_cosine_similarity"):
        return _native.batch_cosine_similarity(query, corpus)

    # NumPy+BLAS fallback (actually faster for this operation)
    if not corpus:
        return []

    query_arr = np.array(query, dtype=np.float32)
    corpus_arr = np.array(corpus, dtype=np.float32)

    if corpus_arr.shape[1] != len(query_arr):
        raise ValueError(
            f"Dimension mismatch: query has {len(query_arr)} dims, "
            f"corpus has {corpus_arr.shape[1]} dims"
        )

    # Normalize
    query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-9)
    corpus_norms = corpus_arr / (np.linalg.norm(corpus_arr, axis=1, keepdims=True) + 1e-9)

    # Compute similarities
    similarities = np.dot(corpus_norms, query_norm)
    return similarities.tolist()


@dispatch_with_observability("top_k_similar")
def top_k_similar(
    query: List[float], corpus: List[List[float]], k: int = 10
) -> List[Tuple[int, float]]:
    """Find top-k most similar vectors from a corpus.

    Args:
        query: Query embedding vector
        corpus: List of corpus embedding vectors
        k: Number of top results to return

    Returns:
        List of (index, similarity) tuples, sorted by similarity descending
    """
    if _NATIVE_AVAILABLE:
        return _native.top_k_similar(query, corpus, k)

    # Pure Python fallback
    similarities = batch_cosine_similarity(query, corpus)
    indexed = list(enumerate(similarities))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:k]


def batch_normalize_vectors(vectors: List[List[float]]) -> List[List[float]]:
    """Normalize vectors to unit length for efficient similarity computation.

    Pre-normalizing vectors allows subsequent similarity computations to
    skip redundant norm calculations, providing ~2x speedup for batch operations.

    Args:
        vectors: List of vectors to normalize

    Returns:
        List of normalized vectors (unit length)
    """
    if _NATIVE_AVAILABLE:
        return _native.batch_normalize_vectors(vectors)

    # Pure Python fallback using NumPy
    if not vectors:
        return []

    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    normalized = arr / norms
    return normalized.tolist()


def batch_cosine_similarity_normalized(
    query: List[float], normalized_corpus: List[List[float]]
) -> List[float]:
    """Compute cosine similarities with a pre-normalized corpus.

    This is faster than batch_cosine_similarity when the corpus has already
    been normalized via batch_normalize_vectors, as it avoids redundant
    norm calculations.

    Args:
        query: Query embedding vector (will be normalized internally)
        normalized_corpus: List of pre-normalized corpus vectors (unit length)

    Returns:
        List of similarity scores, one per corpus vector
    """
    if _NATIVE_AVAILABLE:
        return _native.batch_cosine_similarity_normalized(query, normalized_corpus)

    # Pure Python fallback using NumPy
    if not normalized_corpus:
        return []

    query_arr = np.array(query, dtype=np.float32)
    corpus_arr = np.array(normalized_corpus, dtype=np.float32)

    # Normalize query
    query_normalized = query_arr / (np.linalg.norm(query_arr) + 1e-9)

    # For pre-normalized corpus, similarity is just dot product
    similarities = np.dot(corpus_arr, query_normalized)
    return similarities.tolist()


def top_k_similar_normalized(
    query: List[float], normalized_corpus: List[List[float]], k: int = 10
) -> List[Tuple[int, float]]:
    """Find top-k similar vectors from a pre-normalized corpus.

    More efficient version of top_k_similar when corpus is already normalized
    via batch_normalize_vectors.

    Args:
        query: Query embedding vector
        normalized_corpus: List of pre-normalized corpus vectors
        k: Number of top results to return

    Returns:
        List of (index, similarity) tuples, sorted by similarity descending
    """
    if _NATIVE_AVAILABLE:
        return _native.top_k_similar_normalized(query, normalized_corpus, k)

    # Pure Python fallback using heap for efficiency
    import heapq

    similarities = batch_cosine_similarity_normalized(query, normalized_corpus)

    # Use heapq.nlargest for efficient top-k selection
    indexed = [(sim, i) for i, sim in enumerate(similarities)]
    top_k = heapq.nlargest(k, indexed)

    # Convert back to (index, similarity) format
    return [(i, sim) for sim, i in top_k]
