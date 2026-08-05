// Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! High-performance cosine similarity computation.
//!
//! This module provides SIMD-optimized cosine similarity calculations
//! for embedding vectors, commonly used in semantic search and tool selection.
//!
//! Features:
//! - Portable SIMD using the `wide` crate (works on x86, ARM, etc.)
//! - Batch processing with parallelization via rayon
//! - Memory-efficient operations avoiding unnecessary allocations

use pyo3::prelude::*;
use rayon::prelude::*;
use wide::f32x8;

const EPSILON: f32 = 1e-9;
const SIMD_WIDTH: usize = 8;

/// Compute the L2 norm of a vector using SIMD.
#[inline]
fn simd_norm(vec: &[f32]) -> f32 {
    let chunks = vec.chunks_exact(SIMD_WIDTH);
    let remainder = chunks.remainder();

    let mut sum = f32x8::ZERO;
    for chunk in chunks {
        let v = f32x8::from(unsafe { *(chunk.as_ptr() as *const [f32; 8]) });
        sum += v * v;
    }

    let mut total: f32 = sum.reduce_add();

    // Handle remainder
    for &x in remainder {
        total += x * x;
    }

    total.sqrt().max(EPSILON)
}

/// L2-normalize a vector in place-free form (returns a new Vec).
/// Falls back to the original vector if the norm is ~0.
#[inline]
fn normalize_vec(vec: &[f32]) -> Vec<f32> {
    let norm = simd_norm(vec);
    if norm <= EPSILON {
        return vec.to_vec();
    }
    vec.iter().map(|x| x / norm).collect()
}

/// Compute dot product using SIMD.
#[inline]
fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let a_chunks = a.chunks_exact(SIMD_WIDTH);
    let b_chunks = b.chunks_exact(SIMD_WIDTH);
    let a_remainder = a_chunks.remainder();
    let b_remainder = b_chunks.remainder();

    let mut sum = f32x8::ZERO;
    for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
        let av = f32x8::from(unsafe { *(a_chunk.as_ptr() as *const [f32; 8]) });
        let bv = f32x8::from(unsafe { *(b_chunk.as_ptr() as *const [f32; 8]) });
        sum += av * bv;
    }

    let mut total: f32 = sum.reduce_add();

    // Handle remainder
    for (a, b) in a_remainder.iter().zip(b_remainder.iter()) {
        total += a * b;
    }

    total
}

/// Compute cosine similarity between two vectors.
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity value between -1 and 1
///
/// # Raises
/// * `ValueError` - If vectors have different lengths
#[pyfunction]
pub fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Vectors must have same length: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    if a.is_empty() {
        return Ok(0.0);
    }

    let norm_a = simd_norm(&a);
    let norm_b = simd_norm(&b);
    let dot = simd_dot(&a, &b);

    Ok(dot / (norm_a * norm_b))
}

/// Rust-facing implementation of [`batch_cosine_similarity`] (no GIL, no
/// `PyErr`): the wrapper releases the GIL around this, and errors are plain
/// `String`s because a `PyErr` is `!Ungil` and cannot cross `allow_threads`.
pub fn batch_cosine_similarity_impl(
    query: Vec<f32>,
    corpus: Vec<Vec<f32>>,
) -> Result<Vec<f32>, String> {
    if corpus.is_empty() {
        return Ok(Vec::new());
    }

    // Validate dimensions
    let query_dim = query.len();
    for (i, vec) in corpus.iter().enumerate() {
        if vec.len() != query_dim {
            return Err(format!(
                "Dimension mismatch: query has {} dims, corpus[{}] has {} dims",
                query_dim,
                i,
                vec.len()
            ));
        }
    }

    // Pre-compute query norm
    let query_norm = simd_norm(&query);

    // Use parallel processing for larger corpora
    let results: Vec<f32> = if corpus.len() > 100 {
        corpus
            .par_iter()
            .map(|vec| {
                let corpus_norm = simd_norm(vec);
                let dot = simd_dot(&query, vec);
                dot / (query_norm * corpus_norm)
            })
            .collect()
    } else {
        corpus
            .iter()
            .map(|vec| {
                let corpus_norm = simd_norm(vec);
                let dot = simd_dot(&query, vec);
                dot / (query_norm * corpus_norm)
            })
            .collect()
    };

    Ok(results)
}

/// Compute cosine similarity between a query vector and multiple corpus vectors.
///
/// This function is optimized for the common case of comparing one query
/// against many candidates, using parallel processing for large corpora.
///
/// # Arguments
/// * `query` - Query embedding vector
/// * `corpus` - List of corpus embedding vectors
///
/// # Returns
/// List of similarity scores, one per corpus vector
///
/// # Raises
/// * `ValueError` - If query dimension doesn't match corpus dimensions
#[pyfunction]
pub fn batch_cosine_similarity(
    py: Python<'_>,
    query: Vec<f32>,
    corpus: Vec<Vec<f32>>,
) -> PyResult<Vec<f32>> {
    py.allow_threads(|| batch_cosine_similarity_impl(query, corpus))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Compute a full similarity matrix of `queries` x `corpus` in a SINGLE FFI
/// crossing (rayon-parallel over queries), instead of looping
/// `batch_cosine_similarity` from Python (which crossed once per query).
///
/// Results are identical to per-query `batch_cosine_similarity`.
///
/// # Arguments
/// * `queries` - Query embedding vectors (matrix rows)
/// * `corpus` - Corpus embedding vectors (matrix columns)
/// * `normalize` - If true, pre-normalize the corpus once and reuse it for
///   every query (avoids recomputing corpus norms per row)
pub fn similarity_matrix_impl(
    queries: Vec<Vec<f32>>,
    corpus: Vec<Vec<f32>>,
    normalize: bool,
) -> Result<Vec<Vec<f32>>, String> {
    if queries.is_empty() || corpus.is_empty() {
        return Ok(Vec::new());
    }

    // Validate dimensions: every corpus vector must match the query dim.
    let qdim = queries[0].len();
    for (i, q) in queries.iter().enumerate() {
        if q.len() != qdim {
            return Err(format!(
                "Query dimension mismatch: queries[0] has {} dims, queries[{}] has {}",
                qdim,
                i,
                q.len()
            ));
        }
    }
    for (i, c) in corpus.iter().enumerate() {
        if c.len() != qdim {
            return Err(format!(
                "Dimension mismatch: query has {} dims, corpus[{}] has {}",
                qdim,
                i,
                c.len()
            ));
        }
    }

    if normalize {
        // Pre-normalize corpus once; reuse for every query row.
        let corpus_normed: Vec<Vec<f32>> = corpus.par_iter().map(|c| normalize_vec(c)).collect();
        Ok(queries
            .par_iter()
            .map(|q| {
                let qn = normalize_vec(q);
                corpus_normed
                    .iter()
                    .map(|c| simd_dot(&qn, c))
                    .collect::<Vec<f32>>()
            })
            .collect())
    } else {
        // Cosine similarity normalizes per dot product; same result, slightly
        // more work (recomputes corpus norms per row) but avoids the helper.
        Ok(queries
            .par_iter()
            .map(|q| {
                let qn = simd_norm(q);
                corpus
                    .iter()
                    .map(|c| {
                        let cn = simd_norm(c);
                        simd_dot(q, c) / (qn * cn)
                    })
                    .collect::<Vec<f32>>()
            })
            .collect())
    }
}

#[pyfunction]
#[pyo3(signature = (queries, corpus, normalize = true))]
pub fn similarity_matrix(
    py: Python<'_>,
    queries: Vec<Vec<f32>>,
    corpus: Vec<Vec<f32>>,
    normalize: bool,
) -> PyResult<Vec<Vec<f32>>> {
    py.allow_threads(|| similarity_matrix_impl(queries, corpus, normalize))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Find top-k most similar vectors from a corpus.
///
/// # Arguments
/// * `query` - Query embedding vector
/// * `corpus` - List of corpus embedding vectors
/// * `k` - Number of top results to return
///
/// # Returns
/// List of (index, similarity) tuples, sorted by similarity descending
#[pyfunction]
#[pyo3(signature = (query, corpus, k = 10))]
pub fn top_k_similar(
    query: Vec<f32>,
    corpus: Vec<Vec<f32>>,
    k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    if k == 0 || corpus.is_empty() {
        return Ok(Vec::new());
    }

    let similarities = batch_cosine_similarity_impl(query, corpus)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    // Create (index, similarity) pairs
    let mut indexed: Vec<(usize, f32)> = similarities.into_iter().enumerate().collect();

    // Partial sort for efficiency - only need top k
    let k = k.min(indexed.len());
    if k == 0 {
        return Ok(Vec::new());
    }
    indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top k and sort them
    let mut top_k: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();
    top_k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(top_k)
}

/// Normalize vectors in-place to unit length.
///
/// This is useful for pre-processing embeddings before similarity computation,
/// reducing redundant norm calculations in batch operations.
///
/// # Arguments
/// * `vectors` - List of vectors to normalize in-place
///
/// # Returns
/// List of normalized vectors (unit length)
pub fn batch_normalize_vectors_impl(vectors: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    if vectors.is_empty() {
        return Vec::new();
    }

    // Use parallel processing for larger batches
    if vectors.len() > 100 {
        vectors
            .par_iter()
            .map(|vec| normalize_vector(vec))
            .collect()
    } else {
        vectors.iter().map(|vec| normalize_vector(vec)).collect()
    }
}

#[pyfunction]
pub fn batch_normalize_vectors(py: Python<'_>, vectors: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
    Ok(py.allow_threads(|| batch_normalize_vectors_impl(vectors)))
}

/// Normalize a single vector to unit length using SIMD.
#[inline]
fn normalize_vector(vec: &[f32]) -> Vec<f32> {
    let norm = simd_norm(vec);
    vec.iter().map(|&x| x / norm).collect()
}

/// Compute cosine similarities between a query and pre-normalized corpus vectors.
///
/// This is faster than batch_cosine_similarity when the corpus is already normalized,
/// as it avoids redundant norm calculations.
///
/// # Arguments
/// * `query` - Query embedding vector (will be normalized internally)
/// * `normalized_corpus` - List of pre-normalized corpus vectors (unit length)
///
/// # Returns
/// List of similarity scores, one per corpus vector
pub fn batch_cosine_similarity_normalized_impl(
    query: Vec<f32>,
    normalized_corpus: Vec<Vec<f32>>,
) -> Result<Vec<f32>, String> {
    if normalized_corpus.is_empty() {
        return Ok(Vec::new());
    }

    // Validate dimensions
    let query_dim = query.len();
    for (i, vec) in normalized_corpus.iter().enumerate() {
        if vec.len() != query_dim {
            return Err(format!(
                "Dimension mismatch: query has {} dims, corpus[{}] has {} dims",
                query_dim,
                i,
                vec.len()
            ));
        }
    }

    // Normalize query once
    let query_normalized = normalize_vector(&query);

    // For pre-normalized vectors, similarity is just dot product
    let results: Vec<f32> = if normalized_corpus.len() > 100 {
        normalized_corpus
            .par_iter()
            .map(|vec| simd_dot(&query_normalized, vec))
            .collect()
    } else {
        normalized_corpus
            .iter()
            .map(|vec| simd_dot(&query_normalized, vec))
            .collect()
    };

    Ok(results)
}

#[pyfunction]
pub fn batch_cosine_similarity_normalized(
    py: Python<'_>,
    query: Vec<f32>,
    normalized_corpus: Vec<Vec<f32>>,
) -> PyResult<Vec<f32>> {
    py.allow_threads(|| batch_cosine_similarity_normalized_impl(query, normalized_corpus))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Find top-k similar vectors from a pre-normalized corpus.
///
/// More efficient version of top_k_similar when corpus is already normalized.
///
/// # Arguments
/// * `query` - Query embedding vector
/// * `normalized_corpus` - List of pre-normalized corpus vectors
/// * `k` - Number of top results to return
///
/// # Returns
/// List of (index, similarity) tuples, sorted by similarity descending
#[pyfunction]
#[pyo3(signature = (query, normalized_corpus, k = 10))]
pub fn top_k_similar_normalized(
    query: Vec<f32>,
    normalized_corpus: Vec<Vec<f32>>,
    k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    let similarities = batch_cosine_similarity_normalized_impl(query, normalized_corpus)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    // Use a binary heap for efficient top-k selection
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(PartialEq)]
    struct ScoredIndex(usize, f32);

    impl Eq for ScoredIndex {}

    impl Ord for ScoredIndex {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse order for min-heap behavior
            other.1.partial_cmp(&self.1).unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for ScoredIndex {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let k = k.min(similarities.len());
    let mut heap: BinaryHeap<ScoredIndex> = BinaryHeap::with_capacity(k + 1);

    for (idx, sim) in similarities.into_iter().enumerate() {
        heap.push(ScoredIndex(idx, sim));
        if heap.len() > k {
            heap.pop(); // Remove smallest
        }
    }

    // Extract and sort results
    let mut results: Vec<(usize, f32)> = heap.into_iter().map(|si| (si.0, si.1)).collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(v.clone(), v).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(a, b).unwrap();
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(a, b).unwrap();
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_cosine_similarity() {
        let query = vec![1.0, 0.0, 0.0];
        let corpus = vec![
            vec![1.0, 0.0, 0.0],  // identical
            vec![0.0, 1.0, 0.0],  // orthogonal
            vec![-1.0, 0.0, 0.0], // opposite
        ];
        let sims = batch_cosine_similarity_impl(query, corpus).unwrap();
        assert_eq!(sims.len(), 3);
        assert!((sims[0] - 1.0).abs() < 1e-6);
        assert!(sims[1].abs() < 1e-6);
        assert!((sims[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_matrix_matches_per_query() {
        // The matrix result must equal per-query batch_cosine_similarity for both
        // normalize branches (the correctness invariant the Python wrapper relies on).
        let queries = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.7, 0.7, 0.0],
        ];
        let corpus = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![-1.0, 0.0, 0.0],
        ];

        for normalize in [true, false] {
            let matrix =
                similarity_matrix_impl(queries.clone(), corpus.clone(), normalize).unwrap();
            assert_eq!(matrix.len(), queries.len());
            for (i, q) in queries.iter().enumerate() {
                let row = batch_cosine_similarity_impl(q.clone(), corpus.clone()).unwrap();
                assert_eq!(matrix[i].len(), row.len());
                for (a, b) in matrix[i].iter().zip(row.iter()) {
                    assert!(
                        (a - b).abs() < 1e-5,
                        "matrix mismatch at row {i} (normalize={normalize})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_similarity_matrix_empty() {
        assert_eq!(
            similarity_matrix_impl(vec![], vec![vec![1.0, 0.0]], true).unwrap(),
            Vec::<Vec<f32>>::new()
        );
        assert_eq!(
            similarity_matrix_impl(vec![vec![1.0, 0.0]], vec![], true).unwrap(),
            Vec::<Vec<f32>>::new()
        );
    }

    #[test]
    fn test_top_k_similar() {
        let query = vec![1.0, 0.0];
        let corpus = vec![
            vec![0.5, 0.5],
            vec![1.0, 0.0], // Most similar
            vec![0.0, 1.0],
            vec![0.9, 0.1], // Second most similar
        ];
        let top = top_k_similar(query, corpus, 2).unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1); // Index of most similar
        assert_eq!(top[1].0, 3); // Index of second most similar
    }

    #[test]
    fn test_top_k_similar_zero_k() {
        let query = vec![1.0, 0.0];
        let corpus = vec![vec![1.0, 0.0]];
        let top = top_k_similar(query, corpus, 0).unwrap();
        assert!(top.is_empty());
    }

    #[test]
    fn test_simd_norm() {
        // Test with various sizes to exercise SIMD and remainder
        let v3 = vec![3.0, 4.0, 0.0];
        assert!((simd_norm(&v3) - 5.0).abs() < 1e-6);

        let v9 = vec![1.0; 9];
        assert!((simd_norm(&v9) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(cosine_similarity(a, b).is_err());
    }

    #[test]
    fn test_batch_normalize_vectors() {
        let vectors = vec![
            vec![3.0, 4.0], // norm = 5
            vec![1.0, 0.0], // norm = 1
            vec![0.0, 2.0], // norm = 2
        ];
        let normalized = batch_normalize_vectors_impl(vectors);

        assert_eq!(normalized.len(), 3);

        // Check first vector is normalized to unit length
        let norm0: f32 = normalized[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm0 - 1.0).abs() < 1e-6);

        // Check values
        assert!((normalized[0][0] - 0.6).abs() < 1e-6);
        assert!((normalized[0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_batch_cosine_similarity_normalized() {
        // Pre-normalized vectors (unit length)
        let corpus = vec![
            vec![1.0, 0.0, 0.0], // unit vector along x
            vec![0.0, 1.0, 0.0], // unit vector along y
            vec![0.0, 0.0, 1.0], // unit vector along z
        ];
        let query = vec![1.0, 0.0, 0.0]; // Will be normalized internally

        let sims = batch_cosine_similarity_normalized_impl(query, corpus).unwrap();
        assert_eq!(sims.len(), 3);
        assert!((sims[0] - 1.0).abs() < 1e-6); // Identical
        assert!(sims[1].abs() < 1e-6); // Orthogonal
        assert!(sims[2].abs() < 1e-6); // Orthogonal
    }

    #[test]
    fn test_top_k_similar_normalized() {
        // Pre-normalized vectors
        let corpus = vec![
            vec![0.6, 0.8], // ~53 degrees from x-axis
            vec![1.0, 0.0], // x-axis (most similar to query)
            vec![0.0, 1.0], // y-axis (least similar)
            vec![0.8, 0.6], // ~37 degrees (second most similar)
        ];
        let query = vec![1.0, 0.0]; // x-axis

        let top = top_k_similar_normalized(query, corpus, 2).unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1); // Index 1 is most similar (x-axis)
        assert_eq!(top[1].0, 3); // Index 3 is second most similar
    }

    #[test]
    fn test_normalize_vector() {
        let v = vec![3.0, 4.0];
        let normalized = normalize_vector(&v);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);

        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// EmbeddingIndex — pre-normalized corpus for batch similarity queries
// ---------------------------------------------------------------------------

/// Pre-normalized embedding corpus for batch similarity queries.
/// Stores vectors in a contiguous flat buffer for SIMD-friendly access.
#[pyclass]
pub struct EmbeddingIndex {
    data: Vec<f32>, // flat: dim * n_vectors
    dim: usize,
    n_vectors: usize,
    labels: Vec<String>,
}

#[pymethods]
impl EmbeddingIndex {
    /// Create a new EmbeddingIndex from a list of vectors and their labels.
    ///
    /// Each vector is L2-normalized before storage so that dot product
    /// equals cosine similarity at query time.
    ///
    /// # Arguments
    /// * `vectors` - List of embedding vectors (must all have the same dimension)
    /// * `labels` - List of labels, one per vector
    ///
    /// # Raises
    /// * `ValueError` - If vectors/labels lengths differ or dimensions are inconsistent
    #[new]
    pub fn new(vectors: Vec<Vec<f32>>, labels: Vec<String>) -> PyResult<Self> {
        if vectors.len() != labels.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vectors length ({}) must match labels length ({})",
                vectors.len(),
                labels.len()
            )));
        }
        if vectors.is_empty() {
            return Ok(Self {
                data: Vec::new(),
                dim: 0,
                n_vectors: 0,
                labels,
            });
        }

        let dim = vectors[0].len();
        if dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Vectors must have non-zero dimension",
            ));
        }
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Dimension mismatch: vectors[0] has {} dims, vectors[{}] has {} dims",
                    dim,
                    i,
                    v.len()
                )));
            }
        }

        let n_vectors = vectors.len();
        let mut data = Vec::with_capacity(dim * n_vectors);
        for v in &vectors {
            let norm = simd_norm(v);
            for &x in v {
                data.push(x / norm);
            }
        }

        Ok(Self {
            data,
            dim,
            n_vectors,
            labels,
        })
    }

    /// Add more vectors to the index.
    ///
    /// # Arguments
    /// * `vectors` - Additional embedding vectors
    /// * `labels` - Labels for the new vectors
    ///
    /// # Raises
    /// * `ValueError` - If dimensions don't match existing index or vectors/labels mismatch
    pub fn add(&mut self, vectors: Vec<Vec<f32>>, labels: Vec<String>) -> PyResult<()> {
        if vectors.len() != labels.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vectors length ({}) must match labels length ({})",
                vectors.len(),
                labels.len()
            )));
        }
        if vectors.is_empty() {
            return Ok(());
        }

        // If the index is empty, initialize dim from the first vector
        if self.n_vectors == 0 {
            self.dim = vectors[0].len();
            if self.dim == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Vectors must have non-zero dimension",
                ));
            }
        }

        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Dimension mismatch: index has {} dims, vectors[{}] has {} dims",
                    self.dim,
                    i,
                    v.len()
                )));
            }
        }

        self.data.reserve(self.dim * vectors.len());
        for v in &vectors {
            let norm = simd_norm(v);
            for &x in v {
                self.data.push(x / norm);
            }
        }

        self.n_vectors += vectors.len();
        self.labels.extend(labels);
        Ok(())
    }

    /// Query the index for the top-k most similar vectors.
    ///
    /// The query vector is L2-normalized internally.  Similarity is computed
    /// as the dot product with each stored (pre-normalized) vector, which
    /// equals cosine similarity for unit vectors.
    ///
    /// # Arguments
    /// * `query` - Query embedding vector
    /// * `k` - Number of top results to return (default 10)
    /// * `threshold` - Minimum similarity score to include (default 0.0)
    ///
    /// # Returns
    /// List of (label, similarity) tuples sorted by similarity descending
    #[pyo3(signature = (query, k = 10, threshold = 0.0))]
    pub fn query(&self, query: Vec<f32>, k: usize, threshold: f32) -> Vec<(String, f32)> {
        if self.n_vectors == 0 || query.len() != self.dim {
            return Vec::new();
        }

        // Normalize query
        let q_norm = simd_norm(&query);
        let q_normalized: Vec<f32> = query.iter().map(|&x| x / q_norm).collect();

        // Compute dot products against all stored vectors
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(self.n_vectors);
        for i in 0..self.n_vectors {
            let offset = i * self.dim;
            let vec_slice = &self.data[offset..offset + self.dim];
            let sim = simd_dot(&q_normalized, vec_slice);
            if sim >= threshold {
                scored.push((i, sim));
            }
        }

        let k = k.min(scored.len());
        if k == 0 {
            return Vec::new();
        }

        // Partially select the top-k instead of sorting every matching vector.
        scored.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut top_k: Vec<(usize, f32)> = scored.into_iter().take(k).collect();
        top_k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        top_k
            .into_iter()
            .map(|(idx, sim)| (self.labels[idx].clone(), sim))
            .collect()
    }

    /// Return similarity scores for every vector in the index.
    ///
    /// # Arguments
    /// * `query` - Query embedding vector
    ///
    /// # Returns
    /// List of similarity scores, one per stored vector, in insertion order
    ///
    /// # Raises
    /// * `ValueError` - If query dimension doesn't match index dimension
    pub fn similarities(&self, query: Vec<f32>) -> PyResult<Vec<f32>> {
        if self.n_vectors == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimension mismatch: index has {} dims, query has {} dims",
                self.dim,
                query.len()
            )));
        }

        let q_norm = simd_norm(&query);
        let q_normalized: Vec<f32> = query.iter().map(|&x| x / q_norm).collect();

        let mut results = Vec::with_capacity(self.n_vectors);
        for i in 0..self.n_vectors {
            let offset = i * self.dim;
            let vec_slice = &self.data[offset..offset + self.dim];
            results.push(simd_dot(&q_normalized, vec_slice));
        }

        Ok(results)
    }

    /// Return the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Return the dimensionality of vectors in the index.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod embedding_index_tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let idx = EmbeddingIndex::new(vec![], vec![]).unwrap();
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.dim(), 0);
    }

    #[test]
    fn test_new_stores_normalized() {
        let vectors = vec![vec![3.0, 4.0], vec![1.0, 0.0]];
        let labels = vec!["a".to_string(), "b".to_string()];
        let idx = EmbeddingIndex::new(vectors, labels).unwrap();

        assert_eq!(idx.len(), 2);
        assert_eq!(idx.dim(), 2);

        // First stored vector should be [0.6, 0.8]
        assert!((idx.data[0] - 0.6).abs() < 1e-6);
        assert!((idx.data[1] - 0.8).abs() < 1e-6);
        // Second stored vector should be [1.0, 0.0]
        assert!((idx.data[2] - 1.0).abs() < 1e-6);
        assert!((idx.data[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_new_dimension_mismatch() {
        let vectors = vec![vec![1.0, 2.0], vec![1.0, 2.0, 3.0]];
        let labels = vec!["a".to_string(), "b".to_string()];
        assert!(EmbeddingIndex::new(vectors, labels).is_err());
    }

    #[test]
    fn test_new_labels_mismatch() {
        let vectors = vec![vec![1.0, 2.0]];
        let labels = vec!["a".to_string(), "b".to_string()];
        assert!(EmbeddingIndex::new(vectors, labels).is_err());
    }

    #[test]
    fn test_add_vectors() {
        let mut idx = EmbeddingIndex::new(vec![vec![1.0, 0.0]], vec!["a".to_string()]).unwrap();
        assert_eq!(idx.len(), 1);

        idx.add(vec![vec![0.0, 1.0]], vec!["b".to_string()])
            .unwrap();
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.labels, vec!["a", "b"]);
    }

    #[test]
    fn test_add_to_empty_index() {
        let mut idx = EmbeddingIndex::new(vec![], vec![]).unwrap();
        idx.add(vec![vec![1.0, 0.0]], vec!["a".to_string()])
            .unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.dim(), 2);
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let mut idx = EmbeddingIndex::new(vec![vec![1.0, 0.0]], vec!["a".to_string()]).unwrap();
        assert!(idx
            .add(vec![vec![1.0, 0.0, 0.0]], vec!["b".to_string()])
            .is_err());
    }

    #[test]
    fn test_query_identical() {
        let idx = EmbeddingIndex::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec!["x-axis".to_string(), "y-axis".to_string()],
        )
        .unwrap();

        let results = idx.query(vec![1.0, 0.0], 1, 0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "x-axis");
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_query_with_threshold() {
        let idx = EmbeddingIndex::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, 0.7]],
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        )
        .unwrap();

        // Query close to a, threshold should filter out b
        let results = idx.query(vec![1.0, 0.0], 10, 0.5);
        // "a" similarity ~1.0, "c" similarity ~0.7, "b" similarity ~0.0
        assert!(results.len() == 2);
        assert_eq!(results[0].0, "a");
        assert_eq!(results[1].0, "c");
    }

    #[test]
    fn test_query_top_k_limit() {
        let idx = EmbeddingIndex::new(
            vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.8, 0.2]],
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        )
        .unwrap();

        let results = idx.query(vec![1.0, 0.0], 2, 0.0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_zero_k() {
        let idx = EmbeddingIndex::new(
            vec![vec![1.0, 0.0], vec![0.9, 0.1]],
            vec!["a".to_string(), "b".to_string()],
        )
        .unwrap();

        let results = idx.query(vec![1.0, 0.0], 0, 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_wrong_dimension() {
        let idx = EmbeddingIndex::new(vec![vec![1.0, 0.0]], vec!["a".to_string()]).unwrap();

        let results = idx.query(vec![1.0, 0.0, 0.0], 1, 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_similarities() {
        let idx = EmbeddingIndex::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]],
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        )
        .unwrap();

        let sims = idx.similarities(vec![1.0, 0.0]).unwrap();
        assert_eq!(sims.len(), 3);
        assert!((sims[0] - 1.0).abs() < 1e-6); // identical
        assert!(sims[1].abs() < 1e-6); // orthogonal
        assert!((sims[2] + 1.0).abs() < 1e-6); // opposite
    }

    #[test]
    fn test_similarities_wrong_dimension() {
        let idx = EmbeddingIndex::new(vec![vec![1.0, 0.0]], vec!["a".to_string()]).unwrap();
        assert!(idx.similarities(vec![1.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn test_query_empty_index() {
        let idx = EmbeddingIndex::new(vec![], vec![]).unwrap();
        let results = idx.query(vec![1.0, 0.0], 10, 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_similarities_empty_index() {
        let idx = EmbeddingIndex::new(vec![], vec![]).unwrap();
        let sims = idx.similarities(vec![1.0, 0.0]).unwrap();
        assert!(sims.is_empty());
    }
}
