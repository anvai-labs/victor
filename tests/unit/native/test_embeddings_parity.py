# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Native embeddings parity against the NumPy reference spec (FOU-C3).

The embeddings operations (``quantize_embedding`` and friends) ship only in
``victor_native`` — there is no in-repo Python fallback wrapper. The Python
reference here is therefore the SPEC implementation: the exact algorithm
documented in ``rust/crates/python-bindings/src/embeddings.rs`` written in
NumPy float32. Reference-implementation notes that matter for parity:

- symmetric int8 quantization: ``scale = abs_max / 127`` (f32), rounding is
  half-away-from-zero (Rust ``f32::round`` — NOT NumPy's banker's rounding),
- ``quantized_cosine_similarity`` computes the dot/norms in exact i32 then
  takes an f32 sqrt/divide,
- ``random_projection`` uses a documented LCG stream
  (``state = state*6364136223846793005 + 1 (mod 2^64)``, top 31 bits scaled)
  that the reference reproduces bit-exactly; only the matmul accumulation
  order (SIMD 8-lane vs BLAS) is covered by a tolerance.

Selected by the native-parity CI job (``pytest -m native_parity``), which
builds the Rust extension first; in ordinary unit runs these still execute
and skip themselves when the wheel is missing/stale.
"""

from __future__ import annotations

import math

import pytest

from victor.processing.native._base import _NATIVE_AVAILABLE, _native

pytestmark = pytest.mark.native_parity

_EMBEDDINGS_OPS = (
    "quantize_embedding",
    "batch_quantize_embeddings",
    "dequantize_embedding",
    "quantized_cosine_similarity",
    "batch_quantized_cosine_similarity",
    "matmul_vector",
    "batch_matmul_vector",
    "random_projection",
    "pairwise_distances",
    "knn_graph",
)

np = pytest.importorskip("numpy")


def _native_has_embeddings() -> bool:
    return _NATIVE_AVAILABLE and all(hasattr(_native, op) for op in _EMBEDDINGS_OPS)


_requires_native = pytest.mark.skipif(
    not _native_has_embeddings(),
    reason="native wheel missing embeddings ops — rebuild with maturin",
)

# Corpora: ordinary embeddings, boundary shapes, and exact .5 quantization
# points that would catch a banker's-rounding regression in the reference.
_VECTORS = [
    [0.1, -0.5, 0.3, 0.8, -0.2],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [-7.0, 3.5, 1.75],
    [0.0, 1.27, -2.54],  # x/scale lands on exact halves for the 1.27 entry
    [1e-7, -5e-8],  # sub-epsilon scale branch
]


def _ref_quantize(vec):
    """Spec reference for symmetric int8 quantization (see module docstring)."""
    v = np.asarray(vec, dtype=np.float32)
    if v.size == 0:
        return np.array([], dtype=np.int8), np.float32(1.0), 0
    abs_max = np.float32(max(abs(float(v.min())), abs(float(v.max()))))
    scale = np.float32(abs_max / np.float32(127.0))
    eps = np.float32(1e-9)
    if scale < eps:
        data = np.zeros(v.size, dtype=np.int8)
    else:
        q = v / scale
        rounded = np.sign(q) * np.floor(np.abs(q) + np.float32(0.5))
        data = np.clip(rounded, -128.0, 127.0).astype(np.int8)
    return data, np.float32(max(scale, float(eps))), 0


def _ref_dequantize(data, scale, zero_point=0):
    return (np.asarray(data, dtype=np.float32) - np.float32(zero_point)) * np.float32(scale)


def _ref_quantized_cosine(a, b) -> float:
    da = np.asarray(a.data, dtype=np.int64)
    db = np.asarray(b.data, dtype=np.int64)
    if da.size == 0:
        return 0.0
    dot = int(np.dot(da, db))
    norm_a = np.float32(math.sqrt(float(np.dot(da, da))))
    norm_b = np.float32(math.sqrt(float(np.dot(db, db))))
    if float(norm_a) < 1e-9 or float(norm_b) < 1e-9:
        return 0.0
    return float(np.float32(dot) / (norm_a * norm_b))


def _lcg_projection(target_dim: int, input_dim: int, seed: int) -> np.ndarray:
    """Bit-exact replication of the Rust LCG projection matrix."""
    state = seed
    scale = np.sqrt(np.float32(1.0) / np.float32(target_dim))
    out = np.empty((target_dim, input_dim), dtype=np.float32)
    mask = (1 << 64) - 1
    for r in range(target_dim):
        for c in range(input_dim):
            state = (state * 6364136223846793005 + 1) & mask
            r01 = np.float32(state >> 33) / np.float32(4294967295.0)
            out[r, c] = (r01 * np.float32(2.0) - np.float32(1.0)) * scale
    return out


def _random_corpus(n: int, dim: int, seed: int = 0):
    rs = np.random.RandomState(seed)
    return rs.rand(n, dim).astype(np.float32).tolist()


class TestQuantizationParity:
    @_requires_native
    @pytest.mark.parametrize("vector", _VECTORS)
    def test_quantize_matches_reference(self, vector):
        q = _native.quantize_embedding([float(x) for x in vector])
        ref_data, ref_scale, ref_zp = _ref_quantize(vector)
        assert list(q.data) == list(ref_data)
        assert q.scale == float(ref_scale)
        assert q.zero_point == ref_zp

    @_requires_native
    def test_empty_vector_quantizes_to_empty(self):
        q = _native.quantize_embedding([])
        assert q.data == []
        assert q.scale == 1.0

    @_requires_native
    @pytest.mark.parametrize("vector", _VECTORS)
    def test_dequantize_matches_reference(self, vector):
        q = _native.quantize_embedding([float(x) for x in vector])
        ref_data, ref_scale, ref_zp = _ref_quantize(vector)
        assert np.allclose(
            _native.dequantize_embedding(q),
            _ref_dequantize(ref_data, ref_scale, ref_zp),
            atol=1e-9,
        )

    @_requires_native
    @pytest.mark.parametrize("vector", _VECTORS)
    def test_roundtrip_error_bounded(self, vector):
        """int8 quantization error stays within half a quantization step."""
        original = np.asarray(vector, dtype=np.float32)
        q = _native.quantize_embedding([float(x) for x in vector])
        restored = np.asarray(_native.dequantize_embedding(q), dtype=np.float32)
        step = float(q.scale)
        assert np.max(np.abs(original - restored)) <= step / 2.0 + 1e-6

    @_requires_native
    def test_batch_quantize_matches_singles(self):
        corpus = [[float(x) for x in v] for v in _VECTORS]
        batch = _native.batch_quantize_embeddings(corpus)
        singles = [_native.quantize_embedding(v) for v in corpus]
        assert len(batch) == len(singles)
        for qb, qs in zip(batch, singles):
            assert list(qb.data) == list(qs.data)
            assert qb.scale == qs.scale
            assert qb.zero_point == qs.zero_point


class TestQuantizedCosineParity:
    # Same-dimension corpus (4-dim) so cross-pair comparison is defined.
    COSINE_CORPUS = [
        [0.1, 0.2, 0.3, 0.4],
        [0.1, -0.2, 0.3, -0.4],
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

    @_requires_native
    @pytest.mark.parametrize("i,j", [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (2, 3), (0, 4)])
    def test_quantized_cosine_matches_reference(self, i, j):
        quantized = [_native.quantize_embedding([float(x) for x in v]) for v in self.COSINE_CORPUS]
        a, b = quantized[i], quantized[j]
        assert _native.quantized_cosine_similarity(a, b) == pytest.approx(
            _ref_quantized_cosine(a, b), abs=1e-6
        )

    @_requires_native
    def test_zero_vector_scores_zero(self):
        a = _native.quantize_embedding([1.0, 2.0, 3.0])
        z = _native.quantize_embedding([0.0, 0.0, 0.0])
        assert _native.quantized_cosine_similarity(a, z) == 0.0
        assert _native.quantized_cosine_similarity(z, a) == 0.0

    @_requires_native
    def test_dimension_mismatch_raises(self):
        a = _native.quantize_embedding([1.0, 2.0])
        b = _native.quantize_embedding([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            _native.quantized_cosine_similarity(a, b)

    @_requires_native
    def test_batch_cosine_matches_singles(self):
        quantized = [_native.quantize_embedding([float(x) for x in v]) for v in self.COSINE_CORPUS]
        query = quantized[0]
        batch = _native.batch_quantized_cosine_similarity(query, quantized)
        singles = [_native.quantized_cosine_similarity(query, q) for q in quantized]
        assert batch == pytest.approx(singles, abs=1e-6)

    @_requires_native
    def test_batch_cosine_empty_corpus(self):
        query = _native.quantize_embedding([1.0, 2.0])
        assert _native.batch_quantized_cosine_similarity(query, []) == []


class TestMatrixOpsParity:
    @_requires_native
    def test_matmul_matches_numpy_reference(self):
        matrix = _random_corpus(6, 16, seed=1)
        vector = _random_corpus(1, 16, seed=2)[0]
        result = np.asarray(_native.matmul_vector(matrix, vector), dtype=np.float32)
        ref = np.asarray(matrix, dtype=np.float32) @ np.asarray(vector, dtype=np.float32)
        np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)

    @_requires_native
    def test_matmul_ragged_row_raises(self):
        with pytest.raises(ValueError, match="cols"):
            _native.matmul_vector([[1.0, 2.0], [1.0]], [1.0, 2.0])

    @_requires_native
    def test_batch_matmul_matches_per_vector_matmul(self):
        matrix = _random_corpus(5, 8, seed=3)
        vectors = _random_corpus(4, 8, seed=4)
        batch = _native.batch_matmul_vector(matrix, vectors)
        assert len(batch) == len(vectors)
        for row, v in zip(batch, vectors):
            single = _native.matmul_vector(matrix, v)
            np.testing.assert_allclose(
                np.asarray(row, dtype=np.float32),
                np.asarray(single, dtype=np.float32),
                rtol=1e-6,
                atol=1e-7,
            )

    @_requires_native
    def test_batch_matmul_dim_mismatch_raises(self):
        matrix = _random_corpus(3, 4, seed=5)
        with pytest.raises(ValueError):
            _native.batch_matmul_vector(matrix, [[1.0, 2.0]])


class TestRandomProjectionParity:
    @_requires_native
    def test_matches_lcg_reference(self):
        """The projection is the documented LCG stream times the input — the
        replicated reference agrees within matmul accumulation error."""
        embeddings = _random_corpus(6, 16, seed=6)
        target_dim, seed = 8, 42
        projected = np.asarray(
            _native.random_projection(embeddings, target_dim, seed), dtype=np.float32
        )
        projection = _lcg_projection(target_dim, 16, seed)
        ref = np.asarray(embeddings, dtype=np.float32) @ projection.T
        np.testing.assert_allclose(projected, ref, rtol=1e-4, atol=1e-5)

    @_requires_native
    def test_deterministic_for_same_seed(self):
        embeddings = _random_corpus(4, 8, seed=7)
        first = _native.random_projection(embeddings, 4, 42)
        second = _native.random_projection(embeddings, 4, 42)
        assert first == second

    @_requires_native
    def test_seed_sensitive(self):
        embeddings = _random_corpus(4, 8, seed=8)
        assert _native.random_projection(embeddings, 4, 42) != _native.random_projection(
            embeddings, 4, 43
        )

    @_requires_native
    def test_target_dim_above_input_raises(self):
        with pytest.raises(ValueError, match="Target dimension"):
            _native.random_projection(_random_corpus(2, 4, seed=9), 8, 42)

    @_requires_native
    def test_empty_input_returns_empty(self):
        assert _native.random_projection([], 4, 42) == []


class TestDistanceParity:
    @_requires_native
    def test_pairwise_distances_matches_reference(self):
        """Upper-triangular flattened 1 - cosine over L2-normalized rows."""
        embeddings = _random_corpus(7, 12, seed=10)
        distances = np.asarray(_native.pairwise_distances(embeddings), dtype=np.float32)
        matrix = np.asarray(embeddings, dtype=np.float32)
        norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        normalized = matrix / norms
        ref = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                ref.append(1.0 - float(np.dot(normalized[i], normalized[j])))
        assert distances.shape == (len(ref),)
        np.testing.assert_allclose(distances, np.asarray(ref), rtol=1e-4, atol=1e-5)

    @_requires_native
    @pytest.mark.parametrize("count", [0, 1])
    def test_pairwise_degenerate_inputs_empty(self, count):
        assert _native.pairwise_distances(_random_corpus(count, 4, seed=11)) == []

    @_requires_native
    def test_knn_graph_matches_reference(self):
        embeddings = _random_corpus(8, 12, seed=12)
        k = 3
        knn = _native.knn_graph(embeddings, k)
        matrix = np.asarray(embeddings, dtype=np.float32)
        norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        normalized = matrix / norms
        sims = normalized @ normalized.T
        for i, (neighbors, distances) in enumerate(knn):
            order = np.argsort(-sims[i])[1:]  # self excluded
            assert list(neighbors) == list(order[:k])
            ref_distances = 1.0 - sims[i][order[:k]]
            np.testing.assert_allclose(
                np.asarray(distances, dtype=np.float32),
                np.asarray(ref_distances, dtype=np.float32),
                rtol=1e-4,
                atol=1e-5,
            )

    @_requires_native
    def test_knn_k_clamped_to_corpus(self):
        embeddings = _random_corpus(3, 6, seed=13)
        neighbors, _ = _native.knn_graph(embeddings, 10)[0]
        assert len(neighbors) == 2  # n - 1

    @_requires_native
    def test_knn_empty_corpus(self):
        assert _native.knn_graph([], 3) == []
