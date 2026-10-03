# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Dispatch pressure tests for victor/processing/native/similarity.py.

numpy is an optional accelerator, never a requirement: with numpy hidden
(probe + lazy import patched), every similarity function still returns correct
values — via the native engine when present, else pure-Python vecmath — and
numpy-present environments keep their BLAS batch path.

Mirrored location for scripts/ci/select_changed_tests.py: changed source
``victor/processing/native/similarity.py`` maps here.
"""

from __future__ import annotations

import pytest

from victor.core.vecmath import batch_cosine as vec_batch_cosine
from victor.core.vecmath import normalize_rows as vec_normalize_rows
from victor.processing.native import similarity
from victor.processing.native import _base as native_base

QUERY = [1.0, 0.0, 1.0, 0.5]
CORPUS = [
    [1.0, 0.0, 1.0, 0.5],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
]


@pytest.fixture(autouse=True)
def _reset_numpy_probe():
    native_base.reset_numpy_probe_for_tests()
    yield
    native_base.reset_numpy_probe_for_tests()


def hide_numpy(monkeypatch):
    """Simulate a bare install: probe says absent, lazy import returns None.

    similarity.py imports _NATIVE_AVAILABLE/_native BY VALUE at module import,
    so its own bindings must be patched too — _base's globals don't propagate.
    """
    monkeypatch.setattr(native_base, "numpy_accelerator_available", lambda: False)
    monkeypatch.setattr(similarity, "_ensure_numpy", lambda: None)
    monkeypatch.setattr(similarity, "_NATIVE_AVAILABLE", False)


def test_numpy_present_uses_numpy_batch_path(monkeypatch):
    """numpy-installed environments keep the BLAS batch path (no regression)."""
    if native_base.numpy_accelerator_available() is False:
        pytest.skip("numpy not installed in this environment")
    called = {}
    real = similarity._batch_cosine_numpy

    def spy(np, query, corpus):
        called["hit"] = True
        return real(np, query, corpus)

    monkeypatch.setattr(similarity, "_batch_cosine_numpy", spy)
    result = similarity.batch_cosine_similarity(QUERY, CORPUS)
    assert called.get("hit") is True
    assert len(result) == len(CORPUS)
    assert result[0] == pytest.approx(1.0, abs=1e-6)


def test_numpy_absent_batch_dispatches_to_native(monkeypatch):
    if not native_base.is_native_available():
        pytest.skip("victor_native not installed in this environment")
    hide_numpy(monkeypatch)
    called = {}

    def spy(query, corpus):
        called["hit"] = True
        return [0.0] * len(corpus)

    # The no-numpy policy must route the batch op to the native engine.
    monkeypatch.setattr(native_base, "_NATIVE_AVAILABLE", True)
    monkeypatch.setattr(similarity, "_NATIVE_AVAILABLE", True)
    monkeypatch.setattr(
        similarity, "_native", type("_N", (), {"batch_cosine_similarity": staticmethod(spy)})
    )
    result = similarity.batch_cosine_similarity(QUERY, CORPUS)
    assert called.get("hit") is True
    assert result == [0.0] * len(CORPUS)


def test_numpy_absent_no_native_batch_uses_pure(monkeypatch):
    hide_numpy(monkeypatch)
    monkeypatch.setattr(native_base, "_NATIVE_AVAILABLE", False)

    result = similarity.batch_cosine_similarity(QUERY, CORPUS)

    assert result == vec_batch_cosine(QUERY, CORPUS)
    # eps-damped float64 norms give 1 - ~2e-9, not exactly 1.0 (float32
    # numpy hides the eps; the pure path keeps it).
    assert result[0] == pytest.approx(1.0, abs=1e-6)
    # Zero vector: eps-damped norm must give 0.0, not NaN.
    assert result[3] == 0.0


def test_numpy_absent_single_pair_uses_pure(monkeypatch):
    hide_numpy(monkeypatch)
    monkeypatch.setattr(native_base, "_NATIVE_AVAILABLE", False)

    assert similarity.cosine_similarity(QUERY, QUERY) == pytest.approx(1.0, abs=1e-6)
    assert similarity.cosine_similarity(QUERY, [0.0] * len(QUERY)) == 0.0


def test_numpy_absent_normalized_variants_use_pure(monkeypatch):
    hide_numpy(monkeypatch)
    monkeypatch.setattr(native_base, "_NATIVE_AVAILABLE", False)

    normalized = vec_normalize_rows(CORPUS)
    assert similarity.batch_normalize_vectors(CORPUS) == normalized
    scores = similarity.batch_cosine_similarity_normalized(QUERY, normalized)
    assert scores == pytest.approx(vec_batch_cosine(QUERY, CORPUS), rel=1e-9)
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    top = similarity.top_k_similar_normalized(QUERY, normalized, k=2)
    assert [i for i, _ in top] == [0, 2]


def test_pure_matches_numpy_values(monkeypatch):
    """vecmath must agree with the numpy formulation it replaced (float64 vs
    float32 differs only in low-order bits)."""
    np = pytest.importorskip("numpy")
    hide_numpy(monkeypatch)
    monkeypatch.setattr(native_base, "_NATIVE_AVAILABLE", False)

    rng = np.random.default_rng(42)
    query = rng.standard_normal(384).tolist()
    corpus = rng.standard_normal((17, 384)).tolist()

    pure = similarity.batch_cosine_similarity(query, corpus)

    q = np.array(query, dtype=np.float32)
    c = np.array(corpus, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-9)
    cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    expected = np.dot(cn, qn).tolist()

    for got, want in zip(pure, expected):
        # float32 (numpy) vs float64 (pure) plus near-zero values: rel alone is
        # too tight, so floor the tolerance absolutely.
        assert got == pytest.approx(want, rel=1e-4, abs=1e-6)


def test_error_semantics_preserved(monkeypatch):
    hide_numpy(monkeypatch)
    monkeypatch.setattr(native_base, "_NATIVE_AVAILABLE", False)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        similarity.batch_cosine_similarity(QUERY, [[1.0, 2.0]])
    with pytest.raises(ValueError, match="same length"):
        similarity.cosine_similarity([1.0], [1.0, 2.0])
    assert similarity.batch_cosine_similarity(QUERY, []) == []
    assert similarity.batch_normalize_vectors([]) == []
