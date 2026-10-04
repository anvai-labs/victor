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

"""InMemoryVectorStore: numpy-optional import safety and pure-math fallback.

Mirrored location for scripts/ci/select_changed_tests.py: changed source
``victor/contrib/vectorstores/store.py`` maps here (stem ``store``).
"""

from __future__ import annotations

import asyncio
import importlib
import sys

import pytest


def _poison_numpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)


def _fresh_store(monkeypatch):
    """Import the store with numpy poisoned, returning a fresh instance."""
    monkeypatch.delitem(sys.modules, "victor.contrib.vectorstores.store", raising=False)
    monkeypatch.delitem(sys.modules, "victor.contrib.vectorstores", raising=False)
    module = importlib.import_module("victor.contrib.vectorstores.store")
    return module, module.InMemoryVectorStore()


def test_module_import_survives_without_numpy(monkeypatch):
    """Reload with numpy poisoned takes the ImportError guard branch."""
    _poison_numpy(monkeypatch)
    module = importlib.import_module("victor.contrib.vectorstores.store")
    importlib.reload(module)
    assert module.NUMPY_AVAILABLE is False
    assert module.np is None


def test_pure_cosine_path_without_numpy(monkeypatch):
    """add/search work with numpy poisoned (lists + stdlib math fallback)."""
    _poison_numpy(monkeypatch)
    module, store = _fresh_store(monkeypatch)
    assert module.NUMPY_AVAILABLE is False

    async def run():
        ids = await store.add_documents(["alpha", "beta"], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        results = await store.search([1.0, 0.0, 1.0], top_k=2)
        return ids, results

    ids, results = asyncio.run(run())
    assert results, "pure-path search returned nothing"
    assert results[0].document_id == ids[0]  # "alpha" ranks first for its own vector
    score = store._cosine_similarity([1.0, 0.0], [0.0, 0.0])
    assert score == 0.0  # zero-norm guard, exact


def test_numpy_path_agrees_with_pure_path(monkeypatch):
    np = pytest.importorskip("numpy")

    async def build(module):
        store = module.InMemoryVectorStore()
        ids = await store.add_documents(["alpha", "beta"], [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        results = await store.search([1.0, 0.0, 1.0], top_k=2)
        rank = [r.document_id for r in results]
        return rank, ids, results

    numpy_module = importlib.import_module("victor.contrib.vectorstores.store")
    numpy_rank, numpy_ids, numpy_results = asyncio.run(build(numpy_module))

    _poison_numpy(monkeypatch)
    monkeypatch.delitem(sys.modules, "victor.contrib.vectorstores.store", raising=False)
    monkeypatch.delitem(sys.modules, "victor.contrib.vectorstores", raising=False)
    pure_module = importlib.import_module("victor.contrib.vectorstores.store")
    pure_rank, pure_ids, pure_results = asyncio.run(build(pure_module))

    # Same ranking: each build's top results are its own doc0 then doc1
    # (uuid-minted ids differ between builds, positions must not).
    assert numpy_rank == numpy_ids
    assert pure_rank == pure_ids
    for r_p, r_n in zip(pure_results, numpy_results):
        assert r_p.score == pytest.approx(float(r_n.score), abs=1e-6)
    assert np is not None  # keep the importorskip binding used above
