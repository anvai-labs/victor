# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Durability contract: an indexed graph and its vectors must survive a restart.

Both backends persist to disk, so "index it, close it, open it again, it is still
there" is the minimum either has to satisfy. Nothing asserted that before, and two
defects hid in the gap:

- ProximaRecord writes — which hold Tier-A node properties *and* the embedding —
  report success, are readable in the same session, and are gone after a restart.
- ``victor init`` disables embeddings outright, so neither backend had vectors at
  all and the missing durability was invisible.

The equivalent close/reopen assertion already exists one tier down, for the CPG
fragment store (``tests/unit/storage/graph/test_cpg_fragment_store.py``). These
tests apply it to Tier-A and to the vector path.

**Never assert through ``stats()`` or ``count()``.** Both are stubs that answer 0
regardless of contents, which is precisely what masked the record bug — a
collection reporting ``record_count: 0`` looked identical whether it held 200 rows
or none. Every assertion here reads rows back through a real retrieval path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import pytest

try:  # importorskip only catches ImportError; a broken install raises RuntimeError
    import proximadb_sdk  # noqa: F401

    _PROXIMA_IMPORTABLE = True
except Exception:  # pragma: no cover - environment-dependent
    _PROXIMA_IMPORTABLE = False

from victor.core.graph_rag.config import GraphIndexConfig  # noqa: E402
from victor.core.graph_rag.indexing import GraphIndexingPipeline  # noqa: E402
from victor.storage.graph.registry import create_graph_store  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Proxima is expected to fail both assertions today: ORION drops edges across a
# restart, and the ProximaRecords holding the vectors are not persisted at all.
# Filed upstream. Marked xfail(strict=True) rather than skipped so the day the
# engine fixes it these turn XPASS and fail the build, prompting the marker's
# removal — a skip would just go quiet and we would never notice the fix.
_PROXIMA_NOT_DURABLE = pytest.mark.xfail(
    strict=True,
    reason=(
        "ProximaDB does not persist ORION edges or ProximaRecords across a "
        "restart; see anvai-labs/proximaDB#1524"
    ),
)

BACKENDS = [
    "sqlite",
    pytest.param("proxima", marks=_PROXIMA_NOT_DURABLE),
]

_SOURCE = '''
"""Fixture module."""


class StorageEngine:
    """Persists things."""

    def write(self, key, value):
        return self._encode(key, value)

    def _encode(self, key, value):
        return f"{key}={value}"


def open_engine():
    return StorageEngine()
'''


def _make_corpus(root: Path) -> Path:
    """A corpus small enough to index quickly but real enough to produce symbols."""
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "engine.py").write_text(_SOURCE, encoding="utf-8")
    (src / "helper.py").write_text(
        "from .engine import open_engine\n\n\ndef run():\n    return open_engine()\n",
        encoding="utf-8",
    )
    return root


def _skip_if_backend_unavailable(backend: str) -> None:
    if backend == "proxima" and not _PROXIMA_IMPORTABLE:
        pytest.skip("proximadb_sdk unavailable")


async def _index_once(backend: str, project: Path, corpus: Path, *, embeddings: bool) -> None:
    """Index the corpus, then close the store so buffers must reach disk."""
    store = create_graph_store(backend, project_path=project)
    config = GraphIndexConfig(
        root_path=corpus,
        enable_ccg=False,  # Tier-A only; Tier-B durability is covered elsewhere
        enable_embeddings=embeddings,
        enable_subgraph_cache=False,
        incremental=False,
    )
    try:
        await GraphIndexingPipeline(store, config).index_repository(root_path=corpus)
    finally:
        await store.close()


async def _reopen(backend: str, project: Path) -> Tuple[Any, List[Any], List[Any]]:
    """Open a fresh store over the same directory and read the graph back."""
    store = create_graph_store(backend, project_path=project)
    await store.initialize()
    return store, await store.get_all_nodes(), await store.get_all_edges()


async def _vector_hits(store: Any) -> int:
    """Count hits from the backend's own vector-retrieval path.

    The two backends expose different surfaces — which is the whole point of the
    correlated-store design — so this bridges them:

    * Proxima co-locates vectors with graph nodes and takes a query **vector**.
    * SQLite keeps them in LanceDB behind an EmbeddingProvider that takes **text**
      and embeds it internally.

    Any query returns nearest neighbours when the index is non-empty, so a non-zero
    count is exactly the "vectors survived" signal — no need to reconstruct the
    exact vector that was written.
    """
    semantic = getattr(store, "semantic_search", None)
    if semantic is not None:
        probe = [0.05] * 384
        return len(await semantic(probe, top_k=5))

    build_config = getattr(store, "_build_embedding_config", None)
    if build_config is None:
        pytest.skip(f"{type(store).__name__} exposes no vector retrieval path")

    from victor.storage.vector_stores.registry import EmbeddingRegistry

    provider = EmbeddingRegistry.create(build_config())
    await provider.initialize()
    try:
        return len(await provider.search_similar("storage engine", limit=5))
    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            await close()


@pytest.mark.parametrize("backend", BACKENDS)
async def test_graph_survives_restart(backend: str, tmp_path: Path) -> None:
    """Nodes and edges written by one process must be readable by the next."""
    _skip_if_backend_unavailable(backend)
    corpus = _make_corpus(tmp_path / "repo")
    project = tmp_path / "project"
    project.mkdir()

    await _index_once(backend, project, corpus, embeddings=False)

    store, nodes, edges = await _reopen(backend, project)
    try:
        assert nodes, f"{backend}: no nodes survived the restart"
        assert any(n.type in ("function", "class", "method") for n in nodes), (
            f"{backend}: symbols were lost across the restart; got "
            f"{sorted({n.type for n in nodes})}"
        )
        assert edges, f"{backend}: no edges survived the restart"
    finally:
        await store.close()


@pytest.mark.parametrize("backend", BACKENDS)
async def test_vectors_survive_restart(backend: str, tmp_path: Path) -> None:
    """Embeddings must outlive the process that produced them.

    Proxima keeps them inside ProximaRecords, SQLite in LanceDB. Either way a
    semantic query after reopening must still find something — otherwise the agent
    silently loses code search on the next run, with no error to notice.
    """
    _skip_if_backend_unavailable(backend)
    corpus = _make_corpus(tmp_path / "repo")
    project = tmp_path / "project"
    project.mkdir()

    await _index_once(backend, project, corpus, embeddings=True)

    store, nodes, _ = await _reopen(backend, project)
    try:
        assert nodes, f"{backend}: graph itself did not survive; vector check is moot"
        hits = await _vector_hits(store)
        assert hits > 0, (
            f"{backend}: no vectors survived the restart. The write reported success "
            f"and was readable in-session, so this is a durability failure, not a "
            f"write failure."
        )
    finally:
        await store.close()
