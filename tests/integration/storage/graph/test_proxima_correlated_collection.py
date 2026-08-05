# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""One correlated collection: vector↔graph share one oid + one instance (TD-11/12).

Verifies the unified-substrate work: a symbol's ORION graph node and its HNSW
vector live in ONE embedded ProximaDB instance under a single ``oid``, so a
vector hit resolves to its graph node by identity (no cross-store join), and the
graph store + embedding provider share one ref-counted instance per repo.

Skips when the embedded ``proximadb-server`` binary is unavailable.
"""

from __future__ import annotations

import pytest

try:  # importorskip only catches ImportError; a broken install (e.g. grpc codegen
    # mismatch) raises RuntimeError and would fail collection for the whole suite.
    import proximadb_sdk  # noqa: F401
except Exception as _exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"proximadb_sdk unavailable: {_exc}", allow_module_level=True)

from victor.storage.graph import GraphNode  # noqa: E402
from victor.storage.graph.proxima_store import ProximaGraphStore  # noqa: E402
from victor.storage.proxima_runtime import (  # noqa: E402
    ProximaRepoConnection,
    ProximaUnavailableError,
    start_embedded_db,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


async def _require_embedded(data_dir):
    """Skip the test cleanly if an embedded instance cannot start here."""
    try:
        db = await start_embedded_db(data_dir)
    except ProximaUnavailableError as exc:
        pytest.skip(f"Embedded ProximaDB unavailable: {exc}")
    await db.stop()


async def _require_vector_path(store):
    """Skip until the embedded vector CRUD (create/insert/search) is functional.

    The embedded vector path is gated on ProximaDB-side boundary work (v2 embedded
    collection registration + Arrow Flight transport) — see
    ``roadmap/handoff/EMBEDDED_CODEGRAPH_BOUNDARY_HANDOFF.md`` in the proximaDB repo.
    These assertions auto-enable once that lands; until then they skip rather than
    fail (the graph path and the shared-instance refcount test stay active).
    """
    if store._conn is None:  # injected/service mode — no embedded vectors
        pytest.skip("no embedded vector connection")
    try:
        probe = await store._conn.get_or_create_collection("__probe__", dimension=4)
        await probe.insert_records([{"id": "p", "vector": [1.0, 0.0, 0.0, 0.0]}])
        hits = await probe.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    except Exception as exc:  # noqa: BLE001 - any drift means "not ready yet"
        pytest.skip(f"embedded vector path not functional yet (see handoff): {exc}")
    if not hits:
        pytest.skip("embedded vector path returns no hits — boundary work pending (handoff)")


async def test_connection_is_shared_and_refcounted(tmp_path):
    """Two acquires of the same data_dir return ONE instance; refcount governs stop."""
    data_dir = tmp_path / "proximadb"
    await _require_embedded(data_dir)

    a = await ProximaRepoConnection.acquire(data_dir)
    b = await ProximaRepoConnection.acquire(data_dir)
    try:
        assert a is b  # same shared connection object
        assert a.embedded_db is b.embedded_db  # one subprocess
        assert a.embedded_db is not None
    finally:
        await a.release()
        # One holder remains → instance still alive.
        assert b.embedded_db is not None
        await b.release()
        # Last holder released → stopped and unregistered.
        assert b.embedded_db is None


async def test_oid_bridges_vector_to_graph(tmp_path):
    """Atomic record vector → graph projection by identity (one oid)."""
    data_dir = tmp_path / "proximadb"
    await _require_embedded(data_dir)

    store = ProximaGraphStore(repo="fix", data_dir=data_dir)
    await store.initialize()
    await _require_vector_path(store)
    try:
        await store.upsert_nodes(
            [
                GraphNode(node_id="o:alpha", type="function", name="alpha", file="a.py"),
                GraphNode(node_id="o:beta", type="function", name="beta", file="b.py"),
                GraphNode(node_id="o:gamma", type="function", name="gamma", file="c.py"),
            ]
        )
        # Atomically replace each pending record with graph props + a 384-d vector.
        alpha = [1.0, *([0.0] * 383)]
        beta = [0.0, 1.0, *([0.0] * 382)]
        gamma = [0.0, 0.0, 1.0, *([0.0] * 381)]
        await store.upsert_node_record("o:alpha", alpha)
        await store.upsert_node_record("o:beta", beta)
        await store.upsert_node_record("o:gamma", gamma)

        # A query near alpha's vector must resolve to alpha's GRAPH node by oid.
        near_alpha = [0.9, 0.1, *([0.0] * 382)]
        hits = await store.semantic_search(near_alpha, top_k=1)
        assert [n.node_id for n in hits] == ["o:alpha"]
        assert hits[0].name == "alpha"  # vector hit → full graph node, not a join

        # And near beta → beta.
        near_beta = [0.05, 0.95, *([0.0] * 382)]
        hits = await store.semantic_search(near_beta, top_k=1)
        assert [n.node_id for n in hits] == ["o:beta"]

        # Deleting the node drops the co-indexed vector too (one entity).
        await store.delete_by_file("a.py")
        remaining = await store.semantic_search(near_alpha, top_k=3)
        assert "o:alpha" not in {n.node_id for n in remaining}
    finally:
        await store.close()


async def test_semantic_search_empty_without_vectors(tmp_path):
    """No co-indexed vectors → semantic_search returns [] (no crash)."""
    data_dir = tmp_path / "proximadb"
    await _require_embedded(data_dir)
    store = ProximaGraphStore(repo="fix", data_dir=data_dir)
    await store.initialize()
    await _require_vector_path(store)
    try:
        assert await store.semantic_search([1.0, 0.0, 0.0, 0.0], top_k=5) == []
    finally:
        await store.close()
