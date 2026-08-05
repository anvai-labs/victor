# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Embedded ProximaDB parity test (TD-11/12/13) — the real verification gate.

Runs ``ProximaGraphStore`` against a **real** embedded ProximaDB (in-RAM vectors,
the ``EmbeddingMode::Memory`` equivalent) and asserts impact_analysis
(forward/backward) and hybrid seed→expand match ``SqliteGraphStore`` on known
symbols, per ``docs/architecture/proximadb-codegraph-backend.md``.

This test **skips** when the embedded server binary is unavailable (e.g. the
proximadb-server binary has not been built). The fast adapter-level parity that
always runs lives in ``tests/unit/storage/graph/test_proxima_store_parity.py``.

The multi-tenant **service** mode (``server_url=``) is intentionally not covered
here — it is WIP, gated on ProximaDB TD-127 (secondary indexes) + TD-130/131
(graph bulk-load + REST v2 hybrid).
"""

from __future__ import annotations

from typing import List

import pytest

try:  # importorskip only catches ImportError; a broken install (e.g. grpc codegen
    # mismatch) raises RuntimeError and would fail collection for the whole suite.
    import proximadb_sdk  # noqa: F401
except Exception as _exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"proximadb_sdk unavailable: {_exc}", allow_module_level=True)

from victor.storage.graph import GraphEdge, GraphNode, SqliteGraphStore  # noqa: E402
from victor.storage.graph.proxima_store import ProximaGraphStore  # noqa: E402
from victor.storage.proxima_runtime import (  # noqa: E402
    ProximaUnavailableError,
    start_embedded_db,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _nodes() -> List[GraphNode]:
    return [
        GraphNode(node_id="n:main", type="function", name="main", file="a.py", line=1),
        GraphNode(node_id="n:parse", type="function", name="parse", file="a.py", line=10),
        GraphNode(node_id="n:validate", type="function", name="validate", file="b.py", line=5),
        GraphNode(node_id="n:helper", type="function", name="helper", file="b.py", line=20),
    ]


def _edges() -> List[GraphEdge]:
    return [
        GraphEdge(src="n:main", dst="n:parse", type="CALLS"),
        GraphEdge(src="n:main", dst="n:validate", type="CALLS"),
        GraphEdge(src="n:parse", dst="n:helper", type="CALLS"),
        GraphEdge(src="n:validate", dst="n:helper", type="CALLS"),
    ]


def _edge_keys(edges: List[GraphEdge]) -> set:
    return {(e.src, e.dst, e.type) for e in edges}


@pytest.fixture
async def embedded_proxima_store(tmp_path):
    """Start a real embedded ProximaDB store, or skip if unavailable."""
    try:
        db = await start_embedded_db(tmp_path / "proximadb")
    except ProximaUnavailableError as exc:
        pytest.skip(f"Embedded ProximaDB unavailable: {exc}")

    from proximadb_sdk.graph import ProximaDBGraph
    from proximadb_sdk.unified_client import ProximaDBClient

    client = ProximaDBClient(url=db.rest_url, protocol="rest")
    try:
        client.create_graph("fixture_codegraph")
    except Exception:
        pass
    graph = ProximaDBGraph(client, "fixture_codegraph")
    store = ProximaGraphStore(graph=graph, client=client, repo="fixture")
    await store.upsert_nodes(_nodes())
    await store.upsert_edges(_edges())
    try:
        yield store
    finally:
        await db.stop()


async def test_embedded_impact_and_hybrid_parity(tmp_path, embedded_proxima_store):
    sqlite = SqliteGraphStore(project_path=tmp_path / "sqlite")
    await sqlite.initialize()
    await sqlite.upsert_nodes(_nodes())
    await sqlite.upsert_edges(_edges())
    proxima = embedded_proxima_store
    try:
        for target in ("n:main", "n:helper"):
            for direction in ("in", "out"):
                s = await sqlite.get_neighbors(target, direction=direction, max_depth=3)
                p = await proxima.get_neighbors(target, direction=direction, max_depth=3)
                assert _edge_keys(s) == _edge_keys(p), (target, direction)

        s_exp = await sqlite.multi_hop_traverse_parallel(["n:main"], max_hops=3)
        p_exp = await proxima.multi_hop_traverse_parallel(["n:main"], max_hops=3)
        assert {n.node_id for n in s_exp.nodes} == {n.node_id for n in p_exp.nodes}
        assert _edge_keys(s_exp.edges) == _edge_keys(p_exp.edges)

        # Exercise the v2 query/nodes envelope path (find_nodes/get_nodes_by_file).
        assert {n.node_id for n in await proxima.find_nodes(name="helper")} == {"n:helper"}
        assert {n.node_id for n in await proxima.get_nodes_by_file("b.py")} == {
            "n:validate",
            "n:helper",
        }
    finally:
        await sqlite.close()
