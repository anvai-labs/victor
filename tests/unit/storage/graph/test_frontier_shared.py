"""Shared frontier-SELECT behavior across the SQL graph stores (round 7).

The one-hop frontier extraction (`_frontier.select_frontier_edges`) unified
byte-identical copies in the sqlite and duckdb stores; the duckdb frontier
path had NO behavioral test anywhere (optional dep, registry tests are
signature-only). This pins both delegates through each store's public
`get_neighbors`, including the metadata-decoder seam (stdlib json for
sqlite, orjson for duckdb) and the edge-type filter.
"""

from __future__ import annotations

import pytest

from victor.storage.graph.protocol import GraphEdge, GraphNode


def _nodes():
    return [
        GraphNode(node_id="a", type="function", name="a", file="m.py"),
        GraphNode(node_id="b", type="function", name="b", file="m.py"),
        GraphNode(node_id="c", type="function", name="c", file="m.py"),
    ]


def _edges():
    return [
        GraphEdge(src="a", dst="b", type="CALLS", weight=0.5, metadata={"k": 1}),
        GraphEdge(src="a", dst="c", type="CONTAINS", weight=0.2, metadata={}),
    ]


async def _assert_shared_behavior(store):
    await store.initialize()
    try:
        await store.upsert_nodes(_nodes())
        await store.upsert_edges(_edges())

        both = await store.get_neighbors("a", direction="out", max_depth=1)
        pairs = sorted((e.type, e.dst) for e in both)
        assert pairs == [("CALLS", "b"), ("CONTAINS", "c")]

        # Metadata survives the store-specific decoder.
        calls = next(e for e in both if e.type == "CALLS")
        assert calls.metadata == {"k": 1}

        # Edge-type filter flows through the shared type clause.
        calls_only = await store.get_neighbors("a", edge_types=["CALLS"], direction="out")
        assert [(e.type, e.dst) for e in calls_only] == [("CALLS", "b")]

        # Direction="in" exercises the dst frontier side.
        inbound = await store.get_neighbors("b", direction="in", max_depth=1)
        assert [(e.src, e.dst) for e in inbound] == [("a", "b")]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_frontier_shared_behavior(tmp_path):
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    store = SqliteGraphStore(tmp_path)
    await _assert_shared_behavior(store)


@pytest.mark.asyncio
async def test_duckdb_frontier_shared_behavior(tmp_path):
    duckdb = pytest.importorskip("duckdb")

    from victor.storage.graph.duckdb_store import DuckDBGraphStore

    store = DuckDBGraphStore(tmp_path / "graph.duckdb")
    await _assert_shared_behavior(store)
