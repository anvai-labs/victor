# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Parity tests: ProximaGraphStore vs SqliteGraphStore (TD-11/12/13).

These tests drive the **real** ``proximadb_sdk.graph.ProximaDBGraph`` against an
in-memory fake ProximaDB client, so the actual ``ProximaGraphStore`` adapter and
ProximaDB's real traversal/search logic run without needing the embedded server
binary. The verification gate from
``docs/architecture/proximadb-codegraph-backend.md`` — ``impact_analysis``
(forward/backward) and hybrid seed→expand must match the SQLite store on known
symbols — is asserted here against the default SQLite backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

try:  # importorskip only catches ImportError; a broken install (e.g. grpc codegen
    # mismatch) raises RuntimeError and would fail collection for the whole suite.
    import proximadb_sdk  # noqa: F401
except Exception as _exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"proximadb_sdk unavailable: {_exc}", allow_module_level=True)

from proximadb_sdk.graph import ProximaDBGraph  # noqa: E402

from victor.storage.graph import GraphEdge, GraphNode, SqliteGraphStore  # noqa: E402
from victor.storage.graph.proxima_store import ProximaGraphStore  # noqa: E402


# ---------------------------------------------------------------------------
# In-memory fake ProximaDB client (matches the contract ProximaDBGraph needs)
# ---------------------------------------------------------------------------
class FakeProximaClient:
    """Minimal in-memory client implementing the methods ProximaDBGraph calls."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []

    # graph lifecycle
    def create_graph(self, graph_id: str, *a: Any, **k: Any) -> Dict[str, Any]:
        return {"success": True}

    def delete_graph(self, graph_id: str, *a: Any, **k: Any) -> Dict[str, Any]:
        self.nodes.clear()
        self.edges.clear()
        return {"success": True}

    # writes
    def create_node(
        self,
        graph_id: str,
        node_id: str,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.nodes[node_id] = {
            "id": node_id,
            "labels": list(labels or []),
            "properties": dict(properties or {}),
        }
        return {"success": True}

    def create_edge(
        self,
        graph_id: str,
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
        properties: Optional[Dict[str, Any]] = None,
        weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.edges.append(
            {
                "id": edge_id,
                "from_node_id": from_node_id,
                "to_node_id": to_node_id,
                "edge_type": edge_type,
                "properties": dict(properties or {}),
                "weight": weight,
            }
        )
        return {"success": True}

    def delete_node(self, node_id: str, graph_id: Optional[str] = None) -> Dict[str, Any]:
        self.nodes.pop(node_id, None)
        self.edges = [
            e for e in self.edges if e["from_node_id"] != node_id and e["to_node_id"] != node_id
        ]
        return {"success": True}

    # reads
    def get_node(self, node_id: str, graph_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self.nodes.get(node_id)

    def query_nodes(
        self,
        graph_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        matched: List[Dict[str, Any]] = []
        for node in self.nodes.values():
            if labels and not any(label in node["labels"] for label in labels):
                continue
            if properties and any(node["properties"].get(k) != v for k, v in properties.items()):
                continue
            matched.append(node)
        offset = offset or 0
        page = matched[offset:]
        if limit is not None:
            page = page[:limit]
        return {"nodes": page}

    def get_outgoing_edges(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        graph_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return [
            e
            for e in self.edges
            if e["from_node_id"] == node_id and (not edge_types or e["edge_type"] in edge_types)
        ]

    def get_incoming_edges(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        graph_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return [
            e
            for e in self.edges
            if e["to_node_id"] == node_id and (not edge_types or e["edge_type"] in edge_types)
        ]

    def get_graph_stats(self, graph_id: Optional[str] = None) -> Dict[str, Any]:
        return {"node_count": len(self.nodes), "edge_count": len(self.edges)}


# ---------------------------------------------------------------------------
# Fixture repo: a tiny but real call graph with known symbols
# ---------------------------------------------------------------------------
def _fixture_nodes() -> List[GraphNode]:
    return [
        GraphNode(
            node_id="n:main",
            type="function",
            name="main",
            file="a.py",
            line=1,
            signature="def main()",
            docstring="entrypoint",
        ),
        GraphNode(
            node_id="n:parse",
            type="function",
            name="parse",
            file="a.py",
            line=10,
            signature="def parse(x)",
            docstring="parse input",
        ),
        GraphNode(
            node_id="n:validate",
            type="function",
            name="validate",
            file="b.py",
            line=5,
            signature="def validate(x)",
            docstring="validate input",
        ),
        GraphNode(
            node_id="n:helper",
            type="function",
            name="helper",
            file="b.py",
            line=20,
            signature="def helper()",
            docstring="shared helper",
        ),
    ]


def _fixture_edges() -> List[GraphEdge]:
    return [
        GraphEdge(src="n:main", dst="n:parse", type="CALLS"),
        GraphEdge(src="n:main", dst="n:validate", type="CALLS"),
        GraphEdge(src="n:parse", dst="n:helper", type="CALLS"),
        GraphEdge(src="n:validate", dst="n:helper", type="CALLS"),
    ]


async def _make_sqlite_store(tmp_path) -> SqliteGraphStore:
    store = SqliteGraphStore(project_path=tmp_path)
    await store.initialize()
    await store.upsert_nodes(_fixture_nodes())
    await store.upsert_edges(_fixture_edges())
    return store


async def _make_proxima_store() -> ProximaGraphStore:
    client = FakeProximaClient()
    graph = ProximaDBGraph(client, "fixture_codegraph")
    store = ProximaGraphStore(graph=graph, client=client, repo="fixture")
    await store.upsert_nodes(_fixture_nodes())
    await store.upsert_edges(_fixture_edges())
    return store


def _edge_keys(edges: List[GraphEdge]) -> set:
    return {(e.src, e.dst, e.type) for e in edges}


def _impacted_ids(edges: List[GraphEdge], direction: str) -> set:
    # Mirrors victor.tools.graph_query_tool.impact_analysis node collection.
    return {(e.src if direction == "in" else e.dst) for e in edges}


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["n:main", "n:parse", "n:validate", "n:helper"])
@pytest.mark.parametrize("max_depth", [1, 2, 3])
async def test_impact_analysis_forward_parity(tmp_path, target, max_depth):
    """Forward impact (incoming edges) must match SQLite for known symbols."""
    sqlite = await _make_sqlite_store(tmp_path)
    proxima = await _make_proxima_store()
    try:
        s_edges = await sqlite.get_neighbors(target, direction="in", max_depth=max_depth)
        p_edges = await proxima.get_neighbors(target, direction="in", max_depth=max_depth)
        assert _edge_keys(s_edges) == _edge_keys(p_edges)
        assert _impacted_ids(s_edges, "in") == _impacted_ids(p_edges, "in")
    finally:
        await sqlite.close()


@pytest.mark.parametrize("target", ["n:main", "n:parse", "n:validate", "n:helper"])
@pytest.mark.parametrize("max_depth", [1, 2, 3])
async def test_impact_analysis_backward_parity(tmp_path, target, max_depth):
    """Backward impact (outgoing edges) must match SQLite for known symbols."""
    sqlite = await _make_sqlite_store(tmp_path)
    proxima = await _make_proxima_store()
    try:
        s_edges = await sqlite.get_neighbors(target, direction="out", max_depth=max_depth)
        p_edges = await proxima.get_neighbors(target, direction="out", max_depth=max_depth)
        assert _edge_keys(s_edges) == _edge_keys(p_edges)
        assert _impacted_ids(s_edges, "out") == _impacted_ids(p_edges, "out")
    finally:
        await sqlite.close()


async def test_hybrid_seed_expand_parity(tmp_path):
    """Hybrid seed→expand: same seed yields identical expanded node/edge sets."""
    sqlite = await _make_sqlite_store(tmp_path)
    proxima = await _make_proxima_store()
    try:
        # Seed discovery: exact-name search must surface the known symbol in both.
        s_seed = await sqlite.search_symbols("helper", limit=5)
        p_seed = await proxima.search_symbols("helper", limit=5)
        assert "n:helper" in {n.node_id for n in s_seed}
        assert "n:helper" in {n.node_id for n in p_seed}

        # Expand from the same known seeds via parallel multi-hop traversal.
        seeds = ["n:main"]
        s_result = await sqlite.multi_hop_traverse_parallel(seeds, max_hops=3)
        p_result = await proxima.multi_hop_traverse_parallel(seeds, max_hops=3)
        assert {n.node_id for n in s_result.nodes} == {n.node_id for n in p_result.nodes}
        assert _edge_keys(s_result.edges) == _edge_keys(p_result.edges)
    finally:
        await sqlite.close()


async def test_node_lookup_parity(tmp_path):
    """find_nodes / get_node_by_id / get_nodes_by_file agree with SQLite."""
    sqlite = await _make_sqlite_store(tmp_path)
    proxima = await _make_proxima_store()
    try:
        s_node = await sqlite.get_node_by_id("n:parse")
        p_node = await proxima.get_node_by_id("n:parse")
        assert s_node is not None and p_node is not None
        assert (s_node.name, s_node.type, s_node.file, s_node.line) == (
            p_node.name,
            p_node.type,
            p_node.file,
            p_node.line,
        )

        s_by_file = {n.node_id for n in await sqlite.get_nodes_by_file("b.py")}
        p_by_file = {n.node_id for n in await proxima.get_nodes_by_file("b.py")}
        assert s_by_file == p_by_file == {"n:validate", "n:helper"}

        s_found = {n.node_id for n in await sqlite.find_nodes(name="helper")}
        p_found = {n.node_id for n in await proxima.find_nodes(name="helper")}
        assert s_found == p_found == {"n:helper"}
    finally:
        await sqlite.close()


async def test_all_edges_parity(tmp_path):
    sqlite = await _make_sqlite_store(tmp_path)
    proxima = await _make_proxima_store()
    try:
        assert _edge_keys(await sqlite.get_all_edges()) == _edge_keys(await proxima.get_all_edges())
    finally:
        await sqlite.close()


async def test_oid_is_the_only_correlation_key():
    """The graph node id IS the oid; embedding_ref is not used (retired)."""
    proxima = await _make_proxima_store()
    node = await proxima.get_node_by_id("n:helper")
    assert node is not None
    # embedding_ref is never round-tripped through the ProximaDB backend.
    assert node.embedding_ref is None


async def test_indexing_pipeline_node_updates():
    """update_node_metadata round-trips; set_node_embedding is a safe no-throw."""
    proxima = await _make_proxima_store()

    await proxima.update_node_metadata("n:parse", {"complexity": 7, "hotspot": True})
    node = await proxima.get_node_by_id("n:parse")
    assert node is not None
    assert node.metadata.get("complexity") == 7
    assert node.metadata.get("hotspot") is True

    # set_node_embedding must not raise even though the vector is owned by the
    # co-oid'd vector collection, not the graph node.
    await proxima.set_node_embedding("n:parse", [0.1, 0.2, 0.3])
    # Unknown node is a no-op, not an error.
    await proxima.update_node_metadata("n:missing", {"x": 1})
    await proxima.set_node_embedding("n:missing", [0.0])
