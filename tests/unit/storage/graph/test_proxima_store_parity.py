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
from victor.storage.graph.cpg_fragments import CpgFragmentStore  # noqa: E402
from victor.storage.graph.edge_types import EdgeType  # noqa: E402
from victor.storage.graph.proxima_store import ProximaGraphStore  # noqa: E402


# ---------------------------------------------------------------------------
# In-memory fake ProximaDB client (matches the contract ProximaDBGraph needs)
# ---------------------------------------------------------------------------
class FakeProximaClient:
    """Minimal in-memory client implementing the methods ProximaDBGraph calls."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.fail_node_writes = False

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
        if self.fail_node_writes:
            raise RuntimeError("projection unavailable")
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


class FakeRecordCollection:
    """Record-native authority used to verify one-envelope node writes."""

    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, Any]] = {}
        self.writes: List[List[Dict[str, Any]]] = []
        self.fail_with: Exception | None = None

    async def insert_records(self, records):
        if self.fail_with is not None:
            raise self.fail_with
        copied = [
            {
                **record,
                "vector": list(record["vector"]),
                "props": dict(record.get("props") or {}),
            }
            for record in records
        ]
        self.writes.append(copied)
        for record in copied:
            self.records[record["id"]] = record
        return {
            "inserted_count": len(copied),
            "failed_count": 0,
            "errors": [],
            "inserted_ids": [record["id"] for record in copied],
        }

    async def search(self, query_vector, top_k=10, filters=None):
        matches = []
        for record in self.records.values():
            props = record.get("props") or {}
            if filters and any(props.get(key) != value for key, value in filters.items()):
                continue
            matches.append({"id": record["id"], "score": 1.0})
        return matches[:top_k]

    async def delete(self, ids):
        for record_id in ids:
            self.records.pop(record_id, None)
        return len(ids)

    async def clear(self):
        self.records.clear()


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
    store = ProximaGraphStore(
        graph=graph,
        client=client,
        repo="fixture",
        record_collection=FakeRecordCollection(),
    )
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
    """Node metadata updates round-trip through the graph adapter."""
    proxima = await _make_proxima_store()

    await proxima.update_node_metadata("n:parse", {"complexity": 7, "hotspot": True})
    node = await proxima.get_node_by_id("n:parse")
    assert node is not None
    assert node.metadata.get("complexity") == 7
    assert node.metadata.get("hotspot") is True

    # Unknown node is a no-op, not an error.
    await proxima.update_node_metadata("n:missing", {"x": 1})


async def test_hot_node_is_one_atomic_record_before_projection():
    client = FakeProximaClient()
    collection = FakeRecordCollection()
    graph = ProximaDBGraph(client, "atomic_codegraph")
    store = ProximaGraphStore(
        graph=graph,
        client=client,
        repo="atomic",
        record_collection=collection,
    )
    node = GraphNode(
        "n:parse",
        "function",
        "parse",
        "a.py",
        line=10,
        signature="def parse(x)",
        metadata={"complexity": 3},
    )

    await store.upsert_nodes([node])

    record = collection.records["n:parse"]
    assert record["props"]["record_kind"] == "graph_node"
    assert record["props"]["graph_id"] == "atomic_codegraph"
    assert record["props"]["type"] == "function"
    assert record["props"]["name"] == "parse"
    assert record["props"]["metadata"] == {
        "complexity": 3,
        "has_embedding": False,
    }
    assert record["props"]["has_embedding"] is False
    assert record["vector"] == [0.0] * 384
    assert set(client.nodes) == {"n:parse"}


async def test_record_failure_never_creates_graph_only_node():
    client = FakeProximaClient()
    collection = FakeRecordCollection()
    collection.fail_with = RuntimeError("disk full")
    graph = ProximaDBGraph(client, "atomic_codegraph")
    store = ProximaGraphStore(
        graph=graph,
        client=client,
        repo="atomic",
        record_collection=collection,
    )

    with pytest.raises(RuntimeError, match="disk full"):
        await store.upsert_nodes([GraphNode("n:a", "function", "a", "a.py")])

    assert client.nodes == {}


async def test_atomic_embedding_record_contains_graph_props_vector_and_staleness():
    proxima = await _make_proxima_store()
    collection = proxima._symbol_collection
    writes_before = len(collection.writes)
    embedding = [0.1] * 384

    await proxima.upsert_node_record(
        "n:parse",
        embedding,
        metadata={"has_embedding": True, "content_version": "v1"},
    )

    assert len(collection.writes) == writes_before + 1
    assert len(collection.writes[-1]) == 1
    record = collection.records["n:parse"]
    assert record["vector"] == embedding
    assert record["props"]["name"] == "parse"
    assert record["props"]["file"] == "a.py"
    assert record["props"]["metadata"]["has_embedding"] is True
    assert record["props"]["metadata"]["content_version"] == "v1"
    assert record["props"]["has_embedding"] is True
    node = await proxima.get_node_by_id("n:parse")
    assert node.metadata["content_version"] == "v1"


async def test_batched_embedding_records_write_once_per_batch():
    # Plural upsert: TWO round trips per BATCH (one record write + one
    # projection call) instead of two per node — the per-node loop was a
    # dominant term of repo-scale embeddings-on ingest.
    proxima = await _make_proxima_store()
    collection = proxima._symbol_collection
    writes_before = len(collection.writes)

    await proxima.upsert_node_records(
        [
            ("n:parse", [0.1] * 384, {"content_version": "v1"}),
            ("n:helper", [0.2] * 384, {"content_version": "v2"}),
        ],
        metadata={"has_embedding": True},
    )

    assert len(collection.writes) == writes_before + 1, "one batch write, not per-node"
    assert len(collection.writes[-1]) == 2
    assert collection.records["n:parse"]["vector"] == [0.1] * 384
    assert collection.records["n:parse"]["props"]["metadata"]["content_version"] == "v1"
    assert collection.records["n:helper"]["props"]["metadata"]["has_embedding"] is True
    assert collection.records["n:helper"]["props"]["metadata"]["content_version"] == "v2"


async def test_batched_embedding_unknown_node_raises_before_any_write():
    proxima = await _make_proxima_store()
    collection = proxima._symbol_collection
    writes_before = len(collection.writes)

    with pytest.raises(KeyError, match="n:ghost"):
        await proxima.upsert_node_records(
            [
                ("n:parse", [0.1] * 384, None),
                ("n:ghost", [0.2] * 384, None),
            ]
        )

    assert len(collection.writes) == writes_before, "all-or-nothing per batch"


async def test_semantic_search_excludes_pending_placeholder_records():
    proxima = await _make_proxima_store()
    query = [0.1] * 384

    assert await proxima.semantic_search(query, top_k=10) == []

    await proxima.upsert_node_record(
        "n:helper",
        query,
        metadata={"content_version": "v1"},
    )
    hits = await proxima.semantic_search(query, top_k=10)
    assert [node.node_id for node in hits] == ["n:helper"]


async def test_atomic_embedding_failure_preserves_pending_projection():
    proxima = await _make_proxima_store()
    collection = proxima._symbol_collection
    collection.fail_with = RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        await proxima.upsert_node_record(
            "n:parse",
            [0.1] * 384,
            metadata={"has_embedding": True, "content_version": "v1"},
        )

    assert collection.records["n:parse"]["props"]["has_embedding"] is False
    node = await proxima.get_node_by_id("n:parse")
    assert not node.metadata.get("has_embedding")
    assert not hasattr(proxima, "set_node_embedding")


async def test_metadata_update_preserves_committed_vector_atomically():
    proxima = await _make_proxima_store()
    collection = proxima._symbol_collection
    embedding = [0.25] * 384
    await proxima.upsert_node_record(
        "n:parse",
        embedding,
        metadata={"has_embedding": True, "content_version": "v1"},
    )
    writes_before = len(collection.writes)

    await proxima.update_node_metadata("n:parse", {"hotspot": True})

    assert len(collection.writes) == writes_before + 1
    record = collection.records["n:parse"]
    assert record["vector"] == embedding
    assert record["props"]["metadata"] == {
        "has_embedding": True,
        "content_version": "v1",
        "hotspot": True,
    }


async def test_projection_failure_keeps_committed_record_authoritative():
    client = FakeProximaClient()
    collection = FakeRecordCollection()
    store = ProximaGraphStore(
        graph=ProximaDBGraph(client, "atomic_codegraph"),
        client=client,
        repo="atomic",
        record_collection=collection,
    )
    client.fail_node_writes = True

    with pytest.raises(RuntimeError, match="ORION node projection failed"):
        await store.upsert_nodes([GraphNode("n:a", "function", "a", "a.py")])

    assert "n:a" in collection.records
    assert client.nodes == {}
    committed = await store.get_node_by_id("n:a")
    assert committed is not None
    assert committed.name == "a"


async def test_wrong_embedding_dimension_fails_before_record_mutation():
    proxima = await _make_proxima_store()
    collection = proxima._symbol_collection
    writes_before = len(collection.writes)

    with pytest.raises(ValueError, match="3 dimensions; expected 384"):
        await proxima.upsert_node_record("n:parse", [0.1, 0.2, 0.3])

    assert len(collection.writes) == writes_before


async def test_delete_by_repo_clears_records_and_sidecars():
    class FakeEmbeddedDB:
        def __init__(self):
            self.deleted = []

        async def delete_collection(self, name):
            self.deleted.append(name)
            return True

    class FakeConnection:
        def __init__(self):
            self.embedded_db = FakeEmbeddedDB()
            self.forgotten = []

        def forget_collection(self, name):
            self.forgotten.append(name)

    proxima = await _make_proxima_store()
    connection = FakeConnection()
    proxima._conn = connection
    proxima._symbol_collection = object()
    proxima._file_mtimes["a.py"] = 1.0
    proxima._file_hashes["a.py"] = "abc"

    await proxima.delete_by_repo(clear_embeddings=True)

    # Both tiers must be dropped. Clearing only the symbol records left the edge
    # collection on disk, so a force rebuild — which is what delete_by_repo
    # serves — resurrected every previous edge in the next session.
    assert connection.embedded_db.deleted == [
        "fixture_codegraph_records",
        "fixture_codegraph_edges",
    ]
    assert connection.forgotten == ["fixture_codegraph_records", "fixture_codegraph_edges"]
    assert proxima._symbol_collection is None
    assert proxima._file_mtimes == {}
    assert proxima._file_hashes == {}


async def test_delete_by_repo_stops_before_projection_reset_when_record_cleanup_fails():
    class FailingEmbeddedDB:
        async def delete_collection(self, name):
            return False

    class FailingConnection:
        embedded_db = FailingEmbeddedDB()

        def forget_collection(self, name):
            raise AssertionError("failed collection deletion must not invalidate the handle")

    proxima = await _make_proxima_store()
    proxima._conn = FailingConnection()
    proxima._file_hashes["a.py"] = "abc"

    with pytest.raises(RuntimeError, match="Failed to delete ProximaDB collection"):
        await proxima.delete_by_repo(clear_embeddings=True)

    assert await proxima.get_all_nodes()
    assert proxima._file_hashes == {"a.py": "abc"}


async def test_tier_boundary_keeps_cpg_out_of_hot_graph_and_drills_down_on_demand(tmp_path):
    client = FakeProximaClient()
    graph = ProximaDBGraph(client, "tiered_codegraph")
    fragments = CpgFragmentStore(tmp_path / "cpg-fragments.sqlite3")
    store = ProximaGraphStore(
        graph=graph,
        client=client,
        repo="tiered",
        cpg_store=fragments,
        record_collection=FakeRecordCollection(),
    )
    symbol = GraphNode("fn:a", "function", "a", "src/a.py", line=1)
    statements = [
        GraphNode(
            "stmt:1",
            "statement",
            "assignment:2",
            "src/a.py",
            line=2,
            scope_id="fn:a",
            statement_type="assignment",
        ),
        GraphNode(
            "stmt:2",
            "statement",
            "return:3",
            "src/a.py",
            line=3,
            scope_id="fn:a",
            statement_type="return",
        ),
    ]

    await store.upsert_nodes([symbol, *statements])
    await store.upsert_edges(
        [
            GraphEdge("fn:a", "fn:b", EdgeType.CALLS),
            GraphEdge("stmt:1", "stmt:2", EdgeType.CFG_SUCCESSOR),
            GraphEdge("stmt:1", "stmt:2", EdgeType.DDG_DEF_USE),
        ]
    )

    # Tier A remains the only globally traversable/scan-visible graph.
    assert set(client.nodes) == {"fn:a"}
    assert {edge["edge_type"] for edge in client.edges} == {EdgeType.CALLS}
    assert [node.node_id for node in await store.get_all_nodes()] == ["fn:a"]
    assert {edge.type for edge in await store.get_all_edges()} == {EdgeType.CALLS}
    assert {edge.type for edge in await store.get_neighbors("stmt:1")} == set()

    # Explicit statement and CCG requests drill into Tier B without promoting it.
    assert [node.node_id for node in await store.get_nodes_by_statement_type("assignment")] == [
        "stmt:1"
    ]
    assert {node.node_id for node in await store.get_nodes_by_scope("fn:a")} == {
        "stmt:1",
        "stmt:2",
    }
    assert [
        node.node_id
        for node in await store.find_nodes(name="return:3", type="statement", file="src/a.py")
    ] == ["stmt:2"]
    assert (await store.get_node_by_id("stmt:2")).statement_type == "return"
    cold_edges = await store.get_neighbors(
        "stmt:1",
        edge_types={EdgeType.CFG_SUCCESSOR, EdgeType.DDG_DEF_USE},
        direction="out",
    )
    assert {edge.type for edge in cold_edges} == {
        EdgeType.CFG_SUCCESSOR,
        EdgeType.DDG_DEF_USE,
    }
    cold_batches = [
        batch
        async for batch in store.iter_edges(
            batch_size=1,
            edge_types={EdgeType.CFG_SUCCESSOR, EdgeType.DDG_DEF_USE},
        )
    ]
    assert all(len(batch) == 1 for batch in cold_batches)
    assert {edge.type for batch in cold_batches for edge in batch} == {
        EdgeType.CFG_SUCCESSOR,
        EdgeType.DDG_DEF_USE,
    }

    stats = await store.stats()
    assert stats["tier_a_nodes"] == 1
    assert stats["tier_a_edges"] == 1
    assert stats["tier_b_nodes"] == 2
    assert stats["tier_b_edges"] == 2
    await store.close()


async def test_tier_boundary_deletes_hot_and_cold_file_state(tmp_path):
    client = FakeProximaClient()
    graph = ProximaDBGraph(client, "tiered_codegraph")
    store = ProximaGraphStore(
        graph=graph,
        client=client,
        repo="tiered",
        cpg_store=CpgFragmentStore(tmp_path / "cpg-fragments.sqlite3"),
        record_collection=FakeRecordCollection(),
    )
    await store.upsert_nodes(
        [
            GraphNode("fn:a", "function", "a", "src/a.py"),
            GraphNode(
                "stmt:a",
                "statement",
                "return:2",
                "src/a.py",
                line=2,
                scope_id="fn:a",
                statement_type="return",
            ),
        ]
    )
    await store.upsert_edges([GraphEdge("stmt:a", "stmt:a", EdgeType.CFG_LOOP_BACK)])

    await store.delete_by_file("src/a.py")

    assert client.nodes == {}
    assert await store.get_node_by_id("stmt:a") is None
    assert (await store.stats())["tier_b_nodes"] == 0

    await store.upsert_nodes(
        [GraphNode("stmt:b", "statement", "return:1", "src/b.py", statement_type="return")]
    )
    await store.delete_by_repo()
    assert (await store.stats())["tier_b_nodes"] == 0
    await store.close()


async def test_file_delete_reopens_record_authority_before_projection_delete():
    store = await _make_proxima_store()
    collection = store._symbol_collection

    class FakeConnection:
        async def get_or_create_collection(self, name, *, dimension):
            # delete_by_file now touches BOTH tiers: the symbol records (384-dim)
            # and the durable edge records (dim 1), since edges incident to the
            # deleted nodes must go too or the graph keeps dangling edges.
            assert name in ("fixture_codegraph_records", "fixture_codegraph_edges")
            assert dimension == (1 if name.endswith("_edges") else 384)
            return collection

    # Simulate a restarted store whose collection handle/cache has not yet been
    # hydrated while the ORION projection is already available.
    store._symbol_collection = None
    store._record_nodes.clear()
    store._record_vectors.clear()
    store._record_node_ids.clear()
    store._conn = FakeConnection()

    await store.delete_by_file("a.py")

    assert "n:main" not in collection.records
    assert "n:parse" not in collection.records


async def test_stats_contract_holds_for_both_backends(tmp_path):
    """Both backends must expose integer ``nodes``/``edges`` totals.

    ``victor init`` indexes these directly (``db_stats['nodes']``). ProximaDB
    previously reported only ``tier_a_*``/``tier_b_*``, so a repo whose
    ``.victor/graph_backend`` marker selected proxima died with
    KeyError('nodes') and printed just "CCG indexing skipped: 'nodes'" — the
    graph index was silently never built.
    """
    sqlite_store = await _make_sqlite_store(tmp_path)
    proxima_store = await _make_proxima_store()
    try:
        for store in (sqlite_store, proxima_store):
            stats = await store.stats()
            for key in ("nodes", "edges"):
                assert key in stats, f"{type(store).__name__}.stats() is missing {key!r}"
                assert isinstance(stats[key], int), (
                    f"{type(store).__name__}.stats()[{key!r}] must be an int, "
                    f"got {type(stats[key]).__name__}"
                )
    finally:
        await sqlite_store.close()
        await proxima_store.close()


async def test_proxima_stats_totals_span_both_tiers():
    """``nodes``/``edges`` are whole-graph totals, not just the ORION tier.

    Tier-B holds every statement node and CFG/CDG/DDG edge, which is the bulk of
    a CCG index. Reporting only Tier-A would under-count the graph by most of it.
    """
    store = await _make_proxima_store()
    try:
        stats = await store.stats()
        assert stats["nodes"] == stats["tier_a_nodes"] + stats["tier_b_nodes"]
        assert stats["edges"] == stats["tier_a_edges"] + stats["tier_b_edges"]
    finally:
        await store.close()


def test_orion_stat_reads_the_envelope_shape():
    """ORION answers {'success':..., 'data': {'total_nodes': N}}, not a flat map.

    Reading `node_count`/`nodes` off the top level returned 0 for a graph holding
    1,328 symbols, which silently dropped the whole Tier-A tier from the totals.
    """
    from victor.storage.graph.proxima_store import _orion_stat

    envelope = {
        "success": True,
        "data": {"total_nodes": 1328, "total_edges": 1258, "average_degree": 0.94},
    }
    assert _orion_stat(envelope, "nodes") == 1328
    assert _orion_stat(envelope, "edges") == 1258


def test_orion_stat_still_reads_flat_spellings():
    from victor.storage.graph.proxima_store import _orion_stat

    assert _orion_stat({"node_count": 7, "edge_count": 9}, "nodes") == 7
    assert _orion_stat({"nodes": 3, "edges": 4}, "edges") == 4


def test_orion_stat_returns_zero_for_an_unknown_shape():
    """A 0 must mean 'unreported' so stats() falls back to a real recount."""
    from victor.storage.graph.proxima_store import _orion_stat

    assert _orion_stat({}, "nodes") == 0
    assert _orion_stat({"data": {"unexpected": 1}}, "edges") == 0


async def test_upsert_edges_filters_duplicates_before_writing():
    """Duplicates must never reach ORION's batch_create_edges.

    That call aborts the batch at the first "already exists" and silently
    discards every edge after it — a 3-edge batch whose middle edge is a repeat
    answers created=1, failed=1 and drops the third without counting it. Because
    the cross-file resolution passes deliberately re-emit edges per-file
    indexing already wrote, whichever edge happened to be the first duplicate
    decided how many survived, and the same corpus indexed to a different edge
    count on every run.
    """
    store = await _make_proxima_store()
    try:
        graph = await store._ensure()
        edge = GraphEdge(src="n:main", dst="n:parse", type="CALLS")

        await store.upsert_edges([edge])
        first = len(await store.get_all_edges())

        calls_before = len(getattr(graph, "created_edge_batches", []) or [])
        await store.upsert_edges([edge])  # exact repeat
        assert len(await store.get_all_edges()) == first, "repeat must not add an edge"

        # The repeat must be filtered client-side, not handed to the server.
        batches = getattr(graph, "created_edge_batches", None)
        if batches is not None:
            assert len(batches) == calls_before, "duplicate batch should not be sent"
    finally:
        await store.close()


async def test_upsert_edges_raises_when_the_server_drops_edges_silently():
    """created + failed short of the batch size means edges vanished."""
    store = await _make_proxima_store()
    try:
        with pytest.raises(RuntimeError, match="silently dropped"):
            store._validate_edge_write(
                {"success": True, "created": 1, "failed": 1, "errors": []},
                3,
                [GraphEdge(src="a", dst="b", type="CALLS")],
            )
    finally:
        await store.close()
