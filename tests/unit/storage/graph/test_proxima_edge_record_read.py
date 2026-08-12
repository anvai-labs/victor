# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Reading the durable Tier-A edge tier must be complete or fail closed.

The server caps scan pages at 10,000 records. A single request therefore cannot
implement ``get_all_edges`` for larger graphs, regardless of the client's limit.
These tests cross the cursor boundary and reject partial prefixes when a later
page fails.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from victor.storage.graph.protocol import GraphEdge, GraphNode
from victor.storage.graph.proxima_store import ProximaGraphStore


class _Response:
    def __init__(self, status_code: int, payload: Dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Dict[str, Any]:
        return self._payload


class _Client:
    """Serves one page and then fails, the way the server actually behaves."""

    def __init__(self, pages: List[_Response]) -> None:
        self._pages = pages
        self.requests: List[Dict[str, Any]] = []

    async def __aenter__(self) -> "_Client":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def post(self, url: str, json: Dict[str, Any] | None = None, timeout: float = 0) -> Any:
        self.requests.append(json or {})
        return self._pages[min(len(self.requests) - 1, len(self._pages) - 1)]


def _record(src: str, dst: str, etype: str) -> Dict[str, Any]:
    """A record in the server's typed-envelope read shape."""
    return {
        "id": f"{src}|{etype}|{dst}",
        "props": {
            "record_kind": {"type": "string", "value": "graph_edge"},
            "src": {"type": "string", "value": src},
            "dst": {"type": "string", "value": dst},
            "type": {"type": "string", "value": etype},
            "metadata": {"type": "jsonb", "value": {}},
        },
    }


def _store_with_client(client: _Client) -> ProximaGraphStore:
    store = ProximaGraphStore(repo="t", graph=object())

    class _Db:
        rest_url = "http://unused"

        def _http_client(self) -> _Client:
            return client

    class _Conn:
        embedded_db = _Db()

    store._conn = _Conn()  # type: ignore[assignment]
    store._edge_collection = object()  # short-circuit collection creation
    return store


async def test_edge_read_follows_every_scan_page() -> None:
    """A server cursor means more authoritative rows, not optional work."""
    client = _Client(
        [
            _Response(
                200,
                {
                    "records": [_record("a", "b", "CALLS")],
                    "next_cursor": "page-2",
                },
            ),
            _Response(200, {"records": [_record("b", "c", "CALLS")]}),
        ]
    )
    store = _store_with_client(client)

    edges = await store._read_edge_records()

    assert edges == [
        GraphEdge(src="a", dst="b", type="CALLS", weight=None, metadata={}),
        GraphEdge(src="b", dst="c", type="CALLS", weight=None, metadata={}),
    ]
    assert len(client.requests) == 2
    assert "cursor" not in client.requests[0]
    assert client.requests[1]["cursor"] == "page-2"


async def test_edge_read_rejects_a_partial_prefix_when_a_later_page_fails(caplog: Any) -> None:
    """A prefix is not an authoritative graph and must never be cached."""
    store = _store_with_client(
        _Client(
            [
                _Response(
                    200,
                    {
                        "records": [_record("a", "b", "CALLS")],
                        "next_cursor": "page-2",
                    },
                ),
                _Response(500, {}),
            ]
        )
    )

    with caplog.at_level("WARNING"):
        edges = await store._read_edge_records()

    assert edges is None
    assert store._edge_record_cache is None
    assert any("failed" in r.message.lower() for r in caplog.records)


async def test_edge_read_caches_within_a_session() -> None:
    """`get_all_edges` runs per retrieval; re-scanning each time cost 732 ms."""
    client = _Client([_Response(200, {"records": [_record("a", "b", "CALLS")]})])
    store = _store_with_client(client)

    first = await store._read_edge_records()
    second = await store._read_edge_records()

    assert first == second
    assert len(client.requests) == 1, "the second read should be served from cache"


async def test_get_all_edges_uses_only_the_durable_authority() -> None:
    """A healthy authoritative read must not consult the ORION projection."""
    client = _Client([_Response(200, {"records": [_record("a", "b", "CALLS")]})])
    store = _store_with_client(client)

    class _CountingGraph:
        calls = 0

        def get_all_edges(self) -> List[Any]:
            self.calls += 1
            return []

    graph = _CountingGraph()
    store._graph = graph

    first = await store.get_all_edges()
    second = await store.get_all_edges()

    assert first == second
    assert graph.calls == 0
    assert len(client.requests) == 1


async def test_get_all_edges_does_not_resurrect_a_deleted_edge_from_orion() -> None:
    """Projection lag after a canonical delete must not change the logical result."""
    client = _Client([_Response(200, {"records": []})])
    store = _store_with_client(client)

    class _StaleGraph:
        calls = 0

        def get_all_edges(self) -> List[Any]:
            self.calls += 1
            return [
                type(
                    "ProjectedEdge",
                    (),
                    {
                        "from_node": "deleted",
                        "to_node": "target",
                        "edge_type": "CALLS",
                        "properties": {},
                        "weight": None,
                    },
                )()
            ]

    graph = _StaleGraph()
    store._graph = graph

    assert await store.get_all_edges() == []
    assert graph.calls == 0


async def test_get_all_edges_fails_closed_when_durable_authority_is_unreadable() -> None:
    """An embedded store must never downgrade to a stale projection on scan failure."""
    store = _store_with_client(_Client([_Response(500, {})]))

    class _GraphWithEdge:
        def get_all_edges(self) -> List[Any]:
            return [object()]

    store._graph = _GraphWithEdge()

    with pytest.raises(RuntimeError, match="authoritative Tier-A edge records"):
        await store.get_all_edges()


async def test_sibling_mutation_invalidates_a_generation_tagged_cache() -> None:
    """Incremental reindex through a sibling store must invalidate this view."""
    client = _Client([_Response(200, {"records": [_record("b", "c", "CALLS")]})])
    store = _store_with_client(client)

    class _SharedConnection:
        embedded_db = store._conn.embedded_db

        def __init__(self) -> None:
            self.generation = 0

        def collection_generation(self, name: str) -> int:
            return self.generation

        def mark_collection_mutated(self, name: str) -> int:
            self.generation += 1
            return self.generation

    connection = _SharedConnection()
    store._conn = connection  # type: ignore[assignment]
    store._edge_record_cache = [GraphEdge(src="a", dst="b", type="CALLS")]
    store._edge_record_cache_generation = 0

    connection.mark_collection_mutated(store._edge_collection_name)
    edges = await store._read_edge_records()

    assert edges == [GraphEdge(src="b", dst="c", type="CALLS", weight=None, metadata={})]
    assert len(client.requests) == 1


async def test_edge_read_reports_failure_instead_of_returning_empty(caplog: Any) -> None:
    """An unreadable tier must not masquerade as an empty one."""
    store = _store_with_client(_Client([_Response(500, {})]))

    with caplog.at_level("WARNING"):
        edges = await store._read_edge_records()

    assert edges is None, "a failed read must be distinguishable from an empty graph"
    assert any("failed" in r.message.lower() for r in caplog.records)


class _RecordingCollection:
    """Captures deletes so the edge tier's participation can be asserted."""

    def __init__(self) -> None:
        self.deleted: List[str] = []
        self.cleared = False

    async def delete(self, ids: List[str]) -> int:
        self.deleted.extend(ids)
        return len(ids)

    async def insert_records(self, payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"inserted_count": len(payload), "failed_count": 0}

    async def clear(self) -> None:
        self.cleared = True


async def test_deleting_a_file_invalidates_the_cached_edge_set() -> None:
    """A cached read must not outlive the data it describes.

    `_edge_record_cache` exists so `get_all_edges` — called once per retrieval —
    does not re-scan. It was invalidated only by `upsert_edges`, so after a
    deletion the cache kept serving edges that no longer existed, for the rest of
    the session.
    """
    store = _store_with_client(_Client([_Response(200, {"records": []})]))
    store._edge_record_cache = [GraphEdge(src="a", dst="b", type="CALLS")]

    store._invalidate_edge_records()

    assert store._edge_record_cache is None, (
        "the cached edge set survived a deletion; the next get_all_edges would "
        "return edges that are gone"
    )


async def test_clearing_the_repo_also_clears_the_edge_tier() -> None:
    """`delete_by_repo` must drop edge records, not just symbol records.

    It deleted only `{repo}_codegraph_records`. The edge collection survived, so a
    force rebuild — which is exactly what `delete_by_repo` serves — left every
    previous edge on disk to be read back by the next session.
    """
    deleted_collections: List[str] = []

    class _Db:
        rest_url = "http://unused"

        async def delete_collection(self, name: str) -> bool:
            deleted_collections.append(name)
            return True

    class _Conn:
        embedded_db = _Db()

        def forget_collection(self, name: str) -> None:
            return None

    store = ProximaGraphStore(repo="t", graph=object())
    store._conn = _Conn()  # type: ignore[assignment]
    store._edge_record_cache = [GraphEdge(src="a", dst="b", type="CALLS")]

    await store._clear_edge_records()

    assert store._edge_collection_name in deleted_collections, (
        f"edge collection was not deleted; dropped only {deleted_collections}"
    )
    assert store._edge_record_cache is None


class _Graph:
    def get_nodes_by_file(self, file: str) -> List[Any]:
        return []


class _CpgStore:
    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        return []

    async def delete_by_file(self, file: str) -> None:
        return None

    async def clear(self) -> None:
        return None

    async def close(self) -> None:
        return None


def _store_for_delete() -> tuple[ProximaGraphStore, _RecordingCollection]:
    symbols = _RecordingCollection()
    edges = _RecordingCollection()
    store = ProximaGraphStore(
        repo="t",
        graph=_Graph(),
        record_collection=symbols,
        cpg_store=_CpgStore(),
    )
    store._edge_collection = edges
    return store, edges


async def test_delete_by_file_removes_cached_and_durable_incident_edges() -> None:
    """A file delete must remove the durable edge, not merely its ORION projection."""
    store, edge_collection = _store_for_delete()
    node = GraphNode(node_id="a", type="function", name="a", file="a.py")
    deleted = GraphEdge(src="a", dst="b", type="CALLS")
    retained = GraphEdge(src="x", dst="y", type="CALLS")
    store._record_nodes[node.node_id] = node
    store._record_node_ids.add(node.node_id)
    store._edge_record_cache = [deleted, retained]
    store._edge_record_cache_generation = store._edge_generation()
    store._orion_edge_ids = {"a|CALLS|b", "x|CALLS|y"}

    await store.delete_by_file("a.py")

    assert edge_collection.deleted == ["a|CALLS|b"]
    assert store._edge_record_cache is None
    assert store._orion_edge_ids is None


async def test_delete_by_file_refreshes_cache_before_discovering_incident_edges() -> None:
    """A sibling store may have added an edge after this store cached its scan."""
    store, edge_collection = _store_for_delete()
    node = GraphNode(node_id="a", type="function", name="a", file="a.py")
    added_by_sibling = GraphEdge(src="a", dst="new", type="CALLS")
    store._record_nodes[node.node_id] = node
    store._record_node_ids.add(node.node_id)
    store._edge_record_cache = []

    class _EmbeddedDB:
        _http_client = object()

    class _Connection:
        embedded_db = _EmbeddedDB()

    store._conn = _Connection()  # type: ignore[assignment]  # production scan-capable path

    async def fresh_edge_records() -> List[GraphEdge]:
        assert store._edge_record_cache is None, "delete trusted a possibly stale session cache"
        return [added_by_sibling]

    store._read_edge_records = fresh_edge_records  # type: ignore[method-assign]

    await store.delete_by_file("a.py")

    assert edge_collection.deleted == ["a|CALLS|new"]


async def test_delete_by_repo_clears_edge_record_collection_and_cache() -> None:
    """Repository deletion must clear both record collections and every edge cache."""
    store, edge_collection = _store_for_delete()
    store._edge_record_cache = [GraphEdge(src="a", dst="b", type="CALLS")]
    store._orion_edge_ids = {"a|CALLS|b"}

    await store.delete_by_repo()

    assert edge_collection.cleared
    assert store._edge_record_cache is None
    assert store._orion_edge_ids is None
