# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Reading the durable Tier-A edge tier must degrade loudly, never to silence.

The first version of this read paginated with the server's scan cursor and
returned ``None`` if any page failed. The server mints that cursor against the
collection's numeric object id and validates it against the name in the request
path, so page 2 always 400s for a name-addressed client
(anvai-labs/proximaDB#1542). The result: a graph with more than one page of edges
reported **zero** edges — indistinguishable from an empty collection — and the
failure was logged at debug.

A unit test catches this where an integration test did not: the durability
fixture holds ~7 edges, one page, so it never reached the second request.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from victor.storage.graph.protocol import GraphEdge
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


async def test_edge_read_issues_a_single_request_not_a_cursor_loop() -> None:
    """Following the cursor is impossible by name, so it must not be attempted."""
    client = _Client([_Response(200, {"records": [_record("a", "b", "CALLS")]})])
    store = _store_with_client(client)

    edges = await store._read_edge_records()

    assert edges == [GraphEdge(src="a", dst="b", type="CALLS", weight=None, metadata={})]
    assert len(client.requests) == 1, "the read must not paginate"
    assert "cursor" not in client.requests[0]


async def test_edge_read_keeps_rows_when_the_server_reports_more(caplog: Any) -> None:
    """A truncated read returns what it has AND says so.

    Previously this returned None, so a partial read looked like an empty graph.
    """
    payload = {
        "records": [_record("a", "b", "CALLS"), _record("b", "c", "CALLS")],
        "next_cursor": "opaque",
    }
    store = _store_with_client(_Client([_Response(200, payload)]))

    with caplog.at_level("WARNING"):
        edges = await store._read_edge_records()

    assert len(edges) == 2, "rows already read must not be discarded"
    assert any("truncated" in r.message.lower() for r in caplog.records), (
        "truncation must be reported at WARNING; a silent prefix of the graph is "
        "the failure mode this test exists to prevent"
    )


async def test_edge_read_caches_within_a_session() -> None:
    """`get_all_edges` runs per retrieval; re-scanning each time cost 732 ms."""
    client = _Client([_Response(200, {"records": [_record("a", "b", "CALLS")]})])
    store = _store_with_client(client)

    first = await store._read_edge_records()
    second = await store._read_edge_records()

    assert first == second
    assert len(client.requests) == 1, "the second read should be served from cache"


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

    async def delete(self, ids: List[str]) -> int:
        self.deleted.extend(ids)
        return len(ids)

    async def insert_records(self, payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"inserted_count": len(payload), "failed_count": 0}


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

    assert (
        store._edge_collection_name in deleted_collections
    ), f"edge collection was not deleted; dropped only {deleted_collections}"
    assert store._edge_record_cache is None
