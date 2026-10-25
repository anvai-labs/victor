"""ADR-033 phase 1: the interactive explorer (viz_queries + server).

Golden-element determinism, the read-only surface, allowlist hardening,
the no-graph 409, and statement-node exclusion — against a seeded store,
so CI exercises everything without a pre-indexed checkout.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from victor.core.graph_rag import viz_queries  # noqa: E402
from victor.integrations.api.viz_server import create_viz_app  # noqa: E402


async def _seed(tmp_path: Path) -> Path:
    from victor.storage.graph.protocol import GraphEdge, GraphNode
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    def node(node_id, name, file, type_, line=1):
        return GraphNode(
            node_id=node_id,
            type=type_,
            name=name,
            file=file,
            line=line,
            end_line=line + 1,
            lang="python",
        )

    nodes = [
        node("orch", "AgentOrchestrator", "victor/agent/orchestrator.py", "class"),
        node("chat", "chat", "victor/agent/chat_service.py", "function"),
        node("cfg", "Settings", "victor/config/settings.py", "class"),
        node("msg", "Message", "victor/providers/base.py", "class"),
        # A statement node: must never appear in explorer output.
        node("stmt", "assignment:1", "victor/agent/orchestrator.py", "statement"),
    ]
    edges = [
        GraphEdge(src="orch", dst="cfg", type="CALLS"),
        GraphEdge(src="chat", dst="orch", type="CALLS"),
        GraphEdge(src="chat", dst="cfg", type="CALLS"),
        GraphEdge(src="orch", dst="chat", type="REFERENCES"),
        GraphEdge(src="chat", dst="msg", type="CALLS"),
    ]
    store = SqliteGraphStore(tmp_path)
    await store.initialize()
    await store.upsert_nodes(nodes)
    await store.upsert_edges(edges)
    await store.close()
    return tmp_path / ".victor" / "project.db"


@pytest.fixture
def db_path(tmp_path):
    return asyncio.get_event_loop().run_until_complete(_seed(tmp_path))


@pytest.fixture
def client(tmp_path):
    # A workspace with the seeded graph under its .victor dir.
    asyncio.get_event_loop().run_until_complete(_seed(tmp_path))
    return TestClient(create_viz_app(tmp_path))


def test_viz_queries_are_deterministic(db_path):
    first = viz_queries.package_graph(db_path)
    second = viz_queries.package_graph(db_path)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["nodes"], "seeded graph must yield package nodes"


def test_class_graph_excludes_statement_nodes(db_path):
    result = viz_queries.class_graph(db_path, "victor/agent/orchestrator.py")
    kinds = {n["data"].get("kind") for n in result["nodes"]}
    assert "statement" not in kinds
    assert "class" in kinds or "function" in kinds


def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Victor Code Explorer" in res.text


def test_static_allowlist_hardening(client):
    for good in ("explorer.js", "explorer.css"):
        assert client.get(f"/static/{good}").status_code == 200
    assert client.get("/static/secret.txt").status_code == 404
    assert client.get("/static/vendor/cytoscape.min.js").status_code in (200, 404)
    assert client.get("/static/vendor/evil.js").status_code == 404


def test_no_graph_is_a_409(tmp_path):
    empty = tmp_path / "empty-ws"
    empty.mkdir()
    res = TestClient(create_viz_app(empty)).get("/api/graph/package")
    assert res.status_code == 409
    assert "index graph" in res.json()["detail"]


def test_api_routes_return_element_lists(client):
    pkg = client.get("/api/graph/package")
    assert pkg.status_code == 200
    body = pkg.json()
    assert set(body) == {"nodes", "edges"}
    assert all(n["data"]["kind"] == "package" for n in body["nodes"])
    mod = client.get("/api/graph/module/victor/agent")
    assert mod.status_code == 200
    hits = client.get("/api/graph/search?q=chat")
    assert hits.status_code == 200
    assert any(r["data"]["label"] == "chat" for r in hits.json()["results"])
