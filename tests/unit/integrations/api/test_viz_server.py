"""ADR-033: the interactive explorer (viz_queries + server).

Phase 1: golden-element determinism, the read-only surface, allowlist
hardening, the no-graph 409, and statement-node exclusion. Phase 2: the
call-flow sequence view (call_line ordering, cycle guard, truncation caps,
unordered-graph fallback) — against a seeded store, so CI exercises
everything without a pre-indexed checkout.
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

    def node(node_id, name, file, type_, line=1, parent_id=None, signature=None):
        return GraphNode(
            node_id=node_id,
            type=type_,
            name=name,
            file=file,
            line=line,
            end_line=line + 1,
            lang="python",
            parent_id=parent_id,
            signature=signature,
        )

    nodes = [
        node("orch", "AgentOrchestrator", "victor/agent/orchestrator.py", "class"),
        node("chat", "chat", "victor/agent/chat_service.py", "function"),
        node("cfg", "Settings", "victor/config/settings.py", "class"),
        node("msg", "Message", "victor/providers/base.py", "class"),
        # A statement node: must never appear in explorer output.
        node("stmt", "assignment:1", "victor/agent/orchestrator.py", "statement"),
        # Flow-shape (ADR-033 phase 2): a service class with methods, a repo
        # class, and a free function — calls carry call_line metadata.
        node("svc_cls", "TurnService", "victor/agent/services/turn.py", "class"),
        node("svc_execute", "execute", "victor/agent/services/turn.py", "method", 30, "svc_cls"),
        node("svc_step", "_step", "victor/agent/services/turn.py", "method", 80, "svc_cls"),
        node("svc_helper", "helper", "victor/agent/services/turn.py", "method", 120, "svc_cls"),
        node("svc_handle", "handle", "victor/agent/services/turn.py", "method", 200, "svc_cls"),
        node("repo_cls", "Repo", "victor/core/repo.py", "class"),
        node("repo_save", "save", "victor/core/repo.py", "method", 10, "repo_cls"),
        node("repo_handle", "handle", "victor/core/repo.py", "method", 50, "repo_cls"),
        node("util_fn", "serialize", "victor/core/util.py", "function", 5),
        node("plain_fn", "plain", "victor/core/util.py", "function", 60),
        # Hostile name: graph names are untrusted text (polyglot harvest);
        # mermaid output must never carry raw markup through.
        node("evil_fn", 'bad<svg>"onload"&', "victor/core/util.py", "function", 90),
        # Class-diagram shape: a base, a focal service on it, a subclass,
        # and classes the focal methods use.
        node("base_cls", "BaseService", "victor/core/base.py", "class", 5),
        node("svc2_cls", "SessionService", "victor/agent/services/session.py", "class", 20),
        node(
            "svc2_start",
            "start",
            "victor/agent/services/session.py",
            "method",
            30,
            "svc2_cls",
            signature="start(session - Session) -> None",
        ),
        node("svc2_stop", "stop", "victor/agent/services/session.py", "method", 40, "svc2_cls"),
        node("sub_cls", "CachedSessionService", "victor/agent/services/cache.py", "class", 10),
        node("used_cls", "Metrics", "victor/core/metrics.py", "class", 1),
    ]
    edges = [
        GraphEdge(src="orch", dst="cfg", type="CALLS"),
        GraphEdge(src="chat", dst="orch", type="CALLS"),
        GraphEdge(src="chat", dst="cfg", type="CALLS"),
        GraphEdge(src="orch", dst="chat", type="REFERENCES"),
        GraphEdge(src="chat", dst="msg", type="CALLS"),
        # execute: constructor (L20), method (L40), function (L60) — inserted
        # deliberately out of source order; flow must sort by call_line.
        GraphEdge(
            src="svc_execute",
            dst="repo_cls",
            type="CALLS",
            metadata={"call_line": 20, "provenance": "scope_resolver"},
        ),
        GraphEdge(
            src="svc_execute",
            dst="svc_step",
            type="CALLS",
            metadata={"call_line": 40, "provenance": "scope_resolver"},
        ),
        GraphEdge(
            src="svc_execute",
            dst="util_fn",
            type="CALLS",
            metadata={"call_line": 60, "provenance": "scope_resolver"},
        ),
        GraphEdge(
            src="svc_execute",
            dst="svc_helper",
            type="CALLS",
            metadata={"call_line": 70, "provenance": "scope_resolver"},
        ),
        GraphEdge(
            src="svc_step",
            dst="repo_save",
            type="CALLS",
            metadata={"call_line": 99, "provenance": "scope_resolver"},
        ),
        GraphEdge(
            src="svc_step",
            dst="svc_execute",
            type="CALLS",
            metadata={"call_line": 105, "provenance": "scope_resolver"},
        ),
        GraphEdge(
            src="svc_helper",
            dst="svc_helper",
            type="CALLS",
            metadata={"call_line": 7, "provenance": "scope_resolver"},
        ),
        # Pre-flow edge shape: no call_line anywhere on this path.
        GraphEdge(src="plain_fn", dst="util_fn", type="CALLS"),
        GraphEdge(src="plain_fn", dst="evil_fn", type="CALLS", metadata={"call_line": 91}),
        # INHERITS edges are subclass -> base (src is the child).
        GraphEdge(src="svc2_cls", dst="base_cls", type="INHERITS"),
        GraphEdge(src="sub_cls", dst="svc2_cls", type="INHERITS"),
        GraphEdge(
            src="svc2_start",
            dst="used_cls",
            type="CALLS",
            metadata={"call_line": 35, "provenance": "scope_resolver"},
        ),
        GraphEdge(
            src="svc2_stop",
            dst="used_cls",
            type="CALLS",
            metadata={"call_line": 45, "provenance": "scope_resolver"},
        ),
        GraphEdge(src="svc2_start", dst="repo_cls", type="REFERENCES"),
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
    res = client.get("/viz")
    assert res.status_code == 200
    assert "Victor Code Explorer" in res.text


def test_index_uses_viz_absolute_asset_paths(client):
    """Assets must be /viz-absolute. Relative paths break in a real browser:
    at /viz (no trailing slash) the base directory is /, so "assets/x.js"
    resolves to /assets/x.js and 404s — the page ships but never renders.
    Both mounts (main server, standalone wrapper) serve the page at the
    identical /viz path, so root-absolute is correct everywhere."""
    res = client.get("/viz")
    assert res.status_code == 200
    assert 'src="/viz/assets/explorer.js"' in res.text
    assert 'src="/viz/assets/vendor/mermaid.min.js"' in res.text
    assert 'href="/viz/assets/explorer.css"' in res.text
    # No bare-relative or foreign-prefix references remain.
    assert 'src="assets/' not in res.text
    assert 'src="/static' not in res.text


def test_static_allowlist_hardening(client):
    for good in ("explorer.js", "explorer.css"):
        assert client.get(f"/viz/assets/{good}").status_code == 200
    assert client.get("/viz/assets/vendor/cytoscape.min.js").status_code in (200, 404)
    assert client.get("/viz/assets/vendor/cytoscape-dagre.min.js").status_code in (200, 404)
    # Path escape attempts are rejected.
    for bad in (
        "/viz/assets/secret.txt",
        "/viz/assets/../../pyproject.toml",
        "/viz/assets/vendor/evil.js",
        "/viz/assets/vendor/../../pyproject.toml",
    ):
        assert client.get(bad).status_code == 404, bad


def test_no_graph_is_a_409(tmp_path):
    empty = tmp_path / "empty-ws"
    empty.mkdir()
    res = TestClient(create_viz_app(empty)).get("/api/graph/package")
    assert res.status_code == 409
    assert "index graph" in res.json()["detail"]


def test_factory_registered_in_main_server_router_list():
    """create_all_routers must yield the viz routes (the main-server mount)."""
    from types import SimpleNamespace

    from victor.integrations.api.routes import create_all_routers

    stub = SimpleNamespace(workspace_root="/nonexistent")
    routers = create_all_routers(stub)
    paths = {getattr(r, "path", None) for r in routers for r in getattr(r, "routes", [])}
    assert "/viz" in paths
    assert "/api/graph/package" in paths
    assert "/api/graph/flow" in paths
    assert "/api/graph/classdiagram" in paths


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


# --- Call-flow sequence view (ADR-033 phase 2) ------------------------------


def _labels(result):
    return [m["label"] for m in result["messages"]]


def test_flow_orders_messages_by_call_line(db_path):
    result = viz_queries.flow(db_path, "execute", depth=1)
    assert result["ordered"] is True
    assert result["entry"]["name"] == "execute"
    first_caller = [m for m in result["messages"] if m["depth"] == 1]
    assert _labels(result) == ["«new» Repo", "_step", "serialize", "helper"]
    assert [m["call_line"] for m in first_caller] == [20, 40, 60, 70]
    assert "sequenceDiagram" in result["mermaid"]
    assert "«new» Repo" in result["mermaid"]
    assert 'as "TurnService"' in result["mermaid"]


def test_flow_class_granularity_participants(db_path):
    result = viz_queries.flow(db_path, "execute", depth=2)
    labels = {p["label"] for p in result["participants"].values()}
    assert "TurnService" in labels  # methods lane onto their class
    assert "Repo" in labels  # constructor target is its own lane
    assert "util.py" in labels  # free functions lane onto their module


def test_flow_cycle_and_recursion_are_guarded(db_path):
    result = viz_queries.flow(db_path, "execute", depth=4)
    revisits = [m for m in result["messages"] if m["revisit"]]
    assert revisits, "the execute -> _step -> execute cycle must be a revisit"
    # Recursion (helper -> helper) appears exactly once and terminates.
    self_calls = [m for m in result["messages"] if m["label"] == "helper" and m["revisit"]]
    assert len(self_calls) == 1
    assert len(result["messages"]) < 20, "guards must bound the expansion"


def test_flow_without_call_lines_is_honestly_unordered(db_path):
    result = viz_queries.flow(db_path, "plain", depth=1)
    assert result["ordered"] is False
    # call_line order (91) beats the no-line edge; the no-line edge follows.
    assert _labels(result) == ['bad<svg>"onload"&', "serialize"]


def test_flow_sanitizes_hostile_symbol_names(db_path):
    """Symbol names are untrusted text; mermaid output must never carry raw
    markup (angle brackets, quotes, ampersands) through to the renderer."""
    result = viz_queries.flow(db_path, "plain", depth=1)
    assert "<svg>" not in result["mermaid"]
    assert 'bad"' not in result["mermaid"]
    assert "&" not in result["mermaid"] or "&amp;" in result["mermaid"]
    # JSON keeps the raw name (data fidelity); the UI escapes at the DOM
    # boundary — the renderer-facing mermaid source is what must be clean.
    assert any("<svg>" in m["label"] for m in result["messages"])


def test_flow_ambiguity_returns_candidates(db_path):
    result = viz_queries.flow(db_path, "handle")
    assert "candidates" in result
    assert {c["file"] for c in result["candidates"]} == {
        "victor/agent/services/turn.py",
        "victor/core/repo.py",
    }
    pinned = viz_queries.flow(db_path, "handle", node_id="repo_handle")
    assert pinned["entry"]["file"] == "victor/core/repo.py"


def test_flow_unknown_symbol_has_no_entry(db_path):
    assert viz_queries.flow(db_path, "does_not_exist") == {"entry": None}


def test_flow_fanout_cap_reports_truncation(db_path):
    result = viz_queries.flow(db_path, "execute", depth=1, max_fanout=2)
    assert result["truncated"] is True
    assert result["pruned"] == 2  # 4 callees, cap 2 — reported, never silent


def test_flow_truncation_is_reported(db_path):
    result = viz_queries.flow(db_path, "execute", depth=3, max_messages=2)
    assert result["truncated"] is True
    assert result["pruned"] > 0
    assert len(result["messages"]) == 2
    assert "truncated" in result["mermaid"]


def test_flow_endpoint_happy_path_and_errors(client):
    ok = client.get("/api/graph/flow", params={"symbol": "execute", "depth": "2"})
    assert ok.status_code == 200
    body = ok.json()
    assert body["entry"]["name"] == "execute"
    assert body["mermaid"].startswith("sequenceDiagram")

    missing = client.get("/api/graph/flow", params={"symbol": "nope"})
    assert missing.status_code == 404

    ambiguous = client.get("/api/graph/flow", params={"symbol": "handle"})
    assert ambiguous.status_code == 200
    assert len(ambiguous.json()["candidates"]) == 2

    pinned = client.get("/api/graph/flow", params={"symbol": "handle", "node_id": "repo_handle"})
    assert pinned.json()["entry"]["file"] == "victor/core/repo.py"

    assert client.get("/api/graph/flow", params={"symbol": "x", "depth": "9"}).status_code == 422
    assert (
        client.get("/api/graph/flow", params={"symbol": "x", "granularity": "both"})
    ).status_code == 422


def test_flow_409_without_graph(tmp_path):
    empty = tmp_path / "empty-ws"
    empty.mkdir()
    res = TestClient(create_viz_app(empty)).get("/api/graph/flow", params={"symbol": "x"})
    assert res.status_code == 409


def test_explorer_assets_are_parseable():
    """The explorer JS must survive a real engine parse — phase 1 shipped a
    strict-mode reserved word (`package` as a parameter) that curl smoke
    tests cannot catch. Skip when node is unavailable."""
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node not available")
    asset = (
        Path(viz_queries.__file__).parent.parent.parent
        / "integrations"
        / "api"
        / "static"
        / "viz"
        / "explorer.js"
    )
    proc = subprocess.run(["node", "--check", str(asset)], capture_output=True, text=True)
    assert proc.returncode == 0, f"explorer.js failed to parse:\n{proc.stderr}"


# --- Class diagrams (ADR-033 diagram families) ------------------------------


def test_class_diagram_hierarchy_direction(db_path):
    """INHERITS is subclass -> base; the diagram must put bases above."""
    result = viz_queries.class_diagram(db_path, "SessionService")
    assert result["entry"]["name"] == "SessionService"
    assert [b["name"] for b in result["bases"]] == ["BaseService"]
    assert [c["name"] for c in result["subclasses"]] == ["CachedSessionService"]
    assert "BaseService <|-- SessionService" in result["mermaid"]
    assert "SessionService <|-- CachedSessionService" in result["mermaid"]


def test_class_diagram_members_and_uses(db_path):
    result = viz_queries.class_diagram(db_path, "SessionService")
    assert [m["name"] for m in result["members"]] == ["start", "stop"]
    # start uses Metrics twice + Repo once -> Metrics ranks first; the focal
    # class never appears as its own associate.
    assert [a["name"] for a in result["associates"]] == ["Metrics", "Repo"]
    assert "start(session - Session)" in result["mermaid"]
    assert "--> Metrics : uses" in result["mermaid"]
    related = {r["name"]: r["relation"] for r in result["related"]}
    assert related == {
        "BaseService": "base",
        "CachedSessionService": "subclass",
        "Metrics": "associate",
        "Repo": "associate",
    }


def test_class_diagram_member_cap_is_reported(db_path):
    result = viz_queries.class_diagram(db_path, "SessionService", max_members=1)
    assert result["members_truncated"] is True
    assert result["members_total"] == 2
    assert len(result["members"]) == 1
    assert "2 members - showing first 1" in result["mermaid"]


def test_class_diagram_route_and_errors(client):
    ok = client.get("/api/graph/classdiagram", params={"symbol": "SessionService"})
    assert ok.status_code == 200
    assert ok.json()["mermaid"].startswith("classDiagram")

    # "handle" is a method, not a class -> unknown at the class surface.
    assert client.get("/api/graph/classdiagram", params={"symbol": "handle"}).status_code == 404
    assert (
        client.get("/api/graph/classdiagram", params={"symbol": "x", "max_members": "1"})
    ).status_code == 422
    pinned = client.get(
        "/api/graph/classdiagram", params={"symbol": "SessionService", "node_id": "svc2_cls"}
    )
    assert pinned.json()["entry"]["file"] == "victor/agent/services/session.py"
    # Pinning a non-class id is rejected like flow's callable-only pin.
    bad_pin = client.get(
        "/api/graph/classdiagram", params={"symbol": "x", "node_id": "svc_execute"}
    )
    assert bad_pin.status_code == 404


def test_class_diagram_sanitizes_names(db_path):
    """Hostile names must not reach the mermaid source."""
    result = viz_queries.class_diagram(db_path, "SessionService")
    assert "<svg>" not in result["mermaid"]


# --- Polyglot representation (monorepo trees in the package view) ---------


def _seed_polyglot(tmp_path: Path) -> Path:
    """A tiny monorepo: python coupling (visible by weight), a rust tree
    coupling across crates (visible only at per-tree depth + partner
    pull-in), and a vscode-like tree with purely internal coupling
    (visible only by activity admission)."""
    import asyncio

    from victor.storage.graph.protocol import GraphEdge, GraphNode
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    def node(node_id, name, file, type_, line=1, parent_id=None):
        return GraphNode(
            node_id=node_id,
            type=type_,
            name=name,
            file=file,
            line=line,
            end_line=line + 1,
            lang="python",
            parent_id=parent_id,
        )

    nodes = [
        node("a1", "alpha", "victor/agent/alpha.py", "function"),
        node("a2", "beta", "victor/core/beta.py", "function"),
        node("r1", "rust_one", "rust/crates/edge-runtime/src/a.rs", "function"),
        node("r2", "rust_two", "rust/crates/edge-runtime/src/b.rs", "function"),
        node("r3", "rust_three", "rust/crates/protocol/src/c.rs", "function"),
        node("r4", "rust_four", "rust/crates/tools/src/d.rs", "function"),
        node("v1", "vs_one", "vscode-victor/src/one.ts", "function"),
        node("v2", "vs_two", "vscode-victor/src/two.ts", "function"),
    ]
    edges = [
        GraphEdge(src="a1", dst="a2", type="CALLS"),  # python coupling
        GraphEdge(src="r1", dst="r2", type="CALLS"),  # rust same-crate
        GraphEdge(src="r1", dst="r3", type="CALLS"),  # rust cross-crate
        GraphEdge(src="r3", dst="r4", type="CALLS"),  # rust cross-crate
        GraphEdge(src="v1", dst="v2", type="CALLS"),  # vscode internal only
    ]
    store = SqliteGraphStore(tmp_path)
    asyncio.run(store.initialize())
    asyncio.run(store.upsert_nodes(nodes))
    asyncio.run(store.upsert_edges(edges))
    asyncio.run(store.close())
    return tmp_path / ".victor" / "project.db"


def test_package_view_represents_all_trees(db_path, tmp_path):
    """Every top-level tree appears — including a rust tree whose coupling
    is cross-CRATE (needs depth-3 + partner pull-in) and a vscode-like
    tree with purely INTERNAL coupling (needs activity admission)."""
    poly = _seed_polyglot(tmp_path)
    result = viz_queries.package_graph(poly, top_packages=2)
    ids = {n["data"]["id"] for n in result["nodes"]}
    assert any(i.startswith("rust/crates/") for i in ids), f"rust missing: {ids}"
    assert any(i.startswith("vscode-victor") for i in ids), f"vscode missing: {ids}"
    # Cross-crate edges render when both crates are present.
    rust_edges = [
        e
        for e in result["edges"]
        if "rust/crates" in e["data"]["source"] or "rust/crates" in e["data"]["target"]
    ]
    assert rust_edges, "rust cross-crate edges must render"
    # Internally-coupled-only packages carry an honest internal degree.
    vs = next(n["data"] for n in result["nodes"] if n["data"]["id"].startswith("vscode"))
    assert vs["degree"] >= 1


def test_package_view_never_labels_generic_components(db_path, tmp_path):
    poly = _seed_polyglot(tmp_path)
    result = viz_queries.package_graph(poly, top_packages=2)
    labels = {n["data"]["label"] for n in result["nodes"]}
    assert "src" not in labels, "generic component must not display as label"


def test_internal_activity_packages_ranks_unrepresented(tmp_path):
    poly = _seed_polyglot(tmp_path)
    conn = viz_queries._connect(poly)
    try:
        ranked = viz_queries._internal_activity_packages(conn, exclude=set())
    finally:
        conn.close()
    assert ranked, "activity ranking must see the seeded graph"
    names = [p for p, _ in ranked]
    assert any(p.startswith("vscode-victor") for p in names)
    assert any(p.startswith("rust/crates") for p in names)
