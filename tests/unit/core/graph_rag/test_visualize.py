"""Hierarchical coupling visualizations (``victor.core.graph_rag.visualize``).

Deterministic query -> mermaid functions over the SQLite project store;
golden-content assertions plus the render path with and without mmdc.
"""

from __future__ import annotations

import pytest

from victor.core.graph_rag import visualize


async def _seed_store(tmp_path):
    from victor.storage.graph.protocol import GraphEdge, GraphNode
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    def node(node_id, name, file, type_):
        return GraphNode(
            node_id=node_id, type=type_, name=name, file=file, line=1, end_line=2, lang="python"
        )

    nodes = [
        node("orch", "AgentOrchestrator", "victor/agent/orchestrator.py", "class"),
        node("chat", "chat", "victor/agent/chat_service.py", "function"),
        node("base", "Settings", "victor/config/settings.py", "class"),
        node("msg", "Message", "victor/providers/base.py", "class"),
    ]
    edges = [
        GraphEdge(src="orch", dst="base", type="CALLS"),
        GraphEdge(src="chat", dst="orch", type="CALLS"),
        GraphEdge(src="chat", dst="base", type="CALLS"),
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
    from victor.core.async_utils import run_sync

    return run_sync(_seed_store(tmp_path))


def test_package_view_lists_heaviest_couplings_deterministically(db_path):
    mermaid = visualize.package_coupling_mermaid(db_path)
    assert visualize.package_coupling_mermaid(db_path) == mermaid  # deterministic
    assert "graph LR" in mermaid
    # chat -> config (2 edges) is the heaviest coupling and leads the edges.
    assert "victor_agent --2--> victor_config" in mermaid
    assert "victor_agent --1--> victor_providers" in mermaid
    # agent <-> (nothing back) but chat<->orch mutual REFERENCES forms a
    # cross-package cycle island? No — both files are IN victor/agent; the
    # cycle class appears only for real cross-package cycles.
    assert "classDef cycle" in mermaid


def test_module_view_scopes_to_one_package(db_path):
    mermaid = visualize.module_coupling_mermaid(db_path, "victor/agent")
    assert "orchestrator.py" in mermaid
    assert "chat_service.py" in mermaid
    # Cross-package edges (agent -> config/providers) are out of scope.
    assert "victor_config" not in mermaid
    assert "settings.py" not in mermaid


def test_class_view_renders_symbols_of_one_module(db_path):
    mermaid = visualize.class_graph_mermaid(db_path, "victor/agent/chat_service.py")
    assert "chat" in mermaid
    # Symbols from other modules are out of scope.
    assert "AgentOrchestrator" not in mermaid


def test_render_without_mmdc_writes_mmd_only(db_path, tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = visualize.render("graph LR\n  a-->b\n", tmp_path / "viz" / "out.mmd")
    assert result.mmd_path.read_text().startswith("graph LR")
    assert result.svg_path is None


def test_render_with_mmdc_produces_svg(tmp_path):
    import shutil

    if shutil.which("mmdc") is None:
        pytest.skip("mmdc not installed")
    result = visualize.render("graph LR\n  a-->b\n", tmp_path / "out.mmd")
    assert result.svg_path is not None and result.svg_path.exists()


def test_cli_writes_package_view_and_errors_honestly(tmp_path, db_path, monkeypatch):
    from unittest.mock import patch

    from rich.console import Console

    from victor.ui.commands import graph

    project_root = db_path.parents[1]
    output = Console(record=True, width=160)
    with patch.object(graph, "console", output):
        # Direct calls must pass level explicitly — typer defaults are
        # OptionInfo objects outside a real CLI invocation.
        graph.graph_viz(
            level="package", top=12, path=str(project_root), out=tmp_path / "p.mmd"
        )
        assert (tmp_path / "p.mmd").read_text().startswith("%%")
        assert "✓ package view" in output.export_text()

    err_console = Console(record=True, width=160)
    with patch.object(graph, "console", err_console):
        with pytest.raises(Exception):
            graph.graph_viz(
                level="module", package=None, module=None, top=12, path=str(project_root)
            )
        assert "requires --package" in err_console.export_text()
        with pytest.raises(Exception):
            graph.graph_viz(
                level="bogus", package=None, module=None, top=12, path=str(project_root)
            )
        assert "Unknown level" in err_console.export_text()
