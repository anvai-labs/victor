from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from victor.core.graph_rag.codegraph_repository import (
    prepare_repository_snapshot,
    project_resolved_relations,
)


def test_project_relations_requires_both_endpoints_to_be_materialized() -> None:
    """A repository snapshot is broader than the persisted Graph-RAG graph.

    Parser symbols from a file whose v2 identities did not match the nodes
    produced by the active language provider must not leak dangling edges into
    permissive stores such as SQLite (ORION rejects the same write).
    """
    relation_type = SimpleNamespace(name="CALLS")
    repository = SimpleNamespace(
        relations=[
            SimpleNamespace(
                from_symbol_id="persisted-a",
                to_symbol_id="persisted-b",
                target_ref=None,
                provenance="repository_resolver",
                relation_type=relation_type,
            ),
            SimpleNamespace(
                from_symbol_id="snapshot-only-a",
                to_symbol_id="snapshot-only-b",
                target_ref=None,
                provenance="repository_resolver",
                relation_type=relation_type,
            ),
        ]
    )

    projection = project_resolved_relations(
        repository,
        materialized_symbol_ids={"persisted-a", "persisted-b"},
    )

    assert projection.edges == (("persisted-a", "persisted-b", "CALLS"),)
    assert projection.calls == 1
    assert projection.dropped_unmaterialized == 1


def test_prepare_repository_snapshot_filters_non_indexed_files(tmp_path: Path) -> None:
    pytest.importorskip("victor_codegraph")
    (tmp_path / "included.py").write_text("def included(): pass\n", encoding="utf-8")
    (tmp_path / "excluded.py").write_text("def excluded(): pass\n", encoding="utf-8")

    repository = prepare_repository_snapshot(tmp_path, ["included.py"])

    assert repository is not None
    assert set(repository.files) == {"included.py"}
    assert {symbol.location.file_path for symbol in repository.symbols} == {"included.py"}
    emitted_ids = {symbol.id for symbol in repository.symbols}
    assert all(
        relation.from_symbol_id in emitted_ids
        and (relation.target_ref is not None or relation.to_symbol_id in emitted_ids)
        for relation in repository.relations
    )


def test_prepare_repository_snapshot_propagates_unexpected_failure(
    monkeypatch, tmp_path: Path
) -> None:
    codegraph = pytest.importorskip("victor_codegraph")

    def _raise(_root):
        raise RuntimeError("parser unavailable")

    monkeypatch.setattr(codegraph, "parse_repo", _raise)

    with pytest.raises(RuntimeError, match="parser unavailable"):
        prepare_repository_snapshot(tmp_path, [])


@pytest.mark.asyncio
async def test_index_snapshot_preserves_symbols_edges_and_opaque_lookup(tmp_path, monkeypatch):
    from victor_codegraph import parse_repo
    from victor_codegraph import treesitter_parser
    from tree_sitter import Language, Parser
    import tree_sitter_typescript

    # Use the installed grammar, independent of HOME-scoped download caches.
    original_parser = treesitter_parser._get_parser
    monkeypatch.setattr(
        treesitter_parser,
        "_get_parser",
        lambda grammar: (
            Parser(Language(tree_sitter_typescript.language_tsx()))
            if grammar == "tsx"
            else original_parser(grammar)
        ),
    )
    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.storage.graph.sqlite_store import SqliteGraphStore
    from victor.ui.commands.graph import _resolve_target

    (tmp_path / "a.py").write_text("def target(): return 1\n")
    (tmp_path / "b.py").write_text(
        "from a import target\n@decorate\ndef caller():\n target()\n target()\n"
    )
    (tmp_path / "c.py").write_text("def target(): return 2\n")
    (tmp_path / "widget.tsx").write_text("export function Widget() { return <div/>; }")
    from unittest.mock import AsyncMock

    # This fixture owns no vectors; isolate graph persistence from model downloads.
    monkeypatch.setattr(SqliteGraphStore, "_delete_embeddings_for_file", AsyncMock())
    store = SqliteGraphStore(tmp_path)
    await store.initialize()
    try:
        config = GraphIndexConfig(
            root_path=tmp_path,
            enable_ccg=False,
            enable_embeddings=False,
            enable_subgraph_cache=False,
            enable_module_metrics=False,
        )
        pipeline = GraphIndexingPipeline(store, config)
        stats = await pipeline.index_repository()
        assert not stats.errors, stats.errors
        shared = parse_repo(tmp_path)
        nodes = await store.get_all_nodes()
        assert {n.node_id for n in nodes} == {s.id for s in shared.symbols}, [
            (s.simple_name, s.location.file_path)
            for s in shared.symbols
            if s.id not in {n.node_id for n in nodes}
        ]
        assert stats.repository_relations_dropped_unmaterialized == 0
        projected = project_resolved_relations(shared)
        stored_edges = {(e.src, e.dst, e.type) for e in await store.get_all_edges()}
        assert projected.edges and set(projected.edges) <= stored_edges
        target = next(s for s in shared.symbols if s.simple_name == "target")
        assert await _resolve_target(target.id, store) == target.id
        caller = next(s for s in shared.symbols if s.simple_name == "caller")
        assert await _resolve_target("caller", store) == caller.id
        assert await _resolve_target("target", store) is None  # ambiguous names are not IDs
        counts = await store.stats()
        assert stats.persisted_nodes == counts["nodes"]
        assert stats.persisted_edges == counts["edges"]
        assert stats.files_processed == len(shared.files)
    finally:
        await store.close()
