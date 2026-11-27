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

    assert [(e.src, e.dst, e.type) for e in projection.edges] == [
        ("persisted-a", "persisted-b", "CALLS")
    ]
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

    def _raise(_root, **kwargs):
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
        assert projected.edges and {(e.src, e.dst, e.type) for e in projected.edges} <= stored_edges
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


def test_canonical_projection_carries_body_hash_even_when_summary_is_unchanged():
    import hashlib
    from victor_codegraph import parse
    from victor.core.graph_rag.codegraph_repository import project_file_symbols

    before = parse("def f(): return 1", file_path="a.py")
    after = parse("def f(): return 2", file_path="a.py")
    old, new = project_file_symbols(before)[0], project_file_symbols(after)[0]
    assert old.node_id == new.node_id
    assert old.signature == new.signature
    assert old.metadata["source_hash"] != new.metadata["source_hash"]
    assert (
        new.metadata["source_hash"]
        == hashlib.sha256(after.symbols[0].source_code.encode()).hexdigest()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("incremental", [True, False])
async def test_collision_file_keeps_unique_symbols_without_ambiguous_edges(
    tmp_path, monkeypatch, incremental
):
    """A collision leaves its own declarations out, not their file: the file's
    unique symbols are indexed, and no call to the ambiguous name is guessed,
    in that file, from its importer, or through a namesake elsewhere."""
    from unittest.mock import AsyncMock, Mock

    from victor_codegraph import parse_repo
    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    (tmp_path / "ambiguous.py").write_text(
        "def duplicate(): return 1\ndef duplicate(): return 2\n"
        "def otherwise_valid(): return duplicate()\n"
    )
    (tmp_path / "normal.py").write_text(
        "from ambiguous import duplicate\ndef consumer(): return duplicate()\n"
    )
    (tmp_path / "twin.py").write_text("def duplicate(): return 3\n")
    shared = parse_repo(tmp_path)
    monkeypatch.setattr(SqliteGraphStore, "_clear_all_embeddings", AsyncMock())
    collision = shared.files["ambiguous.py"]
    duplicate_ids = {s.id for s in collision.symbols if s.simple_name == "duplicate"}
    marker_id = next(
        s.id
        for s in shared.symbols
        if s.symbol_type.name == "FILE" and s.location.file_path == "ambiguous.py"
    )
    valid_id = next(s.id for s in shared.symbols if s.simple_name == "otherwise_valid")
    consumer_id = next(s.id for s in shared.symbols if s.simple_name == "consumer")
    twin_id = next(
        s.id
        for s in shared.symbols
        if s.location.file_path == "twin.py" and s.simple_name == "duplicate"
    )
    monkeypatch.setattr(SqliteGraphStore, "_delete_embeddings_for_file", AsyncMock())
    store = SqliteGraphStore(tmp_path)
    await store.initialize()
    try:
        pipeline = GraphIndexingPipeline(
            store,
            GraphIndexConfig(
                root_path=tmp_path,
                enable_ccg=True,
                incremental=incremental,
                enable_embeddings=False,
                enable_subgraph_cache=False,
                enable_module_metrics=False,
            ),
        )
        # Statement nodes do not depend on symbol identity, so every file gets them.
        ccg = Mock(return_value=([], []))
        monkeypatch.setattr(pipeline, "_build_ccg_inline", ccg)
        stats = await pipeline.index_repository()
        assert stats.error_count == 0, stats.errors
        assert {call.args[0] for call in ccg.call_args_list} == {
            tmp_path / "ambiguous.py",
            tmp_path / "normal.py",
            tmp_path / "twin.py",
        }
        assert stats.files_partial == 1 and stats.files_processed == 3
        nodes = await store.get_all_nodes()
        collision_nodes = [n for n in nodes if n.file == "ambiguous.py"]
        assert {n.node_id for n in collision_nodes} == {marker_id, valid_id}
        marker = next(n for n in collision_nodes if n.node_id == marker_id)
        assert marker.metadata["parse_status"] == "partial"
        assert marker.metadata["projection_status"] == "symbols_quarantined"
        assert marker.metadata["omitted_symbol_count"] == 2
        assert marker.metadata["omitted_relation_count"] == 0
        assert marker.metadata["ambiguous_symbol_names"] == ["duplicate"]
        assert marker.metadata["diagnostics"][0]["code"] == "identity_collision"
        details = marker.metadata["identity_collisions"]
        assert {d["symbol_id"] for d in details} == duplicate_ids
        assert [d["source_span"]["start_line"] for d in details[0]["declarations"]] == [1, 2]
        assert not duplicate_ids.intersection(n.node_id for n in nodes)
        edges = await store.get_all_edges()
        assert not any(e.src in duplicate_ids or e.dst in duplicate_ids for e in edges)
        assert not any(e.type == "CALLS" for e in edges)
        assert not any(e.type == "CALLS" and e.dst == twin_id for e in edges)

        # Repairing the source clears the quarantine and restores canonical edges.
        (tmp_path / "ambiguous.py").write_text("def duplicate(): return 1\n")
        recovered = await pipeline.index_repository()
        assert recovered.error_count == 0 and recovered.files_partial == 0
        nodes = await store.get_all_nodes()
        marker = next(n for n in nodes if n.node_id == marker_id)
        assert marker.metadata["projection_status"] == "symbols"
        assert not marker.metadata["diagnostics"]
        target = next(n for n in nodes if n.name == "duplicate" and n.file == "ambiguous.py")
        # The repaired file declares `duplicate` alone, so only the importer's
        # call is left to bind, and it binds to this file, not to the twin.
        calls = [e for e in await store.get_all_edges() if e.type == "CALLS"]
        assert {(e.src, e.dst) for e in calls} == {(consumer_id, target.node_id)}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_survivor_of_a_quarantined_path_gets_no_guessed_call(tmp_path, monkeypatch):
    """Victor graph: `class f` survives two colliding `def f`; an importer's call
    to `f` must not become a CALLS edge to it."""
    from unittest.mock import AsyncMock

    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    (tmp_path / "x.py").write_text(
        "class f:\n    pass\n\n\ndef f():\n    return 1\n\n\ndef f():\n    return 2\n"
    )
    (tmp_path / "y.py").write_text("from x import f\n\n\ndef g():\n    return f()\n")
    monkeypatch.setattr(SqliteGraphStore, "_clear_all_embeddings", AsyncMock())
    monkeypatch.setattr(SqliteGraphStore, "_delete_embeddings_for_file", AsyncMock())
    store = SqliteGraphStore(tmp_path)
    await store.initialize()
    try:
        pipeline = GraphIndexingPipeline(
            store,
            GraphIndexConfig(
                root_path=tmp_path,
                enable_ccg=False,
                enable_embeddings=False,
                enable_subgraph_cache=False,
                enable_module_metrics=False,
            ),
        )
        stats = await pipeline.index_repository()
        assert stats.error_count == 0, stats.errors
        nodes = await store.get_all_nodes()
        survivor = next(n for n in nodes if n.file == "x.py" and n.name == "f")
        assert survivor.type == "class"
        edges = await store.get_all_edges()
        assert not any(e.type == "CALLS" and e.dst == survivor.node_id for e in edges)
    finally:
        await store.close()


def test_prepared_identity_tracks_selected_content_and_intent_only(tmp_path):
    from victor_codegraph import parse_repo
    from victor.core.graph_rag.codegraph_repository import qualify_repository_snapshot

    selected = tmp_path / "selected.py"
    excluded = tmp_path / "excluded.py"
    selected.write_text("def selected(): return 1\n")
    excluded.write_text("x = 1\n")
    repository = parse_repo(tmp_path)
    source_id = repository.generation_id
    symbol_ids = [s.id for s in repository.symbols]
    settings = {"enable_ccg": False, "embedding_profile": None}
    versions = {"python": "3.12", "victor-codegraph": "test-version"}

    def qualify(repo, **overrides):
        return qualify_repository_snapshot(
            repo,
            ["selected.py"],
            settings=overrides.get("settings", settings),
            parser_versions=overrides.get("parser_versions", versions),
        )

    original = qualify(repository)
    assert original.status == "complete" and original.prepared_id
    assert qualify(repository, settings=dict(reversed(list(settings.items())))) == original
    assert qualify(repository, parser_versions=dict(reversed(list(versions.items())))) == original
    excluded.write_text("x = 2\n")
    assert qualify(parse_repo(tmp_path)) == original
    selected.write_text("def selected(): return 2\n")
    assert qualify(parse_repo(tmp_path)).prepared_id != original.prepared_id
    assert (
        qualify(repository, settings={**settings, "enable_ccg": True}).prepared_id
        != original.prepared_id
    )
    assert (
        qualify(repository, parser_versions={**versions, "python": "3.13"}).prepared_id
        != original.prepared_id
    )
    assert repository.generation_id == source_id
    assert [s.id for s in repository.symbols] == symbol_ids


@pytest.mark.parametrize(
    "defect", ["missing_file", "missing_manifest", "hash_mismatch", "error", "fallback", "partial"]
)
def test_snapshot_qualification_distinguishes_invalid_and_partial_inputs(tmp_path, defect):
    from dataclasses import replace
    from victor_codegraph import parse_repo, ParseStatus
    from victor.core.graph_rag.codegraph_repository import qualify_repository_snapshot

    (tmp_path / "a.py").write_text("x = 1\n")
    repository = parse_repo(tmp_path)
    if defect == "missing_file":
        repository.files.clear()
    elif defect == "missing_manifest":
        repository.manifest.clear()
    elif defect == "hash_mismatch":
        repository.manifest["a.py"] = "wrong"
    else:
        repository.files["a.py"] = replace(repository.files["a.py"], status=ParseStatus(defect))
    result = qualify_repository_snapshot(
        repository, ["a.py"], settings={}, parser_versions={"parser": "1"}
    )
    assert result.selected_count == 1
    if defect == "partial":
        assert result.status == "partial"
        assert result.partial_files == ("a.py",) and result.prepared_id
    else:
        assert result.status == "failed" and result.prepared_id is None
        assert (result.missing_files or result.failed_files) == ("a.py",)


def test_missing_optional_parser_or_version_never_invents_prepared_identity(tmp_path, monkeypatch):
    from victor_codegraph import parse_repo
    from victor.core.graph_rag import codegraph_repository as boundary
    from importlib.metadata import PackageNotFoundError

    (tmp_path / "a.py").write_text("x = 1\n")

    def unavailable(package):
        raise PackageNotFoundError(package)

    monkeypatch.setattr(boundary.metadata, "version", unavailable)
    repository = parse_repo(tmp_path)
    versions = boundary.snapshot_parser_versions(repository)
    assert versions["victor-codegraph"] is None and versions["python"]
    qualified = boundary.qualify_repository_snapshot(
        repository, ["a.py"], settings={}, parser_versions=versions
    )
    assert qualified.status == "complete" and qualified.prepared_id is None
    legacy = boundary.qualify_repository_snapshot(
        None, ["a.py"], settings={}, parser_versions=versions
    )
    assert legacy.status == "legacy-unqualified" and legacy.prepared_id is None


@pytest.mark.parametrize("missing", ["victor_codegraph", "parser_dependency", None])
def test_optional_snapshot_absence_is_distinct_from_broken_install(tmp_path, monkeypatch, missing):
    import builtins

    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "victor_codegraph":
            if missing is None:
                raise ImportError("installed parser export is broken")
            raise ModuleNotFoundError("unavailable", name=missing)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    if missing == "victor_codegraph":
        assert prepare_repository_snapshot(tmp_path, []) is None
    else:
        with pytest.raises(ImportError):
            prepare_repository_snapshot(tmp_path, [])
