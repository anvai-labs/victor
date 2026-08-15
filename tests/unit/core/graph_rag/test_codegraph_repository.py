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


def test_prepare_repository_snapshot_degrades_softly(monkeypatch, tmp_path: Path) -> None:
    codegraph = pytest.importorskip("victor_codegraph")

    def _raise(_root):
        raise RuntimeError("parser unavailable")

    monkeypatch.setattr(codegraph, "parse_repo", _raise)

    assert prepare_repository_snapshot(tmp_path, []) is None
