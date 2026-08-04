from __future__ import annotations

from pathlib import Path

import pytest

from victor.core.graph_rag.codegraph_repository import prepare_repository_snapshot


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
