"""CLI previews are read-only; apply backs up WAL state before graph repair."""

import sqlite3
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from victor.ui.commands.graph import graph_app
from victor.storage.graph.sqlite_store import _TABLE_SCHEMA, FTS_SCHEMA


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "repo"
    directory = root / ".victor"
    directory.mkdir(parents=True)
    db = sqlite3.connect(directory / "project.db")
    db.executescript("PRAGMA journal_mode=WAL;" + _TABLE_SCHEMA + FTS_SCHEMA)
    db.executemany(
        "INSERT INTO graph_node (node_id,type,name,file,line,end_line,signature) VALUES (?,'function','run','a.py',1,2,'def run():')",
        [("a",), ("b",)],
    )
    db.commit()
    yield root, db
    db.close()


def test_preview_does_not_mutate_or_backup(project):
    root, db = project
    before = list(db.iterdump())
    result = CliRunner().invoke(graph_app, ["dedup", "--path", str(root), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Duplicate groups: 1; removable nodes: 1" in result.output
    assert list(db.iterdump()) == before
    assert not list((root / ".victor").glob("*.bak"))
    assert not (root / ".victor" / "index.lock").exists()


def test_apply_backs_up_before_merge_and_reports_refresh_failure(project, monkeypatch):
    root, db = project
    monkeypatch.setattr(
        "victor.core.graph_rag.dedup.update_embeddings",
        AsyncMock(side_effect=RuntimeError("offline")),
    )
    result = CliRunner().invoke(graph_app, ["dedup", "--path", str(root), "--apply"])
    assert result.exit_code == 1, result.output
    assert "Graph merges committed" in result.output
    assert "repair remains pending" in result.output
    assert db.execute("SELECT count(*) FROM graph_node").fetchone()[0] == 1
    backups = list((root / ".victor").glob("*.bak"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        assert backup.execute("SELECT count(*) FROM graph_node").fetchone()[0] == 2
    assert backups[0].stat().st_mode & 0o777 == 0o600
    refresh = AsyncMock(return_value=1)
    monkeypatch.setattr("victor.core.graph_rag.dedup.update_embeddings", refresh)
    retry = CliRunner().invoke(graph_app, ["dedup", "--path", str(root), "--apply"])
    assert retry.exit_code == 0, retry.output
    assert "Duplicate groups: 0" in retry.output
    refresh.assert_awaited_once_with(root)


def test_missing_database_and_non_sqlite_fail_without_creating_database(tmp_path):
    tmp_path = tmp_path / "missing"
    result = CliRunner().invoke(graph_app, ["dedup", "--path", str(tmp_path)])
    assert result.exit_code != 0
    assert not (tmp_path / ".victor").exists()
    (tmp_path / ".victor").mkdir(parents=True)
    (tmp_path / ".victor" / "graph_backend").write_text("proxima")
    result = CliRunner().invoke(graph_app, ["dedup", "--path", str(tmp_path), "--apply"])
    assert result.exit_code == 1
    assert "SQLite" in result.output
    assert not (tmp_path / ".victor" / "project.db").exists()


def test_contradictory_flags_fail_before_opening_database(tmp_path):
    tmp_path = tmp_path / "missing"
    result = CliRunner().invoke(
        graph_app, ["dedup", "--path", str(tmp_path), "--apply", "--dry-run"]
    )
    assert result.exit_code == 2
    assert not (tmp_path / ".victor").exists()


def test_codegraph_alias_and_canonical_command_invoke_same_operation(tmp_path, monkeypatch):
    from victor.ui.cli import app

    operation = AsyncMock()
    monkeypatch.setattr("victor.ui.commands.graph._dedup_async", operation)
    for prefix in (["codegraph"], ["index", "graph"]):
        result = CliRunner().invoke(app, [*prefix, "dedup", "--path", str(tmp_path), "--dry-run"])
        assert result.exit_code == 0, result.output
    assert operation.await_count == 2
    operation.assert_awaited_with(tmp_path.resolve(), False)
