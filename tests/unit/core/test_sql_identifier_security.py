"""Real SQLite regressions for dynamic identifier handling."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
import typer

from victor.core.database import DatabaseManager, ProjectDatabaseManager
from victor.core.repository import Entity, SQLiteRepository
from victor.core.sql_utils import quote_identifier
from victor.framework.observability.persistence import SQLiteMetricsStore
from victor.framework.rl.hierarchical_policy import HierarchicalPolicy
from victor.ui.commands.db import db_prune as prune


@pytest.mark.parametrize("table", ["select", 'odd"name]--', "items WHERE 1=1 --"])
async def test_repository_quotes_table_and_index_names(tmp_path, table):
    repo = SQLiteRepository(tmp_path / "repo.db", table, Entity)
    entity = Entity(id="one")
    await repo.add(entity)
    assert (await repo.get("one")).id == "one"
    assert await repo.count() == 1
    await repo.update(entity)
    assert await repo.exists("one")
    assert len(await repo.list()) == 1
    await repo.delete("one")
    assert await repo.count() == 0


@pytest.mark.parametrize("name", ["", "bad; --", "x UNION SELECT 1", "a.b", "a\0b"])
def test_learner_rejects_non_identifier_prefix_before_database_access(name):
    conn = MagicMock()
    with pytest.raises(ValueError):
        HierarchicalPolicy(name=name, db_connection=conn)
    assert not conn.mock_calls


def test_learner_accepts_custom_identifier():
    with sqlite3.connect(":memory:") as conn:
        learner = HierarchicalPolicy(name="custom_policy_2", db_connection=conn)
        assert learner.name == "custom_policy_2"
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='custom_policy_2_q_table'"
        ).fetchone()


@pytest.mark.parametrize("manager_class", [DatabaseManager, ProjectDatabaseManager])
def test_stats_quotes_names_from_database_schema(tmp_path, manager_class):
    # Bypass manager singleton/setup so these tests never touch a user's DB.
    manager = object.__new__(manager_class)
    manager.db_path = tmp_path / "stats.db"
    with sqlite3.connect(manager.db_path) as conn:
        table = 'odd"name]--'
        quoted = quote_identifier(table)
        conn.execute(f"CREATE TABLE {quoted} (created_at TEXT)")
        conn.execute(f"INSERT INTO {quoted} VALUES (?)", ("2026-09-13",))
        manager.get_tables = lambda: [table]
        manager.query_one = lambda sql: conn.execute(sql).fetchone()
        manager.get_connection = lambda: conn
        assert manager.get_stats()["tables"][table] == 1
        if manager_class is DatabaseManager:
            assert manager.get_table_stats()[0]["rows"] == 1
            assert manager.get_table_stats()[0]["min_date"] == "2026-09-13"


@pytest.mark.parametrize("existing", [False, True])
def test_legacy_migration_quotes_source_and_table(tmp_path, existing):
    table = 'odd"name]--'
    quoted = quote_identifier(table)
    source = tmp_path / "legacy.db"
    with sqlite3.connect(source) as conn:
        conn.execute(f"CREATE TABLE {quoted} (id INTEGER)")
        conn.execute(f"INSERT INTO {quoted} VALUES (42)")
    manager = object.__new__(DatabaseManager)
    with sqlite3.connect(":memory:") as target:
        if existing:
            target.execute(f"CREATE TABLE {quoted} (id INTEGER)")
        manager.table_exists = lambda name: bool(
            target.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone()
        )
        assert manager._migrate_from_legacy(target, 'odd"source', source) == 1
        assert target.execute(f"SELECT id FROM {quoted}").fetchone() == (42,)


@pytest.mark.parametrize("dry_run", [False, True])
def test_prune_rejects_unknown_table_before_count(dry_run):
    db = MagicMock()
    db.get_tables_for_group.return_value = ["rl_outcome"]
    with patch("victor.core.database.get_database", return_value=db):
        with pytest.raises(typer.Exit):
            prune(
                older_than=None,
                table="rl_outcome] WHERE 1=1 --",
                group=None,
                keep_last=1,
                dry_run=dry_run,
                yes=True,
            )
    db.get_connection.assert_not_called()
    db.prune_table.assert_not_called()


async def test_metrics_groups_are_values_and_unknown_fields_are_rejected():
    store = SQLiteMetricsStore()
    try:
        store._conn.executemany(
            "INSERT INTO agent_metrics (agent_id, session_id, created_at, total_input_tokens) VALUES (?,?,?,?)",
            [("a", "one", -1, 10), ("a", "one", 0, 2), ("b", "two", 1, 3)],
        )
        with pytest.raises(ValueError):
            await store.aggregate("total_tokens", group_by=["agent_id) FROM sqlite_master --"])
        rows = await store.aggregate("total_input_tokens", group_by=["agent_id"], start_time=0)
        assert {row["group_key"]: row["total"] for row in rows} == {"a": 2, "b": 3}
        rows = await store.aggregate(
            "total_tokens", group_by=["agent_id", "session_id"], end_time=0
        )
        assert rows[0]["group_key"] == '["a","one"]'
        assert rows[0]["total"] == 12
        assert (await store.aggregate("total_tokens"))[0]["total"] == 15
    finally:
        store._conn.close()
