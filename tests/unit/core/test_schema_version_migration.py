# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Migration authority: scoped upgrades, atomic publication and legacy preservation.

Modern unversioned lazy stores retain their existing adoption policy. Explicitly
versioned/legacy stores must fail closed on incompatible schema or SQL errors.
"""

from __future__ import annotations

import logging
import sqlite3

import pytest

from victor.core.database import DatabaseManager, ProjectDatabaseManager, _apply_migration_sql
from victor.core.schema import CURRENT_SCHEMA_VERSION, Schema


def _manager() -> DatabaseManager:
    # Build an uninitialised instance so we can call the migration method
    # directly against a connection we control (avoids the ~/.victor singleton).
    return object.__new__(DatabaseManager)


def _schema_version(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT value FROM sys_metadata WHERE key = 'schema_version'").fetchone()
    return row[0] if row else None


def test_modern_db_marks_version_without_replaying(caplog):
    """A modern DB (tables but no version row, no legacy sentinel) is marked
    at the current version without replaying rename migrations."""
    conn = sqlite3.connect(":memory:")
    conn.execute(Schema.SYS_METADATA)
    conn.execute("CREATE TABLE rl_outcome (id INTEGER PRIMARY KEY)")  # a lazily-created table

    with caplog.at_level(logging.WARNING):
        _manager()._run_schema_version_migrations(conn)

    assert _schema_version(conn) == str(CURRENT_SCHEMA_VERSION)
    assert "Legacy database detected" not in caplog.text
    assert "Migration SQL" not in caplog.text  # no replay → no failed-statement noise


def test_modern_db_migration_is_idempotent():
    """Re-running against an already-versioned DB is a silent no-op."""
    conn = sqlite3.connect(":memory:")
    conn.execute(Schema.SYS_METADATA)
    conn.execute("CREATE TABLE rl_outcome (id INTEGER PRIMARY KEY)")

    mgr = _manager()
    mgr._run_schema_version_migrations(conn)
    first = _schema_version(conn)
    mgr._run_schema_version_migrations(conn)  # must not error or re-migrate
    assert _schema_version(conn) == first == str(CURRENT_SCHEMA_VERSION)


def test_legacy_db_migrates_and_persists_version():
    """A genuine pre-v2 legacy DB (``_db_metadata`` sentinel) migrates and,
    crucially, records its version so it does not re-migrate every startup."""
    conn = sqlite3.connect(":memory:")
    conn.execute(Schema.SYS_METADATA)
    conn.execute("CREATE TABLE _db_metadata (key TEXT, value TEXT)")
    conn.execute(
        "CREATE TABLE rl_outcomes (id INTEGER PRIMARY KEY, created_at TEXT, repo_id TEXT, vertical TEXT)"
    )

    _manager()._run_schema_version_migrations(conn)

    # The version must persist (the legacy branch previously issued an UPDATE
    # that matched no row, leaving schema_version NULL and forcing a replay).
    assert _schema_version(conn) == str(CURRENT_SCHEMA_VERSION)


def test_apply_migration_sql_handles_multi_statement_block():
    """Multi-statement blocks (e.g. the ``*_INDEXES`` constants) execute in the caller transaction instead of raising 'one statement at a time'."""
    conn = sqlite3.connect(":memory:")
    conn.execute(Schema.RL_OUTCOME.split(";")[0])  # create rl_outcome table first

    _apply_migration_sql(conn, Schema.RL_OUTCOME_INDEXES)

    names = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_rl_outcome%'"
        ).fetchall()
    }
    assert {"idx_rl_outcome_learner", "idx_rl_outcome_context"} <= names


@pytest.mark.parametrize("sql", ["THIS IS NOT VALID SQL", "ALTER TABLE missing RENAME TO other"])
def test_unexpected_migration_error_propagates(sql):
    with sqlite3.connect(":memory:") as conn:
        with pytest.raises(sqlite3.OperationalError):
            _apply_migration_sql(conn, sql)


@pytest.mark.parametrize(
    "manager_type,metadata",
    [(DatabaseManager, "sys_metadata"), (ProjectDatabaseManager, "_project_metadata")],
)
def test_failed_migration_keeps_schema_and_version_atomic(
    tmp_path, monkeypatch, manager_type, metadata
):
    from victor.core import schema

    path = tmp_path / "migration.db"
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE {metadata} (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    conn.execute(f"INSERT INTO {metadata} VALUES ('schema_version', '8', NULL)")
    conn.commit()
    monkeypatch.setattr(
        schema,
        "get_migration_steps",
        lambda *args: [
            schema.MigrationStep(
                "global" if manager_type is DatabaseManager else "project",
                "CREATE TABLE new_state (id INTEGER); INSERT INTO new_state VALUES (1);",
            ),
            schema.MigrationStep(
                "global" if manager_type is DatabaseManager else "project",
                "INSERT INTO missing_table VALUES (1)",
            ),
        ],
    )
    manager = object.__new__(manager_type)
    with pytest.raises(sqlite3.OperationalError):
        manager._run_schema_version_migrations(conn)
    conn.close()
    with sqlite3.connect(path) as observer:
        assert (
            observer.execute(f"SELECT value FROM {metadata} WHERE key='schema_version'").fetchone()[
                0
            ]
            == "8"
        )
        assert (
            observer.execute("SELECT name FROM sqlite_master WHERE name='new_state'").fetchone()
            is None
        )


def test_multistatement_helper_preserves_caller_transaction():
    conn = sqlite3.connect(":memory:")
    conn.execute("BEGIN")
    with pytest.raises(sqlite3.OperationalError):
        _apply_migration_sql(
            conn, "CREATE TABLE transient (value TEXT); INSERT INTO missing VALUES (1);"
        )
    assert conn.in_transaction
    conn.rollback()
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='transient'").fetchone() is None
    conn.close()


@pytest.mark.parametrize(
    "manager_type,metadata",
    [(DatabaseManager, "sys_metadata"), (ProjectDatabaseManager, "_project_metadata")],
)
def test_future_schema_version_is_rejected(manager_type, metadata):
    conn = sqlite3.connect(":memory:")
    conn.execute(f"CREATE TABLE {metadata} (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    conn.execute(f"INSERT INTO {metadata} VALUES ('schema_version', '999', NULL)")
    conn.commit()
    with pytest.raises(ValueError, match="newer"):
        object.__new__(manager_type)._run_schema_version_migrations(conn)
    assert conn.execute(f"SELECT value FROM {metadata}").fetchone()[0] == "999"
    conn.close()


LEGACY_DELTA = """CREATE TABLE local_classifier_delta (
    decision_type TEXT, feature_hash INTEGER, weight REAL, samples INTEGER,
    sum_reward REAL, updated_at TEXT, PRIMARY KEY (decision_type, feature_hash))"""


def _versioned(conn, scope, version):
    metadata = "sys_metadata" if scope == "global" else "_project_metadata"
    conn.execute(f"CREATE TABLE {metadata} (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    conn.execute(f"INSERT INTO {metadata} VALUES ('schema_version', ?, NULL)", (str(version),))
    conn.commit()


def test_nested_migration_does_not_commit_caller_and_rolls_back_with_it():
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        _versioned(conn, "global", 7)
        conn.execute("BEGIN")
        migrate(conn, "global")
        assert conn.in_transaction
        assert _schema_version(conn) == "9"
        conn.rollback()
        assert _schema_version(conn) == "7"
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE name='decision_log'").fetchone()
            is None
        )


def test_failed_nested_migration_preserves_earlier_caller_write(monkeypatch):
    from victor.core import schema
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        _versioned(conn, "global", 8)
        conn.execute("INSERT INTO sys_metadata VALUES ('caller', 'preserved', NULL)")
        monkeypatch.setattr(
            schema,
            "get_migration_steps",
            lambda *args: [
                schema.MigrationStep(
                    "global", "CREATE TABLE partial (id INTEGER); INSERT INTO missing VALUES (1)"
                )
            ],
        )
        with pytest.raises(sqlite3.OperationalError):
            migrate(conn, "global")
        assert conn.in_transaction
        assert _schema_version(conn) == "8"
        assert (
            conn.execute("SELECT value FROM sys_metadata WHERE key='caller'").fetchone()[0]
            == "preserved"
        )
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE name='partial'").fetchone() is None
        )


@pytest.mark.parametrize(
    "scope,forbidden", [("global", "local_classifier_delta"), ("project", "decision_log")]
)
def test_scope_excludes_other_database_tables(scope, forbidden):
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        _versioned(conn, scope, 0)
        conn.execute("CREATE TABLE sessions (id TEXT)")
        migrate(conn, scope)
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE name=?", (forbidden,)).fetchone()
            is None
        )
        if scope == "project":
            assert conn.execute("SELECT name FROM sqlite_master WHERE name='sessions'").fetchone()


@pytest.mark.parametrize(
    "column,passes",
    [("session_id TEXT", True), ("session_id INTEGER", False), ("session_id TEXT NOT NULL", False)],
)
def test_existing_column_requires_compatible_shape(column, passes):
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        _versioned(conn, "global", 3)
        conn.execute(
            f"CREATE TABLE rl_outcome (created_at TEXT, repo_id TEXT, vertical TEXT, {column})"
        )
        if passes:
            migrate(conn, "global")
            assert _schema_version(conn) == "9"
        else:
            with pytest.raises(ValueError, match="Incompatible existing column"):
                migrate(conn, "global")
            assert _schema_version(conn) == "3"


def test_present_optional_table_with_missing_index_column_fails():
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        _versioned(conn, "global", 3)
        conn.execute("CREATE TABLE rl_outcome (id INTEGER)")
        with pytest.raises(sqlite3.OperationalError, match="created_at"):
            migrate(conn, "global")
        assert _schema_version(conn) == "3"
        assert "session_id" not in {row[1] for row in conn.execute("PRAGMA table_info(rl_outcome)")}


def test_legacy_rename_collision_preserves_both_tables():
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        _versioned(conn, "global", 1)
        for table, value in [("rl_outcomes", 1), ("rl_outcome", 2)]:
            conn.execute(f"CREATE TABLE {table} (id INTEGER)")
            conn.execute(f"INSERT INTO {table} VALUES (?)", (value,))
        conn.commit()
        with pytest.raises(ValueError, match="Conflicting migration tables"):
            migrate(conn, "global")
        assert _schema_version(conn) == "1"
        assert conn.execute("SELECT id FROM rl_outcomes").fetchone()[0] == 1
        assert conn.execute("SELECT id FROM rl_outcome").fetchone()[0] == 2


@pytest.mark.parametrize(
    "case", ["new", "equal", "conflict", "duplicate", "missing_column", "null_key"]
)
def test_legacy_metadata_reconciliation(case):
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        _versioned(conn, "global", 1)
        if case == "missing_column":
            conn.execute("CREATE TABLE _db_metadata (value TEXT)")
        else:
            conn.execute("CREATE TABLE _db_metadata (key TEXT, value TEXT)")
            conn.execute(
                "INSERT INTO _db_metadata VALUES (?, 'kept')",
                (None if case == "null_key" else "custom",),
            )
            conn.execute("INSERT INTO _db_metadata VALUES ('schema_version', '1')")
            if case == "duplicate":
                conn.execute("INSERT INTO _db_metadata VALUES ('custom', 'kept')")
        if case in {"equal", "conflict"}:
            conn.execute(
                "INSERT INTO sys_metadata VALUES ('custom', ?, NULL)",
                ("kept" if case == "equal" else "other",),
            )
        conn.commit()
        if case in {"new", "equal"}:
            migrate(conn, "global")
            assert _schema_version(conn) == "9"
            assert (
                conn.execute("SELECT value FROM sys_metadata WHERE key='custom'").fetchone()[0]
                == "kept"
            )
        else:
            with pytest.raises(ValueError):
                migrate(conn, "global")
            assert _schema_version(conn) == "1"
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='_db_metadata'").fetchone()


@pytest.mark.parametrize(
    "layout", ["absent", "current_populated", "legacy_empty", "legacy_populated", "unknown_empty"]
)
def test_v9_migration_preserves_data_or_rejects(layout):
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        _versioned(conn, "project", 8)
        if layout == "current_populated":
            conn.execute(Schema.LOCAL_CLASSIFIER_DELTA)
            conn.execute(
                "INSERT INTO local_classifier_delta (decision_type, feature_hash, label) VALUES ('task', 1, 'pass')"
            )
        elif layout.startswith("legacy"):
            conn.execute(LEGACY_DELTA)
            if layout == "legacy_populated":
                conn.execute(
                    "INSERT INTO local_classifier_delta VALUES ('task', 1, .5, 1, .5, 't')"
                )
        elif layout == "unknown_empty":
            conn.execute("CREATE TABLE local_classifier_delta (id INTEGER)")
        conn.commit()
        before = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='local_classifier_delta'"
        ).fetchone()
        if layout in {"legacy_populated", "unknown_empty"}:
            with pytest.raises(ValueError, match="reconcil"):
                migrate(conn, "project", initialize=True)
            assert (
                conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name='local_classifier_delta'"
                ).fetchone()
                == before
            )
            assert conn.execute("SELECT value FROM _project_metadata").fetchone()[0] == "8"
        else:
            migrate(conn, "project", initialize=True)
            count = conn.execute("SELECT COUNT(*) FROM local_classifier_delta").fetchone()[0]
            assert count == (1 if layout == "current_populated" else 0)
            assert "feature_spec_version" in {
                row[1] for row in conn.execute("PRAGMA table_info(local_classifier_delta)")
            }


def test_project_initialization_upgrades_before_current_indexes(tmp_path):
    path = tmp_path / ".victor" / "project.db"
    path.parent.mkdir(exist_ok=True)
    with sqlite3.connect(path) as conn:
        _versioned(conn, "project", 8)
        conn.execute(LEGACY_DELTA)
    manager = ProjectDatabaseManager(tmp_path)
    try:
        assert (
            manager.get_connection()
            .execute("SELECT value FROM _project_metadata WHERE key='schema_version'")
            .fetchone()[0]
            == "9"
        )
        assert (
            manager.get_connection()
            .execute("SELECT name FROM sqlite_master WHERE name='idx_local_classifier_delta_type'")
            .fetchone()
        )
    finally:
        manager.close()


def test_final_index_failure_rolls_back_schema_and_can_retry(monkeypatch):
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        _versioned(conn, "project", 8)
        conn.execute(LEGACY_DELTA)
        conn.commit()
        with monkeypatch.context() as patch:
            patch.setattr(
                Schema,
                "get_project_indexes",
                classmethod(lambda cls: ["CREATE INDEX broken ON graph_node(no_such_column)"]),
            )
            with pytest.raises(sqlite3.OperationalError):
                migrate(conn, "project", initialize=True)
        assert conn.execute("SELECT value FROM _project_metadata").fetchone()[0] == "8"
        assert "label" not in {
            row[1] for row in conn.execute("PRAGMA table_info(local_classifier_delta)")
        }
        migrate(conn, "project", initialize=True)
        assert conn.execute("SELECT value FROM _project_metadata").fetchone()[0] == "9"


def test_sql_block_parser_preserves_quoted_semicolons_comments_and_trigger():
    with sqlite3.connect(":memory:") as conn:
        conn.execute("BEGIN")
        _apply_migration_sql(
            conn,
            """
            -- a comment with a ; delimiter
            CREATE TABLE sample (value TEXT);
            CREATE TABLE audit (value TEXT);
            CREATE TRIGGER log_sample AFTER INSERT ON sample BEGIN
                INSERT INTO audit VALUES ('trigger;value');
                INSERT INTO audit VALUES (new.value);
            END;
            INSERT INTO sample VALUES ('literal;value'); -- trailing ; comment
        """,
        )
        assert conn.execute("SELECT value FROM audit ORDER BY rowid").fetchall() == [
            ("trigger;value",),
            ("literal;value",),
        ]
        conn.rollback()
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='sample'").fetchone() is None


@pytest.mark.parametrize(
    "sql",
    [
        "COMMIT",
        "-- comment\nCOMMIT;",
        "/* comment */ END;",
        "PRAGMA foreign_keys=OFF;",
        "ATTACH ':memory:' AS bypass",
    ],
)
def test_migration_sql_cannot_escape_transaction(sql):
    with sqlite3.connect(":memory:") as conn:
        conn.execute("BEGIN")
        with pytest.raises(sqlite3.OperationalError, match="Unsupported"):
            _apply_migration_sql(conn, sql)
        assert conn.in_transaction


def test_concurrent_migrators_reread_version_after_writer_admission(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from victor.core import schema
    from victor.core.migrations import migrate

    path = tmp_path / "concurrent.db"
    with sqlite3.connect(path) as conn:
        _versioned(conn, "global", 8)
    admitted, release, second_started = Event(), Event(), Event()
    calls = []

    def steps(*args):
        calls.append(args)
        if args[0] == 8:
            admitted.set()
            assert release.wait(5)
            return [schema.MigrationStep("global", "CREATE TABLE once_only (id INTEGER)")]
        return []

    monkeypatch.setattr(schema, "get_migration_steps", steps)

    def run(second=False):
        with sqlite3.connect(path, timeout=10) as conn:
            if second:
                second_started.set()
            migrate(conn, "global")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run)
        try:
            assert admitted.wait(5)
            second = pool.submit(run, True)
            assert second_started.wait(5)
        finally:
            release.set()
        first.result(timeout=10)
        second.result(timeout=10)
    assert calls == [(8, 9), (9, 9)]
    with sqlite3.connect(path) as conn:
        assert _schema_version(conn) == "9"


@pytest.mark.parametrize(
    "scope,version,name",
    [
        ("global", 6, "rl_param"),
        ("project", 8, "local_classifier_delta"),
    ],
)
@pytest.mark.parametrize("kind", ["view", "index"])
def test_wrong_kind_object_is_not_an_absent_optional_table(scope, version, name, kind):
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        _versioned(conn, scope, version)
        if kind == "view":
            conn.execute(f"CREATE VIEW {name} AS SELECT 1 AS value")
        else:
            conn.execute("CREATE TABLE unrelated (value TEXT)")
            conn.execute(f"CREATE INDEX {name} ON unrelated(value)")
        with pytest.raises(ValueError, match="Expected table"):
            migrate(conn, scope)
        metadata = "sys_metadata" if scope == "global" else "_project_metadata"
        assert conn.execute(f"SELECT value FROM {metadata}").fetchone()[0] == str(version)
        assert (
            conn.execute("SELECT type FROM sqlite_master WHERE name=?", (name,)).fetchone()[0]
            == kind
        )


def test_deferred_constraint_failure_at_commit_rolls_back_publication(monkeypatch):
    from victor.core import schema
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        _versioned(conn, "global", 8)
        monkeypatch.setattr(
            schema,
            "get_migration_steps",
            lambda *args: [
                schema.MigrationStep(
                    "global",
                    """
                CREATE TABLE parent (id INTEGER PRIMARY KEY);
                CREATE TABLE child (parent_id INTEGER REFERENCES parent(id)
                    DEFERRABLE INITIALLY DEFERRED);
                INSERT INTO child VALUES (99);
            """,
                )
            ],
        )
        with pytest.raises(sqlite3.IntegrityError):
            migrate(conn, "global")
        assert not conn.in_transaction
        assert _schema_version(conn) == "8"
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='child'").fetchone() is None


def test_interrupt_rolls_back_migration(monkeypatch):
    from victor.core import schema
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        _versioned(conn, "global", 8)

        def interrupted(*args):
            conn.execute("CREATE TABLE partial (id INTEGER)")
            raise KeyboardInterrupt

        monkeypatch.setattr(schema, "get_migration_steps", interrupted)
        with pytest.raises(KeyboardInterrupt):
            migrate(conn, "global")
        assert not conn.in_transaction
        assert _schema_version(conn) == "8"
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE name='partial'").fetchone() is None
        )


@pytest.mark.parametrize("versions", [["999"], ["1", "1"]])
def test_unversioned_target_rejects_invalid_legacy_version(versions):
    from victor.core.migrations import migrate

    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE _db_metadata (key TEXT, value TEXT)")
        conn.executemany(
            "INSERT INTO _db_metadata VALUES ('schema_version', ?)", [(v,) for v in versions]
        )
        conn.commit()
        with pytest.raises(ValueError):
            migrate(conn, "global")
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE name='sys_metadata'").fetchone()
            is None
        )
        assert conn.execute("SELECT value FROM _db_metadata").fetchall() == [(v,) for v in versions]
