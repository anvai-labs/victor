# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Regression tests for global schema-version migration.

Covers the noise/idempotency issues seen on real global databases:

* A modern DB that accumulated tables lazily (``sys_metadata`` present, no
  ``schema_version`` row, no ``_db_metadata`` sentinel) must NOT replay the
  historical rename/ALTER chain — doing so only logged "no such table" /
  "duplicate column" warnings on every startup.
* The recorded ``schema_version`` must persist so migrations run at most once.
* Genuine pre-v2 legacy DBs (``_db_metadata`` sentinel present) must still
  migrate and record their version.
* Multi-statement migration blocks must not raise "You can only execute one
  statement at a time", and benign already-satisfied errors must not surface
  as WARNING.
"""

from __future__ import annotations

import logging
import sqlite3

from victor.core.database import DatabaseManager, _apply_migration_sql
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
    conn.execute("CREATE TABLE rl_outcomes (id INTEGER PRIMARY KEY)")

    _manager()._run_schema_version_migrations(conn)

    # The version must persist (the legacy branch previously issued an UPDATE
    # that matched no row, leaving schema_version NULL and forcing a replay).
    assert _schema_version(conn) == str(CURRENT_SCHEMA_VERSION)


def test_apply_migration_sql_handles_multi_statement_block():
    """Multi-statement blocks (e.g. the ``*_INDEXES`` constants) are routed to
    executescript instead of raising 'one statement at a time'."""
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


def test_apply_migration_sql_downgrades_benign_errors(caplog):
    """Already-satisfied errors are DEBUG; genuine errors stay WARNING."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE rl_outcome (id INTEGER PRIMARY KEY)")

    with caplog.at_level(logging.WARNING):
        _apply_migration_sql(conn, "ALTER TABLE missing_table RENAME TO other")  # no such table
        _apply_migration_sql(conn, "ALTER TABLE rl_outcome ADD COLUMN id TEXT")  # duplicate column
    assert caplog.text == ""  # benign → nothing at WARNING or above

    with caplog.at_level(logging.WARNING):
        _apply_migration_sql(conn, "THIS IS NOT VALID SQL")
    assert "Migration SQL failed" in caplog.text
