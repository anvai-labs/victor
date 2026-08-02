# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for the dedicated undo-history database (``.victor/undo.db``).

The undo store lives in its own SQLite file so its per-edit writes never contend
with the graph indexer's continuous writes to ``project.db``.
"""

from pathlib import Path

import pytest

from victor.core.undo_database import (
    UNDO_SCHEMA_VERSION,
    UndoDatabaseManager,
    get_undo_database,
    reset_undo_databases,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_undo_databases()
    yield
    reset_undo_databases()


def test_opens_dedicated_undo_db_file(tmp_path):
    mgr = get_undo_database(tmp_path)
    assert mgr.db_path == tmp_path / ".victor" / "undo.db"
    assert mgr.db_path.name == "undo.db"
    assert mgr.db_path.exists()


def test_does_not_create_project_db(tmp_path):
    get_undo_database(tmp_path)
    # Undo store must be isolated: project.db is NOT created by this manager.
    assert not (tmp_path / ".victor" / "project.db").exists()


def test_wal_mode_enabled(tmp_path):
    mgr = get_undo_database(tmp_path)
    mode = mgr.get_connection().execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_schema_v1_tables_and_version(tmp_path):
    mgr = get_undo_database(tmp_path)
    conn = mgr.get_connection()
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"undo_meta", "change_groups", "file_changes"}.issubset(tables)
    assert mgr.schema_version() == UNDO_SCHEMA_VERSION == 1


def test_file_changes_has_new_columns(tmp_path):
    mgr = get_undo_database(tmp_path)
    cols = {
        r[1] for r in mgr.get_connection().execute("PRAGMA table_info(file_changes)").fetchall()
    }
    assert {"seq", "session_id", "message_id"}.issubset(cols)


def test_change_groups_has_message_id(tmp_path):
    mgr = get_undo_database(tmp_path)
    cols = {
        r[1] for r in mgr.get_connection().execute("PRAGMA table_info(change_groups)").fetchall()
    }
    assert "message_id" in cols


def test_factory_caches_per_path(tmp_path):
    a = get_undo_database(tmp_path)
    b = get_undo_database(tmp_path)
    assert a is b


def test_distinct_paths_get_distinct_managers(tmp_path):
    p1 = tmp_path / "proj1"
    p2 = tmp_path / "proj2"
    p1.mkdir()
    p2.mkdir()
    assert get_undo_database(p1) is not get_undo_database(p2)


def test_ensure_schema_is_idempotent(tmp_path):
    mgr = get_undo_database(tmp_path)
    mgr.ensure_schema()
    mgr.ensure_schema()
    assert mgr.schema_version() == 1


def test_construct_directly(tmp_path):
    mgr = UndoDatabaseManager(project_path=tmp_path)
    assert isinstance(mgr, UndoDatabaseManager)
    assert mgr.db_path.exists()
