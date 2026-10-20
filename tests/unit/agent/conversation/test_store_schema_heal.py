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

"""ConversationStore schema self-heal for legacy `sessions` tables."""

import sqlite3

from victor.agent.conversation.store_schema import ConversationStoreSchema


def test_migrate_sessions_table_heals_all_missing_core_columns(tmp_path):
    """A legacy sessions table missing core columns is healed in one pass.

    Regression: an earlier fix added only session_id, so init still failed on the
    next missing column (last_activity). The heal must add the whole core set.
    """
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    # Ancient table — missing session_id, last_activity, and most core columns.
    conn.execute("CREATE TABLE sessions (id TEXT, created_at TIMESTAMP)")
    conn.commit()

    schema = ConversationStoreSchema.__new__(ConversationStoreSchema)
    schema.migrate_sessions_table(conn)

    cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    for required in (
        "session_id",
        "last_activity",
        "project_path",
        "model",
        "profile",
        "max_tokens",
        "reserved_tokens",
        "metadata",
    ):
        assert required in cols, f"missing healed column: {required}"

    # An INSERT OR REPLACE using the core columns now succeeds (no "no such column").
    conn.execute(
        "INSERT OR REPLACE INTO sessions (session_id, created_at, last_activity) "
        "VALUES (?, ?, ?)",
        ("s1", "2026-01-01", "2026-01-01"),
    )
    conn.commit()
    conn.close()


def _core_tables(conn) -> set:
    return {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }


def test_critical_ddl_constant_creates_core_tables():
    """jscpd extraction: _CRITICAL_TABLES_DDL is the one shared definition —
    executing it directly creates exactly the four core tables, idempotently."""
    import sqlite3

    from victor.agent.conversation.store_schema import _CRITICAL_TABLES_DDL

    conn = sqlite3.connect(":memory:")
    conn.executescript(_CRITICAL_TABLES_DDL)
    assert _core_tables(conn) == {"sessions", "messages", "context_summaries", "compaction_events"}
    # Idempotent: a second run is a no-op on the same tables.
    conn.executescript(_CRITICAL_TABLES_DDL)
    assert _core_tables(conn) == {"sessions", "messages", "context_summaries", "compaction_events"}


def test_ensure_critical_tables_and_normalized_schema_both_apply_the_ddl(tmp_path):
    """Both call sites interpolate the shared constant; each leaves the four
    core tables behind (the normalized path additionally creates lookups)."""
    import sqlite3

    from types import SimpleNamespace

    schema = ConversationStoreSchema.__new__(ConversationStoreSchema)
    schema.store = SimpleNamespace(
        _model_family_ids={},
        _model_size_ids={},
        _context_size_ids={},
        _provider_ids={},
        cleanup_stale_sessions=lambda **kwargs: None,
        db_path=str(tmp_path / "normalized.db"),
    )

    critical = sqlite3.connect(":memory:")
    schema.ensure_critical_tables(critical)
    assert {"sessions", "messages", "context_summaries", "compaction_events"} <= _core_tables(
        critical
    )

    normalized = sqlite3.connect(":memory:")
    schema.apply_normalized_schema(normalized)
    tables = _core_tables(normalized)
    assert {"sessions", "messages", "context_summaries", "compaction_events"} <= tables
    # The normalized path creates the lookup tables the DDL does not own.
    assert {"providers", "model_families"} <= tables
