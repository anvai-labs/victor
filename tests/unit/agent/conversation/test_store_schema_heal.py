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


def _build_legacy_ui_session_db(conn: sqlite3.Connection) -> None:
    """Create a pre-migration project.db: messages FK -> ui_session plus the
    old-shape messages_fts (stale message_id column) and its triggers."""
    conn.executescript("""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY, created_at TIMESTAMP NOT NULL,
            last_active TIMESTAMP NOT NULL, project_path TEXT,
            provider TEXT, model TEXT, is_archived INTEGER DEFAULT 0
        );
        CREATE TABLE ui_session (
            session_id TEXT PRIMARY KEY, created_at TIMESTAMP NOT NULL
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, timestamp TIMESTAMP NOT NULL,
            token_count INTEGER NOT NULL, priority INTEGER NOT NULL,
            tool_name TEXT, tool_call_id TEXT, metadata TEXT, agent_id TEXT,
            parent_session_id TEXT, team_id TEXT, member_id TEXT, plan_id TEXT,
            plan_step_id TEXT,
            FOREIGN KEY (session_id) REFERENCES ui_session(session_id)
                ON DELETE CASCADE
        );
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            content, message_id UNINDEXED, session_id UNINDEXED,
            content='messages', content_rowid='rowid'
        );
        CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content, message_id, session_id)
            VALUES (NEW.rowid, NEW.content, NEW.id, NEW.session_id);
        END;
        CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content, message_id, session_id)
            VALUES ('delete', OLD.rowid, OLD.content, OLD.id, OLD.session_id);
        END;
    """)
    conn.execute("INSERT INTO sessions VALUES ('s1','t','t','/p','x','y',0)")
    conn.execute("INSERT INTO ui_session VALUES ('s1','t')")
    # A session that only ever existed in ui_session (the legacy workaround).
    conn.execute("INSERT INTO ui_session VALUES ('s_ghost','t')")
    rows = [
        ("m1", "s1", "alpha tok_one"),
        ("m2", "s1", "bravo tok_two"),
        ("m3", "s1", "charlie tok_three"),
        ("m4", "s1", "delta tok_four"),
    ]
    for mid, sid, content in rows:
        conn.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                mid,
                sid,
                "user",
                content,
                "t",
                1,
                1,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )
    conn.execute(
        "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "m_orphan",
            "s_ghost",
            "user",
            "orphan tok_six",
            "t",
            1,
            1,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    # Compaction-style hard delete leaves a rowid gap (m3 gone, rowids 1,2,4,5).
    conn.execute("DELETE FROM messages WHERE id='m3'")
    conn.commit()


def _migrated_schema():
    from types import SimpleNamespace

    schema = ConversationStoreSchema.__new__(ConversationStoreSchema)
    schema.store = SimpleNamespace(
        _model_family_ids={},
        _model_size_ids={},
        _context_size_ids={},
        _provider_ids={},
        cleanup_stale_sessions=lambda **kwargs: None,
        db_path=":memory:",
    )
    return schema


def test_migrate_fk_to_sessions_rebuilds_fts_index(tmp_path):
    """The FK migration must recreate messages_fts, not carry it over.

    Regression (negative): the old-shape index has a stale message_id column
    (content queries error out) and its rowids no longer match the re-inserted
    messages — searches returned the wrong message or nothing. The migration
    must rebuild the index in the current shape and repopulate it.
    """
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    _build_legacy_ui_session_db(conn)

    schema = _migrated_schema()
    schema.migrate_fk_to_sessions(conn)
    schema.ensure_critical_tables(conn)
    conn.commit()

    # FK now targets sessions; original UI payloads remain available for recovery.
    fk_targets = {row[2] for row in conn.execute("PRAGMA foreign_key_list(messages)")}
    assert fk_targets == {"sessions"}
    assert (
        conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='ui_session'").fetchone()[0]
        == 1
    )

    # FTS recreated in the current shape: no stale message_id column, and the
    # COUNT query that errored on the old shape ("no such column: T.message_id")
    # now works.
    fts_sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='messages_fts'").fetchone()[0]
    assert "message_id" not in fts_sql
    indexed = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
    surviving = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert indexed == surviving == 4  # m1, m2, m4, and recovered m_orphan

    # Rowids align: every index entry maps to a live message and vice versa.
    fts_rowids = {r[0] for r in conn.execute("SELECT rowid FROM messages_fts")}
    msg_rowids = {r[0] for r in conn.execute("SELECT rowid FROM messages")}
    assert fts_rowids == msg_rowids

    # Every surviving message is findable by a token unique to it — including
    # m4, which sat after the compaction rowid gap and was unfindable before.
    for token, expected_id in [("tok_one", "m1"), ("tok_two", "m2"), ("tok_four", "m4")]:
        hits = conn.execute(
            "SELECT m.id FROM messages_fts f JOIN messages m ON m.rowid = f.rowid"
            " WHERE messages_fts MATCH ?",
            (token,),
        ).fetchall()
        assert [h[0] for h in hits] == [expected_id], (token, hits)

    conn.close()


def test_migrate_fk_to_sessions_is_idempotent(tmp_path):
    """A second init on a migrated DB must be a no-op for data and index."""
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    _build_legacy_ui_session_db(conn)

    schema = _migrated_schema()
    schema.migrate_fk_to_sessions(conn)
    schema.ensure_critical_tables(conn)
    conn.commit()
    before = conn.execute("SELECT id, content FROM messages ORDER BY id").fetchall()
    before_hits = conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'tok_four'"
    ).fetchone()[0]
    conn.close()

    conn = sqlite3.connect(db)
    schema.migrate_fk_to_sessions(conn)  # second run: early-return path
    schema.ensure_critical_tables(conn)
    conn.commit()
    after = conn.execute("SELECT id, content FROM messages ORDER BY id").fetchall()
    after_hits = conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'tok_four'"
    ).fetchone()[0]

    assert before == after
    assert before_hits == after_hits == 1
    conn.close()


def test_migrate_fk_to_sessions_logs_recovered_sessions(tmp_path, caplog):
    """Legacy-only histories retain a session header and emit a recovery warning."""
    import logging

    conn = sqlite3.connect(tmp_path / "legacy.db")
    _build_legacy_ui_session_db(conn)
    with caplog.at_level(logging.WARNING, logger="victor.agent.conversation.store_schema"):
        _migrated_schema().migrate_fk_to_sessions(conn)
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE id='m_orphan'").fetchone()[0] == 1
    assert any("recovered 1 legacy session" in rec.message for rec in caplog.records)
    conn.close()


def test_migrate_fk_to_sessions_preserves_metadata_json(tmp_path):
    """metadata_json values survive the table recreation when the legacy
    table has the column; a legacy table without it keeps the '{}' default."""
    db = tmp_path / "legacy_json.db"
    conn = sqlite3.connect(db)
    _build_legacy_ui_session_db(conn)
    # Simulate a database that ran apply_hybrid_compaction_schema.
    conn.execute("ALTER TABLE messages ADD COLUMN metadata_json TEXT DEFAULT '{}'")
    conn.execute("UPDATE messages SET metadata_json = ? WHERE id = 'm1'", ('{"k": 1}',))
    conn.commit()

    schema = _migrated_schema()
    schema.migrate_fk_to_sessions(conn)
    schema.ensure_critical_tables(conn)
    conn.commit()

    preserved = conn.execute("SELECT metadata_json FROM messages WHERE id = 'm1'").fetchone()[0]
    default_kept = conn.execute("SELECT metadata_json FROM messages WHERE id = 'm4'").fetchone()[0]
    assert preserved == '{"k": 1}'
    assert default_kept == "{}"
    conn.close()


class _FailingMigrationConnection(sqlite3.Connection):
    fail_fts = False

    def execute(self, sql, parameters=()):
        if self.fail_fts and "CREATE VIRTUAL TABLE" in sql:
            raise sqlite3.OperationalError("injected FTS failure")
        return super().execute(sql, parameters)


def _add_legacy_compaction(conn):
    from victor.agent.conversation.store_schema import _COMPACTION_EVENTS_DDL

    conn.execute(_COMPACTION_EVENTS_DDL.replace("REFERENCES sessions", "REFERENCES ui_session"))
    conn.execute(
        "INSERT INTO compaction_events (id, session_id, strategy, created_at) "
        "VALUES ('c1', 's1', 'tiered', 't')"
    )
    conn.commit()


def test_fk_migration_failure_rolls_back_data_schema_and_restores_enforcement(tmp_path):
    import pytest

    conn = sqlite3.connect(tmp_path / "atomic.db", factory=_FailingMigrationConnection)
    _build_legacy_ui_session_db(conn)
    _add_legacy_compaction(conn)
    conn.execute("PRAGMA foreign_keys = ON")
    before = conn.execute("SELECT * FROM messages ORDER BY id").fetchall()
    conn.fail_fts = True
    with pytest.raises(sqlite3.OperationalError, match="injected FTS failure"):
        _migrated_schema().migrate_fk_to_sessions(conn)
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("SELECT * FROM messages ORDER BY id").fetchall() == before
    assert conn.execute("PRAGMA foreign_key_list(messages)").fetchone()[2] == "ui_session"
    assert "messages_old" not in _core_tables(conn)
    conn.fail_fts = False
    _migrated_schema().migrate_fk_to_sessions(conn)
    assert conn.execute("PRAGMA foreign_key_check(messages)").fetchall() == []
    assert conn.execute("PRAGMA foreign_key_list(compaction_events)").fetchone()[2] == "sessions"
    conn.close()


def test_fk_migration_repairs_compaction_when_messages_already_migrated():
    conn = sqlite3.connect(":memory:")
    _build_legacy_ui_session_db(conn)
    _add_legacy_compaction(conn)
    schema = _migrated_schema()
    # Simulate the previous implementation failing after messages committed.
    conn.execute("ALTER TABLE compaction_events RENAME TO saved_compaction")
    schema.migrate_fk_to_sessions(conn)
    conn.execute("ALTER TABLE saved_compaction RENAME TO compaction_events")
    conn.execute("DROP TABLE messages_fts")
    conn.commit()
    schema.migrate_fk_to_sessions(conn)
    assert conn.execute("PRAGMA foreign_key_list(compaction_events)").fetchone()[2] == "sessions"
    assert conn.execute("SELECT id FROM compaction_events").fetchall() == [("c1",)]
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'tok_one'"
        ).fetchone()[0]
        == 1
    )
    conn.close()


def test_fk_migration_preserves_legacy_session_only_messages():
    conn = sqlite3.connect(":memory:")
    _build_legacy_ui_session_db(conn)
    _migrated_schema().migrate_fk_to_sessions(conn)
    assert conn.execute("SELECT content FROM messages WHERE id='m_orphan'").fetchone() == (
        "orphan tok_six",
    )
    assert conn.execute("SELECT session_id FROM sessions WHERE session_id='s_ghost'").fetchone()
    conn.close()


def test_fk_migration_fts_tracks_update_and_delete_with_foreign_keys_enabled():
    conn = sqlite3.connect(":memory:")
    _build_legacy_ui_session_db(conn)
    conn.execute("PRAGMA foreign_keys = ON")
    schema = _migrated_schema()
    schema.migrate_fk_to_sessions(conn)
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    conn.execute("UPDATE messages SET content='replacementtoken' WHERE id='m1'")
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'tok_one'"
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'replacementtoken'"
        ).fetchone()[0]
        == 1
    )
    conn.execute("DELETE FROM sessions WHERE session_id='s1'")
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id='s1'").fetchone()[0] == 0
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'replacementtoken'"
        ).fetchone()[0]
        == 0
    )
    conn.close()


def test_fk_migration_merges_interrupted_tables_without_losing_new_metadata():
    conn = sqlite3.connect(":memory:")
    _build_legacy_ui_session_db(conn)
    conn.execute("ALTER TABLE messages RENAME TO messages_old")
    from victor.agent.conversation.store_schema import _MESSAGES_TABLE_DDL

    conn.execute(_MESSAGES_TABLE_DDL)
    conn.execute(
        "INSERT INTO messages (id, session_id, role, content, timestamp, token_count, priority, metadata_json) "
        "VALUES ('m1', 's1', 'user', 'newer content', 't', 1, 1, ?)",
        ('{"new": true}',),
    )
    conn.commit()
    _migrated_schema().migrate_fk_to_sessions(conn)
    assert conn.execute("SELECT content, metadata_json FROM messages WHERE id='m1'").fetchone() == (
        "newer content",
        '{"new": true}',
    )
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 4
    assert "messages_old" not in _core_tables(conn)
    conn.close()


def test_fk_migration_repairs_missing_fts_even_without_pending_foreign_keys():
    conn = sqlite3.connect(":memory:")
    _build_legacy_ui_session_db(conn)
    schema = _migrated_schema()
    schema.migrate_fk_to_sessions(conn)
    conn.execute("DROP TABLE messages_fts")
    conn.commit()
    schema.migrate_fk_to_sessions(conn)
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'tok_one'"
        ).fetchone()[0]
        == 1
    )
    conn.close()
