"""Priority 4 Phase 1: Database Schema Migration Tests.

Tests for the session_id column migration to rl_outcome table.
Verifies:
- Migration applies correctly
- Backward compatibility maintained
- New indexes created
- Existing queries still work
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import pytest

from victor.core.schema import Tables, get_migration_sql, CURRENT_SCHEMA_VERSION
from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import get_rl_coordinator


class TestPriority4Migration:
    """Test Priority 4 database schema migration (version 3 -> 4)."""

    def test_migration_version_4_exists(self):
        """Verify migration version 4 is defined."""
        # Migration from 3 to 4 should exist
        migrations = get_migration_sql(3, 4)
        assert len(migrations) > 0, "Migration version 4 should have SQL statements"

    def test_migration_adds_session_id_column(self):
        """Verify migration adds session_id column to rl_outcome."""
        # Create test database with schema version 3
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Initialize database with version 3 schema
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Create rl_outcome table (version 3 structure)
            cursor.execute(f"""
                CREATE TABLE {Tables.RL_OUTCOME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    task_type TEXT,
                    vertical TEXT DEFAULT '',
                    repo_id TEXT DEFAULT NULL,
                    success INTEGER,
                    quality_score REAL,
                    metadata TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # Verify session_id doesn't exist yet
            cursor.execute(f"PRAGMA table_info({Tables.RL_OUTCOME})")
            columns_v3 = {row[1] for row in cursor.fetchall()}
            assert "session_id" not in columns_v3

            # Apply migration version 4
            migrations = get_migration_sql(3, 4)
            for sql in migrations:
                cursor.execute(sql)

            # Verify session_id column now exists
            cursor.execute(f"PRAGMA table_info({Tables.RL_OUTCOME})")
            columns_v4 = {row[1] for row in cursor.fetchall()}
            assert "session_id" in columns_v4, "session_id column should be added"

            # Verify all original columns still exist
            assert columns_v3.issubset(columns_v4), "All original columns should remain"

            conn.close()

    def test_migration_creates_indexes(self):
        """Verify migration creates performance indexes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Create rl_outcome table
            cursor.execute(f"""
                CREATE TABLE {Tables.RL_OUTCOME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    task_type TEXT,
                    vertical TEXT DEFAULT '',
                    repo_id TEXT DEFAULT NULL,
                    success INTEGER,
                    quality_score REAL,
                    metadata TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # Apply migration
            migrations = get_migration_sql(3, 4)
            for sql in migrations:
                cursor.execute(sql)

            # Verify indexes were created
            cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='index' "
                f"AND tbl_name='{Tables.RL_OUTCOME}'"
            )
            indexes = {row[0] for row in cursor.fetchall()}

            assert "idx_rl_outcome_session" in indexes, "session index should be created"
            assert "idx_rl_outcome_repo" in indexes, "repo index should be created"

            conn.close()

    def test_backward_compatibility_insert(self):
        """Verify old INSERT statements still work (without session_id)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Create table with migration applied
            cursor.execute(f"""
                CREATE TABLE {Tables.RL_OUTCOME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    task_type TEXT,
                    vertical TEXT DEFAULT '',
                    repo_id TEXT DEFAULT NULL,
                    success INTEGER,
                    quality_score REAL,
                    metadata TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    session_id TEXT
                )
            """)

            # Old-style insert (without session_id) should still work
            cursor.execute(f"""
                INSERT INTO {Tables.RL_OUTCOME}
                (learner_id, provider, model, task_type, success, quality_score)
                VALUES
                ('test_learner', 'anthropic', 'claude', 'test_task', 1, 0.85)
            """)

            # Verify insert worked
            cursor.execute(f"SELECT * FROM {Tables.RL_OUTCOME}")
            row = cursor.fetchone()
            assert row is not None
            assert row[1] == "test_learner"  # learner_id
            assert row[11] is None  # session_id should be NULL

            conn.close()

    def test_backward_compatibility_select(self):
        """Verify old SELECT statements still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Create table with migration applied
            cursor.execute(f"""
                CREATE TABLE {Tables.RL_OUTCOME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    task_type TEXT,
                    vertical TEXT DEFAULT '',
                    repo_id TEXT DEFAULT NULL,
                    success INTEGER,
                    quality_score REAL,
                    metadata TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    session_id TEXT
                )
            """)

            # Insert test data
            cursor.execute(f"""
                INSERT INTO {Tables.RL_OUTCOME}
                (learner_id, provider, model, task_type, success, quality_score, session_id)
                VALUES
                ('test_learner', 'anthropic', 'claude', 'test_task', 1, 0.85, 'session123')
            """)

            # Old-style select (without session_id) should still work
            cursor.execute(f"""
                SELECT learner_id, provider, model, quality_score
                FROM {Tables.RL_OUTCOME}
                WHERE provider = 'anthropic'
            """)
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "test_learner"
            assert row[3] == 0.85

            conn.close()

    def test_new_functionality_session_queries(self):
        """Verify new session_id-based queries work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Create table with migration
            cursor.execute(f"""
                CREATE TABLE {Tables.RL_OUTCOME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    task_type TEXT,
                    vertical TEXT DEFAULT '',
                    repo_id TEXT DEFAULT NULL,
                    success INTEGER,
                    quality_score REAL,
                    metadata TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    session_id TEXT
                )
            """)

            # Insert test data with session_id
            for i in range(5):
                cursor.execute(f"""
                    INSERT INTO {Tables.RL_OUTCOME}
                    (learner_id, provider, task_type, success, session_id)
                    VALUES
                    ('test_learner', 'test_provider', 'test_task', 1, 'session_abc')
                """)

            # Query by session_id
            cursor.execute(f"""
                SELECT COUNT(*) FROM {Tables.RL_OUTCOME}
                WHERE session_id = 'session_abc'
            """)
            count = cursor.fetchone()[0]
            assert count == 5, "Should find 5 outcomes for session_abc"

            conn.close()

    def test_new_functionality_repo_queries(self):
        """Verify new repo_id-based queries work with index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Create table with migration and indexes
            cursor.execute(f"""
                CREATE TABLE {Tables.RL_OUTCOME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    task_type TEXT,
                    vertical TEXT DEFAULT '',
                    repo_id TEXT DEFAULT NULL,
                    success INTEGER,
                    quality_score REAL,
                    metadata TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    session_id TEXT
                )
            """)

            # Create indexes
            cursor.execute(
                f"CREATE INDEX idx_rl_outcome_repo ON {Tables.RL_OUTCOME}(repo_id, created_at)"
            )

            # Insert test data with repo_id
            for i in range(3):
                cursor.execute(f"""
                    INSERT INTO {Tables.RL_OUTCOME}
                    (learner_id, provider, task_type, success, repo_id)
                    VALUES
                    ('test_learner', 'test_provider', 'test_task', 1, 'vijaysingh/codingagent')
                """)

            # Query by repo_id
            cursor.execute(f"""
                SELECT COUNT(*) FROM {Tables.RL_OUTCOME}
                WHERE repo_id = 'vijaysingh/codingagent'
            """)
            count = cursor.fetchone()[0]
            assert count == 3, "Should find 3 outcomes for repo"

            conn.close()

    def test_current_schema_version_is_10(self):
        """Verify current schema version is 10 (v10: RL outcome category)."""
        assert (
            CURRENT_SCHEMA_VERSION == 10
        ), f"Current schema version should be 10, got {CURRENT_SCHEMA_VERSION}"

    def test_rl_outcome_with_session_id(self):
        """Test RLOutcome can be created with session_id."""
        outcome = RLOutcome(
            provider="user",
            model="feedback",
            task_type="feedback",
            success=True,
            quality_score=0.9,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "session_id": "test_session_123",
                "feedback_source": "user",
            },
            vertical="general",
        )

        assert outcome.metadata["session_id"] == "test_session_123"
        assert outcome.quality_score == 0.9


class TestPriority4BackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_existing_queries_still_work(self):
        """Verify all existing query patterns still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Create table with migration
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {Tables.RL_OUTCOME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    task_type TEXT,
                    vertical TEXT DEFAULT '',
                    repo_id TEXT DEFAULT NULL,
                    success INTEGER,
                    quality_score REAL,
                    metadata TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    session_id TEXT
                )
            """)

            # Insert test data
            cursor.execute(f"""
                INSERT INTO {Tables.RL_OUTCOME}
                (learner_id, provider, model, task_type, success, quality_score, repo_id, session_id)
                VALUES
                ('tool_selector', 'anthropic', 'claude-sonnet-4-5-20250929', 'tool_call', 1, 0.85, 'test/repo', 'session_123')
            """)

            # Test existing query patterns
            queries = [
                # Query by learner_id
                f"SELECT * FROM {Tables.RL_OUTCOME} WHERE learner_id = 'tool_selector'",
                # Query by provider
                f"SELECT * FROM {Tables.RL_OUTCOME} WHERE provider = 'anthropic'",
                # Query by task_type
                f"SELECT * FROM {Tables.RL_OUTCOME} WHERE task_type = 'tool_call'",
                # Query by quality_score
                f"SELECT * FROM {Tables.RL_OUTCOME} WHERE quality_score > 0.8",
                # Query by repo_id
                f"SELECT * FROM {Tables.RL_OUTCOME} WHERE repo_id = 'test/repo'",
            ]

            for query in queries:
                cursor.execute(query)
                row = cursor.fetchone()
                assert row is not None, f"Query should return results: {query}"

            conn.close()

    def test_rl_coordinator_integration(self):
        """Test RLCoordinator works with new schema."""
        coordinator = get_rl_coordinator()

        # Create outcome with session_id in metadata
        outcome = RLOutcome(
            provider="user",
            model="feedback",
            task_type="feedback",
            success=True,
            quality_score=0.9,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "session_id": "test_session",
                "feedback_source": "user",
            },
            vertical="general",
        )

        # Should not raise
        try:
            coordinator.record_outcome("user_feedback", outcome)
        except Exception as e:
            pytest.fail(f"record_outcome should work with new schema: {e}")


class TestPriority4Performance:
    """Test performance impact of migration."""

    def test_session_index_performance(self):
        """Verify session index improves query performance."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Create table with migration
            cursor.execute(f"""
                CREATE TABLE {Tables.RL_OUTCOME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    learner_id TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    task_type TEXT,
                    vertical TEXT DEFAULT '',
                    repo_id TEXT DEFAULT NULL,
                    success INTEGER,
                    quality_score REAL,
                    metadata TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    session_id TEXT
                )
            """)

            # Create index
            cursor.execute(
                f"CREATE INDEX idx_rl_outcome_session ON {Tables.RL_OUTCOME}(session_id, created_at)"
            )

            # Insert test data (1000 rows)
            for i in range(1000):
                session_id = f"session_{i % 100}"  # 100 unique sessions
                cursor.execute(f"""
                    INSERT INTO {Tables.RL_OUTCOME}
                    (learner_id, provider, task_type, success, session_id)
                    VALUES
                    ('test_learner', 'test_provider', 'test_task', 1, '{session_id}')
                """)

            # Test query performance with index
            start = time.time()
            cursor.execute(f"""
                SELECT * FROM {Tables.RL_OUTCOME}
                WHERE session_id = 'session_50'
                ORDER BY created_at DESC
                LIMIT 10
            """)
            rows = cursor.fetchall()
            elapsed_ms = (time.time() - start) * 1000

            assert len(rows) == 10, "Should find 10 outcomes"
            assert elapsed_ms < 100, f"Query with index should be fast, took {elapsed_ms:.1f}ms"

            conn.close()
