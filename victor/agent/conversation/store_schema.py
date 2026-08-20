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

"""Schema manager for ConversationStore."""

from __future__ import annotations
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.conversation.store import ConversationStore

from victor.core.json_utils import json_loads
from victor.agent.ml_metadata import (
    ContextSize,
    ModelFamily,
    ModelSize,
)

logger = logging.getLogger(__name__)


class ConversationStoreSchema:
    """Manages SQLite database schema setup, lookup caches, and migrations for ConversationStore."""

    def __init__(self, store: ConversationStore):
        """Initialize schema manager.

        Args:
            store: Reference to parent ConversationStore
        """
        self.store = store

    @property
    def db_path(self):
        return self.store.db_path

    @property
    def SCHEMA_VERSION(self):
        return self.store.SCHEMA_VERSION

    def init_database(self) -> None:
        """Initialize SQLite database for persistence with normalized schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Run database migrations first
        try:
            from victor.agent.conversation.migrations import migrate_database

            migrate_database(str(self.db_path))
            logger.debug(f"Database migrations completed for {self.db_path}")
        except Exception as e:
            logger.warning(f"Database migration failed (will continue with existing schema): {e}")

        with sqlite3.connect(self.db_path) as conn:
            # SQLite performance optimizations
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = -2000")
            conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA foreign_keys = ON")

            # Create schema version tracking table
            from victor.agent.conversation.migrations import ensure_schema_version_table

            ensure_schema_version_table(conn)

            # Check if current schema version is applied
            cursor = conn.execute(
                "SELECT version FROM schema_version WHERE version = ?",
                (self.SCHEMA_VERSION,),
            )
            version_applied = cursor.fetchone() is not None

            if not version_applied:
                # Migrate existing sessions table if it exists
                self.migrate_sessions_table(conn)

                # Apply normalized schema
                self.apply_normalized_schema(conn)
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (self.SCHEMA_VERSION, datetime.now().isoformat()),
                )
                conn.commit()
                logger.info(f"Database schema upgraded to v{self.SCHEMA_VERSION}")
            else:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='model_families'"
                )
                if cursor.fetchone() is None:
                    logger.info(
                        "Normalized lookup tables absent in versioned DB — creating them (%s)",
                        self.db_path,
                    )
                    self.ensure_lookup_tables(conn)
                    conn.commit()

                # Load ID caches for existing database
                self.load_lookup_caches(conn)

                # Run migration for any missing columns
                self.migrate_sessions_table(conn)
                conn.commit()

            self.ensure_critical_tables(conn)
            conn.commit()

        logger.debug(f"Database initialized at {self.db_path}")

    def ensure_lookup_tables(self, conn: sqlite3.Connection) -> None:
        """Create only the four normalized lookup tables on an existing database."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS model_families (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                provider_id INTEGER
            );
            CREATE TABLE IF NOT EXISTS model_sizes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                family_id INTEGER,
                num_parameters INTEGER
            );
            CREATE TABLE IF NOT EXISTS context_sizes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                min_tokens INTEGER,
                max_tokens INTEGER
            );
            CREATE TABLE IF NOT EXISTS providers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT
            );
        """)
        self.populate_lookup_tables(conn)
        self.populate_fts5_index(conn)

    def ensure_critical_tables(self, conn: sqlite3.Connection) -> None:
        """Idempotently create core data tables that must always exist."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                last_activity TIMESTAMP NOT NULL,
                project_path TEXT,
                model TEXT,
                profile TEXT,
                max_tokens INTEGER DEFAULT 100000,
                reserved_tokens INTEGER DEFAULT 4096,
                metadata TEXT,
                provider_id INTEGER REFERENCES providers(id),
                model_family_id INTEGER REFERENCES model_families(id),
                model_size_id INTEGER REFERENCES model_sizes(id),
                context_size_id INTEGER REFERENCES context_sizes(id),
                model_params_b REAL,
                context_tokens INTEGER,
                tool_capable INTEGER DEFAULT 0,
                is_moe INTEGER DEFAULT 0,
                is_reasoning INTEGER DEFAULT 0,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                cached_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                cost_usd_micros INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                token_count INTEGER NOT NULL,
                priority INTEGER NOT NULL,
                tool_name TEXT,
                tool_call_id TEXT,
                metadata TEXT,
                agent_id TEXT,
                parent_session_id TEXT,
                team_id TEXT,
                member_id TEXT,
                plan_id TEXT,
                plan_step_id TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS context_summaries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                messages_summarized TEXT,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS compaction_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                agent_id TEXT,
                parent_session_id TEXT,
                team_id TEXT,
                member_id TEXT,
                plan_id TEXT,
                plan_step_id TEXT,
                strategy TEXT NOT NULL,
                messages_removed INTEGER DEFAULT 0,
                tokens_freed INTEGER DEFAULT 0,
                summary TEXT,
                created_at TIMESTAMP NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            -- Full-text search index for message content (FTS5)
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                session_id,
                content_rowid=rowid,
                content=messages
            );
        """)

        try:
            conn.executescript("""
                -- Trigger to insert into FTS5 when message is inserted
                CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content, session_id)
                    VALUES (NEW.rowid, NEW.content, NEW.session_id);
                END;

                -- Trigger to update FTS5 when message is updated
                CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE OF content, session_id ON messages BEGIN
                    UPDATE messages_fts SET content = NEW.content, session_id = NEW.session_id
                    WHERE rowid = NEW.rowid;
                END;

                -- Trigger to delete from FTS5 when message is deleted
                CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
                    DELETE FROM messages_fts WHERE rowid = OLD.rowid;
                END;
            """)
        except Exception as e:
            logger.debug(f"FTS5 triggers creation skipped (may already exist): {e}")

        try:
            self.load_lookup_caches(conn)
        except Exception:
            pass

    def apply_normalized_schema(self, conn: sqlite3.Connection) -> None:
        """Apply the normalized schema with lookup tables."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS model_families (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                provider_id INTEGER
            );

            CREATE TABLE IF NOT EXISTS model_sizes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                family_id INTEGER,
                num_parameters INTEGER,
                FOREIGN KEY (family_id) REFERENCES model_families(id)
                    ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS context_sizes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                min_tokens INTEGER,
                max_tokens INTEGER
            );

            CREATE TABLE IF NOT EXISTS providers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                last_activity TIMESTAMP NOT NULL,
                project_path TEXT,
                model TEXT,
                profile TEXT,
                max_tokens INTEGER DEFAULT 100000,
                reserved_tokens INTEGER DEFAULT 4096,
                metadata TEXT,
                provider_id INTEGER REFERENCES providers(id),
                model_family_id INTEGER REFERENCES model_families(id),
                model_size_id INTEGER REFERENCES model_sizes(id),
                context_size_id INTEGER REFERENCES context_sizes(id),
                model_params_b REAL,
                context_tokens INTEGER,
                tool_capable INTEGER DEFAULT 0,
                is_moe INTEGER DEFAULT 0,
                is_reasoning INTEGER DEFAULT 0,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                cached_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                cost_usd_micros INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                token_count INTEGER NOT NULL,
                priority INTEGER NOT NULL,
                tool_name TEXT,
                tool_call_id TEXT,
                metadata TEXT,
                agent_id TEXT,
                parent_session_id TEXT,
                team_id TEXT,
                member_id TEXT,
                plan_id TEXT,
                plan_step_id TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS context_summaries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                messages_summarized TEXT,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS compaction_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                agent_id TEXT,
                parent_session_id TEXT,
                team_id TEXT,
                member_id TEXT,
                plan_id TEXT,
                plan_step_id TEXT,
                strategy TEXT NOT NULL,
                messages_removed INTEGER DEFAULT 0,
                tokens_freed INTEGER DEFAULT 0,
                summary TEXT,
                created_at TIMESTAMP NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_time
            ON messages(session_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_messages_priority
            ON messages(session_id, priority DESC);

            CREATE INDEX IF NOT EXISTS idx_summaries_session
            ON context_summaries(session_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_compaction_events_session_agent_time
            ON compaction_events(session_id, agent_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_compaction_events_team_time
            ON compaction_events(team_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_sessions_provider
            ON sessions(provider_id);

            CREATE INDEX IF NOT EXISTS idx_sessions_model_family
            ON sessions(model_family_id);

            CREATE INDEX IF NOT EXISTS idx_sessions_model_size
            ON sessions(model_size_id);

            CREATE INDEX IF NOT EXISTS idx_sessions_context_size
            ON sessions(context_size_id);

            CREATE INDEX IF NOT EXISTS idx_sessions_tool_capable
            ON sessions(tool_capable);

            CREATE INDEX IF NOT EXISTS idx_sessions_family_size
            ON sessions(model_family_id, model_size_id);

            CREATE INDEX IF NOT EXISTS idx_sessions_provider_family
            ON sessions(provider_id, model_family_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                message_id UNINDEXED,
                session_id UNINDEXED,
                content='messages',
                content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content, message_id, session_id)
                VALUES (NEW.rowid, NEW.content, NEW.id, NEW.session_id);
            END;

            CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content, message_id, session_id)
                VALUES ('delete', OLD.rowid, OLD.content, OLD.id, OLD.session_id);
            END;

            CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content, message_id, session_id)
                VALUES ('delete', OLD.rowid, OLD.content, OLD.id, OLD.session_id);
                INSERT INTO messages_fts(rowid, content, message_id, session_id)
                VALUES (NEW.rowid, NEW.content, NEW.id, NEW.session_id);
            END;

            CREATE INDEX IF NOT EXISTS idx_messages_tool_results
            ON messages(session_id, role, tool_name, timestamp DESC)
            WHERE role IN ('tool_call', 'tool');

            CREATE INDEX IF NOT EXISTS idx_messages_exchange
            ON messages(session_id, role, timestamp)
            WHERE role IN ('user', 'assistant');

            CREATE INDEX IF NOT EXISTS idx_messages_session_agent_time
            ON messages(session_id, agent_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_messages_agent_time
            ON messages(agent_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_messages_team_time
            ON messages(team_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_messages_plan_step_time
            ON messages(plan_id, plan_step_id, timestamp);
            """)

        self.populate_lookup_tables(conn)
        self.load_lookup_caches(conn)
        self.apply_hybrid_compaction_schema(conn)
        self.auto_maintenance()

    def auto_maintenance(self) -> None:
        """Run lightweight maintenance on startup."""
        _FREELIST_VACUUM_THRESHOLD = 100

        try:
            self.store.cleanup_stale_sessions(
                max_age_days=0,
                purge_test_models=True,
                purge_empty=True,
            )

            with sqlite3.connect(self.db_path) as conn:
                freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
                if freelist > _FREELIST_VACUUM_THRESHOLD:
                    conn.execute("PRAGMA incremental_vacuum")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    logger.info(f"Auto-vacuum: reclaimed {freelist} free pages")
        except sqlite3.Error as e:
            logger.debug(f"Auto-maintenance skipped: {e}")

    def apply_hybrid_compaction_schema(self, conn: sqlite3.Connection) -> None:
        """Apply enhanced schema for hybrid compaction system."""
        try:
            conn.execute("SELECT json_extract('{\"test\": 1}', '$.test')")

            messages_columns = [
                ("metadata_json", "TEXT DEFAULT '{}'"),
                ("priority", "INTEGER DEFAULT 50"),
            ]

            for col_name, col_def in messages_columns:
                try:
                    conn.execute(f"ALTER TABLE messages ADD COLUMN {col_name} {col_def}")
                    logger.info(f"Added column {col_name} to messages table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        logger.warning(f"Failed to add column {col_name} to messages: {e}")

            lineage_columns = [
                ("agent_id", "TEXT"),
                ("parent_session_id", "TEXT"),
                ("team_id", "TEXT"),
                ("member_id", "TEXT"),
                ("plan_id", "TEXT"),
                ("plan_step_id", "TEXT"),
            ]
            for col_name, col_def in lineage_columns:
                try:
                    conn.execute(f"ALTER TABLE messages ADD COLUMN {col_name} {col_def}")
                    logger.info(f"Added column {col_name} to messages table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        logger.warning(f"Failed to add column {col_name} to messages: {e}")
            self.backfill_message_lineage_columns(conn)

            summary_columns = [
                ("summary_format", "TEXT DEFAULT 'natural'"),
                ("summary_xml", "TEXT"),
                ("summary_text", "TEXT"),
                ("summary_json", "TEXT DEFAULT '{}'"),
                ("messages_summarized_json", "TEXT DEFAULT '[]'"),
                ("strategy_used", "TEXT"),
                ("complexity_score", "REAL"),
                ("estimated_tokens_saved", "INTEGER DEFAULT 0"),
            ]

            for col_name, col_def in summary_columns:
                try:
                    conn.execute(f"ALTER TABLE context_summaries ADD COLUMN {col_name} {col_def}")
                    logger.info(f"Added column {col_name} to context_summaries table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        logger.warning(f"Failed to add column {col_name} to context_summaries: {e}")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS compaction_history (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    strategy_used TEXT NOT NULL,
                    message_count_before INTEGER NOT NULL,
                    message_count_after INTEGER NOT NULL,
                    token_count_before INTEGER NOT NULL,
                    token_count_after INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    llm_provider TEXT,
                    llm_model TEXT,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS topic_segments (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    topic_name TEXT NOT NULL,
                    start_message_id TEXT NOT NULL,
                    end_message_id TEXT,
                    message_count INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY (start_message_id) REFERENCES messages(id),
                    FOREIGN KEY (end_message_id) REFERENCES messages(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_strategy
                ON context_summaries(strategy_used, created_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_compaction_history_session
                ON compaction_history(session_id, created_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_compaction_history_strategy
                ON compaction_history(strategy_used, created_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic_segments_session
                ON topic_segments(session_id, created_at)
            """)

            logger.info("Applied hybrid compaction schema enhancements")

        except sqlite3.Error as e:
            logger.warning(f"Failed to apply hybrid compaction schema: {e}")

    def backfill_message_lineage_columns(self, conn: sqlite3.Connection) -> None:
        """Backfill queryable lineage columns from metadata JSON for legacy rows."""
        try:
            rows = conn.execute("""
                SELECT id, metadata FROM messages
                WHERE metadata IS NOT NULL
                  AND metadata != ''
                  AND (
                    agent_id IS NULL OR parent_session_id IS NULL OR team_id IS NULL
                    OR member_id IS NULL OR plan_id IS NULL OR plan_step_id IS NULL
                  )
                """).fetchall()
        except sqlite3.Error as exc:
            logger.debug("Skipping lineage backfill: %s", exc)
            return

        updated = 0
        for message_id, metadata_raw in rows:
            try:
                metadata = json_loads(metadata_raw or "{}")
            except Exception:
                continue
            if not isinstance(metadata, dict):
                continue
            values = {
                "agent_id": metadata.get("agent_id"),
                "parent_session_id": metadata.get("parent_session_id"),
                "team_id": metadata.get("team_id"),
                "member_id": metadata.get("member_id"),
                "plan_id": metadata.get("plan_id"),
                "plan_step_id": metadata.get("plan_step_id"),
            }
            if not any(value is not None for value in values.values()):
                continue
            conn.execute(
                """
                UPDATE messages
                SET agent_id = COALESCE(agent_id, ?),
                    parent_session_id = COALESCE(parent_session_id, ?),
                    team_id = COALESCE(team_id, ?),
                    member_id = COALESCE(member_id, ?),
                    plan_id = COALESCE(plan_id, ?),
                    plan_step_id = COALESCE(plan_step_id, ?)
                WHERE id = ?
                """,
                (
                    values["agent_id"],
                    values["parent_session_id"],
                    values["team_id"],
                    values["member_id"],
                    values["plan_id"],
                    values["plan_step_id"],
                    message_id,
                ),
            )
            updated += 1

        if updated:
            logger.info("Backfilled message lineage columns for %s messages", updated)

    def populate_fts5_index(self, conn: sqlite3.Connection) -> None:
        """Populate FTS5 index from existing messages."""
        try:
            fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
            msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

            if fts_count >= msg_count:
                logger.debug("FTS5 index already populated (%d messages)", fts_count)
                return

            conn.execute("""
                INSERT INTO messages_fts(rowid, content, session_id)
                SELECT rowid, content, session_id FROM messages
            """)
            indexed = conn.execute("SELECT changes()").fetchone()[0]
            logger.info("Populated FTS5 index with %d existing messages", indexed)

        except sqlite3.Error as e:
            logger.debug("FTS5 population skipped: %s", e)

    def populate_lookup_tables(self, conn: sqlite3.Connection) -> None:
        """Populate lookup tables with predefined enum values."""
        families = [
            ("llama", None),
            ("qwen", None),
            ("mistral", None),
            ("mixtral", None),
            ("claude", None),
            ("gpt", None),
            ("gemini", None),
            ("deepseek", None),
            ("phi", None),
            ("codellama", None),
            ("command", None),
            ("grok", None),
            ("unknown", None),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO model_families (name, provider_id) VALUES (?, ?)",
            families,
        )

        sizes = [
            ("tiny", None, 0),
            ("small", None, 4),
            ("medium", None, 20),
            ("large", None, 51),
            ("xlarge", None, 122),
            ("xxlarge", None, 175),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO model_sizes (name, family_id, num_parameters) VALUES (?, ?, ?)",
            sizes,
        )

        ctx_sizes = [
            ("small", 0, 8000),
            ("medium", 8000, 32000),
            ("large", 32000, 128000),
            ("xlarge", 128000, 999999999),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO context_sizes (name, min_tokens, max_tokens) VALUES (?, ?, ?)",
            ctx_sizes,
        )

        providers = [
            ("anthropic", "Anthropic Claude API"),
            ("openai", "OpenAI API"),
            ("ollama", "Local Ollama server"),
            ("groq", "Groq Cloud API"),
            ("google", "Google Gemini API"),
            ("xai", "xAI Grok API"),
            ("deepseek", "DeepSeek API"),
            ("mistral", "Mistral AI API"),
            ("lmstudio", "Local LMStudio server"),
            ("vllm", "vLLM server"),
            ("together", "Together AI API"),
            ("openrouter", "OpenRouter API"),
            ("moonshot", "Moonshot/Kimi API"),
            ("fireworks", "Fireworks AI API"),
            ("cerebras", "Cerebras API"),
            ("unknown", "Unknown provider"),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO providers (name, description) VALUES (?, ?)",
            providers,
        )

        conn.commit()

    def load_lookup_caches(self, conn: sqlite3.Connection) -> None:
        """Load lookup table IDs into memory caches for fast lookup."""
        rows = conn.execute("SELECT name, id FROM model_families").fetchall()
        self.store._model_family_ids = {row[0]: row[1] for row in rows}

        rows = conn.execute("SELECT name, id FROM model_sizes").fetchall()
        self.store._model_size_ids = {row[0]: row[1] for row in rows}

        rows = conn.execute("SELECT name, id FROM context_sizes").fetchall()
        self.store._context_size_ids = {row[0]: row[1] for row in rows}

        rows = conn.execute("SELECT name, id FROM providers").fetchall()
        self.store._provider_ids = {row[0]: row[1] for row in rows}

        logger.debug(
            f"Loaded lookup caches: {len(self.store._model_family_ids)} families, "
            f"{len(self.store._provider_ids)} providers"
        )

    def migrate_sessions_table(self, conn: sqlite3.Connection) -> None:
        """Add missing columns to sessions table for schema migration."""
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        result = cursor.fetchone()
        if not result:
            return

        table_sql = result[0]

        # Self-heal very old `sessions` tables that predate the current core
        # columns. Without this, ConversationStore init fails hard (e.g. "table
        # sessions has no column named session_id" / "... last_activity"). Add the
        # whole core set in one pass rather than one column per restart. ALTER
        # cannot re-add a PRIMARY KEY or a NOT NULL without a default, but plain/
        # defaulted columns are enough to unblock INSERT OR REPLACE on this cache.
        core_columns = [
            ("session_id", "TEXT"),
            ("created_at", "TIMESTAMP"),
            ("last_activity", "TIMESTAMP"),
            ("project_path", "TEXT"),
            ("model", "TEXT"),
            ("profile", "TEXT"),
            ("max_tokens", "INTEGER DEFAULT 100000"),
            ("reserved_tokens", "INTEGER DEFAULT 4096"),
            ("metadata", "TEXT"),
        ]
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        healed = []
        for col_name, col_def in core_columns:
            if col_name not in existing_cols:
                try:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_def}")
                    healed.append(col_name)
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        logger.warning(f"Failed to add {col_name} to sessions: {e}")
        if healed:
            logger.warning(f"Healed legacy sessions table: added missing columns {healed}")

        new_columns = [
            ("provider_id", "INTEGER"),
            ("model_family_id", "INTEGER"),
            ("model_size_id", "INTEGER"),
            ("context_size_id", "INTEGER"),
            ("model_params_b", "REAL"),
            ("context_tokens", "INTEGER"),
            ("tool_capable", "INTEGER DEFAULT 0"),
            ("is_moe", "INTEGER DEFAULT 0"),
            ("is_reasoning", "INTEGER DEFAULT 0"),
            ("prompt_tokens", "INTEGER DEFAULT 0"),
            ("completion_tokens", "INTEGER DEFAULT 0"),
            ("cached_tokens", "INTEGER DEFAULT 0"),
            ("reasoning_tokens", "INTEGER DEFAULT 0"),
            ("cost_usd_micros", "INTEGER DEFAULT 0"),
        ]

        columns_added = []
        for col_name, col_def in new_columns:
            if col_name not in table_sql:
                try:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_def}")
                    columns_added.append(col_name)
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        logger.warning(f"Failed to add column {col_name} to sessions: {e}")

        if columns_added:
            logger.info(f"Migrated sessions table: added columns {columns_added}")

    def get_or_create_provider_id(
        self, conn: sqlite3.Connection, provider: Optional[str]
    ) -> Optional[int]:
        """Get or create provider ID, with caching."""
        if not provider:
            return None

        provider_lower = provider.lower()

        if provider_lower in self.store._provider_ids:
            return self.store._provider_ids[provider_lower]

        try:
            conn.execute(
                "INSERT OR IGNORE INTO providers (name) VALUES (?)",
                (provider_lower,),
            )
            cursor = conn.execute(
                "SELECT id FROM providers WHERE name = ?",
                (provider_lower,),
            )
            row = cursor.fetchone()
            if row:
                self.store._provider_ids[provider_lower] = row[0]
                return row[0]
        except sqlite3.Error as e:
            logger.warning(f"Failed to get/create provider ID for {provider}: {e}")

        return None

    def get_model_family_id(self, family: Optional[ModelFamily]) -> Optional[int]:
        """Get model family ID from cache."""
        if not family:
            return None
        return self.store._model_family_ids.get(family.value)

    def get_model_size_id(self, size: Optional[ModelSize]) -> Optional[int]:
        """Get model size ID from cache."""
        if not size:
            return None
        return self.store._model_size_ids.get(size.value)

    def get_context_size_id(self, ctx_size: Optional[ContextSize]) -> Optional[int]:
        """Get context size ID from cache."""
        if not ctx_size:
            return None
        return self.store._context_size_ids.get(ctx_size.value)
