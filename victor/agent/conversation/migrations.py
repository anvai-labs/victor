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

"""Database migration system for ConversationStore.

This module provides schema migration support to handle database schema
changes over time without losing data.
"""

from victor.core.json_utils import json_loads
import sqlite3
import logging
from typing import Optional, List, Callable
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class Migration:
    """Represents a single database migration."""

    def __init__(
        self,
        version: str,
        description: str,
        up_sql: str,
        down_sql: Optional[str] = None,
    ):
        """Initialize a migration.

        Args:
            version: Schema version (e.g., "0.3.0")
            description: Human-readable description
            up_sql: SQL to apply migration
            down_sql: SQL to rollback migration (optional)
        """
        self.version = version
        self.description = description
        self.up_sql = up_sql
        self.down_sql = down_sql


class MigrationRunner:
    """Runs database migrations in order."""

    def __init__(self, db_path: str):
        """Initialize migration runner.

        Args:
            db_path: Path to database file
        """
        self.db_path = db_path
        self.migrations: List[Migration] = []

    def register(self, migration: Migration) -> None:
        """Register a migration.

        Args:
            migration: Migration to register
        """
        self.migrations.append(migration)
        # Sort by version
        self.migrations.sort(key=lambda m: m.version)

    def get_current_version(self) -> Optional[str]:
        """Get current schema version from database.

        Returns:
            Current version or None if no schema_version table
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Check if schema_version table exists first
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if cursor.fetchone() is None:
                # Create schema_version table if it doesn't exist
                from victor.agent.conversation.migrations import (
                    ensure_schema_version_table,
                )

                ensure_schema_version_table(conn)

            cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()

    def run_migrations(self) -> None:
        """Run all pending migrations."""
        current_version = self.get_current_version()

        for migration in self.migrations:
            if current_version is None or migration.version > current_version:
                logger.info(f"Running migration {migration.version}: {migration.description}")
                self._apply_migration(migration)

    def _apply_migration(self, migration: Migration) -> None:
        """Apply a single migration.

        Args:
            migration: Migration to apply
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Execute migration in transaction
            with conn:
                # Apply migration SQL
                conn.executescript(migration.up_sql)

                # Update schema version
                conn.execute(
                    """
                    INSERT INTO schema_version (version, applied_at)
                    VALUES (?, ?)
                    """,
                    (migration.version, datetime.utcnow().isoformat()),
                )

            logger.info(f"Migration {migration.version} applied successfully")
        except Exception as e:
            conn.rollback()
            logger.error(f"Migration {migration.version} failed: {e}")
            raise
        finally:
            conn.close()


# Define migrations
MIGRATIONS = [
    Migration(
        version="0.3.0",
        description="Add timestamp column to messages table and align schemas",
        up_sql="""
            -- This migration is handled by custom Python logic below
            -- due to the need for conditional schema changes
        """,
    ),
]


def apply_migration_0_3_0(db_path: str) -> bool:
    """Apply migration 0.3.0 - Add timestamp column and align schemas.

    This is a custom migration that checks the current schema and applies
    appropriate changes based on what exists.

    Args:
        db_path: Path to database file

    Returns:
        True if migration was applied, False if already up-to-date
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # Migrate messages table
        messages_migrated = _migrate_messages_table(cursor)

        # Migrate model_families table
        model_families_migrated = _migrate_model_families_table(cursor)

        # Migrate model_sizes table
        model_sizes_migrated = _migrate_model_sizes_table(cursor)

        # Migrate context_sizes table
        context_sizes_migrated = _migrate_context_sizes_table(cursor)

        # Ensure multi-agent compaction event ledger exists
        compaction_events_migrated = _ensure_compaction_events_table(cursor)

        # Commit if any migration was applied
        if (
            messages_migrated
            or model_families_migrated
            or model_sizes_migrated
            or context_sizes_migrated
            or compaction_events_migrated
        ):
            conn.commit()
            return True

        return False

    except Exception as e:
        conn.rollback()
        logger.error(f"Migration 0.3.0 failed: {e}")
        raise
    finally:
        conn.close()


def _migrate_messages_table(cursor: sqlite3.Cursor) -> bool:
    """Migrate messages table to add timestamp column.

    Args:
        cursor: Database cursor

    Returns:
        True if migration was applied, False if already up-to-date
    """
    # Check current messages table schema
    cursor.execute("PRAGMA table_info(messages)")
    columns = {col[1]: col[2] for col in cursor.fetchall()}

    if not columns:
        logger.debug("Messages table does not exist yet, skipping migration")
        return False

    logger.debug(f"Current messages table columns: {list(columns.keys())}")

    # Define target schema
    required_columns = {
        "id": "TEXT",
        "session_id": "TEXT",
        "role": "TEXT",
        "content": "TEXT",
        "timestamp": "TIMESTAMP",
        "token_count": "INTEGER",
        "priority": "INTEGER",
        "tool_name": "TEXT",
        "tool_call_id": "TEXT",
        "metadata": "TEXT",
        "agent_id": "TEXT",
        "parent_session_id": "TEXT",
        "team_id": "TEXT",
        "member_id": "TEXT",
        "plan_id": "TEXT",
        "plan_step_id": "TEXT",
    }

    # Check if we're already at the target schema
    has_all_columns = all(col in columns for col in required_columns.keys())
    if has_all_columns:
        backfilled_count = _backfill_agent_lineage_from_metadata(cursor)
        _create_agent_lineage_indexes(cursor)
        logger.debug("Messages table schema is already up-to-date")
        return backfilled_count > 0

    # Check if we need a full table recreation (schema is too different)
    needs_recreation = (
        "id" in columns
        and columns["id"] == "INTEGER"  # Wrong type
        or "created_at" in columns  # Old column name
        or "tool_calls" in columns  # Old column name
    )

    if needs_recreation:
        logger.info("Performing full table recreation for schema migration")
        _recreate_messages_table(cursor, columns)
        _backfill_agent_lineage_from_metadata(cursor)
        logger.info("Successfully migrated messages table via recreation")
        return True
    else:
        # Just add missing columns
        logger.info("Adding missing columns to messages table")
        missing_cols = [
            (col_name, col_type)
            for col_name, col_type in required_columns.items()
            if col_name not in columns
        ]
        for col_name, col_type in missing_cols:
            try:
                cursor.execute(f"ALTER TABLE messages ADD COLUMN {col_name} {col_type}")
                logger.debug(f"Added column {col_name}")
            except Exception as e:
                logger.warning(f"Failed to add column {col_name}: {e}")
        logger.info(f"Added {len(missing_cols)} columns to messages table")
        _backfill_agent_lineage_from_metadata(cursor)
        _create_agent_lineage_indexes(cursor)
        return True


def _backfill_agent_lineage_from_metadata(cursor: sqlite3.Cursor) -> int:
    """Backfill lineage columns from metadata JSON for existing messages."""
    try:
        cursor.execute("""
            SELECT id, metadata FROM messages
            WHERE metadata IS NOT NULL
              AND metadata != ''
              AND (
                agent_id IS NULL OR parent_session_id IS NULL OR team_id IS NULL
                OR member_id IS NULL OR plan_id IS NULL OR plan_step_id IS NULL
              )
            """)
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        logger.debug("Skipping message lineage metadata backfill: %s", exc)
        return 0

    updated = 0
    for message_id, metadata_raw in rows:
        try:
            metadata = json_loads(metadata_raw or "{}")
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue
        values = (
            metadata.get("agent_id"),
            metadata.get("parent_session_id"),
            metadata.get("team_id"),
            metadata.get("member_id"),
            metadata.get("plan_id"),
            metadata.get("plan_step_id"),
            message_id,
        )
        if not any(value is not None for value in values[:-1]):
            continue
        cursor.execute(
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
            values,
        )
        updated += 1

    if updated:
        logger.info("Backfilled message lineage columns from metadata for %s rows", updated)
    return updated


def _create_agent_lineage_indexes(cursor: sqlite3.Cursor) -> None:
    """Create indexes used for multi-agent message isolation."""
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_session_agent_time ON messages(session_id, agent_id, timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_agent_time ON messages(agent_id, timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_team_time ON messages(team_id, timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_plan_step_time ON messages(plan_id, plan_step_id, timestamp)"
    )


def _ensure_compaction_events_table(cursor: sqlite3.Cursor) -> bool:
    """Ensure per-agent compaction audit table exists."""
    required_columns = {
        "id": "TEXT",
        "session_id": "TEXT",
        "agent_id": "TEXT",
        "parent_session_id": "TEXT",
        "team_id": "TEXT",
        "member_id": "TEXT",
        "plan_id": "TEXT",
        "plan_step_id": "TEXT",
        "strategy": "TEXT",
        "messages_removed": "INTEGER DEFAULT 0",
        "tokens_freed": "INTEGER DEFAULT 0",
        "summary": "TEXT",
        "created_at": "TIMESTAMP",
        "metadata": "TEXT",
    }

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='compaction_events'")
    if cursor.fetchone() is not None:
        cursor.execute("PRAGMA table_info(compaction_events)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        changed = False
        for column_name, column_type in required_columns.items():
            if column_name in existing_columns:
                continue
            cursor.execute(f"ALTER TABLE compaction_events ADD COLUMN {column_name} {column_type}")
            changed = True
        _create_compaction_event_indexes(cursor)
        return changed

    cursor.execute("""
        CREATE TABLE compaction_events (
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
        )
    """)
    _create_compaction_event_indexes(cursor)
    return True


def _create_compaction_event_indexes(cursor: sqlite3.Cursor) -> None:
    """Create indexes used for per-agent compaction event queries."""
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_compaction_events_session_agent_time ON compaction_events(session_id, agent_id, created_at DESC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_compaction_events_team_time ON compaction_events(team_id, created_at DESC)"
    )


def _migrate_model_sizes_table(cursor: sqlite3.Cursor) -> bool:
    """Migrate model_sizes table from old schema to new normalized schema.

    Old schema: (id, name, min_params_b, max_params_b)
    New schema: (id, name, family_id, num_parameters)

    Args:
        cursor: Database cursor

    Returns:
        True if migration was applied, False if already up-to-date
    """
    # Check current model_sizes table schema
    cursor.execute("PRAGMA table_info(model_sizes)")
    columns = {col[1]: col[2] for col in cursor.fetchall()}

    logger.debug(f"Current model_sizes table columns: {list(columns.keys())}")

    # Check if already using new schema
    if "family_id" in columns and "num_parameters" in columns:
        logger.debug("model_sizes table already using new schema")
        return False

    # Check if using old schema
    if "min_params_b" in columns and "max_params_b" in columns:
        logger.info("Migrating model_sizes table from old schema to new schema")

        # Read existing data
        cursor.execute("SELECT id, name, min_params_b, max_params_b FROM model_sizes")
        existing_rows = cursor.fetchall()

        # Rename old table
        cursor.execute("ALTER TABLE model_sizes RENAME TO model_sizes_old")

        # Create new table with correct schema
        cursor.execute("""
            CREATE TABLE model_sizes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                family_id INTEGER,
                num_parameters INTEGER,
                FOREIGN KEY (family_id) REFERENCES model_families(id)
                    ON DELETE SET NULL
            )
        """)

        # Migrate data (use midpoint of range as num_parameters)
        for row_id, name, min_params, max_params in existing_rows:
            # Calculate midpoint as num_parameters
            midpoint = int((min_params + max_params) / 2) if min_params is not None else 0
            cursor.execute(
                "INSERT INTO model_sizes (id, name, family_id, num_parameters) VALUES (?, ?, ?, ?)",
                (row_id, name, None, midpoint),
            )

        # Drop old table
        cursor.execute("DROP TABLE model_sizes_old")

        logger.info(f"Successfully migrated {len(existing_rows)} rows in model_sizes table")
        return True

    logger.debug("model_sizes table schema is already up-to-date")
    return False


def _migrate_model_families_table(cursor: sqlite3.Cursor) -> bool:
    """Migrate model_families table from old schema to new normalized schema.

    Old schema: (id, name, description)
    New schema: (id, name, provider_id)

    Args:
        cursor: Database cursor

    Returns:
        True if migration was applied, False if already up-to-date
    """
    # Check current model_families table schema
    cursor.execute("PRAGMA table_info(model_families)")
    columns = {col[1]: col[2] for col in cursor.fetchall()}

    logger.debug(f"Current model_families table columns: {list(columns.keys())}")

    # Check if already using new schema
    if "provider_id" in columns:
        logger.debug("model_families table already using new schema")
        return False

    # Check if using old schema
    if "description" in columns:
        logger.info("Migrating model_families table from old schema to new schema")

        # Read existing data
        cursor.execute("SELECT id, name FROM model_families")
        existing_rows = cursor.fetchall()

        # Rename old table
        cursor.execute("ALTER TABLE model_families RENAME TO model_families_old")

        # Create new table with correct schema
        cursor.execute("""
            CREATE TABLE model_families (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                provider_id INTEGER,
                FOREIGN KEY (provider_id) REFERENCES providers(id)
                    ON DELETE SET NULL
            )
        """)

        # Migrate data (provider_id will be NULL for all existing rows)
        for row_id, name in existing_rows:
            cursor.execute(
                "INSERT INTO model_families (id, name, provider_id) VALUES (?, ?, ?)",
                (row_id, name, None),
            )

        # Drop old table
        cursor.execute("DROP TABLE model_families_old")

        logger.info(f"Successfully migrated {len(existing_rows)} rows in model_families table")
        return True

    logger.debug("model_families table schema is already up-to-date")
    return False


def _migrate_context_sizes_table(cursor: sqlite3.Cursor) -> bool:
    """Migrate context_sizes table to correct lookup table schema.

    Correct schema: (id, name, min_tokens, max_tokens)
    Wrong schema: (id, session_id, token_count, message_count, created_at)

    Args:
        cursor: Database cursor

    Returns:
        True if migration was applied, False if already up-to-date
    """
    # Check current context_sizes table schema
    cursor.execute("PRAGMA table_info(context_sizes)")
    columns = {col[1]: col[2] for col in cursor.fetchall()}

    logger.debug(f"Current context_sizes table columns: {list(columns.keys())}")

    # Check if already using correct schema
    if "name" in columns and "min_tokens" in columns and "max_tokens" in columns:
        logger.debug("context_sizes table already using correct schema")
        return False

    # Wrong schema detected - need to recreate
    logger.info("Migrating context_sizes table to correct lookup schema")

    # Drop the wrong table (no data to preserve since it's a lookup table)
    cursor.execute("DROP TABLE IF EXISTS context_sizes")

    # Create correct table
    cursor.execute("""
        CREATE TABLE context_sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            min_tokens INTEGER,
            max_tokens INTEGER
        )
    """)

    logger.info("Successfully recreated context_sizes table with correct schema")
    return True


def _recreate_messages_table(cursor: sqlite3.Cursor, current_columns: dict) -> None:
    """Recreate messages table with correct schema.

    Args:
        cursor: Database cursor
        current_columns: Current column mapping
    """
    # Determine data source based on current schema
    use_created_at = "created_at" in current_columns
    use_tool_calls = "tool_calls" in current_columns
    has_integer_id = "id" in current_columns and current_columns["id"] == "INTEGER"

    # Create new table with correct schema
    cursor.execute("""
        CREATE TABLE messages_new (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0,
            priority INTEGER NOT NULL DEFAULT 0,
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
        )
    """)

    # Build SELECT clause based on current columns
    select_parts = []
    if has_integer_id:
        select_parts.append("CAST(id AS TEXT) as id")
    else:
        select_parts.append("id")

    select_parts.extend(
        [
            "session_id",
            "role",
            "content",
        ]
    )

    # Add timestamp source
    if "timestamp" in current_columns:
        select_parts.append("timestamp")
    elif use_created_at:
        select_parts.append("created_at as timestamp")
    else:
        select_parts.append("'2024-01-01T00:00:00' as timestamp")

    # Add other columns with defaults
    if "token_count" in current_columns:
        select_parts.append("token_count")
    else:
        select_parts.append("0 as token_count")

    if "priority" in current_columns:
        select_parts.append("priority")
    else:
        select_parts.append("0 as priority")

    if "tool_name" in current_columns:
        select_parts.append("tool_name")
    elif use_tool_calls:
        select_parts.append("tool_calls as tool_name")
    else:
        select_parts.append("NULL as tool_name")

    if "tool_call_id" in current_columns:
        select_parts.append("tool_call_id")
    else:
        select_parts.append("NULL as tool_call_id")

    if "metadata" in current_columns:
        select_parts.append("metadata")
    else:
        select_parts.append("NULL as metadata")

    lineage_columns = [
        "agent_id",
        "parent_session_id",
        "team_id",
        "member_id",
        "plan_id",
        "plan_step_id",
    ]
    for column in lineage_columns:
        if column in current_columns:
            select_parts.append(column)
        else:
            select_parts.append(f"NULL as {column}")

    # Copy data to new table
    cursor.execute(f"""
        INSERT INTO messages_new (
            id, session_id, role, content, timestamp,
            token_count, priority, tool_name, tool_call_id, metadata,
            agent_id, parent_session_id, team_id, member_id, plan_id, plan_step_id
        )
        SELECT {', '.join(select_parts)} FROM messages
    """)

    # Drop old table and rename
    cursor.execute("DROP TABLE messages")
    cursor.execute("ALTER TABLE messages_new RENAME TO messages")

    # Recreate indexes
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_session_time ON messages(session_id, timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_priority ON messages(session_id, priority DESC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_tool_results ON messages(session_id, role, tool_name, timestamp DESC) WHERE role IN ('tool_call', 'tool_result')"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_exchange ON messages(session_id, role, timestamp) WHERE role IN ('user', 'assistant')"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_session_agent_time ON messages(session_id, agent_id, timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_agent_time ON messages(agent_id, timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_team_time ON messages(team_id, timestamp)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_plan_step_time ON messages(plan_id, plan_step_id, timestamp)"
    )


def migrate_database(db_path: str) -> None:
    """Migrate a database to the latest schema version.

    Args:
        db_path: Path to database file
    """
    # Run custom migration 0.3.0 for timestamp column
    try:
        apply_migration_0_3_0(db_path)
    except Exception as e:
        logger.warning(f"Migration 0.3.0 failed (may already be applied): {e}")

    # Update schema version to 0.3.0 only for existing databases that already have tables.
    # Fresh databases get their schema version set by ConversationStore._apply_normalized_schema()
    # after all tables are created. Inserting the version here on a fresh DB would cause
    # store.__init__ to skip _apply_normalized_schema and call _load_lookup_caches instead,
    # failing because model_families doesn't exist yet.
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema_version_table(conn)

        # Only record the version if the core tables already exist (existing DB)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        if cursor.fetchone() is None:
            # Fresh DB — schema version will be set by _apply_normalized_schema
            logger.debug("Skipping schema version record for fresh database (no tables yet)")
            return

        # Check if version 0.3.0 is already recorded
        cursor.execute("SELECT version FROM schema_version WHERE version = ?", ("0.3.0",))
        if cursor.fetchone() is None:
            # Insert version record for existing DB that was just migrated
            cursor.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                ("0.3.0", datetime.utcnow().isoformat()),
            )
            conn.commit()
            logger.info(f"Database {db_path} upgraded to schema version 0.3.0")
    finally:
        conn.close()


def ensure_schema_version_table(conn: sqlite3.Connection) -> None:
    """Ensure schema_version table exists.

    Args:
        conn: Database connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL
        );
        """)
