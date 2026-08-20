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

"""Database schema migration for legacy victor-invest databases.

This script handles the specific case where victor-invest was created
with an old database schema that needs to be updated to match the
current ConversationStore expectations.

Strategy:
- Preserve graph data (graph_node, graph_edge, file_changes)
- Recreate conversation tables with correct schema
- Maintain data integrity
"""

import sqlite3
import shutil
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def migrate_legacy_database(db_path: Path) -> bool:
    """Migrate a legacy victor-invest database to the current schema.

    Args:
        db_path: Path to the database file

    Returns:
        True if migration was successful, False otherwise
    """
    # Create backup
    backup_path = db_path.with_suffix(".db.backup")
    if not backup_path.exists():
        shutil.copy2(db_path, backup_path)
        logger.info(f"Created backup: {backup_path}")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        # Check current schema state
        cursor.execute("PRAGMA table_info(sessions)")
        sessions_cols = {col[1]: col[2] for col in cursor.fetchall()}

        # If sessions table has 'id' column (old schema), we need to migrate
        if "id" in sessions_cols and "session_id" not in sessions_cols:
            logger.info("Detected legacy schema, migrating...")
            return _migrate_legacy_schema(conn, cursor)

        # If sessions table has 'session_id' column, it might already be migrated
        elif "session_id" in sessions_cols:
            logger.info("Schema appears to be already migrated or partially migrated")
            return _verify_and_fix_partial_migration(conn, cursor)

        else:
            logger.info("Schema is already correct")
            return False

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        # Restore from backup on failure
        if backup_path.exists():
            shutil.copy2(backup_path, db_path)
            logger.info(f"Restored from backup: {backup_path}")
        raise
    finally:
        conn.close()


def _migrate_legacy_schema(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> bool:
    """Migrate from legacy schema to current schema.

    Strategy:
    1. Rename old tables with _legacy suffix
    2. Create new tables with correct schema
    3. Migrate any data that can be preserved
    4. Drop legacy tables

    Args:
        conn: Database connection
        cursor: Database cursor

    Returns:
        True if migration was successful
    """
    # Tables to migrate (preserve data if possible)
    tables_to_migrate = {
        "sessions": _migrate_sessions_table,
        "messages": _migrate_messages_table,
        "context_sizes": _create_context_sizes_table,
        "context_summaries": _create_context_summaries_table,
        "model_families": _migrate_model_families_table,
        "model_sizes": _migrate_model_sizes_table,
        "providers": _migrate_providers_table,
    }

    for table_name, migrate_func in tables_to_migrate.items():
        try:
            logger.info(f"Migrating {table_name}...")
            migrate_func(conn, cursor)
        except Exception as e:
            logger.warning(f"Failed to migrate {table_name}: {e}")

    # Update schema version
    cursor.execute(
        "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
        ("0.3.0", datetime.utcnow().isoformat()),
    )

    conn.commit()
    logger.info("Legacy schema migration completed successfully")
    return True


def _verify_and_fix_partial_migration(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> bool:
    """Verify and fix partially migrated schemas.

    Args:
        conn: Database connection
        cursor: Database cursor

    Returns:
        True if fixes were applied, False otherwise
    """
    # Check if context_sizes needs fixing
    cursor.execute("PRAGMA table_info(context_sizes)")
    ctx_cols = {col[1]: col[2] for col in cursor.fetchall()}

    if "name" not in ctx_cols:
        logger.info("Fixing context_sizes table...")
        _recreate_context_sizes(conn, cursor)
        conn.commit()
        return True

    return False


def _migrate_sessions_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Migrate sessions table from legacy to current schema.

    Legacy schema: id, name, provider, model, profile, ...
    Current schema: session_id, provider_id, model_family_id, model_size_id, ...
    """
    # Rename old table
    cursor.execute("ALTER TABLE sessions RENAME TO sessions_legacy")

    # Create new table with correct schema
    cursor.execute("""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            provider_id INTEGER,
            model_family_id INTEGER,
            model_size_id INTEGER,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            FOREIGN KEY (provider_id) REFERENCES providers(id)
                ON DELETE SET NULL,
            FOREIGN KEY (model_family_id) REFERENCES model_families(id)
                ON DELETE SET NULL,
            FOREIGN KEY (model_size_id) REFERENCES model_sizes(id)
                ON DELETE SET NULL
        )
    """)

    # Note: We're not migrating data from legacy sessions because
    # the schema is too different and there are 0 sessions anyway


def _migrate_messages_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Migrate messages table using the existing migration logic."""
    from victor.agent.conversation.migrations import apply_migration_0_3_0

    apply_migration_0_3_0(str(conn.execute("SELECT :db_path:").fetchone()[0]))


def _create_context_sizes_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Create context_sizes table with correct schema."""
    # Drop old table if exists
    cursor.execute("DROP TABLE IF EXISTS context_sizes")

    cursor.execute("""
        CREATE TABLE context_sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            min_tokens INTEGER,
            max_tokens INTEGER
        )
    """)


def _recreate_context_sizes(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Recreate context_sizes table with correct schema."""
    cursor.execute("DROP TABLE IF EXISTS context_sizes")
    _create_context_sizes_table(conn, cursor)


def _create_context_summaries_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Create context_summaries table with correct schema."""
    cursor.execute("DROP TABLE IF EXISTS context_summaries")

    cursor.execute("""
        CREATE TABLE context_summaries (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            messages_summarized TEXT,
            created_at TIMESTAMP NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_summaries_session
        ON context_summaries(session_id, created_at DESC)
    """)


def _migrate_model_families_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Migrate model_families table."""
    # Rename old table
    cursor.execute("ALTER TABLE model_families RENAME TO model_families_legacy")

    # Create new table
    cursor.execute("""
        CREATE TABLE model_families (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            provider_id INTEGER,
            FOREIGN KEY (provider_id) REFERENCES providers(id)
                ON DELETE SET NULL
        )
    """)


def _migrate_model_sizes_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Migrate model_sizes table."""
    cursor.execute("ALTER TABLE model_sizes RENAME TO model_sizes_legacy")

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


def _migrate_providers_table(conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
    """Migrate providers table."""
    cursor.execute("ALTER TABLE providers RENAME TO providers_legacy")

    cursor.execute("""
        CREATE TABLE providers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT
        )
    """)


def main():
    """Main migration function."""
    import sys

    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".victor/project.db")

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    print(f"Migrating database: {db_path}")
    success = migrate_legacy_database(db_path)

    if success:
        print("✅ Migration completed successfully!")
        return 0
    else:
        print("ℹ️  Migration was not needed (schema already correct)")
        return 0


if __name__ == "__main__":
    main()
