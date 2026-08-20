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

"""Unified database managers for Victor.

Includes:
- DatabaseManager: Global singleton for user-level data (sync + async APIs)
- ProjectDatabaseManager: Per-project data

Provides two database scopes to separate user-level and project-level data.

See victor.core.schema for table constants and SQL definitions.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │              GLOBAL DATABASE (~/.victor/victor.db)                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │  User-level data shared across all projects:                        │
    │  ├── RL Learning: rl_outcomes, *_q_values, tool/model preferences   │
    │  ├── Teams: team_composition_stats, team_execution_history          │
    │  ├── Sessions: sessions (interactive CLI session persistence)       │
    │  ├── Signatures: failed_signatures (loop prevention)                │
    │  └── Cross-Vertical: patterns learned across projects               │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │              PROJECT DATABASE (.victor/project.db)                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Project-specific data (per-repo):                                  │
    │  ├── Graph: symbols, references, definitions                        │
    │  ├── Conversations: project conversation history                    │
    │  ├── Entities: project entity memory                                │
    │  └── Mode Learning: project-specific mode preferences               │
    └─────────────────────────────────────────────────────────────────────┘

Why Two Databases:
    - Keeps user preferences/RL separate from project-specific data
    - Prevents pollution of global learnings with project-specific patterns
    - Allows sharing project.db in team settings if desired
    - Enables independent backup/migration of each scope

Migration:
    Existing databases are automatically migrated on first access:
    - ~/.victor/graph/graph.db → global (RL tables) + project (graph tables)
    - ~/.victor/team_learning.db → global
    - ~/.victor/sessions.db → global
    - ~/.victor/signatures.db → global
    - .victor/conversation.db → project

Usage:
    from victor.core.database import get_database, get_project_database

    # Global database (user-level)
    global_db = get_database()
    global_db.execute("INSERT INTO rl_outcomes ...", params)

    # Project database (repo-level)
    project_db = get_project_database()
    project_db.query("SELECT * FROM symbols WHERE name = ?", ("MyClass",))
"""

from __future__ import annotations

import asyncio
import logging
import queue
import shutil
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Substrings of sqlite3.OperationalError messages that are expected when a
# migration statement is applied to a database that already satisfies it (an
# idempotent re-run, or a cross-scope statement that targets tables owned by the
# other database). These are benign and logged at DEBUG rather than WARNING.
_BENIGN_MIGRATION_ERRORS = (
    "no such table",
    "duplicate column name",
    "already exists",
    "already another table",
)


def _apply_migration_sql(conn: sqlite3.Connection, sql: str) -> None:
    """Apply a single migration statement (or multi-statement block) idempotently.

    Migration entries may be a single statement or a multi-statement block (e.g.
    the ``Schema.*_INDEXES`` constants). ``sqlite3.Connection.execute`` rejects
    the latter ("You can only execute one statement at a time"), so blocks are
    routed to ``executescript``. Errors that are expected on an already-migrated
    or cross-scope database are logged at DEBUG; anything else stays a WARNING.

    Args:
        conn: Target database connection.
        sql: A SQL statement or ``;``-separated block from ``get_migration_sql``.
    """
    # A block still containing ';' after dropping trailing whitespace/semicolons
    # holds more than one statement and must go through executescript.
    is_multi = ";" in sql.strip().rstrip(";").rstrip()
    try:
        if is_multi:
            conn.executescript(sql)
        else:
            conn.execute(sql)
    except sqlite3.OperationalError as e:
        message = str(e).lower()
        if any(token in message for token in _BENIGN_MIGRATION_ERRORS):
            logger.debug(f"Migration SQL skipped (already satisfied): {e}")
        else:
            logger.warning(f"Migration SQL failed: {e}")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Migration SQL failed: {e}")


# =============================================================================
# DATABASE CONSOLIDATION
# =============================================================================


class DatabaseConsolidator:
    """Handle database consolidation migration to canonical two-database structure.

    Target Architecture:
        ~/.victor/victor.db - Global database (user-wide data)
            - Settings, API keys, profiles
            - RL learning (rl_outcomes, rl_q_values, etc.)
            - Team composition stats
            - Cross-vertical patterns
            - Interactive CLI session persistence

        ./.victor/project.db - Project database (project-specific data)
            - Graph nodes/edges
            - Conversations and messages
            - Project sessions
            - Entity memory
            - Mode learning
            - Change tracking

    Migration Strategy:
        1. Backup existing databases with .bak.<timestamp> suffix
        2. Migrate data to canonical databases
        3. Validate integrity
        4. Update metadata
    """

    def __init__(self, victor_dir: Optional[Path] = None):
        """Initialize consolidator.

        Args:
            victor_dir: Path to ~/.victor directory
        """
        if victor_dir is None:
            victor_dir = Path.home() / ".victor"
        self.victor_dir = victor_dir
        self._consolidated_flag = victor_dir / ".consolidated_v6"

    def needs_consolidation(self) -> bool:
        """Check if database consolidation is needed.

        Returns:
            True if consolidation has not been completed
        """
        # Check if consolidation flag exists
        if self._consolidated_flag.exists():
            return False

        # Check if there are any legacy databases to migrate
        legacy_dbs = [
            self.victor_dir / "conversation.db",
            self.victor_dir / "profile_learning.db",
            self.victor_dir / "project.db",  # Old mixed database
        ]

        for db_path in legacy_dbs:
            if db_path.exists() and db_path.stat().st_size > 0:
                return True

        return False

    def consolidate(self) -> None:
        """Consolidate databases to canonical two-database structure.

        Raises:
            Exception: If consolidation fails
        """
        if not self.needs_consolidation():
            logger.debug("Database consolidation already completed")
            return

        logger.info("Starting database consolidation...")

        try:
            # Step 1: Backup existing databases
            self._backup_legacy_databases()

            # Step 2: Migrate global data to ~/.victor/victor.db
            self._migrate_global_database()

            # Step 3: Migrate project data to ./.victor/project.db
            self._migrate_project_databases()

            # Step 4: Create consolidation flag
            self._consolidated_flag.write_text(
                f"consolidated_at={datetime.now().isoformat()}\n" f"schema_version=6\n"
            )

            logger.info("Database consolidation complete")

        except Exception as e:
            logger.error(f"Database consolidation failed: {e}")
            raise

    def _backup_legacy_databases(self) -> None:
        """Backup legacy databases with .bak.<timestamp> suffix."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        legacy_dbs = [
            self.victor_dir / "conversation.db",
            self.victor_dir / "profile_learning.db",
            self.victor_dir / "project.db",
            self.victor_dir / "graph" / "graph.db",
        ]

        for db_path in legacy_dbs:
            if db_path.exists() and db_path.stat().st_size > 0:
                backup_path = Path(f"{db_path}.bak.{timestamp}")
                shutil.copy2(db_path, backup_path)
                logger.info(f"Backed up {db_path.name} to {backup_path.name}")

    def _migrate_global_database(self) -> None:
        """Migrate global user data to ~/.victor/victor.db."""
        target_path = self.victor_dir / "victor.db"

        # Source databases with global data
        sources = [
            self.victor_dir / "conversation.db",
            self.victor_dir / "profile_learning.db",
            self.victor_dir / "project.db",  # Old mixed database
        ]

        for source_path in sources:
            if source_path.exists() and source_path.stat().st_size > 0:
                self._migrate_global_tables(source_path, target_path)

    def _migrate_global_tables(
        self,
        source_path: Path,
        target_path: Path,
    ) -> None:
        """Migrate global tables from source to target database.

        Global tables (user-wide data):
        - rl_* (reinforcement learning)
        - agent_* (agent learning, teams)
        - ui_* (UI sessions, preferences)
        - interaction_history (profile learning)
        - profile_metrics (profile learning)
        """
        # Global tables to migrate
        global_tables = {
            # RL tables
            "rl_learner",
            "rl_outcome",
            "rl_metric",
            "rl_q_value",
            "rl_transition",
            "rl_param",
            "rl_task_stat",
            "rl_mode_q",
            "rl_mode_history",
            "rl_mode_task",
            "rl_model_q",
            "rl_model_task",
            "rl_model_state",
            "rl_tool_q",
            "rl_tool_task",
            "rl_tool_outcome",
            "rl_cache_q",
            "rl_cache_tool",
            "rl_cache_history",
            "rl_grounding_param",
            "rl_grounding_stat",
            "rl_grounding_history",
            "rl_semantic_stat",
            "rl_patience_stat",
            "rl_prompt_stat",
            "rl_quality_weight",
            "rl_quality_history",
            "rl_provider_stat",
            "rl_context_pruning",
            "rl_pattern",
            "rl_pattern_use",
            # Agent tables
            "agent_team_config",
            "agent_team_run",
            "agent_workflow_run",
            "agent_workflow_q",
            "agent_prompt_style",
            "agent_prompt_element",
            "agent_prompt_history",
            "agent_prompt_candidate",
            "agent_prompt_pareto_instance",
            "agent_curriculum_stage",
            "agent_curriculum_metric",
            "agent_curriculum_history",
            "agent_policy_snapshot",
            # UI tables
            "ui_session",
            "ui_failed_call",
            # Profile learning
            "interaction_history",
            "profile_metrics",
        }

        self._migrate_tables(source_path, target_path, global_tables)

    def _migrate_project_databases(self) -> None:
        """Migrate project data to ./.victor/project.db."""
        # For each project directory with a legacy database
        project_dir = Path.cwd() / ".victor"

        # Check for standalone graph.db in project
        graph_db = project_dir / "graph.db"
        if graph_db.exists() and graph_db.stat().st_size > 0:
            target_path = project_dir / "project.db"
            self._migrate_project_tables(graph_db, target_path)

    def _migrate_project_tables(
        self,
        source_path: Path,
        target_path: Path,
    ) -> None:
        """Migrate project tables from source to target database.

        Project tables (project-specific data):
        - graph_* (graph nodes, edges, file mtime)
        - messages, sessions (conversations)
        - context_sizes, context_summaries
        - entities (entity memory)
        - change_groups, file_changes (change tracking)
        """
        # Project tables to migrate
        project_tables = {
            # Graph tables
            "graph_node",
            "graph_edge",
            "graph_file_mtime",
            "graph_module_metric",
            "graph_module_metric_history",
            "graph_requirement",
            "graph_subgraph",
            "graph_subgraph_node",
            # Conversation tables
            "messages",
            "sessions",
            "context_sizes",
            "context_summaries",
            # Entity memory
            "entities",
            # Changes
            "change_groups",
            "file_changes",
        }

        self._migrate_tables(source_path, target_path, project_tables)

    def _migrate_tables(
        self,
        source_path: Path,
        target_path: Path,
        table_names: set,
    ) -> None:
        """Migrate tables from source to target database.

        Args:
            source_path: Path to source database
            target_path: Path to target database
            table_names: Set of table names to migrate
        """
        import sqlite3

        # Connect to both databases
        source_conn = sqlite3.connect(str(source_path))
        target_conn = sqlite3.connect(str(target_path))

        try:
            # Attach source database
            target_conn.execute("ATTACH DATABASE ? AS source_db", (str(source_path),))

            # Get tables in source that we want to migrate
            cursor = target_conn.execute("""
                SELECT name FROM source_db.sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            source_tables = {row[0] for row in cursor.fetchall()}

            # Migrate matching tables
            migrated = 0
            for table in table_names & source_tables:
                try:
                    # Check if table exists in target
                    target_cursor = target_conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    )
                    if target_cursor.fetchone():
                        # Table exists, copy data if target is empty
                        source_count = target_conn.execute(
                            f"SELECT COUNT(*) FROM source_db.{table}"
                        ).fetchone()[0]
                        target_count = target_conn.execute(
                            f"SELECT COUNT(*) FROM {table}"
                        ).fetchone()[0]

                        if source_count > 0 and target_count == 0:
                            target_conn.execute(f"""
                                INSERT INTO {table}
                                SELECT * FROM source_db.{table}
                            """)
                            logger.debug(f"Copied {source_count} rows from {table}")
                            migrated += 1
                    else:
                        # Create table and copy data
                        schema_row = target_conn.execute(
                            """
                            SELECT sql FROM source_db.sqlite_master
                            WHERE type='table' AND name=?
                        """,
                            (table,),
                        ).fetchone()

                        if schema_row and schema_row[0]:
                            target_conn.execute(schema_row[0])
                            target_conn.execute(f"""
                                INSERT INTO {table}
                                SELECT * FROM source_db.{table}
                            """)

                            # Copy indexes
                            idx_cursor = target_conn.execute(
                                """
                                SELECT sql FROM source_db.sqlite_master
                                WHERE type='index' AND tbl_name=? AND sql IS NOT NULL
                            """,
                                (table,),
                            )
                            for idx_row in idx_cursor:
                                try:
                                    target_conn.execute(idx_row[0])
                                except sqlite3.OperationalError:
                                    pass  # Index may already exist

                            migrated += 1
                            logger.debug(f"Migrated table: {table}")

                except Exception as e:
                    logger.warning(f"Failed to migrate {table}: {e}")

            target_conn.commit()

            if migrated > 0:
                logger.info(f"Migrated {migrated} tables from {source_path.name}")

        finally:
            target_conn.execute("DETACH DATABASE source_db")
            source_conn.close()
            target_conn.close()


def _find_existing_project_dir(start: Path) -> Optional[Path]:
    """Walk up from ``start`` to the nearest ancestor holding ``.victor/project.db``.

    A project owns a single ``.victor/project.db`` at its root. Resolving a
    subdirectory (e.g. ``src/network``) must therefore reuse the repository's
    project database rather than fabricate a fresh empty one under the subdir.

    Only an *existing* ``project.db`` is matched, so a brand-new project still
    initializes in place. The search is bounded at the git repository root: it
    never ascends past a directory that contains ``.git`` without first finding
    a project database there. It is also bounded at the system temp root: the
    temp dir is world-writable, so a stray ``/tmp/.victor`` — planted by
    another user, or left behind by a past process — must never own every
    resolution under it (pytest tmp dirs have no ``.git`` bound and would all
    silently share that one polluted database).

    Args:
        start: Directory to begin the upward search from (already resolved).

    Returns:
        The ``.victor`` directory of the owning project, or ``None`` if no
        existing project database is found within the bound.
    """
    temp_root = Path(tempfile.gettempdir()).resolve()
    for ancestor in (start, *start.parents):
        # Temp-root boundary before the adoption check: a ``.victor`` sitting
        # directly in the shared temp dir is never a legitimate ancestor
        # project. (start == temp root still resolves there via the caller's
        # in-place fallback, so direct use of the temp dir is unaffected.)
        if ancestor == temp_root:
            break
        victor_dir = ancestor / ".victor"
        if (victor_dir / "project.db").exists():
            return victor_dir
        # Stop at the repository boundary: a subdir of a not-yet-indexed repo
        # should initialize at the repo root via the caller, not somewhere above.
        if (ancestor / ".git").exists():
            break
    return None


def _normalize_project_database_paths(
    project_path: Optional[Path],
) -> tuple[Path, Path, Path]:
    """Normalize project DB input to project root, state dir, and db file."""
    if project_path is None:
        resolved = Path.cwd().resolve()
    else:
        resolved = Path(project_path).resolve()

    if resolved.suffix == ".db":
        project_dir = resolved.parent
        db_path = resolved
    elif resolved.name == ".victor":
        project_dir = resolved
        db_path = project_dir / "project.db"
    else:
        # Prefer an existing project DB in an ancestor so subdirectory lookups
        # resolve to the repository's single project.db instead of creating an
        # empty DB (and stray ``.victor/`` dir) under the subdirectory.
        existing = _find_existing_project_dir(resolved)
        project_dir = existing if existing is not None else resolved / ".victor"
        db_path = project_dir / "project.db"

    project_root = project_dir.parent
    return project_root, project_dir, db_path


class _DatabaseManagerBase:
    """Base class for database managers with shared connection logic.

    Provides common functionality for managing SQLite database connections:
    - Thread-local connection storage
    - Raw connection retrieval with lazy initialization
    - Connection cleanup

    This base class is inherited by both DatabaseManager and ProjectDatabaseManager
    to avoid code duplication.
    """

    def __init__(self):
        """Initialize thread-local storage for database connection."""
        import threading

        self._local = threading.local()

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Apply manager-specific settings to a newly opened connection."""
        return None

    def _get_raw_connection(self) -> sqlite3.Connection:
        """Get raw SQLite connection (thread-local).

        Creates a new connection if one doesn't exist for the current thread.
        All connections use Row factory for convenient column access.

        Returns:
            Thread-local SQLite connection with Row factory

        ENHANCEMENT: Added retry logic for concurrent access during init.
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            import time

            max_retries = 3
            last_error = None

            for attempt in range(max_retries):
                try:
                    conn = sqlite3.connect(
                        str(self.db_path),
                        check_same_thread=False,
                        timeout=30.0,  # 30 second timeout for SQLite locks
                    )
                    conn.row_factory = sqlite3.Row
                    self._configure_connection(conn)
                    self._local.conn = conn
                    break  # Success
                except sqlite3.OperationalError as e:
                    last_error = e
                    if "locked" in str(e).lower() and attempt < max_retries - 1:
                        # Brief wait before retry (exponential backoff)
                        wait_time = 0.1 * (2**attempt)
                        logger.debug(
                            "Database locked (attempt %d/%d), retrying in %.2fs",
                            attempt + 1,
                            max_retries,
                            wait_time,
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        # Re-raise if not a lock error or out of retries
                        raise

            # Should not reach here, but handle it
            if not hasattr(self._local, "conn") or self._local.conn is None:
                raise sqlite3.OperationalError(
                    f"Failed to open database after {max_retries} attempts: {last_error}"
                )

        return self._local.conn

    def close(self) -> None:
        """Close the database connection.

        Closes the thread-local connection if it exists and clears the reference.
        Safe to call multiple times.
        """
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def maintain(self) -> Dict[str, Any]:
        """Checkpoint the WAL and reclaim free pages.

        Long-lived databases (especially the project graph DB) accumulate a large WAL and
        many free pages after deletes/re-indexing; with ``auto_vacuum`` off the file never
        shrinks. This truncates the WAL and runs ``VACUUM`` to reclaim space (which also
        applies any ``auto_vacuum`` mode going forward). Idempotent and safe to rerun.

        Returns a dict with ``path``, ``size_before``, and ``size_after`` (bytes).
        """
        before = self.db_path.stat().st_size if self.db_path.exists() else 0
        conn = self._get_raw_connection()
        # Order matters in WAL mode: VACUUM rebuilds the compacted database into the WAL, and
        # the main file is only rewritten/shrunk by a subsequent checkpoint. So VACUUM first,
        # then wal_checkpoint(TRUNCATE) to flush the compacted pages back and truncate the WAL.
        conn.execute("VACUUM")  # must run outside an explicit transaction
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            logger.debug("wal_checkpoint(TRUNCATE) failed", exc_info=True)
        conn.commit()
        after = self.db_path.stat().st_size if self.db_path.exists() else 0
        logger.info(
            "Database maintained: %s (%.1f MB -> %.1f MB)",
            self.db_path,
            before / 1024 / 1024,
            after / 1024 / 1024,
        )
        return {"path": str(self.db_path), "size_before": before, "size_after": after}


class DatabaseManager(_DatabaseManagerBase):
    """Unified database manager for Victor.

    Provides a single SQLite database for all components, with:
    - Thread-safe connection pooling
    - Automatic migration from legacy databases
    - Table namespace management
    - Connection health monitoring

    Attributes:
        db_path: Path to the unified database file
        _connection: Thread-local connection storage
        _lock: Threading lock for migrations
    """

    # Singleton instance
    _instance: Optional["DatabaseManager"] = None
    _init_lock = threading.Lock()

    # Legacy database paths relative to ~/.victor/
    LEGACY_DATABASES = {
        "graph": "graph/graph.db",
        "team_learning": "team_learning.db",
        "sessions": "sessions.db",
        "signatures": "signatures.db",
        "rl_learning": "rl_learning.db",  # May be empty
    }

    def __new__(cls, db_path: Optional[Path] = None) -> "DatabaseManager":
        """Get or create singleton instance."""
        with cls._init_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database manager.

        Args:
            db_path: Path to database file. Defaults to ~/.victor/victor.db
        """
        if self._initialized:
            return

        # Initialize base class (sets up self._local)
        super().__init__()

        self._victor_dir = Path.home() / ".victor"
        self._victor_dir.mkdir(parents=True, exist_ok=True)

        if db_path is None:
            db_path = self._victor_dir / "victor.db"

        self.db_path = db_path
        self._migration_lock = threading.Lock()
        self._migrated = False

        # Ensure database exists and run migrations
        self._ensure_database()

        self._initialized = True
        logger.info(f"DatabaseManager initialized: {self.db_path}")

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Apply default SQLite settings for the global database."""
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")

    def _ensure_database(self) -> None:
        """Ensure database exists and is up to date."""
        from victor.core.schema import Tables, Schema

        # Run database consolidation if needed
        consolidator = DatabaseConsolidator(self._victor_dir)
        if consolidator.needs_consolidation():
            logger.info("Running database consolidation...")
            consolidator.consolidate()

        # Create database if needed
        conn = self._get_raw_connection()

        # Create metadata table using schema constant
        conn.execute(Schema.SYS_METADATA)
        conn.commit()

        # Run schema version migrations if needed
        self._run_schema_version_migrations(conn)

        # Run legacy migrations if needed
        self._run_migrations(conn)

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection.

        Returns:
            Thread-local SQLite connection with Row factory
        """
        return self._get_raw_connection()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database transactions.

        Yields:
            Database connection with automatic commit/rollback

        Example:
            with db.transaction() as conn:
                conn.execute("INSERT INTO ...")
                conn.execute("UPDATE ...")
            # Auto-committed on success, rolled back on exception
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> sqlite3.Cursor:
        """Execute SQL with auto-commit.

        Args:
            sql: SQL statement
            params: Optional parameters

        Returns:
            Cursor with results
        """
        conn = self.get_connection()
        cursor = conn.execute(sql, params or ())
        conn.commit()
        return cursor

    def executemany(
        self,
        sql: str,
        params_list: List[Tuple[Any, ...]],
    ) -> sqlite3.Cursor:
        """Execute SQL for multiple parameter sets.

        Args:
            sql: SQL statement
            params_list: List of parameter tuples

        Returns:
            Cursor
        """
        conn = self.get_connection()
        cursor = conn.executemany(sql, params_list)
        conn.commit()
        return cursor

    def query(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> List[sqlite3.Row]:
        """Execute query and return all rows.

        Args:
            sql: SQL query
            params: Optional parameters

        Returns:
            List of Row objects
        """
        conn = self.get_connection()
        cursor = conn.execute(sql, params or ())
        return cursor.fetchall()

    def query_one(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> Optional[sqlite3.Row]:
        """Execute query and return first row.

        Args:
            sql: SQL query
            params: Optional parameters

        Returns:
            Row object or None
        """
        conn = self.get_connection()
        cursor = conn.execute(sql, params or ())
        return cursor.fetchone()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists.

        Args:
            table_name: Name of table

        Returns:
            True if table exists
        """
        row = self.query_one(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return row is not None

    def get_tables(self) -> List[str]:
        """Get list of all tables.

        Returns:
            List of table names
        """
        rows = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in rows]

    def _run_schema_version_migrations(self, conn: sqlite3.Connection) -> None:
        """Run schema version migrations.

        Args:
            conn: Database connection
        """
        from victor.core.schema import Tables, CURRENT_SCHEMA_VERSION, get_migration_sql

        # Get current schema version
        cursor = conn.execute(
            f"SELECT value FROM {Tables.SYS_METADATA} WHERE key = 'schema_version'"
        )
        row = cursor.fetchone()

        if row:
            current_version = int(row[0])
        else:
            # Check if this is a legacy database (has tables but no schema_version tracking)
            # vs a truly empty database (first ever access)
            existing_tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            non_meta_tables = existing_tables - {Tables.SYS_METADATA, "sqlite_sequence"}
            # Only treat as a pre-v2 legacy Victor DB (needing the table-rename
            # migrations) when the pre-migration sentinel table `_db_metadata` is
            # present. A modern DB that merely accumulated tables lazily (via the
            # Schema.* constants) has no version row yet but is already at the
            # current schema — replaying the rename/ALTER chain against it only
            # produces "no such table"/"duplicate column" noise. Mirrors the
            # project-DB guard in ProjectDatabaseManager._run_schema_version_migrations.
            if non_meta_tables and "_db_metadata" in existing_tables:
                # Legacy database — run all migrations so missing columns get added
                current_version = 0
                logger.info(
                    f"Legacy database detected (tables: {len(non_meta_tables)}), "
                    f"running migrations from 0 to {CURRENT_SCHEMA_VERSION}"
                )
            else:
                # Truly empty DB, or a modern DB pre-populated by lazy table
                # creation — mark at current version without replaying migrations.
                conn.execute(
                    f"""
                    INSERT INTO {Tables.SYS_METADATA} (key, value, updated_at)
                    VALUES ('schema_version', ?, datetime('now'))
                    """,
                    (str(CURRENT_SCHEMA_VERSION),),
                )
                conn.commit()
                logger.info(f"Database initialized at schema version {CURRENT_SCHEMA_VERSION}")
                return

        # Run migrations if needed
        if current_version < CURRENT_SCHEMA_VERSION:
            logger.info(
                f"Migrating database from version {current_version} to {CURRENT_SCHEMA_VERSION}"
            )

            migration_sqls = get_migration_sql(current_version, CURRENT_SCHEMA_VERSION)

            for sql in migration_sqls:
                _apply_migration_sql(conn, sql)

            # Record the new schema version. Use an upsert: the legacy branch
            # reaches here with current_version=0 but no schema_version row, so a
            # plain UPDATE would match nothing and the version would never persist
            # — forcing a full migration replay on every startup.
            conn.execute(
                f"""
                INSERT INTO {Tables.SYS_METADATA} (key, value, updated_at)
                VALUES ('schema_version', ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (str(CURRENT_SCHEMA_VERSION),),
            )
            conn.commit()
            logger.info(f"Database migrated to schema version {CURRENT_SCHEMA_VERSION}")

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Run database migrations from legacy databases."""
        from victor.core.schema import Tables

        with self._migration_lock:
            if self._migrated:
                return

            # Check if already migrated
            cursor = conn.execute(
                f"SELECT value FROM {Tables.SYS_METADATA} WHERE key = 'migrated_at'"
            )
            row = cursor.fetchone()
            if row:
                logger.debug(f"Database already migrated at {row[0]}")
                self._migrated = True
                return

            # Migrate from legacy databases
            migrated_count = 0
            for name, rel_path in self.LEGACY_DATABASES.items():
                legacy_path = self._victor_dir / rel_path
                if legacy_path.exists() and legacy_path.stat().st_size > 0:
                    try:
                        migrated = self._migrate_from_legacy(conn, name, legacy_path)
                        migrated_count += migrated
                        logger.info(f"Migrated {migrated} tables from {name}")
                    except Exception as e:
                        logger.warning(f"Failed to migrate {name}: {e}")

            # Record migration timestamp
            now = datetime.now().isoformat()
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {Tables.SYS_METADATA} (key, value, updated_at)
                VALUES ('migrated_at', ?, ?)
                """,
                (now, now),
            )
            conn.commit()

            self._migrated = True
            if migrated_count > 0:
                logger.info(f"Database migration complete: {migrated_count} tables")

    def _migrate_from_legacy(
        self,
        conn: sqlite3.Connection,
        source_name: str,
        legacy_path: Path,
    ) -> int:
        """Migrate tables from a legacy database.

        Args:
            conn: Target database connection
            source_name: Name of source database (for logging)
            legacy_path: Path to legacy database

        Returns:
            Number of tables migrated
        """
        # Attach legacy database
        conn.execute(f"ATTACH DATABASE ? AS legacy_{source_name}", (str(legacy_path),))

        try:
            # Get tables from legacy database (excluding sqlite internal tables)
            cursor = conn.execute(f"""
                SELECT name FROM legacy_{source_name}.sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]

            migrated = 0
            for table in tables:
                # Skip if table already exists in target
                if self.table_exists(table):
                    # Check if source has data and target is empty
                    source_count = conn.execute(
                        f"SELECT COUNT(*) FROM legacy_{source_name}.{table}"
                    ).fetchone()[0]
                    target_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

                    if source_count > 0 and target_count == 0:
                        # Copy data from source to existing target
                        conn.execute(f"""
                            INSERT INTO {table}
                            SELECT * FROM legacy_{source_name}.{table}
                        """)
                        logger.debug(f"Copied {source_count} rows from legacy {table}")
                        migrated += 1
                    continue

                # Get table schema
                schema_row = conn.execute(
                    f"""
                    SELECT sql FROM legacy_{source_name}.sqlite_master
                    WHERE type='table' AND name=?
                """,
                    (table,),
                ).fetchone()

                if schema_row and schema_row[0]:
                    # Create table in target
                    try:
                        conn.execute(schema_row[0])
                    except sqlite3.OperationalError as e:
                        if "already exists" not in str(e):
                            raise

                    # Copy data
                    conn.execute(f"""
                        INSERT INTO {table}
                        SELECT * FROM legacy_{source_name}.{table}
                    """)

                    # Copy indexes
                    idx_cursor = conn.execute(
                        f"""
                        SELECT sql FROM legacy_{source_name}.sqlite_master
                        WHERE type='index' AND tbl_name=? AND sql IS NOT NULL
                    """,
                        (table,),
                    )
                    for idx_row in idx_cursor:
                        try:
                            conn.execute(idx_row[0])
                        except sqlite3.OperationalError:
                            pass  # Index may already exist

                    migrated += 1
                    logger.debug(f"Migrated table: {table}")

            conn.commit()
            return migrated

        finally:
            conn.execute(f"DETACH DATABASE legacy_{source_name}")

    def backup(self, backup_path: Optional[Path] = None) -> Path:
        """Create a backup of the database.

        Args:
            backup_path: Optional custom backup path

        Returns:
            Path to backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self._victor_dir / "backups" / f"victor_{timestamp}.db"

        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Use SQLite backup API
        conn = self.get_connection()
        backup_conn = sqlite3.connect(str(backup_path))
        try:
            conn.backup(backup_conn)
            logger.info(f"Database backed up to: {backup_path}")
        finally:
            backup_conn.close()

        return backup_path

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dict with table counts and sizes
        """
        stats: Dict[str, Any] = {
            "path": str(self.db_path),
            "size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "tables": {},
        }

        for table in self.get_tables():
            if table.startswith("_"):
                continue
            count = self.query_one(f"SELECT COUNT(*) FROM {table}")
            stats["tables"][table] = count[0] if count else 0

        return stats

    # Table groups for bulk operations
    TABLE_GROUPS = {
        "rl": ["rl_outcome", "rl_metric"],
        "agent": ["agent_prompt_history", "agent_workflow_run", "agent_team_run"],
    }

    # Column name mapping: table → date column used for time-based pruning
    # Tables use different names: created_at, timestamp, executed_at
    _DATE_COLUMNS = {
        "rl_outcome": "created_at",
        "rl_metric": "created_at",
        "agent_prompt_history": "timestamp",
        "agent_workflow_run": "executed_at",
        "agent_team_run": "created_at",
        "rl_cache_history": "created_at",
        "rl_grounding_history": "created_at",
        "rl_tool_outcome": "created_at",
    }

    def _get_date_column(self, table: str) -> str:
        """Get the date column name for a table, with auto-detection fallback."""
        if table in self._DATE_COLUMNS:
            return self._DATE_COLUMNS[table]
        # Auto-detect
        conn = self.get_connection()
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info([{table}])").fetchall()}
        for candidate in ("created_at", "timestamp", "executed_at", "updated_at"):
            if candidate in cols:
                return candidate
        raise ValueError(f"No date column found in '{table}' (columns: {cols})")

    def get_tables_for_group(self, group: str) -> list:
        """Get table names for a logical group. 'all' returns all prunable tables."""
        if group == "all":
            tables = []
            for g in self.TABLE_GROUPS.values():
                tables.extend(g)
            return tables
        return list(self.TABLE_GROUPS.get(group, []))

    def prune_table(
        self,
        table: str,
        *,
        older_than_days: int | None = None,
        keep_last: int | None = None,
    ) -> int:
        """Delete old rows from a table.

        Args:
            table: Table name (must be in a known group for safety)
            older_than_days: Delete rows older than N days (uses created_at)
            keep_last: Keep only the most recent N rows

        Returns:
            Number of rows deleted
        """
        # Safety: only allow pruning known tables
        all_prunable = self.get_tables_for_group("all")
        if table not in all_prunable:
            raise ValueError(f"Table '{table}' not in prunable tables: {all_prunable}")

        conn = self.get_connection()

        if older_than_days is not None:
            from datetime import datetime, timedelta, timezone

            date_col = self._get_date_column(table)
            cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
            cursor = conn.execute(f"DELETE FROM [{table}] WHERE [{date_col}] < ?", (cutoff,))
            conn.commit()
            deleted = cursor.rowcount
            logger.info(
                "Pruned %d rows from %s (older than %d days)",
                deleted,
                table,
                older_than_days,
            )
            return deleted

        if keep_last is not None:
            cursor = conn.execute(
                f"DELETE FROM [{table}] WHERE id NOT IN "
                f"(SELECT id FROM [{table}] ORDER BY id DESC LIMIT ?)",
                (keep_last,),
            )
            conn.commit()
            deleted = cursor.rowcount
            logger.info("Pruned %d rows from %s (kept last %d)", deleted, table, keep_last)
            return deleted

        return 0

    def archive_table(self, table: str, before_date: str, output_path: "Path") -> int:
        """Export rows before a date to gzip JSONL, then delete them.

        Args:
            table: Table name
            before_date: ISO date string (YYYY-MM-DD)
            output_path: Path for the .jsonl.gz archive file

        Returns:
            Number of rows archived
        """
        import gzip
        import json

        all_prunable = self.get_tables_for_group("all")
        if table not in all_prunable:
            raise ValueError(f"Table '{table}' not prunable")

        conn = self.get_connection()
        date_col = self._get_date_column(table)
        cursor = conn.execute(f"SELECT * FROM [{table}] WHERE [{date_col}] < ?", (before_date,))
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        if not rows:
            return 0

        # Write to gzip JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            for row in rows:
                record = dict(zip(columns, row))
                f.write(json.dumps(record, default=str) + "\n")

        # Delete archived rows
        conn.execute(f"DELETE FROM [{table}] WHERE [{date_col}] < ?", (before_date,))
        conn.commit()

        logger.info("Archived %d rows from %s to %s", len(rows), table, output_path)
        return len(rows)

    def get_table_stats(self) -> list:
        """Get per-table statistics for all tables with data.

        Returns:
            List of dicts with table, rows, min_date, max_date, est_size_kb
        """
        conn = self.get_connection()
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]

        stats = []
        for table in sorted(tables):
            try:
                count = conn.execute(f"SELECT count(*) FROM [{table}]").fetchone()[0]
                if count == 0:
                    continue

                # Try to get date range (check multiple column names)
                min_date = max_date = None
                try:
                    date_col = self._get_date_column(table)
                    row = conn.execute(
                        f"SELECT MIN([{date_col}]), MAX([{date_col}]) FROM [{table}]"
                    ).fetchone()
                    min_date, max_date = row[0], row[1]
                except Exception:
                    pass

                # Estimate size from sample
                sample = conn.execute(f"SELECT * FROM [{table}] LIMIT 5").fetchall()
                avg_row = sum(len(str(r)) for r in sample) / max(len(sample), 1)
                est_kb = int(count * avg_row / 1024)

                stats.append(
                    {
                        "table": table,
                        "rows": count,
                        "min_date": str(min_date)[:10] if min_date else None,
                        "max_date": str(max_date)[:10] if max_date else None,
                        "est_size_kb": est_kb,
                    }
                )
            except Exception:
                continue

        return stats

    def vacuum(self) -> None:
        """Vacuum the database to reclaim space."""
        conn = self.get_connection()
        conn.execute("VACUUM")
        logger.info("Database vacuumed")

    # =========================================================================
    # Async API (wraps sync methods via thread pool)
    # =========================================================================

    def _ensure_async_state(self) -> None:
        """Initialize async-related state if not already done."""
        if not hasattr(self, "_write_queue"):
            self._write_queue: queue.Queue[Tuple[str, Optional[Tuple[Any, ...]]]] = queue.Queue()
            self._write_lock = threading.Lock()
            self._batch_size = 100
            self._flush_interval = 5.0
            self._flush_task: Optional[asyncio.Task] = None
            self._is_flushing = False

    async def execute_async(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> sqlite3.Cursor:
        """Execute SQL asynchronously.

        Runs the sync execute in a thread pool to avoid blocking.

        Args:
            sql: SQL statement
            params: Optional parameters

        Returns:
            Cursor with results
        """
        return await asyncio.to_thread(self.execute, sql, params)

    async def executemany_async(
        self,
        sql: str,
        params_list: List[Tuple[Any, ...]],
    ) -> sqlite3.Cursor:
        """Execute SQL for multiple parameter sets asynchronously.

        Args:
            sql: SQL statement
            params_list: List of parameter tuples

        Returns:
            Cursor
        """
        return await asyncio.to_thread(self.executemany, sql, params_list)

    async def query_async(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> List[sqlite3.Row]:
        """Execute query asynchronously.

        Args:
            sql: SQL query
            params: Optional parameters

        Returns:
            List of Row objects
        """
        return await asyncio.to_thread(self.query, sql, params)

    async def query_one_async(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> Optional[sqlite3.Row]:
        """Execute query and return first row asynchronously.

        Args:
            sql: SQL query
            params: Optional parameters

        Returns:
            Row object or None
        """
        return await asyncio.to_thread(self.query_one, sql, params)

    # =========================================================================
    # Write Batching
    # =========================================================================

    def queue_write(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        """Queue a write operation for batched execution.

        Writes are batched and executed together for better performance.
        Call flush_writes() to execute all queued writes.

        Args:
            sql: SQL statement (INSERT, UPDATE, DELETE)
            params: Optional parameters
        """
        self._ensure_async_state()
        self._write_queue.put((sql, params))

        # Auto-flush if queue is large
        if self._write_queue.qsize() >= self._batch_size:
            self.flush_writes_sync()

    def flush_writes_sync(self) -> int:
        """Flush all queued writes synchronously.

        Returns:
            Number of writes executed
        """
        self._ensure_async_state()

        with self._write_lock:
            if self._is_flushing:
                return 0
            self._is_flushing = True

        try:
            writes: List[Tuple[str, Optional[Tuple[Any, ...]]]] = []

            # Drain the queue
            while not self._write_queue.empty():
                try:
                    writes.append(self._write_queue.get_nowait())
                except Exception:
                    break

            if not writes:
                return 0

            # Execute in a transaction
            conn = self.get_connection()
            try:
                for sql, params in writes:
                    conn.execute(sql, params or ())
                conn.commit()
                logger.debug(f"Flushed {len(writes)} batched writes")
                return len(writes)
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to flush writes: {e}")
                # Re-queue failed writes
                for write in writes:
                    self._write_queue.put(write)
                raise

        finally:
            self._is_flushing = False

    async def flush_writes_async(self) -> int:
        """Flush all queued writes asynchronously.

        Returns:
            Number of writes executed
        """
        return await asyncio.to_thread(self.flush_writes_sync)

    def get_write_queue_size(self) -> int:
        """Get number of pending writes.

        Returns:
            Queue size
        """
        self._ensure_async_state()
        return self._write_queue.qsize()

    async def start_auto_flush(self, interval_seconds: Optional[float] = None) -> None:
        """Start background task that periodically flushes writes.

        Args:
            interval_seconds: Flush interval (default 5.0)
        """
        self._ensure_async_state()

        if interval_seconds is not None:
            self._flush_interval = interval_seconds

        async def _flush_loop() -> None:
            while True:
                await asyncio.sleep(self._flush_interval)
                try:
                    await self.flush_writes_async()
                except Exception as e:
                    logger.warning(f"Auto-flush error: {e}")

        self._flush_task = asyncio.create_task(_flush_loop())

    async def stop_auto_flush(self) -> None:
        """Stop background flush task and flush remaining writes."""
        if hasattr(self, "_flush_task") and self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Final flush
        await self.flush_writes_async()


class ProjectDatabaseManager(_DatabaseManagerBase):
    """Project-level database manager for repo-specific data.

    Manages project-scoped data in .victor/project.db:
    - Graph data (symbols, references, definitions)
    - Conversation memory
    - Entity memory
    - Project-specific mode learning

    Unlike the global DatabaseManager, this is NOT a singleton since
    different projects have different databases.

    Attributes:
        db_path: Path to the project database file
        project_dir: Path to the project's .victor directory
    """

    # Legacy project databases to migrate
    LEGACY_PROJECT_DATABASES = {
        "conversation": "conversation.db",
        "entities": "entities.db",
        "mode_learning": "mode_learning.db",
        "profile_learning": "profile_learning.db",
        "changes": "changes/changes.db",
        "graph": "graph/graph.db",
    }

    # Tables that should stay in project database (not global)
    PROJECT_TABLES = {
        # Graph tables
        "graph_node",
        "graph_edge",
        "graph_file_mtime",
        "graph_node_fts",
        "graph_module_metric",
        "graph_module_metric_history",
        # Conversation tables
        "messages",
        "sessions",
        "context_sizes",
        "context_summaries",
        # Mode learning tables
        "rl_mode_q",
        "rl_mode_task",
        "rl_mode_history",
        # Profile learning tables
        "interaction_history",
        "profile_metrics",
        # Changes tables
        "change_groups",
        "file_changes",
    }
    _WAL_AUTOCHECKPOINT_PAGES = 10000

    def __init__(self, project_path: Optional[Path] = None):
        """Initialize project database manager.

        Args:
            project_path: Path to project root. If None, uses current directory.
        """
        project_root, project_dir, db_path = _normalize_project_database_paths(project_path)

        # Initialize base class (sets up self._local)
        super().__init__()

        self.project_root = project_root
        self.project_dir = project_dir
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._migration_lock = threading.Lock()
        self._migrated = False

        # Ensure database exists
        self._ensure_database()

        logger.info(f"ProjectDatabaseManager initialized: {self.db_path}")

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Apply SQLite settings for the project database.

        The larger auto-checkpoint threshold keeps large graph-indexing writes
        from triggering frequent WAL checkpoints on small default page budgets.
        """
        # auto_vacuum must be set before tables are created to take effect without a VACUUM;
        # on a fresh project DB this lets incremental_vacuum/maintain() reclaim deleted graph
        # rows so the file does not grow unbounded. On an existing DB it is a no-op until the
        # next VACUUM (which maintain() runs).
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA wal_autocheckpoint={self._WAL_AUTOCHECKPOINT_PAGES}")
        # Deliberately 5000ms, NOT the 30s connect() timeout: non-critical writers
        # (e.g. the edit tool's undo-log bookkeeping) no longer hard-fail on a lock,
        # so a long stall is worse than a skipped undo record. This overrides the
        # 30000ms busy handler sqlite3.connect(timeout=30.0) installs.
        conn.execute("PRAGMA busy_timeout = 5000")

    def _ensure_database(self) -> None:
        """Ensure database exists and is up to date."""
        from victor.core.schema import Schema

        conn = self._get_raw_connection()

        # Create metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _project_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)

        # Create project-level tables (graph, etc.)
        for schema_sql in Schema.get_project_schemas():
            conn.execute(schema_sql)

        # Create project-level indexes
        for index_sql in Schema.get_project_indexes():
            conn.executescript(index_sql)

        conn.commit()

        # Run schema version migrations (adds columns like ast_kind to existing tables)
        self._run_schema_version_migrations(conn)

        # Run migrations if needed
        self._run_migrations(conn)

    def _run_schema_version_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply versioned column migrations to existing project tables."""
        from victor.core.schema import CURRENT_SCHEMA_VERSION, get_migration_sql

        cursor = conn.execute("SELECT value FROM _project_metadata WHERE key = 'schema_version'")
        row = cursor.fetchone()

        if row:
            current_version = int(row[0])
        else:
            # Check if this is an existing project DB (has tables but no version tracking)
            existing_tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            non_meta = existing_tables - {"_project_metadata", "sqlite_sequence"}
            # Only treat as a legacy Victor DB (needing rename migrations) when the
            # pre-migration sentinel table _db_metadata is present.  A database that
            # has ConversationStore tables (sessions, messages, …) but no _db_metadata
            # is NOT a legacy Victor global DB — running rename migrations would destroy
            # the ConversationStore session table by renaming it to UI_SESSION.
            if non_meta and "_db_metadata" in existing_tables:
                # Genuine legacy Victor DB — run all migrations to pick up schema renames
                current_version = 0
                logger.info(
                    f"Legacy project DB detected ({len(non_meta)} tables), "
                    f"migrating to schema v{CURRENT_SCHEMA_VERSION}"
                )
            else:
                # Brand-new project DB or project DB pre-populated by ConversationStore —
                # mark at current version without running rename migrations.
                conn.execute(
                    "INSERT OR REPLACE INTO _project_metadata (key, value, updated_at) "
                    "VALUES ('schema_version', ?, datetime('now'))",
                    (str(CURRENT_SCHEMA_VERSION),),
                )
                conn.commit()
                return

        if current_version < CURRENT_SCHEMA_VERSION:
            migration_sqls = get_migration_sql(current_version, CURRENT_SCHEMA_VERSION)
            for sql in migration_sqls:
                _apply_migration_sql(conn, sql)
            conn.execute(
                "INSERT OR REPLACE INTO _project_metadata (key, value, updated_at) "
                "VALUES ('schema_version', ?, datetime('now'))",
                (str(CURRENT_SCHEMA_VERSION),),
            )
            conn.commit()
            logger.info(f"Project DB migrated to schema version {CURRENT_SCHEMA_VERSION}")

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return self._get_raw_connection()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database transactions."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> sqlite3.Cursor:
        """Execute SQL with auto-commit."""
        conn = self.get_connection()
        cursor = conn.execute(sql, params or ())
        conn.commit()
        return cursor

    def query(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> List[sqlite3.Row]:
        """Execute query and return all rows."""
        conn = self.get_connection()
        cursor = conn.execute(sql, params or ())
        return cursor.fetchall()

    def query_one(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> Optional[sqlite3.Row]:
        """Execute query and return first row."""
        conn = self.get_connection()
        cursor = conn.execute(sql, params or ())
        return cursor.fetchone()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        row = self.query_one(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return row is not None

    def get_tables(self) -> List[str]:
        """Get list of all tables."""
        rows = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in rows]

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Run database migrations from legacy project databases."""
        with self._migration_lock:
            if self._migrated:
                return

            # Check if already migrated
            cursor = conn.execute("SELECT value FROM _project_metadata WHERE key = 'migrated_at'")
            row = cursor.fetchone()
            if row:
                logger.debug(f"Project database already migrated at {row[0]}")
                self._migrated = True
                return

            # Migrate from legacy databases
            migrated_count = 0
            for name, rel_path in self.LEGACY_PROJECT_DATABASES.items():
                legacy_path = self.project_dir / rel_path
                if legacy_path.exists() and legacy_path.stat().st_size > 0:
                    try:
                        migrated = self._migrate_project_tables(conn, name, legacy_path)
                        migrated_count += migrated
                        if migrated > 0:
                            logger.info(f"Migrated {migrated} tables from {name}")
                    except Exception as e:
                        logger.warning(f"Failed to migrate {name}: {e}")

            # Record migration timestamp
            now = datetime.now().isoformat()
            conn.execute(
                """
                INSERT OR REPLACE INTO _project_metadata (key, value, updated_at)
                VALUES ('migrated_at', ?, ?)
                """,
                (now, now),
            )
            conn.commit()

            self._migrated = True
            if migrated_count > 0:
                logger.info(f"Project database migration complete: {migrated_count} tables")

    def _migrate_project_tables(
        self,
        conn: sqlite3.Connection,
        source_name: str,
        legacy_path: Path,
    ) -> int:
        """Migrate project-specific tables from a legacy database."""
        # Attach legacy database
        conn.execute(f"ATTACH DATABASE ? AS legacy_{source_name}", (str(legacy_path),))

        try:
            # Get tables from legacy database
            cursor = conn.execute(f"""
                SELECT name FROM legacy_{source_name}.sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]

            migrated = 0
            for table in tables:
                # Only migrate project-specific tables
                if table not in self.PROJECT_TABLES and not table.startswith("_"):
                    continue

                # Skip if table already exists
                if self.table_exists(table):
                    continue

                # Get table schema
                schema_row = conn.execute(
                    f"""
                    SELECT sql FROM legacy_{source_name}.sqlite_master
                    WHERE type='table' AND name=?
                """,
                    (table,),
                ).fetchone()

                if schema_row and schema_row[0]:
                    try:
                        conn.execute(schema_row[0])
                    except sqlite3.OperationalError as e:
                        if "already exists" not in str(e):
                            raise

                    # Copy data
                    conn.execute(f"""
                        INSERT INTO {table}
                        SELECT * FROM legacy_{source_name}.{table}
                    """)

                    migrated += 1
                    logger.debug(f"Migrated project table: {table}")

            conn.commit()
            return migrated

        finally:
            conn.execute(f"DETACH DATABASE legacy_{source_name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats: Dict[str, Any] = {
            "path": str(self.db_path),
            "size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "tables": {},
        }

        for table in self.get_tables():
            if table.startswith("_"):
                continue
            count = self.query_one(f"SELECT COUNT(*) FROM {table}")
            stats["tables"][table] = count[0] if count else 0

        return stats


# Module-level singleton accessors
_database: Optional[DatabaseManager] = None
_project_databases: Dict[str, ProjectDatabaseManager] = {}


def get_database(db_path: Optional[Path] = None) -> DatabaseManager:
    """Get the global database manager instance.

    Args:
        db_path: Optional custom database path (only used on first call)

    Returns:
        DatabaseManager singleton instance
    """
    global _database
    if _database is None:
        _database = DatabaseManager(db_path)
    return _database


def get_project_database(project_path: Optional[Path] = None) -> ProjectDatabaseManager:
    """Get project database manager for a specific project.

    Args:
        project_path: Path to project root. If None, uses current directory.

    Returns:
        ProjectDatabaseManager instance for the project
    """
    global _project_databases
    _project_root, _project_dir, db_path = _normalize_project_database_paths(project_path)
    project_key = str(db_path)
    if project_key not in _project_databases:
        _project_databases[project_key] = ProjectDatabaseManager(project_path)
    return _project_databases[project_key]


def resolve_project_db_root(project_path: Optional[Path] = None) -> Path:
    """Resolve the canonical project root that owns the project database.

    Walks up from ``project_path`` (or cwd) to the repository's ``.victor``
    project, so callers operating on a subdirectory get the single project-level
    database root rather than the literal subpath. Falls back to the resolved
    path itself when no existing project database is found (fresh project).

    Args:
        project_path: A path inside the project (file or directory). Defaults to cwd.

    Returns:
        The directory that holds (or will hold) ``.victor/project.db``.
    """
    project_root, _project_dir, _db_path = _normalize_project_database_paths(project_path)
    return project_root


def reset_database() -> None:
    """Reset the global database instance (for testing)."""
    global _database
    if _database is not None:
        _database.close()
        _database = None
    DatabaseManager._instance = None


def reset_project_database(project_path: Optional[Path] = None) -> None:
    """Reset project database instance (for testing)."""
    global _project_databases
    _project_root, _project_dir, db_path = _normalize_project_database_paths(project_path)
    project_key = str(db_path)
    if project_key in _project_databases:
        _project_databases[project_key].close()
        del _project_databases[project_key]


def reset_all_databases() -> None:
    """Reset all database instances (for testing)."""
    global _database, _project_databases

    if _database is not None:
        _database.close()
        _database = None
    DatabaseManager._instance = None

    for db in _project_databases.values():
        db.close()
    _project_databases.clear()


__all__ = [
    "DatabaseManager",
    "ProjectDatabaseManager",
    "get_database",
    "get_project_database",
    "resolve_project_db_root",
    "reset_database",
    "reset_project_database",
    "reset_all_databases",
]
