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

"""Dedicated project-scoped SQLite store for file-edit undo/redo history.

Undo history used to live in the shared ``project.db``. That file is written
continuously by the graph indexer (reindex-on-save); SQLite serializes writers
even under WAL, so the tiny per-edit undo write kept losing the write-lock and
failing with ``database is locked`` — silently dropping undo history.

``undo.db`` is a separate file (sibling of ``project.db`` under ``.victor/``) so
the undo writer holds its **own** write-lock and never contends with the graph
indexer. It is intentionally small, self-versioned, synchronous, and rebuildable
(history is not durability-critical — file backups cover recoverability).

Concurrency: WAL + ``busy_timeout`` + per-thread connections (inherited from
``_DatabaseManagerBase``) let multiple victor sessions/processes editing the same
project write here safely; their small writes briefly serialize but never block
on the indexer.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional

from victor.core.database import _DatabaseManagerBase, _normalize_project_database_paths

logger = logging.getLogger(__name__)

# Schema version for undo.db. Bump + add a migration block when the DDL changes.
UNDO_SCHEMA_VERSION = 1

# Canonical DDL (idempotent). `change_groups.data` holds the full JSON of a
# ChangeGroup and is the restore source of truth; `file_changes` mirrors columns
# for querying (seq gives deterministic reverse-replay order).
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS undo_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS change_groups (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    message_id TEXT,
    timestamp REAL,
    description TEXT,
    tool_name TEXT,
    undone INTEGER DEFAULT 0,
    data TEXT
);

CREATE TABLE IF NOT EXISTS file_changes (
    id TEXT PRIMARY KEY,
    group_id TEXT,
    seq INTEGER,
    change_type TEXT,
    file_path TEXT,
    timestamp REAL,
    tool_name TEXT,
    original_content TEXT,
    new_content TEXT,
    original_path TEXT,
    checksum_before TEXT,
    checksum_after TEXT,
    session_id TEXT,
    message_id TEXT,
    FOREIGN KEY (group_id) REFERENCES change_groups(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_groups_session ON change_groups(session_id);
CREATE INDEX IF NOT EXISTS idx_groups_timestamp ON change_groups(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_groups_message ON change_groups(message_id);
CREATE INDEX IF NOT EXISTS idx_changes_group ON file_changes(group_id);
CREATE INDEX IF NOT EXISTS idx_changes_path ON file_changes(file_path);
"""


class UndoDatabaseManager(_DatabaseManagerBase):
    """Manages the dedicated ``.victor/undo.db`` store.

    Reuses the trusted per-thread-connection + locked-retry machinery from
    ``_DatabaseManagerBase``; only the pragmas and schema are undo-specific.
    Not a singleton — one instance per resolved db-path (see
    :func:`get_undo_database`), mirroring ``get_project_database``.
    """

    _BUSY_TIMEOUT_MS = 5000

    def __init__(self, project_path: Optional[Path] = None):
        # Resolve the same project ``.victor`` dir as project.db (ancestor-aware),
        # then swap the filename so undo.db sits alongside project.db.
        _root, project_dir, _project_db = _normalize_project_database_paths(project_path)
        super().__init__()
        self.project_dir = project_dir
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = project_dir / "undo.db"
        self._schema_lock = threading.Lock()
        self._ensure_database()
        logger.info("UndoDatabaseManager initialized: %s", self.db_path)

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """WAL + short busy_timeout: own lock, never a long stall on contention."""
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout = {self._BUSY_TIMEOUT_MS}")

    def _ensure_database(self) -> None:
        """Create tables/indexes and stamp the schema version (idempotent)."""
        conn = self._get_raw_connection()
        with self._schema_lock:
            conn.executescript(_SCHEMA_SQL)
            conn.execute(
                "INSERT INTO undo_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO NOTHING",
                (str(UNDO_SCHEMA_VERSION),),
            )
            conn.commit()

    def ensure_schema(self) -> None:
        """Public, idempotent schema ensure (callers may re-verify)."""
        self._ensure_database()

    def get_connection(self) -> sqlite3.Connection:
        """Return the thread-local connection (mirrors ProjectDatabaseManager)."""
        return self._get_raw_connection()

    def schema_version(self) -> int:
        """Return the stored schema version (0 if unset)."""
        try:
            row = (
                self._get_raw_connection()
                .execute("SELECT value FROM undo_meta WHERE key='schema_version'")
                .fetchone()
            )
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0


# One manager per resolved undo.db path (mirrors get_project_database).
_undo_databases: Dict[str, UndoDatabaseManager] = {}
_undo_databases_lock = threading.Lock()


def get_undo_database(project_path: Optional[Path] = None) -> UndoDatabaseManager:
    """Get the undo-history database manager for a project.

    Args:
        project_path: Project root (or a path within it). None = current directory.

    Returns:
        A cached :class:`UndoDatabaseManager` for the resolved ``.victor/undo.db``.
    """
    _root, project_dir, _project_db = _normalize_project_database_paths(project_path)
    key = str(project_dir / "undo.db")
    with _undo_databases_lock:
        mgr = _undo_databases.get(key)
        if mgr is None:
            mgr = UndoDatabaseManager(project_path)
            _undo_databases[key] = mgr
        return mgr


def reset_undo_databases() -> None:
    """Close and clear all cached undo-database managers (test hook)."""
    with _undo_databases_lock:
        for mgr in _undo_databases.values():
            try:
                mgr.close()
            except Exception:  # pragma: no cover - best effort
                pass
        _undo_databases.clear()


__all__ = [
    "UndoDatabaseManager",
    "get_undo_database",
    "reset_undo_databases",
    "UNDO_SCHEMA_VERSION",
]
