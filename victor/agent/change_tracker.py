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

"""File change tracking system for undo/redo functionality.

This module provides a robust system for tracking file modifications made by
Victor, enabling undo/redo operations similar to OpenCode's /undo and /redo.
"""

import hashlib
from victor.core.json_utils import json_dumps, json_loads
import logging
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _current_correlation_id() -> Optional[str]:
    """Best-effort current trace/turn id, used as the change-group ``message_id``.

    Reads the ambient ``TraceContext`` contextvar (cheap, None if no active
    trace); groups edits made within one turn under the same id. Never raises.
    """
    try:
        from victor.runtime.trace_context import get_correlation_id

        return get_correlation_id()
    except Exception:  # pragma: no cover - defensive, tracing is optional
        return None


class ChangeType(Enum):
    """Types of file changes."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


@dataclass
class FileChange:
    """Represents a single file change."""

    id: str
    change_type: ChangeType
    file_path: str
    timestamp: float
    tool_name: str
    tool_args: Dict[str, Any]
    original_content: Optional[str] = None  # Content before change
    new_content: Optional[str] = None  # Content after change
    original_path: Optional[str] = None  # For renames
    checksum_before: Optional[str] = None
    checksum_after: Optional[str] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None  # Links to conversation message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "change_type": self.change_type.value,
            "file_path": self.file_path,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "original_content": self.original_content,
            "new_content": self.new_content,
            "original_path": self.original_path,
            "checksum_before": self.checksum_before,
            "checksum_after": self.checksum_after,
            "session_id": self.session_id,
            "message_id": self.message_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileChange":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            change_type=ChangeType(data["change_type"]),
            file_path=data["file_path"],
            timestamp=data["timestamp"],
            tool_name=data["tool_name"],
            tool_args=data.get("tool_args", {}),
            original_content=data.get("original_content"),
            new_content=data.get("new_content"),
            original_path=data.get("original_path"),
            checksum_before=data.get("checksum_before"),
            checksum_after=data.get("checksum_after"),
            session_id=data.get("session_id"),
            message_id=data.get("message_id"),
        )


@dataclass
class ChangeGroup:
    """A group of related changes (e.g., from a single tool execution)."""

    id: str
    changes: List[FileChange] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    description: str = ""
    tool_name: str = ""
    undone: bool = False
    session_id: Optional[str] = None
    message_id: Optional[str] = None  # Links the group to a conversation message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "changes": [c.to_dict() for c in self.changes],
            "timestamp": self.timestamp,
            "description": self.description,
            "tool_name": self.tool_name,
            "undone": self.undone,
            "session_id": self.session_id,
            "message_id": self.message_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChangeGroup":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            changes=[FileChange.from_dict(c) for c in data.get("changes", [])],
            timestamp=data.get("timestamp", time.time()),
            description=data.get("description", ""),
            tool_name=data.get("tool_name", ""),
            undone=data.get("undone", False),
            session_id=data.get("session_id"),
            message_id=data.get("message_id"),
        )


class FileChangeHistory:
    """Tracks file changes for undo/redo functionality.

    Uses SQLite for persistent storage and maintains an in-memory
    stack for fast undo/redo operations.
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        max_history: int = 10000,
        session_id: Optional[str] = None,
        project_path: Optional[Path] = None,
    ):
        """Initialize the file change history.

        Args:
            storage_dir: Deprecated, ignored. Uses consolidated project.db.
            max_history: Maximum number of change groups to keep (default 10000 for historical analysis)
            session_id: Current session identifier
            project_path: Path to project root for database access.
        """
        from victor.config.settings import get_project_paths
        from victor.core.undo_database import get_undo_database

        self.storage_dir = storage_dir or get_project_paths().changes_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.session_id = session_id or self._generate_session_id()

        # In-memory stacks for fast operations
        self._undo_stack: List[ChangeGroup] = []
        self._redo_stack: List[ChangeGroup] = []

        # Current change group being built
        self._current_group: Optional[ChangeGroup] = None

        # Dedicated undo.db (own write-lock; never contends with the graph indexer
        # on project.db). Schema owned by UndoDatabaseManager.
        self._db = get_undo_database(project_path)
        self._db_path = self._db.db_path
        self._init_database()

        # Load recent history
        self._load_recent_history()

        logger.info(f"ChangeTracker initialized with session {self.session_id}")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{int(time.time() * 1000)}"

    def _init_database(self) -> None:
        """Ensure the undo schema exists.

        The dedicated ``undo.db`` is a fresh, self-contained file whose schema is
        owned and versioned by :class:`UndoDatabaseManager`; there is no legacy
        table to destructively rebuild (that lived in the shared ``project.db``).
        """
        self._db.ensure_schema()

    def _load_recent_history(self) -> None:
        """Load recent change history from database."""
        conn = self._db.get_connection()
        cursor = conn.cursor()

        # Load recent non-undone groups
        cursor.execute(
            """
            SELECT data FROM change_groups
            WHERE session_id = ? AND undone = 0
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (self.session_id, self.max_history),
        )

        rows = cursor.fetchall()
        for (data,) in reversed(rows):  # Reverse to maintain order
            try:
                group = ChangeGroup.from_dict(json_loads(data))
                self._undo_stack.append(group)
            except Exception as e:
                logger.warning(f"Failed to load change group: {e}")

        # Connection managed by ProjectDatabaseManager
        logger.debug(f"Loaded {len(self._undo_stack)} change groups from history")

    @staticmethod
    def compute_checksum(content: str) -> str:
        """Compute MD5 checksum of content."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def begin_change_group(
        self, tool_name: str, description: str = "", message_id: Optional[str] = None
    ) -> str:
        """Begin a new change group.

        Args:
            tool_name: Name of the tool making changes
            description: Human-readable description
            message_id: Optional conversation message id this group belongs to

        Returns:
            Group ID
        """
        group_id = f"grp_{int(time.time() * 1000)}_{tool_name}"
        # Auto-resolve the turn/message id when the caller didn't supply one, so
        # edit/write/patch groups are attributed without touching each writer.
        if message_id is None:
            message_id = _current_correlation_id()
        self._current_group = ChangeGroup(
            id=group_id,
            tool_name=tool_name,
            description=description,
            timestamp=time.time(),
            session_id=self.session_id,
            message_id=message_id,
        )
        logger.debug(f"Started change group: {group_id}")
        return group_id

    def record_change(
        self,
        file_path: str,
        change_type: ChangeType,
        original_content: Optional[str] = None,
        new_content: Optional[str] = None,
        tool_name: str = "",
        tool_args: Optional[Dict[str, Any]] = None,
        original_path: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> FileChange:
        """Record a file change.

        Args:
            file_path: Path to the file being changed
            change_type: Type of change
            original_content: File content before change
            new_content: File content after change
            tool_name: Name of tool making the change
            tool_args: Arguments passed to the tool
            original_path: Original path (for renames)
            message_id: Optional conversation message id (defaults to the group's)

        Returns:
            The recorded FileChange
        """
        change_id = f"chg_{int(time.time() * 1000)}_{os.path.basename(file_path)}"

        group_message_id = self._current_group.message_id if self._current_group else None
        change = FileChange(
            id=change_id,
            change_type=change_type,
            file_path=str(file_path),
            timestamp=time.time(),
            tool_name=tool_name or (self._current_group.tool_name if self._current_group else ""),
            tool_args=tool_args or {},
            original_content=original_content,
            new_content=new_content,
            original_path=original_path,
            checksum_before=(self.compute_checksum(original_content) if original_content else None),
            checksum_after=self.compute_checksum(new_content) if new_content else None,
            session_id=self.session_id,
            message_id=message_id if message_id is not None else group_message_id,
        )

        if self._current_group:
            self._current_group.changes.append(change)
        else:
            # Auto-create a group for standalone changes
            self.begin_change_group(tool_name or "unknown", f"Changed {file_path}")
            if self._current_group:  # Always true after begin_change_group
                self._current_group.changes.append(change)
            self.commit_change_group()

        logger.debug(f"Recorded change: {change_type.value} on {file_path}")
        return change

    def commit_change_group(self) -> Optional[ChangeGroup]:
        """Commit the current change group.

        Returns:
            The committed ChangeGroup, or None if no current group
        """
        if not self._current_group or not self._current_group.changes:
            self._current_group = None
            return None

        group = self._current_group
        self._current_group = None

        # Add to undo stack
        self._undo_stack.append(group)

        # Clear redo stack on new change
        self._redo_stack.clear()

        # Persist to database
        self._save_group(group)

        # Trim history if needed
        self._trim_history()

        logger.info(f"Committed change group: {group.id} with {len(group.changes)} changes")
        return group

    # Bounded retry for transient "database is locked" errors: 2 retries, 100ms
    # apart. After that the error propagates (callers such as the edit tool treat
    # a failed undo-log write as non-fatal bookkeeping, not an edit failure).
    _SAVE_LOCK_RETRIES = 2
    _SAVE_LOCK_RETRY_DELAY_S = 0.1

    def _save_group(self, group: ChangeGroup) -> None:
        """Save a change group to the database, retrying transient lock errors."""
        attempts = 0
        while True:
            try:
                self._save_group_once(group)
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempts >= self._SAVE_LOCK_RETRIES:
                    raise
                attempts += 1
                logger.debug(
                    "change_tracker save hit a locked database (attempt %d/%d): %s",
                    attempts,
                    self._SAVE_LOCK_RETRIES,
                    exc,
                )
                time.sleep(self._SAVE_LOCK_RETRY_DELAY_S)

    def _save_group_once(self, group: ChangeGroup) -> None:
        """Single attempt at persisting a change group to the database.

        Group row + all file rows are written in one implicit transaction and a
        single ``commit`` so the group is DB-atomic (all-or-nothing); the
        write-lock is held only for that short commit, minimizing contention
        across concurrent sessions.
        """
        conn = self._db.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO change_groups
                (id, session_id, message_id, timestamp, description, tool_name, undone, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    group.id,
                    self.session_id,
                    group.message_id,
                    group.timestamp,
                    group.description,
                    group.tool_name,
                    1 if group.undone else 0,
                    json_dumps(group.to_dict()),
                ),
            )

            # Replace child rows so seq stays consistent on re-save (undo/redo).
            cursor.execute("DELETE FROM file_changes WHERE group_id = ?", (group.id,))

            # Also save individual changes for querying (seq = deterministic order).
            for seq, change in enumerate(group.changes):
                cursor.execute(
                    """
                    INSERT INTO file_changes
                    (id, group_id, seq, change_type, file_path, timestamp, tool_name,
                     original_content, new_content, original_path, checksum_before,
                     checksum_after, session_id, message_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        change.id,
                        group.id,
                        seq,
                        change.change_type.value,
                        change.file_path,
                        change.timestamp,
                        change.tool_name,
                        change.original_content,
                        change.new_content,
                        change.original_path,
                        change.checksum_before,
                        change.checksum_after,
                        change.session_id,
                        change.message_id,
                    ),
                )

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _trim_history(self) -> None:
        """Trim history to max_history size."""
        while len(self._undo_stack) > self.max_history:
            oldest = self._undo_stack.pop(0)
            # Optionally delete from database
            self._delete_group(oldest.id)

    def _delete_group(self, group_id: str) -> None:
        """Delete a change group from the database."""
        conn = self._db.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM file_changes WHERE group_id = ?", (group_id,))
        cursor.execute("DELETE FROM change_groups WHERE id = ?", (group_id,))
        conn.commit()
        # Connection managed by ProjectDatabaseManager

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    def undo(self, force: bool = False) -> Tuple[bool, str, List[str]]:
        """Undo the last change group for this session (group-atomic).

        Reverts every file in the most recent non-undone group. Writes are
        crash-safe (temp file + ``os.replace``); if any file fails mid-replay the
        already-reverted files are restored (all-or-nothing). A file another
        session changed since (checksum mismatch) is skipped with a warning
        unless ``force`` is set.

        Returns:
            Tuple of (success, message, list of affected files)
        """
        if not self._undo_stack:
            return False, "Nothing to undo", []

        group = self._undo_stack[-1]
        applied, skipped, error = self._replay_group(group, phase="undo", force=force)

        if error is not None:
            return False, f"Undo failed: {error} (changes rolled back)", []
        if not applied and skipped:
            return (
                False,
                f"Skipped {len(skipped)} file(s) changed by another session since; "
                "nothing undone. Re-run with force to override.",
                [],
            )

        # Commit the state transition only after a successful replay.
        self._undo_stack.pop()
        group.undone = True
        self._redo_stack.append(group)
        self._save_group(group)

        timestamp = datetime.fromtimestamp(group.timestamp).strftime("%H:%M:%S")
        plural = "s" if len(applied) != 1 else ""
        msg = f"Undid '{group.tool_name}' ({len(applied)} file{plural}) from {timestamp}"
        if skipped:
            msg += f"; skipped {len(skipped)} changed externally"
        return True, msg, applied

    def redo(self, force: bool = False) -> Tuple[bool, str, List[str]]:
        """Redo the last undone change group (group-atomic, crash-safe).

        Returns:
            Tuple of (success, message, list of affected files)
        """
        if not self._redo_stack:
            return False, "Nothing to redo", []

        group = self._redo_stack[-1]
        applied, skipped, error = self._replay_group(group, phase="redo", force=force)

        if error is not None:
            return False, f"Redo failed: {error} (changes rolled back)", []
        if not applied and skipped:
            return (
                False,
                f"Skipped {len(skipped)} file(s) changed by another session since; "
                "nothing redone. Re-run with force to override.",
                [],
            )

        self._redo_stack.pop()
        group.undone = False
        self._undo_stack.append(group)
        self._save_group(group)

        plural = "s" if len(applied) != 1 else ""
        msg = f"Redid '{group.tool_name}' ({len(applied)} file{plural})"
        if skipped:
            msg += f"; skipped {len(skipped)} changed externally"
        return True, msg, applied

    def _replay_group(
        self, group: ChangeGroup, *, phase: str, force: bool = False
    ) -> Tuple[List[str], List[str], Optional[str]]:
        """Replay a group's changes atomically (``undo`` reverses, ``redo`` applies).

        Returns ``(applied_files, skipped_files, error)``. On error, every file
        mutated in this call is restored to its pre-replay content and ``error``
        is set (``applied`` empty) — the group is all-or-nothing on the filesystem.
        Conflict-guarded files are skipped (recorded in ``skipped_files``).
        """
        changes = list(reversed(group.changes)) if phase == "undo" else list(group.changes)
        applied: List[str] = []
        skipped: List[str] = []
        journal: List[Dict[str, Tuple[bool, Optional[str]]]] = []

        for change in changes:
            if not force and self._has_conflict(change, phase):
                skipped.append(change.file_path)
                logger.warning(
                    "%s conflict guard skipped %s (changed by another session)",
                    phase,
                    change.file_path,
                )
                continue

            snap = self._snapshot_paths(self._paths_touched(change))
            try:
                if phase == "undo":
                    self._reverse_change(change)
                else:
                    self._apply_change(change)
            except Exception as e:
                # Roll back this change and everything applied earlier in the group.
                self._restore_snapshot(snap)
                for prev in reversed(journal):
                    self._restore_snapshot(prev)
                logger.error("Failed to %s %s: %s", phase, change.file_path, e)
                return [], skipped, f"{change.file_path}: {e}"
            journal.append(snap)
            applied.append(change.file_path)

        return applied, skipped, None

    @staticmethod
    def _paths_touched(change: FileChange) -> List[str]:
        """Filesystem paths a change may mutate (target + rename source)."""
        paths = [change.file_path]
        if change.original_path:
            paths.append(change.original_path)
        return paths

    @staticmethod
    def _snapshot_paths(paths: List[str]) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Snapshot (exists, content) for each path, for partial-failure rollback."""
        snap: Dict[str, Tuple[bool, Optional[str]]] = {}
        for pth in paths:
            p = Path(pth)
            if p.exists() and p.is_file():
                try:
                    snap[pth] = (True, p.read_text())
                except (OSError, UnicodeDecodeError):
                    snap[pth] = (True, None)  # unreadable/binary: existence-only
            else:
                snap[pth] = (False, None)
        return snap

    def _restore_snapshot(self, snap: Dict[str, Tuple[bool, Optional[str]]]) -> None:
        """Restore files to a snapshot captured by :meth:`_snapshot_paths`."""
        for pth, (existed, content) in snap.items():
            p = Path(pth)
            if existed:
                if content is not None:
                    self._atomic_write(p, content)
            elif p.exists():
                try:
                    p.unlink()
                except OSError:
                    logger.error("rollback: could not remove %s", p)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Crash-safe write: temp file in the same dir + fsync + ``os.replace``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".undo_tmp_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _current_checksum(self, file_path: str) -> Optional[str]:
        """MD5 of the file's current on-disk content, or None if absent/unreadable."""
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return None
        try:
            return self.compute_checksum(p.read_text())
        except (OSError, UnicodeDecodeError):
            return None

    def _has_conflict(self, change: FileChange, phase: str) -> bool:
        """True if the file is not in the state this replay step expects.

        ``undo`` expects the change's *after* state on disk; ``redo`` expects the
        *before* state. A mismatch means another session/tool touched the file
        since, so we skip rather than clobber it.
        """
        ct = change.change_type
        cur = self._current_checksum(change.file_path)
        if phase == "undo":
            if ct in (ChangeType.CREATE, ChangeType.MODIFY):
                return cur != change.checksum_after
            if ct == ChangeType.DELETE:
                return cur is not None  # expected absent (it was deleted)
            if ct == ChangeType.RENAME:
                return not Path(change.file_path).exists()  # expected at new path
        else:  # redo → expect the 'before' state
            if ct == ChangeType.CREATE:
                return cur is not None  # expected absent (not yet created)
            if ct == ChangeType.MODIFY:
                return cur != change.checksum_before
            if ct == ChangeType.DELETE:
                return cur != change.checksum_before  # expected original present
            if ct == ChangeType.RENAME and change.original_path:
                return not Path(change.original_path).exists()
        return False

    def _reverse_change(self, change: FileChange) -> None:
        """Reverse a single file change (crash-safe writes)."""
        path = Path(change.file_path)

        if change.change_type == ChangeType.CREATE:
            # Reverse create = delete
            if path.exists():
                path.unlink()
        elif change.change_type == ChangeType.DELETE:
            # Reverse delete = restore original content
            if change.original_content is not None:
                self._atomic_write(path, change.original_content)
        elif change.change_type == ChangeType.MODIFY:
            # Reverse modify = restore original content
            if change.original_content is not None:
                self._atomic_write(path, change.original_content)
        elif change.change_type == ChangeType.RENAME:
            # Reverse rename = rename back
            if change.original_path and path.exists():
                original = Path(change.original_path)
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(original))

    def _apply_change(self, change: FileChange) -> None:
        """Apply a single file change for redo (crash-safe writes)."""
        path = Path(change.file_path)

        if change.change_type == ChangeType.CREATE:
            if change.new_content is not None:
                self._atomic_write(path, change.new_content)
        elif change.change_type == ChangeType.DELETE:
            if path.exists():
                path.unlink()
        elif change.change_type == ChangeType.MODIFY:
            if change.new_content is not None:
                self._atomic_write(path, change.new_content)
        elif change.change_type == ChangeType.RENAME:
            if change.original_path:
                original = Path(change.original_path)
                if original.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(original), str(path))

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent change history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of change group summaries
        """
        history = []
        for group in reversed(self._undo_stack[-limit:]):
            history.append(
                {
                    "id": group.id,
                    "timestamp": datetime.fromtimestamp(group.timestamp).isoformat(),
                    "tool_name": group.tool_name,
                    "description": group.description,
                    "file_count": len(group.changes),
                    "files": [c.file_path for c in group.changes[:5]],  # First 5 files
                    "undone": group.undone,
                }
            )
        return history

    def get_file_history(self, file_path: str, limit: int = 10) -> List[FileChange]:
        """Get change history for a specific file.

        Args:
            file_path: Path to the file
            limit: Maximum number of entries

        Returns:
            List of FileChange objects
        """
        conn = self._db.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, change_type, file_path, timestamp, tool_name,
                   original_content, new_content, original_path,
                   checksum_before, checksum_after
            FROM file_changes
            WHERE file_path = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (str(file_path), limit),
        )

        changes = []
        for row in cursor.fetchall():
            changes.append(
                FileChange(
                    id=row[0],
                    change_type=ChangeType(row[1]),
                    file_path=row[2],
                    timestamp=row[3],
                    tool_name=row[4],
                    tool_args={},
                    original_content=row[5],
                    new_content=row[6],
                    original_path=row[7],
                    checksum_before=row[8],
                    checksum_after=row[9],
                )
            )

        # Connection managed by ProjectDatabaseManager
        return changes

    def clear_history(self) -> int:
        """Clear all change history.

        Returns:
            Number of groups cleared
        """
        count = len(self._undo_stack) + len(self._redo_stack)

        self._undo_stack.clear()
        self._redo_stack.clear()

        conn = self._db.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM file_changes WHERE 1=1")
        cursor.execute("DELETE FROM change_groups WHERE 1=1")
        conn.commit()
        # Connection managed by ProjectDatabaseManager

        logger.info(f"Cleared {count} change groups from history")
        return count


# Global instance for easy access
_file_change_history: Optional[FileChangeHistory] = None


def get_change_tracker() -> FileChangeHistory:
    """Get or create the global file change history instance.

    Note: Function name kept for backward compatibility.
    """
    global _file_change_history
    if _file_change_history is None:
        _file_change_history = FileChangeHistory()
    return _file_change_history


def set_change_tracker(tracker: FileChangeHistory) -> None:
    """Set the global file change history instance.

    Note: Function name kept for backward compatibility.
    """
    global _file_change_history
    _file_change_history = tracker


def reset_change_tracker() -> None:
    """Reset the global file change history instance.

    Used primarily for testing to ensure a fresh history between tests.

    Note: Function name kept for backward compatibility.
    """
    global _file_change_history
    _file_change_history = None
