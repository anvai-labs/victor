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

"""Workspace snapshots for safe AI-assisted code changes.

This module provides:
- Capture workspace state before AI makes changes
- Restore to previous state if changes fail or are unwanted
- Diff between snapshots and current state
- Automatic cleanup of old snapshots

Usage:
    from victor.agent.snapshots import SnapshotManager

    manager = SnapshotManager()

    # Before making changes
    snapshot_id = manager.create_snapshot(
        files=["src/api.py", "src/utils.py"],
        description="Before refactoring API"
    )

    # If something goes wrong
    manager.restore_snapshot(snapshot_id)

    # See what changed
    diff = manager.diff_snapshot(snapshot_id)
"""

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FileSnapshot:
    """Snapshot of a single file."""

    path: str  # Relative path from workspace root
    content: Optional[str] = None  # File content (None if file didn't exist)
    content_hash: str = ""  # SHA256 hash for quick comparison
    permissions: int = 0o644  # File permissions
    existed: bool = True  # Whether file existed when snapshot was taken

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "content": self.content,
            "content_hash": self.content_hash,
            "permissions": self.permissions,
            "existed": self.existed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileSnapshot":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            content=data.get("content"),
            content_hash=data.get("content_hash", ""),
            permissions=data.get("permissions", 0o644),
            existed=data.get("existed", True),
        )


@dataclass
class WorkspaceSnapshot:
    """Complete snapshot of workspace state."""

    snapshot_id: str
    created_at: str  # ISO format timestamp
    description: str
    files: List[FileSnapshot] = field(default_factory=list)
    workspace_root: str = ""
    git_ref: Optional[str] = None  # Git HEAD ref at snapshot time
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "description": self.description,
            "files": [f.to_dict() for f in self.files],
            "workspace_root": self.workspace_root,
            "git_ref": self.git_ref,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceSnapshot":
        """Create from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            created_at=data["created_at"],
            description=data.get("description", ""),
            files=[FileSnapshot.from_dict(f) for f in data.get("files", [])],
            workspace_root=data.get("workspace_root", ""),
            git_ref=data.get("git_ref"),
            metadata=data.get("metadata", {}),
        )

    @property
    def file_count(self) -> int:
        """Number of files in snapshot."""
        return len(self.files)

    def get_file(self, path: str) -> Optional[FileSnapshot]:
        """Get file snapshot by path."""
        for f in self.files:
            if f.path == path:
                return f
        return None


@dataclass
class FileDiff:
    """Difference between snapshot and current state."""

    path: str
    status: str  # "added", "modified", "deleted", "unchanged"
    snapshot_content: Optional[str] = None
    current_content: Optional[str] = None
    diff_lines: List[str] = field(default_factory=list)


class FileSnapshotStore:
    """Stores workspace snapshots for safe AI-assisted changes.

    Snapshots are stored in memory during a session and can optionally
    be persisted to disk for recovery across sessions.

    Note: Previously named `SnapshotManager`. Alias kept for backward compatibility.
    """

    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        max_snapshots: int = 20,
        persist_dir: Optional[Path] = None,
    ):
        """Initialize snapshot store.

        Args:
            workspace_root: Root directory of workspace (default: cwd)
            max_snapshots: Maximum number of snapshots to keep
            persist_dir: Directory to persist snapshots (optional)
        """
        self.workspace_root = workspace_root or Path.cwd()
        self.max_snapshots = max_snapshots
        self.persist_dir = persist_dir
        self._snapshots: Dict[str, WorkspaceSnapshot] = {}
        self._snapshot_order: List[str] = []  # Oldest to newest
        self._counter = 0

        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted_snapshots()

    def _generate_snapshot_id(self) -> str:
        """Generate a unique snapshot ID."""
        self._counter += 1
        timestamp = datetime.now().strftime("%H%M%S")
        return f"snap_{self._counter}_{timestamp}"

    def _hash_content(self, content: str) -> str:
        """Generate SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _get_git_ref(self) -> Optional[str]:
        """Get current git HEAD reference."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self.workspace_root,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception as e:
            logger.debug("Failed to resolve git HEAD for snapshot: %s", e)
        return None

    def _read_file_safe(self, path: Path) -> Optional[str]:
        """Safely read file content, return None if unreadable."""
        try:
            # Skip binary files
            if self._is_binary(path):
                return None
            return path.read_text(encoding="utf-8")
        except Exception:
            return None

    def _is_binary(self, path: Path) -> bool:
        """Check if file is likely binary."""
        binary_extensions = {
            ".pyc",
            ".pyo",
            ".so",
            ".dll",
            ".exe",
            ".bin",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".ico",
            ".pdf",
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".7z",
            ".db",
            ".sqlite",
            ".sqlite3",
        }
        return path.suffix.lower() in binary_extensions

    def create_snapshot(
        self,
        files: Optional[List[str]] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a snapshot of specified files or tracked changes.

        Args:
            files: List of file paths to snapshot (relative to workspace root)
                   If None, snapshots all modified files (git status)
            description: Human-readable description
            metadata: Additional metadata to store

        Returns:
            Snapshot ID
        """
        snapshot_id = self._generate_snapshot_id()

        # If no files specified, get modified files from git
        if files is None:
            files = self._get_modified_files()

        file_snapshots = []
        for file_path in files:
            full_path = self.workspace_root / file_path

            if full_path.exists():
                content = self._read_file_safe(full_path)
                file_snapshots.append(
                    FileSnapshot(
                        path=file_path,
                        content=content,
                        content_hash=self._hash_content(content) if content else "",
                        permissions=full_path.stat().st_mode & 0o777,
                        existed=True,
                    )
                )
            else:
                # File doesn't exist yet - record that
                file_snapshots.append(
                    FileSnapshot(
                        path=file_path,
                        content=None,
                        content_hash="",
                        existed=False,
                    )
                )

        snapshot = WorkspaceSnapshot(
            snapshot_id=snapshot_id,
            created_at=datetime.now().isoformat(),
            description=description,
            files=file_snapshots,
            workspace_root=str(self.workspace_root),
            git_ref=self._get_git_ref(),
            metadata=metadata or {},
        )

        # Store snapshot
        self._snapshots[snapshot_id] = snapshot
        self._snapshot_order.append(snapshot_id)

        # Cleanup old snapshots
        self._cleanup_old_snapshots()

        # Persist if configured
        if self.persist_dir:
            self._persist_snapshot(snapshot)

        logger.info(f"Created snapshot {snapshot_id}: {description} ({len(files)} files)")
        return snapshot_id

    def _get_modified_files(self) -> List[str]:
        """Get list of modified files from git status."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.workspace_root,
            )
            if result.returncode == 0:
                files = []
                for line in result.stdout.strip().split("\n"):
                    if line:
                        # Format: "XY filename" where X=index, Y=worktree
                        line[:2]
                        filename = line[3:].strip()
                        # Handle renamed files: "R  old -> new"
                        if " -> " in filename:
                            filename = filename.split(" -> ")[1]
                        if filename:
                            files.append(filename)
                return files
        except Exception as e:
            logger.warning(f"Failed to get git status: {e}")
        return []

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore workspace to snapshot state.

        Args:
            snapshot_id: ID of snapshot to restore

        Returns:
            True if restoration successful
        """
        snapshot = self._snapshots.get(snapshot_id)
        if not snapshot:
            logger.error(f"Snapshot not found: {snapshot_id}")
            return False

        restored_count = 0
        errors = []

        for file_snap in snapshot.files:
            full_path = self.workspace_root / file_snap.path

            try:
                if not file_snap.existed:
                    # File didn't exist in snapshot - delete it if it exists now
                    if full_path.exists():
                        full_path.unlink()
                        logger.debug(f"Deleted: {file_snap.path}")
                        restored_count += 1
                elif file_snap.content is not None:
                    # Restore file content
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(file_snap.content, encoding="utf-8")
                    os.chmod(full_path, file_snap.permissions)
                    logger.debug(f"Restored: {file_snap.path}")
                    restored_count += 1
            except Exception as e:
                errors.append(f"{file_snap.path}: {e}")
                logger.error(f"Failed to restore {file_snap.path}: {e}")

        if errors:
            logger.warning(f"Restored {restored_count} files with {len(errors)} errors")
            return False

        logger.info(f"Restored snapshot {snapshot_id}: {restored_count} files")
        return True

    def diff_snapshot(self, snapshot_id: str) -> List[FileDiff]:
        """Get diff between snapshot and current state.

        Args:
            snapshot_id: ID of snapshot to compare

        Returns:
            List of file differences
        """
        snapshot = self._snapshots.get(snapshot_id)
        if not snapshot:
            return []

        diffs = []
        current_files: Set[str] = set()

        for file_snap in snapshot.files:
            full_path = self.workspace_root / file_snap.path
            current_content = self._read_file_safe(full_path) if full_path.exists() else None
            current_files.add(file_snap.path)

            if not file_snap.existed and current_content is not None:
                # File was added
                diffs.append(
                    FileDiff(
                        path=file_snap.path,
                        status="added",
                        snapshot_content=None,
                        current_content=current_content,
                    )
                )
            elif file_snap.existed and current_content is None:
                # File was deleted
                diffs.append(
                    FileDiff(
                        path=file_snap.path,
                        status="deleted",
                        snapshot_content=file_snap.content,
                        current_content=None,
                    )
                )
            elif file_snap.content != current_content:
                # File was modified
                diffs.append(
                    FileDiff(
                        path=file_snap.path,
                        status="modified",
                        snapshot_content=file_snap.content,
                        current_content=current_content,
                        diff_lines=self._generate_diff_lines(
                            file_snap.content or "", current_content or ""
                        ),
                    )
                )
            else:
                # File unchanged
                diffs.append(
                    FileDiff(
                        path=file_snap.path,
                        status="unchanged",
                    )
                )

        return diffs

    def _generate_diff_lines(self, old: str, new: str) -> List[str]:
        """Generate unified diff lines between old and new content."""
        import difflib

        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="snapshot",
            tofile="current",
            lineterm="",
        )
        return list(diff)

    def list_snapshots(self, limit: int = 10) -> List[WorkspaceSnapshot]:
        """List recent snapshots.

        Args:
            limit: Maximum number to return

        Returns:
            List of snapshots (newest first)
        """
        snapshot_ids = self._snapshot_order[-limit:][::-1]
        return [self._snapshots[sid] for sid in snapshot_ids if sid in self._snapshots]

    def get_snapshot(self, snapshot_id: str) -> Optional[WorkspaceSnapshot]:
        """Get snapshot by ID."""
        return self._snapshots.get(snapshot_id)

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot.

        Args:
            snapshot_id: ID of snapshot to delete

        Returns:
            True if deleted
        """
        if snapshot_id not in self._snapshots:
            return False

        del self._snapshots[snapshot_id]
        self._snapshot_order.remove(snapshot_id)

        if self.persist_dir:
            persist_path = self.persist_dir / f"{snapshot_id}.json"
            if persist_path.exists():
                persist_path.unlink()

        return True

    def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots exceeding max_snapshots limit."""
        while len(self._snapshot_order) > self.max_snapshots:
            oldest_id = self._snapshot_order.pop(0)
            if oldest_id in self._snapshots:
                del self._snapshots[oldest_id]

            if self.persist_dir:
                persist_path = self.persist_dir / f"{oldest_id}.json"
                if persist_path.exists():
                    persist_path.unlink()

    def _persist_snapshot(self, snapshot: WorkspaceSnapshot) -> None:
        """Persist snapshot to disk."""
        if not self.persist_dir:
            return

        persist_path = self.persist_dir / f"{snapshot.snapshot_id}.json"
        with open(persist_path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

    def _load_persisted_snapshots(self) -> None:
        """Load persisted snapshots from disk."""
        if not self.persist_dir or not self.persist_dir.exists():
            return

        for snapshot_file in sorted(self.persist_dir.glob("snap_*.json")):
            try:
                with open(snapshot_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                snapshot = WorkspaceSnapshot.from_dict(data)
                self._snapshots[snapshot.snapshot_id] = snapshot
                self._snapshot_order.append(snapshot.snapshot_id)
            except Exception as e:
                logger.warning(f"Failed to load snapshot {snapshot_file}: {e}")

    def clear_all(self) -> int:
        """Clear all snapshots.

        Returns:
            Number of snapshots cleared
        """
        count = len(self._snapshots)
        self._snapshots.clear()
        self._snapshot_order.clear()

        if self.persist_dir:
            for snapshot_file in self.persist_dir.glob("snap_*.json"):
                snapshot_file.unlink()

        return count


# Default singleton instance
_default_store: Optional[FileSnapshotStore] = None


def get_snapshot_store() -> FileSnapshotStore:
    """Get the default snapshot store instance."""
    global _default_store
    if _default_store is None:
        _default_store = FileSnapshotStore()
    return _default_store


def set_snapshot_store(store: FileSnapshotStore) -> None:
    """Set the default snapshot store instance."""
    global _default_store
    _default_store = store


def reset_snapshot_store() -> None:
    """Reset the default snapshot store singleton."""
    global _default_store
    _default_store = None
