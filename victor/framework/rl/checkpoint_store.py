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

"""Checkpoint store for RL policy persistence.

This module provides checkpoint storage for RL policies, enabling:
- Periodic automatic checkpointing
- Version tagging for policy states
- Rollback to previous versions
- Checkpoint comparison and diff

Checkpoint Format:
- Learner name
- Version tag (semantic or timestamp-based)
- Serialized Q-values/weights
- Metadata (samples, performance metrics)
- Timestamp

Sprint 6: Observability & Polish
"""

import gzip
import hashlib
import json
from victor.core.json_utils import json_dumps, json_loads
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PolicyCheckpoint:
    """A checkpoint of a learner's policy state.

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        learner_name: Name of the learner
        version: Version tag (e.g., "v1.2.3" or timestamp)
        state: Serialized policy state (Q-values, weights, etc.)
        metadata: Performance metrics at checkpoint time
        timestamp: When checkpoint was created
        parent_id: ID of parent checkpoint (for lineage)
        tags: User-defined tags
    """

    checkpoint_id: str
    learner_name: str
    version: str
    state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def state_hash(self) -> str:
        """Compute hash of the state for comparison."""
        state_json = json_dumps(self.state, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "learner_name": self.learner_name,
            "version": self.version,
            "state": self.state,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "state_hash": self.state_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyCheckpoint":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            learner_name=data["learner_name"],
            version=data["version"],
            state=data["state"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
        )


@dataclass
class CheckpointDiff:
    """Difference between two checkpoints.

    Attributes:
        from_version: Source version
        to_version: Target version
        added_keys: Keys added in target
        removed_keys: Keys removed from source
        changed_keys: Keys with different values
        unchanged_keys: Keys with same values
    """

    from_version: str
    to_version: str
    added_keys: List[str] = field(default_factory=list)
    removed_keys: List[str] = field(default_factory=list)
    changed_keys: List[str] = field(default_factory=list)
    unchanged_keys: List[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if there are any differences."""
        return bool(self.added_keys or self.removed_keys or self.changed_keys)

    @property
    def change_ratio(self) -> float:
        """Compute ratio of changed keys to total keys."""
        total = (
            len(self.added_keys)
            + len(self.removed_keys)
            + len(self.changed_keys)
            + len(self.unchanged_keys)
        )
        if total == 0:
            return 0.0
        changed = len(self.added_keys) + len(self.removed_keys) + len(self.changed_keys)
        return changed / total


class CheckpointStore:
    """Storage for policy checkpoints.

    Provides persistent storage for policy checkpoints with support
    for versioning, tagging, and lineage tracking.

    Storage Options:
    - SQLite database (default)
    - File system (compressed JSON)

    Usage:
        store = CheckpointStore(storage_path)

        # Create checkpoint
        checkpoint = store.create_checkpoint(
            learner_name="tool_selector",
            version="v1.0.0",
            state=learner.export_state(),
            metadata={"success_rate": 0.85},
        )

        # List checkpoints
        checkpoints = store.list_checkpoints("tool_selector")

        # Rollback
        old_state = store.get_checkpoint("tool_selector", "v0.9.0")
    """

    # Maximum checkpoints to keep per learner
    MAX_CHECKPOINTS_PER_LEARNER = 50

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        db_connection: Optional[Any] = None,
    ):
        """Initialize checkpoint store.

        Args:
            storage_path: Path for file-based storage
            db_connection: SQLite connection for database storage
        """
        self.storage_path = storage_path or Path.home() / ".victor" / "checkpoints"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.db = db_connection

        # In-memory cache for fast access
        self._cache: Dict[str, Dict[str, PolicyCheckpoint]] = {}

        # Auto-increment for checkpoint IDs
        self._next_id = 1

        if db_connection:
            self._ensure_tables()
            self._load_from_db()
        else:
            self._load_from_files()

    def _ensure_tables(self) -> None:
        """Create database tables for checkpoint storage."""
        if not self.db:
            return

        cursor = self.db.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS policy_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                learner_name TEXT NOT NULL,
                version TEXT NOT NULL,
                state TEXT NOT NULL,
                metadata TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                parent_id TEXT,
                tags TEXT NOT NULL,
                state_hash TEXT NOT NULL,
                UNIQUE(learner_name, version)
            )
            """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkpoints_learner
            ON policy_checkpoints(learner_name, timestamp DESC)
            """)

        self.db.commit()

    def _load_from_db(self) -> None:
        """Load checkpoints from database."""
        if not self.db:
            return

        cursor = self.db.cursor()

        try:
            cursor.execute("SELECT * FROM policy_checkpoints ORDER BY timestamp DESC")
            for row in cursor.fetchall():
                row_dict = dict(row)
                checkpoint = PolicyCheckpoint(
                    checkpoint_id=row_dict["checkpoint_id"],
                    learner_name=row_dict["learner_name"],
                    version=row_dict["version"],
                    state=json_loads(row_dict["state"]),
                    metadata=json_loads(row_dict["metadata"]),
                    timestamp=row_dict["timestamp"],
                    parent_id=row_dict["parent_id"],
                    tags=json_loads(row_dict["tags"]),
                )

                if checkpoint.learner_name not in self._cache:
                    self._cache[checkpoint.learner_name] = {}
                self._cache[checkpoint.learner_name][checkpoint.version] = checkpoint

                # Update next ID
                try:
                    id_num = int(checkpoint.checkpoint_id.split("_")[-1])
                    self._next_id = max(self._next_id, id_num + 1)
                except (ValueError, IndexError):
                    pass

            logger.info(
                f"CheckpointStore: Loaded {sum(len(v) for v in self._cache.values())} checkpoints"
            )

        except Exception as e:
            logger.debug(f"CheckpointStore: Could not load from database: {e}")

    def _load_from_files(self) -> None:
        """Load checkpoints from file system."""
        for learner_dir in self.storage_path.iterdir():
            if not learner_dir.is_dir():
                continue

            learner_name = learner_dir.name
            self._cache[learner_name] = {}

            for checkpoint_file in learner_dir.glob("*.json.gz"):
                try:
                    with gzip.open(checkpoint_file, "rt") as f:
                        data = json.load(f)
                        checkpoint = PolicyCheckpoint.from_dict(data)
                        self._cache[learner_name][checkpoint.version] = checkpoint

                        # Update next ID
                        try:
                            id_num = int(checkpoint.checkpoint_id.split("_")[-1])
                            self._next_id = max(self._next_id, id_num + 1)
                        except (ValueError, IndexError):
                            pass

                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")

    def create_checkpoint(
        self,
        learner_name: str,
        version: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> PolicyCheckpoint:
        """Create a new checkpoint.

        Args:
            learner_name: Name of the learner
            version: Version tag
            state: Policy state to checkpoint
            metadata: Optional performance metrics
            parent_id: Optional parent checkpoint ID
            tags: Optional user tags

        Returns:
            Created PolicyCheckpoint
        """
        checkpoint_id = f"ckpt_{learner_name}_{self._next_id}"
        self._next_id += 1

        checkpoint = PolicyCheckpoint(
            checkpoint_id=checkpoint_id,
            learner_name=learner_name,
            version=version,
            state=state,
            metadata=metadata or {},
            parent_id=parent_id,
            tags=tags or [],
        )

        # Add to cache
        if learner_name not in self._cache:
            self._cache[learner_name] = {}
        self._cache[learner_name][version] = checkpoint

        # Persist
        self._save_checkpoint(checkpoint)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(learner_name)

        logger.info(
            f"CheckpointStore: Created checkpoint {checkpoint_id} " f"for {learner_name} v{version}"
        )

        return checkpoint

    def _save_checkpoint(self, checkpoint: PolicyCheckpoint) -> None:
        """Save checkpoint to storage."""
        if self.db:
            self._save_to_db(checkpoint)
        else:
            self._save_to_file(checkpoint)

    def _save_to_db(self, checkpoint: PolicyCheckpoint) -> None:
        """Save checkpoint to database."""
        if not self.db:
            return

        cursor = self.db.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO policy_checkpoints
            (checkpoint_id, learner_name, version, state, metadata,
             timestamp, parent_id, tags, state_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint.checkpoint_id,
                checkpoint.learner_name,
                checkpoint.version,
                json_dumps(checkpoint.state),
                json_dumps(checkpoint.metadata),
                checkpoint.timestamp,
                checkpoint.parent_id,
                json_dumps(checkpoint.tags),
                checkpoint.state_hash,
            ),
        )

        self.db.commit()

    def _save_to_file(self, checkpoint: PolicyCheckpoint) -> None:
        """Save checkpoint to compressed file."""
        learner_dir = self.storage_path / checkpoint.learner_name
        learner_dir.mkdir(exist_ok=True)

        # Safe version string for filename
        safe_version = checkpoint.version.replace("/", "_").replace("\\", "_")
        filepath = learner_dir / f"{safe_version}.json.gz"

        with gzip.open(filepath, "wt") as f:
            json.dump(checkpoint.to_dict(), f)

    def _cleanup_old_checkpoints(self, learner_name: str) -> None:
        """Remove old checkpoints beyond the limit."""
        if learner_name not in self._cache:
            return

        checkpoints = list(self._cache[learner_name].values())
        if len(checkpoints) <= self.MAX_CHECKPOINTS_PER_LEARNER:
            return

        # Sort by timestamp (oldest first)
        checkpoints.sort(key=lambda c: c.timestamp)

        # Remove oldest checkpoints
        to_remove = checkpoints[: len(checkpoints) - self.MAX_CHECKPOINTS_PER_LEARNER]

        for checkpoint in to_remove:
            self._delete_checkpoint(checkpoint)

    def _delete_checkpoint(self, checkpoint: PolicyCheckpoint) -> None:
        """Delete a checkpoint from storage."""
        # Remove from cache
        if checkpoint.learner_name in self._cache:
            self._cache[checkpoint.learner_name].pop(checkpoint.version, None)

        # Remove from storage
        if self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "DELETE FROM policy_checkpoints WHERE checkpoint_id = ?",
                (checkpoint.checkpoint_id,),
            )
            self.db.commit()
        else:
            safe_version = checkpoint.version.replace("/", "_").replace("\\", "_")
            filepath = self.storage_path / checkpoint.learner_name / f"{safe_version}.json.gz"
            if filepath.exists():
                filepath.unlink()

    def get_checkpoint(self, learner_name: str, version: str) -> Optional[PolicyCheckpoint]:
        """Get a specific checkpoint.

        Args:
            learner_name: Learner name
            version: Version tag

        Returns:
            PolicyCheckpoint or None
        """
        return self._cache.get(learner_name, {}).get(version)

    def get_latest_checkpoint(self, learner_name: str) -> Optional[PolicyCheckpoint]:
        """Get the most recent checkpoint for a learner.

        Args:
            learner_name: Learner name

        Returns:
            Most recent PolicyCheckpoint or None
        """
        checkpoints = self._cache.get(learner_name, {})
        if not checkpoints:
            return None

        return max(checkpoints.values(), key=lambda c: c.timestamp)

    def list_checkpoints(
        self,
        learner_name: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[PolicyCheckpoint]:
        """List checkpoints with optional filtering.

        Args:
            learner_name: Optional filter by learner
            tag: Optional filter by tag

        Returns:
            List of matching checkpoints
        """
        result = []

        if learner_name:
            checkpoints = list(self._cache.get(learner_name, {}).values())
        else:
            checkpoints = [
                cp for learner_cps in self._cache.values() for cp in learner_cps.values()
            ]

        for checkpoint in checkpoints:
            if tag and tag not in checkpoint.tags:
                continue
            result.append(checkpoint)

        # Sort by timestamp (newest first)
        result.sort(key=lambda c: c.timestamp, reverse=True)
        return result

    def diff_checkpoints(
        self,
        learner_name: str,
        from_version: str,
        to_version: str,
    ) -> Optional[CheckpointDiff]:
        """Compare two checkpoints.

        Args:
            learner_name: Learner name
            from_version: Source version
            to_version: Target version

        Returns:
            CheckpointDiff or None if versions not found
        """
        from_cp = self.get_checkpoint(learner_name, from_version)
        to_cp = self.get_checkpoint(learner_name, to_version)

        if not from_cp or not to_cp:
            return None

        from_keys = set(self._flatten_keys(from_cp.state))
        to_keys = set(self._flatten_keys(to_cp.state))

        added = list(to_keys - from_keys)
        removed = list(from_keys - to_keys)

        common = from_keys & to_keys
        changed = []
        unchanged = []

        for key in common:
            from_val = self._get_nested_value(from_cp.state, key)
            to_val = self._get_nested_value(to_cp.state, key)

            if from_val != to_val:
                changed.append(key)
            else:
                unchanged.append(key)

        return CheckpointDiff(
            from_version=from_version,
            to_version=to_version,
            added_keys=added,
            removed_keys=removed,
            changed_keys=changed,
            unchanged_keys=unchanged,
        )

    def _flatten_keys(self, d: Dict[str, Any], prefix: str = "") -> List[str]:
        """Flatten nested dictionary keys."""
        keys = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys.extend(self._flatten_keys(v, key))
            else:
                keys.append(key)
        return keys

    def _get_nested_value(self, d: Dict[str, Any], key: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        parts = key.split(".")
        current = d
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def tag_checkpoint(self, learner_name: str, version: str, tag: str) -> bool:
        """Add a tag to a checkpoint.

        Args:
            learner_name: Learner name
            version: Version tag
            tag: Tag to add

        Returns:
            True if successful
        """
        checkpoint = self.get_checkpoint(learner_name, version)
        if not checkpoint:
            return False

        if tag not in checkpoint.tags:
            checkpoint.tags.append(tag)
            self._save_checkpoint(checkpoint)

        return True

    def export_metrics(self) -> Dict[str, Any]:
        """Export checkpoint store metrics.

        Returns:
            Dictionary with metrics
        """
        total_checkpoints = sum(len(cps) for cps in self._cache.values())
        learner_counts = {name: len(cps) for name, cps in self._cache.items()}

        return {
            "total_checkpoints": total_checkpoints,
            "learners": len(self._cache),
            "checkpoints_per_learner": learner_counts,
            "storage_path": str(self.storage_path),
        }


# Global singleton
_checkpoint_store: Optional[CheckpointStore] = None


def get_checkpoint_store(
    storage_path: Optional[Path] = None,
    db_connection: Optional[Any] = None,
) -> CheckpointStore:
    """Get global checkpoint store (lazy init).

    Args:
        storage_path: Optional storage path
        db_connection: Optional database connection

    Returns:
        CheckpointStore singleton
    """
    global _checkpoint_store
    if _checkpoint_store is None:
        _checkpoint_store = CheckpointStore(storage_path, db_connection)
    return _checkpoint_store
