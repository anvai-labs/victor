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

"""Protocol definitions for conversation state checkpointing.

This module defines the abstract interfaces and data structures for
time-travel debugging, inspired by LangGraph's checkpoint system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import uuid


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""

    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when a checkpoint cannot be found."""

    pass


class CheckpointStorageError(CheckpointError):
    """Raised when checkpoint storage operations fail."""

    pass


class DiffType(str, Enum):
    """Types of differences between checkpoints."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class CheckpointMetadata:
    """Metadata for a conversation state checkpoint.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint
        session_id: ID of the session this checkpoint belongs to
        timestamp: When the checkpoint was created
        stage: Current conversation stage (from ConversationStage)
        tool_count: Number of tools executed at checkpoint time
        message_count: Number of messages in conversation
        parent_id: ID of parent checkpoint (for fork tracking)
        description: Human-readable description
        tags: Optional tags for categorization
        version: Schema version for forward compatibility
    """

    checkpoint_id: str
    session_id: str
    timestamp: datetime
    stage: str
    tool_count: int
    message_count: int
    parent_id: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    version: int = 1

    @classmethod
    def create(
        cls,
        session_id: str,
        stage: str,
        tool_count: int,
        message_count: int,
        parent_id: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> "CheckpointMetadata":
        """Factory method to create a new checkpoint metadata.

        Args:
            session_id: ID of the session
            stage: Current conversation stage
            tool_count: Number of tools executed
            message_count: Number of messages
            parent_id: Optional parent checkpoint ID
            description: Optional description
            tags: Optional list of tags

        Returns:
            New CheckpointMetadata instance
        """
        return cls(
            checkpoint_id=f"ckpt_{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            stage=stage,
            tool_count=tool_count,
            message_count=message_count,
            parent_id=parent_id,
            description=description,
            tags=tags or [],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "stage": self.stage,
            "tool_count": self.tool_count,
            "message_count": self.message_count,
            "parent_id": self.parent_id,
            "description": self.description,
            "tags": self.tags,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Deserialize metadata from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            checkpoint_id=data["checkpoint_id"],
            session_id=data["session_id"],
            timestamp=timestamp,
            stage=data.get("stage", "INITIAL"),
            tool_count=data.get("tool_count", 0),
            message_count=data.get("message_count", 0),
            parent_id=data.get("parent_id"),
            description=data.get("description"),
            tags=data.get("tags", []),
            version=data.get("version", 1),
        )


@dataclass
class FieldDiff:
    """Represents a difference in a single field."""

    field_name: str
    diff_type: DiffType
    old_value: Any = None
    new_value: Any = None


@dataclass
class CheckpointDiff:
    """Difference between two checkpoints.

    Provides a structured view of what changed between two conversation states,
    enabling debugging and understanding of agent behavior.

    Attributes:
        checkpoint_a: ID of the first checkpoint
        checkpoint_b: ID of the second checkpoint
        metadata_diff: Differences in metadata fields
        messages_added: Messages added between checkpoints
        messages_removed: Messages removed between checkpoints
        tools_added: Tool executions added between checkpoints
        files_observed_diff: Changes to observed files
        files_modified_diff: Changes to modified files
        stage_changes: List of stage transitions
    """

    checkpoint_a: str
    checkpoint_b: str
    metadata_diff: List[FieldDiff] = field(default_factory=list)
    messages_added: int = 0
    messages_removed: int = 0
    tools_added: List[str] = field(default_factory=list)
    tools_removed: List[str] = field(default_factory=list)
    files_observed_diff: List[FieldDiff] = field(default_factory=list)
    files_modified_diff: List[FieldDiff] = field(default_factory=list)
    stage_changes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize diff to dictionary."""
        return {
            "checkpoint_a": self.checkpoint_a,
            "checkpoint_b": self.checkpoint_b,
            "metadata_diff": [
                {
                    "field_name": d.field_name,
                    "diff_type": d.diff_type.value,
                    "old_value": d.old_value,
                    "new_value": d.new_value,
                }
                for d in self.metadata_diff
            ],
            "messages_added": self.messages_added,
            "messages_removed": self.messages_removed,
            "tools_added": self.tools_added,
            "tools_removed": self.tools_removed,
            "files_observed_diff": [
                {
                    "field_name": d.field_name,
                    "diff_type": d.diff_type.value,
                    "old_value": d.old_value,
                    "new_value": d.new_value,
                }
                for d in self.files_observed_diff
            ],
            "files_modified_diff": [
                {
                    "field_name": d.field_name,
                    "diff_type": d.diff_type.value,
                    "old_value": d.old_value,
                    "new_value": d.new_value,
                }
                for d in self.files_modified_diff
            ],
            "stage_changes": self.stage_changes,
        }

    def summary(self) -> str:
        """Generate a human-readable summary of the diff."""
        lines = [f"Diff: {self.checkpoint_a} -> {self.checkpoint_b}"]

        if self.stage_changes:
            lines.append(f"  Stage changes: {' -> '.join(self.stage_changes)}")

        if self.messages_added or self.messages_removed:
            lines.append(f"  Messages: +{self.messages_added} / -{self.messages_removed}")

        if self.tools_added:
            lines.append(f"  Tools added: {', '.join(self.tools_added[:5])}")
            if len(self.tools_added) > 5:
                lines.append(f"    ... and {len(self.tools_added) - 5} more")

        if self.files_observed_diff:
            added = sum(1 for d in self.files_observed_diff if d.diff_type == DiffType.ADDED)
            removed = sum(1 for d in self.files_observed_diff if d.diff_type == DiffType.REMOVED)
            lines.append(f"  Files observed: +{added} / -{removed}")

        if self.files_modified_diff:
            added = sum(1 for d in self.files_modified_diff if d.diff_type == DiffType.ADDED)
            removed = sum(1 for d in self.files_modified_diff if d.diff_type == DiffType.REMOVED)
            lines.append(f"  Files modified: +{added} / -{removed}")

        return "\n".join(lines)


@dataclass
class CheckpointData:
    """Complete checkpoint data including metadata and serialized state.

    Attributes:
        metadata: Checkpoint metadata
        state_data: Serialized conversation state (JSON-compatible dict)
        compressed: Whether state_data is compressed
        checksum: Optional integrity checksum
    """

    metadata: CheckpointMetadata
    state_data: Dict[str, Any]
    compressed: bool = False
    checksum: Optional[str] = None


@runtime_checkable
class CheckpointManagerProtocol(Protocol):
    """Protocol for checkpoint persistence backends.

    Implementations provide storage for conversation state checkpoints,
    enabling time-travel debugging across different storage backends
    (SQLite, PostgreSQL, in-memory, etc.).
    """

    async def save_checkpoint(
        self,
        session_id: str,
        state_data: Dict[str, Any],
        metadata: CheckpointMetadata,
    ) -> str:
        """Save a checkpoint and return its ID.

        Args:
            session_id: Session identifier
            state_data: Serialized conversation state
            metadata: Checkpoint metadata

        Returns:
            Checkpoint ID

        Raises:
            CheckpointStorageError: If save fails
        """
        ...

    async def load_checkpoint(self, checkpoint_id: str) -> CheckpointData:
        """Load a checkpoint by ID.

        Args:
            checkpoint_id: ID of checkpoint to load

        Returns:
            CheckpointData with metadata and state

        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
            CheckpointStorageError: If load fails
        """
        ...

    async def list_checkpoints(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[CheckpointMetadata]:
        """List checkpoints for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number to return
            offset: Number to skip for pagination

        Returns:
            List of checkpoint metadata, ordered by timestamp descending
        """
        ...

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    async def get_checkpoint_metadata(self, checkpoint_id: str) -> CheckpointMetadata:
        """Get metadata for a checkpoint without loading state.

        Args:
            checkpoint_id: ID of checkpoint

        Returns:
            Checkpoint metadata

        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
        """
        ...

    async def cleanup_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
    ) -> int:
        """Remove old checkpoints, keeping the N most recent.

        Args:
            session_id: Session identifier
            keep_count: Number of recent checkpoints to keep

        Returns:
            Number of checkpoints removed
        """
        ...
