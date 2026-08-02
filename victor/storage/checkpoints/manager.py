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

"""High-level checkpoint management API.

Provides a convenient interface for time-travel debugging operations
including save, restore, fork, and diff capabilities.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from victor.storage.checkpoints.protocol import (
    CheckpointData,
    CheckpointDiff,
    CheckpointManagerProtocol,
    CheckpointMetadata,
    CheckpointNotFoundError,
    DiffType,
    FieldDiff,
)

logger = logging.getLogger(__name__)


class ConversationCheckpointManager:
    """High-level API for conversation state checkpointing.

    Wraps a storage backend (SQLite, memory, etc.) with convenient
    methods for time-travel debugging:

    - save_checkpoint: Create a named checkpoint
    - load_checkpoint: Restore state from checkpoint
    - fork_from_checkpoint: Create new session from checkpoint
    - diff_checkpoints: Compare two checkpoints
    - auto_checkpoint: Automatic checkpointing based on tool count

    Note: Renamed from CheckpointManager to ConversationCheckpointManager to be
    semantically distinct from:
    - GitCheckpointManager (victor.agent.checkpoints): Git stash-based checkpoints
    - GraphCheckpointManager (victor.framework.graph): Graph state checkpoints

    Usage:
        from victor.storage.checkpoints import ConversationCheckpointManager, SQLiteCheckpointBackend

        backend = SQLiteCheckpointBackend()
        manager = ConversationCheckpointManager(backend)

        # Manual checkpoint
        cp_id = await manager.save_checkpoint(session_id, state, "before refactor")

        # Later, restore
        state = await manager.restore_checkpoint(cp_id)

        # Or fork for experimentation
        new_session, new_state = await manager.fork_from_checkpoint(cp_id)
    """

    def __init__(
        self,
        backend: CheckpointManagerProtocol,
        auto_checkpoint_interval: int = 5,
        max_checkpoints_per_session: int = 50,
    ):
        """Initialize the checkpoint manager.

        Args:
            backend: Storage backend implementing CheckpointManagerProtocol
            auto_checkpoint_interval: Tool executions between auto-checkpoints
            max_checkpoints_per_session: Maximum checkpoints to keep per session
        """
        self.backend = backend
        self.auto_checkpoint_interval = auto_checkpoint_interval
        self.max_checkpoints_per_session = max_checkpoints_per_session

        # Track tool count for auto-checkpointing
        self._tool_counts: Dict[str, int] = {}
        self._last_auto_checkpoint: Dict[str, int] = {}

    async def save_checkpoint(
        self,
        session_id: str,
        state: Dict[str, Any],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Save a checkpoint of the current conversation state.

        Args:
            session_id: Session identifier
            state: Conversation state dictionary (from ConversationState.to_dict())
            description: Optional human-readable description
            tags: Optional tags for categorization
            parent_id: Optional parent checkpoint ID (for forks)

        Returns:
            Checkpoint ID
        """
        # Extract metadata from state
        stage = state.get("stage", "INITIAL")
        tool_count = len(state.get("tool_history", []))
        message_count = state.get("message_count", 0)

        metadata = CheckpointMetadata.create(
            session_id=session_id,
            stage=stage,
            tool_count=tool_count,
            message_count=message_count,
            parent_id=parent_id,
            description=description,
            tags=tags,
        )

        checkpoint_id = await self.backend.save_checkpoint(
            session_id=session_id,
            state_data=state,
            metadata=metadata,
        )

        # Cleanup old checkpoints if needed
        await self.backend.cleanup_old_checkpoints(session_id, self.max_checkpoints_per_session)

        logger.info(
            f"Checkpoint saved: {checkpoint_id[:20]}... " f"({description or 'no description'})"
        )

        return checkpoint_id

    async def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore conversation state from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            Restored conversation state dictionary

        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
        """
        data = await self.backend.load_checkpoint(checkpoint_id)

        logger.info(
            f"Restored checkpoint: {checkpoint_id[:20]}... "
            f"(stage={data.metadata.stage}, tools={data.metadata.tool_count})"
        )

        return data.state_data

    async def fork_from_checkpoint(
        self,
        checkpoint_id: str,
        new_session_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Create a new session fork from a checkpoint.

        This allows exploring alternative paths from a previous state
        without affecting the original session.

        Args:
            checkpoint_id: ID of checkpoint to fork from
            new_session_id: Optional ID for new session (auto-generated if None)

        Returns:
            Tuple of (new_session_id, forked_state)

        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
        """
        # Load the checkpoint
        data = await self.backend.load_checkpoint(checkpoint_id)

        # Generate new session ID if not provided
        if new_session_id is None:
            new_session_id = f"fork_{uuid.uuid4().hex[:12]}"

        # Create a copy of the state for the new session
        forked_state = data.state_data.copy()

        # Save initial checkpoint for the fork
        await self.save_checkpoint(
            session_id=new_session_id,
            state=forked_state,
            description=f"Fork from {checkpoint_id[:12]}...",
            parent_id=checkpoint_id,
            tags=["fork", f"from:{data.metadata.session_id}"],
        )

        logger.info(f"Forked session {new_session_id} from checkpoint {checkpoint_id[:20]}...")

        return new_session_id, forked_state

    async def diff_checkpoints(
        self,
        checkpoint_a: str,
        checkpoint_b: str,
    ) -> CheckpointDiff:
        """Compare two checkpoints and return their differences.

        Args:
            checkpoint_a: ID of first checkpoint (older)
            checkpoint_b: ID of second checkpoint (newer)

        Returns:
            CheckpointDiff with detailed differences

        Raises:
            CheckpointNotFoundError: If either checkpoint doesn't exist
        """
        data_a = await self.backend.load_checkpoint(checkpoint_a)
        data_b = await self.backend.load_checkpoint(checkpoint_b)

        state_a = data_a.state_data
        state_b = data_b.state_data

        diff = CheckpointDiff(
            checkpoint_a=checkpoint_a,
            checkpoint_b=checkpoint_b,
        )

        # Compare metadata
        if data_a.metadata.stage != data_b.metadata.stage:
            diff.metadata_diff.append(
                FieldDiff(
                    field_name="stage",
                    diff_type=DiffType.MODIFIED,
                    old_value=data_a.metadata.stage,
                    new_value=data_b.metadata.stage,
                )
            )
            diff.stage_changes = [data_a.metadata.stage, data_b.metadata.stage]

        # Compare messages
        diff.messages_added = max(0, data_b.metadata.message_count - data_a.metadata.message_count)
        diff.messages_removed = max(
            0, data_a.metadata.message_count - data_b.metadata.message_count
        )

        # Compare tool history
        tools_a = set(state_a.get("tool_history", []))
        tools_b = set(state_b.get("tool_history", []))
        diff.tools_added = list(tools_b - tools_a)
        diff.tools_removed = list(tools_a - tools_b)

        # Compare observed files
        files_a = set(state_a.get("observed_files", []))
        files_b = set(state_b.get("observed_files", []))
        self._compute_set_diff(files_a, files_b, "observed_file", diff.files_observed_diff)

        # Compare modified files
        mod_a = set(state_a.get("modified_files", []))
        mod_b = set(state_b.get("modified_files", []))
        self._compute_set_diff(mod_a, mod_b, "modified_file", diff.files_modified_diff)

        return diff

    def _compute_set_diff(
        self,
        set_a: Set[str],
        set_b: Set[str],
        field_prefix: str,
        diff_list: List[FieldDiff],
    ) -> None:
        """Compute differences between two sets."""
        for item in set_b - set_a:
            diff_list.append(
                FieldDiff(
                    field_name=f"{field_prefix}:{item}",
                    diff_type=DiffType.ADDED,
                    new_value=item,
                )
            )
        for item in set_a - set_b:
            diff_list.append(
                FieldDiff(
                    field_name=f"{field_prefix}:{item}",
                    diff_type=DiffType.REMOVED,
                    old_value=item,
                )
            )

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
        return await self.backend.list_checkpoints(session_id, limit, offset)

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        return await self.backend.delete_checkpoint(checkpoint_id)

    async def get_checkpoint_metadata(self, checkpoint_id: str) -> CheckpointMetadata:
        """Get metadata for a checkpoint without loading full state.

        Args:
            checkpoint_id: ID of checkpoint

        Returns:
            Checkpoint metadata

        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
        """
        return await self.backend.get_checkpoint_metadata(checkpoint_id)

    async def maybe_auto_checkpoint(
        self,
        session_id: str,
        state: Dict[str, Any],
        force: bool = False,
    ) -> Optional[str]:
        """Create an auto-checkpoint if enough tools have been executed.

        Args:
            session_id: Session identifier
            state: Current conversation state
            force: Force checkpoint regardless of interval

        Returns:
            Checkpoint ID if created, None otherwise
        """
        tool_count = len(state.get("tool_history", []))

        # Initialize tracking if needed
        if session_id not in self._tool_counts:
            self._tool_counts[session_id] = 0
            self._last_auto_checkpoint[session_id] = 0

        # Check if we should checkpoint
        tools_since_last = tool_count - self._last_auto_checkpoint[session_id]

        if force or tools_since_last >= self.auto_checkpoint_interval:
            checkpoint_id = await self.save_checkpoint(
                session_id=session_id,
                state=state,
                description=f"Auto-checkpoint at {tool_count} tools",
                tags=["auto"],
            )
            self._last_auto_checkpoint[session_id] = tool_count
            return checkpoint_id

        return None

    async def get_timeline(
        self,
        session_id: str,
        include_forks: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get a timeline view of checkpoints for visualization.

        Args:
            session_id: Session identifier
            include_forks: Whether to include forked sessions

        Returns:
            List of checkpoint timeline entries
        """
        checkpoints = await self.list_checkpoints(session_id, limit=100)

        timeline = []
        for cp in checkpoints:
            entry = {
                "id": cp.checkpoint_id,
                "timestamp": cp.timestamp.isoformat(),
                "stage": cp.stage,
                "tool_count": cp.tool_count,
                "message_count": cp.message_count,
                "description": cp.description or "",
                "tags": cp.tags,
                "is_fork": cp.parent_id is not None,
                "parent_id": cp.parent_id,
            }
            timeline.append(entry)

        return timeline

    def format_timeline_ascii(self, timeline: List[Dict[str, Any]]) -> str:
        """Format timeline as ASCII art for CLI display.

        Args:
            timeline: Timeline from get_timeline()

        Returns:
            ASCII art representation
        """
        if not timeline:
            return "No checkpoints found."

        lines = ["Checkpoint Timeline", "=" * 60]

        for i, entry in enumerate(timeline):
            prefix = "├─" if i < len(timeline) - 1 else "└─"
            fork_marker = " (fork)" if entry.get("is_fork") else ""

            lines.append(
                f"{prefix} [{entry['id'][:8]}...] " f"{entry['timestamp'][:19]}{fork_marker}"
            )
            lines.append(
                f"│  Stage: {entry['stage']}, "
                f"Tools: {entry['tool_count']}, "
                f"Messages: {entry['message_count']}"
            )
            if entry.get("description"):
                lines.append(f"│  {entry['description']}")
            lines.append("│")

        return "\n".join(lines)
