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

"""In-memory checkpoint storage backend for testing."""

import copy
from datetime import datetime
from typing import Any, Dict, List

from victor.storage.checkpoints.protocol import (
    CheckpointData,
    CheckpointManagerProtocol,
    CheckpointMetadata,
    CheckpointNotFoundError,
)


class MemoryCheckpointBackend(CheckpointManagerProtocol):
    """In-memory backend for checkpoint storage.

    Useful for testing and ephemeral sessions. Data is lost when
    the process exits.
    """

    def __init__(self):
        """Initialize empty storage."""
        self._checkpoints: Dict[str, CheckpointData] = {}
        self._by_session: Dict[str, List[str]] = {}

    async def save_checkpoint(
        self,
        session_id: str,
        state_data: Dict[str, Any],
        metadata: CheckpointMetadata,
    ) -> str:
        """Save checkpoint to memory."""
        checkpoint_id = metadata.checkpoint_id

        self._checkpoints[checkpoint_id] = CheckpointData(
            metadata=metadata,
            state_data=copy.deepcopy(state_data),
            compressed=False,
            checksum=None,
        )

        if session_id not in self._by_session:
            self._by_session[session_id] = []
        self._by_session[session_id].append(checkpoint_id)

        return checkpoint_id

    async def load_checkpoint(self, checkpoint_id: str) -> CheckpointData:
        """Load checkpoint from memory."""
        if checkpoint_id not in self._checkpoints:
            raise CheckpointNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        data = self._checkpoints[checkpoint_id]
        return CheckpointData(
            metadata=data.metadata,
            state_data=copy.deepcopy(data.state_data),
            compressed=data.compressed,
            checksum=data.checksum,
        )

    async def list_checkpoints(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[CheckpointMetadata]:
        """List checkpoints for session."""
        if session_id not in self._by_session:
            return []

        checkpoint_ids = self._by_session[session_id]
        checkpoints = [
            self._checkpoints[cid].metadata for cid in checkpoint_ids if cid in self._checkpoints
        ]

        # Sort by timestamp descending
        checkpoints.sort(key=lambda m: m.timestamp, reverse=True)

        return checkpoints[offset : offset + limit]

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from memory."""
        if checkpoint_id not in self._checkpoints:
            return False

        data = self._checkpoints.pop(checkpoint_id)
        session_id = data.metadata.session_id

        if session_id in self._by_session:
            try:
                self._by_session[session_id].remove(checkpoint_id)
            except ValueError:
                pass

        return True

    async def get_checkpoint_metadata(self, checkpoint_id: str) -> CheckpointMetadata:
        """Get checkpoint metadata."""
        if checkpoint_id not in self._checkpoints:
            raise CheckpointNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        return self._checkpoints[checkpoint_id].metadata

    async def cleanup_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
    ) -> int:
        """Remove old checkpoints."""
        if session_id not in self._by_session:
            return 0

        checkpoints = await self.list_checkpoints(session_id, limit=1000)

        if len(checkpoints) <= keep_count:
            return 0

        to_delete = checkpoints[keep_count:]
        deleted = 0

        for metadata in to_delete:
            if await self.delete_checkpoint(metadata.checkpoint_id):
                deleted += 1

        return deleted

    def clear(self) -> None:
        """Clear all checkpoints (testing utility)."""
        self._checkpoints.clear()
        self._by_session.clear()
