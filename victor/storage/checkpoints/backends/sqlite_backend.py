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

"""SQLite-based checkpoint storage backend.

Provides embedded, persistent storage for conversation state checkpoints
with support for large states via compression.
"""

try:
    import aiosqlite
except ImportError:  # optional dependency — provided by victor-ai[checkpoints]
    aiosqlite = None  # type: ignore[assignment]
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.storage.checkpoints.protocol import (
    CheckpointData,
    CheckpointManagerProtocol,
    CheckpointMetadata,
    CheckpointNotFoundError,
    CheckpointStorageError,
)
from victor.storage.checkpoints.state_serializer import (
    serialize_conversation_state,
    deserialize_conversation_state,
)

logger = logging.getLogger(__name__)


# SQL schema for checkpoint storage
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    stage TEXT NOT NULL,
    tool_count INTEGER NOT NULL,
    message_count INTEGER NOT NULL,
    parent_id TEXT,
    description TEXT,
    tags TEXT,
    version INTEGER DEFAULT 1,
    state_data TEXT NOT NULL,
    state_compressed INTEGER DEFAULT 0,
    state_checksum TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_session_timestamp
ON checkpoints (session_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_parent_id
ON checkpoints (parent_id);
"""


class SQLiteCheckpointBackend(CheckpointManagerProtocol):
    """SQLite backend for checkpoint storage.

    Provides embedded persistence for conversation state checkpoints,
    suitable for single-user CLI applications and development.

    Features:
    - Automatic schema creation
    - Compression for large states
    - Pagination for checkpoint listing
    - Automatic cleanup of old checkpoints
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        db_name: str = "checkpoints.db",
    ):
        """Initialize SQLite backend.

        Args:
            storage_path: Directory for database file (default: ~/.victor/)
            db_name: Database filename

        Raises:
            RuntimeError: If the optional ``aiosqlite`` dependency is missing.
        """
        if aiosqlite is None:
            raise RuntimeError(
                "Conversation checkpoints require the optional 'aiosqlite' package. "
                "Install it with `pip install victor-ai[checkpoints]` "
                "(or `pip install aiosqlite`)."
            )
        if storage_path is None:
            storage_path = Path.home() / ".victor"

        storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = storage_path / db_name
        self._initialized = False

        logger.debug(f"SQLite checkpoint backend: {self.db_path}")

    async def _ensure_initialized(self) -> None:
        """Ensure database schema is created."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()

        self._initialized = True
        logger.debug("Checkpoint database initialized")

    async def save_checkpoint(
        self,
        session_id: str,
        state_data: Dict[str, Any],
        metadata: CheckpointMetadata,
    ) -> str:
        """Save a checkpoint to SQLite.

        Args:
            session_id: Session identifier
            state_data: Serialized conversation state
            metadata: Checkpoint metadata

        Returns:
            Checkpoint ID

        Raises:
            CheckpointStorageError: If save fails
        """
        await self._ensure_initialized()

        try:
            # Serialize state with optional compression
            serialized = serialize_conversation_state(state_data, compress=True)

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO checkpoints (
                        checkpoint_id, session_id, timestamp, stage,
                        tool_count, message_count, parent_id, description,
                        tags, version, state_data, state_compressed, state_checksum
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metadata.checkpoint_id,
                        session_id,
                        metadata.timestamp.isoformat(),
                        metadata.stage,
                        metadata.tool_count,
                        metadata.message_count,
                        metadata.parent_id,
                        metadata.description,
                        json.dumps(metadata.tags),
                        metadata.version,
                        (
                            json.dumps(serialized["data"])
                            if not serialized["compressed"]
                            else serialized["data"]
                        ),
                        1 if serialized["compressed"] else 0,
                        serialized["checksum"],
                    ),
                )
                await db.commit()

            logger.info(
                f"Saved checkpoint {metadata.checkpoint_id} "
                f"(session={session_id}, tools={metadata.tool_count})"
            )
            return metadata.checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise CheckpointStorageError(f"Failed to save checkpoint: {e}") from e

    async def load_checkpoint(self, checkpoint_id: str) -> CheckpointData:
        """Load a checkpoint from SQLite.

        Args:
            checkpoint_id: ID of checkpoint to load

        Returns:
            CheckpointData with metadata and state

        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
            CheckpointStorageError: If load fails
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT * FROM checkpoints WHERE checkpoint_id = ?
                    """,
                    (checkpoint_id,),
                )
                row = await cursor.fetchone()

                if not row:
                    raise CheckpointNotFoundError(f"Checkpoint not found: {checkpoint_id}")

                # Reconstruct metadata
                metadata = CheckpointMetadata(
                    checkpoint_id=row["checkpoint_id"],
                    session_id=row["session_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    stage=row["stage"],
                    tool_count=row["tool_count"],
                    message_count=row["message_count"],
                    parent_id=row["parent_id"],
                    description=row["description"],
                    tags=json.loads(row["tags"]) if row["tags"] else [],
                    version=row["version"],
                )

                # Deserialize state
                compressed = bool(row["state_compressed"])
                if compressed:
                    stored = {
                        "data": row["state_data"],
                        "compressed": True,
                        "checksum": row["state_checksum"],
                        "version": row["version"],
                    }
                else:
                    stored = {
                        "data": json.loads(row["state_data"]),
                        "compressed": False,
                        "checksum": row["state_checksum"],
                        "version": row["version"],
                    }

                state_data = deserialize_conversation_state(stored)

                return CheckpointData(
                    metadata=metadata,
                    state_data=state_data,
                    compressed=compressed,
                    checksum=row["state_checksum"],
                )

        except CheckpointNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise CheckpointStorageError(f"Failed to load checkpoint: {e}") from e

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
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT checkpoint_id, session_id, timestamp, stage,
                           tool_count, message_count, parent_id, description,
                           tags, version
                    FROM checkpoints
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (session_id, limit, offset),
                )
                rows = await cursor.fetchall()

                return [
                    CheckpointMetadata(
                        checkpoint_id=row["checkpoint_id"],
                        session_id=row["session_id"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        stage=row["stage"],
                        tool_count=row["tool_count"],
                        message_count=row["message_count"],
                        parent_id=row["parent_id"],
                        description=row["description"],
                        tags=json.loads(row["tags"]) if row["tags"] else [],
                        version=row["version"],
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                )
                await db.commit()
                deleted = cursor.rowcount > 0

                if deleted:
                    logger.info(f"Deleted checkpoint {checkpoint_id}")

                return deleted

        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False

    async def get_checkpoint_metadata(self, checkpoint_id: str) -> CheckpointMetadata:
        """Get metadata for a checkpoint without loading state.

        Args:
            checkpoint_id: ID of checkpoint

        Returns:
            Checkpoint metadata

        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT checkpoint_id, session_id, timestamp, stage,
                           tool_count, message_count, parent_id, description,
                           tags, version
                    FROM checkpoints
                    WHERE checkpoint_id = ?
                    """,
                    (checkpoint_id,),
                )
                row = await cursor.fetchone()

                if not row:
                    raise CheckpointNotFoundError(f"Checkpoint not found: {checkpoint_id}")

                return CheckpointMetadata(
                    checkpoint_id=row["checkpoint_id"],
                    session_id=row["session_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    stage=row["stage"],
                    tool_count=row["tool_count"],
                    message_count=row["message_count"],
                    parent_id=row["parent_id"],
                    description=row["description"],
                    tags=json.loads(row["tags"]) if row["tags"] else [],
                    version=row["version"],
                )

        except CheckpointNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get checkpoint metadata: {e}")
            raise CheckpointStorageError(f"Failed to get metadata: {e}") from e

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
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get IDs to keep
                cursor = await db.execute(
                    """
                    SELECT checkpoint_id FROM checkpoints
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, keep_count),
                )
                keep_ids = [row[0] for row in await cursor.fetchall()]

                if not keep_ids:
                    return 0

                # Delete older checkpoints
                placeholders = ",".join("?" * len(keep_ids))
                cursor = await db.execute(
                    f"""
                    DELETE FROM checkpoints
                    WHERE session_id = ?
                    AND checkpoint_id NOT IN ({placeholders})
                    """,
                    (session_id, *keep_ids),
                )
                await db.commit()
                deleted = cursor.rowcount

                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old checkpoints for session {session_id}")

                return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")
            return 0

    async def get_checkpoint_count(self, session_id: str) -> int:
        """Get total number of checkpoints for a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of checkpoints
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE session_id = ?",
                    (session_id,),
                )
                row = await cursor.fetchone()
                return row[0] if row else 0

        except Exception as e:
            logger.error(f"Failed to count checkpoints: {e}")
            return 0

    async def get_all_sessions(self) -> List[str]:
        """Get all session IDs with checkpoints.

        Returns:
            List of session IDs
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "SELECT DISTINCT session_id FROM checkpoints ORDER BY session_id"
                )
                rows = await cursor.fetchall()
                return [row[0] for row in rows]

        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            return []
