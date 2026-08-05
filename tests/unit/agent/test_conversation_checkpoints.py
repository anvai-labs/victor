# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for conversation state checkpoint system.

This tests the new time-travel debugging checkpoints (victor/checkpoints/),
not the git-based checkpoints in victor/agent/checkpoints.py.
"""

import pytest
from datetime import datetime, timezone

from victor.storage.checkpoints.protocol import (
    CheckpointMetadata,
    CheckpointNotFoundError,
)
from victor.storage.checkpoints.state_serializer import (
    StateSerializer,
    serialize_conversation_state,
    deserialize_conversation_state,
)
from victor.storage.checkpoints.backends.memory_backend import MemoryCheckpointBackend
from victor.storage.checkpoints.manager import ConversationCheckpointManager


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata."""

    def test_create_metadata(self):
        """Test creating checkpoint metadata."""
        metadata = CheckpointMetadata.create(
            session_id="session_123",
            stage="READING",
            tool_count=5,
            message_count=10,
            description="Test checkpoint",
        )

        assert metadata.checkpoint_id.startswith("ckpt_")
        assert metadata.session_id == "session_123"
        assert metadata.stage == "READING"
        assert metadata.tool_count == 5
        assert metadata.message_count == 10
        assert metadata.description == "Test checkpoint"
        assert isinstance(metadata.timestamp, datetime)

    def test_metadata_serialization(self):
        """Test metadata to_dict and from_dict."""
        metadata = CheckpointMetadata.create(
            session_id="session_456",
            stage="EXECUTION",
            tool_count=15,
            message_count=25,
            tags=["auto", "important"],
        )

        data = metadata.to_dict()
        restored = CheckpointMetadata.from_dict(data)

        assert restored.checkpoint_id == metadata.checkpoint_id
        assert restored.session_id == metadata.session_id
        assert restored.stage == metadata.stage
        assert restored.tool_count == metadata.tool_count
        assert restored.tags == metadata.tags


class TestStateSerializer:
    """Tests for state serialization."""

    def test_serialize_simple_state(self):
        """Test serializing simple state."""
        serializer = StateSerializer(compress=False)
        state = {
            "stage": "READING",
            "tool_history": ["read_file", "search_files"],
            "message_count": 5,
        }

        result = serializer.serialize(state)
        assert result.checksum
        assert not result.compressed

        restored = serializer.deserialize(result)
        assert restored == state

    def test_serialize_with_datetime(self):
        """Test serializing state with datetime."""
        serializer = StateSerializer(compress=False)
        now = datetime.now(timezone.utc)
        state = {
            "timestamp": now,
            "data": "test",
        }

        result = serializer.serialize(state)
        restored = serializer.deserialize(result)

        assert restored["timestamp"] == now
        assert restored["data"] == "test"

    def test_serialize_with_sets(self):
        """Test serializing state with sets."""
        serializer = StateSerializer(compress=False)
        state = {
            "observed_files": {"file1.py", "file2.py"},
            "modified_files": {"file3.py"},
        }

        result = serializer.serialize(state)
        restored = serializer.deserialize(result)

        assert restored["observed_files"] == {"file1.py", "file2.py"}
        assert restored["modified_files"] == {"file3.py"}

    def test_compression(self):
        """Test that large states get compressed."""
        serializer = StateSerializer(compress=True, compression_threshold=100)
        # Create a state larger than threshold
        state = {
            "data": "x" * 1000,
            "more_data": list(range(100)),
        }

        result = serializer.serialize(state)
        assert result.compressed

        restored = serializer.deserialize(result)
        assert restored["data"] == state["data"]


class TestMemoryCheckpointBackend:
    """Tests for in-memory checkpoint backend."""

    @pytest.fixture
    def backend(self):
        return MemoryCheckpointBackend()

    @pytest.mark.asyncio
    async def test_save_and_load(self, backend):
        """Test saving and loading checkpoints."""
        metadata = CheckpointMetadata.create(
            session_id="session_1",
            stage="READING",
            tool_count=5,
            message_count=10,
        )
        state_data = {"test": "data", "count": 42}

        # Save
        checkpoint_id = await backend.save_checkpoint(
            session_id="session_1",
            state_data=state_data,
            metadata=metadata,
        )

        assert checkpoint_id == metadata.checkpoint_id

        # Load
        loaded = await backend.load_checkpoint(checkpoint_id)
        assert loaded.metadata.checkpoint_id == checkpoint_id
        assert loaded.state_data["test"] == "data"
        assert loaded.state_data["count"] == 42

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, backend):
        """Test listing checkpoints for a session."""
        # Create multiple checkpoints
        for i in range(5):
            metadata = CheckpointMetadata.create(
                session_id="session_list",
                stage="READING",
                tool_count=i,
                message_count=i * 2,
            )
            await backend.save_checkpoint(
                session_id="session_list",
                state_data={"index": i},
                metadata=metadata,
            )

        checkpoints = await backend.list_checkpoints("session_list")
        assert len(checkpoints) == 5

        # Should be ordered by timestamp descending
        for cp in checkpoints:
            assert cp.session_id == "session_list"

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, backend):
        """Test deleting a checkpoint."""
        metadata = CheckpointMetadata.create(
            session_id="session_del",
            stage="READING",
            tool_count=1,
            message_count=2,
        )
        checkpoint_id = await backend.save_checkpoint(
            session_id="session_del",
            state_data={"test": "delete"},
            metadata=metadata,
        )

        # Delete
        deleted = await backend.delete_checkpoint(checkpoint_id)
        assert deleted

        # Try to load - should fail
        with pytest.raises(CheckpointNotFoundError):
            await backend.load_checkpoint(checkpoint_id)


class TestConversationCheckpointManager:
    """Tests for ConversationCheckpointManager."""

    @pytest.fixture
    def manager(self):
        backend = MemoryCheckpointBackend()
        return ConversationCheckpointManager(backend)

    @pytest.mark.asyncio
    async def test_save_and_restore(self, manager):
        """Test saving and restoring checkpoints."""
        state = {
            "stage": "READING",
            "tool_history": ["read_file", "search_files"],
            "observed_files": ["file1.py"],
            "modified_files": [],
            "message_count": 5,
        }

        # Save
        checkpoint_id = await manager.save_checkpoint(
            session_id="session_mgr",
            state=state,
            description="Test checkpoint",
        )

        # Restore
        restored = await manager.restore_checkpoint(checkpoint_id)
        assert restored["stage"] == "READING"
        assert restored["tool_history"] == ["read_file", "search_files"]
        assert restored["message_count"] == 5

    @pytest.mark.asyncio
    async def test_fork_from_checkpoint(self, manager):
        """Test forking a session from a checkpoint."""
        state = {
            "stage": "EXECUTION",
            "tool_history": ["read_file"],
            "message_count": 3,
        }

        checkpoint_id = await manager.save_checkpoint(
            session_id="original_session",
            state=state,
        )

        # Fork
        new_session_id, forked_state = await manager.fork_from_checkpoint(checkpoint_id)

        assert new_session_id.startswith("fork_")
        assert forked_state["stage"] == "EXECUTION"
        assert forked_state["tool_history"] == ["read_file"]

        # Verify fork created its own checkpoint
        fork_checkpoints = await manager.list_checkpoints(new_session_id)
        assert len(fork_checkpoints) >= 1
        assert any("Fork" in (cp.description or "") for cp in fork_checkpoints)

    @pytest.mark.asyncio
    async def test_diff_checkpoints(self, manager):
        """Test comparing two checkpoints."""
        state1 = {
            "stage": "READING",
            "tool_history": ["read_file"],
            "observed_files": ["file1.py"],
            "modified_files": [],
            "message_count": 2,
        }

        state2 = {
            "stage": "EXECUTION",
            "tool_history": ["read_file", "write_file"],
            "observed_files": ["file1.py", "file2.py"],
            "modified_files": ["file1.py"],
            "message_count": 5,
        }

        cp1 = await manager.save_checkpoint("session_diff", state1)
        cp2 = await manager.save_checkpoint("session_diff", state2)

        diff = await manager.diff_checkpoints(cp1, cp2)

        assert diff.checkpoint_a == cp1
        assert diff.checkpoint_b == cp2
        assert diff.messages_added == 3
        assert "write_file" in diff.tools_added

    @pytest.mark.asyncio
    async def test_auto_checkpoint(self, manager):
        """Test auto-checkpointing based on tool count."""
        manager.auto_checkpoint_interval = 3

        # Not enough tools - no checkpoint
        state1 = {"tool_history": ["a", "b"], "message_count": 1, "stage": "READING"}
        result = await manager.maybe_auto_checkpoint("session_auto", state1)
        assert result is None

        # Enough tools - should checkpoint
        state2 = {
            "tool_history": ["a", "b", "c", "d"],
            "message_count": 2,
            "stage": "READING",
        }
        result = await manager.maybe_auto_checkpoint("session_auto", state2)
        assert result is not None

        # Verify checkpoint was created
        checkpoints = await manager.list_checkpoints("session_auto")
        assert len(checkpoints) == 1

    @pytest.mark.asyncio
    async def test_timeline(self, manager):
        """Test getting checkpoint timeline."""
        for i in range(3):
            state = {
                "stage": f"STAGE_{i}",
                "tool_history": [f"tool_{j}" for j in range(i + 1)],
                "message_count": i + 1,
            }
            await manager.save_checkpoint(
                "session_timeline",
                state,
                description=f"Checkpoint {i}",
            )

        timeline = await manager.get_timeline("session_timeline")
        assert len(timeline) == 3

        # Verify timeline entries have expected fields
        for entry in timeline:
            assert "id" in entry
            assert "timestamp" in entry
            assert "stage" in entry
            assert "description" in entry
