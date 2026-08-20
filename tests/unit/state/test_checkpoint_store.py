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

"""Unit tests for CheckpointStore.

Tests the policy checkpoint storage and versioning system.
"""

import gzip
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from victor.framework.rl.checkpoint_store import (
    CheckpointStore,
    PolicyCheckpoint,
    CheckpointDiff,
    get_checkpoint_store,
)


@pytest.fixture
def temp_storage_path() -> Path:
    """Fixture for temporary storage path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "checkpoints"


@pytest.fixture
def store(temp_storage_path: Path) -> CheckpointStore:
    """Fixture for CheckpointStore with temp storage."""
    return CheckpointStore(storage_path=temp_storage_path)


class TestPolicyCheckpoint:
    """Tests for PolicyCheckpoint dataclass."""

    def test_checkpoint_creation(self) -> None:
        """Test creating a checkpoint."""
        checkpoint = PolicyCheckpoint(
            checkpoint_id="ckpt_test_1",
            learner_name="tool_selector",
            version="v1.0.0",
            state={"q_values": {"ctx1": 0.7}},
            metadata={"success_rate": 0.85},
            tags=["production"],
        )

        assert checkpoint.checkpoint_id == "ckpt_test_1"
        assert checkpoint.learner_name == "tool_selector"
        assert checkpoint.version == "v1.0.0"
        assert checkpoint.state["q_values"]["ctx1"] == 0.7

    def test_checkpoint_state_hash(self) -> None:
        """Test state hash computation."""
        checkpoint1 = PolicyCheckpoint(
            checkpoint_id="ckpt1",
            learner_name="test",
            version="v1",
            state={"q": 0.5},
        )
        checkpoint2 = PolicyCheckpoint(
            checkpoint_id="ckpt2",
            learner_name="test",
            version="v2",
            state={"q": 0.5},
        )
        checkpoint3 = PolicyCheckpoint(
            checkpoint_id="ckpt3",
            learner_name="test",
            version="v3",
            state={"q": 0.6},
        )

        # Same state = same hash
        assert checkpoint1.state_hash == checkpoint2.state_hash
        # Different state = different hash
        assert checkpoint1.state_hash != checkpoint3.state_hash

    def test_checkpoint_to_dict(self) -> None:
        """Test converting checkpoint to dictionary."""
        checkpoint = PolicyCheckpoint(
            checkpoint_id="ckpt_1",
            learner_name="test",
            version="v1.0",
            state={"key": "value"},
            tags=["tag1"],
        )

        data = checkpoint.to_dict()

        assert data["checkpoint_id"] == "ckpt_1"
        assert data["learner_name"] == "test"
        assert data["state"] == {"key": "value"}
        assert data["tags"] == ["tag1"]
        assert "state_hash" in data

    def test_checkpoint_from_dict(self) -> None:
        """Test creating checkpoint from dictionary."""
        data = {
            "checkpoint_id": "ckpt_1",
            "learner_name": "test",
            "version": "v1.0",
            "state": {"key": "value"},
            "metadata": {"metric": 0.5},
            "tags": ["tag1"],
        }

        checkpoint = PolicyCheckpoint.from_dict(data)

        assert checkpoint.checkpoint_id == "ckpt_1"
        assert checkpoint.version == "v1.0"
        assert checkpoint.metadata["metric"] == 0.5

    def test_checkpoint_defaults(self) -> None:
        """Test default values for checkpoint."""
        checkpoint = PolicyCheckpoint(
            checkpoint_id="ckpt",
            learner_name="test",
            version="v1",
            state={},
        )

        assert checkpoint.metadata == {}
        assert checkpoint.parent_id is None
        assert checkpoint.tags == []
        assert checkpoint.timestamp is not None


class TestCheckpointDiff:
    """Tests for CheckpointDiff dataclass."""

    def test_diff_no_changes(self) -> None:
        """Test diff with no changes."""
        diff = CheckpointDiff(
            from_version="v1",
            to_version="v2",
            unchanged_keys=["key1", "key2"],
        )

        assert diff.has_changes is False
        assert diff.change_ratio == 0.0

    def test_diff_with_changes(self) -> None:
        """Test diff with changes."""
        diff = CheckpointDiff(
            from_version="v1",
            to_version="v2",
            added_keys=["new_key"],
            removed_keys=["old_key"],
            changed_keys=["modified"],
            unchanged_keys=["stable"],
        )

        assert diff.has_changes is True
        assert diff.change_ratio == 0.75  # 3 changed out of 4 total

    def test_diff_empty(self) -> None:
        """Test empty diff."""
        diff = CheckpointDiff(from_version="v1", to_version="v2")

        assert diff.has_changes is False
        assert diff.change_ratio == 0.0


class TestCheckpointStore:
    """Tests for CheckpointStore."""

    def test_initialization(self, store: CheckpointStore) -> None:
        """Test store initialization."""
        assert store.storage_path.exists()
        assert store._cache == {}
        assert store._next_id == 1

    def test_create_checkpoint(self, store: CheckpointStore) -> None:
        """Test creating a checkpoint."""
        checkpoint = store.create_checkpoint(
            learner_name="tool_selector",
            version="v1.0.0",
            state={"q_values": {"ctx1": 0.7}},
            metadata={"success_rate": 0.85},
        )

        assert checkpoint.learner_name == "tool_selector"
        assert checkpoint.version == "v1.0.0"
        assert "ckpt_tool_selector_" in checkpoint.checkpoint_id

    def test_create_checkpoint_increments_id(self, store: CheckpointStore) -> None:
        """Test checkpoint ID increments."""
        cp1 = store.create_checkpoint("learner", "v1", {"state": 1})
        cp2 = store.create_checkpoint("learner", "v2", {"state": 2})

        assert cp1.checkpoint_id != cp2.checkpoint_id

    def test_create_checkpoint_with_parent(self, store: CheckpointStore) -> None:
        """Test creating checkpoint with parent."""
        cp1 = store.create_checkpoint("learner", "v1", {"state": 1})
        cp2 = store.create_checkpoint("learner", "v2", {"state": 2}, parent_id=cp1.checkpoint_id)

        assert cp2.parent_id == cp1.checkpoint_id

    def test_create_checkpoint_with_tags(self, store: CheckpointStore) -> None:
        """Test creating checkpoint with tags."""
        checkpoint = store.create_checkpoint(
            "learner", "v1", {"state": 1}, tags=["production", "stable"]
        )

        assert "production" in checkpoint.tags
        assert "stable" in checkpoint.tags

    def test_get_checkpoint(self, store: CheckpointStore) -> None:
        """Test getting a checkpoint."""
        store.create_checkpoint("learner", "v1.0.0", {"state": "test"})

        retrieved = store.get_checkpoint("learner", "v1.0.0")

        assert retrieved is not None
        assert retrieved.version == "v1.0.0"
        assert retrieved.state == {"state": "test"}

    def test_get_checkpoint_not_found(self, store: CheckpointStore) -> None:
        """Test getting non-existent checkpoint."""
        result = store.get_checkpoint("unknown", "v1")
        assert result is None

    def test_get_latest_checkpoint(self, store: CheckpointStore) -> None:
        """Test getting latest checkpoint."""
        store.create_checkpoint("learner", "v1", {"v": 1})
        store.create_checkpoint("learner", "v2", {"v": 2})
        store.create_checkpoint("learner", "v3", {"v": 3})

        latest = store.get_latest_checkpoint("learner")

        assert latest is not None
        assert latest.version == "v3"

    def test_get_latest_checkpoint_empty(self, store: CheckpointStore) -> None:
        """Test getting latest checkpoint when none exist."""
        result = store.get_latest_checkpoint("unknown")
        assert result is None

    def test_list_checkpoints_by_learner(self, store: CheckpointStore) -> None:
        """Test listing checkpoints by learner."""
        store.create_checkpoint("learner1", "v1", {})
        store.create_checkpoint("learner1", "v2", {})
        store.create_checkpoint("learner2", "v1", {})

        checkpoints = store.list_checkpoints("learner1")

        assert len(checkpoints) == 2
        assert all(cp.learner_name == "learner1" for cp in checkpoints)

    def test_list_checkpoints_all(self, store: CheckpointStore) -> None:
        """Test listing all checkpoints."""
        store.create_checkpoint("learner1", "v1", {})
        store.create_checkpoint("learner2", "v1", {})

        checkpoints = store.list_checkpoints()

        assert len(checkpoints) == 2

    def test_list_checkpoints_by_tag(self, store: CheckpointStore) -> None:
        """Test listing checkpoints by tag."""
        store.create_checkpoint("learner", "v1", {}, tags=["production"])
        store.create_checkpoint("learner", "v2", {}, tags=["staging"])
        store.create_checkpoint("learner", "v3", {}, tags=["production", "stable"])

        checkpoints = store.list_checkpoints(tag="production")

        assert len(checkpoints) == 2

    def test_list_checkpoints_sorted_by_timestamp(self, store: CheckpointStore) -> None:
        """Test checkpoints are sorted by timestamp (newest first)."""
        store.create_checkpoint("learner", "v1", {})
        store.create_checkpoint("learner", "v2", {})
        store.create_checkpoint("learner", "v3", {})

        checkpoints = store.list_checkpoints("learner")

        # Newest first
        assert checkpoints[0].version == "v3"
        assert checkpoints[-1].version == "v1"

    def test_diff_checkpoints(self, store: CheckpointStore) -> None:
        """Test comparing checkpoints."""
        store.create_checkpoint("learner", "v1", {"a": 1, "b": 2, "c": 3})
        store.create_checkpoint("learner", "v2", {"a": 1, "b": 5, "d": 4})

        diff = store.diff_checkpoints("learner", "v1", "v2")

        assert diff is not None
        assert diff.from_version == "v1"
        assert diff.to_version == "v2"
        assert "d" in diff.added_keys
        assert "c" in diff.removed_keys
        assert "b" in diff.changed_keys
        assert "a" in diff.unchanged_keys

    def test_diff_checkpoints_nested(self, store: CheckpointStore) -> None:
        """Test comparing checkpoints with nested state."""
        store.create_checkpoint(
            "learner",
            "v1",
            {
                "q_values": {"ctx1": 0.5, "ctx2": 0.6},
                "config": {"lr": 0.01},
            },
        )
        store.create_checkpoint(
            "learner",
            "v2",
            {
                "q_values": {"ctx1": 0.7, "ctx2": 0.6},
                "config": {"lr": 0.01},
            },
        )

        diff = store.diff_checkpoints("learner", "v1", "v2")

        assert diff is not None
        assert "q_values.ctx1" in diff.changed_keys
        assert "q_values.ctx2" in diff.unchanged_keys

    def test_diff_checkpoints_not_found(self, store: CheckpointStore) -> None:
        """Test diff with non-existent checkpoint."""
        store.create_checkpoint("learner", "v1", {})

        result = store.diff_checkpoints("learner", "v1", "v999")
        assert result is None

    def test_tag_checkpoint(self, store: CheckpointStore) -> None:
        """Test adding tag to checkpoint."""
        store.create_checkpoint("learner", "v1", {})

        result = store.tag_checkpoint("learner", "v1", "production")

        assert result is True
        checkpoint = store.get_checkpoint("learner", "v1")
        assert "production" in checkpoint.tags

    def test_tag_checkpoint_duplicate(self, store: CheckpointStore) -> None:
        """Test adding duplicate tag."""
        store.create_checkpoint("learner", "v1", {}, tags=["existing"])

        store.tag_checkpoint("learner", "v1", "existing")

        checkpoint = store.get_checkpoint("learner", "v1")
        assert checkpoint.tags.count("existing") == 1

    def test_tag_checkpoint_not_found(self, store: CheckpointStore) -> None:
        """Test tagging non-existent checkpoint."""
        result = store.tag_checkpoint("unknown", "v1", "tag")
        assert result is False

    def test_file_persistence(self, temp_storage_path: Path) -> None:
        """Test checkpoints are persisted to files."""
        store = CheckpointStore(storage_path=temp_storage_path)
        store.create_checkpoint("learner", "v1.0.0", {"state": "test"})

        # Check file exists
        file_path = temp_storage_path / "learner" / "v1.0.0.json.gz"
        assert file_path.exists()

        # Read and verify
        with gzip.open(file_path, "rt") as f:
            data = json.load(f)
            assert data["version"] == "v1.0.0"

    def test_file_loading(self, temp_storage_path: Path) -> None:
        """Test loading checkpoints from files."""
        # Create checkpoint with first store
        store1 = CheckpointStore(storage_path=temp_storage_path)
        store1.create_checkpoint("learner", "v1", {"q": 0.5})

        # Create new store and verify loaded
        store2 = CheckpointStore(storage_path=temp_storage_path)
        checkpoint = store2.get_checkpoint("learner", "v1")

        assert checkpoint is not None
        assert checkpoint.state == {"q": 0.5}

    def test_cleanup_old_checkpoints(self, temp_storage_path: Path) -> None:
        """Test cleanup of old checkpoints."""
        # Create store with low limit for testing
        store = CheckpointStore(storage_path=temp_storage_path)
        store.MAX_CHECKPOINTS_PER_LEARNER = 3

        # Create more than limit
        for i in range(5):
            store.create_checkpoint("learner", f"v{i}", {"v": i})

        # Should only have 3 checkpoints
        checkpoints = store.list_checkpoints("learner")
        assert len(checkpoints) == 3

        # Should have newest ones
        versions = [cp.version for cp in checkpoints]
        assert "v4" in versions
        assert "v3" in versions
        assert "v2" in versions

    def test_export_metrics(self, store: CheckpointStore) -> None:
        """Test metrics export."""
        store.create_checkpoint("learner1", "v1", {})
        store.create_checkpoint("learner1", "v2", {})
        store.create_checkpoint("learner2", "v1", {})

        metrics = store.export_metrics()

        assert metrics["total_checkpoints"] == 3
        assert metrics["learners"] == 2
        assert metrics["checkpoints_per_learner"]["learner1"] == 2
        assert metrics["checkpoints_per_learner"]["learner2"] == 1

    def test_special_characters_in_version(self, store: CheckpointStore) -> None:
        """Test version strings with special characters."""
        store.create_checkpoint("learner", "v1.0.0/rc1", {"state": 1})

        checkpoint = store.get_checkpoint("learner", "v1.0.0/rc1")
        assert checkpoint is not None
        assert checkpoint.version == "v1.0.0/rc1"


class TestGlobalSingleton:
    """Tests for global singleton."""

    def test_get_checkpoint_store(self) -> None:
        """Test getting global singleton."""
        import victor.framework.rl.checkpoint_store as module

        module._checkpoint_store = None

        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = get_checkpoint_store(storage_path=Path(tmpdir))
            store2 = get_checkpoint_store()

            assert store1 is store2

    def test_singleton_preserves_state(self) -> None:
        """Test singleton preserves state."""
        import victor.framework.rl.checkpoint_store as module

        module._checkpoint_store = None

        with tempfile.TemporaryDirectory() as tmpdir:
            store = get_checkpoint_store(storage_path=Path(tmpdir))
            store.create_checkpoint("test", "v1", {})

            store2 = get_checkpoint_store()
            assert store2.get_checkpoint("test", "v1") is not None
