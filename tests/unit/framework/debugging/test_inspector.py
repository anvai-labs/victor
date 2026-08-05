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

"""Unit tests for state inspection."""

import pytest

from victor.framework.debugging.inspector import (
    StateInspector,
    StateSnapshot,
    StateDiff,
)


@pytest.mark.unit
class TestStateSnapshot:
    """Test StateSnapshot dataclass."""

    def test_capture_snapshot(self, sample_state):
        """Test capturing state snapshot."""
        snapshot = StateSnapshot.capture(sample_state, "analyze")

        assert snapshot.node_id == "analyze"
        assert snapshot.state == sample_state
        assert snapshot.state_summary == {
            "task": "str",
            "file_path": "str",
            "errors": "int",
            "results": "list",
            "metadata": "dict",
        }

    def test_snapshot_to_dict(self, sample_state):
        """Test snapshot serialization."""
        snapshot = StateSnapshot.capture(sample_state, "analyze")

        data = snapshot.to_dict()

        assert "timestamp" in data
        assert data["node_id"] == "analyze"
        assert "state_summary" in data
        assert "size_bytes" in data


@pytest.mark.unit
class TestStateDiff:
    """Test StateDiff comparison."""

    def test_compare_states_no_changes(self, sample_state):
        """Test comparing identical states."""
        diff = StateDiff.compare(sample_state, sample_state)

        assert diff.has_changes() is False
        assert len(diff.changed_keys) == 0
        assert len(diff.added_keys) == 0
        assert len(diff.removed_keys) == 0

    def test_compare_states_with_changes(self, sample_state):
        """Test comparing different states."""
        modified = sample_state.copy()
        modified["errors"] = 5
        modified["new_key"] = "new_value"
        del modified["task"]

        diff = StateDiff.compare(sample_state, modified)

        assert diff.has_changes() is True
        assert diff.changed_keys["errors"] == (0, 5)
        assert "new_key" in diff.added_keys
        assert "task" in diff.removed_keys

    def test_diff_to_dict(self, sample_state):
        """Test diff serialization."""
        modified = sample_state.copy()
        modified["errors"] = 5

        diff = StateDiff.compare(sample_state, modified)
        data = diff.to_dict()

        assert "added_keys" in data
        assert "removed_keys" in data
        assert "changed_keys" in data
        assert "unchanged_count" in data


@pytest.mark.unit
class TestStateInspector:
    """Test StateInspector."""

    def test_capture_snapshot(self, state_inspector, sample_state):
        """Test capturing state snapshot."""
        snapshot = state_inspector.capture_snapshot(sample_state, "analyze")

        assert snapshot.node_id == "analyze"
        assert snapshot.state == sample_state
        assert snapshot.state_summary == {
            "task": "str",
            "file_path": "str",
            "errors": "int",
            "results": "list",
            "metadata": "dict",
        }

    def test_compare_states_no_changes(self, state_inspector, sample_state):
        """Test comparing identical states."""
        diff = state_inspector.compare_states(sample_state, sample_state)

        assert diff.has_changes() is False
        assert len(diff.changed_keys) == 0
        assert len(diff.added_keys) == 0
        assert len(diff.removed_keys) == 0

    def test_compare_states_with_changes(self, state_inspector, sample_state):
        """Test comparing different states."""
        modified = sample_state.copy()
        modified["errors"] = 5
        modified["new_key"] = "new_value"
        del modified["task"]

        diff = state_inspector.compare_states(sample_state, modified)

        assert diff.has_changes() is True
        assert diff.changed_keys["errors"] == (0, 5)
        assert "new_key" in diff.added_keys
        assert "task" in diff.removed_keys

    def test_get_value_nested(self, state_inspector, sample_state):
        """Test getting nested value."""
        value = state_inspector.get_value(sample_state, "metadata.iteration")

        assert value == 1

    def test_get_value_missing(self, state_inspector, sample_state):
        """Test getting missing value with default."""
        value = state_inspector.get_value(sample_state, "missing_key", default="default")

        assert value == "default"

    def test_get_state_summary(self, state_inspector, sample_state):
        """Test getting state summary."""
        summary = state_inspector.get_state_summary(sample_state)

        assert summary == {
            "task": "str",
            "file_path": "str",
            "errors": "int",
            "results": "list",
            "metadata": "dict",
        }

    def test_get_snapshots(self, state_inspector, sample_state):
        """Test getting snapshots."""
        snapshot1 = state_inspector.capture_snapshot(sample_state, "analyze")
        snapshot2 = state_inspector.capture_snapshot(sample_state, "process")

        snapshots = state_inspector.get_snapshots()

        assert len(snapshots) == 2
        assert snapshot1 in snapshots
        assert snapshot2 in snapshots

    def test_get_snapshots_with_limit(self, state_inspector, sample_state):
        """Test getting snapshots with limit."""
        state_inspector.capture_snapshot(sample_state, "analyze")
        state_inspector.capture_snapshot(sample_state, "process")
        state_inspector.capture_snapshot(sample_state, "review")

        snapshots = state_inspector.get_snapshots(limit=2)

        assert len(snapshots) == 2

    def test_clear_snapshots(self, state_inspector, sample_state):
        """Test clearing snapshots."""
        state_inspector.capture_snapshot(sample_state, "analyze")
        state_inspector.clear_snapshots()

        assert len(state_inspector.get_snapshots()) == 0

    def test_get_large_state_keys(self, state_inspector):
        """Test finding large state keys."""
        large_state = {
            "small_key": "small",
            "large_key": "x" * 2000,  # > 1024 bytes
        }

        large_keys = state_inspector.get_large_state_keys(large_state)

        assert "large_key" in large_keys
        assert "small_key" not in large_keys

    def test_snapshot_history_limit(self, state_inspector, sample_state):
        """Test snapshot history is limited."""
        # Create inspector with max 3 snapshots
        inspector = StateInspector(max_snapshots=3)

        # Add 5 snapshots
        for i in range(5):
            inspector.capture_snapshot(sample_state, f"node_{i}")

        # Should only keep last 3
        snapshots = inspector.get_snapshots()
        assert len(snapshots) == 3
