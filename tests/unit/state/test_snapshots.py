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

"""Tests for the workspace snapshots module."""

import tempfile
from pathlib import Path

import pytest

from victor.agent.snapshot_store import (
    FileSnapshot,
    FileSnapshotStore,
    WorkspaceSnapshot,
    get_snapshot_store,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        # Create some test files
        (workspace / "file1.py").write_text("print('hello')\n")
        (workspace / "file2.py").write_text("x = 1\ny = 2\n")
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "file3.py").write_text("# nested file\n")
        yield workspace


@pytest.fixture
def snapshot_manager(temp_workspace):
    """Create a snapshot manager with temporary workspace."""
    return FileSnapshotStore(workspace_root=temp_workspace, max_snapshots=5)


class TestFileSnapshot:
    """Tests for FileSnapshot dataclass."""

    def test_to_dict(self):
        """to_dict should return all fields."""
        snap = FileSnapshot(
            path="src/api.py",
            content="print('api')",
            content_hash="abc123",
            permissions=0o644,
            existed=True,
        )
        data = snap.to_dict()

        assert data["path"] == "src/api.py"
        assert data["content"] == "print('api')"
        assert data["content_hash"] == "abc123"
        assert data["permissions"] == 0o644
        assert data["existed"] is True

    def test_from_dict(self):
        """from_dict should restore all fields."""
        data = {
            "path": "test.py",
            "content": "# test",
            "content_hash": "def456",
            "permissions": 0o755,
            "existed": True,
        }
        snap = FileSnapshot.from_dict(data)

        assert snap.path == "test.py"
        assert snap.content == "# test"
        assert snap.content_hash == "def456"
        assert snap.permissions == 0o755

    def test_nonexistent_file(self):
        """FileSnapshot should track non-existent files."""
        snap = FileSnapshot(
            path="new_file.py",
            content=None,
            existed=False,
        )
        assert snap.existed is False
        assert snap.content is None


class TestWorkspaceSnapshot:
    """Tests for WorkspaceSnapshot dataclass."""

    def test_to_dict_and_from_dict(self):
        """WorkspaceSnapshot should serialize and deserialize correctly."""
        file_snap = FileSnapshot(path="test.py", content="# test", existed=True)
        snapshot = WorkspaceSnapshot(
            snapshot_id="snap_1_120000",
            created_at="2025-01-01T12:00:00",
            description="Test snapshot",
            files=[file_snap],
            workspace_root="/home/user/project",
            git_ref="abc123def",
            metadata={"tool": "edit_files"},
        )

        # Serialize
        data = snapshot.to_dict()

        # Deserialize
        restored = WorkspaceSnapshot.from_dict(data)

        assert restored.snapshot_id == "snap_1_120000"
        assert restored.description == "Test snapshot"
        assert restored.file_count == 1
        assert restored.git_ref == "abc123def"
        assert restored.metadata["tool"] == "edit_files"

    def test_get_file(self):
        """get_file should return file by path."""
        file1 = FileSnapshot(path="src/api.py", content="api", existed=True)
        file2 = FileSnapshot(path="src/utils.py", content="utils", existed=True)
        snapshot = WorkspaceSnapshot(
            snapshot_id="test",
            created_at="2025-01-01T12:00:00",
            description="",
            files=[file1, file2],
        )

        assert snapshot.get_file("src/api.py") == file1
        assert snapshot.get_file("src/utils.py") == file2
        assert snapshot.get_file("nonexistent.py") is None


class TestFileSnapshotStore:
    """Tests for FileSnapshotStore class."""

    def test_create_snapshot(self, snapshot_manager, temp_workspace):
        """create_snapshot should capture file state."""
        snapshot_id = snapshot_manager.create_snapshot(
            files=["file1.py", "file2.py"],
            description="Test snapshot",
        )

        assert snapshot_id is not None
        assert snapshot_id.startswith("snap_")

        snap = snapshot_manager.get_snapshot(snapshot_id)
        assert snap is not None
        assert snap.file_count == 2
        assert snap.description == "Test snapshot"

    def test_create_snapshot_auto_detect_files(self, snapshot_manager, temp_workspace):
        """create_snapshot with no files should auto-detect from git status."""
        # Without git, this should return empty list but not crash
        snapshot_id = snapshot_manager.create_snapshot(description="Auto")
        snap = snapshot_manager.get_snapshot(snapshot_id)
        assert snap is not None

    def test_restore_snapshot(self, snapshot_manager, temp_workspace):
        """restore_snapshot should restore file content."""
        # Create snapshot
        snapshot_id = snapshot_manager.create_snapshot(
            files=["file1.py"],
            description="Before change",
        )

        # Modify file
        (temp_workspace / "file1.py").write_text("print('modified')\n")
        assert (temp_workspace / "file1.py").read_text() == "print('modified')\n"

        # Restore
        result = snapshot_manager.restore_snapshot(snapshot_id)
        assert result is True

        # Verify restored
        assert (temp_workspace / "file1.py").read_text() == "print('hello')\n"

    def test_restore_deletes_new_files(self, snapshot_manager, temp_workspace):
        """restore_snapshot should delete files that didn't exist."""
        # Create snapshot of non-existent file
        snapshot_id = snapshot_manager.create_snapshot(
            files=["new_file.py"],
            description="Before new file",
        )

        # Create new file
        (temp_workspace / "new_file.py").write_text("# new\n")
        assert (temp_workspace / "new_file.py").exists()

        # Restore
        result = snapshot_manager.restore_snapshot(snapshot_id)
        assert result is True

        # File should be deleted
        assert not (temp_workspace / "new_file.py").exists()

    def test_diff_snapshot_modified(self, snapshot_manager, temp_workspace):
        """diff_snapshot should detect modified files."""
        snapshot_id = snapshot_manager.create_snapshot(
            files=["file1.py"],
            description="Original",
        )

        # Modify file
        (temp_workspace / "file1.py").write_text("print('changed')\n")

        diffs = snapshot_manager.diff_snapshot(snapshot_id)
        assert len(diffs) == 1
        assert diffs[0].status == "modified"
        assert diffs[0].path == "file1.py"

    def test_diff_snapshot_added(self, snapshot_manager, temp_workspace):
        """diff_snapshot should detect added files."""
        # Snapshot non-existent file
        snapshot_id = snapshot_manager.create_snapshot(
            files=["new_file.py"],
            description="Before add",
        )

        # Create file
        (temp_workspace / "new_file.py").write_text("# new\n")

        diffs = snapshot_manager.diff_snapshot(snapshot_id)
        assert len(diffs) == 1
        assert diffs[0].status == "added"

    def test_diff_snapshot_deleted(self, snapshot_manager, temp_workspace):
        """diff_snapshot should detect deleted files."""
        snapshot_id = snapshot_manager.create_snapshot(
            files=["file1.py"],
            description="Before delete",
        )

        # Delete file
        (temp_workspace / "file1.py").unlink()

        diffs = snapshot_manager.diff_snapshot(snapshot_id)
        assert len(diffs) == 1
        assert diffs[0].status == "deleted"

    def test_list_snapshots(self, snapshot_manager):
        """list_snapshots should return recent snapshots."""
        # Create multiple snapshots
        id1 = snapshot_manager.create_snapshot(files=["file1.py"], description="First")
        id2 = snapshot_manager.create_snapshot(files=["file2.py"], description="Second")
        id3 = snapshot_manager.create_snapshot(files=["file1.py", "file2.py"], description="Third")

        snapshots = snapshot_manager.list_snapshots(limit=10)
        assert len(snapshots) == 3

        # Should be newest first
        assert snapshots[0].snapshot_id == id3
        assert snapshots[1].snapshot_id == id2
        assert snapshots[2].snapshot_id == id1

    def test_list_snapshots_with_limit(self, snapshot_manager):
        """list_snapshots should respect limit parameter."""
        for i in range(5):
            snapshot_manager.create_snapshot(files=["file1.py"], description=f"Snap {i}")

        snapshots = snapshot_manager.list_snapshots(limit=2)
        assert len(snapshots) == 2

    def test_delete_snapshot(self, snapshot_manager):
        """delete_snapshot should remove snapshot."""
        snapshot_id = snapshot_manager.create_snapshot(
            files=["file1.py"],
            description="To delete",
        )

        assert snapshot_manager.get_snapshot(snapshot_id) is not None

        result = snapshot_manager.delete_snapshot(snapshot_id)
        assert result is True

        assert snapshot_manager.get_snapshot(snapshot_id) is None

    def test_delete_nonexistent_snapshot(self, snapshot_manager):
        """delete_snapshot should return False for missing snapshots."""
        result = snapshot_manager.delete_snapshot("nonexistent")
        assert result is False

    def test_max_snapshots_cleanup(self, temp_workspace):
        """Old snapshots should be cleaned up when max_snapshots exceeded."""
        manager = FileSnapshotStore(workspace_root=temp_workspace, max_snapshots=3)

        ids = []
        for i in range(5):
            ids.append(manager.create_snapshot(files=["file1.py"], description=f"Snap {i}"))

        # Should only have 3 snapshots
        assert len(manager.list_snapshots(limit=10)) == 3

        # Oldest should be gone
        assert manager.get_snapshot(ids[0]) is None
        assert manager.get_snapshot(ids[1]) is None

        # Newest should exist
        assert manager.get_snapshot(ids[4]) is not None

    def test_clear_all(self, snapshot_manager):
        """clear_all should remove all snapshots."""
        for i in range(3):
            snapshot_manager.create_snapshot(files=["file1.py"], description=f"Snap {i}")

        count = snapshot_manager.clear_all()
        assert count == 3
        assert len(snapshot_manager.list_snapshots()) == 0


class TestGlobalFileSnapshotStore:
    """Tests for global snapshot store functions."""

    def test_get_snapshot_manager_singleton(self):
        """get_snapshot_store should return same instance."""
        manager1 = get_snapshot_store()
        manager2 = get_snapshot_store()
        assert manager1 is manager2
