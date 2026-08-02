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

"""Tests for the change tracker undo/redo system."""

import os
import tempfile
import pytest

from victor.agent.change_tracker import (
    FileChangeHistory,
    ChangeType,
    FileChange,
    ChangeGroup,
    get_change_tracker,
    reset_change_tracker,
)


@pytest.fixture
def tracker():
    """Create a fresh change tracker for each test."""
    from pathlib import Path

    reset_change_tracker()
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = FileChangeHistory(project_path=Path(tmpdir), max_history=50)
        yield tracker


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_change_types_exist(self):
        """Verify all change types exist."""
        assert ChangeType.CREATE.value == "create"
        assert ChangeType.MODIFY.value == "modify"
        assert ChangeType.DELETE.value == "delete"
        assert ChangeType.RENAME.value == "rename"


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_create_file_change(self):
        """Test creating a FileChange instance."""
        import time

        change = FileChange(
            id="test-change-1",
            change_type=ChangeType.CREATE,
            file_path="/tmp/test.py",
            timestamp=time.time(),
            tool_name="write_file",
            tool_args={"path": "/tmp/test.py"},
            original_content=None,
            new_content="print('hello')",
        )

        assert change.file_path == "/tmp/test.py"
        assert change.change_type == ChangeType.CREATE
        assert change.original_content is None
        assert change.new_content == "print('hello')"
        assert change.tool_name == "write_file"

    def test_modify_file_change(self):
        """Test creating a modification change."""
        import time

        change = FileChange(
            id="test-change-2",
            change_type=ChangeType.MODIFY,
            file_path="/tmp/test.py",
            timestamp=time.time(),
            tool_name="edit_files",
            tool_args={},
            original_content="old content",
            new_content="new content",
        )

        assert change.change_type == ChangeType.MODIFY
        assert change.original_content == "old content"
        assert change.new_content == "new content"

    def test_rename_file_change(self):
        """Test creating a rename change."""
        import time

        change = FileChange(
            id="test-change-3",
            change_type=ChangeType.RENAME,
            file_path="/tmp/new.py",
            timestamp=time.time(),
            tool_name="edit_files",
            tool_args={},
            original_path="/tmp/old.py",
        )

        assert change.change_type == ChangeType.RENAME
        assert change.original_path == "/tmp/old.py"


class TestChangeGroup:
    """Tests for ChangeGroup dataclass."""

    def test_create_change_group(self):
        """Test creating a ChangeGroup."""
        import time

        changes = [
            FileChange(
                id="test-change-1",
                change_type=ChangeType.CREATE,
                file_path="/tmp/test.py",
                timestamp=time.time(),
                tool_name="write_file",
                tool_args={},
                new_content="content",
            )
        ]
        group = ChangeGroup(
            id="test-123",
            changes=changes,
            tool_name="write_file",
            description="Create test file",
        )

        assert group.id == "test-123"
        assert group.tool_name == "write_file"
        assert group.description == "Create test file"
        assert len(group.changes) == 1
        assert group.timestamp is not None


class TestFileChangeHistory:
    """Tests for the FileChangeHistory class."""

    def test_begin_change_group(self, tracker):
        """Test beginning a change group."""
        group_id = tracker.begin_change_group("test_tool", "Test description")

        assert group_id is not None
        assert tracker._current_group is not None
        assert tracker._current_group.tool_name == "test_tool"
        assert tracker._current_group.description == "Test description"

    def test_record_change(self, tracker):
        """Test recording a change."""
        tracker.begin_change_group("write_file", "Create file")
        tracker.record_change(
            file_path="/tmp/test.py",
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="print('hello')",
            tool_name="write_file",
        )

        assert len(tracker._current_group.changes) == 1
        change = tracker._current_group.changes[0]
        assert change.file_path == "/tmp/test.py"
        assert change.change_type == ChangeType.CREATE

    def test_commit_change_group(self, tracker):
        """Test committing a change group."""
        tracker.begin_change_group("write_file", "Create file")
        tracker.record_change(
            file_path="/tmp/test.py",
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="content",
            tool_name="write_file",
        )
        group = tracker.commit_change_group()

        assert group is not None
        assert tracker._current_group is None
        assert tracker.can_undo()

    def test_can_undo_empty(self, tracker):
        """Test can_undo when no changes exist."""
        assert not tracker.can_undo()

    def test_can_redo_empty(self, tracker):
        """Test can_redo when no undone changes exist."""
        assert not tracker.can_redo()


class TestUndoRedo:
    """Tests for undo/redo functionality with actual file operations."""

    def test_undo_file_create(self, tracker, temp_dir):
        """Test undoing a file creation."""
        file_path = os.path.join(temp_dir, "test.py")
        content = "print('hello')"

        # Create the file
        with open(file_path, "w") as f:
            f.write(content)

        # Record the creation
        tracker.begin_change_group("write_file", "Create test.py")
        tracker.record_change(
            file_path=file_path,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content=content,
            tool_name="write_file",
        )
        tracker.commit_change_group()

        # Verify file exists
        assert os.path.exists(file_path)

        # Undo the creation
        success, message, files = tracker.undo()

        assert success
        assert not os.path.exists(file_path)
        assert file_path in files

    def test_redo_file_create(self, tracker, temp_dir):
        """Test redoing a file creation after undo."""
        file_path = os.path.join(temp_dir, "test.py")
        content = "print('hello')"

        # Create the file
        with open(file_path, "w") as f:
            f.write(content)

        # Record the creation
        tracker.begin_change_group("write_file", "Create test.py")
        tracker.record_change(
            file_path=file_path,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content=content,
            tool_name="write_file",
        )
        tracker.commit_change_group()

        # Undo
        tracker.undo()
        assert not os.path.exists(file_path)

        # Redo
        success, message, files = tracker.redo()

        assert success
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            assert f.read() == content

    def test_undo_file_modify(self, tracker, temp_dir):
        """Test undoing a file modification."""
        file_path = os.path.join(temp_dir, "test.py")
        original = "original content"
        modified = "modified content"

        # Create original file
        with open(file_path, "w") as f:
            f.write(original)

        # Modify and record
        with open(file_path, "w") as f:
            f.write(modified)

        tracker.begin_change_group("write_file", "Modify test.py")
        tracker.record_change(
            file_path=file_path,
            change_type=ChangeType.MODIFY,
            original_content=original,
            new_content=modified,
            tool_name="write_file",
        )
        tracker.commit_change_group()

        # Verify modified content
        with open(file_path, "r") as f:
            assert f.read() == modified

        # Undo the modification
        success, message, files = tracker.undo()

        assert success
        with open(file_path, "r") as f:
            assert f.read() == original

    def test_undo_file_delete(self, tracker, temp_dir):
        """Test undoing a file deletion."""
        file_path = os.path.join(temp_dir, "test.py")
        content = "file content"

        # Create and then delete the file
        with open(file_path, "w") as f:
            f.write(content)

        # Record and delete
        tracker.begin_change_group("edit_files", "Delete test.py")
        tracker.record_change(
            file_path=file_path,
            change_type=ChangeType.DELETE,
            original_content=content,
            new_content=None,
            tool_name="edit_files",
        )
        tracker.commit_change_group()

        os.remove(file_path)
        assert not os.path.exists(file_path)

        # Undo the deletion (restore the file)
        success, message, files = tracker.undo()

        assert success
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            assert f.read() == content

    def test_multiple_undo(self, tracker, temp_dir):
        """Test multiple undo operations."""
        file1 = os.path.join(temp_dir, "file1.py")
        file2 = os.path.join(temp_dir, "file2.py")

        # Create first file
        with open(file1, "w") as f:
            f.write("content1")

        tracker.begin_change_group("write_file", "Create file1")
        tracker.record_change(
            file_path=file1,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="content1",
            tool_name="write_file",
        )
        tracker.commit_change_group()

        # Create second file
        with open(file2, "w") as f:
            f.write("content2")

        tracker.begin_change_group("write_file", "Create file2")
        tracker.record_change(
            file_path=file2,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="content2",
            tool_name="write_file",
        )
        tracker.commit_change_group()

        assert os.path.exists(file1)
        assert os.path.exists(file2)

        # Undo second
        success1, _, _ = tracker.undo()
        assert success1
        assert os.path.exists(file1)
        assert not os.path.exists(file2)

        # Undo first
        success2, _, _ = tracker.undo()
        assert success2
        assert not os.path.exists(file1)
        assert not os.path.exists(file2)

    def test_undo_clears_redo_on_new_change(self, tracker, temp_dir):
        """Test that new changes clear the redo stack."""
        file1 = os.path.join(temp_dir, "file1.py")
        file2 = os.path.join(temp_dir, "file2.py")

        # Create and commit first file
        with open(file1, "w") as f:
            f.write("content1")

        tracker.begin_change_group("write_file", "Create file1")
        tracker.record_change(
            file_path=file1,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="content1",
            tool_name="write_file",
        )
        tracker.commit_change_group()

        # Undo
        tracker.undo()
        assert tracker.can_redo()

        # New change should clear redo stack
        with open(file2, "w") as f:
            f.write("content2")

        tracker.begin_change_group("write_file", "Create file2")
        tracker.record_change(
            file_path=file2,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="content2",
            tool_name="write_file",
        )
        tracker.commit_change_group()

        assert not tracker.can_redo()


class TestGetChangeTracker:
    """Tests for singleton pattern."""

    def test_get_change_tracker_returns_same_instance(self):
        """Test that get_change_tracker returns the same instance."""
        reset_change_tracker()
        tracker1 = get_change_tracker()
        tracker2 = get_change_tracker()
        assert tracker1 is tracker2

    def test_reset_change_tracker(self):
        """Test resetting the tracker creates a new instance."""
        reset_change_tracker()
        tracker1 = get_change_tracker()
        reset_change_tracker()
        tracker2 = get_change_tracker()
        assert tracker1 is not tracker2


class TestHistory:
    """Tests for change history."""

    def test_get_history_empty(self, tracker):
        """Test getting history when empty."""
        history = tracker.get_history()
        assert history == []

    def test_get_history_with_changes(self, tracker, temp_dir):
        """Test getting history with changes."""
        file_path = os.path.join(temp_dir, "test.py")

        with open(file_path, "w") as f:
            f.write("content")

        tracker.begin_change_group("write_file", "Create test.py")
        tracker.record_change(
            file_path=file_path,
            change_type=ChangeType.CREATE,
            original_content=None,
            new_content="content",
            tool_name="write_file",
        )
        tracker.commit_change_group()

        history = tracker.get_history()

        assert len(history) == 1
        assert history[0]["tool_name"] == "write_file"
        assert history[0]["description"] == "Create test.py"
        assert history[0]["undone"] is False  # Not undone = applied

    def test_get_history_limit(self, tracker, temp_dir):
        """Test history limit."""
        # Create multiple changes
        for i in range(5):
            file_path = os.path.join(temp_dir, f"test{i}.py")
            with open(file_path, "w") as f:
                f.write(f"content{i}")

            tracker.begin_change_group("write_file", f"Create test{i}.py")
            tracker.record_change(
                file_path=file_path,
                change_type=ChangeType.CREATE,
                original_content=None,
                new_content=f"content{i}",
                tool_name="write_file",
            )
            tracker.commit_change_group()

        history = tracker.get_history(limit=3)
        assert len(history) == 3


class TestSaveGroupLockRetry:
    """P6: _save_group retries transient 'database is locked' errors."""

    def _record_one(self, tracker):
        tracker.begin_change_group("edit", "Test lock retry")
        tracker.record_change(
            file_path="/tmp/test.py",
            change_type=ChangeType.MODIFY,
            original_content="a",
            new_content="b",
            tool_name="edit",
        )

    def test_save_group_retries_transient_lock_then_succeeds(self, tracker, monkeypatch):
        """Two 'locked' failures then success -> commit succeeds after 2 retries."""
        import sqlite3

        import victor.agent.change_tracker as ct_mod

        sleeps = []
        monkeypatch.setattr(ct_mod.time, "sleep", lambda s: sleeps.append(s))

        calls = {"n": 0}

        def _flaky(group):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(tracker, "_save_group_once", _flaky)

        self._record_one(tracker)
        group = tracker.commit_change_group()

        assert group is not None
        assert calls["n"] == 3
        assert sleeps == [0.1, 0.1]

    def test_save_group_reraises_after_bounded_retries(self, tracker, monkeypatch):
        """Three consecutive 'locked' failures -> the error propagates."""
        import sqlite3

        import victor.agent.change_tracker as ct_mod

        sleeps = []
        monkeypatch.setattr(ct_mod.time, "sleep", lambda s: sleeps.append(s))

        calls = {"n": 0}

        def _always_locked(group):
            calls["n"] += 1
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(tracker, "_save_group_once", _always_locked)

        self._record_one(tracker)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            tracker.commit_change_group()

        assert calls["n"] == 3  # 1 initial attempt + 2 retries
        assert sleeps == [0.1, 0.1]

    def test_save_group_non_lock_error_not_retried(self, tracker, monkeypatch):
        """A non-lock OperationalError propagates immediately (no retries)."""
        import sqlite3

        import victor.agent.change_tracker as ct_mod

        sleeps = []
        monkeypatch.setattr(ct_mod.time, "sleep", lambda s: sleeps.append(s))

        calls = {"n": 0}

        def _broken(group):
            calls["n"] += 1
            raise sqlite3.OperationalError("no such table: change_groups")

        monkeypatch.setattr(tracker, "_save_group_once", _broken)

        self._record_one(tracker)
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            tracker.commit_change_group()

        assert calls["n"] == 1
        assert sleeps == []


class TestDedicatedUndoDb:
    """The tracker persists to the dedicated undo.db, not project.db."""

    def test_uses_undo_db_file(self, tracker):
        assert tracker._db_path.name == "undo.db"

    def test_does_not_create_project_db(self, temp_dir):
        from pathlib import Path

        proj = Path(temp_dir)
        FileChangeHistory(project_path=proj)
        assert (proj / ".victor" / "undo.db").exists()
        assert not (proj / ".victor" / "project.db").exists()


class TestConflictGuard:
    """Session-scoped conflict guard: don't clobber files changed elsewhere."""

    def _commit_modify(self, tracker, f):
        with open(f, "w") as fh:
            fh.write("original")
        with open(f, "w") as fh:
            fh.write("modified")
        tracker.begin_change_group("edit", "mod")
        tracker.record_change(f, ChangeType.MODIFY, "original", "modified", tool_name="edit")
        tracker.commit_change_group()

    def test_undo_skips_externally_modified_file(self, tracker, temp_dir):
        f = os.path.join(temp_dir, "x.py")
        self._commit_modify(tracker, f)
        # Another session changes the file after our edit.
        with open(f, "w") as fh:
            fh.write("someone elses change")

        success, message, files = tracker.undo()

        assert not success
        assert "another session" in message.lower()
        assert files == []
        with open(f) as fh:
            assert fh.read() == "someone elses change"  # not clobbered
        assert tracker.can_undo()  # not consumed; can retry with force

    def test_force_undo_overrides_conflict(self, tracker, temp_dir):
        f = os.path.join(temp_dir, "x.py")
        self._commit_modify(tracker, f)
        with open(f, "w") as fh:
            fh.write("someone elses change")

        success, _message, _files = tracker.undo(force=True)

        assert success
        with open(f) as fh:
            assert fh.read() == "original"
        assert tracker.can_redo()


class TestAtomicGroupReplay:
    """Group undo is all-or-nothing on the filesystem (crash-safe rollback)."""

    def test_partial_failure_rolls_back_applied_files(self, tracker, temp_dir, monkeypatch):
        f1 = os.path.join(temp_dir, "f1.py")
        f2 = os.path.join(temp_dir, "f2.py")
        for f in (f1, f2):
            with open(f, "w") as fh:
                fh.write("orig")
        for f in (f1, f2):
            with open(f, "w") as fh:
                fh.write("mod")

        tracker.begin_change_group("edit", "two files")
        tracker.record_change(f1, ChangeType.MODIFY, "orig", "mod", tool_name="edit")
        tracker.record_change(f2, ChangeType.MODIFY, "orig", "mod", tool_name="edit")
        tracker.commit_change_group()

        # undo reverses in reverse order (f2 then f1); fail on f1 AFTER f2 reverted.
        real_reverse = tracker._reverse_change

        def flaky_reverse(change):
            if change.file_path == f1:
                raise OSError("simulated write failure")
            return real_reverse(change)

        monkeypatch.setattr(tracker, "_reverse_change", flaky_reverse)

        success, message, files = tracker.undo()

        assert not success
        assert "rolled back" in message.lower()
        assert files == []
        # Both files restored to their pre-undo (modified) content — nothing half-done.
        with open(f1) as fh:
            assert fh.read() == "mod"
        with open(f2) as fh:
            assert fh.read() == "mod"
        assert tracker.can_undo()  # state transition not committed

    def test_atomic_write_leaves_no_temp_on_success(self, tracker, temp_dir):
        f = os.path.join(temp_dir, "x.py")
        with open(f, "w") as fh:
            fh.write("v2")
        tracker.begin_change_group("edit", "mod")
        tracker.record_change(f, ChangeType.MODIFY, "v1", "v2", tool_name="edit")
        tracker.commit_change_group()

        tracker.undo()

        with open(f) as fh:
            assert fh.read() == "v1"
        # No leftover .undo_tmp_* temp files in the dir.
        leftovers = [p for p in os.listdir(temp_dir) if p.startswith(".undo_tmp_")]
        assert leftovers == []


class TestMessageIdAndSeq:
    """message_id (turn attribution) and seq (deterministic order) persist."""

    def test_message_id_persists_on_group_and_changes(self, tracker, temp_dir):
        f = os.path.join(temp_dir, "x.py")
        with open(f, "w") as fh:
            fh.write("c")
        tracker.begin_change_group("edit", "d", message_id="turn-7")
        tracker.record_change(f, ChangeType.CREATE, None, "c", tool_name="edit")
        grp = tracker.commit_change_group()

        conn = tracker._db.get_connection()
        gmid = conn.execute(
            "SELECT message_id FROM change_groups WHERE id=?", (grp.id,)
        ).fetchone()[0]
        assert gmid == "turn-7"
        seq, cmid = conn.execute(
            "SELECT seq, message_id FROM file_changes WHERE group_id=?", (grp.id,)
        ).fetchone()
        assert seq == 0
        assert cmid == "turn-7"

    def test_seq_orders_multiple_files(self, tracker, temp_dir):
        f1 = os.path.join(temp_dir, "a.py")
        f2 = os.path.join(temp_dir, "b.py")
        for f in (f1, f2):
            with open(f, "w") as fh:
                fh.write("x")
        tracker.begin_change_group("edit", "two")
        tracker.record_change(f1, ChangeType.CREATE, None, "x", tool_name="edit")
        tracker.record_change(f2, ChangeType.CREATE, None, "x", tool_name="edit")
        grp = tracker.commit_change_group()

        rows = (
            tracker._db.get_connection()
            .execute(
                "SELECT file_path, seq FROM file_changes WHERE group_id=? ORDER BY seq",
                (grp.id,),
            )
            .fetchall()
        )
        assert [r[1] for r in rows] == [0, 1]
        assert rows[0][0] == f1 and rows[1][0] == f2


class TestCrossSessionScoping:
    """Multiple sessions share undo.db but each undoes only its own group."""

    def test_two_sessions_undo_own_group(self, tmp_path):
        from pathlib import Path

        from victor.core.undo_database import reset_undo_databases

        reset_change_tracker()
        reset_undo_databases()
        proj = tmp_path / "proj"
        proj.mkdir()
        fdir = tmp_path / "files"
        fdir.mkdir()

        t_a = FileChangeHistory(project_path=proj, session_id="A")
        t_b = FileChangeHistory(project_path=proj, session_id="B")
        assert t_a._db is t_b._db  # same dedicated undo.db manager (cached per path)

        fa = str(fdir / "a.py")
        fb = str(fdir / "b.py")
        Path(fa).write_text("a")
        Path(fb).write_text("b")

        t_a.begin_change_group("edit", "A change")
        t_a.record_change(fa, ChangeType.CREATE, None, "a", tool_name="edit")
        t_a.commit_change_group()

        t_b.begin_change_group("edit", "B change")
        t_b.record_change(fb, ChangeType.CREATE, None, "b", tool_name="edit")
        t_b.commit_change_group()

        ok_a, _, files_a = t_a.undo()
        assert ok_a and fa in files_a and fb not in files_a
        ok_b, _, files_b = t_b.undo()
        assert ok_b and fb in files_b
