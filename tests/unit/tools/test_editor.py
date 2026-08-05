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

"""Tests for editing/editor.py module."""

import pytest

from victor.processing.editing.editor import (
    OperationType,
    EditOperation,
    EditTransaction,
    FileEditor,
)


class TestOperationType:
    """Tests for OperationType enum."""

    def test_operation_types(self):
        """Test operation type values."""
        assert OperationType.CREATE == "create"
        assert OperationType.MODIFY == "modify"
        assert OperationType.DELETE == "delete"
        assert OperationType.RENAME == "rename"


class TestEditOperation:
    """Tests for EditOperation model."""

    def test_create_operation(self):
        """Test creating an EditOperation."""
        op = EditOperation(
            type=OperationType.CREATE, path="/test/file.txt", new_content="Hello World"
        )

        assert op.type == OperationType.CREATE
        assert op.path == "/test/file.txt"
        assert op.new_content == "Hello World"
        assert op.applied is False

    def test_modify_operation(self):
        """Test modify operation."""
        op = EditOperation(
            type=OperationType.MODIFY,
            path="/test/file.txt",
            old_content="Old",
            new_content="New",
        )

        assert op.type == OperationType.MODIFY
        assert op.old_content == "Old"
        assert op.new_content == "New"


class TestEditTransaction:
    """Tests for EditTransaction model."""

    def test_transaction_creation(self):
        """Test creating an EditTransaction."""
        tx = EditTransaction(id="tx_123", description="Test transaction")

        assert tx.id == "tx_123"
        assert tx.description == "Test transaction"
        assert tx.committed is False
        assert tx.rolled_back is False
        assert len(tx.operations) == 0


class TestFileEditor:
    """Tests for FileEditor class."""

    @pytest.fixture
    def editor(self, tmp_path):
        """Create a FileEditor with temporary backup dir."""
        backup_dir = tmp_path / "backups"
        return FileEditor(backup_dir=str(backup_dir), auto_backup=True)

    def test_editor_initialization(self, tmp_path):
        """Test FileEditor initialization."""
        backup_dir = tmp_path / "backups"
        editor = FileEditor(backup_dir=str(backup_dir))

        assert editor.backup_dir == backup_dir
        assert editor.auto_backup is True
        assert editor.current_transaction is None
        assert len(editor.transaction_history) == 0

    def test_start_transaction(self, editor):
        """Test starting a transaction."""
        tx_id = editor.start_transaction("Test transaction")

        assert tx_id is not None
        assert editor.current_transaction is not None
        assert editor.current_transaction.id == tx_id
        assert editor.current_transaction.description == "Test transaction"

    def test_start_transaction_when_active(self, editor):
        """Test starting transaction when one is already active."""
        editor.start_transaction("First")

        with pytest.raises(RuntimeError, match="Transaction already in progress"):
            editor.start_transaction("Second")

    def test_add_create(self, editor):
        """Test adding a create operation."""
        editor.start_transaction("Add file")
        editor.add_create("/test/new.txt", "Content")

        assert len(editor.current_transaction.operations) == 1
        op = editor.current_transaction.operations[0]
        assert op.type == OperationType.CREATE
        assert op.path == "/test/new.txt"
        assert op.new_content == "Content"

    def test_add_modify(self, editor, tmp_path):
        """Test adding a modify operation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        editor.start_transaction("Modify file")
        editor.add_modify(str(test_file), "New content")

        assert len(editor.current_transaction.operations) == 1
        op = editor.current_transaction.operations[0]
        assert op.type == OperationType.MODIFY
        assert op.path == str(test_file)
        assert op.old_content == "Original content"
        assert op.new_content == "New content"

    def test_add_delete(self, editor, tmp_path):
        """Test adding a delete operation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("To be deleted")

        editor.start_transaction("Delete file")
        editor.add_delete(str(test_file))

        assert len(editor.current_transaction.operations) == 1
        op = editor.current_transaction.operations[0]
        assert op.type == OperationType.DELETE
        assert op.path == str(test_file)
        assert op.old_content == "To be deleted"

    def test_add_rename(self, editor, tmp_path):
        """Test adding a rename operation."""
        test_file = tmp_path / "old.txt"
        test_file.write_text("Content")

        editor.start_transaction("Rename file")
        editor.add_rename(str(test_file), str(tmp_path / "new.txt"))

        assert len(editor.current_transaction.operations) == 1
        op = editor.current_transaction.operations[0]
        assert op.type == OperationType.RENAME
        assert op.path == str(test_file)
        assert op.new_path == str(tmp_path / "new.txt")

    def test_commit_create_operation(self, editor, tmp_path):
        """Test committing a create operation."""
        new_file = tmp_path / "new.txt"

        editor.start_transaction("Create file")
        editor.add_create(str(new_file), "Hello World")
        success = editor.commit()

        assert success is True
        assert new_file.exists()
        assert new_file.read_text() == "Hello World"
        assert editor.current_transaction is None

    def test_commit_modify_operation(self, editor, tmp_path):
        """Test committing a modify operation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original")

        editor.start_transaction("Modify file")
        editor.add_modify(str(test_file), "Modified")
        success = editor.commit()

        assert success is True
        assert test_file.read_text() == "Modified"

    def test_commit_delete_operation(self, editor, tmp_path):
        """Test committing a delete operation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("To delete")

        editor.start_transaction("Delete file")
        editor.add_delete(str(test_file))
        success = editor.commit()

        assert success is True
        assert not test_file.exists()

    def test_commit_dry_run(self, editor, tmp_path):
        """Test dry-run mode doesn't apply changes."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original")

        editor.start_transaction("Dry run")
        editor.add_modify(str(test_file), "Modified")
        success = editor.commit(dry_run=True)

        assert success is True
        assert test_file.read_text() == "Original"  # Not modified

    def test_rollback(self, editor, tmp_path):
        """Test rolling back committed changes."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original")

        editor.start_transaction("Modify and rollback")
        editor.add_modify(str(test_file), "Modified")
        editor.commit()

        # File should be modified
        assert test_file.read_text() == "Modified"

        # Rollback the last transaction
        if editor.transaction_history:
            # Set the last transaction as current to rollback
            last_tx = editor.transaction_history[-1]
            editor.current_transaction = last_tx
            success = editor.rollback()

            assert success is True
            assert test_file.read_text() == "Original"

    def test_abort_transaction(self, editor, tmp_path):
        """Test aborting a transaction."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original")

        editor.start_transaction("Abort test")
        editor.add_modify(str(test_file), "Modified")

        editor.abort()

        assert editor.current_transaction is None
        assert test_file.read_text() == "Original"

    def test_get_transaction_summary(self, editor, tmp_path):
        """Test getting transaction summary."""
        # Create a file to modify
        test_file = tmp_path / "file2.txt"
        test_file.write_text("Original")

        editor.start_transaction("Test summary")
        editor.add_create(str(tmp_path / "file1.txt"), "Content 1")
        editor.add_modify(str(test_file), "Content 2")

        summary = editor.get_transaction_summary()

        assert summary["id"] is not None
        assert summary["description"] == "Test summary"
        assert summary["operations"] == 2
        assert summary["committed"] is False

    def test_operations_without_transaction(self, editor):
        """Test that operations fail without active transaction."""
        with pytest.raises(RuntimeError, match="No active transaction"):
            editor.add_create("/test/file.txt", "Content")

    def test_multiple_operations(self, editor, tmp_path):
        """Test multiple operations in one transaction."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file2.write_text("Original")

        editor.start_transaction("Multiple ops")
        editor.add_create(str(file1), "New file")
        editor.add_modify(str(file2), "Modified")

        success = editor.commit()

        assert success is True
        assert file1.exists()
        assert file1.read_text() == "New file"
        assert file2.read_text() == "Modified"

    def test_preview_diff(self, editor, tmp_path):
        """Test preview_diff doesn't error."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original")

        editor.start_transaction("Preview test")
        editor.add_modify(str(test_file), "Modified")

        # Should not raise error (just displays to console)
        editor.preview_diff()
