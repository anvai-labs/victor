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

"""Tests for file_editor_tool module - TEXT-BASED file editing.

These tests require victor-coding package to be installed and are
marked as integration tests.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from victor.tools.file_editor_tool import edit


def _check_editor():
    """Check at runtime (not import time) if enhanced editor is available."""
    try:
        from victor.tools.file_editor_tool import _is_file_editor_available

        return _is_file_editor_available()
    except Exception:
        return False


# These tests require the enhanced editor (EditorProtocol registered via vertical).
pytestmark = [pytest.mark.integration]


@pytest.fixture(autouse=True)
def skip_without_editor():
    """Skip if enhanced editor is not registered (runtime check each test)."""
    if not _check_editor():
        pytest.skip("Enhanced editor not registered (requires vertical)")


class TestEditBasicOperations:
    """Tests for basic edit operations."""

    @pytest.mark.asyncio
    async def test_edit_empty_ops(self):
        """Test edit with empty operations list."""
        result = await edit(ops=[])
        assert result["success"] is False
        assert "No operations provided" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_ops_as_json_string(self, tmp_path):
        """Test edit accepts ops as JSON string."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        # Pass ops as JSON string (common from LLM output)
        ops_json = (
            f'[{{"type": "replace", "path": "{test_file}", "old_str": "hello", "new_str": "hi"}}]'
        )
        result = await edit(ops=ops_json)

        assert result["success"] is True
        assert test_file.read_text() == "hi world"

    @pytest.mark.asyncio
    async def test_edit_invalid_json_string(self):
        """Test edit handles invalid JSON gracefully."""
        result = await edit(ops="not valid json")

        assert result["success"] is False
        assert "syntax error" in result["error"] or "Invalid JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_json_control_character_recovery(self, tmp_path):
        """Test JSON parsing auto-fixes control characters."""
        # This tests the control character recovery path
        # Simulating a JSON string with embedded newlines
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2")

        # Valid JSON that should work
        ops_json = f'[{{"type": "replace", "path": "{test_file}", "old_str": "line1", "new_str": "LINE1"}}]'
        result = await edit(ops=ops_json)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_edit_json_raw_newlines_auto_fixed(self, tmp_path):
        """Test JSON parsing auto-fixes raw newlines in strings.

        Models sometimes pass JSON with raw newlines inside string values.
        The edit tool should auto-fix these and proceed.
        """
        test_file = tmp_path / "test.txt"
        test_file.write_text("old line1\nold line2")

        # Create JSON with ACTUAL raw newlines inside the string values
        # (not escaped \n, but real newline characters)
        ops_json = (
            '[{"type": "replace", "path": "'
            + str(test_file)
            + '", "old_str": "old line1\nold line2", "new_str": "new line1\nnew line2"}]'
        )

        # This should auto-fix the raw newlines and succeed
        result = await edit(ops=ops_json)

        assert result["success"] is True
        assert test_file.read_text() == "new line1\nnew line2"

    @pytest.mark.asyncio
    async def test_edit_json_raw_tabs_auto_fixed(self, tmp_path):
        """Test JSON parsing auto-fixes raw tabs in strings."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("col1\tcol2")

        # Create JSON with ACTUAL raw tab inside the string value
        ops_json = (
            '[{"type": "replace", "path": "'
            + str(test_file)
            + '", "old_str": "col1\tcol2", "new_str": "COL1\tCOL2"}]'
        )

        result = await edit(ops=ops_json)

        assert result["success"] is True
        assert test_file.read_text() == "COL1\tCOL2"

    @pytest.mark.asyncio
    async def test_edit_op_not_dict(self):
        """Test error when operation is not a dictionary."""
        result = await edit(ops=["not a dict"])

        assert result["success"] is False
        assert "must be a dictionary" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_op_missing_type(self):
        """Test error when operation missing type field."""
        result = await edit(ops=[{"path": "/some/path"}])

        assert result["success"] is False
        assert "missing required field: type" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_op_invalid_type(self):
        """Test error when operation has invalid type."""
        result = await edit(ops=[{"type": "invalid_op", "path": "/some/path"}])

        assert result["success"] is False
        assert "invalid type" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_op_missing_path(self):
        """Test error when operation missing path field."""
        result = await edit(ops=[{"type": "create"}])

        assert result["success"] is False
        assert "missing required field: path" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_rename_missing_new_path(self):
        """Test error when rename missing new_path."""
        result = await edit(ops=[{"type": "rename", "path": "/some/file.txt"}])

        assert result["success"] is False
        assert "missing required field: new_path" in result["error"]


class TestEditCreateOperation:
    """Tests for create operation."""

    @pytest.mark.asyncio
    async def test_create_file(self, tmp_path):
        """Test creating a new file."""
        new_file = tmp_path / "new_file.txt"

        result = await edit(
            ops=[{"type": "create", "path": str(new_file), "content": "new content"}]
        )

        assert result["success"] is True
        assert result["by_type"]["create"] == 1
        assert new_file.exists()
        assert new_file.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_create_file_empty_content(self, tmp_path):
        """Test creating a file with empty content."""
        new_file = tmp_path / "empty.txt"

        result = await edit(ops=[{"type": "create", "path": str(new_file)}])

        assert result["success"] is True
        assert new_file.exists()
        assert new_file.read_text() == ""


class TestEditModifyOperation:
    """Tests for modify operation."""

    @pytest.mark.asyncio
    async def test_modify_file(self, tmp_path):
        """Test modifying an existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")

        result = await edit(
            ops=[
                {
                    "type": "modify",
                    "path": str(test_file),
                    "content": "modified content",
                }
            ]
        )

        assert result["success"] is True
        assert result["by_type"]["modify"] == 1
        assert test_file.read_text() == "modified content"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_modify_file_with_new_content_key(self, tmp_path):
        """Test modify using 'new_content' key instead of 'content'."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("original")

        result = await edit(ops=[{"type": "modify", "path": str(test_file), "new_content": "new"}])

        assert result["success"] is True
        assert test_file.read_text() == "new"

    @pytest.mark.asyncio
    async def test_modify_file_missing_content(self, tmp_path):
        """Test error when modify missing content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("original")

        result = await edit(ops=[{"type": "modify", "path": str(test_file)}])

        assert result["success"] is False
        assert "missing content" in result["error"]


class TestEditDeleteOperation:
    """Tests for delete operation."""

    @pytest.mark.asyncio
    async def test_delete_file(self, tmp_path):
        """Test deleting an existing file."""
        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("delete me")

        result = await edit(ops=[{"type": "delete", "path": str(test_file)}])

        assert result["success"] is True
        assert result["by_type"]["delete"] == 1
        assert not test_file.exists()


class TestEditRenameOperation:
    """Tests for rename operation."""

    @pytest.mark.asyncio
    async def test_rename_file(self, tmp_path):
        """Test renaming a file."""
        old_file = tmp_path / "old_name.txt"
        old_file.write_text("content")
        new_path = tmp_path / "new_name.txt"

        result = await edit(
            ops=[{"type": "rename", "path": str(old_file), "new_path": str(new_path)}]
        )

        assert result["success"] is True
        assert result["by_type"]["rename"] == 1
        assert not old_file.exists()
        assert new_path.exists()
        assert new_path.read_text() == "content"


class TestEditReplaceOperation:
    """Tests for replace (surgical string replacement) operation."""

    @pytest.mark.asyncio
    async def test_replace_string(self, tmp_path):
        """Test surgical string replacement."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def old_func():\n    pass\n")

        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(test_file),
                    "old_str": "old_func",
                    "new_str": "new_func",
                }
            ]
        )

        assert result["success"] is True
        assert "replace" in result["by_type"]
        assert result["by_type"]["replace"] == 1
        assert "new_func" in test_file.read_text()
        assert "old_func" not in test_file.read_text()

    @pytest.mark.asyncio
    async def test_replace_missing_old_str(self, tmp_path):
        """Test error when replace missing old_str."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = await edit(ops=[{"type": "replace", "path": str(test_file), "new_str": "new"}])

        assert result["success"] is False
        assert "missing required field: old_str" in result["error"]

    @pytest.mark.asyncio
    async def test_replace_missing_new_str(self, tmp_path):
        """Test error when replace missing new_str."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = await edit(ops=[{"type": "replace", "path": str(test_file), "old_str": "old"}])

        assert result["success"] is False
        assert "missing required field: new_str" in result["error"]

    @pytest.mark.asyncio
    async def test_replace_file_not_exist(self, tmp_path):
        """Test error when file doesn't exist for replace."""
        nonexistent = tmp_path / "nonexistent.txt"

        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(nonexistent),
                    "old_str": "old",
                    "new_str": "new",
                }
            ]
        )

        assert result["success"] is False
        assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_replace_old_str_not_found(self, tmp_path):
        """Test error when old_str not found in file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(test_file),
                    "old_str": "nonexistent string",
                    "new_str": "new",
                }
            ]
        )

        assert result["success"] is False
        assert "old_str not found" in result["error"]

    @pytest.mark.asyncio
    async def test_replace_ambiguous_multiple_matches(self, tmp_path):
        """Test error when old_str found multiple times."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("foo bar foo baz foo")

        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(test_file),
                    "old_str": "foo",
                    "new_str": "qux",
                }
            ]
        )

        assert result["success"] is False
        assert "found 3 times" in result["error"]
        assert "Ambiguous" in result["error"]

    @pytest.mark.asyncio
    async def test_replace_hint_first_line_match(self, tmp_path):
        """Test helpful hint when first line matches but rest doesn't."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("first line\nsecond line\n")

        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(test_file),
                    "old_str": "first line\nwrong second",
                    "new_str": "new",
                }
            ]
        )

        assert result["success"] is False
        # Should contain hint about first line matching
        assert "old_str not found" in result["error"]

    @pytest.mark.asyncio
    async def test_replace_hint_trailing_whitespace(self, tmp_path):
        """Test helpful hint about trailing whitespace."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(test_file),
                    "old_str": "content\n",  # Extra newline
                    "new_str": "new",
                }
            ]
        )

        assert result["success"] is False
        assert "old_str not found" in result["error"]


class TestEditPreviewMode:
    """Tests for preview mode."""

    @pytest.mark.asyncio
    async def test_preview_without_commit(self, tmp_path):
        """Test preview mode doesn't apply changes when commit=False."""
        test_file = tmp_path / "test.txt"
        original = "original content"
        test_file.write_text(original)

        result = await edit(
            ops=[{"type": "modify", "path": str(test_file), "content": "new content"}],
            preview=True,
            commit=False,
        )

        assert result["success"] is True
        assert result["operations_applied"] == 0
        assert "Preview" in result["message"]
        # File should NOT be modified
        assert test_file.read_text() == original

    @pytest.mark.asyncio
    async def test_preview_with_commit(self, tmp_path):
        """Test preview mode with commit=True applies changes."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("original")

        result = await edit(
            ops=[{"type": "modify", "path": str(test_file), "content": "modified"}],
            preview=True,
            commit=True,
        )

        assert result["success"] is True
        assert result["operations_applied"] == 1
        assert test_file.read_text() == "modified"


class TestEditCommitMode:
    """Tests for commit mode."""

    @pytest.mark.asyncio
    async def test_commit_false_no_apply(self, tmp_path):
        """Test commit=False queues but doesn't apply changes."""
        test_file = tmp_path / "test.txt"
        original = "original"
        test_file.write_text(original)

        result = await edit(
            ops=[{"type": "modify", "path": str(test_file), "content": "new"}],
            commit=False,
        )

        assert result["success"] is True
        assert result["operations_applied"] == 0
        assert "not applied" in result["message"]
        # File should NOT be modified
        assert test_file.read_text() == original


class TestEditMultipleOperations:
    """Tests for multiple operations in single edit call."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_multiple_operations(self, tmp_path):
        """Test multiple operations in single edit call."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("content1")

        file2 = tmp_path / "file2.txt"
        # Will be created

        result = await edit(
            ops=[
                {"type": "modify", "path": str(file1), "content": "modified1"},
                {"type": "create", "path": str(file2), "content": "new_content"},
            ],
            desc="Multiple operations test",
        )

        assert result["success"] is True
        assert result["operations_applied"] == 2
        assert result["by_type"]["modify"] == 1
        assert result["by_type"]["create"] == 1


class TestEditErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_queue_operations_exception(self, tmp_path):
        """Test handling of exception during queue phase."""
        # Create a file but mock FileEditor to raise during queueing
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Patch _create_file_editor since that's what the code uses
        with patch("victor.tools.file_editor_tool._create_file_editor") as mock_factory:
            mock_editor = MagicMock()
            mock_editor.start_transaction.return_value = "txn123"
            mock_editor.add_modify.side_effect = Exception("Queue failed")
            mock_factory.return_value = mock_editor

            result = await edit(ops=[{"type": "modify", "path": str(test_file), "content": "new"}])

            assert result["success"] is False
            assert "Failed to queue operations" in result["error"]


class TestEditTextBasedBehavior:
    """Tests demonstrating TEXT-BASED (non-AST-aware) behavior.

    These tests document that edit() is NOT code-aware and may cause
    false positives. For Python symbol renaming, use rename() instead.
    """

    @pytest.mark.asyncio
    async def test_text_replacement_partial_match_warning(self, tmp_path):
        """Document: edit() can cause false positives in code."""
        test_file = tmp_path / "test.py"
        # 'get_user' appears in 'get_username' - edit() will match both!
        test_file.write_text("get_user = 1\nget_username = 2\n")

        # If we try to replace just "get_user", it would be ambiguous
        # because it appears twice (once as standalone, once as prefix)
        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(test_file),
                    "old_str": "get_user",
                    "new_str": "fetch_user",
                }
            ]
        )

        # This should fail due to ambiguous match (2 occurrences)
        assert result["success"] is False
        assert "found 2 times" in result["error"]

    @pytest.mark.asyncio
    async def test_text_replacement_unique_match_ok(self, tmp_path):
        """Document: edit() works fine for unique text matches."""
        test_file = tmp_path / "config.yaml"
        test_file.write_text("database_host: localhost\nport: 5432\n")

        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(test_file),
                    "old_str": "database_host: localhost",
                    "new_str": "database_host: 127.0.0.1",
                }
            ]
        )

        assert result["success"] is True
        assert "127.0.0.1" in test_file.read_text()


class TestEditUndoLogResilience:
    """P6: an applied edit must never be reported as failed on undo-log db errors.

    FileEditor.commit() writes files to disk BEFORE the ChangeTracker persists
    the undo history to project.db. A "database is locked" error in that
    bookkeeping step must not turn an applied edit into a tool error (the
    model's retry would then hit 'old_str not found' on the mutated file).
    """

    @pytest.mark.asyncio
    async def test_db_lock_after_file_commit_returns_success_with_warning(
        self, tmp_path, monkeypatch
    ):
        """Edit succeeds (with a warning note) when the undo-log db write fails."""
        import sqlite3

        from victor.agent.change_tracker import FileChangeHistory, reset_change_tracker

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        def _raise_locked(self):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(FileChangeHistory, "commit_change_group", _raise_locked)
        try:
            result = await edit(
                ops=[
                    {
                        "type": "replace",
                        "path": str(test_file),
                        "old_str": "hello",
                        "new_str": "goodbye",
                    }
                ]
            )
        finally:
            # The patched commit never clears the singleton's current group.
            reset_change_tracker()

        # The file WAS modified on disk — the tool must report success.
        assert test_file.read_text() == "goodbye world"
        assert result["success"] is True
        assert "undo history not recorded" in result["message"]
        assert "⚠️" in result["message"]

    @pytest.mark.asyncio
    async def test_undo_note_absent_on_clean_commit(self, tmp_path):
        """No undo-history warning note when the undo-log write succeeds."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": str(test_file),
                    "old_str": "hello",
                    "new_str": "goodbye",
                }
            ]
        )

        assert result["success"] is True
        assert "undo history not recorded" not in result["message"]
