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

"""Tests for file_editor_tool module.

These tests require victor-coding package to be installed and are
marked as integration tests.
"""

import pytest
import tempfile
from pathlib import Path

from victor.tools.file_editor_tool import edit


def _check_editor():
    """Check at runtime (not import time) if enhanced editor is available."""
    try:
        from victor.tools.file_editor_tool import _is_file_editor_available

        return _is_file_editor_available()
    except Exception:
        return False


# These tests require the enhanced editor (EditorProtocol registered via vertical)
pytestmark = [pytest.mark.integration]


@pytest.fixture(autouse=True)
def skip_without_editor():
    """Skip if enhanced editor is not registered (runtime check each test)."""
    if not _check_editor():
        pytest.skip("Enhanced editor not registered (requires vertical)")


class TestEditFiles:
    """Tests for edit_files function."""

    @pytest.mark.asyncio
    async def test_edit_files_empty_operations(self):
        """Test editing with empty operations."""
        result = await edit(ops=[])
        assert result["success"] is False
        assert "No operations" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_files_invalid_operation(self):
        """Test editing with invalid operation type."""
        result = await edit(ops=[{"path": "/tmp/test.txt"}])
        assert result["success"] is False
        assert "type" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_edit_files_create_success(self):
        """Test successful file creation."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/new_file.txt"

        try:
            result = await edit(
                ops=[{"type": "create", "path": temp_path, "content": "Hello World"}]
            )
            assert result["success"] is True
            assert result["operations_queued"] == 1

            # Verify file content
            with open(temp_path) as f:
                content = f.read()
            assert content == "Hello World"
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    @pytest.mark.asyncio
    async def test_edit_files_modify_nonexistent_file(self):
        """Test modifying nonexistent file."""
        result = await edit(
            ops=[
                {
                    "type": "modify",
                    "path": "/nonexistent/file.txt",
                    "content": "new content",
                }
            ]
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_edit_files_preview_mode(self):
        """Test preview mode."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/test.txt"
        Path(temp_path).write_text("original content")

        try:
            result = await edit(
                ops=[{"type": "modify", "path": temp_path, "content": "updated content"}],
                preview=True,
                commit=False,
            )
            # Preview should succeed but not apply changes
            assert result["success"] is True

            # Verify file content unchanged
            with open(temp_path) as f:
                content = f.read()
            assert content == "original content"
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    @pytest.mark.asyncio
    async def test_edit_files_json_string_operations(self):
        """Test passing operations as JSON string."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/json_test.txt"

        try:
            import json

            ops_json = json.dumps([{"type": "create", "path": temp_path, "content": "From JSON"}])
            result = await edit(ops=ops_json)
            assert result["success"] is True
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    @pytest.mark.asyncio
    async def test_edit_files_invalid_json_string(self):
        """Test invalid JSON string operations."""
        result = await edit(ops="not valid json")
        assert result["success"] is False
        assert "JSON" in result["error"]


class TestReplaceOperation:
    """Tests for the 'replace' operation type (surgical str_replace)."""

    @pytest.mark.asyncio
    async def test_replace_basic_success(self):
        """Test basic string replacement in a file."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/replace_test.py"
        Path(temp_path).write_text("def foo():\n    return 1\n")

        try:
            result = await edit(
                ops=[
                    {
                        "type": "replace",
                        "path": temp_path,
                        "old_str": "def foo():\n    return 1",
                        "new_str": "def foo():\n    return 42",
                    }
                ]
            )
            assert result["success"] is True
            assert result["operations_applied"] == 1

            # Verify content changed
            content = Path(temp_path).read_text()
            assert "return 42" in content
            assert "return 1" not in content
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    @pytest.mark.asyncio
    async def test_replace_insert_new_code(self):
        """Test inserting new code after existing code."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/insert_test.py"
        original = """class Foo:
    @property
    def name(self):
        return "foo"
"""
        Path(temp_path).write_text(original)

        try:
            result = await edit(
                ops=[
                    {
                        "type": "replace",
                        "path": temp_path,
                        "old_str": '    @property\n    def name(self):\n        return "foo"',
                        "new_str": '    @property\n    def name(self):\n        return "foo"\n\n    @property\n    def version(self):\n        return "1.0.0"',
                    }
                ]
            )
            assert result["success"] is True

            content = Path(temp_path).read_text()
            assert "def version(self)" in content
            assert 'return "1.0.0"' in content
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    @pytest.mark.asyncio
    async def test_replace_old_str_not_found(self):
        """Test error when old_str is not found in file."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/not_found_test.py"
        Path(temp_path).write_text("def bar():\n    pass\n")

        try:
            result = await edit(
                ops=[
                    {
                        "type": "replace",
                        "path": temp_path,
                        "old_str": "def foo():",  # Does not exist
                        "new_str": "def foo_new():",
                    }
                ]
            )
            assert result["success"] is False
            assert "not found" in result["error"].lower()
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    @pytest.mark.asyncio
    async def test_replace_multiple_occurrences_error(self):
        """Test error when old_str appears multiple times."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/multiple_test.py"
        Path(temp_path).write_text("foo\nbar\nfoo\n")  # "foo" appears twice

        try:
            result = await edit(
                ops=[
                    {
                        "type": "replace",
                        "path": temp_path,
                        "old_str": "foo",
                        "new_str": "baz",
                    }
                ]
            )
            # Should fail because "foo" is ambiguous
            assert result["success"] is False
            assert "multiple" in result["error"].lower() or "ambiguous" in result["error"].lower()
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    @pytest.mark.asyncio
    async def test_replace_missing_old_str(self):
        """Test error when old_str is missing from operation."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/missing_old_test.py"
        Path(temp_path).write_text("content")

        try:
            result = await edit(
                ops=[
                    {
                        "type": "replace",
                        "path": temp_path,
                        "new_str": "new content",  # Missing old_str
                    }
                ]
            )
            assert result["success"] is False
            assert "old_str" in result["error"].lower()
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    @pytest.mark.asyncio
    async def test_replace_missing_new_str(self):
        """Test error when new_str is missing from operation."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/missing_new_test.py"
        Path(temp_path).write_text("old content")

        try:
            result = await edit(
                ops=[
                    {
                        "type": "replace",
                        "path": temp_path,
                        "old_str": "old content",  # Missing new_str
                    }
                ]
            )
            assert result["success"] is False
            assert "new_str" in result["error"].lower()
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()

    @pytest.mark.asyncio
    async def test_replace_nonexistent_file(self):
        """Test error when file does not exist."""
        result = await edit(
            ops=[
                {
                    "type": "replace",
                    "path": "/nonexistent/path/file.py",
                    "old_str": "foo",
                    "new_str": "bar",
                }
            ]
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_replace_preserves_rest_of_file(self):
        """Test that replace only modifies the matched portion."""
        temp_dir = tempfile.mkdtemp()
        temp_path = f"{temp_dir}/preserve_test.py"
        original = """# Header comment
def first():
    pass

def second():
    return 2

# Footer comment
"""
        Path(temp_path).write_text(original)

        try:
            result = await edit(
                ops=[
                    {
                        "type": "replace",
                        "path": temp_path,
                        "old_str": "def second():\n    return 2",
                        "new_str": "def second():\n    return 42",
                    }
                ]
            )
            assert result["success"] is True

            content = Path(temp_path).read_text()
            # Changed part
            assert "return 42" in content
            # Preserved parts
            assert "# Header comment" in content
            assert "def first():" in content
            assert "# Footer comment" in content
        finally:
            Path(temp_path).unlink(missing_ok=True)
            Path(temp_dir).rmdir()
