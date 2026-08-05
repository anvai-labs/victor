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

"""Unit tests for victor.contrib.editing package."""

import pytest
from pathlib import Path

from victor.contrib.editing import DiffEditor
from victor.framework.vertical_protocols import (
    EditOperation,
    EditResult,
    EditValidationResult,
)


@pytest.fixture
def temp_file(tmp_path: Path) -> Path:
    """Create a temporary test file."""
    return tmp_path / "test.txt"


@pytest.fixture
def temp_python_file(tmp_path: Path) -> Path:
    """Create a temporary Python test file."""
    return tmp_path / "test.py"


class TestDiffEditor:
    """Test DiffEditor implementation."""

    def test_editor_info(self) -> None:
        """Test editor metadata retrieval."""
        editor = DiffEditor()
        info = editor.get_editor_info()

        assert info["name"] == "DiffEditor"
        assert info["version"] == "1.0.0"
        assert "case_sensitive" in info
        assert "capabilities" in info
        assert "string_replacement" in info["capabilities"]

    @pytest.mark.asyncio
    async def test_edit_file_simple_replace(self, temp_file: Path) -> None:
        """Test simple string replacement."""
        temp_file.write_text("hello world")

        editor = DiffEditor()
        result = await editor.edit_file(
            file_path=str(temp_file),
            edits=[EditOperation(old_str="hello", new_str="hi")],
        )

        assert result.success
        assert result.edits_applied == 1
        assert result.edits_failed == 0
        assert temp_file.read_text() == "hi world"

    @pytest.mark.asyncio
    async def test_edit_file_multiple_edits(self, temp_file: Path) -> None:
        """Test multiple edits in one call."""
        temp_file.write_text("hello world test")

        editor = DiffEditor()
        result = await editor.edit_file(
            file_path=str(temp_file),
            edits=[
                EditOperation(old_str="hello", new_str="hi"),
                EditOperation(old_str="world", new_str="earth"),
                EditOperation(old_str="test", new_str="example"),
            ],
        )

        assert result.success
        assert result.edits_applied == 3
        assert temp_file.read_text() == "hi earth example"

    @pytest.mark.asyncio
    async def test_edit_file_not_found(self, temp_file: Path) -> None:
        """Test error when file doesn't exist."""
        editor = DiffEditor()
        result = await editor.edit_file(
            file_path=str(temp_file / "nonexistent.txt"),
            edits=[EditOperation(old_str="x", new_str="y")],
        )

        assert not result.success
        assert "File not found" in result.error

    @pytest.mark.asyncio
    async def test_edit_file_string_not_found(self, temp_file: Path) -> None:
        """Test error when old_str not in file."""
        temp_file.write_text("hello world")

        editor = DiffEditor()
        result = await editor.edit_file(
            file_path=str(temp_file),
            edits=[EditOperation(old_str="goodbye", new_str="hi")],
        )

        assert not result.success
        assert result.edits_failed == 1
        assert "String not found" in result.error

    @pytest.mark.asyncio
    async def test_edit_file_multiple_occurrences(self, temp_file: Path) -> None:
        """Test error when old_str appears multiple times without allow_multiple."""
        temp_file.write_text("hello hello hello")

        editor = DiffEditor()
        result = await editor.edit_file(
            file_path=str(temp_file),
            edits=[EditOperation(old_str="hello", new_str="hi", allow_multiple=False)],
        )

        assert not result.success
        assert "appears 3 times" in result.error

    @pytest.mark.asyncio
    async def test_edit_file_allow_multiple(self, temp_file: Path) -> None:
        """Test replacing all occurrences with allow_multiple=True."""
        temp_file.write_text("hello hello hello")

        editor = DiffEditor()
        result = await editor.edit_file(
            file_path=str(temp_file),
            edits=[EditOperation(old_str="hello", new_str="hi", allow_multiple=True)],
        )

        assert result.success
        assert result.edits_applied == 1
        assert temp_file.read_text() == "hi hi hi"

    @pytest.mark.asyncio
    async def test_validate_edit_valid(self, temp_file: Path) -> None:
        """Test validation of valid edit."""
        temp_file.write_text("hello world")

        editor = DiffEditor()
        result = await editor.validate_edit(
            file_path=str(temp_file),
            old_str="hello",
            new_str="hi",
        )

        assert result.valid
        assert result.old_str_found
        assert result.match_count == 1

    @pytest.mark.asyncio
    async def test_validate_edit_not_found(self, temp_file: Path) -> None:
        """Test validation when old_str not found."""
        temp_file.write_text("hello world")

        editor = DiffEditor()
        result = await editor.validate_edit(
            file_path=str(temp_file),
            old_str="goodbye",
            new_str="hi",
        )

        assert not result.valid
        assert not result.old_str_found
        assert "String not found" in result.error

    @pytest.mark.asyncio
    async def test_preview_mode(self, temp_file: Path) -> None:
        """Test preview mode doesn't modify file."""
        temp_file.write_text("hello world")

        editor = DiffEditor()
        result = await editor.edit_file(
            file_path=str(temp_file),
            edits=[EditOperation(old_str="hello", new_str="hi")],
            preview=True,
        )

        assert result.success
        assert result.preview is not None
        assert result.preview == "hi world"
        # File should be unchanged
        assert temp_file.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_case_sensitive_default(self, temp_file: Path) -> None:
        """Test default case-sensitive matching."""
        temp_file.write_text("Hello WORLD")

        editor = DiffEditor()  # case_sensitive=True by default
        result = await editor.edit_file(
            file_path=str(temp_file),
            edits=[
                EditOperation(old_str="hello", new_str="hi"),  # Won't match
                EditOperation(old_str="Hello", new_str="Hi"),  # Will match
            ],
        )

        assert result.edits_applied == 1
        assert result.edits_failed == 1
        assert temp_file.read_text() == "Hi WORLD"

    @pytest.mark.asyncio
    async def test_case_insensitive(self, temp_file: Path) -> None:
        """Test case-insensitive matching."""
        temp_file.write_text("Hello WORLD")

        editor = DiffEditor(case_sensitive=False)
        result = await editor.edit_file(
            file_path=str(temp_file),
            edits=[
                EditOperation(old_str="hello", new_str="hi"),  # Will match now
                EditOperation(old_str="WORLD", new_str="earth"),  # Will match
            ],
        )

        assert result.edits_applied == 2
        # Case-insensitive matching finds "Hello" and "WORLD"
        # but replaces with exact new_str ("hi" and "earth")
        assert temp_file.read_text() == "hi earth"

    @pytest.mark.asyncio
    async def test_empty_edits_list(self, temp_file: Path) -> None:
        """Test with empty edits list."""
        temp_file.write_text("hello world")

        editor = DiffEditor()
        result = await editor.edit_file(
            file_path=str(temp_file),
            edits=[],
        )

        assert result.success
        assert result.edits_applied == 0

    @pytest.mark.asyncio
    async def test_edit_python_file(self, temp_python_file: Path) -> None:
        """Test editing a Python file."""
        temp_python_file.write_text("""
def foo():
    '''Old docstring.'''
    pass
def bar():
    return None
""")

        editor = DiffEditor()
        result = await editor.edit_file(
            file_path=str(temp_python_file),
            edits=[
                EditOperation(
                    old_str="'''Old docstring.'''",
                    new_str="'''New docstring.'''",
                ),
                EditOperation(
                    old_str="def foo():",
                    new_str="def foo():\n    '''Function docstring.'''",
                ),
            ],
        )

        assert result.success
        assert result.edits_applied == 2
        assert "New docstring" in temp_python_file.read_text()


class TestEditOperation:
    """Test EditOperation dataclass."""

    def test_edit_operation_basic(self) -> None:
        """Test basic edit operation creation."""
        op = EditOperation(old_str="old", new_str="new")
        assert op.old_str == "old"
        assert op.new_str == "new"
        assert op.start_line is None
        assert op.end_line is None
        assert op.allow_multiple is False

    def test_edit_operation_with_context(self) -> None:
        """Test edit operation with line context."""
        op = EditOperation(
            old_str="old",
            new_str="new",
            start_line=10,
            end_line=20,
            allow_multiple=True,
        )
        assert op.start_line == 10
        assert op.end_line == 20
        assert op.allow_multiple is True

    def test_edit_operation_empty_old_str_raises(self) -> None:
        """Test that empty old_str raises validation error."""
        with pytest.raises(ValueError, match="old_str cannot be empty"):
            EditOperation(old_str="", new_str="new")

    def test_edit_operation_negative_line_raises(self) -> None:
        """Test that negative line numbers raise validation error."""
        with pytest.raises(ValueError, match="start_line must be >= 0"):
            EditOperation(
                old_str="old",
                new_str="new",
                start_line=-1,
            )

    def test_edit_operation_end_before_start_raises(self) -> None:
        """Test that end_line < start_line raises validation error."""
        with pytest.raises(ValueError, match="start_line.*end_line"):
            EditOperation(
                old_str="old",
                new_str="new",
                start_line=10,
                end_line=5,
            )


class TestEditResult:
    """Test EditResult dataclass."""

    def test_edit_result_success(self) -> None:
        """Test successful edit result."""
        result = EditResult(
            success=True,
            file_path="/path/to/file.txt",
            edits_applied=3,
            edits_failed=1,
        )

        assert result.success
        assert result.file_path == "/path/to/file.txt"
        assert result.total_edits == 4

    def test_edit_result_failure(self) -> None:
        """Test failed edit result."""
        result = EditResult(
            success=False,
            file_path="/path/to/file.txt",
            edits_applied=0,
            error="File not found",
        )

        assert not result.success
        assert result.error == "File not found"
        assert result.total_edits == 0


class TestEditValidationResult:
    """Test EditValidationResult dataclass."""

    def test_validation_result_valid(self) -> None:
        """Test valid edit result."""
        result = EditValidationResult(
            valid=True,
            file_path="/path/to/file.txt",
            old_str_found=True,
            match_count=1,
        )

        assert result.valid
        assert result.is_safe_to_apply

    def test_validation_result_invalid(self) -> None:
        """Test invalid edit result."""
        result = EditValidationResult(
            valid=False,
            file_path="/path/to/file.txt",
            old_str_found=False,
            match_count=0,
            error="String not found",
        )

        assert not result.valid
        assert not result.is_safe_to_apply

    def test_validation_result_multiple_matches(self) -> None:
        """Test validation result with multiple matches."""
        result = EditValidationResult(
            valid=False,
            file_path="/path/to/file.txt",
            old_str_found=True,
            match_count=3,
            error="appears 3 times",
        )

        assert not result.valid
        assert result.old_str_found
        assert result.match_count == 3
        assert not result.is_safe_to_apply
