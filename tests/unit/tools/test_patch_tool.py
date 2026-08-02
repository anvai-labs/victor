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

"""Tests for the patch tool."""

import os
import tempfile
import pytest

from victor.tools.patch_tool import (
    Hunk,
    PatchFile,
    parse_unified_diff,
    apply_patch_to_content,
    patch,
)


class TestParseUnifiedDiff:
    """Tests for parse_unified_diff function."""

    def test_parse_simple_diff(self):
        """Test parsing a simple unified diff."""
        diff = """--- a/hello.py
+++ b/hello.py
@@ -1,3 +1,4 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
+    return True
"""
        patches = parse_unified_diff(diff)

        assert len(patches) == 1
        patch = patches[0]
        assert patch.old_path == "hello.py"
        assert patch.new_path == "hello.py"
        assert len(patch.hunks) == 1
        assert patch.hunks[0].old_start == 1
        assert patch.hunks[0].old_count == 3
        assert patch.hunks[0].new_start == 1
        assert patch.hunks[0].new_count == 4

    def test_parse_new_file(self):
        """Test parsing a diff for a new file."""
        diff = """--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+# New file
+def new_func():
+    pass
"""
        patches = parse_unified_diff(diff)

        assert len(patches) == 1
        patch = patches[0]
        assert patch.is_new_file is True
        assert patch.old_path is None
        assert patch.new_path == "new_file.py"

    def test_parse_deleted_file(self):
        """Test parsing a diff for a deleted file."""
        diff = """--- a/old_file.py
+++ /dev/null
@@ -1,3 +0,0 @@
-# Old file
-def old_func():
-    pass
"""
        patches = parse_unified_diff(diff)

        assert len(patches) == 1
        patch = patches[0]
        assert patch.is_deleted is True
        assert patch.old_path == "old_file.py"
        assert patch.new_path is None

    def test_parse_multiple_hunks(self):
        """Test parsing diff with multiple hunks."""
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def func1():
-    return 1
+    return 10
@@ -10,3 +10,3 @@
 def func2():
-    return 2
+    return 20
"""
        patches = parse_unified_diff(diff)

        assert len(patches) == 1
        assert len(patches[0].hunks) == 2
        assert patches[0].hunks[0].old_start == 1
        assert patches[0].hunks[1].old_start == 10

    def test_parse_multiple_files(self):
        """Test parsing diff with multiple files."""
        diff = """--- a/file1.py
+++ b/file1.py
@@ -1 +1 @@
-old1
+new1
--- a/file2.py
+++ b/file2.py
@@ -1 +1 @@
-old2
+new2
"""
        patches = parse_unified_diff(diff)

        assert len(patches) == 2
        assert patches[0].old_path == "file1.py"
        assert patches[1].old_path == "file2.py"


class TestApplyPatchToContent:
    """Tests for apply_patch_to_content function."""

    def test_apply_simple_change(self):
        """Test applying a simple line change."""
        content = "line1\nline2\nline3"
        hunks = [
            Hunk(
                old_start=1,
                old_count=3,
                new_start=1,
                new_count=3,
                lines=[" line1", "-line2", "+line2_modified", " line3"],
            )
        ]

        success, result, warnings = apply_patch_to_content(content, hunks)

        assert success is True
        assert result == "line1\nline2_modified\nline3"
        assert len(warnings) == 0

    def test_apply_insertion(self):
        """Test applying an insertion."""
        content = "line1\nline3"
        hunks = [
            Hunk(
                old_start=1,
                old_count=2,
                new_start=1,
                new_count=3,
                lines=[" line1", "+line2", " line3"],
            )
        ]

        success, result, warnings = apply_patch_to_content(content, hunks)

        assert success is True
        assert result == "line1\nline2\nline3"

    def test_apply_deletion(self):
        """Test applying a deletion."""
        content = "line1\nline2\nline3"
        hunks = [
            Hunk(
                old_start=1,
                old_count=3,
                new_start=1,
                new_count=2,
                lines=[" line1", "-line2", " line3"],
            )
        ]

        success, result, warnings = apply_patch_to_content(content, hunks)

        assert success is True
        assert result == "line1\nline3"

    def test_apply_with_context_mismatch(self):
        """Test applying patch with mismatched context."""
        content = "different\nline2\nline3"
        hunks = [
            Hunk(
                old_start=1,
                old_count=3,
                new_start=1,
                new_count=3,
                lines=[
                    " line1",
                    "-line2",
                    "+line2_modified",
                    " line3",
                ],  # line1 doesn't match
            )
        ]

        success, result, warnings = apply_patch_to_content(content, hunks)

        # Should fail due to context mismatch
        assert success is False
        assert len(warnings) > 0

    def test_apply_with_fuzz(self):
        """Test applying patch with fuzz factor."""
        # Content is shifted by one line
        content = "extra_line\nline1\nline2\nline3"
        hunks = [
            Hunk(
                old_start=1,  # Original expects line1 at line 1
                old_count=3,
                new_start=1,
                new_count=3,
                lines=[" line1", "-line2", "+line2_modified", " line3"],
            )
        ]

        success, result, warnings = apply_patch_to_content(content, hunks, fuzz=5)

        assert success is True
        assert "line2_modified" in result


class TestHunk:
    """Tests for Hunk dataclass."""

    def test_create_hunk(self):
        """Test creating a Hunk."""
        hunk = Hunk(
            old_start=10,
            old_count=5,
            new_start=10,
            new_count=6,
            lines=[" context", "-old", "+new", " context"],
        )

        assert hunk.old_start == 10
        assert hunk.old_count == 5
        assert hunk.new_start == 10
        assert hunk.new_count == 6
        assert len(hunk.lines) == 4


class TestPatchFile:
    """Tests for PatchFile dataclass."""

    def test_create_patch_file(self):
        """Test creating a PatchFile."""
        patch = PatchFile(
            old_path="test.py",
            new_path="test.py",
            hunks=[],
            is_new_file=False,
            is_deleted=False,
        )

        assert patch.old_path == "test.py"
        assert patch.new_path == "test.py"
        assert patch.is_binary is False


@pytest.mark.asyncio
class TestApplyPatch:
    """Tests for apply_patch function."""

    async def test_apply_patch_to_file(self):
        """Test applying a patch to an actual file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("def hello():\n    print('Hello')\n")

            diff_content = f"""--- a/{test_file}
+++ b/{test_file}
@@ -1,2 +1,3 @@
 def hello():
-    print('Hello')
+    print('Hello, World!')
+    return True
"""
            result = await patch(patch_content=diff_content, file_path=test_file, backup=False)

            assert result["success"] is True
            # Path may be resolved differently on macOS (/var vs /private/var)
            assert len(result["files_modified"]) == 1
            assert "test.py" in result["files_modified"][0]

            # Verify file content
            with open(test_file, "r") as f:
                content = f.read()
            assert "Hello, World!" in content
            assert "return True" in content

    async def test_apply_patch_dry_run(self):
        """Test dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.py")
            original_content = "old content"
            with open(test_file, "w") as f:
                f.write(original_content)

            diff_content = f"""--- a/{test_file}
+++ b/{test_file}
@@ -1 +1 @@
-old content
+new content
"""
            result = await patch(patch_content=diff_content, file_path=test_file, dry_run=True)

            assert "preview" in result

            # Verify file was NOT modified
            with open(test_file, "r") as f:
                content = f.read()
            assert content == original_content

    async def test_apply_patch_creates_backup(self):
        """Test that backup is created when backup=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("original")

            diff_content = f"""--- a/{test_file}
+++ b/{test_file}
@@ -1 +1 @@
-original
+modified
"""
            await patch(patch_content=diff_content, file_path=test_file, backup=True)

            # Check backup exists
            backup_file = test_file + ".orig"
            assert os.path.exists(backup_file)
            with open(backup_file, "r") as f:
                assert f.read() == "original"

    async def test_apply_patch_creates_new_file(self):
        """Test creating a new file via patch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_file = os.path.join(tmpdir, "new_file.py")

            diff_content = f"""--- /dev/null
+++ b/{new_file}
@@ -0,0 +1,2 @@
+# New file
+print('Hello')
"""
            result = await patch(patch_content=diff_content, backup=False)

            assert result["success"] is True
            assert os.path.exists(new_file)
            with open(new_file, "r") as f:
                content = f.read()
            assert "# New file" in content

    async def test_apply_patch_invalid(self):
        """Test handling invalid patch."""
        result = await patch(patch_content="this is not a valid patch")

        assert result["success"] is False
        assert "error" in result or "No valid patches" in str(result.get("error", ""))


@pytest.mark.asyncio
class TestCreatePatch:
    """Tests for create_patch function."""

    async def test_create_patch_for_existing_file(self):
        """Test creating a patch for an existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("line1\nline2\nline3")

            result = await patch(
                operation="create",
                file_path=test_file,
                new_content="line1\nline2_modified\nline3",
            )

            assert result["success"] is True
            assert "patch" in result
            assert "-line2" in result["patch"]
            assert "+line2_modified" in result["patch"]
            assert result["stats"]["additions"] == 1
            assert result["stats"]["deletions"] == 1

    async def test_create_patch_for_new_file(self):
        """Test creating a patch for a new file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_file = os.path.join(tmpdir, "nonexistent.py")

            result = await patch(
                operation="create",
                file_path=new_file,
                new_content="# New content\nprint('hello')",
            )

            assert result["success"] is True
            assert "patch" in result
            assert "+# New content" in result["patch"]

    async def test_create_patch_context_lines(self):
        """Test that context lines are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.py")
            with open(test_file, "w") as f:
                f.write("line1\nline2\nline3\nline4\nline5")

            result = await patch(
                operation="create",
                file_path=test_file,
                new_content="line1\nline2\nline3_modified\nline4\nline5",
                context_lines=2,
            )

            assert result["success"] is True
            # Should include context lines around the change
            assert "line2" in result["patch"]
            assert "line4" in result["patch"]
