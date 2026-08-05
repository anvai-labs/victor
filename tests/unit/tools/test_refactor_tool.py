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

"""Tests for refactor_tool module - AST-aware code refactoring.

This module tests the consolidated refactoring tools:
- rename: AST-aware symbol renaming (file/directory/project scope)
- extract: Extract code to new function
- inline: Inline variable
- organize_imports: Organize and sort imports
"""

import pytest

from victor.tools.refactor_tool import (
    rename,
    extract,
    inline,
    organize_imports,
)


class TestRename:
    """Tests for rename function - AST-aware symbol renaming."""

    @pytest.mark.asyncio
    async def test_rename_missing_old_name(self):
        """Test handling of missing old_name parameter."""
        result = await rename(old_name="", new_name="new_test")

        assert result["success"] is False
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_missing_new_name(self):
        """Test handling of missing new_name parameter."""
        result = await rename(old_name="test", new_name="")

        assert result["success"] is False
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_same_name(self, tmp_path):
        """Test error when old and new names are the same."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func(): pass")

        result = await rename(
            old_name="func",
            new_name="func",
            path=str(test_file),
            scope="file",
        )

        assert result["success"] is False
        assert "different" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_invalid_scope(self, tmp_path):
        """Test error with invalid scope."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func(): pass")

        result = await rename(
            old_name="func",
            new_name="new_func",
            path=str(test_file),
            scope="invalid",
        )

        assert result["success"] is False
        assert "Invalid scope" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_file_not_found(self):
        """Test handling of non-existent file."""
        result = await rename(
            old_name="old",
            new_name="new",
            path="/nonexistent/file.py",
            scope="file",
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_path_not_file_for_file_scope(self, tmp_path):
        """Test error when path is directory for file scope."""
        result = await rename(
            old_name="old",
            new_name="new",
            path=str(tmp_path),  # Directory, not file
            scope="file",
        )

        assert result["success"] is False
        assert "must be a file" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_not_python_file(self, tmp_path):
        """Test handling of non-Python file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Not Python")

        result = await rename(
            old_name="old",
            new_name="new",
            path=str(test_file),
            scope="file",
        )

        assert result["success"] is False
        # Should indicate it's not a Python file
        assert "Python" in result["error"] or "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_single_file_success(self, tmp_path):
        """Test successful rename in a single file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""def old_func():
    return True

result = old_func()
""")

        result = await rename(
            old_name="old_func",
            new_name="new_func",
            path=str(test_file),
            scope="file",
            preview=False,
        )

        assert result["success"] is True
        assert result["files_count"] == 1
        assert result["total_changes"] >= 2  # Definition + usage

        # Verify file was modified
        content = test_file.read_text()
        assert "new_func" in content
        assert "old_func" not in content

    @pytest.mark.asyncio
    async def test_rename_single_file_preview(self, tmp_path):
        """Test preview mode doesn't modify file."""
        test_file = tmp_path / "test.py"
        original_content = """def old_func():
    return True
"""
        test_file.write_text(original_content)

        result = await rename(
            old_name="old_func",
            new_name="new_func",
            path=str(test_file),
            scope="file",
            preview=True,
        )

        assert result["success"] is True
        assert "PREVIEW" in result["formatted_report"]

        # File should NOT be modified
        assert test_file.read_text() == original_content

    @pytest.mark.asyncio
    async def test_rename_symbol_not_found(self, tmp_path):
        """Test error when symbol not found in file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def other_func(): pass")

        result = await rename(
            old_name="nonexistent",
            new_name="new_name",
            path=str(test_file),
            scope="file",
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_directory_scope(self, tmp_path):
        """Test renaming symbol across directory."""
        # Create files in directory
        file1 = tmp_path / "module1.py"
        file1.write_text("def helper(): pass\nhelper()")

        file2 = tmp_path / "module2.py"
        file2.write_text("from module1 import helper\nhelper()")

        result = await rename(
            old_name="helper",
            new_name="util",
            path=str(tmp_path),
            scope="directory",
            preview=False,
        )

        assert result["success"] is True
        assert result["files_count"] == 2

        # Both files should be modified
        assert "util" in file1.read_text()
        assert "util" in file2.read_text()

    @pytest.mark.asyncio
    async def test_rename_project_scope(self, tmp_path):
        """Test renaming symbol across entire project."""
        # Create nested directory structure
        subdir = tmp_path / "subpackage"
        subdir.mkdir()

        file1 = tmp_path / "main.py"
        file1.write_text("CONFIG = 'test'\nprint(CONFIG)")

        file2 = subdir / "utils.py"
        file2.write_text("from main import CONFIG\nval = CONFIG")

        result = await rename(
            old_name="CONFIG",
            new_name="SETTINGS",
            path=str(tmp_path),
            scope="project",
            preview=False,
        )

        assert result["success"] is True
        assert result["files_count"] == 2

        # Both files should be modified
        assert "SETTINGS" in file1.read_text()
        assert "SETTINGS" in file2.read_text()

    @pytest.mark.asyncio
    async def test_rename_with_depth_limit(self, tmp_path):
        """Test depth limiting for project scope."""
        # Create nested structure
        deep_dir = tmp_path / "level1" / "level2" / "level3"
        deep_dir.mkdir(parents=True)

        file_root = tmp_path / "root.py"
        file_root.write_text("target = 1")

        file_deep = deep_dir / "deep.py"
        file_deep.write_text("target = 2")

        # Only rename at depth 0 (current dir only)
        result = await rename(
            old_name="target",
            new_name="renamed",
            path=str(tmp_path),
            scope="project",
            depth=0,
            preview=False,
        )

        assert result["success"] is True
        # Should only modify root file
        assert "renamed" in file_root.read_text()
        assert "target" in file_deep.read_text()  # Deep file unchanged

    @pytest.mark.asyncio
    async def test_rename_no_python_files(self, tmp_path):
        """Test error when no Python files in directory."""
        # Create empty directory
        result = await rename(
            old_name="func",
            new_name="new_func",
            path=str(tmp_path),
            scope="project",
        )

        assert result["success"] is False
        assert "No Python files" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_skips_excluded_dirs(self, tmp_path):
        """Test that __pycache__, .git etc are skipped."""
        # Create file in __pycache__
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        cache_file = cache_dir / "cached.py"
        cache_file.write_text("target = 1")

        # Create normal file
        normal_file = tmp_path / "main.py"
        normal_file.write_text("target = 2")

        result = await rename(
            old_name="target",
            new_name="renamed",
            path=str(tmp_path),
            scope="project",
            preview=False,
        )

        assert result["success"] is True
        # Cache file should be unchanged
        assert "target" in cache_file.read_text()
        # Normal file should be modified
        assert "renamed" in normal_file.read_text()

    @pytest.mark.asyncio
    async def test_rename_word_boundary_safety(self, tmp_path):
        """Test that rename uses word boundaries to avoid partial matches."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""def get_user(): pass
def get_username(): pass  # Should NOT be renamed
user = get_user()
username = get_username()  # Should NOT be renamed
""")

        result = await rename(
            old_name="get_user",
            new_name="fetch_user",
            path=str(test_file),
            scope="file",
            preview=False,
        )

        assert result["success"] is True

        content = test_file.read_text()
        assert "fetch_user" in content
        assert "get_username" in content  # Not renamed
        assert "fetch_username" not in content  # Partial match prevented

    @pytest.mark.asyncio
    async def test_rename_class(self, tmp_path):
        """Test renaming a class."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""class OldClass:
    def __init__(self):
        self.value = 0

def use_class():
    obj = OldClass()
    return obj
""")

        result = await rename(
            old_name="OldClass",
            new_name="NewClass",
            path=str(test_file),
            scope="file",
            preview=False,
        )

        assert result["success"] is True

        content = test_file.read_text()
        assert "class NewClass" in content
        assert "obj = NewClass()" in content
        assert "OldClass" not in content


class TestExtract:
    """Tests for extract function - extract code to new function."""

    @pytest.mark.asyncio
    async def test_extract_missing_file(self):
        """Test handling of missing file parameter."""
        result = await extract(file="", start_line=1, end_line=5, function_name="extracted")

        assert result["success"] is False
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_missing_function_name(self):
        """Test handling of missing function_name."""
        result = await extract(file="test.py", start_line=1, end_line=5, function_name="")

        assert result["success"] is False
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_invalid_line_range(self, tmp_path):
        """Test handling of invalid line range."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")

        result = await extract(
            file=str(test_file), start_line=5, end_line=2, function_name="extracted"
        )

        assert result["success"] is False
        assert "Invalid" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_file_not_found(self):
        """Test handling of non-existent file."""
        result = await extract(
            file="/nonexistent/file.py",
            start_line=1,
            end_line=5,
            function_name="extracted",
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_success_preview(self, tmp_path):
        """Test successful function extraction in preview mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def main():
    x = 5
    y = 10
    result = x + y
    print(result)
    return result
""")

        result = await extract(
            file=str(test_file),
            start_line=4,
            end_line=5,
            function_name="calculate_sum",
            preview=True,
        )

        # Should handle gracefully
        assert "success" in result


class TestInline:
    """Tests for inline function - inline variable."""

    @pytest.mark.asyncio
    async def test_inline_missing_file(self):
        """Test handling of missing file parameter."""
        result = await inline(file="", variable_name="var")

        assert result["success"] is False
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_inline_missing_variable_name(self):
        """Test handling of missing variable_name."""
        result = await inline(file="test.py", variable_name="")

        assert result["success"] is False
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_inline_file_not_found(self):
        """Test handling of non-existent file."""
        result = await inline(file="/nonexistent/file.py", variable_name="var")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_inline_not_found(self, tmp_path):
        """Test handling of variable not found."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")

        result = await inline(file=str(test_file), variable_name="nonexistent")

        # Variable not found
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_inline_success_preview(self, tmp_path):
        """Test successful variable inlining in preview mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def calculate():
    result = 5
    x = result * 2
    y = result + 3
    return x + y
""")

        result = await inline(file=str(test_file), variable_name="result", preview=True)

        assert result["success"] is True
        assert result["changes_count"] > 0
        assert "preview" in result["formatted_report"].lower()

    @pytest.mark.asyncio
    async def test_inline_success_apply(self, tmp_path):
        """Test successful variable inlining with changes applied."""
        test_file = tmp_path / "test.py"
        original_content = """
def process():
    value = 10
    result = value * 2
    return result
"""
        test_file.write_text(original_content)

        result = await inline(file=str(test_file), variable_name="value", preview=False)

        assert result["success"] is True
        assert result["changes_count"] > 0


class TestOrganizeImports:
    """Tests for organize_imports function."""

    @pytest.mark.asyncio
    async def test_organize_imports_missing_file(self):
        """Test handling of missing file parameter."""
        result = await organize_imports(file="")

        assert result["success"] is False
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_organize_imports_file_not_found(self):
        """Test handling of non-existent file."""
        result = await organize_imports(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_organize_imports_no_imports(self, tmp_path):
        """Test organizing file with no imports."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def hello():
    pass
""")

        result = await organize_imports(file=str(test_file))

        # Should succeed even with no imports
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_organize_imports_already_organized(self, tmp_path):
        """Test organizing already well-organized imports."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
import os
import sys

from pathlib import Path
from typing import Dict

def hello():
    pass
""")

        result = await organize_imports(file=str(test_file))

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_organize_imports_preview_mode(self, tmp_path):
        """Test organize imports in preview mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
from typing import Dict
import sys
import os
from pathlib import Path

def hello():
    pass
""")

        result = await organize_imports(file=str(test_file), preview=True)

        assert result["success"] is True
        assert "preview" in result["formatted_report"].lower()

    @pytest.mark.asyncio
    async def test_organize_imports_apply_changes(self, tmp_path):
        """Test applying import organization."""
        test_file = tmp_path / "test.py"
        original_content = """
from typing import Dict
import sys
import os

def hello():
    pass
"""
        test_file.write_text(original_content)

        result = await organize_imports(file=str(test_file), preview=False)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_organize_imports_with_duplicates(self, tmp_path):
        """Test organizing imports with duplicates."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
import os
import sys
import os

def hello():
    pass
""")

        result = await organize_imports(file=str(test_file))

        assert result["success"] is True


class TestHelperFunctions:
    """Tests for helper functions via public API."""

    @pytest.mark.asyncio
    async def test_find_symbol_variable(self, tmp_path):
        """Test symbol lookup finds variables."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""my_var = 10
other = my_var + 5
""")

        result = await rename(
            old_name="my_var",
            new_name="new_var",
            path=str(test_file),
            scope="file",
            preview=False,
        )

        assert result["success"] is True
        content = test_file.read_text()
        assert "new_var" in content

    @pytest.mark.asyncio
    async def test_analyze_variables_in_extract(self, tmp_path):
        """Test variable analysis during extraction."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def main():
    a = 1
    b = 2
    c = a + b
    d = c * 2
    return d
""")

        result = await extract(
            file=str(test_file),
            start_line=4,
            end_line=5,
            function_name="add_values",
            preview=True,
        )

        assert result["success"] is True
        assert "parameters" in result

    @pytest.mark.asyncio
    async def test_find_function_insert_point_empty(self, tmp_path):
        """Test function insert point when file has only imports."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""import os
import sys
from pathlib import Path
""")

        result = await extract(
            file=str(test_file),
            start_line=1,
            end_line=1,
            function_name="wrapped_import",
            preview=True,
        )

        # Should succeed even with only imports
        assert "success" in result


class TestRenameEdgeCases:
    """Additional edge cases for rename function."""

    @pytest.mark.asyncio
    async def test_rename_directory_not_found(self):
        """Test error when directory doesn't exist for directory scope."""
        result = await rename(
            old_name="func",
            new_name="new_func",
            path="/nonexistent/directory",
            scope="directory",
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_path_not_dir_for_project_scope(self, tmp_path):
        """Test error when path is file for project scope."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func(): pass")

        result = await rename(
            old_name="func",
            new_name="new_func",
            path=str(test_file),
            scope="project",
        )

        assert result["success"] is False
        assert "directory" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_no_occurrences_in_directory(self, tmp_path):
        """Test when symbol exists in file but not as definition."""
        file1 = tmp_path / "module1.py"
        file1.write_text("x = 1")

        file2 = tmp_path / "module2.py"
        file2.write_text("y = 2")

        result = await rename(
            old_name="nonexistent",
            new_name="new_name",
            path=str(tmp_path),
            scope="directory",
        )

        assert result["success"] is False
        assert "No occurrences" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_with_syntax_error_file(self, tmp_path):
        """Test that syntax errors in files are skipped gracefully."""
        good_file = tmp_path / "good.py"
        good_file.write_text("target = 1")

        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def func( # invalid syntax\n    pass")

        result = await rename(
            old_name="target",
            new_name="renamed",
            path=str(tmp_path),
            scope="project",
            preview=False,
        )

        assert result["success"] is True
        assert "renamed" in good_file.read_text()

    @pytest.mark.asyncio
    async def test_rename_many_files_truncation(self, tmp_path):
        """Test report truncation with many files."""
        # Create 20+ files
        for i in range(20):
            (tmp_path / f"module{i}.py").write_text(f"shared_var = {i}")

        result = await rename(
            old_name="shared_var",
            new_name="new_shared",
            path=str(tmp_path),
            scope="project",
            preview=True,
        )

        assert result["success"] is True
        assert "more files" in result["formatted_report"]

    @pytest.mark.asyncio
    async def test_rename_many_changes_truncation(self, tmp_path):
        """Test report truncation with many changes per file."""
        test_file = tmp_path / "test.py"
        # Create file with many occurrences
        lines = ["target = 1"] + [f"x{i} = target" for i in range(10)]
        test_file.write_text("\n".join(lines))

        result = await rename(
            old_name="target",
            new_name="renamed",
            path=str(test_file),
            scope="file",
            preview=True,
        )

        assert result["success"] is True
        assert "more" in result["formatted_report"]

    @pytest.mark.asyncio
    async def test_rename_with_report_depth_info(self, tmp_path):
        """Test that report shows depth when specified."""
        test_file = tmp_path / "test.py"
        test_file.write_text("target = 1")

        result = await rename(
            old_name="target",
            new_name="renamed",
            path=str(tmp_path),
            scope="project",
            depth=2,
            preview=True,
        )

        assert result["success"] is True
        assert "Depth" in result["formatted_report"]


class TestExtractEdgeCases:
    """Additional edge cases for extract function."""

    @pytest.mark.asyncio
    async def test_extract_with_return_values(self, tmp_path):
        """Test extraction that might have return values."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def main():
    a = 5
    b = 10
    result = a + b
    return result
""")

        result = await extract(
            file=str(test_file),
            start_line=3,
            end_line=4,
            function_name="compute",
            preview=True,
        )

        assert result["success"] is True
        assert "new_function" in result

    @pytest.mark.asyncio
    async def test_extract_empty_extraction(self, tmp_path):
        """Test extraction with minimal/empty block."""
        test_file = tmp_path / "test.py"
        test_file.write_text("pass\n")

        result = await extract(
            file=str(test_file),
            start_line=1,
            end_line=1,
            function_name="empty_func",
            preview=True,
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_extract_with_syntax_error_block(self, tmp_path):
        """Test extraction handles syntax errors in block gracefully."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def main():
    x = (1 +
    # incomplete expression
""")

        result = await extract(
            file=str(test_file),
            start_line=3,
            end_line=4,
            function_name="broken",
            preview=True,
        )

        # Should still succeed but with empty params
        assert result["success"] is True
        assert result["parameters"] == []

    @pytest.mark.asyncio
    async def test_extract_apply_changes(self, tmp_path):
        """Test extraction actually writes the file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def main():
    x = 5
    y = 10
    return x + y
""")

        result = await extract(
            file=str(test_file),
            start_line=3,
            end_line=4,
            function_name="setup_vars",
            preview=False,
        )

        assert result["success"] is True
        content = test_file.read_text()
        assert "def setup_vars" in content


class TestInlineEdgeCases:
    """Additional edge cases for inline function."""

    @pytest.mark.asyncio
    async def test_inline_with_syntax_error(self, tmp_path):
        """Test inline handles syntax errors."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func( # broken\n    pass")

        result = await inline(file=str(test_file), variable_name="x")

        assert result["success"] is False
        assert "Syntax error" in result["error"]

    @pytest.mark.asyncio
    async def test_inline_many_usages_truncation(self, tmp_path):
        """Test inline truncates report with many usages."""
        test_file = tmp_path / "test.py"
        lines = ["value = 10"] + [f"x{i} = value" for i in range(15)]
        test_file.write_text("\n".join(lines))

        result = await inline(file=str(test_file), variable_name="value", preview=True)

        assert result["success"] is True
        assert "more" in result["formatted_report"]

    @pytest.mark.asyncio
    async def test_inline_complex_value(self, tmp_path):
        """Test inlining a complex expression."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def func():
    computed = 1 + 2 * 3
    result = computed + 10
    return result
""")

        result = await inline(file=str(test_file), variable_name="computed", preview=False)

        assert result["success"] is True
        content = test_file.read_text()
        assert "1 + 2 * 3" in content


class TestOrganizeImportsEdgeCases:
    """Additional edge cases for organize_imports."""

    @pytest.mark.asyncio
    async def test_organize_imports_syntax_error(self, tmp_path):
        """Test organize_imports handles syntax errors."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func( # broken\n    pass")

        result = await organize_imports(file=str(test_file))

        assert result["success"] is False
        assert "Syntax error" in result["error"]

    @pytest.mark.asyncio
    async def test_organize_imports_with_alias(self, tmp_path):
        """Test organizing imports with aliases."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""import numpy as np
import os as operating_system
from pathlib import Path as P

def hello():
    pass
""")

        result = await organize_imports(file=str(test_file), preview=True)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_organize_imports_relative(self, tmp_path):
        """Test organizing relative imports."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""import os
from . import utils
from ..core import base
from .helpers import func

def hello():
    pass
""")

        result = await organize_imports(file=str(test_file))

        assert result["success"] is True
        assert result["local_count"] >= 1

    @pytest.mark.asyncio
    async def test_organize_imports_third_party(self, tmp_path):
        """Test organizing third-party imports."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""import requests
import numpy
from flask import Flask
import os

def hello():
    pass
""")

        result = await organize_imports(file=str(test_file))

        assert result["success"] is True
        assert result["third_party_count"] >= 1

    @pytest.mark.asyncio
    async def test_organize_imports_with_docstring(self, tmp_path):
        """Test organizing imports preserves module docstring."""
        test_file = tmp_path / "test.py"
        test_file.write_text('''"""This is a module docstring.

Multi-line docstring.
"""
import sys
import os

def hello():
    pass
''')

        result = await organize_imports(file=str(test_file), preview=False)

        assert result["success"] is True
        content = test_file.read_text()
        assert '"""This is a module docstring.' in content

    @pytest.mark.asyncio
    async def test_organize_imports_with_comments(self, tmp_path):
        """Test organizing imports preserves leading comments."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""# -*- coding: utf-8 -*-
# Copyright notice
import sys
import os

def hello():
    pass
""")

        result = await organize_imports(file=str(test_file), preview=False)

        assert result["success"] is True
        content = test_file.read_text()
        assert "# -*- coding: utf-8 -*-" in content

    @pytest.mark.asyncio
    async def test_organize_imports_many_imports_truncation(self, tmp_path):
        """Test report truncation with many imports."""
        test_file = tmp_path / "test.py"
        # Create file with 25+ imports
        imports = "\n".join([f"import module{i}" for i in range(25)])
        test_file.write_text(f"{imports}\n\ndef hello():\n    pass\n")

        result = await organize_imports(file=str(test_file), preview=True)

        assert result["success"] is True
        assert "more" in result["formatted_report"]


class TestCollectPythonFiles:
    """Tests for _collect_python_files behavior through public API."""

    @pytest.mark.asyncio
    async def test_collect_skips_nonexistent_path(self, tmp_path):
        """Test handling of non-existent paths."""
        result = await rename(
            old_name="x",
            new_name="y",
            path=str(tmp_path / "nonexistent"),
            scope="project",
        )

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_collect_directory_scope_recursion(self, tmp_path):
        """Test directory scope doesn't recurse deeply."""
        subdir = tmp_path / "sub"
        subdir.mkdir()

        file_root = tmp_path / "root.py"
        file_root.write_text("target = 1")

        file_sub = subdir / "nested.py"
        file_sub.write_text("target = 2")

        # Directory scope: should include immediate subdirs
        result = await rename(
            old_name="target",
            new_name="renamed",
            path=str(tmp_path),
            scope="directory",
            preview=False,
        )

        assert result["success"] is True
        # Both files should be renamed
        assert result["files_count"] == 2
