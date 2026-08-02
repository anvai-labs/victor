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

"""
Diff-based file editor implementation.

This module provides the default file editor implementation using
simple string replacement (diff-based editing).

SOLID Principles:
- SRP: DiffEditor only handles diff-based editing
- OCP: Extends BaseEditor without modification
- LSP: Implements EditorProtocol completely
- ISP: Focused on diff-based operations
- DIP: No dependencies on concrete implementations

Usage:
    from victor.contrib.editing import DiffEditor

    editor = DiffEditor()
    result = await editor.edit_file(
        file_path="/path/to/file.py",
        edits=[EditOperation(old_str="old", new_str="new")],
    )
"""

from __future__ import annotations

import logging
from typing import Any, List

from victor.framework.vertical_protocols import (
    EditOperation,
    EditResult,
    EditValidationResult,
    EditorProtocol,
)

logger = logging.getLogger(__name__)


class DiffEditor(EditorProtocol):
    """
    Diff-based file editor using string replacement.

    This editor applies edits by replacing old_str with new_str in the
    file content. It's the default implementation for file editing
    in Victor.

    The editor supports:
    - Single replacement (default)
    - Multiple replacements (allow_multiple=True)
    - Preview mode without applying changes

    Example:
        editor = DiffEditor()

        # Single replacement
        result = await editor.edit_file(
            "myfile.py",
            [EditOperation(old_str="def foo():", new_str="def bar():")],
        )

        # Multiple replacements
        result = await editor.edit_file(
            "myfile.py",
            [EditOperation(old_str="TODO", new_str="FIXME", allow_multiple=True)],
        )
    """

    def __init__(
        self,
        case_sensitive: bool = True,
    ) -> None:
        """Initialize the diff editor.

        Args:
            case_sensitive: Whether string matching is case-sensitive
        """
        self._case_sensitive = case_sensitive

    async def edit_file(
        self,
        file_path: str,
        edits: list[EditOperation],
        preview: bool = False,
        **kwargs: Any,
    ) -> EditResult:
        """Apply edits to a file using string replacement."""
        from pathlib import Path

        path = Path(file_path)

        # Validate inputs
        if not path.exists():
            return EditResult(
                success=False,
                file_path=file_path,
                edits_applied=0,
                error=f"File not found: {file_path}",
            )

        if not path.is_file():
            return EditResult(
                success=False,
                file_path=file_path,
                edits_applied=0,
                error=f"Not a file: {file_path}",
            )

        # Read file content
        try:
            content = path.read_text(encoding="utf-8")
        except PermissionError:
            return EditResult(
                success=False,
                file_path=file_path,
                edits_applied=0,
                error=f"Permission denied reading file: {file_path}",
            )
        except Exception as e:
            return EditResult(
                success=False,
                file_path=file_path,
                edits_applied=0,
                error=f"Error reading file: {e}",
            )

        # Apply edits sequentially
        edits_applied = 0
        edits_failed = 0

        errors: List[str] = []

        for edit in edits:
            try:
                # Validate edit
                validation = await self.validate_edit(
                    file_path,
                    edit.old_str,
                    edit.new_str,
                    content=content,
                    allow_multiple=edit.allow_multiple,
                )

                if not validation.valid:
                    edits_failed += 1
                    errors.append(validation.error or "Unknown validation error")
                    logger.warning(f"Edit validation failed: {validation.error}")
                    continue

                # Apply replacement
                content = self._apply_replacement(
                    content,
                    edit.old_str,
                    edit.new_str,
                    edit.allow_multiple,
                )
                edits_applied += 1

            except Exception as e:
                edits_failed += 1
                errors.append(str(e))
                logger.error(f"Error applying edit: {e}")

        # Write or preview result
        if preview:
            return EditResult(
                success=edits_failed == 0,
                file_path=file_path,
                edits_applied=edits_applied,
                edits_failed=edits_failed,
                preview=content,
                error="; ".join(errors) if errors else None,
            )

        # Only write if all edits succeeded (or at least one succeeded)
        if edits_applied == 0 and errors:
            return EditResult(
                success=False,
                file_path=file_path,
                edits_applied=0,
                edits_failed=edits_failed,
                error="; ".join(errors),
            )

        try:
            path.write_text(content, encoding="utf-8")
        except PermissionError:
            return EditResult(
                success=False,
                file_path=file_path,
                edits_applied=0,
                error=f"Permission denied writing to file: {file_path}",
            )
        except Exception as e:
            return EditResult(
                success=False,
                file_path=file_path,
                edits_applied=0,
                error=f"Error writing file: {e}",
            )

        return EditResult(
            success=edits_failed == 0,
            file_path=file_path,
            edits_applied=edits_applied,
            edits_failed=edits_failed,
            error="; ".join(errors) if errors else None,
        )

    async def validate_edit(
        self,
        file_path: str,
        old_str: str,
        new_str: str,
        content: str | None = None,
        **kwargs: Any,
    ) -> EditValidationResult:
        """Validate an edit operation."""
        from pathlib import Path
        import re

        # Read content if not provided
        if content is None:
            try:
                content = Path(file_path).read_text(encoding="utf-8")
            except Exception as e:
                return EditValidationResult(
                    valid=False,
                    file_path=file_path,
                    old_str_found=False,
                    error=f"Cannot read file: {e}",
                )

        # Check if old_str exists (respect case sensitivity)
        if self._case_sensitive:
            old_str_found = old_str in content
            match_count = content.count(old_str)
        else:
            # Case-insensitive check
            pattern = re.compile(re.escape(old_str), re.IGNORECASE)
            matches = pattern.findall(content)
            old_str_found = len(matches) > 0
            match_count = len(matches)

        if not old_str_found:
            return EditValidationResult(
                valid=False,
                file_path=file_path,
                old_str_found=False,
                error=f"String not found in file: {old_str[:50]}...",
            )

        # Check for multiple occurrences
        allow_multiple = kwargs.get("allow_multiple", False)
        if match_count > 1 and not allow_multiple:
            return EditValidationResult(
                valid=False,
                file_path=file_path,
                old_str_found=True,
                match_count=match_count,
                error=f"String appears {match_count} times. "
                f"Use allow_multiple=True to replace all.",
            )

        return EditValidationResult(
            valid=True,
            file_path=file_path,
            old_str_found=True,
            match_count=match_count,
        )

    def get_editor_info(self) -> dict[str, Any]:
        """Get editor metadata."""
        return {
            "name": "DiffEditor",
            "version": "1.0.0",
            "case_sensitive": self._case_sensitive,
            "capabilities": ["string_replacement", "multiple_replacements"],
        }

    def _apply_replacement(
        self,
        content: str,
        old_str: str,
        new_str: str,
        allow_multiple: bool,
    ) -> str:
        """Apply string replacement to content."""
        if self._case_sensitive:
            return content.replace(
                old_str,
                new_str,
                1 if not allow_multiple else -1,
            )

        # Case-insensitive replacement
        import re

        pattern = re.escape(old_str)
        return re.sub(
            pattern,
            new_str,
            content,
            count=1 if not allow_multiple else 0,
            flags=re.IGNORECASE,
        )


__all__ = ["DiffEditor"]
