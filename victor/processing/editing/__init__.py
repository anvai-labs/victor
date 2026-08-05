# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Generic file editing and transaction support for Victor framework.

This module provides safe, transaction-based file editing with:
- Atomic multi-file operations
- Automatic backups
- Rollback capability
- Diff preview
- Dry-run mode

Example:
    ```python
    from victor.processing.editing import FileEditor

    editor = FileEditor(backup_dir="~/.victor/backups")

    # Start a transaction
    editor.start_transaction("Add authentication module")

    # Queue edits
    editor.create_file("auth.py", "class Auth:\\n    pass")
    editor.modify_file("config.py", old_content, new_content)

    # Preview changes
    print(editor.preview_changes())

    # Commit or rollback
    editor.commit()  # or editor.rollback()
    ```

This is a core framework module available to all verticals, not just coding.
"""

from victor.processing.editing.editor import (
    OperationType,
    EditOperation,
    EditTransaction,
    FileEditor,
)

__all__ = [
    "OperationType",
    "EditOperation",
    "EditTransaction",
    "FileEditor",
]
