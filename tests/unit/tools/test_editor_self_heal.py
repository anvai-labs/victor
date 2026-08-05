# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FileEditor backup self-heal — regression for the benchmark edit-commit P0.

Root cause: the FileEditor is a long-lived singleton whose backup_dir is mkdir'd
once at __init__. The SWE-bench harness runs `git clean -fd` between tasks,
which deletes the workspace's `.victor/` (untracked) — including the backup_dir.
The next edit's `_create_backup` then does `shutil.copy2` into a non-existent
dir → FileNotFoundError → the whole commit rolls back → every later edit fails.

Fix: `_create_backup` re-ensures the backup_dir exists before each copy.
"""

from __future__ import annotations

import shutil

import pytest


def test_edit_self_heals_after_backup_dir_deleted(tmp_path):
    """An edit must still commit after an external process deletes backup_dir."""
    pytest.importorskip("victor_coding")  # FileEditor lives in the victor-coding vertical
    import os
    from pathlib import Path

    from victor_coding.editing.editor import EditOperation, FileEditor, OperationType

    f = tmp_path / "x.py"
    f.write_text("a = 1\n")
    backup_dir = tmp_path / "backups"

    os.chdir(tmp_path)
    editor = FileEditor(backup_dir=str(backup_dir))

    def _commit(path: str, content: str) -> bool:
        editor.current_transaction = None
        editor.start_transaction("test")
        editor.current_transaction.operations.append(
            EditOperation(type=OperationType.MODIFY, path=path, new_content=content)
        )
        return editor.commit(dry_run=False)

    # First edit succeeds (backup_dir exists from __init__).
    assert _commit("x.py", "a = 2\n") is True

    # Simulate `git clean -fd` wiping .victor/backups between tasks.
    shutil.rmtree(backup_dir)
    assert not backup_dir.exists()

    # Previously FileNotFoundError -> commit rolled back. Now self-heals.
    assert _commit("x.py", "a = 3\n") is True
    assert editor.last_commit_error is None
    assert f.read_text() == "a = 3\n"
