# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FileEditor backup self-heal — regression for the benchmark edit-commit P0.

Root cause: the FileEditor is a long-lived singleton whose backup_dir is mkdir'd
once at __init__. The SWE-bench harness runs `git clean -fd` between tasks,
which deletes the workspace's `.victor/` (untracked) — including the backup_dir.
The next edit's `_create_backup` then does `shutil.copy2` into a non-existent
dir → FileNotFoundError → the whole commit rolls back → every later edit fails.

Fix: `_create_backup` re-ensures the backup_dir exists before each copy
(ported to core `victor/processing/editing/editor.py` when the victor-coding
editor fork was unified away — the fork had carried the fix since #346).
"""

from __future__ import annotations

import shutil


def test_edit_self_heals_after_backup_dir_deleted(tmp_path):
    """An edit must still commit after an external process deletes backup_dir."""
    import os

    from victor.processing.editing.editor import EditOperation, FileEditor, OperationType

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


def test_reset_state_clears_transaction_history_and_error(tmp_path):
    """reset_state (#368): the FileEditor is a singleton; per-task state must
    not leak across tasks — transaction history bounded, stale commit error
    cleared, backup_dir re-ensured."""
    from victor.processing.editing.editor import EditOperation, FileEditor, OperationType

    editor = FileEditor(backup_dir=str(tmp_path / "backups"))
    editor.start_transaction("task N")
    editor.current_transaction.operations.append(
        EditOperation(
            type=OperationType.CREATE,
            path=str(tmp_path / "a.txt"),
            content="hello",
        )
    )
    editor.commit()
    editor.last_commit_error = "StaleError: from task N"
    editor.reset_state()
    assert editor.current_transaction is None
    assert editor.transaction_history == []
    assert editor.last_commit_error is None
    # Re-ensures a deleted backup dir (same hygiene as _create_backup).
    import shutil as _shutil

    _shutil.rmtree(tmp_path / "backups")
    editor.reset_state()
    assert (tmp_path / "backups").is_dir()


def test_commit_failure_populates_last_commit_error(tmp_path, caplog):
    """last_commit_error (#344): commit failures are captured and logged —
    the victor edit tool suppresses the editor's rich-console stdout, so the
    attribute is the only surface for the real error."""
    import logging

    from victor.processing.editing.editor import EditOperation, FileEditor, OperationType

    editor = FileEditor(backup_dir=str(tmp_path / "backups"))
    editor.start_transaction("doomed")
    # RENAME without new_path raises deterministically INSIDE the apply loop
    # (bypassing the add_* helpers' up-front validation) -> rollback path.
    editor.current_transaction.operations.append(
        EditOperation(
            type=OperationType.RENAME,
            path=str(tmp_path / "anything.txt"),
        )
    )
    with caplog.at_level(logging.ERROR, logger="victor.processing.editing.editor"):
        committed = editor.commit()
    assert committed is False
    assert editor.last_commit_error is not None
    assert editor.last_commit_error.startswith("ValueError:")
    assert any("FileEditor.commit failed" in r.message for r in caplog.records)
