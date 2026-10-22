"""Shared helpers for change-tracker and patch routes."""

from __future__ import annotations

from typing import Any, Dict


def undo_last_change() -> Dict[str, Any]:
    """Undo the last tracked change and return a response payload."""
    from victor.core.change_tracker import get_change_tracker

    tracker = get_change_tracker()
    success, message, files = tracker.undo()
    return {"success": success, "message": message, "files": files}


def redo_last_change() -> Dict[str, Any]:
    """Redo the last undone change and return a response payload."""
    from victor.core.change_tracker import get_change_tracker

    tracker = get_change_tracker()
    success, message, files = tracker.redo()
    return {"success": success, "message": message, "files": files}


def change_history(limit: int) -> Dict[str, Any]:
    """Return change history limited to the requested number of entries."""
    from victor.core.change_tracker import get_change_tracker

    tracker = get_change_tracker()
    history = tracker.get_history(limit=limit)
    return {"history": history}


async def apply_patch_request(patch: str, dry_run: bool) -> Dict[str, Any]:
    """Apply a patch via patch_tool and return its result payload."""
    from victor.tools import patch_tool

    return await patch_tool.apply_patch(patch=patch, dry_run=dry_run)


async def create_patch_request(file_path: str, new_content: str) -> Dict[str, Any]:
    """Create a patch via patch_tool and return its result payload."""
    from victor.tools import patch_tool

    return await patch_tool.create_patch(file_path=file_path, new_content=new_content)


__all__ = [
    "undo_last_change",
    "redo_last_change",
    "change_history",
    "apply_patch_request",
    "create_patch_request",
]
