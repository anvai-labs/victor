"""Per-member tool paths without changing process-wide working directories.

This is execution isolation, not a shell security sandbox. Shell commands can
explicitly address other locations; callers retain their normal safety policy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from victor.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WorkspaceBoundTool(BaseTool):
    """Delegate supported tools with paths rooted in an assigned worktree."""

    SUPPORTED = frozenset({"read", "write", "edit", "shell", "ls", "find", "overview"})

    def __init__(self, tool: BaseTool, root: str):
        if tool.name not in self.SUPPORTED:
            logger.warning("Tool %s has no worktree path adapter", tool.name)
            raise ValueError(f"Tool {tool.name!r} has no worktree path adapter")
        self._tool = tool._tool if isinstance(tool, WorkspaceBoundTool) else tool
        self._root = Path(root).resolve()
        if not self._root.is_dir():
            raise ValueError(f"Member workspace does not exist: {root}")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    @property
    def name(self):
        return self._tool.name

    @property
    def description(self):
        return self._tool.description

    @property
    def parameters(self):
        return self._tool.parameters

    @property
    def metadata(self):
        return self._tool.metadata

    @property
    def cost_tier(self):
        return self._tool.cost_tier

    @property
    def is_idempotent(self):
        return self._tool.is_idempotent

    def _path(self, value: str) -> str:
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else self._root / path)

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        args = dict(kwargs)
        if self.name == "shell":
            args["cwd"] = self._path(args.get("cwd") or ".")
        elif self.name == "edit":
            ops = args.get("ops")
            if not isinstance(ops, list) or not all(isinstance(op, dict) for op in ops):
                return ToolResult(
                    success=False, output=None, error="Worktree edits require a structured ops list"
                )
            args["ops"] = [
                {
                    key: self._path(value) if key in {"path", "new_path"} else value
                    for key, value in op.items()
                }
                for op in ops
            ]
        else:
            args["path"] = self._path(args.get("path") or ".")
        return await self._tool.execute({**_exec_ctx, "workspace_root": str(self._root)}, **args)
