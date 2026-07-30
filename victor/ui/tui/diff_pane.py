# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Dedicated unified-diff pane for the interactive TUI (ADR-020, Phase 2).

When a file-editing tool (``edit`` / ``patch`` / ``replace_in_file``) completes,
:func:`extract_edit_diff` turns its result into an :class:`EditDiff` by reusing
the shared :class:`~victor.ui.rendering.tool_preview.ToolPreviewRenderer` diff
strategy (which prefers the tool's own ``diff`` / ``diff_formatted`` output and
falls back to a ``difflib`` diff over the edit arguments). :class:`DiffPane`
renders the latest edit as a colored, scrollable unified diff and lets the user
cycle through every edit in the turn. Pure extraction is Textual-free and
unit-testable; only :class:`DiffPane` touches Textual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rich.markup import escape

from victor.ui.rendering.tool_preview import ToolPreviewRenderer

#: Tools whose output is a file edit worth showing as a diff.
_EDIT_TOOLS = ("edit", "patch", "replace_in_file")

#: Upper bound on diff lines kept per edit (guards against pathological diffs).
_MAX_DIFF_LINES = 500

_RENDERER = ToolPreviewRenderer()

# Diff-line colours, matched to the TUI theme (Dracula family).
_ADD = "#50fa7b"
_DEL = "#ff5555"
_HUNK = "#8be9fd"


@dataclass
class EditDiff:
    """A single file edit's rendered diff."""

    path: str
    stats: str = ""  # e.g. "+3 -1"
    lines: List[str] = field(default_factory=list)
    colored: bool = False  # True when `lines` already carry Rich markup


def is_edit_tool(tool_name: str) -> bool:
    """True when ``tool_name`` names a file-editing tool."""
    name = (tool_name or "").lower()
    return any(name == tool or name.startswith(tool) for tool in _EDIT_TOOLS)


def _path_from_args(arguments: Dict[str, Any]) -> Optional[str]:
    path = arguments.get("path") or arguments.get("file_path")
    if path:
        return str(path)
    ops = arguments.get("ops")
    if isinstance(ops, list):
        for op in ops:
            if isinstance(op, dict) and op.get("path"):
                return str(op["path"])
    return None


def extract_edit_diff(
    tool_name: str,
    arguments: Optional[Dict[str, Any]],
    result: str,
    max_lines: int = _MAX_DIFF_LINES,
) -> Optional[EditDiff]:
    """Build an :class:`EditDiff` for an edit tool call, or ``None``.

    Returns ``None`` for non-edit tools and for edit calls that produced no diff
    (e.g. a bare "N operations applied" summary), so callers can feed every
    ``TOOL_END`` action and only diffs surface.
    """
    if not is_edit_tool(tool_name):
        return None
    args = dict(arguments or {})
    try:
        preview = _RENDERER.render(tool_name, args, result or "", max_lines)
    except Exception:  # noqa: BLE001 - a preview must never break the turn
        return None
    if preview.syntax_hint != "diff" or not preview.lines:
        return None
    path = _path_from_args(args) or "(edit)"
    return EditDiff(
        path=path,
        stats=(preview.header or "").strip(),
        lines=list(preview.lines),
        colored=preview.contains_rich_markup,
    )


def _colorize(line: str) -> str:
    """Apply +/-/@@ diff colouring to a plain unified-diff line."""
    escaped = escape(line)
    if line.startswith("+") and not line.startswith("+++"):
        return f"[{_ADD}]{escaped}[/]"
    if line.startswith("-") and not line.startswith("---"):
        return f"[{_DEL}]{escaped}[/]"
    if line.startswith("@@"):
        return f"[{_HUNK}]{escaped}[/]"
    return f"[dim]{escaped}[/]"


try:  # Textual is a core dep; keep the pure pieces importable without it.
    from textual.widgets import RichLog
except Exception:  # pragma: no cover - only in a broken install
    RichLog = None  # type: ignore[assignment,misc]


if RichLog is not None:

    class DiffPane(RichLog):
        """Collapsible unified-diff pane; shows one edit at a time, newest first."""

        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("wrap", False)
            kwargs.setdefault("markup", True)
            kwargs.setdefault("highlight", False)
            super().__init__(**kwargs)
            self._edits: List[EditDiff] = []
            self._index = 0

        @property
        def has_edits(self) -> bool:
            """Whether any edit diff has been captured this turn."""
            return bool(self._edits)

        def add_edit(self, edit: EditDiff) -> None:
            """Append an edit, focus it, and reveal the pane."""
            self._edits.append(edit)
            self._index = len(self._edits) - 1
            self.display = True
            self._render_current()

        def cycle(self) -> None:
            """Advance to the next captured edit (wraps), revealing the pane."""
            if not self._edits:
                return
            self._index = (self._index + 1) % len(self._edits)
            self.display = True
            self._render_current()

        def clear_edits(self) -> None:
            """Drop all captured edits and hide the pane (e.g. on a new turn)."""
            self._edits = []
            self._index = 0
            self.clear()
            self.display = False

        def _render_current(self) -> None:
            self.clear()
            if not self._edits:
                return
            edit = self._edits[self._index]
            counter = f"({self._index + 1}/{len(self._edits)})"
            stats = f"  [dim]{escape(edit.stats)}[/]" if edit.stats else ""
            self.write(f"[bold]▾ diff · {escape(edit.path)}[/]  [dim]{counter}[/]{stats}")
            for line in edit.lines:
                self.write(line if edit.colored else _colorize(line))
