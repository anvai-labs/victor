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

"""Jupyter notebook cell-level editing tool.

Supports replace, insert, and delete operations on individual cells
within .ipynb files, preserving notebook metadata and kernel specs.
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
from json import JSONDecodeError
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.tools.decorators import tool
from victor.tools.enums import (
    AccessMode,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    Priority,
)

logger = logging.getLogger(__name__)


class NotebookEditMode(Enum):
    """Supported notebook edit operations."""

    REPLACE = "replace"
    INSERT = "insert"
    DELETE = "delete"


class NotebookCellType(Enum):
    """Supported Jupyter cell types."""

    CODE = "code"
    MARKDOWN = "markdown"


@dataclass
class NotebookEditResult:
    """Result of a notebook edit operation.

    Args:
        success: Whether the operation completed successfully.
        message: Human-readable description of the outcome.
        original_cell_count: Number of cells before the edit.
        new_cell_count: Number of cells after the edit.
    """

    success: bool
    message: str
    original_cell_count: int = 0
    new_cell_count: int = 0


def _resolve_cell_index(
    cells: List[Dict[str, Any]],
    cell_id: Optional[str],
    cell_index: Optional[int],
) -> int:
    """Resolve a cell target to a concrete index.

    Looks up by ``cell_id`` first (matching the ``id`` field in cell
    metadata or at the cell top-level). Falls back to ``cell_index``.

    Args:
        cells: The notebook cells list.
        cell_id: Optional cell id string.
        cell_index: Optional zero-based index.

    Returns:
        Resolved zero-based index into *cells*.

    Raises:
        ValueError: If neither identifier is provided, or the target
            cannot be found.
    """
    if cell_id is not None:
        for idx, cell in enumerate(cells):
            # nbformat >= 4.5 stores id at top level; older versions may
            # store it inside metadata.
            top_id = cell.get("id")
            meta_id = cell.get("metadata", {}).get("id")
            if cell_id in (top_id, meta_id):
                return idx
        raise ValueError(f"Cell with id '{cell_id}' not found in notebook")

    if cell_index is not None:
        if cell_index < 0 or cell_index >= len(cells):
            raise ValueError(
                f"cell_index {cell_index} out of range " f"(notebook has {len(cells)} cells)"
            )
        return cell_index

    raise ValueError("Either cell_id or cell_index must be provided")


def _make_cell(
    cell_type: NotebookCellType,
    source: str,
) -> Dict[str, Any]:
    """Create a new notebook cell dictionary.

    Args:
        cell_type: The type of cell to create.
        source: The source content of the cell.

    Returns:
        A dict conforming to the nbformat v4 cell schema.
    """
    base: Dict[str, Any] = {
        "cell_type": cell_type.value,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == NotebookCellType.CODE:
        base["execution_count"] = None
        base["outputs"] = []
    return base


def _read_notebook(path: Path) -> Dict[str, Any]:
    """Read and parse a notebook file.

    Args:
        path: Filesystem path to the ``.ipynb`` file.

    Returns:
        Parsed notebook dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is not valid JSON or lacks a cells array.
    """
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    if not path.suffix == ".ipynb":
        raise ValueError(f"File is not an .ipynb notebook: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        notebook = json_loads(text)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in notebook: {exc}") from exc

    if not isinstance(notebook.get("cells"), list):
        raise ValueError("Notebook JSON does not contain a 'cells' array")

    return notebook


def _write_notebook(path: Path, notebook: Dict[str, Any]) -> None:
    """Write a notebook dictionary back to disk.

    Uses 1-space indent and a trailing newline to match the default
    Jupyter on-disk format.

    Args:
        path: Filesystem path to write.
        notebook: The notebook dictionary to serialize.
    """
    text = json_dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


@tool(
    category="filesystem",
    priority=Priority.MEDIUM,
    access_mode=AccessMode.WRITE,
    danger_level=DangerLevel.LOW,
    cost_tier=CostTier.FREE,
    execution_category=ExecutionCategory.WRITE,
    stages=["execution"],
    task_types=["action"],
    keywords=["notebook", "jupyter", "ipynb", "cell", "edit notebook"],
)
async def notebook_edit(
    notebook_path: str,
    edit_mode: NotebookEditMode = NotebookEditMode.REPLACE,
    cell_type: NotebookCellType = NotebookCellType.CODE,
    cell_id: Optional[str] = None,
    cell_index: Optional[int] = None,
    new_source: Optional[str] = None,
) -> NotebookEditResult:
    """Edit a Jupyter notebook at the cell level.

    Supports three modes:

    * **REPLACE** -- update the source of an existing cell, preserving its
      outputs and metadata.
    * **INSERT** -- insert a new cell at a given position.
    * **DELETE** -- remove a cell at a given position.

    Args:
        notebook_path: Path to the .ipynb file.
        edit_mode: The edit operation to perform (replace, insert, delete).
        cell_type: Cell type for insert operations (code or markdown).
        cell_id: Optional cell id string to locate the target cell.
        cell_index: Optional zero-based cell index to locate the target cell.
        new_source: New cell content for replace/insert operations.

    Returns:
        A :class:`NotebookEditResult` describing the outcome.
    """
    path = Path(notebook_path).expanduser().resolve()

    try:
        notebook = _read_notebook(path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to read notebook: %s", exc)
        return NotebookEditResult(success=False, message=str(exc))

    cells: List[Dict[str, Any]] = notebook["cells"]
    original_count = len(cells)

    mode = edit_mode

    # --- REPLACE ---
    if mode == NotebookEditMode.REPLACE:
        if new_source is None:
            return NotebookEditResult(
                success=False,
                message="new_source is required for REPLACE mode",
                original_cell_count=original_count,
                new_cell_count=original_count,
            )
        try:
            idx = _resolve_cell_index(cells, cell_id, cell_index)
        except ValueError as exc:
            return NotebookEditResult(
                success=False,
                message=str(exc),
                original_cell_count=original_count,
                new_cell_count=original_count,
            )

        cells[idx]["source"] = new_source.splitlines(keepends=True)
        # Clear stale outputs for code cells after content change.
        if cells[idx].get("cell_type") == "code":
            cells[idx]["execution_count"] = None
            cells[idx]["outputs"] = []

        _write_notebook(path, notebook)
        logger.info("Replaced cell %d in %s", idx, path)
        return NotebookEditResult(
            success=True,
            message=f"Replaced cell {idx}",
            original_cell_count=original_count,
            new_cell_count=original_count,
        )

    # --- INSERT ---
    if mode == NotebookEditMode.INSERT:
        if new_source is None:
            return NotebookEditResult(
                success=False,
                message="new_source is required for INSERT mode",
                original_cell_count=original_count,
                new_cell_count=original_count,
            )

        # Determine insertion index.  If neither cell_id nor cell_index is
        # given, append to the end.
        if cell_id is not None or cell_index is not None:
            try:
                insert_idx = _resolve_cell_index(cells, cell_id, cell_index)
            except ValueError as exc:
                return NotebookEditResult(
                    success=False,
                    message=str(exc),
                    original_cell_count=original_count,
                    new_cell_count=original_count,
                )
        else:
            insert_idx = len(cells)

        new_cell = _make_cell(cell_type, new_source)
        cells.insert(insert_idx, new_cell)

        _write_notebook(path, notebook)
        new_count = len(cells)
        logger.info(
            "Inserted %s cell at index %d in %s",
            cell_type.value,
            insert_idx,
            path,
        )
        return NotebookEditResult(
            success=True,
            message=f"Inserted {cell_type.value} cell at index {insert_idx}",
            original_cell_count=original_count,
            new_cell_count=new_count,
        )

    # --- DELETE ---
    if mode == NotebookEditMode.DELETE:
        if original_count == 0:
            return NotebookEditResult(
                success=False,
                message="Notebook has no cells to delete",
                original_cell_count=0,
                new_cell_count=0,
            )
        try:
            idx = _resolve_cell_index(cells, cell_id, cell_index)
        except ValueError as exc:
            return NotebookEditResult(
                success=False,
                message=str(exc),
                original_cell_count=original_count,
                new_cell_count=original_count,
            )

        del cells[idx]

        _write_notebook(path, notebook)
        new_count = len(cells)
        logger.info("Deleted cell %d from %s", idx, path)
        return NotebookEditResult(
            success=True,
            message=f"Deleted cell {idx}",
            original_cell_count=original_count,
            new_cell_count=new_count,
        )

    # Should be unreachable, but guard against future enum additions.
    return NotebookEditResult(
        success=False,
        message=f"Unsupported edit mode: {mode}",
        original_cell_count=original_count,
        new_cell_count=original_count,
    )
