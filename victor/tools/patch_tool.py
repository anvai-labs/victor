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

"""Patch tool for applying unified diff patches to files.

This tool provides functionality similar to the Unix `patch` command,
allowing AI-generated diffs to be applied to files safely.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)


@dataclass
class Hunk:
    """Represents a single hunk from a unified diff."""

    old_start: int  # Line number in original file (1-indexed)
    old_count: int  # Number of lines in original
    new_start: int  # Line number in new file (1-indexed)
    new_count: int  # Number of lines in new
    lines: List[str]  # The actual diff lines (with +/-/space prefix)


@dataclass
class PatchFile:
    """Represents a patch for a single file."""

    old_path: Optional[str]  # Original file path (None for new files)
    new_path: Optional[str]  # New file path (None for deleted files)
    hunks: List[Hunk]
    is_binary: bool = False
    is_new_file: bool = False
    is_deleted: bool = False


def parse_unified_diff(diff_text: str) -> List[PatchFile]:
    """Parse a unified diff into structured patch data.

    Args:
        diff_text: The unified diff text

    Returns:
        List of PatchFile objects
    """
    patches = []
    lines = diff_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Look for file header
        if line.startswith("--- "):
            old_path = _parse_path(line[4:])
            i += 1

            if i >= len(lines) or not lines[i].startswith("+++ "):
                logger.warning(f"Expected +++ line at position {i}")
                continue

            new_path = _parse_path(lines[i][4:])
            i += 1

            # Determine if new or deleted file
            is_new_file = old_path in ("/dev/null", "")
            is_deleted = new_path in ("/dev/null", "")

            # Parse hunks
            hunks = []
            while i < len(lines) and lines[i].startswith("@@"):
                hunk, i = _parse_hunk(lines, i)
                if hunk:
                    hunks.append(hunk)

            patches.append(
                PatchFile(
                    old_path=None if is_new_file else old_path,
                    new_path=None if is_deleted else new_path,
                    hunks=hunks,
                    is_new_file=is_new_file,
                    is_deleted=is_deleted,
                )
            )
        else:
            i += 1

    return patches


def _parse_path(path_str: str) -> str:
    """Parse a path from diff header line.

    Handles various formats:
    - a/path/to/file.py
    - b/path/to/file.py
    - /dev/null
    - "path with spaces/file.py"
    """
    path_str = path_str.strip()

    # Remove quotes if present
    if path_str.startswith('"') and path_str.endswith('"'):
        path_str = path_str[1:-1]

    # Remove a/ or b/ prefix (git diff format)
    if path_str.startswith("a/") or path_str.startswith("b/"):
        path_str = path_str[2:]

    # Handle timestamp suffix (e.g., "file.py\t2024-01-01 12:00:00")
    path_str = path_str.split("\t")[0].strip()

    return path_str


def _parse_hunk(lines: List[str], start_idx: int) -> Tuple[Optional[Hunk], int]:
    """Parse a single hunk from diff lines.

    Args:
        lines: All diff lines
        start_idx: Index of the @@ line

    Returns:
        Tuple of (Hunk or None, next index)
    """
    header = lines[start_idx]

    # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
    match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
    if not match:
        logger.warning(f"Could not parse hunk header: {header}")
        return None, start_idx + 1

    old_start = int(match.group(1))
    old_count = int(match.group(2)) if match.group(2) else 1
    new_start = int(match.group(3))
    new_count = int(match.group(4)) if match.group(4) else 1

    # Collect hunk lines
    i = start_idx + 1
    hunk_lines = []

    while i < len(lines):
        line = lines[i]

        # Stop at next hunk or file header
        if line.startswith("@@") or line.startswith("---") or line.startswith("diff "):
            break

        # Accept context (+/-/space) lines
        if line.startswith("+") or line.startswith("-") or line.startswith(" ") or line == "":
            hunk_lines.append(line)
            i += 1
        # Handle "\ No newline at end of file" marker
        elif line.startswith("\\"):
            i += 1
        else:
            break

    return (
        Hunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            lines=hunk_lines,
        ),
        i,
    )


def apply_patch_to_content(
    content: str, hunks: List[Hunk], fuzz: int = 2
) -> Tuple[bool, str, List[str]]:
    """Apply patch hunks to file content.

    Args:
        content: Original file content
        hunks: List of hunks to apply
        fuzz: Allowed fuzz factor for matching (lines of context to ignore)

    Returns:
        Tuple of (success, new_content, list of warnings)
    """
    lines = content.split("\n")
    warnings = []
    offset = 0  # Track line offset from applied hunks

    for hunk in hunks:
        target_line = hunk.old_start - 1 + offset  # Convert to 0-indexed

        # Extract expected old lines and new lines from hunk
        old_lines = []
        new_lines = []

        for line in hunk.lines:
            if line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith("+"):
                new_lines.append(line[1:])
            elif line.startswith(" ") or line == "":
                # Context line
                context = line[1:] if line.startswith(" ") else ""
                old_lines.append(context)
                new_lines.append(context)

        # Try to find matching location (with fuzz)
        match_line = _find_matching_location(lines, old_lines, target_line, fuzz)

        if match_line is None:
            warnings.append(f"Could not apply hunk at line {hunk.old_start}: context mismatch")
            continue

        # Apply the hunk
        lines = lines[:match_line] + new_lines + lines[match_line + len(old_lines) :]

        # Update offset for subsequent hunks
        offset += len(new_lines) - len(old_lines)

    return len(warnings) == 0, "\n".join(lines), warnings


def _find_matching_location(
    lines: List[str], old_lines: List[str], target: int, fuzz: int
) -> Optional[int]:
    """Find where old_lines match in the file, with fuzz tolerance.

    Args:
        lines: Current file lines
        old_lines: Lines we're looking for
        target: Expected line number (0-indexed)
        fuzz: How many lines to search around target

    Returns:
        Matching line number (0-indexed) or None
    """
    if not old_lines:
        return target

    # Try exact location first
    if _lines_match(lines, old_lines, target):
        return target

    # Try with fuzz
    for offset in range(1, fuzz + 1):
        # Try before target
        if target - offset >= 0 and _lines_match(lines, old_lines, target - offset):
            return target - offset
        # Try after target
        if _lines_match(lines, old_lines, target + offset):
            return target + offset

    # Try searching more broadly
    for i in range(len(lines) - len(old_lines) + 1):
        if _lines_match(lines, old_lines, i):
            return i

    return None


def _lines_match(lines: List[str], expected: List[str], start: int) -> bool:
    """Check if expected lines match at position start."""
    if start + len(expected) > len(lines):
        return False

    for i, exp in enumerate(expected):
        if lines[start + i].rstrip() != exp.rstrip():
            return False

    return True


@tool(
    category="patch",
    priority=Priority.HIGH,  # Important for applying code changes
    access_mode=AccessMode.WRITE,  # Modifies files
    danger_level=DangerLevel.LOW,  # Changes are undoable via backup
    keywords=["patch", "apply", "diff", "unified diff", "create"],
)
async def patch(
    operation: str = "apply",
    patch_content: str = "",
    file_path: Optional[str] = None,
    new_content: str = "",
    dry_run: bool = False,
    fuzz: int = 2,
    backup: bool = True,
    context_lines: int = 3,
) -> Dict[str, Any]:
    """Unified patch operations: create diffs or apply patches.

    Operations:
    - "apply": Apply a unified diff patch to files (default)
    - "create": Create a unified diff from file and new content

    Args:
        operation: "apply" (default) or "create"
        patch_content: Unified diff content (for "apply" operation)
        file_path: Target file path
        new_content: New content to diff against (for "create" operation)
        dry_run: Preview changes without applying (for "apply" operation)
        fuzz: Fuzzy matching factor (for "apply" operation)
        backup: Create backup before modifying (for "apply" operation)
        context_lines: Context lines to include (for "create" operation)
    """
    op = operation.lower().strip()

    # Route to create diff operation
    if op == "create":
        if not file_path:
            return {
                "success": False,
                "error": "file_path required for 'create' operation",
            }
        if not new_content:
            return {
                "success": False,
                "error": "new_content required for 'create' operation",
            }
        return await _create_diff(file_path, new_content, context_lines)

    # Apply patch operation (default)
    if op != "apply":
        return {
            "success": False,
            "error": f"Unknown operation '{operation}'. Use 'apply' or 'create'.",
        }

    if not patch_content:
        return {
            "success": False,
            "error": "patch_content required for 'apply' operation",
        }
    from victor.agent.change_tracker import ChangeType, get_change_tracker

    # Parse the patch
    try:
        patches = parse_unified_diff(patch_content)
    except Exception as e:
        return {"success": False, "error": f"Failed to parse patch: {e}"}

    if not patches:
        return {"success": False, "error": "No valid patches found in input"}

    # Track results
    files_modified = []
    files_created = []
    files_deleted = []
    warnings = []
    preview_output = []

    tracker = get_change_tracker()
    tracker.begin_change_group("apply_patch", f"Apply patch to {len(patches)} file(s)")

    for patch_file in patches:
        # Determine target path
        target_path = file_path or patch_file.new_path or patch_file.old_path
        if not target_path:
            warnings.append("Patch missing file path")
            continue

        target = Path(target_path).expanduser().resolve()

        try:
            if patch_file.is_deleted:
                # Handle file deletion
                if dry_run:
                    preview_output.append(f"Would delete: {target_path}")
                else:
                    original_content = None
                    if target.exists():
                        original_content = target.read_text()
                        target.unlink()
                        files_deleted.append(str(target))

                        tracker.record_change(
                            file_path=str(target),
                            change_type=ChangeType.DELETE,
                            original_content=original_content,
                            new_content=None,
                            tool_name="apply_patch",
                        )

            elif patch_file.is_new_file:
                # Handle new file creation
                new_content = _get_new_content_from_hunks(patch_file.hunks)

                if dry_run:
                    preview_output.append(f"Would create: {target_path}")
                    preview_output.append(f"Content:\n{new_content[:500]}...")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(new_content)
                    files_created.append(str(target))

                    tracker.record_change(
                        file_path=str(target),
                        change_type=ChangeType.CREATE,
                        original_content=None,
                        new_content=new_content,
                        tool_name="apply_patch",
                    )

            else:
                # Handle file modification
                if not target.exists():
                    warnings.append(f"File not found: {target_path}")
                    continue

                original_content = target.read_text()

                if backup and not dry_run:
                    backup_path = target.with_suffix(target.suffix + ".orig")
                    backup_path.write_text(original_content)

                success, new_content, hunk_warnings = apply_patch_to_content(
                    original_content, patch_file.hunks, fuzz
                )

                warnings.extend(hunk_warnings)

                if dry_run:
                    preview_output.append(f"Would modify: {target_path}")
                    # Show diff preview
                    changes = _compute_simple_diff(original_content, new_content)
                    preview_output.append(f"Changes:\n{changes}")
                else:
                    target.write_text(new_content)
                    files_modified.append(str(target))

                    tracker.record_change(
                        file_path=str(target),
                        change_type=ChangeType.MODIFY,
                        original_content=original_content,
                        new_content=new_content,
                        tool_name="apply_patch",
                    )

        except Exception as e:
            warnings.append(f"Error processing {target_path}: {e}")
            logger.exception(f"Error applying patch to {target_path}")

    if not dry_run:
        tracker.commit_change_group()

    result = {
        "success": len(warnings) == 0,
        "files_modified": files_modified,
        "files_created": files_created,
        "files_deleted": files_deleted,
        "warnings": warnings,
    }

    if dry_run:
        result["preview"] = "\n".join(preview_output)
        result["message"] = "Dry run - no files were modified"
    else:
        total = len(files_modified) + len(files_created) + len(files_deleted)
        result["message"] = f"Applied patch to {total} file(s). Use /undo to revert."

    return result


def _get_new_content_from_hunks(hunks: List[Hunk]) -> str:
    """Extract new file content from hunks (for new files)."""
    lines = []
    for hunk in hunks:
        for line in hunk.lines:
            if line.startswith("+"):
                lines.append(line[1:])
            elif line.startswith(" "):
                lines.append(line[1:])
    return "\n".join(lines)


def _compute_simple_diff(old: str, new: str) -> str:
    """Compute a simple line-by-line diff for preview."""
    old_lines = old.split("\n")
    new_lines = new.split("\n")

    result = []
    max_lines = max(len(old_lines), len(new_lines))

    for i in range(min(max_lines, 20)):  # Limit preview to 20 lines
        old_line = old_lines[i] if i < len(old_lines) else ""
        new_line = new_lines[i] if i < len(new_lines) else ""

        if old_line != new_line:
            if old_line:
                result.append(f"- {old_line}")
            if new_line:
                result.append(f"+ {new_line}")
        else:
            result.append(f"  {old_line}")

    if max_lines > 20:
        result.append(f"... ({max_lines - 20} more lines)")

    return "\n".join(result)


async def _create_diff(
    file_path: str,
    new_content: str,
    context_lines: int = 3,
) -> Dict[str, Any]:
    """Internal: Create a unified diff patch from a file and new content."""
    import difflib

    try:
        target = Path(file_path).expanduser().resolve()

        if target.exists():
            original_content = target.read_text()
            original_lines = original_content.splitlines(keepends=True)
        else:
            original_content = ""
            original_lines = []

        new_lines = new_content.splitlines(keepends=True)

        # Generate unified diff
        diff_result = difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=context_lines,
        )

        patch_text = "".join(diff_result)

        # Compute stats
        additions = sum(
            1
            for line in patch_text.split("\n")
            if line.startswith("+") and not line.startswith("+++")
        )
        deletions = sum(
            1
            for line in patch_text.split("\n")
            if line.startswith("-") and not line.startswith("---")
        )

        return {
            "success": True,
            "operation": "create",
            "patch": patch_text,
            "stats": {
                "additions": additions,
                "deletions": deletions,
                "file": file_path,
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create patch: {e}",
        }
