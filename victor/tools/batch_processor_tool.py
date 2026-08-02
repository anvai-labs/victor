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

"""Batch processing tool for multi-file operations.

Features:
- Process multiple files in parallel
- Pattern-based file selection
- Bulk operations (search, replace, analyze, list, transform)
- Progress tracking
- Error handling and reporting
- Dry-run mode
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool

logger = logging.getLogger(__name__)

# Constants
_DEFAULT_MAX_WORKERS: int = 4

# Lazy-loaded presentation adapter for icons
_presentation = None


def _get_icon(name: str) -> str:
    """Get icon from presentation adapter (lazy initialization)."""
    global _presentation
    if _presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        _presentation = create_presentation_adapter()
    return _presentation.icon(name, with_color=False)


# Helper functions for parallel processing


async def _parallel_search(
    files: List[Path],
    pattern: str,
    use_regex: bool,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> List[Dict[str, Any]]:
    """Search files in parallel with progress indication."""
    results = []

    def search_file(file_path: Path) -> Optional[Dict[str, Any]]:
        try:
            content = file_path.read_text()
            matches = []

            if use_regex:
                for line_num, line in enumerate(content.split("\n"), 1):
                    if re.search(pattern, line):
                        matches.append({"line": line_num, "text": line.strip()[:100]})
            else:
                for line_num, line in enumerate(content.split("\n"), 1):
                    if pattern in line:
                        matches.append({"line": line_num, "text": line.strip()[:100]})

            if matches:
                return {"file": str(file_path), "matches": matches}

        except Exception as e:
            logger.warning("Failed to search %s: %s", file_path, e)

        return None

    # Execute in parallel with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Searching {len(files)} files...", total=len(files))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(search_file, f): f for f in files}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                progress.update(task, advance=1)

    return results


async def _parallel_replace(
    files: List[Path],
    find_text: str,
    replace_text: str,
    use_regex: bool,
    dry_run: bool,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> List[Dict[str, Any]]:
    """Replace in files in parallel with progress indication."""
    results = []

    def replace_in_file(file_path: Path) -> Optional[Dict[str, Any]]:
        try:
            content = file_path.read_text()
            original = content

            if use_regex:
                new_content = re.sub(find_text, replace_text, content)
            else:
                new_content = content.replace(find_text, replace_text)

            if new_content != original:
                count = (
                    content.count(find_text)
                    if not use_regex
                    else len(re.findall(find_text, content))
                )

                if not dry_run:
                    file_path.write_text(new_content)

                return {
                    "file": str(file_path),
                    "replacements": count,
                    "modified": not dry_run,
                }

        except Exception as e:
            logger.warning("Failed to process %s: %s", file_path, e)
            return {"file": str(file_path), "error": str(e)}

        return None

    # Execute in parallel with progress tracking
    mode = "DRY RUN" if dry_run else "REPLACING"
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"{mode} in {len(files)} files...", total=len(files))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(replace_in_file, f): f for f in files}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                progress.update(task, advance=1)

    return results


async def _parallel_analyze(
    files: List[Path], max_workers: int = _DEFAULT_MAX_WORKERS
) -> List[Dict[str, Any]]:
    """Analyze files in parallel with progress indication."""
    results = []

    def analyze_file(file_path: Path) -> Dict[str, Any]:
        try:
            content = file_path.read_text()
            lines = content.split("\n")

            return {
                "file": str(file_path),
                "lines": len(lines),
                "size": file_path.stat().st_size,
                "extension": file_path.suffix,
            }

        except Exception as e:
            logger.warning("Failed to analyze %s: %s", file_path, e)
            return {
                "file": str(file_path),
                "error": str(e),
            }

    # Execute in parallel with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Analyzing {len(files)} files...", total=len(files))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(analyze_file, f): f for f in files}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                progress.update(task, advance=1)

    return results


@tool(
    cost_tier=CostTier.HIGH,
    category="batch",
    priority=Priority.LOW,  # Specialized bulk operations
    access_mode=AccessMode.MIXED,  # Reads and can modify files
    danger_level=DangerLevel.MEDIUM,  # Bulk file modifications
    keywords=["batch", "bulk", "search", "replace", "transform"],
)
async def batch(
    operation: str,
    path: str,
    file_pattern: str = "*.*",
    pattern: Optional[str] = None,
    find: Optional[str] = None,
    replace: Optional[str] = "",
    regex: bool = False,
    dry_run: bool = False,
    max_files: int = 1000,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    r"""
    Unified batch processing tool for multi-file operations.

    Performs parallel batch operations on multiple files including search,
    replace, analyze, and list operations. Consolidates all batch processing
    functionality into a single interface.

    Args:
        operation: Operation to perform. Options: "search", "replace", "analyze",
            "list", "transform".
        path: Directory path to process.
        file_pattern: Glob pattern to match files (default: *.*).
        pattern: Search pattern (for search operation).
        find: Text to find (for replace operation).
        replace: Replacement text (for replace operation, default: empty string).
        regex: Use regex for pattern/find matching (default: False).
        dry_run: Preview changes without modifying (for replace, default: False).
        max_files: Maximum number of files to process (default: 1000).
        options: Additional operation-specific options.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - operation: Operation performed
        - results: Operation-specific results
        - formatted_report: Human-readable report
        - error: Error message if failed

    Examples:
        # Search for pattern across files
        batch(operation="search", path="./src", pattern="TODO", file_pattern="*.py")

        # Find and replace across files
        batch(operation="replace", path="./src", find="old_name", replace="new_name",
              file_pattern="*.py", dry_run=True)

        # Analyze files and get statistics
        batch(operation="analyze", path="./src", file_pattern="*.py")

        # List files matching pattern
        batch(operation="list", path="./src", file_pattern="*.ts")

        # Using regex for advanced search/replace
        batch(operation="search", path="./", pattern=r"def \w+\(", regex=True)
    """
    if not operation:
        return {"success": False, "error": "Missing required parameter: operation"}

    if not path:
        return {"success": False, "error": "Missing required parameter: path"}

    if options is None:
        options = {}

    path_obj = Path(path)
    if not path_obj.exists():
        return {"success": False, "error": f"Path not found: {path}"}

    # Find matching files
    files = list(path_obj.rglob(file_pattern))[:max_files]

    if not files:
        return {
            "success": True,
            "operation": operation,
            "results": [],
            "message": f"No files matching pattern '{file_pattern}' found",
        }

    # Search operation
    if operation == "search":
        if not pattern:
            return {
                "success": False,
                "error": "Search operation requires 'pattern' parameter",
            }

        results = await _parallel_search(files, pattern, regex)
        total_matches = sum(len(r["matches"]) for r in results)

        # Build report
        report = []
        report.append(f"Batch Search Results: '{pattern}' in {path}")
        report.append("=" * 70)
        report.append("")
        report.append(f"Found in {len(results)} files ({total_matches} matches)")
        report.append("")

        for result in results[:20]:
            rel_path = (
                Path(result["file"]).relative_to(path_obj)
                if path_obj.is_dir()
                else Path(result["file"]).name
            )
            report.append(f"{rel_path}")
            for match in result["matches"][:5]:
                report.append(f"  Line {match['line']}: {match['text']}")
            if len(result["matches"]) > 5:
                report.append(f"  ... and {len(result['matches']) - 5} more matches")
            report.append("")

        if len(results) > 20:
            report.append(f"... and {len(results) - 20} more files")

        return {
            "success": True,
            "operation": "search",
            "results": results,
            "total_files": len(results),
            "total_matches": total_matches,
            "formatted_report": "\n".join(report),
        }

    # Replace operation
    elif operation == "replace":
        if not find:
            return {
                "success": False,
                "error": "Replace operation requires 'find' parameter",
            }

        results = await _parallel_replace(files, find, replace, regex, dry_run)
        total_replacements = sum(r.get("replacements", 0) for r in results)

        # Build report
        report = []
        mode = "DRY RUN" if dry_run else "EXECUTED"
        report.append(f"Batch Replace Results ({mode}): '{find}' → '{replace}' in {path}")
        report.append("=" * 70)
        report.append("")
        report.append(f"Modified {len(results)} files ({total_replacements} replacements)")
        report.append("")

        for result in results[:30]:
            rel_path = (
                Path(result["file"]).relative_to(path_obj)
                if path_obj.is_dir()
                else Path(result["file"]).name
            )
            if "error" in result:
                report.append(f"{_get_icon('error')} {rel_path}: {result['error']}")
            else:
                status = "Preview" if dry_run else "Modified"
                report.append(
                    f"{_get_icon('success')} {rel_path}: {result.get('replacements', 0)} replacements ({status})"
                )

        if dry_run:
            report.append("")
            report.append(f"{_get_icon('warning')}  This was a DRY RUN - no files were modified")
            report.append("   Run with dry_run=False to apply changes")

        return {
            "success": True,
            "operation": "replace",
            "results": results,
            "total_files": len(results),
            "total_replacements": total_replacements,
            "dry_run": dry_run,
            "formatted_report": "\n".join(report),
        }

    # Analyze operation
    elif operation == "analyze":
        results = await _parallel_analyze(files)

        # Calculate stats
        total_lines = sum(r.get("lines", 0) for r in results)
        total_size = sum(r.get("size", 0) for r in results)

        # Group by extension
        by_ext: Dict[str, Dict[str, int]] = {}
        for result in results:
            ext = result.get("extension", "no extension")
            if ext not in by_ext:
                by_ext[ext] = {"count": 0, "lines": 0, "size": 0}

            by_ext[ext]["count"] += 1
            by_ext[ext]["lines"] += result.get("lines", 0)
            by_ext[ext]["size"] += result.get("size", 0)

        # Build report
        report = []
        report.append(f"Batch Analysis Report: {path}")
        report.append("=" * 70)
        report.append("")
        report.append(f"Files analyzed: {len(results)}")
        report.append(f"Total lines: {total_lines:,}")
        report.append(f"Total size: {total_size / 1024:.2f} KB")
        report.append("")
        report.append("By file type:")
        for ext, stats in sorted(by_ext.items(), key=lambda x: x[1]["count"], reverse=True):
            report.append(
                f"  {ext}: {stats['count']} files, {stats['lines']:,} lines, {stats['size'] / 1024:.2f} KB"
            )

        return {
            "success": True,
            "operation": "analyze",
            "results": results,
            "total_files": len(results),
            "total_lines": total_lines,
            "total_size": total_size,
            "by_extension": by_ext,
            "formatted_report": "\n".join(report),
        }

    # List operation
    elif operation == "list":
        # Prepare file info
        file_info = []
        for file_path in sorted(files):
            rel_path = file_path.relative_to(path_obj)
            size = file_path.stat().st_size
            file_info.append({"path": str(rel_path), "size": size})

        # Build report
        report = []
        report.append(f"Files matching '{file_pattern}' in {path}:")
        report.append("=" * 70)
        report.append("")
        report.append(f"Total: {len(files)} files")
        report.append("")

        for info in file_info[:100]:
            report.append(f"  {info['path']} ({info['size']} bytes)")

        if len(files) > 100:
            report.append("")
            report.append(f"... and {len(files) - 100} more files")

        return {
            "success": True,
            "operation": "list",
            "files": file_info,
            "total_files": len(files),
            "formatted_report": "\n".join(report),
        }

    # Transform operation (placeholder)
    elif operation == "transform":
        return {"success": False, "error": "Transform operation not yet implemented"}

    else:
        return {
            "success": False,
            "error": f"Unknown operation: {operation}. Valid operations: search, replace, analyze, list, transform",
        }
