# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Shared search helpers for the unified command tools.

These pure functions back the literal-search surfaces that several unified
dispatchers reach for:

* :func:`grep_search` — literal/regex content search across files. Used by
  ``code grep`` (the canonical literal content search) and as the graceful
  fallback for ``code search`` when the semantic ``victor_coding`` package is
  absent. Also used by the legacy ``search grep`` back-compat shim.

Extracted here (out of ``search_tool.py``) so the helpers survive the
``search`` domain's retirement and can be reused without re-importing a
deprecated tool module.

When ripgrep (``rg``) is on PATH it is used as the search engine (binary and
size skipping for free); otherwise a guarded Python walk runs. Both engines
return the same result shape.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.framework.tool_progress import emit_tool_progress, has_progress_sink

logger = logging.getLogger(__name__)

# Directories never worth scanning for code search. ``.victor`` holds the
# project database and LanceDB embedding fragments — multi-GB binaries that
# must never be text-scanned. ``.claude`` holds linked worktrees whose contents
# would duplicate every match.
_SEARCH_SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".victor",
    ".claude",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".tox",
    ".eggs",
    "htmlcov",
}

# Files larger than this are skipped by the Python walk (and via
# ``--max-filesize`` for ripgrep): source files this big are almost always
# generated artifacts, and reading them dominates scan time.
_MAX_FILE_BYTES = 2 * 1024 * 1024

# Bytes sniffed from the head of each file to detect binary content.
_BINARY_SNIFF_BYTES = 8192

# How often (in files) the Python walk yields to the event loop and emits a
# progress heartbeat to the live renderer.
_PROGRESS_EVERY_FILES = 200

# A slow-scan threshold that promotes the completion log line to WARNING so
# multi-minute regressions are visible in the log instead of silent.
_SLOW_SCAN_WARN_SECONDS = 5.0


async def grep_search(
    query: str,
    path: str,
    regex: bool = False,
    case_sensitive: bool = False,
    include_glob: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search file contents and return grep-like match dictionaries.

    Args:
        query: Text or regex pattern to search for.
        path: File or directory to search under.
        regex: When ``True``, treat ``query`` as a regular expression;
            otherwise escape it for a literal substring match.
        case_sensitive: When ``True``, match case-sensitively; otherwise
            case-insensitively (the default, matching grep ``-i``).
        include_glob: When set, only search files whose basename matches this
            glob (grep ``--include=GLOB`` semantics, e.g. ``*.py``).

    Returns:
        List of ``{"file": str, "line": int, "content": str}`` dicts, one per
        matching line.
    """
    root = Path(path).expanduser()
    started = time.monotonic()

    results = await _ripgrep_search(
        query, root, regex=regex, case_sensitive=case_sensitive, include_glob=include_glob
    )
    engine = "rg"
    if results is None:
        engine = "python"
        results = await _python_walk_search(
            query, root, regex=regex, case_sensitive=case_sensitive, include_glob=include_glob
        )

    elapsed = time.monotonic() - started
    log = logger.warning if elapsed >= _SLOW_SCAN_WARN_SECONDS else logger.debug
    log(
        "grep_search engine=%s query=%r root=%s matches=%d elapsed=%.2fs",
        engine,
        query,
        root,
        len(results),
        elapsed,
    )
    return results


async def _ripgrep_search(
    query: str,
    root: Path,
    *,
    regex: bool,
    case_sensitive: bool,
    include_glob: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Run ripgrep when available; return ``None`` to fall back to the walk."""
    rg = shutil.which("rg")
    if rg is None or not root.exists():
        return None

    args = [
        rg,
        "--line-number",
        "--no-heading",
        "--with-filename",
        "--color",
        "never",
        "--no-messages",
        "--hidden",
        f"--max-filesize={_MAX_FILE_BYTES}",
    ]
    for skip in sorted(_SEARCH_SKIP_DIRS):
        args.append(f"--glob=!{skip}/")
    if include_glob:
        args.append(f"--glob={include_glob}")
    if not case_sensitive:
        args.append("--ignore-case")
    if not regex:
        args.append("--fixed-strings")
    args += ["--", query, str(root)]

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
    except OSError as exc:  # rg vanished or failed to spawn — use the walk
        logger.debug("ripgrep unavailable (%s); falling back to Python walk", exc)
        return None
    if proc.returncode not in (0, 1):  # 1 = no matches; >=2 = usage/runtime error
        logger.debug("ripgrep exited %s; falling back to Python walk", proc.returncode)
        return None

    results: List[Dict[str, Any]] = []
    for raw in stdout.decode("utf-8", errors="ignore").splitlines():
        file_part, sep, rest = raw.partition(":")
        line_part, sep2, content = rest.partition(":")
        if not sep or not sep2 or not line_part.isdigit():
            continue
        results.append({"file": file_part, "line": int(line_part), "content": content})
    return results


async def _python_walk_search(
    query: str,
    root: Path,
    *,
    regex: bool,
    case_sensitive: bool,
    include_glob: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Guarded pure-Python fallback scan (no ripgrep on PATH)."""
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query if regex else re.escape(query), flags)
    results: List[Dict[str, Any]] = []
    scanned = 0

    for file_path in _iter_search_files(root):
        if include_glob and not fnmatch.fnmatch(file_path.name, include_glob):
            # grep --include matches on the basename, not the full path.
            continue
        scanned += 1
        if scanned % _PROGRESS_EVERY_FILES == 0:
            if has_progress_sink():
                emit_tool_progress(
                    name="code",
                    stdout=f"scanning… {scanned} files, {len(results)} matches",
                )
            # Keep the event loop responsive during large scans.
            await asyncio.sleep(0)
        text = _read_search_text(file_path)
        if text is None:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                results.append(
                    {
                        "file": str(file_path),
                        "line": line_no,
                        "content": line,
                    }
                )
    return results


def _iter_search_files(root: Path):
    """Yield candidate files under ``root``, pruning skip-dirs without descending."""
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SEARCH_SKIP_DIRS]
        for filename in filenames:
            yield Path(dirpath) / filename


def _read_search_text(file_path: Path) -> Optional[str]:
    """Read a file for scanning, or ``None`` if oversized, binary, or unreadable."""
    try:
        if file_path.stat().st_size > _MAX_FILE_BYTES:
            return None
        with open(file_path, "rb") as fh:
            head = fh.read(_BINARY_SNIFF_BYTES)
            if b"\x00" in head:
                return None
            rest = fh.read()
        return (head + rest).decode("utf-8", errors="ignore")
    except OSError:
        return None


def format_grep_results(
    results: List[Dict[str, Any]], *, limit: int = 50, files_only: bool = False
) -> str:
    """Render grep matches as ``file:line: content`` lines with a truncation hint.

    Args:
        results: Output of :func:`grep_search`.
        limit: Maximum number of matches to render before truncating.
        files_only: When ``True``, emit only the unique file paths (first-seen
            order, no ``:line:`` parts) — grep ``-l`` semantics.

    Returns:
        A human-readable string; ``"No matches found."`` when empty.
    """
    if not isinstance(results, list):
        return str(results)

    out: List[str] = []
    if files_only:
        seen: List[str] = []
        for match in results:
            file_path = match.get("file", "unknown")
            if file_path not in seen:
                seen.append(file_path)
        out = seen[:limit]
        if len(seen) > limit:
            out.append(
                f"\n### 💡 SYSTEM HINT\nToo many matching files ({len(seen)}). "
                "Results truncated. Please refine your search query or directory."
            )
        if not out:
            return "No matches found."
        return "\n".join(out)

    for i, match in enumerate(results):
        if i >= limit:
            break
        file_path = match.get("file", "unknown")
        line_number = match.get("line", "?")
        content = match.get("content", "").strip()
        out.append(f"{file_path}:{line_number}: {content}")

    if len(results) > limit:
        out.append(
            f"\n### 💡 SYSTEM HINT\nToo many matches found ({len(results)}). "
            "Results truncated. Please refine your search query or directory."
        )

    if not out:
        return "No matches found."
    return "\n".join(out)


__all__ = ["grep_search", "format_grep_results"]
