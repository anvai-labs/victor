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

"""Common utilities shared across Victor tools.

This module provides shared constants and utilities to avoid
code duplication across tool implementations.
"""

import os
from pathlib import Path
from typing import List, Optional, Set

# Directories to exclude when walking/searching codebases
EXCLUDE_DIRS: Set[str] = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    "*.egg-info",
    ".eggs",
}

# Default file extensions for code search
DEFAULT_CODE_EXTENSIONS: Set[str] = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".scss",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    # .NET / C# extensions
    ".cs",
    ".csx",
    ".vb",
    ".fs",
    ".fsx",
    ".csproj",
    ".vbproj",
    ".fsproj",
    ".sln",
    ".xaml",
}


def safe_walk(
    root: str,
    exclude_dirs: Optional[Set[str]] = None,
    extensions: Optional[Set[str]] = None,
) -> List[str]:
    """Walk directory tree safely, excluding common non-code directories.

    Args:
        root: Root directory to walk
        exclude_dirs: Directories to exclude (defaults to EXCLUDE_DIRS)
        extensions: File extensions to include (None = all files)

    Returns:
        List of file paths relative to root
    """
    exclude = exclude_dirs if exclude_dirs is not None else EXCLUDE_DIRS
    files: List[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out excluded directories in-place to prevent descent
        dirnames[:] = [d for d in dirnames if d not in exclude and not d.startswith(".")]

        for fname in filenames:
            # Skip hidden files
            if fname.startswith("."):
                continue

            # Filter by extension if specified
            if extensions is not None:
                ext = os.path.splitext(fname)[1]
                if ext not in extensions:
                    continue

            files.append(os.path.join(dirpath, fname))

    return files


def gather_code_files(
    root: str,
    extensions: Optional[Set[str]] = None,
    exclude_dirs: Optional[Set[str]] = None,
) -> List[str]:
    """Gather code files from directory tree.

    Args:
        root: Root directory to search
        extensions: File extensions to include (defaults to DEFAULT_CODE_EXTENSIONS)
        exclude_dirs: Directories to exclude (defaults to EXCLUDE_DIRS)

    Returns:
        List of file paths
    """
    exts = extensions if extensions is not None else DEFAULT_CODE_EXTENSIONS
    return safe_walk(root, exclude_dirs=exclude_dirs, extensions=exts)


def gather_files_by_pattern(
    root: Path,
    pattern: str = "*",
    exclude_dirs: Optional[Set[str]] = None,
) -> List[Path]:
    """Gather files matching a glob pattern, excluding common non-code directories.

    This is the canonical function for file gathering. All tools should use this
    instead of raw rglob to ensure consistent exclusion of venv, node_modules, etc.

    Args:
        root: Root directory to search
        pattern: Glob pattern for files (e.g., "*.py", "*.json")
        exclude_dirs: Directories to exclude (defaults to EXCLUDE_DIRS)

    Returns:
        List of Path objects matching pattern, excluding unwanted directories

    Example:
        # Get all Python files, excluding venv/node_modules
        files = gather_files_by_pattern(Path("."), "*.py")

        # Get all JSON files with custom exclusions
        files = gather_files_by_pattern(Path("src"), "*.json", {"tests"})
    """
    exclude = exclude_dirs if exclude_dirs is not None else EXCLUDE_DIRS
    files: List[Path] = []

    for path in root.rglob(pattern):
        if not path.is_file():
            continue

        # Check if any parent directory is in exclusion list
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            # Path is not relative to root
            parts = path.parts

        # Skip if any part is in excluded directories
        if any(part in exclude for part in parts):
            continue

        # Skip hidden directories (but not hidden files like .env)
        if any(part.startswith(".") for part in parts[:-1]):
            continue

        files.append(path)

    return files


def latest_mtime(root: Path, exclude_dirs: Optional[Set[str]] = None) -> float:
    """Find latest modification time under root, respecting exclusions.

    Args:
        root: Root directory to search
        exclude_dirs: Directories to exclude (defaults to EXCLUDE_DIRS)

    Returns:
        Latest modification time as Unix timestamp, or 0.0 if no files found
    """
    exclude = exclude_dirs if exclude_dirs is not None else EXCLUDE_DIRS
    latest = 0.0

    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude and not d.startswith(".")]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            try:
                mtime = fpath.stat().st_mtime
                if mtime > latest:
                    latest = mtime
            except OSError:
                pass

    return latest
