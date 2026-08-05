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

"""Shared ignore patterns and path filtering logic for codebase indexing.

This module centralizes the logic for determining which files and directories
should be excluded from indexing, ensuring consistency across all indexer
components (CodebaseIndexer, SymbolStore, CodebaseAnalyzer).

Design Principles:
- Hidden directories (starting with '.') are excluded by convention
- Non-hidden skip directories are explicitly listed
- Custom exclude patterns can be added per-project
"""

from pathlib import Path
from typing import Iterable, Optional, Set

# Default directories to skip (non-hidden only)
# Hidden directories (starting with '.') are excluded automatically by should_ignore_path()
DEFAULT_SKIP_DIRS: Set[str] = {
    # Python
    "__pycache__",
    "venv",
    "env",
    # Node.js
    "node_modules",
    # Build outputs
    "build",
    "dist",
    "target",
    "out",
    "egg-info",
    # Coverage
    "coverage",
    "htmlcov",
    # Third party / vendor
    "vendor",
    "third_party",
    # Archive/legacy code (not actively maintained)
    "archive",
}


def is_hidden_path(path: Path) -> bool:
    """Check if any component of the path is a hidden directory.

    Hidden directories follow Unix convention: they start with '.'
    Excludes '.' and '..' which are special directory entries.

    Args:
        path: Path to check

    Returns:
        True if path contains any hidden directory components
    """
    for part in path.parts:
        if part.startswith(".") and part not in (".", ".."):
            return True
    return False


def should_ignore_path(
    path: Path,
    skip_dirs: Optional[Set[str]] = None,
    extra_skip_dirs: Optional[Iterable[str]] = None,
) -> bool:
    """Check if a path should be ignored during indexing.

    This function provides consistent ignore logic across all indexer components.
    It automatically excludes:
    1. Hidden directories (Unix convention: starting with '.')
    2. Directories in the skip_dirs set

    Args:
        path: Path to check
        skip_dirs: Set of directory names to skip. Defaults to DEFAULT_SKIP_DIRS.
        extra_skip_dirs: Additional directory names to skip (merged with skip_dirs).

    Returns:
        True if the path should be ignored

    Example:
        >>> from pathlib import Path
        >>> should_ignore_path(Path("src/main.py"))
        False
        >>> should_ignore_path(Path(".git/config"))
        True
        >>> should_ignore_path(Path("node_modules/lodash/index.js"))
        True
        >>> should_ignore_path(Path("src/__pycache__/main.cpython-311.pyc"))
        True
    """
    # Check for hidden directories first (most common case)
    if is_hidden_path(path):
        return True

    # Build effective skip dirs set
    effective_skip_dirs = skip_dirs if skip_dirs is not None else DEFAULT_SKIP_DIRS
    if extra_skip_dirs:
        effective_skip_dirs = effective_skip_dirs | set(extra_skip_dirs)

    # Check if any path component is in skip dirs
    return any(part in effective_skip_dirs for part in path.parts)


def get_effective_skip_dirs(
    base_skip_dirs: Optional[Set[str]] = None,
    extra_skip_dirs: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Get the effective set of directories to skip.

    Args:
        base_skip_dirs: Base set of directories to skip. Defaults to DEFAULT_SKIP_DIRS.
        extra_skip_dirs: Additional directories to skip.

    Returns:
        Combined set of directory names to skip
    """
    effective = base_skip_dirs if base_skip_dirs is not None else DEFAULT_SKIP_DIRS.copy()
    if extra_skip_dirs:
        effective = effective | set(extra_skip_dirs)
    return effective
