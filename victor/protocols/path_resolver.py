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

"""Centralized Path Resolution.

This module provides unified path normalization and resolution, replacing
per-tool path handling scattered throughout the codebase.

Design:
- Normalization pipeline with composable normalizers
- Fuzzy matching for similar path suggestions
- Cached resolution for performance
- Clear separation of file vs directory resolution

Common Issues Fixed:
- LLMs using paths like "project/utils" when cwd is already "project"
- Relative paths with incorrect depth
- Mixed path separators
- Trailing slashes on files

Usage:
    resolver = PathResolver()
    result = resolver.resolve_file("investor_homelab/models/news.py")

    if result.was_normalized:
        logger.info(f"Path normalized: {result.original_path} -> {result.resolved_path}")

    # Fuzzy matching for suggestions
    suggestions = resolver.suggest_similar("modls/news.py")  # Typo
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Path Resolution Result
# =============================================================================


@dataclass
class PathResolution:
    """Result of path resolution.

    Attributes:
        original_path: The path as provided by caller
        resolved_path: The normalized, resolved Path object
        was_normalized: True if any normalization was applied
        normalization_applied: Description of normalization (if any)
        exists: Whether the resolved path exists
        is_file: True if resolved path is a file
        is_directory: True if resolved path is a directory
    """

    original_path: str
    resolved_path: Path
    was_normalized: bool = False
    normalization_applied: Optional[str] = None
    exists: bool = False
    is_file: bool = False
    is_directory: bool = False

    @property
    def path_str(self) -> str:
        """Get resolved path as string."""
        return str(self.resolved_path)

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.was_normalized:
            return (
                f"'{self.original_path}' -> '{self.resolved_path}' ({self.normalization_applied})"
            )
        return str(self.resolved_path)


# =============================================================================
# Path Resolver Protocol
# =============================================================================


@runtime_checkable
class IPathResolver(Protocol):
    """Protocol for path resolution.

    Provides centralized path normalization and resolution
    for all filesystem operations.
    """

    def resolve(self, path: str, must_exist: bool = True) -> PathResolution:
        """Resolve a path with normalization.

        Args:
            path: Path to resolve (relative or absolute)
            must_exist: If True, raises error if path doesn't exist

        Returns:
            PathResolution with resolved path and metadata
        """
        ...

    def resolve_file(self, path: str) -> PathResolution:
        """Resolve a file path.

        Args:
            path: File path to resolve

        Returns:
            PathResolution, raises if not a file
        """
        ...

    def resolve_directory(self, path: str) -> PathResolution:
        """Resolve a directory path.

        Args:
            path: Directory path to resolve

        Returns:
            PathResolution, raises if not a directory
        """
        ...

    def suggest_similar(self, path: str, limit: int = 5) -> List[str]:
        """Suggest similar paths that exist.

        Useful for providing helpful error messages when a path doesn't exist.

        Args:
            path: Non-existent path to find matches for
            limit: Maximum suggestions to return

        Returns:
            List of similar existing paths
        """
        ...


# =============================================================================
# Normalizer Functions
# =============================================================================


def strip_cwd_prefix(path: str, cwd: Path) -> Tuple[Optional[str], str]:
    """Strip redundant cwd prefix from path.

    When working in a subdirectory, LLMs sometimes include the directory name
    in paths (e.g., "project/utils" when cwd is already "project/").

    Args:
        path: Path to normalize
        cwd: Current working directory

    Returns:
        Tuple of (normalized_path or None, description)
    """
    if not path or path.startswith("/") or path.startswith("~"):
        return None, ""

    cwd_name = cwd.name

    # Check if path starts with cwd name
    if path.startswith(f"{cwd_name}/"):
        stripped = path[len(cwd_name) + 1 :]
        if stripped:
            return stripped, f"stripped_cwd_prefix:{cwd_name}"

    return None, ""


def strip_human_root_prefix(path: str, cwd: Path) -> Tuple[Optional[str], str]:
    """Strip natural-language root/repository prefixes from valid paths.

    LLMs sometimes turn "root rust/Cargo.toml" into a literal path even
    though "root" was descriptive. Only strip when the remaining path exists.
    """
    if not path or path.startswith("/") or path.startswith("~"):
        return None, ""

    match = re.match(r"^(?P<prefix>root|repo|repository|project)(?:\s+|/)(?P<rest>.+)$", path)
    if not match:
        return None, ""

    remainder = match.group("rest").strip()
    if not remainder:
        return None, ""

    if (cwd / remainder).exists():
        return remainder, f"stripped_human_root_prefix:{match.group('prefix')}"

    return None, ""


def strip_first_component(path: str, cwd: Path) -> Tuple[Optional[str], str]:
    """Try stripping first path component if it matches a cwd component.

    Args:
        path: Path to normalize
        cwd: Current working directory

    Returns:
        Tuple of (normalized_path or None, description)
    """
    if not path or "/" not in path:
        return None, ""

    parts = path.split("/", 1)
    if len(parts) != 2:
        return None, ""

    first_component = parts[0]
    remainder = parts[1]

    # Check if first component is any part of cwd
    cwd_parts = set(cwd.parts)
    if first_component in cwd_parts:
        # Verify the remainder exists
        check_path = cwd / remainder
        if check_path.exists():
            return remainder, f"stripped_component:{first_component}"

    return None, ""


def strip_common_prefix(path: str, cwd: Path) -> Tuple[Optional[str], str]:
    """Strip common prefix if path has nested duplicate.

    Handles cases like "src/components/src/components/Button.tsx"
    where the path was accidentally duplicated.

    Args:
        path: Path to normalize
        cwd: Current working directory

    Returns:
        Tuple of (normalized_path or None, description)
    """
    if "/" not in path:
        return None, ""

    parts = path.split("/")
    n = len(parts)

    # Look for repeated sequences
    for seq_len in range(1, n // 2 + 1):
        prefix = parts[:seq_len]
        if parts[seq_len : seq_len + seq_len] == prefix:
            # Found duplicate prefix
            stripped = "/".join(parts[seq_len:])
            check_path = cwd / stripped
            if check_path.exists():
                return stripped, f"stripped_duplicate:{'/'.join(prefix)}"

    return None, ""


def restore_absolute_path(path: str, cwd: Path) -> Tuple[Optional[str], str]:
    """Restore leading slash for Unix absolute paths.

    LLMs sometimes drop the leading '/' from absolute paths. This normalizer
    detects paths that look like they should be absolute and restores the slash.

    Common Unix root directories that indicate an absolute path:
    - var/ (system variable data)
    - tmp/ (temporary files)
    - home/ (user home directories)
    - usr/ (user programs)
    - etc/ (configuration)
    - opt/ (optional software)
    - private/ (macOS private directory containing /var, /tmp, etc.)

    Args:
        path: Path to check
        cwd: Current working directory (unused but required by normalizer signature)

    Returns:
        Tuple of (normalized_path with leading /, description) or (None, "")
    """
    # Skip if already absolute or home-relative
    if path.startswith("/") or path.startswith("~"):
        return None, ""

    # Common Unix root directories that indicate the path should be absolute
    UNIX_ROOT_PREFIXES = (
        "var/",
        "tmp/",
        "home/",
        "usr/",
        "etc/",
        "opt/",
        "private/",  # macOS: /private/var, /private/tmp
        "Users/",  # macOS user directories
        "Library/",  # macOS Library
        "System/",  # macOS System
        "Applications/",  # macOS Applications
    )

    if path.startswith(UNIX_ROOT_PREFIXES):
        absolute_path = "/" + path
        # Verify the absolute path exists before suggesting it
        if Path(absolute_path).exists():
            return absolute_path, "restored_absolute_path"

    return None, ""


def normalize_separators(path: str, cwd: Path) -> Tuple[Optional[str], str]:
    """Normalize path separators and trailing slashes.

    Args:
        path: Path to normalize
        cwd: Current working directory

    Returns:
        Tuple of (normalized_path or None, description)
    """
    # Replace backslashes with forward slashes (Windows paths)
    if "\\" in path:
        normalized = path.replace("\\", "/")
        return normalized, "normalized_separators"

    # Remove trailing slashes (except for root)
    if path.endswith("/") and len(path) > 1:
        normalized = path.rstrip("/")
        return normalized, "stripped_trailing_slash"

    return None, ""


# =============================================================================
# Path Resolver Implementation
# =============================================================================


@dataclass
class PathResolver(IPathResolver):
    """Centralized path resolution with normalization pipeline.

    Applies a series of normalizers to resolve paths that may have
    common issues (redundant prefixes, wrong separators, etc.).

    Multi-Root Support:
    - Primary cwd is always checked first
    - Additional search roots can be added for nested project structures
    - Useful for monorepos or projects with multiple working directories
    - Example: project/project/utils/ structure where files could be
      in either root

    Attributes:
        cwd: Base directory for resolution (defaults to os.getcwd())
        additional_roots: Additional directories to search for paths
        normalizers: List of normalizer functions to apply
        _cache: Cache for resolved paths
    """

    cwd: Optional[Path] = None
    additional_roots: List[Path] = field(default_factory=list)
    normalizers: List[Callable[[str, Path], Tuple[Optional[str], str]]] = field(
        default_factory=list
    )
    _cache: Dict[str, PathResolution] = field(default_factory=dict)
    _known_paths: Optional[Set[str]] = None  # For fuzzy matching

    def __post_init__(self) -> None:
        """Initialize with defaults if not provided."""
        if self.cwd is None:
            self.cwd = Path.cwd()
        else:
            self.cwd = Path(self.cwd).resolve()

        # Convert additional_roots to resolved Paths
        self.additional_roots = [
            Path(root).resolve() if not isinstance(root, Path) else root.resolve()
            for root in self.additional_roots
        ]

        # Auto-detect nested project structure (project/project pattern)
        self._auto_detect_nested_roots()

        if not self.normalizers:
            self.normalizers = [
                normalize_separators,
                restore_absolute_path,  # Handle LLMs dropping leading /
                strip_human_root_prefix,
                strip_cwd_prefix,
                strip_first_component,
                strip_common_prefix,
            ]

    def _auto_detect_nested_roots(self) -> None:
        """Auto-detect nested project structure and add as additional root.

        Handles common pattern where project structure is:
            my_project/
                my_project/
                    __init__.py
                    utils/
                tests/
                setup.py

        In this case, if cwd is my_project/, we should also search in
        my_project/my_project/ for files referenced as 'utils/foo.py'.
        """
        # Check for directory with same name as cwd
        cwd_name = self.cwd.name
        nested_dir = self.cwd / cwd_name

        if nested_dir.is_dir() and nested_dir not in self.additional_roots:
            # Check if it looks like a Python package (has __init__.py or .py files)
            has_python_files = any(nested_dir.glob("*.py")) or any(
                nested_dir.glob("**/__init__.py")
            )

            if has_python_files:
                self.additional_roots.append(nested_dir)
                logger.debug(
                    f"PathResolver: Auto-detected nested project structure, "
                    f"added search root: {nested_dir}"
                )

    def _apply_normalizers(self, path: str) -> Tuple[str, Optional[str]]:
        """Apply normalizers in sequence until one succeeds.

        Args:
            path: Path to normalize

        Returns:
            Tuple of (normalized_path, description or None)
        """
        current_path = path
        applied_normalizations: List[str] = []

        for normalizer in self.normalizers:
            result, description = normalizer(current_path, self.cwd)
            if result is not None:
                current_path = result
                applied_normalizations.append(description)

        if applied_normalizations:
            return current_path, ", ".join(applied_normalizations)
        return path, None

    def resolve(self, path: str, must_exist: bool = True) -> PathResolution:
        """Resolve a path with normalization and multi-root search.

        Resolution order:
        1. Try path directly from cwd
        2. Try path from each additional_root
        3. Apply normalizers and try again from cwd
        4. Apply normalizers and try from each additional_root

        Args:
            path: Path to resolve
            must_exist: If True, raises FileNotFoundError if not found

        Returns:
            PathResolution with resolved path

        Raises:
            FileNotFoundError: If must_exist and path not found
        """
        # Check cache
        cache_key = f"{path}:{must_exist}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Handle empty path
        if not path:
            result = PathResolution(
                original_path=path,
                resolved_path=self.cwd,
                exists=True,
                is_directory=True,
            )
            self._cache[cache_key] = result
            return result

        # Build list of search roots: cwd first, then additional roots
        search_roots = [self.cwd] + self.additional_roots

        # Try original path from each root
        for root in search_roots:
            try:
                resolved = Path(path).expanduser()
                if not resolved.is_absolute():
                    resolved = (root / resolved).resolve()
                else:
                    resolved = resolved.resolve()

                if resolved.exists():
                    result = PathResolution(
                        original_path=path,
                        resolved_path=resolved,
                        was_normalized=(root != self.cwd),
                        normalization_applied=(
                            f"resolved_from:{root.name}" if root != self.cwd else None
                        ),
                        exists=True,
                        is_file=resolved.is_file(),
                        is_directory=resolved.is_dir(),
                    )
                    self._cache[cache_key] = result
                    return result
            except (OSError, ValueError):
                continue

        # Apply normalizers and try each root again
        normalized, description = self._apply_normalizers(path)

        if normalized != path:
            for root in search_roots:
                try:
                    resolved = (root / normalized).resolve()
                    if resolved.exists():
                        # Build combined description
                        full_description = description
                        if root != self.cwd:
                            full_description = f"{description}, resolved_from:{root.name}"

                        result = PathResolution(
                            original_path=path,
                            resolved_path=resolved,
                            was_normalized=True,
                            normalization_applied=full_description,
                            exists=True,
                            is_file=resolved.is_file(),
                            is_directory=resolved.is_dir(),
                        )
                        logger.debug(f"Path resolved: {result}")
                        self._cache[cache_key] = result
                        return result
                except (OSError, ValueError):
                    continue

        # Path not found
        if must_exist:
            # Provide helpful error with suggestions
            suggestions = self.suggest_similar(path, limit=3)
            suggestion_text = ""
            if suggestions:
                suggestion_text = f"\nDid you mean: {', '.join(suggestions)}"
            raise FileNotFoundError(f"Path not found: {path}{suggestion_text}")

        # Return non-existent path resolution
        try:
            resolved = Path(path).expanduser()
            if not resolved.is_absolute():
                resolved = (self.cwd / resolved).resolve()
        except (OSError, ValueError):
            resolved = self.cwd / path

        result = PathResolution(
            original_path=path,
            resolved_path=resolved,
            exists=False,
        )
        self._cache[cache_key] = result
        return result

    def resolve_file(self, path: str) -> PathResolution:
        """Resolve a file path.

        Args:
            path: File path to resolve

        Returns:
            PathResolution for an existing file

        Raises:
            FileNotFoundError: If file doesn't exist
            IsADirectoryError: If path is a directory
        """
        result = self.resolve(path, must_exist=True)

        if result.is_directory:
            raise IsADirectoryError(
                f"Cannot read directory as file: {path}\n"
                f"Suggestion: Use ls(path='{path}') to explore its contents."
            )

        return result

    def resolve_directory(self, path: str) -> PathResolution:
        """Resolve a directory path.

        Args:
            path: Directory path to resolve

        Returns:
            PathResolution for an existing directory

        Raises:
            FileNotFoundError: If directory doesn't exist
            NotADirectoryError: If path is a file
        """
        result = self.resolve(path, must_exist=True)

        if result.is_file:
            raise NotADirectoryError(
                f"Path is not a directory: {path}\n"
                f"Suggestion: Use read(path='{path}') to read file contents."
            )

        return result

    def suggest_similar(self, path: str, limit: int = 5) -> List[str]:
        """Suggest similar paths that exist.

        Uses difflib for fuzzy matching against known paths in cwd.

        Args:
            path: Non-existent path to find matches for
            limit: Maximum suggestions to return

        Returns:
            List of similar existing paths
        """
        suggestions: List[str] = []

        # Prefer package/module entry files when the requested module path is
        # actually backed by a same-named directory (e.g. foo.py -> foo/base.py).
        suggestions.extend(self._suggest_package_entry_paths(path, limit=limit))

        # Build known paths if not cached
        if self._known_paths is None:
            self._known_paths = self._scan_directory_names()

        # Get filename component for matching
        path_obj = Path(path)
        filename = path_obj.name
        full_paths = self._get_relative_file_paths()

        # For duplicated manifest names (Cargo.toml, package.json, etc.), a bare
        # filename is not actionable. Prefer concrete relative paths that exist.
        exact_relative_matches = [
            candidate
            for candidate in full_paths
            if Path(candidate.rstrip("/")).name == filename and candidate.rstrip("/") != path
        ]
        suggestions.extend(exact_relative_matches[:limit])

        # Find close matches
        matches = get_close_matches(
            filename,
            list(self._known_paths),
            n=limit,
            cutoff=0.5,
        )

        # If the path has multiple components, try matching against full paths
        if "/" in path or "\\" in path:
            # Also try matching the full path
            full_matches = get_close_matches(
                path,
                full_paths,
                n=limit,
                cutoff=0.4,
            )
            matches = list(dict.fromkeys(full_matches + matches))[:limit]

        suggestions.extend(matches)
        return [
            candidate
            for candidate in dict.fromkeys(suggestions)
            if candidate.rstrip("/") != path.rstrip("/")
        ][:limit]

    def _suggest_package_entry_paths(self, path: str, limit: int = 5) -> List[str]:
        """Suggest package entry files for missing module-style paths.

        This improves recoveries such as `registry.py` where the actual source
        lives in a sibling package directory like `registry/base.py`.
        """
        requested = Path(path).expanduser()
        if requested.suffix != ".py":
            return []

        package_dir = requested.with_suffix("")
        if not package_dir.is_absolute():
            package_dir = self.cwd / package_dir

        try:
            package_dir = package_dir.resolve()
        except (OSError, RuntimeError):
            return []

        if not package_dir.is_dir():
            return []

        preferred_names = ("base.py", "__init__.py")
        suggestions: List[str] = []

        for name in preferred_names:
            candidate = package_dir / name
            if candidate.is_file():
                suggestions.append(self._to_relative_display_path(candidate))

        other_candidates = sorted(
            candidate
            for candidate in package_dir.iterdir()
            if candidate.is_file()
            and candidate.suffix == ".py"
            and candidate.name not in preferred_names
        )
        suggestions.extend(
            self._to_relative_display_path(candidate) for candidate in other_candidates
        )
        return suggestions[:limit]

    def _to_relative_display_path(self, path: Path) -> str:
        """Format a path relative to cwd when possible for tool suggestions."""
        try:
            return str(path.relative_to(self.cwd))
        except ValueError:
            return str(path)

    def _scan_directory_names(self, max_depth: int = 3) -> Set[str]:
        """Scan directory for file/directory names.

        Args:
            max_depth: Maximum directory depth to scan

        Returns:
            Set of file and directory names
        """
        names: Set[str] = set()

        def _scan(path: Path, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                for entry in path.iterdir():
                    if entry.name.startswith("."):
                        continue
                    names.add(entry.name)
                    if entry.is_dir():
                        _scan(entry, depth + 1)
            except PermissionError:
                pass

        _scan(self.cwd, 0)
        return names

    def _get_relative_file_paths(self, max_depth: int = 4) -> List[str]:
        """Get relative file paths from cwd.

        Args:
            max_depth: Maximum depth to scan

        Returns:
            List of relative file paths
        """
        paths: List[str] = []

        def _scan(path: Path, depth: int, prefix: str) -> None:
            if depth > max_depth:
                return
            try:
                for entry in path.iterdir():
                    if entry.name.startswith("."):
                        continue
                    rel_path = f"{prefix}/{entry.name}" if prefix else entry.name
                    if entry.is_file():
                        paths.append(rel_path)
                    elif entry.is_dir():
                        paths.append(rel_path + "/")
                        _scan(entry, depth + 1, rel_path)
            except PermissionError:
                pass

        _scan(self.cwd, 0, "")
        return paths

    def set_cwd(self, cwd: Path) -> None:
        """Update the base directory.

        Clears all caches when cwd changes and re-runs auto-detection.

        Args:
            cwd: New working directory
        """
        self.cwd = Path(cwd).resolve()
        self.additional_roots = []  # Reset additional roots
        self._auto_detect_nested_roots()  # Re-detect nested structure
        self._cache.clear()
        self._known_paths = None

    def add_search_root(self, root: Path) -> None:
        """Add an additional search root.

        Useful for monorepos or projects with multiple working directories.

        Args:
            root: Additional directory to search for paths
        """
        resolved_root = Path(root).resolve()
        if resolved_root.is_dir() and resolved_root not in self.additional_roots:
            self.additional_roots.append(resolved_root)
            self._cache.clear()  # Clear cache since search paths changed
            self._known_paths = None
            logger.debug(f"PathResolver: Added search root: {resolved_root}")

    def remove_search_root(self, root: Path) -> bool:
        """Remove an additional search root.

        Args:
            root: Directory to remove from search paths

        Returns:
            True if root was removed, False if not found
        """
        resolved_root = Path(root).resolve()
        if resolved_root in self.additional_roots:
            self.additional_roots.remove(resolved_root)
            self._cache.clear()
            self._known_paths = None
            return True
        return False

    @property
    def search_roots(self) -> List[Path]:
        """Get all search roots (cwd + additional roots)."""
        return [self.cwd] + self.additional_roots

    def clear_cache(self) -> None:
        """Clear all cached resolutions."""
        self._cache.clear()
        self._known_paths = None


# =============================================================================
# Factory Function
# =============================================================================


def create_path_resolver(
    cwd: Optional[Path] = None,
    additional_roots: Optional[List[Path]] = None,
) -> PathResolver:
    """Create a configured PathResolver instance.

    Factory function for DI registration.

    Args:
        cwd: Base directory (defaults to os.getcwd())
        additional_roots: Additional directories to search for paths

    Returns:
        Configured PathResolver instance with multi-root support
    """
    return PathResolver(
        cwd=cwd,
        additional_roots=additional_roots or [],
    )
