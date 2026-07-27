# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Filesystem tools for reading, writing, and listing contents."""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

import aiofiles

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool

# PathResolver for centralized path normalization
from victor.protocols.path_resolver import PathResolver, create_path_resolver

logger = logging.getLogger(__name__)

# Global PathResolver instance (lazy initialized)
_path_resolver: Optional[PathResolver] = None


def get_path_resolver() -> PathResolver:
    """Get or create the global PathResolver instance."""
    global _path_resolver
    if _path_resolver is None:
        _path_resolver = create_path_resolver()
    return _path_resolver


def reset_path_resolver() -> None:
    """Reset the path resolver (useful when cwd changes)."""
    global _path_resolver
    _path_resolver = None


def _list_available_directories(base_path: Path, max_count: int = 10) -> List[str]:
    """List available directories at the given path level.

    Args:
        base_path: Parent path to search
        max_count: Maximum number of directories to return

    Returns:
        List of directory names
    """
    if not base_path.exists():
        return []

    try:
        return [
            d.name for d in sorted(base_path.iterdir()) if d.is_dir() and not d.name.startswith(".")
        ][:max_count]
    except PermissionError:
        return []


def _validate_directory_with_suggestions(path: Path) -> None:
    """Validate directory exists, providing helpful suggestions if not.

    Args:
        path: Path to validate

    Raises:
        FileNotFoundError: With helpful suggestions if directory doesn't exist
    """
    if path.exists():
        return

    # Try to find similar directories
    if path.parent.exists():
        available = _list_available_directories(path.parent)
        if available:
            # Find directories with similar names
            from difflib import get_close_matches

            suggestions = get_close_matches(path.name, available, n=3, cutoff=0.6)
            if suggestions:
                raise FileNotFoundError(
                    f"Directory not found: {path}\n\n"
                    f"Did you mean one of these?\n  - "
                    + "\n  - ".join(suggestions)
                    + f"\n\nAvailable directories in {path.parent.name}:\n  - "
                    + "\n  - ".join(available[:5])
                )

    # Fallback: general error
    raise FileNotFoundError(
        f"Directory not found: {path}\n\n"
        f"Tip: Check the path is correct and the directory exists."
    )


# ============================================================================
# PATH NORMALIZATION
# ============================================================================
# LLMs sometimes use paths with redundant prefixes when working in subdirectories.
# For example, when cwd is "investor_homelab/", they might use "investor_homelab/utils"
# instead of just "utils". This helper delegates to PathResolver for consistency.


def _normalize_path(path: str) -> Optional[str]:
    """Normalize a path by stripping redundant directory prefixes.

    When working in a subdirectory, LLMs sometimes include the directory name
    in paths (e.g., "project/utils" when cwd is already "project/").
    This function delegates to PathResolver for consistent normalization.

    Args:
        path: The original path that doesn't exist

    Returns:
        A normalized path if a valid one is found, None otherwise

    Note:
        This function is maintained for backward compatibility.
        New code should use PathResolver directly.
    """
    if not path or path.startswith("/") or path.startswith("~"):
        return None  # Absolute paths shouldn't be normalized

    try:
        resolver = get_path_resolver()
        result = resolver.resolve(path, must_exist=True)

        # If the path was normalized, return the relative path
        if result.was_normalized:
            # Return relative path string
            try:
                return str(result.resolved_path.relative_to(Path.cwd()))
            except ValueError:
                return str(result.resolved_path)

        return None
    except FileNotFoundError:
        # Path doesn't exist even after normalization - try legacy fallback
        # for cases where we just want a normalized path string (not validation)
        cwd = Path.cwd()
        cwd_name = cwd.name

        # Check if path starts with cwd name (common pattern)
        if path.startswith(f"{cwd_name}/"):
            stripped = path[len(cwd_name) + 1 :]
            if stripped and (cwd / stripped).exists():
                return stripped

        # Try stripping first path component
        parts = Path(path).parts
        if len(parts) > 1:
            stripped = str(Path(*parts[1:]))
            if (cwd / stripped).exists():
                return stripped

        return None


# ============================================================================
# SESSION-LEVEL FILE CONTENT CACHE (P3-2)
# ============================================================================
# Prevents redundant file reads within a session. Especially helpful for
# providers like DeepSeek that tend to re-read the same files repeatedly.
# Cache is keyed by normalized absolute path and tracks modification time.


@dataclass
class CachedFileContent:
    """Cached file content with metadata for invalidation."""

    content: str
    mtime: float  # File modification time when cached
    cached_at: float  # When the cache entry was created
    size: int  # File size in bytes
    hits: int = 0  # Number of cache hits


@dataclass
class FileContentCacheStats:
    """Statistics for the file content cache."""

    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    evictions: int = 0
    total_bytes_cached: int = 0
    total_bytes_saved: int = 0  # Bytes not re-read due to cache hits


class FileContentCache:
    """Session-level cache for file contents.

    NOTE: DISABLED by default (_cache_enabled=False).
    The OS page cache is more efficient for this purpose - it keeps
    recently accessed file pages in memory at the kernel level with
    better memory management across the entire system.

    This cache adds unnecessary memory pressure in the Python process
    for minimal benefit. Re-enable only if profiling shows a proven
    bottleneck in file I/O for repeated reads of the same files.

    Features (when enabled):
    - Caches file contents keyed by normalized absolute path
    - Auto-invalidates when file modification time changes
    - Thread-safe operations
    - Memory-bounded with LRU eviction
    - Tracks hit/miss statistics
    """

    def __init__(
        self,
        max_entries: int = 100,
        max_total_bytes: int = 10_000_000,  # 10MB default
        ttl_seconds: int = 300,  # 5 minute TTL
    ):
        """Initialize file content cache.

        Args:
            max_entries: Maximum number of files to cache
            max_total_bytes: Maximum total bytes to cache (soft limit)
            ttl_seconds: Time-to-live for cache entries
        """
        self._cache: Dict[str, CachedFileContent] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._max_total_bytes = max_total_bytes
        self._ttl_seconds = ttl_seconds
        self._stats = FileContentCacheStats()
        self._access_order: List[str] = []  # For LRU eviction

    def _normalize_path(self, path: str) -> str:
        """Normalize path to absolute resolved form."""
        return str(Path(path).expanduser().resolve())

    def get(self, path: str) -> Optional[str]:
        """Get cached content if valid.

        Validates:
        - File still exists
        - Modification time hasn't changed
        - TTL hasn't expired

        Args:
            path: File path (will be normalized)

        Returns:
            Cached content or None if miss/invalid
        """
        normalized = self._normalize_path(path)

        with self._lock:
            entry = self._cache.get(normalized)
            if entry is None:
                self._stats.misses += 1
                return None

            # Check TTL
            if time.time() - entry.cached_at > self._ttl_seconds:
                self._invalidate_entry(normalized, reason="ttl_expired")
                self._stats.misses += 1
                return None

            # Check if file still exists and mtime matches
            try:
                file_path = Path(normalized)
                if not file_path.exists():
                    self._invalidate_entry(normalized, reason="file_deleted")
                    self._stats.misses += 1
                    return None

                current_mtime = file_path.stat().st_mtime
                if current_mtime != entry.mtime:
                    self._invalidate_entry(normalized, reason="mtime_changed")
                    self._stats.misses += 1
                    return None

            except OSError:
                self._invalidate_entry(normalized, reason="os_error")
                self._stats.misses += 1
                return None

            # Cache hit!
            entry.hits += 1
            self._stats.hits += 1
            self._stats.total_bytes_saved += entry.size

            # Update access order for LRU
            if normalized in self._access_order:
                self._access_order.remove(normalized)
            self._access_order.append(normalized)

            logger.debug(
                "File cache HIT: %s (hits=%d, saved=%d bytes)",
                normalized,
                entry.hits,
                entry.size,
            )
            return entry.content

    def set(self, path: str, content: str, mtime: float, size: int) -> bool:
        """Cache file content.

        May trigger LRU eviction if cache is full.

        Args:
            path: File path (will be normalized)
            content: File content to cache
            mtime: File modification time
            size: File size in bytes

        Returns:
            True if cached successfully
        """
        normalized = self._normalize_path(path)

        with self._lock:
            # Check if we need to evict entries
            self._maybe_evict(size)

            # Store the entry
            self._cache[normalized] = CachedFileContent(
                content=content,
                mtime=mtime,
                cached_at=time.time(),
                size=size,
                hits=0,
            )
            self._stats.total_bytes_cached += size

            # Update access order
            if normalized in self._access_order:
                self._access_order.remove(normalized)
            self._access_order.append(normalized)

            logger.debug("File cache SET: %s (%d bytes)", normalized, size)
            return True

    def invalidate(self, path: str) -> bool:
        """Invalidate a cache entry.

        Call this when a file is written to ensure fresh reads.

        Args:
            path: File path to invalidate

        Returns:
            True if entry was found and invalidated
        """
        normalized = self._normalize_path(path)
        with self._lock:
            return self._invalidate_entry(normalized, reason="explicit_invalidate")

    def _invalidate_entry(self, normalized: str, reason: str = "") -> bool:
        """Internal invalidation (already holds lock)."""
        entry = self._cache.pop(normalized, None)
        if entry:
            self._stats.invalidations += 1
            self._stats.total_bytes_cached -= entry.size
            if normalized in self._access_order:
                self._access_order.remove(normalized)
            logger.debug("File cache INVALIDATE: %s (reason=%s)", normalized, reason)
            return True
        return False

    def _maybe_evict(self, incoming_bytes: int) -> None:
        """Evict entries if necessary (already holds lock)."""
        # Check entry count
        while len(self._cache) >= self._max_entries and self._access_order:
            oldest = self._access_order.pop(0)
            entry = self._cache.pop(oldest, None)
            if entry:
                self._stats.evictions += 1
                self._stats.total_bytes_cached -= entry.size
                logger.debug("File cache EVICT (count): %s", oldest)

        # Check total bytes (soft limit - only evict if significantly over)
        while (
            self._stats.total_bytes_cached + incoming_bytes > self._max_total_bytes * 1.2
            and self._access_order
        ):
            oldest = self._access_order.pop(0)
            entry = self._cache.pop(oldest, None)
            if entry:
                self._stats.evictions += 1
                self._stats.total_bytes_cached -= entry.size
                logger.debug("File cache EVICT (bytes): %s", oldest)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            self._stats.total_bytes_cached = 0
            logger.info("File cache CLEARED: %d entries removed", count)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (
                self._stats.hits / (self._stats.hits + self._stats.misses) * 100
                if (self._stats.hits + self._stats.misses) > 0
                else 0.0
            )
            return {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "hit_rate_percent": round(hit_rate, 1),
                "invalidations": self._stats.invalidations,
                "evictions": self._stats.evictions,
                "entries": len(self._cache),
                "total_bytes_cached": self._stats.total_bytes_cached,
                "total_bytes_saved": self._stats.total_bytes_saved,
                "max_entries": self._max_entries,
                "max_total_bytes": self._max_total_bytes,
            }

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)


# Global file content cache instance (session-level)
# DISABLED by default - OS page cache is more efficient.
# Can be re-enabled via settings if needed for specific use cases.
_file_content_cache: Optional[FileContentCache] = None
_cache_enabled: bool = False  # Disabled - OS page cache handles this better


# ============================================================================
# SANDBOX PATH ENFORCEMENT (P4 - Explore/Plan Mode Restrictions)
# ============================================================================
# In EXPLORE and PLAN modes, file writes are restricted to the sandbox directory
# (.victor/sandbox/) to prevent accidental modifications to the main codebase.


def get_sandbox_path() -> Optional[Path]:
    """Get the sandbox directory path for the current mode.

    Returns:
        Path to sandbox directory if in restricted mode, None if unrestricted.
    """
    try:
        from victor.agent.mode_controller import get_mode_controller

        controller = get_mode_controller()
        config = controller.config

        if config.sandbox_dir and config.allow_sandbox_edits:
            # Sandbox is enabled for this mode
            return Path.cwd() / config.sandbox_dir
        elif not config.allow_all_tools and (
            "write_file" in config.disallowed_tools or "edit_files" in config.disallowed_tools
        ):
            # Writing is disallowed in this mode (no sandbox exception)
            return None

        # Unrestricted mode (BUILD) - no sandbox enforcement
        return None
    except Exception:
        # If mode controller not available, default to unrestricted
        return None


def enforce_sandbox_path(file_path: Path) -> None:
    """Enforce sandbox restrictions for file writes in restricted modes.

    Args:
        file_path: The resolved file path being written to

    Raises:
        PermissionError: If path is outside the active write policy or sandbox
    """
    from victor.tools.write_path_policy import get_active_write_policy

    policy = get_active_write_policy()
    if policy is not None:
        if not policy.allows(file_path):
            try:
                from victor.agent.mode_controller import get_mode_controller

                mode = get_mode_controller().current_mode.value.upper()
            except Exception:
                mode = "RESTRICTED"
            raise PermissionError(
                f"[{mode} MODE] Cannot write to '{file_path}'.\n"
                f"Analysis-safe paths: .victor/analysis/, ./tmp/, ./docs/*.md, "
                f"./ANALYSIS-*.md, ./CHECKPOINT-*.md, /tmp/**"
            )
        return  # Policy satisfied — skip legacy sandbox check

    # Legacy sandbox path (no policy set — unchanged behavior)
    sandbox = get_sandbox_path()

    if sandbox is None:
        return

    sandbox.mkdir(parents=True, exist_ok=True)

    try:
        file_path.resolve().relative_to(sandbox.resolve())
    except ValueError:
        try:
            from victor.agent.mode_controller import get_mode_controller

            mode = get_mode_controller().current_mode.value.upper()
        except Exception:
            mode = "RESTRICTED"
        raise PermissionError(
            f"[{mode} MODE] Cannot write to '{file_path}'.\n"
            f"In {mode} mode, edits are restricted to the sandbox directory: {sandbox}\n"
            f"Use /mode build to switch to build mode for unrestricted file access.\n"
            f"Or prefix your path with .victor/sandbox/ to write within the sandbox."
        )


def get_file_content_cache() -> FileContentCache:
    """Get or create the global file content cache."""
    global _file_content_cache
    if _file_content_cache is None:
        _file_content_cache = FileContentCache()
    return _file_content_cache


def clear_file_content_cache(reset_stats: bool = True) -> None:
    """Clear the global file content cache (call on session end).

    Args:
        reset_stats: If True (default), also reset statistics.
                     Set to False to preserve stats across clear.
    """
    global _file_content_cache
    if _file_content_cache is not None:
        _file_content_cache.clear()
        if reset_stats:
            # Reset the stats for a fresh session
            _file_content_cache._stats = FileContentCacheStats()


def is_file_cache_enabled() -> bool:
    """Check if file content cache is enabled."""
    return _cache_enabled


# ============================================================================
# READ-BEFORE-WRITE STATE TRACKING (P0 - overwrite safety)
# ============================================================================
# Tracks which files the agent has actually READ this session (with the mtime
# observed at read time) so we can refuse to overwrite an existing file that
# was never read — the inverse of Claude Code's "Overwriting an existing file
# you haven't Read will fail". This is independent of FileContentCache because
# the cache is disabled by default (_cache_enabled=False); the read-state
# tracker is always on.


class ReadStateTracker:
    """Thread-safe record of files read this session, keyed by normalized path.

    Stores the mtime observed at read time so a later external modification
    (mtime changed) re-invalidates the read even within the same session.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # normalized absolute path -> mtime observed at read time
        self._read_mtime: Dict[str, float] = {}

    @staticmethod
    def _normalize(path: str) -> str:
        try:
            return str(Path(path).expanduser().resolve())
        except Exception:
            return str(path)

    def record(self, path: str) -> None:
        """Mark a path as read, capturing its current mtime.

        No-op if the file does not exist (stat failed) — only real reads count.
        """
        normalized = self._normalize(path)
        try:
            mtime = Path(normalized).stat().st_mtime
        except OSError:
            return
        with self._lock:
            self._read_mtime[normalized] = mtime

    def is_current(self, path: str) -> bool:
        """True iff path was read this session and has not changed since.

        Returns False when the file was never read, was deleted, changed on
        disk (mtime differs), or stat fails — i.e. the safe default is "unread".
        """
        normalized = self._normalize(path)
        with self._lock:
            recorded = self._read_mtime.get(normalized)
        if recorded is None:
            return False
        try:
            return Path(normalized).stat().st_mtime == recorded
        except OSError:
            return False

    def clear(self) -> None:
        with self._lock:
            self._read_mtime.clear()


_read_state_tracker: Optional[ReadStateTracker] = None


def get_read_state() -> ReadStateTracker:
    """Get or create the global read-state tracker (always-on, not cache-gated)."""
    global _read_state_tracker
    if _read_state_tracker is None:
        _read_state_tracker = ReadStateTracker()
    return _read_state_tracker


def record_read(path: str) -> None:
    """Record that a file was read this session. Called from read()."""
    get_read_state().record(path)


def _read_before_write_enabled() -> bool:
    """Master switch for the read-before-overwrite guard.

    Default ON. Set VICTOR_ENFORCE_READ_BEFORE_WRITE=0 (or false/no/off) to
    disable for batch evals/benchmarks that legitimately overwrite unread files.
    """
    raw = os.environ.get("VICTOR_ENFORCE_READ_BEFORE_WRITE", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def enforce_read_before_write(file_path: Path, *, force: bool = False) -> None:
    """Refuse to overwrite an existing file that was not read this session.

    Creating a new file is always allowed. An explicit ``force=True`` opts out
    (intentional full rewrite / generated output). Raises PermissionError when
    the guard blocks the write so the caller surfaces a retryable error.
    """
    if force or not _read_before_write_enabled():
        return
    try:
        if not file_path.exists():
            return  # new file — always allowed
    except OSError:
        return  # cannot stat — don't block on an unrelated error
    if get_read_state().is_current(str(file_path)):
        return  # read this session and unchanged
    raise PermissionError(
        f"Refusing to overwrite existing file '{file_path}' that was not read "
        f"in this session (or changed since read). Read it first, then edit, or "
        f"pass force=True to acknowledge the overwrite."
    )


# ============================================================================
# FILE TYPE DETECTION SYSTEM
# ============================================================================
# Extensible architecture for detecting file types via magic bytes and extensions.
# Future: Add handlers for PDF, images, archives, etc. via the handler registry.


class FileCategory(Enum):
    """High-level file category for routing to appropriate handlers."""

    TEXT = auto()  # Plain text, code, config files
    DOCUMENT = auto()  # PDF, DOCX, ODT, etc.
    IMAGE = auto()  # PNG, JPEG, GIF, WebP, etc.
    ARCHIVE = auto()  # ZIP, TAR, RAR, 7Z, etc.
    DATABASE = auto()  # SQLite, etc.
    MEDIA = auto()  # Audio/video files
    COMPILED = auto()  # Executables, bytecode, shared libraries
    DATA = auto()  # Generic binary data


@dataclass
class FileTypeInfo:
    """Information about a detected file type."""

    category: FileCategory
    mime_type: str
    description: str
    extensions: Tuple[str, ...]  # Expected extensions for this type
    magic_bytes: Optional[bytes] = None  # Magic signature if detected
    magic_offset: int = 0  # Offset where magic bytes are found
    suggestion: str = ""  # Help message for unsupported types


# Magic bytes signatures for common file types
# Format: (magic_bytes, offset, FileTypeInfo)
# Order matters: more specific signatures should come first
MAGIC_SIGNATURES: List[Tuple[bytes, int, FileTypeInfo]] = [
    # Documents
    (
        b"%PDF",
        0,
        FileTypeInfo(
            category=FileCategory.DOCUMENT,
            mime_type="application/pdf",
            description="PDF document",
            extensions=(".pdf",),
            suggestion="PDF reading support coming soon. For now, use `pdftotext` externally.",
        ),
    ),
    (
        b"PK\x03\x04",
        0,
        FileTypeInfo(
            category=FileCategory.ARCHIVE,
            mime_type="application/zip",
            description="ZIP archive (or DOCX/XLSX/PPTX)",
            extensions=(
                ".zip",
                ".docx",
                ".xlsx",
                ".pptx",
                ".odt",
                ".ods",
                ".odp",
                ".jar",
                ".whl",
                ".apk",
            ),
            suggestion="Use `unzip -l` to list contents, or extract and read individual files.",
        ),
    ),
    (
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
        0,
        FileTypeInfo(
            category=FileCategory.DOCUMENT,
            mime_type="application/msword",
            description="Microsoft Office legacy format (DOC/XLS/PPT)",
            extensions=(".doc", ".xls", ".ppt"),
            suggestion="Convert to .docx/.xlsx/.pptx or use external tools to extract text.",
        ),
    ),
    # Images
    (
        b"\x89PNG\r\n\x1a\n",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/png",
            description="PNG image",
            extensions=(".png",),
            suggestion="Image viewing support coming soon. Check for related text/alt-text files.",
        ),
    ),
    (
        b"\xff\xd8\xff",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/jpeg",
            description="JPEG image",
            extensions=(".jpg", ".jpeg"),
            suggestion="Image viewing support coming soon. Check for related text/alt-text files.",
        ),
    ),
    (
        b"GIF87a",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/gif",
            description="GIF image (GIF87a)",
            extensions=(".gif",),
            suggestion="Image viewing support coming soon.",
        ),
    ),
    (
        b"GIF89a",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/gif",
            description="GIF image (GIF89a)",
            extensions=(".gif",),
            suggestion="Image viewing support coming soon.",
        ),
    ),
    (
        b"RIFF",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/webp",
            description="WebP image (RIFF container)",
            extensions=(".webp",),
            suggestion="Image viewing support coming soon.",
        ),
    ),
    (
        b"BM",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/bmp",
            description="BMP image",
            extensions=(".bmp",),
            suggestion="Image viewing support coming soon.",
        ),
    ),
    (
        b"II*\x00",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/tiff",
            description="TIFF image (little-endian)",
            extensions=(".tiff", ".tif"),
            suggestion="Image viewing support coming soon.",
        ),
    ),
    (
        b"MM\x00*",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/tiff",
            description="TIFF image (big-endian)",
            extensions=(".tiff", ".tif"),
            suggestion="Image viewing support coming soon.",
        ),
    ),
    (
        b"\x00\x00\x01\x00",
        0,
        FileTypeInfo(
            category=FileCategory.IMAGE,
            mime_type="image/x-icon",
            description="ICO icon file",
            extensions=(".ico",),
            suggestion="Icon files are binary images. Check for SVG alternatives.",
        ),
    ),
    # Archives
    (
        b"Rar!\x1a\x07",
        0,
        FileTypeInfo(
            category=FileCategory.ARCHIVE,
            mime_type="application/x-rar-compressed",
            description="RAR archive",
            extensions=(".rar",),
            suggestion="Use `unrar l` to list contents.",
        ),
    ),
    (
        b"7z\xbc\xaf\x27\x1c",
        0,
        FileTypeInfo(
            category=FileCategory.ARCHIVE,
            mime_type="application/x-7z-compressed",
            description="7-Zip archive",
            extensions=(".7z",),
            suggestion="Use `7z l` to list contents.",
        ),
    ),
    (
        b"\x1f\x8b",
        0,
        FileTypeInfo(
            category=FileCategory.ARCHIVE,
            mime_type="application/gzip",
            description="Gzip compressed file",
            extensions=(".gz", ".tgz"),
            suggestion="Use `zcat` to view contents, or `tar -tzf` for tarballs.",
        ),
    ),
    (
        b"BZh",
        0,
        FileTypeInfo(
            category=FileCategory.ARCHIVE,
            mime_type="application/x-bzip2",
            description="Bzip2 compressed file",
            extensions=(".bz2",),
            suggestion="Use `bzcat` to view contents.",
        ),
    ),
    (
        b"\xfd7zXZ\x00",
        0,
        FileTypeInfo(
            category=FileCategory.ARCHIVE,
            mime_type="application/x-xz",
            description="XZ compressed file",
            extensions=(".xz",),
            suggestion="Use `xzcat` to view contents.",
        ),
    ),
    # Database
    (
        b"SQLite format 3",
        0,
        FileTypeInfo(
            category=FileCategory.DATABASE,
            mime_type="application/x-sqlite3",
            description="SQLite database",
            extensions=(".db", ".sqlite", ".sqlite3"),
            suggestion="Use shell(cmd='sqlite3 file.db \".tables\"') to list tables, or database(action='connect', connection=DatabaseConnection(db_type='sqlite', database='file.db')) for structured access.",
        ),
    ),
    # Media
    (
        b"ID3",
        0,
        FileTypeInfo(
            category=FileCategory.MEDIA,
            mime_type="audio/mpeg",
            description="MP3 audio (ID3 tag)",
            extensions=(".mp3",),
            suggestion="Media files cannot be read as text.",
        ),
    ),
    (
        b"\xff\xfb",
        0,
        FileTypeInfo(
            category=FileCategory.MEDIA,
            mime_type="audio/mpeg",
            description="MP3 audio (frame sync)",
            extensions=(".mp3",),
            suggestion="Media files cannot be read as text.",
        ),
    ),
    (
        b"fLaC",
        0,
        FileTypeInfo(
            category=FileCategory.MEDIA,
            mime_type="audio/flac",
            description="FLAC audio",
            extensions=(".flac",),
            suggestion="Media files cannot be read as text.",
        ),
    ),
    (
        b"OggS",
        0,
        FileTypeInfo(
            category=FileCategory.MEDIA,
            mime_type="audio/ogg",
            description="Ogg container (audio/video)",
            extensions=(".ogg", ".ogv", ".oga"),
            suggestion="Media files cannot be read as text.",
        ),
    ),
    # Compiled/executables
    (
        b"\x7fELF",
        0,
        FileTypeInfo(
            category=FileCategory.COMPILED,
            mime_type="application/x-elf",
            description="ELF executable/shared library",
            extensions=(".so", ".o", ""),
            suggestion="This is a compiled binary. Check for source files instead.",
        ),
    ),
    (
        b"MZ",
        0,
        FileTypeInfo(
            category=FileCategory.COMPILED,
            mime_type="application/x-msdownload",
            description="Windows executable (PE/MZ)",
            extensions=(".exe", ".dll"),
            suggestion="This is a compiled binary. Check for source files instead.",
        ),
    ),
    (
        b"\xca\xfe\xba\xbe",
        0,
        FileTypeInfo(
            category=FileCategory.COMPILED,
            mime_type="application/java-archive",
            description="Java class file / Mach-O fat binary",
            extensions=(".class",),
            suggestion="This is compiled Java bytecode. Check for .java source files.",
        ),
    ),
    (
        b"\xcf\xfa\xed\xfe",
        0,
        FileTypeInfo(
            category=FileCategory.COMPILED,
            mime_type="application/x-mach-binary",
            description="Mach-O binary (64-bit)",
            extensions=("", ".dylib"),
            suggestion="This is a compiled macOS binary. Check for source files instead.",
        ),
    ),
    (
        b"\xce\xfa\xed\xfe",
        0,
        FileTypeInfo(
            category=FileCategory.COMPILED,
            mime_type="application/x-mach-binary",
            description="Mach-O binary (32-bit)",
            extensions=("", ".dylib"),
            suggestion="This is a compiled macOS binary. Check for source files instead.",
        ),
    ),
    # Python bytecode
    # Note: Python bytecode magic changes with versions, so we check extension primarily
]

# Maximum bytes to read for magic detection
MAGIC_BYTES_READ_SIZE = 32


def detect_file_type_by_magic(file_path: Path) -> Optional[FileTypeInfo]:
    """Detect file type by reading magic bytes from file header.

    Args:
        file_path: Path to the file to inspect

    Returns:
        FileTypeInfo if a known signature is found, None otherwise
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(MAGIC_BYTES_READ_SIZE)
    except (IOError, PermissionError):
        return None

    if not header:
        return None

    for magic_bytes, offset, type_info in MAGIC_SIGNATURES:
        if len(header) >= offset + len(magic_bytes):
            if header[offset : offset + len(magic_bytes)] == magic_bytes:
                # Return a copy with the detected magic bytes
                return FileTypeInfo(
                    category=type_info.category,
                    mime_type=type_info.mime_type,
                    description=type_info.description,
                    extensions=type_info.extensions,
                    magic_bytes=magic_bytes,
                    magic_offset=offset,
                    suggestion=type_info.suggestion,
                )

    return None


def check_extension_magic_mismatch(
    file_path: Path, magic_type: Optional[FileTypeInfo]
) -> Optional[str]:
    """Check if file extension matches detected magic bytes.

    Returns a warning message if there's a mismatch, None otherwise.
    """
    if magic_type is None:
        return None

    ext = file_path.suffix.lower()

    # If extension matches expected extensions, no warning
    if ext in magic_type.extensions:
        return None

    # Extension doesn't match magic bytes - potential mismatch
    expected_exts = ", ".join(magic_type.extensions) if magic_type.extensions else "(none)"
    return (
        f"Warning: File extension '{ext}' doesn't match detected content type.\n"
        f"Detected: {magic_type.description} (mime: {magic_type.mime_type})\n"
        f"Expected extensions: {expected_exts}\n"
        f"The file may have been renamed or misidentified."
    )


# ============================================================================
# BINARY FILE HANDLER REGISTRY (Extensible Architecture)
# ============================================================================
# Future handlers for different binary types can be registered here.
# Each handler takes a Path and returns extracted text content.

BinaryFileHandler = Callable[[Path], str]

# Registry for binary file handlers by category
# Future: Add handlers like:
#   FileCategory.DOCUMENT: pdf_handler, docx_handler
#   FileCategory.IMAGE: image_description_handler
#   FileCategory.ARCHIVE: archive_list_handler
_BINARY_HANDLERS: Dict[FileCategory, BinaryFileHandler] = {}


def register_binary_handler(category: FileCategory, handler: BinaryFileHandler) -> None:
    """Register a handler for a binary file category.

    Args:
        category: The file category this handler supports
        handler: Function that takes a Path and returns extracted text content

    Example:
        def pdf_handler(path: Path) -> str:
            # Extract text from PDF
            return extracted_text

        register_binary_handler(FileCategory.DOCUMENT, pdf_handler)
    """
    _BINARY_HANDLERS[category] = handler


def unregister_binary_handler(category: FileCategory) -> None:
    """Unregister a handler for a binary file category.

    Useful for test cleanup to avoid test pollution.

    Args:
        category: The file category to unregister
    """
    _BINARY_HANDLERS.pop(category, None)


def get_binary_handler(category: FileCategory) -> Optional[BinaryFileHandler]:
    """Get the handler for a binary file category, if one is registered."""
    return _BINARY_HANDLERS.get(category)


# ============================================================================
# FILE TYPE CONSTANTS
# ============================================================================

# Supported text/code file extensions
TEXT_EXTENSIONS = {
    # Code
    ".py",
    ".pyi",
    ".pyx",
    ".pxd",  # Python
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",  # JavaScript/TypeScript
    ".java",
    ".kt",
    ".kts",
    ".scala",
    ".groovy",  # JVM
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".hxx",  # Systems
    ".cs",
    ".fs",
    ".vb",  # .NET
    ".rb",
    ".php",
    ".pl",
    ".pm",
    ".lua",
    ".r",
    ".R",  # Scripting
    ".swift",
    ".m",
    ".mm",  # Apple
    ".sol",
    ".vy",  # Blockchain
    # Config
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",
    ".xsl",
    ".xsd",
    ".dtd",
    ".env",
    ".env.example",
    ".env.local",
    ".properties",
    ".plist",
    # Web
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",
    ".svelte",
    ".astro",
    # Data
    ".csv",
    ".tsv",
    ".sql",
    # Docs
    ".md",
    ".markdown",
    ".rst",
    ".txt",
    ".adoc",
    # Shell
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".bat",
    ".cmd",
    # Build
    ".make",
    ".cmake",
    ".gradle",
    ".sbt",
    ".dockerfile",
    ".containerfile",
    # Other
    ".graphql",
    ".proto",
    ".thrift",
    ".avsc",
    ".tf",
    ".tfvars",
    ".hcl",  # Terraform
    ".vim",
    ".el",
    ".clj",
    ".cljs",
    ".edn",
    ".ex",
    ".exs",
    ".erl",
    ".hrl",  # Elixir/Erlang
    ".hs",
    ".lhs",
    ".ml",
    ".mli",
    ".fsi",  # Functional
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
}


# Cache of allowed read roots per project root: (expiry_ts, frozenset_of_roots).
# Git worktrees are legitimate working dirs for the same repo but resolve OUTSIDE
# the main project root, so the workspace guard must treat every linked worktree
# as in-workspace. Discovering them shells out to ``git worktree list`` once and
# caches the result (TTL) so reading many files inside a worktree stays cheap.
_ALLOWED_READ_ROOTS_CACHE: "Dict[str, Tuple[float, frozenset]]" = {}
_ALLOWED_READ_ROOTS_TTL = 60.0  # seconds


EXTRA_READ_ROOTS_ENV = "VICTOR_EXTRA_READ_ROOTS"


def _extra_read_roots() -> "Tuple[str, ...]":
    """Operator-declared roots outside the project, for cross-repo work.

    ``VICTOR_EXTRA_READ_ROOTS`` holds ``os.pathsep``-separated paths (``~`` is
    expanded). This is the supported way to let one Victor session read a second
    repo — co-designing a wire contract across two codebases, or inspecting
    Victor's own ``~/.victor`` state — without disabling the guard wholesale.

    Opt-in only: with the variable unset, behaviour is exactly as before.
    """
    raw = os.environ.get(EXTRA_READ_ROOTS_ENV, "")
    roots = set()
    for entry in raw.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        try:
            roots.add(str(Path(entry).expanduser().resolve()))
        except OSError:
            continue
    return tuple(sorted(roots))


def _discover_read_roots(project_root: Path) -> "frozenset[str]":
    """Return the project root, every linked git-worktree root, and extra roots."""
    roots = {str(project_root), *_extra_read_roots()}
    try:
        import subprocess

        proc = subprocess.run(
            ["git", "-C", str(project_root), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("worktree "):
                    wt = line[len("worktree ") :].strip()
                    if wt:
                        roots.add(str(Path(wt).resolve()))
    except Exception:
        # git absent / not a repo / timeout → fall back to project root only.
        pass
    return frozenset(roots)


def _allowed_read_roots(project_root: Path, *, refresh: bool = False) -> "frozenset[str]":
    """Cached set of in-workspace roots (project root + worktrees + extra roots)."""
    # The extra roots are part of the key: changing VICTOR_EXTRA_READ_ROOTS
    # mid-session must take effect immediately, not after the TTL expires.
    key = "\x00".join((str(project_root), *_extra_read_roots()))
    now = time.monotonic()
    if not refresh:
        cached = _ALLOWED_READ_ROOTS_CACHE.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]
    roots = _discover_read_roots(project_root)
    _ALLOWED_READ_ROOTS_CACHE[key] = (now + _ALLOWED_READ_ROOTS_TTL, roots)
    return roots


def _out_of_workspace_message(path: str, file_path: Path, project_root: Path) -> str:
    """Explain the guard *and* how to get the content anyway.

    The previous message ended at "Use ls('.') to see files", which reads as a
    hard stop: sessions asked to inspect a second repo concluded the path was
    unreachable and gave up, even though ``ls`` and ``shell`` have no workspace
    restriction at all. A guard that cannot be routed around is
    indistinguishable from a missing capability, so state every route.
    """
    is_dir = file_path.is_dir()
    if is_dir:
        # ls() is not scoped, so it reaches the directory with no extra setup.
        one_off = f"ls(path='{file_path}')"
    elif file_path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        # A text pager on a binary store is the advice that stalls sessions.
        one_off = f"shell(cmd='sqlite3 {file_path} \".tables\"', action='read')"
    else:
        one_off = f"shell(cmd='sed -n \"1,200p\" {file_path}', action='read')"
    declared_root = file_path if is_dir else file_path.parent
    return (
        f"read() is scoped to the {project_root.name} workspace, and '{path}' is "
        f"outside it. Only read() is scoped — ls() and shell() are not, so this "
        f"is not a sandbox and the path is still reachable:\n"
        f"  - now: {one_off}\n"
        f"  - for the whole session: launch victor with "
        f"{EXTRA_READ_ROOTS_ENV}={declared_root}, which makes read() accept it "
        f"directly (keeping pagination and in-file search).\n"
        f"Use ls('.') for files inside {project_root.name}."
    )


def _path_within_workspace(file_path: Path, project_root: Path) -> bool:
    """True if ``file_path`` is under the project root or any linked git worktree.

    On a cache-miss for the path (e.g. a worktree created mid-session), the
    worktree list is refreshed once before deciding, so newly added worktrees
    are recognized immediately.
    """
    target = str(file_path)

    def _under(roots: "frozenset[str]") -> bool:
        return any(target == r or target.startswith(r + os.sep) for r in roots)

    if _under(_allowed_read_roots(project_root)):
        return True
    # Miss: a worktree may have appeared since the cache was populated.
    return _under(_allowed_read_roots(project_root, refresh=True))


@tool(
    category="filesystem",
    priority=Priority.CRITICAL,  # Always available for selection
    access_mode=AccessMode.READONLY,  # Only reads files
    danger_level=DangerLevel.SAFE,  # No side effects
    # Registry-driven metadata for tool selection and loop detection
    signature_params=[
        "path",
    ],  # Only path matters for loop detection. offset/limit excluded to enable pagination - reading different chunks of same file is exploration, not a loop
    stages=[
        "reading",
        "initial",
        "analysis",
        "verification",
    ],  # Conversation stages where relevant
    task_types=["analysis", "search"],  # Task types for classification-aware selection
    execution_category=ExecutionCategory.READ_ONLY,  # Safe for parallel execution
    keywords=[
        "read",
        "file",
        "open",
        "view",
        "content",
        "source",
        "code",
        "show",
        "display",
        "cat",
        "look",
        "examine",
        "inspect",
    ],
    use_cases=[
        "reading source code files",
        "viewing configuration files",
        "examining text documents",
        "pattern search within a single file (use code_search for multi-file)",
        "viewing specific line ranges with offset/limit",
    ],
    examples=[
        'read(path="src/main.py")',
        'read(path="config.yaml")',
        'read(path="README.md")',
        'read(path="main.py", offset=0, limit=50)',
        'read(path="utils.py", pattern="def calculate")',
    ],
    mandatory_keywords=[
        "read file",
        "show file",
        "explain this code",
        "what does this",
        # Additional keywords from MANDATORY_TOOL_KEYWORDS
        "explain",
        "describe",
        "what does",
    ],  # Force inclusion
    priority_hints=[
        "TRUNCATION: 10000 lines/64KB. Always ends on complete lines.",
        "PAGINATION: When truncated, output includes 'Use offset=N to continue' - use that exact offset value.",
        "LARGE FILES: prefer extract_skeleton (signatures only) or code_search within the file to find the relevant section, instead of reading the whole file in many chunks.",
        "Use for TEXT and CODE files only (.py, .js, .json, .yaml, .md, etc.)",
        "NOT for binary files (.pdf, .docx, .db, .pyc, images, archives)",
        "SINGLE-FILE SEARCH: Use search parameter for grep-like pattern search within one file",
        "MULTI-FILE SEARCH: Use code_search(mode='text', query='pattern', path='.') for searching across multiple files",
        "Use ls first to check file sizes before reading",
    ],
)
async def read(
    path: str,
    offset: int = 0,
    limit: int = 0,
    search: str = "",
    ctx: int = 2,
    regex: bool = False,
    # Parameter aliases for models that use different names (e.g., gpt-oss)
    line_start: int = None,
    line_end: int = None,
) -> str:
    """Read text/code file. Binary files rejected.

    Use this tool for READING FILE CONTENTS.
    For listing directory contents, use ls(path='directory/') instead.

    TRUNCATION LIMITS:
    - Maximum 10000 lines OR 64KB (whichever is reached first)
    - Always truncates at complete line boundaries (never mid-line)
    - When truncated, includes: "[... N more lines. Use offset=X to continue ...]"

    PAGINATION (for large files):
    - Use offset/limit: read(path, offset=0, limit=200), then offset=200, etc.
    - Or let auto-truncation guide you with the offset value in output

    SEARCH OPTIONS:
    - Single-file: Use search param for grep-like pattern search within one file
    - Multi-file: Use code_search(mode='text', query='pattern', path='.') for searching across files
    - Semantic: Use code_search(mode='semantic', query='concept') for conceptual search

    Args:
        path: File path (MUST be a file, NOT a directory)
        offset: Start line (0=beginning). Use for pagination of large files.
        limit: Max lines to read (0=auto, applies configured limits).
               Set explicit limit to override auto-truncation.
        search: Grep pattern for single-file search (use code_search for multi-file)
        ctx: Context lines around search matches
        regex: Pattern is regex (default: False for literal match)
        line_start: Alias for offset (some models use this name)
        line_end: Alias for limit (some models use this name)

    Returns:
        File content with line numbers. If truncated, includes continuation hint
        with exact offset to use for next read. If search param is used, returns
        matching lines with context.

    Examples:
        # Read entire file (with auto-truncation)
        read('src/main.py')

        # Read with pagination
        read('src/main.py', offset=0, limit=200)
        read('src/main.py', offset=200, limit=200)

        # Single-file pattern search
        read('src/main.py', search='TODO', ctx=2)

        # Multi-file search (use code_search instead)
        code_search(mode='text', query='TODO', path='src/', k=10)

    Note: If you need to list files in a directory, use ls(path='directory/') instead.
    """
    # Handle parameter aliases from models that use different names
    if line_start is not None and offset == 0:
        offset = line_start
    if line_end is not None and limit == 0:
        # Convert line_end to limit (line_end is absolute, limit is count)
        if line_start is not None:
            limit = max(0, line_end - line_start)
        else:
            limit = line_end
    file_path = Path(path).expanduser().resolve()

    # Workspace guard: warn if path is outside the current project root.
    # Returns helpful message so model self-corrects (doesn't raise).
    # Disabled via VICTOR_DISABLE_WORKSPACE_GUARD=1 for testing.
    if not os.environ.get("VICTOR_DISABLE_WORKSPACE_GUARD"):
        try:
            from victor.config.settings import get_project_paths

            _project_root = Path(get_project_paths().project_root).resolve()
            # Allow the project root AND any linked git worktree of the same repo:
            # worktrees resolve outside the main root but are legitimate (and the
            # documented) working dirs, so a naive prefix check would wrongly block
            # reads that edit/write already permit.
            if not _path_within_workspace(file_path, _project_root):
                return _out_of_workspace_message(path, file_path, _project_root)
        except Exception:
            pass

    # Early return for directory paths (handles "", ".", "src/", etc.)
    if file_path.is_dir():
        logger.info("Auto-converting read('%s') → ls('%s')", path, path)
        try:
            dir_result = await ls(path=str(file_path), depth=1)
            return f"Note: '{path}' is a directory, showing contents:\n\n{dir_result}"
        except Exception as e:
            return (
                f"'{path}' is a directory. Listing failed: {e}\n"
                f"Use ls(path='{path}') to explore."
            )

    if not file_path.exists():
        rust_mod_path = file_path.with_suffix("") / "mod.rs" if file_path.suffix == ".rs" else None
        if rust_mod_path and rust_mod_path.is_file():
            logger.info("Resolved Rust module path '%s' -> '%s'", path, rust_mod_path)
            file_path = rust_mod_path
            try:
                path = str(file_path.relative_to(Path.cwd()))
            except ValueError:
                path = str(file_path)

    if not file_path.exists():
        # Try PathResolver for intelligent path normalization and suggestions
        resolver = get_path_resolver()
        try:
            result = resolver.resolve_file(path)
            # PathResolver found a valid path after normalization
            file_path = result.resolved_path
            if result.was_normalized:
                logger.debug(
                    f"Path normalized by PathResolver: '{path}' -> '{result.resolved_path}' "
                    f"(applied: {result.normalization_applied})"
                )
                try:
                    path = str(result.resolved_path.relative_to(Path.cwd()))
                except ValueError:
                    path = str(result.resolved_path)
        except FileNotFoundError:
            # Try fuzzy resolve — search for bare filename in project tree
            basename = Path(path).name
            if basename and basename != path:
                # Already has directory prefix, don't fuzzy search
                pass
            else:
                try:
                    cwd = Path.cwd()
                    matches = [
                        m
                        for m in cwd.rglob(basename)
                        if m.is_file() and ".victor" not in str(m) and ".git" not in str(m)
                    ]
                    if len(matches) == 1:
                        file_path = matches[0]
                        logger.info(f"Fuzzy resolved {path} → {file_path}")
                        try:
                            path = str(file_path.relative_to(cwd))
                        except ValueError:
                            path = str(file_path)
                        # Skip to reading — file_path is now valid
                        if file_path.exists():
                            # Continue to the reading logic below
                            pass
                        else:
                            raise FileNotFoundError(f"File not found: {path}")
                    else:
                        raise FileNotFoundError("multiple or none")
                except (FileNotFoundError, OSError):
                    pass  # Fall through to suggestions

            # PathResolver couldn't find the file - provide helpful suggestions
            if not file_path.exists():
                suggestions = resolver.suggest_similar(path, limit=5)
                if suggestions:
                    suggestion_list = "\n  - ".join(suggestions[:5])
                    raise FileNotFoundError(
                        f"File not found: {path}\n"
                        f"Did you mean one of these?\n  - {suggestion_list}"
                    )
                else:
                    raise FileNotFoundError(f"File not found: {path}")
        except IsADirectoryError:
            # Auto-convert read(dir) → ls(dir)
            logger.info(f"Auto-converting read('{path}') → ls('{path}')")
            try:
                dir_result = await ls(path=path, depth=1)
                return f"Note: '{path}' is a directory, showing contents:\n\n" f"{dir_result}"
            except Exception as e:
                logger.warning("ls() fallback failed for directory '%s': %s", path, e)
                return (
                    f"'{path}' is a directory. Listing failed: {e}\n"
                    f"Use ls(path='{path}') to explore."
                )
    if not file_path.is_file():
        if file_path.is_dir():
            # Auto-convert read(dir) → ls(dir) instead of erroring
            logger.info(f"Auto-converting read('{path}') → ls('{path}')")
            try:
                dir_result = await ls(path=path, depth=1)
                return f"Note: '{path}' is a directory, showing contents:\n\n" f"{dir_result}"
            except Exception as e:
                logger.warning("ls() fallback failed for directory '%s': %s", path, e)
                return (
                    f"'{path}' is a directory. Listing failed: {e}\n"
                    f"Use ls(path='{path}') to explore."
                )
        raise IsADirectoryError(f"Path is not a file: {path}")

    # Binary file categories with helpful suggestions
    BINARY_CATEGORIES = {
        # Documents - suggest extraction approach
        "document": {
            "extensions": {
                ".pdf",
                ".doc",
                ".docx",
                ".xls",
                ".xlsx",
                ".ppt",
                ".pptx",
                ".odt",
                ".ods",
                ".odp",
            },
            "suggestion": "For document content, consider: 1) Convert to text externally (e.g., pdftotext), "
            "2) Use code_search to find related text files, 3) Check for a .txt or .md version.",
        },
        # Images - suggest description
        "image": {
            "extensions": {
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".bmp",
                ".ico",
                ".webp",
                ".svg",
                ".tiff",
                ".tif",
            },
            "suggestion": "Images cannot be read as text. If you need to understand image content, "
            "describe what you expect based on filename/context, or check for alt-text in nearby HTML/MD files.",
        },
        # Databases - suggest querying
        "database": {
            "extensions": {".db", ".sqlite", ".sqlite3"},
            "suggestion": "This is a database file. Use shell with 'sqlite3' to query it, "
            "e.g., `sqlite3 file.db '.tables'` or `sqlite3 file.db 'SELECT * FROM table LIMIT 5'`.",
        },
        # Python bytecode/cache
        "python_cache": {
            "extensions": {".pyc", ".pyo", ".pyd"},
            "suggestion": "This is compiled Python bytecode. Read the source .py file instead.",
        },
        # Pickled data
        "pickle": {
            "extensions": {".pkl", ".pickle"},
            "suggestion": "This is serialized Python data. To inspect, use: "
            "`python -c \"import pickle; print(pickle.load(open('file.pkl', 'rb')))\"`",
        },
        # Archives
        "archive": {
            "extensions": {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tgz"},
            "suggestion": "This is an archive. Use shell to list contents: "
            "`unzip -l file.zip`, `tar -tf file.tar.gz`, etc.",
        },
        # Compiled/binary
        "compiled": {
            "extensions": {
                ".so",
                ".o",
                ".a",
                ".dylib",
                ".dll",
                ".exe",
                ".bin",
                ".whl",
                ".egg",
                ".class",
                ".jar",
                ".war",
            },
            "suggestion": "This is a compiled binary. Check for source code files (.py, .java, .c, etc.) instead.",
        },
        # Media
        "media": {
            "extensions": {
                ".mp3",
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".wav",
                ".flac",
                ".ogg",
                ".webm",
            },
            "suggestion": "Media files cannot be read as text. Check for subtitles (.srt, .vtt) or metadata files.",
        },
        # Lock files
        "lock": {
            "extensions": {".lock"},
            "suggestion": "Lock files are often binary or auto-generated. Check the non-lock version if available.",
        },
        # Data files
        "data": {
            "extensions": {".dat", ".bin"},
            "suggestion": "This appears to be a binary data file. Look for documentation or schema files nearby.",
        },
    }

    # Known binary filenames (no extension)
    BINARY_FILENAMES = {".coverage"}

    def _get_binary_error(ext: str, filename: str) -> str:
        """Generate helpful error message based on file type."""
        ext_lower = ext.lower()
        for category, info in BINARY_CATEGORIES.items():
            if ext_lower in info["extensions"]:
                return (
                    f"Cannot read binary file: {filename}\n"
                    f"Type: {category} ({ext})\n"
                    f"Suggestion: {info['suggestion']}"
                )
        return f"Cannot read binary file: {filename} (extension: {ext}). This is not a text file."

    # Flatten all binary extensions
    ALL_BINARY_EXTENSIONS = set()
    for info in BINARY_CATEGORIES.values():
        ALL_BINARY_EXTENSIONS.update(info["extensions"])

    # =========================================================================
    # MAGIC BYTES DETECTION (Primary check - cannot be spoofed by extension)
    # =========================================================================
    # Detect actual file type via magic bytes first
    magic_type = detect_file_type_by_magic(file_path)

    if magic_type is not None:
        # Check for extension/magic mismatch (potential spoofing or misnamed file)
        mismatch_warning = check_extension_magic_mismatch(file_path, magic_type)
        if mismatch_warning:
            logger.warning(mismatch_warning)

        # If magic bytes indicate binary content, check if we have a handler
        if magic_type.category != FileCategory.TEXT:
            handler = get_binary_handler(magic_type.category)
            if handler is not None:
                # Future: Use registered handler to extract text
                try:
                    return handler(file_path)
                except Exception as e:
                    logger.error(f"Binary handler failed for {path}: {e}")
                    # Fall through to error

            # No handler available - provide helpful error with magic-based info
            raise ValueError(
                f"Cannot read binary file: {path}\n"
                f"Detected type: {magic_type.description}\n"
                f"MIME type: {magic_type.mime_type}\n"
                f"Category: {magic_type.category.name}\n"
                + (f"Suggestion: {magic_type.suggestion}" if magic_type.suggestion else "")
                + (f"\n\n{mismatch_warning}" if mismatch_warning else "")
            )

    # =========================================================================
    # EXTENSION-BASED CHECK (Fallback for files without recognized magic bytes)
    # =========================================================================
    # Check by extension
    if file_path.suffix.lower() in ALL_BINARY_EXTENSIONS:
        raise ValueError(_get_binary_error(file_path.suffix, path))

    # Check by filename (for dotfiles without extensions like .coverage)
    if file_path.name in BINARY_FILENAMES or file_path.name.startswith(".coverage."):
        raise ValueError(
            f"Cannot read binary file: {path}\n"
            f"Type: coverage database\n"
            f"Suggestion: This is a pytest-cov SQLite database. Use shell with sqlite3 to query, "
            f"or use 'coverage report' to see formatted coverage data."
        )

    # =========================================================================
    # TEXT FILE READING (with session-level caching)
    # =========================================================================
    # Check session cache first (prevents redundant reads for providers like DeepSeek)
    content: Optional[str] = None
    cache_hit = False

    if is_file_cache_enabled():
        cache = get_file_content_cache()
        cached_content = cache.get(str(file_path))
        if cached_content is not None:
            content = cached_content
            cache_hit = True
            logger.debug("File read from cache: %s", path)

    # Try to read the file if not in cache
    if content is None:
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
        except UnicodeDecodeError as e:
            # File contains binary content - provide helpful message
            file_size = file_path.stat().st_size
            raise ValueError(
                f"Cannot read file: {path}\n"
                f"Reason: Contains binary/non-UTF-8 content (error at byte {e.start})\n"
                f"Size: {file_size:,} bytes\n"
                f"Suggestion: This file appears to be binary despite its extension. "
                f"Check if it's the correct file, or look for a text-based alternative."
            )

        # Cache the content for future reads
        if is_file_cache_enabled() and not cache_hit:
            try:
                file_stat = file_path.stat()
                cache = get_file_content_cache()
                cache.set(
                    str(file_path),
                    content,
                    mtime=file_stat.st_mtime,
                    size=file_stat.st_size,
                )
            except OSError:
                pass  # Don't fail if we can't cache

    # Record that this file was read this session (read-before-overwrite guard).
    # Independent of the content cache (which is off by default) so the gate
    # always has read-state to consult.
    record_read(str(file_path))

    # Normalize parameters (handle non-int input from model)
    def _to_int(val, default: int) -> int:
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
        return default

    offset = _to_int(offset, 0)
    limit = _to_int(limit, 0)
    ctx = _to_int(ctx, 2)

    # TOKEN-EFFICIENT MODE: Search/grep
    if search:
        from victor.tools.output_utils import grep_lines

        result = grep_lines(
            content=content,
            pattern=search,
            context_before=ctx,
            context_after=ctx,
            case_sensitive=True,
            is_regex=regex,
            max_matches=50,
            file_path=path,
        )
        return result.to_string(show_line_numbers=True, max_matches=50)

    # Import truncation utilities
    from victor.tools.output_utils import truncate_by_lines, format_with_line_numbers

    # Determine truncation limits based on model context
    # Cloud models (Anthropic, OpenAI, etc.): ~2500 lines / 100KB (~25K tokens)
    # Local models (Ollama, LMStudio, vLLM): ~2000 lines / 32KB (~8K tokens)
    def _get_truncation_limits() -> tuple:
        """Get appropriate truncation limits based on current provider."""
        try:
            from victor.config.settings import get_settings

            settings = get_settings()

            # Check for airgapped mode or local providers
            if settings.security.airgapped_mode:
                return 10000, 65536  # ~10000 lines, 64KB for local models (airgapped)

            # Check provider name for local indicators
            provider_obj = getattr(settings, "provider", None)
            provider = (
                provider_obj.default_provider
                if hasattr(provider_obj, "default_provider")
                else str(provider_obj or "")
            ).lower()
            local_providers = {"ollama", "lmstudio", "vllm", "llamacpp", "local"}
            if any(p in provider for p in local_providers):
                # Try to get model context size from capabilities
                try:
                    from victor.providers.model_capabilities import (
                        get_model_capabilities,
                    )

                    model = getattr(settings, "model", "")
                    caps = get_model_capabilities(model)
                    context_window = caps.get("context_window", 0)
                    if context_window > 0:
                        # For local models with 64K+ context, use ~25% for reads (~16K tokens)
                        # This gives good file reading while leaving room for the rest of the conversation
                        max_tokens = context_window // 4  # Use 25% of context for file reads
                        # Estimate ~3 bytes per token (average), ~40 chars per line
                        max_lines = min(
                            10000, max(100, max_tokens // 3)
                        )  # Ensure at least 100 lines
                        max_bytes = min(65536, max_tokens * 3)  # ~3 bytes per token average
                        return max_lines, max_bytes
                except Exception as e:
                    logger.debug("Failed to compute context-aware truncation limits: %s", e)
                return 10000, 65536  # Fallback for local models: 10000 lines, 64KB

            # Cloud models get higher limits (large context windows)
            # Anthropic: 200K tokens, GPT-4: 128K tokens
            # 100KB ≈ 25K tokens at 4 bytes/token, reasonable for large context
            return 10000, 65536  # ~10000 lines, 64KB for cloud models
        except Exception as e:
            # Default to cloud limits if settings unavailable
            logger.debug("Settings unavailable for truncation limits, using defaults: %s", e)
            return 10000, 65536  # ~10000 lines, 64KB

    MAX_LINES, MAX_BYTES = _get_truncation_limits()

    # Check file size and adapt limit if needed for large files
    file_size = len(content.encode("utf-8"))
    if limit == 0 and file_size > 500000:  # 500KB threshold
        # For large files, use higher default limit
        MAX_LINES = 50000
        logger.info(
            f"Large file detected ({file_size} bytes), "
            f"increasing read limit to {MAX_LINES} lines"
        )

    lines = content.split("\n")
    total_lines = len(lines)

    # Clamp offset to valid range
    offset = max(0, min(offset, total_lines))

    # Apply offset first
    if offset > 0:
        remaining_content = "\n".join(lines[offset:])
    else:
        remaining_content = content

    # Determine effective limit
    if limit > 0:
        # User specified explicit limit - honor it (but still apply byte limit for safety)
        effective_max_lines = limit
    else:
        # Auto-truncation mode - apply default limits
        effective_max_lines = MAX_LINES

    # Apply truncation (always ends on complete lines)
    truncated_content, info = truncate_by_lines(
        remaining_content,
        max_lines=effective_max_lines,
        max_bytes=MAX_BYTES,
        start_line=0,  # Already applied offset above
    )

    # Format with line numbers (1-indexed, adjusted for offset)
    numbered_content = format_with_line_numbers(truncated_content, start_line=offset + 1)

    # Build informative header
    actual_end_line = offset + info.lines_returned
    header_parts = [
        f"[File: {path}]",
        f"[Lines {offset + 1}-{actual_end_line} of {total_lines}]",
        f"[Size: {info.bytes_returned:,} bytes]",
    ]

    # Add truncation warning with specific details
    if info.was_truncated:
        remaining = total_lines - actual_end_line
        if info.truncation_reason == "line_limit":
            # Distinguish between user-requested limit and system limit
            if limit > 0:
                # User explicitly requested this limit
                header_parts.append(
                    f"[READ {actual_end_line - offset}/{total_lines} lines requested via limit={limit}. "
                    f"{remaining} more lines available. Use offset={actual_end_line} to continue]"
                )
            else:
                # System-imposed limit
                header_parts.append(
                    f"[TRUNCATED: Hit {effective_max_lines} line system limit. "
                    f"{remaining} lines remaining. Use offset={actual_end_line} to continue]"
                )
        elif info.truncation_reason == "byte_limit":
            header_parts.append(
                f"[TRUNCATED: Hit {MAX_BYTES // 1024}KB byte limit at line {actual_end_line}. "
                f"{remaining} lines remaining. Use offset={actual_end_line} to continue]"
            )

    header = "\n".join(header_parts) + "\n\n"
    return header + numbered_content


# Extensions auto-eligible for LSP validate/format on write (see write()).
_LSP_SUPPORTED_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyi",
        ".pyx",  # Python
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cc",
        ".cxx",  # C/C++
        ".rs",  # Rust
        ".ts",
        ".tsx",
        ".js",
        ".jsx",  # JavaScript/TypeScript
        ".go",  # Go
        ".java",  # Java
        ".kt",
        ".kts",  # Kotlin
        ".swift",  # Swift
        ".scala",  # Scala
        ".cs",  # C#
        ".php",  # PHP
        ".rb",  # Ruby
        ".lua",  # Lua
        ".ex",
        ".exs",  # Elixir
        ".hs",  # Haskell
        ".r",
        ".R",  # R
        ".json",
        ".jsonc",  # JSON
        ".yaml",
        ".yml",  # YAML
        ".toml",  # TOML
        ".xml",  # XML
        ".html",
        ".htm",  # HTML
        ".css",
        ".scss",
        ".less",  # CSS
        ".sh",
        ".bash",
        ".zsh",  # Shell
        ".sql",  # SQL
        ".md",
        ".markdown",  # Markdown
    }
)


async def _delegate_write(path: str, content: str, force: bool) -> Dict[str, Any]:
    """Delegate a full-content write to ``edit()`` — the single write path + gate.

    Uses op type ``create`` for a new file and ``modify`` to overwrite an existing
    one (FileEditor's ``create`` refuses existing files). Consolidating write()
    onto edit() yields one write path and one read-before-overwrite gate; lazy
    import avoids a filesystem ↔ file_editor_tool circular import at module load.
    """
    from victor.tools.file_editor_tool import edit as _edit

    op_type = "modify" if Path(path).expanduser().exists() else "create"
    return await _edit(ops=[{"type": op_type, "path": path, "content": content, "force": force}])


def _editor_available() -> bool:
    """True when the enhanced FileEditor (the optional ``victor-coding`` extra) is registered.

    ``edit()`` requires victor-coding; ``write()`` must keep working without it,
    so the facade checks this and falls back to a direct write when False.
    """
    try:
        from victor.core.capability_registry import CapabilityRegistry
        from victor.framework.vertical_protocols import EditorProtocol

        return CapabilityRegistry.get_instance().is_enhanced(EditorProtocol)
    except Exception:
        return False


@tool(
    category="filesystem",
    priority=Priority.CRITICAL,  # Always available for selection
    access_mode=AccessMode.WRITE,  # Creates/overwrites files
    danger_level=DangerLevel.LOW,  # Minor risk, easily undoable
    # Registry-driven metadata for tool selection and cache invalidation
    signature_params=["path"],  # Same file = loop, regardless of content
    stages=["execution"],  # Conversation stages where relevant
    task_types=[
        "edit",
        "generation",
        "action",
    ],  # Task types for classification-aware selection
    execution_category=ExecutionCategory.WRITE,  # Cannot run in parallel with conflicting ops
    keywords=[
        "write",
        "file",
        "create",
        "save",
        "new",
        "output",
        "overwrite",
        "store",
        "put",
        "generate",
    ],
    use_cases=[
        "creating new files",
        "writing content to files",
        "saving generated code",
        "creating configuration files",
        "overwriting existing files",
    ],
    examples=[
        'write(path="utils.py", content="# Utility functions\\ndef helper():\\n    pass")',
        'write(path="main.py", content="def hello():\\n    print(\'hi\')")',
        'write(path="config.yaml", content="app:\\n  name: myapp")',
        'write(path="README.md", content="# Project\\n\\nDescription here")',
    ],
    priority_hints=[
        "Use for creating new files or completely replacing file content",
        "For surgical edits to existing files, use edit with 'replace' operation instead",
        "Supports undo via /undo command",
        "Automatically uses LSP validation/formatting when available for the language",
        "REQUIRED: 'path' and 'content' must be separate keyword arguments, not a JSON object",
    ],
)
async def write(
    path: str,
    content: str,
    *,
    validate: bool = False,
    format_code: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> Union[str, Dict[str, Any]]:
    """Write a file — thin facade over ``edit(create)``.

    Requires `path` AND `content`; overwriting an existing file requires
    reading it first in this session.

    Consolidating write() onto edit() gives ONE write path and ONE
    read-before-overwrite gate (see ``enforce_read_before_write``). Behavior is
    preserved from the previous standalone implementation:

    * **Simple mode** (default): auto LSP validate/format for code files, then
      create/overwrite. Returns a success message string.
    * **Enhanced mode** (``validate`` / ``format_code`` / ``dry_run``): returns
      a Dict with LSP diagnostics; ``dry_run`` writes nothing.

    Safety: overwriting an existing file that was NOT read this session is
    refused. Pass ``force=True`` to opt out (intentional full rewrite /
    generated output).

    Note:
        In EXPLORE/PLAN modes, writes are restricted to .victor/sandbox/.
        Use /mode build to enable unrestricted file writes.
        Falls back to a regular write if LSP is not available for the file type.
    """
    file_path = Path(path).expanduser().resolve()

    # Enforce sandbox restrictions in EXPLORE/PLAN modes (fail fast, before any
    # LSP work). edit() also enforces this per-op.
    enforce_sandbox_path(file_path)

    if file_path.exists() and file_path.is_dir():
        raise IsADirectoryError(f"Cannot write to directory: {path}")

    enhanced_mode = validate or format_code or dry_run

    # ------------------------------------------------------------------
    # LSP enhance (validate/format). Runs write_with_lsp with write=False so it
    # only RETURNS formatted content + diagnostics; the actual write is done by
    # edit(create) below — single write path.
    # ------------------------------------------------------------------
    final_content = content
    lsp_suffix = ""
    run_lsp = file_path.suffix.lower() in _LSP_SUPPORTED_EXTENSIONS or enhanced_mode

    if run_lsp:
        try:
            from victor.tools.lsp_write_enhancer import DiagnosticSeverity, write_with_lsp

            result = await write_with_lsp(
                path=str(file_path),
                content=content,
                validate=True,
                format_code=format_code or not enhanced_mode,
                write=False,  # never write here — delegate to edit()
            )

            if result.written_content is not None:
                final_content = result.written_content

            # Enhanced mode: return diagnostics dict; write only when not dry_run
            # and LSP found no errors (matches prior "has errors → not written").
            if enhanced_mode:
                if not dry_run and result.success:
                    edit_res = await _delegate_write(path, final_content, force)
                    if isinstance(edit_res, dict) and edit_res.get("success") is False:
                        return edit_res
                return result.to_dict()

            # Simple mode: build the "(formatted ..., validation passed)" suffix.
            lsp_info: List[str] = []
            if result.formatted:
                lsp_info.append(f"formatted with {result.formatter_used}")
            if result.validated:
                # Diagnostic.severity is a DiagnosticSeverity Enum (not a str), so compare
                # against the enum members — a string compare was always False and silently
                # reported "validation passed" even for error-level diagnostics.
                error_count = sum(
                    1 for d in result.diagnostics if d.severity == DiagnosticSeverity.ERROR
                )
                warning_count = sum(
                    1 for d in result.diagnostics if d.severity == DiagnosticSeverity.WARNING
                )
                if error_count or warning_count:
                    lsp_info.append(f"{error_count} errors, {warning_count} warnings")
                else:
                    lsp_info.append("validation passed")
            lsp_suffix = f" ({', '.join(lsp_info)})" if lsp_info else ""

        except Exception as lsp_error:
            logger.debug("LSP enhancement failed, falling back to plain write: %s", lsp_error)
            if enhanced_mode:
                return {
                    "success": False,
                    "path": path,
                    "error": f"LSP enhancement failed: {lsp_error}",
                    "validated": False,
                    "formatted": False,
                }
            lsp_suffix = " (LSP unavailable for this file type)"

    # ------------------------------------------------------------------
    # Write the file. Prefer the consolidated edit(create) path (single write
    # path + single gate); fall back to a direct write when the enhanced editor
    # (the optional victor-coding extra) is unavailable so write() keeps working
    # standalone. The read-before-overwrite gate applies to BOTH paths.
    # ------------------------------------------------------------------
    existed_before = file_path.exists()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if _editor_available():
        edit_result = await _delegate_write(path, final_content, force)  # edit gates internally
        if isinstance(edit_result, dict) and edit_result.get("success") is False:
            # Surface structured failures (e.g. read-before-overwrite gate) verbatim.
            return edit_result
    else:
        # Direct write fallback (victor-coding unavailable) — apply the gate here.
        try:
            enforce_read_before_write(file_path, force=force)
        except PermissionError as pe:
            return {"success": False, "path": path, "error": str(pe)}

        from victor.agent.change_tracker import ChangeType, get_change_tracker

        tracker = get_change_tracker()
        change_type = ChangeType.MODIFY if existed_before else ChangeType.CREATE
        tracker.begin_change_group("write", f"Write to {path}")
        tracker.record_change(
            file_path=str(file_path),
            change_type=change_type,
            original_content=original_or_none(file_path),
            new_content=final_content,
            tool_name="write",
            tool_args={"path": path},
        )
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(final_content)
        tracker.commit_change_group()

        if is_file_cache_enabled():
            get_file_content_cache().invalidate(str(file_path))

    action = "modified" if existed_before else "created"
    return (
        f"Successfully {action} {path} ({len(final_content)} characters)"
        f"{lsp_suffix}. Use /undo to revert."
    )


def original_or_none(file_path: Path) -> Optional[str]:
    """Read existing file content for undo tracking, or None if absent/unreadable."""
    try:
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return None


@tool(
    category="filesystem",
    priority=Priority.CRITICAL,  # Always available for selection
    access_mode=AccessMode.READONLY,  # Only reads directory contents
    danger_level=DangerLevel.SAFE,  # No side effects
    # Registry-driven metadata for tool selection and loop detection
    signature_params=[
        "path",
        "depth",
        "pattern",
    ],  # Params indicating exploration progress
    stages=[
        "initial",
        "planning",
        "reading",
        "analysis",
    ],  # Conversation stages where relevant
    task_types=["search", "analysis"],  # Task types for classification-aware selection
    execution_category=ExecutionCategory.READ_ONLY,  # Safe for parallel execution
    keywords=[
        "list",
        "directory",
        "ls",
        "dir",
        "files",
        "folders",
        "find",
        "browse",
        "explore",
        "contents",
        "tree",
    ],
    use_cases=[
        "listing directory contents",
        "finding files by pattern",
        "exploring project structure",
        "listing specific file types",
    ],
    examples=[
        'ls(path="src/", depth=2)',
        'ls(path=".", pattern="*.py")',
        'ls(path="docs/", depth=1)',
        'ls(path="tests/", pattern="test_*")',
        'ls(path=".", include_dirs=True, include_files=False)',
    ],
    mandatory_keywords=[
        "list files",
        "show files",
        "how many files",
        "count files",
    ],  # Force inclusion
    priority_hints=[
        "Use for browsing directory contents (default depth=2 shows subdirectories)",
        "Use pattern parameter for filtering (e.g., '*.py', 'test_*')",
        "For searching specific files, use find(name='filename') instead",
    ],
)
async def ls(
    path: str,
    recursive: bool = False,
    depth: int = 2,
    pattern: str = "",
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """List directory contents with file sizes.

    ⚠️ CRITICAL: This tool ONLY works on DIRECTORIES, not files.
    If you need to read file contents, use read(path='file_path') instead.

    Before calling ls():
    1. Check if the path is a file or directory
    2. If it's a file (e.g., victor/ui/cli.py), use read() instead
    3. Only call ls() on directory paths (e.g., victor/ui/)

    WRONG: ls(path='victor/ui/cli.py')  # This is a FILE
    RIGHT: ls(path='victor/ui/')        # This is a DIRECTORY
    RIGHT: read(path='victor/ui/cli.py') # Use read() for files

    Args:
        path: Directory path (MUST be a directory, NOT a file)
        recursive: All levels (ignores depth)
        depth: Levels to explore (default=2 for subdirectory visibility)
        pattern: Glob filter (*.py, test_*)
        limit: Max entries

    Returns:
        List of {name/path, type, depth, size, hint}.
        - size: File size in bytes (files only)
        - hint: For large files (>100KB), pagination instruction

    Tip: Default depth=2 shows contents of immediate subdirectories.
         Use depth=1 for just the top level, or recursive=True for all.
    """
    import fnmatch

    try:
        dir_path = Path(path).expanduser().resolve()

        if not dir_path.exists():
            # Try PathResolver for intelligent path normalization and suggestions
            resolver = get_path_resolver()
            try:
                result = resolver.resolve_directory(path)
                # PathResolver found a valid directory after normalization
                dir_path = result.resolved_path
                if result.was_normalized:
                    logger.debug(
                        f"Directory path normalized by PathResolver: '{path}' -> '{result.resolved_path}' "
                        f"(applied: {result.normalization_applied})"
                    )
                    try:
                        path = str(result.resolved_path.relative_to(Path.cwd()))
                    except ValueError:
                        path = str(result.resolved_path)
            except FileNotFoundError:
                # PathResolver couldn't find the directory - provide helpful suggestions
                suggestions = resolver.suggest_similar(path, limit=5)
                if suggestions:
                    suggestion_list = "\n  - ".join(suggestions[:5])
                    raise FileNotFoundError(
                        f"❌ Directory not found: {path}\n"
                        f"💡 Did you mean one of these?\n"
                        f"  - {suggestion_list}\n"
                        f"🔍 Use find(name='filename') to search for files"
                    )
                else:
                    raise FileNotFoundError(
                        f"❌ Directory not found: {path}\n"
                        f"💡 Use find(name='filename') to search for files\n"
                        f"💡 Use overview(path='.') to see project structure"
                    )
            except NotADirectoryError:
                raise NotADirectoryError(
                    f"❌ ERROR: The path '{path}' is a FILE, NOT a DIRECTORY.\n"
                    f"🔧 CORRECT ACTION: Use read(path='{path}') instead of ls(path='{path}')\n"
                    f"📖 EXAMPLE: To read this file, call: read(path='{path}')\n"
                    f"💡 TIP: The ls() tool ONLY works on directories (e.g., ls(path='victor/ui/'))"
                )
        if not dir_path.is_dir():
            if dir_path.is_file():
                # ls() on a file returns comprehensive filesystem metadata with content preview
                logger.info(f"ls() called on file '{path}', returning file metadata with preview")
                try:
                    stat_info = dir_path.stat()
                    import pwd
                    import grp

                    # Get ownership (fallback to numeric IDs if names unavailable)
                    try:
                        owner = pwd.getpwuid(stat_info.st_uid).pw_name
                    except KeyError:
                        owner = str(stat_info.st_uid)
                    try:
                        group = grp.getgrgid(stat_info.st_gid).gr_name
                    except KeyError:
                        group = str(stat_info.st_gid)

                    # Get permissions in octal and symbolic form
                    perms_octal = oct(stat_info.st_mode)[-3:]
                    import stat as stat_module

                    perms_symbolic = stat_module.filemode(stat_info.st_mode)

                    # Get timestamps
                    import datetime

                    created = datetime.datetime.fromtimestamp(stat_info.st_ctime)
                    modified = datetime.datetime.fromtimestamp(stat_info.st_mtime)
                    accessed = datetime.datetime.fromtimestamp(stat_info.st_atime)

                    # Try to get a content preview (first 500 chars or 10 lines, whichever is less)
                    content_preview = None
                    try:
                        if stat_info.st_size < 100_000:  # Only preview files smaller than 100KB
                            with open(dir_path, "r", encoding="utf-8", errors="ignore") as f:
                                lines = []
                                for i, line in enumerate(f):
                                    if i >= 10:
                                        break
                                    lines.append(line.rstrip("\n"))
                                preview_text = "\n".join(lines)
                                if len(preview_text) > 500:
                                    preview_text = preview_text[:500] + "..."
                                content_preview = preview_text
                    except Exception as preview_error:
                        logger.debug(f"Failed to generate content preview: {preview_error}")

                    # Build file metadata item
                    item = {
                        "name": dir_path.name,
                        "path": str(dir_path),
                        "type": "file",
                        "depth": 0,
                        "size": stat_info.st_size,
                        "extension": dir_path.suffix.lower() if dir_path.suffix else "",
                        # Permissions and ownership
                        "permissions": perms_octal,
                        "permissions_symbolic": perms_symbolic,
                        "owner": owner,
                        "group": group,
                        # Timestamps
                        "created": created.isoformat(),
                        "modified": modified.isoformat(),
                        "accessed": accessed.isoformat(),
                        # Additional metadata
                        "inode": stat_info.st_ino,
                        "nlinks": stat_info.st_nlink,
                        "auto_converted": True,
                        "hint": f"Note: '{path}' is a file. Use read() to see full contents.",
                    }

                    # Add content preview if available
                    if content_preview:
                        item["content_preview"] = content_preview

                    # Return in same format as directory listing
                    cwd = str(Path.cwd())
                    try:
                        relative_target = str(dir_path.relative_to(Path.cwd()))
                    except ValueError:
                        relative_target = str(dir_path)

                    return {
                        "cwd": cwd,
                        "target": relative_target if relative_target != "." else ".",
                        "items": [item],
                        "count": 1,
                        "auto_converted_from_file": True,
                    }
                except Exception as e:
                    # Fallback to error if metadata collection fails
                    logger.warning(f"ls() file metadata collection failed: {e}")
                    raise NotADirectoryError(
                        f"Path is not a directory: {path}\n"
                        f"Suggestion: Use read(path='{path}') to read this file instead."
                    )
            raise NotADirectoryError(f"Path is not a directory: {path}")

        # Normalize limit (handle non-int input from model)
        if not isinstance(limit, int):
            limit = int(limit) if isinstance(limit, str) and limit.isdigit() else 1000

        items = []
        count = 0

        # Normalize depth (handle non-int input from model)
        if not isinstance(depth, int):
            depth = int(depth) if isinstance(depth, str) and depth.isdigit() else 1

        # Determine max depth for traversal
        max_depth = float("inf") if recursive else depth

        def walk_breadth_first(base: Path):
            """Walk directory breadth-first (level by level)."""
            from collections import deque

            queue = deque([(base, 0)])  # (path, current_depth)

            while queue:
                current, current_depth = queue.popleft()
                if current_depth > 0:  # Don't yield the root
                    yield current, current_depth

                if current.is_dir() and current_depth < max_depth:
                    try:
                        children = sorted(current.iterdir())
                        for child in children:
                            queue.append((child, current_depth + 1))
                    except PermissionError:
                        pass

        def walk_depth_first(base: Path, current_depth: int):
            """Walk directory depth-first (recursive)."""
            if current_depth > max_depth:
                return
            try:
                for p in sorted(base.iterdir()):
                    yield p, current_depth
                    if p.is_dir():
                        yield from walk_depth_first(p, current_depth + 1)
            except PermissionError:
                pass

        if depth == 1 and not recursive:
            # Only immediate children (optimized path)
            iterator = ((p, 1) for p in sorted(dir_path.iterdir()))
        else:
            # Breadth-first: see all children at each level first (better coverage when truncated)
            iterator = walk_breadth_first(dir_path)

        # Use relative paths when exploring beyond depth 1
        use_relative_paths = recursive or depth > 1

        for p, entry_depth in iterator:
            if count >= limit:
                break

            is_dir = p.is_dir()
            name = str(p.relative_to(dir_path)) if use_relative_paths else p.name

            # Pattern filter (glob)
            if pattern and not fnmatch.fnmatch(name, pattern):
                continue

            # Compute full path relative to cwd for use in subsequent tool calls
            try:
                full_path = str(p.relative_to(Path.cwd()))
            except ValueError:
                # If path is not relative to cwd, use absolute
                full_path = str(p)

            item = {
                "path" if use_relative_paths else "name": name,
                "full_path": full_path,  # Full path for model to use in subsequent calls
                "type": "directory" if is_dir else "file",
                "depth": entry_depth,
            }

            # Add file size for files (helps LLM plan read operations)
            if not is_dir:
                try:
                    size = p.stat().st_size
                    item["size"] = size
                    # Direct hint for large files - tell LLM exactly what to do
                    if size > 100_000:
                        item["hint"] = "USE read(offset=0,limit=500) to paginate"
                except OSError:
                    pass  # Skip size on permission errors

            items.append(item)
            count += 1

        # Build result with cwd context for better LLM orientation
        cwd = str(Path.cwd())

        # Try to express the target path relative to cwd for clarity
        try:
            relative_target = str(dir_path.relative_to(Path.cwd()))
        except ValueError:
            relative_target = str(dir_path)  # Use absolute if outside cwd

        # Always include cwd context and relative target path in response
        result = {
            "cwd": cwd,
            "target": relative_target if relative_target != "." else ".",
            "items": items,
            "count": len(items),
        }

        # Add optional metadata
        if pattern:
            result["filter"] = pattern
        if count >= limit:
            result["truncated"] = True

        return result

    except Exception as e:
        # Let the decorator handle the exception and format it
        raise e


@tool(
    category="filesystem",
    priority=Priority.HIGH,  # Very useful for file discovery
    access_mode=AccessMode.READONLY,  # Only searches files
    danger_level=DangerLevel.SAFE,  # No side effects
    keywords=[
        "find",
        "search",
        "locate",
        "where",
        "which",
        "discover",
        "lookup",
    ],
    use_cases=[
        "finding files by name pattern",
        "locating a specific file",
        "discovering where a file exists",
        "searching for files recursively",
    ],
    examples=[
        "find tool_executor.py",
        "where is the config file",
        "locate all test files",
        "find files named *_tool.py",
    ],
)
async def find(
    name: str,
    path: str = ".",
    type: str = "all",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Find files by name pattern (like Unix find -name).

    Searches recursively through the directory tree to locate files
    matching the given name pattern. Supports glob patterns.

    Args:
        name: File name pattern to find (supports glob: *.py, *test*, tool_*.py)
        path: Root directory to search from (default: current directory)
        type: Filter by type: 'file', 'dir', or 'all' (default: all)
        limit: Maximum results to return (default: 50)

    Returns:
        List of matching files with path, type, and size.

    Examples:
        find("tool_executor.py")  # Find exact filename anywhere
        find("*_tool.py")         # Find files ending in _tool.py
        find("test*", type="dir") # Find directories starting with 'test'
    """
    import fnmatch

    try:
        base_path = Path(path).expanduser().resolve()

        if not base_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        results = []
        count = 0

        # Walk the directory tree
        for root, dirs, files in base_path.walk():
            # Skip hidden and common excluded directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in {"node_modules", "__pycache__", "venv", ".venv", "build", "dist"}
            ]

            # Check directories if type allows
            if type in ("all", "dir"):
                for d in dirs:
                    if fnmatch.fnmatch(d, name) or fnmatch.fnmatch(d.lower(), name.lower()):
                        dir_path = root / d
                        results.append(
                            {
                                "path": str(dir_path.relative_to(base_path)),
                                "type": "directory",
                                "size": 0,
                            }
                        )
                        count += 1
                        if count >= limit:
                            break

            # Check files if type allows
            if type in ("all", "file") and count < limit:
                for f in files:
                    if fnmatch.fnmatch(f, name) or fnmatch.fnmatch(f.lower(), name.lower()):
                        file_path = root / f
                        try:
                            size = file_path.stat().st_size
                        except OSError:
                            size = 0
                        results.append(
                            {
                                "path": str(file_path.relative_to(base_path)),
                                "type": "file",
                                "size": size,
                            }
                        )
                        count += 1
                        if count >= limit:
                            break

            if count >= limit:
                break

        if not results:
            return [
                {
                    "message": f"No files matching '{name}' found in {path}",
                    "suggestion": "Try a broader pattern like '*{name}*' or search from project root with path='.'",
                }
            ]

        return results

    except Exception as e:
        raise e


# Important documentation file patterns (case-insensitive)
IMPORTANT_DOC_PATTERNS = [
    "readme*",
    "architecture*",
    "index*",
    "claude*",
    "contributing*",
    "changelog*",
    "license*",
    "design*",
    "overview*",
    "getting*started*",
    "setup*",
    "install*",
]


async def _get_directory_summaries(root: Path, directory_paths: List[str]) -> Dict[str, str]:
    """Get summaries for directories based on symbol index.

    Uses the project's SQLite symbol graph to identify key classes and functions
    within each directory, providing a high-level view of its purpose.

    Args:
        root: Codebase root path
        directory_paths: List of relative directory paths to summarize

    Returns:
        Dict mapping directory path to summary string
    """
    from victor.core.database import get_project_database

    try:
        # Use absolute path to resolve the correct project database
        # This is safe to call even if the database is already open
        project_db = get_project_database(root)
        if not project_db.table_exists("graph_node"):
            logger.debug("graph_node table does not exist yet (init still in progress)")
            return {}

        # Limit number of summaries to avoid excessive DB queries on large repos
        max_summaries = 40
        target_paths = directory_paths[:max_summaries]

        summaries: Dict[str, str] = {}

        for rel_path in target_paths:
            # Normalize path for SQLite LIKE (use / even on Windows)
            # We look for recursive matches (file LIKE 'rel_path/%')
            db_path_prefix = rel_path.replace("\\", "/") + "/%"

            # Query for top symbols (prioritizing classes then functions)
            # This is a lightweight query that avoids full graph traversal
            query = """
                SELECT type, name
                FROM graph_node
                WHERE (file LIKE ? OR file = ?)
                AND type IN ('class', 'function', 'interface', 'module')
                ORDER BY CASE type
                    WHEN 'class' THEN 1
                    WHEN 'interface' THEN 2
                    WHEN 'function' THEN 3
                    ELSE 4
                END
                LIMIT 15
            """
            rows = project_db.query(query, (db_path_prefix, rel_path))

            if not rows:
                continue

            # Group by type and build summary
            by_type: Dict[str, List[str]] = {}
            for row in rows:
                t = row["type"]
                if t not in by_type:
                    by_type[t] = []
                by_type[t].append(row["name"])

            parts = []
            if "class" in by_type:
                classes = sorted(list(set(by_type["class"])))[:4]
                parts.append(f"Classes: {', '.join(classes)}")
            if "function" in by_type:
                funcs = sorted(list(set(by_type["function"])))[:4]
                parts.append(f"Funcs: {', '.join(funcs)}")

            if parts:
                summaries[rel_path] = " | ".join(parts)

        return summaries
    except Exception as e:
        # ENHANCEMENT: Better error logging for debugging
        error_msg = str(e) if str(e) else f"{type(e).__name__}"

        # Log with context for debugging
        logger.warning(
            "Failed to get directory summaries from symbol graph: %s "
            "(this is OK during init - will retry on next tool call)",
            error_msg,
        )

        # Return partial results if available
        return summaries if "summaries" in locals() else {}


@tool(
    category="filesystem",
    name="project_overview",
    priority=Priority.HIGH,
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    # Registry-driven metadata for tool selection
    signature_params=[
        "path",
        "max_depth",
    ],  # Params indicating exploration progress
    stages=[
        "initial",
        "exploration",
        "analysis",
    ],  # Conversation stages where relevant
    task_types=["analysis", "exploration"],  # Task types for classification
    execution_category=ExecutionCategory.READ_ONLY,  # Safe for parallel execution
    keywords=[
        "overview",
        "summary",
        "structure",
        "directories",
        "project",
        "architecture",
        "layout",
        "components",
        "organization",
        "tree",
    ],
    use_cases=[
        "getting initial project overview",
        "understanding directory structure",
        "identifying important files",
        "project exploration",
        "finding documentation files",
        "discovering main components",
    ],
    examples=[
        "give me a project overview",
        "show me the directory structure",
        "what are the main components",
        "summarize the project layout",
        "what does this codebase contain",
        "how is this project organized",
    ],
    mandatory_keywords=[
        "project overview",
        "directory structure",
        "main components",
        "codebase layout",
    ],  # Force inclusion for these intents
)
async def overview(
    path: str = ".",
    max_depth: int = 2,
    top_files_by_size: int = 100,
    top_doc_files: int = 15,
) -> Dict[str, Any]:
    """Get a curated project overview for initial exploration.

    Provides:
    1. Directory structure at max_depth (default: 2 levels)
    2. Top documentation files (README*, ARCHITECTURE*, INDEX*, etc.)
    3. Largest source files by size (helps identify core modules)

    Args:
        path: Root directory to explore (default: current directory)
        max_depth: Directory exploration depth (default: 2)
        top_files_by_size: Number of largest files to include (default: 100)
        top_doc_files: Number of documentation files to include (default: 15)

    Returns:
        Dict with directories, important_docs, and largest_files sections
    """
    import fnmatch

    try:
        root = Path(path).expanduser().resolve()
        # Validate directory exists with helpful suggestions
        _validate_directory_with_suggestions(root)

        if not root.is_dir():
            # GAP-14 FIX: If a file path is given, use its parent directory
            # This is a common model mistake - be helpful and auto-correct
            parent = root.parent
            if parent.is_dir():
                root = parent
            else:
                raise NotADirectoryError(
                    f"Path is not a directory: {path}. Use the parent directory or a directory path."
                )

        # Excluded directories
        exclude_dirs = {
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
            ".eggs",
            "htmlcov",
            ".cache",
            ".ruff_cache",
        }

        # Documentation extensions
        doc_extensions = {".md", ".rst", ".txt", ".adoc"}

        # Collect all files with metadata
        all_files: List[Dict[str, Any]] = []
        directories: List[Dict[str, Any]] = []
        important_docs: List[Dict[str, Any]] = []

        def walk_tree(base: Path, current_depth: int):
            """Walk tree collecting files and directories."""
            if current_depth > max_depth:
                return
            try:
                for p in sorted(base.iterdir()):
                    rel_path = str(p.relative_to(root))

                    # Skip excluded directories
                    if p.is_dir() and p.name in exclude_dirs:
                        continue
                    # Skip hidden files/dirs
                    if p.name.startswith("."):
                        continue

                    if p.is_dir():
                        directories.append(
                            {
                                "path": rel_path,
                                "depth": current_depth,
                            }
                        )
                        walk_tree(p, current_depth + 1)
                    else:
                        try:
                            size = p.stat().st_size
                        except OSError:
                            size = 0

                        file_info = {
                            "path": rel_path,
                            "name": p.name,
                            "size": size,
                            "extension": p.suffix.lower(),
                            "depth": current_depth,
                        }
                        all_files.append(file_info)

                        # Check if it's an important doc file
                        name_lower = p.name.lower()
                        is_doc = p.suffix.lower() in doc_extensions

                        # Check against important doc patterns (case-insensitive)
                        for pattern in IMPORTANT_DOC_PATTERNS:
                            if fnmatch.fnmatch(name_lower, pattern):
                                file_info["importance"] = "high"
                                important_docs.append(file_info)
                                break
                        else:
                            # Also include .md files at root level as potentially important
                            if is_doc and current_depth == 1:
                                file_info["importance"] = "medium"
                                important_docs.append(file_info)

            except PermissionError:
                pass

        walk_tree(root, 1)

        # Get summaries for directories from symbol index
        # ENHANCEMENT: Add retry logic for init-time database locks
        max_retries = 2
        dir_summaries = {}

        for attempt in range(max_retries):
            dir_summaries = await _get_directory_summaries(root, [d["path"] for d in directories])

            if dir_summaries:
                break  # Success

            if attempt < max_retries - 1:
                logger.debug("Retry %d/%d for directory summaries", attempt + 1, max_retries)
                await asyncio.sleep(0.5)  # Brief wait before retry

        # Attach summaries to directories
        for d in directories:
            if d["path"] in dir_summaries:
                d["summary"] = dir_summaries[d["path"]]

        # Sort important docs by importance then size
        importance_order = {"high": 0, "medium": 1}
        important_docs.sort(
            key=lambda x: (
                importance_order.get(x.get("importance", "medium"), 1),
                -x.get("size", 0),
            )
        )
        important_docs = important_docs[:top_doc_files]

        # Get largest files (excluding docs already included)
        doc_paths = {d["path"] for d in important_docs}
        source_files = [f for f in all_files if f["path"] not in doc_paths]
        source_files.sort(key=lambda x: -x.get("size", 0))
        largest_files = source_files[:top_files_by_size]

        # Format sizes for readability
        def format_size(size: int) -> str:
            if size < 1024:
                return f"{size}B"
            elif size < 1024 * 1024:
                return f"{size // 1024}KB"
            else:
                return f"{size // (1024 * 1024)}MB"

        for f in important_docs + largest_files:
            f["size_formatted"] = format_size(f.get("size", 0))

        return {
            "success": True,
            "root": str(root),
            "directories": directories,
            "directory_count": len(directories),
            "important_docs": important_docs,
            "largest_files": largest_files,
            "total_files_scanned": len(all_files),
            "hints": [
                "Summary-First: Use directory summaries to identify relevant subsystems before deep diving.",
                "Use read to examine important documentation first",
                "Large files often indicate core modules",
                "Check README.md and CLAUDE.md for project-specific guidance",
            ],
        }

    except Exception as e:
        raise e
