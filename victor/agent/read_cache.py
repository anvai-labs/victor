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

"""Read result deduplication cache for efficient file access.

This module provides caching for file read operations to prevent redundant
reads of the same file content, improving agent efficiency.

Issue Reference: workflow-test-issues-v2.md Issue #2
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class CachedRead:
    """Represents a cached file read result."""

    path: str
    content: str
    content_hash: str
    timestamp: float
    size_bytes: int
    read_count: int = 1
    last_access: float = field(default_factory=time)

    def touch(self) -> None:
        """Update last access time and increment read count."""
        self.last_access = time()
        self.read_count += 1


@dataclass
class ReadCacheStats:
    """Statistics for read cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    redundant_warnings: int = 0
    total_bytes_saved: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": f"{self.hit_rate:.1%}",
            "redundant_warnings": self.redundant_warnings,
            "bytes_saved": self.total_bytes_saved,
        }


@runtime_checkable
class IReadCache(Protocol):
    """Protocol for read result caching."""

    def get(self, path: str) -> Optional[str]:
        """Get cached content if available."""
        ...

    def put(self, path: str, content: str) -> None:
        """Cache file content."""
        ...

    def invalidate(self, path: str) -> None:
        """Invalidate cache for a modified file."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        ...


class ReadResultCache:
    """Caches file read results to prevent redundant reads.

    Features:
    - TTL-based expiration
    - LRU eviction when at capacity
    - Content hash for change detection
    - Redundant read detection and warnings
    - Thread-safe operations

    Usage:
        cache = ReadResultCache(ttl_seconds=300)

        # Check cache before reading
        content = cache.get("/path/to/file.py")
        if content is None:
            content = read_file("/path/to/file.py")
            cache.put("/path/to/file.py", content)

        # Invalidate on write
        cache.invalidate("/path/to/file.py")
    """

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_entries: int = 100,
        max_total_bytes: int = 10 * 1024 * 1024,  # 10MB
        redundant_threshold: int = 2,
        redundant_window: float = 60.0,
    ):
        """Initialize the read cache.

        Args:
            ttl_seconds: Time-to-live for cached entries
            max_entries: Maximum number of cached files
            max_total_bytes: Maximum total cache size in bytes
            redundant_threshold: Read count threshold for redundant warning
            redundant_window: Time window for redundant detection
        """
        self._cache: Dict[str, CachedRead] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._max_total_bytes = max_total_bytes
        self._redundant_threshold = redundant_threshold
        self._redundant_window = redundant_window
        self._stats = ReadCacheStats()
        self._current_bytes = 0

    def get(self, path: str, offset: int = 0, limit: Optional[int] = None) -> Optional[str]:
        """Get cached content if available and not expired.

        Args:
            path: File path to look up
            offset: Line offset (for partial reads)
            limit: Line limit (for partial reads)

        Returns:
            Cached content or None if not available/expired
        """
        # Normalize path
        normalized_path = self._normalize_path(path)

        with self._lock:
            if normalized_path not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[normalized_path]

            # Check expiration
            if time() - entry.timestamp > self._ttl:
                self._evict(normalized_path)
                self._stats.misses += 1
                return None

            # Update access info
            entry.touch()
            self._stats.hits += 1
            self._stats.total_bytes_saved += entry.size_bytes

            logger.debug(
                f"Cache hit for {path} (read #{entry.read_count}, "
                f"hit rate: {self._stats.hit_rate:.1%})"
            )

            # Handle partial reads
            content = entry.content
            if offset > 0 or limit is not None:
                lines = content.splitlines(keepends=True)
                if offset > 0:
                    lines = lines[offset:]
                if limit is not None:
                    lines = lines[:limit]
                content = "".join(lines)

            return content

    def put(self, path: str, content: str) -> None:
        """Cache file content.

        Args:
            path: File path
            content: File content to cache
        """
        normalized_path = self._normalize_path(path)
        content_bytes = len(content.encode("utf-8"))
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]

        with self._lock:
            # Check if already cached with same content
            if normalized_path in self._cache:
                existing = self._cache[normalized_path]
                if existing.content_hash == content_hash:
                    existing.touch()
                    return
                # Content changed, remove old entry
                self._evict(normalized_path)

            # Evict if at capacity
            while (
                len(self._cache) >= self._max_entries
                or self._current_bytes + content_bytes > self._max_total_bytes
            ):
                if not self._evict_lru():
                    break

            # Add new entry
            entry = CachedRead(
                path=normalized_path,
                content=content,
                content_hash=content_hash,
                timestamp=time(),
                size_bytes=content_bytes,
            )
            self._cache[normalized_path] = entry
            self._current_bytes += content_bytes

            logger.debug(f"Cached {path} ({content_bytes} bytes, hash: {content_hash})")

    def invalidate(self, path: str) -> None:
        """Invalidate cache for a modified file.

        Args:
            path: File path to invalidate
        """
        normalized_path = self._normalize_path(path)

        with self._lock:
            if normalized_path in self._cache:
                self._evict(normalized_path)
                logger.debug(f"Invalidated cache for {path}")

    def invalidate_all(self) -> None:
        """Invalidate all cached entries."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0
            logger.debug("Invalidated all cache entries")

    def should_warn_redundant(self, path: str) -> bool:
        """Check if file was read recently (potential redundant read).

        Args:
            path: File path to check

        Returns:
            True if this appears to be a redundant read
        """
        normalized_path = self._normalize_path(path)

        with self._lock:
            if normalized_path not in self._cache:
                return False

            entry = self._cache[normalized_path]

            # Check if read multiple times within window
            is_redundant = (
                entry.read_count >= self._redundant_threshold
                and (time() - entry.timestamp) < self._redundant_window
            )

            if is_redundant:
                self._stats.redundant_warnings += 1
                logger.warning(
                    f"Redundant read detected: {path} read {entry.read_count} times "
                    f"in {time() - entry.timestamp:.1f}s"
                )

            return is_redundant

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            stats = self._stats.to_dict()
            stats["entries"] = len(self._cache)
            stats["total_bytes"] = self._current_bytes
            stats["max_entries"] = self._max_entries
            stats["max_bytes"] = self._max_total_bytes

            # Find most read file
            if self._cache:
                most_read = max(self._cache.values(), key=lambda x: x.read_count)
                stats["most_read"] = {
                    "path": most_read.path,
                    "count": most_read.read_count,
                }

            return stats

    def get_cached_paths(self) -> list:
        """Get list of currently cached paths.

        Returns:
            List of cached file paths
        """
        with self._lock:
            return list(self._cache.keys())

    def _normalize_path(self, path: str) -> str:
        """Normalize a file path for consistent caching.

        Args:
            path: File path to normalize

        Returns:
            Normalized path string
        """
        # Remove leading/trailing whitespace
        path = path.strip()

        # Normalize path separators
        path = path.replace("\\", "/")

        # Remove duplicate slashes
        while "//" in path:
            path = path.replace("//", "/")

        return path

    def _evict(self, path: str) -> None:
        """Evict a specific entry from cache.

        Args:
            path: Path to evict
        """
        if path in self._cache:
            entry = self._cache.pop(path)
            self._current_bytes -= entry.size_bytes
            self._stats.evictions += 1

    def _evict_lru(self) -> bool:
        """Evict least recently used entry.

        Returns:
            True if an entry was evicted
        """
        if not self._cache:
            return False

        # Find LRU entry
        lru_path = min(self._cache.keys(), key=lambda k: self._cache[k].last_access)
        self._evict(lru_path)
        return True


class ReadCacheMiddleware:
    """Middleware to integrate ReadResultCache with tool pipeline.

    Wraps file read operations to use caching automatically.
    """

    def __init__(self, cache: Optional[ReadResultCache] = None):
        """Initialize middleware.

        Args:
            cache: ReadResultCache instance (creates default if None)
        """
        self._cache = cache or ReadResultCache()

    def wrap_read(self, read_func):
        """Wrap a read function with caching.

        Args:
            read_func: Original read function

        Returns:
            Wrapped function with caching
        """

        def cached_read(path: str, **kwargs):
            # Try cache first
            offset = kwargs.get("offset", 0)
            limit = kwargs.get("limit")

            cached = self._cache.get(path, offset=offset, limit=limit)
            if cached is not None:
                return {"content": cached, "cached": True}

            # Call original function
            result = read_func(path, **kwargs)

            # Cache the full content if successful
            if result.get("content"):
                # Only cache full reads
                if offset == 0 and limit is None:
                    self._cache.put(path, result["content"])

            return result

        return cached_read

    def invalidate_on_write(self, path: str) -> None:
        """Invalidate cache when a file is written.

        Args:
            path: Path of written file
        """
        self._cache.invalidate(path)

    @property
    def cache(self) -> ReadResultCache:
        """Get the underlying cache."""
        return self._cache


def create_read_cache(
    ttl_seconds: float = 300.0,
    max_entries: int = 100,
) -> ReadResultCache:
    """Factory function for creating ReadResultCache.

    Args:
        ttl_seconds: Cache TTL in seconds
        max_entries: Maximum cache entries

    Returns:
        Configured ReadResultCache instance
    """
    return ReadResultCache(ttl_seconds=ttl_seconds, max_entries=max_entries)
