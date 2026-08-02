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

"""Centralized cache management for tools.

This module provides a ToolCacheManager that replaces module-level caches
with an injectable service, enabling:
- Proper test isolation (caches can be cleared between tests)
- DI-based tool configuration
- Unified cache statistics and monitoring

Example:
    # Create cache manager (typically via DI)
    cache_manager = ToolCacheManager()

    # Get namespaced cache
    index_cache = cache_manager.get_namespace("code_search_index")
    index_cache["my_project"] = index_data

    # Clear for testing
    cache_manager.clear_namespace("code_search_index")

    # Or clear all
    cache_manager.clear_all()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class GenericCacheEntry:
    """A cached value with metadata.

    Attributes:
        value: The cached value
        created_at: Timestamp when entry was created
        accessed_at: Timestamp when entry was last accessed
        access_count: Number of times entry was accessed
        ttl: Time-to-live in seconds (None for no expiry)
    """

    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if entry has expired.

        Returns:
            True if entry has TTL and has exceeded it
        """
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """Update access timestamp and count."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistics for a cache namespace.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of entries evicted
        total_entries: Current number of entries
        total_size_bytes: Estimated size in bytes (if tracked)
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as percentage (0-100)
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100


class CacheNamespace:
    """A namespaced cache with TTL and statistics support.

    Provides a dict-like interface with additional features:
    - Optional TTL per entry
    - Access statistics
    - Max size limits
    - Thread safety
    """

    def __init__(
        self,
        name: str,
        max_entries: Optional[int] = None,
        default_ttl: Optional[float] = None,
    ):
        """Initialize cache namespace.

        Args:
            name: Namespace identifier
            max_entries: Maximum entries (None for unlimited)
            default_ttl: Default TTL in seconds (None for no expiry)
        """
        self.name = name
        self.max_entries = max_entries
        self.default_ttl = default_ttl

        self._data: Dict[str, GenericCacheEntry] = {}
        self._stats = CacheStats()
        self._lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a cached value.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        with self._lock:
            entry = self._data.get(key)

            if entry is None:
                self._stats.misses += 1
                return default

            if entry.is_expired():
                del self._data[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.total_entries -= 1
                return default

            entry.touch()
            self._stats.hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a cached value.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
        """
        with self._lock:
            # Check if we need to evict
            if self.max_entries and len(self._data) >= self.max_entries:
                self._evict_oldest()

            # Create or update entry
            entry = GenericCacheEntry(
                value=value,
                ttl=ttl if ttl is not None else self.default_ttl,
            )

            is_new = key not in self._data
            self._data[key] = entry

            if is_new:
                self._stats.total_entries += 1

    def delete(self, key: str) -> bool:
        """Delete a cached entry.

        Args:
            key: Cache key

        Returns:
            True if entry existed and was deleted
        """
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._stats.total_entries -= 1
                return True
            return False

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._data)
            self._data.clear()
            self._stats.total_entries = 0
            return count

    def keys(self) -> Iterator[str]:
        """Iterate over valid cache keys.

        Yields:
            Cache keys (excluding expired entries)
        """
        with self._lock:
            # Build list to avoid modification during iteration
            valid_keys = []
            expired_keys = []

            for key, entry in self._data.items():
                if entry.is_expired():
                    expired_keys.append(key)
                else:
                    valid_keys.append(key)

            # Clean up expired entries
            for key in expired_keys:
                del self._data[key]
                self._stats.evictions += 1
                self._stats.total_entries -= 1

            for key in valid_keys:
                yield key

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._data[key]
                self._stats.evictions += 1
                self._stats.total_entries -= 1
                return False
            return True

    def __len__(self) -> int:
        """Return number of entries (may include expired)."""
        return len(self._data)

    def __getitem__(self, key: str) -> Any:
        """Dict-style get (raises KeyError if not found)."""
        value = self.get(key, default=_MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-style set."""
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        """Dict-style delete."""
        if not self.delete(key):
            raise KeyError(key)

    def _evict_oldest(self) -> None:
        """Evict the oldest accessed entry."""
        if not self._data:
            return

        oldest_key = min(
            self._data.keys(),
            key=lambda k: self._data[k].accessed_at,
        )
        del self._data[oldest_key]
        self._stats.evictions += 1
        self._stats.total_entries -= 1

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


# Sentinel for missing values
_MISSING = object()


class ToolCacheManager:
    """Centralized cache management for tools.

    Replaces module-level caches with an injectable service that:
    - Provides namespaced caches for different tools
    - Enables test isolation via clear methods
    - Tracks statistics for monitoring
    - Supports TTL and size limits

    This is designed to be registered as a singleton in the DI container
    and injected into tools that need caching.

    Example:
        # In ServiceProvider
        container.register_singleton(ToolCacheManager, ToolCacheManager)

        # In tool
        class CodeSearchTool:
            def __init__(self, cache_manager: Optional[ToolCacheManager] = None):
                self._cache_manager = cache_manager

            def get_index(self, project: str) -> Dict:
                cache = self._get_cache()
                return cache.get(project, {})

            def _get_cache(self) -> CacheNamespace:
                if self._cache_manager:
                    return self._cache_manager.index_cache
                # Fallback to global
                return _INDEX_CACHE
    """

    def __init__(
        self,
        default_max_entries: Optional[int] = 1000,
        default_ttl: Optional[float] = None,
    ):
        """Initialize the cache manager.

        Args:
            default_max_entries: Default max entries per namespace
            default_ttl: Default TTL in seconds (None for no expiry)
        """
        self._namespaces: Dict[str, CacheNamespace] = {}
        self._default_max_entries = default_max_entries
        self._default_ttl = default_ttl
        self._lock = threading.RLock()

        # Pre-create well-known namespaces
        self._init_standard_namespaces()

    def _init_standard_namespaces(self) -> None:
        """Initialize standard cache namespaces for tools."""
        # Code search index cache
        self._namespaces["code_search_index"] = CacheNamespace(
            name="code_search_index",
            max_entries=50,  # Max 50 projects
            default_ttl=3600,  # 1 hour
        )

        # File content cache
        self._namespaces["file_content"] = CacheNamespace(
            name="file_content",
            max_entries=200,  # Max 200 files
            default_ttl=300,  # 5 minutes
        )

        # Database connection cache (no TTL, managed manually)
        self._namespaces["database_connections"] = CacheNamespace(
            name="database_connections",
            max_entries=20,  # Max 20 connections
            default_ttl=None,  # No expiry, managed by tool
        )

        # Path resolver cache
        self._namespaces["path_resolver"] = CacheNamespace(
            name="path_resolver",
            max_entries=100,
            default_ttl=600,  # 10 minutes
        )

        # Binary handler cache
        self._namespaces["binary_handlers"] = CacheNamespace(
            name="binary_handlers",
            max_entries=50,
            default_ttl=None,  # No expiry
        )

    def get_namespace(self, name: str) -> CacheNamespace:
        """Get or create a cache namespace.

        Args:
            name: Namespace identifier

        Returns:
            CacheNamespace for the given name
        """
        with self._lock:
            if name not in self._namespaces:
                self._namespaces[name] = CacheNamespace(
                    name=name,
                    max_entries=self._default_max_entries,
                    default_ttl=self._default_ttl,
                )
            return self._namespaces[name]

    def clear_namespace(self, name: str) -> int:
        """Clear a specific namespace.

        Args:
            name: Namespace to clear

        Returns:
            Number of entries cleared
        """
        with self._lock:
            if name in self._namespaces:
                return self._namespaces[name].clear()
            return 0

    def clear_all(self) -> int:
        """Clear all namespaces.

        Returns:
            Total number of entries cleared
        """
        with self._lock:
            total = 0
            for ns in self._namespaces.values():
                total += ns.clear()
            return total

    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all namespaces.

        Returns:
            Dict mapping namespace names to their stats
        """
        with self._lock:
            return {name: ns.stats for name, ns in self._namespaces.items()}

    def get_total_entries(self) -> int:
        """Get total entries across all namespaces.

        Returns:
            Total entry count
        """
        with self._lock:
            return sum(len(ns) for ns in self._namespaces.values())

    # =========================================================================
    # Specific Cache Accessors (for type safety and convenience)
    # =========================================================================

    @property
    def index_cache(self) -> CacheNamespace:
        """Get code search index cache."""
        return self.get_namespace("code_search_index")

    @property
    def file_content_cache(self) -> CacheNamespace:
        """Get file content cache."""
        return self.get_namespace("file_content")

    @property
    def connection_pool(self) -> CacheNamespace:
        """Get database connection pool cache."""
        return self.get_namespace("database_connections")

    @property
    def path_resolver_cache(self) -> CacheNamespace:
        """Get path resolver cache."""
        return self.get_namespace("path_resolver")

    @property
    def binary_handler_cache(self) -> CacheNamespace:
        """Get binary handler cache."""
        return self.get_namespace("binary_handlers")


# =========================================================================
# Singleton for backward compatibility
# =========================================================================

_default_manager: Optional[ToolCacheManager] = None
_manager_lock = threading.Lock()


def get_tool_cache_manager() -> ToolCacheManager:
    """Get or create the default tool cache manager.

    This provides backward compatibility for code that doesn't
    use DI yet. Prefer injecting ToolCacheManager directly.

    Returns:
        Default ToolCacheManager instance
    """
    global _default_manager
    with _manager_lock:
        if _default_manager is None:
            _default_manager = ToolCacheManager()
        return _default_manager


def reset_tool_cache_manager() -> None:
    """Reset the default tool cache manager.

    Useful for testing to ensure clean state between tests.
    """
    global _default_manager
    with _manager_lock:
        if _default_manager is not None:
            _default_manager.clear_all()
        _default_manager = None


__all__ = [
    # Main class
    "ToolCacheManager",
    # Supporting classes
    "CacheNamespace",
    "GenericCacheEntry",
    "CacheStats",
    # Factory functions
    "get_tool_cache_manager",
    "reset_tool_cache_manager",
]
