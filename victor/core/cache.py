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

"""Generic cache base classes and protocols.

This module provides unified cache abstractions for use across the framework.
It defines common protocols and base implementations to reduce code duplication
across the 40+ cache implementations in the codebase.

Key distinctions:
- victor/core/cache.py → Generic cache protocols and base classes
- victor/storage/cache/tiered_cache.py → Two-tier memory/disk cache implementation
- victor/storage/cache/tool_cache.py → Tool-specific caching with path indexing
- victor/runtime/cache_registry.py → Centralized cache registration and invalidation

Design Principles:
1. Protocol-based design for dependency inversion
2. Thread-safe operations by default
3. Pluggable eviction policies (LRU, TTL, custom)
4. Statistics tracking for observability
5. Namespace support for cache isolation

Usage:
    from victor.core.cache import Cache, LRUCache, TTLCache

    # Simple LRU cache
    cache = LRUCache(max_size=1000)
    cache.set("key", "value")
    value = cache.get("key")

    # TTL cache with namespace
    cache = TTLCache(ttl_seconds=300, default_namespace="my_cache")
    cache.set("key", "value", namespace="custom")
    value = cache.get("key", namespace="custom")
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

# Type variables for generic cache types
K = TypeVar("K")
V = TypeVar("V")


class EvictionPolicy(str, Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    FIFO = "fifo"  # First In First Out
    LIFO = "lifo"  # Last In First Out
    TTL = "ttl"  # Time To Live
    CUSTOM = "custom"  # Custom eviction logic


@dataclass
class CacheEntry:
    """A cache entry with metadata.

    Attributes:
        key: The cache key
        value: The cached value
        created_at: Timestamp when entry was created
        last_accessed_at: Timestamp when entry was last accessed
        access_count: Number of times this entry was accessed
        ttl: Optional TTL in seconds
        size: Estimated size in bytes (for size-based eviction)
        metadata: Optional custom metadata
    """

    key: Any
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if this entry has expired based on TTL.

        Args:
            current_time: Current timestamp (uses time.time() if None)

        Returns:
            True if entry is expired
        """
        if self.ttl is None:
            return False

        t = current_time or time.time()
        return (t - self.created_at) > self.ttl

    def touch(self, current_time: Optional[float] = None) -> None:
        """Update last_accessed_at and increment access_count.

        Args:
            current_time: Current timestamp (uses time.time() if None)
        """
        self.last_accessed_at = current_time or time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        sets: Number of set operations
        deletes: Number of delete operations
        evictions: Number of evictions
        current_size: Current number of entries
        max_size: Maximum number of entries
        hit_rate: Cache hit rate (0-1)
    """

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0

    @property
    def total_requests(self) -> int:
        """Total number of get requests."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        total = self.total_requests
        return self.hits / total if total > 0 else 0.0

    def __add__(self, other: "CacheStats") -> "CacheStats":
        """Combine two stats objects."""
        return CacheStats(
            hits=self.hits + other.hits,
            misses=self.misses + other.misses,
            sets=self.sets + other.sets,
            deletes=self.deletes + other.deletes,
            evictions=self.evictions + other.evictions,
            current_size=self.current_size,  # Size doesn't add
            max_size=max(self.max_size, other.max_size),
        )


@runtime_checkable
class CacheProtocol(Protocol[K, V]):
    """Protocol for basic cache operations.

    This protocol defines the minimal interface that all caches should implement.
    It enables dependency inversion and allows swapping cache implementations.

    Example:
        def process_with_cache(cache: CacheProtocol[str, dict]) -> dict:
            data = cache.get("key")
            if data is None:
                data = expensive_operation()
                cache.set("key", data)
            return data
    """

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get a value from the cache.

        Args:
            key: The cache key
            default: Value to return if key not found

        Returns:
            The cached value or default if not found
        """
        ...

    def set(self, key: K, value: V) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
        """
        ...

    def delete(self, key: K) -> bool:
        """Delete a value from the cache.

        Args:
            key: The cache key

        Returns:
            True if the key was found and deleted
        """
        ...

    def clear(self) -> None:
        """Clear all entries from the cache."""
        ...

    def __contains__(self, key: K) -> bool:
        """Check if a key is in the cache."""
        ...

    def __len__(self) -> int:
        """Get the number of entries in the cache."""
        ...


@runtime_checkable
class NamespacedCacheProtocol(CacheProtocol[K, V], Protocol):
    """Protocol for caches with namespace support.

    Namespaces allow logical separation of cache entries within the same
    cache instance. Useful for multi-tenant scenarios or cache isolation.

    Example:
        cache.set("key", "value1", namespace="user1")
        cache.set("key", "value2", namespace="user2")
        # Different values, same key, different namespaces
    """

    def get(self, key: K, default: Optional[V] = None, namespace: str = "default") -> Optional[V]:
        """Get a value from the cache.

        Args:
            key: The cache key
            default: Value to return if key not found
            namespace: Cache namespace for isolation

        Returns:
            The cached value or default if not found
        """
        ...

    def set(self, key: K, value: V, namespace: str = "default") -> None:
        """Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
            namespace: Cache namespace for isolation
        """
        ...

    def delete(self, key: K, namespace: str = "default") -> bool:
        """Delete a value from the cache.

        Args:
            key: The cache key
            namespace: Cache namespace

        Returns:
            True if the key was found and deleted
        """
        ...

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear entries from the cache.

        Args:
            namespace: If provided, only clear this namespace. Otherwise clear all.
        """
        ...

    def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a specific namespace.

        Args:
            namespace: The namespace to clear

        Returns:
            Number of entries cleared
        """
        ...


@runtime_checkable
class ObservableCacheProtocol(CacheProtocol[K, V], Protocol):
    """Protocol for caches with statistics tracking.

    Provides observability into cache performance through hit rates,
    eviction counts, and other metrics.
    """

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats object with current metrics
        """
        ...

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        ...


class BaseCache(ABC, Generic[K, V]):
    """Abstract base class for cache implementations.

    Provides common functionality like statistics tracking, thread safety,
    and utility methods. Subclasses only need to implement the core storage
    operations.

    Attributes:
        _lock: Thread lock for thread-safe operations
        _stats: Cache statistics
    """

    def __init__(self) -> None:
        """Initialize the base cache."""
        self._lock = threading.RLock()
        self._stats = CacheStats()

    @abstractmethod
    def _get_entry(self, key: K) -> Optional[CacheEntry]:
        """Get a cache entry by key (subclass implementation).

        Args:
            key: The cache key

        Returns:
            CacheEntry or None if not found
        """
        ...

    @abstractmethod
    def _set_entry(self, key: K, entry: CacheEntry) -> Optional[CacheEntry]:
        """Set a cache entry (subclass implementation).

        Args:
            key: The cache key
            entry: The entry to set

        Returns:
            The evicted entry if any, None otherwise
        """
        ...

    @abstractmethod
    def _delete_entry(self, key: K) -> bool:
        """Delete a cache entry (subclass implementation).

        Args:
            key: The cache key

        Returns:
            True if the entry was found and deleted
        """
        ...

    @abstractmethod
    def _clear_entries(self) -> None:
        """Clear all cache entries (subclass implementation)."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of entries in the cache."""
        ...

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get a value from the cache.

        Args:
            key: The cache key
            default: Value to return if key not found

        Returns:
            The cached value or default if not found
        """
        with self._lock:
            entry = self._get_entry(key)
            if entry is None:
                self._stats.misses += 1
                return default

            if entry.is_expired():
                self._delete_entry(key)
                self._stats.misses += 1
                return default

            entry.touch()
            self._stats.hits += 1
            return entry.value

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (None = no expiration)
        """
        with self._lock:
            entry = CacheEntry(key=key, value=value, ttl=ttl)
            evicted = self._set_entry(key, entry)
            if evicted:
                self._stats.evictions += 1
            self._stats.sets += 1

    def delete(self, key: K) -> bool:
        """Delete a value from the cache.

        Args:
            key: The cache key

        Returns:
            True if the key was found and deleted
        """
        with self._lock:
            deleted = self._delete_entry(key)
            if deleted:
                self._stats.deletes += 1
            return deleted

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._clear_entries()

    def __contains__(self, key: K) -> bool:
        """Check if a key is in the cache."""
        with self._lock:
            entry = self._get_entry(key)
            return entry is not None and not entry.is_expired()

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats object with current metrics
        """
        with self._lock:
            stats = self._stats
            return CacheStats(
                hits=stats.hits,
                misses=stats.misses,
                sets=stats.sets,
                deletes=stats.deletes,
                evictions=stats.evictions,
                current_size=len(self),
                max_size=stats.max_size,
            )

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._stats = CacheStats(max_size=self._stats.max_size)

    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes.

        Args:
            value: The value to measure

        Returns:
            Estimated size in bytes
        """
        try:
            import sys

            return sys.getsizeof(value)
        except Exception:
            return 0


class LRUCache(BaseCache[K, V]):
    """Thread-safe LRU (Least Recently Used) cache.

    Evicts the least recently used items when the cache is full.

    Attributes:
        max_size: Maximum number of entries in the cache
        _cache: OrderedDict storage (acts as LRU cache)

    Example:
        cache = LRUCache(max_size=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        value = cache.get("key1")  # Returns "value1", marks key1 as recently used
    """

    def __init__(self, max_size: int = 128) -> None:
        """Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries (default: 128)
        """
        super().__init__()
        self.max_size = max_size
        self._cache: OrderedDict[K, CacheEntry] = OrderedDict()
        self._stats.max_size = max_size

    def _get_entry(self, key: K) -> Optional[CacheEntry]:
        """Get a cache entry, moving it to the end (most recently used)."""
        entry = self._cache.get(key)
        if entry is not None:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
        return entry

    def _set_entry(self, key: K, entry: CacheEntry) -> Optional[CacheEntry]:
        """Set a cache entry, evicting LRU if necessary."""
        evicted = None

        # Update max_size if changed
        self._stats.max_size = self.max_size

        if key in self._cache:
            # Update existing, move to end
            self._cache.move_to_end(key)

        # Check if we need to evict
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Pop first item (least recently used)
            lru_key, lru_entry = self._cache.popitem(last=False)
            evicted = lru_entry

        self._cache[key] = entry
        self._cache.move_to_end(key)
        return evicted

    def _delete_entry(self, key: K) -> bool:
        """Delete a cache entry."""
        try:
            del self._cache[key]
            return True
        except KeyError:
            return False

    def _clear_entries(self) -> None:
        """Clear all entries."""
        self._cache.clear()

    def __len__(self) -> int:
        """Get the number of entries in the cache."""
        return len(self._cache)

    def keys(self) -> Iterator[K]:
        """Get an iterator over cache keys in LRU order."""
        with self._lock:
            return iter(self._cache.keys())

    def values(self) -> Iterator[V]:
        """Get an iterator over cache values in LRU order."""
        with self._lock:
            return (entry.value for entry in self._cache.values())

    def items(self) -> Iterator[tuple[K, V]]:
        """Get an iterator over cache items in LRU order."""
        with self._lock:
            return ((key, entry.value) for key, entry in self._cache.items())


class TTLCache(BaseCache[K, V]):
    """Thread-safe TTL (Time To Live) cache.

    Items expire after a configured time period. Expired items are
    lazily removed on access.

    Attributes:
        ttl_seconds: Default TTL for all entries in seconds
        max_size: Maximum number of entries (0 = unlimited)
        _cache: Dict storage

    Example:
        cache = TTLCache(ttl_seconds=60, max_size=1000)
        cache.set("key", "value")
        # 60 seconds later...
        value = cache.get("key")  # Returns None (expired)
    """

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 0) -> None:
        """Initialize the TTL cache.

        Args:
            ttl_seconds: Default time-to-live in seconds (default: 300)
            max_size: Maximum number of entries (0 = unlimited)
        """
        super().__init__()
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[K, CacheEntry] = {}
        self._stats.max_size = max_size

    def _get_entry(self, key: K) -> Optional[CacheEntry]:
        """Get a cache entry if not expired."""
        entry = self._cache.get(key)
        if entry is not None and entry.is_expired():
            # Lazy expiration
            del self._cache[key]
            return None
        return entry

    def _set_entry(self, key: K, entry: CacheEntry) -> Optional[CacheEntry]:
        """Set a cache entry with default TTL."""
        # Set default TTL if not specified
        if entry.ttl is None:
            entry.ttl = self.ttl_seconds

        evicted = None

        # Check size limit
        if self.max_size > 0 and len(self._cache) >= self.max_size and key not in self._cache:
            # Evict a random entry (simple FIFO-like eviction)
            # For smarter eviction, use LRUCache instead
            evict_key = next(iter(self._cache))
            evicted = self._cache.pop(evict_key)

        self._cache[key] = entry
        return evicted

    def _delete_entry(self, key: K) -> bool:
        """Delete a cache entry."""
        try:
            del self._cache[key]
            return True
        except KeyError:
            return False

    def _clear_entries(self) -> None:
        """Clear all entries."""
        self._cache.clear()

    def __len__(self) -> int:
        """Get the number of entries in the cache."""
        return len(self._cache)

    def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def keys(self) -> Iterator[K]:
        """Get an iterator over cache keys."""
        with self._lock:
            # Filter out expired entries
            self.cleanup_expired()
            return iter(self._cache.keys())

    def values(self) -> Iterator[V]:
        """Get an iterator over cache values."""
        with self._lock:
            self.cleanup_expired()
            return (entry.value for entry in self._cache.values())

    def items(self) -> Iterator[tuple[K, V]]:
        """Get an iterator over cache items."""
        with self._lock:
            self.cleanup_expired()
            return ((key, entry.value) for key, entry in self._cache.items())


class NamespacedCache(BaseCache[K, V]):
    """Thread-safe cache with namespace support.

    Each namespace is an independent cache within the same instance.
    Useful for logical separation of cached data.

    Attributes:
        default_namespace: Default namespace to use
        namespace_factory: Factory function for creating namespace caches
        _namespaces: Dict of namespace -> cache

    Example:
        cache = NamespacedCache(default_namespace="app")
        cache.set("key", "value1")  # Uses "app" namespace
        cache.set("key", "value2", namespace="user")  # Uses "user" namespace
    """

    def __init__(
        self,
        default_namespace: str = "default",
        cache_factory: Optional[Callable[[], BaseCache]] = None,
    ) -> None:
        """Initialize the namespaced cache.

        Args:
            default_namespace: Default namespace name
            cache_factory: Factory for creating namespace caches (default: LRUCache(128))
        """
        super().__init__()
        self.default_namespace = default_namespace
        self._cache_factory = cache_factory or (lambda: LRUCache(max_size=128))
        self._namespaces: Dict[str, BaseCache] = {}

    def _get_namespace(self, namespace: str) -> BaseCache:
        """Get or create a namespace cache."""
        if namespace not in self._namespaces:
            self._namespaces[namespace] = self._cache_factory()
        return self._namespaces[namespace]

    def get(
        self, key: K, default: Optional[V] = None, namespace: Optional[str] = None
    ) -> Optional[V]:
        """Get a value from the cache.

        Args:
            key: The cache key
            default: Value to return if key not found
            namespace: Cache namespace (uses default if None)

        Returns:
            The cached value or default if not found
        """
        ns = namespace or self.default_namespace
        cache = self._get_namespace(ns)
        return cache.get(key, default)

    def set(
        self,
        key: K,
        value: V,
        namespace: Optional[str] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
            namespace: Cache namespace (uses default if None)
            ttl: Time-to-live in seconds
        """
        ns = namespace or self.default_namespace
        cache = self._get_namespace(ns)
        cache.set(key, value, ttl=ttl)

    def delete(self, key: K, namespace: Optional[str] = None) -> bool:
        """Delete a value from the cache.

        Args:
            key: The cache key
            namespace: Cache namespace (uses default if None)

        Returns:
            True if the key was found and deleted
        """
        ns = namespace or self.default_namespace
        cache = self._get_namespace(ns)
        return cache.delete(key)

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear entries from the cache.

        Args:
            namespace: If provided, only clear this namespace. Otherwise clear all.
        """
        if namespace is not None:
            if namespace in self._namespaces:
                self._namespaces[namespace].clear()
        else:
            for cache in self._namespaces.values():
                cache.clear()

    def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a specific namespace.

        Args:
            namespace: The namespace to clear

        Returns:
            Number of entries cleared
        """
        if namespace not in self._namespaces:
            return 0

        cache = self._namespaces[namespace]
        count = len(cache)
        cache.clear()
        return count

    def _get_entry(self, key: K) -> Optional[CacheEntry]:
        """Get a cache entry (uses default namespace)."""
        return self._get_namespace(self.default_namespace)._get_entry(key)

    def _set_entry(self, key: K, entry: CacheEntry) -> Optional[CacheEntry]:
        """Set a cache entry (uses default namespace)."""
        return self._get_namespace(self.default_namespace)._set_entry(key, entry)

    def _delete_entry(self, key: K) -> bool:
        """Delete a cache entry (uses default namespace)."""
        return self._get_namespace(self.default_namespace)._delete_entry(key)

    def _clear_entries(self) -> None:
        """Clear all entries from all namespaces."""
        for cache in self._namespaces.values():
            cache.clear()

    def __len__(self) -> int:
        """Get total number of entries across all namespaces."""
        return sum(len(cache) for cache in self._namespaces.values())

    def __contains__(self, key: K) -> bool:
        """Check if a key is in any namespace."""
        return any(key in cache for cache in self._namespaces.values())

    def get_stats(self) -> CacheStats:
        """Get combined statistics from all namespaces."""
        combined = CacheStats()
        for cache in self._namespaces.values():
            combined += cache.get_stats()
        combined.current_size = len(self)
        return combined


def create_hash_cache_key(*args: Any, **kwargs: Any) -> str:
    """Create a stable hash key for caching.

    Useful for creating cache keys from function arguments.

    Args:
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key

    Returns:
        Stable hash string

    Example:
        key = create_hash_cache_key("model", "prompt", temperature=0.7)
        # Same inputs always produce the same hash
    """
    import json

    try:
        # Sort kwargs for stable output
        key_dict = {"args": args, "kwargs": sorted(kwargs.items())}
        key_str = json.dumps(key_dict, sort_keys=True, default=str)
    except Exception:
        # Fallback for non-serializable objects
        key_parts = [str(a) for a in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = "|".join(key_parts)

    return hashlib.sha256(key_str.encode()).hexdigest()


def cached(
    cache: Optional[BaseCache] = None,
    key_fn: Optional[Callable[..., str]] = None,
    ttl: Optional[float] = None,
):
    """Decorator for caching function results.

    Args:
        cache: Cache instance to use (creates default LRUCache if None)
        key_fn: Function to generate cache keys (uses create_hash_cache_key if None)
        ttl: Default TTL for cached values

    Returns:
        Decorated function with caching

    Example:
        @cached(ttl=60)
        def expensive_function(x, y):
            return x * y

        result1 = expensive_function(2, 3)  # Computes
        result2 = expensive_function(2, 3)  # Returns cached result
    """
    if cache is None:
        cache = LRUCache(max_size=128)

    if key_fn is None:
        key_fn = create_hash_cache_key

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            key = key_fn(func.__name__, *args, **kwargs)
            result = cache.get(key)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result, ttl=ttl)
            return result

        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.original_function = func  # type: ignore[attr-defined]
        return wrapper

    return decorator


__all__ = [
    # Protocols
    "CacheProtocol",
    "NamespacedCacheProtocol",
    "ObservableCacheProtocol",
    # Enums
    "EvictionPolicy",
    # Data classes
    "CacheEntry",
    "CacheStats",
    # Base classes
    "BaseCache",
    # Implementations
    "LRUCache",
    "TTLCache",
    "NamespacedCache",
    # Utilities
    "create_hash_cache_key",
    "cached",
]
