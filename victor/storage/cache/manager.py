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

"""Unified Cache Manager for Victor.

This module provides a centralized cache management system with:
- Namespace-scoped access for logical separation
- Unified interface across different cache types
- Integration with the DI container
- Global cache management utilities

Design Principles:
- Facade pattern: Simple interface over TieredCache
- Namespace isolation: Each component gets its own namespace
- Consistent API: Same interface for all cache operations
- Observable: Statistics and monitoring support

Example Usage:
    from victor.storage.cache.manager import get_cache_manager

    # Get namespace-scoped cache
    cache = get_cache_manager()
    tool_cache = cache.namespace("tools")

    # Use cache
    tool_cache.set("read_file:abc123", result)
    cached = tool_cache.get("read_file:abc123")

    # Clear specific namespace
    cache.clear_namespace("tools")

    # Get statistics
    stats = cache.get_stats()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from victor.storage.cache.config import CacheConfig
from victor.storage.cache.tiered_cache import TieredCache

logger = logging.getLogger(__name__)


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache implementations."""

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a cached value."""
        ...

    def delete(self, key: str) -> bool:
        """Delete a cached value."""
        ...

    def clear(self) -> int:
        """Clear all cached values."""
        ...


class CacheLifecycleProtocol(Protocol):
    """Extended protocol for cache lifecycle management.

    Provides a unified interface for TTL enforcement, eviction policies,
    size management, and cache health reporting. All cache implementations
    should satisfy this protocol in addition to CacheProtocol.
    """

    def get_stats(self) -> "CacheStats":
        """Return current cache statistics (hits, misses, evictions)."""
        ...

    def evict_expired(self) -> int:
        """Remove all entries past their TTL. Returns count of evicted entries."""
        ...

    def get_size(self) -> int:
        """Return current number of entries in the cache."""
        ...

    def get_memory_usage_bytes(self) -> int:
        """Estimate current memory usage in bytes."""
        ...

    def shutdown(self) -> None:
        """Gracefully shut down the cache (flush pending writes, release resources)."""
        ...


@dataclass
class CacheStats:
    """Cache statistics."""

    memory_hits: int = 0
    memory_misses: int = 0
    disk_hits: int = 0
    disk_misses: int = 0
    total_sets: int = 0
    total_evictions: int = 0
    namespaces: List[str] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        """Calculate overall hit rate."""
        total_hits = self.memory_hits + self.disk_hits
        total_requests = total_hits + self.memory_misses + self.disk_misses
        if total_requests == 0:
            return 0.0
        return total_hits / total_requests

    @property
    def memory_hit_rate(self) -> float:
        """Calculate memory cache hit rate."""
        total = self.memory_hits + self.memory_misses
        if total == 0:
            return 0.0
        return self.memory_hits / total


class CacheNamespace:
    """Namespace-scoped cache access.

    Provides a simplified interface for cache operations within a namespace.
    All operations are automatically scoped to the namespace.

    Example:
        cache = CacheNamespace(manager, "embeddings")
        cache.set("model_v1", embeddings)
        result = cache.get("model_v1")
    """

    def __init__(self, manager: "CacheManager", namespace: str):
        """Initialize namespace cache.

        Args:
            manager: Parent cache manager
            namespace: Namespace identifier
        """
        self._manager = manager
        self._namespace = namespace

    @property
    def namespace(self) -> str:
        """Get namespace name."""
        return self._namespace

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value.

        Args:
            key: Cache key within namespace

        Returns:
            Cached value or None if not found
        """
        return self._manager.get(key, self._namespace)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a cached value.

        Args:
            key: Cache key within namespace
            value: Value to cache
            ttl: Optional TTL in seconds

        Returns:
            True if successfully cached
        """
        return self._manager.set(key, value, self._namespace, ttl)

    def delete(self, key: str) -> bool:
        """Delete a cached value.

        Args:
            key: Cache key within namespace

        Returns:
            True if deleted
        """
        return self._manager.delete(key, self._namespace)

    def clear(self) -> int:
        """Clear all values in this namespace.

        Returns:
            Number of entries cleared
        """
        return self._manager.clear_namespace(self._namespace)

    def get_or_set(
        self,
        key: str,
        factory: callable,
        ttl: Optional[int] = None,
    ) -> Any:
        """Get cached value or compute and cache it.

        Args:
            key: Cache key
            factory: Callable to compute value if not cached
            ttl: Optional TTL in seconds

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl)
        return value


class CacheManager:
    """Unified cache manager providing namespace-scoped access.

    This class provides a centralized interface for all caching needs,
    with logical separation via namespaces.

    Namespaces:
    - "tools": Tool execution results
    - "embeddings": Embedding vectors
    - "responses": LLM responses
    - "code_search": Code search results
    - "metadata": File/project metadata

    Example:
        manager = CacheManager()

        # Get namespace-scoped cache
        tool_cache = manager.namespace("tools")
        tool_cache.set("read_file:hash123", content)

        # Direct access
        manager.set("key", value, namespace="embeddings")

        # Statistics
        stats = manager.get_stats()
        print(f"Hit rate: {stats.hit_rate:.2%}")
    """

    # Well-known namespace names
    NAMESPACE_TOOLS = "tools"
    NAMESPACE_EMBEDDINGS = "embeddings"
    NAMESPACE_RESPONSES = "responses"
    NAMESPACE_CODE_SEARCH = "code_search"
    NAMESPACE_METADATA = "metadata"

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self._config = config or CacheConfig()
        self._cache = TieredCache(self._config)
        self._namespaces: Dict[str, CacheNamespace] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> CacheConfig:
        """Get cache configuration."""
        return self._config

    def namespace(self, name: str) -> CacheNamespace:
        """Get or create a namespace-scoped cache.

        Args:
            name: Namespace identifier

        Returns:
            CacheNamespace for scoped operations
        """
        with self._lock:
            if name not in self._namespaces:
                self._namespaces[name] = CacheNamespace(self, name)
            return self._namespaces[name]

    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get a cached value.

        Args:
            key: Cache key
            namespace: Cache namespace

        Returns:
            Cached value or None if not found
        """
        return self._cache.get(key, namespace)

    def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a cached value.

        Args:
            key: Cache key
            value: Value to cache
            namespace: Cache namespace
            ttl: Optional TTL in seconds

        Returns:
            True if successfully cached
        """
        return self._cache.set(key, value, namespace, ttl)

    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a cached value.

        Args:
            key: Cache key
            namespace: Cache namespace

        Returns:
            True if deleted
        """
        return self._cache.delete(key, namespace)

    def clear_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of entries cleared
        """
        return self._cache.clear(namespace)

    def clear_all(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries cleared
        """
        return self._cache.clear()

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with hit/miss counts
        """
        raw_stats = self._cache.get_stats()
        return CacheStats(
            memory_hits=int(raw_stats.get("memory_hits", 0)),
            memory_misses=int(raw_stats.get("memory_misses", 0)),
            disk_hits=int(raw_stats.get("disk_hits", 0)),
            disk_misses=int(raw_stats.get("disk_misses", 0)),
            total_sets=int(raw_stats.get("sets", 0)),
            total_evictions=int(raw_stats.get("evictions", 0)),
            namespaces=list(self._namespaces.keys()),
        )

    def close(self) -> None:
        """Close cache and release resources."""
        self._cache.close()


# =============================================================================
# Global Cache Manager
# =============================================================================

_global_manager: Optional[CacheManager] = None
_global_lock = threading.Lock()


def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Get or create the global cache manager.

    Args:
        config: Cache configuration (only used on first call)

    Returns:
        Global CacheManager instance
    """
    global _global_manager
    with _global_lock:
        if _global_manager is None:
            _global_manager = CacheManager(config)
        return _global_manager


def set_cache_manager(manager: CacheManager) -> None:
    """Set the global cache manager.

    Args:
        manager: CacheManager to use as global
    """
    global _global_manager
    with _global_lock:
        if _global_manager is not None:
            _global_manager.close()
        _global_manager = manager


def reset_cache_manager() -> None:
    """Reset the global cache manager (for testing)."""
    global _global_manager
    with _global_lock:
        if _global_manager is not None:
            _global_manager.close()
            _global_manager = None


# =============================================================================
# Convenience Functions
# =============================================================================


def get_tools_cache() -> CacheNamespace:
    """Get the tools cache namespace."""
    return get_cache_manager().namespace(CacheManager.NAMESPACE_TOOLS)


def get_embeddings_cache() -> CacheNamespace:
    """Get the embeddings cache namespace."""
    return get_cache_manager().namespace(CacheManager.NAMESPACE_EMBEDDINGS)


def get_responses_cache() -> CacheNamespace:
    """Get the LLM responses cache namespace."""
    return get_cache_manager().namespace(CacheManager.NAMESPACE_RESPONSES)


def get_code_search_cache() -> CacheNamespace:
    """Get the code search cache namespace."""
    return get_cache_manager().namespace(CacheManager.NAMESPACE_CODE_SEARCH)
