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

"""Async-safe cache manager for vertical extension instances.

This module provides lock-per-key caching to avoid contention in parallel
loading scenarios. Each cache key has its own lock, allowing concurrent
access to different keys without blocking.

Design Principles:
    - Lock-per-key for reduced contention
    - Double-check locking pattern for efficiency
    - Async-safe operations
    - Telemetry integration
    - Backward compatible with sync code
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class AsyncSafeCacheManager:
    """Async-safe cache with lock-per-key for parallel loading.

    Unlike the single-lock ExtensionCacheManager, this cache uses a separate
    lock for each key, allowing concurrent access to different keys without
    contention.

    Example:
        cache = AsyncSafeCacheManager()

        # Async usage
        value = await cache.get_or_create_async(
            "MyVertical:mod:qual",
            "middleware",
            async_factory_fn
        )

        # Sync usage (thread-safe but not async)
        value = cache.get_or_create(
            "MyVertical:mod:qual",
            "middleware",
            factory_fn
        )
    """

    def __init__(self) -> None:
        """Initialize the async-safe cache manager."""
        self._cache: Dict[str, Any] = {}
        self._locks: Dict[str, threading.RLock] = {}
        self._meta_lock = threading.RLock()
        self._hit_count: int = 0
        self._miss_count: int = 0

    def _make_key(self, namespace: str, key: str) -> str:
        """Build composite cache key from namespace and key."""
        return f"{namespace}:{key}"

    def _get_lock(self, cache_key: str) -> threading.RLock:
        """Get or create lock for a specific cache key.

        Args:
            cache_key: The composite cache key

        Returns:
            RLock for this specific key
        """
        with self._meta_lock:
            if cache_key not in self._locks:
                self._locks[cache_key] = threading.RLock()
            return self._locks[cache_key]

    def get_or_create(
        self,
        namespace: str,
        key: str,
        factory: Callable[[], Any],
    ) -> Any:
        """Get cached value or create via factory. Thread-safe with lock-per-key.

        Uses double-check locking pattern for efficiency:
        1. Check cache without lock (fast path)
        2. Acquire lock for this key
        3. Check cache again with lock (another thread may have created it)
        4. Create if still needed

        Args:
            namespace: Cache namespace (typically derived from the vertical class)
            key: Extension key (e.g., "middleware", "safety_extension")
            factory: Zero-argument callable that creates the value

        Returns:
            The cached or newly created value
        """
        cache_key = self._make_key(namespace, key)

        # Fast path: check cache without lock
        if cache_key in self._cache:
            self._hit_count += 1
            return self._cache[cache_key]

        # Slow path: acquire lock for this specific key
        lock = self._get_lock(cache_key)
        with lock:
            # Double-check: another thread may have created it while we waited
            if cache_key in self._cache:
                self._hit_count += 1
                return self._cache[cache_key]

            # Create value
            self._miss_count += 1
            value = factory()
            self._cache[cache_key] = value
            return value

    async def get_or_create_async(
        self,
        namespace: str,
        key: str,
        factory: Awaitable[Callable[[], Any]] | Callable[[], Any],
    ) -> Any:
        """Get cached value or create via async factory. Async-safe with lock-per-key.

        Uses double-check locking pattern adapted for async:
        1. Check cache without lock (fast path)
        2. Acquire lock for this key
        3. Check cache again with lock
        4. Create if still needed (awaiting factory if async)

        Args:
            namespace: Cache namespace
            key: Extension key
            factory: Async or sync callable that creates the value

        Returns:
            The cached or newly created value

        Example:
            async def create_middleware():
                return await SomeAsyncClass.create()

            middleware = await cache.get_or_create_async(
                "MyVertical:mod:qual",
                "middleware",
                create_middleware
            )
        """
        cache_key = self._make_key(namespace, key)

        # Fast path: check cache without lock
        if cache_key in self._cache:
            self._hit_count += 1
            return self._cache[cache_key]

        # Slow path: acquire lock for this specific key
        lock = self._get_lock(cache_key)
        with lock:
            # Double-check: another thread/task may have created it
            if cache_key in self._cache:
                self._hit_count += 1
                return self._cache[cache_key]

            # Create value
            self._miss_count += 1

            # Handle both sync and async factories
            if asyncio.iscoroutinefunction(factory):
                # Factory is async coroutine function
                value = await factory()
            elif asyncio.iscoroutine(factory):
                # Factory is coroutine object
                value = await factory
            else:
                # Factory is sync callable
                value = await asyncio.to_thread(factory)

            self._cache[cache_key] = value
            return value

    def get_if_cached(self, namespace: str, key: str) -> Tuple[bool, Any]:
        """Return (True, value) if cached, (False, None) otherwise.

        No factory invocation occurs; this is a pure lookup.

        Args:
            namespace: Cache namespace
            key: Extension key

        Returns:
            Tuple of (found, value). If found is False, value is None
        """
        cache_key = self._make_key(namespace, key)

        # No lock needed for read (Python dict reads are thread-safe)
        if cache_key in self._cache:
            self._hit_count += 1
            return True, self._cache[cache_key]

        self._miss_count += 1
        return False, None

    def load_optional(
        self,
        namespace: str,
        key: str,
        loader: Callable[[], Optional[Any]],
    ) -> Optional[Any]:
        """Load optional extension - cached on hit, invokes loader on miss.

        Unlike get_or_create, this method returns None on cache miss instead of
        invoking a factory. The loader is only called if the value is not cached.

        Args:
            namespace: Cache namespace
            key: Extension key
            loader: Zero-argument callable that loads the value

        Returns:
            Cached value or result of loader (may be None)

        Example:
            safety = cache.load_optional(
                "MyVertical:mod:qual",
                "safety",
                lambda: get_safety_extension()
            )
        """
        cache_key = self._make_key(namespace, key)

        # Fast path: check cache without lock
        if cache_key in self._cache:
            self._hit_count += 1
            return self._cache[cache_key]

        # Slow path: acquire lock and load
        lock = self._get_lock(cache_key)
        with lock:
            # Double-check
            if cache_key in self._cache:
                self._hit_count += 1
                return self._cache[cache_key]

            # Load value (may be None)
            self._miss_count += 1
            value = loader()

            # Cache even if None (avoid repeated failed loads)
            if value is not None:
                self._cache[cache_key] = value

            return value

    def invalidate(self, namespace: Optional[str] = None, key: Optional[str] = None) -> int:
        """Invalidate cache entries.

        Args:
            namespace: If specified, only invalidate entries in this namespace.
                        If None, invalidates all entries.
            key: If specified (with namespace), only invalidate this specific key.
                  If None (with namespace), invalidates all keys in namespace.

        Returns:
            Number of entries invalidated

        Example:
            # Invalidate all entries
            count = cache.invalidate()

            # Invalidate all entries for a namespace
            count = cache.invalidate(namespace="MyVertical:mod:qual")

            # Invalidate specific key
            count = cache.invalidate(
                namespace="MyVertical:mod:qual",
                key="middleware"
            )
        """
        with self._meta_lock:
            if namespace is None:
                # Invalidate all
                count = len(self._cache)
                self._cache.clear()
                self._locks.clear()
                return count

            if key is None:
                # Invalidate all keys in namespace
                prefix = f"{namespace}:"
                to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
                for k in to_delete:
                    del self._cache[k]
                    if k in self._locks:
                        del self._locks[k]
                return len(to_delete)

            # Invalidate specific key
            cache_key = self._make_key(namespace, key)
            if cache_key in self._cache:
                del self._cache[cache_key]
            if cache_key in self._locks:
                del self._locks[cache_key]
            return 1

        return 0

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with keys: cache_size, hit_count, miss_count, hit_rate
        """
        with self._meta_lock:
            total = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total if total > 0 else 0.0

            return {
                "cache_size": len(self._cache),
                "lock_count": len(self._locks),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(hit_rate, 4),
            }

    def clear(self) -> None:
        """Clear all cache entries and locks."""
        with self._meta_lock:
            self._cache.clear()
            self._locks.clear()
            self._hit_count = 0
            self._miss_count = 0


__all__ = ["AsyncSafeCacheManager"]
