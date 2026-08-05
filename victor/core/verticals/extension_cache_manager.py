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

"""Thread-safe cache manager for vertical extension instances.

Extracts caching logic from VerticalExtensionLoader into a focused component
responsible for storing, retrieving, and invalidating cached extension instances
using namespaced composite keys and RLock for thread safety.

The composite cache key format is ``{namespace}:{key}`` which matches the
existing pattern used by VerticalExtensionLoader.

Usage:
    cache = ExtensionCacheManager()

    # Get or create a cached extension
    value = cache.get_or_create("MyVertical:mod:qual", "middleware", factory_fn)

    # Check if cached without invoking a factory
    found, value = cache.get_if_cached("MyVertical:mod:qual", "middleware")

    # Load optional extension (cache hits but not misses)
    value = cache.load_optional("MyVertical:mod:qual", "safety", loader_fn)

    # Invalidate entries
    removed = cache.invalidate(namespace="MyVertical:mod:qual")
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional, Tuple


class ExtensionCacheManager:
    """Thread-safe cache for vertical extension instances.

    Extracts caching logic from VerticalExtensionLoader into a focused component.
    Uses namespaced composite keys and RLock for thread safety.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def _make_key(self, namespace: str, key: str) -> str:
        """Build composite cache key from namespace and key."""
        return f"{namespace}:{key}"

    def get_or_create(self, namespace: str, key: str, factory: Callable[[], Any]) -> Any:
        """Get cached value or create via factory. Thread-safe.

        If the key is not present in the cache, the factory callable is invoked
        and its return value is stored before being returned.

        Args:
            namespace: Cache namespace (typically derived from the vertical class).
            key: Extension key (e.g., "middleware", "safety_extension").
            factory: Zero-argument callable that creates the value. Only called
                     on cache miss.

        Returns:
            The cached or newly created value.
        """
        cache_key = self._make_key(namespace, key)
        with self._lock:
            if cache_key not in self._cache:
                self._cache[cache_key] = factory()
            return self._cache[cache_key]

    def get_if_cached(self, namespace: str, key: str) -> Tuple[bool, Any]:
        """Return (True, value) if cached, (False, None) otherwise.

        No factory invocation occurs; this is a pure lookup.

        Args:
            namespace: Cache namespace.
            key: Extension key.

        Returns:
            A tuple of (found, value). If found is False, value is None.
        """
        cache_key = self._make_key(namespace, key)
        with self._lock:
            if cache_key in self._cache:
                return True, self._cache[cache_key]
        return False, None

    def load_optional(
        self,
        namespace: str,
        key: str,
        loader: Callable[[], Optional[Any]],
    ) -> Optional[Any]:
        """Load an optional extension: cache hits but not misses.

        If the extension is already cached, return it immediately. Otherwise
        invoke the loader. If the loader returns a non-None value, cache and
        return it. If the loader returns None, return None without caching
        the miss so future calls can retry.

        Args:
            namespace: Cache namespace.
            key: Extension key.
            loader: Zero-argument callable that returns the extension or None.

        Returns:
            The cached/loaded extension instance, or None.
        """
        cached, value = self.get_if_cached(namespace, key)
        if cached:
            return value

        resolved = loader()
        if resolved is None:
            return None

        return self.get_or_create(namespace, key, lambda: resolved)

    def invalidate(
        self,
        namespace: Optional[str] = None,
        key: Optional[str] = None,
    ) -> int:
        """Invalidate cache entries. Returns count of entries removed.

        If namespace is None, clear all entries.
        If namespace is provided but key is None, clear all entries in that namespace.
        If both namespace and key are provided, clear that specific entry.

        Args:
            namespace: Optional namespace to scope invalidation.
            key: Optional key to target a specific entry.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            if namespace is None:
                count = len(self._cache)
                self._cache.clear()
                return count

            if key is not None:
                cache_key = self._make_key(namespace, key)
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    return 1
                return 0

            # Remove all entries whose key starts with the namespace prefix.
            prefix = f"{namespace}:"
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)
