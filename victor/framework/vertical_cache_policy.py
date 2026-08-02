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

"""Cache policy abstraction for vertical integration pipeline.

This module provides a pluggable policy interface so cache behavior can be
swapped without modifying `VerticalIntegrationPipeline`.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class VerticalIntegrationCachePolicy(Protocol):
    """Protocol for vertical integration cache behavior."""

    def load(
        self,
        cache_key: str,
        *,
        cache: Dict[str, str],
        timestamps: Dict[str, float],
        cache_ttl: int,
    ) -> Optional[str]:
        """Load cached payload for key, returning None on miss."""

    def save(
        self,
        cache_key: str,
        payload: str,
        *,
        cache: Dict[str, str],
        timestamps: Dict[str, float],
        max_entries: int,
    ) -> None:
        """Persist payload for key, enforcing policy constraints."""

    def get_stats(
        self,
        *,
        cache: Dict[str, str],
        max_entries: int,
    ) -> Dict[str, int]:
        """Return cache statistics."""

    def clear(
        self,
        *,
        cache: Dict[str, str],
        timestamps: Dict[str, float],
    ) -> None:
        """Clear cache state."""


class InMemoryLRUVerticalIntegrationCachePolicy:
    """In-memory LRU+TTL cache policy for integration metadata."""

    def __init__(self) -> None:
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def load(
        self,
        cache_key: str,
        *,
        cache: Dict[str, str],
        timestamps: Dict[str, float],
        cache_ttl: int,
    ) -> Optional[str]:
        payload = cache.get(cache_key)
        if payload is None:
            self._misses += 1
            return None

        timestamp = timestamps.get(cache_key)
        if timestamp is not None and cache_ttl > 0:
            age_seconds = time.monotonic() - timestamp
            if age_seconds > cache_ttl:
                cache.pop(cache_key, None)
                timestamps.pop(cache_key, None)
                self._misses += 1
                self._expirations += 1
                return None

        # LRU touch by reinserting key at end
        value = cache.pop(cache_key)
        cache[cache_key] = value
        self._hits += 1
        return value

    def save(
        self,
        cache_key: str,
        payload: str,
        *,
        cache: Dict[str, str],
        timestamps: Dict[str, float],
        max_entries: int,
    ) -> None:
        if cache_key in cache:
            cache.pop(cache_key, None)
        cache[cache_key] = payload
        timestamps[cache_key] = time.monotonic()

        if max_entries <= 0:
            return

        while len(cache) > max_entries:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key, None)
            timestamps.pop(oldest_key, None)
            self._evictions += 1

    def get_stats(
        self,
        *,
        cache: Dict[str, str],
        max_entries: int,
    ) -> Dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "expirations": self._expirations,
            "size": len(cache),
            "max_entries": max_entries,
        }

    def clear(
        self,
        *,
        cache: Dict[str, str],
        timestamps: Dict[str, float],
    ) -> None:
        cache.clear()
        timestamps.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0


__all__ = [
    "VerticalIntegrationCachePolicy",
    "InMemoryLRUVerticalIntegrationCachePolicy",
]
