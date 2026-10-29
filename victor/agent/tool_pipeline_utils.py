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

"""Standalone tool pipeline utilities (extracted from tool_pipeline.py).

ToolRateLimiter and LRUToolCache are self-contained utilities — they have
no dependency on the ToolPipeline class.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from victor.config.tool_selection_defaults import ToolPipelineDefaults


class ToolRateLimiter:
    """Token bucket rate limiter for tool execution.

    Implements a token bucket algorithm to limit the rate of tool executions.
    This prevents overwhelming external services or local resources.

    Attributes:
        rate: Tokens per second to add (refill rate)
        burst: Maximum tokens that can accumulate (bucket size)
        tokens: Current available tokens
        last_update: Timestamp of last token refill

    Example:
        limiter = ToolRateLimiter(rate=10.0, burst=5)

        # Synchronous check
        if limiter.acquire():
            execute_tool()

        # Asynchronous wait
        await limiter.wait()
        execute_tool()
    """

    def __init__(self, rate: float = 10.0, burst: int = 5):
        """Initialize the rate limiter.

        Args:
            rate: Tokens per second (refill rate)
            burst: Maximum tokens (bucket capacity)
        """
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)  # Start with full bucket
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if token acquired, False if no tokens available.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def wait(self) -> None:
        """Wait until a token is available.

        This method blocks asynchronously until a token can be acquired.
        """
        while not self.acquire():
            # Sleep for time needed to get one token
            await asyncio.sleep(1.0 / self.rate)


class LRUToolCache:
    """LRU (Least Recently Used) cache with TTL for idempotent tool results.

    Provides efficient caching with LRU eviction policy and time-based expiry.
    This prevents stale results from being returned after files change and
    bounds memory growth for long-running sessions.

    Attributes:
        _cache: OrderedDict maintaining insertion/access order
        _timestamps: Dict mapping keys to creation timestamps
        _max_size: Maximum number of entries to cache (default: 50)
        _ttl_seconds: Time-to-live in seconds (default: 300s = 5 minutes)

    Example:
        cache = LRUToolCache(max_size=50, ttl_seconds=300)
        cache.set("key", result)
        result = cache.get("key")  # Returns cached result if not expired
    """

    def __init__(
        self,
        max_size: int = ToolPipelineDefaults.IDEMPOTENT_CACHE_MAX_SIZE,
        ttl_seconds: float = ToolPipelineDefaults.IDEMPOTENT_CACHE_TTL,
    ):
        """Initialize the LRU cache with TTL.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live in seconds
        """
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        # Co-design review item 16: concurrent tool dispatch means multiple
        # coroutines can now call get()/set() around the SAME cache instance
        # without an intervening await inside either method, so CPython's
        # GIL already makes each individual call atomic — no torn
        # OrderedDict state is possible. This lock exists for a caller that
        # needs an atomic read-check-then-write span ACROSS an await
        # boundary (e.g. "miss the cache, run the tool, then populate it"
        # without a duplicate concurrent run for the same key).
        #
        # NOTE this is NOT guaranteed today by upstream batch deduplication:
        # deduplicate_tool_calls() signatures raw, pre-normalization
        # arguments, while get_cached_result()/cache_result() signature
        # fully-normalized arguments (post parameter-enforcement defaults,
        # post search-routing rewrite) — two calls with different raw args
        # can still converge on the same cache signature after
        # normalization and land in the same parallel gather batch. Without
        # this lock actually being used by those two methods, that race is
        # merely wasteful (both miss, both execute the same idempotent
        # operation, one cache write clobbers the other) rather than
        # corrupting, because both sides of the race represent the exact
        # same operation by construction — but it is a real race, not one
        # dedup rules out.
        self.lock = asyncio.Lock()

    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry has expired.

        Args:
            key: Cache key to check

        Returns:
            True if entry is expired or timestamp missing
        """
        if key not in self._timestamps:
            return True
        age = time.time() - self._timestamps[key]
        return age > self._ttl_seconds

    def _evict_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries evicted
        """
        expired_keys = [k for k in self._cache if self._is_expired(k)]
        for key in expired_keys:
            del self._cache[key]
            self._timestamps.pop(key, None)
        return len(expired_keys)

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.

        If found and not expired, the entry is moved to the end (most recently used).
        Expired entries are automatically removed.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        if key in self._cache:
            # Check TTL expiration
            if self._is_expired(key):
                # Remove expired entry
                del self._cache[key]
                self._timestamps.pop(key, None)
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache.

        If the key exists, it's updated and moved to the end with refreshed TTL.
        If cache is full, expired entries are evicted first, then LRU eviction.

        Args:
            key: Cache key
            value: Value to cache
        """
        current_time = time.time()

        if key in self._cache:
            # Update existing and move to end
            self._cache.move_to_end(key)
        self._cache[key] = value
        self._timestamps[key] = current_time

        # Evict expired entries first (cheaper than LRU for memory)
        if len(self._cache) > self._max_size:
            self._evict_expired()

        # Still over capacity? LRU eviction
        while len(self._cache) > self._max_size:
            oldest_key, _ = self._cache.popitem(last=False)
            self._timestamps.pop(oldest_key, None)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        self._timestamps.clear()

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache)

    def items(self):
        """Return all non-expired items in the cache."""
        # Filter out expired entries
        return [(k, v) for k, v in self._cache.items() if not self._is_expired(k)]

    def remove(self, key: str) -> bool:
        """Remove a specific key from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was found and removed, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            self._timestamps.pop(key, None)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring.

        Returns:
            Dict with size, max_size, ttl_seconds, expired_count
        """
        expired_count = sum(1 for k in self._cache if self._is_expired(k))
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "expired_count": expired_count,
            "active_count": len(self._cache) - expired_count,
        }
