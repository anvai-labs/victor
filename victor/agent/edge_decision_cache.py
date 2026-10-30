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

"""Shared core for the edge-model micro-deciders.

ComplexityEstimator and TaskClassifier run the same flow: count the request,
fall back to a heuristic when disabled, serve a TTL-cached result, otherwise
ask the edge model, cache its answer, and fall back to the heuristic on any
edge failure. They also parse the same ``LABEL: value`` response format.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable, ClassVar, Dict, Generic, Optional, TypeVar

T = TypeVar("T")


def extract_labeled_float(response: str, label: str, default: float) -> float:
    """Return the float after ``LABEL:`` in an edge-model response, else ``default``."""
    match = re.search(rf"{label}:\s*([0-9.]+)", response, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return default


def extract_labeled_text(response: str, label: str, default: str) -> str:
    """Return the stripped rest-of-line after ``LABEL:``, else ``default``."""
    match = re.search(rf"{label}:\s*(.+)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return default


class CachedEdgeDecider(Generic[T]):
    """TTL cache, request statistics and fallback flow for an edge decider.

    Subclasses set ``_STATS_TOTAL_KEY`` (the request-count key reported by
    :meth:`get_statistics`) and ``_LOG_NAME``, and call :meth:`_decide`.
    """

    _STATS_TOTAL_KEY: ClassVar[str] = "total_requests"
    _LOG_NAME: ClassVar[str] = "EdgeDecider"

    def __init__(
        self,
        enabled: bool = True,
        use_cache: bool = True,
        cache_ttl: int = 3600,
    ):
        """Initialize the decider.

        Args:
            enabled: Whether the edge model is consulted (heuristic otherwise)
            use_cache: Whether to cache edge-model results
            cache_ttl: Cache time-to-live in seconds
        """
        self.enabled = enabled
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[T, float]] = {}
        self._request_count = 0
        self._cache_hits = 0
        self._logger = logging.getLogger(type(self).__module__)

    async def _decide(
        self,
        query: str,
        edge: Callable[[float], Awaitable[T]],
        heuristic: Callable[[float], T],
    ) -> T:
        """Run the shared decision flow; both callables receive the start time."""
        import time

        self._request_count += 1
        start_time = time.time()

        if not self.enabled:
            return heuristic(start_time)

        if self.use_cache:
            cached = self._get_from_cache(query)
            if cached:
                self._cache_hits += 1
                self._logger.debug(f"[{self._LOG_NAME}] Cache hit for query")
                return cached

        try:
            result = await edge(start_time)
            if self.use_cache:
                self._add_to_cache(query, result)
            return result
        except Exception as e:
            self._logger.warning(f"[{self._LOG_NAME}] Edge model failed: {e}, using heuristic")
            return heuristic(start_time)

    def _get_from_cache(self, query: str) -> Optional[T]:
        """Return a cached result that is younger than ``cache_ttl``."""
        import time

        if query in self._cache:
            result, timestamp = self._cache[query]
            if time.time() - timestamp < self.cache_ttl:
                return result
            del self._cache[query]
        return None

    def _add_to_cache(self, query: str, result: T) -> None:
        import time

        self._cache[query] = (result, time.time())

    def get_statistics(self) -> Dict[str, Any]:
        """Return request counts and cache statistics."""
        cache_hit_rate = (
            (self._cache_hits / self._request_count) if self._request_count > 0 else 0.0
        )
        return {
            self._STATS_TOTAL_KEY: self._request_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
        }

    def reset_statistics(self) -> None:
        """Reset counters and clear the cache."""
        self._request_count = 0
        self._cache_hits = 0
        self._cache.clear()
