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

"""Aggregation Service for observability metrics.

Provides time-series aggregation, tool statistics, and performance metrics.
Reuses percentile calculation from streaming_metrics.py.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import statistics
import logging

from victor.core.events import MessagingEvent
from victor.observability.query_service import QueryService

logger = logging.getLogger(__name__)


@dataclass
class AggregationServiceConfig:
    """Configuration for AggregationService."""

    time_windows: Dict[str, int] = None
    cache_ttl_seconds: int = 300
    max_events_per_query: int = 10000

    def __post_init__(self):
        if self.time_windows is None:
            self.time_windows = {"1h": 3600, "24h": 86400, "7d": 604800}


class AggregationService:
    """Service for aggregating observability metrics over time windows."""

    def __init__(self, config: Optional[AggregationServiceConfig] = None):
        """Initialize AggregationService.

        Args:
            config: Service configuration
        """
        self.config = config or AggregationServiceConfig()
        self._query_service = QueryService()
        self._cache: Dict[str, Tuple[datetime, Dict]] = {}

    def _parse_time_window(self, window: str) -> timedelta:
        """Parse time window string to timedelta.

        Args:
            window: Time window (e.g., "1h", "24h", "7d")

        Returns:
            Timedelta for the window

        Raises:
            ValueError: If window format is invalid
        """
        if window in self.config.time_windows:
            return timedelta(seconds=self.config.time_windows[window])

        # Parse custom format (e.g., "30m", "2h", "7d")
        try:
            unit = window[-1]
            value = int(window[:-1])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid time window format: {window}")

        if unit == "s":
            return timedelta(seconds=value)
        elif unit == "m":
            return timedelta(minutes=value)
        elif unit == "h":
            return timedelta(hours=value)
        elif unit == "d":
            return timedelta(days=value)
        else:
            raise ValueError(f"Invalid time window format: {window}")

    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for method call.

        Args:
            method: Method name
            **kwargs: Method arguments

        Returns:
            Cache key string
        """
        parts = [method]
        for k, v in sorted(kwargs.items()):
            parts.append(f"{k}={v}")
        return ":".join(parts)

    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached result if not expired.

        Args:
            key: Cache key

        Returns:
            Cached result or None
        """
        if key in self._cache:
            timestamp, result = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.config.cache_ttl_seconds):
                return result
            else:
                del self._cache[key]
        return None

    def _set_cache(self, key: str, result: Dict) -> None:
        """Cache result with current timestamp.

        Args:
            key: Cache key
            result: Result to cache
        """
        self._cache[key] = (datetime.now(), result)

    async def get_metrics_history(
        self,
        time_window: str = "1h",
        bucket_size: Optional[str] = None,
    ) -> Dict:
        """Get historical metrics aggregated over time buckets.

        Args:
            time_window: Time window (e.g., "1h", "24h", "7d")
            bucket_size: Bucket size (e.g., "5m", "1h", "1d"). Default: auto-calculated

        Returns:
            Dict with time-series metrics
        """
        cache_key = self._get_cache_key(
            "metrics_history", time_window=time_window, bucket_size=bucket_size
        )
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Calculate time range
        window_delta = self._parse_time_window(time_window)
        end_time = datetime.now()
        start_time = end_time - window_delta

        # Calculate bucket size if not specified
        if bucket_size is None:
            # Auto-calculate: ~50 buckets for the time window
            total_seconds = window_delta.total_seconds()
            bucket_seconds = total_seconds / 50
            if bucket_seconds < 60:
                bucket_size = f"{int(bucket_seconds)}s"
            elif bucket_seconds < 3600:
                bucket_size = f"{int(bucket_seconds / 60)}m"
            else:
                bucket_size = f"{int(bucket_seconds / 3600)}h"

        bucket_delta = self._parse_time_window(bucket_size)

        # Query events
        events = await self._query_service.get_recent_events(
            start_time=start_time,
            end_time=end_time,
            limit=self.config.max_events_per_query,
        )

        # Bucket events
        buckets = {}
        current_bucket = start_time

        while current_bucket < end_time:
            bucket_key = current_bucket.strftime("%Y-%m-%dT%H:%M:%S")
            buckets[bucket_key] = {
                "tool_calls": 0,
                "errors": 0,
                "durations": [],
            }
            current_bucket += bucket_delta

        # Fill buckets
        for event in events:
            # Find bucket
            event_time = event.datetime
            bucket_idx = int(
                (event_time - start_time).total_seconds() / bucket_delta.total_seconds()
            )
            bucket_keys = list(buckets.keys())
            if 0 <= bucket_idx < len(bucket_keys):
                bucket_key = bucket_keys[bucket_idx]
                bucket = buckets[bucket_key]

                # Aggregate metrics
                if event.topic.startswith("tool."):
                    bucket["tool_calls"] += 1
                    if event.data and "duration_ms" in event.data:
                        bucket["durations"].append(event.data["duration_ms"])
                    if not event.data.get("success", True):
                        bucket["errors"] += 1

        # Convert to response format
        response = {
            "time_window": time_window,
            "bucket_size": bucket_size,
            "buckets": list(buckets.keys()),
            "metrics": {
                "tool_calls": [b["tool_calls"] for b in buckets.values()],
                "errors": [b["errors"] for b in buckets.values()],
                "avg_duration_ms": [
                    statistics.mean(b["durations"]) if b["durations"] else None
                    for b in buckets.values()
                ],
            },
        }

        self._set_cache(cache_key, response)
        return response

    async def get_tool_statistics(self) -> Dict:
        """Get tool usage statistics with success rates and duration percentiles.

        Returns:
            Dict with tool statistics
        """
        cache_key = self._get_cache_key("tool_stats")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Query all tool events
        events = await self._query_service.get_recent_events(
            topic_pattern="tool.*",
            limit=self.config.max_events_per_query,
        )

        # Aggregate by tool
        tool_stats: Dict[str, Dict] = {}

        for event in events:
            if not event.data:
                continue

            tool_name = event.data.get("tool_name", "unknown")

            if tool_name not in tool_stats:
                tool_stats[tool_name] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "durations": [],
                }

            stats = tool_stats[tool_name]
            stats["total_calls"] += 1

            if event.data.get("success", True):
                stats["successful_calls"] += 1

            if "duration_ms" in event.data:
                stats["durations"].append(event.data["duration_ms"])

        # Calculate percentiles using streaming_metrics pattern
        def percentile(values: List[float], p: float) -> Optional[float]:
            if not values or len(values) < 2:
                return None
            sorted_values = sorted(values)
            idx = int(len(sorted_values) * p)
            return sorted_values[min(idx, len(sorted_values) - 1)]

        # Convert to response format
        tools = []
        for tool_name, stats in sorted(tool_stats.items()):
            durations = stats["durations"]
            tools.append(
                {
                    "tool_name": tool_name,
                    "total_calls": stats["total_calls"],
                    "success_rate": (
                        stats["successful_calls"] / stats["total_calls"]
                        if stats["total_calls"] > 0
                        else 0
                    ),
                    "avg_duration_ms": (statistics.mean(durations) if durations else None),
                    "p50_duration_ms": percentile(durations, 0.50),
                    "p95_duration_ms": percentile(durations, 0.95),
                    "p99_duration_ms": percentile(durations, 0.99),
                }
            )

        response = {"tools": tools}
        self._set_cache(cache_key, response)
        return response

    async def get_performance_metrics(self) -> Dict:
        """Get performance metrics including latency percentiles and throughput.

        Returns:
            Dict with performance metrics
        """
        cache_key = self._get_cache_key("performance")
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Query recent events (last hour)
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        events = await self._query_service.get_recent_events(
            start_time=start_time,
            end_time=end_time,
            limit=self.config.max_events_per_query,
        )

        # Collect metrics
        durations = []
        errors = 0
        tool_calls = 0

        for event in events:
            if event.topic.startswith("tool."):
                tool_calls += 1

                if event.data:
                    if not event.data.get("success", True):
                        errors += 1

                    if "duration_ms" in event.data:
                        durations.append(event.data["duration_ms"])

        # Calculate percentiles
        def percentile(values: List[float], p: float) -> Optional[float]:
            if not values or len(values) < 2:
                return None
            sorted_values = sorted(values)
            idx = int(len(sorted_values) * p)
            return sorted_values[min(idx, len(sorted_values) - 1)]

        # Calculate throughput
        time_window_seconds = 3600  # 1 hour
        total_events = len(events)

        response = {
            "latency_ms": {
                "p50": percentile(durations, 0.50),
                "p95": percentile(durations, 0.95),
                "p99": percentile(durations, 0.99),
                "avg": statistics.mean(durations) if durations else None,
            },
            "throughput": {
                "requests_per_second": total_events / time_window_seconds,
                "tool_calls_per_second": tool_calls / time_window_seconds,
            },
            "errors": {
                "total_errors": errors,
                "error_rate": errors / total_events if total_events > 0 else 0,
            },
        }

        self._set_cache(cache_key, response)
        return response
