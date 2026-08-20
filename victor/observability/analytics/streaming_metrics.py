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

"""
Streaming Performance Monitoring for LLM Responses.

This module provides comprehensive metrics collection for streaming responses:
- Time to First Token (TTFT)
- Tokens per second throughput
- Streaming latency percentiles
- Tool call timing within streams

Usage:
    collector = get_metrics_collector()
    metrics = collector.create_metrics("req_123", "claude-3-5-haiku", "anthropic")

    async for chunk in MetricsStreamWrapper(stream, metrics, collector):
        # Process chunk
        pass

    # Get summary
    summary = collector.get_summary(provider="anthropic")
"""

import asyncio
import json
import logging
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MetricType(Enum):
    """Types of streaming metrics."""

    TTFT = "time_to_first_token"
    TTLT = "time_to_last_token"
    TOKENS_PER_SECOND = "tokens_per_second"
    TOTAL_TOKENS = "total_tokens"
    TOOL_CALL_LATENCY = "tool_call_latency"
    CHUNK_INTERVAL = "chunk_interval"


@dataclass
class StreamMetrics:
    """Metrics for a single streaming response (ANALYTICS-FOCUSED).

    .. deprecated::
        For new code, prefer victor.agent.stream_handler.StreamMetrics for operational metrics.
        This class is analytics-focused and will be deprecated in version 0.10.0.

    This class is designed for analytics and observability tracking with:
    - Request context tracking (request_id, model, provider)
    - Detailed chunk interval statistics (P50, P95, P99)
    - Error tracking for monitoring

    For operational streaming metrics (during agent execution), use:
        from victor.agent.stream_handler import StreamMetrics

    Differences from canonical StreamMetrics:
    - Analytics-focused (this) vs operational (canonical)
    - Request context fields (request_id, model, provider) vs metadata dict
    - Unix timestamps vs relative timing
    - More detailed percentile analysis

    Attributes:
        request_id: Unique identifier for the request
        model: Model name/identifier
        provider: Provider name (anthropic, openai, etc.)
        start_time: Unix timestamp when stream started
        first_token_time: Unix timestamp of first token
        last_token_time: Unix timestamp of last token
        total_chunks: Number of chunks received
        total_tokens: Estimated total tokens (content + tool calls)
        content_tokens: Tokens in content only
        tool_calls_count: Number of tool calls in response
        tool_call_times: Timestamps of tool call chunks
        chunk_intervals: Time intervals between chunks
        errors: List of errors encountered
    """

    request_id: str
    model: str
    provider: str

    # Timing metrics (Unix timestamps)
    start_time: float = 0
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None

    # Token metrics
    total_chunks: int = 0
    total_tokens: int = 0
    content_tokens: int = 0

    # Tool call metrics
    tool_calls_count: int = 0
    tool_call_times: List[float] = field(default_factory=list)

    # Chunk timing (in seconds)
    chunk_intervals: List[float] = field(default_factory=list)

    # Error tracking
    errors: List[str] = field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.first_token_time and self.start_time:
            return (self.first_token_time - self.start_time) * 1000
        return None

    @property
    def total_duration_ms(self) -> Optional[float]:
        """Total stream duration in milliseconds."""
        if self.last_token_time and self.start_time:
            return (self.last_token_time - self.start_time) * 1000
        return None

    @property
    def tokens_per_second(self) -> Optional[float]:
        """Average tokens per second."""
        duration = self.total_duration_ms
        if duration and duration > 0 and self.total_tokens > 0:
            return self.total_tokens / (duration / 1000)
        return None

    @property
    def avg_chunk_interval_ms(self) -> Optional[float]:
        """Average time between chunks in milliseconds."""
        if self.chunk_intervals:
            return statistics.mean(self.chunk_intervals) * 1000
        return None

    @property
    def p50_chunk_interval_ms(self) -> Optional[float]:
        """Median chunk interval in milliseconds."""
        if self.chunk_intervals:
            return statistics.median(self.chunk_intervals) * 1000
        return None

    @property
    def p95_chunk_interval_ms(self) -> Optional[float]:
        """95th percentile chunk interval in milliseconds."""
        if len(self.chunk_intervals) >= 20:
            sorted_intervals = sorted(self.chunk_intervals)
            idx = int(len(sorted_intervals) * 0.95)
            return sorted_intervals[idx] * 1000
        return None

    @property
    def p99_chunk_interval_ms(self) -> Optional[float]:
        """99th percentile chunk interval in milliseconds."""
        if len(self.chunk_intervals) >= 100:
            sorted_intervals = sorted(self.chunk_intervals)
            idx = int(len(sorted_intervals) * 0.99)
            return sorted_intervals[idx] * 1000
        return None

    @property
    def is_complete(self) -> bool:
        """Check if stream completed (has last token time)."""
        return self.last_token_time is not None

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "provider": self.provider,
            "start_time": self.start_time,
            "ttft_ms": self.ttft_ms,
            "total_duration_ms": self.total_duration_ms,
            "tokens_per_second": self.tokens_per_second,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "content_tokens": self.content_tokens,
            "tool_calls_count": self.tool_calls_count,
            "avg_chunk_interval_ms": self.avg_chunk_interval_ms,
            "p50_chunk_interval_ms": self.p50_chunk_interval_ms,
            "p95_chunk_interval_ms": self.p95_chunk_interval_ms,
            "p99_chunk_interval_ms": self.p99_chunk_interval_ms,
            "errors": self.errors,
            "is_complete": self.is_complete,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Export metrics as JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class MetricsSummary:
    """Aggregated metrics summary."""

    count: int = 0
    ttft_ms: Dict[str, Optional[float]] = field(default_factory=dict)
    tokens_per_second: Dict[str, Optional[float]] = field(default_factory=dict)
    duration_ms: Dict[str, Optional[float]] = field(default_factory=dict)
    error_rate: float = 0.0
    total_tokens: int = 0
    total_tool_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Export summary as dictionary."""
        return {
            "count": self.count,
            "ttft_ms": self.ttft_ms,
            "tokens_per_second": self.tokens_per_second,
            "duration_ms": self.duration_ms,
            "error_rate": self.error_rate,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
        }


class StreamingMetricsCollector:
    """
    Collects and aggregates streaming metrics across requests.

    Features:
    - Real-time metric collection
    - Histogram aggregation with percentiles
    - Export to monitoring systems
    - Callback support for real-time processing

    Usage:
        collector = StreamingMetricsCollector()

        # Create metrics for a request
        metrics = collector.create_metrics("req_123", "claude-3-5-haiku", "anthropic")

        # ... stream processing ...

        # Record completed metrics
        collector.record_metrics(metrics)

        # Get summary
        summary = collector.get_summary(provider="anthropic", last_n=100)
    """

    def __init__(
        self,
        max_history: int = 1000,
        export_path: Optional[Path] = None,
    ):
        """Initialize the metrics collector.

        Args:
            max_history: Maximum metrics to keep in history
            export_path: Optional path for metrics export
        """
        self.max_history = max_history
        self.export_path = export_path

        self._metrics_history: List[StreamMetrics] = []
        self._callbacks: List[Callable[[StreamMetrics], None]] = []
        self._lock = asyncio.Lock()

        logger.debug(f"StreamingMetricsCollector initialized. Max history: {max_history}")

    def create_metrics(
        self,
        request_id: Optional[str] = None,
        model: str = "unknown",
        provider: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StreamMetrics:
        """Create a new metrics instance for a stream.

        Args:
            request_id: Unique request ID. Generated if not provided.
            model: Model name/identifier
            provider: Provider name
            metadata: Additional metadata to attach

        Returns:
            New StreamMetrics instance
        """
        if request_id is None:
            request_id = f"req_{uuid.uuid4().hex[:12]}"

        metrics = StreamMetrics(
            request_id=request_id,
            model=model,
            provider=provider,
            start_time=time.time(),
            metadata=metadata or {},
        )

        logger.debug(f"Created metrics for request {request_id}")

        return metrics

    async def record_metrics(self, metrics: StreamMetrics):
        """Record completed metrics.

        Args:
            metrics: Completed StreamMetrics instance
        """
        async with self._lock:
            self._metrics_history.append(metrics)

            # Trim history
            if len(self._metrics_history) > self.max_history:
                self._metrics_history = self._metrics_history[-self.max_history :]

        # Export if configured
        if self.export_path:
            await self._export_metrics(metrics)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.warning(f"Metrics callback failed: {e}")

        logger.debug(
            f"Recorded metrics for {metrics.request_id}. "
            f"TTFT: {metrics.ttft_ms:.1f}ms, TPS: {metrics.tokens_per_second:.1f}"
            if metrics.ttft_ms and metrics.tokens_per_second
            else f"Recorded metrics for {metrics.request_id}"
        )

    def record_metrics_sync(self, metrics: StreamMetrics):
        """Record completed metrics (synchronous version).

        Args:
            metrics: Completed StreamMetrics instance
        """
        self._metrics_history.append(metrics)

        # Trim history
        if len(self._metrics_history) > self.max_history:
            self._metrics_history = self._metrics_history[-self.max_history :]

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.warning(f"Metrics callback failed: {e}")

    def on_metrics(self, callback: Callable[[StreamMetrics], None]):
        """Register callback for completed metrics.

        Args:
            callback: Function to call with completed metrics
        """
        self._callbacks.append(callback)
        logger.debug(f"Registered metrics callback. Total: {len(self._callbacks)}")

    def get_summary(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        last_n: int = 100,
    ) -> MetricsSummary:
        """Get summary statistics for recent metrics.

        Args:
            provider: Filter by provider name
            model: Filter by model name
            last_n: Number of recent metrics to analyze

        Returns:
            MetricsSummary with aggregated statistics
        """
        # Filter metrics
        metrics = self._metrics_history[-last_n:]
        if provider:
            metrics = [m for m in metrics if m.provider == provider]
        if model:
            metrics = [m for m in metrics if m.model == model]

        if not metrics:
            return MetricsSummary()

        # Collect values
        ttft_values = [m.ttft_ms for m in metrics if m.ttft_ms is not None]
        tps_values = [m.tokens_per_second for m in metrics if m.tokens_per_second]
        duration_values = [m.total_duration_ms for m in metrics if m.total_duration_ms]

        summary = MetricsSummary(
            count=len(metrics),
            ttft_ms={
                "avg": statistics.mean(ttft_values) if ttft_values else None,
                "min": min(ttft_values) if ttft_values else None,
                "max": max(ttft_values) if ttft_values else None,
                "p50": statistics.median(ttft_values) if ttft_values else None,
                "p95": self._percentile(ttft_values, 0.95),
                "p99": self._percentile(ttft_values, 0.99),
            },
            tokens_per_second={
                "avg": statistics.mean(tps_values) if tps_values else None,
                "min": min(tps_values) if tps_values else None,
                "max": max(tps_values) if tps_values else None,
                "p50": statistics.median(tps_values) if tps_values else None,
            },
            duration_ms={
                "avg": statistics.mean(duration_values) if duration_values else None,
                "min": min(duration_values) if duration_values else None,
                "max": max(duration_values) if duration_values else None,
                "p50": statistics.median(duration_values) if duration_values else None,
                "p95": self._percentile(duration_values, 0.95),
            },
            error_rate=sum(1 for m in metrics if m.has_errors) / len(metrics),
            total_tokens=sum(m.total_tokens for m in metrics),
            total_tool_calls=sum(m.tool_calls_count for m in metrics),
        )

        return summary

    def get_recent_metrics(
        self,
        count: int = 10,
        provider: Optional[str] = None,
    ) -> List[StreamMetrics]:
        """Get recent metrics.

        Args:
            count: Number of metrics to return
            provider: Filter by provider name

        Returns:
            List of recent StreamMetrics
        """
        metrics = self._metrics_history
        if provider:
            metrics = [m for m in metrics if m.provider == provider]
        return metrics[-count:]

    def clear_history(self):
        """Clear metrics history."""
        self._metrics_history.clear()
        logger.info("Metrics history cleared")

    # =========================================================================
    # Export Functions
    # =========================================================================

    def export_to_json(
        self,
        path: Optional[Path] = None,
        include_summary: bool = True,
    ) -> str:
        """Export metrics history to JSON.

        Args:
            path: Optional file path to write. If None, returns string.
            include_summary: Include summary statistics

        Returns:
            JSON string of metrics
        """
        import json

        data = {
            "exported_at": datetime.now().isoformat(),
            "metrics_count": len(self._metrics_history),
            "metrics": [
                {
                    "start_time": m.start_time,
                    "first_token_time": m.first_token_time,
                    "end_time": m.end_time,
                    "model": m.model,
                    "provider": m.provider,
                    "total_tokens": m.total_tokens,
                    "total_chunks": m.total_chunks,
                    "ttft_ms": m.ttft_ms,
                    "total_duration_ms": m.total_duration_ms,
                    "tokens_per_second": m.tokens_per_second,
                    "has_errors": m.has_errors,
                    "error_count": len(m.errors),
                }
                for m in self._metrics_history
            ],
        }

        if include_summary:
            summary = self.get_summary()
            data["summary"] = {
                "count": summary.count,
                "ttft_ms": summary.ttft_ms,
                "tokens_per_second": summary.tokens_per_second,
                "duration_ms": summary.duration_ms,
                "error_rate": summary.error_rate,
                "total_tokens": summary.total_tokens,
                "total_tool_calls": summary.total_tool_calls,
            }

        json_str = json.dumps(data, indent=2, default=str)

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str)
            logger.info(f"Exported {len(self._metrics_history)} metrics to {path}")

        return json_str

    def export_to_csv(
        self,
        path: Optional[Path] = None,
    ) -> str:
        """Export metrics history to CSV.

        Args:
            path: Optional file path to write. If None, returns string.

        Returns:
            CSV string of metrics
        """
        import csv
        import io

        output = io.StringIO()
        fieldnames = [
            "timestamp",
            "model",
            "provider",
            "ttft_ms",
            "duration_ms",
            "tokens_per_second",
            "total_tokens",
            "total_chunks",
            "has_errors",
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for m in self._metrics_history:
            writer.writerow(
                {
                    "timestamp": m.start_time,
                    "model": m.model or "",
                    "provider": m.provider or "",
                    "ttft_ms": f"{m.ttft_ms:.2f}" if m.ttft_ms else "",
                    "duration_ms": (f"{m.total_duration_ms:.2f}" if m.total_duration_ms else ""),
                    "tokens_per_second": (
                        f"{m.tokens_per_second:.2f}" if m.tokens_per_second else ""
                    ),
                    "total_tokens": m.total_tokens,
                    "total_chunks": m.total_chunks,
                    "has_errors": m.has_errors,
                }
            )

        csv_str = output.getvalue()

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(csv_str)
            logger.info(f"Exported {len(self._metrics_history)} metrics to {path}")

        return csv_str

    def generate_report(self, provider: Optional[str] = None) -> str:
        """Generate a text report of streaming metrics.

        Args:
            provider: Filter by provider name

        Returns:
            Formatted text report
        """
        summary = self.get_summary(provider=provider)
        recent = self.get_recent_metrics(count=5, provider=provider)

        lines = [
            "=" * 60,
            "STREAMING METRICS REPORT",
            "=" * 60,
            "",
            f"Total Requests: {summary.count}",
            f"Total Tokens: {summary.total_tokens:,}",
            f"Total Tool Calls: {summary.total_tool_calls}",
            f"Error Rate: {summary.error_rate:.1%}",
            "",
            "--- Time to First Token (TTFT) ---",
        ]

        if summary.ttft_ms.get("avg"):
            lines.extend(
                [
                    f"  Average: {summary.ttft_ms['avg']:.0f}ms",
                    f"  Min: {summary.ttft_ms['min']:.0f}ms",
                    f"  Max: {summary.ttft_ms['max']:.0f}ms",
                    f"  P50: {summary.ttft_ms['p50']:.0f}ms",
                    (
                        f"  P95: {summary.ttft_ms['p95']:.0f}ms"
                        if summary.ttft_ms.get("p95")
                        else ""
                    ),
                ]
            )
        else:
            lines.append("  No data available")

        lines.extend(
            [
                "",
                "--- Tokens Per Second ---",
            ]
        )

        if summary.tokens_per_second.get("avg"):
            lines.extend(
                [
                    f"  Average: {summary.tokens_per_second['avg']:.1f} tok/s",
                    f"  Min: {summary.tokens_per_second['min']:.1f} tok/s",
                    f"  Max: {summary.tokens_per_second['max']:.1f} tok/s",
                ]
            )
        else:
            lines.append("  No data available")

        lines.extend(
            [
                "",
                "--- Recent Requests ---",
            ]
        )

        for i, m in enumerate(recent, 1):
            lines.append(
                f"  {i}. {m.model or 'unknown'} - "
                f"TTFT: {m.ttft_ms:.0f}ms, "
                f"Duration: {m.total_duration_ms:.0f}ms, "
                f"Tokens: {m.total_tokens}"
                if m.ttft_ms
                else f"  {i}. {m.model or 'unknown'} - No timing data"
            )

        lines.extend(["", "=" * 60])

        return "\n".join(lines)

    async def _export_metrics(self, metrics: StreamMetrics):
        """Export metrics to file.

        Args:
            metrics: Metrics to export
        """
        if not self.export_path:
            return

        try:
            self.export_path.parent.mkdir(parents=True, exist_ok=True)

            # Append to JSONL file
            with open(self.export_path, "a") as f:
                f.write(metrics.to_json() + "\n")

        except Exception as e:
            logger.warning(f"Failed to export metrics: {e}")

    @staticmethod
    def _percentile(values: List[float], p: float) -> Optional[float]:
        """Calculate percentile.

        Args:
            values: List of values
            p: Percentile (0.0 to 1.0)

        Returns:
            Percentile value or None if insufficient data
        """
        if not values or len(values) < 2:
            return None
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p)
        return sorted_values[min(idx, len(sorted_values) - 1)]


class MetricsStreamWrapper:
    """
    Wraps a streaming response to collect metrics.

    Usage:
        metrics = collector.create_metrics("req_123", "claude-3-5-haiku", "anthropic")

        async for chunk in MetricsStreamWrapper(stream, metrics, collector):
            # Process chunk
            pass
    """

    def __init__(
        self,
        stream: AsyncIterator[T],
        metrics: StreamMetrics,
        collector: Optional[StreamingMetricsCollector] = None,
        token_estimator: Optional[Callable[[Any], int]] = None,
    ):
        """Initialize the wrapper.

        Args:
            stream: Async iterator to wrap
            metrics: StreamMetrics instance to populate
            collector: Optional collector to record completed metrics
            token_estimator: Function to estimate tokens from chunk content
        """
        self.stream = stream
        self.metrics = metrics
        self.collector = collector
        self.token_estimator = token_estimator or self._default_token_estimator
        self._last_chunk_time: Optional[float] = None
        self._iterator: Optional[AsyncIterator[T]] = None

    def __aiter__(self) -> "MetricsStreamWrapper":
        return self

    async def aclose(self) -> None:
        """Propagate close to the wrapped stream.

        Without this, closing the wrapper (e.g. via contextlib.aclosing on an early break)
        would not finalize the underlying provider/httpx generator, leaving it to GC-driven
        off-task cleanup ("async generator ignored GeneratorExit" / cross-task cancel scope).
        """
        inner_close = getattr(self.stream, "aclose", None)
        if callable(inner_close):
            await inner_close()

    async def __anext__(self) -> T:
        try:
            chunk = await self.stream.__anext__()

            current_time = time.time()

            # Record first token time
            if self.metrics.first_token_time is None:
                self.metrics.first_token_time = current_time

            # Record chunk interval
            if self._last_chunk_time is not None:
                interval = current_time - self._last_chunk_time
                self.metrics.chunk_intervals.append(interval)
            self._last_chunk_time = current_time

            # Update metrics
            self.metrics.total_chunks += 1
            self.metrics.last_token_time = current_time

            # Count tokens from content
            content = self._extract_content(chunk)
            if content:
                tokens = self.token_estimator(content)
                self.metrics.content_tokens += tokens

            # Track tool calls
            tool_calls = self._extract_tool_calls(chunk)
            if tool_calls:
                self.metrics.tool_calls_count += len(tool_calls)
                self.metrics.tool_call_times.append(current_time)

            return chunk

        except StopAsyncIteration:
            # Stream complete - finalize metrics
            self.metrics.total_tokens = self.metrics.content_tokens

            # Record with collector
            if self.collector:
                self.collector.record_metrics_sync(self.metrics)

            raise

        except Exception as e:
            # Record error
            self.metrics.errors.append(str(e))

            # Still record metrics even on error
            if self.collector:
                self.collector.record_metrics_sync(self.metrics)

            raise

    def _extract_content(self, chunk: Any) -> Optional[str]:
        """Extract content from chunk."""
        if hasattr(chunk, "content"):
            return chunk.content
        if hasattr(chunk, "delta") and hasattr(chunk.delta, "content"):
            return chunk.delta.content
        if isinstance(chunk, dict):
            return chunk.get("content") or chunk.get("delta", {}).get("content")
        return None

    def _extract_tool_calls(self, chunk: Any) -> Optional[List[Any]]:
        """Extract tool calls from chunk."""
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            return chunk.tool_calls
        if isinstance(chunk, dict):
            return chunk.get("tool_calls")
        return None

    @staticmethod
    def _default_token_estimator(content: str) -> int:
        """Estimate tokens from content (4 chars per token)."""
        if not content:
            return 0
        return len(content) // 4 + 1


# Singleton collector instance
_metrics_collector: Optional[StreamingMetricsCollector] = None


def get_metrics_collector(
    max_history: int = 1000,
    export_path: Optional[Path] = None,
) -> StreamingMetricsCollector:
    """Get the global metrics collector.

    Args:
        max_history: Maximum metrics to keep in history
        export_path: Optional path for metrics export

    Returns:
        Global StreamingMetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = StreamingMetricsCollector(
            max_history=max_history,
            export_path=export_path,
        )
    return _metrics_collector


def reset_metrics_collector():
    """Reset the global metrics collector."""
    global _metrics_collector
    _metrics_collector = None
