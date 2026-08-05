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

"""Performance profiling for agent operations.

This module provides fine-grained timing instrumentation for:
- Tool execution latency (per tool breakdown)
- Provider API latency (time-to-first-token, total response time)
- Internal operations (tool selection, context building, etc.)
- Hierarchical span tracking for nested operations

Design Pattern: Context Manager + Decorator
===========================================
PerformanceProfiler provides both context managers and decorators
for flexible instrumentation with minimal code changes.

Usage:
    profiler = PerformanceProfiler()

    # Context manager for explicit timing
    with profiler.span("tool_execution", tool="read_file") as span:
        result = tool.execute(**args)
        span.add_metadata("file_size", len(result))

    # Decorator for automatic method timing
    @profiler.profile("provider_call")
    async def call_provider(self, messages):
        return await provider.chat(messages)

    # Get timing report
    report = profiler.get_report()
    print(report.to_markdown())

Integration with UsageAnalytics:
================================
PerformanceProfiler complements UsageAnalytics by providing:
- Fine-grained timing within a single request
- Hierarchical span tracking (parent/child relationships)
- Real-time profiling vs. historical analytics

Example span hierarchy:
    chat_turn (2500ms)
    ├── tool_selection (50ms)
    │   ├── semantic_search (30ms)
    │   └── keyword_matching (15ms)
    ├── tool_execution (1800ms)
    │   ├── read_file (200ms)
    │   ├── grep_search (400ms)
    │   └── write_file (1200ms)
    └── provider_call (650ms)
        ├── time_to_first_token (200ms)
        └── streaming_completion (450ms)
"""

from __future__ import annotations

import functools
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class SpanStatus(Enum):
    """Status of a profiling span."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Span:
    """A timed span representing an operation.

    Spans can be nested to create a hierarchy of operations,
    enabling drill-down analysis of where time is spent.

    Attributes:
        name: Human-readable name for this operation
        category: Category for grouping (e.g., "tool", "provider", "internal")
        span_id: Unique identifier for this span
        parent_id: ID of parent span (if nested)
        start_time: Start timestamp (monotonic)
        end_time: End timestamp (monotonic, None if running)
        status: Current status of the span
        metadata: Additional context about the operation
        children: Child span IDs (for hierarchy)
    """

    name: str
    category: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.RUNNING
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds, or None if still running."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def is_complete(self) -> bool:
        """Check if span has completed."""
        return self.status in (SpanStatus.COMPLETED, SpanStatus.FAILED)

    def complete(self, error: Optional[str] = None) -> None:
        """Mark span as completed."""
        self.end_time = time.perf_counter()
        if error:
            self.status = SpanStatus.FAILED
            self.error = error
        else:
            self.status = SpanStatus.COMPLETED

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the span."""
        self.metadata[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "category": self.category,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "metadata": self.metadata,
            "children": self.children,
            "error": self.error,
        }


@dataclass
class ProfileReport:
    """Aggregated profiling report.

    Provides summary statistics and detailed breakdown of timing
    across different operation categories.
    """

    total_duration_ms: float
    span_count: int
    spans_by_category: Dict[str, List[Span]]
    root_spans: List[Span]
    created_at: float = field(default_factory=time.time)

    def get_category_stats(self, category: str) -> Dict[str, float]:
        """Get statistics for a specific category.

        Returns:
            Dict with total_ms, avg_ms, min_ms, max_ms, count
        """
        spans = self.spans_by_category.get(category, [])
        if not spans:
            return {
                "total_ms": 0,
                "avg_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "count": 0,
            }

        durations = [s.duration_ms for s in spans if s.duration_ms is not None]
        if not durations:
            return {
                "total_ms": 0,
                "avg_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "count": len(spans),
            }

        return {
            "total_ms": sum(durations),
            "avg_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "count": len(durations),
        }

    def get_slowest_spans(self, n: int = 5) -> List[Span]:
        """Get the N slowest spans across all categories."""
        all_spans = []
        for spans in self.spans_by_category.values():
            all_spans.extend(spans)

        completed = [s for s in all_spans if s.duration_ms is not None]
        return sorted(completed, key=lambda s: s.duration_ms or 0, reverse=True)[:n]

    def to_markdown(self) -> str:
        """Generate a markdown report."""
        lines = [
            "# Performance Profile Report",
            "",
            f"**Total Duration:** {self.total_duration_ms:.2f}ms",
            f"**Span Count:** {self.span_count}",
            "",
            "## Category Breakdown",
            "",
            "| Category | Count | Total (ms) | Avg (ms) | Min (ms) | Max (ms) |",
            "|----------|-------|------------|----------|----------|----------|",
        ]

        for category in sorted(self.spans_by_category.keys()):
            stats = self.get_category_stats(category)
            lines.append(
                f"| {category} | {stats['count']} | {stats['total_ms']:.2f} | "
                f"{stats['avg_ms']:.2f} | {stats['min_ms']:.2f} | {stats['max_ms']:.2f} |"
            )

        lines.extend(["", "## Slowest Operations", ""])
        for span in self.get_slowest_spans(5):
            metadata_str = ", ".join(f"{k}={v}" for k, v in span.metadata.items())
            lines.append(
                f"- **{span.name}** ({span.category}): {span.duration_ms:.2f}ms"
                + (f" [{metadata_str}]" if metadata_str else "")
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_duration_ms": self.total_duration_ms,
            "span_count": self.span_count,
            "categories": {
                cat: self.get_category_stats(cat) for cat in self.spans_by_category.keys()
            },
            "slowest_spans": [s.to_dict() for s in self.get_slowest_spans(10)],
            "created_at": self.created_at,
        }


class PerformanceProfiler:
    """Hierarchical performance profiler for agent operations.

    Thread-safe profiler that tracks timing spans with parent/child
    relationships, enabling detailed analysis of where time is spent.

    Features:
    - Context manager for explicit span timing
    - Decorator for automatic method timing
    - Hierarchical span tracking
    - Category-based aggregation
    - Markdown and JSON reports

    Example:
        profiler = PerformanceProfiler()

        # Using context manager
        with profiler.span("tool_execution", tool="read_file"):
            result = await tool.execute()

        # Using decorator
        @profiler.profile("provider_call")
        async def call_provider():
            pass

        # Get report
        report = profiler.get_report()
    """

    def __init__(self, enabled: bool = True):
        """Initialize the profiler.

        Args:
            enabled: If False, profiling is disabled (zero overhead)
        """
        self._enabled = enabled
        self._spans: Dict[str, Span] = {}
        self._root_spans: List[str] = []
        self._current_span: threading.local = threading.local()
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None

    @property
    def enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable profiling."""
        self._enabled = True

    def disable(self) -> None:
        """Disable profiling."""
        self._enabled = False

    def reset(self) -> None:
        """Reset all spans and start fresh."""
        with self._lock:
            self._spans.clear()
            self._root_spans.clear()
            self._start_time = None

    def _get_current_span_id(self) -> Optional[str]:
        """Get the current span ID for this thread."""
        return getattr(self._current_span, "span_id", None)

    def _set_current_span_id(self, span_id: Optional[str]) -> None:
        """Set the current span ID for this thread."""
        self._current_span.span_id = span_id

    @contextmanager
    def span(
        self,
        name: str,
        category: str = "default",
        **metadata: Any,
    ) -> Generator[Span, None, None]:
        """Context manager for timing a span.

        Args:
            name: Human-readable name for the operation
            category: Category for grouping (e.g., "tool", "provider")
            **metadata: Additional metadata to attach to the span

        Yields:
            The Span object (can add more metadata during execution)

        Example:
            with profiler.span("read_file", category="tool", path="/foo") as span:
                result = read(path)
                span.add_metadata("bytes_read", len(result))
        """
        if not self._enabled:
            # Return a dummy span that does nothing
            dummy = Span(name=name, category=category, metadata=metadata)
            yield dummy
            return

        # Track start time on first span
        if self._start_time is None:
            self._start_time = time.perf_counter()

        # Create span with parent relationship
        parent_id = self._get_current_span_id()
        span_obj = Span(
            name=name,
            category=category,
            parent_id=parent_id,
            metadata=dict(metadata),
        )

        # Register span
        with self._lock:
            self._spans[span_obj.span_id] = span_obj
            if parent_id is None:
                self._root_spans.append(span_obj.span_id)
            elif parent_id in self._spans:
                self._spans[parent_id].children.append(span_obj.span_id)

        # Set as current span
        old_span_id = self._get_current_span_id()
        self._set_current_span_id(span_obj.span_id)

        try:
            yield span_obj
            span_obj.complete()
        except Exception as e:
            span_obj.complete(error=str(e))
            raise
        finally:
            self._set_current_span_id(old_span_id)

    def profile(
        self,
        name: Optional[str] = None,
        category: str = "default",
    ) -> Callable[[F], F]:
        """Decorator for automatically timing a function.

        Args:
            name: Span name (defaults to function name)
            category: Category for grouping

        Example:
            @profiler.profile(category="provider")
            async def call_api(self, messages):
                return await self.provider.chat(messages)
        """

        def decorator(func: F) -> F:
            span_name = name or func.__name__

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.span(span_name, category=category):
                    return func(*args, **kwargs)

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.span(span_name, category=category):
                    return await func(*args, **kwargs)

            # Return appropriate wrapper based on function type
            import asyncio

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a span by ID."""
        return self._spans.get(span_id)

    def get_active_spans(self) -> List[Span]:
        """Get all currently running spans."""
        return [s for s in self._spans.values() if not s.is_complete]

    def get_spans_by_category(self, category: str) -> List[Span]:
        """Get all spans in a category."""
        return [s for s in self._spans.values() if s.category == category]

    def get_report(self) -> ProfileReport:
        """Generate a profiling report.

        Returns:
            ProfileReport with aggregated statistics
        """
        # Calculate total duration
        if self._start_time is None:
            total_duration = 0.0
        else:
            total_duration = (time.perf_counter() - self._start_time) * 1000

        # Group spans by category
        spans_by_category: Dict[str, List[Span]] = {}
        for span_obj in self._spans.values():
            if span_obj.category not in spans_by_category:
                spans_by_category[span_obj.category] = []
            spans_by_category[span_obj.category].append(span_obj)

        # Get root spans
        root_spans = [self._spans[sid] for sid in self._root_spans if sid in self._spans]

        return ProfileReport(
            total_duration_ms=total_duration,
            span_count=len(self._spans),
            spans_by_category=spans_by_category,
            root_spans=root_spans,
        )

    def log_summary(self, level: int = logging.INFO) -> None:
        """Log a summary of profiling results."""
        if not self._enabled or not self._spans:
            return

        report = self.get_report()
        logger.log(level, "Performance Profile Summary:")
        logger.log(level, f"  Total Duration: {report.total_duration_ms:.2f}ms")
        logger.log(level, f"  Span Count: {report.span_count}")

        for category in sorted(report.spans_by_category.keys()):
            stats = report.get_category_stats(category)
            logger.log(
                level,
                f"  {category}: {stats['count']} ops, "
                f"total={stats['total_ms']:.2f}ms, avg={stats['avg_ms']:.2f}ms",
            )


# Global profiler instance (disabled by default for production)
_global_profiler: Optional[PerformanceProfiler] = None
_global_lock = threading.Lock()


def get_profiler(enabled: bool = True) -> PerformanceProfiler:
    """Get or create the global profiler instance.

    Args:
        enabled: Whether profiling should be enabled

    Returns:
        Global PerformanceProfiler instance
    """
    global _global_profiler
    with _global_lock:
        if _global_profiler is None:
            _global_profiler = PerformanceProfiler(enabled=enabled)
        return _global_profiler


def reset_profiler() -> None:
    """Reset the global profiler."""
    global _global_profiler
    with _global_lock:
        if _global_profiler is not None:
            _global_profiler.reset()


__all__ = [
    "Span",
    "SpanStatus",
    "ProfileReport",
    "PerformanceProfiler",
    "get_profiler",
    "reset_profiler",
]
