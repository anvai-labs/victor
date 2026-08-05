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

"""Performance profiling utilities for graph operations (PH4-008).

This module provides profiling and optimization tools for graph operations:
- Performance timing for graph traversal
- Hot path identification
- Memory usage tracking
- Optimization recommendations

Usage:
    from victor.processing.graph_profiler import profile_graph_operation, GraphProfiler

    profiler = GraphProfiler()

    @profile_graph_operation(profiler)
    async def my_traversal():
        # Your graph traversal code
        pass

    # Get performance report
    report = profiler.get_report()
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class OperationMetrics:
    """Metrics for a single graph operation.

    Attributes:
        name: Operation name
        call_count: Number of times called
        total_time_ms: Total execution time in milliseconds
        avg_time_ms: Average execution time in milliseconds
        min_time_ms: Minimum execution time in milliseconds
        max_time_ms: Maximum execution time in milliseconds
        last_time_ms: Last execution time in milliseconds
        error_count: Number of errors
        memory_bytes: Estimated memory usage in bytes
    """

    name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    last_time_ms: float = 0.0
    error_count: int = 0
    memory_bytes: int = 0

    @property
    def avg_time_ms(self) -> float:
        """Average execution time in milliseconds."""
        return self.total_time_ms / self.call_count if self.call_count > 0 else 0.0


@dataclass
class ProfileReport:
    """Performance profile report for graph operations.

    Attributes:
        operations: Metrics for each operation
        hot_paths: List of operation names ordered by total time
        recommendations: Optimization recommendations
        total_time_ms: Total execution time across all operations
    """

    operations: Dict[str, OperationMetrics] = field(default_factory=dict)
    hot_paths: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    total_time_ms: float = 0.0

    def get_hot_paths(self, top_n: int = 10) -> List[tuple[str, float]]:
        """Get top N operations by total time.

        Args:
            top_n: Number of top operations to return

        Returns:
            List of (operation_name, total_time_ms) tuples
        """
        return sorted(
            [(name, metrics.total_time_ms) for name, metrics in self.operations.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

    def get_slowest_operations(self, top_n: int = 5) -> List[tuple[str, float]]:
        """Get top N operations by average time.

        Args:
            top_n: Number of top operations to return

        Returns:
            List of (operation_name, avg_time_ms) tuples
        """
        return sorted(
            [(name, metrics.avg_time_ms) for name, metrics in self.operations.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

    def get_most_frequent(self, top_n: int = 5) -> List[tuple[str, int]]:
        """Get top N most frequently called operations.

        Args:
            top_n: Number of top operations to return

        Returns:
            List of (operation_name, call_count) tuples
        """
        return sorted(
            [(name, metrics.call_count) for name, metrics in self.operations.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]


class GraphProfiler:
    """Profiler for graph operations (PH4-008).

    Tracks execution time, call frequency, and memory usage for graph
    operations to identify hot paths and optimization opportunities.

    Attributes:
        enabled: Whether profiling is enabled
        track_memory: Whether to track memory usage
        operations: Dict of operation metrics
        call_stack: Current call stack for nested profiling
    """

    def __init__(
        self,
        enabled: bool = True,
        track_memory: bool = False,
    ) -> None:
        """Initialize the profiler.

        Args:
            enabled: Whether profiling is enabled
            track_memory: Whether to track memory usage (can be expensive)
        """
        self._enabled = enabled
        self._track_memory = track_memory
        self._operations: Dict[str, OperationMetrics] = {}
        self._call_stack: List[str] = []

    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self._enabled

    @contextlib.contextmanager
    def profile_operation(
        self,
        operation_name: str,
    ):
        """Context manager for profiling an operation.

        Args:
            operation_name: Name of the operation

        Yields:
            None

        Example:
            profiler = GraphProfiler()

            with profiler.profile_operation("graph_traversal"):
                result = await graph_store.get_neighbors(node_id)
        """
        if not self._enabled:
            yield
            return

        # Get or create metrics
        metrics = self._operations.setdefault(
            operation_name,
            OperationMetrics(name=operation_name),
        )

        # Track call stack
        self._call_stack.append(operation_name)

        # Start timing
        start_time = time.perf_counter()

        # Get start memory if tracking
        start_memory = 0
        if self._track_memory:
            import tracemalloc

            if not tracemalloc.is_tracing():
                tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]

        try:
            yield
        finally:
            # End timing
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000

            # Update metrics
            metrics.call_count += 1
            metrics.total_time_ms += elapsed_ms
            metrics.last_time_ms = elapsed_ms
            metrics.min_time_ms = min(metrics.min_time_ms, elapsed_ms)
            metrics.max_time_ms = max(metrics.max_time_ms, elapsed_ms)

            # Track memory if enabled
            if self._track_memory:
                import tracemalloc

                current_memory = tracemalloc.get_traced_memory()[0]
                metrics.memory_bytes = max(metrics.memory_bytes, current_memory - start_memory)

            # Pop call stack
            if self._call_stack:
                self._call_stack.pop()

    def record_call(
        self,
        operation_name: str,
        duration_ms: float,
        error: bool = False,
    ) -> None:
        """Record a manual operation call.

        Args:
            operation_name: Name of the operation
            duration_ms: Execution time in milliseconds
            error: Whether the operation resulted in an error
        """
        if not self._enabled:
            return

        metrics = self._operations.setdefault(
            operation_name,
            OperationMetrics(name=operation_name),
        )

        metrics.call_count += 1
        metrics.total_time_ms += duration_ms
        metrics.last_time_ms = duration_ms
        metrics.min_time_ms = min(metrics.min_time_ms, duration_ms)
        metrics.max_time_ms = max(metrics.max_time_ms, duration_ms)

        if error:
            metrics.error_count += 1

    def get_metrics(self, operation_name: str) -> Optional[OperationMetrics]:
        """Get metrics for a specific operation.

        Args:
            operation_name: Name of the operation

        Returns:
            OperationMetrics or None if not found
        """
        return self._operations.get(operation_name)

    def get_report(self) -> ProfileReport:
        """Generate a performance profile report.

        Returns:
            ProfileReport with all metrics and recommendations
        """
        report = ProfileReport(operations=self._operations.copy())

        # Calculate total time
        report.total_time_ms = sum(m.total_time_ms for m in self._operations.values())

        # Identify hot paths (operations with most total time)
        hot_paths = report.get_hot_paths()
        report.hot_paths = [name for name, _ in hot_paths]

        # Generate recommendations
        report.recommendations = self._generate_recommendations()

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if not self._operations:
            return recommendations

        # Find frequently called slow operations
        for op_name, metrics in self._operations.items():
            if metrics.call_count >= 10 and metrics.avg_time_ms > 100:
                recommendations.append(
                    f"High-frequency slow operation: {op_name} "
                    f"({metrics.call_count} calls, {metrics.avg_time_ms:.1f}ms avg) "
                    f"- consider caching or batching"
                )

        # Find operations with high variance
        for op_name, metrics in self._operations.items():
            if metrics.call_count >= 5:
                variance = metrics.max_time_ms - metrics.min_time_ms
                if variance > 100:
                    recommendations.append(
                        f"High variance in {op_name}: "
                        f"{metrics.min_time_ms:.1f}ms - {metrics.max_time_ms:.1f}ms "
                        f"({variance:.1f}ms range) - check for locking issues"
                    )

        # Find error-prone operations
        for op_name, metrics in self._operations.items():
            error_rate = metrics.error_count / metrics.call_count if metrics.call_count > 0 else 0
            if error_rate > 0.1:
                recommendations.append(
                    f"High error rate in {op_name}: "
                    f"{metrics.error_count}/{metrics.call_count} errors "
                    f"({error_rate:.1%}) - add retry logic or validation"
                )

        # Sort by priority (total time impact)
        recommendations.sort(key=lambda r: r.split(":")[0], reverse=False)

        return recommendations

    def reset(self) -> None:
        """Reset all profiling data."""
        self._operations.clear()
        self._call_stack.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get profiler statistics.

        Returns:
            Dictionary with profiler stats
        """
        total_calls = sum(m.call_count for m in self._operations.values())
        total_time = sum(m.total_time_ms for m in self._operations.values())

        return {
            "enabled": self._enabled,
            "track_memory": self._track_memory,
            "operation_count": len(self._operations),
            "total_calls": total_calls,
            "total_time_ms": total_time,
            "avg_time_ms": total_time / total_calls if total_calls > 0 else 0,
        }


# Global profiler instance
_global_profiler: Optional[GraphProfiler] = None


def get_graph_profiler() -> GraphProfiler:
    """Get the global graph profiler singleton.

    Returns:
        Global GraphProfiler instance
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = GraphProfiler()
    return _global_profiler


def reset_graph_profiler() -> None:
    """Reset the global graph profiler."""
    global _global_profiler
    _global_profiler = None


def configure_graph_profiler(
    enabled: bool = True,
    track_memory: bool = False,
) -> GraphProfiler:
    """Configure the global graph profiler.

    Args:
        enabled: Whether profiling is enabled
        track_memory: Whether to track memory usage

    Returns:
        Configured GraphProfiler instance
    """
    global _global_profiler
    _global_profiler = GraphProfiler(
        enabled=enabled,
        track_memory=track_memory,
    )
    return _global_profiler


@dataclass
class ProfilingConfig:
    """Configuration for graph profiling (PH4-008).

    Attributes:
        enabled: Whether profiling is enabled
        track_memory: Whether to track memory usage
        report_threshold_ms: Minimum time to include in reports
        sample_rate: Sampling rate (1.0 = all, 0.1 = 10%)
        max_tracked_operations: Maximum number of operations to track
    """

    enabled: bool = False
    track_memory: bool = False
    report_threshold_ms: float = 10.0
    sample_rate: float = 1.0
    max_tracked_operations: int = 100


def profile_graph_operation(
    profiler: Optional[GraphProfiler] = None,
    operation_name: Optional[str] = None,
):
    """Decorator for profiling graph operations.

    Args:
        profiler: GraphProfiler instance (uses global if None)
        operation_name: Operation name (uses function name if None)

    Example:
        profiler = GraphProfiler()

        @profile_graph_operation(profiler, "get_neighbors")
        async def get_neighbors(node_id: str):
            return await graph_store.get_neighbors(node_id)
    """
    from functools import wraps

    if profiler is None:
        profiler = get_graph_profiler()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Use function name if operation name not provided
        op_name = operation_name or func.__name__

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            with profiler.profile_operation(op_name):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            with profiler.profile_operation(op_name):
                return func(*args, **kwargs)

        # Return appropriate wrapper based on whether function is async
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


__all__ = [
    "GraphProfiler",
    "OperationMetrics",
    "ProfileReport",
    "ProfilingConfig",
    "get_graph_profiler",
    "reset_graph_profiler",
    "configure_graph_profiler",
    "profile_graph_operation",
]
