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

"""Observability decorators for native dispatch functions.

Provides low-overhead timing, logging, and metrics for flat dispatch
functions in victor.processing.native modules.

This is the preferred approach over protocol-based dispatch because:
- Zero abstraction overhead (direct function calls)
- Composable with other decorators
- Can be enabled/disabled globally via settings
- Works seamlessly with existing flat dispatch pattern

Usage:
    from victor.processing.native.observability import dispatch_with_observability

    @dispatch_with_observability("batch_cosine_similarity")
    def batch_cosine_similarity(query, corpus):
        if _NATIVE_AVAILABLE:
            return _native.batch_cosine_similarity(query, corpus)
        # Python fallback...
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Callable, ParamSpec, TypeVar

from victor.config.settings import get_settings
from victor.processing.native._base import is_native_available as _native_is_available

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# Global observability state
_observability_enabled: bool | None = None  # None = not initialized

# Cap on retained per-operation latency samples (prevents unbounded list growth).
_MAX_RETAINED_SAMPLES = 1000


def is_observability_enabled() -> bool:
    """Check if native dispatch observability is enabled.

    Returns False if:
    - Disabled via settings (native.observability_enabled = False)
    - Running in test mode (detected via pytest or unittest)
    - Explicitly disabled via disable_observability()

    Returns:
        True if observability should be added to dispatch functions
    """
    global _observability_enabled

    if _observability_enabled is not None:
        return _observability_enabled

    # Check settings
    try:
        settings = get_settings()
        enabled = getattr(settings.observability, "native_observability_enabled", True)
    except Exception:
        enabled = True  # Default to enabled if settings unavailable

    # Disable in test mode for performance
    import sys

    if "pytest" in sys.modules or "unittest" in sys.modules:
        enabled = False

    _observability_enabled = enabled
    return enabled


def enable_observability() -> None:
    """Enable native dispatch observability."""
    global _observability_enabled
    _observability_enabled = True


def disable_observability() -> None:
    """Disable native dispatch observability (for tests, benchmarks)."""
    global _observability_enabled
    _observability_enabled = False


def dispatch_with_observability(
    operation_name: str,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that adds observability to native dispatch functions.

    Features:
    - Timing measurement (elapsed time in microseconds)
    - Backend tracking (rust/python)
    - Debug logging when enabled
    - Minimal overhead when disabled (<1ns per call)

    Args:
        operation_name: Name of the operation for logging/metrics

    Returns:
        Decorated function with observability hooks

    Example:
        @dispatch_with_observability("batch_cosine_similarity")
        def batch_cosine_similarity(query, corpus):
            if _NATIVE_AVAILABLE:
                return _native.batch_cosine_similarity(query, corpus)
            return _python_fallback(query, corpus)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not is_observability_enabled():
                return func(*args, **kwargs)

            start = time.perf_counter()
            # Best-effort backend attribution for AGGREGATE stats only. The flat
            # dispatch functions branch internally, so we cannot confirm which
            # backend ran per call — hence we forward update_ewma=False so the
            # per-backend EWMA (consumed by adaptive dispatch) stays honest and
            # is populated only by InstrumentedAccelerator._timed_call, which
            # knows its real backend.
            backend = "rust" if _native_is_available() else "python"

            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                elapsed_us = elapsed * 1_000_000  # Convert to microseconds

                # Feed the unified NativeMetrics sink (aggregate counts only).
                try:
                    DispatchMetrics.get_instance().record(operation_name, backend, elapsed_us)
                except Exception:  # pragma: no cover - never let metrics break the call
                    logger.debug("DispatchMetrics.record failed", exc_info=True)

                # Only log slow operations (>1ms) to reduce noise
                if elapsed_us > 1000:
                    logger.debug(
                        "native_dispatch: operation=%s backend=%s elapsed_us=%.0f",
                        operation_name,
                        backend,
                        elapsed_us,
                    )

        return wrapper

    return decorator


class DispatchMetrics:
    """Metrics collector for native dispatch operations.

    Thread-safe singleton for tracking dispatch statistics.
    Can be queried for monitoring and performance analysis.
    """

    _instance: "DispatchMetrics | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "DispatchMetrics":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._reset()
            return cls._instance

    def _reset(self) -> None:
        """Reset all metrics."""
        self._counts: dict[str, int] = {}
        self._times: dict[str, list[float]] = {}
        self._backends: dict[str, dict[str, int]] = {}

    def record(
        self,
        operation: str,
        backend: str,
        elapsed_us: float,
    ) -> None:
        """Record a dispatch operation.

        Also forwards the observation to the unified ``NativeMetrics`` sink so
        there is a single stats registry. The forward uses
        ``update_ewma=False`` because the flat dispatch path cannot confirm
        which backend actually ran; adaptive dispatch's per-backend EWMA is
        populated only by ``InstrumentedAccelerator._timed_call``.

        Args:
            operation: Operation name
            backend: Backend used ("rust" or "python")
            elapsed_us: Elapsed time in microseconds
        """
        self._counts[operation] = self._counts.get(operation, 0) + 1

        if operation not in self._times:
            self._times[operation] = []
        self._times[operation].append(elapsed_us)
        # Bound the retained sample window to avoid unbounded growth.
        if len(self._times[operation]) > _MAX_RETAINED_SAMPLES:
            del self._times[operation][:-_MAX_RETAINED_SAMPLES]

        if operation not in self._backends:
            self._backends[operation] = {}
        self._backends[operation][backend] = self._backends[operation].get(backend, 0) + 1

        # Forward to the canonical NativeMetrics sink (aggregate-only).
        try:
            from victor.native.observability import NativeMetrics

            NativeMetrics.get_instance().record_call(
                operation,
                elapsed_us / 1000.0,
                used_rust=(backend == "rust"),
                update_ewma=False,
            )
        except Exception:  # pragma: no cover - metrics must never break dispatch
            logger.debug("NativeMetrics forward failed", exc_info=True)

    def get_stats(self, operation: str) -> dict[str, Any] | None:
        """Get statistics for a specific operation.

        Args:
            operation: Operation name

        Returns:
            Dict with count, avg_time_us, backend_counts or None if no data
        """
        if operation not in self._counts:
            return None

        times = self._times.get(operation, [])
        avg_time = sum(times) / len(times) if times else 0

        return {
            "operation": operation,
            "count": self._counts[operation],
            "avg_time_us": avg_time,
            "backend_counts": self._backends.get(operation, {}),
        }

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all operations.

        Returns:
            Dict mapping operation names to their stats
        """
        return {op: self.get_stats(op) for op in self._counts if self.get_stats(op) is not None}

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._reset()


import threading
