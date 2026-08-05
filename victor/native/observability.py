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

"""Observability hooks for native accelerations.

Provides metrics collection, tracing, and instrumentation for all
native (Rust) and fallback (Python) operations.

Features:
- Histogram-based latency tracking with percentiles
- Counter-based usage tracking (Rust vs Python calls)
- EventBus integration for async event publishing
- OpenTelemetry span creation for distributed tracing
- Instrumented wrapper for automatic metrics collection

Usage:
    from victor.native.observability import instrumented_call, NativeMetrics

    # Automatic instrumentation
    result = instrumented_call(
        operation="symbol_extraction",
        native_fn=rust_extract,
        fallback_fn=python_extract,
        source_code,
        lang="python",
    )

    # Manual metrics access
    metrics = NativeMetrics.get_instance()
    print(f"Rust calls: {metrics.rust_calls.value}")
    print(f"Avg latency: {metrics.extraction_latency.mean():.2f}ms")
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic instrumented calls
T = TypeVar("T")


# -----------------------------------------------------------------------------
# Adaptive-dispatch tunables (Gap 3: close the measurement→dispatch loop).
#
# Per-backend EWMA latency is maintained on OperationStats and consumed by
# accelerator.get_preferred_backend() to pick the empirically faster backend
# once enough samples exist. These defaults can be overridden via
# set_adaptive_ewma_alpha() / set_adaptive_min_samples() (wired from settings
# in victor.processing.native.accelerator).
# -----------------------------------------------------------------------------
_EWMA_ALPHA: float = 0.3
_DEFAULT_MIN_SAMPLES: int = 20


def set_adaptive_ewma_alpha(alpha: float) -> None:
    """Set the EWMA smoothing factor used for adaptive dispatch (clamped 0.01–1.0)."""
    global _EWMA_ALPHA
    _EWMA_ALPHA = max(0.01, min(1.0, float(alpha)))


def set_adaptive_min_samples(min_samples: int) -> None:
    """Set the minimum per-backend samples before adaptive dispatch kicks in."""
    global _DEFAULT_MIN_SAMPLES
    _DEFAULT_MIN_SAMPLES = max(1, int(min_samples))


def _ewma_update(current: float, observed: float, alpha: float, sample_count: int) -> float:
    """Blend a new observation into the EWMA. The first observation seeds it."""
    if sample_count <= 1:
        return observed
    return alpha * observed + (1.0 - alpha) * current


@dataclass
class OperationStats:
    """Statistics for a single operation type.

    Thread-safe accumulator for operation metrics. In addition to aggregate
    counters, it tracks per-backend EWMA latency so adaptive dispatch can pick
    the empirically faster backend (Gap 3). EWMA is only updated by observers
    that KNOW which backend ran (InstrumentedAccelerator._timed_call);
    best-effort observers pass ``update_ewma=False`` to avoid corrupting it.
    """

    calls: int = 0
    duration_ms_total: float = 0.0
    errors: int = 0
    rust_calls: int = 0
    python_calls: int = 0
    # Per-backend EWMA latency (ms) + sample counts for adaptive dispatch.
    rust_ewma_ms: float = 0.0
    python_ewma_ms: float = 0.0
    rust_ewma_samples: int = 0
    python_ewma_samples: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        duration_ms: float,
        used_rust: bool,
        error: bool = False,
        *,
        update_ewma: bool = True,
        ewma_alpha: Optional[float] = None,
    ) -> None:
        """Record an operation execution.

        Args:
            duration_ms: Duration of the call in milliseconds.
            used_rust: Whether the Rust backend was used.
            error: Whether the call resulted in an error.
            update_ewma: Only observers that know the real backend should update
                the per-backend EWMA (default True for the class-based timed
                path). Best-effort observers pass False to keep EWMA honest.
            ewma_alpha: Override the module EWMA alpha for this sample.
        """
        alpha = _EWMA_ALPHA if ewma_alpha is None else ewma_alpha
        with self._lock:
            self.calls += 1
            self.duration_ms_total += duration_ms
            if error:
                self.errors += 1
            if used_rust:
                self.rust_calls += 1
                if update_ewma:
                    self.rust_ewma_samples += 1
                    self.rust_ewma_ms = _ewma_update(
                        self.rust_ewma_ms, duration_ms, alpha, self.rust_ewma_samples
                    )
            else:
                self.python_calls += 1
                if update_ewma:
                    self.python_ewma_samples += 1
                    self.python_ewma_ms = _ewma_update(
                        self.python_ewma_ms,
                        duration_ms,
                        alpha,
                        self.python_ewma_samples,
                    )

    def preferred_backend(self, min_samples: Optional[int] = None) -> Optional[str]:
        """Adaptive backend hint from observed per-backend EWMA latency.

        Returns ``"rust"`` or ``"python"`` once BOTH backends have at least
        ``min_samples`` EWMA observations (picks the lower-latency one); else
        ``None`` so the caller falls back to the static benchmark prior.

        Args:
            min_samples: Minimum per-backend samples required (default: module
                ``_DEFAULT_MIN_SAMPLES``).
        """
        threshold = _DEFAULT_MIN_SAMPLES if min_samples is None else min_samples
        with self._lock:
            if self.rust_ewma_samples >= threshold and self.python_ewma_samples >= threshold:
                return "rust" if self.rust_ewma_ms <= self.python_ewma_ms else "python"
        return None

    @property
    def duration_ms_avg(self) -> float:
        """Average duration in milliseconds."""
        return self.duration_ms_total / self.calls if self.calls > 0 else 0.0

    @property
    def rust_ratio(self) -> float:
        """Ratio of Rust calls to total calls."""
        return self.rust_calls / self.calls if self.calls > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for metrics export."""
        return {
            "calls_total": float(self.calls),
            "duration_ms_total": self.duration_ms_total,
            "duration_ms_avg": self.duration_ms_avg,
            "errors_total": float(self.errors),
            "rust_calls": float(self.rust_calls),
            "python_calls": float(self.python_calls),
            "rust_ratio": self.rust_ratio,
            "rust_ewma_ms": self.rust_ewma_ms,
            "python_ewma_ms": self.python_ewma_ms,
            "rust_ewma_samples": float(self.rust_ewma_samples),
            "python_ewma_samples": float(self.python_ewma_samples),
        }


class NativeMetrics:
    """Singleton metrics collector for native accelerations.

    Integrates with Victor's observability infrastructure:
    - MetricsRegistry for histogram/counter metrics
    - EventBus for async event publishing
    - OpenTelemetry for distributed tracing

    Example:
        metrics = NativeMetrics.get_instance()
        metrics.record_call("symbol_extraction", 5.2, used_rust=True)
        print(metrics.get_stats())
    """

    _instance: Optional[NativeMetrics] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize metrics (use get_instance() instead)."""
        self._stats: Dict[str, OperationStats] = {}
        self._stats_lock = threading.Lock()

        # Lazy-loaded observability integrations
        self._registry = None
        self._bus = None
        self._tracer = None

        # Pre-defined operations
        self._operations = [
            "symbol_extraction",
            "arg_normalization",
            "similarity_compute",
            "text_chunking",
            "stdlib_check",
            "reference_extraction",
            "json_repair",
            "type_coercion",
        ]

        # Initialize stats for known operations
        for op in self._operations:
            self._stats[op] = OperationStats()

    @classmethod
    def get_instance(cls) -> NativeMetrics:
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _get_registry(self):
        """Lazy-load MetricsRegistry."""
        if self._registry is None:
            try:
                from victor.observability.metrics import MetricsRegistry

                self._registry = MetricsRegistry.get_instance()
            except ImportError:
                logger.debug("MetricsRegistry not available")
        return self._registry

    def _get_bus(self):
        """Lazy-load ObservabilityBus."""
        if self._bus is None:
            try:
                from victor.core.events import get_observability_bus

                self._bus = get_observability_bus()
            except ImportError:
                logger.debug("ObservabilityBus not available (import error)")
            except Exception:
                # ServiceNotFoundError or other errors when bus not registered
                logger.debug("ObservabilityBus not available (service not registered)")
        return self._bus

    def _get_tracer(self):
        """Lazy-load OpenTelemetry tracer."""
        if self._tracer is None:
            try:
                from victor.observability.telemetry import get_tracer

                self._tracer = get_tracer("victor.native")
            except ImportError:
                logger.debug("OpenTelemetry not available")
        return self._tracer

    def record_call(
        self,
        operation: str,
        duration_ms: float,
        used_rust: bool,
        error: bool = False,
        tags: Optional[Dict[str, str]] = None,
        update_ewma: bool = True,
    ) -> None:
        """Record a native operation call.

        Args:
            operation: Operation name (e.g., "symbol_extraction")
            duration_ms: Duration in milliseconds
            used_rust: Whether Rust implementation was used
            error: Whether the call resulted in an error
            tags: Optional additional tags for metrics
            update_ewma: Forwarded to :meth:`OperationStats.record`. Pass False
                from best-effort observers that cannot confirm which backend ran.
        """
        # Update internal stats
        with self._stats_lock:
            if operation not in self._stats:
                self._stats[operation] = OperationStats()
            self._stats[operation].record(duration_ms, used_rust, error, update_ewma=update_ewma)

        # Emit to MetricsRegistry if available
        registry = self._get_registry()
        if registry is not None:
            try:
                # Get or create histogram
                histogram = registry.histogram(
                    f"native_{operation}_duration_ms",
                    f"Duration of native {operation} operation",
                    buckets=(0.1, 0.5, 1, 5, 10, 50, 100, 500),
                )
                histogram.observe(duration_ms)

                # Increment counter
                counter = registry.counter(
                    f"native_{operation}_calls_total",
                    f"Total calls to native {operation}",
                )
                counter.inc()
            except Exception as e:
                logger.debug(f"Failed to record to MetricsRegistry: {e}")

        # Emit to EventBus if available
        bus = self._get_bus()
        if bus is not None:
            try:
                bus.emit_metric(
                    metric_name=f"native_{operation}_duration_ms",
                    value=duration_ms,
                    unit="ms",
                    tags={
                        "backend": "rust" if used_rust else "python",
                        "error": str(error).lower(),
                        **(tags or {}),
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to emit to EventBus: {e}")

    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for operations.

        Args:
            operation: Specific operation, or None for all

        Returns:
            Dictionary of statistics
        """
        with self._stats_lock:
            if operation:
                stats = self._stats.get(operation)
                return stats.to_dict() if stats else {}
            return {op: stats.to_dict() for op, stats in self._stats.items()}

    def get_operation_stats(self, operation: str) -> Optional[OperationStats]:
        """Return the live ``OperationStats`` for ``operation``, or ``None``.

        Used by adaptive dispatch (:func:`accelerator.get_preferred_backend`) to
        read the per-backend EWMA. The returned object is thread-safe (mutated
        under its own lock), so callers may read ``preferred_backend()`` directly.
        """
        with self._stats_lock:
            return self._stats.get(operation)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics across all operations."""
        with self._stats_lock:
            total_calls = sum(s.calls for s in self._stats.values())
            total_duration = sum(s.duration_ms_total for s in self._stats.values())
            total_rust = sum(s.rust_calls for s in self._stats.values())
            total_errors = sum(s.errors for s in self._stats.values())

            return {
                "total_calls": float(total_calls),
                "total_duration_ms": total_duration,
                "avg_duration_ms": total_duration / total_calls if total_calls else 0,
                "rust_ratio": total_rust / total_calls if total_calls else 0,
                "error_rate": total_errors / total_calls if total_calls else 0,
            }


@contextmanager
def traced_native_call(
    operation: str, attributes: Optional[Dict[str, Any]] = None
) -> Generator[Any, None, None]:
    """Create an OpenTelemetry span for a native operation.

    Args:
        operation: Operation name
        attributes: Optional span attributes

    Yields:
        The span object (or None if OTEL not available)

    Example:
        with traced_native_call("symbol_extraction", {"file": "foo.py"}) as span:
            result = extract_symbols(source)
            if span:
                span.set_attribute("symbols_found", len(result))
    """
    metrics = NativeMetrics.get_instance()
    tracer = metrics._get_tracer()

    if tracer is not None:
        try:
            with tracer.start_as_current_span(f"native.{operation}") as span:
                # Set base attributes
                try:
                    from victor.processing.native import is_native_available

                    span.set_attribute(
                        "native.backend", "rust" if is_native_available() else "python"
                    )
                except ImportError:
                    span.set_attribute("native.backend", "unknown")

                # Set custom attributes
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)

                yield span
        except Exception as e:
            logger.debug(f"Tracing error: {e}")
            yield None
    else:
        yield None


def instrumented_call(
    operation: str,
    native_fn: Optional[Callable[..., T]],
    fallback_fn: Callable[..., T],
    *args: Any,
    native_available: Optional[bool] = None,
    **kwargs: Any,
) -> T:
    """Execute a function with automatic metrics collection.

    Chooses between native (Rust) and fallback (Python) implementations
    and records timing and usage metrics.

    Args:
        operation: Operation name for metrics
        native_fn: Rust implementation (or None if not available)
        fallback_fn: Python fallback implementation
        *args: Positional arguments to pass
        native_available: Override native availability check
        **kwargs: Keyword arguments to pass

    Returns:
        Result from the chosen implementation

    Raises:
        Exception: Any exception from the underlying function

    Example:
        result = instrumented_call(
            "json_repair",
            native_fn=victor_native.repair_json if is_native_available() else None,
            fallback_fn=python_repair_json,
            malformed_json,
        )
    """
    metrics = NativeMetrics.get_instance()

    # Determine which implementation to use
    if native_available is None:
        try:
            from victor.processing.native import is_native_available

            native_available = is_native_available()
        except ImportError:
            native_available = False

    use_native = native_available and native_fn is not None

    start = time.perf_counter()
    error = False

    try:
        if use_native:
            result = native_fn(*args, **kwargs)
        else:
            result = fallback_fn(*args, **kwargs)
        return result
    except Exception:
        error = True
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        metrics.record_call(operation, duration_ms, used_rust=use_native, error=error)


class InstrumentedAccelerator:
    """Base class for instrumented accelerator implementations.

    Provides automatic metrics collection for all operations.
    Subclass and implement the protocol methods.

    Example:
        class MyAccelerator(InstrumentedAccelerator):
            def __init__(self):
                super().__init__(backend="python")

            def do_work(self, data):
                with self._timed_call("do_work"):
                    return self._process(data)
    """

    def __init__(self, backend: str = "python"):
        """Initialize with backend identifier.

        Args:
            backend: "rust" or "python"
        """
        self._backend = backend
        self._metrics = NativeMetrics.get_instance()
        self._call_count = 0
        self._total_duration_ms = 0.0
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        """Check if this accelerator is available."""
        return True

    def get_version(self) -> Optional[str]:
        """Get version string."""
        return None

    def get_backend(self) -> str:
        """Get backend identifier."""
        return self._backend

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this accelerator."""
        with self._lock:
            return {
                "calls_total": float(self._call_count),
                "duration_ms_total": self._total_duration_ms,
                "duration_ms_avg": (
                    self._total_duration_ms / self._call_count if self._call_count > 0 else 0.0
                ),
            }

    @contextmanager
    def _timed_call(self, operation: str, **tags: str) -> Generator[None, None, None]:
        """Context manager for timing an operation.

        Args:
            operation: Operation name for metrics
            **tags: Additional metric tags

        Example:
            with self._timed_call("extract_functions", lang="python"):
                result = self._do_extraction(source)
        """
        start = time.perf_counter()
        error = False
        try:
            yield
        except Exception:
            error = True
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000

            with self._lock:
                self._call_count += 1
                self._total_duration_ms += duration_ms

            self._metrics.record_call(
                operation,
                duration_ms,
                used_rust=(self._backend == "rust"),
                error=error,
                tags=tags if tags else None,
            )


# -----------------------------------------------------------------------------
# Canonical metrics sink (Gap 1 unification).
#
# This module is the single source of truth for native-acceleration metrics:
# ``InstrumentedAccelerator._timed_call`` feeds accurate per-backend EWMA here,
# and ``victor.processing.native.observability.DispatchMetrics`` delegates here
# too, so there is ONE stats registry (not two parallel ones). Physical
# relocation of the ``victor/native/`` impl packages is tracked separately
# (118 import sites — FEP-grade); do not move ``InstrumentedAccelerator``.
# -----------------------------------------------------------------------------


def get_operation_stats(operation: str) -> Optional[OperationStats]:
    """Return the live ``OperationStats`` for ``operation`` from the shared sink.

    Convenience accessor over :meth:`NativeMetrics.get_operation_stats`. Returns
    ``None`` when the operation has never been recorded.
    """
    return NativeMetrics.get_instance().get_operation_stats(operation)


def reset_native_metrics() -> None:
    """Drop the singleton metrics instance (test helper)."""
    NativeMetrics.reset_instance()
