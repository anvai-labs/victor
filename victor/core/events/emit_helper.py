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

"""Helper utilities for emitting events from synchronous code.

This module provides utilities to safely call async event emission
functions from synchronous code without RuntimeWarnings.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_BACKGROUND_LOOP_LOCK = threading.Lock()
_BACKGROUND_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BACKGROUND_LOOP_THREAD: Optional[threading.Thread] = None
_EMIT_STATS_LOCK = threading.Lock()
_EMIT_SYNC_STATS: Dict[str, int] = {
    "scheduled": 0,
    "completed": 0,
    "failed": 0,
    "cancelled": 0,
    "dropped_no_loop": 0,
    "dropped_backend_disconnected": 0,
    "schedule_errors": 0,
}
_REPORTER_LOCK = threading.Lock()
_REPORTER_SINGLETON: Optional["EmitSyncMetricsReporter"] = None


def _increment_emit_stat(name: str) -> None:
    """Increment a sync emit metric counter."""
    with _EMIT_STATS_LOCK:
        _EMIT_SYNC_STATS[name] = _EMIT_SYNC_STATS.get(name, 0) + 1


def get_emit_sync_stats() -> Dict[str, int]:
    """Return snapshot of sync emit scheduling/execution metrics."""
    with _EMIT_STATS_LOCK:
        return dict(_EMIT_SYNC_STATS)


def reset_emit_sync_stats() -> None:
    """Reset sync emit metrics (useful for tests and diagnostics windows)."""
    with _EMIT_STATS_LOCK:
        for key in list(_EMIT_SYNC_STATS.keys()):
            _EMIT_SYNC_STATS[key] = 0


def _ensure_background_emit_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Get or create a background event loop for sync emit fallback."""
    global _BACKGROUND_LOOP, _BACKGROUND_LOOP_THREAD

    with _BACKGROUND_LOOP_LOCK:
        if _BACKGROUND_LOOP and _BACKGROUND_LOOP.is_running():
            return _BACKGROUND_LOOP

        loop_ready = threading.Event()
        loop_holder: Dict[str, asyncio.AbstractEventLoop] = {}

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop_holder["loop"] = loop
            loop_ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=_run_loop,
            name="events-sync-emitter",
            daemon=True,
        )
        thread.start()

        if not loop_ready.wait(timeout=1.0):
            logger.debug("Failed to initialize background event loop for sync emits")
            return None

        loop = loop_holder.get("loop")
        if loop is None:
            logger.debug("Background event loop unavailable for sync emits")
            return None

        _BACKGROUND_LOOP = loop
        _BACKGROUND_LOOP_THREAD = thread
        return loop


def emit_event_sync(
    event_bus,
    topic: str,
    data: Dict[str, Any],
    *,
    source: str = "victor",
    correlation_id: Optional[str] = None,
    use_background_loop: bool = False,
    track_metrics: bool = True,
) -> None:
    """Safely emit an event from synchronous code.

    This function handles the async event emission by creating a task
    on the running event loop and adding a done callback to handle errors.
    If no event loop is running or the backend is not connected, it silently
    skips the emission.

    Args:
        event_bus: The ObservabilityBus instance
        topic: Event topic (e.g., "tool.start", "metric.latency")
        data: Event payload
        source: Event source identifier
        correlation_id: Optional correlation ID
        use_background_loop: If True, emit via a background loop when no
            running loop exists in current thread.
        track_metrics: If True, update internal sync emit counters.

    Example:
        >>> from victor.core.events.emit_helper import emit_event_sync
        >>> emit_event_sync(bus, "tool.start", {"tool": "read_file"})
    """

    async def _emit_and_handle_errors():
        """Emit event and handle any errors."""
        if correlation_id is None:
            await event_bus.emit(
                topic=topic,
                data=data,
                source=source,
            )
        else:
            try:
                await event_bus.emit(
                    topic=topic,
                    data=data,
                    source=source,
                    correlation_id=correlation_id,
                )
            except TypeError:
                # Compatibility for emit() implementations without
                # correlation_id support.
                await event_bus.emit(
                    topic=topic,
                    data=data,
                    source=source,
                )

    def _on_emit_done(done_future: Any) -> None:
        """Track completion state of scheduled emit tasks/futures."""
        if not track_metrics:
            return
        if done_future.cancelled():
            _increment_emit_stat("cancelled")
            return

        try:
            error = done_future.exception()
        except Exception as callback_error:
            _increment_emit_stat("failed")
            logger.debug(f"Failed reading emit task result for {topic}: {callback_error}")
            return

        if error is None:
            _increment_emit_stat("completed")
        else:
            _increment_emit_stat("failed")
            logger.debug(f"Event emission failed for {topic}: {error}")

    try:
        # Check if backend is connected before attempting to emit
        backend = event_bus.backend if hasattr(event_bus, "backend") else None
        if backend and hasattr(backend, "_is_connected") and not backend._is_connected:
            if track_metrics:
                _increment_emit_stat("dropped_backend_disconnected")
            logger.debug(f"Backend not connected, skipping event emission: {topic}")
            return

        loop = asyncio.get_running_loop()
        task = loop.create_task(_emit_and_handle_errors())
        if track_metrics:
            _increment_emit_stat("scheduled")
        task.add_done_callback(_on_emit_done)
    except RuntimeError:
        # No running loop - optionally fallback to background loop.
        if not use_background_loop:
            if track_metrics:
                _increment_emit_stat("dropped_no_loop")
            logger.debug(f"No event loop running, skipping event emission: {topic}")
            return

        loop = _ensure_background_emit_loop()
        if loop is None:
            if track_metrics:
                _increment_emit_stat("dropped_no_loop")
            logger.debug(f"No background loop available, skipping event emission: {topic}")
            return

        try:
            future = asyncio.run_coroutine_threadsafe(_emit_and_handle_errors(), loop)
            if track_metrics:
                _increment_emit_stat("scheduled")
            future.add_done_callback(_on_emit_done)
        except Exception as e:
            if track_metrics:
                _increment_emit_stat("schedule_errors")
            logger.debug(f"Failed to schedule background event emission for {topic}: {e}")


def emit_sync_metrics_event(
    *,
    event_bus: Optional[Any] = None,
    topic: str = "core.events.emit_sync.metrics",
    source: str = "EmitHelper",
    reset_after_emit: bool = False,
    use_background_loop: bool = True,
) -> Dict[str, int]:
    """Emit current sync emitter metrics as an observability event.

    Args:
        event_bus: Optional event bus to use. If omitted, resolves from
            `get_observability_bus()`.
        topic: Event topic for metrics payload.
        source: Event source value.
        reset_after_emit: If True, reset counters after scheduling emission.
        use_background_loop: If True, allow emission from sync no-loop contexts.

    Returns:
        Snapshot of metrics that was scheduled for emission.
    """
    stats_snapshot = get_emit_sync_stats()
    payload = {"stats": stats_snapshot}

    bus = event_bus
    if bus is None:
        try:
            from victor.core.events import get_observability_bus

            bus = get_observability_bus()
        except Exception as e:
            logger.debug(f"Failed to resolve observability bus for sync metrics event: {e}")
            bus = None

    if bus is not None:
        emit_event_sync(
            bus,
            topic,
            payload,
            source=source,
            use_background_loop=use_background_loop,
            track_metrics=False,
        )

    if reset_after_emit:
        reset_emit_sync_stats()

    return stats_snapshot


class EmitSyncMetricsReporter:
    """Periodic publisher for sync emitter metrics snapshots."""

    def __init__(
        self,
        *,
        interval_seconds: float = 60.0,
        topic: str = "core.events.emit_sync.metrics",
        source: str = "EmitSyncMetricsReporter",
        reset_after_emit: bool = False,
        use_background_loop: bool = True,
        event_bus_provider: Optional[Callable[[], Any]] = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")

        self._interval_seconds = interval_seconds
        self._topic = topic
        self._source = source
        self._reset_after_emit = reset_after_emit
        self._use_background_loop = use_background_loop
        self._event_bus_provider = event_bus_provider
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        """Return True when reporter loop thread is active."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start periodic emission loop if not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="emit-sync-metrics-reporter",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop periodic emission loop."""
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()

        thread.join(timeout=timeout)
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def _run_loop(self) -> None:
        """Loop that periodically emits metrics snapshot events."""
        while not self._stop_event.wait(self._interval_seconds):
            try:
                bus = self._resolve_event_bus()
                emit_sync_metrics_event(
                    event_bus=bus,
                    topic=self._topic,
                    source=self._source,
                    reset_after_emit=self._reset_after_emit,
                    use_background_loop=self._use_background_loop,
                )
            except Exception as e:
                logger.debug(f"EmitSyncMetricsReporter loop error: {e}")

    def _resolve_event_bus(self) -> Optional[Any]:
        """Resolve event bus from provider when configured."""
        if self._event_bus_provider is None:
            return None
        try:
            return self._event_bus_provider()
        except Exception as e:
            logger.debug(f"EmitSyncMetricsReporter bus provider failed: {e}")
            return None


def start_emit_sync_metrics_reporter(
    *,
    interval_seconds: float = 60.0,
    topic: str = "core.events.emit_sync.metrics",
    source: str = "EmitSyncMetricsReporter",
    reset_after_emit: bool = False,
    use_background_loop: bool = True,
    event_bus_provider: Optional[Callable[[], Any]] = None,
) -> EmitSyncMetricsReporter:
    """Start singleton periodic reporter for sync emit metrics."""
    global _REPORTER_SINGLETON

    with _REPORTER_LOCK:
        if _REPORTER_SINGLETON is None:
            _REPORTER_SINGLETON = EmitSyncMetricsReporter(
                interval_seconds=interval_seconds,
                topic=topic,
                source=source,
                reset_after_emit=reset_after_emit,
                use_background_loop=use_background_loop,
                event_bus_provider=event_bus_provider,
            )

        _REPORTER_SINGLETON.start()
        return _REPORTER_SINGLETON


def stop_emit_sync_metrics_reporter(*, timeout: float = 2.0) -> None:
    """Stop and clear singleton periodic reporter for sync emit metrics."""
    global _REPORTER_SINGLETON

    with _REPORTER_LOCK:
        reporter = _REPORTER_SINGLETON
        _REPORTER_SINGLETON = None

    if reporter is not None:
        reporter.stop(timeout=timeout)
