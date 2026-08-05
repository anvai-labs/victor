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

"""Tests for sync event emission helpers."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from victor.core.events.emit_helper import (
    EmitSyncMetricsReporter,
    emit_event_sync,
    emit_sync_metrics_event,
    get_emit_sync_stats,
    reset_emit_sync_stats,
    start_emit_sync_metrics_reporter,
    stop_emit_sync_metrics_reporter,
)


@pytest.fixture(autouse=True)
def _reset_emit_metrics():
    """Ensure emit helper metrics are isolated per test."""
    stop_emit_sync_metrics_reporter()
    reset_emit_sync_stats()
    yield
    stop_emit_sync_metrics_reporter()
    reset_emit_sync_stats()


def test_emit_event_sync_skips_without_running_loop_by_default():
    """Without loop and without fallback, sync emit should not schedule emission."""
    emitted = threading.Event()

    class _Bus:
        async def emit(self, topic, data, source):
            emitted.set()

    emit_event_sync(
        _Bus(),
        "test.topic",
        {"value": 1},
        source="test",
    )

    assert not emitted.wait(timeout=0.1)


def test_emit_event_sync_emits_without_running_loop_when_fallback_enabled():
    """Fallback mode should emit from sync contexts without a running loop."""
    emitted = threading.Event()
    payloads = []

    class _Bus:
        async def emit(self, topic, data, source):
            payloads.append((topic, data, source))
            emitted.set()

    emit_event_sync(
        _Bus(),
        "test.topic",
        {"value": 2},
        source="test",
        use_background_loop=True,
    )

    assert emitted.wait(timeout=1.0)
    topic, data, source = payloads[-1]
    assert topic == "test.topic"
    assert data == {"value": 2}
    assert source == "test"


@pytest.mark.asyncio
async def test_emit_event_sync_emits_on_running_loop():
    """Sync helper should schedule task on active loop."""
    payloads = []
    emitted = asyncio.Event()

    class _Bus:
        async def emit(self, topic, data, source):
            payloads.append((topic, data, source))
            emitted.set()

    emit_event_sync(
        _Bus(),
        "test.running",
        {"ok": True},
        source="running",
    )
    await emitted.wait()

    topic, data, source = payloads[-1]
    assert topic == "test.running"
    assert data == {"ok": True}
    assert source == "running"

    # Done-callback metric update can lag one event-loop tick.
    stats = get_emit_sync_stats()
    for _ in range(20):
        if stats["completed"] >= 1:
            break
        await asyncio.sleep(0.01)
        stats = get_emit_sync_stats()

    assert stats["scheduled"] >= 1
    assert stats["completed"] >= 1


def test_emit_event_sync_metrics_track_dropped_without_loop():
    """Metrics should record dropped emits when no loop/fallback is available."""

    class _Bus:
        async def emit(self, topic, data, source):
            return None

    emit_event_sync(
        _Bus(),
        "test.drop",
        {"value": 3},
        source="drop",
    )

    stats = get_emit_sync_stats()
    assert stats["dropped_no_loop"] >= 1


@pytest.mark.asyncio
async def test_emit_event_sync_metrics_track_emit_failures():
    """Metrics should record completion failures from emit task exceptions."""
    done = asyncio.Event()

    class _Bus:
        async def emit(self, topic, data, source):
            try:
                raise RuntimeError("emit failed")
            finally:
                done.set()

    emit_event_sync(
        _Bus(),
        "test.failure",
        {"value": 4},
        source="failure",
    )
    await done.wait()
    await asyncio.sleep(0)

    stats = get_emit_sync_stats()
    assert stats["scheduled"] >= 1
    assert stats["failed"] >= 1


def test_emit_sync_metrics_event_emits_snapshot_without_running_loop():
    """Metrics snapshot emitter should publish stats from sync no-loop contexts."""

    # Create at least one metric value.
    class _NoopBus:
        async def emit(self, topic, data, source):
            return None

    emit_event_sync(
        _NoopBus(),
        "test.drop",
        {"value": 1},
        source="drop",
    )

    emitted = threading.Event()
    payloads = []

    class _MetricsBus:
        async def emit(self, topic, data, source):
            payloads.append((topic, data, source))
            emitted.set()

    snapshot = emit_sync_metrics_event(event_bus=_MetricsBus())

    assert emitted.wait(timeout=1.0)
    topic, data, source = payloads[-1]
    assert topic == "core.events.emit_sync.metrics"
    assert source == "EmitHelper"
    assert data["stats"] == snapshot
    assert data["stats"]["dropped_no_loop"] >= 1


def test_emit_sync_metrics_event_can_reset_after_emit():
    """Metrics snapshot emitter should support reset-after-emit semantics."""

    class _NoopBus:
        async def emit(self, topic, data, source):
            return None

    emit_event_sync(
        _NoopBus(),
        "test.drop",
        {"value": 2},
        source="drop",
    )
    assert get_emit_sync_stats()["dropped_no_loop"] >= 1

    class _MetricsBus:
        async def emit(self, topic, data, source):
            return None

    emit_sync_metrics_event(
        event_bus=_MetricsBus(),
        reset_after_emit=True,
    )

    stats = get_emit_sync_stats()
    assert all(value == 0 for value in stats.values())


def test_emit_event_sync_concurrent_fallback_threads_are_delivered_and_counted():
    """Concurrent fallback scheduling should deliver all events and track metrics."""
    total = 250
    delivered = 0
    delivered_lock = threading.Lock()
    delivered_event = threading.Event()

    class _Bus:
        async def emit(self, topic, data, source):
            nonlocal delivered
            with delivered_lock:
                delivered += 1
                if delivered >= total:
                    delivered_event.set()

    bus = _Bus()

    def _schedule_emit(i: int) -> None:
        emit_event_sync(
            bus,
            "test.concurrent",
            {"i": i},
            source="stress",
            use_background_loop=True,
        )

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(_schedule_emit, i) for i in range(total)]
        for future in futures:
            future.result()

    assert delivered_event.wait(timeout=3.0)

    deadline = time.time() + 3.0
    stats = get_emit_sync_stats()
    while stats["completed"] < total and time.time() < deadline:
        time.sleep(0.01)
        stats = get_emit_sync_stats()

    assert stats["scheduled"] >= total
    assert stats["completed"] >= total
    assert stats["failed"] == 0


def test_emit_sync_metrics_reporter_periodically_emits_and_stops():
    """Reporter should periodically emit metrics events and stop cleanly."""
    emitted_count = 0
    lock = threading.Lock()
    emitted = threading.Event()

    class _Bus:
        async def emit(self, topic, data, source):
            nonlocal emitted_count
            with lock:
                emitted_count += 1
                if emitted_count >= 2:
                    emitted.set()

    bus = _Bus()
    reporter = EmitSyncMetricsReporter(
        interval_seconds=0.02,
        event_bus_provider=lambda: bus,
    )

    reporter.start()
    assert reporter.is_running is True
    assert emitted.wait(timeout=1.0)

    reporter.stop(timeout=1.0)
    assert reporter.is_running is False


def test_start_stop_emit_sync_metrics_reporter_singleton():
    """Singleton start API should return the same running reporter instance."""
    emitted = threading.Event()

    class _Bus:
        async def emit(self, topic, data, source):
            emitted.set()

    bus = _Bus()
    reporter1 = start_emit_sync_metrics_reporter(
        interval_seconds=0.02,
        event_bus_provider=lambda: bus,
    )
    reporter2 = start_emit_sync_metrics_reporter(
        interval_seconds=0.02,
        event_bus_provider=lambda: bus,
    )

    assert reporter1 is reporter2
    assert reporter1.is_running is True
    assert emitted.wait(timeout=1.0)

    stop_emit_sync_metrics_reporter(timeout=1.0)
    assert reporter1.is_running is False
