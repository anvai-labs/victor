# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Bounded event-bridge queues with drop-oldest overflow.

Both the DeliveryEngine event queue and per-client sender queues were
unbounded asyncio.Queues whose QueueFull handlers could never fire
(co-design review U7-F7): one slow consumer grew memory without limit.
Queues are now bounded (MAX_QUEUE_SIZE) with drop-oldest overflow policy —
the newest event survives, drops are counted, and the client is NOT
disconnected.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from victor.integrations.api.event_bridge import DeliveryEngine, MetricsCollector


def _event(name: str):
    import time

    return SimpleNamespace(
        type="test.event",
        timestamp=time.time(),
        to_json=lambda: f'{{"name": "{name}"}}',
    )


class TestEngineQueueBounds:
    async def test_queues_created_with_maxsize(self):
        engine = DeliveryEngine(
            registry=SimpleNamespace(items=lambda: []), metrics=MetricsCollector()
        )
        await engine.start()
        try:
            assert engine._event_queue.maxsize == DeliveryEngine.MAX_QUEUE_SIZE
        finally:
            await engine.stop()

    async def test_broadcast_sync_drops_oldest_on_full(self):
        metrics = MetricsCollector()
        engine = DeliveryEngine(registry=SimpleNamespace(items=lambda: []), metrics=metrics)
        # No start(): we drive the overflow branch directly with a tiny queue.
        engine._event_queue = asyncio.Queue(maxsize=2)

        e1, e2, e3 = _event("e1"), _event("e2"), _event("e3")
        engine.broadcast_sync(e1)
        engine.broadcast_sync(e2)
        engine.broadcast_sync(e3)  # queue full -> e1 dropped

        assert engine._event_queue.qsize() == 2
        assert engine._event_queue.get_nowait() is e2
        assert engine._event_queue.get_nowait() is e3
        assert metrics._queue_drop_count == 1

    async def test_dashboard_exposes_drop_counter(self):
        metrics = MetricsCollector()
        metrics._queue_drop_count = 5
        dashboard = metrics.get_reliability_dashboard()
        assert dashboard["queue_dropped_events"] == 5


class TestClientQueueOverflow:
    async def test_slow_client_drops_oldest_and_stays_registered(self):
        metrics = MetricsCollector()
        engine = DeliveryEngine(registry=SimpleNamespace(items=lambda: []), metrics=metrics)
        engine._loop = asyncio.get_running_loop()

        full_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        full_queue.put_nowait("old-payload")
        client = SimpleNamespace(
            accepts=lambda event: True,
            sender_queue=full_queue,
            sender_task=SimpleNamespace(done=lambda: False),
            consecutive_send_failures=0,
        )
        engine._registry = {"c1": client}  # dict satisfies .get/.items usage

        await engine._send_to_clients(_event("new"))

        assert full_queue.qsize() == 1
        assert full_queue.get_nowait() != "old-payload"  # oldest dropped
        assert metrics._queue_drop_count == 1
        assert "c1" in engine._registry  # client NOT disconnected
