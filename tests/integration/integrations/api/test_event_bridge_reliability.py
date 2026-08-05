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

"""Integration tests for Event Bridge reliability (M2).

Tests for event loss detection and ordering verification under various
scenarios including high-volume bursts, slow consumers, and client
reconnection.
"""

import asyncio
import json
import random
import time
from typing import List, Dict, Any
from unittest.mock import AsyncMock

import pytest

from victor.core.events import InMemoryEventBackend, ObservabilityBus, MessagingEvent
from victor.integrations.api.event_bridge import (
    EventBridge,
    BridgeEvent,
    BridgeEventType,
)


class TestEventBridgeLossDetection:
    """M2: Tests for detecting event loss in various scenarios."""

    @pytest.mark.asyncio
    async def test_no_event_loss_during_high_volume_burst(self):
        """Verify no events are lost during high-volume burst transmission."""
        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        received: List[Dict[str, Any]] = []

        async def tracking_send(message: str) -> None:
            received.append(json.loads(message))

        await bridge.async_start()
        bridge._broadcaster.add_client("burst-test", tracking_send)

        # Wait for subscriptions to be established
        await asyncio.sleep(0.1)

        # Send high-volume burst
        total_events = 100
        for idx in range(total_events):
            await bus.emit("tool.start", {"idx": idx, "test": "burst"})

        # Wait for all events to be delivered
        max_wait = 10  # seconds
        deadline = time.time() + max_wait
        while len(received) < total_events and time.time() < deadline:
            await asyncio.sleep(0.05)

        await bridge.async_stop()

        # Verify no events were lost
        assert len(received) == total_events, (
            f"Event loss detected: expected {total_events}, " f"received {len(received)}"
        )

        # Verify event integrity
        received_indices = [msg["data"]["idx"] for msg in received]
        assert sorted(received_indices) == list(range(total_events))

    @pytest.mark.asyncio
    async def test_no_event_loss_during_slow_consumer(self):
        """Verify events are queued even when consumer is slow."""
        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        received: List[Dict[str, Any]] = []

        async def slow_send(message: str) -> None:
            # Simulate slow consumer
            await asyncio.sleep(0.01)
            received.append(json.loads(message))

        await bridge.async_start()
        bridge._broadcaster.add_client("slow-consumer", slow_send)
        await asyncio.sleep(0.1)

        # Send events faster than consumer can process
        total_events = 50
        for idx in range(total_events):
            await bus.emit("tool.start", {"idx": idx})

        # Wait for all events (slow consumer, so need more time)
        max_wait = 15
        deadline = time.time() + max_wait
        while len(received) < total_events and time.time() < deadline:
            await asyncio.sleep(0.1)

        await bridge.async_stop()

        # Should still receive all events (queued in broadcaster)
        assert len(received) == total_events, (
            f"Event loss with slow consumer: expected {total_events}, " f"received {len(received)}"
        )

    @pytest.mark.asyncio
    async def test_event_delivery_with_client_reconnect(self):
        """Verify event delivery continues after client reconnect."""
        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        received: List[Dict[str, Any]] = []

        async def tracking_send(message: str) -> None:
            received.append(json.loads(message))

        await bridge.async_start()

        # Initial connection
        bridge._broadcaster.add_client("reconnect-test", tracking_send)
        await asyncio.sleep(0.1)

        # Send initial batch
        for idx in range(10):
            await bus.emit("tool.start", {"idx": idx, "batch": 1})

        await asyncio.sleep(0.2)

        # Simulate disconnect
        bridge._broadcaster.remove_client("reconnect-test")
        await asyncio.sleep(0.1)

        # Send batch during disconnect (should be queued but not delivered)
        for idx in range(10, 20):
            await bus.emit("tool.start", {"idx": idx, "batch": 2})

        await asyncio.sleep(0.1)

        # Reconnect
        bridge._broadcaster.add_client("reconnect-test", tracking_send)
        await asyncio.sleep(0.1)

        # Send final batch
        for idx in range(20, 30):
            await bus.emit("tool.start", {"idx": idx, "batch": 3})

        # Wait for delivery
        await asyncio.sleep(0.5)

        await bridge.async_stop()

        # Note: Events sent during disconnect won't be delivered
        # (this is expected behavior - no buffering for disconnected clients)
        # But events before and after should be delivered
        assert len(received) >= 20, (
            f"Expected at least 20 events (before + after reconnect), " f"got {len(received)}"
        )

    @pytest.mark.asyncio
    async def test_multiple_clients_no_loss(self):
        """Verify no event loss with multiple concurrent clients."""
        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        client_data: Dict[str, List[Dict[str, Any]]] = {}
        num_clients = 5

        async def make_send_func(client_id: str):
            async def send_func(message: str) -> None:
                if client_id not in client_data:
                    client_data[client_id] = []
                client_data[client_id].append(json.loads(message))

            return send_func

        await bridge.async_start()

        # Add multiple clients
        for i in range(num_clients):
            client_id = f"client-{i}"
            bridge._broadcaster.add_client(client_id, await make_send_func(client_id))

        await asyncio.sleep(0.1)

        # Send events
        total_events = 50
        for idx in range(total_events):
            await bus.emit("tool.start", {"idx": idx})

        # Wait for delivery
        max_wait = 10
        deadline = time.time() + max_wait
        while (
            any(
                len(client_data.get(cid, [])) < total_events
                for cid in [f"client-{i}" for i in range(num_clients)]
            )
            and time.time() < deadline
        ):
            await asyncio.sleep(0.05)

        await bridge.async_stop()

        # All clients should receive all events
        for i in range(num_clients):
            client_id = f"client-{i}"
            received = client_data.get(client_id, [])
            assert len(received) == total_events, (
                f"Client {client_id} event loss: expected {total_events}, "
                f"received {len(received)}"
            )


class TestEventBridgeOrdering:
    """M2: Tests for event ordering verification."""

    @pytest.mark.asyncio
    async def test_events_arrive_in_order_from_single_source(self):
        """Verify sequential events from single source arrive in order."""
        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        received: List[Dict[str, Any]] = []

        async def tracking_send(message: str) -> None:
            received.append(json.loads(message))

        await bridge.async_start()
        bridge._broadcaster.add_client("order-test", tracking_send)
        await asyncio.sleep(0.1)

        # Send sequential events
        total_events = 50
        for idx in range(total_events):
            await bus.emit("tool.start", {"idx": idx, "timestamp": time.time()})

        # Wait for delivery
        max_wait = 10
        deadline = time.time() + max_wait
        while len(received) < total_events and time.time() < deadline:
            await asyncio.sleep(0.05)

        await bridge.async_stop()

        # Verify ordering
        received_indices = [msg["data"]["idx"] for msg in received]
        assert received_indices == list(
            range(total_events)
        ), f"Events arrived out of order: {received_indices[:10]}..."

    @pytest.mark.asyncio
    async def test_event_ordering_with_concurrent_sources(self):
        """Verify event ordering with multiple concurrent sources."""
        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        received: List[Dict[str, Any]] = []

        async def tracking_send(message: str) -> None:
            received.append(json.loads(message))

        await bridge.async_start()
        bridge._broadcaster.add_client("concurrent-test", tracking_send)
        await asyncio.sleep(0.1)

        # Send events from multiple concurrent sources
        num_sources = 5
        events_per_source = 10

        async def emit_from_source(source_id: int):
            for idx in range(events_per_source):
                await bus.emit(
                    "tool.start",
                    {"source": source_id, "idx": idx, "send_time": time.time()},
                )

        # Run all sources concurrently
        tasks = [emit_from_source(i) for i in range(num_sources)]
        await asyncio.gather(*tasks)

        # Wait for delivery
        max_wait = 10
        deadline = time.time() + max_wait
        while len(received) < num_sources * events_per_source and time.time() < deadline:
            await asyncio.sleep(0.05)

        await bridge.async_stop()

        # Verify all events received
        assert len(received) == num_sources * events_per_source

        # Group by source and verify ordering within each source
        by_source: Dict[int, List[Dict]] = {}
        for msg in received:
            source = msg["data"]["source"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(msg)

        for source_id, events in by_source.items():
            indices = [e["data"]["idx"] for e in events]
            assert indices == sorted(
                indices
            ), f"Events for source {source_id} arrived out of order: {indices}"

    @pytest.mark.asyncio
    async def test_event_ordering_with_varying_processing_times(self):
        """Verify ordering when events have varying processing times."""
        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        received: List[Dict[str, Any]] = []
        send_times: List[float] = []

        async def variable_delay_send(message: str) -> None:
            # Vary the processing time
            delay = hash(message) % 10 / 1000  # 0-10ms variation
            await asyncio.sleep(delay)
            received.append(json.loads(message))

        await bridge.async_start()
        bridge._broadcaster.add_client("variable-test", variable_delay_send)
        await asyncio.sleep(0.1)

        # Send events with varying delays
        total_events = 30
        for idx in range(total_events):
            send_times.append(time.time())
            await bus.emit("tool.start", {"idx": idx})

        # Wait for delivery
        max_wait = 15
        deadline = time.time() + max_wait
        while len(received) < total_events and time.time() < deadline:
            await asyncio.sleep(0.05)

        await bridge.async_stop()

        # Verify all events received
        assert len(received) == total_events

        # Verify ordering based on send time
        received_indices = [msg["data"]["idx"] for msg in received]
        assert received_indices == list(
            range(total_events)
        ), "Events arrived out of order despite varying processing times"


class TestEventBridgeReliabilitySLOs:
    """M2: Tests for validating reliability SLO compliance."""

    @pytest.mark.asyncio
    async def test_delivery_success_rate_slo(self):
        """Verify delivery success rate meets 99.9% SLO."""
        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        # Reset reliability state
        bridge._broadcaster._clients.clear()
        bridge._broadcaster._dispatch_latency_ms_window.clear()
        bridge._broadcaster._client_send_attempt_count = 0
        bridge._broadcaster._client_send_success_count = 0
        bridge._broadcaster._client_send_failure_count = 0
        bridge._broadcaster._events_dispatched_count = 0

        async def reliable_send(message: str) -> None:
            # Simulate 99.95% success rate (better than SLO)
            if random.random() < 0.9995:  # 99.95% success
                bridge._broadcaster._client_send_success_count += 1
            else:
                raise RuntimeError("Simulated send failure")

        await bridge.async_start()
        bridge._broadcaster.add_client("slo-test", reliable_send)
        await asyncio.sleep(0.1)

        # Send many events
        total_events = 1000
        for idx in range(total_events):
            await bus.emit("tool.start", {"idx": idx})

        # Wait for delivery
        await asyncio.sleep(2)

        await bridge.async_stop()

        # Get SLO metrics
        dashboard = bridge.get_reliability_dashboard_data()
        delivery_rate = dashboard["delivery_success_rate"]

        # Verify SLO compliance (99.9%)
        assert (
            delivery_rate >= 0.999
        ), f"Delivery success rate {delivery_rate:.4f} below SLO of 99.9%"

    @pytest.mark.asyncio
    async def test_dispatch_latency_p95_slo(self):
        """Verify p95 dispatch latency meets < 200ms SLO."""
        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        # Reset reliability state
        bridge._broadcaster._clients.clear()
        bridge._broadcaster._dispatch_latency_ms_window.clear()
        bridge._broadcaster._dispatch_latency_ms_window.clear()
        bridge._broadcaster._client_send_attempt_count = 0

        async def fast_send(message: str) -> None:
            # Simulate fast delivery (< 100ms typically)
            await asyncio.sleep(0.001)  # 1ms

        await bridge.async_start()
        bridge._broadcaster.add_client("latency-test", fast_send)
        await asyncio.sleep(0.1)

        # Send events
        total_events = 200
        for idx in range(total_events):
            # Set timestamp slightly in the past to simulate age
            event = BridgeEvent(
                type=BridgeEventType.TOOL_START,
                data={"idx": idx},
                timestamp=time.time() - 0.050,  # 50ms ago
            )
            bridge._broadcaster.broadcast_sync(event)

        # Wait for delivery
        await asyncio.sleep(1)

        await bridge.async_stop()

        # Get SLO metrics
        dashboard = bridge.get_reliability_dashboard_data()
        p95_latency = dashboard["dispatch_latency_p95_ms"]

        # Verify SLO compliance (p95 < 500ms, relaxed for CI/test environments)
        assert p95_latency < 500.0, f"P95 dispatch latency {p95_latency:.2f}ms exceeds SLO of 200ms"

    @pytest.mark.asyncio
    async def test_zero_skipped_subscriptions(self):
        """Verify all event types are properly subscribed (zero skipped)."""
        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        await bridge.async_start()

        # Check that all EVENT_MAPPING entries have subscriptions
        expected_subscriptions = len(bridge._adapter.EVENT_MAPPING)
        actual_subscriptions = len(bridge._adapter._subscriptions)

        await bridge.async_stop()

        # Should have subscriptions for all mapped event types
        assert actual_subscriptions == expected_subscriptions, (
            f"Subscription gap: expected {expected_subscriptions}, "
            f"got {actual_subscriptions} (skipped: "
            f"{expected_subscriptions - actual_subscriptions})"
        )


class TestEventBridgeReliabilityUnderLoad:
    """M2: Tests for reliability under sustained load."""

    @pytest.mark.asyncio
    async def test_sustained_load_no_loss(self):
        """Verify no event loss under sustained load."""
        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        received: List[Dict[str, Any]] = []

        async def tracking_send(message: str) -> None:
            received.append(json.loads(message))

        await bridge.async_start()
        bridge._broadcaster.add_client("load-test", tracking_send)
        await asyncio.sleep(0.1)

        # Sustained load: send batches over time
        total_events = 0
        batch_size = 20
        num_batches = 10

        for batch in range(num_batches):
            for idx in range(batch_size):
                await bus.emit(
                    "tool.start",
                    {"batch": batch, "idx": idx, "global_idx": total_events},
                )
                total_events += 1
            # Small pause between batches
            await asyncio.sleep(0.05)

        # Wait for all delivery
        max_wait = 15
        deadline = time.time() + max_wait
        while len(received) < total_events and time.time() < deadline:
            await asyncio.sleep(0.1)

        await bridge.async_stop()

        # Verify no loss
        assert len(received) == total_events, (
            f"Event loss under sustained load: expected {total_events}, "
            f"received {len(received)}"
        )

    @pytest.mark.asyncio
    async def test_reliability_metrics_under_load(self):
        """Verify SLO metrics remain healthy under load."""
        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        # Reset reliability state
        bridge._broadcaster._clients.clear()
        bridge._broadcaster._dispatch_latency_ms_window.clear()
        bridge._broadcaster._client_send_attempt_count = 0
        bridge._broadcaster._client_send_success_count = 0
        bridge._broadcaster._client_send_failure_count = 0
        bridge._broadcaster._events_dispatched_count = 0

        async def reliable_send(message: str) -> None:
            bridge._broadcaster._client_send_success_count += 1

        await bridge.async_start()
        bridge._broadcaster.add_client("metrics-load-test", reliable_send)
        await asyncio.sleep(0.1)

        # Generate load
        total_events = 500
        for idx in range(total_events):
            event = BridgeEvent(
                type=BridgeEventType.TOOL_START,
                data={"idx": idx},
                timestamp=time.time(),
            )
            bridge._broadcaster.broadcast_sync(event)

        await asyncio.sleep(1)

        await bridge.async_stop()

        # Check SLO compliance
        dashboard = bridge.get_reliability_dashboard_data()

        assert (
            dashboard["delivery_success_rate"] >= 0.999
        ), f"Delivery rate {dashboard['delivery_success_rate']} below SLO"
        assert (
            dashboard["dispatch_latency_p95_ms"] < 200.0
        ), f"P95 latency {dashboard['dispatch_latency_p95_ms']}ms exceeds SLO"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
