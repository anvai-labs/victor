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

"""Tests for EventBridge module.

TDD tests for real-time event streaming to VS Code and other clients.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.core.events import MessagingEvent

# EventBroadcaster singleton isolation is handled by the shared autouse fixture
# `_reset_event_broadcaster_singleton` in tests/unit/conftest.py (covers every
# bridge test file, not just this one), via EventBroadcaster.reset_instance().


class TestEventBridgeEventTypes:
    """Tests for EventBridge event type handling."""

    @pytest.mark.asyncio
    async def test_bridge_event_type_enum(self):
        """Test BridgeEventType enum values."""
        from victor.integrations.api.event_bridge import BridgeEventType

        # Tool events
        assert BridgeEventType.TOOL_START.value == "tool.start"
        assert BridgeEventType.TOOL_COMPLETE.value == "tool.complete"
        assert BridgeEventType.TOOL_ERROR.value == "tool.error"

        # File events
        assert BridgeEventType.FILE_CREATED.value == "file.created"
        assert BridgeEventType.FILE_MODIFIED.value == "file.modified"
        assert BridgeEventType.FILE_DELETED.value == "file.deleted"

        # Provider events
        assert BridgeEventType.PROVIDER_SWITCH.value == "provider.switch"
        assert BridgeEventType.PROVIDER_ERROR.value == "provider.error"

    @pytest.mark.asyncio
    async def test_bridge_event_serialization(self):
        """Test BridgeEvent serialization to JSON."""
        from victor.integrations.api.event_bridge import BridgeEvent, BridgeEventType

        event = BridgeEvent(
            type=BridgeEventType.TOOL_START, data={"tool_name": "read_file", "args": {}}
        )

        event_dict = event.to_dict()

        assert event_dict["type"] == "tool.start"
        assert event_dict["data"]["tool_name"] == "read_file"
        assert "id" in event_dict
        assert "timestamp" in event_dict

        # Test JSON serialization
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "tool.start"

    @pytest.mark.asyncio
    async def test_bridge_event_has_unique_ids(self):
        """Test that each BridgeEvent gets a unique ID."""
        from victor.integrations.api.event_bridge import BridgeEvent, BridgeEventType

        event1 = BridgeEvent(type=BridgeEventType.TOOL_START, data={})
        event2 = BridgeEvent(type=BridgeEventType.TOOL_START, data={})

        assert event1.id != event2.id

    @pytest.mark.asyncio
    async def test_bridge_event_has_timestamp(self):
        """Test that BridgeEvent has a timestamp."""
        from victor.integrations.api.event_bridge import BridgeEvent, BridgeEventType

        event = BridgeEvent(type=BridgeEventType.TOOL_START, data={})

        assert event.timestamp > 0


class TestEventBridgeIntegration:
    """Integration tests for EventBridge with EventBus."""

    @pytest.mark.asyncio
    async def test_event_bridge_import(self):
        """Test that EventBridge can be imported."""
        from victor.integrations.api.event_bridge import EventBridge

        assert EventBridge is not None

    @pytest.mark.asyncio
    async def test_event_bridge_creation(self):
        """Test that EventBridge can be created with ObservabilityBus."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        assert bridge is not None
        assert bridge._event_bus == bus

    @pytest.mark.asyncio
    async def test_event_bridge_start_stop(self):
        """Test EventBridge start and stop lifecycle."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        bridge.start()
        assert bridge._running is True

        bridge.stop()
        assert bridge._running is False

    @pytest.mark.asyncio
    async def test_event_bridge_broadcaster_exists(self):
        """Test that EventBridge has a broadcaster."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        assert hasattr(bridge, "_broadcaster")


class TestEventBridgeConvenienceEmitters:
    """Tests for direct helper emitters."""

    def test_emit_tool_complete_includes_follow_up_suggestions(self):
        """emit_tool_complete should preserve follow-up suggestion payloads."""
        from victor.integrations.api.event_bridge import emit_tool_complete

        broadcaster = MagicMock()

        with patch(
            "victor.integrations.api.event_bridge.EventBroadcaster",
            return_value=broadcaster,
        ):
            emit_tool_complete(
                "tool-123",
                "done",
                duration_ms=42,
                follow_up_suggestions=[
                    {
                        "command": 'graph(mode="callers", node="parse_json", depth=1)',
                        "description": "Show callers of parse_json.",
                    }
                ],
            )

        broadcaster.broadcast_sync.assert_called_once()
        event = broadcaster.broadcast_sync.call_args.args[0]
        assert event.data["tool_id"] == "tool-123"
        assert event.data["follow_up_suggestions"] == [
            {
                "command": 'graph(mode="callers", node="parse_json", depth=1)',
                "description": "Show callers of parse_json.",
            }
        ]


class TestEventBusAdapterCompatibility:
    """Compatibility tests for sync and async EventBus APIs.

    M1: Async subscribe is now the primary and only supported path.
    Sync subscribe is no longer supported - use ObservabilityBus or other
    async-compatible event bus.
    """

    @pytest.mark.asyncio
    async def test_event_bus_adapter_supports_async_subscribe(self):
        """Adapter should subscribe and unsubscribe using async handles."""
        from victor.integrations.api.event_bridge import EventBusAdapter

        class FakeHandle:
            def __init__(self, pattern: str):
                self.pattern = pattern
                self.unsubscribed = False

            async def unsubscribe(self):
                self.unsubscribed = True

        class FakeAsyncBus:
            def __init__(self):
                self.subscriptions = []
                self.handles = []

            async def subscribe(self, pattern, handler):
                handle = FakeHandle(pattern)
                self.subscriptions.append((pattern, handler))
                self.handles.append(handle)
                return handle

        event_bus = FakeAsyncBus()
        adapter = EventBusAdapter()

        adapter.connect(event_bus)
        await asyncio.sleep(0.01)

        assert len(event_bus.subscriptions) == len(adapter.EVENT_MAPPING)
        assert len(adapter._subscriptions) == len(adapter.EVENT_MAPPING)
        assert all(asyncio.iscoroutinefunction(handler) for _, handler in event_bus.subscriptions)

        adapter.disconnect()
        await asyncio.sleep(0.01)

        assert all(handle.unsubscribed for handle in event_bus.handles)

    @pytest.mark.asyncio
    async def test_event_bus_adapter_rejects_sync_subscribe_m1(self):
        """M1: Adapter should reject event buses with sync subscribe (not supported)."""
        from victor.integrations.api.event_bridge import EventBusAdapter

        class FakeSyncBus:
            def __init__(self):
                self.subscribe_calls = []

            def subscribe(self, pattern, handler):
                self.subscribe_calls.append((pattern, handler))
                return None

        event_bus = FakeSyncBus()
        adapter = EventBusAdapter()

        # The sync connect() should fail to schedule the operation
        # and the async connect_async() should raise RuntimeError
        with pytest.raises(RuntimeError, match="async subscribe"):
            await adapter.connect_async(event_bus)

    def test_event_bus_adapter_preserves_correlation_id_in_bridge_payload(self):
        """Bridge payloads should expose correlation_id for websocket clients."""
        from victor.integrations.api.event_bridge import EventBusAdapter

        broadcaster = MagicMock()
        adapter = EventBusAdapter(broadcaster=broadcaster)
        event = MessagingEvent(
            topic="tool.complete",
            data={"tool_name": "graph", "success": True},
            correlation_id="chat_req_456",
        )

        adapter._on_event(event)

        broadcaster.broadcast_sync.assert_called_once()
        bridge_event = broadcaster.broadcast_sync.call_args.args[0]
        assert bridge_event.data["correlation_id"] == "chat_req_456"


class TestEventBridgeReliability:
    """Reliability checks for bridge event delivery."""

    @pytest.mark.asyncio
    async def test_event_bridge_burst_delivery_has_no_loss_or_reordering(self):
        """A burst of events should be delivered completely and in-order."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)
        received = []

        async def send_func(message: str):
            received.append(json.loads(message))

        # Generous deadline: this poller returns the instant the predicate is
        # true (10ms poll), so a wide timeout costs nothing on success and only
        # tolerates a saturated CI runner (the full test matrix runs 72 jobs in
        # parallel). 2.0s was too tight and flaked under load.
        async def wait_for(predicate, timeout: float = 15.0):
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                if predicate():
                    return
                await asyncio.sleep(0.01)
            pytest.fail("Timed out waiting for event bridge condition")

        bridge.start()
        bridge._broadcaster.add_client("event-loss-check", send_func)

        await wait_for(lambda: bridge._broadcaster._running)
        await wait_for(
            lambda: backend.get_subscription_count() >= len(bridge._adapter.EVENT_MAPPING)
        )

        total_events = 25
        for idx in range(total_events):
            await bus.emit("tool.start", {"idx": idx})

        await wait_for(lambda: len(received) >= total_events)
        assert len(received) == total_events
        assert [msg["data"]["idx"] for msg in received] == list(range(total_events))

        bridge._broadcaster.remove_client("event-loss-check")
        bridge.stop()
        await wait_for(lambda: not bridge._broadcaster._running)
        await bus.disconnect()

    def _reset_reliability_state(self, bridge) -> None:
        """Reset singleton broadcaster reliability state between tests."""
        bridge._broadcaster._clients.clear()
        bridge._broadcaster._dispatch_latency_ms_window.clear()
        bridge._broadcaster._client_send_attempt_count = 0
        bridge._broadcaster._client_send_success_count = 0
        bridge._broadcaster._client_send_failure_count = 0
        bridge._broadcaster._events_dispatched_count = 0
        bridge._broadcaster._last_slo_breach_log_ts = 0.0

    @pytest.mark.asyncio
    async def test_reliability_dashboard_defaults_before_delivery(self):
        """Dashboard should expose healthy defaults before any send attempts."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)
        self._reset_reliability_state(bridge)

        dashboard = bridge.get_reliability_dashboard_data()

        assert dashboard["events_dispatched"] == 0
        assert dashboard["total_send_attempts"] == 0
        assert dashboard["send_successes"] == 0
        assert dashboard["send_failures"] == 0
        assert dashboard["delivery_success_rate"] == 1.0
        assert dashboard["dispatch_latency_p95_ms"] == 0.0
        assert dashboard["slo_status"]["delivery_success_rate"] is True
        assert dashboard["slo_status"]["dispatch_latency_p95_ms"] is True

    @pytest.mark.asyncio
    async def test_reliability_dashboard_tracks_delivery_and_slos(self):
        """Dashboard should track success/failure counts, p95 latency, and SLO status."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import (
            BridgeEvent,
            BridgeEventType,
            EventBridge,
        )

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)
        self._reset_reliability_state(bridge)

        async def successful_send(_message: str) -> None:
            return None

        async def failing_send(_message: str) -> None:
            raise RuntimeError("network timeout")

        bridge._broadcaster.add_client("ok-client", successful_send)
        bridge._broadcaster.add_client("bad-client", failing_send)

        event = BridgeEvent(
            type=BridgeEventType.TOOL_START,
            data={"idx": 1},
            timestamp=time.time() - 0.35,
        )
        await bridge._broadcaster._send_to_clients(event)

        dashboard = bridge.get_reliability_dashboard_data()

        assert dashboard["events_dispatched"] == 1
        assert dashboard["total_send_attempts"] == 2
        assert dashboard["send_successes"] == 1
        assert dashboard["send_failures"] == 1
        assert dashboard["delivery_success_rate"] == 0.5
        assert dashboard["dispatch_latency_p95_ms"] > 200.0
        assert dashboard["slo_status"]["delivery_success_rate"] is False
        assert dashboard["slo_status"]["dispatch_latency_p95_ms"] is False

        # A single transient failure should not evict the client immediately.
        assert "bad-client" in bridge._broadcaster._clients

    @pytest.mark.asyncio
    async def test_reliability_dashboard_evicts_after_repeated_failures(self):
        """Broadcaster should evict clients after repeated consecutive send failures."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import (
            BridgeEvent,
            BridgeEventType,
            EventBridge,
        )

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)
        self._reset_reliability_state(bridge)

        async def failing_send(_message: str) -> None:
            raise RuntimeError("network timeout")

        bridge._broadcaster.add_client("bad-client", failing_send)

        for idx in range(bridge._broadcaster._max_consecutive_send_failures):
            event = BridgeEvent(
                type=BridgeEventType.TOOL_START,
                data={"idx": idx},
                timestamp=time.time(),
            )
            await bridge._broadcaster._send_to_clients(event)

        assert "bad-client" not in bridge._broadcaster._clients


class TestEventBridgeBroadcaster:
    """Tests for EventBridge broadcaster functionality."""

    @pytest.mark.asyncio
    async def test_broadcaster_add_client(self):
        """Test adding a client to the broadcaster."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)
        bridge.start()

        # Mock send function
        send_func = AsyncMock()
        bridge._broadcaster.add_client("client-1", send_func)

        assert "client-1" in bridge._broadcaster._clients

        bridge.stop()

    @pytest.mark.asyncio
    async def test_broadcaster_remove_client(self):
        """Test removing a client from the broadcaster."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)
        bridge.start()

        send_func = AsyncMock()
        bridge._broadcaster.add_client("client-1", send_func)
        bridge._broadcaster.remove_client("client-1")

        assert "client-1" not in bridge._broadcaster._clients

        bridge.stop()


class TestEventBridgeFiltering:
    """Tests for event filtering in EventBridge."""

    @pytest.mark.asyncio
    async def test_event_bridge_all_event_types(self):
        """Test that all expected event types exist."""
        from victor.integrations.api.event_bridge import BridgeEventType

        expected_types = [
            "TOOL_START",
            "TOOL_PROGRESS",
            "TOOL_COMPLETE",
            "TOOL_ERROR",
            "FILE_CREATED",
            "FILE_MODIFIED",
            "FILE_DELETED",
            "PROVIDER_SWITCH",
            "PROVIDER_ERROR",
            "PROVIDER_RECOVERY",
            "SESSION_START",
            "SESSION_END",
            "SESSION_ERROR",
            "METRICS_UPDATE",
            "BUDGET_WARNING",
            "BUDGET_EXHAUSTED",
            "NOTIFICATION",
            "ERROR",
        ]

        for type_name in expected_types:
            assert hasattr(BridgeEventType, type_name), f"Missing event type: {type_name}"

    @pytest.mark.asyncio
    async def test_broadcaster_filters_events_by_correlation_id(self):
        """Clients with a correlation filter should only receive matching events."""
        from victor.integrations.api.event_bridge import (
            BridgeEvent,
            BridgeEventType,
            EventBroadcaster,
        )

        broadcaster = EventBroadcaster()
        broadcaster._clients.clear()

        matching = AsyncMock()
        general = AsyncMock()

        broadcaster.add_client(
            "filtered",
            matching,
            subscriptions={"tool.complete"},
            correlation_id="chat_123",
        )
        broadcaster.add_client("general", general, subscriptions={"tool.complete"})

        await broadcaster._send_to_clients(
            BridgeEvent(
                type=BridgeEventType.TOOL_COMPLETE,
                data={"tool_name": "graph", "correlation_id": "chat_999"},
            )
        )

        matching.assert_not_awaited()
        general.assert_awaited_once()

        await broadcaster._send_to_clients(
            BridgeEvent(
                type=BridgeEventType.TOOL_COMPLETE,
                data={"tool_name": "graph", "correlation_id": "chat_123"},
            )
        )

        assert matching.await_count == 1
        assert general.await_count == 2
        broadcaster._clients.clear()

    @pytest.mark.asyncio
    async def test_websocket_handler_subscribe_supports_categories_and_correlation_id(
        self,
    ):
        """Subscribe messages should normalize categories and preserve correlation filters."""
        from victor.integrations.api.event_bridge import WebSocketEventHandler

        broadcaster = MagicMock()
        handler = WebSocketEventHandler(broadcaster=broadcaster)

        await handler._handle_message(
            "client-1",
            json.dumps(
                {
                    "type": "subscribe",
                    "categories": ["tool.complete", "tool.error"],
                    "correlation_id": "chat_abc",
                }
            ),
        )

        broadcaster.normalize_subscriptions.assert_called_once_with(["tool.complete", "tool.error"])
        broadcaster.update_subscriptions.assert_called_once_with(
            "client-1",
            broadcaster.normalize_subscriptions.return_value,
            correlation_id="chat_abc",
        )

    def test_broadcaster_recent_events_supports_same_filters(self):
        """Recent-event snapshots should use the same category/correlation matching rules."""
        from victor.integrations.api.event_bridge import (
            BridgeEvent,
            BridgeEventType,
            EventBroadcaster,
        )

        broadcaster = EventBroadcaster()
        broadcaster._recent_events.clear()

        broadcaster.broadcast_sync(
            BridgeEvent(
                type=BridgeEventType.TOOL_COMPLETE,
                data={"tool_name": "graph", "correlation_id": "chat_1"},
            )
        )
        broadcaster.broadcast_sync(
            BridgeEvent(
                type=BridgeEventType.TOOL_ERROR,
                data={"tool_name": "graph", "correlation_id": "chat_2"},
            )
        )

        filtered = broadcaster.get_recent_events(
            limit=5,
            subscriptions={"tool.complete"},
            correlation_id="chat_1",
        )

        assert len(filtered) == 1
        assert filtered[0].type is BridgeEventType.TOOL_COMPLETE
        assert filtered[0].data["correlation_id"] == "chat_1"
        broadcaster._recent_events.clear()


class TestEventBusAdapterAsyncPath:
    """M1 Tests for async subscribe path in EventBusAdapter."""

    @pytest.mark.asyncio
    async def test_connect_async_with_async_event_bus(self):
        """Test connect_async successfully subscribes to async event bus."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBusAdapter

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        adapter = EventBusAdapter()

        await adapter.connect_async(bus)

        # Verify subscriptions were created
        assert len(adapter._subscriptions) > 0
        assert len(adapter._subscription_handles) == len(adapter._subscriptions)

    @pytest.mark.asyncio
    async def test_connect_async_fails_without_subscribe_method(self):
        """Test connect_async raises error when event bus has no subscribe method."""
        from victor.integrations.api.event_bridge import EventBusAdapter

        class InvalidBus:
            pass

        bus = InvalidBus()
        adapter = EventBusAdapter()

        with pytest.raises(RuntimeError, match="no subscribe\\(\\) method"):
            await adapter.connect_async(bus)

    @pytest.mark.asyncio
    async def test_connect_async_fails_with_sync_subscribe(self):
        """Test connect_async raises error when subscribe is not async."""
        from victor.integrations.api.event_bridge import EventBusAdapter

        class SyncBus:
            def subscribe(self, pattern, handler):
                return None

        bus = SyncBus()
        adapter = EventBusAdapter()

        with pytest.raises(RuntimeError, match="async subscribe"):
            await adapter.connect_async(bus)

    @pytest.mark.asyncio
    async def test_connect_async_handles_partial_subscription_failures(self):
        """Test connect_async continues when some subscriptions fail."""
        from victor.integrations.api.event_bridge import EventBusAdapter

        class FlakyAsyncBus:
            def __init__(self):
                self.call_count = 0

            async def subscribe(self, pattern, handler):
                self.call_count += 1
                # Fail on first subscription
                if self.call_count == 1:
                    raise RuntimeError("First subscription failed")
                # Return mock handle for others
                from unittest.mock import Mock

                handle = Mock()
                handle.unsubscribe = AsyncMock()
                return handle

        bus = FlakyAsyncBus()
        adapter = EventBusAdapter()

        # Should not raise, should continue with other subscriptions
        await adapter.connect_async(bus)

        # Should have some subscriptions despite failures
        assert len(adapter._subscription_handles) > 0

    @pytest.mark.asyncio
    async def test_disconnect_async_properly_unsubscribes(self):
        """Test disconnect_async properly awaits all unsubscribe calls."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBusAdapter

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        adapter = EventBusAdapter()

        await adapter.connect_async(bus)
        initial_sub_count = len(adapter._subscription_handles)

        await adapter.disconnect_async()

        # All handles should be cleared
        assert len(adapter._subscription_handles) == 0
        assert len(adapter._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_disconnect_async_handles_disconnect_during_subscribe(self):
        """Test disconnect_async handles rapid connect/disconnect cycles."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBusAdapter

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        adapter = EventBusAdapter()

        await adapter.connect_async(bus)
        # Immediately disconnect
        await adapter.disconnect_async()

        # Should cleanly handle the rapid cycle
        assert len(adapter._subscription_handles) == 0

    @pytest.mark.asyncio
    async def test_disconnect_async_idempotent(self):
        """Test disconnect_async can be called multiple times safely."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBusAdapter

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        adapter = EventBusAdapter()

        await adapter.connect_async(bus)
        await adapter.disconnect_async()
        # Second disconnect should be safe
        await adapter.disconnect_async()

        assert len(adapter._subscription_handles) == 0


class TestEventBridgeAsyncPath:
    """M1 Tests for async path in EventBridge."""

    @pytest.mark.asyncio
    async def test_async_start_stop_lifecycle(self):
        """Test async_start and async_stop lifecycle."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        await bridge.async_start()
        assert bridge._running is True
        assert len(bridge._adapter._subscription_handles) > 0

        await bridge.async_stop()
        assert bridge._running is False
        assert len(bridge._adapter._subscription_handles) == 0

    @pytest.mark.asyncio
    async def test_async_start_idempotent(self):
        """Test async_start can be called multiple times safely."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        await bridge.async_start()
        await bridge.async_start()  # Second call should be no-op

        assert bridge._running is True

        await bridge.async_stop()

    @pytest.mark.asyncio
    async def test_async_stop_idempotent(self):
        """Test async_stop can be called multiple times safely."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        await bridge.async_start()
        await bridge.async_stop()
        await bridge.async_stop()  # Second call should be no-op

        assert bridge._running is False

    @pytest.mark.asyncio
    async def test_async_start_without_event_bus(self):
        """Test async_start works without an event bus (broadcast only)."""
        from victor.integrations.api.event_bridge import EventBridge

        bridge = EventBridge(event_bus=None)

        await bridge.async_start()
        assert bridge._running is True

        await bridge.async_stop()
        assert bridge._running is False


class TestEventBridgeAsyncBackwardCompatibility:
    """M1 Tests for backward compatibility of sync methods."""

    @pytest.mark.asyncio
    async def test_sync_start_still_works(self):
        """Test sync start() still works for backward compatibility."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        # Sync start in async context should schedule the operation
        bridge.start()
        await asyncio.sleep(0.1)  # Let the scheduled operation complete

        # May or may not be running depending on timing
        # but should not raise an error
        await bridge.async_stop()

    @pytest.mark.asyncio
    async def test_sync_stop_still_works(self):
        """Test sync stop() still works for backward compatibility."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        await bridge.async_start()

        # Sync stop in async context should schedule the operation
        bridge.stop()
        await asyncio.sleep(0.1)  # Let the scheduled operation complete

    @pytest.mark.asyncio
    async def test_adapter_sync_connect_still_works(self):
        """Test adapter sync connect() still works for backward compatibility."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBusAdapter

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        adapter = EventBusAdapter()

        # Sync connect in async context should schedule the operation
        adapter.connect(bus)
        await asyncio.sleep(0.1)  # Let the scheduled operation complete

        # Clean up
        await adapter.disconnect_async()


class TestEventBroadcasterLoopIsolation:
    def test_stop_tolerates_broadcast_task_from_different_loop(self):
        """Stopping the singleton broadcaster should not hang across event loops."""
        from victor.integrations.api.event_bridge import EventBroadcaster

        broadcaster = EventBroadcaster()
        old_loop = asyncio.new_event_loop()
        try:
            old_loop.run_until_complete(broadcaster.start())

            async def _stop_on_new_loop() -> None:
                await asyncio.wait_for(broadcaster.stop(), timeout=0.5)

            asyncio.run(_stop_on_new_loop())

            assert broadcaster._running is False
            assert broadcaster._broadcast_task is None
        finally:
            pending = asyncio.all_tasks(old_loop)
            for task in pending:
                task.cancel()
            if pending:
                old_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            old_loop.close()

    def test_delivery_survives_loop_swap_no_loss_or_reordering(self):
        """Regression: a leaked singleton bound to a closed loop must not break
        delivery on a later loop. Start on loop A (left bound), then run a
        25-event burst on loop B; the engine must rebind and deliver all 25 in
        order. Before the SRP/loop-correct redesign the stale DeliveryEngine
        created tasks on the closed loop A -> 'Event loop is closed' / timeout.
        """
        import json as _json

        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge, EventBroadcaster

        EventBroadcaster.reset_instance()

        async def _leak_on_loop_a():
            bridge = EventBridge(ObservabilityBus(backend=InMemoryEventBackend()))
            await bridge.async_start()  # binds the singleton engine to loop A

        async def _burst_on_loop_b():
            bus = ObservabilityBus(backend=InMemoryEventBackend())
            bridge = EventBridge(bus)
            received = []

            async def _send(message):
                received.append(_json.loads(message))

            async def _wait(predicate, timeout=5.0):
                deadline = asyncio.get_running_loop().time() + timeout
                while asyncio.get_running_loop().time() < deadline:
                    if predicate():
                        return True
                    await asyncio.sleep(0.01)
                return False

            await bridge.async_start()  # same singleton -> must rebind A -> B
            bridge._broadcaster.add_client("loss-check", _send)
            await _wait(lambda: bridge._broadcaster._running)
            total = 25
            for idx in range(total):
                await bus.emit("tool.start", {"idx": idx})
            ok = await _wait(lambda: len(received) >= total)
            ordered = [m["data"]["idx"] for m in received] == list(range(total))
            await bridge.async_stop()
            return len(received), (ok and ordered)

        asyncio.run(_leak_on_loop_a())  # loop A closes; singleton left bound to it
        count, ok = asyncio.run(_burst_on_loop_b())  # loop B
        assert count == 25, f"expected 25 delivered after loop swap, got {count}"
        assert ok, "events lost or reordered after loop swap"
        EventBroadcaster.reset_instance()


class TestEventBridgeSyncBridge:
    def test_adapter_connect_uses_shared_sync_bridge_without_running_loop(self):
        """Sync adapter connect should bridge with run_sync outside async contexts."""
        from victor.integrations.api import event_bridge as event_bridge_module

        adapter = event_bridge_module.EventBusAdapter()
        event_bus = object()
        coro = object()
        mock_async = MagicMock(return_value=coro)

        with (
            patch.object(
                event_bridge_module.asyncio,
                "get_running_loop",
                side_effect=RuntimeError,
            ),
            patch.object(adapter, "connect_async", mock_async),
            patch.object(event_bridge_module, "run_sync", return_value=None) as mock_run_sync,
        ):
            adapter.connect(event_bus)

        mock_async.assert_called_once_with(event_bus)
        mock_run_sync.assert_called_once_with(coro)

    def test_adapter_disconnect_uses_shared_sync_bridge_without_running_loop(self):
        """Sync adapter disconnect should bridge with run_sync outside async contexts."""
        from victor.integrations.api import event_bridge as event_bridge_module

        adapter = event_bridge_module.EventBusAdapter()
        coro = object()
        mock_async = MagicMock(return_value=coro)

        with (
            patch.object(
                event_bridge_module.asyncio,
                "get_running_loop",
                side_effect=RuntimeError,
            ),
            patch.object(adapter, "disconnect_async", mock_async),
            patch.object(event_bridge_module, "run_sync", return_value=None) as mock_run_sync,
        ):
            adapter.disconnect()

        mock_async.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)

    def test_adapter_run_async_operation_uses_shared_sync_bridge_without_running_loop(
        self,
    ):
        """Internal adapter bridge helper should delegate to run_sync."""
        from victor.integrations.api import event_bridge as event_bridge_module

        adapter = event_bridge_module.EventBusAdapter()
        awaitable = object()
        on_success = MagicMock()

        with (
            patch.object(
                event_bridge_module.asyncio,
                "get_running_loop",
                side_effect=RuntimeError,
            ),
            patch.object(event_bridge_module, "run_sync", return_value="done") as mock_run_sync,
        ):
            adapter._run_async_operation(awaitable, description="connect", on_success=on_success)

        mock_run_sync.assert_called_once_with(awaitable)
        on_success.assert_called_once_with("done")

    def test_bridge_start_uses_shared_sync_bridge_without_running_loop(self):
        """Sync bridge start should delegate to run_sync outside async contexts."""
        from victor.integrations.api import event_bridge as event_bridge_module

        bridge = event_bridge_module.EventBridge()
        coro = object()
        mock_async = MagicMock(return_value=coro)

        with (
            patch.object(
                event_bridge_module.asyncio,
                "get_running_loop",
                side_effect=RuntimeError,
            ),
            patch.object(bridge, "_start_and_set_flag", mock_async),
            patch.object(event_bridge_module, "run_sync", return_value=None) as mock_run_sync,
        ):
            bridge.start()

        mock_async.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)

    def test_bridge_stop_uses_shared_sync_bridge_without_running_loop(self):
        """Sync bridge stop should delegate to run_sync outside async contexts."""
        from victor.integrations.api import event_bridge as event_bridge_module

        bridge = event_bridge_module.EventBridge()
        bridge._running = True
        coro = object()
        mock_async = MagicMock(return_value=coro)

        with (
            patch.object(
                event_bridge_module.asyncio,
                "get_running_loop",
                side_effect=RuntimeError,
            ),
            patch.object(bridge, "_stop_and_cleanup", mock_async),
            patch.object(event_bridge_module, "run_sync", return_value=None) as mock_run_sync,
        ):
            bridge.stop()

        mock_async.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)

    def test_bridge_run_async_operation_uses_shared_sync_bridge_without_running_loop(
        self,
    ):
        """EventBridge lifecycle helper should delegate to run_sync."""
        from victor.integrations.api import event_bridge as event_bridge_module

        bridge = event_bridge_module.EventBridge()
        awaitable = object()

        with (
            patch.object(
                event_bridge_module.asyncio,
                "get_running_loop",
                side_effect=RuntimeError,
            ),
            patch.object(event_bridge_module, "run_sync", return_value=None) as mock_run_sync,
        ):
            bridge._run_async_operation(awaitable, description="start")

        mock_run_sync.assert_called_once_with(awaitable)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
