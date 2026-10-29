"""Unit tests for InMemoryEventBackend per-topic drop counters and alert threshold (Item 5)."""

from __future__ import annotations

from unittest.mock import MagicMock


from victor.core.events.backends import InMemoryEventBackend, ObservabilityBus
from victor.core.events.protocols import MessagingEvent


def _event(topic: str = "test.topic") -> MessagingEvent:
    return MessagingEvent(topic=topic, data={})


def _make_backend(
    *,
    maxsize: int = 2,
    drop_alert_threshold: int = 2,
    drop_alert_callback=None,
) -> InMemoryEventBackend:
    backend = InMemoryEventBackend(
        queue_maxsize=maxsize,
        drop_alert_threshold=drop_alert_threshold,
        drop_alert_callback=drop_alert_callback,
    )
    return backend


# ---------------------------------------------------------------------------
# Alert fires when threshold crossed
# ---------------------------------------------------------------------------


class TestDropAlertFires:
    async def test_alert_fires_when_threshold_crossed(self):
        calls = []
        backend = _make_backend(maxsize=1, drop_alert_threshold=2, drop_alert_callback=calls.append)
        await backend.connect()
        # Fill the queue
        await backend.publish(_event("a"))
        # Drops #1 and #2 → threshold reached on #2
        await backend.publish(_event("b"))
        await backend.publish(_event("c"))
        assert len(calls) >= 1
        payload = calls[0]
        assert "total_drops" in payload
        assert payload["total_drops"] >= 2
        await backend.disconnect()

    async def test_no_alert_without_callback(self):
        """Should not raise even with many drops when no callback is set."""
        backend = _make_backend(maxsize=1, drop_alert_threshold=1, drop_alert_callback=None)
        await backend.connect()
        await backend.publish(_event())
        for _ in range(5):
            await backend.publish(_event())  # all drop silently
        await backend.disconnect()


# ---------------------------------------------------------------------------
# Alert is dampened (fires once per threshold block)
# ---------------------------------------------------------------------------


class TestAlertDampening:
    async def test_alert_dampened_after_first_fire(self):
        calls = []
        backend = _make_backend(maxsize=1, drop_alert_threshold=2, drop_alert_callback=calls.append)
        await backend.connect()
        await backend.publish(_event("fill"))
        # Generate 10 drops — threshold=2 so alert fires at 2, 4, 6, 8, 10
        for _ in range(10):
            await backend.publish(_event("overflow"))
        # Must NOT fire once per drop — should be ~5 calls, not 10
        assert len(calls) < 10
        # Must fire at least once
        assert len(calls) >= 1
        await backend.disconnect()


# ---------------------------------------------------------------------------
# Per-topic drop tracking
# ---------------------------------------------------------------------------


class TestPerTopicDrops:
    async def test_per_topic_drops_tracked(self):
        backend = _make_backend(maxsize=1, drop_alert_threshold=100)
        await backend.connect()
        await backend.publish(_event("fill"))
        await backend.publish(_event("topic.a"))
        await backend.publish(_event("topic.b"))
        await backend.publish(_event("topic.a"))
        stats = backend.get_queue_pressure_stats()
        per_topic = stats["per_topic_drops"]
        assert per_topic.get("topic.a", 0) == 2
        assert per_topic.get("topic.b", 0) == 1
        await backend.disconnect()

    async def test_get_queue_pressure_stats_includes_per_topic(self):
        backend = _make_backend(maxsize=1)
        await backend.connect()
        await backend.publish(_event())
        await backend.publish(_event())  # drop
        stats = backend.get_queue_pressure_stats()
        assert "per_topic_drops" in stats
        await backend.disconnect()


# ---------------------------------------------------------------------------
# ObservabilityBus.register_drop_alert_handler
# ---------------------------------------------------------------------------


class TestObservabilityBusAlertWiring:
    async def test_register_handler_wires_to_backend(self):
        handler = MagicMock()
        inner = _make_backend(maxsize=1, drop_alert_threshold=1)
        await inner.connect()
        bus = ObservabilityBus(backend=inner)
        bus.register_drop_alert_handler(handler, threshold=5)
        assert inner._drop_alert_callback is handler
        assert inner._drop_alert_threshold == 5
        await inner.disconnect()

    async def test_register_handler_noop_for_unsupported_backend(self):
        """Should not raise when backend lacks _drop_alert_callback."""
        mock_backend = MagicMock()
        del mock_backend._drop_alert_callback  # ensure hasattr returns False
        bus = ObservabilityBus(backend=mock_backend)
        bus.register_drop_alert_handler(lambda p: None, threshold=10)  # must not raise

    async def test_alert_payload_has_required_fields(self):
        calls = []
        backend = _make_backend(maxsize=1, drop_alert_threshold=1, drop_alert_callback=calls.append)
        await backend.connect()
        await backend.publish(_event("fill"))
        await backend.publish(_event("overflow"))
        assert calls, "alert callback was never called"
        payload = calls[0]
        for field in (
            "total_drops",
            "per_topic_drops",
            "overflow_policy",
            "queue_depth",
            "queue_maxsize",
        ):
            assert field in payload, f"missing field: {field}"
        await backend.disconnect()
