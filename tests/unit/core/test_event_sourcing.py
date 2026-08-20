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

"""Tests for event sourcing module."""

import pytest
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from victor.core.event_sourcing import (
    Aggregate,
    ConcurrencyError,
    DomainEvent,
    EventDispatcher,
    EventEnvelope,
    EventSourcedRepository,
    InMemoryEventStore,
    Projection,
    SQLiteEventStore,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
    ToolCalledEvent,
    ToolResultEvent,
    StateChangedEvent,
)

# =============================================================================
# Test Events
# =============================================================================


@dataclass
class OrderCreated(DomainEvent):
    """Test event for order creation."""

    order_id: str = ""
    customer: str = ""


@dataclass
class ItemAdded(DomainEvent):
    """Test event for adding item."""

    item: str = ""
    quantity: int = 1
    price: float = 0.0


@dataclass
class OrderCompleted(DomainEvent):
    """Test event for order completion."""

    total: float = 0.0


# =============================================================================
# Test Aggregate
# =============================================================================


class OrderAggregate(Aggregate):
    """Test aggregate for orders."""

    def __init__(self, aggregate_id: str):
        super().__init__(aggregate_id)
        self.customer = ""
        self.items: List[Dict] = []
        self.total = 0.0
        self.status = "pending"

    def apply_OrderCreated(self, event: OrderCreated):
        self.customer = event.customer

    def apply_ItemAdded(self, event: ItemAdded):
        self.items.append(
            {
                "item": event.item,
                "quantity": event.quantity,
                "price": event.price,
            }
        )
        self.total += event.price * event.quantity

    def apply_OrderCompleted(self, event: OrderCompleted):
        self.status = "completed"
        self.total = event.total

    def create(self, customer: str):
        """Create order."""
        self.apply(OrderCreated(order_id=self.aggregate_id, customer=customer))

    def add_item(self, item: str, quantity: int, price: float):
        """Add item to order."""
        self.apply(ItemAdded(item=item, quantity=quantity, price=price))

    def complete(self):
        """Complete order."""
        self.apply(OrderCompleted(total=self.total))


# =============================================================================
# Event Tests
# =============================================================================


class TestDomainEvent:
    """Tests for DomainEvent base class."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = DomainEvent()

        assert event.event_id is not None
        assert event.timestamp is not None
        assert event.version == 1

    def test_event_type(self):
        """Test event type property."""
        event = OrderCreated(order_id="123", customer="Test")

        assert event.event_type == "OrderCreated"

    def test_to_dict(self):
        """Test serialization."""
        event = OrderCreated(order_id="123", customer="Test")
        data = event.to_dict()

        assert data["event_type"] == "OrderCreated"
        assert data["data"]["order_id"] == "123"
        assert data["data"]["customer"] == "Test"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "event_id": "test-id",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": 1,
            "data": {"order_id": "123", "customer": "Test"},
        }

        event = OrderCreated.from_dict(data)

        assert event.event_id == "test-id"
        assert event.order_id == "123"
        assert event.customer == "Test"

    def test_correlation_id(self):
        """Test correlation ID."""
        event = DomainEvent(correlation_id="corr-123")

        assert event.correlation_id == "corr-123"


class TestEventEnvelope:
    """Tests for EventEnvelope."""

    def test_envelope_creation(self):
        """Test envelope creation."""
        event = OrderCreated(order_id="123")
        envelope = EventEnvelope(
            stream_id="order-123",
            stream_version=1,
            event=event,
        )

        assert envelope.stream_id == "order-123"
        assert envelope.stream_version == 1
        assert envelope.event == event

    def test_to_dict(self):
        """Test envelope serialization."""
        event = OrderCreated(order_id="123")
        envelope = EventEnvelope(
            stream_id="order-123",
            stream_version=1,
            event=event,
        )

        data = envelope.to_dict()

        assert data["stream_id"] == "order-123"
        assert data["stream_version"] == 1
        assert "event" in data


# =============================================================================
# Aggregate Tests
# =============================================================================


class TestAggregate:
    """Tests for Aggregate base class."""

    def test_aggregate_creation(self):
        """Test aggregate creation."""
        order = OrderAggregate("order-123")

        assert order.aggregate_id == "order-123"
        assert order.version == 0

    def test_apply_event(self):
        """Test applying events."""
        order = OrderAggregate("order-123")
        order.create(customer="Test Customer")

        assert order.customer == "Test Customer"
        assert order.version == 1
        assert len(order.uncommitted_events) == 1

    def test_apply_multiple_events(self):
        """Test applying multiple events."""
        order = OrderAggregate("order-123")
        order.create(customer="Test")
        order.add_item("Widget", 2, 10.0)
        order.add_item("Gadget", 1, 25.0)

        assert order.customer == "Test"
        assert len(order.items) == 2
        assert order.total == 45.0
        assert order.version == 3

    def test_load_from_history(self):
        """Test rebuilding from history."""
        # Create events
        events = [
            OrderCreated(order_id="123", customer="Test"),
            ItemAdded(item="Widget", quantity=2, price=10.0),
            ItemAdded(item="Gadget", quantity=1, price=25.0),
        ]

        # Rebuild aggregate
        order = OrderAggregate("order-123")
        order.load_from_history(events)

        assert order.customer == "Test"
        assert len(order.items) == 2
        assert order.total == 45.0
        assert order.version == 3
        assert len(order.uncommitted_events) == 0

    def test_clear_uncommitted_events(self):
        """Test clearing uncommitted events."""
        order = OrderAggregate("order-123")
        order.create(customer="Test")

        assert len(order.uncommitted_events) == 1

        order.clear_uncommitted_events()

        assert len(order.uncommitted_events) == 0

    def test_get_snapshot(self):
        """Test getting snapshot."""
        order = OrderAggregate("order-123")
        order.create(customer="Test")
        order.add_item("Widget", 1, 10.0)

        snapshot = order.get_snapshot()

        assert snapshot["aggregate_id"] == "order-123"
        assert snapshot["version"] == 2
        assert snapshot["state"]["customer"] == "Test"

    def test_load_from_snapshot(self):
        """Test loading from snapshot."""
        snapshot = {
            "aggregate_id": "order-123",
            "version": 5,
            "state": {
                "customer": "Test",
                "items": [{"item": "Widget", "quantity": 1, "price": 10.0}],
                "total": 10.0,
                "status": "pending",
            },
        }

        order = OrderAggregate("order-123")
        order.load_from_snapshot(snapshot)

        assert order.customer == "Test"
        assert order.version == 5
        assert len(order.items) == 1


# =============================================================================
# InMemoryEventStore Tests
# =============================================================================


class TestInMemoryEventStore:
    """Tests for InMemoryEventStore."""

    @pytest.mark.asyncio
    async def test_append_events(self):
        """Test appending events."""
        store = InMemoryEventStore()

        events = [
            OrderCreated(order_id="123", customer="Test"),
            ItemAdded(item="Widget", quantity=1, price=10.0),
        ]

        version = await store.append("order-123", events)

        assert version == 2

    @pytest.mark.asyncio
    async def test_read_events(self):
        """Test reading events."""
        store = InMemoryEventStore()

        events = [
            OrderCreated(order_id="123", customer="Test"),
            ItemAdded(item="Widget", quantity=1, price=10.0),
        ]
        await store.append("order-123", events)

        envelopes = await store.read("order-123")

        assert len(envelopes) == 2
        assert envelopes[0].stream_version == 1
        assert envelopes[1].stream_version == 2

    @pytest.mark.asyncio
    async def test_read_from_version(self):
        """Test reading from specific version."""
        store = InMemoryEventStore()

        events = [
            OrderCreated(order_id="123", customer="Test"),
            ItemAdded(item="Widget", quantity=1, price=10.0),
            ItemAdded(item="Gadget", quantity=1, price=20.0),
        ]
        await store.append("order-123", events)

        envelopes = await store.read("order-123", from_version=2)

        assert len(envelopes) == 2

    @pytest.mark.asyncio
    async def test_read_nonexistent_stream(self):
        """Test reading nonexistent stream."""
        store = InMemoryEventStore()

        envelopes = await store.read("nonexistent")

        assert len(envelopes) == 0

    @pytest.mark.asyncio
    async def test_get_stream_version(self):
        """Test getting stream version."""
        store = InMemoryEventStore()

        events = [
            OrderCreated(order_id="123", customer="Test"),
            ItemAdded(item="Widget", quantity=1, price=10.0),
        ]
        await store.append("order-123", events)

        version = await store.get_stream_version("order-123")

        assert version == 2

    @pytest.mark.asyncio
    async def test_optimistic_concurrency(self):
        """Test optimistic concurrency check."""
        store = InMemoryEventStore()

        await store.append("order-123", [OrderCreated(order_id="123")])

        # This should fail - expected version 0, but actual is 1
        with pytest.raises(ConcurrencyError):
            await store.append(
                "order-123",
                [ItemAdded(item="Widget")],
                expected_version=0,
            )

    @pytest.mark.asyncio
    async def test_read_all(self):
        """Test reading all events."""
        store = InMemoryEventStore()

        await store.append("order-1", [OrderCreated(order_id="1")])
        await store.append("order-2", [OrderCreated(order_id="2")])

        all_events = await store.read_all()

        assert len(all_events) == 2

    @pytest.mark.asyncio
    async def test_save_aggregate(self):
        """Test saving aggregate."""
        store = InMemoryEventStore()

        order = OrderAggregate("order-123")
        order.create(customer="Test")
        order.add_item("Widget", 1, 10.0)

        await store.save(order)

        assert len(order.uncommitted_events) == 0

        version = await store.get_stream_version("order-123")
        assert version == 2

    @pytest.mark.asyncio
    async def test_load_aggregate(self):
        """Test loading aggregate."""
        store = InMemoryEventStore()

        # Save aggregate
        order1 = OrderAggregate("order-123")
        order1.create(customer="Test")
        order1.add_item("Widget", 1, 10.0)
        await store.save(order1)

        # Load aggregate
        order2 = await store.load(OrderAggregate, "order-123")

        assert order2 is not None
        assert order2.customer == "Test"
        assert len(order2.items) == 1
        assert order2.version == 2


# =============================================================================
# SQLiteEventStore Tests
# =============================================================================


class TestSQLiteEventStore:
    """Tests for SQLiteEventStore."""

    @pytest.mark.asyncio
    async def test_append_and_read(self):
        """Test basic append and read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "events.db"
            store = SQLiteEventStore(db_path)

            events = [
                OrderCreated(order_id="123", customer="Test"),
                ItemAdded(item="Widget", quantity=1, price=10.0),
            ]

            version = await store.append("order-123", events)
            assert version == 2

            envelopes = await store.read("order-123")
            assert len(envelopes) == 2

    @pytest.mark.asyncio
    async def test_persistence(self):
        """Test data persists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "events.db"

            # Write events
            store1 = SQLiteEventStore(db_path)
            await store1.append("order-123", [OrderCreated(order_id="123")])

            # Read from new store
            store2 = SQLiteEventStore(db_path)
            envelopes = await store2.read("order-123")

            assert len(envelopes) == 1

    @pytest.mark.asyncio
    async def test_optimistic_concurrency(self):
        """Test optimistic concurrency in SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "events.db"
            store = SQLiteEventStore(db_path)

            await store.append("order-123", [OrderCreated(order_id="123")])

            with pytest.raises(ConcurrencyError):
                await store.append(
                    "order-123",
                    [ItemAdded(item="Widget")],
                    expected_version=0,
                )

    @pytest.mark.asyncio
    async def test_snapshots(self):
        """Test snapshot save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "events.db"
            store = SQLiteEventStore(db_path)

            snapshot = {
                "customer": "Test",
                "items": [{"item": "Widget"}],
            }

            await store.save_snapshot("order-123", 5, snapshot)

            loaded = await store.load_snapshot("order-123")

            assert loaded is not None
            assert loaded["version"] == 5
            assert loaded["data"]["customer"] == "Test"


# =============================================================================
# EventDispatcher Tests
# =============================================================================


class TestEventDispatcher:
    """Tests for EventDispatcher."""

    @pytest.mark.asyncio
    async def test_subscribe_and_dispatch(self):
        """Test basic subscribe and dispatch."""
        dispatcher = EventDispatcher()
        handled = []

        def handler(event):
            handled.append(event)

        dispatcher.subscribe("OrderCreated", handler)
        await dispatcher.dispatch(OrderCreated(order_id="123"))

        assert len(handled) == 1

    @pytest.mark.asyncio
    async def test_async_handler(self):
        """Test async handler."""
        dispatcher = EventDispatcher()
        handled = []

        async def handler(event):
            handled.append(event)

        dispatcher.subscribe("OrderCreated", handler)
        await dispatcher.dispatch(OrderCreated(order_id="123"))

        assert len(handled) == 1

    @pytest.mark.asyncio
    async def test_subscribe_all(self):
        """Test subscribing to all events."""
        dispatcher = EventDispatcher()
        handled = []

        def handler(event):
            handled.append(event)

        dispatcher.subscribe_all(handler)

        await dispatcher.dispatch(OrderCreated(order_id="1"))
        await dispatcher.dispatch(ItemAdded(item="Widget"))

        assert len(handled) == 2

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        """Test multiple handlers for same event."""
        dispatcher = EventDispatcher()
        results = []

        def handler1(event):
            results.append("handler1")

        def handler2(event):
            results.append("handler2")

        dispatcher.subscribe("OrderCreated", handler1)
        dispatcher.subscribe("OrderCreated", handler2)

        await dispatcher.dispatch(OrderCreated(order_id="123"))

        assert "handler1" in results
        assert "handler2" in results


# =============================================================================
# EventSourcedRepository Tests
# =============================================================================


class TestEventSourcedRepository:
    """Tests for EventSourcedRepository."""

    @pytest.mark.asyncio
    async def test_get_and_save(self):
        """Test basic get and save."""
        store = InMemoryEventStore()
        repo = EventSourcedRepository(store, OrderAggregate)

        # Create and save
        order = OrderAggregate("order-123")
        order.create(customer="Test")
        await repo.save(order)

        # Get
        loaded = await repo.get("order-123")

        assert loaded is not None
        assert loaded.customer == "Test"

    @pytest.mark.asyncio
    async def test_exists(self):
        """Test exists check."""
        store = InMemoryEventStore()
        repo = EventSourcedRepository(store, OrderAggregate)

        assert await repo.exists("order-123") is False

        order = OrderAggregate("order-123")
        order.create(customer="Test")
        await repo.save(order)

        assert await repo.exists("order-123") is True

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test getting nonexistent aggregate."""
        store = InMemoryEventStore()
        repo = EventSourcedRepository(store, OrderAggregate)

        loaded = await repo.get("nonexistent")

        assert loaded is None


# =============================================================================
# Common Event Tests
# =============================================================================


class TestCommonEvents:
    """Tests for common domain events."""

    def test_task_started_event(self):
        """Test TaskStartedEvent."""
        event = TaskStartedEvent(
            task_id="task-123",
            prompt="Hello world",
            provider="anthropic",
            model="claude-3",
        )

        assert event.event_type == "TaskStartedEvent"
        assert event.task_id == "task-123"

    def test_task_completed_event(self):
        """Test TaskCompletedEvent."""
        event = TaskCompletedEvent(
            task_id="task-123",
            result="Success",
            duration_ms=1500.0,
            tokens_used=100,
        )

        assert event.tokens_used == 100
        assert event.duration_ms == 1500.0

    def test_tool_called_event(self):
        """Test ToolCalledEvent."""
        event = ToolCalledEvent(
            task_id="task-123",
            tool_name="read",
            arguments={"path": "/tmp/test.txt"},
        )

        assert event.tool_name == "read"
        assert event.arguments["path"] == "/tmp/test.txt"

    def test_state_changed_event(self):
        """Test StateChangedEvent."""
        event = StateChangedEvent(
            task_id="task-123",
            from_state="PLANNING",
            to_state="EXECUTING",
            reason="Plan approved",
        )

        assert event.from_state == "PLANNING"
        assert event.to_state == "EXECUTING"
