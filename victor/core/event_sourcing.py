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

"""Event Sourcing pattern for audit trail and state replay.

This module provides a comprehensive event sourcing implementation that:
- Stores all state changes as immutable events
- Supports rebuilding state from event history
- Provides audit trail capabilities
- Enables temporal queries and debugging

Design Patterns:
- Event Sourcing: Store state changes as events
- Aggregate Root: Encapsulates domain logic
- Event Store: Persistent event storage
- Snapshot: Periodic state snapshots for performance

Example:
    from victor.core.event_sourcing import (
        DomainEvent,
        EventStore,
        Aggregate,
        InMemoryEventStore,
    )

    # Define events
    @dataclass
    class TaskCreated(DomainEvent):
        task_id: str
        prompt: str

    @dataclass
    class TaskCompleted(DomainEvent):
        task_id: str
        result: str

    # Create aggregate
    class TaskAggregate(Aggregate):
        def apply_TaskCreated(self, event):
            self.task_id = event.task_id
            self.prompt = event.prompt
            self.status = "created"

        def apply_TaskCompleted(self, event):
            self.result = event.result
            self.status = "completed"

    # Use event store
    store = InMemoryEventStore()
    task = TaskAggregate("task-123")
    task.apply(TaskCreated(task_id="task-123", prompt="Hello"))
    await store.save(task)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Event Base Classes
# =============================================================================


# Event type registry for deserialization (module-level to avoid dataclass issues)
_EVENT_REGISTRY: Dict[str, Type["DomainEvent"]] = {}


@dataclass
class DomainEvent:
    """Base class for all domain events.

    Events are immutable records of state changes that have occurred.
    They should be named in past tense (e.g., TaskCreated, FileModified).

    Event subclasses are automatically registered for deserialization.
    Use DomainEvent.from_dict() to deserialize events with proper type resolution.
    """

    # Metadata
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register event subclasses."""
        super().__init_subclass__(**kwargs)
        _EVENT_REGISTRY[cls.__name__] = cls

    @classmethod
    def register(cls, event_class: Type["DomainEvent"]) -> Type["DomainEvent"]:
        """Explicitly register an event type.

        Normally event types are auto-registered, but this can be used
        for dynamic registration or decorator patterns.

        Args:
            event_class: Event class to register

        Returns:
            The registered class (for decorator use)
        """
        _EVENT_REGISTRY[event_class.__name__] = event_class
        return event_class

    @classmethod
    def get_registered_types(cls) -> Dict[str, Type["DomainEvent"]]:
        """Get all registered event types."""
        return dict(_EVENT_REGISTRY)

    @property
    def event_type(self) -> str:
        """Get event type name."""
        return self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "metadata": self.metadata,
            "data": self._get_data(),
        }

    def _get_data(self) -> Dict[str, Any]:
        """Get event-specific data."""
        result = {}
        for key, value in self.__dict__.items():
            if key not in (
                "event_id",
                "timestamp",
                "version",
                "correlation_id",
                "causation_id",
                "metadata",
            ):
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainEvent":
        """Deserialize event from dictionary.

        Uses the event type registry to instantiate the correct event class.
        If the event type is not registered, creates a base DomainEvent.

        Args:
            data: Dictionary with event data including 'event_type' and 'data'

        Returns:
            DomainEvent instance of the appropriate type
        """
        event_type = data.get("event_type", "DomainEvent")
        event_data = data.get("data", {})

        # Look up the correct class from registry
        event_cls = _EVENT_REGISTRY.get(event_type, cls)

        return event_cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data.get("version", 1),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            metadata=data.get("metadata", {}),
            **event_data,
        )


@dataclass
class EventEnvelope:
    """Wrapper for storing events with stream metadata."""

    stream_id: str
    stream_version: int
    event: DomainEvent
    stored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize envelope to dictionary."""
        return {
            "stream_id": self.stream_id,
            "stream_version": self.stream_version,
            "stored_at": self.stored_at.isoformat(),
            "event": self.event.to_dict(),
        }


# =============================================================================
# Aggregate Root Pattern
# =============================================================================


T = TypeVar("T", bound="Aggregate")


class Aggregate(ABC):
    """Base class for aggregate roots.

    Aggregates encapsulate domain logic and produce events.
    They can be rebuilt from event history.

    Example:
        class OrderAggregate(Aggregate):
            def __init__(self, aggregate_id: str):
                super().__init__(aggregate_id)
                self.items = []
                self.total = 0

            def apply_ItemAdded(self, event):
                self.items.append(event.item)
                self.total += event.price

            def add_item(self, item: str, price: float):
                self.apply(ItemAdded(item=item, price=price))
    """

    def __init__(self, aggregate_id: str) -> None:
        """Initialize aggregate.

        Args:
            aggregate_id: Unique identifier for this aggregate.
        """
        self._aggregate_id = aggregate_id
        self._version = 0
        self._uncommitted_events: List[DomainEvent] = []

    @property
    def aggregate_id(self) -> str:
        """Get aggregate ID."""
        return self._aggregate_id

    @property
    def version(self) -> int:
        """Get current version."""
        return self._version

    @property
    def uncommitted_events(self) -> List[DomainEvent]:
        """Get events that haven't been persisted."""
        return self._uncommitted_events.copy()

    def apply(self, event: DomainEvent) -> None:
        """Apply an event to the aggregate.

        This both mutates state and records the event.

        Args:
            event: Event to apply.
        """
        self._apply_event(event)
        self._uncommitted_events.append(event)
        self._version += 1

    def _apply_event(self, event: DomainEvent) -> None:
        """Apply event to internal state.

        Args:
            event: Event to apply.
        """
        handler_name = f"apply_{event.event_type}"
        handler = getattr(self, handler_name, None)

        if handler is None:
            logger.warning(f"No handler for event type: {event.event_type}")
            return

        handler(event)

    def load_from_history(self, events: List[DomainEvent]) -> None:
        """Rebuild aggregate state from event history.

        Args:
            events: Historical events in order.
        """
        for event in events:
            self._apply_event(event)
            self._version += 1

    def clear_uncommitted_events(self) -> None:
        """Clear uncommitted events after save."""
        self._uncommitted_events.clear()

    def get_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of current state.

        Returns:
            State snapshot dictionary.
        """
        return {
            "aggregate_id": self._aggregate_id,
            "version": self._version,
            "state": self._get_state(),
        }

    def _get_state(self) -> Dict[str, Any]:
        """Get internal state for snapshot.

        Returns:
            State dictionary.
        """
        state = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                state[key] = value
        return state

    def load_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Load aggregate from snapshot.

        Args:
            snapshot: State snapshot dictionary.
        """
        self._aggregate_id = snapshot["aggregate_id"]
        self._version = snapshot["version"]

        for key, value in snapshot.get("state", {}).items():
            setattr(self, key, value)


# =============================================================================
# Event Store Interface
# =============================================================================


class EventStore(ABC):
    """Abstract base class for event stores.

    Event stores are responsible for persisting and retrieving events.
    """

    @abstractmethod
    async def append(
        self,
        stream_id: str,
        events: List[DomainEvent],
        expected_version: Optional[int] = None,
    ) -> int:
        """Append events to a stream.

        Args:
            stream_id: Stream identifier.
            events: Events to append.
            expected_version: Expected stream version for optimistic concurrency.

        Returns:
            New stream version.

        Raises:
            ConcurrencyError: If expected_version doesn't match.
        """
        pass

    @abstractmethod
    async def read(
        self,
        stream_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None,
    ) -> List[EventEnvelope]:
        """Read events from a stream.

        Args:
            stream_id: Stream identifier.
            from_version: Starting version (inclusive).
            to_version: Ending version (inclusive).

        Returns:
            List of event envelopes.
        """
        pass

    @abstractmethod
    async def read_all(
        self,
        from_position: int = 0,
        limit: int = 100,
    ) -> List[EventEnvelope]:
        """Read all events across streams.

        Args:
            from_position: Global position to start from.
            limit: Maximum events to return.

        Returns:
            List of event envelopes.
        """
        pass

    @abstractmethod
    async def get_stream_version(self, stream_id: str) -> int:
        """Get current stream version.

        Args:
            stream_id: Stream identifier.

        Returns:
            Current version or 0 if stream doesn't exist.
        """
        pass

    async def save(self, aggregate: Aggregate) -> None:
        """Save aggregate's uncommitted events.

        Args:
            aggregate: Aggregate to save.
        """
        events = aggregate.uncommitted_events
        if not events:
            return

        expected_version = aggregate.version - len(events)
        await self.append(
            aggregate.aggregate_id,
            events,
            expected_version if expected_version > 0 else None,
        )
        aggregate.clear_uncommitted_events()

    async def load(
        self,
        aggregate_type: Type[T],
        aggregate_id: str,
    ) -> Optional[T]:
        """Load aggregate from event history.

        Args:
            aggregate_type: Aggregate class.
            aggregate_id: Aggregate ID.

        Returns:
            Loaded aggregate or None if not found.
        """
        envelopes = await self.read(aggregate_id)
        if not envelopes:
            return None

        aggregate = aggregate_type(aggregate_id)
        events = [env.event for env in envelopes]
        aggregate.load_from_history(events)
        return aggregate


class ConcurrencyError(Exception):
    """Raised when optimistic concurrency check fails."""

    def __init__(
        self,
        stream_id: str,
        expected_version: int,
        actual_version: int,
    ) -> None:
        """Initialize error.

        Args:
            stream_id: Stream identifier.
            expected_version: Expected version.
            actual_version: Actual version.
        """
        super().__init__(
            f"Concurrency conflict for stream {stream_id}: "
            f"expected {expected_version}, actual {actual_version}"
        )
        self.stream_id = stream_id
        self.expected_version = expected_version
        self.actual_version = actual_version


# =============================================================================
# In-Memory Event Store
# =============================================================================


class InMemoryEventStore(EventStore):
    """In-memory event store for testing and development.

    Thread-safe implementation using asyncio locks.
    """

    def __init__(self) -> None:
        """Initialize in-memory store."""
        self._streams: Dict[str, List[EventEnvelope]] = {}
        self._all_events: List[EventEnvelope] = []
        self._lock = asyncio.Lock()

    async def append(
        self,
        stream_id: str,
        events: List[DomainEvent],
        expected_version: Optional[int] = None,
    ) -> int:
        """Append events to stream."""
        async with self._lock:
            current_version = len(self._streams.get(stream_id, []))

            if expected_version is not None and current_version != expected_version:
                raise ConcurrencyError(stream_id, expected_version, current_version)

            if stream_id not in self._streams:
                self._streams[stream_id] = []

            for event in events:
                current_version += 1
                envelope = EventEnvelope(
                    stream_id=stream_id,
                    stream_version=current_version,
                    event=event,
                )
                self._streams[stream_id].append(envelope)
                self._all_events.append(envelope)

            return current_version

    async def read(
        self,
        stream_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None,
    ) -> List[EventEnvelope]:
        """Read events from stream."""
        async with self._lock:
            if stream_id not in self._streams:
                return []

            events = self._streams[stream_id]

            # Filter by version
            result = []
            for env in events:
                if env.stream_version < from_version:
                    continue
                if to_version is not None and env.stream_version > to_version:
                    break
                result.append(env)

            return result

    async def read_all(
        self,
        from_position: int = 0,
        limit: int = 100,
    ) -> List[EventEnvelope]:
        """Read all events."""
        async with self._lock:
            return self._all_events[from_position : from_position + limit]

    async def get_stream_version(self, stream_id: str) -> int:
        """Get stream version."""
        async with self._lock:
            return len(self._streams.get(stream_id, []))

    def clear(self) -> None:
        """Clear all events (for testing)."""
        self._streams.clear()
        self._all_events.clear()


# =============================================================================
# SQLite Event Store
# =============================================================================


class SQLiteEventStore(EventStore):
    """SQLite-based event store for persistence.

    Provides durable event storage with atomic operations.
    All blocking sqlite3 I/O is offloaded to a dedicated thread via
    ThreadPoolExecutor to avoid blocking the asyncio event loop.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        """Initialize SQLite store.

        Args:
            db_path: Path to SQLite database.
        """
        self._db_path = Path(db_path)
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlite-event-store")
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    global_position INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream_id TEXT NOT NULL,
                    stream_version INTEGER NOT NULL,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    stored_at TEXT NOT NULL,
                    UNIQUE(stream_id, stream_version)
                )
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_stream
                ON events(stream_id, stream_version)
                """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    stream_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    snapshot_data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """)
            conn.commit()

    async def close(self) -> None:
        """Shut down the executor. Call when done with the store."""
        self._executor.shutdown(wait=True)

    def _append_sync(
        self,
        stream_id: str,
        events: List[DomainEvent],
        expected_version: Optional[int],
    ) -> int:
        """Synchronous append — runs in executor thread."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT COALESCE(MAX(stream_version), 0) " "FROM events WHERE stream_id = ?",
                (stream_id,),
            )
            current_version = cursor.fetchone()[0]

            if expected_version is not None and current_version != expected_version:
                raise ConcurrencyError(stream_id, expected_version, current_version)

            for event in events:
                current_version += 1
                conn.execute(
                    "INSERT INTO events "
                    "(stream_id, stream_version, event_id, event_type, event_data, stored_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        stream_id,
                        current_version,
                        event.event_id,
                        event.event_type,
                        json.dumps(event.to_dict()),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

            conn.commit()
            return current_version

    def _read_sync(
        self,
        stream_id: str,
        from_version: int,
        to_version: Optional[int],
    ) -> List[EventEnvelope]:
        """Synchronous read — runs in executor thread."""
        with sqlite3.connect(self._db_path) as conn:
            if to_version is not None:
                cursor = conn.execute(
                    "SELECT stream_id, stream_version, event_data, stored_at "
                    "FROM events "
                    "WHERE stream_id = ? AND stream_version >= ? AND stream_version <= ? "
                    "ORDER BY stream_version",
                    (stream_id, from_version, to_version),
                )
            else:
                cursor = conn.execute(
                    "SELECT stream_id, stream_version, event_data, stored_at "
                    "FROM events "
                    "WHERE stream_id = ? AND stream_version >= ? "
                    "ORDER BY stream_version",
                    (stream_id, from_version),
                )

            return self._rows_to_envelopes(cursor)

    def _read_all_sync(self, from_position: int, limit: int) -> List[EventEnvelope]:
        """Synchronous read_all — runs in executor thread."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT stream_id, stream_version, event_data, stored_at "
                "FROM events WHERE global_position > ? "
                "ORDER BY global_position LIMIT ?",
                (from_position, limit),
            )
            return self._rows_to_envelopes(cursor)

    def _get_stream_version_sync(self, stream_id: str) -> int:
        """Synchronous get_stream_version — runs in executor thread."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT COALESCE(MAX(stream_version), 0) " "FROM events WHERE stream_id = ?",
                (stream_id,),
            )
            return cursor.fetchone()[0]

    def _save_snapshot_sync(self, stream_id: str, version: int, snapshot: Dict[str, Any]) -> None:
        """Synchronous save_snapshot — runs in executor thread."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO snapshots "
                "(stream_id, version, snapshot_data, created_at) VALUES (?, ?, ?, ?)",
                (
                    stream_id,
                    version,
                    json.dumps(snapshot),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

    def _load_snapshot_sync(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous load_snapshot — runs in executor thread."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT version, snapshot_data FROM snapshots WHERE stream_id = ?",
                (stream_id,),
            )
            row = cursor.fetchone()
            if row:
                return {"version": row[0], "data": json.loads(row[1])}
            return None

    @staticmethod
    def _rows_to_envelopes(cursor) -> List[EventEnvelope]:
        """Convert cursor rows to EventEnvelope list with error handling."""
        result = []
        for row in cursor:
            try:
                event_data = json.loads(row[2])
                event = DomainEvent.from_dict(event_data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("Skipping corrupt event in stream %s v%s: %s", row[0], row[1], e)
                continue
            envelope = EventEnvelope(
                stream_id=row[0],
                stream_version=row[1],
                event=event,
                stored_at=datetime.fromisoformat(row[3]),
            )
            result.append(envelope)
        return result

    async def append(
        self,
        stream_id: str,
        events: List[DomainEvent],
        expected_version: Optional[int] = None,
    ) -> int:
        """Append events to stream."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._append_sync, stream_id, events, expected_version
            )

    async def read(
        self,
        stream_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None,
    ) -> List[EventEnvelope]:
        """Read events from stream."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._read_sync, stream_id, from_version, to_version
            )

    async def read_all(
        self,
        from_position: int = 0,
        limit: int = 100,
    ) -> List[EventEnvelope]:
        """Read all events."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._read_all_sync, from_position, limit
            )

    async def get_stream_version(self, stream_id: str) -> int:
        """Get stream version."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._get_stream_version_sync, stream_id
            )

    async def save_snapshot(
        self,
        stream_id: str,
        version: int,
        snapshot: Dict[str, Any],
    ) -> None:
        """Save aggregate snapshot.

        Args:
            stream_id: Stream identifier.
            version: Version at snapshot.
            snapshot: Snapshot data.
        """
        async with self._lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor, self._save_snapshot_sync, stream_id, version, snapshot
            )

    async def load_snapshot(
        self,
        stream_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Load aggregate snapshot.

        Args:
            stream_id: Stream identifier.

        Returns:
            Snapshot data or None if not found.
        """
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._load_snapshot_sync, stream_id)


# =============================================================================
# Event Handlers and Projections
# =============================================================================


EventHandler = Callable[[DomainEvent], None]
AsyncEventHandler = Callable[[DomainEvent], Any]


class EventDispatcher:
    """Dispatches events to registered handlers.

    Implements the Observer pattern for event distribution.
    """

    def __init__(self) -> None:
        """Initialize dispatcher."""
        self._handlers: Dict[str, List[AsyncEventHandler]] = {}
        self._all_handlers: List[AsyncEventHandler] = []

    def subscribe(
        self,
        event_type: str,
        handler: AsyncEventHandler,
    ) -> None:
        """Subscribe to specific event type.

        Args:
            event_type: Event type to handle.
            handler: Handler function.
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: AsyncEventHandler) -> None:
        """Subscribe to all events.

        Args:
            handler: Handler function.
        """
        self._all_handlers.append(handler)

    async def _safe_execute_handler(self, handler, event: DomainEvent) -> None:
        """Safely execute a single event handler.

        Args:
            handler: Handler function (sync or async)
            event: Event to handle
        """
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Event handler error: {e}")

    async def dispatch(self, event: DomainEvent) -> None:
        """Dispatch event to handlers in parallel.

        Performance: Handlers execute in parallel using asyncio.gather().
        Errors in one handler don't prevent other handlers from running.

        Args:
            event: Event to dispatch.
        """
        # Combine type-specific and all-event handlers
        handlers = self._handlers.get(event.event_type, []) + self._all_handlers

        # Execute all handlers in parallel
        if handlers:
            await asyncio.gather(
                *[self._safe_execute_handler(handler, event) for handler in handlers],
                return_exceptions=True,  # Don't fail on handler errors
            )


class Projection(ABC):
    """Base class for read model projections.

    Projections build read-optimized views from events.
    """

    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle an event and update projection.

        Args:
            event: Event to handle.
        """
        pass

    @abstractmethod
    async def rebuild(self, events: List[DomainEvent]) -> None:
        """Rebuild projection from event history.

        Args:
            events: All events to replay.
        """
        pass


# =============================================================================
# Event Sourced Repository
# =============================================================================


class EventSourcedRepository(Generic[T]):
    """Repository for event-sourced aggregates.

    Provides a clean interface for loading and saving aggregates.
    """

    def __init__(
        self,
        event_store: EventStore,
        aggregate_type: Type[T],
        snapshot_interval: int = 100,
    ) -> None:
        """Initialize repository.

        Args:
            event_store: Event store to use.
            aggregate_type: Aggregate class.
            snapshot_interval: Events between snapshots.
        """
        self._store = event_store
        self._aggregate_type = aggregate_type
        self._snapshot_interval = snapshot_interval

    async def get(self, aggregate_id: str) -> Optional[T]:
        """Get aggregate by ID.

        Args:
            aggregate_id: Aggregate ID.

        Returns:
            Aggregate or None if not found.
        """
        return await self._store.load(self._aggregate_type, aggregate_id)

    async def save(self, aggregate: T) -> None:
        """Save aggregate.

        Args:
            aggregate: Aggregate to save.
        """
        await self._store.save(aggregate)

        # Create snapshot if needed
        if (
            isinstance(self._store, SQLiteEventStore)
            and aggregate.version % self._snapshot_interval == 0
        ):
            snapshot = aggregate.get_snapshot()
            await self._store.save_snapshot(
                aggregate.aggregate_id,
                aggregate.version,
                snapshot,
            )

    async def exists(self, aggregate_id: str) -> bool:
        """Check if aggregate exists.

        Args:
            aggregate_id: Aggregate ID.

        Returns:
            True if exists.
        """
        version = await self._store.get_stream_version(aggregate_id)
        return version > 0


# =============================================================================
# Common Domain Events
# =============================================================================


@dataclass
class TaskStartedEvent(DomainEvent):
    """Event for when a task is started."""

    task_id: str = ""
    prompt: str = ""
    provider: str = ""
    model: str = ""


@dataclass
class TaskCompletedEvent(DomainEvent):
    """Event for when a task completes."""

    task_id: str = ""
    result: str = ""
    duration_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class TaskFailedEvent(DomainEvent):
    """Event for when a task fails."""

    task_id: str = ""
    error: str = ""
    error_type: str = ""


@dataclass
class ToolCalledEvent(DomainEvent):
    """Event for when a tool is called."""

    task_id: str = ""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultEvent(DomainEvent):
    """Event for tool result."""

    task_id: str = ""
    tool_name: str = ""
    success: bool = True
    result: str = ""
    duration_ms: float = 0.0


@dataclass
class StateChangedEvent(DomainEvent):
    """Event for state machine transitions."""

    task_id: str = ""
    from_state: str = ""
    to_state: str = ""
    reason: str = ""


@dataclass
class ContextCompactedEvent(DomainEvent):
    """Event emitted when conversation context is compacted."""

    messages_removed: int = 0
    chars_freed: int = 0
    trigger: str = ""
    summary: str = ""


@dataclass
class SessionResumedEvent(DomainEvent):
    """Event emitted when a session is resumed from persistence."""

    session_id: str = ""
    messages_restored: int = 0
    ledger_entries_restored: int = 0
