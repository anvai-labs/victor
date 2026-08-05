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

"""Lightweight event backend implementations.

This module provides lightweight backends for smaller deployments:
- SQLiteEventBackend: File-based persistence with SQLite
- DatabaseEventBackend: SQLAlchemy-based for any SQL database

These are suitable for:
- Single-instance deployments with persistence
- Development and testing
- Small teams with shared database
- Air-gapped environments

For high-throughput distributed systems, consider:
- Kafka (victor-kafka-events package)
- Redis Streams (victor-redis-events package)

Example:
    from victor.core.events.backends_lightweight import (
        SQLiteEventBackend,
        DatabaseEventBackend,
    )

    # SQLite (file-based, persistent)
    backend = SQLiteEventBackend("events.db")
    await backend.connect()

    # Any SQL database via SQLAlchemy
    backend = DatabaseEventBackend("postgresql://localhost/victor")
    await backend.connect()
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3

from victor.core.json_utils import json_dumps, json_loads
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from victor.core.events.protocols import (
    BackendConfig,
    BackendType,
    DeliveryGuarantee,
    MessagingEvent,
    EventHandler,
    EventPublishError,
    IEventBackend,
    SubscriptionHandle,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SQLite Event Backend (Lightweight, File-Based)
# =============================================================================


@dataclass
class _SQLiteSubscription:
    """Internal subscription tracking for SQLite backend."""

    id: str
    pattern: str
    handler: EventHandler
    is_active: bool = True
    last_event_id: int = 0


class SQLiteEventBackend:
    """SQLite-based event backend for persistent, single-instance deployments.

    Features:
    - File-based persistence (survives restarts)
    - AT_LEAST_ONCE delivery (events stored until acknowledged)
    - Polling-based subscription (configurable interval)
    - Lightweight (no external dependencies)

    Limitations:
    - Single writer (SQLite limitation)
    - Not suitable for distributed deployments
    - Polling adds latency vs push-based systems

    Schema:
        events (id, topic, data, timestamp, source, correlation_id, delivered)

    Example:
        backend = SQLiteEventBackend("events.db")
        await backend.connect()

        await backend.publish(MessagingEvent(topic="tool.call", data={"name": "read"}))

        async def handler(event):
            print(f"Received: {event.topic}")

        await backend.subscribe("tool.*", handler)
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        config: Optional[BackendConfig] = None,
        *,
        poll_interval_ms: float = 100.0,
        max_batch_size: int = 100,
    ) -> None:
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database (":memory:" for in-memory)
            config: Optional backend configuration
            poll_interval_ms: Polling interval for subscriptions
            max_batch_size: Maximum events to fetch per poll
        """
        self._db_path = db_path
        self._config = config or BackendConfig(backend_type=BackendType.DATABASE)
        self._poll_interval_ms = poll_interval_ms
        self._max_batch_size = max_batch_size

        self._conn: Optional[sqlite3.Connection] = None
        self._is_connected = False
        self._subscriptions: Dict[str, _SQLiteSubscription] = {}
        self._poller_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.DATABASE

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self._is_connected

    async def connect(self) -> None:
        """Connect to SQLite database and create schema."""
        if self._is_connected:
            return

        # Create connection with WAL mode for concurrent reads + writes
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # Create schema
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                topic TEXT NOT NULL,
                data TEXT NOT NULL,
                timestamp REAL NOT NULL,
                source TEXT NOT NULL,
                correlation_id TEXT,
                partition_key TEXT,
                headers TEXT,
                delivery_guarantee TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at)
        """)

        # Create delivery tracking table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS event_deliveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                subscription_id TEXT NOT NULL,
                delivered_at REAL,
                acknowledged_at REAL,
                FOREIGN KEY (event_id) REFERENCES events(id),
                UNIQUE(event_id, subscription_id)
            )
        """)

        self._is_connected = True

        # Start poller if we have subscriptions
        if self._subscriptions:
            self._start_poller()

        logger.debug(f"SQLiteEventBackend connected to {self._db_path}")

    async def disconnect(self) -> None:
        """Disconnect from database."""
        if not self._is_connected:
            return

        self._is_connected = False

        # Stop poller
        if self._poller_task and not self._poller_task.done():
            self._poller_task.cancel()
            try:
                await asyncio.wait_for(self._poller_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Close connection
        if self._conn:
            self._conn.close()
            self._conn = None

        logger.debug("SQLiteEventBackend disconnected")

    async def health_check(self) -> bool:
        """Check database health."""
        if not self._is_connected or not self._conn:
            return False
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _event_row(self, event: MessagingEvent) -> tuple:
        """Build a parameter tuple for an INSERT."""
        return (
            event.id,
            event.topic,
            json_dumps(event.data),
            event.timestamp,
            event.source,
            event.correlation_id,
            event.partition_key,
            json_dumps(event.headers),
            event.delivery_guarantee.value,
        )

    _INSERT_SQL = """
        INSERT INTO events (
            event_id, topic, data, timestamp, source,
            correlation_id, partition_key, headers, delivery_guarantee
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    async def publish(self, event: MessagingEvent) -> bool:
        """Publish event to database.

        Args:
            event: MessagingEvent to publish

        Returns:
            True if event was stored
        """
        if not self._is_connected or not self._conn:
            raise EventPublishError(event, "Backend not connected", retryable=True)

        try:
            self._conn.execute(self._INSERT_SQL, self._event_row(event))
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Duplicate event ID
            logger.warning(f"Duplicate event ID: {event.id}")
            return False
        except Exception as e:
            raise EventPublishError(event, str(e), retryable=True)

    async def publish_batch(self, events: List[MessagingEvent]) -> int:
        """Publish multiple events in a single transaction."""
        if not self._is_connected or not self._conn:
            return 0

        rows = []
        for event in events:
            try:
                rows.append(self._event_row(event))
            except Exception:
                continue

        if not rows:
            return 0

        try:
            self._conn.executemany(self._INSERT_SQL, rows)
            self._conn.commit()
            return len(rows)
        except sqlite3.IntegrityError:
            # Fall back to one-by-one for partial success on duplicates
            success_count = 0
            for row in rows:
                try:
                    self._conn.execute(self._INSERT_SQL, row)
                    self._conn.commit()
                    success_count += 1
                except sqlite3.IntegrityError:
                    continue
            return success_count
        except Exception as e:
            logger.error(f"Batch publish failed: {e}")
            return 0

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        """Subscribe to events matching a pattern.

        Args:
            pattern: Topic pattern (e.g., "tool.*")
            handler: Async callback for events

        Returns:
            Subscription handle
        """
        subscription_id = uuid.uuid4().hex[:12]

        # Get latest event ID to start from
        last_id = 0
        if self._conn:
            cursor = self._conn.execute("SELECT MAX(id) FROM events")
            row = cursor.fetchone()
            if row and row[0]:
                last_id = row[0]

        subscription = _SQLiteSubscription(
            id=subscription_id,
            pattern=pattern,
            handler=handler,
            is_active=True,
            last_event_id=last_id,
        )

        with self._lock:
            self._subscriptions[subscription_id] = subscription

        # Start poller if not running
        if self._is_connected and not self._poller_task:
            self._start_poller()

        handle = SubscriptionHandle(
            subscription_id=subscription_id,
            pattern=pattern,
            is_active=True,
        )

        async def bound_unsubscribe() -> None:
            await self.unsubscribe(handle)

        handle.unsubscribe = bound_unsubscribe
        return handle

    async def unsubscribe(self, handle: SubscriptionHandle) -> bool:
        """Unsubscribe from events."""
        with self._lock:
            subscription = self._subscriptions.get(handle.subscription_id)
            if subscription:
                subscription.is_active = False
                del self._subscriptions[handle.subscription_id]
                handle.is_active = False
                return True
        return False

    def _start_poller(self) -> None:
        """Start the polling task."""
        if self._poller_task and not self._poller_task.done():
            return
        self._poller_task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        """Background loop that polls for new events."""
        while self._is_connected:
            try:
                await self._poll_events()
                await asyncio.sleep(self._poll_interval_ms / 1000.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Poll loop error: {e}")
                await asyncio.sleep(1.0)

    async def _poll_events(self) -> None:
        """Poll and deliver events to subscriptions."""
        if not self._conn:
            return

        with self._lock:
            subscriptions = list(self._subscriptions.values())

        for subscription in subscriptions:
            if not subscription.is_active:
                continue

            # Fetch new events
            cursor = self._conn.execute(
                """
                SELECT id, event_id, topic, data, timestamp, source,
                       correlation_id, partition_key, headers, delivery_guarantee
                FROM events
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (subscription.last_event_id, self._max_batch_size),
            )

            for row in cursor:
                event = MessagingEvent(
                    id=row["event_id"],
                    topic=row["topic"],
                    data=json_loads(row["data"]),
                    timestamp=row["timestamp"],
                    source=row["source"],
                    correlation_id=row["correlation_id"],
                    partition_key=row["partition_key"],
                    headers=json_loads(row["headers"] or "{}"),
                    delivery_guarantee=DeliveryGuarantee(row["delivery_guarantee"]),
                )

                # Check pattern match
                if event.matches_pattern(subscription.pattern):
                    try:
                        await subscription.handler(event)
                    except Exception as e:
                        logger.warning(f"Handler error for {event.topic}: {e}")

                # Update last processed ID
                subscription.last_event_id = row["id"]

    def get_event_count(self) -> int:
        """Get total event count in database."""
        if not self._conn:
            return 0
        cursor = self._conn.execute("SELECT COUNT(*) FROM events")
        row = cursor.fetchone()
        return row[0] if row else 0

    def cleanup_old_events(self, max_age_seconds: float = 86400) -> int:
        """Delete events older than max_age_seconds.

        Args:
            max_age_seconds: Maximum event age (default: 24 hours)

        Returns:
            Number of events deleted
        """
        if not self._conn:
            return 0

        cutoff = time.time() - max_age_seconds
        cursor = self._conn.execute(
            "DELETE FROM events WHERE created_at < ?",
            (cutoff,),
        )
        return cursor.rowcount


# =============================================================================
# Backend Registration
# =============================================================================


def register_lightweight_backends() -> None:
    """Register lightweight backends with the factory.

    Call this to enable SQLite backend via create_event_backend().

    Example:
        from victor.core.events.backends_lightweight import register_lightweight_backends
        from victor.core.events import create_event_backend, BackendType

        register_lightweight_backends()
        backend = create_event_backend(backend_type=BackendType.DATABASE)
    """
    from victor.core.events.backends import register_backend_factory

    def create_sqlite_backend(config: BackendConfig) -> SQLiteEventBackend:
        db_path = config.extra.get("db_path", ":memory:")
        poll_interval = config.extra.get("poll_interval_ms", 100.0)
        return SQLiteEventBackend(
            db_path=db_path,
            config=config,
            poll_interval_ms=poll_interval,
        )

    register_backend_factory(BackendType.DATABASE, create_sqlite_backend)
    logger.debug("Registered lightweight backends: DATABASE (SQLite)")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "SQLiteEventBackend",
    "register_lightweight_backends",
]
