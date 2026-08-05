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

"""Tests for lightweight event backends (SQLite).

Run with: pytest tests/unit/core/events/test_lightweight_backends.py -v
"""

import asyncio
import tempfile
from pathlib import Path
from typing import List

import pytest

from victor.core.events import MessagingEvent, BackendType
from victor.core.events.backends_lightweight import (
    SQLiteEventBackend,
    register_lightweight_backends,
)


@pytest.mark.unit
class TestSQLiteEventBackend:
    """Tests for SQLiteEventBackend."""

    @pytest.fixture
    async def backend(self):
        """Create in-memory SQLite backend for each test."""
        backend = SQLiteEventBackend(":memory:")
        await backend.connect()
        yield backend
        await backend.disconnect()

    @pytest.fixture
    async def file_backend(self, tmp_path):
        """Create file-based SQLite backend."""
        db_path = str(tmp_path / "events.db")
        backend = SQLiteEventBackend(db_path)
        await backend.connect()
        yield backend
        await backend.disconnect()

    def test_backend_type(self):
        """Backend should report DATABASE type."""
        backend = SQLiteEventBackend()
        assert backend.backend_type == BackendType.DATABASE

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Backend should connect and disconnect cleanly."""
        backend = SQLiteEventBackend(":memory:")
        assert backend.is_connected is False

        await backend.connect()
        assert backend.is_connected is True

        await backend.disconnect()
        assert backend.is_connected is False

    @pytest.mark.asyncio
    async def test_health_check(self, backend):
        """Connected backend should pass health check."""
        assert await backend.health_check() is True

    @pytest.mark.asyncio
    async def test_publish_stores_event(self, backend):
        """Publish should store event in database."""
        event = MessagingEvent(topic="test.topic", data={"key": "value"})
        result = await backend.publish(event)

        assert result is True
        assert backend.get_event_count() == 1

    @pytest.mark.asyncio
    async def test_publish_subscribe_basic(self, backend):
        """Basic pub/sub should work with polling."""
        received: List[MessagingEvent] = []

        async def handler(event: MessagingEvent):
            received.append(event)

        # Subscribe first
        handle = await backend.subscribe("test.*", handler)
        assert handle.is_active is True

        # Publish
        event = MessagingEvent(topic="test.topic", data={"msg": "hello"})
        await backend.publish(event)

        # Wait for polling to deliver
        await asyncio.sleep(0.3)

        assert len(received) == 1
        assert received[0].data["msg"] == "hello"

    @pytest.mark.asyncio
    async def test_pattern_filtering(self, backend):
        """Subscriptions should filter by pattern."""
        tool_events: List[MessagingEvent] = []
        agent_events: List[MessagingEvent] = []

        async def tool_handler(event: MessagingEvent):
            tool_events.append(event)

        async def agent_handler(event: MessagingEvent):
            agent_events.append(event)

        await backend.subscribe("tool.*", tool_handler)
        await backend.subscribe("agent.*", agent_handler)

        await backend.publish(MessagingEvent(topic="tool.call"))
        await backend.publish(MessagingEvent(topic="agent.message"))
        await backend.publish(MessagingEvent(topic="tool.result"))

        await asyncio.sleep(0.3)

        assert len(tool_events) == 2
        assert len(agent_events) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, backend):
        """Unsubscribe should stop event delivery."""
        received: List[MessagingEvent] = []

        async def handler(event: MessagingEvent):
            received.append(event)

        handle = await backend.subscribe("test.*", handler)

        await backend.publish(MessagingEvent(topic="test.1"))
        await asyncio.sleep(0.3)
        assert len(received) == 1

        await handle.unsubscribe()

        await backend.publish(MessagingEvent(topic="test.2"))
        await asyncio.sleep(0.3)
        assert len(received) == 1  # Still 1

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path):
        """Events should persist across connections."""
        db_path = str(tmp_path / "persistent.db")

        # First connection - publish events
        backend1 = SQLiteEventBackend(db_path)
        await backend1.connect()
        await backend1.publish(MessagingEvent(topic="persist.1", data={"seq": 1}))
        await backend1.publish(MessagingEvent(topic="persist.2", data={"seq": 2}))
        await backend1.disconnect()

        # Second connection - events should exist
        backend2 = SQLiteEventBackend(db_path)
        await backend2.connect()
        assert backend2.get_event_count() == 2
        await backend2.disconnect()

    @pytest.mark.asyncio
    async def test_batch_publish(self, backend):
        """Batch publish should store all events."""
        events = [
            MessagingEvent(topic="batch.1", data={"i": 1}),
            MessagingEvent(topic="batch.2", data={"i": 2}),
            MessagingEvent(topic="batch.3", data={"i": 3}),
        ]

        count = await backend.publish_batch(events)
        assert count == 3
        assert backend.get_event_count() == 3

    @pytest.mark.asyncio
    async def test_cleanup_old_events(self, backend):
        """Cleanup should remove old events."""
        # Publish events
        for i in range(5):
            await backend.publish(MessagingEvent(topic=f"cleanup.{i}"))

        assert backend.get_event_count() == 5

        # Cleanup with 0 max age (delete all)
        deleted = backend.cleanup_old_events(max_age_seconds=0)
        assert deleted == 5
        assert backend.get_event_count() == 0

    @pytest.mark.asyncio
    async def test_duplicate_event_id_rejected(self, backend):
        """Duplicate event IDs should be rejected."""
        event1 = MessagingEvent(id="duplicate123", topic="test.1")
        event2 = MessagingEvent(id="duplicate123", topic="test.2")

        result1 = await backend.publish(event1)
        result2 = await backend.publish(event2)

        assert result1 is True
        assert result2 is False  # Duplicate rejected
        assert backend.get_event_count() == 1


@pytest.mark.unit
class TestBackendRegistration:
    """Tests for backend registration."""

    def test_register_lightweight_backends(self):
        """Registration should add DATABASE backend to factory."""
        from victor.core.events.backends import _backend_factories

        # Register
        register_lightweight_backends()

        # Should be registered
        assert BackendType.DATABASE in _backend_factories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
