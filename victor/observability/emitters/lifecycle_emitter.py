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

"""Lifecycle event emitter for tracking session lifecycle.

SOLID Principles:
- SRP: Focused solely on lifecycle events
- OCP: Extensible via inheritance (can add custom lifecycle tracking)
- LSP: Substitutable with ILifecycleEventEmitter
- ISP: Implements focused interface
- DIP: Depends on ObservabilityBus abstraction, not concrete implementation

Migration Notes:
- Migrated from legacy EventBus to canonical core/events system
- Uses topic-based routing ("lifecycle.session.start", "lifecycle.session.end")
- Fully async API with sync wrappers for gradual migration
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from victor.observability.emitters.base import ILifecycleEventEmitter
from victor.core.events import ObservabilityBus, SyncEventWrapper

logger = logging.getLogger(__name__)


class LifecycleEventEmitter(ILifecycleEventEmitter):
    """Emits lifecycle events to ObservabilityBus.

    Thread-safe, performant lifecycle tracking.
    Handles errors gracefully to avoid impacting execution.

    Example:
        >>> emitter = LifecycleEventEmitter()
        >>> emitter.session_start("session-123", agent_id="agent-1")
        >>> # ... session ...
        >>> emitter.session_end("session-123", 5000.0)

    Migration:
        This emitter now uses the canonical core/events system instead of
        the legacy observability/event_bus.py. Events use topic-based routing
        ("lifecycle.session.start", "lifecycle.session.end") instead of
        category-based routing.
    """

    def __init__(self, bus: Optional[ObservabilityBus] = None):
        """Initialize the lifecycle event emitter.

        Args:
            bus: Optional ObservabilityBus instance. If None, uses DI container.
        """
        self._bus = bus
        self._sync_wrapper: Optional[SyncEventWrapper] = None
        self._enabled = True

    def _get_bus(self) -> Optional[ObservabilityBus]:
        """Get ObservabilityBus instance.

        Returns:
            ObservabilityBus instance or None if unavailable
        """
        if self._bus:
            return self._bus

        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    def _get_sync_wrapper(self) -> Optional[SyncEventWrapper]:
        """Get sync wrapper for gradual migration.

        Returns:
            SyncEventWrapper instance or None if unavailable
        """
        if self._sync_wrapper:
            return self._sync_wrapper

        bus = self._get_bus()
        if bus:
            self._sync_wrapper = SyncEventWrapper(bus.backend)
            return self._sync_wrapper

        return None

    async def emit_async(
        self,
        topic: str,
        data: Dict[str, Any],
    ) -> bool:
        """Emit a lifecycle event asynchronously.

        Args:
            topic: Event topic (e.g., "lifecycle.session.start")
            data: Event payload

        Returns:
            True if emission succeeded, False otherwise
        """
        if not self._enabled:
            return False

        bus = self._get_bus()
        if bus:
            try:
                # Add category metadata for observability features
                data_with_category = {
                    **data,
                    "category": "lifecycle",
                }
                return await bus.emit(topic, data_with_category)
            except Exception as e:
                logger.debug(f"Failed to emit lifecycle event: {e}")
                return False

        return False

    def emit(
        self,
        topic: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit a lifecycle event synchronously (for gradual migration).

        This method wraps the async emit_async() method using emit_event_sync()
        so sync callers can reuse the shared async/sync bridge behavior.

        Args:
            topic: Event topic (e.g., "lifecycle.session.start")
            data: Event payload
        """
        try:
            from victor.core.events.emit_helper import emit_event_sync

            bus = self._get_bus()
            if bus:
                emit_event_sync(
                    bus,
                    topic=topic,
                    data=data,
                    source="LifecycleEventEmitter",
                )
        except Exception as e:
            logger.debug(f"Failed to emit lifecycle event: {e}")

    async def session_start_async(
        self,
        session_id: str,
        **metadata: Any,
    ) -> bool:
        """Emit session start event asynchronously.

        Args:
            session_id: Unique session identifier
            **metadata: Additional metadata

        Returns:
            True if emission succeeded
        """
        return await self.emit_async(
            topic="lifecycle.session.start",
            data={
                "session_id": session_id,
                **metadata,
            },
        )

    def session_start(
        self,
        session_id: str,
        **metadata: Any,
    ) -> None:
        """Emit session start event (sync wrapper).

        Args:
            session_id: Unique session identifier
            **metadata: Additional metadata
        """
        self.emit(
            topic="lifecycle.session.start",
            data={
                "session_id": session_id,
                **metadata,
            },
        )

    async def session_end_async(
        self,
        session_id: str,
        duration_ms: float,
        **metadata: Any,
    ) -> bool:
        """Emit session end event asynchronously.

        Args:
            session_id: Unique session identifier
            duration_ms: Session duration in milliseconds
            **metadata: Additional metadata

        Returns:
            True if emission succeeded
        """
        return await self.emit_async(
            topic="lifecycle.session.end",
            data={
                "session_id": session_id,
                "duration_ms": duration_ms,
                **metadata,
            },
        )

    def session_end(
        self,
        session_id: str,
        duration_ms: float,
        **metadata: Any,
    ) -> None:
        """Emit session end event (sync wrapper).

        Args:
            session_id: Unique session identifier
            duration_ms: Session duration in milliseconds
            **metadata: Additional metadata
        """
        self.emit(
            topic="lifecycle.session.end",
            data={
                "session_id": session_id,
                "duration_ms": duration_ms,
                **metadata,
            },
        )

    @contextmanager
    def track_session(
        self,
        session_id: str,
        **metadata: Any,
    ):
        """Context manager for tracking session lifecycle.

        Automatically emits start/end events.

        Args:
            session_id: Unique session identifier
            **metadata: Additional metadata

        Yields:
            None

        Example:
            >>> with emitter.track_session("session-123", agent_id="agent-1"):
            ...     # ... session work ...
        """
        start_time = time.time()
        self.session_start(session_id, **metadata)

        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.session_end(session_id, duration_ms, **metadata)

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if event emission is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self._enabled
