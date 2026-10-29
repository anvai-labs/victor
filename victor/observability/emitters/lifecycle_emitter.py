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

import time
from contextlib import contextmanager
from typing import Any

from victor.observability.emitters.base import BaseEventEmitter, ILifecycleEventEmitter


class LifecycleEventEmitter(BaseEventEmitter, ILifecycleEventEmitter):
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

    _CATEGORY = "lifecycle"

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
