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

"""Synchronous wrapper for async event backend.

This module provides synchronous wrappers around the async IEventBackend,
allowing gradual migration from sync to async APIs.

Usage:
    from victor.core.events.sync_wrapper import SyncEventWrapper
    from victor.core.events import create_event_backend

    # Create async backend
    backend = create_event_backend()
    await backend.connect()

    # Wrap for sync usage
    sync_wrapper = SyncEventWrapper(backend)

    # Use sync API
    event = MessagingEvent(topic="test", data={})
    sync_wrapper.publish(event)  # Sync call, runs async internally
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

from victor.core.async_utils import run_sync, run_sync_in_thread
from victor.core.events.protocols import (
    EventHandler,
    MessagingEvent,
    IEventBackend,
    SubscriptionHandle,
)

if TYPE_CHECKING:
    from victor.core.events import ObservabilityBus

logger = logging.getLogger(__name__)

# Type alias for sync event handlers
SyncEventHandler = Callable[[MessagingEvent], None]


class SyncEventWrapper:
    """Synchronous wrapper around IEventBackend.

    This wrapper allows synchronous code to use the async event backend
    by delegating to the shared async/sync bridge helpers.

    This is intended for gradual migration from sync to async APIs.
    New code should use async/await directly.

    Example:
        >>> from victor.core.events import create_event_backend, MessagingEvent
        >>> from victor.core.events.sync_wrapper import SyncEventWrapper
        >>>
        >>> # Create async backend
        >>> backend = create_event_backend()
        >>> await backend.connect()
        >>>
        >>> # Wrap for sync usage
        >>> sync_wrapper = SyncEventWrapper(backend)
        >>>
        >>> # Use sync API
        >>> event = MessagingEvent(topic="tool.start", data={"tool": "read_file"})
        >>> sync_wrapper.publish(event)
        True

    Thread Safety:
        - Sync callers use the canonical `run_sync()` helper
        - Async-context callers hop to a worker thread via `run_sync_in_thread()`
        - For thread safety and best performance, use backend directly with async/await
    """

    def __init__(self, backend: IEventBackend) -> None:
        """Initialize the sync wrapper.

        Args:
            backend: Async event backend to wrap
        """
        self._backend = backend
        logger.debug(f"[SyncEventWrapper] Created wrapper for {type(backend).__name__}")

    def _run_async_synchronous(self, coro: Awaitable) -> any:
        """Run an async coroutine in a synchronous context.

        This method handles both cases:
        1. Called from sync context - uses the shared `run_sync()` helper
        2. Called from async context - uses `run_sync_in_thread()` to avoid nested loops

        Args:
            coro: Awaitable coroutine to execute

        Returns:
            Result of the coroutine

        Raises:
            TimeoutError: If operation takes longer than 5 seconds
            Exception: If coroutine raises an exception
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            logger.warning(
                "[SyncEventWrapper] Method called from async context. "
                "This indicates incorrect usage. Use async backend directly."
            )
            return run_sync_in_thread(coro, timeout=5.0)
        except RuntimeError:
            return run_sync(coro)

    def publish(self, event: MessagingEvent) -> bool:
        """Publish an event synchronously.

        This method runs the async publish operation in a new event loop.

        Args:
            event: MessagingEvent to publish

        Returns:
            True if event was published successfully

        Example:
            >>> wrapper = SyncEventWrapper(backend)
            >>> event = MessagingEvent(topic="test", data={})
            >>> wrapper.publish(event)
            True
        """
        return self._run_async_synchronous(self._backend.publish(event))

    def publish_batch(self, events: list[MessagingEvent]) -> int:
        """Publish multiple events synchronously.

        Args:
            events: List of events to publish

        Returns:
            Number of events successfully published
        """
        return self._run_async_synchronous(self._backend.publish_batch(events))

    def subscribe(
        self,
        pattern: str,
        handler: SyncEventHandler,
    ) -> SubscriptionHandle:
        """Subscribe to events synchronously.

        Note: The handler will be called from async context. If the handler
        is CPU-intensive or blocking, consider running it in a thread pool.

        Args:
            pattern: Topic pattern to subscribe to (e.g., "tool.*")
            handler: Synchronous event handler

        Returns:
            Subscription handle for unsubscription

        Example:
            >>> def handler(event):
            ...     print(f"Received: {event.topic}")
            >>>
            >>> wrapper = SyncEventWrapper(backend)
            >>> handle = wrapper.subscribe("tool.*", handler)
            >>>
            >>> # Later: unsubscribe
            >>> handle.unsubscribe()
        """

        async def async_wrapper(event: MessagingEvent) -> None:
            """Wrap sync handler for async backend."""
            try:
                handler(event)
            except Exception as e:
                logger.error(f"[SyncEventWrapper] Handler error for {event.topic}: {e}")

        return self._run_async_synchronous(self._backend.subscribe(pattern, async_wrapper))

    def unsubscribe(self, handle: SubscriptionHandle) -> bool:
        """Unsubscribe from events synchronously.

        Args:
            handle: Subscription handle from subscribe()

        Returns:
            True if unsubscribed successfully
        """
        return self._run_async_synchronous(self._backend.unsubscribe(handle))

    def health_check(self) -> bool:
        """Check backend health synchronously.

        Returns:
            True if backend is healthy
        """
        return self._run_async_synchronous(self._backend.health_check())

    # SyncObservabilityBus removed — had zero external callers.
    # Use SyncEventWrapper directly for sync→async bridging.
