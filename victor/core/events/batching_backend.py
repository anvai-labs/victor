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

"""Batching event backend decorator for optimized high-frequency event delivery.

This module provides a decorator that wraps any IEventBackend to add
asynchronous batching support. It's particularly useful for:
- Reducing overhead of high-frequency events (e.g., streaming chunks)
- Optimizing remote backend throughput (Kafka, SQS, Redis)
- Preventing event fanout bottlenecks in the local dispatcher
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

from victor.core.events.protocols import (
    BackendConfig,
    BackendType,
    MessagingEvent,
    EventHandler,
    IEventBackend,
    SubscriptionHandle,
)

logger = logging.getLogger(__name__)


class BatchingEventBackend:
    """Decorator that adds batching to an underlying event backend.

    Attributes:
        max_batch_size: Maximum number of events in a single batch
        flush_interval: Maximum time (seconds) to wait before flushing a partial batch
    """

    def __init__(
        self,
        wrapped_backend: IEventBackend,
        config: Optional[BackendConfig] = None,
    ) -> None:
        """Initialize the batching decorator.

        Args:
            wrapped_backend: The backend to wrap
            config: Configuration containing batching parameters
        """
        self._wrapped = wrapped_backend
        self._config = config or BackendConfig()

        # Batching parameters
        self._max_batch_size = getattr(self._config, "max_batch_size", 100)
        self._flush_interval = getattr(self._config, "flush_interval_ms", 1000.0) / 1000.0

        # Internal state
        self._batch: List[MessagingEvent] = []
        self._batch_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._is_connected = False

    @property
    def backend_type(self) -> BackendType:
        return self._wrapped.backend_type

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._wrapped.is_connected

    async def connect(self) -> None:
        """Connect the underlying backend and start the flush loop."""
        if self._is_connected:
            return

        await self._wrapped.connect()
        self._is_connected = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.debug(
            "BatchingEventBackend connected (max_batch_size=%d, flush_interval=%.2fs)",
            self._max_batch_size,
            self._flush_interval,
        )

    async def disconnect(self) -> None:
        """Flush remaining events and disconnect."""
        if not self._is_connected:
            return

        self._is_connected = False

        # Cancel flush loop
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_now()

        await self._wrapped.disconnect()
        logger.debug("BatchingEventBackend disconnected")

    async def health_check(self) -> bool:
        return self._is_connected and await self._wrapped.health_check()

    async def publish(self, event: MessagingEvent) -> bool:
        """Queue an event for batching.

        Args:
            event: Event to publish

        Returns:
            True if event was accepted for batching
        """
        async with self._batch_lock:
            self._batch.append(event)

            # Flush immediately if batch is full
            if len(self._batch) >= self._max_batch_size:
                # Use a background task to avoid blocking the publisher
                asyncio.create_task(self._flush_now())

        return True

    async def publish_batch(self, events: List[MessagingEvent]) -> int:
        """Queue multiple events for batching.

        Args:
            events: List of events

        Returns:
            Number of events accepted
        """
        async with self._batch_lock:
            self._batch.extend(events)

            # Flush if batch limit exceeded
            if len(self._batch) >= self._max_batch_size:
                asyncio.create_task(self._flush_now())

        return len(events)

    async def _flush_now(self) -> None:
        """Perform an immediate flush of the current batch."""
        async with self._batch_lock:
            if not self._batch:
                return

            batch_to_flush = self._batch
            self._batch = []

        try:
            # Delegate to wrapped backend's batch publishing
            await self._wrapped.publish_batch(batch_to_flush)
        except Exception as e:
            logger.error("Failed to flush event batch: %s", e)

    async def _flush_loop(self) -> None:
        """Periodic background task to flush the batch."""
        while self._is_connected:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_now()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in event flush loop: %s", e)

    async def subscribe(self, pattern: str, handler: EventHandler) -> SubscriptionHandle:
        """Subscriptions bypass batching and go directly to the wrapped backend."""
        return await self._wrapped.subscribe(pattern, handler)

    async def unsubscribe(self, handle: SubscriptionHandle) -> bool:
        return await self._wrapped.unsubscribe(handle)

    def __getattr__(self, name: str) -> Any:
        """Proxy any other attributes to the wrapped backend."""
        return getattr(self._wrapped, name)
