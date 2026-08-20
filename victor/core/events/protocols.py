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

"""Protocol definitions for distributed event backends.

This module defines the protocols (interfaces) for swappable event backends,
enabling distributed messaging via Kafka, SQS, RabbitMQ, Redis, or in-memory.

Design Principles:
- Protocol-first: All backends implement IEventBackend protocol
- Pluggable: Backends can be swapped via configuration
- Async-native: All operations are async for non-blocking I/O
- Delivery guarantees: Support at-most-once, at-least-once, exactly-once

Two Primary Use Cases:
1. ObservabilityBus: High-volume telemetry (lossy OK, sampling/batching)
2. AgentMessageBus: Cross-agent communication (delivery guarantees required)

Example:
    # Using the protocol directly
    async def publish_events(backend: IEventBackend):
        await backend.publish(MessagingEvent(topic="tool.start", data={}))

    # Factory creates appropriate backend
    backend = create_event_backend(backend_type="kafka", config=kafka_config)
"""

from __future__ import annotations

import uuid
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)


class DeliveryGuarantee(str, Enum):
    """Delivery guarantee levels for event messaging.

    Different backends support different guarantee levels:
    - InMemory: AT_MOST_ONCE only (no persistence)
    - Kafka: All three levels (configurable acks)
    - SQS: AT_LEAST_ONCE (with visibility timeout)
    - RabbitMQ: AT_LEAST_ONCE (with ack mode)

    Attributes:
        AT_MOST_ONCE: Fire-and-forget, may lose events (fastest)
        AT_LEAST_ONCE: Retry until ack, may duplicate (balanced)
        EXACTLY_ONCE: Transactional, no loss or duplicates (slowest)
    """

    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class BackendType(str, Enum):
    """Supported event backend types.

    Attributes:
        IN_MEMORY: In-process queue (default, for single-instance)
        KAFKA: Apache Kafka (distributed, high throughput)
        SQS: AWS SQS (serverless, managed)
        RABBITMQ: RabbitMQ (traditional MQ, flexible routing)
        REDIS: Redis Streams (fast, simple)
        DATABASE: Database-backed (postgres, sqlite for persistence)
    """

    IN_MEMORY = "in_memory"
    KAFKA = "kafka"
    SQS = "sqs"
    RABBITMQ = "rabbitmq"
    REDIS = "redis"
    DATABASE = "database"


@dataclass
class MessagingEvent:
    """Canonical event format for all distributed messaging.

    This is the unified event structure used across all backends.
    It's designed to be serializable and backend-agnostic.

    Attributes:
        topic: Dot-separated topic string (e.g., "tool.call", "agent.message")
        data: Event payload (must be JSON-serializable)
        id: Unique event identifier
        timestamp: Unix timestamp when event was created
        source: Component that generated the event
        correlation_id: Optional ID for correlating related events
        partition_key: Optional key for partitioning (Kafka, Redis)
        headers: Optional metadata headers
        delivery_guarantee: Required delivery level

    Example:
        event = MessagingEvent(
            topic="tool.read",
            data={"file": "/path/to/file", "lines": 100},
            source="agent_1",
            correlation_id="task_abc123",
        )
    """

    topic: str
    data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
    source: str = "victor"
    correlation_id: Optional[str] = field(default=None)
    partition_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_MOST_ONCE

    def __post_init__(self):
        """Initialize correlation ID and headers from context if not provided."""
        from victor.core.context import get_trace_id, get_session_id

        # Auto-propagate trace ID as correlation ID
        if self.correlation_id is None:
            self.correlation_id = get_trace_id()

        # Auto-propagate session ID in headers
        sid = get_session_id()
        if sid and "session_id" not in self.headers:
            self.headers["session_id"] = sid

    @property
    def category(self) -> str:
        """Extract category from topic (first part before the dot).

        For example, "tool.call" -> "tool", "agent.message" -> "agent".

        Returns:
            Category string (first part of topic)
        """
        return self.topic.split(".")[0] if self.topic else "unknown"

    @property
    def datetime(self) -> "datetime":
        """Convert timestamp to datetime object for display purposes.

        Returns:
            datetime object corresponding to the timestamp
        """
        from datetime import datetime

        return datetime.fromtimestamp(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary for transport."""
        return {
            "id": self.id,
            "topic": self.topic,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "partition_key": self.partition_key,
            "headers": self.headers,
            "delivery_guarantee": self.delivery_guarantee.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessagingEvent":
        """Deserialize event from dictionary."""
        return cls(
            id=data.get("id", uuid.uuid4().hex[:16]),
            topic=data["topic"],
            data=data.get("data", {}),
            timestamp=data.get("timestamp", time.time()),
            source=data.get("source", "victor"),
            correlation_id=data.get("correlation_id"),
            partition_key=data.get("partition_key"),
            headers=data.get("headers", {}),
            delivery_guarantee=DeliveryGuarantee(
                data.get("delivery_guarantee", DeliveryGuarantee.AT_MOST_ONCE.value)
            ),
        )

    def matches_pattern(self, pattern: str) -> bool:
        """Check if event topic matches a pattern.

        Supports wildcard patterns:
        - "tool.*" matches "tool.call", "tool.result"
        - "*.error" matches "tool.error", "agent.error"
        - "*" matches everything

        Args:
            pattern: Pattern string with optional wildcards

        Returns:
            True if topic matches pattern
        """
        if pattern == "*":
            return True

        pattern_parts = pattern.split(".")
        topic_parts = self.topic.split(".")

        if len(pattern_parts) != len(topic_parts):
            # Handle trailing wildcard
            if pattern_parts[-1] == "*" and len(pattern_parts) <= len(topic_parts):
                return all(
                    p == "*" or p == t
                    for p, t in zip(pattern_parts[:-1], topic_parts[: len(pattern_parts) - 1])
                )
            return False

        return all(p == "*" or p == t for p, t in zip(pattern_parts, topic_parts))


@dataclass
class SubscriptionHandle:
    """Handle returned from subscribe() for managing subscriptions.

    Use this to unsubscribe or check subscription status.

    Attributes:
        subscription_id: Unique identifier for this subscription
        pattern: Topic pattern being subscribed to
        is_active: Whether subscription is currently active
    """

    subscription_id: str
    pattern: str
    is_active: bool = True

    async def unsubscribe(self) -> None:
        """Unsubscribe from the topic pattern.

        Note: This is a placeholder - actual implementation provided by backend.
        """
        self.is_active = False


# Type alias for event handlers
EventHandler = Callable[[MessagingEvent], Awaitable[None]]
SyncEventHandler = Callable[[MessagingEvent], None]


@dataclass
class BackendConfig:
    """Configuration for event backends.

    This provides common configuration options that apply to all backends,
    with backend-specific options in the `extra` dictionary.

    Attributes:
        backend_type: Type of backend to create
        delivery_guarantee: Default delivery level for events
        max_batch_size: Maximum events per batch
        flush_interval_ms: Interval to flush batches (milliseconds)
        max_retries: Maximum retry attempts for failed publishes
        retry_delay_ms: Delay between retries (milliseconds)
        extra: Backend-specific configuration options

    Backend-Specific Extra Options:
        InMemory:
            - queue_overflow_policy: "drop_newest" (default), "drop_oldest", or
              "block_with_timeout"
            - queue_overflow_block_timeout_ms: Timeout for block-with-timeout policy
            - queue_overflow_topic_policies: Optional per-topic policy overrides
            - queue_overflow_topic_block_timeout_ms: Optional per-topic timeout overrides
            - overflow_durable_sink: Optional callable/write/persist sink for dropped events

        Kafka:
            - bootstrap_servers: Kafka broker addresses
            - security_protocol: PLAINTEXT, SSL, SASL_SSL
            - group_id: Consumer group ID

        SQS:
            - queue_url: SQS queue URL
            - region: AWS region
            - visibility_timeout: Message visibility timeout

        RabbitMQ:
            - host: RabbitMQ host
            - port: RabbitMQ port
            - exchange: Exchange name
            - routing_key: Default routing key

        Redis:
            - host: Redis host
            - port: Redis port
            - stream_name: Redis stream name
            - max_len: Maximum stream length

        Database:
            - connection_string: Database connection string
            - table_name: Table for events
            - poll_interval_ms: Polling interval for consumers
    """

    backend_type: BackendType = BackendType.IN_MEMORY
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_MOST_ONCE
    max_batch_size: int = 100
    flush_interval_ms: float = 1000.0
    max_retries: int = 3
    retry_delay_ms: float = 100.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_observability(cls) -> "BackendConfig":
        """Create config optimized for observability (lossy OK).

        Returns:
            Config with AT_MOST_ONCE delivery, batching enabled
        """
        return cls(
            backend_type=BackendType.IN_MEMORY,
            delivery_guarantee=DeliveryGuarantee.AT_MOST_ONCE,
            max_batch_size=100,
            flush_interval_ms=1000.0,
        )

    @classmethod
    def for_agent_messaging(cls) -> "BackendConfig":
        """Create config optimized for agent communication (reliable).

        Returns:
            Config with AT_LEAST_ONCE delivery, small batches
        """
        return cls(
            backend_type=BackendType.IN_MEMORY,
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
            max_batch_size=10,
            flush_interval_ms=100.0,
            max_retries=5,
        )


@runtime_checkable
class IEventPublisher(Protocol):
    """Protocol for event publishers.

    Implementations should handle serialization and transport.
    """

    async def publish(self, event: MessagingEvent) -> bool:
        """Publish a single event.

        Args:
            event: Event to publish

        Returns:
            True if published successfully (or queued for delivery)

        Raises:
            EventPublishError: If publish fails and cannot retry
        """
        ...

    async def publish_batch(self, events: List[MessagingEvent]) -> int:
        """Publish multiple events in a batch.

        Args:
            events: List of events to publish

        Returns:
            Number of events successfully published

        Raises:
            EventPublishError: If batch publish fails
        """
        ...


@runtime_checkable
class IEventSubscriber(Protocol):
    """Protocol for event subscribers.

    Implementations should handle deserialization and delivery.
    """

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        """Subscribe to events matching a pattern.

        Args:
            pattern: Topic pattern (supports wildcards like "tool.*")
            handler: Async callback for received events

        Returns:
            Handle for managing the subscription

        Example:
            async def on_tool_event(event: MessagingEvent):
                print(f"Tool: {event.topic}")

            handle = await backend.subscribe("tool.*", on_tool_event)
            # Later: await handle.unsubscribe()
        """
        ...

    async def unsubscribe(self, handle: SubscriptionHandle) -> bool:
        """Unsubscribe using a subscription handle.

        Args:
            handle: Subscription handle from subscribe()

        Returns:
            True if unsubscribed successfully
        """
        ...


@runtime_checkable
class IEventBackend(IEventPublisher, IEventSubscriber, Protocol):
    """Protocol for complete event backend implementations.

    Combines publisher and subscriber capabilities with lifecycle methods.
    All backends (InMemory, Kafka, SQS, etc.) implement this protocol.

    Example:
        class KafkaEventBackend:
            '''Kafka implementation of IEventBackend.'''

            async def connect(self) -> None:
                self._producer = AIOKafkaProducer(...)
                await self._producer.start()

            async def publish(self, event: MessagingEvent) -> bool:
                await self._producer.send(event.topic, event.to_dict())
                return True

            async def subscribe(self, pattern, handler) -> SubscriptionHandle:
                # Create Kafka consumer for pattern
                ...
    """

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected and ready."""
        ...

    async def connect(self) -> None:
        """Connect to the backend service.

        Called once during initialization. Should establish connections,
        create topics/queues if needed, and prepare for pub/sub.

        Raises:
            ConnectionError: If connection fails
        """
        ...

    async def disconnect(self) -> None:
        """Disconnect from the backend service.

        Should cleanly close connections and flush pending events.
        """
        ...

    async def health_check(self) -> bool:
        """Check backend health.

        Returns:
            True if backend is healthy and responsive
        """
        ...


class EventPublishError(Exception):
    """Raised when event publishing fails."""

    def __init__(self, event: MessagingEvent, message: str, retryable: bool = True):
        self.event = event
        self.retryable = retryable
        super().__init__(f"Failed to publish event {event.id}: {message}")


class EventSubscriptionError(Exception):
    """Raised when subscription management fails."""

    def __init__(self, pattern: str, message: str):
        self.pattern = pattern
        super().__init__(f"Subscription error for pattern '{pattern}': {message}")


# Type for backend factory functions
BackendFactory = Callable[[BackendConfig], IEventBackend]
