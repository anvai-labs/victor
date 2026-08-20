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

"""Event backend implementations.

This module provides concrete implementations of IEventBackend:
- InMemoryEventBackend: Default in-process implementation
- (Future) KafkaEventBackend, SQSEventBackend, RabbitMQEventBackend, RedisEventBackend

The factory function create_event_backend() creates the appropriate backend
based on configuration.

Example:
    from victor.core.events.backends import (
        create_event_backend,
        InMemoryEventBackend,
        BackendConfig,
        BackendType,
    )

    # Use factory (recommended)
    backend = create_event_backend()
    await backend.connect()
    await backend.publish(MessagingEvent(topic="test", data={}))

    # Or create directly
    backend = InMemoryEventBackend()
    await backend.connect()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from fnmatch import fnmatchcase
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from victor.core.events.protocols import (
    BackendConfig,
    BackendType,
    DeliveryGuarantee,
    MessagingEvent,
    EventHandler,
    EventPublishError,
    EventSubscriptionError,
    IEventBackend,
    SubscriptionHandle,
)

logger = logging.getLogger(__name__)

_QUEUE_POLICY_DROP_NEWEST = "drop_newest"
_QUEUE_POLICY_DROP_OLDEST = "drop_oldest"
_QUEUE_POLICY_BLOCK_WITH_TIMEOUT = "block_with_timeout"
_VALID_QUEUE_POLICIES = {
    _QUEUE_POLICY_DROP_NEWEST,
    _QUEUE_POLICY_DROP_OLDEST,
    _QUEUE_POLICY_BLOCK_WITH_TIMEOUT,
}
_DELIVERY_GUARANTEE_MAP: Dict[str, DeliveryGuarantee] = {
    "at_most_once": DeliveryGuarantee.AT_MOST_ONCE,
    "at_least_once": DeliveryGuarantee.AT_LEAST_ONCE,
    "exactly_once": DeliveryGuarantee.EXACTLY_ONCE,
}
_BACKEND_TYPE_MAP: Dict[str, BackendType] = {
    "in_memory": BackendType.IN_MEMORY,
    "memory": BackendType.IN_MEMORY,
    "sqlite": BackendType.DATABASE,
    "database": BackendType.DATABASE,
    "redis": BackendType.REDIS,
    "kafka": BackendType.KAFKA,
    "sqs": BackendType.SQS,
    "rabbitmq": BackendType.RABBITMQ,
}


def _parse_backend_type(raw: Any) -> BackendType:
    """Parse backend type from settings value with safe fallback."""
    if isinstance(raw, BackendType):
        return raw
    normalized = str(raw).strip().lower()
    backend_type = _BACKEND_TYPE_MAP.get(normalized)
    if backend_type is None:
        logger.warning("Unknown event_backend_type '%s'; defaulting to '%s'", raw, "in_memory")
        return BackendType.IN_MEMORY
    return backend_type


def _parse_delivery_guarantee(raw: Any) -> DeliveryGuarantee:
    """Parse delivery guarantee from settings value with safe fallback."""
    if isinstance(raw, DeliveryGuarantee):
        return raw
    normalized = str(raw).strip().lower()
    guarantee = _DELIVERY_GUARANTEE_MAP.get(normalized)
    if guarantee is None:
        logger.warning(
            "Unknown event_delivery_guarantee '%s'; defaulting to '%s'",
            raw,
            DeliveryGuarantee.AT_MOST_ONCE.value,
        )
        return DeliveryGuarantee.AT_MOST_ONCE
    return guarantee


def _parse_int_setting(raw: Any, *, default: int, minimum: int) -> int:
    """Parse int setting with fallback and lower-bound clamp."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _parse_float_setting(raw: Any, *, default: float, minimum: float) -> float:
    """Parse float setting with fallback and lower-bound clamp."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _normalize_topic_policy_overrides(raw: Any) -> Dict[str, str]:
    """Normalize optional per-topic overflow policy overrides."""
    if not isinstance(raw, dict):
        return {}

    normalized: Dict[str, str] = {}
    for topic_pattern, policy in raw.items():
        pattern = str(topic_pattern).strip()
        if not pattern:
            continue
        normalized_policy = str(policy).strip().lower()
        if normalized_policy not in _VALID_QUEUE_POLICIES:
            logger.warning(
                "Ignoring invalid topic overflow policy '%s' for pattern '%s'",
                policy,
                topic_pattern,
            )
            continue
        normalized[pattern] = normalized_policy
    return normalized


def _normalize_topic_block_timeout_overrides(raw: Any) -> Dict[str, float]:
    """Normalize optional per-topic timeout overrides for block-with-timeout policy."""
    if not isinstance(raw, dict):
        return {}

    normalized: Dict[str, float] = {}
    for topic_pattern, timeout_ms in raw.items():
        pattern = str(topic_pattern).strip()
        if not pattern:
            continue
        try:
            parsed_timeout = float(timeout_ms)
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring non-numeric topic block timeout '%s' for pattern '%s'",
                timeout_ms,
                topic_pattern,
            )
            continue
        if parsed_timeout < 0:
            logger.warning(
                "Ignoring negative topic block timeout '%s' for pattern '%s'",
                timeout_ms,
                topic_pattern,
            )
            continue
        normalized[pattern] = parsed_timeout
    return normalized


def build_backend_config_from_settings(settings: Any) -> BackendConfig:
    """Build backend config from settings object with normalized defaults."""
    backend_type = _parse_backend_type(getattr(settings, "event_backend_type", "in_memory"))
    delivery_guarantee = _parse_delivery_guarantee(
        getattr(settings, "event_delivery_guarantee", DeliveryGuarantee.AT_MOST_ONCE.value)
    )
    max_batch_size = _parse_int_setting(
        getattr(settings, "event_max_batch_size", 100),
        default=100,
        minimum=1,
    )
    flush_interval_ms = _parse_float_setting(
        getattr(settings, "event_flush_interval_ms", 1000.0),
        default=1000.0,
        minimum=0.0,
    )

    overflow_policy = (
        str(getattr(settings, "event_queue_overflow_policy", _QUEUE_POLICY_DROP_NEWEST))
        .strip()
        .lower()
    )
    if overflow_policy not in _VALID_QUEUE_POLICIES:
        logger.warning(
            "Unknown event_queue_overflow_policy '%s'; defaulting to '%s'",
            overflow_policy,
            _QUEUE_POLICY_DROP_NEWEST,
        )
        overflow_policy = _QUEUE_POLICY_DROP_NEWEST

    queue_maxsize = _parse_int_setting(
        getattr(settings, "event_queue_maxsize", 10000),
        default=10000,
        minimum=1,
    )
    block_timeout_ms = _parse_float_setting(
        getattr(settings, "event_queue_overflow_block_timeout_ms", 50.0),
        default=50.0,
        minimum=0.0,
    )
    topic_policy_overrides = _normalize_topic_policy_overrides(
        getattr(settings, "event_queue_overflow_topic_policies", {})
    )
    topic_block_timeout_overrides = _normalize_topic_block_timeout_overrides(
        getattr(settings, "event_queue_overflow_topic_block_timeout_ms", {})
    )

    return BackendConfig(
        backend_type=backend_type,
        delivery_guarantee=delivery_guarantee,
        max_batch_size=max_batch_size,
        flush_interval_ms=flush_interval_ms,
        extra={
            "queue_maxsize": queue_maxsize,
            "queue_overflow_policy": overflow_policy,
            "queue_overflow_block_timeout_ms": block_timeout_ms,
            "queue_overflow_topic_policies": topic_policy_overrides,
            "queue_overflow_topic_block_timeout_ms": topic_block_timeout_overrides,
        },
    )


# =============================================================================
# In-Memory Event Backend
# =============================================================================


@dataclass
class _Subscription:
    """Internal subscription tracking."""

    id: str
    pattern: str
    handler: EventHandler
    is_active: bool = True


class InMemoryEventBackend:
    """In-memory event backend for single-instance deployments.

    This is the default backend that provides fast, in-process pub/sub.
    It's suitable for:
    - Single-instance deployments
    - Development and testing
    - Low-latency observability

    Limitations:
    - Events are lost on process restart
    - Cannot scale horizontally
    - Only AT_MOST_ONCE delivery (no persistence)

    Thread Safety:
    - Subscriptions are thread-safe (uses lock)
    - Publish is thread-safe (uses asyncio.Queue)
    - Handlers run in asyncio tasks (non-blocking)

    Example:
        backend = InMemoryEventBackend()
        await backend.connect()

        async def handler(event):
            print(f"Received: {event.topic}")

        handle = await backend.subscribe("tool.*", handler)
        await backend.publish(MessagingEvent(topic="tool.call", data={"name": "read"}))

        await backend.disconnect()
    """

    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        *,
        queue_maxsize: int = 10000,
        drop_alert_threshold: int = 100,
        drop_alert_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the in-memory backend.

        Args:
            config: Optional backend configuration
            queue_maxsize: Maximum queue size (0 for unbounded)
            drop_alert_threshold: Fire drop alert after this many cumulative drops.
            drop_alert_callback: Optional callable invoked when threshold is crossed.
        """
        self._drop_alert_threshold = drop_alert_threshold
        self._drop_alert_callback = drop_alert_callback
        self._per_topic_drop_counts: Dict[str, int] = {}
        self._last_alert_total_drops: int = 0
        self._config = config or BackendConfig()
        extra = self._config.extra if isinstance(self._config.extra, dict) else {}
        configured_queue_maxsize = extra.get("queue_maxsize", queue_maxsize)
        try:
            resolved_queue_maxsize = int(configured_queue_maxsize)
        except (TypeError, ValueError):
            resolved_queue_maxsize = int(queue_maxsize)
        if resolved_queue_maxsize < 0:
            resolved_queue_maxsize = max(0, int(queue_maxsize))
        self._queue_maxsize = resolved_queue_maxsize

        overflow_policy = (
            str(extra.get("queue_overflow_policy", _QUEUE_POLICY_DROP_NEWEST)).strip().lower()
        )
        if overflow_policy not in _VALID_QUEUE_POLICIES:
            logger.warning(
                "Unknown queue_overflow_policy '%s'; defaulting to '%s'",
                overflow_policy,
                _QUEUE_POLICY_DROP_NEWEST,
            )
            overflow_policy = _QUEUE_POLICY_DROP_NEWEST
        self._queue_overflow_policy = overflow_policy

        timeout_raw = extra.get("queue_overflow_block_timeout_ms", 50.0)
        try:
            timeout_ms = max(0.0, float(timeout_raw))
        except (TypeError, ValueError):
            timeout_ms = 50.0
        self._queue_overflow_block_timeout_ms = timeout_ms
        self._queue_overflow_topic_policies = _normalize_topic_policy_overrides(
            extra.get("queue_overflow_topic_policies", {})
        )
        # Prioritize specific patterns first (fewer wildcards, then longer string).
        self._queue_overflow_topic_policy_items = sorted(
            self._queue_overflow_topic_policies.items(),
            key=lambda item: (item[0].count("*"), -len(item[0])),
        )
        self._queue_overflow_topic_block_timeout_ms = _normalize_topic_block_timeout_overrides(
            extra.get("queue_overflow_topic_block_timeout_ms", {})
        )
        self._overflow_durable_sink = extra.get("overflow_durable_sink")

        self._subscriptions: Dict[str, _Subscription] = {}
        # Topic index: exact topic -> list of subscription IDs for O(1) exact dispatch.
        # Wildcard subscriptions (containing "*") are stored separately for linear scan.
        self._exact_topic_index: Dict[str, List[str]] = {}
        self._wildcard_subscriptions: List[str] = []
        self._compiled_patterns: Dict[str, Callable[[str], bool]] = {}
        # Cache: topic -> list of matching wildcard subscription IDs.
        # Invalidated on subscribe/unsubscribe to keep consistent.
        self._wildcard_match_cache: Dict[str, List[str]] = {}
        # Cache: topic -> (policy, timeout_ms) for overflow policy resolution.
        self._overflow_policy_cache: Dict[str, tuple[str, float]] = {}
        self._event_queue: asyncio.Queue[MessagingEvent] = asyncio.Queue(
            maxsize=self._queue_maxsize
        )
        self._is_connected = False
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self._pending_tasks: Set[asyncio.Task] = set()
        self._publish_stats_lock = threading.Lock()
        self._publish_stats: Dict[str, int] = {
            "queued": 0,
            "dropped_newest": 0,
            "dropped_oldest": 0,
            "blocked_success": 0,
            "blocked_timeout": 0,
            "durable_sink_success": 0,
            "durable_sink_failures": 0,
            "max_queue_depth": 0,
            "topic_policy_override_hits": 0,
        }

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.IN_MEMORY

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected and ready."""
        return self._is_connected

    def _invalidate_dispatch_caches(self) -> None:
        """Invalidate wildcard match and overflow policy caches.

        Called when subscriptions change to ensure cache consistency.
        """
        self._wildcard_match_cache.clear()
        self._overflow_policy_cache.clear()

    def _index_subscription(self, subscription: "_Subscription") -> None:
        """Add subscription to the appropriate index."""
        if "*" in subscription.pattern:
            self._wildcard_subscriptions.append(subscription.id)
            self._compiled_patterns[subscription.id] = self._compile_pattern(subscription.pattern)
            self._invalidate_dispatch_caches()
        else:
            # Exact topic match - O(1) lookup
            if subscription.pattern not in self._exact_topic_index:
                self._exact_topic_index[subscription.pattern] = []
            self._exact_topic_index[subscription.pattern].append(subscription.id)

    def _deindex_subscription(self, subscription: "_Subscription") -> None:
        """Remove subscription from the index."""
        sid = subscription.id
        if "*" in subscription.pattern:
            if sid in self._wildcard_subscriptions:
                self._wildcard_subscriptions.remove(sid)
            self._compiled_patterns.pop(sid, None)
            self._invalidate_dispatch_caches()
        else:
            subs = self._exact_topic_index.get(subscription.pattern, [])
            if sid in subs:
                subs.remove(sid)
                if not subs:
                    del self._exact_topic_index[subscription.pattern]

    @staticmethod
    def _compile_pattern(pattern: str) -> Callable[[str], bool]:
        """Compile a topic pattern into an efficient matcher function.

        Avoids repeated string splitting on every event dispatch.
        """
        if pattern == "*":
            return lambda topic: True

        pattern_parts = pattern.split(".")
        has_trailing_wildcard = pattern_parts[-1] == "*"

        def matcher(topic: str) -> bool:
            topic_parts = topic.split(".")
            if has_trailing_wildcard:
                if len(topic_parts) < len(pattern_parts) - 1:
                    return False
                return all(
                    p == "*" or p == t
                    for p, t in zip(pattern_parts[:-1], topic_parts[: len(pattern_parts) - 1])
                )
            if len(pattern_parts) != len(topic_parts):
                return False
            return all(p == "*" or p == t for p, t in zip(pattern_parts, topic_parts))

        return matcher

    async def connect(self) -> None:
        """Connect and start the event dispatcher.

        The dispatcher runs in the background, delivering events to handlers.
        """
        if self._is_connected:
            return

        self._is_connected = True
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        logger.debug("InMemoryEventBackend connected")

    async def disconnect(self) -> None:
        """Disconnect and stop the dispatcher.

        Waits for pending events to be delivered (with timeout).
        """
        if not self._is_connected:
            return

        self._is_connected = False

        # Cancel dispatcher
        if self._dispatcher_task and not self._dispatcher_task.done():
            self._dispatcher_task.cancel()
            try:
                await asyncio.wait_for(self._dispatcher_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Cancel pending handler tasks
        for task in list(self._pending_tasks):
            if not task.done():
                task.cancel()

        self._pending_tasks.clear()
        logger.debug("InMemoryEventBackend disconnected")

    async def health_check(self) -> bool:
        """Check backend health.

        Returns:
            True if connected and dispatcher is running
        """
        return (
            self._is_connected
            and self._dispatcher_task is not None
            and not self._dispatcher_task.done()
        )

    def _increment_publish_stat(self, key: str, delta: int = 1) -> None:
        """Increment queue publish counter."""
        with self._publish_stats_lock:
            self._publish_stats[key] = self._publish_stats.get(key, 0) + delta

    def _fire_drop_alert_if_needed(self, topic: str, drop_kind: str) -> None:
        """Track per-topic drops and fire the alert callback when threshold is crossed.

        Dampened: only one alert fires per threshold increment to prevent spam.
        Safe to call from the publish hot-path (no I/O; lock held only briefly).
        """
        alert_payload: Optional[Dict[str, Any]] = None
        with self._publish_stats_lock:
            self._per_topic_drop_counts[topic] = self._per_topic_drop_counts.get(topic, 0) + 1
            total = self._publish_stats.get("dropped_newest", 0) + self._publish_stats.get(
                "dropped_oldest", 0
            )
            delta = total - self._last_alert_total_drops
            if delta >= self._drop_alert_threshold and self._drop_alert_callback is not None:
                self._last_alert_total_drops = total
                alert_payload = {
                    "total_drops": total,
                    "per_topic_drops": dict(self._per_topic_drop_counts),
                    "overflow_policy": str(self._queue_overflow_policy),
                    "queue_depth": self._event_queue.qsize(),
                    "queue_maxsize": self._queue_maxsize,
                    "drop_kind": drop_kind,
                }
        if alert_payload is not None:
            try:
                self._drop_alert_callback(alert_payload)  # type: ignore[misc]
            except Exception as exc:
                logger.debug("Drop alert callback raised: %s", exc)

    def _update_max_queue_depth(self) -> None:
        """Update max queue depth watermark."""
        depth = self.get_queue_depth()
        with self._publish_stats_lock:
            if depth > self._publish_stats.get("max_queue_depth", 0):
                self._publish_stats["max_queue_depth"] = depth

    def _write_to_durable_sink(self, event: MessagingEvent, reason: str) -> None:
        """Write dropped events to optional durable sink."""
        sink = self._overflow_durable_sink
        if sink is None:
            return

        try:
            if callable(sink):
                sink(event=event, reason=reason)
            elif hasattr(sink, "write") and callable(sink.write):
                sink.write(event=event, reason=reason)
            elif hasattr(sink, "persist") and callable(sink.persist):
                sink.persist(event=event, reason=reason)
            else:
                raise TypeError("overflow_durable_sink must be callable/write/persist compatible")
            self._increment_publish_stat("durable_sink_success")
        except Exception as e:
            self._increment_publish_stat("durable_sink_failures")
            logger.debug("Durable sink write failed: %s", e)

    def _resolve_overflow_policy_for_topic(self, topic: str) -> tuple[str, float]:
        """Resolve effective overflow policy + timeout for a topic.

        Results are cached per topic and invalidated when subscriptions change.
        """
        cached = self._overflow_policy_cache.get(topic)
        if cached is not None:
            return cached

        for pattern, policy in self._queue_overflow_topic_policy_items:
            if fnmatchcase(topic, pattern):
                timeout_ms = self._queue_overflow_topic_block_timeout_ms.get(
                    pattern,
                    self._queue_overflow_block_timeout_ms,
                )
                result = (policy, timeout_ms)
                self._overflow_policy_cache[topic] = result
                return result
        result = (self._queue_overflow_policy, self._queue_overflow_block_timeout_ms)
        self._overflow_policy_cache[topic] = result
        return result

    async def publish(self, event: MessagingEvent) -> bool:
        """Publish an event to all matching subscribers.

        Args:
            event: Event to publish

        Returns:
            True if event was queued for delivery

        Raises:
            EventPublishError: If queue is full and cannot accept event
        """
        if not self._is_connected:
            raise EventPublishError(event, "Backend not connected", retryable=True)

        try:
            # Non-blocking put
            self._event_queue.put_nowait(event)
            self._increment_publish_stat("queued")
            self._update_max_queue_depth()
            return True
        except asyncio.QueueFull:
            effective_policy, effective_block_timeout_ms = self._resolve_overflow_policy_for_topic(
                event.topic
            )
            if (
                effective_policy != self._queue_overflow_policy
                or effective_block_timeout_ms != self._queue_overflow_block_timeout_ms
            ):
                self._increment_publish_stat("topic_policy_override_hits")

            if effective_policy == _QUEUE_POLICY_DROP_OLDEST:
                try:
                    oldest = self._event_queue.get_nowait()
                    self._increment_publish_stat("dropped_oldest")
                    self._fire_drop_alert_if_needed(oldest.topic, "drop_oldest")
                    self._write_to_durable_sink(oldest, "drop_oldest")
                except asyncio.QueueEmpty:
                    oldest = None

                try:
                    self._event_queue.put_nowait(event)
                    self._increment_publish_stat("queued")
                    self._update_max_queue_depth()
                    return True
                except asyncio.QueueFull:
                    self._increment_publish_stat("dropped_newest")
                    self._fire_drop_alert_if_needed(event.topic, "drop_newest_after_drop_oldest")
                    self._write_to_durable_sink(event, "drop_newest_after_drop_oldest")
                    logger.warning(
                        "Event queue remained full after drop_oldest policy, dropping event: %s",
                        event.topic,
                    )
                    return False

            if effective_policy == _QUEUE_POLICY_BLOCK_WITH_TIMEOUT:
                timeout_s = effective_block_timeout_ms / 1000.0
                try:
                    await asyncio.wait_for(self._event_queue.put(event), timeout=timeout_s)
                    self._increment_publish_stat("blocked_success")
                    self._increment_publish_stat("queued")
                    self._update_max_queue_depth()
                    return True
                except asyncio.TimeoutError:
                    self._increment_publish_stat("blocked_timeout")
                    self._fire_drop_alert_if_needed(event.topic, "block_timeout")
                    self._write_to_durable_sink(event, "block_timeout")
                    logger.warning(
                        "Event queue full (block timeout %.1fms), dropping event: %s",
                        effective_block_timeout_ms,
                        event.topic,
                    )
                    return False

            # Default: drop newest event (AT_MOST_ONCE semantics)
            self._increment_publish_stat("dropped_newest")
            self._fire_drop_alert_if_needed(event.topic, "drop_newest")
            self._write_to_durable_sink(event, "drop_newest")
            logger.warning("Event queue full, dropping event: %s", event.topic)
            return False

    async def publish_batch(self, events: List[MessagingEvent]) -> int:
        """Publish multiple events.

        Args:
            events: List of events to publish

        Returns:
            Number of events successfully queued
        """
        success_count = 0
        for event in events:
            try:
                if await self.publish(event):
                    success_count += 1
            except EventPublishError:
                continue
        return success_count

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        """Subscribe to events matching a pattern.

        Args:
            pattern: Topic pattern (e.g., "tool.*", "agent.message")
            handler: Async callback for events

        Returns:
            Handle for managing the subscription
        """
        subscription_id = uuid.uuid4().hex[:12]

        subscription = _Subscription(
            id=subscription_id,
            pattern=pattern,
            handler=handler,
            is_active=True,
        )

        with self._lock:
            self._subscriptions[subscription_id] = subscription
            self._index_subscription(subscription)

        # Create handle with unsubscribe capability
        handle = SubscriptionHandle(
            subscription_id=subscription_id,
            pattern=pattern,
            is_active=True,
        )

        # Bind unsubscribe method

        async def bound_unsubscribe() -> None:
            await self.unsubscribe(handle)

        handle.unsubscribe = bound_unsubscribe

        logger.debug(f"Subscribed to pattern '{pattern}' (id: {subscription_id})")
        return handle

    async def unsubscribe(self, handle: SubscriptionHandle) -> bool:
        """Unsubscribe using a handle.

        Args:
            handle: Subscription handle from subscribe()

        Returns:
            True if unsubscribed successfully
        """
        with self._lock:
            subscription = self._subscriptions.get(handle.subscription_id)
            if subscription:
                subscription.is_active = False
                self._deindex_subscription(subscription)
                del self._subscriptions[handle.subscription_id]
                handle.is_active = False
                logger.debug(f"Unsubscribed from pattern '{handle.pattern}'")
                return True
        return False

    async def _dispatch_loop(self) -> None:
        """Background loop that dispatches events to handlers."""
        while self._is_connected:
            try:
                # Wait for event with timeout to allow checking is_connected
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    continue

                # Get matching subscriptions via indexed lookup
                with self._lock:
                    matched_ids: List[str] = []
                    # O(1) exact topic match
                    exact_ids = self._exact_topic_index.get(event.topic, [])
                    matched_ids.extend(exact_ids)
                    # Wildcard matching with cache (O(1) on cache hit)
                    cached_wildcards = self._wildcard_match_cache.get(event.topic)
                    if cached_wildcards is not None:
                        matched_ids.extend(cached_wildcards)
                    else:
                        wildcard_matches = [
                            sid
                            for sid in self._wildcard_subscriptions
                            if self._compiled_patterns.get(sid, lambda t: False)(event.topic)
                        ]
                        self._wildcard_match_cache[event.topic] = wildcard_matches
                        matched_ids.extend(wildcard_matches)
                    subscriptions = [
                        self._subscriptions[sid]
                        for sid in matched_ids
                        if sid in self._subscriptions and self._subscriptions[sid].is_active
                    ]

                # Dispatch to handlers concurrently
                for subscription in subscriptions:
                    try:
                        task = asyncio.create_task(
                            self._safe_call_handler(subscription.handler, event)
                        )
                        self._pending_tasks.add(task)
                        task.add_done_callback(self._pending_tasks.discard)
                    except Exception:
                        logger.warning(
                            "Failed to dispatch event handler",
                            exc_info=True,
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.warning(
                    "Unexpected error in event dispatch loop",
                    exc_info=True,
                )

    async def _safe_call_handler(self, handler: EventHandler, event: MessagingEvent) -> None:
        """Safely call a handler, catching exceptions."""
        try:
            await handler(event)
        except Exception as e:
            logger.warning(f"Event handler error for {event.topic}: {e}")

    def get_subscription_count(self) -> int:
        """Get number of active subscriptions."""
        with self._lock:
            return sum(1 for s in self._subscriptions.values() if s.is_active)

    def get_queue_depth(self) -> int:
        """Get current event queue depth."""
        return self._event_queue.qsize()

    def get_queue_pressure_stats(self) -> Dict[str, Any]:
        """Get queue overflow policy and pressure/drop counters."""
        with self._publish_stats_lock:
            stats = dict(self._publish_stats)
            per_topic = dict(self._per_topic_drop_counts)
        return {
            "queue_depth": self.get_queue_depth(),
            "queue_maxsize": self._queue_maxsize,
            "overflow_policy": self._queue_overflow_policy,
            "block_timeout_ms": self._queue_overflow_block_timeout_ms,
            "topic_overflow_policies": dict(self._queue_overflow_topic_policies),
            "topic_block_timeout_ms": dict(self._queue_overflow_topic_block_timeout_ms),
            "per_topic_drops": per_topic,
            "stats": stats,
        }


# =============================================================================
# Lazy Backend Proxy
# =============================================================================


def _is_backend_connected(backend: IEventBackend) -> bool:
    """Check connection state using protocol property with compatibility fallback."""
    try:
        return bool(backend.is_connected)
    except Exception:
        return bool(getattr(backend, "_is_connected", False))


class LazyInitEventBackend:
    """Proxy backend that defers backend construction until first use.

    Useful for optional/heavy distributed backends (Kafka/Redis/SQS/RabbitMQ),
    where importing clients or constructing backend objects can add startup cost
    even if no events are emitted in a process.
    """

    def __init__(
        self,
        *,
        config: BackendConfig,
        backend_type: BackendType,
        factory: Callable[[BackendConfig], IEventBackend],
    ) -> None:
        self._config = config
        self._selected_type = backend_type
        self._factory = factory
        self._backend: Optional[IEventBackend] = None
        self._lock = threading.Lock()

    @property
    def backend_type(self) -> BackendType:
        """Requested backend type (available before concrete construction)."""
        return self._selected_type

    @property
    def is_connected(self) -> bool:
        """Connection status for the underlying backend (False if not initialized)."""
        backend = self._backend
        if backend is None:
            return False
        return _is_backend_connected(backend)

    def _get_or_create_backend(self) -> IEventBackend:
        """Create the underlying backend once, thread-safely."""
        backend = self._backend
        if backend is not None:
            return backend

        with self._lock:
            backend = self._backend
            if backend is None:
                backend = self._factory(self._config)
                self._backend = backend
        return backend

    async def connect(self) -> None:
        backend = self._get_or_create_backend()
        await backend.connect()

    async def disconnect(self) -> None:
        backend = self._backend
        if backend is None:
            return
        await backend.disconnect()

    async def health_check(self) -> bool:
        backend = self._get_or_create_backend()
        return await backend.health_check()

    async def publish(self, event: MessagingEvent) -> bool:
        backend = self._get_or_create_backend()
        return await backend.publish(event)

    async def publish_batch(self, events: List[MessagingEvent]) -> int:
        backend = self._get_or_create_backend()
        return await backend.publish_batch(events)

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        backend = self._get_or_create_backend()
        return await backend.subscribe(pattern, handler)

    async def unsubscribe(self, handle: SubscriptionHandle) -> bool:
        backend = self._backend
        if backend is None:
            return False
        return await backend.unsubscribe(handle)


# =============================================================================
# Backend Factory
# =============================================================================

# Registry of backend factories
_backend_factories: Dict[BackendType, Callable[[BackendConfig], IEventBackend]] = {}


def register_backend_factory(
    backend_type: BackendType,
    factory: Callable[[BackendConfig], IEventBackend],
) -> None:
    """Register a backend factory.

    External packages can register their own backend implementations:

    Example:
        from victor.core.events.backends import register_backend_factory
        from my_package import KafkaEventBackend

        register_backend_factory(
            BackendType.KAFKA,
            lambda config: KafkaEventBackend(config)
        )
    """
    _backend_factories[backend_type] = factory


def create_event_backend(
    config: Optional[BackendConfig] = None,
    *,
    backend_type: Optional[BackendType] = None,
    lazy_init: bool = False,
    enable_batching: bool = False,
) -> IEventBackend:
    """Factory function to create event backends.

    Creates the appropriate backend based on configuration or explicit type.
    If no backend is registered for the requested type, falls back to InMemory.

    Args:
        config: Optional backend configuration
        backend_type: Override backend type (uses config.backend_type if not set)
        lazy_init: If True, defer backend object construction until first operation.
            Recommended for heavyweight distributed backends.
        enable_batching: If True, wrap the backend in a BatchingEventBackend decorator.

    Returns:
        IEventBackend implementation
    """
    config = config or BackendConfig()
    selected_type = backend_type or config.backend_type

    # Resolve factory, falling back to in-memory if unregistered.
    factory = _backend_factories.get(selected_type)
    if factory is None and selected_type != BackendType.IN_MEMORY:
        logger.warning(
            f"Backend type '{selected_type.value}' not registered, " f"falling back to IN_MEMORY"
        )
        selected_type = BackendType.IN_MEMORY
        factory = _backend_factories.get(BackendType.IN_MEMORY)

    # Guaranteed by built-in registration below, but keep defensive fallback.
    if factory is None:

        def factory(cfg: BackendConfig) -> InMemoryEventBackend:
            return InMemoryEventBackend(cfg)

    # In-memory backend is lightweight; instantiate directly.
    if selected_type == BackendType.IN_MEMORY or not lazy_init:
        backend = factory(config)
    else:
        backend = LazyInitEventBackend(
            config=config,
            backend_type=selected_type,
            factory=factory,
        )

    # Wrap in batching decorator if requested
    if enable_batching:
        from victor.core.events.batching_backend import BatchingEventBackend

        return BatchingEventBackend(backend, config)

    return backend


# Register built-in backend
register_backend_factory(
    BackendType.IN_MEMORY,
    lambda config: InMemoryEventBackend(config),
)


# =============================================================================
# Specialized Bus Classes
# =============================================================================


class ObservabilityBus:
    """Specialized event bus for observability (metrics, tracing, logging).

    Optimized for:
    - High throughput
    - Lossy delivery OK (AT_MOST_ONCE)
    - Batching and sampling
    - Low latency

    Example:
        bus = ObservabilityBus()
        await bus.connect()

        # Emit metrics
        await bus.emit("metric.latency", {"value": 42.5, "unit": "ms"})

        # Subscribe to all metrics
        await bus.subscribe("metric.*", lambda e: print(e.data))
    """

    def __init__(
        self,
        backend: Optional[IEventBackend] = None,
        config: Optional[BackendConfig] = None,
    ) -> None:
        """Initialize observability bus.

        Args:
            backend: Optional pre-configured backend
            config: Configuration (ignored if backend provided)
        """
        if backend:
            self._backend = backend
        else:
            cfg = config or BackendConfig.for_observability()
            self._backend = create_event_backend(cfg)

        # Track exporters (for compatibility with old EventBus API)
        self._exporters: List[Any] = []
        self._exporter_handles: List[SubscriptionHandle] = []
        self._pending_exporters: List[tuple[Any, EventHandler]] = (
            []
        )  # (exporter, handler) awaiting subscription

        # Track background tasks for proper cleanup (prevents pytest hangs)
        self._background_tasks: Set[asyncio.Task] = set()

    @property
    def backend(self) -> IEventBackend:
        """Get underlying backend."""
        return self._backend

    def get_delivery_pressure_stats(self) -> Dict[str, Any]:
        """Get backend queue/drop pressure stats when available."""
        getter = getattr(self._backend, "get_queue_pressure_stats", None)
        if callable(getter):
            try:
                return getter()
            except Exception as e:
                logger.debug("Failed to get backend pressure stats: %s", e)
        return {}

    def register_drop_alert_handler(
        self,
        handler: Callable[[Dict[str, Any]], None],
        *,
        threshold: int = 100,
    ) -> None:
        """Wire a drop-alert callback into the underlying backend.

        No-op when the backend does not support per-topic drop tracking
        (e.g. distributed backends that do not expose drop counters).
        """
        backend = self._backend
        if hasattr(backend, "_drop_alert_callback"):
            backend._drop_alert_callback = handler
            backend._drop_alert_threshold = threshold

    async def connect(self) -> None:
        """Connect the backend."""
        await self._backend.connect()

    async def disconnect(self) -> None:
        """Disconnect the backend."""
        await self._backend.disconnect()

    def add_exporter(self, exporter: Any) -> None:
        """Add an event exporter to the bus.

        Exporters are handlers that write events to external systems
        (files, databases, APIs). They receive all events emitted through the bus.

        Note: The exporter will be subscribed on the next call to emit().

        Args:
            exporter: Exporter object with export(event) method
        """

        # Create handler for this exporter
        async def _export_handler(event):
            try:
                if hasattr(exporter, "export"):
                    await exporter.export(event)
                elif callable(exporter):
                    # Exporter is a callable function
                    await exporter(event)
            except Exception as e:
                logger.debug(f"Exporter error: {e}")

        # Store exporter and handler for subscription on next emit
        self._exporters.append(exporter)
        self._pending_exporters.append((exporter, _export_handler))

    async def _subscribe_exporter(self, exporter: Any, handler: EventHandler) -> None:
        """Subscribe an exporter to the event bus.

        This is called asynchronously to handle subscription when event loop is available.

        Args:
            exporter: The exporter object
            handler: The event handler for the exporter
        """
        # Subscribe to all events ("*" pattern)
        try:
            handle = await self.subscribe("*", handler)
            # Store the handle so we can unsubscribe later
            self._exporter_handles.append((exporter, handle))

            # Remove from pending list
            self._pending_exporters = [
                (exp, h) for exp, h in self._pending_exporters if exp != exporter
            ]
        except Exception as e:
            logger.debug(f"Failed to subscribe exporter: {e}")

    def remove_exporter(self, exporter: Any) -> None:
        """Remove an event exporter from the bus.

        Args:
            exporter: Exporter object to remove
        """
        # Remove from list
        if exporter in self._exporters:
            self._exporters.remove(exporter)

        # Remove from pending
        self._pending_exporters = [
            (exp, h) for exp, h in self._pending_exporters if exp != exporter
        ]

        # Unsubscribe from backend (sync - deactivates subscription immediately)
        for i, (exp, handle) in enumerate(self._exporter_handles):
            if exp == exporter:
                # Deactivate subscription directly (sync operation)
                handle.is_active = False
                # Remove from backend's subscription dict
                if hasattr(self._backend, "_subscriptions"):
                    with self._backend._lock:
                        if handle.subscription_id in self._backend._subscriptions:
                            del self._backend._subscriptions[handle.subscription_id]
                # Remove from handles list
                del self._exporter_handles[i]
                break

    async def emit(
        self,
        topic: str,
        data: Dict[str, Any],
        *,
        source: str = "victor",
        correlation_id: Optional[str] = None,
    ) -> bool:
        """Emit an observability event.

        Args:
            topic: Event topic (e.g., "metric.latency", "trace.span")
            data: Event payload
            source: Event source identifier
            correlation_id: Optional correlation ID

        Returns:
            True if event was emitted
        """
        # Process pending exporter subscriptions
        if self._pending_exporters:
            pending = list(self._pending_exporters)  # Copy to avoid modification during iteration
            for exporter, handler in pending:
                await self._subscribe_exporter(exporter, handler)

        event = MessagingEvent(
            topic=topic,
            data=data,
            source=source,
            correlation_id=correlation_id,
            delivery_guarantee=DeliveryGuarantee.AT_MOST_ONCE,
        )

        try:
            # Auto-connect backend if not connected (lazy initialization)
            if not _is_backend_connected(self._backend):
                await self._backend.connect()

            return await self._backend.publish(event)
        except EventPublishError as e:
            # Log but don't propagate - observability errors shouldn't crash the app
            logger.debug(f"Failed to emit observability event: {e}")
            return False

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        """Subscribe to observability events."""
        return await self._backend.subscribe(pattern, handler)

    async def unsubscribe(self, handle: SubscriptionHandle) -> bool:
        """Unsubscribe from observability events.

        Args:
            handle: Subscription handle returned by subscribe()

        Returns:
            True if unsubscribed successfully
        """
        return await self._backend.unsubscribe(handle)

    def emit_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ) -> None:
        """Emit an error event (fire-and-forget).

        This is a compatibility method for code that expects EventBus-like interface.
        Events are emitted asynchronously without waiting for completion.

        Args:
            error: The exception.
            context: Optional error context.
            recoverable: Whether error is recoverable.
        """
        # Schedule emit as fire-and-forget task
        import asyncio

        asyncio.create_task(
            self.emit(
                topic="error",
                data={
                    "message": str(error),
                    "type": type(error).__name__,
                    "recoverable": recoverable,
                    **(context or {}),
                },
                source="observability",
            )
        )

    def emit_lifecycle_event(
        self,
        event_name: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a lifecycle event (fire-and-forget).

        Convenience method for graph/workflow lifecycle events
        (graph_started, node_start, node_end, graph_completed).
        """
        from victor.core.events.emit_helper import emit_event_sync

        emit_event_sync(
            self,
            f"lifecycle.{event_name}",
            data or {},
            source="observability",
        )

    def emit_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Emit a metric event (fire-and-forget).

        Convenience method that wraps emit() for metric data.
        Safely handles both async and sync calling contexts.

        Args:
            metric_name: Name of the metric (e.g., "latency", "token_count").
            value: Metric value.
            unit: Unit of measurement (e.g., "ms", "count").
            tags: Optional key-value tags for the metric.
        """
        import asyncio

        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop - skip emission in sync context
            # This prevents "coroutine was never awaited" warnings
            return

        # Create and track the background task for proper cleanup
        task = loop.create_task(
            self.emit(
                topic=f"metric.{metric_name}",
                data={
                    "metric_name": metric_name,
                    "value": value,
                    "unit": unit,
                    "tags": tags or {},
                },
                source="observability",
            )
        )
        # Track task for cleanup (prevents pytest hangs)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def cleanup_background_tasks(self) -> None:
        """Clean up all pending background tasks.

        Waits for all fire-and-forget tasks to complete.
        Call this before shutting down the bus to prevent task leaks.
        """
        if self._background_tasks:
            # Wait for all tasks with timeout to prevent hanging
            await asyncio.wait(self._background_tasks, timeout=5.0)
            self._background_tasks.clear()


class AgentMessageBus:
    """Specialized event bus for cross-agent communication.

    Optimized for:
    - Reliable delivery (AT_LEAST_ONCE)
    - Point-to-point and broadcast messaging
    - Request-response patterns
    - Agent coordination

    Example:
        bus = AgentMessageBus()
        await bus.connect()

        # Send to specific agent
        await bus.send("agent.task", {"action": "analyze"}, to_agent="researcher")

        # Broadcast to all agents
        await bus.broadcast("agent.status", {"phase": "planning"})

        # Subscribe to messages for this agent
        await bus.subscribe_agent("executor", handler)
    """

    def __init__(
        self,
        backend: Optional[IEventBackend] = None,
        config: Optional[BackendConfig] = None,
    ) -> None:
        """Initialize agent message bus.

        Args:
            backend: Optional pre-configured backend
            config: Configuration (ignored if backend provided)
        """
        if backend:
            self._backend = backend
        else:
            cfg = config or BackendConfig.for_agent_messaging()
            self._backend = create_event_backend(cfg)

    @property
    def backend(self) -> IEventBackend:
        """Get underlying backend."""
        return self._backend

    async def connect(self) -> None:
        """Connect the backend."""
        await self._backend.connect()

    async def disconnect(self) -> None:
        """Disconnect the backend."""
        await self._backend.disconnect()

    async def send(
        self,
        topic: str,
        data: Dict[str, Any],
        *,
        to_agent: str,
        from_agent: str = "coordinator",
        correlation_id: Optional[str] = None,
    ) -> bool:
        """Send a message to a specific agent.

        Args:
            topic: Message topic
            data: Message payload
            to_agent: Target agent ID
            from_agent: Sender agent ID
            correlation_id: Optional correlation ID

        Returns:
            True if message was sent
        """
        event = MessagingEvent(
            topic=f"agent.{to_agent}.{topic}",
            data={
                **data,
                "_from_agent": from_agent,
                "_to_agent": to_agent,
            },
            source=from_agent,
            correlation_id=correlation_id,
            partition_key=to_agent,
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        )

        # Auto-connect backend if not connected (lazy initialization)
        if not _is_backend_connected(self._backend):
            await self._backend.connect()

        return await self._backend.publish(event)

    async def broadcast(
        self,
        topic: str,
        data: Dict[str, Any],
        *,
        from_agent: str = "coordinator",
        correlation_id: Optional[str] = None,
    ) -> bool:
        """Broadcast a message to all agents.

        Args:
            topic: Message topic
            data: Message payload
            from_agent: Sender agent ID
            correlation_id: Optional correlation ID

        Returns:
            True if broadcast was sent
        """
        event = MessagingEvent(
            topic=f"agent.broadcast.{topic}",
            data={
                **data,
                "_from_agent": from_agent,
                "_broadcast": True,
            },
            source=from_agent,
            correlation_id=correlation_id,
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        )

        # Auto-connect backend if not connected (lazy initialization)
        if not _is_backend_connected(self._backend):
            await self._backend.connect()

        return await self._backend.publish(event)

    async def subscribe_agent(
        self,
        agent_id: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        """Subscribe to messages for a specific agent.

        Args:
            agent_id: Agent ID to subscribe for
            handler: Message handler

        Returns:
            Subscription handle
        """
        # Subscribe to directed messages and broadcasts
        directed_handle = await self._backend.subscribe(
            f"agent.{agent_id}.*",
            handler,
        )
        broadcast_handle = await self._backend.subscribe(
            "agent.broadcast.*",
            handler,
        )

        # Create combined handle
        combined = SubscriptionHandle(
            subscription_id=f"{directed_handle.subscription_id}+{broadcast_handle.subscription_id}",
            pattern=f"agent.{agent_id}.* + agent.broadcast.*",
            is_active=True,
        )

        async def unsubscribe_both() -> None:
            await directed_handle.unsubscribe()
            await broadcast_handle.unsubscribe()
            combined.is_active = False

        combined.unsubscribe = unsubscribe_both
        return combined


# =============================================================================
# Convenience Functions for DI Integration
# =============================================================================


def get_observability_bus() -> ObservabilityBus:
    """Get the ObservabilityBus instance from DI container.

    This is a convenience function that retrieves the ObservabilityBus
    from the global service container. If not registered, it will
    automatically register and return a new instance using backend from settings.

    The backend type is determined by settings.event_backend_type, which defaults
    to "in_memory" if not configured.

    Returns:
        ObservabilityBus instance

    Example:
        >>> from victor.core.events.backends import get_observability_bus
        >>>
        >>> bus = get_observability_bus()
        >>> await bus.emit("tool.start", {"tool": "read_file"})
    """
    from victor.core.container import get_container
    from victor.config.settings import get_settings

    container = get_container()

    # Auto-register if not exists
    if not container.is_registered(ObservabilityBus):

        def create_bus(container):
            # Read settings to determine backend
            settings = get_settings()
            lazy_init = bool(getattr(settings, "event_backend_lazy_init", True))
            enable_batching = bool(getattr(settings, "event_batching_enabled", False))
            backend_config = build_backend_config_from_settings(settings)
            backend = create_event_backend(
                config=backend_config,
                lazy_init=lazy_init,
                enable_batching=enable_batching,
            )

            return ObservabilityBus(backend=backend)

        from victor.core.container import ServiceLifetime

        container.register(ObservabilityBus, create_bus, ServiceLifetime.SINGLETON)

    bus = container.get(ObservabilityBus)

    # Optional sync emit metrics reporter bootstrap.
    # This is idempotent via start_emit_sync_metrics_reporter singleton semantics.
    try:
        settings = get_settings()
        if getattr(settings, "event_emit_sync_metrics_enabled", False):
            from victor.core.events.emit_helper import start_emit_sync_metrics_reporter

            start_emit_sync_metrics_reporter(
                interval_seconds=getattr(
                    settings, "event_emit_sync_metrics_interval_seconds", 60.0
                ),
                topic=getattr(
                    settings,
                    "event_emit_sync_metrics_topic",
                    "core.events.emit_sync.metrics",
                ),
                reset_after_emit=getattr(
                    settings, "event_emit_sync_metrics_reset_after_emit", False
                ),
                event_bus_provider=lambda bus=bus: bus,
            )
    except Exception as e:
        logger.debug("Failed to initialize sync emit metrics reporter: %s", e)

    return bus


def get_agent_message_bus() -> AgentMessageBus:
    """Get the AgentMessageBus instance from DI container.

    This is a convenience function that retrieves the AgentMessageBus
    from the global service container. If not registered, it will
    automatically register and return a new instance using backend from settings.

    The backend type is determined by settings.event_backend_type, which defaults
    to "in_memory" if not configured.

    Returns:
        AgentMessageBus instance

    Example:
        >>> from victor.core.events.backends import get_agent_message_bus
        >>>
        >>> bus = get_agent_message_bus()
        >>> await bus.send("task", {"action": "analyze"}, to_agent="researcher")
    """
    from victor.core.container import get_container
    from victor.config.settings import get_settings

    container = get_container()

    # Auto-register if not exists
    if not container.is_registered(AgentMessageBus):

        def create_bus(container):
            # Read settings to determine backend
            settings = get_settings()
            lazy_init = bool(getattr(settings, "event_backend_lazy_init", True))
            enable_batching = bool(
                getattr(settings, "event_agent_messaging_batching_enabled", False)
            )
            backend_config = build_backend_config_from_settings(settings)
            backend = create_event_backend(
                config=backend_config,
                lazy_init=lazy_init,
                enable_batching=enable_batching,
            )

            return AgentMessageBus(backend=backend)

        from victor.core.container import ServiceLifetime

        container.register(AgentMessageBus, create_bus, ServiceLifetime.SINGLETON)

    return container.get(AgentMessageBus)


def get_event_backend() -> IEventBackend:
    """Get the IEventBackend instance from DI container.

    This is a convenience function that retrieves the event backend
    from the global service container.

    Returns:
        IEventBackend instance

    Example:
        >>> from victor.core.events.backends import get_event_backend
        >>>
        >>> backend = get_event_backend()
        >>> await backend.publish(MessagingEvent(topic="test", data={}))
    """
    from victor.core.container import get_container

    container = get_container()
    return container.get(IEventBackend)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Backends
    "InMemoryEventBackend",
    "LazyInitEventBackend",
    # Specialized buses
    "ObservabilityBus",
    "AgentMessageBus",
    # Factory
    "create_event_backend",
    "register_backend_factory",
    # Convenience functions for DI integration
    "get_observability_bus",
    "get_agent_message_bus",
    "get_event_backend",
]
