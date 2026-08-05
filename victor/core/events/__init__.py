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

"""Event System for Victor.

This package provides:
1. Protocol-based backends for distributed messaging
2. Specialized buses for observability and agent communication

Key Components:
- IEventBackend: Protocol for swappable backends (Kafka, SQS, RabbitMQ, etc.)
- ObservabilityBus: High-throughput telemetry (lossy OK)
- AgentMessageBus: Reliable cross-agent communication

Usage - Protocol-Based Backends:
    from victor.core.events import (
        create_event_backend,
        MessagingEvent,
        ObservabilityBus,
        AgentMessageBus,
    )

    # Observability (high volume, lossy OK)
    obs_bus = ObservabilityBus()
    await obs_bus.connect()
    await obs_bus.emit("metric.latency", {"value": 42.5})

    # Agent communication (delivery guarantees)
    agent_bus = AgentMessageBus()
    await agent_bus.connect()
    await agent_bus.send("task", {"action": "analyze"}, to_agent="researcher")

Backend Types:
    - IN_MEMORY: Default, single-instance (victor.core.events.backends)
    - KAFKA: Distributed, high-throughput (register external implementation)
    - SQS: AWS serverless (register external implementation)
    - RABBITMQ: Traditional MQ (register external implementation)
    - REDIS: Fast streams (register external implementation)

See Also:
    - victor.core.events.protocols: Protocol definitions
    - victor.core.events.backends: Backend implementations

Note:
    The canonical event type system is victor.framework.events.EventType.
"""

# Protocol-based event system
from victor.core.events.protocols import (
    # Core types
    MessagingEvent,
    SubscriptionHandle,
    DeliveryGuarantee,
    BackendType,
    BackendConfig,
    # Protocols
    IEventBackend,
    IEventPublisher,
    IEventSubscriber,
    # Exceptions
    EventPublishError,
    EventSubscriptionError,
)

from victor.core.events.backends import (
    # Backends
    InMemoryEventBackend,
    LazyInitEventBackend,
    # Specialized buses
    ObservabilityBus,
    AgentMessageBus,
    # Factory
    create_event_backend,
    register_backend_factory,
    # Convenience functions for DI integration
    get_observability_bus,
    get_agent_message_bus,
    get_event_backend,
)

# Lightweight backends (optional import - may not be needed in all contexts)
try:
    from victor.core.events.backends_lightweight import (
        SQLiteEventBackend,
        register_lightweight_backends,
    )

    _LIGHTWEIGHT_AVAILABLE = True
except ImportError:
    _LIGHTWEIGHT_AVAILABLE = False
    SQLiteEventBackend = None  # type: ignore
    register_lightweight_backends = None  # type: ignore

# Adapters for bridging with existing systems
from victor.core.events.adapter import (
    victor_event_to_event as victor_event_to_event_simple,
    EventBusAdapter,
    TeamMessageBusAdapter,
)

# Migration utilities - REMOVED (migration complete, no longer needed)
# The migrator.py module caused circular imports after event_bus.py deletion
# Use adapter.py for backward compatibility if needed

# Sync wrappers for gradual migration
from victor.core.events.sync_wrapper import (
    SyncEventWrapper,
    SyncEventHandler,
)
from victor.core.events.emit_helper import (
    emit_event_sync,
    get_emit_sync_stats,
    reset_emit_sync_stats,
    emit_sync_metrics_event,
    EmitSyncMetricsReporter,
    start_emit_sync_metrics_reporter,
    stop_emit_sync_metrics_reporter,
)
from victor.core.events.taxonomy import EventTaxonomyRegistry

__all__ = [
    # Protocol-based - Core types
    "MessagingEvent",
    "SubscriptionHandle",
    "DeliveryGuarantee",
    "BackendType",
    "BackendConfig",
    # Protocol-based - Protocols
    "IEventBackend",
    "IEventPublisher",
    "IEventSubscriber",
    # Protocol-based - Exceptions
    "EventPublishError",
    "EventSubscriptionError",
    # Protocol-based - Backends
    "InMemoryEventBackend",
    "LazyInitEventBackend",
    # Protocol-based - Specialized buses
    "ObservabilityBus",
    "AgentMessageBus",
    # Protocol-based - Factory
    "create_event_backend",
    "register_backend_factory",
    # Convenience functions for DI integration
    "get_observability_bus",
    "get_agent_message_bus",
    "get_event_backend",
    # Lightweight backends (optional)
    "SQLiteEventBackend",
    "register_lightweight_backends",
    # Adapters for bridging (simple versions)
    "victor_event_to_event_simple",
    "EventBusAdapter",
    "TeamMessageBusAdapter",
    # Migration utilities removed (migration complete)
    # Sync wrappers for gradual migration
    "SyncEventWrapper",
    "SyncEventHandler",
    # Sync emit helper APIs
    # Event taxonomy
    "EventTaxonomyRegistry",
    # Sync emit helper APIs
    "emit_event_sync",
    "get_emit_sync_stats",
    "reset_emit_sync_stats",
    "emit_sync_metrics_event",
    "EmitSyncMetricsReporter",
    "start_emit_sync_metrics_reporter",
    "stop_emit_sync_metrics_reporter",
]
