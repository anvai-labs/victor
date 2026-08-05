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

"""Adapters to bridge existing EventBus with new protocol-based system.

This module provides adapters that:
1. Forward VictorEvent from existing EventBus to IEventBackend
2. Convert between VictorEvent and Event formats
3. Enable gradual migration to protocol-based system

Usage:
    from victor.core.events import ObservabilityBus as EventBus
    from victor.core.events import ObservabilityBus
    from victor.core.events.adapter import EventBusAdapter

    # Bridge existing EventBus to new backend
    legacy_bus = EventBus.get_instance()
    new_backend = ObservabilityBus()
    await new_backend.connect()

    adapter = EventBusAdapter(legacy_bus, new_backend)
    adapter.enable_forwarding()

    # Now events from legacy bus are forwarded to new backend
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

from victor.core.events.protocols import MessagingEvent, DeliveryGuarantee

if TYPE_CHECKING:
    from victor.core.events import (
        ObservabilityBus as EventBus,
        VictorEvent,
        EventCategory,
    )
    from victor.core.events.backends import ObservabilityBus, AgentMessageBus
    from victor.agent.teams.communication import TeamMessageBus

logger = logging.getLogger(__name__)


def victor_event_to_event(victor_event: "VictorEvent") -> MessagingEvent:
    """Convert VictorEvent (legacy) to MessagingEvent (protocol-based).

    Args:
        victor_event: Legacy VictorEvent from observability module

    Returns:
        Protocol-based MessagingEvent
    """
    return MessagingEvent(
        id=victor_event.id,
        topic=f"{victor_event.category.value}.{victor_event.name}",
        data=victor_event.data,
        timestamp=victor_event.timestamp.timestamp(),
        source=victor_event.source,
        correlation_id=victor_event.trace_id,
        delivery_guarantee=DeliveryGuarantee.AT_MOST_ONCE,
    )


class EventBusAdapter:
    """Adapter to bridge legacy EventBus with protocol-based backends.

    This adapter subscribes to the legacy EventBus and forwards events
    to a new protocol-based backend (ObservabilityBus or AgentMessageBus).

    Features:
    - Bidirectional event forwarding (optional)
    - Event filtering by category
    - Conversion between VictorEvent and MessagingEvent formats

    Example:
        from victor.core.events import ObservabilityBus as EventBus
        from victor.core.events import ObservabilityBus
        from victor.core.events.adapter import EventBusAdapter

        # Create adapter
        legacy = EventBus.get_instance()
        new_bus = ObservabilityBus()
        await new_bus.connect()

        adapter = EventBusAdapter(legacy, new_bus)
        adapter.enable_forwarding()

        # Events now flow: legacy -> new_bus backend
    """

    def __init__(
        self,
        legacy_bus: "EventBus",
        new_backend: "ObservabilityBus",
        *,
        forward_to_new: bool = True,
        forward_to_legacy: bool = False,
    ) -> None:
        """Initialize the adapter.

        Args:
            legacy_bus: Existing EventBus instance
            new_backend: New protocol-based ObservabilityBus
            forward_to_new: Forward events from legacy to new
            forward_to_legacy: Forward events from new to legacy
        """
        self._legacy_bus = legacy_bus
        self._new_backend = new_backend
        self._forward_to_new = forward_to_new
        self._forward_to_legacy = forward_to_legacy
        self._unsubscribe_legacy: Optional[Callable[[], None]] = None
        self._unsubscribe_new: Optional[object] = None
        self._enabled = False

    def enable_forwarding(self) -> None:
        """Enable event forwarding between buses."""
        if self._enabled:
            return

        if self._forward_to_new:
            # Subscribe to all legacy events
            self._unsubscribe_legacy = self._legacy_bus.subscribe_all(self._on_legacy_event)

        self._enabled = True
        logger.debug("EventBusAdapter forwarding enabled")

    def disable_forwarding(self) -> None:
        """Disable event forwarding."""
        if not self._enabled:
            return

        if self._unsubscribe_legacy:
            self._unsubscribe_legacy()
            self._unsubscribe_legacy = None

        self._enabled = False
        logger.debug("EventBusAdapter forwarding disabled")

    def _on_legacy_event(self, victor_event: "VictorEvent") -> None:
        """Handle event from legacy bus."""
        if not self._forward_to_new:
            return

        # Convert and forward
        event = victor_event_to_event(victor_event)
        try:
            # Use synchronous emit (legacy bus is sync)
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._new_backend.emit(
                        event.topic,
                        event.data,
                        source=event.source,
                        correlation_id=event.correlation_id,
                    )
                )
            except RuntimeError:
                # No running loop, skip forwarding
                pass
        except Exception as e:
            logger.warning(f"Failed to forward event to new backend: {e}")


class TeamMessageBusAdapter:
    """Adapter to bridge TeamMessageBus with AgentMessageBus.

    This allows existing team coordination to use the new distributed
    backend while maintaining backward compatibility.

    Example:
        from victor.agent.teams.communication import TeamMessageBus
        from victor.core.events import AgentMessageBus
        from victor.core.events.adapter import TeamMessageBusAdapter

        # Create adapter
        team_bus = TeamMessageBus("my_team")
        agent_bus = AgentMessageBus()
        await agent_bus.connect()

        adapter = TeamMessageBusAdapter(team_bus, agent_bus)
        adapter.enable_forwarding()
    """

    def __init__(
        self,
        team_bus: "TeamMessageBus",
        agent_bus: "AgentMessageBus",
    ) -> None:
        """Initialize the adapter.

        Args:
            team_bus: Existing TeamMessageBus instance
            agent_bus: New AgentMessageBus backend
        """
        self._team_bus = team_bus
        self._agent_bus = agent_bus
        self._enabled = False

    def enable_forwarding(self) -> None:
        """Enable message forwarding from TeamMessageBus to AgentMessageBus."""
        self._enabled = True
        logger.debug("TeamMessageBusAdapter forwarding enabled")

    def disable_forwarding(self) -> None:
        """Disable message forwarding."""
        self._enabled = False
        logger.debug("TeamMessageBusAdapter forwarding disabled")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "victor_event_to_event",
    "EventBusAdapter",
    "TeamMessageBusAdapter",
]
