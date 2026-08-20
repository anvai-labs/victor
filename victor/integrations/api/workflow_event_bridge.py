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

"""Workflow Event Bridge for Real-Time Visualization.

Bridges workflow execution events from ObservabilityBus to WebSocket clients
for real-time workflow visualization updates.

This module handles:
- Subscribing to workflow lifecycle events
- Filtering events by workflow_id
- Broadcasting to WebSocket clients
- Converting WorkflowStreamChunk to WebSocket events
- Managing per-workflow subscriptions

Architecture:
    ObservabilityBus → WorkflowEventBridge → WebSocket → Browser (Cytoscape.js)

Example:
    from victor.integrations.api.workflow_event_bridge import WorkflowEventBridge
    from victor.core.events import get_observability_bus

    bus = get_observability_bus()
    bridge = WorkflowEventBridge(bus)
    await bridge.start()

    # Subscribe to workflow events
    await bridge.subscribe_workflow("wf_abc123", websocket_handler)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from victor.core.events import (
    ObservabilityBus,
    MessagingEvent,
    get_observability_bus,
    SubscriptionHandle,
)
from victor.workflows.streaming import WorkflowEventType, WorkflowStreamChunk

logger = logging.getLogger(__name__)


class WSMessageType(str, Enum):
    """WebSocket message types."""

    # Client → Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"

    # Server → Client
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"
    PONG = "pong"
    EVENT = "event"
    ERROR = "error"


@dataclass
class WSMessage:
    """WebSocket message.

    Attributes:
        type: Message type
        workflow_id: Workflow identifier
        data: Message data
        timestamp: Message timestamp
        id: Unique message ID
    """

    type: WSMessageType
    workflow_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "id": self.id,
        }

        if self.workflow_id:
            result["workflow_id"] = self.workflow_id

        if self.data:
            result["data"] = self.data

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class WorkflowSubscription:
    """Subscription to workflow events.

    Attributes:
        workflow_id: Workflow identifier
        client_id: Client identifier
        send_func: Async function to send messages to client
        subscribed_at: Subscription timestamp
        last_activity: Last activity timestamp
    """

    workflow_id: str
    client_id: str
    send_func: Callable[[str], Any]
    subscribed_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    async def send(self, message: WSMessage) -> None:
        """Send a message to this subscriber.

        Args:
            message: WSMessage to send
        """
        try:
            await self.send_func(message.to_json())
            self.last_activity = time.time()
        except Exception as e:
            logger.error(f"Failed to send message to client {self.client_id}: {e}")
            raise


class WorkflowEventBridge:
    """Bridges workflow events to WebSocket clients.

    Subscribes to ObservabilityBus for workflow events and broadcasts
    them to subscribed WebSocket clients filtered by workflow_id.

    Attributes:
        _event_bus: ObservabilityBus instance
        _subscriptions: Dict mapping (workflow_id, client_id) to subscription
        _running: Whether the bridge is running
        _event_handlers: Registered event handlers per workflow
    """

    def __init__(self, event_bus: Optional[ObservabilityBus] = None):
        """Initialize the WorkflowEventBridge.

        Args:
            event_bus: Optional ObservabilityBus instance
        """
        self._event_bus = event_bus
        self._subscriptions: Dict[tuple[str, str], WorkflowSubscription] = {}
        self._running = False
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._event_subscriptions: List[SubscriptionHandle] = []

    async def start(self) -> None:
        """Start the event bridge.

        Subscribes to workflow lifecycle events on the ObservabilityBus.
        """
        if self._running:
            return

        if not self._event_bus:
            self._event_bus = get_observability_bus()

        # Subscribe to all workflow events
        workflow_patterns = [
            "workflow.*",
            "lifecycle.workflow.*",
        ]

        for pattern in workflow_patterns:
            try:
                handle = await self._event_bus.subscribe(pattern, self._on_workflow_event)
                self._event_subscriptions.append(handle)
            except Exception as e:
                logger.warning(f"Failed to subscribe to {pattern}: {e}")

        self._running = True
        logger.info("WorkflowEventBridge started")

    async def stop(self) -> None:
        """Stop the event bridge.

        Unsubscribes from events and clears all subscriptions.
        """
        if not self._running:
            return

        # Unsubscribe from events using the handles
        for handle in self._event_subscriptions:
            try:
                await handle.unsubscribe()
            except Exception as e:
                logger.warning(f"Failed to unsubscribe from event bus: {e}")

        self._event_subscriptions.clear()
        self._subscriptions.clear()
        self._event_handlers.clear()
        self._running = False

        logger.info("WorkflowEventBridge stopped")

    async def subscribe_workflow(
        self,
        workflow_id: str,
        client_id: str,
        send_func: Callable[[str], Any],
    ) -> WorkflowSubscription:
        """Subscribe a client to workflow events.

        Args:
            workflow_id: Workflow identifier
            client_id: Client identifier
            send_func: Async function to send messages

        Returns:
            WorkflowSubscription instance

        Example:
            async def ws_send(msg: str):
                await websocket.send_text(msg)

            subscription = await bridge.subscribe_workflow("wf_123", "client_1", ws_send)
        """
        key = (workflow_id, client_id)

        if key in self._subscriptions:
            # Update existing subscription
            self._subscriptions[key].send_func = send_func
            self._subscriptions[key].last_activity = time.time()
        else:
            # Create new subscription
            self._subscriptions[key] = WorkflowSubscription(
                workflow_id=workflow_id,
                client_id=client_id,
                send_func=send_func,
            )

        logger.info(f"Client {client_id} subscribed to workflow {workflow_id}")

        # Send subscription confirmation
        await self._subscriptions[key].send(
            WSMessage(
                type=WSMessageType.SUBSCRIBED,
                workflow_id=workflow_id,
                data={"message": f"Subscribed to workflow {workflow_id}"},
            )
        )

        return self._subscriptions[key]

    async def unsubscribe_workflow(
        self,
        workflow_id: str,
        client_id: str,
    ) -> None:
        """Unsubscribe a client from workflow events.

        Args:
            workflow_id: Workflow identifier
            client_id: Client identifier
        """
        key = (workflow_id, client_id)

        if key in self._subscriptions:
            del self._subscriptions[key]
            logger.info(f"Client {client_id} unsubscribed from workflow {workflow_id}")

    async def handle_websocket_connection(
        self,
        websocket: Any,  # WebSocket object
        workflow_id: str,
        client_id: Optional[str] = None,
    ) -> None:
        """Handle a WebSocket connection for workflow events.

        Args:
            websocket: WebSocket connection object
            workflow_id: Workflow to subscribe to
            client_id: Optional client identifier

        Example:
            @app.websocket("/workflows/{workflow_id}/stream")
            async def workflow_stream(websocket: WebSocket, workflow_id: str):
                await websocket.accept()
                bridge = WorkflowEventBridge()
                await bridge.handle_websocket_connection(websocket, workflow_id)
        """
        client_id = client_id or uuid.uuid4().hex[:12]

        # Subscribe to workflow
        await self.subscribe_workflow(
            workflow_id,
            client_id,
            websocket.send_text,  # type: ignore
        )

        try:
            # Handle incoming messages
            while True:
                data = await websocket.receive_text()
                await self._handle_client_message(workflow_id, client_id, data)

        except Exception as e:
            logger.warning(f"WebSocket connection error: {e}")
        finally:
            # Cleanup
            await self.unsubscribe_workflow(workflow_id, client_id)

    async def _handle_client_message(
        self,
        workflow_id: str,
        client_id: str,
        message: str,
    ) -> None:
        """Handle an incoming message from a client.

        Args:
            workflow_id: Workflow identifier
            client_id: Client identifier
            message: Message content
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "ping":
                # Respond with pong
                key = (workflow_id, client_id)
                if key in self._subscriptions:
                    await self._subscriptions[key].send(
                        WSMessage(
                            type=WSMessageType.PONG,
                            data={"timestamp": time.time()},
                        )
                    )

            elif msg_type == "unsubscribe":
                # Unsubscribe from workflow
                await self.unsubscribe_workflow(workflow_id, client_id)

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client {client_id}")
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")

    async def _on_workflow_event(self, event: MessagingEvent) -> None:
        """Handle a workflow event from ObservabilityBus.

        Args:
            event: MessagingEvent from ObservabilityBus
        """
        try:
            # Extract workflow_id from event
            workflow_id = event.data.get("workflow_id")
            if not workflow_id:
                return

            # Convert to WS message
            ws_message = self._event_to_ws_message(event)
            if not ws_message:
                return

            # Broadcast to all subscribers of this workflow
            tasks = []
            for key, subscription in self._subscriptions.items():
                sub_workflow_id, _ = key
                if sub_workflow_id == workflow_id:
                    tasks.append(subscription.send(ws_message))

            # Send to all subscribers concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error handling workflow event: {e}")

    def _event_to_ws_message(self, event: MessagingEvent) -> Optional[WSMessage]:
        """Convert ObservabilityBus event to WS message.

        Args:
            event: MessagingEvent from ObservabilityBus

        Returns:
            WSMessage or None if event should be ignored
        """
        # Map event topics to workflow event types
        topic = event.topic

        # Determine event type
        event_type = None
        if "workflow.start" in topic:
            event_type = "workflow_start"
        elif "workflow.complete" in topic or "workflow.end" in topic:
            event_type = "workflow_complete"
        elif "workflow.error" in topic:
            event_type = "workflow_error"
        elif "node.start" in topic:
            event_type = "node_start"
        elif "node.complete" in topic or "node.end" in topic:
            event_type = "node_complete"
        elif "node.error" in topic:
            event_type = "node_error"
        elif "progress" in topic:
            event_type = "progress_update"

        if not event_type:
            return None

        # Build message data
        message_data = {
            "event_type": event_type,
            "timestamp": event.timestamp or datetime.now(timezone.utc).isoformat(),
        }

        # Add event-specific data
        if "node" in topic:
            message_data["node_id"] = event.data.get("node_id")
            message_data["node_name"] = event.data.get("node_name")
        elif "progress" in topic:
            message_data["progress"] = event.data.get("progress", 0.0)

        # Merge in additional event data
        message_data.update(event.data)

        # Check if this is a final event
        is_final = event_type in ["workflow_complete", "workflow_error"]
        if is_final:
            message_data["is_final"] = True

        return WSMessage(
            type=WSMessageType.EVENT,
            workflow_id=event.data.get("workflow_id"),
            data=message_data,
            timestamp=event.timestamp or time.time(),
        )

    async def broadcast_workflow_event(
        self,
        workflow_id: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Broadcast a workflow event to all subscribers.

        Args:
            workflow_id: Workflow identifier
            event_type: Event type string
            data: Event data

        Example:
            await bridge.broadcast_workflow_event(
                "wf_123",
                "node_complete",
                {"node_id": "analyze", "progress": 50.0}
            )
        """
        message = WSMessage(
            type=WSMessageType.EVENT,
            workflow_id=workflow_id,
            data={
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **data,
            },
        )

        tasks = []
        for key, subscription in self._subscriptions.items():
            sub_workflow_id, _ = key
            if sub_workflow_id == workflow_id:
                tasks.append(subscription.send(message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_subscriber_count(self, workflow_id: Optional[str] = None) -> int:
        """Get number of active subscribers.

        Args:
            workflow_id: Optional workflow filter

        Returns:
            Number of subscribers
        """
        if workflow_id:
            return sum(1 for (wid, _) in self._subscriptions.keys() if wid == workflow_id)
        return len(self._subscriptions)

    def get_workflow_ids(self) -> List[str]:
        """Get list of workflow IDs with active subscribers.

        Returns:
            List of workflow IDs
        """
        return list(set(wid for (wid, _) in self._subscriptions.keys()))


def workflow_stream_chunk_to_ws_event(
    chunk: WorkflowStreamChunk,
) -> Dict[str, Any]:
    """Convert WorkflowStreamChunk to WebSocket event format.

    Args:
        chunk: WorkflowStreamChunk from streaming executor

    Returns:
        Dictionary compatible with WSMessage data field

    Example:
        from victor.workflows.streaming import WorkflowStreamChunk, WorkflowEventType

        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_START,
            workflow_id="wf_123",
            node_id="analyze",
        )

        event_data = workflow_stream_chunk_to_ws_event(chunk)
        # Returns: {"event_type": "node_start", "node_id": "analyze", ...}
    """
    event_data = {
        "event_type": chunk.event_type.value,
        "workflow_id": chunk.workflow_id,
        "timestamp": (
            chunk.timestamp.isoformat()
            if chunk.timestamp
            else datetime.now(timezone.utc).isoformat()
        ),
    }

    # Add optional fields
    if chunk.node_id:
        event_data["node_id"] = chunk.node_id

    if chunk.node_name:
        event_data["node_name"] = chunk.node_name

    if chunk.error:
        event_data["error"] = str(chunk.error)

    if chunk.metadata:
        event_data.update(chunk.metadata)

    # Add progress if available
    if chunk.progress is not None:
        event_data["progress"] = chunk.progress

    return event_data
