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

"""Unit tests for workflow_event_bridge module.

Tests workflow event bridging and WebSocket streaming functionality.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.integrations.api.workflow_event_bridge import (
    WSMessageType,
    WSMessage,
    WorkflowSubscription,
    WorkflowEventBridge,
    workflow_stream_chunk_to_ws_event,
)
from victor.core.events import MessagingEvent
from victor.workflows.streaming import (
    WorkflowEventType,
    WorkflowStreamChunk,
)


class TestWSMessageType:
    """Test WSMessageType enum."""

    def test_message_types(self):
        """Test that all expected message types exist."""
        assert WSMessageType.SUBSCRIBE == "subscribe"
        assert WSMessageType.UNSUBSCRIBE == "unsubscribe"
        assert WSMessageType.PING == "ping"
        assert WSMessageType.SUBSCRIBED == "subscribed"
        assert WSMessageType.PONG == "pong"
        assert WSMessageType.EVENT == "event"
        assert WSMessageType.ERROR == "error"


class TestWSMessage:
    """Test WSMessage dataclass."""

    def test_ws_message_creation(self):
        """Test creating a basic WebSocket message."""
        message = WSMessage(
            type=WSMessageType.EVENT,
            workflow_id="test_workflow",
        )

        assert message.type == WSMessageType.EVENT
        assert message.workflow_id == "test_workflow"
        assert message.data == {}
        assert message.id is not None

    def test_ws_message_with_data(self):
        """Test creating a WebSocket message with data."""
        data = {"event_type": "node_start", "node_id": "test_node"}
        message = WSMessage(
            type=WSMessageType.EVENT,
            workflow_id="test_workflow",
            data=data,
        )

        assert message.data == data

    def test_ws_message_to_dict(self):
        """Test converting WebSocket message to dictionary."""
        message = WSMessage(
            type=WSMessageType.EVENT,
            workflow_id="test_workflow",
            data={"test": "value"},
        )

        message_dict = message.to_dict()

        assert message_dict["type"] == "event"
        assert message_dict["workflow_id"] == "test_workflow"
        assert message_dict["data"]["test"] == "value"
        assert "timestamp" in message_dict
        assert "id" in message_dict

    def test_ws_message_to_json(self):
        """Test converting WebSocket message to JSON."""
        message = WSMessage(
            type=WSMessageType.SUBSCRIBED,
            workflow_id="test_workflow",
        )

        json_str = message.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "subscribed"
        assert parsed["workflow_id"] == "test_workflow"


class TestWorkflowSubscription:
    """Test WorkflowSubscription dataclass."""

    @pytest.mark.asyncio
    async def test_subscription_send(self):
        """Test sending a message through subscription."""
        # Create a mock send function
        send_mock = AsyncMock()

        subscription = WorkflowSubscription(
            workflow_id="test_workflow",
            client_id="client_1",
            send_func=send_mock,
        )

        # Create a message
        message = WSMessage(
            type=WSMessageType.EVENT,
            workflow_id="test_workflow",
            data={"test": "data"},
        )

        # Send the message
        await subscription.send(message)

        # Verify send was called
        send_mock.assert_called_once()
        sent_json = send_mock.call_args[0][0]
        sent_data = json.loads(sent_json)

        assert sent_data["type"] == "event"
        assert sent_data["data"]["test"] == "data"

    @pytest.mark.asyncio
    async def test_subscription_send_updates_activity(self):
        """Test that sending updates last_activity timestamp."""
        import time

        send_mock = AsyncMock()

        subscription = WorkflowSubscription(
            workflow_id="test_workflow",
            client_id="client_1",
            send_func=send_mock,
        )

        old_activity = subscription.last_activity

        # Wait a bit to ensure timestamp changes
        await asyncio.sleep(0.01)

        message = WSMessage(
            type=WSMessageType.EVENT,
            workflow_id="test_workflow",
        )

        await subscription.send(message)

        assert subscription.last_activity > old_activity


class TestWorkflowEventBridge:
    """Test WorkflowEventBridge class."""

    @pytest.fixture
    def event_bridge(self):
        """Create a WorkflowEventBridge instance for testing."""
        bridge = WorkflowEventBridge(event_bus=None)
        return bridge

    @pytest.mark.asyncio
    async def test_bridge_start_stop(self, event_bridge):
        """Test starting and stopping the bridge."""
        # Mock the event bus
        mock_bus = MagicMock()
        mock_bus.subscribe = AsyncMock()
        mock_bus.unsubscribe = AsyncMock()

        await event_bridge.start()
        assert event_bridge._running is True

        await event_bridge.stop()
        assert event_bridge._running is False

    @pytest.mark.asyncio
    async def test_subscribe_workflow(self, event_bridge):
        """Test subscribing to a workflow."""
        send_mock = AsyncMock()

        await event_bridge.subscribe_workflow(
            workflow_id="test_workflow",
            client_id="client_1",
            send_func=send_mock,
        )

        # Verify subscription was created
        key = ("test_workflow", "client_1")
        assert key in event_bridge._subscriptions

        # Verify subscription confirmation was sent
        send_mock.assert_called_once()
        sent_json = send_mock.call_args[0][0]
        sent_data = json.loads(sent_json)

        assert sent_data["type"] == "subscribed"

    @pytest.mark.asyncio
    async def test_unsubscribe_workflow(self, event_bridge):
        """Test unsubscribing from a workflow."""
        send_mock = AsyncMock()

        # First subscribe
        await event_bridge.subscribe_workflow(
            workflow_id="test_workflow",
            client_id="client_1",
            send_func=send_mock,
        )

        # Then unsubscribe
        await event_bridge.unsubscribe_workflow(
            workflow_id="test_workflow",
            client_id="client_1",
        )

        # Verify subscription was removed
        key = ("test_workflow", "client_1")
        assert key not in event_bridge._subscriptions

    def test_get_subscriber_count(self, event_bridge):
        """Test getting subscriber count."""
        # Initially should be 0
        assert event_bridge.get_subscriber_count() == 0

        # Add a mock subscription
        event_bridge._subscriptions[("wf1", "client1")] = MagicMock()
        event_bridge._subscriptions[("wf1", "client2")] = MagicMock()
        event_bridge._subscriptions[("wf2", "client3")] = MagicMock()

        # Total count
        assert event_bridge.get_subscriber_count() == 3

        # Filtered by workflow
        assert event_bridge.get_subscriber_count("wf1") == 2
        assert event_bridge.get_subscriber_count("wf2") == 1

    def test_get_workflow_ids(self, event_bridge):
        """Test getting list of workflow IDs."""
        # Add mock subscriptions
        event_bridge._subscriptions[("wf1", "client1")] = MagicMock()
        event_bridge._subscriptions[("wf1", "client2")] = MagicMock()
        event_bridge._subscriptions[("wf2", "client3")] = MagicMock()

        workflow_ids = event_bridge.get_workflow_ids()

        assert len(workflow_ids) == 2
        assert "wf1" in workflow_ids
        assert "wf2" in workflow_ids

    @pytest.mark.asyncio
    async def test_broadcast_workflow_event(self, event_bridge):
        """Test broadcasting a workflow event."""
        # Create mock subscriptions
        send_mock1 = AsyncMock()
        send_mock2 = AsyncMock()

        subscription1 = WorkflowSubscription(
            workflow_id="test_workflow",
            client_id="client_1",
            send_func=send_mock1,
        )

        subscription2 = WorkflowSubscription(
            workflow_id="test_workflow",
            client_id="client_2",
            send_func=send_mock2,
        )

        event_bridge._subscriptions[("test_workflow", "client_1")] = subscription1
        event_bridge._subscriptions[("test_workflow", "client_2")] = subscription2

        # Broadcast event
        await event_bridge.broadcast_workflow_event(
            workflow_id="test_workflow",
            event_type="node_complete",
            data={"node_id": "test_node", "progress": 50.0},
        )

        # Verify both subscribers received the event
        assert send_mock1.call_count == 1
        assert send_mock2.call_count == 1

    def test_event_to_ws_message_workflow_start(self, event_bridge):
        """Test converting workflow start event to WS message."""
        event = MessagingEvent(
            topic="workflow.start",
            data={"workflow_id": "test_workflow"},
        )

        ws_message = event_bridge._event_to_ws_message(event)

        assert ws_message is not None
        assert ws_message.type == WSMessageType.EVENT
        assert ws_message.workflow_id == "test_workflow"
        assert ws_message.data["event_type"] == "workflow_start"

    def test_event_to_ws_message_node_complete(self, event_bridge):
        """Test converting node complete event to WS message."""
        event = MessagingEvent(
            topic="workflow.node.complete",
            data={
                "workflow_id": "test_workflow",
                "node_id": "test_node",
                "node_name": "Test Node",
            },
        )

        ws_message = event_bridge._event_to_ws_message(event)

        assert ws_message is not None
        assert ws_message.data["event_type"] == "node_complete"
        assert ws_message.data["node_id"] == "test_node"
        assert ws_message.data["node_name"] == "Test Node"

    def test_event_to_ws_message_ignored_topic(self, event_bridge):
        """Test that events with unrelated topics are ignored."""
        event = MessagingEvent(
            topic="tool.start",
            data={"workflow_id": "test_workflow"},
        )

        ws_message = event_bridge._event_to_ws_message(event)

        assert ws_message is None


class TestWorkflowStreamChunkToWSEvent:
    """Test workflow_stream_chunk_to_ws_event function."""

    def test_convert_node_start_chunk(self):
        """Test converting NODE_START chunk to WS event."""
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_START,
            workflow_id="test_workflow",
            node_id="test_node",
            node_name="Test Node",
        )

        event_data = workflow_stream_chunk_to_ws_event(chunk)

        assert event_data["event_type"] == "node_start"
        assert event_data["workflow_id"] == "test_workflow"
        assert event_data["node_id"] == "test_node"
        assert event_data["node_name"] == "Test Node"
        assert "timestamp" in event_data

    def test_convert_node_complete_chunk_with_progress(self):
        """Test converting NODE_COMPLETE chunk with progress."""
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_COMPLETE,
            workflow_id="test_workflow",
            node_id="test_node",
            progress=75.5,
        )

        event_data = workflow_stream_chunk_to_ws_event(chunk)

        assert event_data["event_type"] == "node_complete"
        assert event_data["progress"] == 75.5

    def test_convert_node_error_chunk(self):
        """Test converting NODE_ERROR chunk."""
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_ERROR,
            workflow_id="test_workflow",
            node_id="test_node",
            error="Test error message",
        )

        event_data = workflow_stream_chunk_to_ws_event(chunk)

        assert event_data["event_type"] == "node_error"
        assert event_data["error"] == "Test error message"

    def test_convert_workflow_complete_chunk(self):
        """Test converting WORKFLOW_COMPLETE chunk."""
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.WORKFLOW_COMPLETE,
            workflow_id="test_workflow",
            progress=100.0,
        )

        event_data = workflow_stream_chunk_to_ws_event(chunk)

        assert event_data["event_type"] == "workflow_complete"
        assert event_data["progress"] == 100.0

    def test_convert_chunk_with_metadata(self):
        """Test converting chunk with additional metadata."""
        metadata = {
            "tool_calls": 5,
            "tokens_used": 1500,
            "duration_seconds": 120.0,
        }

        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_COMPLETE,
            workflow_id="test_workflow",
            node_id="test_node",
            metadata=metadata,
        )

        event_data = workflow_stream_chunk_to_ws_event(chunk)

        assert event_data["tool_calls"] == 5
        assert event_data["tokens_used"] == 1500
        assert event_data["duration_seconds"] == 120.0
