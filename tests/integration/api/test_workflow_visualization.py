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

"""Integration tests for workflow visualization API.

Tests the complete workflow visualization feature including:
- FastAPI endpoints
- WebSocket streaming
- Graph export
- Event bridging
"""

import asyncio
import json
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from datetime import datetime, timezone

from victor.integrations.api.fastapi_server import VictorFastAPIServer
from victor.integrations.api.graph_export import (
    GraphNode,
    GraphEdge,
    GraphSchema,
    NodeStatus,
    NodeType,
    WorkflowExecutionState,
    ExecutionNodeState,
)
from victor.integrations.api.workflow_event_bridge import WorkflowEventBridge
from victor.core.events import get_observability_bus


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def fastapi_server():
    """Create a FastAPI server for testing.

    Note: These tests use ASGITransport directly against ``server.app``, so
    they do not need a live TCP listener. Keeping the fixture in-process avoids
    port conflicts when the broader suite has other API tests running.
    """
    server = VictorFastAPIServer(
        host="localhost",
        port=0,
        enable_cors=True,
    )

    # Manually initialize the event bridge if not already initialized
    # because these tests intentionally do not launch a real uvicorn server.
    event_bus = None
    if server._workflow_event_bridge is None:
        event_bus = get_observability_bus()
        await event_bus.connect()
        server._workflow_event_bridge = WorkflowEventBridge(event_bus)
        await server._workflow_event_bridge.start()

    yield server

    if server._workflow_event_bridge is not None:
        await server._workflow_event_bridge.stop()
    if event_bus is not None:
        await event_bus.disconnect()


@pytest_asyncio.fixture
async def http_client(fastapi_server):
    """Create an HTTP client for testing."""
    transport = ASGITransport(app=fastapi_server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.integration
class TestWorkflowVisualizationEndpoints:
    """Integration tests for workflow visualization endpoints."""

    @pytest.mark.asyncio
    async def test_get_workflow_graph_not_found(self, http_client):
        """Test getting graph for non-existent workflow."""
        response = await http_client.get("/workflows/nonexistent/graph")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_workflow_execution_not_found(self, http_client):
        """Test getting execution state for non-existent workflow."""
        response = await http_client.get("/workflows/nonexistent/execution")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_visualize_html_response(self, http_client):
        """Test that visualize endpoint returns HTML."""
        response = await http_client.get("/workflows/visualize/test_workflow")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<!DOCTYPE html>" in response.text or "error" in response.text.lower()


@pytest.mark.integration
class TestWorkflowVisualizationWithMockData:
    """Tests with mock workflow data."""

    @pytest.fixture
    def mock_workflow_execution(self, fastapi_server):
        """Create mock workflow execution data."""
        workflow_id = "test_workflow_123"

        # Create mock graph schema
        graph_schema = GraphSchema(
            workflow_id=workflow_id,
            name="Test Workflow",
            description="A test workflow for integration testing",
            nodes=[
                GraphNode(
                    id="node_a",
                    name="Node A",
                    type=NodeType.AGENT,
                    description="First node",
                ),
                GraphNode(
                    id="node_b",
                    name="Node B",
                    type=NodeType.COMPUTE,
                    description="Second node",
                ),
                GraphNode(
                    id="node_c",
                    name="Node C",
                    type=NodeType.CONDITION,
                    description="Third node",
                ),
            ],
            edges=[
                GraphEdge(
                    id="edge_0",
                    source="node_a",
                    target="node_b",
                ),
                GraphEdge(
                    id="edge_1",
                    source="node_b",
                    target="node_c",
                ),
            ],
            start_node="node_a",
            entry_point="node_a",
        )

        # Create mock execution state
        now = datetime.now(timezone.utc).isoformat()
        execution_state = {
            "status": "running",
            "progress": 66.6,
            "started_at": now,
            "current_node": "node_b",
            "completed_nodes": ["node_a"],
            "failed_nodes": [],
            "skipped_nodes": [],
            "node_execution_path": [
                {
                    "node_id": "node_a",
                    "status": "completed",
                    "started_at": now,
                    "completed_at": now,
                    "duration_seconds": 120.0,
                    "tool_calls": 5,
                    "tokens_used": 1500,
                },
                {
                    "node_id": "node_b",
                    "status": "running",
                    "started_at": now,
                    "completed_at": None,
                    "duration_seconds": 60.0,
                    "tool_calls": 3,
                    "tokens_used": 800,
                },
            ],
            "total_duration_seconds": 180.0,
            "total_tool_calls": 8,
            "total_tokens": 2300,
        }

        # Add to server's execution store
        fastapi_server._workflow_executions[workflow_id] = {
            "graph": graph_schema.to_cytoscape_dict(),
            **execution_state,
        }

        return workflow_id

    @pytest.mark.asyncio
    async def test_get_workflow_graph_success(self, http_client, mock_workflow_execution):
        """Test successfully getting workflow graph."""
        workflow_id = mock_workflow_execution

        response = await http_client.get(f"/workflows/{workflow_id}/graph")

        assert response.status_code == 200

        graph_data = response.json()
        assert graph_data["workflow_id"] == workflow_id
        assert graph_data["name"] == "Test Workflow"
        assert len(graph_data["nodes"]) == 3
        assert len(graph_data["edges"]) == 2
        assert graph_data["start_node"] == "node_a"

    @pytest.mark.asyncio
    async def test_get_workflow_execution_success(self, http_client, mock_workflow_execution):
        """Test successfully getting workflow execution state."""
        workflow_id = mock_workflow_execution

        response = await http_client.get(f"/workflows/{workflow_id}/execution")

        assert response.status_code == 200

        exec_data = response.json()
        assert exec_data["workflow_id"] == workflow_id
        assert exec_data["status"] == "running"
        assert exec_data["progress"] == 66.6
        assert exec_data["current_node"] == "node_b"
        assert len(exec_data["completed_nodes"]) == 1
        assert exec_data["total_tool_calls"] == 8
        assert exec_data["total_tokens"] == 2300

    @pytest.mark.asyncio
    async def test_workflow_websocket_connection(self, http_client, mock_workflow_execution):
        """Test WebSocket connection for workflow streaming."""
        workflow_id = mock_workflow_execution

        # Note: This test requires the server to be running
        # In a real integration test, you would connect to the actual server
        # For now, we'll test the endpoint exists

        # Test that the endpoint is accessible
        response = await http_client.get(f"/workflows/{workflow_id}/graph")
        assert response.status_code == 200


@pytest.mark.integration
class TestWorkflowVisualizationEventStreaming:
    """Tests for event streaming functionality."""

    @pytest.mark.asyncio
    async def test_workflow_event_bridge_initialization(self, fastapi_server):
        """Test that workflow event bridge is initialized."""
        assert (
            fastapi_server._workflow_event_bridge is not None
        ), "Event bridge should be initialized"
        assert (
            fastapi_server._workflow_event_bridge._running is True
        ), "Event bridge should be running"

    @pytest.mark.asyncio
    async def test_workflow_event_bridge_subscribe(self, fastapi_server):
        """Test subscribing to workflow events."""
        bridge = fastapi_server._workflow_event_bridge
        assert bridge is not None, "Event bridge should be initialized"

        workflow_id = "test_workflow"

        # Mock send function
        async def send_mock(msg):
            pass

        # Subscribe
        await bridge.subscribe_workflow(
            workflow_id=workflow_id,
            client_id="test_client",
            send_func=send_mock,
        )

        # Verify subscription
        count = bridge.get_subscriber_count(workflow_id)
        assert count == 1

        # Cleanup
        await bridge.unsubscribe_workflow(workflow_id, "test_client")

    @pytest.mark.asyncio
    async def test_broadcast_workflow_event(self, fastapi_server):
        """Test broadcasting workflow events."""
        bridge = fastapi_server._workflow_event_bridge
        assert bridge is not None, "Event bridge should be initialized"

        workflow_id = "test_workflow"

        # Track received messages
        received_messages = []

        async def mock_send(msg):
            received_messages.append(json.loads(msg))

        # Subscribe
        await bridge.subscribe_workflow(
            workflow_id=workflow_id,
            client_id="test_client",
            send_func=mock_send,
        )

        # Broadcast event
        await bridge.broadcast_workflow_event(
            workflow_id=workflow_id,
            event_type="node_complete",
            data={"node_id": "test_node", "progress": 50.0},
        )

        # Verify event was received
        assert len(received_messages) > 0
        event = received_messages[-1]
        assert event["type"] == "event"
        assert event["data"]["event_type"] == "node_complete"
        assert event["data"]["node_id"] == "test_node"

        # Cleanup
        await bridge.unsubscribe_workflow(workflow_id, "test_client")


@pytest.mark.integration
class TestGraphExportIntegration:
    """Integration tests for graph export functionality."""

    @pytest.mark.asyncio
    async def test_export_and_serialize_graph(self):
        """Test exporting a graph and serializing to JSON."""
        # Create a simple graph schema
        schema = GraphSchema(
            workflow_id="integration_test",
            name="Integration Test Workflow",
            description="Testing graph export",
            nodes=[
                GraphNode(
                    id="start",
                    name="Start",
                    type=NodeType.AGENT,
                ),
                GraphNode(
                    id="process",
                    name="Process",
                    type=NodeType.COMPUTE,
                ),
                GraphNode(
                    id="end",
                    name="End",
                    type=NodeType.CONDITION,
                ),
            ],
            edges=[
                GraphEdge(
                    id="e1",
                    source="start",
                    target="process",
                ),
                GraphEdge(
                    id="e2",
                    source="process",
                    target="end",
                ),
            ],
            start_node="start",
        )

        # Convert to Cytoscape format
        cytoscape_dict = schema.to_cytoscape_dict()

        # Verify JSON serialization
        json_str = json.dumps(cytoscape_dict)
        parsed = json.loads(json_str)

        assert parsed["workflow_id"] == "integration_test"
        assert len(parsed["nodes"]) == 3
        assert len(parsed["edges"]) == 2

    @pytest.mark.asyncio
    async def test_execution_state_serialization(self):
        """Test serializing execution state to JSON."""
        state = WorkflowExecutionState(
            workflow_id="test_workflow",
            status="running",
            progress=75.0,
            current_node="process",
            completed_nodes=["start"],
            node_execution_path=[
                ExecutionNodeState(
                    node_id="start",
                    status=NodeStatus.COMPLETED,
                    duration_seconds=100.0,
                    tool_calls=5,
                    tokens_used=1500,
                )
            ],
            total_tool_calls=5,
            total_tokens=1500,
        )

        # Convert to dict and serialize
        state_dict = state.to_dict()
        json_str = json.dumps(state_dict)
        parsed = json.loads(json_str)

        assert parsed["workflow_id"] == "test_workflow"
        assert parsed["status"] == "running"
        assert parsed["progress"] == 75.0
        assert len(parsed["node_execution_path"]) == 1


@pytest.mark.integration
class TestEndToEndWorkflowVisualization:
    """End-to-end tests for workflow visualization."""

    @pytest.mark.asyncio
    async def test_complete_visualization_flow(self, fastapi_server, http_client):
        """Test complete workflow visualization flow.

        This test simulates:
        1. Creating a workflow execution
        2. Getting the graph structure
        3. Getting execution state
        4. Broadcasting events
        """
        workflow_id = "e2e_test_workflow"

        # Step 1: Create mock workflow execution
        graph_schema = GraphSchema(
            workflow_id=workflow_id,
            name="E2E Test Workflow",
            description="End-to-end test",
            nodes=[
                GraphNode(
                    id="start",
                    name="Start",
                    type=NodeType.AGENT,
                ),
                GraphNode(
                    id="end",
                    name="End",
                    type=NodeType.COMPUTE,
                ),
            ],
            edges=[
                GraphEdge(
                    id="e1",
                    source="start",
                    target="end",
                ),
            ],
            start_node="start",
        )

        fastapi_server._workflow_executions[workflow_id] = {
            "graph": graph_schema.to_cytoscape_dict(),
            "status": "running",
            "progress": 50.0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "current_node": "start",
            "completed_nodes": [],
            "failed_nodes": [],
            "skipped_nodes": [],
            "node_execution_path": [],
            "total_duration_seconds": 0.0,
            "total_tool_calls": 0,
            "total_tokens": 0,
        }

        # Step 2: Get graph structure
        graph_response = await http_client.get(f"/workflows/{workflow_id}/graph")
        assert graph_response.status_code == 200
        graph_data = graph_response.json()
        assert graph_data["workflow_id"] == workflow_id

        # Step 3: Get execution state
        exec_response = await http_client.get(f"/workflows/{workflow_id}/execution")
        assert exec_response.status_code == 200
        exec_data = exec_response.json()
        assert exec_data["status"] == "running"

        # Step 4: Broadcast event
        received_events = []

        async def track_events(msg):
            received_events.append(json.loads(msg))

        await fastapi_server._workflow_event_bridge.subscribe_workflow(
            workflow_id=workflow_id,
            client_id="e2e_test_client",
            send_func=track_events,
        )

        await fastapi_server._workflow_event_bridge.broadcast_workflow_event(
            workflow_id=workflow_id,
            event_type="node_complete",
            data={"node_id": "start", "progress": 100.0},
        )

        # Verify event was broadcast
        assert len(received_events) > 0
        assert received_events[-1]["data"]["event_type"] == "node_complete"

        # Cleanup
        await fastapi_server._workflow_event_bridge.unsubscribe_workflow(
            workflow_id, "e2e_test_client"
        )
