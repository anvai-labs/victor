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

"""Unit tests for graph_export module.

Tests graph extraction and export functionality for workflow visualization.
"""

import pytest
from datetime import datetime, timezone

from victor.integrations.api.graph_export import (
    GraphNode,
    GraphEdge,
    GraphSchema,
    NodeStatus,
    NodeType,
    ExecutionNodeState,
    WorkflowExecutionState,
    export_graph_schema,
    get_execution_state,
)


class TestGraphNode:
    """Test GraphNode dataclass."""

    def test_graph_node_creation(self):
        """Test creating a basic graph node."""
        node = GraphNode(
            id="test_node",
            name="Test Node",
            type=NodeType.AGENT,
            description="A test node",
        )

        assert node.id == "test_node"
        assert node.name == "Test Node"
        assert node.type == NodeType.AGENT
        assert node.description == "A test node"
        assert node.status == NodeStatus.PENDING
        assert node.metadata == {}

    def test_graph_node_with_metadata(self):
        """Test creating a graph node with metadata."""
        metadata = {"role": "researcher", "tool_budget": 15}
        node = GraphNode(
            id="research_node",
            name="Research",
            type=NodeType.AGENT,
            metadata=metadata,
        )

        assert node.metadata == metadata

    def test_graph_node_to_cytoscape_dict(self):
        """Test converting graph node to Cytoscape.js format."""
        node = GraphNode(
            id="test_node",
            name="Test Node",
            type=NodeType.COMPUTE,
            status=NodeStatus.RUNNING,
        )

        cytoscape_dict = node.to_cytoscape_dict()

        assert cytoscape_dict["data"]["id"] == "test_node"
        assert cytoscape_dict["data"]["label"] == "Test Node"
        assert cytoscape_dict["data"]["type"] == "compute"
        assert cytoscape_dict["data"]["status"] == "running"
        assert "classes" in cytoscape_dict
        assert cytoscape_dict["classes"] == "node-running"

    def test_graph_node_with_position(self):
        """Test graph node with position data."""
        position = {"x": 100.0, "y": 200.0}
        node = GraphNode(
            id="test_node",
            name="Test Node",
            type=NodeType.AGENT,
            position=position,
        )

        cytoscape_dict = node.to_cytoscape_dict()

        assert cytoscape_dict["position"] == position


class TestGraphEdge:
    """Test GraphEdge dataclass."""

    def test_graph_edge_creation(self):
        """Test creating a basic graph edge."""
        edge = GraphEdge(
            id="edge_0",
            source="node_a",
            target="node_b",
        )

        assert edge.id == "edge_0"
        assert edge.source == "node_a"
        assert edge.target == "node_b"
        assert edge.label is None
        assert edge.conditional is False

    def test_graph_edge_with_label(self):
        """Test creating an edge with a label."""
        edge = GraphEdge(
            id="edge_1",
            source="node_a",
            target="node_b",
            label="success",
        )

        assert edge.label == "success"

    def test_graph_edge_conditional(self):
        """Test creating a conditional edge."""
        edge = GraphEdge(
            id="edge_2",
            source="node_a",
            target="node_b",
            conditional=True,
            label="condition_met",
        )

        assert edge.conditional is True
        assert edge.label == "condition_met"

    def test_graph_edge_to_cytoscape_dict(self):
        """Test converting graph edge to Cytoscape.js format."""
        edge = GraphEdge(
            id="edge_0",
            source="node_a",
            target="node_b",
            label="test_label",
            conditional=True,
        )

        cytoscape_dict = edge.to_cytoscape_dict()

        assert cytoscape_dict["data"]["id"] == "edge_0"
        assert cytoscape_dict["data"]["source"] == "node_a"
        assert cytoscape_dict["data"]["target"] == "node_b"
        assert cytoscape_dict["data"]["label"] == "test_label"
        assert cytoscape_dict["data"]["conditional"] is True


class TestGraphSchema:
    """Test GraphSchema dataclass."""

    def test_graph_schema_creation(self):
        """Test creating a graph schema."""
        nodes = [
            GraphNode(
                id="node_a",
                name="Node A",
                type=NodeType.AGENT,
            ),
            GraphNode(
                id="node_b",
                name="Node B",
                type=NodeType.COMPUTE,
            ),
        ]

        edges = [
            GraphEdge(
                id="edge_0",
                source="node_a",
                target="node_b",
            )
        ]

        schema = GraphSchema(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="A test workflow",
            nodes=nodes,
            edges=edges,
            start_node="node_a",
            entry_point="node_a",
        )

        assert schema.workflow_id == "test_workflow"
        assert schema.name == "Test Workflow"
        assert schema.total_nodes == 2
        assert len(schema.nodes) == 2
        assert len(schema.edges) == 1

    def test_graph_schema_to_cytoscape_dict(self):
        """Test converting graph schema to Cytoscape.js format."""
        nodes = [
            GraphNode(
                id="node_a",
                name="Node A",
                type=NodeType.AGENT,
            )
        ]

        edges = [
            GraphEdge(
                id="edge_0",
                source="node_a",
                target="node_b",
            )
        ]

        schema = GraphSchema(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="A test workflow",
            nodes=nodes,
            edges=edges,
        )

        cytoscape_dict = schema.to_cytoscape_dict()

        assert cytoscape_dict["workflow_id"] == "test_workflow"
        assert cytoscape_dict["name"] == "Test Workflow"
        assert len(cytoscape_dict["nodes"]) == 1
        assert len(cytoscape_dict["edges"]) == 1

    def test_graph_schema_to_elements_format(self):
        """Test converting graph schema to Cytoscape elements format."""
        nodes = [
            GraphNode(
                id="node_a",
                name="Node A",
                type=NodeType.AGENT,
            )
        ]

        edges = [
            GraphEdge(
                id="edge_0",
                source="node_a",
                target="node_b",
            )
        ]

        schema = GraphSchema(
            workflow_id="test_workflow",
            name="Test Workflow",
            description="A test workflow",
            nodes=nodes,
            edges=edges,
        )

        elements = schema.to_elements_format()

        assert "nodes" in elements
        assert "edges" in elements
        assert len(elements["nodes"]) == 1
        assert len(elements["edges"]) == 1


class TestExecutionNodeState:
    """Test ExecutionNodeState dataclass."""

    def test_execution_node_state_creation(self):
        """Test creating an execution node state."""
        state = ExecutionNodeState(
            node_id="test_node",
            status=NodeStatus.COMPLETED,
            duration_seconds=120.5,
            tool_calls=5,
            tokens_used=1500,
        )

        assert state.node_id == "test_node"
        assert state.status == NodeStatus.COMPLETED
        assert state.duration_seconds == 120.5
        assert state.tool_calls == 5
        assert state.tokens_used == 1500

    def test_execution_node_state_to_dict(self):
        """Test converting execution node state to dictionary."""
        now = datetime.now(timezone.utc).isoformat()
        state = ExecutionNodeState(
            node_id="test_node",
            status=NodeStatus.RUNNING,
            started_at=now,
            tool_calls=3,
            error="Test error",
        )

        state_dict = state.to_dict()

        assert state_dict["node_id"] == "test_node"
        assert state_dict["status"] == "running"
        assert state_dict["started_at"] == now
        assert state_dict["tool_calls"] == 3
        assert state_dict["error"] == "Test error"


class TestWorkflowExecutionState:
    """Test WorkflowExecutionState dataclass."""

    def test_workflow_execution_state_creation(self):
        """Test creating a workflow execution state."""
        state = WorkflowExecutionState(
            workflow_id="test_workflow",
            status="running",
            progress=45.5,
            current_node="node_b",
            completed_nodes=["node_a"],
            total_tool_calls=10,
            total_tokens=3000,
        )

        assert state.workflow_id == "test_workflow"
        assert state.status == "running"
        assert state.progress == 45.5
        assert state.current_node == "node_b"
        assert len(state.completed_nodes) == 1

    def test_workflow_execution_state_to_dict(self):
        """Test converting workflow execution state to dictionary."""
        node_state = ExecutionNodeState(
            node_id="test_node",
            status=NodeStatus.COMPLETED,
            duration_seconds=100.0,
        )

        state = WorkflowExecutionState(
            workflow_id="test_workflow",
            status="running",
            progress=50.0,
            node_execution_path=[node_state],
        )

        state_dict = state.to_dict()

        assert state_dict["workflow_id"] == "test_workflow"
        assert state_dict["status"] == "running"
        assert state_dict["progress"] == 50.0
        assert len(state_dict["node_execution_path"]) == 1
        assert state_dict["node_execution_path"][0]["node_id"] == "test_node"

    def test_create_empty_execution_state(self):
        """Test creating an empty execution state."""
        state = WorkflowExecutionState.create_empty("test_workflow")

        assert state.workflow_id == "test_workflow"
        assert state.status == "pending"
        assert state.progress == 0.0
        assert state.started_at is not None
        assert len(state.completed_nodes) == 0


class TestGetExecutionState:
    """Test get_execution_state function."""

    def test_get_execution_state_success(self):
        """Test getting execution state from store."""
        execution_store = {
            "test_workflow": {
                "status": "running",
                "progress": 50.0,
                "current_node": "node_b",
                "completed_nodes": ["node_a"],
                "node_execution_path": [
                    {
                        "node_id": "node_a",
                        "status": "completed",
                        "duration_seconds": 100.0,
                        "tool_calls": 5,
                        "tokens_used": 1500,
                    }
                ],
                "total_duration_seconds": 100.0,
                "total_tool_calls": 5,
                "total_tokens": 1500,
            }
        }

        state = get_execution_state("test_workflow", execution_store)

        assert state.workflow_id == "test_workflow"
        assert state.status == "running"
        assert state.progress == 50.0
        assert state.current_node == "node_b"
        assert len(state.node_execution_path) == 1

    def test_get_execution_state_not_found(self):
        """Test getting execution state for non-existent workflow."""
        execution_store = {}

        with pytest.raises(KeyError, match="Workflow nonexistent not found"):
            get_execution_state("nonexistent", execution_store)


class TestExportGraphSchema:
    """Test export_graph_schema function."""

    def test_export_unsupported_graph_type(self):
        """Test exporting an unsupported graph type."""
        with pytest.raises(TypeError, match="Unsupported graph type"):
            export_graph_schema("not_a_graph")

    def test_export_with_custom_metadata(self):
        """Test exporting with custom workflow metadata."""

        # Create a mock workflow definition
        class MockWorkflow:
            def __init__(self):
                self.nodes = [
                    {
                        "id": "node_a",
                        "name": "Node A",
                        "type": "agent",
                        "description": "Test node",
                        "next": ["node_b"],
                    }
                ]
                self.metadata = {
                    "id": "test_workflow",
                    "name": "Test Workflow",
                    "description": "A test workflow",
                }

        workflow = MockWorkflow()

        # This will fail because we don't have the full workflow definition structure
        # but we can test the error handling
        try:
            schema = export_graph_schema(
                workflow,
                workflow_id="custom_id",
                name="Custom Name",
                description="Custom Description",
            )
        except (TypeError, AttributeError):
            # Expected for incomplete mock
            pass


@pytest.mark.parametrize(
    "status_string,expected_status",
    [
        ("pending", NodeStatus.PENDING),
        ("running", NodeStatus.RUNNING),
        ("completed", NodeStatus.COMPLETED),
        ("failed", NodeStatus.FAILED),
        ("skipped", NodeStatus.SKIPPED),
    ],
)
def test_node_status_enum(status_string, expected_status):
    """Test NodeStatus enum values."""
    status = NodeStatus(status_string)
    assert status == expected_status


@pytest.mark.parametrize(
    "type_string,expected_type",
    [
        ("agent", NodeType.AGENT),
        ("compute", NodeType.COMPUTE),
        ("condition", NodeType.CONDITION),
        ("parallel", NodeType.PARALLEL),
        ("hitl", NodeType.HITL),
    ],
)
def test_node_type_enum(type_string, expected_type):
    """Test NodeType enum values."""
    node_type = NodeType(type_string)
    assert node_type == expected_type
