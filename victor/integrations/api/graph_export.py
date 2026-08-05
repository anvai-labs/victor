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

"""Graph Export for Workflow Visualization.

Exports StateGraph and WorkflowDefinition structures to Cytoscape.js-compatible
JSON format for web-based workflow visualization.

This module provides the canonical graph extraction logic for:
- StateGraph compiled workflows
- WorkflowDefinition YAML workflows
- Node and edge metadata
- Execution state mapping

Example:
    from victor.integrations.api.graph_export import export_graph_schema
    from victor.framework.graph import StateGraph

    graph = StateGraph(state_schema)
    graph.add_node("analyze", analyze_fn)
    compiled = graph.compile()

    # Export to Cytoscape.js format
    schema = export_graph_schema(compiled)
    # Returns: {"nodes": [...], "edges": [...], ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.graph import CompiledGraph, Node, Edge
    from victor.workflows.definition import WorkflowDefinition
    from victor.workflows.visualization import WorkflowVisualizer

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Execution status of a workflow node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeType(str, Enum):
    """Type of workflow node."""

    AGENT = "agent"
    COMPUTE = "compute"
    CONDITION = "condition"
    PARALLEL = "parallel"
    HITL = "hitl"  # Human-in-the-loop
    TRANSFORM = "transform"
    START = "start"
    END = "end"


@dataclass
class GraphNode:
    """A node in the workflow graph.

    Attributes:
        id: Unique node identifier
        name: Human-readable name
        type: Node type (agent, compute, condition, etc.)
        description: Node description
        status: Execution status
        metadata: Additional node metadata
        position: Optional position for layout (x, y)
    """

    id: str
    name: str
    type: NodeType
    description: str = ""
    status: NodeStatus = NodeStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    position: Optional[Dict[str, float]] = None

    def to_cytoscape_dict(self) -> Dict[str, Any]:
        """Convert to Cytoscape.js element format.

        Returns:
            Dictionary compatible with Cytoscape.js element format
        """
        element = {
            "data": {
                "id": self.id,
                "label": self.name,
                "type": self.type.value,
                "status": self.status.value,
                "description": self.description,
                **self.metadata,
            }
        }

        # Add status class for styling
        if self.status != NodeStatus.PENDING:
            element["classes"] = f"node-{self.status.value}"

        # Add position if provided
        if self.position:
            element["position"] = self.position

        return element


@dataclass
class GraphEdge:
    """An edge in the workflow graph.

    Attributes:
        id: Unique edge identifier
        source: Source node ID
        target: Target node ID
        label: Optional edge label
        conditional: Whether this is a conditional edge
        metadata: Additional edge metadata
    """

    id: str
    source: str
    target: str
    label: Optional[str] = None
    conditional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_cytoscape_dict(self) -> Dict[str, Any]:
        """Convert to Cytoscape.js element format.

        Returns:
            Dictionary compatible with Cytoscape.js element format
        """
        element = {
            "data": {
                "id": self.id,
                "source": self.source,
                "target": self.target,
                "conditional": self.conditional,
                **self.metadata,
            }
        }

        if self.label:
            element["data"]["label"] = self.label

        return element


@dataclass
class GraphSchema:
    """Complete workflow graph schema.

    Attributes:
        workflow_id: Workflow identifier
        name: Workflow name
        description: Workflow description
        nodes: List of nodes
        edges: List of edges
        start_node: Entry point node ID
        entry_point: Entry point node ID (same as start_node)
        total_nodes: Total number of nodes
        metadata: Additional workflow metadata
    """

    workflow_id: str
    name: str
    description: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    start_node: Optional[str] = None
    entry_point: Optional[str] = None
    total_nodes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate total nodes after initialization."""
        self.total_nodes = len(self.nodes)

    def to_cytoscape_dict(self) -> Dict[str, Any]:
        """Convert to Cytoscape.js format.

        Returns:
            Dictionary with nodes and edges arrays for Cytoscape.js
        """
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "nodes": [node.to_cytoscape_dict() for node in self.nodes],
            "edges": [edge.to_cytoscape_dict() for edge in self.edges],
            "start_node": self.start_node,
            "entry_point": self.entry_point,
            "total_nodes": self.total_nodes,
            "metadata": self.metadata,
        }

    def to_elements_format(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert to Cytoscape.js elements format (nodes + edges).

        Returns:
            Dictionary with 'nodes' and 'edges' keys for cytoscape()
        """
        return {
            "nodes": [node.to_cytoscape_dict() for node in self.nodes],
            "edges": [edge.to_cytoscape_dict() for edge in self.edges],
        }


def export_graph_schema(
    graph: Any,
    workflow_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> GraphSchema:
    """Export a StateGraph or WorkflowDefinition to Cytoscape.js schema.

    Args:
        graph: CompiledGraph or WorkflowDefinition to export
        workflow_id: Optional workflow identifier
        name: Optional workflow name
        description: Optional workflow description

    Returns:
        GraphSchema in Cytoscape.js-compatible format

    Raises:
        TypeError: If graph is not a supported type
        ValueError: If graph structure is invalid
    """
    # Detect graph type
    if hasattr(graph, "get_graph_schema"):
        # StateGraph.CompiledGraph
        return _export_from_compiled_graph(graph, workflow_id, name, description)  # type: ignore
    elif hasattr(graph, "nodes"):
        # WorkflowDefinition
        return _export_from_workflow_definition(
            graph, workflow_id, name, description  # type: ignore
        )
    elif hasattr(graph, "_nodes") and hasattr(graph, "_edges"):
        # WorkflowVisualizer
        return _export_from_visualizer(graph, workflow_id, name, description)  # type: ignore
    else:
        raise TypeError(
            f"Unsupported graph type: {type(graph)}. "
            "Expected CompiledGraph, WorkflowDefinition, or WorkflowVisualizer"
        )


def _export_from_compiled_graph(
    compiled_graph: Any,
    workflow_id: Optional[str],
    name: Optional[str],
    description: Optional[str],
) -> GraphSchema:
    """Export from StateGraph.CompiledGraph.

    Args:
        compiled_graph: CompiledGraph instance
        workflow_id: Optional workflow identifier
        name: Optional workflow name
        description: Optional workflow description

    Returns:
        GraphSchema instance
    """
    try:
        schema = compiled_graph.get_graph_schema()
    except Exception as e:
        logger.error(f"Failed to get graph schema: {e}")
        raise ValueError(f"Invalid CompiledGraph: {e}")

    # Extract nodes
    nodes = []
    node_ids = schema.get("nodes", [])

    for node_id in node_ids:
        # Try to get node metadata
        node_metadata = {}
        if "nodes_map" in schema:
            node_metadata = schema["nodes_map"].get(node_id, {})

        nodes.append(
            GraphNode(
                id=node_id,
                name=node_metadata.get("name", node_id),
                type=NodeType(node_metadata.get("type", "compute")),
                description=node_metadata.get("description", ""),
                metadata=node_metadata,
            )
        )

    # Extract edges
    edges = []
    edge_id = 0
    edges_dict = schema.get("edges", {})

    for source, targets in edges_dict.items():
        if not isinstance(targets, list):
            targets = [targets]

        for target_info in targets:
            if isinstance(target_info, dict):
                target = target_info.get("target")
                conditional = target_info.get("conditional", False)
                label = target_info.get("label")
            else:
                target = target_info
                conditional = False
                label = None

            edges.append(
                GraphEdge(
                    id=f"edge_{edge_id}",
                    source=source,
                    target=target,
                    label=label,
                    conditional=conditional,
                )
            )
            edge_id += 1

    # Get entry point
    entry_point = schema.get("entry_point")
    start_node = entry_point or (node_ids[0] if node_ids else None)

    return GraphSchema(
        workflow_id=workflow_id or compiled_graph.__class__.__name__,
        name=name or "StateGraph Workflow",
        description=description or "StateGraph compiled workflow",
        nodes=nodes,
        edges=edges,
        start_node=start_node,
        entry_point=entry_point,
    )


def _export_from_workflow_definition(
    workflow: Any,
    workflow_id: Optional[str],
    name: Optional[str],
    description: Optional[str],
) -> GraphSchema:
    """Export from WorkflowDefinition.

    Args:
        workflow: WorkflowDefinition instance
        workflow_id: Optional workflow identifier
        name: Optional workflow name
        description: Optional workflow description

    Returns:
        GraphSchema instance
    """
    # Extract nodes from workflow definition
    nodes = []
    edges = []

    workflow_nodes = workflow.nodes if hasattr(workflow, "nodes") else []

    for node_def in workflow_nodes:
        node_id = node_def.get("id")
        if not node_id:
            continue

        # Determine node type
        node_type_str = node_def.get("type", "agent")
        try:
            node_type = NodeType(node_type_str.lower())
        except ValueError:
            # Map common aliases
            type_mapping = {
                "AgentNode": NodeType.AGENT,
                "ComputeNode": NodeType.COMPUTE,
                "ConditionNode": NodeType.CONDITION,
                "ParallelNode": NodeType.PARALLEL,
                "HITLNode": NodeType.HITL,
                "TransformNode": NodeType.TRANSFORM,
            }
            node_type = type_mapping.get(node_type_str, NodeType.COMPUTE)

        nodes.append(
            GraphNode(
                id=node_id,
                name=node_def.get("name", node_id),
                type=node_type,
                description=node_def.get("description", ""),
                metadata={
                    k: v
                    for k, v in node_def.items()
                    if k not in ["id", "name", "type", "description", "next"]
                },
            )
        )

    # Extract edges from 'next' field
    edge_id = 0
    for node_def in workflow_nodes:
        source_id = node_def.get("id")
        if not source_id:
            continue

        next_nodes = node_def.get("next", [])
        if not isinstance(next_nodes, list):
            next_nodes = [next_nodes]

        for next_node in next_nodes:
            if isinstance(next_node, dict):
                # Conditional edge
                for condition_name, target_id in next_node.items():
                    edges.append(
                        GraphEdge(
                            id=f"edge_{edge_id}",
                            source=source_id,
                            target=target_id,
                            label=condition_name,
                            conditional=True,
                        )
                    )
                    edge_id += 1
            else:
                # Normal edge
                edges.append(
                    GraphEdge(
                        id=f"edge_{edge_id}",
                        source=source_id,
                        target=next_node,
                        conditional=False,
                    )
                )
                edge_id += 1

    # Get metadata
    workflow_meta = workflow.metadata if hasattr(workflow, "metadata") else {}

    return GraphSchema(
        workflow_id=workflow_id or workflow_meta.get("id", "unknown"),
        name=name or workflow_meta.get("name", "Workflow"),
        description=description or workflow_meta.get("description", ""),
        nodes=nodes,
        edges=edges,
        start_node=workflow_meta.get("start_node"),
        entry_point=workflow_meta.get("entry_point"),
        metadata=workflow_meta,
    )


def _export_from_visualizer(
    visualizer: Any,
    workflow_id: Optional[str],
    name: Optional[str],
    description: Optional[str],
) -> GraphSchema:
    """Export from WorkflowVisualizer.

    Args:
        visualizer: WorkflowVisualizer instance
        workflow_id: Optional workflow identifier
        name: Optional workflow name
        description: Optional workflow description

    Returns:
        GraphSchema instance
    """
    # Access internal _nodes and _edges
    internal_nodes = visualizer._nodes if hasattr(visualizer, "_nodes") else []
    internal_edges = visualizer._edges if hasattr(visualizer, "_edges") else []

    # Convert nodes
    nodes = []
    for node in internal_nodes:
        node_type_str = getattr(node, "node_type", "compute")
        try:
            node_type = NodeType(node_type_str.lower())
        except ValueError:
            node_type = NodeType.COMPUTE

        nodes.append(
            GraphNode(
                id=node.id,
                name=node.name,
                type=node_type,
                description=getattr(node, "description", ""),
                metadata=getattr(node, "metadata", {}),
            )
        )

    # Convert edges
    edges = []
    for edge in internal_edges:
        edges.append(
            GraphEdge(
                id=f"{edge.source}-{edge.target}",
                source=edge.source,
                target=edge.target,
                label=getattr(edge, "label", None),
                conditional=getattr(edge, "conditional", False),
                metadata=getattr(edge, "metadata", {}),
            )
        )

    # Try to get workflow info
    workflow = visualizer.workflow if hasattr(visualizer, "workflow") else None
    workflow_meta = workflow.metadata if workflow and hasattr(workflow, "metadata") else {}

    return GraphSchema(
        workflow_id=workflow_id or workflow_meta.get("id", "unknown"),
        name=name or workflow_meta.get("name", "Workflow"),
        description=description or workflow_meta.get("description", ""),
        nodes=nodes,
        edges=edges,
        start_node=workflow_meta.get("start_node"),
        entry_point=workflow_meta.get("entry_point"),
        metadata=workflow_meta,
    )


@dataclass
class ExecutionNodeState:
    """Execution state for a single node.

    Attributes:
        node_id: Node identifier
        status: Execution status
        started_at: Start timestamp
        completed_at: Completion timestamp
        duration_seconds: Execution duration
        tool_calls: Number of tool calls made
        tokens_used: Number of tokens used
        error: Error message if failed
    """

    node_id: str
    status: NodeStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    tool_calls: int = 0
    tokens_used: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "error": self.error,
        }


@dataclass
class WorkflowExecutionState:
    """Complete workflow execution state.

    Attributes:
        workflow_id: Workflow identifier
        status: Overall workflow status
        progress: Progress percentage (0-100)
        started_at: Workflow start timestamp
        completed_at: Workflow completion timestamp
        current_node: Currently executing node
        completed_nodes: List of completed node IDs
        failed_nodes: List of failed node IDs
        skipped_nodes: List of skipped node IDs
        node_execution_path: Ordered list of node execution states
        total_duration_seconds: Total execution duration
        total_tool_calls: Total tool calls across all nodes
        total_tokens: Total tokens used
    """

    workflow_id: str
    status: str
    progress: float
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_node: Optional[str] = None
    completed_nodes: List[str] = field(default_factory=list)
    failed_nodes: List[str] = field(default_factory=list)
    skipped_nodes: List[str] = field(default_factory=list)
    node_execution_path: List[ExecutionNodeState] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    total_tool_calls: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "progress": self.progress,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "current_node": self.current_node,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "skipped_nodes": self.skipped_nodes,
            "node_execution_path": [node.to_dict() for node in self.node_execution_path],
            "total_duration_seconds": self.total_duration_seconds,
            "total_tool_calls": self.total_tool_calls,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def create_empty(cls, workflow_id: str) -> "WorkflowExecutionState":
        """Create an empty execution state for a new workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            New WorkflowExecutionState with default values
        """
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            workflow_id=workflow_id,
            status="pending",
            progress=0.0,
            started_at=now,
        )


def get_execution_state(
    workflow_id: str,
    execution_store: Dict[str, Any],
) -> WorkflowExecutionState:
    """Get current execution state for a workflow.

    Args:
        workflow_id: Workflow identifier
        execution_store: In-memory execution store from FastAPI server

    Returns:
        WorkflowExecutionState instance

    Raises:
        KeyError: If workflow_id not found in execution store
    """
    if workflow_id not in execution_store:
        raise KeyError(f"Workflow {workflow_id} not found in execution store")

    exec_data = execution_store[workflow_id]

    # Convert execution data to WorkflowExecutionState
    node_states = []
    for node_data in exec_data.get("node_execution_path", []):
        node_states.append(
            ExecutionNodeState(
                node_id=node_data["node_id"],
                status=NodeStatus(node_data.get("status", "pending")),
                started_at=node_data.get("started_at"),
                completed_at=node_data.get("completed_at"),
                duration_seconds=node_data.get("duration_seconds", 0.0),
                tool_calls=node_data.get("tool_calls", 0),
                tokens_used=node_data.get("tokens_used", 0),
                error=node_data.get("error"),
            )
        )

    return WorkflowExecutionState(
        workflow_id=workflow_id,
        status=exec_data.get("status", "unknown"),
        progress=exec_data.get("progress", 0.0),
        started_at=exec_data.get("started_at"),
        completed_at=exec_data.get("completed_at"),
        current_node=exec_data.get("current_node"),
        completed_nodes=exec_data.get("completed_nodes", []),
        failed_nodes=exec_data.get("failed_nodes", []),
        skipped_nodes=exec_data.get("skipped_nodes", []),
        node_execution_path=node_states,
        total_duration_seconds=exec_data.get("total_duration_seconds", 0.0),
        total_tool_calls=exec_data.get("total_tool_calls", 0),
        total_tokens=exec_data.get("total_tokens", 0),
    )
