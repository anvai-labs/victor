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

"""Adapters for workflow engine interoperability.

This module provides adapters that bridge the gap between different
workflow engines in Victor:

1. WorkflowBuilder (victor/workflows) - DAG-based, dict context
2. StateGraph (victor/framework/graph.py) - Cyclic, typed state

The adapters enable unified execution while maintaining backward
compatibility with existing workflows.

Design Goals:
- Preserve existing WorkflowBuilder API
- Enable StateGraph features (cycles, typed state) for new workflows
- Single execution path for both workflow types
- Gradual migration from WorkflowBuilder to StateGraph

Usage:
    from victor.workflows.adapters import WorkflowToGraphAdapter

    # Adapt a WorkflowBuilder workflow to StateGraph
    adapter = WorkflowToGraphAdapter()
    state_graph = adapter.adapt(workflow_definition)

    # Execute with StateGraph runtime
    app = state_graph.compile()
    result = await app.invoke(initial_state)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, ConfigDict, field_validator

from victor.core.async_utils import run_sync, run_sync_in_thread
from victor.workflows.definition import (
    WorkflowDefinition,
    WorkflowNode,
    WorkflowNodeType,
)

if TYPE_CHECKING:
    from victor.framework.graph import StateGraph
    from victor.workflows.protocols import IWorkflowNodeRuntimeExecutor

logger = logging.getLogger(__name__)


class AdapterWorkflowStateModel(BaseModel):
    """Standard state for adapted WorkflowBuilder-to-StateGraph flows (Pydantic v2).

    This state model is adapter-specific and intentionally distinct from the
    compiled runtime `victor.workflows.runtime_types.WorkflowState`.

    Migrated from TypedDict to Pydantic for better type safety and validation.

    Attributes:
        context: Original workflow context (from WorkflowBuilder)
        messages: Conversation messages
        current_node: Current node being executed
        visited_nodes: Set of nodes already visited
        results: Results from each node
        error: Error message if any
        is_complete: Whether workflow has completed
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    context: Dict[str, Any] = Field(default_factory=dict)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    current_node: str = Field(default="")
    visited_nodes: List[str] = Field(default_factory=list)
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    is_complete: bool = Field(default=False)

    @field_validator("visited_nodes")
    @classmethod
    def validate_visited_nodes(cls, v: List[str]) -> List[str]:
        """Validate visited_nodes list."""
        if v and len(set(v)) != len(v):
            raise ValueError("visited_nodes must be unique (no duplicates)")
        return v

    # Dict-like interface for StateGraph compatibility
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key (dict-like interface)."""
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        """Return list of keys (dict-like interface)."""
        return [
            "context",
            "messages",
            "current_node",
            "visited_nodes",
            "results",
            "error",
            "is_complete",
        ]

    def values(self) -> List[Any]:
        """Return list of values (dict-like interface)."""
        return [
            self.context,
            self.messages,
            self.current_node,
            self.visited_nodes,
            self.results,
            self.error,
            self.is_complete,
        ]

    def items(self) -> List[Tuple[str, Any]]:
        """Return list of (key, value) tuples (dict-like interface)."""
        return [
            ("context", self.context),
            ("messages", self.messages),
            ("current_node", self.current_node),
            ("visited_nodes", self.visited_nodes),
            ("results", self.results),
            ("error", self.error),
            ("is_complete", self.is_complete),
        ]

    def __getitem__(self, key: str) -> Any:
        """Get item by key (dict-like subscript access)."""
        return getattr(self, key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item by key (dict-like subscript access)."""
        setattr(self, key, value)


# Type alias for backward compatibility
AdapterWorkflowState = AdapterWorkflowStateModel
WorkflowState = AdapterWorkflowStateModel


@dataclass
class AdaptedNode:
    """A workflow node adapted for StateGraph execution.

    Attributes:
        name: Node name
        node_type: Original node type
        handler: Async function to execute
        next_nodes: Static next nodes
        conditional_edges: Conditional routing
        tool_budget: Tool budget for this node
        allowed_tools: Tools allowed in this node
    """

    name: str
    node_type: WorkflowNodeType
    handler: Callable[[AdapterWorkflowStateModel], AdapterWorkflowStateModel]
    next_nodes: List[str] = field(default_factory=list)
    conditional_edges: Dict[str, str] = field(default_factory=dict)
    tool_budget: int = 10
    allowed_tools: List[str] = field(default_factory=list)


class WorkflowToGraphAdapter:
    """Adapts WorkflowDefinition to StateGraph for unified execution.

    This adapter enables existing WorkflowBuilder workflows to run
    on the StateGraph runtime, providing access to features like:
    - Proper cycle detection and max iteration limits
    - Typed state management
    - Checkpointing and recovery
    - Streaming execution

    Example:
        adapter = WorkflowToGraphAdapter()
        state_graph = adapter.adapt(feature_workflow)

        # Execute with StateGraph runtime
        app = state_graph.compile()
        result = await app.invoke({
            "context": {"task": "implement feature"},
            "messages": [],
        })
    """

    def __init__(
        self,
        *,
        max_iterations: int = 50,
        enable_cycles: bool = False,
    ):
        """Initialize the adapter.

        Args:
            max_iterations: Maximum iterations for cycle detection
            enable_cycles: Whether to allow cycles in adapted workflows
        """
        self.max_iterations = max_iterations
        self.enable_cycles = enable_cycles

    def adapt(self, workflow: WorkflowDefinition) -> "StateGraph":
        """Adapt a WorkflowDefinition to StateGraph.

        Creates a StateGraph that mimics the behavior of the
        original WorkflowBuilder workflow.

        Args:
            workflow: WorkflowDefinition to adapt

        Returns:
            StateGraph ready for compilation and execution
        """
        # Import here to avoid circular imports
        from victor.framework.graph import StateGraph, END

        # Create StateGraph with workflow state (Pydantic model)
        graph: StateGraph[AdapterWorkflowStateModel] = StateGraph(AdapterWorkflowStateModel)

        # Convert each node
        for node in workflow.nodes:
            adapted = self._adapt_node(node, workflow)
            graph.add_node(adapted.name, adapted.handler)

        # Add edges based on node connections
        for node in workflow.nodes:
            if node.next_nodes:
                for next_node in node.next_nodes:
                    if next_node:
                        graph.add_edge(node.name, next_node)
            else:
                # Terminal node - connect to END
                graph.add_edge(node.name, END)

        # Set entry point
        if workflow.nodes:
            graph.set_entry_point(workflow.nodes[0].name)

        logger.debug(
            f"Adapted workflow '{workflow.name}' to StateGraph " f"({len(workflow.nodes)} nodes)"
        )

        return graph

    def _adapt_node(
        self,
        node: WorkflowNode,
        workflow: WorkflowDefinition,
    ) -> AdaptedNode:
        """Adapt a single WorkflowNode to an AdaptedNode.

        Args:
            node: WorkflowNode to adapt
            workflow: Parent workflow for context

        Returns:
            AdaptedNode ready for StateGraph
        """

        def create_handler(
            n: WorkflowNode,
        ) -> Callable[[AdapterWorkflowStateModel], AdapterWorkflowStateModel]:
            """Create a state-updating handler for the node."""

            def handler(state: AdapterWorkflowStateModel) -> AdapterWorkflowStateModel:
                # Update state with node execution (Pydantic model)
                new_state = state.model_copy(deep=True)
                new_state.current_node = n.name

                # Track visited nodes
                visited = list(state.visited_nodes)
                visited.append(n.name)
                new_state.visited_nodes = visited

                # Add placeholder result
                # In production, this would execute the actual agent
                results = dict(state.results)
                results[n.name] = {
                    "status": "pending",
                    "node_type": (
                        n.node_type.value if hasattr(n.node_type, "value") else str(n.node_type)
                    ),
                }
                new_state.results = results

                return new_state

            return handler

        return AdaptedNode(
            name=node.name,
            node_type=node.node_type,
            handler=create_handler(node),
            next_nodes=node.next_nodes,
            tool_budget=node.tool_budget,
            allowed_tools=node.allowed_tools,
        )

    def adapt_with_execution(
        self,
        workflow: WorkflowDefinition,
        executor: "IWorkflowNodeRuntimeExecutor",
    ) -> "StateGraph":
        """Adapt with real execution handlers.

        This version connects the adapted graph to the actual
        WorkflowExecutor for real agent execution.

        Args:
            workflow: WorkflowDefinition to adapt
            executor: Node-capable workflow runtime for running agents

        Returns:
            StateGraph with real execution handlers
        """
        # Import here to avoid circular imports
        from victor.framework.graph import StateGraph, END

        graph: StateGraph[AdapterWorkflowState] = StateGraph(AdapterWorkflowState)

        # Convert each node with real execution
        for node in workflow.nodes:
            handler = self._create_execution_handler(node, executor)
            graph.add_node(node.name, handler)

        # Add edges
        for node in workflow.nodes:
            if node.next_nodes:
                for next_node in node.next_nodes:
                    if next_node:
                        graph.add_edge(node.name, next_node)
            else:
                graph.add_edge(node.name, END)

        if workflow.nodes:
            graph.set_entry_point(workflow.nodes[0].name)

        return graph

    def _create_execution_handler(
        self,
        node: WorkflowNode,
        executor: "IWorkflowNodeRuntimeExecutor",
    ) -> Callable[[AdapterWorkflowState], AdapterWorkflowState]:
        """Create an execution handler that uses the workflow executor.

        Args:
            node: WorkflowNode to execute
            executor: Node-capable workflow runtime

        Returns:
            Handler function for StateGraph
        """

        async def async_handler(state: AdapterWorkflowState) -> AdapterWorkflowState:
            """Execute the node using the workflow executor."""
            new_state = dict(state)
            new_state["current_node"] = node.name

            # Get context from state
            context = state.get("context", {})

            try:
                # Execute the node using the executor
                # This is a simplified version - production would handle
                # all the agent orchestration details
                result = await executor.execute_node(node, context)

                # Update state with result
                results = dict(state.get("results", {}))
                results[node.name] = result
                new_state["results"] = results

            except Exception as e:
                new_state["error"] = str(e)
                logger.error(f"Node {node.name} failed: {e}")

            # Track visited
            visited = list(state.get("visited_nodes", []))
            visited.append(node.name)
            new_state["visited_nodes"] = visited

            return new_state  # type: ignore

        # Keep a sync compatibility shell for graph callbacks while routing
        # event-loop ownership through the shared async bridge helpers.
        def sync_handler(state: AdapterWorkflowState) -> AdapterWorkflowState:
            try:
                asyncio.get_running_loop()
                return run_sync_in_thread(async_handler(state))
            except RuntimeError:
                return run_sync(async_handler(state))

        return sync_handler


class GraphToWorkflowAdapter:
    """Adapts StateGraph to WorkflowDefinition for compatibility.

    This adapter enables new StateGraph workflows to work with
    existing WorkflowBuilder-based infrastructure.

    Use case: A new workflow written with StateGraph needs to be
    registered in the existing workflow registry.
    """

    def adapt(self, graph: "StateGraph", name: str) -> WorkflowDefinition:
        """Adapt a StateGraph to WorkflowDefinition.

        Args:
            graph: StateGraph to adapt
            name: Name for the workflow

        Returns:
            WorkflowDefinition compatible with existing infrastructure
        """
        from victor.workflows.builder import WorkflowBuilder

        builder = WorkflowBuilder(name)

        # Get nodes from graph
        # Note: This is a simplified adaptation - StateGraph
        # internal structure may vary
        nodes = getattr(graph, "_nodes", {})
        edges = getattr(graph, "_edges", {})
        entry_point = getattr(graph, "_entry_point", None)

        # Convert nodes, starting with entry point if defined
        ordered_nodes = list(nodes.keys())
        if entry_point and entry_point in ordered_nodes:
            ordered_nodes.remove(entry_point)
            ordered_nodes.insert(0, entry_point)

        for node_name in ordered_nodes:
            builder.add_agent(
                name=node_name,
                role="adapted_node",
                goal=f"Execute {node_name}",
            )

        # Set up edges
        for from_node, to_nodes in edges.items():
            for to_node in to_nodes:
                # Note: WorkflowBuilder uses next_nodes on nodes
                pass  # Edge setup handled by builder

        return builder.build()


__all__ = [
    "AdapterWorkflowState",
    "WorkflowState",
    "AdaptedNode",
    "WorkflowToGraphAdapter",
    "GraphToWorkflowAdapter",
]
