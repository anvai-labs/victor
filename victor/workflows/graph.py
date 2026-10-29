"""Workflow graph implementation for defining workflow structures.

This module provides concrete implementations of workflow graph components:
- WorkflowNode: A node in the workflow graph
- WorkflowEdge: A simple edge connecting two nodes
- ConditionalEdge: An edge with conditional traversal logic
- BasicWorkflowGraph: The basic graph container for nodes and edges

For typed workflow graphs with state, see victor.workflows.graph_dsl.WorkflowGraph.
For LangGraph-compatible StateGraph, see victor.framework.graph.StateGraph.

These classes implement the protocols defined in victor.workflows.protocols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from victor.core.graph_algorithms import detect_cycle, find_reachable
from victor.workflows.protocols import (
    RetryPolicy,
    NodeResult,
    IWorkflowNode,
    IWorkflowEdge,
)

# Type alias for router functions with Literal return types for better type safety
# Example: RouterFunction[Literal["success", "error"]] for a router that returns those values
RouterFunction = Callable[[Dict[str, Any]], str]


class DuplicateNodeError(Exception):
    """Raised when attempting to add a node with a duplicate ID."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        super().__init__(f"Node with ID '{node_id}' already exists in the graph")


class InvalidEdgeError(Exception):
    """Raised when an edge references non-existent nodes."""

    def __init__(self, message: str):
        super().__init__(message)


class GraphValidationError(Exception):
    """Raised when graph validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Graph validation failed: {'; '.join(errors)}")


# Type alias for handler functions
NodeHandler = Callable[[Dict[str, Any], Optional[Dict[str, Any]]], NodeResult]


@dataclass
class WorkflowNode:
    """A node in the workflow graph.

    Represents a single step in a workflow that can be executed.

    Attributes:
        id: Unique identifier for this node.
        name: Human-readable name for display.
        handler: Async function to execute for this node.
        retry_policy: Policy for retrying failed executions.
        metadata: Optional metadata about this node.

    Example:
        async def process_data(state, context=None):
            processed = transform(state["data"])
            return NodeResult(
                status=ProtocolNodeStatus.COMPLETED,
                output={"data": processed}
            )

        node = WorkflowNode(
            id="process",
            name="Process Data",
            handler=process_data,
            retry_policy=RetryPolicy(max_retries=3)
        )
    """

    id: str
    name: str
    handler: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], NodeResult]
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    metadata: Dict[str, Any] = field(default_factory=dict)

    async def execute(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> NodeResult:
        """Execute this node with the given state.

        Args:
            state: Current workflow state.
            context: Optional execution context.

        Returns:
            NodeResult with status and output.
        """
        return await self.handler(state, context)


@dataclass
class WorkflowEdge:
    """A simple edge connecting two nodes.

    This edge is always traversable - it has no conditional logic.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.

    Example:
        edge = WorkflowEdge(source_id="start", target_id="process")
    """

    source_id: str
    target_id: str

    def should_traverse(self, state: Dict[str, Any]) -> bool:
        """Check if this edge should be traversed.

        Simple edges are always traversable.

        Args:
            state: Current workflow state (ignored).

        Returns:
            Always True.
        """
        return True


@dataclass
class ConditionalEdge:
    """An edge with conditional traversal logic.

    The condition function is evaluated against the workflow state
    to determine if the edge should be traversed.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        condition: Function that takes state and returns bool.

    Example:
        def is_successful(state):
            return state.get("status") == "success"

        edge = ConditionalEdge(
            source_id="check",
            target_id="success_handler",
            condition=is_successful
        )
    """

    source_id: str
    target_id: str
    condition: Callable[[Dict[str, Any]], bool]

    def should_traverse(self, state: Dict[str, Any]) -> bool:
        """Check if this edge should be traversed.

        Evaluates the condition function against the state.

        Args:
            state: Current workflow state.

        Returns:
            True if condition returns True, False otherwise.
        """
        return self.condition(state)


class _RouterEdge:
    """Internal edge type for router-based conditional branching.

    Used by add_conditional_edge to create multiple conditional edges
    based on a router function's output.

    Type Safety:
        Router functions should use Literal return types for compile-time checking:
        ```python
        from typing import Literal

        def route_by_status(state: Dict[str, Any]) -> Literal["success", "error", "retry"]:
            if state.get("error"):
                return "error"
            if state.get("needs_retry"):
                return "retry"
            return "success"
        ```
    """

    def __init__(
        self,
        source_id: str,
        target_id: str,
        router: RouterFunction,
        route_key: str,
    ):
        self.source_id = source_id
        self.target_id = target_id
        self._router = router
        self._route_key = route_key

    @property
    def source_id(self) -> str:
        return self._source_id

    @source_id.setter
    def source_id(self, value: str) -> None:
        self._source_id = value

    @property
    def target_id(self) -> str:
        return self._target_id

    @target_id.setter
    def target_id(self, value: str) -> None:
        self._target_id = value

    def should_traverse(self, state: Dict[str, Any]) -> bool:
        """Check if this edge should be traversed.

        Compares the router function's output to this edge's route key.

        Args:
            state: Current workflow state.

        Returns:
            True if router returns this edge's route key.
        """
        return self._router(state) == self._route_key


class BasicWorkflowGraph:
    """A basic directed graph container for workflow nodes and edges.

    Provides methods for building, querying, and validating workflow graphs.
    Supports fluent interface for chaining method calls.

    Note: For typed workflow graphs with state, see victor.workflows.graph_dsl.WorkflowGraph.
    For LangGraph-compatible StateGraph, see victor.framework.graph.StateGraph.

    Example:
        graph = (
            BasicWorkflowGraph()
            .add_node(start_node)
            .add_node(process_node)
            .add_node(end_node)
            .add_edge(WorkflowEdge("start", "process"))
            .add_edge(WorkflowEdge("process", "end"))
        )
        graph.set_entry_node("start")
        graph.set_exit_nodes(["end"])

        errors = graph.validate()
        if not errors:
            # Graph is valid
            pass
    """

    def __init__(self) -> None:
        """Initialize an empty workflow graph."""
        self._nodes: Dict[str, IWorkflowNode] = {}
        self._edges: List[IWorkflowEdge] = []
        self._entry_node_id: Optional[str] = None
        self._exit_node_ids: Set[str] = set()

    def add_node(self, node: IWorkflowNode) -> "BasicWorkflowGraph":
        """Add a node to the graph.

        Args:
            node: The node to add.

        Returns:
            Self for fluent interface.

        Raises:
            DuplicateNodeError: If a node with the same ID already exists.
        """
        if node.id in self._nodes:
            raise DuplicateNodeError(node.id)

        self._nodes[node.id] = node
        return self

    def add_edge(self, edge: IWorkflowEdge) -> "BasicWorkflowGraph":
        """Add an edge to the graph.

        Args:
            edge: The edge to add.

        Returns:
            Self for fluent interface.

        Raises:
            InvalidEdgeError: If source or target node doesn't exist.
        """
        if edge.source_id not in self._nodes:
            raise InvalidEdgeError(f"Source node '{edge.source_id}' does not exist in the graph")

        if edge.target_id not in self._nodes:
            raise InvalidEdgeError(f"Target node '{edge.target_id}' does not exist in the graph")

        self._edges.append(edge)
        return self

    def add_conditional_edge(
        self,
        source_id: str,
        router: RouterFunction,
        targets: Dict[str, str],
    ) -> "BasicWorkflowGraph":
        """Add conditional branching edges based on a router function.

        Creates multiple edges from the source node, each activated
        when the router returns the corresponding key.

        Args:
            source_id: ID of the source node.
            router: Function that takes state and returns a route key.
            targets: Mapping from route keys to target node IDs.

        Returns:
            Self for fluent interface.

        Raises:
            InvalidEdgeError: If source or any target node doesn't exist.

        Example:
            from typing import Literal

            def route_by_status(state: Dict[str, Any]) -> Literal["success", "error", "retry"]:
                if state.get("error"):
                    return "error"
                if state.get("needs_retry"):
                    return "retry"
                return "success"

            graph.add_conditional_edge(
                source_id="check",
                router=route_by_status,
                targets={
                    "success": "handle_success",
                    "error": "handle_error",
                    "retry": "retry_step"
                }
            )
        """
        if source_id not in self._nodes:
            raise InvalidEdgeError(f"Source node '{source_id}' does not exist in the graph")

        for route_key, target_id in targets.items():
            if target_id not in self._nodes:
                raise InvalidEdgeError(
                    f"Target node '{target_id}' for route '{route_key}' does not exist"
                )

            edge = _RouterEdge(
                source_id=source_id,
                target_id=target_id,
                router=router,
                route_key=route_key,
            )
            self._edges.append(edge)

        return self

    def set_entry_node(self, node_id: str) -> "BasicWorkflowGraph":
        """Set the entry point of the workflow.

        Args:
            node_id: ID of the entry node.

        Returns:
            Self for fluent interface.

        Raises:
            InvalidEdgeError: If node doesn't exist.
        """
        if node_id not in self._nodes:
            raise InvalidEdgeError(f"Entry node '{node_id}' does not exist in the graph")

        self._entry_node_id = node_id
        return self

    def set_exit_nodes(self, node_ids: List[str]) -> "BasicWorkflowGraph":
        """Set the exit points of the workflow.

        Args:
            node_ids: List of exit node IDs.

        Returns:
            Self for fluent interface.

        Raises:
            InvalidEdgeError: If any node doesn't exist.
        """
        for node_id in node_ids:
            if node_id not in self._nodes:
                raise InvalidEdgeError(f"Exit node '{node_id}' does not exist in the graph")

        self._exit_node_ids = set(node_ids)
        return self

    def get_node(self, node_id: str) -> Optional[IWorkflowNode]:
        """Get a node by its ID.

        Args:
            node_id: The node's unique identifier.

        Returns:
            The node if found, None otherwise.
        """
        return self._nodes.get(node_id)

    def get_entry_node(self) -> Optional[IWorkflowNode]:
        """Get the entry node of the graph.

        Returns:
            The entry node if set, None otherwise.
        """
        if self._entry_node_id is None:
            return None
        return self._nodes.get(self._entry_node_id)

    def get_exit_nodes(self) -> List[IWorkflowNode]:
        """Get all exit nodes of the graph.

        Returns:
            List of exit nodes.
        """
        return [self._nodes[node_id] for node_id in self._exit_node_ids if node_id in self._nodes]

    def get_next_nodes(self, node_id: str, state: Dict[str, Any]) -> List[IWorkflowNode]:
        """Get the next nodes to execute after the given node.

        Evaluates all edges from the node and returns nodes whose
        edges should be traversed based on the current state.

        Args:
            node_id: The current node's ID.
            state: Current workflow state (for conditional edges).

        Returns:
            List of nodes to execute next.
        """
        next_nodes = []

        for edge in self._edges:
            if edge.source_id == node_id and edge.should_traverse(state):
                target_node = self._nodes.get(edge.target_id)
                if target_node is not None:
                    next_nodes.append(target_node)

        return next_nodes

    def validate(self) -> List[str]:
        """Validate the graph structure.

        Checks for:
        - Entry node is set
        - No orphan nodes (unreachable from entry)
        - No cycles in the graph

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        # Check for entry node
        if self._entry_node_id is None:
            errors.append("No entry node set for the workflow")
            return errors  # Can't check other things without entry node

        if self._entry_node_id not in self._nodes:
            errors.append(f"Entry node '{self._entry_node_id}' not found in graph")
            return errors

        # Check for orphan nodes
        reachable = self._find_reachable_nodes()
        orphans = set(self._nodes.keys()) - reachable

        for orphan_id in orphans:
            errors.append(f"Orphan node '{orphan_id}' is not reachable from entry")

        # Check for cycles
        cycles = self._find_cycles()
        for cycle in cycles:
            cycle_str = " -> ".join(cycle)
            errors.append(f"Cycle detected: {cycle_str}")

        return errors

    def _find_reachable_nodes(self) -> Set[str]:
        """Find all nodes reachable from the entry node.

        Uses BFS to traverse the graph and collect all reachable node IDs.

        Returns:
            Set of reachable node IDs.
        """
        if self._entry_node_id is None:
            return set()

        adjacency: Dict[str, List[str]] = {node_id: [] for node_id in self._nodes}
        for edge in self._edges:
            adjacency.setdefault(edge.source_id, []).append(edge.target_id)

        return find_reachable(self._entry_node_id, adjacency)

    def _find_cycles(self) -> List[List[str]]:
        """Find all cycles in the graph.

        Uses DFS with coloring to detect cycles.
        WHITE (0) = not visited
        GRAY (1) = currently in recursion stack
        BLACK (2) = fully processed

        Returns:
            List of cycles, each cycle is a list of node IDs.
        """
        adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in self._nodes}
        for edge in self._edges:
            adjacency.setdefault(edge.source_id, set()).add(edge.target_id)

        cycle = detect_cycle(set(self._nodes), adjacency)
        return [cycle] if cycle else []


LegacyWorkflowGraph = BasicWorkflowGraph
"""Explicit legacy alias for callers migrating off ``WorkflowGraph``."""


# Deprecation alias for backward compatibility
import warnings as _warnings

_WORKFLOW_GRAPH_ALIAS_DEPRECATION_MESSAGE = (
    "WorkflowGraph from victor.workflows.graph is deprecated. "
    "Use BasicWorkflowGraph for the simple graph container or "
    "victor.workflows.graph_dsl.WorkflowGraph for the typed workflow DSL. "
    "This warning-backed alias remains supported through v0.8.0 (2026-12-31). "
    "See docs/architecture/migration.md for migration guidance."
)


def __getattr__(name: str):
    """Provide deprecation warning for WorkflowGraph alias."""
    if name == "WorkflowGraph":
        _warnings.warn(
            _WORKFLOW_GRAPH_ALIAS_DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=2,
        )
        return BasicWorkflowGraph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
