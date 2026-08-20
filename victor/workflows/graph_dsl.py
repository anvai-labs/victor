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

"""WorkflowGraph DSL for declarative workflow definitions.

Provides a LangGraph-inspired declarative API for defining workflow graphs
with type-safe state, operator chaining, and branch/merge patterns.

This module offers an alternative to YAML-based workflow definitions,
allowing workflows to be defined directly in Python with:
- Type-safe state via dataclass-based State
- Operator chaining with `>>` for linear flows
- Conditional branching with `add_conditional_edges`
- Parallel execution with `add_parallel_nodes`
- Entry/exit point declaration
- Compilation to WorkflowDefinition for execution

Note: This module provides WorkflowGraph which compiles to WorkflowDefinition.
For the LangGraph-compatible StateGraph that compiles to CompiledGraph,
see victor.framework.graph.StateGraph.

Example - Basic linear workflow:
    from victor.workflows.graph_dsl import WorkflowGraph, State
    from dataclasses import dataclass
    from typing import List, Optional

    @dataclass
    class CodeReviewState(State):
        files: List[str]
        analysis: Optional[str] = None
        issues: Optional[List[str]] = None
        report: Optional[str] = None

    async def analyze_code(state: CodeReviewState) -> CodeReviewState:
        # Analyze files...
        state.analysis = "Found patterns X, Y, Z"
        return state

    async def find_issues(state: CodeReviewState) -> CodeReviewState:
        state.issues = ["Issue 1", "Issue 2"]
        return state

    async def generate_report(state: CodeReviewState) -> CodeReviewState:
        state.report = f"Analysis: {state.analysis}\\nIssues: {state.issues}"
        return state

    # Build workflow using operator chaining
    graph = WorkflowGraph(CodeReviewState)
    graph.add_node("analyze", analyze_code)
    graph.add_node("find_issues", find_issues)
    graph.add_node("report", generate_report)

    # Chain: analyze >> find_issues >> report
    graph.add_edge("analyze", "find_issues")
    graph.add_edge("find_issues", "report")
    graph.set_entry_point("analyze")
    graph.set_finish_point("report")

    workflow = graph.compile()

Example - Conditional branching:
    def route_decision(state: CodeReviewState) -> str:
        if state.issues and len(state.issues) > 0:
            return "has_issues"
        return "no_issues"

    graph.add_conditional_edges(
        "analyze",
        route_decision,
        {
            "has_issues": "fix",
            "no_issues": "report"
        }
    )

Example - Operator chaining (>>):
    # Fluent chaining API
    (graph.node("analyze") >> graph.node("find_issues") >> graph.node("report"))

    # Or using the chain method
    graph.chain("analyze", "find_issues", "report")
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

from victor.core.async_utils import run_sync, run_sync_in_thread
from victor.workflows.definition import (
    AgentNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    WorkflowDefinition,
    WorkflowNode,
)

logger = logging.getLogger(__name__)


# Type variable for state
S = TypeVar("S", bound="State")


class State(ABC):
    """Base class for workflow state.

    Workflow state should be defined as a dataclass inheriting from State.
    This enables type-safe state management and automatic serialization.

    Example:
        @dataclass
        class MyState(State):
            files: List[str]
            analysis: Optional[str] = None
            results: Dict[str, Any] = field(default_factory=dict)
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary.

        Returns:
            Dictionary representation of state
        """
        if is_dataclass(self):
            return {f.name: getattr(self, f.name) for f in fields(self)}
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls: Type[S], data: Dict[str, Any]) -> S:
        """Create state from dictionary.

        Args:
            data: Dictionary with state values

        Returns:
            New state instance
        """
        if is_dataclass(cls):
            field_names = {f.name for f in fields(cls)}
            filtered = {k: v for k, v in data.items() if k in field_names}
            return cls(**filtered)
        instance = object.__new__(cls)
        instance.__dict__.update(data)
        return instance

    def copy(self: S) -> S:
        """Create a shallow copy of the state.

        Returns:
            New state instance with copied values
        """
        return self.from_dict(self.to_dict())

    def merge(self: S, updates: Dict[str, Any]) -> S:
        """Create a new state with merged updates.

        Args:
            updates: Dictionary of updates to apply

        Returns:
            New state instance with merged values
        """
        data = self.to_dict()
        data.update(updates)
        return self.from_dict(data)


# Node function types
NodeFunc = Callable[[S], Union[S, Awaitable[S]]]
RouterFunc = Callable[[S], str]


class GraphNodeType(str, Enum):
    """Types of graph DSL nodes.

    Renamed from NodeType to be semantically distinct:
    - GraphNodeType (here): Graph DSL nodes (FUNCTION, CONDITIONAL, SUBGRAPH)
    - WorkflowNodeType (victor.workflows.definition): Workflow definition nodes (COMPUTE, HITL, START, END)
    - YAMLNodeType (victor.workflows.yaml_loader): YAML loader validation nodes
    """

    FUNCTION = "function"  # Transform function
    AGENT = "agent"  # Agent execution
    CONDITIONAL = "conditional"  # Router/branch
    PARALLEL = "parallel"  # Parallel execution
    SUBGRAPH = "subgraph"  # Nested graph


@dataclass
class GraphNode(Generic[S]):
    """A node in the state graph.

    Attributes:
        name: Unique node identifier
        func: The node function or agent config
        node_type: Type of node
        metadata: Additional node configuration
    """

    name: str
    func: Optional[NodeFunc[S]] = None
    node_type: GraphNodeType = GraphNodeType.FUNCTION
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For agent nodes
    agent_role: Optional[str] = None
    agent_goal: Optional[str] = None
    tool_budget: int = 15
    allowed_tools: Optional[List[str]] = None

    # For chaining (>> operator)
    _graph: Optional["WorkflowGraph[S]"] = field(default=None, repr=False)

    def __rshift__(self, other: "GraphNode[S]") -> "GraphNode[S]":
        """Chain nodes using >> operator.

        Example:
            graph.node("a") >> graph.node("b") >> graph.node("c")

        Returns:
            The right-hand node (for chaining)
        """
        if self._graph is None or other._graph is None:
            raise ValueError("Nodes must be added to a graph before chaining")

        if self._graph is not other._graph:
            raise ValueError("Cannot chain nodes from different graphs")

        self._graph.add_edge(self.name, other.name)
        return other

    def __or__(self, other: "GraphNode[S]") -> "GraphNode[S]":
        """Alternative chaining using | operator.

        Same behavior as >> but may be preferred in some contexts.
        """
        return self.__rshift__(other)


@runtime_checkable
class Compilable(Protocol):
    """Protocol for objects that can compile to WorkflowDefinition."""

    def compile(self) -> WorkflowDefinition:
        """Compile to a workflow definition."""
        ...


class WorkflowGraph(Generic[S]):
    """A typed workflow graph for declarative workflow definitions.

    WorkflowGraph provides a LangGraph-inspired API for defining workflows
    as directed graphs with typed state. Nodes are functions that transform
    state, and edges define the flow between nodes.

    This class compiles to WorkflowDefinition for execution via WorkflowExecutor.
    For the LangGraph-compatible StateGraph that compiles to CompiledGraph,
    see victor.framework.graph.StateGraph.

    Type Parameters:
        S: The state type (must inherit from State)

    Attributes:
        state_type: The state class for this graph
        name: Optional graph name

    Example:
        @dataclass
        class MyState(State):
            value: int = 0

        def increment(state: MyState) -> MyState:
            state.value += 1
            return state

        graph = WorkflowGraph(MyState, name="counter")
        graph.add_node("inc", increment)
        graph.set_entry_point("inc")
        workflow = graph.compile()
    """

    # Special node names
    START = "__start__"
    END = "__end__"

    def __init__(
        self,
        state_type: Type[S],
        name: Optional[str] = None,
        description: str = "",
    ):
        """Initialize a new state graph.

        Args:
            state_type: The state dataclass type
            name: Optional workflow name (defaults to state type name)
            description: Optional workflow description
        """
        self.state_type = state_type
        self.name = name or state_type.__name__.replace("State", "").lower() + "_workflow"
        self.description = description

        self._nodes: Dict[str, GraphNode[S]] = {}
        self._edges: Dict[str, List[str]] = {}  # from_node -> [to_nodes]
        self._conditional_edges: Dict[str, tuple[RouterFunc[S], Dict[str, str]]] = {}
        self._entry_point: Optional[str] = None
        self._finish_points: Set[str] = set()
        self._metadata: Dict[str, Any] = {}

    def add_node(
        self,
        name: str,
        func: Optional[NodeFunc[S]] = None,
        *,
        node_type: GraphNodeType = GraphNodeType.FUNCTION,
        **kwargs: Any,
    ) -> "WorkflowGraph[S]":
        """Add a node to the graph.

        Args:
            name: Unique node name
            func: Node function (transforms state)
            node_type: Type of node
            **kwargs: Additional node configuration

        Returns:
            Self for chaining

        Raises:
            ValueError: If node name already exists

        Example:
            graph.add_node("process", process_fn)
            graph.add_node("analyze", analyze_fn, timeout=60)
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")

        node = GraphNode(
            name=name,
            func=func,
            node_type=node_type,
            metadata=kwargs,
            _graph=self,
        )
        self._nodes[name] = node
        self._edges[name] = []

        logger.debug(f"Added node '{name}' to graph '{self.name}'")
        return self

    def add_agent_node(
        self,
        name: str,
        role: str,
        goal: str,
        *,
        tool_budget: int = 15,
        allowed_tools: Optional[List[str]] = None,
        output_key: Optional[str] = None,
    ) -> "WorkflowGraph[S]":
        """Add an agent node to the graph.

        Agent nodes spawn sub-agents to perform tasks using the
        existing SubAgent infrastructure.

        Args:
            name: Unique node name
            role: Agent role (researcher, planner, executor, etc.)
            goal: Task description for the agent
            tool_budget: Maximum tool calls
            allowed_tools: Specific tools to allow
            output_key: Key for storing agent output in state

        Returns:
            Self for chaining

        Example:
            graph.add_agent_node(
                "analyze",
                role="researcher",
                goal="Analyze the codebase for patterns",
                tool_budget=20
            )
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")

        node = GraphNode(
            name=name,
            func=None,
            node_type=GraphNodeType.AGENT,
            agent_role=role,
            agent_goal=goal,
            tool_budget=tool_budget,
            allowed_tools=allowed_tools,
            metadata={"output_key": output_key or name},
            _graph=self,
        )
        self._nodes[name] = node
        self._edges[name] = []

        logger.debug(f"Added agent node '{name}' (role={role}) to graph '{self.name}'")
        return self

    def add_edge(self, from_node: str, to_node: str) -> "WorkflowGraph[S]":
        """Add an edge between nodes.

        Args:
            from_node: Source node name
            to_node: Target node name

        Returns:
            Self for chaining

        Raises:
            ValueError: If nodes don't exist

        Example:
            graph.add_edge("analyze", "report")
        """
        if from_node not in self._nodes and from_node != self.START:
            raise ValueError(f"Source node '{from_node}' not found")
        if to_node not in self._nodes and to_node != self.END:
            raise ValueError(f"Target node '{to_node}' not found")

        if from_node not in self._edges:
            self._edges[from_node] = []

        if to_node not in self._edges[from_node]:
            self._edges[from_node].append(to_node)

        logger.debug(f"Added edge: {from_node} -> {to_node}")
        return self

    def add_conditional_edges(
        self,
        from_node: str,
        router: RouterFunc[S],
        routes: Dict[str, str],
        *,
        default: Optional[str] = None,
    ) -> "WorkflowGraph[S]":
        """Add conditional edges from a node.

        The router function examines state and returns a route key.
        The routes dictionary maps route keys to target node names.

        Args:
            from_node: Source node name
            router: Function that returns route key
            routes: Mapping from route keys to node names
            default: Default route if router returns unknown key

        Returns:
            Self for chaining

        Example:
            def decide(state: MyState) -> str:
                return "fix" if state.has_issues else "done"

            graph.add_conditional_edges(
                "analyze",
                decide,
                {"fix": "fix_node", "done": "report_node"}
            )
        """
        if from_node not in self._nodes:
            raise ValueError(f"Source node '{from_node}' not found")

        for target in routes.values():
            if target not in self._nodes and target != self.END:
                raise ValueError(f"Target node '{target}' not found")

        if default:
            routes = routes.copy()
            routes["__default__"] = default

        self._conditional_edges[from_node] = (router, routes)
        logger.debug(f"Added conditional edges from '{from_node}': {list(routes.keys())}")
        return self

    def set_entry_point(self, node: str) -> "WorkflowGraph[S]":
        """Set the entry point (start node) for the graph.

        Args:
            node: Name of the entry node

        Returns:
            Self for chaining

        Raises:
            ValueError: If node doesn't exist
        """
        if node not in self._nodes:
            raise ValueError(f"Entry node '{node}' not found")

        self._entry_point = node
        self.add_edge(self.START, node)
        logger.debug(f"Set entry point: {node}")
        return self

    def set_finish_point(self, node: str) -> "WorkflowGraph[S]":
        """Set a finish point (end node) for the graph.

        Multiple finish points can be set.

        Args:
            node: Name of the finish node

        Returns:
            Self for chaining
        """
        if node not in self._nodes:
            raise ValueError(f"Finish node '{node}' not found")

        self._finish_points.add(node)
        self.add_edge(node, self.END)
        logger.debug(f"Set finish point: {node}")
        return self

    def node(self, name: str) -> GraphNode[S]:
        """Get a node reference for chaining.

        Used with the >> operator for fluent edge definitions.

        Args:
            name: Node name

        Returns:
            GraphNode for chaining

        Example:
            graph.node("a") >> graph.node("b") >> graph.node("c")
        """
        if name not in self._nodes:
            raise ValueError(f"Node '{name}' not found")
        return self._nodes[name]

    def chain(self, *nodes: str) -> "WorkflowGraph[S]":
        """Chain multiple nodes in sequence.

        Convenience method for creating linear flows.

        Args:
            *nodes: Node names in order

        Returns:
            Self for chaining

        Example:
            graph.chain("analyze", "process", "report")
        """
        for i in range(len(nodes) - 1):
            self.add_edge(nodes[i], nodes[i + 1])
        return self

    def branch(
        self,
        from_node: str,
        *branches: str,
    ) -> "WorkflowGraph[S]":
        """Create branches from a node (parallel edges).

        All branches will be connected from the source node.
        Use with ParallelNode or conditional edges.

        Args:
            from_node: Source node
            *branches: Target nodes

        Returns:
            Self for chaining

        Example:
            graph.branch("split", "branch_a", "branch_b", "branch_c")
        """
        for target in branches:
            self.add_edge(from_node, target)
        return self

    def merge(
        self,
        to_node: str,
        *sources: str,
    ) -> "WorkflowGraph[S]":
        """Merge multiple nodes into one.

        Creates edges from all sources to the target.

        Args:
            to_node: Target node
            *sources: Source nodes

        Returns:
            Self for chaining

        Example:
            graph.merge("join", "branch_a", "branch_b")
        """
        for source in sources:
            self.add_edge(source, to_node)
        return self

    def add_parallel_nodes(
        self,
        name: str,
        parallel: List[str],
        *,
        join_strategy: str = "all",
        next_nodes: Optional[List[str]] = None,
    ) -> "WorkflowGraph[S]":
        """Add a parallel execution node.

        Executes multiple nodes concurrently and joins results.

        Args:
            name: Name for the parallel node
            parallel: List of node names to execute in parallel
            join_strategy: How to combine results (all, any, merge)
            next_nodes: Nodes to execute after parallel completion

        Returns:
            Self for chaining
        """
        for node_name in parallel:
            if node_name not in self._nodes:
                raise ValueError(f"Parallel node '{node_name}' not found")

        node = GraphNode(
            name=name,
            func=None,
            node_type=GraphNodeType.PARALLEL,
            metadata={
                "parallel_nodes": parallel,
                "join_strategy": join_strategy,
            },
            _graph=self,
        )
        self._nodes[name] = node
        self._edges[name] = next_nodes or []

        return self

    def set_metadata(self, key: str, value: Any) -> "WorkflowGraph[S]":
        """Set graph metadata.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for chaining
        """
        self._metadata[key] = value
        return self

    def validate(self) -> List[str]:
        """Validate the graph structure.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self._nodes:
            errors.append("Graph has no nodes")

        if not self._entry_point:
            errors.append("No entry point set")
        elif self._entry_point not in self._nodes:
            errors.append(f"Entry point '{self._entry_point}' not found")

        if not self._finish_points:
            # Find terminal nodes (nodes with no outgoing edges)
            terminal = [
                n
                for n, edges in self._edges.items()
                if not edges and n not in self._conditional_edges
            ]
            if terminal:
                errors.append(f"No finish points set. Terminal nodes: {terminal}")

        # Check for unreachable nodes
        reachable = self._find_reachable()
        for node_name in self._nodes:
            if node_name not in reachable:
                errors.append(f"Node '{node_name}' is unreachable")

        # Check for broken edges
        for from_node, targets in self._edges.items():
            for target in targets:
                if target != self.END and target not in self._nodes:
                    errors.append(f"Edge from '{from_node}' to non-existent node '{target}'")

        return errors

    def _find_reachable(self) -> Set[str]:
        """Find all nodes reachable from entry point."""
        if not self._entry_point:
            return set()

        reachable = set()
        to_visit = [self._entry_point]

        while to_visit:
            node_name = to_visit.pop()
            if node_name in reachable or node_name == self.END:
                continue
            reachable.add(node_name)

            # Regular edges
            to_visit.extend(self._edges.get(node_name, []))

            # Conditional edges
            if node_name in self._conditional_edges:
                _, routes = self._conditional_edges[node_name]
                to_visit.extend(routes.values())

        return reachable

    def compile(self, name: Optional[str] = None) -> WorkflowDefinition:
        """Compile the graph to a WorkflowDefinition.

        Converts the StateGraph to the existing WorkflowDefinition
        format for execution by WorkflowExecutor.

        Args:
            name: Optional override for workflow name

        Returns:
            WorkflowDefinition ready for execution

        Raises:
            ValueError: If graph validation fails
        """
        errors = self.validate()
        if errors:
            raise ValueError(f"Graph validation failed: {'; '.join(errors)}")

        workflow_name = name or self.name
        nodes: Dict[str, WorkflowNode] = {}

        # Convert graph nodes to workflow nodes
        for node_name, graph_node in self._nodes.items():
            workflow_node = self._convert_node(graph_node)
            nodes[node_name] = workflow_node

        # Set next_nodes from edges
        for from_node, targets in self._edges.items():
            if from_node in nodes:
                # Filter out END marker
                real_targets = [t for t in targets if t != self.END]
                nodes[from_node].next_nodes = real_targets

        # Handle conditional edges
        for from_node, (router, routes) in self._conditional_edges.items():
            if from_node in nodes:
                # Convert function node to condition node
                original = nodes[from_node]
                condition_node = ConditionNode(
                    id=from_node,
                    name=original.name,
                    condition=lambda ctx, r=router, st=self.state_type: r(st.from_dict(ctx)),
                    branches={k: v for k, v in routes.items() if v != self.END},
                    next_nodes=[],  # Conditions use branches, not next_nodes
                )
                nodes[from_node] = condition_node

        return WorkflowDefinition(
            name=workflow_name,
            description=self.description,
            nodes=nodes,
            start_node=self._entry_point,
            metadata={
                **self._metadata,
                "state_type": self.state_type.__name__,
                "compiled_from": "WorkflowGraph",
            },
        )

    def _convert_node(self, graph_node: GraphNode[S]) -> WorkflowNode:
        """Convert a GraphNode to a WorkflowNode.

        Args:
            graph_node: The graph node to convert

        Returns:
            Appropriate WorkflowNode subclass
        """
        if graph_node.node_type == GraphNodeType.AGENT:
            return AgentNode(
                id=graph_node.name,
                name=graph_node.name,
                role=graph_node.agent_role or "executor",
                goal=graph_node.agent_goal or "",
                tool_budget=graph_node.tool_budget,
                allowed_tools=graph_node.allowed_tools,
                output_key=graph_node.metadata.get("output_key", graph_node.name),
                next_nodes=[],
            )

        elif graph_node.node_type == GraphNodeType.PARALLEL:
            return ParallelNode(
                id=graph_node.name,
                name=graph_node.name,
                parallel_nodes=graph_node.metadata.get("parallel_nodes", []),
                join_strategy=graph_node.metadata.get("join_strategy", "all"),
                next_nodes=[],
            )

        else:
            # Function nodes become TransformNode
            func = graph_node.func

            def transform_wrapper(
                ctx: Dict[str, Any],
                f: Optional[NodeFunc[S]] = func,
                st: Type[S] = self.state_type,
            ) -> Dict[str, Any]:
                if f is None:
                    return ctx
                state = st.from_dict(ctx)
                result = f(state)
                # Handle both sync and async functions
                if inspect.iscoroutine(result):
                    try:
                        asyncio.get_running_loop()
                        result = run_sync_in_thread(result)
                    except RuntimeError:
                        result = run_sync(result)
                return result.to_dict() if hasattr(result, "to_dict") else ctx

            return TransformNode(
                id=graph_node.name,
                name=graph_node.name,
                transform=transform_wrapper,
                next_nodes=[],
            )

    def __repr__(self) -> str:
        """String representation of the graph."""
        return (
            f"WorkflowGraph(name={self.name!r}, "
            f"nodes={list(self._nodes.keys())}, "
            f"entry={self._entry_point!r})"
        )


# Convenience functions for creating graphs
def create_graph(
    state_type: Type[S],
    name: Optional[str] = None,
    description: str = "",
) -> WorkflowGraph[S]:
    """Create a new workflow graph.

    Args:
        state_type: The state dataclass type
        name: Optional workflow name
        description: Optional description

    Returns:
        New WorkflowGraph instance

    Example:
        graph = create_graph(MyState, "my_workflow")
    """
    return WorkflowGraph(state_type, name=name, description=description)


def compile_graph(graph: WorkflowGraph[S], name: Optional[str] = None) -> WorkflowDefinition:
    """Compile a state graph to workflow definition.

    Args:
        graph: The graph to compile
        name: Optional name override

    Returns:
        WorkflowDefinition ready for execution
    """
    return graph.compile(name=name)


# Export all public symbols
__all__ = [
    # Core classes
    "State",
    "WorkflowGraph",
    "GraphNode",
    "GraphNodeType",
    # Type variables
    "S",
    "NodeFunc",
    "RouterFunc",
    # Protocols
    "Compilable",
    # Convenience functions
    "create_graph",
    "compile_graph",
]
