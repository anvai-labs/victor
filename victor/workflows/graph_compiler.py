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

"""Workflow graph compiler for unified execution via CompiledGraph.

This module provides compilers that convert various workflow representations
to CompiledGraph, enabling a single execution engine for all workflow types.

Compilers:
- WorkflowGraphCompiler: Compiles WorkflowGraph (graph_dsl) to CompiledGraph
- WorkflowDefinitionCompiler: Compiles WorkflowDefinition to CompiledGraph

This is Phase 4 of the workflow consolidation plan, implementing the
Single Responsibility Principle (SRP) by using CompiledGraph as the
single execution engine.

Example:
    from victor.workflows.graph_compiler import (
        WorkflowGraphCompiler,
        compile_to_graph,
    )
    from victor.workflows.graph_dsl import WorkflowGraph, State

    # Define a workflow graph
    @dataclass
    class MyState(State):
        value: int = 0

    graph = WorkflowGraph(MyState)
    graph.add_node("process", lambda s: s)
    graph.set_entry_point("process")
    graph.set_finish_point("process")

    # Compile to CompiledGraph
    compiler = WorkflowGraphCompiler()
    compiled = compiler.compile(graph)

    # Execute via CompiledGraph.invoke()
    result = await compiled.invoke({"value": 42})
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from victor.framework.graph import (
    CompiledGraph,
    Edge,
    EdgeType,
    Node,
    END,
)
from victor.workflows.context import ExecutionContext, create_execution_context
from victor.workflows.protocols import NodeRunner, NodeRunnerResult

logger = logging.getLogger(__name__)


class NodeCategory(str, Enum):
    """Workflow node type classification for runner lookup."""

    AGENT = "agent"
    COMPUTE = "compute"
    CONDITION = "condition"
    PARALLEL = "parallel"
    TRANSFORM = "transform"
    UNKNOWN = "unknown"


if TYPE_CHECKING:
    from victor.workflows.graph_dsl import (
        WorkflowGraph,
        GraphNode,
        GraphNodeType,
        State,
    )
    from victor.workflows.definition import WorkflowDefinition, WorkflowNode
    from victor.workflows.node_runners import NodeRunnerRegistry

logger = logging.getLogger(__name__)

S = TypeVar("S")  # State type


# =============================================================================
# Compiler Configuration
# =============================================================================


@dataclass
class CompilerConfig:
    """Configuration for workflow graph compilation.

    Attributes:
        use_node_runners: Whether to use NodeRunner protocol for execution.
        runner_registry: Registry of node runners (if use_node_runners=True).
        validate_before_compile: Whether to validate graph before compilation.
        preserve_state_type: Whether to preserve typed state or use dict.
        emitter: Optional ObservabilityEmitter for streaming events (Phase 5).
        enable_observability: Whether to emit observability events during execution.
    """

    use_node_runners: bool = False
    runner_registry: Optional["NodeRunnerRegistry"] = None
    validate_before_compile: bool = True
    preserve_state_type: bool = False
    emitter: Optional[Any] = None  # ObservabilityEmitter, use Any to avoid circular import
    enable_observability: bool = False


# =============================================================================
# Node Wrapper for NodeRunner Integration
# =============================================================================


class NodeRunnerWrapper:
    """Wraps NodeRunner execution as a CompiledGraph-compatible node function.

    This adapter allows NodeRunner implementations to be used within
    CompiledGraph execution, bridging the two execution models.

    Supports observability events via optional emitter (Phase 5).
    """

    def __init__(
        self,
        node_id: str,
        node_config: Dict[str, Any],
        runner: NodeRunner,
        emitter: Optional[Any] = None,  # ObservabilityEmitter
    ):
        """Initialize the wrapper.

        Args:
            node_id: ID of the node being wrapped.
            node_config: Configuration for the node.
            runner: NodeRunner to delegate execution to.
            emitter: Optional ObservabilityEmitter for streaming events.
        """
        self._node_id = node_id
        self._node_config = node_config
        self._runner = runner
        self._emitter = emitter

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node via NodeRunner.

        Args:
            state: Current workflow state (as dict).

        Returns:
            Updated state after node execution.
        """
        import time

        start_time = time.time()
        node_name = self._node_config.get("name", self._node_id)

        # Emit NODE_START if emitter is configured
        if self._emitter:
            self._emitter.emit_node_start(self._node_id, node_name)

        # Convert state to ExecutionContext if needed
        if "_workflow_id" not in state:
            # Wrap in ExecutionContext structure
            context: Dict[str, Any] = {
                "data": state.copy(),
                "_workflow_id": state.get("_workflow_id", ""),
                "_current_node": self._node_id,
                "_node_results": state.get("_node_results", {}),
            }
        else:
            context = state

        try:
            # Execute via NodeRunner
            updated_context, result = await self._runner.execute(
                self._node_id,
                self._node_config,
                context,
            )

            duration = time.time() - start_time

            # Emit NODE_COMPLETE or NODE_ERROR if emitter is configured
            if self._emitter:
                if result.success:
                    self._emitter.emit_node_complete(
                        self._node_id,
                        node_name,
                        duration=duration,
                        output=result.output,
                    )
                else:
                    self._emitter.emit_node_error(
                        self._node_id,
                        error=result.error or "Unknown error",
                        node_name=node_name,
                        duration=duration,
                    )

            # Return updated state
            if "_workflow_id" not in state:
                # Return just the data for simple state
                return updated_context.get("data", state)
            return updated_context

        except Exception as e:
            duration = time.time() - start_time
            if self._emitter:
                self._emitter.emit_node_error(
                    self._node_id,
                    error=str(e),
                    node_name=node_name,
                    duration=duration,
                )
            raise


# =============================================================================
# WorkflowGraph Compiler
# =============================================================================


class WorkflowGraphCompiler(Generic[S]):
    """Compiles WorkflowGraph to CompiledGraph for unified execution.

    This compiler transforms a WorkflowGraph (from graph_dsl.py) into a
    CompiledGraph (from framework/graph.py), enabling execution through
    the single CompiledGraph.invoke() engine.

    The compiler handles:
    - Node function wrapping (sync/async)
    - Edge conversion (normal and conditional)
    - State type preservation
    - Optional NodeRunner integration

    Example:
        compiler = WorkflowGraphCompiler()
        compiled = compiler.compile(my_graph)
        result = await compiled.invoke(initial_state)
    """

    def __init__(self, config: Optional[CompilerConfig] = None):
        """Initialize the compiler.

        Args:
            config: Compilation configuration.
        """
        self._config = config or CompilerConfig()

    def compile(
        self,
        graph: "WorkflowGraph[S]",
        name: Optional[str] = None,
    ) -> CompiledGraph:
        """Compile a WorkflowGraph to CompiledGraph.

        Args:
            graph: The WorkflowGraph to compile.
            name: Optional name override.

        Returns:
            CompiledGraph ready for execution.

        Raises:
            ValueError: If graph validation fails.
        """
        # Validate if configured
        if self._config.validate_before_compile:
            errors = graph.validate()
            if errors:
                raise ValueError(f"Graph validation failed: {'; '.join(errors)}")

        # Import here to avoid circular imports
        from victor.workflows.graph_dsl import GraphNodeType

        # Build nodes
        nodes: Dict[str, Node] = {}
        for node_name, graph_node in graph._nodes.items():
            node_func = self._create_node_function(graph_node, GraphNodeType, graph.state_type)
            nodes[node_name] = Node(
                id=node_name,
                func=node_func,
                metadata={
                    "node_type": graph_node.node_type.value,
                    "original_name": graph_node.name,
                },
            )

        # Build edges
        edges: Dict[str, List[Edge]] = {}

        # Normal edges
        for from_node, targets in graph._edges.items():
            if from_node not in edges:
                edges[from_node] = []

            for target in targets:
                # Convert END marker
                target_id = END if target == graph.END else target
                edges[from_node].append(
                    Edge(
                        source=from_node,
                        target=target_id,
                        edge_type=EdgeType.NORMAL,
                    )
                )

        # Conditional edges
        for from_node, (router, routes) in graph._conditional_edges.items():
            if from_node not in edges:
                edges[from_node] = []

            # Convert routes, replacing END markers
            converted_routes = {k: (END if v == graph.END else v) for k, v in routes.items()}

            # Create state-aware router
            state_type = graph.state_type

            def make_router(r, st):
                def wrapped_router(state: Dict[str, Any]) -> str:
                    # Convert dict state to typed state for router
                    if hasattr(st, "from_dict"):
                        typed_state = st.from_dict(state)
                    else:
                        typed_state = st(**state) if isinstance(state, dict) else state
                    return r(typed_state)

                return wrapped_router

            edges[from_node].append(
                Edge(
                    source=from_node,
                    target=converted_routes,
                    edge_type=EdgeType.CONDITIONAL,
                    condition=make_router(router, state_type),
                )
            )

        # Get entry point
        entry_point = graph._entry_point
        if not entry_point:
            raise ValueError("Graph has no entry point")

        return CompiledGraph(
            nodes=nodes,
            edges=edges,
            entry_point=entry_point,
            state_schema=graph.state_type if self._config.preserve_state_type else None,
        )

    def _create_node_function(
        self,
        graph_node: "GraphNode[S]",
        GraphNodeType: Type,
        state_type: Type[S],
    ) -> Callable[[Any], Any]:
        """Create a node function for CompiledGraph.

        Args:
            graph_node: The graph node to wrap.
            GraphNodeType: Enum for node types.
            state_type: The state type for conversion.

        Returns:
            Callable that executes the node.
        """
        # Check if using NodeRunner
        if self._config.use_node_runners and self._config.runner_registry:
            runner = self._config.runner_registry.get_runner(graph_node.node_type.value)
            if runner:
                node_config = self._extract_node_config(graph_node, GraphNodeType)
                emitter = self._config.emitter if self._config.enable_observability else None
                return NodeRunnerWrapper(graph_node.name, node_config, runner, emitter)

        # Default: wrap the original function with state conversion
        original_func = graph_node.func
        if original_func is None:
            # No-op node
            async def noop(state: Any) -> Any:
                return state

            return noop

        # Create wrapper that handles dict <-> typed state conversion
        async def state_converting_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            # Convert dict to typed state
            if hasattr(state_type, "from_dict"):
                typed_state = state_type.from_dict(state)
            else:
                # Try direct construction
                try:
                    typed_state = state_type(**state)
                except TypeError:
                    # If that fails, pass dict directly
                    typed_state = state

            # Call original function
            if asyncio.iscoroutinefunction(original_func):
                result = await original_func(typed_state)
            else:
                result = original_func(typed_state)

            # Convert result back to dict
            if hasattr(result, "to_dict"):
                return result.to_dict()
            elif hasattr(result, "__dataclass_fields__"):
                # Dataclass - convert to dict
                from dataclasses import asdict

                return asdict(result)
            elif isinstance(result, dict):
                return result
            else:
                # Unknown type, return as-is
                return result

        return state_converting_wrapper

    def _extract_node_config(
        self,
        graph_node: "GraphNode[S]",
        GraphNodeType: Type,
    ) -> Dict[str, Any]:
        """Extract node configuration for NodeRunner.

        Args:
            graph_node: The graph node.
            GraphNodeType: Enum for node types.

        Returns:
            Configuration dict for NodeRunner.
        """
        config: Dict[str, Any] = {
            "type": graph_node.node_type.value,
            "name": graph_node.name,
        }

        # Add type-specific config
        if graph_node.node_type == GraphNodeType.AGENT:
            config.update(
                {
                    "role": graph_node.agent_role or "executor",
                    "goal": graph_node.agent_goal or "",
                    "tool_budget": graph_node.agent_tool_budget or 10,
                }
            )
        elif graph_node.node_type == GraphNodeType.COMPUTE:
            config.update(
                {
                    "handler": graph_node.compute_handler,
                    "tools": graph_node.compute_tools or [],
                }
            )
        elif graph_node.node_type == GraphNodeType.TRANSFORM:
            config.update(
                {
                    "transform": graph_node.func,
                }
            )

        return config


# =============================================================================
# WorkflowDefinition Compiler
# =============================================================================


class WorkflowDefinitionCompiler:
    """Compiles WorkflowDefinition to CompiledGraph.

    This compiler transforms a WorkflowDefinition into a CompiledGraph,
    enabling legacy workflow definitions to use the unified execution engine.

    Example:
        compiler = WorkflowDefinitionCompiler(runner_registry=my_registry)
        compiled = compiler.compile(my_definition)
        result = await compiled.invoke(initial_state)
    """

    def __init__(
        self,
        runner_registry: Optional["NodeRunnerRegistry"] = None,
    ):
        """Initialize the compiler.

        Args:
            runner_registry: Registry of node runners for execution.
        """
        self._runner_registry = runner_registry

    def compile(
        self,
        definition: "WorkflowDefinition",
    ) -> CompiledGraph:
        """Compile a WorkflowDefinition to CompiledGraph.

        Args:
            definition: The workflow definition to compile.

        Returns:
            CompiledGraph ready for execution.
        """
        from victor.workflows.definition import (
            AgentNode as DefAgentNode,
            ComputeNode as DefComputeNode,
            ConditionNode as DefConditionNode,
            ParallelNode as DefParallelNode,
            TransformNode as DefTransformNode,
        )

        nodes: Dict[str, Node] = {}
        edges: Dict[str, List[Edge]] = {}

        # Convert nodes
        for node_id, workflow_node in definition.nodes.items():
            node_func = self._create_node_function(workflow_node)
            nodes[node_id] = Node(
                id=node_id,
                func=node_func,
                metadata={
                    "node_type": type(workflow_node).__name__,
                    "original_name": workflow_node.name,
                },
            )

            # Build edges from next_nodes
            if workflow_node.next_nodes:
                edges[node_id] = [
                    Edge(
                        source=node_id,
                        target=target,
                        edge_type=EdgeType.NORMAL,
                    )
                    for target in workflow_node.next_nodes
                ]

            # Handle condition nodes
            if isinstance(workflow_node, DefConditionNode):
                if node_id not in edges:
                    edges[node_id] = []

                edges[node_id].append(
                    Edge(
                        source=node_id,
                        target=workflow_node.branches,
                        edge_type=EdgeType.CONDITIONAL,
                        condition=lambda state, cond=workflow_node.condition: cond(state),
                    )
                )

        # Add finish edge to END
        for node_id, workflow_node in definition.nodes.items():
            if not workflow_node.next_nodes and node_id not in edges:
                edges[node_id] = [
                    Edge(
                        source=node_id,
                        target=END,
                        edge_type=EdgeType.NORMAL,
                    )
                ]

        entry_point = definition.start_node
        if not entry_point:
            raise ValueError("WorkflowDefinition has no start_node")

        return CompiledGraph(
            nodes=nodes,
            edges=edges,
            entry_point=entry_point,
        )

    def _create_node_function(
        self,
        workflow_node: "WorkflowNode",
    ) -> Callable[[Any], Any]:
        """Create a node function for the workflow node.

        Args:
            workflow_node: The workflow node.

        Returns:
            Async callable for node execution.
        """
        from victor.workflows.definition import (
            AgentNode as DefAgentNode,
            ComputeNode as DefComputeNode,
            ConditionNode as DefConditionNode,
            ParallelNode as DefParallelNode,
            TransformNode as DefTransformNode,
        )

        # Use NodeRunner if available
        if self._runner_registry:
            node_type = self._get_node_type(workflow_node)
            runner = self._runner_registry.get_runner(node_type)
            if runner:
                config = self._extract_config(workflow_node)
                return NodeRunnerWrapper(workflow_node.id, config, runner)

        # Default implementations
        if isinstance(workflow_node, DefTransformNode):
            transform_fn = workflow_node.transform

            async def transform_exec(state: Dict[str, Any]) -> Dict[str, Any]:
                result = transform_fn(state)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, dict):
                    return {**state, **result}
                return state

            return transform_exec

        if isinstance(workflow_node, DefConditionNode):
            # Condition nodes just pass through state
            # Routing is handled by edges
            async def condition_exec(state: Dict[str, Any]) -> Dict[str, Any]:
                return state

            return condition_exec

        # Default pass-through
        async def passthrough(state: Dict[str, Any]) -> Dict[str, Any]:
            return state

        return passthrough

    def _get_node_type(self, workflow_node: "WorkflowNode") -> NodeCategory:
        """Get the node type string for runner lookup."""
        from victor.workflows.definition import (
            AgentNode as DefAgentNode,
            ComputeNode as DefComputeNode,
            ConditionNode as DefConditionNode,
            ParallelNode as DefParallelNode,
            TransformNode as DefTransformNode,
        )

        if isinstance(workflow_node, DefAgentNode):
            return NodeCategory.AGENT
        elif isinstance(workflow_node, DefComputeNode):
            return NodeCategory.COMPUTE
        elif isinstance(workflow_node, DefConditionNode):
            return NodeCategory.CONDITION
        elif isinstance(workflow_node, DefParallelNode):
            return NodeCategory.PARALLEL
        elif isinstance(workflow_node, DefTransformNode):
            return NodeCategory.TRANSFORM
        return NodeCategory.UNKNOWN

    def _extract_config(self, workflow_node: "WorkflowNode") -> Dict[str, Any]:
        """Extract configuration from workflow node."""
        from victor.workflows.definition import (
            AgentNode as DefAgentNode,
            ComputeNode as DefComputeNode,
            TransformNode as DefTransformNode,
        )

        config: Dict[str, Any] = {
            "id": workflow_node.id,
            "name": workflow_node.name,
        }

        if isinstance(workflow_node, DefAgentNode):
            config.update(
                {
                    "role": workflow_node.role,
                    "goal": workflow_node.goal,
                    "tool_budget": workflow_node.tool_budget,
                    "allowed_tools": workflow_node.allowed_tools,
                    "output_key": workflow_node.output_key,
                }
            )
        elif isinstance(workflow_node, DefComputeNode):
            config.update(
                {
                    "handler": workflow_node.handler,
                    "tools": workflow_node.tools,
                }
            )
        elif isinstance(workflow_node, DefTransformNode):
            config.update(
                {
                    "transform": workflow_node.transform,
                }
            )

        return config


# =============================================================================
# Convenience Functions
# =============================================================================


def compile_workflow_graph(
    graph: "WorkflowGraph[S]",
    config: Optional[CompilerConfig] = None,
) -> CompiledGraph:
    """Compile a WorkflowGraph to CompiledGraph.

    Convenience function for one-off compilation.

    Args:
        graph: The WorkflowGraph to compile.
        config: Optional compilation configuration.

    Returns:
        CompiledGraph ready for execution.
    """
    compiler = WorkflowGraphCompiler(config)
    return compiler.compile(graph)


def compile_workflow_definition(
    definition: "WorkflowDefinition",
    runner_registry: Optional["NodeRunnerRegistry"] = None,
) -> CompiledGraph:
    """Compile a WorkflowDefinition to CompiledGraph.

    Convenience function for one-off compilation.

    Args:
        definition: The WorkflowDefinition to compile.
        runner_registry: Optional runner registry for execution.

    Returns:
        CompiledGraph ready for execution.
    """
    compiler = WorkflowDefinitionCompiler(runner_registry)
    return compiler.compile(definition)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "CompilerConfig",
    # Wrappers
    "NodeRunnerWrapper",
    # Compilers
    "WorkflowGraphCompiler",
    "WorkflowDefinitionCompiler",
    # Convenience functions
    "compile_workflow_graph",
    "compile_workflow_definition",
]
