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

"""StateGraph - LangGraph-compatible graph workflow engine.

This module provides a LangGraph-inspired StateGraph implementation
for building cyclic, stateful agent workflows with typed state management.

Design Principles (SOLID):
    - Single Responsibility: Each class handles one aspect (state, nodes, edges, execution)
    - Open/Closed: Extensible via protocols without modifying core classes
    - Liskov Substitution: All node types implement NodeProtocol
    - Interface Segregation: Small, focused protocols (StateProtocol, EdgeProtocol)
    - Dependency Inversion: Depend on abstractions (protocols) not concretions

Key Differences from victor.workflows:
    - StateGraph: Typed state management (vs. dict-based context)
    - Cyclic Support: Allows cycles with configurable max iterations
    - Edge Types: Explicit conditional/normal edges (vs. next_nodes list)
    - Checkpointing: Full state persistence for recovery
    - Compile Step: Validates and optimizes graph before execution

Example:
    from victor.framework.graph import StateGraph, Node, Edge, END
    from pydantic import BaseModel, Field
    from typing import List, Optional

    # Define typed state (Pydantic RECOMMENDED for type safety)
    class AgentState(BaseModel):
        messages: List[str] = Field(default_factory=list)
        task: str = Field(default="")
        result: Optional[str] = None

        # Dict-like interface for StateGraph compatibility
        class Config:
            arbitrary_types_allowed = True

    # Create graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze", analyze_task)
    graph.add_node("execute", execute_task)
    graph.add_node("review", review_result)

    # Add edges (including cycles)
    graph.add_edge("analyze", "execute")
    graph.add_conditional_edge(
        "execute",
        should_retry,
        {"retry": "analyze", "done": "review"}
    )
    graph.add_edge("review", END)

    # Set entry point
    graph.set_entry_point("analyze")

    # Compile and run
    app = graph.compile()
    result = await app.invoke({"messages": [], "task": "Fix bug"})

Note: TypedDict is still supported for backward compatibility, but Pydantic models
are recommended for better validation and error messages in production code.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import uuid
from typing import (
    Any,
    Awaitable,
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

# Pydantic for type-safe state models (recommended over TypedDict)
from pydantic import BaseModel

from victor.framework.graph_checkpoint import (
    CheckpointerProtocol,
    MemoryCheckpointer,
    RLCheckpointerAdapter,
    WorkflowCheckpoint,
)
from victor.framework.graph_execution import (
    GraphCheckpointManager,
    GraphEventEmitter,
    GraphExecutionResult,
    InterruptHandler,
    IterationController,
    NodeExecutor,
    TimeoutManager,
    snapshot_state_for_result,
)
from victor.framework.graph_algorithms import find_reachable
from victor.framework.graph_examples import AgentStateModel
from victor.framework.graph_merge import (
    default_state_merger,
    resolve_state_merger,
    strict_state_merger,
)
from victor.framework.graph_primitives import (
    ConditionFunctionProtocol,
    Edge,
    EdgeType,
    FrameworkNodeStatus,
    Node,
    NodeFunctionProtocol,
    ParallelBranchExecutionError,
    Send,
    StateProtocol,
    SubgraphNode,
    _MAX_SUBGRAPH_DEPTH,
    _SUBGRAPH_DEPTH_KEY,
)
from victor.framework.graph_runtime import (
    GraphRuntimeOutcome,
    run_graph_execution,
    stream_graph_execution,
)
from victor.framework.graph_state import CopyOnWriteState
from victor.framework.graph_validation import StateValidationError, StateValidator

logger = logging.getLogger(__name__)

# Import focused configs for ISP compliance
from victor.framework.config import (
    GraphConfig as GraphConfig,
    ExecutionConfig,
    CheckpointConfig,
    InterruptConfig,
    PerformanceConfig,
    ObservabilityConfig,
    ValidationConfig,
)

# Type variables for generic state
StateType = TypeVar("StateType", bound=Dict[str, Any])
T = TypeVar("T")

# Sentinel for end of graph
END = "__end__"
START = "__start__"


# Note: GraphConfig is now imported from victor.framework.graph.config
# This provides ISP compliance through focused config classes:
# - ExecutionConfig: execution limits
# - CheckpointConfig: state persistence
# - InterruptConfig: interrupt behavior
# - PerformanceConfig: performance optimizations
# - ObservabilityConfig: observability and eventing
# GraphConfig remains as a facade composing these focused configs


class CompiledGraph(Generic[StateType]):
    """Compiled graph ready for execution.

    The compilation step validates the graph structure and
    creates an optimized execution plan.
    """

    def __init__(
        self,
        nodes: Dict[str, Node],
        edges: Dict[str, List[Edge]],
        entry_point: str,
        state_schema: Optional[Type[StateType]] = None,
        config: Optional[GraphConfig] = None,
        strict_edges: bool = False,
    ):
        """Initialize compiled graph.

        Args:
            nodes: Node registry
            edges: Edge registry (source -> list of edges)
            entry_point: Starting node ID
            state_schema: Optional type for state validation
            config: Execution configuration
            strict_edges: If True, raise EdgeResolutionError when a conditional
                edge doesn't match any case instead of falling through to END.
        """
        self._nodes = nodes
        self._edges = edges
        self._entry_point = entry_point
        self._state_schema = state_schema
        self._config = config or GraphConfig()
        self._strict_edges = strict_edges
        self._debug_hook: Optional[Any] = None  # DebugHook for debugging
        self._state_merger: Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]] = (
            resolve_state_merger(
                self._config.performance.parallel_state_merge_strategy,
                self._config.performance.custom_state_merger,
            )
        )

        # Initialize state validator
        validation_config = self._config.validation
        self._validator: Optional[StateValidator] = None
        if validation_config.enabled and state_schema is not None:
            self._validator = StateValidator(schema=state_schema, strict=validation_config.strict)
        else:
            self._validator = None

    def set_debug_hook(self, hook: Optional[Any]) -> None:
        """Set debug hook for execution.

        Args:
            hook: DebugHook instance or None to disable debugging
        """
        self._debug_hook = hook

    @property
    def graph(self) -> "CompiledGraph[StateType]":
        """Return the compiled graph itself.

        This property provides a self-reference for compatibility
        with APIs that expect a .graph attribute.

        Returns:
            Self reference to the compiled graph
        """
        return self

    def _should_use_cow(self, exec_config: GraphConfig) -> bool:
        """Determine if copy-on-write should be used.

        Args:
            exec_config: Execution configuration (with focused configs)

        Returns:
            True if COW should be enabled
        """
        # Explicit config takes precedence
        if exec_config.performance.use_copy_on_write is not None:
            return exec_config.performance.use_copy_on_write

        # Fall back to settings
        try:
            from victor.config.settings import load_settings

            settings = load_settings()
            return settings.workflow.stategraph_copy_on_write_enabled
        except Exception:
            # Default to True if settings can't be loaded
            return True

    def _validate_state_after_node(
        self,
        current_node: str,
        state: Any,
        exec_config: GraphConfig,
    ) -> Optional[str]:
        """Validate post-node state and return an error string when strict validation fails."""
        if not exec_config.validation.enabled or self._state_schema is None:
            return None

        validator = self._validator or StateValidator(
            schema=self._state_schema,
            strict=exec_config.validation.strict,
        )
        state_dict = state if isinstance(state, dict) else dict(state)
        errors = validator.validate(state_dict)
        if not errors:
            return None

        if exec_config.validation.strict:
            return f"State validation failed after node '{current_node}': {errors}"
        if exec_config.validation.log_errors:
            logger.warning(f"State validation failed after node '{current_node}': {errors}")
        return None

    def _validate_input_state(self, input_state: StateType, exec_config: GraphConfig) -> None:
        """Validate input state according to graph validation config."""
        if not exec_config.validation.enabled or self._state_schema is None:
            return

        validator = self._validator or StateValidator(
            schema=self._state_schema,
            strict=exec_config.validation.strict,
        )
        state_dict = input_state if isinstance(input_state, dict) else dict(input_state)
        errors = validator.validate(state_dict)
        if not errors:
            return

        if exec_config.validation.strict:
            raise StateValidationError(errors, state_dict)
        if exec_config.validation.log_errors:
            logger.warning(f"State validation failed on entry: {errors}")

    def _build_runtime_collaborators(
        self,
        exec_config: GraphConfig,
        *,
        graph_id: str,
        use_copy_on_write: bool,
    ) -> tuple[
        IterationController,
        TimeoutManager,
        InterruptHandler,
        NodeExecutor,
        GraphCheckpointManager,
        GraphEventEmitter,
    ]:
        """Build the focused runtime collaborators used by invoke() and stream()."""
        iteration_controller = IterationController(
            max_iterations=exec_config.execution.max_iterations,
            recursion_limit=exec_config.execution.recursion_limit,
        )
        timeout_manager = TimeoutManager(timeout=exec_config.execution.timeout)
        interrupt_handler = InterruptHandler(
            interrupt_before=exec_config.interrupt.interrupt_before,
            interrupt_after=exec_config.interrupt.interrupt_after,
        )
        node_executor = NodeExecutor(nodes=self._nodes, use_copy_on_write=use_copy_on_write)
        checkpoint_manager = GraphCheckpointManager(
            checkpointer=exec_config.checkpoint.checkpointer
        )
        event_emitter = GraphEventEmitter(
            graph_id=graph_id,
            emit_events=exec_config.observability.emit_events,
        )
        return (
            iteration_controller,
            timeout_manager,
            interrupt_handler,
            node_executor,
            checkpoint_manager,
            event_emitter,
        )

    async def invoke(
        self,
        input_state: StateType,
        *,
        config: Optional[GraphConfig] = None,
        thread_id: Optional[str] = None,
        debug_hook: Optional[Any] = None,
        start_node: Optional[str] = None,
    ) -> GraphExecutionResult[StateType]:
        """Execute the graph using focused runtime helpers."""
        exec_config = config or self._config
        self._validate_input_state(input_state, exec_config)
        thread_id = thread_id or uuid.uuid4().hex
        use_cow = self._should_use_cow(exec_config)
        graph_id = exec_config.observability.graph_id or thread_id

        # Use parameter debug hook or instance debug hook
        hook = debug_hook or self._debug_hook

        (
            iteration_controller,
            timeout_manager,
            interrupt_handler,
            node_executor,
            checkpoint_manager,
            event_emitter,
        ) = self._build_runtime_collaborators(
            exec_config,
            graph_id=graph_id,
            use_copy_on_write=use_cow,
        )

        # Load initial state (from checkpoint or input)
        state, current_node = await checkpoint_manager.load_initial_state(
            thread_id=thread_id,
            input_state=input_state,
            entry_point=self._entry_point,
        )

        # Explicit start_node overrides both checkpoint and entry point
        if start_node is not None:
            current_node = start_node
        runtime_outcome: GraphRuntimeOutcome = await run_graph_execution(
            state=state,
            current_node=current_node,
            end_node_token=END,
            entry_point=self._entry_point,
            node_count=len(self._nodes),
            thread_id=thread_id,
            iteration_controller=iteration_controller,
            timeout_manager=timeout_manager,
            interrupt_handler=interrupt_handler,
            node_executor=node_executor,
            checkpoint_manager=checkpoint_manager,
            event_emitter=event_emitter,
            hook=hook,
            validate_state=lambda node_id, node_state: self._validate_state_after_node(
                node_id,
                node_state,
                exec_config,
            ),
            snapshot_state=snapshot_state_for_result,
            get_next_node=self._get_next_node,
            execute_parallel=lambda sends, executor, timeout_mgr, base_state: self._execute_parallel(
                sends=sends,
                node_executor=executor,
                timeout_manager=timeout_mgr,
                base_state=base_state,
            ),
            record_state_history=exec_config.checkpoint.record_state_history,
        )
        return GraphExecutionResult(
            state=runtime_outcome.state,
            success=runtime_outcome.success,
            error=runtime_outcome.error,
            iterations=runtime_outcome.iterations,
            duration=runtime_outcome.duration,
            node_history=runtime_outcome.node_history,
            state_history=runtime_outcome.state_history,
        )

    def _get_next_node(self, current_node: str, state: Any) -> Union[str, List[Send]]:
        """Determine next node based on edges and state.

        Args:
            current_node: Current node ID
            state: Current state

        Returns:
            Next node ID, a list of Send directives, or END
        """
        edges = self._edges.get(current_node, [])
        if not edges:
            return END

        for edge in edges:
            target = edge.get_target(state)
            if target:
                return target

        # No conditional edge matched
        if self._strict_edges:
            from victor.core.errors import EdgeResolutionError

            raise EdgeResolutionError(
                f"No conditional edge from '{current_node}' matched the "
                f"current state. Use strict_edges=False to allow fallthrough."
            )

        # Default to first edge if no conditional matches
        if edges and edges[0].edge_type == EdgeType.NORMAL:
            return edges[0].target if isinstance(edges[0].target, str) else END

        return END

    async def _execute_parallel(
        self,
        sends: List[Send],
        node_executor: "NodeExecutor",
        timeout_manager: "TimeoutManager",
        base_state: Any,
    ) -> Any:
        """Execute parallel fan-out branches and merge results.

        Args:
            sends: List of Send directives
            node_executor: Executor for individual nodes
            timeout_manager: Timeout manager
            base_state: State before the fan-out

        Returns:
            Merged state after all branches complete
        """

        async def _run_branch(
            send: Send,
        ) -> tuple[str, bool, Optional[str], Dict[str, Any]]:
            branch_state = copy.deepcopy(send.state)
            success, error, result_state = await node_executor.execute(
                node_id=send.node,
                state=branch_state,
                timeout_manager=timeout_manager,
            )
            state_dict = result_state if isinstance(result_state, dict) else dict(result_state)
            return send.node, success, error, state_dict

        branch_results = await asyncio.gather(*[_run_branch(s) for s in sends])

        failures = [
            f"{node}: {error or 'unknown error'}"
            for node, success, error, _state_dict in branch_results
            if not success
        ]
        if failures:
            logger.warning("Parallel branch failures: %s", "; ".join(failures))
            raise ParallelBranchExecutionError("Parallel branch failure(s): " + "; ".join(failures))

        base = base_state if isinstance(base_state, dict) else dict(base_state)
        return self._state_merger(
            base,
            [state_dict for _node, _success, _error, state_dict in branch_results],
        )

    async def stream(
        self,
        input_state: StateType,
        *,
        config: Optional[GraphConfig] = None,
        thread_id: Optional[str] = None,
    ):
        """Stream execution yielding state after each node.

        Args:
            input_state: Initial state
            config: Override execution config
            thread_id: Thread ID for checkpointing

        Yields:
            Tuple of (node_id, state) after each execution
        """
        exec_config = config or self._config
        self._validate_input_state(input_state, exec_config)
        thread_id = thread_id or uuid.uuid4().hex
        use_cow = self._should_use_cow(exec_config)
        graph_id = exec_config.observability.graph_id or thread_id
        (
            iteration_controller,
            timeout_manager,
            interrupt_handler,
            node_executor,
            checkpoint_manager,
            event_emitter,
        ) = self._build_runtime_collaborators(
            exec_config,
            graph_id=graph_id,
            use_copy_on_write=use_cow,
        )
        state, current_node = await checkpoint_manager.load_initial_state(
            thread_id=thread_id,
            input_state=input_state,
            entry_point=self._entry_point,
        )

        async for node_id, node_state in stream_graph_execution(
            state=state,
            current_node=current_node,
            end_node_token=END,
            entry_point=self._entry_point,
            node_count=len(self._nodes),
            thread_id=thread_id,
            iteration_controller=iteration_controller,
            timeout_manager=timeout_manager,
            interrupt_handler=interrupt_handler,
            node_executor=node_executor,
            checkpoint_manager=checkpoint_manager,
            event_emitter=event_emitter,
            hook=self._debug_hook,
            validate_state=lambda node_id, node_state: self._validate_state_after_node(
                node_id,
                node_state,
                exec_config,
            ),
            snapshot_state=snapshot_state_for_result,
            get_next_node=self._get_next_node,
            execute_parallel=lambda sends, executor, timeout_mgr, base_state: self._execute_parallel(
                sends=sends,
                node_executor=executor,
                timeout_manager=timeout_mgr,
                base_state=base_state,
            ),
            record_state_history=exec_config.checkpoint.record_state_history,
        ):
            yield node_id, node_state

    def get_graph_schema(self) -> Dict[str, Any]:
        """Get graph structure as dictionary.

        Returns:
            Dictionary describing nodes and edges
        """
        return {
            "nodes": list(self._nodes.keys()),
            "edges": {
                src: [
                    {
                        "target": e.target,
                        "type": e.edge_type.value,
                    }
                    for e in edges
                ]
                for src, edges in self._edges.items()
            },
            "entry_point": self._entry_point,
        }

    # -----------------------------------------------------------------
    # State History & Replay (LangGraph parity)
    # -----------------------------------------------------------------

    async def get_state_history(
        self,
        thread_id: str,
    ) -> List[WorkflowCheckpoint]:
        """Return all checkpoints for a thread, ordered chronologically.

        Requires a checkpointer to be configured on the graph config.

        Args:
            thread_id: Thread identifier

        Returns:
            List of WorkflowCheckpoint instances (empty if no checkpointer)
        """
        checkpointer = self._config.checkpoint.checkpointer
        if checkpointer is None:
            return []
        return await checkpointer.list(thread_id)

    async def replay_from(
        self,
        thread_id: str,
        checkpoint_id: str,
        *,
        config: Optional[GraphConfig] = None,
    ) -> GraphExecutionResult[StateType]:
        """Load a checkpoint and replay the graph from its node.

        A new thread id is generated to avoid polluting existing history.

        Args:
            thread_id: Original thread id that owns the checkpoint
            checkpoint_id: The specific checkpoint to replay from
            config: Optional execution config override

        Returns:
            GraphExecutionResult from the replayed execution

        Raises:
            ValueError: If no checkpointer is configured or the
                checkpoint is not found
        """
        checkpointer = self._config.checkpoint.checkpointer
        if checkpointer is None:
            raise ValueError("Cannot replay without a checkpointer configured on the graph.")

        all_checkpoints = await checkpointer.list(thread_id)
        target = None
        for cp in all_checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                target = cp
                break

        if target is None:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found for thread '{thread_id}'.")

        replay_thread = f"replay_{uuid.uuid4().hex}"
        return await self.invoke(
            input_state=target.state,
            config=config,
            thread_id=replay_thread,
            start_node=target.node_id,
        )


class StateGraph(Generic[StateType]):
    """StateGraph builder for creating stateful workflows.

    Provides a LangGraph-compatible API for building graph workflows
    with typed state, cyclic support, and checkpointing.

    Example:
        graph = StateGraph(AgentState)
        graph.add_node("analyze", analyze_func)
        graph.add_node("execute", execute_func)
        graph.add_edge("analyze", "execute")
        graph.add_conditional_edge(
            "execute",
            should_retry,
            {"retry": "analyze", "done": END}
        )
        graph.set_entry_point("analyze")

        app = graph.compile()
        result = await app.invoke(initial_state)
    """

    def __init__(
        self,
        state_schema: Optional[Type[StateType]] = None,
        config_schema: Optional[Type] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize StateGraph.

        Args:
            state_schema: Optional type for state validation
            config_schema: Optional type for config validation
            metadata: Optional graph-level metadata
        """
        self._state_schema = state_schema
        self._config_schema = config_schema
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, List[Edge]] = {}
        self._entry_point: Optional[str] = None
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self._state_merger: Optional[
            Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]
        ] = None

    def add_node(
        self,
        node_id: str,
        func: Callable[[StateType], Union[StateType, Awaitable[StateType]]],
        **metadata: Any,
    ) -> "StateGraph[StateType]":
        """Add a node to the graph.

        Args:
            node_id: Unique node identifier
            func: Node execution function
            **metadata: Additional metadata

        Returns:
            Self for chaining

        Raises:
            ValueError: If node already exists
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")

        self._nodes[node_id] = Node(id=node_id, func=func, metadata=metadata)
        logger.debug(f"Added node: {node_id}")
        return self

    @property
    def node_ids(self) -> List[str]:
        """Return the graph's node identifiers."""
        return list(self._nodes.keys())

    def add_edge(
        self,
        source: str,
        target: str,
    ) -> "StateGraph[StateType]":
        """Add a normal edge between nodes.

        Args:
            source: Source node ID
            target: Target node ID (or END)

        Returns:
            Self for chaining
        """
        if source not in self._edges:
            self._edges[source] = []

        edge = Edge(source=source, target=target, edge_type=EdgeType.NORMAL)
        self._edges[source].append(edge)
        logger.debug(f"Added edge: {source} -> {target}")
        return self

    def add_conditional_edge(
        self,
        source: str,
        condition: Callable[[StateType], str],
        branches: Dict[str, str],
    ) -> "StateGraph[StateType]":
        """Add a conditional edge with multiple branches.

        Args:
            source: Source node ID
            condition: Function that returns branch name
            branches: Mapping from branch names to target node IDs

        Returns:
            Self for chaining
        """
        if source not in self._edges:
            self._edges[source] = []

        edge = Edge(
            source=source,
            target=branches,
            edge_type=EdgeType.CONDITIONAL,
            condition=condition,
        )
        self._edges[source].append(edge)
        logger.debug(f"Added conditional edge: {source} -> {list(branches.values())}")
        return self

    def add_subgraph(
        self,
        node_id: str,
        compiled_graph: "CompiledGraph",
        *,
        input_mapper: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        output_mapper: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        **metadata: Any,
    ) -> "StateGraph[StateType]":
        """Add a subgraph node for modular graph-of-graphs composition.

        The inner ``compiled_graph`` is invoked via
        :meth:`CompiledGraph.invoke` when this node executes.

        Args:
            node_id: Unique node identifier
            compiled_graph: A pre-compiled graph to run as a subgraph
            input_mapper: Optional function to map parent state to subgraph input
            output_mapper: Optional function to map subgraph output back to parent
            **metadata: Additional metadata

        Returns:
            Self for chaining

        Raises:
            ValueError: If node already exists
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")

        subgraph_node = SubgraphNode(
            id=node_id,
            compiled_graph=compiled_graph,
            input_mapper=input_mapper,
            output_mapper=output_mapper,
            metadata=metadata,
        )
        # SubgraphNode has the same execute() signature as Node,
        # so NodeExecutor handles it uniformly.
        self._nodes[node_id] = subgraph_node  # type: ignore[assignment]
        logger.debug(f"Added subgraph node: {node_id}")
        return self

    def add_state_merger(
        self,
        func: Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]],
    ) -> "StateGraph[StateType]":
        """Set a custom state merger for fan-out / parallel branches.

        The merger receives the base state and a list of branch result
        states and must return the merged state.

        Args:
            func: Merger function ``(base_state, branch_states) -> merged``

        Returns:
            Self for chaining
        """
        self._state_merger = func
        return self

    def set_entry_point(self, node_id: str) -> "StateGraph[StateType]":
        """Set the entry point node.

        Args:
            node_id: Node to start execution from

        Returns:
            Self for chaining
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' not found")
        self._entry_point = node_id
        return self

    def set_finish_point(self, node_id: str) -> "StateGraph[StateType]":
        """Set a node as finish point (adds edge to END).

        Args:
            node_id: Node that finishes the graph

        Returns:
            Self for chaining
        """
        return self.add_edge(node_id, END)

    def compile(
        self,
        checkpointer: Optional[CheckpointerProtocol] = None,
        strict_edges: bool = False,
        **config_kwargs: Any,
    ) -> CompiledGraph[StateType]:
        """Compile the graph for execution.

        Validates graph structure and creates optimized execution plan.

        Args:
            checkpointer: Optional checkpointer for persistence
            strict_edges: If True, raise EdgeResolutionError when a conditional
                edge doesn't match any case instead of falling through to END.
            **config_kwargs: Additional config options

        Returns:
            CompiledGraph ready for execution

        Raises:
            ValueError: If graph is invalid
        """
        # Validate
        errors = self._validate()
        if errors:
            raise ValueError(f"Invalid graph: {'; '.join(errors)}")
        warnings = self._validation_warnings()
        for warning in warnings:
            logger.warning("StateGraph validation warning: %s", warning)

        # Create config (use from_legacy to support both legacy and focused config formats)
        config = GraphConfig.from_legacy(
            checkpointer=checkpointer,
            **config_kwargs,
        )

        compiled = CompiledGraph(
            nodes=self._nodes.copy(),
            edges={k: list(v) for k, v in self._edges.items()},
            entry_point=self._entry_point,
            state_schema=self._state_schema,
            config=config,
            strict_edges=strict_edges,
        )
        if self._state_merger is not None:
            compiled._state_merger = self._state_merger
        return compiled

    def _validate(self) -> List[str]:
        """Validate graph structure.

        Returns:
            List of error messages
        """
        errors = []

        if not self._nodes:
            errors.append("Graph has no nodes")

        if not self._entry_point:
            errors.append("No entry point set")
        elif self._entry_point not in self._nodes:
            errors.append(f"Entry point '{self._entry_point}' not found")

        # Check edge targets exist
        for source, edges in self._edges.items():
            if source not in self._nodes:
                errors.append(f"Edge source '{source}' not found")

            for edge in edges:
                if isinstance(edge.target, str):
                    if edge.target != END and edge.target not in self._nodes:
                        errors.append(f"Edge target '{edge.target}' not found")
                elif isinstance(edge.target, dict):
                    for branch, target in edge.target.items():
                        if target != END and target not in self._nodes:
                            errors.append(
                                f"Conditional target '{target}' not found " f"(branch: {branch})"
                            )

        # Check all nodes are reachable
        reachable = self._find_reachable()
        for node_id in self._nodes:
            if node_id not in reachable and node_id != self._entry_point:
                errors.append(f"Node '{node_id}' is unreachable")

        return errors

    def _validation_warnings(self) -> List[str]:
        """Return non-fatal structural warnings for the graph."""
        warnings: List[str] = []
        reachable = self._find_reachable()

        for node_id in reachable:
            if node_id == END:
                continue
            outgoing_edges = self._edges.get(node_id, [])
            if outgoing_edges:
                continue
            warnings.append(f"Node '{node_id}' has no outgoing edges and will terminate implicitly")

        return warnings

    def _find_reachable(self) -> Set[str]:
        """Find all reachable nodes from entry point."""
        if not self._entry_point:
            return set()

        adjacency: Dict[str, List[str]] = {}
        for source, edges in self._edges.items():
            targets: List[str] = []
            for edge in edges:
                if isinstance(edge.target, str):
                    targets.append(edge.target)
                elif isinstance(edge.target, dict):
                    targets.extend(edge.target.values())
            adjacency[source] = targets

        return find_reachable(self._entry_point, adjacency, sentinel=END)

    @classmethod
    def from_schema(
        cls,
        schema: Union[Dict[str, Any], str],
        state_schema: Optional[Type[StateType]] = None,
        node_registry: Optional[Dict[str, Callable]] = None,
        condition_registry: Optional[Dict[str, Callable]] = None,
    ) -> "StateGraph[StateType]":
        """Create StateGraph from schema dictionary or YAML string.

        This enables dynamic graph generation from serialized schemas,
        supporting Phase 3.0 requirements for workflow persistence and
        external graph definition.

        Args:
            schema: Either a dictionary schema or YAML string containing:
                - nodes: List of node definitions with id and type
                - edges: List of edge definitions with source, target, type
                - entry_point: Starting node ID
                - Optional: state_schema, metadata
            state_schema: Optional TypedDict type for state validation
            node_registry: Registry of node functions (for 'function' type nodes)
                Maps node function names to callable functions
            condition_registry: Registry of condition functions (for conditional edges)
                Maps condition function names to callable functions

        Returns:
            StateGraph instance ready for compilation

        Raises:
            ValueError: If schema is invalid or missing required fields
            TypeError: If node/condition types are unsupported

        Example:
            # Define schema
            schema = {
                "nodes": [
                    {"id": "analyze", "type": "function", "func": "analyze_task"},
                    {"id": "execute", "type": "function", "func": "execute_task"},
                ],
                "edges": [
                    {"source": "analyze", "target": "execute", "type": "normal"},
                    {
                        "source": "execute",
                        "target": {"retry": "analyze", "done": "__end__"},
                        "type": "conditional",
                        "condition": "should_retry"
                    }
                ],
                "entry_point": "analyze"
            }

            # Create registries
            node_registry = {
                "analyze_task": analyze_task_func,
                "execute_task": execute_task_func,
            }
            condition_registry = {
                "should_retry": should_retry_func,
            }

            # Deserialize
            graph = StateGraph.from_schema(
                schema,
                state_schema=AgentState,
                node_registry=node_registry,
                condition_registry=condition_registry
            )

            # Compile and execute
            app = graph.compile()
            result = await app.invoke(initial_state)

        Example with YAML:
            yaml_schema = \"""
            nodes:
              - id: analyze
                type: function
                func: analyze_task
              - id: execute
                type: function
                func: execute_task
            edges:
              - source: analyze
                target: execute
                type: normal
              - source: execute
                target:
                  retry: analyze
                  done: __end__
                type: conditional
                condition: should_retry
            entry_point: analyze
            \"""

            graph = StateGraph.from_schema(
                yaml_schema,
                node_registry=node_registry,
                condition_registry=condition_registry
            )
        """
        import yaml

        # Parse YAML if string input
        if isinstance(schema, str):
            try:
                schema_dict = yaml.safe_load(schema)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML schema: {e}") from e
        else:
            schema_dict = schema

        # Validate required fields
        required_fields = ["nodes", "edges", "entry_point"]
        missing_fields = [f for f in required_fields if f not in schema_dict]
        if missing_fields:
            raise ValueError(f"Schema missing required fields: {missing_fields}")

        # Initialize registries with defaults
        node_registry = node_registry or {}
        condition_registry = condition_registry or {}

        # Create StateGraph instance
        graph = cls(
            state_schema=state_schema,
            metadata=schema_dict.get("metadata"),
        )

        # Add nodes
        for node_def in schema_dict["nodes"]:
            if not isinstance(node_def, dict):
                raise ValueError(f"Invalid node definition: {node_def}")

            node_id = node_def.get("id")
            if not node_id:
                raise ValueError("Node definition must have 'id' field")

            node_type = node_def.get("type", "function")

            if node_type == "function":
                # Function node - look up in registry
                func_name = node_def.get("func")
                if not func_name:
                    raise ValueError(f"Function node '{node_id}' must specify 'func'")

                if func_name not in node_registry:
                    raise ValueError(
                        f"Node function '{func_name}' not found in node_registry. "
                        f"Available: {list(node_registry.keys())}"
                    )

                node_func = node_registry[func_name]
                metadata = {k: v for k, v in node_def.items() if k not in ["id", "type", "func"]}
                graph.add_node(node_id, node_func, **metadata)

            elif node_type == "passthrough":
                # Passthrough node (identity function)
                def passthrough_func(state):
                    return state

                metadata = {k: v for k, v in node_def.items() if k not in ["id", "type"]}
                graph.add_node(node_id, passthrough_func, **metadata)

            elif node_type == "agent":
                # Agent node - placeholder for workflow execution
                # The actual agent execution is handled by the workflow executor
                def create_agent_placeholder(node_config):
                    def agent_placeholder(state):
                        # Store node config in state for executor to use
                        return {
                            **state,
                            "_pending_agent": node_config,
                        }

                    return agent_placeholder

                metadata = {k: v for k, v in node_def.items() if k not in ["id", "type"]}
                graph.add_node(node_id, create_agent_placeholder(node_def), **metadata)

            elif node_type == "compute":
                # Compute node - placeholder for handler execution
                # The actual compute execution is handled by the workflow executor
                def create_compute_placeholder(node_config):
                    def compute_placeholder(state):
                        # Store node config in state for executor to use
                        return {
                            **state,
                            "_pending_compute": node_config,
                        }

                    return compute_placeholder

                metadata = {k: v for k, v in node_def.items() if k not in ["id", "type"]}
                graph.add_node(node_id, create_compute_placeholder(node_def), **metadata)

            elif node_type == "subgraph":
                # Subgraph node - look up compiled graph in node_registry
                subgraph_key = node_def.get("graph")
                if not subgraph_key:
                    raise ValueError(f"Subgraph node '{node_id}' must specify 'graph' key")
                if subgraph_key not in node_registry:
                    raise ValueError(
                        f"Subgraph '{subgraph_key}' not found in node_registry. "
                        f"Available: {list(node_registry.keys())}"
                    )
                inner_graph = node_registry[subgraph_key]
                graph.add_subgraph(node_id, inner_graph)

            else:
                raise TypeError(f"Unsupported node type: {node_type}")

        # Add edges
        for edge_def in schema_dict["edges"]:
            if not isinstance(edge_def, dict):
                raise ValueError(f"Invalid edge definition: {edge_def}")

            source = edge_def.get("source")
            if not source:
                raise ValueError("Edge definition must have 'source' field")

            target = edge_def.get("target")
            if target is None:
                raise ValueError("Edge definition must have 'target' field")

            edge_type = edge_def.get("type", "normal")

            if edge_type == "normal":
                graph.add_edge(source, target)

            elif edge_type == "conditional":
                condition_name = edge_def.get("condition")
                if not condition_name:
                    raise ValueError(f"Conditional edge from '{source}' must specify 'condition'")

                if condition_name not in condition_registry:
                    raise ValueError(
                        f"Condition function '{condition_name}' not found in "
                        f"condition_registry. Available: {list(condition_registry.keys())}"
                    )

                if not isinstance(target, dict):
                    raise ValueError(
                        f"Conditional edge target must be dict mapping branches to nodes, "
                        f"got: {type(target)}"
                    )

                condition_func = condition_registry[condition_name]
                graph.add_conditional_edge(source, condition_func, target)

            else:
                raise TypeError(f"Unsupported edge type: {edge_type}")

        # Set entry point
        entry_point = schema_dict["entry_point"]
        if entry_point not in graph._nodes:
            raise ValueError(
                f"Entry point '{entry_point}' not found in nodes. "
                f"Available nodes: {list(graph._nodes.keys())}"
            )

        graph.set_entry_point(entry_point)

        return graph


# Convenience factory functions
def create_graph(
    state_schema: Optional[Type[StateType]] = None,
) -> StateGraph[StateType]:
    """Create a new StateGraph.

    Args:
        state_schema: Optional type for state validation

    Returns:
        New StateGraph instance
    """
    return StateGraph(state_schema)


__all__ = [
    # Core types
    "StateGraph",
    "CompiledGraph",
    "Node",
    "Edge",
    "EdgeType",
    "FrameworkNodeStatus",
    # Subgraph composition
    "SubgraphNode",
    # Fan-out / parallel execution
    "Send",
    "default_state_merger",
    "strict_state_merger",
    "ParallelBranchExecutionError",
    # State management
    "CopyOnWriteState",  # Copy-on-write state wrapper (P2 scalability)
    "AgentStateModel",  # Example Pydantic state model (recommended over TypedDict)
    # Execution
    "GraphExecutionResult",
    "GraphConfig",
    # Checkpointing
    "WorkflowCheckpoint",
    "CheckpointerProtocol",
    "MemoryCheckpointer",
    "RLCheckpointerAdapter",  # Uses existing RL CheckpointStore
    # Protocols
    "StateProtocol",
    "NodeFunctionProtocol",
    "ConditionFunctionProtocol",
    # Constants
    "END",
    "START",
    # Factory
    "create_graph",
]
