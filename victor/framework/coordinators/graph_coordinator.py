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

"""Graph Execution Coordinator.

Handles execution and streaming of StateGraph/CompiledGraph workflows.
This coordinator encapsulates graph execution functionality with proper
LSP-compliant handling of GraphExecutionResult types.

Features:
- Execute compiled StateGraphs
- Stream execution events for real-time updates
- LSP-compliant polymorphic result handling
- Support for WorkflowGraph compilation
- Integration with NodeRunner registry
"""

from __future__ import annotations

import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
)

if TYPE_CHECKING:
    from victor.framework.graph import CompiledGraph
    from victor.framework.workflow_engine import WorkflowExecutionResult, WorkflowEvent
    from victor.workflows.graph_dsl import WorkflowGraph
    from victor.workflows.node_runners import NodeRunnerRegistry
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

logger = logging.getLogger(__name__)


class GraphTurnExecutor:
    """Coordinator for StateGraph/CompiledGraph execution.

    Handles all aspects of graph-based workflow execution including:
    - Direct CompiledGraph execution
    - WorkflowGraph compilation and execution
    - Streaming execution events
    - LSP-compliant result handling

    The coordinator properly handles the polymorphic return type from
    CompiledGraph.invoke(), which may return either a GraphExecutionResult
    object with a .state attribute or a raw state dictionary.

    Example:
        coordinator = GraphTurnExecutor()

        # Execute a compiled graph
        result = await coordinator.execute(
            compiled_graph,
            initial_state={"input": "data"},
        )

        # Stream execution events
        async for event in coordinator.stream(compiled_graph, state):
            print(f"{event.node_id}: {event.event_type}")
    """

    def __init__(
        self,
        runner_registry: Optional["NodeRunnerRegistry"] = None,
    ) -> None:
        """Initialize the graph execution coordinator.

        Args:
            runner_registry: Optional NodeRunnerRegistry for unified execution
        """
        self._runner_registry = runner_registry
        self._unified_compilers: Dict[bool, "UnifiedWorkflowCompiler"] = {}

    def set_runner_registry(self, registry: "NodeRunnerRegistry") -> None:
        """Set the NodeRunner registry for unified execution.

        Args:
            registry: NodeRunnerRegistry with configured runners
        """
        self._runner_registry = registry
        self._unified_compilers = {}

    def _get_unified_compiler(self, *, use_node_runners: bool) -> "UnifiedWorkflowCompiler":
        """Get or create a canonical unified compiler for graph execution."""
        compiler = self._unified_compilers.get(use_node_runners)
        if compiler is None:
            from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

            compiler = UnifiedWorkflowCompiler(
                runner_registry=self._runner_registry if use_node_runners else None,
                enable_caching=False,
            )
            self._unified_compilers[use_node_runners] = compiler
        return compiler

    async def execute(
        self,
        graph: "CompiledGraph",
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "WorkflowExecutionResult":
        """Execute a compiled StateGraph.

        Properly handles the polymorphic return type from graph.invoke(),
        ensuring LSP compliance by checking for the .state attribute.

        Args:
            graph: Compiled StateGraph to execute
            initial_state: Initial workflow state
            **kwargs: Additional execution parameters

        Returns:
            WorkflowExecutionResult with final state and metadata
        """
        from victor.framework.workflow_engine import WorkflowExecutionResult

        start_time = time.time()

        try:
            # Execute the graph directly
            result = await graph.invoke(initial_state or {})

            duration = time.time() - start_time

            # Handle polymorphic return type (LSP compliance)
            # CompiledGraph.invoke() returns GraphExecutionResult with .state attribute
            # Some graphs may return state dict directly for backward compatibility
            if hasattr(result, "state"):
                final_state = result.state
                nodes_executed = getattr(result, "node_history", [])
                success = getattr(result, "success", True)
                error = getattr(result, "error", None)
            else:
                # Backward compatibility: result is the final state dict
                final_state = result
                nodes_executed = []
                success = True
                error = None

            return WorkflowExecutionResult(
                success=success,
                final_state=final_state,
                nodes_executed=(
                    nodes_executed
                    if nodes_executed
                    else (
                        list(graph._execution_order) if hasattr(graph, "_execution_order") else []
                    )
                ),
                duration_seconds=duration,
                error=error,
            )

        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            return WorkflowExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def stream(
        self,
        graph: "CompiledGraph",
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator["WorkflowEvent"]:
        """Stream events from StateGraph execution.

        Uses the graph's stream method if available, otherwise falls
        back to invoke.

        Args:
            graph: Compiled StateGraph to execute
            initial_state: Initial workflow state
            **kwargs: Additional execution parameters

        Yields:
            WorkflowEvent for each execution step
        """
        from victor.framework.workflow_engine import WorkflowEvent

        try:
            # Use the graph's stream method if available
            if hasattr(graph, "stream"):
                async for node_id, state in graph.stream(initial_state or {}):
                    yield WorkflowEvent(
                        event_type="node_complete",
                        node_id=node_id,
                        timestamp=time.time(),
                        state_snapshot=state,
                    )
            else:
                # Fallback to invoke
                result = await graph.invoke(initial_state or {})

                # Handle polymorphic return type
                if hasattr(result, "state"):
                    final_state = result.state
                else:
                    final_state = result

                yield WorkflowEvent(
                    event_type="complete",
                    node_id="",
                    timestamp=time.time(),
                    state_snapshot=final_state,
                )

        except Exception as e:
            logger.error(f"Graph streaming failed: {e}")
            yield WorkflowEvent(
                event_type="error",
                node_id="",
                timestamp=time.time(),
                data={"error": str(e)},
            )

    async def execute_workflow_graph(
        self,
        graph: "WorkflowGraph",
        initial_state: Optional[Dict[str, Any]] = None,
        use_node_runners: bool = False,
        **kwargs: Any,
    ) -> "WorkflowExecutionResult":
        """Execute a WorkflowGraph via CompiledGraph (unified execution path).

        This method compiles a WorkflowGraph to CompiledGraph and executes
        it through the single CompiledGraph.invoke() engine, providing a
        unified execution path for all workflow types.

        Args:
            graph: WorkflowGraph to compile and execute
            initial_state: Initial workflow state
            use_node_runners: Whether to use NodeRunner protocol for execution
            **kwargs: Additional execution parameters

        Returns:
            WorkflowExecutionResult with final state and metadata
        """
        from victor.framework.workflow_engine import WorkflowExecutionResult

        start_time = time.time()

        try:
            compiler = self._get_unified_compiler(use_node_runners=use_node_runners)
            compiled = compiler.compile_graph(graph)

            # Execute via CompiledGraph.invoke()
            result = await compiled.invoke(initial_state or {})

            duration = time.time() - start_time

            # Extract execution info from result (LSP compliance)
            if hasattr(result, "state"):
                final_state = result.state
                nodes_executed = getattr(result, "node_history", [])
                success = getattr(result, "success", True)
                error = getattr(result, "error", None)
            else:
                # Result is the final state dict
                final_state = result
                nodes_executed = []
                success = True
                error = None

            return WorkflowExecutionResult(
                success=success,
                final_state=final_state,
                nodes_executed=nodes_executed,
                duration_seconds=duration,
                error=error,
            )

        except Exception as e:
            logger.error(f"WorkflowGraph execution failed: {e}")
            return WorkflowExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def execute_definition_compiled(
        self,
        workflow: Any,
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "WorkflowExecutionResult":
        """Execute a WorkflowDefinition via CompiledGraph (unified execution path).

        This method compiles a WorkflowDefinition to CompiledGraph and executes
        it through the single CompiledGraph.invoke() engine.

        Args:
            workflow: WorkflowDefinition to compile and execute
            initial_state: Initial workflow state
            **kwargs: Additional execution parameters

        Returns:
            WorkflowExecutionResult with final state and metadata
        """
        from victor.framework.workflow_engine import WorkflowExecutionResult

        start_time = time.time()

        try:
            compiler = self._get_unified_compiler(
                use_node_runners=self._runner_registry is not None
            )
            compiled = compiler.compile_definition(workflow)

            # Execute via CompiledGraph.invoke()
            result = await compiled.invoke(initial_state or {})

            duration = time.time() - start_time

            # Extract execution info from result (LSP compliance)
            if hasattr(result, "state"):
                final_state = result.state
                nodes_executed = getattr(result, "node_history", [])
                success = getattr(result, "success", True)
                error = getattr(result, "error", None)
            else:
                final_state = result
                nodes_executed = []
                success = True
                error = None

            return WorkflowExecutionResult(
                success=success,
                final_state=final_state,
                nodes_executed=nodes_executed,
                duration_seconds=duration,
                error=error,
            )

        except Exception as e:
            logger.error(f"WorkflowDefinition compiled execution failed: {e}")
            return WorkflowExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


__all__ = ["GraphTurnExecutor"]
