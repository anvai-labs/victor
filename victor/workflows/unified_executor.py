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

"""Victor StateGraph Workflow Executor.

Victor's native workflow execution engine based on StateGraph, providing:
- LangGraph-compatible API for seamless migration
- TypedDict state schemas for type-safe workflow state
- Checkpointing for resumable long-running workflows
- YAML DSL that maps directly to StateGraph patterns
- Streaming execution with real-time state updates

Example:
    from victor.workflows import StateGraphExecutor

    # Execute workflow
    executor = StateGraphExecutor(orchestrator)
    result = await executor.execute(workflow, {"input": "data"})

    # With checkpointing for resumable workflows
    result = await executor.execute(
        workflow,
        {"input": "data"},
        thread_id="session-123",
    )

    # Streaming execution
    async for node_id, state in executor.stream(workflow, {"input": "data"}):
        print(f"Node {node_id} completed: {state}")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.framework.graph import CheckpointerProtocol
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import WorkflowDefinition
from victor.framework.graph import MemoryCheckpointer
from victor.workflows.compiler.boundary import (
    NativeWorkflowGraphCompiler,
    ParsedWorkflowDefinition,
    WorkflowCompilationRequest,
)
from victor.workflows.executors.compatibility import CompatibilityNodeExecutorFactory
from victor.workflows.runtime_types import WorkflowState, create_initial_workflow_state

logger = logging.getLogger(__name__)


@dataclass
class ExecutorConfig:
    """Configuration for Victor's StateGraph workflow execution.

    Attributes:
        enable_checkpointing: Enable checkpointing for resumable workflows
        max_iterations: Max loop iterations for cyclic workflows
        timeout: Execution timeout in seconds
        interrupt_nodes: Nodes to interrupt before (for human-in-the-loop)
    """

    enable_checkpointing: bool = True
    max_iterations: int = 25
    timeout: Optional[float] = None
    interrupt_nodes: List[str] = field(default_factory=list)
    default_profile: Optional[str] = None  # Default profile for nodes without explicit profile


@dataclass
class ExecutorResult:
    """Result from Victor StateGraph workflow execution.

    Attributes:
        success: Whether execution succeeded
        state: Final state/context data
        error: Error message if failed
        duration_seconds: Total execution time
        nodes_executed: List of node IDs that were executed
        iterations: Number of iterations (for cyclic workflows)
        checkpoints_saved: Number of checkpoints saved
        interrupted: Whether execution was interrupted (for HITL)
        interrupt_node: Node where execution was interrupted
    """

    success: bool
    state: Dict[str, Any]
    error: Optional[str] = None
    duration_seconds: float = 0.0
    nodes_executed: List[str] = field(default_factory=list)
    iterations: int = 0
    checkpoints_saved: int = 0
    interrupted: bool = False
    interrupt_node: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from final state."""
        return self.state.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "success": self.success,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 3),
            "nodes_executed": self.nodes_executed,
            "iterations": self.iterations,
            "checkpoints_saved": self.checkpoints_saved,
            "interrupted": self.interrupted,
            "interrupt_node": self.interrupt_node,
            "state_keys": (
                list(self.state.to_dict().keys())
                if hasattr(self.state, "to_dict")
                else list(self.state.keys())
            ),
        }


class StateGraphExecutor:
    """Victor's native StateGraph workflow executor.

    A production-grade workflow executor with LangGraph-compatible API:
    - TypedDict state schemas for type-safe workflow state
    - Checkpointing for resumable long-running workflows
    - YAML DSL that maps directly to StateGraph patterns
    - Streaming execution with real-time state updates
    - Human-in-the-loop interrupt support

    Example:
        # Create executor
        executor = StateGraphExecutor(orchestrator)

        # Execute workflow with checkpointing
        result = await executor.execute(
            workflow,
            initial_context={"data_path": "/data/input.csv"},
            thread_id="session-123",
        )

        if result.success:
            print(f"Output: {result.get('output')}")
            print(f"Nodes executed: {result.nodes_executed}")
        else:
            print(f"Failed: {result.error}")

        # Streaming execution
        async for node_id, state in executor.stream(workflow, {"input": "data"}):
            print(f"Completed: {node_id}")
    """

    def __init__(
        self,
        orchestrator: Optional["AgentOrchestrator"] = None,
        orchestrators: Optional[Dict[str, "AgentOrchestrator"]] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        config: Optional[ExecutorConfig] = None,
    ):
        """Initialize the workflow executor.

        Args:
            orchestrator: Single agent orchestrator for all agent nodes (legacy)
            orchestrators: Dict mapping profile names to orchestrators (new)
            tool_registry: Tool registry for compute nodes
            config: Execution configuration
        """
        # Support both single orchestrator (legacy) and multiple orchestrators (new)
        if orchestrator is not None and orchestrators is not None:
            raise ValueError("Cannot specify both orchestrator and orchestrators")

        if orchestrators:
            self.orchestrators = orchestrators
            self.orchestrator = orchestrators.get(
                config.default_profile
                if config and config.default_profile
                else next(iter(orchestrators))
            )
        elif orchestrator:
            self.orchestrators = {"default": orchestrator}
            self.orchestrator = orchestrator
        else:
            self.orchestrators = {}
            self.orchestrator = None

        self.tool_registry = tool_registry or self._get_default_tool_registry()
        self.config = config or ExecutorConfig()
        self._compiler: Optional[NativeWorkflowGraphCompiler] = None

    def _get_default_tool_registry(self) -> Optional["ToolRegistry"]:
        """Get the default tool registry if available."""
        try:
            from victor.tools.registry import get_tool_registry

            return get_tool_registry()
        except Exception:
            return None

    def _create_executor_factory(self) -> CompatibilityNodeExecutorFactory:
        """Create the canonical node executor factory for workflow compilation."""
        return CompatibilityNodeExecutorFactory(
            orchestrator=self.orchestrator,
            orchestrators=self.orchestrators,
            tool_registry=self.tool_registry,
        )

    def _build_checkpointer_factory(
        self,
        checkpointer: Optional["CheckpointerProtocol"] = None,
    ) -> Callable[[], Optional["CheckpointerProtocol"]]:
        """Create the checkpointer factory expected by the boundary compiler."""

        def create_checkpointer() -> Optional["CheckpointerProtocol"]:
            return checkpointer or MemoryCheckpointer()

        return create_checkpointer

    def _get_compiler(self) -> NativeWorkflowGraphCompiler:
        """Get or create the canonical StateGraph compiler."""
        if self._compiler is None:
            self._compiler = NativeWorkflowGraphCompiler(
                node_executor_factory=self._create_executor_factory(),
                enable_checkpointing=self.config.enable_checkpointing,
                interrupt_on_hitl=bool(self.config.interrupt_nodes),
            )
        return self._compiler

    def _compile_workflow(
        self,
        workflow: "WorkflowDefinition",
        *,
        checkpointer: Optional["CheckpointerProtocol"] = None,
    ) -> Any:
        """Compile a workflow definition through the canonical boundary compiler."""
        errors = workflow.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {'; '.join(errors)}")

        compiled_workflow = replace(
            workflow,
            max_iterations=self.config.max_iterations,
            max_execution_timeout_seconds=self.config.timeout,
        )
        parsed = ParsedWorkflowDefinition(
            request=WorkflowCompilationRequest(
                source=f"workflow://{workflow.name}",
                workflow_name=workflow.name,
                validate=True,
            ),
            workflow=compiled_workflow,
        )

        if checkpointer is None:
            compiler = self._get_compiler()
        else:
            compiler = NativeWorkflowGraphCompiler(
                node_executor_factory=self._create_executor_factory(),
                checkpointer_factory=self._build_checkpointer_factory(checkpointer),
                enable_checkpointing=True,
                interrupt_on_hitl=bool(self.config.interrupt_nodes),
            )
        return compiler.compile(parsed)

    async def execute(
        self,
        workflow: "WorkflowDefinition",
        initial_context: Optional[Dict[str, Any]] = None,
        *,
        thread_id: Optional[str] = None,
        checkpointer: Optional["CheckpointerProtocol"] = None,
    ) -> ExecutorResult:
        """Execute a workflow using Victor's StateGraph engine.

        Args:
            workflow: The workflow definition to execute
            initial_context: Initial context/state data
            thread_id: Thread ID for checkpointing
            checkpointer: Custom checkpointer

        Returns:
            ExecutorResult with execution outcome
        """
        start_time = time.time()

        logger.info(f"Executing workflow '{workflow.name}'")

        try:
            compiled = self._compile_workflow(workflow, checkpointer=checkpointer)

            state = create_initial_workflow_state(
                workflow_id=thread_id,
                current_node=workflow.start_node or "",
                initial_state=initial_context,
            )

            # Execute
            result = await compiled.invoke(state, thread_id=thread_id)

            # Extract user state (exclude internal fields)
            # Convert Pydantic model to dict for compatibility
            state_dict = (
                result.state.to_dict() if hasattr(result.state, "to_dict") else result.state
            )
            user_state = {k: v for k, v in state_dict.items() if not k.startswith("_")}

            return ExecutorResult(
                success=result.success,
                state=user_state,
                error=result.error,
                duration_seconds=time.time() - start_time,
                nodes_executed=result.node_history,
                iterations=result.iterations,
            )

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return ExecutorResult(
                success=False,
                state=initial_context or {},
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def stream(
        self,
        workflow: "WorkflowDefinition",
        initial_context: Optional[Dict[str, Any]] = None,
        *,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[tuple]:
        """Stream workflow execution, yielding after each node.

        Args:
            workflow: The workflow to execute
            initial_context: Initial context data
            thread_id: Thread ID for checkpointing

        Yields:
            Tuple of (node_id, current_state) after each node execution
        """
        compiled = self._compile_workflow(workflow)

        state = create_initial_workflow_state(
            workflow_id=thread_id,
            current_node=workflow.start_node or "",
            initial_state=initial_context,
        )

        # Stream execution
        async for node_id, current_state in compiled.stream(state, thread_id=thread_id):
            # Filter internal fields for user
            user_state = {k: v for k, v in current_state.items() if not k.startswith("_")}
            yield (node_id, user_state)


# Global default executor instance
_default_executor: Optional[StateGraphExecutor] = None


def get_executor(
    orchestrator: Optional["AgentOrchestrator"] = None,
    config: Optional[ExecutorConfig] = None,
) -> StateGraphExecutor:
    """Get or create the default StateGraph executor.

    Args:
        orchestrator: Agent orchestrator (uses existing if not provided)
        config: Execution config (uses existing if not provided)

    Returns:
        StateGraphExecutor instance
    """
    global _default_executor

    if _default_executor is None or orchestrator is not None:
        _default_executor = StateGraphExecutor(
            orchestrator=orchestrator,
            config=config,
        )
    return _default_executor


async def execute_workflow(
    workflow: "WorkflowDefinition",
    initial_context: Optional[Dict[str, Any]] = None,
    *,
    orchestrator: Optional["AgentOrchestrator"] = None,
    thread_id: Optional[str] = None,
) -> ExecutorResult:
    """Execute a workflow using the StateGraph executor.

    Convenience function for one-off workflow execution.

    Args:
        workflow: The workflow to execute
        initial_context: Initial context data
        orchestrator: Agent orchestrator
        thread_id: Thread ID for checkpointing

    Returns:
        ExecutorResult
    """
    executor = get_executor(orchestrator)
    return await executor.execute(
        workflow,
        initial_context,
        thread_id=thread_id,
    )


__all__ = [
    # Config
    "ExecutorConfig",
    # Result
    "ExecutorResult",
    # Executor
    "StateGraphExecutor",
    # Convenience
    "get_executor",
    "execute_workflow",
    # Compiled graph executor (consolidated from compiled_executor.py)
    "CompiledWorkflowExecutor",
    "ExecutionResult",
    "WorkflowExecutor",
]


# ---------------------------------------------------------------------------
# Compiled graph executor — consolidated from victor/workflows/compiled_executor.py
# ---------------------------------------------------------------------------


class CompiledWorkflowExecutor:
    """Executor for compiled workflow graphs.

    Wraps compiled StateGraph execution with consistent interface.
    Delegates to the compiled graph's invoke/stream methods.
    """

    def __init__(
        self,
        orchestrator_pool: Any,
        *,
        max_parallel: int = 4,
        default_timeout: float = 300.0,
        checkpointer: Optional[Any] = None,
        cache: Optional[Any] = None,
        cache_config: Optional[Any] = None,
    ):
        """Initialize the compiled workflow executor.

        Args:
            orchestrator_pool: Agent orchestrator or pool of orchestrators
            max_parallel: Maximum parallel executions (for compatibility)
            default_timeout: Default timeout for node execution
            checkpointer: Optional checkpointer for persistence
            cache: Optional workflow cache
            cache_config: Optional cache configuration
        """
        self._orchestrator_pool = orchestrator_pool
        self.orchestrator = orchestrator_pool
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout
        self._checkpointer = checkpointer
        self._cache_config = cache_config
        self._active_executions: Dict[str, Any] = {}
        self._sub_agents: Any = None

        # Create cache from config if provided and no explicit cache
        if cache is None and cache_config is not None:
            try:
                from victor.workflows.cache import WorkflowCache

                self.cache = WorkflowCache(config=cache_config)
            except ImportError:
                self.cache = None
        else:
            self.cache = cache

    @property
    def sub_agents(self) -> Any:
        """Lazily materialize the delegated sub-agent runtime for legacy workflows."""
        if self._sub_agents is None:
            from victor.agent.subagents import SubAgentOrchestrator

            self._sub_agents = SubAgentOrchestrator(self.orchestrator)
        return self._sub_agents

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return workflow cache statistics for backward compatibility."""
        if self.cache is None or not hasattr(self.cache, "get_stats"):
            return {
                "enabled": False,
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "evictions": 0,
                "skipped_non_cacheable": 0,
                "hit_rate": 0.0,
                "current_size": 0,
                "max_size": 0,
            }
        return self.cache.get_stats()

    async def execute(
        self,
        workflow_or_graph: Any,
        initial_state: Optional[Dict[str, Any]] = None,
        *,
        initial_context: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        checkpoint: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute a workflow or compiled workflow graph.

        Args:
            workflow_or_graph: WorkflowDefinition or compiled graph
            initial_state: Initial state data (alias for initial_context)
            initial_context: Initial context data (for backward compatibility)
            thread_id: Thread ID for checkpointing
            checkpoint: Checkpoint ID to resume from
            timeout: Execution timeout

        Returns:
            ExecutionResult or WorkflowResult
        """
        import time
        from victor.workflows.context import WorkflowResult, WorkflowContext

        # Support both initial_state and initial_context parameters
        context_data = initial_context or initial_state or {}

        logger.info(f"Executing workflow '{getattr(workflow_or_graph, 'name', 'unknown')}'...")

        start_time = time.time()

        # Check if this is a WorkflowDefinition (not yet compiled)
        if hasattr(workflow_or_graph, "start_node") and hasattr(workflow_or_graph, "nodes"):
            # Execute WorkflowDefinition directly
            context = WorkflowContext(data=context_data.copy())
            context.metadata.update(
                {
                    "thread_id": thread_id,
                    "workflow_name": getattr(workflow_or_graph, "name", "unknown"),
                    "workflow": workflow_or_graph,
                    "execution_id": uuid.uuid4().hex,
                }
            )

            if not workflow_or_graph.start_node:
                return WorkflowResult(
                    workflow_name=getattr(workflow_or_graph, "name", "unknown"),
                    success=False,
                    context=context,
                    error="Workflow has no start node",
                    total_duration=time.time() - start_time,
                )

            try:
                await self._execute_workflow(
                    workflow_or_graph, context, timeout=timeout, thread_id=thread_id
                )
                total_duration = time.time() - start_time
                total_tool_calls = sum(
                    result.tool_calls_used for result in context.node_results.values()
                )

                return WorkflowResult(
                    workflow_name=getattr(workflow_or_graph, "name", "unknown"),
                    success=not context.has_failures(),
                    context=context,
                    error=context.get_error() if context.has_failures() else None,
                    total_duration=total_duration,
                    total_tool_calls=total_tool_calls,
                )
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}", exc_info=True)
                return WorkflowResult(
                    workflow_name=getattr(workflow_or_graph, "name", "unknown"),
                    success=False,
                    context=context,
                    error=str(e),
                    total_duration=time.time() - start_time,
                    total_tool_calls=sum(
                        result.tool_calls_used for result in context.node_results.values()
                    ),
                )

        # Handle compiled graph with invoke method
        if hasattr(workflow_or_graph, "invoke"):
            return await workflow_or_graph.invoke(
                initial_state or {},
                thread_id=thread_id,
                checkpoint=checkpoint,
            )

        # Fallback for unknown types
        return WorkflowResult(
            workflow_name="unknown",
            success=False,
            context=WorkflowContext(data=context_data),
            error="Unable to execute workflow: unsupported type",
            total_duration=time.time() - start_time,
        )

    async def _execute_workflow(
        self,
        workflow: Any,
        context: Any,
        timeout: Optional[float] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Execute all nodes in a WorkflowDefinition.

        Args:
            workflow: WorkflowDefinition with nodes and start_node
            context: WorkflowContext to update
            timeout: Optional timeout in seconds
            thread_id: Optional thread ID for checkpointing
        """
        if not workflow.start_node:
            return

        # Track executed nodes to prevent loops
        executed = set()
        to_execute = [workflow.start_node]

        while to_execute:
            node_id = to_execute.pop(0)

            if node_id in executed:
                continue

            node = workflow.get_node(node_id) if hasattr(workflow, "get_node") else None
            if not node:
                continue

            # Execute the node with timeout if provided
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(
                        self._execute_node(node, context),
                        timeout=timeout,
                    )
                else:
                    result = await self._execute_node(node, context)
                context.add_result(result)
                executed.add(node_id)

                if not result.success and not workflow.metadata.get("continue_on_failure", False):
                    break

                # Save checkpoint after each node if checkpointer is available
                if self._checkpointer:
                    self._save_workflow_checkpoint(
                        state={
                            "last_node": node_id,
                            "next_node": (
                                self._get_next_nodes(node, context)[0]
                                if self._get_next_nodes(node, context)
                                else None
                            ),
                            "context": dict(context.data),
                        },
                    )

                # Get next nodes
                next_nodes = self._get_next_nodes(node, context)
                to_execute.extend(next_nodes)

            except asyncio.TimeoutError:
                # Record timeout failure
                from victor_contracts.workflows import NodeResult, ExecutorNodeStatus

                context.add_result(
                    NodeResult(
                        node_id=node_id,
                        status=ExecutorNodeStatus.FAILED,
                        error=f"Node execution timed out after {timeout}s",
                    )
                )
                break

            except Exception as e:
                # Record failure
                from victor_contracts.workflows import NodeResult, ExecutorNodeStatus

                context.add_result(
                    NodeResult(
                        node_id=node_id,
                        status=ExecutorNodeStatus.FAILED,
                        error=str(e),
                    )
                )

                # Stop on failure unless continue_on_failure is set
                if not workflow.metadata.get("continue_on_failure", False):
                    break

    async def stream(
        self,
        compiled_graph: Any,
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[Any]:
        """Stream execution events from a compiled workflow."""
        if hasattr(compiled_graph, "stream"):
            async for event in compiled_graph.stream(initial_state, thread_id=thread_id):
                yield event

    async def _execute_agent_node(
        self,
        node: Any,
        context: Any,
        start_time: float,
    ) -> Any:
        """Execute an agent node.

        This method is provided for test compatibility. Tests may patch
        this method to provide mock implementations.

        Args:
            node: The agent node to execute
            context: The execution context
            start_time: When execution started

        Returns:
            NodeResult with execution outcome
        """
        from victor_contracts.workflows import NodeResult, ExecutorNodeStatus

        task = self._build_agent_task(node, context)
        role = self._map_role_to_enum(getattr(node, "role", "executor"))
        result = await self.sub_agents.spawn(
            role=role,
            task=task,
            tool_budget=getattr(node, "tool_budget", 15),
            allowed_tools=getattr(node, "allowed_tools", None),
            timeout_seconds=int(getattr(node, "timeout_seconds", None) or self.default_timeout),
            disable_embeddings=getattr(node, "disable_embeddings", False),
        )

        if result.success and result.summary:
            context.set(getattr(node, "output_key", None) or node.id, result.summary)

        return NodeResult(
            node_id=node.id,
            status=(ExecutorNodeStatus.COMPLETED if result.success else ExecutorNodeStatus.FAILED),
            output=result.summary,
            error=result.error,
            tool_calls_used=getattr(result, "tool_calls_used", 0),
            duration_seconds=time.time() - start_time,
        )

    async def _execute_node(self, node: Any, context: Any) -> Any:
        """Execute a single workflow node.

        Args:
            node: The workflow node to execute
            context: The execution context

        Returns:
            NodeResult with execution outcome
        """
        from victor.workflows.definition import (
            AgentNode,
            ConditionNode,
            ParallelNode,
            TransformNode,
        )
        from victor_contracts.workflows import NodeResult, ExecutorNodeStatus
        import time

        start_time = time.time()
        context_snapshot = dict(getattr(context, "data", {}))

        try:
            if self.cache is not None:
                cached_result = self.cache.get(node, context_snapshot)
                if cached_result is not None:
                    self._apply_cached_result_to_context(node, context, cached_result)
                    return cached_result

            if isinstance(node, AgentNode):
                return await self._execute_agent_node(node, context, start_time)

            if isinstance(node, ConditionNode):
                try:
                    branch = node.condition(context.data)
                    next_node = node.branches.get(branch)
                    result = NodeResult(
                        node_id=node.id,
                        status=ExecutorNodeStatus.COMPLETED,
                        output={"branch": branch, "next_node": next_node},
                        duration_seconds=time.time() - start_time,
                    )
                    self._cache_result(node, context_snapshot, result)
                    return result
                except Exception as exc:
                    return NodeResult(
                        node_id=node.id,
                        status=ExecutorNodeStatus.FAILED,
                        error=f"Condition evaluation failed: {exc}",
                        duration_seconds=time.time() - start_time,
                    )

            if isinstance(node, ParallelNode):
                workflow = context.metadata.get("workflow")
                parallel_nodes = []
                for node_id in getattr(node, "parallel_nodes", []):
                    candidate = workflow.get_node(node_id) if workflow is not None else None
                    if candidate is not None:
                        parallel_nodes.append(candidate)

                if not parallel_nodes:
                    return NodeResult(
                        node_id=node.id,
                        status=ExecutorNodeStatus.SKIPPED,
                        duration_seconds=time.time() - start_time,
                    )

                semaphore = asyncio.Semaphore(self.max_parallel)

                async def execute_with_semaphore(parallel_node: Any) -> Any:
                    async with semaphore:
                        return await self._execute_node(parallel_node, context)

                results = await asyncio.gather(
                    *(execute_with_semaphore(parallel_node) for parallel_node in parallel_nodes),
                    return_exceptions=True,
                )

                node_results = []
                for result in results:
                    if isinstance(result, Exception):
                        node_results.append(
                            NodeResult(
                                node_id="unknown",
                                status=ExecutorNodeStatus.FAILED,
                                error=str(result),
                            )
                        )
                    else:
                        node_results.append(result)
                        context.add_result(result)

                if node.join_strategy == "all":
                    success = all(result.success for result in node_results)
                elif node.join_strategy == "any":
                    success = any(result.success for result in node_results)
                else:
                    success = True

                return NodeResult(
                    node_id=node.id,
                    status=(ExecutorNodeStatus.COMPLETED if success else ExecutorNodeStatus.FAILED),
                    output={"results": [result.output for result in node_results if result.output]},
                    tool_calls_used=sum(result.tool_calls_used for result in node_results),
                    duration_seconds=time.time() - start_time,
                )

            if isinstance(node, TransformNode):
                try:
                    result_data = node.transform(context.data)
                    if isinstance(result_data, dict):
                        context.update(result_data)
                    result = NodeResult(
                        node_id=node.id,
                        status=ExecutorNodeStatus.COMPLETED,
                        output=result_data,
                        duration_seconds=time.time() - start_time,
                    )
                    self._cache_result(node, context_snapshot, result)
                    return result
                except Exception as exc:
                    return NodeResult(
                        node_id=node.id,
                        status=ExecutorNodeStatus.FAILED,
                        error=f"Transform failed: {exc}",
                        duration_seconds=time.time() - start_time,
                    )

            return NodeResult(
                node_id=getattr(node, "id", "unknown"),
                status=ExecutorNodeStatus.SKIPPED,
                duration_seconds=time.time() - start_time,
            )
        except Exception as exc:
            return NodeResult(
                node_id=getattr(node, "id", "unknown"),
                status=ExecutorNodeStatus.FAILED,
                error=str(exc),
                duration_seconds=time.time() - start_time,
            )

    def _apply_cached_result_to_context(self, node: Any, context: Any, result: Any) -> None:
        """Replay deterministic cached side effects onto the workflow context."""
        output = getattr(result, "output", None)
        output_key = getattr(node, "output_key", None)

        if isinstance(output, dict):
            context.update(output)
        elif output_key and output is not None:
            context.set(output_key, output)

    def _cache_result(self, node: Any, context_snapshot: Dict[str, Any], result: Any) -> None:
        """Store deterministic node results in the workflow cache when available."""
        if self.cache is None or not hasattr(self.cache, "set"):
            return
        self.cache.set(node, context_snapshot, result)

    async def _execute_chain_handler(
        self,
        node: Any,
        context: Any,
        chain_name: str,
        start_time: float,
    ) -> Any:
        """Execute a legacy chain handler via the deprecated registry seam."""
        from victor.workflows import executor as legacy_executor
        from victor_contracts.workflows import NodeResult, ExecutorNodeStatus

        registry = legacy_executor.get_chain_registry()
        chain = registry.create(chain_name)
        input_data = {
            param: context.get(source)
            for param, source in (getattr(node, "input_mapping", None) or {}).items()
        }

        try:
            if hasattr(chain, "invoke"):
                output = await legacy_executor.asyncio.to_thread(chain.invoke, input_data)
            else:
                output = await legacy_executor.asyncio.to_thread(chain, **input_data)
        except Exception as exc:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=f"Chain handler '{chain_name}' failed: {exc}",
                duration_seconds=time.time() - start_time,
            )

        if getattr(node, "output_key", None):
            context.set(node.output_key, output)

        return NodeResult(
            node_id=node.id,
            status=ExecutorNodeStatus.COMPLETED,
            output=output,
            duration_seconds=time.time() - start_time,
        )

    def _get_next_nodes(self, node: Any, context: Any) -> List[str]:
        """Get next nodes based on current node type.

        Args:
            node: The current node
            context: The execution context

        Returns:
            List of next node IDs
        """
        from victor.workflows.definition import ConditionNode

        if isinstance(node, ConditionNode):
            route = node.condition(context.data)
            return [node.branches.get(route)] if route in node.branches else []
        else:
            return list(node.next_nodes) if hasattr(node, "next_nodes") else []

    def _save_workflow_checkpoint(self, **kwargs: Any) -> None:
        """Save a workflow checkpoint."""
        if self._checkpointer and hasattr(self._checkpointer, "create_checkpoint"):
            self._checkpointer.create_checkpoint(**kwargs)

    def _emit_workflow_step_event(self, **kwargs: Any) -> None:
        """Emit workflow step event."""
        pass

    def _emit_workflow_completed_event(self, **kwargs: Any) -> None:
        """Emit workflow completed event."""
        pass

    def _build_agent_task(self, node: Any, context: Any) -> str:
        """Build agent task from node and context.

        Args:
            node: The agent node
            context: The execution context

        Returns:
            Task string for the agent
        """
        import json

        goal = getattr(node, "goal", "") or ""
        substitutions: Dict[str, str] = {}
        for param, key in getattr(node, "input_mapping", {}).items():
            value = context.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                try:
                    value = json.dumps(value, indent=2, default=str)
                except (TypeError, ValueError):
                    value = str(value)
            substitutions[param] = value

        for key, value in substitutions.items():
            goal = goal.replace(f"{{{key}}}", value)
        return goal

    def _map_role_to_enum(self, role: str) -> Any:
        """Map a legacy workflow role string to SubAgentRole."""
        from victor.agent.subagents import SubAgentRole

        role_map = {
            "researcher": SubAgentRole.RESEARCHER,
            "planner": SubAgentRole.PLANNER,
            "executor": SubAgentRole.EXECUTOR,
            "reviewer": SubAgentRole.REVIEWER,
            "tester": SubAgentRole.TESTER,
        }
        return role_map.get(role.lower(), SubAgentRole.EXECUTOR)

    async def execute_by_name(
        self,
        workflow_name: str,
        initial_context: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute a workflow looked up from the global registry."""
        from victor.workflows.registry import get_global_registry
        from victor.workflows.context import WorkflowResult, WorkflowContext

        registry = get_global_registry()
        workflow = registry.get(workflow_name)
        if not workflow:
            return WorkflowResult(
                workflow_name=workflow_name,
                success=False,
                context=WorkflowContext(),
                error=f"Workflow '{workflow_name}' not found",
            )
        return await self.execute(workflow, initial_context, timeout=timeout)

    def cancel(self) -> None:
        """Cancel all active executions."""
        self._active_executions.clear()


class ExecutionResult:
    """Execution result from compiled workflow executor."""

    def __init__(self, final_state: Dict[str, Any], metrics: Dict[str, Any]):
        self._final_state = final_state
        self._metrics = metrics

    @property
    def final_state(self) -> Dict[str, Any]:
        return self._final_state

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics


# Backward-compatibility alias
WorkflowExecutor = CompiledWorkflowExecutor
