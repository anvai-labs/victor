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

import logging
import time
from contextlib import aclosing
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
from victor.framework.graph_runtime import NodeLifecycleObserver
from victor.workflows.compiler.boundary import (
    NativeWorkflowGraphCompiler,
    ParsedWorkflowDefinition,
    WorkflowCompilationRequest,
)
from victor.workflows.executors.compatibility import CompatibilityNodeExecutorFactory
from victor.workflows.runtime_types import create_initial_workflow_state

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
    max_parallel: Optional[int] = None


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
    node_results: Dict[str, Any] = field(default_factory=dict)

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
                max_parallel=self.config.max_parallel,
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
                max_parallel=self.config.max_parallel,
            )
        return compiler.compile(parsed)

    async def execute(
        self,
        workflow: "WorkflowDefinition",
        initial_context: Optional[Dict[str, Any]] = None,
        *,
        thread_id: Optional[str] = None,
        checkpointer: Optional["CheckpointerProtocol"] = None,
        node_observer: Optional[NodeLifecycleObserver] = None,
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
            result = await compiled.invoke(state, thread_id=thread_id, node_observer=node_observer)

            # Extract user state (exclude internal fields)
            # Convert Pydantic model to dict for compatibility
            # WorkflowStateModel.items() exposes the flat runtime mapping;
            # to_dict() is its persistence representation with nested user data.
            state_dict = dict(result.state.items())
            user_state = {k: v for k, v in state_dict.items() if not k.startswith("_")}

            return ExecutorResult(
                success=result.success,
                state=user_state,
                error=result.error,
                duration_seconds=time.time() - start_time,
                nodes_executed=result.node_history,
                iterations=result.iterations,
                node_results=state_dict.get("_node_results", {}),
                interrupted=getattr(result, "interrupted", False),
                interrupt_node=getattr(result, "interrupt_node", None),
                checkpoints_saved=getattr(result, "checkpoints_saved", 0),
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
        checkpointer: Optional["CheckpointerProtocol"] = None,
    ) -> AsyncIterator[tuple]:
        """Stream workflow execution, yielding after each node.

        Args:
            workflow: The workflow to execute
            initial_context: Initial context data
            thread_id: Thread ID for checkpointing

        Yields:
            Tuple of (node_id, current_state) after each node execution
        """
        compiled = self._compile_workflow(workflow, checkpointer=checkpointer)

        state = create_initial_workflow_state(
            workflow_id=thread_id,
            current_node=workflow.start_node or "",
            initial_state=initial_context,
        )

        # Stream execution
        async with aclosing(compiled.stream(state, thread_id=thread_id)) as events:
            async for node_id, current_state in events:
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
    "CompiledWorkflowExecutor",  # noqa: F822 - lazy compatibility export
    "ExecutionResult",
    "WorkflowExecutor",  # noqa: F822 - lazy compatibility export
]


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


def __getattr__(name: str) -> Any:
    """Resolve old executor imports without retaining a second engine."""
    if name in {"WorkflowExecutor", "CompiledWorkflowExecutor"}:
        from victor.workflows.state_graph_adapter import StateGraphWorkflowExecutor

        return StateGraphWorkflowExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
