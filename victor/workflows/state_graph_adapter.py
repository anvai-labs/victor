"""WorkflowResult compatibility at the StateGraph execution boundary (ADR-030).

This adapter owns argument/result conversion only. Traversal, joins, checkpoints,
and failure handling belong to StateGraphExecutor and CompiledGraph.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, AsyncIterator, Dict, Optional

from victor_contracts.workflows import ExecutorNodeStatus, NodeResult
from victor.workflows.context import WorkflowContext, WorkflowResult
from victor.workflows.definition import AgentNode, WorkflowDefinition
from victor.workflows.orchestrator_pool import OrchestratorPool
from victor.workflows.unified_executor import ExecutorConfig, ExecutorResult, StateGraphExecutor


def to_workflow_result(workflow_name: str, result: ExecutorResult) -> WorkflowResult:
    """Preserve execution metadata while converting compiled node result types."""
    node_results = {}
    for node_id, value in result.node_results.items():
        if isinstance(value, NodeResult):
            node_results[node_id] = value
            continue
        fields = value if isinstance(value, dict) else vars(value)
        node_results[node_id] = NodeResult(
            node_id=node_id,
            status=(
                ExecutorNodeStatus.COMPLETED
                if fields.get("success", False)
                else ExecutorNodeStatus.FAILED
            ),
            output=fields.get("output"),
            error=fields.get("error"),
            duration_seconds=fields.get("duration_seconds", 0.0),
            tool_calls_used=fields.get("tool_calls_used", 0),
        )
    return WorkflowResult(
        workflow_name=workflow_name,
        success=result.success,
        context=WorkflowContext(data=result.state, node_results=node_results),
        error=result.error,
        total_duration=result.duration_seconds,
        total_tool_calls=sum(node.tool_calls_used for node in node_results.values()),
        nodes_executed=list(result.nodes_executed),
        interrupted=result.interrupted,
        interrupt_node=result.interrupt_node,
    )


class StateGraphWorkflowExecutor:
    """Adapt definition-based workflow callers to the compiled graph engine."""

    def __init__(
        self,
        orchestrator_pool: Any = None,
        *,
        max_parallel: int = 4,
        default_timeout: float = 300.0,
        checkpointer: Any = None,
        cache: Any = None,
        cache_config: Any = None,
    ) -> None:
        if cache is not None or cache_config is not None:
            raise ValueError("Node-result caching is unsupported by the StateGraph adapter")
        if max_parallel < 1:
            raise ValueError("max_parallel must be positive")
        self.orchestrator = orchestrator_pool
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout
        self._checkpointer = checkpointer

    def _executor(
        self, workflow: WorkflowDefinition, timeout: Optional[float]
    ) -> StateGraphExecutor:
        # Configuration is per invocation: concurrent batch runs must not overwrite
        # one another's deadline, profile mapping, or compiled graph.
        config = ExecutorConfig(
            max_parallel=self.max_parallel,
            max_iterations=workflow.max_iterations,
            timeout=(timeout if timeout is not None else workflow.max_execution_timeout_seconds),
        )
        if isinstance(self.orchestrator, OrchestratorPool):
            profiles = {
                node.profile
                for node in workflow.nodes.values()
                if isinstance(node, AgentNode) and node.profile
            }
            orchestrators = {
                profile: self.orchestrator.get_orchestrator(profile) for profile in profiles
            }
            orchestrators["default"] = self.orchestrator.get_default_orchestrator()
            config.default_profile = "default"
            return StateGraphExecutor(orchestrators=orchestrators, config=config)
        return StateGraphExecutor(orchestrator=self.orchestrator, config=config)

    async def execute(
        self,
        workflow: Any,
        initial_state: Optional[Dict[str, Any]] = None,
        *,
        initial_context: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        checkpoint: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute a definition, or delegate an already compiled graph unchanged."""
        state = initial_context if initial_context is not None else initial_state
        state = state if state is not None else {}
        if not isinstance(workflow, WorkflowDefinition):
            if hasattr(workflow, "invoke"):
                kwargs: Dict[str, Any] = {"thread_id": thread_id}
                if checkpoint is not None:
                    kwargs["checkpoint"] = checkpoint
                return await workflow.invoke(state, **kwargs)
            raise TypeError("Expected WorkflowDefinition or a compiled graph with invoke()")
        if checkpoint is not None:
            raise ValueError("Resume definitions using thread_id and a graph checkpointer")
        if workflow.metadata.get("continue_on_failure", False):
            raise ValueError(
                "continue_on_failure is unsupported by StateGraph; handle recoverable failures explicitly inside nodes"
            )
        # Apply the compatibility default without mutating the caller's definition.
        prepared = replace(
            workflow,
            nodes={
                node_id: (
                    replace(node, timeout_seconds=self.default_timeout)
                    if isinstance(node, AgentNode) and node.timeout_seconds is None
                    else node
                )
                for node_id, node in workflow.nodes.items()
            },
        )
        result = await self._executor(prepared, timeout).execute(
            prepared, state, thread_id=thread_id, checkpointer=self._checkpointer
        )
        return to_workflow_result(workflow.name, result)

    async def stream(
        self,
        workflow: Any,
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[Any]:
        """Stream compiled state updates through the surviving engine."""
        runtime = (
            self._executor(workflow, None) if isinstance(workflow, WorkflowDefinition) else None
        )
        source = (
            runtime.stream(workflow, initial_state, thread_id=thread_id)
            if runtime is not None
            else workflow.stream(initial_state, thread_id=thread_id)
        )
        async for event in source:
            yield event
