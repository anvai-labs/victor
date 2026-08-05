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

"""Streaming workflow executor with astream() method.

Phase 3 of Graph Orchestration Streaming API.

Provides StreamingWorkflowExecutor that layers streaming over the
compatibility workflow executor runtime
streaming capabilities via the astream() method, yielding WorkflowStreamChunk
events during workflow execution.

Example:
    executor = StreamingWorkflowExecutor(orchestrator)

    async for chunk in executor.astream(workflow, initial_context):
        if chunk.event_type == WorkflowEventType.NODE_START:
            print(f"Starting node: {chunk.node_name}")
        elif chunk.event_type == WorkflowEventType.AGENT_CONTENT:
            print(chunk.content, end="")
        elif chunk.event_type == WorkflowEventType.WORKFLOW_COMPLETE:
            print(f"Workflow completed! Progress: {chunk.progress}%")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
)

from victor_contracts.workflows import ExecutorNodeStatus, NodeResult
from victor.workflows.context import WorkflowContext, WorkflowResult
from victor.workflows.definition import (
    AgentNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    WorkflowDefinition,
    WorkflowNode,
)
from victor.workflows.runtime_executor_factory import create_legacy_workflow_executor

# Import canonical streaming types from streaming.py (DRY - Phase 5 consolidation)
from victor.workflows.streaming import (
    WorkflowEventType,
    WorkflowStreamChunk,
    WorkflowStreamContext,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.framework.rl.checkpoint_store import CheckpointStore
    from victor.workflows.cache import WorkflowCache, WorkflowCacheConfig

logger = logging.getLogger(__name__)


@dataclass
class _ExecutorStreamContext:
    """Internal context for tracking executor streaming state.

    This is an executor-specific wrapper that provides percentage-based
    progress (0-100) while the canonical WorkflowStreamContext uses
    fraction-based progress (0.0-1.0).

    Attributes:
        workflow_id: Unique identifier for this workflow execution
        total_nodes: Total number of nodes in the workflow
        completed_nodes: Number of nodes that have completed
        is_cancelled: Whether cancellation has been requested
        start_time: When workflow execution started
    """

    workflow_id: str
    total_nodes: int
    completed_nodes: int = 0
    is_cancelled: bool = False
    start_time: float = field(default_factory=time.time)

    @property
    def progress(self) -> float:
        """Calculate progress percentage.

        Returns:
            Progress as a percentage (0.0 to 100.0)
        """
        if self.total_nodes == 0:
            return 0.0
        return (self.completed_nodes / self.total_nodes) * 100.0


@dataclass
class _Subscription:
    """Internal subscription data."""

    event_types: List["WorkflowEventType"]
    callback: Callable[["WorkflowStreamChunk"], None]
    active: bool = True


class StreamingWorkflowExecutor:
    """Workflow executor with streaming support.

    Wraps the compatibility workflow executor to add astream() that yields
    WorkflowStreamChunk events during execution, enabling real-time
    progress tracking and content streaming from agent nodes.

    Features:
    - Yields events for workflow start/complete/error
    - Yields events for node start/complete/error
    - Supports progress tracking as percentage
    - Supports cancellation of active workflows
    - Supports event subscriptions via subscribe()
    - Backward compatible with base WorkflowExecutor.execute()

    Example:
        executor = StreamingWorkflowExecutor(orchestrator)

        # Stream workflow execution
        async for chunk in executor.astream(workflow, {"input": "data"}):
            print(f"[{chunk.event_type.value}] Progress: {chunk.progress}%")

        # Subscribe to specific events
        def on_node_start(chunk):
            print(f"Node started: {chunk.node_name}")

        unsubscribe = executor.subscribe([WorkflowEventType.NODE_START], on_node_start)

        # Cancel active workflow
        executor.cancel_workflow(workflow_id)
    """

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        *,
        max_parallel: int = 4,
        default_timeout: float = 300.0,
        checkpointer: Optional["CheckpointStore"] = None,
        cache: Optional["WorkflowCache"] = None,
        cache_config: Optional["WorkflowCacheConfig"] = None,
    ):
        """Initialize streaming executor.

        Args:
            orchestrator: Agent orchestrator instance
            max_parallel: Maximum parallel executions
            default_timeout: Default timeout per node
            checkpointer: Optional CheckpointStore for persistence
            cache: Optional WorkflowCache for node result caching
            cache_config: Optional config to create a cache
        """
        self._runtime = create_legacy_workflow_executor(
            orchestrator,
            max_parallel=max_parallel,
            default_timeout=default_timeout,
            checkpointer=checkpointer,
            cache=cache,
            cache_config=cache_config,
        )

        # Streaming state (uses _ExecutorStreamContext for percentage-based progress)
        self._active_workflows: Dict[str, _ExecutorStreamContext] = {}
        self._subscriptions: List[_Subscription] = []
        self._queue_poll_interval = 0.1  # 100ms poll interval

    @property
    def sub_agents(self) -> Any:
        """Expose the delegated sub-agent runtime for compatibility."""
        return self._runtime.sub_agents

    @property
    def default_timeout(self) -> float:
        """Expose delegated default timeout for streaming node execution."""
        return self._runtime.default_timeout

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate legacy execute() calls to the wrapped runtime."""
        return await self._runtime.execute(*args, **kwargs)

    def cancel(self) -> None:
        """Delegate cancellation to the wrapped legacy runtime."""
        self._runtime.cancel()

    def __getattr__(self, name: str) -> Any:
        """Delegate compatibility attributes to the wrapped runtime."""
        return getattr(self._runtime, name)

    async def astream(
        self,
        workflow: WorkflowDefinition,
        initial_context: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[WorkflowStreamChunk]:
        """Stream workflow execution events.

        Executes the workflow and yields WorkflowStreamChunk events
        as execution progresses. Events include workflow start/complete,
        node start/complete, and agent content.

        Args:
            workflow: Workflow definition to execute
            initial_context: Initial context data
            timeout: Overall timeout (None = no limit)
            thread_id: Thread ID for checkpointing (enables resume)

        Yields:
            WorkflowStreamChunk events during execution

        Example:
            async for chunk in executor.astream(workflow):
                if chunk.event_type == WorkflowEventType.WORKFLOW_START:
                    print(f"Started workflow: {chunk.workflow_id}")
                elif chunk.event_type == WorkflowEventType.NODE_COMPLETE:
                    print(f"Completed node: {chunk.node_name}")
                elif chunk.is_final:
                    print("Workflow finished!")
        """
        workflow_id = uuid.uuid4().hex[:8]
        thread_id = thread_id or workflow_id

        # Count total nodes for progress tracking
        total_nodes = len(workflow.nodes)

        # Create stream context (uses _ExecutorStreamContext for percentage-based progress)
        stream_ctx = _ExecutorStreamContext(
            workflow_id=workflow_id,
            total_nodes=total_nodes,
        )
        self._active_workflows[workflow_id] = stream_ctx

        # Yield WORKFLOW_START first
        start_chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.WORKFLOW_START,
            workflow_id=workflow_id,
            progress=0.0,
            metadata={
                "workflow_name": workflow.name,
                "total_nodes": total_nodes,
            },
        )
        self._notify_subscribers(start_chunk)
        yield start_chunk

        # Create context for execution
        context = WorkflowContext(
            data=initial_context.copy() if initial_context else {},
            metadata={
                "execution_id": workflow_id,
                "workflow_name": workflow.name,
                "thread_id": thread_id,
            },
        )

        start_time = time.time()
        error_message: Optional[str] = None
        success = True

        try:
            # Execute workflow with streaming
            async for chunk in self._execute_workflow_streaming(
                workflow, context, stream_ctx, thread_id
            ):
                self._notify_subscribers(chunk)
                yield chunk

                # Check for cancellation
                if stream_ctx.is_cancelled:
                    error_message = "Workflow cancelled"
                    success = False
                    break

        except asyncio.TimeoutError:
            error_message = f"Workflow timed out after {timeout}s"
            success = False
            error_chunk = WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_ERROR,
                workflow_id=workflow_id,
                progress=stream_ctx.progress,
                error=error_message,
            )
            self._notify_subscribers(error_chunk)
            yield error_chunk

        except Exception as e:
            logger.error(f"Workflow '{workflow.name}' failed: {e}", exc_info=True)
            error_message = str(e)
            success = False
            error_chunk = WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_ERROR,
                workflow_id=workflow_id,
                progress=stream_ctx.progress,
                error=error_message,
            )
            self._notify_subscribers(error_chunk)
            yield error_chunk

        finally:
            # Cleanup
            self._active_workflows.pop(workflow_id, None)

        # Yield WORKFLOW_COMPLETE (or final error state)
        if success and not context.has_failures():
            total_duration = time.time() - start_time
            complete_chunk = WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_COMPLETE,
                workflow_id=workflow_id,
                progress=100.0,
                is_final=True,
                metadata={
                    "workflow_name": workflow.name,
                    "total_duration": total_duration,
                    "success": True,
                },
            )
            self._notify_subscribers(complete_chunk)
            yield complete_chunk
        else:
            # Yield error completion
            total_duration = time.time() - start_time
            final_chunk = WorkflowStreamChunk(
                event_type=(
                    WorkflowEventType.WORKFLOW_ERROR
                    if error_message
                    else WorkflowEventType.WORKFLOW_COMPLETE
                ),
                workflow_id=workflow_id,
                progress=100.0,
                is_final=True,
                error=error_message,
                metadata={
                    "workflow_name": workflow.name,
                    "total_duration": total_duration,
                    "success": False,
                },
            )
            self._notify_subscribers(final_chunk)
            yield final_chunk

    async def _execute_workflow_streaming(
        self,
        workflow: WorkflowDefinition,
        context: WorkflowContext,
        stream_ctx: _ExecutorStreamContext,
        thread_id: str,
    ) -> AsyncIterator[WorkflowStreamChunk]:
        """Execute workflow DAG with streaming events.

        Traverses the workflow DAG and yields streaming chunks for
        node start/complete events.

        Args:
            workflow: Workflow to execute
            context: Execution context
            stream_ctx: Streaming context for progress
            thread_id: Thread ID for checkpointing

        Yields:
            WorkflowStreamChunk events for node execution
        """
        if not workflow.start_node:
            raise ValueError("Workflow has no start node")

        # Track executed nodes to prevent loops
        executed: Set[str] = set()
        to_execute: List[str] = [workflow.start_node]

        while to_execute:
            if stream_ctx.is_cancelled:
                break

            node_id = to_execute.pop(0)

            if node_id in executed:
                continue

            node = workflow.get_node(node_id)
            if not node:
                logger.warning(f"Node '{node_id}' not found in workflow")
                continue

            # Yield NODE_START
            node_start_chunk = WorkflowStreamChunk(
                event_type=WorkflowEventType.NODE_START,
                workflow_id=stream_ctx.workflow_id,
                progress=stream_ctx.progress,
                node_id=node.id,
                node_name=node.name,
                metadata={"node_type": node.node_type.value},
            )
            yield node_start_chunk

            start_time = time.time()

            # Execute the node (with streaming for agent nodes)
            if isinstance(node, AgentNode):
                result = await self._execute_agent_node_streaming(
                    node, context, stream_ctx, start_time
                )
            else:
                result = await self._execute_node(node, context)

            context.add_result(result)
            executed.add(node_id)
            stream_ctx.completed_nodes += 1

            # Yield NODE_COMPLETE or NODE_ERROR
            if result.success:
                node_complete_chunk = WorkflowStreamChunk(
                    event_type=WorkflowEventType.NODE_COMPLETE,
                    workflow_id=stream_ctx.workflow_id,
                    progress=stream_ctx.progress,
                    node_id=node.id,
                    node_name=node.name,
                    metadata={
                        "duration_seconds": result.duration_seconds,
                        "tool_calls_used": result.tool_calls_used,
                    },
                )
                yield node_complete_chunk
            else:
                node_error_chunk = WorkflowStreamChunk(
                    event_type=WorkflowEventType.NODE_ERROR,
                    workflow_id=stream_ctx.workflow_id,
                    progress=stream_ctx.progress,
                    node_id=node.id,
                    node_name=node.name,
                    error=result.error,
                    metadata={"duration_seconds": result.duration_seconds},
                )
                yield node_error_chunk

                # Stop if not configured to continue on failure
                if not workflow.metadata.get("continue_on_failure", False):
                    logger.warning(f"Stopping workflow due to node failure: {node_id}")
                    break

            # Determine next nodes
            next_nodes = self._get_next_nodes(node, context)

            # Save checkpoint after each node
            if self._runtime._checkpointer and thread_id:
                self._save_workflow_checkpoint(
                    thread_id=thread_id,
                    workflow_name=workflow.name,
                    last_node=node_id,
                    next_node=next_nodes[0] if next_nodes else None,
                    context_data=dict(context.data),
                )

            # Emit RL event
            self._emit_workflow_step_event(
                workflow_name=workflow.name,
                node_id=node_id,
                node_type=(node.node_type.value if hasattr(node, "node_type") else "unknown"),
                success=result.status == ExecutorNodeStatus.COMPLETED,
                duration=result.duration_seconds,
            )

            to_execute.extend(next_nodes)

    async def _execute_agent_node_streaming(
        self,
        node: AgentNode,
        context: WorkflowContext,
        stream_ctx: _ExecutorStreamContext,
        start_time: float,
    ) -> NodeResult:
        """Execute agent node with streaming content.

        Executes an agent node and can yield AGENT_CONTENT chunks
        for streamed content from the agent.

        Args:
            node: Agent node to execute
            context: Execution context
            stream_ctx: Streaming context
            start_time: Node start time

        Returns:
            NodeResult with agent output
        """
        from victor.agent.subagents import SubAgentRole

        # Build agent task from goal and context
        task = self._build_agent_task(node, context)

        # Map role string to SubAgentRole
        role_map = {
            "researcher": SubAgentRole.RESEARCHER,
            "planner": SubAgentRole.PLANNER,
            "executor": SubAgentRole.EXECUTOR,
            "reviewer": SubAgentRole.REVIEWER,
            "tester": SubAgentRole.TESTER,
        }
        role = role_map.get(node.role.lower(), SubAgentRole.EXECUTOR)

        # Spawn and execute agent
        # Note: For future enhancement, we could integrate with agent's
        # streaming capabilities to yield AGENT_CONTENT chunks
        result = await self.sub_agents.spawn(
            role=role,
            task=task,
            tool_budget=node.tool_budget,
            allowed_tools=node.allowed_tools,
            timeout_seconds=int(self.default_timeout),
        )

        # Store output in context
        if result.success and result.summary:
            context.set(node.output_key or node.id, result.summary)

        return NodeResult(
            node_id=node.id,
            status=(ExecutorNodeStatus.COMPLETED if result.success else ExecutorNodeStatus.FAILED),
            output=result.summary,
            error=result.error,
            duration_seconds=time.time() - start_time,
            tool_calls_used=result.tool_calls_used,
        )

    async def _execute_node(self, node: WorkflowNode, context: WorkflowContext) -> NodeResult:
        """Delegate single-node execution to the wrapped compatibility runtime."""
        return await self._runtime._execute_node(node, context)

    def _get_next_nodes(self, node: WorkflowNode, context: WorkflowContext) -> List[str]:
        """Delegate next-node resolution to the wrapped compatibility runtime."""
        return self._runtime._get_next_nodes(node, context)

    def _save_workflow_checkpoint(self, **kwargs: Any) -> None:
        """Delegate checkpoint persistence to the wrapped compatibility runtime."""
        self._runtime._save_workflow_checkpoint(**kwargs)

    def _emit_workflow_step_event(self, **kwargs: Any) -> None:
        """Delegate workflow-step event emission to the wrapped compatibility runtime."""
        self._runtime._emit_workflow_step_event(**kwargs)

    def _build_agent_task(self, node: AgentNode, context: WorkflowContext) -> str:
        """Delegate agent-task construction to the wrapped compatibility runtime."""
        return self._runtime._build_agent_task(node, context)

    def subscribe(
        self,
        event_types: List[WorkflowEventType],
        callback: Callable[[WorkflowStreamChunk], None],
    ) -> Callable[[], None]:
        """Subscribe to workflow events.

        Register a callback to be invoked when events of specified
        types are emitted during workflow execution.

        Args:
            event_types: List of event types to subscribe to
            callback: Function to call with matching chunks

        Returns:
            Unsubscribe function that removes the subscription

        Example:
            def on_progress(chunk):
                print(f"Progress: {chunk.progress}%")

            unsubscribe = executor.subscribe(
                [WorkflowEventType.NODE_COMPLETE],
                on_progress
            )

            # Later, to unsubscribe:
            unsubscribe()
        """
        subscription = _Subscription(
            event_types=event_types,
            callback=callback,
        )
        self._subscriptions.append(subscription)

        def unsubscribe() -> None:
            subscription.active = False
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)

        return unsubscribe

    def _notify_subscribers(self, chunk: WorkflowStreamChunk) -> None:
        """Notify all matching subscribers of an event.

        Args:
            chunk: The chunk to notify about
        """
        for subscription in self._subscriptions:
            if not subscription.active:
                continue
            if chunk.event_type in subscription.event_types:
                try:
                    subscription.callback(chunk)
                except Exception as e:
                    logger.warning(f"Subscriber callback failed: {e}")

    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow.

        Requests cancellation of a workflow that is currently executing.
        The workflow will stop at the next checkpoint.

        Args:
            workflow_id: ID of the workflow to cancel

        Returns:
            True if cancellation was requested, False if workflow not found

        Example:
            async for chunk in executor.astream(workflow):
                if should_cancel:
                    executor.cancel_workflow(chunk.workflow_id)
                    break
        """
        stream_ctx = self._active_workflows.get(workflow_id)
        if stream_ctx is None:
            return False

        stream_ctx.is_cancelled = True
        logger.info(f"Cancellation requested for workflow: {workflow_id}")
        return True

    def get_active_workflows(self) -> List[str]:
        """Get list of active workflow IDs.

        Returns:
            List of workflow IDs that are currently executing
        """
        return list(self._active_workflows.keys())

    def get_workflow_progress(self, workflow_id: str) -> Optional[float]:
        """Get progress of an active workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Progress percentage (0.0 to 100.0) or None if not found
        """
        stream_ctx = self._active_workflows.get(workflow_id)
        if stream_ctx is None:
            return None
        return stream_ctx.progress


__all__ = [
    "WorkflowEventType",
    "WorkflowStreamChunk",
    "WorkflowStreamContext",
    "StreamingWorkflowExecutor",
]
