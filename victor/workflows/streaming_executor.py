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

"""Workflow lifecycle chunks backed exclusively by canonical graph execution.

This adapter transports node events; the compiled graph owns all traversal,
parallel execution, failures, and checkpoints. Agent token events remain data
model capabilities, not a claim that sub-agent spawn streams content.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing, suppress
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from victor.workflows.context import WorkflowResult
from victor.workflows.definition import ParallelNode, WorkflowDefinition
from victor.workflows.runtime_executor_factory import create_legacy_workflow_executor
from victor.workflows.streaming import WorkflowEventType, WorkflowStreamChunk, WorkflowStreamContext

logger = logging.getLogger(__name__)


@dataclass
class _ExecutorStreamContext:
    workflow_id: str
    total_nodes: int
    completed_nodes: int = 0
    is_cancelled: bool = False
    start_time: float = field(default_factory=time.time)
    task: Optional[asyncio.Task[Any]] = None

    @property
    def progress(self) -> float:
        return (
            min(100.0, 100.0 * self.completed_nodes / self.total_nodes) if self.total_nodes else 0.0
        )


@dataclass
class _Subscription:
    event_types: List[WorkflowEventType]
    callback: Callable[[WorkflowStreamChunk], None]
    active: bool = True


class StreamingWorkflowExecutor:
    """Expose lifecycle events, subscriptions, and cancellation for compiled runs."""

    def __init__(
        self,
        orchestrator: Any = None,
        *,
        max_parallel: int = 4,
        default_timeout: float = 300.0,
        checkpointer: Any = None,
        cache: Any = None,
        cache_config: Any = None,
    ) -> None:
        self._runtime = create_legacy_workflow_executor(
            orchestrator,
            max_parallel=max_parallel,
            default_timeout=default_timeout,
            checkpointer=checkpointer,
            cache=cache,
            cache_config=cache_config,
        )
        self._active_workflows: Dict[str, _ExecutorStreamContext] = {}
        self._subscriptions: List[_Subscription] = []

    @property
    def default_timeout(self) -> float:
        return self._runtime.default_timeout

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        return await self._runtime.execute(*args, **kwargs)

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Expose canonical completed-node state tuples for framework callers."""
        async with aclosing(self._runtime.stream(*args, **kwargs)) as events:
            async for event in events:
                yield event

    def cancel(self) -> None:
        """Cancel streams owned by this wrapper, leaving other executors alone."""
        for workflow_id in list(self._active_workflows):
            self.cancel_workflow(workflow_id)

    async def astream(
        self,
        workflow: WorkflowDefinition,
        initial_context: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[WorkflowStreamChunk]:
        """Yield lifecycle chunks with backpressure, then one terminal outcome.

        Acknowledging each delivered event before graph execution continues lets
        callers cancel at node boundaries. Closing this generator cancels and
        awaits its invocation task. All queues and observers belong to this run.
        """
        workflow_id = uuid.uuid4().hex[:8]
        ctx = _ExecutorStreamContext(workflow_id, len(workflow.nodes))
        self._active_workflows[workflow_id] = ctx
        queue: asyncio.Queue[Any] = asyncio.Queue()
        finished = object()
        result: Optional[WorkflowResult] = None
        error: Optional[str] = None

        async def observe(
            kind: str, node_id: str, state: Any, message: Optional[str], duration: float
        ) -> None:
            node = workflow.nodes[node_id]
            event_type = {
                "start": WorkflowEventType.NODE_START,
                "complete": WorkflowEventType.NODE_COMPLETE,
                "error": WorkflowEventType.NODE_ERROR,
            }[kind]
            metadata: Dict[str, Any] = {"node_type": node.node_type.value}
            if kind != "start":
                ctx.completed_nodes += 1
                metadata["duration_seconds"] = duration
                # Parallel groups emit one lifecycle event while their leaf
                # executors own usage records. Count each leaf once, excluding
                # aggregate group records and unrelated earlier nodes.
                pending = [node_id]
                visited = set()
                tool_calls = 0
                while pending:
                    result_id = pending.pop()
                    if result_id in visited:
                        continue
                    visited.add(result_id)
                    result_node = workflow.nodes[result_id]
                    if isinstance(result_node, ParallelNode):
                        pending.extend(result_node.parallel_nodes)
                        continue
                    node_result = state.get("_node_results", {}).get(result_id)
                    tool_calls += (
                        node_result.get("tool_calls_used", 0)
                        if isinstance(node_result, dict)
                        else getattr(node_result, "tool_calls_used", 0)
                    )
                metadata["tool_calls_used"] = tool_calls
            chunk = WorkflowStreamChunk(
                event_type=event_type,
                workflow_id=workflow_id,
                node_id=node_id,
                node_name=node.name,
                progress=ctx.progress,
                error=message,
                metadata=metadata,
            )
            acknowledged = asyncio.get_running_loop().create_future()
            queue.put_nowait((chunk, acknowledged))
            await acknowledged

        try:
            start = WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_START,
                workflow_id=workflow_id,
                progress=0.0,
                metadata={"workflow_name": workflow.name, "total_nodes": ctx.total_nodes},
            )
            self._notify_subscribers(start)
            yield start
            if ctx.is_cancelled:
                error = "Workflow cancelled"
            else:
                ctx.task = asyncio.create_task(
                    self._runtime.execute(
                        workflow,
                        initial_context=initial_context,
                        timeout=timeout,
                        thread_id=thread_id or workflow_id,
                        node_observer=observe,
                    )
                )
                # A done callback also fires if cancellation happens before the
                # task starts, unlike a runner's finally block.
                ctx.task.add_done_callback(lambda task: queue.put_nowait(finished))
                while True:
                    item = await queue.get()
                    if item is finished or ctx.is_cancelled:
                        break
                    chunk, acknowledged = item
                    self._notify_subscribers(chunk)
                    yield chunk
                    if not acknowledged.done():
                        acknowledged.set_result(None)
                try:
                    result = await ctx.task
                except asyncio.CancelledError:
                    if not ctx.is_cancelled:
                        raise
                    error = "Workflow cancelled"
        except Exception as exc:
            logger.error("Workflow '%s' failed: %s", workflow.name, exc)
            error = str(exc)
        finally:
            if ctx.task is not None:
                if not ctx.task.done():
                    ctx.task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await ctx.task
            self._active_workflows.pop(workflow_id, None)

        interrupted = result.interrupted if result is not None else False
        success = result.success if result is not None else False
        error = error or (result.error if result is not None else None)
        if ctx.is_cancelled:
            success = False
            error = "Workflow cancelled"
        final = WorkflowStreamChunk(
            event_type=(
                WorkflowEventType.WORKFLOW_PAUSED
                if interrupted
                else (
                    WorkflowEventType.WORKFLOW_COMPLETE
                    if success
                    else WorkflowEventType.WORKFLOW_ERROR
                )
            ),
            workflow_id=workflow_id,
            progress=100.0 if success and not interrupted else ctx.progress,
            is_final=True,
            error=error,
            metadata={
                "workflow_name": workflow.name,
                "total_duration": time.time() - ctx.start_time,
                "success": success,
                "interrupted": interrupted,
                "interrupt_node": result.interrupt_node if result is not None else None,
                "status": "paused" if interrupted else "completed" if success else "failed",
            },
        )
        self._notify_subscribers(final)
        yield final

    def subscribe(
        self, event_types: List[WorkflowEventType], callback: Callable[[WorkflowStreamChunk], None]
    ) -> Callable[[], None]:
        subscription = _Subscription(event_types, callback)
        self._subscriptions.append(subscription)

        def unsubscribe() -> None:
            subscription.active = False
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)

        return unsubscribe

    def _notify_subscribers(self, chunk: WorkflowStreamChunk) -> None:
        for subscription in list(self._subscriptions):
            if subscription.active and chunk.event_type in subscription.event_types:
                try:
                    subscription.callback(chunk)
                except Exception as exc:
                    logger.warning("Subscriber callback failed: %s", exc)

    def cancel_workflow(self, workflow_id: str) -> bool:
        ctx = self._active_workflows.get(workflow_id)
        if ctx is None:
            return False
        ctx.is_cancelled = True
        if ctx.task is not None:
            ctx.task.cancel()
        return True

    def get_active_workflows(self) -> List[str]:
        return list(self._active_workflows)

    def get_workflow_progress(self, workflow_id: str) -> Optional[float]:
        ctx = self._active_workflows.get(workflow_id)
        return ctx.progress if ctx is not None else None


__all__ = [
    "WorkflowEventType",
    "WorkflowStreamChunk",
    "WorkflowStreamContext",
    "StreamingWorkflowExecutor",
]
