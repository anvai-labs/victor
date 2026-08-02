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

"""Execution collaborators for compiled StateGraph workflows."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

from pydantic import BaseModel

from victor.framework.execution_checkpoint import normalize_execution_checkpoint_context
from victor.framework.graph_checkpoint import CheckpointerProtocol, WorkflowCheckpoint
from victor.framework.graph_state import CopyOnWriteState

logger = logging.getLogger(__name__)

StateType = TypeVar("StateType")


@dataclass
class GraphExecutionResult(Generic[StateType]):
    """Result from graph execution."""

    state: StateType
    success: bool
    error: Optional[str] = None
    iterations: int = 0
    duration: float = 0.0
    node_history: List[str] = field(default_factory=list)
    state_history: List[Tuple[str, Any]] = field(default_factory=list)


def snapshot_state_for_result(state: Any) -> Any:
    """Create a safe state snapshot for execution traces."""
    if isinstance(state, BaseModel):
        return state.model_copy(deep=True)
    return copy.deepcopy(state)


class IterationController:
    """Controls graph iteration and recursion tracking."""

    def __init__(self, max_iterations: int, recursion_limit: int):
        self.max_iterations = max_iterations
        self.recursion_limit = recursion_limit
        self.iterations = 0
        self.visited_count: Dict[str, int] = {}

    def should_continue(self, current_node: str) -> tuple[bool, Optional[str]]:
        self.iterations += 1
        if self.iterations > self.max_iterations:
            return False, f"Max iterations ({self.max_iterations}) exceeded"

        self.visited_count[current_node] = self.visited_count.get(current_node, 0) + 1
        if self.visited_count[current_node] > self.recursion_limit:
            return False, f"Recursion limit exceeded at node: {current_node}"

        return True, None

    def reset(self):
        self.iterations = 0
        self.visited_count.clear()


class TimeoutManager:
    """Tracks elapsed time and enforces execution timeouts."""

    def __init__(self, timeout: Optional[float]):
        self.timeout = timeout
        self.start_time: Optional[float] = None

    def start(self):
        self.start_time = time.time()

    def get_remaining(self) -> Optional[float]:
        if self.timeout is None or self.start_time is None:
            return None
        return self.timeout - (time.time() - self.start_time)

    def is_expired(self) -> bool:
        remaining = self.get_remaining()
        return remaining is not None and remaining <= 0

    def get_elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


class InterruptHandler:
    """Handles graph interrupts for human-in-the-loop workflows."""

    def __init__(self, interrupt_before: List[str], interrupt_after: List[str]):
        self.interrupt_before = set(interrupt_before)
        self.interrupt_after = set(interrupt_after)

    def should_interrupt_before(self, node_id: str) -> bool:
        return node_id in self.interrupt_before

    def should_interrupt_after(self, node_id: str) -> bool:
        return node_id in self.interrupt_after


@dataclass
class NodeExecutionResult:
    """Typed result from a single node execution.

    Preserves the original exception (``error``) so callers can dispatch on
    ``error_type`` (e.g. "TimeoutError" vs "ValidationError") without catching
    and re-raising.  ``error_message`` always holds ``str(error)`` for logging.
    """

    success: bool
    state: Any
    error: Optional[BaseException] = None
    error_type: str = ""
    error_message: str = ""

    @classmethod
    def ok(cls, state: Any) -> "NodeExecutionResult":
        return cls(success=True, state=state)

    @classmethod
    def fail(cls, error: BaseException, state: Any) -> "NodeExecutionResult":
        # str(asyncio.TimeoutError()) is "" on Python 3.11+; fall back to type name
        msg = str(error) or type(error).__name__
        return cls(
            success=False,
            state=state,
            error=error,
            error_type=type(error).__name__,
            error_message=msg,
        )


class NodeExecutor:
    """Executes graph nodes with timeout and copy-on-write support."""

    def __init__(self, nodes: Dict[str, Any], use_copy_on_write: bool):
        self.nodes = nodes
        self.use_copy_on_write = use_copy_on_write

    async def execute(
        self,
        node_id: str,
        state: StateType,
        timeout_manager: TimeoutManager,
    ) -> tuple[bool, Optional[str], StateType]:
        if not self.nodes.get(node_id):
            return False, f"Node not found: {node_id}", state
        typed = await self.execute_typed(node_id, state, timeout_manager)
        return typed.success, typed.error_message or None, typed.state

    async def execute_typed(
        self,
        node_id: str,
        state: Any,
        timeout_manager: "TimeoutManager",
    ) -> NodeExecutionResult:
        """Execute a node and return a typed NodeExecutionResult.

        Preserves the original exception instance and its type name so callers
        can dispatch on error category without string parsing.
        """
        node = self.nodes.get(node_id)
        if not node:
            return NodeExecutionResult.fail(
                error=LookupError(f"Node not found: {node_id}"), state=state
            )

        try:
            if timeout_manager.is_expired():
                return NodeExecutionResult.fail(
                    error=asyncio.TimeoutError("Execution timeout"), state=state
                )

            remaining = timeout_manager.get_remaining()
            use_copy_on_write = self.use_copy_on_write and not isinstance(state, BaseModel)

            if use_copy_on_write:
                cow_state: CopyOnWriteState[Any] = CopyOnWriteState(state)
                if remaining is not None:
                    result = await asyncio.wait_for(
                        node.execute(cow_state), timeout=remaining  # type: ignore[arg-type]
                    )
                else:
                    result = await node.execute(cow_state)  # type: ignore[arg-type]

                if isinstance(result, CopyOnWriteState):
                    state = result.get_state()
                elif isinstance(result, (BaseModel, dict)):
                    state = result
                else:
                    state = cow_state.get_state()
            else:
                if remaining is not None:
                    state = await asyncio.wait_for(node.execute(state), timeout=remaining)
                else:
                    state = await node.execute(state)

            return NodeExecutionResult.ok(state=state)
        except asyncio.TimeoutError as exc:
            return NodeExecutionResult.fail(error=exc, state=state)
        except Exception as exc:
            return NodeExecutionResult.fail(error=exc, state=state)


class GraphCheckpointManager:
    """Loads and saves graph checkpoints around execution."""

    def __init__(self, checkpointer: Optional[CheckpointerProtocol]):
        self.checkpointer = checkpointer

    async def load_initial_state(
        self,
        thread_id: str,
        input_state: StateType,
        entry_point: str,
    ) -> tuple[StateType, str]:
        if self.checkpointer:
            checkpoint = await self.checkpointer.load(thread_id)
            if checkpoint:
                logger.info("Resuming from checkpoint at node: %s", checkpoint.node_id)
                return checkpoint.state.copy(), checkpoint.node_id
        return copy.deepcopy(input_state), entry_point

    async def save_checkpoint(
        self,
        thread_id: str,
        node_id: str,
        state: StateType,
    ) -> None:
        if self.checkpointer:
            checkpoint_id = f"{thread_id}_{node_id}_{time.time()}"
            self._bind_graph_checkpoint_id(state, checkpoint_id)
            checkpoint = WorkflowCheckpoint(
                checkpoint_id=checkpoint_id,
                thread_id=thread_id,
                node_id=node_id,
                state=state,
                timestamp=time.time(),
                metadata=self._checkpoint_metadata(state),
            )
            await self.checkpointer.save(checkpoint)

    @staticmethod
    def _bind_graph_checkpoint_id(state: StateType, checkpoint_id: str) -> None:
        if isinstance(state, dict):
            context = state.get("context")
            if isinstance(context, dict):
                state["context"] = normalize_execution_checkpoint_context(
                    context,
                    graph_checkpoint_id=checkpoint_id,
                )
            return

        context = getattr(state, "context", None)
        if isinstance(context, dict):
            try:
                state.context = normalize_execution_checkpoint_context(
                    context,
                    graph_checkpoint_id=checkpoint_id,
                )
            except Exception:
                logger.debug(
                    "Unable to attach graph checkpoint id to state context",
                    exc_info=True,
                )

    @staticmethod
    def _checkpoint_metadata(state: StateType) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if isinstance(state, dict):
            plan_execution_state = state.get("plan_execution_state")
        else:
            plan_execution_state = getattr(state, "plan_execution_state", None)
        if isinstance(plan_execution_state, dict) and plan_execution_state:
            metadata["plan_execution_state"] = plan_execution_state
        return metadata


class GraphEventEmitter:
    """Emits graph execution events for observability."""

    def __init__(self, graph_id: str, emit_events: bool):
        self.graph_id = graph_id
        self.emit_events = emit_events

    def _emit(self, event_name: str, payload: Dict[str, Any]) -> None:
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            get_event_bus().emit_lifecycle_event(
                event_name,
                {
                    "graph_id": self.graph_id,
                    "source": "StateGraph",
                    **payload,
                },
            )
        except Exception as error:
            logger.warning("Failed to emit %s event: %s", event_name, error)

    def emit_graph_started(self, entry_point: str, node_count: int, thread_id: str):
        self._emit(
            "graph_started",
            {
                "entry_point": entry_point,
                "node_count": node_count,
                "thread_id": thread_id,
            },
        )

    def emit_node_start(self, node_id: str, iteration: int):
        self._emit(
            "node_start",
            {
                "node_id": node_id,
                "iteration": iteration,
            },
        )

    def emit_node_complete(self, node_id: str, iteration: int, duration: float):
        self._emit(
            "node_end",
            {
                "node_id": node_id,
                "iteration": iteration,
                "duration": duration,
                "success": True,
            },
        )

    def emit_graph_completed(
        self, success: bool, iterations: int, duration: float, node_count: int
    ):
        self._emit_rl_completion(
            success=success,
            iterations=iterations,
            duration=duration,
        )
        self._emit(
            "graph_completed",
            {
                "success": success,
                "iterations": iterations,
                "duration": duration,
                "node_count": node_count,
            },
        )

    def _emit_rl_completion(self, success: bool, iterations: int, duration: float) -> None:
        try:
            from victor.framework.rl.hooks import RLEvent, RLEventType, get_rl_hooks

            hooks = get_rl_hooks()
            if hooks is None:
                return

            quality = 0.8 if success else 0.2
            if success and iterations < 10:
                quality += 0.1
            if success and duration < 30:
                quality += 0.1

            hooks.emit(
                RLEvent(
                    type=RLEventType.WORKFLOW_COMPLETED,
                    workflow_name="state_graph",
                    success=success,
                    quality_score=min(1.0, quality),
                    metadata={
                        "iterations": iterations,
                        "duration_seconds": duration,
                        "graph_type": "state_graph",
                    },
                )
            )
        except Exception as error:
            logger.debug("Graph RL completion emission failed: %s", error)

    def emit_graph_error(self, error: str, iterations: int, duration: float):
        self._emit(
            "graph_error",
            {
                "error": error,
                "iterations": iterations,
                "duration": duration,
            },
        )
