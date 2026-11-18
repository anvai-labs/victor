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

"""Agentic loop graph executor - executes StateGraph-based agentic loop.

This module provides AgenticLoopGraphExecutor which runs the agentic
loop graph with proper service injection and result handling.
"""

from __future__ import annotations

from victor.core.errors import ProviderPolicyError, is_terminal_agent_error

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, TYPE_CHECKING

from victor.framework.agentic_graph.state import (
    AgenticLoopStateModel,
    create_initial_state,
)
from victor.framework.agentic_graph.builder import (
    AgenticLoopDependencies,
    create_agentic_loop_graph,
)
from victor.framework.agentic_graph.turn_record import TurnRecord, TurnRecordChannel
from victor.framework.execution_checkpoint import execution_checkpoint_trace_metadata

if TYPE_CHECKING:
    from victor.agent.services.turn_execution_runtime import TurnExecutor
    from victor.framework.graph import CompiledGraph

logger = logging.getLogger(__name__)


@dataclass
class LoopResult:
    """Result from agentic loop execution.

    Attributes:
        success: Whether execution completed successfully
        response: Final response content
        iterations: Number of iterations executed
        termination_reason: Why execution terminated
        metadata: Additional execution metadata
        error: Error message if execution failed
    """

    success: bool
    response: Optional[str]
    iterations: int
    termination_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def paused(self) -> bool:
        """The run stopped at a graph boundary and can resume (FEP-0032/FOU-D4)."""
        info = self.metadata.get("paused")
        return isinstance(info, dict) and bool(info.get("thread_id"))

    @classmethod
    def from_graph_result(
        cls,
        graph_result: Any,
        final_state: AgenticLoopStateModel,
        thread_id: Optional[str] = None,
    ) -> "LoopResult":
        """Create LoopResult from graph execution result.

        Args:
            graph_result: Raw graph execution result
            final_state: Final state after execution
            thread_id: The executor-owned per-run thread id; when the run
                stopped at a graph boundary (FEP-0032 ``interrupted``), the
                paused position is surfaced under ``metadata["paused"]`` so
                callers can resume via ``compiled.resume(thread_id)``. A
                pause without a thread id is not surfaced (unresumable).

        Returns:
            LoopResult instance
        """
        # Extract response from final state
        response = None
        action_result = final_state.action_result
        if isinstance(action_result, TurnRecord):
            response = action_result.content
        elif isinstance(action_result, dict):
            response = action_result.get("response")

        # Determine termination reason
        termination_reason = "unknown"
        if final_state.evaluation:
            decision = final_state.evaluation.get("decision", "")
            if decision == "complete":
                termination_reason = "complete"
            elif decision == "fail":
                termination_reason = "failed"
            elif final_state.iteration >= final_state.max_iterations:
                termination_reason = "max_iterations"

        metadata = {"final_state": final_state.to_dict()}
        plan_execution_state = _extract_plan_execution_state(final_state)
        if plan_execution_state:
            metadata["plan_execution_state"] = plan_execution_state
        checkpoint_metadata = _extract_execution_checkpoint_trace_metadata(final_state)
        if checkpoint_metadata:
            metadata["execution_checkpoint"] = checkpoint_metadata

        state_history = getattr(graph_result, "state_history", None)
        if state_history:
            metadata["state_history"] = [
                {
                    "node_name": node_name,
                    "state": state.to_dict() if hasattr(state, "to_dict") else state,
                }
                for node_name, state in state_history
            ]

        interrupted = bool(getattr(graph_result, "interrupted", False))
        if interrupted and thread_id:
            metadata["paused"] = {
                "thread_id": thread_id,
                "next_node": getattr(graph_result, "next_node", None),
            }

        return cls(
            success=termination_reason != "failed",
            response=response,
            iterations=final_state.iteration,
            termination_reason=termination_reason,
            metadata=metadata,
        )


def _turn_record_observer(
    channel: Optional[TurnRecordChannel],
) -> Any:
    """Publish completed ACT records without putting a live sink in graph state."""

    async def observe(
        event: str,
        node_id: str,
        state: Any,
        error: Optional[str],
        duration: float,
    ) -> None:
        if event != "complete" or node_id != "act" or channel is None:
            return
        record = getattr(state, "action_result", None)
        if isinstance(record, TurnRecord):
            channel.publish(record)

    return observe


def _resolve_default_checkpointer():
    """Durable-by-default graph checkpointing (FEP-0047, FOU-D2).

    Composition-root seam: resolves the settings-driven checkpointer from the
    storage layer. The import is local so importing this module never pulls
    storage infrastructure; any unexpected resolution failure warns and keeps
    the framework's in-memory default rather than failing agent construction.
    """
    from victor.storage.checkpoints.graph_adapter import (
        get_default_graph_checkpointer,
    )

    return get_default_graph_checkpointer()


class AgenticLoopGraphExecutor:
    """Executes the agentic loop using StateGraph.

    This executor manages:
    - Graph creation and compilation
    - Service injection into nodes
    - Execution lifecycle (run, stream)
    - Result aggregation and formatting

    Example:
        executor = AgenticLoopGraphExecutor(
            execution_context=context,
            max_iterations=10,
        )
        result = await executor.run("Write tests")

    Note:
        The executor always constructs a canonical ``AgenticLoopStateModel`` via
        ``create_initial_state()`` before invoking the compiled graph. Raw-dict
        state compatibility is primarily for direct ``compiled.invoke(...)``
        callers outside this executor facade.
    """

    def __init__(
        self,
        execution_context: Any,
        max_iterations: int = 10,
        enable_fulfillment: bool = True,
        enable_adaptive_iterations: bool = True,
    ):
        """Initialize the executor.

        Args:
            execution_context: RuntimeExecutionContext for service injection
            max_iterations: Maximum number of loop iterations
            enable_fulfillment: Whether to enable fulfillment checks
            enable_adaptive_iterations: Whether to enable adaptive termination
        """
        self.execution_context = execution_context
        self.max_iterations = max_iterations
        self.enable_fulfillment = enable_fulfillment
        self.enable_adaptive_iterations = enable_adaptive_iterations

        # Services (injected lazily or set externally)
        self.runtime_intelligence: Optional[Any] = None
        self.planning_coordinator: Optional[Any] = None
        self.turn_executor: Optional[TurnExecutor] = None
        self.evaluator: Optional[Any] = None
        self.fulfillment_detector: Optional[Any] = None
        self.dependencies = AgenticLoopDependencies(
            resolvers={
                "runtime_intelligence": lambda: self.runtime_intelligence,
                "execution_context": lambda: self.execution_context,
                "planning_coordinator": lambda: self.planning_coordinator,
                "use_llm_planning": lambda: self.planning_coordinator is not None,
                "turn_executor": lambda: self.turn_executor,
                "evaluator": lambda: self.evaluator,
                "fulfillment_detector": lambda: self.fulfillment_detector,
            }
        )

        # Create and compile graph
        self.graph = create_agentic_loop_graph(
            max_iterations=max_iterations,
            enable_fulfillment=enable_fulfillment,
            enable_adaptive_iterations=enable_adaptive_iterations,
            include_prompt_node=True,
            dependencies=self.dependencies,
        )
        # record_state_history=True: this executor reconstructs legacy
        # per-iteration LoopResult data from GraphExecutionResult.state_history
        # (see LoopResult.from_graph_result below) — the one production
        # consumer of state_history, which is opt-in elsewhere (item 11).
        self.compiled: CompiledGraph = self.graph.compile(
            record_state_history=True,
            checkpointer=_resolve_default_checkpointer(),
            checkpoint_after_nodes={"act"},
        )

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        *,
        turn_record_channel: Optional[TurnRecordChannel] = None,
    ) -> LoopResult:
        """Run the agentic loop to completion.

        Args:
            query: User's query or task
            context: Optional additional context
            conversation_history: Optional prior conversation turns

        Returns:
            LoopResult with execution outcome
        """
        if self.turn_executor is not None:
            self.turn_executor.require_supported_loop_executor(stategraph=True)

        # Create initial state
        initial_state = create_initial_state(
            query=query,
            context=context or {},
            max_iterations=self.max_iterations,
            conversation_history=conversation_history,
        )

        # Runtime services and the run-owned record sink stay outside durable
        # state. They are injected through collaborators and lifecycle hooks.
        observer = _turn_record_observer(turn_record_channel)

        # Per-run thread identity (FOU-D2 ledger note): each run gets a fresh
        # id so durable checkpoints never hijack a later run's input, and the
        # paused result can name its resume point (FOU-D4).
        run_thread_id = uuid.uuid4().hex
        try:
            # Execute graph
            graph_result = await self.compiled.invoke(
                initial_state,
                thread_id=run_thread_id,
                node_observer=observer,
            )

            # Extract final state
            final_state = self._extract_final_state(graph_result)

            # Create result
            result = LoopResult.from_graph_result(
                graph_result, final_state, thread_id=run_thread_id
            )

            logger.info(
                f"Agentic loop completed: {result.iterations} iterations, "
                f"termination={result.termination_reason}"
            )

            return result

        except ProviderPolicyError:
            raise
        except Exception as e:
            if is_terminal_agent_error(e):
                raise
            logger.error(f"Agentic loop execution failed: {e}")
            return LoopResult(
                success=False,
                response=None,
                iterations=0,
                termination_reason="error",
                error=str(e),
            )

    async def stream(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        *,
        turn_record_channel: Optional[TurnRecordChannel] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream agentic loop execution events.

        Args:
            query: User's query or task
            context: Optional additional context
            conversation_history: Optional prior conversation turns

        Yields:
            Event dictionaries with node_name, state, and event_type
        """
        if self.turn_executor is not None:
            self.turn_executor.require_supported_loop_executor(stategraph=True)

        # Create initial state
        initial_state = create_initial_state(
            query=query,
            context=context or {},
            max_iterations=self.max_iterations,
            conversation_history=conversation_history,
        )

        observer = _turn_record_observer(turn_record_channel)

        try:
            # Stream graph execution
            async for node_name, state in self.compiled.stream(
                initial_state,
                node_observer=observer,
            ):
                event = {
                    "node_name": node_name,
                    "state": state.to_dict() if hasattr(state, "to_dict") else state,
                    "event_type": "node_complete",
                }
                plan_execution_state = _extract_plan_execution_state(state)
                if plan_execution_state:
                    event["plan_execution_state"] = plan_execution_state
                checkpoint_metadata = _extract_execution_checkpoint_trace_metadata(state)
                if checkpoint_metadata:
                    event["execution_checkpoint"] = checkpoint_metadata
                yield event

        except ProviderPolicyError:
            raise
        except Exception as e:
            if is_terminal_agent_error(e):
                raise
            yield {
                "node_name": "error",
                "state": {},
                "event_type": "error",
                "error": str(e),
            }

    def _extract_final_state(self, graph_result: Any) -> AgenticLoopStateModel:
        """Extract final state from graph result.

        Args:
            graph_result: Raw graph execution result

        Returns:
            Final AgenticLoopStateModel
        """
        # Handle different graph result formats
        if hasattr(graph_result, "state"):
            return graph_result.state
        elif isinstance(graph_result, dict):
            return AgenticLoopStateModel(**graph_result)
        else:
            # Assume it's already a state model
            return graph_result

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with execution metrics
        """
        graph_nodes = list(getattr(self.graph, "node_ids", []))

        return {
            "max_iterations": self.max_iterations,
            "enable_fulfillment": self.enable_fulfillment,
            "enable_adaptive_iterations": self.enable_adaptive_iterations,
            "graph_nodes": graph_nodes,
        }


def _extract_execution_checkpoint_trace_metadata(
    state: Any,
) -> Optional[Dict[str, Any]]:
    """Return trace-safe execution checkpoint metadata carried by graph state."""
    context = _state_context(state)
    if not context:
        return None

    checkpoint = context.get("execution_checkpoint")
    if checkpoint is None:
        checkpoint = context.get("execution_checkpoint_metadata")
    if checkpoint is None:
        return None

    return execution_checkpoint_trace_metadata(checkpoint)


def _extract_plan_execution_state(state: Any) -> Optional[Dict[str, Any]]:
    """Return checkpointable plan execution state carried by graph state."""
    if isinstance(state, Mapping):
        plan_execution_state = state.get("plan_execution_state")
    else:
        plan_execution_state = getattr(state, "plan_execution_state", None)
    return plan_execution_state if isinstance(plan_execution_state, dict) else None


def _state_context(state: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(state, Mapping):
        context = state.get("context")
    else:
        context = getattr(state, "context", None)
    return context if isinstance(context, Mapping) else None
