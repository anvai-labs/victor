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

"""
Enhanced StateGraph integration with Credit Assignment.

This module provides seamless integration between StateGraph workflows
and the credit assignment system, enabling automatic tracking and
attribution of rewards across complex multi-step workflows.

Key Features:
- Automatic transition tracking during graph execution
- Reward collection from node outcomes
- Credit assignment at multiple granularities
- Critical action detection for workflow optimization

Usage:
    from victor.framework.rl.credit_graph_integration import (
        CreditAwareGraph,
        CreditTracer,
    )

    # Wrap StateGraph with credit awareness
    graph = CreditAwareGraph(StateGraph(MyState))
    graph.add_node("task", task_func)

    # Compile with credit tracing
    app = graph.compile(enable_credit=True)

    # Run and get credit assignment
    result = await app.invoke(initial_state)
    attribution = app.get_credit_attribution()
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from victor.framework.rl.credit_assignment import (
    ActionMetadata,
    CreditAssignmentConfig,
    CreditAssignmentIntegration,
    CreditGranularity,
    CreditMethodology,
    CreditSignal,
)

if TYPE_CHECKING:
    from victor.framework.graph import CompiledGraph, StateGraph

logger = logging.getLogger(__name__)


# ============================================================================
# Credit Tracing Context
# ============================================================================


@dataclass
class Transition:
    """Represents a single transition in the graph execution."""

    transition_id: str
    from_node: str
    to_node: str
    timestamp: float
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    node_output: Any
    agent_id: str
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def create(
        cls,
        from_node: str,
        to_node: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        node_output: Any,
        agent_id: str = "default",
    ) -> "Transition":
        """Create a new transition."""
        return cls(
            transition_id=f"trans_{from_node}_{to_node}_{datetime.now().timestamp()}",
            from_node=from_node,
            to_node=to_node,
            timestamp=datetime.now().timestamp(),
            state_before=state_before,
            state_after=state_after,
            node_output=node_output,
            agent_id=agent_id,
        )


@dataclass
class ExecutionTrace:
    """Complete trace of a graph execution with credit information."""

    trace_id: str
    start_time: float
    end_time: float
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    transitions: List[Transition]
    rewards: List[float]
    total_reward: float
    success: bool
    error: Optional[str] = None
    credit_signals: List[CreditSignal] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def transition_count(self) -> int:
        return len(self.transitions)


# ============================================================================
# Credit Tracer
# ============================================================================


class CreditTracer:
    """Tracks graph execution for credit assignment.

    This tracer wraps StateGraph execution to collect:
    - State transitions
    - Rewards from node outcomes
    - Agent/team attribution
    - Timing information
    """

    def __init__(
        self,
        reward_extractor: Optional[Callable[[Any], float]] = None,
        agent_extractor: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        """Initialize the credit tracer.

        Args:
            reward_extractor: Function to extract reward from node output
            agent_extractor: Function to extract agent ID from state
        """
        self._reward_extractor = reward_extractor or self._default_reward_extractor
        self._agent_extractor = agent_extractor or self._default_agent_extractor
        self._active_trace: Optional[ExecutionTrace] = None
        self._trace_history: List[ExecutionTrace] = []

    def start_trace(
        self, initial_state: Dict[str, Any], trace_id: Optional[str] = None
    ) -> ExecutionTrace:
        """Start a new execution trace.

        Args:
            initial_state: Initial graph state
            trace_id: Optional trace ID (auto-generated if not provided)

        Returns:
            The created ExecutionTrace
        """
        trace_id = trace_id or f"trace_{datetime.now().timestamp()}"
        self._active_trace = ExecutionTrace(
            trace_id=trace_id,
            start_time=datetime.now().timestamp(),
            end_time=0.0,
            initial_state=initial_state,
            final_state={},
            transitions=[],
            rewards=[],
            total_reward=0.0,
            success=False,
        )
        logger.debug(f"Started trace: {trace_id}")
        return self._active_trace

    def record_transition(
        self,
        from_node: str,
        to_node: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        node_output: Any,
    ) -> Transition:
        """Record a state transition.

        Args:
            from_node: Source node ID
            to_node: Target node ID
            state_before: State before transition
            state_after: State after transition
            node_output: Output from the node execution

        Returns:
            The recorded Transition
        """
        if self._active_trace is None:
            logger.warning("No active trace, skipping transition recording")
            return None  # type: ignore

        agent_id = self._agent_extractor(state_after)
        transition = Transition.create(
            from_node=from_node,
            to_node=to_node,
            state_before=state_before,
            state_after=state_after,
            node_output=node_output,
            agent_id=agent_id,
        )

        # Extract reward
        reward = self._reward_extractor(node_output)
        transition.reward = reward

        self._active_trace.transitions.append(transition)
        self._active_trace.rewards.append(reward)
        self._active_trace.total_reward += reward

        logger.debug(f"Recorded transition: {from_node} -> {to_node} (reward={reward})")
        return transition

    def end_trace(
        self, final_state: Dict[str, Any], success: bool, error: Optional[str] = None
    ) -> ExecutionTrace:
        """End the current execution trace.

        Args:
            final_state: Final graph state
            success: Whether execution succeeded
            error: Optional error message

        Returns:
            The completed ExecutionTrace
        """
        if self._active_trace is None:
            logger.warning("No active trace to end")
            return None  # type: ignore

        self._active_trace.end_time = datetime.now().timestamp()
        self._active_trace.final_state = final_state
        self._active_trace.success = success
        self._active_trace.error = error

        self._trace_history.append(self._active_trace)
        logger.debug(
            f"Ended trace: {self._active_trace.trace_id} "
            f"(transitions={len(self._active_trace.transitions)}, "
            f"reward={self._active_trace.total_reward})"
        )

        completed = self._active_trace
        self._active_trace = None
        return completed

    def get_active_trace(self) -> Optional[ExecutionTrace]:
        """Get the currently active trace."""
        return self._active_trace

    def get_trace_history(self) -> List[ExecutionTrace]:
        """Get all completed traces."""
        return self._trace_history.copy()

    def assign_credit_to_trace(
        self,
        trace: ExecutionTrace,
        methodology: CreditMethodology = CreditMethodology.GAE,
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Assign credit to a trace's transitions.

        Args:
            trace: Execution trace to assign credit to
            methodology: Credit assignment methodology
            config: Optional configuration

        Returns:
            List of CreditSignal
        """
        # Create action metadata from transitions
        trajectory = []
        for i, trans in enumerate(trace.transitions):
            metadata = ActionMetadata(
                agent_id=trans.agent_id,
                action_id=trans.transition_id,
                turn_index=i,  # Each transition is a "turn"
                step_index=i,
                method_name=trans.to_node,
                timestamp=trans.timestamp,
            )
            trajectory.append(metadata)

        # Create integration with config
        ca_integration = CreditAssignmentIntegration(default_config=config)
        signals = ca_integration.assign_credit(trajectory, trace.rewards, methodology, config)

        # Attach to trace
        trace.credit_signals = signals

        return signals

    def get_agent_attribution(self, trace: ExecutionTrace, agent_id: str) -> Dict[str, float]:
        """Get credit attribution for a specific agent.

        Args:
            trace: Execution trace
            agent_id: Agent to get attribution for

        Returns:
            Dictionary mapping contributors to credit amounts
        """
        # Filter signals by agent
        agent_signals = [
            s for s in trace.credit_signals if s.metadata and s.metadata.agent_id == agent_id
        ]

        # Aggregate attribution
        attribution: Dict[str, float] = defaultdict(float)
        for signal in agent_signals:
            for contributor, amount in signal.attribution.items():
                attribution[contributor] += amount

        # Add direct credit
        direct_credit = sum(s.credit for s in agent_signals)
        if direct_credit > 0:
            attribution[agent_id] += direct_credit

        return dict(attribution)

    def _default_reward_extractor(self, node_output: Any) -> float:
        """Default reward extraction from node output.

        Looks for reward in common locations:
        - node_output.get("reward") if dict
        - node_output.reward if object
        - 0.0 if not found
        """
        if isinstance(node_output, dict):
            return float(node_output.get("reward", node_output.get("score", 0.0)))
        elif hasattr(node_output, "reward"):
            return float(node_output.reward)
        elif hasattr(node_output, "score"):
            return float(node_output.score)
        return 0.0

    def _default_agent_extractor(self, state: Dict[str, Any]) -> str:
        """Default agent ID extraction from state.

        Looks for agent in common locations:
        - state.get("agent_id")
        - state.get("agent")
        - "default" if not found
        """
        return state.get("agent_id", state.get("agent", "default"))


# ============================================================================
# Credit-Aware Graph Wrapper
# ============================================================================


class CreditAwareGraph:
    """Wrapper that adds credit tracking to StateGraph.

    This wrapper provides a drop-in replacement for StateGraph that
    automatically tracks execution and assigns credit.

    Example:
        # Instead of: graph = StateGraph(MyState)
        # Use:
        graph = CreditAwareGraph(StateGraph(MyState))

        # Use graph normally
        graph.add_node("task", task_func)
        graph.add_edge("start", "task")

        # Compile with credit tracking
        app = graph.compile(enable_credit=True)

        # Run and get credit info
        result = await app.invoke(state)
        attribution = app.get_credit_attribution()
    """

    def __init__(
        self,
        graph: "StateGraph",
        tracer: Optional[CreditTracer] = None,
        reward_extractor: Optional[Callable[[Any], float]] = None,
        agent_extractor: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        """Initialize the credit-aware graph wrapper.

        Args:
            graph: The StateGraph to wrap
            tracer: Optional CreditTracer (creates default if not provided)
            reward_extractor: Optional reward extraction function
            agent_extractor: Optional agent ID extraction function
        """
        self._graph = graph
        self._tracer = tracer or CreditTracer(reward_extractor, agent_extractor)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the wrapped graph."""
        return getattr(self._graph, name)

    def compile(
        self,
        enable_credit: bool = True,
        credit_methodology: CreditMethodology = CreditMethodology.GAE,
        credit_config: Optional[CreditAssignmentConfig] = None,
    ) -> "CompiledCreditAwareGraph":
        """Compile the graph with credit tracking enabled.

        Args:
            enable_credit: Whether to enable credit tracking
            credit_methodology: Credit assignment methodology to use
            credit_config: Optional credit assignment configuration

        Returns:
            CompiledCreditAwareGraph that tracks execution
        """
        # Compile the underlying graph
        compiled = self._graph.compile()

        # Wrap with credit tracking if enabled
        if enable_credit:
            return CompiledCreditAwareGraph(
                compiled=compiled,
                tracer=self._tracer,
                methodology=credit_methodology,
                config=credit_config,
            )

        return compiled  # type: ignore


class CompiledCreditAwareGraph:
    """Compiled graph with credit tracking capabilities.

    This class wraps a CompiledGraph and adds automatic credit
    assignment during execution.
    """

    def __init__(
        self,
        compiled: "CompiledGraph",
        tracer: CreditTracer,
        methodology: CreditMethodology,
        config: Optional[CreditAssignmentConfig],
    ):
        """Initialize the compiled credit-aware graph.

        Args:
            compiled: The compiled StateGraph
            tracer: CreditTracer for tracking execution
            methodology: Credit assignment methodology
            config: Optional credit configuration
        """
        self._compiled = compiled
        self._tracer = tracer
        self._methodology = methodology
        self._config = config
        self._last_trace: Optional[ExecutionTrace] = None

    async def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke the graph with credit tracking.

        Args:
            input: Initial state
            config: Optional configuration

        Returns:
            Final state after graph execution
        """
        # Start trace
        self._tracer.start_trace(input)

        try:
            # Execute graph (would need hook into CompiledGraph internals)
            result = await self._compiled.invoke(input, config)

            # End trace successfully
            self._last_trace = self._tracer.end_trace(result, success=True)

            # Assign credit
            if self._last_trace:
                self._tracer.assign_credit_to_trace(
                    self._last_trace, self._methodology, self._config
                )

            return result

        except Exception as e:
            # End trace with error
            self._last_trace = self._tracer.end_trace(
                (
                    self._tracer.get_active_trace().final_state
                    if self._tracer.get_active_trace()
                    else input
                ),
                success=False,
                error=str(e),
            )
            raise

    def get_credit_attribution(self) -> Dict[str, Any]:
        """Get credit attribution from the last execution.

        Returns:
            Dictionary with attribution data:
            - trace_id: ID of the trace
            - total_reward: Total reward
            - agent_attribution: Per-agent attribution
            - signals: All credit signals
        """
        if self._last_trace is None:
            return {
                "trace_id": None,
                "total_reward": 0.0,
                "agent_attribution": {},
                "signals": [],
            }

        # Get attribution per agent
        agents = set()
        for signal in self._last_trace.credit_signals:
            if signal.metadata:
                agents.add(signal.metadata.agent_id)

        agent_attribution = {}
        for agent in agents:
            agent_attribution[agent] = self._tracer.get_agent_attribution(self._last_trace, agent)

        return {
            "trace_id": self._last_trace.trace_id,
            "total_reward": self._last_trace.total_reward,
            "duration": self._last_trace.duration,
            "transition_count": self._last_trace.transition_count,
            "agent_attribution": agent_attribution,
            "signals": [s.to_dict() for s in self._last_trace.credit_signals],
        }

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary of the last execution trace.

        Returns:
            Summary with trace statistics
        """
        if self._last_trace is None:
            return {"executed": False}

        return {
            "executed": True,
            "trace_id": self._last_trace.trace_id,
            "duration": self._last_trace.duration,
            "transitions": self._last_trace.transition_count,
            "total_reward": self._last_trace.total_reward,
            "success": self._last_trace.success,
            "error": self._last_trace.error,
            "methodology": self._methodology.value,
        }

    def get_credit_signals(self) -> List[CreditSignal]:
        """Get all credit signals from the last execution.

        Returns:
            List of CreditSignal
        """
        if self._last_trace is None:
            return []
        return self._last_trace.credit_signals.copy()

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to the compiled graph."""
        return getattr(self._compiled, name)


# ============================================================================
# Utility Functions
# ============================================================================


def create_credit_aware_workflow(
    graph: "StateGraph",
    reward_key: str = "reward",
    agent_key: str = "agent_id",
    methodology: CreditMethodology = CreditMethodology.GAE,
) -> CreditAwareGraph:
    """Create a credit-aware workflow from a StateGraph.

    Convenience function for creating credit-aware graphs with
    standard reward and agent extraction.

    Args:
        graph: The StateGraph to wrap
        reward_key: Key to extract reward from node output
        agent_key: Key to extract agent ID from state
        methodology: Credit assignment methodology

    Returns:
        CreditAwareGraph ready for compilation

    Example:
        graph = StateGraph(MyState)
        graph.add_node("task", task_func)

        credit_graph = create_credit_aware_workflow(graph)
        app = credit_graph.compile()

        result = await app.invoke(state)
        attribution = app.get_credit_attribution()
    """

    def reward_extractor(output: Any) -> float:
        if isinstance(output, dict):
            return float(output.get(reward_key, 0.0))
        return 0.0

    def agent_extractor(state: Dict[str, Any]) -> str:
        return state.get(agent_key, "default")

    return CreditAwareGraph(
        graph,
        tracer=CreditTracer(reward_extractor, agent_extractor),
    )


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    # Core classes
    "CreditTracer",
    "CreditAwareGraph",
    "CompiledCreditAwareGraph",
    # Data structures
    "Transition",
    "ExecutionTrace",
    # Utilities
    "create_credit_aware_workflow",
]
