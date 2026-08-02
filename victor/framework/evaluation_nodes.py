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

"""Evaluation Nodes for StateGraph - Extension for agentic loop evaluation.

This module extends StateGraph with evaluation capabilities:
- EvaluationNode: Evaluates state and decides next action
- DecisionEdge: Routes to different nodes based on evaluation
- EvaluationCheckpoint: Saves state at evaluation points

Design Principle: Extend existing StateGraph, don't replace.

Based on research from:
- arXiv:2601.21268 - Meta-evaluation without ground truth
- arXiv:2510.13220 - EvoTest runtime adaptation

Example:
    from victor.framework.graph import StateGraph
    from victor.framework.evaluation_nodes import EvaluationNode, add_evaluation

    graph = StateGraph(AgentState)

    # Add evaluation checkpoint
    def evaluate_progress(state):
        return EvaluationResult(
            score=state.get("progress", 0.0),
            decision="continue" if state["progress"] < 1.0 else "complete"
        )

    graph = add_evaluation(
        graph,
        node_id="check_progress",
        evaluator=evaluate_progress,
        decision_edges={
            "continue": "next_step",
            "complete": "__end__",
        }
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.framework.graph import StateGraph, StateType


class EvaluationDecision(Enum):
    """Standard evaluation decisions for agentic loops."""

    CONTINUE = "continue"  # Continue to next node
    RETRY = "retry"  # Retry current node
    COMPLETE = "complete"  # Task complete, exit graph
    ESCALATE = "escalate"  # Escalate to human/higher level
    FAIL = "fail"  # Task failed, exit with error


@dataclass
class EvaluationResult:
    """Result from evaluating state at checkpoint.

    Attributes:
        decision: What to do next (CONTINUE, RETRY, COMPLETE, etc.)
        score: Confidence/quality score (0.0-1.0)
        reason: Human-readable explanation
        metrics: Additional metrics for learning
        metadata: Additional data
    """

    decision: Union[str, EvaluationDecision]
    score: float = 0.5
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_continue(self) -> bool:
        """Check if execution should continue."""
        return self.decision == EvaluationDecision.CONTINUE or self.decision == "continue"

    @property
    def should_retry(self) -> bool:
        """Check if should retry current step."""
        return self.decision == EvaluationDecision.RETRY or self.decision == "retry"

    @property
    def should_complete(self) -> bool:
        """Check if task is complete."""
        return self.decision == EvaluationDecision.COMPLETE or self.decision == "complete"

    @property
    def should_fail(self) -> bool:
        """Check if task failed."""
        return self.decision == EvaluationDecision.FAIL or self.decision == "fail"


# Type alias for evaluator functions
Evaluator = Callable[[Dict[str, Any]], Union[EvaluationResult, Awaitable[EvaluationResult]]]
"""Evaluator function signature.

Takes current state dict, returns EvaluationResult.
Can be sync or async.
"""


@dataclass
class EvaluationCheckpoint:
    """Checkpoint data for evaluation point.

    Attributes:
        checkpoint_id: Unique identifier
        node_id: Which evaluation node
        timestamp: When evaluation occurred
        state: State at evaluation time
        result: Evaluation result
        iteration: Loop iteration number
    """

    checkpoint_id: str
    node_id: str
    timestamp: datetime
    state: Dict[str, Any]
    result: EvaluationResult
    iteration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "state_keys": (
                list(self.state.to_dict().keys())
                if hasattr(self.state, "to_dict")
                else list(self.state.keys())
            ),
            "decision": str(self.result.decision),
            "score": self.result.score,
            "reason": self.result.reason,
            "iteration": self.iteration,
        }


class EvaluationNode:
    """Node that evaluates state and routes execution.

    This node type extends StateGraph with evaluation capabilities
    without modifying the core graph.py file.

    Example:
        def check_quality(state):
            errors = state.get("errors", [])
            if errors:
                return EvaluationResult(
                    decision="retry",
                    score=0.0,
                    reason=f"Found {len(errors)} errors"
                )
            return EvaluationResult(
                decision="continue",
                score=1.0,
                reason="No errors found"
            )

        node = EvaluationNode(
            id="quality_check",
            evaluator=check_quality,
            decision_edges={
                "retry": "fix_errors",
                "continue": "next_step",
            }
        )
    """

    def __init__(
        self,
        id: str,
        evaluator: Evaluator,
        decision_edges: Dict[str, str],
        checkpoint: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize evaluation node.

        Args:
            id: Unique node identifier
            evaluator: Function that evaluates state and returns decision
            decision_edges: Mapping from decision strings to target node IDs
            checkpoint: Whether to save checkpoint at this node
            metadata: Additional node metadata
        """
        self.id = id
        self.evaluator = evaluator
        self.decision_edges = decision_edges
        self.checkpoint = checkpoint
        self.metadata = metadata or {}

    async def evaluate(self, state: Dict[str, Any]) -> EvaluationResult:
        """Evaluate state and return decision.

        Args:
            state: Current state dictionary

        Returns:
            EvaluationResult with decision and metadata
        """
        # Call evaluator (sync or async)
        import asyncio

        result = self.evaluator(state)
        if asyncio.iscoroutine(result):
            result = await result

        return result

    def get_next_node(self, result: EvaluationResult) -> str:
        """Get next node ID based on evaluation result.

        Handles both string decisions ("continue") and enum decisions
        (EvaluationDecision.CONTINUE).

        Args:
            result: Evaluation result from evaluator

        Returns:
            Next node ID to execute

        Raises:
            ValueError: If decision has no mapped edge
        """
        decision = result.decision

        # Try exact match first (works for both str and enum keys)
        decision_str = str(decision)
        if decision_str in self.decision_edges:
            return self.decision_edges[decision_str]

        # Try .value for enum decisions against string keys
        if hasattr(decision, "value") and decision.value in self.decision_edges:
            return self.decision_edges[decision.value]

        # Try matching string against enum keys
        for key in self.decision_edges:
            if hasattr(key, "value") and key.value == decision:
                return self.decision_edges[key]

        raise ValueError(
            f"Decision '{decision_str}' has no mapped edge. "
            f"Available decisions: {list(self.decision_edges.keys())}"
        )


# ============================================================================
# StateGraph Extension Functions
# ============================================================================


def add_evaluation(
    graph: "StateGraph",
    node_id: str,
    evaluator: Evaluator,
    decision_edges: Dict[str, str],
    checkpoint: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> "StateGraph":
    """Add evaluation node to existing StateGraph.

    This function extends StateGraph without modifying core code.

    Args:
        graph: Existing StateGraph instance
        node_id: Unique ID for evaluation node
        evaluator: Function that evaluates state
        decision_edges: Mapping from decisions to target nodes
        checkpoint: Whether to save checkpoint
        metadata: Additional metadata

    Returns:
        Modified StateGraph with evaluation node added

    Example:
        graph = StateGraph(AgentState)
        graph.add_node("analyze", analyze_fn)
        graph.add_node("execute", execute_fn)

        # Add evaluation checkpoint
        graph = add_evaluation(
            graph,
            node_id="check_result",
            evaluator=lambda s: EvaluationResult(
                decision="continue" if s.get("success") else "retry"
            ),
            decision_edges={
                "continue": "__end__",
                "retry": "analyze",
            }
        )

        graph.add_edge("execute", "check_result")
    """
    # Create evaluation node wrapper
    eval_node = EvaluationNode(
        id=node_id,
        evaluator=evaluator,
        decision_edges=decision_edges,
        checkpoint=checkpoint,
        metadata=metadata,
    )

    # Define node function that wraps evaluation
    async def evaluation_node_fn(state: Dict[str, Any]) -> Dict[str, Any]:
        # Evaluate state
        result = await eval_node.evaluate(state)

        # Store evaluation result in state
        state["_evaluation_result"] = result
        state["_evaluation_node_id"] = node_id

        # Add checkpoint if enabled
        if checkpoint:
            state.setdefault("_evaluation_checkpoints", []).append(
                EvaluationCheckpoint(
                    checkpoint_id=f"{node_id}_{datetime.now(tz=timezone.utc).timestamp()}",
                    node_id=node_id,
                    timestamp=datetime.now(tz=timezone.utc),
                    state=state.copy(),
                    result=result,
                    iteration=state.get("_iteration", 0),
                )
            )

        # The actual routing happens via conditional edges
        # This just stores the result for the edge function to use
        return state

    # Add node to graph
    graph.add_node(node_id, evaluation_node_fn)

    # Add conditional edge for routing
    def route_decision(state: Dict[str, Any]) -> str:
        """Route to next node based on evaluation decision."""
        result = state.get("_evaluation_result")
        if not result:
            logger.warning(f"No evaluation result found at {node_id}")
            return list(decision_edges.values())[0]  # Default to first edge

        return eval_node.get_next_node(result)

    # For each decision in decision_edges, add conditional routing
    # This is a simplified version - full implementation would integrate
    # with StateGraph's conditional edge system
    graph.add_conditional_edge(
        node_id,
        route_decision,
        decision_edges,
    )

    return graph


def create_agentic_loop_graph(
    state_type: type,
    perception_fn: Callable,
    planning_fn: Callable,
    execution_fn: Callable,
    evaluator_fn: Evaluator,
    max_iterations: int = 10,
) -> "StateGraph":
    """Create a complete agentic loop graph with evaluation.

    This creates a PERCEIVE → PLAN → ACT → EVALUATE → DECIDE loop
    with an iteration guard to enforce max_iterations.

    Args:
        state_type: Type annotation for state
        perception_fn: Perception function
        planning_fn: Planning function
        execution_fn: Execution function
        evaluator_fn: Evaluation function
        max_iterations: Maximum loop iterations

    Returns:
        Configured StateGraph ready to compile

    Example:
        graph = create_agentic_loop_graph(
            state_type=AgentState,
            perception_fn=perceive_task,
            planning_fn=create_plan,
            execution_fn=execute_plan,
            evaluator_fn=evaluate_result,
            max_iterations=5,
        )

        app = graph.compile()
        result = await app.invoke({"task": "Fix the bug"})
    """
    from victor.framework.graph import StateGraph, END

    graph = StateGraph(state_type)

    # Wrap perception with iteration counter
    async def perceive_with_iteration(state: Dict[str, Any]) -> Dict[str, Any]:
        state["_iteration"] = state.get("_iteration", 0) + 1
        import asyncio

        result = perception_fn(state)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    graph.add_node("perceive", perceive_with_iteration)
    graph.add_node("plan", planning_fn)
    graph.add_node("act", execution_fn)

    # Wrap evaluator to enforce max_iterations
    async def evaluator_with_guard(state: Dict[str, Any]) -> EvaluationResult:
        iteration = state.get("_iteration", 0)
        if iteration >= max_iterations:
            return EvaluationResult(
                decision=EvaluationDecision.FAIL,
                score=0.0,
                reason=f"Max iterations ({max_iterations}) exceeded",
            )

        import asyncio

        result = evaluator_fn(state)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    # Add evaluation node with routing
    graph = add_evaluation(
        graph,
        node_id="evaluate",
        evaluator=evaluator_with_guard,
        decision_edges={
            str(EvaluationDecision.CONTINUE): "perceive",
            str(EvaluationDecision.COMPLETE): END,
            str(EvaluationDecision.RETRY): "act",
            str(EvaluationDecision.FAIL): END,
        },
    )

    # Set entry point and connect edges
    graph.set_entry_point("perceive")
    graph.add_edge("perceive", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "evaluate")

    return graph


# ============================================================================
# Utility Functions
# ============================================================================


def simple_score_evaluator(
    threshold: float = 0.8,
    score_key: str = "score",
) -> Evaluator:
    """Create a simple threshold-based evaluator.

    Args:
        threshold: Score threshold for continuing
        score_key: Key in state to check for score

    Returns:
        Evaluator function

    Example:
        evaluator = simple_score_evaluator(threshold=0.7)
        # Returns CONTINUE if state[score_key] >= 0.7
        # Returns RETRY otherwise
    """

    def evaluator(state: Dict[str, Any]) -> EvaluationResult:
        score = state.get(score_key, 0.0)

        if score >= threshold:
            return EvaluationResult(
                decision="continue",
                score=score,
                reason=f"Score {score:.2f} >= threshold {threshold}",
            )

        return EvaluationResult(
            decision="retry",
            score=score,
            reason=f"Score {score:.2f} < threshold {threshold}",
        )

    return evaluator


def progress_tracking_evaluator(
    score_key: str = "score",
    complete_threshold: float = 0.9,
    plateau_window: int = 3,
    plateau_tolerance: float = 0.02,
) -> Evaluator:
    """Create a progress-aware evaluator with plateau detection.

    Inspired by SubSearch (arXiv:2604.07415): provides intermediate
    rewards based on meaningful progress checkpoints rather than
    binary pass/fail. Detects stalled progress and triggers
    early termination.

    Args:
        score_key: State key containing current score
        complete_threshold: Score at which task is complete
        plateau_window: Number of iterations to check for plateau
        plateau_tolerance: Min score improvement to not be a plateau

    Returns:
        Evaluator function that tracks progress across invocations

    Example:
        evaluator = progress_tracking_evaluator(
            complete_threshold=0.9,
            plateau_window=3,
        )
    """
    # Track score history across invocations via closure
    score_history: List[float] = []

    def evaluator(state: Dict[str, Any]) -> EvaluationResult:
        score = state.get(score_key, 0.0)
        score_history.append(score)
        iteration = len(score_history)

        # Check completion
        if score >= complete_threshold:
            return EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=score,
                reason=f"Score {score:.2f} >= threshold {complete_threshold}",
                metrics={"progress_history": list(score_history)},
            )

        # Check for plateau (no improvement over window)
        if len(score_history) >= plateau_window:
            recent = score_history[-plateau_window:]
            improvement = max(recent) - min(recent)
            if improvement < plateau_tolerance:
                return EvaluationResult(
                    decision=EvaluationDecision.FAIL,
                    score=score,
                    reason=(
                        f"Progress plateaued: {improvement:.3f} improvement "
                        f"over {plateau_window} iterations"
                    ),
                    metrics={"progress_history": list(score_history)},
                )

        # Check if making progress (score increasing)
        if len(score_history) >= 2:
            delta = score_history[-1] - score_history[-2]
            if delta > 0:
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=score,
                    reason=f"Progress: +{delta:.3f} (iteration {iteration})",
                    metrics={"delta": delta, "progress_history": list(score_history)},
                )
            else:
                return EvaluationResult(
                    decision=EvaluationDecision.RETRY,
                    score=score,
                    reason=f"Regression: {delta:.3f} (iteration {iteration})",
                    metrics={"delta": delta, "progress_history": list(score_history)},
                )

        # First iteration — continue
        return EvaluationResult(
            decision=EvaluationDecision.CONTINUE,
            score=score,
            reason=f"Initial evaluation (iteration {iteration})",
            metrics={"progress_history": list(score_history)},
        )

    return evaluator


def composite_evaluator(
    evaluators: List[Evaluator],
    strategy: str = "worst",
) -> Evaluator:
    """Chain multiple evaluators with configurable aggregation.

    Args:
        evaluators: List of evaluator functions to combine
        strategy: Aggregation strategy:
            - "worst": Use worst (most pessimistic) result
            - "best": Use best (most optimistic) result
            - "average": Average scores, use majority decision

    Returns:
        Combined evaluator function

    Example:
        evaluator = composite_evaluator([
            simple_score_evaluator(threshold=0.7),
            error_count_evaluator(max_errors=0),
        ], strategy="worst")
    """

    def evaluator(state: Dict[str, Any]) -> EvaluationResult:
        results = [e(state) for e in evaluators]
        if not results:
            return EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.5)

        if strategy == "worst":
            # Pick result with lowest score
            worst = min(results, key=lambda r: r.score)
            return EvaluationResult(
                decision=worst.decision,
                score=worst.score,
                reason=f"Composite (worst of {len(results)}): {worst.reason}",
                metrics={"individual_scores": [r.score for r in results]},
            )
        elif strategy == "best":
            best = max(results, key=lambda r: r.score)
            return EvaluationResult(
                decision=best.decision,
                score=best.score,
                reason=f"Composite (best of {len(results)}): {best.reason}",
                metrics={"individual_scores": [r.score for r in results]},
            )
        else:  # average
            avg_score = sum(r.score for r in results) / len(results)
            # Majority decision
            from collections import Counter

            decisions = [str(r.decision) for r in results]
            majority = Counter(decisions).most_common(1)[0][0]
            return EvaluationResult(
                decision=majority,
                score=avg_score,
                reason=f"Composite (avg of {len(results)}): score={avg_score:.2f}",
                metrics={"individual_scores": [r.score for r in results]},
            )

    return evaluator


def convergence_evaluator(
    score_key: str = "score",
    min_iterations: int = 2,
    convergence_threshold: float = 0.01,
    min_score: float = 0.6,
) -> Evaluator:
    """Create evaluator that detects convergence to a stable solution.

    Inspired by iterative refinement patterns from RefineRL
    (arXiv:2604.00790): stop when further iterations yield
    diminishing returns.

    Args:
        score_key: State key with score
        min_iterations: Minimum iterations before convergence check
        convergence_threshold: Max score delta to consider converged
        min_score: Minimum score to accept convergence

    Returns:
        Evaluator that detects convergence
    """
    score_history: List[float] = []

    def evaluator(state: Dict[str, Any]) -> EvaluationResult:
        score = state.get(score_key, 0.0)
        score_history.append(score)

        if len(score_history) < min_iterations:
            return EvaluationResult(
                decision=EvaluationDecision.CONTINUE,
                score=score,
                reason=f"Need {min_iterations - len(score_history)} more iterations",
            )

        # Check convergence: last two scores close enough?
        delta = abs(score_history[-1] - score_history[-2])
        if delta < convergence_threshold and score >= min_score:
            return EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=score,
                reason=(
                    f"Converged: delta={delta:.4f} < {convergence_threshold}, "
                    f"score={score:.2f} >= {min_score}"
                ),
                metrics={"iterations": len(score_history), "final_delta": delta},
            )

        if score < min_score and delta < convergence_threshold:
            return EvaluationResult(
                decision=EvaluationDecision.FAIL,
                score=score,
                reason=f"Converged below minimum: score={score:.2f} < {min_score}",
            )

        return EvaluationResult(
            decision=EvaluationDecision.CONTINUE,
            score=score,
            reason=f"Not converged: delta={delta:.4f}",
        )

    return evaluator


def error_count_evaluator(
    max_errors: int = 0,
    error_key: str = "errors",
) -> Evaluator:
    """Create an error-count-based evaluator.

    Args:
        max_errors: Maximum allowed errors
        error_key: Key in state containing errors (list or count)

    Returns:
        Evaluator function

    Example:
        evaluator = error_count_evaluator(max_errors=0)
        # Returns CONTINUE if no errors
        # Returns RETRY if errors found
    """

    def evaluator(state: Dict[str, Any]) -> EvaluationResult:
        errors = state.get(error_key, [])

        # Handle both list and integer count
        if isinstance(errors, list):
            error_count = len(errors)
        else:
            error_count = int(errors)

        if error_count <= max_errors:
            return EvaluationResult(
                decision="continue",
                score=1.0 - (error_count * 0.1),
                reason=f"{error_count} errors (max: {max_errors})",
            )

        return EvaluationResult(
            decision="retry",
            score=0.0,
            reason=f"{error_count} errors exceeds max {max_errors}",
        )

    return evaluator
