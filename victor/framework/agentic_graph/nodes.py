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

"""Agentic loop nodes for StateGraph-based execution.

This module provides the core nodes for the PERCEIVE-PLAN-ACT-EVALUATE
agentic loop when expressed as a StateGraph.

Nodes:
    perceive_node: Understand user intent and task requirements
    plan_node: Generate execution plan (LLM or fast-path)
    act_node: Execute plan via TurnExecutor
    evaluate_node: Assess progress and quality
    decide_edge: Conditional routing based on evaluation

Each node is a pure function that receives state and returns updated state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.framework.agentic_graph._state_utils import normalize_state_argument
from victor.framework.agentic_graph.state import AgenticLoopStateModel
from victor.framework.evaluation_nodes import EvaluationDecision

if TYPE_CHECKING:
    from victor.framework.evaluation_nodes import EnhancedCompletionEvaluator
    from victor.framework.fulfillment import FulfillmentDetector
    from victor.agent.services.turn_execution_runtime import TurnExecutor

logger = logging.getLogger(__name__)


# =============================================================================
# PERCEIVE Node
# =============================================================================


@normalize_state_argument()
async def perceive_node(
    state: AgenticLoopStateModel,
    runtime_intelligence: Optional[Any] = None,
    perception_integration: Optional[Any] = None,
) -> AgenticLoopStateModel:
    """PERCEIVE stage: Understand user intent and task requirements.

    This node analyzes the user's query to determine:
    - Action intent (read, write, execute, etc.)
    - Task complexity (low, medium, high)
    - Task type (code_generation, debugging, etc.)
    - Confidence level

    Args:
        state: Canonical agentic loop state. Legacy dict/COW inputs are
            normalized before the node body runs.
        runtime_intelligence: Optional RuntimeIntelligenceService instance
        perception_integration: Optional PerceptionIntegration instance (alternative)

    Returns:
        Updated state with perception results
    """
    try:
        # Use provided service or fallback
        if perception_integration is not None:
            # Use PerceptionIntegration directly
            perception_result = await perception_integration.perceive(
                query=state.query,
                context=state.context or {},
            )
        elif runtime_intelligence is not None:
            # Use RuntimeIntelligenceService
            perception_result = await runtime_intelligence.analyze_turn(
                query=state.query,
                context=state.context or {},
            )
        else:
            # Fallback: create simple perception
            perception_result = _create_fallback_perception(state.query)

        # Extract perception data
        perception = {
            "intent": getattr(perception_result, "intent", None),
            "complexity": getattr(perception_result, "complexity", "medium"),
            "confidence": getattr(perception_result, "confidence", 0.5),
        }

        # Get task type from task_analysis if available
        task_type = "general"
        complexity = "medium"

        if hasattr(perception_result, "task_analysis"):
            task_analysis = perception_result.task_analysis
            if task_analysis:
                task_type = getattr(task_analysis, "task_type", task_type)
                complexity = getattr(perception_result, "complexity", complexity)

        # Update state
        return state.model_copy(
            update={
                "stage": "perceive",
                "perception": perception,
                "task_type": task_type,
                "complexity": complexity,
                "iteration": state.iteration + 1,
            }
        )

    except Exception as e:
        logger.warning(f"Perception failed: {e}, using fallback")

        # Fallback perception
        return state.model_copy(
            update={
                "stage": "perceive",
                "perception": {
                    "intent": "query",
                    "complexity": "medium",
                    "confidence": 0.5,
                    "error": str(e),
                },
                "task_type": "general",
                "complexity": "medium",
                "iteration": state.iteration + 1,
            }
        )


# =============================================================================
# PLAN Node
# =============================================================================


@normalize_state_argument()
async def plan_node(
    state: AgenticLoopStateModel,
    planning_coordinator: Optional[Any] = None,
    use_llm_planning: bool = False,
    runtime_intelligence: Optional[Any] = None,
) -> AgenticLoopStateModel:
    """PLAN stage: Generate execution plan.

    This node creates an execution plan based on the perception results.
    Can use LLM-based planning or fast-path heuristics.

    Args:
        state: Canonical agentic loop state. Legacy dict/COW inputs are
            normalized before the node body runs.
        planning_coordinator: Optional PlanningCoordinator instance
        use_llm_planning: Whether to use LLM for planning (default: False)
        runtime_intelligence: Optional runtime intelligence service for
            planning-routing hints

    Returns:
        Updated state with execution plan
    """
    planning_events = list(state.planning_events)
    planning_routing_hints = dict(state.planning_routing_hints)
    structured_routing_policy = dict(state.structured_routing_policy)
    effective_use_llm_planning = use_llm_planning

    if runtime_intelligence is not None:
        planning_scope_context = {
            **(state.context or {}),
            "task_type": state.task_type or "general",
            "tool_budget": (state.context or {}).get("tool_budget", 10),
        }
        if hasattr(runtime_intelligence, "get_structured_routing_policy"):
            policy = runtime_intelligence.get_structured_routing_policy(
                query=state.query,
                scope_context=planning_scope_context,
            )
            if policy is not None:
                if hasattr(policy, "planning_context"):
                    resolved_planning_hints = policy.planning_context()
                    if isinstance(resolved_planning_hints, dict):
                        planning_routing_hints = dict(resolved_planning_hints)
                if hasattr(policy, "to_dict"):
                    serialized_policy = policy.to_dict()
                    if isinstance(serialized_policy, dict):
                        structured_routing_policy = serialized_policy
        elif hasattr(runtime_intelligence, "get_planning_routing_context"):
            resolved_planning_hints = runtime_intelligence.get_planning_routing_context(
                query=state.query,
                scope_context=planning_scope_context,
            )
            if isinstance(resolved_planning_hints, dict):
                planning_routing_hints = dict(resolved_planning_hints)

    if planning_routing_hints.get("planning_force_llm") and planning_coordinator is not None:
        effective_use_llm_planning = True
    elif planning_routing_hints.get("planning_prefer_fast_path"):
        effective_use_llm_planning = False

    if effective_use_llm_planning and planning_coordinator is not None:
        # LLM-based planning
        try:
            plan_result = await planning_coordinator.chat_with_planning(
                message=state.query,
                task_type=state.task_type or "general",
                complexity=state.complexity or "medium",
            )

            plan = {
                "content": getattr(plan_result, "content", ""),
                "tool_calls": getattr(plan_result, "tool_calls", []),
                "reasoning": getattr(plan_result, "reasoning", ""),
            }
            response_metadata = getattr(plan_result, "metadata", None)
            if not isinstance(response_metadata, dict):
                response_metadata = {}
            plan_execution_state = response_metadata.get("plan_execution_state")
            if not isinstance(plan_execution_state, dict):
                plan_execution_state = {}
            planning_events.append(
                {
                    "selection_policy": (
                        "experiment_forced_slow_path"
                        if planning_routing_hints.get("planning_force_llm")
                        else "llm_planning"
                    ),
                    "used_llm_planning": True,
                    "task_type": state.task_type or "general",
                    "skip_reason": None,
                    "forced_by_runtime_feedback": bool(
                        planning_routing_hints.get("planning_force_llm")
                    ),
                    "force_reason": planning_routing_hints.get("planning_force_reason"),
                    "preference_reason": planning_routing_hints.get("planning_prefer_reason"),
                    "preferred_by_runtime_feedback": bool(
                        planning_routing_hints.get("planning_prefer_fast_path")
                    ),
                    "constraint_tags": list(
                        planning_routing_hints.get("planning_constraint_tags") or []
                    ),
                    "experiment_support": planning_routing_hints.get("planning_experiment_support"),
                    "has_plan_execution_state": bool(plan_execution_state),
                }
            )

            return state.model_copy(
                update={
                    "stage": "plan",
                    "plan": plan,
                    "plan_execution_state": plan_execution_state,
                    "planning_events": planning_events,
                    "planning_routing_hints": planning_routing_hints,
                    "structured_routing_policy": structured_routing_policy,
                }
            )

        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, falling back to fast path")

    # Fast-path planning (rule-based)
    plan = _build_fast_path_plan(state)
    planning_events.append(
        {
            "selection_policy": (
                "heuristic_fast_path"
                if not planning_routing_hints.get("planning_prefer_fast_path")
                else "heuristic_fast_path"
            ),
            "used_llm_planning": False,
            "task_type": state.task_type or "general",
            "skip_reason": None,
            "forced_by_runtime_feedback": bool(planning_routing_hints.get("planning_force_llm")),
            "force_reason": planning_routing_hints.get("planning_force_reason"),
            "preference_reason": planning_routing_hints.get("planning_prefer_reason"),
            "preferred_by_runtime_feedback": bool(
                planning_routing_hints.get("planning_prefer_fast_path")
            ),
            "constraint_tags": list(planning_routing_hints.get("planning_constraint_tags") or []),
            "experiment_support": planning_routing_hints.get("planning_experiment_support"),
        }
    )

    return state.model_copy(
        update={
            "stage": "plan",
            "plan": plan,
            "planning_events": planning_events,
            "planning_routing_hints": planning_routing_hints,
            "structured_routing_policy": structured_routing_policy,
        }
    )


def _build_fast_path_plan(state: AgenticLoopStateModel) -> Dict[str, Any]:
    """Build a fast-path execution plan using heuristics.

    Args:
        state: Current agentic loop state

    Returns:
        Fast-path plan dictionary
    """
    perception = state.perception or {}
    intent = perception.get("intent")
    task_type = state.task_type or "general"
    complexity = state.complexity or "medium"

    # Default plan
    plan = {
        "approach": "fast_path",
        "tool_calls": [],
        "reasoning": "Rule-based planning for simple tasks",
    }

    # Determine tools based on intent
    if intent and "write" in str(intent).lower():
        plan["tool_calls"] = ["code_search", "write_file"]
        plan["reasoning"] = "Write operation requires code search and file writing"
    elif intent and "read" in str(intent).lower():
        plan["tool_calls"] = ["code_search", "read_file"]
        plan["reasoning"] = "Read operation requires code search and file reading"
    elif task_type == "debugging":
        plan["tool_calls"] = ["code_search", "read_file", "grep"]
        plan["reasoning"] = "Debugging requires search and analysis tools"
    elif task_type == "code_generation":
        plan["tool_calls"] = ["code_search", "write_file"]
        plan["reasoning"] = "Code generation requires search and writing"
    elif complexity == "low":
        plan["tool_calls"] = ["chat"]
        plan["reasoning"] = "Simple query, chat only needed"

    return plan


# =============================================================================
# ACT Node
# =============================================================================


@normalize_state_argument()
async def act_node(
    state: AgenticLoopStateModel,
    turn_executor: Optional[Any] = None,
) -> AgenticLoopStateModel:
    """ACT stage: Execute the plan.

    This node executes the plan via TurnExecutor (single-turn execution).

    Args:
        state: Canonical agentic loop state. Legacy dict/COW inputs are
            normalized before the node body runs.
        turn_executor: Optional TurnExecutor instance

    Returns:
        Updated state with action results
    """
    degradation_events = list(state.degradation_events)

    try:
        if turn_executor is not None:
            # Execute via TurnExecutor
            result = await turn_executor.execute_turn(
                user_message=state.query,
                task_classification=state.task_type,
                runtime_context_overrides=state.context,
            )

            # Extract results
            action_result = {
                "response": getattr(result, "response", ""),
                "tool_results": getattr(result, "tool_results", []),
            }

            tool_results = action_result.get("tool_results", [])

            return state.model_copy(
                update={
                    "stage": "act",
                    "action_result": action_result,
                    "tool_results": tool_results,
                }
            )

        else:
            # Fallback: create basic action result
            action_result = {
                "response": f"Processed: {state.query}",
                "tool_results": [],
                "fallback": True,
            }

            return state.model_copy(
                update={
                    "stage": "act",
                    "action_result": action_result,
                    "tool_results": [],
                }
            )

    except Exception as e:
        logger.warning(f"Action execution failed: {e}")

        # Return error state
        action_result = {
            "response": f"Execution failed: {e}",
            "tool_results": [],
            "error": str(e),
        }
        degradation_events.append(
            {
                "source": "agentic_graph",
                "failure_type": "ACTION_ERROR",
                "recovered": False,
                "error": str(e),
            }
        )

        return state.model_copy(
            update={
                "stage": "act",
                "action_result": action_result,
                "tool_results": [],
                "degradation_events": degradation_events,
            }
        )


# =============================================================================
# EVALUATE Node
# =============================================================================


@normalize_state_argument()
async def evaluate_node(
    state: AgenticLoopStateModel,
    evaluator: Optional[Any] = None,
    fulfillment_detector: Optional[Any] = None,
    enable_fulfillment_check: bool = False,
) -> AgenticLoopStateModel:
    """EVALUATE stage: Assess progress and quality.

    This node evaluates the action result to determine:
    - Whether to continue, complete, or fail
    - Progress score for tracking
    - Optional fulfillment check

    Args:
        state: Canonical agentic loop state. Legacy dict/COW inputs are
            normalized before the node body runs.
        evaluator: Optional EnhancedCompletionEvaluator instance
        fulfillment_detector: Optional FulfillmentDetector instance
        enable_fulfillment_check: Whether to run fulfillment check

    Returns:
        Updated state with evaluation results
    """
    degradation_events = list(state.degradation_events)

    try:
        if evaluator is not None:
            # Use provided evaluator
            eval_result = await evaluator.evaluate(
                perception=state.perception,
                action_result=state.action_result,
                state=state.to_dict(),
            )

            evaluation = {
                "decision": getattr(eval_result, "decision", "continue"),
                "score": getattr(eval_result, "score", 0.5),
                "reason": getattr(eval_result, "reason", ""),
            }

        else:
            # Default evaluation logic
            evaluation = _default_evaluation(state)

        # Update progress scores
        progress_scores = list(state.progress_scores)
        progress_scores.append(evaluation.get("score", 0.5))

        update_dict = {
            "stage": "evaluate",
            "evaluation": evaluation,
            "progress_scores": progress_scores,
        }

        # Optional fulfillment check
        if enable_fulfillment_check and fulfillment_detector is not None:
            fulfillment_result = fulfillment_detector.get_fulfillment_result(
                query=state.query,
                action_result=state.action_result,
                perception=state.perception,
            )

            update_dict["fulfillment"] = fulfillment_result

        return state.model_copy(update=update_dict)

    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")

        # Fallback evaluation
        evaluation = {
            "decision": "continue",
            "score": 0.5,
            "error": str(e),
        }
        degradation_events.append(
            {
                "source": "agentic_graph",
                "failure_type": "EVALUATION_ERROR",
                "recovered": False,
                "error": str(e),
            }
        )

        return state.model_copy(
            update={
                "stage": "evaluate",
                "evaluation": evaluation,
                "progress_scores": list(state.progress_scores) + [0.5],
                "degradation_events": degradation_events,
            }
        )


def _default_evaluation(state: AgenticLoopStateModel) -> Dict[str, Any]:
    """Default evaluation logic when no evaluator provided.

    Args:
        state: Current agentic loop state

    Returns:
        Evaluation dictionary with decision and score
    """
    action_result = state.action_result or {}

    # Check for errors
    if action_result.get("error"):
        return {
            "decision": "fail",
            "score": 0.0,
            "reason": "Execution error occurred",
        }

    # Check for completion signals
    response = action_result.get("response", "")
    if response and any(
        marker in response.lower() for marker in ["done", "complete", "finished", "successfully"]
    ):
        return {
            "decision": "complete",
            "score": 0.9,
            "reason": "Task completed successfully",
        }

    # Default: continue
    return {
        "decision": "continue",
        "score": 0.5,
        "reason": "Progress made, continue execution",
    }


# =============================================================================
# DECIDE Edge (Conditional Routing)
# =============================================================================


@normalize_state_argument()
def decide_edge(state: AgenticLoopStateModel) -> str:
    """DECIDE stage: Route to next node based on evaluation.

    This is a conditional edge function for StateGraph that determines
    the next node based on the evaluation decision.

    Args:
        state: Canonical agentic loop state. Legacy dict/COW inputs are
            normalized before routing.

    Returns:
        Next node name: "perceive", "act", or "__end__"
    """
    # Check max iterations first
    if state.iteration >= state.max_iterations:
        return "__end__"

    # Get evaluation decision
    evaluation = state.evaluation or {}
    decision = evaluation.get("decision", "continue")

    # Normalize decision to string
    if isinstance(decision, EvaluationDecision):
        decision_str = decision.value
    else:
        decision_str = str(decision).lower()

    # Route based on decision
    if decision_str == EvaluationDecision.COMPLETE.value or decision_str == "complete":
        return "__end__"
    elif decision_str == EvaluationDecision.FAIL.value or decision_str == "fail":
        return "__end__"
    elif decision_str == EvaluationDecision.RETRY.value or decision_str == "retry":
        return "act"
    else:  # continue or any other
        return "perceive"


def _create_fallback_perception(query: str) -> Any:
    """Create a fallback perception result when no service is available.

    Args:
        query: User's query

    Returns:
        Mock perception result
    """

    # Create a simple mock object
    class FallbackPerception:
        def __init__(self, query: str):
            query_lower = query.lower()
            if any(word in query_lower for word in ["write", "create", "implement"]):
                self.intent = type("Intent", (), {"value": "write"})()
            elif any(word in query_lower for word in ["read", "show", "display"]):
                self.intent = type("Intent", (), {"value": "read"})()
            else:
                self.intent = type("Intent", (), {"value": "query"})()

            self.complexity = "medium"
            self.confidence = 0.5

            self.task_analysis = type(
                "TaskAnalysis",
                (),
                {"task_type": "general"},
            )()

    return FallbackPerception(query)
