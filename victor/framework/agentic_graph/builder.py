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

"""Agentic loop graph builder - creates StateGraph for agentic loop execution.

This module provides the create_agentic_loop_graph function that builds
a StateGraph implementing the PERCEIVE-PLAN-ACT-EVALUATE-DECIDE loop.

Graph Structure:
    START -> perceive -> plan -> act -> evaluate -> decide
    decide -> perceive (loop) or __end__ (terminate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from victor.framework.graph import CopyOnWriteState, StateGraph, END
from victor.framework.agentic_graph.state import AgenticLoopStateModel
from victor.framework.agentic_graph.nodes import (
    perceive_node,
    plan_node,
    act_node,
    evaluate_node,
    decide_edge,
)

logger = logging.getLogger(__name__)


@dataclass
class AgenticLoopDependencies:
    """Typed dependency container for agentic-graph node construction.

    The container is the canonical builder seam. Legacy ``(value, resolver)``
    parameters are still accepted by ``create_agentic_loop_graph()`` and are
    normalized into this object for backward compatibility.
    """

    runtime_intelligence: Optional[Any] = None
    planning_coordinator: Optional[Any] = None
    use_llm_planning: Optional[bool] = None
    turn_executor: Optional[Any] = None
    evaluator: Optional[Any] = None
    fulfillment_detector: Optional[Any] = None
    resolvers: dict[str, Callable[[], Any]] = field(default_factory=dict)

    def resolve(self, name: str) -> Optional[Any]:
        """Resolve a dependency from a live resolver or a static fallback."""
        resolver = self.resolvers.get(name)
        if resolver is not None:
            return resolver()
        return getattr(self, name)


def _build_dependencies(
    *,
    dependencies: Optional[AgenticLoopDependencies],
    runtime_intelligence: Optional[Any],
    runtime_intelligence_resolver: Optional[Callable[[], Any]],
    planning_coordinator: Optional[Any],
    planning_coordinator_resolver: Optional[Callable[[], Any]],
    use_llm_planning: bool,
    use_llm_planning_resolver: Optional[Callable[[], bool]],
    turn_executor: Optional[Any],
    turn_executor_resolver: Optional[Callable[[], Any]],
    evaluator: Optional[Any],
    evaluator_resolver: Optional[Callable[[], Any]],
    fulfillment_detector: Optional[Any],
    fulfillment_detector_resolver: Optional[Callable[[], Any]],
) -> AgenticLoopDependencies:
    """Normalize legacy builder arguments into the canonical dependency container."""
    resolved = dependencies or AgenticLoopDependencies()
    if runtime_intelligence is not None:
        resolved.runtime_intelligence = runtime_intelligence
    if planning_coordinator is not None:
        resolved.planning_coordinator = planning_coordinator
    if turn_executor is not None:
        resolved.turn_executor = turn_executor
    if evaluator is not None:
        resolved.evaluator = evaluator
    if fulfillment_detector is not None:
        resolved.fulfillment_detector = fulfillment_detector
    if use_llm_planning or resolved.use_llm_planning is None:
        resolved.use_llm_planning = use_llm_planning

    if runtime_intelligence_resolver is not None:
        resolved.resolvers["runtime_intelligence"] = runtime_intelligence_resolver
    if planning_coordinator_resolver is not None:
        resolved.resolvers["planning_coordinator"] = planning_coordinator_resolver
    if use_llm_planning_resolver is not None:
        resolved.resolvers["use_llm_planning"] = use_llm_planning_resolver
    if turn_executor_resolver is not None:
        resolved.resolvers["turn_executor"] = turn_executor_resolver
    if evaluator_resolver is not None:
        resolved.resolvers["evaluator"] = evaluator_resolver
    if fulfillment_detector_resolver is not None:
        resolved.resolvers["fulfillment_detector"] = fulfillment_detector_resolver

    return resolved


def _apply_agentic_state_defaults(
    state: Any,
    *,
    max_iterations: int,
) -> Any:
    """Populate builder-owned state defaults for raw dict compatibility paths."""
    if isinstance(state, dict):
        if "max_iterations" in state:
            return state
        normalized_state = dict(state)
        normalized_state["max_iterations"] = max_iterations
        return normalized_state

    if isinstance(state, CopyOnWriteState):
        current_state = state.get_state()
        if isinstance(current_state, dict) and "max_iterations" not in current_state:
            state["max_iterations"] = max_iterations

    return state


def _bind_configured_node(
    node_fn: Callable[..., Any],
    /,
    *,
    max_iterations: int,
    dependencies: AgenticLoopDependencies,
    dependency_names: tuple[str, ...] = (),
    **static_dependencies: Any,
) -> Callable[[Any], Any]:
    """Create a named node wrapper that resolves dependencies at execution time."""

    def _configured_node(state: Any) -> Any:
        state = _apply_agentic_state_defaults(state, max_iterations=max_iterations)
        resolved_dependencies = {name: dependencies.resolve(name) for name in dependency_names}
        resolved_dependencies.update(static_dependencies)
        return node_fn(state, **resolved_dependencies)

    _configured_node.__name__ = getattr(node_fn, "__name__", "configured_node")
    return _configured_node


def create_agentic_loop_graph(
    max_iterations: int = 10,
    enable_fulfillment: bool = True,
    enable_adaptive_iterations: bool = True,
    *,
    dependencies: Optional[AgenticLoopDependencies] = None,
    include_prompt_node: bool = False,
    prompt_node: Optional[Callable[[Any], Any]] = None,
    runtime_intelligence: Optional[Any] = None,
    runtime_intelligence_resolver: Optional[Callable[[], Any]] = None,
    planning_coordinator: Optional[Any] = None,
    planning_coordinator_resolver: Optional[Callable[[], Any]] = None,
    use_llm_planning: bool = False,
    use_llm_planning_resolver: Optional[Callable[[], bool]] = None,
    turn_executor: Optional[Any] = None,
    turn_executor_resolver: Optional[Callable[[], Any]] = None,
    evaluator: Optional[Any] = None,
    evaluator_resolver: Optional[Callable[[], Any]] = None,
    fulfillment_detector: Optional[Any] = None,
    fulfillment_detector_resolver: Optional[Callable[[], Any]] = None,
) -> StateGraph:
    """Create a StateGraph that implements the agentic loop.

    The graph implements the PERCEIVE-PLAN-ACT-EVALUATE-DECIDE loop
    using StateGraph nodes and conditional edges.

    Args:
        max_iterations: Maximum number of loop iterations (default: 10)
        enable_fulfillment: Whether to enable fulfillment checks (default: True)
        enable_adaptive_iterations: Whether to enable adaptive termination (default: True)

    Returns:
        Compiled StateGraph ready for execution

    Example:
        graph = create_agentic_loop_graph(max_iterations=5)
        compiled = graph.compile()
        state = AgenticLoopStateModel(query="Write code", max_iterations=5)
        result = await compiled.invoke(state)

    Compatibility:
        ``AgenticLoopStateModel`` is the canonical input type. Raw ``dict`` input
        remains supported for compatibility; when used, builder-owned defaults
        such as ``max_iterations`` are injected before node execution.

    Dependency injection:
        ``dependencies=AgenticLoopDependencies(...)`` is the canonical builder
        seam. Legacy ``*_resolver`` arguments remain supported and are folded
        into the container for backward compatibility.
    """
    resolved_dependencies = _build_dependencies(
        dependencies=dependencies,
        runtime_intelligence=runtime_intelligence,
        runtime_intelligence_resolver=runtime_intelligence_resolver,
        planning_coordinator=planning_coordinator,
        planning_coordinator_resolver=planning_coordinator_resolver,
        use_llm_planning=use_llm_planning,
        use_llm_planning_resolver=use_llm_planning_resolver,
        turn_executor=turn_executor,
        turn_executor_resolver=turn_executor_resolver,
        evaluator=evaluator,
        evaluator_resolver=evaluator_resolver,
        fulfillment_detector=fulfillment_detector,
        fulfillment_detector_resolver=fulfillment_detector_resolver,
    )

    graph = StateGraph(
        AgenticLoopStateModel,
        metadata={
            "max_iterations": max_iterations,
            "enable_fulfillment": enable_fulfillment,
            "enable_adaptive_iterations": enable_adaptive_iterations,
            "include_prompt_node": include_prompt_node,
        },
    )

    resolved_prompt_node = prompt_node
    if include_prompt_node and resolved_prompt_node is None:
        from victor.framework.agentic_graph.service_nodes import prompt_service_node

        resolved_prompt_node = prompt_service_node

    # Add nodes
    # Note: We use lambda functions to inject configuration into nodes
    if include_prompt_node and resolved_prompt_node is not None:
        graph.add_node(
            "prompt",
            _bind_configured_node(
                resolved_prompt_node,
                max_iterations=max_iterations,
                dependencies=resolved_dependencies,
            ),
        )

    graph.add_node(
        "perceive",
        _bind_configured_node(
            perceive_node,
            max_iterations=max_iterations,
            dependencies=resolved_dependencies,
            dependency_names=("runtime_intelligence",),
        ),
    )

    graph.add_node(
        "plan",
        _bind_configured_node(
            plan_node,
            max_iterations=max_iterations,
            dependencies=resolved_dependencies,
            dependency_names=(
                "planning_coordinator",
                "use_llm_planning",
                "runtime_intelligence",
            ),
        ),
    )

    graph.add_node(
        "act",
        _bind_configured_node(
            act_node,
            max_iterations=max_iterations,
            dependencies=resolved_dependencies,
            dependency_names=("turn_executor",),
        ),
    )

    graph.add_node(
        "evaluate",
        _bind_configured_node(
            evaluate_node,
            max_iterations=max_iterations,
            dependencies=resolved_dependencies,
            dependency_names=("evaluator", "fulfillment_detector"),
            enable_fulfillment_check=enable_fulfillment,
        ),
    )

    # Set entry point
    graph.set_entry_point("prompt" if include_prompt_node else "perceive")

    # Add sequential edges (PROMPT -> PERCEIVE -> PLAN -> ACT -> EVALUATE)
    if include_prompt_node and resolved_prompt_node is not None:
        graph.add_edge("prompt", "perceive")
    graph.add_edge("perceive", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "evaluate")

    # Add conditional edge from EVALUATE to DECIDE
    # The decide_edge function returns the next node name
    graph.add_conditional_edge(
        "evaluate",
        decide_edge,
        {
            "perceive": "perceive",  # Continue loop
            "act": "act",  # Retry
            "__end__": END,  # Complete or fail
        },
    )

    return graph
