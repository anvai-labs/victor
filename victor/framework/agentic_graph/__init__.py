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

"""Agentic Graph - StateGraph-based agentic loop implementation.

This module provides a StateGraph-based implementation of the agentic loop,
replacing the custom while loop with a declarative graph-based approach.

Architecture:
    StateGraph (PERCEIVE -> PLAN -> ACT -> EVALUATE -> DECIDE)
        ├── perceive_node: Intent understanding via PerceptionIntegration
        ├── plan_node: Execution planning via PlanningCoordinator
        ├── act_node: Tool execution via TurnExecutor
        ├── evaluate_node: Progress evaluation via FulfillmentDetector
        └── decide: Conditional edge routing back to PERCEIVE or END

Components:
    - state: AgenticLoopStateModel and helpers
    - nodes: Core agentic loop nodes (perceive, plan, act, evaluate)
    - builder: Graph construction utilities
    - executor: AgenticLoopGraphExecutor for running the graph
    - coordinator_nodes: State-passed coordinator adapters
    - service_nodes: Service provider nodes (chat, tool, context, provider)
    - team_selector: Formation selection logic

Example:
    from victor.framework.agentic_graph import (
        AgenticLoopStateModel,
        create_agentic_loop_graph,
        AgenticLoopGraphExecutor,
    )

    # Canonical typed-state usage
    graph = create_agentic_loop_graph(max_iterations=10)
    compiled = graph.compile()
    state = AgenticLoopStateModel(query="Fix the authentication bug", max_iterations=10)
    result = await compiled.invoke(state)

    # Runtime executor usage
    executor = AgenticLoopGraphExecutor(execution_context, max_iterations=10)
    result = await executor.run("Fix the authentication bug")

Compatibility note:
    ``AgenticLoopStateModel`` is the canonical state type. Passing a raw ``dict``
    to ``compiled.invoke()`` is supported only as a compatibility path. Builder-
    owned defaults such as ``max_iterations`` are injected automatically, but
    callers should prefer the typed model for explicitness and validation.

Team Architecture Note:
    Teams are formations, not separate graphs. Use UnifiedTeamCoordinator
    directly as a StateGraph node:

        from victor.teams import UnifiedTeamCoordinator, TeamFormation
        coordinator = UnifiedTeamCoordinator(orchestrator)
        coordinator.set_formation(TeamFormation.PARALLEL)
        graph.add_node("team", coordinator)  # Direct usage!
"""

# Core state and graph components
from victor.framework.agentic_graph.state import (
    AgenticLoopStateModel,
    create_initial_state,
    should_continue_loop,
)
from victor.framework.agentic_graph.nodes import (
    perceive_node,
    plan_node,
    act_node,
    evaluate_node,
    decide_edge,
)
from victor.framework.agentic_graph.builder import (
    AgenticLoopDependencies,
    create_agentic_loop_graph,
)
from victor.framework.agentic_graph.executor import (
    AgenticLoopGraphExecutor,
    LoopResult,
)

# Coordinator adapters
from victor.framework.agentic_graph.coordinator_nodes import (
    CoordinatorAdapter,
    exploration_node,
    safety_node,
    system_prompt_node,
)

# Service provider nodes
from victor.framework.agentic_graph.service_nodes import (
    chat_service_node,
    tool_service_node,
    context_service_node,
    provider_service_node,
    prompt_service_node,
    inject_execution_context,
)

# Team selector (formation selection logic)
from victor.framework.agentic_graph.team_selector import (
    select_formation,
    FormationCriteria,
    DEFAULT_FORMATION,
)


def __getattr__(name: str):
    """Lazy re-export of deprecated ``AgenticLoopState`` alias."""
    if name == "AgenticLoopState":
        # Import triggers the deprecation warning in state.py's __getattr__
        from victor.framework.agentic_graph import state as _state_mod

        return _state_mod.AgenticLoopState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # State (canonical model — single source of truth)
    "AgenticLoopStateModel",
    "AgenticLoopState",  # deprecated — emits DeprecationWarning
    "create_initial_state",
    "should_continue_loop",
    # Core nodes
    "perceive_node",
    "plan_node",
    "act_node",
    "evaluate_node",
    "decide_edge",
    # Builder and executor
    "AgenticLoopDependencies",
    "create_agentic_loop_graph",
    "AgenticLoopGraphExecutor",
    "LoopResult",
    # Coordinator adapters
    "CoordinatorAdapter",
    "exploration_node",
    "safety_node",
    "system_prompt_node",
    # Service nodes
    "chat_service_node",
    "tool_service_node",
    "context_service_node",
    "provider_service_node",
    "prompt_service_node",
    "inject_execution_context",
    # Team selector
    "select_formation",
    "FormationCriteria",
    "DEFAULT_FORMATION",
]
