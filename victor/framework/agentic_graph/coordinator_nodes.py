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

"""Coordinator adapter nodes for StateGraph integration (Phase 2 consolidation).

This module provides adapter nodes that bridge state-passed coordinators
with StateGraph execution. Each coordinator becomes a pure function node
that takes AgenticLoopStateModel and returns updated state.

Key Components:
- CoordinatorAdapter: Generic adapter for state-passed coordinators
- exploration_node: Parallel codebase exploration
- safety_node: Tool call safety checking
- system_prompt_node: Task classification and prompt building

Pattern:
    StateGraph Node (pure function)
        ├─ Extracts context from AgenticLoopStateModel
        ├─ Creates ContextSnapshot for coordinator
        ├─ Calls state-passed coordinator
        └─ Returns updated AgenticLoopStateModel with transitions applied
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.framework.agentic_graph._state_utils import normalize_state_argument
from victor.framework.agentic_graph.state import AgenticLoopStateModel

if TYPE_CHECKING:
    from victor.agent.coordinators.state_context import (
        CoordinatorResult,
        ContextSnapshot,
        StateTransition,
        TransitionBatch,
    )

logger = logging.getLogger(__name__)


def _create_context_snapshot(
    state: AgenticLoopStateModel,
    orchestrator: Optional[Any] = None,
) -> "ContextSnapshot":
    """Create a ContextSnapshot from AgenticLoopStateModel.

    Args:
        state: Current agentic loop state
        orchestrator: Optional orchestrator for full snapshot

    Returns:
        ContextSnapshot for state-passed coordinators
    """
    from victor.agent.coordinators.state_context import ContextSnapshot

    # Extract conversation state from state model (stored in context field)
    conversation_state = dict(state.context or {})
    session_state = dict(conversation_state.get("session_state", {}))

    # Build messages tuple from conversation_history
    messages = tuple(state.conversation_history or [])

    # Extract capabilities from context
    capabilities = dict(conversation_state.get("capabilities", {}))

    # If orchestrator provided, delegate to create_snapshot
    if orchestrator is not None:
        from victor.agent.coordinators.state_context import create_snapshot

        return create_snapshot(orchestrator)

    # Otherwise, create minimal snapshot from state
    return ContextSnapshot(
        messages=messages,
        session_id=conversation_state.get("session_id", ""),
        conversation_stage=conversation_state.get("stage", state.stage or "perceive"),
        settings=None,  # Not available in state model
        model=conversation_state.get("model", ""),
        provider=conversation_state.get("provider", ""),
        max_tokens=conversation_state.get("max_tokens", 4096),
        temperature=conversation_state.get("temperature", 0.7),
        conversation_state=conversation_state,
        session_state=session_state,
        observed_files=tuple(conversation_state.get("observed_files", [])),
        capabilities=capabilities,
    )


def _apply_transitions_to_state(
    state: AgenticLoopStateModel,
    transitions: "TransitionBatch",
) -> AgenticLoopStateModel:
    """Apply coordinator transitions to state model.

    Args:
        state: Current agentic loop state
        transitions: Transition batch from coordinator

    Returns:
        Updated state with transitions applied
    """
    # Update context with transition data
    context_updates = dict(state.context or {})

    for transition in transitions.transitions:
        from victor.agent.coordinators.state_context import TransitionType

        if transition.transition_type == TransitionType.UPDATE_STATE:
            key = transition.data.get("key")
            value = transition.data.get("value")
            scope = transition.data.get("scope", "conversation")

            if scope == "conversation":
                context_updates[key] = value
            else:
                # Store session-level updates in session_state sub-key
                if "session_state" not in context_updates:
                    context_updates["session_state"] = {}
                context_updates["session_state"][key] = value

    # Return updated state
    return state.model_copy(
        update={
            "context": context_updates,
        }
    )


# =============================================================================
# Generic Coordinator Adapter
# =============================================================================


class CoordinatorAdapter:
    """Generic adapter for state-passed coordinators.

    This adapter wraps any state-passed coordinator and makes it callable
    as a StateGraph node function.

    Usage:
        adapter = CoordinatorAdapter(MyStatePassedCoordinator())
        result = await adapter.call(state, coordinator_method="my_method")
    """

    def __init__(
        self,
        coordinator: Any,
        orchestrator: Optional[Any] = None,
    ):
        """Initialize the adapter.

        Args:
            coordinator: State-passed coordinator instance
            orchestrator: Optional orchestrator for full context snapshots
        """
        self._coordinator = coordinator
        self._orchestrator = orchestrator

    @normalize_state_argument(position=1)
    async def call(
        self,
        state: AgenticLoopStateModel,
        method_name: str,
        **kwargs,
    ) -> AgenticLoopStateModel:
        """Call coordinator method and apply transitions.

        Args:
            state: Canonical agentic loop state. Legacy dict/COW inputs are
                normalized before adapter logic runs.
            method_name: Name of coordinator method to call
            **kwargs: Additional arguments to pass to coordinator method

        Returns:
            Updated state with coordinator transitions applied
        """
        # Create context snapshot
        snapshot = _create_context_snapshot(state, self._orchestrator)

        # Call coordinator method
        method = getattr(self._coordinator, method_name)
        result = await method(snapshot, **kwargs)

        # Apply transitions to state
        if hasattr(result, "transitions"):
            state = _apply_transitions_to_state(state, result.transitions)

        # Store coordinator metadata in context
        if hasattr(result, "metadata") and result.metadata:
            context = dict(state.context or {})
            if "coordinator_results" not in context:
                context["coordinator_results"] = {}
            context["coordinator_results"][method_name] = result.metadata
            state = state.model_copy(update={"context": context})

        return state


# =============================================================================
# Exploration Node
# =============================================================================


@normalize_state_argument()
async def exploration_node(
    state: AgenticLoopStateModel,
    exploration_coordinator: Optional[Any] = None,
    orchestrator: Optional[Any] = None,
) -> AgenticLoopStateModel:
    """Exploration node: Parallel codebase exploration.

    This node runs parallel exploration to discover relevant files and
    code patterns before executing the main task.

    Args:
        state: Canonical agentic loop state. Legacy dict/COW inputs are
            normalized before the node body runs.
        exploration_coordinator: Optional ExplorationStatePassedCoordinator
        orchestrator: Optional orchestrator for full context

    Returns:
        Updated state with exploration findings
    """
    # Skip if no query
    if not state.query:
        return state

    # Use provided coordinator or create default
    if exploration_coordinator is None:
        try:
            from victor.agent.coordinators.exploration_state_passed import (
                ExplorationStatePassedCoordinator,
            )

            exploration_coordinator = ExplorationStatePassedCoordinator()
        except ImportError:
            logger.warning("ExplorationStatePassedCoordinator not available")
            return state

    try:
        # Create adapter and call explore method
        adapter = CoordinatorAdapter(exploration_coordinator, orchestrator)
        state = await adapter.call(state, "explore", user_message=state.query)

        logger.info("Exploration node completed")

    except Exception as e:
        logger.warning(f"Exploration node failed: {e}")
        # Continue without exploration - not critical

    return state


# =============================================================================
# Safety Node
# =============================================================================


@normalize_state_argument()
async def safety_node(
    state: AgenticLoopStateModel,
    safety_coordinator: Optional[Any] = None,
    orchestrator: Optional[Any] = None,
) -> AgenticLoopStateModel:
    """Safety node: Check tool call safety.

    This node validates proposed tool calls against safety rules.
    Can block dangerous operations like git push --force.

    Args:
        state: Canonical agentic loop state. Legacy dict/COW inputs are
            normalized before the node body runs.
        safety_coordinator: Optional SafetyStatePassedCoordinator
        orchestrator: Optional orchestrator for full context

    Returns:
        Updated state with safety check results
    """
    # Extract proposed tool calls from plan
    plan = state.plan or {}
    tool_calls = plan.get("tool_calls", [])

    if not tool_calls:
        return state  # No tools to check

    # Use provided coordinator or create default
    if safety_coordinator is None:
        try:
            from victor.agent.coordinators.safety_state_passed import (
                SafetyStatePassedCoordinator,
            )

            safety_coordinator = SafetyStatePassedCoordinator()
        except ImportError:
            logger.warning("SafetyStatePassedCoordinator not available")
            return state

    try:
        # Check each tool call
        all_safe = True
        safety_results = []

        for tool_call in tool_calls:
            if isinstance(tool_call, str):
                tool_name = tool_call
                tool_args = []
            elif isinstance(tool_call, dict):
                tool_name = tool_call.get("name", tool_call.get("tool", ""))
                tool_args = tool_call.get("arguments", tool_call.get("args", []))
            else:
                continue

            # Create adapter and call check method
            adapter = CoordinatorAdapter(safety_coordinator, orchestrator)
            result = await adapter.call(state, "check", tool_name=tool_name, tool_args=tool_args)

            # Extract safety decision from context
            context = result.context or {}
            check_result = context.get("last_safety_check", {})
            if not check_result.get("is_safe", True):
                all_safe = False
            safety_results.append(check_result)

        # Store safety results in context
        context = dict(state.context or {})
        context["safety_results"] = safety_results
        context["all_tools_safe"] = all_safe

        state = state.model_copy(update={"context": context})

        if not all_safe:
            logger.warning("Safety node: Some tools were blocked")

    except Exception as e:
        logger.error(f"Safety node failed: {e}")
        context = dict(state.context or {})
        context["all_tools_safe"] = False
        context["safety_error"] = str(e)
        state = state.model_copy(update={"context": context})

    return state


# =============================================================================
# System Prompt Node
# =============================================================================


@normalize_state_argument()
async def system_prompt_node(
    state: AgenticLoopStateModel,
    system_prompt_coordinator: Optional[Any] = None,
    orchestrator: Optional[Any] = None,
) -> AgenticLoopStateModel:
    """System prompt node: Task classification and prompt building.

    This node classifies the task and builds appropriate system prompts.

    Args:
        state: Canonical agentic loop state. Legacy dict/COW inputs are
            normalized before the node body runs.
        system_prompt_coordinator: Optional SystemPromptStatePassedCoordinator
        orchestrator: Optional orchestrator for full context

    Returns:
        Updated state with task classification
    """
    # Skip if no query
    if not state.query:
        return state

    # Use provided coordinator or try to get from orchestrator
    if system_prompt_coordinator is None:
        try:
            # Try to get task analyzer from orchestrator
            if orchestrator and hasattr(orchestrator, "task_analyzer"):
                from victor.agent.coordinators.system_prompt_state_passed import (
                    SystemPromptStatePassedCoordinator,
                )

                system_prompt_coordinator = SystemPromptStatePassedCoordinator(
                    orchestrator.task_analyzer
                )
            else:
                # No task analyzer available - return early
                return state
        except ImportError:
            logger.warning("SystemPromptStatePassedCoordinator not available")
            return state

    try:
        # Create adapter and call classify method
        adapter = CoordinatorAdapter(system_prompt_coordinator, orchestrator)
        state = await adapter.call(state, "classify", user_message=state.query)

        logger.info("System prompt node completed")

    except Exception as e:
        logger.warning(f"System prompt node failed: {e}")
        # Continue without classification - not critical

    return state
