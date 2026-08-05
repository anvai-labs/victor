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

"""Dynamic router formation for task-based agent routing.

This module provides DynamicRouterFormation, which routes tasks to
appropriate agents based on task analysis and agent capabilities.

Formation Pattern:
    Router → (Agent 1 | Agent 2 | Agent 3 | ...)

Routes tasks to the agent best suited for the task based on:
- Task category (coding, research, analysis, writing, etc.)
- Agent capabilities and roles
- Keyword-based fallback routing

SOLID Principles:
- SRP: Routing logic only
- OCP: Extensible via custom routing strategies
- LSP: Substitutable with other formations
- DIP: Depends on TeamContext and BaseFormationStrategy abstractions

Usage:
    from victor.coordination.formations.dynamic_router import DynamicRouterFormation
    from victor.coordination.formations.base import TeamContext

    # Create router with custom mappings
    formation = DynamicRouterFormation(
        category_to_role={
            "coding": "coder",
            "research": "researcher",
            "analysis": "analyst",
            "writing": "writer",
        },
        keyword_to_agent={
            "code": "coder",
            "search": "researcher",
            "analyze": "analyst",
        }
    )

    # Create context with multiple agents
    context = TeamContext("team-1", "dynamic_router")
    context.set("coder", coder_agent)
    context.set("researcher", researcher_agent)
    context.set("analyst", analyst_agent)

    # Execute with automatic routing
    results = await formation.execute(agents, context, task)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult

logger = logging.getLogger(__name__)


class DynamicRouterFormation(BaseFormationStrategy):
    """Routes tasks to appropriate agents dynamically.

    Uses task analysis to determine agent capabilities required
    and routes to best-suited agent.

    Formation Pattern:
        Router → (Agent 1 | Agent 2 | Agent 3 | ...)

    Routing Strategy:
        1. If TaskAnalyzer available: Use category-based routing
        2. Fallback: Keyword-based routing
        3. Final fallback: First available agent

    SOLID: SRP (routing logic only), OCP (extensible routing strategies)

    Attributes:
        category_to_role: Maps task categories to agent roles
        keyword_to_agent: Maps keywords to agent IDs (fallback)

    Example:
        >>> formation = DynamicRouterFormation(
        ...     category_to_role={
        ...         "coding": "coder",
        ...         "research": "researcher",
        ...         "analysis": "analyst",
        ...     }
        ... )
        >>>
        >>> context = TeamContext("team-1", "dynamic_router")
        >>> context.set("coder", coder_agent)
        >>> context.set("researcher", researcher_agent)
        >>>
        >>> results = await formation.execute(agents, context, task)
        >>> selected_agent = results[0].metadata['selected_agent']
    """

    def __init__(
        self,
        category_to_role: Optional[Dict[str, str]] = None,
        keyword_to_agent: Optional[Dict[str, str]] = None,
    ):
        """Initialize the dynamic router formation.

        Args:
            category_to_role: Maps task categories to agent roles
                (default: standard mappings for coding, research, analysis, writing)
            keyword_to_agent: Maps keywords to agent IDs for fallback routing
                (default: standard keyword mappings)
        """
        # Default category to role mappings
        default_category_to_role = {
            "coding": "coder",
            "development": "coder",
            "programming": "coder",
            "research": "researcher",
            "search": "researcher",
            "investigation": "researcher",
            "analysis": "analyst",
            "analytics": "analyst",
            "writing": "writer",
            "documentation": "writer",
            "general": "generalist",
        }

        # Default keyword to agent mappings (fallback)
        default_keyword_to_agent = {
            "code": "coder",
            "function": "coder",
            "class": "coder",
            "search": "researcher",
            "find": "researcher",
            "look up": "researcher",
            "analyze": "analyst",
            "investigate": "analyst",
            "review": "analyst",
            "write": "writer",
            "document": "writer",
            "explain": "writer",
        }

        self.category_to_role = category_to_role or default_category_to_role
        self.keyword_to_agent = keyword_to_agent or default_keyword_to_agent

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute task by routing to appropriate agent.

        Args:
            agents: List of available agents
            context: Team context with agent references
            task: Task message to process

        Returns:
            List of MemberResult with metadata:
                - result: Result from selected agent
                - selected_agent: ID of selected agent
                - routing_method: How agent was selected (category/keyword/default)
                - routing_reason: Explanation for routing choice
        """
        # Select agent based on task analysis
        selected_agent, method, reason = self._select_agent(context, task.content)

        # Execute task with selected agent
        try:
            result = await selected_agent.execute(task.content, context=context.shared_state)

            return [
                MemberResult(
                    member_id=selected_agent.id,
                    success=True,
                    output=str(result) if result else "",
                    metadata={
                        "selected_agent": selected_agent.id,
                        "routing_method": method,
                        "routing_reason": reason,
                        "formation": "dynamic_router",
                    },
                )
            ]
        except Exception as e:
            logger.error(f"Agent {selected_agent.id} failed: {e}")
            return [
                MemberResult(
                    member_id=selected_agent.id,
                    success=False,
                    output="",
                    error=str(e),
                    metadata={
                        "selected_agent": selected_agent.id,
                        "routing_method": method,
                        "routing_reason": reason,
                    },
                )
            ]

    def _select_agent(self, context: TeamContext, task: str) -> tuple[Any, str, str]:
        """Select best agent for the task.

        Args:
            context: Team context with available agents
            task: Task description

        Returns:
            Tuple of (selected_agent, routing_method, routing_reason)
        """
        # Try task analyzer if available
        task_analyzer = context.get("task_analyzer")
        if task_analyzer:
            try:
                # Try to analyze task category
                category = self._analyze_task_category(task)

                if category:
                    # Map category to role
                    role = self.category_to_role.get(category)
                    if role:
                        agent = context.get(role)
                        if agent:
                            return agent, "category", f"Task category: {category}"
            except Exception as e:
                logger.debug(f"Task analysis failed: {e}")

        # Fallback to keyword-based routing
        return self._keyword_routing(context, task)

    def _analyze_task_category(self, task: str) -> Optional[str]:
        """Analyze task to determine category.

        Args:
            task: Task description

        Returns:
            Category string or None if unable to determine
        """
        task_lower = task.lower()

        # Check for category keywords
        if any(
            kw in task_lower
            for kw in ["code", "function", "class", "implement", "refactor", "debug"]
        ):
            return "coding"
        elif any(kw in task_lower for kw in ["search", "find", "look up", "investigate"]):
            return "research"
        elif any(kw in task_lower for kw in ["analyze", "review", "compare", "evaluate"]):
            return "analysis"
        elif any(kw in task_lower for kw in ["write", "document", "explain", "describe"]):
            return "writing"

        return None

    def _keyword_routing(self, context: TeamContext, task: str) -> tuple[Any, str, str]:
        """Fallback routing using keyword matching.

        Args:
            context: Team context with available agents
            task: Task description

        Returns:
            Tuple of (selected_agent, routing_method, routing_reason)
        """
        task_lower = task.lower()

        # Try keyword matches
        for keyword, agent_id in self.keyword_to_agent.items():
            if keyword in task_lower:
                agent = context.get(agent_id)
                if agent:
                    return agent, "keyword", f"Keyword match: '{keyword}'"

        # Final fallback: first available agent
        all_agents = list(context.shared_state.values())
        if all_agents:
            # Filter for agent-like objects
            for obj in all_agents:
                if hasattr(obj, "execute") and hasattr(obj, "id"):
                    return obj, "default", "First available agent"

        # No agent found - return error
        error_msg = "No agents available for routing"
        logger.error(error_msg)
        return None, "error", error_msg

    def validate_context(self, context: TeamContext) -> bool:
        """Validate that context has at least one agent.

        Args:
            context: Team context to validate

        Returns:
            True if context has at least one agent
        """
        # Check if any agents are registered in context
        agent_count = 0
        for value in context.shared_state.values():
            if hasattr(value, "execute") and hasattr(value, "id"):
                agent_count += 1

        if agent_count == 0:
            logger.warning("DynamicRouterFormation context has no agents")

        return agent_count > 0

    def supports_early_termination(self) -> bool:
        """Check if formation supports early termination.

        Dynamic router doesn't need early termination as it
        only executes one agent.

        Returns:
            False (no early termination needed)
        """
        return False


__all__ = ["DynamicRouterFormation"]
