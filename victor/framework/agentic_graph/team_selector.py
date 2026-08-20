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

"""Team formation selector for StateGraph integration (Phase 4 consolidation).

This module provides logic to select the appropriate team formation
based on task characteristics, complexity, and team size.

Key Components:
- FormationCriteria: Dataclass for formation selection inputs
- select_formation: Main selection function
- DEFAULT_FORMATION: Fallback formation when no criteria match

Selection Rules:
1. Single agent (team_size <= 1) → SEQUENTIAL
2. Consensus required → CONSENSUS
3. Large teams (team_size >= 5) → HIERARCHICAL
4. Workflow task with dependencies → PIPELINE
5. Independent tasks → PARALLEL
6. Dependent tasks → SEQUENTIAL
7. High complexity → HIERARCHICAL
8. Task-specific overrides (debugging → SEQUENTIAL, research → PARALLEL)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from victor.framework.agentic_graph.state import AgenticLoopStateModel
from victor.teams.types import TeamFormation

logger = logging.getLogger(__name__)

# Default formation when no criteria match
DEFAULT_FORMATION = TeamFormation.SEQUENTIAL

# Team size thresholds
LARGE_TEAM_THRESHOLD = 5
MEDIUM_TEAM_THRESHOLD = 3

# Task types that prefer specific formations
TASK_TYPE_OVERRIDES = {
    "debugging": TeamFormation.SEQUENTIAL,
    "research": TeamFormation.PARALLEL,
    "analysis": TeamFormation.PARALLEL,
    "code_generation": TeamFormation.PIPELINE,
    "refactoring": TeamFormation.PIPELINE,
    "testing": TeamFormation.PARALLEL,
    "documentation": TeamFormation.PARALLEL,
}


@dataclass
class FormationCriteria:
    """Criteria for selecting team formation.

    Attributes:
        complexity: Task complexity (low, medium, high)
        task_type: Type of task (general, feature, debugging, etc.)
        team_size: Number of agents in the team
        has_dependencies: Whether tasks have dependencies
        requires_consensus: Whether consensus is required
    """

    complexity: str = "medium"
    task_type: str = "general"
    team_size: int = 1
    has_dependencies: bool = False
    requires_consensus: bool = False

    @classmethod
    def from_state(cls, state: Union[AgenticLoopStateModel, dict]) -> "FormationCriteria":
        """Create FormationCriteria from AgenticLoopStateModel.

        Args:
            state: Current agentic loop state

        Returns:
            FormationCriteria instance
        """
        context = {}
        if isinstance(state, AgenticLoopStateModel):
            context = state.context or {}
        elif isinstance(state, dict):
            context = state.get("context", state)

        return cls(
            complexity=context.get("complexity", "medium"),
            task_type=context.get("task_type", "general"),
            team_size=context.get("team_size", 1),
            has_dependencies=context.get("has_dependencies", False),
            requires_consensus=context.get("requires_consensus", False),
        )


def select_formation(
    criteria: Union[FormationCriteria, AgenticLoopStateModel, dict],
) -> TeamFormation:
    """Select appropriate team formation based on criteria.

    Selection priority (highest to lowest):
    1. Consensus requirement → CONSENSUS
    2. Task type override → Specific formation
    3. Large teams → HIERARCHICAL
    4. Workflow with dependencies → PIPELINE
    5. Independent tasks → PARALLEL
    6. Default → SEQUENTIAL

    Args:
        criteria: Formation criteria or state object

    Returns:
        Selected TeamFormation
    """
    # Convert to FormationCriteria if needed
    if not isinstance(criteria, FormationCriteria):
        criteria = FormationCriteria.from_state(criteria)

    # Normalize team size
    team_size = max(0, criteria.team_size)

    # Priority 1: Consensus requirement
    if criteria.requires_consensus:
        logger.debug("Selected CONSENSUS formation (consensus required)")
        return TeamFormation.CONSENSUS

    # Priority 2: Task type overrides
    if criteria.task_type in TASK_TYPE_OVERRIDES:
        formation = TASK_TYPE_OVERRIDES[criteria.task_type]
        logger.debug(f"Selected {formation.value} formation (task type: {criteria.task_type})")
        return formation

    # Priority 3: Single agent
    if team_size <= 1:
        logger.debug("Selected SEQUENTIAL formation (single agent)")
        return TeamFormation.SEQUENTIAL

    # Priority 4: Large teams need hierarchical coordination
    if team_size >= LARGE_TEAM_THRESHOLD:
        logger.debug(f"Selected HIERARCHICAL formation (large team: {team_size})")
        return TeamFormation.HIERARCHICAL

    # Priority 5: Workflow tasks with dependencies prefer pipeline
    if criteria.task_type == "workflow" and criteria.has_dependencies:
        logger.debug("Selected PIPELINE formation (workflow with dependencies)")
        return TeamFormation.PIPELINE

    # Priority 6: High complexity benefits from hierarchical coordination
    if criteria.complexity == "high" and team_size >= MEDIUM_TEAM_THRESHOLD:
        logger.debug("Selected HIERARCHICAL formation (high complexity)")
        return TeamFormation.HIERARCHICAL

    # Priority 7: Independent tasks can run in parallel
    if not criteria.has_dependencies and team_size >= MEDIUM_TEAM_THRESHOLD:
        logger.debug("Selected PARALLEL formation (independent tasks)")
        return TeamFormation.PARALLEL

    # Default: sequential execution
    logger.debug("Selected SEQUENTIAL formation (default)")
    return DEFAULT_FORMATION
