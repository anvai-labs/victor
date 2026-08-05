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

"""Agent teams for coordinated multi-agent execution.

This module provides infrastructure for creating and coordinating teams of
specialized agents working toward shared goals. Similar to CrewAI's Crew
concept but built on Victor's existing SubAgent infrastructure.

Key Concepts:
- TeamFormation: How agents are organized (sequential, parallel, hierarchical, pipeline)
- TeamMember: Individual agent in a team with role and goal
- TeamCoordinator: Orchestrates team execution
- TeamMessageBus: Inter-agent communication

Example Usage:
    from victor.agent.teams import (
        TeamConfig,
        TeamMember,
        TeamFormation,
        TeamCoordinator,
    )
    from victor.agent.subagents import SubAgentRole

    # Define team members
    members = [
        TeamMember(
            id="researcher",
            role=SubAgentRole.RESEARCHER,
            name="Code Researcher",
            goal="Find all authentication-related code",
        ),
        TeamMember(
            id="planner",
            role=SubAgentRole.PLANNER,
            name="Implementation Planner",
            goal="Create implementation plan based on research",
        ),
        TeamMember(
            id="executor",
            role=SubAgentRole.EXECUTOR,
            name="Code Implementer",
            goal="Implement the planned changes",
        ),
    ]

    # Create team configuration
    config = TeamConfig(
        name="Auth Refactoring Team",
        goal="Refactor authentication to use JWT",
        members=members,
        formation=TeamFormation.PIPELINE,
    )

    # Execute team
    coordinator = TeamCoordinator(orchestrator)
    result = await coordinator.execute_team(config)

    print(f"Success: {result.success}")
    print(f"Total tool calls: {result.total_tool_calls}")
"""

# Import canonical types from victor.teams.types (single source of truth)
from victor.teams.types import (
    AgentMessage,
    MemberResult,
    MemoryConfig,
    MemberStatus,
    MessageType,
    TeamConfig,
    TeamFormation,
    TeamMember,
    TeamResult,
)
from victor.agent.teams.coordinator import TeamCoordinator
from victor.agent.teams.communication import (
    TeamMessageBus,
    TeamSharedMemory,
)
from victor.agent.teams.metrics import (
    TaskCategory,
    TeamMetrics,
    CompositionStats,
    categorize_task,
)
from victor.agent.teams.learner import (
    TeamRecommendation,
    TeamCompositionLearner,
    get_team_learner,
    DEFAULT_COMPOSITIONS,
)

__all__ = [
    # Core team types
    "MemoryConfig",
    "TeamFormation",
    "MemberStatus",
    "TeamMember",
    "TeamConfig",
    "MemberResult",
    "TeamResult",
    # Coordination
    "TeamCoordinator",
    # Communication
    "MessageType",
    "AgentMessage",
    "TeamMessageBus",
    "TeamSharedMemory",
    # Metrics
    "TaskCategory",
    "TeamMetrics",
    "CompositionStats",
    "categorize_task",
    # Learning
    "TeamRecommendation",
    "TeamCompositionLearner",
    "get_team_learner",
    "DEFAULT_COMPOSITIONS",
]
