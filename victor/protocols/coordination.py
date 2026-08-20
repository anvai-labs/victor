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

"""Coordination protocols for mode-team-workflow integration.

This module defines SOLID-compliant protocols for coordinating the selection
and suggestion of teams and workflows based on agent mode and task analysis.

Design Principles:
- Single Responsibility: Each protocol has one clear purpose
- Open/Closed: Extend via new implementations, not modifications
- Liskov Substitution: All implementations are interchangeable
- Interface Segregation: Small, focused protocols
- Dependency Inversion: Depend on protocols, not implementations

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Coordination Layer                            │
    ├─────────────────────────────────────────────────────────────────┤
    │  TaskAnalyzer  →  CoordinationAdvisorRuntime  →  Orchestrator   │
    │       ↓                       ↓                        ↓         │
    │  TaskAnalysis         CoordinationSuggestion      Execution      │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from victor.protocols.coordination import (
        CoordinationAdvisorProtocol,
        CoordinationSuggestion,
        TeamSelectionStrategyProtocol,
    )

    class MyCoordinator(CoordinationAdvisorProtocol):
        def suggest_for_task(self, task_type, complexity, mode):
            ...
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.teams import TeamConfig, TeamFormation


class TeamSuggestionAction(str, Enum):
    """Action to take for team suggestion.

    Determines how the system should respond when a team is suggested.
    """

    NONE = "none"
    """No team suggestion - single agent execution."""

    SUGGEST = "suggest"
    """Suggest team to user, await confirmation."""

    AUTO_SPAWN = "auto_spawn"
    """Automatically spawn the suggested team."""

    REQUIRE_APPROVAL = "require_approval"
    """Suggest team and require explicit approval."""


class ComplexityLevel(str, Enum):
    """Task complexity levels for coordination decisions.

    Maps to existing complexity classifications but provides
    specific semantics for team/workflow coordination.
    """

    TRIVIAL = "trivial"
    """Single-step task, no coordination needed."""

    LOW = "low"
    """Simple task, single agent sufficient."""

    MEDIUM = "medium"
    """Moderate task, team suggestion optional."""

    HIGH = "high"
    """Complex task, team strongly recommended."""

    EXTREME = "extreme"
    """Very complex task, team with extended budget."""

    @classmethod
    def from_string(cls, value: str) -> "ComplexityLevel":
        """Convert string to ComplexityLevel.

        Args:
            value: String complexity value

        Returns:
            ComplexityLevel enum value
        """
        mapping = {
            "trivial": cls.TRIVIAL,
            "low": cls.LOW,
            "simple": cls.LOW,
            "medium": cls.MEDIUM,
            "moderate": cls.MEDIUM,
            "high": cls.HIGH,
            "complex": cls.HIGH,
            "extreme": cls.EXTREME,
            "very_high": cls.EXTREME,
        }
        return mapping.get(value.lower(), cls.MEDIUM)


@dataclass
class TeamRecommendation:
    """Recommendation for a specific team.

    Attributes:
        team_name: Name of the recommended team
        confidence: Confidence score 0.0-1.0
        reason: Explanation for recommendation
        formation: Suggested team formation
        suggested_budget: Suggested total tool budget
        role_distribution: Suggested role counts
        source: Source of recommendation (rule, learning, hybrid)
    """

    team_name: str
    confidence: float
    reason: str
    formation: Optional[str] = None
    suggested_budget: Optional[int] = None
    role_distribution: Optional[Dict[str, int]] = None
    source: str = "rule"

    def __lt__(self, other: "TeamRecommendation") -> bool:
        """Compare by confidence for sorting."""
        return self.confidence < other.confidence


@dataclass
class WorkflowRecommendation:
    """Recommendation for a specific workflow.

    Attributes:
        workflow_name: Name of the recommended workflow
        confidence: Confidence score 0.0-1.0
        reason: Explanation for recommendation
        trigger_condition: What triggered this recommendation
        estimated_steps: Estimated workflow steps
    """

    workflow_name: str
    confidence: float
    reason: str
    trigger_condition: Optional[str] = None
    estimated_steps: Optional[int] = None


@dataclass
class CoordinationSuggestion:
    """Complete coordination suggestion for a task.

    Encapsulates all recommendations for team and workflow selection
    based on task analysis and mode configuration.

    Attributes:
        mode: Current agent mode
        task_type: Classified task type
        complexity: Task complexity level
        action: Suggested action for team spawning
        team_recommendations: Ordered list of team recommendations
        workflow_recommendations: Ordered list of workflow recommendations
        system_prompt_additions: Additional prompt content for the mode
        tool_priorities: Tool priority adjustments
        metadata: Additional coordination metadata
    """

    mode: str
    task_type: str
    complexity: ComplexityLevel
    action: TeamSuggestionAction = TeamSuggestionAction.NONE
    team_recommendations: List[TeamRecommendation] = field(default_factory=list)
    workflow_recommendations: List[WorkflowRecommendation] = field(default_factory=list)
    system_prompt_additions: str = ""
    tool_priorities: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_team_suggestion(self) -> bool:
        """Check if any team is suggested."""
        return len(self.team_recommendations) > 0

    @property
    def has_workflow_suggestion(self) -> bool:
        """Check if any workflow is suggested."""
        return len(self.workflow_recommendations) > 0

    @property
    def primary_team(self) -> Optional[TeamRecommendation]:
        """Get the primary (highest confidence) team recommendation."""
        if self.team_recommendations:
            return max(self.team_recommendations, key=lambda r: r.confidence)
        return None

    @property
    def primary_workflow(self) -> Optional[WorkflowRecommendation]:
        """Get the primary (highest confidence) workflow recommendation."""
        if self.workflow_recommendations:
            return max(self.workflow_recommendations, key=lambda r: r.confidence)
        return None

    @property
    def should_spawn_team(self) -> bool:
        """Check if team should be auto-spawned."""
        return self.action == TeamSuggestionAction.AUTO_SPAWN and self.has_team_suggestion

    @property
    def should_suggest_team(self) -> bool:
        """Check if team should be suggested to user."""
        return (
            self.action
            in (
                TeamSuggestionAction.SUGGEST,
                TeamSuggestionAction.REQUIRE_APPROVAL,
            )
            and self.has_team_suggestion
        )


@runtime_checkable
class CoordinationAdvisorProtocol(Protocol):
    """Canonical protocol for coordination advice across teams and workflows.

    Implementations coordinate between:
    1. Agent modes (explore/plan/build)
    2. Team specifications (from verticals)
    3. Workflow definitions (from YAML)
    4. Task analysis results

    Example:
        class DefaultCoordinator(CoordinationAdvisorProtocol):
            def suggest_for_task(
                self,
                task_type: str,
                complexity: str,
                mode: str,
            ) -> CoordinationSuggestion:
                level = ComplexityLevel.from_string(complexity)
                action = self._determine_action(level, mode)
                teams = self._get_matching_teams(task_type)
                return CoordinationSuggestion(
                    mode=mode,
                    task_type=task_type,
                    complexity=level,
                    action=action,
                    team_recommendations=teams,
                )
    """

    @abstractmethod
    def suggest_for_task(
        self,
        task_type: str,
        complexity: str,
        mode: str,
    ) -> CoordinationSuggestion:
        """Suggest teams and workflows for a task.

        This is the main coordination method called by TaskAnalyzer
        after classifying a task.

        Args:
            task_type: Classified task type (feature, bugfix, refactor, etc.)
            complexity: Complexity level string
            mode: Current agent mode (explore, plan, build)

        Returns:
            CoordinationSuggestion with recommendations
        """
        ...

    @abstractmethod
    def get_default_workflow(self, mode: str) -> Optional[str]:
        """Get the default workflow for a mode.

        Args:
            mode: Agent mode name

        Returns:
            Default workflow name or None
        """
        ...

    @abstractmethod
    def get_suggested_teams(
        self,
        task_type: str,
        complexity: str,
    ) -> List[TeamRecommendation]:
        """Get team suggestions without mode context.

        Useful for querying available teams for a task type.

        Args:
            task_type: Classified task type
            complexity: Complexity level string

        Returns:
            List of team recommendations
        """
        ...

    @abstractmethod
    def get_action_for_complexity(
        self,
        complexity: str,
        mode: str,
    ) -> TeamSuggestionAction:
        """Determine action based on complexity and mode.

        Args:
            complexity: Complexity level string
            mode: Agent mode name

        Returns:
            TeamSuggestionAction to take
        """
        ...


@runtime_checkable
class TeamSelectionStrategyProtocol(Protocol):
    """Protocol for team selection strategies.

    Implementations provide different approaches to selecting
    teams from available options:
    - RuleBased: Match task keywords to team names
    - LearningBased: Use TeamCompositionLearner
    - Hybrid: Combine rules with learning

    Example:
        class RuleBasedStrategy(TeamSelectionStrategyProtocol):
            def select(self, task_type, complexity, available_teams):
                # Match task type to team name
                for name, spec in available_teams.items():
                    if task_type in name:
                        return name
                return None
    """

    @abstractmethod
    def select(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
    ) -> Optional[str]:
        """Select a single team for the task.

        Args:
            task_type: Classified task type
            complexity: Complexity level string
            available_teams: Dict of team names to specs

        Returns:
            Selected team name or None
        """
        ...

    @abstractmethod
    def recommend(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
        top_k: int = 3,
    ) -> List[TeamRecommendation]:
        """Recommend multiple teams for the task.

        Args:
            task_type: Classified task type
            complexity: Complexity level string
            available_teams: Dict of team names to specs
            top_k: Maximum recommendations to return

        Returns:
            List of team recommendations ordered by confidence
        """
        ...


@runtime_checkable
class WorkflowSelectionStrategyProtocol(Protocol):
    """Protocol for workflow selection strategies.

    Implementations select workflows based on task characteristics.
    """

    @abstractmethod
    def select(
        self,
        task_type: str,
        mode: str,
        available_workflows: Dict[str, Any],
    ) -> Optional[str]:
        """Select a workflow for the task.

        Args:
            task_type: Classified task type
            mode: Current agent mode
            available_workflows: Dict of workflow names to definitions

        Returns:
            Selected workflow name or None
        """
        ...

    @abstractmethod
    def recommend(
        self,
        task_type: str,
        mode: str,
        available_workflows: Dict[str, Any],
        top_k: int = 3,
    ) -> List[WorkflowRecommendation]:
        """Recommend workflows for the task.

        Args:
            task_type: Classified task type
            mode: Current agent mode
            available_workflows: Dict of workflow names to definitions
            top_k: Maximum recommendations to return

        Returns:
            List of workflow recommendations ordered by confidence
        """
        ...


@runtime_checkable
class CoordinationConfigProviderProtocol(Protocol):
    """Protocol for providing coordination configuration.

    Implementations load configuration from YAML or other sources.
    """

    @abstractmethod
    def get_mode_config(self, mode: str) -> Dict[str, Any]:
        """Get configuration for a mode.

        Args:
            mode: Agent mode name

        Returns:
            Mode configuration dict
        """
        ...

    @abstractmethod
    def get_complexity_thresholds(self, mode: str) -> Dict[str, TeamSuggestionAction]:
        """Get complexity thresholds for a mode.

        Args:
            mode: Agent mode name

        Returns:
            Dict mapping complexity levels to actions
        """
        ...

    @abstractmethod
    def get_default_teams(self, mode: str) -> List[str]:
        """Get default team names for a mode.

        Args:
            mode: Agent mode name

        Returns:
            List of team names
        """
        ...

    @abstractmethod
    def get_default_workflows(self, mode: str) -> List[str]:
        """Get default workflow names for a mode.

        Args:
            mode: Agent mode name

        Returns:
            List of workflow names
        """
        ...


__all__ = [
    # Enums
    "TeamSuggestionAction",
    "ComplexityLevel",
    # Data classes
    "TeamRecommendation",
    "WorkflowRecommendation",
    "CoordinationSuggestion",
    # Protocols
    "CoordinationAdvisorProtocol",
    "TeamSelectionStrategyProtocol",
    "WorkflowSelectionStrategyProtocol",
    "CoordinationConfigProviderProtocol",
]
