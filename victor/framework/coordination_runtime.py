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

"""Framework-first coordination recommendation helpers."""

from __future__ import annotations

import logging
from inspect import getattr_static
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from victor.framework.team_runtime import VerticalCoordinationCatalog
from victor.protocols.coordination import (
    CoordinationAdvisorProtocol,
    CoordinationSuggestion,
    ComplexityLevel,
    TeamRecommendation,
    TeamSelectionStrategyProtocol,
    TeamSuggestionAction,
    WorkflowRecommendation,
    WorkflowSelectionStrategyProtocol,
)

if TYPE_CHECKING:
    from victor.agent.teams.learner import TeamCompositionLearner

logger = logging.getLogger(__name__)


@dataclass
class ModeCoordinationConfig:
    """Configuration for a specific mode."""

    mode_name: str
    default_workflows: List[str] = field(default_factory=list)
    default_teams: List[str] = field(default_factory=list)
    team_suggestion_enabled: bool = True
    complexity_thresholds: Dict[str, TeamSuggestionAction] = field(
        default_factory=lambda: {
            "trivial": TeamSuggestionAction.NONE,
            "low": TeamSuggestionAction.NONE,
            "medium": TeamSuggestionAction.SUGGEST,
            "high": TeamSuggestionAction.SUGGEST,
            "extreme": TeamSuggestionAction.AUTO_SPAWN,
        }
    )
    tool_priorities: Dict[str, float] = field(default_factory=dict)
    system_prompt_addition: str = ""


class VerticalCoordinationAdvisor(CoordinationAdvisorProtocol):
    """Framework-native advisor for team/workflow coordination on one vertical context."""

    def __init__(
        self,
        *,
        vertical_context: Optional[Any] = None,
        team_selector: Optional[Any] = None,
        workflow_selector: Optional[Any] = None,
        mode_configs: Optional[Dict[str, ModeCoordinationConfig]] = None,
    ) -> None:
        self._vertical_context = vertical_context
        self._team_selector = team_selector or HybridTeamSelector(
            RuleBasedTeamSelector(),
            LearningBasedTeamSelector(),
        )
        self._workflow_selector = workflow_selector or RuleBasedWorkflowSelector()
        self._mode_configs = mode_configs or DEFAULT_MODE_CONFIGS.copy()

    def set_vertical_context(self, context: Any) -> None:
        """Update the bound vertical context."""
        self._vertical_context = context

    def set_team_learner(self, learner: "TeamCompositionLearner") -> None:
        """Bind a team-composition learner to the configured selector when supported."""
        if isinstance(self._team_selector, HybridTeamSelector):
            self._team_selector._learning_selector.set_learner(learner)
        elif isinstance(self._team_selector, LearningBasedTeamSelector):
            self._team_selector.set_learner(learner)

    def suggest_for_task(
        self,
        task_type: str,
        complexity: str,
        mode: str,
    ) -> CoordinationSuggestion:
        """Suggest teams and workflows for a task."""
        return build_coordination_suggestion(
            task_type=task_type,
            complexity=complexity,
            mode=mode,
            coordination_catalog=self._get_coordination_catalog(),
            team_selector=self._team_selector,
            workflow_selector=self._workflow_selector,
            mode_configs=self._mode_configs,
        )

    def get_default_workflow(self, mode: str) -> Optional[str]:
        """Get the default workflow for a mode."""
        return resolve_default_workflow_for_mode(
            mode,
            coordination_catalog=self._get_coordination_catalog(),
            mode_configs=self._mode_configs,
        )

    def get_suggested_teams(
        self,
        task_type: str,
        complexity: str,
    ) -> List[TeamRecommendation]:
        """Get team suggestions for a task."""
        return recommend_teams_for_catalog(
            task_type=task_type,
            complexity=complexity,
            coordination_catalog=self._get_coordination_catalog(),
            team_selector=self._team_selector,
        )

    def get_action_for_complexity(
        self,
        complexity: str,
        mode: str,
    ) -> TeamSuggestionAction:
        """Determine the action for a task complexity/mode pair."""
        return get_action_for_complexity(
            complexity,
            mode,
            mode_configs=self._mode_configs,
        )

    def get_workflow_recommendations(
        self,
        task_type: str,
        mode: str,
    ) -> List[WorkflowRecommendation]:
        """Get workflow recommendations for a task."""
        return recommend_workflows_for_catalog(
            task_type=task_type,
            mode=mode,
            coordination_catalog=self._get_coordination_catalog(),
            workflow_selector=self._workflow_selector,
        )

    def _get_coordination_catalog(self) -> VerticalCoordinationCatalog:
        """Resolve the shared team/workflow catalog for the current vertical context."""
        from victor.framework.team_runtime import resolve_vertical_coordination_catalog

        return resolve_vertical_coordination_catalog(self._vertical_context)


def create_vertical_coordination_advisor(
    *,
    vertical_context: Optional[Any] = None,
    team_learner: Optional["TeamCompositionLearner"] = None,
    selection_strategy: str = "hybrid",
    workflow_selector: Optional[Any] = None,
    mode_configs: Optional[Dict[str, ModeCoordinationConfig]] = None,
) -> VerticalCoordinationAdvisor:
    """Create a framework-native coordination advisor with a configured selector policy."""
    if selection_strategy == "rule":
        team_selector = RuleBasedTeamSelector()
    elif selection_strategy == "learning":
        team_selector = LearningBasedTeamSelector(team_learner)
    else:
        learning_selector = LearningBasedTeamSelector(team_learner)
        team_selector = HybridTeamSelector(
            RuleBasedTeamSelector(),
            learning_selector,
        )

    return VerticalCoordinationAdvisor(
        vertical_context=vertical_context,
        team_selector=team_selector,
        workflow_selector=workflow_selector or RuleBasedWorkflowSelector(),
        mode_configs=mode_configs,
    )


@dataclass(frozen=True)
class CatalogCoordinationSuggestion:
    """Serializable recommendation resolved from a shared coordination catalog."""

    vertical: str
    suggestion: CoordinationSuggestion
    available_teams: Tuple[str, ...] = ()
    available_workflows: Tuple[str, ...] = ()
    default_workflow: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return serialize_coordination_suggestion(
            self.suggestion,
            vertical=self.vertical,
            available_teams=self.available_teams,
            available_workflows=self.available_workflows,
            default_workflow=self.default_workflow,
        )


DEFAULT_MODE_CONFIGS: Dict[str, ModeCoordinationConfig] = {
    "explore": ModeCoordinationConfig(
        mode_name="explore",
        default_workflows=[],
        default_teams=["research_team"],
        team_suggestion_enabled=False,
        complexity_thresholds={
            "trivial": TeamSuggestionAction.NONE,
            "low": TeamSuggestionAction.NONE,
            "medium": TeamSuggestionAction.NONE,
            "high": TeamSuggestionAction.SUGGEST,
            "extreme": TeamSuggestionAction.SUGGEST,
        },
    ),
    "plan": ModeCoordinationConfig(
        mode_name="plan",
        default_workflows=["planning_workflow"],
        default_teams=["research_team", "analysis_team"],
        team_suggestion_enabled=True,
        complexity_thresholds={
            "trivial": TeamSuggestionAction.NONE,
            "low": TeamSuggestionAction.NONE,
            "medium": TeamSuggestionAction.SUGGEST,
            "high": TeamSuggestionAction.SUGGEST,
            "extreme": TeamSuggestionAction.SUGGEST,
        },
    ),
    "review": ModeCoordinationConfig(
        mode_name="review",
        default_workflows=["code_review"],
        default_teams=["review_team", "code_review_team", "qa_team"],
        team_suggestion_enabled=True,
        complexity_thresholds={
            "trivial": TeamSuggestionAction.NONE,
            "low": TeamSuggestionAction.NONE,
            "medium": TeamSuggestionAction.SUGGEST,
            "high": TeamSuggestionAction.SUGGEST,
            "extreme": TeamSuggestionAction.AUTO_SPAWN,
        },
    ),
    "delegate": ModeCoordinationConfig(
        mode_name="delegate",
        default_workflows=["planning_workflow", "feature_implementation"],
        default_teams=["feature_team", "bug_fix_team", "review_team"],
        team_suggestion_enabled=True,
        complexity_thresholds={
            "trivial": TeamSuggestionAction.NONE,
            "low": TeamSuggestionAction.SUGGEST,
            "medium": TeamSuggestionAction.AUTO_SPAWN,
            "high": TeamSuggestionAction.AUTO_SPAWN,
            "extreme": TeamSuggestionAction.AUTO_SPAWN,
        },
    ),
    "build": ModeCoordinationConfig(
        mode_name="build",
        default_workflows=["feature_implementation", "bug_fix"],
        default_teams=["feature_team", "bug_fix_team", "refactoring_team"],
        team_suggestion_enabled=True,
        complexity_thresholds={
            "trivial": TeamSuggestionAction.NONE,
            "low": TeamSuggestionAction.NONE,
            "medium": TeamSuggestionAction.SUGGEST,
            "high": TeamSuggestionAction.AUTO_SPAWN,
            "extreme": TeamSuggestionAction.AUTO_SPAWN,
        },
    ),
}


class RuleBasedTeamSelector:
    """Rule-based team selection using keyword matching."""

    TASK_TEAM_PATTERNS: Dict[str, List[str]] = {
        "feature": ["feature_team", "implementation_team"],
        "bugfix": ["bug_fix_team", "debugging_team"],
        "bug": ["bug_fix_team", "debugging_team"],
        "refactor": ["refactoring_team", "code_review_team"],
        "review": ["review_team", "code_review_team"],
        "test": ["testing_team", "qa_team"],
        "documentation": ["documentation_team", "docs_team"],
        "research": ["research_team", "deep_research_team"],
        "analysis": ["analysis_team", "eda_team"],
        "deployment": ["deployment_team", "devops_team"],
        "container": ["container_team", "docker_team"],
        "monitoring": ["monitoring_team", "observability_team"],
        "visualization": ["visualization_team", "charting_team"],
        "ml": ["ml_team", "machine_learning_team"],
        "data": ["data_team", "cleaning_team", "eda_team"],
    }

    def select(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
    ) -> Optional[str]:
        """Select a team based on task type matching."""
        task_lower = task_type.lower()
        for pattern, team_names in self.TASK_TEAM_PATTERNS.items():
            if pattern in task_lower:
                for team_name in team_names:
                    if team_name in available_teams:
                        return team_name

        for team_name in available_teams:
            if task_lower in team_name.lower():
                return team_name

        return None

    def recommend(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
        top_k: int = 3,
    ) -> List[TeamRecommendation]:
        """Recommend teams with confidence scores."""
        recommendations = []
        task_lower = task_type.lower()

        for team_name, spec in available_teams.items():
            confidence = 0.0
            reason = ""
            for pattern, team_names in self.TASK_TEAM_PATTERNS.items():
                if pattern in task_lower and team_name in team_names:
                    confidence = max(confidence, 0.8)
                    reason = f"Task type '{task_type}' matches pattern '{pattern}'"
                    break

            if task_lower in team_name.lower():
                confidence = max(confidence, 0.6)
                reason = reason or "Team name contains task type"

            if hasattr(spec, "description") and task_lower in spec.description.lower():
                confidence = max(confidence, 0.5)
                reason = reason or "Team description matches task"

            if confidence > 0:
                recommendations.append(
                    TeamRecommendation(
                        team_name=team_name,
                        confidence=confidence,
                        reason=reason,
                        formation=getattr(spec, "formation", None),
                        source="rule",
                    )
                )

        recommendations.sort(reverse=True)
        return recommendations[:top_k]


class LearningBasedTeamSelector:
    """Learning-based team selection using TeamCompositionLearner."""

    def __init__(self, learner: Optional["TeamCompositionLearner"] = None):
        self._learner = learner

    def set_learner(self, learner: "TeamCompositionLearner") -> None:
        """Set the learner instance."""
        self._learner = learner

    def select(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
    ) -> Optional[str]:
        """Select team using learner recommendations."""
        if self._learner is None:
            return None

        try:
            recommendation = self._learner.get_recommendation(task_type)
            if recommendation and not recommendation.is_baseline:
                return self._find_team_for_recommendation(recommendation, available_teams)
        except Exception as exc:
            logger.debug("Learner selection failed: %s", exc)

        return None

    def recommend(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
        top_k: int = 3,
    ) -> List[TeamRecommendation]:
        """Recommend teams using learner feedback."""
        if self._learner is None:
            return []

        try:
            recommendation = self._learner.get_recommendation(task_type)
            if recommendation:
                return [
                    TeamRecommendation(
                        team_name=self._find_team_for_recommendation(
                            recommendation, available_teams
                        )
                        or "feature_team",
                        confidence=recommendation.confidence,
                        reason=recommendation.reason,
                        formation=(
                            recommendation.formation.value if recommendation.formation else None
                        ),
                        suggested_budget=recommendation.suggested_budget,
                        role_distribution=recommendation.role_distribution,
                        source="learning",
                    )
                ]
        except Exception as exc:
            logger.debug("Learner recommendation failed: %s", exc)

        return []

    def _find_team_for_recommendation(
        self,
        recommendation: Any,
        available_teams: Dict[str, Any],
    ) -> Optional[str]:
        """Find team matching learner recommendation."""
        if hasattr(recommendation, "formation") and recommendation.formation:
            formation_name = (
                recommendation.formation.value
                if hasattr(recommendation.formation, "value")
                else str(recommendation.formation)
            )
            for team_name, spec in available_teams.items():
                if hasattr(spec, "formation"):
                    spec_formation = (
                        spec.formation.value
                        if hasattr(spec.formation, "value")
                        else str(spec.formation)
                    )
                    if spec_formation == formation_name:
                        return team_name

        if available_teams:
            return next(iter(available_teams))
        return None


class HybridTeamSelector:
    """Hybrid team selection combining rules and learning."""

    def __init__(
        self,
        rule_selector: RuleBasedTeamSelector,
        learning_selector: LearningBasedTeamSelector,
        rule_weight: float = 0.4,
        learning_weight: float = 0.6,
    ):
        self._rule_selector = rule_selector
        self._learning_selector = learning_selector
        self._rule_weight = rule_weight
        self._learning_weight = learning_weight

    def select(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
    ) -> Optional[str]:
        """Select team using hybrid approach."""
        learning_pick = self._learning_selector.select(task_type, complexity, available_teams)
        if learning_pick:
            return learning_pick
        return self._rule_selector.select(task_type, complexity, available_teams)

    def recommend(
        self,
        task_type: str,
        complexity: str,
        available_teams: Dict[str, Any],
        top_k: int = 3,
    ) -> List[TeamRecommendation]:
        """Recommend teams using hybrid approach."""
        rule_recs = self._rule_selector.recommend(task_type, complexity, available_teams, top_k)
        learning_recs = self._learning_selector.recommend(
            task_type, complexity, available_teams, top_k
        )

        combined: Dict[str, TeamRecommendation] = {}
        for rec in rule_recs:
            combined[rec.team_name] = TeamRecommendation(
                team_name=rec.team_name,
                confidence=rec.confidence * self._rule_weight,
                reason=rec.reason,
                formation=rec.formation,
                source="hybrid-rule",
            )

        for rec in learning_recs:
            if rec.team_name in combined:
                existing = combined[rec.team_name]
                combined[rec.team_name] = TeamRecommendation(
                    team_name=rec.team_name,
                    confidence=(existing.confidence + rec.confidence * self._learning_weight),
                    reason=f"{existing.reason}; {rec.reason}",
                    formation=rec.formation or existing.formation,
                    suggested_budget=rec.suggested_budget,
                    role_distribution=rec.role_distribution,
                    source="hybrid",
                )
            else:
                combined[rec.team_name] = TeamRecommendation(
                    team_name=rec.team_name,
                    confidence=rec.confidence * self._learning_weight,
                    reason=rec.reason,
                    formation=rec.formation,
                    suggested_budget=rec.suggested_budget,
                    role_distribution=rec.role_distribution,
                    source="hybrid-learning",
                )

        return sorted(combined.values(), reverse=True)[:top_k]


class RuleBasedWorkflowSelector:
    """Rule-based workflow selection."""

    TASK_WORKFLOW_PATTERNS: Dict[str, List[str]] = {
        "feature": ["feature_implementation", "quick_feature"],
        "bugfix": ["bug_fix", "quick_fix"],
        "bug": ["bug_fix", "quick_fix"],
        "refactor": ["refactoring", "code_review"],
        "review": ["code_review", "pr_review", "quick_review"],
        "test": ["testing", "test_coverage"],
    }

    def select(
        self,
        task_type: str,
        mode: str,
        available_workflows: Dict[str, Any],
    ) -> Optional[str]:
        """Select workflow based on task type and mode."""
        task_lower = task_type.lower()
        for pattern, workflow_names in self.TASK_WORKFLOW_PATTERNS.items():
            if pattern in task_lower:
                for name in workflow_names:
                    if name in available_workflows:
                        return name
        return None

    def recommend(
        self,
        task_type: str,
        mode: str,
        available_workflows: Dict[str, Any],
        top_k: int = 3,
    ) -> List[WorkflowRecommendation]:
        """Recommend workflows with confidence."""
        recommendations = []
        task_lower = task_type.lower()

        for name, workflow in available_workflows.items():
            confidence = 0.0
            reason = ""
            for pattern, workflow_names in self.TASK_WORKFLOW_PATTERNS.items():
                if pattern in task_lower and name in workflow_names:
                    confidence = max(confidence, 0.8)
                    reason = f"Task type '{task_type}' matches workflow pattern"
                    break

            if task_lower in name.lower():
                confidence = max(confidence, 0.6)
                reason = reason or "Workflow name matches task"

            if confidence > 0:
                recommendations.append(
                    WorkflowRecommendation(
                        workflow_name=name,
                        confidence=confidence,
                        reason=reason,
                    )
                )

        recommendations.sort(key=lambda rec: rec.confidence, reverse=True)
        return recommendations[:top_k]


def get_action_for_complexity(
    complexity: str,
    mode: str,
    *,
    mode_configs: Optional[Dict[str, ModeCoordinationConfig]] = None,
) -> TeamSuggestionAction:
    """Determine action based on complexity and mode."""
    resolved_mode_configs = mode_configs or DEFAULT_MODE_CONFIGS
    mode_config = resolved_mode_configs.get(mode, resolved_mode_configs["build"])
    complexity_lower = complexity.lower()

    if complexity_lower in mode_config.complexity_thresholds:
        return mode_config.complexity_thresholds[complexity_lower]

    complexity_level = ComplexityLevel.from_string(complexity)
    default_actions = {
        ComplexityLevel.TRIVIAL: TeamSuggestionAction.NONE,
        ComplexityLevel.LOW: TeamSuggestionAction.NONE,
        ComplexityLevel.MEDIUM: TeamSuggestionAction.SUGGEST,
        ComplexityLevel.HIGH: TeamSuggestionAction.SUGGEST,
        ComplexityLevel.EXTREME: TeamSuggestionAction.AUTO_SPAWN,
    }
    return default_actions.get(complexity_level, TeamSuggestionAction.NONE)


def recommend_teams_for_catalog(
    *,
    task_type: str,
    complexity: str,
    coordination_catalog: VerticalCoordinationCatalog,
    team_selector: TeamSelectionStrategyProtocol,
    top_k: int = 3,
) -> List[TeamRecommendation]:
    """Recommend teams from a shared coordination catalog."""
    available_teams = (
        dict(coordination_catalog.team_catalog.team_specs)
        if coordination_catalog.has_team_specs
        else {}
    )
    if not available_teams:
        return []

    return team_selector.recommend(
        task_type=task_type,
        complexity=complexity,
        available_teams=available_teams,
        top_k=top_k,
    )


def recommend_workflows_for_catalog(
    *,
    task_type: str,
    mode: str,
    coordination_catalog: VerticalCoordinationCatalog,
    workflow_selector: WorkflowSelectionStrategyProtocol,
    top_k: int = 3,
) -> List[WorkflowRecommendation]:
    """Recommend workflows from a shared coordination catalog."""
    available_workflows = (
        dict(coordination_catalog.workflow_catalog.workflow_specs)
        if coordination_catalog.has_workflow_specs
        else {}
    )
    if not available_workflows:
        return []

    return workflow_selector.recommend(
        task_type=task_type,
        mode=mode,
        available_workflows=available_workflows,
        top_k=top_k,
    )


def resolve_default_workflow_for_mode(
    mode: str,
    *,
    coordination_catalog: VerticalCoordinationCatalog,
    mode_configs: Optional[Dict[str, ModeCoordinationConfig]] = None,
) -> Optional[str]:
    """Resolve the default workflow for a mode from a shared coordination catalog."""
    resolved_mode_configs = mode_configs or DEFAULT_MODE_CONFIGS
    mode_config = resolved_mode_configs.get(mode)
    if mode_config is None or not mode_config.default_workflows:
        return None

    available_workflows = set(coordination_catalog.list_workflow_names())
    for workflow_name in mode_config.default_workflows:
        if workflow_name in available_workflows:
            return workflow_name
    return mode_config.default_workflows[0]


def build_coordination_suggestion(
    *,
    task_type: str,
    complexity: str,
    mode: str,
    coordination_catalog: VerticalCoordinationCatalog,
    team_selector: TeamSelectionStrategyProtocol,
    workflow_selector: WorkflowSelectionStrategyProtocol,
    mode_configs: Optional[Dict[str, ModeCoordinationConfig]] = None,
) -> CoordinationSuggestion:
    """Build a shared coordination recommendation from a framework catalog."""
    resolved_mode_configs = mode_configs or DEFAULT_MODE_CONFIGS
    complexity_level = ComplexityLevel.from_string(complexity)
    mode_config = resolved_mode_configs.get(mode, resolved_mode_configs["build"])
    action = get_action_for_complexity(
        complexity,
        mode,
        mode_configs=resolved_mode_configs,
    )

    team_recommendations: List[TeamRecommendation] = []
    if mode_config.team_suggestion_enabled:
        team_recommendations = recommend_teams_for_catalog(
            task_type=task_type,
            complexity=complexity,
            coordination_catalog=coordination_catalog,
            team_selector=team_selector,
        )

    workflow_recommendations = recommend_workflows_for_catalog(
        task_type=task_type,
        mode=mode,
        coordination_catalog=coordination_catalog,
        workflow_selector=workflow_selector,
    )

    return CoordinationSuggestion(
        mode=mode,
        task_type=task_type,
        complexity=complexity_level,
        action=action,
        team_recommendations=team_recommendations,
        workflow_recommendations=workflow_recommendations,
        system_prompt_additions=mode_config.system_prompt_addition,
        tool_priorities=mode_config.tool_priorities,
        metadata={
            "mode_config": mode_config.mode_name,
            "team_suggestion_enabled": mode_config.team_suggestion_enabled,
        },
    )


def build_registered_coordination_suggestions(
    *,
    task_type: str,
    complexity: str,
    mode: str = "build",
    vertical: Optional[str] = None,
    team_selector: Optional[TeamSelectionStrategyProtocol] = None,
    workflow_selector: Optional[WorkflowSelectionStrategyProtocol] = None,
    mode_configs: Optional[Dict[str, ModeCoordinationConfig]] = None,
) -> List[CatalogCoordinationSuggestion]:
    """Build shared coordination recommendations for registered vertical catalogs."""
    from victor.framework.team_runtime import resolve_registered_coordination_catalogs

    requested_vertical = (
        vertical.strip() if isinstance(vertical, str) and vertical.strip() else None
    )
    resolved_mode_configs = mode_configs or DEFAULT_MODE_CONFIGS
    resolved_team_selector = team_selector or HybridTeamSelector(
        RuleBasedTeamSelector(),
        LearningBasedTeamSelector(),
    )
    resolved_workflow_selector = workflow_selector or RuleBasedWorkflowSelector()
    coordination_catalogs = resolve_registered_coordination_catalogs()

    if requested_vertical is not None:
        selected_verticals = (
            [requested_vertical] if requested_vertical in coordination_catalogs else []
        )
    else:
        selected_verticals = sorted(coordination_catalogs)

    suggestions: List[CatalogCoordinationSuggestion] = []
    for vertical_name in selected_verticals:
        coordination_catalog = coordination_catalogs.get(vertical_name)
        if coordination_catalog is None:
            continue

        suggestion = build_coordination_suggestion(
            task_type=task_type,
            complexity=complexity,
            mode=mode,
            coordination_catalog=coordination_catalog,
            team_selector=resolved_team_selector,
            workflow_selector=resolved_workflow_selector,
            mode_configs=resolved_mode_configs,
        )
        catalog_suggestion = CatalogCoordinationSuggestion(
            vertical=vertical_name,
            suggestion=suggestion,
            available_teams=tuple(coordination_catalog.list_team_names()),
            available_workflows=tuple(coordination_catalog.list_workflow_names()),
            default_workflow=resolve_default_workflow_for_mode(
                mode,
                coordination_catalog=coordination_catalog,
                mode_configs=resolved_mode_configs,
            ),
        )
        if _should_include_catalog_suggestion(
            catalog_suggestion,
            force_include=requested_vertical is not None,
        ):
            suggestions.append(catalog_suggestion)

    return suggestions


def build_runtime_coordination_suggestion(
    *,
    runtime_subject: Any,
    task_type: str,
    complexity: str,
    mode: Optional[str] = None,
) -> CoordinationSuggestion:
    """Build a coordination recommendation for an orchestrator-like runtime.

    This helper keeps user-facing runtime surfaces on the shared framework
    recommendation engine while still honoring configured selectors and mode
    policies when the runtime already has them.
    """
    from victor.framework.team_runtime import resolve_vertical_coordination_catalog

    coordination = _resolve_runtime_coordination(runtime_subject)
    if _should_delegate_to_coordination_surface(coordination):
        try:
            return coordination.suggest_for_task(
                task_type=task_type,
                complexity=complexity,
                mode=mode or _resolve_runtime_mode(runtime_subject),
            )
        except Exception as exc:
            logger.debug("Fallback coordination surface failed: %s", exc)

    resolved_mode = mode or _resolve_runtime_mode(runtime_subject)
    return build_coordination_suggestion(
        task_type=task_type,
        complexity=complexity,
        mode=resolved_mode,
        coordination_catalog=resolve_vertical_coordination_catalog(
            _resolve_runtime_vertical_subject(runtime_subject)
        ),
        team_selector=_resolve_runtime_team_selector(runtime_subject),
        workflow_selector=_resolve_runtime_workflow_selector(runtime_subject),
        mode_configs=_resolve_runtime_mode_configs(runtime_subject),
    )


def get_runtime_coordination_suggestion(
    *,
    runtime_subject: Any,
    task_type: str,
    complexity: str,
    mode: Optional[str] = None,
) -> CoordinationSuggestion:
    """Get a runtime coordination suggestion via the subject's canonical API when available."""
    getter = getattr(runtime_subject, "get_coordination_suggestion", None)
    if callable(getter):
        try:
            return getter(task_type, complexity, mode=mode)
        except TypeError:
            if mode is None:
                try:
                    return getter(task_type, complexity)
                except Exception as exc:
                    logger.debug("Runtime coordination getter fallback failed: %s", exc)
            else:
                logger.debug("Runtime coordination getter rejected mode=%s", mode)
        except Exception as exc:
            logger.debug("Runtime coordination getter failed: %s", exc)

    return build_runtime_coordination_suggestion(
        runtime_subject=runtime_subject,
        task_type=task_type,
        complexity=complexity,
        mode=mode,
    )


def _resolve_runtime_mode(runtime_subject: Any) -> str:
    """Resolve the current mode from a runtime-like object."""
    mode_controller = getattr(runtime_subject, "mode_controller", None)
    current_mode = getattr(mode_controller, "current_mode", None)
    current_mode_value = getattr(current_mode, "value", None)
    if isinstance(current_mode_value, str) and current_mode_value:
        return current_mode_value
    return "build"


def _resolve_runtime_vertical_subject(runtime_subject: Any) -> Any:
    """Resolve the vertical or vertical context for a runtime-like object."""
    get_vertical_context = getattr(runtime_subject, "get_vertical_context", None)
    if callable(get_vertical_context):
        try:
            vertical_context = get_vertical_context()
        except Exception:
            vertical_context = None
        if vertical_context is not None:
            return vertical_context

    vertical_context = (
        getattr(runtime_subject, "vertical_context", None)
        if _has_declared_attribute(runtime_subject, "vertical_context")
        else None
    )
    if vertical_context is not None:
        return vertical_context
    if _has_declared_attribute(runtime_subject, "_vertical_context"):
        return getattr(runtime_subject, "_vertical_context", None)
    return None


def _resolve_runtime_coordination(runtime_subject: Any) -> Any:
    """Resolve the declared coordination surface for a runtime-like object."""
    for attr_name in ("coordination_advisor", "coordination"):
        if not _has_declared_attribute(runtime_subject, attr_name):
            continue
        try:
            coordination = getattr(runtime_subject, attr_name, None)
        except Exception:
            coordination = None
        if coordination is not None:
            return coordination
    return None


def _resolve_runtime_team_selector(runtime_subject: Any) -> Any:
    """Resolve the configured team selector for a runtime-like object."""
    coordination = _resolve_runtime_coordination(runtime_subject)
    team_selector = getattr(coordination, "_team_selector", None)
    if team_selector is not None:
        return team_selector
    return HybridTeamSelector(
        RuleBasedTeamSelector(),
        LearningBasedTeamSelector(),
    )


def _resolve_runtime_workflow_selector(runtime_subject: Any) -> Any:
    """Resolve the configured workflow selector for a runtime-like object."""
    coordination = _resolve_runtime_coordination(runtime_subject)
    workflow_selector = getattr(coordination, "_workflow_selector", None)
    if workflow_selector is not None:
        return workflow_selector
    return RuleBasedWorkflowSelector()


def _resolve_runtime_mode_configs(
    runtime_subject: Any,
) -> Dict[str, ModeCoordinationConfig]:
    """Resolve the configured mode policies for a runtime-like object."""
    coordination = _resolve_runtime_coordination(runtime_subject)
    mode_configs = getattr(coordination, "_mode_configs", None)
    if isinstance(mode_configs, dict) and mode_configs:
        return mode_configs
    return DEFAULT_MODE_CONFIGS


def serialize_coordination_suggestion(
    suggestion: CoordinationSuggestion,
    *,
    vertical: Optional[str] = None,
    available_teams: Optional[Tuple[str, ...]] = None,
    available_workflows: Optional[Tuple[str, ...]] = None,
    default_workflow: Optional[str] = None,
) -> Dict[str, Any]:
    """Serialize a coordination suggestion for CLI/API surfaces."""
    payload: Dict[str, Any] = {
        "mode": suggestion.mode,
        "task_type": suggestion.task_type,
        "complexity": suggestion.complexity.value,
        "action": suggestion.action.value,
        "should_spawn_team": suggestion.should_spawn_team,
        "should_suggest_team": suggestion.should_suggest_team,
        "primary_team": _serialize_team_recommendation(suggestion.primary_team),
        "primary_workflow": _serialize_workflow_recommendation(suggestion.primary_workflow),
        "team_recommendations": [
            _serialize_team_recommendation(recommendation)
            for recommendation in suggestion.team_recommendations
        ],
        "workflow_recommendations": [
            _serialize_workflow_recommendation(recommendation)
            for recommendation in suggestion.workflow_recommendations
        ],
        "system_prompt_additions": suggestion.system_prompt_additions,
        "tool_priorities": dict(suggestion.tool_priorities),
        "metadata": dict(suggestion.metadata),
    }
    if vertical is not None:
        payload["vertical"] = vertical
    if available_teams is not None:
        payload["available_teams"] = list(available_teams)
    if available_workflows is not None:
        payload["available_workflows"] = list(available_workflows)
    payload["default_workflow"] = default_workflow
    return payload


def serialize_catalog_coordination_suggestions(
    suggestions: List[CatalogCoordinationSuggestion],
) -> List[Dict[str, Any]]:
    """Serialize catalog coordination suggestions for user-facing surfaces."""
    return [suggestion.to_dict() for suggestion in suggestions]


def _should_delegate_to_coordination_surface(coordination: Any) -> bool:
    """Whether to fall back to a runtime-provided coordination surface."""
    if coordination is None:
        return False
    suggest_for_task = getattr(coordination, "suggest_for_task", None)
    if not callable(suggest_for_task):
        return False
    return not any(
        _has_declared_attribute(coordination, attr_name)
        for attr_name in ("_team_selector", "_workflow_selector", "_mode_configs")
    )


def _serialize_team_recommendation(
    recommendation: Optional[TeamRecommendation],
) -> Optional[Dict[str, Any]]:
    """Serialize a team recommendation for JSON output."""
    if recommendation is None:
        return None
    return {
        "team_name": recommendation.team_name,
        "confidence": recommendation.confidence,
        "reason": recommendation.reason,
        "formation": recommendation.formation,
        "suggested_budget": recommendation.suggested_budget,
        "role_distribution": dict(recommendation.role_distribution or {}),
        "source": recommendation.source,
    }


def _serialize_workflow_recommendation(
    recommendation: Optional[WorkflowRecommendation],
) -> Optional[Dict[str, Any]]:
    """Serialize a workflow recommendation for JSON output."""
    if recommendation is None:
        return None
    return {
        "workflow_name": recommendation.workflow_name,
        "confidence": recommendation.confidence,
        "reason": recommendation.reason,
        "trigger_condition": recommendation.trigger_condition,
        "estimated_steps": recommendation.estimated_steps,
    }


def _should_include_catalog_suggestion(
    suggestion: CatalogCoordinationSuggestion,
    *,
    force_include: bool,
) -> bool:
    """Decide whether a registered catalog recommendation is worth surfacing."""
    if force_include:
        return True
    if suggestion.suggestion.has_team_suggestion or suggestion.suggestion.has_workflow_suggestion:
        return True
    return suggestion.default_workflow is not None


def _has_declared_attribute(subject: Any, attr_name: str) -> bool:
    """Check for a real attribute without triggering MagicMock fallback."""
    if subject is None:
        return False
    try:
        getattr_static(subject, attr_name)
    except AttributeError:
        return False
    return True


__all__ = [
    "CatalogCoordinationSuggestion",
    "DEFAULT_MODE_CONFIGS",
    "HybridTeamSelector",
    "LearningBasedTeamSelector",
    "ModeCoordinationConfig",
    "RuleBasedTeamSelector",
    "RuleBasedWorkflowSelector",
    "VerticalCoordinationAdvisor",
    "create_vertical_coordination_advisor",
    "build_registered_coordination_suggestions",
    "build_coordination_suggestion",
    "build_runtime_coordination_suggestion",
    "get_runtime_coordination_suggestion",
    "get_action_for_complexity",
    "recommend_teams_for_catalog",
    "recommend_workflows_for_catalog",
    "resolve_default_workflow_for_mode",
    "serialize_catalog_coordination_suggestions",
    "serialize_coordination_suggestion",
]
