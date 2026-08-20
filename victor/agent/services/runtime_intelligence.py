# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical runtime-intelligence service for live agent execution.

This service consolidates the currently scattered runtime collaborators used for:
- turn perception and task understanding
- prompt optimization bundle retrieval
- decision-service turn budget management

It composes the existing strong building blocks instead of replacing them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.agent.optimization_injector import OptimizationInjector
from victor.agent.task_analyzer import TaskAnalyzer, get_task_analyzer
from victor.evaluation.experiment_memory import ExperimentMemoryStore
from victor.evaluation.topology_feedback import (
    aggregate_topology_feedback,
    summarize_topology_feedback,
)
from victor.evaluation.runtime_feedback import (
    RuntimeEvaluationFeedbackScope,
    load_runtime_evaluation_feedback,
    runtime_evaluation_feedback_scope_from_context,
    save_session_topology_runtime_feedback,
)
from victor.framework.perception_integration import PerceptionIntegration
from victor.framework.routing_policy import StructuredRoutingPolicy
from victor.framework.runtime_evaluation_policy import (
    ClarificationDecision,
    RuntimeEvaluationFeedback,
    RuntimeEvaluationPolicy,
)

if TYPE_CHECKING:
    from victor.agent.services.protocols.decision_service import (
        LLMDecisionServiceProtocol,
    )

logger = logging.getLogger(__name__)
_FEEDBACK_OVERRIDE_FIELDS = {
    "completion_threshold",
    "enhanced_progress_threshold",
    "minimum_supported_evidence_score",
}


@dataclass(frozen=True)
class PromptOptimizationIdentity:
    """Canonical identity for a live prompt optimization payload."""

    provider: Optional[str] = None
    prompt_candidate_hash: Optional[str] = None
    section_name: Optional[str] = None
    prompt_section_name: Optional[str] = None
    strategy_name: Optional[str] = None
    source: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize identity using the shared prompt-optimization schema."""
        resolved_section = self.prompt_section_name or self.section_name
        return {
            "provider": self.provider,
            "prompt_candidate_hash": self.prompt_candidate_hash,
            "section_name": resolved_section,
            "prompt_section_name": resolved_section,
            "strategy_name": self.strategy_name,
            "source": self.source,
        }


@dataclass(frozen=True)
class PromptOptimizationBundle:
    """Typed prompt-optimization payload for a single turn."""

    evolved_sections: List[str] = field(default_factory=list)
    few_shots: Optional[str] = None
    failure_hint: Optional[str] = None
    identities: List[PromptOptimizationIdentity] = field(default_factory=list)
    experiment_guidance: List[str] = field(default_factory=list)
    experiment_memory_hints: Dict[str, Any] = field(default_factory=dict)

    def to_session_metadata(self) -> Dict[str, Any]:
        """Serialize applied live prompt optimizations for session artifacts."""
        entries = [identity.to_metadata() for identity in self.identities]
        by_section: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            section_name = str(
                entry.get("section_name") or entry.get("prompt_section_name") or ""
            ).strip()
            if section_name:
                by_section[section_name] = dict(entry)
        metadata = {"entries": entries, "by_section": by_section}
        if self.experiment_guidance or self.experiment_memory_hints:
            experiment_memory = dict(self.experiment_memory_hints)
            if self.experiment_guidance:
                experiment_memory["prompt_guidance"] = list(self.experiment_guidance)
            metadata["experiment_memory"] = experiment_memory
        return metadata


@dataclass(frozen=True)
class RuntimeIntelligenceSnapshot:
    """Typed runtime view of a turn analysis."""

    query: str
    perception: Optional[Any] = None
    task_analysis: Optional[Any] = None
    decision_service_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyRoutingFeedback:
    """Learned topology preference derived from validated runtime feedback."""

    coverage: float = 0.0
    avg_reward: float = 0.0
    avg_confidence: float = 0.0
    observation_count_hint: int = 0
    action_agreement: float = 0.0
    topology_agreement: float = 0.0
    provider_agreement: float = 0.0
    formation_agreement: float = 0.0
    conflict_score: float = 0.0
    preferred_action: Optional[str] = None
    preferred_topology: Optional[str] = None
    preferred_execution_mode: Optional[str] = None
    preferred_provider: Optional[str] = None
    preferred_formation: Optional[str] = None
    action_distribution: Dict[str, float] = field(default_factory=dict)
    topology_distribution: Dict[str, float] = field(default_factory=dict)
    execution_mode_distribution: Dict[str, float] = field(default_factory=dict)
    provider_distribution: Dict[str, float] = field(default_factory=dict)
    formation_distribution: Dict[str, float] = field(default_factory=dict)
    selection_policy_distribution: Dict[str, float] = field(default_factory=dict)
    selection_policy_reward_totals: Dict[str, float] = field(default_factory=dict)
    selection_policy_optimization_counts: Dict[str, float] = field(default_factory=dict)
    selection_policy_optimization_reward_totals: Dict[str, float] = field(default_factory=dict)
    selection_policy_feasible_counts: Dict[str, float] = field(default_factory=dict)
    selection_policy_scope_metrics: Dict[str, Dict[str, Dict[str, Any]]] = field(
        default_factory=dict
    )

    @property
    def support(self) -> float:
        """Normalized trust in the learned topology preference."""
        base_support = max(0.0, min(1.0, (self.coverage * 0.55) + (self.avg_reward * 0.45)))
        consistency_multiplier = max(0.25, 1.0 - (self.conflict_score * 0.75))
        return round(max(0.0, min(1.0, base_support * consistency_multiplier)), 4)

    @property
    def observation_count(self) -> int:
        """Return the approximate number of observations backing this feedback."""
        totals: List[float] = []
        for distribution in (
            self.action_distribution,
            self.topology_distribution,
            self.execution_mode_distribution,
            self.provider_distribution,
            self.formation_distribution,
        ):
            total = 0.0
            for value in distribution.values():
                try:
                    total += float(value)
                except (TypeError, ValueError):
                    continue
            if total > 0.0:
                totals.append(total)

        if totals:
            return max(self.observation_count_hint, max(1, int(round(max(totals)))))
        if any((self.coverage > 0.0, self.avg_reward > 0.0, self.avg_confidence > 0.0)):
            return max(self.observation_count_hint, 1)
        return max(0, self.observation_count_hint)

    def is_actionable(
        self,
        *,
        min_coverage: float = 0.2,
        min_reward: float = 0.45,
        min_observations: int = 2,
    ) -> bool:
        """Return whether the feedback is strong enough to steer routing."""
        return (
            self.observation_count >= min_observations
            and self.coverage >= min_coverage
            and self.avg_reward >= min_reward
            and self.support > 0.0
            and any(
                (
                    self.preferred_action,
                    self.preferred_topology,
                    self.preferred_execution_mode,
                    self.preferred_provider,
                    self.preferred_formation,
                )
            )
        )

    @property
    def avg_reward_by_selection_policy(self) -> Dict[str, float]:
        """Return average topology reward grouped by topology selection policy."""
        averages: Dict[str, float] = {}
        for policy, count_value in self.selection_policy_distribution.items():
            try:
                count = float(count_value)
                reward_total = float(self.selection_policy_reward_totals.get(policy, 0.0))
            except (TypeError, ValueError):
                continue
            if count <= 0.0:
                continue
            averages[policy] = round(reward_total / count, 4)
        return averages

    @property
    def learned_override_reward_delta(self) -> Optional[float]:
        """Compare learned-close overrides against heuristic topology selection."""
        averages = self.avg_reward_by_selection_policy
        try:
            learned = float(averages["learned_close_override"])
            heuristic = float(averages["heuristic"])
        except (KeyError, TypeError, ValueError):
            return None
        return round(learned - heuristic, 4)

    @property
    def avg_optimization_reward_by_selection_policy(self) -> Dict[str, float]:
        """Return average optimization reward grouped by topology selection policy."""
        averages: Dict[str, float] = {}
        for policy, count_value in self.selection_policy_optimization_counts.items():
            try:
                count = float(count_value)
                reward_total = float(
                    self.selection_policy_optimization_reward_totals.get(policy, 0.0)
                )
            except (TypeError, ValueError):
                continue
            if count <= 0.0:
                continue
            averages[policy] = round(reward_total / count, 4)
        return averages

    @property
    def feasibility_rate_by_selection_policy(self) -> Dict[str, float]:
        """Return feasibility rate grouped by topology selection policy."""
        rates: Dict[str, float] = {}
        for policy, count_value in self.selection_policy_optimization_counts.items():
            try:
                count = float(count_value)
                feasible_count = float(self.selection_policy_feasible_counts.get(policy, 0.0))
            except (TypeError, ValueError):
                continue
            if count <= 0.0:
                continue
            rates[policy] = round(feasible_count / count, 4)
        return rates

    @property
    def learned_override_optimization_reward_delta(self) -> Optional[float]:
        """Compare learned-close overrides against heuristic policy using PR2 reward."""
        averages = self.avg_optimization_reward_by_selection_policy
        try:
            learned = float(averages["learned_close_override"])
            heuristic = float(averages["heuristic"])
        except (KeyError, TypeError, ValueError):
            return None
        return round(learned - heuristic, 4)

    @property
    def learned_override_feasibility_delta(self) -> Optional[float]:
        """Compare learned-close overrides against heuristic policy using feasibility rate."""
        rates = self.feasibility_rate_by_selection_policy
        try:
            learned = float(rates["learned_close_override"])
            heuristic = float(rates["heuristic"])
        except (KeyError, TypeError, ValueError):
            return None
        return round(learned - heuristic, 4)

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize the learned topology preference for logs and snapshots."""
        return {
            "coverage": round(self.coverage, 4),
            "avg_reward": round(self.avg_reward, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "support": self.support,
            "observation_count": self.observation_count,
            "observation_count_hint": self.observation_count_hint,
            "action_agreement": round(self.action_agreement, 4),
            "topology_agreement": round(self.topology_agreement, 4),
            "provider_agreement": round(self.provider_agreement, 4),
            "formation_agreement": round(self.formation_agreement, 4),
            "conflict_score": round(self.conflict_score, 4),
            "preferred_action": self.preferred_action,
            "preferred_topology": self.preferred_topology,
            "preferred_execution_mode": self.preferred_execution_mode,
            "preferred_provider": self.preferred_provider,
            "preferred_formation": self.preferred_formation,
            "action_distribution": dict(self.action_distribution),
            "topology_distribution": dict(self.topology_distribution),
            "execution_mode_distribution": dict(self.execution_mode_distribution),
            "provider_distribution": dict(self.provider_distribution),
            "formation_distribution": dict(self.formation_distribution),
            "selection_policy_distribution": dict(self.selection_policy_distribution),
            "selection_policy_reward_totals": dict(self.selection_policy_reward_totals),
            "avg_reward_by_selection_policy": dict(self.avg_reward_by_selection_policy),
            "learned_override_reward_delta": self.learned_override_reward_delta,
            "selection_policy_optimization_counts": dict(self.selection_policy_optimization_counts),
            "selection_policy_optimization_reward_totals": dict(
                self.selection_policy_optimization_reward_totals
            ),
            "avg_optimization_reward_by_selection_policy": dict(
                self.avg_optimization_reward_by_selection_policy
            ),
            "selection_policy_feasible_counts": dict(self.selection_policy_feasible_counts),
            "feasibility_rate_by_selection_policy": dict(self.feasibility_rate_by_selection_policy),
            "learned_override_optimization_reward_delta": (
                self.learned_override_optimization_reward_delta
            ),
            "learned_override_feasibility_delta": self.learned_override_feasibility_delta,
            "selection_policy_scope_metrics": dict(self.selection_policy_scope_metrics),
        }

    def to_routing_context(
        self,
        *,
        min_coverage: float = 0.2,
        min_reward: float = 0.45,
        min_observations: int = 2,
        min_agreement: float = 0.6,
        scope_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert learned feedback into soft topology-routing hints."""
        if (
            self.observation_count < min_observations
            or self.coverage < min_coverage
            or self.avg_reward < min_reward
            or self.support <= 0.0
        ):
            return {}

        routing_context = {
            "learned_topology_support": self.support,
            "learned_topology_observation_count": self.observation_count,
            "learned_topology_action_agreement": round(self.action_agreement, 4),
            "learned_topology_kind_agreement": round(self.topology_agreement, 4),
            "learned_provider_agreement": round(self.provider_agreement, 4),
            "learned_formation_agreement": round(self.formation_agreement, 4),
            "learned_topology_conflict_score": round(self.conflict_score, 4),
            "topology_feedback_coverage": round(self.coverage, 4),
            "avg_topology_reward": round(self.avg_reward, 4),
            "avg_topology_confidence": round(self.avg_confidence, 4),
        }
        selection_policy_metrics = self.resolve_selection_policy_metrics(
            scope_context=scope_context
        )
        learned_override_reward = selection_policy_metrics.get("learned_override_policy_reward")
        heuristic_reward = selection_policy_metrics.get("heuristic_policy_reward")
        if learned_override_reward is not None:
            routing_context["learned_override_policy_reward"] = learned_override_reward
            routing_context["learned_override_policy_count"] = int(
                selection_policy_metrics.get("learned_override_policy_count", 0)
            )
        if heuristic_reward is not None:
            routing_context["heuristic_policy_reward"] = heuristic_reward
            routing_context["heuristic_policy_count"] = int(
                selection_policy_metrics.get("heuristic_policy_count", 0)
            )
        if selection_policy_metrics.get("learned_override_policy_reward_delta") is not None:
            routing_context["learned_override_policy_reward_delta"] = selection_policy_metrics[
                "learned_override_policy_reward_delta"
            ]
        if selection_policy_metrics.get("learned_override_policy_optimization_reward") is not None:
            routing_context["learned_override_policy_optimization_reward"] = (
                selection_policy_metrics["learned_override_policy_optimization_reward"]
            )
        if selection_policy_metrics.get("heuristic_policy_optimization_reward") is not None:
            routing_context["heuristic_policy_optimization_reward"] = selection_policy_metrics[
                "heuristic_policy_optimization_reward"
            ]
        if (
            selection_policy_metrics.get("learned_override_policy_optimization_reward_delta")
            is not None
        ):
            routing_context["learned_override_policy_optimization_reward_delta"] = (
                selection_policy_metrics["learned_override_policy_optimization_reward_delta"]
            )
        if selection_policy_metrics.get("learned_override_policy_feasibility_rate") is not None:
            routing_context["learned_override_policy_feasibility_rate"] = selection_policy_metrics[
                "learned_override_policy_feasibility_rate"
            ]
        if selection_policy_metrics.get("heuristic_policy_feasibility_rate") is not None:
            routing_context["heuristic_policy_feasibility_rate"] = selection_policy_metrics[
                "heuristic_policy_feasibility_rate"
            ]
        if selection_policy_metrics.get("learned_override_policy_feasibility_delta") is not None:
            routing_context["learned_override_policy_feasibility_delta"] = selection_policy_metrics[
                "learned_override_policy_feasibility_delta"
            ]
        if selection_policy_metrics.get("scope_dimension") is not None:
            routing_context["learned_override_policy_scope_dimension"] = selection_policy_metrics[
                "scope_dimension"
            ]
        if selection_policy_metrics.get("scope_label") is not None:
            routing_context["learned_override_policy_scope_label"] = selection_policy_metrics[
                "scope_label"
            ]
        if self.preferred_action and self.action_agreement >= min_agreement:
            routing_context["learned_topology_action"] = self.preferred_action
        if self.preferred_topology and self.topology_agreement >= min_agreement:
            routing_context["learned_topology_kind"] = self.preferred_topology
        if (
            self.preferred_execution_mode
            and max(self.action_agreement, self.topology_agreement) >= min_agreement
        ):
            routing_context["learned_topology_execution_mode"] = self.preferred_execution_mode
        if self.preferred_provider and self.provider_agreement >= min_agreement:
            routing_context["learned_provider_hint"] = self.preferred_provider
        if self.preferred_formation and self.formation_agreement >= min_agreement:
            routing_context["learned_formation_hint"] = self.preferred_formation
        return routing_context

    def _effective_policy_count(self, policy: str) -> int:
        """Return a conservative integer evidence count for one selection policy."""
        return self._effective_policy_count_from_distribution(
            self.selection_policy_distribution,
            policy,
        )

    @staticmethod
    def _effective_policy_count_from_distribution(
        distribution: Dict[str, float],
        policy: str,
    ) -> int:
        """Return a conservative integer evidence count from a policy-count distribution."""
        try:
            count = float(distribution.get(policy, 0.0))
        except (TypeError, ValueError):
            return 0
        if count <= 0.0:
            return 0
        return max(1, int(math.ceil(count)))

    @staticmethod
    def _normalize_scope_token(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

    @classmethod
    def _model_family_token(cls, value: Any) -> Optional[str]:
        text = cls._normalize_scope_token(value)
        if text is None:
            return None
        return text.split("-", 1)[0].split("_", 1)[0].split("/", 1)[0]

    @classmethod
    def _scope_context_label(
        cls,
        scope_context: Optional[Dict[str, Any]],
        *,
        dimension: str,
    ) -> Optional[str]:
        context = scope_context or {}
        if dimension == "task_type":
            return cls._normalize_scope_token(context.get("task_type"))
        if dimension == "provider":
            provider_value = (
                context.get("provider_hint")
                or context.get("provider")
                or context.get("provider_name")
            )
            if provider_value is None:
                candidates = context.get("provider_candidates")
                if isinstance(candidates, list) and candidates:
                    provider_value = candidates[0]
            return cls._normalize_scope_token(provider_value)
        if dimension == "model_family":
            return cls._model_family_token(context.get("model") or context.get("model_name"))
        return None

    def resolve_selection_policy_metrics(
        self,
        *,
        scope_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve the most relevant learned-override policy metrics for the current scope."""
        for dimension in ("model_family", "provider", "task_type"):
            label = self._scope_context_label(scope_context, dimension=dimension)
            if label is None:
                continue
            dimension_metrics = self.selection_policy_scope_metrics.get(dimension) or {}
            bucket = dimension_metrics.get(label)
            if not isinstance(bucket, dict):
                continue
            policy_counts = dict(bucket.get("policy_counts") or {})
            avg_reward_by_policy = dict(bucket.get("avg_reward_by_policy") or {})
            avg_optimization_reward_by_policy = dict(
                bucket.get("avg_optimization_reward_by_policy") or {}
            )
            feasibility_rate_by_policy = dict(bucket.get("feasibility_rate_by_policy") or {})
            learned_override_reward = avg_reward_by_policy.get("learned_close_override")
            heuristic_reward = avg_reward_by_policy.get("heuristic")
            learned_override_reward_delta = bucket.get("learned_override_reward_delta")
            learned_override_optimization_reward = avg_optimization_reward_by_policy.get(
                "learned_close_override"
            )
            heuristic_optimization_reward = avg_optimization_reward_by_policy.get("heuristic")
            learned_override_optimization_reward_delta = bucket.get(
                "learned_override_optimization_reward_delta"
            )
            learned_override_feasibility_rate = feasibility_rate_by_policy.get(
                "learned_close_override"
            )
            heuristic_feasibility_rate = feasibility_rate_by_policy.get("heuristic")
            learned_override_feasibility_delta = bucket.get("learned_override_feasibility_delta")
            if (
                learned_override_reward is None
                and heuristic_reward is None
                and learned_override_reward_delta is None
                and learned_override_optimization_reward is None
                and heuristic_optimization_reward is None
                and learned_override_optimization_reward_delta is None
                and learned_override_feasibility_rate is None
                and heuristic_feasibility_rate is None
                and learned_override_feasibility_delta is None
            ):
                continue
            return {
                "scope_dimension": dimension,
                "scope_label": label,
                "learned_override_policy_reward": learned_override_reward,
                "heuristic_policy_reward": heuristic_reward,
                "learned_override_policy_count": self._effective_policy_count_from_distribution(
                    policy_counts,
                    "learned_close_override",
                ),
                "heuristic_policy_count": self._effective_policy_count_from_distribution(
                    policy_counts,
                    "heuristic",
                ),
                "learned_override_policy_reward_delta": learned_override_reward_delta,
                "learned_override_policy_optimization_reward": (
                    learned_override_optimization_reward
                ),
                "heuristic_policy_optimization_reward": heuristic_optimization_reward,
                "learned_override_policy_optimization_reward_delta": (
                    learned_override_optimization_reward_delta
                ),
                "learned_override_policy_feasibility_rate": (learned_override_feasibility_rate),
                "heuristic_policy_feasibility_rate": heuristic_feasibility_rate,
                "learned_override_policy_feasibility_delta": (learned_override_feasibility_delta),
            }

        return {
            "scope_dimension": "global",
            "scope_label": "global",
            "learned_override_policy_reward": self.avg_reward_by_selection_policy.get(
                "learned_close_override"
            ),
            "heuristic_policy_reward": self.avg_reward_by_selection_policy.get("heuristic"),
            "learned_override_policy_count": self._effective_policy_count("learned_close_override"),
            "heuristic_policy_count": self._effective_policy_count("heuristic"),
            "learned_override_policy_reward_delta": self.learned_override_reward_delta,
            "learned_override_policy_optimization_reward": (
                self.avg_optimization_reward_by_selection_policy.get("learned_close_override")
            ),
            "heuristic_policy_optimization_reward": (
                self.avg_optimization_reward_by_selection_policy.get("heuristic")
            ),
            "learned_override_policy_optimization_reward_delta": (
                self.learned_override_optimization_reward_delta
            ),
            "learned_override_policy_feasibility_rate": (
                self.feasibility_rate_by_selection_policy.get("learned_close_override")
            ),
            "heuristic_policy_feasibility_rate": (
                self.feasibility_rate_by_selection_policy.get("heuristic")
            ),
            "learned_override_policy_feasibility_delta": (self.learned_override_feasibility_delta),
        }


@dataclass(frozen=True)
class TeamRoutingFeedback:
    """Learned team/worktree preference derived from validated runtime feedback."""

    coverage: float = 0.0
    task_count: int = 0
    worktree_plan_count: int = 0
    worktree_materialized_count: int = 0
    worktree_dry_run_count: int = 0
    low_risk_task_count: int = 0
    medium_risk_task_count: int = 0
    high_risk_task_count: int = 0
    merge_conflict_task_count: int = 0
    cleanup_task_count: int = 0
    cleanup_error_task_count: int = 0
    avg_team_assignments: float = 0.0
    avg_team_scoped_members: float = 0.0
    avg_team_members_with_changes: float = 0.0
    avg_team_changed_file_count: float = 0.0
    formation_distribution: Dict[str, float] = field(default_factory=dict)
    merge_risk_distribution: Dict[str, float] = field(default_factory=dict)
    scope_dimension: Optional[str] = None
    scope_label: Optional[str] = None

    @property
    def formation_agreement(self) -> float:
        """Estimate agreement from the dominant learned team formation."""
        return RuntimeIntelligenceService._distribution_agreement(self.formation_distribution)

    @property
    def preferred_formation(self) -> Optional[str]:
        """Return the safest supported formation hint for team execution."""
        if self.support <= 0.0:
            return None
        if self.risk_score >= 0.5:
            return "hierarchical"
        return RuntimeIntelligenceService._preferred_label(self.formation_distribution)

    @property
    def low_risk_rate(self) -> float:
        return round(self.low_risk_task_count / max(1, self.task_count), 4)

    @property
    def medium_risk_rate(self) -> float:
        return round(self.medium_risk_task_count / max(1, self.task_count), 4)

    @property
    def high_risk_rate(self) -> float:
        return round(self.high_risk_task_count / max(1, self.task_count), 4)

    @property
    def merge_conflict_rate(self) -> float:
        return round(self.merge_conflict_task_count / max(1, self.task_count), 4)

    @property
    def cleanup_error_rate(self) -> float:
        return round(self.cleanup_error_task_count / max(1, self.task_count), 4)

    @property
    def materialization_rate(self) -> float:
        return round(self.worktree_materialized_count / max(1, self.task_count), 4)

    @property
    def dry_run_rate(self) -> float:
        return round(self.worktree_dry_run_count / max(1, self.task_count), 4)

    @property
    def cleanup_failure_rate(self) -> float:
        if self.cleanup_task_count <= 0:
            return self.cleanup_error_rate
        return round(self.cleanup_error_task_count / max(1, self.cleanup_task_count), 4)

    @property
    def risk_score(self) -> float:
        """Compress merge-risk and cleanup instability into one bounded penalty."""
        return round(
            max(
                0.0,
                min(
                    1.0,
                    (self.high_risk_rate * 0.55)
                    + (self.merge_conflict_rate * 0.25)
                    + (self.cleanup_error_rate * 0.20),
                ),
            ),
            4,
        )

    @property
    def support(self) -> float:
        """Normalized trust in the learned team/worktree preference."""
        return round(
            max(
                0.0,
                min(
                    1.0,
                    (self.coverage * 0.35)
                    + (self.materialization_rate * 0.25)
                    + ((1.0 - self.risk_score) * 0.25)
                    + (self.low_risk_rate * 0.15),
                ),
            ),
            4,
        )

    @property
    def recommended_max_workers(self) -> Optional[int]:
        """Return a conservative worker-count hint derived from merge-safe history."""
        if self.support <= 0.0:
            return None
        baseline = max(self.avg_team_scoped_members, self.avg_team_assignments, 0.0)
        if baseline <= 0.0:
            return None
        workers = int(round(baseline))
        if self.high_risk_rate >= 0.25 or self.medium_risk_rate >= 0.5:
            workers -= 1
        if self.merge_conflict_task_count > 0 or self.cleanup_error_task_count > 0:
            workers = min(workers, 2)
        return max(2, min(4, workers))

    @property
    def recommends_worktree_isolation(self) -> bool:
        """Whether team execution should use isolated worktrees by default."""
        if self.support <= 0.0 or self.worktree_plan_count <= 0:
            return False
        if self.avg_team_scoped_members >= 2.0 or self.avg_team_changed_file_count >= 2.0:
            return True
        return self.worktree_materialized_count > 0

    @property
    def recommends_materialized_worktrees(self) -> bool:
        """Whether isolated worktrees should be materialized, not just planned."""
        if (
            not self.recommends_worktree_isolation
            or self.worktree_materialized_count <= 0
            or self.recommends_dry_run_worktrees
        ):
            return False
        return self.materialization_rate >= 0.4 and self.cleanup_failure_rate < 0.5

    @property
    def recommends_dry_run_worktrees(self) -> bool:
        """Whether the current scope should prefer planned/dry-run worktrees."""
        if not self.recommends_worktree_isolation or self.worktree_dry_run_count <= 0:
            return False
        if self.worktree_materialized_count <= 0:
            return True
        if self.dry_run_rate >= self.materialization_rate:
            return True
        return self.risk_score >= 0.45 or self.cleanup_failure_rate >= 0.35

    @property
    def recommended_cleanup_worktrees(self) -> Optional[bool]:
        """Whether materialized worktrees should be auto-cleaned after execution."""
        if not self.recommends_worktree_isolation or self.cleanup_task_count <= 0:
            return None
        return self.cleanup_failure_rate < 0.35

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize the learned team/worktree preference for logs and snapshots."""
        metadata = {
            "coverage": round(self.coverage, 4),
            "support": round(self.support, 4),
            "task_count": self.task_count,
            "worktree_plan_count": self.worktree_plan_count,
            "worktree_materialized_count": self.worktree_materialized_count,
            "worktree_dry_run_count": self.worktree_dry_run_count,
            "low_risk_task_count": self.low_risk_task_count,
            "medium_risk_task_count": self.medium_risk_task_count,
            "high_risk_task_count": self.high_risk_task_count,
            "merge_conflict_task_count": self.merge_conflict_task_count,
            "cleanup_task_count": self.cleanup_task_count,
            "cleanup_error_task_count": self.cleanup_error_task_count,
            "formation_agreement": round(self.formation_agreement, 4),
            "preferred_formation": self.preferred_formation,
            "recommended_max_workers": self.recommended_max_workers,
            "low_risk_rate": self.low_risk_rate,
            "medium_risk_rate": self.medium_risk_rate,
            "high_risk_rate": self.high_risk_rate,
            "merge_conflict_rate": self.merge_conflict_rate,
            "cleanup_error_rate": self.cleanup_error_rate,
            "cleanup_failure_rate": self.cleanup_failure_rate,
            "materialization_rate": self.materialization_rate,
            "dry_run_rate": self.dry_run_rate,
            "risk_score": self.risk_score,
            "recommends_worktree_isolation": self.recommends_worktree_isolation,
            "recommends_materialized_worktrees": self.recommends_materialized_worktrees,
            "recommends_dry_run_worktrees": self.recommends_dry_run_worktrees,
            "recommended_cleanup_worktrees": self.recommended_cleanup_worktrees,
            "avg_team_assignments": round(self.avg_team_assignments, 4),
            "avg_team_scoped_members": round(self.avg_team_scoped_members, 4),
            "avg_team_members_with_changes": round(self.avg_team_members_with_changes, 4),
            "avg_team_changed_file_count": round(self.avg_team_changed_file_count, 4),
            "formation_distribution": dict(self.formation_distribution),
            "merge_risk_distribution": dict(self.merge_risk_distribution),
        }
        if self.scope_dimension:
            metadata["scope_dimension"] = self.scope_dimension
        if self.scope_label:
            metadata["scope_label"] = self.scope_label
        return metadata

    def to_routing_context(
        self,
        *,
        min_coverage: float = 0.2,
        min_support: float = 0.2,
        min_agreement: float = 0.55,
    ) -> Dict[str, Any]:
        """Convert learned team feedback into soft formation and parallelism hints."""
        if self.coverage < min_coverage or self.support < min_support or self.task_count <= 0:
            return {}

        routing_context: Dict[str, Any] = {
            "learned_team_support": round(self.support, 4),
            "team_feedback_coverage": round(self.coverage, 4),
            "learned_team_risk_score": round(self.risk_score, 4),
            "learned_team_formation_agreement": round(self.formation_agreement, 4),
            "team_worktree_materialized_count": self.worktree_materialized_count,
            "team_worktree_dry_run_count": self.worktree_dry_run_count,
            "team_high_risk_task_count": self.high_risk_task_count,
            "team_merge_conflict_task_count": self.merge_conflict_task_count,
            "team_cleanup_error_task_count": self.cleanup_error_task_count,
        }
        if self.scope_dimension:
            routing_context["learned_team_policy_scope_dimension"] = self.scope_dimension
        if self.scope_label:
            routing_context["learned_team_policy_scope_label"] = self.scope_label
        preferred_formation = self.preferred_formation
        if preferred_formation and self.formation_agreement >= min_agreement:
            routing_context["learned_formation_hint"] = preferred_formation
            routing_context["learned_formation_agreement"] = round(self.formation_agreement, 4)
        recommended_max_workers = self.recommended_max_workers
        if recommended_max_workers is not None:
            routing_context["learned_team_max_workers_hint"] = recommended_max_workers
        if self.recommends_worktree_isolation:
            routing_context["learned_worktree_isolation_hint"] = True
        if self.recommends_dry_run_worktrees:
            routing_context["learned_dry_run_worktrees_hint"] = True
            routing_context["learned_materialize_worktrees_hint"] = False
        elif self.recommends_materialized_worktrees:
            routing_context["learned_materialize_worktrees_hint"] = True
            routing_context["learned_dry_run_worktrees_hint"] = False
        cleanup_hint = self.recommended_cleanup_worktrees
        if cleanup_hint is not None:
            routing_context["learned_cleanup_worktrees_hint"] = cleanup_hint
        return routing_context


@dataclass(frozen=True)
class DegradationRoutingFeedback:
    """Learned degradation and recovery preference derived from runtime feedback."""

    coverage: float = 0.0
    event_count: int = 0
    degraded_task_count: int = 0
    recovered_task_count: int = 0
    recovery_rate: float = 0.0
    avg_adaptation_cost: float = 0.0
    avg_time_to_recover_seconds: float = 0.0
    avg_cost_variance: float = 0.0
    avg_recovery_time_variance: float = 0.0
    avg_intervention_count: float = 0.0
    avg_confidence: float = 0.0
    avg_drift_score: float = 0.0
    content_degradation_task_count: int = 0
    confidence_degradation_task_count: int = 0
    provider_degradation_task_count: int = 0
    persistent_degradation_task_count: int = 0
    drift_task_count: int = 0
    intervention_task_count: int = 0
    high_adaptation_cost_task_count: int = 0
    drift_rate: float = 0.0
    intervention_rate: float = 0.0
    high_cost_rate: float = 0.0
    confidence_rate: float = 0.0
    stability_score: float = 1.0
    source_distribution: Dict[str, float] = field(default_factory=dict)
    kind_distribution: Dict[str, float] = field(default_factory=dict)
    failure_type_distribution: Dict[str, float] = field(default_factory=dict)
    provider_distribution: Dict[str, float] = field(default_factory=dict)
    reason_distribution: Dict[str, float] = field(default_factory=dict)

    @property
    def support(self) -> float:
        """Normalized trust in the long-horizon degradation signal."""
        return round(
            max(
                0.0,
                min(
                    1.0,
                    (self.coverage * 0.45)
                    + (min(1.0, self.event_count / 8.0) * 0.2)
                    + (min(1.0, self.degraded_task_count / 4.0) * 0.2)
                    + ((1.0 - self.stability_score) * 0.15),
                ),
            ),
            4,
        )

    @property
    def severity_score(self) -> float:
        """Compress drift, intervention, and recovery cost into one bounded penalty."""
        return round(
            max(
                0.0,
                min(
                    1.0,
                    (self.avg_drift_score * 0.35)
                    + (self.drift_rate * 0.2)
                    + (self.intervention_rate * 0.15)
                    + (self.high_cost_rate * 0.1)
                    + (min(1.0, self.avg_adaptation_cost / 4.0) * 0.1)
                    + (min(1.0, self.avg_cost_variance / 4.0) * 0.05)
                    + ((1.0 - self.stability_score) * 0.05),
                ),
            ),
            4,
        )

    @property
    def dominant_source(self) -> Optional[str]:
        return RuntimeIntelligenceService._preferred_label(self.source_distribution)

    @property
    def dominant_kind(self) -> Optional[str]:
        return RuntimeIntelligenceService._preferred_label(self.kind_distribution)

    @property
    def dominant_failure_type(self) -> Optional[str]:
        return RuntimeIntelligenceService._preferred_label(self.failure_type_distribution)

    @property
    def dominant_provider(self) -> Optional[str]:
        return RuntimeIntelligenceService._preferred_label(self.provider_distribution)

    @property
    def recommends_conservative_routing(self) -> bool:
        """Whether the runtime should treat this scope as degradation-prone."""
        return (
            self.severity_score >= 0.45
            or self.persistent_degradation_task_count > 0
            or self.drift_rate >= 0.5
            or self.high_cost_rate >= 0.35
        )

    @property
    def recommends_recovery_budget_buffer(self) -> bool:
        """Whether the runtime should expect retries or extra recovery work."""
        return (
            self.intervention_rate >= 0.25
            or self.avg_adaptation_cost >= 1.5
            or self.avg_time_to_recover_seconds >= 3.0
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize the learned degradation preference for logs and snapshots."""
        return {
            "coverage": round(self.coverage, 4),
            "support": round(self.support, 4),
            "event_count": self.event_count,
            "degraded_task_count": self.degraded_task_count,
            "recovered_task_count": self.recovered_task_count,
            "recovery_rate": round(self.recovery_rate, 4),
            "avg_adaptation_cost": round(self.avg_adaptation_cost, 4),
            "avg_time_to_recover_seconds": round(self.avg_time_to_recover_seconds, 4),
            "avg_cost_variance": round(self.avg_cost_variance, 4),
            "avg_recovery_time_variance": round(self.avg_recovery_time_variance, 4),
            "avg_intervention_count": round(self.avg_intervention_count, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_drift_score": round(self.avg_drift_score, 4),
            "content_degradation_task_count": self.content_degradation_task_count,
            "confidence_degradation_task_count": self.confidence_degradation_task_count,
            "provider_degradation_task_count": self.provider_degradation_task_count,
            "persistent_degradation_task_count": self.persistent_degradation_task_count,
            "drift_task_count": self.drift_task_count,
            "intervention_task_count": self.intervention_task_count,
            "high_adaptation_cost_task_count": self.high_adaptation_cost_task_count,
            "drift_rate": round(self.drift_rate, 4),
            "intervention_rate": round(self.intervention_rate, 4),
            "high_cost_rate": round(self.high_cost_rate, 4),
            "confidence_rate": round(self.confidence_rate, 4),
            "stability_score": round(self.stability_score, 4),
            "severity_score": round(self.severity_score, 4),
            "dominant_source": self.dominant_source,
            "dominant_kind": self.dominant_kind,
            "dominant_failure_type": self.dominant_failure_type,
            "dominant_provider": self.dominant_provider,
            "recommends_conservative_routing": self.recommends_conservative_routing,
            "recommends_recovery_budget_buffer": self.recommends_recovery_budget_buffer,
            "source_distribution": dict(self.source_distribution),
            "kind_distribution": dict(self.kind_distribution),
            "failure_type_distribution": dict(self.failure_type_distribution),
            "provider_distribution": dict(self.provider_distribution),
            "reason_distribution": dict(self.reason_distribution),
        }

    def to_routing_context(
        self,
        *,
        min_coverage: float = 0.2,
        min_support: float = 0.2,
    ) -> Dict[str, Any]:
        """Convert long-horizon degradation feedback into soft routing hints."""
        if self.coverage < min_coverage or self.support < min_support or self.event_count <= 0:
            return {}

        routing_context: Dict[str, Any] = {
            "learned_degradation_support": round(self.support, 4),
            "degradation_feedback_coverage": round(self.coverage, 4),
            "learned_degradation_severity_score": round(self.severity_score, 4),
            "learned_degradation_stability_score": round(self.stability_score, 4),
            "learned_degradation_event_count": self.event_count,
            "learned_degradation_recovery_rate": round(self.recovery_rate, 4),
            "learned_degradation_drift_rate": round(self.drift_rate, 4),
            "learned_degradation_intervention_rate": round(self.intervention_rate, 4),
            "learned_degradation_high_cost_rate": round(self.high_cost_rate, 4),
            "learned_degradation_confidence_rate": round(self.confidence_rate, 4),
            "learned_degradation_avg_adaptation_cost": round(self.avg_adaptation_cost, 4),
            "learned_degradation_avg_time_to_recover_seconds": round(
                self.avg_time_to_recover_seconds,
                4,
            ),
            "learned_degradation_avg_drift_score": round(self.avg_drift_score, 4),
            "learned_degradation_persistent_task_count": self.persistent_degradation_task_count,
            "learned_degradation_intervention_task_count": self.intervention_task_count,
        }
        if self.dominant_source:
            routing_context["learned_degradation_dominant_source"] = self.dominant_source
        if self.dominant_kind:
            routing_context["learned_degradation_dominant_kind"] = self.dominant_kind
        if self.dominant_failure_type:
            routing_context["learned_degradation_dominant_failure_type"] = (
                self.dominant_failure_type
            )
        if self.dominant_provider:
            routing_context["learned_degradation_dominant_provider"] = self.dominant_provider
        if self.recommends_conservative_routing:
            routing_context["learned_degradation_conservative_routing_hint"] = True
        if self.recommends_recovery_budget_buffer:
            routing_context["learned_degradation_recovery_buffer_hint"] = True
        return routing_context


class RuntimeIntelligenceService:
    """Service-first boundary for runtime task intelligence."""

    def __init__(
        self,
        task_analyzer: Optional[TaskAnalyzer] = None,
        perception_integration: Optional[PerceptionIntegration] = None,
        optimization_injector: Optional[OptimizationInjector] = None,
        decision_service: Optional["LLMDecisionServiceProtocol"] = None,
        evaluation_policy: Optional[RuntimeEvaluationPolicy] = None,
        evaluation_feedback_path: Optional[Any] = None,
        evaluation_feedback_scope: Optional[Any] = None,
        experiment_memory_path: Optional[Any] = None,
    ) -> None:
        derived_policy = evaluation_policy
        if derived_policy is None and perception_integration is not None:
            derived_policy = getattr(perception_integration, "evaluation_policy", None)
        if not isinstance(derived_policy, RuntimeEvaluationPolicy):
            derived_policy = None
        resolved_policy = derived_policy or RuntimeEvaluationPolicy()
        locked_feedback_fields = self._resolve_locked_feedback_fields(perception_integration)
        resolved_feedback_scope = self._resolve_feedback_scope(
            evaluation_feedback_scope,
            perception_integration,
        )
        self._evaluation_feedback_path = evaluation_feedback_path
        self._evaluation_feedback_scope = resolved_feedback_scope
        persisted_feedback = self._load_persisted_feedback(
            evaluation_feedback_path,
            resolved_feedback_scope,
        )
        if persisted_feedback is not None:
            resolved_policy = self._apply_feedback(
                resolved_policy,
                persisted_feedback,
                locked_fields=locked_feedback_fields,
            )
        decision_feedback = None
        if decision_service is not None:
            decision_feedback = self._resolve_runtime_feedback(decision_service)
            if decision_feedback is not None:
                resolved_policy = self._apply_feedback(
                    resolved_policy,
                    decision_feedback,
                    locked_fields=locked_feedback_fields,
                )
        self._runtime_feedback_metadata = self._merge_feedback_metadata(
            getattr(persisted_feedback, "metadata", None),
            getattr(decision_feedback, "metadata", None),
        )
        self._experiment_memory_path = self._resolve_experiment_memory_path(
            explicit_path=experiment_memory_path,
            evaluation_feedback_path=evaluation_feedback_path,
        )
        self._experiment_memory_store = self._load_experiment_memory_store(
            self._experiment_memory_path
        )
        self._persisted_topology_routing_feedback = self._extract_topology_routing_feedback(
            self._runtime_feedback_metadata
        )
        self._persisted_team_routing_feedback = self._extract_team_routing_feedback(
            self._runtime_feedback_metadata
        )
        self._persisted_degradation_routing_feedback = self._extract_degradation_routing_feedback(
            self._runtime_feedback_metadata
        )
        self._session_topology_feedback_records: List[Dict[str, Any]] = []
        self._session_topology_routing_feedback: Optional[TopologyRoutingFeedback] = None
        self._evaluation_policy = resolved_policy
        self._task_analyzer = task_analyzer or get_task_analyzer()
        self._perception_integration = perception_integration or PerceptionIntegration(
            config=self._evaluation_policy.to_config()
        )
        self._synchronize_perception_policy(self._perception_integration)
        self._optimization_injector = optimization_injector
        self._decision_service = decision_service
        if hasattr(self._task_analyzer, "set_runtime_intelligence"):
            try:
                self._task_analyzer.set_runtime_intelligence(self)
            except Exception as exc:
                logger.debug("Runtime intelligence could not bind task analyzer: %s", exc)

    @classmethod
    def from_orchestrator(
        cls,
        orchestrator: Any,
        *,
        task_analyzer: Optional[TaskAnalyzer] = None,
        perception_integration: Optional[PerceptionIntegration] = None,
        optimization_injector: Optional[OptimizationInjector] = None,
        evaluation_feedback_path: Optional[Any] = None,
        evaluation_feedback_scope: Optional[Any] = None,
        experiment_memory_path: Optional[Any] = None,
    ) -> "RuntimeIntelligenceService":
        """Create a service instance by resolving the decision service from an orchestrator."""
        container = getattr(orchestrator, "_container", None)
        if container is None:
            return cls(
                task_analyzer=task_analyzer,
                perception_integration=perception_integration,
                optimization_injector=optimization_injector,
                evaluation_feedback_path=evaluation_feedback_path,
                evaluation_feedback_scope=evaluation_feedback_scope,
                experiment_memory_path=experiment_memory_path,
            )

        return cls.from_container(
            container,
            task_analyzer=task_analyzer,
            perception_integration=perception_integration,
            optimization_injector=optimization_injector,
            evaluation_feedback_path=evaluation_feedback_path,
            evaluation_feedback_scope=evaluation_feedback_scope,
            experiment_memory_path=experiment_memory_path,
        )

    @classmethod
    def from_container(
        cls,
        container: Any,
        *,
        task_analyzer: Optional[TaskAnalyzer] = None,
        perception_integration: Optional[PerceptionIntegration] = None,
        optimization_injector: Optional[OptimizationInjector] = None,
        evaluation_feedback_path: Optional[Any] = None,
        evaluation_feedback_scope: Optional[Any] = None,
        experiment_memory_path: Optional[Any] = None,
    ) -> "RuntimeIntelligenceService":
        """Create a service instance by resolving the decision service from a container."""
        decision_service = cls._resolve_decision_service(container)
        return cls(
            task_analyzer=task_analyzer,
            perception_integration=perception_integration,
            optimization_injector=optimization_injector,
            decision_service=decision_service,
            evaluation_feedback_path=evaluation_feedback_path,
            evaluation_feedback_scope=evaluation_feedback_scope,
            experiment_memory_path=experiment_memory_path,
        )

    @staticmethod
    def _resolve_decision_service(
        container: Any,
    ) -> Optional["LLMDecisionServiceProtocol"]:
        """Resolve the decision service from a DI container when available."""
        from victor.agent.services.protocols.decision_service import (
            get_decision_service,
        )

        return get_decision_service(container)

    @staticmethod
    def _resolve_runtime_feedback(decision_service: Any) -> Optional[Any]:
        """Resolve runtime evaluation feedback from a decision service when supported."""
        if decision_service is None or not hasattr(
            decision_service, "get_runtime_evaluation_feedback"
        ):
            return None
        try:
            return decision_service.get_runtime_evaluation_feedback()
        except Exception as exc:
            logger.debug(
                "Runtime intelligence could not resolve decision-service feedback: %s",
                exc,
            )
            return None

    @staticmethod
    def _resolve_feedback_scope(
        explicit_scope: Optional[Any],
        perception_integration: Any,
    ) -> RuntimeEvaluationFeedbackScope:
        """Resolve the requested runtime-feedback scope from explicit input or config."""
        resolved_scope = RuntimeEvaluationFeedbackScope.from_value(explicit_scope)
        if not resolved_scope.is_empty():
            return resolved_scope
        config = getattr(perception_integration, "config", None)
        return runtime_evaluation_feedback_scope_from_context(config)

    @staticmethod
    def _load_persisted_feedback(
        path: Optional[Any],
        scope: Optional[Any],
    ) -> Optional[Any]:
        """Load persisted benchmark-truth feedback when available."""
        try:
            return load_runtime_evaluation_feedback(path, scope=scope)
        except Exception as exc:
            logger.debug("Runtime intelligence could not load persisted feedback: %s", exc)
            return None

    @staticmethod
    def _resolve_experiment_memory_path(
        *,
        explicit_path: Optional[Any],
        evaluation_feedback_path: Optional[Any],
    ) -> Optional[Path]:
        """Resolve the experiment-memory artifact path next to runtime feedback by default."""
        candidate = explicit_path if explicit_path is not None else evaluation_feedback_path
        if candidate is None:
            return None
        try:
            path = Path(candidate)
        except TypeError:
            return None
        if explicit_path is not None:
            return path
        if path.suffix:
            return path.parent / "experiment_memory.json"
        return path / "experiment_memory.json"

    @staticmethod
    def _load_experiment_memory_store(
        path: Optional[Path],
    ) -> Optional[ExperimentMemoryStore]:
        """Load the persisted experiment-memory store when available."""
        if path is None:
            return None
        try:
            return ExperimentMemoryStore(persist_path=path)
        except Exception as exc:
            logger.debug("Runtime intelligence could not load experiment memory: %s", exc)
            return None

    @staticmethod
    def _resolve_locked_feedback_fields(perception_integration: Any) -> set[str]:
        """Keep explicit runtime config authoritative over persisted calibration overlays."""
        config = getattr(perception_integration, "config", None)
        if not isinstance(config, dict):
            return set()
        return {key for key in _FEEDBACK_OVERRIDE_FIELDS if config.get(key) is not None}

    @staticmethod
    def _apply_feedback(
        policy: RuntimeEvaluationPolicy,
        feedback: Any,
        *,
        locked_fields: Optional[set[str]] = None,
    ) -> RuntimeEvaluationPolicy:
        """Apply runtime feedback without overriding explicitly configured thresholds."""
        if feedback is None:
            return policy

        locked = locked_fields or set()
        overrides = {
            field_name: getattr(feedback, field_name, None)
            for field_name in _FEEDBACK_OVERRIDE_FIELDS
            if field_name not in locked and getattr(feedback, field_name, None) is not None
        }
        if not overrides:
            return policy
        return policy.with_overrides(**overrides)

    @staticmethod
    def _merge_feedback_metadata(*metadata_payloads: Any) -> Dict[str, Any]:
        """Merge persisted and live feedback metadata for downstream routing hints."""
        merged: Dict[str, Any] = {}
        sources: List[str] = []
        for metadata in metadata_payloads:
            if not isinstance(metadata, dict):
                continue
            source = metadata.get("source")
            if isinstance(source, str) and source:
                sources.append(source)
            for key, value in metadata.items():
                if value in (None, "", {}, []):
                    continue
                merged[key] = value
        if sources:
            merged["feedback_sources"] = list(dict.fromkeys(sources))
        return merged

    @staticmethod
    def _coerce_feedback_float(value: Any) -> float:
        """Normalize optional feedback metrics into bounded floats."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _coerce_non_negative_float(value: Any) -> float:
        """Normalize optional feedback metrics into non-negative floats."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, numeric)

    @staticmethod
    def _coerce_feedback_int(value: Any) -> int:
        """Normalize optional feedback counts into non-negative integers."""
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, numeric)

    @staticmethod
    def _normalize_distribution(mapping: Any) -> Dict[str, float]:
        """Normalize serialized count mappings into compact float distributions."""
        if not isinstance(mapping, dict):
            return {}
        normalized: Dict[str, float] = {}
        for key, value in mapping.items():
            label = str(key).strip()
            if not label:
                continue
            try:
                count = float(value)
            except (TypeError, ValueError):
                continue
            if count <= 0.0:
                continue
            normalized[label] = round(count, 4)
        return normalized

    @staticmethod
    def _preferred_label(distribution: Any) -> Optional[str]:
        """Return the most-supported label from a serialized count mapping."""
        if not isinstance(distribution, dict):
            return None
        best_label: Optional[str] = None
        best_count = float("-inf")
        for key, value in distribution.items():
            label = str(key).strip()
            if not label:
                continue
            try:
                count = float(value)
            except (TypeError, ValueError):
                continue
            if count > best_count:
                best_label = label
                best_count = count
        return best_label

    @staticmethod
    def _distribution_agreement(distribution: Any) -> float:
        """Estimate agreement from the dominant label share in a distribution."""
        if not isinstance(distribution, dict):
            return 0.0
        total = 0.0
        dominant = 0.0
        for value in distribution.values():
            try:
                count = float(value)
            except (TypeError, ValueError):
                continue
            if count <= 0.0:
                continue
            total += count
            dominant = max(dominant, count)
        if total <= 0.0:
            return 0.0
        return round(dominant / total, 4)

    @staticmethod
    def _topology_conflict_score(
        *,
        action_agreement: float,
        topology_agreement: float,
        provider_agreement: float,
        formation_agreement: float,
    ) -> float:
        """Compress dimension-level agreement into one conflict score."""
        weighted_agreements = [
            (action_agreement, 0.35),
            (topology_agreement, 0.30),
            (provider_agreement, 0.20),
            (formation_agreement, 0.15),
        ]
        usable = [(value, weight) for value, weight in weighted_agreements if value > 0.0]
        if not usable:
            return 0.0
        weighted_consensus = sum(value * weight for value, weight in usable) / sum(
            weight for _, weight in usable
        )
        return round(max(0.0, min(1.0, 1.0 - weighted_consensus)), 4)

    @classmethod
    def _extract_topology_routing_feedback(
        cls,
        metadata: Dict[str, Any],
    ) -> Optional[TopologyRoutingFeedback]:
        """Build learned topology routing feedback from persisted runtime metadata."""
        if not isinstance(metadata, dict):
            return None

        coverage = cls._coerce_feedback_float(metadata.get("topology_feedback_coverage"))
        avg_reward = cls._coerce_feedback_float(metadata.get("avg_topology_reward"))
        avg_confidence = cls._coerce_feedback_float(metadata.get("avg_topology_confidence"))
        observation_count_hint = cls._coerce_feedback_int(
            metadata.get("topology_observation_count") or metadata.get("task_count")
        )
        action_distribution = cls._normalize_distribution(
            metadata.get("topology_final_actions") or metadata.get("topology_actions") or {}
        )
        topology_distribution = cls._normalize_distribution(
            metadata.get("topology_final_kinds") or metadata.get("topology_kinds") or {}
        )
        execution_mode_distribution = cls._normalize_distribution(
            metadata.get("topology_execution_modes") or {}
        )
        provider_distribution = cls._normalize_distribution(
            metadata.get("topology_providers") or {}
        )
        formation_distribution = cls._normalize_distribution(
            metadata.get("topology_formations") or {}
        )
        selection_policy_distribution = cls._normalize_distribution(
            metadata.get("topology_selection_policies") or {}
        )
        selection_policy_reward_totals = cls._normalize_distribution(
            metadata.get("topology_selection_policy_reward_totals") or {}
        )
        selection_policy_optimization_counts = cls._normalize_distribution(
            metadata.get("topology_selection_policy_optimization_counts") or {}
        )
        selection_policy_optimization_reward_totals = cls._normalize_distribution(
            metadata.get("topology_selection_policy_optimization_reward_totals") or {}
        )
        selection_policy_feasible_counts = cls._normalize_distribution(
            metadata.get("topology_selection_policy_feasible_counts") or {}
        )
        selection_policy_scope_metrics = dict(
            metadata.get("topology_selection_policy_scope_metrics") or {}
        )
        action_agreement = cls._coerce_feedback_float(
            metadata.get("topology_action_agreement")
        ) or cls._distribution_agreement(action_distribution)
        topology_agreement = cls._coerce_feedback_float(
            metadata.get("topology_kind_agreement")
        ) or cls._distribution_agreement(topology_distribution)
        provider_agreement = cls._coerce_feedback_float(
            metadata.get("topology_provider_agreement")
        ) or cls._distribution_agreement(provider_distribution)
        formation_agreement = cls._coerce_feedback_float(
            metadata.get("topology_formation_agreement")
        ) or cls._distribution_agreement(formation_distribution)
        conflict_score = cls._coerce_feedback_float(
            metadata.get("topology_conflict_score")
        ) or cls._topology_conflict_score(
            action_agreement=action_agreement,
            topology_agreement=topology_agreement,
            provider_agreement=provider_agreement,
            formation_agreement=formation_agreement,
        )

        if not any(
            (
                coverage > 0.0,
                avg_reward > 0.0,
                action_distribution,
                topology_distribution,
                execution_mode_distribution,
                provider_distribution,
                formation_distribution,
            )
        ):
            return None

        return TopologyRoutingFeedback(
            coverage=coverage,
            avg_reward=avg_reward,
            avg_confidence=avg_confidence,
            observation_count_hint=observation_count_hint,
            action_agreement=action_agreement,
            topology_agreement=topology_agreement,
            provider_agreement=provider_agreement,
            formation_agreement=formation_agreement,
            conflict_score=conflict_score,
            preferred_action=cls._preferred_label(action_distribution),
            preferred_topology=cls._preferred_label(topology_distribution),
            preferred_execution_mode=cls._preferred_label(execution_mode_distribution),
            preferred_provider=cls._preferred_label(provider_distribution),
            preferred_formation=cls._preferred_label(formation_distribution),
            action_distribution=action_distribution,
            topology_distribution=topology_distribution,
            execution_mode_distribution=execution_mode_distribution,
            provider_distribution=provider_distribution,
            formation_distribution=formation_distribution,
            selection_policy_distribution=selection_policy_distribution,
            selection_policy_reward_totals=selection_policy_reward_totals,
            selection_policy_optimization_counts=selection_policy_optimization_counts,
            selection_policy_optimization_reward_totals=selection_policy_optimization_reward_totals,
            selection_policy_feasible_counts=selection_policy_feasible_counts,
            selection_policy_scope_metrics=selection_policy_scope_metrics,
        )

    @classmethod
    def _extract_team_routing_feedback(
        cls,
        metadata: Dict[str, Any],
    ) -> Optional[TeamRoutingFeedback]:
        """Build learned team/worktree routing feedback from persisted runtime metadata."""
        return cls._extract_team_routing_feedback_from_mapping(metadata)

    @classmethod
    def _extract_team_routing_feedback_from_mapping(
        cls,
        metadata: Dict[str, Any],
        *,
        scope_dimension: Optional[str] = None,
        scope_label: Optional[str] = None,
    ) -> Optional[TeamRoutingFeedback]:
        """Build learned team/worktree routing feedback from one metadata mapping."""
        if not isinstance(metadata, dict):
            return None

        coverage = cls._coerce_feedback_float(metadata.get("team_feedback_coverage"))
        task_count = cls._coerce_feedback_int(metadata.get("tasks_with_team_feedback"))
        worktree_plan_count = cls._coerce_feedback_int(metadata.get("team_worktree_plan_count"))
        worktree_materialized_count = cls._coerce_feedback_int(
            metadata.get("team_worktree_materialized_count")
        )
        worktree_dry_run_count = cls._coerce_feedback_int(
            metadata.get("team_worktree_dry_run_count")
        )
        low_risk_task_count = cls._coerce_feedback_int(metadata.get("team_low_risk_task_count"))
        medium_risk_task_count = cls._coerce_feedback_int(
            metadata.get("team_medium_risk_task_count")
        )
        high_risk_task_count = cls._coerce_feedback_int(metadata.get("team_high_risk_task_count"))
        merge_conflict_task_count = cls._coerce_feedback_int(
            metadata.get("team_merge_conflict_task_count")
        )
        cleanup_task_count = cls._coerce_feedback_int(metadata.get("team_cleanup_task_count"))
        cleanup_error_task_count = cls._coerce_feedback_int(
            metadata.get("team_cleanup_error_task_count")
        )
        avg_team_assignments = cls._coerce_non_negative_float(metadata.get("avg_team_assignments"))
        avg_team_scoped_members = cls._coerce_non_negative_float(
            metadata.get("avg_team_scoped_members")
        )
        avg_team_members_with_changes = cls._coerce_non_negative_float(
            metadata.get("avg_team_members_with_changes")
        )
        avg_team_changed_file_count = cls._coerce_non_negative_float(
            metadata.get("avg_team_changed_file_count")
        )
        formation_distribution = cls._normalize_distribution(metadata.get("team_formations") or {})
        merge_risk_distribution = cls._normalize_distribution(
            metadata.get("team_merge_risk_levels") or {}
        )

        if not any(
            (
                coverage > 0.0,
                task_count > 0,
                worktree_plan_count > 0,
                formation_distribution,
                merge_risk_distribution,
            )
        ):
            return None

        return TeamRoutingFeedback(
            coverage=coverage,
            task_count=task_count,
            worktree_plan_count=worktree_plan_count,
            worktree_materialized_count=worktree_materialized_count,
            worktree_dry_run_count=worktree_dry_run_count,
            low_risk_task_count=low_risk_task_count,
            medium_risk_task_count=medium_risk_task_count,
            high_risk_task_count=high_risk_task_count,
            merge_conflict_task_count=merge_conflict_task_count,
            cleanup_task_count=cleanup_task_count,
            cleanup_error_task_count=cleanup_error_task_count,
            avg_team_assignments=avg_team_assignments,
            avg_team_scoped_members=avg_team_scoped_members,
            avg_team_members_with_changes=avg_team_members_with_changes,
            avg_team_changed_file_count=avg_team_changed_file_count,
            formation_distribution=formation_distribution,
            merge_risk_distribution=merge_risk_distribution,
            scope_dimension=scope_dimension,
            scope_label=scope_label,
        )

    @classmethod
    def _extract_degradation_routing_feedback(
        cls,
        metadata: Dict[str, Any],
    ) -> Optional[DegradationRoutingFeedback]:
        """Build long-horizon degradation routing feedback from runtime metadata."""
        if not isinstance(metadata, dict):
            return None

        coverage = cls._coerce_feedback_float(metadata.get("degradation_feedback_coverage"))
        event_count = cls._coerce_feedback_int(metadata.get("degradation_event_count"))
        degraded_task_count = cls._coerce_feedback_int(metadata.get("degraded_task_count"))
        recovered_task_count = cls._coerce_feedback_int(metadata.get("recovered_task_count"))
        recovery_rate = cls._coerce_feedback_float(metadata.get("degradation_recovery_rate"))
        avg_adaptation_cost = cls._coerce_non_negative_float(
            metadata.get("avg_degradation_adaptation_cost")
        )
        avg_time_to_recover_seconds = cls._coerce_non_negative_float(
            metadata.get("avg_degradation_time_to_recover_seconds")
        )
        avg_cost_variance = cls._coerce_non_negative_float(
            metadata.get("avg_degradation_cost_variance")
        )
        avg_recovery_time_variance = cls._coerce_non_negative_float(
            metadata.get("avg_degradation_recovery_time_variance")
        )
        avg_intervention_count = cls._coerce_non_negative_float(
            metadata.get("avg_degradation_intervention_count")
        )
        avg_confidence = cls._coerce_feedback_float(metadata.get("avg_degradation_confidence"))
        avg_drift_score = cls._coerce_feedback_float(metadata.get("avg_degradation_drift_score"))
        content_degradation_task_count = cls._coerce_feedback_int(
            metadata.get("content_degradation_task_count")
        )
        confidence_degradation_task_count = cls._coerce_feedback_int(
            metadata.get("confidence_degradation_task_count")
        )
        provider_degradation_task_count = cls._coerce_feedback_int(
            metadata.get("provider_degradation_task_count")
        )
        persistent_degradation_task_count = cls._coerce_feedback_int(
            metadata.get("persistent_degradation_task_count")
        )
        drift_task_count = cls._coerce_feedback_int(metadata.get("drift_task_count"))
        intervention_task_count = cls._coerce_feedback_int(
            metadata.get("degradation_intervention_task_count")
        )
        high_adaptation_cost_task_count = cls._coerce_feedback_int(
            metadata.get("high_adaptation_cost_task_count")
        )
        drift_rate = cls._coerce_feedback_float(metadata.get("degradation_drift_rate"))
        intervention_rate = cls._coerce_feedback_float(
            metadata.get("degradation_intervention_rate")
        )
        high_cost_rate = cls._coerce_feedback_float(metadata.get("degradation_high_cost_rate"))
        confidence_rate = cls._coerce_feedback_float(metadata.get("degradation_confidence_rate"))
        stability_score_raw = metadata.get("degradation_stability_score")
        if stability_score_raw is None:
            stability_score = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - (drift_rate * 0.35)
                    - (intervention_rate * 0.15)
                    - (high_cost_rate * 0.15)
                    - (min(1.0, avg_adaptation_cost / 4.0) * 0.15)
                    - (min(1.0, avg_cost_variance / 4.0) * 0.1)
                    - (confidence_rate * 0.05)
                    + (recovery_rate * 0.1),
                ),
            )
        else:
            stability_score = cls._coerce_feedback_float(stability_score_raw)

        source_distribution = cls._normalize_distribution(metadata.get("degradation_sources") or {})
        kind_distribution = cls._normalize_distribution(metadata.get("degradation_kinds") or {})
        failure_type_distribution = cls._normalize_distribution(
            metadata.get("degradation_failure_types") or {}
        )
        provider_distribution = cls._normalize_distribution(
            metadata.get("degradation_providers") or {}
        )
        reason_distribution = cls._normalize_distribution(metadata.get("degradation_reasons") or {})

        if not any(
            (
                coverage > 0.0,
                event_count > 0,
                degraded_task_count > 0,
                drift_task_count > 0,
                source_distribution,
                kind_distribution,
            )
        ):
            return None

        return DegradationRoutingFeedback(
            coverage=coverage,
            event_count=event_count,
            degraded_task_count=degraded_task_count,
            recovered_task_count=recovered_task_count,
            recovery_rate=recovery_rate,
            avg_adaptation_cost=avg_adaptation_cost,
            avg_time_to_recover_seconds=avg_time_to_recover_seconds,
            avg_cost_variance=avg_cost_variance,
            avg_recovery_time_variance=avg_recovery_time_variance,
            avg_intervention_count=avg_intervention_count,
            avg_confidence=avg_confidence,
            avg_drift_score=avg_drift_score,
            content_degradation_task_count=content_degradation_task_count,
            confidence_degradation_task_count=confidence_degradation_task_count,
            provider_degradation_task_count=provider_degradation_task_count,
            persistent_degradation_task_count=persistent_degradation_task_count,
            drift_task_count=drift_task_count,
            intervention_task_count=intervention_task_count,
            high_adaptation_cost_task_count=high_adaptation_cost_task_count,
            drift_rate=drift_rate,
            intervention_rate=intervention_rate,
            high_cost_rate=high_cost_rate,
            confidence_rate=confidence_rate,
            stability_score=round(stability_score, 4),
            source_distribution=source_distribution,
            kind_distribution=kind_distribution,
            failure_type_distribution=failure_type_distribution,
            provider_distribution=provider_distribution,
            reason_distribution=reason_distribution,
        )

    @staticmethod
    def _feedback_observation_weight(feedback: TopologyRoutingFeedback) -> float:
        """Return the weighting factor for merging historical and live feedback."""
        observations = float(feedback.observation_count)
        if observations > 0.0:
            return observations
        return 1.0 if feedback.support > 0.0 else 0.0

    @staticmethod
    def _merge_distributions(*distributions: Dict[str, float]) -> Dict[str, float]:
        """Merge distribution counts by summing counts for matching labels."""
        merged: Dict[str, float] = {}
        for distribution in distributions:
            if not isinstance(distribution, dict):
                continue
            for key, value in distribution.items():
                label = str(key).strip()
                if not label:
                    continue
                try:
                    count = float(value)
                except (TypeError, ValueError):
                    continue
                if count <= 0.0:
                    continue
                merged[label] = round(merged.get(label, 0.0) + count, 4)
        return merged

    @classmethod
    def _merge_topology_routing_feedback(
        cls,
        persisted: Optional[TopologyRoutingFeedback],
        session: Optional[TopologyRoutingFeedback],
    ) -> Optional[TopologyRoutingFeedback]:
        """Merge persisted validated feedback with session-local live observations."""
        if persisted is None:
            return session
        if session is None:
            return persisted

        persisted_weight = cls._feedback_observation_weight(persisted)
        session_weight = cls._feedback_observation_weight(session)
        total_weight = persisted_weight + session_weight
        if total_weight <= 0.0:
            return session

        action_distribution = cls._merge_distributions(
            persisted.action_distribution,
            session.action_distribution,
        )
        topology_distribution = cls._merge_distributions(
            persisted.topology_distribution,
            session.topology_distribution,
        )
        execution_mode_distribution = cls._merge_distributions(
            persisted.execution_mode_distribution,
            session.execution_mode_distribution,
        )
        provider_distribution = cls._merge_distributions(
            persisted.provider_distribution,
            session.provider_distribution,
        )
        formation_distribution = cls._merge_distributions(
            persisted.formation_distribution,
            session.formation_distribution,
        )
        selection_policy_distribution = cls._merge_distributions(
            persisted.selection_policy_distribution,
            session.selection_policy_distribution,
        )
        selection_policy_reward_totals = cls._merge_distributions(
            persisted.selection_policy_reward_totals,
            session.selection_policy_reward_totals,
        )
        selection_policy_optimization_counts = cls._merge_distributions(
            persisted.selection_policy_optimization_counts,
            session.selection_policy_optimization_counts,
        )
        selection_policy_optimization_reward_totals = cls._merge_distributions(
            persisted.selection_policy_optimization_reward_totals,
            session.selection_policy_optimization_reward_totals,
        )
        selection_policy_feasible_counts = cls._merge_distributions(
            persisted.selection_policy_feasible_counts,
            session.selection_policy_feasible_counts,
        )
        selection_policy_scope_metrics = dict(persisted.selection_policy_scope_metrics)
        for dimension, entries in session.selection_policy_scope_metrics.items():
            if not isinstance(entries, dict):
                continue
            dimension_bucket = selection_policy_scope_metrics.setdefault(dimension, {})
            for label, bucket in entries.items():
                if not isinstance(bucket, dict):
                    continue
                existing_bucket = dimension_bucket.get(label) or {}
                merged_policy_counts = cls._merge_distributions(
                    dict(existing_bucket.get("policy_counts") or {}),
                    dict(bucket.get("policy_counts") or {}),
                )
                merged_policy_reward_totals = cls._merge_distributions(
                    dict(existing_bucket.get("policy_reward_totals") or {}),
                    dict(bucket.get("policy_reward_totals") or {}),
                )
                merged_policy_optimization_counts = cls._merge_distributions(
                    dict(existing_bucket.get("policy_optimization_counts") or {}),
                    dict(bucket.get("policy_optimization_counts") or {}),
                )
                merged_policy_optimization_reward_totals = cls._merge_distributions(
                    dict(existing_bucket.get("policy_optimization_reward_totals") or {}),
                    dict(bucket.get("policy_optimization_reward_totals") or {}),
                )
                merged_policy_feasible_counts = cls._merge_distributions(
                    dict(existing_bucket.get("policy_feasible_counts") or {}),
                    dict(bucket.get("policy_feasible_counts") or {}),
                )
                avg_reward_by_policy: Dict[str, float] = {}
                for policy, count_value in merged_policy_counts.items():
                    try:
                        count = float(count_value)
                        reward_total = float(merged_policy_reward_totals.get(policy, 0.0))
                    except (TypeError, ValueError):
                        continue
                    if count <= 0.0:
                        continue
                    avg_reward_by_policy[policy] = round(reward_total / count, 4)
                avg_optimization_reward_by_policy: Dict[str, float] = {}
                for policy, count_value in merged_policy_optimization_counts.items():
                    try:
                        count = float(count_value)
                        reward_total = float(
                            merged_policy_optimization_reward_totals.get(policy, 0.0)
                        )
                    except (TypeError, ValueError):
                        continue
                    if count <= 0.0:
                        continue
                    avg_optimization_reward_by_policy[policy] = round(reward_total / count, 4)
                feasibility_rate_by_policy: Dict[str, float] = {}
                for policy, count_value in merged_policy_optimization_counts.items():
                    try:
                        count = float(count_value)
                        feasible_count = float(merged_policy_feasible_counts.get(policy, 0.0))
                    except (TypeError, ValueError):
                        continue
                    if count <= 0.0:
                        continue
                    feasibility_rate_by_policy[policy] = round(feasible_count / count, 4)
                learned_override_reward_delta = None
                try:
                    learned_override_reward_delta = round(
                        float(avg_reward_by_policy["learned_close_override"])
                        - float(avg_reward_by_policy["heuristic"]),
                        4,
                    )
                except (KeyError, TypeError, ValueError):
                    learned_override_reward_delta = None
                learned_override_optimization_reward_delta = None
                try:
                    learned_override_optimization_reward_delta = round(
                        float(avg_optimization_reward_by_policy["learned_close_override"])
                        - float(avg_optimization_reward_by_policy["heuristic"]),
                        4,
                    )
                except (KeyError, TypeError, ValueError):
                    learned_override_optimization_reward_delta = None
                learned_override_feasibility_delta = None
                try:
                    learned_override_feasibility_delta = round(
                        float(feasibility_rate_by_policy["learned_close_override"])
                        - float(feasibility_rate_by_policy["heuristic"]),
                        4,
                    )
                except (KeyError, TypeError, ValueError):
                    learned_override_feasibility_delta = None
                dimension_bucket[label] = {
                    "policy_counts": merged_policy_counts,
                    "policy_reward_totals": merged_policy_reward_totals,
                    "policy_optimization_counts": merged_policy_optimization_counts,
                    "policy_optimization_reward_totals": (merged_policy_optimization_reward_totals),
                    "policy_feasible_counts": merged_policy_feasible_counts,
                    "avg_reward_by_policy": avg_reward_by_policy,
                    "learned_override_reward_delta": learned_override_reward_delta,
                    "avg_optimization_reward_by_policy": avg_optimization_reward_by_policy,
                    "feasibility_rate_by_policy": feasibility_rate_by_policy,
                    "learned_override_optimization_reward_delta": (
                        learned_override_optimization_reward_delta
                    ),
                    "learned_override_feasibility_delta": learned_override_feasibility_delta,
                }

        def weighted_average(first: float, second: float) -> float:
            return round(
                ((first * persisted_weight) + (second * session_weight)) / total_weight,
                4,
            )

        action_agreement = cls._distribution_agreement(action_distribution)
        topology_agreement = cls._distribution_agreement(topology_distribution)
        provider_agreement = cls._distribution_agreement(provider_distribution)
        formation_agreement = cls._distribution_agreement(formation_distribution)

        return TopologyRoutingFeedback(
            coverage=weighted_average(persisted.coverage, session.coverage),
            avg_reward=weighted_average(persisted.avg_reward, session.avg_reward),
            avg_confidence=weighted_average(persisted.avg_confidence, session.avg_confidence),
            observation_count_hint=persisted.observation_count + session.observation_count,
            action_agreement=action_agreement,
            topology_agreement=topology_agreement,
            provider_agreement=provider_agreement,
            formation_agreement=formation_agreement,
            conflict_score=cls._topology_conflict_score(
                action_agreement=action_agreement,
                topology_agreement=topology_agreement,
                provider_agreement=provider_agreement,
                formation_agreement=formation_agreement,
            ),
            preferred_action=cls._preferred_label(action_distribution),
            preferred_topology=cls._preferred_label(topology_distribution),
            preferred_execution_mode=cls._preferred_label(execution_mode_distribution),
            preferred_provider=cls._preferred_label(provider_distribution),
            preferred_formation=cls._preferred_label(formation_distribution),
            action_distribution=action_distribution,
            topology_distribution=topology_distribution,
            execution_mode_distribution=execution_mode_distribution,
            provider_distribution=provider_distribution,
            formation_distribution=formation_distribution,
            selection_policy_distribution=selection_policy_distribution,
            selection_policy_reward_totals=selection_policy_reward_totals,
            selection_policy_optimization_counts=selection_policy_optimization_counts,
            selection_policy_optimization_reward_totals=selection_policy_optimization_reward_totals,
            selection_policy_feasible_counts=selection_policy_feasible_counts,
            selection_policy_scope_metrics=selection_policy_scope_metrics,
        )

    def _synchronize_perception_policy(self, perception_integration: Any) -> None:
        """Align the underlying perception integration with the shared runtime policy."""
        if perception_integration is None:
            return
        try:
            if hasattr(perception_integration, "_evaluation_policy"):
                perception_integration._evaluation_policy = self._evaluation_policy
            if hasattr(perception_integration, "config") and isinstance(
                perception_integration.config, dict
            ):
                perception_integration.config.update(self._evaluation_policy.to_config())
        except Exception as exc:
            logger.debug("Runtime intelligence could not synchronize perception policy: %s", exc)

    @property
    def perception_integration(self) -> PerceptionIntegration:
        """Expose the underlying perception integration for compatibility."""
        return self._perception_integration

    @property
    def task_analyzer(self) -> TaskAnalyzer:
        """Expose the task analyzer for compatibility."""
        return self._task_analyzer

    @property
    def evaluation_policy(self) -> RuntimeEvaluationPolicy:
        """Expose the canonical runtime evaluation policy."""
        return self._evaluation_policy

    def get_topology_routing_feedback(self) -> Optional[TopologyRoutingFeedback]:
        """Return learned topology preferences derived from runtime feedback metadata."""
        return self._merge_topology_routing_feedback(
            self._persisted_topology_routing_feedback,
            self._session_topology_routing_feedback,
        )

    def get_team_routing_feedback(self) -> Optional[TeamRoutingFeedback]:
        """Return learned team/worktree preferences derived from runtime feedback metadata."""
        return self._persisted_team_routing_feedback

    def get_degradation_routing_feedback(self) -> Optional[DegradationRoutingFeedback]:
        """Return long-horizon degradation preferences derived from runtime feedback metadata."""
        return self._persisted_degradation_routing_feedback

    def get_structured_routing_policy(
        self,
        *,
        query: Optional[str] = None,
        scope_context: Optional[Dict[str, Any]] = None,
        min_coverage: float = 0.2,
        min_reward: float = 0.45,
        min_observations: int = 2,
    ) -> StructuredRoutingPolicy:
        """Return one typed routing policy spanning topology, planning, and recovery."""
        resolved_scope_context = self._resolve_routing_scope_context(scope_context)

        topology_hints: Dict[str, Any] = {}
        topology_feedback = self.get_topology_routing_feedback()
        if topology_feedback is not None:
            topology_hints.update(
                topology_feedback.to_routing_context(
                    min_coverage=min_coverage,
                    min_reward=min_reward,
                    min_observations=min_observations,
                    scope_context=resolved_scope_context,
                )
            )

        team_hints: Dict[str, Any] = {}
        team_feedback = self.resolve_team_routing_feedback(scope_context=resolved_scope_context)
        if team_feedback is not None:
            team_hints.update(team_feedback.to_routing_context())
        global_team_feedback = self.get_team_routing_feedback()
        if global_team_feedback is not None and global_team_feedback is not team_feedback:
            for key, value in global_team_feedback.to_routing_context().items():
                team_hints.setdefault(key, value)

        degradation_hints: Dict[str, Any] = {}
        degradation_feedback = self.get_degradation_routing_feedback()
        if degradation_feedback is not None:
            degradation_hints.update(degradation_feedback.to_routing_context())

        experiment_hints = self.get_experiment_routing_context(
            query=query,
            scope_context=resolved_scope_context,
        )
        planning_hints = self.get_planning_routing_context(
            query=query,
            scope_context=resolved_scope_context,
        )

        metadata: Dict[str, Any] = {}
        if topology_feedback is not None:
            metadata["topology_feedback"] = topology_feedback.to_metadata()
        if team_feedback is not None:
            metadata["team_feedback"] = team_feedback.to_metadata()
        elif global_team_feedback is not None:
            metadata["team_feedback"] = global_team_feedback.to_metadata()
        if degradation_feedback is not None:
            metadata["degradation_feedback"] = degradation_feedback.to_metadata()
        if experiment_hints:
            metadata["experiment_hints"] = dict(experiment_hints)
        if planning_hints:
            metadata["planning_hints"] = dict(planning_hints)

        return StructuredRoutingPolicy(
            scope_context=dict(resolved_scope_context),
            topology_hints=topology_hints,
            team_hints=team_hints,
            degradation_hints=degradation_hints,
            experiment_hints=dict(experiment_hints),
            planning_hints=dict(planning_hints),
            metadata=metadata,
        )

    def resolve_team_routing_feedback(
        self,
        *,
        scope_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[TeamRoutingFeedback]:
        """Resolve the most relevant team/worktree routing feedback for a runtime scope."""
        scoped_feedback = self._resolve_scoped_team_routing_feedback(scope_context=scope_context)
        if scoped_feedback is not None:
            return scoped_feedback
        return self.get_team_routing_feedback()

    def _resolve_scoped_team_routing_feedback(
        self,
        *,
        scope_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[TeamRoutingFeedback]:
        """Resolve scoped team/worktree routing feedback for the requested runtime context."""
        scope_metrics = self._runtime_feedback_metadata.get("team_worktree_scope_metrics") or {}
        if not isinstance(scope_metrics, dict):
            return None
        for dimension in ("model_family", "provider", "task_type"):
            label = TopologyRoutingFeedback._scope_context_label(
                scope_context,
                dimension=dimension,
            )
            if label is None:
                continue
            dimension_metrics = scope_metrics.get(dimension) or {}
            bucket = dimension_metrics.get(label)
            if not isinstance(bucket, dict):
                continue
            feedback = self._extract_team_routing_feedback_from_mapping(
                dict(bucket),
                scope_dimension=dimension,
                scope_label=label,
            )
            if feedback is not None:
                return feedback
        return None

    def get_topology_routing_context(
        self,
        *,
        min_coverage: float = 0.2,
        min_reward: float = 0.45,
        min_observations: int = 2,
        query: Optional[str] = None,
        scope_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return soft topology-routing hints from validated feedback and experiment memory."""
        policy = self.get_structured_routing_policy(
            query=query,
            scope_context=scope_context,
            min_coverage=min_coverage,
            min_reward=min_reward,
            min_observations=min_observations,
        )
        return policy.selector_context()

    def _resolve_routing_scope_context(
        self,
        scope_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge runtime routing scope hints with the persisted feedback scope defaults."""
        resolved_context = dict(scope_context or {})
        if "provider" not in resolved_context and resolved_context.get("provider_hint"):
            resolved_context["provider"] = resolved_context.get("provider_hint")
        if self._evaluation_feedback_scope.task_type and "task_type" not in resolved_context:
            resolved_context["task_type"] = self._evaluation_feedback_scope.task_type
        if self._evaluation_feedback_scope.provider and "provider" not in resolved_context:
            resolved_context["provider"] = self._evaluation_feedback_scope.provider
        if self._evaluation_feedback_scope.model and "model" not in resolved_context:
            resolved_context["model"] = self._evaluation_feedback_scope.model
        return resolved_context

    @staticmethod
    def _coerce_optional_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_optional_text(value: Any) -> Optional[str]:
        text = str(value).strip() if value is not None else ""
        return text or None

    def _resolve_experiment_scope_filters(
        self,
        scope_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Optional[str]]:
        """Resolve query filters for experiment-memory retrieval."""
        resolved_context = self._resolve_routing_scope_context(scope_context)
        benchmark = self._coerce_optional_text(
            resolved_context.get("benchmark") or self._runtime_feedback_metadata.get("benchmark")
        )
        provider = self._coerce_optional_text(
            resolved_context.get("provider") or resolved_context.get("provider_hint")
        )
        model = self._coerce_optional_text(
            resolved_context.get("model") or resolved_context.get("model_name")
        )
        prompt_candidate_hash = self._coerce_optional_text(
            resolved_context.get("prompt_candidate_hash")
        )
        section_name = self._coerce_optional_text(
            resolved_context.get("section_name") or resolved_context.get("prompt_section_name")
        )
        return {
            "benchmark": benchmark,
            "provider": provider,
            "model": model,
            "prompt_candidate_hash": prompt_candidate_hash,
            "section_name": section_name,
        }

    def get_experiment_routing_context(
        self,
        *,
        query: Optional[str] = None,
        scope_context: Optional[Dict[str, Any]] = None,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """Return soft routing hints distilled from reusable experiment-memory records."""
        store = self._experiment_memory_store
        if store is None or len(store) == 0 or limit <= 0:
            return {}

        scope_filters = self._resolve_experiment_scope_filters(scope_context)
        records = store.search(
            query or "",
            benchmark=scope_filters["benchmark"],
            provider=scope_filters["provider"],
            model=scope_filters["model"],
            prompt_candidate_hash=scope_filters["prompt_candidate_hash"],
            section_name=scope_filters["section_name"],
            limit=limit,
        )
        if not records and query:
            records = store.search(
                "",
                benchmark=scope_filters["benchmark"],
                provider=scope_filters["provider"],
                model=scope_filters["model"],
                prompt_candidate_hash=scope_filters["prompt_candidate_hash"],
                section_name=scope_filters["section_name"],
                limit=limit,
            )
        if not records and query:
            records = store.search(
                query,
                provider=scope_filters["provider"],
                model=scope_filters["model"],
                limit=limit,
            )
        if not records:
            return {}

        bias_total = 0.0
        bias_weight = 0.0
        planning_bias_total = 0.0
        planning_bias_weight = 0.0
        constraint_tags: set[str] = set()
        next_candidate_hints: list[str] = []
        record_ids: list[str] = []
        for record in records:
            record_ids.append(record.record_id)
            summary_delta = self._coerce_optional_float(
                record.summary_metrics.get("topology_learned_override_optimization_reward_delta")
            )
            if summary_delta is not None:
                normalized_delta = max(-1.0, min(1.0, summary_delta / 0.25))
                weight = min(1.0, max(0.25, abs(normalized_delta)))
                bias_total += normalized_delta * weight
                bias_weight += weight

            for insight in record.insights:
                summary_text = insight.summary.lower()
                confidence = max(0.0, min(float(insight.confidence), 1.0))
                if insight.kind == "failed_hypothesis":
                    if "learned close override" in summary_text and "heuristic" in summary_text:
                        bias_total -= max(0.25, confidence)
                        bias_weight += max(0.25, confidence)
                    if (
                        "forced llm planning" in summary_text
                        and "heuristic fast-path" in summary_text
                    ):
                        planning_bias_total -= max(0.25, confidence)
                        planning_bias_weight += max(0.25, confidence)
                elif insight.kind == "successful_transformation":
                    if "learned close override" in summary_text and "heuristic" in summary_text:
                        bias_total += max(0.25, confidence)
                        bias_weight += max(0.25, confidence)
                    if (
                        "forced llm planning" in summary_text
                        and "heuristic fast-path" in summary_text
                    ):
                        planning_bias_total += max(0.25, confidence)
                        planning_bias_weight += max(0.25, confidence)
                elif insight.kind == "environment_constraint":
                    gate_failure = self._coerce_optional_text(
                        insight.evidence.get("gate_failure")
                        or insight.evidence.get("failure_category")
                    )
                    if gate_failure:
                        constraint_tags.add(gate_failure)
                elif insight.kind == "next_candidate":
                    if insight.summary and insight.summary not in next_candidate_hints:
                        next_candidate_hints.append(insight.summary)

        if bias_weight <= 0.0:
            bias = 0.0
        else:
            bias = round(max(-1.0, min(1.0, bias_total / bias_weight)), 4)
        if planning_bias_weight <= 0.0:
            planning_bias = 0.0
        else:
            planning_bias = round(
                max(-1.0, min(1.0, planning_bias_total / planning_bias_weight)),
                4,
            )

        preferred_selection_policy = None
        if bias <= -0.1:
            preferred_selection_policy = "heuristic"
        elif bias >= 0.1:
            preferred_selection_policy = "learned_close_override"
        preferred_planning_policy = None
        if planning_bias <= -0.1:
            preferred_planning_policy = "heuristic_fast_path"
        elif planning_bias >= 0.1:
            preferred_planning_policy = "experiment_forced_slow_path"

        return {
            "experiment_memory_match_count": len(records),
            "experiment_memory_support": round(min(1.0, len(records) / max(1.0, float(limit))), 4),
            "experiment_memory_selection_policy_bias": bias,
            "experiment_memory_preferred_selection_policy": preferred_selection_policy,
            "experiment_memory_planning_policy_bias": planning_bias,
            "experiment_memory_preferred_planning_policy": preferred_planning_policy,
            "experiment_memory_constraint_tags": sorted(constraint_tags),
            "experiment_memory_next_candidate_hints": next_candidate_hints[:2],
            "experiment_memory_record_ids": record_ids[:3],
        }

    def get_planning_routing_context(
        self,
        *,
        query: Optional[str] = None,
        scope_context: Optional[Dict[str, Any]] = None,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """Return planning-gate hints distilled from experiment memory."""
        experiment_hints = self.get_experiment_routing_context(
            query=query,
            scope_context=scope_context,
            limit=limit,
        )
        if not experiment_hints:
            return {}

        constraint_tags = [
            str(tag).strip()
            for tag in list(experiment_hints.get("experiment_memory_constraint_tags") or [])
            if str(tag).strip()
        ]
        next_candidate_hints = [
            str(item).strip()
            for item in list(experiment_hints.get("experiment_memory_next_candidate_hints") or [])
            if str(item).strip()
        ]
        planning_hints: Dict[str, Any] = {
            "planning_match_count": int(experiment_hints.get("experiment_memory_match_count") or 0),
            "planning_experiment_support": float(
                experiment_hints.get("experiment_memory_support") or 0.0
            ),
            "planning_constraint_tags": constraint_tags,
            "planning_next_candidate_hints": next_candidate_hints[:2],
            "planning_policy_bias": float(
                experiment_hints.get("experiment_memory_planning_policy_bias") or 0.0
            ),
            "planning_preferred_policy": self._coerce_optional_text(
                experiment_hints.get("experiment_memory_preferred_planning_policy")
            ),
        }
        if constraint_tags:
            planning_hints["planning_force_llm"] = True
            planning_hints["planning_force_reason"] = (
                f"experiment_constraints: {', '.join(constraint_tags[:2])}"
            )
        elif (
            planning_hints["planning_preferred_policy"] == "experiment_forced_slow_path"
            and planning_hints["planning_policy_bias"] >= 0.15
        ):
            planning_hints["planning_force_llm"] = True
            planning_hints["planning_force_reason"] = (
                "experiment_policy_bias: experiment_forced_slow_path"
            )
        elif (
            planning_hints["planning_preferred_policy"] == "heuristic_fast_path"
            and planning_hints["planning_policy_bias"] <= -0.15
        ):
            absolute_bias = min(1.0, abs(float(planning_hints["planning_policy_bias"])))
            planning_hints["planning_prefer_fast_path"] = True
            planning_hints["planning_prefer_reason"] = "experiment_policy_bias: heuristic_fast_path"
            planning_hints["planning_fast_path_tool_budget_limit"] = 4 + int(absolute_bias >= 0.6)
            planning_hints["planning_fast_path_query_length_limit"] = int(
                round(60 + (20 * absolute_bias))
            )
            planning_hints["planning_fast_path_complexity_threshold"] = round(
                min(0.45, 0.3 + (0.12 * absolute_bias)),
                4,
            )
        return planning_hints

    def _build_persistable_session_topology_feedback(
        self,
    ) -> Optional[RuntimeEvaluationFeedback]:
        """Build the scoped live-topology feedback payload for persistence."""
        feedback = self._session_topology_routing_feedback
        if feedback is None or feedback.observation_count <= 0:
            return None

        return RuntimeEvaluationFeedback(
            metadata={
                "topology_feedback_coverage": round(feedback.coverage, 4),
                "avg_topology_reward": round(feedback.avg_reward, 4),
                "avg_topology_confidence": round(feedback.avg_confidence, 4),
                "topology_actions": dict(feedback.action_distribution),
                "topology_final_actions": dict(feedback.action_distribution),
                "topology_kinds": dict(feedback.topology_distribution),
                "topology_final_kinds": dict(feedback.topology_distribution),
                "topology_execution_modes": dict(feedback.execution_mode_distribution),
                "topology_providers": dict(feedback.provider_distribution),
                "topology_formations": dict(feedback.formation_distribution),
                "topology_selection_policies": dict(feedback.selection_policy_distribution),
                "topology_selection_policy_reward_totals": dict(
                    feedback.selection_policy_reward_totals
                ),
                "topology_selection_policy_optimization_counts": dict(
                    feedback.selection_policy_optimization_counts
                ),
                "topology_selection_policy_optimization_reward_totals": dict(
                    feedback.selection_policy_optimization_reward_totals
                ),
                "topology_selection_policy_feasible_counts": dict(
                    feedback.selection_policy_feasible_counts
                ),
                "avg_topology_reward_by_selection_policy": dict(
                    feedback.avg_reward_by_selection_policy
                ),
                "topology_learned_override_reward_delta": feedback.learned_override_reward_delta,
                "avg_topology_optimization_reward_by_selection_policy": dict(
                    feedback.avg_optimization_reward_by_selection_policy
                ),
                "topology_selection_policy_feasibility_rates": dict(
                    feedback.feasibility_rate_by_selection_policy
                ),
                "topology_learned_override_optimization_reward_delta": (
                    feedback.learned_override_optimization_reward_delta
                ),
                "topology_learned_override_feasibility_delta": (
                    feedback.learned_override_feasibility_delta
                ),
                "topology_selection_policy_scope_metrics": dict(
                    feedback.selection_policy_scope_metrics
                ),
                "topology_observation_count": feedback.observation_count,
                "topology_action_agreement": round(feedback.action_agreement, 4),
                "topology_kind_agreement": round(feedback.topology_agreement, 4),
                "topology_provider_agreement": round(feedback.provider_agreement, 4),
                "topology_formation_agreement": round(feedback.formation_agreement, 4),
                "topology_conflict_score": round(feedback.conflict_score, 4),
                "task_count": feedback.observation_count,
            }
        )

    def _persist_session_topology_feedback(self) -> Optional[Any]:
        """Persist scoped live-topology feedback for future runtime sessions."""
        if self._evaluation_feedback_scope.is_empty():
            return None
        feedback = self._build_persistable_session_topology_feedback()
        if feedback is None:
            return None
        try:
            return save_session_topology_runtime_feedback(
                feedback,
                base_path=self._evaluation_feedback_path,
                scope=self._evaluation_feedback_scope,
            )
        except Exception as exc:
            logger.debug("Runtime intelligence could not persist topology feedback: %s", exc)
            return None

    def record_topology_outcome(self, payload: Any) -> Optional[TopologyRoutingFeedback]:
        """Record one live topology outcome and refresh session-local routing hints."""
        summary = summarize_topology_feedback(payload)
        if summary is None:
            return self.get_topology_routing_feedback()

        self._session_topology_feedback_records.append({"topology_summary": dict(summary)})
        aggregate = aggregate_topology_feedback(
            self._session_topology_feedback_records,
            total_tasks=len(self._session_topology_feedback_records),
        )
        self._session_topology_routing_feedback = self._extract_topology_routing_feedback(aggregate)
        self._persist_session_topology_feedback()
        return self.get_topology_routing_feedback()

    def has_decision_service(self) -> bool:
        """Return whether the runtime has an attached decision service."""
        return self._decision_service is not None

    def clear_session_cache(self) -> None:
        """Clear cached prompt-optimization state when the session changes."""
        if self._optimization_injector is not None:
            self._optimization_injector.clear_session_cache()
        self._session_topology_feedback_records.clear()
        self._session_topology_routing_feedback = None

    def reset_decision_budget(self) -> None:
        """Reset the decision-service per-turn budget when available."""
        if self._decision_service is not None and hasattr(self._decision_service, "reset_budget"):
            self._decision_service.reset_budget()

    def decide_sync(
        self,
        decision_type: Any,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> Optional[Any]:
        """Delegate a synchronous decision to the underlying decision service."""
        if self._decision_service is None or not hasattr(self._decision_service, "decide_sync"):
            return None
        kwargs = {"heuristic_confidence": heuristic_confidence}
        if heuristic_result is not None:
            kwargs["heuristic_result"] = heuristic_result
        return self._decision_service.decide_sync(
            decision_type,
            context,
            **kwargs,
        )

    async def analyze_turn(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> RuntimeIntelligenceSnapshot:
        """Analyze a turn and return a typed snapshot."""
        perception = None
        task_analysis = None

        if self._perception_integration is not None:
            perception = await self._perception_integration.perceive(
                query,
                context,
                conversation_history,
            )
            task_analysis = getattr(perception, "task_analysis", None)

        if task_analysis is None and self._task_analyzer is not None:
            analysis_context = dict(context or {})
            if conversation_history:
                analysis_context["history"] = conversation_history
            task_analysis = self._task_analyzer.analyze(query, context=analysis_context)

        metadata = {
            "has_context": bool(context),
            "history_length": len(conversation_history or []),
        }
        topology_feedback = self.get_topology_routing_feedback()
        if topology_feedback is not None:
            metadata["topology_feedback"] = topology_feedback.to_metadata()
        team_feedback = self.get_team_routing_feedback()
        if team_feedback is not None:
            metadata["team_feedback"] = team_feedback.to_metadata()
        degradation_feedback = self.get_degradation_routing_feedback()
        if degradation_feedback is not None:
            metadata["degradation_feedback"] = degradation_feedback.to_metadata()
        routing_scope_context = dict(context or {})
        if task_analysis is not None and "task_type" not in routing_scope_context:
            routing_scope_context["task_type"] = getattr(task_analysis, "task_type", None)
        routing_policy = self.get_structured_routing_policy(
            query=query,
            scope_context=routing_scope_context,
        )
        routing_hints = routing_policy.selector_context()
        if not routing_policy.is_empty():
            metadata["structured_routing_policy"] = routing_policy.to_dict()
        if routing_hints:
            metadata["topology_routing_hints"] = routing_hints
            experiment_memory_hints = {
                key: value
                for key, value in routing_hints.items()
                if str(key).startswith("experiment_memory_")
            }
            if experiment_memory_hints:
                metadata["experiment_memory_hints"] = experiment_memory_hints
        planning_hints = routing_policy.planning_context()
        if planning_hints:
            metadata["planning_routing_hints"] = planning_hints

        return RuntimeIntelligenceSnapshot(
            query=query,
            perception=perception,
            task_analysis=task_analysis,
            decision_service_available=self.has_decision_service(),
            metadata=metadata,
        )

    def classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task intent using the underlying task analyzer when available."""
        if self._task_analyzer and hasattr(self._task_analyzer, "classify_keywords"):
            try:
                return self._task_analyzer.classify_keywords(user_message)
            except Exception as exc:
                logger.debug("Runtime intelligence keyword classification failed: %s", exc)
        return {"task_type": "default", "confidence": 0.0}

    def classify_task_with_context(
        self,
        user_message: str,
        history: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Classify task intent with history context when supported."""
        if self._task_analyzer and hasattr(self._task_analyzer, "classify_with_context"):
            try:
                return self._task_analyzer.classify_with_context(user_message, history or [])
            except Exception as exc:
                logger.debug(
                    "Runtime intelligence contextual task classification failed: %s",
                    exc,
                )
        return {"task_type": "default", "confidence": 0.0}

    @staticmethod
    def get_clarification_decision(
        perception: Optional[Any],
        *,
        default_prompt: Optional[str] = (
            "Please clarify the target file, component, or bug before I continue."
        ),
        policy: Optional[RuntimeEvaluationPolicy] = None,
    ) -> ClarificationDecision:
        """Normalize clarification policy into one typed runtime decision."""
        selected_policy = (policy or RuntimeEvaluationPolicy()).with_overrides(
            default_clarification_prompt=default_prompt
        )
        return selected_policy.get_clarification_decision(perception)

    @staticmethod
    def evaluate_confidence_progress(
        confidence: float,
        state: Dict[str, Any],
        *,
        retry_limit: Optional[int] = 2,
        high_confidence_threshold: Optional[float] = 0.8,
        medium_confidence_threshold: Optional[float] = 0.5,
        policy: Optional[RuntimeEvaluationPolicy] = None,
    ) -> Any:
        """Apply the canonical confidence-band policy for live-loop evaluation."""
        selected_policy = (policy or RuntimeEvaluationPolicy()).with_overrides(
            high_confidence_threshold=high_confidence_threshold,
            medium_confidence_threshold=medium_confidence_threshold,
            low_confidence_retry_limit=retry_limit,
        )
        return selected_policy.evaluate_confidence_progress(
            confidence,
            state,
            retry_limit=retry_limit,
        )

    @staticmethod
    def get_confidence_evaluation(
        confidence: float,
        *,
        high_confidence_threshold: Optional[float] = 0.8,
        medium_confidence_threshold: Optional[float] = 0.5,
        policy: Optional[RuntimeEvaluationPolicy] = None,
    ) -> Any:
        """Emit the canonical confidence-band evaluation without mutating retry state."""
        selected_policy = (policy or RuntimeEvaluationPolicy()).with_overrides(
            high_confidence_threshold=high_confidence_threshold,
            medium_confidence_threshold=medium_confidence_threshold,
        )
        return selected_policy.get_confidence_evaluation(confidence)

    @staticmethod
    def apply_low_confidence_retry_budget(
        evaluation: Any,
        state: Dict[str, Any],
        *,
        retry_limit: Optional[int] = 2,
        low_confidence_threshold: Optional[float] = 0.5,
        policy: Optional[RuntimeEvaluationPolicy] = None,
    ) -> Any:
        """Apply the canonical retry-budget policy to low-confidence retry results."""
        selected_policy = (policy or RuntimeEvaluationPolicy()).with_overrides(
            low_confidence_retry_limit=retry_limit,
            medium_confidence_threshold=low_confidence_threshold,
        )
        return selected_policy.apply_retry_budget(
            evaluation,
            state,
            retry_limit=retry_limit,
            low_confidence_threshold=low_confidence_threshold,
        )

    def get_prompt_optimization_bundle(
        self,
        user_message: str,
        turn_context: Any,
    ) -> PromptOptimizationBundle:
        """Collect prompt optimizations for the current turn."""
        provider_name = getattr(turn_context, "provider_name", "")
        model_name = getattr(turn_context, "model", "")
        task_type = getattr(turn_context, "task_type", "default")
        evolved_sections: List[str] = []
        identities: List[PromptOptimizationIdentity] = []
        few_shots = None
        failure_hint = None
        if self._optimization_injector is not None:
            payloads: Optional[Any] = None
            if hasattr(self._optimization_injector, "get_evolved_section_payloads"):
                payloads = self._optimization_injector.get_evolved_section_payloads(
                    provider=provider_name,
                    model=model_name,
                    task_type=task_type,
                )
            used_payload_sections = False
            if isinstance(payloads, (list, tuple)):
                for payload in payloads:
                    if not isinstance(payload, dict):
                        continue
                    used_payload_sections = True
                    text = str(payload.get("text", "")).strip()
                    if text:
                        evolved_sections.append(text)
                    identity = self._build_prompt_identity(payload)
                    if identity is not None:
                        identities.append(identity)
            if not used_payload_sections:
                evolved_sections = self._optimization_injector.get_evolved_sections(
                    provider=provider_name,
                    model=model_name,
                    task_type=task_type,
                )

            few_shot_payload: Optional[Any] = None
            if hasattr(self._optimization_injector, "get_few_shot_payload"):
                few_shot_payload = self._optimization_injector.get_few_shot_payload(
                    user_message,
                    provider=provider_name,
                    model=model_name,
                    task_type=task_type,
                )
            used_payload_few_shot = False
            if isinstance(few_shot_payload, dict):
                if few_shot_payload:
                    used_payload_few_shot = True
                    few_shots = str(few_shot_payload.get("text", "")).strip() or None
                    identity = self._build_prompt_identity(few_shot_payload)
                    if identity is not None:
                        identities.append(identity)
            if not used_payload_few_shot:
                few_shots = self._optimization_injector.get_few_shots(
                    user_message,
                    provider=provider_name,
                    model=model_name,
                    task_type=task_type,
                )

            if getattr(turn_context, "last_turn_failed", False):
                failure_hint = self._optimization_injector.get_failure_hint(
                    getattr(turn_context, "last_failure_category", None),
                    getattr(turn_context, "last_failure_error", None),
                )

        experiment_memory_hints = self.get_experiment_routing_context(
            query=user_message,
            scope_context={
                "provider": provider_name,
                "model": model_name,
                "task_type": task_type,
                "prompt_candidate_hash": getattr(turn_context, "prompt_candidate_hash", None),
                "section_name": getattr(turn_context, "prompt_section_name", None)
                or getattr(turn_context, "section_name", None),
            },
        )
        experiment_guidance = self._build_experiment_prompt_guidance(experiment_memory_hints)

        # Cache the identities actually served this turn so the agentic loop can
        # attribute the turn's outcome to them at turn-evaluation time. Only
        # identities carrying a candidate hash are rewardable.
        self._last_served_prompt_identities = [
            identity for identity in identities if identity.prompt_candidate_hash
        ]

        return PromptOptimizationBundle(
            evolved_sections=list(evolved_sections or []),
            few_shots=few_shots,
            failure_hint=failure_hint,
            identities=identities,
            experiment_guidance=experiment_guidance,
            experiment_memory_hints=experiment_memory_hints,
        )

    @staticmethod
    def _build_experiment_prompt_guidance(hints: Optional[Dict[str, Any]]) -> List[str]:
        """Convert experiment-memory routing hints into prompt-ready planning guidance."""
        if not isinstance(hints, dict) or not hints:
            return []

        guidance: List[str] = []
        constraint_tags = [
            str(tag).strip()
            for tag in list(hints.get("experiment_memory_constraint_tags") or [])
            if str(tag).strip()
        ]
        if constraint_tags:
            constraint_label = "constraint" if len(constraint_tags) == 1 else "constraints"
            guidance.append(
                "Experiment "
                f"{constraint_label} from similar runs: satisfy "
                f"{', '.join(constraint_tags[:2])} before broadening the plan."
            )

        next_candidate_hints = [
            str(item).strip()
            for item in list(hints.get("experiment_memory_next_candidate_hints") or [])
            if str(item).strip()
        ]
        for hint in next_candidate_hints[:2]:
            guidance.append(f"Experiment-guided next candidate: {hint}")
        return guidance

    @staticmethod
    def _build_prompt_identity(payload: Any) -> Optional[PromptOptimizationIdentity]:
        """Extract canonical prompt identity from an optimization payload."""
        if not isinstance(payload, dict):
            return None

        def _optional_text(value: Any) -> Optional[str]:
            text = str(value).strip() if value is not None else ""
            return text or None

        section_name = _optional_text(
            payload.get("section_name") or payload.get("prompt_section_name")
        )
        provider = _optional_text(payload.get("provider"))
        prompt_candidate_hash = _optional_text(payload.get("prompt_candidate_hash"))
        strategy_name = _optional_text(payload.get("strategy_name"))
        source = _optional_text(payload.get("source"))
        if not any([section_name, provider, prompt_candidate_hash, strategy_name, source]):
            return None
        return PromptOptimizationIdentity(
            provider=provider,
            prompt_candidate_hash=prompt_candidate_hash,
            section_name=section_name,
            prompt_section_name=section_name,
            strategy_name=strategy_name,
            source=source,
        )
