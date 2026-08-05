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

"""Heuristic topology selector for runtime structure decisions.

This module implements the first executable version of topology-aware routing.
The selector intentionally uses transparent heuristics so the runtime can emit
inspectable rationale and telemetry before moving to learned policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from victor.agent.topology_contract import (
    TopologyAction,
    TopologyDecision,
    TopologyDecisionInput,
    TopologyGroundingRequirements,
    TopologyKind,
)


@dataclass
class TopologySelectorConfig:
    """Configuration for heuristic topology selection."""

    fallback_score_gap: float = 0.12
    low_confidence_threshold: float = 0.65
    learned_override_score_gap: float = 0.1
    learned_override_min_support: float = 0.4
    learned_override_min_agreement: float = 0.7
    learned_override_max_conflict: float = 0.35
    learned_override_margin: float = 0.01
    learned_override_min_policy_count: int = 2
    learned_override_disable_reward_delta: float = -0.2
    learned_override_disable_feasibility_delta: float = -0.2
    learned_override_reward_delta_cap: float = 0.25
    learned_override_score_gap_tuning_gain: float = 0.3
    learned_override_support_tuning_gain: float = 0.2
    learned_override_agreement_tuning_gain: float = 0.2
    learned_override_conflict_tuning_gain: float = 0.2
    experiment_memory_disable_bias_threshold: float = -0.75
    experiment_memory_bias_tuning_gain: float = 0.25
    min_parallel_workers: int = 2
    max_parallel_workers: int = 4


@dataclass(frozen=True)
class LearnedOverrideThresholds:
    """Effective learned-override thresholds after policy-aware tuning."""

    score_gap: float
    min_support: float
    min_agreement: float
    max_conflict: float
    reward_delta: Optional[float] = None
    optimization_reward_delta: Optional[float] = None
    feasibility_delta: Optional[float] = None
    reward_evidence: int = 0
    experiment_memory_bias: Optional[float] = None
    experiment_memory_match_count: int = 0
    experiment_memory_preferred_policy: Optional[str] = None
    profile: str = "static"
    disabled: bool = False


class TopologySelector:
    """Choose a runtime topology from request-time execution hints."""

    LOCAL_PROVIDER_HINTS = {
        "local",
        "ollama",
        "vllm",
        "lmstudio",
        "llamacpp",
        "llama.cpp",
    }

    def __init__(self, config: Optional[TopologySelectorConfig] = None):
        self.config = config or TopologySelectorConfig()

    def select(self, decision_input: TopologyDecisionInput) -> TopologyDecision:
        """Return a heuristic topology decision for the given runtime hints."""
        heuristic_scores = {
            TopologyAction.DIRECT_RESPONSE: self._score_direct_response(decision_input),
            TopologyAction.SINGLE_AGENT: self._score_single_agent(decision_input),
            TopologyAction.PARALLEL_EXPLORATION: self._score_parallel_exploration(decision_input),
            TopologyAction.TEAM_PLAN: self._score_team_plan(decision_input),
            TopologyAction.ESCALATE_MODEL: self._score_escalate_model(decision_input),
            TopologyAction.SAFE_STOP: self._score_safe_stop(decision_input),
        }
        selected_action, score_breakdown, policy_metadata = self._apply_learned_policy_overlay(
            decision_input,
            heuristic_scores,
        )
        ordered = sorted(score_breakdown.items(), key=lambda item: item[1], reverse=True)
        best_action, best_score = ordered[0]
        second_action, second_score = ordered[1]
        final_action = selected_action or best_action

        confidence = self._normalize_confidence(score_breakdown.get(final_action, best_score))
        fallback_action = self._resolve_fallback_action(
            final_action=final_action,
            score_breakdown=score_breakdown,
            confidence=confidence,
            heuristic_best_action=policy_metadata.get("heuristic_action"),
            heuristic_best_score=policy_metadata.get("heuristic_score"),
        )

        provider = self._select_provider_hint(decision_input, final_action)
        formation = self._select_formation_hint(decision_input, final_action)
        grounding = self._build_grounding(
            decision_input=decision_input,
            action=final_action,
            provider=provider,
            formation=formation,
        )
        grounding.metadata.update(self._grounding_policy_metadata(policy_metadata))

        return TopologyDecision(
            action=final_action,
            topology=self._topology_kind_for_action(final_action),
            confidence=confidence,
            rationale=self._build_rationale(
                final_action,
                decision_input,
                policy_metadata=policy_metadata,
            ),
            score_breakdown={action.value: round(score, 4) for action, score in ordered},
            grounding_requirements=grounding,
            fallback_action=fallback_action,
            telemetry_tags=self._build_telemetry_tags(decision_input, policy_metadata),
            provider=provider,
            formation=formation,
        )

    def _score_direct_response(self, decision_input: TopologyDecisionInput) -> float:
        score = 0.0
        if self._is_low(decision_input.task_complexity):
            score += 0.25
        if self._is_low(decision_input.expected_depth):
            score += 0.2
        if self._is_low(decision_input.expected_breadth):
            score += 0.1
        if decision_input.tool_budget <= 1:
            score += 0.15
        if self._is_high(decision_input.latency_sensitivity):
            score += 0.1
        if decision_input.confidence_hint >= 0.75:
            score += 0.1
        if decision_input.prior_failures > 0:
            score -= 0.2
        if self._is_low(decision_input.observability_level):
            score -= 0.1
        return score

    def _score_single_agent(self, decision_input: TopologyDecisionInput) -> float:
        score = 0.2
        if self._is_low(decision_input.task_complexity):
            score += 0.25
        elif self._is_medium(decision_input.task_complexity):
            score += 0.15
        else:
            score -= 0.05

        if self._is_low(decision_input.expected_depth):
            score += 0.15
        elif self._is_medium(decision_input.expected_depth):
            score += 0.05
        else:
            score -= 0.15

        if self._is_low(decision_input.expected_breadth):
            score += 0.1
        elif self._is_high(decision_input.expected_breadth):
            score -= 0.15

        if self._is_high(decision_input.observability_level):
            score += 0.15
        elif self._is_medium(decision_input.observability_level):
            score += 0.05
        else:
            score -= 0.1

        if decision_input.prior_failures == 0:
            score += 0.05
        else:
            score -= min(0.08 * decision_input.prior_failures, 0.24)

        if self._is_high(decision_input.latency_sensitivity):
            score += 0.05
        if self._is_high(decision_input.privacy_sensitivity):
            score += 0.05
        if decision_input.confidence_hint >= 0.6:
            score += 0.05
        elif decision_input.confidence_hint < 0.4:
            score -= 0.05
        return score

    def _score_parallel_exploration(self, decision_input: TopologyDecisionInput) -> float:
        score = 0.1
        if self._is_high(decision_input.expected_breadth):
            score += 0.35
        elif self._is_medium(decision_input.expected_breadth):
            score += 0.1
        else:
            score -= 0.05

        if self._is_high(decision_input.task_complexity):
            score += 0.1
        elif self._is_medium(decision_input.task_complexity):
            score += 0.05

        if self._is_low(decision_input.observability_level):
            score += 0.15
        elif self._is_medium(decision_input.observability_level):
            score += 0.05

        if decision_input.prior_failures > 0:
            score += 0.1

        if self._is_high(decision_input.latency_sensitivity):
            score -= 0.2
        elif self._is_medium(decision_input.latency_sensitivity):
            score -= 0.05

        if self._is_high(decision_input.token_cost_pressure):
            score -= 0.1
        if decision_input.tool_budget < 4:
            score -= 0.15
        return score

    def _score_team_plan(self, decision_input: TopologyDecisionInput) -> float:
        score = 0.05
        if self._is_high(decision_input.expected_depth):
            score += 0.35
        elif self._is_medium(decision_input.expected_depth):
            score += 0.1
        else:
            score -= 0.05

        if self._is_high(decision_input.task_complexity):
            score += 0.25
        elif self._is_medium(decision_input.task_complexity):
            score += 0.1

        if decision_input.available_team_formations:
            score += 0.1
        else:
            score -= 0.2

        if self._is_low(decision_input.observability_level):
            score += 0.1

        if self._is_high(decision_input.latency_sensitivity):
            score -= 0.2
        elif self._is_medium(decision_input.latency_sensitivity):
            score -= 0.05

        if self._is_high(decision_input.token_cost_pressure):
            score -= 0.15
        if decision_input.prior_failures > 1:
            score += 0.05
        return score

    def _score_escalate_model(self, decision_input: TopologyDecisionInput) -> float:
        score = 0.05
        if decision_input.confidence_hint < 0.4:
            score += 0.25
        elif decision_input.confidence_hint < 0.6:
            score += 0.1

        if self._is_low(decision_input.observability_level):
            score += 0.2
        elif self._is_medium(decision_input.observability_level):
            score += 0.05

        if self._is_high(decision_input.task_complexity):
            score += 0.15
        elif self._is_medium(decision_input.task_complexity):
            score += 0.05

        if decision_input.prior_failures > 0:
            score += 0.15

        if self._is_high(decision_input.token_cost_pressure):
            score -= 0.1
        if self._is_high(decision_input.privacy_sensitivity) and self._is_remote_only(
            decision_input.provider_candidates
        ):
            score -= 0.4
        if not decision_input.provider_candidates:
            score -= 0.1
        return score

    def _score_safe_stop(self, decision_input: TopologyDecisionInput) -> float:
        score = 0.0
        remote_only = self._is_remote_only(decision_input.provider_candidates)
        if self._is_high(decision_input.privacy_sensitivity) and remote_only:
            score += 0.5
        if self._is_high(decision_input.privacy_sensitivity) and self._is_low(
            decision_input.observability_level
        ):
            score += 0.2
        if decision_input.confidence_hint < 0.3 and decision_input.prior_failures > 1:
            score += 0.15
        if not decision_input.provider_candidates:
            score += 0.1
        if (
            self._is_high(decision_input.task_complexity)
            and self._is_low(decision_input.observability_level)
            and self._is_high(decision_input.privacy_sensitivity)
            and remote_only
        ):
            score += 0.1
        return score

    def _select_fallback_action(
        self,
        *,
        confidence: float,
        best_action: TopologyAction,
        best_score: float,
        second_action: TopologyAction,
        second_score: float,
    ) -> Optional[TopologyAction]:
        gap = best_score - second_score
        if (
            confidence < self.config.low_confidence_threshold
            or gap <= self.config.fallback_score_gap
        ):
            if second_action is not best_action:
                return second_action
        return None

    def _select_provider_hint(
        self,
        decision_input: TopologyDecisionInput,
        action: TopologyAction,
    ) -> Optional[str]:
        candidates = decision_input.provider_candidates
        if not candidates or action == TopologyAction.SAFE_STOP:
            return None

        if action == TopologyAction.ESCALATE_MODEL:
            remote_candidates = [
                provider for provider in candidates if not self._is_local_provider(provider)
            ]
            learned_provider = self._preferred_provider_from_feedback(
                decision_input,
                remote_candidates,
            )
            if learned_provider is not None:
                return learned_provider
            if remote_candidates:
                return remote_candidates[0]
            return candidates[0]

        if self._is_high(decision_input.privacy_sensitivity):
            local_candidates = [
                provider for provider in candidates if self._is_local_provider(provider)
            ]
            learned_provider = self._preferred_provider_from_feedback(
                decision_input,
                local_candidates,
            )
            if learned_provider is not None:
                return learned_provider
            if local_candidates:
                return local_candidates[0]

        learned_provider = self._preferred_provider_from_feedback(decision_input, candidates)
        if learned_provider is not None:
            return learned_provider

        return candidates[0]

    def _select_formation_hint(
        self,
        decision_input: TopologyDecisionInput,
        action: TopologyAction,
    ) -> Optional[str]:
        if action != TopologyAction.TEAM_PLAN:
            return None

        formations = decision_input.available_team_formations
        if not formations:
            return "adaptive"
        learned_formation = self._preferred_formation_from_feedback(decision_input, formations)
        if learned_formation is not None:
            return learned_formation
        if self._is_high(decision_input.expected_depth) and "hierarchical" in formations:
            return "hierarchical"
        if "adaptive" in formations:
            return "adaptive"
        return formations[0]

    def _build_grounding(
        self,
        *,
        decision_input: TopologyDecisionInput,
        action: TopologyAction,
        provider: Optional[str],
        formation: Optional[str],
    ) -> TopologyGroundingRequirements:
        if action == TopologyAction.DIRECT_RESPONSE:
            return TopologyGroundingRequirements(
                provider=provider,
                tool_budget=0,
                iteration_budget=1,
                metadata={"topology": "direct"},
            )

        if action == TopologyAction.SINGLE_AGENT:
            return TopologyGroundingRequirements(
                provider=provider,
                tool_budget=decision_input.tool_budget,
                iteration_budget=decision_input.iteration_budget,
                metadata={"topology": "single_agent"},
            )

        if action == TopologyAction.PARALLEL_EXPLORATION:
            return TopologyGroundingRequirements(
                provider=provider,
                max_workers=self._parallel_worker_count(decision_input),
                tool_budget=decision_input.tool_budget,
                iteration_budget=min(decision_input.iteration_budget, 2),
                metadata={"topology": "parallel_exploration"},
            )

        if action == TopologyAction.TEAM_PLAN:
            metadata = {"topology": "team_plan"}
            metadata.update(self._team_execution_metadata(decision_input))
            return TopologyGroundingRequirements(
                provider=provider,
                formation=formation,
                max_workers=self._team_worker_count(decision_input),
                tool_budget=decision_input.tool_budget,
                iteration_budget=decision_input.iteration_budget,
                metadata=metadata,
            )

        if action == TopologyAction.ESCALATE_MODEL:
            return TopologyGroundingRequirements(
                provider=provider,
                tool_budget=max(decision_input.tool_budget, 4),
                iteration_budget=decision_input.iteration_budget,
                escalation_target=provider,
                metadata={"topology": "escalated_single_agent"},
            )

        return TopologyGroundingRequirements(
            safety_mode="privacy_block",
            metadata={"topology": "safe_stop"},
        )

    def _parallel_worker_count(self, decision_input: TopologyDecisionInput) -> int:
        if self._is_high(decision_input.expected_breadth):
            return min(
                self.config.max_parallel_workers,
                max(self.config.min_parallel_workers, decision_input.tool_budget // 2),
            )
        return self.config.min_parallel_workers

    def _team_worker_count(self, decision_input: TopologyDecisionInput) -> int:
        learned_hint = self._learned_team_worker_count(decision_input)
        if learned_hint is not None:
            return learned_hint
        return self._parallel_worker_count(decision_input)

    def _learned_team_worker_count(
        self,
        decision_input: TopologyDecisionInput,
    ) -> Optional[int]:
        context = decision_input.context or {}
        support = self._coerce_unit_float(context.get("learned_team_support"))
        if support <= 0.0:
            return None
        try:
            worker_hint = int(context.get("learned_team_max_workers_hint"))
        except (TypeError, ValueError):
            return None
        return max(
            self.config.min_parallel_workers,
            min(self.config.max_parallel_workers, worker_hint),
        )

    def _team_execution_metadata(
        self,
        decision_input: TopologyDecisionInput,
    ) -> Dict[str, Any]:
        context = decision_input.context or {}
        support = self._coerce_unit_float(context.get("learned_team_support"))
        if support <= 0.0:
            return {}

        metadata: Dict[str, Any] = {}
        for context_key, metadata_key in (
            ("learned_worktree_isolation_hint", "worktree_isolation"),
            ("learned_materialize_worktrees_hint", "materialize_worktrees"),
            ("learned_dry_run_worktrees_hint", "dry_run_worktrees"),
            ("learned_cleanup_worktrees_hint", "cleanup_worktrees"),
        ):
            value = self._coerce_optional_bool(context.get(context_key))
            if value is not None:
                metadata[metadata_key] = value
        return metadata

    def _topology_kind_for_action(self, action: TopologyAction) -> TopologyKind:
        return {
            TopologyAction.DIRECT_RESPONSE: TopologyKind.DIRECT,
            TopologyAction.SINGLE_AGENT: TopologyKind.SINGLE_AGENT,
            TopologyAction.PARALLEL_EXPLORATION: TopologyKind.PARALLEL_EXPLORATION,
            TopologyAction.TEAM_PLAN: TopologyKind.TEAM,
            TopologyAction.ESCALATE_MODEL: TopologyKind.ESCALATED_SINGLE_AGENT,
            TopologyAction.SAFE_STOP: TopologyKind.SAFE_STOP,
        }[action]

    def _build_rationale(
        self,
        action: TopologyAction,
        decision_input: TopologyDecisionInput,
        *,
        policy_metadata: Optional[Dict[str, object]] = None,
    ) -> str:
        del decision_input  # reserved for future rationale enrichment
        if action == TopologyAction.DIRECT_RESPONSE:
            rationale = "Low-complexity, low-depth request favors a direct response path."
        elif action == TopologyAction.SINGLE_AGENT:
            rationale = "Manageable complexity and good observability favor a single-agent loop."
        elif action == TopologyAction.PARALLEL_EXPLORATION:
            rationale = "Breadth-heavy search with enough budget favors parallel exploration."
        elif action == TopologyAction.TEAM_PLAN:
            rationale = "High depth and coordination needs favor an explicit team plan."
        elif action == TopologyAction.ESCALATE_MODEL:
            rationale = (
                "Low confidence or weak observability favors escalating to a stronger model path."
            )
        else:
            rationale = (
                "High privacy sensitivity combined with remote-only execution pressure favors a "
                "safe stop."
            )

        metadata = policy_metadata or {}
        if metadata.get("selection_policy") == "learned_close_override":
            heuristic_action = metadata.get("heuristic_action")
            if isinstance(heuristic_action, TopologyAction):
                heuristic_label = heuristic_action.value.replace("_", " ")
            else:
                heuristic_label = "heuristic default"
            rationale = (
                f"{rationale} High-agreement learned topology feedback overrode the close "
                f"heuristic preference for {heuristic_label}."
            )
        return rationale

    def _normalize_confidence(self, score: float) -> float:
        return max(0.0, min(score, 1.0))

    def _apply_learned_policy_overlay(
        self,
        decision_input: TopologyDecisionInput,
        heuristic_scores: Dict[TopologyAction, float],
    ) -> Tuple[Optional[TopologyAction], Dict[TopologyAction, float], Dict[str, object]]:
        """Apply a guarded learned-policy override for close topology choices."""
        score_breakdown = dict(heuristic_scores)
        ordered = sorted(score_breakdown.items(), key=lambda item: item[1], reverse=True)
        heuristic_best_action, heuristic_best_score = ordered[0]
        metadata: Dict[str, object] = {
            "selection_policy": "heuristic",
            "heuristic_action": heuristic_best_action,
            "heuristic_score": heuristic_best_score,
        }
        context = decision_input.context or {}
        preferred_action = self._feedback_preferred_action(context)
        if preferred_action is None or preferred_action == heuristic_best_action:
            return heuristic_best_action, score_breakdown, metadata

        thresholds = self._resolve_learned_override_thresholds(context)
        metadata.update(self._threshold_metadata(thresholds))
        agreement = self._feedback_agreement(
            context,
            agreement_keys=(
                "learned_topology_action_agreement",
                "learned_topology_kind_agreement",
            ),
        )
        if agreement < thresholds.min_agreement:
            return heuristic_best_action, score_breakdown, metadata
        support = self._feedback_strength(
            context,
            agreement_keys=(
                "learned_topology_action_agreement",
                "learned_topology_kind_agreement",
            ),
        )
        if support < thresholds.min_support:
            return heuristic_best_action, score_breakdown, metadata
        conflict = self._coerce_unit_float(context.get("learned_topology_conflict_score"))
        if conflict > thresholds.max_conflict or thresholds.disabled:
            return heuristic_best_action, score_breakdown, metadata

        preferred_score = score_breakdown.get(preferred_action)
        if preferred_score is None:
            return heuristic_best_action, score_breakdown, metadata
        heuristic_gap = heuristic_best_score - preferred_score
        if heuristic_gap <= 0.0:
            return heuristic_best_action, score_breakdown, metadata
        if heuristic_gap > thresholds.score_gap:
            return heuristic_best_action, score_breakdown, metadata

        score_breakdown[preferred_action] = (
            heuristic_best_score + self.config.learned_override_margin
        )
        metadata.update(
            {
                "selection_policy": "learned_close_override",
                "learned_action": preferred_action,
                "learned_support": round(support, 4),
                "learned_agreement": round(agreement, 4),
                "learned_conflict": round(conflict, 4),
                "learned_gap": round(heuristic_gap, 4),
            }
        )
        return preferred_action, score_breakdown, metadata

    def _resolve_learned_override_thresholds(
        self,
        context: Dict[str, object],
    ) -> LearnedOverrideThresholds:
        """Tune learned-override thresholds from observed policy reward history."""
        reward_delta = self._coerce_optional_float(
            context.get("learned_override_policy_reward_delta")
        )
        optimization_reward_delta = self._coerce_optional_float(
            context.get("learned_override_policy_optimization_reward_delta")
        )
        feasibility_delta = self._coerce_optional_float(
            context.get("learned_override_policy_feasibility_delta")
        )
        experiment_memory_bias = self._coerce_optional_float(
            context.get("experiment_memory_selection_policy_bias")
        )
        experiment_memory_match_count = self._coerce_non_negative_int(
            context.get("experiment_memory_match_count")
        )
        experiment_memory_preferred_policy = self._optional_text(
            context.get("experiment_memory_preferred_selection_policy")
        )
        reward_evidence = min(
            self._coerce_non_negative_int(context.get("learned_override_policy_count")),
            self._coerce_non_negative_int(context.get("heuristic_policy_count")),
        )
        thresholds = LearnedOverrideThresholds(
            score_gap=self.config.learned_override_score_gap,
            min_support=self.config.learned_override_min_support,
            min_agreement=self.config.learned_override_min_agreement,
            max_conflict=self.config.learned_override_max_conflict,
            reward_delta=reward_delta,
            optimization_reward_delta=optimization_reward_delta,
            feasibility_delta=feasibility_delta,
            reward_evidence=reward_evidence,
            experiment_memory_bias=experiment_memory_bias,
            experiment_memory_match_count=experiment_memory_match_count,
            experiment_memory_preferred_policy=experiment_memory_preferred_policy,
        )
        effective_reward_delta = (
            optimization_reward_delta if optimization_reward_delta is not None else reward_delta
        )
        if (
            experiment_memory_match_count > 0
            and experiment_memory_bias is not None
            and experiment_memory_bias <= self.config.experiment_memory_disable_bias_threshold
        ):
            return LearnedOverrideThresholds(
                score_gap=thresholds.score_gap,
                min_support=thresholds.min_support,
                min_agreement=thresholds.min_agreement,
                max_conflict=thresholds.max_conflict,
                reward_delta=reward_delta,
                optimization_reward_delta=optimization_reward_delta,
                feasibility_delta=feasibility_delta,
                reward_evidence=reward_evidence,
                experiment_memory_bias=experiment_memory_bias,
                experiment_memory_match_count=experiment_memory_match_count,
                experiment_memory_preferred_policy=experiment_memory_preferred_policy,
                profile="experiment_memory_disabled_negative",
                disabled=True,
            )
        has_live_policy_signal = not (
            (effective_reward_delta is None and feasibility_delta is None)
            or reward_evidence < self.config.learned_override_min_policy_count
        )
        if not has_live_policy_signal and (
            experiment_memory_match_count <= 0 or experiment_memory_bias is None
        ):
            return thresholds
        if (
            feasibility_delta is not None
            and feasibility_delta <= self.config.learned_override_disable_feasibility_delta
        ):
            return LearnedOverrideThresholds(
                score_gap=thresholds.score_gap,
                min_support=thresholds.min_support,
                min_agreement=thresholds.min_agreement,
                max_conflict=thresholds.max_conflict,
                reward_delta=reward_delta,
                optimization_reward_delta=optimization_reward_delta,
                feasibility_delta=feasibility_delta,
                reward_evidence=reward_evidence,
                experiment_memory_bias=experiment_memory_bias,
                experiment_memory_match_count=experiment_memory_match_count,
                experiment_memory_preferred_policy=experiment_memory_preferred_policy,
                profile="adaptive_disabled_feasibility",
                disabled=True,
            )
        if (
            effective_reward_delta is not None
            and effective_reward_delta <= self.config.learned_override_disable_reward_delta
        ):
            return LearnedOverrideThresholds(
                score_gap=thresholds.score_gap,
                min_support=thresholds.min_support,
                min_agreement=thresholds.min_agreement,
                max_conflict=thresholds.max_conflict,
                reward_delta=reward_delta,
                optimization_reward_delta=optimization_reward_delta,
                feasibility_delta=feasibility_delta,
                reward_evidence=reward_evidence,
                experiment_memory_bias=experiment_memory_bias,
                experiment_memory_match_count=experiment_memory_match_count,
                experiment_memory_preferred_policy=experiment_memory_preferred_policy,
                profile="adaptive_disabled_negative",
                disabled=True,
            )

        tuning_signal = effective_reward_delta if effective_reward_delta is not None else 0.0
        if feasibility_delta is not None:
            if effective_reward_delta is None:
                tuning_signal = feasibility_delta
            else:
                tuning_signal = (effective_reward_delta * 0.7) + (feasibility_delta * 0.3)
        if experiment_memory_match_count > 0 and experiment_memory_bias is not None:
            experiment_memory_signal = self._clamp_float(
                experiment_memory_bias * self.config.experiment_memory_bias_tuning_gain,
                -self.config.learned_override_reward_delta_cap,
                self.config.learned_override_reward_delta_cap,
            )
            if has_live_policy_signal:
                tuning_signal = (tuning_signal * 0.75) + (experiment_memory_signal * 0.25)
            else:
                tuning_signal = experiment_memory_signal
        bounded_delta = self._clamp_float(
            tuning_signal,
            -self.config.learned_override_reward_delta_cap,
            self.config.learned_override_reward_delta_cap,
        )
        if not has_live_policy_signal and experiment_memory_match_count > 0:
            profile = (
                "experiment_memory_positive"
                if bounded_delta > 0.0
                else (
                    "experiment_memory_negative"
                    if bounded_delta < 0.0
                    else "experiment_memory_neutral"
                )
            )
        else:
            profile = (
                "adaptive_positive"
                if bounded_delta > 0.0
                else "adaptive_negative" if bounded_delta < 0.0 else "adaptive_neutral"
            )
        return LearnedOverrideThresholds(
            score_gap=self._clamp_float(
                self.config.learned_override_score_gap
                + (bounded_delta * self.config.learned_override_score_gap_tuning_gain),
                0.05,
                0.2,
            ),
            min_support=self._clamp_float(
                self.config.learned_override_min_support
                - (bounded_delta * self.config.learned_override_support_tuning_gain),
                0.25,
                0.65,
            ),
            min_agreement=self._clamp_float(
                self.config.learned_override_min_agreement
                - (bounded_delta * self.config.learned_override_agreement_tuning_gain),
                0.55,
                0.9,
            ),
            max_conflict=self._clamp_float(
                self.config.learned_override_max_conflict
                + (bounded_delta * self.config.learned_override_conflict_tuning_gain),
                0.15,
                0.6,
            ),
            reward_delta=reward_delta,
            optimization_reward_delta=optimization_reward_delta,
            feasibility_delta=feasibility_delta,
            reward_evidence=reward_evidence,
            experiment_memory_bias=experiment_memory_bias,
            experiment_memory_match_count=experiment_memory_match_count,
            experiment_memory_preferred_policy=experiment_memory_preferred_policy,
            profile=profile,
        )

    def _threshold_metadata(
        self,
        thresholds: LearnedOverrideThresholds,
    ) -> Dict[str, object]:
        """Serialize effective learned-override threshold metadata."""
        metadata: Dict[str, object] = {
            "learned_override_threshold_profile": thresholds.profile,
            "learned_override_effective_score_gap": round(thresholds.score_gap, 4),
            "learned_override_effective_min_support": round(thresholds.min_support, 4),
            "learned_override_effective_min_agreement": round(thresholds.min_agreement, 4),
            "learned_override_effective_max_conflict": round(thresholds.max_conflict, 4),
            "learned_override_disabled": thresholds.disabled,
        }
        if thresholds.reward_delta is not None:
            metadata["learned_override_policy_reward_delta"] = round(
                thresholds.reward_delta,
                4,
            )
        if thresholds.optimization_reward_delta is not None:
            metadata["learned_override_policy_optimization_reward_delta"] = round(
                thresholds.optimization_reward_delta,
                4,
            )
        if thresholds.feasibility_delta is not None:
            metadata["learned_override_policy_feasibility_delta"] = round(
                thresholds.feasibility_delta,
                4,
            )
        if thresholds.reward_evidence > 0:
            metadata["learned_override_policy_evidence"] = thresholds.reward_evidence
        if thresholds.experiment_memory_bias is not None:
            metadata["experiment_memory_selection_policy_bias"] = round(
                thresholds.experiment_memory_bias,
                4,
            )
        if thresholds.experiment_memory_match_count > 0:
            metadata["experiment_memory_match_count"] = thresholds.experiment_memory_match_count
        if thresholds.experiment_memory_preferred_policy is not None:
            metadata["experiment_memory_preferred_selection_policy"] = (
                thresholds.experiment_memory_preferred_policy
            )
        return metadata

    def _feedback_support(self, context: Dict[str, object]) -> float:
        """Estimate how strongly learned feedback should influence routing."""
        explicit_support = self._coerce_unit_float(context.get("learned_topology_support"))
        if explicit_support > 0.0:
            return explicit_support
        coverage = self._coerce_unit_float(context.get("topology_feedback_coverage"))
        reward = self._coerce_unit_float(context.get("avg_topology_reward"))
        return max(0.0, min(1.0, (coverage * 0.55) + (reward * 0.45)))

    def _feedback_strength(
        self,
        context: Dict[str, object],
        *,
        agreement_keys: Tuple[str, ...] = (),
    ) -> float:
        """Blend learned support with agreement/conflict so split hints back off."""
        support = self._feedback_support(context)
        if support <= 0.0:
            return 0.0
        agreement_values = [
            self._coerce_unit_float(context.get(key))
            for key in agreement_keys
            if self._coerce_unit_float(context.get(key)) > 0.0
        ]
        if agreement_values:
            support *= max(agreement_values)
        conflict = self._coerce_unit_float(context.get("learned_topology_conflict_score"))
        if conflict > 0.0:
            support *= max(0.0, 1.0 - (0.75 * conflict))
        return max(0.0, min(1.0, support))

    def _feedback_agreement(
        self,
        context: Dict[str, object],
        *,
        agreement_keys: Tuple[str, ...] = (),
    ) -> float:
        """Return the strongest agreement signal for the requested hint family."""
        agreement_values = [
            self._coerce_unit_float(context.get(key))
            for key in agreement_keys
            if self._coerce_unit_float(context.get(key)) > 0.0
        ]
        if not agreement_values:
            return 1.0
        return max(agreement_values)

    def _feedback_preferred_action(
        self,
        context: Dict[str, object],
    ) -> Optional[TopologyAction]:
        """Resolve the preferred learned action from routing context."""
        preferred_action = self._optional_text(context.get("learned_topology_action"))
        if preferred_action:
            try:
                return TopologyAction(preferred_action)
            except ValueError:
                pass
        preferred_topology = self._optional_text(context.get("learned_topology_kind"))
        if preferred_topology:
            return self._action_for_topology_kind(preferred_topology)
        return None

    def _action_for_topology_kind(self, topology_kind: str) -> Optional[TopologyAction]:
        """Map a serialized topology kind back to its canonical action."""
        normalized = topology_kind.strip().lower()
        mapping = {
            TopologyKind.DIRECT.value: TopologyAction.DIRECT_RESPONSE,
            TopologyKind.SINGLE_AGENT.value: TopologyAction.SINGLE_AGENT,
            TopologyKind.PARALLEL_EXPLORATION.value: TopologyAction.PARALLEL_EXPLORATION,
            TopologyKind.TEAM.value: TopologyAction.TEAM_PLAN,
            TopologyKind.ESCALATED_SINGLE_AGENT.value: TopologyAction.ESCALATE_MODEL,
            TopologyKind.SAFE_STOP.value: TopologyAction.SAFE_STOP,
        }
        return mapping.get(normalized)

    def _resolve_fallback_action(
        self,
        *,
        final_action: TopologyAction,
        score_breakdown: Dict[TopologyAction, float],
        confidence: float,
        heuristic_best_action: Optional[object],
        heuristic_best_score: Optional[object],
    ) -> Optional[TopologyAction]:
        """Choose the runtime fallback while preserving heuristic intent after overrides."""
        if (
            isinstance(heuristic_best_action, TopologyAction)
            and heuristic_best_action != final_action
            and heuristic_best_score is not None
        ):
            return heuristic_best_action

        ordered = sorted(score_breakdown.items(), key=lambda item: item[1], reverse=True)
        best_action, best_score = ordered[0]
        second_action, second_score = ordered[1]
        return self._select_fallback_action(
            confidence=confidence,
            best_action=best_action,
            best_score=best_score,
            second_action=second_action,
            second_score=second_score,
        )

    def _build_telemetry_tags(
        self,
        decision_input: TopologyDecisionInput,
        policy_metadata: Dict[str, object],
    ) -> Dict[str, str]:
        """Build stable telemetry tags for topology selection provenance."""
        tags = {
            "task_type": decision_input.task_type,
            "task_complexity": decision_input.task_complexity,
            "privacy_sensitivity": decision_input.privacy_sensitivity,
            "latency_sensitivity": decision_input.latency_sensitivity,
            "selection_policy": str(policy_metadata.get("selection_policy") or "heuristic"),
        }
        threshold_profile = policy_metadata.get("learned_override_threshold_profile")
        if isinstance(threshold_profile, str) and threshold_profile:
            tags["learned_override_threshold_profile"] = threshold_profile
        heuristic_action = policy_metadata.get("heuristic_action")
        if isinstance(heuristic_action, TopologyAction):
            tags["heuristic_action"] = heuristic_action.value
        learned_action = policy_metadata.get("learned_action")
        if isinstance(learned_action, TopologyAction):
            tags["learned_action"] = learned_action.value
        return tags

    def _grounding_policy_metadata(self, policy_metadata: Dict[str, object]) -> Dict[str, str]:
        """Expose selector provenance on grounded execution metadata."""
        metadata = {
            "selection_policy": str(policy_metadata.get("selection_policy") or "heuristic"),
        }
        for key in (
            "learned_override_threshold_profile",
            "learned_override_effective_score_gap",
            "learned_override_effective_min_support",
            "learned_override_effective_min_agreement",
            "learned_override_effective_max_conflict",
            "learned_override_policy_reward_delta",
            "learned_override_policy_optimization_reward_delta",
            "learned_override_policy_feasibility_delta",
            "learned_override_policy_evidence",
            "experiment_memory_selection_policy_bias",
            "experiment_memory_match_count",
            "experiment_memory_preferred_selection_policy",
            "learned_override_disabled",
        ):
            if key in policy_metadata:
                metadata[key] = policy_metadata[key]
        heuristic_action = policy_metadata.get("heuristic_action")
        if isinstance(heuristic_action, TopologyAction):
            metadata["heuristic_action"] = heuristic_action.value
        learned_action = policy_metadata.get("learned_action")
        if isinstance(learned_action, TopologyAction):
            metadata["learned_action"] = learned_action.value
        return metadata

    def _preferred_provider_from_feedback(
        self,
        decision_input: TopologyDecisionInput,
        candidates: List[str],
    ) -> Optional[str]:
        agreement = self._feedback_agreement(
            decision_input.context or {},
            agreement_keys=("learned_provider_agreement",),
        )
        if (
            not candidates
            or agreement < 0.6
            or self._feedback_strength(
                decision_input.context or {},
                agreement_keys=("learned_provider_agreement",),
            )
            <= 0.0
        ):
            return None
        preferred_provider = self._optional_text(
            (decision_input.context or {}).get("learned_provider_hint")
        )
        if preferred_provider in candidates:
            return preferred_provider
        return None

    def _preferred_formation_from_feedback(
        self,
        decision_input: TopologyDecisionInput,
        formations: List[str],
    ) -> Optional[str]:
        agreement = self._feedback_agreement(
            decision_input.context or {},
            agreement_keys=("learned_formation_agreement",),
        )
        if (
            not formations
            or agreement < 0.6
            or self._feedback_strength(
                decision_input.context or {},
                agreement_keys=("learned_formation_agreement",),
            )
            <= 0.0
        ):
            return None
        preferred_formation = self._optional_text(
            (decision_input.context or {}).get("learned_formation_hint")
        )
        if preferred_formation in formations:
            return preferred_formation
        return None

    def _is_remote_only(self, provider_candidates: List[str]) -> bool:
        if not provider_candidates:
            return False
        return all(not self._is_local_provider(provider) for provider in provider_candidates)

    def _is_local_provider(self, provider_name: str) -> bool:
        normalized = provider_name.strip().lower()
        return any(hint in normalized for hint in self.LOCAL_PROVIDER_HINTS)

    @staticmethod
    def _optional_text(value: object) -> Optional[str]:
        text = str(value).strip() if value is not None else ""
        return text or None

    @staticmethod
    def _coerce_unit_float(value: object) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(numeric, 1.0))

    @staticmethod
    def _coerce_optional_float(value: object) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_optional_bool(value: object) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return None

    @staticmethod
    def _coerce_non_negative_int(value: object) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, numeric)

    @staticmethod
    def _clamp_float(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _is_low(value: str) -> bool:
        return value.strip().lower() == "low"

    @staticmethod
    def _is_medium(value: str) -> bool:
        return value.strip().lower() == "medium"

    @staticmethod
    def _is_high(value: str) -> bool:
        return value.strip().lower() == "high"


__all__ = [
    "TopologySelector",
    "TopologySelectorConfig",
]
