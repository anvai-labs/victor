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

"""Tests for heuristic topology selector."""

from victor.agent.topology_contract import (
    TopologyAction,
    TopologyDecisionInput,
    TopologyKind,
)
from victor.agent.topology_selector import TopologySelector


class TestTopologySelector:
    """Test heuristic topology choices."""

    def test_selects_single_agent_for_shallow_observable_task(self):
        """Low-complexity observable work should stay in a single-agent loop."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="rename this variable",
                task_type="edit",
                task_complexity="low",
                confidence_hint=0.7,
                expected_depth="low",
                expected_breadth="low",
                observability_level="high",
                provider_candidates=["ollama", "openai"],
            )
        )

        assert decision.action == TopologyAction.SINGLE_AGENT
        assert decision.topology == TopologyKind.SINGLE_AGENT
        assert decision.provider == "ollama"
        assert "single-agent loop" in decision.rationale

    def test_selects_parallel_exploration_for_breadth_heavy_task(self):
        """Breadth-heavy search should favor parallel exploration."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="search for all auth-related failures",
                task_type="analysis",
                task_complexity="high",
                confidence_hint=0.5,
                tool_budget=8,
                expected_depth="medium",
                expected_breadth="high",
                observability_level="low",
                latency_sensitivity="low",
                prior_failures=1,
                provider_candidates=["ollama", "openai"],
            )
        )

        assert decision.action == TopologyAction.PARALLEL_EXPLORATION
        assert decision.topology == TopologyKind.PARALLEL_EXPLORATION
        assert decision.grounding_requirements.max_workers >= 2
        assert "parallel exploration" in decision.rationale

    def test_selects_team_plan_for_deep_complex_task(self):
        """Deep architectural work should favor team planning when formations exist."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="redesign the orchestration architecture",
                task_type="design",
                task_complexity="high",
                confidence_hint=0.6,
                tool_budget=12,
                iteration_budget=3,
                expected_depth="high",
                expected_breadth="medium",
                observability_level="medium",
                latency_sensitivity="low",
                available_team_formations=["parallel", "hierarchical", "adaptive"],
                provider_candidates=["openai", "anthropic"],
            )
        )

        assert decision.action == TopologyAction.TEAM_PLAN
        assert decision.topology == TopologyKind.TEAM
        assert decision.formation == "hierarchical"
        assert decision.grounding_requirements.formation == "hierarchical"
        assert "team plan" in decision.rationale

    def test_selects_escalate_model_for_ambiguous_low_observability_task(self):
        """Low-confidence, low-observability work should escalate model strength."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="determine the root cause from sparse logs",
                task_type="debug",
                task_complexity="high",
                confidence_hint=0.2,
                tool_budget=6,
                expected_depth="medium",
                expected_breadth="low",
                observability_level="low",
                prior_failures=1,
                provider_candidates=["openai", "anthropic"],
            )
        )

        assert decision.action == TopologyAction.ESCALATE_MODEL
        assert decision.topology == TopologyKind.ESCALATED_SINGLE_AGENT
        assert decision.provider == "openai"
        assert decision.grounding_requirements.escalation_target == "openai"
        assert "stronger model path" in decision.rationale

    def test_selects_safe_stop_for_high_privacy_remote_only_case(self):
        """High-privacy remote-only cases should stop safely instead of escalating."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="inspect confidential customer incident data",
                task_type="analysis",
                task_complexity="high",
                confidence_hint=0.2,
                tool_budget=6,
                expected_depth="high",
                expected_breadth="medium",
                privacy_sensitivity="high",
                observability_level="low",
                prior_failures=2,
                provider_candidates=["openai", "anthropic"],
            )
        )

        assert decision.action == TopologyAction.SAFE_STOP
        assert decision.topology == TopologyKind.SAFE_STOP
        assert decision.provider is None
        assert decision.grounding_requirements.safety_mode == "privacy_block"
        assert "safe stop" in decision.rationale

    def test_sets_fallback_action_when_decision_is_close(self):
        """Close or weak decisions should preserve a fallback action."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="investigate an intermittent issue",
                task_type="analysis",
                task_complexity="medium",
                confidence_hint=0.55,
                tool_budget=6,
                expected_depth="medium",
                expected_breadth="medium",
                observability_level="medium",
                latency_sensitivity="medium",
                prior_failures=1,
                provider_candidates=["ollama", "openai"],
            )
        )

        assert decision.fallback_action is not None
        assert decision.fallback_action != decision.action
        assert decision.score_breakdown
        assert "single_agent" in decision.score_breakdown
        assert "escalate_model" in decision.score_breakdown

    def test_learned_override_can_shift_close_decision_toward_learned_action(self):
        """High-agreement learned feedback may override a close heuristic call."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="investigate an intermittent issue",
                task_type="analysis",
                task_complexity="medium",
                confidence_hint=0.52,
                tool_budget=6,
                expected_depth="medium",
                expected_breadth="low",
                observability_level="medium",
                latency_sensitivity="medium",
                prior_failures=1,
                provider_candidates=["ollama", "openai"],
                context={
                    "learned_topology_action": "escalate_model",
                    "learned_topology_kind": "escalated_single_agent",
                    "learned_topology_support": 0.9,
                    "learned_topology_action_agreement": 0.88,
                    "learned_topology_kind_agreement": 0.88,
                    "learned_topology_conflict_score": 0.1,
                },
            )
        )

        assert decision.action == TopologyAction.ESCALATE_MODEL
        assert decision.topology == TopologyKind.ESCALATED_SINGLE_AGENT
        assert decision.fallback_action == TopologyAction.SINGLE_AGENT
        assert decision.telemetry_tags["selection_policy"] == "learned_close_override"
        assert decision.telemetry_tags["learned_override_threshold_profile"] == "static"
        assert decision.telemetry_tags["heuristic_action"] == "single_agent"
        assert decision.telemetry_tags["learned_action"] == "escalate_model"
        assert decision.grounding_requirements.metadata["selection_policy"] == (
            "learned_close_override"
        )
        assert decision.grounding_requirements.metadata["learned_override_effective_score_gap"] == (
            0.1
        )
        assert "overrode the close heuristic preference" in decision.rationale

    def test_feedback_hints_prefer_provider_and_formation_when_supported(self):
        """Learned provider and formation hints should be honored when valid."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="redesign the orchestration architecture",
                task_type="design",
                task_complexity="high",
                confidence_hint=0.6,
                tool_budget=12,
                iteration_budget=3,
                expected_depth="high",
                expected_breadth="medium",
                observability_level="medium",
                latency_sensitivity="low",
                available_team_formations=["parallel", "hierarchical", "adaptive"],
                provider_candidates=["openai", "anthropic"],
                context={
                    "learned_provider_hint": "anthropic",
                    "learned_formation_hint": "parallel",
                    "learned_topology_action": "team_plan",
                    "learned_topology_kind": "team",
                    "learned_topology_support": 0.8,
                },
            )
        )

        assert decision.action == TopologyAction.TEAM_PLAN
        assert decision.provider == "anthropic"
        assert decision.formation == "parallel"

    def test_team_feedback_hint_can_constrain_team_worker_count(self):
        """Merge-safe team feedback should tighten the grounded team worker count."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="redesign the orchestration architecture",
                task_type="design",
                task_complexity="high",
                confidence_hint=0.6,
                tool_budget=12,
                iteration_budget=3,
                expected_depth="high",
                expected_breadth="medium",
                observability_level="medium",
                latency_sensitivity="low",
                available_team_formations=["parallel", "hierarchical", "adaptive"],
                provider_candidates=["openai", "anthropic"],
                context={
                    "learned_formation_hint": "parallel",
                    "learned_team_support": 0.72,
                    "learned_team_max_workers_hint": 2,
                },
            )
        )

        assert decision.action == TopologyAction.TEAM_PLAN
        assert decision.grounding_requirements.max_workers == 2

    def test_team_feedback_hints_can_enable_worktree_isolation(self):
        """Learned team feedback should ground worktree isolation policy into team plans."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="implement the refactor across multiple modules",
                task_type="feature",
                task_complexity="high",
                confidence_hint=0.58,
                tool_budget=10,
                iteration_budget=3,
                expected_depth="high",
                expected_breadth="medium",
                observability_level="medium",
                latency_sensitivity="low",
                available_team_formations=["parallel", "hierarchical", "adaptive"],
                provider_candidates=["openai", "anthropic"],
                context={
                    "learned_team_support": 0.74,
                    "learned_formation_hint": "parallel",
                    "learned_worktree_isolation_hint": True,
                    "learned_dry_run_worktrees_hint": True,
                    "learned_cleanup_worktrees_hint": False,
                },
            )
        )

        assert decision.action == TopologyAction.TEAM_PLAN
        assert decision.grounding_requirements.metadata["worktree_isolation"] is True
        assert decision.grounding_requirements.metadata["dry_run_worktrees"] is True
        assert decision.grounding_requirements.metadata["cleanup_worktrees"] is False

    def test_low_agreement_feedback_does_not_override_base_action(self):
        """Low-agreement learned hints should not override the base heuristic action."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="rename this variable",
                task_type="edit",
                task_complexity="low",
                confidence_hint=0.7,
                expected_depth="low",
                expected_breadth="low",
                observability_level="high",
                provider_candidates=["ollama", "openai"],
                context={
                    "learned_topology_action": "parallel_exploration",
                    "learned_topology_kind": "parallel_exploration",
                    "learned_topology_support": 0.95,
                    "learned_topology_action_agreement": 0.4,
                    "learned_topology_kind_agreement": 0.4,
                    "learned_topology_conflict_score": 0.8,
                },
            )
        )

        assert decision.action == TopologyAction.SINGLE_AGENT
        assert decision.provider == "ollama"

    def test_low_agreement_feedback_does_not_override_provider_or_formation(self):
        """Low-agreement provider and formation hints should fall back to heuristics."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="redesign the orchestration architecture",
                task_type="design",
                task_complexity="high",
                confidence_hint=0.6,
                tool_budget=12,
                iteration_budget=3,
                expected_depth="high",
                expected_breadth="medium",
                observability_level="medium",
                latency_sensitivity="low",
                available_team_formations=["parallel", "hierarchical", "adaptive"],
                provider_candidates=["openai", "anthropic"],
                context={
                    "learned_provider_hint": "anthropic",
                    "learned_formation_hint": "parallel",
                    "learned_topology_action": "team_plan",
                    "learned_topology_kind": "team",
                    "learned_topology_support": 0.9,
                    "learned_provider_agreement": 0.45,
                    "learned_formation_agreement": 0.48,
                    "learned_topology_conflict_score": 0.7,
                },
            )
        )

        assert decision.action == TopologyAction.TEAM_PLAN
        assert decision.provider == "openai"
        assert decision.formation == "hierarchical"

    def test_high_agreement_feedback_does_not_override_when_gap_is_not_close(self):
        """Learned policy should not override when heuristic preference is decisive."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="rename this variable",
                task_type="edit",
                task_complexity="low",
                confidence_hint=0.7,
                expected_depth="low",
                expected_breadth="low",
                observability_level="high",
                provider_candidates=["ollama", "openai"],
                context={
                    "learned_topology_action": "parallel_exploration",
                    "learned_topology_kind": "parallel_exploration",
                    "learned_topology_support": 0.95,
                    "learned_topology_action_agreement": 0.92,
                    "learned_topology_kind_agreement": 0.92,
                    "learned_topology_conflict_score": 0.05,
                },
            )
        )

        assert decision.action == TopologyAction.SINGLE_AGENT
        assert decision.telemetry_tags["selection_policy"] == "heuristic"
        assert decision.telemetry_tags["heuristic_action"] == "single_agent"
        assert "learned_action" not in decision.telemetry_tags
        assert decision.telemetry_tags["learned_override_threshold_profile"] == "static"

    def test_positive_override_reward_delta_relaxes_gap_threshold(self):
        """Positive reward history should widen the close-call gap threshold."""
        selector = TopologySelector()

        baseline = selector.select(
            TopologyDecisionInput(
                query="probe reward-tuned close call",
                task_type="analysis",
                task_complexity="low",
                confidence_hint=0.45,
                tool_budget=4,
                expected_depth="low",
                expected_breadth="medium",
                observability_level="medium",
                latency_sensitivity="low",
                prior_failures=2,
                provider_candidates=["ollama", "openai"],
                context={
                    "learned_topology_action": "escalate_model",
                    "learned_topology_kind": "escalated_single_agent",
                    "learned_topology_support": 0.9,
                    "learned_topology_action_agreement": 0.88,
                    "learned_topology_kind_agreement": 0.88,
                    "learned_topology_conflict_score": 0.1,
                },
            )
        )
        tuned = selector.select(
            TopologyDecisionInput(
                query="probe reward-tuned close call",
                task_type="analysis",
                task_complexity="low",
                confidence_hint=0.45,
                tool_budget=4,
                expected_depth="low",
                expected_breadth="medium",
                observability_level="medium",
                latency_sensitivity="low",
                prior_failures=2,
                provider_candidates=["ollama", "openai"],
                context={
                    "learned_topology_action": "escalate_model",
                    "learned_topology_kind": "escalated_single_agent",
                    "learned_topology_support": 0.9,
                    "learned_topology_action_agreement": 0.88,
                    "learned_topology_kind_agreement": 0.88,
                    "learned_topology_conflict_score": 0.1,
                    "learned_override_policy_reward_delta": 0.25,
                    "learned_override_policy_count": 4,
                    "heuristic_policy_count": 4,
                },
            )
        )

        assert baseline.action == TopologyAction.SINGLE_AGENT
        assert tuned.action == TopologyAction.ESCALATE_MODEL
        assert tuned.fallback_action == TopologyAction.SINGLE_AGENT
        assert tuned.telemetry_tags["selection_policy"] == "learned_close_override"
        assert tuned.telemetry_tags["learned_override_threshold_profile"] == "adaptive_positive"
        assert tuned.grounding_requirements.metadata["learned_override_effective_score_gap"] == (
            0.175
        )

    def test_negative_override_reward_delta_tightens_learned_close_override(self):
        """Mildly negative reward history should tighten, not only hard-disable, overrides."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="investigate an intermittent issue",
                task_type="analysis",
                task_complexity="medium",
                confidence_hint=0.52,
                tool_budget=6,
                expected_depth="medium",
                expected_breadth="low",
                observability_level="medium",
                latency_sensitivity="medium",
                prior_failures=1,
                provider_candidates=["ollama", "openai"],
                context={
                    "learned_topology_action": "escalate_model",
                    "learned_topology_kind": "escalated_single_agent",
                    "learned_topology_support": 0.9,
                    "learned_topology_action_agreement": 0.88,
                    "learned_topology_kind_agreement": 0.88,
                    "learned_topology_conflict_score": 0.1,
                    "learned_override_policy_reward_delta": -0.12,
                    "learned_override_policy_count": 4,
                    "heuristic_policy_count": 4,
                },
            )
        )

        assert decision.action == TopologyAction.SINGLE_AGENT
        assert decision.fallback_action == TopologyAction.ESCALATE_MODEL
        assert decision.telemetry_tags["selection_policy"] == "heuristic"
        assert decision.telemetry_tags["learned_override_threshold_profile"] == "adaptive_negative"
        assert decision.grounding_requirements.metadata["learned_override_effective_score_gap"] == (
            0.064
        )

    def test_strongly_negative_reward_delta_hard_disables_learned_close_override(self):
        """Severely negative reward history should still hard-disable learned overrides."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="investigate an intermittent issue",
                task_type="analysis",
                task_complexity="medium",
                confidence_hint=0.52,
                tool_budget=6,
                expected_depth="medium",
                expected_breadth="low",
                observability_level="medium",
                latency_sensitivity="medium",
                prior_failures=1,
                provider_candidates=["ollama", "openai"],
                context={
                    "learned_topology_action": "escalate_model",
                    "learned_topology_kind": "escalated_single_agent",
                    "learned_topology_support": 0.9,
                    "learned_topology_action_agreement": 0.88,
                    "learned_topology_kind_agreement": 0.88,
                    "learned_topology_conflict_score": 0.1,
                    "learned_override_policy_reward_delta": -0.25,
                    "learned_override_policy_count": 4,
                    "heuristic_policy_count": 4,
                },
            )
        )

        assert decision.action == TopologyAction.SINGLE_AGENT
        assert decision.telemetry_tags["selection_policy"] == "heuristic"
        assert decision.telemetry_tags["learned_override_threshold_profile"] == (
            "adaptive_disabled_negative"
        )
        assert decision.grounding_requirements.metadata["learned_override_disabled"] is True

    def test_positive_optimization_reward_delta_takes_precedence_for_tuning(self):
        """PR2 optimization reward should tune overrides ahead of topology-only reward history."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="probe reward-tuned close call",
                task_type="analysis",
                task_complexity="low",
                confidence_hint=0.45,
                tool_budget=4,
                expected_depth="low",
                expected_breadth="medium",
                observability_level="medium",
                latency_sensitivity="low",
                prior_failures=2,
                provider_candidates=["ollama", "openai"],
                context={
                    "learned_topology_action": "escalate_model",
                    "learned_topology_kind": "escalated_single_agent",
                    "learned_topology_support": 0.9,
                    "learned_topology_action_agreement": 0.88,
                    "learned_topology_kind_agreement": 0.88,
                    "learned_topology_conflict_score": 0.1,
                    "learned_override_policy_reward_delta": -0.1,
                    "learned_override_policy_optimization_reward_delta": 0.25,
                    "learned_override_policy_count": 4,
                    "heuristic_policy_count": 4,
                },
            )
        )

        assert decision.action == TopologyAction.ESCALATE_MODEL
        assert decision.telemetry_tags["selection_policy"] == "learned_close_override"
        assert decision.telemetry_tags["learned_override_threshold_profile"] == "adaptive_positive"
        assert (
            decision.grounding_requirements.metadata[
                "learned_override_policy_optimization_reward_delta"
            ]
            == 0.25
        )

    def test_negative_feasibility_delta_hard_disables_learned_close_override(self):
        """Feasibility regressions should disable learned overrides even with positive reward."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="investigate an intermittent issue",
                task_type="analysis",
                task_complexity="medium",
                confidence_hint=0.52,
                tool_budget=6,
                expected_depth="medium",
                expected_breadth="low",
                observability_level="medium",
                latency_sensitivity="medium",
                prior_failures=1,
                provider_candidates=["ollama", "openai"],
                context={
                    "learned_topology_action": "escalate_model",
                    "learned_topology_kind": "escalated_single_agent",
                    "learned_topology_support": 0.9,
                    "learned_topology_action_agreement": 0.88,
                    "learned_topology_kind_agreement": 0.88,
                    "learned_topology_conflict_score": 0.1,
                    "learned_override_policy_optimization_reward_delta": 0.2,
                    "learned_override_policy_feasibility_delta": -0.3,
                    "learned_override_policy_count": 4,
                    "heuristic_policy_count": 4,
                },
            )
        )

        assert decision.action == TopologyAction.SINGLE_AGENT
        assert decision.telemetry_tags["selection_policy"] == "heuristic"
        assert decision.telemetry_tags["learned_override_threshold_profile"] == (
            "adaptive_disabled_feasibility"
        )
        assert decision.grounding_requirements.metadata["learned_override_disabled"] is True

    def test_negative_experiment_memory_bias_disables_learned_close_override(self):
        """Experiment-memory evidence should suppress learned overrides before live policy data exists."""
        selector = TopologySelector()

        decision = selector.select(
            TopologyDecisionInput(
                query="investigate an intermittent issue",
                task_type="analysis",
                task_complexity="medium",
                confidence_hint=0.52,
                tool_budget=6,
                expected_depth="medium",
                expected_breadth="low",
                observability_level="medium",
                latency_sensitivity="medium",
                prior_failures=1,
                provider_candidates=["ollama", "openai"],
                context={
                    "learned_topology_action": "escalate_model",
                    "learned_topology_kind": "escalated_single_agent",
                    "learned_topology_support": 0.9,
                    "learned_topology_action_agreement": 0.88,
                    "learned_topology_kind_agreement": 0.88,
                    "learned_topology_conflict_score": 0.1,
                    "experiment_memory_match_count": 2,
                    "experiment_memory_support": 0.7,
                    "experiment_memory_selection_policy_bias": -0.8,
                    "experiment_memory_preferred_selection_policy": "heuristic",
                },
            )
        )

        assert decision.action == TopologyAction.SINGLE_AGENT
        assert decision.telemetry_tags["selection_policy"] == "heuristic"
        assert decision.telemetry_tags["learned_override_threshold_profile"] == (
            "experiment_memory_disabled_negative"
        )
        assert decision.grounding_requirements.metadata["learned_override_disabled"] is True
        assert decision.grounding_requirements.metadata[
            "experiment_memory_selection_policy_bias"
        ] == (-0.8)
