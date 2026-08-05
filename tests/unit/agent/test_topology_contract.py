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

"""Tests for topology contract types."""

from victor.agent.topology_contract import (
    TopologyAction,
    TopologyDecision,
    TopologyDecisionInput,
    TopologyGroundingRequirements,
    TopologyKind,
)


class TestTopologyAction:
    """Test topology action enum values."""

    def test_action_values_are_stable(self):
        """Topology actions should serialize to stable string values."""
        assert TopologyAction.DIRECT_RESPONSE.value == "direct_response"
        assert TopologyAction.SINGLE_AGENT.value == "single_agent"
        assert TopologyAction.PARALLEL_EXPLORATION.value == "parallel_exploration"
        assert TopologyAction.TEAM_PLAN.value == "team_plan"
        assert TopologyAction.ESCALATE_MODEL.value == "escalate_model"
        assert TopologyAction.SAFE_STOP.value == "safe_stop"


class TestTopologyDecisionInput:
    """Test topology decision input contract."""

    def test_decision_input_defaults(self):
        """Decision input should provide safe defaults for optional fields."""
        decision_input = TopologyDecisionInput(query="analyze this repo")

        assert decision_input.task_type == "default"
        assert decision_input.task_complexity == "medium"
        assert decision_input.confidence_hint == 0.5
        assert decision_input.tool_budget == 10
        assert decision_input.iteration_budget == 1
        assert decision_input.available_tools == []
        assert decision_input.available_team_formations == []
        assert decision_input.provider_candidates == []
        assert decision_input.context == {}

    def test_decision_input_to_dict(self):
        """Decision input should serialize to plain Python structures."""
        decision_input = TopologyDecisionInput(
            query="review the architecture",
            task_type="analysis",
            task_complexity="high",
            confidence_hint=0.7,
            tool_budget=8,
            iteration_budget=3,
            expected_depth="high",
            expected_breadth="medium",
            privacy_sensitivity="high",
            bandwidth_pressure="low",
            latency_sensitivity="medium",
            token_cost_pressure="high",
            observability_level="low",
            prior_failures=2,
            similar_experience_count=5,
            available_tools=["read", "code_search"],
            available_team_formations=["parallel", "hierarchical"],
            provider_candidates=["openai", "anthropic"],
            context={"repo": "victor"},
        )

        result = decision_input.to_dict()

        assert result["query"] == "review the architecture"
        assert result["task_type"] == "analysis"
        assert result["tool_budget"] == 8
        assert result["available_tools"] == ["read", "code_search"]
        assert result["available_team_formations"] == ["parallel", "hierarchical"]
        assert result["provider_candidates"] == ["openai", "anthropic"]
        assert result["context"] == {"repo": "victor"}


class TestTopologyGroundingRequirements:
    """Test topology grounding requirements."""

    def test_grounding_requirements_to_dict(self):
        """Grounding requirements should serialize nested execution hints."""
        grounding = TopologyGroundingRequirements(
            provider="anthropic",
            formation="hierarchical",
            max_workers=3,
            tool_budget=12,
            iteration_budget=4,
            escalation_target="claude-opus",
            safety_mode="strict",
            metadata={"phase": "planning"},
        )

        result = grounding.to_dict()

        assert result == {
            "provider": "anthropic",
            "formation": "hierarchical",
            "max_workers": 3,
            "tool_budget": 12,
            "iteration_budget": 4,
            "escalation_target": "claude-opus",
            "safety_mode": "strict",
            "metadata": {"phase": "planning"},
        }


class TestTopologyDecision:
    """Test topology decision serialization."""

    def test_decision_to_dict_serializes_nested_contract(self):
        """Decision should serialize enums and nested requirements cleanly."""
        decision = TopologyDecision(
            action=TopologyAction.TEAM_PLAN,
            topology=TopologyKind.TEAM,
            confidence=0.82,
            rationale="High depth task with low observability benefits from a team plan.",
            score_breakdown={
                "depth_score": 0.9,
                "observability_penalty": 0.6,
            },
            grounding_requirements=TopologyGroundingRequirements(
                provider="openai",
                formation="hierarchical",
                max_workers=4,
                tool_budget=14,
                metadata={"source": "heuristic"},
            ),
            fallback_action=TopologyAction.ESCALATE_MODEL,
            telemetry_tags={"task_type": "design", "privacy_sensitivity": "medium"},
            provider="openai",
            formation="hierarchical",
        )

        result = decision.to_dict()

        assert result["action"] == "team_plan"
        assert result["topology"] == "team"
        assert result["confidence"] == 0.82
        assert result["rationale"].startswith("High depth task")
        assert result["score_breakdown"] == {
            "depth_score": 0.9,
            "observability_penalty": 0.6,
        }
        assert result["grounding_requirements"]["provider"] == "openai"
        assert result["grounding_requirements"]["formation"] == "hierarchical"
        assert result["fallback_action"] == "escalate_model"
        assert result["telemetry_tags"]["task_type"] == "design"
        assert result["provider"] == "openai"
        assert result["formation"] == "hierarchical"

    def test_decision_without_fallback_serializes_none(self):
        """Fallback action should serialize to None when not provided."""
        decision = TopologyDecision(
            action=TopologyAction.SINGLE_AGENT,
            topology=TopologyKind.SINGLE_AGENT,
            confidence=0.61,
            rationale="Simple, observable task.",
        )

        result = decision.to_dict()

        assert result["fallback_action"] is None
        assert result["grounding_requirements"]["provider"] is None
