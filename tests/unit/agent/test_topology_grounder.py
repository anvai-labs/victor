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

"""Tests for topology grounder."""

from victor.agent.topology_contract import (
    TopologyAction,
    TopologyDecision,
    TopologyGroundingRequirements,
    TopologyKind,
)
from victor.agent.topology_grounder import GroundedTopologyPlan, TopologyGrounder


class TestGroundedTopologyPlan:
    """Test grounded topology plan utilities."""

    def test_to_context_overrides_omits_empty_optional_fields(self):
        """Context overrides should only include populated optional hints."""
        plan = GroundedTopologyPlan(
            action=TopologyAction.SINGLE_AGENT,
            topology=TopologyKind.SINGLE_AGENT,
            execution_mode="single_agent",
            provider="ollama",
            tool_budget=8,
            iteration_budget=2,
        )

        result = plan.to_context_overrides()

        assert result["topology_action"] == "single_agent"
        assert result["provider_hint"] == "ollama"
        assert result["tool_budget"] == 8
        assert result["iteration_budget"] == 2
        assert "formation_hint" not in result
        assert "max_workers" not in result
        assert "stop_reason" not in result

    def test_to_context_overrides_omits_tool_budget_when_zero(self):
        """tool_budget=0 means 'not set'; omit it so _apply_tool_budget_override
        doesn't zero out the service budget."""
        plan = GroundedTopologyPlan(
            action=TopologyAction.SINGLE_AGENT,
            topology=TopologyKind.SINGLE_AGENT,
            execution_mode="single_agent",
            tool_budget=0,
            iteration_budget=1,
        )

        result = plan.to_context_overrides()

        assert "tool_budget" not in result

    def test_to_context_overrides_includes_tool_budget_when_positive(self):
        """Positive tool_budget values should appear in the overrides."""
        plan = GroundedTopologyPlan(
            action=TopologyAction.SINGLE_AGENT,
            topology=TopologyKind.SINGLE_AGENT,
            execution_mode="single_agent",
            tool_budget=25,
            iteration_budget=3,
        )

        result = plan.to_context_overrides()

        assert result["tool_budget"] == 25


class TestTopologyGrounder:
    """Test grounding topology decisions into runtime plans."""

    def test_ground_single_agent_decision(self):
        """Single-agent decisions should preserve provider and budgets."""
        decision = TopologyDecision(
            action=TopologyAction.SINGLE_AGENT,
            topology=TopologyKind.SINGLE_AGENT,
            confidence=0.91,
            rationale="Simple observable task.",
            grounding_requirements=TopologyGroundingRequirements(
                provider="ollama",
                tool_budget=7,
                iteration_budget=2,
                metadata={"source": "heuristic"},
            ),
            provider="ollama",
        )

        plan = TopologyGrounder().ground(decision)

        assert plan.execution_mode == "single_agent"
        assert plan.provider == "ollama"
        assert plan.tool_budget == 7
        assert plan.iteration_budget == 2
        assert plan.metadata["confidence"] == 0.91
        assert plan.metadata["source"] == "heuristic"

    def test_ground_team_plan_decision(self):
        """Team decisions should preserve formation and worker hints."""
        decision = TopologyDecision(
            action=TopologyAction.TEAM_PLAN,
            topology=TopologyKind.TEAM,
            confidence=0.84,
            rationale="High depth and synthesis needs.",
            grounding_requirements=TopologyGroundingRequirements(
                provider="openai",
                formation="hierarchical",
                max_workers=4,
                tool_budget=12,
                iteration_budget=3,
            ),
            provider="openai",
            formation="hierarchical",
        )

        plan = TopologyGrounder().ground(decision)

        assert plan.execution_mode == "team_execution"
        assert plan.provider == "openai"
        assert plan.formation == "hierarchical"
        assert plan.max_workers == 4
        assert plan.tool_budget == 12
        assert plan.iteration_budget == 3

    def test_ground_safe_stop_decision_uses_safety_mode_as_reason(self):
        """Safe-stop decisions should carry a stop reason into the plan."""
        decision = TopologyDecision(
            action=TopologyAction.SAFE_STOP,
            topology=TopologyKind.SAFE_STOP,
            confidence=0.97,
            rationale="Privacy rules block remote escalation.",
            grounding_requirements=TopologyGroundingRequirements(
                safety_mode="privacy_block",
                metadata={"topology": "safe_stop"},
            ),
        )

        plan = TopologyGrounder().ground(decision)

        assert plan.execution_mode == "safe_stop"
        assert plan.stop_reason == "privacy_block"
        assert plan.provider is None
        assert plan.to_context_overrides()["stop_reason"] == "privacy_block"
