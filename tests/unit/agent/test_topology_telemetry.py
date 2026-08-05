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

"""Tests for topology telemetry payloads."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.agent.topology_contract import (
    TopologyAction,
    TopologyDecision,
    TopologyDecisionInput,
    TopologyGroundingRequirements,
    TopologyKind,
)
from victor.agent.topology_grounder import GroundedTopologyPlan
from victor.agent.topology_telemetry import (
    TopologyTelemetryEvent,
    build_topology_telemetry_event,
    emit_topology_telemetry_event,
)


class TestTopologyTelemetryEvent:
    """Test topology telemetry event serialization."""

    def test_to_dict_serializes_event(self):
        """Telemetry events should serialize all fields into plain structures."""
        event = TopologyTelemetryEvent(
            query="review auth flow",
            task_type="analysis",
            task_complexity="high",
            privacy_sensitivity="medium",
            bandwidth_pressure="low",
            latency_sensitivity="medium",
            token_cost_pressure="high",
            action="team_plan",
            topology="team",
            confidence=0.83,
            rationale="Deep task favors coordination.",
            fallback_action="escalate_model",
            provider="openai",
            formation="hierarchical",
            execution_mode="team_execution",
            score_breakdown={"team_plan": 0.83},
            telemetry_tags={"task_type": "analysis"},
            outcome={"success": True, "duration_ms": 5200},
        )

        result = event.to_dict()

        assert result["action"] == "team_plan"
        assert result["topology"] == "team"
        assert result["provider"] == "openai"
        assert result["formation"] == "hierarchical"
        assert result["outcome"]["success"] is True


class TestBuildTopologyTelemetryEvent:
    """Test topology telemetry event builder."""

    def test_builds_event_from_decision_and_plan(self):
        """Builder should merge input, decision, grounded plan, and outcome."""
        decision_input = TopologyDecisionInput(
            query="design a new orchestration flow",
            task_type="design",
            task_complexity="high",
            privacy_sensitivity="medium",
            bandwidth_pressure="medium",
            latency_sensitivity="low",
            token_cost_pressure="medium",
        )
        decision = TopologyDecision(
            action=TopologyAction.TEAM_PLAN,
            topology=TopologyKind.TEAM,
            confidence=0.88,
            rationale="High depth and synthesis needs.",
            score_breakdown={"team_plan": 0.88, "single_agent": 0.41},
            grounding_requirements=TopologyGroundingRequirements(
                provider="openai",
                formation="hierarchical",
                max_workers=4,
            ),
            fallback_action=TopologyAction.ESCALATE_MODEL,
            telemetry_tags={"task_type": "design"},
            provider="openai",
            formation="hierarchical",
        )
        plan = GroundedTopologyPlan(
            action=TopologyAction.TEAM_PLAN,
            topology=TopologyKind.TEAM,
            execution_mode="team_execution",
            provider="openai",
            formation="hierarchical",
            max_workers=4,
            tool_budget=12,
            iteration_budget=3,
        )

        event = build_topology_telemetry_event(
            decision_input,
            decision,
            grounded_plan=plan,
            outcome={"success": True, "tool_calls": 7},
        )

        assert isinstance(event, TopologyTelemetryEvent)
        assert event.query == "design a new orchestration flow"
        assert event.action == "team_plan"
        assert event.topology == "team"
        assert event.execution_mode == "team_execution"
        assert event.provider == "openai"
        assert event.formation == "hierarchical"
        assert event.fallback_action == "escalate_model"
        assert event.outcome["tool_calls"] == 7


@pytest.mark.asyncio
async def test_emit_topology_telemetry_event_uses_observability_bus(monkeypatch):
    bus = SimpleNamespace(emit=AsyncMock(return_value=True))
    events_module = __import__("victor.core.events", fromlist=["get_observability_bus"])
    monkeypatch.setattr(events_module, "get_observability_bus", lambda: bus)

    event = TopologyTelemetryEvent(
        query="route query",
        task_type="analysis",
        task_complexity="medium",
        privacy_sensitivity="medium",
        bandwidth_pressure="low",
        latency_sensitivity="medium",
        token_cost_pressure="medium",
        action="single_agent",
        topology="single_agent",
        confidence=0.75,
        rationale="Balanced task",
    )

    emitted = await emit_topology_telemetry_event(event)

    assert emitted is True
    bus.emit.assert_awaited_once()
    assert bus.emit.await_args.kwargs["topic"] == "topology.decision"
    assert bus.emit.await_args.kwargs["data"]["category"] == "topology"
