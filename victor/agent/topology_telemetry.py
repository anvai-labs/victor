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

"""Stable telemetry payloads for topology routing decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import logging

from victor.agent.topology_contract import TopologyDecision, TopologyDecisionInput
from victor.agent.topology_grounder import GroundedTopologyPlan

logger = logging.getLogger(__name__)


@dataclass
class TopologyTelemetryEvent:
    """Serialized telemetry event for one topology decision lifecycle."""

    query: str
    task_type: str
    task_complexity: str
    privacy_sensitivity: str
    bandwidth_pressure: str
    latency_sensitivity: str
    token_cost_pressure: str
    action: str
    topology: str
    confidence: float
    rationale: str
    fallback_action: Optional[str] = None
    provider: Optional[str] = None
    formation: Optional[str] = None
    execution_mode: Optional[str] = None
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    telemetry_tags: Dict[str, str] = field(default_factory=dict)
    outcome: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "query": self.query,
            "task_type": self.task_type,
            "task_complexity": self.task_complexity,
            "privacy_sensitivity": self.privacy_sensitivity,
            "bandwidth_pressure": self.bandwidth_pressure,
            "latency_sensitivity": self.latency_sensitivity,
            "token_cost_pressure": self.token_cost_pressure,
            "action": self.action,
            "topology": self.topology,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "fallback_action": self.fallback_action,
            "provider": self.provider,
            "formation": self.formation,
            "execution_mode": self.execution_mode,
            "score_breakdown": dict(self.score_breakdown),
            "telemetry_tags": dict(self.telemetry_tags),
            "outcome": dict(self.outcome),
        }


def build_topology_telemetry_event(
    decision_input: TopologyDecisionInput,
    decision: TopologyDecision,
    grounded_plan: Optional[GroundedTopologyPlan] = None,
    outcome: Optional[Dict[str, Any]] = None,
) -> TopologyTelemetryEvent:
    """Build a topology telemetry event from runtime decisions and outcome."""
    return TopologyTelemetryEvent(
        query=decision_input.query,
        task_type=decision_input.task_type,
        task_complexity=decision_input.task_complexity,
        privacy_sensitivity=decision_input.privacy_sensitivity,
        bandwidth_pressure=decision_input.bandwidth_pressure,
        latency_sensitivity=decision_input.latency_sensitivity,
        token_cost_pressure=decision_input.token_cost_pressure,
        action=decision.action.value,
        topology=decision.topology.value,
        confidence=decision.confidence,
        rationale=decision.rationale,
        fallback_action=(decision.fallback_action.value if decision.fallback_action else None),
        provider=(grounded_plan.provider if grounded_plan else None) or decision.provider,
        formation=(grounded_plan.formation if grounded_plan else None) or decision.formation,
        execution_mode=grounded_plan.execution_mode if grounded_plan else None,
        score_breakdown=dict(decision.score_breakdown),
        telemetry_tags=dict(decision.telemetry_tags),
        outcome=dict(outcome or {}),
    )


async def emit_topology_telemetry_event(
    event: TopologyTelemetryEvent,
    *,
    topic: str = "topology.decision",
    source: str = "victor.agent.topology",
) -> bool:
    """Emit a topology event to the observability bus."""
    try:
        from victor.core.events import get_observability_bus

        bus = get_observability_bus()
        payload = event.to_dict()
        payload.setdefault("category", "topology")
        return await bus.emit(topic=topic, data=payload, source=source)
    except Exception as exc:
        logger.debug("Failed to emit topology telemetry event: %s", exc)
        return False


__all__ = [
    "TopologyTelemetryEvent",
    "build_topology_telemetry_event",
    "emit_topology_telemetry_event",
]
