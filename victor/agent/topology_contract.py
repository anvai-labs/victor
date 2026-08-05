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

"""Canonical topology decision contract for agent runtime structure selection.

This module defines the shared data structures used by topology-aware routing.
It intentionally stays separate from existing provider and paradigm routing
objects because topology selection operates at a different level:

- provider routing chooses *where* a request runs
- paradigm routing chooses *how deeply* the request should be processed
- topology routing chooses *what runtime structure* should be used

The first implementation wave focuses on latency, bandwidth, privacy, token
cost, and coordination overhead. Battery and power concerns are intentionally
out of scope for this contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TopologyAction(str, Enum):
    """Action selected by the topology router."""

    DIRECT_RESPONSE = "direct_response"
    SINGLE_AGENT = "single_agent"
    PARALLEL_EXPLORATION = "parallel_exploration"
    TEAM_PLAN = "team_plan"
    ESCALATE_MODEL = "escalate_model"
    SAFE_STOP = "safe_stop"


class TopologyKind(str, Enum):
    """Grounded runtime structure chosen for execution."""

    DIRECT = "direct"
    SINGLE_AGENT = "single_agent"
    PARALLEL_EXPLORATION = "parallel_exploration"
    TEAM = "team"
    ESCALATED_SINGLE_AGENT = "escalated_single_agent"
    SAFE_STOP = "safe_stop"


@dataclass
class TopologyGroundingRequirements:
    """Execution requirements needed to ground a topology decision."""

    provider: Optional[str] = None
    formation: Optional[str] = None
    max_workers: Optional[int] = None
    tool_budget: Optional[int] = None
    iteration_budget: Optional[int] = None
    escalation_target: Optional[str] = None
    safety_mode: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "provider": self.provider,
            "formation": self.formation,
            "max_workers": self.max_workers,
            "tool_budget": self.tool_budget,
            "iteration_budget": self.iteration_budget,
            "escalation_target": self.escalation_target,
            "safety_mode": self.safety_mode,
            "metadata": dict(self.metadata),
        }


@dataclass
class TopologyDecisionInput:
    """Input signals used to choose a runtime topology."""

    query: str
    task_type: str = "default"
    task_complexity: str = "medium"
    confidence_hint: float = 0.5
    tool_budget: int = 10
    iteration_budget: int = 1
    expected_depth: str = "medium"
    expected_breadth: str = "medium"
    privacy_sensitivity: str = "medium"
    bandwidth_pressure: str = "medium"
    latency_sensitivity: str = "medium"
    token_cost_pressure: str = "medium"
    observability_level: str = "high"
    prior_failures: int = 0
    similar_experience_count: int = 0
    available_tools: List[str] = field(default_factory=list)
    available_team_formations: List[str] = field(default_factory=list)
    provider_candidates: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        data["available_tools"] = list(self.available_tools)
        data["available_team_formations"] = list(self.available_team_formations)
        data["provider_candidates"] = list(self.provider_candidates)
        data["context"] = dict(self.context)
        return data


@dataclass
class TopologyDecision:
    """Selected topology decision plus grounding and telemetry metadata."""

    action: TopologyAction
    topology: TopologyKind
    confidence: float
    rationale: str
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    grounding_requirements: TopologyGroundingRequirements = field(
        default_factory=TopologyGroundingRequirements
    )
    fallback_action: Optional[TopologyAction] = None
    telemetry_tags: Dict[str, str] = field(default_factory=dict)
    provider: Optional[str] = None
    formation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "action": self.action.value,
            "topology": self.topology.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "score_breakdown": dict(self.score_breakdown),
            "grounding_requirements": self.grounding_requirements.to_dict(),
            "fallback_action": (self.fallback_action.value if self.fallback_action else None),
            "telemetry_tags": dict(self.telemetry_tags),
            "provider": self.provider,
            "formation": self.formation,
        }


__all__ = [
    "TopologyAction",
    "TopologyDecision",
    "TopologyDecisionInput",
    "TopologyGroundingRequirements",
    "TopologyKind",
]
