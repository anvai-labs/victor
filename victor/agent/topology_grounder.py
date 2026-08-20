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

"""Ground topology decisions into executable runtime plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from victor.agent.topology_contract import (
    TopologyAction,
    TopologyDecision,
    TopologyKind,
)


@dataclass
class GroundedTopologyPlan:
    """Executable runtime plan derived from a topology decision."""

    action: TopologyAction
    topology: TopologyKind
    execution_mode: str
    provider: Optional[str] = None
    formation: Optional[str] = None
    max_workers: Optional[int] = None
    tool_budget: int = 0
    iteration_budget: int = 1
    escalation_target: Optional[str] = None
    stop_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "action": self.action.value,
            "topology": self.topology.value,
            "execution_mode": self.execution_mode,
            "provider": self.provider,
            "formation": self.formation,
            "max_workers": self.max_workers,
            "tool_budget": self.tool_budget,
            "iteration_budget": self.iteration_budget,
            "escalation_target": self.escalation_target,
            "stop_reason": self.stop_reason,
            "metadata": dict(self.metadata),
        }

    def to_context_overrides(self) -> Dict[str, Any]:
        """Convert the plan into runtime context hints."""
        overrides: Dict[str, Any] = {
            "topology_action": self.action.value,
            "topology_kind": self.topology.value,
            "execution_mode": self.execution_mode,
            "iteration_budget": self.iteration_budget,
        }
        # Only include tool_budget when it carries a real constraint (> 0).
        # A value of 0 means "not set" and would zero out the service if
        # applied via _apply_tool_budget_override.
        if self.tool_budget:
            overrides["tool_budget"] = self.tool_budget
        if self.provider is not None:
            overrides["provider_hint"] = self.provider
        if self.formation is not None:
            overrides["formation_hint"] = self.formation
        if self.max_workers is not None:
            overrides["max_workers"] = self.max_workers
        if self.escalation_target is not None:
            overrides["escalation_target"] = self.escalation_target
        if self.stop_reason is not None:
            overrides["stop_reason"] = self.stop_reason
        if self.metadata:
            overrides["topology_metadata"] = dict(self.metadata)
        return overrides


class TopologyGrounder:
    """Ground a topology decision into a plan consumable by the runtime."""

    def ground(self, decision: TopologyDecision) -> GroundedTopologyPlan:
        """Convert a topology decision into an executable plan."""
        requirements = decision.grounding_requirements
        execution_mode = self._execution_mode_for(decision.action, decision.topology)
        metadata = {
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            **requirements.metadata,
        }

        stop_reason = None
        if decision.action == TopologyAction.SAFE_STOP:
            stop_reason = requirements.safety_mode or decision.rationale

        return GroundedTopologyPlan(
            action=decision.action,
            topology=decision.topology,
            execution_mode=execution_mode,
            provider=decision.provider or requirements.provider,
            formation=decision.formation or requirements.formation,
            max_workers=requirements.max_workers,
            tool_budget=requirements.tool_budget or 0,
            iteration_budget=requirements.iteration_budget or 1,
            escalation_target=requirements.escalation_target,
            stop_reason=stop_reason,
            metadata=metadata,
        )

    @staticmethod
    def _execution_mode_for(action: TopologyAction, topology: TopologyKind) -> str:
        if action == TopologyAction.DIRECT_RESPONSE:
            return "direct_response"
        if action == TopologyAction.SINGLE_AGENT:
            return "single_agent"
        if action == TopologyAction.PARALLEL_EXPLORATION:
            return "parallel_exploration"
        if action == TopologyAction.TEAM_PLAN:
            return "team_execution"
        if action == TopologyAction.ESCALATE_MODEL:
            return "escalated_single_agent"
        if action == TopologyAction.SAFE_STOP:
            return "safe_stop"
        return topology.value


__all__ = [
    "GroundedTopologyPlan",
    "TopologyGrounder",
]
