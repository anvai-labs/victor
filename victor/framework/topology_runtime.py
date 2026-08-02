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

"""Shared topology runtime preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from victor.agent.topology_contract import TopologyAction
from victor.framework.team_runtime import (
    ResolvedTeamExecutionPlan,
    resolve_configured_team,
)


@dataclass
class PreparedTopologyRuntime:
    """Shared runtime-preparation contract for grounded topology plans."""

    action: Optional[str]
    execution_mode: Optional[str]
    runtime_context_overrides: Dict[str, Any] = field(default_factory=dict)
    team_plan: Optional[ResolvedTeamExecutionPlan] = None
    parallel_exploration: Optional[Dict[str, Any]] = None

    def to_result(self, *, prepared: bool) -> Dict[str, Any]:
        """Convert the prepared contract into a serializable runtime payload."""
        result: Dict[str, Any] = {
            "action": self.action,
            "prepared": prepared,
            "execution_mode": self.execution_mode,
        }
        if self.runtime_context_overrides:
            result["runtime_context_overrides"] = dict(self.runtime_context_overrides)
        if self.parallel_exploration:
            result["parallel_exploration"] = dict(self.parallel_exploration)
        if self.team_plan is not None:
            result.update(
                {
                    "team_name": self.team_plan.team_name,
                    "display_name": self.team_plan.display_name,
                    "formation": self.team_plan.formation.value,
                    "member_count": self.team_plan.member_count,
                }
            )
        return result


def derive_topology_task_context(
    task_classification: Any,
    *,
    fallback_task_type: str = "unknown",
    fallback_complexity: str = "medium",
) -> tuple[str, str]:
    """Normalize task classification into topology-routing context."""
    task_type = str(
        getattr(task_classification, "task_type", None)
        or getattr(task_classification, "intent", None)
        or fallback_task_type
    )
    complexity = getattr(task_classification, "complexity", None)
    complexity_value = getattr(complexity, "value", complexity)
    return task_type, str(complexity_value or fallback_complexity)


def prepare_topology_runtime_contract(
    topology_plan: Any,
    *,
    orchestrator: Optional[Any] = None,
    task_type: str = "unknown",
    complexity: str = "medium",
) -> PreparedTopologyRuntime:
    """Prepare shared runtime state for a grounded topology plan."""
    action = getattr(topology_plan, "action", None)
    execution_mode = getattr(topology_plan, "execution_mode", None)
    overrides = _coerce_topology_overrides(topology_plan)
    team_plan = None
    parallel_exploration = None

    if action == TopologyAction.PARALLEL_EXPLORATION:
        parallel_exploration = {
            "force": True,
            "max_results_override": getattr(topology_plan, "tool_budget", None),
        }
    elif action == TopologyAction.TEAM_PLAN and orchestrator is not None:
        team_plan = resolve_configured_team(
            orchestrator,
            task_type=task_type,
            complexity=complexity,
            preferred_team=overrides.get("team_name"),
            preferred_formation=getattr(topology_plan, "formation", None),
            max_workers=getattr(topology_plan, "max_workers", None),
            tool_budget=getattr(topology_plan, "tool_budget", None),
        )
        if team_plan is not None:
            overrides.update(team_plan.to_runtime_context_overrides())

    return PreparedTopologyRuntime(
        action=getattr(action, "value", action),
        execution_mode=execution_mode,
        runtime_context_overrides=overrides,
        team_plan=team_plan,
        parallel_exploration=parallel_exploration,
    )


def _coerce_topology_overrides(topology_plan: Any) -> Dict[str, Any]:
    """Materialize runtime overrides from a grounded topology plan."""
    to_context_overrides = getattr(topology_plan, "to_context_overrides", None)
    if callable(to_context_overrides):
        overrides = to_context_overrides()
        if isinstance(overrides, dict):
            resolved = dict(overrides)
            _promote_team_runtime_policy_hints(resolved)
            return resolved
    return {}


def _promote_team_runtime_policy_hints(overrides: Dict[str, Any]) -> None:
    """Promote team runtime policy hints from topology metadata into flat overrides."""
    metadata = overrides.get("topology_metadata")
    if not isinstance(metadata, dict):
        return
    for key in (
        "worktree_isolation",
        "materialize_worktrees",
        "dry_run_worktrees",
        "cleanup_worktrees",
    ):
        if key in overrides:
            continue
        value = _coerce_optional_bool(metadata.get(key))
        if value is not None:
            overrides[key] = value


def _coerce_optional_bool(value: Any) -> Optional[bool]:
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


__all__ = [
    "PreparedTopologyRuntime",
    "derive_topology_task_context",
    "prepare_topology_runtime_contract",
]
