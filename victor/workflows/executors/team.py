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

"""Team step executor.

Executes declarative team-step workflow definitions using the framework
team-step adapter runtime.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import TYPE_CHECKING, Any

from victor.teams.types import TeamAgentCategory, TeamFormation, TeamMember

if TYPE_CHECKING:
    from victor.workflows.definition import TeamStepWorkflow
    from victor.workflows.runtime_types import WorkflowState

logger = logging.getLogger(__name__)


class TeamStepExecutor:
    """Executor for declarative team workflow steps."""

    def __init__(self, context: Any = None):
        self._context = context

    async def execute(self, node: "TeamStepWorkflow", state: "WorkflowState") -> "WorkflowState":
        """Execute a team step and merge its output back into workflow state."""
        from victor.agent.subagents import SubAgentRole
        from victor.framework.state_merging import MergeMode
        from victor.framework.workflows.nodes import TeamStep, TeamStepConfig
        from victor.workflows.runtime_types import GraphNodeResult

        logger.info("Executing team step: %s", node.id)
        start_time = time.time()
        current_state = dict(state)

        try:
            team_step = TeamStep(
                id=node.id,
                name=node.name,
                goal=node.goal,
                team_formation=self._resolve_team_formation(node.team_formation, TeamFormation),
                members=self._build_members(node.members, SubAgentRole, TeamMember),
                config=TeamStepConfig(
                    timeout_seconds=node.timeout_seconds,
                    merge_strategy=node.merge_strategy,
                    merge_mode=self._resolve_merge_mode(node.merge_mode, MergeMode),
                    output_key=node.output_key,
                    continue_on_error=node.continue_on_error,
                ),
                shared_context=node.shared_context,
                max_iterations=node.max_iterations,
                total_tool_budget=node.total_tool_budget,
                next_nodes=node.next_nodes,
                retry_policy=node.retry_policy,
            )

            merged_state = await team_step.execute_async(
                self._get_orchestrator(),
                current_state,
            )
            output = merged_state.get(node.output_key, {})

            if "_node_results" not in merged_state:
                merged_state["_node_results"] = {}
            merged_state["_node_results"][node.id] = GraphNodeResult(
                node_id=node.id,
                success=output.get("success", True),
                output=output,
                duration_seconds=time.time() - start_time,
            )
            return merged_state

        except Exception as exc:
            if "_node_results" not in current_state:
                current_state["_node_results"] = {}
            current_state["_error"] = f"Team step '{node.id}' failed: {exc}"
            current_state["_node_results"][node.id] = GraphNodeResult(
                node_id=node.id,
                success=False,
                error=str(exc),
                duration_seconds=time.time() - start_time,
            )
            return current_state

    def _build_members(
        self, members: list[dict[str, Any]], role_enum: Any, member_type: Any
    ) -> list[Any]:
        role_map = {
            "researcher": getattr(role_enum, "RESEARCHER", role_enum.EXECUTOR),
            "planner": getattr(role_enum, "PLANNER", role_enum.EXECUTOR),
            "executor": getattr(role_enum, "EXECUTOR", role_enum.RESEARCHER),
            "reviewer": getattr(role_enum, "REVIEWER", role_enum.EXECUTOR),
            "writer": getattr(role_enum, "WRITER", getattr(role_enum, "EXECUTOR", None)),
            "analyst": getattr(role_enum, "ANALYST", getattr(role_enum, "RESEARCHER", None)),
        }

        built_members = []
        for index, member in enumerate(members):
            role = role_map.get(
                member.get("role", "executor").lower(),
                getattr(role_enum, "EXECUTOR", getattr(role_enum, "RESEARCHER", None)),
            )
            built_members.append(
                member_type(
                    id=member.get("id", f"{role.value}_{index}"),
                    role=role,
                    name=member.get("name", member.get("id", "Member")),
                    goal=member.get("goal", ""),
                    tool_budget=member.get("tool_budget", 15),
                    allowed_tools=member.get("allowed_tools"),
                    agent_category=TeamAgentCategory(
                        member.get(
                            "agent_category",
                            ("supervisor" if member.get("is_manager", False) else "specialist"),
                        )
                    ),
                    can_delegate=member.get("can_delegate", False),
                    delegation_targets=member.get("delegation_targets"),
                    is_manager=member.get("is_manager", False),
                    backstory=member.get("backstory", ""),
                    expertise=member.get("expertise", []),
                    personality=member.get("personality", ""),
                )
            )
        return built_members

    def _resolve_team_formation(self, formation: str, formation_enum: Any) -> Any:
        formation_map = {
            "sequential": formation_enum.SEQUENTIAL,
            "parallel": getattr(formation_enum, "PARALLEL", formation_enum.SEQUENTIAL),
            "hierarchical": getattr(
                formation_enum,
                "HIERARCHICAL",
                formation_enum.SEQUENTIAL,
            ),
            "pipeline": getattr(formation_enum, "PIPELINE", formation_enum.SEQUENTIAL),
            "consensus": getattr(formation_enum, "CONSENSUS", formation_enum.SEQUENTIAL),
        }
        return formation_map.get((formation or "").lower(), formation_enum.SEQUENTIAL)

    def _resolve_merge_mode(self, merge_mode: str, merge_mode_enum: Any) -> Any:
        merge_mode_map = {
            "team_wins": merge_mode_enum.TEAM_WINS,
            "graph_wins": getattr(merge_mode_enum, "GRAPH_WINS", merge_mode_enum.TEAM_WINS),
            "merge": getattr(merge_mode_enum, "MERGE", merge_mode_enum.TEAM_WINS),
            "error": getattr(merge_mode_enum, "ERROR", merge_mode_enum.TEAM_WINS),
        }
        return merge_mode_map.get((merge_mode or "").lower(), merge_mode_enum.TEAM_WINS)

    def _get_orchestrator(self) -> Any:
        if self._context and hasattr(self._context, "orchestrator") and self._context.orchestrator:
            return self._context.orchestrator

        if (
            self._context
            and hasattr(self._context, "services")
            and self._context.services is not None
        ):
            from victor.workflows.orchestrator_pool import OrchestratorPool

            services = self._context.services
            if hasattr(services, "get_optional"):
                pool = services.get_optional(OrchestratorPool)
                if pool is not None:
                    return pool.get_default_orchestrator()

        return None

    def supports_node_type(self, node_type: str) -> bool:
        """Return whether this executor supports the given workflow node type."""
        return node_type == "team"


_DEPRECATED_ALIAS_MAP = {
    "TeamNodeExecutor": TeamStepExecutor,
}
_TEAM_NODE_REMOVAL_VERSION = "v0.9.0"
_TEAM_NODE_REMOVAL_DATE = "2027-03-31"


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_ALIAS_MAP:
        warnings.warn(
            f"{name} is deprecated; use "
            f"{_DEPRECATED_ALIAS_MAP[name].__name__} instead. "
            f"This compatibility alias remains supported through "
            f"{_TEAM_NODE_REMOVAL_VERSION} ({_TEAM_NODE_REMOVAL_DATE}) and "
            "will be removed after that milestone. "
            "See docs/architecture/migration.md for migration guidance.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_ALIAS_MAP[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_DEPRECATED_ALIAS_MAP))
