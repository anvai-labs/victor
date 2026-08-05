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

"""TeamStep - declarative workflow adapter for multi-agent teams.

This module provides the workflow-layer adapter for invoking ad-hoc
multi-agent teams within workflow graphs. Teams use the ``victor.teams``
runtime infrastructure for coordination and merge results back into graph
state.

Design Principles (SOLID):
    - Single Responsibility: TeamStep only handles team orchestration
    - Open/Closed: Extensible via custom merge strategies
    - Liskov Substitution: Compatible with other WorkflowNode types
    - Interface Segregation: Lean interfaces for specific use cases
    - Dependency Inversion: Depends on protocols (ITeamCoordinator) not concretions

Key Features:
    - Spawn ad-hoc teams with any formation (SEQUENTIAL, PARALLEL, etc.)
    - Automatic state merging with conflict resolution
    - Timeout management for long-running teams
    - Error propagation with configurable behavior
    - Full compatibility with checkpointing

Example:
    from victor.framework.workflows.nodes import TeamStep
    from victor.teams import TeamFormation, TeamMember, TeamConfig
    from victor.agent.subagents import SubAgentRole

    # Define team members
    members = [
        TeamMember(
            id="researcher",
            role=SubAgentRole.RESEARCHER,
            name="Researcher",
            goal="Find information",
        ),
        TeamMember(
            id="writer",
            role=SubAgentRole.WRITER,
            name="Writer",
            goal="Write report",
        ),
    ]

    # Create declarative team step
    team_step = TeamStep(
        id="research_team",
        name="Research Team",
        goal="Conduct comprehensive research",
        team_formation=TeamFormation.SEQUENTIAL,
        members=members,
        timeout_seconds=300,
        merge_strategy="dict",
        output_key="research_result",
    )

    # Use in workflow
    workflow.add_node(team_step)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
)

from victor.framework.state_merging import (
    MergeMode,
    MergeStrategy,
    StateMergeError,
    create_merge_strategy,
)
from victor.core.async_utils import run_sync
from victor.teams.types import TeamAgentCategory, TeamConfig, TeamFormation, TeamMember

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.teams import ITeamCoordinator, TeamResult

logger = logging.getLogger(__name__)


@dataclass
class TeamStepConfig:
    """Configuration for TeamStep execution.

    Attributes:
        timeout_seconds: Maximum execution time (None = no limit)
        merge_strategy: Strategy for merging state ("dict", "list", "custom", "selective")
        merge_mode: How to handle conflicts (TEAM_WINS, GRAPH_WINS, MERGE, ERROR)
        output_key: Key to store team result in graph state
        continue_on_error: Whether to continue graph execution if team fails
        validate_before_merge: Whether to validate merged state
        required_keys: Keys required in team result for validation
        custom_merge: Custom merge function (for merge_strategy="custom")
    """

    timeout_seconds: Optional[float] = None
    merge_strategy: str = "dict"
    merge_mode: MergeMode = MergeMode.TEAM_WINS
    output_key: str = "team_result"
    continue_on_error: bool = True
    validate_before_merge: bool = False
    required_keys: Optional[List[str]] = None
    custom_merge: Optional[Callable[[str, Any, Any], Any]] = None


@dataclass
class TeamStep:
    """Declarative workflow adapter that invokes a multi-agent team.

    ``UnifiedTeamCoordinator`` is the primary runtime team abstraction.
    ``TeamStep`` exists for workflow-definition surfaces that need to invoke
    that capability from declarative config. It should be treated as an
    adapter layer, not as a distinct team runtime model.

    Attributes:
        id: Unique node identifier
        name: Human-readable node name
        goal: Overall goal for the team
        team_formation: How to organize the team (SEQUENTIAL, PARALLEL, etc.)
        members: List of team members with their configurations
        config: TeamStep execution configuration
        shared_context: Initial context to share with all team members
        max_iterations: Maximum iterations for team execution
        total_tool_budget: Total tool budget across all members
        next_nodes: IDs of nodes to execute after this one
        retry_policy: Optional retry policy
    """

    id: str
    name: str
    goal: str
    team_formation: TeamFormation
    members: List[TeamMember]
    config: TeamStepConfig = field(default_factory=TeamStepConfig)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 50
    total_tool_budget: int = 100
    next_nodes: List[str] = field(default_factory=list)
    retry_policy: Optional[Any] = None  # RetryPolicy from victor.workflows.protocols

    def execute(
        self,
        orchestrator: Optional["AgentOrchestrator"],
        graph_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the team step (sync wrapper).

        Args:
            orchestrator: Agent orchestrator for spawning sub-agents
            graph_state: Current workflow graph state

        Returns:
            Updated graph state with team result merged in
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return run_sync(self.execute_async(orchestrator, graph_state))

        # Preserve the sync API inside async callers by hopping to a worker
        # thread, where run_sync() can safely own the event loop.
        import concurrent.futures

        def _execute_in_worker_thread() -> Dict[str, Any]:
            return run_sync(self.execute_async(orchestrator, graph_state))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute_in_worker_thread)
            return future.result()

    async def execute_async(
        self,
        orchestrator: Optional["AgentOrchestrator"],
        graph_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the team step asynchronously.

        This method:
        1. Creates a team coordinator
        2. Adds configured members
        3. Executes the team task
        4. Merges team result back into graph state

        Args:
            orchestrator: Agent orchestrator for spawning sub-agents
            graph_state: Current workflow graph state

        Returns:
            Updated graph state with team result merged in

        Raises:
            StateMergeError: If merge fails and mode is ERROR
            asyncio.TimeoutError: If team execution times out
        """
        start_time = time.time()
        state = dict(graph_state)  # Make mutable copy

        try:
            # Import team coordinator factory
            from victor.teams import create_coordinator, TeamConfig

            # Create team coordinator
            coordinator = create_coordinator(
                orchestrator=orchestrator,
                enable_observability=False,  # Disable to avoid double-events
                enable_rl=False,
            )

            # Create team config
            team_config = TeamConfig(
                name=self.name,
                goal=self._build_goal(state),
                members=self.members,
                formation=self.team_formation,
                max_iterations=self.max_iterations,
                total_tool_budget=self.total_tool_budget,
                shared_context={
                    **self.shared_context,
                    **self._extract_context(state),
                },
                timeout_seconds=int(self.config.timeout_seconds or 600),
            )

            # Members are passed through the TeamConfig — UnifiedTeamCoordinator
            # adapts them to ITeamMember instances inside ``execute_team_config``
            # using the supplied orchestrator. No node-side wiring needed.

            # Execute team with timeout
            if self.config.timeout_seconds:
                team_result = await asyncio.wait_for(
                    self._execute_team(coordinator, team_config),
                    timeout=self.config.timeout_seconds,
                )
            else:
                team_result = await self._execute_team(coordinator, team_config)

            # Merge team result into graph state
            merged_state = self._merge_team_result(state, team_result)

            # Store team result
            output_key = self.config.output_key
            merged_state[output_key] = {
                "success": team_result.success,
                "final_output": team_result.final_output,
                "formation": team_result.formation.value,
                "total_duration": team_result.total_duration,
                "total_tool_calls": team_result.total_tool_calls,
                "member_count": len(team_result.member_results),
            }

            # Log success
            duration = time.time() - start_time
            logger.info(
                f"Team step '{self.id}' completed successfully in {duration:.2f}s "
                f"({len(team_result.member_results)} members)"
            )

            return merged_state

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            error_msg = f"Team step '{self.id}' timed out after {duration:.2f}s"
            logger.warning(error_msg)

            if self.config.continue_on_error:
                state["_error"] = error_msg
                state["_timeout"] = True
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][self.id] = {
                    "success": False,
                    "error": error_msg,
                    "duration_seconds": duration,
                }
                return state
            else:
                raise

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Team step '{self.id}' failed: {e}"
            logger.error(error_msg, exc_info=True)

            if self.config.continue_on_error:
                state["_error"] = error_msg
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][self.id] = {
                    "success": False,
                    "error": str(e),
                    "duration_seconds": duration,
                }
                return state
            else:
                raise

    def _build_goal(self, graph_state: Dict[str, Any]) -> str:
        """Build goal with context substitution.

        Args:
            graph_state: Current graph state

        Returns:
            Goal string with context variables substituted
        """
        goal = self.goal

        # Substitute context variables
        for key, value in graph_state.items():
            if not key.startswith("_"):
                goal = goal.replace(f"${{{key}}}", str(value))
                goal = goal.replace(f"$ctx.{key}", str(value))

        return goal

    def _extract_context(self, graph_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant context for team from graph state.

        Args:
            graph_state: Current graph state

        Returns:
            Context dictionary for team
        """
        # Exclude internal keys
        return {
            k: v for k, v in graph_state.items() if not k.startswith("_") or k in ["_task", "_goal"]
        }

    async def _execute_team(
        self,
        coordinator: "ITeamCoordinator",
        team_config: TeamConfig,
    ) -> "TeamResult":
        """Execute team using coordinator.

        Args:
            coordinator: Team coordinator instance
            team_config: Team configuration

        Returns:
            Team result from execution
        """
        follow_up_contract = (
            team_config.shared_context.get("delegate_follow_up_contract")
            if isinstance(team_config.shared_context, Mapping)
            else None
        )
        if isinstance(follow_up_contract, Mapping):
            from victor.protocols.team import IDelegateFollowUpCoordinator

            if isinstance(coordinator, IDelegateFollowUpCoordinator):
                result = coordinator.execute_follow_up_contract(
                    follow_up_contract,
                    step_id=team_config.shared_context.get("delegate_next_step_id"),
                )
                if inspect.isawaitable(result):
                    result = await result
                return self._coerce_team_result(result, formation=team_config.formation)

        def _get_explicit_callable(name: str):
            if hasattr(type(coordinator), name) or name in vars(coordinator):
                candidate = getattr(coordinator, name, None)
                if callable(candidate):
                    return candidate
            return None

        execute_team_config = _get_explicit_callable("execute_team_config")
        if execute_team_config is not None:
            result = execute_team_config(team_config)
            if inspect.isawaitable(result):
                return await result
            return result

        run = _get_explicit_callable("run")
        if run is not None:
            result = run(team_config)
            if inspect.isawaitable(result):
                return await result
            return result

        # Fallback for lightweight coordinators
        from victor.teams.types import TeamResult, MemberResult

        return TeamResult(
            success=True,
            final_output=f"Team '{team_config.name}' executed with {len(team_config.members)} members",
            member_results={
                m.id: MemberResult(
                    member_id=m.id,
                    success=True,
                    output=f"Member {m.name} completed task",
                )
                for m in team_config.members
            },
            formation=team_config.formation,
        )

    @staticmethod
    def _coerce_team_result(result: Any, *, formation: TeamFormation) -> "TeamResult":
        """Normalize follow-up execution results into workflow ``TeamResult``."""
        from victor.teams.types import MemberResult, TeamResult

        if isinstance(result, TeamResult):
            return result
        if not isinstance(result, Mapping):
            raise TypeError("team follow-up execution must return a mapping or TeamResult")

        raw_member_results = result.get("member_results", {})
        member_results: Dict[str, MemberResult] = {}
        if isinstance(raw_member_results, Mapping):
            for member_id, member_result in raw_member_results.items():
                key = str(member_id)
                if isinstance(member_result, MemberResult):
                    member_results[key] = member_result
                elif isinstance(member_result, Mapping):
                    member_results[key] = MemberResult(
                        member_id=str(member_result.get("member_id") or key),
                        success=bool(member_result.get("success", False)),
                        output=str(member_result.get("output", "")),
                        error=member_result.get("error"),
                        metadata=dict(member_result.get("metadata") or {}),
                        tool_calls_used=int(member_result.get("tool_calls_used", 0)),
                        duration_seconds=float(member_result.get("duration_seconds", 0.0)),
                        discoveries=list(member_result.get("discoveries") or []),
                    )

        return TeamResult(
            success=bool(result.get("success", False)),
            final_output=str(result.get("final_output", "")),
            member_results=member_results,
            formation=formation,
            total_tool_calls=int(result.get("total_tool_calls", 0)),
            total_duration=float(result.get("total_duration", 0.0)),
            communication_log=list(result.get("communication_log", [])),
            shared_context=dict(result.get("shared_context", {})),
            consensus_achieved=result.get("consensus_achieved"),
            consensus_rounds=result.get("consensus_rounds"),
            error=result.get("error"),
        )

    def _merge_team_result(
        self,
        graph_state: Dict[str, Any],
        team_result: "TeamResult",
    ) -> Dict[str, Any]:
        """Merge team result into graph state.

        Args:
            graph_state: Current graph state
            team_result: Result from team execution

        Returns:
            Merged state dictionary

        Raises:
            StateMergeError: If merge fails and mode is ERROR
        """
        # Extract team state from result
        # Include shared_context items directly for merging
        team_state = {
            "team_success": team_result.success,
            "team_output": team_result.final_output,
            "formation": team_result.formation.value,
            # Also store full shared_context for reference
            "_team_shared_context": team_result.shared_context,
        }

        # Add shared_context items directly to team_state for merging
        if team_result.shared_context:
            team_state.update(team_result.shared_context)

        # Create merge strategy
        if self.config.merge_strategy == "custom" and self.config.custom_merge:
            merge_strategy = create_merge_strategy(
                strategy_type="custom",
                mode=self.config.merge_mode,
                conflict_resolver=self.config.custom_merge,
            )
        else:
            merge_strategy = create_merge_strategy(
                strategy_type=self.config.merge_strategy,
                mode=self.config.merge_mode,
            )

        # Perform merge
        try:
            merged_state = merge_strategy.merge(graph_state, team_state)

            # Validate if configured
            if self.config.validate_before_merge and self.config.required_keys:
                from victor.framework.state_merging import validate_merged_state

                validate_merged_state(
                    merged_state,
                    required_keys=self.config.required_keys,
                )

            return merged_state

        except StateMergeError as e:
            if self.config.merge_mode == MergeMode.ERROR:
                raise
            else:
                # Log warning but continue with graph state
                logger.warning(
                    f"State merge failed for team step '{self.id}': {e}. "
                    f"Continuing with graph state."
                )
                return graph_state

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary.

        Returns:
            Dictionary representation of the node
        """
        return {
            "id": self.id,
            "name": self.name,
            "type": "team",
            "goal": self.goal,
            "team_formation": self.team_formation.value,
            "members": [m.to_dict() for m in self.members],
            "config": {
                "timeout_seconds": self.config.timeout_seconds,
                "merge_strategy": self.config.merge_strategy,
                "merge_mode": self.config.merge_mode.value,
                "output_key": self.config.output_key,
                "continue_on_error": self.config.continue_on_error,
                "validate_before_merge": self.config.validate_before_merge,
                "required_keys": self.config.required_keys,
            },
            "shared_context": self.shared_context,
            "max_iterations": self.max_iterations,
            "total_tool_budget": self.total_tool_budget,
            "next_nodes": self.next_nodes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamStep":
        """Deserialize node from dictionary.

        Args:
            data: Dictionary representation of the node

        Returns:
            TeamStep instance
        """
        from victor.core.shared_types import SubAgentRole

        # Deserialize members
        members = []
        for member_data in data.get("members", []):
            # Map role string to enum
            role_map = {
                "researcher": SubAgentRole.RESEARCHER,
                "planner": SubAgentRole.PLANNER,
                "executor": SubAgentRole.EXECUTOR,
                "reviewer": SubAgentRole.REVIEWER,
                "tester": SubAgentRole.TESTER,
            }
            role = role_map.get(member_data["role"].lower(), SubAgentRole.EXECUTOR)

            member = TeamMember(
                id=member_data["id"],
                role=role,
                name=member_data["name"],
                goal=member_data["goal"],
                tool_budget=member_data.get("tool_budget", 15),
                allowed_tools=member_data.get("allowed_tools"),
                agent_category=TeamAgentCategory(
                    member_data.get(
                        "agent_category",
                        ("supervisor" if member_data.get("is_manager", False) else "specialist"),
                    )
                ),
                can_delegate=member_data.get("can_delegate", False),
                delegation_targets=member_data.get("delegation_targets"),
                is_manager=member_data.get("is_manager", False),
                backstory=member_data.get("backstory", ""),
                expertise=member_data.get("expertise", []),
                personality=member_data.get("personality", ""),
            )
            members.append(member)

        # Deserialize config
        config_data = data.get("config", {})
        config = TeamStepConfig(
            timeout_seconds=config_data.get("timeout_seconds"),
            merge_strategy=config_data.get("merge_strategy", "dict"),
            merge_mode=MergeMode(config_data.get("merge_mode", "team_wins")),
            output_key=config_data.get("output_key", "team_result"),
            continue_on_error=config_data.get("continue_on_error", True),
            validate_before_merge=config_data.get("validate_before_merge", False),
            required_keys=config_data.get("required_keys"),
        )

        # Deserialize formation
        formation_map = {
            "sequential": TeamFormation.SEQUENTIAL,
            "parallel": TeamFormation.PARALLEL,
            "hierarchical": TeamFormation.HIERARCHICAL,
            "pipeline": TeamFormation.PIPELINE,
            "consensus": TeamFormation.CONSENSUS,
        }
        formation = formation_map.get(data["team_formation"].lower(), TeamFormation.SEQUENTIAL)

        return cls(
            id=data["id"],
            name=data["name"],
            goal=data["goal"],
            team_formation=formation,
            members=members,
            config=config,
            shared_context=data.get("shared_context", {}),
            max_iterations=data.get("max_iterations", 50),
            total_tool_budget=data.get("total_tool_budget", 100),
            next_nodes=data.get("next_nodes", []),
        )


_DEPRECATED_ALIAS_MAP = {
    "TeamNode": TeamStep,
    "TeamNodeConfig": TeamStepConfig,
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


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_DEPRECATED_ALIAS_MAP))


if TYPE_CHECKING:
    TeamNode = TeamStep
    TeamNodeConfig = TeamStepConfig


__all__ = [
    "TeamStep",
    "TeamStepConfig",
    "TeamNode",
    "TeamNodeConfig",
]
