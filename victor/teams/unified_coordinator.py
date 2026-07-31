# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Unified team coordinator implementing ITeamCoordinator.

This is the production-ready coordinator that combines:
- ITeamCoordinator protocol compliance (LSP)
- All 5 formations including CONSENSUS
- EventBus observability via ObservabilityMixin
- RL integration via RLMixin
- Message bus and shared memory support

Example:
    from victor.teams import UnifiedTeamCoordinator, TeamFormation

    coordinator = UnifiedTeamCoordinator(orchestrator)
    coordinator.set_execution_context(task_type="feature", complexity="high")

    coordinator.add_member(researcher).add_member(executor)
    coordinator.set_formation(TeamFormation.PIPELINE)

    result = await coordinator.execute_task("Implement authentication", {})
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.framework.graph_checkpoint import CheckpointerProtocol, WorkflowCheckpoint
from victor.framework.member_event_sink import MemberEvent, MemberEventSink, current_member_sink
from victor.coordination.formations import (
    SequentialFormation,
    ParallelFormation,
    HierarchicalFormation,
    PipelineFormation,
    ConsensusFormation,
    ReflectionFormation,
)
from victor.teams.mixins.observability import ObservabilityMixin
from victor.teams.mixins.rl import RLMixin
from victor.teams.types import (
    AgentMessage,
    MemberResult,
    MessageType,
    TeamFormation,
    TeamParticipant,
    TeamResult,
)
from victor.teams.workspace_isolation import (
    WorkspaceIsolationService,
    WorkspaceMaterializationOutcome,
)
from victor.teams.worktree_runtime import (
    MaterializedWorktreeAssignment,
    WorktreeAssignment,
    WorktreeExecutionPlan,
    WorktreeMaterializationSession,
)

if TYPE_CHECKING:
    from victor.protocols.team import ITeamMember

logger = logging.getLogger(__name__)


# =============================================================================
# StateGraph Node Configuration
# =============================================================================


@dataclass(frozen=True)
class StateGraphNodeConfig:
    """Configuration for ``UnifiedTeamCoordinator`` used as a StateGraph node.

    The default values match the historical behaviour of ``__call__``: the
    node reads the task from ``state["task"]`` (falling back to
    ``state["query"]``) and writes the team result under ``"result"`` /
    ``"team_output"`` / ``"error"``. Override any field to map the node onto
    an existing graph schema without renaming state keys.

    ``formation_strategy`` is an optional synchronous callable that receives
    the original state and returns a ``TeamFormation``. Its result is injected
    into the per-call context as ``formation_hint`` so the coordinator's
    existing ``_resolve_effective_formation`` does the work —
    ``self._formation`` is never mutated, which keeps concurrent ``__call__``
    invocations safe. Async strategies are intentionally not supported here.
    """

    task_key: str = "task"
    query_key: str = "query"
    result_key: str = "result"
    output_key: str = "team_output"
    error_key: str = "error"
    formation_strategy: Optional[Callable[[Any], TeamFormation]] = None


@dataclass
class _CoordinatorExecutionState:
    """Per-call execution state for concurrent-safe team runs."""

    members: List["ITeamMember"]
    formation: TeamFormation
    supervisor: Optional["ITeamMember"]
    shared_context: Dict[str, Any]
    message_history: List[AgentMessage] = field(default_factory=list)


# =============================================================================
# Team Member Adapter
# =============================================================================


@runtime_checkable
class ContextAgent(Protocol):
    """A role-named agent that context-driven formations invoke directly.

    Reflection-style formations pull "generator"/"critic" agents out of the
    team context and call ``execute(prompt, context)`` on them, using the
    returned string as the produced/critiqued text. This is a narrower contract
    than ``ITeamMember`` — implement it to supply a custom role agent.
    """

    async def execute(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Produce (or critique) text for ``prompt`` and return it as a string."""
        ...


class _MemberContextAgent:
    """Adapts an ``ITeamMember`` to the :class:`ContextAgent` contract.

    Bridges the interface mismatch between the coordinator's members
    (``execute_task(str, dict) -> Any``) and what context-driven formations
    expect (``execute(str, context) -> str``), normalizing str/mapping output.
    """

    def __init__(self, member: "ITeamMember") -> None:
        self._member = member
        self.id = getattr(member, "id", "context_agent")

    async def execute(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        raw = await self._member.execute_task(prompt, context or {})
        if isinstance(raw, Mapping):
            value = raw.get("output") or raw.get("final_output") or raw.get("content") or ""
            return str(value)
        return "" if raw is None else str(raw)


def _member_formation_role(member: Any) -> Optional[str]:
    """Read a member's declared formation_role, looking through adapters.

    Members may be raw ``ITeamMember`` adapters wrapping a ``TeamMember`` dataclass
    (the dataclass carries ``formation_role``), so check both levels.
    """
    role = getattr(member, "formation_role", None)
    if role is None:
        inner = getattr(member, "member", None)
        role = getattr(inner, "formation_role", None)
    return role


class UnifiedTeamCoordinator(ObservabilityMixin, RLMixin):
    """Production-ready team coordinator implementing ITeamCoordinator.

    This coordinator unifies the framework and agent-layer implementations,
    providing full protocol compliance with all production features.

    Features:
        - ITeamCoordinator protocol compliance
        - 5 formation patterns (SEQUENTIAL, PARALLEL, HIERARCHICAL, PIPELINE, CONSENSUS)
        - EventBus observability integration
        - RL integration for team composition learning
        - Message bus for inter-agent communication
        - Shared memory for team context

    Attributes:
        _orchestrator: Agent orchestrator (optional, for SubAgent spawning)
        _members: List of team members
        _formation: Current team formation
        _manager: Supervisor member for HIERARCHICAL formation
        _message_history: Log of inter-agent messages
        _shared_context: Shared context dictionary
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        *,
        enable_observability: bool = True,
        enable_rl: bool = True,
        lightweight_mode: bool = False,
        worktree_planner: Optional[Any] = None,
        merge_analyzer: Optional[Any] = None,
        worktree_runtime: Optional[Any] = None,
        checkpointer: Optional["CheckpointerProtocol"] = None,
    ) -> None:
        """Initialize the unified coordinator.

        Args:
            orchestrator: Optional agent orchestrator for SubAgent spawning
            enable_observability: Enable EventBus integration
            enable_rl: Enable RL integration
            lightweight_mode: If True, disable mixins (for testing without dependencies)
        """
        # Initialize mixins conditionally based on lightweight_mode
        if not lightweight_mode:
            ObservabilityMixin.__init__(self, enable_observability=enable_observability)
            RLMixin.__init__(self, enable_rl=enable_rl)
            self._enable_observability = enable_observability
            self._enable_rl = enable_rl
        else:
            # In lightweight mode, skip mixin initialization
            self._enable_observability = False
            self._enable_rl = False

        # Core state
        self._orchestrator = orchestrator
        # ADR-023: opt-in member-granular checkpointer. None → no checkpoint/resume.
        self._checkpointer: Optional["CheckpointerProtocol"] = checkpointer
        self._members: List[ITeamMember] = []
        self._formation = TeamFormation.SEQUENTIAL
        self._manager: Optional[ITeamMember] = None
        self._lightweight_mode = lightweight_mode

        # Communication
        self._message_history: List[AgentMessage] = []
        self._message_lock = asyncio.Lock()
        self._shared_context: Dict[str, Any] = {}
        self._workspace_isolation = WorkspaceIsolationService(
            planner=worktree_planner,
            runtime=worktree_runtime,
            merge_analyzer=merge_analyzer,
        )

        # LSP capability for language intelligence
        self._lsp: Optional[Any] = None

        # Formation strategies (composition over inheritance)
        self._formations: Dict[TeamFormation, BaseFormationStrategy] = {
            TeamFormation.SEQUENTIAL: SequentialFormation(),
            TeamFormation.PARALLEL: ParallelFormation(),
            TeamFormation.HIERARCHICAL: HierarchicalFormation(),
            TeamFormation.PIPELINE: PipelineFormation(),
            TeamFormation.CONSENSUS: ConsensusFormation(),
            TeamFormation.REFLECTION: ReflectionFormation(),
        }

        # StateGraph node config (used when the coordinator is invoked as a
        # graph node via ``__call__``). Default preserves historical keys.
        self._state_graph_config: StateGraphNodeConfig = StateGraphNodeConfig()

        # Per-task execution state keeps parameterised runs concurrency-safe
        # without mutating coordinator defaults like ``_formation`` or
        # ``_members``.
        self._execution_state: ContextVar[Optional[_CoordinatorExecutionState]] = ContextVar(
            "unified_team_execution_state",
            default=None,
        )

    # =========================================================================
    # ITeamCoordinator Protocol Methods
    # =========================================================================

    def add_member(self, member: "ITeamMember") -> "UnifiedTeamCoordinator":
        """Add a member to the team.

        Args:
            member: Team member implementing ITeamMember protocol

        Returns:
            Self for fluent chaining
        """
        self._members.append(member)
        return self

    def set_formation(self, formation: TeamFormation) -> "UnifiedTeamCoordinator":
        """Set the team formation pattern.

        Args:
            formation: Formation to use

        Returns:
            Self for fluent chaining
        """
        self._formation = formation
        return self

    def set_manager(self, manager: "ITeamMember") -> "UnifiedTeamCoordinator":
        """Compatibility alias for set_supervisor()."""
        return self.set_supervisor(manager)

    def set_supervisor(self, supervisor: "ITeamMember") -> "UnifiedTeamCoordinator":
        """Set the supervisor for HIERARCHICAL formation.

        Args:
            supervisor: Supervisor member

        Returns:
            Self for fluent chaining
        """
        self._manager = supervisor
        if supervisor not in self._members:
            self._members.insert(0, supervisor)
        return self

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with the team.

        Dispatches to the appropriate formation executor based on
        the current formation setting.

        Args:
            task: Task description
            context: Execution context with shared state

        Returns:
            Result dictionary with:
                - success: Whether execution succeeded
                - member_results: Results from each member
                - final_output: Synthesized final output
                - formation: Formation used
        """
        return await self._execute_with(
            task=task,
            context=context,
            formation=self._formation,
            members=list(self._members),
            supervisor=self._manager,
            persist_execution_state=True,
        )

    async def execute_team(
        self,
        config: Any,
        on_member_complete: Optional[Callable[[str, MemberResult], None]] = None,
    ) -> TeamResult:
        """Compatibility entry point for legacy team callers.

        Accepts a ``TeamConfig`` and executes it as a one-shot team run.

        New code should prefer ``execute_task`` or ``execute_team_config``; this
        method is retained to keep higher-level framework APIs stable.
        """
        result = await self.execute_team_config(config)
        if on_member_complete:
            for member in list(config.members):
                member_result = result.member_results.get(member.id)
                if member_result is not None:
                    on_member_complete(member.id, member_result)
        return result

    async def execute_follow_up_request(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Execute a surface-ready delegate follow-up request.

        Args:
            request: Task-plus-context envelope from a delegate follow-up contract

        Returns:
            Team execution result
        """
        task, context = self._normalize_delegate_execution_request(request)
        return await self.execute_task(task, context)

    async def execute_follow_up_contract(
        self,
        contract: Mapping[str, Any],
        *,
        step_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a delegate follow-up contract directly.

        Args:
            contract: Follow-up contract emitted by delegate-mode execution
            step_id: Optional explicit step selection override

        Returns:
            Team execution result
        """
        request = self._resolve_delegate_execution_request_from_follow_up_contract(
            contract,
            step_id=step_id,
        )
        return await self.execute_follow_up_request(request)

    async def broadcast(self, message: AgentMessage) -> List[Optional[AgentMessage]]:
        """Broadcast a message to all team members.

        Args:
            message: Message to broadcast (recipient_id should be None)

        Returns:
            List of responses from members
        """
        async with self._message_lock:
            self._active_message_history().append(message)
        responses: List[Optional[AgentMessage]] = []

        for member in self._active_members():
            if member.id != message.sender_id:  # Don't send to sender
                try:
                    response = await member.receive_message(message)
                    if response:
                        async with self._message_lock:
                            self._active_message_history().append(response)
                    responses.append(response)
                except Exception as e:
                    logger.warning(f"Member {member.id} failed to receive: {e}")
                    responses.append(None)

        return responses

    async def send_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Send a message to a specific team member.

        Args:
            message: Message with recipient_id set

        Returns:
            Response from the recipient, or None if not found
        """
        if not message.recipient_id:
            return None

        async with self._message_lock:
            self._active_message_history().append(message)

        # Find recipient
        for member in self._active_members():
            if member.id == message.recipient_id:
                try:
                    response = await member.receive_message(message)
                    if response:
                        async with self._message_lock:
                            self._active_message_history().append(response)
                    return response
                except Exception as e:
                    logger.warning(f"Member {member.id} failed to receive: {e}")
                    return None

        return None

    # =========================================================================
    # IMessageBusProvider / ISharedMemoryProvider
    # =========================================================================

    @property
    def message_bus(self) -> Optional[Any]:
        """Get message bus (self provides message routing)."""
        return self

    @property
    def shared_memory(self) -> Dict[str, Any]:
        """Get shared memory context."""
        return self._active_shared_context()

    def _current_execution_state(self) -> Optional[_CoordinatorExecutionState]:
        """Return the active per-call execution state, if any."""
        return self._execution_state.get()

    def _active_members(self) -> List["ITeamMember"]:
        state = self._current_execution_state()
        if state is not None:
            return list(state.members)
        return list(self._members)

    def _active_formation(self) -> TeamFormation:
        state = self._current_execution_state()
        if state is not None:
            return state.formation
        return self._formation

    def _active_supervisor(self) -> Optional["ITeamMember"]:
        state = self._current_execution_state()
        if state is not None:
            return state.supervisor
        return self._manager

    def _active_manager(self) -> Optional["ITeamMember"]:
        """Compatibility alias for _active_supervisor()."""
        return self._active_supervisor()

    def _active_shared_context(self) -> Dict[str, Any]:
        state = self._current_execution_state()
        if state is not None:
            return state.shared_context
        return self._shared_context

    def _active_message_history(self) -> List[AgentMessage]:
        state = self._current_execution_state()
        if state is not None:
            return state.message_history
        return self._message_history

    @property
    def lsp(self) -> Optional[Any]:
        """Get the LSP capability for code intelligence in team coordination.

        Returns:
            LSPCapability instance or None
        """
        return self._lsp

    def set_lsp(self, lsp_capability: Any) -> None:
        """Set the LSP capability for team coordination.

        Enables language intelligence features for code-related team
        operations and member coordination.

        Args:
            lsp_capability: LSPCapability instance
        """
        self._lsp = lsp_capability

    # =========================================================================
    # Formation Executors
    # =========================================================================

    def with_checkpointer(
        self, checkpointer: Optional[CheckpointerProtocol]
    ) -> "UnifiedTeamCoordinator":
        """Attach an opt-in member-granular checkpointer (fluent). ADR-023."""
        self._checkpointer = checkpointer
        return self

    async def _load_member_resume(
        self, checkpointer: CheckpointerProtocol, thread_id: str, team_node_id: str
    ) -> Optional[Dict[str, Any]]:
        """Build a ``resume_completed`` payload from the latest member checkpoint.

        Returns ``None`` when there is nothing to resume (fresh run).
        """
        try:
            checkpoints = await checkpointer.list(thread_id)
        except Exception as exc:  # noqa: BLE001 - resume is best-effort
            logger.debug("Member checkpoint list failed (starting fresh): %s", exc)
            return None
        relevant = [c for c in checkpoints if c.metadata.get("team_node_id") == team_node_id]
        if not relevant:
            return None
        latest = max(relevant, key=lambda c: c.timestamp)
        state = latest.state or {}
        member_results = [MemberResult.from_dict(r) for r in state.get("member_results") or []]
        return {
            "member_ids": state.get("completed_member_ids")
            or [r.member_id for r in member_results],
            "member_results": member_results,
            "shared_state": state.get("shared_state") or {},
            "last_output": state.get("last_output"),
            "last_agent_id": state.get("last_agent_id"),
        }

    def _make_member_checkpoint_hook(
        self,
        checkpointer: CheckpointerProtocol,
        thread_id: str,
        team_node_id: str,
        formation: str,
    ) -> Callable[..., Any]:
        """Build the per-member checkpoint callback passed to the formation (ADR-023)."""

        async def _hook(
            index: int,
            member_result: MemberResult,
            results: List[MemberResult],
            shared_state: Dict[str, Any],
        ) -> None:
            last_success = next((r for r in reversed(results) if r.success and r.output), None)
            checkpoint = WorkflowCheckpoint(
                checkpoint_id=f"{thread_id}:{team_node_id}:member:{index}",
                thread_id=thread_id,
                node_id=f"{team_node_id}:member:{member_result.member_id}",
                state={
                    "completed_member_ids": [r.member_id for r in results],
                    "member_results": [r.to_dict() for r in results],
                    "shared_state": dict(shared_state or {}),
                    "last_output": last_success.output if last_success else None,
                    "last_agent_id": last_success.member_id if last_success else None,
                    "next_member_index": index + 1,
                },
                timestamp=time.time(),
                metadata={
                    "team_node_id": team_node_id,
                    "member_id": member_result.member_id,
                    "member_index": index,
                    "formation": formation,
                },
            )
            try:
                await checkpointer.save(checkpoint)
            except Exception as exc:  # noqa: BLE001 - checkpointing must not break the run
                logger.warning("Member checkpoint save failed: %s", exc)

        return _hook

    def _make_member_pause_hook(
        self,
        checkpointer: CheckpointerProtocol,
        thread_id: str,
        team_node_id: str,
        formation: str,
    ) -> Callable[..., Any]:
        """Build the durable pause callback passed to the formation (ADR-023 pillar 2b).

        Invoked when a member reports awaiting-approval. Persists a checkpoint marking the
        pause: ``completed_member_ids`` holds only members finished *before* this one (the
        paused member is excluded so a resumed run re-executes it), plus the pending
        ``approval_request`` so the team can durably resume once the human decides.
        """

        async def _hook(
            index: int,
            member_result: MemberResult,
            results: List[MemberResult],
            shared_state: Dict[str, Any],
        ) -> None:
            # ``results`` holds only the members completed BEFORE the paused one.
            last_success = next((r for r in reversed(results) if r.success and r.output), None)
            checkpoint = WorkflowCheckpoint(
                checkpoint_id=f"{thread_id}:{team_node_id}:member:{index}:pause",
                thread_id=thread_id,
                node_id=f"{team_node_id}:member:{member_result.member_id}",
                state={
                    "completed_member_ids": [r.member_id for r in results],
                    "member_results": [r.to_dict() for r in results],
                    "shared_state": dict(shared_state or {}),
                    "last_output": last_success.output if last_success else None,
                    "last_agent_id": last_success.member_id if last_success else None,
                    "next_member_index": index,
                    "awaiting_approval": True,
                    "paused_member_id": member_result.member_id,
                    "approval_request": (member_result.metadata or {}).get("approval_request"),
                },
                timestamp=time.time(),
                metadata={
                    "team_node_id": team_node_id,
                    "member_id": member_result.member_id,
                    "member_index": index,
                    "formation": formation,
                    "awaiting_approval": True,
                },
            )
            try:
                await checkpointer.save(checkpoint)
            except Exception as exc:  # noqa: BLE001 - checkpointing must not break the run
                logger.warning("Member pause checkpoint save failed: %s", exc)

        return _hook

    def _make_member_event_hook(
        self,
        sink: "MemberEventSink",
        formation: str,
    ) -> Callable[..., Any]:
        """Build the per-member streaming callback passed to the formation (ADR-023).

        The formation awaits this around each member with
        ``(kind, member_id, index, *, success=True, content="")``; it pushes a
        :class:`MemberEvent` onto the stream sink. Emission is bounded/lossy, so this
        never blocks member execution.
        """

        async def _hook(
            kind: str,
            member_id: str,
            index: int,
            *,
            success: bool = True,
            content: str = "",
        ) -> None:
            try:
                await sink.emit(
                    MemberEvent(
                        kind=kind,
                        member_id=member_id,
                        formation=formation,
                        index=index,
                        content=content,
                        success=success,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - streaming must not break the run
                logger.debug("Member event emit failed: %s", exc)

        return _hook

    async def _execute_formation(
        self,
        task: str,
        context: Dict[str, Any],
        formation_override: Optional[TeamFormation] = None,
    ) -> Dict[str, Any]:
        """Execute using formation strategies.

        This replaces the old per-formation methods with a single method
        that delegates to the appropriate formation strategy.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Result dictionary
        """
        # Get the formation strategy
        active_formation = formation_override or self._active_formation()
        strategy = self._formations[active_formation]
        delegate_reentry_contract = self._extract_delegate_reentry_contract(context)
        effective_context = self._apply_delegate_reentry_context(
            context,
            delegate_reentry_contract=delegate_reentry_contract,
        )

        # Wrap team members with participants
        shared_state_with_supervisor = self._active_shared_context()
        context_shared_state = effective_context.get("shared_state")
        if isinstance(context_shared_state, Mapping):
            shared_state_with_supervisor.update(context_shared_state)
        active_supervisor = self._active_supervisor()
        if active_supervisor is not None:
            shared_state_with_supervisor["explicit_supervisor_id"] = active_supervisor.id
            shared_state_with_supervisor["explicit_manager_id"] = active_supervisor.id

        max_workers = self._extract_max_workers(effective_context, shared_state_with_supervisor)
        candidate_members = self._filter_execution_members(
            self._active_members(),
            member_ids=self._extract_delegate_reentry_member_ids(delegate_reentry_contract),
        )
        execution_members = self._limit_execution_members(
            candidate_members,
            active_formation,
            max_workers,
            supervisor=active_supervisor,
        )
        member_context_overrides = self._extract_delegate_reentry_member_context_overrides(
            delegate_reentry_contract
        )
        worktree_plan = None
        worktree_session = None
        workspace_diagnostics: list[Dict[str, Any]] = []
        if not member_context_overrides:
            worktree_plan = self._plan_worktree_execution(
                execution_members,
                context=effective_context,
                formation=active_formation,
            )
            materialization = self._materialize_worktree_plan_with_diagnostics(
                worktree_plan,
                context=effective_context,
            )
            worktree_session = materialization.session
            workspace_diagnostics = materialization.diagnostics_payload()
            worktree_overrides_source = (
                worktree_session.assignments
                if worktree_session
                else (worktree_plan.assignments if worktree_plan is not None else ())
            )
            member_context_overrides = (
                {
                    assignment.member_id: assignment.to_context_overrides()
                    for assignment in worktree_overrides_source
                }
                if worktree_overrides_source
                else {}
            )
        participants = [
            TeamParticipant(
                member=m,
                executor=m.execute_task,
                message_handler=m.receive_message,
                base_context=effective_context,
                context_overrides=member_context_overrides.get(m.id, {}),
            )
            for m in execution_members
        ]

        # Context-driven formations (e.g. reflection) read role-named agents
        # ("generator"/"critic") from context rather than the agents list. Bind
        # them generically — no-op for the other formations.
        context_agents = self._populate_context_agents(strategy, execution_members)
        if context_agents:
            shared_state_with_supervisor.update(context_agents)

        # Create TeamContext
        if max_workers is not None:
            shared_state_with_supervisor["max_workers"] = max_workers
        shared_state_with_supervisor["effective_formation"] = active_formation.value
        if worktree_plan is not None:
            shared_state_with_supervisor["worktree_plan"] = worktree_plan.to_dict()
        if worktree_session is not None:
            shared_state_with_supervisor["worktree_session"] = worktree_session.to_dict()
        if workspace_diagnostics:
            shared_state_with_supervisor["workspace_isolation_diagnostics"] = list(
                workspace_diagnostics
            )

        team_context_id = (
            effective_context.get("team_id") or effective_context.get("team_name") or "UnifiedTeam"
        )
        shared_state_with_supervisor["team_id"] = team_context_id
        team_context_metadata = dict(effective_context)
        for reserved_key in (
            "team_id",
            "formation",
            "shared_state",
            "state_manager",
            "lsp_capability",
        ):
            team_context_metadata.pop(reserved_key, None)

        team_context = TeamContext(
            team_id=team_context_id,
            formation=active_formation.value,
            shared_state=shared_state_with_supervisor,
            **team_context_metadata,
        )

        # ADR-023: member-granular checkpoint/resume — active only when a
        # checkpointer and a stable thread_id are both provided (fully opt-in).
        thread_id = effective_context.get("thread_id") or effective_context.get("__thread_id__")
        if self._checkpointer is not None and thread_id:
            team_node_id = str(team_context_id)
            team_context.resume_completed = await self._load_member_resume(
                self._checkpointer, str(thread_id), team_node_id
            )
            team_context.checkpoint_hook = self._make_member_checkpoint_hook(
                self._checkpointer, str(thread_id), team_node_id, active_formation.value
            )
            # ADR-023 pillar 2b: durable pause when a member awaits approval.
            team_context.pause_hook = self._make_member_pause_hook(
                self._checkpointer, str(thread_id), team_node_id, active_formation.value
            )
            # On resume, surface the human's decision to the re-run member.
            approval_decision = effective_context.get("approval_decision")
            if approval_decision is not None:
                team_context.shared_state["approval_decision"] = approval_decision

        # ADR-023: per-member streaming — when a member event sink is published on the
        # stream context var (i.e. this team runs inside a streamed turn), emit member
        # lifecycle events tagged with member_id so the client can render per-member lanes.
        # Fully opt-in: no sink → hook stays None → no events, unchanged behavior.
        member_sink = current_member_sink.get()
        if member_sink is not None:
            team_context.member_event_hook = self._make_member_event_hook(
                member_sink, active_formation.value
            )

        # Create AgentMessage for the task
        agent_task = AgentMessage(
            sender_id="coordinator",
            content=task,
            message_type=MessageType.TASK,
            data=effective_context,
        )

        result_dict: Optional[Dict[str, Any]] = None
        try:
            # Execute using formation strategy
            member_results_list = await strategy.execute(participants, team_context, agent_task)

            # Convert list of MemberResults to dict
            member_results: Dict[str, MemberResult] = {r.member_id: r for r in member_results_list}
            self._inject_worktree_changed_files(
                member_results,
                worktree_session=worktree_session,
            )

            # Build final output
            success = all(r.success for r in member_results_list) if member_results_list else False
            final_outputs = [r.output for r in member_results_list if r.success]
            total_tool_calls = sum(r.tool_calls_used for r in member_results_list)

            # Determine final output based on formation
            # For pipeline, use only the last stage's output
            # For other formations, join all outputs
            if active_formation == TeamFormation.PIPELINE and final_outputs:
                final_output = final_outputs[-1]  # Last stage's output only
            else:
                final_output = "\n\n".join(final_outputs)

            # Extract consensus metadata if present (from ConsensusFormation)
            result_dict = {
                "success": success,
                "member_results": member_results,
                "final_output": final_output,
                "formation": active_formation.value,
                "total_tool_calls": total_tool_calls,
                "communication_log": list(self._active_message_history()),
                "shared_context": dict(shared_state_with_supervisor),
            }

            # ADR-023 pillar 2b: the formation stopped at a member awaiting approval —
            # surface a durable paused aggregate. The run is resumable on the same thread_id
            # (with an ``approval_decision`` in context) via the persisted pause checkpoint.
            awaiting = team_context.shared_state.get("__awaiting_approval__")
            if awaiting:
                result_dict["success"] = False  # team did not complete — it is paused
                result_dict["status"] = "awaiting_approval"
                result_dict["paused_member_id"] = awaiting.get("member_id")
                result_dict["approval_request"] = awaiting.get("approval_request")
                if thread_id:
                    result_dict["thread_id"] = str(thread_id)

            # Add consensus metadata if any member result has it
            if member_results_list:
                first_metadata = member_results_list[0].metadata
                if "consensus_achieved" in first_metadata:
                    result_dict["consensus_achieved"] = first_metadata["consensus_achieved"]
                if "consensus_rounds" in first_metadata:
                    result_dict["consensus_rounds"] = first_metadata["consensus_rounds"]
            if worktree_plan is not None:
                result_dict["worktree_plan"] = worktree_plan.to_dict()
            if worktree_session is not None:
                result_dict["worktree_session"] = worktree_session.to_dict()
            if workspace_diagnostics:
                result_dict["workspace_isolation_diagnostics"] = list(workspace_diagnostics)
            merge_analysis = self._analyze_merge(member_results, worktree_plan=worktree_plan)
            worker_return_contracts = self._build_worker_return_contracts(
                member_results,
                merge_analysis=merge_analysis,
            )
            if worker_return_contracts:
                result_dict["worker_return_contracts"] = worker_return_contracts
            merge_orchestration: Optional[Dict[str, Any]] = None
            if merge_analysis is not None:
                result_dict["merge_analysis"] = merge_analysis.to_dict()
                result_dict["merge_risk_level"] = merge_analysis.risk_level.value
                if worktree_session is not None:
                    merge_orchestration = self._build_merge_orchestration(
                        worktree_session,
                        merge_analysis=merge_analysis.to_dict(),
                    )
                    if merge_orchestration is not None:
                        result_dict["merge_orchestration"] = merge_orchestration
            if worker_return_contracts:
                merge_review_contract = self._build_merge_review_contract(
                    worker_return_contracts,
                    merge_analysis=merge_analysis,
                    merge_orchestration=merge_orchestration,
                )
                if merge_review_contract:
                    result_dict["merge_review_contract"] = merge_review_contract
                    # Gate: evaluate before executing — review contract gates execution
                    if worktree_session is not None:
                        approval_decision = (
                            self._workspace_isolation.should_execute_merge_with_review(
                                effective_context,
                                merge_review_contract=merge_review_contract,
                                merge_orchestration=merge_orchestration,
                            )
                        )
                        import dataclasses as _dc

                        result_dict["merge_approval_decision"] = _dc.asdict(approval_decision)
                        if approval_decision.approved:
                            merge_execution = self._execute_merge_orchestration(
                                worktree_session,
                                merge_analysis=(
                                    merge_analysis.to_dict() if merge_analysis is not None else None
                                ),
                                context=effective_context,
                            )
                            if merge_execution is not None:
                                result_dict["merge_execution"] = merge_execution
                                merge_execution_diagnostics = self._normalize_workspace_diagnostics(
                                    merge_execution.get("diagnostics")
                                )
                                if merge_execution_diagnostics:
                                    workspace_diagnostics.extend(merge_execution_diagnostics)
                                    result_dict["workspace_isolation_diagnostics"] = list(
                                        workspace_diagnostics
                                    )
                    delegate_follow_up_contract = self._build_delegate_follow_up_contract(
                        worker_return_contracts,
                        merge_review_contract=merge_review_contract,
                        worktree_session=worktree_session,
                        merge_execution=result_dict.get("merge_execution"),
                        merge_analysis=result_dict.get("merge_analysis"),
                        merge_orchestration=merge_orchestration,
                        workspace_diagnostics=workspace_diagnostics,
                        preserve_merge_follow_up=self._resolve_context_mode(effective_context)
                        == "delegate",
                    )
                    if delegate_follow_up_contract:
                        result_dict["delegate_follow_up_contract"] = delegate_follow_up_contract
                        self._attach_delegate_follow_up_suggestions(
                            result_dict,
                            delegate_follow_up_contract,
                        )

            return result_dict
        finally:
            if worktree_session is not None:
                if self._should_cleanup_worktrees(effective_context, result_dict=result_dict):
                    cleanup_summary = self._cleanup_worktree_session(worktree_session)
                else:
                    cleanup_summary = self._build_preserved_worktree_cleanup_summary(
                        worktree_session,
                        reason="preserved_for_follow_up",
                    )
                if result_dict is not None:
                    result_dict["worktree_cleanup"] = cleanup_summary

    def _plan_worktree_execution(
        self,
        members: List["ITeamMember"],
        *,
        context: Dict[str, Any],
        formation: TeamFormation,
    ) -> Optional[WorktreeExecutionPlan]:
        return self._workspace_isolation.plan(members, context=context, formation=formation)

    @staticmethod
    def _coerce_context_flag(
        context: Dict[str, Any],
        key: str,
        *,
        default: bool = False,
    ) -> bool:
        raw_value = context.get(key)
        if raw_value is None:
            return default
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _resolve_context_mode(cls, context: Dict[str, Any]) -> Optional[str]:
        for key in ("mode", "current_mode", "active_mode"):
            raw_value = context.get(key)
            if raw_value is None:
                continue
            value = str(raw_value).strip().lower()
            if value:
                return value
        return None

    def _materialize_worktree_plan(
        self,
        worktree_plan: Optional[WorktreeExecutionPlan],
        *,
        context: Dict[str, Any],
    ) -> Optional[WorktreeMaterializationSession]:
        return self._workspace_isolation.materialize(worktree_plan, context=context)

    def _materialize_worktree_plan_with_diagnostics(
        self,
        worktree_plan: Optional[WorktreeExecutionPlan],
        *,
        context: Dict[str, Any],
    ) -> WorkspaceMaterializationOutcome:
        return self._workspace_isolation.materialize_with_diagnostics(
            worktree_plan,
            context=context,
        )

    def _analyze_merge(
        self,
        member_results: Dict[str, MemberResult],
        *,
        worktree_plan: Optional[WorktreeExecutionPlan],
    ) -> Optional[Any]:
        return self._workspace_isolation.analyze_merge(
            member_results,
            worktree_plan=worktree_plan,
        )

    @staticmethod
    def _coerce_optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _normalize_path_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        normalized: list[str] = []
        for item in list(value or []):
            text = cls._coerce_optional_text(item)
            if text is None:
                continue
            normalized.append(text.rstrip("/"))
        return list(dict.fromkeys(normalized))

    @classmethod
    def _normalize_path_map(cls, value: Any) -> Dict[str, List[str]]:
        if not isinstance(value, Mapping):
            return {}
        normalized: Dict[str, List[str]] = {}
        for member_id, paths in value.items():
            key = cls._coerce_optional_text(member_id)
            if key is None:
                continue
            normalized[key] = cls._normalize_path_list(paths)
        return normalized

    @classmethod
    def _normalize_member_id_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        normalized: list[str] = []
        for item in list(value or []):
            text = cls._coerce_optional_text(item)
            if text is None:
                continue
            normalized.append(text)
        return list(dict.fromkeys(normalized))

    @classmethod
    def _normalize_workspace_diagnostics(cls, value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            value = [value]
        normalized: list[Dict[str, Any]] = []
        for item in list(value or []):
            if not isinstance(item, Mapping):
                continue
            diagnostic = dict(item)
            reason = (
                cls._coerce_optional_text(diagnostic.get("reason"))
                or cls._coerce_optional_text(diagnostic.get("blocked_reason"))
                or cls._coerce_optional_text(diagnostic.get("type"))
                or "workspace_isolation"
            )
            message = (
                cls._coerce_optional_text(diagnostic.get("message"))
                or cls._coerce_optional_text(diagnostic.get("error"))
                or reason
            )
            operation = (
                cls._coerce_optional_text(diagnostic.get("operation")) or "workspace_isolation"
            )
            severity = cls._coerce_optional_text(diagnostic.get("severity")) or "warning"
            details = diagnostic.get("details")
            diagnostic["operation"] = operation
            diagnostic["reason"] = reason
            diagnostic["message"] = message
            diagnostic["severity"] = severity
            diagnostic["details"] = dict(details) if isinstance(details, Mapping) else {}
            normalized.append(diagnostic)
        return normalized

    @classmethod
    def _extract_delegate_reentry_contract(cls, context: Mapping[str, Any]) -> Dict[str, Any]:
        raw_value = context.get("delegate_reentry_contract")
        return dict(raw_value) if isinstance(raw_value, Mapping) else {}

    @classmethod
    def _extract_delegate_merge_contract(cls, context: Mapping[str, Any]) -> Dict[str, Any]:
        raw_value = context.get("delegate_merge_contract")
        return dict(raw_value) if isinstance(raw_value, Mapping) else {}

    @classmethod
    def _extract_delegate_follow_up_contract(cls, context: Mapping[str, Any]) -> Dict[str, Any]:
        raw_value = context.get("delegate_follow_up_contract")
        return dict(raw_value) if isinstance(raw_value, Mapping) else {}

    @classmethod
    def _extract_delegate_approval_contract(cls, context: Mapping[str, Any]) -> Dict[str, Any]:
        raw_value = context.get("delegate_approval_contract")
        if isinstance(raw_value, Mapping):
            return dict(raw_value)
        follow_up_contract = cls._extract_delegate_follow_up_contract(context)
        nested = follow_up_contract.get("approval_contract")
        return dict(nested) if isinstance(nested, Mapping) else {}

    @classmethod
    def _extract_delegate_follow_up_next_steps(
        cls, context: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        follow_up_contract = cls._extract_delegate_follow_up_contract(context)
        top_level_steps = cls._normalize_delegate_next_steps(follow_up_contract.get("next_steps"))
        if top_level_steps:
            return top_level_steps
        approval_contract = cls._extract_delegate_approval_contract(context)
        return cls._normalize_delegate_next_steps(approval_contract.get("next_steps"))

    @classmethod
    def _normalize_delegate_next_steps(cls, value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return []
        normalized: List[Dict[str, Any]] = []
        seen_step_ids: Dict[str, int] = {}
        for index, item in enumerate(list(value), start=1):
            if not isinstance(item, Mapping):
                continue
            payload = dict(item)
            base_step_id = (
                cls._coerce_optional_text(payload.get("step_id"))
                or cls._coerce_optional_text(payload.get("step"))
                or f"delegate_step_{index}"
            )
            occurrence = seen_step_ids.get(base_step_id, 0) + 1
            seen_step_ids[base_step_id] = occurrence
            payload["step_id"] = base_step_id if occurrence == 1 else f"{base_step_id}_{occurrence}"
            normalized.append(payload)
        return normalized

    @classmethod
    def _build_delegate_next_step_request(
        cls,
        *,
        step_id: str,
        next_steps: Sequence[Mapping[str, Any]],
        primary_step_id: Optional[str],
    ) -> Dict[str, Any]:
        request_payload: Dict[str, Any] = {
            "mode": "delegate",
            "delegate_next_step_id": step_id,
            "delegate_follow_up_contract": {
                "next_steps": copy.deepcopy(list(next_steps)),
            },
        }
        if primary_step_id is not None:
            request_payload["delegate_follow_up_contract"]["primary_step_id"] = primary_step_id
        return request_payload

    @classmethod
    def _build_delegate_follow_up_step_requests(
        cls,
        *,
        next_steps: Sequence[Mapping[str, Any]],
        primary_step_id: Optional[str],
    ) -> Dict[str, Dict[str, Any]]:
        normalized_steps = cls._normalize_delegate_next_steps(next_steps)
        step_requests: Dict[str, Dict[str, Any]] = {}
        for step in normalized_steps:
            step_id = cls._coerce_optional_text(step.get("step_id"))
            if step_id is None:
                continue
            step_requests[step_id] = cls._build_delegate_next_step_request(
                step_id=step_id,
                next_steps=normalized_steps,
                primary_step_id=primary_step_id,
            )
        return step_requests

    @classmethod
    def _build_delegate_next_step_execution_request(
        cls,
        *,
        step: Mapping[str, Any],
        step_request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        task = (
            cls._coerce_optional_text(step.get("instruction"))
            or cls._coerce_optional_text(step.get("step"))
            or "Continue delegate follow-up"
        )
        return {
            "task": task,
            "context": copy.deepcopy(dict(step_request)),
        }

    @classmethod
    def _build_delegate_follow_up_step_execution_requests(
        cls,
        *,
        next_steps: Sequence[Mapping[str, Any]],
        step_requests: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        normalized_steps = cls._normalize_delegate_next_steps(next_steps)
        step_execution_requests: Dict[str, Dict[str, Any]] = {}
        for step in normalized_steps:
            step_id = cls._coerce_optional_text(step.get("step_id"))
            if step_id is None:
                continue
            step_request = step_requests.get(step_id)
            if not isinstance(step_request, Mapping):
                continue
            step_execution_requests[step_id] = cls._build_delegate_next_step_execution_request(
                step=step,
                step_request=step_request,
            )
        return step_execution_requests

    @classmethod
    def _build_delegate_follow_up_suggestions(
        cls,
        next_steps: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build compact runtime suggestions from selectable delegate follow-up steps."""
        suggestions: List[Dict[str, Any]] = []
        for step in cls._normalize_delegate_next_steps(next_steps):
            step_id = cls._coerce_optional_text(step.get("step_id"))
            if step_id is None:
                continue
            summary = (
                cls._coerce_optional_text(step.get("instruction"))
                or cls._coerce_optional_text(step.get("description"))
                or cls._coerce_optional_text(step.get("step"))
                or step_id
            )
            description = f"{step_id}: {summary}"
            if len(description) > 120:
                description = description[:117] + "..."
            escaped_step_id = step_id.replace("\\", "\\\\").replace("'", "\\'")
            suggestions.append(
                {
                    "tool": "delegate_follow_up",
                    "command": f"delegate_follow_up(step_id='{escaped_step_id}')",
                    "description": description,
                    "step_id": step_id,
                    "requires_approval": bool(step.get("requires_approval", False)),
                }
            )
        return suggestions

    @classmethod
    def _attach_delegate_follow_up_suggestions(
        cls,
        result_dict: Dict[str, Any],
        delegate_follow_up_contract: Mapping[str, Any],
    ) -> None:
        """Mirror delegate follow-up suggestions onto result artifacts for renderers."""
        raw_suggestions = delegate_follow_up_contract.get("follow_up_suggestions")
        suggestions: List[Dict[str, Any]] = []
        if isinstance(raw_suggestions, Sequence) and not isinstance(raw_suggestions, (str, bytes)):
            suggestions = [
                dict(suggestion)
                for suggestion in raw_suggestions
                if isinstance(suggestion, Mapping)
            ]
        if not suggestions:
            suggestions = cls._build_delegate_follow_up_suggestions(
                delegate_follow_up_contract.get("next_steps") or []
            )
        if not suggestions:
            return
        result_dict["delegate_follow_up_suggestions"] = suggestions
        result_dict["follow_up_suggestions"] = suggestions

    @classmethod
    def _normalize_delegate_execution_request(
        cls,
        request: Mapping[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        if not isinstance(request, Mapping):
            raise TypeError("delegate follow-up request must be a mapping")
        task = cls._coerce_optional_text(request.get("task"))
        if task is None:
            raise ValueError("delegate follow-up request must include a non-empty task")
        context = request.get("context")
        if not isinstance(context, Mapping):
            raise ValueError("delegate follow-up request must include a mapping context")
        return task, copy.deepcopy(dict(context))

    @classmethod
    def _resolve_delegate_execution_request_from_follow_up_contract(
        cls,
        contract: Mapping[str, Any],
        *,
        step_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(contract, Mapping):
            raise TypeError("delegate follow-up contract must be a mapping")

        follow_up_contract = dict(contract)
        normalized_step_id = cls._coerce_optional_text(step_id) or cls._coerce_optional_text(
            follow_up_contract.get("primary_step_id")
        )

        step_execution_requests = (
            dict(follow_up_contract.get("step_execution_requests") or {})
            if isinstance(follow_up_contract.get("step_execution_requests"), Mapping)
            else {}
        )
        if normalized_step_id is not None:
            candidate_request = step_execution_requests.get(normalized_step_id)
            if isinstance(candidate_request, Mapping):
                return copy.deepcopy(dict(candidate_request))
        else:
            primary_request = follow_up_contract.get("primary_step_execution_request")
            if isinstance(primary_request, Mapping):
                return copy.deepcopy(dict(primary_request))

        step_requests = (
            dict(follow_up_contract.get("step_requests") or {})
            if isinstance(follow_up_contract.get("step_requests"), Mapping)
            else {}
        )
        next_steps = cls._normalize_delegate_next_steps(follow_up_contract.get("next_steps"))
        if normalized_step_id is None and next_steps:
            normalized_step_id = cls._coerce_optional_text(next_steps[0].get("step_id"))
        if normalized_step_id is None:
            raise ValueError("delegate follow-up contract does not include a selectable step")

        candidate_step_request = step_requests.get(normalized_step_id)
        if not isinstance(candidate_step_request, Mapping):
            raise ValueError("delegate follow-up contract does not include a runnable step request")

        for step in next_steps:
            candidate_step_id = cls._coerce_optional_text(step.get("step_id"))
            if candidate_step_id == normalized_step_id:
                return cls._build_delegate_next_step_execution_request(
                    step=step,
                    step_request=candidate_step_request,
                )

        raise ValueError("delegate follow-up contract does not include the selected next step")

    @classmethod
    def _resolve_delegate_next_step_by_id(
        cls,
        context: Mapping[str, Any],
        *,
        step_id: str,
    ) -> Dict[str, Any]:
        next_steps = cls._extract_delegate_follow_up_next_steps(context)
        normalized_step_id = cls._coerce_optional_text(step_id)
        if normalized_step_id is None:
            return {}
        for step in next_steps:
            candidate_id = cls._coerce_optional_text(step.get("step_id"))
            candidate_step = cls._coerce_optional_text(step.get("step"))
            if normalized_step_id in {candidate_id, candidate_step}:
                return step
        return {}

    @classmethod
    def _extract_delegate_next_step(cls, context: Mapping[str, Any]) -> Dict[str, Any]:
        raw_value = context.get("delegate_next_step")
        if isinstance(raw_value, Mapping):
            return dict(raw_value)
        step_id = cls._coerce_optional_text(raw_value)
        if step_id is None:
            step_id = cls._coerce_optional_text(context.get("delegate_next_step_id"))
        if step_id is None:
            return {}
        return cls._resolve_delegate_next_step_by_id(context, step_id=step_id)

    @classmethod
    def _apply_delegate_reentry_context(
        cls,
        context: Dict[str, Any],
        *,
        delegate_reentry_contract: Mapping[str, Any],
    ) -> Dict[str, Any]:
        context_overrides = (
            dict(delegate_reentry_contract.get("context_overrides") or {})
            if isinstance(delegate_reentry_contract.get("context_overrides"), Mapping)
            else {}
        )
        effective_context = dict(context_overrides)
        effective_context.update(context)
        return effective_context

    @classmethod
    def _apply_delegate_next_step_context(
        cls,
        context: Dict[str, Any],
        *,
        delegate_next_step: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not delegate_next_step:
            return dict(context)
        selected_step = dict(context)
        step_name = cls._coerce_optional_text(delegate_next_step.get("step"))
        instruction = cls._coerce_optional_text(delegate_next_step.get("instruction"))
        if step_name is not None:
            selected_step.setdefault("delegate_selected_step", step_name)
        if instruction is not None:
            selected_step.setdefault("delegate_selected_instruction", instruction)

        resume_context = (
            dict(delegate_next_step.get("resume_context") or {})
            if isinstance(delegate_next_step.get("resume_context"), Mapping)
            else {}
        )
        execution_context = (
            dict(delegate_next_step.get("execution_context") or {})
            if isinstance(delegate_next_step.get("execution_context"), Mapping)
            else {}
        )
        effective_context = dict(resume_context)
        effective_context.update(execution_context)
        effective_context.update(selected_step)
        return effective_context

    @classmethod
    def _apply_delegate_merge_context(
        cls,
        context: Dict[str, Any],
        *,
        delegate_merge_contract: Mapping[str, Any],
    ) -> Dict[str, Any]:
        context_overrides = (
            dict(delegate_merge_contract.get("context_overrides") or {})
            if isinstance(delegate_merge_contract.get("context_overrides"), Mapping)
            else {}
        )
        effective_context = dict(context_overrides)
        effective_context.update(context)
        return effective_context

    @classmethod
    def _extract_delegate_reentry_member_ids(
        cls,
        delegate_reentry_contract: Mapping[str, Any],
    ) -> List[str]:
        member_ids = cls._normalize_member_id_list(
            delegate_reentry_contract.get("retry_member_ids")
        )
        if member_ids:
            return member_ids
        overrides = delegate_reentry_contract.get("resume_member_context_overrides")
        if isinstance(overrides, Mapping):
            return cls._normalize_member_id_list(overrides.keys())
        return []

    @classmethod
    def _extract_delegate_reentry_member_context_overrides(
        cls,
        delegate_reentry_contract: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        overrides = delegate_reentry_contract.get("resume_member_context_overrides")
        if not isinstance(overrides, Mapping):
            return {}
        next_action = cls._coerce_optional_text(delegate_reentry_contract.get("next_action"))
        retry_tasks_by_member = (
            dict(delegate_reentry_contract.get("retry_tasks_by_member") or {})
            if isinstance(delegate_reentry_contract.get("retry_tasks_by_member"), Mapping)
            else {}
        )
        normalized: Dict[str, Dict[str, Any]] = {}
        for member_id, payload in overrides.items():
            key = cls._coerce_optional_text(member_id)
            if key is None or not isinstance(payload, Mapping):
                continue
            override = dict(payload)
            if next_action is not None:
                override["delegate_reentry_next_action"] = next_action
            follow_up_task_brief = cls._coerce_optional_text(retry_tasks_by_member.get(key))
            if follow_up_task_brief is not None:
                override["follow_up_task_brief"] = follow_up_task_brief
            normalized[key] = override
        return normalized

    @classmethod
    def _filter_execution_members(
        cls,
        members: List["ITeamMember"],
        *,
        member_ids: List[str],
    ) -> List["ITeamMember"]:
        if not member_ids:
            return list(members)
        selected_ids = set(member_ids)
        filtered = [member for member in members if member.id in selected_ids]
        return filtered or list(members)

    @classmethod
    def _summarize_member_output(cls, output: str) -> Optional[str]:
        text = cls._coerce_optional_text(output)
        if text is None:
            return None
        compact = " ".join(text.split())
        if len(compact) <= 280:
            return compact
        return compact[:277].rstrip() + "..."

    @classmethod
    def _normalize_validation_run(cls, metadata: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        raw_validation = metadata.get("validation_run")
        payload = dict(raw_validation) if isinstance(raw_validation, Mapping) else {}
        status = cls._coerce_optional_text(
            payload.get("status")
            or payload.get("result")
            or payload.get("outcome")
            or metadata.get("validation_status")
        )
        command = cls._coerce_optional_text(
            payload.get("command")
            or payload.get("test_command")
            or metadata.get("validation_command")
            or metadata.get("test_command")
        )
        summary = cls._coerce_optional_text(
            payload.get("summary")
            or payload.get("output")
            or payload.get("result_summary")
            or metadata.get("validation_summary")
        )
        normalized: Dict[str, Any] = {}
        if status is not None:
            normalized["status"] = status
        if command is not None:
            normalized["command"] = command
        if summary is not None:
            normalized["summary"] = summary
        return normalized or None

    @classmethod
    def _build_worker_return_contracts(
        cls,
        member_results: Dict[str, MemberResult],
        *,
        merge_analysis: Optional[Any],
    ) -> Dict[str, Dict[str, Any]]:
        if hasattr(merge_analysis, "to_dict"):
            merge_payload = merge_analysis.to_dict()
        elif isinstance(merge_analysis, Mapping):
            merge_payload = dict(merge_analysis)
        else:
            merge_payload = {}

        member_changed_files = cls._normalize_path_map(merge_payload.get("member_changed_files"))
        out_of_scope_writes = cls._normalize_path_map(merge_payload.get("out_of_scope_writes"))
        readonly_violations = cls._normalize_path_map(merge_payload.get("readonly_violations"))
        overlapping_files_by_member: Dict[str, List[str]] = {}
        for conflict in list(merge_payload.get("overlapping_files") or []):
            if not isinstance(conflict, Mapping):
                continue
            path = cls._coerce_optional_text(conflict.get("path"))
            if path is None:
                continue
            for member_id in list(conflict.get("members") or []):
                normalized_member_id = cls._coerce_optional_text(member_id)
                if normalized_member_id is None:
                    continue
                overlapping_files_by_member.setdefault(normalized_member_id, []).append(path)

        contracts: Dict[str, Dict[str, Any]] = {}
        for member_id, result in member_results.items():
            normalized_member_id = cls._coerce_optional_text(member_id)
            if normalized_member_id is None:
                continue
            metadata = dict(result.metadata or {})
            changed_files = cls._normalize_path_list(
                metadata.get("changed_files")
                or metadata.get("files_touched")
                or metadata.get("modified_files")
                or member_changed_files.get(normalized_member_id)
            )
            overlap_paths = cls._normalize_path_list(
                overlapping_files_by_member.get(normalized_member_id)
            )
            out_of_scope_paths = cls._normalize_path_list(
                out_of_scope_writes.get(normalized_member_id)
            )
            readonly_paths = cls._normalize_path_list(readonly_violations.get(normalized_member_id))
            reasons: list[str] = []
            if overlap_paths:
                reasons.append("overlapping_files")
            if readonly_paths:
                reasons.append("readonly_violations")
            if out_of_scope_paths:
                reasons.append("out_of_scope_writes")
            risk_level = "low"
            if overlap_paths or readonly_paths:
                risk_level = "high"
            elif out_of_scope_paths:
                risk_level = "medium"

            task_summary = cls._coerce_optional_text(
                metadata.get("task_summary")
                or metadata.get("summary")
                or metadata.get("result_summary")
            ) or cls._summarize_member_output(result.output)

            contracts[normalized_member_id] = {
                "member_id": normalized_member_id,
                "success": bool(result.success),
                "task_summary": task_summary,
                "changed_files": changed_files,
                "validation_run": cls._normalize_validation_run(metadata),
                "merge_risk": {
                    "level": risk_level,
                    "reasons": reasons,
                    "overlapping_files": overlap_paths,
                    "out_of_scope_writes": out_of_scope_paths,
                    "readonly_violations": readonly_paths,
                },
            }

        return contracts

    def _build_merge_review_contract(
        self,
        worker_return_contracts: Mapping[str, Mapping[str, Any]],
        *,
        merge_analysis: Optional[Any],
        merge_orchestration: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._workspace_isolation.build_merge_review_contract(
            worker_return_contracts,
            merge_analysis=merge_analysis,
            merge_orchestration=merge_orchestration,
        )

    @classmethod
    def _build_delegate_follow_up_contract(
        cls,
        worker_return_contracts: Mapping[str, Mapping[str, Any]],
        *,
        merge_review_contract: Mapping[str, Any],
        worktree_session: Optional[WorktreeMaterializationSession],
        merge_execution: Optional[Mapping[str, Any]] = None,
        merge_analysis: Optional[Mapping[str, Any]] = None,
        merge_orchestration: Optional[Mapping[str, Any]] = None,
        workspace_diagnostics: Optional[Sequence[Mapping[str, Any]]] = None,
        preserve_merge_follow_up: bool = False,
    ) -> Dict[str, Any]:
        next_action = (
            cls._coerce_optional_text(merge_review_contract.get("next_action")) or "inspect"
        )
        workspace_diagnostics_payload = cls._normalize_workspace_diagnostics(workspace_diagnostics)
        merge_execution_payload = (
            dict(merge_execution or {}) if isinstance(merge_execution, Mapping) else {}
        )
        merge_executed = bool(merge_execution_payload.get("executed", False))
        validation_failed_members = cls._normalize_path_list(
            merge_review_contract.get("validation_failed_members")
        )
        review_required_members = cls._normalize_path_list(
            merge_review_contract.get("review_required_members")
        )

        fix_validation_queue: list[Dict[str, Any]] = []
        for member_id in validation_failed_members:
            contract = (
                dict(worker_return_contracts.get(member_id) or {})
                if isinstance(worker_return_contracts.get(member_id), Mapping)
                else {}
            )
            validation_run = (
                dict(contract.get("validation_run") or {})
                if isinstance(contract.get("validation_run"), Mapping)
                else {}
            )
            fix_validation_queue.append(
                {
                    "member_id": member_id,
                    "validation_command": cls._coerce_optional_text(validation_run.get("command")),
                    "validation_summary": cls._coerce_optional_text(validation_run.get("summary")),
                    "changed_files": cls._normalize_path_list(contract.get("changed_files")),
                }
            )

        review_queue: list[Dict[str, Any]] = []
        for member_id in review_required_members:
            contract = (
                dict(worker_return_contracts.get(member_id) or {})
                if isinstance(worker_return_contracts.get(member_id), Mapping)
                else {}
            )
            merge_risk = (
                dict(contract.get("merge_risk") or {})
                if isinstance(contract.get("merge_risk"), Mapping)
                else {}
            )
            review_queue.append(
                {
                    "member_id": member_id,
                    "merge_risk_level": cls._coerce_optional_text(merge_risk.get("level")) or "low",
                    "merge_risk_reasons": list(merge_risk.get("reasons") or []),
                    "changed_files": cls._normalize_path_list(contract.get("changed_files")),
                    "task_summary": cls._coerce_optional_text(contract.get("task_summary")),
                }
            )

        preserve_worktrees = bool(
            worktree_session is not None
            and (
                next_action in {"fix_validation", "review", "inspect"}
                or (preserve_merge_follow_up and next_action == "merge" and not merge_executed)
            )
        )
        preserved_worktree_paths: list[str] = []
        if preserve_worktrees and worktree_session is not None:
            assignments = getattr(worktree_session, "assignments", [])
            for assignment in list(assignments or []):
                path = cls._coerce_optional_text(getattr(assignment, "worktree_path", None))
                if path is not None:
                    preserved_worktree_paths.append(path)

        reentry_contract: Optional[Dict[str, Any]] = None
        merge_execution_contract: Optional[Dict[str, Any]] = None
        if preserve_worktrees and worktree_session is not None:
            if next_action == "merge" and not merge_executed:
                merge_execution_contract = cls._build_delegate_merge_execution_contract(
                    worktree_session=worktree_session,
                    merge_analysis=merge_analysis,
                    merge_orchestration=merge_orchestration,
                    merge_review_contract=merge_review_contract,
                    worker_return_contracts=worker_return_contracts,
                )
            else:
                retry_member_ids = cls._resolve_delegate_reentry_member_ids(
                    next_action=next_action,
                    validation_failed_members=validation_failed_members,
                    review_required_members=review_required_members,
                )
                retry_tasks_by_member = cls._build_delegate_reentry_retry_tasks(
                    next_action=next_action,
                    fix_validation_queue=fix_validation_queue,
                    review_queue=review_queue,
                )
                resume_member_context_overrides: Dict[str, Dict[str, Any]] = {}
                resume_worktree_paths: Dict[str, str] = {}
                for member_id in retry_member_ids:
                    assignment = worktree_session.assignment_for(member_id)
                    if assignment is None:
                        continue
                    override = cls._build_delegate_reentry_member_context_override(assignment)
                    if not override:
                        continue
                    resume_member_context_overrides[member_id] = override
                    path = cls._coerce_optional_text(override.get("worktree_path"))
                    if path is not None:
                        resume_worktree_paths[member_id] = path
                if resume_member_context_overrides:
                    reentry_contract = {
                        "mode": "delegate",
                        "next_action": next_action,
                        "retry_member_ids": retry_member_ids,
                        "resume_worktree_paths": resume_worktree_paths,
                        "retry_tasks_by_member": retry_tasks_by_member,
                        "resume_member_context_overrides": resume_member_context_overrides,
                        "context_overrides": {
                            "mode": "delegate",
                            "worktree_isolation": True,
                            "materialize_worktrees": False,
                            "cleanup_worktrees": False,
                        },
                    }

        approval_contract = cls._build_delegate_approval_contract(
            next_action=next_action,
            merge_review_contract=merge_review_contract,
            reentry_contract=reentry_contract,
            merge_execution_contract=merge_execution_contract,
            fix_validation_queue=fix_validation_queue,
            review_queue=review_queue,
            merge_execution=merge_execution,
        )
        if workspace_diagnostics_payload:
            approval_contract = dict(approval_contract)
            approval_contract["workspace_isolation_diagnostics"] = list(
                workspace_diagnostics_payload
            )
        follow_up_steps = cls._normalize_delegate_next_steps(approval_contract.get("next_steps"))

        contract = {
            "next_action": next_action,
            "preserve_worktrees": preserve_worktrees,
            "fix_validation_queue": fix_validation_queue,
            "review_queue": review_queue,
            "review_required_members": review_required_members,
            "validation_failed_members": validation_failed_members,
            "preserved_worktree_paths": preserved_worktree_paths,
            "approval_contract": approval_contract,
        }
        if workspace_diagnostics_payload:
            contract["workspace_isolation_diagnostics"] = list(workspace_diagnostics_payload)
        if follow_up_steps:
            contract["next_steps"] = follow_up_steps
            primary_step_id = cls._coerce_optional_text(follow_up_steps[0].get("step_id"))
            if primary_step_id is not None:
                contract["primary_step_id"] = primary_step_id
            step_requests = cls._build_delegate_follow_up_step_requests(
                next_steps=follow_up_steps,
                primary_step_id=primary_step_id,
            )
            if step_requests:
                contract["step_requests"] = step_requests
                if primary_step_id is not None and primary_step_id in step_requests:
                    contract["primary_step_request"] = copy.deepcopy(step_requests[primary_step_id])
                step_execution_requests = cls._build_delegate_follow_up_step_execution_requests(
                    next_steps=follow_up_steps,
                    step_requests=step_requests,
                )
                if step_execution_requests:
                    contract["step_execution_requests"] = step_execution_requests
                    if primary_step_id is not None and primary_step_id in step_execution_requests:
                        contract["primary_step_execution_request"] = copy.deepcopy(
                            step_execution_requests[primary_step_id]
                        )
            follow_up_suggestions = cls._build_delegate_follow_up_suggestions(follow_up_steps)
            if follow_up_suggestions:
                contract["follow_up_suggestions"] = follow_up_suggestions
        if reentry_contract is not None:
            contract["reentry_contract"] = reentry_contract
        if merge_execution_contract is not None:
            contract["merge_execution_contract"] = merge_execution_contract
        return contract

    @classmethod
    def _build_delegate_approval_contract(
        cls,
        *,
        next_action: str,
        merge_review_contract: Mapping[str, Any],
        reentry_contract: Optional[Mapping[str, Any]],
        merge_execution_contract: Optional[Mapping[str, Any]],
        fix_validation_queue: List[Dict[str, Any]],
        review_queue: List[Dict[str, Any]],
        merge_execution: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        merge_execution_payload = (
            dict(merge_execution or {}) if isinstance(merge_execution, Mapping) else {}
        )
        reentry_payload = (
            dict(reentry_contract or {}) if isinstance(reentry_contract, Mapping) else {}
        )
        retry_member_ids = cls._normalize_member_id_list(reentry_payload.get("retry_member_ids"))
        recommended_merge_order = cls._normalize_member_id_list(
            merge_review_contract.get("recommended_merge_order")
        )
        validation_target_ids = cls._normalize_member_id_list(
            item.get("member_id") for item in fix_validation_queue if isinstance(item, Mapping)
        )
        review_target_ids = cls._normalize_member_id_list(
            item.get("member_id") for item in review_queue if isinstance(item, Mapping)
        )
        merge_executed = bool(merge_execution_payload.get("executed"))
        if not merge_executed:
            merge_status = cls._coerce_optional_text(merge_execution_payload.get("status"))
            if merge_status is not None:
                merge_executed = merge_status.lower() in {
                    "success",
                    "succeeded",
                    "merged",
                    "completed",
                }
        recommended_mode = cls._coerce_optional_text(merge_review_contract.get("recommended_mode"))
        if recommended_mode is None:
            if merge_executed or bool(merge_review_contract.get("merge_execution_eligible")):
                recommended_mode = "auto_apply_safe"
            elif next_action in {"fix_validation", "review", "inspect"}:
                recommended_mode = "manual_review"

        if merge_executed:
            target_member_ids = recommended_merge_order or retry_member_ids
            return cls._finalize_delegate_approval_contract(
                {
                    "required": False,
                    "reason": "merge_executed",
                    "recommended_action": "merged",
                    "recommended_mode": recommended_mode,
                    "resume_ready": False,
                    "auto_retry_eligible": False,
                    "merge_executed": True,
                    "target_member_ids": target_member_ids,
                    "summary": cls._build_delegate_approval_summary(
                        "Merge orchestration already executed for",
                        target_member_ids=target_member_ids,
                    ),
                },
                reentry_payload=reentry_payload,
                merge_execution_contract=merge_execution_contract,
            )

        if next_action == "fix_validation":
            target_member_ids = retry_member_ids or validation_target_ids
            resume_ready = bool(reentry_payload) and bool(target_member_ids)
            return cls._finalize_delegate_approval_contract(
                {
                    "required": not resume_ready,
                    "reason": "validation_failed",
                    "recommended_action": "retry" if resume_ready else "approve_retry",
                    "recommended_mode": recommended_mode,
                    "resume_ready": resume_ready,
                    "auto_retry_eligible": resume_ready,
                    "merge_executed": False,
                    "target_member_ids": target_member_ids,
                    "summary": cls._build_delegate_approval_summary(
                        (
                            "Resume preserved worktrees to fix failing validation for"
                            if resume_ready
                            else "Approve a validation retry for"
                        ),
                        target_member_ids=target_member_ids,
                    ),
                },
                reentry_payload=reentry_payload,
                merge_execution_contract=merge_execution_contract,
            )

        if next_action == "review":
            target_member_ids = retry_member_ids or review_target_ids
            return cls._finalize_delegate_approval_contract(
                {
                    "required": True,
                    "reason": "review_required",
                    "recommended_action": "review_then_retry",
                    "recommended_mode": recommended_mode,
                    "resume_ready": bool(reentry_payload),
                    "auto_retry_eligible": False,
                    "merge_executed": False,
                    "target_member_ids": target_member_ids,
                    "summary": cls._build_delegate_approval_summary(
                        "Review merge risks before retrying preserved worktrees for",
                        target_member_ids=target_member_ids,
                    ),
                },
                reentry_payload=reentry_payload,
                merge_execution_contract=merge_execution_contract,
            )

        if next_action == "merge":
            target_member_ids = recommended_merge_order or retry_member_ids
            return cls._finalize_delegate_approval_contract(
                {
                    "required": True,
                    "reason": "merge_ready",
                    "recommended_action": "approve_merge",
                    "recommended_mode": recommended_mode,
                    "resume_ready": False,
                    "auto_retry_eligible": False,
                    "merge_executed": False,
                    "target_member_ids": target_member_ids,
                    "summary": cls._build_delegate_approval_summary(
                        "Review and approve merge execution for",
                        target_member_ids=target_member_ids,
                    ),
                },
                reentry_payload=reentry_payload,
                merge_execution_contract=merge_execution_contract,
            )

        target_member_ids = retry_member_ids or list(
            dict.fromkeys([*validation_target_ids, *review_target_ids])
        )
        return cls._finalize_delegate_approval_contract(
            {
                "required": True,
                "reason": "inspect_required",
                "recommended_action": "inspect_worktrees",
                "recommended_mode": recommended_mode,
                "resume_ready": bool(reentry_payload),
                "auto_retry_eligible": False,
                "merge_executed": False,
                "target_member_ids": target_member_ids,
                "summary": cls._build_delegate_approval_summary(
                    "Inspect preserved worktrees before retrying work for",
                    target_member_ids=target_member_ids,
                ),
            },
            reentry_payload=reentry_payload,
            merge_execution_contract=merge_execution_contract,
        )

    @classmethod
    def _finalize_delegate_approval_contract(
        cls,
        contract: Mapping[str, Any],
        *,
        reentry_payload: Mapping[str, Any],
        merge_execution_contract: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        approval_contract = dict(contract)
        target_member_ids = cls._normalize_member_id_list(
            approval_contract.get("target_member_ids")
        )
        resume_context = cls._build_delegate_approval_resume_context(
            reentry_payload,
            target_member_ids=target_member_ids,
        )
        if resume_context is not None:
            approval_contract["resume_context"] = resume_context
        task_briefs = cls._build_delegate_approval_task_briefs(
            reentry_payload,
            target_member_ids=target_member_ids,
        )
        if task_briefs:
            approval_contract["task_briefs_by_member"] = task_briefs
        if "next_steps" not in approval_contract:
            next_steps = cls._build_delegate_approval_next_steps(
                approval_contract,
                target_member_ids=target_member_ids,
                resume_context=resume_context,
                task_briefs=task_briefs,
                merge_execution_contract=merge_execution_contract,
            )
            if next_steps:
                approval_contract["next_steps"] = next_steps
        normalized_next_steps = cls._normalize_delegate_next_steps(
            approval_contract.get("next_steps")
        )
        if normalized_next_steps:
            approval_contract["next_steps"] = normalized_next_steps
        return approval_contract

    @classmethod
    def _build_delegate_approval_resume_context(
        cls,
        reentry_payload: Mapping[str, Any],
        *,
        target_member_ids: List[str],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(reentry_payload, Mapping) or not reentry_payload:
            return None
        normalized_payload = dict(reentry_payload)
        retry_member_ids = cls._normalize_member_id_list(normalized_payload.get("retry_member_ids"))
        resume_overrides = normalized_payload.get("resume_member_context_overrides")
        resume_paths = normalized_payload.get("resume_worktree_paths")
        has_resume_details = bool(retry_member_ids)
        if isinstance(resume_overrides, Mapping):
            has_resume_details = has_resume_details or bool(resume_overrides)
        if isinstance(resume_paths, Mapping):
            has_resume_details = has_resume_details or bool(resume_paths)
        if not has_resume_details:
            return None
        if not retry_member_ids and target_member_ids:
            normalized_payload["retry_member_ids"] = list(target_member_ids)
        return {
            "mode": "delegate",
            "delegate_reentry_contract": normalized_payload,
        }

    @classmethod
    def _build_delegate_approval_task_briefs(
        cls,
        reentry_payload: Mapping[str, Any],
        *,
        target_member_ids: List[str],
    ) -> Dict[str, str]:
        if not isinstance(reentry_payload, Mapping):
            return {}
        raw_briefs = reentry_payload.get("retry_tasks_by_member")
        if not isinstance(raw_briefs, Mapping):
            return {}
        prioritized_ids = cls._normalize_member_id_list(target_member_ids)
        if not prioritized_ids:
            prioritized_ids = cls._normalize_member_id_list(raw_briefs.keys())
        task_briefs: Dict[str, str] = {}
        for member_id in prioritized_ids:
            task_brief = cls._coerce_optional_text(raw_briefs.get(member_id))
            if task_brief is None:
                continue
            task_briefs[member_id] = task_brief
        return task_briefs

    @classmethod
    def _build_delegate_approval_summary(
        cls,
        prefix: str,
        *,
        target_member_ids: List[str],
    ) -> str:
        normalized_targets = cls._normalize_member_id_list(target_member_ids)
        if normalized_targets:
            return f"{prefix}: {', '.join(normalized_targets)}."
        return f"{prefix} the delegate worktree set."

    @classmethod
    def _build_delegate_approval_next_steps(
        cls,
        approval_contract: Mapping[str, Any],
        *,
        target_member_ids: List[str],
        resume_context: Optional[Mapping[str, Any]],
        task_briefs: Mapping[str, str],
        merge_execution_contract: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        recommended_action = cls._coerce_optional_text(approval_contract.get("recommended_action"))
        summary = cls._coerce_optional_text(approval_contract.get("summary"))
        requires_approval = bool(approval_contract.get("required", False))
        normalized_targets = cls._normalize_member_id_list(target_member_ids)

        def build_step(
            step: str,
            instruction: Optional[str],
            *,
            step_requires_approval: bool,
            include_resume: bool = False,
        ) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "step": step,
                "instruction": instruction
                or cls._build_delegate_approval_summary(
                    "Continue delegate follow-up for",
                    target_member_ids=normalized_targets,
                ),
                "target_member_ids": list(normalized_targets),
                "requires_approval": step_requires_approval,
            }
            if include_resume and resume_context:
                payload["resume_context"] = dict(resume_context)
            if include_resume and task_briefs:
                payload["task_briefs_by_member"] = dict(task_briefs)
            return payload

        if recommended_action == "merged" or bool(approval_contract.get("merge_executed", False)):
            return [
                build_step(
                    "status_merged",
                    summary,
                    step_requires_approval=False,
                )
            ]
        if recommended_action == "retry":
            return [
                build_step(
                    "resume_delegate_retry",
                    summary,
                    step_requires_approval=False,
                    include_resume=True,
                )
            ]
        if recommended_action == "approve_retry":
            return [
                build_step(
                    "approve_delegate_retry",
                    summary,
                    step_requires_approval=True,
                )
            ]
        if recommended_action == "review_then_retry":
            steps = [
                build_step(
                    "review_worktrees",
                    summary,
                    step_requires_approval=True,
                )
            ]
            if resume_context:
                steps.append(
                    build_step(
                        "resume_delegate_retry",
                        cls._build_delegate_approval_summary(
                            "Resume preserved worktrees after review for",
                            target_member_ids=normalized_targets,
                        ),
                        step_requires_approval=requires_approval,
                        include_resume=True,
                    )
                )
            return steps
        if recommended_action == "approve_merge":
            step = build_step(
                "approve_merge_execution",
                summary,
                step_requires_approval=True,
            )
            if merge_execution_contract:
                step["execution_context"] = {
                    "mode": "delegate",
                    "delegate_merge_contract": dict(merge_execution_contract),
                }
            return [step]
        if recommended_action == "inspect_worktrees":
            steps = [
                build_step(
                    "inspect_worktrees",
                    summary,
                    step_requires_approval=True,
                )
            ]
            if resume_context:
                steps.append(
                    build_step(
                        "resume_delegate_retry",
                        cls._build_delegate_approval_summary(
                            "Resume preserved worktrees after inspection for",
                            target_member_ids=normalized_targets,
                        ),
                        step_requires_approval=requires_approval,
                        include_resume=True,
                    )
                )
            return steps
        return []

    @classmethod
    def _resolve_delegate_reentry_member_ids(
        cls,
        *,
        next_action: str,
        validation_failed_members: List[str],
        review_required_members: List[str],
    ) -> List[str]:
        if next_action == "fix_validation":
            return list(validation_failed_members)
        if next_action == "review":
            return list(review_required_members)
        combined = list(dict.fromkeys([*validation_failed_members, *review_required_members]))
        return combined

    @classmethod
    def _build_delegate_reentry_retry_tasks(
        cls,
        *,
        next_action: str,
        fix_validation_queue: List[Dict[str, Any]],
        review_queue: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        retry_tasks: Dict[str, str] = {}
        for item in fix_validation_queue:
            if not isinstance(item, Mapping):
                continue
            member_id = cls._coerce_optional_text(item.get("member_id"))
            if member_id is None:
                continue
            command = cls._coerce_optional_text(item.get("validation_command"))
            summary = cls._coerce_optional_text(item.get("validation_summary"))
            changed_files = cls._normalize_path_list(item.get("changed_files"))
            parts = [f"Fix the failing validation run for {member_id}."]
            if command is not None:
                parts.append(f"Re-run `{command}`.")
            if summary is not None:
                parts.append(f"Last result: {summary}.")
            if changed_files:
                parts.append(f"Focus on: {', '.join(changed_files)}.")
            retry_tasks[member_id] = " ".join(parts)

        for item in review_queue:
            if not isinstance(item, Mapping):
                continue
            member_id = cls._coerce_optional_text(item.get("member_id"))
            if member_id is None or member_id in retry_tasks:
                continue
            risk_level = cls._coerce_optional_text(item.get("merge_risk_level")) or "low"
            reasons = cls._normalize_path_list(item.get("merge_risk_reasons"))
            changed_files = cls._normalize_path_list(item.get("changed_files"))
            task_summary = cls._coerce_optional_text(item.get("task_summary"))
            parts = [f"Review the pending merge risk for {member_id} ({risk_level})."]
            if reasons:
                parts.append(f"Address: {', '.join(reasons)}.")
            if changed_files:
                parts.append(f"Inspect: {', '.join(changed_files)}.")
            if task_summary is not None:
                parts.append(f"Prior output: {task_summary}.")
            retry_tasks[member_id] = " ".join(parts)

        if not retry_tasks and next_action == "inspect":
            for item in review_queue:
                if not isinstance(item, Mapping):
                    continue
                member_id = cls._coerce_optional_text(item.get("member_id"))
                if member_id is None:
                    continue
                retry_tasks[member_id] = f"Inspect the preserved worktree state for {member_id}."
        return retry_tasks

    @classmethod
    def _build_delegate_reentry_member_context_override(
        cls,
        assignment: Any,
    ) -> Dict[str, Any]:
        override = {}
        to_context_overrides = getattr(assignment, "to_context_overrides", None)
        if callable(to_context_overrides):
            raw_override = to_context_overrides()
            if isinstance(raw_override, Mapping):
                override.update(dict(raw_override))

        worktree_path = cls._coerce_optional_text(getattr(assignment, "worktree_path", None))
        branch_name = cls._coerce_optional_text(getattr(assignment, "branch_name", None))
        if worktree_path is not None:
            override.setdefault("workspace_root", worktree_path)
            override["worktree_path"] = worktree_path
        if branch_name is not None:
            override["branch_name"] = branch_name

        assignment_payload: Dict[str, Any] = {}
        to_dict = getattr(assignment, "to_dict", None)
        if callable(to_dict):
            raw_payload = to_dict()
            if isinstance(raw_payload, Mapping):
                assignment_payload.update(dict(raw_payload))
        if not assignment_payload:
            member_id = cls._coerce_optional_text(getattr(assignment, "member_id", None))
            if member_id is not None:
                assignment_payload["member_id"] = member_id
            if branch_name is not None:
                assignment_payload["branch_name"] = branch_name
            if worktree_path is not None:
                assignment_payload["worktree_path"] = worktree_path
        if assignment_payload:
            override["worktree_assignment"] = assignment_payload

        return override

    @classmethod
    def _rebuild_worktree_assignment(
        cls,
        payload: Mapping[str, Any],
    ) -> Optional[WorktreeAssignment]:
        member_id = cls._coerce_optional_text(payload.get("member_id"))
        branch_name = cls._coerce_optional_text(payload.get("branch_name"))
        worktree_name = cls._coerce_optional_text(payload.get("worktree_name"))
        worktree_path = cls._coerce_optional_text(payload.get("worktree_path"))
        if None in (member_id, branch_name, worktree_name, worktree_path):
            return None
        return WorktreeAssignment(
            member_id=member_id,
            branch_name=branch_name,
            worktree_name=worktree_name,
            worktree_path=worktree_path,
            claimed_paths=tuple(cls._normalize_path_list(payload.get("claimed_paths"))),
            readonly_paths=tuple(cls._normalize_path_list(payload.get("readonly_paths"))),
            merge_priority=int(payload.get("merge_priority") or 0),
            metadata=(
                dict(payload.get("metadata") or {})
                if isinstance(payload.get("metadata"), Mapping)
                else {}
            ),
        )

    @classmethod
    def _rebuild_worktree_plan(
        cls,
        payload: Mapping[str, Any],
    ) -> Optional[WorktreeExecutionPlan]:
        team_name = cls._coerce_optional_text(payload.get("team_name"))
        repo_root = cls._coerce_optional_text(payload.get("repo_root"))
        parent_dir = cls._coerce_optional_text(payload.get("parent_dir"))
        base_ref = cls._coerce_optional_text(payload.get("base_ref"))
        branch_prefix = cls._coerce_optional_text(payload.get("branch_prefix"))
        formation_name = (
            cls._coerce_optional_text(payload.get("formation")) or TeamFormation.SEQUENTIAL.value
        )
        if None in (team_name, repo_root, parent_dir, base_ref, branch_prefix):
            return None
        try:
            formation = TeamFormation(formation_name)
        except ValueError:
            formation = TeamFormation.SEQUENTIAL
        assignments = tuple(
            assignment
            for assignment in (
                cls._rebuild_worktree_assignment(item)
                for item in list(payload.get("assignments") or [])
                if isinstance(item, Mapping)
            )
            if assignment is not None
        )
        return WorktreeExecutionPlan(
            team_name=team_name,
            repo_root=repo_root,
            parent_dir=parent_dir,
            base_ref=base_ref,
            branch_prefix=branch_prefix,
            formation=formation,
            assignments=assignments,
            merge_order=tuple(cls._normalize_member_id_list(payload.get("merge_order"))),
            shared_readonly_paths=tuple(
                cls._normalize_path_list(payload.get("shared_readonly_paths"))
            ),
            rationale=cls._coerce_optional_text(payload.get("rationale")),
            metadata=(
                dict(payload.get("metadata") or {})
                if isinstance(payload.get("metadata"), Mapping)
                else {}
            ),
        )

    @classmethod
    def _rebuild_materialized_worktree_assignment(
        cls,
        payload: Mapping[str, Any],
    ) -> Optional[MaterializedWorktreeAssignment]:
        assignment = cls._rebuild_worktree_assignment(payload)
        if assignment is None:
            return None
        runtime_metadata = payload.get("runtime_metadata")
        return MaterializedWorktreeAssignment(
            assignment=assignment,
            materialized=bool(payload.get("materialized", False)),
            cleanup_required=bool(payload.get("cleanup_required", False)),
            metadata=(
                dict(runtime_metadata or {}) if isinstance(runtime_metadata, Mapping) else {}
            ),
        )

    @classmethod
    def _rebuild_worktree_session(
        cls,
        payload: Mapping[str, Any],
    ) -> Optional[WorktreeMaterializationSession]:
        plan = cls._rebuild_worktree_plan(
            dict(payload.get("plan") or {}) if isinstance(payload.get("plan"), Mapping) else {}
        )
        if plan is None:
            return None
        assignments = tuple(
            assignment
            for assignment in (
                cls._rebuild_materialized_worktree_assignment(item)
                for item in list(payload.get("assignments") or [])
                if isinstance(item, Mapping)
            )
            if assignment is not None
        )
        metadata = payload.get("metadata")
        return WorktreeMaterializationSession(
            plan=plan,
            assignments=assignments,
            materialized=bool(payload.get("materialized", False)),
            dry_run=bool(payload.get("dry_run", False)),
            metadata=dict(metadata or {}) if isinstance(metadata, Mapping) else {},
        )

    @classmethod
    def _build_delegate_merge_execution_contract(
        cls,
        *,
        worktree_session: WorktreeMaterializationSession,
        merge_analysis: Optional[Mapping[str, Any]],
        merge_orchestration: Optional[Mapping[str, Any]],
        merge_review_contract: Mapping[str, Any],
        worker_return_contracts: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "mode": "delegate",
            "next_action": "merge",
            "worktree_session": worktree_session.to_dict(),
            "merge_analysis": dict(merge_analysis or {}),
            "merge_orchestration": dict(merge_orchestration or {}),
            "merge_review_contract": dict(merge_review_contract or {}),
            "worker_return_contracts": {
                member_id: dict(contract)
                for member_id, contract in worker_return_contracts.items()
                if isinstance(contract, Mapping)
            },
            "context_overrides": {
                "mode": "delegate",
                "worktree_isolation": True,
                "materialize_worktrees": False,
                "cleanup_worktrees": True,
            },
        }

    def _execute_delegate_merge_contract(
        self,
        task: str,
        context: Dict[str, Any],
        *,
        formation: TeamFormation,
        delegate_merge_contract: Mapping[str, Any],
    ) -> Dict[str, Any]:
        effective_context = self._apply_delegate_merge_context(
            context,
            delegate_merge_contract=delegate_merge_contract,
        )
        session_payload = (
            dict(delegate_merge_contract.get("worktree_session") or {})
            if isinstance(delegate_merge_contract.get("worktree_session"), Mapping)
            else {}
        )
        worktree_session = self._rebuild_worktree_session(session_payload)
        if worktree_session is None:
            return {
                "success": False,
                "error": "Invalid delegate merge contract",
                "member_results": {},
                "final_output": "",
                "formation": formation.value,
            }

        merge_analysis = (
            dict(delegate_merge_contract.get("merge_analysis") or {})
            if isinstance(delegate_merge_contract.get("merge_analysis"), Mapping)
            else {}
        )
        merge_orchestration = (
            dict(delegate_merge_contract.get("merge_orchestration") or {})
            if isinstance(delegate_merge_contract.get("merge_orchestration"), Mapping)
            else {}
        )
        merge_review_contract = (
            dict(delegate_merge_contract.get("merge_review_contract") or {})
            if isinstance(delegate_merge_contract.get("merge_review_contract"), Mapping)
            else {}
        )
        worker_return_contracts = {
            member_id: dict(contract)
            for member_id, contract in (
                dict(delegate_merge_contract.get("worker_return_contracts") or {})
                if isinstance(delegate_merge_contract.get("worker_return_contracts"), Mapping)
                else {}
            ).items()
            if isinstance(contract, Mapping)
        }
        merge_execution = self._execute_merge_orchestration(
            worktree_session,
            merge_analysis=merge_analysis or None,
            context=effective_context,
        ) or {
            "status": "blocked",
            "executed": False,
            "blocked_reason": "merge_execution_unavailable",
        }

        result: Dict[str, Any] = {
            "success": bool(merge_execution.get("executed", False)),
            "member_results": {},
            "final_output": (
                "Approved merge orchestration executed."
                if merge_execution.get("executed", False)
                else "Approved merge orchestration did not execute."
            ),
            "formation": formation.value,
            "total_tool_calls": 0,
            "communication_log": list(self._active_message_history()),
            "shared_context": dict(self._active_shared_context()),
            "worktree_session": worktree_session.to_dict(),
            "merge_execution": merge_execution,
        }
        if merge_analysis:
            result["merge_analysis"] = dict(merge_analysis)
        if merge_orchestration:
            result["merge_orchestration"] = dict(merge_orchestration)
        else:
            built_orchestration = self._build_merge_orchestration(
                worktree_session,
                merge_analysis=merge_analysis or None,
            )
            if built_orchestration is not None:
                merge_orchestration = built_orchestration
                result["merge_orchestration"] = built_orchestration
        if merge_review_contract:
            result["merge_review_contract"] = dict(merge_review_contract)
        if worker_return_contracts:
            result["worker_return_contracts"] = dict(worker_return_contracts)
        if worker_return_contracts and merge_review_contract:
            delegate_follow_up_contract = self._build_delegate_follow_up_contract(
                worker_return_contracts,
                merge_review_contract=merge_review_contract,
                worktree_session=worktree_session,
                merge_execution=merge_execution,
                merge_analysis=merge_analysis,
                merge_orchestration=merge_orchestration,
                preserve_merge_follow_up=self._resolve_context_mode(effective_context)
                == "delegate",
            )
            if delegate_follow_up_contract:
                result["delegate_follow_up_contract"] = delegate_follow_up_contract
                self._attach_delegate_follow_up_suggestions(
                    result,
                    delegate_follow_up_contract,
                )

        if self._should_cleanup_worktrees(effective_context, result_dict=result):
            cleanup_summary = self._cleanup_worktree_session(worktree_session)
        else:
            cleanup_summary = self._build_preserved_worktree_cleanup_summary(
                worktree_session,
                reason="preserved_for_follow_up",
            )
        result["worktree_cleanup"] = cleanup_summary
        return result

    def _inject_worktree_changed_files(
        self,
        member_results: Dict[str, MemberResult],
        *,
        worktree_session: Optional[WorktreeMaterializationSession],
    ) -> None:
        self._workspace_isolation.inject_changed_files(
            member_results,
            worktree_session=worktree_session,
        )

    def _build_merge_orchestration(
        self,
        worktree_session: WorktreeMaterializationSession,
        *,
        merge_analysis: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        return self._workspace_isolation.build_merge_orchestration(
            worktree_session,
            merge_analysis=merge_analysis,
        )

    def _should_execute_merge_orchestration(
        self,
        context: Dict[str, Any],
        *,
        merge_orchestration: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        return self._workspace_isolation.should_execute_merge(
            context,
            merge_orchestration=merge_orchestration,
        )

    def _execute_merge_orchestration(
        self,
        worktree_session: WorktreeMaterializationSession,
        *,
        merge_analysis: Optional[Dict[str, Any]] = None,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        return self._workspace_isolation.execute_merge(
            worktree_session,
            merge_analysis=merge_analysis,
            context=context,
        )

    def _should_cleanup_worktrees(
        self,
        context: Dict[str, Any],
        *,
        result_dict: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self._workspace_isolation.should_cleanup(context, result_dict=result_dict)

    def _build_preserved_worktree_cleanup_summary(
        self,
        worktree_session: WorktreeMaterializationSession,
        *,
        reason: str,
    ) -> Dict[str, Any]:
        return self._workspace_isolation.preserved_cleanup_summary(
            worktree_session,
            reason=reason,
        )

    def _cleanup_worktree_session(
        self,
        worktree_session: WorktreeMaterializationSession,
    ) -> Dict[str, Any]:
        return self._workspace_isolation.cleanup(worktree_session)

    def _resolve_effective_formation(
        self,
        context: Dict[str, Any],
        *,
        default_formation: Optional[TeamFormation] = None,
    ) -> TeamFormation:
        """Resolve formation override hints without mutating the default formation."""
        fallback_formation = default_formation or self._active_formation()
        raw_hint = context.get("formation_hint") or context.get("topology_formation_hint")
        if not raw_hint:
            return fallback_formation

        normalized = str(raw_hint).strip().lower()
        for formation in TeamFormation:
            if formation.value == normalized:
                return formation
        return fallback_formation

    @staticmethod
    def _extract_max_workers(
        context: Dict[str, Any],
        shared_state: Dict[str, Any],
    ) -> Optional[int]:
        """Extract max worker hint from the execution context."""
        raw_value = context.get("max_workers", shared_state.get("max_workers"))
        if raw_value is None:
            return None
        try:
            max_workers = int(raw_value)
        except (TypeError, ValueError):
            return None
        return max_workers if max_workers > 0 else None

    def _limit_execution_members(
        self,
        members: List["ITeamMember"],
        formation: TeamFormation,
        max_workers: Optional[int],
        *,
        supervisor: Optional["ITeamMember"] = None,
    ) -> List["ITeamMember"]:
        """Limit members for a single execution while preserving a supervisor."""
        if max_workers is None or max_workers >= len(members):
            return list(members)

        if formation == TeamFormation.HIERARCHICAL and supervisor in members:
            selected: List["ITeamMember"] = [supervisor]
            for member in members:
                if member is supervisor:
                    continue
                selected.append(member)
                if len(selected) >= max_workers:
                    break
            return selected

        return list(members[:max_workers])

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear(self) -> "UnifiedTeamCoordinator":
        """Clear all members and reset state.

        Returns:
            Self for fluent chaining
        """
        self._members.clear()
        self._manager = None
        self._message_history.clear()
        self._shared_context.clear()
        return self

    @property
    def members(self) -> List["ITeamMember"]:
        """Get list of team members."""
        return list(self._members)

    @property
    def formation(self) -> TeamFormation:
        """Get current formation."""
        return self._formation

    @property
    def manager(self) -> Optional["ITeamMember"]:
        """Compatibility alias for supervisor."""
        return self.supervisor

    @property
    def supervisor(self) -> Optional["ITeamMember"]:
        """Get team supervisor (for hierarchical formation)."""
        return self._manager

    # =========================================================================
    # Parameterised execution & TeamConfig adapter
    # =========================================================================

    async def _execute_with(
        self,
        task: str,
        context: Dict[str, Any],
        *,
        formation: TeamFormation,
        members: List["ITeamMember"],
        supervisor: Optional["ITeamMember"] = None,
        persist_execution_state: bool = False,
    ) -> Dict[str, Any]:
        """Execute a team run with per-call members/formation overrides.

        The coordinator's defaults remain unchanged. Per-call state lives in a
        task-local ``ContextVar`` so concurrent invocations can execute against
        the same coordinator instance without serialisation or instance swaps.
        """
        delegate_next_step = self._extract_delegate_next_step(context)
        effective_context = self._apply_delegate_next_step_context(
            context,
            delegate_next_step=delegate_next_step,
        )
        execution_members = list(members)
        delegate_merge_contract = self._extract_delegate_merge_contract(effective_context)
        if not execution_members and not delegate_merge_contract:
            return {
                "success": False,
                "error": "No team members added",
                "member_results": {},
                "final_output": "",
                "formation": formation.value,
            }

        execution_state = _CoordinatorExecutionState(
            members=execution_members,
            formation=formation,
            supervisor=supervisor,
            shared_context=copy.deepcopy(dict(effective_context)),
        )
        token = self._execution_state.set(execution_state)
        try:
            start_time = time.time()
            effective_formation = self._resolve_effective_formation(
                effective_context,
                default_formation=formation,
            )

            self._emit_team_event(
                "started",
                {
                    "task": task,
                    "formation": effective_formation.value,
                    "member_count": len(execution_members),
                },
            )
            try:
                if delegate_merge_contract:
                    result = self._execute_delegate_merge_contract(
                        task,
                        effective_context,
                        formation=effective_formation,
                        delegate_merge_contract=delegate_merge_contract,
                    )
                else:
                    result = await self._execute_formation(
                        task,
                        effective_context,
                        formation_override=effective_formation,
                    )

                duration = time.time() - start_time
                failed_count = sum(
                    1 for r in result.get("member_results", {}).values() if not r.success
                )
                quality = self._compute_quality_score(
                    success=result.get("success", False),
                    member_count=len(execution_members),
                    total_tool_calls=result.get("total_tool_calls", 0),
                    duration_seconds=duration,
                    failed_members=failed_count,
                )

                self._record_team_rl_outcome(
                    team_name=context.get("team_name", "UnifiedTeam"),
                    formation=effective_formation.value,
                    success=result.get("success", False),
                    quality_score=quality,
                    metadata={
                        "member_count": len(execution_members),
                        "duration": duration,
                        "delegate_selected_step": effective_context.get("delegate_selected_step"),
                    },
                )

                self._emit_team_event(
                    "completed",
                    {
                        "success": result.get("success", False),
                        "duration": duration,
                    },
                )
                if persist_execution_state:
                    self._shared_context = dict(
                        result.get("shared_context", execution_state.shared_context)
                    )
                    self._message_history = list(execution_state.message_history)
                return result

            except Exception as e:
                self._emit_team_event("error", {"error": str(e)})
                logger.error(f"Team execution failed: {e}")
                if persist_execution_state:
                    self._shared_context = dict(execution_state.shared_context)
                    self._message_history = list(execution_state.message_history)
                return {
                    "success": False,
                    "error": str(e),
                    "member_results": {},
                    "final_output": "",
                    "formation": effective_formation.value,
                }
        finally:
            self._execution_state.reset(token)

    def _populate_context_agents(
        self, strategy: "BaseFormationStrategy", members: List["ITeamMember"]
    ) -> Dict[str, "ContextAgent"]:
        """Bind members to the role-named agents a context-driven formation needs.

        Returns ``{}`` unless ``strategy.consumes_context_agents()`` is True.
        Each role from ``strategy.get_required_roles()`` is bound to the member
        whose ``formation_role`` matches, falling back to positional order.
        Returns fewer entries than required roles when members are short — the
        formation's own validation then reports the shortfall.
        """
        if not strategy.consumes_context_agents():
            return {}
        roles = strategy.get_required_roles() or []
        if not roles:
            return {}

        agents: Dict[str, "ContextAgent"] = {}
        remaining = list(members)

        # First pass: explicit binding by declared formation_role.
        for role in roles:
            for idx, member in enumerate(remaining):
                if _member_formation_role(member) == role:
                    agents[role] = _MemberContextAgent(member)
                    remaining.pop(idx)
                    break

        # Second pass: positional fallback for any unbound role.
        for role in roles:
            if role in agents or not remaining:
                continue
            agents[role] = _MemberContextAgent(remaining.pop(0))

        return agents

    def _adapt_team_members(self, members: List[Any]) -> List["ITeamMember"]:
        """Adapt ``TeamMember`` dataclasses to ``ITeamMember`` adapters.

        Uses ``self._orchestrator`` to build a ``SubAgentOrchestrator`` that
        actually executes each member. Raises ``ValueError`` if no
        orchestrator is configured — callers can bypass this requirement by
        passing pre-built ``ITeamMember`` instances via the ``members=``
        parameter on ``execute_team_config``.
        """
        if self._orchestrator is None:
            raise ValueError(
                "UnifiedTeamCoordinator requires an orchestrator to adapt "
                "TeamMember dataclasses into ITeamMember instances. Either "
                "construct the coordinator with an orchestrator, or pass "
                "pre-built members via execute_team_config(config, members=...)."
            )

        from victor.teams.types import TeamParticipant
        from victor.agent.subagents.orchestrator import SubAgentOrchestrator

        sub_orchestrator = SubAgentOrchestrator(self._orchestrator)

        def _make_executor(team_member):
            async def executor(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
                spawn_result = await sub_orchestrator.spawn(
                    role=team_member.role,
                    task=task,
                    tool_budget=team_member.tool_budget,
                    allowed_tools=team_member.allowed_tools,
                    provider=getattr(team_member, "provider", None),
                    model=getattr(team_member, "model", None),
                    temperature=getattr(team_member, "temperature", None),
                    reasoning_effort=getattr(team_member, "reasoning_effort", None),
                )
                return {
                    "success": getattr(spawn_result, "success", False),
                    "output": getattr(spawn_result, "summary", "") or "",
                    "error": getattr(spawn_result, "error", None),
                    "tool_calls_used": getattr(spawn_result, "tool_calls_used", 0),
                    "duration_seconds": getattr(spawn_result, "duration_seconds", 0.0),
                }

            return executor

        return [TeamParticipant(member=m, executor=_make_executor(m)) for m in members]

    def _dict_result_to_team_result(
        self,
        result: Dict[str, Any],
        *,
        formation: TeamFormation,
    ) -> TeamResult:
        """Convert ``execute_task``'s dict result into a ``TeamResult``."""
        return TeamResult(
            success=bool(result.get("success", False)),
            final_output=str(result.get("final_output", "")),
            member_results=dict(result.get("member_results", {})),
            formation=formation,
            total_tool_calls=int(result.get("total_tool_calls", 0)),
            total_duration=float(result.get("total_duration", 0.0)),
            communication_log=list(result.get("communication_log", [])),
            shared_context=dict(result.get("shared_context", {})),
            consensus_achieved=result.get("consensus_achieved"),
            consensus_rounds=result.get("consensus_rounds"),
            error=result.get("error"),
        )

    async def execute_team_config(
        self,
        config: Any,
        *,
        members: Optional[List["ITeamMember"]] = None,
    ) -> TeamResult:
        """Execute a ``TeamConfig`` and return a ``TeamResult``.

        This is the entry point used by the declarative team-step adapter in
        ``victor.framework.workflows.nodes`` and any other caller that
        already has a ``TeamConfig`` in hand.

        Args:
            config: A ``TeamConfig`` describing the run (goal, formation,
                shared context, members).
            members: Optional pre-built ``ITeamMember`` instances. If
                omitted, the coordinator builds adapters from
                ``config.members`` using its orchestrator (raises
                ``ValueError`` if no orchestrator is configured).

        Returns:
            A ``TeamResult`` matching the formation in the config.
        """
        if members is None:
            members = self._adapt_team_members(list(config.members))

        result = await self._execute_with(
            task=config.goal,
            context=dict(config.shared_context or {}),
            formation=config.formation,
            members=members,
        )
        return self._dict_result_to_team_result(result, formation=config.formation)

    # =========================================================================
    # StateGraph Integration
    # =========================================================================

    def with_state_graph_config(self, config: StateGraphNodeConfig) -> "UnifiedTeamCoordinator":
        """Configure how this coordinator behaves when used as a StateGraph node.

        ``__call__`` reads/writes the input/output keys named in ``config``.
        The default config preserves historical behaviour (``task`` / ``query``
        in, ``result`` / ``team_output`` / ``error`` out).

        Returns ``self`` for fluent chaining.
        """
        self._state_graph_config = config
        return self

    @staticmethod
    def _classify_state(state: Any) -> str:
        """Classify the incoming graph state.

        Returns one of ``"cow"``, ``"pydantic"``, or ``"dict"``. The
        coordinator branches on this when reading task/context and when
        writing back the team result, so the integration works against any
        of the state schemas StateGraph supports.
        """
        # Lazy imports to keep victor/teams from depending on pydantic at
        # module load time.
        from victor.framework.graph import CopyOnWriteState

        if isinstance(state, CopyOnWriteState):
            return "cow"

        try:
            from pydantic import BaseModel
        except ImportError:  # pragma: no cover - pydantic is a hard dep here
            return "dict"
        if isinstance(state, BaseModel):
            return "pydantic"
        return "dict"

    @staticmethod
    def _read_state(state: Any, kind: str, key: str, default: Any = None) -> Any:
        if kind == "pydantic":
            return getattr(state, key, default)
        if kind == "cow":
            return state.get(key, default)
        return state.get(key, default)

    @staticmethod
    def _build_call_context(state: Any, kind: str, *, exclude: set) -> Dict[str, Any]:
        if kind == "pydantic":
            dump = state.model_dump()
            return {k: v for k, v in dump.items() if k not in exclude}
        if kind == "cow":
            return {k: state[k] for k in state.keys() if k not in exclude}
        return {k: v for k, v in state.items() if k not in exclude}

    @staticmethod
    def _apply_updates(state: Any, kind: str, updates: Dict[str, Any]) -> Any:
        """Write ``updates`` back into the original state container.

        - ``dict``: returns a new dict (caller's input is not mutated).
        - ``pydantic``: returns ``state.model_copy(update=updates)``. If the
          model rejects unknown fields, raises ``ValueError`` naming the
          offending key so users know to either widen ``model_config`` with
          ``extra='allow'`` or remap keys via ``StateGraphNodeConfig``.
        - ``cow``: assigns each key via ``__setitem__`` and returns the
          same wrapper, so the StateGraph executor sees the mutation.
        """
        if kind == "pydantic":
            # ``model_copy(update=...)`` in Pydantic v2 does not run validation,
            # so a strict (extra='forbid') model would silently drop unknown
            # keys. Detect that case up front and raise a clear error so users
            # know to either remap keys or widen ``model_config``.
            extra_policy = getattr(state.model_config, "get", lambda *_: None)("extra")
            if extra_policy is None and isinstance(state.model_config, dict):
                extra_policy = state.model_config.get("extra")
            declared = set(getattr(type(state), "model_fields", {}).keys())
            unknown = [k for k in updates if k not in declared]
            if unknown and extra_policy not in ("allow",):
                hint = (
                    "Either remap the keys via StateGraphNodeConfig "
                    "(e.g. result_key='context') or set "
                    "model_config = {'extra': 'allow'} on your state model."
                )
                raise ValueError(
                    f"Cannot write team result into Pydantic state of type "
                    f"{type(state).__name__}: fields {unknown} are not declared "
                    f"and the model does not allow extras. {hint}"
                )
            try:
                return state.model_copy(update=updates)
            except Exception as exc:  # pydantic.ValidationError on assignment
                raise ValueError(
                    f"Failed to write team result into Pydantic state of type "
                    f"{type(state).__name__}: {exc}. Consider remapping keys "
                    f"via StateGraphNodeConfig."
                ) from exc
        if kind == "cow":
            for k, v in updates.items():
                state[k] = v
            return state
        # dict
        new_state = dict(state)
        new_state.update(updates)
        return new_state

    async def __call__(self, state: Any) -> Any:
        """Execute as a StateGraph node.

        This makes the coordinator directly usable in StateGraph:
            graph.add_node("team", coordinator)

        The coordinator reads the task from ``state[task_key]`` (with a
        fallback to ``state[query_key]``), executes the team with the
        configured formation, and returns a *new* state dict with the
        results. The caller's input dict is never mutated.

        Use ``with_state_graph_config(...)`` to map the node onto a graph
        schema with different key names. For declarative workflow YAML with
        timeout/retry/merge-strategy configuration, use ``TeamStep`` from
        ``victor.framework.workflows.nodes`` instead — ``__call__`` is the
        lean programmatic path.

        Args:
            state: Current graph state. The task is read from
                ``state[config.task_key]`` (default ``"task"``), falling
                back to ``state[config.query_key]`` (default ``"query"``).
                All other keys are passed through as execution context.

        Returns:
            New state dict with:
                - ``config.result_key`` (default ``"result"``): final output
                  string when execution succeeded.
                - ``config.output_key`` (default ``"team_output"``): full
                  team execution result dict.
                - ``config.error_key`` (default ``"error"``): error message
                  when execution failed.

        Example::

            from victor.framework import StateGraph
            from victor.teams import (
                UnifiedTeamCoordinator,
                TeamFormation,
                StateGraphNodeConfig,
            )

            coordinator = UnifiedTeamCoordinator(orchestrator)
            coordinator.set_formation(TeamFormation.PARALLEL)
            coordinator.add_member(agent1).add_member(agent2)
            coordinator.with_state_graph_config(
                StateGraphNodeConfig(task_key="instruction", result_key="answer")
            )

            graph = StateGraph(AgentState)
            graph.add_node("research_team", coordinator)
        """
        config = self._state_graph_config
        kind = self._classify_state(state)

        # Extract task using configured keys (task takes precedence over query).
        # ``None`` from a missing pydantic optional is treated like missing.
        task = self._read_state(state, kind, config.task_key, default=None)
        if not task:
            task = self._read_state(state, kind, config.query_key, default="")

        # Build context excluding task/query keys.
        context = self._build_call_context(state, kind, exclude={config.task_key, config.query_key})
        strategy_state = state.get_state() if kind == "cow" else state
        if callable(config.formation_strategy):
            try:
                selected_formation = config.formation_strategy(strategy_state)
            except Exception as exc:
                logger.debug("Formation strategy failed; keeping default formation: %s", exc)
            else:
                if asyncio.iscoroutine(selected_formation):
                    close_coro = getattr(selected_formation, "close", None)
                    if callable(close_coro):
                        close_coro()
                    raise TypeError(
                        "StateGraphNodeConfig.formation_strategy must return a "
                        "TeamFormation synchronously; async strategies are not supported."
                    )
                if isinstance(selected_formation, TeamFormation):
                    context["formation_hint"] = selected_formation.value
                elif selected_formation is not None:
                    context["formation_hint"] = str(selected_formation)

        # Execute team task.
        result = await self.execute_task(task, context)

        # Compose updates and apply via the type-aware writer. dict inputs
        # are returned as a new dict (caller's state is not mutated), Pydantic
        # inputs are returned via model_copy, CoW wrappers are mutated via
        # __setitem__ so the executor sees the change.
        updates: Dict[str, Any] = {}
        if result.get("success"):
            updates[config.result_key] = result.get("final_output", "")
            updates[config.output_key] = result
        else:
            updates[config.error_key] = result.get("error", "Unknown error")
            updates[config.output_key] = result

        return self._apply_updates(state, kind, updates)
