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
import json
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

from victor.coordination.formations.base import (
    BaseFormationStrategy,
    MemberCheckpointError,
    TeamContext,
)
from victor.framework.graph_checkpoint import CheckpointerProtocol, WorkflowCheckpoint
from victor.agent.approval_binding import require_complete_batch_evidence
from victor.framework.member_event_sink import MemberEvent, MemberEventSink, current_member_sink
from victor.coordination.formations import create_formation_registry
from victor.teams.mixins.observability import ObservabilityMixin
from victor.teams.mixins.rl import RLMixin
from victor.teams.types import (
    AgentMessage,
    MemberResult,
    MessageType,
    TeamFormation,
    TeamParticipant,
    TeamResult,
    normalize_supervisor_context,
)
from victor.teams.workspace_isolation import (
    WorkspaceIsolationService,
    WorkspaceMaterializationOutcome,
)
from victor.teams import delegate_contracts
from victor.teams.worktree_runtime import (
    MaterializedWorktreeAssignment,
    WorktreeAssignment,
    WorktreeExecutionPlan,
    WorktreeMaterializationSession,
)

if TYPE_CHECKING:
    from victor.protocols.team import ITeamMember

logger = logging.getLogger(__name__)


def _require_uncancelled_member_task() -> None:
    """A handled cancellation request is not permission to admit work or publish a pause."""
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError("Member execution cancellation requested")


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
    team_goal: str = ""
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
        _supervisor: Supervisor member for HIERARCHICAL formation
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
        self._supervisor: Optional[ITeamMember] = None
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
        self._formations: Dict[TeamFormation, BaseFormationStrategy] = create_formation_registry()

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
        self._supervisor = supervisor
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
            supervisor=self._supervisor,
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
        task, context = delegate_contracts.normalize_delegate_execution_request(request)
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
        request = delegate_contracts.resolve_delegate_execution_request_from_follow_up_contract(
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
        return self._supervisor

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
        _require_uncancelled_member_task()
        try:
            checkpoints = await checkpointer.list(thread_id)
        except Exception as exc:
            _require_uncancelled_member_task()
            raise MemberCheckpointError("load") from exc
        _require_uncancelled_member_task()
        relevant = [c for c in checkpoints if c.metadata.get("team_node_id") == team_node_id]
        if not relevant:
            return None
        latest = max(relevant, key=lambda c: c.timestamp)
        state = latest.state or {}
        require_complete_batch_evidence(state.get("approval_request"))
        for pending in state.get("awaiting_approvals") or []:
            require_complete_batch_evidence(pending.get("approval_request"))
        member_results = [MemberResult.from_dict(r) for r in state.get("member_results") or []]
        return {
            "member_ids": state.get("completed_member_ids")
            or [r.member_id for r in member_results],
            "member_results": member_results,
            "shared_state": state.get("shared_state") or {},
            "last_output": state.get("last_output"),
            "last_agent_id": state.get("last_agent_id"),
        }

    @staticmethod
    async def _save_member_checkpoint(
        checkpointer: CheckpointerProtocol, checkpoint: WorkflowCheckpoint, operation: str
    ) -> None:
        """Require a save acknowledgement; a lost acknowledgement never permits retry."""
        _require_uncancelled_member_task()
        try:
            await checkpointer.save(checkpoint)
        except Exception as exc:
            _require_uncancelled_member_task()
            raise MemberCheckpointError(operation) from exc
        _require_uncancelled_member_task()

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
            await self._save_member_checkpoint(checkpointer, checkpoint, "save_member")

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
            await self._save_member_checkpoint(checkpointer, checkpoint, "save_pause")

        return _hook

    def _make_member_batch_pause_hook(
        self,
        checkpointer: CheckpointerProtocol,
        thread_id: str,
        team_node_id: str,
        formation: str,
    ) -> Callable[..., Any]:
        """Build the concurrent *multi-member* durable pause callback (ADR-023 pillar 2b).

        A concurrent wave (PARALLEL) can have several members awaiting approval at once. Unlike
        the sequential :meth:`_make_member_pause_hook` (one paused member, called mid-loop), this
        is invoked once after the wave with the full set of awaiting members. It persists a single
        pause checkpoint: ``completed_member_ids`` holds only the members that *finished* (awaiting
        ones are excluded so a resumed run re-runs them), plus an ``awaiting_approvals`` list of
        every pending approval so the run can durably resume once the human(s) decide.
        """

        async def _hook(
            awaiting_results: List[MemberResult],
            completed_results: List[MemberResult],
            shared_state: Dict[str, Any],
        ) -> None:
            last_success = next(
                (r for r in reversed(completed_results) if r.success and r.output), None
            )
            checkpoint = WorkflowCheckpoint(
                checkpoint_id=f"{thread_id}:{team_node_id}:pause",
                thread_id=thread_id,
                node_id=f"{team_node_id}:pause",
                state={
                    "completed_member_ids": [r.member_id for r in completed_results],
                    "member_results": [r.to_dict() for r in completed_results],
                    "shared_state": dict(shared_state or {}),
                    "last_output": last_success.output if last_success else None,
                    "last_agent_id": last_success.member_id if last_success else None,
                    "awaiting_approval": True,
                    "awaiting_approvals": [
                        {
                            "member_id": r.member_id,
                            "approval_request": (r.metadata or {}).get("approval_request"),
                        }
                        for r in awaiting_results
                    ],
                },
                timestamp=time.time(),
                metadata={
                    "team_node_id": team_node_id,
                    "formation": formation,
                    "awaiting_approval": True,
                    "awaiting_member_ids": [r.member_id for r in awaiting_results],
                },
            )
            await self._save_member_checkpoint(checkpointer, checkpoint, "save_batch_pause")

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
        delegate_reentry_contract = delegate_contracts.extract_delegate_reentry_contract(context)
        effective_context = delegate_contracts.apply_delegate_reentry_context(
            context,
            delegate_reentry_contract=delegate_reentry_contract,
        )

        # Resolve authoritative recovery state before allocating workspaces or admitting members.
        team_context_id = (
            effective_context.get("team_id") or effective_context.get("team_name") or "UnifiedTeam"
        )
        thread_id = effective_context.get("thread_id") or effective_context.get("__thread_id__")
        checkpointer = self._checkpointer
        resume_completed = None
        if checkpointer is not None and thread_id:
            resume_completed = await self._load_member_resume(
                checkpointer, str(thread_id), str(team_context_id)
            )

        parallel_isolation = bool(
            active_formation == TeamFormation.PARALLEL
            and effective_context.get("parallel_worktree_isolation", False)
        )
        if parallel_isolation:
            effective_context = dict(effective_context)
            effective_context.update(worktree_isolation=True, materialize_worktrees=True)
            # Preserve deliverables for review; merging/cleanup remain explicit.
            effective_context.setdefault("cleanup_worktrees", False)
            effective_context.setdefault("auto_merge_worktrees", False)

        # Wrap team members with participants
        shared_state_with_supervisor = self._active_shared_context()
        context_shared_state = effective_context.get("shared_state")
        if isinstance(context_shared_state, Mapping):
            shared_state_with_supervisor.update(context_shared_state)
        active_supervisor = self._active_supervisor()
        if active_supervisor is not None:
            shared_state_with_supervisor["explicit_supervisor_id"] = active_supervisor.id
        normalize_supervisor_context(shared_state_with_supervisor)

        max_workers = self._extract_max_workers(effective_context, shared_state_with_supervisor)
        candidate_members = delegate_contracts.filter_execution_members(
            self._active_members(),
            member_ids=delegate_contracts.extract_delegate_reentry_member_ids(
                delegate_reentry_contract
            ),
        )
        admission_limit = None
        if active_formation == TeamFormation.PARALLEL and effective_context.get(
            "capacity_aware_parallelism",
            shared_state_with_supervisor.get("capacity_aware_parallelism", False),
        ):
            provider = getattr(self._orchestrator, "provider", None)
            capacity_query = getattr(provider, "get_parallel_capacity", None)
            if not callable(capacity_query):
                raise ValueError(
                    "Capacity-aware parallelism requires a provider capacity declaration"
                )
            admission_limit = await capacity_query(getattr(self._orchestrator, "model", ""))
            if (
                not isinstance(admission_limit, int)
                or isinstance(admission_limit, bool)
                or admission_limit < 1
            ):
                raise ValueError("Provider parallel capacity must be a positive integer")
            if max_workers is not None:
                admission_limit = min(admission_limit, max_workers)
            # Admission queues every member. Legacy max_workers truncation is
            # preserved only when the additive admission contract is disabled.
            execution_members = list(candidate_members)
            shared_state_with_supervisor["member_concurrency_limit"] = admission_limit
        else:
            execution_members = self._limit_execution_members(
                candidate_members,
                active_formation,
                max_workers,
                supervisor=active_supervisor,
            )
        member_context_overrides = (
            delegate_contracts.extract_delegate_reentry_member_context_overrides(
                delegate_reentry_contract
            )
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
        if parallel_isolation and (worktree_session is None or not worktree_session.materialized):
            logger.warning("PARALLEL workspace isolation could not be materialized")
            raise ValueError("PARALLEL workspace isolation requires materialized git worktrees")
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
        if checkpointer is not None and thread_id:
            team_node_id = str(team_context_id)
            team_context.resume_completed = resume_completed
            team_context.checkpoint_hook = self._make_member_checkpoint_hook(
                checkpointer, str(thread_id), team_node_id, active_formation.value
            )
            # ADR-023 pillar 2b: durable pause when a member awaits approval.
            team_context.pause_hook = self._make_member_pause_hook(
                checkpointer, str(thread_id), team_node_id, active_formation.value
            )
            # ADR-023 pillar 2b (concurrent): multi-member pause for concurrent formations
            # (PARALLEL) — several members can await approval in one wave.
            team_context.batch_pause_hook = self._make_member_batch_pause_hook(
                checkpointer, str(thread_id), team_node_id, active_formation.value
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
        checkpoint_failure: Optional[MemberCheckpointError] = None
        durable_cancelled = False
        # ADR-023 pillar 2b: arm durable member pause for the duration of member execution when a
        # checkpointer + thread_id are configured (pause_hook set) AND the formation actually
        # implements durable pause. A member's ASK then raises MemberApprovalPause (durable) instead
        # of blocking on the modal. Gating on supports_durable_pause() is essential: arming a
        # formation that can't stop-and-checkpoint (e.g. PARALLEL) would silently abort the gated
        # tool. Non-supporting formations keep slice-2a inline approval.
        from victor.agent.member_approval_context import current_member_durable_pause_enabled

        _durable_pause_token = (
            current_member_durable_pause_enabled.set(True)
            if team_context.pause_hook is not None and strategy.supports_durable_pause()
            else None
        )
        try:
            # Execute using formation strategy (durable-pause arming reset immediately after).
            try:
                member_results_list = await strategy.execute(participants, team_context, agent_task)
            finally:
                if _durable_pause_token is not None:
                    current_member_durable_pause_enabled.reset(_durable_pause_token)

            # Convert list of MemberResults to dict
            member_results: Dict[str, MemberResult] = {r.member_id: r for r in member_results_list}
            self._inject_worktree_changed_files(
                member_results,
                worktree_session=worktree_session,
            )

            # Build final output
            if not member_results_list:
                success = False
            elif active_formation == TeamFormation.PARALLEL:
                # PARALLEL members are independent by contract (redundancy,
                # diverse perspectives): the team succeeds when at least one
                # member delivered. Failed members stay visible in
                # member_results instead of failing the whole team.
                success = any(r.success for r in member_results_list)
            else:
                success = all(r.success for r in member_results_list)
            final_outputs = [r.output for r in member_results_list if r.success]
            total_tool_calls = sum(r.tool_calls_used for r in member_results_list)

            # Determine final output based on formation
            # For pipeline, use only the last stage's output
            # For other formations, join all outputs
            if active_formation == TeamFormation.PIPELINE and final_outputs:
                final_output = final_outputs[-1]  # Last stage's output only
            else:
                final_output = "\n\n".join(final_outputs)

            if active_formation == TeamFormation.PARALLEL:
                failed = [result for result in member_results_list if not result.success]
                if failed:
                    summary = "Member failures:\n" + "\n".join(
                        f"- {result.member_id}: {result.error or 'execution failed'}"
                        for result in failed
                    )
                    final_output = f"{final_output}\n\n{summary}" if final_output else summary
            if active_formation == TeamFormation.CONSENSUS and member_results_list:
                metadata = member_results_list[0].metadata
                success = success and bool(
                    metadata.get("consensus_achieved") or "consensus_decision" in metadata
                )
                if "consensus_decision" in metadata:
                    final_output = metadata["consensus_decision"]

            if member_results_list and "ensemble_success" in member_results_list[0].metadata:
                metadata = member_results_list[0].metadata
                success = bool(metadata["ensemble_success"]) and all(
                    r.success for r in member_results_list
                )
                final_output = (
                    metadata["ensemble_decision"] if success else metadata["ensemble_error"]
                )

            if member_results_list and "conversation_success" in member_results_list[0].metadata:
                metadata = member_results_list[0].metadata
                success = bool(metadata["conversation_success"]) and all(
                    r.success for r in member_results_list
                )
                final_output = metadata["conversation_output"]

            if (
                active_formation == TeamFormation.REFLECTION
                and effective_context.get("capture_member_usage", False)
                and member_results_list
                and "reflection_success" in member_results_list[0].metadata
            ):
                metadata = member_results_list[0].metadata
                success = bool(metadata["reflection_success"]) and all(
                    r.success for r in member_results_list
                )
                final_output = metadata["reflection_output"]

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
            awaiting_many = team_context.shared_state.get("__awaiting_approvals__")
            if awaiting:
                result_dict["success"] = False  # team did not complete — it is paused
                result_dict["status"] = "awaiting_approval"
                result_dict["paused_member_id"] = awaiting.get("member_id")
                result_dict["approval_request"] = awaiting.get("approval_request")
                if thread_id:
                    result_dict["thread_id"] = str(thread_id)
            elif awaiting_many:
                # ADR-023 concurrent pause: a PARALLEL wave stopped with one or more members
                # awaiting approval — surface the multi-pause aggregate. Resumable on the same
                # thread_id via the persisted pause checkpoint (re-runs exactly the paused set).
                result_dict["success"] = False  # team did not complete — it is paused
                result_dict["status"] = "awaiting_approval"
                result_dict["awaiting_approvals"] = list(awaiting_many)
                result_dict["paused_member_ids"] = [a.get("member_id") for a in awaiting_many]
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
            worker_return_contracts = delegate_contracts.build_worker_return_contracts(
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
                                merge_execution_diagnostics = (
                                    delegate_contracts.normalize_workspace_diagnostics(
                                        merge_execution.get("diagnostics")
                                    )
                                )
                                if merge_execution_diagnostics:
                                    workspace_diagnostics.extend(merge_execution_diagnostics)
                                    result_dict["workspace_isolation_diagnostics"] = list(
                                        workspace_diagnostics
                                    )
                    delegate_follow_up_contract = (
                        delegate_contracts.build_delegate_follow_up_contract(
                            worker_return_contracts,
                            merge_review_contract=merge_review_contract,
                            worktree_session=worktree_session,
                            merge_execution=result_dict.get("merge_execution"),
                            merge_analysis=result_dict.get("merge_analysis"),
                            merge_orchestration=merge_orchestration,
                            workspace_diagnostics=workspace_diagnostics,
                            preserve_merge_follow_up=delegate_contracts.resolve_context_mode(
                                effective_context
                            )
                            == "delegate",
                        )
                    )
                    if delegate_follow_up_contract:
                        result_dict["delegate_follow_up_contract"] = delegate_follow_up_contract
                        delegate_contracts.attach_delegate_follow_up_suggestions(
                            result_dict,
                            delegate_follow_up_contract,
                        )

            return result_dict
        except MemberCheckpointError as exc:
            checkpoint_failure = exc
            raise
        except asyncio.CancelledError:
            durable_cancelled = checkpointer is not None and bool(thread_id)
            raise
        finally:
            cleanup_summary = None
            if worktree_session is not None:
                if checkpoint_failure is not None:
                    cleanup_summary = self._build_preserved_worktree_cleanup_summary(
                        worktree_session,
                        reason="member_checkpoint_failed",
                    )
                    checkpoint_failure.worktree_cleanup = cleanup_summary
                elif durable_cancelled:
                    cleanup_summary = self._build_preserved_worktree_cleanup_summary(
                        worktree_session,
                        reason="member_execution_cancelled",
                    )
                elif self._should_cleanup_worktrees(effective_context, result_dict=result_dict):
                    cleanup_summary = self._cleanup_worktree_session(worktree_session)
                else:
                    cleanup_summary = self._build_preserved_worktree_cleanup_summary(
                        worktree_session,
                        reason="preserved_for_follow_up",
                    )
                if result_dict is not None:
                    result_dict["worktree_cleanup"] = cleanup_summary
            if durable_cancelled:
                self._emit_team_event(
                    "cancelled",
                    {
                        "team_id": team_context_id,
                        "thread_id": str(thread_id),
                        "reconciliation_required": True,
                        **({"worktree_cleanup": cleanup_summary} if cleanup_summary else {}),
                    },
                )

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

    def _execute_delegate_merge_contract(
        self,
        task: str,
        context: Dict[str, Any],
        *,
        formation: TeamFormation,
        delegate_merge_contract: Mapping[str, Any],
    ) -> Dict[str, Any]:
        effective_context = delegate_contracts.apply_delegate_merge_context(
            context,
            delegate_merge_contract=delegate_merge_contract,
        )
        session_payload = (
            dict(delegate_merge_contract.get("worktree_session") or {})
            if isinstance(delegate_merge_contract.get("worktree_session"), Mapping)
            else {}
        )
        worktree_session = delegate_contracts.rebuild_worktree_session(session_payload)
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
            delegate_follow_up_contract = delegate_contracts.build_delegate_follow_up_contract(
                worker_return_contracts,
                merge_review_contract=merge_review_contract,
                worktree_session=worktree_session,
                merge_execution=merge_execution,
                merge_analysis=merge_analysis,
                merge_orchestration=merge_orchestration,
                preserve_merge_follow_up=delegate_contracts.resolve_context_mode(effective_context)
                == "delegate",
            )
            if delegate_follow_up_contract:
                result["delegate_follow_up_contract"] = delegate_follow_up_contract
                delegate_contracts.attach_delegate_follow_up_suggestions(
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
        self._supervisor = None
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
        return self._supervisor

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
        delegate_next_step = delegate_contracts.extract_delegate_next_step(context)
        effective_context = delegate_contracts.apply_delegate_next_step_context(
            context,
            delegate_next_step=delegate_next_step,
        )
        execution_members = list(members)
        delegate_merge_contract = delegate_contracts.extract_delegate_merge_contract(
            effective_context
        )
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
            team_goal=(task or "").strip(),
        )
        token = self._execution_state.set(execution_state)
        try:
            start_time = time.time()
            effective_formation = self._resolve_effective_formation(
                effective_context,
                default_formation=formation,
            )
            execution_state.formation = effective_formation

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
                    **(
                        {
                            "error_code": "member_checkpoint_failed",
                            "checkpoint_operation": e.operation,
                            "reconciliation_required": True,
                            **(
                                {"worktree_cleanup": e.worktree_cleanup}
                                if e.worktree_cleanup is not None
                                else {}
                            ),
                        }
                        if isinstance(e, MemberCheckpointError)
                        else {}
                    ),
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
                # The formation passes the TEAM goal to every member; a
                # member's own spec.goal is its actual assignment. Lead with
                # it (team goal as context) so parallel members each know
                # what - and only what - they are responsible for.
                # HIERARCHICAL is exempt: its `task` is the supervisor's
                # dynamic delegation, which outranks the static spec.goal.
                member_goal = (getattr(team_member, "goal", "") or "").strip()
                team_task = (task or "").strip()
                execution_state = self._current_execution_state()
                binding_context = execution_state.shared_context if execution_state else context
                binding = binding_context.get("member_task_binding")
                if binding is not None and binding != "structured-v1":
                    raise ValueError("member_task_binding must be 'structured-v1' or absent")
                is_dynamic_delegation = self._active_formation() == TeamFormation.HIERARCHICAL or (
                    team_task != (execution_state.team_goal if execution_state else "")
                )
                if binding == "structured-v1":
                    # Keep the formation's task opaque: it owns delegation and
                    # the response contract. Bind identity/assignment once here,
                    # never infer either from a rewritten task or role string.
                    effective_task = json.dumps(
                        {
                            "version": 1,
                            "member": {
                                "id": team_member.id,
                                "name": team_member.name,
                                "role": getattr(team_member.role, "value", team_member.role),
                                "assignment": team_member.goal,
                            },
                            "formation_task": task,
                            "instruction_priority": ["formation_task", "member.assignment"],
                            "instructions": (
                                "Execute formation_task for this member. Its active delegation "
                                "and response contract take priority over the declared assignment. "
                                "Use member.assignment to identify your own deliverables when "
                                "formation_task does not replace them. Return the formation's "
                                "response, not this input envelope."
                            ),
                        }
                    )
                elif member_goal and member_goal != team_task and not is_dynamic_delegation:
                    effective_task = (
                        f"{member_goal}\n\n(Team objective, for context: {team_task})"
                        if team_task
                        else member_goal
                    )
                else:
                    effective_task = task
                spawn_result = await sub_orchestrator.spawn(
                    role=team_member.role,
                    task=effective_task,
                    tool_budget=team_member.tool_budget,
                    allowed_tools=team_member.allowed_tools,
                    provider=getattr(team_member, "provider", None),
                    model=getattr(team_member, "model", None),
                    temperature=getattr(team_member, "temperature", None),
                    reasoning_effort=getattr(team_member, "reasoning_effort", None),
                    member_id=team_member.id,
                    display_name=getattr(team_member, "name", None),
                    team_id=context.get("team_id"),
                    plan_id=context.get("plan_id"),
                    plan_step_id=context.get("plan_step_id"),
                    parent_session_id=context.get("parent_session_id"),
                    child_session_id=context.get("child_session_id"),
                    working_directory=context.get("worktree_path"),
                    capture_usage=bool(context.get("capture_member_usage", False)),
                )
                return {
                    "success": getattr(spawn_result, "success", False),
                    "output": getattr(spawn_result, "summary", "") or "",
                    "error": getattr(spawn_result, "error", None),
                    "tool_calls_used": getattr(spawn_result, "tool_calls_used", 0),
                    "duration_seconds": getattr(spawn_result, "duration_seconds", 0.0),
                    "metadata": {
                        key: value
                        for key, value in (getattr(spawn_result, "details", {}) or {}).items()
                        if key
                        in {
                            "member_id",
                            "agent_id",
                            "display_name",
                            "team_id",
                            "plan_id",
                            "plan_step_id",
                            "parent_session_id",
                            "child_session_id",
                            "session_id",
                            "usage",
                            "awaiting_approval",
                            "approval_request",
                        }
                    },
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
        raw_formation = result.get("formation")
        try:
            effective_formation = (
                raw_formation
                if isinstance(raw_formation, TeamFormation)
                else TeamFormation(str(raw_formation))
            )
        except (TypeError, ValueError):
            effective_formation = formation
        return TeamResult(
            success=bool(result.get("success", False)),
            final_output=str(result.get("final_output", "")),
            member_results=dict(result.get("member_results", {})),
            formation=effective_formation,
            total_tool_calls=int(result.get("total_tool_calls", 0)),
            total_duration=float(result.get("total_duration", 0.0)),
            communication_log=list(result.get("communication_log", [])),
            shared_context=dict(result.get("shared_context", {})),
            consensus_achieved=result.get("consensus_achieved"),
            consensus_rounds=result.get("consensus_rounds"),
            error=result.get("error"),
            status=result.get("status"),
            paused_member_id=result.get("paused_member_id"),
            approval_request=result.get("approval_request"),
            thread_id=result.get("thread_id"),
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
                if selected_formation is not None and not asyncio.iscoroutine(selected_formation):
                    selected_formation = TeamFormation(
                        getattr(
                            selected_formation, "value", str(selected_formation).strip().lower()
                        )
                    )
            except Exception as exc:
                logger.warning("Formation strategy failed; keeping default formation: %s", exc)
                sink = current_member_sink.get()
                if sink is not None:
                    from victor.framework.member_event_sink import (
                        TEAM_FORMATION_WARNING,
                        MemberEvent,
                    )

                    await sink.emit(
                        MemberEvent(
                            kind=TEAM_FORMATION_WARNING,
                            member_id="coordinator",
                            formation=self._formation.value,
                            success=False,
                            content="Formation strategy failed; using configured default",
                            metadata={"reason": type(exc).__name__, "level": "warning"},
                        )
                    )
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
