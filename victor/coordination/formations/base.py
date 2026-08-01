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

"""Base formation strategy for multi-agent coordination.

MIGRATION NOTICE: TeamContext state storage is migrating to canonical system.

For team state storage, use the canonical state management system:
    - victor.state.TeamStateManager - Team scope state
    - victor.state.get_global_manager() - Unified access to all scopes

The TeamContext class now uses TeamStateManager internally for
shared_state storage.

---

Legacy Documentation:

This module defines the abstract base for all formation strategies,
following the Open/Closed Principle (OCP) and Strategy pattern.
"""

import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional

from victor.teams.types import AgentMessage, MemberResult

logger = logging.getLogger(__name__)


class TeamContext:
    """Simple context for team execution.

    MIGRATION: This class now integrates with TeamStateManager for
    canonical state storage. The shared_state dict is kept for
    backward compatibility but is synced to TeamStateManager.

    Attributes:
        team_id: Team identifier
        formation: Formation pattern being used
        shared_state: Shared state between agents (DEPRECATED - use state_manager)
        state_manager: Optional TeamStateManager for canonical state storage
        metadata: Additional metadata

    Example:
        # OLD (using shared_state dict):
        context = TeamContext("team-1", "orchestration")
        context.shared_state["coordinator"] = "agent-1"

        # NEW (using canonical state manager):
        from victor.state import TeamStateManager

        mgr = TeamStateManager()
        await mgr.set("coordinator", "agent-1")

        # OR with TeamContext integration:
        context = TeamContext("team-1", "orchestration", state_manager=mgr)
        await context.set("coordinator", "agent-1")  # Uses manager
    """

    def __init__(
        self,
        team_id: str,
        formation: str,
        shared_state: Optional[Dict[str, Any]] = None,
        state_manager: Optional[Any] = None,
        lsp_capability: Optional[Any] = None,
        **metadata: Any,
    ):
        self.team_id = team_id
        self.formation = formation
        self.shared_state = shared_state or {}
        self._lsp = lsp_capability  # Language Server Protocol capability
        self.metadata = metadata
        self._state_manager = state_manager
        self._lock = threading.Lock()

        # ADR-023 member durability (opt-in; set by UnifiedTeamCoordinator when a
        # checkpointer is configured). None → no checkpoint/resume, unchanged behavior.
        # checkpoint_hook: awaited after each member completes with
        #   (index, member_result, results_so_far, shared_state).
        # resume_completed: seeds a resumed run — {member_ids, member_results,
        #   shared_state, last_output, last_agent_id}.
        self.checkpoint_hook: Optional[Callable[..., Awaitable[None]]] = None
        self.resume_completed: Optional[Dict[str, Any]] = None

        # ADR-023 per-member streaming (opt-in; set by UnifiedTeamCoordinator when a
        # member event sink is present on the stream context var). Awaited by a formation
        # around each member with (kind, member_id, index, *, success=, content=). None →
        # no member events emitted, unchanged behavior.
        self.member_event_hook: Optional[Callable[..., Awaitable[None]]] = None

        # ADR-023 pillar 2b: durable member pause (opt-in; set by UnifiedTeamCoordinator when a
        # checkpointer is configured). Awaited when a member reports awaiting-approval, with
        # (index, member_result, completed_results, shared_state); persists a pause checkpoint
        # and the formation stops. None → the awaiting-approval flag is ignored (unchanged).
        self.pause_hook: Optional[Callable[..., Awaitable[None]]] = None

        # ADR-023 pillar 2b (concurrent): durable *multi-member* pause (opt-in; set alongside
        # pause_hook by UnifiedTeamCoordinator when a checkpointer is configured). A concurrent
        # wave (PARALLEL) can have several members awaiting approval at once; this batch hook is
        # awaited once, after the wave, with (awaiting_results, completed_results, shared_state)
        # and persists a single pause checkpoint recording all pending approvals. None → the
        # concurrent runner never collects awaiting members (unchanged).
        self.batch_pause_hook: Optional[Callable[..., Awaitable[None]]] = None

        # Initialize manager with existing shared_state
        if self._state_manager and self.shared_state:
            self._sync_to_manager()

    def _sync_to_manager(self) -> None:
        """Sync shared_state to the canonical state manager.

        Called once during __init__. Uses direct _state access since
        the async manager API is unavailable in a sync constructor.
        This is safe because no concurrent access exists at construction time.
        """
        if not self._state_manager:
            return

        try:
            for key, value in self.shared_state.items():
                self._state_manager._state[key] = value
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to sync state to manager: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared state.

        DEPRECATED: Use state_manager.get() instead.

        Args:
            key: Key to retrieve
            default: Default value if key doesn't exist

        Returns:
            Value associated with key, or default
        """
        with self._lock:
            if self._state_manager:
                try:
                    return self._state_manager._state.get(key, default)
                except Exception:
                    pass
            return self.shared_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in shared state.

        DEPRECATED: Use state_manager.set() instead.

        Args:
            key: Key to set
            value: Value to store
        """
        with self._lock:
            if self._state_manager:
                try:
                    self._state_manager._state[key] = value
                except Exception:
                    pass
            self.shared_state[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple values in shared state.

        DEPRECATED: Use state_manager.update() instead.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        with self._lock:
            if self._state_manager:
                try:
                    self._state_manager._state.update(updates)
                except Exception:
                    pass
            self.shared_state.update(updates)

    @property
    def lsp(self) -> Optional[Any]:
        """Get the LSP capability for code intelligence in team context.

        Returns:
            LSPCapability instance or None
        """
        return self._lsp

    def set_lsp(self, lsp_capability: Any) -> None:
        """Set the LSP capability for team context.

        Enables language intelligence features for team coordination
        and code-related operations.

        Args:
            lsp_capability: LSPCapability instance
        """
        self._lsp = lsp_capability


class BaseFormationStrategy(ABC):
    """Abstract base for formation strategies.

    This class defines the contract for all formation strategies,
    allowing the coordinator to delegate execution logic while
    remaining independent of specific implementations (OCP).

    Implementations must define how to:
    - Execute agents in the formation pattern
    - Handle results from agents
    - Manage context flow between agents
    """

    @abstractmethod
    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute agents using this formation strategy.

        Args:
            agents: List of agents to execute
            context: Team context with shared state
            task: Task message to process

        Returns:
            List of results from each agent
        """
        pass

    @abstractmethod
    def validate_context(self, context: "TeamContext") -> bool:
        """Validate that context has required fields for this formation.

        Args:
            context: Team context to validate

        Returns:
            True if context is valid for this formation
        """
        pass

    def get_required_roles(self) -> Optional[List[str]]:
        """Get required roles for this formation (if any).

        Returns:
            List of required role names, or None if any role is acceptable
        """
        return None

    def supports_early_termination(self) -> bool:
        """Check if this formation supports early termination.

        Returns:
            True if formation can terminate before all agents complete
        """
        return False

    def supports_durable_pause(self) -> bool:
        """Whether this formation implements ADR-023 durable member pause/resume.

        Only formations that stop-and-checkpoint on an ``awaiting_approval`` member (today
        just SEQUENTIAL) may arm durable pause; otherwise a member ``ASK`` under a
        checkpointer would raise ``MemberApprovalPause`` with no handler to stop the run,
        silently aborting the gated tool. Non-supporting formations keep slice-2a inline
        approval. Returns ``False`` by default.
        """
        return False

    async def _execute_member_with_events(
        self,
        agent: Any,
        agent_task: "AgentMessage",
        exec_context: "TeamContext",
        index: int,
        *,
        member_event_hook: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> MemberResult:
        """Run one member, emitting ADR-023 per-member lane events around it.

        Shared by the concurrent formations (PARALLEL / HIERARCHICAL) so per-member streaming
        lanes work without each reimplementing the emit logic — the sink ``ContextVar`` and the
        coordinator's single ``member_event_hook`` propagate into ``gather``-spawned tasks. Emits
        ``member_start`` before execution and, by outcome, ``member_awaiting_approval`` /
        ``member_completed`` / ``member_error`` after (mirroring SEQUENTIAL), so the same lanes
        render. Never raises: a member exception becomes a failed ``MemberResult`` (so
        ``gather(return_exceptions=True)`` post-processing stays a harmless safety net). With no
        hook, this is just ``await agent.execute(...)`` — byte-identical.
        """
        if member_event_hook is not None:
            await member_event_hook("member_start", agent.id, index)
        try:
            result = await agent.execute(agent_task, exec_context)
        except Exception as e:  # noqa: BLE001 - normalized to a failed member result
            logger.error(f"{type(self).__name__}: member {agent.id} failed: {e}")
            result = MemberResult(
                member_id=agent.id,
                success=False,
                output="",
                error=str(e),
                metadata={"index": index},
            )
        if member_event_hook is not None:
            metadata = result.metadata or {}
            if metadata.get("awaiting_approval"):
                approval_request = metadata.get("approval_request") or {}
                detail = str(
                    approval_request.get("title") or approval_request.get("tool_name") or ""
                )
                await member_event_hook("member_awaiting_approval", agent.id, index, content=detail)
            else:
                await member_event_hook(
                    "member_completed" if result.success else "member_error",
                    agent.id,
                    index,
                    success=result.success,
                    content="" if result.success else (result.error or ""),
                )
        return result

    async def _execute_members_concurrently(
        self,
        agents: List[Any],
        task: "AgentMessage",
        context: "TeamContext",
        exec_contexts: List["TeamContext"],
        *,
        tasks: Optional[List["AgentMessage"]] = None,
        indices: Optional[List[int]] = None,
        resume_override: Optional[Dict[str, Any]] = None,
    ) -> List[MemberResult]:
        """Run members concurrently with ADR-023 durable checkpoint/resume + streaming lanes.

        Reused by the concurrent formations (PARALLEL and the HIERARCHICAL specialist wave).
        Members execute concurrently via ``asyncio.gather`` (each in its own ``exec_context``);
        only the brief completion handler — record the result + checkpoint the cumulative
        completed set — is serialized under an ``asyncio.Lock``, so parallelism is preserved. On
        resume, members already in the checkpoint's completed set are skipped and their results
        pre-seeded; the rest re-run. All hooks are read off the original ``context`` (the
        isolated per-member contexts don't carry them). No checkpointer ⇒ this is just a
        lane-emitting gather.

        The keyword extensions are additive with PARALLEL-preserving defaults:

        - ``tasks``: per-member task messages (HIERARCHICAL delegates a *different* task to each
          specialist). ``None`` → every member receives the shared ``task``.
        - ``indices``: per-member lane/checkpoint indices (HIERARCHICAL specialists are 1-based —
          the supervisor is 0). ``None`` → ``enumerate`` order.
        - ``resume_override``: an explicit ``resume_completed``-shaped payload (HIERARCHICAL
          filters the coordinator payload down to specialist ids — the supervisor's phase results
          must not seed the wave). ``None`` → read ``context.resume_completed``; pass ``{}`` to
          force a fresh wave regardless of the context payload.

        ADR-023 concurrent durable pause: when ``batch_pause_hook`` is set (durable team run whose
        formation ``supports_durable_pause()``), members that come back awaiting approval are
        *not* recorded as completed — they are collected and, after the wave, a single pause
        aggregate (``__awaiting_approvals__``) + pause checkpoint records every pending approval,
        so a resumed run re-runs exactly the awaiting members while completed ones are skipped.
        """
        member_event_hook = getattr(context, "member_event_hook", None)
        checkpoint_hook = getattr(context, "checkpoint_hook", None)
        batch_pause_hook = getattr(context, "batch_pause_hook", None)
        member_tasks = tasks if tasks is not None else [task] * len(agents)
        member_indices = indices if indices is not None else list(range(len(agents)))

        # ADR-023 resume: pre-seed completed members and skip them below.
        seeded: List[MemberResult] = []
        completed_ids: set = set()
        resume = (
            resume_override
            if resume_override is not None
            else getattr(context, "resume_completed", None)
        )
        if resume:
            for raw in resume.get("member_results") or []:
                seeded.append(raw if isinstance(raw, MemberResult) else MemberResult.from_dict(raw))
            completed_ids = set(resume.get("member_ids") or [r.member_id for r in seeded])

        lock = asyncio.Lock()
        cumulative: List[MemberResult] = list(seeded)
        awaiting: List[MemberResult] = []
        _SKIPPED = object()
        _AWAITING = object()

        async def _run(
            agent: Any, exec_context: "TeamContext", index: int, agent_task: "AgentMessage"
        ) -> Any:
            if agent.id in completed_ids:
                logger.debug(
                    f"{type(self).__name__}: skipping completed member {agent.id} (resume)"
                )
                return _SKIPPED
            result = await self._execute_member_with_events(
                agent, agent_task, exec_context, index, member_event_hook=member_event_hook
            )
            # A member awaiting approval durably pauses (only when the formation supports it, i.e.
            # a batch pause hook is wired): it is NOT recorded as completed, so a resumed run
            # re-runs it. Collect it for the post-wave pause aggregate instead.
            is_awaiting = batch_pause_hook is not None and bool(
                (result.metadata or {}).get("awaiting_approval")
            )
            if checkpoint_hook is not None or is_awaiting:
                # Serialize only the record/checkpoint/collect — execution above stayed concurrent.
                async with lock:
                    if is_awaiting:
                        awaiting.append(result)
                    else:
                        cumulative.append(result)
                        if checkpoint_hook is not None:
                            await checkpoint_hook(
                                index, result, list(cumulative), context.shared_state
                            )
            return _AWAITING if is_awaiting else result

        gathered = await asyncio.gather(
            *[
                _run(a, c, i, t)
                for a, c, i, t in zip(agents, exec_contexts, member_indices, member_tasks)
            ],
            return_exceptions=True,
        )

        fresh: List[MemberResult] = []
        for i, r in enumerate(gathered):
            if r is _SKIPPED or r is _AWAITING:
                continue
            if isinstance(r, BaseException):
                logger.error(f"{type(self).__name__}: member {agents[i].id} failed: {r}")
                fresh.append(
                    MemberResult(
                        member_id=agents[i].id,
                        success=False,
                        output="",
                        error=str(r),
                        metadata={"index": member_indices[i]},
                    )
                )
            else:
                fresh.append(r)

        # ADR-023 concurrent durable pause: one or more members awaited approval this wave.
        # Publish the multi-pause aggregate and persist a single pause checkpoint recording every
        # pending approval; the coordinator surfaces the paused aggregate. Awaiting members are
        # absent from the returned results (excluded from the completed set), so a resumed run
        # re-runs exactly them.
        if awaiting and batch_pause_hook is not None:
            context.shared_state["__awaiting_approvals__"] = [
                {
                    "member_id": r.member_id,
                    "approval_request": (r.metadata or {}).get("approval_request"),
                }
                for r in awaiting
            ]
            await batch_pause_hook(list(awaiting), list(cumulative), context.shared_state)

        return seeded + fresh

    def consumes_context_agents(self) -> bool:
        """Whether this formation reads role-named agents from the context.

        Most formations operate on the ``agents`` list passed to ``execute()``.
        A few (e.g. reflection) instead pull named role agents — "generator",
        "critic" — out of ``context`` (see :meth:`get_required_roles`). When this
        returns True, the coordinator binds members to those roles and injects
        them into the context before execution.

        Returns:
            True if ``execute()`` reads agents from context by role name.
        """
        return False

    async def prepare_context(
        self,
        context: "TeamContext",
        task: AgentMessage,
    ) -> "TeamContext":
        """Prepare context before execution.

        Args:
            context: Initial team context
            task: Task message

        Returns:
            Prepared context with formation-specific initialization
        """
        return context

    async def process_results(
        self,
        results: List[MemberResult],
        context: "TeamContext",
    ) -> List[MemberResult]:
        """Process results after execution.

        Args:
            results: Raw results from agents
            context: Team context after execution

        Returns:
            Processed results
        """
        return results
