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

"""Hierarchical formation strategy.

Supervisor agent delegates to specialist agents,
then synthesizes results.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult, MessageType

logger = logging.getLogger(__name__)


class HierarchicalFormation(BaseFormationStrategy):
    """Execute with supervisor-specialist delegation pattern.

    In hierarchical formation:
    - One supervisor delegates tasks to specialists
    - Specialists execute in parallel
    - Supervisor synthesizes specialist results
    - Prefers an explicit supervisor category/contract

    Use case: Complex task decomposition, supervised execution
    """

    def get_required_roles(self) -> Optional[List[str]]:
        """Hierarchical formation requires a coordinating supervisor role."""
        return ["supervisor", "coordinator", "lead", "manager"]

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute the supervisor → specialists → synthesis phases.

        ADR-023 phase-granular durable checkpoint/resume (opt-in — active only when the
        coordinator wired ``checkpoint_hook``/``resume_completed``, i.e. a checkpointer +
        thread_id). The three phases — supervisor **plan**, concurrent **specialists**, supervisor
        **synthesis** — are snapshotted into ``shared_state["__hier__"]`` as each completes (the
        existing member checkpoint hook already persists ``shared_state``), so a crash resumes at
        the last *completed* phase: a completed plan is **restored, not re-run** (its
        ``delegated_tasks`` drive phase 2), completed specialists are restored, and only the
        unreached phase(s) execute. The specialist wave runs through the shared concurrent runner,
        which checkpoints the cumulative completed set mid-wave — a crash mid-wave resumes by
        re-running only the unfinished specialists under the *restored* plan (no replan).
        No checkpointer ⇒ byte-identical.

        ADR-023 durable pause (opt-in on ``pause_hook``/``batch_pause_hook``): every phase handles
        an awaiting-approval result — the supervisor **plan** and **synthesis** pause via the
        singular ``__awaiting_approval__`` aggregate (mirroring SEQUENTIAL; the paused phase is
        *not* snapshotted, so a resumed run re-executes exactly it), and the specialist wave pauses
        via the concurrent multi-pause aggregate (``__awaiting_approvals__`` + one batch pause
        checkpoint; completed specialists are checkpointed and skipped on resume). Covering all
        three pause points is what makes arming durable pause safe here
        (see :meth:`supports_durable_pause`).
        """
        if len(agents) < 2:
            raise ValueError(
                "Hierarchical formation requires at least 2 agents " "(supervisor + specialists)"
            )

        supervisor, specialists = self._resolve_supervisor(agents, context)
        logger.debug(
            f"HierarchicalFormation: supervisor={supervisor.id}, "
            f"specialists={[s.id for s in specialists]}"
        )

        # ADR-023: phase-granular checkpoint/resume state (opt-in). ``saved`` accumulates the
        # per-phase snapshot; ``phase_done`` is the highest phase already persisted (0 = fresh).
        checkpoint_hook = getattr(context, "checkpoint_hook", None)
        pause_hook = getattr(context, "pause_hook", None)
        resume = getattr(context, "resume_completed", None)
        saved: Dict[str, Any] = dict(
            ((resume or {}).get("shared_state") or {}).get("__hier__") or {}
        )
        phase_done = int(saved.get("phase", 0))

        async def _checkpoint_phase(phase: int, marker: MemberResult) -> None:
            """Persist progress up to ``phase`` as one immutable snapshot via the member hook."""
            if checkpoint_hook is None:
                return
            saved["phase"] = phase
            # A fresh dict each save so every checkpoint captures its own snapshot (the member
            # hook stores ``dict(shared_state)`` — a shallow copy — so aliasing __hier__ would
            # let a later phase mutate an earlier checkpoint).
            context.shared_state["__hier__"] = dict(saved)
            await checkpoint_hook(phase - 1, marker, [marker], context.shared_state)

        # ── Phase 1: supervisor plans and delegates ──
        if phase_done >= 1 and saved.get("plan") is not None:
            supervisor_result = MemberResult.from_dict(saved["plan"])
            logger.info(
                "HierarchicalFormation: resume — restored supervisor plan (phase 1 skipped)"
            )
        else:
            supervisor_result = await supervisor.execute(task, context)
            # ADR-023 pillar 2b: the supervisor's plan hit an approval gate — durably pause
            # (mirroring SEQUENTIAL). The plan is NOT snapshotted, so a resumed run re-executes it.
            if pause_hook is not None and (supervisor_result.metadata or {}).get(
                "awaiting_approval"
            ):
                await self._pause_on_supervisor(
                    supervisor_result, [], context, pause_hook, phase="plan"
                )
                return []
            saved["plan"] = supervisor_result.to_dict()
            await _checkpoint_phase(1, supervisor_result)

        # Results: supervisor first, then specialists in original order.
        results: List[MemberResult] = [supervisor_result]

        delegated_tasks = supervisor_result.metadata.get("delegated_tasks")
        used_delegation = bool(supervisor_result.success and delegated_tasks)

        # ── Phase 2: specialists execute in parallel ──
        if phase_done >= 2 and saved.get("specialists") is not None:
            results.extend(MemberResult.from_dict(r) for r in saved["specialists"])
            if checkpoint_hook is not None:
                # Keep the live snapshot current so a phase-3 pause checkpoint embeds phases 1–2.
                context.shared_state["__hier__"] = dict(saved)
            logger.info(
                "HierarchicalFormation: resume — restored %d specialists (phase 2 skipped)",
                len(saved["specialists"]),
            )
        else:
            if not used_delegation:
                logger.info(
                    "HierarchicalFormation: supervisor did not delegate tasks, "
                    "executing all specialists with the original task"
                )
                pairs = [(sp, task, i) for i, sp in enumerate(specialists)]
            else:
                if len(delegated_tasks) != len(specialists):
                    logger.warning(
                        f"HierarchicalFormation: task count mismatch: "
                        f"{len(delegated_tasks)} tasks vs {len(specialists)} specialists"
                    )
                # Only as many specialists as there are delegated tasks run (original semantics).
                pairs = [
                    (sp, delegated_tasks[i], i)
                    for i, sp in enumerate(specialists)
                    if i < len(delegated_tasks)
                ]
            # Write the phase-1 snapshot into live shared_state BEFORE the wave — even on the
            # restored-plan resume path — so the runner's mid-wave cumulative checkpoints (and a
            # batch pause checkpoint) embed the plan: a mid-wave crash or pause resumes under the
            # restored plan (no replan) and re-runs only the unfinished specialists.
            if checkpoint_hook is not None:
                context.shared_state["__hier__"] = dict(saved)
            specialist_results = await self._run_specialists(pairs, task, context, resume)
            # ADR-023 concurrent durable pause: one or more specialists awaited approval — the
            # shared runner already published the multi-pause aggregate (__awaiting_approvals__)
            # and persisted one batch pause checkpoint. Return supervisor + completed specialists
            # WITHOUT snapshotting phase 2 and WITHOUT synthesizing: the resumed run restores the
            # plan and re-runs exactly the paused specialists (completed ones are skipped via the
            # pause checkpoint's completed set).
            if context.shared_state.get("__awaiting_approvals__"):
                logger.info("HierarchicalFormation: specialist wave awaiting approval; pausing")
                return results + specialist_results
            results.extend(specialist_results)
            saved["specialists"] = [r.to_dict() for r in specialist_results]
            await _checkpoint_phase(
                2, specialist_results[-1] if specialist_results else supervisor_result
            )

        # The fallback (no-delegation) path has no synthesis phase — it ends at phase 2.
        if not used_delegation:
            return results

        # ── Phase 3: supervisor synthesizes specialist results ──
        if phase_done >= 3 and saved.get("synthesis") is not None:
            results[0] = MemberResult.from_dict(saved["synthesis"])
            logger.info("HierarchicalFormation: resume — restored synthesis (phase 3 skipped)")
            return results

        synthesis_inputs = [
            {
                "agent_id": r.member_id,
                "success": r.success,
                "content": r.output,
            }
            for r in results[1:]
        ]
        synthesis_task = AgentMessage(
            message_type=MessageType.RESULT,
            sender_id="system",
            recipient_id=supervisor.id,
            content={
                "task": "Synthesize specialist results",
                "specialist_results": synthesis_inputs,
                "worker_results": synthesis_inputs,
            },
        )

        synthesis_result = await supervisor.execute(synthesis_task, context)
        # ADR-023 pillar 2b: the synthesis hit an approval gate — durably pause. The synthesis is
        # NOT snapshotted (and results[0] stays the plan), so a resumed run restores phases 1–2
        # from the pause checkpoint's __hier__ snapshot and re-executes only the synthesis.
        if pause_hook is not None and (synthesis_result.metadata or {}).get("awaiting_approval"):
            await self._pause_on_supervisor(
                synthesis_result, results, context, pause_hook, phase="synthesis"
            )
            return results
        results[0] = synthesis_result  # Replace with final synthesis
        saved["synthesis"] = synthesis_result.to_dict()
        await _checkpoint_phase(3, synthesis_result)

        return results

    async def _pause_on_supervisor(
        self,
        supervisor_result: MemberResult,
        completed: List[MemberResult],
        context: TeamContext,
        pause_hook: Any,
        *,
        phase: str,
    ) -> None:
        """Durably pause on an awaiting-approval supervisor phase (plan or synthesis).

        Mirrors ``SequentialFormation``'s pause handling: publish the singular
        ``__awaiting_approval__`` aggregate (the supervisor's lane index is 0), emit the awaiting
        lane event, and persist the pause checkpoint via ``pause_hook`` with only the results
        completed *before* the paused phase — the paused phase is not snapshotted into
        ``__hier__``, so a resumed run re-executes exactly it.
        """
        logger.info(f"HierarchicalFormation: supervisor {phase} awaiting approval; pausing")
        metadata = supervisor_result.metadata or {}
        approval_request = metadata.get("approval_request")
        member_event_hook = getattr(context, "member_event_hook", None)
        if member_event_hook is not None:
            detail = str(
                (approval_request or {}).get("title")
                or (approval_request or {}).get("tool_name")
                or ""
            )
            await member_event_hook(
                "member_awaiting_approval", supervisor_result.member_id, 0, content=detail
            )
        context.shared_state["__awaiting_approval__"] = {
            "member_id": supervisor_result.member_id,
            "index": 0,
            "approval_request": approval_request,
        }
        await pause_hook(0, supervisor_result, completed, context.shared_state)

    def _resolve_supervisor(self, agents: List[Any], context: TeamContext) -> Tuple[Any, List[Any]]:
        """Detect the supervisor and specialists from the agent list.

        Prefers an explicit ``explicit_supervisor_id`` (legacy ``explicit_manager_id``) in the
        shared state, then supervisor/delegate contract signals, falling back to the first agent.
        """
        # Check for explicit supervisor in context first. Keep the older
        # explicit_manager_id key readable for serialized compatibility.
        explicit_supervisor_id = context.shared_state.get(
            "explicit_supervisor_id",
            context.shared_state.get("explicit_manager_id"),
        )

        supervisor = None
        specialists: List[Any] = []

        for agent in agents:
            # If explicit supervisor is set, use it.
            if explicit_supervisor_id and agent.id == explicit_supervisor_id:
                supervisor = agent
                continue
            if explicit_supervisor_id:
                specialists.append(agent)
                continue

            # Otherwise check for explicit supervisor/member metadata.
            is_supervisor = False
            if getattr(agent, "is_supervisor", False):
                is_supervisor = True

            member = getattr(agent, "_member", None)
            if not is_supervisor and member is not None and getattr(member, "is_supervisor", False):
                is_supervisor = True

            if not is_supervisor and getattr(agent, "can_delegate", False):
                is_supervisor = True
            if (
                not is_supervisor
                and member is not None
                and getattr(member, "can_delegate", False)
                and getattr(member, "is_manager", False)
            ):
                is_supervisor = True

            if hasattr(agent, "role") and hasattr(agent.role, "capabilities"):
                from victor.framework.agent_protocols import AgentCapability

                if AgentCapability.DELEGATE in agent.role.capabilities:
                    is_supervisor = True

            # Check if agent has _role attribute with a coordinator-like name.
            if not is_supervisor and hasattr(agent, "_role") and hasattr(agent._role, "name"):
                role_name = agent._role.name.lower()
                if (
                    "supervisor" in role_name
                    or "manager" in role_name
                    or "lead" in role_name
                    or "coordinator" in role_name
                ):
                    is_supervisor = True

            if is_supervisor:
                supervisor = agent
            else:
                specialists.append(agent)

        # If no supervisor found by contract/capability, use first agent as fallback.
        if supervisor is None:
            logger.warning(
                "HierarchicalFormation: no supervisor detected by contract/capability, "
                "using first agent as supervisor"
            )
            supervisor = agents[0]
            specialists = agents[1:]
        elif explicit_supervisor_id:
            logger.info(f"HierarchicalFormation: using explicit supervisor={supervisor.id}")
        else:
            logger.info(f"HierarchicalFormation: auto-detected supervisor={supervisor.id}")

        return supervisor, specialists

    async def _run_specialists(
        self,
        pairs: List[Tuple[Any, AgentMessage, int]],
        task: AgentMessage,
        context: TeamContext,
        resume: Optional[Dict[str, Any]] = None,
    ) -> List[MemberResult]:
        """Run ``(specialist, task, index)`` tuples through the shared concurrent runner.

        Reuses ``BaseFormationStrategy._execute_members_concurrently`` (the PARALLEL wave runner)
        with per-specialist tasks and 1-based lane indices (the supervisor is 0), so the wave gets
        the full ADR-023 concurrent contract for free: per-member streaming lanes, lock-protected
        mid-wave cumulative checkpoints (= per-specialist partial resume), and the multi-member
        awaiting-approval collection + batch pause. Specialists share the live team context
        (hierarchical semantics — unlike PARALLEL's isolated copies), and a specialist exception
        still becomes a failed ``MemberResult`` tagged with its 1-based index.

        ``resume`` (the coordinator's ``resume_completed`` payload) is filtered down to specialist
        ids — the supervisor's phase results ride ``__hier__``, not the wave's completed set — and
        always passed as an explicit override (``{}`` when nothing matches) so the runner never
        seeds the wave from the unfiltered context payload.
        """
        agents = [sp for sp, _tk, _i in pairs]
        specialist_tasks = [tk for _sp, tk, _i in pairs]
        indices = [i + 1 for _sp, _tk, i in pairs]

        # Filter the resume payload to this wave's specialists (degrades gracefully on old-format
        # checkpoints — no matching ids ⇒ a fresh wave).
        specialist_ids = {a.id for a in agents}
        resume_override: Dict[str, Any] = {}
        if resume:
            kept: List[MemberResult] = []
            for raw in resume.get("member_results") or []:
                r = raw if isinstance(raw, MemberResult) else MemberResult.from_dict(raw)
                if r.member_id in specialist_ids:
                    kept.append(r)
            kept_ids = [m for m in (resume.get("member_ids") or []) if m in specialist_ids] or [
                r.member_id for r in kept
            ]
            if kept_ids:
                resume_override = {"member_ids": kept_ids, "member_results": kept}

        return await self._execute_members_concurrently(
            agents,
            task,
            context,
            [context] * len(agents),
            tasks=specialist_tasks,
            indices=indices,
            resume_override=resume_override,
        )

    def validate_context(self, context: TeamContext) -> bool:
        """Hierarchical formation requires delegation support."""
        return context is not None and hasattr(context, "shared_state")

    def supports_early_termination(self) -> bool:
        """Hierarchical formation requires all specialists to complete."""
        return False

    def supports_durable_pause(self) -> bool:
        """HIERARCHICAL implements ADR-023 durable pause at all three phases.

        Arming durable pause makes a member ``ASK`` raise ``MemberApprovalPause`` in *any* phase,
        so every phase must stop-and-checkpoint on an awaiting result (or the gated tool would
        silently abort — the #740 class of bug). The supervisor **plan** and **synthesis** pause
        via the singular ``__awaiting_approval__`` aggregate (the paused phase is not snapshotted,
        so resume re-executes exactly it); the **specialist wave** pauses via the shared concurrent
        runner's multi-pause aggregate (``__awaiting_approvals__`` + one batch pause checkpoint;
        completed specialists are skipped on resume). Arming is therefore safe.
        """
        return True
