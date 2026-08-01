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

"""Hierarchical formation strategy.

Supervisor agent delegates to specialist agents,
then synthesizes results.
"""

import asyncio
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
        unreached phase(s) execute. A crash *mid* specialist wave replans (per-specialist partial
        resume is the concurrent-checkpoint refinement, deferred). No checkpointer ⇒ byte-identical.
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
            saved["plan"] = supervisor_result.to_dict()
            await _checkpoint_phase(1, supervisor_result)

        # Results: supervisor first, then specialists in original order.
        results: List[MemberResult] = [supervisor_result]

        delegated_tasks = supervisor_result.metadata.get("delegated_tasks")
        used_delegation = bool(supervisor_result.success and delegated_tasks)

        # ── Phase 2: specialists execute in parallel ──
        if phase_done >= 2 and saved.get("specialists") is not None:
            results.extend(MemberResult.from_dict(r) for r in saved["specialists"])
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
            specialist_results = await self._run_specialists(pairs, context)
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
        results[0] = synthesis_result  # Replace with final synthesis
        saved["synthesis"] = synthesis_result.to_dict()
        await _checkpoint_phase(3, synthesis_result)

        return results

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
        context: TeamContext,
    ) -> List[MemberResult]:
        """Run ``(specialist, task, index)`` tuples concurrently, normalizing failures.

        Reuses the shared per-member lane helper (``_execute_specialist`` → the streaming
        ``member_event_hook``); a specialist exception becomes a failed ``MemberResult`` tagged
        with its 1-based index, matching the prior inline gather.
        """
        gathered = await asyncio.gather(
            *[self._execute_specialist(sp, tk, context, idx) for sp, tk, idx in pairs],
            return_exceptions=True,
        )
        out: List[MemberResult] = []
        for (sp, _tk, idx), result in zip(pairs, gathered):
            if isinstance(result, Exception):
                logger.error(f"HierarchicalFormation: specialist {sp.id} failed: {result}")
                out.append(
                    MemberResult(
                        member_id=sp.id,
                        success=False,
                        output="",
                        error=str(result),
                        metadata={"index": idx + 1, "role": "specialist"},
                    )
                )
            else:
                out.append(result)
        return out

    async def _execute_specialist(
        self,
        specialist: Any,
        task: AgentMessage,
        context: TeamContext,
        index: int,
    ) -> MemberResult:
        """Execute a single specialist, emitting per-member lane events (ADR-023).

        Specialists gather concurrently; the sink ContextVar propagates into the tasks, so
        reusing the shared helper gives per-member lanes without touching the gather flow.
        """
        logger.debug(f"HierarchicalFormation: executing specialist {index + 1}: {specialist.id}")
        member_event_hook = getattr(context, "member_event_hook", None)
        return await self._execute_member_with_events(
            specialist, task, context, index + 1, member_event_hook=member_event_hook
        )

    def validate_context(self, context: TeamContext) -> bool:
        """Hierarchical formation requires delegation support."""
        return context is not None and hasattr(context, "shared_state")

    def supports_early_termination(self) -> bool:
        """Hierarchical formation requires all specialists to complete."""
        return False
