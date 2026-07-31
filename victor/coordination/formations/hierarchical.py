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

import logging
from typing import Any, Dict, List, Optional

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
        """Execute with supervisor-specialist pattern."""
        if len(agents) < 2:
            raise ValueError(
                "Hierarchical formation requires at least 2 agents " "(supervisor + specialists)"
            )

        # Check for explicit supervisor in context first. Keep the older
        # explicit_manager_id key readable for serialized compatibility.
        explicit_supervisor_id = context.shared_state.get(
            "explicit_supervisor_id",
            context.shared_state.get("explicit_manager_id"),
        )

        # Auto-detect supervisor: prefer the explicit category/contract, then
        # fall back to legacy manager/lead role signals.
        supervisor = None
        specialists = []

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
        else:
            if explicit_supervisor_id:
                logger.info(f"HierarchicalFormation: using explicit supervisor={supervisor.id}")
            else:
                logger.info(f"HierarchicalFormation: auto-detected supervisor={supervisor.id}")

        logger.debug(
            f"HierarchicalFormation: supervisor={supervisor.id}, "
            f"specialists={[s.id for s in specialists]}"
        )

        # Phase 1: Supervisor plans and delegates (executes first).
        supervisor_result = await supervisor.execute(task, context)

        # Create results list with supervisor first, followed by specialists in original order.
        results = []
        results.append(supervisor_result)

        # Check if supervisor created delegation tasks.
        if not supervisor_result.success or not supervisor_result.metadata.get("delegated_tasks"):
            logger.info(
                "HierarchicalFormation: supervisor did not delegate tasks, "
                "executing all specialists with original task"
            )
            # Fallback: Execute all specialists with the original task.
            specialist_tasks = [
                self._execute_specialist(specialist, task, context, i)
                for i, specialist in enumerate(specialists)
            ]

            # Execute specialists in parallel.
            import asyncio

            specialist_results = await asyncio.gather(
                *specialist_tasks,
                return_exceptions=True,
            )

            # Process specialist results.
            for i, result in enumerate(specialist_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"HierarchicalFormation: specialist {specialists[i].id} "
                        f"failed: {result}"
                    )
                    results.append(
                        MemberResult(
                            member_id=specialists[i].id,
                            success=False,
                            output="",
                            error=str(result),
                            metadata={"index": i + 1, "role": "specialist"},
                        )
                    )
                else:
                    results.append(result)

            return results

        # Phase 2: Specialists execute delegated tasks in parallel.
        delegated_tasks = supervisor_result.metadata["delegated_tasks"]

        if len(delegated_tasks) != len(specialists):
            logger.warning(
                f"HierarchicalFormation: task count mismatch: "
                f"{len(delegated_tasks)} tasks vs {len(specialists)} specialists"
            )

        specialist_tasks = []
        for i, specialist in enumerate(specialists):
            if i < len(delegated_tasks):
                specialist_task = delegated_tasks[i]
                specialist_tasks.append(
                    self._execute_specialist(specialist, specialist_task, context, i)
                )

        # Execute specialists in parallel.
        import asyncio

        specialist_results = await asyncio.gather(
            *specialist_tasks,
            return_exceptions=True,
        )

        # Process specialist results.
        for i, result in enumerate(specialist_results):
            if isinstance(result, Exception):
                logger.error(
                    f"HierarchicalFormation: specialist {specialists[i].id} " f"failed: {result}"
                )
                results.append(
                    MemberResult(
                        member_id=specialists[i].id,
                        success=False,
                        output="",
                        error=str(result),
                        metadata={"index": i + 1, "role": "specialist"},
                    )
                )
            else:
                results.append(result)

        # Phase 3: Supervisor synthesizes results.
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

        return results

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
