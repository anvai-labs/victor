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

"""Sequential formation strategy.

Agents execute one after another, with context chaining:
output of agent N becomes input for agent N+1.
"""

import logging
from typing import Any, Dict, List

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult, MessageType

logger = logging.getLogger(__name__)


class SequentialFormation(BaseFormationStrategy):
    """Execute agents sequentially with context accumulation.

    In sequential formation:
    - All agents receive the same task
    - Agent 1 executes with initial context
    - Agent 2 executes with Agent 1's output in context
    - Agent N executes with all previous outputs in context
    - Context accumulates through shared_state

    Use case: Multi-perspective analysis, sequential review
    """

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute agents sequentially with context chaining."""
        results: List[MemberResult] = []
        previous_output = None
        previous_agent_id = None

        # ADR-023: resume from a prior member checkpoint — pre-seed completed
        # members and skip them below. Fully opt-in (None → unchanged behavior).
        completed_ids: set = set()
        resume = getattr(context, "resume_completed", None)
        if resume:
            for raw in resume.get("member_results") or []:
                results.append(
                    raw if isinstance(raw, MemberResult) else MemberResult.from_dict(raw)
                )
            completed_ids = set(resume.get("member_ids") or [r.member_id for r in results])
            previous_output = resume.get("last_output")
            previous_agent_id = resume.get("last_agent_id")

        checkpoint_hook = getattr(context, "checkpoint_hook", None)
        # ADR-023: per-member streaming — emit lifecycle events when a sink is wired.
        member_event_hook = getattr(context, "member_event_hook", None)
        # ADR-023 pillar 2b: durable pause when a member reports awaiting-approval.
        pause_hook = getattr(context, "pause_hook", None)

        for i, agent in enumerate(agents):
            if agent.id in completed_ids:
                logger.debug(f"SequentialFormation: skipping completed member {agent.id} (resume)")
                continue

            logger.debug(f"SequentialFormation: executing agent {i+1}/{len(agents)}: {agent.id}")

            if member_event_hook is not None:
                await member_event_hook("member_start", agent.id, i)

            # Add previous output and agent to context
            if previous_output:
                context.shared_state["previous_output"] = previous_output
            if previous_agent_id is not None:
                context.shared_state["previous_agent"] = previous_agent_id

            # Create task for this agent with previous_agent in data
            agent_task = task
            if previous_agent_id is not None:
                # Create a copy of task data with previous_agent
                task_data = dict(task.data)
                task_data["previous_agent"] = previous_agent_id
                agent_task = AgentMessage(
                    sender_id=task.sender_id,
                    content=task.content,
                    message_type=task.message_type,
                    recipient_id=agent.id,
                    data=task_data,
                    timestamp=task.timestamp,
                    reply_to=task.reply_to,
                    priority=task.priority,
                )

            # Execute agent with task
            try:
                result = await agent.execute(agent_task, context)

                # ADR-023 pillar 2b: a member awaiting human approval durably pauses the
                # formation (opt-in on a pause_hook). The paused member is NOT appended, so a
                # resumed run re-executes it; completed members are checkpointed and skipped.
                if pause_hook is not None and result.metadata.get("awaiting_approval"):
                    logger.info(
                        f"SequentialFormation: member {agent.id} awaiting approval; pausing"
                    )
                    approval_request = result.metadata.get("approval_request") or {}
                    detail = str(
                        approval_request.get("title") or approval_request.get("tool_name") or ""
                    )
                    if member_event_hook is not None:
                        await member_event_hook(
                            "member_awaiting_approval", agent.id, i, content=detail
                        )
                    context.shared_state["__awaiting_approval__"] = {
                        "member_id": agent.id,
                        "index": i,
                        "approval_request": result.metadata.get("approval_request"),
                    }
                    await pause_hook(i, result, results, context.shared_state)
                    return results

                results.append(result)

                # Store output and agent ID for next agent's context
                if result.success and result.output:
                    previous_output = result.output
                    previous_agent_id = agent.id

            except Exception as e:
                logger.error(f"SequentialFormation: agent {agent.id} failed: {e}")
                result = MemberResult(
                    member_id=agent.id,
                    success=False,
                    output="",
                    error=str(e),
                    metadata={"index": i},
                )
                results.append(result)
                # Continue with next agent even if one fails

            # ADR-023: per-member streaming — completed/error lifecycle event.
            if member_event_hook is not None:
                await member_event_hook(
                    "member_completed" if result.success else "member_error",
                    agent.id,
                    i,
                    success=result.success,
                    content="" if result.success else (result.error or ""),
                )

            # ADR-023: checkpoint after each member (success or failure).
            if checkpoint_hook is not None:
                await checkpoint_hook(i, result, results, context.shared_state)

        return results

    def validate_context(self, context: TeamContext) -> bool:
        """Sequential formation requires minimal context."""
        return context is not None

    def supports_early_termination(self) -> bool:
        """Sequential formation can terminate on first failure."""
        return True

    def supports_durable_pause(self) -> bool:
        """SEQUENTIAL implements ADR-023 durable pause (stop + checkpoint on awaiting-approval)."""
        return True
