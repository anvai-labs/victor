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

"""Pipeline formation strategy.

Output of each agent feeds directly into the next agent.
Each agent transforms the data, creating a processing pipeline.
"""

import logging
from typing import Any, Dict, List

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult, MessageType

logger = logging.getLogger(__name__)


class PipelineFormation(BaseFormationStrategy):
    """Execute agents as a processing pipeline.

    In pipeline formation:
    - Agent 1 processes initial data
    - Agent 2 transforms Agent 1's output
    - Agent N transforms Agent N-1's output
    - Each agent must produce output for next agent
    - Similar to Unix pipes: agent1 | agent2 | agent3

    Use case: Multi-stage data processing, transformation pipelines
    """

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute agents as a pipeline with ADR-023 durability.

        Like SEQUENTIAL, a durable run (checkpointer configured) checkpoints after each
        stage, resumes by skipping completed stages, durably pauses on an awaiting-approval
        stage, and streams per-member lanes — reusing the shared
        :meth:`_execute_member_with_events` helper + the coordinator hooks. Pipeline-specific:
        stages **stop** on failure or empty output (a broken stage can't feed the next).
        """
        results: List[MemberResult] = []
        previous_output = None

        # ADR-023: resume from a prior stage checkpoint — pre-seed completed stages + skip.
        completed_ids: set = set()
        resume = getattr(context, "resume_completed", None)
        if resume:
            for raw in resume.get("member_results") or []:
                results.append(
                    raw if isinstance(raw, MemberResult) else MemberResult.from_dict(raw)
                )
            completed_ids = set(resume.get("member_ids") or [r.member_id for r in results])
            previous_output = resume.get("last_output")

        member_event_hook = getattr(context, "member_event_hook", None)
        checkpoint_hook = getattr(context, "checkpoint_hook", None)
        pause_hook = getattr(context, "pause_hook", None)

        for i, agent in enumerate(agents):
            if agent.id in completed_ids:
                logger.debug(f"PipelineFormation: skipping completed stage {agent.id} (resume)")
                continue

            logger.debug(f"PipelineFormation: stage {i+1}/{len(agents)}: {agent.id}")

            # Add previous output to context (pipeline chaining).
            if previous_output is not None:
                context.shared_state["previous_output"] = previous_output

            # Shared helper: emits member_start → executes → emits the terminal lane event
            # (completed / error / awaiting); normalizes exceptions to a failed MemberResult.
            result = await self._execute_member_with_events(
                agent, task, context, i, member_event_hook=member_event_hook
            )

            # ADR-023 pillar 2b: durable pause — an awaiting-approval stage stops + checkpoints
            # (via pause_hook) and is NOT appended, so a resumed run re-executes it.
            if pause_hook is not None and (result.metadata or {}).get("awaiting_approval"):
                logger.info(f"PipelineFormation: stage {agent.id} awaiting approval; pausing")
                context.shared_state["__awaiting_approval__"] = {
                    "member_id": agent.id,
                    "index": i,
                    "approval_request": result.metadata.get("approval_request"),
                }
                await pause_hook(i, result, results, context.shared_state)
                return results

            results.append(result)

            # ADR-023: checkpoint after each stage (success or failure).
            if checkpoint_hook is not None:
                await checkpoint_hook(i, result, results, context.shared_state)

            if not result.success:
                logger.error(f"PipelineFormation: stage {i} failed, stopping pipeline")
                break

            # Store output for the next stage; an empty output can't feed the pipeline.
            if result.output:
                previous_output = result.output
            else:
                logger.warning(f"PipelineFormation: stage {i} produced no output, stopping")
                break

        return results

    def validate_context(self, context: TeamContext) -> bool:
        """Pipeline formation requires minimal context."""
        return context is not None

    def supports_early_termination(self) -> bool:
        """Pipeline formation terminates on first failure."""
        return True

    def supports_durable_pause(self) -> bool:
        """PIPELINE is sequential — it implements ADR-023 durable pause (stop + checkpoint)."""
        return True
