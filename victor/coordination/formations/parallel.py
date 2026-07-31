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

"""Parallel formation strategy.

Agents execute simultaneously with independent contexts.
All agents receive the same task and work independently.
"""

import copy
import logging
from typing import Any, List

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult

logger = logging.getLogger(__name__)


class ParallelFormation(BaseFormationStrategy):
    """Execute all agents in parallel with independent contexts.

    In parallel formation:
    - All agents receive the same task simultaneously
    - Each agent works independently
    - Results are collected and combined at the end
    - No inter-agent communication during execution

    Use case: Diverse perspectives, redundancy, independent analysis
    """

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute all agents in parallel with isolated contexts.

        Each agent receives a deep copy of the shared state to prevent
        race conditions. After execution, agent contexts are merged back
        into the parent context using last-writer-wins semantics.
        """
        # Create isolated context copies for each agent
        agent_contexts = [
            TeamContext(
                team_id=context.team_id,
                formation=context.formation,
                shared_state=copy.deepcopy(context.shared_state),
                lsp_capability=context.lsp,
                **context.metadata,
            )
            for _ in agents
        ]

        # ADR-023: run members concurrently with per-member streaming lanes + durable
        # checkpoint/resume (a resumed wave skips completed members). The shared runner handles
        # the gather, lane hooks, exception normalization, and the lock-protected cumulative
        # checkpoint; hooks are read off the original team_context (the isolated contexts don't
        # carry them). Member execution stays fully concurrent.
        results = await self._execute_members_concurrently(agents, task, context, agent_contexts)

        # Merge agent contexts back into parent (last-writer-wins per key)
        for agent_ctx in agent_contexts:
            context.update(agent_ctx.shared_state)

        return results

    def validate_context(self, context: TeamContext) -> bool:
        """Parallel formation requires minimal context."""
        return context is not None

    def supports_early_termination(self) -> bool:
        """Parallel formation waits for all agents to complete."""
        return False
