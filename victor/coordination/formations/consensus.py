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

"""Consensus formation strategy.

All agents must agree on result.
Multiple rounds if needed until consensus or timeout.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult, MessageType

logger = logging.getLogger(__name__)


class ConsensusFormation(BaseFormationStrategy):
    """Execute agents until consensus is reached.

    In consensus formation:
    - All agents analyze the same task
    - Results are compared for agreement
    - If no consensus, agents see each other's results and try again
    - Multiple rounds until consensus or max rounds reached

    Use case: Critical decisions, validation, quality assurance
    """

    def __init__(self, max_rounds: int = 1, agreement_threshold: float = 0.7):
        """Initialize consensus formation.

        Args:
            max_rounds: Maximum number of consensus rounds (default: 1 for testing)
            agreement_threshold: Fraction of agents that must agree (0.0-1.0)
        """
        self.max_rounds = max_rounds
        self.agreement_threshold = agreement_threshold

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute agents until consensus reached.

        ADR-023 round-granular durable checkpoint/resume (opt-in — active only when the
        coordinator wired ``checkpoint_hook``/``resume_completed``, i.e. a checkpointer +
        thread_id). Rounds are sequential (each builds its task from the previous round's
        results), so after each round the loop state — completed round count, accumulated
        results, and the next round's rebuilt task — is snapshotted into
        ``shared_state["__consensus__"]`` (persisted by the existing member checkpoint hook). A
        crash resumes at the next unfinished round: completed rounds are restored, not re-run.
        No checkpointer ⇒ byte-identical.
        """
        checkpoint_hook = getattr(context, "checkpoint_hook", None)
        resume = getattr(context, "resume_completed", None)
        saved: Dict[str, Any] = dict(
            ((resume or {}).get("shared_state") or {}).get("__consensus__") or {}
        )

        # Terminal resume: the run already finished — return the persisted final results.
        if saved.get("done"):
            return [MemberResult.from_dict(r) for r in saved.get("final") or []]

        round_start = int(saved.get("round_done", 0))
        all_results: List[MemberResult] = [
            MemberResult.from_dict(r) for r in saved.get("all_results") or []
        ]
        current_task = task
        if round_start > 0 and saved.get("next_task_content") is not None:
            current_task = AgentMessage(
                message_type=MessageType.TASK,
                sender_id="system",
                content=saved["next_task_content"],
                data={"consensus_round": round_start},
            )
            logger.info(
                "ConsensusFormation: resume — continuing from round %d/%d",
                round_start + 1,
                self.max_rounds,
            )

        async def _checkpoint(
            round_done: int,
            next_task_content: Optional[str],
            done: bool,
            final: Optional[List[MemberResult]],
        ) -> None:
            """Snapshot the round loop into shared_state and persist via the member hook."""
            if checkpoint_hook is None:
                return
            context.shared_state["__consensus__"] = {
                "round_done": round_done,
                "all_results": [r.to_dict() for r in all_results],
                "next_task_content": next_task_content,
                "done": done,
                "final": [r.to_dict() for r in final] if final is not None else None,
            }
            marker = (final or all_results or [MemberResult(member_id="consensus", success=True)])[
                -1
            ]
            await checkpoint_hook(round_done - 1, marker, list(all_results), context.shared_state)

        for round_num in range(round_start, self.max_rounds):
            logger.info(f"ConsensusFormation: round {round_num + 1}/{self.max_rounds}")

            # Execute all agents in parallel for this round
            round_tasks = [
                self._execute_agent(agent, current_task, context, round_num) for agent in agents
            ]

            round_results = await asyncio.gather(*round_tasks, return_exceptions=True)

            # Process results
            processed_results = []
            for i, result in enumerate(round_results):
                if isinstance(result, Exception):
                    processed_results.append(
                        MemberResult(
                            member_id=agents[i].id,
                            success=False,
                            output="",
                            error=str(result),
                            metadata={"round": round_num},
                        )
                    )
                else:
                    processed_results.append(result)

            all_results.extend(processed_results)

            # Check for consensus
            consensus = self._check_consensus(processed_results)

            if consensus:
                logger.info(f"ConsensusFormation: consensus reached in round {round_num + 1}")
                # Mark results with consensus metadata
                for r in processed_results:
                    r.metadata["consensus_achieved"] = True
                    r.metadata["consensus_rounds"] = round_num + 1
                await _checkpoint(round_num + 1, None, True, processed_results)
                # Return results from final consensus round
                return processed_results

            # Prepare task for next round with previous results
            current_task = AgentMessage(
                message_type=MessageType.TASK,
                sender_id="system",
                content=str(
                    {
                        "original_task": task.content,
                        "round": round_num + 1,
                        "previous_results": [
                            {
                                "agent_id": r.member_id,
                                "output": r.output,
                                "success": r.success,
                            }
                            for r in processed_results
                        ],
                        "instruction": "Review previous results and reach consensus",
                    }
                ),
                data={"consensus_round": round_num + 1},
            )
            await _checkpoint(round_num + 1, current_task.content, False, None)

        # Max rounds reached without consensus
        logger.warning(f"ConsensusFormation: no consensus after {self.max_rounds} rounds")
        # Return final round results (last round executed)
        final_round_num = self.max_rounds - 1
        final_round_results = [
            r for r in all_results if r.metadata.get("round", 0) == final_round_num
        ]
        # Mark that consensus was not achieved
        for r in final_round_results:
            r.metadata["consensus_achieved"] = False
            r.metadata["consensus_rounds"] = self.max_rounds
        await _checkpoint(self.max_rounds, None, True, final_round_results)
        return final_round_results

    async def _execute_agent(
        self,
        agent: Any,
        task: AgentMessage,
        context: TeamContext,
        round_num: int,
    ) -> MemberResult:
        """Execute a single agent, emitting per-member lane events (ADR-023).

        Members gather concurrently per round; reusing the shared helper gives per-member
        lanes (the index is the round, so multi-round runs read clearly). The sink ContextVar
        propagates into the gather tasks.
        """
        logger.debug(f"ConsensusFormation: round {round_num + 1}, agent {agent.id}")
        member_event_hook = getattr(context, "member_event_hook", None)
        return await self._execute_member_with_events(
            agent, task, context, round_num, member_event_hook=member_event_hook
        )

    def _check_consensus(self, results: List[MemberResult]) -> bool:
        """Check if results indicate consensus.

        Args:
            results: Results from all agents in this round

        Returns:
            True if consensus reached
        """
        if not results:
            return False

        # Filter successful results
        successful = [r for r in results if r.success]

        # Consensus is achieved if all agents succeed
        # (All agents agree to execute successfully)
        if len(successful) == len(results):
            return True

        # Not all agents succeeded, check if enough succeeded for threshold
        if len(successful) < len(results) * self.agreement_threshold:
            return False

        # Simple consensus check: all successful results have similar content
        # (In practice, this might use more sophisticated comparison)
        if len(successful) == 0:
            return False

        # Get first successful result as reference
        reference = successful[0].output

        # Count how many match reference
        matches = 0
        for result in successful:
            if self._content_matches(result.output, reference):
                matches += 1

        # Check if enough matches
        return matches >= len(results) * self.agreement_threshold

    def _content_matches(self, content1: Any, content2: Any) -> bool:
        """Check if two contents match for consensus.

        This is a simple implementation. In practice, you might use:
        - Semantic similarity
        - Fuzzy matching
        - Content-specific comparison logic

        Args:
            content1: First content
            content2: Second content

        Returns:
            True if contents match
        """
        if content1 is None or content2 is None:
            return False

        # For strings, use simple equality
        if isinstance(content1, str) and isinstance(content2, str):
            return content1.lower() == content2.lower()

        # For dicts, compare keys and basic structure
        if isinstance(content1, dict) and isinstance(content2, dict):
            return set(content1.keys()) == set(content2.keys())

        # Default: exact match
        return content1 == content2

    def validate_context(self, context: TeamContext) -> bool:
        """Consensus formation requires shared state for comparing results."""
        return context is not None and hasattr(context, "shared_state")

    def supports_early_termination(self) -> bool:
        """Consensus formation terminates early when consensus reached."""
        return True
