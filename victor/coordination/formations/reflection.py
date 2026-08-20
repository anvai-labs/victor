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

"""Reflection formation for iterative refinement with feedback.

This module provides ReflectionFormation, which implements the reflection
loop pattern: generator → critic → refinement → (repeat).

Useful for:
- Code review and refinement
- Iterative problem solving
- Quality improvement through feedback

SOLID Principles:
- SRP: Reflection logic only
- OCP: Extensible via max_iterations parameter
- LSP: Substitutable with other formations
- DIP: Depends on TeamContext and BaseFormationStrategy abstractions

Usage:
    from victor.coordination.formations.reflection import ReflectionFormation
    from victor.coordination.formations.base import TeamContext

    # Create formation with max iterations
    formation = ReflectionFormation(max_iterations=3)

    # Create context with generator and critic agents
    context = TeamContext("team-1", "reflection")
    context.set("generator", generator_agent)
    context.set("critic", critic_agent)

    # Execute with reflection loop
    results = await formation.execute(agents, context, task)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult, MessageType

logger = logging.getLogger(__name__)

# Explicit verdict line the critic is asked to emit so satisfaction is judged by
# the critic LLM rather than inferred from incidental positive words.
_VERDICT_RE = re.compile(r"verdict\W+(satisfied|needs[ _]work)", re.IGNORECASE)
# Negations immediately preceding a satisfaction keyword (keyword fallback only).
_NEGATION_RE = re.compile(r"\b(not|no|never|isn't|aren't|wasn't|n't)\b\W*$", re.IGNORECASE)


class ReflectionFormation(BaseFormationStrategy):
    """Implements reflection loop pattern for iterative refinement.

    Formation Pattern:
        Generator Agent → Critic Agent → Refinement → (repeat)

    The generator produces a solution, the critic provides feedback,
    and the generator refines based on the feedback. This continues
    until the critic is satisfied or max iterations is reached.

    Useful for:
    - Code review and iterative improvement
    - Problem refinement through feedback
    - Quality assurance cycles

    SOLID: SRP (reflection logic only), OCP (extensible via max_iterations)

    Attributes:
        max_iterations: Maximum number of refinement cycles
        satisfaction_keywords: Keywords that indicate critic satisfaction

    Example:
        >>> formation = ReflectionFormation(max_iterations=3)
        >>> context = TeamContext("team-1", "reflection")
        >>> context.set("generator", generator_agent)
        >>> context.set("critic", critic_agent)
        >>>
        >>> results = await formation.execute(
        ...     agents=[generator_agent, critic_agent],
        ...     context=context,
        ...     task=AgentMessage("Refactor this code")
        ... )
        >>>
        >>> # Results include iteration count and final feedback
        >>> print(f"Iterations: {results[0].metadata['iterations']}")
        >>> print(f"Final feedback: {results[0].metadata['final_feedback']}")
    """

    def __init__(
        self,
        max_iterations: int = 3,
        satisfaction_keywords: Optional[List[str]] = None,
    ):
        """Initialize the reflection formation.

        Args:
            max_iterations: Maximum number of refinement cycles (default: 3)
            satisfaction_keywords: Keywords that indicate critic satisfaction
                (default: ["good", "satisfactory", "acceptable", "approved", "excellent"])
        """
        self.max_iterations = max_iterations
        self.satisfaction_keywords = satisfaction_keywords or [
            "good",
            "satisfactory",
            "acceptable",
            "approved",
            "excellent",
            "perfect",
            "great",
        ]

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute reflection loop: generator → critic → refine → repeat.

        Args:
            agents: List of agents (should contain generator and critic)
            context: Team context with agent references
            task: Task message to process

        Returns:
            List of MemberResult with metadata:
                - result: Final generated result
                - iterations: Number of iterations performed
                - final_feedback: Critic's final feedback
                - satisfied: Whether critic was satisfied
        """
        # Get agents from context
        generator = context.get("generator")
        critic = context.get("critic")

        if not generator or not critic:
            logger.error("ReflectionFormation requires 'generator' and 'critic' agents in context")
            return [
                MemberResult(
                    member_id="reflection_formation",
                    success=False,
                    output="",
                    error="Missing generator or critic agent in context",
                )
            ]

        result = None
        feedback = None

        # Preserve the original task so every critique judges fitness against
        # what was actually asked (not just the candidate solution in isolation),
        # and so refinement rounds keep the goal in view.
        original_task = task.content

        # Per-run iteration bound: a team may override via context
        # ("reflection_max_iterations") so the shared formation instance stays
        # stateless and concurrency-safe. Falls back to the instance default.
        max_iterations = context.get("reflection_max_iterations") or self.max_iterations

        # ADR-023 per-member streaming lanes. Reflection's generator/critic use a different
        # execute signature (str + shared_state) and produce one aggregate result, so the
        # shared helper doesn't fit — emit via the same coordinator hook inline instead.
        member_event_hook = getattr(context, "member_event_hook", None)

        # ADR-023 iteration-granular durable checkpoint/resume (opt-in). Iterations are sequential
        # (each refines from the previous critique), so after each the loop state — completed
        # iterations, the refined generator input, and the last result/feedback — is snapshotted
        # into shared_state["__reflection__"] (persisted by the existing member checkpoint hook).
        # A crash resumes at the next unfinished iteration; completed iterations are not re-run.
        checkpoint_hook = getattr(context, "checkpoint_hook", None)
        resume = getattr(context, "resume_completed", None)
        saved: Dict[str, Any] = dict(
            ((resume or {}).get("shared_state") or {}).get("__reflection__") or {}
        )

        if saved.get("done"):
            # Terminal resume: the loop already finished — rebuild the final aggregate result.
            return [
                self._final_result(
                    saved.get("result"), saved.get("feedback"), int(saved["iter_done"])
                )
            ]

        iter_start = int(saved.get("iter_done", 0))
        if iter_start > 0:
            result = saved.get("result")
            feedback = saved.get("feedback")
            if saved.get("task_content") is not None:
                task = AgentMessage(
                    sender_id="reflection_formation",
                    content=saved["task_content"],
                    message_type=MessageType.TASK,
                )
            logger.info(
                "ReflectionFormation: resume — continuing from iteration %d/%d",
                iter_start + 1,
                max_iterations,
            )

        async def _checkpoint(iter_done: int, task_content: Optional[str], done: bool) -> None:
            """Snapshot the iteration loop into shared_state and persist via the member hook."""
            if checkpoint_hook is None:
                return
            context.shared_state["__reflection__"] = {
                "iter_done": iter_done,
                "task_content": task_content,
                "result": str(result) if result is not None else None,
                "feedback": feedback,
                "done": done,
            }
            marker = MemberResult(
                member_id="reflection_formation",
                success=True,
                output=str(result) if result else "",
                metadata={"iterations": iter_done},
            )
            await checkpoint_hook(iter_done - 1, marker, [marker], context.shared_state)

        # Reflection loop
        iteration = iter_start - 1
        for iteration in range(iter_start, max_iterations):
            logger.info(f"Reflection iteration {iteration + 1}/{max_iterations}")

            # Generate solution
            if member_event_hook is not None:
                await member_event_hook("member_start", generator.id, iteration)
            try:
                generator_response = await generator.execute(
                    task.content, context=context.shared_state
                )
                result = generator_response
            except Exception as e:
                logger.error(f"Generator failed in iteration {iteration + 1}: {e}")
                if member_event_hook is not None:
                    await member_event_hook(
                        "member_error", generator.id, iteration, success=False, content=str(e)
                    )
                return [
                    MemberResult(
                        member_id=generator.id,
                        success=False,
                        output=str(result) if result else "",
                        error=f"Generator failed: {str(e)}",
                    )
                ]
            if member_event_hook is not None:
                await member_event_hook("member_completed", generator.id, iteration, success=True)

            # Critique the solution against the original task.
            critique_prompt = self._build_critique_prompt(original_task, result)
            if member_event_hook is not None:
                await member_event_hook("member_start", critic.id, iteration)
            try:
                critique_response = await critic.execute(
                    critique_prompt, context=context.shared_state
                )
                feedback = critique_response
                if member_event_hook is not None:
                    await member_event_hook("member_completed", critic.id, iteration, success=True)
            except Exception as e:
                logger.warning(f"Critic failed in iteration {iteration + 1}: {e}")
                feedback = f"Critique unavailable: {str(e)}"
                if member_event_hook is not None:
                    await member_event_hook(
                        "member_error", critic.id, iteration, success=False, content=str(e)
                    )

            # Check if satisfied (critic verdict preferred; keyword fallback)
            if self._is_satisfied(feedback):
                logger.info(f"Critic satisfied after {iteration + 1} iterations")
                break

            # Refine with feedback for next iteration
            task = AgentMessage(
                sender_id="reflection_formation",
                content=self._build_refine_prompt(original_task, result, feedback),
                message_type=MessageType.TASK,
            )
            # ADR-023: checkpoint the completed (unsatisfied) iteration + the refined input.
            await _checkpoint(iteration + 1, task.content, done=False)

        # ADR-023: terminal checkpoint so a resume after completion returns without re-running.
        await _checkpoint(iteration + 1, None, done=True)

        # Return final result with metadata
        return [self._final_result(result, feedback, iteration + 1)]

    def _final_result(self, result: Any, feedback: Optional[str], iterations: int) -> MemberResult:
        """Build the reflection formation's single aggregate result (shared by resume)."""
        return MemberResult(
            member_id="reflection_formation",
            success=True,
            output=str(result) if result else "",
            metadata={
                "iterations": iterations,
                "final_feedback": feedback,
                "satisfied": self._is_satisfied(feedback),
                "formation": "reflection",
            },
        )

    def validate_context(self, context: TeamContext) -> bool:
        """Validate that context has required generator and critic.

        Args:
            context: Team context to validate

        Returns:
            True if context has 'generator' and 'critic' keys
        """
        generator = context.get("generator")
        critic = context.get("critic")

        has_generator = generator is not None
        has_critic = critic is not None

        if not has_generator:
            logger.warning("ReflectionFormation context missing 'generator' agent")

        if not has_critic:
            logger.warning("ReflectionFormation context missing 'critic' agent")

        return has_generator and has_critic

    def get_required_roles(self) -> Optional[List[str]]:
        """Get required roles for reflection formation.

        Returns:
            List of required role names: ['generator', 'critic']
        """
        return ["generator", "critic"]

    def supports_early_termination(self) -> bool:
        """Check if formation supports early termination.

        Reflection formation can terminate early if critic is satisfied.

        Returns:
            True (supports early termination)
        """
        return True

    def consumes_context_agents(self) -> bool:
        """Reflection reads its 'generator' and 'critic' agents from context."""
        return True

    @staticmethod
    def _build_critique_prompt(original_task: str, result: Any) -> str:
        """Build a critique prompt that judges the solution against the task.

        Includes the original task so the critic assesses fitness-for-purpose,
        and requests an explicit verdict line so satisfaction can be judged by
        the critic rather than inferred from incidental positive words.
        """
        return (
            "You are reviewing a candidate solution against the original task.\n\n"
            f"ORIGINAL TASK:\n{original_task}\n\n"
            f"CANDIDATE SOLUTION:\n{result}\n\n"
            "Assess whether the solution fully and correctly satisfies the "
            "original task. Give specific, actionable feedback on anything that "
            "is missing, incorrect, or could be improved.\n"
            "End your reply with a final line exactly of the form "
            "'VERDICT: SATISFIED' if no further work is needed, or "
            "'VERDICT: NEEDS_WORK' otherwise."
        )

    @staticmethod
    def _build_refine_prompt(original_task: str, result: Any, feedback: Optional[str]) -> str:
        """Build a refinement prompt that keeps the original task in view."""
        return (
            "Refine your solution to better satisfy the original task.\n\n"
            f"ORIGINAL TASK:\n{original_task}\n\n"
            f"CURRENT SOLUTION:\n{result}\n\n"
            f"REVIEWER FEEDBACK:\n{feedback}\n\n"
            "Produce an improved solution that addresses the feedback while "
            "fully satisfying the original task."
        )

    def _is_satisfied(self, feedback: Optional[str]) -> bool:
        """Check whether critic feedback indicates satisfaction.

        Prefers the critic's explicit ``VERDICT: SATISFIED`` / ``NEEDS_WORK``
        line (LLM-judged). Falls back to satisfaction-keyword matching only when
        no verdict line is present, with a guard so a negated keyword (e.g.
        "not good") does not count as satisfaction.

        Args:
            feedback: Critic's feedback text

        Returns:
            True if the critic is satisfied.
        """
        if not feedback:
            return False

        verdict = self._parse_verdict(feedback)
        if verdict is not None:
            return verdict

        return self._keyword_satisfied(feedback)

    @staticmethod
    def _parse_verdict(feedback: str) -> Optional[bool]:
        """Return True/False from an explicit VERDICT line, or None if absent."""
        match = _VERDICT_RE.search(feedback)
        if not match:
            return None
        return match.group(1).lower().startswith("satisfied")

    def _keyword_satisfied(self, feedback: str) -> bool:
        """Heuristic fallback: a non-negated satisfaction keyword is present."""
        text = feedback.lower()
        for keyword in self.satisfaction_keywords:
            for hit in re.finditer(r"\b" + re.escape(keyword.lower()) + r"\b", text):
                preceding = text[max(0, hit.start() - 12) : hit.start()]
                if _NEGATION_RE.search(preceding):
                    continue
                return True
        return False


__all__ = ["ReflectionFormation"]
