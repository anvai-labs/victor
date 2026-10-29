"""State-passed system prompt coordinator (SPA-3).

Provides the canonical state-passed task-classification seam for prompt
handling. Prompt building itself stays on the service-owned prompt pipeline,
but classification results should be stored as state transitions for
downstream coordinators.

Usage:
    snapshot = create_snapshot(orchestrator)
    coord = SystemPromptStatePassedCoordinator(task_analyzer)
    result = await coord.classify(snapshot, user_message)
    # result.transitions has task_type, complexity, keywords in state
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from victor.agent.coordinators.state_context import (
    CoordinatorResult,
    ContextSnapshot,
    TransitionBatch,
)

if TYPE_CHECKING:
    from victor.agent.task_analyzer import TaskAnalyzer

logger = logging.getLogger(__name__)


class SystemPromptStatePassedCoordinator:
    """State-passed wrapper for system prompt task classification.

    Takes a ContextSnapshot (immutable) and returns a CoordinatorResult
    with task classification transitions. No orchestrator reference,
    no direct state mutation.
    """

    def __init__(self, task_analyzer: "TaskAnalyzer") -> None:
        self._task_analyzer = task_analyzer

    async def classify(
        self,
        context: ContextSnapshot,
        user_message: str,
    ) -> CoordinatorResult:
        """Classify user message and return classification as transitions.

        Reads conversation context from snapshot, delegates to TaskAnalyzer,
        and returns results as UPDATE_STATE transitions.

        Args:
            context: Immutable snapshot of orchestrator state
            user_message: The user's input message

        Returns:
            CoordinatorResult with task classification transitions
        """
        # Build conversation history from snapshot messages
        history = self._extract_history(context)

        # Classify with context (pure read operation)
        classification = self._task_analyzer.classify_task_with_context(user_message, history)

        if not classification:
            return CoordinatorResult.no_op(reasoning="No classification result")

        # Convert classification to state transitions
        batch = TransitionBatch()

        task_type = classification.get("task_type", "general")
        batch.update_state("task_type", task_type, scope="conversation")

        complexity = classification.get("complexity", "medium")
        batch.update_state("task_complexity", complexity, scope="conversation")

        keywords = classification.get("keywords", [])
        if keywords:
            batch.update_state("task_keywords", keywords, scope="conversation")

        confidence = classification.get("confidence", 0.5)

        return CoordinatorResult(
            transitions=batch,
            reasoning=f"Classified as {task_type} (complexity={complexity})",
            confidence=float(confidence),
            metadata=classification,
        )

    @staticmethod
    def _extract_history(context: ContextSnapshot) -> List[Dict[str, Any]]:
        """Extract conversation history from snapshot messages.

        Converts immutable message tuples to dict format expected by
        TaskAnalyzer.classify_task_with_context().
        """
        history = []
        for msg in context.messages[-10:]:  # Last 10 messages for context
            if hasattr(msg, "role") and hasattr(msg, "content"):
                history.append({"role": msg.role, "content": msg.content})
        return history
