"""State-passed exploration coordinator (SPA-2).

Wraps ExplorationCoordinator with the state-passed pattern:
- Input: ContextSnapshot (immutable) + user_message
- Output: CoordinatorResult with StateTransitions
- No orchestrator reference, no direct state mutation

This demonstrates the state-passed migration pattern on a real coordinator.
The underlying ExplorationCoordinator logic is reused unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from victor.agent.services.exploration_runtime import (
    ExplorationCoordinator,
    ExplorationResult,
)
from victor.agent.coordinators.state_context import (
    CoordinatorResult,
    ContextSnapshot,
    TransitionBatch,
)

logger = logging.getLogger(__name__)


class ExplorationStatePassedCoordinator:
    """State-passed wrapper for ExplorationCoordinator.

    Reads configuration from ContextSnapshot, delegates to the existing
    ExplorationCoordinator, and returns results as StateTransitions.

    Usage:
        snapshot = create_snapshot(orchestrator)
        coordinator = ExplorationStatePassedCoordinator()
        result = await coordinator.explore(snapshot, user_message)
        # result.transitions contains UPDATE_STATE transitions with findings
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        max_results: int = 5,
    ) -> None:
        self._inner = ExplorationCoordinator()
        self._project_root = project_root
        self._max_results = max_results

    async def explore(
        self,
        context: ContextSnapshot,
        user_message: str,
        *,
        project_root: Optional[Path] = None,
        max_results: Optional[int] = None,
    ) -> CoordinatorResult:
        """Run parallel exploration using context snapshot.

        Reads provider/model/settings from the snapshot instead of
        requiring an orchestrator reference.

        Args:
            context: Immutable snapshot of orchestrator state
            user_message: The user's task description

        Returns:
            CoordinatorResult with exploration findings as transitions
        """
        # Read configuration from snapshot (no orchestrator needed)
        provider = context.provider
        model = context.model
        resolved_project_root = project_root or self._project_root or Path(".")
        resolved_max_results = max_results if isinstance(max_results, int) else self._max_results

        # Determine complexity from snapshot capabilities
        complexity = "action"
        if context.has_capability("task_complexity"):
            complexity = context.get_capability_value("task_complexity") or "action"

        # Delegate to existing coordinator (reuse, don't rewrite)
        exploration_result = await self._inner.explore_parallel(
            task_description=user_message,
            project_root=resolved_project_root,
            max_results=resolved_max_results,
            provider=provider,
            model=model,
            complexity=complexity,
        )

        # Convert result to state transitions
        return self._to_coordinator_result(exploration_result)

    def _to_coordinator_result(
        self,
        exploration: ExplorationResult,
    ) -> CoordinatorResult:
        """Convert ExplorationResult to CoordinatorResult with transitions."""
        if not exploration.file_paths and not exploration.summary:
            return CoordinatorResult.no_op(
                reasoning="No exploration results found",
            )

        batch = TransitionBatch()

        # Store discovered files in conversation state
        if exploration.file_paths:
            batch.update_state(
                "explored_files",
                exploration.file_paths,
                scope="conversation",
            )

        # Store exploration summary
        if exploration.summary:
            batch.update_state(
                "exploration_summary",
                exploration.summary,
                scope="conversation",
            )

        # Store metrics
        batch.update_state(
            "exploration_metrics",
            {
                "duration_seconds": exploration.duration_seconds,
                "tool_calls": exploration.tool_calls,
                "files_found": len(exploration.file_paths),
            },
            scope="conversation",
        )

        return CoordinatorResult(
            transitions=batch,
            reasoning=f"Found {len(exploration.file_paths)} files in {exploration.duration_seconds:.1f}s",
            confidence=min(1.0, len(exploration.file_paths) / 3.0),
            metadata={
                "file_paths": exploration.file_paths,
                "summary": exploration.summary,
                "tool_calls": exploration.tool_calls,
                "duration_seconds": exploration.duration_seconds,
            },
        )
