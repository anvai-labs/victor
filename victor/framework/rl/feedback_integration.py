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

"""Integration layer for implicit feedback collection.

This module provides a bridge between the ImplicitFeedbackCollector
and other system components, enabling easy adoption without requiring
extensive modifications to existing code.

Usage:
    # At session start
    integration = FeedbackIntegration.get_instance()
    session = integration.start_tracking(
        session_id="...",
        task_type="analysis",
        provider="anthropic",
        model="claude-3-opus"
    )

    # During execution (can be called from anywhere)
    integration.record_tool(session_id, "code_search", True, 150.0)
    integration.record_grounding(session_id, 0.85)

    # At session end
    feedback = integration.end_tracking(session_id, completed=True)

    # Feedback is automatically distributed to relevant learners

Sprint 4: Implicit Feedback Enhancement
"""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from victor.framework.rl.implicit_feedback import (
    ImplicitFeedback,
    SessionContext,
    get_feedback_collector,
)
from victor.framework.rl.base import RLOutcome

if TYPE_CHECKING:
    from victor.agent.response_quality import QualityResult
    from victor.framework.rl.coordinator import RLCoordinator

logger = logging.getLogger(__name__)


class FeedbackIntegration:
    """Integration layer for implicit feedback collection.

    Provides a simplified interface for recording feedback from
    various system components and distributing it to RL learners.
    """

    _instance: Optional["FeedbackIntegration"] = None

    @classmethod
    def get_instance(cls) -> "FeedbackIntegration":
        """Get singleton instance.

        Returns:
            FeedbackIntegration singleton
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize feedback integration."""
        self._collector = get_feedback_collector()
        self._rl_coordinator: Optional["RLCoordinator"] = None
        self._enabled = True

        # Session tracking
        self._active_sessions: Dict[str, SessionContext] = {}

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable feedback collection.

        Args:
            enabled: Whether to collect feedback
        """
        self._enabled = enabled
        logger.info(f"FeedbackIntegration {'enabled' if enabled else 'disabled'}")

    def _get_coordinator(self) -> Optional["RLCoordinator"]:
        """Lazy load RL coordinator.

        Returns:
            RLCoordinator instance or None
        """
        if self._rl_coordinator is None:
            try:
                from victor.framework.rl.coordinator import get_rl_coordinator

                self._rl_coordinator = get_rl_coordinator()
            except Exception as e:
                logger.debug(f"Failed to get RL coordinator: {e}")

        return self._rl_coordinator

    def start_tracking(
        self,
        session_id: str,
        task_type: str = "general",
        provider: str = "",
        model: str = "",
        max_iterations: int = 30,
    ) -> Optional[SessionContext]:
        """Start tracking a session.

        Args:
            session_id: Unique session identifier
            task_type: Type of task
            provider: LLM provider
            model: Model name
            max_iterations: Maximum iterations

        Returns:
            SessionContext or None if disabled
        """
        if not self._enabled:
            return None

        session = self._collector.start_session(
            session_id=session_id,
            task_type=task_type,
            provider=provider,
            model=model,
            max_iterations=max_iterations,
        )

        self._active_sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get an active session.

        Args:
            session_id: Session identifier

        Returns:
            SessionContext or None
        """
        return self._active_sessions.get(session_id)

    def record_tool(
        self,
        session_id: str,
        tool_name: str,
        success: bool,
        execution_time_ms: float,
        error_message: Optional[str] = None,
        is_retry: bool = False,
    ) -> None:
        """Record a tool execution.

        Args:
            session_id: Session identifier
            tool_name: Name of the tool
            success: Whether it succeeded
            execution_time_ms: Execution duration
            error_message: Optional error message
            is_retry: Whether this is a retry
        """
        if not self._enabled:
            return

        session = self._active_sessions.get(session_id)
        if session:
            self._collector.record_tool_execution(
                session=session,
                tool_name=tool_name,
                success=success,
                execution_time_ms=execution_time_ms,
                error_message=error_message,
                is_retry=is_retry,
            )

    def record_grounding(self, session_id: str, confidence: float) -> None:
        """Record a grounding verification result.

        Args:
            session_id: Session identifier
            confidence: Grounding confidence score
        """
        if not self._enabled:
            return

        session = self._active_sessions.get(session_id)
        if session:
            self._collector.record_grounding_result(session, confidence)

    def record_quality(self, session_id: str, quality_result: "QualityResult") -> None:
        """Record a quality scoring result.

        Args:
            session_id: Session identifier
            quality_result: Quality result from scorer
        """
        if not self._enabled:
            return

        session = self._active_sessions.get(session_id)
        if session:
            self._collector.record_quality_result(session, quality_result)

            # Also update quality weights learner
            self._record_quality_for_learning(session, quality_result)

    def _record_quality_for_learning(
        self, session: SessionContext, quality_result: "QualityResult"
    ) -> None:
        """Record quality result for weight learning.

        Args:
            session: Session context
            quality_result: Quality result to record
        """
        coordinator = self._get_coordinator()
        if not coordinator:
            return

        try:
            # Extract dimension scores
            dimension_scores = {}
            for ds in quality_result.dimension_scores:
                dimension_scores[ds.dimension.value] = ds.score

            outcome = RLOutcome(
                provider=session.provider,
                model=session.model,
                task_type=session.task_type,
                success=quality_result.passes_threshold,
                quality_score=quality_result.overall_score,
                metadata={
                    "dimension_scores": dimension_scores,
                    "overall_success": quality_result.overall_score,
                },
            )

            coordinator.record_outcome("quality_weights", outcome, "coding")

        except Exception as e:
            logger.debug(f"Failed to record quality for learning: {e}")

    def record_iteration(self, session_id: str) -> None:
        """Record an iteration.

        Args:
            session_id: Session identifier
        """
        if not self._enabled:
            return

        session = self._active_sessions.get(session_id)
        if session:
            self._collector.record_iteration(session)

    def record_workflow(
        self, session_id: str, started: bool = True, completed: bool = False
    ) -> None:
        """Record workflow pattern.

        Args:
            session_id: Session identifier
            started: Whether pattern started
            completed: Whether pattern completed
        """
        if not self._enabled:
            return

        session = self._active_sessions.get(session_id)
        if session:
            self._collector.record_workflow_pattern(session, started, completed)

    def end_tracking(self, session_id: str, completed: bool = False) -> Optional[ImplicitFeedback]:
        """End session tracking and compute feedback.

        Args:
            session_id: Session identifier
            completed: Whether task completed successfully

        Returns:
            Computed ImplicitFeedback or None
        """
        if not self._enabled:
            return None

        session = self._active_sessions.pop(session_id, None)
        if not session:
            return None

        feedback = self._collector.end_session(session, completed)

        # Distribute feedback to relevant learners
        self._distribute_feedback(feedback)

        return feedback

    def _distribute_feedback(self, feedback: ImplicitFeedback) -> None:
        """Distribute feedback to relevant RL learners.

        Args:
            feedback: Computed feedback to distribute
        """
        coordinator = self._get_coordinator()
        if not coordinator:
            return

        reward = feedback.compute_reward()

        try:
            # Create outcome for general consumption (reserved for future learners)
            _outcome = RLOutcome(  # noqa: F841
                provider=feedback.provider,
                model=feedback.model,
                task_type=feedback.task_type,
                success=feedback.task_completed,
                quality_score=feedback.quality_score,
                metadata={
                    "session_id": feedback.session_id,
                    "tool_success_rate": feedback.tool_success_rate,
                    "grounding_score": feedback.grounding_score,
                    "efficiency_score": feedback.efficiency_score,
                    "retry_count": feedback.retry_count,
                    "duration_seconds": feedback.duration_seconds,
                    "tool_count": feedback.tool_count,
                    "iteration_count": feedback.iteration_count,
                    "computed_reward": reward,
                },
            )

            # Record to continuation patience learner
            outcome_patience = RLOutcome(
                provider=feedback.provider,
                model=feedback.model,
                task_type=feedback.task_type,
                success=feedback.task_completed,
                quality_score=feedback.quality_score,
                metadata={
                    "iteration_count": feedback.iteration_count,
                    "task_completed": feedback.task_completed,
                    "stuck": feedback.retry_count > 3,
                },
            )
            coordinator.record_outcome("continuation_patience", outcome_patience, "coding")

            logger.debug(
                f"FeedbackIntegration: Distributed feedback for session {feedback.session_id[:8]}... "
                f"(reward={reward:.3f})"
            )

        except Exception as e:
            logger.debug(f"Failed to distribute feedback: {e}")

    def get_quality_weights(self, task_type: str) -> Dict[str, float]:
        """Get learned quality weights for a task type.

        Args:
            task_type: Task type

        Returns:
            Dictionary of dimension -> weight
        """
        coordinator = self._get_coordinator()
        if not coordinator:
            return {}

        try:
            learner = coordinator.get_learner("quality_weights")
            if learner:
                rec = learner.get_recommendation("", "", task_type)
                if rec and not rec.is_baseline:
                    return rec.value
        except Exception as e:
            logger.debug(f"Failed to get quality weights: {e}")

        return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback collection statistics.

        Returns:
            Dictionary with stats
        """
        stats = self._collector.get_aggregate_stats()
        stats["active_sessions"] = len(self._active_sessions)
        stats["enabled"] = self._enabled
        return stats


def get_feedback_integration() -> FeedbackIntegration:
    """Get global feedback integration instance.

    Returns:
        FeedbackIntegration singleton
    """
    return FeedbackIntegration.get_instance()
