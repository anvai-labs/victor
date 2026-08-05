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

"""Implicit feedback collection for RL reward signals.

This module derives reward signals from observable session behavior
without requiring explicit user feedback, minimizing user friction.

Implicit Signals Collected:
- Tool success rate: Did tools execute without errors?
- Task completion: Did the session complete successfully?
- Retry patterns: Did user retry or correct the response?
- Session efficiency: Duration relative to task complexity
- Tool sequence completion: Did workflow patterns complete?
- Grounding verification: Did response pass hallucination checks?
- Quality score: Multi-dimensional response quality

Architecture:
    Session → ImplicitFeedbackCollector → ImplicitFeedback → Reward Signal
                     ↓
            RL Learners (all learners can consume this feedback)

Sprint 4: Implicit Feedback Enhancement
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.response_quality import QualityResult

logger = logging.getLogger(__name__)


@dataclass
class ToolExecution:
    """Record of a single tool execution."""

    tool_name: str
    success: bool
    execution_time_ms: float
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class ImplicitFeedback:
    """Aggregated implicit feedback from a session.

    All values are normalized to [0, 1] where higher is better.

    Attributes:
        tool_success_rate: Fraction of tools that executed successfully
        task_completed: Whether the overall task was completed
        retry_count: Number of retries (lower is better, normalized)
        efficiency_score: Session efficiency (shorter = more efficient)
        sequence_completion_rate: Workflow patterns that completed
        grounding_score: Hallucination verification score
        quality_score: Multi-dimensional response quality
        session_id: Unique session identifier
        task_type: Type of task (analysis, action, create, etc.)
        provider: LLM provider used
        model: Model used
        metadata: Additional context
    """

    tool_success_rate: float = 0.0
    task_completed: bool = False
    retry_count: int = 0
    efficiency_score: float = 0.5
    sequence_completion_rate: float = 0.0
    grounding_score: float = 0.7
    quality_score: float = 0.5
    session_id: str = ""
    task_type: str = "general"
    provider: str = ""
    model: str = ""
    duration_seconds: float = 0.0
    tool_count: int = 0
    iteration_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_reward(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute composite reward from implicit feedback.

        Default weights (can be customized or learned):
        - task_completed: 40% (most important)
        - tool_success_rate: 25%
        - grounding_score: 15%
        - efficiency_score: 10%
        - quality_score: 10%

        Args:
            weights: Optional custom weights for each signal

        Returns:
            Reward value in [-1.0, 1.0]
        """
        default_weights = {
            "task_completed": 0.40,
            "tool_success_rate": 0.25,
            "grounding_score": 0.15,
            "efficiency_score": 0.10,
            "quality_score": 0.10,
        }
        w = weights or default_weights

        # Normalize retry penalty (more retries = lower score)
        retry_penalty = max(0.0, 1.0 - (self.retry_count * 0.2))

        # Compute weighted sum
        reward = (
            float(self.task_completed) * w.get("task_completed", 0.4)
            + self.tool_success_rate * w.get("tool_success_rate", 0.25)
            + self.grounding_score * w.get("grounding_score", 0.15)
            + self.efficiency_score * w.get("efficiency_score", 0.10)
            + self.quality_score * w.get("quality_score", 0.10)
        ) * retry_penalty

        # Shift to [-1, 1] range (centered at 0.5 → 0)
        return (reward - 0.5) * 2


@dataclass
class SessionContext:
    """Context for a session being tracked.

    Mutable state that accumulates during session execution.
    """

    session_id: str
    task_type: str = "general"
    provider: str = ""
    model: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    completed: bool = False

    # Tool tracking
    tool_executions: List[ToolExecution] = field(default_factory=list)
    total_retries: int = 0

    # Iteration tracking
    iteration_count: int = 0
    max_iterations: int = 30

    # Quality tracking
    grounding_results: List[float] = field(default_factory=list)
    quality_results: List[float] = field(default_factory=list)

    # Workflow tracking
    workflow_patterns_started: int = 0
    workflow_patterns_completed: int = 0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImplicitFeedbackCollector:
    """Collects implicit feedback from session behavior.

    This collector is designed to be integrated with the orchestrator
    to passively observe session events and derive reward signals.

    Usage:
        collector = ImplicitFeedbackCollector()
        session = collector.start_session("session_123", "analysis", "anthropic", "claude-3")

        # During session execution
        collector.record_tool_execution(session, tool_name, success, duration_ms)
        collector.record_grounding_result(session, score)
        collector.record_iteration(session)

        # At session end
        feedback = collector.end_session(session, completed=True)
        reward = feedback.compute_reward()
    """

    # Baseline durations for efficiency calculation (seconds)
    BASELINE_DURATIONS = {
        "analysis": 30.0,
        "search": 15.0,
        "action": 45.0,
        "create": 60.0,
        "edit": 30.0,
        "general": 30.0,
    }

    def __init__(self):
        """Initialize the feedback collector."""
        self._active_sessions: Dict[str, SessionContext] = {}
        self._completed_feedback: List[ImplicitFeedback] = []

        # Statistics
        self._total_sessions: int = 0
        self._total_completed: int = 0

    def start_session(
        self,
        session_id: str,
        task_type: str = "general",
        provider: str = "",
        model: str = "",
        max_iterations: int = 30,
    ) -> SessionContext:
        """Start tracking a new session.

        Args:
            session_id: Unique session identifier
            task_type: Type of task being performed
            provider: LLM provider
            model: Model being used
            max_iterations: Maximum iterations allowed

        Returns:
            SessionContext for tracking
        """
        session = SessionContext(
            session_id=session_id,
            task_type=task_type,
            provider=provider,
            model=model,
            max_iterations=max_iterations,
        )

        self._active_sessions[session_id] = session
        self._total_sessions += 1

        logger.debug(f"ImplicitFeedback: Started session {session_id} ({task_type})")
        return session

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get an active session by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionContext or None if not found
        """
        return self._active_sessions.get(session_id)

    def record_tool_execution(
        self,
        session: SessionContext,
        tool_name: str,
        success: bool,
        execution_time_ms: float,
        error_message: Optional[str] = None,
        is_retry: bool = False,
    ) -> None:
        """Record a tool execution.

        Args:
            session: Session context
            tool_name: Name of the tool
            success: Whether execution succeeded
            execution_time_ms: Execution duration in milliseconds
            error_message: Optional error message on failure
            is_retry: Whether this is a retry of a previous execution
        """
        execution = ToolExecution(
            tool_name=tool_name,
            success=success,
            execution_time_ms=execution_time_ms,
            error_message=error_message,
        )

        session.tool_executions.append(execution)

        if is_retry:
            session.total_retries += 1

        logger.debug(
            f"ImplicitFeedback: Tool {tool_name} {'succeeded' if success else 'failed'} "
            f"({execution_time_ms:.1f}ms)"
        )

    def record_iteration(self, session: SessionContext) -> None:
        """Record an iteration completion.

        Args:
            session: Session context
        """
        session.iteration_count += 1

    def record_grounding_result(self, session: SessionContext, confidence: float) -> None:
        """Record a grounding verification result.

        Args:
            session: Session context
            confidence: Grounding confidence score (0-1)
        """
        session.grounding_results.append(confidence)

    def record_quality_result(
        self,
        session: SessionContext,
        quality_result: "QualityResult",
    ) -> None:
        """Record a quality scoring result.

        Args:
            session: Session context
            quality_result: Quality result from ResponseQualityScorer
        """
        session.quality_results.append(quality_result.overall_score)

    def record_workflow_pattern(
        self, session: SessionContext, started: bool = False, completed: bool = False
    ) -> None:
        """Record workflow pattern tracking.

        Args:
            session: Session context
            started: Whether a pattern was started
            completed: Whether a pattern was completed
        """
        if started:
            session.workflow_patterns_started += 1
        if completed:
            session.workflow_patterns_completed += 1

    def end_session(
        self,
        session: SessionContext,
        completed: bool = False,
    ) -> ImplicitFeedback:
        """End a session and compute implicit feedback.

        Args:
            session: Session context
            completed: Whether the task was completed successfully

        Returns:
            Aggregated ImplicitFeedback
        """
        session.end_time = time.time()
        session.completed = completed

        # Remove from active sessions
        self._active_sessions.pop(session.session_id, None)

        if completed:
            self._total_completed += 1

        # Compute feedback
        feedback = self._compute_feedback(session)
        self._completed_feedback.append(feedback)

        logger.debug(
            f"ImplicitFeedback: Session {session.session_id} ended "
            f"(completed={completed}, reward={feedback.compute_reward():.3f})"
        )

        return feedback

    def _compute_feedback(self, session: SessionContext) -> ImplicitFeedback:
        """Compute implicit feedback from session context.

        Args:
            session: Completed session context

        Returns:
            Computed ImplicitFeedback
        """
        # Tool success rate
        if session.tool_executions:
            successful = sum(1 for t in session.tool_executions if t.success)
            tool_success_rate = successful / len(session.tool_executions)
        else:
            tool_success_rate = 0.5  # Neutral if no tools

        # Efficiency score (based on duration vs baseline)
        duration = (session.end_time or time.time()) - session.start_time
        baseline = self.BASELINE_DURATIONS.get(session.task_type, 30.0)

        # Efficiency: 1.0 at baseline, decreasing for longer, increasing for shorter
        if duration > 0:
            efficiency_ratio = baseline / duration
            efficiency_score = min(1.0, max(0.0, efficiency_ratio * 0.5 + 0.25))
        else:
            efficiency_score = 0.5

        # Grounding score (average of all results)
        if session.grounding_results:
            grounding_score = sum(session.grounding_results) / len(session.grounding_results)
        else:
            grounding_score = 0.7  # Default if no grounding

        # Quality score (average of all results)
        if session.quality_results:
            quality_score = sum(session.quality_results) / len(session.quality_results)
        else:
            quality_score = 0.5  # Default if no quality scoring

        # Sequence completion rate
        if session.workflow_patterns_started > 0:
            sequence_completion_rate = (
                session.workflow_patterns_completed / session.workflow_patterns_started
            )
        else:
            sequence_completion_rate = 0.5  # Neutral

        return ImplicitFeedback(
            tool_success_rate=tool_success_rate,
            task_completed=session.completed,
            retry_count=session.total_retries,
            efficiency_score=efficiency_score,
            sequence_completion_rate=sequence_completion_rate,
            grounding_score=grounding_score,
            quality_score=quality_score,
            session_id=session.session_id,
            task_type=session.task_type,
            provider=session.provider,
            model=session.model,
            duration_seconds=duration,
            tool_count=len(session.tool_executions),
            iteration_count=session.iteration_count,
            metadata=session.metadata,
        )

    def get_recent_feedback(self, n: int = 10) -> List[ImplicitFeedback]:
        """Get most recent feedback records.

        Args:
            n: Number of records to return

        Returns:
            List of recent ImplicitFeedback
        """
        return self._completed_feedback[-n:]

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics from collected feedback.

        Returns:
            Dictionary with aggregate stats
        """
        if not self._completed_feedback:
            return {
                "total_sessions": self._total_sessions,
                "completed_sessions": self._total_completed,
                "completion_rate": 0.0,
                "avg_reward": 0.0,
            }

        rewards = [f.compute_reward() for f in self._completed_feedback]
        success_rates = [f.tool_success_rate for f in self._completed_feedback]
        grounding_scores = [f.grounding_score for f in self._completed_feedback]

        return {
            "total_sessions": self._total_sessions,
            "completed_sessions": self._total_completed,
            "completion_rate": self._total_completed / max(1, self._total_sessions),
            "avg_reward": sum(rewards) / len(rewards),
            "avg_tool_success_rate": sum(success_rates) / len(success_rates),
            "avg_grounding_score": sum(grounding_scores) / len(grounding_scores),
            "feedback_count": len(self._completed_feedback),
        }

    def export_for_rl(self) -> List[Dict[str, Any]]:
        """Export feedback data in format suitable for RL learners.

        Returns:
            List of dictionaries with RL-ready data
        """
        return [
            {
                "session_id": f.session_id,
                "provider": f.provider,
                "model": f.model,
                "task_type": f.task_type,
                "reward": f.compute_reward(),
                "tool_success_rate": f.tool_success_rate,
                "task_completed": f.task_completed,
                "grounding_score": f.grounding_score,
                "quality_score": f.quality_score,
                "efficiency_score": f.efficiency_score,
                "retry_count": f.retry_count,
                "duration_seconds": f.duration_seconds,
                "tool_count": f.tool_count,
                "iteration_count": f.iteration_count,
            }
            for f in self._completed_feedback
        ]


# Global singleton for easy access
_feedback_collector: Optional[ImplicitFeedbackCollector] = None


def get_feedback_collector() -> ImplicitFeedbackCollector:
    """Get global feedback collector (lazy init).

    Returns:
        Global ImplicitFeedbackCollector singleton
    """
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = ImplicitFeedbackCollector()
    return _feedback_collector
