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

"""Time-aware execution management for agent workflows.

This module provides mechanisms to track execution time budgets and provide
guidance to agents about remaining time, enabling graceful completion before
timeout.

Issue Reference: workflow-test-issues-v2.md Issue #3
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.presentation import PresentationProtocol

logger = logging.getLogger(__name__)


class TimePhase(Enum):
    """Execution time phases based on remaining budget."""

    NORMAL = "normal"  # >50% time remaining - full exploration allowed
    WARNING = "warning"  # 25-50% time remaining - prioritize deliverables
    CRITICAL = "critical"  # <25% time remaining - summarize and conclude
    EXPIRED = "expired"  # Time exhausted - must stop immediately


@dataclass
class ExecutionCheckpoint:
    """Progress checkpoint during time-aware execution.

    Renamed from Checkpoint to be semantically distinct:
    - GitCheckpoint (victor.agent.checkpoints): Git stash-based
    - ExecutionCheckpoint (here): Time/progress tracking with phase
    - WorkflowCheckpoint (victor.framework.graph): Workflow state persistence
    - HITLCheckpoint (victor.framework.hitl): Human-in-the-loop pause/resume
    """

    timestamp: float
    elapsed: float
    remaining: float
    description: str
    phase: TimePhase
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionBudget:
    """Tracks execution time budget and progress."""

    total_seconds: float
    start_time: float
    checkpoints: List[ExecutionCheckpoint] = field(default_factory=list)
    phase_transitions: List[tuple] = field(default_factory=list)
    _last_phase: TimePhase = field(default=TimePhase.NORMAL)

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def remaining(self) -> float:
        """Get remaining time in seconds."""
        return max(0, self.total_seconds - self.elapsed)

    @property
    def progress_ratio(self) -> float:
        """Get progress as ratio (0.0 to 1.0)."""
        return min(1.0, self.elapsed / self.total_seconds)

    @property
    def remaining_ratio(self) -> float:
        """Get remaining time as ratio (0.0 to 1.0)."""
        return max(0.0, 1.0 - self.progress_ratio)

    @property
    def phase(self) -> TimePhase:
        """Get current execution phase based on remaining time."""
        ratio = self.remaining_ratio

        if ratio <= 0:
            new_phase = TimePhase.EXPIRED
        elif ratio < 0.25:
            new_phase = TimePhase.CRITICAL
        elif ratio < 0.50:
            new_phase = TimePhase.WARNING
        else:
            new_phase = TimePhase.NORMAL

        # Track phase transitions
        if new_phase != self._last_phase:
            self.phase_transitions.append((self.elapsed, self._last_phase, new_phase))
            self._last_phase = new_phase

        return new_phase

    def checkpoint(self, description: str, **metadata) -> ExecutionCheckpoint:
        """Record a checkpoint for progress tracking.

        Args:
            description: Description of checkpoint
            **metadata: Additional metadata to store

        Returns:
            Created ExecutionCheckpoint
        """
        cp = ExecutionCheckpoint(
            timestamp=time.time(),
            elapsed=self.elapsed,
            remaining=self.remaining,
            description=description,
            phase=self.phase,
            metadata=metadata,
        )
        self.checkpoints.append(cp)
        logger.debug(f"ExecutionCheckpoint: {description} (elapsed: {self.elapsed:.1f}s)")
        return cp

    def get_summary(self) -> Dict[str, Any]:
        """Get budget summary."""
        return {
            "total_seconds": self.total_seconds,
            "elapsed": round(self.elapsed, 2),
            "remaining": round(self.remaining, 2),
            "phase": self.phase.value,
            "checkpoints": len(self.checkpoints),
            "phase_transitions": len(self.phase_transitions),
        }


@runtime_checkable
class ITimeAwareExecutor(Protocol):
    """Protocol for time-aware execution."""

    def get_phase(self) -> TimePhase:
        """Get current execution phase."""
        ...

    def get_time_guidance(self) -> str:
        """Get guidance based on remaining time."""
        ...

    def should_summarize_now(self) -> bool:
        """Check if agent should summarize and conclude."""
        ...

    def get_remaining_seconds(self) -> Optional[float]:
        """Get remaining seconds or None if no budget."""
        ...


class TimeAwareExecutor:
    """Manages execution with time awareness.

    Provides:
    - Time budget tracking with phases
    - Guidance messages based on remaining time
    - ExecutionCheckpoint recording for progress tracking
    - Callbacks for phase transitions

    Usage:
        executor = TimeAwareExecutor(timeout_seconds=120)

        # During execution
        guidance = executor.get_time_guidance()
        if executor.should_summarize_now():
            # Prioritize delivering summary

        # Record progress
        executor.checkpoint("Completed file analysis")
    """

    # Phase transition thresholds
    WARNING_THRESHOLD = 0.50  # 50% time remaining
    CRITICAL_THRESHOLD = 0.25  # 25% time remaining

    # Guidance messages per phase (without icons - icons added dynamically)
    # Format: (icon_name, message)
    _PHASE_GUIDANCE_TEMPLATES = {
        TimePhase.NORMAL: (None, ""),
        TimePhase.WARNING: (
            "clock",
            "TIME WARNING: Less than 50% time remaining. "
            "Focus on completing primary deliverables. "
            "Avoid starting new exploration paths.",
        ),
        TimePhase.CRITICAL: (
            "warning",
            "TIME CRITICAL: Less than 25% time remaining. "
            "Prioritize delivering the most important output NOW. "
            "Skip non-essential exploration and provide a summary.",
        ),
        TimePhase.EXPIRED: (
            "stop_sign",
            "TIME EXPIRED: Execution time has been exhausted. "
            "Provide immediate summary of current findings and stop.",
        ),
    }

    def __init__(
        self,
        timeout_seconds: Optional[float] = None,
        on_phase_change: Optional[Callable[[TimePhase, TimePhase], None]] = None,
        presentation: Optional["PresentationProtocol"] = None,
    ):
        """Initialize time-aware executor.

        Args:
            timeout_seconds: Total execution time budget (None for unlimited)
            on_phase_change: Callback when phase changes (old_phase, new_phase)
            presentation: Optional presentation adapter for icons (creates default if None)
        """
        self._budget: Optional[ExecutionBudget] = None
        self._on_phase_change = on_phase_change
        self._last_notified_phase = TimePhase.NORMAL

        # Lazy init for backward compatibility
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            self._presentation = create_presentation_adapter()
        else:
            self._presentation = presentation

        if timeout_seconds is not None and timeout_seconds > 0:
            self._budget = ExecutionBudget(
                total_seconds=timeout_seconds,
                start_time=time.time(),
            )
            logger.info(f"Time-aware execution started: {timeout_seconds}s budget")

    def get_phase(self) -> TimePhase:
        """Get current execution phase.

        Returns:
            Current TimePhase based on remaining time
        """
        if not self._budget:
            return TimePhase.NORMAL

        phase = self._budget.phase

        # Notify on phase change
        if phase != self._last_notified_phase:
            if self._on_phase_change:
                self._on_phase_change(self._last_notified_phase, phase)
            logger.info(f"Phase transition: {self._last_notified_phase.value} -> {phase.value}")
            self._last_notified_phase = phase

        return phase

    def get_time_guidance(self) -> str:
        """Get guidance message based on remaining time.

        Returns:
            Guidance string for agent (empty if NORMAL phase)
        """
        phase = self.get_phase()
        template = self._PHASE_GUIDANCE_TEMPLATES.get(phase, (None, ""))
        icon_name, message = template

        if not message:
            return ""

        # Build guidance with icon
        if icon_name:
            icon = self._presentation.icon(icon_name, with_color=False)
            guidance = f"{icon} {message}"
        else:
            guidance = message

        if self._budget:
            remaining = self._budget.remaining
            guidance += f"\n[Remaining: {remaining:.0f}s]"

        return guidance

    def should_summarize_now(self) -> bool:
        """Check if agent should summarize and conclude.

        Returns:
            True if in CRITICAL or EXPIRED phase
        """
        return self.get_phase() in (TimePhase.CRITICAL, TimePhase.EXPIRED)

    def should_avoid_exploration(self) -> bool:
        """Check if agent should avoid new exploration.

        Returns:
            True if in WARNING, CRITICAL, or EXPIRED phase
        """
        return self.get_phase() in (
            TimePhase.WARNING,
            TimePhase.CRITICAL,
            TimePhase.EXPIRED,
        )

    def get_remaining_seconds(self) -> Optional[float]:
        """Get remaining seconds or None if no budget.

        Returns:
            Remaining seconds or None
        """
        return self._budget.remaining if self._budget else None

    def get_elapsed_seconds(self) -> float:
        """Get elapsed seconds since start.

        Returns:
            Elapsed seconds (0 if no budget)
        """
        return self._budget.elapsed if self._budget else 0.0

    def checkpoint(self, description: str, **metadata) -> Optional[ExecutionCheckpoint]:
        """Record a progress checkpoint.

        Args:
            description: ExecutionCheckpoint description
            **metadata: Additional metadata

        Returns:
            Created ExecutionCheckpoint or None if no budget
        """
        if self._budget:
            return self._budget.checkpoint(description, **metadata)
        return None

    def get_budget_summary(self) -> Optional[Dict[str, Any]]:
        """Get budget summary.

        Returns:
            Budget summary dict or None if no budget
        """
        return self._budget.get_summary() if self._budget else None

    def get_checkpoints(self) -> List[ExecutionCheckpoint]:
        """Get all recorded checkpoints.

        Returns:
            List of checkpoints
        """
        return self._budget.checkpoints if self._budget else []

    def is_expired(self) -> bool:
        """Check if execution time is expired.

        Returns:
            True if time expired
        """
        return self.get_phase() == TimePhase.EXPIRED

    def get_tool_budget_recommendation(self) -> int:
        """Get recommended remaining tool budget based on time.

        Returns:
            Recommended number of remaining tool calls
        """
        phase = self.get_phase()

        if phase == TimePhase.EXPIRED:
            return 0
        elif phase == TimePhase.CRITICAL:
            return 2  # Just enough for summary
        elif phase == TimePhase.WARNING:
            return 5  # Limited exploration
        else:
            return 20  # Full budget

    def extend_budget(self, additional_seconds: float) -> None:
        """Extend the time budget.

        Args:
            additional_seconds: Seconds to add to budget
        """
        if self._budget:
            self._budget.total_seconds += additional_seconds
            logger.info(f"Extended budget by {additional_seconds}s")


class TimeAwareContext:
    """Context manager for time-aware execution blocks."""

    def __init__(
        self,
        timeout_seconds: float,
        on_phase_change: Optional[Callable[[TimePhase, TimePhase], None]] = None,
    ):
        """Initialize context.

        Args:
            timeout_seconds: Execution time budget
            on_phase_change: Phase change callback
        """
        self._executor = TimeAwareExecutor(
            timeout_seconds=timeout_seconds,
            on_phase_change=on_phase_change,
        )

    def __enter__(self) -> TimeAwareExecutor:
        """Enter context."""
        return self._executor

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if self._executor._budget:
            summary = self._executor.get_budget_summary()
            logger.info(f"Time-aware execution complete: {summary}")
        return False


def create_time_aware_executor(
    timeout_seconds: Optional[float] = None,
) -> TimeAwareExecutor:
    """Factory function for creating TimeAwareExecutor.

    Args:
        timeout_seconds: Execution time budget

    Returns:
        Configured TimeAwareExecutor instance
    """
    return TimeAwareExecutor(timeout_seconds=timeout_seconds)
