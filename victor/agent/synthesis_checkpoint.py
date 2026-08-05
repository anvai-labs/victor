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

"""
Synthesis Checkpoint (Template Method Pattern).

This module implements checkpoints that determine when to force synthesis
during agentic tool execution. It helps prevent over-exploration and ensures
that gathered information is consolidated into actionable responses.

SOLID Principles Applied:
- Single Responsibility: Each checkpoint checks one condition
- Open/Closed: New checkpoints can be added without modifying existing code
- Liskov Substitution: All checkpoints are interchangeable
- Interface Segregation: Checkpoint interface is minimal
- Dependency Inversion: Composite depends on abstraction, not concrete checkpoints

Addresses GAP-7: Over-exploration without synthesis
Addresses GAP-8: Missing task completion signal
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class CheckpointResult:
    """Result of a synthesis checkpoint evaluation."""

    should_synthesize: bool
    reason: str
    suggested_prompt: Optional[str] = None
    priority: int = 0  # Higher = more urgent
    metadata: Dict[str, Any] = field(default_factory=dict)


class SynthesisCheckpoint(ABC):
    """
    Abstract checkpoint that determines when to force synthesis.

    Checkpoints are evaluated during tool execution to detect when
    the agent should stop exploring and synthesize findings.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the checkpoint for logging."""
        pass

    @abstractmethod
    def check(
        self, tool_history: List[Dict[str, Any]], task_context: Dict[str, Any]
    ) -> CheckpointResult:
        """
        Evaluate whether synthesis should be triggered.

        Args:
            tool_history: List of tool call records with 'tool', 'args', 'result' keys
            task_context: Context including 'task_type', 'elapsed_time', 'timeout', etc.

        Returns:
            CheckpointResult with synthesis decision and reason
        """
        pass


class ToolCountCheckpoint(SynthesisCheckpoint):
    """Checkpoint based on total tool call count."""

    def __init__(self, max_calls: int = 10) -> None:
        self.max_calls = max_calls

    @property
    def name(self) -> str:
        return "tool_count"

    def check(
        self, tool_history: List[Dict[str, Any]], task_context: Dict[str, Any]
    ) -> CheckpointResult:
        if len(tool_history) >= self.max_calls:
            return CheckpointResult(
                should_synthesize=True,
                reason=f"Reached {self.max_calls} tool calls",
                suggested_prompt=(
                    f"You have made {len(tool_history)} tool calls. "
                    "Synthesize your findings so far before continuing exploration."
                ),
                priority=5,
                metadata={"tool_count": len(tool_history), "threshold": self.max_calls},
            )
        return CheckpointResult(
            should_synthesize=False,
            reason=f"Under limit ({len(tool_history)}/{self.max_calls})",
        )


class DuplicateToolCheckpoint(SynthesisCheckpoint):
    """Checkpoint when same tool called repeatedly with same/similar args."""

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "duplicate_tool"

    def check(
        self, tool_history: List[Dict[str, Any]], task_context: Dict[str, Any]
    ) -> CheckpointResult:
        if len(tool_history) < self.threshold:
            return CheckpointResult(
                should_synthesize=False, reason="Too few calls to detect duplicates"
            )

        # Check for same tool called threshold times in a row
        recent = tool_history[-self.threshold :]
        tool_names = [h.get("tool", "") for h in recent]

        if len(set(tool_names)) == 1 and tool_names[0]:
            tool_name = tool_names[0]
            return CheckpointResult(
                should_synthesize=True,
                reason=f"Same tool '{tool_name}' called {self.threshold} times consecutively",
                suggested_prompt=(
                    f"You have called '{tool_name}' {self.threshold} times in a row. "
                    "This suggests you may be stuck in a loop. "
                    "Please synthesize what you have learned and try a different approach."
                ),
                priority=8,
                metadata={"repeated_tool": tool_name, "count": self.threshold},
            )

        return CheckpointResult(should_synthesize=False, reason="No excessive repetition detected")


class SimilarArgsCheckpoint(SynthesisCheckpoint):
    """Checkpoint when tools are called with similar arguments."""

    def __init__(self, window_size: int = 5, similarity_threshold: float = 0.7) -> None:
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold

    @property
    def name(self) -> str:
        return "similar_args"

    def check(
        self, tool_history: List[Dict[str, Any]], task_context: Dict[str, Any]
    ) -> CheckpointResult:
        if len(tool_history) < 3:
            return CheckpointResult(
                should_synthesize=False, reason="Too few calls to detect similarity"
            )

        recent = tool_history[-self.window_size :]

        # Extract path-like arguments for comparison
        paths_seen: List[str] = []
        queries_seen: List[str] = []

        for h in recent:
            args = h.get("args", {})
            if isinstance(args, dict):
                for key in ("path", "file_path", "file", "directory"):
                    if key in args:
                        paths_seen.append(str(args[key]))
                for key in ("query", "pattern", "search"):
                    if key in args:
                        queries_seen.append(str(args[key]))

        # Check for repeated paths
        if paths_seen:
            unique_paths = set(paths_seen)
            if len(unique_paths) < len(paths_seen) * (1 - self.similarity_threshold):
                return CheckpointResult(
                    should_synthesize=True,
                    reason="Repeated file operations on similar paths",
                    suggested_prompt=(
                        "You are repeatedly accessing the same files. "
                        "Consider synthesizing what you've learned from these files."
                    ),
                    priority=6,
                    metadata={"repeated_paths": list(unique_paths)},
                )

        # Check for similar queries
        if queries_seen:
            unique_queries = set(queries_seen)
            if len(unique_queries) < len(queries_seen) * (1 - self.similarity_threshold):
                return CheckpointResult(
                    should_synthesize=True,
                    reason="Repeated searches with similar queries",
                    suggested_prompt=(
                        "You are running similar searches repeatedly. "
                        "Consider synthesizing the search results you have gathered."
                    ),
                    priority=6,
                    metadata={"repeated_queries": list(unique_queries)},
                )

        return CheckpointResult(should_synthesize=False, reason="No excessive similarity detected")


class TimeoutApproachingCheckpoint(SynthesisCheckpoint):
    """Checkpoint when approaching time limit."""

    def __init__(self, warning_threshold: float = 0.7, critical_threshold: float = 0.9) -> None:
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    @property
    def name(self) -> str:
        return "timeout_approaching"

    def check(
        self, tool_history: List[Dict[str, Any]], task_context: Dict[str, Any]
    ) -> CheckpointResult:
        elapsed = task_context.get("elapsed_time", 0)
        timeout = task_context.get("timeout", 180)

        if timeout <= 0:
            return CheckpointResult(should_synthesize=False, reason="No timeout configured")

        time_ratio = elapsed / timeout
        remaining = timeout - elapsed

        if time_ratio > self.critical_threshold:
            return CheckpointResult(
                should_synthesize=True,
                reason=f"Critical: {int(remaining)}s remaining ({int(time_ratio*100)}% used)",
                suggested_prompt=(
                    f"Time is critically short ({int(remaining)}s remaining). "
                    "Provide your best answer with current findings immediately."
                ),
                priority=10,
                metadata={
                    "elapsed": elapsed,
                    "timeout": timeout,
                    "remaining": remaining,
                },
            )

        if time_ratio > self.warning_threshold:
            return CheckpointResult(
                should_synthesize=True,
                reason=f"Warning: {int(remaining)}s remaining ({int(time_ratio*100)}% used)",
                suggested_prompt=(
                    f"Time is running short ({int(remaining)}s remaining). "
                    "Consider synthesizing your findings soon."
                ),
                priority=7,
                metadata={
                    "elapsed": elapsed,
                    "timeout": timeout,
                    "remaining": remaining,
                },
            )

        return CheckpointResult(
            should_synthesize=False,
            reason=f"Plenty of time ({int(remaining)}s remaining)",
        )


class NoProgressCheckpoint(SynthesisCheckpoint):
    """Checkpoint when tool results show no new information."""

    def __init__(self, window_size: int = 4) -> None:
        self.window_size = window_size

    @property
    def name(self) -> str:
        return "no_progress"

    def check(
        self, tool_history: List[Dict[str, Any]], task_context: Dict[str, Any]
    ) -> CheckpointResult:
        if len(tool_history) < self.window_size:
            return CheckpointResult(
                should_synthesize=False, reason="Too few calls to detect progress"
            )

        recent = tool_history[-self.window_size :]

        # Check for failed tools
        failures = sum(1 for h in recent if not h.get("success", True) or h.get("error"))
        if failures >= self.window_size - 1:
            return CheckpointResult(
                should_synthesize=True,
                reason=f"{failures}/{self.window_size} recent tool calls failed",
                suggested_prompt=(
                    "Most recent tool calls have failed. "
                    "Please provide a response based on available information "
                    "or ask for clarification."
                ),
                priority=8,
                metadata={"failures": failures, "window": self.window_size},
            )

        # Check for empty results
        empty_count = 0
        for h in recent:
            result = h.get("result", "")
            if not result or result == "[]" or result == "{}" or result == "null":
                empty_count += 1

        if empty_count >= self.window_size - 1:
            return CheckpointResult(
                should_synthesize=True,
                reason=f"{empty_count}/{self.window_size} recent results were empty",
                suggested_prompt=(
                    "Recent tool calls returned empty or minimal results. "
                    "Consider synthesizing what you have found or trying a different approach."
                ),
                priority=7,
                metadata={"empty_results": empty_count, "window": self.window_size},
            )

        return CheckpointResult(should_synthesize=False, reason="Progress appears normal")


class ErrorRateCheckpoint(SynthesisCheckpoint):
    """Checkpoint when error rate exceeds threshold."""

    def __init__(self, error_threshold: float = 0.5, min_calls: int = 4) -> None:
        self.error_threshold = error_threshold
        self.min_calls = min_calls

    @property
    def name(self) -> str:
        return "error_rate"

    def check(
        self, tool_history: List[Dict[str, Any]], task_context: Dict[str, Any]
    ) -> CheckpointResult:
        if len(tool_history) < self.min_calls:
            return CheckpointResult(
                should_synthesize=False, reason="Too few calls to calculate error rate"
            )

        errors = sum(1 for h in tool_history if not h.get("success", True) or h.get("error"))
        error_rate = errors / len(tool_history)

        if error_rate > self.error_threshold:
            return CheckpointResult(
                should_synthesize=True,
                reason=f"Error rate {error_rate:.0%} exceeds {self.error_threshold:.0%}",
                suggested_prompt=(
                    f"Error rate is high ({int(error_rate*100)}% of tool calls failed). "
                    "Consider providing a partial answer or asking for assistance."
                ),
                priority=9,
                metadata={"error_rate": error_rate, "total_calls": len(tool_history)},
            )

        return CheckpointResult(
            should_synthesize=False,
            reason=f"Error rate acceptable ({error_rate:.0%})",
        )


class UnifiedTaskTrackerCheckpoint(SynthesisCheckpoint):
    """Checkpoint that delegates to UnifiedTaskTracker for sophisticated loop detection.

    Phase 3 Enhancement: Replaces primitive heuristic-based checkpoints (DuplicateTool,
    SimilarArgs, NoProgress) with UnifiedTaskTracker's sophisticated detection:

    - Offset-aware file read loop detection (same file + offset + limit)
    - Signature-based deduplication with permanent blocking
    - Task-type aware thresholds (different for EDIT vs. ANALYZE)
    - Mode-aware limits (exploration multiplier)
    - Progress tracking and plateau detection

    This is MORE sophisticated than primitive counting and aligns with the refined
    execution optimization plan's goal of trusting UnifiedTaskTracker over heuristics.
    """

    def __init__(self, tracker: Optional[Any] = None) -> None:
        """Initialize with optional UnifiedTaskTracker instance.

        Args:
            tracker: UnifiedTaskTracker instance. If None, will be looked up from task_context.
        """
        self._tracker = tracker

    @property
    def name(self) -> str:
        return "unified_task_tracker"

    def check(
        self, tool_history: List[Dict[str, Any]], task_context: Dict[str, Any]
    ) -> CheckpointResult:
        # Get tracker from instance or task_context
        tracker = self._tracker or task_context.get("unified_task_tracker")
        if not tracker:
            return CheckpointResult(
                should_synthesize=False,
                reason="UnifiedTaskTracker not available",
            )

        # Query UnifiedTaskTracker for loop warning
        try:
            warning = tracker.check_loop_warning()
            if warning:
                return CheckpointResult(
                    should_synthesize=True,
                    reason=f"UnifiedTaskTracker loop detection: {warning}",
                    suggested_prompt=(
                        f"UnifiedTaskTracker detected a potential loop: {warning}\n\n"
                        "Please synthesize your findings so far and try a different approach. "
                        "The system has detected repeated patterns that suggest you may be stuck."
                    ),
                    priority=9,  # High priority (loops are serious)
                    metadata={"warning": warning, "checkpoint": "unified_task_tracker"},
                )
        except Exception as e:
            logger.warning(f"UnifiedTaskTracker checkpoint failed: {e}")
            return CheckpointResult(
                should_synthesize=False,
                reason=f"UnifiedTaskTracker checkpoint error: {e}",
            )

        return CheckpointResult(
            should_synthesize=False,
            reason="UnifiedTaskTracker: No loop detected",
        )


class CompositeSynthesisCheckpoint(SynthesisCheckpoint):
    """Combines multiple checkpoints with priority-based selection."""

    def __init__(self, checkpoints: Optional[List[SynthesisCheckpoint]] = None) -> None:
        self._checkpoints = checkpoints or []

    @property
    def name(self) -> str:
        return "composite"

    def add_checkpoint(self, checkpoint: SynthesisCheckpoint) -> "CompositeSynthesisCheckpoint":
        """Add a checkpoint to the composite."""
        self._checkpoints.append(checkpoint)
        return self

    def check(
        self, tool_history: List[Dict[str, Any]], task_context: Dict[str, Any]
    ) -> CheckpointResult:
        triggered: List[CheckpointResult] = []

        for checkpoint in self._checkpoints:
            try:
                result = checkpoint.check(tool_history, task_context)
                if result.should_synthesize:
                    result.metadata["checkpoint"] = checkpoint.name
                    triggered.append(result)
            except Exception as e:
                logger.warning(f"Checkpoint {checkpoint.name} failed: {e}")

        if not triggered:
            return CheckpointResult(
                should_synthesize=False,
                reason="All checkpoints passed",
            )

        # Return highest priority result
        triggered.sort(key=lambda r: r.priority, reverse=True)
        best = triggered[0]

        # Enhance with aggregate info if multiple triggered
        if len(triggered) > 1:
            best.metadata["other_triggers"] = [
                r.metadata.get("checkpoint", "unknown") for r in triggered[1:]
            ]
            best.reason = f"{best.reason} (+{len(triggered)-1} other triggers)"

        return best


def create_default_checkpoint() -> CompositeSynthesisCheckpoint:
    """Create a checkpoint with default configuration.

    Phase 3 Enhancement: Replaces primitive heuristic-based checkpoints (DuplicateTool,
    SimilarArgs, NoProgress) with UnifiedTaskTracker's sophisticated loop detection.

    Kept: ToolCount, TimeoutApproaching, ErrorRate (provide unique value)
    Replaced: DuplicateTool, SimilarArgs, NoProgress → UnifiedTaskTrackerCheckpoint
    """
    return CompositeSynthesisCheckpoint(
        [
            ToolCountCheckpoint(max_calls=12),
            UnifiedTaskTrackerCheckpoint(),  # Replaces DuplicateTool, SimilarArgs, NoProgress
            TimeoutApproachingCheckpoint(warning_threshold=0.7),
            ErrorRateCheckpoint(error_threshold=0.5),
        ]
    )


def create_aggressive_checkpoint() -> CompositeSynthesisCheckpoint:
    """Create a checkpoint that triggers synthesis earlier (for simple tasks).

    Phase 3 Enhancement: Uses UnifiedTaskTracker instead of primitive heuristics.
    """
    return CompositeSynthesisCheckpoint(
        [
            ToolCountCheckpoint(max_calls=5),
            UnifiedTaskTrackerCheckpoint(),  # Replaces DuplicateTool, SimilarArgs, NoProgress
            TimeoutApproachingCheckpoint(warning_threshold=0.5),
            ErrorRateCheckpoint(error_threshold=0.3),
        ]
    )


def create_relaxed_checkpoint() -> CompositeSynthesisCheckpoint:
    """Create a checkpoint that allows more exploration (for complex tasks).

    Phase 3 Enhancement: Uses UnifiedTaskTracker instead of primitive heuristics.
    """
    return CompositeSynthesisCheckpoint(
        [
            ToolCountCheckpoint(max_calls=20),
            UnifiedTaskTrackerCheckpoint(),  # Replaces DuplicateTool, SimilarArgs, NoProgress
            TimeoutApproachingCheckpoint(warning_threshold=0.8),
            ErrorRateCheckpoint(error_threshold=0.6),
        ]
    )


def get_checkpoint_for_complexity(complexity: str) -> CompositeSynthesisCheckpoint:
    """Get appropriate checkpoint configuration for task complexity."""
    if complexity == "simple":
        return create_aggressive_checkpoint()
    elif complexity == "complex":
        return create_relaxed_checkpoint()
    else:  # medium or default
        return create_default_checkpoint()
