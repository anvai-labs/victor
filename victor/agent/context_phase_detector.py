"""Context phase detector for phase-aware context management.

Detects the current task phase (EXPLORATION, PLANNING, EXECUTION, REVIEW)
based on conversation stage, tool usage patterns, and content analysis.

The phase detector uses multiple strategies:
1. Stage mapping (primary): Maps ConversationStage to TaskPhase
2. Tool patterns (tiebreaker): Uses tool usage to infer phase
3. Content analysis (optional): Analyzes message content for phase signals

Includes cooldown (2s) and thrashing prevention (6 transitions/2min) to avoid
rapid phase switching that could degrade context quality.

Usage:
    from victor.agent.context_phase_detector import PhaseDetector, PhaseTransitionDetector

    detector = PhaseDetector()

    # Detect current phase
    phase = detector.detect_phase(
        current_stage=ConversationStage.EXECUTION,
        recent_tools=["edit", "write"],
        message_content="Implementing the fix"
    )
    # Returns: TaskPhase.EXECUTION
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from victor.core.shared_types import ConversationStage, TaskPhase, STAGE_TO_PHASE_MAP
from victor.tools.core_tool_aliases import canonicalize_core_tool_name
from victor.tools.tool_names import ToolNames, get_canonical_name

logger = logging.getLogger(__name__)


class PhaseDetectionStrategy(str, Enum):
    """Strategies for phase detection."""

    STAGE_MAPPING = "stage_mapping"  # Use stage-to-phase mapping only
    TOOL_PATTERNS = "tool_patterns"  # Use tool usage patterns
    CONTENT_ANALYSIS = "content_analysis"  # Analyze message content
    ENSEMBLE = "ensemble"  # Combine multiple strategies


@dataclass
class PhaseDetectionResult:
    """Result of phase detection with metadata."""

    phase: TaskPhase
    confidence: float
    strategy: str
    matched_signals: List[str]
    timestamp: float = field(default_factory=time.monotonic)


# Tool patterns for each phase
TOOL_PATTERNS: Dict[TaskPhase, List[str]] = {
    TaskPhase.EXPLORATION: [
        "search",
        "find",
        "locate",
        "grep",
        "list",
        ToolNames.READ,
        ToolNames.LS,
    ],
    TaskPhase.PLANNING: [
        "plan",
        "breakdown",
        "decompose",
        "outline",
        "design",
        "architect",
    ],
    TaskPhase.EXECUTION: [
        ToolNames.EDIT,
        ToolNames.WRITE,
        "create",
        "delete",
        "modify",
        "run",
        "execute",
        "test",
        "build",
        "apply",
        ToolNames.SHELL,
    ],
    TaskPhase.REVIEW: [
        "verify",
        "check",
        "validate",
        "test",
        "inspect",
        "review",
        "analyze",
    ],
}

# Content patterns for each phase
CONTENT_PATTERNS: Dict[TaskPhase, List[str]] = {
    TaskPhase.EXPLORATION: [
        r"\b(looking|searching|finding|exploring|reading|understanding)\b",
        r"\bwhat (file|files|code|function|class)\b",
        r"\bwhere is\b",
        r"\bshow me\b",
    ],
    TaskPhase.PLANNING: [
        r"\b(plan|design|approach|strategy|outline|breakdown)\b",
        r"\b(how to|how should|steps to)\b",
        r"\b(first|then|next|after that)\b",
        r"\b(implement|create|build)\b",
    ],
    TaskPhase.EXECUTION: [
        r"\b(implement|implementing|creating|writing|editing|modifying)\b",
        r"\b(fixing|fix|changed|updated|added|removed)\b",
        r"\b(running|executing|testing|building)\b",
        r"\b(apply|commit|save)\b",
    ],
    TaskPhase.REVIEW: [
        r"\b(checking|verifying|validating|testing)\b",
        r"\b(done|complete|finished|ready)\b",
        r"\b(work(ing)? correctly|as expected)\b",
        r"\b(test(s|ing)|review(ing)?|inspect(ing)?)\b",
    ],
}


def _canonical_phase_tool_name(tool_name: str) -> str:
    """Normalize core-tool aliases before phase matching."""
    lowered = tool_name.lower()
    return get_canonical_name(canonicalize_core_tool_name(lowered))


class PhaseDetector:
    """Detects the current task phase based on multiple signals.

    Uses a tiered approach:
    1. Stage mapping (primary): Maps ConversationStage to TaskPhase
    2. Tool patterns (tiebreaker): Uses tool usage to infer phase
    3. Content analysis (optional): Analyzes message content for phase signals

    This allows for accurate phase detection even when stage information
    is ambiguous or transitions are occurring.
    """

    def __init__(
        self,
        strategy: PhaseDetectionStrategy = PhaseDetectionStrategy.STAGE_MAPPING,
        enable_content_analysis: bool = False,
    ):
        """Initialize the phase detector.

        Args:
            strategy: Primary detection strategy
            enable_content_analysis: Whether to analyze message content
        """
        self._strategy = strategy
        self._enable_content_analysis = enable_content_analysis

        # Compile content regex patterns for efficiency
        self._compiled_patterns: Dict[TaskPhase, List[re.Pattern]] = {}
        for phase, patterns in CONTENT_PATTERNS.items():
            self._compiled_patterns[phase] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def detect_phase(
        self,
        current_stage: ConversationStage,
        recent_tools: Optional[List[str]] = None,
        message_content: Optional[str] = None,
    ) -> TaskPhase:
        """Detect the current task phase.

        Args:
            current_stage: Current conversation stage
            recent_tools: List of recently used tools (for tiebreaking)
            message_content: Current message content (for content analysis)

        Returns:
            Detected task phase
        """
        # Strategy 1: Stage mapping (primary)
        if self._strategy in (
            PhaseDetectionStrategy.STAGE_MAPPING,
            PhaseDetectionStrategy.ENSEMBLE,
        ):
            phase = STAGE_TO_PHASE_MAP.get(current_stage)
            if phase:
                return phase

        # Strategy 2: Tool patterns (tiebreaker)
        if (
            self._strategy
            in (
                PhaseDetectionStrategy.TOOL_PATTERNS,
                PhaseDetectionStrategy.ENSEMBLE,
            )
            and recent_tools
        ):
            phase = self._detect_from_tools(recent_tools)
            if phase:
                return phase

        # Strategy 3: Content analysis (optional)
        if (
            self._enable_content_analysis
            and self._strategy
            in (
                PhaseDetectionStrategy.CONTENT_ANALYSIS,
                PhaseDetectionStrategy.ENSEMBLE,
            )
            and message_content
        ):
            phase = self._detect_from_content(message_content)
            if phase:
                return phase

        # Fallback to EXPLORATION
        logger.debug("Phase detection failed, falling back to EXPLORATION")
        return TaskPhase.EXPLORATION

    def _detect_from_tools(self, recent_tools: List[str]) -> Optional[TaskPhase]:
        """Detect phase based on recently used tools.

        Args:
            recent_tools: List of tool names

        Returns:
            Detected phase or None
        """
        if not recent_tools:
            return None

        # Count tool usage per phase
        phase_counts: Dict[TaskPhase, int] = dict.fromkeys(TaskPhase, 0)

        for tool in recent_tools:
            canonical_tool = _canonical_phase_tool_name(tool)
            for phase, patterns in TOOL_PATTERNS.items():
                if canonical_tool in patterns:
                    phase_counts[phase] += 1

        # Find phase with most tool usage
        best_phase = max(phase_counts, key=phase_counts.get)

        if phase_counts[best_phase] > 0:
            return best_phase

        return None

    def _detect_from_content(self, content: str) -> Optional[TaskPhase]:
        """Detect phase based on message content.

        Args:
            content: Message content to analyze

        Returns:
            Detected phase or None
        """
        if not content:
            return None

        # Count pattern matches per phase
        phase_counts: Dict[TaskPhase, int] = dict.fromkeys(TaskPhase, 0)

        for phase, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    phase_counts[phase] += 1

        # Find phase with most pattern matches
        best_phase = max(phase_counts, key=phase_counts.get)

        if phase_counts[best_phase] > 0:
            return best_phase

        return None


class PhaseTransitionDetector:
    """Manages phase transitions with cooldown and thrashing prevention.

    Prevents rapid phase switching that could degrade context quality by:
    - Enforcing cooldown period (2s) between transitions
    - Limiting transitions in time window (6 per 2 minutes)
    - Validating transitions (only allow valid phase changes)

    Valid phase transitions:
    - EXPLORATION → PLANNING
    - PLANNING → EXECUTION
    - EXECUTION → REVIEW
    - REVIEW → EXPLORATION (for multi-step tasks)
    """

    # Valid phase transitions
    VALID_TRANSITIONS: Dict[TaskPhase, List[TaskPhase]] = {
        TaskPhase.EXPLORATION: [TaskPhase.PLANNING, TaskPhase.EXECUTION],
        TaskPhase.PLANNING: [TaskPhase.EXECUTION],
        TaskPhase.EXECUTION: [TaskPhase.REVIEW, TaskPhase.EXPLORATION],
        TaskPhase.REVIEW: [TaskPhase.EXPLORATION, TaskPhase.PLANNING],
    }

    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        max_transitions: int = 6,
        window_seconds: float = 120.0,
    ):
        """Initialize the phase transition detector.

        Args:
            cooldown_seconds: Minimum time between phase transitions
            max_transitions: Maximum transitions in time window
            window_seconds: Time window for transition counting
        """
        self._cooldown_seconds = cooldown_seconds
        self._max_transitions = max_transitions
        self._window_seconds = window_seconds

        # State tracking
        self._current_phase: Optional[TaskPhase] = None
        self._last_transition_time: float = 0.0
        self._transition_history: deque = deque(maxlen=10)  # Store last 10 transitions

    def should_transition(
        self,
        new_phase: TaskPhase,
    ) -> bool:
        """Check if a phase transition should be allowed.

        Args:
            new_phase: Proposed new phase

        Returns:
            True if transition should be allowed
        """
        now = time.monotonic()

        # First transition is always allowed
        if self._current_phase is None:
            return True

        # Check cooldown
        time_since_last = now - self._last_transition_time
        if time_since_last < self._cooldown_seconds:
            logger.debug(
                "Phase transition blocked by cooldown: %.2fs < %.2fs",
                time_since_last,
                self._cooldown_seconds,
            )
            return False

        # Check if this is a valid transition
        valid_transitions = self.VALID_TRANSITIONS.get(self._current_phase, [])
        if new_phase not in valid_transitions:
            logger.debug(
                "Invalid phase transition: %s → %s",
                self._current_phase.value,
                new_phase.value,
            )
            return False

        # Check for thrashing (too many transitions in time window)
        recent_transitions = [t for t in self._transition_history if now - t < self._window_seconds]

        if len(recent_transitions) >= self._max_transitions:
            logger.debug(
                "Phase transition blocked by thrashing: %d transitions in %.1fs",
                len(recent_transitions),
                self._window_seconds,
            )
            return False

        # Allow transition
        return True

    def record_transition(
        self,
        old_phase: TaskPhase,
        new_phase: TaskPhase,
    ) -> None:
        """Record a phase transition.

        Args:
            old_phase: Previous phase
            new_phase: New phase
        """
        now = time.monotonic()

        self._current_phase = new_phase
        self._last_transition_time = now
        self._transition_history.append(now)

        logger.info(
            "Phase transition: %s → %s",
            old_phase.value if old_phase else "None",
            new_phase.value,
        )

    def get_current_phase(self) -> Optional[TaskPhase]:
        """Get the current phase.

        Returns:
            Current phase or None if no phase set yet
        """
        return self._current_phase

    def get_transition_count(self) -> int:
        """Get total number of transitions recorded.

        Returns:
            Number of transitions
        """
        return len(self._transition_history)

    def get_stats(self) -> Dict[str, any]:
        """Get detector statistics.

        Returns:
            Dictionary with detector stats
        """
        now = time.monotonic()

        recent_transitions = [t for t in self._transition_history if now - t < self._window_seconds]

        return {
            "current_phase": self._current_phase.value if self._current_phase else None,
            "last_transition": (
                now - self._last_transition_time if self._transition_history else None
            ),
            "total_transitions": len(self._transition_history),
            "recent_transitions": len(recent_transitions),
        }


def create_phase_detector(
    strategy: str = "stage_mapping",
    **kwargs,
) -> PhaseDetector:
    """Factory function to create a phase detector.

    Args:
        strategy: Detection strategy ("stage_mapping", "tool_patterns", "content_analysis", "ensemble")
        **kwargs: Additional arguments to pass to PhaseDetector

    Returns:
        Configured PhaseDetector instance
    """
    try:
        strategy_enum = PhaseDetectionStrategy(strategy.lower())
    except ValueError:
        logger.warning("Invalid phase detection strategy '%s', using 'stage_mapping'", strategy)
        strategy_enum = PhaseDetectionStrategy.STAGE_MAPPING

    return PhaseDetector(strategy=strategy_enum, **kwargs)
