"""Online tool reputation tracking for mid-turn credit feedback.

Maintains per-tool rolling scores updated after each tool execution
WITHIN a turn. Provides tool selection hints that bias the agent toward
tools with better recent track records.

This is the "online" credit feedback loop: unlike assign_turn_credit()
which runs at turn boundaries, tool reputation updates continuously
during tool execution, influencing subsequent tool choices mid-stream.

Usage:
    tracker = ToolReputationTracker()

    # After each tool execution (called by ToolPipeline)
    tracker.record(tool_name="read", success=True, duration_ms=50)
    tracker.record(tool_name="shell", success=False, duration_ms=5000)

    # Get reputation-based hints for prompt injection
    hints = tracker.get_tool_hints()
    # → {"read": 0.95, "shell": -0.3}

    # Get a concise prompt string for mid-turn injection
    guidance = tracker.get_selection_guidance()
    # → "Prefer read (reliable). Avoid shell (recent failures)."
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from victor.framework.tool_naming import get_canonical_name

logger = logging.getLogger(__name__)


@dataclass
class ToolRecord:
    """Single tool execution record for reputation tracking."""

    success: bool
    duration_ms: float
    reward: float


class ToolReputationTracker:
    """Tracks per-tool reputation within and across turns.

    Uses exponential moving average (EMA) for fast adaptation:
    reputation_new = α * current_reward + (1-α) * reputation_old

    High α (0.3) means recent performance weighs heavily — the agent
    adapts quickly to tool failures within the current turn.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        min_calls_for_guidance: int = 2,
        positive_threshold: float = 0.5,
        negative_threshold: float = -0.2,
    ):
        """Initialize the reputation tracker.

        Args:
            alpha: EMA smoothing factor (higher = more responsive)
            min_calls_for_guidance: Minimum calls before generating guidance
            positive_threshold: Reputation above this → "prefer" hint
            negative_threshold: Reputation below this → "avoid" hint
        """
        self._alpha = alpha
        self._min_calls = min_calls_for_guidance
        self._positive_threshold = positive_threshold
        self._negative_threshold = negative_threshold

        # Per-tool EMA reputation
        self._reputation: Dict[str, float] = defaultdict(lambda: 0.0)
        # Per-tool call count
        self._call_count: Dict[str, int] = defaultdict(int)
        # Per-tool recent records (for detailed analysis)
        self._recent: Dict[str, List[ToolRecord]] = defaultdict(list)
        self._max_recent = 20

    def record(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float = 0.0,
    ) -> float:
        """Record a tool execution result and update reputation.

        Called by ToolPipeline after each tool execution, before the
        next tool in the same turn. This creates the mid-turn feedback.

        Args:
            tool_name: Name of the tool
            success: Whether execution succeeded
            duration_ms: Execution duration

        Returns:
            Updated reputation score for this tool
        """
        tool_name = get_canonical_name(tool_name)

        # Compute reward
        reward = 1.0 if success else -1.0
        if success and duration_ms > 5000:
            reward -= min(0.3, (duration_ms - 5000) / 30000)

        record = ToolRecord(success=success, duration_ms=duration_ms, reward=reward)

        # Update EMA reputation
        old_rep = self._reputation[tool_name]
        self._reputation[tool_name] = self._alpha * reward + (1.0 - self._alpha) * old_rep

        # Track call count and recent history
        self._call_count[tool_name] += 1
        self._recent[tool_name].append(record)
        if len(self._recent[tool_name]) > self._max_recent:
            self._recent[tool_name] = self._recent[tool_name][-self._max_recent :]

        return self._reputation[tool_name]

    def get_tool_hints(self) -> Dict[str, float]:
        """Get reputation scores for all tracked tools.

        Returns:
            Dict mapping tool_name → reputation score in [-1.0, 1.0]
        """
        return {
            tool: score
            for tool, score in self._reputation.items()
            if self._call_count[tool] >= self._min_calls
        }

    def get_selection_guidance(self, max_hints: int = 3) -> Optional[str]:
        """Generate concise tool selection guidance for prompt injection.

        Only generates guidance when there's enough data AND clear
        signal (tools significantly above/below threshold).

        Args:
            max_hints: Maximum number of tool hints to include

        Returns:
            Short guidance string or None if insufficient data
        """
        hints = self.get_tool_hints()
        if not hints:
            return None

        preferred: List[Tuple[str, float]] = []
        avoid: List[Tuple[str, float]] = []

        for tool, score in sorted(hints.items(), key=lambda x: x[1], reverse=True):
            if score > self._positive_threshold:
                preferred.append((tool, score))
            elif score < self._negative_threshold:
                avoid.append((tool, score))

        if not preferred and not avoid:
            return None

        parts = []
        for tool, score in preferred[:max_hints]:
            count = self._call_count[tool]
            parts.append(f"- {tool}: reliable ({count} calls, score {score:+.1f})")
        for tool, score in avoid[:max_hints]:
            count = self._call_count[tool]
            recent_fails = sum(1 for r in self._recent[tool][-5:] if not r.success)
            parts.append(
                f"- {tool}: unreliable ({recent_fails}/last 5 failed, "
                f"score {score:+.1f}). Verify arguments carefully."
            )

        if not parts:
            return None

        return "Mid-turn tool reputation:\n" + "\n".join(parts)

    def get_reputation(self, tool_name: str) -> float:
        """Get current reputation for a specific tool."""
        return self._reputation.get(get_canonical_name(tool_name), 0.0)

    def reset(self) -> None:
        """Reset all reputation data."""
        self._reputation.clear()
        self._call_count.clear()
        self._recent.clear()

    def reset_tool(self, tool_name: str) -> None:
        """Reset reputation for a single tool."""
        tool_name = get_canonical_name(tool_name)
        self._reputation.pop(tool_name, None)
        self._call_count.pop(tool_name, None)
        self._recent.pop(tool_name, None)

    @property
    def tracked_tools(self) -> int:
        """Number of tools with reputation data."""
        return len(self._call_count)


__all__ = [
    "ToolReputationTracker",
    "ToolRecord",
]
