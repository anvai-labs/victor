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

"""Dynamic threshold tuning for adaptive optimization.

Learns optimal thresholds from usage data and adapts over time.
This improves routing accuracy by tuning thresholds based on
actual task outcomes rather than static heuristics.

Tracks metrics like:
- Success rate per paradigm/task type
- Token usage per paradigm
- Latency per paradigm
- Adjusts thresholds to optimize for cost vs quality
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ThresholdType(str, Enum):
    """Types of thresholds that can be tuned."""

    COMPLEXITY_DIRECT = "complexity_direct"  # Max complexity for direct paradigm
    COMPLEXITY_FOCUSED = "complexity_focused"  # Max complexity for focused paradigm
    HISTORY_DIRECT = "history_direct"  # Max history length for direct paradigm
    HISTORY_FOCUSED = "history_focused"  # Max history length for focused paradigm
    QUERY_LENGTH_DIRECT = "query_length_direct"  # Max query length for direct
    TOOL_BUDGET_DIRECT = "tool_budget_direct"  # Max tool budget for direct


@dataclass
class TaskOutcome:
    """Record of a task execution outcome.

    Attributes:
        task_type: Task type that was executed
        paradigm: Processing paradigm used
        model_tier: Model tier used
        success: Whether task completed successfully
        token_count: Tokens consumed
        latency_ms: Execution latency
        timestamp: When the task completed
        routing_confidence: Confidence of routing decision
    """

    task_type: str
    paradigm: str
    model_tier: str
    success: bool
    token_count: int
    latency_ms: float
    timestamp: datetime
    routing_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "paradigm": self.paradigm,
            "model_tier": self.model_tier,
            "success": self.success,
            "token_count": self.token_count,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "routing_confidence": self.routing_confidence,
        }


@dataclass
class ThresholdConfig:
    """Configuration for a threshold.

    Attributes:
        name: Threshold name
        value: Current threshold value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        default_value: Default value
        description: What this threshold controls
    """

    name: ThresholdType
    value: float
    min_value: float
    max_value: float
    default_value: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name.value,
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "default_value": self.default_value,
            "description": self.description,
        }


class ThresholdOptimizer:
    """Optimizes thresholds based on task outcomes.

    Learns from actual task executions to tune thresholds for optimal
    cost-quality tradeoff. Uses success rates and token efficiency
    to adjust thresholds over time.

    Example:
        optimizer = ThresholdOptimizer()
        optimizer.record_outcome(outcome)
        optimizer.optimize_thresholds()  # Adjust based on data
    """

    # Default threshold configurations
    DEFAULT_THRESHOLDS = {
        ThresholdType.COMPLEXITY_DIRECT: ThresholdConfig(
            name=ThresholdType.COMPLEXITY_DIRECT,
            value=0.3,
            min_value=0.1,
            max_value=0.6,
            default_value=0.3,
            description="Max complexity score for direct paradigm",
        ),
        ThresholdType.COMPLEXITY_FOCUSED: ThresholdConfig(
            name=ThresholdType.COMPLEXITY_FOCUSED,
            value=0.6,
            min_value=0.4,
            max_value=0.8,
            default_value=0.6,
            description="Max complexity score for focused paradigm",
        ),
        ThresholdType.HISTORY_DIRECT: ThresholdConfig(
            name=ThresholdType.HISTORY_DIRECT,
            value=0.0,
            min_value=0.0,
            max_value=3.0,
            default_value=0.0,
            description="Max history length for direct paradigm",
        ),
        ThresholdType.HISTORY_FOCUSED: ThresholdConfig(
            name=ThresholdType.HISTORY_FOCUSED,
            value=3.0,
            min_value=1.0,
            max_value=5.0,
            default_value=3.0,
            description="Max history length for focused paradigm",
        ),
        ThresholdType.QUERY_LENGTH_DIRECT: ThresholdConfig(
            name=ThresholdType.QUERY_LENGTH_DIRECT,
            value=100.0,
            min_value=50.0,
            max_value=200.0,
            default_value=100.0,
            description="Max query length for direct paradigm",
        ),
        ThresholdType.TOOL_BUDGET_DIRECT: ThresholdConfig(
            name=ThresholdType.TOOL_BUDGET_DIRECT,
            value=3.0,
            min_value=1.0,
            max_value=5.0,
            default_value=3.0,
            description="Max tool budget for direct paradigm",
        ),
    }

    def __init__(
        self,
        enabled: bool = True,
        min_samples: int = 100,
        optimization_interval: int = 1000,
        adjustment_rate: float = 0.1,
    ):
        """Initialize the threshold optimizer.

        Args:
            enabled: Whether the optimizer is enabled
            min_samples: Minimum samples before optimizing
            optimization_interval: Optimize every N outcomes
            adjustment_rate: How much to adjust thresholds (0-1)
        """
        self.enabled = enabled
        self.min_samples = min_samples
        self.optimization_interval = optimization_interval
        self.adjustment_rate = adjustment_rate

        self._thresholds: Dict[ThresholdType, ThresholdConfig] = dict(self.DEFAULT_THRESHOLDS)
        self._outcomes: List[TaskOutcome] = []
        self._optimization_count = 0

    def get_threshold(self, threshold_type: ThresholdType) -> float:
        """Get current threshold value.

        Args:
            threshold_type: Type of threshold

        Returns:
            Current threshold value
        """
        return self._thresholds[threshold_type].value

    def set_threshold(self, threshold_type: ThresholdType, value: float) -> None:
        """Set threshold value.

        Args:
            threshold_type: Type of threshold
            value: New value (will be clamped to min/max)
        """
        config = self._thresholds[threshold_type]
        config.value = max(config.min_value, min(config.max_value, value))
        logger.info(f"[ThresholdOptimizer] Set {threshold_type.value}={config.value:.2f}")

    def record_outcome(self, outcome: TaskOutcome) -> None:
        """Record a task execution outcome.

        Args:
            outcome: Task execution outcome
        """
        if not self.enabled:
            return

        self._outcomes.append(outcome)

        # Optimize periodically
        if len(self._outcomes) >= self.optimization_interval:
            self.optimize_thresholds()

    def optimize_thresholds(self) -> Dict[str, Any]:
        """Optimize thresholds based on collected outcomes.

        Analyzes success rates and token efficiency per paradigm
        to adjust thresholds for better cost-quality tradeoff.

        Returns:
            Dict with optimization results
        """
        if not self.enabled:
            return {"status": "disabled"}

        if len(self._outcomes) < self.min_samples:
            logger.info(
                f"[ThresholdOptimizer] Not enough samples to optimize "
                f"({len(self._outcomes)} < {self.min_samples})"
            )
            return {
                "status": "insufficient_samples",
                "sample_count": len(self._outcomes),
            }

        self._optimization_count += 1

        # Analyze outcomes by paradigm
        paradigm_stats = self._analyze_paradigms()

        # Adjust thresholds based on analysis
        adjustments = self._compute_adjustments(paradigm_stats)

        # Apply adjustments
        for threshold_type, adjustment in adjustments.items():
            current = self._thresholds[threshold_type].value
            new_value = current + adjustment
            self.set_threshold(threshold_type, new_value)

        # Clear old outcomes to free memory
        self._outcomes = self._outcomes[-self.min_samples :]

        logger.info(
            f"[ThresholdOptimizer] Optimization #{self._optimization_count} "
            f"complete: {len(adjustments)} thresholds adjusted"
        )

        return {
            "status": "success",
            "optimization_count": self._optimization_count,
            "adjustments": {k.value: v for k, v in adjustments.items()},
            "paradigm_stats": paradigm_stats,
        }

    def _analyze_paradigms(self) -> Dict[str, Dict[str, float]]:
        """Analyze outcomes by paradigm.

        Returns:
            Dict mapping paradigm to stats (success_rate, avg_tokens, avg_latency)
        """
        paradigm_outcomes: Dict[str, List[TaskOutcome]] = defaultdict(list)

        for outcome in self._outcomes:
            paradigm_outcomes[outcome.paradigm].append(outcome)

        stats = {}
        for paradigm, outcomes in paradigm_outcomes.items():
            success_rate = sum(o.success for o in outcomes) / len(outcomes)
            avg_tokens = sum(o.token_count for o in outcomes) / len(outcomes)
            avg_latency = sum(o.latency_ms for o in outcomes) / len(outcomes)

            stats[paradigm] = {
                "count": len(outcomes),
                "success_rate": success_rate,
                "avg_tokens": avg_tokens,
                "avg_latency": avg_latency,
            }

        return stats

    def _compute_adjustments(
        self, paradigm_stats: Dict[str, Dict[str, float]]
    ) -> Dict[ThresholdType, float]:
        """Compute threshold adjustments based on paradigm stats.

        Args:
            paradigm_stats: Statistics per paradigm

        Returns:
            Dict mapping threshold type to adjustment amount
        """
        adjustments = {}

        # Get direct paradigm stats
        direct_stats = paradigm_stats.get("direct", {})
        focused_stats = paradigm_stats.get("focused", {})

        # If direct paradigm has high success rate, can relax thresholds
        if direct_stats.get("count", 0) >= 10:
            direct_success = direct_stats.get("success_rate", 0)
            if direct_success > 0.95:
                # Can increase complexity threshold for direct
                adjustments[ThresholdType.COMPLEXITY_DIRECT] = self.adjustment_rate * 0.1
            elif direct_success < 0.85:
                # Need to tighten complexity threshold for direct
                adjustments[ThresholdType.COMPLEXITY_DIRECT] = -self.adjustment_rate * 0.1

        # If focused paradigm is efficient, can expand its range
        if focused_stats.get("count", 0) >= 10:
            focused_tokens = focused_stats.get("avg_tokens", 1000)
            if focused_tokens < 1200:  # Efficient
                # Can expand focused range
                adjustments[ThresholdType.COMPLEXITY_FOCUSED] = self.adjustment_rate * 0.05

        return adjustments

    def get_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Get all current thresholds.

        Returns:
            Dict mapping threshold name to config dict
        """
        return {k.value: v.to_dict() for k, v in self._thresholds.items()}

    def reset_thresholds(self) -> None:
        """Reset all thresholds to default values."""
        for threshold_type, config in self._thresholds.items():
            config.value = config.default_value
        logger.info("[ThresholdOptimizer] All thresholds reset to defaults")

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics.

        Returns:
            Dict with optimization statistics
        """
        return {
            "enabled": self.enabled,
            "total_outcomes": len(self._outcomes),
            "optimization_count": self._optimization_count,
            "thresholds": self.get_thresholds(),
        }

    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._outcomes = []
        self._optimization_count = 0


# Singleton instance
_threshold_optimizer_instance: Optional[ThresholdOptimizer] = None


def get_threshold_optimizer() -> ThresholdOptimizer:
    """Get the singleton ThresholdOptimizer instance.

    Returns:
        ThresholdOptimizer singleton instance
    """
    global _threshold_optimizer_instance
    if _threshold_optimizer_instance is None:
        _threshold_optimizer_instance = ThresholdOptimizer()
    return _threshold_optimizer_instance


__all__ = [
    "ThresholdOptimizer",
    "TaskOutcome",
    "ThresholdConfig",
    "ThresholdType",
    "get_threshold_optimizer",
]
