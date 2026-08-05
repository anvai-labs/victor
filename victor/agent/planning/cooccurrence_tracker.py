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

"""Co-occurrence Tracker for predictive tool selection.

Tracks tool usage patterns to predict which tools will be needed next.
Uses sequential pattern mining and conditional probability for predictions.

Key Concepts:
- Co-occurrence matrix: P(tool_b | tool_a) = probability of tool_b given tool_a
- Sequential patterns: Bigrams (tool_a → tool_b) and trigrams (tool_a → tool_b → tool_c)
- Task-type specificity: Different patterns for different task types
- Success rate weighting: Boost patterns that lead to successful outcomes

Usage:
    from victor.agent.planning.cooccurrence_tracker import CooccurrenceTracker

    tracker = CooccurrenceTracker()

    # Record tool sequence
    tracker.record_tool_sequence(
        tools=["search", "read", "edit"],
        task_type="bugfix",
        success=True,
    )

    # Predict next tools
    predictions = tracker.predict_next_tools(
        current_tools=["search", "read"],
        task_type="bugfix",
    )
    # Returns: [("edit", 0.85), ("write", 0.12), ...]
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ToolPattern:
    """A sequential tool pattern with confidence.

    Attributes:
        sequence: Ordered list of tool names
        support: Number of times this pattern was observed
        confidence: Conditional probability P(next | context)
        success_rate: Success rate when this pattern was used
        last_seen: Timestamp of most recent observation
    """

    sequence: List[str]
    support: int
    confidence: float
    success_rate: float
    last_seen: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate pattern data."""
        if self.support < 0:  # Changed from <= 0 to < 0 (allow 0 for new patterns)
            raise ValueError("Pattern support cannot be negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be in [0, 1]")
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("Success rate must be in [0, 1]")


@dataclass
class ToolPrediction:
    """Prediction for the next tool to use.

    Attributes:
        tool_name: Name of the predicted tool
        probability: Conditional probability (confidence)
        pattern_source: Source pattern that generated this prediction
        success_rate: Historical success rate for this prediction
    """

    tool_name: str
    probability: float
    pattern_source: Optional[str] = None
    success_rate: float = 0.5

    def __post_init__(self):
        """Validate prediction data."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Probability must be in [0, 1]")


class CooccurrenceTracker:
    """Tracks tool co-occurrence patterns for predictive selection.

    Maintains:
    - Co-occurrence matrix: P(tool_b | tool_a) for all tool pairs
    - Sequential patterns: Bigrams and trigrams
    - Task-type specificity: Separate patterns per task type
    - Success rates: Track which patterns lead to successful outcomes

    The tracker uses a sliding window to prioritize recent patterns while
    maintaining long-term statistics.
    """

    def __init__(
        self,
        max_history: int = 10000,
        min_observations: int = 3,
        decay_factor: float = 0.95,
        enable_task_type_specificity: bool = True,
    ):
        """Initialize the co-occurrence tracker.

        Args:
            max_history: Maximum number of tool sequences to keep in memory
            min_observations: Minimum observations before making predictions
            decay_factor: Exponential decay factor for old patterns (0.0-1.0)
            enable_task_type_specificity: Track patterns separately per task type
        """
        self.max_history = max_history
        self.min_observations = min_observations
        self.decay_factor = decay_factor
        self.enable_task_type_specificity = enable_task_type_specificity

        # Co-occurrence matrices: P(tool_b | tool_a)
        # Structure: {task_type: {tool_a: {tool_b: count}}}
        self._cooccurrence_matrices: Dict[str, Dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )

        # Sequential patterns: bigrams and trigrams
        # Structure: {task_type: [ToolPattern]}
        self._bigrams: Dict[str, List[ToolPattern]] = defaultdict(list)
        self._trigrams: Dict[str, List[ToolPattern]] = defaultdict(list)

        # Tool success rates
        # Structure: {task_type: {tool: (success_count, total_count)}}
        self._tool_success: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: (0, 0))
        )

        # Pattern success rates
        # Structure: {task_type: {pattern_key: (success_count, total_count)}}
        self._pattern_success: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: (0, 0))
        )

        # History buffer (for decay and sliding window)
        self._history: List[Tuple[str, List[str], bool, datetime]] = []
        self._total_sequences: int = 0

        logger.info(
            f"CooccurrenceTracker initialized (max_history={max_history}, "
            f"min_observations={min_observations}, decay={decay_factor})"
        )

    def record_tool_sequence(
        self,
        tools: List[str],
        task_type: str = "default",
        success: bool = True,
    ) -> None:
        """Record a tool usage sequence.

        Updates co-occurrence matrices, sequential patterns, and success rates.

        Args:
            tools: Ordered list of tool names used
            task_type: Type of task (for task-specific patterns)
            success: Whether the sequence led to a successful outcome
        """
        if not tools:
            return

        now = datetime.now()

        # Apply decay if history is full
        if len(self._history) >= self.max_history:
            self._apply_decay()

        # Normalize task type
        task_key = task_type if self.enable_task_type_specificity else "default"

        # Update co-occurrence matrix (forward direction only)
        for i, tool_a in enumerate(tools):
            for j in range(i + 1, min(i + 4, len(tools))):  # Look ahead up to 3 tools
                tool_b = tools[j]
                self._cooccurrence_matrices[task_key][tool_a][tool_b] += 1

                # Also ensure tool_b is in the matrix as a key (for backward lookups)
                if tool_b not in self._cooccurrence_matrices[task_key]:
                    self._cooccurrence_matrices[task_key][tool_b] = Counter()

        # Extract and record bigrams
        for i in range(len(tools) - 1):
            bigram = tools[i : i + 2]
            self._record_pattern(bigram, task_key, success, now)

        # Extract and record trigrams
        for i in range(len(tools) - 2):
            trigram = tools[i : i + 3]
            self._record_pattern(trigram, task_key, success, now)

        # Update tool success rates
        for tool in tools:
            success_count, total_count = self._tool_success[task_key][tool]
            self._tool_success[task_key][tool] = (
                success_count + (1 if success else 0),
                total_count + 1,
            )

        # Store in history
        self._history.append((task_key, list(tools), success, now))
        self._total_sequences += 1

        logger.debug(f"Recorded tool sequence: {tools} (task={task_type}, success={success})")

    def _record_pattern(
        self,
        pattern: List[str],
        task_key: str,
        success: bool,
        timestamp: datetime,
    ) -> None:
        """Record a sequential pattern with success tracking.

        Args:
            pattern: Tool sequence (bigram or trigram)
            task_key: Normalized task type
            success: Whether pattern led to success
            timestamp: When pattern was observed
        """
        pattern_key = "→".join(pattern)

        # Update pattern success rate
        success_count, total_count = self._pattern_success[task_key][pattern_key]
        self._pattern_success[task_key][pattern_key] = (
            success_count + (1 if success else 0),
            total_count + 1,
        )

        # Get or create pattern
        pattern_obj = self._find_or_create_pattern(pattern, task_key, timestamp)

        # Update support (count)
        pattern_obj.support += 1
        pattern_obj.last_seen = timestamp

        # Update success rate
        new_success_count, new_total_count = self._pattern_success[task_key][pattern_key]
        pattern_obj.success_rate = (
            new_success_count / new_total_count if new_total_count > 0 else 0.5
        )

    def _find_or_create_pattern(
        self,
        sequence: List[str],
        task_key: str,
        timestamp: datetime,
    ) -> ToolPattern:
        """Find existing pattern or create new one.

        Args:
            sequence: Tool sequence
            task_key: Task type
            timestamp: Observation time

        Returns:
            ToolPattern object
        """
        # Check if pattern exists
        if len(sequence) == 2:
            patterns = self._bigrams[task_key]
        elif len(sequence) == 3:
            patterns = self._trigrams[task_key]
        else:
            # Unsupported pattern length
            return ToolPattern(
                sequence=sequence,
                support=1,
                confidence=0.0,
                success_rate=0.5,
                last_seen=timestamp,
            )

        # Try to find existing pattern
        for p in patterns:
            if p.sequence == sequence:
                return p

        # Create new pattern
        new_pattern = ToolPattern(
            sequence=sequence,
            support=0,  # Will be incremented by _record_pattern
            confidence=0.0,  # Will be calculated on demand
            success_rate=0.5,
            last_seen=timestamp,
        )

        if len(sequence) == 2:
            self._bigrams[task_key].append(new_pattern)
        else:
            self._trigrams[task_key].append(new_pattern)

        return new_pattern

    def predict_next_tools(
        self,
        current_tools: List[str],
        task_type: str = "default",
        top_k: int = 5,
    ) -> List[ToolPrediction]:
        """Predict which tools will be needed next.

        Uses ensemble of:
        1. Direct co-occurrence: P(tool_b | tool_a)
        2. Sequential patterns: Bigram and trigram matching
        3. Success rate boosting: Prefer historically successful patterns

        Args:
            current_tools: Tools used so far in this session
            task_type: Type of task for task-specific predictions
            top_k: Maximum number of predictions to return

        Returns:
            List of ToolPrediction objects, sorted by probability
        """
        if not current_tools:
            return []

        task_key = task_type if self.enable_task_type_specificity else "default"

        # Collect predictions from multiple sources
        predictions: Dict[str, float] = defaultdict(float)

        # Source 1: Bigram matching (highest priority - immediate next tool)
        if len(current_tools) >= 1:
            bigram_predictions = self._predict_from_bigrams(current_tools, task_key)
            for tool, prob in bigram_predictions:
                predictions[tool] += prob * 2.0  # High weight for bigrams

        # Source 2: Direct co-occurrence from most recent tool (lower priority)
        if current_tools:
            last_tool = current_tools[-1]
            cooccurrence_counts = self._cooccurrence_matrices[task_key].get(last_tool, {})

            if cooccurrence_counts:
                total = sum(cooccurrence_counts.values())
                for next_tool, count in cooccurrence_counts.items():
                    # Only add if not already predicted by bigram (avoid double counting)
                    if next_tool not in predictions:
                        predictions[next_tool] += (count / total) * 0.5

        # Source 3: Trigram matching (medium priority)
        if len(current_tools) >= 2:
            trigram_predictions = self._predict_from_trigrams(current_tools, task_key)
            for tool, prob in trigram_predictions:
                predictions[tool] += prob * 1.0  # Medium weight for trigrams

        # Apply success rate boosting
        boosted_predictions = self._apply_success_rate_boosting(predictions, task_key)

        # Normalize probabilities
        total = sum(boosted_predictions.values())
        if total > 0:
            normalized = {k: v / total for k, v in boosted_predictions.items()}
        else:
            normalized = {}

        # Sort and convert to ToolPrediction objects
        sorted_predictions = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result = []
        for tool_name, probability in sorted_predictions:
            success_rate = self._get_tool_success_rate(tool_name, task_key)
            result.append(
                ToolPrediction(
                    tool_name=tool_name,
                    probability=probability,
                    pattern_source="cooccurrence",
                    success_rate=success_rate,
                )
            )

        logger.debug(
            f"Predicted next tools: {[p.tool_name for p in result]} "
            f"(task={task_type}, context={current_tools})"
        )

        return result

    def _predict_from_bigrams(
        self,
        current_tools: List[str],
        task_key: str,
    ) -> List[Tuple[str, float]]:
        """Predict next tools using bigram patterns.

        Args:
            current_tools: Recent tool usage
            task_key: Task type

        Returns:
            List of (tool, probability) tuples
        """
        predictions: Dict[str, float] = defaultdict(float)

        if not current_tools:
            return list(predictions.items())

        last_tool = current_tools[-1]

        # Find all bigrams starting with last_tool and calculate confidence
        matching_patterns = [
            p
            for p in self._bigrams[task_key]
            if len(p.sequence) == 2 and p.sequence[0] == last_tool
        ]

        if not matching_patterns:
            return list(predictions.items())

        # Calculate total support for normalization
        total_support = sum(p.support for p in matching_patterns)

        for pattern in matching_patterns:
            next_tool = pattern.sequence[1]
            # Confidence = support / total_support
            confidence = pattern.support / total_support if total_support > 0 else 0.0
            predictions[next_tool] += confidence * pattern.success_rate

        return list(predictions.items())

    def _predict_from_trigrams(
        self,
        current_tools: List[str],
        task_key: str,
    ) -> List[Tuple[str, float]]:
        """Predict next tools using trigram patterns.

        Args:
            current_tools: Recent tool usage
            task_key: Task type

        Returns:
            List of (tool, probability) tuples
        """
        predictions: Dict[str, float] = defaultdict(float)

        if len(current_tools) < 2:
            return list(predictions.items())

        last_two = current_tools[-2:]

        # Find all trigrams matching the last 2 tools
        matching_patterns = [
            p
            for p in self._trigrams[task_key]
            if (
                len(p.sequence) == 3
                and p.sequence[0] == last_two[0]
                and p.sequence[1] == last_two[1]
            )
        ]

        if not matching_patterns:
            return list(predictions.items())

        # Calculate total support for normalization
        total_support = sum(p.support for p in matching_patterns)

        for pattern in matching_patterns:
            next_tool = pattern.sequence[2]
            # Confidence = support / total_support
            confidence = pattern.support / total_support if total_support > 0 else 0.0
            predictions[next_tool] += confidence * pattern.success_rate

        return list(predictions.items())

    def _apply_success_rate_boosting(
        self,
        predictions: Dict[str, float],
        task_key: str,
    ) -> Dict[str, float]:
        """Apply success rate boosting to predictions.

        Tools with higher success rates get boosted.

        Args:
            predictions: Raw prediction scores
            task_key: Task type

        Returns:
            Boosted predictions
        """
        boosted = {}

        for tool, score in predictions.items():
            success_rate = self._get_tool_success_rate(tool, task_key)

            # Boost factor: 0.5x for low success, 2.0x for high success
            boost_factor = 0.5 + (success_rate * 1.5)

            boosted[tool] = score * boost_factor

        return boosted

    def _get_tool_success_rate(self, tool: str, task_key: str) -> float:
        """Get success rate for a tool.

        Args:
            tool: Tool name
            task_key: Task type

        Returns:
            Success rate (0.0-1.0)
        """
        success_count, total_count = self._tool_success[task_key][tool]

        if total_count == 0:
            return 0.5  # Default for unobserved tools

        return success_count / total_count

    def _apply_decay(self) -> None:
        """Apply exponential decay to old patterns.

        Removes oldest entries from history and decays pattern counts.
        """
        # Remove oldest entries
        if len(self._history) > self.max_history // 2:
            remove_count = len(self._history) - self.max_history // 2
            removed = self._history[:remove_count]
            self._history = self._history[remove_count:]

            # Decay co-occurrence matrices
            for task_key, tools, success, _ in removed:
                for i, tool_a in enumerate(tools):
                    for j in range(i + 1, min(i + 4, len(tools))):
                        tool_b = tools[j]
                        matrix = self._cooccurrence_matrices[task_key][tool_a]
                        if tool_b in matrix:
                            matrix[tool_b] = max(1, int(matrix[tool_b] * self.decay_factor))

            logger.debug(f"Applied decay to {remove_count} old sequences")

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics.

        Returns:
            Dictionary with tracker metrics
        """
        # Count total patterns
        total_bigrams = sum(len(patterns) for patterns in self._bigrams.values())
        total_trigrams = sum(len(patterns) for patterns in self._trigrams.values())

        # Count unique tools
        unique_tools = set()
        for matrix in self._cooccurrence_matrices.values():
            unique_tools.update(matrix.keys())
            for counter in matrix.values():
                unique_tools.update(counter.keys())

        return {
            "total_sequences_recorded": self._total_sequences,
            "history_size": len(self._history),
            "unique_tools": len(unique_tools),
            "total_bigrams": total_bigrams,
            "total_trigrams": total_trigrams,
            "task_types": list(self._cooccurrence_matrices.keys()),
        }

    def reset(self) -> None:
        """Reset all tracking data."""
        self._cooccurrence_matrices.clear()
        self._bigrams.clear()
        self._trigrams.clear()
        self._tool_success.clear()
        self._pattern_success.clear()
        self._history.clear()
        self._total_sequences = 0

        logger.info("CooccurrenceTracker reset")


__all__ = [
    "CooccurrenceTracker",
    "ToolPattern",
    "ToolPrediction",
]
