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

"""Completion Scorer - Combines multiple signals into unified completion score.

This module provides a unified scoring mechanism that combines multiple signals
to determine task completion:
1. Requirement satisfaction (from RequirementValidator)
2. Fulfillment score (from FulfillmentDetector)
3. Keyword heuristics (existing patterns)
4. Perception confidence (calibrated)
5. Task complexity adjustment

Design Principles:
1. Multi-signal fusion: Combine diverse signals for robust detection
2. Configurable weights: Allow tuning via settings
3. Explainable scores: Clear contribution from each signal
4. Adaptive thresholds: Different thresholds for different task types

Based on research from:
- arXiv:2604.07415 - SubSearch intermediate reward design
- arXiv:2601.21268 - Meta-evaluation without ground truth

Example:
    from victor.framework.completion_sorer import CompletionScorer

    scorer = CompletionScorer()

    score = scorer.calculate_completion_score(
        requirement_result=validation_result,
        fulfillment_result=fulfillment,
        keyword_result=completion_signal,
        perception=perception,
        task_type=TaskType.CODE_GENERATION,
    )

    if score >= 0.8:
        print("Task complete!")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.framework.perception_integration import Perception
    from victor.framework.requirement_validator import RequirementResult
    from victor.framework.fulfillment import FulfillmentResult
    from victor.agent.task_analyzer import TaskComplexity


class TaskType(Enum):
    """Task types for completion detection."""

    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    TESTING = "testing"
    DEBUGGING = "debugging"
    SEARCH = "search"
    ANALYSIS = "analysis"
    SETUP = "setup"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    UNKNOWN = "unknown"


@dataclass
class CompletionSignal:
    """Signal from keyword-based completion detection."""

    has_completion_indicator: bool
    has_complete_code: bool
    has_structure: bool
    is_continuation_request: bool
    confidence: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class CompletionScore:
    """Unified completion score with breakdown.

    Attributes:
        total_score: Combined score from all signals (0.0-1.0)
        requirement_score: Requirement satisfaction component
        fulfillment_score: Task-specific fulfillment component
        keyword_score: Keyword heuristic component
        confidence_score: Perception confidence component
        complexity_adjustment: Task complexity adjustment
        is_complete: True if score exceeds threshold
        threshold: Threshold used for completion decision
        breakdown: Detailed breakdown of each component
    """

    total_score: float
    requirement_score: float
    fulfillment_score: float
    keyword_score: float
    confidence_score: float
    complexity_adjustment: float
    is_complete: bool
    threshold: float
    breakdown: Dict[str, Any] = field(default_factory=dict)


class CompletionScorer:
    """Combines multiple signals into unified completion score.

    This scorer implements a weighted sum of multiple signals:
    - Requirement satisfaction (35% weight): Primary signal
    - Fulfillment score (25% weight): Task-specific validation
    - Keyword confidence (20% weight): Existing heuristics
    - Perception confidence (15% weight): Calibrated confidence
    - Complexity adjustment (5% weight): Simpler tasks = higher threshold

    Usage:
        scorer = CompletionScorer()

        score = scorer.calculate_completion_score(
            requirement_result=validation_result,
            fulfillment_result=fulfillment,
            keyword_result=completion_signal,
            perception=perception,
            task_type=TaskType.CODE_GENERATION,
        )
    """

    def __init__(
        self,
        requirement_weight: float = 0.35,
        fulfillment_weight: float = 0.25,
        keyword_weight: float = 0.20,
        confidence_weight: float = 0.15,
        complexity_weight: float = 0.05,
        default_threshold: float = 0.80,
    ):
        """Initialize scorer with weights.

        Args:
            requirement_weight: Weight for requirement satisfaction (default: 0.35)
            fulfillment_weight: Weight for fulfillment score (default: 0.25)
            keyword_weight: Weight for keyword confidence (default: 0.20)
            confidence_weight: Weight for perception confidence (default: 0.15)
            complexity_weight: Weight for complexity adjustment (default: 0.05)
            default_threshold: Default threshold for completion (default: 0.80)
        """
        # Validate weights sum to ~1.0
        total_weight = (
            requirement_weight
            + fulfillment_weight
            + keyword_weight
            + confidence_weight
            + complexity_weight
        )
        if abs(total_weight - 1.0) > 0.1:
            logger.warning(
                f"CompletionScorer weights sum to {total_weight:.2f}, "
                f"expected ~1.0. Normalizing."
            )
            # Normalize weights
            scale = 1.0 / total_weight
            requirement_weight *= scale
            fulfillment_weight *= scale
            keyword_weight *= scale
            confidence_weight *= scale
            complexity_weight *= scale

        self.requirement_weight = requirement_weight
        self.fulfillment_weight = fulfillment_weight
        self.keyword_weight = keyword_weight
        self.confidence_weight = confidence_weight
        self.complexity_weight = complexity_weight
        self.default_threshold = default_threshold

        # Task-specific thresholds
        self.task_thresholds = {
            TaskType.CODE_GENERATION: 0.85,  # Higher threshold for code
            TaskType.TESTING: 0.80,  # High threshold for correctness
            TaskType.DEBUGGING: 0.75,  # Medium threshold (fix verification)
            TaskType.SEARCH: 0.70,  # Lower threshold (information retrieval)
            TaskType.ANALYSIS: 0.75,  # Medium threshold
            TaskType.SETUP: 0.80,  # High threshold (must work)
            TaskType.DOCUMENTATION: 0.70,  # Lower threshold (subjective)
            TaskType.DEPLOYMENT: 0.85,  # High threshold (must succeed)
            TaskType.UNKNOWN: 0.75,  # Default medium threshold
        }

    def calculate_completion_score(
        self,
        requirement_result: Optional[RequirementResult],
        fulfillment_result: Optional[Any],
        keyword_result: Optional[CompletionSignal],
        perception: Optional[Perception],
        task_type: TaskType = TaskType.UNKNOWN,
    ) -> CompletionScore:
        """Calculate unified completion score from multiple signals.

        Args:
            requirement_result: Result from RequirementValidator
            fulfillment_result: Result from FulfillmentDetector
            keyword_result: Result from keyword detection
            perception: Perception from PerceptionIntegration
            task_type: Type of task for threshold selection

        Returns:
            CompletionScore with total score and breakdown
        """
        # Extract individual scores
        requirement_score = self._extract_requirement_score(requirement_result)
        fulfillment_score = self._extract_fulfillment_score(fulfillment_result)
        keyword_score = self._extract_keyword_score(keyword_result)
        confidence_score = self._extract_confidence_score(perception)
        complexity_adjustment = self._calculate_complexity_adjustment(perception, task_type)

        # Calculate weighted sum
        total_score = (
            requirement_score * self.requirement_weight
            + fulfillment_score * self.fulfillment_weight
            + keyword_score * self.keyword_weight
            + confidence_score * self.confidence_weight
            + complexity_adjustment * self.complexity_weight
        )

        # Get threshold for task type
        threshold = self.task_thresholds.get(task_type, self.default_threshold)

        # Determine if complete
        is_complete = total_score >= threshold

        # Build breakdown
        breakdown = {
            "requirement": {
                "score": requirement_score,
                "weight": self.requirement_weight,
                "contribution": requirement_score * self.requirement_weight,
            },
            "fulfillment": {
                "score": fulfillment_score,
                "weight": self.fulfillment_weight,
                "contribution": fulfillment_score * self.fulfillment_weight,
            },
            "keyword": {
                "score": keyword_score,
                "weight": self.keyword_weight,
                "contribution": keyword_score * self.keyword_weight,
            },
            "confidence": {
                "score": confidence_score,
                "weight": self.confidence_weight,
                "contribution": confidence_score * self.confidence_weight,
            },
            "complexity": {
                "score": complexity_adjustment,
                "weight": self.complexity_weight,
                "contribution": complexity_adjustment * self.complexity_weight,
            },
        }

        return CompletionScore(
            total_score=total_score,
            requirement_score=requirement_score,
            fulfillment_score=fulfillment_score,
            keyword_score=keyword_score,
            confidence_score=confidence_score,
            complexity_adjustment=complexity_adjustment,
            is_complete=is_complete,
            threshold=threshold,
            breakdown=breakdown,
        )

    def _extract_requirement_score(self, requirement_result: Optional[RequirementResult]) -> float:
        """Extract requirement satisfaction score."""
        if requirement_result is None:
            # No requirements available - neutral score
            return 0.5

        if requirement_result.is_satisfied:
            # All requirements met - high score
            return 0.95
        else:
            # Use satisfaction score from validation result
            return requirement_result.satisfaction_score

    def _extract_fulfillment_score(self, fulfillment_result: Optional[Any]) -> float:
        """Extract fulfillment score."""
        if fulfillment_result is None:
            # No fulfillment check - neutral score
            return 0.5

        # Check if it's a FulfillmentResult with score attribute
        if hasattr(fulfillment_result, "score"):
            return fulfillment_result.score

        # Check if it's a FulfillmentResult with is_fulfilled attribute
        if hasattr(fulfillment_result, "is_fulfilled"):
            if fulfillment_result.is_fulfilled:
                return 0.95
            elif hasattr(fulfillment_result, "is_partial") and fulfillment_result.is_partial:
                return 0.6
            else:
                return 0.3

        # Unknown format - neutral score
        return 0.5

    def _extract_keyword_score(self, keyword_result: Optional[CompletionSignal]) -> float:
        """Extract keyword confidence score."""
        if keyword_result is None:
            # No keyword detection - neutral score
            return 0.5

        # Use confidence from signal
        if keyword_result.is_continuation_request:
            # Model wants to continue - low completion score
            return 0.3

        if keyword_result.has_completion_indicator:
            # Strong completion signal
            return 0.9

        if keyword_result.has_complete_code or keyword_result.has_structure:
            # Moderate completion signal
            return 0.75

        # Weak or no signal
        return keyword_result.confidence

    def _extract_confidence_score(self, perception: Optional[Perception]) -> float:
        """Extract calibrated perception confidence."""
        if perception is None:
            # No perception - neutral score
            return 0.5

        # Check if perception has confidence attribute
        if hasattr(perception, "confidence"):
            confidence = perception.confidence

            # Calibrate: geometric mean fusion (from PerceptionIntegration)
            # This downgrades overconfident predictions
            if hasattr(perception, "intent_confidence"):
                intent_confidence = perception.intent_confidence
                # Geometric mean: sqrt(confidence * intent_confidence)
                calibrated = (confidence * intent_confidence) ** 0.5
                return calibrated

            return confidence

        # No confidence available - neutral score
        return 0.5

    def _calculate_complexity_adjustment(
        self, perception: Optional[Perception], task_type: TaskType
    ) -> float:
        """Calculate complexity adjustment.

        Simpler tasks get higher scores (easier to complete).
        Complex tasks get lower scores (harder to complete).

        Returns:
            Adjustment factor (0.0-1.0)
        """
        # Default: neutral adjustment
        if perception is None:
            return 0.5

        # Check if perception has complexity information
        complexity = None
        if hasattr(perception, "complexity"):
            complexity = perception.complexity
        elif hasattr(perception, "task_complexity"):
            complexity = perception.task_complexity

        if complexity is None:
            # No complexity info - use task type defaults
            # Simpler tasks (search, analysis) get higher adjustment
            # Complex tasks (code_generation, setup) get lower adjustment
            task_complexity_map = {
                TaskType.SEARCH: 0.8,
                TaskType.ANALYSIS: 0.75,
                TaskType.DOCUMENTATION: 0.8,
                TaskType.DEBUGGING: 0.6,
                TaskType.TESTING: 0.6,
                TaskType.CODE_GENERATION: 0.5,
                TaskType.SETUP: 0.5,
                TaskType.DEPLOYMENT: 0.5,
                TaskType.UNKNOWN: 0.6,
            }
            return task_complexity_map.get(task_type, 0.6)

        # Use complexity from perception
        # Assuming complexity is an enum with values like LOW, MEDIUM, HIGH
        complexity_str = str(complexity).lower()

        if "low" in complexity_str or "simple" in complexity_str:
            return 0.8
        elif "medium" in complexity_str:
            return 0.6
        elif "high" in complexity_str or "complex" in complexity_str:
            return 0.4
        else:
            return 0.6  # Default to medium

    def get_threshold(self, task_type: TaskType) -> float:
        """Get completion threshold for task type.

        Args:
            task_type: Type of task

        Returns:
            Threshold (0.0-1.0)
        """
        return self.task_thresholds.get(task_type, self.default_threshold)

    def set_threshold(self, task_type: TaskType, threshold: float) -> None:
        """Set completion threshold for task type.

        Args:
            task_type: Type of task
            threshold: New threshold (0.0-1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.task_thresholds[task_type] = threshold
        else:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
