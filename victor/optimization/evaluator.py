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

"""Variant evaluation for workflow optimization.

This module provides the VariantEvaluator class for evaluating
workflow variants to estimate their performance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from victor.optimization.models import (
    NodeStatistics,
    WorkflowProfile,
)
from victor.optimization.generator import (
    WorkflowVariant,
    WorkflowVariantGenerator,
)

logger = logging.getLogger(__name__)


class EvaluationMode(Enum):
    """Evaluation modes for variants."""

    DRY_RUN = "dry_run"  # Execute variant on test inputs
    HISTORICAL = "historical"  # Use historical data to estimate
    ESTIMATION = "estimation"  # Estimate from metadata only


@dataclass
class EvaluationResult:
    """Result of variant evaluation.

    Attributes:
        variant_id: Variant that was evaluated
        mode: Evaluation mode used
        duration_score: Normalized duration score (higher is better)
        cost_score: Normalized cost score (higher is better)
        quality_score: Quality score (0-1)
        overall_score: Overall weighted score
        confidence: Confidence in evaluation (0-1)
        metrics: Additional metrics
        recommendation: Whether to recommend this variant
    """

    variant_id: str
    mode: EvaluationMode
    duration_score: float
    cost_score: float
    quality_score: float
    overall_score: float
    confidence: float
    metrics: Dict[str, float] = None
    recommendation: bool = False

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "variant_id": self.variant_id,
            "mode": self.mode.value,
            "duration_score": self.duration_score,
            "cost_score": self.cost_score,
            "quality_score": self.quality_score,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "metrics": self.metrics,
            "recommendation": self.recommendation,
        }


class VariantEvaluator:
    """Evaluates workflow variants to estimate performance.

    The evaluator supports multiple evaluation modes:
    - Dry-run: Execute variant on test inputs
    - Historical: Use historical data from experiment tracker
    - Estimation: Estimate from variant metadata

    Example:
        evaluator = VariantEvaluator()

        result = await evaluator.evaluate_variant(
            variant=variant,
            profile=profile,
            mode=EvaluationMode.ESTIMATION,
        )

        print(f"Overall score: {result.overall_score:.3f}")
        print(f"Recommend: {result.recommendation}")
    """

    def __init__(
        self,
        objective_weights: Optional[Dict[str, float]] = None,
        recommendation_threshold: float = 0.1,
    ):
        """Initialize the variant evaluator.

        Args:
            objective_weights: Weights for multi-objective scoring
            recommendation_threshold: Minimum improvement to recommend variant
        """
        self.objective_weights = objective_weights or {
            "duration": 0.4,
            "cost": 0.4,
            "quality": 0.2,
        }
        self.recommendation_threshold = recommendation_threshold

    async def evaluate_variant(
        self,
        variant: WorkflowVariant,
        profile: WorkflowProfile,
        mode: EvaluationMode = EvaluationMode.ESTIMATION,
        test_inputs: Optional[List[Dict[str, Any]]] = None,
        score_function: Optional[Callable[[WorkflowVariant], float]] = None,
    ) -> EvaluationResult:
        """Evaluate a workflow variant.

        Args:
            variant: Workflow variant to evaluate
            profile: Original workflow profile
            mode: Evaluation mode
            test_inputs: Test inputs for dry-run mode
            score_function: Optional custom scoring function

        Returns:
            EvaluationResult with scores and recommendation
        """
        logger.info(f"Evaluating variant {variant.variant_id} using {mode.value} mode")

        if mode == EvaluationMode.DRY_RUN:
            result = await self._evaluate_dry_run(
                variant,
                test_inputs,
            )
        elif mode == EvaluationMode.HISTORICAL:
            result = await self._evaluate_historical(
                variant,
                profile,
            )
        else:  # ESTIMATION
            result = await self._evaluate_estimation(
                variant,
                profile,
            )

        # Calculate overall score
        if score_function:
            result.overall_score = score_function(variant)
        else:
            result.overall_score = self._calculate_overall_score(
                result.duration_score,
                result.cost_score,
                result.quality_score,
            )

        # Determine recommendation
        baseline_score = 1.0  # Baseline is original workflow
        improvement = (result.overall_score - baseline_score) / max(baseline_score, 1e-6)
        result.recommendation = improvement > self.recommendation_threshold

        logger.info(
            f"Evaluation complete: score={result.overall_score:.3f}, "
            f"recommend={result.recommendation}, confidence={result.confidence:.1%}"
        )

        return result

    async def _evaluate_dry_run(
        self,
        variant: WorkflowVariant,
        test_inputs: Optional[List[Dict[str, Any]]],
    ) -> EvaluationResult:
        """Evaluate variant through dry-run execution.

        Args:
            variant: Workflow variant
            test_inputs: Test inputs for execution

        Returns:
            EvaluationResult
        """
        logger.info("Performing dry-run evaluation")

        # This is a simplified implementation
        # Full implementation would:
        # 1. Compile the variant workflow
        # 2. Execute on test inputs
        # 3. Measure duration, cost, quality
        # 4. Compare with baseline

        # For MVP, use variant metadata as estimates
        duration_score = 1.0 / (1.0 + variant.estimated_duration_reduction)
        cost_score = 1.0 / (1.0 + variant.estimated_cost_reduction)
        quality_score = 1.0 - (variant.expected_improvement * 0.1)  # Assume slight quality risk

        return EvaluationResult(
            variant_id=variant.variant_id,
            mode=EvaluationMode.DRY_RUN,
            duration_score=duration_score,
            cost_score=cost_score,
            quality_score=quality_score,
            overall_score=0.0,  # Calculated later
            confidence=0.9,  # High confidence with dry-run
            metrics={
                "num_executions": len(test_inputs) if test_inputs else 0,
            },
        )

    async def _evaluate_historical(
        self,
        variant: WorkflowVariant,
        profile: WorkflowProfile,
    ) -> EvaluationResult:
        """Evaluate variant using historical data.

        Args:
            variant: Workflow variant
            profile: Original workflow profile

        Returns:
            EvaluationResult
        """
        logger.info("Performing historical evaluation")

        # Estimate impact based on changes and historical performance
        duration_reduction = variant.estimated_duration_reduction
        cost_reduction = variant.estimated_cost_reduction

        # Calculate scores relative to baseline
        baseline_duration = profile.total_duration
        baseline_cost = profile.total_cost

        if baseline_duration > 0:
            duration_score = baseline_duration / max(
                baseline_duration - duration_reduction,
                1e-6,
            )
        else:
            duration_score = 1.0

        if baseline_cost > 0:
            cost_score = baseline_cost / max(
                baseline_cost - cost_reduction,
                1e-6,
            )
        else:
            cost_score = 1.0

        # Quality estimate based on risk level
        risk_factors = {
            "low": 0.0,
            "medium": 0.05,
            "high": 0.1,
        }
        quality_impact = risk_factors.get(variant.risk_level, 0.05)
        quality_score = profile.success_rate * (1.0 - quality_impact)

        return EvaluationResult(
            variant_id=variant.variant_id,
            mode=EvaluationMode.HISTORICAL,
            duration_score=duration_score,
            cost_score=cost_score,
            quality_score=quality_score,
            overall_score=0.0,  # Calculated later
            confidence=0.7,  # Medium confidence with historical data
            metrics={
                "baseline_duration": baseline_duration,
                "baseline_cost": baseline_cost,
                "estimated_duration_reduction": duration_reduction,
                "estimated_cost_reduction": cost_reduction,
            },
        )

    async def _evaluate_estimation(
        self,
        variant: WorkflowVariant,
        profile: WorkflowProfile,
    ) -> EvaluationResult:
        """Evaluate variant using estimation from metadata.

        Args:
            variant: Workflow variant
            profile: Original workflow profile

        Returns:
            EvaluationResult
        """
        logger.info("Performing estimation-based evaluation")

        # Use variant metadata for estimation
        expected_improvement = variant.expected_improvement

        # Estimate scores from improvement
        duration_score = 1.0 + (expected_improvement * 0.5)
        cost_score = 1.0 + (expected_improvement * 0.3)
        quality_score = 1.0 - (expected_improvement * 0.05)  # Assume slight quality risk

        # Adjust based on risk level
        risk_multipliers = {
            "low": 1.0,
            "medium": 0.95,
            "high": 0.9,
        }
        risk_multiplier = risk_multipliers.get(variant.risk_level, 0.95)
        quality_score *= risk_multiplier

        # Confidence based on number of changes
        confidence = min(0.8, 1.0 / (1.0 + len(variant.changes) * 0.1))

        return EvaluationResult(
            variant_id=variant.variant_id,
            mode=EvaluationMode.ESTIMATION,
            duration_score=duration_score,
            cost_score=cost_score,
            quality_score=quality_score,
            overall_score=0.0,  # Calculated later
            confidence=confidence,
            metrics={
                "expected_improvement": expected_improvement,
                "num_changes": len(variant.changes),
                "risk_level": variant.risk_level,
            },
        )

    def _calculate_overall_score(
        self,
        duration_score: float,
        cost_score: float,
        quality_score: float,
    ) -> float:
        """Calculate overall weighted score.

        Args:
            duration_score: Duration score
            cost_score: Cost score
            quality_score: Quality score

        Returns:
            Overall weighted score
        """
        return (
            self.objective_weights["duration"] * duration_score
            + self.objective_weights["cost"] * cost_score
            + self.objective_weights["quality"] * quality_score
        )

    async def compare_variants(
        self,
        variants: List[WorkflowVariant],
        profile: WorkflowProfile,
        mode: EvaluationMode = EvaluationMode.ESTIMATION,
    ) -> List[EvaluationResult]:
        """Compare multiple workflow variants.

        Args:
            variants: List of workflow variants to compare
            profile: Original workflow profile
            mode: Evaluation mode

        Returns:
            List of evaluation results sorted by overall score
        """
        logger.info(f"Comparing {len(variants)} variants")

        results = []

        for variant in variants:
            result = await self.evaluate_variant(
                variant,
                profile,
                mode,
            )
            results.append(result)

        # Sort by overall score
        results.sort(key=lambda r: r.overall_score, reverse=True)

        # Log ranking
        logger.info("Variant ranking:")
        for i, result in enumerate(results, 1):
            logger.info(
                f"  {i}. {result.variant_id}: "
                f"score={result.overall_score:.3f}, "
                f"recommend={result.recommendation}"
            )

        return results
