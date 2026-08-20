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

"""Quality assessment protocols and implementations.

This module defines the interface for response quality assessment using
Liskov Substitution and Dependency Inversion principles.

Design Patterns:
- Strategy Pattern: Different quality assessment strategies
- Chain of Responsibility: Multiple assessors can be chained
- Dependency Inversion: Core code depends on IQualityAssessor interface

Quality Dimensions:
- Grounding: Is the response factually accurate based on code?
- Coverage: Does the response address all parts of the query?
- Clarity: Is the response clear and well-structured?
- Correctness: Is any generated code syntactically and semantically correct?
- Conciseness: Is the response appropriately concise?

Usage:
    assessor = ProviderAwareQualityAssessor(provider_adapter)
    score = assessor.assess(response, context)

    if score.is_acceptable:
        # Response meets quality threshold
        pass
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import re


class ProtocolQualityDimension(str, Enum):
    """Protocol-level quality dimensions for response assessment.

    Renamed from QualityDimension to be semantically distinct:
    - ResponseQualityDimension (victor.agent.response_quality): LLM response quality
    - ProtocolQualityDimension (here): Protocol-level quality (includes GROUNDING, SAFETY)
    """

    GROUNDING = "grounding"
    COVERAGE = "coverage"
    CLARITY = "clarity"
    CORRECTNESS = "correctness"
    CONCISENESS = "conciseness"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"


@dataclass
class DimensionScore:
    """Score for a single quality dimension.

    Attributes:
        dimension: The quality dimension
        score: Score value (0.0-1.0)
        weight: Weight for aggregation
        reason: Explanation for the score
        evidence: Supporting evidence
    """

    dimension: ProtocolQualityDimension
    score: float
    weight: float = 1.0
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityScore:
    """Overall quality assessment result.

    Attributes:
        score: Overall quality score (0.0-1.0)
        is_acceptable: Whether score meets threshold
        threshold: The threshold used for acceptability
        provider: Provider name for context
        dimension_scores: Individual dimension scores
        feedback: Human-readable feedback
        suggestions: Improvement suggestions
    """

    score: float
    is_acceptable: bool
    threshold: float = 0.80
    provider: str = ""
    dimension_scores: Dict[ProtocolQualityDimension, DimensionScore] = field(default_factory=dict)
    feedback: str = ""
    suggestions: List[str] = field(default_factory=list)

    def get_dimension_score(self, dimension: ProtocolQualityDimension) -> float:
        """Get score for a specific dimension."""
        if dimension in self.dimension_scores:
            return self.dimension_scores[dimension].score
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score": self.score,
            "is_acceptable": self.is_acceptable,
            "threshold": self.threshold,
            "provider": self.provider,
            "dimensions": {
                dim.value: {
                    "score": ds.score,
                    "weight": ds.weight,
                    "reason": ds.reason,
                }
                for dim, ds in self.dimension_scores.items()
            },
            "feedback": self.feedback,
            "suggestions": self.suggestions,
        }


@runtime_checkable
class IQualityAssessor(Protocol):
    """Interface for quality assessment.

    Implementations should be interchangeable (Liskov Substitution).
    """

    def assess(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> QualityScore:
        """Assess response quality.

        Args:
            response: The response text to assess
            context: Additional context (query, provider, etc.)

        Returns:
            Quality score with dimension breakdown
        """
        ...

    @property
    def dimensions(self) -> List[ProtocolQualityDimension]:
        """Return dimensions this assessor evaluates."""
        ...


class BaseQualityAssessor(ABC):
    """Abstract base class for quality assessors."""

    def __init__(
        self,
        weights: Optional[Dict[ProtocolQualityDimension, float]] = None,
        threshold: float = 0.80,
    ):
        """Initialize assessor.

        Args:
            weights: Weights for each dimension
            threshold: Minimum score for acceptability
        """
        self._weights = weights or {
            ProtocolQualityDimension.GROUNDING: 0.25,
            ProtocolQualityDimension.COVERAGE: 0.20,
            ProtocolQualityDimension.CLARITY: 0.15,
            ProtocolQualityDimension.CORRECTNESS: 0.25,
            ProtocolQualityDimension.CONCISENESS: 0.15,
        }
        self._threshold = threshold

    @property
    def dimensions(self) -> List[ProtocolQualityDimension]:
        """Return dimensions this assessor evaluates."""
        return list(self._weights.keys())

    @abstractmethod
    def _assess_dimension(
        self,
        dimension: ProtocolQualityDimension,
        response: str,
        context: Dict[str, Any],
    ) -> DimensionScore:
        """Assess a single quality dimension."""
        ...

    def assess(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> QualityScore:
        """Assess response quality across all dimensions."""
        dimension_scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for dimension, weight in self._weights.items():
            score = self._assess_dimension(dimension, response, context)
            score.weight = weight
            dimension_scores[dimension] = score

            weighted_sum += score.score * weight
            total_weight += weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return QualityScore(
            score=overall_score,
            is_acceptable=overall_score >= self._threshold,
            threshold=self._threshold,
            dimension_scores=dimension_scores,
        )


class SimpleQualityAssessor(BaseQualityAssessor):
    """Simple heuristic-based quality assessor."""

    def _assess_dimension(
        self,
        dimension: ProtocolQualityDimension,
        response: str,
        context: Dict[str, Any],
    ) -> DimensionScore:
        """Assess using simple heuristics."""
        if dimension == ProtocolQualityDimension.CLARITY:
            return self._assess_clarity(response)
        elif dimension == ProtocolQualityDimension.CONCISENESS:
            return self._assess_conciseness(response, context)
        elif dimension == ProtocolQualityDimension.CORRECTNESS:
            return self._assess_correctness(response)
        elif dimension == ProtocolQualityDimension.COVERAGE:
            return self._assess_coverage(response, context)
        elif dimension == ProtocolQualityDimension.GROUNDING:
            return self._assess_grounding(response, context)
        else:
            return DimensionScore(dimension=dimension, score=0.5)

    def _assess_clarity(self, response: str) -> DimensionScore:
        """Assess response clarity."""
        score = 0.5

        # Check for structure (headers, lists)
        if re.search(r"^#+\s", response, re.MULTILINE):
            score += 0.1
        if re.search(r"^\s*[-*]\s", response, re.MULTILINE):
            score += 0.1
        if re.search(r"^\s*\d+\.\s", response, re.MULTILINE):
            score += 0.1

        # Check for code blocks
        if "```" in response:
            score += 0.1

        # Penalize very long unbroken paragraphs
        paragraphs = response.split("\n\n")
        long_paragraphs = sum(1 for p in paragraphs if len(p) > 500)
        if long_paragraphs > 2:
            score -= 0.1

        return DimensionScore(
            dimension=ProtocolQualityDimension.CLARITY,
            score=min(1.0, max(0.0, score)),
            reason="Assessed based on structure and formatting",
        )

    def _assess_conciseness(self, response: str, context: Dict[str, Any]) -> DimensionScore:
        """Assess response conciseness."""
        query = context.get("query", "")
        query_len = len(query.split())
        response_len = len(response.split())

        # Ideal ratio depends on query type
        if "explain" in query.lower() or "how" in query.lower():
            ideal_ratio = 20  # Explanations can be longer
        elif "fix" in query.lower() or "bug" in query.lower():
            ideal_ratio = 10  # Fixes should be focused
        else:
            ideal_ratio = 15

        actual_ratio = response_len / max(query_len, 1)

        # Score based on ratio
        if actual_ratio <= ideal_ratio:
            score = 0.9
        elif actual_ratio <= ideal_ratio * 1.5:
            score = 0.7
        elif actual_ratio <= ideal_ratio * 2:
            score = 0.5
        else:
            score = 0.3

        return DimensionScore(
            dimension=ProtocolQualityDimension.CONCISENESS,
            score=score,
            reason=f"Response/query ratio: {actual_ratio:.1f} (ideal: {ideal_ratio})",
            evidence={"response_words": response_len, "query_words": query_len},
        )

    def _assess_correctness(self, response: str) -> DimensionScore:
        """Assess code correctness heuristics."""
        score = 0.7  # Start neutral

        # Check for code blocks
        code_blocks = re.findall(r"```(\w*)\n(.*?)```", response, re.DOTALL)

        if code_blocks:
            for lang, code in code_blocks:
                # Check for obvious syntax issues
                if lang in ("python", "py"):
                    # Check balanced parens/brackets
                    if code.count("(") != code.count(")"):
                        score -= 0.1
                    if code.count("[") != code.count("]"):
                        score -= 0.1
                    if code.count("{") != code.count("}"):
                        score -= 0.1

                    # Check for TODO/FIXME left in
                    if "TODO" in code or "FIXME" in code:
                        score -= 0.05

            score += 0.1  # Bonus for having code

        return DimensionScore(
            dimension=ProtocolQualityDimension.CORRECTNESS,
            score=min(1.0, max(0.0, score)),
            reason="Assessed based on syntax heuristics",
        )

    def _assess_coverage(self, response: str, context: Dict[str, Any]) -> DimensionScore:
        """Assess query coverage."""
        query = context.get("query", "")
        score = 0.5
        covered = 0

        # Extract key terms from query
        key_terms = set(
            word.lower()
            for word in re.findall(r"\b[a-zA-Z]{3,}\b", query)
            if word.lower() not in {"the", "and", "for", "how", "can", "you", "this", "that"}
        )

        if key_terms:
            response_lower = response.lower()
            covered = sum(1 for term in key_terms if term in response_lower)
            coverage_ratio = covered / len(key_terms)
            score = 0.3 + (0.7 * coverage_ratio)

        return DimensionScore(
            dimension=ProtocolQualityDimension.COVERAGE,
            score=score,
            reason=(
                f"Covered {covered}/{len(key_terms)} key terms"
                if key_terms
                else "No query terms to cover"
            ),
            evidence={"key_terms": list(key_terms)},
        )

    def _assess_grounding(self, response: str, context: Dict[str, Any]) -> DimensionScore:
        """Assess factual grounding (placeholder for actual verification)."""
        # This is a placeholder - actual grounding uses IGroundingStrategy
        grounding_result = context.get("grounding_result")

        if grounding_result:
            return DimensionScore(
                dimension=ProtocolQualityDimension.GROUNDING,
                score=grounding_result.get("confidence", 0.5),
                reason=grounding_result.get("reason", "From grounding verifier"),
            )

        return DimensionScore(
            dimension=ProtocolQualityDimension.GROUNDING,
            score=0.7,  # Assume grounded if not verified
            reason="Grounding not verified",
        )


class ProviderAwareQualityAssessor(BaseQualityAssessor):
    """Quality assessor that adapts to provider capabilities.

    Uses provider-specific thresholds and adjustments.
    """

    def __init__(
        self,
        provider_name: str = "",
        provider_threshold: float = 0.80,
        weights: Optional[Dict[ProtocolQualityDimension, float]] = None,
    ):
        """Initialize with provider context.

        Args:
            provider_name: Name of the provider
            provider_threshold: Provider-specific quality threshold
            weights: Dimension weights
        """
        super().__init__(weights=weights, threshold=provider_threshold)
        self._provider_name = provider_name

    def _assess_dimension(
        self,
        dimension: ProtocolQualityDimension,
        response: str,
        context: Dict[str, Any],
    ) -> DimensionScore:
        """Assess with provider-specific adjustments."""
        # Use SimpleQualityAssessor for base assessment
        simple = SimpleQualityAssessor()
        base_score = simple._assess_dimension(dimension, response, context)

        # Apply provider-specific adjustments
        adjustment = self._get_provider_adjustment(dimension)
        adjusted_score = min(1.0, max(0.0, base_score.score + adjustment))

        return DimensionScore(
            dimension=dimension,
            score=adjusted_score,
            weight=base_score.weight,
            reason=f"{base_score.reason} (provider adjustment: {adjustment:+.2f})",
            evidence=base_score.evidence,
        )

    def _get_provider_adjustment(self, dimension: ProtocolQualityDimension) -> float:
        """Get provider-specific score adjustment."""
        # Known provider adjustments
        adjustments = {
            "deepseek": {
                ProtocolQualityDimension.CORRECTNESS: 0.05,  # Good at code
                ProtocolQualityDimension.CONCISENESS: -0.05,  # Can be verbose
            },
            "anthropic": {
                ProtocolQualityDimension.CLARITY: 0.05,
                ProtocolQualityDimension.GROUNDING: 0.05,
            },
            "openai": {
                ProtocolQualityDimension.CLARITY: 0.03,
            },
            "xai": {
                ProtocolQualityDimension.CONCISENESS: -0.10,  # Known for repetition
            },
            "ollama": {
                ProtocolQualityDimension.CORRECTNESS: -0.05,  # Local models vary
                ProtocolQualityDimension.GROUNDING: -0.05,
            },
        }

        provider_adjustments = adjustments.get(self._provider_name.lower(), {})
        return provider_adjustments.get(dimension, 0.0)

    def assess(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> QualityScore:
        """Assess with provider context."""
        result = super().assess(response, context)
        result.provider = self._provider_name
        return result


class CompositeQualityAssessor:
    """Combines multiple quality assessors.

    Uses weighted voting or max/min strategies.
    """

    def __init__(
        self,
        assessors: Optional[List[IQualityAssessor]] = None,
        strategy: str = "weighted",
    ):
        """Initialize composite assessor.

        Args:
            assessors: List of assessors to combine
            strategy: Combination strategy ("weighted", "max", "min", "average")
        """
        self._assessors = assessors or [SimpleQualityAssessor()]
        self._strategy = strategy

    def add_assessor(self, assessor: IQualityAssessor) -> None:
        """Add an assessor to the composite."""
        self._assessors.append(assessor)

    @property
    def dimensions(self) -> List[ProtocolQualityDimension]:
        """Return all dimensions covered by assessors."""
        all_dims = set()
        for assessor in self._assessors:
            all_dims.update(assessor.dimensions)
        return list(all_dims)

    def assess(
        self,
        response: str,
        context: Dict[str, Any],
    ) -> QualityScore:
        """Assess using all assessors and combine results."""
        scores = [a.assess(response, context) for a in self._assessors]

        if not scores:
            return QualityScore(score=0.0, is_acceptable=False)

        if self._strategy == "max":
            overall = max(s.score for s in scores)
        elif self._strategy == "min":
            overall = min(s.score for s in scores)
        else:  # weighted or average
            overall = sum(s.score for s in scores) / len(scores)

        # Combine dimension scores
        combined_dimensions = {}
        for score in scores:
            for dim, ds in score.dimension_scores.items():
                if dim not in combined_dimensions:
                    combined_dimensions[dim] = []
                combined_dimensions[dim].append(ds)

        merged_dimensions = {}
        for dim, scores_list in combined_dimensions.items():
            avg_score = sum(s.score for s in scores_list) / len(scores_list)
            merged_dimensions[dim] = DimensionScore(
                dimension=dim,
                score=avg_score,
                reason=f"Combined from {len(scores_list)} assessors",
            )

        threshold = scores[0].threshold if scores else 0.80

        return QualityScore(
            score=overall,
            is_acceptable=overall >= threshold,
            threshold=threshold,
            dimension_scores=merged_dimensions,
        )
