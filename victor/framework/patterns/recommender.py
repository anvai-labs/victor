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

"""Pattern recommendation implementation.

This module provides the PatternRecommender class that suggests
optimal collaboration patterns for specific tasks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor.framework.patterns.types import (
    CollaborationPattern,
    PatternCategory,
    PatternRecommendation,
    PatternValidationResult,
    TaskContext,
)

logger = logging.getLogger(__name__)


class PatternRecommender:
    """Recommends collaboration patterns for tasks.

    Implements PatternRecommenderProtocol for pattern recommendations.

    Example:
        recommender = PatternRecommender(patterns)

        context = TaskContext(
            task_description="Implement new authentication feature",
            required_capabilities=["coding", "testing", "security"],
            vertical="coding",
            complexity="high",
        )

        recommendations = await recommender.recommend(context, top_k=3)
        for rec in recommendations:
            print(f"{rec.pattern.name}: {rec.score:.2f} - {rec.rationale}")
    """

    def __init__(
        self,
        patterns: Optional[List[CollaborationPattern]] = None,
        complexity_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize pattern recommender.

        Args:
            patterns: Available patterns for recommendation
            complexity_weights: Weights for complexity scoring
        """
        self._patterns = patterns or []
        self._complexity_weights = complexity_weights or {
            "low": 0.3,
            "medium": 0.6,
            "high": 1.0,
        }

    def add_pattern(self, pattern: CollaborationPattern) -> None:
        """Add a pattern to the catalog.

        Args:
            pattern: Pattern to add
        """
        self._patterns.append(pattern)

    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove a pattern from the catalog.

        Args:
            pattern_id: ID of pattern to remove

        Returns:
            True if removed, False if not found
        """
        for i, p in enumerate(self._patterns):
            if p.id == pattern_id:
                del self._patterns[i]
                return True
        return False

    async def recommend(
        self,
        task_context: TaskContext,
        top_k: int = 5,
    ) -> List[PatternRecommendation]:
        """Recommend patterns for task context.

        Args:
            task_context: Task to find pattern for
            top_k: Number of recommendations to return

        Returns:
            List of pattern recommendations, sorted by score
        """
        if not self._patterns:
            logger.warning("No patterns available for recommendation")
            return []

        # Score all patterns
        scored = await self.rank_patterns(self._patterns, task_context)

        # Return top-k
        return scored[:top_k]

    async def rank_patterns(
        self,
        patterns: List[CollaborationPattern],
        task_context: TaskContext,
    ) -> List[PatternRecommendation]:
        """Rank patterns by suitability for task.

        Args:
            patterns: List of patterns to rank
            task_context: Task context

        Returns:
            List of pattern recommendations, ranked by score
        """
        recommendations = []

        for pattern in patterns:
            score = self._calculate_score(pattern, task_context)
            rationale = self._generate_rationale(pattern, task_context, score)
            benefits = self._identify_benefits(pattern, task_context)
            risks = self._identify_risks(pattern, task_context)

            recommendations.append(
                PatternRecommendation(
                    pattern=pattern,
                    score=score,
                    rationale=rationale,
                    expected_benefits=benefits,
                    potential_risks=risks,
                    confidence=self._calculate_confidence(pattern),
                )
            )

        # Sort by score descending
        recommendations.sort(key=lambda r: r.score, reverse=True)
        return recommendations

    async def explain_recommendation(
        self,
        recommendation: PatternRecommendation,
    ) -> str:
        """Explain why pattern was recommended.

        Args:
            recommendation: Pattern recommendation to explain

        Returns:
            Human-readable explanation
        """
        pattern = recommendation.pattern
        lines = [
            f"## Pattern: {pattern.name}",
            "",
            f"**Score:** {recommendation.score:.2f}/1.00",
            f"**Confidence:** {recommendation.confidence:.1%}",
            "",
            "### Rationale",
            recommendation.rationale,
            "",
            "### Expected Benefits",
        ]

        for benefit in recommendation.expected_benefits:
            lines.append(f"- {benefit}")

        lines.extend(
            [
                "",
                "### Potential Risks",
            ]
        )

        for risk in recommendation.potential_risks:
            lines.append(f"- {risk}")

        lines.extend(
            [
                "",
                "### Pattern Statistics",
                f"- Success Rate: {pattern.success_rate:.1%}",
                f"- Usage Count: {pattern.metrics.usage_count}",
                f"- Avg Duration: {pattern.metrics.avg_duration_ms:.0f}ms",
            ]
        )

        return "\n".join(lines)

    def _calculate_score(
        self,
        pattern: CollaborationPattern,
        context: TaskContext,
    ) -> float:
        """Calculate pattern suitability score.

        Args:
            pattern: Pattern to score
            context: Task context

        Returns:
            Score (0-1)
        """
        # Base score from success rate
        score = pattern.success_rate * 0.4

        # Complexity match
        complexity_match = self._match_complexity(pattern, context)
        score += complexity_match * 0.25

        # Capability match
        capability_match = self._match_capabilities(pattern, context)
        score += capability_match * 0.25

        # Recency bonus (patterns used recently are slightly preferred)
        recency_score = self._calculate_recency_score(pattern)
        score += recency_score * 0.1

        return min(score, 1.0)

    def _match_complexity(
        self,
        pattern: CollaborationPattern,
        context: TaskContext,
    ) -> float:
        """Match pattern to task complexity.

        Args:
            pattern: Pattern to match
            context: Task context

        Returns:
            Match score (0-1)
        """
        # Match team size to complexity (weights used for future enhancements)
        _ = self._complexity_weights.get(context.complexity, 0.5)  # Reserved
        team_size = len(pattern.participants)

        if context.complexity == "high" and team_size >= 3:
            return 1.0
        elif context.complexity == "medium" and 2 <= team_size <= 4:
            return 1.0
        elif context.complexity == "low" and team_size <= 2:
            return 1.0

        # Partial match
        return 0.5

    def _match_capabilities(
        self,
        pattern: CollaborationPattern,
        context: TaskContext,
    ) -> float:
        """Match pattern capabilities to task requirements.

        Args:
            pattern: Pattern to match
            context: Task context

        Returns:
            Match score (0-1)
        """
        if not context.required_capabilities:
            return 0.5

        # Collect pattern capabilities
        pattern_capabilities = set()
        for p in pattern.participants:
            caps = p.get("capabilities", [])
            pattern_capabilities.update(caps)
            # Also consider role as capability
            pattern_capabilities.add(p.get("role", ""))

        required = set(context.required_capabilities)

        if not required:
            return 0.5

        # Calculate overlap
        overlap = len(required.intersection(pattern_capabilities))
        return overlap / len(required)

    def _calculate_recency_score(self, pattern: CollaborationPattern) -> float:
        """Calculate recency score for pattern.

        Args:
            pattern: Pattern to score

        Returns:
            Recency score (0-1)
        """
        from datetime import datetime, timedelta

        last_used = pattern.metrics.last_used
        if not last_used:
            return 0.0

        now = datetime.now()
        age = now - last_used

        # Full score if used in last day
        if age < timedelta(days=1):
            return 1.0
        # Linear decay over 30 days
        if age < timedelta(days=30):
            return 1.0 - (age.days / 30.0)
        return 0.0

    def _calculate_confidence(self, pattern: CollaborationPattern) -> float:
        """Calculate confidence in pattern based on usage.

        Args:
            pattern: Pattern to evaluate

        Returns:
            Confidence score (0-1)
        """
        usage = pattern.metrics.usage_count

        # More usage = higher confidence
        if usage >= 100:
            return 0.95
        elif usage >= 50:
            return 0.85
        elif usage >= 20:
            return 0.75
        elif usage >= 10:
            return 0.65
        elif usage >= 5:
            return 0.55
        return 0.4

    def _generate_rationale(
        self,
        pattern: CollaborationPattern,
        context: TaskContext,
        score: float,
    ) -> str:
        """Generate explanation for recommendation.

        Args:
            pattern: Recommended pattern
            context: Task context
            score: Calculated score

        Returns:
            Rationale string
        """
        reasons = []

        # Success rate
        if pattern.success_rate >= 0.8:
            reasons.append(f"High success rate ({pattern.success_rate:.0%})")
        elif pattern.success_rate >= 0.6:
            reasons.append(f"Good success rate ({pattern.success_rate:.0%})")

        # Team size match
        team_size = len(pattern.participants)
        if context.complexity == "high" and team_size >= 3:
            reasons.append("Team size appropriate for high-complexity task")
        elif context.complexity == "low" and team_size <= 2:
            reasons.append("Efficient team size for straightforward task")

        # Category match
        if pattern.category == PatternCategory.PARALLEL:
            reasons.append("Parallel execution can improve throughput")
        elif pattern.category == PatternCategory.HIERARCHICAL:
            reasons.append("Hierarchical structure provides clear coordination")
        elif pattern.category == PatternCategory.COLLABORATIVE:
            reasons.append("Collaborative approach supports complex decisions")

        # Usage
        if pattern.metrics.usage_count >= 10:
            reasons.append(f"Proven pattern with {pattern.metrics.usage_count} prior uses")

        if not reasons:
            reasons.append("General-purpose pattern with reasonable fit")

        return " ".join(reasons) + "."

    def _identify_benefits(
        self,
        pattern: CollaborationPattern,
        context: TaskContext,
    ) -> List[str]:
        """Identify expected benefits of using pattern.

        Args:
            pattern: Pattern to evaluate
            context: Task context

        Returns:
            List of expected benefits
        """
        benefits = []

        if pattern.success_rate >= 0.7:
            benefits.append("High likelihood of successful completion")

        if pattern.category == PatternCategory.PARALLEL:
            benefits.append("Faster execution through parallelization")

        if pattern.category == PatternCategory.HIERARCHICAL:
            benefits.append("Clear accountability and coordination")

        if pattern.metrics.avg_duration_ms < 60000:  # Under 1 minute
            benefits.append("Fast average execution time")

        if len(pattern.participants) >= 2:
            benefits.append("Multiple perspectives on problem solving")

        return benefits or ["Structured collaboration approach"]

    def _identify_risks(
        self,
        pattern: CollaborationPattern,
        context: TaskContext,
    ) -> List[str]:
        """Identify potential risks of using pattern.

        Args:
            pattern: Pattern to evaluate
            context: Task context

        Returns:
            List of potential risks
        """
        risks = []

        if pattern.success_rate < 0.6:
            risks.append(f"Below-average success rate ({pattern.success_rate:.0%})")

        if pattern.metrics.usage_count < 5:
            risks.append("Limited usage history - less proven")

        if len(pattern.participants) >= 5:
            risks.append("Large team may have coordination overhead")

        if pattern.category == PatternCategory.COMPETITIVE:
            risks.append("Competing approaches may waste resources")

        if context.complexity == "high" and len(pattern.participants) < 2:
            risks.append("Small team may struggle with high complexity")

        return risks or ["No significant risks identified"]


class PatternValidator:
    """Validates collaboration patterns.

    Implements PatternValidatorProtocol for pattern validation.
    """

    def __init__(
        self,
        min_participants: int = 1,
        max_participants: int = 10,
        require_description: bool = True,
    ):
        """Initialize validator.

        Args:
            min_participants: Minimum required participants
            max_participants: Maximum allowed participants
            require_description: Whether description is required
        """
        self.min_participants = min_participants
        self.max_participants = max_participants
        self.require_description = require_description

    async def validate(
        self,
        pattern: CollaborationPattern,
        test_cases: Optional[List[TaskContext]] = None,
    ) -> PatternValidationResult:
        """Validate pattern structure and correctness.

        Args:
            pattern: Pattern to validate
            test_cases: Optional test cases for validation

        Returns:
            PatternValidationResult with is_valid flag and any errors
        """
        errors = []
        warnings = []

        # Check name
        if not pattern.name:
            errors.append("Pattern must have a name")

        # Check description
        if self.require_description and not pattern.description:
            warnings.append("Pattern has no description")

        # Check participants
        num_participants = len(pattern.participants)
        if num_participants < self.min_participants:
            errors.append(f"Pattern must have at least {self.min_participants} participant(s)")
        if num_participants > self.max_participants:
            errors.append(f"Pattern cannot have more than {self.max_participants} participants")

        # Check participant specs
        for i, p in enumerate(pattern.participants):
            if not p.get("id") and not p.get("role"):
                warnings.append(f"Participant {i} has no id or role")

        # Quality score based on success rate and usage
        quality_score = self._calculate_quality_score(pattern)

        # Safety score (1.0 for now - would check for harmful patterns)
        safety_score = 1.0

        return PatternValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score,
            safety_score=safety_score,
        )

    async def estimate_success(
        self,
        pattern: CollaborationPattern,
        task_context: TaskContext,
    ) -> float:
        """Estimate success probability for pattern.

        Args:
            pattern: Pattern to estimate
            task_context: Task context

        Returns:
            Success probability (0-1)
        """
        # Start with historical success rate
        base_rate = pattern.success_rate

        # Adjust for confidence based on usage
        confidence = min(pattern.metrics.usage_count / 20.0, 1.0)

        # Estimate: weighted average of historical and prior
        prior = 0.5  # Assume 50% prior
        estimated = (base_rate * confidence) + (prior * (1 - confidence))

        return estimated

    def _calculate_quality_score(self, pattern: CollaborationPattern) -> float:
        """Calculate quality score for pattern.

        Args:
            pattern: Pattern to score

        Returns:
            Quality score (0-1)
        """
        score = 0.0

        # Success rate contribution
        score += pattern.success_rate * 0.4

        # Usage count contribution (more usage = higher quality signal)
        usage_score = min(pattern.metrics.usage_count / 50.0, 1.0)
        score += usage_score * 0.3

        # Documentation quality
        if pattern.description:
            score += 0.15
        if pattern.workflow:
            score += 0.15

        return min(score, 1.0)


__all__ = [
    "PatternRecommender",
    "PatternValidator",
]
