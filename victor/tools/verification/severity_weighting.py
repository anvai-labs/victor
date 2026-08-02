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

"""Severity weighting for codebase analysis issues.

This module calculates severity scores for issues based on their actual
impact on the codebase. Not all issues are equally severe - a cross-layer
dependency in core code is more critical than a style issue in tests.

P4 Priority: Medium impact, medium effort. Enables better prioritization.

Design Patterns:
- Weighted Scoring: Multiple factors contribute to severity
- Category-Based Impact: Different issue types have different impacts
- Configurable Weights: Weights can be customized per project

Usage:
    weighting = SeverityWeighting()
    score = weighting.calculate_severity_score({
        "issue_type": "cross_layer_dependency",
        "file_path": "src/core/lib.rs",
        "category": "architectural",
    })
    severity = weighting.classify_severity(score)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from victor.tools.verification.protocols import (
    ClaimIssue,
    IssueCategory,
    SeverityLevel,
)

logger = logging.getLogger(__name__)


class ImpactFactor(str, Enum):
    """Impact factors for severity calculation."""

    COMPILATION_TIME = "compilation_time"
    RUNTIME_PERFORMANCE = "runtime_performance"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    TEST_RELIABILITY = "test_reliability"
    API_STABILITY = "api_stability"
    CODE_SIZE = "code_size"
    DEPENDENCY_COMPLEXITY = "dependency_complexity"


@dataclass
class SeverityWeights:
    """Configurable weights for impact factors.

    Attributes:
        compilation_time: Impact on compilation time
        runtime_performance: Impact on runtime performance
        maintainability: Impact on code maintainability
        security: Security impact (highest weight)
        test_reliability: Impact on test reliability
        api_stability: Impact on API stability
        code_size: Impact on code size/bloat
        dependency_complexity: Impact on dependency complexity
    """

    compilation_time: float = 0.3
    runtime_performance: float = 0.4
    maintainability: float = 0.2
    security: float = 0.5
    test_reliability: float = 0.15
    api_stability: float = 0.35
    code_size: float = 0.1
    dependency_complexity: float = 0.25

    def get_weight(self, factor: ImpactFactor) -> float:
        """Get weight for an impact factor.

        Args:
            factor: Impact factor to get weight for

        Returns:
            Weight value
        """
        return getattr(self, factor.value, 0.2)


# Default severity weights
DEFAULT_WEIGHTS = SeverityWeights()

# Security-focused weights (higher security priority)
SECURITY_FOCUSED_WEIGHTS = SeverityWeights(
    compilation_time=0.1,
    runtime_performance=0.3,
    maintainability=0.2,
    security=0.8,  # Highest priority
    test_reliability=0.15,
    api_stability=0.3,
    code_size=0.05,
    dependency_complexity=0.2,
)

# Performance-focused weights
PERFORMANCE_FOCUSED_WEIGHTS = SeverityWeights(
    compilation_time=0.5,
    runtime_performance=0.6,
    maintainability=0.15,
    security=0.3,
    test_reliability=0.1,
    api_stability=0.2,
    code_size=0.2,
    dependency_complexity=0.3,
)


class IssueImpactClassifier:
    """Classifies issues by their impact factors."""

    # Patterns for different impact factors
    COMPILATION_TIME_PATTERNS = [
        r"compile.*time",
        r"build.*time",
        r"compilation.*overhead",
        r"header.*include",
        r"macro.*expansion",
        r"template.*instantiation",
    ]

    RUNTIME_PERFORMANCE_PATTERNS = [
        r"performance",
        r"slow",
        r"inefficient",
        r"o\(n\^2\)",
        r"quadratic",
        r"allocation",
        r"copy",
        r"clone",
    ]

    SECURITY_PATTERNS = [
        r"security",
        r"vulnerability",
        r"unsafe",
        r"buffer.*overflow",
        r"memory.*leak",
        r"injection",
        r"xss",
        r"csrf",
        r"auth",
    ]

    MAINTAINABILITY_PATTERNS = [
        r"complex",
        r"confus(ing|ed)",
        r"unclear",
        r"magic.*number",
        r"duplicat",
        r"coupl(ing|ed)",
        r"cohesion",
        r"modular",
    ]

    API_STABILITY_PATTERNS = [
        r"public.*api",
        r"interface",
        r"breaking.*change",
        r"deprecated",
        r"semver",
        r"backward.*compatib",
    ]

    def __init__(self):
        """Compile regex patterns for efficiency."""
        self._patterns = {
            ImpactFactor.COMPILATION_TIME: [
                re.compile(p, re.IGNORECASE) for p in self.COMPILATION_TIME_PATTERNS
            ],
            ImpactFactor.RUNTIME_PERFORMANCE: [
                re.compile(p, re.IGNORECASE) for p in self.RUNTIME_PERFORMANCE_PATTERNS
            ],
            ImpactFactor.SECURITY: [re.compile(p, re.IGNORECASE) for p in self.SECURITY_PATTERNS],
            ImpactFactor.MAINTAINABILITY: [
                re.compile(p, re.IGNORECASE) for p in self.MAINTAINABILITY_PATTERNS
            ],
            ImpactFactor.API_STABILITY: [
                re.compile(p, re.IGNORECASE) for p in self.API_STABILITY_PATTERNS
            ],
        }

    def classify_impacts(
        self,
        issue: Dict[str, Any] | ClaimIssue,
    ) -> Set[ImpactFactor]:
        """Classify which impact factors an issue affects.

        Args:
            issue: Issue to classify

        Returns:
            Set of impact factors
        """
        if isinstance(issue, dict):
            issue_dict = issue
        else:
            issue_dict = issue.model_dump()

        impacts = set()

        # Get text to analyze
        text = " ".join(
            [
                issue_dict.get("issue_type") or "",
                issue_dict.get("description") or "",
                issue_dict.get("snippet") or "",
            ]
        ).lower()

        # Check each impact factor's patterns
        for factor, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    impacts.add(factor)
                    break

        # Add default impacts based on issue type
        issue_type = issue_dict.get("issue_type", "").lower()

        if "dependency" in issue_type or "import" in issue_type:
            impacts.add(ImpactFactor.COMPILATION_TIME)
            impacts.add(ImpactFactor.DEPENDENCY_COMPLEXITY)

        if "global" in issue_type or "static" in issue_type:
            impacts.add(ImpactFactor.MAINTAINABILITY)

        if "cross.*layer" in issue_type or "coupl" in issue_type:
            impacts.add(ImpactFactor.MAINTAINABILITY)
            impacts.add(ImpactFactor.DEPENDENCY_COMPLEXITY)

        return impacts


class SeverityWeighting:
    """Calculates severity scores for codebase issues.

    Uses configurable weights for different impact factors to
    calculate an overall severity score.
    """

    # Severity thresholds
    CRITICAL_THRESHOLD = 0.8
    HIGH_THRESHOLD = 0.6
    MEDIUM_THRESHOLD = 0.4
    LOW_THRESHOLD = 0.2

    def __init__(
        self,
        weights: Optional[SeverityWeights] = None,
        profile: str = "default",
    ):
        """Initialize severity weighting.

        Args:
            weights: Custom severity weights
            profile: Predefined weight profile ("default", "security", "performance")
        """
        if weights:
            self._weights = weights
        else:
            self._weights = self._get_profile_weights(profile)

        self._classifier = IssueImpactClassifier()

    def _get_profile_weights(self, profile: str) -> SeverityWeights:
        """Get weights for a predefined profile.

        Args:
            profile: Profile name

        Returns:
            SeverityWeights for the profile
        """
        profiles = {
            "default": DEFAULT_WEIGHTS,
            "security": SECURITY_FOCUSED_WEIGHTS,
            "performance": PERFORMANCE_FOCUSED_WEIGHTS,
        }
        return profiles.get(profile, DEFAULT_WEIGHTS)

    def calculate_severity_score(
        self,
        issue: Dict[str, Any] | ClaimIssue,
    ) -> float:
        """Calculate weighted severity score for an issue.

        Args:
            issue: Issue to score

        Returns:
            Severity score between 0.0 and 1.0
        """
        if isinstance(issue, dict):
            issue_dict = issue
        else:
            issue_dict = issue.model_dump()

        # Classify impacts
        impacts = self._classifier.classify_impacts(issue)

        if not impacts:
            return 0.3  # Default to low-medium for unknown issues

        # Calculate weighted score
        total_score = 0.0
        max_possible_score = 0.0

        for impact in impacts:
            weight = self._weights.get_weight(impact)
            impact_score = self._calculate_impact_score(impact, issue_dict)
            total_score += weight * impact_score
            max_possible_score += weight

        if max_possible_score > 0:
            return min(total_score / max_possible_score, 1.0)
        return 0.3

    def _calculate_impact_score(
        self,
        impact: ImpactFactor,
        issue: Dict[str, Any],
    ) -> float:
        """Calculate score for a specific impact factor.

        Args:
            impact: Impact factor to score
            issue: Issue dictionary

        Returns:
            Impact score between 0.0 and 1.0
        """
        base_score = 0.5

        # Adjust based on file location
        file_path = issue.get("file_path") or ""

        # Core/src files are more impactful
        if file_path and any(
            core_dir in file_path for core_dir in ["/src/", "/core/", "/lib/", "/kernel/"]
        ):
            base_score += 0.2

        # Test files are less impactful
        if file_path and any(
            test_dir in file_path.lower() for test_dir in ["/test/", "/tests/", "/spec/", "/mock/"]
        ):
            base_score -= 0.3

        # Examples/docs are least impactful
        if file_path and any(
            doc_dir in file_path.lower()
            for doc_dir in ["/example", "/doc/", "/docs/", "/tutorial/"]
        ):
            base_score -= 0.4

        # Adjust based on issue category
        category = issue.get("category")

        if category == IssueCategory.SECURITY:
            if impact == ImpactFactor.SECURITY:
                return 1.0

        elif category == IssueCategory.ARCHITECTURAL:
            if impact in {
                ImpactFactor.MAINTAINABILITY,
                ImpactFactor.DEPENDENCY_COMPLEXITY,
            }:
                base_score += 0.2

        elif category == IssueCategory.TESTING:
            if impact == ImpactFactor.TEST_RELIABILITY:
                base_score += 0.2
            else:
                base_score -= 0.2

        return max(0.0, min(base_score, 1.0))

    def classify_severity(self, score: float) -> SeverityLevel:
        """Map severity score to severity level.

        Args:
            score: Severity score (0.0-1.0)

        Returns:
            SeverityLevel enum value
        """
        if score >= self.CRITICAL_THRESHOLD:
            return SeverityLevel.CRITICAL
        elif score >= self.HIGH_THRESHOLD:
            return SeverityLevel.HIGH
        elif score >= self.MEDIUM_THRESHOLD:
            return SeverityLevel.MEDIUM
        elif score >= self.LOW_THRESHOLD:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.INFO

    def score_and_classify(
        self,
        issue: Dict[str, Any] | ClaimIssue,
    ) -> tuple[float, SeverityLevel]:
        """Calculate score and classify severity in one call.

        Args:
            issue: Issue to score

        Returns:
            Tuple of (score, severity_level)
        """
        score = self.calculate_severity_score(issue)
        severity = self.classify_severity(score)
        return score, severity

    def get_impact_factors(
        self,
        issue: Dict[str, Any] | ClaimIssue,
    ) -> Set[ImpactFactor]:
        """Get impact factors for an issue.

        Args:
            issue: Issue to analyze

        Returns:
            Set of impact factors
        """
        return self._classifier.classify_impacts(issue)

    def batch_score_issues(
        self,
        issues: List[Dict[str, Any] | ClaimIssue],
    ) -> List[Dict[str, Any]]:
        """Score multiple issues efficiently.

        Args:
            issues: List of issues to score

        Returns:
            List of issues with added severity information
        """
        results = []
        for issue in issues:
            score, severity = self.score_and_classify(issue)
            impacts = self.get_impact_factors(issue)

            issue_dict = issue if isinstance(issue, dict) else issue.model_dump()
            issue_dict_with_severity = {
                **issue_dict,
                "severity_score": score,
                "severity_level": severity.value,
                "impact_factors": [i.value for i in impacts],
            }
            results.append(issue_dict_with_severity)

        return results

    def get_severity_distribution(
        self,
        issues: List[Dict[str, Any] | ClaimIssue],
    ) -> Dict[str, int]:
        """Get count of issues by severity level.

        Args:
            issues: List of issues

        Returns:
            Dictionary mapping severity to count
        """
        distribution = {level.value: 0 for level in SeverityLevel}

        for issue in issues:
            _, severity = self.score_and_classify(issue)
            distribution[severity.value] += 1

        return distribution
