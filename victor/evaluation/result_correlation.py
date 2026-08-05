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

"""Test result correlation and scoring for SWE-bench evaluation.

This module provides sophisticated test result analysis:
- Correlates test outputs with SWE-bench expectations
- Computes detailed scoring metrics
- Analyzes failure patterns and root causes
- Generates comprehensive evaluation reports
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from victor.evaluation.test_runners import TestResult
from victor.evaluation.baseline_validator import (
    BaselineValidationResult,
    TestStatus,
    get_test_status,
)

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Categories of test failures for analysis."""

    ASSERTION_ERROR = "assertion_error"  # Test assertion failed
    IMPORT_ERROR = "import_error"  # Module import failed
    SYNTAX_ERROR = "syntax_error"  # Code syntax error
    TYPE_ERROR = "type_error"  # Type mismatch
    ATTRIBUTE_ERROR = "attribute_error"  # Missing attribute
    RUNTIME_ERROR = "runtime_error"  # Generic runtime error
    TIMEOUT = "timeout"  # Test timed out
    SETUP_ERROR = "setup_error"  # Test setup failed
    TEARDOWN_ERROR = "teardown_error"  # Test teardown failed
    FIXTURE_ERROR = "fixture_error"  # Pytest fixture error
    ENVIRONMENT_ERROR = "environment_error"  # Environment setup issue
    UNKNOWN = "unknown"  # Unclassified failure


@dataclass
class TestCorrelation:
    """Correlation between expected and actual test result.

    Attributes:
        test_id: Test identifier
        expected_status: Expected test status from SWE-bench
        actual_status: Actual test result
        matches_expectation: Whether result matches SWE-bench expectation
        failure_category: Category of failure if test failed
        error_message: Error message from test
        is_flaky: Whether test appears to be flaky
        context: Additional context about the correlation
    """

    test_id: str
    expected_status: TestStatus
    actual_status: TestStatus
    matches_expectation: bool
    failure_category: Optional[FailureCategory] = None
    error_message: Optional[str] = None
    is_flaky: bool = False
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "test_id": self.test_id,
            "expected_status": self.expected_status.value,
            "actual_status": self.actual_status.value,
            "matches_expectation": self.matches_expectation,
            "failure_category": (self.failure_category.value if self.failure_category else None),
            "error_message": self.error_message,
            "is_flaky": self.is_flaky,
            "context": self.context,
        }


@dataclass
class SWEBenchScore:
    """Detailed scoring for SWE-bench evaluation.

    Attributes:
        instance_id: SWE-bench instance identifier
        resolved: Whether the issue is fully resolved
        partial: Whether partial progress was made
        fail_to_pass_score: Score for FAIL_TO_PASS tests (0-1)
        pass_to_pass_score: Score for PASS_TO_PASS tests (0-1)
        overall_score: Combined overall score (0-1)
        tests_fixed: Number of failing tests now passing
        tests_broken: Number of passing tests now failing
        total_fail_to_pass: Total FAIL_TO_PASS tests
        total_pass_to_pass: Total PASS_TO_PASS tests
        correlations: Individual test correlations
        metadata: Additional scoring metadata
    """

    instance_id: str
    resolved: bool = False
    partial: bool = False
    fail_to_pass_score: float = 0.0
    pass_to_pass_score: float = 1.0  # Default to perfect (no regression)
    overall_score: float = 0.0
    tests_fixed: int = 0
    tests_broken: int = 0
    total_fail_to_pass: int = 0
    total_pass_to_pass: int = 0
    correlations: list[TestCorrelation] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "instance_id": self.instance_id,
            "resolved": self.resolved,
            "partial": self.partial,
            "fail_to_pass_score": self.fail_to_pass_score,
            "pass_to_pass_score": self.pass_to_pass_score,
            "overall_score": self.overall_score,
            "tests_fixed": self.tests_fixed,
            "tests_broken": self.tests_broken,
            "total_fail_to_pass": self.total_fail_to_pass,
            "total_pass_to_pass": self.total_pass_to_pass,
            "correlations": [c.to_dict() for c in self.correlations],
            "metadata": self.metadata,
        }


@dataclass
class CorrelationReport:
    """Comprehensive evaluation report with correlations.

    Attributes:
        timestamp: When report was generated
        total_instances: Total instances evaluated
        resolved_count: Fully resolved instances
        partial_count: Partially resolved instances
        failed_count: Failed instances
        avg_f2p_score: Average FAIL_TO_PASS score
        avg_p2p_score: Average PASS_TO_PASS score
        avg_overall_score: Average overall score
        by_repo: Scores grouped by repository
        by_difficulty: Scores grouped by difficulty
        failure_analysis: Analysis of failure patterns
        scores: Individual instance scores
    """

    timestamp: datetime = field(default_factory=datetime.now)
    total_instances: int = 0
    resolved_count: int = 0
    partial_count: int = 0
    failed_count: int = 0
    avg_f2p_score: float = 0.0
    avg_p2p_score: float = 0.0
    avg_overall_score: float = 0.0
    by_repo: dict[str, dict] = field(default_factory=dict)
    by_difficulty: dict[str, dict] = field(default_factory=dict)
    failure_analysis: dict = field(default_factory=dict)
    scores: list[SWEBenchScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_instances": self.total_instances,
                "resolved": self.resolved_count,
                "partial": self.partial_count,
                "failed": self.failed_count,
                "resolve_rate": (
                    self.resolved_count / self.total_instances if self.total_instances else 0
                ),
            },
            "scores": {
                "avg_f2p": self.avg_f2p_score,
                "avg_p2p": self.avg_p2p_score,
                "avg_overall": self.avg_overall_score,
            },
            "by_repo": self.by_repo,
            "by_difficulty": self.by_difficulty,
            "failure_analysis": self.failure_analysis,
            "instances": [s.to_dict() for s in self.scores],
        }

    def to_text(self) -> str:
        """Generate human-readable text report."""
        lines = [
            "=" * 60,
            "SWE-bench Evaluation Report",
            "=" * 60,
            f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Instances: {self.total_instances}",
            (
                f"Resolved: {self.resolved_count} ({self.resolved_count/self.total_instances*100:.1f}%)"
                if self.total_instances
                else "Resolved: 0"
            ),
            f"Partial: {self.partial_count}",
            f"Failed: {self.failed_count}",
            "",
            "SCORES",
            "-" * 40,
            f"Average FAIL_TO_PASS Score: {self.avg_f2p_score:.3f}",
            f"Average PASS_TO_PASS Score: {self.avg_p2p_score:.3f}",
            f"Average Overall Score: {self.avg_overall_score:.3f}",
            "",
        ]

        if self.by_repo:
            lines.extend(
                [
                    "BY REPOSITORY",
                    "-" * 40,
                ]
            )
            for repo, stats in sorted(self.by_repo.items()):
                lines.append(f"  {repo}:")
                lines.append(f"    Instances: {stats.get('count', 0)}")
                lines.append(f"    Resolved: {stats.get('resolved', 0)}")
                lines.append(f"    Avg Score: {stats.get('avg_score', 0):.3f}")
            lines.append("")

        if self.by_difficulty:
            lines.extend(
                [
                    "BY DIFFICULTY",
                    "-" * 40,
                ]
            )
            for diff, stats in sorted(self.by_difficulty.items()):
                lines.append(f"  {diff}:")
                lines.append(f"    Instances: {stats.get('count', 0)}")
                lines.append(f"    Resolved: {stats.get('resolved', 0)}")
                lines.append(f"    Avg Score: {stats.get('avg_score', 0):.3f}")
            lines.append("")

        if self.failure_analysis:
            lines.extend(
                [
                    "FAILURE ANALYSIS",
                    "-" * 40,
                ]
            )
            for category, count in sorted(
                self.failure_analysis.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                lines.append(f"  {category}: {count}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


class ResultCorrelator:
    """Correlates test results with SWE-bench expectations."""

    # Patterns for detecting failure categories
    FAILURE_PATTERNS = {
        FailureCategory.ASSERTION_ERROR: [
            r"AssertionError",
            r"assert\s+.*\s+==\s+",
            r"Expected.*but got",
        ],
        FailureCategory.IMPORT_ERROR: [
            r"ImportError",
            r"ModuleNotFoundError",
            r"No module named",
        ],
        FailureCategory.SYNTAX_ERROR: [
            r"SyntaxError",
            r"IndentationError",
            r"invalid syntax",
        ],
        FailureCategory.TYPE_ERROR: [
            r"TypeError",
            r"expected.*type",
        ],
        FailureCategory.ATTRIBUTE_ERROR: [
            r"AttributeError",
            r"has no attribute",
        ],
        FailureCategory.TIMEOUT: [
            r"TimeoutError",
            r"timed out",
            r"TIMEOUT",
        ],
        FailureCategory.SETUP_ERROR: [
            r"setup\s+error",
            r"SetupError",
            r"fixture.*setup",
        ],
        FailureCategory.TEARDOWN_ERROR: [
            r"teardown\s+error",
            r"TeardownError",
        ],
        FailureCategory.FIXTURE_ERROR: [
            r"fixture\s+'.*'\s+not found",
            r"FixtureError",
        ],
        FailureCategory.RUNTIME_ERROR: [
            r"RuntimeError",
            r"ValueError",
            r"KeyError",
            r"IndexError",
        ],
        FailureCategory.ENVIRONMENT_ERROR: [
            r"EnvironmentError",
            r"FileNotFoundError",
            r"PermissionError",
            r"OSError",
        ],
    }

    def __init__(self):
        """Initialize result correlator."""
        self._compiled_patterns = {}
        for category, patterns in self.FAILURE_PATTERNS.items():
            self._compiled_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def categorize_failure(self, error_message: str) -> FailureCategory:
        """Categorize a test failure based on error message.

        Args:
            error_message: Error message from test output

        Returns:
            Failure category
        """
        if not error_message:
            return FailureCategory.UNKNOWN

        for category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(error_message):
                    return category

        return FailureCategory.UNKNOWN

    def correlate_test(
        self,
        test_id: str,
        expected_status: TestStatus,
        actual_result: TestResult,
    ) -> TestCorrelation:
        """Correlate a single test result with expectation.

        Args:
            test_id: Test identifier
            expected_status: Expected status (PASSED for P2P, FAILED for F2P)
            actual_result: Actual test result

        Returns:
            TestCorrelation with analysis
        """
        actual_status = get_test_status(actual_result)
        matches = expected_status == actual_status

        failure_category = None
        if actual_status == TestStatus.FAILED or actual_status == TestStatus.ERROR:
            failure_category = self.categorize_failure(
                actual_result.error_message or actual_result.stderr
            )

        return TestCorrelation(
            test_id=test_id,
            expected_status=expected_status,
            actual_status=actual_status,
            matches_expectation=matches,
            failure_category=failure_category,
            error_message=actual_result.error_message,
            context={
                "duration_ms": actual_result.duration_ms,
            },
        )

    def compute_score(
        self,
        validation_result: BaselineValidationResult,
        instance_metadata: Optional[dict] = None,
    ) -> SWEBenchScore:
        """Compute detailed SWE-bench score from validation result.

        Args:
            validation_result: Result from baseline validation
            instance_metadata: Additional metadata (repo, difficulty, etc.)

        Returns:
            Detailed score with correlations
        """
        baseline = validation_result.baseline
        post_results = validation_result.post_change_results

        # Map post-change results by test name
        post_map = {r.test_name: r for r in post_results.results}

        score = SWEBenchScore(
            instance_id=validation_result.instance_id,
            total_fail_to_pass=len(baseline.fail_to_pass),
            total_pass_to_pass=len(baseline.pass_to_pass),
        )

        # Correlate FAIL_TO_PASS tests
        for test_id in baseline.fail_to_pass:
            post_result = post_map.get(test_id)
            if post_result:
                correlation = self.correlate_test(
                    test_id,
                    TestStatus.FAILED,  # Expected at baseline
                    post_result,
                )
                # Check if now passing (fixed)
                if post_result.passed:
                    score.tests_fixed += 1
                    correlation.context["transition"] = "failing_to_passing"
                score.correlations.append(correlation)

        # Correlate PASS_TO_PASS tests
        for test_id in baseline.pass_to_pass:
            post_result = post_map.get(test_id)
            if post_result:
                correlation = self.correlate_test(
                    test_id,
                    TestStatus.PASSED,  # Expected at baseline
                    post_result,
                )
                # Check if now failing (broken)
                if not post_result.passed:
                    score.tests_broken += 1
                    correlation.context["transition"] = "passing_to_failing"
                score.correlations.append(correlation)

        # Compute scores
        if score.total_fail_to_pass > 0:
            score.fail_to_pass_score = score.tests_fixed / score.total_fail_to_pass
        else:
            score.fail_to_pass_score = 1.0  # No tests to fix

        if score.total_pass_to_pass > 0:
            passing = score.total_pass_to_pass - score.tests_broken
            score.pass_to_pass_score = passing / score.total_pass_to_pass
        else:
            score.pass_to_pass_score = 1.0  # No tests to maintain

        # Overall score: F2P success minus P2P regression penalty
        score.overall_score = max(
            0.0,
            score.fail_to_pass_score - (1.0 - score.pass_to_pass_score),
        )

        # Determine resolution status
        score.resolved = score.tests_fixed == score.total_fail_to_pass and score.tests_broken == 0
        score.partial = score.tests_fixed > 0 and not score.resolved

        # Add metadata
        if instance_metadata:
            score.metadata = instance_metadata

        return score

    def generate_report(
        self,
        scores: list[SWEBenchScore],
        metadata_provider: Optional[callable] = None,
    ) -> CorrelationReport:
        """Generate comprehensive evaluation report.

        Args:
            scores: List of individual instance scores
            metadata_provider: Optional function to get metadata by instance_id

        Returns:
            CorrelationReport with aggregated analysis
        """
        report = CorrelationReport(scores=scores)
        report.total_instances = len(scores)

        if not scores:
            return report

        # Aggregate scores
        total_f2p = 0.0
        total_p2p = 0.0
        total_overall = 0.0

        repo_stats: dict[str, dict] = {}
        difficulty_stats: dict[str, dict] = {}
        failure_counts: dict[str, int] = {}

        for score in scores:
            # Count resolution status
            if score.resolved:
                report.resolved_count += 1
            elif score.partial:
                report.partial_count += 1
            else:
                report.failed_count += 1

            # Accumulate scores
            total_f2p += score.fail_to_pass_score
            total_p2p += score.pass_to_pass_score
            total_overall += score.overall_score

            # Get metadata
            metadata = score.metadata or {}
            if metadata_provider:
                metadata = metadata_provider(score.instance_id)

            # Group by repo
            repo = metadata.get("repo", "unknown")
            if repo not in repo_stats:
                repo_stats[repo] = {
                    "count": 0,
                    "resolved": 0,
                    "total_score": 0.0,
                }
            repo_stats[repo]["count"] += 1
            if score.resolved:
                repo_stats[repo]["resolved"] += 1
            repo_stats[repo]["total_score"] += score.overall_score

            # Group by difficulty
            difficulty = metadata.get("difficulty", "unknown")
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {
                    "count": 0,
                    "resolved": 0,
                    "total_score": 0.0,
                }
            difficulty_stats[difficulty]["count"] += 1
            if score.resolved:
                difficulty_stats[difficulty]["resolved"] += 1
            difficulty_stats[difficulty]["total_score"] += score.overall_score

            # Analyze failures
            for correlation in score.correlations:
                if correlation.failure_category:
                    cat = correlation.failure_category.value
                    failure_counts[cat] = failure_counts.get(cat, 0) + 1

        # Compute averages
        n = len(scores)
        report.avg_f2p_score = total_f2p / n
        report.avg_p2p_score = total_p2p / n
        report.avg_overall_score = total_overall / n

        # Finalize repo stats
        for _repo, stats in repo_stats.items():
            stats["avg_score"] = stats["total_score"] / stats["count"]
            del stats["total_score"]
        report.by_repo = repo_stats

        # Finalize difficulty stats
        for _diff, stats in difficulty_stats.items():
            stats["avg_score"] = stats["total_score"] / stats["count"]
            del stats["total_score"]
        report.by_difficulty = difficulty_stats

        report.failure_analysis = failure_counts

        return report


def correlate_validation_results(
    results: list[BaselineValidationResult],
    metadata: Optional[dict[str, dict]] = None,
) -> CorrelationReport:
    """Correlate multiple validation results into a report.

    Args:
        results: List of validation results
        metadata: Optional metadata keyed by instance_id

    Returns:
        CorrelationReport with full analysis
    """
    correlator = ResultCorrelator()

    scores = []
    for result in results:
        instance_meta = metadata.get(result.instance_id, {}) if metadata else {}
        score = correlator.compute_score(result, instance_meta)
        scores.append(score)

    return correlator.generate_report(scores)


def analyze_failure_patterns(
    scores: list[SWEBenchScore],
) -> dict[str, Any]:
    """Analyze failure patterns across multiple instances.

    Args:
        scores: List of SWE-bench scores

    Returns:
        Analysis of failure patterns
    """
    patterns = {
        "by_category": {},
        "by_test_pattern": {},
        "common_errors": [],
        "flaky_tests": [],
    }

    category_counts: dict[str, int] = {}
    test_pattern_counts: dict[str, int] = {}
    error_messages: list[str] = []

    for score in scores:
        for correlation in score.correlations:
            # Count by category
            if correlation.failure_category:
                cat = correlation.failure_category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1

            # Extract test pattern (e.g., "test_" prefix, module name)
            test_id = correlation.test_id
            if "::" in test_id:
                module = test_id.split("::")[0]
                test_pattern_counts[module] = test_pattern_counts.get(module, 0) + 1

            # Collect error messages
            if correlation.error_message:
                error_messages.append(correlation.error_message[:200])

            # Track flaky tests
            if correlation.is_flaky:
                patterns["flaky_tests"].append(correlation.test_id)

    patterns["by_category"] = category_counts
    patterns["by_test_pattern"] = dict(
        sorted(test_pattern_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    )

    # Find common error patterns
    error_patterns: dict[str, int] = {}
    for msg in error_messages:
        # Extract key error phrase
        for pattern in ["Error:", "Exception:", "Failed:"]:
            if pattern in msg:
                key = msg[msg.index(pattern) : msg.index(pattern) + 50]
                error_patterns[key] = error_patterns.get(key, 0) + 1

    patterns["common_errors"] = sorted(
        [{"pattern": k, "count": v} for k, v in error_patterns.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:10]

    return patterns


def save_correlation_report(
    report: CorrelationReport,
    output_path: Path,
    include_text: bool = True,
) -> None:
    """Save correlation report to files.

    Args:
        report: Report to save
        output_path: Base path for output files
        include_text: Also save text version
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    logger.info(f"Saved JSON report to {json_path}")

    # Save text
    if include_text:
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write(report.to_text())
        logger.info(f"Saved text report to {txt_path}")
