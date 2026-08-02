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

"""Tests for SeverityWeighting module."""

import pytest

from victor.tools.verification import (
    ClaimIssue,
    IssueCategory,
    SeverityLevel,
    SeverityWeighting,
)


class TestSeverityWeighting:
    """Tests for SeverityWeighting class."""

    @pytest.fixture
    def weighting(self) -> SeverityWeighting:
        """Create a SeverityWeighting instance."""
        return SeverityWeighting()

    def test_calculate_severity_score_security_issue(self, weighting: SeverityWeighting):
        """Test severity scoring for security issues."""
        issue = ClaimIssue(
            issue_type="security_vulnerability",
            description="Buffer overflow in core module",
            file_path="src/core/lib.rs",
            category=IssueCategory.SECURITY,
        )

        score = weighting.calculate_severity_score(issue)

        # Security issues should have high severity
        assert score >= 0.6

    def test_calculate_severity_score_test_issue(self, weighting: SeverityWeighting):
        """Test severity scoring for test issues."""
        issue = ClaimIssue(
            issue_type="global_mutable_state",
            description="Global state in test",
            file_path="tests/test_utils.rs",
            category=IssueCategory.TESTING,
        )

        score = weighting.calculate_severity_score(issue)

        # Test issues should have lower severity
        assert score < 0.7

    def test_calculate_severity_score_architectural_issue(self, weighting: SeverityWeighting):
        """Test severity scoring for architectural issues."""
        issue = ClaimIssue(
            issue_type="cross_layer_dependency",
            description="Storage depends on Index",
            file_path="src/storage/lib.rs",
            category=IssueCategory.ARCHITECTURAL,
        )

        score = weighting.calculate_severity_score(issue)

        assert 0.0 <= score <= 1.0

    def test_classify_severity(self, weighting: SeverityWeighting):
        """Test severity classification."""
        assert weighting.classify_severity(0.9) == SeverityLevel.CRITICAL
        assert weighting.classify_severity(0.7) == SeverityLevel.HIGH
        assert weighting.classify_severity(0.5) == SeverityLevel.MEDIUM
        assert weighting.classify_severity(0.3) == SeverityLevel.LOW
        assert weighting.classify_severity(0.1) == SeverityLevel.INFO

    def test_score_and_classify(self, weighting: SeverityWeighting):
        """Test combined scoring and classification."""
        issue = ClaimIssue(
            issue_type="security_vulnerability",
            description="Security issue",
            file_path="src/core/lib.rs",
            category=IssueCategory.SECURITY,
        )

        score, severity = weighting.score_and_classify(issue)

        assert 0.0 <= score <= 1.0
        assert isinstance(severity, SeverityLevel)

    def test_get_impact_factors(self, weighting: SeverityWeighting):
        """Test getting impact factors for an issue."""
        issue = ClaimIssue(
            issue_type="performance_issue",
            description="Slow algorithm causing delays",
        )

        factors = weighting.get_impact_factors(issue)

        assert len(factors) > 0  # Should have some impact factors

    def test_batch_score_issues(self, weighting: SeverityWeighting):
        """Test batch scoring of multiple issues."""
        issues = [
            ClaimIssue(
                issue_type="security",
                description="Security issue",
                category=IssueCategory.SECURITY,
            ),
            ClaimIssue(issue_type="test", description="Test issue", file_path="tests/test.rs"),
        ]

        results = weighting.batch_score_issues(issues)

        assert len(results) == 2
        assert all("severity_score" in r for r in results)
        assert all("severity_level" in r for r in results)

    def test_get_severity_distribution(self, weighting: SeverityWeighting):
        """Test getting severity distribution."""
        issues = [
            ClaimIssue(issue_type="x", description="X", category=IssueCategory.SECURITY),
            ClaimIssue(issue_type="y", description="Y", file_path="tests/test.rs"),
            ClaimIssue(issue_type="z", description="Z"),
        ]

        distribution = weighting.get_severity_distribution(issues)

        assert len(distribution) == len(SeverityLevel)
        assert sum(distribution.values()) == 3

    def test_security_focused_weights(self):
        """Test security-focused weight profile."""
        weighting = SeverityWeighting(profile="security")

        issue = ClaimIssue(
            issue_type="performance",
            description="Performance issue",
            category=IssueCategory.PERFORMANCE,
        )

        # Security profile should down-weight performance vs security
        score = weighting.calculate_severity_score(issue)

        assert 0.0 <= score <= 1.0

    def test_performance_focused_weights(self):
        """Test performance-focused weight profile."""
        weighting = SeverityWeighting(profile="performance")

        issue = ClaimIssue(
            issue_type="compilation_time",
            description="Slow compilation",
            category=IssueCategory.PERFORMANCE,
        )

        # Performance profile should up-weight performance issues
        score = weighting.calculate_severity_score(issue)

        assert score > 0.3
