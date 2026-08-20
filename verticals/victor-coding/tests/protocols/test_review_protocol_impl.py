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

"""Tests for code review protocol types and data structures."""

import pytest
from pathlib import Path

pytest.importorskip("victor_coding")

from victor_coding.review.protocol import (
    ComplexityMetrics,
    DuplicationResult,
    FileReview,
    ReviewCategory,
    ReviewConfig,
    ReviewFinding,
    ReviewResult,
    ReviewRule,
    ReviewRuleSet,
    SecurityIssue,
    Severity,
    SourceLocation,
)

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestSeverity:
    """Tests for Severity enum."""

    def test_error_severity(self):
        """Test error severity."""
        assert Severity.ERROR.value == "error"

    def test_warning_severity(self):
        """Test warning severity."""
        assert Severity.WARNING.value == "warning"

    def test_info_severity(self):
        """Test info severity."""
        assert Severity.INFO.value == "info"

    def test_hint_severity(self):
        """Test hint severity."""
        assert Severity.HINT.value == "hint"


class TestReviewCategory:
    """Tests for ReviewCategory enum."""

    def test_style_category(self):
        """Test style category."""
        assert ReviewCategory.STYLE.value == "style"

    def test_security_category(self):
        """Test security category."""
        assert ReviewCategory.SECURITY.value == "security"

    def test_performance_category(self):
        """Test performance category."""
        assert ReviewCategory.PERFORMANCE.value == "performance"

    def test_all_categories_exist(self):
        """Test all expected categories exist."""
        categories = [c.value for c in ReviewCategory]
        assert "complexity" in categories
        assert "maintainability" in categories
        assert "documentation" in categories
        assert "testing" in categories
        assert "error_handling" in categories
        assert "naming" in categories
        assert "duplication" in categories


# =============================================================================
# SOURCE LOCATION TESTS
# =============================================================================


class TestSourceLocation:
    """Tests for SourceLocation dataclass."""

    def test_creation_minimal(self):
        """Test minimal source location creation."""
        loc = SourceLocation(
            file_path=Path("test.py"),
            start_line=10,
        )
        assert loc.file_path == Path("test.py")
        assert loc.start_line == 10
        assert loc.start_column == 0  # default
        assert loc.end_line == 10  # defaults to start_line

    def test_creation_full(self):
        """Test full source location creation."""
        loc = SourceLocation(
            file_path=Path("test.py"),
            start_line=10,
            start_column=5,
            end_line=15,
            end_column=20,
        )
        assert loc.end_line == 15
        assert loc.end_column == 20

    def test_post_init_sets_end_line(self):
        """Test __post_init__ sets end_line to start_line when None."""
        loc = SourceLocation(Path("test.py"), 10, 5, None)
        assert loc.end_line == 10


# =============================================================================
# REVIEW FINDING TESTS
# =============================================================================


class TestReviewFinding:
    """Tests for ReviewFinding dataclass."""

    @pytest.fixture
    def sample_location(self):
        """Create sample location."""
        return SourceLocation(Path("test.py"), 10, 5)

    @pytest.fixture
    def sample_finding(self, sample_location):
        """Create sample finding."""
        return ReviewFinding(
            rule_id="STYLE001",
            message="Line too long",
            severity=Severity.WARNING,
            category=ReviewCategory.STYLE,
            location=sample_location,
            code_snippet="very_long_line = 'test'",
            suggestion="Break into multiple lines",
            fix_available=True,
            fix_code="line = (\n    'test'\n)",
        )

    def test_creation(self, sample_finding):
        """Test finding creation."""
        assert sample_finding.rule_id == "STYLE001"
        assert sample_finding.severity == Severity.WARNING
        assert sample_finding.category == ReviewCategory.STYLE

    def test_to_dict(self, sample_finding):
        """Test to_dict conversion."""
        d = sample_finding.to_dict()
        assert d["rule_id"] == "STYLE001"
        assert d["severity"] == "warning"
        assert d["category"] == "style"
        assert d["line"] == 10
        assert d["fix_available"] is True

    def test_defaults(self, sample_location):
        """Test finding defaults."""
        finding = ReviewFinding(
            rule_id="TEST",
            message="Test",
            severity=Severity.INFO,
            category=ReviewCategory.STYLE,
            location=sample_location,
        )
        assert finding.code_snippet == ""
        assert finding.suggestion == ""
        assert finding.fix_available is False
        assert finding.metadata == {}


# =============================================================================
# REVIEW RULE TESTS
# =============================================================================


class TestReviewRule:
    """Tests for ReviewRule dataclass."""

    def test_creation(self):
        """Test rule creation."""
        rule = ReviewRule(
            id="STYLE001",
            name="Line length",
            description="Check line length",
            category=ReviewCategory.STYLE,
        )
        assert rule.id == "STYLE001"
        assert rule.severity == Severity.WARNING  # default
        assert rule.enabled is True  # default

    def test_matches_tags_empty_filter(self):
        """Test matches_tags with empty filter."""
        rule = ReviewRule(
            id="TEST",
            name="Test",
            description="Test rule",
            category=ReviewCategory.STYLE,
            tags=["python", "formatting"],
        )
        assert rule.matches_tags([]) is True

    def test_matches_tags_match_found(self):
        """Test matches_tags when match found."""
        rule = ReviewRule(
            id="TEST",
            name="Test",
            description="Test rule",
            category=ReviewCategory.STYLE,
            tags=["python", "formatting"],
        )
        assert rule.matches_tags(["python"]) is True
        assert rule.matches_tags(["formatting"]) is True

    def test_matches_tags_no_match(self):
        """Test matches_tags when no match."""
        rule = ReviewRule(
            id="TEST",
            name="Test",
            description="Test rule",
            category=ReviewCategory.STYLE,
            tags=["python"],
        )
        assert rule.matches_tags(["javascript"]) is False

    def test_matches_tags_rule_no_tags(self):
        """Test matches_tags when rule has no tags."""
        rule = ReviewRule(
            id="TEST",
            name="Test",
            description="Test rule",
            category=ReviewCategory.STYLE,
            tags=[],
        )
        assert rule.matches_tags(["python"]) is False


# =============================================================================
# REVIEW RULESET TESTS
# =============================================================================


class TestReviewRuleSet:
    """Tests for ReviewRuleSet dataclass."""

    @pytest.fixture
    def sample_rules(self):
        """Create sample rules."""
        return [
            ReviewRule(
                id="R1",
                name="Rule 1",
                description="",
                category=ReviewCategory.STYLE,
                enabled=True,
            ),
            ReviewRule(
                id="R2",
                name="Rule 2",
                description="",
                category=ReviewCategory.STYLE,
                enabled=False,
            ),
            ReviewRule(
                id="R3",
                name="Rule 3",
                description="",
                category=ReviewCategory.SECURITY,
                enabled=True,
            ),
        ]

    def test_creation(self, sample_rules):
        """Test ruleset creation."""
        ruleset = ReviewRuleSet(
            name="default",
            description="Default rules",
            rules=sample_rules,
        )
        assert ruleset.name == "default"
        assert len(ruleset.rules) == 3

    def test_get_enabled_rules(self, sample_rules):
        """Test get_enabled_rules."""
        ruleset = ReviewRuleSet(name="test", rules=sample_rules)
        enabled = ruleset.get_enabled_rules()
        assert len(enabled) == 2
        assert all(r.enabled for r in enabled)

    def test_get_rules_by_category(self, sample_rules):
        """Test get_rules_by_category."""
        ruleset = ReviewRuleSet(name="test", rules=sample_rules)
        style_rules = ruleset.get_rules_by_category(ReviewCategory.STYLE)
        # Only enabled rules with STYLE category
        assert len(style_rules) == 1
        assert style_rules[0].id == "R1"


# =============================================================================
# FILE REVIEW TESTS
# =============================================================================


class TestFileReview:
    """Tests for FileReview dataclass."""

    @pytest.fixture
    def sample_findings(self):
        """Create sample findings."""
        loc = SourceLocation(Path("test.py"), 10)
        return [
            ReviewFinding("E1", "Error", Severity.ERROR, ReviewCategory.STYLE, loc),
            ReviewFinding("W1", "Warning", Severity.WARNING, ReviewCategory.STYLE, loc),
            ReviewFinding("W2", "Warning", Severity.WARNING, ReviewCategory.STYLE, loc),
            ReviewFinding("I1", "Info", Severity.INFO, ReviewCategory.STYLE, loc),
        ]

    def test_creation(self, sample_findings):
        """Test file review creation."""
        review = FileReview(
            file_path=Path("test.py"),
            findings=sample_findings,
            lines_analyzed=100,
        )
        assert review.file_path == Path("test.py")
        assert len(review.findings) == 4

    def test_error_count(self, sample_findings):
        """Test error_count property."""
        review = FileReview(Path("test.py"), findings=sample_findings)
        assert review.error_count == 1

    def test_warning_count(self, sample_findings):
        """Test warning_count property."""
        review = FileReview(Path("test.py"), findings=sample_findings)
        assert review.warning_count == 2

    def test_info_count(self, sample_findings):
        """Test info_count property."""
        review = FileReview(Path("test.py"), findings=sample_findings)
        assert review.info_count == 1


# =============================================================================
# REVIEW RESULT TESTS
# =============================================================================


class TestReviewResult:
    """Tests for ReviewResult dataclass."""

    @pytest.fixture
    def file_review_with_error(self):
        """Create file review with error."""
        loc = SourceLocation(Path("error.py"), 10)
        return FileReview(
            file_path=Path("error.py"),
            findings=[
                ReviewFinding("E1", "Error", Severity.ERROR, ReviewCategory.STYLE, loc),
            ],
        )

    @pytest.fixture
    def file_review_clean(self):
        """Create clean file review."""
        loc = SourceLocation(Path("clean.py"), 10)
        return FileReview(
            file_path=Path("clean.py"),
            findings=[
                ReviewFinding("I1", "Info", Severity.INFO, ReviewCategory.STYLE, loc),
            ],
        )

    def test_passed_no_errors(self, file_review_clean):
        """Test passed property with no errors."""
        result = ReviewResult(file_reviews=[file_review_clean])
        assert result.passed is True

    def test_passed_with_errors(self, file_review_with_error):
        """Test passed property with errors."""
        result = ReviewResult(file_reviews=[file_review_with_error])
        assert result.passed is False

    def test_total_errors(self, file_review_with_error, file_review_clean):
        """Test total_errors property."""
        result = ReviewResult(file_reviews=[file_review_with_error, file_review_clean])
        assert result.total_errors == 1

    def test_total_warnings(self):
        """Test total_warnings property."""
        loc = SourceLocation(Path("test.py"), 10)
        review = FileReview(
            Path("test.py"),
            findings=[
                ReviewFinding("W1", "Warn", Severity.WARNING, ReviewCategory.STYLE, loc),
                ReviewFinding("W2", "Warn", Severity.WARNING, ReviewCategory.STYLE, loc),
            ],
        )
        result = ReviewResult(file_reviews=[review])
        assert result.total_warnings == 2

    def test_get_findings_by_category(self):
        """Test get_findings_by_category."""
        loc = SourceLocation(Path("test.py"), 10)
        review = FileReview(
            Path("test.py"),
            findings=[
                ReviewFinding("S1", "Style", Severity.INFO, ReviewCategory.STYLE, loc),
                ReviewFinding("SEC1", "Security", Severity.ERROR, ReviewCategory.SECURITY, loc),
            ],
        )
        result = ReviewResult(file_reviews=[review])
        security_findings = result.get_findings_by_category(ReviewCategory.SECURITY)
        assert len(security_findings) == 1
        assert security_findings[0].rule_id == "SEC1"

    def test_get_findings_by_severity(self):
        """Test get_findings_by_severity."""
        loc = SourceLocation(Path("test.py"), 10)
        review = FileReview(
            Path("test.py"),
            findings=[
                ReviewFinding("E1", "Error", Severity.ERROR, ReviewCategory.STYLE, loc),
                ReviewFinding("W1", "Warn", Severity.WARNING, ReviewCategory.STYLE, loc),
            ],
        )
        result = ReviewResult(file_reviews=[review])
        errors = result.get_findings_by_severity(Severity.ERROR)
        assert len(errors) == 1


# =============================================================================
# REVIEW CONFIG TESTS
# =============================================================================


class TestReviewConfig:
    """Tests for ReviewConfig dataclass."""

    def test_default_config(self):
        """Test default config values."""
        config = ReviewConfig()
        assert len(config.enabled_categories) == len(ReviewCategory)
        assert config.min_severity == Severity.INFO
        assert config.fail_on_error is True
        assert config.fail_on_warning is False
        assert config.max_findings_per_file == 100

    def test_custom_config(self):
        """Test custom config."""
        config = ReviewConfig(
            enabled_categories=[ReviewCategory.SECURITY],
            min_severity=Severity.WARNING,
            fail_on_warning=True,
        )
        assert len(config.enabled_categories) == 1
        assert config.min_severity == Severity.WARNING
        assert config.fail_on_warning is True


# =============================================================================
# COMPLEXITY METRICS TESTS
# =============================================================================


class TestComplexityMetrics:
    """Tests for ComplexityMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metrics."""
        metrics = ComplexityMetrics()
        assert metrics.cyclomatic_complexity == 0
        assert metrics.maintainability_index == 100.0

    def test_custom_metrics(self):
        """Test custom metrics."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=15,
            cognitive_complexity=20,
            lines_of_code=500,
            number_of_functions=10,
            max_nesting_depth=4,
        )
        assert metrics.cyclomatic_complexity == 15
        assert metrics.lines_of_code == 500


# =============================================================================
# SECURITY ISSUE TESTS
# =============================================================================


class TestSecurityIssue:
    """Tests for SecurityIssue dataclass."""

    def test_creation(self):
        """Test security issue creation."""
        issue = SecurityIssue(
            vulnerability_type="sql_injection",
            severity=Severity.ERROR,
            cwe_id="CWE-89",
            description="SQL injection vulnerability",
            remediation="Use parameterized queries",
            confidence=0.95,
        )
        assert issue.vulnerability_type == "sql_injection"
        assert issue.cwe_id == "CWE-89"
        assert issue.confidence == 0.95

    def test_defaults(self):
        """Test default values."""
        issue = SecurityIssue(
            vulnerability_type="test",
            severity=Severity.WARNING,
        )
        assert issue.cwe_id is None
        assert issue.description == ""
        assert issue.confidence == 1.0


# =============================================================================
# DUPLICATION RESULT TESTS
# =============================================================================


class TestDuplicationResult:
    """Tests for DuplicationResult dataclass."""

    def test_default_result(self):
        """Test default duplication result."""
        result = DuplicationResult()
        assert result.duplicate_blocks == []
        assert result.total_duplicate_lines == 0
        assert result.duplication_percentage == 0.0

    def test_with_duplicates(self):
        """Test result with duplicates."""
        loc1 = SourceLocation(Path("a.py"), 10, 0, 20, 0)
        loc2 = SourceLocation(Path("b.py"), 30, 0, 40, 0)
        result = DuplicationResult(
            duplicate_blocks=[(loc1, loc2)],
            total_duplicate_lines=20,
            duplication_percentage=5.5,
        )
        assert len(result.duplicate_blocks) == 1
        assert result.total_duplicate_lines == 20
        assert result.duplication_percentage == 5.5
