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

"""Tests for test result correlation and scoring."""

import tempfile
from pathlib import Path


from victor.evaluation.result_correlation import (
    FailureCategory,
    TestCorrelation,
    SWEBenchScore,
    CorrelationReport,
    ResultCorrelator,
    correlate_validation_results,
    analyze_failure_patterns,
    save_correlation_report,
)
from victor.evaluation.baseline_validator import (
    TestBaseline,
    BaselineValidationResult,
    BaselineStatus,
    TestStatus,
)
from victor.evaluation.test_runners import TestResult, TestRunResults


class TestFailureCategory:
    """Tests for FailureCategory enum."""

    def test_category_values(self):
        """Test all failure category values."""
        assert FailureCategory.ASSERTION_ERROR.value == "assertion_error"
        assert FailureCategory.IMPORT_ERROR.value == "import_error"
        assert FailureCategory.SYNTAX_ERROR.value == "syntax_error"
        assert FailureCategory.TYPE_ERROR.value == "type_error"
        assert FailureCategory.ATTRIBUTE_ERROR.value == "attribute_error"
        assert FailureCategory.RUNTIME_ERROR.value == "runtime_error"
        assert FailureCategory.TIMEOUT.value == "timeout"
        assert FailureCategory.SETUP_ERROR.value == "setup_error"
        assert FailureCategory.TEARDOWN_ERROR.value == "teardown_error"
        assert FailureCategory.FIXTURE_ERROR.value == "fixture_error"
        assert FailureCategory.ENVIRONMENT_ERROR.value == "environment_error"
        assert FailureCategory.UNKNOWN.value == "unknown"


class TestTestCorrelation:
    """Tests for TestCorrelation dataclass."""

    def test_basic_correlation(self):
        """Test creating a basic correlation."""
        correlation = TestCorrelation(
            test_id="test_foo",
            expected_status=TestStatus.PASSED,
            actual_status=TestStatus.PASSED,
            matches_expectation=True,
        )
        assert correlation.test_id == "test_foo"
        assert correlation.matches_expectation is True
        assert correlation.failure_category is None

    def test_correlation_with_failure(self):
        """Test correlation with failure category."""
        correlation = TestCorrelation(
            test_id="test_bar",
            expected_status=TestStatus.PASSED,
            actual_status=TestStatus.FAILED,
            matches_expectation=False,
            failure_category=FailureCategory.ASSERTION_ERROR,
            error_message="AssertionError: expected True",
        )
        assert correlation.matches_expectation is False
        assert correlation.failure_category == FailureCategory.ASSERTION_ERROR

    def test_to_dict(self):
        """Test serialization to dictionary."""
        correlation = TestCorrelation(
            test_id="test_foo",
            expected_status=TestStatus.FAILED,
            actual_status=TestStatus.PASSED,
            matches_expectation=False,
            failure_category=None,
            context={"transition": "failing_to_passing"},
        )
        data = correlation.to_dict()
        assert data["test_id"] == "test_foo"
        assert data["expected_status"] == "failed"
        assert data["actual_status"] == "passed"
        assert data["matches_expectation"] is False
        assert data["context"]["transition"] == "failing_to_passing"


class TestSWEBenchScore:
    """Tests for SWEBenchScore dataclass."""

    def test_default_values(self):
        """Test default score values."""
        score = SWEBenchScore(instance_id="test_instance")
        assert score.resolved is False
        assert score.partial is False
        assert score.fail_to_pass_score == 0.0
        assert score.pass_to_pass_score == 1.0
        assert score.overall_score == 0.0
        assert score.tests_fixed == 0
        assert score.tests_broken == 0

    def test_resolved_score(self):
        """Test a fully resolved score."""
        score = SWEBenchScore(
            instance_id="test_resolved",
            resolved=True,
            partial=False,
            fail_to_pass_score=1.0,
            pass_to_pass_score=1.0,
            overall_score=1.0,
            tests_fixed=2,
            tests_broken=0,
            total_fail_to_pass=2,
            total_pass_to_pass=5,
        )
        assert score.resolved is True
        assert score.overall_score == 1.0

    def test_partial_score(self):
        """Test a partially resolved score."""
        score = SWEBenchScore(
            instance_id="test_partial",
            resolved=False,
            partial=True,
            fail_to_pass_score=0.5,
            pass_to_pass_score=0.8,
            overall_score=0.3,
            tests_fixed=1,
            tests_broken=1,
            total_fail_to_pass=2,
            total_pass_to_pass=5,
        )
        assert score.partial is True
        assert score.tests_fixed == 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        correlation = TestCorrelation(
            test_id="test_a",
            expected_status=TestStatus.FAILED,
            actual_status=TestStatus.PASSED,
            matches_expectation=False,
        )
        score = SWEBenchScore(
            instance_id="test",
            resolved=True,
            fail_to_pass_score=1.0,
            correlations=[correlation],
            metadata={"repo": "test/repo"},
        )
        data = score.to_dict()
        assert data["instance_id"] == "test"
        assert data["resolved"] is True
        assert len(data["correlations"]) == 1
        assert data["metadata"]["repo"] == "test/repo"


class TestCorrelationReport:
    """Tests for CorrelationReport dataclass."""

    def test_default_values(self):
        """Test default report values."""
        report = CorrelationReport()
        assert report.total_instances == 0
        assert report.resolved_count == 0
        assert report.avg_overall_score == 0.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        report = CorrelationReport(
            total_instances=10,
            resolved_count=5,
            partial_count=3,
            failed_count=2,
            avg_f2p_score=0.6,
            avg_p2p_score=0.9,
            avg_overall_score=0.5,
            by_repo={"test/repo": {"count": 10, "resolved": 5}},
        )
        data = report.to_dict()
        assert data["summary"]["total_instances"] == 10
        assert data["summary"]["resolved"] == 5
        assert data["summary"]["resolve_rate"] == 0.5
        assert data["scores"]["avg_f2p"] == 0.6

    def test_to_text(self):
        """Test text report generation."""
        report = CorrelationReport(
            total_instances=10,
            resolved_count=5,
            partial_count=3,
            failed_count=2,
            avg_f2p_score=0.6,
            avg_p2p_score=0.9,
            avg_overall_score=0.5,
            by_repo={"django/django": {"count": 5, "resolved": 3, "avg_score": 0.6}},
            by_difficulty={"easy": {"count": 4, "resolved": 3, "avg_score": 0.7}},
            failure_analysis={"assertion_error": 5, "import_error": 2},
        )
        text = report.to_text()
        assert "SWE-bench Evaluation Report" in text
        assert "Total Instances: 10" in text
        assert "Resolved: 5" in text
        assert "django/django" in text
        assert "assertion_error: 5" in text


class TestResultCorrelator:
    """Tests for ResultCorrelator class."""

    def test_categorize_assertion_error(self):
        """Test categorizing assertion errors."""
        correlator = ResultCorrelator()
        category = correlator.categorize_failure("AssertionError: expected True got False")
        assert category == FailureCategory.ASSERTION_ERROR

    def test_categorize_import_error(self):
        """Test categorizing import errors."""
        correlator = ResultCorrelator()
        category = correlator.categorize_failure("ImportError: No module named 'foo'")
        assert category == FailureCategory.IMPORT_ERROR

        category = correlator.categorize_failure("ModuleNotFoundError: No module named 'bar'")
        assert category == FailureCategory.IMPORT_ERROR

    def test_categorize_syntax_error(self):
        """Test categorizing syntax errors."""
        correlator = ResultCorrelator()
        category = correlator.categorize_failure("SyntaxError: invalid syntax")
        assert category == FailureCategory.SYNTAX_ERROR

    def test_categorize_type_error(self):
        """Test categorizing type errors."""
        correlator = ResultCorrelator()
        category = correlator.categorize_failure("TypeError: expected str, got int")
        assert category == FailureCategory.TYPE_ERROR

    def test_categorize_attribute_error(self):
        """Test categorizing attribute errors."""
        correlator = ResultCorrelator()
        category = correlator.categorize_failure("AttributeError: 'Foo' has no attribute 'bar'")
        assert category == FailureCategory.ATTRIBUTE_ERROR

    def test_categorize_timeout(self):
        """Test categorizing timeout errors."""
        correlator = ResultCorrelator()
        category = correlator.categorize_failure("TimeoutError: test timed out")
        assert category == FailureCategory.TIMEOUT

    def test_categorize_fixture_error(self):
        """Test categorizing fixture errors."""
        correlator = ResultCorrelator()
        category = correlator.categorize_failure("fixture 'db' not found")
        assert category == FailureCategory.FIXTURE_ERROR

    def test_categorize_runtime_error(self):
        """Test categorizing runtime errors."""
        correlator = ResultCorrelator()
        category = correlator.categorize_failure("RuntimeError: cannot run")
        assert category == FailureCategory.RUNTIME_ERROR

        category = correlator.categorize_failure("ValueError: invalid value")
        assert category == FailureCategory.RUNTIME_ERROR

        category = correlator.categorize_failure("KeyError: 'missing_key'")
        assert category == FailureCategory.RUNTIME_ERROR

    def test_categorize_environment_error(self):
        """Test categorizing environment errors."""
        correlator = ResultCorrelator()
        category = correlator.categorize_failure("FileNotFoundError: /path/to/file")
        assert category == FailureCategory.ENVIRONMENT_ERROR

        category = correlator.categorize_failure("PermissionError: access denied")
        assert category == FailureCategory.ENVIRONMENT_ERROR

    def test_categorize_unknown(self):
        """Test categorizing unknown errors."""
        correlator = ResultCorrelator()
        category = correlator.categorize_failure("some random error message")
        assert category == FailureCategory.UNKNOWN

        category = correlator.categorize_failure("")
        assert category == FailureCategory.UNKNOWN

        category = correlator.categorize_failure(None)
        assert category == FailureCategory.UNKNOWN

    def test_correlate_test_passing(self):
        """Test correlating a passing test."""
        correlator = ResultCorrelator()
        result = TestResult(test_name="test_foo", passed=True)
        correlation = correlator.correlate_test("test_foo", TestStatus.PASSED, result)
        assert correlation.matches_expectation is True
        assert correlation.actual_status == TestStatus.PASSED

    def test_correlate_test_failing(self):
        """Test correlating a failing test."""
        correlator = ResultCorrelator()
        # A test with error_message is categorized as ERROR status, not FAILED
        # So we expect TestStatus.ERROR to match correctly
        result = TestResult(
            test_name="test_bar",
            passed=False,
            error_message="AssertionError: test failed",
        )
        correlation = correlator.correlate_test("test_bar", TestStatus.ERROR, result)
        assert correlation.matches_expectation is True
        assert correlation.failure_category == FailureCategory.ASSERTION_ERROR

    def test_correlate_test_mismatch(self):
        """Test correlating a test with unexpected result."""
        correlator = ResultCorrelator()
        result = TestResult(test_name="test_baz", passed=True)
        correlation = correlator.correlate_test("test_baz", TestStatus.FAILED, result)
        assert correlation.matches_expectation is False

    def test_compute_score_resolved(self):
        """Test computing score for resolved instance."""
        correlator = ResultCorrelator()

        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc123",
            fail_to_pass=["test_a", "test_b"],
            pass_to_pass=["test_c"],
            status=BaselineStatus.VALID,
        )

        post_results = TestRunResults(
            results=[
                TestResult(test_name="test_a", passed=True),
                TestResult(test_name="test_b", passed=True),
                TestResult(test_name="test_c", passed=True),
            ]
        )

        validation = BaselineValidationResult(
            instance_id="test",
            baseline=baseline,
            post_change_results=post_results,
        )

        score = correlator.compute_score(validation)
        assert score.resolved is True
        assert score.tests_fixed == 2
        assert score.tests_broken == 0
        assert score.fail_to_pass_score == 1.0
        assert score.pass_to_pass_score == 1.0

    def test_compute_score_partial(self):
        """Test computing score for partially resolved instance."""
        correlator = ResultCorrelator()

        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc123",
            fail_to_pass=["test_a", "test_b"],
            pass_to_pass=["test_c"],
            status=BaselineStatus.VALID,
        )

        post_results = TestRunResults(
            results=[
                TestResult(test_name="test_a", passed=True),  # Fixed
                TestResult(test_name="test_b", passed=False),  # Still failing
                TestResult(test_name="test_c", passed=True),  # Still passing
            ]
        )

        validation = BaselineValidationResult(
            instance_id="test",
            baseline=baseline,
            post_change_results=post_results,
        )

        score = correlator.compute_score(validation)
        assert score.partial is True
        assert score.resolved is False
        assert score.tests_fixed == 1
        assert score.fail_to_pass_score == 0.5

    def test_compute_score_with_regression(self):
        """Test computing score with test regression."""
        correlator = ResultCorrelator()

        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc123",
            fail_to_pass=["test_a"],
            pass_to_pass=["test_c", "test_d"],
            status=BaselineStatus.VALID,
        )

        post_results = TestRunResults(
            results=[
                TestResult(test_name="test_a", passed=True),  # Fixed
                TestResult(test_name="test_c", passed=False),  # Broken!
                TestResult(test_name="test_d", passed=True),  # Still passing
            ]
        )

        validation = BaselineValidationResult(
            instance_id="test",
            baseline=baseline,
            post_change_results=post_results,
        )

        score = correlator.compute_score(validation)
        assert score.tests_fixed == 1
        assert score.tests_broken == 1
        assert score.pass_to_pass_score == 0.5
        assert score.resolved is False

    def test_compute_score_with_metadata(self):
        """Test computing score with instance metadata."""
        correlator = ResultCorrelator()

        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc123",
            fail_to_pass=[],
            pass_to_pass=[],
            status=BaselineStatus.VALID,
        )

        validation = BaselineValidationResult(
            instance_id="test",
            baseline=baseline,
            post_change_results=TestRunResults(),
        )

        metadata = {"repo": "django/django", "difficulty": "hard"}
        score = correlator.compute_score(validation, metadata)
        assert score.metadata["repo"] == "django/django"
        assert score.metadata["difficulty"] == "hard"

    def test_generate_report_empty(self):
        """Test generating report with no scores."""
        correlator = ResultCorrelator()
        report = correlator.generate_report([])
        assert report.total_instances == 0
        assert report.avg_overall_score == 0.0

    def test_generate_report_with_scores(self):
        """Test generating report with multiple scores."""
        correlator = ResultCorrelator()

        scores = [
            SWEBenchScore(
                instance_id="inst_1",
                resolved=True,
                fail_to_pass_score=1.0,
                pass_to_pass_score=1.0,
                overall_score=1.0,
                metadata={"repo": "django/django", "difficulty": "easy"},
            ),
            SWEBenchScore(
                instance_id="inst_2",
                resolved=False,
                partial=True,
                fail_to_pass_score=0.5,
                pass_to_pass_score=1.0,
                overall_score=0.5,
                metadata={"repo": "django/django", "difficulty": "hard"},
            ),
            SWEBenchScore(
                instance_id="inst_3",
                resolved=False,
                partial=False,
                fail_to_pass_score=0.0,
                pass_to_pass_score=0.5,
                overall_score=0.0,
                metadata={"repo": "flask/flask", "difficulty": "hard"},
            ),
        ]

        report = correlator.generate_report(scores)
        assert report.total_instances == 3
        assert report.resolved_count == 1
        assert report.partial_count == 1
        assert report.failed_count == 1
        assert report.avg_f2p_score == 0.5
        assert "django/django" in report.by_repo
        assert "flask/flask" in report.by_repo


class TestCorrelateValidationResults:
    """Tests for correlate_validation_results function."""

    def test_correlate_empty(self):
        """Test correlating empty results."""
        report = correlate_validation_results([])
        assert report.total_instances == 0

    def test_correlate_with_metadata(self):
        """Test correlating with metadata."""
        baseline = TestBaseline(
            instance_id="test_1",
            repo="test/repo",
            base_commit="abc",
            fail_to_pass=["test_a"],
            pass_to_pass=[],
            status=BaselineStatus.VALID,
        )

        results = [
            BaselineValidationResult(
                instance_id="test_1",
                baseline=baseline,
                post_change_results=TestRunResults(
                    results=[TestResult(test_name="test_a", passed=True)]
                ),
            )
        ]

        metadata = {"test_1": {"repo": "django/django", "difficulty": "easy"}}
        report = correlate_validation_results(results, metadata)
        assert report.total_instances == 1
        assert "django/django" in report.by_repo


class TestAnalyzeFailurePatterns:
    """Tests for analyze_failure_patterns function."""

    def test_analyze_empty(self):
        """Test analyzing empty scores."""
        patterns = analyze_failure_patterns([])
        assert patterns["by_category"] == {}
        assert patterns["common_errors"] == []

    def test_analyze_with_failures(self):
        """Test analyzing scores with failures."""
        correlation = TestCorrelation(
            test_id="test_module::test_func",
            expected_status=TestStatus.PASSED,
            actual_status=TestStatus.FAILED,
            matches_expectation=False,
            failure_category=FailureCategory.ASSERTION_ERROR,
            error_message="AssertionError: Expected True",
        )

        scores = [
            SWEBenchScore(
                instance_id="test",
                correlations=[correlation],
            )
        ]

        patterns = analyze_failure_patterns(scores)
        assert "assertion_error" in patterns["by_category"]
        assert patterns["by_category"]["assertion_error"] == 1
        assert "test_module" in patterns["by_test_pattern"]

    def test_analyze_flaky_tests(self):
        """Test detecting flaky tests."""
        correlation = TestCorrelation(
            test_id="test_flaky",
            expected_status=TestStatus.PASSED,
            actual_status=TestStatus.FAILED,
            matches_expectation=False,
            is_flaky=True,
        )

        scores = [
            SWEBenchScore(
                instance_id="test",
                correlations=[correlation],
            )
        ]

        patterns = analyze_failure_patterns(scores)
        assert "test_flaky" in patterns["flaky_tests"]


class TestSaveCorrelationReport:
    """Tests for save_correlation_report function."""

    def test_save_report(self):
        """Test saving report to files."""
        report = CorrelationReport(
            total_instances=1,
            resolved_count=1,
            avg_overall_score=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report"
            save_correlation_report(report, output_path)

            json_path = output_path.with_suffix(".json")
            txt_path = output_path.with_suffix(".txt")

            assert json_path.exists()
            assert txt_path.exists()

            import json

            with open(json_path) as f:
                data = json.load(f)
            assert data["summary"]["total_instances"] == 1

            txt_content = txt_path.read_text()
            assert "SWE-bench Evaluation Report" in txt_content

    def test_save_report_json_only(self):
        """Test saving report without text file."""
        report = CorrelationReport()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report"
            save_correlation_report(report, output_path, include_text=False)

            json_path = output_path.with_suffix(".json")
            txt_path = output_path.with_suffix(".txt")

            assert json_path.exists()
            assert not txt_path.exists()

    def test_save_report_creates_directories(self):
        """Test that save creates parent directories."""
        report = CorrelationReport()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "report"
            save_correlation_report(report, output_path)

            assert output_path.with_suffix(".json").exists()
