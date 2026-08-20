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

"""Tests for baseline test validator."""

import tempfile
from datetime import datetime
from pathlib import Path


from victor.evaluation.baseline_validator import (
    TestStatus,
    get_test_status,
    BaselineStatus,
    TestBaseline,
    BaselineValidationResult,
    BaselineCache,
    BaselineValidator,
    check_environment_health,
)
from victor.evaluation.test_runners import TestResult, TestRunResults


class TestTestStatus:
    """Tests for TestStatus enum."""

    def test_status_values(self):
        """Test all status enum values."""
        assert TestStatus.PASSED.value == "passed"
        assert TestStatus.FAILED.value == "failed"
        assert TestStatus.ERROR.value == "error"
        assert TestStatus.SKIPPED.value == "skipped"


class TestGetTestStatus:
    """Tests for get_test_status helper function."""

    def test_passed_result(self):
        """Test passed result returns PASSED status."""
        result = TestResult(test_name="test_foo", passed=True)
        assert get_test_status(result) == TestStatus.PASSED

    def test_failed_result(self):
        """Test failed result returns FAILED status."""
        result = TestResult(test_name="test_foo", passed=False)
        assert get_test_status(result) == TestStatus.FAILED

    def test_skipped_result(self):
        """Test skipped result returns SKIPPED status."""
        result = TestResult(test_name="test_foo", passed=True, skip_reason="not applicable")
        assert get_test_status(result) == TestStatus.SKIPPED

    def test_error_result(self):
        """Test error result returns ERROR status."""
        result = TestResult(test_name="test_foo", passed=False, error_message="ImportError")
        assert get_test_status(result) == TestStatus.ERROR


class TestBaselineStatus:
    """Tests for BaselineStatus enum."""

    def test_status_values(self):
        """Test all baseline status values."""
        assert BaselineStatus.VALID.value == "valid"
        assert BaselineStatus.INVALID.value == "invalid"
        assert BaselineStatus.ERROR.value == "error"
        assert BaselineStatus.MISSING.value == "missing"
        assert BaselineStatus.ENVIRONMENT_ISSUE.value == "environment_issue"


class TestTestBaseline:
    """Tests for TestBaseline dataclass."""

    def test_default_values(self):
        """Test default baseline values."""
        baseline = TestBaseline(
            instance_id="test_instance",
            repo="test/repo",
            base_commit="abc123",
            fail_to_pass=[],
            pass_to_pass=[],
        )
        assert baseline.instance_id == "test_instance"
        assert baseline.repo == "test/repo"
        assert baseline.base_commit == "abc123"
        assert baseline.fail_to_pass_results == {}
        assert baseline.pass_to_pass_results == {}
        assert baseline.status == BaselineStatus.MISSING
        assert baseline.environment_valid is True

    def test_is_valid_true(self):
        """Test is_valid returns True for valid baseline."""
        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc",
            fail_to_pass=[],
            pass_to_pass=[],
            status=BaselineStatus.VALID,
            environment_valid=True,
        )
        assert baseline.is_valid() is True

    def test_is_valid_false_invalid_status(self):
        """Test is_valid returns False for invalid status."""
        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc",
            fail_to_pass=[],
            pass_to_pass=[],
            status=BaselineStatus.INVALID,
            environment_valid=True,
        )
        assert baseline.is_valid() is False

    def test_is_valid_false_environment_issue(self):
        """Test is_valid returns False for environment issues."""
        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc",
            fail_to_pass=[],
            pass_to_pass=[],
            status=BaselineStatus.VALID,
            environment_valid=False,
        )
        assert baseline.is_valid() is False

    def test_get_validation_summary(self):
        """Test validation summary generation."""
        baseline = TestBaseline(
            instance_id="test_instance",
            repo="test/repo",
            base_commit="abc123",
            fail_to_pass=["test_a", "test_b"],
            pass_to_pass=["test_c"],
            fail_to_pass_results={
                "test_a": TestResult(test_name="test_a", passed=False),
                "test_b": TestResult(test_name="test_b", passed=False),
            },
            pass_to_pass_results={
                "test_c": TestResult(test_name="test_c", passed=True),
            },
            status=BaselineStatus.VALID,
            duration_seconds=5.0,
        )

        summary = baseline.get_validation_summary()
        assert summary["instance_id"] == "test_instance"
        assert summary["status"] == "valid"
        assert summary["fail_to_pass"]["expected"] == 2
        assert summary["fail_to_pass"]["actually_failing"] == 2
        assert summary["fail_to_pass"]["match"] is True
        assert summary["pass_to_pass"]["expected"] == 1
        assert summary["pass_to_pass"]["actually_passing"] == 1
        assert summary["pass_to_pass"]["match"] is True

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        baseline = TestBaseline(
            instance_id="test_instance",
            repo="test/repo",
            base_commit="abc123",
            fail_to_pass=["test_a"],
            pass_to_pass=["test_b"],
            fail_to_pass_results={
                "test_a": TestResult(test_name="test_a", passed=False, duration_ms=100.0),
            },
            pass_to_pass_results={
                "test_b": TestResult(test_name="test_b", passed=True),
            },
            status=BaselineStatus.VALID,
            environment_valid=True,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            duration_seconds=5.0,
        )

        # Serialize
        data = baseline.to_dict()
        assert data["instance_id"] == "test_instance"
        assert data["status"] == "valid"
        assert "test_a" in data["fail_to_pass_results"]

        # Deserialize
        restored = TestBaseline.from_dict(data)
        assert restored.instance_id == baseline.instance_id
        assert restored.repo == baseline.repo
        assert restored.status == baseline.status
        assert "test_a" in restored.fail_to_pass_results
        assert restored.fail_to_pass_results["test_a"].passed is False


class TestBaselineValidationResult:
    """Tests for BaselineValidationResult dataclass."""

    def test_default_values(self):
        """Test default validation result values."""
        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc",
            fail_to_pass=[],
            pass_to_pass=[],
        )
        result = BaselineValidationResult(
            instance_id="test",
            baseline=baseline,
            post_change_results=TestRunResults(),
        )
        assert result.fail_to_pass_fixed == []
        assert result.pass_to_pass_broken == []
        assert result.success is False
        assert result.partial_success is False
        assert result.score == 0.0

    def test_to_dict(self):
        """Test serialization of validation result."""
        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc",
            fail_to_pass=["test_a"],
            pass_to_pass=["test_b"],
        )
        result = BaselineValidationResult(
            instance_id="test",
            baseline=baseline,
            post_change_results=TestRunResults(),
            fail_to_pass_fixed=["test_a"],
            success=True,
            score=1.0,
        )

        data = result.to_dict()
        assert data["instance_id"] == "test"
        assert data["fail_to_pass_fixed"] == ["test_a"]
        assert data["success"] is True
        assert data["score"] == 1.0
        assert data["total_fail_to_pass"] == 1
        assert data["total_pass_to_pass"] == 1


class TestBaselineCache:
    """Tests for BaselineCache."""

    def test_init_creates_directories(self):
        """Test that initialization creates cache directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache = BaselineCache(cache_dir)

            assert cache_dir.exists()
            assert cache.db_path.exists()

    def test_set_and_get(self):
        """Test caching and retrieving baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BaselineCache(Path(tmpdir))

            baseline = TestBaseline(
                instance_id="test_instance",
                repo="test/repo",
                base_commit="abc123",
                fail_to_pass=[],
                pass_to_pass=[],
                status=BaselineStatus.VALID,
            )

            cache.set(baseline)

            retrieved = cache.get("test_instance", "test/repo", "abc123")
            assert retrieved is not None
            assert retrieved.instance_id == "test_instance"
            assert retrieved.status == BaselineStatus.VALID

    def test_get_missing(self):
        """Test getting non-existent baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BaselineCache(Path(tmpdir))

            result = cache.get("nonexistent", "repo", "commit")
            assert result is None

    def test_invalidate(self):
        """Test invalidating cached baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BaselineCache(Path(tmpdir))

            baseline = TestBaseline(
                instance_id="test_instance",
                repo="test/repo",
                base_commit="abc123",
                fail_to_pass=[],
                pass_to_pass=[],
                status=BaselineStatus.VALID,
            )

            cache.set(baseline)
            assert cache.get("test_instance", "test/repo", "abc123") is not None

            cache.invalidate("test/repo", "abc123")
            assert cache.get("test_instance", "test/repo", "abc123") is None

    def test_clear(self):
        """Test clearing all cached baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BaselineCache(Path(tmpdir))

            for i in range(3):
                baseline = TestBaseline(
                    instance_id=f"test_{i}",
                    repo="test/repo",
                    base_commit=f"commit_{i}",
                    fail_to_pass=[],
                    pass_to_pass=[],
                )
                cache.set(baseline)

            stats = cache.get_stats()
            assert stats["total"] == 3

            cache.clear()

            stats = cache.get_stats()
            assert stats["total"] == 0

    def test_get_stats(self):
        """Test getting cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BaselineCache(Path(tmpdir))

            for i in range(2):
                baseline = TestBaseline(
                    instance_id=f"django_{i}",
                    repo="django/django",
                    base_commit=f"commit_{i}",
                    fail_to_pass=[],
                    pass_to_pass=[],
                )
                cache.set(baseline)

            baseline = TestBaseline(
                instance_id="flask_0",
                repo="pallets/flask",
                base_commit="commit",
                fail_to_pass=[],
                pass_to_pass=[],
            )
            cache.set(baseline)

            stats = cache.get_stats()
            assert stats["total"] == 3
            assert stats["valid"] == 3
            assert "django/django" in stats["by_repo"]
            assert stats["by_repo"]["django/django"] == 2


class TestBaselineValidator:
    """Tests for BaselineValidator."""

    def test_init_default(self):
        """Test default initialization."""
        validator = BaselineValidator()
        assert validator.test_registry is not None
        assert validator.use_cache is True
        assert validator.cache is not None

    def test_init_no_cache(self):
        """Test initialization without cache."""
        validator = BaselineValidator(use_cache=False)
        assert validator.cache is None
        assert validator.use_cache is False

    def test_validate_baseline_all_valid(self):
        """Test validating a baseline where all tests match expectations."""
        validator = BaselineValidator(use_cache=False)

        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc",
            fail_to_pass=["test_a"],
            pass_to_pass=["test_b"],
            fail_to_pass_results={
                "test_a": TestResult(test_name="test_a", passed=False),
            },
            pass_to_pass_results={
                "test_b": TestResult(test_name="test_b", passed=True),
            },
        )

        result = validator._validate_baseline(baseline)
        assert result.status == BaselineStatus.VALID

    def test_validate_baseline_f2p_invalid(self):
        """Test validating a baseline where F2P tests pass unexpectedly."""
        validator = BaselineValidator(use_cache=False)

        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc",
            fail_to_pass=["test_a"],
            pass_to_pass=[],
            fail_to_pass_results={
                "test_a": TestResult(test_name="test_a", passed=True),  # Should be failing
            },
        )

        result = validator._validate_baseline(baseline)
        assert result.status == BaselineStatus.INVALID
        assert "FAIL_TO_PASS" in result.error_message

    def test_validate_baseline_p2p_invalid(self):
        """Test validating a baseline where P2P tests fail unexpectedly."""
        validator = BaselineValidator(use_cache=False)

        # Need at least one P2P test passing to avoid environment issue detection.
        # Having some tests pass means it's not a total environment failure.
        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc",
            fail_to_pass=["test_a"],
            pass_to_pass=["test_b", "test_c"],
            fail_to_pass_results={
                "test_a": TestResult(test_name="test_a", passed=False),  # Correctly failing
            },
            pass_to_pass_results={
                "test_b": TestResult(
                    test_name="test_b", passed=False
                ),  # Should be passing but isn't
                "test_c": TestResult(
                    test_name="test_c", passed=True
                ),  # This one passes, so not an environment issue
            },
        )

        result = validator._validate_baseline(baseline)
        assert result.status == BaselineStatus.INVALID
        assert "PASS_TO_PASS" in result.error_message

    def test_validate_baseline_environment_issue_all_failed(self):
        """Test detecting environment issues when all tests fail."""
        validator = BaselineValidator(use_cache=False)

        baseline = TestBaseline(
            instance_id="test",
            repo="test/repo",
            base_commit="abc",
            fail_to_pass=["test_a"],
            pass_to_pass=["test_b"],
            fail_to_pass_results={
                "test_a": TestResult(test_name="test_a", passed=False),
            },
            pass_to_pass_results={
                "test_b": TestResult(test_name="test_b", passed=False),
            },
        )

        result = validator._validate_baseline(baseline)
        assert result.status == BaselineStatus.ENVIRONMENT_ISSUE
        assert result.environment_valid is False


class TestCheckEnvironmentHealth:
    """Tests for check_environment_health function."""

    def test_health_check_python(self):
        """Test health check for Python environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "pyproject.toml").write_text("[project]\nname = 'test'")

            result = check_environment_health(project_dir, language="python")

            assert "healthy" in result
            assert result["tool_available"] is True  # Python should be available
            assert "pyproject.toml" in result["configs_found"]

    def test_health_check_no_runner(self):
        """Test health check when no runner found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            result = check_environment_health(project_dir, language="unknown_lang")

            assert result["healthy"] is False
            assert "error" in result

    def test_health_check_empty_project(self):
        """Test health check for empty project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            result = check_environment_health(project_dir, language="python")

            # Python tool is available but no configs found
            assert result["tool_available"] is True
            assert result["configs_found"] == []
            assert result["healthy"] is False  # No configs = not healthy
