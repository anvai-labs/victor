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

"""Baseline test validator for SWE-bench evaluation.

This module validates test baselines for SWE-bench tasks:
- FAIL_TO_PASS: Tests expected to fail at base commit, pass after fix
- PASS_TO_PASS: Tests expected to pass both before and after fix

The baseline validator ensures the test environment is correct and
caches baseline results for efficient re-evaluation.
"""

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from victor.evaluation.test_runners import (
    Language,
    TestRunnerRegistry,
    TestResult,
    TestRunResults,
    is_test_tool_available,
)


class TestStatus(Enum):
    """Status of a test execution (internal enum for baseline validation)."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


def get_test_status(result: TestResult) -> TestStatus:
    """Convert TestResult to TestStatus enum.

    Args:
        result: Test result from test runner

    Returns:
        TestStatus enum value
    """
    if result.skip_reason:
        return TestStatus.SKIPPED
    if result.error_message and not result.passed:
        # Has error message and failed - likely an error
        return TestStatus.ERROR
    if result.passed:
        return TestStatus.PASSED
    return TestStatus.FAILED


logger = logging.getLogger(__name__)


class BaselineStatus(Enum):
    """Status of baseline validation."""

    VALID = "valid"  # Baseline matches expectations
    INVALID = "invalid"  # Baseline doesn't match expectations
    ERROR = "error"  # Error during validation
    MISSING = "missing"  # No baseline cached
    ENVIRONMENT_ISSUE = "environment_issue"  # All tests failing


@dataclass
class TestBaseline:
    """Baseline test results for a SWE-bench instance.

    Attributes:
        instance_id: SWE-bench instance identifier
        repo: Repository name (e.g., 'django/django')
        base_commit: Git commit hash for baseline
        fail_to_pass: Tests that should fail at base commit
        pass_to_pass: Tests that should pass at base commit
        fail_to_pass_results: Actual results for FAIL_TO_PASS tests
        pass_to_pass_results: Actual results for PASS_TO_PASS tests
        status: Overall baseline status
        environment_valid: Whether environment is set up correctly
        timestamp: When baseline was captured
        duration_seconds: Time taken to run baseline tests
        error_message: Error message if status is ERROR
    """

    instance_id: str
    repo: str
    base_commit: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    fail_to_pass_results: dict[str, TestResult] = field(default_factory=dict)
    pass_to_pass_results: dict[str, TestResult] = field(default_factory=dict)
    status: BaselineStatus = BaselineStatus.MISSING
    environment_valid: bool = True
    timestamp: Optional[datetime] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if baseline is valid for evaluation."""
        return self.status == BaselineStatus.VALID and self.environment_valid

    def get_validation_summary(self) -> dict:
        """Get summary of baseline validation."""
        fail_to_pass_failing = sum(1 for r in self.fail_to_pass_results.values() if not r.passed)
        pass_to_pass_passing = sum(1 for r in self.pass_to_pass_results.values() if r.passed)

        return {
            "instance_id": self.instance_id,
            "status": self.status.value,
            "environment_valid": self.environment_valid,
            "fail_to_pass": {
                "expected": len(self.fail_to_pass),
                "actually_failing": fail_to_pass_failing,
                "match": fail_to_pass_failing == len(self.fail_to_pass),
            },
            "pass_to_pass": {
                "expected": len(self.pass_to_pass),
                "actually_passing": pass_to_pass_passing,
                "match": pass_to_pass_passing == len(self.pass_to_pass),
            },
            "duration_seconds": self.duration_seconds,
        }

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        # Serialize TestResult objects to dicts
        f2p_results = {}
        for k, v in self.fail_to_pass_results.items():
            f2p_results[k] = {
                "test_name": v.test_name,
                "passed": v.passed,
                "duration_ms": v.duration_ms,
                "error_message": v.error_message,
                "stdout": v.stdout,
                "stderr": v.stderr,
            }

        p2p_results = {}
        for k, v in self.pass_to_pass_results.items():
            p2p_results[k] = {
                "test_name": v.test_name,
                "passed": v.passed,
                "duration_ms": v.duration_ms,
                "error_message": v.error_message,
                "stdout": v.stdout,
                "stderr": v.stderr,
            }

        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "fail_to_pass": self.fail_to_pass,
            "pass_to_pass": self.pass_to_pass,
            "fail_to_pass_results": f2p_results,
            "pass_to_pass_results": p2p_results,
            "status": self.status.value,
            "environment_valid": self.environment_valid,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TestBaseline":
        """Deserialize from dictionary."""
        fail_to_pass_results = {}
        for k, v in data.get("fail_to_pass_results", {}).items():
            fail_to_pass_results[k] = TestResult(
                test_name=v.get("test_name", k),
                passed=v.get("passed", False),
                duration_ms=v.get("duration_ms", 0.0),
                stdout=v.get("stdout", ""),
                stderr=v.get("stderr", ""),
                error_message=v.get("error_message", ""),
            )

        pass_to_pass_results = {}
        for k, v in data.get("pass_to_pass_results", {}).items():
            pass_to_pass_results[k] = TestResult(
                test_name=v.get("test_name", k),
                passed=v.get("passed", True),
                duration_ms=v.get("duration_ms", 0.0),
                stdout=v.get("stdout", ""),
                stderr=v.get("stderr", ""),
                error_message=v.get("error_message", ""),
            )

        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            instance_id=data["instance_id"],
            repo=data["repo"],
            base_commit=data["base_commit"],
            fail_to_pass=data.get("fail_to_pass", []),
            pass_to_pass=data.get("pass_to_pass", []),
            fail_to_pass_results=fail_to_pass_results,
            pass_to_pass_results=pass_to_pass_results,
            status=BaselineStatus(data.get("status", "missing")),
            environment_valid=data.get("environment_valid", True),
            timestamp=timestamp,
            duration_seconds=data.get("duration_seconds", 0.0),
            error_message=data.get("error_message"),
        )


@dataclass
class BaselineValidationResult:
    """Result of validating agent changes against baseline.

    Attributes:
        instance_id: SWE-bench instance identifier
        baseline: Original baseline
        post_change_results: Test results after agent changes
        fail_to_pass_fixed: Tests that now pass (were failing at baseline)
        pass_to_pass_broken: Tests that now fail (were passing at baseline)
        success: Whether validation passed (all F2P fixed, no P2P broken)
        partial_success: Whether some F2P tests were fixed
        score: Normalized score (0-1)
    """

    instance_id: str
    baseline: TestBaseline
    post_change_results: TestRunResults
    fail_to_pass_fixed: list[str] = field(default_factory=list)
    pass_to_pass_broken: list[str] = field(default_factory=list)
    success: bool = False
    partial_success: bool = False
    score: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "instance_id": self.instance_id,
            "fail_to_pass_fixed": self.fail_to_pass_fixed,
            "pass_to_pass_broken": self.pass_to_pass_broken,
            "success": self.success,
            "partial_success": self.partial_success,
            "score": self.score,
            "total_fail_to_pass": len(self.baseline.fail_to_pass),
            "total_pass_to_pass": len(self.baseline.pass_to_pass),
        }


class BaselineCache:
    """SQLite-based cache for baseline test results.

    Caches baselines by (repo, commit) to avoid re-running tests
    when evaluating multiple instances from the same repo/commit.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize baseline cache.

        Args:
            cache_dir: Directory for cache database. Defaults to ~/.victor/baseline_cache
        """
        if cache_dir is None:
            try:
                from victor.config.secure_paths import get_victor_dir

                cache_dir = get_victor_dir() / "baseline_cache"
            except ImportError:
                cache_dir = Path.home() / ".victor" / "baseline_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "baselines.db"

        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    cache_key TEXT PRIMARY KEY,
                    instance_id TEXT NOT NULL,
                    repo TEXT NOT NULL,
                    base_commit TEXT NOT NULL,
                    data JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_repo_commit
                ON baselines(repo, base_commit)
            """)

    def _get_cache_key(self, instance_id: str, repo: str, commit: str) -> str:
        """Generate cache key for baseline."""
        key_data = f"{instance_id}:{repo}:{commit}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def get(self, instance_id: str, repo: str, commit: str) -> Optional[TestBaseline]:
        """Get cached baseline if available.

        Args:
            instance_id: SWE-bench instance ID
            repo: Repository name
            commit: Git commit hash

        Returns:
            Cached baseline or None if not found
        """
        cache_key = self._get_cache_key(instance_id, repo, commit)

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT data FROM baselines
                WHERE cache_key = ?
                AND (expires_at IS NULL OR expires_at > datetime('now'))
                """,
                (cache_key,),
            ).fetchone()

            if row:
                data = json.loads(row[0])
                return TestBaseline.from_dict(data)

        return None

    def set(
        self,
        baseline: TestBaseline,
        ttl_hours: Optional[int] = None,
    ):
        """Cache baseline result.

        Args:
            baseline: Baseline to cache
            ttl_hours: Cache TTL in hours (None for no expiration)
        """
        cache_key = self._get_cache_key(
            baseline.instance_id,
            baseline.repo,
            baseline.base_commit,
        )

        expires_at = None
        if ttl_hours:
            from datetime import timedelta

            expires_at = datetime.now() + timedelta(hours=ttl_hours)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO baselines
                (cache_key, instance_id, repo, base_commit, data, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    baseline.instance_id,
                    baseline.repo,
                    baseline.base_commit,
                    json.dumps(baseline.to_dict()),
                    expires_at.isoformat() if expires_at else None,
                ),
            )

    def invalidate(self, repo: str, commit: str):
        """Invalidate all cached baselines for a repo/commit.

        Args:
            repo: Repository name
            commit: Git commit hash
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM baselines WHERE repo = ? AND base_commit = ?",
                (repo, commit),
            )

    def clear(self):
        """Clear all cached baselines."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM baselines")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM baselines").fetchone()[0]
            by_repo = dict(
                conn.execute("SELECT repo, COUNT(*) FROM baselines GROUP BY repo").fetchall()
            )
            valid = conn.execute("""
                SELECT COUNT(*) FROM baselines
                WHERE expires_at IS NULL OR expires_at > datetime('now')
                """).fetchone()[0]

            return {
                "total": total,
                "valid": valid,
                "expired": total - valid,
                "by_repo": by_repo,
            }


class BaselineValidator:
    """Validates test baselines for SWE-bench evaluation.

    This class:
    1. Establishes test baselines at base commits
    2. Validates FAIL_TO_PASS tests are actually failing
    3. Validates PASS_TO_PASS tests are actually passing
    4. Detects environment issues
    5. Compares post-change results against baselines
    """

    def __init__(
        self,
        test_registry: Optional[TestRunnerRegistry] = None,
        cache: Optional[BaselineCache] = None,
        use_cache: bool = True,
        timeout_per_test: int = 60,
        parallel_tests: int = 4,
    ):
        """Initialize baseline validator.

        Args:
            test_registry: Test runner registry
            cache: Baseline cache
            use_cache: Whether to use caching
            timeout_per_test: Timeout per test in seconds
            parallel_tests: Number of parallel test workers
        """
        self.test_registry = test_registry or TestRunnerRegistry()
        self.cache = cache if use_cache else None
        self.use_cache = use_cache
        self.timeout_per_test = timeout_per_test
        self.parallel_tests = parallel_tests

        if use_cache and cache is None:
            self.cache = BaselineCache()

    async def establish_baseline(
        self,
        instance_id: str,
        repo: str,
        base_commit: str,
        workspace_dir: Path,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        language: str = "python",
        force_refresh: bool = False,
    ) -> TestBaseline:
        """Establish baseline test results for an instance.

        Args:
            instance_id: SWE-bench instance identifier
            repo: Repository name
            base_commit: Git commit hash
            workspace_dir: Directory with checked-out repo at base commit
            fail_to_pass: Tests expected to fail
            pass_to_pass: Tests expected to pass
            language: Programming language
            force_refresh: Force re-running tests even if cached

        Returns:
            TestBaseline with validation results
        """
        # Check cache first
        if self.use_cache and self.cache and not force_refresh:
            cached = self.cache.get(instance_id, repo, base_commit)
            if cached:
                logger.info(f"Using cached baseline for {instance_id}")
                return cached

        logger.info(f"Establishing baseline for {instance_id} at {base_commit[:8]}")

        baseline = TestBaseline(
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            timestamp=datetime.now(),
        )

        start_time = time.time()

        try:
            # Get appropriate test runner
            runner = self.test_registry.get_runner(language)
            if not runner:
                baseline.status = BaselineStatus.ERROR
                baseline.error_message = f"No test runner for language: {language}"
                return baseline

            # Run FAIL_TO_PASS tests (expected to fail)
            if fail_to_pass:
                logger.debug(f"Running {len(fail_to_pass)} FAIL_TO_PASS tests")
                f2p_results = await runner.run_tests(
                    workspace_dir,
                    fail_to_pass,
                )
                for result in f2p_results.results:
                    baseline.fail_to_pass_results[result.test_name] = result

            # Run PASS_TO_PASS tests (expected to pass)
            if pass_to_pass:
                logger.debug(f"Running {len(pass_to_pass)} PASS_TO_PASS tests")
                p2p_results = await runner.run_tests(
                    workspace_dir,
                    pass_to_pass,
                )
                for result in p2p_results.results:
                    baseline.pass_to_pass_results[result.test_name] = result

            # Validate baseline
            baseline = self._validate_baseline(baseline)

            baseline.duration_seconds = time.time() - start_time

            # Cache valid baselines
            if self.use_cache and self.cache and baseline.status == BaselineStatus.VALID:
                self.cache.set(baseline)

        except Exception as e:
            logger.error(f"Error establishing baseline: {e}")
            baseline.status = BaselineStatus.ERROR
            baseline.error_message = str(e)
            baseline.duration_seconds = time.time() - start_time

        return baseline

    def _validate_baseline(self, baseline: TestBaseline) -> TestBaseline:
        """Validate baseline results match expectations.

        Args:
            baseline: Baseline to validate

        Returns:
            Updated baseline with status
        """
        # Check FAIL_TO_PASS tests actually fail
        f2p_failing = 0
        f2p_total = len(baseline.fail_to_pass)

        for test_id in baseline.fail_to_pass:
            result = baseline.fail_to_pass_results.get(test_id)
            if result and not result.passed:
                f2p_failing += 1

        # Check PASS_TO_PASS tests actually pass
        p2p_passing = 0
        p2p_total = len(baseline.pass_to_pass)

        for test_id in baseline.pass_to_pass:
            result = baseline.pass_to_pass_results.get(test_id)
            if result and result.passed:
                p2p_passing += 1

        # Detect environment issues
        all_tests = list(baseline.fail_to_pass_results.values()) + list(
            baseline.pass_to_pass_results.values()
        )

        if all_tests:
            all_failed = all(not r.passed for r in all_tests)
            all_errored = all(r.error_message and not r.passed for r in all_tests)

            if all_errored:
                baseline.status = BaselineStatus.ENVIRONMENT_ISSUE
                baseline.environment_valid = False
                baseline.error_message = "All tests errored - likely environment issue"
                return baseline

            if all_failed and p2p_total > 0:
                # All tests failing including PASS_TO_PASS suggests env issue
                baseline.status = BaselineStatus.ENVIRONMENT_ISSUE
                baseline.environment_valid = False
                baseline.error_message = "All tests failing - likely environment issue"
                return baseline

        # Check if baseline matches expectations
        f2p_valid = f2p_failing == f2p_total or f2p_total == 0
        p2p_valid = p2p_passing == p2p_total or p2p_total == 0

        if f2p_valid and p2p_valid:
            baseline.status = BaselineStatus.VALID
        else:
            baseline.status = BaselineStatus.INVALID
            issues = []
            if not f2p_valid:
                issues.append(f"FAIL_TO_PASS: {f2p_failing}/{f2p_total} failing")
            if not p2p_valid:
                issues.append(f"PASS_TO_PASS: {p2p_passing}/{p2p_total} passing")
            baseline.error_message = "; ".join(issues)

        return baseline

    async def validate_changes(
        self,
        baseline: TestBaseline,
        workspace_dir: Path,
        language: str = "python",
    ) -> BaselineValidationResult:
        """Validate agent changes against baseline.

        Args:
            baseline: Established baseline
            workspace_dir: Directory with agent's changes
            language: Programming language

        Returns:
            Validation result comparing post-change to baseline
        """
        result = BaselineValidationResult(
            instance_id=baseline.instance_id,
            baseline=baseline,
            post_change_results=TestRunResults(
                language=language,
                total=0,
                passed=0,
                failed=0,
                errors=0,
                duration_seconds=0.0,
            ),
        )

        runner = self.test_registry.get_runner(language)
        if not runner:
            logger.error(f"No test runner for language: {language}")
            return result

        # Run all tests
        all_tests = baseline.fail_to_pass + baseline.pass_to_pass
        if not all_tests:
            result.success = True
            result.score = 1.0
            return result

        post_results = await runner.run_tests(
            workspace_dir,
            all_tests,
            timeout=self.timeout_per_test * len(all_tests),
        )
        result.post_change_results = post_results

        # Map results by test name
        post_map = {r.test_name: r for r in post_results.results}

        # Check FAIL_TO_PASS tests are now passing
        for test_id in baseline.fail_to_pass:
            post_result = post_map.get(test_id)
            if post_result and post_result.passed:
                result.fail_to_pass_fixed.append(test_id)

        # Check PASS_TO_PASS tests are still passing
        for test_id in baseline.pass_to_pass:
            baseline_result = baseline.pass_to_pass_results.get(test_id)
            post_result = post_map.get(test_id)

            # Only count as broken if it was passing at baseline and now fails
            if baseline_result and baseline_result.passed:
                if post_result and not post_result.passed:
                    result.pass_to_pass_broken.append(test_id)

        # Calculate success criteria
        f2p_total = len(baseline.fail_to_pass)
        f2p_fixed = len(result.fail_to_pass_fixed)
        p2p_broken = len(result.pass_to_pass_broken)

        # Full success: all F2P fixed, no P2P broken
        result.success = (f2p_fixed == f2p_total) and (p2p_broken == 0)

        # Partial success: at least some F2P fixed
        result.partial_success = f2p_fixed > 0

        # Score calculation
        if f2p_total > 0:
            f2p_score = f2p_fixed / f2p_total
        else:
            f2p_score = 1.0

        # Penalty for breaking P2P tests
        p2p_total = len(baseline.pass_to_pass)
        if p2p_total > 0:
            p2p_penalty = p2p_broken / p2p_total
        else:
            p2p_penalty = 0.0

        # Final score: F2P success minus P2P breakage penalty
        result.score = max(0.0, f2p_score - p2p_penalty)

        return result


async def quick_validate_baseline(
    workspace_dir: Path,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    language: str = "python",
    timeout: int = 300,
) -> TestBaseline:
    """Quick utility to validate baseline without full setup.

    Args:
        workspace_dir: Directory with repository
        fail_to_pass: Tests expected to fail
        pass_to_pass: Tests expected to pass
        language: Programming language
        timeout: Total timeout in seconds

    Returns:
        TestBaseline with validation results
    """
    validator = BaselineValidator(use_cache=False)

    return await validator.establish_baseline(
        instance_id="quick_validate",
        repo="unknown",
        base_commit="unknown",
        workspace_dir=workspace_dir,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        language=language,
    )


def check_environment_health(
    workspace_dir: Path,
    language: str = "python",
) -> dict:
    """Check if test environment is healthy.

    Runs a simple smoke test to verify:
    - Test runner is available
    - Basic imports work
    - Project can be discovered

    Args:
        workspace_dir: Directory to check
        language: Programming language

    Returns:
        Health check results
    """
    # Map string language to Language enum
    lang_map = {
        "python": Language.PYTHON,
        "javascript": Language.JAVASCRIPT,
        "typescript": Language.TYPESCRIPT,
        "go": Language.GO,
        "rust": Language.RUST,
        "java": Language.JAVA,
    }
    lang_enum = lang_map.get(language, Language.UNKNOWN)

    registry = TestRunnerRegistry()
    runner = registry.get_runner(lang_enum)

    if not runner:
        return {
            "healthy": False,
            "error": f"No test runner for {language}",
        }

    # Check if test tool is available
    tool_available = is_test_tool_available(lang_enum)

    # Check for test configuration files
    test_configs = {
        "python": ["pytest.ini", "setup.py", "pyproject.toml", "tox.ini"],
        "javascript": ["package.json", "jest.config.js", "mocha.opts"],
        "go": ["go.mod", "go.sum"],
        "rust": ["Cargo.toml"],
        "java": ["pom.xml", "build.gradle"],
    }

    configs_found = []
    for config_file in test_configs.get(language, []):
        if (workspace_dir / config_file).exists():
            configs_found.append(config_file)

    return {
        "healthy": tool_available and len(configs_found) > 0,
        "tool_available": tool_available,
        "configs_found": configs_found,
        "workspace": str(workspace_dir),
        "language": language,
    }
