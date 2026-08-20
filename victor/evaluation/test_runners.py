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

"""Multi-language test runner registry for agentic benchmarks.

This module provides test runner implementations for various programming languages,
enabling SWE-bench style evaluations across Python, JavaScript, Go, Rust, and Java
projects.

Example usage:
    from victor.evaluation.test_runners import (
        TestRunnerRegistry,
        TestResult,
        TestRunnerConfig,
    )

    # Auto-detect and run tests
    registry = TestRunnerRegistry()
    runner = registry.detect_runner(project_dir)

    if runner:
        results = await runner.run_tests(
            project_dir,
            test_filter=["test_specific_function"],
        )
        for result in results:
            print(f"{result.test_name}: {'PASS' if result.passed else 'FAIL'}")
"""

import asyncio
import json
import logging
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    UNKNOWN = "unknown"


class TestExpectedStatus(str, Enum):
    """Expected outcome for a test result."""

    PASS = "pass"  # Test is expected to pass
    FAIL = "fail"  # Test is expected to fail
    NONE = ""  # No expectation (any outcome is acceptable)


@dataclass
class TestRunnerConfig:
    """Configuration for test runner execution."""

    language: Language = Language.UNKNOWN
    test_command: str = ""
    timeout_seconds: int = 300
    env_vars: dict[str, str] = field(default_factory=dict)
    working_dir: Optional[Path] = None
    collect_coverage: bool = False
    verbose: bool = False


@dataclass
class TestResult:
    """Result of a single test execution."""

    test_name: str
    passed: bool
    duration_ms: float = 0.0
    stdout: str = ""
    stderr: str = ""
    expected_status: TestExpectedStatus = TestExpectedStatus.NONE
    error_message: str = ""
    skip_reason: str = ""

    @property
    def is_expected(self) -> bool:
        """Check if test result matches expected status."""
        if self.expected_status == TestExpectedStatus.NONE:
            return True
        if self.expected_status == TestExpectedStatus.FAIL:
            return not self.passed
        return self.passed


@dataclass
class TestRunResults:
    """Aggregated results from running a test suite."""

    results: list[TestResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    test_command: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error_message: str = ""

    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0 and self.errors == 0

    def filter_by_names(self, test_names: list[str]) -> "TestRunResults":
        """Filter results to only include specified test names."""
        filtered = [r for r in self.results if r.test_name in test_names]
        return TestRunResults(
            results=filtered,
            total=len(filtered),
            passed=sum(1 for r in filtered if r.passed),
            failed=sum(1 for r in filtered if not r.passed and not r.skip_reason),
            skipped=sum(1 for r in filtered if r.skip_reason),
            duration_seconds=self.duration_seconds,
            test_command=self.test_command,
        )


class BaseTestRunner(ABC):
    """Abstract base class for language-specific test runners."""

    def __init__(self, config: Optional[TestRunnerConfig] = None):
        self.config = config or TestRunnerConfig()

    @property
    @abstractmethod
    def language(self) -> Language:
        """Return the language this runner handles."""
        pass

    @abstractmethod
    def detect(self, project_dir: Path) -> bool:
        """Check if this runner is appropriate for the project."""
        pass

    @abstractmethod
    def detect_test_command(self, project_dir: Path) -> str:
        """Auto-detect the test command for the project."""
        pass

    @abstractmethod
    async def run_tests(
        self,
        project_dir: Path,
        test_filter: Optional[list[str]] = None,
        config: Optional[TestRunnerConfig] = None,
    ) -> TestRunResults:
        """Run tests and return results."""
        pass

    @abstractmethod
    def parse_test_output(self, stdout: str, stderr: str) -> TestRunResults:
        """Parse test output to extract individual test results."""
        pass

    async def _run_command(
        self,
        command: str,
        cwd: Path,
        timeout: int = 300,
        env: Optional[dict[str, str]] = None,
    ) -> tuple[int, str, str]:
        """Run a shell command and return (exit_code, stdout, stderr)."""
        import os

        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            return (
                process.returncode or 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except Exception:
                pass
            return -1, "", f"Test execution timed out after {timeout}s"
        except Exception as e:
            return -1, "", str(e)


class PythonTestRunner(BaseTestRunner):
    """Test runner for Python projects using pytest or unittest."""

    @property
    def language(self) -> Language:
        return Language.PYTHON

    def detect(self, project_dir: Path) -> bool:
        """Check for Python project indicators."""
        indicators = [
            "setup.py",
            "setup.cfg",
            "pyproject.toml",
            "requirements.txt",
            "pytest.ini",
            "tox.ini",
            "conftest.py",
        ]
        return any((project_dir / f).exists() for f in indicators)

    def detect_test_command(self, project_dir: Path) -> str:
        """Detect pytest vs unittest preference."""
        # Check for pytest configuration
        pytest_indicators = [
            "pytest.ini",
            "pyproject.toml",  # may have [tool.pytest]
            "conftest.py",
        ]

        for indicator in pytest_indicators:
            path = project_dir / indicator
            if path.exists():
                if indicator == "pyproject.toml":
                    content = path.read_text()
                    if "[tool.pytest" in content or "pytest" in content:
                        return "python -m pytest"
                else:
                    return "python -m pytest"

        # Check for tests directory structure
        test_dirs = ["tests", "test", "testing"]
        for test_dir in test_dirs:
            if (project_dir / test_dir).exists():
                return "python -m pytest"

        # Default to unittest
        return "python -m unittest discover"

    async def run_tests(
        self,
        project_dir: Path,
        test_filter: Optional[list[str]] = None,
        config: Optional[TestRunnerConfig] = None,
    ) -> TestRunResults:
        """Run Python tests with pytest or unittest."""
        cfg = config or self.config
        test_cmd = cfg.test_command or self.detect_test_command(project_dir)

        # Build command with filters
        if "pytest" in test_cmd:
            if test_filter:
                # Use pytest -k for keyword filtering
                filter_expr = " or ".join(test_filter)
                test_cmd = f"{test_cmd} -v --tb=short -k '{filter_expr}'"
            else:
                test_cmd = f"{test_cmd} -v --tb=short"

            # Add JSON report for structured output
            test_cmd = f"{test_cmd} --json-report --json-report-file=/dev/stdout 2>&1 || true"
        elif "unittest" in test_cmd:
            if test_filter:
                # Run specific test classes/methods
                test_cmd = f"python -m unittest {' '.join(test_filter)} -v"
            else:
                test_cmd = f"{test_cmd} -v"

        logger.info(f"Running: {test_cmd} in {project_dir}")

        exit_code, stdout, stderr = await self._run_command(
            test_cmd,
            project_dir,
            timeout=cfg.timeout_seconds,
            env=cfg.env_vars,
        )

        results = self.parse_test_output(stdout, stderr)
        results.test_command = test_cmd
        results.exit_code = exit_code
        results.stdout = stdout
        results.stderr = stderr

        return results

    def parse_test_output(self, stdout: str, stderr: str) -> TestRunResults:
        """Parse pytest or unittest output."""
        results = TestRunResults()

        # Try to parse JSON report first (pytest-json-report)
        try:
            # Find JSON in output
            json_match = re.search(r'\{["\']tests["\']\s*:', stdout)
            if json_match:
                json_start = json_match.start()
                json_end = stdout.rfind("}") + 1
                if json_end > json_start:
                    report = json.loads(stdout[json_start:json_end])
                    return self._parse_pytest_json(report)
        except (json.JSONDecodeError, KeyError):
            pass

        # Fall back to regex parsing
        # Pytest output: test_file.py::test_name PASSED/FAILED
        pytest_pattern = r"(\S+::\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)"
        matches = re.findall(pytest_pattern, stdout)

        for test_name, status in matches:
            passed = status == "PASSED"
            skipped = status == "SKIPPED"
            result = TestResult(
                test_name=test_name,
                passed=passed,
                skip_reason="skipped" if skipped else "",
            )
            results.results.append(result)
            results.total += 1
            if passed:
                results.passed += 1
            elif skipped:
                results.skipped += 1
            else:
                results.failed += 1

        # Parse unittest output: test_name (test_module.TestClass) ... ok/FAIL
        unittest_pattern = r"(\w+)\s+\(([^)]+)\)\s+\.\.\.\s+(ok|FAIL|ERROR|skipped)"
        matches = re.findall(unittest_pattern, stdout)

        for method_name, class_path, status in matches:
            full_name = f"{class_path}::{method_name}"
            passed = status == "ok"
            skipped = status == "skipped"
            result = TestResult(
                test_name=full_name,
                passed=passed,
                skip_reason="skipped" if skipped else "",
            )
            results.results.append(result)
            results.total += 1
            if passed:
                results.passed += 1
            elif skipped:
                results.skipped += 1
            else:
                results.failed += 1

        # Parse summary line: X passed, Y failed, Z skipped
        summary_pattern = r"(\d+)\s+(passed|failed|skipped|error)"
        for count, status in re.findall(summary_pattern, stdout.lower()):
            count = int(count)
            if status == "passed" and results.passed == 0:
                results.passed = count
            elif status == "failed" and results.failed == 0:
                results.failed = count
            elif status == "skipped" and results.skipped == 0:
                results.skipped = count
            elif status == "error" and results.errors == 0:
                results.errors = count

        if results.total == 0:
            results.total = results.passed + results.failed + results.skipped + results.errors

        return results

    def _parse_pytest_json(self, report: dict[str, Any]) -> TestRunResults:
        """Parse pytest-json-report output."""
        results = TestRunResults()

        tests = report.get("tests", [])
        for test in tests:
            outcome = test.get("outcome", "")
            result = TestResult(
                test_name=test.get("nodeid", ""),
                passed=outcome == "passed",
                duration_ms=test.get("duration", 0) * 1000,
                skip_reason=test.get("longrepr", "") if outcome == "skipped" else "",
            )
            results.results.append(result)

        summary = report.get("summary", {})
        results.total = summary.get("total", len(tests))
        results.passed = summary.get("passed", 0)
        results.failed = summary.get("failed", 0)
        results.skipped = summary.get("skipped", 0)
        results.errors = summary.get("error", 0)
        results.duration_seconds = report.get("duration", 0)

        return results


class JavaScriptTestRunner(BaseTestRunner):
    """Test runner for JavaScript/TypeScript projects."""

    @property
    def language(self) -> Language:
        return Language.JAVASCRIPT

    def detect(self, project_dir: Path) -> bool:
        """Check for JavaScript/Node.js project indicators."""
        return (project_dir / "package.json").exists()

    def detect_test_command(self, project_dir: Path) -> str:
        """Detect npm/yarn test command from package.json."""
        package_json = project_dir / "package.json"
        if not package_json.exists():
            return "npm test"

        try:
            with open(package_json) as f:
                pkg = json.load(f)

            scripts = pkg.get("scripts", {})

            # Check for test script
            if "test" in scripts:
                test_cmd = scripts["test"]
                # Detect test framework
                if "jest" in test_cmd:
                    return "npm test -- --json"
                elif "mocha" in test_cmd:
                    return "npm test -- --reporter json"
                elif "vitest" in test_cmd:
                    return "npm test -- --reporter json"
                return "npm test"

            # Check devDependencies for test frameworks
            dev_deps = pkg.get("devDependencies", {})
            if "jest" in dev_deps:
                return "npx jest --json"
            elif "mocha" in dev_deps:
                return "npx mocha --reporter json"
            elif "vitest" in dev_deps:
                return "npx vitest run --reporter json"

        except (json.JSONDecodeError, KeyError):
            pass

        return "npm test"

    async def run_tests(
        self,
        project_dir: Path,
        test_filter: Optional[list[str]] = None,
        config: Optional[TestRunnerConfig] = None,
    ) -> TestRunResults:
        """Run JavaScript tests."""
        cfg = config or self.config
        test_cmd = cfg.test_command or self.detect_test_command(project_dir)

        # Add test filter if specified
        if test_filter:
            if "jest" in test_cmd:
                filter_pattern = "|".join(test_filter)
                test_cmd = f"{test_cmd} --testNamePattern='{filter_pattern}'"
            elif "mocha" in test_cmd:
                filter_pattern = "|".join(test_filter)
                test_cmd = f"{test_cmd} --grep '{filter_pattern}'"

        logger.info(f"Running: {test_cmd} in {project_dir}")

        exit_code, stdout, stderr = await self._run_command(
            test_cmd,
            project_dir,
            timeout=cfg.timeout_seconds,
            env=cfg.env_vars,
        )

        results = self.parse_test_output(stdout, stderr)
        results.test_command = test_cmd
        results.exit_code = exit_code
        results.stdout = stdout
        results.stderr = stderr

        return results

    def parse_test_output(self, stdout: str, stderr: str) -> TestRunResults:
        """Parse Jest/Mocha output."""
        results = TestRunResults()

        # Try to parse Jest JSON output
        try:
            report = json.loads(stdout)
            if "testResults" in report:
                return self._parse_jest_json(report)
        except json.JSONDecodeError:
            pass

        # Try to parse Mocha JSON output
        try:
            report = json.loads(stdout)
            if "stats" in report and "tests" in report:
                return self._parse_mocha_json(report)
        except json.JSONDecodeError:
            pass

        # Fall back to regex parsing
        # Jest: ✓ test name (XXms)
        jest_pass = re.findall(r"[✓✔]\s+(.+?)\s+\((\d+)\s*m?s\)", stdout)
        jest_fail = re.findall(r"[✗✘×]\s+(.+)", stdout)

        for test_name, duration in jest_pass:
            results.results.append(
                TestResult(
                    test_name=test_name.strip(),
                    passed=True,
                    duration_ms=float(duration),
                )
            )
            results.passed += 1

        for test_name in jest_fail:
            results.results.append(
                TestResult(
                    test_name=test_name.strip(),
                    passed=False,
                )
            )
            results.failed += 1

        results.total = results.passed + results.failed
        return results

    def _parse_jest_json(self, report: dict[str, Any]) -> TestRunResults:
        """Parse Jest JSON output."""
        results = TestRunResults()

        for test_file in report.get("testResults", []):
            for assertion in test_file.get("assertionResults", []):
                status = assertion.get("status", "")
                results.results.append(
                    TestResult(
                        test_name=assertion.get("fullName", ""),
                        passed=status == "passed",
                        duration_ms=assertion.get("duration", 0),
                        skip_reason="skipped" if status == "skipped" else "",
                    )
                )

        results.total = report.get("numTotalTests", 0)
        results.passed = report.get("numPassedTests", 0)
        results.failed = report.get("numFailedTests", 0)
        results.skipped = report.get("numPendingTests", 0)

        return results

    def _parse_mocha_json(self, report: dict[str, Any]) -> TestRunResults:
        """Parse Mocha JSON output."""
        results = TestRunResults()

        for test in report.get("tests", []):
            results.results.append(
                TestResult(
                    test_name=test.get("fullTitle", ""),
                    passed=test.get("err") is None,
                    duration_ms=test.get("duration", 0),
                )
            )

        stats = report.get("stats", {})
        results.total = stats.get("tests", 0)
        results.passed = stats.get("passes", 0)
        results.failed = stats.get("failures", 0)
        results.skipped = stats.get("pending", 0)
        results.duration_seconds = stats.get("duration", 0) / 1000

        return results


class GoTestRunner(BaseTestRunner):
    """Test runner for Go projects."""

    @property
    def language(self) -> Language:
        return Language.GO

    def detect(self, project_dir: Path) -> bool:
        """Check for Go project indicators."""
        indicators = ["go.mod", "go.sum"]
        return any((project_dir / f).exists() for f in indicators)

    def detect_test_command(self, project_dir: Path) -> str:
        """Return go test command with JSON output."""
        return "go test -json ./..."

    async def run_tests(
        self,
        project_dir: Path,
        test_filter: Optional[list[str]] = None,
        config: Optional[TestRunnerConfig] = None,
    ) -> TestRunResults:
        """Run Go tests."""
        cfg = config or self.config
        test_cmd = cfg.test_command or self.detect_test_command(project_dir)

        # Add test filter
        if test_filter:
            filter_pattern = "|".join(test_filter)
            test_cmd = f"{test_cmd} -run '{filter_pattern}'"

        logger.info(f"Running: {test_cmd} in {project_dir}")

        exit_code, stdout, stderr = await self._run_command(
            test_cmd,
            project_dir,
            timeout=cfg.timeout_seconds,
            env=cfg.env_vars,
        )

        results = self.parse_test_output(stdout, stderr)
        results.test_command = test_cmd
        results.exit_code = exit_code
        results.stdout = stdout
        results.stderr = stderr

        return results

    def parse_test_output(self, stdout: str, stderr: str) -> TestRunResults:
        """Parse go test -json output."""
        results = TestRunResults()

        # Parse JSON lines
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            try:
                event = json.loads(line)
                action = event.get("Action", "")
                test_name = event.get("Test", "")

                if action == "pass" and test_name:
                    results.results.append(
                        TestResult(
                            test_name=test_name,
                            passed=True,
                            duration_ms=event.get("Elapsed", 0) * 1000,
                        )
                    )
                    results.passed += 1
                elif action == "fail" and test_name:
                    results.results.append(
                        TestResult(
                            test_name=test_name,
                            passed=False,
                            duration_ms=event.get("Elapsed", 0) * 1000,
                        )
                    )
                    results.failed += 1
                elif action == "skip" and test_name:
                    results.results.append(
                        TestResult(
                            test_name=test_name,
                            passed=True,
                            skip_reason="skipped",
                        )
                    )
                    results.skipped += 1

            except json.JSONDecodeError:
                continue

        results.total = results.passed + results.failed + results.skipped
        return results


class RustTestRunner(BaseTestRunner):
    """Test runner for Rust projects."""

    @property
    def language(self) -> Language:
        return Language.RUST

    def detect(self, project_dir: Path) -> bool:
        """Check for Rust project indicators."""
        return (project_dir / "Cargo.toml").exists()

    def detect_test_command(self, project_dir: Path) -> str:
        """Return cargo test command."""
        return "cargo test -- --format=json -Z unstable-options"

    async def run_tests(
        self,
        project_dir: Path,
        test_filter: Optional[list[str]] = None,
        config: Optional[TestRunnerConfig] = None,
    ) -> TestRunResults:
        """Run Rust tests."""
        cfg = config or self.config

        # Use basic cargo test if JSON format not available
        test_cmd = "cargo test"
        if test_filter:
            test_cmd = f"{test_cmd} {' '.join(test_filter)}"
        test_cmd = f"{test_cmd} -- --nocapture"

        logger.info(f"Running: {test_cmd} in {project_dir}")

        exit_code, stdout, stderr = await self._run_command(
            test_cmd,
            project_dir,
            timeout=cfg.timeout_seconds,
            env=cfg.env_vars,
        )

        results = self.parse_test_output(stdout, stderr)
        results.test_command = test_cmd
        results.exit_code = exit_code
        results.stdout = stdout
        results.stderr = stderr

        return results

    def parse_test_output(self, stdout: str, stderr: str) -> TestRunResults:
        """Parse cargo test output."""
        results = TestRunResults()
        combined = stdout + stderr

        # Parse: test test_name ... ok/FAILED
        test_pattern = r"test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)"
        matches = re.findall(test_pattern, combined)

        for test_name, status in matches:
            passed = status == "ok"
            skipped = status == "ignored"
            results.results.append(
                TestResult(
                    test_name=test_name,
                    passed=passed,
                    skip_reason="ignored" if skipped else "",
                )
            )
            if passed:
                results.passed += 1
            elif skipped:
                results.skipped += 1
            else:
                results.failed += 1

        # Parse summary: test result: ok. X passed; Y failed; Z ignored
        summary_pattern = r"test result:.*?(\d+)\s+passed.*?(\d+)\s+failed.*?(\d+)\s+ignored"
        summary_match = re.search(summary_pattern, combined)
        if summary_match:
            results.passed = int(summary_match.group(1))
            results.failed = int(summary_match.group(2))
            results.skipped = int(summary_match.group(3))

        results.total = results.passed + results.failed + results.skipped
        return results


class JavaTestRunner(BaseTestRunner):
    """Test runner for Java projects using Maven or Gradle."""

    @property
    def language(self) -> Language:
        return Language.JAVA

    def detect(self, project_dir: Path) -> bool:
        """Check for Java project indicators."""
        maven = (project_dir / "pom.xml").exists()
        gradle = (project_dir / "build.gradle").exists() or (
            project_dir / "build.gradle.kts"
        ).exists()
        return maven or gradle

    def detect_test_command(self, project_dir: Path) -> str:
        """Detect Maven vs Gradle."""
        if (project_dir / "pom.xml").exists():
            return "mvn test -B"
        elif (project_dir / "build.gradle").exists() or (project_dir / "build.gradle.kts").exists():
            return "./gradlew test"
        return "mvn test -B"

    async def run_tests(
        self,
        project_dir: Path,
        test_filter: Optional[list[str]] = None,
        config: Optional[TestRunnerConfig] = None,
    ) -> TestRunResults:
        """Run Java tests."""
        cfg = config or self.config
        test_cmd = cfg.test_command or self.detect_test_command(project_dir)

        # Add test filter
        if test_filter:
            if "mvn" in test_cmd:
                filter_pattern = ",".join(test_filter)
                test_cmd = f"{test_cmd} -Dtest={filter_pattern}"
            elif "gradle" in test_cmd:
                filter_pattern = " --tests ".join(test_filter)
                test_cmd = f"{test_cmd} --tests {filter_pattern}"

        logger.info(f"Running: {test_cmd} in {project_dir}")

        exit_code, stdout, stderr = await self._run_command(
            test_cmd,
            project_dir,
            timeout=cfg.timeout_seconds,
            env=cfg.env_vars,
        )

        results = self.parse_test_output(stdout, stderr)
        results.test_command = test_cmd
        results.exit_code = exit_code
        results.stdout = stdout
        results.stderr = stderr

        return results

    def parse_test_output(self, stdout: str, stderr: str) -> TestRunResults:
        """Parse Maven/Gradle test output."""
        results = TestRunResults()
        combined = stdout + stderr

        # Maven surefire pattern: Tests run: X, Failures: Y, Errors: Z, Skipped: W
        maven_pattern = (
            r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)"
        )
        maven_match = re.search(maven_pattern, combined)

        if maven_match:
            results.total = int(maven_match.group(1))
            results.failed = int(maven_match.group(2))
            results.errors = int(maven_match.group(3))
            results.skipped = int(maven_match.group(4))
            results.passed = results.total - results.failed - results.errors - results.skipped

        # Gradle pattern: X tests completed, Y failed
        gradle_pattern = r"(\d+)\s+tests?\s+completed,\s+(\d+)\s+failed"
        gradle_match = re.search(gradle_pattern, combined)

        if gradle_match and not maven_match:
            results.total = int(gradle_match.group(1))
            results.failed = int(gradle_match.group(2))
            results.passed = results.total - results.failed

        # Try to extract individual test names
        # Maven: Running com.example.TestClass
        test_class_pattern = r"Running\s+([\w.]+)"
        for match in re.finditer(test_class_pattern, combined):
            test_class = match.group(1)
            # Check if this class passed or failed based on context
            # This is a simplified approach
            results.results.append(
                TestResult(
                    test_name=test_class,
                    passed=results.failed == 0,
                )
            )

        return results


class TestRunnerRegistry:
    """Registry for test runners with auto-detection."""

    def __init__(self):
        self._runners: list[BaseTestRunner] = [
            PythonTestRunner(),
            JavaScriptTestRunner(),
            GoTestRunner(),
            RustTestRunner(),
            JavaTestRunner(),
        ]

    def register_runner(self, runner: BaseTestRunner) -> None:
        """Register a custom test runner."""
        self._runners.insert(0, runner)  # Custom runners take priority

    def detect_runner(self, project_dir: Path) -> Optional[BaseTestRunner]:
        """Auto-detect the appropriate test runner for a project."""
        for runner in self._runners:
            if runner.detect(project_dir):
                logger.info(f"Detected {runner.language.value} project")
                return runner
        return None

    def get_runner(self, language: Language) -> Optional[BaseTestRunner]:
        """Get a specific runner by language."""
        for runner in self._runners:
            if runner.language == language:
                return runner
        return None

    async def run_tests(
        self,
        project_dir: Path,
        test_filter: Optional[list[str]] = None,
        config: Optional[TestRunnerConfig] = None,
    ) -> TestRunResults:
        """Auto-detect runner and execute tests."""
        runner = self.detect_runner(project_dir)
        if not runner:
            return TestRunResults(error_message=f"No test runner detected for {project_dir}")

        return await runner.run_tests(project_dir, test_filter, config)


def detect_language(project_dir: Path) -> Language:
    """Detect the primary language of a project."""
    registry = TestRunnerRegistry()
    runner = registry.detect_runner(project_dir)
    return runner.language if runner else Language.UNKNOWN


def is_test_tool_available(language: Language) -> bool:
    """Check if test tools for a language are available."""
    checks = {
        Language.PYTHON: lambda: shutil.which("python") is not None,
        Language.JAVASCRIPT: lambda: shutil.which("npm") is not None
        or shutil.which("yarn") is not None,
        Language.GO: lambda: shutil.which("go") is not None,
        Language.RUST: lambda: shutil.which("cargo") is not None,
        Language.JAVA: lambda: shutil.which("mvn") is not None
        or shutil.which("gradle") is not None,
    }
    return checks.get(language, lambda: False)()
