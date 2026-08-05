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

"""Tests for multi-language test runner registry."""

import tempfile
from pathlib import Path


from victor.evaluation.test_runners import (
    Language,
    TestRunnerConfig,
    TestResult,
    TestRunResults,
    BaseTestRunner,
    PythonTestRunner,
    JavaScriptTestRunner,
    GoTestRunner,
    RustTestRunner,
    JavaTestRunner,
    TestRunnerRegistry,
    detect_language,
    is_test_tool_available,
)


class TestLanguageEnum:
    """Tests for Language enum."""

    def test_language_values(self):
        """Test all language enum values."""
        assert Language.PYTHON.value == "python"
        assert Language.JAVASCRIPT.value == "javascript"
        assert Language.TYPESCRIPT.value == "typescript"
        assert Language.GO.value == "go"
        assert Language.RUST.value == "rust"
        assert Language.JAVA.value == "java"
        assert Language.UNKNOWN.value == "unknown"


class TestTestRunnerConfig:
    """Tests for TestRunnerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TestRunnerConfig()
        assert config.language == Language.UNKNOWN
        assert config.test_command == ""
        assert config.timeout_seconds == 300
        assert config.env_vars == {}
        assert config.working_dir is None
        assert config.collect_coverage is False
        assert config.verbose is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TestRunnerConfig(
            language=Language.PYTHON,
            test_command="pytest -v",
            timeout_seconds=600,
            env_vars={"DEBUG": "1"},
            working_dir=Path("/tmp"),
            collect_coverage=True,
            verbose=True,
        )
        assert config.language == Language.PYTHON
        assert config.test_command == "pytest -v"
        assert config.timeout_seconds == 600
        assert config.env_vars == {"DEBUG": "1"}
        assert config.working_dir == Path("/tmp")
        assert config.collect_coverage is True
        assert config.verbose is True


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = TestResult(test_name="test_foo", passed=True)
        assert result.test_name == "test_foo"
        assert result.passed is True
        assert result.duration_ms == 0.0
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.error_message == ""
        assert result.skip_reason == ""

    def test_is_expected_without_status(self):
        """Test is_expected when no expected status set."""
        result = TestResult(test_name="test_foo", passed=True)
        assert result.is_expected is True

        result = TestResult(test_name="test_foo", passed=False)
        assert result.is_expected is True

    def test_is_expected_fail_expected(self):
        """Test is_expected when fail is expected."""
        result = TestResult(test_name="test_foo", passed=False, expected_status="fail")
        assert result.is_expected is True

        result = TestResult(test_name="test_foo", passed=True, expected_status="fail")
        assert result.is_expected is False

    def test_is_expected_pass_expected(self):
        """Test is_expected when pass is expected."""
        result = TestResult(test_name="test_foo", passed=True, expected_status="pass")
        assert result.is_expected is True

        result = TestResult(test_name="test_foo", passed=False, expected_status="pass")
        assert result.is_expected is False


class TestTestRunResults:
    """Tests for TestRunResults dataclass."""

    def test_default_values(self):
        """Test default result aggregate values."""
        results = TestRunResults()
        assert results.results == []
        assert results.total == 0
        assert results.passed == 0
        assert results.failed == 0
        assert results.skipped == 0
        assert results.errors == 0
        assert results.duration_seconds == 0.0

    def test_success_rate_empty(self):
        """Test success rate with no tests."""
        results = TestRunResults()
        assert results.success_rate == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        results = TestRunResults(total=10, passed=7, failed=3)
        assert results.success_rate == 0.7

    def test_all_passed_true(self):
        """Test all_passed when no failures."""
        results = TestRunResults(total=10, passed=10, failed=0, errors=0)
        assert results.all_passed is True

    def test_all_passed_false_with_failures(self):
        """Test all_passed when there are failures."""
        results = TestRunResults(total=10, passed=8, failed=2, errors=0)
        assert results.all_passed is False

    def test_all_passed_false_with_errors(self):
        """Test all_passed when there are errors."""
        results = TestRunResults(total=10, passed=10, failed=0, errors=1)
        assert results.all_passed is False

    def test_filter_by_names(self):
        """Test filtering results by test names."""
        results = TestRunResults(
            results=[
                TestResult(test_name="test_a", passed=True),
                TestResult(test_name="test_b", passed=False),
                TestResult(test_name="test_c", passed=True),
            ],
            total=3,
            passed=2,
            failed=1,
        )

        filtered = results.filter_by_names(["test_a", "test_c"])
        assert len(filtered.results) == 2
        assert filtered.total == 2
        assert filtered.passed == 2
        assert filtered.failed == 0


class TestPythonTestRunner:
    """Tests for PythonTestRunner."""

    def test_language_property(self):
        """Test runner returns correct language."""
        runner = PythonTestRunner()
        assert runner.language == Language.PYTHON

    def test_detect_python_project(self):
        """Test detection of Python projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # Should not detect empty directory
            runner = PythonTestRunner()
            assert runner.detect(project_dir) is False

            # Should detect with setup.py
            (project_dir / "setup.py").write_text("from setuptools import setup")
            assert runner.detect(project_dir) is True

    def test_detect_with_pyproject(self):
        """Test detection with pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "pyproject.toml").write_text("[project]\nname = 'test'")

            runner = PythonTestRunner()
            assert runner.detect(project_dir) is True

    def test_detect_test_command_pytest(self):
        """Test pytest command detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "pytest.ini").write_text("[pytest]")

            runner = PythonTestRunner()
            cmd = runner.detect_test_command(project_dir)
            assert "pytest" in cmd

    def test_detect_test_command_pyproject_pytest(self):
        """Test pytest detection from pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "pyproject.toml").write_text("[tool.pytest.ini_options]")

            runner = PythonTestRunner()
            cmd = runner.detect_test_command(project_dir)
            assert "pytest" in cmd

    def test_parse_pytest_output(self):
        """Test parsing pytest output."""
        runner = PythonTestRunner()

        stdout = """
test_module.py::test_foo PASSED
test_module.py::test_bar FAILED
test_module.py::test_baz SKIPPED

2 passed, 1 failed, 1 skipped
"""
        results = runner.parse_test_output(stdout, "")

        assert len(results.results) == 3
        # Only PASSED counts as passed, SKIPPED has passed=False with skip_reason
        assert results.passed == 1
        assert results.failed == 1
        assert results.skipped == 1

    def test_parse_unittest_output(self):
        """Test parsing unittest output."""
        runner = PythonTestRunner()

        stdout = """
test_foo (test_module.TestClass) ... ok
test_bar (test_module.TestClass) ... FAIL
test_baz (test_module.TestClass) ... skipped
"""
        results = runner.parse_test_output(stdout, "")

        assert len(results.results) == 3
        # Only 'ok' counts as passed, FAIL and skipped have passed=False
        passed_count = sum(1 for r in results.results if r.passed)
        assert passed_count == 1


class TestJavaScriptTestRunner:
    """Tests for JavaScriptTestRunner."""

    def test_language_property(self):
        """Test runner returns correct language."""
        runner = JavaScriptTestRunner()
        assert runner.language == Language.JAVASCRIPT

    def test_detect_node_project(self):
        """Test detection of Node.js projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            runner = JavaScriptTestRunner()
            assert runner.detect(project_dir) is False

            (project_dir / "package.json").write_text('{"name": "test"}')
            assert runner.detect(project_dir) is True

    def test_detect_test_command_npm(self):
        """Test npm test command detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "package.json").write_text('{"name": "test"}')

            runner = JavaScriptTestRunner()
            cmd = runner.detect_test_command(project_dir)
            assert "npm" in cmd

    def test_detect_test_command_jest(self):
        """Test jest detection from package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "package.json").write_text(
                '{"scripts": {"test": "jest"}, "devDependencies": {"jest": "^29.0.0"}}'
            )

            runner = JavaScriptTestRunner()
            cmd = runner.detect_test_command(project_dir)
            assert "jest" in cmd or "npm test" in cmd


class TestGoTestRunner:
    """Tests for GoTestRunner."""

    def test_language_property(self):
        """Test runner returns correct language."""
        runner = GoTestRunner()
        assert runner.language == Language.GO

    def test_detect_go_project(self):
        """Test detection of Go projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            runner = GoTestRunner()
            assert runner.detect(project_dir) is False

            (project_dir / "go.mod").write_text("module example.com/test")
            assert runner.detect(project_dir) is True

    def test_detect_test_command(self):
        """Test go test command detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            runner = GoTestRunner()
            cmd = runner.detect_test_command(project_dir)
            assert "go test" in cmd
            assert "-json" in cmd

    def test_parse_go_test_output(self):
        """Test parsing go test JSON output."""
        runner = GoTestRunner()

        stdout = """
{"Action":"pass","Test":"TestFoo","Elapsed":0.001}
{"Action":"fail","Test":"TestBar","Elapsed":0.002}
{"Action":"skip","Test":"TestBaz","Elapsed":0.0}
"""
        results = runner.parse_test_output(stdout, "")

        assert len(results.results) == 3
        assert results.passed == 1
        assert results.failed == 1
        assert results.skipped == 1


class TestRustTestRunner:
    """Tests for RustTestRunner."""

    def test_language_property(self):
        """Test runner returns correct language."""
        runner = RustTestRunner()
        assert runner.language == Language.RUST

    def test_detect_rust_project(self):
        """Test detection of Rust projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            runner = RustTestRunner()
            assert runner.detect(project_dir) is False

            (project_dir / "Cargo.toml").write_text('[package]\nname = "test"')
            assert runner.detect(project_dir) is True

    def test_parse_cargo_test_output(self):
        """Test parsing cargo test output."""
        runner = RustTestRunner()

        combined = """
test test_foo ... ok
test test_bar ... FAILED
test test_baz ... ignored

test result: ok. 1 passed; 1 failed; 1 ignored
"""
        results = runner.parse_test_output(combined, "")

        assert len(results.results) == 3
        assert results.passed == 1
        assert results.failed == 1
        assert results.skipped == 1


class TestJavaTestRunner:
    """Tests for JavaTestRunner."""

    def test_language_property(self):
        """Test runner returns correct language."""
        runner = JavaTestRunner()
        assert runner.language == Language.JAVA

    def test_detect_maven_project(self):
        """Test detection of Maven projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            runner = JavaTestRunner()
            assert runner.detect(project_dir) is False

            (project_dir / "pom.xml").write_text("<project></project>")
            assert runner.detect(project_dir) is True

    def test_detect_gradle_project(self):
        """Test detection of Gradle projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "build.gradle").write_text("apply plugin: 'java'")

            runner = JavaTestRunner()
            assert runner.detect(project_dir) is True

    def test_detect_test_command_maven(self):
        """Test Maven test command detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "pom.xml").write_text("<project></project>")

            runner = JavaTestRunner()
            cmd = runner.detect_test_command(project_dir)
            assert "mvn" in cmd

    def test_detect_test_command_gradle(self):
        """Test Gradle test command detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "build.gradle").write_text("apply plugin: 'java'")

            runner = JavaTestRunner()
            cmd = runner.detect_test_command(project_dir)
            assert "gradle" in cmd

    def test_parse_maven_output(self):
        """Test parsing Maven surefire output."""
        runner = JavaTestRunner()

        combined = """
Tests run: 10, Failures: 2, Errors: 1, Skipped: 1
"""
        results = runner.parse_test_output(combined, "")

        assert results.total == 10
        assert results.failed == 2
        assert results.errors == 1
        assert results.skipped == 1
        assert results.passed == 6


class TestTestRunnerRegistry:
    """Tests for TestRunnerRegistry."""

    def test_initial_runners(self):
        """Test registry has all default runners."""
        registry = TestRunnerRegistry()

        assert registry.get_runner(Language.PYTHON) is not None
        assert registry.get_runner(Language.JAVASCRIPT) is not None
        assert registry.get_runner(Language.GO) is not None
        assert registry.get_runner(Language.RUST) is not None
        assert registry.get_runner(Language.JAVA) is not None

    def test_detect_runner_python(self):
        """Test auto-detection of Python runner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "setup.py").write_text("from setuptools import setup")

            registry = TestRunnerRegistry()
            runner = registry.detect_runner(project_dir)

            assert runner is not None
            assert runner.language == Language.PYTHON

    def test_detect_runner_javascript(self):
        """Test auto-detection of JavaScript runner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "package.json").write_text('{"name": "test"}')

            registry = TestRunnerRegistry()
            runner = registry.detect_runner(project_dir)

            assert runner is not None
            assert runner.language == Language.JAVASCRIPT

    def test_detect_runner_unknown(self):
        """Test auto-detection returns None for unknown projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            registry = TestRunnerRegistry()
            runner = registry.detect_runner(project_dir)

            assert runner is None

    def test_register_custom_runner(self):
        """Test registering a custom runner."""

        class CustomRunner(BaseTestRunner):
            @property
            def language(self) -> Language:
                return Language.UNKNOWN

            def detect(self, project_dir: Path) -> bool:
                return (project_dir / "custom.config").exists()

            def detect_test_command(self, project_dir: Path) -> str:
                return "custom-test"

            async def run_tests(self, project_dir, test_filter=None, config=None):
                return TestRunResults()

            def parse_test_output(self, stdout, stderr):
                return TestRunResults()

        registry = TestRunnerRegistry()
        registry.register_runner(CustomRunner())

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "custom.config").write_text("custom")

            runner = registry.detect_runner(project_dir)
            assert runner is not None
            assert isinstance(runner, CustomRunner)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_detect_language_python(self):
        """Test detect_language for Python."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            (project_dir / "setup.py").write_text("from setuptools import setup")

            lang = detect_language(project_dir)
            assert lang == Language.PYTHON

    def test_detect_language_unknown(self):
        """Test detect_language for unknown projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            lang = detect_language(project_dir)
            assert lang == Language.UNKNOWN

    def test_is_test_tool_available_python(self):
        """Test checking if Python test tools are available."""
        # Python should always be available in test environment
        assert is_test_tool_available(Language.PYTHON) is True

    def test_is_test_tool_available_unknown(self):
        """Test checking unknown language returns False."""
        assert is_test_tool_available(Language.UNKNOWN) is False
