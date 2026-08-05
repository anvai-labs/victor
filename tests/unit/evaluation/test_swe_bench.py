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

"""Unit tests for benchmark runners.

Tests for SWE-bench, HumanEval, and MBPP benchmark runners.
"""

import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from victor.evaluation.benchmarks.swe_bench import (
    SWEBenchRunner,
    HumanEvalRunner,
    MBPPRunner,
)
from victor.evaluation.swe_bench_loader import SWEBenchInstance, _normalize_test_list
from victor.evaluation.protocol import (
    BenchmarkTask,
    BenchmarkType,
    EvaluationConfig,
    TaskResult,
    TaskStatus,
)

# =============================================================================
# SWEBenchRunner Tests
# =============================================================================


class TestSWEBenchRunnerInit:
    """Tests for SWEBenchRunner initialization."""

    def test_init_with_defaults(self):
        """Initialize with default parameters."""
        runner = SWEBenchRunner()
        assert runner._split == "test"
        assert runner._dataset_path is None
        assert runner._tasks_cache is None

    def test_init_with_custom_params(self):
        """Initialize with custom parameters."""
        runner = SWEBenchRunner(dataset_path=Path("/data/swe.jsonl"), split="dev")
        assert runner._split == "dev"
        assert runner._dataset_path == Path("/data/swe.jsonl")

    def test_benchmark_type_property(self):
        """Return correct benchmark type."""
        runner = SWEBenchRunner()
        assert runner.benchmark_type == BenchmarkType.SWE_BENCH


class TestParseTestOutput:
    """Tests for SWEBenchRunner._parse_test_output across runner formats.

    Covers pytest, unittest, and django runtests.py outputs — the F3
    "0 collected" failures were caused by django/unittest output not matching
    the old pytest-only patterns.
    """

    def setup_method(self):
        self.runner = SWEBenchRunner()

    def test_pytest_all_passed(self):
        passed, total = self.runner._parse_test_output("=== 5 passed in 1.0s ===")
        assert passed == 5 and total == 5

    def test_pytest_passed_and_failed(self):
        passed, total = self.runner._parse_test_output("=== 3 passed, 2 failed in 1.0s ===")
        assert passed == 3 and total == 5

    def test_pytest_with_errors(self):
        passed, total = self.runner._parse_test_output("1 passed, 2 errors in 1.0s")
        assert passed == 1 and total == 3

    def test_unittest_ok(self):
        out = "....\nRan 4 tests in 0.5s\n\nOK"
        passed, total = self.runner._parse_test_output(out)
        assert passed == 4 and total == 4

    def test_unittest_failed(self):
        out = ".F..\nRan 4 tests in 0.5s\n\nFAILED (failures=1)"
        passed, total = self.runner._parse_test_output(out)
        assert passed == 0 and total == 4

    def test_django_runtests_ok(self):
        # django runtests.py delegates to unittest — "Ran N tests" + "OK".
        out = "Ran 9 tests in 1.2s\n\nOK\nDestroying test database..."
        passed, total = self.runner._parse_test_output(out)
        assert passed == 9 and total == 9

    def test_django_runtests_failed(self):
        out = "Ran 9 tests in 1.2s\n\nFAILED (failures=2, errors=1)\n"
        passed, total = self.runner._parse_test_output(out)
        assert passed == 0 and total == 9

    def test_no_recognizable_output(self):
        # Garbled / unparseable output → (0, 0) → caller logs raw + "0 collected".
        passed, total = self.runner._parse_test_output("some random error trace\nno tests here")
        assert passed == 0 and total == 0


class TestVerifyPatchInContainer:
    """Tests for the refactored _verify_patch_in_container (PR2 split).

    Mocks EvalContainer + the image/test-runner resolvers so the full
    start→apply→run→stop flow is exercised without Docker, validating the
    status mapping consumed by both the post-loop eval and the verify-and-retry
    gate.
    """

    def _make_task(self):
        return BenchmarkTask(
            task_id="org__repo-1",
            benchmark=BenchmarkType.SWE_BENCH,
            description="t",
            repo="org/repo",
            base_commit="abc123",
            test_code="",  # no test patch
            fail_to_pass=["tests/test_x.py::test_a"],
        )

    def _install_fakes(self, monkeypatch, test_stdout: str):
        from types import SimpleNamespace

        from victor.evaluation import container_eval

        class _FakeContainer:
            name = "fake-eval"

            def __init__(self, **kwargs):
                self.started = False

            async def start(self):
                self.started = True

            async def exec(self, command, *, cwd=None, timeout=600, env=None):
                cmd = " ".join(command)
                if "checkout" in cmd or "clean" in cmd:
                    return 0, "", ""
                if ".agent_patch.diff" in cmd or ".test_patch.diff" in cmd:
                    return 0, "", ""
                # the test run (EvalContainer.exec returns decoded str, not bytes)
                return 0, test_stdout, ""

            async def stop(self):
                self.started = False

        monkeypatch.setattr(container_eval, "EvalContainer", _FakeContainer)

        async def _fake_image(task, config):
            return "fake/img:1"

        monkeypatch.setattr(container_eval, "resolve_swebench_image_exact", _fake_image)

        async def _noop_cleanup():
            return 0

        monkeypatch.setattr(container_eval, "cleanup_stale_sandbox_containers", _noop_cleanup)

        import victor.context.test_runner as trm

        monkeypatch.setattr(
            trm,
            "detect_test_runner",
            lambda project_root, test_files=None: SimpleNamespace(
                command=["python", "-m", "pytest", "tests/"],
                env={},
                runner_type="pytest",
            ),
        )

    @pytest.mark.asyncio
    async def test_verify_returns_passed(self, tmp_path, monkeypatch):
        self._install_fakes(monkeypatch, "=== 5 passed in 1.0s ===")
        runner = SWEBenchRunner()
        config = EvaluationConfig(benchmark=BenchmarkType.SWE_BENCH, model="test", max_tasks=1)
        vr = await runner._verify_patch_in_container(self._make_task(), "PATCH", tmp_path, config)
        assert vr.status == "passed"
        assert (vr.passed, vr.total) == (5, 5)
        assert vr.image == "fake/img:1"

    @pytest.mark.asyncio
    async def test_verify_returns_partial(self, tmp_path, monkeypatch):
        self._install_fakes(monkeypatch, "=== 3 passed, 2 failed in 1.0s ===")
        runner = SWEBenchRunner()
        config = EvaluationConfig(benchmark=BenchmarkType.SWE_BENCH, model="test", max_tasks=1)
        vr = await runner._verify_patch_in_container(self._make_task(), "PATCH", tmp_path, config)
        assert vr.status == "partial"
        assert (vr.passed, vr.total) == (3, 5)

    @pytest.mark.asyncio
    async def test_whole_file_scoring_not_node_ids(self, tmp_path, monkeypatch):
        """A task with BOTH fail_to_pass node-IDs and a test_code patch must
        drive the test command from the test_code WHOLE FILE, not the node-IDs
        (node-ID scoring mis-scores — see the 64%→20% validation regression).
        """
        from types import SimpleNamespace

        from victor.evaluation import container_eval

        captured = {}

        class _FakeContainer:
            name = "fake"

            def __init__(self, **kw):
                self.started = False

            async def start(self):
                self.started = True

            async def exec(self, command, *, cwd=None, timeout=600, env=None):
                cmd = " ".join(command)
                if "checkout" in cmd or "clean" in cmd:
                    return 0, "", ""
                if ".agent_patch.diff" in cmd or ".test_patch.diff" in cmd:
                    return 0, "", ""
                return 0, "1 passed", ""

            async def stop(self):
                self.started = False

        monkeypatch.setattr(container_eval, "EvalContainer", _FakeContainer)

        async def _img(task, config):
            return "img"

        async def _noop():
            return 0

        monkeypatch.setattr(container_eval, "resolve_swebench_image_exact", _img)
        monkeypatch.setattr(container_eval, "cleanup_stale_sandbox_containers", _noop)
        import victor.context.test_runner as trm

        def _rec(project_root, test_files=None):
            captured["test_files"] = test_files
            return SimpleNamespace(
                command=["python", "-m", "pytest", "x"], env={}, runner_type="pytest"
            )

        monkeypatch.setattr(trm, "detect_test_runner", _rec)

        task = BenchmarkTask(
            task_id="org__repo-1",
            benchmark=BenchmarkType.SWE_BENCH,
            description="t",
            repo="org/repo",
            base_commit="abc",
            test_code="diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n",
            fail_to_pass=["tests/test_x.py::test_node_id"],  # must be IGNORED
        )
        runner = SWEBenchRunner()
        config = EvaluationConfig(benchmark=BenchmarkType.SWE_BENCH, model="t", max_tasks=1)
        await runner._verify_patch_in_container(task, "PATCH", tmp_path, config)
        # Whole test file from the test patch — NOT the fail_to_pass node-ID.
        assert captured["test_files"] == ["tests/test_x.py"]

    @pytest.mark.asyncio
    async def test_verify_reuses_supplied_persistent_container(self, tmp_path, monkeypatch):
        """A pre-started container is reused: no start()/stop() (caller owns the
        lifecycle), only exec() — this is what makes verify-and-retry cheap
        (one container per task, not N+1)."""
        self._install_fakes(monkeypatch, "=== 5 passed in 1.0s ===")

        class _Persistent:
            name = "persist-1"

            def __init__(self):
                self.start_calls = 0
                self.stop_calls = 0
                self.exec_calls = 0

            async def start(self):
                self.start_calls += 1

            async def stop(self):
                self.stop_calls += 1

            async def exec(self, command, *, cwd=None, timeout=600, env=None):
                self.exec_calls += 1
                cmd = " ".join(command)
                if "checkout" in cmd or "clean" in cmd or ".agent_patch" in cmd:
                    return 0, "", ""
                return 0, "=== 5 passed in 1.0s ===", ""

        persist = _Persistent()
        runner = SWEBenchRunner()
        config = EvaluationConfig(benchmark=BenchmarkType.SWE_BENCH, model="test", max_tasks=1)
        vr = await runner._verify_patch_in_container(
            self._make_task(), "PATCH", tmp_path, config, container=persist
        )
        assert vr.status == "passed"
        assert (vr.passed, vr.total) == (5, 5)
        assert persist.start_calls == 0  # caller owns lifecycle — not re-started
        assert persist.stop_calls == 0  # NOT stopped — left running for reuse
        assert persist.exec_calls > 0  # tests ran inside the supplied container


class TestFailToPassNormalization:
    """Regression: FAIL_TO_PASS is a JSON-stringified list in the dataset.

    Storing it raw broke the test command — list(str) exploded it into single
    chars → cmd.extend(chars) → pytest got one-char args → 0 collected on every
    task (0/20 pass on the PR1 validation run).
    """

    def test_normalize_json_string(self):
        assert _normalize_test_list('["tests/test_a.py::test_a", "tests/test_b.py"]') == [
            "tests/test_a.py::test_a",
            "tests/test_b.py",
        ]

    def test_normalize_already_list(self):
        assert _normalize_test_list(["a", "b"]) == ["a", "b"]

    def test_normalize_empty_or_garbage(self):
        assert _normalize_test_list("") == []
        assert _normalize_test_list("not json") == []
        assert _normalize_test_list(None) == []

    def test_from_jsonl_line_parses_fail_to_pass_string(self):
        raw = {
            "instance_id": "org__repo-1",
            "repo": "org/repo",
            "base_commit": "abc",
            "problem_statement": "ps",
            "hints_text": "",
            "patch": "",
            "test_patch": "",
            "FAIL_TO_PASS": '["tests/test_a.py::test_a"]',
            "PASS_TO_PASS": "[]",
            "created_at": "",
        }
        inst = SWEBenchInstance.from_jsonl_line(raw)
        assert inst.fail_to_pass == ["tests/test_a.py::test_a"]  # list, not chars
        task = inst.to_benchmark_task()
        assert task.fail_to_pass == ["tests/test_a.py::test_a"]


class TestExtractPatch:
    """Tests for _extract_patch method."""

    def test_extract_standard_diff_patch(self):
        """Extract standard diff patch format."""
        runner = SWEBenchRunner()

        output = """Some text
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,1 +1,2 @@
-old line
+new line
"""

        patch = runner._extract_patch(output)
        assert "diff --git a/file.py b/file.py" in patch
        assert "--- a/file.py" in patch
        assert "+new line" in patch

    def test_extract_patch_from_code_block(self):
        """Extract patch from markdown code block."""
        runner = SWEBenchRunner()

        output = """Here's the fix:

```diff
--- a/file.py
+++ b/file.py
@@ -1,1 +1,1 @@
-old
+new
```
"""

        patch = runner._extract_patch(output)
        assert "--- a/file.py" in patch
        assert "+new" in patch

    def test_extract_patch_no_diff_returns_empty(self):
        """Return empty string when no diff found."""
        runner = SWEBenchRunner()

        output = "Just regular text without any diff"

        patch = runner._extract_patch(output)
        assert patch == ""

    def test_extract_patch_with_multiple_diffs(self):
        """Extract first complete diff from multiple diffs."""
        runner = SWEBenchRunner()

        output = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,1 +1,1 @@
-a
+b
diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,1 +1,1 @@
-c
+d
"""

        patch = runner._extract_patch(output)
        # Should extract at least the first diff
        assert "diff --git a/file1.py" in patch


class TestSWEBenchRunnerLoadTasks:
    """Tests for load_tasks method."""

    @pytest.mark.asyncio
    async def test_load_tasks_returns_cached_tasks(self):
        """Load tasks returns cached tasks when available."""
        runner = SWEBenchRunner()
        runner._tasks_cache = []
        config = EvaluationConfig(benchmark=BenchmarkType.SWE_BENCH, model="test", max_tasks=10)

        tasks = await runner.load_tasks(config)
        assert tasks == []

    @pytest.mark.asyncio
    async def test_load_tasks_filters_by_config(self):
        """Load tasks filters based on config max_tasks."""
        from victor.evaluation.protocol import BenchmarkTask

        mock_task1 = BenchmarkTask(
            task_id="task-1",
            benchmark=BenchmarkType.SWE_BENCH,
            description="Task 1",
        )
        mock_task2 = BenchmarkTask(
            task_id="task-2",
            benchmark=BenchmarkType.SWE_BENCH,
            description="Task 2",
        )

        runner = SWEBenchRunner()
        runner._tasks_cache = [mock_task1, mock_task2]
        config = EvaluationConfig(benchmark=BenchmarkType.SWE_BENCH, model="test", max_tasks=1)

        tasks = await runner.load_tasks(config)
        assert len(tasks) == 1
        assert tasks[0].task_id == "task-1"


# =============================================================================
# HumanEvalRunner Tests
# =============================================================================


class TestHumanEvalRunnerInit:
    """Tests for HumanEvalRunner initialization."""

    def test_init_default(self):
        """HumanEvalRunner uses default initialization."""
        runner = HumanEvalRunner()
        # No special attributes since it uses BaseBenchmarkRunner defaults
        assert runner.benchmark_type == BenchmarkType.HUMAN_EVAL

    def test_benchmark_type_property(self):
        """Return correct benchmark type."""
        runner = HumanEvalRunner()
        assert runner.benchmark_type == BenchmarkType.HUMAN_EVAL


# =============================================================================
# MBPPRunner Tests
# =============================================================================


class TestMBPPRunnerInit:
    """Tests for MBPPRunner initialization."""

    def test_init_with_default_split(self):
        """Initialize with default split."""
        runner = MBPPRunner()
        assert runner._split == "test"

    def test_init_with_custom_split(self):
        """Initialize with custom split."""
        runner = MBPPRunner(split="dev")
        assert runner._split == "dev"

    def test_benchmark_type_property(self):
        """Return correct benchmark type."""
        runner = MBPPRunner()
        assert runner.benchmark_type == BenchmarkType.MBPP


class TestMBPPBuildPrompt:
    """Tests for _build_prompt method."""

    def test_build_prompt_from_mbpp_item(self):
        """Build prompt from MBPP item."""
        runner = MBPPRunner()

        item = {
            "task_id": 1,
            "text": "Write a function to add two numbers.",
            "code": "",
            "test_list": ["assert add(1, 2) == 3"],
        }

        prompt = runner._build_prompt(item)
        assert "Write a function to add two numbers." in prompt


class TestMBPPBuildTestCode:
    """Tests for _build_test_code method."""

    def test_build_test_code_from_mbpp_item(self):
        """Build test code from MBPP item."""
        runner = MBPPRunner()

        item = {
            "task_id": 1,
            "text": "Write a function.",
            "code": "",
            "test_list": [
                "assert add(1, 2) == 3",
                "assert add(0, 0) == 0",
            ],
        }

        test_code = runner._build_test_code(item)
        assert "assert add(1, 2) == 3" in test_code
        assert "assert add(0, 0) == 0" in test_code

    def test_build_test_code_with_setup(self):
        """Build test code with setup code."""
        runner = MBPPRunner()

        item = {
            "task_id": 1,
            "text": "Write a function.",
            "code": "",
            "test_setup_code": "def add(a, b): return a + b",
            "test_list": ["assert add(1, 2) == 3"],
        }

        test_code = runner._build_test_code(item)
        assert "def add(a, b): return a + b" in test_code
        assert "assert add(1, 2) == 3" in test_code


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestRunTaskErrorHandling:
    """Tests for run_task error handling."""

    @pytest.mark.asyncio
    async def test_run_task_handles_exception(self):
        """Run task properly handles exceptions."""
        from victor.evaluation.protocol import EvaluationConfig

        runner = SWEBenchRunner()
        task = MagicMock()
        task.task_id = "test-1"
        task.description = "Test task"
        task.test_code = "assert True"
        # Set repo to None so SWEBenchWorkspaceManager returns None for
        # cached_repo, forcing the TaskEnvironment fallback path.
        task.repo = None

        agent_output = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old\n+new"
        config = EvaluationConfig(benchmark=BenchmarkType.SWE_BENCH, model="test", use_docker=False)

        # Mock TaskEnvironment to raise exception on setup
        with patch("victor.evaluation.benchmarks.swe_bench.TaskEnvironment") as MockEnv:
            mock_env_instance = MagicMock()
            mock_env_instance.setup = AsyncMock(side_effect=Exception("Setup failed"))
            mock_env_instance.cleanup = AsyncMock()
            MockEnv.return_value = mock_env_instance

            result = await runner.run_task(task, agent_output, config)

            assert result.status == TaskStatus.ERROR
            assert "Setup failed" in result.error_message

    @pytest.mark.asyncio
    async def test_cached_repo_success_logs_test_output_at_debug_only(self, tmp_path, caplog):
        """Passing test output should not pollute INFO logs."""
        runner = SWEBenchRunner()
        task = MagicMock()
        task.task_id = "test-1"
        task.base_commit = None
        task.test_code = None
        task.repo = None

        result = TaskResult(task_id="test-1", status=TaskStatus.RUNNING, generated_code="diff")
        config = EvaluationConfig(benchmark=BenchmarkType.SWE_BENCH, model="test")
        cached_repo = tmp_path / "repo"
        cached_repo.mkdir()
        diff_patch = (
            "diff --git a/file.py b/file.py\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )

        apply_proc = MagicMock()
        apply_proc.returncode = 0
        apply_proc.communicate = AsyncMock(return_value=(b"", b""))

        test_proc = MagicMock()
        test_proc.communicate = AsyncMock(
            return_value=(
                b"astropy/test_example.py::test_ok PASSED\n================ 1 passed in 0.10s ================\n",
                b"",
            )
        )

        runner_config = MagicMock()
        runner_config.command = ["python", "-m", "pytest", "astropy/test_example.py"]
        runner_config.env = {}
        runner_config.runner_type = "pytest"

        caplog.set_level(logging.DEBUG, logger="victor.evaluation.benchmarks.swe_bench")

        with (
            patch(
                "asyncio.create_subprocess_exec",
                side_effect=[apply_proc, test_proc],
            ),
            patch(
                "victor.context.test_runner.detect_test_runner",
                return_value=runner_config,
            ),
        ):
            updated = await runner._run_tests_in_cached_repo(
                task,
                result,
                diff_patch,
                cached_repo,
                config,
            )

        assert updated.status == TaskStatus.PASSED
        assert any(
            record.levelno == logging.DEBUG and "Test stdout (last 500)" in record.message
            for record in caplog.records
        )
        assert not any(
            record.levelno == logging.INFO and "Test stdout (last 500)" in record.message
            for record in caplog.records
        )


# =============================================================================
# Configuration Tests
# =============================================================================


class TestBenchmarkConfiguration:
    """Tests for benchmark configuration and filtering."""

    def test_swe_bench_config_default_path(self):
        """SWEBenchRunner defaults to no dataset_path."""
        runner = SWEBenchRunner()
        assert runner._dataset_path is None

    def test_mbpp_config_default_split(self):
        """MBPPRunner defaults to 'test' split."""
        runner = MBPPRunner()
        assert runner._split == "test"

    def test_all_runners_have_correct_benchmark_type(self):
        """All runners report correct benchmark type."""
        swe_runner = SWEBenchRunner()
        human_runner = HumanEvalRunner()
        mbpp_runner = MBPPRunner()

        assert swe_runner.benchmark_type == BenchmarkType.SWE_BENCH
        assert human_runner.benchmark_type == BenchmarkType.HUMAN_EVAL
        assert mbpp_runner.benchmark_type == BenchmarkType.MBPP


class TestShadowInstallGuard:
    """Host-path verification must never score an ambient site-packages install.

    Regression guard for the OP-RUN-3 finding: anaconda's astropy 6.1.3 scored
    astropy__astropy-12907 15/15 "passed" against a patch that fixed nothing,
    because the imported package (and its own test files) shadowed the
    base-commit checkout.
    """

    def test_checkout_editable_install_is_accepted(self, tmp_path):
        checkout = tmp_path / "repo"
        module_dir = checkout / "astropy"
        module_dir.mkdir(parents=True)
        assert SWEBenchRunner._module_resolves_to_checkout(module_dir, checkout)

    def test_ambient_site_packages_is_rejected(self, tmp_path):
        checkout = tmp_path / "repo"
        checkout.mkdir()
        ambient = tmp_path / "site-packages" / "astropy"
        ambient.mkdir(parents=True)
        assert not SWEBenchRunner._module_resolves_to_checkout(ambient, checkout)

    def test_symlinked_checkout_resolves(self, tmp_path):
        checkout = tmp_path / "repo"
        (checkout / "pkg").mkdir(parents=True)
        link = tmp_path / "link"
        link.symlink_to(checkout)
        assert SWEBenchRunner._module_resolves_to_checkout(link / "pkg", checkout)


class TestGoldTestPatchApplyGuard:
    """A gold test patch that fails to apply must ERROR, not score old tests."""

    async def test_failed_test_patch_apply_errors_the_task(self, tmp_path, monkeypatch):
        import subprocess

        # A real git repo whose HEAD cannot accept the (bogus) test patch.
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
        (repo / "a.txt").write_text("hello\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

        from victor.evaluation.protocol import (
            BenchmarkTask,
            BenchmarkType,
            EvaluationConfig,
            TaskStatus,
        )

        task = BenchmarkTask(
            task_id="t1",
            benchmark=BenchmarkType.SWE_BENCH,
            description="d",
            test_code=(
                "diff --git a/missing.py b/missing.py\n"
                "--- a/missing.py\n"
                "+++ b/missing.py\n"
                "@@ -1,1 +1,1 @@\n"
                "-not the real content\n"
                "+still not\n"
            ),
        )
        # base_commit=None skips the reset branch; the agent patch is empty-but-valid.
        runner = SWEBenchRunner()
        result = TaskResult(task_id="t1", status=TaskStatus.RUNNING)
        config = EvaluationConfig(benchmark=BenchmarkType.SWE_BENCH, model="m", timeout_per_task=30)
        out = await runner._run_tests_in_cached_repo(task, result, "", repo, config)
        assert out.status == TaskStatus.ERROR
        assert "test patch failed to apply" in (out.error_message or "").lower()
