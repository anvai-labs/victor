# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""SWE-bench benchmark runner.

Implements evaluation against the SWE-bench benchmark for
real-world software engineering tasks.

NOTE: Task loading is delegated to SWEBenchLoader (swe_bench_loader.py)
to avoid code duplication. This module focuses on task execution.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:
    from victor.evaluation.container_eval import EvalContainer

from victor.evaluation.harness import BaseBenchmarkRunner, TaskEnvironment
from victor.evaluation.protocol import (
    BenchmarkTask,
    BenchmarkType,
    EvaluationConfig,
    TaskResult,
    TaskStatus,
)
from victor.evaluation.swe_bench_loader import SWEBenchConfig, SWEBenchLoader

logger = logging.getLogger(__name__)


class _ContainerVerifyResult(NamedTuple):
    """Outcome of running a patch's tests in an official SWE-bench container.

    Returned by :meth:`SWEBenchRunner._verify_patch_in_container` and consumed
    by both the post-loop eval (maps ``status`` → ``TaskResult``) and the
    closed-loop verify-and-retry gate (checks ``status == "passed"``).
    """

    passed: int
    total: int
    raw_output: str
    # "passed" | "partial" | "all_failed" | "zero_collected" | "apply_failed" | "error"
    status: str
    image: str = ""
    container_name: str = ""


class SWEBenchRunner(BaseBenchmarkRunner):
    """Runner for SWE-bench benchmark.

    SWE-bench evaluates the ability to solve REAL GitHub issues
    from popular Python repositories using the official Princeton
    NLP dataset from HuggingFace.

    Dataset: https://huggingface.co/datasets/princeton-nlp/SWE-bench
    Paper: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (2024)
    Leaderboard: https://www.swebench.com/

    IMPORTANT: This runner loads REAL benchmark data from HuggingFace,
    clones REAL repositories, applies patches, and runs REAL tests.
    Results are not simulated - they represent actual test execution.

    Supported splits:
    - test: Full test set (~2294 tasks)
    - dev: Development set
    - lite: SWE-bench Lite (~300 tasks, curated subset)

    Task loading is handled by SWEBenchLoader for consistency.
    """

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        split: str = "test",
    ):
        """Initialize the SWE-bench runner.

        Args:
            dataset_path: Path to SWE-bench dataset (JSONL file)
            split: Dataset split (test, dev, lite)
        """
        self._dataset_path = dataset_path
        self._split = split
        self._tasks_cache: Optional[list[BenchmarkTask]] = None
        self._loader = SWEBenchLoader()

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.SWE_BENCH

    async def load_tasks(
        self,
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Load SWE-bench tasks using SWEBenchLoader.

        Delegates to SWEBenchLoader for consistent loading across
        the codebase. Supports both local files and HuggingFace.
        """
        if self._tasks_cache is not None:
            return self._filter_tasks(self._tasks_cache, config)

        tasks = []

        # Build loader config from evaluation config
        loader_config = SWEBenchConfig(
            max_tasks=config.max_tasks,
        )

        # Try loading from dataset file first
        if self._dataset_path and self._dataset_path.exists():
            tasks = self._loader.load_from_file(self._dataset_path, loader_config)
        else:
            # Load from HuggingFace using the split
            # SWE-bench datasets on HuggingFace:
            # - princeton-nlp/SWE-bench_Lite (curated ~300 tasks)
            # - princeton-nlp/SWE-bench (full ~2300 tasks)
            if self._split == "lite":
                dataset_name = "princeton-nlp/SWE-bench_Lite"
                split = "test"
            else:
                dataset_name = "princeton-nlp/SWE-bench"
                split = "test"
            tasks = await self._loader.load_from_huggingface(
                dataset_name=dataset_name,
                split=split,
                config=loader_config,
            )

        self._tasks_cache = tasks
        return self._filter_tasks(tasks, config)

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run a SWE-bench task and evaluate."""
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            generated_code=agent_output,
        )

        # Extract patch from agent output
        patch = self._extract_patch(agent_output)
        result.generated_patch = patch

        if not patch:
            result.status = TaskStatus.FAILED
            result.error_message = "No valid patch found in output"
            return result

        # Use cached repo for fast local test execution instead of
        # cloning from GitHub. Apply patch to a clean checkout and run tests.
        try:
            from victor.evaluation.swe_bench_loader import SWEBenchWorkspaceManager

            workspace_manager = SWEBenchWorkspaceManager()
            cached_repo = workspace_manager.get_cached_repo_path(task)

            if cached_repo and cached_repo.exists():
                if self._select_backend(config) == "docker":
                    # Containerized eval: each task runs in its correct runtime
                    # image (official SWE-bench per-instance image for Python).
                    # Falls back to the host path if Docker / the image is
                    # unavailable so a missing Docker runtime never hard-fails.
                    from victor.evaluation.container_eval import DockerUnavailable

                    try:
                        result = await self._run_tests_in_container(
                            task, result, patch, cached_repo, config
                        )
                    except DockerUnavailable as exc:
                        logger.warning(
                            "Docker eval unavailable (%s) — falling back to host path",
                            exc,
                        )
                        result = await self._run_tests_in_cached_repo(
                            task, result, patch, cached_repo, config
                        )
                else:
                    result = await self._run_tests_in_cached_repo(
                        task, result, patch, cached_repo, config
                    )
            else:
                # Fallback to TaskEnvironment with remote clone
                env = TaskEnvironment(
                    task=task,
                    workspace_dir=config.workspace_dir,
                    use_docker=config.use_docker,
                    docker_image=config.docker_image,
                )
                try:
                    await env.setup()
                    if not await env.apply_patch(patch):
                        result.status = TaskStatus.FAILED
                        result.error_message = "Failed to apply patch"
                        return result
                    if task.test_code:
                        await env.apply_patch(task.test_code)
                    passed, total, stdout, stderr = await env.run_tests(
                        timeout=config.timeout_per_task
                    )
                    result.tests_passed = passed
                    result.tests_total = total
                    result.tests_failed = total - passed
                    result.stdout = stdout
                    result.stderr = stderr
                    if total > 0 and passed == total:
                        result.status = TaskStatus.PASSED
                    elif passed > 0:
                        result.status = TaskStatus.FAILED
                        result.error_message = f"Partial pass: {passed}/{total}"
                    else:
                        result.status = TaskStatus.FAILED
                        result.error_message = "All tests failed"
                except Exception as e:
                    result.status = TaskStatus.ERROR
                    result.error_message = str(e)
                finally:
                    await env.cleanup()

        except Exception as e:
            result.status = TaskStatus.ERROR
            result.error_message = str(e)

        return result

    async def _run_tests_in_cached_repo(
        self,
        task: BenchmarkTask,
        result: TaskResult,
        patch: str,
        cached_repo: Path,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run tests using the cached repo for fast local execution.

        Resets repo to base_commit, applies agent's patch, runs tests,
        then resets again to leave repo clean.
        """
        import asyncio
        import sys
        import tempfile

        try:
            # Reset to base commit (fetch first — may be outside shallow depth)
            if task.base_commit:
                for cmd in [
                    ["git", "fetch", "--depth", "1", "origin", task.base_commit],
                    ["git", "checkout", "--force", task.base_commit],
                    # Preserve .victor (backups + embeddings). Without -e .victor,
                    # git clean wipes the workspace's .victor/ between tasks: the
                    # singleton FileEditor's backup_dir vanishes -> FileNotFoundError
                    # on every later edit's commit (P0); the LanceDB embeddings
                    # store vanishes -> the index rebuilds every task (P2). Matches
                    # the agent_callback's own clean.
                    ["git", "clean", "-fd", "-e", ".victor"],
                ]:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        cwd=cached_repo,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.communicate()

            # Apply the agent's patch
            patch_file = cached_repo / ".agent_patch.diff"
            patch_file.write_text(patch)
            apply_proc = await asyncio.create_subprocess_exec(
                "git",
                "apply",
                "--allow-empty",
                str(patch_file),
                cwd=cached_repo,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            apply_stdout, apply_stderr = await apply_proc.communicate()

            if apply_proc.returncode != 0:
                result.status = TaskStatus.FAILED
                result.error_message = f"Failed to apply patch: {apply_stderr.decode()[:200]}"
                logger.warning("Patch apply failed: %s", apply_stderr.decode()[:200])
                return result

            logger.info("Patch applied successfully to cached repo")

            # Apply test patch if available (SWE-bench provides test code)
            if task.test_code:
                test_patch_file = cached_repo / ".test_patch.diff"
                test_patch_file.write_text(task.test_code)
                test_proc = await asyncio.create_subprocess_exec(
                    "git",
                    "apply",
                    "--allow-empty",
                    str(test_patch_file),
                    cwd=cached_repo,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await test_proc.communicate()

            # Try to apply patch to the installed package in site-packages.
            # Source checkouts of projects like astropy need compiled C extensions
            # which aren't available from the raw git checkout. If the project is
            # pip-installed, we can patch site-packages and run tests against it.
            import os

            site_pkg_dir = None
            if task.repo:
                # Extract project name from repo URL (e.g., "astropy" from "astropy/astropy")
                repo_name = task.repo.rstrip("/").split("/")[-1].replace(".git", "")

                # Use framework's workspace dependency installer
                from victor.context.workspace_setup import (
                    ensure_project_importable,
                )

                # Build C-extensions for projects like astropy (default 120s is
                # too short for a first C-ext compile; subsequent tasks are fast
                # because pip sees the package already installed).
                await ensure_project_importable(
                    repo_name, cached_repo, install_deps=True, timeout=600
                )

                try:
                    spec = __import__(repo_name)
                    site_pkg_dir = Path(spec.__file__).parent
                    # Apply patch to site-packages
                    site_patch_file = cached_repo / ".site_patch.diff"
                    site_patch_file.write_text(patch)
                    apply_site = await asyncio.create_subprocess_exec(
                        "git",
                        "apply",
                        "--allow-empty",
                        "--directory",
                        str(site_pkg_dir.parent),
                        str(site_patch_file),
                        cwd=cached_repo,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    site_out, site_err = await apply_site.communicate()
                    if apply_site.returncode == 0:
                        logger.info(
                            "Patch also applied to installed package at %s",
                            site_pkg_dir,
                        )
                    site_patch_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.debug("Could not patch installed package: %s", e)

            # Run tests using detected test runner
            from victor.context.test_runner import detect_test_runner
            import re as _re

            # Test files come from the test PATCH (whole files), NOT task.
            # fail_to_pass node-IDs — running only the FAIL_TO_PASS node-IDs
            # (with --noconftest) failed tests that the whole-file run passes
            # (validation run dropped 64%→20%). Whole-file scoring is the
            # proven baseline; fail_to_pass stays carried for reporting/future.
            _test_files = None
            if task.test_code:
                _extracted = _re.findall(r"diff --git a/(\S+)", task.test_code)
                _test_files = [f for f in _extracted if "test" in f.lower()]

            _runner_config = detect_test_runner(cached_repo, test_files=_test_files or None)
            test_cmd = _runner_config.command
            # Add --noconftest only for pytest (avoids conftest conflicts)
            if _runner_config.runner_type == "pytest" and "-m" in test_cmd:
                idx = test_cmd.index("-m")
                test_cmd.insert(idx + 2, "--noconftest")
            logger.info("Running tests: %s", " ".join(test_cmd))

            # Use installed package path as test root if available
            test_cwd = str(site_pkg_dir.parent) if site_pkg_dir else str(cached_repo)

            clean_env = os.environ.copy()
            clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
            # Apply runner-specific env vars (e.g., DJANGO_SETTINGS_MODULE)
            clean_env.update(_runner_config.env)

            test_proc = await asyncio.create_subprocess_exec(
                *test_cmd,
                cwd=test_cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=clean_env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    test_proc.communicate(),
                    timeout=min(config.timeout_per_task, 300),
                )
            except asyncio.TimeoutError:
                test_proc.kill()
                result.status = TaskStatus.FAILED
                result.error_message = "Test execution timed out"
                return result

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            result.stdout = stdout_str[-2000:]  # Last 2KB for diagnostics
            result.stderr = stderr_str[-2000:]

            # Parse test results
            passed, total = self._parse_test_output(stdout_str + stderr_str)
            result.tests_passed = passed
            result.tests_total = total
            result.tests_failed = total - passed

            if total > 0 and passed == total:
                result.status = TaskStatus.PASSED
                if stdout_str:
                    logger.debug("Test stdout (last 500): %s", stdout_str[-500:])
                if stderr_str:
                    logger.debug("Test stderr (last 500): %s", stderr_str[-500:])
                logger.info("Tests PASSED: %d/%d", passed, total)
            elif total > 0 and passed > 0:
                result.status = TaskStatus.FAILED
                result.error_message = f"Partial pass: {passed}/{total}"
                logger.info("Test stdout (last 500): %s", stdout_str[-500:])
                logger.info("Test stderr (last 500): %s", stderr_str[-500:])
                logger.info("Tests partial: %d/%d", passed, total)
            elif total > 0:
                result.status = TaskStatus.FAILED
                result.error_message = f"All tests failed ({total} total)"
                logger.info("Test stdout (last 500): %s", stdout_str[-500:])
                logger.info("Test stderr (last 500): %s", stderr_str[-500:])
                logger.info("Tests FAILED: %d/%d", passed, total)
            else:
                # Tests couldn't run (0 collected) — likely missing project deps.
                # Report as PATCH_APPLIED to distinguish from "no patch generated".
                result.status = TaskStatus.FAILED
                result.error_message = (
                    "Patch applied successfully but tests could not run "
                    "(0 collected). Install project deps or use Docker."
                )
                logger.warning(
                    "Tests not collected (0/0). Project may need "
                    "`pip install -e .` or Docker for test execution."
                )
                # Surface the raw test output so a parser gap vs a genuine
                # collection failure is diagnosable (e.g. django runtests.py
                # output that doesn't match the pytest/unittest patterns).
                logger.warning(
                    "Raw test output (last 1500 chars) for 0-collected task " "%s:\n%s",
                    getattr(task, "task_id", "?"),
                    (stdout_str + stderr_str)[-1500:],
                )

        except Exception as e:
            result.status = TaskStatus.ERROR
            result.error_message = str(e)
            logger.error("Test execution error: %s", e)

        finally:
            # Clean up patch files and reset repo
            for f in [".agent_patch.diff", ".test_patch.diff"]:
                (cached_repo / f).unlink(missing_ok=True)
            if task.base_commit:
                proc = await asyncio.create_subprocess_exec(
                    "git",
                    "checkout",
                    "--force",
                    task.base_commit,
                    cwd=cached_repo,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()

        return result

    @staticmethod
    def _select_backend(config: EvaluationConfig) -> str:
        """Pick the eval backend: ``"docker"`` or ``"local"`` (host).

        ``eval_backend`` is the modern switch; the legacy ``use_docker=True`` is
        treated as ``docker`` for back-compat. Default is ``local``.
        """
        backend = getattr(config, "eval_backend", "local")
        if backend == "docker":
            return "docker"
        # Legacy flag: use_docker=True ⇒ docker (unless eval_backend explicitly local).
        if backend == "local" and getattr(config, "use_docker", False):
            return "docker"
        return "local"

    async def _verify_patch_in_container(
        self,
        task: BenchmarkTask,
        patch: str,
        cached_repo: Path,
        config: EvaluationConfig,
        container: Optional["EvalContainer"] = None,
    ) -> "_ContainerVerifyResult":
        """Apply ``patch`` to ``/testbed`` in an official SWE-bench container
        and run the FAIL_TO_PASS tests.

        By default runs a full container lifecycle (start → exec×N → stop) so it
        is stateless. Pass a pre-started ``container`` to reuse a persistent
        per-task container across verify-and-retry calls (and the final eval):
        the caller owns start/stop; this method only resets /testbed, applies the
        patch, runs tests, and returns WITHOUT stopping it. Raises
        :class:`victor.evaluation.container_eval.DockerUnavailable` (propagated
        to the caller for host fallback) if Docker or the image can't be used.
        """
        import re as _re

        from victor.context.test_runner import detect_test_runner
        from victor.evaluation.container_eval import (
            EVAL_WORKSPACE_MOUNT,
            EvalContainer,
            cleanup_stale_eval_containers,
            resolve_runtime,
            resolve_swebench_image_exact,
        )

        # One-shot: sweep orphaned eval containers from prior crashed runs
        # (process death before stop() → containers accumulate → Docker strain).
        # Skip when a persistent container is supplied — it's already running and
        # would match the cleanup pattern (victor-eval-*) → "No such container".
        if container is None and not getattr(SWEBenchRunner, "_stale_containers_cleaned", False):
            SWEBenchRunner._stale_containers_cleaned = True
            await cleanup_stale_eval_containers()

        image = (
            await resolve_swebench_image_exact(task, config)
            if getattr(task, "repo", None)
            else resolve_runtime(task, config).base_image
        )
        _owns_container = container is None
        if _owns_container:
            container = EvalContainer(image=image, workspace_host_path=str(cached_repo))
        ws = EVAL_WORKSPACE_MOUNT  # "/workspace" — cached_repo mount (patch files)
        # The official image ships the repo EDITABLE-INSTALLED at /testbed (its
        # PWD): apply patches there and run tests — NO mount-and-reinstall.
        testbed = "/testbed"
        passed = total = 0
        raw = ""
        status = "error"
        try:
            if _owns_container:
                await container.start()  # raises DockerUnavailable if unusable

            # Write the agent + test patches into the mounted cached_repo (host
            # side) so the container can read them from /workspace.
            patch_file = cached_repo / ".agent_patch.diff"
            patch_file.write_text(patch)
            if task.test_code:
                (cached_repo / ".test_patch.diff").write_text(task.test_code)

            # 1. Reset /testbed to base_commit (idempotency; commit is local).
            if task.base_commit:
                for cmd in (
                    ["git", "checkout", "--force", task.base_commit],
                    ["git", "clean", "-fd"],
                ):
                    rc, _out, err = await container.exec(cmd, cwd=testbed, timeout=180)
                    if rc != 0:
                        logger.debug("container git %s rc=%d: %s", cmd[1], rc, (err or "")[:200])

            # 2. Apply the agent's patch to /testbed (read from /workspace).
            rc, _out, err = await container.exec(
                ["git", "apply", f"{ws}/.agent_patch.diff"], cwd=testbed
            )
            if rc != 0:
                status = "apply_failed"
                raw = (err or "")[:500]
                logger.warning("Patch apply failed in container: %s", raw)
            else:
                logger.info("Patch applied successfully in container /testbed")
                # 3. Apply the SWE-bench test patch.
                if task.test_code:
                    await container.exec(["git", "apply", f"{ws}/.test_patch.diff"], cwd=testbed)

                # 4. Detect the test runner on the host cached_repo (same
                #    repo/files) and run its command in /testbed. Use whole test
                #    FILES from the test patch — NOT fail_to_pass node-IDs (the
                #    node-ID form mis-scores; see _run_tests_in_cached_repo).
                _test_files = None
                if task.test_code:
                    _test_files = [
                        f
                        for f in _re.findall(r"diff --git a/(\S+)", task.test_code)
                        if "test" in f.lower()
                    ]
                runner_cfg = detect_test_runner(cached_repo, test_files=_test_files or None)
                test_cmd = list(runner_cfg.command)
                # detect_test_runner embeds HOST ABSOLUTE PATHS which don't
                # exist in the container (repo is at /testbed). Rewrite: python
                # binary → bare ``python``; cached_repo prefix → relative path.
                if test_cmd and "python" in test_cmd[0]:
                    test_cmd[0] = "python"
                _repo_prefix = str(cached_repo.resolve()) + "/"
                test_cmd = [
                    (arg.replace(_repo_prefix, "") if arg.startswith(_repo_prefix) else arg)
                    for arg in test_cmd
                ]
                if runner_cfg.runner_type == "pytest" and "-m" in test_cmd:
                    idx = test_cmd.index("-m")
                    test_cmd.insert(idx + 2, "--noconftest")
                logger.info("Running tests in container /testbed: %s", " ".join(test_cmd))

                _rc, stdout, stderr = await container.exec(
                    test_cmd,
                    cwd=testbed,
                    timeout=min(config.timeout_per_task, 300),
                    env=dict(runner_cfg.env),
                )
                raw = stdout + stderr
                passed, total = self._parse_test_output(raw)
                if total > 0 and passed == total:
                    status = "passed"
                    logger.info("Tests PASSED (container): %d/%d", passed, total)
                elif total > 0 and passed > 0:
                    status = "partial"
                    logger.info("Tests partial (container): %d/%d", passed, total)
                elif total > 0:
                    status = "all_failed"
                    logger.info("Tests FAILED (container): %d/%d", passed, total)
                else:
                    status = "zero_collected"
                    logger.warning("Tests not collected in container (0/0).")
                # Surface raw output for ANY non-pass outcome (not just
                # 0-collected) so partial/all_failed causes are diagnosable —
                # the JSON doesn't serialize stdout, so the log is the only
                # record of WHY tests failed.
                if status != "passed":
                    logger.warning(
                        "Raw container test output (last 1500 chars) for " "task %s [%s]:\n%s",
                        getattr(task, "task_id", "?"),
                        status,
                        raw[-1500:],
                    )
        finally:
            # Clean up patch files on the host mount; tear down the container
            # (only if we created it — a caller-supplied persistent container
            # is left running for reuse by the next verify call / final eval).
            for f in (".agent_patch.diff", ".test_patch.diff"):
                (cached_repo / f).unlink(missing_ok=True)
            if _owns_container:
                await container.stop()

        return _ContainerVerifyResult(
            passed=passed,
            total=total,
            raw_output=raw[-4000:],
            status=status,
            image=image,
            container_name=container.name,
        )

    async def _run_tests_in_container(
        self,
        task: BenchmarkTask,
        result: TaskResult,
        patch: str,
        cached_repo: Path,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run a task's tests in the official SWE-bench container (post-loop eval).

        Thin wrapper over :meth:`_verify_patch_in_container` that maps the
        parsed counts + status onto the ``TaskResult``. Raises
        :class:`victor.evaluation.container_eval.DockerUnavailable` (propagated
        to the caller for host fallback) if Docker or the image can't be used.
        """
        vr = await self._verify_patch_in_container(task, patch, cached_repo, config)
        result.runtime_image = vr.image
        result.container_id = vr.container_name
        result.eval_backend = "docker"
        result.language = getattr(task, "language", None) or "python"
        result.tests_passed = vr.passed
        result.tests_total = vr.total
        result.tests_failed = vr.total - vr.passed
        result.stdout = vr.raw_output[-2000:]
        if vr.status == "passed":
            result.status = TaskStatus.PASSED
        elif vr.status == "partial":
            result.status = TaskStatus.FAILED
            result.error_message = f"Partial pass: {vr.passed}/{vr.total}"
        elif vr.status == "all_failed":
            result.status = TaskStatus.FAILED
            result.error_message = f"All tests failed ({vr.total} total)"
        elif vr.status == "apply_failed":
            result.status = TaskStatus.FAILED
            result.error_message = f"Failed to apply patch (container): {vr.raw_output[:200]}"
        else:  # zero_collected or error
            result.status = TaskStatus.FAILED
            result.error_message = (
                "Patch applied but tests could not run in container (0 collected)."
                if vr.status == "zero_collected"
                else f"Container verify error ({vr.status})"
            )
        return result

    def _build_test_command(self, task: BenchmarkTask, repo_dir: Path) -> list:
        """Build the test command for a SWE-bench task.

        Uses framework test runner detection to choose the right runner
        (pytest, django, unittest) based on project structure.
        """
        import re

        from victor.context.test_runner import detect_test_runner

        # Extract test file paths from test_code patch or fail_to_pass
        test_files = None
        if hasattr(task, "fail_to_pass") and task.fail_to_pass:
            test_files = task.fail_to_pass
        elif task.test_code:
            extracted = re.findall(r"diff --git a/(\S+)", task.test_code)
            test_files = [f for f in extracted if "test" in f.lower()]

        # Use framework test runner detection
        config = detect_test_runner(repo_dir, test_files=test_files or None)
        logger.info(
            "Test runner detected: %s (command: %s)",
            config.runner_type,
            " ".join(config.command[:4]),
        )
        return config.command

    def _parse_test_output(self, output: str) -> tuple:
        """Parse test-runner output to extract (passed, total).

        Handles the common SWE-bench runner output formats:
          * pytest: "12 passed, 3 failed, 1 error"
          * unittest / django ``runtests.py``: "Ran 17 tests in 1.2s" + "OK"
            or "FAILED (failures=N, errors=M)"

        Returns ``(0, 0)`` only when no recognizable result line is found —
        callers treat that as "0 collected" and log the raw output so the
        cause (wrong runner, missing test file, unparseable format) is
        diagnosable instead of silent.
        """
        import re

        def _grab(pattern: str) -> int:
            m = re.search(pattern, output)
            return int(m.group(1)) if m else 0

        # pytest-style counts.
        passed = _grab(r"(\d+)\s+passed")
        failed = _grab(r"(\d+)\s+failed")
        errors = _grab(r"(\d+)\s+errors?\b")
        total = passed + failed + errors

        # unittest / django runtests.py fallback: "Ran N tests" + OK/FAILED.
        if total == 0:
            m = re.search(r"Ran\s+(\d+)\s+tests?", output)
            if m:
                total = int(m.group(1))
                # "OK" on its own line ⇒ all passed; otherwise none passed.
                passed = total if re.search(r"(?:^|\n)\s*OK\b", output) else 0

        return passed, total

    def _extract_patch(self, output: str) -> str:
        """Extract diff patch from agent output."""
        lines = output.split("\n")
        patch_lines = []
        in_patch = False

        for line in lines:
            # Start of patch
            if line.startswith("diff --git") or line.startswith("---"):
                in_patch = True

            if in_patch:
                patch_lines.append(line)

            # End of patch (heuristic)
            if in_patch and line.startswith("@@") and len(patch_lines) > 100:
                # Long patch, might have ended
                pass

        if patch_lines:
            return "\n".join(patch_lines)

        # Try to extract from code blocks
        import re

        code_blocks = re.findall(r"```(?:diff)?\n(.*?)```", output, re.DOTALL)
        for block in code_blocks:
            if "diff" in block or "---" in block or "+++" in block:
                return block.strip()

        return ""


class HumanEvalRunner(BaseBenchmarkRunner):
    """Runner for HumanEval benchmark.

    Evaluates code generation from docstrings using the official
    OpenAI HumanEval dataset from HuggingFace.

    Dataset: https://huggingface.co/datasets/openai/openai_humaneval
    Paper: "Evaluating Large Language Models Trained on Code" (2021)

    IMPORTANT: This runner loads REAL benchmark data from HuggingFace
    and executes REAL tests. Results are not simulated.
    """

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.HUMAN_EVAL

    async def load_tasks(
        self,
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Load HumanEval tasks from HuggingFace.

        Requires: pip install datasets
        """
        tasks = []

        try:
            from datasets import load_dataset

            logger.info("Loading HumanEval dataset from HuggingFace...")
            dataset = load_dataset("openai/openai_humaneval", split="test")
            logger.info(f"Loaded {len(dataset)} HumanEval problems")

            for item in dataset:
                task = BenchmarkTask(
                    task_id=item["task_id"],
                    benchmark=BenchmarkType.HUMAN_EVAL,
                    description=item["prompt"],
                    language="python",
                    prompt=item["prompt"],
                    test_code=item["test"],
                    solution=item["canonical_solution"],
                    category="code_generation",
                )
                tasks.append(task)

        except ImportError:
            logger.error("datasets library not installed. " "Install with: pip install datasets")
            raise RuntimeError(
                "Cannot load HumanEval: datasets library required. "
                "Install with: pip install datasets"
            )
        except Exception as e:
            logger.error(f"Failed to load HumanEval from HuggingFace: {e}")
            raise

        return self._filter_tasks(tasks, config)

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run a HumanEval task."""
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            generated_code=agent_output,
        )

        # Extract function name from the prompt to call the check function
        import re

        func_match = re.search(r"def\s+(\w+)\s*\(", task.prompt)
        func_name = func_match.group(1) if func_match else "solution"

        # Combine generated code with test and call the check function
        full_code = agent_output + "\n\n" + task.test_code + f"\n\ncheck({func_name})\n"

        env = TaskEnvironment(task=task, use_docker=config.use_docker)

        try:
            workspace = await env.setup()

            # Write code to file
            code_file = workspace / "solution.py"
            code_file.write_text(full_code)

            # Run tests
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(code_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30,
            )

            result.stdout = stdout.decode()
            result.stderr = stderr.decode()

            if proc.returncode == 0:
                result.status = TaskStatus.PASSED
                result.tests_passed = 1
                result.tests_total = 1
            else:
                result.status = TaskStatus.FAILED
                result.tests_failed = 1
                result.tests_total = 1

        except asyncio.TimeoutError:
            result.status = TaskStatus.TIMEOUT
        except Exception as e:
            result.status = TaskStatus.ERROR
            result.error_message = str(e)
        finally:
            await env.cleanup()

        return result


class MBPPRunner(BaseBenchmarkRunner):
    """Runner for MBPP (Mostly Basic Python Problems) benchmark.

    MBPP contains crowd-sourced Python programming problems
    designed to be solvable by entry-level programmers.

    Dataset: https://huggingface.co/datasets/google-research-datasets/mbpp
    Paper: "Program Synthesis with Large Language Models" (2021)

    IMPORTANT: This runner loads REAL benchmark data from HuggingFace
    and executes REAL tests. Results are not simulated.

    Supported splits:
    - test: Test set (500 tasks)
    - train: Training set
    - validation: Validation set
    - prompt: Few-shot prompt examples (10 tasks)
    """

    def __init__(self, split: str = "test"):
        """Initialize the MBPP runner.

        Args:
            split: Dataset split (test, train, validation, prompt)
        """
        self._split = split
        self._tasks_cache: Optional[list[BenchmarkTask]] = None

    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.MBPP

    async def load_tasks(
        self,
        config: EvaluationConfig,
    ) -> list[BenchmarkTask]:
        """Load MBPP tasks from HuggingFace.

        Requires: pip install datasets
        """
        if self._tasks_cache is not None:
            return self._filter_tasks(self._tasks_cache, config)

        tasks = []

        try:
            from datasets import load_dataset

            logger.info(f"Loading MBPP dataset (split={self._split}) from HuggingFace...")
            dataset = load_dataset(
                "google-research-datasets/mbpp",
                split=self._split,
            )
            logger.info(f"Loaded {len(dataset)} MBPP problems")

            for item in dataset:
                # MBPP format: task_id, text, code, test_list, test_setup_code, challenge_test_list
                task = BenchmarkTask(
                    task_id=str(item["task_id"]),
                    benchmark=BenchmarkType.MBPP,
                    description=item["text"],
                    language="python",
                    prompt=self._build_prompt(item),
                    test_code=self._build_test_code(item),
                    solution=item["code"],
                    category="basic_programming",
                    difficulty="easy",
                )
                tasks.append(task)

        except ImportError:
            logger.error("datasets library not installed. " "Install with: pip install datasets")
            raise RuntimeError(
                "Cannot load MBPP: datasets library required. " "Install with: pip install datasets"
            )
        except Exception as e:
            logger.error(f"Failed to load MBPP from HuggingFace: {e}")
            raise

        self._tasks_cache = tasks
        return self._filter_tasks(tasks, config)

    def _build_prompt(self, item: dict) -> str:
        """Build the prompt for the agent.

        Includes the function name the tests call, extracted from
        ``test_list`` — see :meth:`_expected_function_name` for why revealing
        it is the harness's job, not the prompt's.
        """
        text = item["text"]
        name = self._expected_function_name(item)
        name_hint = (
            f"\n\nThe test suite calls your solution as ``{name}(...)``; define "
            "it with that exact name."
            if name
            else ""
        )
        return f'''"""
{text}{name_hint}
"""
'''

    def _expected_function_name(self, item: dict) -> str | None:
        """The function name the tests call, or None if it can't be parsed.

        MBPP's task ``text`` is prose; the function name lives only in
        ``test_list``, which the runner appends *after* the agent has written
        its solution. A name the agent cannot see is unguessable, so identifier
        mismatch (``NameError``) is benchmark noise rather than a prompt-quality
        signal — an n=60 run produced 38 such failures, and the "match test
        identifiers" prompt candidate was net-negative precisely because the
        identifiers are hidden at solution time. HumanEval hands the agent the
        full signature; revealing the name here isolates logic from
        name-inference luck and leaves residual failures that prompt evolution
        can actually act on.
        """
        import re

        for test in item.get("test_list", []):
            match = re.search(r"\bassert\s+(\w+)\s*\(", test)
            if match:
                return match.group(1)
        return None

    def _build_test_code(self, item: dict) -> str:
        """Build test code from test_list."""
        tests = item.get("test_list", [])
        setup = item.get("test_setup_code", "")

        code_lines = []
        if setup:
            code_lines.append(setup)
            code_lines.append("")

        # Add assertions
        for test in tests:
            code_lines.append(test)

        # Add challenge tests if available
        challenge_tests = item.get("challenge_test_list", [])
        for test in challenge_tests:
            code_lines.append(test)

        return "\n".join(code_lines)

    async def run_task(
        self,
        task: BenchmarkTask,
        agent_output: str,
        config: EvaluationConfig,
    ) -> TaskResult:
        """Run an MBPP task by executing generated code with tests."""
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            generated_code=agent_output,
        )

        # Combine generated code with test assertions
        full_code = agent_output + "\n\n" + task.test_code

        env = TaskEnvironment(task=task, use_docker=config.use_docker)

        try:
            workspace = await env.setup()

            # Write code to file
            code_file = workspace / "solution.py"
            code_file.write_text(full_code)

            # Run tests
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(code_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30,
            )

            result.stdout = stdout.decode()
            result.stderr = stderr.decode()

            if proc.returncode == 0:
                result.status = TaskStatus.PASSED
                result.tests_passed = 1
                result.tests_total = 1
            else:
                result.status = TaskStatus.FAILED
                result.tests_failed = 1
                result.tests_total = 1
                result.error_message = result.stderr[:500] if result.stderr else "Test failed"

        except asyncio.TimeoutError:
            result.status = TaskStatus.TIMEOUT
            result.error_message = "Execution timeout (30s)"
        except Exception as e:
            result.status = TaskStatus.ERROR
            result.error_message = str(e)
        finally:
            await env.cleanup()

        return result


class ContainerTestVerifier:
    """Verifier (FEP-0018) that runs FAIL_TO_PASS tests in an official SWE-bench container.

    Wraps ``_verify_patch_in_container`` as a framework-level ``Verifier`` so the
    agentic loop's DECIDE gate can verify the agent's work — not just the
    adapter's eval-specific gate. Uses ``workspace_git_diff`` (ground truth) to
    snapshot the agent's edits.
    """

    def __init__(self, task, cached_repo, config, container=None):
        self._task = task
        self._cached_repo = Path(cached_repo)
        self._config = config
        self._container = container
        self._runner = SWEBenchRunner()

    async def verify(self, *, workspace=None, state=None):
        """Run the FAIL_TO_PASS tests in-container. Returns VerificationResult."""
        from victor.framework.verification import VerificationResult
        from victor.framework.workspace import workspace_git_diff

        repo = workspace or self._cached_repo
        patch = await workspace_git_diff(repo)
        if not patch.strip():
            return VerificationResult(0, 0, "", "no edits to verify")
        try:
            vr = await self._runner._verify_patch_in_container(
                self._task,
                patch,
                self._cached_repo,
                self._config,
                container=self._container,
            )
            return VerificationResult(
                passed=vr.passed,
                total=vr.total,
                raw_output=vr.raw_output,
                feedback=f"Verification: {vr.passed}/{vr.total} tests passed.",
            )
        except Exception as exc:
            return VerificationResult(
                passed=-1,
                total=-1,
                raw_output=str(exc),
                feedback=f"Verification error: {exc}",
            )
