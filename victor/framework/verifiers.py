# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Built-in verifiers for FEP-0018 (framework verification hook).

Provides ``LocalTestVerifier`` (pytest with a fresh runner-owned JUnit report)
and ``LintVerifier`` (process exit status) so any agent
session can verify-and-retry without benchmark-specific wiring. Both implement
the ``Verifier`` protocol from ``victor.framework.verification``.

Usage:
    from victor.framework.verifiers import LocalTestVerifier

    verifier = LocalTestVerifier()
    loop = AgenticLoop(..., verifier=verifier, max_verify_retries=2)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from victor.framework.verification import VerificationResult
from victor.framework.workspace import workspace_files_modified

logger = logging.getLogger(__name__)


class LocalTestVerifier:
    """Run pytest and require its structured test results plus process success.

    Runner detection remains shared with the test-runner service. Detected runners
    without a supported structured report fail explicitly; they are not parsed as
    prose or replaced with pytest. Use a custom Verifier for other report formats.
    """

    def __init__(
        self,
        command: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        timeout: float = 120,
    ):
        self._override_command = command
        self._override_env = env or {}
        self._timeout = timeout

    def _resolve_command(self, workspace: Path) -> tuple[list[str], dict[str, str]]:
        """Detect the test runner for this workspace, or use the override."""
        if self._override_command is not None:
            return self._override_command, self._override_env
        from victor.context.test_runner import detect_test_runner

        config = detect_test_runner(workspace)
        return list(config.command), {**config.env, **self._override_env}

    async def verify(
        self,
        *,
        workspace: Optional[Path] = None,
        state: Optional[dict] = None,
    ) -> VerificationResult:
        """Require a fresh structured report; test prose is diagnostics only."""
        if workspace is None:
            return VerificationResult(0, 0, "", "no workspace provided")
        try:
            cmd, env = self._resolve_command(workspace)
        except Exception as exc:
            return VerificationResult(
                0, 0, "", f"Test runner detection failed: {type(exc).__name__}"
            )
        if not cmd or not (
            cmd[1:3] == ["-m", "pytest"]
            or Path(cmd[0]).name in {"pytest", "pytest.exe", "py.test", "py.test.exe"}
        ):
            return VerificationResult(
                0, 0, "", "Unsupported structured test runner; supply a custom Verifier"
            )
        with tempfile.TemporaryDirectory(prefix="victor-verification-") as directory:
            report = Path(directory) / "pytest.xml"
            rc, stdout, stderr = await _run_command_async(
                [*cmd, "--junitxml=" + str(report)], workspace, env, self._timeout
            )
            raw = stdout + stderr
            try:
                passed, total = _read_pytest_report(report)
            except (OSError, ValueError, ET.ParseError) as exc:
                return VerificationResult(
                    0,
                    0,
                    raw[-4000:],
                    f"VERIFICATION FAILED: missing or invalid pytest report ({type(exc).__name__}); rc={rc}.",
                )
        if total == 0:
            return VerificationResult(0, 0, raw[-4000:], "VERIFICATION FAILED: no executed tests.")
        # Count process acceptance as one explicit check; preserve actual test
        # counts in feedback instead of pretending a nonzero exit passed tests.
        checks_passed, checks_total = passed + int(rc == 0), total + 1
        verified = checks_passed == checks_total
        feedback = (
            f"{'VERIFIED' if verified else 'VERIFICATION FAILED'}: "
            f"{checks_passed}/{checks_total} checks ({passed}/{total} tests; process rc={rc})."
        )
        if not verified:
            feedback += "\n" + raw[-2000:]
        return VerificationResult(checks_passed, checks_total, raw[-4000:], feedback)


def _read_pytest_report(path: Path) -> tuple[int, int]:
    """Validate JUnit structure/counts; skipped-only runs cannot verify work."""
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("pytest report exceeds 64 MiB")
    root = ET.fromstring(path.read_bytes())
    if root.tag not in {"testsuites", "testsuite"}:
        raise ValueError("Expected a JUnit testsuite report")
    suites = list(root) if root.tag == "testsuites" else [root]
    passed = total = 0
    aggregate = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        if suite.tag != "testsuite":
            raise ValueError("Unexpected JUnit root child")
        declared = {}
        for key in ("tests", "failures", "errors", "skipped"):
            value = suite.get(key, "")
            if not value.isascii() or not value.isdigit():
                raise ValueError("Missing or invalid JUnit count: " + key)
            declared[key] = int(value)
            aggregate[key] += declared[key]
        observed = {"failure": 0, "error": 0, "skipped": 0}
        cases = clean = 0
        for case in suite:
            if case.tag == "properties":
                _validate_junit_properties(case)
                continue
            if case.tag != "testcase" or not case.get("name"):
                raise ValueError("Expected a named JUnit test case")
            cases += 1
            statuses = set()
            for child in case:
                if child.tag == "properties":
                    _validate_junit_properties(child)
                elif child.tag in {"failure", "error", "skipped", "system-out", "system-err"}:
                    if len(child):
                        raise ValueError("Unexpected nested JUnit outcome")
                    if child.tag in observed:
                        observed[child.tag] += 1
                        statuses.add(child.tag)
                else:
                    raise ValueError("Unexpected JUnit test case child")
            failed = bool(statuses & {"failure", "error"})
            if not statuses:
                clean += 1
            if "skipped" in statuses and not failed:
                continue
            total += 1
            passed += int(not failed)
        if any(
            declared[key] != observed[tag]
            for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))
        ):
            raise ValueError("JUnit outcome counts disagree with test cases")
        # Pytest may count multiple setup/call/teardown outcomes for one case.
        # Any such case already fails acceptance; an all-pass suite must match
        # its testcase count exactly, so inflated success summaries cannot pass.
        if not cases <= declared["tests"] <= clean + sum(observed.values()):
            raise ValueError("JUnit test count disagrees with test cases")
    if root.tag == "testsuites":
        for key, expected in aggregate.items():
            value = root.get(key)
            if value is not None and (
                not value.isascii() or not value.isdigit() or int(value) != expected
            ):
                raise ValueError("JUnit aggregate count disagrees with suites: " + key)
    return passed, total


def _validate_junit_properties(node: ET.Element) -> None:
    """Metadata cannot hide nested test results or errors."""
    if any(child.tag != "property" or len(child) for child in node):
        raise ValueError("Unexpected JUnit property child")


class LintVerifier:
    """Verifier that runs a linter in the workspace.

    Selects ``cargo clippy`` for Rust, ``go vet`` for Go and ``ruff check``
    otherwise. Acceptance is one explicit process-exit check; diagnostic text
    never supplies an inferred issue count.
    """

    def __init__(
        self,
        command: Optional[list[str]] = None,
        timeout: float = 60,
    ):
        self._override_command = command
        self._timeout = timeout

    def _resolve_command(self, workspace: Path) -> list[str]:
        """Detect the appropriate linter for this workspace."""
        if self._override_command is not None:
            return self._override_command
        # Rust
        if (workspace / "Cargo.toml").exists():
            return ["cargo", "clippy", "--", "-D", "warnings"]
        # Go
        if (workspace / "go.mod").exists():
            return ["go", "vet", "./..."]
        # Python (default)
        return ["ruff", "check", "--output-format=concise"]

    async def verify(
        self,
        *,
        workspace: Optional[Path] = None,
        state: Optional[dict] = None,
    ) -> VerificationResult:
        """Run the linter; only successful process completion verifies it."""
        if workspace is None:
            return VerificationResult(0, 0, "", "no workspace provided")
        cmd = self._resolve_command(workspace)
        if not cmd:
            return VerificationResult(0, 0, "", "No lint command configured")
        rc, stdout, stderr = await _run_command_async(cmd, workspace, None, self._timeout)
        raw = stdout + stderr
        passed = int(rc == 0)
        feedback = (
            "VERIFIED: lint process exited successfully."
            if passed
            else f"VERIFICATION FAILED: lint process rc={rc}.\n{raw[-2000:]}"
        )
        return VerificationResult(passed, 1, raw[-4000:], feedback)


class LSPVerifier:
    """Verifier that checks LSP diagnostics on edited files.

    Uses the orchestrator's LSP capability (``orchestrator.lsp``) to fetch
    diagnostics — zero command execution, instant, multi-language (Python,
    TypeScript, Rust, Go, etc. via pyright/rust-analyzer/gopls). Catches
    type errors, undefined references, and syntax issues that tests may not
    cover. Unavailable LSP or no edited files yields zero checks and does not
    establish verified completion.
    """

    def __init__(self, lsp_capability: Any = None, include_warnings: bool = False):
        self._lsp = lsp_capability
        self._include_warnings = include_warnings

    async def verify(
        self,
        *,
        workspace: Optional[Path] = None,
        state: Optional[dict] = None,
    ) -> VerificationResult:
        """Check LSP diagnostics on edited files."""
        if self._lsp is None:
            return VerificationResult(0, 0, "", "LSP not available")

        # Get edited files from the workspace's git state.
        edited_files = []
        if workspace is not None:
            edited_files = await workspace_files_modified(workspace)

        if not edited_files:
            return VerificationResult(0, 0, "", "no edited files to check")

        all_errors: list[str] = []
        total_errors = 0
        for file_path in edited_files:
            try:
                diagnostics = self._lsp.get_diagnostics(str(file_path))
            except Exception:
                total_errors += 1
                all_errors.append(f"  {file_path} — LSP diagnostics unavailable")
                continue
            for diag in diagnostics:
                severity = getattr(diag, "severity", 1)
                if severity == 1 or (self._include_warnings and severity <= 2):
                    total_errors += 1
                    line = getattr(getattr(diag, "range", None), "start", None)
                    line_num = getattr(line, "line", "?") if line else "?"
                    all_errors.append(f"  {file_path}:{line_num} — {diag.message}")

        total = total_errors if total_errors > 0 else 1
        passed = 0 if total_errors > 0 else 1
        raw = "\n".join(all_errors)
        if passed == total:
            feedback = "VERIFIED: no LSP errors in edited files."
        else:
            feedback = f"VERIFICATION FAILED: {total_errors} LSP error(s).\n\n{raw}"
        logger.info("LSPVerifier: %d error(s) in %d files", total_errors, len(edited_files))
        return VerificationResult(passed, total, raw, feedback)


async def _run_command_async(
    cmd: list[str],
    workspace: Path,
    env: Optional[dict[str, str]] = None,
    timeout: float = 120,
) -> tuple[int, str, str]:
    """Run a buffered verifier with bounded cleanup and retained diagnostics.

    Return status 124 on timeout, 125 on cleanup failure, or the actual exit code.
    File output avoids inherited-pipe hangs; detached-child containment is outside
    this helper's scope. Cancellation propagates, retaining cleanup notes.
    """
    process = None
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        status = 1
        diagnostic = ""
        primary_error: BaseException | None = None
        cleanup_errors: list[str] = []
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(workspace),
                stdout=output,
                stderr=errors,
                env=env,
                start_new_session=(os.name == "posix"),
            )
            await asyncio.wait_for(process.wait(), timeout=timeout)
            status = process.returncode if process.returncode is not None else 1
        except TimeoutError:
            status = 124
            diagnostic = f"Command timed out after {timeout}s"
        except Exception as exc:
            diagnostic = f"Command failed: {type(exc).__name__}: {exc}"
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if process is not None:
                if process.returncode is None and os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except Exception as exc:
                        cleanup_errors.append("kill_group:" + type(exc).__name__)
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    except Exception as exc:
                        cleanup_errors.append("kill_process:" + type(exc).__name__)
                try:
                    await asyncio.wait_for(process.wait(), timeout=1)
                except Exception as exc:
                    cleanup_errors.append("reap_process:" + type(exc).__name__)
            if cleanup_errors:
                detail = "Cleanup failed: " + ", ".join(cleanup_errors)
                if primary_error is not None:
                    primary_error.add_note(detail)
                diagnostic += "\n" + detail
                status = 125
        for stream in (output, errors):
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 4000))
        return (
            status,
            output.read(4000).decode("utf-8", "replace"),
            errors.read(4000).decode("utf-8", "replace") + diagnostic,
        )
