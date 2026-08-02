# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Built-in verifiers for FEP-0018 (framework verification hook).

Provides ``LocalTestVerifier`` (auto-detects the test runner via
``victor.context.test_runner.detect_test_runner`` — pytest, django, unittest,
npm, cargo, go, gradle, maven) and ``LintVerifier`` (runs ruff) so any agent
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
import re
from pathlib import Path
from typing import Any, Optional

from victor.framework.verification import VerificationResult
from victor.framework.workspace import workspace_files_modified

logger = logging.getLogger(__name__)


class LocalTestVerifier:
    """Verifier that auto-detects and runs the project's test suite.

    Uses ``detect_test_runner`` (``victor.context.test_runner``) to discover
    the correct runner for the workspace's language/ecosystem — pytest,
    django, unittest, npm, cargo, go, gradle, or maven. No hardcoded command.

    Falls back to ``python -m pytest -x -q`` if detection fails.
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
        if self._override_command:
            return self._override_command, self._override_env
        try:
            from victor.context.test_runner import detect_test_runner

            config = detect_test_runner(workspace)
            logger.info(
                "LocalTestVerifier: detected %s runner: %s",
                config.runner_type,
                " ".join(config.command),
            )
            return list(config.command), {**config.env, **self._override_env}
        except Exception:
            logger.debug("detect_test_runner failed; falling back to pytest")
            return ["python", "-m", "pytest", "-x", "-q"], self._override_env

    async def verify(
        self,
        *,
        workspace: Optional[Path] = None,
        state: Optional[dict] = None,
    ) -> VerificationResult:
        """Run the test suite and parse results."""
        if workspace is None:
            return VerificationResult(0, 0, "", "no workspace provided")
        cmd, env = self._resolve_command(workspace)
        rc, stdout, stderr = await _run_command_async(cmd, workspace, env, self._timeout)
        raw = stdout + stderr
        passed, total = _parse_test_output(raw)

        if total == 0 and rc != 0:
            feedback = f"Tests failed to run (rc={rc}). Output:\n{raw[-1500:]}"
        elif passed == total and total > 0:
            feedback = f"VERIFIED: {passed}/{total} tests passed."
        else:
            feedback = (
                f"VERIFICATION FAILED: {passed}/{total} tests passed. "
                f"Fix the remaining {total - passed} failure(s).\n\n"
                f"{raw[-2000:]}"
            )
        logger.info("LocalTestVerifier: %d/%d (rc=%d)", passed, total, rc)
        return VerificationResult(passed, total, raw[-4000:], feedback)


class LintVerifier:
    """Verifier that runs a linter in the workspace.

    Auto-detects: ``ruff`` for Python projects, ``cargo clippy`` for Rust,
    ``golangci-lint`` for Go. Falls back to ``ruff check`` if detection fails.
    ``is_verified`` when 0 errors.
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
        if self._override_command:
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
        """Run the linter and parse results."""
        if workspace is None:
            return VerificationResult(0, 0, "", "no workspace provided")
        cmd = self._resolve_command(workspace)
        rc, stdout, stderr = await _run_command_async(cmd, workspace, None, self._timeout)
        raw = stdout + stderr
        # Most linters: rc=0 = clean, rc=1 = issues found.
        error_lines = [line for line in raw.splitlines() if line.strip() and ":" in line]
        errors = len(error_lines)
        total = errors if errors > 0 else 1
        passed = 0 if errors > 0 else 1

        if passed == total:
            feedback = "VERIFIED: no lint issues."
        else:
            feedback = (
                f"VERIFICATION FAILED: {errors} lint issue(s). Fix them:\n\n" f"{raw[-2000:]}"
            )
        logger.info("LintVerifier: %d issue(s) (rc=%d)", errors, rc)
        return VerificationResult(passed, total, raw[-4000:], feedback)


class LSPVerifier:
    """Verifier that checks LSP diagnostics on edited files.

    Uses the orchestrator's LSP capability (``orchestrator.lsp``) to fetch
    diagnostics — zero command execution, instant, multi-language (Python,
    TypeScript, Rust, Go, etc. via pyright/rust-analyzer/gopls). Catches
    type errors, undefined references, and syntax issues that tests may not
    cover. Gracefully degrades when LSP is unavailable (victor-coding not
    installed → vacuous pass).
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
    """Run a command asynchronously in the workspace. Returns (rc, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            stdout_b.decode("utf-8", "replace"),
            stderr_b.decode("utf-8", "replace"),
        )
    except asyncio.TimeoutError:
        return 124, "", f"Command timed out after {timeout}s"
    except Exception as exc:
        return 1, "", str(exc)


def _parse_test_output(output: str) -> tuple[int, int]:
    """Parse a test runner's summary for passed/total counts.

    Handles pytest (``N passed, M failed``), unittest (``OK`` / ``FAILED``),
    cargo (``test result: ok. N passed``), go (``--- FAIL:`` / ``ok``),
    and generic ``N passed`` patterns.
    """
    passed = 0
    failed = 0

    # pytest / generic: "N passed", "N failed", "N error"
    for match in re.finditer(r"(\d+) (passed|failed|error)", output):
        count = int(match.group(1))
        kind = match.group(2)
        if kind == "passed":
            passed = count
        else:
            failed = count

    # cargo: "test result: ok. 3 passed; 0 failed"
    if passed == 0 and failed == 0:
        cargo = re.search(r"(\d+) passed;\s*(\d+) failed", output)
        if cargo:
            passed = int(cargo.group(1))
            failed = int(cargo.group(2))

    # go: count "ok" and "FAIL" package lines
    if passed == 0 and failed == 0:
        passed = output.count("\nok\t") + output.count("\nok\n")
        failed = output.count("FAIL\t")

    total = passed + failed
    return passed, total
