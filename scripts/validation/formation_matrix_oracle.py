"""Independent numeric oracle for the opt-in formation matrix (contract v2).

This runner checks the specified finite domain; it is not a sandbox for hostile
member code. It uses a separate process without gateway credentials or pytest
plugins, and verifies persisted source identities before accepting its report.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from typing import Any

CONTRACT_VERSION = 2
NUMERIC_DOMAIN = (
    "Numeric contract v2: inputs are Python ints in [-1024, 1024] or Python floats "
    "that are multiples of 0.25 in [-1024, 1024]. Return a Python int or float equal "
    "to twice the input. Booleans, sequences, non-finite values and other inputs "
    "are outside the contract. "
)
INPUTS = tuple(range(-1024, 1025)) + tuple(value / 4 for value in range(-4096, 4097))
EXPECTED_CHECKS = len(INPUTS)
CLEANUP_TIMEOUT_SECONDS = 1.0
_ORACLE_SOURCE = Path(__file__).read_bytes()
ORACLE_SHA256 = hashlib.sha256(_ORACLE_SOURCE).hexdigest()


def _evaluate(path: Path, name: str) -> dict[str, Any]:
    """Executed only in the child; emit counts and bounded counterexamples."""
    report: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "total": EXPECTED_CHECKS,
        "checked": 0,
        "passed": 0,
        "failures": [],
    }
    try:
        namespace: dict[str, Any] = {"__name__": "matrix_candidate", "__file__": str(path)}
        # No stale .pyc or member pytest/conftest can supply the implementation.
        exec(compile(path.read_bytes(), str(path), "exec"), namespace)
        function = namespace[name]
        for value in INPUTS:
            actual = function(value)
            report["checked"] += 1
            if type(actual) in (int, float) and actual == value * 2:
                report["passed"] += 1
            elif len(report["failures"]) < 8:
                report["failures"].append(
                    {"input": value, "expected": value * 2, "actual": repr(actual)[:200]}
                )
    except BaseException as exc:
        report["error_type"] = type(exc).__name__
    return report


async def _cleanup_process(process: asyncio.subprocess.Process) -> list[dict[str, str]]:
    """Stop only the owned process/group; retain bounded cleanup failures."""
    errors = []
    # Once exit is observed the PID may belong to an unrelated process/group.
    if process.returncode is None and os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception as exc:
            errors.append({"operation": "kill_group", "error_type": type(exc).__name__})
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except Exception as exc:
            errors.append({"operation": "kill_process", "error_type": type(exc).__name__})
    try:
        await asyncio.wait_for(process.wait(), CLEANUP_TIMEOUT_SECONDS)
    except Exception as exc:
        errors.append({"operation": "reap_process", "error_type": type(exc).__name__})
    return errors


async def check_numeric_oracle(
    path: Path, name: str, python: Path, *, timeout: float = 10
) -> dict[str, Any]:
    """Require the frozen oracle's full report, clean exit and unchanged artifact."""
    entry: dict[str, Any] = {
        "passed": False,
        "returncode": None,
        "contract_version": CONTRACT_VERSION,
        "expected_checks": EXPECTED_CHECKS,
        "oracle_sha256": ORACLE_SHA256,
        "artifact": str(path),
        "timeout_seconds": timeout,
        "cleanup_timeout_seconds": CLEANUP_TIMEOUT_SECONDS,
        "cleanup_errors": [],
    }
    process = None
    try:
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        entry["artifact_sha256"] = source_hash
        if Path(__file__).read_bytes() != _ORACLE_SOURCE:
            raise ValueError("Oracle source changed since harness startup")
        with tempfile.TemporaryDirectory(prefix="victor-matrix-oracle-") as temporary:
            runner = Path(temporary) / "oracle.py"
            runner.write_bytes(_ORACLE_SOURCE)
            # Files prevent a detached descendant retaining a pipe from extending
            # process.wait(). This is bounded verification, not child containment.
            with (
                (Path(temporary) / "stdout").open("w+b") as output_file,
                (Path(temporary) / "stderr").open("w+b") as error_file,
            ):
                process = await asyncio.create_subprocess_exec(
                    str(python),
                    "-I",
                    str(runner),
                    str(path.resolve()),
                    name,
                    cwd=temporary,
                    env={},
                    start_new_session=(os.name == "posix"),
                    stdout=output_file,
                    stderr=error_file,
                )
                primary_error: BaseException | None = None
                try:
                    await asyncio.wait_for(process.wait(), timeout)
                except BaseException as exc:
                    primary_error = exc
                    raise
                finally:
                    errors = await _cleanup_process(process)
                    entry["cleanup_errors"] = errors
                    if errors and primary_error is not None:
                        primary_error.add_note("Oracle cleanup failed: " + json.dumps(errors))
                output_file.seek(0)
                stdout = output_file.read(65537)
                if len(stdout) > 65536:
                    raise ValueError("Oracle report exceeds 64 KiB")
                error_file.seek(0, 2)
                error_file.seek(max(0, error_file.tell() - 2000))
                stderr = error_file.read(2000)
            entry.update(returncode=process.returncode, stderr=stderr.decode(errors="replace"))
            report = json.loads(stdout)
            entry["report"] = report
            stable = (
                runner.read_bytes() == _ORACLE_SOURCE
                and Path(__file__).read_bytes() == _ORACLE_SOURCE
                and hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
            )
            entry["source_unchanged"] = stable
            counts = {
                "total": EXPECTED_CHECKS,
                "checked": EXPECTED_CHECKS,
                "passed": EXPECTED_CHECKS,
            }
            entry["passed"] = (
                process.returncode == 0
                and stable
                and not entry["cleanup_errors"]
                and isinstance(report, dict)
                and set(report) == {"contract_version", "total", "checked", "passed", "failures"}
                and type(report["contract_version"]) is int
                and report["contract_version"] == CONTRACT_VERSION
                and all(
                    type(report[key]) is int and report[key] == value
                    for key, value in counts.items()
                )
                and report["failures"] == []
            )
    except (OSError, ValueError, TimeoutError) as exc:
        entry["error_type"] = type(exc).__name__
        if process is not None:
            entry["returncode"] = process.returncode
    return entry


if __name__ == "__main__":
    # Candidate diagnostics cannot masquerade as the oracle's structured report.
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        result = _evaluate(Path(sys.argv[1]), sys.argv[2])
    print(json.dumps(result, allow_nan=False))
    raise SystemExit(0 if result["passed"] == EXPECTED_CHECKS and "error_type" not in result else 1)
