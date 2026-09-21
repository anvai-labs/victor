"""Qualify the independent oracle with semantic mutants and process failures."""

import asyncio
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from scripts.validation import formation_matrix_oracle as oracle


@pytest.mark.parametrize(
    "expression,passed",
    [
        ("x * 2", True),
        ("abs(x) * 2", False),
        ("int(x) * 2", False),
        ("True", False),
        ("float('nan')", False),
    ],
)
async def test_finite_domain_rejects_semantic_mutants(tmp_path, expression, passed):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(f"def double(x): return {expression}\n")
    result = await oracle.check_numeric_oracle(candidate, "double", Path(sys.executable))
    assert result["passed"] is passed
    assert result["report"]["checked"] == 10242
    assert result["source_unchanged"]
    assert (result["report"]["failures"] == []) is passed
    assert len(result["artifact_sha256"]) == len(result["oracle_sha256"]) == 64


@pytest.mark.parametrize("exit_method", ["raise SystemExit(0)", "os._exit(0)"])
async def test_candidate_exit_cannot_become_verified(tmp_path, exit_method):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(f"import os\n{exit_method}\n")
    result = await oracle.check_numeric_oracle(candidate, "double", Path(sys.executable))
    assert result["passed"] is False
    if exit_method.startswith("raise"):
        assert result["report"]["error_type"] == "SystemExit"
    else:
        assert result["returncode"] == 0
        assert result["error_type"] == "JSONDecodeError"


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "nonzero",
        "empty",
        "malformed",
        "incomplete",
        "counts_type",
        "runner_changed",
        "artifact_changed",
        "oversized",
    ],
)
async def test_reports_require_complete_counts_clean_exit_and_source_identity(
    tmp_path, monkeypatch, fault
):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def double(x): return x * 2\n")
    report = {
        "contract_version": oracle.CONTRACT_VERSION,
        "total": oracle.EXPECTED_CHECKS,
        "checked": oracle.EXPECTED_CHECKS,
        "passed": oracle.EXPECTED_CHECKS,
        "failures": [],
    }
    if fault == "incomplete":
        report["checked"] = 0
    if fault == "counts_type":
        report["total"] = float(oracle.EXPECTED_CHECKS)
    output = json.dumps(report).encode()
    if fault in {"empty", "malformed"}:
        output = b"" if fault == "empty" else b"VERIFIED: 10242 passed"
    if fault == "oversized":
        output = b"x" * 65537
    process = SimpleNamespace(
        returncode=1 if fault == "nonzero" else 0,
        pid=765432,
        wait=AsyncMock(),
    )
    runner = None

    async def spawn(*args, **kwargs):
        nonlocal runner
        runner = Path(args[2])
        assert args[1] == "-I"
        assert kwargs["env"] == {}
        assert runner.read_bytes() == oracle._ORACLE_SOURCE
        kwargs["stdout"].write(output)
        kwargs["stdout"].flush()
        if fault == "runner_changed":
            runner.write_text("changed oracle")
        if fault == "artifact_changed":
            candidate.write_text("changed artifact")
        return process

    monkeypatch.setattr(oracle.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(
        oracle.os,
        "killpg",
        Mock(),
        raising=False,
    )
    result = await oracle.check_numeric_oracle(candidate, "double", Path(sys.executable))
    assert result["passed"] is (fault is None)
    assert not runner.exists()
    assert process.wait.await_count == 2


@pytest.mark.parametrize("cancelled", [False, True])
async def test_timeout_and_cancellation_reap_owned_process(tmp_path, monkeypatch, cancelled):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def double(x): return x * 2\n")
    failure = asyncio.CancelledError() if cancelled else TimeoutError()
    process = SimpleNamespace(
        returncode=None,
        pid=765432,
        wait=AsyncMock(side_effect=[failure, None]),
        kill=Mock(),
    )
    kill_group = Mock()
    monkeypatch.setattr(oracle.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(oracle.os, "killpg", kill_group, raising=False)
    operation = oracle.check_numeric_oracle(candidate, "double", Path(sys.executable))
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        result = await operation
        assert not result["passed"]
        assert result["error_type"] == "TimeoutError"
    assert process.wait.await_count == 2
    if os.name == "posix":
        kill_group.assert_called_once()
    else:
        process.kill.assert_called_once()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process liveness probe")
async def test_real_timeout_reaps_candidate_process(tmp_path):
    marker = tmp_path / "candidate.pid"
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "import os, time\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\ndef double(x): return x * 2\n"
    )
    result = await oracle.check_numeric_oracle(candidate, "double", Path(sys.executable), timeout=1)
    assert result["passed"] is False
    assert result["error_type"] == "TimeoutError"
    assert result["returncode"] == -9
    assert marker.exists(), "Candidate must have started before testing cleanup"
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
async def test_cleanup_fault_preserves_cancellation_and_attempts_reap(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def double(x): return x * 2\n")
    cancellation = asyncio.CancelledError()
    process = SimpleNamespace(
        returncode=None,
        pid=765432,
        wait=AsyncMock(side_effect=[cancellation, None]),
        kill=Mock(),
    )
    monkeypatch.setattr(oracle.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(oracle.os, "killpg", Mock(side_effect=OSError("group cleanup failed")))
    with pytest.raises(asyncio.CancelledError) as caught:
        await oracle.check_numeric_oracle(candidate, "double", Path(sys.executable))
    assert caught.value is cancellation
    assert process.wait.await_count == 2
    assert any("OSError" in note for note in cancellation.__notes__)


@pytest.mark.skipif(os.name != "posix", reason="POSIX detached process probe")
async def test_detached_child_cannot_extend_oracle_deadline(tmp_path):
    import signal
    import time

    marker = tmp_path / "detached.pid"
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'], start_new_session=True)\n"
        f"Path({str(marker)!r}).write_text(str(child.pid))\n"
        "time.sleep(10)\n"
    )
    started = time.monotonic()
    try:
        result = await oracle.check_numeric_oracle(
            candidate, "double", Path(sys.executable), timeout=0.5
        )
        assert not result["passed"]
        assert result["error_type"] == "TimeoutError"
        assert marker.exists(), "Detached child must have started to test pipe ownership"
        assert time.monotonic() - started < 1.7
    finally:
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


async def test_incomplete_reaping_is_bounded_and_recorded(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def double(x): return x * 2\n")
    waits = 0

    async def wait():
        nonlocal waits
        waits += 1
        if waits == 1:
            raise TimeoutError()
        await asyncio.Event().wait()

    process = SimpleNamespace(returncode=None, pid=765432, wait=wait, kill=Mock())
    monkeypatch.setattr(oracle.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(oracle.os, "killpg", Mock(), raising=False)
    monkeypatch.setattr(oracle, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    result = await asyncio.wait_for(
        oracle.check_numeric_oracle(candidate, "double", Path(sys.executable)), 1
    )
    assert not result["passed"]
    assert result["error_type"] == "TimeoutError"
    assert result["cleanup_errors"] == [{"operation": "reap_process", "error_type": "TimeoutError"}]


async def test_completed_process_never_signals_a_recycled_group(monkeypatch):
    process = SimpleNamespace(returncode=0, pid=765432, wait=AsyncMock(), kill=Mock())
    kill_group = Mock()
    monkeypatch.setattr(oracle.os, "killpg", kill_group, raising=False)
    assert await oracle._cleanup_process(process) == []
    kill_group.assert_not_called()
    process.kill.assert_not_called()
    process.wait.assert_awaited_once()
