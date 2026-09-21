"""Structured acceptance must agree with actual process outcomes."""

from pathlib import Path
import sys
from unittest.mock import AsyncMock

import pytest

from victor.framework import verifiers


@pytest.mark.parametrize("returncode", [0, 1])
async def test_local_test_counts_cannot_override_failed_process(tmp_path, monkeypatch, returncode):
    async def run(command, workspace, env, timeout):
        reports = [arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")]
        if reports:
            Path(reports[-1]).write_text(
                '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="ok"/></testsuite></testsuites>'
            )
        return returncode, "1 passed in 0.1s", ""

    monkeypatch.setattr(verifiers, "_run_command_async", run)
    result = await verifiers.LocalTestVerifier([sys.executable, "-m", "pytest"]).verify(
        workspace=tmp_path
    )
    assert result.is_verified is (returncode == 0)


@pytest.mark.parametrize(
    "returncode,output", [(1, ""), (2, "ruff unavailable"), (0, "progress: done")]
)
async def test_lint_acceptance_uses_process_status_not_colons(
    tmp_path, monkeypatch, returncode, output
):
    monkeypatch.setattr(
        verifiers, "_run_command_async", AsyncMock(return_value=(returncode, output, ""))
    )
    result = await verifiers.LintVerifier().verify(workspace=tmp_path)
    assert result.is_verified is (returncode == 0)


@pytest.mark.parametrize(
    "body,verified",
    [
        ("def test_ok(): assert 2 + 2 == 4\n", True),
        ("def test_bad(): assert False\n", False),
        ("import pytest\ndef test_skip(): pytest.skip('intentional')\n", False),
        ("", False),
    ],
)
async def test_actual_pytest_report_acceptance(tmp_path, body, verified):
    (tmp_path / "test_example.py").write_text(body)
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    result = await verifiers.LocalTestVerifier().verify(workspace=tmp_path)
    assert result.is_verified is verified
    if verified:
        assert (result.passed, result.total) == (2, 2)
        assert "1/1 tests; process rc=0" in result.feedback


@pytest.mark.parametrize(
    "report",
    [
        None,
        "1 passed",
        "<wrong/>",
        '<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase/></testsuite>',
    ],
)
async def test_missing_or_malformed_reports_never_fall_back_to_prose(tmp_path, monkeypatch, report):
    async def run(command, *args):
        destination = Path(
            next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml="))
        )
        assert not destination.exists()
        if report is not None:
            destination.write_text(report)
        return 0, "100 passed", ""

    monkeypatch.setattr(verifiers, "_run_command_async", run)
    result = await verifiers.LocalTestVerifier([sys.executable, "-m", "pytest"]).verify(
        workspace=tmp_path
    )
    assert not result.is_verified
    assert "missing or invalid pytest report" in result.feedback


@pytest.mark.parametrize(
    "command", [[], ["npm", "test"], ["cargo", "test"], [sys.executable, "-m", "unittest"]]
)
async def test_unsupported_report_formats_do_not_execute_or_substitute(
    tmp_path, monkeypatch, command
):
    run = AsyncMock()
    monkeypatch.setattr(verifiers, "_run_command_async", run)
    result = await verifiers.LocalTestVerifier(command).verify(workspace=tmp_path)
    assert not result.is_verified
    assert "Unsupported structured test runner" in result.feedback
    run.assert_not_called()


async def test_runner_detection_failure_is_explicit(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(
        "victor.context.test_runner.detect_test_runner", Mock(side_effect=RuntimeError("detection"))
    )
    run = AsyncMock()
    monkeypatch.setattr(verifiers, "_run_command_async", run)
    result = await verifiers.LocalTestVerifier().verify(workspace=tmp_path)
    assert not result.is_verified
    assert "detection failed: RuntimeError" in result.feedback
    run.assert_not_called()


async def test_timeout_retains_diagnostics_and_reaps_process(tmp_path):
    import os

    marker = tmp_path / "pid"
    command = [
        sys.executable,
        "-c",
        (
            "import os,time\nfrom pathlib import Path\n"
            f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
            "print('started', flush=True)\ntime.sleep(60)\n"
        ),
    ]
    status, stdout, stderr = await verifiers._run_command_async(command, tmp_path, timeout=1)
    assert status == 124
    assert "started" in stdout
    assert "timed out" in stderr
    assert marker.exists()
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)


async def test_completed_process_is_never_signalled(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    process = SimpleNamespace(returncode=0, pid=765432, wait=AsyncMock(), kill=Mock())
    kill_group = Mock()
    monkeypatch.setattr(
        verifiers.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    monkeypatch.setattr(verifiers.os, "killpg", kill_group, raising=False)
    assert (await verifiers._run_command_async(["command"], tmp_path))[0] == 0
    process.kill.assert_not_called()
    kill_group.assert_not_called()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group cleanup fault")
async def test_cancellation_survives_cleanup_fault_and_attempts_reap(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock

    cancellation = asyncio.CancelledError()
    process = SimpleNamespace(
        returncode=None, pid=765432, wait=AsyncMock(side_effect=[cancellation, None]), kill=Mock()
    )
    monkeypatch.setattr(
        verifiers.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    monkeypatch.setattr(verifiers.os, "killpg", Mock(side_effect=OSError("cannot kill group")))
    with pytest.raises(asyncio.CancelledError) as caught:
        await verifiers._run_command_async(["command"], tmp_path)
    assert caught.value is cancellation
    assert process.wait.await_count == 2
    assert "kill_group:OSError" in " ".join(cancellation.__notes__)


async def test_incomplete_cleanup_cannot_be_reported_as_success(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock

    count = 0

    async def wait():
        nonlocal count
        count += 1
        if count == 1:
            return 0
        await asyncio.Event().wait()

    process = SimpleNamespace(returncode=0, pid=765432, wait=wait, kill=Mock())
    monkeypatch.setattr(
        verifiers.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    result = await asyncio.wait_for(verifiers._run_command_async(["command"], tmp_path), 2)
    assert result[0] == 125
    assert "reap_process:TimeoutError" in result[2]


async def test_failure_is_not_hidden_by_a_conflicting_skip(tmp_path, monkeypatch):
    async def run(command, *args):
        destination = Path(
            next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml="))
        )
        destination.write_text(
            '<testsuite tests="2" failures="1" errors="0" skipped="1"><testcase name="bad"><skipped/><failure/></testcase><testcase name="ok"/></testsuite>'
        )
        return 0, "1 passed, 1 skipped", ""

    monkeypatch.setattr(verifiers, "_run_command_async", run)
    result = await verifiers.LocalTestVerifier([sys.executable, "-m", "pytest"]).verify(
        workspace=tmp_path
    )
    assert not result.is_verified
    assert "1/2 tests" in result.feedback


@pytest.mark.parametrize(
    "report",
    [
        '<testsuites><testsuite tests="2" failures="1" errors="0" skipped="0"><testcase name="ok"/></testsuite></testsuites>',
        '<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="ok"/><error message="collection failed"/></testsuite>',
        '<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="ok"><unexpected/></testcase></testsuite>',
        pytest.param(
            '<testsuites tests="2" failures="1" errors="0" skipped="0"><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="ok"/></testsuite></testsuites>',
            id="aggregate_counts",
        ),
    ],
)
async def test_contradictory_junit_trees_are_rejected(tmp_path, monkeypatch, report):
    async def run(command, *args):
        destination = Path(
            next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml="))
        )
        destination.write_text(report)
        return 0, "1 passed", ""

    monkeypatch.setattr(verifiers, "_run_command_async", run)
    result = await verifiers.LocalTestVerifier([sys.executable, "-m", "pytest"]).verify(
        workspace=tmp_path
    )
    assert not result.is_verified
    assert "invalid pytest report" in result.feedback


@pytest.mark.parametrize(
    "body,verified",
    [
        (
            "def test_ok(): pass\n@pytest.mark.skip(reason='intentional')\ndef test_skip(): pass\n",
            True,
        ),
        ("@pytest.fixture\ndef f(): raise RuntimeError('setup')\ndef test_f(f): pass\n", False),
        (
            "@pytest.fixture\ndef f():\n    yield\n    raise RuntimeError('teardown')\ndef test_f(f): pass\n",
            False,
        ),
        (
            "@pytest.fixture\ndef f():\n    yield\n    raise RuntimeError('teardown')\ndef test_f(f): assert False\n",
            False,
        ),
        (
            "@pytest.fixture\ndef f(request):\n    def fail(): raise RuntimeError('teardown')\n    request.addfinalizer(fail)\n    pytest.skip('setup skipped')\ndef test_f(f): pass\n",
            False,
        ),
        ("def test_properties(record_property): record_property('key', 'value')\n", True),
    ],
)
async def test_native_pytest_phase_reports_remain_supported(tmp_path, body, verified):
    (tmp_path / "test_phases.py").write_text("import pytest\n" + body)
    result = await verifiers.LocalTestVerifier(
        [sys.executable, "-m", "pytest", "test_phases.py", "-q"]
    ).verify(workspace=tmp_path)
    assert result.is_verified is verified
    assert "invalid pytest report" not in result.feedback


async def test_explicit_empty_lint_command_does_not_select_a_default(tmp_path, monkeypatch):
    run = AsyncMock()
    monkeypatch.setattr(verifiers, "_run_command_async", run)
    result = await verifiers.LintVerifier([]).verify(workspace=tmp_path)
    assert not result.is_verified
    assert "No lint command" in result.feedback
    run.assert_not_called()
