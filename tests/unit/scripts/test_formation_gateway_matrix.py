"""Check artifact acceptance independently of model-generated status claims."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from victor.teams.types import MemberResult, TeamFormation, TeamResult

_PATH = Path(__file__).resolve().parents[3] / "scripts/validation/formation_gateway_matrix.py"
_SPEC = importlib.util.spec_from_file_location("formation_gateway_matrix", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(matrix)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX detached descendant reproduction")
async def test_member_pytest_deadline_survives_inherited_output(tmp_path, monkeypatch):
    import asyncio
    from dataclasses import replace
    import os
    import signal
    import time

    case = await matrix.build_case(MagicMock(), "parallel", tmp_path, Path(sys.executable), 240)
    case = replace(
        case, artifacts={"first": ("first.py", "test_first.py")}, pytest_names=("first",)
    )
    (tmp_path / "first.py").write_text("def first(x): return 2 * x\n")
    marker = tmp_path / "child.pid"
    (tmp_path / "test_first.py").write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "def test_candidate(capfd):\n"
        "    with capfd.disabled():\n"
        "        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'], "
        "start_new_session=True)\n"
        f"    Path({str(marker)!r}).write_text(str(child.pid))\n"
        "    time.sleep(10)\n"
    )
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    original_wait_for = asyncio.wait_for

    async def short_execution_deadline(awaitable, timeout):
        return await original_wait_for(awaitable, 1 if timeout == 60 else timeout)

    monkeypatch.setattr(asyncio, "wait_for", short_execution_deadline)
    # Isolate the real member-pytest lifecycle from the separately tested oracle.
    monkeypatch.setattr(matrix, "check_numeric_oracle", AsyncMock(return_value={"passed": True}))
    started = time.monotonic()
    try:
        report = await matrix.check_case(
            case, None, tmp_path, Path(sys.executable), "reference", [], MagicMock()
        )
        elapsed = time.monotonic() - started
        assert marker.exists(), "The actual pytest test must launch its descendant"
        assert "pytest:first" in report["failures"]
        assert report["pytest"][0]["timed_out"] is True
        assert report["pytest"][0]["runner_status"] == 124
        assert report["pytest"][0]["returncode"] == -signal.SIGKILL
        assert report["pytest"][0]["cleanup_errors"] == []
        assert elapsed < 3, f"pytest cleanup exceeded its bounded grace: {elapsed:.3f}s"
    finally:
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("outcome", ["cleanup_failure", "spawn_failure", "cancelled"])
async def test_member_pytest_runner_failures_cannot_pass(tmp_path, monkeypatch, outcome):
    import asyncio
    from dataclasses import replace
    from victor.framework.verifiers import _BufferedCommandResult

    case = await matrix.build_case(MagicMock(), "parallel", tmp_path, Path(sys.executable), 240)
    case = replace(
        case, artifacts={"first": ("first.py", "test_first.py")}, pytest_names=("first",)
    )
    (tmp_path / "first.py").write_text("def first(x): return 2 * x\n")
    (tmp_path / "test_first.py").write_text("def test_first(): pass\n")
    runner = AsyncMock(
        return_value=_BufferedCommandResult(
            status=125 if outcome == "cleanup_failure" else 1,
            returncode=0 if outcome == "cleanup_failure" else None,
            timed_out=False,
            error_type=None if outcome == "cleanup_failure" else "OSError",
            cleanup_errors=("reap_process:TimeoutError",) if outcome == "cleanup_failure" else (),
            stdout="retained output",
            stderr="retained error",
            stdout_truncated=True,
            stderr_truncated=False,
        )
    )
    cancellation = asyncio.CancelledError()
    if outcome == "cancelled":
        runner.side_effect = cancellation
    monkeypatch.setattr(matrix, "_run_buffered_command", runner)
    oracle = AsyncMock(return_value={"passed": True})
    monkeypatch.setattr(matrix, "check_numeric_oracle", oracle)
    operation = matrix.check_case(
        case, None, tmp_path, Path(sys.executable), "reference", [], MagicMock()
    )
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError) as caught:
            await operation
        assert caught.value is cancellation
        oracle.assert_not_awaited()
    else:
        report = await operation
        assert "pytest:first" in report["failures"]
        entry = report["pytest"][0]
        assert entry["returncode"] == runner.return_value.returncode
        assert entry["runner_status"] == runner.return_value.status
        assert entry["cleanup_errors"] == list(runner.return_value.cleanup_errors)
        assert entry["error_type"] == runner.return_value.error_type
        assert entry["output"] == "retained output"
        assert entry["stderr"] == "retained error"
        assert entry["stdout_truncated"] is True
        assert entry["stderr_truncated"] is False
        oracle.assert_awaited_once()


@pytest.mark.parametrize(
    "fault", [None, "missing", "duplicate_session", "no_result", "constant_function"]
)
@pytest.mark.parametrize("profile", ["standard", "single-file"])
async def test_artifacts_usage_and_pytest_remain_independent(tmp_path, fault, profile):
    case = await matrix.build_case(
        MagicMock(), "parallel", tmp_path, Path(sys.executable), 240, task_profile=profile
    )
    results = {}
    records = []
    for member in case.team._config.members:
        implementation = f"def {member.name}(x): return x * 2\n"
        test = f"def test_{member.name}(): assert {member.name}(4) == 8\n"
        if profile == "standard":
            (tmp_path / f"{member.name}.py").write_text(implementation)
            implementation = f"from {member.name} import {member.name}\n"
        (tmp_path / f"test_{member.name}.py").write_text(implementation + test)
        session = "duplicated" if fault == "duplicate_session" else "session-" + member.id
        results[member.id] = MemberResult(
            member_id=member.id,
            success=True,
            output="ready",
            metadata={
                "session_id": session,
                "usage": {"input_tokens": 15, "output_tokens": 2, "total_tokens": 17},
            },
        )
        records.append(
            {
                "session_id": session,
                "run_id": session,
                "model": "reference",
                "http_status": 200,
                "elapsed_seconds": 1,
            }
        )
    result = TeamResult(True, "ready", results, TeamFormation.PARALLEL)
    observer = SimpleNamespace(
        admin_call=AsyncMock(
            return_value={
                "run": {
                    "total": {
                        "tokens_in": 10,
                        "tokens_out": 2,
                        "cache_read_tokens": 5,
                        "cache_creation_tokens": 0,
                    }
                }
            }
        )
    )
    code_file = tmp_path / ("first.py" if profile == "standard" else "test_first.py")
    if fault == "constant_function":
        code_file.write_text(code_file.read_text().replace("return x * 2", "return 8"))
    if fault == "missing":
        code_file.unlink()
    if fault == "no_result":
        result = None
    report = await matrix.check_case(
        case, result, tmp_path, Path(sys.executable), "reference", records, observer
    )
    assert report["passed"] == (fault is None)
    if fault == "constant_function":
        assert report["pytest"][0]["returncode"] == 0
        assert "semantic_oracle:first" in report["failures"]
    assert len(report["pytest"]) == 2
    assert report["pytest"][-1]["returncode"] == 0
    if fault != "no_result":
        assert len(report["usage_reconciliation"]) == 2
        assert all(item["passed"] for item in report["usage_reconciliation"])


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("profile", ["standard", "single-file"])
async def test_case_always_closes_agent_when_final_checks_fail(
    tmp_path, monkeypatch, cancelled, profile
):
    import asyncio

    agent = MagicMock()
    agent.close = AsyncMock()
    agent.get_orchestrator.return_value.provider.extra_config = {
        "gateway": {"url": "http://127.0.0.1:18084"}
    }
    monkeypatch.setattr(matrix.Agent, "create", AsyncMock(return_value=agent))
    case = await matrix.build_case(MagicMock(), "parallel", tmp_path, Path(sys.executable), 240)
    case.team.run = AsyncMock(return_value=TeamResult(True, "ready", {}, TeamFormation.PARALLEL))
    monkeypatch.setattr(matrix, "build_case", AsyncMock(return_value=case))
    error = asyncio.CancelledError() if cancelled else RuntimeError("acceptance failed")
    monkeypatch.setattr(matrix, "check_case", AsyncMock(side_effect=error))
    args = SimpleNamespace(
        provider="zai", model="reference", proxy_port=18084, timeout=240, task_profile=profile
    )
    operation = matrix.run_case("parallel", tmp_path / "case", args, SimpleNamespace(records=[]))
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        assert (await operation)["passed"] is False
    agent.close.assert_awaited_once()
    import json

    saved = json.loads((tmp_path / "case/evidence.json").read_text())
    assert saved.get("task_profile", "standard") == profile
    matrix.build_case.assert_awaited_once()
    assert matrix.build_case.await_args.kwargs["task_profile"] == profile


@pytest.mark.parametrize(
    "profile,oauth_enabled", [("standard", False), ("single-file", False), ("standard", True)]
)
async def test_matrix_report_stays_pending_until_accounting_and_cleanup_complete(
    tmp_path, monkeypatch, profile, oauth_enabled
):
    import copy
    import json
    from scripts.validation import multiagent_gateway_live as mixed
    from scripts.validation.gateway_oauth import GatewayOAuth

    state = tmp_path / "state"
    state.mkdir()
    (state / "client.json").write_text(
        json.dumps({"url": "http://gateway", "virtual_key": "private-test"})
    )
    (state / "admin-token").write_text("private-admin")
    observer = SimpleNamespace(
        records=[],
        start=AsyncMock(),
        close=AsyncMock(),
        admin_call=AsyncMock(return_value={}),
        collect=AsyncMock(return_value={"failures": []}),
    )
    factory = Mock(return_value=observer)
    monkeypatch.setattr(matrix, "GatewayObserver", factory)
    auth = SimpleNamespace(
        gateway_url="https://gateway",
        database=tmp_path / "oauth.db",
        capability=lambda provider: "loopback-capability",
    )
    monkeypatch.setattr(GatewayOAuth, "load", Mock(return_value=auth))
    monkeypatch.setattr(
        matrix,
        "run_case",
        AsyncMock(
            side_effect=[
                {"case": name, "passed": True, "elapsed_seconds": 1}
                for name in ("sequential", "parallel")
            ]
        ),
    )
    snapshots = []
    original = mixed.save_report

    def save(root, report):
        original(root, report)
        snapshots.append(copy.deepcopy(report))

    monkeypatch.setattr(mixed, "save_report", save)
    args = SimpleNamespace(
        task_profile=profile,
        provider="zai",
        model="reference",
        proxy_port=18084,
        timeout=240,
        gateway_state=state,
        output_dir=tmp_path / "out",
        cases=["sequential", "parallel"],
        auth_profile=tmp_path / "profile.json" if oauth_enabled else None,
    )
    assert await matrix.main(args) == 0
    assert factory.call_args.kwargs == ({"oauth": auth} if oauth_enabled else {})
    assert factory.call_args.args[2] == (auth.database if oauth_enabled else state / "usage.db")
    assert [report["passed"] for report in snapshots] == [False, False, True]
    for report in snapshots:
        assert report.get("task_profile", "standard") == profile
        assert ("task_profile" in report) == (profile != "standard")
        assert ("authentication" in report) == oauth_enabled


@pytest.mark.parametrize("cancelled", [False, True])
async def test_sink_failure_still_closes_agent_and_preserves_evidence(
    tmp_path, monkeypatch, cancelled
):
    import asyncio

    agent = MagicMock()
    agent.close = AsyncMock()
    agent.get_orchestrator.return_value.provider.extra_config = {
        "gateway": {"url": "http://127.0.0.1:18084"}
    }
    monkeypatch.setattr(matrix.Agent, "create", AsyncMock(return_value=agent))
    case = await matrix.build_case(MagicMock(), "parallel", tmp_path, Path(sys.executable), 240)
    case.team.run = AsyncMock(return_value=TeamResult(False, "failed", {}, TeamFormation.PARALLEL))
    monkeypatch.setattr(matrix, "build_case", AsyncMock(return_value=case))
    error = asyncio.CancelledError() if cancelled else RuntimeError("sink failure")
    monkeypatch.setattr(matrix.MemberEventSink, "close", AsyncMock(side_effect=error))
    monkeypatch.setattr(
        matrix, "check_case", AsyncMock(return_value={"failures": ["failed"], "passed": False})
    )
    args = SimpleNamespace(
        provider="zai", model="reference", proxy_port=18084, timeout=240, task_profile="standard"
    )
    operation = matrix.run_case("parallel", tmp_path / "case", args, SimpleNamespace(records=[]))
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        assert (await operation)["passed"] is False
    agent.close.assert_awaited_once()
    assert (tmp_path / "case/evidence.json").exists()


@pytest.mark.parametrize("mode", ["debate", "ensemble_judge", "ensemble_synthesizer"])
@pytest.mark.parametrize("fault", [None, "malformed", "contradiction", "runtime_mismatch"])
async def test_decision_artifact_must_satisfy_formation_contract(tmp_path, mode, fault):
    import dataclasses
    import json

    case = await matrix.build_case(MagicMock(), mode, tmp_path, Path(sys.executable), 240)
    role = "synthesizer" if mode.endswith("synthesizer") else "judge"
    case = dataclasses.replace(
        case, artifacts={role: ("decision.json",)}, pytest_names=(), isolated=False
    )
    decision = (
        {"answer": "verdict"}
        if role == "synthesizer"
        else {"selected_member_id": case.team._config.members[0].id}
    )
    if mode == "debate":
        decision["verdict"] = "verdict"
    if fault == "contradiction":
        decision["answer" if role == "synthesizer" else "selected_member_id"] = "unknown"
    if fault == "runtime_mismatch":
        if mode == "ensemble_judge":
            decision["selected_member_id"] = case.team._config.members[1].id
        else:
            decision["answer" if role == "synthesizer" else "verdict"] = "different verdict"
    (tmp_path / "decision.json").write_text(
        "not JSON" if fault == "malformed" else json.dumps(decision)
    )
    member = next(member for member in case.team._config.members if member.name == role)
    result = TeamResult(
        True,
        "verdict",
        {
            member.id: MemberResult(
                member.id,
                True,
                (
                    json.dumps({"selected_member_id": case.team._config.members[0].id})
                    if mode == "ensemble_judge"
                    else "verdict"
                ),
                metadata={
                    "session_id": "session",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    "ensemble_success": True,
                    "conversation_success": True,
                    "conversation_termination": "judge",
                },
            )
        },
        case.team._config.formation,
    )
    observer = SimpleNamespace(
        admin_call=AsyncMock(
            return_value={
                "run": {
                    "total": {
                        "tokens_in": 1,
                        "tokens_out": 1,
                        "cache_read_tokens": 0,
                        "cache_creation_tokens": 0,
                    }
                }
            }
        )
    )
    report = await matrix.check_case(
        case,
        result,
        tmp_path,
        Path(sys.executable),
        "reference",
        [
            {
                "session_id": "session",
                "run_id": "session",
                "model": "reference",
                "http_status": 200,
                "elapsed_seconds": 1,
            }
        ],
        observer,
    )
    assert report["passed"] == (fault is None)
    assert any("decision" in item for item in report["failures"]) == (fault is not None)
