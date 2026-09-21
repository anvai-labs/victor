"""Check artifact acceptance independently of model-generated status claims."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.teams.types import MemberResult, TeamFormation, TeamResult

_PATH = Path(__file__).resolve().parents[3] / "scripts/validation/formation_gateway_matrix.py"
_SPEC = importlib.util.spec_from_file_location("formation_gateway_matrix", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(matrix)


@pytest.mark.parametrize(
    "fault", [None, "missing", "duplicate_session", "no_result", "constant_function"]
)
async def test_artifacts_usage_and_pytest_remain_independent(tmp_path, fault):
    case = await matrix.build_case(MagicMock(), "parallel", tmp_path, Path(sys.executable), 240)
    results = {}
    records = []
    for member in case.team._config.members:
        (tmp_path / f"{member.name}.py").write_text(f"def {member.name}(x): return x * 2\n")
        (tmp_path / f"test_{member.name}.py").write_text(
            f"from {member.name} import {member.name}\ndef test_{member.name}(): assert {member.name}(4) == 8\n"
        )
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
    if fault == "constant_function":
        (tmp_path / "first.py").write_text("def first(x): return 8\n")
    if fault == "missing":
        (tmp_path / "first.py").unlink()
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
async def test_case_always_closes_agent_when_final_checks_fail(tmp_path, monkeypatch, cancelled):
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
    args = SimpleNamespace(provider="zai", model="reference", proxy_port=18084, timeout=240)
    operation = matrix.run_case("parallel", tmp_path / "case", args, SimpleNamespace(records=[]))
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        assert (await operation)["passed"] is False
    agent.close.assert_awaited_once()
    assert (tmp_path / "case/evidence.json").exists()


async def test_matrix_report_stays_pending_until_accounting_and_cleanup_complete(
    tmp_path, monkeypatch
):
    import copy
    import json
    from scripts.validation import multiagent_gateway_live as mixed

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
    monkeypatch.setattr(matrix, "GatewayObserver", lambda *args: observer)
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
        provider="zai",
        model="reference",
        proxy_port=18084,
        timeout=240,
        gateway_state=state,
        output_dir=tmp_path / "out",
        cases=["sequential", "parallel"],
    )
    assert await matrix.main(args) == 0
    assert [report["passed"] for report in snapshots] == [False, False, True]


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
    args = SimpleNamespace(provider="zai", model="reference", proxy_port=18084, timeout=240)
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
