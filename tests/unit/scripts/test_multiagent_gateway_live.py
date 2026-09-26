"""Offline acceptance tests; these are not live provider evidence."""

from __future__ import annotations

import json
import importlib.util
import asyncio
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from victor.teams.types import MemberResult

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/validation/multiagent_gateway_live.py"
_SPEC = importlib.util.spec_from_file_location("multiagent_gateway_live", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
live = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(live)


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    members = []
    for name in live.MEMBER_NAMES:
        (tmp_path / f"{name}.py").write_text(f"def {name}(x): return x * 2\n")
        (tmp_path / f"test_{name}.py").write_text(
            f"from {name} import {name}\ndef test_{name}(): assert {name}(4) == 8\n"
        )
        members.append(
            SimpleNamespace(
                member_id=name,
                metadata={
                    "session_id": f"run-{name}",
                    "usage": {"input_tokens": 15, "output_tokens": 2, "total_tokens": 17},
                },
            )
        )
    (tmp_path / "review.json").write_text(json.dumps({"verdict": "approved", "findings": []}))
    response = AsyncMock()
    response.status = 200
    response.json.return_value = {
        "run": {
            "total": {
                "tokens_in": 10,
                "tokens_out": 2,
                "cache_read_tokens": 4,
                "cache_creation_tokens": 1,
            }
        }
    }
    context = AsyncMock()
    context.__aenter__.return_value = response
    client = Mock()
    client.get.return_value = context
    pytest_run = Mock(return_value=subprocess.CompletedProcess([], 0, "7 passed", ""))
    monkeypatch.setattr(live.subprocess, "run", pytest_run)
    return members, client, pytest_run


async def collect(tmp_path, completed_run, **kwargs):
    members, client, _ = completed_run
    evidence = {"failures": []}
    await live.collect_acceptance(
        root=tmp_path,
        python=Path(".venv-codesign/bin/python"),
        mixed=True,
        results=members,
        client=client,
        gateway_url="http://localhost:18788",
        admin="private-test-token",
        evidence=evidence,
        **kwargs,
    )
    return evidence


@pytest.mark.parametrize("verdict", ["needs_work", "malformed", "missing"])
async def test_review_failure_preserves_all_usage_and_independent_tests(
    tmp_path, completed_run, verdict
):
    review = tmp_path / "review.json"
    if verdict == "missing":
        review.unlink()
    elif verdict == "malformed":
        review.write_text('{"verdict":"approved","findings":[123]}')
    else:
        review.write_text(json.dumps({"verdict": verdict, "findings": ["counterexample"]}))
    evidence = await collect(tmp_path, completed_run)
    assert any(item["check"] == "review" for item in evidence["failures"])
    assert len(evidence["usage_reconciliation"]) == 7
    assert all(item["passed"] for item in evidence["usage_reconciliation"])
    assert evidence["pytest"]["returncode"] == 0
    completed_run[2].assert_called_once()


async def test_missing_artifact_and_bad_member_usage_do_not_hide_other_results(
    tmp_path, completed_run
):
    (tmp_path / "fallbacka.py").unlink()
    completed_run[0][0].metadata["usage"]["input_tokens"] = 10
    evidence = await collect(tmp_path, completed_run)
    assert evidence["missing_deliverables"] == ["fallbacka.py"]
    assert {item["check"] for item in evidence["failures"]} == {"deliverables", "usage"}
    assert [item["passed"] for item in evidence["usage_reconciliation"]] == [False] + [True] * 6
    assert evidence["pytest"]["returncode"] == 0


@pytest.mark.parametrize("oauth_enabled", [False, True])
async def test_complete_run_reconciles_cache_inclusive_usage(
    tmp_path, completed_run, oauth_enabled
):
    auth = Mock()
    auth.request.return_value = completed_run[1].get.return_value
    evidence = await collect(tmp_path, completed_run, **({"oauth": auth} if oauth_enabled else {}))
    assert evidence["failures"] == []
    assert len(evidence["usage_reconciliation"]) == 7
    assert evidence["missing_deliverables"] == []
    assert evidence["review"] == {"verdict": "approved", "findings": []}
    if oauth_enabled:
        completed_run[1].get.assert_not_called()
        assert auth.request.call_count == 7
        assert all(call.kwargs == {"purpose": "accounting"} for call in auth.request.call_args_list)


async def test_pytest_timeout_is_failure_with_retained_usage(tmp_path, completed_run):
    completed_run[2].side_effect = subprocess.TimeoutExpired("pytest", 60)
    evidence = await collect(tmp_path, completed_run)
    assert evidence["pytest"]["timeout"] is True
    assert any(item["check"] == "pytest" for item in evidence["failures"])
    assert len(evidence["usage_reconciliation"]) == 7


@pytest.mark.parametrize(
    "phase",
    ["startup", "team", "pause", "cancelled", "opaque", "provenance", "collect_cancel", "oauth"],
)
async def test_validate_writes_failure_evidence_and_restores_process_state(
    tmp_path, monkeypatch, phase
):
    state = tmp_path / "state"
    state.mkdir()
    settings = {"url": "http://localhost:18788", "virtual_key": "private-test-key"}
    for name in ("client.json", "inferflux.json"):
        (state / name).write_text(json.dumps(settings))
    (state / "admin-token").write_text("private-test-admin")
    if phase == "oauth":
        for path in state.iterdir():
            path.unlink()
        auth = Mock(gateway_url="https://gateway")
        auth.capability.side_effect = lambda provider: "private-test-" + provider
        monkeypatch.setattr(live.GatewayOAuth, "load", Mock(return_value=auth))
    monkeypatch.setenv("SANDHI_GATEWAY_URL", "http://previous")
    monkeypatch.delenv("SANDHI_GATEWAY_VIRTUAL_KEY_ZAI", raising=False)
    monkeypatch.setattr(live.web.TCPSite, "start", AsyncMock())
    agent = Mock()
    agent.get_orchestrator.return_value.provider.extra_config = {
        "gateway": {"url": "http://localhost:18082"}
    }
    create = AsyncMock(return_value=agent)
    if phase == "startup":
        create.side_effect = RuntimeError("private-test-key must not be serialized")
    if phase == "cancelled":
        create.side_effect = asyncio.CancelledError()
    monkeypatch.setattr(live.Agent, "create", create)
    team = AsyncMock(side_effect=RuntimeError("private-test-admin must not be serialized"))
    if phase in {"pause", "opaque"}:
        member = MemberResult(member_id="writer", success=True, output="partial")
        if phase == "opaque":
            member.metadata["opaque"] = object()
        paused = SimpleNamespace(
            status="failed",
            member_results={"writer": member},
            to_dict=lambda: {"status": "failed", "member_results": {"writer": member.to_dict()}},
        )
        team.side_effect = None
        team.return_value = SimpleNamespace(
            _config=SimpleNamespace(
                members=[SimpleNamespace(id=name) for name in live.MEMBER_NAMES[:3]]
            ),
            _coordinator=Mock(),
            run=AsyncMock(return_value=paused),
        )
    monkeypatch.setattr(live.AgentTeam, "create_review_team", team)
    if phase == "provenance":
        monkeypatch.setattr(
            live.subprocess, "check_output", Mock(side_effect=OSError("git unavailable"))
        )
    if phase == "collect_cancel":
        monkeypatch.setattr(
            live, "collect_acceptance", AsyncMock(side_effect=asyncio.CancelledError())
        )
    # Keep independent pytest real: no generated tests must return nonzero.
    previous_cwd = Path.cwd()
    args = SimpleNamespace(
        output_dir=tmp_path / "runs",
        gateway_state=state,
        mixed=True,
        proxy_port=18082,
        auth_profile=tmp_path / "profile.json" if phase == "oauth" else None,
    )
    if phase in {"cancelled", "collect_cancel"}:
        with pytest.raises(asyncio.CancelledError):
            await live.validate(args)
        result = 1
    else:
        result = await live.validate(args)
    (report_path,) = (tmp_path / "runs").glob("*/evidence.json")
    serialized = report_path.read_text()
    report = json.loads(serialized)
    assert result == 1 and report["passed"] is False
    if phase != "collect_cancel":
        assert report["pytest"]["returncode"] != 0
    assert any(item["check"] == "execution" for item in report["failures"])
    assert "private-test-" not in serialized
    assert ("authentication" in report) == (phase == "oauth")
    if phase == "oauth":
        assert [call.args for call in auth.capability.call_args_list] == [("zai",), ("inferflux",)]
    assert json.loads(report_path.with_name("requests.json").read_text()) == []
    assert Path.cwd() == previous_cwd
    assert os.environ["SANDHI_GATEWAY_URL"] == "http://previous"
    assert "SANDHI_GATEWAY_VIRTUAL_KEY_ZAI" not in os.environ
    if phase in {"pause", "opaque"}:
        assert report["pipeline"]["pause"]["member_results"]["writer"]["output"] == "partial"
        assert report["usage_reconciliation"][0]["member_id"] == "writer"
        assert report["usage_reconciliation"][0]["passed"] is False
    if phase == "team":
        for name in ("writer", "reviewer", "reviser"):
            assert live.NUMERIC_DOMAIN in team.call_args.kwargs[name].goal

    if phase == "opaque":
        assert any(item["check"] == "serialization" for item in report["failures"])
    if phase == "provenance":
        assert any(item["check"] == "provenance" for item in report["failures"])
    if phase == "collect_cancel":
        assert any(
            item["check"] == "acceptance_collection" and item["error_type"] == "CancelledError"
            for item in report["failures"]
        )
