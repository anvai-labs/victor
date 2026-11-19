# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Unit tests for the deterministic per_turn_auditor A/B battery (FOU-S4, ADR-036).

The battery certifies WIRING + COST (flag-ON consults the judge exactly where the
documented contract says, flag-OFF is a strict no-op, cost model bounded); it does
NOT decide judge quality — that needs the live flag_ab A/B, so the verdict maps to
ESCALATE_LIVE_AB_REQUIRED, pinned here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from victor.evaluation.auditor_ab import (
    AUDITOR_AB_ENV,
    AUDITOR_AB_FLAG,
    auditor_ab_corpus,
    run_auditor_ab,
)
from victor.framework.per_turn_auditor import resolve_per_turn_auditor_enabled

# ── corpus ────────────────────────────────────────────────────────────────


def test_corpus_is_deterministic_and_covers_the_contract_cells() -> None:
    a = auditor_ab_corpus()
    b = auditor_ab_corpus()
    assert [t.task_id for t in a] == [t.task_id for t in b]
    ids = {t.task_id for t in a}
    # Healthy COMPLETE turns (judge consulted), the degenerate COMPLETE (heuristic
    # alarm, judge skipped), and a non-COMPLETE pass-through.
    assert "healthy_content_complete" in ids
    assert "successful_tool_complete" in ids
    assert "degenerate_complete" in ids
    assert "non_complete_pass_through" in ids


# ── arm behavior: wiring correctness ──────────────────────────────────────


def test_off_arm_is_a_strict_noop(tmp_path: Path) -> None:
    result = run_auditor_ab(out_dir=str(tmp_path))
    baseline = result["baseline_battery"]
    assert baseline["judge_calls_total"] == 0
    assert baseline["downgrades"] == 0
    assert all(task["decision_unchanged"] for task in baseline["tasks"])


def test_on_arm_consults_the_judge_once_per_healthy_complete_turn(tmp_path: Path) -> None:
    result = run_auditor_ab(out_dir=str(tmp_path))
    by_id = {t["task_id"]: t for t in result["candidate_battery"]["tasks"]}
    for task_id in ("healthy_content_complete", "successful_tool_complete"):
        report = by_id[task_id]
        assert report["judge_calls"] == 1, task_id
        assert report["final_decision"] == "complete", task_id
        assert report["downgraded"] is False, task_id


def test_on_arm_degenerate_complete_downgrades_without_consulting_the_judge(
    tmp_path: Path,
) -> None:
    result = run_auditor_ab(out_dir=str(tmp_path))
    by_id = {t["task_id"]: t for t in result["candidate_battery"]["tasks"]}
    report = by_id["degenerate_complete"]
    # The deterministic heuristic alarms first; the judge is never spent on it.
    assert report["judge_calls"] == 0
    assert report["downgraded"] is True
    assert report["final_decision"] == "retry"


def test_on_arm_non_complete_turn_is_untouched(tmp_path: Path) -> None:
    result = run_auditor_ab(out_dir=str(tmp_path))
    by_id = {t["task_id"]: t for t in result["candidate_battery"]["tasks"]}
    report = by_id["non_complete_pass_through"]
    assert report["judge_calls"] == 0
    assert report["decision_unchanged"] is True
    assert report["final_decision"] == "retry"


def test_alarm_budget_annotates_and_allows_instead_of_trapping() -> None:
    """max_alarms exhausted → the COMPLETE is allowed with an annotation, no loop trap."""
    from types import SimpleNamespace

    from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
    from victor.framework.per_turn_auditor import PerTurnAuditor, PerTurnAuditorConfig

    auditor = PerTurnAuditor(PerTurnAuditorConfig(enabled=True, max_alarms=0))
    ev = EvaluationResult(decision=EvaluationDecision.COMPLETE, score=0.9)
    degenerate = SimpleNamespace(content="", tool_results=[])
    out = auditor.apply(ev, degenerate)
    assert out is ev  # allowed, not downgraded
    assert out.metadata.get("per_turn_auditor_exhausted") is True


# ── cost model ────────────────────────────────────────────────────────────


def test_cost_model_matches_the_documented_bounds(tmp_path: Path) -> None:
    from victor.agent.edge_turn_judge import (
        _PREFIX_STEPS,
        _PREFIX_SUMMARY_CHARS,
        _TAIL_CHARS,
    )

    result = run_auditor_ab(out_dir=str(tmp_path))
    cost = result["cost_model"]
    assert cost["judge_invocations_per_complete_turn_max"] == 1
    assert cost["off_arm_judge_calls"] == 0
    # The edge prompt is bounded: 500-char turn tail + 8×160-char prefix summaries.
    assert cost["edge_prompt_char_budget"] == _TAIL_CHARS + _PREFIX_STEPS * _PREFIX_SUMMARY_CHARS
    assert cost["edge_output_token_cap"] == 128
    assert cost["main_loop_token_overhead"] == 0  # edge model is local/token-free


# ── wiring audit (placement-not-existence guard) ──────────────────────────


def test_wiring_audit_covers_both_execution_paths() -> None:
    from victor.evaluation.auditor_ab import auditor_wiring_audit

    wiring = auditor_wiring_audit()
    assert wiring["streaming_path_resolves_flag"]
    assert wiring["streaming_path_injects_judge"]
    assert wiring["buffered_path_resolves_flag"]
    assert wiring["buffered_path_injects_judge"]
    assert wiring["loop_applies_auditor_on_evaluate_seam"]


def test_flag_resolver_env_overrides_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    settings = SimpleNamespace(agent=SimpleNamespace(per_turn_auditor=False))
    monkeypatch.setenv(AUDITOR_AB_ENV, "true")
    assert resolve_per_turn_auditor_enabled(settings) is True
    monkeypatch.setenv(AUDITOR_AB_ENV, "false")
    assert resolve_per_turn_auditor_enabled(settings) is False
    monkeypatch.delenv(AUDITOR_AB_ENV)
    assert resolve_per_turn_auditor_enabled(settings) is False


# ── artifacts + verdict ───────────────────────────────────────────────────


def test_run_auditor_ab_writes_snapshots_and_restores_env(tmp_path: Path) -> None:
    os.environ[AUDITOR_AB_ENV] = "sentinel"
    try:
        result = run_auditor_ab(out_dir=str(tmp_path))
    finally:
        restored = os.environ.get(AUDITOR_AB_ENV)
        os.environ.pop(AUDITOR_AB_ENV, None)
    assert restored == "sentinel"

    assert result["flag"] == AUDITOR_AB_FLAG
    off_path = Path(result["baseline_path"])
    on_path = Path(result["candidate_path"])
    assert off_path.name == f"{AUDITOR_AB_FLAG}_off.json"
    assert on_path.name == f"{AUDITOR_AB_FLAG}_on.json"
    assert json.loads(off_path.read_text(encoding="utf-8"))["enabled"] is False
    assert json.loads(on_path.read_text(encoding="utf-8"))["enabled"] is True
    assert result["verdict"]["verdict"] in (
        "WIRING_VERIFIED_COST_N_KNOWN",
        "ESCALATE_LIVE_AB_REQUIRED",
        "HOLD_WIRING_INCOMPLETE",
    )


def test_verdict_escalates_to_live_ab_when_wiring_verified(tmp_path: Path) -> None:
    result = run_auditor_ab(out_dir=str(tmp_path))
    verdict = result["verdict"]
    # Judge quality is not deterministically decidable → ADR-036 graduation needs the
    # live flag_ab battery; the flag stays at its current default (off) meanwhile.
    assert verdict["verdict"] == "ESCALATE_LIVE_AB_REQUIRED"
    assert verdict["should_flip_default"] is False
    assert verdict["wiring_verified"] is True
    assert "python -m victor.evaluation.flag_ab" in verdict["live_ab_protocol"]
    assert "--flag per_turn_auditor" in verdict["live_ab_protocol"]


def test_verdict_maps_broken_wiring_to_hold() -> None:
    from victor.evaluation.auditor_ab import (
        AuditorABArmResult,
        auditor_wiring_audit,
        assess_auditor_ab,
    )

    baseline = AuditorABArmResult(enabled=False)
    candidate = AuditorABArmResult(enabled=True)
    broken = dict.fromkeys(auditor_wiring_audit(), False)
    verdict = assess_auditor_ab(baseline, candidate, wiring=broken)
    assert verdict.verdict == "HOLD_WIRING_INCOMPLETE"
    assert verdict.should_flip_default is False
    assert verdict.wiring_verified is False


def test_verdict_maps_contract_violation_to_hold(tmp_path: Path) -> None:
    """Static wiring present but ON-arm behavior contradicts the contract → HOLD."""
    from victor.evaluation.auditor_ab import (
        AuditorABArmResult,
        auditor_wiring_audit,
        assess_auditor_ab,
    )

    # A candidate arm that missed its documented consultations (e.g. the judge
    # never invoked on a healthy COMPLETE turn) fails the behavioral contract.
    candidate = AuditorABArmResult(
        enabled=True,
        downgrades=0,
        contract_violations=1,
        task_reports=[
            {
                "task_id": "healthy_content_complete",
                "judge_calls": 0,
                "upstream_decision": "complete",
                "final_decision": "complete",
                "decision_unchanged": True,
                "downgraded": False,
                "expect_judge_call": True,
                "expect_downgrade": False,
            }
        ],
    )
    baseline = AuditorABArmResult(enabled=False)
    verdict = assess_auditor_ab(baseline, candidate, wiring=auditor_wiring_audit())
    assert verdict.verdict == "HOLD_WIRING_INCOMPLETE"
