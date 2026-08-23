# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for the EVR-6 offline HTIR-oracle measurement gate."""

from __future__ import annotations

from victor.evaluation.htir import (
    ArtifactEffect,
    ETCLOVGLayer,
    HTIRStep,
    HTIRTrace,
    Role,
    StepStatus,
)
from victor.evaluation.turn_auditor_eval import (
    HTIRAuditorCase,
    TurnAuditObservation,
    TurnAuditorGate,
    TurnAuditorGateVerdict,
    assess_evidence_payload,
    assess_turn_auditor,
    case_from_dict,
)
from victor.framework.per_turn_auditor import AuditVerdict


def _step(index: int, *, failed: bool = False) -> HTIRStep:
    return HTIRStep(
        index=index,
        role=Role.TOOL,
        status=StepStatus.FAILED if failed else StepStatus.OK,
        effect=ArtifactEffect.NONE,
        layer=ETCLOVGLayer.EXECUTION,
        tool_name="shell",
        summary="failed" if failed else "ok",
    )


def _case(
    task_id: str,
    *,
    family: str,
    oracle_alarm_step: int | None,
    predicted_alarm_step: int | None,
    latency_ms: float = 10.0,
) -> HTIRAuditorCase:
    trace = HTIRTrace(
        task_id=task_id,
        steps=(_step(0), _step(1), _step(2, failed=oracle_alarm_step is not None)),
    )
    observations = tuple(
        TurnAuditObservation(
            step_index=index,
            verdict=(
                AuditVerdict.ALARM if index == predicted_alarm_step else AuditVerdict.CONTINUE
            ),
            latency_ms=latency_ms,
        )
        for index in range(3)
    )
    return HTIRAuditorCase(
        trace=trace,
        observations=observations,
        oracle_alarm_step=oracle_alarm_step,
        family=family,
        oracle_source="verifier:v1",
    )


def _passing_cases() -> tuple[HTIRAuditorCase, ...]:
    cases = []
    families = ("code-fix", "docs", "file-create", "qa")
    for index in range(24):
        positive = index < 12
        cases.append(
            _case(
                f"task-{index}",
                family=families[index % len(families)],
                oracle_alarm_step=2 if positive else None,
                predicted_alarm_step=1 if positive else None,
            )
        )
    return tuple(cases)


def test_passing_gate_reports_early_detection_and_latency() -> None:
    report = assess_turn_auditor(_passing_cases(), auditor_id="ollama:qwen@sha256:abc")
    assert report.verdict is TurnAuditorGateVerdict.PASS
    assert report.passed
    assert report.metrics["precision"] == 1.0
    assert report.metrics["recall"] == 1.0
    assert report.metrics["false_alarm_rate"] == 0.0
    assert report.metrics["mean_lead_steps"] == 1.0
    assert report.metrics["p95_latency_ms"] == 10.0
    assert report.metrics["family_metrics"]["code-fix"]["recall"] == 1.0


def test_late_alarm_counts_as_false_negative() -> None:
    cases = list(_passing_cases())
    cases[0] = _case(
        "task-0",
        family="code-fix",
        oracle_alarm_step=1,
        predicted_alarm_step=2,
    )
    report = assess_turn_auditor(cases, auditor_id="judge@revision:pinned")
    assert report.metrics["false_negatives"] == 1
    assert report.metrics["late_alarm_task_ids"] == ["task-0"]


def test_quality_and_latency_regressions_hold() -> None:
    cases = list(_passing_cases())
    # One healthy false alarm makes FAR 1/12, above 5%. Two slow traces put
    # more than 5% of all prefix observations over the p95 budget.
    for index in (12, 13):
        cases[index] = _case(
            f"task-{index}",
            family="code-fix" if index == 12 else "docs",
            oracle_alarm_step=None,
            predicted_alarm_step=0 if index == 12 else None,
            latency_ms=2500.0,
        )
    report = assess_turn_auditor(cases, auditor_id="judge@revision:pinned")
    assert report.verdict is TurnAuditorGateVerdict.HOLD
    assert any("false-alarm rate" in reason for reason in report.reasons)
    assert any("p95 latency" in reason for reason in report.reasons)


def test_insufficient_distribution_and_unpinned_identity_hold() -> None:
    report = assess_turn_auditor(
        _passing_cases()[:4],
        auditor_id="mutable-tag",
        gate=TurnAuditorGate(min_precision=0.0, min_recall=0.0),
    )
    assert report.verdict is TurnAuditorGateVerdict.HOLD
    assert any("identity" in reason for reason in report.reasons)
    assert any("insufficient traces" in reason for reason in report.reasons)
    assert any("negative traces" in reason for reason in report.reasons)


def test_integrity_failures_hold_even_with_good_predictions() -> None:
    cases = list(_passing_cases())
    first = cases[0]
    cases[0] = HTIRAuditorCase(
        trace=first.trace,
        observations=first.observations[:-1],
        oracle_alarm_step=first.oracle_alarm_step,
        family=first.family,
        oracle_source="",
    )
    report = assess_turn_auditor(cases, auditor_id="judge@revision:pinned")
    assert report.verdict is TurnAuditorGateVerdict.HOLD
    assert any("oracle source" in reason for reason in report.reasons)
    assert any("cover every HTIR prefix" in reason for reason in report.reasons)


def test_non_sequential_htir_steps_hold() -> None:
    cases = list(_passing_cases())
    first = cases[0]
    bad_trace = HTIRTrace(
        task_id=first.task_id,
        steps=(first.trace.steps[0], first.trace.steps[2]),
    )
    cases[0] = HTIRAuditorCase(
        trace=bad_trace,
        observations=first.observations[:2],
        oracle_alarm_step=1,
        family=first.family,
        oracle_source=first.oracle_source,
    )
    report = assess_turn_auditor(cases, auditor_id="judge@revision:pinned")
    assert any("step indices are not sequential" in reason for reason in report.reasons)


def test_per_family_regression_cannot_hide_in_aggregate() -> None:
    cases = list(_passing_cases())
    # Miss one of three positives in a family: overall recall remains 11/12,
    # but code-fix recall falls below the 0.8 per-family gate.
    cases[0] = _case(
        "task-0",
        family="code-fix",
        oracle_alarm_step=2,
        predicted_alarm_step=None,
    )
    report = assess_turn_auditor(cases, auditor_id="judge@revision:pinned")
    assert report.metrics["recall"] > 0.8
    assert any("family 'code-fix' recall" in reason for reason in report.reasons)


def test_duplicate_task_ids_hold() -> None:
    cases = list(_passing_cases())
    cases[-1] = cases[0]
    report = assess_turn_auditor(cases, auditor_id="judge@revision:pinned")
    assert any("duplicate task ids" in reason for reason in report.reasons)


def test_case_json_round_trip_and_payload_assessment() -> None:
    cases = _passing_cases()
    parsed = case_from_dict(cases[0].to_dict())
    assert parsed == cases[0]

    report = assess_evidence_payload(
        {
            "schema_version": 1,
            "auditor_id": "ollama:qwen@sha256:abc",
            "cases": [case.to_dict() for case in cases],
        }
    )
    assert report.passed
    assert report.to_dict()["scope"].startswith("offline-prerequisite-only")


def test_unsupported_schema_is_rejected() -> None:
    try:
        assess_evidence_payload({"schema_version": 2, "cases": []})
    except ValueError as exc:
        assert "schema_version" in str(exc)
    else:  # pragma: no cover - makes the expected exception explicit
        raise AssertionError("unsupported schema must raise")
