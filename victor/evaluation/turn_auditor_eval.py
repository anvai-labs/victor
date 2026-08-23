# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Offline HTIR-oracle measurement gate for the EVR-6 per-turn auditor.

The live auditor is intentionally opt-in.  This module scores prefix-only
CONTINUE/ALARM predictions against independently labelled HTIR traces before a
default change can be considered.  It is pure and deterministic: model calls
happen while producing the evidence artifact, never while assessing it.

Input JSON schema::

    {
      "schema_version": 1,
      "auditor_id": "ollama:qwen...@sha256:...",
      "cases": [{
        "task_id": "task-1",
        "family": "code-fix",
        "oracle_source": "verifier:v1",
        "oracle_alarm_step": 2,
        "trace": {"steps": [...]},
        "observations": [
          {"step_index": 0, "verdict": "continue", "latency_ms": 12.0},
          {"step_index": 1, "verdict": "alarm", "latency_ms": 11.0}
        ]
      }]
    }

``oracle_alarm_step`` is the first independently labelled point at which an
intervention is justified; ``null`` denotes a healthy trace.  An ALARM is an
early true positive only when it occurs at or before that step.  The gate emits
PASS/HOLD evidence only and never changes runtime configuration.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from victor.evaluation.htir import (
    ArtifactEffect,
    ETCLOVGLayer,
    HTIRStep,
    HTIRTrace,
    Role,
    StepStatus,
)
from victor.framework.per_turn_auditor import AuditVerdict


class TurnAuditorGateVerdict(str, Enum):
    """Whether an auditor clears the offline EVR-6 evidence gate."""

    PASS = "pass"
    HOLD = "hold"


@dataclass(frozen=True)
class TurnAuditObservation:
    """One auditor prediction made from ``trace.steps[:step_index + 1]`` only."""

    step_index: int
    verdict: AuditVerdict
    latency_ms: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "verdict": self.verdict.value,
            "latency_ms": round(self.latency_ms, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HTIRAuditorCase:
    """An HTIR trace, independent oracle label, and prefix predictions."""

    trace: HTIRTrace
    observations: tuple[TurnAuditObservation, ...]
    oracle_alarm_step: Optional[int]
    family: str
    oracle_source: str

    @property
    def task_id(self) -> str:
        return self.trace.task_id

    @property
    def is_positive(self) -> bool:
        return self.oracle_alarm_step is not None

    @property
    def first_predicted_alarm(self) -> Optional[int]:
        alarms = [
            observation.step_index
            for observation in self.observations
            if observation.verdict is AuditVerdict.ALARM
        ]
        return min(alarms) if alarms else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "oracle_source": self.oracle_source,
            "oracle_alarm_step": self.oracle_alarm_step,
            "trace": self.trace.to_dict(),
            "observations": [observation.to_dict() for observation in self.observations],
        }


@dataclass(frozen=True)
class TurnAuditorGate:
    """Pre-registered evidence, quality, and latency thresholds."""

    min_traces: int = 24
    min_positive_traces: int = 8
    min_negative_traces: int = 8
    min_families: int = 4
    min_traces_per_family: int = 4
    min_precision: float = 0.80
    min_recall: float = 0.80
    max_false_alarm_rate: float = 0.05
    max_p95_latency_ms: float = 2000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_traces": self.min_traces,
            "min_positive_traces": self.min_positive_traces,
            "min_negative_traces": self.min_negative_traces,
            "min_families": self.min_families,
            "min_traces_per_family": self.min_traces_per_family,
            "min_precision": self.min_precision,
            "min_recall": self.min_recall,
            "max_false_alarm_rate": self.max_false_alarm_rate,
            "max_p95_latency_ms": self.max_p95_latency_ms,
        }


@dataclass(frozen=True)
class TurnAuditorReport:
    """Serializable EVR-6 offline decision."""

    verdict: TurnAuditorGateVerdict
    auditor_id: str
    gate: TurnAuditorGate
    metrics: dict[str, Any]
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return self.verdict is TurnAuditorGateVerdict.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "verdict": self.verdict.value,
            "passed": self.passed,
            "auditor_id": self.auditor_id,
            "gate": self.gate.to_dict(),
            "metrics": self.metrics,
            "reasons": list(self.reasons),
            "scope": "offline-prerequisite-only; does not authorize a default flip",
        }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _identity_is_pinned(identity: str) -> bool:
    """Require an explicit immutable/versioned suffix (``name@revision``)."""
    name, separator, revision = identity.strip().rpartition("@")
    return bool(separator and name.strip() and revision.strip())


def _percentile_95(values: Sequence[float]) -> float:
    """Return the deterministic nearest-rank p95 (zero for no observations)."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


def _integrity_errors(cases: Sequence[HTIRAuditorCase]) -> list[str]:
    errors: list[str] = []
    task_ids = [case.task_id for case in cases]
    if len(set(task_ids)) != len(task_ids):
        errors.append("duplicate task ids in evidence")

    for case in cases:
        prefix = f"{case.task_id or '<missing-task-id>'}: "
        if not case.task_id:
            errors.append(prefix + "missing task id")
        if not case.family.strip():
            errors.append(prefix + "missing task family")
        if not case.oracle_source.strip():
            errors.append(prefix + "missing independent oracle source")
        n_steps = len(case.trace.steps)
        if n_steps == 0:
            errors.append(prefix + "empty HTIR trace")
        if [step.index for step in case.trace.steps] != list(range(n_steps)):
            errors.append(prefix + "HTIR step indices are not sequential")
        if case.oracle_alarm_step is not None and not 0 <= case.oracle_alarm_step < n_steps:
            errors.append(prefix + "oracle alarm step is outside the HTIR trace")

        indices = [observation.step_index for observation in case.observations]
        if len(indices) != len(set(indices)):
            errors.append(prefix + "duplicate prefix observations")
        if indices != sorted(indices):
            errors.append(prefix + "prefix observations are not ordered")
        if set(indices) != set(range(n_steps)):
            errors.append(prefix + "predictions do not cover every HTIR prefix")
        if any(observation.latency_ms < 0 for observation in case.observations):
            errors.append(prefix + "negative latency")
    return errors


def assess_turn_auditor(
    cases: Sequence[HTIRAuditorCase],
    *,
    auditor_id: str,
    gate: Optional[TurnAuditorGate] = None,
) -> TurnAuditorReport:
    """Score prefix predictions against independent HTIR oracle labels."""
    resolved_gate = gate or TurnAuditorGate()
    cases = tuple(cases)
    positives = tuple(case for case in cases if case.is_positive)
    negatives = tuple(case for case in cases if not case.is_positive)

    true_positives = 0
    false_negatives = 0
    false_positives = 0
    lead_steps: list[int] = []
    late_alarm_task_ids: list[str] = []
    missed_task_ids: list[str] = []
    false_alarm_task_ids: list[str] = []

    for case in positives:
        predicted = case.first_predicted_alarm
        assert case.oracle_alarm_step is not None
        if predicted is not None and predicted <= case.oracle_alarm_step:
            true_positives += 1
            lead_steps.append(case.oracle_alarm_step - predicted)
        else:
            false_negatives += 1
            if predicted is None:
                missed_task_ids.append(case.task_id)
            else:
                late_alarm_task_ids.append(case.task_id)
    for case in negatives:
        if case.first_predicted_alarm is not None:
            false_positives += 1
            false_alarm_task_ids.append(case.task_id)

    latencies = [observation.latency_ms for case in cases for observation in case.observations]
    family_counts: dict[str, int] = {}
    for case in cases:
        family_counts[case.family] = family_counts.get(case.family, 0) + 1

    family_metrics: dict[str, dict[str, Any]] = {}
    for family in sorted(family_counts):
        family_positive = tuple(case for case in positives if case.family == family)
        family_negative = tuple(case for case in negatives if case.family == family)
        family_true_positive = sum(
            case.first_predicted_alarm is not None
            and case.oracle_alarm_step is not None
            and case.first_predicted_alarm <= case.oracle_alarm_step
            for case in family_positive
        )
        family_false_positive = sum(
            case.first_predicted_alarm is not None for case in family_negative
        )
        family_metrics[family] = {
            "n": family_counts[family],
            "n_positive": len(family_positive),
            "n_negative": len(family_negative),
            "true_positives": family_true_positive,
            "false_positives": family_false_positive,
            "recall": round(_rate(family_true_positive, len(family_positive)), 6),
            "false_alarm_rate": round(_rate(family_false_positive, len(family_negative)), 6),
        }

    precision = _rate(true_positives, true_positives + false_positives)
    recall = _rate(true_positives, len(positives))
    false_alarm_rate = _rate(false_positives, len(negatives))
    p95_latency_ms = _percentile_95(latencies)
    metrics: dict[str, Any] = {
        "n_traces": len(cases),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "n_observations": len(latencies),
        "family_counts": dict(sorted(family_counts.items())),
        "family_metrics": family_metrics,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "false_alarm_rate": round(false_alarm_rate, 6),
        "mean_lead_steps": round(sum(lead_steps) / len(lead_steps), 6) if lead_steps else 0.0,
        "p95_latency_ms": round(p95_latency_ms, 4),
        "missed_task_ids": missed_task_ids,
        "late_alarm_task_ids": late_alarm_task_ids,
        "false_alarm_task_ids": false_alarm_task_ids,
    }

    reasons = _integrity_errors(cases)
    if not _identity_is_pinned(auditor_id):
        reasons.append("auditor identity is not pinned as name@revision")
    if len(cases) < resolved_gate.min_traces:
        reasons.append(f"insufficient traces: n={len(cases)}, require {resolved_gate.min_traces}")
    if len(positives) < resolved_gate.min_positive_traces:
        reasons.append(
            "insufficient positive traces: "
            f"n={len(positives)}, require {resolved_gate.min_positive_traces}"
        )
    if len(negatives) < resolved_gate.min_negative_traces:
        reasons.append(
            "insufficient negative traces: "
            f"n={len(negatives)}, require {resolved_gate.min_negative_traces}"
        )
    if len(family_counts) < resolved_gate.min_families:
        reasons.append(
            f"insufficient family coverage: {len(family_counts)}, "
            f"require {resolved_gate.min_families}"
        )
    thin_families = {
        family: count
        for family, count in family_counts.items()
        if count < resolved_gate.min_traces_per_family
    }
    if thin_families:
        reasons.append(
            "insufficient per-family evidence: "
            + ", ".join(f"{family}={count}" for family, count in sorted(thin_families.items()))
        )
    for family, family_result in family_metrics.items():
        if family_result["n_positive"] == 0 or family_result["n_negative"] == 0:
            reasons.append(f"family {family!r} lacks both positive and negative oracle traces")
            continue
        if family_result["recall"] < resolved_gate.min_recall:
            reasons.append(
                f"family {family!r} recall below gate: "
                f"{family_result['recall']:.3f} < {resolved_gate.min_recall:.3f}"
            )
        if family_result["false_alarm_rate"] > resolved_gate.max_false_alarm_rate:
            reasons.append(
                f"family {family!r} false-alarm rate above gate: "
                f"{family_result['false_alarm_rate']:.3f} > "
                f"{resolved_gate.max_false_alarm_rate:.3f}"
            )
    if precision < resolved_gate.min_precision:
        reasons.append(f"precision below gate: {precision:.3f} < {resolved_gate.min_precision:.3f}")
    if recall < resolved_gate.min_recall:
        reasons.append(f"recall below gate: {recall:.3f} < {resolved_gate.min_recall:.3f}")
    if false_alarm_rate > resolved_gate.max_false_alarm_rate:
        reasons.append(
            "false-alarm rate above gate: "
            f"{false_alarm_rate:.3f} > {resolved_gate.max_false_alarm_rate:.3f}"
        )
    if p95_latency_ms > resolved_gate.max_p95_latency_ms:
        reasons.append(
            "p95 latency above gate: "
            f"{p95_latency_ms:.1f}ms > {resolved_gate.max_p95_latency_ms:.1f}ms"
        )

    verdict = TurnAuditorGateVerdict.HOLD if reasons else TurnAuditorGateVerdict.PASS
    return TurnAuditorReport(verdict, auditor_id, resolved_gate, metrics, tuple(reasons))


def _trace_from_dict(task_id: str, data: Mapping[str, Any]) -> HTIRTrace:
    steps = tuple(
        HTIRStep(
            index=int(step["index"]),
            role=Role(step["role"]),
            status=StepStatus(step["status"]),
            effect=ArtifactEffect(step["effect"]),
            layer=ETCLOVGLayer(step["layer"]),
            tool_name=str(step.get("tool_name", "")),
            summary=str(step.get("summary", "")),
        )
        for step in data.get("steps", [])
    )
    return HTIRTrace(
        task_id=task_id,
        steps=steps,
        session_id=str(data.get("session_id", "")),
        benchmark=str(data.get("benchmark", "")),
        metadata=dict(data.get("metadata", {})),
    )


def case_from_dict(data: Mapping[str, Any]) -> HTIRAuditorCase:
    """Parse one evidence case from the stable JSON representation."""
    task_id = str(data.get("task_id", ""))
    trace = _trace_from_dict(task_id, data.get("trace", {}))
    observations = tuple(
        TurnAuditObservation(
            step_index=int(observation["step_index"]),
            verdict=AuditVerdict(observation["verdict"]),
            latency_ms=float(observation.get("latency_ms", 0.0)),
            reason=str(observation.get("reason", "")),
        )
        for observation in data.get("observations", [])
    )
    alarm_step = data.get("oracle_alarm_step")
    return HTIRAuditorCase(
        trace=trace,
        observations=observations,
        oracle_alarm_step=int(alarm_step) if alarm_step is not None else None,
        family=str(data.get("family", "")),
        oracle_source=str(data.get("oracle_source", "")),
    )


def assess_evidence_payload(
    payload: Mapping[str, Any], *, gate: Optional[TurnAuditorGate] = None
) -> TurnAuditorReport:
    """Parse and assess an EVR-6 evidence payload."""
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported EVR-6 evidence schema_version")
    cases = tuple(case_from_dict(case) for case in payload.get("cases", []))
    return assess_turn_auditor(
        cases,
        auditor_id=str(payload.get("auditor_id", "")),
        gate=gate,
    )


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", help="Path to the HTIR-oracle evidence JSON")
    parser.add_argument("--output", help="Optional path for the gate report JSON")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    report = assess_evidence_payload(payload)
    rendered = json.dumps(report.to_dict(), indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
