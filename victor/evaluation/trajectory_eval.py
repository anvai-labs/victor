# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Trajectory-level evaluation harness (EVR-1, FEP-0008 Phase E).

Scores a *full* agent trajectory (not just its final answer) on orthogonal dimensions —
Planning, Tool-Grounding, Recovery, Refusal — and aggregates a battery of trajectories with
**confidence intervals** rather than point numbers (per arXiv:2605.10448; TRBench 2604.08178).

This module is the *measurement substrate* for the evaluation-centric runtime: it consumes the
existing :class:`~victor.evaluation.agentic_harness.AgenticExecutionTrace`, so no new instrumentation
is required. The default scorers here are **deterministic** (derived from the trace's structured
fields) — they introduce no LLM-judge dependency. Rubric / LLM-as-judge dimension scorers
(AdaRubric 2603.21362) plug into the same :class:`DimensionScorer` seam later, gated by the
judge-reliability harness (EVR-2 / ADR-011).

Each :class:`DimensionScore` carries a ``confidence`` in [0, 1] that is low when a turn/trajectory
does not engage that dimension (e.g. Refusal on a benign task), so the confidence-weighted aggregate
does not let an un-engaged dimension drag the score — and a future DimensionAwareFilter (ADR-009)
can gate only on *engaged* dimensions.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

from victor.core.utils import clamp

if TYPE_CHECKING:
    from victor.evaluation.agentic_harness import AgenticExecutionTrace


class TrajectoryDimension(str, Enum):
    """Orthogonal axes a trajectory is scored on (TRBench families)."""

    PLANNING = "planning"
    TOOL_GROUNDING = "tool_grounding"
    RECOVERY = "recovery"
    REFUSAL = "refusal"
    # ADR-010 / EVR-4: did the trajectory's completion come with a verifiable effect
    # (workspace delta / verified check) or at least read-grounded tool evidence?
    EFFECT_GROUNDING = "effect_grounding"


# Refusal markers (first-line / phrase level). Conservative — detection only; appropriateness vs the
# task is judged later by an LLM judge (EVR-3), so deterministic refusal scoring stays low-confidence.
_REFUSAL_PATTERNS = (
    "i can't",
    "i cannot",
    "i won't",
    "i will not",
    "i'm unable to",
    "i am unable to",
    "i'm not able to",
    "against policy",
    "violates",
    "cannot assist",
    "can't help with that",
)


@dataclass(frozen=True)
class DimensionScore:
    """Score for one trajectory dimension.

    ``score`` and ``confidence`` are both in [0, 1]; ``confidence`` is low when the trajectory does
    not exercise the dimension, so the aggregate can down-weight un-engaged axes.
    """

    dimension: TrajectoryDimension
    score: float
    confidence: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrajectoryScore:
    """All dimension scores for a single trajectory plus a confidence-weighted aggregate."""

    task_id: str
    dimensions: tuple[DimensionScore, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def aggregate(self) -> float:
        """Confidence-weighted mean of dimension scores (AdaRubric-style).

        Dimensions a trajectory does not engage (low confidence) contribute proportionally less, so
        e.g. an un-engaged Refusal axis does not drag a successful coding trajectory's score.
        """
        total_w = sum(d.confidence for d in self.dimensions)
        if total_w <= 0:
            return 0.0
        return sum(d.score * d.confidence for d in self.dimensions) / total_w

    def get(self, dimension: TrajectoryDimension) -> Optional[DimensionScore]:
        for d in self.dimensions:
            if d.dimension is dimension:
                return d
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "aggregate": round(self.aggregate, 4),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IntervalStat:
    """A mean with a confidence interval over ``n`` samples (dimension-agnostic)."""

    mean: float
    lower: float
    upper: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 4),
            "ci_lower": round(self.lower, 4),
            "ci_upper": round(self.upper, 4),
            "n": self.n,
        }


@dataclass(frozen=True)
class DimensionInterval:
    """Mean + confidence interval for one dimension across a battery of trajectories."""

    dimension: TrajectoryDimension
    stat: IntervalStat

    def to_dict(self) -> dict[str, Any]:
        return {"dimension": self.dimension.value, **self.stat.to_dict()}


@dataclass(frozen=True)
class BatteryResult:
    """Aggregated battery result: per-dimension and overall means with confidence intervals.

    Reporting intervals (not point numbers) is the EVR-1 discipline — a single battery run can't
    distinguish a real improvement from noise without them (arXiv:2605.10448).
    """

    scores: tuple[TrajectoryScore, ...]
    per_dimension: tuple[DimensionInterval, ...]
    overall: IntervalStat | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": len(self.scores),
            "overall": self.overall.to_dict() if self.overall else None,
            "per_dimension": [d.to_dict() for d in self.per_dimension],
        }


def mean_confidence_interval(
    values: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Return ``(mean, lower, upper)`` for ``values`` at the given confidence level.

    Dependency-light: normal approximation ``mean ± z·s/√n`` with a small-sample Student-t
    multiplier table (no scipy/numpy). For ``n < 2`` the interval collapses to the mean. Bounds are
    clamped to [0, 1] since dimension scores are normalized.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = statistics.fmean(vals)
    if n < 2:
        return mean, mean, mean
    stdev = statistics.stdev(vals)
    if stdev == 0.0:
        return mean, mean, mean
    t = _t_multiplier(n - 1, confidence)
    half = t * stdev / math.sqrt(n)
    return mean, clamp(mean - half, 0.0, 1.0), clamp(mean + half, 0.0, 1.0)


# Student-t two-sided multipliers for 95% / 90% confidence by degrees of freedom. Covers small
# samples (where the normal z=1.96 under-covers); falls back to the z value for large df.
_T_TABLE_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}
_T_TABLE_90 = {
    1: 6.314,
    2: 2.920,
    3: 2.353,
    4: 2.132,
    5: 2.015,
    6: 1.943,
    7: 1.895,
    8: 1.860,
    9: 1.833,
    10: 1.812,
    15: 1.753,
    20: 1.725,
    30: 1.697,
}


def _t_multiplier(df: int, confidence: float) -> float:
    table = _T_TABLE_90 if abs(confidence - 0.90) < 1e-6 else _T_TABLE_95
    asymptote = 1.645 if table is _T_TABLE_90 else 1.96
    if df <= 0:
        return asymptote
    if df in table:
        return table[df]
    # Use the nearest tabulated df at or below; beyond the table, approach the normal asymptote.
    keys = [k for k in table if k <= df]
    return table[max(keys)] if keys else asymptote


class DimensionScorer(Protocol):
    """Scores one dimension of a trajectory. The seam rubric/LLM-judge scorers plug into later."""

    dimension: TrajectoryDimension

    def score(self, trace: "AgenticExecutionTrace") -> DimensionScore:
        """Return this dimension's :class:`DimensionScore` for ``trace``."""
        ...


# --------------------------------------------------------------------------------------------------
# Deterministic scorers (no LLM). Derived purely from AgenticExecutionTrace structured fields.
# --------------------------------------------------------------------------------------------------


@dataclass
class ToolGroundingScorer:
    """Fraction of tool calls that succeeded; confidence grows with the number of calls."""

    dimension: TrajectoryDimension = TrajectoryDimension.TOOL_GROUNDING

    def score(self, trace: "AgenticExecutionTrace") -> DimensionScore:
        total = trace.total_tool_calls
        if total == 0:
            # No tools used — could be a legitimate Q&A turn; don't penalize, but low confidence.
            return DimensionScore(self.dimension, 0.5, 0.2, "no tool calls (dimension not engaged)")
        ok = trace.successful_tool_calls
        score = ok / total
        confidence = clamp(total / 3.0, 0.3, 1.0)
        return DimensionScore(
            self.dimension, score, confidence, f"{ok}/{total} tool calls succeeded"
        )


@dataclass
class RecoveryScorer:
    """Did the agent recover after a failed tool call?

    A failure is "recovered" when a later tool call succeeds. With no failures the dimension is not
    engaged (low confidence). Honors ``correction_metrics['recovered']`` when the trace provides it.
    """

    dimension: TrajectoryDimension = TrajectoryDimension.RECOVERY

    def score(self, trace: "AgenticExecutionTrace") -> DimensionScore:
        calls = list(getattr(trace, "tool_calls", []) or [])
        failure_idxs = [i for i, c in enumerate(calls) if not getattr(c, "success", True)]
        if not failure_idxs:
            return DimensionScore(
                self.dimension, 0.5, 0.2, "no tool failures (dimension not engaged)"
            )
        metrics = getattr(trace, "correction_metrics", {}) or {}
        if isinstance(metrics, dict) and "recovered" in metrics:
            recovered = bool(metrics["recovered"])
            return DimensionScore(
                self.dimension,
                1.0 if recovered else 0.0,
                0.9,
                f"correction_metrics.recovered={recovered}",
            )
        recovered = sum(
            1 for i in failure_idxs if any(getattr(c, "success", False) for c in calls[i + 1 :])
        )
        score = recovered / len(failure_idxs)
        confidence = clamp(len(failure_idxs) / 2.0, 0.4, 1.0)
        return DimensionScore(
            self.dimension,
            score,
            confidence,
            f"recovered {recovered}/{len(failure_idxs)} failures",
        )


@dataclass
class PlanningScorer:
    """Heuristic planning signal: explicit topology/plan events, else multi-step tool structure.

    Deterministic planning assessment is inherently weak, so confidence is capped moderate; an
    LLM-judge planning scorer (EVR-3) supersedes this when the judge clears its reliability gate.
    """

    dimension: TrajectoryDimension = TrajectoryDimension.PLANNING

    def score(self, trace: "AgenticExecutionTrace") -> DimensionScore:
        topology = list(getattr(trace, "topology_events", []) or [])
        turns = int(getattr(trace, "turns", 0) or 0)
        total_tools = trace.total_tool_calls
        if topology:
            return DimensionScore(self.dimension, 0.8, 0.5, f"{len(topology)} topology/plan events")
        if turns > 1 and total_tools > 0:
            return DimensionScore(
                self.dimension,
                0.6,
                0.35,
                f"multi-step execution ({turns} turns, {total_tools} tools)",
            )
        return DimensionScore(self.dimension, 0.4, 0.3, "no explicit planning signal")


@dataclass
class RefusalScorer:
    """Detects a refusal in the final assistant message.

    Deterministic detection only — whether a refusal is *appropriate* needs the task's intent label,
    so confidence stays low (the LLM judge decides appropriateness later). Score 1.0 = no refusal.
    """

    dimension: TrajectoryDimension = TrajectoryDimension.REFUSAL

    def score(self, trace: "AgenticExecutionTrace") -> DimensionScore:
        final = ""
        for msg in reversed(list(getattr(trace, "messages", []) or [])):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                final = str(msg["content"])
                break
        first_line = final.strip().splitlines()[0].lower() if final.strip() else ""
        refused = any(p in first_line for p in _REFUSAL_PATTERNS)
        if refused:
            return DimensionScore(
                self.dimension, 0.0, 0.4, "refusal detected (appropriateness TBD)"
            )
        return DimensionScore(self.dimension, 1.0, 0.2, "no refusal detected")


@dataclass
class EffectGroundingScorer:
    """Effect-grounded completion measurement (ADR-010 / EVR-4).

    Scores whether the trajectory's outcome is backed by a verifiable effect, using the same
    classification rules as the runtime :class:`~victor.framework.effect_gate.EffectGate`
    (:func:`~victor.framework.effect_gate.classify_tool_result`):

    - a workspace delta or verified check (or recorded ``file_edits``) → 1.0
    - only read-grounded tool evidence (a grounded claim) → 0.7, medium confidence
    - tools attempted but nothing succeeded → 0.0 (completion-without-effect)
    - no tool calls at all → 0.5, low confidence (dimension not engaged — pure conversation)

    **Not** part of :func:`default_scorers` — adding it would shift existing EVR-1 battery
    aggregates. Compose it explicitly (``TrajectoryEvaluator(default_scorers() + (EffectGroundingScorer(),))``);
    the EVR-5 acceptance oracle adopts it later.
    """

    dimension: TrajectoryDimension = TrajectoryDimension.EFFECT_GROUNDING

    def score(self, trace: "AgenticExecutionTrace") -> DimensionScore:
        from victor.framework.effect_gate import classify_tool_result

        calls = list(getattr(trace, "tool_calls", []) or [])
        effects = 0
        successes = 0
        for i, call in enumerate(calls):
            result = {
                "tool_name": getattr(call, "name", ""),
                "args": getattr(call, "arguments", {}) or {},
                "success": getattr(call, "success", False),
            }
            if result["success"]:
                successes += 1
            if classify_tool_result(result, i) is not None:
                effects += 1
        file_edits = len(getattr(trace, "file_edits", []) or [])
        if effects or file_edits:
            n = effects + file_edits
            return DimensionScore(
                self.dimension,
                1.0,
                clamp(n / 2.0, 0.4, 1.0),
                f"{n} verifiable effect(s) recorded (deltas/checks/file edits)",
            )
        if not calls:
            return DimensionScore(self.dimension, 0.5, 0.2, "no tool calls (dimension not engaged)")
        if successes:
            return DimensionScore(
                self.dimension,
                0.7,
                0.5,
                f"read-grounded only: {successes} successful tool call(s), no workspace "
                "delta or verified check",
            )
        return DimensionScore(
            self.dimension,
            0.0,
            0.6,
            f"completion-without-effect: {len(calls)} tool call(s) attempted, none succeeded",
        )


def default_scorers() -> tuple[DimensionScorer, ...]:
    """The deterministic EVR-1 scorer set (no LLM dependency).

    Intentionally excludes :class:`EffectGroundingScorer` (ADR-010) so existing battery
    aggregates are unchanged; compose it explicitly where effect grounding is measured.
    """
    return (PlanningScorer(), ToolGroundingScorer(), RecoveryScorer(), RefusalScorer())


class TrajectoryEvaluator:
    """Scores trajectories across a pluggable set of :class:`DimensionScorer`."""

    def __init__(self, scorers: Optional[Sequence[DimensionScorer]] = None) -> None:
        self._scorers: tuple[DimensionScorer, ...] = (
            tuple(scorers) if scorers else default_scorers()
        )

    def score_trajectory(self, trace: "AgenticExecutionTrace") -> TrajectoryScore:
        dims = tuple(s.score(trace) for s in self._scorers)
        return TrajectoryScore(
            task_id=getattr(trace, "task_id", ""),
            dimensions=dims,
            metadata={
                "turns": int(getattr(trace, "turns", 0) or 0),
                "tool_calls": trace.total_tool_calls,
                "benchmark": getattr(trace, "benchmark", ""),
            },
        )

    def score_battery(
        self, traces: Sequence["AgenticExecutionTrace"], confidence: float = 0.95
    ) -> BatteryResult:
        """Score many trajectories and aggregate per-dimension + overall means with CIs."""
        scores = tuple(self.score_trajectory(t) for t in traces)
        per_dim: list[DimensionInterval] = []
        for dim in TrajectoryDimension:
            vals = [d.score for s in scores for d in s.dimensions if d.dimension is dim]
            if vals:
                m, lo, hi = mean_confidence_interval(vals, confidence)
                per_dim.append(DimensionInterval(dim, IntervalStat(m, lo, hi, len(vals))))
        overall: IntervalStat | None = None
        if scores:
            agg = [s.aggregate for s in scores]
            m, lo, hi = mean_confidence_interval(agg, confidence)
            overall = IntervalStat(m, lo, hi, len(agg))
        return BatteryResult(scores=scores, per_dimension=tuple(per_dim), overall=overall)
