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

"""Regression-gated harness acceptance oracle (EVR-5, ADR-012).

The named gate the ADR asks for: *any* change to prompts (GEPA/MIPROv2 output), the loop, the
completion strategy, or recovery must clear this oracle before it ships. The oracle compares a
**candidate** harness against a **baseline** on two batteries and returns a single verdict:

1. **Trajectory battery** (EVR-1) — a candidate must show *no unacceptable regression* in the
   confidence-weighted aggregate, judged with the batteries' confidence intervals so noise is not
   mistaken for a real drop (arXiv:2605.10448). Consumes the existing
   :class:`~victor.evaluation.trajectory_eval.BatteryResult`.
2. **Characterization battery** (FEP-0007) — the parity/characterization transcripts must be
   **byte-stable-or-justified**: an unexplained change is a rejection; a change with a written
   rationale is accepted and recorded (the FEP-0007 discipline made explicit).

Results are reported at **(model, harness-config)** granularity — Harness-Bench's central finding is
that capability is a property of the *(model, harness)* pair, not the model alone (arXiv:2605.27922),
so a verdict without its config is meaningless.

This module is the pure *decision engine*: it takes already-computed battery results and renders the
verdict. Producing those results from live runs and wiring the verdict into CI as the promotion gate
is the follow-on (ADR-012 prong 3); keeping the decision logic pure keeps it unit-testable and
deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from victor.evaluation.trajectory_eval import (
    BatteryResult,
    IntervalStat,
    TrajectoryDimension,
)


@dataclass(frozen=True)
class HarnessConfig:
    """Identifies the *(model, harness-config)* a battery result was produced under."""

    model: str
    completion_strategy: str = "enhanced"
    effect_gate: bool = False
    label: str = ""

    def resolved_label(self) -> str:
        """A stable, human-readable identity for reporting/grouping."""
        parts = [self.model, self.completion_strategy, f"effect_gate={self.effect_gate}"]
        if self.label:
            parts.append(self.label)
        return " · ".join(parts)


class AcceptanceVerdict(str, Enum):
    """The oracle's decision for one *(model, harness-config)* candidate."""

    ACCEPT = "accept"  # no regression; within tolerance / not statistically significant
    ACCEPT_JUSTIFIED = "accept_justified"  # characterization delta present but justified
    REJECT_REGRESSION = "reject_regression"  # significant trajectory regression
    REJECT_CHARACTERIZATION = "reject_characterization"  # unjustified characterization delta
    INSUFFICIENT_DATA = "insufficient_data"  # too few samples to gate on noise


@dataclass(frozen=True)
class DimensionDelta:
    """Baseline→candidate movement on one trajectory dimension."""

    dimension: TrajectoryDimension
    baseline_mean: float
    candidate_mean: float
    regressed: bool

    @property
    def delta(self) -> float:
        return self.candidate_mean - self.baseline_mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "baseline_mean": round(self.baseline_mean, 4),
            "candidate_mean": round(self.candidate_mean, 4),
            "delta": round(self.delta, 4),
            "regressed": self.regressed,
        }


@dataclass(frozen=True)
class CharacterizationDelta:
    """Outcome of the byte-stability battery: identical, or a set of changed transcripts."""

    stable: bool
    changed: tuple[str, ...] = ()
    justification: str = ""

    @property
    def justified(self) -> bool:
        return bool(self.justification.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable": self.stable,
            "changed": list(self.changed),
            "justified": self.justified,
            "justification": self.justification,
        }


def characterization_from_signatures(
    baseline: Mapping[str, str],
    candidate: Mapping[str, str],
    *,
    justification: str = "",
) -> CharacterizationDelta:
    """Diff two ``name → transcript-signature`` maps into a :class:`CharacterizationDelta`.

    A signature is any stable digest of a characterization transcript (e.g. a hash). Keys present in
    one map but not the other count as changed, so an added/removed scenario is not silently stable.
    """
    changed = sorted(
        key for key in set(baseline) | set(candidate) if baseline.get(key) != candidate.get(key)
    )
    return CharacterizationDelta(
        stable=not changed, changed=tuple(changed), justification=justification
    )


@dataclass(frozen=True)
class AcceptanceReport:
    """The full, serializable verdict for one candidate at a *(model, harness-config)*."""

    config: HarnessConfig
    verdict: AcceptanceVerdict
    overall_baseline: Optional[IntervalStat]
    overall_candidate: Optional[IntervalStat]
    dimension_deltas: tuple[DimensionDelta, ...]
    characterization: Optional[CharacterizationDelta]
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def accepted(self) -> bool:
        return self.verdict in (AcceptanceVerdict.ACCEPT, AcceptanceVerdict.ACCEPT_JUSTIFIED)

    @property
    def overall_delta(self) -> float:
        if self.overall_baseline is None or self.overall_candidate is None:
            return 0.0
        return self.overall_candidate.mean - self.overall_baseline.mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.resolved_label(),
            "verdict": self.verdict.value,
            "accepted": self.accepted,
            "overall_baseline": self.overall_baseline.to_dict() if self.overall_baseline else None,
            "overall_candidate": (
                self.overall_candidate.to_dict() if self.overall_candidate else None
            ),
            "overall_delta": round(self.overall_delta, 4),
            "dimension_deltas": [d.to_dict() for d in self.dimension_deltas],
            "characterization": self.characterization.to_dict() if self.characterization else None,
            "reasons": list(self.reasons),
        }


class HarnessAcceptanceOracle:
    """Renders a regression-gated accept/reject verdict for a candidate harness.

    Args:
        regression_tolerance: Aggregate/dimension drop (baseline − candidate) tolerated before a
            regression is even considered — absorbs trivial numeric wobble. Default 0.02.
        min_samples: Minimum battery ``n`` on *both* arms to gate on; below it the verdict is
            ``INSUFFICIENT_DATA`` (you cannot separate signal from noise with too few tasks).
        require_significant: When True (default), a drop beyond tolerance is only a *regression* if
            it also clears the confidence-interval significance test — the candidate's optimistic
            bound stays below the baseline mean. When False, any drop beyond tolerance rejects
            (stricter; useful for a locked-down promotion gate).
    """

    def __init__(
        self,
        *,
        regression_tolerance: float = 0.02,
        min_samples: int = 5,
        require_significant: bool = True,
    ) -> None:
        self._tolerance = regression_tolerance
        self._min_samples = min_samples
        self._require_significant = require_significant

    def evaluate(
        self,
        *,
        config: HarnessConfig,
        baseline: BatteryResult,
        candidate: BatteryResult,
        characterization: Optional[CharacterizationDelta] = None,
    ) -> AcceptanceReport:
        """Return the :class:`AcceptanceReport` comparing ``candidate`` against ``baseline``."""
        reasons: list[str] = []
        dimension_deltas = self._dimension_deltas(baseline, candidate)

        # 1. Characterization: an unjustified byte-instability is a hard rejection (highest priority).
        if characterization is not None and not characterization.stable:
            if not characterization.justified:
                reasons.append(
                    "characterization battery changed without justification: "
                    + ", ".join(characterization.changed)
                )
                return self._report(
                    config,
                    AcceptanceVerdict.REJECT_CHARACTERIZATION,
                    baseline,
                    candidate,
                    dimension_deltas,
                    characterization,
                    reasons,
                )

        # 2. Trajectory regression on the overall confidence-weighted aggregate.
        insufficient = self._insufficient(baseline.overall, candidate.overall)
        overall_regressed = (not insufficient) and self._is_regression(
            baseline.overall, candidate.overall
        )
        dim_regressed = [d for d in dimension_deltas if d.regressed]

        if overall_regressed or dim_regressed:
            if overall_regressed and baseline.overall and candidate.overall:
                reasons.append(
                    f"overall aggregate regressed {baseline.overall.mean:.4f} → "
                    f"{candidate.overall.mean:.4f} (Δ {self._delta(baseline.overall, candidate.overall):+.4f})"
                )
            for d in dim_regressed:
                reasons.append(f"dimension {d.dimension.value} regressed (Δ {d.delta:+.4f})")
            return self._report(
                config,
                AcceptanceVerdict.REJECT_REGRESSION,
                baseline,
                candidate,
                dimension_deltas,
                characterization,
                reasons,
            )

        # 3. Not enough data to certify no-regression (and nothing rejected above).
        if insufficient:
            reasons.append(
                f"insufficient samples to gate (baseline n="
                f"{baseline.overall.n if baseline.overall else 0}, "
                f"candidate n={candidate.overall.n if candidate.overall else 0}, "
                f"min={self._min_samples})"
            )
            return self._report(
                config,
                AcceptanceVerdict.INSUFFICIENT_DATA,
                baseline,
                candidate,
                dimension_deltas,
                characterization,
                reasons,
            )

        # 4. Accepted — justified characterization delta, or fully stable.
        if characterization is not None and not characterization.stable:
            reasons.append(
                "characterization changed but justified: " + characterization.justification
            )
            verdict = AcceptanceVerdict.ACCEPT_JUSTIFIED
        else:
            reasons.append("no unacceptable regression; batteries within tolerance")
            verdict = AcceptanceVerdict.ACCEPT
        return self._report(
            config, verdict, baseline, candidate, dimension_deltas, characterization, reasons
        )

    # ── internals ─────────────────────────────────────────────────

    def _dimension_deltas(
        self, baseline: BatteryResult, candidate: BatteryResult
    ) -> tuple[DimensionDelta, ...]:
        base_by_dim = {d.dimension: d.stat for d in baseline.per_dimension}
        cand_by_dim = {d.dimension: d.stat for d in candidate.per_dimension}
        deltas: list[DimensionDelta] = []
        for dim in TrajectoryDimension:
            b = base_by_dim.get(dim)
            c = cand_by_dim.get(dim)
            if b is None or c is None:
                continue
            regressed = (not self._insufficient(b, c)) and self._is_regression(b, c)
            deltas.append(DimensionDelta(dim, b.mean, c.mean, regressed))
        return tuple(deltas)

    def _insufficient(self, base: Optional[IntervalStat], cand: Optional[IntervalStat]) -> bool:
        if base is None or cand is None:
            return True
        return base.n < self._min_samples or cand.n < self._min_samples

    def _is_regression(self, base: Optional[IntervalStat], cand: Optional[IntervalStat]) -> bool:
        """A regression is a drop beyond tolerance; optionally also CI-significant."""
        if base is None or cand is None:
            return False
        drop = base.mean - cand.mean
        if drop <= self._tolerance:
            return False
        if not self._require_significant:
            return True
        # Significant when the candidate's optimistic upper bound still sits below the baseline mean.
        return cand.upper < base.mean

    @staticmethod
    def _delta(base: IntervalStat, cand: IntervalStat) -> float:
        return cand.mean - base.mean

    @staticmethod
    def _report(
        config: HarnessConfig,
        verdict: AcceptanceVerdict,
        baseline: BatteryResult,
        candidate: BatteryResult,
        dimension_deltas: tuple[DimensionDelta, ...],
        characterization: Optional[CharacterizationDelta],
        reasons: list[str],
    ) -> AcceptanceReport:
        return AcceptanceReport(
            config=config,
            verdict=verdict,
            overall_baseline=baseline.overall,
            overall_candidate=candidate.overall,
            dimension_deltas=dimension_deltas,
            characterization=characterization,
            reasons=tuple(reasons),
        )
