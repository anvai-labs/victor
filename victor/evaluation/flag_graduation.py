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

"""Flag-graduation harness (flag-graduation policy × EVR-5 acceptance oracle).

Turns the regression-gated acceptance oracle into the concrete *decision* the flag-graduation
policy needs: **is opt-in flag X safe to flip default-on?** It compares a **candidate** battery
(flag ON) against a **baseline** battery (flag OFF) via
:class:`~victor.evaluation.acceptance_oracle.HarnessAcceptanceOracle` — GRADUATE when enabling the
flag *matches-or-beats* the baseline (the oracle accepts: no unacceptable regression), otherwise
HOLD (regression, or insufficient battery evidence to decide).

This is the pure decision layer. Producing the two :class:`BatteryResult` inputs from real A/B runs
(the flag toggled on/off over a task set) is the measurement step the policy governs; a
``batteries → BatteryResult.to_dict()`` snapshot for each arm feeds the CLI
(``python -m victor.evaluation.flag_graduation --flag X --baseline off.json --candidate on.json``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from victor.evaluation.acceptance_oracle import (
    AcceptanceReport,
    AcceptanceVerdict,
    HarnessAcceptanceOracle,
    HarnessConfig,
)
from victor.evaluation.trajectory_eval import (
    BatteryResult,
    DimensionInterval,
    IntervalStat,
    TrajectoryDimension,
)


class GraduationVerdict(str, Enum):
    """Should an opt-in flag be graduated to default-on?"""

    GRADUATE = "graduate"  # enabling the flag matches-or-beats baseline — safe to flip default-on
    HOLD = "hold"  # regression, or not enough battery evidence to decide


@dataclass(frozen=True)
class GraduationReport:
    """The graduation decision for one flag, wrapping the oracle's acceptance report."""

    flag: str
    verdict: GraduationVerdict
    acceptance: AcceptanceReport
    recommendation: str

    @property
    def should_graduate(self) -> bool:
        return self.verdict is GraduationVerdict.GRADUATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "flag": self.flag,
            "verdict": self.verdict.value,
            "should_graduate": self.should_graduate,
            "recommendation": self.recommendation,
            "acceptance": self.acceptance.to_dict(),
        }


def assess_graduation(
    flag_name: str,
    baseline: BatteryResult,
    candidate: BatteryResult,
    *,
    oracle: Optional[HarnessAcceptanceOracle] = None,
    neutral_band: float = 0.005,
) -> GraduationReport:
    """Assess whether ``flag_name`` is safe *and worth it* to graduate default-on.

    The oracle's ACCEPT is only the **safety** bar (no unacceptable regression). Flipping a
    default-on additionally needs **benefit or genuine neutrality** — otherwise you take on cost
    for nothing. So GRADUATE requires ACCEPT *and* an overall delta ≥ ``-neutral_band``; an accepted
    candidate that is measurably worse (delta below the band) is safe but HOLDs ("no benefit").

    Args:
        flag_name: The opt-in flag under assessment (e.g. ``"effect_gated_completion"``).
        baseline: Battery with the flag **off**.
        candidate: Battery with the flag **on**.
        oracle: Acceptance oracle to gate with; a default is used when omitted.
        neutral_band: How far below baseline still counts as "neutral" (default 0.005). A candidate
            worse than this — even if not *significantly* worse — earns no default flip.

    Returns:
        A :class:`GraduationReport`: GRADUATE when the flag beats-or-matches baseline (within the
        neutral band) with no unacceptable regression; HOLD on regression, no benefit, or too little
        data.
    """
    oracle = oracle or HarnessAcceptanceOracle()
    config = HarnessConfig(model="battery", label=f"graduate:{flag_name}")
    report = oracle.evaluate(config=config, baseline=baseline, candidate=candidate)

    if report.verdict is AcceptanceVerdict.INSUFFICIENT_DATA:
        return GraduationReport(
            flag=flag_name,
            verdict=GraduationVerdict.HOLD,
            acceptance=report,
            recommendation=(
                f"HOLD: insufficient battery evidence to graduate '{flag_name}' "
                "— run a larger A/B."
            ),
        )
    if report.accepted:
        delta = report.overall_delta
        if delta >= -neutral_band:
            stance = (
                "beats-or-matches baseline"
                if delta >= 0
                else f"is statistically neutral (within ±{neutral_band:.3f})"
            )
            return GraduationReport(
                flag=flag_name,
                verdict=GraduationVerdict.GRADUATE,
                acceptance=report,
                recommendation=(
                    f"GRADUATE: enabling '{flag_name}' {stance} "
                    f"(Δ {delta:+.4f}, no unacceptable regression)."
                ),
            )
        # Accepted (no significant regression) but measurably worse → safe, but not worth flipping.
        return GraduationReport(
            flag=flag_name,
            verdict=GraduationVerdict.HOLD,
            acceptance=report,
            recommendation=(
                f"HOLD: enabling '{flag_name}' is safe (no unacceptable regression) but shows no "
                f"benefit — measurably worse (Δ {delta:+.4f}, beyond the ±{neutral_band:.3f} "
                "neutral band). Keep it opt-in."
            ),
        )
    return GraduationReport(
        flag=flag_name,
        verdict=GraduationVerdict.HOLD,
        acceptance=report,
        recommendation=(
            f"HOLD: enabling '{flag_name}' regressed the battery "
            f"({report.verdict.value}); do not flip default-on."
        ),
    )


def battery_from_dict(data: dict[str, Any]) -> BatteryResult:
    """Reconstruct a :class:`BatteryResult` from a ``BatteryResult.to_dict()`` snapshot.

    Only the fields the oracle reads (``overall`` + ``per_dimension``) are restored; ``scores`` is
    left empty (the oracle does not consult it).
    """
    overall = None
    ov = data.get("overall")
    if ov:
        overall = IntervalStat(ov["mean"], ov["ci_lower"], ov["ci_upper"], ov["n"])
    per_dim = tuple(
        DimensionInterval(
            TrajectoryDimension(d["dimension"]),
            IntervalStat(d["mean"], d["ci_lower"], d["ci_upper"], d["n"]),
        )
        for d in data.get("per_dimension", [])
    )
    return BatteryResult(scores=(), per_dimension=per_dim, overall=overall)


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m victor.evaluation.flag_graduation",
        description="Decide whether an opt-in flag is safe to graduate default-on (A/B via the "
        "acceptance oracle).",
    )
    parser.add_argument("--flag", required=True, help="Flag name under assessment")
    parser.add_argument(
        "--baseline", required=True, help="Path to the flag-OFF BatteryResult.to_dict() JSON"
    )
    parser.add_argument(
        "--candidate", required=True, help="Path to the flag-ON BatteryResult.to_dict() JSON"
    )
    args = parser.parse_args(argv)

    with open(args.baseline, encoding="utf-8") as fh:
        baseline = battery_from_dict(json.load(fh))
    with open(args.candidate, encoding="utf-8") as fh:
        candidate = battery_from_dict(json.load(fh))

    report = assess_graduation(args.flag, baseline, candidate)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.should_graduate else 1


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_main())
