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

"""Unit tests for the regression-gated acceptance oracle (EVR-5, ADR-012)."""

from __future__ import annotations

from typing import Optional

from victor.evaluation.acceptance_oracle import (
    AcceptanceVerdict,
    HarnessAcceptanceOracle,
    HarnessConfig,
    characterization_from_signatures,
)
from victor.evaluation.trajectory_eval import (
    BatteryResult,
    DimensionInterval,
    IntervalStat,
    TrajectoryDimension,
)


def _battery(
    mean: float,
    n: int = 20,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    dims: Optional[dict[TrajectoryDimension, float]] = None,
) -> BatteryResult:
    overall = IntervalStat(
        mean, mean if lower is None else lower, mean if upper is None else upper, n
    )
    per_dim = tuple(
        DimensionInterval(dim, IntervalStat(m, m, m, n)) for dim, m in (dims or {}).items()
    )
    return BatteryResult(scores=(), per_dimension=per_dim, overall=overall)


CONFIG = HarnessConfig(model="qwen3:8b", completion_strategy="rubric", effect_gate=True)


def test_accept_when_candidate_matches_baseline() -> None:
    oracle = HarnessAcceptanceOracle()
    report = oracle.evaluate(config=CONFIG, baseline=_battery(0.80), candidate=_battery(0.80))
    assert report.verdict is AcceptanceVerdict.ACCEPT
    assert report.accepted


def test_reject_on_significant_overall_regression() -> None:
    oracle = HarnessAcceptanceOracle()
    report = oracle.evaluate(
        config=CONFIG,
        baseline=_battery(0.80, lower=0.75, upper=0.85),
        candidate=_battery(0.60, lower=0.55, upper=0.65),
    )
    assert report.verdict is AcceptanceVerdict.REJECT_REGRESSION
    assert not report.accepted
    assert report.overall_delta < 0


def test_noise_within_ci_is_not_a_regression() -> None:
    oracle = HarnessAcceptanceOracle()
    # Drop of 0.05 exceeds tolerance, but candidate's upper bound (0.95) sits above baseline mean
    # (0.80) — not statistically separable, so it is not a regression.
    report = oracle.evaluate(
        config=CONFIG,
        baseline=_battery(0.80, lower=0.60, upper=1.0),
        candidate=_battery(0.75, lower=0.55, upper=0.95),
    )
    assert report.verdict is AcceptanceVerdict.ACCEPT


def test_require_significant_false_rejects_any_drop_beyond_tolerance() -> None:
    oracle = HarnessAcceptanceOracle(require_significant=False)
    report = oracle.evaluate(
        config=CONFIG,
        baseline=_battery(0.80, lower=0.60, upper=1.0),
        candidate=_battery(0.75, lower=0.55, upper=0.95),
    )
    assert report.verdict is AcceptanceVerdict.REJECT_REGRESSION


def test_insufficient_data_when_samples_below_minimum() -> None:
    oracle = HarnessAcceptanceOracle(min_samples=5)
    report = oracle.evaluate(
        config=CONFIG, baseline=_battery(0.80, n=2), candidate=_battery(0.50, n=2)
    )
    assert report.verdict is AcceptanceVerdict.INSUFFICIENT_DATA
    assert not report.accepted


def test_reject_on_unjustified_characterization_delta() -> None:
    oracle = HarnessAcceptanceOracle()
    char = characterization_from_signatures({"s1": "a"}, {"s1": "b"})
    report = oracle.evaluate(
        config=CONFIG, baseline=_battery(0.80), candidate=_battery(0.80), characterization=char
    )
    assert report.verdict is AcceptanceVerdict.REJECT_CHARACTERIZATION
    assert "s1" in report.characterization.changed


def test_accept_justified_characterization_delta() -> None:
    oracle = HarnessAcceptanceOracle()
    char = characterization_from_signatures(
        {"s1": "a"}, {"s1": "b"}, justification="intentional prompt reflow, output equivalent"
    )
    report = oracle.evaluate(
        config=CONFIG, baseline=_battery(0.80), candidate=_battery(0.80), characterization=char
    )
    assert report.verdict is AcceptanceVerdict.ACCEPT_JUSTIFIED
    assert report.accepted


def test_regression_dominates_a_justified_characterization() -> None:
    oracle = HarnessAcceptanceOracle()
    char = characterization_from_signatures({"s1": "a"}, {"s1": "b"}, justification="reflow")
    report = oracle.evaluate(
        config=CONFIG,
        baseline=_battery(0.80, lower=0.75, upper=0.85),
        candidate=_battery(0.60, lower=0.55, upper=0.65),
        characterization=char,
    )
    assert report.verdict is AcceptanceVerdict.REJECT_REGRESSION


def test_dimension_regression_rejects_even_when_overall_stable() -> None:
    oracle = HarnessAcceptanceOracle()
    base = _battery(0.80, dims={TrajectoryDimension.TOOL_GROUNDING: 0.9})
    cand = _battery(0.80, dims={TrajectoryDimension.TOOL_GROUNDING: 0.5})
    report = oracle.evaluate(config=CONFIG, baseline=base, candidate=cand)
    assert report.verdict is AcceptanceVerdict.REJECT_REGRESSION
    regressed = [d for d in report.dimension_deltas if d.regressed]
    assert regressed and regressed[0].dimension is TrajectoryDimension.TOOL_GROUNDING


def test_characterization_from_signatures_detects_added_key() -> None:
    delta = characterization_from_signatures({"s1": "a"}, {"s1": "a", "s2": "new"})
    assert not delta.stable
    assert delta.changed == ("s2",)


def test_config_label_and_report_serialization() -> None:
    oracle = HarnessAcceptanceOracle()
    report = oracle.evaluate(config=CONFIG, baseline=_battery(0.80), candidate=_battery(0.80))
    assert "qwen3:8b" in CONFIG.resolved_label()
    assert "rubric" in CONFIG.resolved_label()
    payload = report.to_dict()
    assert payload["verdict"] == "accept"
    assert payload["accepted"] is True
    assert payload["config"] == CONFIG.resolved_label()
