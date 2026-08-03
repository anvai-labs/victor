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

"""Unit tests for the flag-graduation harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from victor.evaluation.flag_graduation import (
    GraduationVerdict,
    assess_graduation,
    battery_from_dict,
    _main,
)
from victor.evaluation.trajectory_eval import BatteryResult, IntervalStat


def _battery(
    mean: float, n: int = 20, *, lower: Optional[float] = None, upper: Optional[float] = None
) -> BatteryResult:
    overall = IntervalStat(
        mean, mean if lower is None else lower, mean if upper is None else upper, n
    )
    return BatteryResult(scores=(), per_dimension=(), overall=overall)


FLAG = "effect_gated_completion"


def test_graduate_when_candidate_matches_baseline() -> None:
    report = assess_graduation(FLAG, _battery(0.80), _battery(0.80))
    assert report.verdict is GraduationVerdict.GRADUATE
    assert report.should_graduate
    assert FLAG in report.recommendation


def test_graduate_when_candidate_beats_baseline() -> None:
    report = assess_graduation(FLAG, _battery(0.70), _battery(0.85))
    assert report.verdict is GraduationVerdict.GRADUATE


def test_hold_on_significant_regression() -> None:
    report = assess_graduation(
        FLAG,
        _battery(0.80, lower=0.75, upper=0.85),
        _battery(0.60, lower=0.55, upper=0.65),
    )
    assert report.verdict is GraduationVerdict.HOLD
    assert not report.should_graduate
    assert "regressed" in report.recommendation


def test_hold_on_insufficient_data() -> None:
    report = assess_graduation(FLAG, _battery(0.80, n=2), _battery(0.50, n=2))
    assert report.verdict is GraduationVerdict.HOLD
    assert "insufficient" in report.recommendation


def test_report_serialization() -> None:
    payload = assess_graduation(FLAG, _battery(0.80), _battery(0.80)).to_dict()
    assert payload["flag"] == FLAG
    assert payload["verdict"] == "graduate"
    assert payload["should_graduate"] is True
    assert "acceptance" in payload


def test_battery_from_dict_roundtrip() -> None:
    original = _battery(0.77, n=13, lower=0.6, upper=0.9)
    restored = battery_from_dict(original.to_dict())
    assert restored.overall is not None
    assert restored.overall.mean == 0.77
    assert restored.overall.n == 13


def test_cli_graduate_exits_zero(tmp_path: Path) -> None:
    off = tmp_path / "off.json"
    on = tmp_path / "on.json"
    off.write_text(json.dumps(_battery(0.80).to_dict()), encoding="utf-8")
    on.write_text(json.dumps(_battery(0.82).to_dict()), encoding="utf-8")
    rc = _main(["--flag", FLAG, "--baseline", str(off), "--candidate", str(on)])
    assert rc == 0


def test_cli_hold_exits_one(tmp_path: Path) -> None:
    off = tmp_path / "off.json"
    on = tmp_path / "on.json"
    off.write_text(json.dumps(_battery(0.80, lower=0.75, upper=0.85).to_dict()), encoding="utf-8")
    on.write_text(json.dumps(_battery(0.60, lower=0.55, upper=0.65).to_dict()), encoding="utf-8")
    rc = _main(["--flag", FLAG, "--baseline", str(off), "--candidate", str(on)])
    assert rc == 1
