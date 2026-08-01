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

"""EVR-5 / ADR-012: the harness-acceptance oracle as the promotion merge gate.

Drives the FEP-0007 QA battery through the *real* scripted streaming loop (no LLM, no network),
maps each run to an :class:`~victor.evaluation.agentic_harness.AgenticExecutionTrace`, scores the
battery with the trajectory evaluator, and asks the
:class:`~victor.evaluation.acceptance_oracle.HarnessAcceptanceOracle` whether the current harness
regresses against a checked-in baseline — the ADR-012 gate ("no harness/prompt edit ships without
passing it"), reported at *(model, harness-config)* granularity.

This lives under ``tests/integration/`` so it runs in the develop→main promotion battery
(``ci-integration.yml``), not the fast develop gate — the heavy oracle belongs on promotion.

Regenerate the baseline after an *intended* behavior change::

    EVR5_RECORD_BASELINE=1 pytest tests/integration/streaming/test_acceptance_oracle_gate.py

then review + commit ``acceptance_baseline.json``. A change that moves the battery without a baseline
update surfaces as a REJECT (regression) or an unjustified characterization delta — by design.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import pytest

from victor.evaluation.acceptance_oracle import (
    HarnessAcceptanceOracle,
    HarnessConfig,
    characterization_from_signatures,
)
from victor.evaluation.agentic_harness import AgenticExecutionTrace, EvalToolCall
from victor.evaluation.trajectory_eval import (
    BatteryResult,
    DimensionInterval,
    EffectGroundingScorer,
    IntervalStat,
    TrajectoryDimension,
    TrajectoryEvaluator,
    default_scorers,
)

from .parity_harness import (
    SCENARIOS,
    Scenario,
    StreamTranscript,
    build_for_scenario,
    capture_transcript,
)

pytestmark = [pytest.mark.integration, pytest.mark.agents]

_BASELINE_PATH = Path(__file__).parent / "acceptance_baseline.json"
# The (model, harness-config) the baseline was recorded under — the scripted, deterministic harness.
_CONFIG = HarnessConfig(model="scripted", completion_strategy="enhanced", effect_gate=False)


def _trace_from_scenario(scenario: Scenario, transcript: StreamTranscript) -> AgenticExecutionTrace:
    """Map a scripted run's :class:`StreamTranscript` into a scorable trajectory trace."""
    failing = set(scenario.failing_tools)
    calls = [
        EvalToolCall(name=name, arguments={}, success=name not in failing)
        for name in transcript.tool_calls
    ]
    messages: List[dict] = [{"role": "user", "content": scenario.message}]
    if transcript.content:
        messages.append({"role": "assistant", "content": transcript.content})
    return AgenticExecutionTrace(
        task_id=scenario.id,
        start_time=0.0,
        end_time=0.0,
        benchmark="fep0007-parity-battery",
        turns=len(scenario.turns),
        messages=messages,
        tool_calls=calls,
    )


def _signature(transcript: StreamTranscript) -> str:
    """A stable digest of one scenario's observable behavior (final content + tool sequence)."""
    payload = transcript.content + "␟" + "|".join(transcript.tool_calls)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _evaluator() -> TrajectoryEvaluator:
    # Adopt EffectGroundingScorer (ADR-010) here — the acceptance oracle measures effect grounding
    # even though it is out of the default EVR-1 battery to keep those aggregates stable.
    return TrajectoryEvaluator(tuple(default_scorers()) + (EffectGroundingScorer(),))


async def _run_battery() -> tuple[BatteryResult, Dict[str, str]]:
    traces: List[AgenticExecutionTrace] = []
    signatures: Dict[str, str] = {}
    for scenario in SCENARIOS:
        orch = build_for_scenario(scenario)
        transcript = await capture_transcript(orch, scenario.message)
        traces.append(_trace_from_scenario(scenario, transcript))
        signatures[scenario.id] = _signature(transcript)
    return _evaluator().score_battery(traces), signatures


def _battery_from_snapshot(data: dict) -> BatteryResult:
    ov = data["overall"]
    overall = IntervalStat(ov["mean"], ov["ci_lower"], ov["ci_upper"], ov["n"])
    per_dim = tuple(
        DimensionInterval(
            TrajectoryDimension(d["dimension"]),
            IntervalStat(d["mean"], d["ci_lower"], d["ci_upper"], d["n"]),
        )
        for d in data["per_dimension"]
    )
    return BatteryResult(scores=(), per_dimension=per_dim, overall=overall)


def _write_baseline(candidate: BatteryResult, signatures: Dict[str, str]) -> None:
    _BASELINE_PATH.write_text(
        json.dumps({"battery": candidate.to_dict(), "signatures": signatures}, indent=2) + "\n",
        encoding="utf-8",
    )


async def test_promotion_acceptance_gate() -> None:
    """The current harness must clear the acceptance oracle against the recorded baseline."""
    candidate, signatures = await _run_battery()

    if os.getenv("EVR5_RECORD_BASELINE") == "1" or not _BASELINE_PATH.exists():
        _write_baseline(candidate, signatures)
        pytest.skip(f"recorded acceptance baseline → {_BASELINE_PATH.name}")

    data = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    baseline = _battery_from_snapshot(data["battery"])
    characterization = characterization_from_signatures(data["signatures"], signatures)

    report = HarnessAcceptanceOracle().evaluate(
        config=_CONFIG,
        baseline=baseline,
        candidate=candidate,
        characterization=characterization,
    )

    detail = (
        f"acceptance oracle REJECTED promotion for {_CONFIG.resolved_label()}:\n  - "
        + "\n  - ".join(report.reasons)
        + "\n"
        + json.dumps(report.to_dict(), indent=2)
    )
    assert report.accepted, detail


def test_baseline_snapshot_is_committed() -> None:
    """Guard: the recorded baseline must be present in the tree (not regenerated silently in CI)."""
    assert _BASELINE_PATH.exists(), (
        f"{_BASELINE_PATH.name} is missing — regenerate with "
        "EVR5_RECORD_BASELINE=1 pytest tests/integration/streaming/test_acceptance_oracle_gate.py "
        "and commit it."
    )
