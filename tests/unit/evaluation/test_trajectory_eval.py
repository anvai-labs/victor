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

"""Tests for the trajectory-level evaluation harness (EVR-1, FEP-0008 Phase E).

A duck-typed fake stands in for ``AgenticExecutionTrace`` (the scorers only read its attributes /
properties), keeping these fast and free of the heavy harness imports.
"""

from types import SimpleNamespace

from victor.evaluation.trajectory_eval import (
    DimensionScore,
    PlanningScorer,
    RecoveryScorer,
    RefusalScorer,
    ToolGroundingScorer,
    TrajectoryDimension,
    TrajectoryEvaluator,
    TrajectoryScore,
    mean_confidence_interval,
)


def _trace(*, tools=(), correction=None, topology=(), turns=1, messages=(), task_id="t"):
    calls = [SimpleNamespace(success=bool(s)) for s in tools]
    return SimpleNamespace(
        task_id=task_id,
        turns=turns,
        tool_calls=calls,
        total_tool_calls=len(calls),
        successful_tool_calls=sum(1 for c in calls if c.success),
        correction_metrics=correction or {},
        topology_events=list(topology),
        messages=list(messages),
        benchmark="unit",
    )


# --- ToolGrounding ---------------------------------------------------------------------------------


def test_tool_grounding_all_success():
    s = ToolGroundingScorer().score(_trace(tools=[True, True, True]))
    assert s.score == 1.0 and s.confidence == 1.0


def test_tool_grounding_partial():
    s = ToolGroundingScorer().score(_trace(tools=[True, False]))
    assert s.score == 0.5


def test_tool_grounding_no_tools_low_confidence():
    s = ToolGroundingScorer().score(_trace(tools=[]))
    assert s.score == 0.5 and s.confidence == 0.2  # dimension not engaged


# --- Recovery --------------------------------------------------------------------------------------


def test_recovery_failure_then_success_is_recovered():
    s = RecoveryScorer().score(_trace(tools=[False, True]))
    assert s.score == 1.0


def test_recovery_failure_never_recovers():
    s = RecoveryScorer().score(_trace(tools=[True, False]))
    assert s.score == 0.0


def test_recovery_no_failures_low_confidence():
    s = RecoveryScorer().score(_trace(tools=[True, True]))
    assert s.confidence == 0.2


def test_recovery_honors_correction_metrics():
    s = RecoveryScorer().score(_trace(tools=[False], correction={"recovered": True}))
    assert s.score == 1.0 and s.confidence == 0.9


# --- Planning --------------------------------------------------------------------------------------


def test_planning_topology_events():
    s = PlanningScorer().score(_trace(topology=[{"e": 1}], tools=[True]))
    assert s.score == 0.8


def test_planning_multistep():
    s = PlanningScorer().score(_trace(turns=3, tools=[True]))
    assert s.score == 0.6


def test_planning_none():
    s = PlanningScorer().score(_trace(turns=1, tools=[]))
    assert s.score == 0.4


# --- Refusal ---------------------------------------------------------------------------------------


def test_refusal_detected():
    msgs = [{"role": "assistant", "content": "I can't help with that request."}]
    s = RefusalScorer().score(_trace(messages=msgs))
    assert s.score == 0.0


def test_refusal_absent():
    msgs = [{"role": "assistant", "content": "The file is at app.py with 3 functions."}]
    s = RefusalScorer().score(_trace(messages=msgs))
    assert s.score == 1.0 and s.confidence == 0.2


# --- Aggregate / evaluator -------------------------------------------------------------------------


def test_aggregate_is_confidence_weighted():
    score = TrajectoryScore(
        task_id="t",
        dimensions=(
            DimensionScore(TrajectoryDimension.TOOL_GROUNDING, 1.0, 1.0, ""),
            DimensionScore(TrajectoryDimension.REFUSAL, 0.0, 0.0, ""),  # zero confidence -> ignored
        ),
    )
    assert (
        score.aggregate == 1.0
    )  # the un-engaged (0-confidence) refusal axis does not drag it down


def test_evaluator_scores_all_default_dimensions():
    ev = TrajectoryEvaluator()
    score = ev.score_trajectory(_trace(tools=[True, True]))
    dims = {d.dimension for d in score.dimensions}
    # EFFECT_GROUNDING (ADR-010 / EVR-4) is intentionally NOT in default_scorers() so existing
    # battery aggregates are unchanged; it is composed explicitly (see
    # test_trajectory_effect_grounding.py).
    assert dims == set(TrajectoryDimension) - {TrajectoryDimension.EFFECT_GROUNDING}


# --- Confidence intervals --------------------------------------------------------------------------


def test_ci_single_value_collapses():
    assert mean_confidence_interval([0.7]) == (0.7, 0.7, 0.7)


def test_ci_zero_variance_collapses():
    assert mean_confidence_interval([0.5, 0.5, 0.5]) == (0.5, 0.5, 0.5)


def test_ci_brackets_mean_and_clamps():
    mean, lo, hi = mean_confidence_interval([0.2, 0.4, 0.6, 0.8])
    assert lo <= mean <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_score_battery_reports_intervals():
    ev = TrajectoryEvaluator()
    traces = [
        _trace(tools=[True, True]),
        _trace(tools=[True, False]),
        _trace(tools=[False, True]),
    ]
    result = ev.score_battery(traces)
    assert result.overall is not None and result.overall.n == 3
    dims = {d.dimension for d in result.per_dimension}
    assert TrajectoryDimension.TOOL_GROUNDING in dims
    d = result.to_dict()
    assert (
        d["n"] == 3 and d["overall"]["ci_lower"] <= d["overall"]["mean"] <= d["overall"]["ci_upper"]
    )
