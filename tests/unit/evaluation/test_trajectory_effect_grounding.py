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

"""Tests for the EFFECT_GROUNDING trajectory dimension (EVR-4, ADR-010)."""

from victor.evaluation.agentic_harness import AgenticExecutionTrace, EvalToolCall, FileEdit
from victor.evaluation.trajectory_eval import (
    EffectGroundingScorer,
    TrajectoryDimension,
    TrajectoryEvaluator,
    default_scorers,
)


def _trace(*, tool_calls=(), file_edits=(), task_id="t1"):
    return AgenticExecutionTrace(
        task_id=task_id,
        start_time=0.0,
        end_time=1.0,
        tool_calls=list(tool_calls),
        file_edits=list(file_edits),
    )


def _call(name, arguments=None, success=True):
    return EvalToolCall(name=name, arguments=arguments or {}, success=success)


# --- scorer --------------------------------------------------------------------------------------


def test_workspace_delta_scores_full():
    score = EffectGroundingScorer().score(_trace(tool_calls=[_call("write", {"path": "a.py"})]))
    assert score.dimension is TrajectoryDimension.EFFECT_GROUNDING
    assert score.score == 1.0


def test_verified_check_scores_full():
    score = EffectGroundingScorer().score(_trace(tool_calls=[_call("shell", {"cmd": "pytest -q"})]))
    assert score.score == 1.0


def test_file_edits_count_as_effects():
    score = EffectGroundingScorer().score(
        _trace(file_edits=[FileEdit(path="a.py", action="modify")])
    )
    assert score.score == 1.0


def test_alias_tool_names_classify_identically():
    score = EffectGroundingScorer().score(
        _trace(tool_calls=[_call("write_file", {"file_path": "a.py"})])
    )
    assert score.score == 1.0


def test_read_only_trajectory_is_grounded_claim():
    score = EffectGroundingScorer().score(
        _trace(tool_calls=[_call("read", {"path": "a.py"}), _call("grep", {"pattern": "x"})])
    )
    assert score.score == 0.7
    assert "read-grounded" in score.reason


def test_completion_without_effect_scores_zero():
    # ADR-010 battery case: tools attempted, nothing succeeded — a completion here is ungrounded.
    score = EffectGroundingScorer().score(
        _trace(tool_calls=[_call("write", {"path": "a.py"}, success=False)])
    )
    assert score.score == 0.0
    assert "completion-without-effect" in score.reason


def test_no_tool_trajectory_not_engaged():
    score = EffectGroundingScorer().score(_trace())
    assert score.score == 0.5
    assert score.confidence <= 0.2


# --- no-completion-without-effect battery cases (ADR-010) ----------------------------------------


def test_battery_separates_effectful_from_empty_completions():
    scorer = EffectGroundingScorer()
    effectful = scorer.score(_trace(tool_calls=[_call("edit", {"path": "b.py"})]))
    empty = scorer.score(
        _trace(tool_calls=[_call("edit", {"path": "b.py"}, success=False)], task_id="t2")
    )
    assert effectful.score == 1.0
    assert empty.score == 0.0


def test_evaluator_composes_effect_grounding_dimension():
    evaluator = TrajectoryEvaluator(default_scorers() + (EffectGroundingScorer(),))
    trajectory = evaluator.score_trajectory(_trace(tool_calls=[_call("write", {"path": "a.py"})]))
    dim = trajectory.get(TrajectoryDimension.EFFECT_GROUNDING)
    assert dim is not None and dim.score == 1.0


# --- default_scorers exclusion (aggregate stability) ---------------------------------------------


def test_not_in_default_scorers():
    dims = {s.dimension for s in default_scorers()}
    assert TrajectoryDimension.EFFECT_GROUNDING not in dims


def test_default_battery_aggregates_unshifted():
    # Default evaluator must produce no EFFECT_GROUNDING dimension at all.
    trajectory = TrajectoryEvaluator().score_trajectory(
        _trace(tool_calls=[_call("write", {"path": "a.py"})])
    )
    assert trajectory.get(TrajectoryDimension.EFFECT_GROUNDING) is None
