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

"""Tests for rubric-based completion evaluation (EVR-3, ADR-009)."""

from dataclasses import dataclass

import pytest

from victor.framework.rubric_completion import (
    DefaultRubricGenerator,
    DimensionAwareFilter,
    HeuristicRubricJudge,
    Rubric,
    RubricCache,
    RubricCompletionEvaluator,
    RubricDimension,
    RubricDimensionScore,
    confidence_weighted_mean,
)


@dataclass
class _ScriptedJudge:
    """Returns scripted (score, confidence) per dimension name; counts calls."""

    scores: dict
    default: tuple = (0.9, 0.9)
    calls: int = 0

    def score(self, dimension, content, context):
        self.calls += 1
        s, c = self.scores.get(dimension.name, self.default)
        return RubricDimensionScore(dimension.name, s, c, "scripted")


# --- Rubric model ----------------------------------------------------------------------------------


def test_rubric_validate_rejects_duplicates():
    with pytest.raises(ValueError):
        Rubric("t", (RubricDimension("a"), RubricDimension("a"))).validate()


def test_rubric_normalized_sums_to_one():
    r = Rubric("t", (RubricDimension("a", 2.0), RubricDimension("b", 2.0))).normalized()
    assert pytest.approx(sum(d.weight for d in r.dimensions)) == 1.0


def test_default_generator_produces_valid_rubric():
    r = DefaultRubricGenerator().generate("coding", {})
    assert len(r.dimensions) == 4
    assert pytest.approx(sum(d.weight for d in r.dimensions)) == 1.0


# --- Heuristic judge / aggregate -------------------------------------------------------------------


def test_heuristic_judge_rewards_substantial_structured_content():
    judge = HeuristicRubricJudge()
    dim = RubricDimension("correctness")
    short = judge.score(dim, "ok", {})
    long_structured = judge.score(dim, "x" * 400 + "\n- a\n- b", {})
    assert long_structured.score > short.score
    assert short.confidence == 0.3  # fallback stays low-confidence


def test_confidence_weighted_mean_ignores_zero_confidence():
    scores = (
        RubricDimensionScore("a", 1.0, 1.0),
        RubricDimensionScore("b", 0.0, 0.0),  # un-engaged
    )
    assert confidence_weighted_mean(scores) == 1.0


# --- DimensionAwareFilter --------------------------------------------------------------------------


def test_filter_flags_engaged_below_threshold():
    rubric = DefaultRubricGenerator().generate("t", {})
    scores = [
        RubricDimensionScore("correctness", 0.5, 0.9),  # engaged, below 0.6 -> fail
        RubricDimensionScore("tool_grounding", 0.9, 0.9),
        RubricDimensionScore("completeness", 0.9, 0.9),
        RubricDimensionScore("recovery", 0.9, 0.9),
    ]
    failed = DimensionAwareFilter().failed(rubric, scores)
    assert {f.name for f in failed} == {"correctness"}


def test_filter_ignores_unengaged_low_confidence_dimension():
    rubric = DefaultRubricGenerator().generate("t", {})
    scores = [
        RubricDimensionScore("correctness", 0.9, 0.9),
        RubricDimensionScore("tool_grounding", 0.9, 0.9),
        RubricDimensionScore("completeness", 0.9, 0.9),
        RubricDimensionScore("recovery", 0.1, 0.1),  # below threshold but un-engaged -> not gated
    ]
    assert DimensionAwareFilter(confidence_floor=0.25).failed(rubric, scores) == ()


# --- Cache -----------------------------------------------------------------------------------------


def test_cache_generates_once_per_family():
    @dataclass
    class _CountingGen:
        calls: int = 0

        def generate(self, task_family, context):
            self.calls += 1
            return Rubric(task_family, (RubricDimension("a", 1.0),))

    gen = _CountingGen()
    cache = RubricCache()
    cache.get_or_generate("fam", gen, {})
    cache.get_or_generate("fam", gen, {})
    assert gen.calls == 1  # cached on the second call


# --- Evaluator end-to-end --------------------------------------------------------------------------


def test_evaluator_complete_when_all_dimensions_clear():
    ev = RubricCompletionEvaluator(judge=_ScriptedJudge(scores={}))  # all default (0.9, 0.9)
    result = ev.evaluate(task_family="coding", content="done", context={})
    assert result.complete is True and result.failed_dimensions == ()
    assert result.aggregate == pytest.approx(0.9)


def test_evaluator_incomplete_when_engaged_dimension_fails():
    judge = _ScriptedJudge(scores={"completeness": (0.4, 0.9)})  # engaged, below 0.6
    ev = RubricCompletionEvaluator(judge=judge)
    result = ev.evaluate(task_family="coding", content="partial", context={})
    assert result.complete is False
    assert "completeness" in result.failed_dimensions


def test_evaluator_completes_despite_unengaged_low_dimension():
    # recovery scored low but un-engaged (low confidence) -> does not block completion.
    judge = _ScriptedJudge(scores={"recovery": (0.0, 0.1)})
    ev = RubricCompletionEvaluator(judge=judge)
    result = ev.evaluate(task_family="coding", content="done", context={})
    assert result.complete is True


def test_evaluator_default_construction_runs_without_llm():
    ev = RubricCompletionEvaluator()  # default generator + heuristic judge
    result = ev.evaluate(task_family="qa", content="x" * 500 + "\n- point", context={})
    assert isinstance(result.complete, bool) and 0.0 <= result.aggregate <= 1.0
