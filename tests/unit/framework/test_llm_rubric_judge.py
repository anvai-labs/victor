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

"""Tests for the LLM-backed rubric judge (EVR-3c, FEP-0008 Phase A / ADR-009)."""

from victor.framework.rubric_completion import (
    AsyncRubricCompletionEvaluator,
    DefaultRubricGenerator,
    LLMRubricJudge,
    Rubric,
    RubricDimension,
    _build_rubric_prompt,
    _parse_rubric_scores,
)


def _rubric():
    return DefaultRubricGenerator().generate("coding", {})


# --- parsing ---------------------------------------------------------------------------------------


def test_parse_extracts_score_and_confidence_per_dimension():
    rubric = _rubric()
    text = (
        "correctness: score=0.9 confidence=0.8\n"
        "tool_grounding: score=0.7 confidence=0.6\n"
        "completeness: score=0.85 confidence=0.7\n"
        "recovery: score=0.5 confidence=0.3\n"
    )
    scores = _parse_rubric_scores(rubric, text)
    by_name = {s.name: s for s in scores}
    assert by_name["correctness"].score == 0.9 and by_name["correctness"].confidence == 0.8
    assert by_name["recovery"].score == 0.5


def test_parse_missing_dimension_falls_back_low_confidence():
    rubric = Rubric("t", (RubricDimension("alpha"), RubricDimension("beta")))
    scores = _parse_rubric_scores(rubric, "alpha: score=0.9 confidence=0.9")
    by_name = {s.name: s for s in scores}
    assert by_name["beta"].score == 0.5 and by_name["beta"].confidence == 0.2


def test_parse_missing_numbers_use_defaults_and_clamp():
    rubric = Rubric("t", (RubricDimension("alpha"),))
    # Line present but no parseable score/confidence -> neutral score, confidence BELOW
    # the DimensionAwareFilter engagement floor (0.25) so it cannot gate completion.
    s = _parse_rubric_scores(rubric, "alpha: looks good")[0]
    assert s.score == 0.5 and s.confidence == 0.2


def test_build_prompt_includes_dimensions_and_content():
    rubric = _rubric()
    prompt = _build_rubric_prompt(rubric, "the response text")
    assert "correctness" in prompt and "the response text" in prompt


# --- LLM judge -------------------------------------------------------------------------------------


async def test_llm_judge_scores_via_complete_fn():
    rubric = _rubric()

    async def fake_complete(prompt):
        return "\n".join(f"{d.name}: score=0.9 confidence=0.9" for d in rubric.dimensions)

    scores = await LLMRubricJudge(fake_complete).score_rubric(rubric, "answer", {})
    assert len(scores) == len(rubric.dimensions)
    assert all(s.score == 0.9 for s in scores)


async def test_llm_judge_degrades_on_error():
    async def boom(prompt):
        raise RuntimeError("provider down")

    rubric = _rubric()
    scores = await LLMRubricJudge(boom).score_rubric(rubric, "answer", {})
    assert all(s.score == 0.5 and s.confidence == 0.2 for s in scores)  # neutral fallback


# --- async evaluator end-to-end --------------------------------------------------------------------


async def test_async_evaluator_complete_when_all_clear():
    async def fake_complete(prompt):
        return (
            "correctness: score=0.9 confidence=0.9\n"
            "tool_grounding: score=0.9 confidence=0.9\n"
            "completeness: score=0.9 confidence=0.9\n"
            "recovery: score=0.9 confidence=0.9\n"
        )

    ev = AsyncRubricCompletionEvaluator(LLMRubricJudge(fake_complete))
    result = await ev.aevaluate(task_family="coding", content="done", context={})
    assert result.complete is True and result.failed_dimensions == ()


async def test_async_evaluator_incomplete_when_dimension_fails():
    async def fake_complete(prompt):
        return (
            "correctness: score=0.3 confidence=0.9\n"  # engaged, below 0.6 -> fail
            "tool_grounding: score=0.9 confidence=0.9\n"
            "completeness: score=0.9 confidence=0.9\n"
            "recovery: score=0.9 confidence=0.9\n"
        )

    ev = AsyncRubricCompletionEvaluator(LLMRubricJudge(fake_complete))
    result = await ev.aevaluate(task_family="coding", content="partial", context={})
    assert result.complete is False and "correctness" in result.failed_dimensions
