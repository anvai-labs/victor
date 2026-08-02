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

"""Tests for the rubric judge adapter (EVR-2 × EVR-3, ADR-009 × ADR-011)."""

from pathlib import Path
from typing import Any, Mapping

import pytest

from victor.evaluation.calibration_corpus import default_corpus
from victor.evaluation.calibration_rubric_judge import (
    make_llm_rubric_judge,
    make_rubric_judge,
    render_judged_content,
)
from victor.evaluation.judge_calibration_harness import (
    JudgeCalibrationHarness,
    Transcript,
    TranscriptStep,
    make_scripted_executor,
)
from victor.framework.rubric_completion import (
    RubricCompletionEvaluator,
    RubricDimension,
    RubricDimensionScore,
)


def _transcript(*, tools: int = 1, final: str = "Done — created the file.") -> Transcript:
    steps = tuple(
        TranscriptStep(kind="tool", content=f"write_file(path='f{i}.txt')") for i in range(tools)
    )
    return Transcript(steps=steps, final_message=final)


# --- Content rendering -----------------------------------------------------------------------------


def test_render_includes_all_blinded_sections(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "settings.toml").write_text("port = 8000\n")
    (workspace / "docs").mkdir()
    (workspace / "docs" / "guide.md").write_text("# Guide\n")
    content = render_judged_content("Create settings.toml", _transcript(), workspace)
    assert "TASK:\nCreate settings.toml" in content
    assert "write_file(path='f0.txt')" in content
    # Workspace file contents are visible (a judge cannot assess correctness from names).
    assert "--- settings.toml ---" in content and "port = 8000" in content
    assert "--- docs/guide.md ---" in content
    assert "FINAL RESPONSE:\nDone — created the file." in content


def test_render_omits_oversized_file_contents(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "big.txt").write_text("x" * 5_000)
    content = render_judged_content("Task", _transcript(), workspace)
    assert "content omitted" in content
    assert "x" * 100 not in content


def test_render_handles_empty_transcript_and_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "empty_ws"  # fresh dir: conftest materializes .victor/ in tmp_path
    workspace.mkdir()
    content = render_judged_content("Task", Transcript(), workspace)
    assert "(none)" in content
    assert "(empty)" in content


def test_render_caps_tool_step_listing(tmp_path: Path) -> None:
    content = render_judged_content("Task", _transcript(tools=40), tmp_path)
    assert "... 10 more" in content


# --- Sync rubric judge -----------------------------------------------------------------------------


class _RecordingJudge:
    """Rubric-dimension judge stub that records the content it was asked to grade."""

    def __init__(self, score: float, confidence: float = 0.9) -> None:
        self.score_value = score
        self.confidence = confidence
        self.contents: list[str] = []

    def score(
        self, dimension: RubricDimension, content: str, context: Mapping[str, Any]
    ) -> RubricDimensionScore:
        self.contents.append(content)
        return RubricDimensionScore(dimension.name, self.score_value, self.confidence, "stub")


def test_rubric_judge_projects_complete_verdict_to_binary(tmp_path: Path) -> None:
    passing = make_rubric_judge(RubricCompletionEvaluator(judge=_RecordingJudge(0.9)))
    failing = make_rubric_judge(RubricCompletionEvaluator(judge=_RecordingJudge(0.1)))
    assert passing("Task", _transcript(), tmp_path) == 1.0
    assert failing("Task", _transcript(), tmp_path) == 0.0


def test_rubric_judge_aggregate_mode_returns_weighted_mean(tmp_path: Path) -> None:
    judge = make_rubric_judge(
        RubricCompletionEvaluator(judge=_RecordingJudge(0.8)), score_mode="aggregate"
    )
    assert judge("Task", _transcript(), tmp_path) == pytest.approx(0.8)


def test_rubric_judge_grades_the_rendered_blinded_view(tmp_path: Path) -> None:
    recording = _RecordingJudge(0.9)
    judge = make_rubric_judge(RubricCompletionEvaluator(judge=recording))
    judge("The task prompt", _transcript(final="All done."), tmp_path)
    assert recording.contents, "judge was never invoked"
    graded = recording.contents[0]
    assert "The task prompt" in graded
    assert "All done." in graded
    assert "gold" not in graded.lower()


def test_default_rubric_judge_is_deterministic_and_offline(tmp_path: Path) -> None:
    judge = make_rubric_judge()
    first = judge("Task", _transcript(), tmp_path)
    second = judge("Task", _transcript(), tmp_path)
    assert first == second
    assert first in (0.0, 1.0)


def test_default_rubric_judge_composes_with_harness(tmp_path: Path) -> None:
    harness = JudgeCalibrationHarness(default_corpus(variants=1))
    report = harness.run(
        make_scripted_executor(lambda _t: True),
        make_rubric_judge(),
        workspace_root=tmp_path,
    )
    assert len(report.samples) == 6
    assert all(sample.judged in (0.0, 1.0) for sample in report.samples)


# --- LLM rubric judge ------------------------------------------------------------------------------


def _grade_lines(score: float, confidence: float = 0.9) -> str:
    names = ("correctness", "tool_grounding", "completeness", "recovery")
    return "\n".join(f"{name}: score={score} confidence={confidence}" for name in names)


def test_llm_rubric_judge_scores_from_completion_text(tmp_path: Path) -> None:
    async def good_fn(_prompt: str) -> str:
        return _grade_lines(0.9)

    async def bad_fn(_prompt: str) -> str:
        return _grade_lines(0.1)

    assert make_llm_rubric_judge(good_fn)("Task", _transcript(), tmp_path) == 1.0
    assert make_llm_rubric_judge(bad_fn)("Task", _transcript(), tmp_path) == 0.0


def test_llm_rubric_judge_receives_rendered_view(tmp_path: Path) -> None:
    prompts: list[str] = []

    async def spy_fn(prompt: str) -> str:
        prompts.append(prompt)
        return _grade_lines(0.9)

    make_llm_rubric_judge(spy_fn)("Fix the bug in util.py", _transcript(), tmp_path)
    assert len(prompts) == 1
    assert "Fix the bug in util.py" in prompts[0]


def test_llm_rubric_judge_provider_error_is_credulous_by_design(tmp_path: Path) -> None:
    """Documents the sharp edge: error fallback scores are below the engagement floor,
    so no dimension gates and the verdict is COMPLETE (1.0). Calibration must surface
    this as poor agreement rather than the adapter masking it."""

    async def failing_fn(_prompt: str) -> str:
        raise ConnectionError("provider down")

    assert make_llm_rubric_judge(failing_fn)("Task", _transcript(), tmp_path) == 1.0


def test_llm_rubric_judge_reuses_one_event_loop(tmp_path: Path) -> None:
    """Regression: per-call asyncio.run broke pooled-HTTP providers on the second call
    ('Event loop is closed' — hit live with DeepSeek). All grading calls must run on the
    same persistent loop so provider connection pools stay bound to a live loop."""
    import asyncio

    loops: list[int] = []

    async def loop_spy_fn(_prompt: str) -> str:
        loops.append(id(asyncio.get_running_loop()))
        return _grade_lines(0.9)

    judge = make_llm_rubric_judge(loop_spy_fn)
    judge("Task one", _transcript(), tmp_path)
    judge("Task two", _transcript(), tmp_path)
    judge("Task three", _transcript(), tmp_path)
    assert len(loops) == 3
    assert len(set(loops)) == 1, "each call ran on a different event loop"


def test_ungradable_output_is_detected_and_voids_cleanliness(tmp_path: Path) -> None:
    """Regression for the qwen3.5 signature: transport succeeds, no grade lines parse,
    every dimension falls back below the engagement floor, verdict defaults to COMPLETE.
    Constant credulity with clean call stats — must be counted as ungradable."""
    from victor.evaluation.calibration_rubric_judge import JudgeCallStats

    async def thinking_preamble_fn(_prompt: str) -> str:
        return "Let me carefully consider whether this task was completed properly..."

    stats = JudgeCallStats()
    judge = make_llm_rubric_judge(thinking_preamble_fn, stats=stats)
    verdict = judge("Task", _transcript(), tmp_path)
    assert verdict == 1.0  # the filter's default — a grade-shaped non-grade
    assert stats.ungradable == 1
    assert not stats.clean


def test_parseable_output_does_not_count_as_ungradable(tmp_path: Path) -> None:
    from victor.evaluation.calibration_rubric_judge import JudgeCallStats

    async def good_fn(_prompt: str) -> str:
        return _grade_lines(0.9)

    stats = JudgeCallStats()
    judge = make_llm_rubric_judge(good_fn, stats=stats)
    judge("Task", _transcript(), tmp_path)
    assert stats.ungradable == 0
    assert stats.clean


# --- Provider complete_fn adapter ------------------------------------------------------------------


class _StubProvider:
    """Mimics BaseProvider.chat for the complete_fn seam."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def chat(self, messages, *, model, temperature=0.7, max_tokens=4096, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        class _Response:
            content = self.reply

        return _Response()


def test_provider_complete_fn_grades_at_temperature_zero(tmp_path: Path) -> None:
    from victor.evaluation.calibration_rubric_judge import make_provider_complete_fn

    provider = _StubProvider(_grade_lines(0.9))
    judge = make_llm_rubric_judge(make_provider_complete_fn(provider, "test-model"))
    assert judge("Task", _transcript(), tmp_path) == 1.0
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["model"] == "test-model"
    assert call["temperature"] == 0.0  # a non-deterministic grader cannot be calibrated
    assert call["messages"][0].role == "user"
    assert "TASK:" in call["messages"][0].content


class _RateLimitedProvider(_StubProvider):
    """Raises rate-limit errors for the first ``fail_n`` calls, then succeeds."""

    def __init__(self, reply: str, fail_n: int) -> None:
        super().__init__(reply)
        self.fail_n = fail_n

    async def chat(self, messages, **kwargs):
        if len(self.calls) < self.fail_n:
            self.calls.append({"failed": True})
            raise ConnectionError("Rate limit exceeded: 429 Too Many Requests")
        return await super().chat(messages, **kwargs)


def test_provider_complete_fn_retries_rate_limits_and_tracks_stats(
    tmp_path: Path,
) -> None:
    from victor.evaluation.calibration_rubric_judge import (
        JudgeCallStats,
        make_provider_complete_fn,
    )

    stats = JudgeCallStats()
    provider = _RateLimitedProvider(_grade_lines(0.9), fail_n=2)
    judge = make_llm_rubric_judge(
        make_provider_complete_fn(provider, "test-model", retry_backoff_seconds=0.0, stats=stats)
    )
    assert judge("Task", _transcript(), tmp_path) == 1.0
    assert stats.calls == 1
    assert stats.retries == 2
    assert stats.failures == 0
    assert stats.clean


def test_provider_complete_fn_exhausted_retries_count_as_failure(
    tmp_path: Path,
) -> None:
    from victor.evaluation.calibration_rubric_judge import (
        JudgeCallStats,
        make_provider_complete_fn,
    )

    stats = JudgeCallStats()
    provider = _RateLimitedProvider(_grade_lines(0.9), fail_n=99)
    judge = make_llm_rubric_judge(
        make_provider_complete_fn(
            provider,
            "test-model",
            max_attempts=3,
            retry_backoff_seconds=0.0,
            stats=stats,
        )
    )
    # LLMRubricJudge degrades the propagated error to the neutral fallback (verdict 1.0) —
    # which is exactly why stats.failures must void the measurement.
    assert judge("Task", _transcript(), tmp_path) == 1.0
    assert stats.failures == 1
    assert not stats.clean


def test_provider_complete_fn_does_not_retry_non_rate_limit_errors(
    tmp_path: Path,
) -> None:
    from victor.evaluation.calibration_rubric_judge import (
        JudgeCallStats,
        make_provider_complete_fn,
    )

    class _AuthFailProvider(_StubProvider):
        async def chat(self, messages, **kwargs):
            self.calls.append({"failed": True})
            raise PermissionError("invalid api key")

    stats = JudgeCallStats()
    provider = _AuthFailProvider("")
    judge = make_llm_rubric_judge(
        make_provider_complete_fn(provider, "test-model", retry_backoff_seconds=0.0, stats=stats)
    )
    judge("Task", _transcript(), tmp_path)
    assert len(provider.calls) == 1  # no retries on non-retryable errors
    assert stats.failures == 1 and stats.retries == 0
