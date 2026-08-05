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

"""Rubric judge adapter for the calibration harness (EVR-2 × EVR-3, ADR-009 × ADR-011).

Wraps the ADR-009 rubric completion machinery
(:mod:`victor.framework.rubric_completion`) in the blinded
``(prompt, transcript, workspace) -> float`` interface of
:class:`~victor.evaluation.judge_calibration_harness.CalibrationJudge`, so the judge that
``completion_strategy=rubric`` would put in production is exactly the judge being calibrated.

Two constructors:

- :func:`make_rubric_judge` — wraps a (sync) :class:`RubricCompletionEvaluator`. Default-
  constructed it uses the deterministic ``HeuristicRubricJudge`` fallback, so the whole
  calibration runs with zero LLM calls. Calibrating this fallback is itself informative: it
  measures whether the no-LLM path the loop degrades to is trustworthy (expect: it is not —
  that is why ADR-011 gates the LLM judge instead of trusting the fallback blindly).
- :func:`make_llm_rubric_judge` — wraps :class:`AsyncRubricCompletionEvaluator` around an
  injected async ``complete_fn(prompt) -> text`` (a local Ollama model keeps it offline).
  Synchronous per the harness contract; owns its event loop via ``asyncio.run``.

What the rubric grades: the judge-visible view is rendered into a single content block
(task, tool activity, final response) via :func:`render_judged_content`. Verifier verdicts
never appear in it — blinding is inherited from the harness contract.

Score projection: by default the returned float is the **binary completion verdict**
(``1.0`` if the DimensionAwareFilter says COMPLETE) because that is the production decision
ADR-009 routes on — calibrate what you ship. ``score_mode="aggregate"`` returns the
confidence-weighted mean instead, for inspecting judge behavior below the decision boundary.

Known sharp edge worth watching in reports: when every dimension comes back at confidence
below the filter's floor (e.g. the LLM judge's error fallback scores everything at
confidence 0.2 < floor 0.25), no dimension is *engaged*, so the filter reports COMPLETE.
A judge whose provider is failing therefore looks systematically credulous — which the
α measurement will surface as poor agreement, and which is precisely why the gate exists.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Literal, Optional

from victor.evaluation.judge_calibration_harness import (
    CalibrationJudge,
    PersistentLoopRunner,
    Transcript,
)
from victor.framework.rubric_completion import (
    AsyncRubricCompletionEvaluator,
    LLMRubricJudge,
    RubricCompletionEvaluator,
    RubricCompletionResult,
)

if TYPE_CHECKING:
    from victor.providers.base import BaseProvider

ScoreMode = Literal["complete", "aggregate"]

# The corpus family is task metadata, not gold; but the blinded judge signature does not
# carry it, so all calibration content is graded under one rubric family.
CALIBRATION_TASK_FAMILY = "calibration"

_MAX_TOOL_STEPS_RENDERED = 30
_MAX_FILE_BYTES_RENDERED = 2_000
_MAX_TOTAL_FILE_BYTES_RENDERED = 8_000


def _render_workspace(workspace: Path) -> list[str]:
    """Shallow listing plus capped contents of small text files.

    Contents matter: without them a judge cannot assess the *correctness* of an edit even
    in principle (first calibration run: nearly every genuinely-solved code-fix task was
    judged incomplete on a names-only view). Showing workspace state is within the blinding
    contract — the judge sees what exists, never what a verifier concluded about it.
    """
    try:
        files = sorted(p for p in workspace.rglob("*") if p.is_file())
    except OSError:
        return ["(unreadable)"]
    if not files:
        return ["(empty)"]
    lines: list[str] = []
    budget = _MAX_TOTAL_FILE_BYTES_RENDERED
    for path in files:
        rel = path.relative_to(workspace)
        try:
            size = path.stat().st_size
            if size > _MAX_FILE_BYTES_RENDERED or budget <= 0:
                lines.append(f"--- {rel} ({size} bytes, content omitted) ---")
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            lines.append(f"--- {rel} (unreadable) ---")
            continue
        budget -= len(text)
        lines.append(f"--- {rel} ---")
        lines.append(text.rstrip("\n"))
    return lines


def render_judged_content(prompt: str, transcript: Transcript, workspace: Path) -> str:
    """Render the blinded calibration view into the content block the rubric grades.

    Includes the task, the tool activity (evidence the ``tool_grounding`` dimension needs),
    the workspace state (small-file contents, capped), and the final response. Verifier
    verdicts are structurally absent.
    """
    tool_steps = transcript.tool_steps()
    tool_lines = [f"- {step.content}" for step in tool_steps[:_MAX_TOOL_STEPS_RENDERED]]
    if len(tool_steps) > _MAX_TOOL_STEPS_RENDERED:
        tool_lines.append(f"- ... {len(tool_steps) - _MAX_TOOL_STEPS_RENDERED} more")
    # WORKSPACE STATE renders last: the framework's rubric prompt caps content length,
    # and workspace detail is the section that can safely lose its tail.
    return "\n".join(
        [
            "TASK:",
            prompt,
            "",
            "TOOL ACTIVITY:",
            *(tool_lines or ["(none)"]),
            "",
            "FINAL RESPONSE:",
            transcript.final_message,
            "",
            "WORKSPACE STATE:",
            *_render_workspace(workspace),
        ]
    )


def _project(result: RubricCompletionResult, score_mode: ScoreMode) -> float:
    if score_mode == "complete":
        return 1.0 if result.complete else 0.0
    return result.aggregate


def make_rubric_judge(
    evaluator: Optional[RubricCompletionEvaluator] = None,
    *,
    score_mode: ScoreMode = "complete",
) -> CalibrationJudge:
    """Wrap a sync rubric evaluator as a CalibrationJudge.

    With no arguments this is the fully deterministic ADR-009 fallback path
    (default generator + HeuristicRubricJudge) — zero LLM calls.
    """
    rubric_evaluator = evaluator or RubricCompletionEvaluator()

    def judge(prompt: str, transcript: Transcript, workspace: Path) -> float:
        result = rubric_evaluator.evaluate(
            task_family=CALIBRATION_TASK_FAMILY,
            content=render_judged_content(prompt, transcript, workspace),
        )
        return _project(result, score_mode)

    return judge


@dataclass
class JudgeCallStats:
    """Integrity accounting for judge grading calls.

    Two ways a calibration silently stops measuring the judge, both producing neutral,
    filter-inert scores that masquerade as credulous grades:

    - **Transport failure** (``failures``): the provider call errored after retries; the
      ``LLMRubricJudge`` error fallback kicks in. Observed live: a fully rate-limited
      GLM-5.2 run reproduced the heuristic judge's α to three decimals.
    - **Ungradable output** (``ungradable``): the call SUCCEEDED but no dimension parsed
      from the reply — wrong format, or a thinking-family model exhausting max_tokens on
      reasoning before emitting any ``score=`` line. Observed live: qwen3.5:27b returned
      48/48 clean calls and judged everything complete, again matching the heuristic's α
      exactly.

    Check ``clean`` before trusting a report built from this judge.
    """

    calls: int = 0
    retries: int = 0
    failures: int = 0
    ungradable: int = 0

    @property
    def clean(self) -> bool:
        return self.failures == 0 and self.ungradable == 0

    def summary(self) -> str:
        return (
            f"calls={self.calls} retries={self.retries} "
            f"failures={self.failures} ungradable={self.ungradable}"
        )


def _no_dimension_graded(result: RubricCompletionResult) -> bool:
    """True when every dimension carries the parse/error fallback signature.

    All three ungraded paths in :mod:`victor.framework.rubric_completion` — missing row,
    present-but-unparseable row, provider-error fallback — emit ``score=0.5`` with
    ``confidence<=0.2`` (below the engagement floor). A result made entirely of such
    scores contains zero judge signal; the filter's COMPLETE verdict is a default, not a
    grade.
    """
    return bool(result.scores) and all(
        s.score == 0.5 and s.confidence <= 0.2 for s in result.scores
    )


def _looks_rate_limited(exc: Exception) -> bool:
    if getattr(exc, "retryable", False):
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text


def make_provider_complete_fn(
    provider: "BaseProvider",
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_attempts: int = 5,
    retry_backoff_seconds: float = 20.0,
    pre_call_delay_seconds: float = 0.0,
    stats: Optional[JudgeCallStats] = None,
) -> Callable[[str], Awaitable[str]]:
    """Adapt any Victor provider into the ``complete_fn`` seam of the LLM rubric judge.

    Grading defaults to temperature 0.0 for reproducibility (the ADR-013 concern applies
    doubly to a judge: a non-deterministic grader cannot be meaningfully calibrated).

    Unlike the production completion path (which correctly degrades to neutral rather than
    stall the agentic loop), calibration should WAIT: rate-limited calls retry up to
    ``max_attempts`` with linear backoff (``retry_backoff_seconds`` × attempt), and
    ``pre_call_delay_seconds`` paces successive tasks under strict provider RPM caps.
    Pass a :class:`JudgeCallStats` to detect runs whose α is voided by residual failures.

    Example (fully offline)::

        from victor.providers.registry import ProviderRegistry

        provider = ProviderRegistry.create("ollama")
        judge = make_llm_rubric_judge(
            make_provider_complete_fn(provider, "qwen2.5-coder:7b")
        )
    """

    async def complete_fn(prompt: str) -> str:
        from victor.providers.base import Message

        if stats:
            stats.calls += 1
        if pre_call_delay_seconds > 0:
            await asyncio.sleep(pre_call_delay_seconds)
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = await provider.chat(
                    [Message(role="user", content=prompt)],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.content or ""
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts or not _looks_rate_limited(exc):
                    break
                if stats:
                    stats.retries += 1
                await asyncio.sleep(retry_backoff_seconds * attempt)
        if stats:
            stats.failures += 1
        assert last_exc is not None
        raise last_exc

    return complete_fn


def make_llm_rubric_judge(
    complete_fn: Callable[[str], Awaitable[str]],
    *,
    evaluator: Optional[AsyncRubricCompletionEvaluator] = None,
    score_mode: ScoreMode = "complete",
    stats: Optional[JudgeCallStats] = None,
) -> CalibrationJudge:
    """Wrap the LLM-backed rubric evaluator (single grading call per task) as a CalibrationJudge.

    ``complete_fn(prompt) -> text`` is the provider seam — pass a local-model completion for a
    fully offline run. ``evaluator`` overrides the default AsyncRubricCompletionEvaluator (e.g.
    to supply a task-adaptive generator). All grading calls share ONE persistent event loop
    (provider connection pools bind to their creating loop; per-call ``asyncio.run`` broke
    every call after the first against pooled-HTTP providers). Raises RuntimeError if invoked
    from a running loop — use the async evaluator directly there.
    """
    rubric_evaluator = evaluator or AsyncRubricCompletionEvaluator(LLMRubricJudge(complete_fn))
    runner = PersistentLoopRunner()

    def judge(prompt: str, transcript: Transcript, workspace: Path) -> float:
        result = runner.run(
            rubric_evaluator.aevaluate(
                task_family=CALIBRATION_TASK_FAMILY,
                content=render_judged_content(prompt, transcript, workspace),
            )
        )
        if stats is not None and _no_dimension_graded(result):
            stats.ungradable += 1
        return _project(result, score_mode)

    return judge
