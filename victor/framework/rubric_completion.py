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

"""Rubric-based completion evaluation (EVR-3, ADR-009, FEP-0008 Phase A).

Replaces a single scalar completion score with **task-adaptive rubric dimensions**, each scored with
a confidence, and gated by a **DimensionAwareFilter** that requires *every engaged* dimension to
clear its threshold before COMPLETE (AdaRubric 2603.21362). This fixes the two failure modes the
FEP-0007 cutover exposed: one strong axis masking a failed one (premature stop), and a finished
answer being under-scored overall (restatement).

Two evaluators share the same rubric model + DimensionAwareFilter:

- :class:`RubricCompletionEvaluator` (sync) with a per-dimension :class:`RubricJudge`. The default
  :class:`HeuristicRubricJudge` needs no LLM (the safe baseline).
- :class:`AsyncRubricCompletionEvaluator` (EVR-3c) with :class:`LLMRubricJudge`, which scores all
  dimensions in a single LLM call via an injected async ``complete_fn`` — the task-adaptive judge
  AdaRubric's gains come from.

``AgenticLoop`` selects between them via ``completion_strategy="rubric"`` (EVR-3b): an injected
``rubric_complete_fn`` activates the LLM path, otherwise the heuristic baseline. Default strategy is
"enhanced" so behavior is unchanged until the rubric path is proven on the batteries.

**Reliability note:** before trusting the LLM judge in production, gate it with EVR-2
(``JudgeReliabilityGate`` over a human-labeled set) and consider wrapping it with
``OrderSwapEnsembleJudge`` for position-bias robustness; on the α-gate failing, fall back to the
heuristic judge. Rubrics are cached per task family (>95% cost cut, AdaRubric).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional, Protocol, Sequence

from victor.core.utils import clamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RubricDimension:
    """One orthogonal completion axis: a weight, a pass threshold, and grading guidance."""

    name: str
    weight: float = 0.25
    threshold: float = 0.6
    description: str = ""


@dataclass(frozen=True)
class Rubric:
    """A task-family rubric: a set of weighted, thresholded dimensions."""

    task_family: str
    dimensions: tuple[RubricDimension, ...]

    def validate(self) -> "Rubric":
        """Validate dimension names are distinct and weights are sane; return self.

        Weights are normalized to sum to 1.0 by the caller if needed; here we only reject empties and
        duplicate names (AdaRubric requires orthogonal, non-overlapping dimensions).
        """
        if not self.dimensions:
            raise ValueError("rubric must have at least one dimension")
        names = [d.name for d in self.dimensions]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate rubric dimension names: {names}")
        return self

    def normalized(self) -> "Rubric":
        """Return a copy with weights normalized to sum to 1.0 (no-op if already ~1.0)."""
        total = sum(d.weight for d in self.dimensions)
        if total <= 0:
            raise ValueError("rubric weights must be positive")
        if math.isclose(total, 1.0, abs_tol=1e-6):
            return self
        dims = tuple(
            RubricDimension(d.name, d.weight / total, d.threshold, d.description)
            for d in self.dimensions
        )
        return Rubric(self.task_family, dims)


@dataclass(frozen=True)
class RubricDimensionScore:
    """A judge's score for one rubric dimension. ``score``/``confidence`` are in [0, 1]."""

    name: str
    score: float
    confidence: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RubricCompletionResult:
    """Outcome of a rubric completion evaluation."""

    complete: bool
    aggregate: float
    scores: tuple[RubricDimensionScore, ...]
    failed_dimensions: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "aggregate": round(self.aggregate, 4),
            "failed_dimensions": list(self.failed_dimensions),
            "reason": self.reason,
            "scores": [s.to_dict() for s in self.scores],
        }


def confidence_weighted_mean(scores: Sequence[RubricDimensionScore]) -> float:
    """Confidence-weighted mean of dimension scores (un-engaged axes contribute less)."""
    total_w = sum(s.confidence for s in scores)
    if total_w <= 0:
        return 0.0
    return sum(s.score * s.confidence for s in scores) / total_w


class RubricJudge(Protocol):
    """Scores a single rubric dimension for given content. The LLM-judge seam (wrap + gate it)."""

    def score(
        self, dimension: RubricDimension, content: str, context: Mapping[str, Any]
    ) -> RubricDimensionScore:
        """Return the dimension's :class:`RubricDimensionScore` for ``content``."""
        ...


class RubricGenerator(Protocol):
    """Produces a :class:`Rubric` for a task family. The task-adaptive (LLM) generator seam."""

    def generate(self, task_family: str, context: Mapping[str, Any]) -> Rubric:
        """Return a validated rubric for ``task_family``."""
        ...


# General default rubric (deterministic, non-LLM). A task-adaptive LLM generator supersedes this in
# production; these four orthogonal axes are a sensible baseline for agent task completion.
_DEFAULT_DIMENSIONS = (
    RubricDimension("correctness", 0.40, 0.6, "Is the result/answer correct for the request?"),
    RubricDimension("tool_grounding", 0.25, 0.5, "Is the answer grounded in tool/file evidence?"),
    RubricDimension(
        "completeness", 0.25, 0.6, "Does it address the full request, nothing missing?"
    ),
    RubricDimension("recovery", 0.10, 0.4, "Were tool failures recovered from?"),
)


@dataclass
class DefaultRubricGenerator:
    """Returns a fixed, sensible rubric (the non-LLM baseline)."""

    dimensions: tuple[RubricDimension, ...] = _DEFAULT_DIMENSIONS

    def generate(self, task_family: str, context: Mapping[str, Any]) -> Rubric:
        return Rubric(task_family, self.dimensions).validate().normalized()


@dataclass
class HeuristicRubricJudge:
    """Deterministic, no-LLM fallback judge.

    A coarse content-shape heuristic — substantial, structured content scores higher — used only as
    the fallback when no reliable LLM judge is available (EVR-2 gate). Low/moderate confidence by
    design so the DimensionAwareFilter does not over-trust it.
    """

    def score(
        self, dimension: RubricDimension, content: str, context: Mapping[str, Any]
    ) -> RubricDimensionScore:
        text = (content or "").strip()
        length = len(text)
        has_structure = any(m in text for m in ("\n-", "\n*", "\n#", "1.", "```"))
        base = clamp(length / 400.0, 0.0, 1.0)
        if has_structure:
            base = clamp(base + 0.15, 0.0, 1.0)
        return RubricDimensionScore(
            dimension.name,
            base,
            0.3,
            f"heuristic: len={length}, structured={has_structure}",
        )


@dataclass
class RubricCache:
    """Per-task-family rubric cache (AdaRubric caches rubrics per family for ~free reuse)."""

    _store: dict[str, Rubric] = field(default_factory=dict)

    def get_or_generate(
        self, task_family: str, generator: RubricGenerator, context: Mapping[str, Any]
    ) -> Rubric:
        cached = self._store.get(task_family)
        if cached is not None:
            return cached
        rubric = generator.generate(task_family, context).validate().normalized()
        self._store[task_family] = rubric
        return rubric

    def clear(self) -> None:
        self._store.clear()


@dataclass
class DimensionAwareFilter:
    """COMPLETE only when every *engaged* dimension clears its threshold (AdaRubric).

    A dimension is "engaged" when its score confidence is at or above ``confidence_floor``; un-engaged
    dimensions (e.g. recovery on a no-error task) are not gated. This is the structural fix that stops
    one strong axis from masking a failed one — and stops an under-scored axis the trajectory never
    engaged from blocking completion.
    """

    confidence_floor: float = 0.25

    def failed(
        self, rubric: Rubric, scores: Sequence[RubricDimensionScore]
    ) -> tuple[RubricDimensionScore, ...]:
        thresholds = {d.name: d.threshold for d in rubric.dimensions}
        return tuple(
            s
            for s in scores
            if s.confidence >= self.confidence_floor and s.score < thresholds.get(s.name, 0.6)
        )


class RubricCompletionEvaluator:
    """Generate → score → DimensionAwareFilter, producing a :class:`RubricCompletionResult`.

    Default-constructed it is fully deterministic (default generator + heuristic judge), so it runs
    with no LLM. Inject an LLM judge (wrapped per the module note) and a task-adaptive generator for
    the real thing.
    """

    def __init__(
        self,
        *,
        judge: Optional[RubricJudge] = None,
        generator: Optional[RubricGenerator] = None,
        dimension_filter: Optional[DimensionAwareFilter] = None,
        cache: Optional[RubricCache] = None,
    ) -> None:
        self._judge: RubricJudge = judge or HeuristicRubricJudge()
        self._generator: RubricGenerator = generator or DefaultRubricGenerator()
        self._filter = dimension_filter or DimensionAwareFilter()
        self._cache = cache or RubricCache()

    def evaluate(
        self,
        *,
        task_family: str,
        content: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> RubricCompletionResult:
        ctx: Mapping[str, Any] = context or {}
        rubric = self._cache.get_or_generate(task_family, self._generator, ctx)
        scores = tuple(self._judge.score(d, content, ctx) for d in rubric.dimensions)
        return _assemble_result(rubric, scores, self._filter)


def _assemble_result(
    rubric: Rubric,
    scores: Sequence[RubricDimensionScore],
    dimension_filter: DimensionAwareFilter,
) -> RubricCompletionResult:
    """Apply the DimensionAwareFilter + confidence-weighted aggregate to per-dimension scores."""
    failed = dimension_filter.failed(rubric, scores)
    aggregate = confidence_weighted_mean(scores)
    complete = not failed
    if complete:
        reason = f"all engaged dimensions cleared their thresholds (aggregate={aggregate:.2f})"
    else:
        names = ", ".join(f"{f.name}={f.score:.2f}" for f in failed)
        reason = f"dimensions below threshold: {names}"
    return RubricCompletionResult(
        complete=complete,
        aggregate=aggregate,
        scores=tuple(scores),
        failed_dimensions=tuple(f.name for f in failed),
        reason=reason,
    )


# ----------------------------------------------------------------------------------------------------
# LLM-backed judge (EVR-3c) — the task-adaptive judge AdaRubric's gains come from.
# ----------------------------------------------------------------------------------------------------

_SCORE_RE = re.compile(r"score\s*[=:]\s*(1(?:\.0+)?|0?\.\d+|[01])", re.IGNORECASE)
_CONF_RE = re.compile(r"conf(?:idence)?\s*[=:]\s*(1(?:\.0+)?|0?\.\d+|[01])", re.IGNORECASE)


def _build_rubric_prompt(rubric: Rubric, content: str) -> str:
    """A compact grading prompt: one ``<name>: score=.. confidence=..`` line per dimension."""
    lines = [
        "Grade whether the RESPONSE satisfies the task on each dimension below.",
        "For EACH dimension output exactly one line: <name>: score=<0.0-1.0> confidence=<0.0-1.0>",
        "score = how well the response satisfies the dimension; confidence = how sure you are.",
        # The numeric convention matters: DimensionAwareFilter treats confidence >= 0.25 as an
        # engaged dimension. A vague "use LOW confidence" led judges to emit 0.3 for
        # not-applicable axes (e.g. recovery with no failures), wrongly blocking completion.
        "If a dimension does not apply to this task (e.g. recovery when nothing failed), "
        "output exactly: score=0.0 confidence=0.0 for it.",
        "",
        "Dimensions:",
    ]
    lines += [f"- {d.name}: {d.description}" for d in rubric.dimensions]
    lines += [
        "",
        "RESPONSE:",
        (content or "")[:4000],
        "",
        "Grades (one line per dimension):",
    ]
    return "\n".join(lines)


def _parse_rubric_scores(rubric: Rubric, text: str) -> tuple[RubricDimensionScore, ...]:
    """Parse one ``score``/``confidence`` per dimension from the judge's text; tolerant fallbacks."""
    out = []
    rows = (text or "").splitlines()
    for d in rubric.dimensions:
        row = next((r for r in rows if d.name.lower() in r.lower()), "")
        s_match = _SCORE_RE.search(row)
        c_match = _CONF_RE.search(row)
        if not row:
            out.append(RubricDimensionScore(d.name, 0.5, 0.2, "llm: dimension not in output"))
            continue
        # Unparseable rows must default BELOW the DimensionAwareFilter engagement floor
        # (0.25): a row like "correctness: looks good" carries no gradable verdict and
        # must not gate completion as an engaged 0.5-score failure.
        score = clamp(float(s_match.group(1)), 0.0, 1.0) if s_match else 0.5
        confidence = clamp(float(c_match.group(1)), 0.0, 1.0) if c_match else 0.2
        out.append(RubricDimensionScore(d.name, score, confidence, "llm"))
    return tuple(out)


@dataclass
class LLMRubricJudge:
    """Scores all rubric dimensions in a SINGLE LLM call via an injected async ``complete_fn``.

    Provider-agnostic: ``complete_fn(prompt) -> text`` is supplied by the caller (provider or edge
    model). One call per completion check keeps latency/cost bounded; the result should be cached per
    task family upstream. For position-bias robustness it can be wrapped to run under multiple
    dimension orderings (EVR-2 ``OrderSwapEnsembleJudge``); a provider error degrades to neutral,
    low-confidence scores so completion never hard-fails on the judge.
    """

    complete_fn: Callable[[str], Awaitable[str]]

    async def score_rubric(
        self, rubric: Rubric, content: str, context: Mapping[str, Any]
    ) -> tuple[RubricDimensionScore, ...]:
        try:
            text = await self.complete_fn(_build_rubric_prompt(rubric, content))
        except Exception as exc:  # degrade, don't crash the loop's completion check
            logger.warning("LLM rubric judge call failed (%s); using neutral fallback", exc)
            return tuple(
                RubricDimensionScore(d.name, 0.5, 0.2, "llm error fallback")
                for d in rubric.dimensions
            )
        return _parse_rubric_scores(rubric, text or "")


class AsyncRubricCompletionEvaluator:
    """Async counterpart of :class:`RubricCompletionEvaluator` for an LLM (single-call) judge."""

    def __init__(
        self,
        judge: LLMRubricJudge,
        *,
        generator: Optional[RubricGenerator] = None,
        dimension_filter: Optional[DimensionAwareFilter] = None,
        cache: Optional[RubricCache] = None,
    ) -> None:
        self._judge = judge
        self._generator: RubricGenerator = generator or DefaultRubricGenerator()
        self._filter = dimension_filter or DimensionAwareFilter()
        self._cache = cache or RubricCache()

    async def aevaluate(
        self,
        *,
        task_family: str,
        content: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> RubricCompletionResult:
        ctx: Mapping[str, Any] = context or {}
        rubric = self._cache.get_or_generate(task_family, self._generator, ctx)
        scores = await self._judge.score_rubric(rubric, content, ctx)
        return _assemble_result(rubric, scores, self._filter)
