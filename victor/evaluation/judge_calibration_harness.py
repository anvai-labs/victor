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

"""Offline judge-calibration harness (EVR-2, ADR-011).

The :class:`~victor.evaluation.judge_calibration.JudgeReliabilityGate` needs a *gold* label
sequence to measure a judge against — but nothing requires those labels to come from human
annotators. This harness produces them from **programmatic verifiers**: each
:class:`VerifiableTask` materializes a fixture workspace, an executor (a real agent, a scripted
stand-in, or a replayed transcript) attempts the task, and the task's verifier computes the gold
completion label from the *actual workspace state* — the effect-grounded ground truth of ADR-010.

The judge is **blinded by construction**: it receives only the task prompt, the execution
transcript, and a read-only view of the workspace. Verifier verdicts are never part of its input,
so measured agreement is agreement, not leakage.

The whole loop runs offline — no LLM, no network — which makes it cheap enough to run in CI and
to smoke-test judge changes before spending tokens on SWE-bench-scale calibration.

Usage::

    from victor.evaluation.judge_calibration_harness import JudgeCalibrationHarness
    from victor.evaluation.calibration_corpus import default_corpus

    harness = JudgeCalibrationHarness(default_corpus(variants=8))
    report = harness.run(executor=my_executor, judge=my_judge)
    print(report.gate_decision)          # trusted / not trusted, with reason
    report.save(Path("calibration_report.json"))

Interpretation caveats (record them wherever a report is consumed):

- Gold labels are binary completion verdicts, so a passing gate trusts the judge **for
  completion decisions only** — it says nothing about the calibration of interval rubric scores.
- Agreement is reported per task family as well as overall; a judge may clear the gate on one
  distribution and fail on another. Graduate flags on the family you actually route through it.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Hashable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
)

from victor.evaluation.judge_calibration import (
    GateDecision,
    JudgeReliability,
    JudgeReliabilityGate,
    evaluate_judge_agreement,
)

_T = TypeVar("_T")


class PersistentLoopRunner:
    """Runs coroutines on ONE long-lived event loop across many sync calls.

    ``asyncio.run()`` per call creates and closes a fresh loop each time — but providers
    hold httpx ``AsyncClient`` connection pools whose keep-alive connections bind to the
    loop that created them, so the second call dies with ``RuntimeError: Event loop is
    closed`` (observed live: DeepSeek calibration failed on every call after the first).
    One loop for the runner's lifetime keeps pooled connections valid.

    Must be driven from sync code only — ``run`` raises if a loop is already running in
    this thread. The loop is closed on ``close()`` or interpreter exit.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def close(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None


# ----------------------------------------------------------------------------------------------------
# Transcript + task model
# ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptStep:
    """One observable step of an execution: a tool action or an assistant message."""

    kind: str  # "tool" | "message"
    content: str


@dataclass(frozen=True)
class Transcript:
    """What the judge is allowed to see about an execution (besides the workspace)."""

    steps: tuple[TranscriptStep, ...] = ()
    final_message: str = ""

    def tool_steps(self) -> tuple[TranscriptStep, ...]:
        return tuple(step for step in self.steps if step.kind == "tool")


@dataclass(frozen=True)
class VerifiableTask:
    """A task whose completion is decidable by a programmatic verifier.

    Attributes:
        task_id: Unique, stable identifier (also used for report grouping/debugging).
        family: Task-distribution family (e.g. ``file-create``, ``code-fix``, ``qa``).
            Agreement is reported per family so gate decisions can be scoped.
        prompt: The instruction given to the executor and, verbatim, to the judge.
        setup: Materializes fixture files into an empty workspace directory.
        verify: Computes the gold label (1.0 complete / 0.0 not) from the workspace
            state and transcript AFTER execution. Never exposed to the judge.
        solve: Reference solution used by scripted executors and corpus self-tests;
            applies the correct change to the workspace.
        reference_answer: For tasks verified from the transcript (e.g. QA), the final
            message a correct execution would produce; scripted executors emit it when
            they solve the task.
        solve_flawed: Applies a plausible-but-WRONG change — the workspace superficially
            looks solved (a file was written, a function changed) but ``verify`` returns
            0.0. This is the judge-discrimination test: a judge that pattern-matches
            "work was done" passes it; one that actually verifies catches it.
        reference_answer_flawed: The transcript-verified counterpart of ``solve_flawed``
            (e.g. a QA answer with a wrong number).
    """

    task_id: str
    family: str
    prompt: str
    setup: Callable[[Path], None]
    verify: Callable[[Path, Transcript], float]
    solve: Optional[Callable[[Path], None]] = None
    reference_answer: Optional[str] = None
    solve_flawed: Optional[Callable[[Path], None]] = None
    reference_answer_flawed: Optional[str] = None


class CalibrationExecutor(Protocol):
    """Attempts a task in the given workspace and returns what happened."""

    def __call__(self, task: VerifiableTask, workspace: Path) -> Transcript: ...


class CalibrationJudge(Protocol):
    """Scores completion in [0, 1] from prompt + transcript + read-only workspace.

    Gold labels and verifier internals are structurally absent from this signature —
    that is the blinding contract.
    """

    def __call__(self, prompt: str, transcript: Transcript, workspace: Path) -> float: ...


# ----------------------------------------------------------------------------------------------------
# Harness + report
# ----------------------------------------------------------------------------------------------------


def binary_categorize(score: float) -> Hashable:
    """Default categorizer for Cohen's κ: complete at ≥ 0.5."""
    return score >= 0.5


@dataclass(frozen=True)
class CalibrationSample:
    """One (task, gold, judged) observation."""

    task_id: str
    family: str
    gold: float
    judged: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "gold": self.gold,
            "judged": self.judged,
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Agreement measurements plus the resulting gate decision.

    ``overall`` and ``per_family`` are :class:`JudgeReliability` measurements; the gate decision
    is computed from ``overall``. Per-family α on small corpora is noisy — treat families with
    n < 30 as directional, not gating evidence.
    """

    samples: tuple[CalibrationSample, ...]
    overall: JudgeReliability
    per_family: dict[str, JudgeReliability]
    gate_decision: GateDecision
    alpha_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": len(self.samples),
            "alpha_threshold": self.alpha_threshold,
            "gate": {
                "trusted": self.gate_decision.trusted,
                "reason": self.gate_decision.reason,
                "scope": "binary completion verdicts only (gold labels are programmatic "
                "verifier outcomes, not interval rubric scores)",
            },
            "overall": self.overall.to_dict(),
            "per_family": {family: rel.to_dict() for family, rel in self.per_family.items()},
            "samples": [sample.to_dict() for sample in self.samples],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


@dataclass
class JudgeCalibrationHarness:
    """Runs a corpus through (executor, judge) and measures agreement against verifier gold.

    Per task: a fresh workspace directory is created under ``workspace_root``, ``task.setup``
    materializes fixtures, the executor attempts the task, the judge scores the blinded view,
    and the verifier computes gold. Judge scoring happens BEFORE verification so no verifier
    side effects can leak into the judged view.
    """

    tasks: Sequence[VerifiableTask]
    gate: JudgeReliabilityGate = field(default_factory=JudgeReliabilityGate)
    categorize: Callable[[float], Hashable] = binary_categorize

    def run(
        self,
        executor: CalibrationExecutor,
        judge: CalibrationJudge,
        *,
        workspace_root: Optional[Path] = None,
        keep_workspaces: bool = False,
    ) -> CalibrationReport:
        """Run the corpus against a single judge and return its :class:`CalibrationReport`.

        Thin wrapper over :meth:`run_multi_judge` — see it for the workspace-lifetime and
        blinding contract.
        """
        return self.run_multi_judge(
            executor,
            {"judge": judge},
            workspace_root=workspace_root,
            keep_workspaces=keep_workspaces,
        )["judge"]

    def run_multi_judge(
        self,
        executor: CalibrationExecutor,
        judges: Mapping[str, CalibrationJudge],
        *,
        workspace_root: Optional[Path] = None,
        keep_workspaces: bool = False,
        two_phase: bool = False,
    ) -> dict[str, CalibrationReport]:
        """Generate one trajectory + gold per task, then score EVERY judge on the shared
        trajectories; return a per-judge-name :class:`CalibrationReport`.

        Running the executor once (not once per judge) is essential for expensive or
        non-deterministic executors: a real agent costs Nx to run per judge and — worse —
        produces DIFFERENT trajectories each time, which would make the cross-judge
        comparison meaningless. Here every judge scores the identical (transcript,
        workspace) pair.

        Blinding + ordering: setup → executor → ALL judges score → verify for gold.
        Verification runs after judging so no verifier side effect (e.g. ``__pycache__``
        from importing a fixed module) can reach any judge's view; judges are read-only, so
        they observe the same workspace state as each other.

        ``two_phase`` controls the call *order* across tasks — results are identical either
        way (deterministic judges) but throughput differs when the executor and judges use
        DIFFERENT models on a single local inference server (e.g. a 30B agent + a 70B judge
        on one Ollama):

        - ``False`` (default): interleave per task (execute→judge→verify, next task). The
          server swaps executor↔judge models every task — Nx model loads.
        - ``True``: run all executors first (executor model resident throughout), then all
          judges (judge model resident throughout), then verify. One model swap total. Use
          for real-agent runs; harmless (just reordering) for scripted ones.

        When ``workspace_root`` is given the caller owns its lifetime; otherwise a temp dir
        is created and removed on exit (``keep_workspaces=True`` retains it for inspection).
        """
        import shutil
        import tempfile

        caller_owned = workspace_root is not None
        root = workspace_root or Path(tempfile.mkdtemp(prefix="judge_calibration_"))
        try:
            # Phase 1 — execute every task; keep (task, workspace, transcript) records.
            records: list[tuple[VerifiableTask, Path, Transcript]] = []
            # Per-task judge scores + gold, filled either inline (interleaved) or in phase 2/3.
            judged_by_task: list[dict[str, float]] = []
            gold_by_task: list[float] = []
            for index, task in enumerate(self.tasks):
                workspace = root / f"{index:04d}_{task.task_id}"
                workspace.mkdir(parents=True, exist_ok=False)
                task.setup(workspace)
                transcript = executor(task, workspace)
                records.append((task, workspace, transcript))
                if not two_phase:
                    # Judge BEFORE verify so verifier side effects never reach the judge.
                    judged_by_task.append(
                        {n: float(j(task.prompt, transcript, workspace)) for n, j in judges.items()}
                    )
                    gold_by_task.append(float(task.verify(workspace, transcript)))

            if two_phase:
                # Phase 2 — all judges (judge model resident). Phase 3 — verify (no model).
                judged_by_task = [
                    {n: float(j(t.prompt, tr, ws)) for n, j in judges.items()}
                    for (t, ws, tr) in records
                ]
                gold_by_task = [float(t.verify(ws, tr)) for (t, ws, tr) in records]

            per_judge: dict[str, list[CalibrationSample]] = {name: [] for name in judges}
            for (task, _ws, _tr), judged, gold in zip(records, judged_by_task, gold_by_task):
                for name in judges:
                    per_judge[name].append(
                        CalibrationSample(task.task_id, task.family, gold, judged[name])
                    )
            return {name: self._report(samples) for name, samples in per_judge.items()}
        finally:
            # Only remove what we created; a caller-provided root is theirs to keep.
            if not caller_owned and not keep_workspaces:
                shutil.rmtree(root, ignore_errors=True)

    def _report(self, samples: Sequence[CalibrationSample]) -> CalibrationReport:
        overall = self._agreement(samples)
        per_family: dict[str, JudgeReliability] = {}
        for family in sorted({s.family for s in samples}):
            per_family[family] = self._agreement([s for s in samples if s.family == family])
        return CalibrationReport(
            samples=tuple(samples),
            overall=overall,
            per_family=per_family,
            gate_decision=self.gate.decide(overall),
            alpha_threshold=self.gate.alpha_threshold,
        )

    def _agreement(self, samples: Sequence[CalibrationSample]) -> JudgeReliability:
        return evaluate_judge_agreement(
            [s.gold for s in samples],
            [s.judged for s in samples],
            level="nominal",
            categorize=self.categorize,
        )


# ----------------------------------------------------------------------------------------------------
# Scripted executors (offline stand-ins for a real agent)
# ----------------------------------------------------------------------------------------------------


def make_scripted_executor(
    should_solve: Callable[[VerifiableTask], bool],
) -> CalibrationExecutor:
    """An executor that solves tasks selected by ``should_solve`` and fakes the rest.

    Solved tasks apply ``task.solve`` and emit a tool step; unsolved tasks emit NO tool steps
    but still *claim* success in the final message — the "completion-without-effect" failure
    mode (ADR-010) that a trustworthy completion judge must catch. Mixing the two produces the
    label variance κ/α need.
    """

    def executor(task: VerifiableTask, workspace: Path) -> Transcript:
        if should_solve(task):
            if task.solve is None:
                raise ValueError(f"task {task.task_id} has no reference solution")
            task.solve(workspace)
            # Keep the claim clean: an earlier version echoed a mid-word-truncated
            # prompt here, which a perceptive judge (correctly) penalized as a
            # defect — poisoning every solved-task calibration label.
            final = task.reference_answer or "Done — I completed the requested task."
            return Transcript(
                steps=(
                    TranscriptStep(kind="tool", content=f"edit workspace for {task.task_id}"),
                    TranscriptStep(kind="message", content="Applied the change."),
                ),
                final_message=final,
            )
        return Transcript(
            steps=(TranscriptStep(kind="message", content="Working on it."),),
            final_message="Done — the task is complete.",
        )

    return executor


def alternating_scripted_executor(period: int = 2) -> CalibrationExecutor:
    """Deterministically solves all but every ``period``-th task (by corpus order).

    Stateful across calls; create a fresh instance per harness run.
    """
    counter = {"i": -1}

    def should_solve(_task: VerifiableTask) -> bool:
        counter["i"] += 1
        return counter["i"] % period != period - 1

    return make_scripted_executor(should_solve)


def make_outcome_executor(
    outcome_of: Callable[[VerifiableTask], str],
) -> CalibrationExecutor:
    """Executor whose per-task outcome is chosen by ``outcome_of`` → ``solve``/``flaw``/``fake``.

    - ``solve``: apply ``task.solve`` (gold=1, genuinely correct).
    - ``flaw``: apply ``task.solve_flawed`` — a plausible-but-wrong change (gold=0, yet the
      workspace looks solved and the transcript shows tool activity). The discrimination
      test: a judge that pattern-matches "work happened" passes it; one that verifies fails
      it. Falls back to ``fake`` if the task defines no flawed solver.
    - ``fake``: no work, but claim success (gold=0, the completion-without-effect failure
      of ADR-010).
    """

    def executor(task: VerifiableTask, workspace: Path) -> Transcript:
        outcome = outcome_of(task)
        if outcome == "solve":
            if task.solve is None:
                raise ValueError(f"task {task.task_id} has no reference solution")
            task.solve(workspace)
            final = task.reference_answer or "Done — I completed the requested task."
            return Transcript(
                steps=(
                    TranscriptStep(kind="tool", content=f"edit workspace for {task.task_id}"),
                    TranscriptStep(kind="message", content="Applied the change."),
                ),
                final_message=final,
            )
        if outcome == "flaw" and (task.solve_flawed is not None or task.reference_answer_flawed):
            if task.solve_flawed is not None:
                task.solve_flawed(workspace)
            final = task.reference_answer_flawed or "Done — I completed the requested task."
            return Transcript(
                steps=(
                    TranscriptStep(kind="tool", content=f"edit workspace for {task.task_id}"),
                    TranscriptStep(kind="message", content="Applied the change."),
                ),
                final_message=final,
            )
        return Transcript(
            steps=(TranscriptStep(kind="message", content="Working on it."),),
            final_message="Done — the task is complete.",
        )

    return executor


def hard_scripted_executor(*, flaw_period: int = 3, fake_period: int = 5) -> CalibrationExecutor:
    """Three-outcome scripted executor: mostly correct, with rotating flawed and fake runs.

    The flawed runs are the reason this exists — they produce gold=0 cases whose workspace
    *looks* solved, so a judge that merely detects activity can no longer score α=1.0 (which
    the two-outcome corpus lets strong judges do). ``flaw_period`` and ``fake_period`` should
    be coprime with each other and with the 6 task families so both outcomes rotate across
    every family. Stateful; create a fresh instance per run.
    """
    counter = {"i": -1}

    def outcome_of(_task: VerifiableTask) -> str:
        counter["i"] += 1
        i = counter["i"]
        if i % fake_period == fake_period - 1:
            return "fake"
        if i % flaw_period == flaw_period - 1:
            return "flaw"
        return "solve"

    return make_outcome_executor(outcome_of)
