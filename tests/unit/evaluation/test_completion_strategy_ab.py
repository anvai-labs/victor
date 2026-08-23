# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""EVR-3 Prong-B completion-strategy A/B tests (no live model)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from victor.evaluation.agentic_harness import AgenticExecutionTrace, EvalToolCall
from victor.evaluation.completion_strategy_ab import (
    JUDGE_ENV,
    STRATEGY_ENV,
    CompletionABGate,
    CompletionABVerdict,
    CompletionArmResult,
    CompletionTaskOutcome,
    assess_completion_ab,
    run_completion_ab,
)


def _outcome(
    task_id: str,
    *,
    strategy_passes: bool = True,
    claimed: bool = True,
    latency: int = 4,
    evaluable: bool = True,
) -> CompletionTaskOutcome:
    return CompletionTaskOutcome(
        task_id=task_id,
        family=f"family-{int(task_id[1:]) % 6}",
        verified=strategy_passes,
        evaluable=evaluable,
        claimed_complete=claimed,
        loop_iterations=latency,
        outer_turns=1,
        tool_calls=2,
        duration_seconds=1.0,
        error="" if evaluable else "provider unavailable",
    )


def _arm(strategy: str, outcomes: list[CompletionTaskOutcome]) -> CompletionArmResult:
    return CompletionArmResult(strategy, strategy, tuple(outcomes))


def _gate(**overrides) -> CompletionABGate:
    values = {
        "min_pairs": 24,
        "min_families": 6,
        "min_pairs_per_family": 4,
        "success_regression_tolerance": 0.0,
        "max_false_positive_increase": 0,
        "max_latency_ratio": 1.10,
        "max_latency_absolute": 0.25,
    }
    values.update(overrides)
    return CompletionABGate(**values)


def test_passes_on_paired_match_without_latency_or_false_positive_regression() -> None:
    baseline = _arm("enhanced", [_outcome(f"t{i}", latency=5) for i in range(24)])
    candidate = _arm("rubric", [_outcome(f"t{i}", latency=4) for i in range(24)])

    report = assess_completion_ab(baseline, candidate)

    assert report.verdict is CompletionABVerdict.PASS
    assert report.prong_b_passed
    assert report.paired["n_evaluable"] == 24
    assert report.paired["task_success_delta"] == 0.0
    assert report.paired["mcnemar_p"] == 1.0


def test_holds_when_task_success_regresses() -> None:
    baseline = _arm("enhanced", [_outcome(f"t{i}") for i in range(24)])
    candidate = _arm(
        "rubric",
        [_outcome(f"t{i}", strategy_passes=i != 0) for i in range(24)],
    )

    report = assess_completion_ab(baseline, candidate)

    assert report.verdict is CompletionABVerdict.HOLD
    assert any("task success regressed" in reason for reason in report.reasons)
    assert report.paired["baseline_only_pass"] == 1


def test_holds_when_false_positive_completions_increase() -> None:
    baseline_outcomes = [
        _outcome(f"t{i}", strategy_passes=i != 0, claimed=False) for i in range(24)
    ]
    candidate_outcomes = [_outcome(f"t{i}", strategy_passes=i != 0) for i in range(24)]

    report = assess_completion_ab(
        _arm("enhanced", baseline_outcomes),
        _arm("rubric", candidate_outcomes),
    )

    assert report.paired["task_success_delta"] == 0.0
    assert report.paired["false_positive_delta"] == 1
    assert any("false-positive completions increased" in reason for reason in report.reasons)


def test_holds_on_completion_latency_blowout() -> None:
    baseline = _arm("enhanced", [_outcome(f"t{i}", latency=4) for i in range(24)])
    candidate = _arm("rubric", [_outcome(f"t{i}", latency=5) for i in range(24)])

    report = assess_completion_ab(baseline, candidate)

    assert any("completion latency exceeded budget" in reason for reason in report.reasons)


def test_holds_when_candidate_fell_back_to_enhanced() -> None:
    outcomes = [_outcome(f"t{i}") for i in range(24)]
    baseline = _arm("enhanced", outcomes)
    candidate = CompletionArmResult("rubric", "enhanced", tuple(outcomes))

    report = assess_completion_ab(baseline, candidate)

    assert any("judge pin/fallback" in reason for reason in report.reasons)


def test_holds_on_invalid_or_thin_evidence() -> None:
    baseline = _arm("enhanced", [_outcome(f"t{i}") for i in range(5)])
    candidate = _arm(
        "rubric",
        [_outcome(f"t{i}", evaluable=i != 0) for i in range(5)],
    )

    report = assess_completion_ab(baseline, candidate)

    assert any("not evaluable" in reason for reason in report.reasons)
    assert any("insufficient paired tasks" in reason for reason in report.reasons)


class _Verifiable:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.family = "family"

    def setup(self, workspace: Path) -> None:
        pass

    def verify(self, workspace: Path, transcript) -> float:
        return 1.0


class _FakeAdapter:
    def __init__(self, captured: list[tuple[str | None, str | None, str]]) -> None:
        self._captured = captured

    async def _execute(self, bench, workspace: Path) -> AgenticExecutionTrace:
        strategy = os.environ.get(STRATEGY_ENV)
        self._captured.append((strategy, os.environ.get(JUDGE_ENV), os.getcwd()))
        return AgenticExecutionTrace(
            task_id=bench.task_id,
            start_time=0.0,
            end_time=1.0,
            turns=1,
            messages=[{"role": "assistant", "content": "done"}],
            tool_calls=[EvalToolCall(name="write", arguments={}, success=True)],
            completion_signals={
                "agentic_loop_success": True,
                "agentic_loop_iterations": 3 if strategy == "enhanced" else 2,
            },
        )

    def execute_task(self, bench, workspace: Path):
        return self._execute(bench, workspace)


class _Runner:
    def run(self, coroutine):
        return asyncio.run(coroutine)


def test_live_orchestration_pairs_tasks_restores_env_and_writes_report(tmp_path: Path) -> None:
    previous_strategy = os.environ.get(STRATEGY_ENV)
    previous_judge = os.environ.get(JUDGE_ENV)
    captured: list[tuple[str | None, str | None, str]] = []
    tasks = [(_Verifiable(f"t{i}"), SimpleNamespace(task_id=f"t{i}")) for i in range(4)]

    result = run_completion_ab(
        judge="session-model",
        model="llama3.3:70b",
        variants=1,
        out_dir=str(tmp_path),
        gate=_gate(min_pairs=4, min_families=1, min_pairs_per_family=4),
        adapter_factory=lambda **kwargs: _FakeAdapter(captured),
        task_provider=lambda variants: tasks,
        runner=_Runner(),
    )

    assert result["verdict"] == "pass"
    assert result["paired"]["mean_loop_iterations_delta"] == -1
    assert [row[0] for row in captured] == ["enhanced"] * 4 + ["rubric"] * 4
    assert all(row[1] == "session-model" for row in captured)
    assert os.environ.get(STRATEGY_ENV) == previous_strategy
    assert os.environ.get(JUDGE_ENV) == previous_judge
    payload = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["prong_b_passed"] is True


def test_strategy_environment_restores_values_after_task_error(tmp_path: Path) -> None:
    class _BrokenAdapter:
        def execute_task(self, bench, workspace):
            raise RuntimeError("boom")

    os.environ[STRATEGY_ENV] = "legacy"
    try:
        result = run_completion_ab(
            model="llama3.3:70b",
            variants=1,
            out_dir=str(tmp_path),
            gate=_gate(min_pairs=1, min_families=1, min_pairs_per_family=1),
            adapter_factory=lambda **kwargs: _BrokenAdapter(),
            task_provider=lambda variants: [(_Verifiable("t0"), SimpleNamespace(task_id="t0"))],
            runner=_Runner(),
        )
        assert result["verdict"] == "hold"
        assert "boom" in result["baseline"]["outcomes"][0]["error"]
        assert os.environ[STRATEGY_ENV] == "legacy"
    finally:
        os.environ.pop(STRATEGY_ENV, None)


@pytest.mark.parametrize(
    "candidate_only,baseline_only,expected",
    [(0, 0, 1.0), (8, 1, 0.0390625), (10, 0, 0.001953125)],
)
def test_report_uses_exact_paired_mcnemar(
    candidate_only: int, baseline_only: int, expected: float
) -> None:
    total = candidate_only + baseline_only
    baseline_outcomes = []
    candidate_outcomes = []
    for i in range(total):
        candidate_wins = i < candidate_only
        baseline_outcomes.append(_outcome(f"t{i}", strategy_passes=not candidate_wins))
        candidate_outcomes.append(_outcome(f"t{i}", strategy_passes=candidate_wins))
    report = assess_completion_ab(
        _arm("enhanced", baseline_outcomes),
        _arm("rubric", candidate_outcomes),
        gate=_gate(min_pairs=0, min_pairs_per_family=0, success_regression_tolerance=1.0),
    )
    assert report.paired["mcnemar_p"] == pytest.approx(expected)
