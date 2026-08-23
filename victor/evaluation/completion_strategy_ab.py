# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Verifier-backed completion-strategy A/B battery (EVR-3 Prong B).

Runs the same verifiable tasks through ``enhanced`` and ``rubric`` completion,
pairs outcomes by task id, and emits EVR-3 Prong-B evidence: task success,
false-positive completion, and loop latency. This is one prerequisite for an
ADR-009 default decision, not a substitute for the judge-reliability gate.
The runner never changes the configured default; it only sets per-arm process
overrides while constructing and executing the real agent.

Example::

    python -m victor.evaluation.completion_strategy_ab \
        --model qwen3-coder-tools:30b \
        --judge llm:llama3.3:70b@http://localhost:11434 \
        --base-url http://localhost:11434 --variants 4 --out-dir artifacts/evr3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

STRATEGY_ENV = "VICTOR_COMPLETION_STRATEGY"
JUDGE_ENV = "VICTOR_COMPLETION_JUDGE"


class CompletionABVerdict(str, Enum):
    """Whether the rubric strategy clears EVR-3's end-to-end prong."""

    PASS = "pass"
    HOLD = "hold"


@dataclass(frozen=True)
class CompletionTaskOutcome:
    """Verifier truth and completion telemetry for one task in one arm."""

    task_id: str
    family: str
    verified: bool
    evaluable: bool
    claimed_complete: Optional[bool]
    loop_iterations: int
    outer_turns: int
    tool_calls: int
    duration_seconds: float
    error: str = ""

    @property
    def false_positive(self) -> bool:
        return self.evaluable and self.claimed_complete is True and not self.verified

    @property
    def latency_turns(self) -> int:
        """Prefer the strategy-sensitive inner-loop count, with outer turns as fallback."""
        return self.loop_iterations if self.loop_iterations > 0 else self.outer_turns

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "verified": self.verified,
            "evaluable": self.evaluable,
            "claimed_complete": self.claimed_complete,
            "false_positive": self.false_positive,
            "loop_iterations": self.loop_iterations,
            "outer_turns": self.outer_turns,
            "tool_calls": self.tool_calls,
            "duration_seconds": round(self.duration_seconds, 4),
            "error": self.error,
        }


@dataclass(frozen=True)
class CompletionArmResult:
    """All outcomes for one requested/effective completion strategy."""

    requested_strategy: str
    effective_strategy: str
    outcomes: tuple[CompletionTaskOutcome, ...] = field(default_factory=tuple)

    @property
    def evaluable_outcomes(self) -> tuple[CompletionTaskOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.evaluable)

    def summary(self) -> dict[str, Any]:
        outcomes = self.evaluable_outcomes
        n = len(outcomes)
        latencies = [outcome.latency_turns for outcome in outcomes]
        return {
            "requested_strategy": self.requested_strategy,
            "effective_strategy": self.effective_strategy,
            "n_total": len(self.outcomes),
            "n_evaluable": n,
            "task_success_rate": sum(outcome.verified for outcome in outcomes) / n if n else 0.0,
            "completion_claim_rate": (
                sum(outcome.claimed_complete is True for outcome in outcomes) / n if n else 0.0
            ),
            "false_positive_completions": sum(outcome.false_positive for outcome in outcomes),
            "false_positive_rate": (
                sum(outcome.false_positive for outcome in outcomes) / n if n else 0.0
            ),
            "mean_loop_iterations": statistics.mean(latencies) if latencies else 0.0,
            "mean_outer_turns": (
                statistics.mean(outcome.outer_turns for outcome in outcomes) if outcomes else 0.0
            ),
            "mean_tool_calls": (
                statistics.mean(outcome.tool_calls for outcome in outcomes) if outcomes else 0.0
            ),
            "mean_duration_seconds": (
                statistics.mean(outcome.duration_seconds for outcome in outcomes)
                if outcomes
                else 0.0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


@dataclass(frozen=True)
class CompletionABGate:
    """Explicit non-regression and evidence thresholds for Prong B."""

    min_pairs: int = 24
    min_families: int = 6
    min_pairs_per_family: int = 4
    success_regression_tolerance: float = 0.0
    max_false_positive_increase: int = 0
    max_latency_ratio: float = 1.10
    max_latency_absolute: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_pairs": self.min_pairs,
            "min_families": self.min_families,
            "min_pairs_per_family": self.min_pairs_per_family,
            "success_regression_tolerance": self.success_regression_tolerance,
            "max_false_positive_increase": self.max_false_positive_increase,
            "max_latency_ratio": self.max_latency_ratio,
            "max_latency_absolute": self.max_latency_absolute,
        }


@dataclass(frozen=True)
class CompletionABReport:
    """Serializable paired graduation decision."""

    verdict: CompletionABVerdict
    baseline: CompletionArmResult
    candidate: CompletionArmResult
    gate: CompletionABGate
    paired: dict[str, Any]
    reasons: tuple[str, ...]

    @property
    def prong_b_passed(self) -> bool:
        """Prong-B pass only; judge reliability remains an independent prerequisite."""
        return self.verdict is CompletionABVerdict.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "prong_b_passed": self.prong_b_passed,
            "reasons": list(self.reasons),
            "gate": self.gate.to_dict(),
            "paired": self.paired,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
        }


def _mcnemar_p(candidate_only: int, baseline_only: int) -> float:
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    extreme = max(candidate_only, baseline_only)
    tail = sum(math.comb(discordant, i) for i in range(extreme, discordant + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def assess_completion_ab(
    baseline: CompletionArmResult,
    candidate: CompletionArmResult,
    *,
    gate: Optional[CompletionABGate] = None,
) -> CompletionABReport:
    """Apply the EVR-3 Prong-B gate to two task-id-paired arms."""
    resolved_gate = gate or CompletionABGate()
    base_by_id = {outcome.task_id: outcome for outcome in baseline.outcomes}
    candidate_by_id = {outcome.task_id: outcome for outcome in candidate.outcomes}
    shared_ids = sorted(set(base_by_id) & set(candidate_by_id))
    valid_pairs = [
        (base_by_id[task_id], candidate_by_id[task_id])
        for task_id in shared_ids
        if base_by_id[task_id].evaluable and candidate_by_id[task_id].evaluable
    ]
    invalid_ids = [
        task_id
        for task_id in shared_ids
        if not (base_by_id[task_id].evaluable and candidate_by_id[task_id].evaluable)
    ]
    missing_completion_telemetry = [
        base.task_id
        for base, cand in valid_pairs
        if base.claimed_complete is None or cand.claimed_complete is None
    ]
    missing_latency_telemetry = [
        base.task_id
        for base, cand in valid_pairs
        if base.latency_turns <= 0 or cand.latency_turns <= 0
    ]

    both_pass = sum(base.verified and cand.verified for base, cand in valid_pairs)
    both_fail = sum(not base.verified and not cand.verified for base, cand in valid_pairs)
    candidate_only = sum(not base.verified and cand.verified for base, cand in valid_pairs)
    baseline_only = sum(base.verified and not cand.verified for base, cand in valid_pairs)
    n = len(valid_pairs)
    baseline_success = (both_pass + baseline_only) / n if n else 0.0
    candidate_success = (both_pass + candidate_only) / n if n else 0.0
    baseline_fp = sum(base.false_positive for base, _ in valid_pairs)
    candidate_fp = sum(cand.false_positive for _, cand in valid_pairs)
    baseline_latency = (
        statistics.mean(base.latency_turns for base, _ in valid_pairs) if valid_pairs else 0.0
    )
    candidate_latency = (
        statistics.mean(cand.latency_turns for _, cand in valid_pairs) if valid_pairs else 0.0
    )
    family_counts: dict[str, int] = {}
    for base, cand in valid_pairs:
        family = cand.family or base.family or "unknown"
        family_counts[family] = family_counts.get(family, 0) + 1

    paired = {
        "n_shared": len(shared_ids),
        "n_evaluable": n,
        "invalid_task_ids": invalid_ids,
        "missing_completion_telemetry_task_ids": missing_completion_telemetry,
        "missing_latency_telemetry_task_ids": missing_latency_telemetry,
        "family_counts": dict(sorted(family_counts.items())),
        "both_pass": both_pass,
        "both_fail": both_fail,
        "candidate_only_pass": candidate_only,
        "baseline_only_pass": baseline_only,
        "discordant": candidate_only + baseline_only,
        "task_success_baseline": baseline_success,
        "task_success_candidate": candidate_success,
        "task_success_delta": candidate_success - baseline_success,
        "mcnemar_p": _mcnemar_p(candidate_only, baseline_only),
        "false_positive_baseline": baseline_fp,
        "false_positive_candidate": candidate_fp,
        "false_positive_delta": candidate_fp - baseline_fp,
        "mean_loop_iterations_baseline": baseline_latency,
        "mean_loop_iterations_candidate": candidate_latency,
        "mean_loop_iterations_delta": candidate_latency - baseline_latency,
    }

    reasons: list[str] = []
    if baseline.effective_strategy != "enhanced":
        reasons.append(
            f"baseline effective strategy was {baseline.effective_strategy!r}, not 'enhanced'"
        )
    if candidate.effective_strategy != "rubric":
        reasons.append(
            f"candidate effective strategy was {candidate.effective_strategy!r}, not 'rubric' "
            "(judge pin/fallback prevented a rubric measurement)"
        )
    if set(base_by_id) != set(candidate_by_id):
        reasons.append("arms did not contain the identical task-id set")
    if len(base_by_id) != len(baseline.outcomes) or len(candidate_by_id) != len(candidate.outcomes):
        reasons.append("an arm contained duplicate task ids")
    if invalid_ids:
        reasons.append(
            f"{len(invalid_ids)} paired task(s) were not evaluable: {', '.join(invalid_ids[:8])}"
        )
    if n < resolved_gate.min_pairs:
        reasons.append(f"insufficient paired tasks: n={n}, require {resolved_gate.min_pairs}")
    if len(family_counts) < resolved_gate.min_families:
        reasons.append(
            f"insufficient family coverage: {len(family_counts)}, "
            f"require {resolved_gate.min_families}"
        )
    thin_families = {
        family: count
        for family, count in family_counts.items()
        if count < resolved_gate.min_pairs_per_family
    }
    if thin_families:
        reasons.append(
            "insufficient per-family evidence: "
            + ", ".join(f"{family}={count}" for family, count in sorted(thin_families.items()))
        )
    if missing_completion_telemetry:
        reasons.append(
            f"missing completion-claim telemetry for {len(missing_completion_telemetry)} pair(s)"
        )
    if missing_latency_telemetry:
        reasons.append(
            f"missing completion-latency telemetry for {len(missing_latency_telemetry)} pair(s)"
        )
    if candidate_success + resolved_gate.success_regression_tolerance < baseline_success:
        reasons.append(
            "task success regressed "
            f"{baseline_success:.3f} -> {candidate_success:.3f} "
            f"(tolerance {resolved_gate.success_regression_tolerance:.3f})"
        )
    if candidate_fp - baseline_fp > resolved_gate.max_false_positive_increase:
        reasons.append(
            f"false-positive completions increased {baseline_fp} -> {candidate_fp} "
            f"(allowed increase {resolved_gate.max_false_positive_increase})"
        )
    if valid_pairs:
        latency_limit = max(
            baseline_latency * resolved_gate.max_latency_ratio,
            baseline_latency + resolved_gate.max_latency_absolute,
        )
        if candidate_latency > latency_limit:
            reasons.append(
                "completion latency exceeded budget: "
                f"{baseline_latency:.3f} -> {candidate_latency:.3f} "
                f"(limit {latency_limit:.3f})"
            )

    verdict = CompletionABVerdict.HOLD if reasons else CompletionABVerdict.PASS
    return CompletionABReport(verdict, baseline, candidate, resolved_gate, paired, tuple(reasons))


@contextmanager
def _strategy_environment(strategy: str, judge: Optional[str]) -> Iterator[None]:
    previous_strategy = os.environ.get(STRATEGY_ENV)
    previous_judge = os.environ.get(JUDGE_ENV)
    os.environ[STRATEGY_ENV] = strategy
    if judge:
        os.environ[JUDGE_ENV] = judge
    else:
        os.environ.pop(JUDGE_ENV, None)
    try:
        yield
    finally:
        if previous_strategy is None:
            os.environ.pop(STRATEGY_ENV, None)
        else:
            os.environ[STRATEGY_ENV] = previous_strategy
        if previous_judge is None:
            os.environ.pop(JUDGE_ENV, None)
        else:
            os.environ[JUDGE_ENV] = previous_judge


def _adapter_session_model(adapter: Any, requested_model: Optional[str]) -> Optional[str]:
    if requested_model:
        return requested_model
    orchestrator = getattr(adapter, "orchestrator", None)
    turn_executor = getattr(orchestrator, "turn_executor", None)
    provider_context = getattr(turn_executor, "_provider_context", None)
    model = getattr(provider_context, "model", None)
    return str(model) if model else None


def _effective_strategy(strategy: str, session_model: Optional[str]) -> str:
    from victor.agent.services.judge_calibration_gate import resolve_completion_strategy
    from victor.config.settings import load_settings

    return resolve_completion_strategy(load_settings(), session_model)


def _outcome_from_trace(verifiable: Any, workspace: Path, trace: Any) -> CompletionTaskOutcome:
    from victor.evaluation.flag_ab import _verify_task

    task_id = str(getattr(verifiable, "task_id", None) or getattr(trace, "task_id", "unknown"))
    family = str(getattr(verifiable, "family", "unknown") or "unknown")
    signals = dict(getattr(trace, "completion_signals", None) or {})
    messages = list(getattr(trace, "messages", None) or [])
    has_assistant_response = any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and (message.get("content") or message.get("reasoning"))
        for message in messages
    )
    validation_errors = dict(getattr(trace, "validation_errors", None) or {})
    error = str(validation_errors.get("execution", "") or "")
    evaluable = has_assistant_response and not error
    claimed = signals.get("agentic_loop_success")
    if claimed is None:
        claimed = signals.get("outer_completion_claimed")
    return CompletionTaskOutcome(
        task_id=task_id,
        family=family,
        verified=bool(_verify_task(verifiable, workspace, trace)) if evaluable else False,
        evaluable=evaluable,
        claimed_complete=bool(claimed) if claimed is not None else None,
        loop_iterations=int(signals.get("agentic_loop_iterations", 0) or 0),
        outer_turns=int(getattr(trace, "turns", 0) or 0),
        tool_calls=len(getattr(trace, "tool_calls", None) or []),
        duration_seconds=float(getattr(trace, "duration_seconds", 0.0) or 0.0),
        error=error or ("no assistant response" if not has_assistant_response else ""),
    )


def run_completion_arm(
    strategy: str,
    tasks: Sequence[Tuple[Any, Any]],
    *,
    judge: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    max_turns: int = 12,
    adapter_factory: Optional[Callable[..., Any]] = None,
    runner: Optional[Any] = None,
) -> CompletionArmResult:
    """Execute one strategy arm over a fixed task list."""
    from victor.evaluation.flag_ab import (
        _default_adapter_factory,
        _default_runner,
        _drain_and_close,
    )

    run = runner if runner is not None else _default_runner()
    owns_runner = runner is None
    outcomes: list[CompletionTaskOutcome] = []
    arm_cwd = os.getcwd()
    try:
        with _strategy_environment(strategy, judge):
            adapter = (adapter_factory or _default_adapter_factory)(
                base_url=base_url, model=model, max_turns=max_turns
            )
            effective = _effective_strategy(strategy, _adapter_session_model(adapter, model))
            for verifiable, bench in tasks:
                with tempfile.TemporaryDirectory(prefix="victor-completion-ab-") as tmp:
                    workspace = Path(tmp)
                    setup = getattr(verifiable, "setup", None)
                    try:
                        if callable(setup):
                            setup(workspace)
                        trace = run.run(adapter.execute_task(bench, workspace))
                        outcomes.append(_outcome_from_trace(verifiable, workspace, trace))
                    except Exception as exc:  # one broken task invalidates evidence, not the run
                        logger.exception(
                            "completion A/B task %s failed", getattr(bench, "task_id", "?")
                        )
                        outcomes.append(
                            CompletionTaskOutcome(
                                task_id=str(getattr(bench, "task_id", "unknown")),
                                family=str(getattr(verifiable, "family", "unknown")),
                                verified=False,
                                evaluable=False,
                                claimed_complete=None,
                                loop_iterations=0,
                                outer_turns=0,
                                tool_calls=0,
                                duration_seconds=0.0,
                                error=str(exc),
                            )
                        )
                    finally:
                        os.chdir(arm_cwd)
            return CompletionArmResult(strategy, effective, tuple(outcomes))
    finally:
        if owns_runner:
            _drain_and_close(run)


def run_completion_ab(
    *,
    baseline_strategy: str = "enhanced",
    candidate_strategy: str = "rubric",
    judge: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    variants: int = 4,
    max_turns: int = 12,
    corpus: str = "calibration",
    out_dir: str = ".",
    workdir: Optional[str] = None,
    gate: Optional[CompletionABGate] = None,
    adapter_factory: Optional[Callable[..., Any]] = None,
    task_provider: Optional[Callable[[int], Sequence[Tuple[Any, Any]]]] = None,
    runner: Optional[Any] = None,
) -> dict[str, Any]:
    """Run both paired arms in an isolated cwd and write one evidence report."""
    from victor.evaluation.flag_ab import _corpus_task_provider

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()
    scratch: Optional[tempfile.TemporaryDirectory[str]] = None
    if workdir:
        run_cwd = Path(workdir).resolve()
        run_cwd.mkdir(parents=True, exist_ok=True)
    else:
        scratch = tempfile.TemporaryDirectory(prefix="victor-completion-ab-cwd-")
        run_cwd = Path(scratch.name)

    try:
        os.chdir(run_cwd)
        tasks = tuple((task_provider or _corpus_task_provider(corpus))(variants))
        arm_kwargs = {
            "judge": judge,
            "model": model,
            "base_url": base_url,
            "max_turns": max_turns,
            "adapter_factory": adapter_factory,
            "runner": runner,
        }
        baseline = run_completion_arm(baseline_strategy, tasks, **arm_kwargs)
        candidate = run_completion_arm(candidate_strategy, tasks, **arm_kwargs)
        report = assess_completion_ab(baseline, candidate, gate=gate)
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "judge": judge or "session-model",
            "corpus": corpus,
            "variants": variants,
            "max_turns": max_turns,
            **report.to_dict(),
        }
        report_path = out / "completion_strategy_ab.json"
        report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return {"report_path": str(report_path), **payload}
    finally:
        os.chdir(original_cwd)
        if scratch is not None:
            scratch.cleanup()


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="enhanced")
    parser.add_argument("--candidate", default="rubric")
    parser.add_argument("--judge", default=None, help="Completion judge backend")
    parser.add_argument("--model", default=None, help="Session model")
    parser.add_argument("--base-url", default=None, help="Session provider base URL")
    parser.add_argument("--variants", type=int, default=4, help="Variants per task family")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--corpus", default="calibration", choices=("calibration", "effect-gate"))
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--min-pairs", type=int, default=24)
    parser.add_argument("--min-families", type=int, default=6)
    parser.add_argument("--min-pairs-per-family", type=int, default=4)
    parser.add_argument("--success-regression-tolerance", type=float, default=0.0)
    parser.add_argument("--max-false-positive-increase", type=int, default=0)
    parser.add_argument("--max-latency-ratio", type=float, default=1.10)
    args = parser.parse_args(argv)
    gate = CompletionABGate(
        min_pairs=args.min_pairs,
        min_families=args.min_families,
        min_pairs_per_family=args.min_pairs_per_family,
        success_regression_tolerance=args.success_regression_tolerance,
        max_false_positive_increase=args.max_false_positive_increase,
        max_latency_ratio=args.max_latency_ratio,
    )
    result = run_completion_ab(
        baseline_strategy=args.baseline,
        candidate_strategy=args.candidate,
        judge=args.judge,
        model=args.model,
        base_url=args.base_url,
        variants=args.variants,
        max_turns=args.max_turns,
        corpus=args.corpus,
        out_dir=args.out_dir,
        workdir=args.workdir,
        gate=gate,
    )
    print(
        json.dumps(
            {
                "report_path": result["report_path"],
                "verdict": result["verdict"],
                "prong_b_passed": result["prong_b_passed"],
                "reasons": result["reasons"],
                "paired": result["paired"],
            },
            indent=2,
        )
    )
    return 0 if result["prong_b_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
