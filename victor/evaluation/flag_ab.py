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

"""Flag A/B battery-snapshot generator (flag-graduation policy).

Produces the two :class:`BatteryResult` snapshots the flag-graduation decision needs: it runs a
task battery through the **real agent** twice — an opt-in flag OFF (baseline) then ON (candidate) —
captures an ``AgenticExecutionTrace`` per task, scores with the trajectory evaluator, and writes
each arm's ``BatteryResult.to_dict()`` to JSON. When the graduation decider is importable it also
prints the GRADUATE/HOLD verdict directly.

The flag is toggled per arm via its env var (read at agent construction by the resolvers) — the
baseline arm sets it off, the candidate arm on. Runs against any provider the agent supports;
point ``--base-url`` at a local Ollama (e.g. ``http://172.31.160.1:11434``) for a token-free A/B.

This is a *runner* (it drives a live model), so heavy imports are lazy and the orchestration is
dependency-injected — the unit tests exercise the toggle/score/emit logic with fakes, no model.

    python -m victor.evaluation.flag_ab --flag effect_gated_completion \\
        --base-url http://172.31.160.1:11434 --model qwen3-coder-tools:30b --variants 2
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from victor.evaluation.trajectory_eval import BatteryResult

logger = logging.getLogger(__name__)

#: Opt-in flag → the env var its resolver reads (mirrors resolve_*_enabled in framework).
FLAG_ENV = {
    "effect_gated_completion": "VICTOR_EFFECT_GATED_COMPLETION",
    "per_turn_auditor": "VICTOR_PER_TURN_AUDITOR",
}


def _flag_env_var(flag: str) -> str:
    return FLAG_ENV.get(flag, f"VICTOR_{flag.upper()}")


#: Named task corpora. "calibration" = the general battery; "effect-gate" = tasks that tempt
#: premature completion (a fair test for the effect gate).
CORPORA = ("calibration", "effect-gate")


def _corpus_loader(corpus: str) -> Any:
    if corpus == "effect-gate":
        from victor.evaluation.effect_gate_corpus import effect_gate_corpus

        return effect_gate_corpus
    from victor.evaluation.calibration_corpus import default_corpus

    return default_corpus


def _corpus_task_provider(corpus: str) -> Callable[[int], list[Tuple[Any, Any]]]:
    def provider(variants: int) -> list[Tuple[Any, Any]]:
        from victor.evaluation.calibration_agent_executor import verifiable_to_benchmark_task

        loader = _corpus_loader(corpus)
        return [(t, verifiable_to_benchmark_task(t)) for t in loader(variants=variants)]

    return provider


def _default_task_provider(variants: int) -> list[Tuple[Any, Any]]:
    return _corpus_task_provider("calibration")(variants)


def _verify_task(verifiable: Any, workspace: Path, trace: Any) -> float:
    """Run a task's programmatic verifier on the final workspace → 1.0 pass / 0.0 fail."""
    verify = getattr(verifiable, "verify", None)
    if not callable(verify):
        return 0.0
    try:
        from victor.evaluation.calibration_agent_executor import trace_to_transcript

        transcript = trace_to_transcript(trace)
    except Exception:  # noqa: BLE001 — transcript is optional; most verifiers use the workspace
        transcript = None
    try:
        return 1.0 if float(verify(workspace, transcript)) >= 1.0 else 0.0
    except Exception:  # noqa: BLE001 — a verifier error scores the task as failed, not the arm
        logger.debug("verify failed for %s", getattr(verifiable, "task_id", "?"), exc_info=True)
        return 0.0


def _success_battery(successes: Sequence[float]) -> Any:
    """Aggregate per-task pass/fail into a BatteryResult whose overall mean is the task pass rate."""
    from victor.evaluation.trajectory_eval import (
        BatteryResult,
        IntervalStat,
        mean_confidence_interval,
    )

    vals = [float(s) for s in successes]
    if not vals:
        return BatteryResult(scores=(), per_dimension=(), overall=None)
    mean, lo, hi = mean_confidence_interval(vals)
    return BatteryResult(scores=(), per_dimension=(), overall=IntervalStat(mean, lo, hi, len(vals)))


def _default_adapter_factory(
    *, base_url: Optional[str], model: Optional[str], max_turns: int = 12
) -> Any:
    from victor.evaluation.agent_adapter import AdapterConfig, VictorAgentAdapter

    # Bound per-task cost (the stock 20-turn / 20-min budget is too loose for a battery), but the
    # TOOL budget must stay generous and decoupled from max_turns: a real task reads a few files,
    # edits, then verifies — ~10-15 tool calls — and an enabled effect gate forces extra work by
    # design. A tight tool budget starves it (COMPLETE→RETRY exhausts the budget → the task FAILs
    # with no effect), which silently poisons an effect-gate A/B (observed: gate-ON → 0% pass at
    # tool_budget=8). Keep it well clear of that.
    return VictorAgentAdapter.from_profile(
        profile="default",
        base_url=base_url,
        model_override=model,
        config=AdapterConfig(
            max_turns=max_turns,
            tool_budget=max(30, max_turns * 3),
            total_timeout=600,
            min_turn_timeout=90,
        ),
    )


def _default_runner() -> Any:
    from victor.evaluation.judge_calibration_harness import PersistentLoopRunner

    return PersistentLoopRunner()


async def _stop_background_services() -> None:
    """Reap the agent's leaked background tasks on the current loop.

    Each eval task starts a per-workspace file watcher (graph indexing) and event-bus loop that
    nothing shuts down (VictorAgentAdapter has no close()). Left pending, they log "Task was
    destroyed but it is pending" at teardown and accumulate across the battery. Stop the watchers
    via their registry, then cancel any remaining pending tasks. Best-effort — never raises.
    """
    try:
        from victor.core.indexing.file_watcher import FileWatcherRegistry

        await FileWatcherRegistry.get_instance().stop_all()
    except Exception:  # noqa: BLE001 — cleanup must never break the run
        logger.debug("file-watcher stop_all failed", exc_info=True)

    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _drain_and_close(run: Any) -> None:
    """Drain leaked background tasks on ``run``'s loop, then close it (for a runner we own)."""
    try:
        run.run(_stop_background_services())
    except Exception:  # noqa: BLE001
        logger.debug("background drain failed", exc_info=True)
    close = getattr(run, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            logger.debug("runner close failed", exc_info=True)


def _default_evaluator() -> Any:
    # Adopt EffectGroundingScorer so an effect-gate A/B actually measures effect grounding
    # (matches the acceptance-oracle promotion gate).
    from victor.evaluation.trajectory_eval import (
        EffectGroundingScorer,
        TrajectoryEvaluator,
        default_scorers,
    )

    return TrajectoryEvaluator(tuple(default_scorers()) + (EffectGroundingScorer(),))


def run_battery_arm(
    flag: str,
    enabled: bool,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    variants: int = 2,
    max_turns: int = 12,
    corpus: str = "calibration",
    score: str = "trajectory",
    adapter_factory: Optional[Callable[..., Any]] = None,
    task_provider: Optional[Callable[[int], Sequence[Tuple[Any, Any]]]] = None,
    runner: Optional[Any] = None,
    evaluator: Optional[Any] = None,
) -> "BatteryResult":
    """Run one A/B arm: toggle ``flag`` to ``enabled``, run the battery, score → ``BatteryResult``.

    The flag's env var is set for the whole arm (the resolvers read it at agent construction) and
    restored afterward. Each task runs in a fresh temp workspace with its fixture set up (when the
    task provides one). ``max_turns`` bounds per-task cost.

    ``corpus`` picks the task set ("calibration" | "effect-gate"). ``score`` picks the metric:
    "trajectory" (dimension scores from the trace) or "verify" (run each task's programmatic
    verifier on the final workspace → overall = task pass rate — the fairest signal for the effect
    gate). Collaborators are injectable for testing.
    """
    env_var = _flag_env_var(flag)
    previous = os.environ.get(env_var)
    os.environ[env_var] = "true" if enabled else "false"
    run = runner if runner is not None else _default_runner()
    owns_runner = runner is None
    try:
        adapter = (adapter_factory or _default_adapter_factory)(
            base_url=base_url, model=model, max_turns=max_turns
        )
        provider = task_provider or _corpus_task_provider(corpus)
        tasks = provider(variants)
        traces: list = []
        successes: list = []
        for verifiable, bench in tasks:
            with tempfile.TemporaryDirectory(prefix="victor-flagab-") as tmp:
                workspace = Path(tmp)
                setup = getattr(verifiable, "setup", None)
                if callable(setup):
                    try:
                        setup(workspace)
                    except Exception:  # noqa: BLE001 — a fixture failure must not kill the arm
                        logger.debug(
                            "setup failed for %s", getattr(bench, "task_id", "?"), exc_info=True
                        )
                trace = run.run(adapter.execute_task(bench, workspace))
                if score == "verify":
                    # Verify on the final workspace *before* the temp dir is torn down.
                    successes.append(_verify_task(verifiable, workspace, trace))
                else:
                    traces.append(trace)
        if score == "verify":
            return _success_battery(successes)
        return (evaluator or _default_evaluator()).score_battery(traces)
    finally:
        # Reap leaked agent background services (file watchers + event loop) before the loop is
        # torn down — but only for a runner we created; an injected runner is the caller's to own.
        if owns_runner:
            _drain_and_close(run)
        if previous is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = previous


def run_flag_ab(
    flag: str,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    variants: int = 2,
    max_turns: int = 12,
    out_dir: str = ".",
    workdir: Optional[str] = None,
    isolate_cwd: bool = True,
    **arm_kwargs: Any,
) -> dict:
    """Run both arms (flag OFF then ON), write each snapshot to JSON, and assess graduation.

    Runs in an **isolated working directory** by default (``isolate_cwd``): the agent keys project
    discovery/indexing off the process CWD, so invoking from a large repo would make every task
    re-index that tree (huge CPU, no A/B progress). We chdir into ``workdir`` (or a throwaway temp
    dir) for the run and restore the original CWD afterward. Output paths are resolved absolutely
    first, so ``out_dir`` still lands where you expect.

    Returns a dict with the two snapshot paths, both battery ``to_dict()``s, and — when the
    graduation decider is importable — the GRADUATE/HOLD report.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    original_cwd = os.getcwd()
    scratch: Optional[tempfile.TemporaryDirectory] = None
    if isolate_cwd:
        if workdir:
            Path(workdir).mkdir(parents=True, exist_ok=True)
            os.chdir(workdir)
        else:
            scratch = tempfile.TemporaryDirectory(prefix="victor-flagab-cwd-")
            os.chdir(scratch.name)
    try:
        return _run_flag_ab_arms(
            flag,
            out,
            model=model,
            base_url=base_url,
            variants=variants,
            max_turns=max_turns,
            **arm_kwargs,
        )
    finally:
        os.chdir(original_cwd)
        if scratch is not None:
            scratch.cleanup()


def _run_flag_ab_arms(
    flag: str,
    out: Path,
    *,
    model: Optional[str],
    base_url: Optional[str],
    variants: int,
    max_turns: int,
    **arm_kwargs: Any,
) -> dict:
    baseline = run_battery_arm(
        flag,
        False,
        model=model,
        base_url=base_url,
        variants=variants,
        max_turns=max_turns,
        **arm_kwargs,
    )
    candidate = run_battery_arm(
        flag,
        True,
        model=model,
        base_url=base_url,
        variants=variants,
        max_turns=max_turns,
        **arm_kwargs,
    )
    off_path = out / f"{flag}_off.json"
    on_path = out / f"{flag}_on.json"
    off_path.write_text(json.dumps(baseline.to_dict(), indent=2) + "\n", encoding="utf-8")
    on_path.write_text(json.dumps(candidate.to_dict(), indent=2) + "\n", encoding="utf-8")

    graduation = None
    try:
        from victor.evaluation.flag_graduation import assess_graduation

        graduation = assess_graduation(flag, baseline, candidate).to_dict()
    except Exception:  # noqa: BLE001 — decider is optional; snapshots stand alone
        logger.debug("flag_graduation unavailable; emitting snapshots only", exc_info=True)

    return {
        "flag": flag,
        "baseline_path": str(off_path),
        "candidate_path": str(on_path),
        "baseline_battery": baseline.to_dict(),
        "candidate_battery": candidate.to_dict(),
        "graduation": graduation,
    }


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m victor.evaluation.flag_ab",
        description="A/B a runtime flag (OFF vs ON) over the agent; emit battery snapshots + a "
        "graduation verdict.",
    )
    parser.add_argument("--flag", required=True, choices=sorted(FLAG_ENV))
    parser.add_argument("--model", default=None, help="Provider model (e.g. qwen3-coder-tools:30b)")
    parser.add_argument(
        "--base-url", default=None, help="Provider base URL (e.g. http://172.31.160.1:11434)"
    )
    parser.add_argument("--variants", type=int, default=2, help="Task variants per family (6×N)")
    parser.add_argument(
        "--max-turns", type=int, default=12, help="Per-task turn budget (bounds cost; default 12)"
    )
    parser.add_argument(
        "--corpus",
        default="calibration",
        choices=CORPORA,
        help="Task set: 'calibration' (general) or 'effect-gate' (tempts premature completion)",
    )
    parser.add_argument(
        "--score",
        default="trajectory",
        choices=("trajectory", "verify"),
        help="Metric: 'trajectory' dimension scores, or 'verify' task pass-rate "
        "(the fair effect-gate signal)",
    )
    parser.add_argument("--out-dir", default=".", help="Where to write the two snapshot JSONs")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Working dir to run the agent from (default: a throwaway temp dir). Keep this OUT of "
        "a large repo — the agent indexes its CWD, so running inside the source tree re-indexes it "
        "per task.",
    )
    args = parser.parse_args(argv)

    result = run_flag_ab(
        args.flag,
        model=args.model,
        base_url=args.base_url,
        variants=args.variants,
        max_turns=args.max_turns,
        corpus=args.corpus,
        score=args.score,
        out_dir=args.out_dir,
        workdir=args.workdir,
    )
    graduation = result.get("graduation")
    print(
        json.dumps(
            {
                "flag": result["flag"],
                "baseline_path": result["baseline_path"],
                "candidate_path": result["candidate_path"],
                "graduation": graduation,
            },
            indent=2,
        )
    )
    if graduation is not None:
        return 0 if graduation.get("should_graduate") else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_main())
