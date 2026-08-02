#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Offline judge-calibration demo (EVR-2, ADR-011) — runs with zero LLM calls.

Runs the repo-local verifiable corpus through a scripted executor (which solves most tasks but
fakes some with "completion-without-effect" claims) and calibrates two heuristic judges against
the programmatic gold labels:

- **credulous**: trusts the agent's claim of completion — the failure mode ADR-011 exists to
  catch; expected to FAIL the α ≥ 0.7 gate (claims are always positive, gold is not).
- **evidence**: requires observable tool activity in the transcript before believing a
  completion claim; expected to PASS on this corpus.

To calibrate a *real* judge (rubric/LLM), implement the same
``(prompt, transcript, workspace) -> float`` callable around it and pass it to
``JudgeCalibrationHarness.run`` — the blinding contract and the report format are identical.
To generate trajectories with a *real* agent instead of the scripted executor, use
``victor.evaluation.calibration_agent_executor.make_agent_executor`` with a
``VictorAgentAdapter.from_profile(...)`` (an Ollama profile keeps the loop fully offline).

Usage:
    python benchmarks/judge_calibration/run_offline_calibration.py [--variants 8] [--out DIR]

    # Calibrate the LLM rubric judge via a Victor profile (provider/model/key from
    # profiles.yaml). If the key lives in the system keyring, bridge it into env first —
    # keyring is skipped without a TTY, but env always resolves:
    eval "$(victor auth env -p zai)"
    python benchmarks/judge_calibration/run_offline_calibration.py \
        --variants 8 --judge-profile zai-coding   # e.g. zai / glm-5.2 (1M context)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from victor.evaluation.calibration_corpus import default_corpus
from victor.evaluation.calibration_enhanced_judge import make_enhanced_judge
from victor.evaluation.calibration_rubric_judge import make_rubric_judge
from victor.evaluation.judge_calibration_harness import (
    JudgeCalibrationHarness,
    Transcript,
    alternating_scripted_executor,
)

# Framework loggers that emit high-volume, per-turn / per-workspace lines during a
# real-agent run — the source of the stdout flood.
_FLOOD_LOGGERS = (
    "victor.agent",
    "victor.providers",
    "victor.core.database",
    "victor.evaluation.agent_adapter",
    "victor.framework.agentic_loop",
)


def configure_logging(verbose: bool) -> None:
    """Keep captured output bounded by silencing the framework's stdout flood.

    A real-agent calibration drives the full orchestrator + provider, which log every
    turn, every provider round-trip, and a per-workspace DB migration. Redirected to a file
    (``> run.log 2>&1``) that stream is UNBOUNDED — a stuck run once wrote ~350 GB of
    trajectories to a single log. Quiet by default (only ERROR from the flood sources,
    WARNING elsewhere); ``--verbose`` restores INFO. When you genuinely want verbose but
    bounded output, pipe through a rotating appender, e.g.
    ``... 2>&1 | python3 ~/tools/caplog.py run.log``.
    """
    logging.getLogger("victor").setLevel(logging.INFO if verbose else logging.WARNING)
    if not verbose:
        for name in _FLOOD_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)


def credulous_judge(_prompt: str, transcript: Transcript, _workspace: Path) -> float:
    """Believes any completion claim in the final message."""
    final = transcript.final_message.lower()
    return 1.0 if ("done" in final or "complete" in final) else 0.0


def evidence_judge(_prompt: str, transcript: Transcript, _workspace: Path) -> float:
    """Scores from evidence of work (tool activity), not from what the agent claims."""
    return 1.0 if transcript.tool_steps() else 0.0


def run_human_label_overlay(
    *, reports_dir: Path, labels_path: Path, secondary_path: Path | None
) -> int:
    """Overlay mode: saved reports + human labels -> agreement report. Zero LLM calls."""
    from victor.evaluation.human_label_overlay import (
        LabelsError,
        compute_overlay,
        load_labels,
        load_report_samples,
    )

    report_files = sorted(f for f in reports_dir.glob("*.json") if f.name != "human_overlay.json")
    if not report_files:
        print(f"no report JSONs found in {reports_dir}/ — run a calibration first")
        return 1
    try:
        judge_reports = {f.stem: load_report_samples(f) for f in report_files}
        human = load_labels(labels_path)
        secondary = load_labels(secondary_path) if secondary_path else None
        overlay = compute_overlay(judge_reports, human, secondary_labels=secondary)
    except LabelsError as exc:
        print(f"overlay VOID: {exc}")
        return 1
    out = reports_dir / "human_overlay.json"
    overlay.save(out)
    for verdict in overlay.verdicts:
        mark = "PASS" if verdict.passed else "FAIL"
        print(f"[{mark}] {verdict.name}: {verdict.detail}")
    if overlay.stop_the_line:
        print(
            "\nSTOP THE LINE: human labels disagree with the programmatic verifier gold "
            "beyond the pre-registered threshold. FINDINGS runs based on that gold are void "
            "until the corpus verifiers are fixed and re-validated."
        )
    print(f"\nOverlay written to {out}")
    return 0 if all(v.passed for v in overlay.verdicts) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", type=int, default=8, help="Variants per task family (6 families)"
    )
    parser.add_argument("--out", type=Path, default=Path("benchmarks/judge_calibration/reports"))
    parser.add_argument(
        "--llm-judge-provider",
        default=None,
        help="Also calibrate the real ADR-009 LLM rubric judge via this provider "
        "(e.g. 'ollama' — keeps the run fully offline). One grading call per task.",
    )
    parser.add_argument(
        "--llm-judge-model",
        default="qwen2.5-coder:7b",
        help="Model for --llm-judge-provider grading calls",
    )
    parser.add_argument(
        "--llm-judge-base-url",
        default=None,
        help="Provider base URL override (e.g. http://<windows-host>:11434 from WSL)",
    )
    parser.add_argument(
        "--judge-profile",
        default=None,
        help="Resolve the LLM judge's provider/model/api-key from a Victor profile "
        "(profiles.yaml), e.g. --judge-profile cerebras-glm. Overrides the "
        "--llm-judge-provider/--llm-judge-model pair.",
    )
    parser.add_argument(
        "--judge-delay",
        type=float,
        default=2.0,
        help="Seconds to pace between LLM grading calls (strict provider RPM caps, "
        "e.g. Z.AI, need 15+; local Ollama can use 0)",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=1024,
        help="Token budget per grading call. Thinking-family judge models spend tokens "
        "on reasoning before the grade lines — if the integrity line reports "
        "ungradable>0, raise this (2048+).",
    )
    parser.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="Keep the per-task workspace dirs for inspection (default: cleaned up). "
        "Useful for diagnosing what a judge actually saw on a bad run.",
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Use the three-outcome scripted executor (correct / flawed / fake) instead of "
        "the two-outcome one. Flawed cases look solved but are subtly wrong — the "
        "discrimination test for judges that saturate the easy corpus at α=1.0.",
    )
    parser.add_argument(
        "--export-labeling-pack",
        type=Path,
        default=None,
        metavar="DIR",
        help="Export the EVR-2 human-labeling pack (blinded per-task records + labels "
        "template) to DIR during the run. See "
        "docs/architecture/evr2-human-validation-protocol.md.",
    )
    parser.add_argument(
        "--human-labels",
        type=Path,
        default=None,
        metavar="JSONL",
        help="OVERLAY MODE (no executor, no LLM calls): recompute human-vs-verifier and "
        "human-vs-judge agreement from saved reports in --out plus this labels file, "
        "write human_overlay.json, and exit. Thresholds are pre-registered in the "
        "EVR-2 protocol doc.",
    )
    parser.add_argument(
        "--secondary-labels",
        type=Path,
        default=None,
        metavar="JSONL",
        help="Optional second annotator's labels for the overlay's annotation-QC κ "
        "(reported, never gating).",
    )
    parser.add_argument(
        "--classifier-judge",
        type=Path,
        default=None,
        metavar="NPZ",
        help="Also score a trained linear classifier judge (E2 arm A artifact from "
        "benchmarks/judge_training/train_linear.py) as judge name 'classifier'.",
    )
    parser.add_argument(
        "--encoder-judge",
        type=Path,
        default=None,
        metavar="DIR",
        help="Also score a trained ModernBERT encoder judge (E2 arm B artifact from "
        "benchmarks/judge_training/train_encoder.py) as judge name 'encoder'. "
        "Requires the [judge-training] extra.",
    )
    parser.add_argument(
        "--replay-pack",
        type=Path,
        default=None,
        metavar="DIR",
        help="Replay an exported labeling pack instead of running any executor: judges "
        "score the recorded (prompt, transcript, workspace) views. Use to measure NEW "
        "judges on a prior run's exact trajectories (e.g. llama3.3:70b on run 12). "
        "Requires the same --variants as the original run.",
    )
    parser.add_argument(
        "--two-phase",
        action="store_true",
        help="Run all trajectories first, then all judging (one model swap instead of one "
        "per task). Auto-enabled in --agent-profile mode; use this to force it for scripted "
        "runs too.",
    )
    parser.add_argument(
        "--no-two-phase",
        action="store_true",
        help="Disable the automatic two-phase scheduling in --agent-profile mode.",
    )
    parser.add_argument(
        "--agent-profile",
        default=None,
        help="Generate REAL agent trajectories with this Victor profile instead of the "
        "scripted executor — the production-distribution validation. Expensive (one full "
        "agent run per task). A capable agent may solve everything (single-class gold); "
        "watch the printed gold distribution.",
    )
    parser.add_argument(
        "--agent-base-url",
        default=None,
        help="Base URL override for the agent's provider (e.g. WSL→Windows Ollama gateway).",
    )
    parser.add_argument(
        "--agent-model",
        default=None,
        help="Model override for the agent profile (default: the profile's model).",
    )
    parser.add_argument(
        "--agent-timeout",
        type=int,
        default=240,
        help="Per-task timeout for a real agent run (seconds).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Restore INFO-level framework logging. OFF by default — the framework floods "
        "stdout with per-turn/per-workspace lines that an unbounded redirect turns into "
        "hundreds of GB.",
    )
    args = parser.parse_args()

    # Silence the framework flood BEFORE any adapter/agent runs (a stuck run wrote ~350 GB
    # to a single redirected log before this existed).
    configure_logging(args.verbose)

    if args.human_labels:
        return run_human_label_overlay(
            reports_dir=args.out,
            labels_path=args.human_labels,
            secondary_path=args.secondary_labels,
        )

    judges = {
        "credulous": credulous_judge,
        "evidence": evidence_judge,
        # The real ADR-009 no-LLM fallback path (HeuristicRubricJudge behind the
        # DimensionAwareFilter) — calibrated as shipped, binary completion verdicts.
        "rubric-heuristic": make_rubric_judge(),
        # The production default completion evaluator (EVR-3 parity, deterministic,
        # zero LLM calls) — see calibration_enhanced_judge for the fidelity contract.
        "enhanced": make_enhanced_judge(),
    }
    if args.classifier_judge:
        from victor.evaluation.calibration_classifier_judge import load_linear_judge

        judges["classifier"] = load_linear_judge(args.classifier_judge)
    if args.encoder_judge:
        from victor.evaluation.calibration_classifier_judge import load_encoder_judge

        judges["encoder"] = load_encoder_judge(args.encoder_judge)
    llm_stats = None
    if args.judge_profile or args.llm_judge_provider:
        from victor.evaluation.calibration_rubric_judge import (
            JudgeCallStats,
            make_llm_rubric_judge,
            make_provider_complete_fn,
        )
        from victor.providers.registry import ProviderRegistry

        provider_name = args.llm_judge_provider
        model = args.llm_judge_model
        provider_kwargs = {}
        if args.llm_judge_base_url:
            provider_kwargs["base_url"] = args.llm_judge_base_url
        if args.judge_profile:
            import os

            from victor.config.settings import load_settings

            profiles = load_settings().load_profiles()
            if args.judge_profile not in profiles:
                parser.error(
                    f"profile '{args.judge_profile}' not found. "
                    f"Available: {', '.join(sorted(profiles))}"
                )
            profile = profiles[args.judge_profile]
            provider_name, model = profile.provider, profile.model
            # Pass through profile extras; ${ENV} references resolve like from_profile.
            api_key = getattr(profile, "api_key", None)
            if api_key and api_key.startswith("${") and api_key.endswith("}"):
                api_key = os.environ.get(api_key[2:-1])
            if api_key:
                provider_kwargs["api_key"] = api_key
            profile_base_url = getattr(profile, "base_url", None) or getattr(
                profile, "endpoint", None
            )
            if profile_base_url and not args.llm_judge_base_url:
                provider_kwargs["base_url"] = profile_base_url
        provider = ProviderRegistry.create(provider_name, **provider_kwargs)
        llm_stats = JudgeCallStats()
        judges["rubric-llm"] = make_llm_rubric_judge(
            make_provider_complete_fn(
                provider,
                model,
                max_tokens=args.judge_max_tokens,
                pre_call_delay_seconds=args.judge_delay,
                stats=llm_stats,
            ),
            stats=llm_stats,
        )
    # Executor: real agent trajectories (--agent-profile) or the deterministic scripted
    # stand-in. period=5 is coprime with the 6 task families, so scripted failures rotate
    # across every family instead of always hitting the same ones.
    if args.replay_pack:
        from victor.evaluation.labeling_pack import make_pack_replay_executor

        executor = make_pack_replay_executor(args.replay_pack)
        print(f"executor: pack replay ({args.replay_pack})")
    elif args.agent_profile:
        from victor.evaluation.agent_adapter import VictorAgentAdapter
        from victor.evaluation.calibration_agent_executor import make_agent_executor

        adapter = VictorAgentAdapter.from_profile(
            args.agent_profile,
            base_url=args.agent_base_url,
            model_override=args.agent_model,
        )
        executor = make_agent_executor(adapter, timeout_seconds=args.agent_timeout)
        print(f"executor: real agent (profile={args.agent_profile})")
    elif args.hard:
        from victor.evaluation.judge_calibration_harness import hard_scripted_executor

        executor = hard_scripted_executor()
        print("executor: scripted (HARD — includes flawed 'looks-solved-but-wrong' cases)")
    else:
        executor = alternating_scripted_executor(period=5)
        print("executor: scripted")

    # ONE executor pass; every judge scores the identical trajectories (essential for the
    # real-agent executor, which is expensive and non-deterministic per run). two-phase
    # (execute all → judge all) avoids per-task model swapping on a single inference server;
    # default it on in agent mode, where the executor and judge use different models.
    two_phase = args.two_phase or (args.agent_profile is not None and not args.no_two_phase)
    if two_phase:
        print("scheduling: two-phase (all trajectories, then all judging) — one model swap")
    harness = JudgeCalibrationHarness(default_corpus(variants=args.variants))
    record_sink = None
    if args.export_labeling_pack:
        from victor.evaluation.labeling_pack import make_labeling_pack_sink

        record_sink = make_labeling_pack_sink(args.export_labeling_pack)
        print(f"labeling pack: exporting blinded records to {args.export_labeling_pack}/")
    reports = harness.run_multi_judge(
        executor,
        judges,
        keep_workspaces=args.keep_workspaces,
        two_phase=two_phase,
        record_sink=record_sink,
    )

    # Gold distribution: agreement (α) is meaningless if every task shares one gold class.
    # A strong real agent can solve everything (all gold=1); flag that the run measures only
    # true-positive rate, not discrimination.
    any_report = next(iter(reports.values()))
    gold_counts: dict[float, int] = {}
    for s in any_report.samples:
        gold_counts[s.gold] = gold_counts.get(s.gold, 0) + 1
    print(f"gold distribution: {gold_counts}")
    degenerate_gold = len(gold_counts) < 2
    if degenerate_gold:
        print(
            "⚠ gold is single-class — α cannot measure discrimination (no negative/positive "
            "contrast). This run only shows whether the judge agrees on that one class."
        )

    exit_code = 0
    for name, report in reports.items():
        report.save(args.out / f"{name}.json")
        decision = report.gate_decision
        verdict = "TRUSTED" if decision.trusted else "NOT TRUSTED"
        if degenerate_gold and verdict == "TRUSTED":
            verdict = "TRUSTED (single-class gold — not discriminative)"
        if name == "rubric-llm" and llm_stats is not None:
            print(f"[rubric-llm] grading-call integrity: {llm_stats.summary()}")
            if not llm_stats.clean:
                verdict = "VOID"
                exit_code = 1
                if llm_stats.failures:
                    print(
                        f"[rubric-llm] ⚠ {llm_stats.failures} grading call(s) failed after "
                        "retries — failed calls degrade to neutral fallback scores, so this α "
                        "measures the error path, NOT the judge. Re-run with a higher "
                        "--judge-delay or a less rate-limited provider."
                    )
                if llm_stats.ungradable:
                    print(
                        f"[rubric-llm] ⚠ {llm_stats.ungradable} grading call(s) returned "
                        "output with NO parseable dimension scores — wrong output format, or a "
                        "thinking-family model exhausting its token budget on reasoning before "
                        "the grade lines. This α measures the parse-fallback path, NOT the "
                        "judge. Try --judge-max-tokens 2048+ or a non-thinking judge model."
                    )
        print(f"[{name}] n={len(report.samples)}  {verdict}  ({decision.reason})")
        for family, rel in report.per_family.items():
            alpha = rel.krippendorff_alpha
            shown = "n/a" if alpha is None or alpha != alpha else f"{alpha:.3f}"
            print(f"    {family:<12} n={rel.n:<3} α={shown}")
    print(f"\nReports written to {args.out}/")
    return exit_code


if __name__ == "__main__":
    import os
    import sys

    _code = main()
    # Force exit after the reports are written. The agent orchestrator leaves non-daemon
    # threads and open event loops alive (SharedSignalPool, DB connections, the per-judge
    # PersistentLoopRunner), which make a normal SystemExit hang WAITING for them — a stuck
    # run once lingered 7.5 h holding its (redirected) log open and filled the disk. os._exit
    # skips buffer flushing, so flush first.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
