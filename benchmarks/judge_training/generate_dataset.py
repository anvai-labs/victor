#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Generate the judge-training dataset (E2).

Scripted mode (default, zero LLM calls): easy alternating executors at several
periods (outcome diversity + class balance) plus the hard three-outcome
executor (flawed "looks-solved-but-wrong" cases). Labels are programmatic
verifier gold (κ=1.0-validated, FINDINGS run 12).

Real-agent mode (``--agent-profile``): runs the real agent on each task — the
same executor the calibration harness uses — so the training distribution
MATCHES production. This is the fix for the arm-A/arm-B distribution-shift
failure (both scripted-trained judges scored α=−0.266 on real trajectories,
FINDINGS run 12 / benchmarks/judge_training/FINDINGS.md). ``--variant-start``
drops the low variants so the training set is DISJOINT from the run-12 eval
pack (variants 0–15) — the eval pack stays pure.

Train/dev split is by variant index so surface-variant near-duplicates never
straddle the split.

Usage:
    # Scripted (fast baseline data)
    python benchmarks/judge_training/generate_dataset.py \
        --variants 350 --dev-variant-start 300

    # Real-agent (production distribution), variants 16+ to avoid the eval pack
    python benchmarks/judge_training/generate_dataset.py \
        --agent-profile default --agent-base-url http://192.168.1.20:11434 \
        --variants 48 --variant-start 16 --dev-variant-start 44 \
        --out ~/.victor/models/judge/dataset-real
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from victor.evaluation.calibration_corpus import default_corpus
from victor.evaluation.judge_calibration_harness import (
    alternating_scripted_executor,
    hard_scripted_executor,
)
from victor.evaluation.judge_training_data import (
    generate_training_examples,
    split_by_variant,
    write_jsonl,
)

# (source tag, executor factory). Periods chosen for outcome diversity and a
# roughly balanced label mix; the hard executor contributes the flawed cases.
_SOURCES = (
    ("easy-p2", lambda: alternating_scripted_executor(period=2)),
    ("easy-p3", lambda: alternating_scripted_executor(period=3)),
    ("easy-p5", lambda: alternating_scripted_executor(period=5)),
    ("hard", lambda: hard_scripted_executor()),
)


def _agent_sources(profile: str, base_url: str | None, model: str | None, timeout: int):
    """A single real-agent executor source (production distribution)."""
    from victor.evaluation.agent_adapter import VictorAgentAdapter
    from victor.evaluation.calibration_agent_executor import make_agent_executor

    def factory():
        adapter = VictorAgentAdapter.from_profile(profile, base_url=base_url, model_override=model)
        return make_agent_executor(adapter, timeout_seconds=timeout)

    return (("real-agent", factory),)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=int, default=350)
    parser.add_argument(
        "--variant-start",
        type=int,
        default=0,
        help="Drop variants below this index (use 16+ to stay disjoint from the "
        "run-12 eval pack, which is variants 0-15).",
    )
    parser.add_argument(
        "--dev-variant-start",
        type=int,
        default=300,
        help="Variants >= this index go to dev.jsonl (split by variant, not randomly).",
    )
    parser.add_argument(
        "--agent-profile",
        default=None,
        help="Generate REAL agent trajectories with this profile instead of the scripted "
        "executors (the production-distribution training data).",
    )
    parser.add_argument("--agent-base-url", default=None)
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--agent-timeout", type=int, default=240)
    parser.add_argument("--out", type=Path, default=Path.home() / ".victor/models/judge/dataset")
    args = parser.parse_args()

    # Silence the framework flood before any agent runs (real-agent mode drives
    # the full orchestrator, which logs per turn/workspace).
    import logging

    logging.getLogger("victor").setLevel(logging.WARNING)

    corpus = default_corpus(variants=args.variants)
    if args.variant_start > 0:
        corpus = corpus[args.variant_start * 6 :]  # 6 families per variant

    if args.agent_profile:
        sources = _agent_sources(
            args.agent_profile, args.agent_base_url, args.agent_model, args.agent_timeout
        )
        print(f"executor: real agent (profile={args.agent_profile})")
    else:
        sources = _SOURCES

    all_examples = []
    for source, make_executor in sources:
        examples = generate_training_examples(corpus, make_executor(), source=source)
        all_examples.extend(examples)
        labels = Counter(e.label for e in examples)
        print(f"{source}: {len(examples)} examples, labels {dict(labels)}")

    train, dev = split_by_variant(all_examples, dev_variant_start=args.dev_variant_start)
    n_train = write_jsonl(train, args.out / "train.jsonl")
    n_dev = write_jsonl(dev, args.out / "dev.jsonl")
    total_labels = Counter(e.label for e in all_examples)
    print(
        f"\ntotal {len(all_examples)} (labels {dict(total_labels)}) → "
        f"train {n_train}, dev {n_dev} → {args.out}/"
    )
    return 0


if __name__ == "__main__":
    import os
    import sys

    _code = main()
    # Force exit after the JSONL is written. In --agent-profile mode the full
    # orchestrator runs, leaving non-daemon threads and open event loops alive
    # (SharedSignalPool, DB connections); a normal SystemExit then HANGS waiting
    # on them — observed live: a real-agent run finished writing its dataset but
    # spun a CPU core in the event-dispatch loop for ~90 min afterward. Same
    # os._exit workaround as run_offline_calibration.py. Flush first (os._exit
    # skips buffer flushing).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
