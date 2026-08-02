#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Generate the judge-training dataset (E2) — zero LLM calls.

Composition (per the approved experiment plan): the easy alternating executor
at several periods (outcome diversity + class balance) plus the hard
three-outcome executor (flawed "looks-solved-but-wrong" cases — the
discrimination signal). Labels are programmatic verifier gold
(κ=1.0-validated, FINDINGS run 12). Train/dev split is by variant index so
surface-variant near-duplicates never straddle the split.

Usage:
    python benchmarks/judge_training/generate_dataset.py \
        --variants 350 --dev-variant-start 300 \
        --out ~/.victor/models/judge/dataset
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=int, default=350)
    parser.add_argument(
        "--dev-variant-start",
        type=int,
        default=300,
        help="Variants >= this index go to dev.jsonl (split by variant, not randomly).",
    )
    parser.add_argument("--out", type=Path, default=Path.home() / ".victor/models/judge/dataset")
    args = parser.parse_args()

    corpus = default_corpus(variants=args.variants)
    all_examples = []
    for source, make_executor in _SOURCES:
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
    raise SystemExit(main())
