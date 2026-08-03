#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Build a real-styled BALANCED judge dataset (E2 iteration 2, no GPU).

Real positives come from a prior real-agent run's rendered examples (label 1).
Real-styled negatives are synthesized from those same positives: each real
transcript is re-rendered against its task's UNSOLVED fixture (the ADR-010
completion-without-effect case, in real narration). Both classes are therefore
in real style — the fix for the scripted/real collapse documented in
benchmarks/judge_training/FINDINGS.md. Train/dev split by variant index.

Usage:
    python benchmarks/judge_training/synthesize_real_dataset.py \
        --real-positives ~/.victor/models/judge/dataset-real \
        --max-variant 40 --dev-variant-start 36 \
        --out ~/.victor/models/judge/dataset-realbal
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from victor.evaluation.calibration_corpus import default_corpus
from victor.evaluation.judge_training_data import (
    TrainingExample,
    parse_rendered_view,
    split_by_variant,
    synthesize_effect_removed_negative,
    write_jsonl,
)


def _load_positives(dataset_dir: Path) -> list[TrainingExample]:
    out = []
    for split in ("train.jsonl", "dev.jsonl"):
        p = dataset_dir / split
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["label"]) == 1:
                out.append(
                    TrainingExample(
                        task_id=row["task_id"],
                        family=row["family"],
                        source=row["source"],
                        text=row["text"],
                        label=1,
                    )
                )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-positives", type=Path, default=Path.home() / ".victor/models/judge/dataset-real"
    )
    parser.add_argument("--max-variant", type=int, default=40)
    parser.add_argument("--dev-variant-start", type=int, default=36)
    parser.add_argument(
        "--out", type=Path, default=Path.home() / ".victor/models/judge/dataset-realbal"
    )
    args = parser.parse_args()

    positives = _load_positives(args.real_positives)
    tasks = {t.task_id: t for t in default_corpus(variants=args.max_variant)}
    print(f"real positives: {len(positives)}")

    negatives = []
    skipped = 0
    for ex in positives:
        task = tasks.get(ex.task_id)
        if task is None:
            skipped += 1
            continue
        _prompt, transcript = parse_rendered_view(ex.text)
        try:
            negatives.append(synthesize_effect_removed_negative(task, transcript))
        except ValueError:
            skipped += 1  # fixture already verifies complete (e.g. some qa forms)
    print(f"synthesized real-styled negatives: {len(negatives)} (skipped {skipped})")

    combined = positives + negatives
    train, dev = split_by_variant(combined, dev_variant_start=args.dev_variant_start)
    n_train = write_jsonl(train, args.out / "train.jsonl")
    n_dev = write_jsonl(dev, args.out / "dev.jsonl")
    print(
        f"total {len(combined)} (labels {dict(Counter(e.label for e in combined))}) → "
        f"train {n_train} {dict(Counter(e.label for e in train))}, "
        f"dev {n_dev} {dict(Counter(e.label for e in dev))} → {args.out}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
