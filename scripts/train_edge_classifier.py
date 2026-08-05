#!/usr/bin/env python
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Train the edge-classifier artifact (FEP-0012 Phase 3).

DEV-ONLY (requires the ``[ml]`` extra: scikit-learn/scipy).

Trains one linear head per DecisionType from a CSV of ``(text, label,
decision_type)`` rows and writes a shippable ``.npz`` artifact.

    python scripts/train_edge_classifier.py \
        --csv training_data.csv \
        --out victor/models/edge_classifier_v1.npz \
        --version 1

CSV format (header required): ``text,label,decision_type``.

v1 supervision: the caller supplies labels (e.g. LLM-sourced from
``decisions.jsonl``). The production path supplies reward-derived labels
(``decision_outcome.attributed_reward``) — only the data source changes, not the
trainer.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def load_csv(path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Load ``text,label,decision_type`` rows grouped by decision_type."""
    per_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            sys.exit("CSV must have columns: text,label,decision_type")
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            dtype = (row.get("decision_type") or "").strip()
            if text and label and dtype:
                per_type[dtype].append((text, label))
    return per_type


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the FEP-0012 edge classifier.")
    parser.add_argument("--csv", required=True, help="training data CSV (text,label,decision_type)")
    parser.add_argument("--out", required=True, help="output .npz artifact path")
    parser.add_argument("--version", default="1", help="model version stamp")
    parser.add_argument("--threshold", type=float, default=0.6, help="confidence gate tau")
    args = parser.parse_args()

    from victor.ml.trainer import train_model

    per_type = load_csv(args.csv)
    if not per_type:
        sys.exit("no training rows found in CSV")
    counts = {k: len(v) for k, v in per_type.items()}
    print(f"training heads: {counts}")

    model = train_model(per_type, model_version=args.version, threshold=args.threshold)
    model.save(args.out)
    print(
        f"wrote artifact: {args.out} "
        f"(model_version={model.model_version}, "
        f"feature_spec={model.feature_spec_version}, "
        f"heads={list(model.heads)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
