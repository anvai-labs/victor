#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Train the arm-A linear completion judge (E2, zero new dependencies).

Hashing-trick features (victor/ml/features.py, versioned spec) + sklearn
LogisticRegression on the generated dataset's ``text``/``label`` JSONL.
Reports dev accuracy AND dev Krippendorff α (the gate metric — accuracy on
an imbalanced set flatters). Artifact: sparse-weight ``.npz`` loadable by
``victor.evaluation.calibration_classifier_judge.load_linear_judge`` with a
pure-numpy scorer.

Usage:
    python benchmarks/judge_training/train_linear.py \
        --dataset ~/.victor/models/judge/dataset \
        --out ~/.victor/models/judge/linear_v1.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts.append(row["text"])
        labels.append(int(row["label"]))
    return texts, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path.home() / ".victor/models/judge/dataset"
    )
    parser.add_argument(
        "--out", type=Path, default=Path.home() / ".victor/models/judge/linear_v1.npz"
    )
    parser.add_argument("--c", type=float, default=1.0, help="LogisticRegression C")
    args = parser.parse_args()

    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression

    from victor.evaluation.judge_calibration import evaluate_judge_agreement
    from victor.ml.features import FEATURE_SPEC_VERSION, HASH_SPACE, extract_features

    def featurize(texts: list[str]) -> csr_matrix:
        rows, cols, vals = [], [], []
        for i, text in enumerate(texts):
            for feature_hash, count in extract_features(text).items():
                rows.append(i)
                cols.append(feature_hash)
                vals.append(count)
        return csr_matrix((vals, (rows, cols)), shape=(len(texts), HASH_SPACE))

    train_texts, train_labels = _load(args.dataset / "train.jsonl")
    dev_texts, dev_labels = _load(args.dataset / "dev.jsonl")
    print(f"train n={len(train_texts)}, dev n={len(dev_texts)}")

    model = LogisticRegression(C=args.c, max_iter=500, solver="lbfgs", class_weight="balanced")
    model.fit(featurize(train_texts), train_labels)

    dev_pred = model.predict_proba(featurize(dev_texts))[:, 1]
    dev_binary = [1.0 if p >= 0.5 else 0.0 for p in dev_pred]
    accuracy = sum(int(b == g) for b, g in zip(dev_binary, dev_labels)) / len(dev_labels)
    reliability = evaluate_judge_agreement(
        [float(g) for g in dev_labels], dev_binary, level="nominal"
    )
    print(f"dev accuracy={accuracy:.4f}  dev α={reliability.krippendorff_alpha:.4f}")

    coef = model.coef_[0]
    nonzero = np.nonzero(coef)[0]
    np.savez(
        args.out,
        indices=nonzero.astype(np.int64),
        coefs=coef[nonzero].astype(np.float64),
        intercept=np.float64(model.intercept_[0]),
        feature_spec_version=FEATURE_SPEC_VERSION,
    )
    print(f"artifact: {args.out} ({len(nonzero)} nonzero weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
