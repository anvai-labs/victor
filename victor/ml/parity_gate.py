# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012 Phase 7: the parity gate for shipping a trained classifier.

The adoption wiring already exists — ``auto`` adopts ``LocalClassifierDecisionService``
when an artifact is healthy (Phase 5). What was missing is the **validation**
that says *ship only at parity*: does the classifier, on held-out decisions,
beat a naive baseline often enough to be worth adopting?

This module is that gate. The ``decision_outcome`` junction (Phase 6) is the
ground-truth dataset: each held-out decision has a true reward label
(pass/partial/fail). We hold out a fraction, train on the rest, and evaluate
the classifier's accuracy on the unseen holdout against a **majority-class
baseline** (always predict the most frequent label — the bar the classifier
must clear to add signal).

Metrics (per decision type)
---------------------------
- ``coverage``   — fraction of holdout decisions the classifier gave an
  opinion on (``predict`` returns non-None above τ). Abstaining forever is
  useless even if "never wrong".
- ``calibrated_accuracy`` — accuracy among the decisions it opined on.
- ``baseline_accuracy``   — majority-class accuracy (the naive bar).
- ``overall_accuracy``    — correct/total (abstentions count as wrong).

Ship verdict
------------
A type ships when it has enough samples, adequate coverage, and
``calibrated_accuracy ≥ baseline + min_margin``. Overall ship requires the key
type (``task_completion``) to clear the bar — the premature-completion signal
is the classifier's reason to exist.

CLI: ``python -m victor.ml.parity_gate [--holdout-frac 0.2]``
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any, Optional

from victor.ml.outcome_training import load_outcome_samples

logger = logging.getLogger(__name__)

# The decision type whose head must clear the bar for an overall "ship".
KEY_TYPE = "task_completion"


def train_test_split_by_type(
    samples: dict[str, list[tuple[str, str]]],
    *,
    holdout_frac: float = 0.2,
    seed: int = 0,
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]]:
    """Split per-type samples into (train, holdout) deterministically."""
    rng = random.Random(seed)
    train: dict[str, list[tuple[str, str]]] = {}
    holdout: dict[str, list[tuple[str, str]]] = {}
    for dtype, items in samples.items():
        items = list(items)
        rng.shuffle(items)
        n_hold = max(1, int(round(len(items) * holdout_frac))) if len(items) > 1 else 0
        if n_hold == 0:
            train[dtype] = items
            continue
        holdout[dtype] = items[:n_hold]
        train[dtype] = items[n_hold:]
    return train, holdout


def _majority_label(items: list[tuple[str, str]]) -> str:
    counts = Counter(label for _, label in items)
    return counts.most_common(1)[0][0]


def evaluate_on_holdout(
    model: Any, holdout: dict[str, list[tuple[str, str]]]
) -> dict[str, dict[str, float]]:
    """Score the classifier on held-out decisions per type.

    Returns ``{decision_type: {n, coverage, calibrated_accuracy,
    overall_accuracy, baseline_accuracy}}``. Types the model has no head for
    get ``coverage=0``.
    """
    metrics: dict[str, dict[str, float]] = {}
    for dtype, items in holdout.items():
        n = len(items)
        if n == 0:
            continue
        baseline_label = _majority_label(items)
        baseline_correct = sum(1 for _, lbl in items if lbl == baseline_label)
        opinions = 0
        opinion_correct = 0
        overall_correct = 0
        for text, true_label in items:
            pred, _conf = model.predict(dtype, text)
            if pred is not None:
                opinions += 1
                if pred == true_label:
                    opinion_correct += 1
                    overall_correct += 1
            # abstentions (None) count as wrong for overall_accuracy
        metrics[dtype] = {
            "n": float(n),
            "coverage": opinions / n,
            "calibrated_accuracy": (opinion_correct / opinions) if opinions else 0.0,
            "overall_accuracy": overall_correct / n,
            "baseline_accuracy": baseline_correct / n,
            "margin": ((opinion_correct / opinions) - (baseline_correct / n) if opinions else 0.0),
        }
    return metrics


def ship_verdict(
    metrics: dict[str, dict[str, float]],
    *,
    min_samples: int = 20,
    min_coverage: float = 0.5,
    min_margin: float = 0.0,
    min_calibrated_accuracy: Optional[float] = None,
    key_type: str = KEY_TYPE,
) -> dict[str, Any]:
    """Decide whether to ship per type and overall.

    A type ships when it has ≥ ``min_samples`` holdout decisions, coverage ≥
    ``min_coverage``, and ``calibrated_accuracy ≥ baseline + min_margin`` (and
    ≥ ``min_calibrated_accuracy`` if set). Overall ships when the key type
    (``task_completion``) ships.
    """
    per_type: dict[str, dict[str, Any]] = {}
    for dtype, m in metrics.items():
        enough_samples = m["n"] >= min_samples
        enough_coverage = m["coverage"] >= min_coverage
        beats_baseline = m["margin"] >= min_margin
        meets_floor = (
            min_calibrated_accuracy is None or m["calibrated_accuracy"] >= min_calibrated_accuracy
        )
        ship = enough_samples and enough_coverage and beats_baseline and meets_floor
        per_type[dtype] = {
            "ship": ship,
            "n": int(m["n"]),
            "coverage": round(m["coverage"], 3),
            "calibrated_accuracy": round(m["calibrated_accuracy"], 3),
            "baseline_accuracy": round(m["baseline_accuracy"], 3),
            "margin": round(m["margin"], 3),
        }

    key = per_type.get(key_type)
    overall_ship = bool(key and key["ship"])
    return {
        "ship": overall_ship,
        "key_type": key_type,
        "key_ships": overall_ship,
        "per_type": per_type,
    }


def validate_outcome_training(
    *,
    holdout_frac: float = 0.2,
    seed: int = 0,
    decision_types: Optional[list[str]] = None,
    min_samples: int = 20,
    min_coverage: float = 0.5,
    min_margin: float = 0.0,
) -> dict[str, Any]:
    """End-to-end parity gate: load outcomes → split → train → evaluate → verdict.

    Trains a fresh classifier on the train split and scores it on the unseen
    holdout (so the verdict isn't inflated by training-data leakage). Returns
    the ship verdict + metrics + counts. Needs the [ml] extra to train.
    """
    samples = load_outcome_samples(decision_types)
    if not samples:
        return {
            "ship": False,
            "reason": "no reward-labeled outcomes (run a benchmark + record_session_outcome first)",
            "per_type": {},
        }

    train_split, holdout = train_test_split_by_type(samples, holdout_frac=holdout_frac, seed=seed)

    # Need ≥2 distinct labels on the train split for train_head to fit a type.
    trainable = {
        dtype: items
        for dtype, items in train_split.items()
        if len({lbl for _, lbl in items}) >= 2 and len(items) >= 2
    }
    if not trainable:
        return {
            "ship": False,
            "reason": "insufficient label diversity to train (need pass AND fail tasks)",
            "per_type": {},
            "n_train_types": 0,
        }

    try:
        from victor.ml.trainer import train_model
    except ImportError as exc:
        return {
            "ship": False,
            "reason": f"training needs the [ml] extra (scikit-learn/scipy): {exc}",
            "per_type": {},
        }

    model = train_model(trainable, model_version="parity-gate")
    metrics = evaluate_on_holdout(model, holdout)
    verdict = ship_verdict(
        metrics,
        min_samples=min_samples,
        min_coverage=min_coverage,
        min_margin=min_margin,
    )
    verdict["n_train"] = {dtype: len(items) for dtype, items in trainable.items()}
    verdict["n_holdout"] = {
        dtype: len(items) for dtype, items in holdout.items() if dtype in metrics
    }
    return verdict


def _main(argv: Optional[list[str]] = None) -> int:
    """``python -m victor.ml.parity_gate [--holdout-frac 0.2]``."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="FEP-0012 Phase 7 parity gate")
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.0)
    args = parser.parse_args(argv)

    verdict = validate_outcome_training(
        holdout_frac=args.holdout_frac,
        min_samples=args.min_samples,
        min_coverage=args.min_coverage,
        min_margin=args.min_margin,
    )
    if "reason" in verdict:
        print(f"NO VERDICT: {verdict['reason']}")
        return 1

    print(f"\n{'type':<20} {'ship':<6} {'n':>4} {'cov':>6} {'acc':>6} {'base':>6} {'margin':>7}")
    print("-" * 60)
    for dtype, m in verdict["per_type"].items():
        print(
            f"{dtype:<20} {'YES' if m['ship'] else 'no':<6} {m['n']:>4} "
            f"{m['coverage']:>6.2f} {m['calibrated_accuracy']:>6.2f} "
            f"{m['baseline_accuracy']:>6.2f} {m['margin']:>+7.2f}"
        )
    print("-" * 60)
    print(
        f"OVERALL: {'SHIP' if verdict['ship'] else 'DO NOT SHIP'} (key type={verdict['key_type']})"
    )
    return 0 if verdict["ship"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())
