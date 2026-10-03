#!/usr/bin/env python3
"""Generate FEP-0012 prediction goldens with the ORIGINAL numpy formulation.

One-off (rerun when the artifact or featurizer changes): loads the shipped
artifact via numpy and computes predictions exactly as ``victor.ml.model``
did BEFORE the pure-Python rewrite — ``score = bias + W·x + α·(delta·x)``,
max-shifted ``np.exp`` softmax, ``np.argmax``. The committed goldens then pin
the numpy-free runtime implementation against those numbers
(``tests/unit/ml/test_predict_golden.py``), so the parity evidence is
generated once with numpy and verified everywhere without it.

Usage:
    python scripts/generate_fep0012_goldens.py   # writes the goldens JSON

Requires numpy + the shipped artifact. Inputs with a top-2 probability gap
below 1e-3 are skipped (argmax could flip between implementations).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "victor" / "models" / "edge_classifier_v1.npz"
OUT = ROOT / "tests" / "unit" / "ml" / "goldens" / "fep0012_predict_goldens.json"

sys.path.insert(0, str(ROOT))
from victor.ml.features import FEATURE_SPEC_VERSION, extract_features  # noqa: E402

# Realistic decision excerpts (the texts the classifier sees at runtime).
TEXTS = [
    "implement the feature and add tests for the parser",
    "run the full test suite and report failures",
    "fix the race condition in the connection pool",
    "refactor the storage layer to use the repository pattern",
    "write documentation for the public API",
    "review the pull request and flag security issues",
    "deploy the service to staging and run smoke tests",
    "debug why the scheduler skips every second task",
    "add retry logic with exponential backoff to the client",
    "extract the validation logic into a shared module",
    "migrate the database schema and backfill rows",
    "profile the hot loop and reduce allocations",
    "update dependencies and fix breaking changes",
    "search the codebase for all usages of the config loader",
    "draft the migration plan for the auth service",
    "the build failed on CI after the dependency bump",
    "tests are flaky: the websocket test times out intermittently",
    "the parser crashes on empty input files",
    "memory usage grows during long indexing sessions",
    "shipping the caching layer today; benchmarks look good",
    "still need to wire the config through the CLI flags",
    "blocked: waiting on the API contract review",
    "completed the rollout; error rate flat at 0.1 percent",
    "TODO add pagination to the results endpoint",
    "this milestone covers indexing plus query rewriting",
]


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def numpy_predict(head_data: dict, text: str, alpha: float, delta=None):
    """The pre-rewrite formulation, verbatim (see model.py history)."""
    features = extract_features(text)
    if not features:
        return None
    scores = head_data["bias"].astype(float).copy()
    weights = head_data["weights"]
    for h, val in features.items():
        row = weights.get(h)
        if row is not None:
            scores += row * val
        if delta is not None:
            drow = delta.get(h)
            if drow is not None:
                scores += alpha * np.asarray(drow, dtype=float) * val
    probs = softmax(scores)
    best = int(np.argmax(probs))
    return head_data["labels"][best], float(probs[best]), float(np.sort(probs)[-2])


def main() -> None:
    data = np.load(ARTIFACT, allow_pickle=False)
    assert str(data["feature_spec_version"]) == FEATURE_SPEC_VERSION
    alpha = float(data["alpha"])

    goldens = []
    skipped = 0
    for name in data["head_names"].tolist():
        name = str(name)
        hashes = data[f"{name}__hashes"]
        coefs = data[f"{name}__coefs"]
        head_data = {
            "bias": data[f"{name}__bias"],
            "labels": [str(x) for x in data[f"{name}__labels"]],
            "weights": {int(h): coefs[i] for i, h in enumerate(hashes)},
        }
        threshold = float(data[f"{name}__threshold"])

        for text in TEXTS:
            features = extract_features(text)

            # Deterministic synthetic delta drawn from the text's own features,
            # exercising the alpha-blend branch (per-label weight lists).
            n_labels = len(head_data["labels"])
            delta = {h: [0.05] * n_labels for h in list(features)[:10]}

            for label_tag, delta_arg in (("plain", None), ("delta", delta)):
                out = numpy_predict(head_data, text, alpha, delta_arg)
                if out is None:
                    skipped += 1
                    continue
                label, confidence, runner_up = out
                if confidence - runner_up < 1e-3:
                    skipped += 1  # near-tie: argmax could flip between impls
                    continue
                goldens.append(
                    {
                        "decision_type": name,
                        "text": text,
                        "delta": (
                            {str(k): list(v) for k, v in delta_arg.items()} if delta_arg else None
                        ),
                        "expected_label": label,
                        "expected_confidence": confidence,
                        "threshold": threshold,
                    }
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"feature_spec_version": FEATURE_SPEC_VERSION, "goldens": goldens}, indent=1)
    )
    print(f"wrote {len(goldens)} goldens ({skipped} skipped: empty features or near-ties) to {OUT}")


if __name__ == "__main__":
    main()
