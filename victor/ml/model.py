# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Edge-classifier artifact + pure-Python inference (FEP-0012).

The shipped artifact is a set of per-DecisionType linear heads stored as sparse
weights in an ``.npz``. Inference is pure-Python (numpy-optional: heads are
2-3-label score vectors over sparse n-gram lookups — vectorized math buys
nothing at this size), read via :mod:`victor.ml.npz_reader`; no sklearn/torch
at runtime. numpy is used only by the dev-only ``save()`` writer. The
per-project RL delta is blended in at predict time:
``score = bias + W·x + α·(delta·x)``.

A decision whose calibrated confidence is below the head's ``threshold`` (τ)
returns ``(None, confidence)`` so the caller defers to the heuristic (or, if the
LLM edge tier is present, escalates to it) — preserving the existing
confidence-gated fallback behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from victor.core import vecmath
from victor.ml.features import FEATURE_SPEC_VERSION, extract_features
from victor.ml.npz_reader import (
    NpyArray,
    as_float_rows,
    as_float_vector,
    as_int_vector,
    as_scalar_float,
    as_scalar_str,
    as_str_list,
    read_npz,
)


@dataclass
class DecisionHead:
    """One linear classifier head for a DecisionType.

    Attributes:
        decision_type: The DecisionType this head decides (e.g. task_type).
        labels: Ordered class labels.
        weights: Sparse ``{feature_hash: coef_per_label}`` (only non-zero rows).
        bias: Per-label bias, shape ``[len(labels)]``.
        threshold: Confidence gate τ; below it predict() defers (returns None).
    """

    decision_type: str
    labels: List[str]
    weights: Dict[int, List[float]]
    bias: List[float]
    threshold: float = 0.6


@dataclass
class EdgeClassifierModel:
    """A shipped edge-classifier artifact (universal baseline)."""

    heads: Dict[str, DecisionHead] = field(default_factory=dict)
    feature_spec_version: str = FEATURE_SPEC_VERSION
    model_version: str = "0"
    alpha: float = 0.3  # delta blend weight

    # ------------------------------------------------------------------ predict
    def predict(
        self,
        decision_type: str,
        text: str,
        delta: Optional[Dict[int, List[float]]] = None,
    ) -> Tuple[Optional[str], float]:
        """Predict the label for ``text`` under ``decision_type``.

        Args:
            decision_type: The DecisionType key.
            text: Decision input text (featurized via the versioned extractor).
            delta: Optional per-project RL overlay ``{hash: coef_per_label}``.

        Returns:
            ``(label, confidence)``. ``label`` is ``None`` when the head is
            unknown for this type OR confidence < τ (caller defers to heuristic).
        """
        head = self.heads.get(decision_type)
        if head is None:
            return None, 0.0

        features = extract_features(text)
        if not features:
            return None, 0.0

        scores = list(head.bias)
        weights = head.weights
        for h, val in features.items():
            row = weights.get(h)
            if row is not None:
                scores = [s + c * val for s, c in zip(scores, row)]
            if delta is not None:
                drow = delta.get(h)
                if drow is not None:
                    scores = [s + self.alpha * c * val for s, c in zip(scores, drow)]

        probs = vecmath.softmax(scores)
        best = vecmath.argmax(probs)
        confidence = float(probs[best])
        if confidence < head.threshold:
            return None, confidence
        return head.labels[best], confidence

    # ------------------------------------------------------------ save / load
    def save(self, path: str) -> None:
        """Persist the model to an ``.npz`` artifact (sparse weights).

        Dev-only path (the [ml]/dev extras ship numpy); runtime only reads.
        """
        import numpy as np

        arrays: Dict[str, np.ndarray] = {
            "feature_spec_version": np.array(self.feature_spec_version),
            "model_version": np.array(self.model_version),
            "alpha": np.array(self.alpha),
            "head_names": np.array(sorted(self.heads.keys())),
        }
        for name, head in self.heads.items():
            hashes = np.fromiter(head.weights.keys(), dtype=np.int64, count=len(head.weights))
            coefs = (
                np.stack([np.asarray(row, dtype=float) for row in head.weights.values()])
                if head.weights
                else np.zeros((0, len(head.labels)), dtype=float)
            )
            arrays[f"{name}__hashes"] = hashes
            arrays[f"{name}__coefs"] = coefs
            arrays[f"{name}__bias"] = np.asarray(head.bias, dtype=float)
            arrays[f"{name}__labels"] = np.array(head.labels)
            arrays[f"{name}__threshold"] = np.array(head.threshold)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "EdgeClassifierModel":
        """Load a model from an ``.npz`` artifact.

        Raises ``ValueError`` if the artifact's ``feature_spec_version`` does not
        match the current extractor (prevents silent feature-space drift).
        """
        data: Dict[str, NpyArray] = read_npz(path)
        spec = as_scalar_str(data["feature_spec_version"])
        if spec != FEATURE_SPEC_VERSION:
            raise ValueError(
                f"edge-classifier artifact feature_spec_version={spec!r} "
                f"does not match current {FEATURE_SPEC_VERSION!r}; retrain or "
                "invalidate per-project deltas."
            )
        model = cls(
            feature_spec_version=spec,
            model_version=as_scalar_str(data["model_version"]),
            alpha=as_scalar_float(data["alpha"]),
            heads={},
        )
        for name in as_str_list(data["head_names"]):
            hashes = as_int_vector(data[f"{name}__hashes"])
            coefs = as_float_rows(data[f"{name}__coefs"])
            weights: Dict[int, List[float]] = {int(h): coefs[i] for i, h in enumerate(hashes)}
            model.heads[name] = DecisionHead(
                decision_type=name,
                labels=as_str_list(data[f"{name}__labels"]),
                weights=weights,
                bias=as_float_vector(data[f"{name}__bias"]),
                threshold=as_scalar_float(data[f"{name}__threshold"]),
            )
        return model
