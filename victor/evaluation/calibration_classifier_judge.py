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

"""Trained-classifier calibration judges (E2 — small independent judge).

One wrapper serves every trained arm: anything that maps the rendered blinded
judge view (``render_judged_content``) to p(complete) becomes a
:class:`CalibrationJudge` and goes through the SAME κ/α gate as every LLM
judge — no special pleading for our own models.

``load_linear_judge`` loads the arm-A artifact (hashing-trick logistic,
trained by ``benchmarks/judge_training/train_linear.py``) with a pure-numpy
scorer — no sklearn needed at inference. The npz stores the versioned feature
spec; a spec mismatch refuses to load rather than silently mis-featurizing.

``load_encoder_judge`` loads the arm-B ModernBERT artifact (a saved
transformers sequence-classification directory). It needs the dev-only
``[judge-training]`` deps (torch, transformers); CPU inference is fine for
the ~100-example gate. Neither loader is a runtime dependency — both are
evaluation-only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from victor.evaluation.calibration_rubric_judge import render_judged_content
from victor.evaluation.judge_calibration_harness import CalibrationJudge, Transcript


def make_classifier_judge(
    score_fn: Callable[[str], float], *, threshold: float = 0.5
) -> CalibrationJudge:
    """Wrap a ``rendered_text -> p(complete)`` scorer as a CalibrationJudge.

    The verdict is BINARIZED at ``threshold`` (default 0.5). This is required,
    not cosmetic: the calibration harness computes Krippendorff α at the
    *nominal* level (see ``judge_calibration.krippendorff_alpha``), so a judge
    that returns a raw probability makes every distinct float its own category
    — ``0.999`` and ``1.0`` count as a disagreement against a ``1.0`` gold, and
    α collapses to a meaningless negative even for a near-perfect judge (this
    silently mis-scored every trained-judge experiment before the fix: a
    ModernBERT judge at 95/96 agreement reported α=−0.266 instead of its true
    0.928). LLM judges already return binary completion verdicts (rubric via
    ``score_mode="complete"``); classifier judges must match that contract.
    """

    def judge(prompt: str, transcript: Transcript, workspace: Path) -> float:
        prob = float(score_fn(render_judged_content(prompt, transcript, workspace)))
        return 1.0 if prob >= threshold else 0.0

    return judge


def load_linear_judge(path: Path) -> CalibrationJudge:
    """Load the arm-A linear artifact and return a CalibrationJudge."""
    import numpy as np

    from victor.ml.features import FEATURE_SPEC_VERSION, extract_features

    data = np.load(path, allow_pickle=False)
    spec = str(data["feature_spec_version"])
    if spec != FEATURE_SPEC_VERSION:
        raise ValueError(
            f"{path}: feature spec {spec!r} != runtime {FEATURE_SPEC_VERSION!r} — "
            "retrain the artifact against the current feature module"
        )
    weights = dict(zip(data["indices"].tolist(), data["coefs"].tolist()))
    bias = float(data["intercept"])

    def score(text: str) -> float:
        z = bias
        for feature_hash, count in extract_features(text).items():
            coef = weights.get(feature_hash)
            if coef is not None:
                z += coef * count
        return 1.0 / (1.0 + math.exp(-z))

    return make_classifier_judge(score)


def load_encoder_judge(path: Path, *, max_length: int = 1024) -> CalibrationJudge:
    """Load the arm-B ModernBERT artifact and return a CalibrationJudge.

    Loads once and scores each rendered view as p(label==1) from the softmax.
    Requires the ``[judge-training]`` extra (torch, transformers).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.eval()

    def score(text: str) -> float:
        enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits[0]
        return float(torch.softmax(logits, dim=-1)[1])

    return make_classifier_judge(score)
