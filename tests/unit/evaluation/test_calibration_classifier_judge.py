"""Tests for the trained-classifier calibration judges (E2 arms A + B)."""

import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from victor.evaluation.calibration_classifier_judge import (
    load_linear_judge,
    make_classifier_judge,
)
from victor.evaluation.judge_calibration_harness import Transcript, TranscriptStep
from victor.ml.features import FEATURE_SPEC_VERSION, extract_features


def _save_artifact(path: Path, weights: dict[int, float], intercept: float, spec: str):
    indices = np.array(sorted(weights), dtype=np.int64)
    np.savez(
        path,
        indices=indices,
        coefs=np.array([weights[i] for i in indices], dtype=np.float64),
        intercept=np.float64(intercept),
        feature_spec_version=spec,
    )


class TestLoadLinearJudge:
    def test_scores_match_logistic_of_features(self, tmp_path: Path):
        # Build an artifact whose weights fire on a known text's features.
        text_transcript = Transcript(
            steps=(TranscriptStep(kind="tool", content="edit workspace"),),
            final_message="Done.",
        )
        # Render once through the wrapper to learn the exact text, then weight
        # one of its features positively.
        seen = {}
        probe = make_classifier_judge(lambda text: (seen.setdefault("text", text), 0.0)[1])
        probe("Create a file", text_transcript, tmp_path)
        features = extract_features(seen["text"])
        some_hash = next(iter(features))
        artifact = tmp_path / "m.npz"
        _save_artifact(artifact, {some_hash: 2.0}, intercept=-1.0, spec=FEATURE_SPEC_VERSION)

        judge = load_linear_judge(artifact)
        expected_z = -1.0 + 2.0 * features[some_hash]
        expected = 1.0 / (1.0 + math.exp(-expected_z))
        assert judge("Create a file", text_transcript, tmp_path) == pytest.approx(expected)

    def test_feature_spec_mismatch_refuses_to_load(self, tmp_path: Path):
        artifact = tmp_path / "m.npz"
        _save_artifact(artifact, {1: 1.0}, intercept=0.0, spec="not-the-spec")
        with pytest.raises(ValueError, match="feature spec"):
            load_linear_judge(artifact)

    def test_wrapper_returns_probability_range(self, tmp_path: Path):
        judge = make_classifier_judge(lambda _text: 0.73)
        score = judge("p", Transcript(final_message="x"), tmp_path)
        assert score == 0.73


class TestLoadEncoderJudge:
    """Arm B loader — torch/transformers are dev-only, so stub them here."""

    def test_encoder_judge_softmax_of_logits(self, monkeypatch, tmp_path: Path):
        # Stub torch: softmax over [0.0, 2.0] → p(label==1) = e^2/(1+e^2).
        torch_stub = types.ModuleType("torch")

        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, *a):
                return False

        torch_stub.no_grad = _NoGrad
        torch_stub.softmax = lambda logits, dim: logits  # identity; we control values below

        class _Logits:
            def __init__(self, vals):
                self._vals = vals

            def __getitem__(self, i):
                return self._vals[i]

        # model(**enc).logits[0] → the two-class logits; softmax stub returns them,
        # and score reads index [1]. Make softmax real via a tiny closure instead.
        import math as _math

        def _softmax(logits, dim):
            a, b = logits._vals
            denom = _math.exp(a) + _math.exp(b)
            return _Logits([_math.exp(a) / denom, _math.exp(b) / denom])

        torch_stub.softmax = _softmax
        monkeypatch.setitem(sys.modules, "torch", torch_stub)

        transformers_stub = types.ModuleType("transformers")
        tok = MagicMock()
        tok.return_value = {"input_ids": [[1, 2, 3]]}
        transformers_stub.AutoTokenizer = MagicMock(from_pretrained=MagicMock(return_value=tok))
        model = MagicMock()
        model.return_value.logits = _Logits([_Logits([0.0, 2.0])])
        transformers_stub.AutoModelForSequenceClassification = MagicMock(
            from_pretrained=MagicMock(return_value=model)
        )
        monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

        from victor.evaluation.calibration_classifier_judge import load_encoder_judge

        judge = load_encoder_judge(tmp_path)
        expected = math.exp(2.0) / (math.exp(0.0) + math.exp(2.0))
        assert judge("p", Transcript(final_message="x"), tmp_path) == pytest.approx(expected)
