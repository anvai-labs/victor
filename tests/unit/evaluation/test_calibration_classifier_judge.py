"""Tests for the trained-classifier calibration judge (E2 arm A)."""

import math
from pathlib import Path

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
