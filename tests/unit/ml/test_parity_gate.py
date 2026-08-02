# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012 Phase 7: parity gate (ship a classifier only at parity)."""

import pytest

pytest.importorskip("sklearn")  # validate_outcome_training trains via the [ml] extra

from victor.ml import parity_gate as pg  # noqa: E402
from victor.ml.parity_gate import (  # noqa: E402
    evaluate_on_holdout,
    ship_verdict,
    train_test_split_by_type,
)
from victor.ml.trainer import train_model  # noqa: E402

_PASS = [
    "all parser tests pass and the fix is verified",
    "the patch is complete and tests are green",
    "verified the handler fix works correctly",
    "pytest passes and the issue is resolved",
]
_FAIL = [
    "I think the bug is probably fixed now",
    "maybe this addresses the issue hopefully",
    "attempted a fix but not sure it works",
    "the change might resolve it i believe",
]


def _separable_samples(n_each: int = 20) -> dict:
    return {
        "task_completion": [(t, "pass") for t in _PASS] * n_each
        + [(t, "fail") for t in _FAIL] * n_each
    }


class _FakeModel:
    """Predicts a fixed label — for deterministic metric math."""

    def __init__(self, label):
        self._label = label

    def predict(self, decision_type, text, delta=None):
        return self._label, 0.9


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def test_split_preserves_total_and_types():
    samples = _separable_samples()
    train, holdout = train_test_split_by_type(samples, holdout_frac=0.25, seed=0)
    total = len(samples["task_completion"])
    assert len(train["task_completion"]) + len(holdout["task_completion"]) == total
    assert 0 < len(holdout["task_completion"]) < total


def test_split_is_deterministic():
    s = _separable_samples()
    a = train_test_split_by_type(s, holdout_frac=0.2, seed=7)
    b = train_test_split_by_type(s, holdout_frac=0.2, seed=7)
    assert a == b


# --------------------------------------------------------------------------- #
# evaluate_on_holdout
# --------------------------------------------------------------------------- #
def test_evaluate_metrics_math():
    holdout = {"task_completion": [("a", "pass"), ("b", "pass"), ("c", "fail")]}
    # model predicts "pass" always -> 2/3 correct
    m = evaluate_on_holdout(_FakeModel("pass"), holdout)["task_completion"]
    assert m["n"] == 3
    assert m["coverage"] == 1.0
    assert m["calibrated_accuracy"] == pytest.approx(2 / 3)
    assert m["overall_accuracy"] == pytest.approx(2 / 3)
    assert m["baseline_accuracy"] == pytest.approx(2 / 3)  # majority = pass


class _AbstainingModel:
    """Returns None (below τ) for some inputs."""

    def __init__(self, abstain_on):
        self._abstain_on = abstain_on

    def predict(self, decision_type, text, delta=None):
        if text == self._abstain_on:
            return None, 0.1
        return "pass", 0.9


def test_coverage_counts_abstentions():
    holdout = {"t": [("a", "pass"), ("b", "pass"), ("c", "pass")]}
    m = evaluate_on_holdout(_AbstainingModel("b"), holdout)["t"]
    assert m["coverage"] == pytest.approx(2 / 3)  # abstained on "b"
    assert m["calibrated_accuracy"] == 1.0  # 2 opinions, both right
    assert m["overall_accuracy"] == pytest.approx(2 / 3)  # abstention counts wrong


# --------------------------------------------------------------------------- #
# ship_verdict
# --------------------------------------------------------------------------- #
def test_ship_verdict_ships_when_beats_baseline():
    metrics = {
        "task_completion": {
            "n": 30.0,
            "coverage": 0.9,
            "calibrated_accuracy": 0.8,
            "overall_accuracy": 0.7,
            "baseline_accuracy": 0.6,
            "margin": 0.2,
        }
    }
    v = ship_verdict(metrics, min_samples=20, min_coverage=0.5, min_margin=0.0)
    assert v["ship"] is True
    assert v["per_type"]["task_completion"]["ship"] is True


def test_ship_verdict_rejects_low_margin():
    metrics = {
        "task_completion": {
            "n": 30.0,
            "coverage": 0.9,
            "calibrated_accuracy": 0.55,
            "overall_accuracy": 0.5,
            "baseline_accuracy": 0.6,
            "margin": -0.05,
        }
    }
    assert ship_verdict(metrics)["ship"] is False


def test_ship_verdict_rejects_low_coverage():
    metrics = {
        "task_completion": {
            "n": 30.0,
            "coverage": 0.1,
            "calibrated_accuracy": 1.0,
            "overall_accuracy": 0.1,
            "baseline_accuracy": 0.6,
            "margin": 0.4,
        }
    }
    v = ship_verdict(metrics, min_coverage=0.5)
    assert v["ship"] is False  # abstains too often


def test_ship_verdict_rejects_few_samples():
    metrics = {
        "task_completion": {
            "n": 5.0,
            "coverage": 1.0,
            "calibrated_accuracy": 0.9,
            "overall_accuracy": 0.9,
            "baseline_accuracy": 0.5,
            "margin": 0.4,
        }
    }
    assert ship_verdict(metrics, min_samples=20)["ship"] is False


def test_ship_verdict_empty_no_key_type():
    assert ship_verdict({})["ship"] is False


def test_ship_verdict_key_type_required():
    # A non-key type shipping doesn't make the overall ship.
    metrics = {
        "stage_detection": {
            "n": 30.0,
            "coverage": 0.9,
            "calibrated_accuracy": 0.9,
            "overall_accuracy": 0.9,
            "baseline_accuracy": 0.5,
            "margin": 0.4,
        }
    }
    v = ship_verdict(metrics, min_samples=20)
    assert v["ship"] is False  # task_completion (key) absent


# --------------------------------------------------------------------------- #
# validate_outcome_training (end-to-end with a monkeypatched outcome source)
# --------------------------------------------------------------------------- #
def test_validate_outcome_training_ships_separable(monkeypatch):
    monkeypatch.setattr(pg, "load_outcome_samples", lambda dt=None: _separable_samples(20))
    v = pg.validate_outcome_training(holdout_frac=0.2, min_samples=5, min_coverage=0.5)
    assert v["ship"] is True
    assert v["per_type"]["task_completion"]["calibrated_accuracy"] >= 0.8


def test_validate_outcome_training_no_data(monkeypatch):
    monkeypatch.setattr(pg, "load_outcome_samples", lambda dt=None: {})
    v = pg.validate_outcome_training()
    assert v["ship"] is False
    assert "no reward-labeled" in v["reason"]


def test_validate_outcome_training_no_diversity(monkeypatch):
    # All one label -> can't train (need ≥2 distinct labels).
    monkeypatch.setattr(
        pg,
        "load_outcome_samples",
        lambda dt=None: {"task_completion": [("only pass text", "pass")] * 40},
    )
    v = pg.validate_outcome_training()
    assert v["ship"] is False
    assert "diversity" in v["reason"]
