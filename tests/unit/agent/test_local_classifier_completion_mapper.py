# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012: the task_completion head predicts reward buckets (pass/partial/fail),
so the service mapper must translate them to is_complete correctly. The old
``_map_bool`` expected ``complete/true/yes`` and was a no-op for the shipped head.
"""

from __future__ import annotations

import numpy as np

from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.local_classifier_service import (
    LocalClassifierDecisionService,
    _map_completion,
)
from victor.ml.model import DecisionHead, EdgeClassifierModel
import victor.ml.model as model_mod
import victor.ml.features as features_mod

# task_completion head labels are reward buckets (alphabetical, as the trainer emits).
HEAD_LABELS = ["fail", "partial", "pass"]


def test_map_completion_reward_buckets():
    assert _map_completion("pass", 0.9).is_complete is True
    assert _map_completion("fail", 0.8).is_complete is False
    assert _map_completion("partial", 0.6) is None  # defer on partial
    assert _map_completion("unknown_label", 0.5) is None  # defer on unknown


def test_map_completion_backward_compat_legacy_labels():
    """A future head trained with complete/incomplete labels still maps correctly."""
    assert _map_completion("complete", 0.9).is_complete is True
    assert _map_completion("incomplete", 0.9).is_complete is False


def _service_with_completion_head():
    # Bias toward "pass"; a feature hash 1 boosts "fail". Featurizer is forced below.
    head = DecisionHead(
        decision_type="task_completion",
        labels=list(HEAD_LABELS),
        weights={1: np.array([-3.0, 0.0, 3.0])},  # hash 1 -> boost pass
        bias=np.array([0.0, 0.0, 0.0]),
        threshold=0.5,
    )
    model = EdgeClassifierModel(heads={"task_completion": head})
    return LocalClassifierDecisionService(model=model)


def test_decide_sync_pass_prediction_is_complete(monkeypatch):
    """An integrated confident 'pass' prediction -> is_complete=True (was always False before)."""
    monkeypatch.setattr(features_mod, "extract_features", lambda txt: {1: 1.0})
    monkeypatch.setattr(model_mod, "extract_features", lambda txt: {1: 1.0})
    svc = _service_with_completion_head()
    res = svc.decide_sync(
        DecisionType.TASK_COMPLETION, {"response_tail": "x"}, heuristic_confidence=0.0
    )
    assert res.source == "local_classifier"
    assert res.result.is_complete is True


def test_decide_sync_fail_prediction_not_complete(monkeypatch):
    monkeypatch.setattr(features_mod, "extract_features", lambda txt: {1: 1.0})
    monkeypatch.setattr(model_mod, "extract_features", lambda txt: {1: 1.0})
    # Flip the head so hash 1 boosts "fail" instead.
    head = DecisionHead(
        decision_type="task_completion",
        labels=list(HEAD_LABELS),
        weights={1: np.array([3.0, 0.0, -3.0])},
        bias=np.array([0.0, 0.0, 0.0]),
        threshold=0.5,
    )
    svc = LocalClassifierDecisionService(model=EdgeClassifierModel(heads={"task_completion": head}))
    res = svc.decide_sync(
        DecisionType.TASK_COMPLETION, {"response_tail": "x"}, heuristic_confidence=0.0
    )
    assert res.result.is_complete is False
