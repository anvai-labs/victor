# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012 Phase 6: LocalClassifierDecisionService blends the per-project delta.

The service's read side loads ``local_classifier_delta`` (cached, gated) and
passes it into ``EdgeClassifierModel.predict(delta=...)`` so a project's learned
overlay can change the served label. These tests use a tiny synthetic head and
monkeypatch the featurizer for determinism (independent of the shipped artifact).
"""

from __future__ import annotations

import numpy as np
import pytest

from victor.agent.decisions import local_delta as ld
from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.local_classifier_service import (
    LocalClassifierDecisionService,
)
from victor.core.database import get_project_database
from victor.core.schema import Tables
from victor.ml.model import DecisionHead, EdgeClassifierModel
import victor.ml.model as model_mod
import victor.ml.features as features_mod

LABELS = ["no", "yes"]


def _service_with_head() -> LocalClassifierDecisionService:
    # bias favors "no"; a weight on hash 1 favors "yes".
    head = DecisionHead(
        decision_type="tool_necessity",
        labels=list(LABELS),
        weights={1: np.array([-2.0, 2.0])},
        bias=np.array([1.0, -1.0]),
        threshold=0.5,
    )
    model = EdgeClassifierModel(heads={"tool_necessity": head}, alpha=0.3)
    return LocalClassifierDecisionService(model=model)


def _insert_delta(decision_type: str, feature_hash: int, label: str, weight: float) -> None:
    get_project_database().execute(
        f"INSERT INTO {Tables.LOCAL_CLASSIFIER_DELTA}"
        " (decision_type, feature_hash, label, weight, feature_spec_version)"
        " VALUES (?,?,?,?,?)",
        (decision_type, feature_hash, label, weight, "1"),
    )


@pytest.fixture(autouse=True)
def _fixed_features(monkeypatch):
    """Force every decision's text to a single known feature hash."""
    monkeypatch.setattr(features_mod, "extract_features", lambda txt: {1: 1.0})
    monkeypatch.setattr(model_mod, "extract_features", lambda txt: {1: 1.0})
    ld.clear_delta_for_tests()
    yield
    ld.clear_delta_for_tests()


def test_decide_without_delta_uses_universal_head():
    svc = _service_with_head()
    res = svc.decide_sync(DecisionType.TOOL_NECESSITY, {"message": "x"}, heuristic_confidence=0.0)
    # scores = bias + W·x = [1-2, -1+2] = [-1, 1] -> "yes".
    assert res.result.requires_tools is True
    assert res.source == "local_classifier"


def test_delta_blends_and_flips_decision():
    svc = _service_with_head()
    # Without delta -> "yes".
    base = svc.decide_sync(DecisionType.TOOL_NECESSITY, {"message": "x"}, heuristic_confidence=0.0)
    assert base.result.requires_tools is True

    # A large delta on hash 1 toward "no" flips the blended logit to "no".
    _insert_delta("tool_necessity", 1, "no", 100.0)
    _insert_delta("tool_necessity", 1, "yes", -100.0)
    svc._delta_cache.clear()  # force reload
    flipped = svc.decide_sync(
        DecisionType.TOOL_NECESSITY, {"message": "x"}, heuristic_confidence=0.0
    )
    assert flipped.result.requires_tools is False


def test_delta_not_consulted_when_disabled(monkeypatch):
    class _Disabled:
        local_learning_enabled = False
        local_learning_lr = 0.1
        local_learning_top_k = 2000
        local_learning_decay = 0.995

    monkeypatch.setattr("victor.config.decision_settings.DecisionServiceSettings", _Disabled)

    svc = _service_with_head()
    _insert_delta("tool_necessity", 1, "no", 100.0)  # would flip if consulted
    _insert_delta("tool_necessity", 1, "yes", -100.0)
    svc._delta_cache.clear()
    res = svc.decide_sync(DecisionType.TOOL_NECESSITY, {"message": "x"}, heuristic_confidence=0.0)
    # Disabled -> universal head wins -> "yes" (delta ignored).
    assert res.result.requires_tools is True


def test_delta_cache_hits_within_ttl(monkeypatch):
    """Within the TTL the project DB is not re-read (sub-ms predict)."""
    svc = _service_with_head()
    calls = {"n": 0}
    real_load = ld.load_delta

    def _counting_load(decision_type, labels):
        calls["n"] += 1
        return real_load(decision_type, labels)

    monkeypatch.setattr(ld, "load_delta", _counting_load)
    # First call loads; subsequent calls within TTL hit the cache.
    for _ in range(5):
        svc.decide_sync(DecisionType.TOOL_NECESSITY, {"message": "x"}, heuristic_confidence=0.0)
    assert calls["n"] == 1
