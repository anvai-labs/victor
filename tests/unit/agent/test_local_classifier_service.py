# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""FEP-0012 Phase 5: LocalClassifierDecisionService — classify-with-artifact,
defer-without-artifact, heuristic fast-path, unknown-type defer, unconfident
defer, protocol conformance. Also the DecisionBackend enum config."""

import pytest

pytest.importorskip("sklearn")  # trainer is dev-only

from victor.agent.decisions.schemas import (  # noqa: E402
    DecisionType,
    TaskCategoryType,
    TaskTypeDecision,
)
from victor.agent.services.decision_backend import DecisionBackend  # noqa: E402
from victor.agent.services.local_classifier_service import (  # noqa: E402
    LocalClassifierDecisionService,
)
from victor.agent.services.protocols.decision_service import (  # noqa: E402
    LLMDecisionServiceProtocol,
)
from victor.ml.trainer import train_model  # noqa: E402

# Labels are valid TaskCategoryType values so the mapper resolves them.
_SAMPLES = {
    "task_type_classification": [
        ("run the test suite", "action"),
        ("deploy the application", "action"),
        ("build the project", "action"),
        ("explain the architecture", "analysis"),
        ("review the module design", "analysis"),
        ("describe how routing works", "analysis"),
        ("find the config file", "search"),
        ("locate the failing test", "search"),
        ("grep for the symbol", "search"),
    ]
    * 4
}


@pytest.fixture
def trained_artifact(tmp_path):
    model = train_model(_SAMPLES, model_version="test", threshold=0.5)
    path = tmp_path / "edge.npz"
    model.save(str(path))
    return str(path)


def test_protocol_conformance():
    svc = LocalClassifierDecisionService(model=None)
    assert isinstance(svc, LLMDecisionServiceProtocol)


def test_classify_with_artifact(trained_artifact):
    svc = LocalClassifierDecisionService.from_artifact(trained_artifact)
    assert svc.is_healthy()
    r = svc.decide_sync(
        DecisionType.TASK_TYPE_CLASSIFICATION,
        {"message_excerpt": "explain how the routing works"},
        heuristic_result=None,
        heuristic_confidence=0.0,
    )
    assert r.source == "local_classifier"
    assert isinstance(r.result, TaskTypeDecision)
    assert r.result.task_type == TaskCategoryType.ANALYSIS
    assert r.confidence >= 0.5


def test_defer_without_artifact():
    svc = LocalClassifierDecisionService(model=None)
    assert not svc.is_healthy()
    r = svc.decide_sync(
        DecisionType.TASK_TYPE_CLASSIFICATION,
        {"message_excerpt": "explain routing"},
        heuristic_result="fallback",
        heuristic_confidence=0.1,
    )
    assert r.source == "heuristic"
    assert r.result == "fallback"


def test_heuristic_fast_path(trained_artifact):
    svc = LocalClassifierDecisionService.from_artifact(trained_artifact)
    r = svc.decide_sync(
        DecisionType.TASK_TYPE_CLASSIFICATION,
        {"message_excerpt": "explain routing"},
        heuristic_result="confident-guess",
        heuristic_confidence=0.9,  # >= default threshold 0.6 -> fast path
    )
    assert r.source == "heuristic"
    assert r.result == "confident-guess"


def test_unknown_decision_type_defers(trained_artifact):
    svc = LocalClassifierDecisionService.from_artifact(trained_artifact)
    r = svc.decide_sync(
        DecisionType.COMPACTION,  # no head / no mapper -> defer
        {"messages": 10},
        heuristic_result="heuristic-compaction",
        heuristic_confidence=0.2,
    )
    assert r.source == "heuristic"


def test_unconfident_defers(trained_artifact):
    # Force every prediction below the gate.
    svc = LocalClassifierDecisionService.from_artifact(trained_artifact)
    svc._model.heads["task_type_classification"].threshold = 0.999
    r = svc.decide_sync(
        DecisionType.TASK_TYPE_CLASSIFICATION,
        {"message_excerpt": "explain routing"},
        heuristic_result="fallback",
        heuristic_confidence=0.0,
    )
    assert r.source == "heuristic"


def test_decision_backend_enum_and_setting():
    from victor.config.decision_settings import DecisionServiceSettings

    assert DecisionBackend.parse("edge") == DecisionBackend.EDGE
    assert DecisionBackend.parse("local_classifier") == DecisionBackend.LOCAL_CLASSIFIER
    assert DecisionBackend.parse("garbage") == DecisionBackend.AUTO  # safe default
    assert DecisionServiceSettings().decision_backend == DecisionBackend.AUTO
