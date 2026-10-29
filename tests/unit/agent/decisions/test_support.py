"""Tests for the shared DecisionSupportMixin."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from victor.agent.context_compactor import ContextCompactor
from victor.agent.continuation_strategy import ContinuationStrategy
from victor.agent.decisions.support import DecisionSupportMixin
from victor.agent.error_classifier import ToolErrorClassifier
from victor.agent.task_completion import TaskCompletionDetector
from victor.agent.thinking_detector import ThinkingPatternDetector
from victor.agent.unified_classifier import UnifiedTaskClassifier


class _Host(DecisionSupportMixin):
    def __init__(self, decision_service: Any = None, runtime_intelligence: Any = None) -> None:
        self._decision_service = decision_service
        self._runtime_intelligence = runtime_intelligence


@pytest.mark.parametrize(
    ("decision_service", "runtime_intelligence", "expected"),
    [
        (None, None, False),
        (object(), None, True),
        (None, object(), True),
        (object(), object(), True),
    ],
)
def test_has_decision_support_truth_table(
    decision_service: Any, runtime_intelligence: Any, expected: bool
) -> None:
    host = _Host(decision_service, runtime_intelligence)
    assert host._has_decision_support() is expected


def test_runtime_intelligence_preferred_over_decision_service() -> None:
    runtime = MagicMock()
    runtime.decide_sync.return_value = "runtime"
    service = MagicMock()
    service.decide_sync.return_value = "service"
    host = _Host(service, runtime)

    assert host._decide_sync("type", {"k": 1}) == "runtime"
    runtime.decide_sync.assert_called_once()
    service.decide_sync.assert_not_called()


def test_decision_service_fallback_when_no_runtime() -> None:
    service = MagicMock()
    service.decide_sync.return_value = "service"
    host = _Host(decision_service=service)

    assert host._decide_sync("type", {}) == "service"
    service.decide_sync.assert_called_once()


def test_no_backend_returns_none() -> None:
    assert _Host()._decide_sync("type", {"k": 1}) is None


@pytest.mark.parametrize("backend", ["runtime", "service"])
def test_arguments_forwarded(backend: str) -> None:
    target = MagicMock()
    host = _Host(runtime_intelligence=target) if backend == "runtime" else _Host(target)
    context = {"message": "hi"}

    host._decide_sync("dtype", context, heuristic_result="guess", heuristic_confidence=0.4)

    target.decide_sync.assert_called_once_with(
        "dtype", context, heuristic_result="guess", heuristic_confidence=0.4
    )


def test_default_heuristic_kwargs_forwarded() -> None:
    target = MagicMock()
    _Host(target)._decide_sync("dtype", {})
    target.decide_sync.assert_called_once_with(
        "dtype", {}, heuristic_result=None, heuristic_confidence=0.0
    )


@pytest.mark.parametrize(
    "cls",
    [
        ContextCompactor,
        ContinuationStrategy,
        ToolErrorClassifier,
        TaskCompletionDetector,
        ThinkingPatternDetector,
        UnifiedTaskClassifier,
    ],
)
def test_classifiers_share_mixin_implementation(cls: type) -> None:
    assert issubclass(cls, DecisionSupportMixin)
    assert cls._decide_sync is DecisionSupportMixin._decide_sync
    assert cls._has_decision_support is DecisionSupportMixin._has_decision_support
