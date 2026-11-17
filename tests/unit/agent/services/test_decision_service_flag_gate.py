"""Tests: decision-service call sites honor the ``decision_backend`` enum, not
the legacy ``USE_LLM_DECISION_SERVICE`` flag (FEP-0012 "one knob" alignment).

Locks in the removal of the stale flag gates in
``chat_stream_executor._get_decision_service`` and
``orchestrator._check_tool_necessity_via_edge``. The ``legacy_flag_off`` fixture
forces the flag off (it defaults on via the manager's ``default_enabled``), then
asserts a registered decision service is still consulted — i.e. the call sites
treat registration (the enum) as the single source of truth for "enabled". A
re-introduced gate would fail these tests with the flag forced off.
"""

from unittest.mock import MagicMock

import pytest

from victor.agent.decisions.schemas import DecisionType
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.services.chat_stream_executor import _get_decision_service
from victor.agent.services.protocols.decision_service import (
    DecisionResult,
    LLMDecisionServiceProtocol,
)
from victor.core.container import ServiceContainer
from victor.core.feature_flags import (
    FeatureFlag,
    disable_feature,
    enable_feature,
    is_feature_enabled,
)


@pytest.fixture
def legacy_flag_off():
    """Force USE_LLM_DECISION_SERVICE off for the test, restoring afterward."""
    disable_feature(FeatureFlag.USE_LLM_DECISION_SERVICE)
    assert is_feature_enabled(FeatureFlag.USE_LLM_DECISION_SERVICE) is False
    yield
    enable_feature(FeatureFlag.USE_LLM_DECISION_SERVICE)


def _container_with(service: object) -> ServiceContainer:
    """A container with ``service`` registered as the decision service."""
    container = ServiceContainer()
    container.register_instance(LLMDecisionServiceProtocol, service)
    return container


def test_chat_stream_returns_registered_service_with_flag_off(legacy_flag_off):
    """chat_stream_executor honors a registered service even with the flag off."""
    svc = MagicMock(name="local_classifier")
    orch = MagicMock()
    orch.get_service_container.return_value = _container_with(svc)
    assert _get_decision_service(orch) is svc


def test_orchestrator_consults_registered_service_with_flag_off(legacy_flag_off):
    """Tool-necessity consults a registered service even with the flag off,
    honoring its decision over the raw heuristic."""
    svc = MagicMock(name="local_classifier")
    svc.decide_sync.return_value = DecisionResult(
        decision_type=DecisionType.TOOL_NECESSITY,
        result={"requires_tools": False, "confidence": 0.9},
        source="llm",
        confidence=0.9,
    )
    orch = MagicMock(spec=AgentOrchestrator)
    orch._check_tool_necessity_via_edge = AgentOrchestrator._check_tool_necessity_via_edge.__get__(
        orch
    )
    orch._container = _container_with(svc)

    result = orch._check_tool_necessity_via_edge("what is python", heuristic_conf=0.85)

    # The service was consulted despite the legacy flag being forced off.
    svc.decide_sync.assert_called_once()
    # And its decision (Q&A -> skip tools) is honored over the raw heuristic.
    assert result is True
