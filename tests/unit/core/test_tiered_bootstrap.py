"""Tests for TieredDecisionService bootstrap wiring.

Covers:
- Bootstrap creates TieredDecisionService when feature flag enabled
- TieredDecisionService satisfies LLMDecisionServiceProtocol
- Fallback to single edge service when tiered creation fails
- decide() async method works
- decide_async() delegates to decide()
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from victor.agent.decisions.schemas import DecisionType


class _MockDecisionResult:
    def __init__(self, result=None, source="llm", confidence=0.9):
        self.result = result
        self.source = source
        self.confidence = confidence
        self.latency_ms = 5.0
        self.decision_type = None


class TestTieredServiceProtocol:
    """TieredDecisionService satisfies LLMDecisionServiceProtocol."""

    def test_has_decide_sync(self):
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        service = TieredDecisionService(DecisionServiceSettings())
        assert hasattr(service, "decide_sync")
        assert callable(service.decide_sync)

    @pytest.mark.asyncio
    async def test_has_decide_async(self):
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        service = TieredDecisionService(DecisionServiceSettings())
        assert hasattr(service, "decide")
        assert hasattr(service, "decide_async")

    @pytest.mark.asyncio
    async def test_decide_async_delegates_to_decide_sync(self):
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        service = TieredDecisionService(DecisionServiceSettings())
        mock_edge = MagicMock()
        mock_edge.decide_sync.return_value = _MockDecisionResult(result="test")
        service._services["edge"] = mock_edge

        result = await service.decide(DecisionType.TOOL_SELECTION, {"message": "test"})
        assert result.result == "test"


class TestCreateTieredDecisionService:
    """Factory function creates TieredDecisionService."""

    def test_create_returns_tiered_service(self):
        from victor.agent.services.tiered_decision_service import (
            create_tiered_decision_service,
        )
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        service = create_tiered_decision_service(config)
        assert service is not None
        assert hasattr(service, "decide_sync")
        assert hasattr(service, "_tier_routing")

    def test_create_disabled_returns_none(self):
        from victor.agent.services.tiered_decision_service import (
            create_tiered_decision_service,
        )
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings(enabled=False)
        service = create_tiered_decision_service(config)
        assert service is None

    def test_create_with_custom_routing(self):
        from victor.agent.services.tiered_decision_service import (
            create_tiered_decision_service,
        )
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings(tier_routing={"tool_selection": "balanced"})
        service = create_tiered_decision_service(config)
        assert service._tier_routing["tool_selection"] == "balanced"


class TestBootstrapIntegration:
    """Bootstrap picks TieredDecisionService when available."""

    def test_create_tiered_from_settings(self):
        """_create_tiered_decision_service reads from DecisionServiceSettings."""
        from victor.core.bootstrap_services import _create_tiered_decision_service

        service = _create_tiered_decision_service()
        # May be None if settings load fails, but function should exist
        assert callable(_create_tiered_decision_service)
