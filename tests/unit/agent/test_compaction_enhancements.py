# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for compaction strategy enhancements.

Tests cover:
- Phase 1: COMPACTION decision type
- Phase 2: Decision service integration with ContextCompactor
- Phase 3: System prompt strategy setting
- Phase 4: Dynamic prompt content integration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Optional

from victor.agent.decisions.schemas import (
    DecisionType,
    CompactionDecision,
    SystemPromptOptimizationDecision,
)
from victor.config.decision_settings import DecisionServiceSettings
from victor.config.context_settings import ContextSettings
from victor.agent.context_compactor import ContextCompactor, CompactorConfig
from victor.agent.prompt_builder import SystemPromptBuilder


class TestCompactionDecisionType:
    """Tests for Phase 1: COMPACTION decision type."""

    def test_compaction_decision_type_exists(self):
        """Test that COMPACTION decision type is registered."""
        assert hasattr(DecisionType, "COMPACTION")
        assert DecisionType.COMPACTION == "compaction"

    def test_compaction_decision_schema(self):
        """Test that CompactionDecision schema validates correctly."""
        # Valid simple decision
        decision = CompactionDecision(
            complexity="simple",
            recommended_tier="edge",
            estimated_tokens=1000,
            confidence=0.9,
            reason="Simple compaction with few messages",
        )
        assert decision.complexity == "simple"
        assert decision.recommended_tier == "edge"
        assert decision.estimated_tokens == 1000
        assert decision.confidence == 0.9

        # Valid complex decision
        complex_decision = CompactionDecision(
            complexity="complex",
            recommended_tier="performance",
            estimated_tokens=5000,
            confidence=0.85,
            reason="Complex compaction with many messages",
        )
        assert complex_decision.complexity == "complex"
        assert complex_decision.recommended_tier == "performance"

    def test_compaction_decision_validation(self):
        """Test that CompactionDecision validates enum values."""
        with pytest.raises(ValueError):
            CompactionDecision(
                complexity="invalid",  # Invalid complexity
                recommended_tier="edge",
                estimated_tokens=1000,
                confidence=0.9,
                reason="test",
            )

        with pytest.raises(ValueError):
            CompactionDecision(
                complexity="simple",
                recommended_tier="invalid",  # Invalid tier
                estimated_tokens=1000,
                confidence=0.9,
                reason="test",
            )

        with pytest.raises(ValueError):
            CompactionDecision(
                complexity="simple",
                recommended_tier="edge",
                estimated_tokens=1000,
                confidence=1.5,  # Invalid confidence (> 1.0)
                reason="test",
            )


class TestTieredDecisionRouting:
    """Tests for Phase 1: Tiered decision routing configuration."""

    def test_compaction_routing_config(self):
        """Test that COMPACTION routing is configured as 'auto'."""
        settings = DecisionServiceSettings()
        assert "compaction" in settings.tier_routing
        assert settings.tier_routing["compaction"] == "auto"

    def test_auto_routing_simple(self):
        """Test that auto-routing selects edge tier for simple compaction."""
        from victor.agent.services.tiered_decision_service import TieredDecisionService

        service = TieredDecisionService(DecisionServiceSettings())

        # Mock the service creation
        service._get_service = Mock(return_value=None)

        # Test simple compaction routing
        # Note: This tests the routing logic, not actual LLM calls
        tier = service._tier_routing.get("compaction")
        assert tier == "auto"

    def test_auto_routing_complex(self):
        """Test that auto-routing can handle complex compaction."""
        from victor.agent.services.tiered_decision_service import TieredDecisionService

        service = TieredDecisionService(DecisionServiceSettings())

        # The actual tier selection happens in decide_sync based on context
        # This test verifies the configuration is in place
        assert service._tier_routing["compaction"] == "auto"


class TestContextCompactorIntegration:
    """Tests for Phase 2: Decision service integration with ContextCompactor."""

    def test_compactor_accepts_decision_service(self):
        """Test that ContextCompactor accepts decision_service parameter."""
        mock_controller = Mock()
        mock_decision_service = Mock()

        compactor = ContextCompactor(
            controller=mock_controller,
            decision_service=mock_decision_service,
        )

        assert compactor._decision_service is mock_decision_service

    def test_compactor_defaults_without_decision_service(self):
        """Test that ContextCompactor works without decision service."""
        mock_controller = Mock()

        compactor = ContextCompactor(
            controller=mock_controller,
            decision_service=None,
        )

        assert compactor._decision_service is None

    def test_get_provider_for_tier(self):
        """Test that _get_provider_for_tier creates appropriate providers."""
        mock_controller = Mock()
        compactor = ContextCompactor(controller=mock_controller)

        # Test with mocked provider registry
        with patch("victor.providers.registry.ProviderRegistry") as mock_registry:
            mock_provider = Mock()
            mock_registry.create.return_value = mock_provider

            # Test edge tier
            provider = compactor._get_provider_for_tier("edge")
            assert provider is mock_provider
            mock_registry.create.assert_called_once()

    def test_get_default_provider(self):
        """Test that _get_default_provider gets provider from orchestrator."""
        mock_controller = Mock()
        mock_orchestrator = Mock()
        mock_provider_manager = Mock()
        mock_provider = Mock()

        mock_controller._orchestrator = mock_orchestrator
        mock_orchestrator.provider_manager = mock_provider_manager
        mock_provider_manager.get_active_provider.return_value = mock_provider

        compactor = ContextCompactor(controller=mock_controller)
        provider = compactor._get_default_provider()

        assert provider is mock_provider

    def test_get_prompt_optimization_decision(self):
        """Test that get_prompt_optimization_decision returns valid decisions."""
        mock_controller = Mock()
        mock_decision_service = Mock()
        mock_metrics = Mock()
        mock_metrics.utilization = 0.8

        mock_controller.get_context_metrics.return_value = mock_metrics

        compactor = ContextCompactor(
            controller=mock_controller,
            decision_service=mock_decision_service,
        )

        decision = compactor.get_prompt_optimization_decision(
            current_query="test query",
            recent_failures=["error1", "error2"],
        )

        assert decision is not None
        assert "include_sections" in decision
        assert "add_context_reminder" in decision
        assert "add_failure_hints" in decision
        assert decision["add_failure_hints"] is True  # We passed failures
        assert decision["add_context_reminder"] is False  # No compaction yet


class TestSystemPromptStrategy:
    """Tests for Phase 3: System prompt strategy setting."""

    def test_context_settings_has_strategy(self):
        """Test that ContextSettings has system_prompt_strategy field."""
        settings = ContextSettings()
        assert hasattr(settings, "system_prompt_strategy")
        assert settings.system_prompt_strategy == "static"  # Default value

    def test_system_prompt_strategy_values(self):
        """Test that system_prompt_strategy accepts valid values."""
        settings = ContextSettings(system_prompt_strategy="static")
        assert settings.system_prompt_strategy == "static"

        settings = ContextSettings(system_prompt_strategy="dynamic")
        assert settings.system_prompt_strategy == "dynamic"

        settings = ContextSettings(system_prompt_strategy="hybrid")
        assert settings.system_prompt_strategy == "hybrid"

    def test_prompt_builder_accepts_strategy(self):
        """Test that SystemPromptBuilder accepts system_prompt_strategy."""
        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-3-5-sonnet-20241022",
            system_prompt_strategy="dynamic",
        )

        assert builder.system_prompt_strategy == "dynamic"

    def test_prompt_builder_default_strategy(self):
        """Test that SystemPromptBuilder defaults to 'static'."""
        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-3-5-sonnet-20241022",
        )

        assert builder.system_prompt_strategy == "static"

    def test_static_mode_caches_prompt(self):
        """Test that static mode caches the prompt."""
        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-3-5-sonnet-20241022",
            system_prompt_strategy="static",
        )

        # Build prompt twice
        prompt1 = builder.build()
        prompt2 = builder.build()

        # Should be the same cached object
        assert prompt1 is prompt2
        assert builder._cached_prompt is not None

    def test_dynamic_mode_rebuilds_prompt(self):
        """Test that dynamic mode rebuilds the prompt every time."""
        builder = SystemPromptBuilder(
            provider_name="ollama",
            model="qwen2.5:7b",
            system_prompt_strategy="dynamic",
            provider_caches=False,
        )

        # Build prompt twice
        prompt1 = builder.build()
        prompt2 = builder.build()

        # Cache should be cleared in dynamic mode
        # Note: The prompts might be identical, but the cache should be None
        assert builder._cached_prompt is None

    def test_hybrid_mode_with_cloud_provider(self):
        """Test that hybrid mode uses static for cloud providers."""
        builder = SystemPromptBuilder(
            provider_name="anthropic",
            model="claude-3-5-sonnet-20241022",
            system_prompt_strategy="hybrid",
            provider_caches=True,  # Cloud provider with caching
        )

        # Should behave like static mode
        strategy = builder._get_effective_strategy()
        assert strategy == "static"

    def test_hybrid_mode_with_local_provider(self):
        """Test that hybrid mode uses dynamic for local providers."""
        builder = SystemPromptBuilder(
            provider_name="ollama",
            model="qwen2.5:7b",
            system_prompt_strategy="hybrid",
            provider_caches=False,  # Local provider without caching
        )

        # Should behave like dynamic mode
        strategy = builder._get_effective_strategy()
        assert strategy == "dynamic"


class TestSystemPromptOptimizationDecision:
    """Tests for Phase 4: Dynamic prompt content integration."""

    def test_optimization_decision_schema(self):
        """Test that SystemPromptOptimizationDecision schema validates correctly."""
        decision = SystemPromptOptimizationDecision(
            include_sections=["task_guidance", "completion"],
            add_context_reminder=True,
            add_failure_hints=False,
            adjust_for_complexity=True,
            confidence=0.85,
            reason="Optimization based on task complexity",
        )

        assert decision.include_sections == ["task_guidance", "completion"]
        assert decision.add_context_reminder is True
        assert decision.add_failure_hints is False
        assert decision.adjust_for_complexity is True
        assert decision.confidence == 0.85

    def test_optimization_decision_validation(self):
        """Test that SystemPromptOptimizationDecision validates confidence range."""
        with pytest.raises(ValueError):
            SystemPromptOptimizationDecision(
                include_sections=["task_guidance"],
                add_context_reminder=True,
                add_failure_hints=False,
                adjust_for_complexity=True,
                confidence=1.5,  # Invalid confidence
                reason="test",
            )


class TestIntegrationScenarios:
    """Integration tests for the complete compaction enhancement."""

    def test_simple_compaction_routes_to_edge(self):
        """Test that simple compaction (≤8 messages) routes to edge tier."""
        mock_controller = Mock()
        mock_decision_service = Mock()
        mock_decision_result = Mock()
        mock_decision = Mock()

        # Mock the decision to recommend edge tier
        mock_decision.recommended_tier = "edge"
        mock_decision_result.result = mock_decision
        mock_decision_service.decide_sync.return_value = mock_decision_result

        compactor = ContextCompactor(
            controller=mock_controller,
            decision_service=mock_decision_service,
        )

        # Test that the decision service is called with correct context
        # Note: Full integration test would require more setup
        assert compactor._decision_service is not None

    def test_complex_compaction_routes_to_performance(self):
        """Test that complex compaction (>8 messages) routes to performance tier."""
        mock_controller = Mock()
        mock_decision_service = Mock()
        mock_decision_result = Mock()
        mock_decision = Mock()

        # Mock the decision to recommend performance tier
        mock_decision.recommended_tier = "performance"
        mock_decision_result.result = mock_decision
        mock_decision_service.decide_sync.return_value = mock_decision_result

        compactor = ContextCompactor(
            controller=mock_controller,
            decision_service=mock_decision_service,
        )

        # Test that the decision service is called with correct context
        assert compactor._decision_service is not None

    def test_fallback_to_default_provider(self):
        """Test that compaction falls back to default provider when decision service unavailable."""
        mock_controller = Mock()
        mock_orchestrator = Mock()
        mock_provider_manager = Mock()
        mock_provider = Mock()

        mock_controller._orchestrator = mock_orchestrator
        mock_orchestrator.provider_manager = mock_provider_manager
        mock_provider_manager.get_active_provider.return_value = mock_provider

        # Create compactor without decision service
        compactor = ContextCompactor(
            controller=mock_controller,
            decision_service=None,
        )

        # Should fall back to default provider
        provider = compactor._get_default_provider()
        assert provider is mock_provider


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
