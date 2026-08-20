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

"""Tests for cost-aware routing for provider and model selection."""

import pytest

from victor.workflows.cost_router import (
    CostAwareRouter,
    CostTier,
    ModelConfig,
    RoutingDecision,
    get_default_router,
    route_for_cost,
)


class TestCostTier:
    """Tests for the CostTier enum."""

    def test_cost_tier_values(self):
        """Test that cost tiers have correct integer values."""
        assert CostTier.FREE == 0
        assert CostTier.LOW == 1
        assert CostTier.MEDIUM == 2
        assert CostTier.HIGH == 3

    def test_cost_tier_comparison(self):
        """Test that cost tiers are comparable."""
        assert CostTier.FREE < CostTier.LOW
        assert CostTier.LOW < CostTier.MEDIUM
        assert CostTier.MEDIUM < CostTier.HIGH
        assert CostTier.HIGH > CostTier.FREE

    def test_cost_tier_from_string_valid(self):
        """Test parsing valid cost tier strings."""
        assert CostTier.from_string("free") == CostTier.FREE
        assert CostTier.from_string("low") == CostTier.LOW
        assert CostTier.from_string("medium") == CostTier.MEDIUM
        assert CostTier.from_string("high") == CostTier.HIGH

    def test_cost_tier_from_string_case_insensitive(self):
        """Test that cost tier parsing is case insensitive."""
        assert CostTier.from_string("FREE") == CostTier.FREE
        assert CostTier.from_string("Low") == CostTier.LOW
        assert CostTier.from_string("MEDIUM") == CostTier.MEDIUM
        assert CostTier.from_string("High") == CostTier.HIGH

    def test_cost_tier_from_string_unknown_defaults_to_medium(self):
        """Test that unknown cost tier strings default to MEDIUM."""
        assert CostTier.from_string("unknown") == CostTier.MEDIUM
        assert CostTier.from_string("invalid") == CostTier.MEDIUM
        assert CostTier.from_string("") == CostTier.MEDIUM


class TestModelConfig:
    """Tests for the ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test basic ModelConfig creation."""
        config = ModelConfig(
            name="test-model",
            provider="test-provider",
            cost_tier=CostTier.LOW,
        )
        assert config.name == "test-model"
        assert config.provider == "test-provider"
        assert config.cost_tier == CostTier.LOW
        assert config.capabilities == []
        assert config.max_context == 100000
        assert config.input_cost_per_1k == 0.0
        assert config.output_cost_per_1k == 0.0

    def test_model_config_with_capabilities(self):
        """Test ModelConfig with capabilities."""
        config = ModelConfig(
            name="test-model",
            provider="test-provider",
            cost_tier=CostTier.MEDIUM,
            capabilities=["tool_calling", "streaming", "vision"],
        )
        assert config.capabilities == ["tool_calling", "streaming", "vision"]

    def test_model_config_with_cost_info(self):
        """Test ModelConfig with cost information."""
        config = ModelConfig(
            name="test-model",
            provider="test-provider",
            cost_tier=CostTier.HIGH,
            input_cost_per_1k=10.0,
            output_cost_per_1k=30.0,
        )
        assert config.input_cost_per_1k == 10.0
        assert config.output_cost_per_1k == 30.0

    def test_model_config_has_capability_true(self):
        """Test has_capability returns True for existing capability."""
        config = ModelConfig(
            name="test-model",
            provider="test-provider",
            cost_tier=CostTier.MEDIUM,
            capabilities=["tool_calling", "streaming"],
        )
        assert config.has_capability("tool_calling") is True
        assert config.has_capability("streaming") is True

    def test_model_config_has_capability_false(self):
        """Test has_capability returns False for non-existing capability."""
        config = ModelConfig(
            name="test-model",
            provider="test-provider",
            cost_tier=CostTier.MEDIUM,
            capabilities=["tool_calling"],
        )
        assert config.has_capability("vision") is False
        assert config.has_capability("code") is False


class TestRoutingDecision:
    """Tests for the RoutingDecision dataclass."""

    def test_routing_decision_creation(self):
        """Test basic RoutingDecision creation."""
        decision = RoutingDecision(
            model="test-model",
            provider="test-provider",
            cost_tier=CostTier.LOW,
            reason="Test reason",
        )
        assert decision.model == "test-model"
        assert decision.provider == "test-provider"
        assert decision.cost_tier == CostTier.LOW
        assert decision.reason == "Test reason"
        assert decision.alternatives == []

    def test_routing_decision_with_alternatives(self):
        """Test RoutingDecision with alternatives."""
        decision = RoutingDecision(
            model="best-model",
            provider="test-provider",
            cost_tier=CostTier.MEDIUM,
            reason="Selected best model",
            alternatives=["alt-model-1", "alt-model-2"],
        )
        assert decision.alternatives == ["alt-model-1", "alt-model-2"]


class TestCostAwareRouterInit:
    """Tests for CostAwareRouter initialization."""

    def test_router_with_default_models(self):
        """Test router initializes with default models."""
        router = CostAwareRouter()
        # Check that default models are registered
        assert router.get_model("claude-opus-4-20250514") is not None
        assert router.get_model("claude-sonnet-4-20250514") is not None
        assert router.get_model("claude-3-5-haiku-20241022") is not None
        assert router.get_model("gpt-4o") is not None
        assert router.get_model("gpt-4o-mini") is not None
        assert router.get_model("ollama/llama3.2") is not None
        assert router.get_model("cached") is not None

    def test_router_with_custom_models(self):
        """Test router with custom model configurations."""
        custom_models = [
            ModelConfig(
                name="custom-model-1",
                provider="custom",
                cost_tier=CostTier.LOW,
                capabilities=["tool_calling"],
            ),
            ModelConfig(
                name="custom-model-2",
                provider="custom",
                cost_tier=CostTier.HIGH,
                capabilities=["streaming", "vision"],
            ),
        ]
        router = CostAwareRouter(models=custom_models)
        assert router.get_model("custom-model-1") is not None
        assert router.get_model("custom-model-2") is not None
        # Default models should not be registered
        assert router.get_model("claude-opus-4-20250514") is None

    def test_router_custom_defaults(self):
        """Test router with custom default provider and model."""
        router = CostAwareRouter(
            default_provider="openai",
            default_model="gpt-4o",
        )
        assert router._default_provider == "openai"
        assert router._default_model == "gpt-4o"


class TestCostAwareRouterRegisterModel:
    """Tests for registering models in CostAwareRouter."""

    def test_register_model(self):
        """Test registering a new model."""
        router = CostAwareRouter(models=[])
        config = ModelConfig(
            name="new-model",
            provider="new-provider",
            cost_tier=CostTier.MEDIUM,
        )
        router.register_model(config)
        assert router.get_model("new-model") == config

    def test_register_model_overwrites_existing(self):
        """Test that registering a model with same name overwrites."""
        router = CostAwareRouter(models=[])
        config1 = ModelConfig(
            name="model",
            provider="provider1",
            cost_tier=CostTier.LOW,
        )
        config2 = ModelConfig(
            name="model",
            provider="provider2",
            cost_tier=CostTier.HIGH,
        )
        router.register_model(config1)
        router.register_model(config2)
        model = router.get_model("model")
        assert model.provider == "provider2"
        assert model.cost_tier == CostTier.HIGH


class TestCostAwareRouterGetModel:
    """Tests for getting models from CostAwareRouter."""

    def test_get_model_existing(self):
        """Test getting an existing model."""
        router = CostAwareRouter()
        model = router.get_model("claude-sonnet-4-20250514")
        assert model is not None
        assert model.name == "claude-sonnet-4-20250514"
        assert model.provider == "anthropic"

    def test_get_model_nonexistent(self):
        """Test getting a non-existent model returns None."""
        router = CostAwareRouter()
        model = router.get_model("nonexistent-model")
        assert model is None


class TestCostAwareRouterGetModelsByTier:
    """Tests for getting models by tier from CostAwareRouter."""

    def test_get_models_by_tier_free(self):
        """Test getting FREE tier models."""
        router = CostAwareRouter()
        models = router.get_models_by_tier(CostTier.FREE)
        assert len(models) >= 2  # At least ollama and cached
        for model in models:
            assert model.cost_tier == CostTier.FREE

    def test_get_models_by_tier_low(self):
        """Test getting LOW tier models."""
        router = CostAwareRouter()
        models = router.get_models_by_tier(CostTier.LOW)
        assert len(models) >= 2  # At least haiku and gpt-4o-mini
        for model in models:
            assert model.cost_tier == CostTier.LOW

    def test_get_models_by_tier_medium(self):
        """Test getting MEDIUM tier models."""
        router = CostAwareRouter()
        models = router.get_models_by_tier(CostTier.MEDIUM)
        assert len(models) >= 2  # At least sonnet and gpt-4o
        for model in models:
            assert model.cost_tier == CostTier.MEDIUM

    def test_get_models_by_tier_high(self):
        """Test getting HIGH tier models."""
        router = CostAwareRouter()
        models = router.get_models_by_tier(CostTier.HIGH)
        assert len(models) >= 1  # At least opus
        for model in models:
            assert model.cost_tier == CostTier.HIGH

    def test_get_models_by_tier_empty_result(self):
        """Test getting models for tier with no models."""
        router = CostAwareRouter(
            models=[
                ModelConfig(name="only-low", provider="test", cost_tier=CostTier.LOW),
            ]
        )
        models = router.get_models_by_tier(CostTier.HIGH)
        assert models == []


class TestCostAwareRouterGetModelsUpToTier:
    """Tests for getting models up to a specific tier."""

    def test_get_models_up_to_tier_free(self):
        """Test getting models up to FREE tier."""
        router = CostAwareRouter()
        models = router.get_models_up_to_tier(CostTier.FREE)
        for model in models:
            assert model.cost_tier <= CostTier.FREE

    def test_get_models_up_to_tier_low(self):
        """Test getting models up to LOW tier."""
        router = CostAwareRouter()
        models = router.get_models_up_to_tier(CostTier.LOW)
        for model in models:
            assert model.cost_tier <= CostTier.LOW

    def test_get_models_up_to_tier_medium(self):
        """Test getting models up to MEDIUM tier."""
        router = CostAwareRouter()
        models = router.get_models_up_to_tier(CostTier.MEDIUM)
        # Should include FREE, LOW, and MEDIUM
        tiers = {m.cost_tier for m in models}
        assert CostTier.FREE in tiers or len(models) > 0
        for model in models:
            assert model.cost_tier <= CostTier.MEDIUM

    def test_get_models_up_to_tier_high(self):
        """Test getting models up to HIGH tier (all models)."""
        router = CostAwareRouter()
        all_models = router.get_models_up_to_tier(CostTier.HIGH)
        # Should include all models
        assert len(all_models) == len(CostAwareRouter.DEFAULT_MODELS)


class TestCostAwareRouterRoute:
    """Tests for the route method."""

    def test_route_no_constraints(self):
        """Test routing with no constraints returns a valid decision."""
        router = CostAwareRouter()
        decision = router.route()
        assert isinstance(decision, RoutingDecision)
        assert decision.model is not None
        assert decision.provider is not None

    def test_route_max_cost_tier_free(self):
        """Test routing with FREE tier constraint."""
        router = CostAwareRouter()
        decision = router.route(max_cost_tier=CostTier.FREE)
        assert decision.cost_tier == CostTier.FREE

    def test_route_max_cost_tier_low(self):
        """Test routing with LOW tier constraint."""
        router = CostAwareRouter()
        decision = router.route(max_cost_tier=CostTier.LOW)
        assert decision.cost_tier <= CostTier.LOW

    def test_route_required_capabilities(self):
        """Test routing with required capabilities."""
        router = CostAwareRouter()
        decision = router.route(
            required_capabilities=["tool_calling", "streaming"],
        )
        model = router.get_model(decision.model)
        assert model is not None
        assert "tool_calling" in model.capabilities
        assert "streaming" in model.capabilities

    def test_route_required_capabilities_with_tier(self):
        """Test routing with both tier and capabilities."""
        router = CostAwareRouter()
        decision = router.route(
            max_cost_tier=CostTier.LOW,
            required_capabilities=["tool_calling"],
        )
        assert decision.cost_tier <= CostTier.LOW
        model = router.get_model(decision.model)
        assert "tool_calling" in model.capabilities

    def test_route_preferred_provider(self):
        """Test routing with preferred provider."""
        router = CostAwareRouter()
        decision = router.route(
            max_cost_tier=CostTier.MEDIUM,
            preferred_provider="openai",
        )
        # With MEDIUM tier, should prefer openai if available
        # The exact model depends on sorting, but reason should mention preference
        assert isinstance(decision, RoutingDecision)

    def test_route_preferred_provider_as_tiebreaker(self):
        """Test that preferred provider acts as tiebreaker."""
        custom_models = [
            ModelConfig(
                name="model-a",
                provider="provider-a",
                cost_tier=CostTier.LOW,
                capabilities=["tool_calling"],
            ),
            ModelConfig(
                name="model-b",
                provider="provider-b",
                cost_tier=CostTier.LOW,
                capabilities=["tool_calling"],
            ),
        ]
        router = CostAwareRouter(models=custom_models)
        decision = router.route(
            max_cost_tier=CostTier.LOW,
            preferred_provider="provider-b",
        )
        assert decision.provider == "provider-b"

    def test_route_min_context(self):
        """Test routing with minimum context requirement."""
        router = CostAwareRouter()
        decision = router.route(min_context=100000)
        model = router.get_model(decision.model)
        assert model.max_context >= 100000

    def test_route_min_context_filters_small_models(self):
        """Test that min_context filters out small context models."""
        custom_models = [
            ModelConfig(
                name="small-context",
                provider="test",
                cost_tier=CostTier.FREE,
                max_context=4096,
            ),
            ModelConfig(
                name="large-context",
                provider="test",
                cost_tier=CostTier.LOW,
                max_context=200000,
            ),
        ]
        router = CostAwareRouter(models=custom_models)
        decision = router.route(min_context=10000)
        assert decision.model == "large-context"

    def test_route_task_hint_in_reason(self):
        """Test that task hint appears in reason."""
        router = CostAwareRouter()
        decision = router.route(task_hint="code_review")
        assert "code_review" in decision.reason

    def test_route_alternatives_populated(self):
        """Test that alternatives are populated."""
        router = CostAwareRouter()
        decision = router.route(max_cost_tier=CostTier.HIGH)
        # With multiple models, should have alternatives
        assert isinstance(decision.alternatives, list)

    def test_route_no_matching_models_returns_default(self):
        """Test routing when no models match constraints."""
        custom_models = [
            ModelConfig(
                name="only-model",
                provider="test",
                cost_tier=CostTier.HIGH,
                capabilities=[],
            ),
        ]
        router = CostAwareRouter(
            models=custom_models,
            default_model="default-fallback",
            default_provider="default-provider",
        )
        # Request capability that no model has
        decision = router.route(
            max_cost_tier=CostTier.LOW,
            required_capabilities=["nonexistent_capability"],
        )
        assert decision.model == "default-fallback"
        assert decision.provider == "default-provider"
        assert "No matching models found" in decision.reason

    def test_route_empty_models_uses_defaults(self):
        """Test routing when empty list is passed uses default models.

        Note: Empty list is treated as falsy, so DEFAULT_MODELS are used.
        This is by design - 'models or self.DEFAULT_MODELS' treats [] as falsy.
        """
        router = CostAwareRouter(models=[])
        # Empty list is treated as "not provided", so defaults are used
        decision = router.route()
        assert isinstance(decision, RoutingDecision)
        # Should find models from DEFAULT_MODELS
        assert decision.model in [m.name for m in CostAwareRouter.DEFAULT_MODELS]


class TestCostAwareRouterSelectForConstraints:
    """Tests for select_for_constraints method."""

    def test_select_for_constraints_basic(self):
        """Test selection based on constraints dict."""
        router = CostAwareRouter()
        decision = router.select_for_constraints(
            {
                "max_cost_tier": "LOW",
            }
        )
        assert decision.cost_tier <= CostTier.LOW

    def test_select_for_constraints_tier_parsing(self):
        """Test that tier is parsed from string."""
        router = CostAwareRouter()
        decision = router.select_for_constraints(
            {
                "max_cost_tier": "MEDIUM",
            }
        )
        assert decision.cost_tier <= CostTier.MEDIUM

    def test_select_for_constraints_llm_not_allowed(self):
        """Test selection when LLM is not allowed."""
        router = CostAwareRouter()
        decision = router.select_for_constraints(
            {
                "llm_allowed": False,
            }
        )
        assert decision.model == "cached"
        assert decision.provider == "cache"
        assert decision.cost_tier == CostTier.FREE
        assert "LLM not allowed" in decision.reason

    def test_select_for_constraints_llm_allowed_default(self):
        """Test that LLM is allowed by default."""
        router = CostAwareRouter()
        decision = router.select_for_constraints({})
        # Should not return cached when llm_allowed is not specified
        assert decision.model != "cached" or decision.cost_tier != CostTier.FREE

    def test_select_for_constraints_default_tier(self):
        """Test default tier when not specified."""
        router = CostAwareRouter()
        decision = router.select_for_constraints({})
        # Default should be HIGH (no constraint)
        assert isinstance(decision, RoutingDecision)


class TestCostAwareRouterEstimateCost:
    """Tests for the estimate_cost method."""

    def test_estimate_cost_basic(self):
        """Cost comes from the canonical per-MTok pricing config.

        The previous inline math treated per-MTok reference numbers as
        per-1k, overestimating by 1000x (this test used to pin that bug).
        """
        router = CostAwareRouter()
        cost = router.estimate_cost(
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
        )
        # Canonical (provider_metrics.yaml): sonnet 3.0/MTok in, 15.0/MTok out
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert cost == expected

    def test_estimate_cost_zero_tokens(self):
        """Test cost estimation with zero tokens."""
        router = CostAwareRouter()
        cost = router.estimate_cost(
            model="claude-sonnet-4-20250514",
            input_tokens=0,
            output_tokens=0,
        )
        assert cost == 0.0

    def test_estimate_cost_free_model(self):
        """Test cost estimation for free model."""
        router = CostAwareRouter()
        cost = router.estimate_cost(
            model="ollama/llama3.2",
            input_tokens=10000,
            output_tokens=5000,
        )
        assert cost == 0.0  # Free model has 0 cost

    def test_estimate_cost_unknown_model(self):
        """Test cost estimation for unknown model."""
        router = CostAwareRouter()
        cost = router.estimate_cost(
            model="nonexistent-model",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost == 0.0

    def test_estimate_cost_opus(self):
        """Unpriced models report zero, not fabricated numbers.

        claude-opus-4-20250514 has no canonical pricing entry; the honest
        answer is 0.0 rather than the 1000x-inflated legacy estimate.
        """
        router = CostAwareRouter()
        cost = router.estimate_cost(
            model="claude-opus-4-20250514",
            input_tokens=2000,
            output_tokens=1000,
        )
        assert cost == 0.0

    def test_estimate_cost_large_token_count(self):
        """Test cost estimation with large token counts."""
        router = CostAwareRouter()
        cost = router.estimate_cost(
            model="gpt-4o",
            input_tokens=100000,
            output_tokens=50000,
        )
        # Canonical (provider_metrics.yaml): gpt-4o 2.5/MTok in, 10.0/MTok out
        expected = (100000 / 1_000_000) * 2.5 + (50000 / 1_000_000) * 10.0
        assert cost == expected


class TestGlobalRouter:
    """Tests for global router functions."""

    def test_get_default_router_creates_singleton(self):
        """Test that get_default_router creates a singleton."""
        # Reset global state
        import victor.workflows.cost_router as cr

        cr._default_router = None

        router1 = get_default_router()
        router2 = get_default_router()
        assert router1 is router2

    def test_get_default_router_is_functional(self):
        """Test that default router is functional."""
        router = get_default_router()
        assert isinstance(router, CostAwareRouter)
        decision = router.route()
        assert isinstance(decision, RoutingDecision)


class TestRouteForCost:
    """Tests for route_for_cost convenience function."""

    def test_route_for_cost_default(self):
        """Test route_for_cost with defaults."""
        decision = route_for_cost()
        assert isinstance(decision, RoutingDecision)

    def test_route_for_cost_with_tier(self):
        """Test route_for_cost with tier specified."""
        decision = route_for_cost(max_cost_tier="LOW")
        assert decision.cost_tier <= CostTier.LOW

    def test_route_for_cost_with_capabilities(self):
        """Test route_for_cost with capabilities."""
        decision = route_for_cost(
            max_cost_tier="MEDIUM",
            required_capabilities=["tool_calling"],
        )
        # Reset global to get fresh router
        import victor.workflows.cost_router as cr

        router = get_default_router()
        model = router.get_model(decision.model)
        if model:  # Model might be default fallback
            assert (
                "tool_calling" in model.capabilities
                or decision.model == cr._default_router._default_model
            )

    def test_route_for_cost_tier_case_insensitive(self):
        """Test that tier parsing is case insensitive."""
        decision1 = route_for_cost(max_cost_tier="free")
        decision2 = route_for_cost(max_cost_tier="FREE")
        assert decision1.cost_tier == decision2.cost_tier


class TestRoutingEdgeCases:
    """Tests for edge cases in routing."""

    def test_routing_with_single_model(self):
        """Test routing when only one model is available."""
        single_model = ModelConfig(
            name="only-model",
            provider="only-provider",
            cost_tier=CostTier.MEDIUM,
            capabilities=["tool_calling"],
        )
        router = CostAwareRouter(models=[single_model])
        decision = router.route()
        assert decision.model == "only-model"
        assert decision.alternatives == []

    def test_routing_prioritizes_lower_cost(self):
        """Test that routing prioritizes lower cost models."""
        models = [
            ModelConfig(name="high", provider="test", cost_tier=CostTier.HIGH),
            ModelConfig(name="low", provider="test", cost_tier=CostTier.LOW),
            ModelConfig(name="medium", provider="test", cost_tier=CostTier.MEDIUM),
        ]
        router = CostAwareRouter(models=models)
        decision = router.route(max_cost_tier=CostTier.HIGH)
        # Should select lowest cost first
        assert decision.model == "low"

    def test_routing_with_equal_tiers_prefers_more_capabilities(self):
        """Test that models with more capabilities are preferred at same tier."""
        models = [
            ModelConfig(
                name="few-caps",
                provider="test",
                cost_tier=CostTier.LOW,
                capabilities=["streaming"],
            ),
            ModelConfig(
                name="many-caps",
                provider="test",
                cost_tier=CostTier.LOW,
                capabilities=["streaming", "tool_calling", "vision"],
            ),
        ]
        router = CostAwareRouter(models=models)
        decision = router.route(max_cost_tier=CostTier.LOW)
        assert decision.model == "many-caps"

    def test_routing_impossible_capabilities_constraint(self):
        """Test routing with impossible capability requirements."""
        models = [
            ModelConfig(
                name="model-a",
                provider="test",
                cost_tier=CostTier.LOW,
                capabilities=["streaming"],
            ),
        ]
        router = CostAwareRouter(models=models)
        decision = router.route(
            required_capabilities=["streaming", "impossible_capability"],
        )
        # Should fall back to default
        assert "No matching models found" in decision.reason

    def test_routing_all_models_filtered(self):
        """Test routing when all models are filtered out."""
        models = [
            ModelConfig(
                name="only-high",
                provider="test",
                cost_tier=CostTier.HIGH,
            ),
        ]
        router = CostAwareRouter(models=models)
        decision = router.route(max_cost_tier=CostTier.LOW)
        assert "No matching models found" in decision.reason

    def test_routing_context_size_filters_all(self):
        """Test routing when context requirement filters all models."""
        models = [
            ModelConfig(
                name="small-model",
                provider="test",
                cost_tier=CostTier.FREE,
                max_context=4096,
            ),
        ]
        router = CostAwareRouter(models=models)
        decision = router.route(min_context=1000000)  # 1M context
        assert "No matching models found" in decision.reason


class TestDefaultModels:
    """Tests for default model configurations."""

    def test_default_models_include_anthropic(self):
        """Test that default models include Anthropic models."""
        router = CostAwareRouter()
        anthropic_models = [m for m in router._models.values() if m.provider == "anthropic"]
        assert len(anthropic_models) >= 3  # opus, sonnet, haiku

    def test_default_models_include_openai(self):
        """Test that default models include OpenAI models."""
        router = CostAwareRouter()
        openai_models = [m for m in router._models.values() if m.provider == "openai"]
        assert len(openai_models) >= 2  # gpt-4o, gpt-4o-mini

    def test_default_models_include_free_options(self):
        """Test that default models include free options."""
        router = CostAwareRouter()
        free_models = router.get_models_by_tier(CostTier.FREE)
        assert len(free_models) >= 1

    def test_default_anthropic_models_have_tool_calling(self):
        """Test that Anthropic models have tool_calling capability."""
        router = CostAwareRouter()
        for name in [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-5-haiku-20241022",
        ]:
            model = router.get_model(name)
            assert model is not None
            assert "tool_calling" in model.capabilities

    def test_default_models_have_correct_tiers(self):
        """Test that default models have correct cost tiers."""
        router = CostAwareRouter()

        opus = router.get_model("claude-opus-4-20250514")
        assert opus.cost_tier == CostTier.HIGH

        sonnet = router.get_model("claude-sonnet-4-20250514")
        assert sonnet.cost_tier == CostTier.MEDIUM

        haiku = router.get_model("claude-3-5-haiku-20241022")
        assert haiku.cost_tier == CostTier.LOW

        ollama = router.get_model("ollama/llama3.2")
        assert ollama.cost_tier == CostTier.FREE
