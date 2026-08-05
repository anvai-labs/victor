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

"""Tests for config-driven metrics capabilities."""

import pytest

from victor.config.metrics_capabilities import (
    ProviderMetricsCapabilities,
    get_metrics_capabilities,
    get_supported_providers,
    get_provider_models_with_pricing,
    clear_cache,
)


class TestProviderMetricsCapabilities:
    """Tests for ProviderMetricsCapabilities dataclass."""

    def test_default_values(self):
        """Test default values for capabilities."""
        caps = ProviderMetricsCapabilities(provider="test", model="test-model")
        assert caps.supports_prompt_tokens is False
        assert caps.supports_completion_tokens is False
        assert caps.supports_cache_tokens is False
        assert caps.cost_enabled is False
        assert caps.fallback_enabled is True
        assert caps.chars_per_token == 4

    def test_cost_calculation_basic(self):
        """Test basic cost calculation."""
        caps = ProviderMetricsCapabilities(
            provider="test",
            model="test-model",
            cost_enabled=True,
            input_cost_per_mtok=3.0,
            output_cost_per_mtok=15.0,
        )
        costs = caps.calculate_cost(1000, 500)
        assert costs["input_cost"] == pytest.approx(0.003, rel=1e-6)
        assert costs["output_cost"] == pytest.approx(0.0075, rel=1e-6)
        assert costs["cache_cost"] == pytest.approx(0.0, rel=1e-6)
        assert costs["total_cost"] == pytest.approx(0.0105, rel=1e-6)

    def test_cost_calculation_with_cache(self):
        """Test cost calculation with cache tokens."""
        caps = ProviderMetricsCapabilities(
            provider="anthropic",
            model="claude-3-5-sonnet",
            cost_enabled=True,
            input_cost_per_mtok=3.0,
            output_cost_per_mtok=15.0,
            cache_read_cost_per_mtok=0.30,
            cache_write_cost_per_mtok=3.75,
        )
        costs = caps.calculate_cost(1000, 500, cache_read_tokens=200, cache_write_tokens=100)
        assert costs["input_cost"] == pytest.approx(0.003, rel=1e-6)
        assert costs["output_cost"] == pytest.approx(0.0075, rel=1e-6)
        # cache_cost = (200/1M * 0.30) + (100/1M * 3.75) = 0.00006 + 0.000375 = 0.000435
        assert costs["cache_cost"] == pytest.approx(0.000435, rel=1e-6)
        assert costs["total_cost"] == pytest.approx(0.010935, rel=1e-6)

    def test_cost_calculation_disabled(self):
        """Test that cost calculation returns zeros when disabled."""
        caps = ProviderMetricsCapabilities(
            provider="ollama",
            model="llama3",
            cost_enabled=False,
        )
        costs = caps.calculate_cost(10000, 5000)
        assert costs["input_cost"] == 0.0
        assert costs["output_cost"] == 0.0
        assert costs["total_cost"] == 0.0


class TestGetMetricsCapabilities:
    """Tests for get_metrics_capabilities function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_anthropic_claude_sonnet(self):
        """Test Anthropic Claude 3.5 Sonnet capabilities."""
        caps = get_metrics_capabilities("anthropic", "claude-3-5-sonnet-20241022")
        assert caps.provider == "anthropic"
        assert caps.model == "claude-3-5-sonnet-20241022"
        assert caps.supports_prompt_tokens is True
        assert caps.supports_completion_tokens is True
        assert caps.supports_cache_tokens is True
        assert caps.cost_enabled is True
        assert caps.input_cost_per_mtok == 3.0
        assert caps.output_cost_per_mtok == 15.0

    def test_anthropic_claude_haiku(self):
        """Test Anthropic Claude Haiku capabilities."""
        caps = get_metrics_capabilities("anthropic", "claude-3-5-haiku-20241022")
        assert caps.supports_prompt_tokens is True
        assert caps.cost_enabled is True
        assert caps.input_cost_per_mtok == 0.8
        assert caps.output_cost_per_mtok == 4.0

    def test_openai_gpt4o(self):
        """Test OpenAI GPT-4o capabilities."""
        caps = get_metrics_capabilities("openai", "gpt-4o")
        assert caps.provider == "openai"
        assert caps.supports_prompt_tokens is True
        assert caps.supports_cache_tokens is False  # OpenAI doesn't support cache tokens
        assert caps.cost_enabled is True
        assert caps.input_cost_per_mtok == 2.5
        assert caps.output_cost_per_mtok == 10.0

    def test_openai_gpt4o_mini(self):
        """Test OpenAI GPT-4o-mini capabilities."""
        caps = get_metrics_capabilities("openai", "gpt-4o-mini")
        assert caps.cost_enabled is True
        assert caps.input_cost_per_mtok == 0.15
        assert caps.output_cost_per_mtok == 0.60

    def test_ollama_free(self):
        """Test Ollama (free provider) capabilities."""
        caps = get_metrics_capabilities("ollama", "llama3")
        assert caps.supports_prompt_tokens is False
        assert caps.cost_enabled is False
        assert caps.fallback_enabled is True

    def test_unknown_provider(self):
        """Test unknown provider gets defaults."""
        caps = get_metrics_capabilities("unknown_provider", "unknown_model")
        assert caps.provider == "unknown_provider"
        assert caps.model == "unknown_model"
        assert caps.supports_prompt_tokens is False
        assert caps.cost_enabled is False
        assert caps.fallback_enabled is True

    def test_caching(self):
        """Test that config loading is cached (same values returned)."""
        caps1 = get_metrics_capabilities("anthropic", "claude-3-5-sonnet-20241022")
        caps2 = get_metrics_capabilities("anthropic", "claude-3-5-sonnet-20241022")
        # Config is cached, so values should be equal
        assert caps1 == caps2
        assert caps1.input_cost_per_mtok == caps2.input_cost_per_mtok
        assert caps1.output_cost_per_mtok == caps2.output_cost_per_mtok

    def test_different_models_not_cached_together(self):
        """Test that different models have separate cache entries."""
        caps1 = get_metrics_capabilities("anthropic", "claude-3-5-sonnet-20241022")
        caps2 = get_metrics_capabilities("anthropic", "claude-3-5-haiku-20241022")
        assert caps1 is not caps2
        assert caps1.input_cost_per_mtok != caps2.input_cost_per_mtok


class TestSupportedProviders:
    """Tests for provider listing functions."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_get_supported_providers(self):
        """Test getting list of supported providers."""
        providers = get_supported_providers()
        assert isinstance(providers, list)
        assert "anthropic" in providers
        assert "openai" in providers
        assert "ollama" in providers

    def test_get_provider_models_with_pricing_anthropic(self):
        """Test getting models with pricing for Anthropic."""
        models = get_provider_models_with_pricing("anthropic")
        assert isinstance(models, list)
        assert len(models) > 0
        # Check for known model patterns
        assert any("claude" in m for m in models)

    def test_get_provider_models_with_pricing_ollama(self):
        """Test that free providers have no pricing models."""
        models = get_provider_models_with_pricing("ollama")
        assert isinstance(models, list)
        assert len(models) == 0  # Ollama is free, no pricing


class TestModelPricingResolution:
    """Tests for model pricing resolution through public API."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_exact_model_match(self):
        """Test exact model name matching."""
        caps = get_metrics_capabilities("anthropic", "claude-3-5-sonnet-20241022")
        assert caps.cost_enabled is True
        assert caps.input_cost_per_mtok == 3.0
        assert caps.output_cost_per_mtok == 15.0

    def test_pattern_model_match(self):
        """Test pattern-based model matching."""
        # Should fall back to pattern matching
        caps = get_metrics_capabilities("anthropic", "claude-3-5-sonnet-newer-version")
        # Pattern matching should find pricing
        assert caps.cost_enabled is True

    def test_no_pricing_for_free_provider(self):
        """Test that free providers have no cost."""
        caps = get_metrics_capabilities("ollama", "llama3")
        assert caps.cost_enabled is False
        assert caps.input_cost_per_mtok == 0.0

    def test_unknown_model_uses_provider_default(self):
        """Test unknown model uses provider capabilities but no pricing."""
        caps = get_metrics_capabilities("anthropic", "completely-unknown-model-xyz")
        # Should still have provider capabilities
        assert caps.supports_prompt_tokens is True
        # But pricing may be 0 if no pattern matches
        assert caps.cost_enabled is True


class TestCostCalculationIntegration:
    """Integration tests for full cost calculation flow."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_full_anthropic_request(self):
        """Test full request cost calculation for Anthropic."""
        caps = get_metrics_capabilities("anthropic", "claude-3-5-sonnet-20241022")

        # Simulate a typical request: 1000 prompt, 500 completion, 200 cache read
        costs = caps.calculate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            cache_read_tokens=200,
            cache_write_tokens=0,
        )

        # Verify costs are reasonable
        assert 0 < costs["total_cost"] < 1.0  # Should be fraction of a dollar
        assert costs["input_cost"] > 0
        assert costs["output_cost"] > 0
        assert costs["cache_cost"] >= 0

    def test_full_openai_request(self):
        """Test full request cost calculation for OpenAI."""
        caps = get_metrics_capabilities("openai", "gpt-4o")

        costs = caps.calculate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
        )

        # OpenAI doesn't have cache tokens
        assert costs["cache_cost"] == 0
        assert costs["total_cost"] == costs["input_cost"] + costs["output_cost"]

    def test_large_token_counts(self):
        """Test cost calculation with large token counts."""
        caps = get_metrics_capabilities("anthropic", "claude-3-5-sonnet-20241022")

        # 100k tokens input, 50k output
        costs = caps.calculate_cost(
            prompt_tokens=100000,
            completion_tokens=50000,
        )

        # Should be around $1.05 (100k * $3/M + 50k * $15/M = $0.30 + $0.75)
        assert 1.0 < costs["total_cost"] < 1.2
