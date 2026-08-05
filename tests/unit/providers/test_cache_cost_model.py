# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FEP-0011 Phase 1: CacheCostModel is additive and derives from the legacy
booleans by default, so existing providers behave identically."""

from dataclasses import FrozenInstanceError

import pytest

from victor.providers.base import BaseProvider, CacheCostModel


class _StubProvider(BaseProvider):
    """Minimal concrete provider: implements the abstract surface with stubs so
    subclasses only need to override the caching hooks under test."""

    @property
    def name(self) -> str:
        return "stub"

    async def chat(self, *args, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError

    async def stream(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover
        pass


class _NoCachingProvider(_StubProvider):
    """Default provider: no caching declared."""


class _ApiCachingProvider(_StubProvider):
    """Provider that declares API prompt caching via the legacy bool."""

    def supports_prompt_caching(self) -> bool:
        return True


class _KvCachingProvider(_StubProvider):
    def supports_kv_prefix_caching(self) -> bool:
        return True


class _CharacterizedProvider(_StubProvider):
    """Provider that overrides cache_cost_model with real numbers."""

    def cache_cost_model(self) -> CacheCostModel:
        return CacheCostModel(
            supported=True,
            read_discount=0.9,
            write_overhead=1.25,
            ttl_seconds=300,
            min_prefix_tokens=1024,
            max_cache_tokens=90_000,
            prefix_granularity="system_block",
        )


class TestCacheCostModelDefaults:
    def test_defaults(self):
        m = CacheCostModel()
        assert m.supported is False
        assert m.read_discount == 0.0
        assert m.write_overhead == 1.0
        assert m.ttl_seconds == 0.0
        assert m.min_prefix_tokens == 0
        assert m.max_cache_tokens == 0
        assert m.prefix_granularity == "token"

    def test_frozen(self):
        m = CacheCostModel(supported=True)
        with pytest.raises(FrozenInstanceError):
            m.supported = False  # type: ignore[misc]


class TestDerivedFromBooleans:
    def test_default_provider_cache_model_unsupported(self):
        assert _NoCachingProvider().cache_cost_model().supported is False

    def test_api_caching_provider_derives_supported(self):
        assert _ApiCachingProvider().cache_cost_model().supported is True

    def test_kv_caching_provider_kv_model_derives_supported(self):
        assert _KvCachingProvider().kv_cache_cost_model().supported is True
        # API-level still defaults to unsupported for a KV-only provider.
        assert _KvCachingProvider().cache_cost_model().supported is False


class TestOverride:
    def test_characterized_provider_returns_real_numbers(self):
        m = _CharacterizedProvider().cache_cost_model()
        assert m.supported is True
        assert m.read_discount == pytest.approx(0.9)
        assert m.ttl_seconds == 300
        assert m.prefix_granularity == "system_block"


def _bare(cls):
    """Create a provider instance without running ``__init__`` (cache_cost_model
    needs no instance state, so this avoids API-key/config construction)."""
    return object.__new__(cls)


class TestFlagshipProviderCharacterization:
    """FEP-0011 Phase 2: flagship providers override cache_cost_model() with
    documented real numbers (booleans remain the source of truth / derived)."""

    def test_anthropic_characterized(self):
        from victor.providers.anthropic_provider import AnthropicProvider

        m = _bare(AnthropicProvider).cache_cost_model()
        assert m.supported is True
        assert m.read_discount == pytest.approx(0.9)
        assert m.write_overhead == pytest.approx(1.25)
        assert m.ttl_seconds == pytest.approx(300.0)
        assert m.min_prefix_tokens == 1024
        assert m.prefix_granularity == "system_block"
        # The legacy bool still agrees.
        assert AnthropicProvider.supports_prompt_caching(_bare(AnthropicProvider)) is True

    def test_openai_characterized(self):
        from victor.providers.openai_provider import OpenAIProvider

        m = _bare(OpenAIProvider).cache_cost_model()
        assert m.supported is True
        assert m.read_discount == pytest.approx(0.9)
        assert m.min_prefix_tokens == 1024
        assert m.prefix_granularity == "token"

    def test_llamacpp_kv_only(self):
        from victor.providers.llamacpp_provider import LlamaCppProvider

        prov = _bare(LlamaCppProvider)
        # API-level caching is unsupported (bool False → default model unsupported).
        assert prov.cache_cost_model().supported is False
        # KV caching is supported, latency-only (no billing discount).
        kv = prov.kv_cache_cost_model()
        assert kv.supported is True
        assert kv.read_discount == pytest.approx(0.0)


class TestRemainingProviderCharacterization:
    """FEP-0011 Phase 2+: the rest of the providers are characterized with
    numbers drawn from each provider's own caching docstring."""

    def test_bedrock_high_discount(self):
        from victor.providers.bedrock_provider import BedrockProvider

        m = _bare(BedrockProvider).cache_cost_model()
        assert m.read_discount == pytest.approx(0.9)
        assert m.write_overhead == pytest.approx(1.25)
        assert m.min_prefix_tokens == 1024

    def test_deepseek_discount_and_ttl(self):
        from victor.providers.deepseek_provider import DeepSeekProvider

        m = _bare(DeepSeekProvider).cache_cost_model()
        assert m.read_discount == pytest.approx(0.9)
        assert m.ttl_seconds == pytest.approx(3600.0)

    def test_azure_midband_discount(self):
        from victor.providers.azure_openai_provider import AzureOpenAIProvider

        m = _bare(AzureOpenAIProvider).cache_cost_model()
        assert 0.5 < m.read_discount < 0.75 + 1e-6
        assert m.min_prefix_tokens == 1024

    def test_groq_discount_is_conservative_boundary(self):
        # 0.5 discount is NOT < 0.5 → conservative pruning (keep in prefix).
        from victor.agent.prompt_pipeline import detect_cache_economics
        from victor.providers.groq_provider import GroqProvider

        eco = detect_cache_economics(_bare(GroqProvider))
        assert eco.api_discount == pytest.approx(0.5)
        assert eco.pruning_aggressiveness == "conservative"

    def test_ollama_kv_only(self):
        from victor.providers.ollama_provider import OllamaProvider

        prov = _bare(OllamaProvider)
        assert prov.cache_cost_model().supported is False
        kv = prov.kv_cache_cost_model()
        assert kv.supported is True
        assert kv.read_discount == pytest.approx(0.0)

    def test_cerebras_latency_only(self):
        from victor.providers.cerebras_provider import CerebrasProvider

        m = _bare(CerebrasProvider).cache_cost_model()
        # Latency-only cache: no billing discount, but a real TTL.
        assert m.read_discount == pytest.approx(0.0)
        assert m.ttl_seconds == pytest.approx(300.0)
