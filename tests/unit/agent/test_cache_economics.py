# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FEP-0011 Phase 3: the prompt assembler consumes the characterized
``cache_cost_model()`` via ``detect_cache_economics()``, exposing the real
numbers (discount/TTL/min-prefix) and a derived ``pruning_aggressiveness``."""

import pytest

from victor.agent.prompt_pipeline import (
    CacheEconomics,
    ContentRouter,
    Placement,
    ProviderTier,
    detect_cache_economics,
    detect_provider_tier,
)
from victor.agent.content_registry import ContentCategory, ContentItem
from victor.providers.base import BaseProvider, CacheCostModel
from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.llamacpp_provider import LlamaCppProvider


def _bare(cls):
    """Instantiate a provider without ``__init__`` (cache economics need no state)."""
    return object.__new__(cls)


class _NoCacheProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "nocache"

    async def chat(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    async def stream(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    async def close(self):  # pragma: no cover
        pass


class _WeakCacheProvider(_NoCacheProvider):
    """API caching with a low discount (e.g. a future 25%-discount provider)."""

    def cache_cost_model(self) -> CacheCostModel:
        return CacheCostModel(supported=True, read_discount=0.25)


def test_none_provider_is_no_cache():
    eco = detect_cache_economics(None)
    assert eco.tier == detect_provider_tier(None)
    assert eco.pruning_aggressiveness == "aggressive"


def test_anthropic_characterized_and_consumed():
    eco = detect_cache_economics(_bare(AnthropicProvider))
    assert eco.tier.value == "api_and_kv"
    assert eco.api_discount == pytest.approx(0.9)
    assert eco.min_prefix_tokens == 1024
    assert eco.api_ttl_seconds == pytest.approx(300.0)
    # High discount → keep more content (cached tokens are nearly free).
    assert eco.pruning_aggressiveness == "conservative"
    # The tier helper agrees with the economics.
    assert detect_provider_tier(_bare(AnthropicProvider)) == eco.tier


def test_openai_characterized_and_consumed():
    eco = detect_cache_economics(_bare(OpenAIProvider))
    assert eco.tier.value == "api_and_kv"
    assert eco.api_discount == pytest.approx(0.9)
    assert eco.pruning_aggressiveness == "conservative"


def test_llamacpp_kv_only_low_discount_is_aggressive():
    eco = detect_cache_economics(_bare(LlamaCppProvider))
    assert eco.tier.value == "kv_only"
    # No API-level discount → prune hard.
    assert eco.api_discount == pytest.approx(0.0)
    assert eco.pruning_aggressiveness == "aggressive"


def test_no_cache_provider():
    eco = detect_cache_economics(_NoCacheProvider())
    assert eco.tier.value == "no_cache"
    assert eco.pruning_aggressiveness == "aggressive"


def test_weak_discount_is_balanced():
    eco = detect_cache_economics(_WeakCacheProvider())
    assert eco.tier.value == "api_and_kv"
    assert eco.api_discount == pytest.approx(0.25)
    assert eco.pruning_aggressiveness == "balanced"


def test_economics_is_frozen():
    eco = detect_cache_economics(_NoCacheProvider())
    assert isinstance(eco, CacheEconomics)
    with pytest.raises(Exception):
        eco.api_discount = 1.0  # type: ignore[misc]


def _item(required: bool = False) -> ContentItem:
    return ContentItem(
        name="x",
        category=ContentCategory.STATIC,
        default_text="content",
        token_estimate=10,
        evolvable=False,
        required=required,
        section_group="group",
    )


class TestContentRouterAggressiveness:
    """FEP-0011 Phase 3+: pruning_aggressiveness now drives placement. A weak
    (low-discount) API cache prunes optional content out of the stable prefix;
    a strong/uncharacterized cache keeps everything (backward compatible)."""

    def test_conservative_keeps_optional_in_prefix(self):
        eco = CacheEconomics(tier=ProviderTier.API_AND_KV, api_discount=0.9)
        router = ContentRouter(ProviderTier.API_AND_KV, economics=eco)
        assert eco.pruning_aggressiveness == "conservative"
        # Optional content still lands in the cached system prompt.
        assert router.route(_item(required=False)) == Placement.SYSTEM_PROMPT

    def test_balanced_prunes_optional_to_user_prefix(self):
        eco = CacheEconomics(tier=ProviderTier.API_AND_KV, api_discount=0.25)
        router = ContentRouter(ProviderTier.API_AND_KV, economics=eco)
        assert eco.pruning_aggressiveness == "balanced"
        # Optional content is demoted out of the cached prefix.
        assert router.route(_item(required=False)) == Placement.USER_PREFIX
        # Required content stays in the system prompt.
        assert router.route(_item(required=True)) == Placement.SYSTEM_PROMPT

    def test_balanced_keeps_edge_content_in_prefix(self):
        eco = CacheEconomics(tier=ProviderTier.API_AND_KV, api_discount=0.3)
        router = ContentRouter(
            ProviderTier.API_AND_KV, edge_sections={"GROUNDING_RULES"}, economics=eco
        )
        edge_item = ContentItem(
            name="gr",
            category=ContentCategory.STATIC,
            default_text="rules",
            token_estimate=10,
            evolvable=False,
            required=False,
            section_group="GROUNDING_RULES",
        )
        assert router.route(edge_item) == Placement.SYSTEM_PROMPT

    def test_no_economics_keeps_default_behavior(self):
        # Backward compat: ContentRouter built without economics keeps everything.
        router = ContentRouter(ProviderTier.API_AND_KV)
        assert router.route(_item(required=False)) == Placement.SYSTEM_PROMPT
