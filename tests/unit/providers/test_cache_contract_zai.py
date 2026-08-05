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
"""ZAI (z.ai GLM) prompt-caching contract tests.

Encodes the caching contract for the ZAIProvider, grounded in Z.AI's official
context-caching documentation (``docs.z.ai/guides/capabilities/cache.md``):

Contract under test:
1. ZAIProvider.supports_prompt_caching() == True
   Z.AI provides implicit, automatic context caching for GLM-5/4.7/4.6/4.5
   series models. Cached tokens are billed at a discounted rate (~50% of the
   standard price) and reported via ``usage.prompt_tokens_details.cached_tokens``.
2. ZAIProvider.supports_kv_prefix_caching() == True
   The inference engine reuses precomputed KV state for identical leading
   prefixes (system prompt + tools), reducing time-to-first-token.

These flags gate the framework's KV-optimization chain
(``_kv_optimization_enabled`` in ``prompt_builder_runtime.py``), which freezes
the turn-1 tool set and deterministically orders ``tools[]`` so the serialized
prefix hashes identically across turns, maximizing cache hits.
"""

import pytest

from victor.providers.zai_provider import ZAIProvider


@pytest.fixture
def zai_provider():
    """Create ZAIProvider instance for testing (api key resolves from explicit arg)."""
    return ZAIProvider(
        api_key="test-api-key",
        base_url="https://api.z.ai/api/paas/v4/",
        timeout=30,
        max_retries=2,
    )


class TestZAIPromptCachingFlag:
    """Contract: ZAI must declare API-level prompt caching support.

    Before the fix, ZAIProvider inherited supports_prompt_caching()=False from
    BaseProvider (via HttpxOpenAICompatProvider, which does NOT override it),
    so the framework treated ZAI like a non-caching provider and never engaged
    prefix-stable tool selection.
    """

    def test_supports_prompt_caching_returns_true(self, zai_provider):
        """Z.AI offers cached-token billing discounts (implicit context caching)."""
        assert zai_provider.supports_prompt_caching() is True

    def test_supports_kv_prefix_caching_returns_true(self, zai_provider):
        """Z.AI reuses KV cache for stable prompt prefixes (TTFT reduction)."""
        assert zai_provider.supports_kv_prefix_caching() is True

    def test_cache_contract_is_owned_by_the_shared_model_policy(self):
        """The methods resolve to the typed policy shell, never BaseProvider defaults."""
        import inspect

        for method_name in (
            "supports_prompt_caching",
            "supports_kv_prefix_caching",
        ):
            func = dict(inspect.getmembers(ZAIProvider))[method_name]
            owner = func.__qualname__.split(".")[0]
            assert (
                owner == "SandhiOpenAICompatPolicy"
            ), f"{method_name} should resolve to shared model policy, but resolves to {owner}"
