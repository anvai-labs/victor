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

"""Tests for RuntimeBuilders store context-limit resolution.

The store used to read a flat ``settings.max_context_tokens`` that never
existed (nested-only since the context group move), always falling back to
100000 — now resolved via the nested user-set value or the unified window
resolver.
"""

from types import SimpleNamespace

from victor.agent.factory.runtime_builders import RuntimeBuildersMixin
from victor.config.context_settings import ContextSettings


def _builders(context=None, response_reserve=4096):
    builders = RuntimeBuildersMixin.__new__(RuntimeBuildersMixin)
    builders.settings = SimpleNamespace(
        context=context or ContextSettings(),
        response_token_reserve=response_reserve,
    )
    return builders


class TestResolveStoreContextLimits:
    def test_user_set_nested_value_wins(self):
        builders = _builders(ContextSettings(max_context_tokens=262144))
        tokens, reserve = builders._resolve_store_context_limits("openai")
        assert tokens == 262144
        assert reserve == 4096

    def test_nested_default_falls_through_to_resolver(self):
        """The 100000 nested default must NOT satisfy the store by itself."""
        builders = _builders(ContextSettings())
        tokens, _ = builders._resolve_store_context_limits("anthropic")
        assert tokens == 200000  # provider default from provider_context_limits.yaml

    def test_resolver_fallback_for_unknown_provider(self):
        builders = _builders(ContextSettings())
        tokens, _ = builders._resolve_store_context_limits("no-such-provider")
        assert tokens == 128000

    def test_response_reserve_passthrough(self):
        builders = _builders(ContextSettings(max_context_tokens=50000), response_reserve=2048)
        _, reserve = builders._resolve_store_context_limits("openai")
        assert reserve == 2048

    def test_resolver_failure_keeps_historical_fallback(self):
        builders = _builders(ContextSettings())
        builders.settings.context = None  # force the resolver probe path
        tokens, _ = builders._resolve_store_context_limits("openai")
        assert tokens in (128000, 100000)  # resolver or documented fallback
