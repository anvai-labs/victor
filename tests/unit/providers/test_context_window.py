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

"""Tests for BaseProvider.context_window and per-provider lookup tables.

Validates that every provider returns:
  - a positive int for known-model lookups
  - the provider's documented default for unknown-model lookups
  - never crashes on None/empty model
"""

from __future__ import annotations

import pytest

from victor.providers import context_windows as cw
from victor.providers.base import BaseProvider


class TestLookupTable:
    """Direct exercise of the lookup() helper."""

    def test_prefix_match_picks_first_in_table(self):
        table = [("foo-bar", 100), ("foo", 50)]
        assert cw.lookup(table, "foo-bar-baz", default=999) == 100

    def test_no_match_returns_default(self):
        table = [("foo", 50)]
        assert cw.lookup(table, "qux", default=999) == 999

    def test_none_model_returns_default(self):
        table = [("foo", 50)]
        assert cw.lookup(table, None, default=999) == 999

    def test_empty_string_returns_default(self):
        table = [("foo", 50)]
        assert cw.lookup(table, "", default=999) == 999

    def test_exact_match(self):
        table = [("claude-sonnet-4-5", 200_000)]
        assert cw.lookup(table, "claude-sonnet-4-5", default=8192) == 200_000


class TestBaseProvider:
    """Default implementation behavior."""

    def test_default_returns_default_constant(self):
        # BaseProvider has abstract methods (chat/stream/close) so we exercise
        # context_window via the unbound function form.
        assert BaseProvider.context_window(object(), None) == BaseProvider.DEFAULT_CONTEXT_WINDOW
        assert BaseProvider.context_window(object(), "any") == BaseProvider.DEFAULT_CONTEXT_WINDOW

    def test_default_is_8192(self):
        assert BaseProvider.DEFAULT_CONTEXT_WINDOW == 8192


@pytest.mark.parametrize(
    "module_name,class_name,known_model,known_cw,unknown_default",
    [
        (
            "victor.providers.anthropic_provider",
            "AnthropicProvider",
            "claude-sonnet-4-5",
            200_000,
            200_000,
        ),
        (
            "victor.providers.openai_provider",
            "OpenAIProvider",
            "gpt-4o-2024-11-20",
            128_000,
            128_000,
        ),
        (
            "victor.providers.google_provider",
            "GoogleProvider",
            "gemini-1.5-pro-002",
            2_000_000,
            1_000_000,
        ),
        (
            "victor.providers.bedrock_provider",
            "BedrockProvider",
            "claude-3-opus-bedrock",
            200_000,
            200_000,
        ),
        (
            "victor.providers.groq_provider",
            "GroqProvider",
            "llama-3.1-70b-versatile",
            128_000,
            32_768,
        ),
        (
            "victor.providers.deepseek_provider",
            "DeepSeekProvider",
            "deepseek-chat",
            131_072,
            131_072,
        ),
        (
            "victor.providers.ollama_provider",
            "OllamaProvider",
            "qwen2.5-coder:7b",
            32_768,
            8_192,
        ),
        (
            "victor.providers.lmstudio_provider",
            "LMStudioProvider",
            "llama3.1:70b-fancy-quant",
            128_000,
            8_192,
        ),
        ("victor.providers.mlx_provider", "MLXProvider", "qwen2.5:32b", 32_768, 32_768),
        (
            "victor.providers.llamacpp_provider",
            "LlamaCppProvider",
            "phi3-mini.gguf",
            4_096,
            8_192,
        ),
        (
            "victor.providers.vllm_provider",
            "VLLMProvider",
            "llama3.1:8b-vllm",
            128_000,
            32_768,
        ),
        (
            "victor.providers.together_provider",
            "TogetherProvider",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            128_000,
            32_768,
        ),
        (
            "victor.providers.openrouter_provider",
            "OpenRouterProvider",
            "any/model",
            128_000,
            128_000,
        ),  # OpenRouter returns default for all
        ("victor.providers.xai_provider", "XAIProvider", "grok-3", 131_072, 131_072),
        (
            "victor.providers.cerebras_provider",
            "CerebrasProvider",
            "llama3.1-70b",
            128_000,
            128_000,
        ),
        (
            "victor.providers.fireworks_provider",
            "FireworksProvider",
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            128_000,
            32_768,
        ),
        (
            "victor.providers.azure_openai_provider",
            "AzureOpenAIProvider",
            "gpt-4o",
            128_000,
            128_000,
        ),
        (
            "victor.providers.vertex_provider",
            "VertexAIProvider",
            "gemini-2.0-flash-001",
            1_000_000,
            1_000_000,
        ),
    ],
)
def test_provider_context_window(module_name, class_name, known_model, known_cw, unknown_default):
    """Each provider returns a positive int for known and unknown models."""
    import importlib

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    instance = cls.__new__(cls)
    instance._current_model = None

    # Known model returns documented context window
    assert (
        cls.context_window(instance, known_model) == known_cw
    ), f"{class_name}.context_window({known_model!r}) should be {known_cw}"

    # Unknown model returns provider default
    assert cls.context_window(instance, "totally-unknown-model-xyz-9999") == unknown_default

    # None/empty also returns default
    assert cls.context_window(instance, None) == unknown_default
    assert cls.context_window(instance, "") == unknown_default


def test_provider_context_window_uses_current_model_when_none_passed():
    """Providers fall back to self._current_model when model arg is None."""
    from victor.providers.anthropic_provider import AnthropicProvider

    instance = AnthropicProvider.__new__(AnthropicProvider)
    instance._current_model = "claude-3-5-sonnet-20241022"
    assert AnthropicProvider.context_window(instance) == 200_000
