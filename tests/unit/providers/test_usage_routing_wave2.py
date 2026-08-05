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

"""Wave-2 adapter adoption (ADR-0047 D10a step 2): remaining adapters route usage
parsing through ``victor.providers.usage_parsing.parse_usage_dict``.

Covers Ollama (top-level eval counts, slug ``ollama``) and the OpenAI-shape
adapters that carry their own inline usage builds (slug ``openai``): Azure OpenAI,
OpenRouter, Groq, Together, Fireworks, Moonshot, Cerebras, Qwen, Mistral, vLLM,
and LMStudio. Bedrock is intentionally NOT adopted: victor parses the Converse
API's camelCase ``inputTokens``/``outputTokens`` block, which sandhi's bedrock
parser (Anthropic-on-Bedrock snake_case + Titan invoke shapes) parses to zeros —
a silent mis-meter rather than a clean fallback.

Each adapter gets a routed-path test (fake ``_sg`` returning canned neutral usage
distinguishable from the raw block) and a fallback test (``_sg = None`` -> the
exact historical native dict).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict

import pytest

import victor.providers.usage_parsing as up


def fake_sg(returns: Dict[str, int]) -> SimpleNamespace:
    def parse_usage(slug: str, payload_json: str) -> Dict[str, int]:
        return dict(returns)

    return SimpleNamespace(parse_usage=parse_usage)


# ---------------------------------------------------------------------------
# OpenAI-shape adapters (slug "openai"): dict-based _parse_response(result, model)
# ---------------------------------------------------------------------------

RAW_BLOCK = {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120,
    "prompt_tokens_details": {"cached_tokens": 60},
}

NEUTRAL = {
    "tokens_in": 40,  # fresh-only; 40 + 60 cached = 100 full
    "tokens_out": 20,
    "cache_creation_tokens": 0,
    "cache_read_tokens": 60,
}

ROUTED_USAGE = {
    "prompt_tokens": 100,  # FULL count preserved (40 fresh + 60 cached)
    "completion_tokens": 20,
    "total_tokens": 120,  # raw total preserved
    "cache_read_input_tokens": 60,
}

LEGACY_USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120,
}


def _openai_shape_result(usage: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": dict(usage),
    }


def _azure() -> Any:
    from victor.providers.azure_openai_provider import AzureOpenAIProvider

    return AzureOpenAIProvider(
        api_key="test-key", endpoint="https://unit.openai.azure.example", deployment_name="d"
    )


def _openrouter() -> Any:
    from victor.providers.openrouter_provider import OpenRouterProvider

    return OpenRouterProvider(api_key="test-key")


def _groq() -> Any:
    from victor.providers.groq_provider import GroqProvider

    return GroqProvider(api_key="test-key")


def _together() -> Any:
    from victor.providers.together_provider import TogetherProvider

    return TogetherProvider(api_key="test-key")


def _fireworks() -> Any:
    from victor.providers.fireworks_provider import FireworksProvider

    return FireworksProvider(api_key="test-key")


def _moonshot() -> Any:
    from victor.providers.moonshot_provider import MoonshotProvider

    return MoonshotProvider(api_key="test-key")


def _cerebras() -> Any:
    from victor.providers.cerebras_provider import CerebrasProvider

    return CerebrasProvider(api_key="test-key")


def _mistral() -> Any:
    from victor.providers.mistral_provider import MistralProvider

    return MistralProvider(api_key="test-key")


OPENAI_SHAPE_ADAPTERS: list = [
    pytest.param(_azure, id="azure_openai"),
    pytest.param(_openrouter, id="openrouter"),
    pytest.param(_groq, id="groq"),
    pytest.param(_together, id="together"),
    pytest.param(_fireworks, id="fireworks"),
    pytest.param(_moonshot, id="moonshot"),
    pytest.param(_cerebras, id="cerebras"),
    pytest.param(_mistral, id="mistral"),
]


class TestOpenAiShapeAdapters:
    @pytest.mark.parametrize("factory", OPENAI_SHAPE_ADAPTERS)
    def test_routed_path_gains_cached_split(self, monkeypatch, factory: Callable[[], Any]):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL))
        provider = factory()
        parsed = provider._parse_response(_openai_shape_result(RAW_BLOCK), "test-model")
        assert parsed.usage == ROUTED_USAGE

    @pytest.mark.parametrize("factory", OPENAI_SHAPE_ADAPTERS)
    def test_fallback_without_binding(self, monkeypatch, factory: Callable[[], Any]):
        monkeypatch.setattr(up, "_sg", None)
        provider = factory()
        parsed = provider._parse_response(_openai_shape_result(RAW_BLOCK), "test-model")
        # Exactly the historical 3-key native dict.
        assert parsed.usage == LEGACY_USAGE

    @pytest.mark.parametrize("factory", OPENAI_SHAPE_ADAPTERS)
    def test_no_usage_stays_none_or_zero(self, monkeypatch, factory: Callable[[], Any]):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL))
        provider = factory()
        result = _openai_shape_result(RAW_BLOCK)
        del result["usage"]
        parsed = provider._parse_response(result, "test-model")
        # vLLM historically builds a zeroed dict when usage is absent; others keep None.
        assert parsed.usage is None or parsed.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


class TestGroqTimingAugmentation:
    """Groq's queue_time_ms augmentation must survive on both paths."""

    def _result(self) -> Dict[str, Any]:
        result = _openai_shape_result(RAW_BLOCK)
        result["usage"]["queue_time"] = 0.5
        return result

    def test_routed_keeps_queue_time_ms(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL))
        parsed = _groq()._parse_response(self._result(), "test-model")
        assert parsed.usage == {**ROUTED_USAGE, "queue_time_ms": 500}

    def test_fallback_keeps_queue_time_ms(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", None)
        parsed = _groq()._parse_response(self._result(), "test-model")
        assert parsed.usage == {**LEGACY_USAGE, "queue_time_ms": 500}


class TestCerebrasTimingAugmentation:
    """Cerebras' time_info augmentation must survive on both paths."""

    def _result(self) -> Dict[str, Any]:
        result = _openai_shape_result(RAW_BLOCK)
        result["time_info"] = {"total_time": 1.5}
        return result

    def test_routed_keeps_total_time_ms(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL))
        parsed = _cerebras()._parse_response(self._result(), "test-model")
        assert parsed.usage == {**ROUTED_USAGE, "total_time_ms": 1500}

    def test_fallback_keeps_total_time_ms(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", None)
        parsed = _cerebras()._parse_response(self._result(), "test-model")
        assert parsed.usage == {**LEGACY_USAGE, "total_time_ms": 1500}
