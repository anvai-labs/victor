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

"""ADR-0047 D10a step 2 — the httpx OpenAI-compat family routes usage parsing
through ``victor.providers.usage_parsing.parse_usage_dict`` (wave-1 adoption;
replaced the module-local ``_augment_cache_split`` helper).

Behavior pinned: ``prompt_tokens`` stays the FULL count (window/budget logic
depends on it), the cache split is populated when the sandhi binding resolves it,
and the historical 3-key dict is preserved verbatim without the binding.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

import victor.providers.usage_parsing as up
from victor.providers.httpx_openai_compat import HttpxOpenAICompatProvider


class _CompatProvider(HttpxOpenAICompatProvider):
    @property
    def name(self) -> str:
        return "test-compat"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True


def fake_sg(returns: Dict[str, int]) -> SimpleNamespace:
    def parse_usage(slug: str, payload_json: str) -> Dict[str, int]:
        return dict(returns)

    return SimpleNamespace(parse_usage=parse_usage)


def _make_provider() -> _CompatProvider:
    return _CompatProvider(api_key="sk-test-fake", base_url="https://example.invalid/v1")


def _response_payload(usage_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": usage_data,
    }


def _final_chunk_payload(usage_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": usage_data,
    }


USAGE_WITH_CACHE = {
    "prompt_tokens": 100,
    "completion_tokens": 10,
    "total_tokens": 110,
    "prompt_tokens_details": {"cached_tokens": 60},
}

NEUTRAL_WITH_CACHE = {
    "tokens_in": 40,  # fresh-only; 40 + 60 cached = 100 full
    "tokens_out": 10,
    "cache_creation_tokens": 0,
    "cache_read_tokens": 60,
}


class TestParseResponse:
    def test_populates_cache_split_keeping_prompt_full(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL_WITH_CACHE))
        provider = _make_provider()
        parsed = provider._parse_response(_response_payload(USAGE_WITH_CACHE), "test-model")
        # prompt_tokens stays the FULL count — window/budget logic depends on it.
        assert parsed.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "cache_read_input_tokens": 60,
        }
        # OpenAI has no separate cache-creation billing → not added.
        assert "cache_creation_input_tokens" not in parsed.usage

    def test_noop_when_no_cache(self, monkeypatch):
        monkeypatch.setattr(
            up,
            "_sg",
            fake_sg(
                {
                    "tokens_in": 50,
                    "tokens_out": 5,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 0,
                }
            ),
        )
        provider = _make_provider()
        usage_data = {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}
        parsed = provider._parse_response(_response_payload(usage_data), "test-model")
        assert parsed.usage == {
            "prompt_tokens": 50,
            "completion_tokens": 5,
            "total_tokens": 55,
        }

    def test_fallback_without_binding(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", None)
        provider = _make_provider()
        parsed = provider._parse_response(_response_payload(USAGE_WITH_CACHE), "test-model")
        # Historical 3-key dict, cache split dropped (native fallback).
        assert parsed.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
        }

    def test_no_usage_stays_none(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL_WITH_CACHE))
        provider = _make_provider()
        parsed = provider._parse_response(
            {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]},
            "test-model",
        )
        assert parsed.usage is None


class TestParseStreamChunk:
    def test_final_chunk_populates_cache_split(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", fake_sg(NEUTRAL_WITH_CACHE))
        provider = _make_provider()
        chunk = provider._parse_stream_chunk(_final_chunk_payload(USAGE_WITH_CACHE), [])
        assert chunk is not None and chunk.is_final
        assert chunk.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "cache_read_input_tokens": 60,
        }

    def test_final_chunk_fallback_without_binding(self, monkeypatch):
        monkeypatch.setattr(up, "_sg", None)
        provider = _make_provider()
        chunk = provider._parse_stream_chunk(_final_chunk_payload(USAGE_WITH_CACHE), [])
        assert chunk is not None and chunk.is_final
        assert chunk.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
        }


class TestBinding:
    def test_populates_cache_split_with_real_binding(self):
        pytest.importorskip("sandhi_gateway")
        provider = _make_provider()
        parsed = provider._parse_response(_response_payload(USAGE_WITH_CACHE), "test-model")
        assert parsed.usage is not None
        assert parsed.usage["prompt_tokens"] == 100
        assert parsed.usage["cache_read_input_tokens"] == 60
        assert "cache_creation_input_tokens" not in parsed.usage

    def test_noop_when_no_cache_with_real_binding(self):
        pytest.importorskip("sandhi_gateway")
        provider = _make_provider()
        usage_data = {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}
        parsed = provider._parse_response(_response_payload(usage_data), "test-model")
        assert parsed.usage == {
            "prompt_tokens": 50,
            "completion_tokens": 5,
            "total_tokens": 55,
        }
