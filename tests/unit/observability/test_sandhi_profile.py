"""Direct unit tests for the gateway-profile client/translator (mirror gate).

Covers the pure helpers of victor/observability/sandhi_profile.py — URL-root
normalization, model→provider dialect mapping, payload translation (both
modes + contract violations) and secret masking — complementing the
end-to-end coverage in tests/unit/commands/test_profiles_pull.py.
"""
from __future__ import annotations

import pytest

from victor.observability.sandhi_profile import (
    mask_secret,
    normalize_gateway_root,
    provider_for_model,
    translate_gateway_profile,
)


class TestNormalizeGatewayRoot:
    def test_plain_root_unchanged(self):
        assert normalize_gateway_root("https://gw.example:18788") == "https://gw.example:18788"

    def test_trailing_slash_stripped(self):
        assert normalize_gateway_root("https://gw.example:18788/") == "https://gw.example:18788"

    def test_openai_v1_suffix_stripped(self):
        assert normalize_gateway_root("https://gw.example:18788/v1") == "https://gw.example:18788"

    def test_anthropic_root_untouched(self):
        assert normalize_gateway_root("https://gw.example:18788") == "https://gw.example:18788"

    def test_gemini_v1beta_suffix_stripped(self):
        assert normalize_gateway_root("http://gw:18789/v1beta") == "http://gw:18789"


class TestProviderForModel:
    def test_claude_maps_to_anthropic(self):
        assert provider_for_model("claude-sonnet-5-5") == "anthropic"

    def test_gemini_and_gemma_map_to_google(self):
        assert provider_for_model("gemini-3-pro") == "google"
        assert provider_for_model("gemma4-32k") == "google"

    def test_command_maps_to_cohere(self):
        assert provider_for_model("command-a") == "cohere"

    def test_unknown_falls_back_to_openai(self):
        assert provider_for_model("qwen3.7-plus") == "openai"


class TestTranslateGatewayProfile:
    def test_gateway_mode_translates_endpoint_key_and_defaults(self):
        payload = {
            "mode": "gateway",
            "endpoint": "https://dataserver3.singh.local:18788",
            "group": "homelab-developers",
            "models": ["qwen3.7-plus"],
            "defaults": {"timeout_seconds": 300, "max_tokens": 8192},
            "credentials": {"virtual_key": "vk_secret_value"},
        }
        translated = translate_gateway_profile(payload)
        assert translated.endpoint == "https://dataserver3.singh.local:18788"
        assert translated.virtual_key == "vk_secret_value"
        assert translated.model == "qwen3.7-plus"
        assert translated.provider == "openai"
        doc = translated.document
        providers = doc.get("providers") or {}
        gateway_block = next(iter(providers.values())) if providers else {}
        assert gateway_block.get("gateway", {}).get("url") == "https://dataserver3.singh.local:18788"
        assert gateway_block.get("gateway", {}).get("virtual_key") == "vk_secret_value"

    def test_local_mode_requires_no_credentials(self):
        payload = {
            "mode": "local",
            "group": "g1",
            "models": ["qwen3.7-plus"],
            "defaults": {},
        }
        translated = translate_gateway_profile(payload)
        assert translated.model == "qwen3.7-plus"
        assert translated.virtual_key is None
        assert "providers" not in translated.document

    def test_unknown_mode_is_a_value_error(self):
        with pytest.raises(ValueError, match="mode"):
            translate_gateway_profile({"mode": "hybrid", "group": "g1"})

    def test_gateway_mode_without_endpoint_is_a_value_error(self):
        with pytest.raises(ValueError, match="endpoint"):
            translate_gateway_profile({
                "mode": "gateway",
                "group": "g1",
                "credentials": {"virtual_key": "vk_x"},
            })

    def test_gateway_mode_without_key_is_a_value_error(self):
        with pytest.raises(ValueError):
            translate_gateway_profile({
                "mode": "gateway",
                "endpoint": "https://gw:18788",
                "group": "g1",
            })

    def test_empty_models_defaults_to_auto(self):
        translated = translate_gateway_profile({"mode": "local", "group": "g1", "models": []})
        assert translated.model == "auto"


class TestMaskSecret:
    def test_short_prefix_plus_ellipsis(self):
        assert mask_secret("vk_abcdef0123456789") == "vk_abc…"

    def test_never_returns_the_full_secret(self):
        secret = "vk_secret_value_that_is_long"
        masked = mask_secret(secret)
        assert secret not in masked
        assert masked.endswith("…")
