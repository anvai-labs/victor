# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""InferFlux policy tests; transport and codec behavior are covered in Sandhi."""

import pytest

from victor.providers.inferflux_provider import (
    DEFAULT_BASE_URL,
    INFERFLUX_MODELS,
    InferfluxProvider,
)
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy


@pytest.fixture
def provider() -> InferfluxProvider:
    return InferfluxProvider(api_key="local-key")


def test_inferflux_is_thin_typed_policy(provider: InferfluxProvider) -> None:
    assert issubclass(InferfluxProvider, SandhiOpenAICompatPolicy)
    assert provider.name == "inferflux"
    # The default endpoint is Sandhi's catalog fact (ADR-0008), not victor config.
    assert provider.base_url == DEFAULT_BASE_URL == "http://127.0.0.1:8080/v1"
    assert not hasattr(provider, "client")


def test_model_policy_placeholder_and_prefix_routes(provider: InferfluxProvider) -> None:
    # Model ids are operator config server-side; the placeholder satisfies the
    # default_model∈models invariant and the prefix routes catch real ids.
    required = {"description", "context_window", "max_output", "supports_tools"}
    assert all(required <= metadata.keys() for metadata in INFERFLUX_MODELS.values())
    assert INFERFLUX_MODELS["llama3-8b"]["supports_tools"] is True
    assert provider.context_window("llama3-8b") == 8192
    assert provider.context_window("qwen3-coder") == 32768  # prefix-route catch


def test_cache_policy_reports_no_split(provider: InferfluxProvider) -> None:
    # InferFlux reports no per-request cache split today (Sandhi ADR-0008
    # consequences) — the honest answer, not the compat blanket.
    assert provider.supports_prompt_caching() is False


@pytest.mark.asyncio
async def test_model_listing_falls_back_to_local_policy(
    provider: InferfluxProvider, monkeypatch
) -> None:
    # Sandhi's inferflux catalog lineup is deliberately empty (operator-defined
    # ids), so `_models_from_sandhi()` yields nothing and the YAML tier serves.
    monkeypatch.setattr(type(provider), "_models_from_sandhi", lambda self: None)
    assert {entry["id"] for entry in await provider.list_models()} == set(INFERFLUX_MODELS)


def test_keyless_local_construction_needs_no_credential(monkeypatch) -> None:
    """An anonymous InferFlux server is the documented default: constructing the
    provider with no env key must yield the EMPTY key (sandhi 0.3.0 then sends no
    Authorization header at all — ADR-0008 D1), never APIKeyNotFoundError."""
    monkeypatch.delenv("INFERFLUX_API_KEY", raising=False)
    provider = InferfluxProvider()
    assert provider._api_key == ""


def test_inferflux_api_key_env_var_is_live(monkeypatch) -> None:
    """A secured deployment sets INFERFLUX_API_KEY; the registry-YAML mapping (the
    source of truth) must resolve it, not just the fallback dict."""
    monkeypatch.setenv("INFERFLUX_API_KEY", "secret123")
    provider = InferfluxProvider()
    assert provider._api_key == "secret123"
