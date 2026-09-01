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
