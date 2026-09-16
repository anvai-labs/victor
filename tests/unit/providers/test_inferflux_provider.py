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


def test_model_policy_lineup_and_prefix_routes(provider: InferfluxProvider) -> None:
    # Model ids are operator config server-side; the YAML tier pins the
    # production R9700 store lineup (served by inferflux's
    # config/server.rocm.qwen3coder30b.yaml). The 32768 windows mirror the
    # serving pool's per-sequence context (n_ctx 65536 / 2 sequences).
    required = {"description", "context_window", "max_output", "supports_tools"}
    assert all(required <= metadata.keys() for metadata in INFERFLUX_MODELS.values())
    assert provider.get_default_model() == "qwen3-coder-30b"
    assert INFERFLUX_MODELS["qwen3-coder-30b"]["supports_tools"] is True
    assert provider.context_window("qwen3-coder-30b") == 32768
    assert provider.context_window("gpt-oss-20b") == 32768
    # Unknown operator id: the prefix routes (then the default) catch it, so
    # budgeting still matches whatever the server actually loads.
    assert provider.context_window("granite-8b") == 32768


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


def test_keyless_local_construction_needs_no_credential(
    monkeypatch, tmp_path
) -> None:
    """An anonymous InferFlux server is the documented default: constructing the
    provider with no env key must yield the EMPTY key (sandhi 0.3.0 then sends no
    Authorization header at all — ADR-0008 D1), never APIKeyNotFoundError."""
    monkeypatch.delenv("INFERFLUX_API_KEY", raising=False)
    # Hermetic: point the resolver's keys-file source at an empty dir so a
    # developer-machine ~/.victor/api_keys.yaml can't satisfy the lookup.
    import victor.config.api_keys as api_keys_module

    monkeypatch.setattr(
        api_keys_module,
        "_get_secure_keys_file",
        lambda: tmp_path / "missing" / "api_keys.yaml",
    )
    provider = InferfluxProvider()
    assert provider._api_key == ""


def test_inferflux_api_key_env_var_is_live(monkeypatch) -> None:
    """A secured deployment sets INFERFLUX_API_KEY; the registry-YAML mapping (the
    source of truth) must resolve it, not just the fallback dict."""
    monkeypatch.setenv("INFERFLUX_API_KEY", "secret123")
    provider = InferfluxProvider()
    assert provider._api_key == "secret123"
