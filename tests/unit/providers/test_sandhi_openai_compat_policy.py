# Copyright 2026 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Architecture and contract tests for configured OpenAI-compatible providers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

import victor.providers.httpx_openai_compat as httpx_compat
import victor.providers.openai_compat_model_policy as provider_config
import victor.providers.sandhi_transport as sandhi_transport
from victor.framework.session_config import ProviderOverrideConfig
from victor.providers.base import Message
from victor.providers.cerebras_provider import CerebrasProvider
from victor.providers.deepseek_provider import DeepSeekProvider
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy
from victor.providers.fireworks_provider import FireworksProvider
from victor.providers.groq_provider import GroqProvider
from victor.providers.moonshot_provider import MoonshotProvider
from victor.providers.mistral_provider import MistralProvider
from victor.providers.openrouter_provider import OpenRouterProvider
from victor.providers.qwen_provider import QwenProvider
from victor.providers.together_provider import TogetherProvider
from victor.providers.xai_provider import XAIProvider
from victor.providers.zai_provider import ZAIProvider

CONFIGURED_CLASSES = (
    TogetherProvider,
    FireworksProvider,
    OpenRouterProvider,
    MoonshotProvider,
    GroqProvider,
    CerebrasProvider,
    DeepSeekProvider,
    XAIProvider,
    MistralProvider,
    ZAIProvider,
    QwenProvider,
)


def test_parity_safe_adapters_share_one_configured_implementation() -> None:
    assert all(issubclass(cls, SandhiOpenAICompatPolicy) for cls in CONFIGURED_CLASSES)


@pytest.mark.parametrize(
    ("name", "provider_cls"),
    (("moonshot", MoonshotProvider), ("groq", GroqProvider), ("cerebras", CerebrasProvider)),
)
def test_provider_policies_are_already_typed_sandhi_consumers(
    name: str, provider_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandhi_transport, "_sg", SimpleNamespace(ProviderRuntime=object))
    resolved = sandhi_transport.resolve_transport_class(name, provider_cls, {})
    assert issubclass(resolved, provider_cls)
    assert issubclass(resolved, sandhi_transport.SandhiHttpxTransportMixin)


def test_specs_are_immutable_and_combine_sandhi_wire_with_victor_model_policy() -> None:
    specs = provider_config.list_openai_compat_provider_specs()
    assert set(specs) == {
        "together",
        "fireworks",
        "openrouter",
        "moonshot",
        "groq",
        "cerebras",
        "deepseek",
        "xai",
        "mistral",
        "zai",
        "qwen",
        "inferflux",
    }
    assert specs["together"].base_url == "https://api.together.xyz/v1"
    assert specs["fireworks"].base_url == "https://api.fireworks.ai/inference/v1"
    assert specs["openrouter"].base_url == "https://openrouter.ai/api/v1"
    with pytest.raises(TypeError):
        specs["together"] = specs["fireworks"]  # type: ignore[index]
    with pytest.raises(TypeError):
        specs["together"].models["new-model"] = {}  # type: ignore[index]


def test_endpoint_facts_are_loaded_from_sandhi_wire_catalog() -> None:
    sg = pytest.importorskip("sandhi_gateway")
    if not hasattr(sg, "provider_spec"):
        pytest.skip("requires sandhi-gateway 0.1.2+")
    for spec in provider_config.list_openai_compat_provider_specs().values():
        assert sg.provider_spec(spec.slug)["base_url"] == spec.base_url


def test_context_window_routes_preserve_legacy_budgeting_contract() -> None:
    assert (
        TogetherProvider(api_key="key").context_window("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        == 128_000
    )
    assert TogetherProvider(api_key="key").context_window("openai/gpt-oss-120b") == 32_768
    assert (
        FireworksProvider(api_key="key").context_window("accounts/fireworks/models/deepseek-v3p2")
        == 64_000
    )
    assert OpenRouterProvider(api_key="key").context_window("google/gemini-2.5-flash:free") == (
        128_000
    )


@pytest.mark.parametrize("provider_cls", CONFIGURED_CLASSES)
def test_configured_adapters_are_dynamically_sandhi_capable(provider_cls, monkeypatch) -> None:
    monkeypatch.setattr(sandhi_transport, "_sg", SimpleNamespace(ProviderRuntime=object))
    resolved = sandhi_transport.resolve_transport_class(
        provider_cls.provider_spec().slug, provider_cls, {}
    )
    assert issubclass(resolved, provider_cls)
    assert issubclass(resolved, sandhi_transport.SandhiHttpxTransportMixin)
    assert (
        sandhi_transport.resolve_transport_class(
            provider_cls.provider_spec().slug, provider_cls, {}
        )
        is resolved
    )


def test_sandhi_neutral_usage_is_not_reparsed_through_the_binding(monkeypatch) -> None:
    provider = TogetherProvider(api_key="test-key")

    def fail_if_reparsed(*_args, **_kwargs):
        raise AssertionError("transport-neutral usage must not cross the FFI twice")

    monkeypatch.setattr(httpx_compat, "parse_usage_dict", fail_if_reparsed)
    raw = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        httpx_compat.TRANSPORT_NEUTRAL_USAGE_KEY: {
            "tokens_in": 40,
            "tokens_out": 20,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 60,
        },
    }
    response = provider._parse_response(raw, provider.get_default_model())
    assert response.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_input_tokens": 60,
    }
    assert httpx_compat.TRANSPORT_NEUTRAL_USAGE_KEY not in response.raw_response


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_cls", CONFIGURED_CLASSES)
async def test_direct_policy_instances_execute_only_through_typed_ffi(
    provider_cls, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    class Handle:
        async def complete_json(self, request_json: str) -> str:
            calls.append(json.loads(request_json))
            return json.dumps(
                {
                    "schema_version": "1",
                    "model": calls[-1]["model"],
                    "output": {"content": "ok"},
                    "finish_reason": "stop",
                    "usage": {
                        "tokens_in": 2,
                        "tokens_out": 1,
                        "cache_creation_tokens": 0,
                        "cache_read_tokens": 0,
                        "completeness": "final",
                        "attempts": 1,
                        "outcome": "success",
                    },
                }
            )

    class Runtime:
        def provider(self, *_args, **_kwargs):
            return Handle()

    spec = provider_cls.provider_spec()
    monkeypatch.setattr(
        sandhi_transport,
        "_sg",
        SimpleNamespace(
            ProviderRuntime=Runtime,
            provider_spec=lambda _provider: {"base_url": spec.base_url},
        ),
    )
    provider = provider_cls(api_key="test-key", max_retries=0)
    response = await provider.chat(
        [Message(role="user", content="hello")],
        model=provider.get_default_model(),
    )
    assert response.content == "ok"
    assert calls[0]["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize(
    "provider", ["together", "fireworks", "openrouter", "moonshot", "groq", "cerebras"]
)
def test_cli_provider_override_uses_configured_default_model(provider: str) -> None:
    override = ProviderOverrideConfig.from_cli(provider=provider)
    assert override.model == provider_config.get_openai_compat_provider_spec(provider).default_model


@pytest.mark.parametrize(
    "yaml_text,error",
    [
        ("version: 2\nproviders: {}\n", "version must be 1"),
        (
            """version: 1
providers:
  together:
    default_model: missing
    timeout: 30
    max_retries: 1
    default_context_window: 1000
    models: {model: {context_window: 1000}}
""",
            "default_model must appear in models",
        ),
        (
            """version: 1
providers:
  together:
    default_model: model
    timeout: 30
    max_retries: 1
    cache: {supported: 'yes'}
    default_context_window: 1000
    models: {model: {context_window: 1000}}
""",
            "cache.supported must be boolean",
        ),
    ],
)
def test_invalid_config_fails_closed(tmp_path: Path, yaml_text: str, error: str) -> None:
    path = tmp_path / "providers.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(provider_config.OpenAICompatConfigError, match=error):
        provider_config._load_specs(path)


@pytest.mark.asyncio
async def test_list_models_uses_sandhi_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the Sandhi catalog carries a curated lineup, list_models() serves it."""
    provider = DeepSeekProvider(api_key="test-key", max_retries=0)
    monkeypatch.setattr(
        provider,
        "_models_from_sandhi",
        lambda: [
            {
                "id": "deepseek-chat",
                "name": "DeepSeek Chat",
                "context_window": 131_072,
                "max_output_tokens": 8_192,
            }
        ],
    )

    models = await provider.list_models()

    assert models == [
        {
            "id": "deepseek-chat",
            "name": "DeepSeek Chat",
            "context_window": 131_072,
            "max_output_tokens": 8_192,
        }
    ]


@pytest.mark.asyncio
async def test_list_models_falls_back_to_admitted_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without catalog data (aggregators, old sandhi), list_models() serves the spec policy."""
    provider = OpenRouterProvider(api_key="test-key", max_retries=0)
    monkeypatch.setattr(provider, "_models_from_sandhi", lambda: None)

    models = await provider.list_models()

    assert models == [
        {"id": model_id, **dict(model_info)}
        for model_id, model_info in provider.provider_spec().models.items()
    ]


def test_models_from_sandhi_reads_real_catalog_when_available() -> None:
    """Integration: seeded compat vendors resolve through the real Sandhi catalog.

    Skips cleanly when the installed Sandhi predates the catalog surface or the
    compat seeding (sandhi#49), so it is not CI-flaky.
    """
    try:
        import sandhi_gateway as sg
    except Exception:  # pragma: no cover - sandhi absent
        pytest.skip("sandhi-gateway not installed")
    if not hasattr(sg, "provider_models_json"):
        pytest.skip("installed sandhi predates the catalog surface (TD-0004 Phase A)")

    provider = DeepSeekProvider(api_key="test-key", max_retries=0)
    models = provider._models_from_sandhi()
    if models is None:
        pytest.skip("installed sandhi predates compat vendor seeding (sandhi#49)")
    assert any(m["id"] == "deepseek-chat" for m in models)
