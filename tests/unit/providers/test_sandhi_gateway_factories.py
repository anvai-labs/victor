"""Async local-provider factories must not discover an upstream in gateway mode."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from victor.providers.sandhi_transport import (
    SandhiLMStudioProvider,
    SandhiOllamaProvider,
    SandhiVLLMProvider,
    SandhiLlamaCppProvider,
)

CLASSES = [SandhiLMStudioProvider, SandhiOllamaProvider, SandhiVLLMProvider, SandhiLlamaCppProvider]
GATEWAY = {
    "url": "https://gateway.example.test",
    "virtual_key": "vk_factory_fixture",
    "protocol": "responses",
}


@pytest.mark.parametrize("provider_class", CLASSES)
async def test_async_gateway_factory_never_probes_direct_upstream(provider_class):
    with (
        patch.object(
            provider_class,
            "_select_base_url_async",
            AsyncMock(side_effect=AssertionError("direct endpoint discovery forbidden")),
            create=True,
        ) as select,
        patch(
            "victor.providers.ollama_provider.resolve_endpoint_candidates",
            Mock(side_effect=AssertionError("direct endpoint config forbidden")),
        ) as candidates,
        patch.object(
            httpx.AsyncClient,
            "get",
            AsyncMock(side_effect=AssertionError("direct network forbidden")),
        ) as get,
    ):
        provider = await provider_class.create(
            base_url="http://direct.invalid", gateway=dict(GATEWAY)
        )
        try:
            assert provider._gateway_overrides() == (GATEWAY["url"], GATEWAY["virtual_key"])
            assert provider._client_protocol() == "responses"
            assert provider._typed_provider("server-logical-alias") is not None
            select.assert_not_called()
            candidates.assert_not_called()
            get.assert_not_called()
        finally:
            await provider.close()


@pytest.mark.parametrize("provider_class", CLASSES)
@pytest.mark.parametrize(
    "gateway",
    [
        {},
        "malformed",
        {"url": "http://upstream.invalid", "virtual_key": "vk_fixture"},
        {"url": "https://gateway.example.test"},
        {**GATEWAY, "protocol": "chatgpt_responses"},
    ],
)
async def test_async_malformed_gateway_refuses_before_any_direct_discovery(provider_class, gateway):
    with (
        patch.object(
            provider_class,
            "_select_base_url_async",
            AsyncMock(side_effect=AssertionError("direct endpoint discovery forbidden")),
            create=True,
        ) as select,
        patch(
            "victor.providers.ollama_provider.resolve_endpoint_candidates",
            Mock(side_effect=AssertionError("direct endpoint config forbidden")),
        ) as candidates,
        patch.object(
            httpx.AsyncClient,
            "get",
            AsyncMock(side_effect=AssertionError("direct network forbidden")),
        ) as get,
    ):
        with pytest.raises(ValueError, match="gateway"):
            await provider_class.create(gateway=gateway)
        select.assert_not_called()
        candidates.assert_not_called()
        get.assert_not_called()


@pytest.mark.parametrize("provider_class", CLASSES)
async def test_direct_async_factory_keeps_existing_discovery(provider_class):
    select_result = (
        ("http://127.0.0.1:1234", True)
        if provider_class is SandhiLMStudioProvider
        else "http://127.0.0.1:11434"
    )
    with (
        patch.object(
            provider_class,
            "_select_base_url_async",
            AsyncMock(return_value=select_result),
            create=True,
        ) as select,
        patch.object(
            httpx.AsyncClient,
            "get",
            AsyncMock(return_value=httpx.Response(200, json={"data": [{"id": "direct-model"}]})),
        ) as get,
    ):
        provider = await provider_class.create()
        try:
            assert provider._gateway_overrides() is None
            if provider_class in {SandhiLMStudioProvider, SandhiOllamaProvider}:
                select.assert_awaited_once()
                get.assert_not_called()
            else:
                assert get.await_count >= 1
        finally:
            await provider.close()
