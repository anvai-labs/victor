"""Qwen policy tests at the typed Sandhi boundary."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.providers.base import Message
from victor.providers.oauth_manager import OAUTH_PROVIDERS
from victor.providers.qwen_provider import (
    QWEN_BASE_URLS,
    QWEN_MODELS,
    QWEN_OAUTH_CONFIG,
    QwenProvider,
)
from victor.providers.sandhi_transport import SandhiTypedProviderMixin
from victor.security.auth.sso import SSOConfig, SSOProvider, SSOTokens


def _tokens(access_token: str) -> SSOTokens:
    return SSOTokens(
        access_token=access_token,
        refresh_token="refresh",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )


def test_wire_endpoints_and_models_are_derived_policy() -> None:
    assert isinstance(QWEN_BASE_URLS, MappingProxyType)
    assert QWEN_BASE_URLS == {
        "standard": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "portal": "https://portal.qwen.ai/v1",
        "coding": "https://coding.dashscope.aliyuncs.com/v1",
    }
    assert "qwen3.5" in QWEN_MODELS
    assert QWEN_OAUTH_CONFIG == {"oauth_base_url": "https://chat.qwen.ai"}


def test_api_key_mode_is_a_clientless_typed_policy() -> None:
    provider = QwenProvider(api_key="qwen-key")

    assert isinstance(provider, SandhiTypedProviderMixin)
    assert provider.name == "qwen"
    assert provider.base_url == QWEN_BASE_URLS["standard"]
    assert not hasattr(provider, "client")
    assert provider.supports_tools()
    assert provider.supports_streaming()
    assert provider.context_window("qwen3.5") == 131072


def test_custom_endpoint_is_host_policy_not_a_python_transport() -> None:
    provider = QwenProvider(api_key="qwen-key", base_url="https://qwen.internal/v1")

    assert provider.base_url == "https://qwen.internal/v1"
    assert not hasattr(provider, "client")


@pytest.mark.asyncio
async def test_oauth_refreshes_before_typed_execution() -> None:
    with patch("victor.providers.qwen_provider.OAuthTokenManager") as manager_cls:
        manager = MagicMock()
        manager._load_cached.return_value = _tokens("initial")
        manager.get_valid_token = AsyncMock(return_value="refreshed")
        manager_cls.return_value = manager
        provider = QwenProvider(auth_mode="oauth")

    provider._sandhi_complete = AsyncMock(
        return_value={
            "schema_version": "1",
            "model": "qwen3.5",
            "output": {"content": "ok"},
            "usage": {
                "tokens_in": 1,
                "tokens_out": 1,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
            },
        }
    )
    response = await provider.chat([Message(role="user", content="hi")], model="qwen3.5")

    assert provider.base_url == QWEN_BASE_URLS["portal"]
    assert provider._api_key == "refreshed"
    assert provider.api_key == "refreshed"
    assert response.content == "ok"
    manager.get_valid_token.assert_awaited_once()
    provider._sandhi_complete.assert_awaited_once()


def test_qwen_oauth_registry_contract() -> None:
    config = OAUTH_PROVIDERS["qwen"]
    assert config.provider_name == "qwen"
    assert config.sso_provider == SSOProvider.QWEN
    assert "chat.qwen.ai" in config.issuer_url
    with patch.dict(os.environ, {"VICTOR_QWEN_OAUTH_CLIENT_ID": "test-qwen-id"}):
        sso = config.to_sso_config()
    assert isinstance(sso, SSOConfig)
    assert sso.provider == SSOProvider.QWEN
    assert sso.use_pkce
    assert sso.client_id == "test-qwen-id"


def test_qwen_config_registry_keeps_host_auth_selection() -> None:
    from victor.config.provider_config_registry import QwenConfig, get_provider_config_registry

    registry = get_provider_config_registry()
    assert registry._aliases.get("alibaba") == "qwen"
    assert registry._aliases.get("dashscope") == "qwen"

    config = QwenConfig()
    settings = MagicMock()
    assert (
        config.get_settings(settings, {"auth_mode": "oauth"})["base_url"]
        == QWEN_BASE_URLS["portal"]
    )
