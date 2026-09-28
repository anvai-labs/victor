"""Gateway ownership: no OpenAI credential discovery/refresh in Victor clients."""

from unittest.mock import AsyncMock, patch

import pytest

from victor.providers.openai_provider import OpenAIProvider
from victor.providers.sandhi_transport import SandhiOpenAIProvider
from types import SimpleNamespace
from unittest.mock import MagicMock
import victor.providers.sandhi_transport as st


def install_runtime(monkeypatch):
    runtime = MagicMock()
    runtime.handle = object()
    runtime.calls = []

    def provider(*args, **kwargs):
        runtime.calls.append((args, kwargs))
        return runtime.handle

    runtime.provider = provider
    monkeypatch.setattr(st, "_sg", SimpleNamespace(ProviderRuntime=lambda: runtime))
    monkeypatch.setattr(st, "_verify_wire_contract", lambda: None)
    return runtime


@pytest.mark.asyncio
async def test_gateway_only_client_needs_no_upstream_credential_and_never_loads_oauth(monkeypatch):
    runtime = install_runtime(monkeypatch)
    with (
        patch("victor.providers.openai_provider.UnifiedApiKeyResolver") as resolver,
        patch("victor.providers.openai_provider.OAuthTokenManager") as oauth,
    ):
        provider = SandhiOpenAIProvider(
            auth_mode="oauth",
            gateway={"url": "http://127.0.0.1:18789", "virtual_key": "vk_client_a"},
        )
        await provider._ensure_valid_token()
        handle = provider._typed_provider("test-model")
        assert handle is runtime.handle
        resolver.assert_not_called()
        oauth.assert_not_called()
        args, kwargs = runtime.calls[-1]
        assert args[2] == "vk_client_a"
        assert kwargs["base_url"] == "http://127.0.0.1:18789/v1"
        assert kwargs["max_retries"] == 0
        assert kwargs.get("protocol") != "chatgpt_responses"
        assert "ChatGPT-Account-ID" not in kwargs.get("headers_json", "")
        await provider.client.close()


def test_missing_gateway_key_fails_before_local_secret_resolution():
    with (
        patch("victor.providers.openai_provider.UnifiedApiKeyResolver") as resolver,
        patch("victor.providers.openai_provider.OAuthTokenManager") as oauth,
    ):
        with pytest.raises(ValueError, match="virtual key"):
            OpenAIProvider(gateway={"url": "http://127.0.0.1:18789", "virtual_key": ""})
        resolver.assert_not_called()
        oauth.assert_not_called()


def test_unsafe_gateway_url_is_rejected_before_credentials_are_used():
    with pytest.raises(ValueError, match="HTTPS or loopback"):
        OpenAIProvider(gateway={"url": "http://untrusted.example", "virtual_key": "vk_client"})


@pytest.mark.asyncio
async def test_gateway_renewable_credential_rotates_ffi_handles_and_fails_closed(monkeypatch):
    import time
    from victor.core.identity.protocols import AccessToken

    runtime = install_runtime(monkeypatch)
    credential = SimpleNamespace(
        get_token=AsyncMock(
            side_effect=[
                AccessToken("oidc-first", time.time() + 600),
                AccessToken("oidc-second", time.time() + 600),
                AccessToken("oidc-expired", time.time() - 1),
                RuntimeError("private-broker-diagnostic"),
            ]
        )
    )
    with patch("victor.providers.openai_provider.OAuthTokenManager") as oauth:
        provider = SandhiOpenAIProvider(
            gateway={
                "url": "https://gateway.example.test",
                "credential": credential,
                "audience": "sandhi",
                "grant": "cloud",
            }
        )
        for expected in ["oidc-first", "oidc-second"]:
            await provider._ensure_valid_token()
            provider._typed_provider("model")
            assert runtime.calls[-1][0][2] == expected
            assert '"x-sandhi-grant": "cloud"' in runtime.calls[-1][1]["headers_json"]
            assert len(provider._sandhi_typed_providers) == 1
        for _ in range(2):
            with pytest.raises(Exception, match="gateway credential unavailable") as error:
                await provider._ensure_valid_token()
            assert "private-broker-diagnostic" not in str(error.value)
        credential.get_token.assert_called_with("sandhi")
        oauth.assert_not_called()
        await provider.close()


def test_gateway_rejects_competing_credentials():
    credential = SimpleNamespace(get_token=AsyncMock())
    with pytest.raises(ValueError, match="exactly one"):
        OpenAIProvider(
            gateway={
                "url": "https://gateway.example.test",
                "virtual_key": "vk_secret",
                "credential": credential,
                "audience": "sandhi",
            }
        )


@pytest.mark.asyncio
async def test_gateway_oidc_private_file_rotation_and_identity_pinning(monkeypatch, tmp_path):
    import json
    import time

    runtime = install_runtime(monkeypatch)
    token_file = tmp_path / "access.json"
    config = {
        "token_file": str(token_file),
        "issuer": "https://sso.example.test/oauth2/openid/sandhi",
        "audience": "sandhi",
        "subject": "user-one",
    }
    payload = {
        "access_token": "oidc-first",
        "expires_on": time.time() + 600,
        "token_type": "Bearer",
        **{k: config[k] for k in ("issuer", "audience", "subject")},
    }

    def write():
        token_file.write_text(json.dumps(payload))
        token_file.chmod(0o600)

    write()
    provider = SandhiOpenAIProvider(gateway={"url": "https://gateway.example.test", "oidc": config})
    try:
        await provider._ensure_valid_token()
        provider._typed_provider("model")
        assert runtime.calls[-1][0][2] == "oidc-first"
        payload["access_token"] = "oidc-rotated"
        write()
        await provider._ensure_valid_token()
        provider._typed_provider("model")
        assert runtime.calls[-1][0][2] == "oidc-rotated"
        payload["subject"] = "different-user"
        write()
        with pytest.raises(Exception, match="gateway credential unavailable"):
            await provider._ensure_valid_token()
        payload["subject"] = "user-one"
        write()
        token_file.chmod(0o644)
        with pytest.raises(Exception, match="gateway credential unavailable"):
            await provider._ensure_valid_token()
        token_file.chmod(0o600)
        link = tmp_path / "link.json"
        link.symlink_to(token_file)
        linked = SandhiOpenAIProvider(
            gateway={
                "url": "https://gateway.example.test",
                "oidc": {**config, "token_file": str(link)},
            }
        )
        try:
            with pytest.raises(Exception, match="gateway credential unavailable"):
                await linked._ensure_valid_token()
        finally:
            await linked.close()
    finally:
        await provider.close()


def test_gateway_oidc_config_survives_registry_without_env_key_fallback(monkeypatch):
    from victor.config.settings import ProviderGatewayConfig
    from victor.config.provider_config_registry import resolve_provider_gateway

    config = {
        "token_file": "/private/access.json",
        "issuer": "https://sso.example.test",
        "audience": "sandhi",
        "subject": "user-one",
    }
    monkeypatch.setenv("SANDHI_GATEWAY_VIRTUAL_KEY", "vk-unrelated-env")
    block = ProviderGatewayConfig(url="https://gateway.example.test", oidc=config, grant="cloud")
    settings = {"gateway": block}
    resolve_provider_gateway(settings, "openai")
    assert settings["gateway"]["oidc"] == config
    assert settings["gateway"]["grant"] == "cloud"
    assert not settings["gateway"].get("virtual_key")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"expires_on": 0},
        {"expires_on": True},
        {"expires_on": float("nan")},
        {"access_token": "bad\nheader"},
        {"token_type": "DPoP"},
        {"audience": "different-client"},
        {"issuer": "https://another-issuer.test"},
        {"refresh_token": "must-not-be-read"},
    ],
)
async def test_oidc_file_rejects_malformed_or_misbound_tokens(tmp_path, change):
    import json
    import time
    from victor.core.identity.gateway import GatewayTokenFileCredential

    config = {
        "token_file": str(tmp_path / "access.json"),
        "issuer": "https://sso.example.test",
        "audience": "sandhi",
        "subject": "user-one",
    }
    payload = {
        "access_token": "valid-token",
        "token_type": "Bearer",
        "expires_on": time.time() + 600,
        **{k: config[k] for k in ("issuer", "audience", "subject")},
        **change,
    }
    path = tmp_path / "access.json"
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    with pytest.raises(ValueError):
        await GatewayTokenFileCredential(config).get_token("sandhi")


def test_oidc_unsupported_provider_fails_before_secret_fallback():
    from victor.config.provider_config_registry import resolve_provider_gateway

    with pytest.raises(ValueError, match="OpenAI"):
        resolve_provider_gateway(
            {
                "gateway": {
                    "url": "https://gateway.example.test",
                    "oidc": {"token_file": "/private/access.json"},
                }
            },
            "anthropic",
        )


def test_oidc_missing_gateway_url_cannot_disable_gateway(monkeypatch):
    from victor.config.provider_config_registry import resolve_provider_gateway

    monkeypatch.delenv("SANDHI_GATEWAY_URL", raising=False)
    with pytest.raises(ValueError, match="gateway URL"):
        resolve_provider_gateway(
            {"gateway": {"oidc": {"token_file": "/private/access.json"}}}, "openai"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_expired_oidc_prevents_public_inference_dispatch(monkeypatch, streaming):
    import time
    from victor.core.identity.protocols import AccessToken
    from victor.providers.base import Message, ProviderError

    runtime = install_runtime(monkeypatch)
    credential = SimpleNamespace(
        get_token=AsyncMock(return_value=AccessToken("expired", time.time() - 1))
    )
    provider = SandhiOpenAIProvider(
        gateway={"url": "http://127.0.0.1:18789", "credential": credential, "audience": "sandhi"}
    )
    try:
        with pytest.raises(ProviderError, match="gateway credential unavailable"):
            if streaming:
                async for _ in provider.stream(
                    [Message(role="user", content="synthetic")], model="allowed"
                ):
                    pytest.fail("an expired token must yield no output")
            else:
                await provider.chat([Message(role="user", content="synthetic")], model="allowed")
        assert runtime.calls == []
    finally:
        await provider.close()


@pytest.mark.parametrize(
    "gateway", [{"virtual_key": "vk_test"}, {"virtual_key": ""}, {"grant": "owner"}]
)
def test_explicit_gateway_scope_without_url_fails_closed(monkeypatch, gateway):
    from victor.config.provider_config_registry import resolve_provider_gateway

    monkeypatch.delenv("SANDHI_GATEWAY_URL", raising=False)
    settings = {"gateway": gateway}
    with pytest.raises(ValueError, match="gateway URL"):
        resolve_provider_gateway(settings, "openai")
    assert settings["gateway"] == gateway
