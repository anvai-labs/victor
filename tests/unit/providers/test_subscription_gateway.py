"""Gateway ownership: no OpenAI credential discovery/refresh in Victor clients."""

import base64
import hashlib
import ssl
from unittest.mock import AsyncMock, patch

from pathlib import Path

import pytest

from victor.providers.openai_provider import OpenAIProvider
from victor.providers.sandhi_transport import SandhiOpenAIProvider
from types import SimpleNamespace
from unittest.mock import MagicMock
import victor.providers.sandhi_transport as st

_GATEWAY_TEST_CA_DER = base64.b64decode(
    "MIIDHTCCAgWgAwIBAgIUFHDM95WTU/6/VtwGJIVaVZACMDUwDQYJKoZIhvcNAQELBQAwHTEbMBkG"
    "A1UEAwwSVmljdG9yJTIwVGVzdCUyMENBMCAXDTI2MTEyMjAyMDY1OFoYDzIxMjYxMDI5MDIwNjU4"
    "WjAdMRswGQYDVQQDDBJWaWN0b3IlMjBUZXN0JTIwQ0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAw"
    "ggEKAoIBAQDZf796tmxh+k3CKCy26s0xJ+Isd1fkt8Qy9ptgO1qk1FNMGku3EGoLzElTFtgZ299e"
    "LwF+Pa9JufGX96mQCayx4RcPN0EAPbf2d8xCJ8MBSA6CliyXmfVyXWerG+Xn+XLkh2NrucFlQsSf"
    "ivYSod8kDqrIdSkXRN//e7LbnJUFMTqWyzG2pl1AAZhYXpMgnnp9cPOib3Fe0BnTkHKXyQy9AweJ"
    "bvL1J8RrCCugfaJ9j0rradRe24zs8CaxM+RZ7thR5913lIfH163POKN8JnUphzRpAKiRLNz7H75S"
    "3KniMDoDDwy+/TVbVO/G8Z+uvdXa6jNIUgS00jvsFLFl/T+jAgMBAAGjUzBRMB0GA1UdDgQWBBRL"
    "TApjhf9RKgxzaDAW8ALWreqC+TAfBgNVHSMEGDAWgBRLTApjhf9RKgxzaDAW8ALWreqC+TAPBgNV"
    "HRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQAki8m/12B8bFTqVgEdKgrjk9Jqzf3Ga81R"
    "OQbOYb+V+mQ91azdyfYFT5Qrus0IrGaXUL/Dolx09UMGeFue3lVoHBd+9+8vsUsBcKD1KPnCywae"
    "EJOYYl5TjUFYqUDha8tNkZkzLNB3HNFBNdFWalr9nVRKLrl3+rRsLtnNm5W517bdru3FdyPrQLjn"
    "XKzWYL3TXZfxwcHKvacUAbdOu1p/fgO21OWIqFsxY3LhXF9b8GOafQxSfcCPDE8CeD8XgCucV6iJ"
    "+3VxQiIkicLIM1j6RVu1gYIuO8D1bbcgeMu8d03FXyWn7DlkY08HdTZDFXfPOLNtuzHZ97tQ9nmF"
    "CB88"
)


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
@pytest.mark.parametrize("audience", ["sandhi", "https://sandhi.local/resource"])
async def test_gateway_oidc_private_file_rotation_and_identity_pinning(
    monkeypatch, tmp_path, audience
):
    import json
    import time

    runtime = install_runtime(monkeypatch)
    token_file = tmp_path / "access.json"
    config = {
        "token_file": str(token_file),
        "issuer": "https://sso.example.test/oauth2/openid/sandhi",
        "audience": audience,
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
        # Expiry and origin-client substitution cannot trigger upstream fallback.
        payload["expires_on"] = time.time() - 1
        write()
        with pytest.raises(Exception, match="gateway credential unavailable"):
            await provider._ensure_valid_token()
        payload["expires_on"] = time.time() + 600
        payload["audience"] = "victor-local"
        write()
        with pytest.raises(Exception, match="gateway credential unavailable"):
            await provider._ensure_valid_token()
        payload["audience"] = audience
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


def test_oidc_without_gateway_url_refuses_direct_mode(monkeypatch):
    """OIDC credentials without a gateway URL cannot select direct mode."""
    from victor.config.provider_config_registry import resolve_provider_gateway

    monkeypatch.delenv("SANDHI_GATEWAY_URL", raising=False)
    settings = {"gateway": {"oidc": {"token_file": "/private/access.json"}}}
    with pytest.raises(ValueError, match="Gateway configuration requires"):
        resolve_provider_gateway(settings, "openai")
    assert "gateway" in settings


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
def test_explicit_gateway_credentials_without_url_refuses_direct_mode(monkeypatch, gateway):
    """Gateway credentials without a URL fail closed.

    virtual_key and grant are gateway-specific credentials. Dropping them could
    accidentally dispatch through a direct provider account, so partial gateway
    intent remains a configuration error.
    """
    from victor.config.provider_config_registry import resolve_provider_gateway

    monkeypatch.delenv("SANDHI_GATEWAY_URL", raising=False)
    settings = {"gateway": gateway}
    with pytest.raises(ValueError, match="Gateway configuration requires"):
        resolve_provider_gateway(settings, "openai")
    assert settings["gateway"] == gateway


def test_profile_base_url_suppresses_inherited_gateway():
    """When profile_overrides contain base_url but not gateway, the inherited
    gateway from the providers: section must be suppressed so the profile
    gets a direct connection to its custom endpoint."""
    from victor.config.provider_config_registry import ProviderConfigRegistry

    registry = ProviderConfigRegistry()
    # Simulate base_settings with an inherited gateway from providers: section
    base_settings = {
        "base_url": "https://api.openai.com/v1",
        "gateway": {"url": "https://proxy.example.com", "virtual_key": "vk_inherited"},
    }
    # Profile overrides set a custom base_url but do NOT set gateway
    profile_overrides = {
        "base_url": "https://coding-intl.dashscope.aliyuncs.com/v1",
        "api_key": "sk-test",
    }
    result = registry.get_settings(
        provider="openai",
        settings=MagicMock(),
        profile_overrides=profile_overrides,
        account_data=None,
    )
    # The inherited gateway must be suppressed
    assert "gateway" not in result or result.get("gateway") is None
    # The profile's base_url must be preserved
    assert result["base_url"] == "https://coding-intl.dashscope.aliyuncs.com/v1"


def test_gateway_private_ca_requires_a_nonempty_absolute_path(tmp_path):
    from victor.providers.gateway_contract import gateway_config

    valid = {
        "url": "https://gateway.example.test",
        "virtual_key": "vk_fixture",
        "ca_file": str(tmp_path / "gateway-ca.pem"),
    }
    assert gateway_config(valid)["ca_file"] == valid["ca_file"]
    for invalid in ("", "  ", "relative/gateway-ca.pem", None, 7):
        with pytest.raises(ValueError, match="ca_file must be an absolute"):
            gateway_config({**valid, "ca_file": invalid})


def test_gateway_private_ca_survives_profile_model_normalization(tmp_path):
    from victor.config.settings import ProviderGatewayConfig

    ca_file = str(tmp_path / "gateway-ca.pem")
    config = ProviderGatewayConfig(
        url="https://gateway.example.test",
        virtual_key="vk_fixture",
        ca_file=ca_file,
    )
    assert config.model_dump(mode="json")["ca_file"] == ca_file
    with pytest.raises(ValueError, match="ca_file must be an absolute"):
        ProviderGatewayConfig(
            url="https://gateway.example.test", virtual_key="vk_fixture", ca_file="relative.pem"
        )


def test_gateway_private_ca_survives_profile_registry_resolution(tmp_path):
    from victor.config.provider_config_registry import resolve_provider_gateway

    ca_file = str(tmp_path / "gateway-ca.pem")
    settings = {
        "gateway": {
            "url": "https://gateway.example.test/v1",
            "virtual_key": "vk_fixture",
            "grant": "subscription",
            "ca_file": ca_file,
        }
    }
    resolve_provider_gateway(settings, "openai")
    assert settings["gateway"]["ca_file"] == ca_file
    assert settings["gateway"]["url"] == "https://gateway.example.test"


@pytest.mark.asyncio
async def test_gateway_private_ca_scopes_binding_trust_and_handle_cache(monkeypatch, tmp_path):
    runtime = install_runtime(monkeypatch)
    ca_one = tmp_path / "gateway-one.pem"
    ca_two = tmp_path / "gateway-two.pem"
    ca_one.write_bytes(_GATEWAY_TEST_CA_DER)
    ca_two.write_bytes(_GATEWAY_TEST_CA_DER)
    provider = SandhiOpenAIProvider(
        gateway={
            "url": "https://gateway.example.test",
            "virtual_key": "vk_fixture",
            "ca_file": str(ca_one),
        }
    )
    try:
        first_scope = provider._gateway_scope()
        first_snapshot = provider._gateway_ca_snapshot
        assert first_snapshot.source == str(ca_one)
        assert first_snapshot.path != str(ca_one)
        assert Path(first_snapshot.path).stat().st_mode & 0o777 == 0o600
        assert first_scope[3] == first_snapshot.path
        assert first_scope[4] == hashlib.sha256(_GATEWAY_TEST_CA_DER).hexdigest()
        assert provider._typed_provider("model") is runtime.handle
        assert runtime.calls[-1][1]["ca_file"] == first_snapshot.path
        assert runtime.calls[-1][1]["peer_role"] == "sandhi_gateway_v1"

        provider.extra_config["gateway"]["ca_file"] = f"  {ca_one}  "
        assert provider._gateway_scope() == first_scope
        assert provider._gateway_ca_snapshot is first_snapshot
        assert provider._typed_provider("model") is runtime.handle
        assert len(runtime.calls) == 1

        provider.extra_config["gateway"]["ca_file"] = str(ca_two)
        assert provider._typed_provider("model") is runtime.handle
        second_snapshot = provider._gateway_ca_snapshot
        assert second_snapshot is not first_snapshot
        assert second_snapshot.source == str(ca_two)
        assert second_snapshot.path != first_snapshot.path
        assert not Path(first_snapshot.path).exists()
        assert len(runtime.calls) == 2
        assert runtime.calls[-1][1]["ca_file"] == second_snapshot.path
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_gateway_same_path_rotation_pins_one_snapshot(monkeypatch, tmp_path):
    runtime = install_runtime(monkeypatch)
    ca_file = tmp_path / "gateway-ca.der"
    ca_file.write_bytes(_GATEWAY_TEST_CA_DER)
    provider = SandhiOpenAIProvider(
        gateway={
            "url": "https://gateway.example.test",
            "virtual_key": "vk_fixture",
            "ca_file": str(ca_file),
        }
    )
    try:
        before = provider._gateway_scope()
        snapshot = provider._gateway_ca_snapshot
        assert provider._typed_provider("model") is runtime.handle

        ca_file.write_bytes(b"source rotated after the handle was pinned")
        ca_file.unlink()
        after = provider._gateway_scope()
        assert after == before
        assert provider._gateway_ca_snapshot is snapshot
        assert Path(snapshot.path).read_bytes() == _GATEWAY_TEST_CA_DER
        assert provider._typed_provider("model") is runtime.handle
        assert len(runtime.calls) == 1
        assert runtime.calls[0][1]["ca_file"] == snapshot.path
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_gateway_error_mapping_survives_source_ca_removal(monkeypatch, tmp_path):
    import json

    from victor.core.errors import ProviderConnectionError

    runtime = install_runtime(monkeypatch)
    ca_file = tmp_path / "gateway-ca.der"
    ca_file.write_bytes(_GATEWAY_TEST_CA_DER)
    provider = SandhiOpenAIProvider(
        gateway={
            "url": "https://gateway.example.test",
            "virtual_key": "vk_fixture",
            "ca_file": str(ca_file),
        }
    )
    observed = {}

    def diagnose(url, *, ca_file=None):
        observed["url"] = url
        observed["ca_file"] = ca_file
        return "synthetic TLS diagnosis"

    monkeypatch.setattr(st, "_diagnose_endpoint_tls", diagnose)
    original = RuntimeError(
        json.dumps(
            {
                "code": "transport_error",
                "message": "synthetic transport failure",
                "details": {"url": "https://gateway.example.test/v1"},
            }
        )
    )
    runtime.handle = SimpleNamespace(complete_json=AsyncMock(side_effect=original))
    try:
        assert provider._typed_provider("model") is runtime.handle
        ca_file.unlink()
        await provider._sandhi_complete({"model": "model"})
        raise AssertionError("expected the original transport error")
    except ProviderConnectionError as error:
        assert error.raw_error is original
        assert "synthetic transport failure" in str(error)
        assert "synthetic TLS diagnosis" in str(error)
    finally:
        await provider.close()
    assert observed["url"].startswith("https://gateway.example.test/v1")
    assert observed["ca_file"] != str(ca_file)
    assert Path(observed["ca_file"]).exists() is False


def test_gateway_ssl_context_accepts_der_certificate(tmp_path):
    ca_file = tmp_path / "gateway-ca.der"
    ca_file.write_bytes(_GATEWAY_TEST_CA_DER)
    context = st._gateway_ssl_context(str(ca_file))
    assert isinstance(context, ssl.SSLContext)
    assert context.get_ca_certs()


def test_gateway_ssl_context_replaces_ambient_roots_for_der(tmp_path):
    ca_file = tmp_path / "gateway-ca.der"
    ca_file.write_bytes(_GATEWAY_TEST_CA_DER)
    context = st._gateway_ssl_context(str(ca_file))
    assert len(context.get_ca_certs()) == 1
    assert len(ssl.create_default_context().get_ca_certs()) > 1


def test_gateway_ssl_context_loads_validated_pem_bytes(tmp_path):
    ca_file = tmp_path / "gateway-ca.pem"
    ca_file.write_bytes(ssl.DER_cert_to_PEM_cert(_GATEWAY_TEST_CA_DER).encode("ascii"))
    context = st._gateway_ssl_context(str(ca_file))
    assert len(context.get_ca_certs()) == 1


def test_gateway_ssl_context_accepts_symlinked_pem_bundle(tmp_path):
    certificate = tmp_path / "gateway-ca.pem"
    certificate.write_bytes(ssl.DER_cert_to_PEM_cert(_GATEWAY_TEST_CA_DER).encode("ascii"))
    ca_file = tmp_path / "gateway-ca-link.pem"
    try:
        ca_file.symlink_to(certificate)
    except OSError as error:
        pytest.skip(f"temporary symlink support unavailable: {error}")
    context = st._gateway_ssl_context(str(ca_file))
    assert len(context.get_ca_certs()) == 1


def test_gateway_ssl_context_rejects_non_certificate(tmp_path):
    ca_file = tmp_path / "not-a-certificate.der"
    ca_file.write_bytes(b"not a certificate")
    with pytest.raises(ValueError, match="PEM or DER certificate"):
        st._gateway_ssl_context(str(ca_file))


def test_tls_diagnosis_uses_profile_scoped_ca(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    ca_file = str(tmp_path / "gateway-ca.der")
    observed = {}
    context = MagicMock()
    context.wrap_socket.side_effect = ssl.SSLCertVerificationError()

    @contextmanager
    def fake_socket(*args, **kwargs):
        observed["socket_args"] = args
        observed["socket_kwargs"] = kwargs
        yield MagicMock()

    monkeypatch.setattr(st.socket, "create_connection", fake_socket)
    monkeypatch.setattr(
        st, "_gateway_ssl_context", lambda value: observed.setdefault("ca_file", value) and context
    )
    result = st._diagnose_endpoint_tls("https://gateway.example.test:18444", ca_file=ca_file)
    assert observed["ca_file"] == ca_file
    assert "gateway ca_file configured for this Victor profile" in result
    context.wrap_socket.assert_called_once()


@pytest.mark.asyncio
async def test_gateway_private_ca_is_used_by_discovery_without_global_trust(monkeypatch, tmp_path):
    from unittest.mock import Mock

    ca_file = str(tmp_path / "gateway-ca.der")
    Path(ca_file).write_bytes(_GATEWAY_TEST_CA_DER)
    observed = {}

    def discovery(url, **kwargs):
        observed["url"] = url
        observed.update(kwargs)
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "object": "list",
            "sandhi_contract_version": 1,
            "data": [],
        }
        return response

    runtime = install_runtime(monkeypatch)
    monkeypatch.setattr(st.httpx, "get", discovery)
    provider = SandhiOpenAIProvider(
        gateway={
            "url": "https://gateway.example.test",
            "virtual_key": "vk_fixture",
            "ca_file": ca_file,
        }
    )
    try:
        assert provider._load_gateway_model_capabilities() == {}
        assert observed["url"] == "https://gateway.example.test/v1/models"
        assert observed["verify"] is provider._gateway_ca_snapshot.context
        assert observed["verify"].verify_mode == ssl.CERT_REQUIRED
        assert observed["verify"].check_hostname is True
        assert "ca_file" not in runtime.calls
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_gateway_private_ca_is_used_by_async_discovery(monkeypatch, tmp_path):
    from unittest.mock import Mock

    ca_file = str(tmp_path / "gateway-ca.der")
    Path(ca_file).write_bytes(_GATEWAY_TEST_CA_DER)
    observed = {}

    class Client:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, **kwargs):
            observed["request_url"] = url
            observed.update(kwargs)
            response = Mock()
            response.raise_for_status = Mock()
            response.json.return_value = {
                "object": "list",
                "sandhi_contract_version": 1,
                "data": [],
            }
            return response

    monkeypatch.setattr(st.httpx, "AsyncClient", Client)
    provider = SandhiOpenAIProvider(
        gateway={
            "url": "https://gateway.example.test",
            "virtual_key": "vk_fixture",
            "ca_file": ca_file,
        }
    )
    try:
        await provider._refresh_gateway_model_capabilities()
        assert observed["request_url"] == "https://gateway.example.test/v1/models"
        assert observed["verify"] is provider._gateway_ca_snapshot.context
        assert observed["verify"].verify_mode == ssl.CERT_REQUIRED
        assert observed["verify"].check_hostname is True
    finally:
        await provider.close()
