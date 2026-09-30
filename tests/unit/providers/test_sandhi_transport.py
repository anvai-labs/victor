"""Contract tests for Victor's direct typed Sandhi provider boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import sys

import pytest

import victor.providers.sandhi_transport as st
from victor.providers.base import (
    Message,
    ProviderConnectionError,
    ProviderRateLimitError,
    ToolDefinition,
)
from victor.providers.deepseek_provider import DeepSeekProvider
from victor.providers.google_provider import GoogleProvider
from victor.providers.llamacpp_provider import LlamaCppProvider
from victor.providers.lmstudio_provider import LMStudioProvider
from victor.providers.moonshot_provider import MoonshotProvider
from victor.providers.ollama_provider import OllamaProvider
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.qwen_provider import QwenProvider
from victor.providers.vllm_provider import VLLMProvider


class FakeTypedProvider:
    def __init__(self, *, complete_error: BaseException | None = None) -> None:
        self.requests: list[dict] = []
        self.complete_error = complete_error

    async def complete_json(self, request_json: str, wire_headers_json=None) -> str:
        self.requests.append(json.loads(request_json))
        if self.complete_error:
            raise self.complete_error
        return json.dumps(
            {
                "schema_version": "1",
                "id": "r1",
                "model": "deepseek-chat",
                "output": {
                    "content": "hello",
                    "tool_calls": [{"id": "c1", "name": "lookup", "arguments": '{"q":1}'}],
                },
                "finish_reason": "tool_calls",
                "usage": {
                    "tokens_in": 6,
                    "tokens_out": 5,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 4,
                    "completeness": "final",
                    "attempts": 1,
                },
                "extensions": {
                    "openai": {
                        "id": "r1",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                },
            }
        )

    def stream_json(self, request_json: str, wire_headers_json=None):
        self.requests.append(json.loads(request_json))

        async def events():
            for event in (
                {"event": "response_start", "id": "r2", "model": "deepseek-chat"},
                {"event": "text_delta", "delta": "he"},
                {"event": "reasoning_delta", "delta": "think"},
                {"event": "tool_call_start", "index": 0, "id": "c1", "name": "lookup"},
                {"event": "tool_call_arguments_delta", "index": 0, "delta": '{"q":'},
                {"event": "tool_call_arguments_delta", "index": 0, "delta": "1}"},
                {"event": "tool_call_end", "index": 0},
                {"event": "finish", "reason": "tool_calls"},
                {
                    "event": "usage",
                    "usage": {
                        "tokens_in": 6,
                        "tokens_out": 5,
                        "cache_creation_tokens": 0,
                        "cache_read_tokens": 4,
                    },
                },
            ):
                yield json.dumps(event)

        return events()


class FakeRuntime:
    def __init__(self, handle: FakeTypedProvider) -> None:
        self.handle = handle
        self.calls: list[tuple] = []

    def provider(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.handle


def install_runtime(monkeypatch, handle: FakeTypedProvider | None = None) -> FakeRuntime:
    runtime = FakeRuntime(handle or FakeTypedProvider())
    monkeypatch.setattr(st, "_sg", SimpleNamespace(ProviderRuntime=lambda: runtime))
    return runtime


def make_provider() -> DeepSeekProvider:
    return DeepSeekProvider(api_key="k", base_url="https://api.deepseek.com/v1")


def test_resolver_always_uses_sandhi_for_admitted_provider(monkeypatch):
    install_runtime(monkeypatch)
    assert st.resolve_transport_class("deepseek", DeepSeekProvider, {}) is DeepSeekProvider


@pytest.mark.parametrize(
    ("name", "provider_cls", "expected"),
    (
        ("openai", OpenAIProvider, st.SandhiOpenAIProvider),
        ("google", GoogleProvider, st.SandhiGoogleProvider),
        ("ollama", OllamaProvider, st.SandhiOllamaProvider),
        ("qwen", QwenProvider, QwenProvider),
        ("lmstudio", LMStudioProvider, st.SandhiLMStudioProvider),
        ("vllm", VLLMProvider, st.SandhiVLLMProvider),
        ("llama.cpp", LlamaCppProvider, st.SandhiLlamaCppProvider),
    ),
)
def test_native_families_resolve_to_typed_sandhi_handles(
    name: str, provider_cls: type, expected: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_runtime(monkeypatch)
    assert st.resolve_transport_class(name, provider_cls, {}) is expected


@pytest.mark.asyncio
async def test_catalog_default_is_omitted_so_sandhi_owns_model_endpoint_routing(monkeypatch):
    runtime = FakeRuntime(FakeTypedProvider())
    monkeypatch.setattr(
        st,
        "_sg",
        SimpleNamespace(
            ProviderRuntime=lambda: runtime,
            provider_spec=lambda provider: {
                "slug": "moonshot",
                "base_url": "https://api.moonshot.cn/v1",
            },
        ),
    )
    resolved = st.resolve_transport_class("moonshot", MoonshotProvider, {})
    provider = resolved(api_key="k")
    await provider.chat([Message(role="user", content="hi")], model="kimi-k3")

    args, kwargs = runtime.calls[0]
    assert args[:3] == ("moonshot", "kimi-k3", "k")
    assert kwargs["base_url"] is None


@pytest.mark.asyncio
async def test_openai_oauth_explicitly_selects_responses_and_refreshes_before_handle(monkeypatch):
    runtime = install_runtime(monkeypatch)
    with patch("victor.providers.openai_provider.OAuthTokenManager") as manager_cls:
        manager = MagicMock()
        manager._load_cached.return_value = SimpleNamespace(
            access_token="cached-oauth", is_expired=False
        )
        manager.get_valid_token = AsyncMock(return_value="fresh-oauth")
        manager.get_chatgpt_account_id.return_value = "workspace_123"
        manager_cls.return_value = manager
        provider = st.SandhiOpenAIProvider(auth_mode="oauth")

    await provider.chat(
        [Message(role="developer", content="policy"), Message(role="user", content="hi")],
        model="o3",
        reasoning_effort="high",
    )

    args, kwargs = runtime.calls[0]
    assert args[:3] == ("openai", "o3", "fresh-oauth")
    assert kwargs["protocol"] == "chatgpt_responses"
    assert kwargs["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert json.loads(kwargs["headers_json"])["originator"] == "victor"
    assert json.loads(kwargs["headers_json"])["ChatGPT-Account-ID"] == "workspace_123"
    request = runtime.handle.requests[0]
    assert "temperature" not in request
    # At sandhi's >= 0.1.5 floor (contract minor >= 4) reasoning_effort rides the
    # typed ChatRequestV1 field; sandhi's Responses codec maps it to reasoning.effort
    # in the native body (openai_responses_typed.rs), so victor no longer dual-writes
    # it into the extensions bucket.
    assert request["reasoning_effort"] == "high"
    assert request["extensions"] == {"openai_responses": {}}


def test_resolver_fails_closed_when_binding_is_missing(monkeypatch):
    monkeypatch.setattr(st, "_sg", None)
    with pytest.raises(ProviderConnectionError):
        st.resolve_transport_class("deepseek", DeepSeekProvider, {})


def test_unknown_non_admitted_provider_is_unchanged(monkeypatch):
    install_runtime(monkeypatch)

    class Other(BaseException):
        pass

    assert st.resolve_transport_class("other", Other, {}) is Other


@pytest.mark.parametrize("name", sorted(st.VICTOR_NATIVE_ONLY_PROVIDER_ALIASES))
def test_explicit_native_only_boundary_is_preserved(name, monkeypatch):
    install_runtime(monkeypatch)

    class NativeOnly:
        pass

    assert st.resolve_transport_class(name, NativeOnly, {}) is NativeOnly


def test_unclassified_victor_owned_provider_fails_closed(monkeypatch):
    install_runtime(monkeypatch)
    provider_cls = type(
        "FutureProvider",
        (),
        {"__module__": "victor.providers.future_provider"},
    )

    with pytest.raises(ProviderConnectionError, match="not classified"):
        st.resolve_transport_class("future", provider_cls, {})


@pytest.mark.asyncio
async def test_complete_consumes_typed_response_and_reuses_handle(monkeypatch):
    runtime = install_runtime(monkeypatch)
    provider = make_provider()
    messages = [Message(role="developer", content="policy"), Message(role="user", content="hi")]
    tools = [ToolDefinition(name="lookup", description="Lookup", parameters={"type": "object"})]

    first = await provider.chat(messages, model="deepseek-chat", tools=tools)
    second = await provider.chat(messages, model="deepseek-chat", tools=tools)

    assert first.content == "hello"
    assert first.tool_calls == [{"id": "c1", "name": "lookup", "arguments": {"q": 1}}]
    assert first.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 4,
    }
    assert len(runtime.calls) == 1, "the typed provider handle must be persistent"
    assert len(runtime.handle.requests) == 2
    assert runtime.handle.requests[0]["messages"][0]["role"] == "developer"
    assert runtime.handle.requests[0]["tools"][0]["name"] == "lookup"
    assert second.raw_response == first.raw_response
    await provider.close()
    assert provider._sandhi_typed_providers is None
    assert provider._sandhi_runtime is None


@pytest.mark.asyncio
async def test_stream_consumes_typed_events_without_sse_round_trip(monkeypatch):
    install_runtime(monkeypatch)
    provider = make_provider()

    chunks = [
        chunk
        async for chunk in provider.stream(
            [Message(role="user", content="hi")], model="deepseek-chat"
        )
    ]

    assert chunks[0].content == "he"
    assert chunks[1].metadata == {"reasoning_content": "think"}
    assert chunks[-1].is_final
    assert chunks[-1].stop_reason == "tool_calls"
    assert chunks[-1].tool_calls == [{"id": "c1", "name": "lookup", "arguments": {"q": 1}}]
    assert chunks[-1].usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 4,
    }


@pytest.mark.asyncio
async def test_binding_failure_is_mapped_and_never_replayed(monkeypatch):
    error = RuntimeError(
        json.dumps(
            {
                "code": "rate_limited",
                "message": "slow down",
                "retryable": True,
                "http_status": 429,
            }
        )
    )
    install_runtime(monkeypatch, FakeTypedProvider(complete_error=error))
    provider = make_provider()

    with pytest.raises(ProviderRateLimitError):
        await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")


class TestTransportErrorTlsDiagnosis:
    """Transport errors must say WHY when the endpoint is a self-signed or
    private-CA gateway: the default recovery hint ("check network") is wrong
    when TCP connects but TLS trust fails (T8 installed-harness finding F2).
    """

    async def test_transport_error_with_untrusted_cert_carries_tls_diagnosis(self, monkeypatch):
        exc = RuntimeError(
            "error sending request for url (https://aiserver1:18788/v1/chat/completions)"
        )
        monkeypatch.setattr(
            st,
            "_typed_error_payload",
            lambda _s: {"code": "transport_error", "message": str(exc)},
        )
        monkeypatch.setattr(
            st,
            "_diagnose_endpoint_tls",
            lambda _url: "the endpoint's TLS certificate is not trusted by this machine's trust store",
        )
        err = await st.map_sandhi_error(exc, "zai", 60.0)
        assert isinstance(err, ProviderConnectionError)
        assert "TLS certificate is not trusted" in str(err)

    async def test_transport_error_without_tls_problem_keeps_default_detail(self, monkeypatch):
        exc = RuntimeError(
            "error sending request for url (https://aiserver1:18788/v1/chat/completions)"
        )
        monkeypatch.setattr(
            st,
            "_typed_error_payload",
            lambda _s: {"code": "transport_error", "message": str(exc)},
        )
        monkeypatch.setattr(st, "_diagnose_endpoint_tls", lambda _url: None)
        err = await st.map_sandhi_error(exc, "zai", 60.0)
        assert isinstance(err, ProviderConnectionError)
        assert "TLS diagnosis" not in str(err)

    async def test_transport_error_survives_a_raising_probe(self):
        # A misconfigured URL (port out of range) must never replace the
        # original transport error with a probe failure (review F, MEDIUM-HIGH).
        exc = RuntimeError(
            "error sending request for url (https://aiserver1:99999/v1/chat/completions)"
        )
        monkeypatch_like_payload = {"code": "transport_error", "message": str(exc)}
        with patch.object(st, "_typed_error_payload", lambda _s: monkeypatch_like_payload):
            err = await st.map_sandhi_error(exc, "zai", 60.0)
        assert isinstance(err, ProviderConnectionError)
        assert "error sending request" in str(err)

    def test_probe_returns_none_for_plain_http_urls(self):
        assert st._diagnose_endpoint_tls("http://aiserver1:18788/v1") is None

    def test_probe_returns_none_when_endpoint_unreachable(self):
        # Port 1 on loopback: nothing listens; the network hint is correct.
        assert st._diagnose_endpoint_tls("https://127.0.0.1:1/v1") is None

    def test_probe_classifies_self_signed_listener(self):
        import socket as socket_mod
        import ssl as ssl_mod
        import threading

        SERVER_CERT = """-----BEGIN CERTIFICATE-----
MIIDDzCCAfegAwIBAgIUZ0U5BzFXhRO4OO7DfAY5l4aMqkowDQYJKoZIhvcNAQEL
BQAwFzEVMBMGA1UEAwwMdDgtbG9jYWxob3N0MB4XDTI2MDkzMDAxNDYyOVoXDTI2
MTAwMTAxNDYyOVowFzEVMBMGA1UEAwwMdDgtbG9jYWxob3N0MIIBIjANBgkqhkiG
9w0BAQEFAAOCAQ8AMIIBCgKCAQEA8K6/7NOwPl808O5cFZmu5oA+CHenf1czcD6a
wpLIQyQQbeR4KaV9KVTJV0rROEBoiRysawpP7CaAax+T9pMEf9huMK5+LStmmIdQ
UWUO6tnMQIcUOk+9uQAm8AeFPkFS5iuwRvGdAIcEE86/FMQaDOmuzWVmKtH1wrTd
1LwsAhLdGxnCnwTRhz0iztrXnfMz5ZFPwOgpPGh5ewQr0HPxDST+cvsKscKgmfip
QYeR70kLeCA1Dyo/mfnTma8PRvlDWDzOAp66DXwWGIh4OFyx1toCGhHH/X2m8aBH
CaUBKuDt5VPtegGNCMNIQogB1/psS7gWsprO3FgtZvPEMKoFOwIDAQABo1MwUTAd
BgNVHQ4EFgQUqNWXJrzxpxeaKm9vcwhxoTkMoOMwHwYDVR0jBBgwFoAUqNWXJrzx
pxeaKm9vcwhxoTkMoOMwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOC
AQEAgDPsOPgCX112GWVaMgRFPHNCQYPV8p/FOER3GAG6sYWINJaX2MMs/bYUUDoj
NXg+M1kC8otwVtOIG2Z6l7WSg0cJzf11BFNZh1bWBrgKxJYn2OmXAdrREbcrTzcH
iVn2pAXJ7neO8IgT8pFVocwesPlsOfSKcbXGVjeAJbyWJOViIjKop0ywViNOMIJL
KlLpYOAFpHD21DwKrhjhzz32d6xu99JkKEJqSNNMOc/bIR7d48TSHGhTzyzbg7YJ
W14QPqIWLOK/zjyiUmaaLob5D79GRXLWAOvipnHsAc6v59POPyTSW378GBrQes5G
6mcd0MRcdFqKgLl6kGtMVcF94Q==
-----END CERTIFICATE-----
"""
        SERVER_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDwrr/s07A+XzTw
7lwVma7mgD4Id6d/VzNwPprCkshDJBBt5HgppX0pVMlXStE4QGiJHKxrCk/sJoBr
H5P2kwR/2G4wrn4tK2aYh1BRZQ7q2cxAhxQ6T725ACbwB4U+QVLmK7BG8Z0AhwQT
zr8UxBoM6a7NZWYq0fXCtN3UvCwCEt0bGcKfBNGHPSLO2ted8zPlkU/A6Ck8aHl7
BCvQc/ENJP5y+wqxwqCZ+KlBh5HvSQt4IDUPKj+Z+dOZrw9G+UNYPM4CnroNfBYY
iHg4XLHW2gIaEcf9fabxoEcJpQEq4O3lU+16AY0Iw0hCiAHX+mxLuBayms7cWC1m
88QwqgU7AgMBAAECggEAQaD9D6iHmfJfHsV4UaGG+i6E+80Y3NXb3mML0zuwJPK8
EiMnCwAnuXH3tvhdRY/2kVDCyStWMMgs40kIkUd0hiHvphGmsU1w2+2l3pQGdc6u
7feHcgEVdFFQzMnYMOoiH0ZRen7et2qZl4ccPp7claHQ+wwHyGxZLd8g0CYBgAy6
YnGUFzyvT0n922k3cypsLMx5k+1pAwe9E5DG6tZ4kW2T872jWqGlXPhwDPQuc3VP
MGdeqcnWmfJEKpk/s2z2AXcSHCyf4Va6u6jPFpSNmd+GOHphBI4YgnEtY7+O1ml7
pdzGWoRJ/qP5Cz/VJExEjI+6KIqu5zB+YSELs/rsNQKBgQD6U0o9LLUlP32qKHx4
FjO3arRniozuup4c7KeHs+4eat+NMnExHZuc0+86vbYeKSiunOrUziq4paIUtfLV
C6D/NXJnM+rNO5GTav/gezErPGbgiSL8ONrYi8lK85S8YTdg8DsfPMKD1/czVr74
COMtxJntVvXdrVpY8pAcTQd7RwKBgQD2I4AJdOhFu6vFnSY2f0bDHzUlO3anuUwg
KAYuWUj4Z2gVKE5TtlUgvqtZn3+bKMfqksHtdLoLQGXIZ0zFwBv1KzhPrtNh84Hc
/G3xtAzJzeWukeQjVRVWAfCSc93DgKJzmpdCTGv9DQqwD4krrAuXwy/bd7ILG2OL
rFv3LeU4bQKBgBGX1PnjsH+DrNNOsSDHfq7/Ytp8FFea6g3iXAvfi3a70CZeSzJG
gG9PPdsFBk2sWt2aza5TJxF/IpsOBpkOjiwhl37FWVU/QIX52S3vuo7tWdWiDcFo
RYk+mdEYuXVb58Z6W81gOdOGVCtZh2ZrSXwn+yGBIRqJWnYx5gr3JvV1AoGAPxHf
qAyty9iH7k4TUZmRb0Qa4Rx4jge8Cu1WkB/Ow9/zWqCGWYr6CzbwPznQf9iWSXQr
fwYO+f0ZV52onW9ZepwFhN1+SrYTy6VfIrUJJdi9htrZQ3h0zCIZG93WsFbQyaCO
K63bae8ikvSYKHmgStX3+FuWYqQ1AMA8nHzFJI0CgYEA7zDiY7T0FztVlgi+ULoy
p1A8lJIXqbTpoqe67juk4pP6cP0oymKbGiinevt4I7iUfawi1vGLd0ZOdO1uFYeL
FfwJ3n72lmGwchK90mq6DEwz82e4XQu0+QQe+QMaajO2i9MjAtsxVypYrzbecaVG
Q8DdskD5L69EkmDFx9mfIFg=
-----END PRIVATE KEY-----
"""

        context = ssl_mod.SSLContext(ssl_mod.PROTOCOL_TLS_SERVER)
        from tempfile import NamedTemporaryFile

        cert_file = NamedTemporaryFile(mode="w", suffix=".pem", delete=False)
        key_file = NamedTemporaryFile(mode="w", suffix=".pem", delete=False)
        cert_file.write(SERVER_CERT)
        key_file.write(SERVER_KEY)
        cert_file.close()
        key_file.close()
        context.load_cert_chain(cert_file.name, key_file.name)

        server = socket_mod.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def serve_once():
            # The probe opens exactly one socket and handshakes on it.
            conn, _ = server.accept()
            try:
                tls = context.wrap_socket(conn, server_side=True)
                tls.recv(1024)
                tls.close()
            except ssl_mod.SSLError:
                pass
            finally:
                server.close()

        thread = threading.Thread(target=serve_once, daemon=True)
        thread.start()
        diagnosis = st._diagnose_endpoint_tls(f"https://127.0.0.1:{port}/v1")
        thread.join(timeout=5)
        assert diagnosis is not None
        assert "not trusted by this machine's trust store" in diagnosis

    def test_probe_returns_none_for_trusted_handshake(self):
        # Runtime-generated SAN=IP certificate: the probe must complete a REAL
        # handshake (the server records the verified client socket) and then
        # classify it as healthy. The former version of this test never
        # handshaked - its listener closed after the raw reachability socket.
        import datetime
        import ipaddress
        import socket as socket_mod
        import ssl as ssl_mod
        import threading
        from tempfile import NamedTemporaryFile

        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "t8-localhost")])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(hours=1))
            .add_extension(
                x509.SubjectAlternativeName(
                    [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
                ),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        cert_file = NamedTemporaryFile(mode="wb", suffix=".pem", delete=False)
        key_file = NamedTemporaryFile(mode="wb", suffix=".pem", delete=False)
        cert_file.write(cert.public_bytes(serialization.Encoding.PEM))
        key_file.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        cert_file.close()
        key_file.close()

        context = ssl_mod.SSLContext(ssl_mod.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_file.name, key_file.name)
        client_context = ssl_mod.SSLContext(ssl_mod.PROTOCOL_TLS_CLIENT)
        client_context.load_verify_locations(cert_file.name)

        state = {"handshakes": 0}
        server = socket_mod.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def serve_once():
            conn, _ = server.accept()
            try:
                tls = context.wrap_socket(conn, server_side=True)
                state["handshakes"] += 1
                tls.recv(1024)
                tls.close()
            except ssl_mod.SSLError:
                pass
            finally:
                server.close()

        thread = threading.Thread(target=serve_once, daemon=True)
        thread.start()

        original = ssl_mod.create_default_context
        ssl_mod.create_default_context = lambda **_kwargs: client_context
        try:
            diagnosis = st._diagnose_endpoint_tls(f"https://127.0.0.1:{port}/v1")
        finally:
            ssl_mod.create_default_context = original
        thread.join(timeout=5)
        assert diagnosis is None
        assert state["handshakes"] == 1

def test_pilot_and_raw_bridge_symbols_are_gone():
    for obsolete in (
        "SandhiTransportUnavailable",
        "set_sandhi_transport_providers",
        "configure_from_settings",
        "sse_lines",
        "_binding_complete",
    ):
        assert not hasattr(st, obsolete)


def test_non_routine_usage_state_survives_victor_compatibility_mapping():
    assert st._usage_diagnostics(
        {
            "attempts": 3,
            "completeness": "final",
            "outcome": "success",
            "upstream_request_id": "up_1",
        }
    ) == {
        "attempts": 3,
        "completeness": "final",
        "outcome": "success",
        "upstream_request_id": "up_1",
    }
    assert (
        st._usage_diagnostics({"attempts": 1, "completeness": "final", "outcome": "success"})
        is None
    )


# =============================================================================
# Gateway mode (TD-0003 P3) — point the FFI handle at the Sandhi proxy with a vk.
# =============================================================================


def make_gateway_provider() -> DeepSeekProvider:
    return DeepSeekProvider(
        api_key="real-upstream-key",
        base_url="https://api.deepseek.com/v1",
        gateway={"url": "http://localhost:8600", "virtual_key": "vk_test_123"},
    )


@pytest.mark.asyncio
async def test_gateway_mode_points_ffi_handle_at_proxy_with_virtual_key(monkeypatch):
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    args, kwargs = runtime.calls[0]
    # The slug is preserved so the proxy still speaks the openai-compat dialect;
    # the virtual key replaces the credential; the proxy URL replaces the endpoint.
    assert args[:3] == ("deepseek", "deepseek-chat", "vk_test_123")
    # deepseek is openai-compat: the transport derives root + /v1 (the adapter
    # appends /chat/completions; the proxy serves it under /v1).
    assert kwargs["base_url"] == "http://localhost:8600/v1"
    # sandhi >= 0.1.5 (victor's floor) accepts "bearer" family-wide as a no-op,
    # so gateway mode presents it unconditionally for the virtual key.
    assert kwargs["auth_scheme"] == "bearer"


@pytest.mark.asyncio
async def test_gateway_mode_preserves_protocol_alongside_overrides(monkeypatch):
    """A gateway-mode provider still selects its wire protocol (e.g. responses)."""
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()
    # An OAuth/responses provider carries _sandhi_protocol; gateway mode must not drop it.
    provider._sandhi_protocol = "chatgpt_responses"

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    _, kwargs = runtime.calls[0]
    assert kwargs["protocol"] == "chatgpt_responses"
    assert kwargs["base_url"] == "http://localhost:8600/v1"
    assert kwargs["auth_scheme"] == "bearer"


@pytest.mark.asyncio
async def test_gateway_mode_requests_bearer_regardless_of_auth_scheme_family(monkeypatch):
    """sandhi >= 0.1.5 accepts "bearer" family-wide, so gateway mode presents it
    unconditionally — with or without a per-family _sandhi_auth_scheme marker —
    and the virtual key rides Authorization."""
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()
    provider._sandhi_auth_scheme = "api_key"  # marks an auth-scheme family

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    _, kwargs = runtime.calls[0]
    assert kwargs["auth_scheme"] == "bearer"


@pytest.mark.asyncio
async def test_gateway_mode_reuses_handle_across_calls(monkeypatch):
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")
    await provider.chat([Message(role="user", content="again")], model="deepseek-chat")

    # The gateway handle is cached just like a direct-mode handle (one FFI build).
    assert len(runtime.calls) == 1


@pytest.mark.asyncio
async def test_gateway_mode_missing_virtual_key_fails_closed(monkeypatch):
    install_runtime(monkeypatch)
    monkeypatch.delenv("SANDHI_GATEWAY_VIRTUAL_KEY_DEEPSEEK", raising=False)
    monkeypatch.delenv("SANDHI_GATEWAY_VIRTUAL_KEY", raising=False)
    provider = DeepSeekProvider(
        api_key="real-upstream-key",
        base_url="https://api.deepseek.com/v1",
        gateway={"url": "http://localhost:8600"},
    )

    with pytest.raises(ProviderConnectionError, match="virtual_key"):
        await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")


@pytest.mark.asyncio
async def test_direct_mode_is_unchanged_when_gateway_not_configured(monkeypatch):
    """Regression: absent gateway leaves the provider in direct FFI mode."""
    runtime = install_runtime(monkeypatch)
    provider = DeepSeekProvider(api_key="k", base_url="https://api.deepseek.com/v1")

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    args, kwargs = runtime.calls[0]
    assert args[:3] == ("deepseek", "deepseek-chat", "k")
    # No gateway override: auth_scheme is not forced to bearer.
    assert kwargs.get("auth_scheme") in (None, "api_key", "")
    assert kwargs["base_url"] == "https://api.deepseek.com/v1"


def test_resolve_provider_gateway_normalizes_block_and_unwraps_secret():
    from pydantic import SecretStr

    from victor.config.provider_config_registry import resolve_provider_gateway

    base: dict = {"gateway": {"url": "http://localhost:8600", "virtual_key": SecretStr("vk_s")}}
    resolve_provider_gateway(base, "deepseek")
    assert base["gateway"] == {"url": "http://localhost:8600", "virtual_key": "vk_s"}


def test_resolve_provider_gateway_env_fallback_per_provider_then_global(monkeypatch):
    from victor.config.provider_config_registry import resolve_provider_gateway

    base: dict = {"gateway": {"url": "http://localhost:8600"}}
    monkeypatch.setenv("SANDHI_GATEWAY_VIRTUAL_KEY_DEEPSEEK", "vk_per_provider")
    monkeypatch.setenv("SANDHI_GATEWAY_VIRTUAL_KEY", "vk_global")
    resolve_provider_gateway(base, "deepseek")
    assert base["gateway"]["virtual_key"] == "vk_per_provider"

    base = {"gateway": {"url": "http://localhost:8600"}}
    monkeypatch.delenv("SANDHI_GATEWAY_VIRTUAL_KEY_DEEPSEEK", raising=False)
    resolve_provider_gateway(base, "deepseek")
    assert base["gateway"]["virtual_key"] == "vk_global"


def test_resolve_provider_gateway_rejects_credentials_without_url(monkeypatch):
    from victor.config.provider_config_registry import resolve_provider_gateway

    monkeypatch.delenv("SANDHI_GATEWAY_URL", raising=False)
    settings = {"gateway": {"virtual_key": "vk_test"}}
    with pytest.raises(ValueError, match="gateway URL"):
        resolve_provider_gateway(settings, "openai")
    assert settings["gateway"]["virtual_key"] == "vk_test"


def test_resolve_provider_gateway_drops_only_empty_block_without_url(monkeypatch):
    from victor.config.provider_config_registry import resolve_provider_gateway

    monkeypatch.delenv("SANDHI_GATEWAY_URL", raising=False)
    settings = {"gateway": {}}
    resolve_provider_gateway(settings, "openai")
    assert "gateway" not in settings


class TestWireContractHandshake:
    """One-time fail-soft handshake against the installed binding."""

    def setup_method(self):
        st._wire_contract_checked = False

    def teardown_method(self):
        st._wire_contract_checked = False

    def test_mismatch_warns_once(self, monkeypatch, caplog):
        fake_sg = SimpleNamespace(wire_contract_version=lambda: "2")
        import sys

        monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
        with caplog.at_level("WARNING"):
            st._verify_wire_contract()
            st._verify_wire_contract()  # second call must be a no-op
        warnings = [r for r in caplog.records if "wire-contract mismatch" in r.getMessage()]
        assert len(warnings) == 1

    def test_matching_version_is_silent(self, monkeypatch, caplog):
        fake_sg = SimpleNamespace(wire_contract_version=lambda: "1")
        import sys

        monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
        with caplog.at_level("WARNING"):
            st._verify_wire_contract()
        assert not any("wire-contract" in r.getMessage() for r in caplog.records)

    def test_old_binding_without_surface_is_silent(self, monkeypatch, caplog):
        fake_sg = SimpleNamespace()  # predates wire_contract_version
        import sys

        monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
        with caplog.at_level("WARNING"):
            st._verify_wire_contract()
        assert not any("wire-contract" in r.getMessage() for r in caplog.records)


class TestUpstreamBodySurfacing:
    """details.upstream_body from ProviderErrorV1 must reach the surfaced message."""

    def _typed_error(self, details=None):
        import json as _json

        payload = {
            "code": "upstream_error",
            "message": "upstream status 400",
            "retryable": False,
            "http_status": 400,
            "provider": "moonshot",
        }
        if details is not None:
            payload["details"] = details
        return RuntimeError(_json.dumps(payload))

    async def test_upstream_body_appended_to_message(self):
        body = '{"error":{"message":"tool call id call_9 not found"}}'
        err = await st.map_sandhi_error(self._typed_error({"upstream_body": body}), "moonshot", 30.0)
        assert "tool call id call_9 not found" in str(err)

    async def test_no_details_keeps_prior_message(self):
        err = await st.map_sandhi_error(self._typed_error(), "moonshot", 30.0)
        assert "upstream status 400" in str(err)
        assert "upstream body" not in str(err)

    async def test_body_already_in_message_not_duplicated(self):
        body = "duplicate snippet content that is already present"
        payload_err = self._typed_error({"upstream_body": body})
        import json as _json

        parsed = _json.loads(str(payload_err))
        parsed["message"] = f"upstream status 400: {body}"
        err = await st.map_sandhi_error(RuntimeError(_json.dumps(parsed)), "moonshot", 30.0)
        assert str(err).count(body) == 1


# =============================================================================
# Native body optional (foundations strategy F1) — the typed path must behave
# identically when `extensions` is absent; native-only usage fields surface
# through metadata["sandhi_usage"], never through the body.
# =============================================================================


def _typed_payload(**overrides):
    payload = {
        "schema_version": "1",
        "id": "r9",
        "model": "deepseek-chat",
        "output": {"content": "hello", "tool_calls": []},
        "finish_reason": "stop",
        "usage": {
            "tokens_in": 6,
            "tokens_out": 5,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 4,
            "completeness": "final",
            "attempts": 1,
            "outcome": "success",
        },
    }
    payload.update(overrides)
    return payload


def test_extensions_absent_is_behavior_identical(monkeypatch):
    """With no native body, usage/metadata must match the with-body result."""
    install_runtime(monkeypatch)
    provider = make_provider()

    with_native = provider._completion_from_typed(
        _typed_payload(
            extensions={
                "openai": {
                    "id": "r9",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            }
        ),
        "deepseek-chat",
    )
    without_native = provider._completion_from_typed(_typed_payload(), "deepseek-chat")

    assert without_native.usage == with_native.usage
    assert without_native.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 4,
    }
    assert without_native.metadata == with_native.metadata is None
    assert without_native.content == "hello"
    # Debug fallback: with no native body, raw_response is the typed document.
    assert without_native.raw_response["schema_version"] == "1"


def test_native_only_usage_fields_surface_as_diagnostics(monkeypatch):
    """cache-miss/cost exist only in native bodies; the transport boundary
    extracts them into metadata['sandhi_usage'] so the runtime never reads the
    body (unblocks sandhi G8 native-body gating)."""
    install_runtime(monkeypatch)
    provider = make_provider()

    response = provider._completion_from_typed(
        _typed_payload(
            extensions={
                "openai": {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "prompt_cache_miss_tokens": 7,
                        "cost_in_usd_ticks": 123,
                    }
                }
            }
        ),
        "deepseek-chat",
    )

    assert response.metadata["sandhi_usage"] == {
        "cache_miss_tokens": 7,
        "cost_in_usd_ticks": 123,
    }


def test_native_only_extractor_ignores_junk():
    assert st._native_only_usage(None) == {}
    assert st._native_only_usage("usage") == {}
    assert st._native_only_usage({"prompt_cache_miss_tokens": "x", "cost_in_usd_ticks": None}) == {}
    assert st._native_only_usage({"prompt_cache_miss_tokens": 0, "cost_in_usd_ticks": 0}) == {}


class TestTypedErrorClassFastPath:
    """sandhi>=0.1.3 SandhiProviderError: classification without parse dependence."""

    async def test_unparseable_typed_instance_stays_provider_error(self, monkeypatch):
        class FakeSandhiProviderError(RuntimeError):
            pass

        monkeypatch.setattr(st, "_SANDHI_PROVIDER_ERROR_CLS", FakeSandhiProviderError)
        err = await st.map_sandhi_error(
            FakeSandhiProviderError("truncated payload not json"), "moonshot", 30.0
        )
        from victor.providers.base import ProviderConnectionError, ProviderError

        assert isinstance(err, ProviderError)
        assert not isinstance(err, ProviderConnectionError)
        assert "truncated payload not json" in str(err)

    async def test_plain_unparseable_runtime_error_stays_binding_failure(self, monkeypatch):
        class FakeSandhiProviderError(RuntimeError):
            pass

        monkeypatch.setattr(st, "_SANDHI_PROVIDER_ERROR_CLS", FakeSandhiProviderError)
        err = await st.map_sandhi_error(RuntimeError("segfault in binding"), "moonshot", 30.0)
        from victor.providers.base import ProviderConnectionError

        assert isinstance(err, ProviderConnectionError)
        assert "binding failure" in str(err)


# =============================================================================
# W3a soak (sandhi#90): Victor opts out of the native-body echo by default.
# =============================================================================


def test_typed_request_opts_out_of_native_body_by_default():
    request = st._typed_request_from_openai_payload(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert request["include_native_response"] is False


def test_typed_request_honors_native_body_opt_in(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "victor.config.settings.get_settings",
        lambda: SimpleNamespace(provider=SimpleNamespace(sandhi_include_native_response=True)),
    )
    request = st._typed_request_from_openai_payload(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    )
    # Opt-in restores sandhi's default (include): the field stays off the wire.
    assert "include_native_response" not in request


# =============================================================================
# W3b: wire-truth latency flows from the typed usage surface into diagnostics.
# =============================================================================


def test_latency_fields_extracted_from_neutral_usage():
    assert st._latency_fields({"duration_ms": 120, "time_to_first_token_ms": 45}) == {
        "duration_ms": 120,
        "time_to_first_token_ms": 45,
    }
    # Tolerant-absent: pre-W3b runtimes carry neither field.
    assert st._latency_fields({"tokens_in": 1}) == {}
    assert st._latency_fields(None) == {}
    assert st._latency_fields({"duration_ms": "x", "time_to_first_token_ms": -1}) == {}


def test_completion_surfaces_wire_latency_in_diagnostics(monkeypatch):
    install_runtime(monkeypatch)
    provider = make_provider()
    payload = _typed_payload()
    payload["usage"]["duration_ms"] = 120
    response = provider._completion_from_typed(payload, "deepseek-chat")
    assert response.metadata["sandhi_usage"]["duration_ms"] == 120


# =============================================================================
# W3c: minor-version handshake — victor reads the installed contract minor.
# =============================================================================


def test_installed_minor_defaults_to_zero_for_old_bindings(monkeypatch):
    import types

    fake_sg = types.SimpleNamespace(wire_contract_version=lambda: "1")
    monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
    monkeypatch.setattr(st, "_wire_contract_checked", False)
    monkeypatch.setattr(st, "_installed_contract_minor", 0)
    assert st.installed_chat_contract_minor() == 0


def test_installed_minor_read_from_binding(monkeypatch):
    import types

    fake_sg = types.SimpleNamespace(
        wire_contract_version=lambda: "1", chat_contract_minor=lambda: 3
    )
    monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
    monkeypatch.setattr(st, "_wire_contract_checked", False)
    monkeypatch.setattr(st, "_installed_contract_minor", 0)
    assert st.installed_chat_contract_minor() == 3


def test_handshake_accepts_current_known_minor(monkeypatch, caplog):
    """The 0.7.0 pin speaks contract minor 8 (reasoning inclusion + timing sources);
    the handshake reads it exactly and does not warn it is 'ahead'."""
    import types

    assert st.KNOWN_CONTRACT_MINOR == 8
    fake_sg = types.SimpleNamespace(
        wire_contract_version=lambda: "1", chat_contract_minor=lambda: 8
    )
    monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
    monkeypatch.setattr(st, "_wire_contract_checked", False)
    monkeypatch.setattr(st, "_installed_contract_minor", 0)
    with caplog.at_level("INFO"):
        assert st.installed_chat_contract_minor() == 8
    assert not any("ahead of victor" in r.getMessage() for r in caplog.records)


def test_handshake_tolerates_newer_minor_forward_compat(monkeypatch, caplog):
    """installed_minor > KNOWN stays valid forward-compat: victor reads the
    newer minor, logs an informational 'ahead' note, and keeps transporting
    (newer additive fields are simply ignored until victor catches up)."""
    import types

    fake_sg = types.SimpleNamespace(
        wire_contract_version=lambda: "1",
        chat_contract_minor=lambda: st.KNOWN_CONTRACT_MINOR + 1,
    )
    monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
    monkeypatch.setattr(st, "_wire_contract_checked", False)
    monkeypatch.setattr(st, "_installed_contract_minor", 0)
    with caplog.at_level("INFO"):
        assert st.installed_chat_contract_minor() == st.KNOWN_CONTRACT_MINOR + 1
    assert any("ahead of victor" in r.getMessage() for r in caplog.records)


def test_installed_binding_meets_victor_floor_when_export_exists():
    """G-ledger floor pin: once the installed sandhi-gateway exports
    chat_contract_minor, it must be >= victor's known minor (conditional so
    the pinned pre-W3c binding keeps passing until the next pin bump)."""
    sg = pytest.importorskip("sandhi_gateway")
    minor_fn = getattr(sg, "chat_contract_minor", None)
    if not callable(minor_fn):
        pytest.skip("installed sandhi-gateway predates chat_contract_minor")
    assert int(minor_fn()) >= st.KNOWN_CONTRACT_MINOR


# =============================================================================
# W3d/G7: codec purity — promoted typed fields, family-gated bucket, no leak.
# =============================================================================


def _openai_payload(**extra):
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(extra)
    return payload


def test_reasoning_effort_promoted_to_typed_field(monkeypatch):
    monkeypatch.setattr(st, "_wire_contract_checked", True)
    monkeypatch.setattr(st, "_installed_contract_minor", 4)
    request = st._typed_request_from_openai_payload(_openai_payload(reasoning_effort="high"))
    assert request["reasoning_effort"] == "high"
    # At minor >= 4 the extensions copy is dropped (no dual-write).
    assert "reasoning_effort" not in request.get("extensions", {}).get("openai", {})


def test_thinking_normalized_from_victor_shape(monkeypatch):
    monkeypatch.setattr(st, "_wire_contract_checked", True)
    monkeypatch.setattr(st, "_installed_contract_minor", 4)
    request = st._typed_request_from_openai_payload(
        _openai_payload(thinking={"type": "enabled", "budget_tokens": 2048})
    )
    assert request["thinking"] == {"enabled": True, "budget_tokens": 2048}


def test_dual_write_keeps_extensions_copy_below_minor_4(monkeypatch):
    monkeypatch.setattr(st, "_wire_contract_checked", True)
    monkeypatch.setattr(st, "_installed_contract_minor", 3)  # pinned pre-W3d runtime
    request = st._typed_request_from_openai_payload(_openai_payload(reasoning_effort="high"))
    assert request["reasoning_effort"] == "high"  # typed field always emitted
    # ...and the extensions copy is retained so the old runtime still sees it.
    assert request["extensions"]["openai"]["reasoning_effort"] == "high"


def test_internal_kwargs_never_reach_extensions(monkeypatch):
    monkeypatch.setattr(st, "_wire_contract_checked", True)
    monkeypatch.setattr(st, "_installed_contract_minor", 4)
    request = st._typed_request_from_openai_payload(
        _openai_payload(execution_mode="fast", topology_action="escalate", top_p=0.9)
    )
    native = request.get("extensions", {}).get("openai", {})
    assert "execution_mode" not in native
    assert "topology_action" not in native
    # A real (non-internal) passthrough param still rides extensions.
    assert native.get("top_p") == 0.9


def test_normalize_thinking_shapes():
    assert st._normalize_thinking(True) == {"enabled": True}
    assert st._normalize_thinking({"type": "disabled"}) == {"enabled": False}
    assert st._normalize_thinking({"enabled": True, "budget_tokens": 100}) == {
        "enabled": True,
        "budget_tokens": 100,
    }
    assert st._normalize_thinking("nonsense") is None


def test_neutral_mixin_drops_bucket_for_native_encoder_families(monkeypatch):
    """W3d/G7 D3: gemini/cohere/ollama encoders clone extensions[<slug>] as the
    base body, so an OpenAI-shaped bucket must NOT be re-labeled to their key."""
    from victor.providers.base import Message

    monkeypatch.setattr(st, "_wire_contract_checked", True)
    monkeypatch.setattr(st, "_installed_contract_minor", 4)

    class _Stub(st.SandhiNeutralProviderMixin):
        def __init__(self, slug):
            self._slug = slug

        def _sandhi_slug(self):
            return self._slug

    msgs = [Message(role="user", content="hi")]
    # Gemini: native encoder → bucket dropped even with a passthrough param.
    gemini_req = _Stub("gemini")._neutral_request(msgs, "g", 0.7, 100, None, top_p=0.9)
    assert "extensions" not in gemini_req
    # An openai-compat local (lmstudio): bucket re-labeled, not dropped.
    local_req = _Stub("lmstudio")._neutral_request(msgs, "m", 0.7, 100, None, top_p=0.9)
    assert local_req.get("extensions", {}).get("lmstudio", {}).get("top_p") == 0.9


def test_neutral_mixin_maps_disabled_thinking_to_native_ollama_switch(monkeypatch):
    from victor.providers.base import Message

    monkeypatch.setattr(st, "_wire_contract_checked", True)
    monkeypatch.setattr(st, "_installed_contract_minor", 4)

    class _Stub(st.SandhiNeutralProviderMixin):
        def _sandhi_slug(self):
            return "ollama"

    request = _Stub()._neutral_request(
        [Message(role="user", content="hi")],
        "qwen3.5:2b",
        0.0,
        128,
        None,
        thinking=False,
    )
    assert request["thinking"] == {"enabled": False}
    assert request["extensions"] == {"ollama": {"think": False}}


# =============================================================================
# Gateway URL normalization (per-family base derivation + root canonicalization)
# =============================================================================


class _Probe:
    """Capture the base_url each family's gateway branch derives."""

    pass


@pytest.mark.parametrize(
    "slug,root,expected",
    [
        ("deepseek", "http://gw:8600", "http://gw:8600/v1"),
        ("inferflux", "http://gw:8600", "http://gw:8600/v1"),
        ("anthropic", "http://gw:8600", "http://gw:8600"),
        ("claude", "http://gw:8600", "http://gw:8600"),
        ("gemini", "http://gw:8600", "http://gw:8600/v1beta"),
        ("google", "http://gw:8600", "http://gw:8600/v1beta"),
        # Trailing slash and explicit /v1 in the configured root both canonicalize.
        ("deepseek", "http://gw:8600/", "http://gw:8600/v1"),
        ("gemini", "http://gw:8600/v1beta", "http://gw:8600/v1beta"),
    ],
)
def test_gateway_family_base_derivation(slug, root, expected):
    from victor.providers.sandhi_transport import _gateway_family_base

    assert _gateway_family_base(root, slug) == expected


def test_gateway_root_normalization_strips_v1_and_v1beta():
    from victor.config.provider_config_registry import _normalize_gateway_root

    assert _normalize_gateway_root("http://gw:8600") == "http://gw:8600"
    assert _normalize_gateway_root("http://gw:8600/") == "http://gw:8600"
    assert _normalize_gateway_root("http://gw:8600/v1") == "http://gw:8600"
    assert _normalize_gateway_root("http://gw:8600/v1/") == "http://gw:8600"
    assert _normalize_gateway_root("http://gw:8600/v1beta") == "http://gw:8600"
    # A path that merely ends in v1-ish text is not a prefix.
    assert _normalize_gateway_root("http://gw:8600/somev1") == "http://gw:8600/somev1"


def test_resolve_provider_gateway_uses_url_env_fallback(monkeypatch):
    from victor.config.provider_config_registry import resolve_provider_gateway

    monkeypatch.setenv("SANDHI_GATEWAY_URL", "http://env-gw:8600")
    monkeypatch.delenv("SANDHI_GATEWAY_VIRTUAL_KEY", raising=False)
    monkeypatch.delenv("SANDHI_GATEWAY_VIRTUAL_KEY_ACME", raising=False)
    settings: dict = {"gateway": {"virtual_key": "vk_x"}}
    resolve_provider_gateway(settings, "acme")
    assert settings["gateway"]["url"] == "http://env-gw:8600"


def test_resolve_provider_gateway_canonicalizes_v1_suffix(monkeypatch):
    from victor.config.provider_config_registry import resolve_provider_gateway

    monkeypatch.delenv("SANDHI_GATEWAY_URL", raising=False)
    settings: dict = {"gateway": {"url": "http://gw:8600/v1/", "virtual_key": "vk_x"}}
    resolve_provider_gateway(settings, "acme")
    assert settings["gateway"]["url"] == "http://gw:8600"


def test_env_only_gateway_config_materializes_a_block(monkeypatch):
    """SANDHI_GATEWAY_URL + virtual key, no per-provider YAML block at all."""
    from victor.config.provider_config_registry import resolve_provider_gateway

    monkeypatch.setenv("SANDHI_GATEWAY_URL", "http://env-gw:8600")
    monkeypatch.setenv("SANDHI_GATEWAY_VIRTUAL_KEY", "vk_env")
    settings: dict = {}
    resolve_provider_gateway(settings, "acme")
    assert settings["gateway"] == {"url": "http://env-gw:8600", "virtual_key": "vk_env"}


def test_no_gateway_anywhere_leaves_settings_untouched(monkeypatch):
    from victor.config.provider_config_registry import resolve_provider_gateway

    monkeypatch.delenv("SANDHI_GATEWAY_URL", raising=False)
    settings: dict = {}
    resolve_provider_gateway(settings, "acme")
    assert "gateway" not in settings


@pytest.mark.parametrize("outcome", ["success", "failure", "cancellation"])
async def test_close_releases_owned_handles_and_preserves_native_cleanup_outcome(outcome):
    import asyncio
    import weakref

    error = {
        "failure": OSError("native close"),
        "cancellation": asyncio.CancelledError("native cancellation"),
    }.get(outcome)

    class Resource:
        pass

    class Native:
        close = AsyncMock(side_effect=error)

    class Provider(st.SandhiTypedProviderMixin, Native):
        pass

    provider = Provider()
    other = Provider()
    provider._sandhi_runtime = Resource()
    provider._sandhi_typed_providers = {("owned",): Resource()}
    other._sandhi_runtime = Resource()
    runtime_ref = weakref.ref(provider._sandhi_runtime)
    handle_ref = weakref.ref(next(iter(provider._sandhi_typed_providers.values())))
    if error is None:
        await provider.close()
        await provider.close()
        assert Native.close.await_count == 2
    else:
        with pytest.raises(type(error)) as caught:
            await provider.close()
        assert caught.value is error
        Native.close.assert_awaited_once()
    assert runtime_ref() is None
    assert handle_ref() is None
    assert other._sandhi_runtime is not None
