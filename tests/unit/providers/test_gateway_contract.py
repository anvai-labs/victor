"""Gateway selection and optional credentials stay separate from upstream auth."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from victor.config.settings import ProviderGatewayConfig
from victor.providers.base import Message, ProviderError
from victor.providers.gateway_contract import gateway_virtual_key, validate_gateway_auth
from victor.providers.resilience import ProviderRetryConfig, ResilientProvider
from victor.providers.sandhi_transport import (
    SandhiAnthropicProvider,
    SandhiGoogleProvider,
    SandhiOpenAIProvider,
)


@pytest.mark.parametrize(
    "provider", [SandhiOpenAIProvider, SandhiAnthropicProvider, SandhiGoogleProvider]
)
async def test_url_only_gateway_constructs_without_upstream_credentials(provider):
    with (
        patch("victor.config.api_keys.get_api_key", side_effect=AssertionError("upstream key")),
        patch("httpx.get", side_effect=AssertionError("upstream discovery")),
    ):
        instance = provider(gateway={"url": "http://127.0.0.1:8600"})
    try:
        assert instance._gateway_overrides() == ("http://127.0.0.1:8600", "not-needed")
    finally:
        await instance.close()


def test_url_only_gateway_uses_internal_binding_placeholder():
    assert gateway_virtual_key({"url": "https://gateway.example.test"}) == "not-needed"
    assert gateway_virtual_key(None) is None
    validate_gateway_auth({"url": "https://gateway.example.test"}, "openai")


@pytest.mark.parametrize(
    "auth",
    [
        {"virtual_key": ""},
        {"virtual_key": "  "},
        {"virtual_key": None},
        {"virtual_key": 123},
        {"virtual_key": "invalid\nkey"},
        {"oidc": None},
        {"oidc": "invalid"},
        {"oidc": {}},
        {"auth_mode": "none", "virtual_key": "provided-key"},
        {"auth_mode": "none", "oidc": {}},
        {"auth_mode": "tokens"},
        {"auth_mode": "oidc"},
        {"auth_mode": "invalid"},
        {"api_key": "misplaced-credential"},
        {"credential": "malformed-credential", "audience": "resource"},
        {"audience": "missing-credential"},
    ],
)
def test_supplied_invalid_gateway_auth_never_becomes_anonymous(auth):
    with pytest.raises(ValueError):
        validate_gateway_auth({"url": "https://gateway.example.test", **auth}, "openai")


@pytest.mark.parametrize(
    "auth",
    [
        {"virtual_key": ""},
        {"virtual_key": None},
        {"oidc": None},
        {"auth_mode": "none", "virtual_key": "provided-key"},
        {"api_key": "misplaced-credential"},
        {"credential": "silently-dropped-credential", "audience": "resource"},
    ],
)
def test_profile_model_does_not_erase_supplied_invalid_credentials(auth):
    with pytest.raises(ValueError):
        ProviderGatewayConfig(url="https://gateway.example.test", **auth)


@contextmanager
def protected_gateway():
    headers = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("content-length", "0")))
            headers.append(dict(self.headers))
            body = b'{"error":{"type":"authentication_error","message":"protected gateway"}}'
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", headers
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "provider", [SandhiOpenAIProvider, SandhiAnthropicProvider, SandhiGoogleProvider]
)
@pytest.mark.parametrize(
    "protocol", ["chat_completions", "anthropic_messages", "responses", "gemini_generate_content"]
)
@pytest.mark.parametrize("streaming", [False, True])
async def test_noauth_wire_placeholder_has_no_protected_gateway_authority(
    provider, protocol, streaming
):
    with protected_gateway() as (url, observed):
        with (
            patch("victor.config.api_keys.get_api_key", side_effect=AssertionError("upstream")),
            patch("httpx.get", side_effect=AssertionError("upstream discovery")),
        ):
            instance = provider(
                gateway={"url": url, "protocol": protocol}, auth_mode="oauth", max_retries=4
            )
            fallback = SimpleNamespace(name="forbidden-upstream", chat=AsyncMock(), stream=Mock())
            guarded = ResilientProvider(
                instance,
                fallback_providers=[fallback],
                retry_config=ProviderRetryConfig(
                    max_retries=2, base_delay_seconds=0, jitter_factor=0
                ),
            )
            try:
                with pytest.raises(ProviderError) as failure:
                    if streaming:
                        async for _ in guarded.stream(
                            [Message(role="user", content="hello")], model="alias"
                        ):
                            pass
                    else:
                        await guarded.chat([Message(role="user", content="hello")], model="alias")
                assert failure.value.recoverable is False
                fallback.chat.assert_not_awaited()
                fallback.stream.assert_not_called()
                assert len(observed) == 1
                wire = {name.lower(): value for name, value in observed[0].items()}
                assert wire["authorization"] == "Bearer not-needed"
                assert "x-api-key" not in wire and "x-goog-api-key" not in wire
            finally:
                await instance.close()
