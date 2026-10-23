# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for victor.security.auth.sso.

The token, refresh and device-code endpoints are served by a real local
aiohttp app, so request bodies and URL resolution are exercised end to end.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import socket
import ssl
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web

from victor.security.auth import sso
from victor.security.auth.sso import SSOAuthenticator, SSOConfig, SSOProvider, SSOTokens

Handler = Callable[[Dict[str, str]], "tuple[int, Any]"]


class FakeIdP:
    """Local OAuth endpoint server; records each request's path and form body."""

    def __init__(self) -> None:
        self.routes: Dict[str, Handler] = {}
        self.requests: List[tuple[str, Dict[str, str]]] = []
        self.base_url = ""
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.base_url = f"http://127.0.0.1:{port}"

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        form = dict(await request.post())
        self.requests.append((request.path, form))  # type: ignore[arg-type]
        handler = self.routes.get(request.path)
        if handler is None:
            return web.Response(status=404, text=f"no route {request.path}")
        status, body = handler(form)  # type: ignore[arg-type]
        if isinstance(body, str):
            return web.Response(status=status, text=body)
        return web.json_response(body, status=status)


@pytest.fixture
async def idp():
    server = FakeIdP()
    await server.start()
    yield server
    await server.stop()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _config(provider: SSOProvider = SSOProvider.GENERIC_OIDC, **kwargs: Any) -> SSOConfig:
    kwargs.setdefault("issuer_url", "https://idp.example")
    kwargs.setdefault("client_id", "cid")
    return SSOConfig(provider=provider, **kwargs)


# ---------------------------------------------------------------------------
# Configuration and token value objects
# ---------------------------------------------------------------------------


class TestSSOConfig:
    def test_enterprise_factories(self):
        okta = SSOConfig.for_okta("acme.okta.com", "cid", "secret")
        assert okta.provider is SSOProvider.OKTA
        assert okta.issuer_url == "https://acme.okta.com"
        assert okta.client_secret == "secret"
        assert "groups" in okta.scopes

        azure = SSOConfig.for_azure_ad("tenant", "cid")
        assert azure.issuer_url == "https://login.microsoftonline.com/tenant/v2.0"
        assert azure.client_secret is None

        google = SSOConfig.for_google("cid", "secret")
        assert google.provider is SSOProvider.GOOGLE
        assert google.issuer_url == "https://accounts.google.com"

    def test_subscription_factories(self):
        codex = SSOConfig.for_openai_codex()
        assert codex.redirect_uri == "http://localhost:1455/auth/callback"
        assert "offline_access" in codex.scopes and codex.use_pkce

        qwen = SSOConfig.for_qwen()
        assert qwen.issuer_url == "https://chat.qwen.ai" and qwen.client_id == "qwen-code"

        gemini = SSOConfig.for_google_gemini()
        assert gemini.token_url == "https://oauth2.googleapis.com/token"
        assert gemini.authorize_path == "/o/oauth2/v2/auth"
        assert gemini.extra_auth_params == {"access_type": "offline"}

        copilot = SSOConfig.for_github_copilot()
        assert copilot.use_device_flow and not copilot.use_pkce
        assert copilot.device_code_url == "https://github.com/login/device/code"

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: SSOConfig.for_okta("acme.okta.com", "cid"),
            lambda: SSOConfig.for_azure_ad("t", "cid", "s"),
            lambda: SSOConfig.for_google("cid", "s"),
            SSOConfig.for_openai_codex,
            SSOConfig.for_qwen,
            SSOConfig.for_google_gemini,
            SSOConfig.for_github_copilot,
        ],
    )
    def test_dict_round_trip_is_lossless(self, factory):
        config = factory()
        assert SSOConfig.from_dict(config.to_dict()) == config

    def test_from_dict_keeps_endpoint_overrides(self):
        # Regression: from_dict used to drop the fields to_dict writes, so a
        # stored Gemini/Copilot config refreshed against the wrong token URL.
        restored = SSOConfig.from_dict(SSOConfig.for_github_copilot().to_dict())
        assert restored.token_url == "https://github.com/login/oauth/access_token"
        assert restored.use_device_flow is True
        assert restored.device_code_url == "https://github.com/login/device/code"

    def test_from_dict_minimal_uses_defaults(self):
        config = SSOConfig.from_dict(
            {"provider": "okta", "issuer_url": "https://x", "client_id": "c"}
        )
        assert config.redirect_uri == "http://localhost:8400/callback"
        assert config.scopes == ["openid", "profile", "email"]
        assert config.use_pkce is True and config.use_device_flow is False
        assert config.token_url is None and config.extra_auth_params is None

    def test_from_dict_rejects_unknown_provider(self):
        with pytest.raises(ValueError):
            SSOConfig.from_dict({"provider": "nope", "issuer_url": "x", "client_id": "c"})


class TestSSOTokens:
    def test_expiry(self):
        now = datetime.now(timezone.utc)
        assert SSOTokens("a").is_expired is False
        assert SSOTokens("a", expires_at=now - timedelta(seconds=1)).is_expired is True
        assert SSOTokens("a", expires_at=now + timedelta(hours=1)).is_expired is False

    def test_round_trip(self):
        tokens = SSOTokens(
            access_token="a",
            refresh_token="r",
            id_token="i",
            token_type="DPoP",
            expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
            scopes=["openid"],
        )
        assert SSOTokens.from_dict(tokens.to_dict()) == tokens

    def test_from_dict_without_expiry(self):
        tokens = SSOTokens.from_dict({"access_token": "a"})
        assert tokens.expires_at is None and tokens.token_type == "Bearer"
        assert tokens.to_dict()["expires_at"] is None


def test_credentials_module_reexports_same_objects():
    from victor.workflows.services import credentials

    for name in ("SSOAuthenticator", "SSOConfig", "SSOProvider", "SSOTokens"):
        assert getattr(credentials, name) is getattr(sso, name)


# ---------------------------------------------------------------------------
# Token exchange and refresh
# ---------------------------------------------------------------------------


class TestExchangeAndRefresh:
    @pytest.mark.parametrize(
        ("provider", "token_url", "expected_path"),
        [
            (SSOProvider.GENERIC_OIDC, None, "/oauth/token"),
            (SSOProvider.OKTA, None, "/oauth2/v1/token"),
            (SSOProvider.OKTA, "/custom/token", "/custom/token"),
        ],
    )
    async def test_exchange_code_resolves_token_url(self, idp, provider, token_url, expected_path):
        idp.routes[expected_path] = lambda form: (
            200,
            {
                "access_token": "at",
                "refresh_token": "rt",
                "id_token": "it",
                "expires_in": 60,
                "scope": "openid email",
            },
        )
        config = _config(
            provider,
            issuer_url=idp.base_url,
            token_url=f"{idp.base_url}{token_url}" if token_url else None,
        )
        tokens = await SSOAuthenticator(config)._exchange_code("the-code", "verifier")

        assert tokens.access_token == "at" and tokens.refresh_token == "rt"
        assert tokens.id_token == "it" and tokens.scopes == ["openid", "email"]
        assert tokens.expires_at is not None and not tokens.is_expired
        path, form = idp.requests[-1]
        assert path == expected_path
        assert form == {
            "grant_type": "authorization_code",
            "client_id": "cid",
            "code": "the-code",
            "redirect_uri": config.redirect_uri,
            "code_verifier": "verifier",
        }

    async def test_exchange_code_sends_secret_and_omits_absent_verifier(self, idp):
        idp.routes["/oauth/token"] = lambda form: (200, {"access_token": "at"})
        config = _config(issuer_url=idp.base_url, client_secret="shh")
        tokens = await SSOAuthenticator(config)._exchange_code("c", None)

        _, form = idp.requests[-1]
        assert form["client_secret"] == "shh" and "code_verifier" not in form
        assert tokens.expires_at is None and tokens.scopes == []

    async def test_exchange_code_failure_raises(self, idp):
        idp.routes["/oauth/token"] = lambda form: (400, "invalid_grant")
        with pytest.raises(ValueError, match="Token exchange failed: invalid_grant"):
            await SSOAuthenticator(_config(issuer_url=idp.base_url))._exchange_code("c", None)

    @pytest.mark.parametrize(
        ("provider", "expected_path"),
        [(SSOProvider.GENERIC_OIDC, "/oauth/token"), (SSOProvider.OKTA, "/oauth2/v1/token")],
    )
    async def test_refresh_keeps_old_refresh_token_when_not_rotated(
        self, idp, provider, expected_path
    ):
        idp.routes[expected_path] = lambda form: (200, {"access_token": "new", "expires_in": 30})
        config = _config(provider, issuer_url=idp.base_url, client_secret="shh")
        tokens = await SSOAuthenticator(config).refresh("old-rt")

        assert tokens.access_token == "new" and tokens.refresh_token == "old-rt"
        assert tokens.expires_at is not None
        path, form = idp.requests[-1]
        assert path == expected_path
        assert form == {
            "grant_type": "refresh_token",
            "client_id": "cid",
            "refresh_token": "old-rt",
            "client_secret": "shh",
        }

    async def test_refresh_uses_configured_token_url_and_rotated_token(self, idp):
        idp.routes["/t"] = lambda form: (200, {"access_token": "new", "refresh_token": "rotated"})
        config = _config(issuer_url="https://unused.invalid", token_url=f"{idp.base_url}/t")
        tokens = await SSOAuthenticator(config).refresh("old-rt")

        assert tokens.refresh_token == "rotated" and tokens.expires_at is None
        assert "client_secret" not in idp.requests[-1][1]

    async def test_refresh_failure_raises(self, idp):
        idp.routes["/oauth/token"] = lambda form: (401, "expired")
        with pytest.raises(ValueError, match="Token refresh failed: expired"):
            await SSOAuthenticator(_config(issuer_url=idp.base_url)).refresh("rt")


# ---------------------------------------------------------------------------
# Device code flow (RFC 8628)
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_sleep(monkeypatch):
    """Record poll intervals without waiting them out."""
    delays: List[float] = []
    real_sleep = asyncio.sleep

    async def _sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
        delays.append(delay)
        return await real_sleep(0)

    monkeypatch.setattr(sso.asyncio, "sleep", _sleep)
    return delays


@pytest.fixture
def browser(monkeypatch):
    opened = MagicMock()
    monkeypatch.setattr("webbrowser.open", opened)
    return opened


class TestDeviceCodeLogin:
    def _device_config(self, idp: FakeIdP, **kwargs: Any) -> SSOConfig:
        return _config(
            SSOProvider.GITHUB_COPILOT,
            issuer_url=idp.base_url,
            scopes=["copilot"],
            use_pkce=False,
            use_device_flow=True,
            **kwargs,
        )

    async def test_polls_through_pending_and_slow_down(self, idp, fast_sleep, browser, capsys):
        idp.routes["/login/device/code"] = lambda form: (
            200,
            {
                "device_code": "dc",
                "user_code": "ABCD-1234",
                "verification_uri": "https://github.com/login/device",
                "interval": 1,
            },
        )
        replies = iter(
            [
                {"error": "authorization_pending"},
                {"error": "slow_down"},
                {"access_token": "gho", "expires_in": 120},
            ]
        )
        idp.routes["/login/oauth/access_token"] = lambda form: (200, next(replies))

        # login() dispatches to the device flow when use_device_flow is set.
        tokens = await SSOAuthenticator(self._device_config(idp, client_secret="s")).login(
            timeout=60
        )

        assert tokens.access_token == "gho" and tokens.scopes == ["copilot"]
        assert tokens.refresh_token is None and tokens.token_type == "Bearer"
        assert fast_sleep == [1, 1, 6]  # slow_down adds 5s to the interval
        browser.assert_called_once_with("https://github.com/login/device")
        assert "Enter code: ABCD-1234" in capsys.readouterr().out
        device_form = idp.requests[0][1]
        assert device_form == {"client_id": "cid", "scope": "copilot"}
        poll_form = idp.requests[1][1]
        assert poll_form == {
            "client_id": "cid",
            "device_code": "dc",
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "client_secret": "s",
        }

    async def test_configured_urls_and_browser_failure_is_tolerated(self, idp, fast_sleep, browser):
        browser.side_effect = RuntimeError("no display")
        idp.routes["/dev"] = lambda form: (
            200,
            {"device_code": "dc", "user_code": "U", "verification_url": "https://v"},
        )
        idp.routes["/tok"] = lambda form: (200, {"access_token": "t"})
        config = self._device_config(
            idp, device_code_url=f"{idp.base_url}/dev", token_url=f"{idp.base_url}/tok"
        )
        tokens = await SSOAuthenticator(config)._device_code_login(timeout=60)

        assert tokens.access_token == "t"
        assert fast_sleep == [5]  # default interval
        assert "client_secret" not in idp.requests[1][1]

    async def test_device_code_request_failure(self, idp, browser):
        idp.routes["/login/device/code"] = lambda form: (400, "bad client")
        with pytest.raises(ValueError, match="Device code request failed: bad client"):
            await SSOAuthenticator(self._device_config(idp))._device_code_login()

    async def test_denied_authorization_raises(self, idp, fast_sleep, browser):
        idp.routes["/login/device/code"] = lambda form: (
            200,
            {"device_code": "dc", "user_code": "U", "interval": 0},
        )
        idp.routes["/login/oauth/access_token"] = lambda form: (
            200,
            {"error": "access_denied", "error_description": "user said no"},
        )
        with pytest.raises(ValueError, match="access_denied - user said no"):
            await SSOAuthenticator(self._device_config(idp))._device_code_login()

    async def test_expired_device_code_times_out_without_polling(self, idp, fast_sleep, browser):
        idp.routes["/login/device/code"] = lambda form: (
            200,
            {"device_code": "dc", "user_code": "U", "expires_in": 0},
        )
        with pytest.raises(TimeoutError, match="Device code authentication timed out"):
            await SSOAuthenticator(self._device_config(idp))._device_code_login(timeout=60)
        assert fast_sleep == [] and len(idp.requests) == 1


# ---------------------------------------------------------------------------
# Browser (authorization code + PKCE) flow
# ---------------------------------------------------------------------------


class BrowserLogin:
    """Drives login(): captures the auth URL and replays a redirect to the callback."""

    def __init__(self, monkeypatch, reply: Callable[[Dict[str, str]], Dict[str, str]] | None):
        self.auth_url = ""
        self.params: Dict[str, str] = {}
        self._reply = reply
        self._tasks: List[asyncio.Task[Any]] = []
        monkeypatch.setattr("webbrowser.open", self._open)

    def _open(self, url: str) -> bool:
        self.auth_url = url
        self.params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        if self._reply is not None:
            query = self._reply(self.params)
            self._tasks.append(asyncio.get_running_loop().create_task(self._redirect(query)))
        return True

    async def _redirect(self, query: Dict[str, str]) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.get(self.params["redirect_uri"], params=query) as response:
                assert response.status == 200
                assert "Login successful" in await response.text()


def _ok(params: Dict[str, str]) -> Dict[str, str]:
    return {"code": "auth-code", "state": params["state"]}


class TestBrowserLogin:
    async def test_pkce_login_exchanges_code_with_matching_verifier(self, monkeypatch):
        driver = BrowserLogin(monkeypatch, _ok)
        exchange = AsyncMock(return_value=SSOTokens("at"))
        monkeypatch.setattr(SSOAuthenticator, "_exchange_code", exchange)
        config = _config(redirect_uri=f"http://localhost:{_free_port()}/cb", audience="api://x")

        tokens = await SSOAuthenticator(config).login(timeout=10)

        assert tokens.access_token == "at"
        assert driver.auth_url.startswith("https://idp.example/authorize?")
        params = driver.params
        assert params["response_type"] == "code" and params["client_id"] == "cid"
        assert params["scope"] == "openid profile email" and params["audience"] == "api://x"
        assert params["code_challenge_method"] == "S256"
        code, verifier = exchange.await_args.args
        assert code == "auth-code"
        expected_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        assert params["code_challenge"] == expected_challenge

    async def test_without_pkce_no_challenge_or_verifier(self, monkeypatch):
        driver = BrowserLogin(monkeypatch, _ok)
        exchange = AsyncMock(return_value=SSOTokens("at"))
        monkeypatch.setattr(SSOAuthenticator, "_exchange_code", exchange)
        config = _config(
            SSOProvider.OKTA,
            redirect_uri=f"http://localhost:{_free_port()}/cb",
            use_pkce=False,
        )
        await SSOAuthenticator(config).login(timeout=10)

        assert "/oauth2/v1/authorize?" in driver.auth_url
        assert "code_challenge" not in driver.params
        assert exchange.await_args.args == ("auth-code", None)

    @pytest.mark.parametrize(
        ("config_kwargs", "path", "extra"),
        [
            (
                {"provider": SSOProvider.OPENAI_CODEX},
                "/oauth/authorize",
                {"codex_cli_simplified_flow": "true", "originator": "victor"},
            ),
            (
                {
                    "authorize_path": "/o/oauth2/v2/auth",
                    "extra_auth_params": {"access_type": "offline"},
                },
                "/o/oauth2/v2/auth",
                {"access_type": "offline"},
            ),
        ],
    )
    async def test_authorize_url_variants(self, monkeypatch, config_kwargs, path, extra):
        driver = BrowserLogin(monkeypatch, _ok)
        monkeypatch.setattr(
            SSOAuthenticator, "_exchange_code", AsyncMock(return_value=SSOTokens("at"))
        )
        provider = config_kwargs.pop("provider", SSOProvider.GENERIC_OIDC)
        config = _config(
            provider, redirect_uri=f"http://localhost:{_free_port()}/cb", **config_kwargs
        )
        await SSOAuthenticator(config).login(timeout=10)

        assert driver.auth_url.startswith(f"https://idp.example{path}?")
        assert extra.items() <= driver.params.items()

    async def test_state_mismatch_is_rejected(self, monkeypatch):
        BrowserLogin(monkeypatch, lambda params: {"code": "c", "state": "forged"})
        exchange = AsyncMock()
        monkeypatch.setattr(SSOAuthenticator, "_exchange_code", exchange)
        config = _config(redirect_uri=f"http://localhost:{_free_port()}/cb")

        with pytest.raises(ValueError, match="State mismatch"):
            await SSOAuthenticator(config).login(timeout=10)
        exchange.assert_not_awaited()

    async def test_idp_error_is_raised(self, monkeypatch):
        BrowserLogin(monkeypatch, lambda params: {"error": "access_denied"})
        exchange = AsyncMock()
        monkeypatch.setattr(SSOAuthenticator, "_exchange_code", exchange)
        config = _config(redirect_uri=f"http://localhost:{_free_port()}/cb")

        with pytest.raises(ValueError, match="SSO error: access_denied"):
            await SSOAuthenticator(config).login(timeout=10)
        exchange.assert_not_awaited()

    async def test_timeout_releases_callback_port(self, monkeypatch):
        BrowserLogin(monkeypatch, None)
        port = _free_port()
        config = _config(redirect_uri=f"http://localhost:{port}/cb")

        with pytest.raises(TimeoutError, match="SSO login timed out"):
            await SSOAuthenticator(config).login(timeout=0.05)
        with socket.socket() as s:  # runner.cleanup() ran: the port is free again
            s.bind(("localhost", port))


# ---------------------------------------------------------------------------
# Logout and TLS context
# ---------------------------------------------------------------------------


class TestLogoutAndSSL:
    async def test_logout_generic_with_id_token(self, browser):
        await SSOAuthenticator(_config()).logout(id_token="idt")
        browser.assert_called_once_with(
            "https://idp.example/logout?client_id=cid&id_token_hint=idt"
        )

    async def test_logout_okta_without_id_token(self, browser):
        await SSOAuthenticator(_config(SSOProvider.OKTA)).logout()
        browser.assert_called_once_with("https://idp.example/oauth2/v1/logout?client_id=cid")

    def test_ssl_context_uses_certifi(self):
        pytest.importorskip("certifi")
        assert isinstance(SSOAuthenticator._get_ssl_context(), ssl.SSLContext)

    def test_ssl_context_falls_back_without_certifi(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "certifi", None)
        assert SSOAuthenticator._get_ssl_context() is None
