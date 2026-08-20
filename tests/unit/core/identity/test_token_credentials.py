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

"""Unit tests for the victor.core.identity token-credential layer.

Most tests inject fakes through the TokenCredential protocol (no HTTP). The
client-credentials / IMDS flows use a fake ``aiohttp`` module so they never
touch the network and don't require aiohttp to be installed.
"""

from __future__ import annotations

import sys
import time
import types
from typing import Any, Dict, List

import pytest

from victor.core.identity import (
    AccessToken,
    CachingTokenCredential,
    ChainedTokenCredential,
    ClientAssertionCredential,
    ClientSecretCredential,
    CredentialUnavailableError,
    ManagedIdentityCredential,
    StaticTokenCredential,
    build_entra_credential,
    graph_credential_from_env,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _CountingCredential:
    """A TokenCredential that returns a fixed token and counts get_token calls."""

    def __init__(self, token: str = "T", ttl: float = 3600.0) -> None:
        self.calls = 0
        self._token, self._ttl = token, ttl

    async def get_token(self, *scopes: str) -> AccessToken:
        self.calls += 1
        return AccessToken(token=self._token, expires_on=time.time() + self._ttl)


class _Unavailable:
    async def get_token(self, *scopes: str) -> AccessToken:
        raise CredentialUnavailableError("nope")


def _install_fake_aiohttp(monkeypatch, handler) -> List[Dict[str, Any]]:
    """Inject a fake aiohttp whose POST/GET delegate to ``handler(url, **kw)``."""
    calls: List[Dict[str, Any]] = []

    class _Resp:
        def __init__(self, status, payload):
            self.status, self._p = status, payload

        async def json(self):
            return self._p

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def post(self, url, data=None, **k):
            calls.append({"verb": "POST", "url": url, "data": data})
            return _Resp(*handler(url, data=data))

        def get(self, url, params=None, headers=None, **k):
            calls.append({"verb": "GET", "url": url, "params": params, "headers": headers})
            return _Resp(*handler(url, params=params))

    module = types.ModuleType("aiohttp")
    module.ClientSession = _Session
    module.ClientError = Exception
    module.ClientTimeout = lambda **k: None
    monkeypatch.setitem(sys.modules, "aiohttp", module)
    return calls


# --------------------------------------------------------------------------- #
# AccessToken / StaticTokenCredential
# --------------------------------------------------------------------------- #
def test_access_token_expiry():
    assert AccessToken("t", time.time() - 1).expired
    assert not AccessToken("t", time.time() + 100).expired
    assert AccessToken("t", time.time() + 30).expires_within(60)
    assert not AccessToken("t", time.time() + 300).expires_within(60)


async def test_static_token_credential_returns_token():
    cred = StaticTokenCredential("injected-token")
    tok = await cred.get_token("scope")
    assert tok.token == "injected-token"
    assert not tok.expired


def test_static_token_credential_rejects_empty():
    with pytest.raises(CredentialUnavailableError):
        StaticTokenCredential("")


# --------------------------------------------------------------------------- #
# Caching decorator
# --------------------------------------------------------------------------- #
async def test_caching_reuses_token():
    inner = _CountingCredential()
    cached = CachingTokenCredential(inner)
    a = await cached.get_token("s")
    b = await cached.get_token("s")
    assert a.token == b.token
    assert inner.calls == 1  # second call served from cache


async def test_caching_refreshes_when_near_expiry():
    inner = _CountingCredential(ttl=30.0)  # within the default 60s refresh window
    cached = CachingTokenCredential(inner, refresh_before=60.0)
    await cached.get_token("s")
    await cached.get_token("s")
    assert inner.calls == 2  # always considered stale -> re-minted


async def test_caching_keys_by_scope():
    inner = _CountingCredential()
    cached = CachingTokenCredential(inner)
    await cached.get_token("scopeA")
    await cached.get_token("scopeB")
    assert inner.calls == 2  # different scope sets cache separately


# --------------------------------------------------------------------------- #
# Chain (DefaultAzureCredential-style)
# --------------------------------------------------------------------------- #
async def test_chain_skips_unavailable_and_returns_first_success():
    good = _CountingCredential(token="from-second")
    chain = ChainedTokenCredential(_Unavailable(), good)
    tok = await chain.get_token("s")
    assert tok.token == "from-second"


async def test_chain_raises_when_all_unavailable():
    chain = ChainedTokenCredential(_Unavailable(), _Unavailable())
    with pytest.raises(CredentialUnavailableError):
        await chain.get_token("s")


# --------------------------------------------------------------------------- #
# Client-credentials flows (fake aiohttp)
# --------------------------------------------------------------------------- #
async def test_client_secret_credential_posts_client_credentials(monkeypatch):
    calls = _install_fake_aiohttp(
        monkeypatch,
        lambda url, **k: (200, {"access_token": "ABC", "expires_in": 3600}),
    )
    cred = ClientSecretCredential("tenantA", "client1", "secret1")
    tok = await cred.get_token("https://graph.microsoft.com/.default")
    assert tok.token == "ABC"
    form = calls[0]["data"]
    assert "tenantA/oauth2/v2.0/token" in calls[0]["url"]
    assert form["grant_type"] == "client_credentials"
    assert form["client_secret"] == "secret1"
    assert form["scope"] == "https://graph.microsoft.com/.default"


async def test_client_assertion_credential_uses_jwt_bearer(monkeypatch):
    calls = _install_fake_aiohttp(
        monkeypatch,
        lambda url, **k: (200, {"access_token": "XYZ", "expires_in": 3600}),
    )
    cred = ClientAssertionCredential("tenantA", "client1", lambda: "signed.jwt.assertion")
    tok = await cred.get_token("scope")
    assert tok.token == "XYZ"
    form = calls[0]["data"]
    assert form["client_assertion"] == "signed.jwt.assertion"
    assert form["client_assertion_type"].endswith("jwt-bearer")
    assert "client_secret" not in form


async def test_client_secret_credential_raises_on_error(monkeypatch):
    _install_fake_aiohttp(
        monkeypatch,
        lambda url, **k: (
            401,
            {"error": "invalid_client", "error_description": "bad secret"},
        ),
    )
    with pytest.raises(RuntimeError, match="bad secret"):
        await ClientSecretCredential("t", "c", "s").get_token("scope")


def test_client_secret_credential_requires_all_fields():
    with pytest.raises(ValueError):
        ClientSecretCredential("t", "c", "")


# --------------------------------------------------------------------------- #
# Managed identity
# --------------------------------------------------------------------------- #
async def test_managed_identity_unavailable_off_azure(monkeypatch):
    def _handler(url, **k):
        raise OSError("no route to IMDS")

    # ClientError/OSError -> CredentialUnavailableError
    class _Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def get(self, *a, **k):
            raise OSError("no route to IMDS")

    module = types.ModuleType("aiohttp")
    module.ClientSession = _Session
    module.ClientError = Exception
    module.ClientTimeout = lambda **k: None
    monkeypatch.setitem(sys.modules, "aiohttp", module)

    with pytest.raises(CredentialUnavailableError):
        await ManagedIdentityCredential().get_token("https://graph.microsoft.com/.default")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_build_entra_credential_rejects_both_or_neither():
    with pytest.raises(ValueError):
        build_entra_credential(tenant_id="t", client_id="c")  # neither
    with pytest.raises(ValueError):
        build_entra_credential(
            tenant_id="t", client_id="c", client_secret="s", client_assertion="a"
        )


def test_graph_credential_from_env_prefers_azure_then_teams_alias(monkeypatch):
    for var in (
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "TEAMS_TENANT_ID",
        "TEAMS_CLIENT_ID",
        "TEAMS_CLIENT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)
    assert graph_credential_from_env() is None  # nothing configured

    # TEAMS_* alias works when AZURE_* absent
    monkeypatch.setenv("TEAMS_TENANT_ID", "t")
    monkeypatch.setenv("TEAMS_CLIENT_ID", "c")
    monkeypatch.setenv("TEAMS_CLIENT_SECRET", "s")
    assert graph_credential_from_env() is not None
