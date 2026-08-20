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

"""Unit tests for the Entra delegated/SSO (OIDC) identity flow."""

from __future__ import annotations

import sys
import types
import urllib.parse
from typing import Any, Dict, List

import pytest

from victor.core.identity import (
    OidcConfig,
    UserIdentity,
    build_authorize_url,
    resolve_identity_from_code,
)


def _cfg() -> OidcConfig:
    return OidcConfig(
        tenant_id="tenantA",
        client_id="client1",
        client_secret="secret1",
        redirect_uri="https://app.example/hitl/auth/callback",
    )


def _install_fake_aiohttp(monkeypatch, *, token_payload, me_payload, statuses=(200, 200)):
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
            return _Resp(statuses[0], token_payload)

        def get(self, url, headers=None, **k):
            calls.append({"verb": "GET", "url": url, "headers": headers})
            return _Resp(statuses[1], me_payload)

    module = types.ModuleType("aiohttp")
    module.ClientSession = _Session
    monkeypatch.setitem(sys.modules, "aiohttp", module)
    return calls


def test_user_identity_label():
    assert UserIdentity("oid", "Ada Lovelace", "ada@x.com").label() == "Ada Lovelace <ada@x.com>"
    assert UserIdentity("oid", None, "ada@x.com").label() == "ada@x.com"
    assert UserIdentity("oid", None, None).label() == "oid"


def test_build_authorize_url_has_required_params():
    url = build_authorize_url(_cfg(), state="signed-token", nonce="n1")
    parsed = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qs(parsed.query)
    assert parsed.path.endswith("/tenantA/oauth2/v2.0/authorize")
    assert q["response_type"] == ["code"]
    assert q["client_id"] == ["client1"]
    assert q["redirect_uri"] == ["https://app.example/hitl/auth/callback"]
    assert q["state"] == ["signed-token"]
    assert q["nonce"] == ["n1"]
    assert "openid" in q["scope"][0] and "User.Read" in q["scope"][0]


async def test_resolve_identity_from_code(monkeypatch):
    calls = _install_fake_aiohttp(
        monkeypatch,
        token_payload={"access_token": "DELEGATED", "expires_in": 3600},
        me_payload={"id": "oid-123", "displayName": "Ada", "mail": "ada@x.com"},
    )
    identity = await resolve_identity_from_code(_cfg(), "auth-code-xyz")

    token_call = next(c for c in calls if c["verb"] == "POST")
    me_call = next(c for c in calls if c["verb"] == "GET")
    assert token_call["data"]["grant_type"] == "authorization_code"
    assert token_call["data"]["code"] == "auth-code-xyz"
    assert me_call["headers"]["Authorization"] == "Bearer DELEGATED"
    assert identity.subject == "oid-123"
    assert identity.label() == "Ada <ada@x.com>"


async def test_me_falls_back_to_upn(monkeypatch):
    _install_fake_aiohttp(
        monkeypatch,
        token_payload={"access_token": "T"},
        me_payload={
            "id": "oid",
            "displayName": "Bob",
            "userPrincipalName": "bob@corp.com",
        },
    )
    identity = await resolve_identity_from_code(_cfg(), "c")
    assert identity.email == "bob@corp.com"


async def test_code_exchange_error_raises(monkeypatch):
    _install_fake_aiohttp(
        monkeypatch,
        token_payload={"error": "invalid_grant", "error_description": "bad code"},
        me_payload={},
        statuses=(400, 200),
    )
    with pytest.raises(RuntimeError, match="bad code"):
        await resolve_identity_from_code(_cfg(), "c")


def test_from_env_requires_redirect_uri(monkeypatch):
    for v in (
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "TEAMS_TENANT_ID",
        "TEAMS_CLIENT_ID",
        "TEAMS_CLIENT_SECRET",
        "VICTOR_HITL_SSO_REDIRECT_URI",
    ):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("AZURE_TENANT_ID", "t")
    monkeypatch.setenv("AZURE_CLIENT_ID", "c")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "s")
    assert OidcConfig.from_env() is None  # no redirect URI -> SSO off

    monkeypatch.setenv("VICTOR_HITL_SSO_REDIRECT_URI", "https://app/hitl/auth/callback")
    cfg = OidcConfig.from_env()
    assert cfg is not None and cfg.redirect_uri.endswith("/hitl/auth/callback")
