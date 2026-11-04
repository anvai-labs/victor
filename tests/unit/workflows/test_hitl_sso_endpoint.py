"""End-to-end tests for the delegated/SSO HITL approval flow.

With SSO configured, the approve/reject link 302-redirects to Entra; the
/auth/callback endpoint then resolves the signed-in approver via Graph /me and
records the decision *with* that identity. A fake aiohttp covers the token
exchange + /me call; the app is driven in-process via httpx ASGITransport.
"""

from __future__ import annotations

import base64
import json
import sys
import types
import urllib.parse

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from victor.workflows.hitl import HITLNodeType, HITLRequest  # noqa: E402
from victor.workflows.hitl_api import HITLStore, create_hitl_router  # noqa: E402
from victor.workflows.hitl_signing import sign_action  # noqa: E402

SECRET = "sso-secret"
REDIRECT_URI = "https://app.example/hitl/auth/callback"


@pytest.fixture()
def sso_env(monkeypatch):
    monkeypatch.setenv("VICTOR_HITL_SIGNING_SECRET", SECRET)
    monkeypatch.setenv("AZURE_TENANT_ID", "tenantA")
    monkeypatch.setenv("AZURE_CLIENT_ID", "client1")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret1")
    monkeypatch.setenv("VICTOR_HITL_SSO_REDIRECT_URI", REDIRECT_URI)
    return monkeypatch


def _fake_aiohttp(monkeypatch, *, me):
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
            return _Resp(200, {"access_token": "DELEGATED"})

        def get(self, url, headers=None, **k):
            return _Resp(200, me)

    module = types.ModuleType("aiohttp")
    module.ClientSession = _Session
    monkeypatch.setitem(sys.modules, "aiohttp", module)


def _make_request(rid):
    return HITLRequest(
        request_id=rid,
        node_id="n",
        hitl_type=HITLNodeType.APPROVAL,
        prompt="Deploy?",
        timeout=300,
    )


def _app(store):
    app = FastAPI()
    app.include_router(create_hitl_router(store, require_auth=False), prefix="/hitl")
    return app


def _client(app, **kw):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", **kw)


async def test_respond_link_redirects_to_entra_when_sso_enabled(sso_env):
    store = HITLStore()
    await store.store_request(_make_request("rid-redir"), workflow_id="wf")
    token = sign_action("rid-redir", "approve", secret=SECRET)

    async with _client(_app(store), follow_redirects=False) as ac:
        resp = await ac.get(f"/hitl/respond/rid-redir?action=approve&token={token}")

    assert resp.status_code == 302
    loc = resp.headers["location"]
    q = urllib.parse.parse_qs(urllib.parse.urlparse(loc).query)
    assert "tenantA/oauth2/v2.0/authorize" in loc
    assert q["redirect_uri"] == [REDIRECT_URI]
    # state carries request_id|action|token for the callback to re-verify.
    # The state is base64-encoded JSON: {"rid": ..., "act": ..., "tok": ...}
    raw_state = q["state"][0]
    try:
        state_payload = json.loads(base64.urlsafe_b64decode(raw_state.encode()).decode())
        assert state_payload["rid"] == "rid-redir"
        assert state_payload["act"] == "approve"
        assert state_payload["tok"]  # non-empty token
    except Exception:
        # Fallback: legacy pipe-delimited format
        assert raw_state.startswith("rid-redir|approve|")
    # decision NOT recorded yet — still pending
    assert (await store.get_request("rid-redir")).status == "pending"


async def test_callback_records_decision_with_responder_identity(sso_env):
    _fake_aiohttp(sso_env, me={"id": "oid-1", "displayName": "Ada", "mail": "ada@x.com"})
    store = HITLStore()
    await store.store_request(_make_request("rid-cb"), workflow_id="wf")
    token = sign_action("rid-cb", "approve", secret=SECRET)
    state = urllib.parse.quote(f"rid-cb|approve|{token}")

    async with _client(_app(store)) as ac:
        resp = await ac.get(f"/hitl/auth/callback?code=AUTHCODE&state={state}")

    assert resp.status_code == 200
    # Page shows the approver (HTML-escaped — XSS-safe); store holds the raw value.
    assert "Ada &lt;ada@x.com&gt;" in resp.text
    stored = await store.get_request("rid-cb")
    assert stored.response.approved is True
    assert stored.response.responder == "Ada <ada@x.com>"  # identity recorded


async def test_callback_with_error_returns_400(sso_env):
    store = HITLStore()
    async with _client(_app(store)) as ac:
        resp = await ac.get("/hitl/auth/callback?error=access_denied&state=x|approve|y")
    assert resp.status_code == 400


async def test_callback_with_tampered_state_token_is_rejected(sso_env):
    _fake_aiohttp(sso_env, me={"id": "o", "displayName": "X", "mail": "x@x.com"})
    store = HITLStore()
    await store.store_request(_make_request("rid-bad"), workflow_id="wf")
    approve_token = sign_action("rid-bad", "approve", secret=SECRET)
    # state claims action=reject but carries an approve-bound token -> rejected
    state = urllib.parse.quote(f"rid-bad|reject|{approve_token}")

    async with _client(_app(store)) as ac:
        resp = await ac.get(f"/hitl/auth/callback?code=C&state={state}")

    assert resp.status_code == 403
    assert (await store.get_request("rid-bad")).status == "pending"
