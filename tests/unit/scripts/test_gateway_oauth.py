"""Credential-boundary tests; no model calls or live acceptance claims."""

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
import signal
import subprocess
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.validation.gateway_oauth import GatewayOAuth, GatewayOAuthError

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX validation broker process groups")


@pytest.fixture
def profile(tmp_path):
    broker = tmp_path / "broker.py"
    broker.write_text(
        "import json, pathlib, sys, time\n"
        "p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text())\n"
        "c=p.with_suffix('.count'); c.write_text(str(int(c.read_text())+1) if c.exists() else '1')\n"
        "time.sleep(d.pop('delay',0))\n"
        "if d.pop('exit',False): print('secret error',file=sys.stderr); sys.exit(1)\n"
        "print(json.dumps(d))\n"
    )
    identities = {}
    for name in ("member", "accounting"):
        token = tmp_path / (name + ".json")
        token.write_text(
            json.dumps(
                {
                    "access_token": "private-" + name,
                    "token_type": "Bearer",
                    "expires_on": time.time() + 900,
                    "issuer": "https://issuer.example",
                    "audience": "sandhi",
                    "subject": name,
                }
            )
        )
        identities[name] = {
            "command": [sys.executable, str(broker), str(token)],
            "issuer": "https://issuer.example",
            "audience": "sandhi",
            "subject": name,
        }
    document = {
        "schema_version": 1,
        "gateway_url": "https://gateway.example",
        "ca_file": None,
        "database": str(tmp_path / "usage.db"),
        "credentials": identities,
        "accounting": "accounting",
        "routes": {
            "zai": {"credential": "member", "grant": "cloud", "models": ["reference"]},
        },
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(document))
    return path


class Client:
    def __init__(self, status=200):
        self.calls = []
        self.status = status

    @asynccontextmanager
    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        yield SimpleNamespace(status=self.status)


async def infer(auth, client, **overrides):
    args = {
        "method": "POST",
        "url": auth.gateway_url + "/v1/chat/completions",
        "purpose": "inference",
        "headers": {"Authorization": "Bearer " + auth.capability("zai")},
        "data": b'{"model":"reference","messages":[]}',
    }
    args.update(overrides)
    async with auth.request(client, **args) as response:
        return response.status


async def test_roles_stay_separate_and_denials_are_not_replayed(profile):
    auth = GatewayOAuth.load(profile)
    client = Client(status=403)
    headers = {
        "Authorization": "Bearer " + auth.capability("zai"),
        "Cookie": "private-browser",
        "x-api-key": "private-key",
        "x-goog-api-key": "private-key",
        "Proxy-Authorization": "private-proxy",
        "x-sandhi-grant": "admin",
        "x-sandhi-session": "member-session",
    }
    assert await infer(auth, client, headers=headers) == 403
    assert len(client.calls) == 1
    sent = client.calls[0][2]
    assert sent["data"] == b'{"model":"reference","messages":[]}'
    assert sent["allow_redirects"] is False
    assert sent["headers"] == {
        "Authorization": "Bearer private-member",
        "x-sandhi-grant": "cloud",
        "x-sandhi-session": "member-session",
        "Accept-Encoding": "identity",
    }
    async with auth.request(
        client, "GET", auth.gateway_url + "/dashboard/api/usage", purpose="accounting"
    ):
        pass
    assert client.calls[1][2]["headers"] == {
        "Authorization": "Bearer private-accounting",
        "Accept-Encoding": "identity",
    }
    assert "private-member" not in repr(auth)


@pytest.mark.parametrize(
    "fault",
    [
        "method",
        "path",
        "query",
        "encoded",
        "host",
        "model",
        "duplicate_model",
        "non_object",
        "capability",
        "duplicate_auth",
    ],
)
async def test_invalid_member_route_is_rejected_before_acquiring_a_token(profile, fault):
    from multidict import CIMultiDict

    auth, client = GatewayOAuth.load(profile), Client()
    change = {
        "method": {"method": "GET"},
        "path": {"url": "https://gateway.example/admin/usage/diagnostics"},
        "query": {"url": "https://gateway.example/v1/chat/completions?x=1"},
        "encoded": {"url": "https://gateway.example/v1/chat/%63ompletions"},
        "host": {"url": "https://other.example/v1/chat/completions"},
        "model": {"data": b'{"model":"other"}'},
        "duplicate_model": {"data": b'{"model":"other","model":"reference"}'},
        "non_object": {"data": b"[]"},
        "capability": {"headers": {"Authorization": "Bearer forged"}},
        "duplicate_auth": {
            "headers": CIMultiDict(
                [
                    ("Authorization", "Bearer " + auth.capability("zai")),
                    ("authorization", "Bearer forged"),
                ]
            )
        },
    }[fault]
    with pytest.raises(GatewayOAuthError):
        await infer(auth, client, **change)
    assert client.calls == []
    assert not (profile.parent / "member.count").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("subject", "other"),
        ("audience", "other"),
        ("issuer", "https://other.example"),
        ("token_type", "ID"),
        ("access_token", "bad\r\nheader"),
        ("expires_on", float("nan")),
        ("expires_on", True),
        ("expires_on", 0),
        ("unknown", "secret-body"),
    ],
)
async def test_invalid_broker_contract_is_redacted_and_never_sent(profile, field, value):
    token = profile.parent / "member.json"
    document = json.loads(token.read_text())
    document[field] = value
    token.write_text(json.dumps(document))
    auth, client = GatewayOAuth.load(profile), Client()
    with pytest.raises(GatewayOAuthError) as caught:
        await infer(auth, client)
    assert "private-member" not in str(caught.value)
    assert "secret-body" not in str(caught.value)
    assert client.calls == []


async def test_existing_cache_renews_before_request_without_replaying_http(profile, monkeypatch):
    from victor.core.identity import protocols

    auth, client = GatewayOAuth.load(profile), Client()
    await asyncio.gather(*(infer(auth, client) for _ in range(4)))
    assert (profile.parent / "member.count").read_text() == "1"
    old = json.loads((profile.parent / "member.json").read_text())
    future = old["expires_on"] - 100
    old.update(access_token="private-renewed", expires_on=future + 900)
    (profile.parent / "member.json").write_text(json.dumps(old))
    monkeypatch.setattr(protocols.time, "time", lambda: future)
    await infer(auth, client)
    assert (profile.parent / "member.count").read_text() == "2"
    assert len(client.calls) == 5
    assert client.calls[-1][2]["headers"]["Authorization"] == "Bearer private-renewed"


@pytest.mark.parametrize("fault", ["identity_alias", "http", "extra", "unknown_credential"])
def test_profile_rejects_ambiguous_or_insecure_configuration(profile, fault):
    document = json.loads(profile.read_text())
    if fault == "identity_alias":
        document["credentials"]["accounting"]["subject"] = "member"
    elif fault == "http":
        document["gateway_url"] = "http://gateway.example"
    elif fault == "extra":
        document["skip_tls_verification"] = True
    else:
        document["routes"]["zai"]["credential"] = "missing"
    profile.write_text(json.dumps(document))
    with pytest.raises(GatewayOAuthError):
        GatewayOAuth.load(profile)


@pytest.mark.parametrize(
    "method,path",
    [("POST", "/admin/budgets"), ("GET", "/admin/config"), ("POST", "/v1/chat/completions")],
)
async def test_accounting_lane_cannot_administer_or_infer(profile, method, path):
    auth, client = GatewayOAuth.load(profile), Client()
    with pytest.raises(GatewayOAuthError):
        async with auth.request(client, method, auth.gateway_url + path, purpose="accounting"):
            pass
    assert client.calls == []
    assert not (profile.parent / "accounting.count").exists()


@pytest.mark.parametrize(
    "cancel,leader_exits", [(False, False), (True, False), (False, True), (True, True)]
)
async def test_broker_deadline_and_cancellation_bound_inherited_pipes(
    profile, monkeypatch, cancel, leader_exits
):
    from scripts.validation import gateway_oauth

    marker = profile.parent / "child.pid"
    broker = profile.parent / "descendant.py"
    broker.write_text(
        "import pathlib, subprocess, sys, time\n"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(3)'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid))\n"
        + ("" if leader_exits else "time.sleep(3)\n")
    )
    document = json.loads(profile.read_text())
    document["credentials"]["member"]["command"] = [sys.executable, str(broker)]
    profile.write_text(json.dumps(document))
    monkeypatch.setattr(gateway_oauth, "ACQUISITION_TIMEOUT", 0.3, raising=False)
    auth, client = GatewayOAuth.load(profile), Client()
    started = time.monotonic()
    operation = asyncio.create_task(infer(auth, client))
    try:
        if cancel:
            async with asyncio.timeout(2):
                while not marker.exists():
                    await asyncio.sleep(0.01)
            operation.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else GatewayOAuthError):
            await operation
        assert time.monotonic() - started < 2
        assert client.calls == []
        if marker.exists() and not leader_exits:
            status = subprocess.run(
                ["ps", "-p", marker.read_text(), "-o", "stat="],
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
            assert not status or status.startswith("Z"), "broker descendant survived cleanup"
    finally:
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("fault", ["timeout", "signal", "close"])
async def test_cleanup_failure_preserves_cancellation_and_finishes_other_stages(monkeypatch, fault):
    from unittest.mock import AsyncMock, Mock
    from scripts.validation import gateway_oauth

    process = SimpleNamespace(
        returncode=None if fault == "signal" else 0,
        pid=12345,
        stdout=SimpleNamespace(read=AsyncMock(side_effect=asyncio.CancelledError())),
        wait=AsyncMock(side_effect=asyncio.TimeoutError() if fault == "timeout" else None),
        _transport=Mock(),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    signal_group = Mock(
        side_effect=PermissionError("private-detail") if fault == "signal" else None
    )
    if fault == "close":
        process._transport.close.side_effect = OSError("private-detail")
    monkeypatch.setattr(os, "killpg", signal_group)
    source = gateway_oauth.BrokerCredential(
        {
            "command": ["/trusted/broker"],
            "issuer": "issuer",
            "audience": "audience",
            "subject": "subject",
        }
    )
    with pytest.raises(asyncio.CancelledError) as caught:
        await source.get_token("audience")
    if fault == "signal":
        signal_group.assert_called_once()
    else:
        signal_group.assert_not_called()
    process._transport.close.assert_called_once()
    process.wait.assert_awaited_once()
    assert len(caught.value.__notes__) == 1
    assert "private-detail" not in caught.value.__notes__[0]
