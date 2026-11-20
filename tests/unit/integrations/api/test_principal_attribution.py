# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""FOU-A4a/A4b: VerifiedPrincipal resolution + centralized FEP-0020 attribution.

A4a (design: docs/architecture/fou-a4-rbac-sso-design.md §A4a): every
authenticated request carries a ``VerifiedPrincipal`` resolved INSIDE the
auth dependency choke point (``_make_auth_dependency`` /
``_make_ws_auth_dependency`` / the ``/ws`` auth frame) — never an ASGI
middleware. Registry-less keys (Decision D4 pending) synthesize a legacy
principal whose ``principal_id`` IS today's ``client_id`` label, so
key-only deployments are behavior-neutral: same status codes, same
responses, same owner scoping — only ``request.state`` gains the principal.

A4b (design §A4b): the per-route ``bind_attribution`` re-verification
(chat/classify routes + the ``/ws`` turn) is replaced by ONE binding inside
the A4a dependency. ``group_id`` finally gets a producer (principal groups;
unset for legacy keys). FEP-0020 best-effort rule is pinned: attribution
failure NEVER fails the request path.

Dormancy guard (FEP-0041 class, mirrors the #1955 pattern): the
attribution tests below assert the subject/group observed by the REAL
orchestrator/provider during a REAL HTTP/WS request through the REAL
dependency path — no test calls ``bind_attribution`` directly to make the
assertion pass. If the dependency wiring ever goes dormant again, these go
RED, because the route handlers no longer bind anything.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import APIRouter, Request
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from victor.core.context import get_auth_group_id, get_auth_subject_id
from victor.integrations.api import fastapi_server
from victor.integrations.api.auth.principal import (
    AuthMethod,
    PrincipalKind,
    VerifiedPrincipal,
    bind_principal_attribution,
    legacy_api_key_principal,
)


class _FakeOrchestrator:
    """Captures the attribution identity observed during chat execution."""

    def __init__(self) -> None:
        self.chat_identities: list[tuple[str | None, str | None]] = []
        self.stream_identities: list[tuple[str | None, str | None]] = []

    async def chat(self, message: str) -> SimpleNamespace:
        self.chat_identities.append((get_auth_subject_id(), get_auth_group_id()))
        return SimpleNamespace(content=f"echo:{message}", tool_calls=[])

    async def stream_chat(self, message: str):
        self.stream_identities.append((get_auth_subject_id(), get_auth_group_id()))
        yield SimpleNamespace(content=f"stream:{message}", tool_calls=None)

    async def graceful_shutdown(self) -> None:
        return None

    async def resume(self, run_id: str, decision) -> SimpleNamespace:
        self.chat_identities.append((get_auth_subject_id(), get_auth_group_id()))
        return SimpleNamespace(content="resumed", tool_calls=[], status="ok")


def _grouped_principal(client_id: str = "alice", group: str = "eng") -> VerifiedPrincipal:
    """A registry-shaped principal (the D4 registry is pending; the shape is
    not) used to prove the group producer end-to-end through the dependency."""
    return VerifiedPrincipal(
        principal_id=client_id,
        kind=PrincipalKind.SERVICE,
        auth_method=AuthMethod.API_KEY,
        groups=(group,),
    )


def _create_server(monkeypatch, tmp_path: Path, orchestrator, **server_kwargs):
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        allow_unauthenticated=True,
        **server_kwargs,
    )
    server._orchestrator = orchestrator
    return server


def _add_principal_probe(server) -> None:
    """Mount a probe route through the REAL ``_include_with_auth`` choke
    point so the test observes exactly what production routes observe."""
    probe = APIRouter()

    @probe.get("/__principal_probe")
    async def probe_principal(http_request: Request) -> dict:
        state = http_request.state
        principal = getattr(state, "verified_principal", None)
        return {
            "client_id": getattr(state, "client_id", None),
            "principal_id": getattr(principal, "principal_id", None),
            "kind": getattr(principal, "kind", None),
            "auth_method": getattr(principal, "auth_method", None),
            "groups": list(getattr(principal, "groups", ()) or ()),
            "credential_id": getattr(principal, "credential_id", None),
            "issuer": getattr(principal, "issuer", None),
        }

    server._include_with_auth(probe)


# =============================================================================
# A4a — legacy principal shape (registry-less synthesis)
# =============================================================================


def test_legacy_principal_shape():
    """A registry-less key synthesizes the legacy principal: principal_id IS
    the client_id label, kind=service, method=api_key, no groups, and the
    credential_id is a public digest — never the raw secret."""
    principal = legacy_api_key_principal(api_key="sk-alice-secret", client_id="alice")

    assert principal.principal_id == "alice"
    assert principal.kind is PrincipalKind.SERVICE
    assert principal.auth_method is AuthMethod.API_KEY
    assert principal.groups == ()
    assert principal.primary_group is None
    assert principal.issuer is None
    assert principal.scopes == frozenset()
    assert principal.expires_at is None
    # Compatibility shim: today's client_id semantics survive verbatim.
    assert principal.client_id == "alice"
    # D7: key-only attribution subject keeps the legacy key label.
    assert principal.subject_label == "alice"
    # credential_id is a public key ID, never the presented secret.
    assert "sk-alice-secret" not in (principal.credential_id or "")
    assert (principal.credential_id or "").startswith("sha256:")


def test_principal_is_immutable():
    principal = legacy_api_key_principal(api_key="k", client_id="alice")
    with pytest.raises(FrozenInstanceError):
        principal.principal_id = "bob"  # type: ignore[misc]


# =============================================================================
# A4a — principal exposed through the auth dependency (HTTP)
# =============================================================================


@pytest.mark.asyncio
async def test_keyed_request_exposes_verified_principal_on_request_state(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})
    _add_principal_probe(server)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/__principal_probe", headers={"Authorization": "Bearer sk-alice"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["principal_id"] == "alice"
    assert body["client_id"] == "alice"
    # The shim contract: request.state.client_id keeps its exact value and
    # derives from the principal, so owner-scoping consumers stay untouched.
    assert body["client_id"] == body["principal_id"]
    assert body["kind"] == "service"
    assert body["auth_method"] == "api_key"
    assert body["groups"] == []
    assert body["issuer"] is None


@pytest.mark.asyncio
async def test_unknown_key_401_unchanged(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})
    _add_principal_probe(server)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/__principal_probe", headers={"Authorization": "Bearer sk-mallory"}
        )

    assert response.status_code == 401
    assert orchestrator.chat_identities == []


@pytest.mark.asyncio
async def test_keyless_server_touches_no_principal_state(monkeypatch, tmp_path):
    """Behavior-neutral negative: on a keyless server the dependency still
    short-circuits — request.state gains NO principal and the response is
    byte-identical to pre-A4."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator)
    _add_principal_probe(server)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/__principal_probe")

    assert response.status_code == 200
    assert response.json() == {
        "client_id": None,
        "principal_id": None,
        "kind": None,
        "auth_method": None,
        "groups": [],
        "credential_id": None,
        "issuer": None,
    }


@pytest.mark.asyncio
async def test_exempt_paths_still_open_without_key(monkeypatch, tmp_path):
    """Negative: the exempt-path short-circuit must survive the principal
    wiring — /health and /status stay reachable unauthenticated."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for path in ("/health", "/status"):
            response = await client.get(path)
            assert response.status_code in (
                200,
                404,
            ), f"{path} must stay exempt, got {response.status_code}"


# =============================================================================
# A4a — principal through the WS seams
# =============================================================================


def test_ws_auth_frame_resolves_principal(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-carol": "carol"})

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "auth", "api_key": "sk-carol"})
            assert ws.receive_json()["type"] == "auth_success"
            socket = server._ws_clients[-1]
            principal = getattr(socket.state, "verified_principal", None)
            assert principal is not None
            assert principal.principal_id == "carol"
            assert principal.auth_method is AuthMethod.API_KEY
            # Shim equality with the VAS-05j owner-scoping consumer.
            assert socket.state.client_id == principal.client_id == "carol"


def test_ws_events_query_key_resolves_principal(monkeypatch, tmp_path):
    server = _create_server(
        monkeypatch, tmp_path, _FakeOrchestrator(), api_keys={"sk-alice": "alice"}
    )

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/events?api_key=sk-alice"):
            socket = server._event_clients[-1]
            principal = getattr(socket.state, "verified_principal", None)
            assert principal is not None
            assert principal.principal_id == "alice"
            assert socket.state.client_id == "alice"


# =============================================================================
# A4b — attribution through the dependency (dormancy-guarded)
# =============================================================================


@pytest.mark.asyncio
async def test_chat_binds_principal_through_real_dependency(monkeypatch, tmp_path):
    """Dormancy guard: the route handler no longer binds attribution — the
    subject the orchestrator observes can only come from the dependency."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-alice"},
        )

    assert response.status_code == 200
    assert orchestrator.chat_identities == [("alice", None)]


@pytest.mark.asyncio
async def test_chat_resume_binds_principal_through_real_dependency(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})
    # Resume runs against the VictorClient, not the legacy adapter — expose
    # the orchestrator (which has `resume`) as the client directly.
    server._victor_client = orchestrator

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/resume",
            json={"run_id": "run-1", "approved": True},
            headers={"Authorization": "Bearer sk-alice"},
        )

    assert response.status_code == 200
    assert orchestrator.chat_identities == [("alice", None)]


@pytest.mark.asyncio
async def test_stream_binds_principal_through_real_dependency(monkeypatch, tmp_path):
    """Dormancy guard, strongest form: the binding must stay alive inside the
    streaming body generator — produced AFTER the route handler returned and
    where the per-route bind used to live."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-bob": "bob"})

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-bob"},
        ) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.strip() == "data: [DONE]":
                    break

    assert orchestrator.stream_identities == [("bob", None)]


@pytest.mark.asyncio
async def test_grouped_principal_binds_group_through_dependency(monkeypatch, tmp_path):
    """group_id finally gets a producer: principal groups flow into the
    attribution context (the legacy registry-less path leaves it unset)."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})
    monkeypatch.setattr(
        server, "_principal_for_key", lambda raw_key, client_id: _grouped_principal()
    )

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-alice"},
        )

    assert response.status_code == 200
    assert orchestrator.chat_identities == [("alice", "eng")]


@pytest.mark.asyncio
async def test_legacy_keys_bind_subject_with_group_unset(monkeypatch, tmp_path):
    """Registry-less keys keep D7 subject semantics and bind NO group — the
    pre-A4 subject value, byte-identical, plus nothing else."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-alice"},
        )

    assert orchestrator.chat_identities == [("alice", None)]


def test_ws_turn_attribution_uses_principal_group(monkeypatch, tmp_path):
    """The /ws turn binding is fed from the auth-frame principal: subject AND
    group flow into the turn's attribution context."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-carol": "carol"})
    monkeypatch.setattr(
        server,
        "_principal_for_key",
        lambda raw_key, client_id: _grouped_principal(client_id="carol"),
    )

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "auth", "api_key": "sk-carol"})
            assert ws.receive_json()["type"] == "auth_success"
            ws.send_json({"type": "chat", "messages": [{"role": "user", "content": "hi"}]})
            done = False
            while not done:
                msg = ws.receive_json()
                done = msg["type"] in {"done", "error"}
            assert msg["type"] == "done", msg

    assert orchestrator.stream_identities == [("carol", "eng")]


@pytest.mark.asyncio
async def test_classify_binds_principal_through_real_dependency(monkeypatch, tmp_path):
    """The classify route's re-verification bind is deduped away — the
    provider must observe the subject through the dependency's binding."""
    captured: dict = {}

    class _Provider:
        model = "fake-model"

        async def chat(self, messages, *, model, temperature, max_tokens, **kwargs):
            captured["identity"] = (get_auth_subject_id(), get_auth_group_id())
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "sensitive": True,
                        "category": "money",
                        "reason": "payment confirmation requested",
                        "confidence": 0.9,
                    }
                ),
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                model="fake-model",
            )

        async def close(self):
            return None

    import victor.providers.factory as factory_mod

    orchestrator = SimpleNamespace(
        provider_manager=SimpleNamespace(current_provider=_Provider()),
        model="fake-model",
    )
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-hub": "hub"})

    async def fake_factory(provider_name, model, **kwargs):
        return _Provider()

    monkeypatch.setattr(factory_mod.ManagedProviderFactory, "create", staticmethod(fake_factory))

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/classify",
            json={"input": "x", "preset": "triage.v1"},
            headers={"Authorization": "Bearer sk-hub"},
        )

    assert response.status_code == 200, response.text
    assert captured["identity"] == ("hub", None)


@pytest.mark.asyncio
async def test_attribution_failure_never_fails_the_request(monkeypatch, tmp_path):
    """FEP-0020 best-effort rule: a binding failure inside the dependency
    must leave the request fully served — never an error response."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})

    import victor.integrations.api.auth.principal as principal_mod

    def exploding_bind(subject_id=None, group_id=None):
        raise RuntimeError("attribution backend down")

    monkeypatch.setattr(principal_mod, "bind_attribution", exploding_bind)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-alice"},
        )

    assert response.status_code == 200
    assert response.json()["content"] == "echo:hello"
    assert orchestrator.chat_identities != [], "chat must still execute"


def test_ws_turn_attribution_failure_never_fails_the_turn(monkeypatch, tmp_path):
    """Best-effort on the WS turn path too: a broken binding still streams."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-carol": "carol"})

    import victor.integrations.api.auth.principal as principal_mod

    monkeypatch.setattr(
        principal_mod,
        "bind_attribution",
        lambda subject_id=None, group_id=None: (_ for _ in ()).throw(
            RuntimeError("attribution backend down")
        ),
    )

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "auth", "api_key": "sk-carol"})
            assert ws.receive_json()["type"] == "auth_success"
            ws.send_json({"type": "chat", "messages": [{"role": "user", "content": "hi"}]})
            done = False
            while not done:
                msg = ws.receive_json()
                done = msg["type"] in {"done", "error"}

    assert msg["type"] == "done", msg
    assert orchestrator.stream_identities, "the turn must still execute"


@pytest.mark.asyncio
async def test_keyless_attribution_unchanged_byte_identical(monkeypatch, tmp_path):
    """Negative: a keyless server binds NO identity (config defaults own
    attribution, exactly as pre-A4) and the response body is unchanged."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    assert response.json() == {
        "role": "assistant",
        "content": "echo:hello",
        "tool_calls": [],
        "status": "ok",
        "run_id": None,
        "approval_request": None,
        "resume_thread_id": None,
    }
    assert orchestrator.chat_identities == [(None, None)]


@pytest.mark.asyncio
async def test_concurrent_keyed_requests_isolate_attribution(monkeypatch, tmp_path):
    """Two concurrent keyed requests must each attribute to their own
    principal — the context-bound identity never crosses requests. The
    single-trusted-operator declaration (#1981) attests both keys belong to
    one operator so the posture allows both onto the shared chat surface."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(
        monkeypatch,
        tmp_path,
        orchestrator,
        api_keys={"sk-alice": "alice", "sk-bob": "bob"},
        single_operator_mode=True,
    )

    message_by_subject: dict[str, str] = {}

    async def capture_chat(message: str) -> SimpleNamespace:
        subject = get_auth_subject_id()
        message_by_subject[subject or ""] = message
        return SimpleNamespace(content=f"echo:{message}", tool_calls=[])

    orchestrator.chat = capture_chat  # type: ignore[method-assign]

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(
            client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "from-alice"}]},
                headers={"Authorization": "Bearer sk-alice"},
            ),
            client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "from-bob"}]},
                headers={"Authorization": "Bearer sk-bob"},
            ),
        )

    assert [r.status_code for r in responses] == [200, 200]
    assert message_by_subject == {"alice": "from-alice", "bob": "from-bob"}


@pytest.mark.asyncio
async def test_bound_group_reaches_cost_rows(monkeypatch, tmp_path):
    """The bound group joins the cost pipeline: SessionCostTracker rows pick
    up subject+group from the dependency-bound context (the values that feed
    sandhi_meter virtual keys / neutral events)."""
    from victor.agent.session_cost_tracker import SessionCostTracker

    principal = _grouped_principal()
    tracker = SessionCostTracker()
    with bind_principal_attribution(principal):
        request = tracker.begin_request()
    assert request.subject_id == "alice"
    assert request.group_id == "eng"

    tracker2 = SessionCostTracker()
    request2 = tracker2.begin_request()
    assert request2.subject_id is None
    assert request2.group_id is None
