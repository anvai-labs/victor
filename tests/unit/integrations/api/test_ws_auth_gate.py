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

"""WebSocket chat auth gate.

The `auth` message set ``ws.state.authenticated`` but nothing enforced it —
a chat message from an unauthenticated client ran full agent turns when API
keys were configured (co-design review U7-F1).

Outbound delivery (VAS-05h): ``/ws`` registers sockets in the broadcast
recipient set before message-level authentication, and ``_broadcast_ws``
sent protected workflow/team/tool events (including approval resolutions)
to every registered socket without checking authentication.
"""

from __future__ import annotations

import json
import time

import anyio
import pytest
from fastapi import WebSocketDisconnect

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from victor.integrations.api import fastapi_server


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.streamed: list[str] = []

    async def chat(self, message: str) -> SimpleNamespace:
        return SimpleNamespace(content=f"echo:{message}", tool_calls=[])

    async def stream_chat(self, message: str):
        self.streamed.append(message)
        yield SimpleNamespace(content=f"reply:{message}", metadata={})

    async def graceful_shutdown(self) -> None:
        return None


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


def test_ws_chat_rejected_before_auth_when_keys_configured(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "chat", "messages": [{"role": "user", "content": "hi"}]})
            response = ws.receive_json()

    assert response["type"] == "error"
    assert "auth" in response["message"].lower()
    assert orchestrator.streamed == [], "unauthenticated chat must not execute"


def test_ws_chat_allowed_after_auth_message(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "auth", "api_key": "sk-alice"})
            assert ws.receive_json()["type"] == "auth_success"

            ws.send_json({"type": "chat", "messages": [{"role": "user", "content": "hi"}]})
            done = False
            while not done:
                msg = ws.receive_json()
                done = msg["type"] in {"done", "error"}
            assert msg["type"] == "done", msg

    assert orchestrator.streamed == ["hi"]


def test_ws_chat_open_on_keyless_server(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator)

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "chat", "messages": [{"role": "user", "content": "hi"}]})
            done = False
            while not done:
                msg = ws.receive_json()
                done = msg["type"] in {"done", "error"}
            assert msg["type"] == "done", msg

    assert orchestrator.streamed == ["hi"]


def test_ws_wrong_key_then_chat_rejected(monkeypatch, tmp_path):
    """Negative: a failed auth attempt must not count as authenticated."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "auth", "api_key": "sk-mallory"})
            assert ws.receive_json()["type"] == "auth_failed"

            ws.send_json({"type": "chat", "messages": [{"role": "user", "content": "hi"}]})
            response = ws.receive_json()
    assert response["type"] == "error"
    assert orchestrator.streamed == []


def test_ws_auth_state_not_shared_across_connections(monkeypatch, tmp_path):
    """Negative: authenticating on connection A must not authenticate
    connection B (state is per-socket)."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws_a:
            ws_a.send_json({"type": "auth", "api_key": "sk-alice"})
            assert ws_a.receive_json()["type"] == "auth_success"

        with client.websocket_connect("/ws") as ws_b:
            ws_b.send_json({"type": "chat", "messages": [{"role": "user", "content": "hi"}]})
            response = ws_b.receive_json()
    assert response["type"] == "error"
    assert "auth" in response["message"].lower()
    assert orchestrator.streamed == []


def test_ws_events_guarded_when_keys_configured(monkeypatch, tmp_path):
    """Negative: the /ws/events observability stream must not accept
    unauthenticated subscriptions on a keyed server. Found by adversarial
    review."""
    server = _create_server(
        monkeypatch, tmp_path, _FakeOrchestrator(), api_keys={"sk-alice": "alice"}
    )

    with TestClient(server.app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/events") as ws:
                ws.send_json({"type": "subscribe", "categories": ["all"]})
                ws.receive_json()

        # With a key, the socket is accepted.
        with client.websocket_connect("/ws/events?api_key=sk-alice") as ws:
            ws.send_json({"type": "subscribe", "categories": ["all"]})
            ack = ws.receive_json()
            assert ack.get("type") == "subscribed"


# ---------------------------------------------------------------------------
# Outbound broadcast delivery (VAS-05h)
# ---------------------------------------------------------------------------


def _receive_bounded(client: TestClient, ws, timeout: float = 5.0):
    """Receive one WebSocket message with a hard time bound.

    The stock test-session receive blocks forever; these tests must fail
    rather than hang when the server never sends (or never closes).
    """

    async def _recv():
        with anyio.fail_after(timeout):
            return await ws._send_rx.receive()

    message = client.portal.call(_recv)
    ws._raise_on_close(message)
    return json.loads(message["text"])


def _seed_pending_approval(server, approval_id: str) -> str:
    server._pending_tool_approvals[approval_id] = {
        "tool_name": "edit",
        "approved": False,
        "resolved": False,
    }
    return approval_id


def test_protected_broadcast_withheld_from_unauthenticated_ws(monkeypatch, tmp_path):
    """A passive unauthenticated /ws socket must not receive protected events.

    Real route, real delivery path: an authenticated HTTP route (tool
    approval resolution) broadcasts through _broadcast_ws while one
    authenticated and one pending (unauthenticated) socket are registered.
    Ping probes give a bounded, order-deterministic assertion: the pong
    must be the FIRST message the pending socket sees.
    """
    server = _create_server(
        monkeypatch, tmp_path, _FakeOrchestrator(), api_keys={"sk-alice": "alice"}
    )
    approval_id = _seed_pending_approval(server, "apr-vas05h")

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws_pending:
            with client.websocket_connect("/ws") as ws_auth:
                ws_auth.send_json({"type": "auth", "api_key": "sk-alice"})
                assert ws_auth.receive_json()["type"] == "auth_success"

                resp = client.post(
                    "/tools/approve",
                    json={"approval_id": approval_id, "approved": True},
                    headers={"Authorization": "Bearer sk-alice"},
                )
                assert resp.status_code == 200, resp.text

                # Positive control: the authorized recipient receives the
                # protected event before its pong.
                ws_auth.send_json({"type": "ping"})
                first = _receive_bounded(client, ws_auth)
                assert first["type"] == "tool_approval_resolved", first
                assert first["approval_id"] == approval_id
                assert ws_auth.receive_json()["type"] == "pong"

                # Negative: the pending socket's first message must be the
                # pong — no broadcast may precede it.
                ws_pending.send_json({"type": "ping"})
                msg = _receive_bounded(client, ws_pending)
                assert (
                    msg["type"] == "pong"
                ), f"unauthenticated socket received protected event: {msg}"


def test_protected_broadcast_still_reaches_authenticated_recipients(monkeypatch, tmp_path):
    """Delivery to authenticated sockets must survive the pending-socket fix
    (a keyless server broadcasts to all sockets as before)."""
    server = _create_server(
        monkeypatch, tmp_path, _FakeOrchestrator(), api_keys={"sk-alice": "alice"}
    )
    approval_id = _seed_pending_approval(server, "apr-positive")

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws_auth:
            ws_auth.send_json({"type": "auth", "api_key": "sk-alice"})
            assert ws_auth.receive_json()["type"] == "auth_success"

            resp = client.post(
                "/tools/approve",
                json={"approval_id": approval_id, "approved": False},
                headers={"Authorization": "Bearer sk-alice"},
            )
            assert resp.status_code == 200, resp.text

            ws_auth.send_json({"type": "ping"})
            first = _receive_bounded(client, ws_auth)
            assert first["type"] == "tool_approval_resolved", first
            assert first["approved"] is False


def test_broadcast_reaches_all_sockets_on_keyless_server(monkeypatch, tmp_path):
    """Keyless servers keep the old behavior: every socket is a legitimate
    recipient (explicit allow_unauthenticated=True local dev)."""
    server = _create_server(monkeypatch, tmp_path, _FakeOrchestrator())
    approval_id = _seed_pending_approval(server, "apr-keyless")

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws_a:
            with client.websocket_connect("/ws") as ws_b:
                resp = client.post(
                    "/tools/approve",
                    json={"approval_id": approval_id, "approved": True},
                )
                assert resp.status_code == 200, resp.text

                for ws in (ws_a, ws_b):
                    ws.send_json({"type": "ping"})
                    first = _receive_bounded(client, ws)
                    assert first["type"] == "tool_approval_resolved", first
                    assert ws.receive_json()["type"] == "pong"


def test_unauthenticated_ws_closed_after_auth_deadline(monkeypatch, tmp_path):
    """A keyed server must close /ws sockets that never authenticate within
    a bounded handshake deadline, instead of letting them sit in the
    broadcast recipient set indefinitely."""
    server = _create_server(
        monkeypatch, tmp_path, _FakeOrchestrator(), api_keys={"sk-alice": "alice"}
    )
    server._ws_auth_deadline = 0.2

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            start = time.monotonic()
            with pytest.raises(WebSocketDisconnect):
                _receive_bounded(client, ws, timeout=5.0)
            elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"close took {elapsed:.2f}s — deadline not enforced"


def test_authenticated_ws_survives_past_auth_deadline(monkeypatch, tmp_path):
    """The handshake deadline gates only the pre-auth window; an
    authenticated socket stays connected and responsive past it."""
    server = _create_server(
        monkeypatch, tmp_path, _FakeOrchestrator(), api_keys={"sk-alice": "alice"}
    )
    server._ws_auth_deadline = 0.2

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "auth", "api_key": "sk-alice"})
            assert ws.receive_json()["type"] == "auth_success"

            time.sleep(0.4)  # well past the deadline

            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"


def test_failed_auth_still_closes_at_deadline(monkeypatch, tmp_path):
    """A socket that sends only invalid keys is still a pending socket and
    must be closed at the deadline (failed attempts do not earn an
    extension of the unauthenticated window)."""
    server = _create_server(
        monkeypatch, tmp_path, _FakeOrchestrator(), api_keys={"sk-alice": "alice"}
    )
    server._ws_auth_deadline = 0.3

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "auth", "api_key": "sk-mallory"})
            assert ws.receive_json()["type"] == "auth_failed"
            with pytest.raises(WebSocketDisconnect):
                _receive_bounded(client, ws, timeout=5.0)


def test_frame_drip_cannot_extend_auth_deadline(monkeypatch, tmp_path):
    """Adversarial: the handshake deadline is a TOTAL window from accept, not
    a per-message timeout — an unauthenticated client cannot stay pending
    indefinitely by sending frames that each reset a per-receive timer."""
    server = _create_server(
        monkeypatch, tmp_path, _FakeOrchestrator(), api_keys={"sk-alice": "alice"}
    )
    server._ws_auth_deadline = 1.0

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            # Early frame inside the window is answered normally.
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

            # Drip a frame well inside the window (a per-receive timer would
            # be reset here; a total window keeps ticking)...
            time.sleep(0.6)
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

            # ...then cross the total deadline: the next interaction must
            # be the server closing the socket, not another pong.
            time.sleep(0.6)
            with pytest.raises(WebSocketDisconnect):
                _receive_bounded(client, ws, timeout=5.0)
