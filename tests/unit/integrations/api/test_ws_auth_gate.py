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
"""

from __future__ import annotations

import pytest

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
