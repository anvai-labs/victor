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
