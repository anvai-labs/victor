# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""FEP-0020 attribution join at the API-server auth seam.

``_verify_api_key`` already maps ``Authorization: Bearer <key>`` to a
``client_id`` — these tests pin that the resolved identity is bound to the
execution context for the duration of the chat call, so the cost/usage
records emitted downstream (SessionCostTracker -> SandhiMeter) carry the
authenticated subject instead of only the operator-level config default.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi.testclient import TestClient

from victor.core.context import get_auth_subject_id
from victor.integrations.api import fastapi_server


class _FakeOrchestrator:
    """Captures the attribution subject observed during chat execution."""

    def __init__(self) -> None:
        self.chat_subjects: list[str | None] = []
        self.stream_subjects: list[str | None] = []

    async def chat(self, message: str) -> SimpleNamespace:
        self.chat_subjects.append(get_auth_subject_id())
        return SimpleNamespace(content=f"echo:{message}", tool_calls=[])

    async def stream_chat(self, message: str):
        self.stream_subjects.append(get_auth_subject_id())
        yield SimpleNamespace(content=f"stream:{message}", tool_calls=None)

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


@pytest.mark.asyncio
async def test_chat_binds_authenticated_client_id_as_subject(monkeypatch, tmp_path):
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
    assert orchestrator.chat_subjects == ["alice"]


@pytest.mark.asyncio
async def test_chat_without_key_is_rejected_when_auth_configured(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-alice": "alice"})

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 401
    assert orchestrator.chat_subjects == []


@pytest.mark.asyncio
async def test_chat_unauthenticated_server_binds_no_subject(monkeypatch, tmp_path):
    """No api_keys configured (CLI/local use) -> no identity bound; config default applies."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    assert orchestrator.chat_subjects == [None]


@pytest.mark.asyncio
async def test_chat_stream_binds_subject_for_streamed_turn(monkeypatch, tmp_path):
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

    assert orchestrator.stream_subjects == ["bob"]


@pytest.mark.asyncio
async def test_chat_stream_rejects_invalid_key(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-bob": "bob"})

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong"},
        )

    assert response.status_code == 401
    assert orchestrator.stream_subjects == []


def test_websocket_chat_binds_authenticated_subject(monkeypatch, tmp_path):
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator, api_keys={"sk-carol": "carol"})

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "auth", "api_key": "sk-carol"})
            assert ws.receive_json()["type"] == "auth_success"
            ws.send_json({"type": "chat", "messages": [{"role": "user", "content": "hi"}]})
            done = False
            while not done:
                msg = ws.receive_json()
                done = msg["type"] in {"done", "error"}

    assert orchestrator.stream_subjects == ["carol"]
