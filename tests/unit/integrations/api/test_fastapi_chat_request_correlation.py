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

"""Focused tests for chat request correlation in the FastAPI server."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi.testclient import TestClient

from victor.framework.session_config import SessionConfig
from victor.integrations.api import fastapi_server
from victor.integrations.api.event_bridge import BridgeEvent, BridgeEventType
from victor.observability.request_correlation import get_request_correlation_id


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.chat_request_ids: list[str | None] = []
        self.stream_request_ids: list[str | None] = []

    async def chat(self, message: str) -> SimpleNamespace:
        self.chat_request_ids.append(get_request_correlation_id())
        return SimpleNamespace(content=f"echo:{message}", tool_calls=[])

    async def stream_chat(self, message: str):
        self.stream_request_ids.append(get_request_correlation_id())
        yield SimpleNamespace(content=f"stream:{message}", tool_calls=None)
        yield SimpleNamespace(content="", tool_calls=[{"name": "graph"}])

    async def graceful_shutdown(self) -> None:
        return None


def _create_server(monkeypatch, tmp_path: Path, orchestrator: _FakeOrchestrator):
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
    )
    server._orchestrator = orchestrator
    return server


def test_events_websocket_subscribe_acknowledges_correlation_id(
    monkeypatch, tmp_path: Path
) -> None:
    """The events websocket should acknowledge correlation-scoped subscriptions."""
    server = _create_server(monkeypatch, tmp_path, _FakeOrchestrator())

    with TestClient(server.app) as client:
        with client.websocket_connect("/ws/events") as websocket:
            websocket.send_json(
                {
                    "type": "subscribe",
                    "categories": ["tool.complete", "tool.error"],
                    "correlation_id": "chat_req_789",
                }
            )
            response = websocket.receive_json()

    assert response == {
        "type": "subscribed",
        "categories": ["tool.complete", "tool.error"],
        "correlation_id": "chat_req_789",
    }


@pytest.mark.asyncio
async def test_recent_events_endpoint_filters_by_correlation_id(
    monkeypatch, tmp_path: Path
) -> None:
    """Recent-event snapshots should return only matching request timeline events."""
    server = _create_server(monkeypatch, tmp_path, _FakeOrchestrator())
    broadcaster = fastapi_server.EventBroadcaster()
    broadcaster._recent_events.clear()
    broadcaster.broadcast_sync(
        BridgeEvent(
            type=BridgeEventType.TOOL_COMPLETE,
            data={"tool_name": "graph", "correlation_id": "chat_req_keep"},
        )
    )
    broadcaster.broadcast_sync(
        BridgeEvent(
            type=BridgeEventType.TOOL_ERROR,
            data={"tool_name": "graph", "correlation_id": "chat_req_skip"},
        )
    )

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/events/recent",
            params=[
                ("limit", "5"),
                ("correlation_id", "chat_req_keep"),
                ("categories", "tool.complete"),
            ],
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["events"][0]["type"] == "tool.complete"
    assert payload["events"][0]["data"]["correlation_id"] == "chat_req_keep"


@pytest.mark.asyncio
async def test_chat_assigns_request_id_header_and_context(monkeypatch, tmp_path: Path) -> None:
    """POST /chat should expose and use a stable per-request correlation ID."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    request_id = response.headers["x-victor-request-id"]
    assert request_id.startswith("chat_")
    assert orchestrator.chat_request_ids == [request_id]


@pytest.mark.asyncio
async def test_chat_stream_emits_request_event_and_context(monkeypatch, tmp_path: Path) -> None:
    """POST /chat/stream should align SSE request IDs with backend correlation IDs."""
    orchestrator = _FakeOrchestrator()
    server = _create_server(monkeypatch, tmp_path, orchestrator)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"messages": [{"role": "user", "content": "trace main"}]},
        ) as response:
            request_id = response.headers["x-victor-request-id"]
            payloads: list[str] = []
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line.removeprefix("data: ").strip()
                payloads.append(payload)
                if payload == "[DONE]":
                    break

    assert response.status_code == 200
    assert request_id.startswith("chat_")
    assert orchestrator.stream_request_ids == [request_id]

    parsed_events = [json.loads(payload) for payload in payloads if payload != "[DONE]"]

    assert parsed_events[0] == {"type": "request", "request_id": request_id}
    assert parsed_events[1]["type"] == "content"
    assert parsed_events[1]["request_id"] == request_id
    assert parsed_events[2]["type"] == "tool_call"
    assert parsed_events[2]["request_id"] == request_id


@pytest.mark.asyncio
async def test_chat_stream_prefers_execution_context_chat_runtime_over_legacy_attr(
    monkeypatch, tmp_path: Path
) -> None:
    """Streaming chat should resolve the canonical execution-context runtime first."""

    class _LegacyChatRuntime:
        async def stream_chat(self, _message: str):
            raise AssertionError("legacy orchestrator-bound chat runtime should not be used")

    class _RuntimeChatService:
        def __init__(self) -> None:
            self.stream_request_ids: list[str | None] = []

        async def stream_chat(self, message: str):
            self.stream_request_ids.append(get_request_correlation_id())
            yield SimpleNamespace(content=f"runtime:{message}", tool_calls=None)

    orchestrator = _FakeOrchestrator()
    runtime_chat = _RuntimeChatService()
    orchestrator._execution_context = SimpleNamespace(services=SimpleNamespace(chat=runtime_chat))
    orchestrator._chat_service = _LegacyChatRuntime()
    server = _create_server(monkeypatch, tmp_path, orchestrator)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"messages": [{"role": "user", "content": "prefer runtime"}]},
        ) as response:
            payloads: list[str] = []
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line.removeprefix("data: ").strip()
                payloads.append(payload)
                if payload == "[DONE]":
                    break

    request_event = json.loads(payloads[0])
    content_event = json.loads(payloads[1])

    assert response.status_code == 200
    assert request_event["type"] == "request"
    assert runtime_chat.stream_request_ids == [request_event["request_id"]]
    assert content_event["content"] == "runtime:prefer runtime"


@pytest.mark.asyncio
async def test_fastapi_server_uses_framework_client_factory_for_conversation_access(
    monkeypatch, tmp_path: Path
) -> None:
    """Conversation helpers should build their client through the framework seam."""
    fake_client = SimpleNamespace(initialize=AsyncMock(return_value=SimpleNamespace()))

    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )

    with patch(
        "victor.framework.session_runner.create_victor_client",
        return_value=fake_client,
    ) as create_client:
        server = fastapi_server.VictorFastAPIServer(
            workspace_root=str(tmp_path),
            enable_graphql=False,
        )

        client = await server._get_victor_client()
        cached_client = await server._get_victor_client()

    assert client is fake_client
    assert cached_client is fake_client
    create_client.assert_called_once()
    config = create_client.call_args.args[0]
    assert isinstance(config, SessionConfig)
    assert create_client.call_args.kwargs["container"] is server._container
    fake_client.initialize.assert_awaited_once_with()
