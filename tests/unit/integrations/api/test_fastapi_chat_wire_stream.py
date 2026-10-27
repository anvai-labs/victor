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

"""``POST /chat/wire-stream`` — the v1 wire-event SSE surface for remote UIs.

Pins the three contract facts a remote consumer relies on: the payloads are
wire dicts (not the legacy ``type: content|tool_call`` shape), every stream
terminates with exactly one ``stream_end``, and the FEP-0020 attribution join
binds the authenticated subject for the whole streamed turn.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from victor.core.context import get_auth_subject_id
from victor.integrations.api import fastapi_server


class _WireStubClient:
    """Drives stream_sse() the way the real VictorClient does."""

    def __init__(self) -> None:
        self.subjects: list[str | None] = []

    async def stream(self, message: str):
        self.subjects.append(get_auth_subject_id())
        yield SimpleNamespace(event_type="content", content=f"wire:{message}")
        yield SimpleNamespace(
            event_type="tool_call", tool_name="read", metadata={"arguments": {"path": "x"}}
        )


def _create_server(monkeypatch, tmp_path: Path, client, **server_kwargs):
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

    async def _stub_client():
        return client

    server._get_victor_client = _stub_client  # type: ignore[method-assign]
    return server


async def _read_sse(response) -> list[dict]:
    events: list[dict] = []
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            payload = json.loads(line[len("data: ") :])
            if isinstance(payload, dict):
                events.append(payload)
    return events


@pytest.mark.asyncio
async def test_wire_stream_emits_wire_contract_and_binds_subject(monkeypatch, tmp_path):
    stub = _WireStubClient()
    server = _create_server(monkeypatch, tmp_path, stub, api_keys={"sk-carol": "carol"})

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        async with http.stream(
            "POST",
            "/chat/wire-stream",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer sk-carol"},
        ) as response:
            assert response.status_code == 200
            events = await _read_sse(response)

    kinds = [e.get("event") for e in events]
    # Request preamble, streamed content, tool_call, exactly one terminal stream_end.
    assert kinds[0] == "request"
    assert "content" in kinds
    assert "tool_call" in kinds
    assert kinds.count("stream_end") == 1
    assert kinds[-1] == "stream_end"
    content = next(e for e in events if e.get("event") == "content")
    assert content["content"] == "wire:hello"
    tool_call = next(e for e in events if e.get("event") == "tool_call")
    assert tool_call["tool"] == "read"
    # FEP-0020: the authenticated identity is bound for the streamed turn.
    assert stub.subjects == ["carol"]


@pytest.mark.asyncio
async def test_wire_stream_requires_messages_and_auth(monkeypatch, tmp_path):
    stub = _WireStubClient()
    server = _create_server(monkeypatch, tmp_path, stub, api_keys={"sk-carol": "carol"})

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        empty = await http.post(
            "/chat/wire-stream",
            json={"messages": []},
            headers={"Authorization": "Bearer sk-carol"},
        )
        assert empty.status_code == 400

        unauthenticated = await http.post(
            "/chat/wire-stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert unauthenticated.status_code == 401
    assert stub.subjects == []
