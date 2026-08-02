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

"""FEP-0029 Phase 3b: the API durable-pause surface — /chat awaiting_approval + /chat/resume."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from victor.integrations.api import fastapi_server


def _server(monkeypatch, tmp_path: Path, client: Any) -> Any:
    monkeypatch.setattr(
        fastapi_server, "load_fastapi_router_registrations", lambda *, workspace_root: []
    )
    server = fastapi_server.VictorFastAPIServer(workspace_root=str(tmp_path), enable_graphql=False)
    server._victor_client = client  # bypass real client construction
    return server


class _PausingClient:
    """Chat pauses on the first turn; resume returns the continued answer (records the decision)."""

    def __init__(self) -> None:
        self.resumed: list[tuple] = []

    async def chat(self, message: str) -> SimpleNamespace:
        return SimpleNamespace(
            content="",
            tool_calls=[],
            status="awaiting_approval",
            run_id="run-abc",
            approval_request={"id": "req-1", "title": "Approve tool: run_command"},
        )

    async def resume(self, run_id: str, decision: Any) -> SimpleNamespace:
        self.resumed.append((run_id, decision.approved, decision.response))
        return SimpleNamespace(content="all done", tool_calls=[], status="ok", run_id=None)


async def _post(server: Any, path: str, body: dict) -> Any:
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, json=body)


@pytest.mark.asyncio
async def test_chat_surfaces_awaiting_approval_as_202(monkeypatch, tmp_path: Path) -> None:
    server = _server(monkeypatch, tmp_path, _PausingClient())
    resp = await _post(server, "/chat", {"messages": [{"role": "user", "content": "rm -rf x"}]})

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "awaiting_approval"
    assert body["run_id"] == "run-abc"
    assert body["approval_request"]["title"] == "Approve tool: run_command"


@pytest.mark.asyncio
async def test_resume_route_approves_and_returns_continued_answer(
    monkeypatch, tmp_path: Path
) -> None:
    client = _PausingClient()
    server = _server(monkeypatch, tmp_path, client)
    resp = await _post(
        server,
        "/chat/resume",
        {"run_id": "run-abc", "approved": True, "response": "ok go"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["content"] == "all done"
    # The decision was forwarded to VictorClient.resume verbatim.
    assert client.resumed == [("run-abc", True, "ok go")]


@pytest.mark.asyncio
async def test_resume_unknown_run_is_404(monkeypatch, tmp_path: Path) -> None:
    class _Client:
        async def chat(self, message: str) -> SimpleNamespace:  # pragma: no cover - unused
            return SimpleNamespace(content="", status="ok")

        async def resume(self, run_id: str, decision: Any) -> Any:
            raise ValueError(f"Unknown paused run: {run_id}")

    server = _server(monkeypatch, tmp_path, _Client())
    resp = await _post(server, "/chat/resume", {"run_id": "nope", "approved": True})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_resume_unsupported_client_is_501(monkeypatch, tmp_path: Path) -> None:
    class _NoResume:
        async def chat(self, message: str) -> SimpleNamespace:  # pragma: no cover - unused
            return SimpleNamespace(content="", status="ok")

    server = _server(monkeypatch, tmp_path, _NoResume())
    resp = await _post(server, "/chat/resume", {"run_id": "x", "approved": False})
    assert resp.status_code == 501


@pytest.mark.asyncio
async def test_normal_chat_is_unchanged_200_ok(monkeypatch, tmp_path: Path) -> None:
    class _OkClient:
        async def chat(self, message: str) -> SimpleNamespace:
            return SimpleNamespace(content=f"echo:{message}", tool_calls=[], status="ok")

    server = _server(monkeypatch, tmp_path, _OkClient())
    resp = await _post(server, "/chat", {"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "echo:hi"
    assert body["status"] == "ok"
    assert body["run_id"] is None
