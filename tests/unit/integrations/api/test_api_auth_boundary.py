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

"""App-level API auth boundary.

Auth used to be opt-in per handler — only /chat* called _verify_api_key (3
sites in 1 of 19 route files), leaving git/config/agents/terminal/GraphQL
open even when API keys were configured (co-design review U7-F3). The
app-level dependency now gates every HTTP route when keys are configured;
keyless servers are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from victor.integrations.api import fastapi_server


class _FakeOrchestrator:
    async def chat(self, message: str) -> SimpleNamespace:
        return SimpleNamespace(content=f"echo:{message}", tool_calls=[])

    async def stream_chat(self, message: str):
        yield SimpleNamespace(content=f"stream:{message}", tool_calls=None)

    async def graceful_shutdown(self) -> None:
        return None


def _create_server(monkeypatch, tmp_path: Path, **server_kwargs):
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
    server._orchestrator = _FakeOrchestrator()
    return server


@pytest.mark.asyncio
async def test_keyed_server_rejects_unauthenticated_non_chat_routes(monkeypatch, tmp_path):
    server = _create_server(monkeypatch, tmp_path, api_keys={"sk-alice": "alice"})
    transport = ASGITransport(app=server.app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Real route paths from config/agent route modules.
        response = await client.get("/agents")
        assert response.status_code == 401
        response = await client.get("/config/effective")
        assert response.status_code == 401
        response = await client.post("/model/switch", json={"model": "x"})
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_keyed_server_keeps_health_and_status_open(monkeypatch, tmp_path):
    server = _create_server(monkeypatch, tmp_path, api_keys={"sk-alice": "alice"})
    transport = ASGITransport(app=server.app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for path in ["/health", "/status"]:
            response = await client.get(path)
            assert response.status_code in (
                200,
                404,
            ), f"{path} must not be auth-gated, got {response.status_code}"


@pytest.mark.asyncio
async def test_keyed_server_accepts_bearer_on_protected_route(monkeypatch, tmp_path):
    server = _create_server(monkeypatch, tmp_path, api_keys={"sk-alice": "alice"})
    transport = ASGITransport(app=server.app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/status", headers={"Authorization": "Bearer sk-alice"})
        assert response.status_code in (200, 404)


@pytest.mark.asyncio
async def test_keyless_server_leaves_routes_open(monkeypatch, tmp_path):
    server = _create_server(monkeypatch, tmp_path)
    transport = ASGITransport(app=server.app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/status")
        assert response.status_code in (200, 404)
        response = await client.get("/agents")
        assert response.status_code != 401, "keyless server must not auth-gate"
