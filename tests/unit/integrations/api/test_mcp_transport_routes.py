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

"""FastAPI route-level tests for the /mcp streamable-HTTP transport.

Bearer-key auth is inherited from ``_include_with_auth`` — the first two
tests pin that: no key → 401 before any MCP machinery runs; a valid key →
the full initialize handshake through the real assembly server.
"""

from pathlib import Path

from httpx import ASGITransport, AsyncClient

from victor.integrations.api import fastapi_server

API_KEY = "sk-mcp-transport-test"


def _create_server(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "victor.workflows.hitl_api.get_default_hitl_db_path",
        lambda: tmp_path / "hitl.db",
    )
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        allow_unauthenticated=True,
        api_keys={API_KEY: "tester"},
    )
    return server


def _auth_header() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}


async def _client(server) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=server.app), base_url="http://test")


_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "route-tests", "version": "1.0.0"},
    },
}


class TestBearerAuthGate:
    async def test_post_mcp_without_bearer_is_401(self, monkeypatch, tmp_path):
        server = _create_server(monkeypatch, tmp_path)
        async with await _client(server) as client:
            response = await client.post("/mcp", json=_INITIALIZE)
        assert response.status_code == 401

    async def test_delete_mcp_without_bearer_is_401(self, monkeypatch, tmp_path):
        server = _create_server(monkeypatch, tmp_path)
        async with await _client(server) as client:
            response = await client.delete("/mcp", headers={"Mcp-Session-Id": "x"})
        assert response.status_code == 401


class TestHandshakeThroughRoutes:
    async def test_initialize_with_key_returns_session(self, monkeypatch, tmp_path):
        server = _create_server(monkeypatch, tmp_path)
        async with await _client(server) as client:
            response = await client.post("/mcp", json=_INITIALIZE, headers=_auth_header())
        assert response.status_code == 200
        assert response.headers["Mcp-Session-Id"]
        body = response.json()
        assert body["result"]["protocolVersion"] == "2025-03-26"

    async def test_full_handshake_list_tools_on_session(self, monkeypatch, tmp_path):
        server = _create_server(monkeypatch, tmp_path)
        async with await _client(server) as client:
            init = await client.post("/mcp", json=_INITIALIZE, headers=_auth_header())
            sid = init.headers["Mcp-Session-Id"]
            listed = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                headers={**_auth_header(), "Mcp-Session-Id": sid},
            )
        assert listed.status_code == 200
        assert "tools" in listed.json()["result"]

    async def test_notification_is_202(self, monkeypatch, tmp_path):
        server = _create_server(monkeypatch, tmp_path)
        async with await _client(server) as client:
            init = await client.post("/mcp", json=_INITIALIZE, headers=_auth_header())
            sid = init.headers["Mcp-Session-Id"]
            response = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers={**_auth_header(), "Mcp-Session-Id": sid},
            )
        assert response.status_code == 202
        assert response.content == b""

    async def test_delete_teardowns_session(self, monkeypatch, tmp_path):
        server = _create_server(monkeypatch, tmp_path)
        async with await _client(server) as client:
            init = await client.post("/mcp", json=_INITIALIZE, headers=_auth_header())
            sid = init.headers["Mcp-Session-Id"]
            deleted = await client.delete("/mcp", headers={**_auth_header(), "Mcp-Session-Id": sid})
            assert deleted.status_code == 202
            reused = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                headers={**_auth_header(), "Mcp-Session-Id": sid},
            )
        assert reused.status_code == 404

    async def test_get_mcp_is_405(self, monkeypatch, tmp_path):
        """POST-only server: GET /mcp must be 405 (spec-legal, no SSE yet)."""
        server = _create_server(monkeypatch, tmp_path)
        async with await _client(server) as client:
            response = await client.get("/mcp", headers=_auth_header())
        assert response.status_code == 405


class TestCoexistence:
    async def test_mcp_servers_route_still_works_alongside_transport(self, monkeypatch, tmp_path):
        """The management route /mcp/servers and the transport endpoint /mcp
        must coexist without path collisions."""
        server = _create_server(monkeypatch, tmp_path)
        async with await _client(server) as client:
            servers = await client.get("/mcp/servers", headers=_auth_header())
            init = await client.post("/mcp", json=_INITIALIZE, headers=_auth_header())
        assert servers.status_code == 200
        assert "servers" in servers.json()
        assert init.status_code == 200
