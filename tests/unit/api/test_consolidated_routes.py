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

"""Route tests for the endpoints consolidated onto the FastAPI server.

Covers the LSP routes ported from the (removed) aiohttp server, the implemented
credentials routes, and POST /tools/cancel.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from victor.integrations.api.routes.lsp_routes import (
    create_router as create_lsp_router,
)  # noqa: E402
from victor.integrations.api.routes.system_routes import (  # noqa: E402
    create_router as create_system_router,
)
from victor.integrations.api.routes.tool_routes import (  # noqa: E402
    create_router as create_tool_router,
)


def _client(server) -> TestClient:
    app = FastAPI()
    app.include_router(create_lsp_router(server))
    app.include_router(create_system_router(server))
    app.include_router(create_tool_router(server))
    return TestClient(app)


@pytest.fixture
def server():
    return SimpleNamespace(
        _pending_tool_approvals={},
        _broadcast_ws=AsyncMock(),
        _get_orchestrator=AsyncMock(),
        _get_victor_client=AsyncMock(),
    )


class TestLspRoutes:
    def test_lsp_endpoints_degrade_gracefully_without_capability(self, server):
        client = _client(server)
        body = {"file": "a.py", "line": 1, "character": 2}
        # No LSPManager capability is registered in the test env -> graceful empties.
        r = client.post("/lsp/hover", json=body)
        assert r.status_code == 200
        assert r.json() == {"contents": None, "error": "LSP not available"}

        r = client.post("/lsp/definition", json=body)
        assert r.json()["locations"] == []
        r = client.post("/lsp/references", json=body)
        assert r.json()["locations"] == []
        r = client.post("/lsp/completions", json=body)
        assert r.json()["completions"] == []
        r = client.post("/lsp/diagnostics", json={"file": "a.py"})
        assert r.json()["diagnostics"] == []


class TestCredentialRoutes:
    def test_status_reports_keyring_availability(self, server):
        r = _client(server).get("/credentials/status")
        assert r.status_code == 200
        data = r.json()
        assert {"available", "backend", "configured_providers"} <= set(data)
        assert isinstance(data["available"], bool)

    def test_set_get_delete_round_trip(self, server, monkeypatch):
        store: dict[str, str] = {}
        monkeypatch.setattr(
            "victor.config.api_keys.set_api_key",
            lambda provider, key, use_keyring=False: store.__setitem__(provider, key) or True,
        )
        monkeypatch.setattr(
            "victor.config.api_keys.get_api_key", lambda provider: store.get(provider)
        )
        monkeypatch.setattr(
            "victor.config.api_keys.delete_api_key_from_keyring",
            lambda provider: store.pop(provider, None) is not None,
        )
        monkeypatch.setattr("victor.config.api_keys.clear_api_key_cache", lambda: None)
        client = _client(server)

        assert client.post(
            "/credentials/set", json={"provider": "anthropic", "api_key": "sk-x"}
        ).json() == {"provider": "anthropic", "success": True}
        assert (
            client.get("/credentials/get", params={"provider": "anthropic"}).json()["api_key"]
            == "sk-x"
        )
        assert client.request(
            "DELETE", "/credentials/delete", params={"provider": "anthropic"}
        ).json() == {"provider": "anthropic", "success": True}
        assert (
            client.get("/credentials/get", params={"provider": "anthropic"}).json()["api_key"]
            is None
        )


class TestToolsCancel:
    def test_cancel_unknown_returns_not_found(self, server):
        r = _client(server).post("/tools/cancel", json={"tool_call_id": "nope"})
        assert r.json() == {
            "tool_call_id": "nope",
            "cancelled": False,
            "reason": "not_found",
        }

    def test_cancel_resolves_a_pending_approval(self, server):
        server._pending_tool_approvals["call-1"] = {
            "tool_name": "shell",
            "resolved": False,
        }
        r = _client(server).post("/tools/cancel", json={"tool_call_id": "call-1"})
        assert r.json() == {"tool_call_id": "call-1", "cancelled": True}
        assert "call-1" not in server._pending_tool_approvals
        server._broadcast_ws.assert_awaited()
