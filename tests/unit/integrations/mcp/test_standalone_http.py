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

"""Standalone MCP HTTP app (``victor mcp --http``): the fail-closed Bearer
auth contract is load-bearing — this surface executes shell commands, so
an unauthenticated HTTP mode must not exist. Positive AND negative tests
pin the exact-match guard and the no-anonymous-escape construction rule.
"""

import json

import pytest
from fastapi.testclient import TestClient

from victor.integrations.mcp.server import MCPServer
from victor.integrations.mcp.standalone_http import create_standalone_app

API_KEY = "qual-secret-key"

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "qual", "version": "1"},
    },
}


@pytest.fixture
def app():
    return create_standalone_app(MCPServer(name="t", version="1"), API_KEY)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def _auth(v: str) -> dict:
    return {"Authorization": v}


class TestConstructionFailClosed:
    def test_empty_api_key_rejected(self):
        with pytest.raises(ValueError, match="non-empty API key"):
            create_standalone_app(MCPServer(name="t", version="1"), "")

    def test_blank_api_key_rejected(self):
        with pytest.raises(ValueError, match="non-empty API key"):
            create_standalone_app(MCPServer(name="t", version="1"), "   ")

    def test_api_key_is_stripped_for_the_guard(self, client):
        """A padded key configures the same exact-match guard as the bare key."""
        app = create_standalone_app(MCPServer(name="t", version="1"), f"  {API_KEY}  ")
        with TestClient(app) as c:
            assert (
                c.post("/mcp", json=INITIALIZE, headers=_auth(f"Bearer {API_KEY}")).status_code
                == 200
            )


class TestBearerGuard:
    def test_missing_authorization_is_401(self, client):
        assert client.post("/mcp", json=INITIALIZE).status_code == 401

    def test_wrong_key_is_401(self, client):
        r = client.post("/mcp", json=INITIALIZE, headers=_auth("Bearer not-the-key"))
        assert r.status_code == 401

    def test_scheme_downgrade_is_401(self, client):
        """Exact-match: a lowercase scheme must not authenticate."""
        r = client.post("/mcp", json=INITIALIZE, headers=_auth(f"bearer {API_KEY}"))
        assert r.status_code == 401

    def test_prefix_key_is_401(self, client):
        r = client.post("/mcp", json=INITIALIZE, headers=_auth(f"Bearer {API_KEY}-extra"))
        assert r.status_code == 401

    def test_delete_requires_auth_too(self, client):
        assert client.delete("/mcp", headers={"Mcp-Session-Id": "s"}).status_code == 401


class TestAuthenticatedFlow:
    def test_initialize_succeeds_with_valid_key(self, client):
        r = client.post("/mcp", json=INITIALIZE, headers=_auth(f"Bearer {API_KEY}"))
        assert r.status_code == 200
        body = json.loads(r.content)
        assert body["result"]["serverInfo"]["name"] == "t"

    def test_post_and_delete_round_trip(self, client):
        r = client.post("/mcp", json=INITIALIZE, headers=_auth(f"Bearer {API_KEY}"))
        sid = r.headers["Mcp-Session-Id"]
        tools = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            headers={**_auth(f"Bearer {API_KEY}"), "Mcp-Session-Id": sid},
        )
        assert tools.status_code == 200
        # 202 Accepted is the MCP-spec session-termination status.
        assert client.delete(
            "/mcp", headers={**_auth(f"Bearer {API_KEY}"), "Mcp-Session-Id": sid}
        ).status_code in (200, 202, 204)

    def test_docs_surfaces_disabled(self, client):
        """No anonymous introspection: docs/openapi are off in standalone mode."""
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code in (401, 404)
