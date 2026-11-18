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

"""Streamable-HTTP transport tests (Phase 3) — transport level, no HTTP stack.

Each test drives ``StreamableHTTPTransport.handle_post`` directly with raw
bytes and plain header dicts, mirroring exactly what the FastAPI route layer
feeds in. No fastapi import appears in this module.
"""

import json
import uuid
from typing import Dict, Optional

import pytest

from victor.integrations.mcp.http_transport import (
    DEFAULT_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    StreamableHTTPTransport,
)
from victor.integrations.mcp.server import MCPServer


def _initialize_msg(protocol_version: Optional[str] = DEFAULT_PROTOCOL_VERSION) -> dict:
    params: Dict = {
        "capabilities": {},
        "clientInfo": {"name": "phase3-tests", "version": "1.0.0"},
    }
    if protocol_version is not None:
        params["protocolVersion"] = protocol_version
    return {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": params}


def _tools_list_msg() -> dict:
    return {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


def _enc(payload) -> bytes:
    """Encode a JSON-RPC message the way the HTTP layer delivers it."""
    return json.dumps(payload).encode("utf-8")


class _NotInitializedRegistry:
    """Registry whose execute() explodes — proves refusals never execute."""

    async def execute(self, *args, **kwargs):  # pragma: no cover - failure signal
        raise AssertionError("tool execution must not happen in refusal posture")

    def list_tools(self):
        return []


async def _initialize(transport: StreamableHTTPTransport) -> str:
    status, headers, body = await transport.handle_post(_enc(_initialize_msg()), {}, None)
    assert status == 200, body
    session_id = headers["Mcp-Session-Id"]
    # The minted id must be a valid uuid4 string.
    uuid.UUID(session_id)
    return session_id


class TestSessionMinting:
    async def test_initialize_returns_200_with_session_header(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        status, headers, body = await transport.handle_post(_enc(_initialize_msg()), {}, None)
        assert status == 200
        assert "Mcp-Session-Id" in headers
        uuid.UUID(headers["Mcp-Session-Id"])

    async def test_initialize_response_carries_negotiated_protocol_version(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        _, _, body = await transport.handle_post(_enc(_initialize_msg("2025-06-18")), {}, None)
        response = json.loads(body)
        assert response["result"]["protocolVersion"] == "2025-06-18"

    async def test_initialize_without_requested_version_gets_default(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        _, _, body = await transport.handle_post(_enc(_initialize_msg(None)), {}, None)
        response = json.loads(body)
        assert response["result"]["protocolVersion"] == DEFAULT_PROTOCOL_VERSION
        assert DEFAULT_PROTOCOL_VERSION in SUPPORTED_PROTOCOL_VERSIONS

    async def test_reinitialize_with_known_session_reuses_it(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        sid = await _initialize(transport)
        status, headers, _ = await transport.handle_post(_enc(_initialize_msg()), {}, sid)
        assert status == 200
        assert headers["Mcp-Session-Id"] == sid


class TestSessionValidation:
    async def test_unknown_session_is_404(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        status, _, body = await transport.handle_post(
            _enc(_tools_list_msg()), {}, "00000000-0000-0000-0000-000000000000"
        )
        assert status == 404
        assert b"error" in body

    async def test_missing_session_for_non_initialize_is_400(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        status, _, _ = await transport.handle_post(_enc(_tools_list_msg()), {}, None)
        assert status == 400

    async def test_delete_then_reuse_is_404(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        sid = await _initialize(transport)
        assert await transport.handle_delete(sid) == 202
        assert await transport.handle_delete("no-such-session") == 404
        status, _, _ = await transport.handle_post(_enc(_tools_list_msg()), {}, sid)
        assert status == 404


class TestNotifications:
    async def test_notification_returns_202_empty_body(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        sid = await _initialize(transport)
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        status, _, body = await transport.handle_post(_enc(notification), {}, sid)
        assert status == 202
        assert body is None


class TestParseErrors:
    async def test_garbage_body_is_400_parse_error(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        status, headers, body = await transport.handle_post(b"{not json", {}, None)
        assert status == 400
        assert headers["Content-Type"] == "application/json"
        response = json.loads(body)
        assert response["error"]["code"] == -32700

    async def test_non_object_json_is_400(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        status, _, _ = await transport.handle_post(b"[1,2,3]", {}, None)
        assert status == 400


class TestProtocolNegotiation:
    async def test_supported_client_header_is_echoed(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        status, headers, _ = await transport.handle_post(
            _enc(_initialize_msg()),
            {"MCP-Protocol-Version": "2025-06-18"},
            None,
        )
        assert status == 200
        assert headers["MCP-Protocol-Version"] == "2025-06-18"

    async def test_unsupported_client_header_is_not_echoed(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        _, headers, _ = await transport.handle_post(
            _enc(_initialize_msg()),
            {"MCP-Protocol-Version": "1999-01-01"},
            None,
        )
        assert "MCP-Protocol-Version" not in headers

    async def test_header_lookup_is_case_insensitive(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
        _, headers, _ = await transport.handle_post(
            _enc(_initialize_msg()),
            {"mcp-protocol-version": "2025-06-18"},
            None,
        )
        assert headers["MCP-Protocol-Version"] == "2025-06-18"


class TestRefusalPosture:
    async def test_admission_refusal_is_error_without_execution(self):
        server = MCPServer(
            name="t",
            version="1",
            tool_registry=_NotInitializedRegistry(),
            exec_context=None,
            admission_refusal="refused by policy (fail-closed)",
        )
        transport = StreamableHTTPTransport(server)
        sid = await _initialize(transport)
        call = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "shell", "arguments": {"command": "rm -rf /"}},
        }
        status, _, body = await transport.handle_post(_enc(call), {}, sid)
        assert status == 200
        response = json.loads(body)
        assert response["result"]["isError"] is True
        assert "refused by policy" in response["result"]["content"][0]["text"]


class TestSessionTTL:
    async def test_expired_sessions_are_pruned(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"), session_ttl=0.0)
        sid = await _initialize(transport)
        # Any subsequent request prunes everything with age > ttl=0.
        status, _, _ = await transport.handle_post(_enc(_tools_list_msg()), {}, sid)
        assert status == 404

    async def test_live_sessions_survive_pruning(self):
        transport = StreamableHTTPTransport(MCPServer(name="t", version="1"), session_ttl=3600.0)
        sid = await _initialize(transport)
        status, _, body = await transport.handle_post(_enc(_tools_list_msg()), {}, sid)
        assert status == 200
        response = json.loads(body)
        assert "tools" in response["result"]

    async def test_stdio_root_server_state_not_polluted_by_sessions(self):
        """Session views are isolated: initializing via HTTP must not mark
        the root (stdio-shared) server instance as initialized."""
        root = MCPServer(name="t", version="1")
        transport = StreamableHTTPTransport(root)
        sid = await _initialize(transport)
        assert (await transport.handle_post(_enc(_tools_list_msg()), {}, sid))[0] == 200
        assert root.initialized is False


@pytest.mark.parametrize("version", SUPPORTED_PROTOCOL_VERSIONS)
async def test_all_supported_versions_negotiate_to_themselves(version):
    transport = StreamableHTTPTransport(MCPServer(name="t", version="1"))
    _, _, body = await transport.handle_post(_enc(_initialize_msg(version)), {}, None)
    response = json.loads(body)
    assert response["result"]["protocolVersion"] == version
