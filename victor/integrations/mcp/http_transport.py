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

"""Streamable-HTTP transport for Victor's MCP server (Phase 3).

Transport-level core with NO fastapi import: ``StreamableHTTPTransport``
turns a JSON-RPC request body + headers + session id into an HTTP
``(status, headers, body)`` triple. The FastAPI route layer
(``integrations/api/routes/mcp_transport_routes.py``) and the standalone
server (``integrations/mcp/standalone_http.py``) are thin adapters over
this module — all protocol decisions live here.

Spec posture (streamable-HTTP, modelcontextprotocol.io):

- POST-only server: no SSE stream, GET is rejected 405 by the adapters
  (legal — servers MAY not offer streaming).
- Notifications (``handle_message`` → ``None``) get ``202 Accepted``.
- ``initialize`` with no ``Mcp-Session-Id`` mints a uuid4 session;
  every other request must carry a known session id (unknown → 404,
  missing → 400).
- Session teardown via HTTP DELETE → 202.

Session state is a ``dict[str, MCPServer]`` of per-session views (see
``MCPServer.for_session``) plus a last-seen timestamp dict pruned on each
request with a 1h TTL — no external dependency.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

if TYPE_CHECKING:
    # Type-annotation-only dependency on the server (the layering ratchet
    # skips TYPE_CHECKING blocks); no runtime upward import.
    from victor.integrations.mcp.server import MCPServer

# Protocol versions this server can speak. The stdio constant
# (MCP_PROTOCOL_VERSION = "2024-11-05") is NOT flipped — only the HTTP
# transport negotiates.
SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")

# Streamable-HTTP baseline: clients that request nothing (or something
# unknown) are served this version.
DEFAULT_PROTOCOL_VERSION = "2025-03-26"

DEFAULT_SESSION_TTL_SECONDS = 3600.0

_SESSION_HEADER = "Mcp-Session-Id"
_PROTOCOL_HEADER = "MCP-Protocol-Version"
_JSON_CONTENT_TYPE = {"Content-Type": "application/json"}


def negotiate_protocol_version(requested: Optional[str]) -> str:
    """Return the client's requested version if supported, else the default."""
    if requested in SUPPORTED_PROTOCOL_VERSIONS:
        return requested
    return DEFAULT_PROTOCOL_VERSION


class StreamableHTTPTransport:
    """Adapts JSON-RPC request bodies to the transport-agnostic MCPServer."""

    def __init__(self, server: MCPServer, session_ttl: float = DEFAULT_SESSION_TTL_SECONDS):
        self._server = server
        self._session_ttl = session_ttl
        self._sessions: Dict[str, MCPServer] = {}
        self._last_seen: Dict[str, float] = {}

    async def handle_post(
        self,
        body: bytes,
        headers: Mapping[str, str],
        session_id: Optional[str],
    ) -> Tuple[int, Dict[str, str], Optional[bytes]]:
        """JSON-RPC body → ``handle_message`` → (status, headers, response bytes).

        Returns ``(202, headers, None)`` for notifications and JSON-RPC
        error bodies for transport-level failures (400 parse errors, 404
        unknown session, 400 missing session).
        """
        try:
            message = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return (
                400,
                dict(_JSON_CONTENT_TYPE),
                self._error_body(None, -32700, f"Parse error: {exc}"),
            )

        if not isinstance(message, dict):
            return (
                400,
                dict(_JSON_CONTENT_TYPE),
                self._error_body(None, -32700, "Parse error: request must be a JSON object"),
            )

        self._prune_expired_sessions()

        method = message.get("method")
        is_initialize = method == "initialize"

        if session_id is None and not is_initialize:
            return (
                400,
                dict(_JSON_CONTENT_TYPE),
                self._error_body(None, -32000, f"Missing {_SESSION_HEADER} header"),
            )
        if session_id is not None and session_id not in self._sessions:
            # Spec: a 404 on a session id means "session terminated" —
            # the client must re-initialize.
            return (
                404,
                dict(_JSON_CONTENT_TYPE),
                self._error_body(None, -32001, "Unknown or expired session"),
            )

        if is_initialize and session_id is None:
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = self._server.for_session(session_id)

        session_server = self._sessions[session_id]
        self._last_seen[session_id] = time.monotonic()

        response = await session_server.handle_message(message)

        response_headers: Dict[str, str] = dict(_JSON_CONTENT_TYPE)
        client_protocol = _get_header(headers, _PROTOCOL_HEADER)
        if client_protocol in SUPPORTED_PROTOCOL_VERSIONS:
            response_headers[_PROTOCOL_HEADER] = client_protocol

        if response is None:
            # Notification: JSON-RPC never answers these — 202 with no body.
            return 202, response_headers, None

        if is_initialize:
            self._negotiate_initialize_version(message, response)

        if session_id is not None:
            response_headers[_SESSION_HEADER] = session_id

        return 200, response_headers, json.dumps(response).encode("utf-8")

    async def handle_delete(self, session_id: str) -> int:
        """Session teardown → 202 (deleted) or 404 (unknown session)."""
        self._prune_expired_sessions()
        if session_id not in self._sessions:
            return 404
        del self._sessions[session_id]
        self._last_seen.pop(session_id, None)
        return 202

    def _negotiate_initialize_version(
        self, request: Dict[str, Any], response: Dict[str, Any]
    ) -> None:
        """Rewrite the initialize response to the negotiated protocol version.

        The dispatcher always stamps ``MCP_PROTOCOL_VERSION`` (the stdio
        constant); the HTTP transport overrides it here so the stdio wire
        contract stays untouched.
        """
        params = request.get("params") or {}
        requested = params.get("protocolVersion")
        result = response.get("result")
        if isinstance(result, dict):
            result["protocolVersion"] = negotiate_protocol_version(requested)

    def _prune_expired_sessions(self) -> None:
        now = time.monotonic()
        expired = [
            sid
            for sid, last_seen in self._last_seen.items()
            if now - last_seen >= self._session_ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_seen.pop(sid, None)

    @staticmethod
    def _error_body(msg_id: Optional[str], code: int, message: str) -> bytes:
        payload = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }
        return json.dumps(payload).encode("utf-8")


def _get_header(headers: Mapping[str, str], name: str) -> Optional[str]:
    """Case-insensitive header lookup (FastAPI/Starlette headers are
    case-insensitive; plain test dicts are not)."""
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None
