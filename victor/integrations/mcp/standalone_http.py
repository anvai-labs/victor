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

"""Standalone streamable-HTTP mode for Victor's MCP server (``victor mcp --http``).

AUTH RULE (fail-closed, load-bearing): the standalone MCP HTTP endpoint
REQUIRES a Bearer API key — ``--api-key`` or the ``VICTOR_API_KEY``
environment variable. There is deliberately NO allow-unauthenticated
escape and NO anonymous read-only mode: Victor's MCP tools execute shell
commands and code, and an unauthenticated HTTP surface for code execution
is a remote-code-execution endpoint. ``victor serve``'s keyless local-dev
posture does NOT carry over here.

This app mounts ONLY the transport routes (POST/DELETE /mcp) — no
management API, no chat, no websocket surfaces.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response

from victor.integrations.mcp.http_transport import StreamableHTTPTransport

if TYPE_CHECKING:
    from victor.integrations.mcp.server import MCPServer

logger = logging.getLogger(__name__)


def _bearer_guard(api_key: str):
    """Dependency factory: exact Bearer match or 401. Fail-closed."""

    async def _require_bearer(request: Request) -> None:
        authorization = request.headers.get("Authorization", "")
        if authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return _require_bearer


def create_standalone_app(server: "MCPServer", api_key: str) -> FastAPI:
    """Build the minimal FastAPI app serving ONLY the MCP transport.

    Raises ``ValueError`` on an empty api_key — never start an
    unauthenticated MCP HTTP server.
    """
    if not api_key or not api_key.strip():
        raise ValueError(
            "Standalone MCP HTTP mode requires a non-empty API key "
            "(--api-key or VICTOR_API_KEY). Unauthenticated MCP is a "
            "remote-code-execution hole — fail-closed."
        )

    app = FastAPI(
        title="Victor MCP Server",
        description="Victor's tools over the MCP streamable-HTTP transport (Bearer-authenticated).",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    transport = StreamableHTTPTransport(server)
    guard = Depends(_bearer_guard(api_key.strip()))

    @app.post("/mcp", dependencies=[guard])
    async def mcp_post(request: Request) -> Response:
        body = await request.body()
        session_id: Optional[str] = request.headers.get("Mcp-Session-Id")
        status, headers, payload = await transport.handle_post(body, request.headers, session_id)
        return Response(content=payload, status_code=status, headers=headers)

    @app.delete("/mcp", dependencies=[guard])
    async def mcp_delete(request: Request) -> Response:
        session_id: Optional[str] = request.headers.get("Mcp-Session-Id")
        status = await transport.handle_delete(session_id or "")
        return Response(status_code=status)

    return app
