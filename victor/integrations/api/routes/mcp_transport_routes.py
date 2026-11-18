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

"""Streamable-HTTP transport routes for the Victor FastAPI server.

Exposes the MCP JSON-RPC dispatcher at ``POST /mcp`` and session teardown
at ``DELETE /mcp``. Plain APIRoutes only — a ``Mount`` would bypass the
app-level Bearer auth in ``_include_with_auth`` (fastapi_server.py), and
MCP tools execute code, so these routes MUST sit behind the key gate.

The transport endpoint is distinct from the management surface
``/mcp/servers`` (mcp_routes.py): that one lists configured upstream MCP
servers; this one SERVES Victor's own tools over MCP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Request, Response

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer
    from victor.integrations.mcp.http_transport import StreamableHTTPTransport

logger = logging.getLogger(__name__)


def create_mcp_transport_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create the /mcp transport routes bound to *server*.

    The MCP assembly is built lazily on first transport request: building
    it eagerly would run tool discovery + sandbox admission (and mutate
    the process-global graph-tool feature flag) for every API server
    boot, including unrelated test servers and TUI sessions that never
    touch /mcp.
    """
    router = APIRouter(tags=["MCP"])
    transport_holder: list = []

    def _get_transport() -> "StreamableHTTPTransport":
        if not transport_holder:
            from victor.integrations.mcp.assembly import create_mcp_server
            from victor.integrations.mcp.http_transport import StreamableHTTPTransport

            workspace_root = Path(server.workspace_root or Path.cwd())
            assembly = create_mcp_server(workspace_root)
            transport_holder.append(StreamableHTTPTransport(assembly.server))
        return transport_holder[0]

    @router.post("/mcp")
    async def mcp_transport_post(request: Request) -> Response:
        """Streamable-HTTP POST: JSON-RPC request → dispatcher."""
        transport = _get_transport()
        body = await request.body()
        session_id: Optional[str] = request.headers.get("Mcp-Session-Id")
        status, headers, payload = await transport.handle_post(body, request.headers, session_id)
        return Response(content=payload, status_code=status, headers=headers)

    @router.delete("/mcp")
    async def mcp_transport_delete(request: Request) -> Response:
        """Streamable-HTTP DELETE: explicit session teardown."""
        transport = _get_transport()
        session_id: Optional[str] = request.headers.get("Mcp-Session-Id")
        status = await transport.handle_delete(session_id or "")
        return Response(status_code=status)

    return router


# Route-module convention: every module in this package exports
# ``create_router(server)``; the transport module keeps its explicit name
# as the primary symbol and aliases it for the shared factory list.
create_router = create_mcp_transport_router

__all__ = ["create_mcp_transport_router", "create_router"]
