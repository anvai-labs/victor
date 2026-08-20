# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""MCP routes: /mcp/servers, /mcp/connect, /mcp/disconnect."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from victor.integrations.api.fastapi_server import MCPConnectRequest

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create MCP routes bound to *server*."""
    router = APIRouter(tags=["MCP"])

    @router.get("/mcp/servers")
    async def mcp_servers() -> JSONResponse:
        """Get list of configured MCP servers."""
        try:
            from victor.integrations.mcp.registry import get_mcp_registry

            registry = get_mcp_registry()
            servers = []

            for name in registry.list_servers():
                server_info = registry.get_server_info(name)
                servers.append(
                    {
                        "name": name,
                        "connected": server_info.get("connected", False),
                        "tools": server_info.get("tools", []),
                        "endpoint": server_info.get("endpoint"),
                    }
                )

            return JSONResponse({"servers": servers})

        except ImportError:
            return JSONResponse({"servers": [], "error": "MCP not available"})
        except Exception:
            logger.exception("MCP servers error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/mcp/connect")
    async def mcp_connect(request: MCPConnectRequest) -> JSONResponse:
        """Connect to an MCP server."""
        try:
            from victor.integrations.mcp.registry import get_mcp_registry

            registry = get_mcp_registry()
            success = await registry.connect(request.server, endpoint=request.endpoint)

            return JSONResponse({"success": success, "server": request.server})

        except ImportError:
            raise HTTPException(status_code=501, detail="MCP not available")
        except Exception:
            logger.exception("MCP connect error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/mcp/disconnect")
    async def mcp_disconnect(request: MCPConnectRequest) -> JSONResponse:
        """Disconnect from an MCP server."""
        try:
            from victor.integrations.mcp.registry import get_mcp_registry

            registry = get_mcp_registry()
            await registry.disconnect(request.server)

            return JSONResponse({"success": True, "server": request.server})

        except ImportError:
            raise HTTPException(status_code=501, detail="MCP not available")
        except Exception:
            logger.exception("MCP disconnect error")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    return router
