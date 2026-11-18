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

"""Transport-agnostic MCP server assembly (single source of truth).

Both the stdio entry point (``victor mcp`` / ``victor-mcp-server``) and
future transports (streamable-HTTP, SSE) call :func:`create_mcp_server`
to build a fully wired :class:`MCPServer`. The caller then chooses the
transport (stdio server, HTTP handler, …) — the assembly is
transport-agnostic by design.

Admission is fail-closed: the ENTIRE factory body is wrapped in
try/except — any failure (settings read, tool discovery, registry
construction) produces a refusal assembly rather than propagating.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from victor.integrations.mcp.discovery import register_discovered_tools
from victor.integrations.mcp.server import (
    MCPServer,
    _admit_or_refuse,
    apply_graph_tool_flag,
    register_graph_index_resource,
)
from victor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPAssembly:
    """The fully wired output of :func:`create_mcp_server`.

    ``tool_count`` feeds the startup banner in both entry points.
    """

    server: MCPServer
    exec_context: object  # MCPToolContext or None (admission failure)
    admission_refusal: Optional[str]
    tool_count: int


def create_mcp_server(
    workspace_root: Path,
    *,
    no_graph_tools: bool = False,
) -> MCPAssembly:
    """Build a fully wired MCP server (transport-agnostic).

    The ENTIRE factory body is wrapped in try/except: any failure produces
    a refusal assembly rather than propagating (fail-closed boundary).
    """
    admission_refusal: Optional[str] = None
    exec_context: Any = None
    registered_count = 0
    registry: Optional[ToolRegistry] = None
    try:
        registry = ToolRegistry()
        apply_graph_tool_flag(no_graph_tools)
        registered_count = register_discovered_tools(registry)
        exec_context, admission_refusal = _admit_or_refuse(workspace_root)
    except Exception as exc:
        admission_refusal = (
            f"MCP assembly failed ({exc}); refusing tool execution "
            "(fail-closed). Fix the settings or use the escape hatch."
        )
        logger.warning("MCP assembly admission failure: %s", exc)

    server = MCPServer(
        name="Victor MCP Server",
        version="1.0.0",
        tool_registry=registry if registry is not None else ToolRegistry(),
        exec_context=exec_context,
        admission_refusal=admission_refusal,
    )
    if exec_context is not None:
        register_graph_index_resource(server, workspace_root)

    posture = "sandboxed (admitted policy)" if exec_context else "ADMISSION REFUSAL (fail-closed)"
    print(
        f"Starting MCP server with {registered_count} tools; posture: {posture}",
        file=sys.stderr,
    )
    return MCPAssembly(
        server=server,
        exec_context=exec_context,
        admission_refusal=admission_refusal,
        tool_count=registered_count,
    )
