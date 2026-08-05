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

"""Model Context Protocol (MCP) integration for Victor.

MCP is an open protocol that standardizes how AI applications connect to
data sources and tools. This module provides both client and server
implementations for Victor.

Components:
- MCP Server: Expose Victor's tools as MCP resources
- MCP Client: Connect to external MCP servers
- MCP Registry: Auto-discovery and management of multiple MCP servers
- Protocol: Standard message formats and communication
"""

from victor.integrations.mcp.client import MCPClient
from victor.integrations.mcp.protocol import MCPMessage, MCPResource, MCPTool
from victor.integrations.mcp.registry import MCPRegistry, MCPServerConfig, ServerStatus
from victor.integrations.mcp.server import (
    MCPServer,
    create_mcp_server_from_orchestrator,
)

__all__ = [
    "MCPServer",
    "MCPClient",
    "MCPRegistry",
    "MCPServerConfig",
    "ServerStatus",
    "MCPMessage",
    "MCPTool",
    "MCPResource",
    "create_mcp_server_from_orchestrator",
]
