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

"""
MCP integration as a first-class vertical.

This module implements MCP (Model Context Protocol) as a proper vertical,
enabling MCP servers to be discovered, managed, and integrated through
the standard vertical architecture.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from victor_contracts.verticals.protocols.base import VerticalBase
from victor_contracts.verticals.extensions import VerticalExtensions
from victor.tools.base import BaseTool

logger = logging.getLogger(__name__)


class MCPVertical(VerticalBase):
    """
    MCP integration as a first-class vertical.

    Features:
    - Auto-discovery of MCP servers from entry points
    - Dynamic tool registration from MCP servers
    - Resource integration
    - Prompt contribution from MCP knowledge
    - Health monitoring and reconnection
    """

    name = "mcp"
    description = "Model Context Protocol integration for external tool and resource providers"
    version = "1.0.0"

    # Vertical metadata
    tier = "foundation"

    def __init__(self) -> None:
        """Initialize MCP vertical."""
        super().__init__()
        self._registry: Optional[Any] = None
        self._discovered_servers: Set[str] = set()

    @classmethod
    def get_name(cls) -> str:
        """Return vertical identifier."""
        return "mcp"

    @classmethod
    def get_description(cls) -> str:
        """Return human-readable description."""
        return "Model Context Protocol integration for external tool and resource providers"

    @classmethod
    def get_tools(cls) -> List[str]:
        """
        Get list of tool names for this vertical.

        Note: Actual MCP tools are loaded dynamically at runtime.
        This returns the base MCP bridge tool name.
        """
        # Return empty list - tools loaded dynamically
        return []

    @classmethod
    def get_system_prompt(cls) -> str:
        """Return system prompt text for this vertical."""
        return """You are an AI assistant with access to external tools and resources through the Model Context Protocol (MCP).

## MCP Integration

You have access to MCP servers that provide additional tools and resources. These are discovered dynamically and can include:
- External API integrations
- Database connectors
- File system access
- Custom business logic tools

When you need to use a tool or resource, check if it's available through MCP and use it appropriately.

## Guidelines

1. **Tool Discovery**: MCP tools are dynamically registered. Check tool availability before use.
2. **Resource Access**: MCP resources may include files, data, or other external content.
3. **Error Handling**: MCP connections may fail. Handle errors gracefully and provide alternatives.
4. **Performance**: MCP calls may have latency. Use them judiciously.
"""

    @classmethod
    def get_extensions(cls) -> VerticalExtensions:
        """Get vertical extensions."""
        from victor_contracts.verticals.extensions import VerticalExtensions

        # Return extensions with prompt contributor
        # Note: The contributor needs to be instantiated later when vertical is available
        return VerticalExtensions(
            prompt_contributors=lambda: [MCPKnowledgeContributor(cls())],
        )

    async def initialize(self, settings: Any) -> None:
        """
        Initialize MCP vertical.

        Args:
            settings: Victor settings
        """
        from victor.integrations.mcp import MCPRegistry

        # Create MCP registry
        self._registry = MCPRegistry(settings)

        # Discover MCP servers from entry points
        await self._discover_mcp_servers()

        # Connect to all servers
        await self._registry.connect_all()

        logger.info(f"MCP Vertical initialized with {len(self._discovered_servers)} servers")

    async def _discover_mcp_servers(self) -> None:
        """Discover MCP servers from entry points."""
        try:
            import importlib.metadata as metadata

            # Discover from victor.mcp_servers entry point
            eps = metadata.entry_points()

            if hasattr(eps, "select"):
                server_eps = eps.select(group="victor.mcp_servers")
            else:
                server_eps = eps.get("victor.mcp_servers", [])

            for ep in server_eps:
                try:
                    server_config = ep.load()
                    await self._registry.register_server(server_config)
                    self._discovered_servers.add(ep.name)
                    logger.debug(f"Discovered MCP server: {ep.name}")
                except Exception as e:
                    logger.warning(f"Failed to load MCP server {ep.name}: {e}")

        except Exception as e:
            logger.debug(f"No MCP servers configured: {e}")

    async def get_mcp_tools(self) -> List[Any]:
        """
        Get all tools from connected MCP servers.

        Returns:
            List of MCP tools as Victor Tool objects
        """
        if not self._registry:
            return []

        tools = []
        for server in self._registry.get_servers():
            try:
                mcp_tools = await server.list_tools()
                for mcp_tool in mcp_tools:
                    # Convert MCP tool to Victor tool
                    tool = self._mcp_tool_to_victor_tool(mcp_tool, server)
                    tools.append(tool)
            except Exception as e:
                logger.warning(f"Failed to list tools from MCP server: {e}")

        return tools

    def _mcp_tool_to_victor_tool(self, mcp_tool: Any, server: Any) -> Dict[str, Any]:
        """
        Convert MCP tool to Victor tool format.

        Args:
            mcp_tool: MCP tool definition
            server: MCP server instance

        Returns:
            Victor tool definition
        """
        return {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "input_schema": mcp_tool.input_schema,
            "server": server.name,
        }

    async def get_mcp_resources(self) -> List[Dict[str, Any]]:
        """
        Get all resources from connected MCP servers.

        Returns:
            List of MCP resources
        """
        if not self._registry:
            return []

        resources = []
        for server in self._registry.get_servers():
            try:
                mcp_resources = await server.list_resources()
                for resource in mcp_resources:
                    resources.append(
                        {
                            "uri": resource.uri,
                            "name": resource.name,
                            "description": resource.description,
                            "mime_type": resource.mime_type,
                            "server": server.name,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to list resources from MCP server: {e}")

        return resources

    async def call_mcp_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """
        Call an MCP tool on a specific server.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if not self._registry:
            raise RuntimeError("MCP registry not initialized")

        server = self._registry.get_server(server_name)
        if not server:
            raise ValueError(f"MCP server not found: {server_name}")

        return await server.call_tool(tool_name, arguments)

    async def read_mcp_resource(self, server_name: str, resource_uri: str) -> Any:
        """
        Read a resource from an MCP server.

        Args:
            server_name: Name of the MCP server
            resource_uri: URI of the resource to read

        Returns:
            Resource content
        """
        if not self._registry:
            raise RuntimeError("MCP registry not initialized")

        server = self._registry.get_server(server_name)
        if not server:
            raise ValueError(f"MCP server not found: {server_name}")

        return await server.read_resource(resource_uri)

    async def shutdown(self) -> None:
        """Shutdown MCP connections."""
        if self._registry:
            await self._registry.disconnect_all()
            logger.info("MCP Vertical shut down")


class MCPKnowledgeContributor:
    """
    Prompt contributor for MCP knowledge.

    Contributes MCP resources and tools to the system prompt.
    """

    def __init__(self, vertical: MCPVertical) -> None:
        """
        Initialize contributor.

        Args:
            vertical: MCPVertical instance
        """
        self.vertical = vertical

    async def get_prompt_sections(self, context: Any) -> Dict[str, str]:
        """
        Get prompt sections from MCP resources.

        Args:
            context: Execution context

        Returns:
            Dictionary of prompt sections
        """
        sections = {}

        # Add MCP resources section
        resources = await self.vertical.get_mcp_resources()
        if resources:
            sections["mcp_resources"] = self._format_mcp_resources(resources)

        # Add MCP tools section
        tools = await self.vertical.get_mcp_tools()
        if tools:
            sections["mcp_tools"] = self._format_mcp_tools(tools)

        return sections

    def _format_mcp_resources(self, resources: List[Dict[str, Any]]) -> str:
        """Format MCP resources for prompt."""
        if not resources:
            return "No MCP resources available."

        lines = ["## Available MCP Resources", ""]
        for resource in resources:
            lines.append(f"- {resource['name']} ({resource['uri']})")
            if resource.get("description"):
                lines.append(f"  Description: {resource['description']}")
            lines.append(f"  Server: {resource['server']}")

        return "\n".join(lines)

    def _format_mcp_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Format MCP tools for prompt."""
        if not tools:
            return "No MCP tools available."

        lines = ["## Available MCP Tools", ""]
        for tool in tools:
            lines.append(f"- {tool['name']}")
            if tool.get("description"):
                lines.append(f"  Description: {tool['description']}")
            lines.append(f"  Server: {tool['server']}")

        return "\n".join(lines)
