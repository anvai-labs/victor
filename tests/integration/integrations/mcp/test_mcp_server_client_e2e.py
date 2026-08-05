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

"""End-to-end integration tests for MCP server and client.

These tests start a real MCP server process and connect to it with a real
MCP client, testing actual stdio communication over JSON-RPC 2.0.

This is fundamentally an integration test because:
1. It spawns real subprocesses
2. Tests real network/pipe I/O
3. Tests the interaction between server and client components

For unit testing individual components in isolation (with mocks), see:
- tests/unit/test_mcp_server.py
- tests/unit/test_mcp_client.py
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


def _check_mcp_available():
    """Check if MCP module is available."""
    try:
        from victor.integrations.mcp.server import MCPServer
        from victor.integrations.mcp.client import MCPClient
        from victor.integrations.mcp.protocol import MCPTool

        return True
    except ImportError:
        return False


# Skip entire module if MCP is not available
pytestmark = [
    pytest.mark.skipif(not _check_mcp_available(), reason="MCP module not available"),
    pytest.mark.integration,
]


# ============================================================================
# Test Server Script (written to temp file and executed as subprocess)
# ============================================================================

TEST_SERVER_SCRIPT = '''
"""Minimal MCP server for integration testing."""
import asyncio
import json
import sys


class SimpleMCPServer:
    """A minimal MCP server for testing."""

    def __init__(self):
        self.initialized = False
        self.tools = [
            {
                "name": "test_echo",
                "description": "Echo back the input",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to echo"}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "test_add",
                "description": "Add two numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            }
        ]
        self.resources = [
            {
                "uri": "test://greeting",
                "name": "greeting",
                "description": "A test greeting resource",
                "mimeType": "text/plain"
            }
        ]

    async def handle_message(self, message: dict) -> dict:
        """Handle incoming JSON-RPC message."""
        msg_id = message.get("id")
        method = message.get("method", "")
        params = message.get("params", {})

        if method == "initialize":
            self.initialized = True
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": "test-mcp-server", "version": "1.0.0"}
                }
            }

        if method == "notifications/initialized":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": self.tools}}

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name == "test_echo":
                # Return in STANDARD MCP format (per modelcontextprotocol.io spec)
                result = {
                    "content": [{"type": "text", "text": arguments.get("message", "")}],
                    "isError": False
                }
            elif tool_name == "test_add":
                sum_result = arguments.get("a", 0) + arguments.get("b", 0)
                result = {
                    "content": [{"type": "text", "text": str(sum_result)}],
                    "isError": False
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                }

            return {"jsonrpc": "2.0", "id": msg_id, "result": result}

        if method == "resources/list":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"resources": self.resources}}

        if method == "resources/read":
            uri = params.get("uri")
            if uri == "test://greeting":
                # Return in format expected by Victor client (result.content)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": "Hello from MCP!"
                    }
                }
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32001, "message": f"Resource not found: {uri}"}
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }


async def main():
    """Run the MCP server on stdio."""
    server = SimpleMCPServer()

    # Read from stdin, write to stdout
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(
                None, sys.stdin.readline
            )
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            message = json.loads(line)
            response = await server.handle_message(message)

            print(json.dumps(response), flush=True)

        except json.JSONDecodeError as e:
            error = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            print(json.dumps(error), flush=True)
        except Exception as e:
            # Log to stderr (won't interfere with protocol)
            print(f"Server error: {e}", file=sys.stderr)
            break


if __name__ == "__main__":
    asyncio.run(main())
'''


@pytest.fixture
def mcp_server_script(tmp_path):
    """Create a temporary MCP server script for testing."""
    script_path = tmp_path / "test_mcp_server.py"
    script_path.write_text(TEST_SERVER_SCRIPT)
    return str(script_path)


# ============================================================================
# End-to-End Integration Tests
# ============================================================================


class TestMCPServerClientE2E:
    """End-to-end tests for MCP server and client communication.

    These tests start a real server subprocess and connect with a real client.
    """

    @pytest.mark.asyncio
    async def test_client_connects_to_server(self, mcp_server_script):
        """Test that client can connect to server and initialize."""
        from victor.integrations.mcp.client import MCPClient

        client = MCPClient(
            name="test-client",
            version="1.0.0",
            health_check_interval=0,  # Disable health checks for test
        )

        try:
            # Connect to test server
            connected = await client.connect([sys.executable, mcp_server_script])
            assert connected, "Client should connect successfully"

            # Initialize the connection
            initialized = await client.initialize()
            assert initialized, "Client should initialize successfully"

            # Verify server info was received
            assert client.server_info is not None
            assert client.server_info.name == "test-mcp-server"
            assert client.server_info.version == "1.0.0"

        finally:
            await client.cleanup()

    @pytest.mark.asyncio
    async def test_discover_tools_from_server(self, mcp_server_script):
        """Test that client can discover tools from server."""
        from victor.integrations.mcp.client import MCPClient

        client = MCPClient(health_check_interval=0)

        try:
            await client.connect([sys.executable, mcp_server_script])
            await client.initialize()

            # Refresh tools from server
            tools = await client.refresh_tools()

            assert len(tools) == 2
            tool_names = [t.name for t in tools]
            assert "test_echo" in tool_names
            assert "test_add" in tool_names

            # Verify tool metadata
            echo_tool = next(t for t in tools if t.name == "test_echo")
            assert "echo" in echo_tool.description.lower()

        finally:
            await client.cleanup()

    @pytest.mark.asyncio
    async def test_call_tool_on_server(self, mcp_server_script):
        """Test calling a tool on the server and getting results."""
        from victor.integrations.mcp.client import MCPClient

        client = MCPClient(health_check_interval=0)

        try:
            await client.connect([sys.executable, mcp_server_script])
            await client.initialize()
            await client.refresh_tools()

            # Call test_echo tool (note: call_tool uses **kwargs)
            result = await client.call_tool("test_echo", message="Hello MCP!")

            assert result is not None
            assert result.success is True
            assert result.error is None
            # The result should contain the echoed message
            assert "Hello MCP!" in str(result.result)

        finally:
            await client.cleanup()

    @pytest.mark.asyncio
    async def test_call_tool_with_computation(self, mcp_server_script):
        """Test calling a tool that performs computation."""
        from victor.integrations.mcp.client import MCPClient

        client = MCPClient(health_check_interval=0)

        try:
            await client.connect([sys.executable, mcp_server_script])
            await client.initialize()
            await client.refresh_tools()

            # Call test_add tool
            result = await client.call_tool("test_add", a=5, b=3)

            assert result is not None
            assert result.success is True
            # Result should contain "8"
            assert "8" in str(result.result)

        finally:
            await client.cleanup()

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool(self, mcp_server_script):
        """Test calling a tool that doesn't exist."""
        from victor.integrations.mcp.client import MCPClient

        client = MCPClient(health_check_interval=0)

        try:
            await client.connect([sys.executable, mcp_server_script])
            await client.initialize()

            # Call non-existent tool
            result = await client.call_tool("nonexistent_tool")

            assert result is not None
            assert result.success is False
            assert result.error is not None
            assert "unknown" in result.error.lower() or "not found" in result.error.lower()

        finally:
            await client.cleanup()

    @pytest.mark.asyncio
    async def test_discover_resources_from_server(self, mcp_server_script):
        """Test that client can discover resources from server."""
        from victor.integrations.mcp.client import MCPClient

        client = MCPClient(health_check_interval=0)

        try:
            await client.connect([sys.executable, mcp_server_script])
            await client.initialize()

            # Refresh resources from server
            resources = await client.refresh_resources()

            assert len(resources) == 1
            assert resources[0].uri == "test://greeting"
            assert resources[0].name == "greeting"

        finally:
            await client.cleanup()

    @pytest.mark.asyncio
    async def test_read_resource_from_server(self, mcp_server_script):
        """Test reading a resource from the server."""
        from victor.integrations.mcp.client import MCPClient

        client = MCPClient(health_check_interval=0)

        try:
            await client.connect([sys.executable, mcp_server_script])
            await client.initialize()

            # Read the greeting resource
            content = await client.read_resource("test://greeting")

            assert content is not None
            assert "Hello from MCP!" in str(content)

        finally:
            await client.cleanup()

    @pytest.mark.asyncio
    async def test_ping_server(self, mcp_server_script):
        """Test pinging the server for health check."""
        from victor.integrations.mcp.client import MCPClient

        client = MCPClient(health_check_interval=0)

        try:
            await client.connect([sys.executable, mcp_server_script])
            await client.initialize()

            # Ping the server
            is_healthy = await client.ping()
            assert is_healthy is True

        finally:
            await client.cleanup()

    @pytest.mark.asyncio
    async def test_client_as_context_manager(self, mcp_server_script):
        """Test using client as async context manager."""
        from victor.integrations.mcp.client import MCPClient

        client = MCPClient(
            health_check_interval=0,
            command=[sys.executable, mcp_server_script],
        )

        async with client:
            # Should be connected and initialized
            assert client.initialized is True

            # Should be able to call tools
            result = await client.call_tool("test_echo", message="Context manager test")
            assert result.success is True

        # After exit, process should be cleaned up
        assert client.process is None or client.process.poll() is not None

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, mcp_server_script):
        """Test making multiple sequential tool calls."""
        from victor.integrations.mcp.client import MCPClient

        client = MCPClient(health_check_interval=0)

        try:
            await client.connect([sys.executable, mcp_server_script])
            await client.initialize()

            # Make multiple calls
            for i in range(5):
                result = await client.call_tool("test_add", a=i, b=10)
                assert result.success is True
                assert str(i + 10) in str(result.result)

        finally:
            await client.cleanup()


# ============================================================================
# Unit-like Tests (in-process, using pipes - for faster testing)
# ============================================================================


class TestMCPProtocolHandling:
    """Unit-like tests for MCP protocol message handling.

    These tests directly call the server's handle_message method without
    spawning a subprocess, making them faster but less realistic.
    """

    @pytest.mark.asyncio
    async def test_server_initialize_message(self):
        """Test server handles initialize message correctly."""
        from victor.integrations.mcp.server import MCPServer
        from victor.tools.registry import ToolRegistry

        server = MCPServer(name="test", version="1.0.0", tool_registry=ToolRegistry())

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            }
        )

        assert response["id"] == "1"
        assert "result" in response
        assert server.initialized is True

    @pytest.mark.asyncio
    async def test_server_list_tools_message(self):
        """Test server handles list tools message correctly."""
        from victor.integrations.mcp.server import MCPServer
        from victor.tools.base import BaseTool, CostTier
        from victor.tools.registry import ToolRegistry

        # Create a test tool
        class TestTool(BaseTool):
            name = "test_tool"
            description = "A test tool"
            cost_tier = CostTier.FREE
            parameters = {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return "test"

        registry = ToolRegistry()
        registry.register(TestTool())

        server = MCPServer(name="test", version="1.0.0", tool_registry=registry)
        server.initialized = True

        response = await server.handle_message(
            {"jsonrpc": "2.0", "id": "2", "method": "tools/list", "params": {}}
        )

        assert response["id"] == "2"
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) == 1
        assert response["result"]["tools"][0]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_server_ping_message(self):
        """Test server handles ping message correctly."""
        from victor.integrations.mcp.server import MCPServer
        from victor.tools.registry import ToolRegistry

        server = MCPServer(tool_registry=ToolRegistry())
        server.initialized = True

        response = await server.handle_message(
            {"jsonrpc": "2.0", "id": "3", "method": "ping", "params": {}}
        )

        assert response["id"] == "3"
        assert "result" in response

    @pytest.mark.asyncio
    async def test_server_error_on_uninitialized(self):
        """Test server returns error when not initialized."""
        from victor.integrations.mcp.server import MCPServer
        from victor.tools.registry import ToolRegistry

        server = MCPServer(tool_registry=ToolRegistry())
        # Don't initialize

        response = await server.handle_message(
            {"jsonrpc": "2.0", "id": "4", "method": "tools/list", "params": {}}
        )

        assert response["id"] == "4"
        assert "error" in response
        assert response["error"]["code"] == -32002  # Not initialized


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
