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

"""Demo of Victor's MCP Server.

This demonstrates how Victor can expose its tools through the Model Context
Protocol (MCP), allowing other MCP clients to use Victor's capabilities.

Usage:
    # Start MCP server on stdio (for MCP clients)
    python examples/mcp_server_demo.py --stdio

    # Or run demo showing server capabilities
    python examples/mcp_server_demo.py
"""

import asyncio
import argparse
from victor.integrations.mcp import MCPServer, MCPResource
from victor.tools.registry import ToolRegistry
from victor.tools.bash import shell
from victor.tools.filesystem import read, write, ls
from victor.tools.git_tool import git


async def demo_server():
    """Demo MCP server capabilities."""
    print("🎯 Victor MCP Server Demo")
    print("=" * 70)
    print("\nModel Context Protocol (MCP) allows other applications")
    print("to discover and use Victor's tools through a standard interface.\n")

    # Create tool registry with Victor's tools
    print("1️⃣ Setting up tool registry...")
    print("-" * 70)
    tool_registry = ToolRegistry()
    tool_registry.register(read)
    tool_registry.register(write)
    tool_registry.register(ls)
    tool_registry.register(shell)
    tool_registry.register(git)

    tools_list = tool_registry.list_tools()
    print(f"✓ Registered {len(tools_list)} tools:")
    for tool in tools_list:
        print(f"  - {tool.name}")

    # Create MCP server
    print("\n2️⃣ Creating MCP Server...")
    print("-" * 70)
    server = MCPServer(name="Victor MCP Server", version="1.0.0", tool_registry=tool_registry)

    print(f"✓ Server created: {server.name} v{server.version}")
    print("  Capabilities:")
    print(f"    - Tools: {server.info.capabilities.tools}")
    print(f"    - Resources: {server.info.capabilities.resources}")

    # Register some resources
    print("\n3️⃣ Registering resources...")
    print("-" * 70)

    resources = [
        MCPResource(
            uri="file://./README.md",
            name="Victor README",
            description="Victor project documentation",
            mime_type="text/markdown",
        ),
        MCPResource(
            uri="file://./victor/agent/orchestrator.py",
            name="Agent Orchestrator",
            description="Victor's agent orchestration code",
            mime_type="text/x-python",
        ),
        MCPResource(
            uri="file://./examples/mcp_server_demo.py",
            name="MCP Server Demo",
            description="This demo file",
            mime_type="text/x-python",
        ),
    ]

    for resource in resources:
        server.register_resource(resource)
        print(f"✓ Registered: {resource.name} ({resource.uri})")

    # Simulate MCP messages
    print("\n4️⃣ Testing MCP protocol...")
    print("-" * 70)

    # Initialize
    print("\nTest 1: Initialize")
    init_msg = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "initialize",
        "params": {"clientInfo": {"name": "Demo Client", "version": "1.0.0"}},
    }
    response = await server.handle_message(init_msg)
    print(f"✓ Initialized: {response['result']['serverInfo']['name']}")

    # List tools
    print("\nTest 2: List Tools")
    list_tools_msg = {
        "jsonrpc": "2.0",
        "id": "2",
        "method": "tools/list",
    }
    response = await server.handle_message(list_tools_msg)
    if "error" in response:
        print(f"✗ Error: {response['error']}")
    else:
        tools = response["result"]["tools"]
        print(f"✓ Found {len(tools)} tools:")
        for tool in tools[:3]:  # Show first 3
            print(f"  - {tool['name']}: {tool['description'][:50]}...")

    # List resources
    print("\nTest 3: List Resources")
    list_resources_msg = {
        "jsonrpc": "2.0",
        "id": "3",
        "method": "resources/list",
    }
    response = await server.handle_message(list_resources_msg)
    resources_list = response["result"]["resources"]
    print(f"✓ Found {len(resources_list)} resources:")
    for resource in resources_list:
        print(f"  - {resource['name']}: {resource['uri']}")

    # Call a tool
    print("\nTest 4: Call Tool (ls)")
    call_tool_msg = {
        "jsonrpc": "2.0",
        "id": "4",
        "method": "tools/call",
        "params": {"name": "ls", "arguments": {"path": "."}},
    }
    response = await server.handle_message(call_tool_msg)
    result = response["result"]
    print(f"✓ Tool call {'succeeded' if result['success'] else 'failed'}")
    if result["success"]:
        output = result["result"][:200]
        print(f"  Output (preview): {output}...")

    # Ping
    print("\nTest 5: Ping")
    ping_msg = {
        "jsonrpc": "2.0",
        "id": "5",
        "method": "ping",
    }
    response = await server.handle_message(ping_msg)
    print(f"✓ Ping response: {response['result']}")

    # Show server info
    print("\n5️⃣ Server Information")
    print("-" * 70)
    info = server.get_server_info()
    print(f"Name: {info['name']}")
    print(f"Version: {info['version']}")
    print(f"Tools: {info['tools_count']}")
    print(f"Resources: {info['resources_count']}")
    print(f"Capabilities: {info['capabilities']}")

    print("\n\n✨ Demo Complete!")
    print("\nMCP Server Features:")
    print("  ✓ Exposes Victor's tools via standard protocol")
    print("  ✓ Provides resource access (files, data)")
    print("  ✓ JSON-RPC 2.0 based communication")
    print("  ✓ Stdio transport for client integration")
    print("  ✓ Full MCP specification compliance")

    print("\n\n📚 Usage Examples:")
    print("""
# Run as stdio server (for MCP clients to connect)
python examples/mcp_server_demo.py --stdio

# Then from another terminal, connect with an MCP client:
python examples/mcp_client_demo.py

# Or use with any MCP-compatible client:
# - Claude Desktop
# - VS Code with MCP extension
# - Custom MCP clients
""")

    print("\n\n🔌 Integration:")
    print("""
Victor can now be used by any MCP-compatible application!

Example with Claude Desktop:
1. Add to Claude Desktop config:
   {
     "mcpServers": {
       "victor": {
         "command": "python",
         "args": ["examples/mcp_server_demo.py", "--stdio"]
       }
     }
   }

2. Restart Claude Desktop

3. Victor's tools are now available in Claude Desktop!
   - File operations
   - Git commands
   - Bash execution
   - And more!
""")


async def run_stdio_server():
    """Run MCP server on stdio."""
    # Create tool registry
    tool_registry = ToolRegistry()
    tool_registry.register(read)
    tool_registry.register(write)
    tool_registry.register(ls)
    tool_registry.register(shell)
    tool_registry.register(git)

    # Create and start server
    server = MCPServer(name="Victor MCP Server", version="1.0.0", tool_registry=tool_registry)

    # Register resources
    server.register_resource(
        MCPResource(
            uri="file://./README.md",
            name="Victor README",
            description="Victor project documentation",
            mime_type="text/markdown",
        )
    )

    # Start stdio server
    await server.start_stdio_server()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Victor MCP Server Demo")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run as stdio server (for MCP client connections)",
    )

    args = parser.parse_args()

    if args.stdio:
        # Run as stdio server
        asyncio.run(run_stdio_server())
    else:
        # Run demo
        asyncio.run(demo_server())


if __name__ == "__main__":
    main()
