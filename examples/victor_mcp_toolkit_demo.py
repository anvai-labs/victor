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

"""Victor MCP Toolkit Demo.

This demo showcases how to run Victor as an MCP server toolkit, configure which
tools to expose, and connect from external clients.

Victor exposes 33 tool modules across 9 domain verticals:
- Coding: file operations, git, code search, testing
- DevOps: Docker, Terraform, CI/CD, Kubernetes
- RAG: document ingestion, vector search, retrieval
- Data Analysis: Pandas, visualization, statistics
- Research: web search, summarization, citations
- Security: vulnerability scanning, audit, compliance
- IaC: infrastructure as code management
- Classification: text/data classification pipelines
- Benchmark: agent evaluation and benchmarking

Usage:
    # Run demo showing MCP server capabilities
    python examples/victor_mcp_toolkit_demo.py

    # Start MCP server in stdio mode (for Claude Desktop)
    python examples/victor_mcp_toolkit_demo.py --stdio

    # Start with specific tools only
    python examples/victor_mcp_toolkit_demo.py --stdio --tools read,write,ls,git

    # Start with specific vertical
    python examples/victor_mcp_toolkit_demo.py --stdio --vertical coding

    # Show available tools without starting server
    python examples/victor_mcp_toolkit_demo.py --list-tools
"""

import argparse
import asyncio
import importlib
import inspect
import os
import sys
from typing import List, Optional, Set

from victor.integrations.mcp import MCPResource, MCPServer
from victor.tools.base import BaseTool
from victor.tools.registry import ToolRegistry

# Tool categories by vertical
VERTICAL_TOOLS = {
    "coding": [
        "read",
        "write",
        "edit",
        "ls",
        "git",
        "shell",
        "grep",
        "find_files",
        "code_search",
        "code_review",
        "test_generation",
        "refactor",
        "architecture_summary",
        "dependency_graph",
    ],
    "devops": [
        "docker",
        "terraform",
        "cicd",
        "kubernetes",
        "aws",
        "helm",
        "ansible",
    ],
    "rag": [
        "ingest_document",
        "vector_search",
        "retrieve",
        "chunk_document",
        "embed",
    ],
    "dataanalysis": [
        "pandas_query",
        "visualize",
        "statistics",
        "dataframe_info",
        "plot",
    ],
    "research": [
        "web_search",
        "fetch_url",
        "summarize",
        "cite",
        "extract_entities",
    ],
}


def discover_tools(
    filter_tools: Optional[List[str]] = None,
    filter_vertical: Optional[str] = None,
) -> ToolRegistry:
    """Dynamically discover and register Victor tools.

    Args:
        filter_tools: Optional list of specific tool names to include
        filter_vertical: Optional vertical name to filter by

    Returns:
        ToolRegistry with discovered tools
    """
    registry = ToolRegistry()

    # Build allowed tools set
    allowed_tools: Optional[Set[str]] = None
    if filter_tools:
        allowed_tools = set(filter_tools)
    elif filter_vertical and filter_vertical in VERTICAL_TOOLS:
        allowed_tools = set(VERTICAL_TOOLS[filter_vertical])

    # Dynamic tool discovery
    tools_dir = os.path.join(os.path.dirname(__file__), "..", "victor", "tools")
    tools_dir = os.path.abspath(tools_dir)

    excluded_files = {
        "__init__.py",
        "base.py",
        "decorators.py",
        "semantic_selector.py",
        "common.py",
        "enums.py",
        "context.py",
        "alias_resolver.py",
        "hybrid_tool_selector.py",
    }

    registered_count = 0

    for filename in os.listdir(tools_dir):
        if filename.endswith(".py") and filename not in excluded_files:
            module_name = f"victor.tools.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for _name, obj in inspect.getmembers(module):
                    # Register @tool decorated functions
                    if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                        tool_name = getattr(obj, "name", _name)
                        if allowed_tools is None or tool_name in allowed_tools:
                            registry.register(obj)
                            registered_count += 1
                    # Register BaseTool class instances
                    elif (
                        inspect.isclass(obj)
                        and issubclass(obj, BaseTool)
                        and obj is not BaseTool
                        and hasattr(obj, "name")
                    ):
                        try:
                            tool_instance = obj()
                            if allowed_tools is None or tool_instance.name in allowed_tools:
                                registry.register(tool_instance)
                                registered_count += 1
                        except Exception:
                            pass  # Skip tools that need special initialization
            except Exception as e:
                print(f"Warning: Could not load {module_name}: {e}", file=sys.stderr)

    return registry


def list_available_tools():
    """List all available tools organized by vertical."""
    print("Victor MCP Server - Available Tools")
    print("=" * 70)
    print()

    registry = discover_tools()
    all_tools = {tool.name: tool for tool in registry.list_tools()}

    for vertical, tool_names in VERTICAL_TOOLS.items():
        print(f"\n{vertical.upper()} VERTICAL")
        print("-" * 40)
        for name in tool_names:
            if name in all_tools:
                tool = all_tools[name]
                desc = (
                    tool.description[:50] + "..."
                    if len(tool.description) > 50
                    else tool.description
                )
                print(f"  {name:25} {desc}")
            else:
                print(f"  {name:25} (not loaded)")

    # List any tools not in a vertical
    vertical_tool_set = set()
    for tools in VERTICAL_TOOLS.values():
        vertical_tool_set.update(tools)

    other_tools = [t for t in all_tools.keys() if t not in vertical_tool_set]
    if other_tools:
        print("\nOTHER TOOLS")
        print("-" * 40)
        for name in sorted(other_tools):
            tool = all_tools[name]
            desc = tool.description[:50] + "..." if len(tool.description) > 50 else tool.description
            print(f"  {name:25} {desc}")

    print()
    print(f"Total tools available: {len(all_tools)}")


async def run_demo():
    """Run interactive demo of MCP server capabilities."""
    print("Victor MCP Toolkit Demo")
    print("=" * 70)
    print()
    print("Victor can be run as an MCP server, exposing its 33 tool modules to external")
    print("clients like Claude Desktop, VS Code, or any MCP-compatible application.")
    print()

    # Discover tools
    print("1. TOOL DISCOVERY")
    print("-" * 40)
    registry = discover_tools()
    tools = registry.list_tools()
    print(f"   Discovered {len(tools)} tools from Victor's tool registry")
    print()

    # Show tools by vertical
    print("2. TOOLS BY VERTICAL")
    print("-" * 40)
    for vertical, tool_names in VERTICAL_TOOLS.items():
        available = sum(1 for t in tools if t.name in tool_names)
        print(f"   {vertical:15} {available:3} tools")
    print()

    # Create MCP server
    print("3. MCP SERVER CREATION")
    print("-" * 40)
    server = MCPServer(
        name="Victor MCP Toolkit",
        version="1.0.0",
        tool_registry=registry,
    )
    print(f"   Server: {server.name} v{server.version}")
    print(
        f"   Capabilities: tools={server.info.capabilities.tools}, "
        f"resources={server.info.capabilities.resources}"
    )
    print()

    # Register sample resources
    print("4. RESOURCE REGISTRATION")
    print("-" * 40)
    resources = [
        MCPResource(
            uri="file://./README.md",
            name="Victor README",
            description="Victor project documentation",
            mime_type="text/markdown",
        ),
        MCPResource(
            uri="file://./CLAUDE.md",
            name="Claude Code Instructions",
            description="Instructions for Claude Code",
            mime_type="text/markdown",
        ),
    ]
    for resource in resources:
        server.register_resource(resource)
        print(f"   Registered: {resource.name} ({resource.uri})")
    print()

    # Test MCP protocol
    print("5. MCP PROTOCOL TEST")
    print("-" * 40)

    # Initialize
    init_response = await server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "initialize",
            "params": {"clientInfo": {"name": "Demo Client", "version": "1.0.0"}},
        }
    )
    print(f"   Initialize: {init_response['result']['serverInfo']['name']}")

    # List tools
    list_response = await server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tools/list",
        }
    )
    tools_count = len(list_response["result"]["tools"])
    print(f"   List tools: {tools_count} tools available")

    # List resources
    resources_response = await server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": "3",
            "method": "resources/list",
        }
    )
    resources_count = len(resources_response["result"]["resources"])
    print(f"   List resources: {resources_count} resources registered")

    # Ping
    ping_response = await server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": "4",
            "method": "ping",
        }
    )
    print(f"   Ping: {ping_response['result']}")
    print()

    # Usage instructions
    print("6. USAGE INSTRUCTIONS")
    print("-" * 40)
    print("""
   To use Victor as an MCP server with Claude Desktop:

   1. Add to Claude Desktop config:
      (macOS: ~/Library/Application Support/Claude/claude_desktop_config.json)

      {
        "mcpServers": {
          "victor": {
            "command": "victor",
            "args": ["mcp"]
          }
        }
      }

   2. Restart Claude Desktop

   3. Victor's tools are now available!

   For Docker deployment:
      docker run -it victor-ai/mcp-server

   For filtered tools (coding vertical only):
      python examples/victor_mcp_toolkit_demo.py --stdio --vertical coding
""")

    print("=" * 70)
    print("Demo complete!")


async def run_stdio_server(
    filter_tools: Optional[List[str]] = None,
    filter_vertical: Optional[str] = None,
):
    """Run MCP server in stdio mode.

    Args:
        filter_tools: Optional list of specific tool names to expose
        filter_vertical: Optional vertical name to filter tools by
    """
    # Discover tools with filters
    registry = discover_tools(filter_tools=filter_tools, filter_vertical=filter_vertical)
    tools_count = len(registry.list_tools())

    # Create MCP server
    server = MCPServer(
        name="Victor MCP Toolkit",
        version="1.0.0",
        tool_registry=registry,
    )

    # Register default resources
    server.register_resource(
        MCPResource(
            uri="file://./README.md",
            name="Victor README",
            description="Victor project documentation",
            mime_type="text/markdown",
        )
    )

    print(f"Victor MCP Server starting with {tools_count} tools", file=sys.stderr)

    if filter_vertical:
        print(f"Vertical filter: {filter_vertical}", file=sys.stderr)
    elif filter_tools:
        print(f"Tool filter: {', '.join(filter_tools)}", file=sys.stderr)

    await server.start_stdio_server()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Victor MCP Toolkit Demo - Run Victor as an MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo
  python examples/victor_mcp_toolkit_demo.py

  # Start MCP server for Claude Desktop
  python examples/victor_mcp_toolkit_demo.py --stdio

  # Expose only specific tools
  python examples/victor_mcp_toolkit_demo.py --stdio --tools read,write,ls,git

  # Expose only coding vertical
  python examples/victor_mcp_toolkit_demo.py --stdio --vertical coding

  # List all available tools
  python examples/victor_mcp_toolkit_demo.py --list-tools
""",
    )

    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run as stdio server (for MCP client connections)",
    )
    parser.add_argument(
        "--tools",
        type=str,
        help="Comma-separated list of tools to expose (e.g., read,write,ls)",
    )
    parser.add_argument(
        "--vertical",
        type=str,
        choices=list(VERTICAL_TOOLS.keys()),
        help="Expose only tools from a specific vertical",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List all available tools and exit",
    )

    args = parser.parse_args()

    if args.list_tools:
        list_available_tools()
        return

    filter_tools = None
    if args.tools:
        filter_tools = [t.strip() for t in args.tools.split(",")]

    if args.stdio:
        asyncio.run(
            run_stdio_server(
                filter_tools=filter_tools,
                filter_vertical=args.vertical,
            )
        )
    else:
        asyncio.run(run_demo())


if __name__ == "__main__":
    main()
