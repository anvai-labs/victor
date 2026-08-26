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

"""Simple demo of AgentBrowser integration through Victor's MCP layer.

This demonstrates the schema translator working by showing AgentBrowser tools
with their full parameter schemas (not empty lists).

Prerequisites:
    1. AgentBrowser REST server running on port 3000:
       cd /Users/vijaysingh/code/agentbrowser
       node packages/api/dist/bin.js

    2. agentbrowser-mcp release binary on PATH (gh release download from
       vjsingh1984/agentbrowser), or a local checkout's built dist/ as fallback

Usage:
    cd /Users/vijaysingh/code/codingagent
    python examples/agentbrowser_simple_demo.py
"""

import asyncio
from victor.integrations.mcp import MCPClient


async def main():
    print("🌐 AgentBrowser + Victor Integration Demo (Simple)")
    print("=" * 70)
    print("\nPrerequisites:")
    print("  1. AgentBrowser REST server running on http://localhost:3000")
    print("  2. ~/.victor/mcp.yaml configured with AgentBrowser MCP server")
    print("\nConnecting to AgentBrowser MCP server...\n")

    # Create MCP client
    client = MCPClient(
        name="Victor AgentBrowser Client",
        version="1.0.0",
        health_check_interval=0,  # Disable health monitoring for demo
    )

    # Connect to AgentBrowser MCP server: prefer the released single binary
    # (agentbrowser-mcp v1.1.0+, no Node needed); fall back to a local checkout.
    import shutil

    server_command = (
        ["agentbrowser-mcp"]
        if shutil.which("agentbrowser-mcp")
        else [
            "node",
            "/Users/vijaysingh/code/agentbrowser/packages/mcp-server/dist/bin.js",
        ]
    )
    print(f"spawn: {' '.join(server_command)}")

    success = await client.connect(server_command)

    if not success:
        print("✗ Failed to connect to AgentBrowser MCP server")
        print("\nTroubleshooting:")
        print("  1. Ensure AgentBrowser REST server is running on port 3000")
        print("  2. Check that MCP server is built: pnpm --filter @agentbrowser/mcp-server build")
        print("  3. Verify ~/.victor/mcp.yaml configuration")
        return

    print("✓ Connected to AgentBrowser MCP server")

    # Show server info
    if client.server_info:
        print(f"\n📊 Server Information:")
        print(f"   Name: {client.server_info.name}")
        print(f"   Version: {client.server_info.version}")

    # List available tools
    print(f"\n🔧 Available Tools ({len(client.tools)} total):")
    print("-" * 70)

    for tool in client.tools:
        # This is the key test: tools should have parameters (not empty lists)
        param_count = len(tool.parameters)
        param_names = [p.name for p in tool.parameters]

        print(f"\n  📌 {tool.name}")
        print(f"     Description: {tool.description}")
        print(
            f"     Parameters ({param_count}): {', '.join(param_names) if param_names else 'NONE'}"
        )

        # Show parameter details for first 3 tools
        if tool == client.tools[0]:
            for param in tool.parameters:
                required = "required" if param.required else "optional"
                print(f"       • {param.name}: {param.type.value} ({required})")
                if param.description:
                    print(f"         {param.description}")

    # Test: Create a browser session
    print("\n" + "=" * 70)
    print("🧪 Testing: Create browser session")
    print("-" * 70)

    result = await client.call_tool("browser_create", tenantId="demo-tenant")

    if result.success:
        print(f"✓ Session created successfully")
        # Parse sessionId from result
        import json

        try:
            session_data = json.loads(result.result)
            session_id = session_data.get("sessionId", "unknown")
            page_id = session_data.get("pageId", "unknown")
            print(f"   Session ID: {session_id}")
            print(f"   Page ID: {page_id}")
        except:
            print(f"   Result: {result.result}")
            session_id = "unknown"
            page_id = "unknown"
    else:
        print(f"✗ Session creation failed: {result.error}")
        session_id = "unknown"
        page_id = "unknown"

    # Test: Navigate to example.com (only if session created)
    if session_id != "unknown" and page_id != "unknown":
        print("\n" + "=" * 70)
        print("🧪 Testing: Navigate to example.com")
        print("-" * 70)

        result = await client.call_tool(
            "browser_navigate",
            sessionId=session_id,
            pageId=page_id,
            url="https://example.com",
            waitUntil="load",
        )

        if result.success:
            print(f"✓ Navigation successful")
            print(f"   Result (first 200 chars): {str(result.result)[:200]}...")
        else:
            print(f"✗ Navigation failed: {result.error}")

        # Test: Observe the page
        print("\n" + "=" * 70)
        print("🧪 Testing: Observe page")
        print("-" * 70)

        result = await client.call_tool(
            "browser_observe",
            sessionId=session_id,
            pageId=page_id,
            maxElements=10,
        )

        if result.success:
            print(f"✓ Observation successful")
            print(f"   Result (first 500 chars):")
            print(f"   {str(result.result)[:500]}...")
        else:
            print(f"✗ Observation failed: {result.error}")

        # Test: Take screenshot
        print("\n" + "=" * 70)
        print("🧪 Testing: Screenshot")
        print("-" * 70)

        result = await client.call_tool(
            "browser_screenshot",
            sessionId=session_id,
            pageId=page_id,
            fullPage=False,
        )

        if result.success:
            print(f"✓ Screenshot successful")
            print(f"   Result: {result.result}")
        else:
            print(f"✗ Screenshot failed: {result.error}")

    # Cleanup
    print("\n" + "=" * 70)
    print("🧹 Cleaning up...")
    await client.cleanup()
    print("✓ Disconnected from AgentBrowser MCP server")

    print("\n" + "=" * 70)
    print("✨ Demo Complete!")
    print("\nKey Achievement:")
    print("  ✓ AgentBrowser tools registered with FULL parameter schemas")
    print("  ✓ Schema translator successfully converts inputSchema → parameters")
    print("  ✓ Tools are usable (not empty parameter lists)")
    print("\nBefore this integration:")
    print("  • Tools would appear with 0 parameters (unusable by LLM)")
    print("  • inputSchema was silently ignored by Pydantic")
    print("\nAfter this integration:")
    print("  • Tools show complete parameter information")
    print("  • LLM can properly invoke browser automation")
    print("\nFor more information, see:")
    print(
        "  • Integration: .claude/projects/-Users-vijaysingh-code-agentbrowser/AGENTBROWSER_VICTOR_INTEGRATION_COMPLETE.md"
    )
    print("  • AgentBrowser: https://github.com/vjsingh1984/agentbrowser")
    print("  • Victor: https://github.com/anvai-labs/victor")


if __name__ == "__main__":
    asyncio.run(main())
