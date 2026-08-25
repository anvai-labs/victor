#!/usr/bin/env python3
"""Drive AgentBrowser with visible browser (headless=False) through MCP.

This script connects to AgentBrowser's MCP server and performs browser automation
that you can watch in real-time.

Prerequisites:
    1. AgentBrowser REST server running with headless=False:
       AGENTBROWSER_HEADLESS=false node /path/to/agentbrowser/packages/api/dist/bin.js

    2. AgentBrowser MCP server available

Usage:
    python examples/agentbrowser_visible_demo.py
"""

import asyncio
import json
import time
from victor.integrations.mcp import MCPClient


async def main():
    print("🌐 AgentBrowser Visible Browser Demo")
    print("=" * 70)
    print("\n🎬 Watch your screen — browser automation will be VISIBLE!\n")

    # Create MCP client
    client = MCPClient(
        name="Victor Visible Browser Client",
        version="1.0.0",
        health_check_interval=0,
    )

    # Connect to AgentBrowser MCP server
    server_command = [
        "node",
        "/Users/vijaysingh/code/agentbrowser/packages/mcp-server/dist/bin.js",
    ]

    print("📡 Connecting to AgentBrowser MCP server...")
    success = await client.connect(server_command)

    if not success:
        print("✗ Failed to connect to AgentBrowser MCP server")
        print("\nMake sure AgentBrowser REST server is running:")
        print("  AGENTBROWSER_HEADLESS=false node /Users/vijaysingh/code/agentbrowser/packages/api/dist/bin.js")
        return

    print("✓ Connected to AgentBrowser MCP server")

    # Show available tools
    print(f"\n🔧 Available Tools: {len(client.tools)}")
    for tool in client.tools:
        print(f"   • {tool.name} ({len(tool.parameters)} parameters)")

    # Create a browser session with headless=False
    print("\n" + "=" * 70)
    print("🎬 Step 1: Creating browser session with headless=False")
    print("       👀 Watch for browser window to open!")
    print("-" * 70)

    result = await client.call_tool(
        "browser_create",
        tenantId="visible-demo",
        headless=False,  # This makes the browser visible!
        ttlMs=300000,  # 5 minutes
    )

    if result.success:
        print("✓ Browser session created")
        session_data = json.loads(result.result)
        session_id = session_data.get("sessionId")
        page_id = session_data.get("pageId")
        print(f"   Session ID: {session_id}")
        print(f"   Page ID: {page_id}")
        print("\n   👀 You should see a Chromium browser window open now!")
    else:
        print(f"✗ Failed to create session: {result.error}")
        await client.cleanup()
        return

    # Wait a moment for the browser to become visible
    print("\n⏳ Waiting 2 seconds for browser to be fully visible...")
    await asyncio.sleep(2)

    # Navigate to example.com
    print("\n" + "=" * 70)
    print("🎬 Step 2: Navigating to https://example.com")
    print("       👀 Watch the browser load the page!")
    print("-" * 70)

    result = await client.call_tool(
        "browser_navigate",
        sessionId=session_id,
        pageId=page_id,
        url="https://example.com",
        waitUntil="load",
    )

    if result.success:
        print("✓ Navigation successful")
        nav_result = json.loads(result.result)
        print(f"   URL: {nav_result.get('url')}")
        print("\n   👀 You should see example.com loaded in the browser!")
    else:
        print(f"✗ Navigation failed: {result.error}")

    # Wait for user to see the page
    print("\n⏳ Holding for 3 seconds so you can see the page...")
    await asyncio.sleep(3)

    # Observe the page
    print("\n" + "=" * 70)
    print("🎬 Step 3: Observing page elements")
    print("       👀 Analyzing page structure...")
    print("-" * 70)

    result = await client.call_tool(
        "browser_observe",
        sessionId=session_id,
        pageId=page_id,
        maxElements=20,
    )

    if result.success:
        print("✓ Observation successful")
        obs_result = json.loads(result.result)

        print(f"   Title: {obs_result.get('title')}")
        print(f"   URL: {obs_result.get('url')}")
        print(f"   Revision: {obs_result.get('revision')}")
        print(f"   Summary: {obs_result.get('summary')}")

        elements = obs_result.get('elements', [])
        print(f"\n   Elements found ({len(elements)}):")

        for i, elem in enumerate(elements[:5], 1):
            ref = elem.get('ref')
            role = elem.get('role')
            visible = "👁" if elem.get('visible') else "🚫"
            print(f"      {i}. [{ref}] {role} {visible}")

        if len(elements) > 5:
            print(f"      ... and {len(elements) - 5} more")

    # Navigate to a more interactive page
    print("\n" + "=" * 70)
    print("🎬 Step 4: Navigating to Wikipedia (more interactive)")
    print("       👀 Watch the browser load a complex page!")
    print("-" * 70)

    result = await client.call_tool(
        "browser_navigate",
        sessionId=session_id,
        pageId=page_id,
        url="https://en.wikipedia.org/wiki/Artificial_intelligence",
        waitUntil="networkidle",
    )

    if result.success:
        print("✓ Navigation successful")
        print("\n   👀 You should see Wikipedia loading in the browser!")
        print("      This demonstrates handling complex, JavaScript-heavy pages")

    # Wait for user to see the page
    print("\n⏳ Holding for 5 seconds so you can see the page...")
    await asyncio.sleep(5)

    # Take a screenshot
    print("\n" + "=" * 70)
    print("🎬 Step 5: Taking screenshot of visible browser")
    print("       👀 Capturing what you see!")
    print("-" * 70)

    result = await client.call_tool(
        "browser_screenshot",
        sessionId=session_id,
        pageId=page_id,
        fullPage=False,
    )

    if result.success:
        print("✓ Screenshot successful")
        ss_result = json.loads(result.result)
        print(f"   Artifact ID: {ss_result.get('artifactId')}")
        print(f"   Size: {ss_result.get('sizeBytes')} bytes")
        print(f"   Type: {ss_result.get('contentType')}")

    # Final observation
    print("\n" + "=" * 70)
    print("🎬 Step 6: Final observation of Wikipedia page")
    print("       👀 Analyzing complex page structure...")
    print("-" * 70)

    result = await client.call_tool(
        "browser_observe",
        sessionId=session_id,
        pageId=page_id,
        maxElements=30,
    )

    if result.success:
        print("✓ Observation successful")
        obs_result = json.loads(result.result)

        print(f"   Title: {obs_result.get('title')}")
        print(f"   Summary: {obs_result.get('summary')}")

        elements = obs_result.get('elements', [])
        interactive = [e for e in elements if e.get('role') in ['link', 'button', 'textbox', 'combobox']]
        print(f"\n   Interactive elements found: {len(interactive)}")

    # Keep browser open for viewing
    print("\n" + "=" * 70)
    print("🎬 Step 7: Keeping browser open for 10 seconds")
    print("       👀 Take a good look at the browser window!")
    print("       📸 You can take screenshots manually if you want")
    print("-" * 70)

    print("\n   ⏳ Browser will remain open for 10 more seconds...")
    print("   ⏳ Watch the Chromium window on your screen!")

    for i in range(10, 0, -1):
        print(f"   ⏳ {i} seconds remaining...", end='\r')
        await asyncio.sleep(1)

    print("\n   ✓ Time's up!")

    # Close the session
    print("\n" + "=" * 70)
    print("🧹 Step 8: Closing browser session")
    print("       👀 Watch the browser window close!")
    print("-" * 70)

    result = await client.call_tool("browser_close", sessionId=session_id)

    if result.success:
        print("✓ Browser session closed")
        print("\n   👀 The Chromium window should have closed!")

    # Cleanup
    print("\n" + "=" * 70)
    print("🧹 Cleaning up MCP connection...")
    await client.cleanup()
    print("✓ Disconnected from AgentBrowser MCP server")

    print("\n" + "=" * 70)
    print("✨ Demo Complete!")
    print("\n🎉 You witnessed:")
    print("  ✓ Browser automation with VISIBLE browser window")
    print("  ✓ MCP integration working seamlessly")
    print("  ✓ Schema translator providing full parameter information")
    print("  ✓ Complex page navigation (Wikipedia)")
    print("  ✓ Semantic observations and screenshots")
    print("\n🔑 Key Point:")
    print("  The same tools that work with headless=True also work")
    print("  with headless=False — just set headless=False when creating")
    print("  the session, and watch the automation happen in real-time!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
