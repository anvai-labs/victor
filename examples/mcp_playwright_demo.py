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

"""Demo: Using Playwright MCP server with Victor for browser automation.

This example shows how to connect Victor to the Playwright MCP server
for browser automation, screenshots, and web interaction.

Playwright MCP Server provides:
- Browser automation (navigate, click, type, etc.)
- Screenshot capture
- Console log monitoring
- Network request inspection
- PDF generation
- Web scraping capabilities

Prerequisites:
-------------
1. Node.js installed (v18+)

2. Option A - Use npx (recommended, no installation):
   npx @anthropic/mcp-server-playwright

3. Option B - Docker:
   docker pull anthropic/mcp-server-playwright:latest
   docker run -i --rm anthropic/mcp-server-playwright:latest

4. Option C - Global installation:
   npm install -g @anthropic/mcp-server-playwright

Usage:
------
    # Run the demo
    python examples/mcp_playwright_demo.py

    # Run with specific transport
    python examples/mcp_playwright_demo.py --transport npx
    python examples/mcp_playwright_demo.py --transport docker

    # Run specific demo
    python examples/mcp_playwright_demo.py --demo screenshot
    python examples/mcp_playwright_demo.py --demo navigate
    python examples/mcp_playwright_demo.py --demo scrape

NPX Commands (for manual testing):
---------------------------------
    # Start Playwright MCP server via npx
    npx @anthropic/mcp-server-playwright

Docker Commands (for manual testing):
------------------------------------
    # Run Playwright MCP server in Docker
    docker run -i --rm \\
        -v /tmp/screenshots:/screenshots \\
        anthropic/mcp-server-playwright:latest

References:
-----------
    - Playwright MCP: https://github.com/anthropics/mcp-server-playwright
    - Playwright Docs: https://playwright.dev/
    - MCP Protocol: https://modelcontextprotocol.io
"""

import argparse
import asyncio
import base64
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_npx_available() -> bool:
    """Check if npx is available."""
    return shutil.which("npx") is not None


def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    if not shutil.which("docker"):
        return False
    try:
        import subprocess

        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def get_playwright_command(transport: str = "npx") -> List[str]:
    """Get the command to start Playwright MCP server.

    Args:
        transport: Transport method ('npx', 'docker', or 'global')

    Returns:
        Command list to start the server
    """
    if transport == "npx":
        return ["npx", "-y", "@anthropic/mcp-server-playwright"]
    elif transport == "docker":
        return [
            "docker",
            "run",
            "-i",
            "--rm",
            "-v",
            "/tmp/playwright-screenshots:/screenshots",
            "mcp/playwright:latest",
        ]
    elif transport == "global":
        return ["mcp-server-playwright"]
    else:
        raise ValueError(f"Unknown transport: {transport}")


async def demo_playwright_connection(transport_override: str = None):
    """Demo basic connection to Playwright MCP server."""
    print("=" * 70)
    print("Playwright MCP Server Demo - Basic Connection")
    print("=" * 70)

    # Check prerequisites
    has_npx = check_npx_available()
    has_docker = check_docker_available()

    if not has_npx and not has_docker:
        print("\nError: Neither npx nor Docker is available.")
        print("\nInstallation options:")
        print("  1. Install Node.js: https://nodejs.org/")
        print("  2. Install Docker: https://docs.docker.com/get-docker/")
        return

    # Use override if provided, otherwise prefer npx over Docker
    transport = transport_override if transport_override else ("npx" if has_npx else "docker")
    print(f"\nUsing transport: {transport}")

    try:
        from victor.integrations.mcp import MCPClient
    except ImportError:
        print("\nError: Victor MCP integration not available.")
        print("Please install Victor: pip install -e '.[dev]'")
        return

    command = get_playwright_command(transport)
    print(f"Command: {' '.join(command)}")

    # Create MCP client
    client = MCPClient(
        name="Victor-Playwright-Demo",
        version="1.0.0",
        health_check_interval=0,  # Disable health checks for demo
    )

    try:
        print("\n1. Connecting to Playwright MCP server...")
        print("-" * 70)
        success = await client.connect(command)

        if not success:
            print("   Failed to connect to Playwright MCP server.")
            print("\n   Troubleshooting:")
            if transport == "npx":
                print("   1. Ensure Node.js is installed: node --version")
                print("   2. Try running directly: npx @anthropic/mcp-server-playwright")
            else:
                print("   1. Ensure Docker is running: docker info")
                print("   2. Pull the image: docker pull anthropic/mcp-server-playwright:latest")
            return

        print("   Connected successfully!")

        # Show server info
        if client.server_info:
            print(f"\n   Server: {client.server_info.name} v{client.server_info.version}")

        # List available tools
        print("\n2. Discovering available tools...")
        print("-" * 70)
        tools = await client.refresh_tools()
        print(f"   Found {len(tools)} tools:")
        for tool in tools:
            print(f"\n   - {tool.name}")
            desc = tool.description[:80] + "..." if len(tool.description) > 80 else tool.description
            print(f"     {desc}")
            if tool.parameters:
                print(f"     Parameters: {len(tool.parameters)}")

        # List resources
        print("\n3. Discovering available resources...")
        print("-" * 70)
        resources = await client.refresh_resources()
        if resources:
            print(f"   Found {len(resources)} resources:")
            for resource in resources[:5]:
                print(f"   - {resource.name}: {resource.uri}")
        else:
            print("   No resources exposed (normal for Playwright).")

        # Test ping
        print("\n4. Testing connection...")
        print("-" * 70)
        ping_ok = await client.ping()
        print(f"   Ping: {'Success' if ping_ok else 'Failed'}")

        # Show client status
        print("\n5. Client status...")
        print("-" * 70)
        status = client.get_status()
        print(f"   Connected: {status['connected']}")
        print(f"   Tools: {status['tools_count']}")
        print(f"   Resources: {status['resources_count']}")

    except Exception as e:
        print(f"\nError during connection demo: {e}")
        logger.exception("Demo error")
    finally:
        print("\n6. Cleaning up...")
        print("-" * 70)
        client.disconnect()
        print("   Disconnected from server.")


async def demo_playwright_screenshot(transport_override: str = None):
    """Demo taking screenshots with Playwright MCP server."""
    print("\n" + "=" * 70)
    print("Playwright MCP Server Demo - Screenshot Capture")
    print("=" * 70)

    if not check_npx_available() and not check_docker_available():
        print("\nError: Neither npx nor Docker is available.")
        return

    try:
        from victor.integrations.mcp import MCPClient
    except ImportError:
        print("\nError: Victor MCP integration not available.")
        return

    transport = (
        transport_override if transport_override else ("npx" if check_npx_available() else "docker")
    )
    command = get_playwright_command(transport)

    client = MCPClient(
        name="Victor-Playwright-Screenshot",
        version="1.0.0",
        health_check_interval=0,
    )

    try:
        print("\n1. Connecting to Playwright MCP server...")
        print("-" * 70)
        success = await client.connect(command)
        if not success:
            print("   Failed to connect.")
            return

        print("   Connected!")

        # Get tools
        tools = await client.refresh_tools()
        tool_names = [t.name for t in tools]

        # Navigate to a page
        print("\n2. Navigating to example.com...")
        print("-" * 70)

        navigate_tool = next(
            (
                t
                for t in ["browser_navigate", "navigate", "playwright_navigate", "goto"]
                if t in tool_names
            ),
            None,
        )
        if navigate_tool:
            result = await client.call_tool(navigate_tool, url="https://example.com")
            if result.success:
                print("   Navigation successful!")
            else:
                print(f"   Navigation failed: {result.error}")
                return
        else:
            print(f"   No navigate tool found. Available: {tool_names}")
            return

        # Take screenshot
        print("\n3. Taking screenshot...")
        print("-" * 70)

        screenshot_tool = next(
            (
                t
                for t in [
                    "browser_take_screenshot",
                    "screenshot",
                    "playwright_screenshot",
                    "take_screenshot",
                ]
                if t in tool_names
            ),
            None,
        )
        if screenshot_tool:
            result = await client.call_tool(screenshot_tool)
            if result.success:
                print("   Screenshot captured!")

                # Check if result contains base64 image data
                if result.result and isinstance(result.result, str):
                    if result.result.startswith("data:image") or len(result.result) > 1000:
                        # Save screenshot
                        screenshot_dir = Path("/tmp/playwright-demo")
                        screenshot_dir.mkdir(exist_ok=True)
                        screenshot_path = screenshot_dir / "screenshot.png"

                        # Handle base64 data
                        try:
                            if "base64," in result.result:
                                img_data = base64.b64decode(result.result.split("base64,")[1])
                            else:
                                img_data = base64.b64decode(result.result)
                            screenshot_path.write_bytes(img_data)
                            print(f"   Saved to: {screenshot_path}")
                        except Exception as e:
                            print(f"   Could not save screenshot: {e}")
                    else:
                        print(f"   Result: {result.result[:200]}...")
            else:
                print(f"   Screenshot failed: {result.error}")
        else:
            print(f"   No screenshot tool found. Available: {tool_names}")

        # Get page title
        print("\n4. Getting page info...")
        print("-" * 70)

        evaluate_tool = next(
            (
                t
                for t in [
                    "browser_evaluate",
                    "evaluate",
                    "playwright_evaluate",
                    "execute_script",
                ]
                if t in tool_names
            ),
            None,
        )
        if evaluate_tool:
            # browser_evaluate expects 'function' parameter with arrow function syntax
            result = await client.call_tool(evaluate_tool, function="() => document.title")
            if result.success:
                print(f"   Page title: {result.result}")
            else:
                print(f"   Evaluate failed: {result.error}")

    except Exception as e:
        print(f"\nError during screenshot demo: {e}")
        logger.exception("Demo error")
    finally:
        client.disconnect()
        print("\n   Disconnected.")


async def demo_playwright_navigation(transport_override: str = None):
    """Demo web navigation with Playwright MCP server."""
    print("\n" + "=" * 70)
    print("Playwright MCP Server Demo - Web Navigation")
    print("=" * 70)

    if not check_npx_available() and not check_docker_available():
        print("\nError: Neither npx nor Docker is available.")
        return

    try:
        from victor.integrations.mcp import MCPClient
    except ImportError:
        print("\nError: Victor MCP integration not available.")
        return

    transport = (
        transport_override if transport_override else ("npx" if check_npx_available() else "docker")
    )
    command = get_playwright_command(transport)

    client = MCPClient(
        name="Victor-Playwright-Navigation",
        version="1.0.0",
        health_check_interval=0,
    )

    try:
        print("\n1. Connecting to Playwright MCP server...")
        print("-" * 70)
        success = await client.connect(command)
        if not success:
            print("   Failed to connect.")
            return

        tools = await client.refresh_tools()
        tool_names = [t.name for t in tools]
        print(f"   Connected! Available tools: {len(tools)}")

        # Navigation workflow
        print("\n2. Navigation workflow...")
        print("-" * 70)

        # Navigate to page
        navigate_tool = next(
            (
                t
                for t in ["browser_navigate", "navigate", "playwright_navigate", "goto"]
                if t in tool_names
            ),
            None,
        )
        if navigate_tool:
            print("   Navigating to https://httpbin.org/html...")
            result = await client.call_tool(navigate_tool, url="https://httpbin.org/html")
            if result.success:
                print("   Navigation successful!")
            else:
                print(f"   Failed: {result.error}")
                return

        # Get page content
        print("\n3. Getting page content...")
        print("-" * 70)

        content_tool = next(
            (
                t
                for t in [
                    "browser_evaluate",
                    "get_content",
                    "playwright_content",
                    "page_content",
                    "evaluate",
                ]
                if t in tool_names
            ),
            None,
        )
        if content_tool:
            if content_tool in ("evaluate", "browser_evaluate"):
                # browser_evaluate needs arrow function syntax
                result = await client.call_tool(
                    content_tool, function="() => document.body.innerText"
                )
            else:
                result = await client.call_tool(content_tool)

            if result.success:
                content = str(result.result)[:500]
                print(f"   Content preview:\n   {content}...")
            else:
                print(f"   Failed: {result.error}")

        # Check for click/interact tools
        print("\n4. Checking interaction tools...")
        print("-" * 70)

        interaction_tools = ["click", "type", "fill", "select", "hover"]
        available_interactions = [
            t for t in tool_names if any(i in t.lower() for i in interaction_tools)
        ]

        if available_interactions:
            print(f"   Interaction tools available:")
            for tool in available_interactions:
                print(f"   - {tool}")
        else:
            print("   No interaction tools found in this server version.")

    except Exception as e:
        print(f"\nError during navigation demo: {e}")
        logger.exception("Demo error")
    finally:
        client.disconnect()
        print("\n   Disconnected.")


async def demo_playwright_scraping(transport_override: str = None):
    """Demo web scraping with Playwright MCP server."""
    print("\n" + "=" * 70)
    print("Playwright MCP Server Demo - Web Scraping")
    print("=" * 70)

    if not check_npx_available() and not check_docker_available():
        print("\nError: Neither npx nor Docker is available.")
        return

    try:
        from victor.integrations.mcp import MCPClient
    except ImportError:
        print("\nError: Victor MCP integration not available.")
        return

    transport = (
        transport_override if transport_override else ("npx" if check_npx_available() else "docker")
    )
    command = get_playwright_command(transport)

    client = MCPClient(
        name="Victor-Playwright-Scraper",
        version="1.0.0",
        health_check_interval=0,
    )

    try:
        print("\n1. Connecting to Playwright MCP server...")
        print("-" * 70)
        success = await client.connect(command)
        if not success:
            print("   Failed to connect.")
            return

        tools = await client.refresh_tools()
        tool_names = [t.name for t in tools]
        print(f"   Connected! Tools: {len(tools)}")

        # Navigate to a page with data
        print("\n2. Navigating to target page...")
        print("-" * 70)

        navigate_tool = next(
            (
                t
                for t in ["browser_navigate", "navigate", "playwright_navigate", "goto"]
                if t in tool_names
            ),
            None,
        )
        if navigate_tool:
            # Use httpbin.org/json for structured data
            result = await client.call_tool(navigate_tool, url="https://httpbin.org/json")
            if result.success:
                print("   Navigated to httpbin.org/json")
            else:
                print(f"   Navigation failed: {result.error}")
                return

        # Extract data using evaluate
        print("\n3. Extracting JSON data...")
        print("-" * 70)

        evaluate_tool = next(
            (
                t
                for t in [
                    "browser_evaluate",
                    "evaluate",
                    "playwright_evaluate",
                    "execute_script",
                ]
                if t in tool_names
            ),
            None,
        )
        if evaluate_tool:
            result = await client.call_tool(evaluate_tool, function="() => document.body.innerText")
            if result.success:
                print("   Extracted data:")
                print(f"   {result.result[:300]}...")
            else:
                print(f"   Extraction failed: {result.error}")

        # Demo: Extract specific elements
        print("\n4. Demo: Scraping Hacker News...")
        print("-" * 70)

        if navigate_tool and evaluate_tool:
            # Navigate to HN
            result = await client.call_tool(navigate_tool, url="https://news.ycombinator.com")
            if result.success:
                print("   Navigated to Hacker News")

                # Extract titles
                result = await client.call_tool(
                    evaluate_tool,
                    function="""() => Array.from(document.querySelectorAll('.titleline > a')).slice(0, 5).map(a => a.innerText).join('\\n')""",
                )
                if result.success:
                    print("\n   Top 5 stories:")
                    for i, title in enumerate(str(result.result).split("\n")[:5], 1):
                        print(f"   {i}. {title}")
                else:
                    print(f"   Extraction failed: {result.error}")

    except Exception as e:
        print(f"\nError during scraping demo: {e}")
        logger.exception("Demo error")
    finally:
        client.disconnect()
        print("\n   Disconnected.")


def show_integration_examples():
    """Show integration examples for Playwright MCP server."""
    print("\n" + "=" * 70)
    print("Playwright MCP Integration Examples")
    print("=" * 70)

    print("""
1. Claude Desktop Configuration
-------------------------------
Add to ~/Library/Application Support/Claude/claude_desktop_config.json:

{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-playwright"]
    }
  }
}

Or with Docker:
{
  "mcpServers": {
    "playwright": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/tmp/screenshots:/screenshots",
        "anthropic/mcp-server-playwright:latest"
      ]
    }
  }
}

2. Victor MCP Configuration
---------------------------
Add to ~/.victor/mcp.yaml:

servers:
  - name: playwright
    command:
      - npx
      - -y
      - "@anthropic/mcp-server-playwright"
    description: Playwright browser automation MCP server
    tags:
      - browser
      - automation
      - screenshot
    auto_connect: true
    health_check_interval: 60

3. Programmatic Usage
---------------------
from victor.integrations.mcp import MCPClient

async def scrape_website(url: str) -> str:
    client = MCPClient(name="Scraper")

    try:
        await client.connect(["npx", "-y", "@anthropic/mcp-server-playwright"])

        # Navigate
        await client.call_tool("navigate", url=url)

        # Take screenshot
        screenshot = await client.call_tool("screenshot")

        # Extract content
        content = await client.call_tool(
            "evaluate",
            script="document.body.innerText"
        )

        return content.result if content.success else ""

    finally:
        client.disconnect()

# Usage
content = await scrape_website("https://example.com")

4. With MCPRegistry for Multiple Servers
----------------------------------------
from victor.integrations.mcp import MCPRegistry, MCPServerConfig

registry = MCPRegistry()

registry.register_server(MCPServerConfig(
    name="playwright",
    command=["npx", "-y", "@anthropic/mcp-server-playwright"],
    tags=["browser", "automation"],
))

async with registry:
    await registry.connect("playwright")

    # Navigate
    await registry.call_tool("navigate", url="https://example.com")

    # Screenshot
    result = await registry.call_tool("screenshot")
    print(f"Screenshot: {result.success}")

5. Error Handling Example
-------------------------
from victor.integrations.mcp import MCPClient

client = MCPClient(
    name="SafeScraper",
    health_check_interval=30,
    auto_reconnect=True,
    max_reconnect_attempts=3,
)

# Register callbacks
client.on_connect(lambda: print("Connected!"))
client.on_disconnect(lambda reason: print(f"Disconnected: {reason}"))
client.on_health_change(lambda healthy: print(f"Healthy: {healthy}"))

try:
    await client.connect(["npx", "-y", "@anthropic/mcp-server-playwright"])

    result = await client.call_tool("navigate", url="https://example.com")
    if not result.success:
        print(f"Navigation failed: {result.error}")

    result = await client.call_tool("screenshot")
    if not result.success:
        print(f"Screenshot failed: {result.error}")

finally:
    client.disconnect()
""")


async def main():
    """Run Playwright MCP demo."""
    parser = argparse.ArgumentParser(
        description="Playwright MCP Server Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/mcp_playwright_demo.py                  # Run connection demo
  python examples/mcp_playwright_demo.py --demo all       # Run all demos
  python examples/mcp_playwright_demo.py --demo screenshot
  python examples/mcp_playwright_demo.py --demo navigate
  python examples/mcp_playwright_demo.py --demo scrape
  python examples/mcp_playwright_demo.py --examples       # Show integration code
        """,
    )
    parser.add_argument(
        "--transport",
        choices=["npx", "docker", "global"],
        default=None,
        help="Transport method for MCP server",
    )
    parser.add_argument(
        "--demo",
        choices=["connection", "screenshot", "navigate", "scrape", "all"],
        default="connection",
        help="Which demo to run",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Show integration examples and exit",
    )

    args = parser.parse_args()

    print("Playwright MCP Server Demo")
    print("=" * 70)
    print("\nThis demo shows how to use the Playwright MCP server with Victor")
    print("for browser automation, screenshots, and web scraping.")

    if args.examples:
        show_integration_examples()
        return

    if args.demo in ("connection", "all"):
        await demo_playwright_connection(args.transport)

    if args.demo in ("screenshot", "all"):
        await demo_playwright_screenshot(args.transport)

    if args.demo in ("navigate", "all"):
        await demo_playwright_navigation(args.transport)

    if args.demo in ("scrape", "all"):
        await demo_playwright_scraping(args.transport)

    if args.demo == "connection":
        show_integration_examples()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Install: npm install -g @anthropic/mcp-server-playwright")
    print("  2. Or use npx: npx @anthropic/mcp-server-playwright")
    print("  3. Add to Victor config (~/.victor/mcp.yaml)")
    print("  4. Use browser automation in your workflows")


if __name__ == "__main__":
    asyncio.run(main())
