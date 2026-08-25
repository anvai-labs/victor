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

"""Demo of AgentBrowser integration through Victor's MCP layer.

This demonstrates:
1. Headless browser automation through semantic observations
2. Stable element references (e<revision>_<ordinal>)
3. Approval gates for high-risk actions
4. Token-efficient bounded observations

Prerequisites:
    1. AgentBrowser REST server running on port 3000:
       cd /Users/vijaysingh/code/agentbrowser
       pnpm --filter @agentbrowser/api start

    2. Victor configured with AgentBrowser MCP server:
       # ~/.victor/mcp.yaml should contain:
       mcpServers:
         agentbrowser:
           command: node
           args:
             - /Users/vijaysingh/code/agentbrowser/packages/mcp-server/dist/bin.js
           env:
             AGENTBROWSER_BASE_URL: http://localhost:3000

Usage:
    cd /Users/vijaysingh/code/codingagent
    python examples/agentbrowser_demo.py
"""

import asyncio
from victor.framework import Agent, ToolSet


async def main():
    print("🌐 AgentBrowser + Victor Integration Demo")
    print("=" * 70)
    print("\nPrerequisites:")
    print("  1. AgentBrowser REST server running on http://localhost:3000")
    print("  2. ~/.victor/mcp.yaml configured with AgentBrowser MCP server")
    print("\nStarting agent with AgentBrowser tools...\n")

    # Create agent with AgentBrowser tools (auto-registered via MCP)
    agent = await Agent.create(
        provider="anthropic",  # or any configured provider
        tools=ToolSet.from_mcp_servers(["agentbrowser"]),
    )

    # Task 1: Navigate and observe
    print("1️⃣  Navigate and Observe")
    print("-" * 70)
    print("Navigating to https://example.com...\n")

    result = await agent.run(
        "Navigate to https://example.com and tell me what elements you see. "
        "Use the browser_navigate tool followed by browser_observe. "
        "Report the interactive elements you can identify."
    )

    print(result.content)

    # Task 2: Click an interactive element
    print("\n2️⃣  Click Element")
    print("-" * 70)
    print("Clicking the first link on the page...\n")

    result = await agent.run(
        "Click the first link on the page. "
        "Use the browser_click tool with the selector from the observation "
        "(selectors are in the format e<revision>_<ordinal>, e.g., e1_0)."
    )

    print(result.content)

    # Task 3: Take a screenshot
    print("\n3️⃣  Screenshot")
    print("-" * 70)
    print("Taking a full-page screenshot...\n")

    result = await agent.run(
        "Take a full-page screenshot of the current page. "
        "Use the browser_screenshot tool with full_page=true. "
        "Report where the screenshot was saved."
    )

    print(result.content)

    # Task 4: Extract text content
    print("\n4️⃣  Extract Text")
    print("-" * 70)
    print("Extracting visible text from the page...\n")

    result = await agent.run(
        "Extract all visible text content from the current page. "
        "Use the browser_extract_text tool. "
        "Summarize the key information in 2-3 sentences."
    )

    print(result.content)

    print("\n✨ Demo Complete!")
    print("\n" + "=" * 70)
    print("AgentBrowser Features Exposed:")
    print("  ✓ Semantic observations over screenshots")
    print("  ✓ Stable element references with staleness detection")
    print("  ✓ Safety-first (network policy + approval gates)")
    print("  ✓ Token-efficient bounded observations")
    print("  ✓ Approval gates for high-risk actions")
    print("\nAvailable Tools:")
    print("  • mcp_browser_navigate(url, wait_for)")
    print("  • mcp_browser_observe(selector, max_elements)")
    print("  • mcp_browser_click(selector)")
    print("  • mcp_browser_type(selector, text, clear_first)")
    print("  • mcp_browser_scroll(direction, amount)")
    print("  • mcp_browser_screenshot(selector, full_page, name)")
    print("  • mcp_browser_extract_text(selector)")
    print("  • mcp_browser_wait_for(selector, state, timeout)")
    print("  • mcp_browser_close(session_id)")
    print("\nFor more information, see:")
    print("  • AgentBrowser: https://github.com/vjsingh1984/agentbrowser")
    print("  • Victor: https://github.com/anvai-labs/victor")


if __name__ == "__main__":
    asyncio.run(main())
