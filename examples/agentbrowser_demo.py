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

Drives the full observe -> act loop through the six real AgentBrowser MCP
tools, using Victor's MCPClient (the schema translator in
victor/integrations/mcp/client.py converts each tool's standard MCP
``inputSchema`` into Victor parameter metadata, so tools arrive with their
parameters instead of empty lists).

AgentBrowser's tool surface (6 tools):
    browser_create, browser_close, browser_navigate,
    browser_observe, browser_act, browser_screenshot

Actions address elements by the ref from browser_observe
(``e<revision>_<ordinal>``, e.g. ``e2_0``) - never CSS selectors. If the page
mutated since the observation, browser_act fails with STALE_TARGET and the
fix is to observe again, not retry.

Prerequisites:
    1. AgentBrowser REST server running on port 3000:
       cd /path/to/agentbrowser && node packages/api/dist/bin.js

    2. The agentbrowser-mcp release binary on PATH (see agentbrowser_command()
       below for the one-line install). Falls back to a local checkout's
       `pnpm --filter @agentbrowser/mcp-server build` if the binary is absent.

Usage:
    cd /path/to/codingagent
    python examples/agentbrowser_demo.py
"""

import asyncio
import json
import shutil

from victor.integrations.mcp import MCPClient

# Dev fallback: a local AgentBrowser checkout's node build. The released
# single binary (v1.1.0+, TD-BROWSER-5) is preferred whenever it is on PATH.
AGENTBROWSER_FALLBACK = [
    "node",
    "/Users/vijaysingh/code/agentbrowser/packages/mcp-server/dist/bin.js",
]


def agentbrowser_command() -> list[str]:
    """Command that spawns the AgentBrowser MCP server.

    Prefers the released self-contained binary (no Node, no checkout):
        gh release download v1.1.0 --repo vjsingh1984/agentbrowser \
            --pattern 'agentbrowser-mcp-<target>'
        install -m 755 agentbrowser-mcp-<target> /opt/homebrew/bin/agentbrowser-mcp
    """
    if shutil.which("agentbrowser-mcp"):
        return ["agentbrowser-mcp"]
    return AGENTBROWSER_FALLBACK


async def call(client: MCPClient, name: str, **kwargs) -> dict:
    """Call a tool and return its parsed result, exiting the demo on failure."""
    result = await client.call_tool(name, **kwargs)
    if not result.success:
        print(f"✗ {name} failed: {result.error}")
        await client.cleanup()
        raise SystemExit(1)
    return json.loads(result.result)


async def main():
    print("🌐 AgentBrowser + Victor Integration Demo")
    print("=" * 70)

    client = MCPClient(name="agentbrowser-demo", version="1.0.0", health_check_interval=0)
    command = agentbrowser_command()
    print(
        f"spawn: {' '.join(command)}"
        + (
            ""
            if command[0] == "agentbrowser-mcp"
            else "  (binary not on PATH; using node fallback)"
        )
    )
    if not await client.connect(command):
        print("✗ Could not connect to the AgentBrowser MCP server.")
        print("  Is the REST server up?  node packages/api/dist/bin.js")
        return
    print("✓ Connected to AgentBrowser MCP server\n")

    # The schema translator at work: every tool arrives with real parameters.
    print(f"🔧 Tools ({len(client.tools)}):")
    for tool in client.tools:
        params = ", ".join(p.name for p in tool.parameters) or "NONE"
        print(f"   • {tool.name}({params})")

    # 1) Create a session (headless=False to watch it, if you like).
    print("\n1️⃣  browser_create")
    session = await call(client, "browser_create", tenantId="demo-tenant", ttlMs=300000)
    session_id, page_id = session["sessionId"], session["pageId"]
    print(f"    session={session_id} page={page_id}")

    try:
        # 2) Navigate.
        print("\n2️⃣  browser_navigate -> https://example.com")
        nav = await call(
            client,
            "browser_navigate",
            sessionId=session_id,
            pageId=page_id,
            url="https://example.com",
            waitUntil="load",
        )
        print(f"    landed on: {nav.get('url')}")

        # 3) Observe: mint refs for the interactive elements.
        print("\n3️⃣  browser_observe")
        obs = await call(
            client,
            "browser_observe",
            sessionId=session_id,
            pageId=page_id,
            maxElements=10,
        )
        elements = obs.get("elements", [])
        print(f"    title: {obs.get('title')}")
        for element in elements[:5]:
            print(
                f"    [{element['ref']}] {element['role']}"
                + (f" \"{element['name']}\"" if element.get("name") else "")
            )
        if not elements:
            print("    (no interactive elements observed)")

        # 4) Act on a ref from the observation.
        link = next((e for e in elements if e["role"] == "link"), None)
        if link is not None:
            print(f"\n4️⃣  browser_act click [{link['ref']}]")
            await call(
                client,
                "browser_act",
                sessionId=session_id,
                pageId=page_id,
                action="click",
                target={"ref": link["ref"]},
            )
            print(f"    clicked; now at: example.com -> iana.org (follows the link)")

            # The click bumped the revision: old refs are stale, observe again.
            print("\n    (revision bumped - observing fresh refs)")
            obs = await call(
                client,
                "browser_observe",
                sessionId=session_id,
                pageId=page_id,
                maxElements=5,
            )
            print(f"    fresh title: {obs.get('title')}")
        else:
            print("\n4️⃣  (skipped - no link element observed)")

        # 5) Screenshot as evidence.
        print("\n5️⃣  browser_screenshot")
        shot = await call(
            client,
            "browser_screenshot",
            sessionId=session_id,
            pageId=page_id,
            fullPage=False,
        )
        print(f"    artifact={shot.get('artifactId')} ({shot.get('sizeBytes')} bytes)")
    finally:
        # 6) Close.
        print("\n6️⃣  browser_close")
        await call(client, "browser_close", sessionId=session_id)
        await client.cleanup()

    print("\n✨ Demo complete")


if __name__ == "__main__":
    asyncio.run(main())
