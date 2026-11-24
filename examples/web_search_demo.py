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

"""Demo of the Victor web research tools (FOU-A5).

This demonstrates:
1. ``web_search`` — keyless DuckDuckGo search, or a key-gated API provider
   (``provider="tavily" | "brave" | "serper"``; ``provider="auto"`` picks the
   first configured provider and falls back to DuckDuckGo when no key is set)
2. ``web_fetch`` — readable page read: title + main text, download-size cap,
   content-type guard (``render="browser"`` hands a JS-required page to the
   browser path)
3. ``browser_tool`` — headless-browser read for JS-required pages via the
   agentbrowser service (Playwright fallback with ``engine="playwright"``)

Usage:
    python examples/web_search_demo.py

Note: the DuckDuckGo calls below hit the live network. The API-provider demo
needs TAVILY_API_KEY (or BRAVE_API_KEY / SERPER_API_KEY) set via
``victor keys --set-service tavily``; without the key it prints the
structured "provider not configured" result instead of failing.
"""

import asyncio
import os

from victor.tools.web_search_tool import web_fetch, web_search


def _print_result(result: dict, content_key: str = "results") -> None:
    """Print a tool result, structured-error aware."""
    if not result.get("success"):
        print(f"Error: {result.get('error')}")
        return
    provider = result.get("provider")
    if provider:
        print(f"(provider: {provider})")
    content = result.get(content_key, "")
    content = content[:500] + "..." if len(content) > 500 else content
    print(content)


async def main():
    """Demo web search operations."""
    print("Victor Web Research Tools Demo")
    print("=" * 70)
    print("\nKeyless DuckDuckGo search by default; Tavily/Brave/Serper with a key.\n")

    # 1. Basic web search (keyless DuckDuckGo)
    print("\n1. Basic web search (DuckDuckGo, no key)")
    print("-" * 70)
    print("Query: Python async programming best practices")
    result = await web_search(query="Python async programming best practices", max_results=3)
    _print_result(result)

    # 2. Key-gated API provider (FOU-A5). Set TAVILY_API_KEY to see this path;
    #    without a key the tool returns a structured provider_configured=False
    #    result and fires no unauthenticated request.
    print("\n2. API-provider search (provider='tavily', key-gated)")
    print("-" * 70)
    if not os.environ.get("TAVILY_API_KEY"):
        print("TAVILY_API_KEY not set — demonstrating the structured fallback:")
    result = await web_search(
        query="AI coding assistant features comparison", max_results=3, provider="tavily"
    )
    if not result.get("provider_configured", True):
        print(f"Not configured (as designed): {result.get('error')}")
    else:
        _print_result(result)
        print(f"structured_results: {result.get('structured_results')}")

    # 3. Page read (FOU-A5 page-read upgrade: title + main text)
    print("\n3. Readable page fetch (title + main text)")
    print("-" * 70)
    print("URL: https://www.python.org")
    result = await web_fetch(url="https://www.python.org")
    if result.get("success"):
        print(f"Title: {result.get('title')}")
        _print_result(result, content_key="content")
    else:
        print(f"Error: {result.get('error')}")

    # 4. JS-required page: browser_tool via the agentbrowser service
    #    (needs `agentbrowser` running; engine="playwright" for the local
    #    fallback — see docs/guides/tool-reference.md)
    print("\n4. Browser read for JS-required pages (browser_tool)")
    print("-" * 70)
    try:
        from victor.tools.browser_tool import browser_tool

        result = await browser_tool(url="https://example.com", timeout_seconds=20.0)
        if result.get("success"):
            print(f"Title: {result.get('title')}")
            print(f"Final URL: {result.get('url')}")
            _print_result(result, content_key="content")
        else:
            print(f"Structured error (service down is expected here): {result.get('error')}")
    except ImportError:
        print("browser_tool unavailable in this install.")

    print("\n\nDemo complete. Shipped surface:")
    print("  web_search(query, provider='auto'|'duckduckgo'|'tavily'|'brave'|'serper')")
    print("  web_fetch(url, render='auto'|'http'|'browser')")
    print("  browser_tool(url, engine='agentbrowser'|'playwright')")
    print("\nTips:")
    print("  - provider='auto' upgrades automatically when a search key is set")
    print("  - use web_fetch first; browser_tool is for pages that require JS")
    print("  - ai_summarize=True adds an AI summary of the results")


if __name__ == "__main__":
    asyncio.run(main())
