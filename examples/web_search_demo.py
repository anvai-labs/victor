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

"""Demo of Web Search tool.

This demonstrates:
1. DuckDuckGo web search (no API key required)
2. Result parsing and formatting
3. Content extraction from URLs
4. Privacy-focused search

Usage:
    python examples/web_search_demo.py
"""

import asyncio
from victor.tools.web_search_tool import web_search, web_fetch


async def main():
    """Demo web search operations."""
    print("🎯 Web Search Tool Demo")
    print("=" * 70)
    print("\nUsing DuckDuckGo for privacy-focused search")
    print("No API keys required!\n")

    # Test 1: Basic web search
    print("\n1️⃣ Basic Web Search")
    print("-" * 70)
    print("Query: Python async programming best practices")

    result = await web_search(query="Python async programming best practices", max_results=3)

    if result.get("success"):
        print(result.get("results"))
    else:
        print(f"Error: {result.get('error')}")

    # Test 2: Technical search
    print("\n\n2️⃣ Technical Search")
    print("-" * 70)
    print("Query: Victor AI coding assistant features")

    result = await web_search(query="AI coding assistant features comparison", max_results=3)

    if result.get("success"):
        print(result.get("results"))
    else:
        print(f"Error: {result.get('error')}")

    # Test 3: Fetch specific URL
    print("\n\n3️⃣ Fetch Content from URL")
    print("-" * 70)
    print("URL: https://www.python.org")

    result = await web_fetch(url="https://www.python.org")

    if result.get("success"):
        # Show first 500 chars of content
        content = result.get("content", "")
        content = content[:500] + "..." if len(content) > 500 else content
        print(content)
    else:
        print(f"Error: {result.get('error')}")

    # Test 4: Search with more results
    print("\n\n4️⃣ Extended Search")
    print("-" * 70)
    print("Query: machine learning tutorials")

    result = await web_search(query="machine learning tutorials for beginners", max_results=5)

    if result.get("success"):
        print(result.get("results"))
    else:
        print(f"Error: {result.get('error')}")

    # Test 5: Region-specific search
    print("\n\n5️⃣ Region-Specific Search")
    print("-" * 70)
    print("Query: local AI meetups (US region)")

    result = await web_search(
        query="AI and machine learning meetups", max_results=3, region="us-en"
    )

    if result.get("success"):
        print(result.get("results"))
    else:
        print(f"Error: {result.get('error')}")

    print("\n\n✨ Demo Complete!")
    print("\nWeb Search Tool Features:")
    print("  ✓ DuckDuckGo search (privacy-focused)")
    print("  ✓ No API keys required")
    print("  ✓ Result parsing and formatting")
    print("  ✓ Content extraction from URLs")
    print("  ✓ Region-specific results")
    print("  ✓ Safe search filtering")
    print("  ✓ Configurable result limits")

    print("\n\n🤖 With AI Provider Available:")
    print("  • Automatic result summarization")
    print("  • Key point extraction")
    print("  • Conflicting information detection")
    print("  • Smart source citations")
    print("  • Query refinement suggestions")

    print("\n\n📚 Example with AI (requires provider):")
    print("""
# In agent conversation:
User: "What are the latest developments in AI coding assistants?"

Victor: Let me search for that information...
[Calls web_search with operation="summarize"]

Victor: Based on my web search, here are the latest developments:

## Key Findings:
1. AI coding assistants now support multiple LLM providers
   - Not just OpenAI anymore
   - Local models via Ollama gaining popularity
   - Privacy concerns driving local-first approaches

2. New features emerging:
   - Multi-file editing with transaction safety
   - Semantic code search
   - Context-aware suggestions
   - AI-generated commit messages

3. Market trends:
   - Open-source alternatives challenging proprietary tools
   - Focus on developer privacy
   - Integration with existing workflows

Sources:
- TechCrunch: "AI Coding Tools Market Update" (url)
- Developer Blog: "Local-First AI Development" (url)
- GitHub Blog: "The Future of Code Assistance" (url)

Would you like me to dig deeper into any specific area?
""")

    print("\n\n🔍 Search Tips:")
    print("  • Use specific queries for better results")
    print("  • Combine with fetch operation for detailed content")
    print("  • Use region parameter for localized results")
    print("  • Safe search helps filter content")
    print("  • DuckDuckGo is privacy-focused (no tracking)")


if __name__ == "__main__":
    asyncio.run(main())
