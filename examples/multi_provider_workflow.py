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

"""Example workflow using multiple providers strategically."""

import asyncio
import os

from victor import Agent


async def main():
    """Demonstrate strategic use of multiple providers."""
    print("🎯 Multi-Provider Workflow Example\n")
    print("=" * 60)
    print("Strategy: Use the right model for each task\n")

    # Task: Build a complete feature with tests

    # Step 1: Brainstorm with local Ollama (FREE!)
    print("\n📝 Step 1: Brainstorming (Using Ollama - FREE)")
    print("-" * 60)

    try:
        agent_ollama = await Agent.create(
            provider="ollama",
            model="qwen2.5-coder:7b",
            temperature=0.8,
        )
        ollama_available = True

        response = await agent_ollama.run(
            "Brainstorm 3 creative names for a Python function that validates email addresses. "
            "Just list the names, one per line."
        )
        print(f"Ollama: {response.content}")
        chosen_name = "validate_email_format"  # Pick one
        print(f"\n✅ Chose: {chosen_name}")
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
        ollama_available = False
        chosen_name = "validate_email_format"

    # Step 2: Implement with GPT-4o mini (CHEAP & FAST)
    print("\n\n💻 Step 2: Implementation (Using GPT-4o mini - Fast & Cheap)")
    print("-" * 60)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        agent_gpt4o_mini = await Agent.create(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.5,
        )

        response = await agent_gpt4o_mini.run(
            f"Write a Python function called {chosen_name} that validates email addresses. "
            "Use regex, include docstring, handle edge cases."
        )
        print(f"GPT-4o mini: {response.content[:500]}...")
        implementation = response.content
        await agent_gpt4o_mini.close()
    else:
        print("⚠️  OPENAI_API_KEY not set. Skipping GPT-4o mini step.")
        implementation = "# Implementation would go here"

    # Step 3: Review with Claude (BEST QUALITY)
    print("\n\n🔍 Step 3: Code Review (Using Claude - Best Quality)")
    print("-" * 60)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        agent_claude = await Agent.create(
            provider="anthropic",
            model="claude-sonnet-4-5",
            temperature=0.7,
        )

        response = await agent_claude.run(
            f"Review this code for security, edge cases, and best practices:\n\n{implementation}\n\n"
            "Provide specific suggestions for improvement."
        )
        print(f"Claude: {response.content}")
        await agent_claude.close()
    else:
        print("⚠️  ANTHROPIC_API_KEY not set. Skipping Claude review.")

    # Step 4: Generate tests with Ollama again (FREE!)
    print("\n\n🧪 Step 4: Generate Tests (Using Ollama - FREE)")
    print("-" * 60)

    if ollama_available:
        response = await agent_ollama.run(
            f"Write pytest unit tests for this function:\n\n{implementation}\n\n"
            "Include tests for: valid emails, invalid emails, edge cases."
        )
        print(f"Ollama: {response.content[:500]}...")
    else:
        print("⚠️  Ollama not available. Skipping test generation.")

    # Summary
    print("\n\n📊 Workflow Summary")
    print("-" * 60)
    print("✅ Step 1: Brainstorm with Ollama (FREE)")
    print("✅ Step 2: Implement with GPT-4o mini ($0.15 per 1M input tokens)")
    print("✅ Step 3: Review with Claude ($0.003 per 1K tokens)")
    print("✅ Step 4: Generate tests with Ollama (FREE)")
    print("\n💰 Total cost: ~$0.01 vs. using GPT-4 for everything: ~$0.10")
    print("📈 Cost savings: 90% while maintaining quality!")

    print("\n\n💡 Key Takeaways:")
    print("- Use Ollama for brainstorming and simple tasks (FREE)")
    print("- Use GPT-4o mini for quick implementations (CHEAP)")
    print("- Use Claude for critical reviews (QUALITY)")
    print("- Mix and match for optimal cost/quality balance")

    # Clean up
    if ollama_available:
        await agent_ollama.close()

    print("\n✅ Multi-provider workflow completed!")


if __name__ == "__main__":
    asyncio.run(main())
