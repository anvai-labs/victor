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

"""Example using xAI Grok provider."""

import asyncio
import os

from victor import Agent, EventType


async def main():
    """Run examples with Grok."""
    # Check for API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: XAI_API_KEY environment variable not set")
        print("Get your API key from: https://console.x.ai/")
        return

    print("🤖 xAI Grok Example\n")
    print("=" * 60)

    agent = await Agent.create(
        provider="xai",
        model="grok-beta",
        temperature=0.8,
    )

    # Example 1: General question
    print("\n📝 Example 1: General Question")
    print("-" * 60)
    response = await agent.run("What makes Grok unique compared to other AI models?")
    print(f"Grok: {response.content}")

    # Example 2: Technical explanation
    print("\n\n💻 Example 2: Technical Explanation")
    print("-" * 60)
    await agent.reset()

    response = await agent.run(
        "Explain the differences between REST and GraphQL APIs. " "Give a code example for each."
    )
    print(f"Grok: {response.content}")

    # Example 3: Streaming response
    print("\n\n🌊 Example 3: Streaming Response")
    print("-" * 60)
    await agent.reset()

    print("Grok: ", end="", flush=True)
    async for event in agent.stream(
        "Write a Python decorator that measures function execution time."
    ):
        if event.type == EventType.CONTENT:
            print(event.content, end="", flush=True)
    print()

    # Example 4: Creative + Technical
    print("\n\n🎨 Example 4: Creative + Technical")
    print("-" * 60)
    await agent.reset()

    response = await agent.run(
        "Design a whimsical API for a virtual pet simulator. "
        "Include 5 endpoints with fun names and descriptions."
    )
    print(f"Grok: {response.content}")

    # Example 5: Multi-turn debugging
    print("\n\n🐛 Example 5: Multi-turn Debugging")
    print("-" * 60)
    await agent.reset()

    response1 = await agent.run(
        "I'm getting a 'list index out of range' error in Python. What could cause this?"
    )
    print("User: I'm getting a 'list index out of range' error in Python. What could cause this?")
    print(f"Grok: {response1.content[:200]}...")

    response2 = await agent.run("The error happens when I do: result = my_list[len(my_list)]")
    print("\nUser: The error happens when I do: result = my_list[len(my_list)]")
    print(f"Grok: {response2.content}")

    # Clean up
    await agent.close()

    print("\n\n✅ Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
