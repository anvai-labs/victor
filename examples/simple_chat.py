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

"""Simple example of using Victor with Ollama."""

import asyncio

from victor import Agent, EventType


async def main():
    """Run a simple chat example."""
    agent = await Agent.create(
        provider="ollama",
        model="qwen2.5-coder:7b",  # Change to your preferred model
        temperature=0.7,
    )

    # Example 1: Simple chat
    print("Example 1: Simple Chat")
    print("-" * 50)
    response = await agent.run("What is Python?")
    print(response.content)
    print()

    # Example 2: Follow-up question
    print("Example 2: Follow-up Question")
    print("-" * 50)
    response = await agent.run("Can you give me a simple Python code example?")
    print(response.content)
    print()

    # Example 3: Streaming response
    print("Example 3: Streaming Response")
    print("-" * 50)
    print("Streaming: ", end="", flush=True)
    async for event in agent.stream("Count from 1 to 5"):
        if event.type == EventType.CONTENT:
            print(event.content, end="", flush=True)
    print("\n")

    # Example 4: Using tools
    print("Example 4: Using Tools (File Operations)")
    print("-" * 50)
    response = await agent.run(
        "Create a simple hello.txt file with the content 'Hello from CodingAgent!'"
    )
    print(response.content)
    print()

    # Clean up
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
