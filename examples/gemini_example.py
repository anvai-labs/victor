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

"""Example using Google Gemini provider."""

import asyncio
import logging
import os

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Enable DEBUG for victor providers to see safety filter details
logging.getLogger("victor.providers").setLevel(logging.DEBUG)

from victor import Agent, EventType


async def main():
    """Run examples with Gemini."""
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return

    print("🤖 Google Gemini Example\n")
    print("=" * 60)

    # Create agent with Gemini Pro
    agent = await Agent.create(
        model="gemini-2.5-pro",  # Latest Gemini model
        provider="google",
        temperature=0.9,
    )

    # Example 1: Long context understanding
    print("\n📚 Example 1: Long Context (Gemini's Strength)")
    print("-" * 60)

    long_context = """
    Project Requirements:
    1. Build a task management system
    2. Users can create, edit, delete tasks
    3. Tasks have: title, description, priority, due date, tags
    4. Support subtasks (nested)
    5. Real-time collaboration (multiple users)
    6. Email notifications
    7. Mobile responsive
    8. Dark mode
    9. Export to CSV/JSON
    10. Search and filtering

    Tech Stack:
    - Frontend: React, TypeScript, TailwindCSS
    - Backend: FastAPI, PostgreSQL
    - Real-time: WebSockets
    - Auth: JWT
    """

    response = await agent.run(
        f"Here are requirements:\n\n{long_context}\n\n"
        "What are the main technical challenges and how would you solve them?"
    )
    print(f"Gemini: {response.content}")

    # Example 2: Code generation
    print("\n\n💻 Example 2: Code Generation")
    print("-" * 60)
    await agent.reset()

    response = await agent.run(
        "Write a Python class for a LRU Cache with O(1) get and put operations. "
        "Include type hints and docstrings."
    )
    print(f"Gemini: {response.content}")

    # Example 3: Streaming
    print("\n\n🌊 Example 3: Streaming Response")
    print("-" * 60)
    await agent.reset()

    print("Gemini: ", end="", flush=True)
    async for event in agent.stream(
        "Explain the CAP theorem in distributed systems with examples."
    ):
        if event.type == EventType.CONTENT:
            print(event.content, end="", flush=True)
    print()

    # Example 4: Analysis and comparison
    print("\n\n📊 Example 4: Analysis")
    print("-" * 60)
    await agent.reset()

    response = await agent.run(
        "Compare these Python web frameworks: Django, FastAPI, Flask. "
        "When would you use each? Create a comparison table."
    )
    print(f"Gemini: {response.content}")

    # Example 5: Creative coding
    print("\n\n🎨 Example 5: Creative Coding")
    print("-" * 60)
    await agent.reset()

    response = await agent.run(
        "Create an ASCII art generator function in Python. "
        "Make it fun and include emojis in the output!"
    )
    print(f"Gemini: {response.content}")

    # Clean up
    await agent.close()

    print("\n\n✅ Examples completed!")
    print("\n💡 Tip: Gemini excels at long context and multimodal tasks")


if __name__ == "__main__":
    asyncio.run(main())
