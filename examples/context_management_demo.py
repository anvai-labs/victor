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

"""Demo of context management with token budgeting.

This shows how Victor manages conversation context efficiently.

Usage:
    python examples/context_management_demo.py
"""

from victor.context import ProjectContextLoader, PruningStrategy


def main():
    """Demo context management."""
    print("🎯 Context Management Demo\n")
    print("=" * 70)

    # Create context manager
    print("\n📊 Creating context manager...")
    ctx = ProjectContextLoader(
        model="gpt-4",
        max_tokens=128000,
        reserved_tokens=4096,
        pruning_strategy=PruningStrategy.SMART,
        prune_threshold=0.85,  # Prune at 85% usage
    )

    print(f"Max context: {ctx.max_tokens:,} tokens")
    print(f"Reserved for response: {ctx.reserved_tokens:,} tokens")
    print(f"Pruning strategy: {ctx.pruning_strategy}")

    # Add system message
    print("\n\n1️⃣ Adding system message...")
    ctx.add_message(
        role="system",
        content="You are Victor, a helpful coding assistant.",
        priority=10,  # High priority - keep always
    )
    print_stats(ctx)

    # Add file context
    print("\n\n2️⃣ Adding file context...")
    with open("victor/providers/base.py", "r") as f:
        content = f.read()

    ctx.add_file(path="victor/providers/base.py", content=content, relevance_score=0.9)
    print_stats(ctx)

    # Simulate conversation
    print("\n\n3️⃣ Simulating conversation...")
    conversation = [
        ("user", "Explain how the provider system works"),
        (
            "assistant",
            "The provider system uses an abstract base class called BaseProvider...",
        ),
        ("user", "Show me an example"),
        ("assistant", "Here's an example of implementing a custom provider..."),
        ("user", "How do I add tool support?"),
        (
            "assistant",
            "To add tool support, you need to implement the tools parameter...",
        ),
    ]

    for i, (role, content) in enumerate(conversation):
        ctx.add_message(role, content, priority=5 - (i // 2))  # Decreasing priority
        if i % 2 == 1:  # After each Q&A pair
            print(f"\n   After message {i+1}:")
            print(f"   • Messages: {len(ctx.context.messages)}")
            print(f"   • Tokens: {ctx.context.total_tokens:,}")
            print(f"   • Usage: {ctx.context.usage_percentage:.1f}%")

    # Show final stats
    print("\n\n4️⃣ Final context state:")
    print("-" * 70)
    stats = ctx.get_stats()
    print(f"Total messages: {stats['total_messages']}")
    print(f"  - System: {stats['messages_by_role']['system']}")
    print(f"  - User: {stats['messages_by_role']['user']}")
    print(f"  - Assistant: {stats['messages_by_role']['assistant']}")
    print(f"Total files: {stats['total_files']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Available tokens: {stats['available_tokens']:,}")
    print(f"Usage: {stats['usage_percentage']:.1f}%")

    # Test token counting
    print("\n\n5️⃣ Token counting examples:")
    print("-" * 70)
    examples = [
        "Hello, world!",
        "def factorial(n): return 1 if n == 0 else n * factorial(n-1)",
        "The quick brown fox jumps over the lazy dog" * 100,  # Long text
    ]

    for text in examples:
        tokens = ctx.count_tokens(text)
        preview = text[:50] + ("..." if len(text) > 50 else "")
        print(f"Text: {preview}")
        print(f"Tokens: {tokens:,}\n")

    # Test pruning strategies
    print("\n\n6️⃣ Pruning strategies comparison:")
    print("-" * 70)

    for strategy in [
        PruningStrategy.FIFO,
        PruningStrategy.PRIORITY,
        PruningStrategy.SMART,
    ]:
        test_ctx = ProjectContextLoader(
            model="gpt-4",
            max_tokens=1000,  # Small window for demo
            reserved_tokens=100,
            pruning_strategy=strategy,
            prune_threshold=0.7,
        )

        # Add messages until pruning kicks in
        test_ctx.add_message("system", "System prompt", priority=10)
        for i in range(20):
            test_ctx.add_message("user", f"Question {i}" * 10, priority=5 - (i % 5))
            test_ctx.add_message("assistant", f"Answer {i}" * 10, priority=5)

        stats = test_ctx.get_stats()
        print(f"\n{strategy.value.upper()}:")
        print(f"  Messages: {stats['total_messages']}")
        print(f"  Tokens: {stats['total_tokens']:,}")
        print(f"  Usage: {stats['usage_percentage']:.1f}%")

    # Get formatted context
    print("\n\n7️⃣ Formatted context for prompt:")
    print("-" * 70)
    formatted = ctx.get_context_for_prompt()
    print(f"Total messages in prompt: {len(formatted)}")
    print("\nFirst message:")
    print(f"  Role: {formatted[0]['role']}")
    print(f"  Content (preview): {formatted[0]['content'][:100]}...")

    print("\n\n✨ Demo Complete!")
    print("\nContext management provides:")
    print("  ✓ Accurate token counting")
    print("  ✓ Automatic pruning strategies")
    print("  ✓ Message prioritization")
    print("  ✓ Smart file selection")
    print("  ✓ Context window optimization")


def print_stats(ctx: ProjectContextLoader) -> None:
    """Print context stats."""
    stats = ctx.get_stats()
    print(f"   Tokens: {stats['total_tokens']:,}/{ctx.max_tokens:,}")
    print(f"   Usage: {stats['usage_percentage']:.1f}%")
    print(f"   Available: {stats['available_tokens']:,}")


if __name__ == "__main__":
    main()
