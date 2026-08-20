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

"""Demo of codebase indexing capabilities.

This shows how to use the CodebaseIndex to understand a codebase.
"""

import asyncio
from victor_coding.codebase import CodebaseIndex


async def main():
    """Demo codebase indexing."""
    print("🔍 Codebase Indexing Demo\n")
    print("=" * 60)

    # Index the current project
    indexer = CodebaseIndex(root_path=".")

    # Build the index
    await indexer.index_codebase()

    # Show stats
    print("\n📊 Index Statistics:")
    print("-" * 60)
    stats = indexer.get_stats()
    print(f"Total files: {stats['total_files']}")
    print(f"Total symbols: {stats['total_symbols']}")
    print(f"Total lines: {stats['total_lines']:,}")

    # Find files related to "provider"
    print("\n\n🔎 Finding files related to 'provider':")
    print("-" * 60)
    relevant_files = await indexer.find_relevant_files("provider", max_files=5)

    for file in relevant_files:
        print(f"\n📄 {file.path}")
        print(f"   Lines: {file.lines}")
        print(f"   Symbols: {len(file.symbols)}")
        if file.symbols:
            print(f"   Contains: {', '.join(s.name for s in file.symbols[:3])}...")

    # Find a specific symbol
    print("\n\n🎯 Finding symbol 'BaseProvider':")
    print("-" * 60)
    symbol = indexer.find_symbol("BaseProvider")
    if symbol:
        print(f"Found in: {symbol.file_path}:{symbol.line_number}")
        print(f"Type: {symbol.type}")
        if symbol.docstring:
            print(f"Doc: {symbol.docstring[:100]}...")
    else:
        print("Symbol not found")

    # Get context for a file
    print("\n\n📋 Getting context for 'victor/providers/base.py':")
    print("-" * 60)
    context = indexer.get_file_context("victor/providers/base.py")

    if context:
        file = context["file"]
        print(f"File: {file.path}")
        print(f"Symbols: {len(file.symbols)}")
        print(f"Imports: {len(file.imports)}")
        print(f"Dependencies: {len(context['dependencies'])}")
        print(f"Dependents: {len(context['dependents'])}")

        if context["dependents"]:
            print("\nFiles that depend on this:")
            for dep in context["dependents"][:5]:
                print(f"  - {dep.path}")

    print("\n\n✅ Demo completed!")
    print("\nThis indexing enables:")
    print("  - Smart file discovery")
    print("  - Symbol search")
    print("  - Dependency analysis")
    print("  - Context-aware code assistance")


if __name__ == "__main__":
    asyncio.run(main())
