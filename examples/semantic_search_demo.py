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

"""Demo of semantic search with embeddings.

This shows how to use CodebaseIndex with embeddings for semantic code search.

Requirements:
    pip install chromadb sentence-transformers

Usage:
    python examples/semantic_search_demo.py
"""

import asyncio
from victor_coding.codebase import CodebaseIndex


async def main():
    """Demo semantic search."""
    print("🔍 Semantic Search Demo\n")
    print("=" * 70)

    # Create indexer WITH embeddings enabled
    print("\n📊 Creating indexer with embeddings...")
    indexer = CodebaseIndex(
        root_path=".",
        use_embeddings=True,
        embedding_config={
            "vector_store": "chromadb",
            "embedding_model_type": "sentence-transformers",
            "embedding_model_name": "all-mpnet-base-v2",  # Best quality local model
            "persist_directory": "~/.victor/embeddings/demo",
        },
    )

    # Index the codebase
    print("\n🔨 Indexing codebase (this may take a minute)...")
    await indexer.index_codebase()

    # Show stats
    print("\n📈 Index Statistics:")
    print("-" * 70)
    stats = indexer.get_stats()
    print(f"Total files: {stats['total_files']}")
    print(f"Total symbols: {stats['total_symbols']}")
    print(f"Total lines: {stats['total_lines']:,}")
    print(f"Embeddings enabled: {stats['embeddings_enabled']}")
    if "embedding_stats" in stats:
        emb_stats = stats["embedding_stats"]
        print("\nEmbedding stats:")
        print(f"  Model: {emb_stats.get('model_name', 'N/A')}")
        print(f"  Documents: {emb_stats.get('total_documents', 0)}")

    # Semantic search examples
    print("\n\n🔎 Semantic Search Examples:")
    print("=" * 70)

    queries = [
        "How do I connect to a database?",
        "Functions related to HTTP requests",
        "Code for parsing command line arguments",
        "Embedding and vector search implementation",
        "Tool execution and error handling",
    ]

    for query in queries:
        print(f'\n💬 Query: "{query}"')
        print("-" * 70)

        try:
            results = await indexer.semantic_search(query, max_results=3)

            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['file_path']}:{result.get('symbol_name', 'N/A')}")
                    print(f"   Line: {result.get('line_number', 'N/A')}")
                    print(f"   Relevance: {result['score']:.3f}")
                    print(f"   Preview: {result['content'][:100]}...")
            else:
                print("   No results found")

        except Exception as e:
            print(f"   Error: {e}")

    # Compare with keyword search
    print("\n\n📊 Comparison: Semantic vs Keyword Search")
    print("=" * 70)

    test_query = "authentication middleware"
    print(f'\nQuery: "{test_query}"')

    print("\n1️⃣ Keyword Search (traditional):")
    keyword_results = await indexer.find_relevant_files(test_query, max_files=3)
    if keyword_results:
        for result in keyword_results:
            print(f"   - {result.path} ({len(result.symbols)} symbols)")
    else:
        print("   No keyword matches")

    print("\n2️⃣ Semantic Search (with embeddings):")
    try:
        semantic_results = await indexer.semantic_search(test_query, max_results=3)
        if semantic_results:
            for result in semantic_results:
                print(f"   - {result['file_path']}:{result.get('symbol_name', 'N/A')}")
                print(f"     Score: {result['score']:.3f}")
        else:
            print("   No semantic matches")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n\n✨ Demo Complete!")
    print("\nSemantic search allows you to:")
    print("  ✓ Find code by meaning, not just keywords")
    print("  ✓ Discover related functionality")
    print("  ✓ Navigate codebases more intuitively")
    print("  ✓ Ask questions in natural language")

    print("\n💡 Try your own queries:")
    print("   from victor_coding.codebase import CodebaseIndex")
    print("   indexer = CodebaseIndex('.', use_embeddings=True)")
    print("   await indexer.index_codebase()")
    print("   results = await indexer.semantic_search('your question here')")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo cancelled")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nMake sure you have installed:")
        print("  pip install chromadb sentence-transformers")
