# Graph RAG Quick Start Guide

Get started with Victor's graph-based code intelligence in 5 minutes.

## Installation

Install Victor with semantic embedding support:

```bash
pip install "victor-ai[embeddings]"
```

## 1. Index Your Codebase

First, create a graph index of your code:

```bash
victor graph index --path /path/to/your/code --ccg --embeddings
```

This will:
- Extract symbols (functions, classes, variables)
- Build Code Context Graphs (CFG, CDG, DDG)
- Create embeddings for semantic search
- Cache subgraphs for fast retrieval

Illustrative output (counts and formatting depend on the repository and version):
```
Indexing codebase at: /path/to/your/code
Processing files...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%

✓ Indexing complete
  Files processed: 150
  Nodes created: 1,234
  Edges created: 3,456
  CCG nodes: 5,678
  CCG edges: 8,901
```

## 2. Query Your Code

Ask questions about your codebase in natural language:

```bash
victor graph query "how does user authentication work?" --path /path/to/your/code
```

The query will:
1. Find relevant symbols using semantic search
2. Traverse the graph to find dependencies
3. Return results ranked by relevance

## 3. Analyze Impact

Before making changes, see what will be affected:

```bash
# Forward impact: what depends on this function?
victor graph impact authenticate_user --type forward --depth 3 --path /path/to/your/code

# Backward impact: what does this function depend on?
victor graph impact process_payment --type backward --depth 2 --path /path/to/your/code
```

## 4. Check Statistics

View graph statistics:

```bash
victor graph stats --path /path/to/your/code
```

## Python API Usage

### Basic Indexing

```python
import asyncio
from pathlib import Path
from victor.storage.graph import create_graph_store
from victor.core.graph_rag import GraphIndexingPipeline, GraphIndexConfig

async def index_codebase():
    # Create graph store
    graph_store = create_graph_store("sqlite", None, Path("/path/to/code"))
    await graph_store.initialize()

    # Configure indexing
    config = GraphIndexConfig(
        root_path=Path("/path/to/code"),
        enable_ccg=True,
        enable_embeddings=True,
    )

    # Index
    pipeline = GraphIndexingPipeline(graph_store, config)
    stats = await pipeline.index_repository()

    print(f"Indexed {stats.files_processed} files")
    print(f"Created {stats.nodes_created} nodes, {stats.edges_created} edges")

    await graph_store.close()

asyncio.run(index_codebase())
```

### Querying the Graph

```python
import asyncio
from pathlib import Path
from victor.storage.graph import create_graph_store
from victor.core.graph_rag import MultiHopRetriever, RetrievalConfig

async def query_codebase():
    graph_store = create_graph_store("sqlite", None, Path("/path/to/code"))
    await graph_store.initialize()

    # Configure retrieval
    config = RetrievalConfig(
        seed_count=5,
        max_hops=2,
        top_k=10,
    )

    # Create retriever
    retriever = MultiHopRetriever(graph_store, config)

    # Query
    result = await retriever.retrieve("how is password hashing done?")

    print(f"Found {len(result.nodes)} symbols in {result.execution_time_ms:.1f}ms")

    for node in result.nodes[:5]:
        print(f"  - {node.name} ({node.type}) in {node.file}")

    await graph_store.close()

asyncio.run(query_codebase())
```

### Impact Analysis

```python
async def analyze_impact():
    from victor.storage.graph import create_graph_store

    graph_store = create_graph_store("sqlite", None, Path("/path/to/code"))
    await graph_store.initialize()

    # Find the target function
    nodes = await graph_store.find_nodes(name="authenticate_user")
    if nodes:
        target_id = nodes[0].node_id

        # Get downstream dependencies (forward impact)
        edges = await graph_store.get_neighbors(
            target_id,
            direction="out",
            max_depth=3,
        )

        impacted = {edge.dst for edge in edges}

        print(f"Changing authenticate_user impacts {len(impacted)} symbols:")
        for node_id in list(impacted)[:10]:
            node = await graph_store.get_node_by_id(node_id)
            if node:
                print(f"  - {node.name} in {node.file}")

    await graph_store.close()

asyncio.run(analyze_impact())
```

## Using with Tools

### In Chat Sessions

Graph tool exposure depends on the active vertical/tool selection and feature flags.
The registered semantic tool is `graph_semantic_search`; `VICTOR_USE_GRAPH_QUERY_TOOL`
controls its runtime gate. The following exchange illustrates a session where it is enabled:

```
You: Find functions that handle user authentication

Victor: [Uses graph_semantic_search tool]
I found 3 authentication-related functions:
1. authenticate_user() in auth.py:45
2. validate_credentials() in auth.py:12
3. create_session() in session.py:23

[Follows with multi-hop retrieval for context]
```

### In Python Code

```python
import asyncio
from victor.tools.graph_query_tool import graph_semantic_search

async def analyze_code():
    result = await graph_semantic_search(
        query="database error handling",
        path="/path/to/code",
        mode="semantic",
        max_hops=2,
    )

    for node in result["results"]:
        print(f"{node['name']}: {node.get('signature', 'N/A')}")

asyncio.run(analyze_code())
```

## Configuration

Create or update `~/.victor/settings.yaml`:

```yaml
search:
  graph:
    # Enable/disable features
    enable_ccg: true
    enable_graph_rag: true

    # Languages to analyze with CCG
    ccg_languages:
      - python
      - javascript
      - typescript

    # Retrieval settings
    rag_seed_count: 5
    rag_max_hops: 2

    # Performance
    enable_subgraph_cache: true
    subgraph_cache_ttl: 3600
```

## Common Use Cases

### 1. Understanding Legacy Code

```bash
# Map out the data flow
victor graph query "how does data flow from API to database?"

# Find all entry points
victor graph query "HTTP endpoint handlers"
```

### 2. Refactoring Safety

```bash
# Check impact before refactoring
victor graph impact old_function_name --type forward --depth 5

# Find similar code that might need updating
victor graph query "functions that parse JSON"
```

### 3. Onboarding New Developers

```bash
# Get overview of core functionality
victor graph query "main business logic functions"

# Find key architectural patterns
victor graph query "dependency injection pattern"
```

### 4. Debugging

```bash
# Trace error propagation
victor graph impact error_handling --type forward --depth 4

# Find where exceptions are caught
victor graph query "exception handling code"
```

## Tips and Tricks

### Speed Up Queries

```python
# Use structural mode to search indexed symbol text without vector seeds
config = RetrievalConfig(
    mode="structural",  # Skips vector search
    max_hops=1,         # Reduce traversal depth
)
```

### Focus on Specific Files

```python
# Limit search to specific directory
result = await graph_semantic_search(
    query="authentication",
    path="/path/to/project",  # Root of the graph index; not a result-path filter
)
```

### Export for Visualization

```bash
# Export graph as JSON
victor graph export --output graph.json --format json

# Export as DOT for GraphViz
victor graph export --output graph.dot --format dot

# Render DOT to PNG
dot -Tpng graph.dot -o graph.png
```

## Troubleshooting

### No Results Found

```bash
# Check that graph exists
victor graph stats --path /path/to/your/code

# Re-index if needed
victor graph index --force
```

### Slow Performance

```bash
# Disable CCG for faster indexing
victor graph index --no-ccg

# Reduce hop count
victor graph query "query" --hops 1
```

### Import Errors

```bash
# Install graph dependencies
pip install "victor-ai[embeddings]"

# Or install individually
pip install networkx sentence-transformers
```

## Next Steps

- Read the [Graph RAG Guide](graph-rag-guide.md) for detailed concepts
- Check the [API Reference](graph-api-reference.md) for all available functions
- See [Examples](https://github.com/anvai-labs/victor/tree/develop/examples/) for more code samples
