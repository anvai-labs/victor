# Graph-Based Code Intelligence Guide

## Overview

Victor's graph-based code intelligence features provide deep understanding of code structure, dependencies, and context through three key technologies:

1. **Code Context Graph (CCG)** - Statement-level control flow, control dependence, and data dependence graphs
2. **Graph RAG** - Retrieval-augmented generation using graph traversal for context
3. **Multi-Hop Retrieval** - Traverse dependency chains to find implicit code relationships

## Architecture

The [storage architecture diagram](../architecture.md#database-architecture) shows the
current code-graph store, dedicated SQLite worker, and separate workflow persistence.
Graph traversal and semantic retrieval combine context above those storage interfaces;
[backend status](../architecture.md#database-architecture) distinguishes defaults from
opt-in ProximaDB work. The schemas and examples below describe this guide’s feature surfaces,
not a claim that every optional backend is enabled.

## Quick Start

### CLI Commands

```bash
# Index your codebase with CCG
victor graph index --path /path/to/code --ccg --embeddings

# Query using natural language
victor graph query "authentication function" --path /path/to/code

# Analyze impact of changes
victor graph impact login_function --type forward --depth 3

# View graph statistics
victor graph stats --path /path/to/code

# Export graph for visualization
victor graph export --output graph.json --format json
```

### Python API

```python
from victor.core.graph_rag import (
    GraphIndexingPipeline,
    GraphIndexConfig,
    MultiHopRetriever,
    RetrievalConfig,
)
from victor.storage.graph import create_graph_store
from pathlib import Path

# Create graph store
graph_store = create_graph_store("sqlite", None, Path("/path/to/code"))
await graph_store.initialize()

# Index codebase
config = GraphIndexConfig(
    root_path=Path("/path/to/code"),
    enable_ccg=True,
    enable_embeddings=True,
)
pipeline = GraphIndexingPipeline(graph_store, config)
stats = await pipeline.index_repository()

# Multi-hop retrieval
retriever = MultiHopRetriever(graph_store, RetrievalConfig(max_hops=2))
result = await retriever.retrieve("how does authentication work?")
```

## Code Context Graph (CCG)

### What is CCG?

The Code Context Graph extends traditional symbol graphs with statement-level granularity and three types of edges:

1. **Control Flow Graph (CFG)** - Shows execution paths through code
2. **Control Dependence Graph (CDG)** - Shows which statements control others
3. **Data Dependence Graph (DDG)** - Shows variable def-use relationships

### Supported Languages

- Python
- JavaScript / TypeScript
- Go
- Rust
- Java
- C / C++

### Example CCG Structure

```python
# Source code
def process(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results

# CCG creates:
# - Statement nodes for each line
# - CFG edges showing the loop and conditional flow
# - DDG edges tracking 'item', 'results' variable usage
```

### Edge Types

| Category | Edge Type | Description |
|----------|-----------|-------------|
| **CFG** | `CFG_SUCCESSOR` | Sequential execution |
| **CFG** | `CFG_TRUE_BRANCH` | True branch of condition |
| **CFG** | `CFG_FALSE_BRANCH` | False branch of condition |
| **CDG** | `CDG` | Control dependence |
| **DDG** | `DDG_DEF_USE` | Definition to use |
| **DDG** | `DDG_RAW` | Read-after-write |
| **DDG** | `DDG_WAR` | Write-after-read |
| **DDG** | `DDG_WAW` | Write-after-write |

## Graph RAG Pipeline

### Three-Stage Framework

#### 1. G-Indexing

Builds a unified graph combining:
- Symbol definitions (functions, classes, variables)
- CCG edges (control flow, data flow)
- Semantic embeddings for vector search
- Subgraph caches for fast retrieval

```python
from victor.core.graph_rag import GraphIndexingPipeline, GraphIndexConfig

config = GraphIndexConfig(
    root_path=Path("."),
    enable_ccg=True,
    enable_embeddings=True,
    ccg_languages=["python", "javascript"],
)

pipeline = GraphIndexingPipeline(graph_store, config)
stats = await pipeline.index_repository()
# stats.files_processed, stats.nodes_created, stats.edges_created
```

#### 2. G-Retrieval

Multi-hop retrieval combines:
- **Seed selection** - Vector search for relevant starting nodes
- **Graph expansion** - BFS traversal following dependency edges
- **Relevance scoring** - Decay-with-distance ranking
- **Pruning** - Remove low-relevance nodes

```python
from victor.core.graph_rag import MultiHopRetriever, RetrievalConfig

config = RetrievalConfig(
    seed_count=5,      # Number of seed nodes
    max_hops=2,        # Max traversal depth
    top_k=10,          # Final result count
    edge_types=["CALLS", "REFERENCES", "DDG_DEF_USE"],
)

retriever = MultiHopRetriever(graph_store, config)
result = await retriever.retrieve("how is user data validated?")
# result.nodes, result.edges, result.execution_time_ms
```

#### 3. G-Generation

Builds prompts with graph context:

```python
from victor.core.graph_rag import GraphAwarePromptBuilder, PromptConfig

builder = GraphAwarePromptBuilder()
prompt = builder.build_prompt(
    query="explain the authentication flow",
    subgraphs=result.subgraphs,
    config=PromptConfig(format="hierarchical"),
)
```

## Tools

### graph_semantic_search Tool

Query your codebase using natural language:

```python
from victor.tools.graph_query_tool import graph_semantic_search

result = await graph_semantic_search(
    query="database connection handling",
    path="/path/to/code",
    mode="semantic",  # semantic, structural, hybrid
    max_hops=2,
)
```

### impact_analysis Tool

Analyze downstream/upstream dependencies:

```python
from victor.tools.graph_query_tool import impact_analysis

# Forward impact (what depends on this?)
result = await impact_analysis(
    target="validate_user",
    analysis_type="forward",
    max_depth=3,
    path="/path/to/code",
)

# Backward impact (what does this depend on?)
result = await impact_analysis(
    target="handle_request",
    analysis_type="backward",
    max_depth=2,
    path="/path/to/code",
)
```

## Configuration

### Settings

Add to your `settings.yaml`:

```yaml
search:
  graph:
    # CCG Building
    enable_ccg: true
    ccg_languages:
      - python
      - javascript
      - typescript
      - go
      - rust

    # Graph RAG
    enable_graph_rag: true
    rag_seed_count: 5
    rag_max_hops: 2

    # Subgraph Caching
    enable_subgraph_cache: true
    subgraph_cache_ttl: 3600
```

### Feature Flags

Enable/disable features via environment variables:

```bash
# Enable Graph RAG
export VICTOR_USE_GRAPH_RAG=true

# Enable CCG building
export VICTOR_USE_CCG=true

# Enable multi-hop retrieval
export VICTOR_USE_MULTI_HOP_RETRIEVAL=true
```

## Performance

### Benchmarks

| Operation | Target | Notes |
|-----------|--------|-------|
| CCG Building | <100ms/file | Per 100-line file |
| Multi-hop Retrieval | <500ms | 2-hop query |
| Graph Query | <200ms | Symbol search |
| Subgraph Cache Hit | <50ms | Cached retrieval |

### Optimization Tips

1. **Enable subgraph caching** for frequently-accessed code
2. **Limit CCG languages** to only what you need
3. **Use semantic mode** for conceptual queries
4. **Use structural mode** for dependency analysis
5. **Adjust max_hops** based on codebase size

## Advanced Features

### Requirement Graph

Map requirements to code symbols:

```python
from victor.core.graph_rag.requirement_graph import RequirementGraphBuilder

req_builder = RequirementGraphBuilder(graph_store)
symbols = await req_builder.map_requirement(
    requirement="User should be able to reset password via email",
)
```

### Graph-Aware Embeddings

Generate embeddings that capture structure:

```python
from victor.processing.graph_embeddings import GraphAwareEmbedder

embedder = GraphAwareEmbedder()
embedding = await embedder.embed_with_context(
    node=graph_node,
    graph=code_context_graph,
)
```

### Graph Algorithms

Apply NetworkX algorithms:

```python
from victor.processing.graph_algorithms import compute_all_metrics

# nodes and edges are lists of GraphNode and GraphEdge from the indexed graph.
metrics = compute_all_metrics(nodes, edges)
central_nodes = metrics.pagerank
communities = metrics.communities
```

These helpers operate on explicit node/edge lists. There is no `GraphAlgorithmRunner`
API. NetworkX is optional; inspect the helper's fallback behavior before relying on
an algorithm-specific result.

## Schema Reference

### GraphNode Fields

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | str | Unique identifier |
| `type` | str | function, class, statement, requirement |
| `name` | str | Symbol name |
| `file` | str | Source file path |
| `line` | int | Start line number |
| `ast_kind` | str | Tree-sitter node kind (CCG) |
| `scope_id` | str | Hierarchical scope (CCG) |
| `statement_type` | str | Statement category (CCG) |
| `requirement_id` | str | Linked requirement (new) |

### GraphEdge Fields

| Field | Type | Description |
|-------|------|-------------|
| `src` | str | Source node ID |
| `dst` | str | Destination node ID |
| `type` | str | Edge type (see EdgeType enum) |
| `weight` | float | Optional weight (0-1) |

## Migration

### v4 → v5 Schema Migration

The schema migration is automatic on first run after upgrade:

```sql
-- New CCG columns
ALTER TABLE graph_node ADD COLUMN ast_kind TEXT;
ALTER TABLE graph_node ADD COLUMN scope_id TEXT;
ALTER TABLE graph_node ADD COLUMN statement_type TEXT;

-- New tables
CREATE TABLE graph_requirement (...);
CREATE TABLE graph_subgraph (...);
```

No manual intervention required. Existing data is preserved.

## Troubleshooting

### CCG Building Fails

**Issue**: CCG building fails for certain files

**Solution**: Check that Tree-sitter parsers are installed for your language:

```bash
# Install the Python embedding/grammar dependencies used by Victor
pip install "victor-ai[embeddings]"
# Re-run indexing and inspect its skipped-file/parser diagnostics
victor graph index --path /path/to/code --ccg --force
```

### Slow Multi-Hop Queries

**Issue**: Retrieval takes longer than expected

**Solution**:
- Reduce `max_hops` parameter
- Enable subgraph caching
- Limit query to specific directories

### Missing Dependencies

**Issue**: Import errors for graph modules

**Solution**: Install optional dependencies:

```bash
pip install "victor-ai[embeddings]"
```

## Contributing

### Adding New Edge Types

1. Add to `victor/storage/graph/edge_types.py`:

```python
class EdgeType(str, Enum):
    YOUR_NEW_EDGE = "YOUR_NEW_EDGE"
```

2. Update builder to create edges

3. Add tests for new edge type

### Adding New Languages

1. Add Tree-sitter parser to `ccg_builder.py`
2. Implement language-specific mappings
3. Add tests for new language

## References

- [GraphCoder: Enhanced Code Generation with Control Flow Graphs](https://arxiv.org/abs/2401.05867)
- [CodexGraph: Code Knowledge Graph with Retrieval](https://arxiv.org/abs/2405.13123)
- [GraphCodeAgent: Multi-Hop Graph Retrieval](https://arxiv.org/abs/2406.07694)
- [GraphRAG: From Local to Global](https://arxiv.org/abs/2404.16130)
- [CGM: Code Graph Mining](https://arxiv.org/abs/2305.17321)
