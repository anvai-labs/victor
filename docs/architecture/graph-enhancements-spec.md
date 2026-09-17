# Graph-Based Enhancements for Victor: Specification & Design

> **Historical design/review snapshot.** The findings, proposed APIs, diagrams, and rollout
> status below describe their original review context. For current implementation and remaining
> work, use [Architecture](../architecture.md) and the [Roadmap](../roadmap.md); this document
> is not an installation guide or a certification that every proposed feature shipped.

## Executive Summary

This specification defines graph-based mechanisms to enhance Victor's IDE experience and coding agent capabilities, drawing from recent research in repository-level code intelligence. The proposed enhancements integrate Code Context Graphs (CCG), Graph Retrieval-Augmented Generation (GraphRAG), and multi-hop reasoning to significantly improve code completion, bug fixing, and repository-aware generation.

**Status**: Design Phase - Ready for Implementation Planning
**Version**: 1.0
**Date**: 2025-04-28

---

## Table of Contents

1. [Research Summary](#1-research-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Proposed Enhancements](#3-proposed-enhancements)
4. [Architecture Design](#4-architecture-design)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Use Cases](#6-use-cases)
7. [Success Metrics](#7-success-metrics)

---

## 1. Research Summary

### 1.1 Key Papers Analyzed

| Paper | Core Contribution | Applicability |
|-------|------------------|---------------|
| **GraphCoder** (2406.07003) | Code Context Graph (CCG) with CFG+CDG+DDG edges; coarse-to-fine retrieval | HIGH - Statement-level graph structure |
| **CodexGraph** (2408.03910) | Graph database interface; "write then translate" query pattern | HIGH - Graph query translation |
| **GraphCodeAgent** (2504.10046) | Dual graphs (Requirement + Structural-Semantic); multi-hop reasoning | HIGH - Implicit dependency retrieval |
| **CGM** (2505.16901) | Graph-integrated LLM; attention mask modification; semantic+structural integration | MEDIUM - Model-level integration |
| **GraphRAG Survey** (2408.08921) | G-Indexing, G-Retrieval, G-Generation framework | HIGH - Systematic framework |

### 1.2 Core Techniques Identified

1. **Code Context Graph (CCG)**: Superimposition of Control Flow Graph (CFG), Control Dependence Graph (CDG), and Data Dependence Graph (DDG)
2. **Graph RAG Pipeline**: Three-stage framework (Indexing → Retrieval → Generation)
3. **Multi-hop Reasoning**: Traversal through dependency chains to retrieve implicit code
4. **Query Translation**: Natural language to graph query (e.g., Cypher) conversion
5. **Subgraph Extraction**: Context-aware graph slicing with decay-with-distance weighting
6. **Dual Graph Architecture**: Separate Requirement Graph and Code Graph with mapping

### 1.3 Performance Improvements (from papers)

- GraphCoder: +6.06 code match, +6.23 identifier match vs baseline RAG
- CodexGraph: Competitive on SWE-bench Lite (47.99% with GPT-4o)
- GraphCodeAgent: +43.81% relative improvement on DevEval
- CGM: 43.00% resolution on SWE-bench Lite (open-source SOTA)

---

## 2. Current State Analysis

### 2.1 Victor-Coding Graph Implementation

**Location**: `victor/storage/graph/`, `victor_coding/codebase/graph/`

**Existing Capabilities**:
```python
# Protocol (victor/storage/graph/protocol.py)
class GraphStoreProtocol:
    async def upsert_nodes(nodes: Iterable[GraphNode]) -> None
    async def upsert_edges(edges: Iterable[GraphEdge]) -> None
    async def get_neighbors(node_id: str, edge_types, max_depth: int) -> List[GraphEdge]
    async def search_symbols(query: str, limit: int) -> List[GraphNode]
    async def find_nodes(*, name, type, file) -> List[GraphNode]

# Node Types
GraphNode: node_id, type, name, file, line, end_line, lang,
           signature, docstring, parent_id, embedding_ref, metadata

# Edge Types
GraphEdge: src, dst, type (CALLS, REFERENCES, CONTAINS, INHERITS), weight, metadata
```

**Storage Backends**:
- `SqliteGraphStore`: Production default, FTS5 full-text search
- `MemoryGraphStore`: In-memory for testing
- `DuckDBGraphStore`: Alternative (optional)

### 2.2 Victor-Coding Indexer

**Location**: `victor_coding/codebase/indexer.py`

**Current Features**:
- Tree-sitter based symbol extraction (multi-language)
- AST-based symbol resolution
- File watching with staleness detection
- Parallel processing support (ProcessPoolExecutor)
- SymbolResolver for cross-file references

**Extracted Elements**:
- Functions, classes, methods, variables
- Import statements
- Inheritance relationships
- Basic call graph (CALLS edges)

### 2.3 Gaps Identified

| Gap | Impact | Priority |
|-----|--------|----------|
| No statement-level graph (CFG/CDG/DDG) | Missing fine-grained context for completion | HIGH |
| No graph query language (NL→Cypher) | LLM cannot perform structured navigation | HIGH |
| No multi-hop reasoning API | Cannot retrieve implicit dependencies | HIGH |
| No subgraph extraction with context | Retrieved context lacks structural relationships | MEDIUM |
| No requirement-graph mapping | NL→code gap remains | MEDIUM |
| No graph-aware embeddings | Semantic search ignores structure | MEDIUM |

---

## 3. Proposed Enhancements

### 3.1 Statement-Level Code Context Graph (CCG)

**Inspired by**: GraphCoder (2406.07003)

**Objective**: Capture statement-level control flow, data dependence, and control dependence for accurate code completion context.

**Design**:

```python
@dataclass
class StatementNode(GraphNode):
    """Statement-level node for CCG."""
    statement_type: str  # if, for, while, assignment, return, etc.
    variables_defined: List[str]
    variables_used: List[str]
    ast_path: str  # Path in AST for reconstruction

@dataclass
class CodeContextGraph:
    """Code Context Graph = CFG + CDG + DDG"""
    nodes: Dict[str, StatementNode]
    edges: List[GraphEdge]

    # Edge types from GraphCoder
    EDGE_CF = "control_flow"   # Sequential execution
    EDGE_CD = "control_dep"    # Control dependence (if/for affects execution)
    EDGE_DD = "data_dep"       # Data dependence (variable definition→use)

    def slice_context(
        self,
        target_statement: str,
        max_hops: int = 3,
        max_statements: int = 50
    ) -> "CodeContextGraph":
        """Extract h-hop CCG slice with decay-with-distance weighting."""
```

**Implementation Notes**:
- Build CFG from AST using existing Tree-sitter infrastructure
- Compute CDG via dominance frontier algorithm
- Compute DDG via reaching definitions (variable liveness)
- Store as edge annotations on existing graph store

**Use Cases**:
- Repository-level code completion (predict next statement)
- Context-aware refactoring suggestions
- Impact analysis for code changes

### 3.2 Graph RAG Framework

**Inspired by**: GraphRAG Survey (2408.08921)

**Objective**: Systematic three-stage framework for graph-based retrieval and generation.

**Design**:

```python
class GraphRAGPipeline:
    """Three-stage GraphRAG framework."""

    # Stage 1: Graph-Based Indexing (G-Indexing)
    async def index_repository(
        self,
        repo_path: str,
        graph_schema: GraphSchema
    ) -> None:
        """Build and persist code graph with embeddings."""
        # 1. Extract symbols (existing indexer)
        # 2. Build CCG (new)
        # 3. Generate embeddings (existing)
        # 4. Store in graph database (existing + new edges)

    # Stage 2: Graph-Guided Retrieval (G-Retrieval)
    async def retrieve(
        self,
        query: str,
        retrieval_config: RetrievalConfig
    ) -> GraphRetrievalResult:
        """Retrieve relevant subgraph for query."""
        # 1. Query enhancement (decomposition, expansion)
        # 2. Graph traversal (multi-hop)
        # 3. Subgraph extraction
        # 4. Reranking (structural + semantic)

    # Stage 3: Graph-Enhanced Generation (G-Generation)
    async def generate(
        self,
        query: str,
        retrieved_graph: GraphRetrievalResult
    ) -> str:
        """Generate response using graph context."""
        # 1. Format subgraph as context
        # 2. Inject into LLM prompt
        # 3. Generate with graph-aware prompt
```

**Configuration**:

```python
@dataclass
class RetrievalConfig:
    """Configuration for graph retrieval."""
    max_depth: int = 3           # Multi-hop depth
    max_nodes: int = 100         # Context size limit
    edge_types: List[str] = None # Filter edge types
    decay_factor: float = 0.5    # Distance decay
    similarity_threshold: float = 0.7
    use_semantic_search: bool = True
    use_structure_search: bool = True
```

### 3.3 Natural Language to Graph Query Translation

**Inspired by**: CodexGraph "write then translate" pattern

**Objective**: Enable LLM agents to construct and execute graph queries for structured code navigation.

**Design**:

```python
class GraphQueryTranslator:
    """Translates natural language to executable graph queries."""

    async def translate(
        self,
        nl_query: str,
        schema: GraphSchema
    ) -> GraphQuery:
        """Translate NL query to graph query (SQL/Cypher-like)."""
        # Uses LLM to generate structured query

    async def execute(
        self,
        query: GraphQuery,
        graph_store: GraphStoreProtocol
    ) -> GraphQueryResult:
        """Execute translated query and return results."""

# Example queries
QUERIES = {
    "find_classes_with_method": """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method {{name: '{method_name}'}})
        RETURN c.name, c.file, m.signature
    """,
    "trace_dependency_chain": """
        MATCH path = (start:Symbol {{name: '{start}'}})-[:CALLS*1..{depth}]->(end:Symbol)
        RETURN path
    """,
    "find_all_callers": """
        MATCH (caller)-[:CALLS]->(callee:Symbol {{name: '{callee_name}'}})
        RETURN caller
    """
}
```

**Implementation Approach**:
1. Define query templates for common patterns
2. Use few-shot prompting for LLM translation
3. Validate generated queries against schema
4. Execute via graph store (SQLite or dedicated graph DB)

### 3.4 Multi-Hop Dependency Traversal

**Inspired by**: GraphCodeAgent dual graph approach

**Objective**: Retrieve implicit code dependencies through multi-hop reasoning.

**Design**:

```python
class MultiHopRetriever:
    """Multi-hop traversal for implicit dependency retrieval."""

    async def retrieve_implicit_context(
        self,
        target_node: str,
        query: str,
        max_hops: int = 3
    ) -> RetrievalResult:
        """Retrieve all relevant code including implicit dependencies."""
        result = RetrievalResult()

        # Hop 1: Direct dependencies (CALLS, REFERENCES)
        direct = await self.graph_store.get_neighbors(
            target_node,
            edge_types=["CALLS", "REFERENCES", "USES"],
            max_depth=1
        )
        result.add(direct, hop=1)

        # Hop 2+: Transitive dependencies
        for hop in range(2, max_hops + 1):
            frontier = result.get_hop_nodes(hop - 1)
            transitive = await self._traverse_frontier(frontier)
            # Re-rank by relevance to query
            ranked = self._rerank_by_query(transitive, query)
            result.add(ranked, hop=hop)

        return result

    def _rerank_by_query(
        self,
        nodes: List[GraphNode],
        query: str
    ) -> List[GraphNode]:
        """Re-rank nodes by semantic similarity to query."""
        # Combine structural distance + semantic similarity
        for node in nodes:
            node.score = (
                self.decay_factor ** node.hop *  # Structural decay
                self.semantic_similarity(node, query)  # Semantic match
            )
        return sorted(nodes, key=lambda n: n.score, reverse=True)
```

**Edge Types for Traversal**:
- `CALLS`: Function/method invocation
- `REFERENCES`: Variable/type reference
- `USES`: Import/dependency
- `INHERITS`: Class inheritance
- `CONTAINS`: Parent-child (class→method)

### 3.5 Requirement Graph Integration

**Inspired by**: GraphCodeAgent Requirement Graph (RG)

**Objective**: Bridge the gap between natural language requirements and code implementation.

**Design**:

```python
@dataclass
class RequirementNode:
    """Node in Requirement Graph."""
    requirement_id: str
    description: str
    subrequirements: List[str]  # Child requirements
    mapped_symbols: List[str]   # Mapped code symbols

@dataclass
class RequirementGraph:
    """Graph of requirement relationships."""
    nodes: Dict[str, RequirementNode]
    edges: List[GraphEdge]  # PARENT, SEMANTIC_SIMILAR

    # Mapping to Code Graph
    symbol_mappings: Dict[str, str]  # requirement_id → symbol_id

class RequirementMapper:
    """Maps requirements to code symbols."""

    async def map_requirement_to_symbols(
        self,
        requirement: str,
        code_graph: StructuralSemanticCodeGraph
    ) -> List[str]:
        """Map natural language requirement to relevant code symbols."""
        # 1. Semantic search for similar functions/classes
        # 2. Graph traversal for dependencies
        # 3. Return ranked list of symbol_ids
```

### 3.6 Graph-Aware Embeddings

**Inspired by**: CGM semantic+structural integration

**Objective**: Generate embeddings that capture both code semantics and graph structure.

**Design**:

```python
class GraphAwareEmbedder:
    """Generates structure-aware code embeddings."""

    async def embed_with_context(
        self,
        node: GraphNode,
        graph: CodeContextGraph
    ) -> List[float]:
        """Generate embedding considering graph context."""
        # 1. Get node's code/text embedding
        text_emb = await self.text_embedder.embed(node.to_text())

        # 2. Get structural context (neighborhood)
        neighbors = await graph.get_neighbors(node.node_id, max_depth=2)

        # 3. Generate structural embedding (GNN-style)
        struct_emb = self._encode_structure(neighbors)

        # 4. Fuse embeddings
        return self.fusion_fn(text_emb, struct_emb)

    def _encode_structure(self, neighbors: List[GraphEdge]) -> List[float]:
        """Encode graph structure into embedding."""
        # Options: mean aggregation, attention, GNN layer
        # Simplified: aggregate neighbor embeddings with edge-type weights
```

---

## 4. Architecture Design

### 4.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Victor Agent                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │
│  │   Chat/CLI    │  │     TUI       │  │  HTTP API     │          │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘          │
│          │                   │                   │                   │
│          └───────────────────┼───────────────────┘                   │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    GraphRAG Pipeline                          │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │  │
│  │  │ G-Indexing  │→│ G-Retrieval │→│   G-Generation       │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Graph Query Layer                           │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │  ┌───────────────────┐  ┌─────────────────────────────────┐  │  │
│  │  │ NL→Query Translator│  │   Multi-Hop Traversal           │  │  │
│  │  └───────────────────┘  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Graph Store Layer                           │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │  │
│  │  │    SQLite   │  │   Memory    │  │  ProximaDB/Neo4j    │  │  │
│  │  │ Graph Store │  │ Graph Store │  │  (optional)         │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Code Indexers                               │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │  ┌───────────────────┐  ┌─────────────────────────────────┐  │  │
│  │  │ Symbol Indexer    │  │   CCG Builder (NEW)              │  │  │
│  │  │ (Tree-sitter)     │  │   (CFG+CDG+DDG)                  │  │  │
│  │  └───────────────────┘  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Model

```python
# Extended graph schema
class GraphSchema:
    """Unified schema for code graphs."""

    # Node types
    NODE_TYPES = {
        # Existing
        "repository", "module", "package", "file", "class",
        "function", "method", "variable", "parameter",
        # New (CCG)
        "statement", "predicate", "basic_block",
    }

    # Edge types
    EDGE_TYPES = {
        # Existing
        "CONTAINS",      # Parent-child (file→class)
        "CALLS",         # Function invocation
        "REFERENCES",    # Variable reference
        "INHERITS",      # Class inheritance
        "IMPORTS",       # Module import
        # New (CCG from GraphCoder)
        "control_flow",  # CFG edge (sequential execution)
        "control_dep",   # CDG edge (control dependence)
        "data_dep",      # DDG edge (data dependence)
        # New (Requirement Graph)
        "implements",    # Code implements requirement
        "similar_to",    # Semantic similarity
    }

    # Node attributes (metadata)
    NODE_ATTRIBUTES = {
        # Existing
        "name", "file", "line", "end_line", "signature",
        "docstring", "parent_id", "embedding_ref",
        # New
        "variables_defined", "variables_used",  # For statements
        "requirement_id",                       # For requirement mapping
        "context_hash",                         # For CCG slicing
    }
```

### 4.3 API Design

```python
# Main API for graph-enhanced coding agent
class GraphEnhancedAgent:
    """Main interface for graph-based code intelligence."""

    async def complete_code(
        self,
        file_path: str,
        cursor_line: int,
        context_lines: int = 10
    ) -> CodeCompletion:
        """Repository-aware code completion using CCG."""
        # 1. Build CCG for current context
        # 2. Retrieve similar code patterns via graph traversal
        # 3. Generate completion with LLM

    async def fix_bug(
        self,
        issue_description: str,
        file_path: str | None = None
    ) -> BugFix:
        """Bug fixing with multi-hop dependency retrieval."""
        # 1. Map issue to requirement graph
        # 2. Retrieve relevant code via multi-hop traversal
        # 3. Generate fix with full dependency context

    async def search_code(
        self,
        query: str,
        search_type: SearchType = SearchType.SEMANTIC
    ) -> List[CodeResult]:
        """Graph-aware code search."""
        # Options: SEMANTIC, STRUCTURAL, HYBRID

    async def analyze_impact(
        self,
        file_path: str,
        symbol_name: str
    ) -> ImpactAnalysis:
        """Analyze impact of code change via dependency graph."""
        # 1. Find all callers (reverse CALLS edges)
        # 2. Find all inheritors (reverse INHERITS edges)
        # 3. Return impact tree
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

**Goal**: Extend existing graph infrastructure with CCG support.

| Task | Description | Files |
|------|-------------|-------|
| 1.1 | Add statement-level node types | `victor/storage/graph/protocol.py` |
| 1.2 | Implement CCG builder (CFG) | `victor_coding/codebase/graph/ccg_builder.py` |
| 1.3 | Add CDG/DDG computation | `victor_coding/codebase/graph/dependence.py` |
| 1.4 | Extend graph store schema | `victor/storage/graph/sqlite_store.py` |

**Deliverables**:
- CCG builder for Python (proof of concept)
- Statement nodes in graph store
- Unit tests for CCG construction

### Phase 2: Graph RAG Pipeline (Weeks 5-8)

**Goal**: Implement three-stage GraphRAG framework.

| Task | Description | Files |
|------|-------------|-------|
| 2.1 | Define GraphRAG pipeline interface | `victor/framework/graph_rag.py` |
| 2.2 | Implement G-Indexing with CCG | `victor/framework/graph_rag/indexing.py` |
| 2.3 | Implement G-Retrieval with multi-hop | `victor/framework/graph_rag/retrieval.py` |
| 2.4 | Implement G-Generation prompt format | `victor/framework/graph_rag/generation.py` |

**Deliverables**:
- GraphRAG pipeline with CCG integration
- Multi-hop retrieval API
- Prompt templates for graph-enhanced generation

### Phase 3: Query Translation (Weeks 9-12)

**Goal**: Enable NL-to-graph-query translation.

| Task | Description | Files |
|------|-------------|-------|
| 3.1 | Define query templates | `victor/framework/graph_queries/templates.py` |
| 3.2 | Implement NL→Query translator | `victor/framework/graph_queries/translator.py` |
| 3.3 | Add query execution layer | `victor/framework/graph_queries/executor.py` |
| 3.4 | Integrate with Agent tools | `victor/tools/graph_query_tool.py` |

**Deliverables**:
- NL query to graph query translation
- Query execution with result formatting
- Agent tool for graph queries

### Phase 4: Integration & Optimization (Weeks 13-16)

**Goal**: Integrate with Victor workflows and optimize.

| Task | Description | Files |
|------|-------------|-------|
| 4.1 | Integrate with CodeSearchTool | `victor/tools/code_search.py` |
| 4.2 | Add graph context to init.md generation | `victor/tools/context_builder.py` |
| 4.3 | Optimize graph traversal performance | `victor/storage/graph/cache.py` |
| 4.4 | Add benchmarking & metrics | `victor/evaluation/graph_metrics.py` |

**Deliverables**:
- Graph-enhanced code search
- Graph context in init.md
- Performance benchmarks

### Phase 5: Advanced Features (Weeks 17-20)

**Goal**: Add requirement graph and graph-aware embeddings.

| Task | Description | Files |
|------|-------------|-------|
| 5.1 | Implement Requirement Graph | `victor/framework/requirement_graph.py` |
| 5.2 | Build Requirement→Symbol mapper | `victor/framework/requirement_mapper.py` |
| 5.3 | Implement graph-aware embeddings | `victor/processing/graph_embeddings.py` |
| 5.4 | Integrate with ProximaDB ORION | `victor_coding/codebase/embeddings/proximadb_provider.py` |

**Deliverables**:
- Requirement graph construction
- Graph-aware embedding generation
- ProximaDB integration for vector+graph

---

## 6. Use Cases

### 6.1 Repository-Level Code Completion

**Scenario**: Developer is writing code that calls repository APIs.

**Without Graph Enhancement**:
```python
# LLM only sees current file
def process_user(user_id):
    # LLM doesn't know about UserService.get_user()
    # May suggest incorrect API calls
```

**With CCG + Graph RAG**:
1. Build CCG for current cursor position
2. Traverse CALLS edges to find relevant APIs
3. Retrieve usage examples from similar contexts
4. Generate completion with repository-aware API usage

**Expected Improvement**: +30-40% correct API usage in completion

### 6.2 Bug Fixing with Multi-Hop Reasoning

**Scenario**: GitHub issue describes bug in nested function call chain.

**Without Graph Enhancement**:
```python
# Issue: "separability_matrix doesn't work for nested CompoundModels"
# Agent only searches for "separability_matrix" by name
# Misses the nested model handling code
```

**With Multi-Hop Traversal**:
1. Map issue requirement to relevant symbols
2. Traverse CALLS/REFERENCES edges 2-3 hops deep
3. Retrieve nested model handling code (implicit dependency)
4. Generate fix with full dependency context

**Expected Improvement**: +20-30% successful bug fixes

### 6.3 Impact Analysis for Refactoring

**Scenario**: Developer wants to change a function signature.

**With Graph Traversal**:
```python
# Query: "Who calls process_user and what will break?"
async def analyze_impact(symbol_name: str) -> ImpactTree:
    # 1. Reverse CALLS edges (find callers)
    callers = await graph_store.get_neighbors(
        symbol_name,
        edge_types=["CALLS"],
        direction="in",
        max_depth=3
    )
    # 2. Build impact tree
    # 3. Return visualization
```

**Output**:
```
process_user (victor/auth/service.py:42)
├── called_by: authenticate (victor/auth/handler.py:15)
│   └── called_by: login_route (victor/api/routes.py:89)
├── called_by: validate_session (victor/auth/middleware.py:23)
└── called_by: batch_import (victor/jobs/import.py:156)
```

### 6.4 Code Search with Structural Context

**Scenario**: "Find all classes that have a method named 'validate' and inherit from BaseModel"

**With Graph Query**:
```python
query = """
MATCH (c:Class)-[:INHERITS]->(BaseModel {name: 'BaseModel'})
MATCH (c)-[:HAS_METHOD]->(m:Method {name: 'validate'})
RETURN c.name, c.file, m.signature
"""
results = await graph_query.execute(query)
```

### 6.5 init.md Generation with Graph Context

**Scenario**: Agent needs rich context for a new task.

**Enhanced init.md Structure**:
```markdown
# Repository Context

## Project Structure
- Repository: victor-ai
- 1452 files, 23,312 symbols

## Relevant Symbols (via Graph Traversal)

### Direct Dependencies
1. `Agent.run()` (victor/framework/agent.py:156)
   - Calls: `Orchestrator.execute_turn()`
   - Docstring: "Execute single agent turn"

### Transitive Dependencies (2 hops)
1. `Orchestrator.execute_turn()` (victor/agent/orchestrator.py:89)
   - Calls: `ToolPipeline.execute()`, `ChatServiceAdapter.chat()`
2. `ToolPipeline.execute()` (victor/agent/tool_pipeline.py:45)
   - Uses: `ToolRegistry.get_tool()`

### Similar Code (Semantic + Structural)
1. `StreamAgent.run()` - Similar pattern for streaming
2. `WorkflowAgent.run()` - Similar pattern for workflows

## Data Flow Graph
```
[Agent.run] → [Orchestrator] → [ToolPipeline] → [Tools]
              ↓
         [ChatService] → [Provider]
```
```

---

## 7. Success Metrics

### 7.1 Quantitative Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Code completion accuracy (EM) | ~15% | +6% (GraphCoder improvement) | RepoEval benchmark |
| Bug fix success rate (SWE-bench) | ~30% | +12% (CGM improvement) | SWE-bench Lite |
| Context retrieval precision | ~60% | 85% | Custom benchmark |
| Multi-hop dependency recall | ~40% | 75% | Dependency coverage |
| Graph query translation accuracy | N/A | 80% | Query execution success |

### 7.2 Qualitative Metrics

- Developer satisfaction (survey)
- Reduced context switching (fewer file lookups)
- Faster onboarding to new codebases
- Better code suggestions in IDE

### 7.3 Performance Metrics

| Operation | Target Latency | Notes |
|-----------|----------------|-------|
| CCG construction (per file) | <100ms | Parallel processing |
| Multi-hop retrieval (3 hops) | <500ms | Cached where possible |
| Graph query execution | <200ms | Indexed queries |
| Subgraph extraction | <300ms | Incremental updates |

---

## 8. Open Questions & Risks

### 8.1 Technical Risks

| Risk | Mitigation |
|------|------------|
| CCG construction complexity | Start with Python only, extend gradually |
| Graph database scalability | Use SQLite for MVP, ProximaDB for production |
| LLM query translation accuracy | Template-based + fallback to keyword search |
| Multi-hop retrieval noise | Decay weighting + semantic re-ranking |

### 8.2 Open Questions

1. **Graph Database Choice**: Stick with SQLite or adopt Neo4j/ProximaDB?
   - **Recommendation**: SQLite for MVP, ProximaDB for production (vector+graph)

2. **CCG Granularity**: Statement-level vs. block-level?
   - **Recommendation**: Start with block-level (basic blocks), add statement-level incrementally

3. **Embedding Strategy**: Separate text+structure embeddings or fused?
   - **Recommendation**: Separate for flexibility, fusion at retrieval time

4. **Backward Compatibility**: How to handle existing projects without CCG?
   - **Recommendation**: Graceful degradation to symbol-level graph

---

## 9. References

1. **GraphCoder**: Liu et al., "Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval", arXiv:2406.07003
2. **CodexGraph**: Liu et al., "Bridging Large Language Models and Code Repositories via Code Graph Databases", arXiv:2408.03910
3. **GraphCodeAgent**: Li et al., "Dual Graph-Guided LLM Agent for Retrieval-Augmented Repo-Level Code Generation", arXiv:2504.10046
4. **CGM**: Tao et al., "Code Graph Model: A Graph-Integrated LLM for Repository-Level Software Engineering Tasks", arXiv:2505.16901
5. **GraphRAG Survey**: Peng et al., "Graph Retrieval-Augmented Generation: A Survey", arXiv:2408.08921

---

## Appendix A: Glossary

- **CCG**: Code Context Graph - Superimposition of CFG, CDG, and DDG
- **CFG**: Control Flow Graph - Represents execution order
- **CDG**: Control Dependence Graph - Represents control dependencies
- **DDG**: Data Dependence Graph - Represents variable flow
- **GraphRAG**: Graph Retrieval-Augmented Generation
- **Multi-hop reasoning**: Traversing multiple edges in a graph
- **Subgraph extraction**: Selecting relevant portion of a graph
- **Requirement Graph**: Graph mapping natural language requirements to code
- **Semantic+Structural Integration**: Combining text meaning with code structure

---

**Document Status**: Ready for Review
**Next Steps**: Team discussion, priority assignment, Phase 1 kickoff
