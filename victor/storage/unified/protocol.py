# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Unified Symbol Store Protocol - SOLID-based facade for graph + semantic search.

This module defines protocols and types for a unified storage layer that combines:
- Graph database (relationships, FTS search)
- Vector store (semantic similarity search)
- Unified ID scheme for correlation

Design Principles:
- DIP (Dependency Inversion): Protocol-based interfaces for all stores
- SRP (Single Responsibility): Each protocol handles one concern
- OCP (Open/Closed): New backends via factory, not code changes
- LSP (Liskov Substitution): Any implementation works via protocol

Backend Substitution Examples:
- SQLite + LanceDB (default, local)
- ProximaDB (planned single-engine backend)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, TypeVar

# =============================================================================
# UNIFIED ID SCHEME
# =============================================================================


class SymbolType(str, Enum):
    """Symbol type for unified IDs."""

    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"


@dataclass(frozen=True)
class UnifiedId:
    """Unified identifier for graph-embedding correlation.

    Format: {type}:{repo_relative_path}:{symbol_name}

    Examples:
        - symbol:victor/tools/graph_tool.py:find_symbols
        - file:victor/tools/graph_tool.py
        - class:victor/agent/orchestrator.py:AgentOrchestrator

    Using repo-relative paths avoids:
        - Cross-writing conflicts (multiple protocol.py files)
        - Portability issues with absolute paths
        - Hash collisions from filename-only IDs
    """

    type: str  # symbol, file, class, function, etc.
    path: str  # repo-relative path (e.g., victor/tools/graph_tool.py)
    name: str  # symbol name (empty for file type)

    def __str__(self) -> str:
        if self.name:
            return f"{self.type}:{self.path}:{self.name}"
        return f"{self.type}:{self.path}"

    @classmethod
    def from_string(cls, id_str: str) -> "UnifiedId":
        """Parse unified ID from string."""
        parts = id_str.split(":", 2)  # Max 2 splits for type:path:name
        if len(parts) == 3:
            return cls(type=parts[0], path=parts[1], name=parts[2])
        elif len(parts) == 2:
            return cls(type=parts[0], path=parts[1], name="")
        else:
            raise ValueError(f"Invalid unified ID format: {id_str}")

    @classmethod
    def for_symbol(cls, rel_path: str, symbol_name: str) -> "UnifiedId":
        """Create unified ID for a symbol."""
        return cls(type="symbol", path=rel_path, name=symbol_name)

    @classmethod
    def for_file(cls, rel_path: str) -> "UnifiedId":
        """Create unified ID for a file."""
        return cls(type="file", path=rel_path, name="")

    @classmethod
    def for_class(cls, rel_path: str, class_name: str) -> "UnifiedId":
        """Create unified ID for a class."""
        return cls(type="class", path=rel_path, name=class_name)

    @classmethod
    def for_function(cls, rel_path: str, func_name: str) -> "UnifiedId":
        """Create unified ID for a function."""
        return cls(type="function", path=rel_path, name=func_name)


# =============================================================================
# UNIFIED SYMBOL NODE - Combines Graph + Embedding Data
# =============================================================================


@dataclass
class UnifiedSymbol:
    """Symbol with both graph and embedding data.

    This is the primary data structure returned by UnifiedSymbolStoreProtocol.
    It contains:
    - Identity: unified_id, name, type
    - Location: file path, line numbers
    - Content: signature, docstring
    - Graph: relationships (callers, callees, etc.)
    - Semantic: embedding vector reference
    """

    # Identity
    unified_id: str  # UnifiedId as string for serialization
    name: str
    type: str  # function, class, method, file, etc.

    # Location (repo-relative to avoid conflicts)
    file_path: str  # repo-relative path
    line: Optional[int] = None
    end_line: Optional[int] = None
    lang: Optional[str] = None

    # Content (for FTS and context)
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_id: Optional[str] = None  # for nested symbols

    # Graph relationships (populated on demand)
    callers: List[str] = field(default_factory=list)  # unified_ids of callers
    callees: List[str] = field(default_factory=list)  # unified_ids of callees
    inherits: List[str] = field(default_factory=list)  # base classes
    implementors: List[str] = field(default_factory=list)  # classes implementing this

    # Semantic scores (populated during search)
    semantic_score: Optional[float] = None  # 0-1, similarity to query
    graph_score: Optional[float] = None  # PageRank or centrality score
    combined_score: Optional[float] = None  # weighted combination

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedEdge:
    """Edge between unified symbols."""

    src_id: str  # UnifiedId as string
    dst_id: str  # UnifiedId as string
    type: str  # CALLS, REFERENCES, INHERITS, IMPLEMENTS, CONTAINS
    weight: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SEARCH PARAMETERS AND RESULTS
# =============================================================================


class SearchMode(str, Enum):
    """Search mode for unified queries."""

    KEYWORD = "keyword"  # FTS only
    SEMANTIC = "semantic"  # Vector similarity only
    HYBRID = "hybrid"  # Combined (default)
    GRAPH = "graph"  # Graph traversal only


@dataclass
class SearchParams:
    """Parameters for unified search."""

    query: str
    mode: SearchMode = SearchMode.HYBRID
    limit: int = 20

    # Filters
    file_patterns: Optional[List[str]] = None  # glob patterns
    symbol_types: Optional[List[str]] = None  # function, class, etc.

    # Hybrid search weights
    semantic_weight: float = 0.7  # weight for semantic score
    graph_weight: float = 0.3  # weight for graph importance

    # Graph options
    include_neighbors: bool = False  # include callers/callees in results
    max_depth: int = 1  # for graph traversal

    # Semantic options
    similarity_threshold: float = 0.25  # min similarity score


@dataclass
class UnifiedSearchResult:
    """Result from unified search with multi-signal scoring.

    Combines semantic, keyword, and graph-based search signals.
    Renamed from SearchResult to be semantically distinct from other search types.
    """

    symbol: UnifiedSymbol
    score: float  # combined relevance score
    match_type: str  # keyword, semantic, graph, hybrid

    # Score breakdown
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    graph_score: Optional[float] = None

    # Context
    matched_content: Optional[str] = None  # what matched the query


# Backward compatibility alias
# =============================================================================
# PROTOCOLS - Dependency Inversion Principle
# =============================================================================


class VectorStoreProtocol(Protocol):
    """Protocol for vector stores (semantic search).

    Implementations:
    - LanceDB: Default, disk-based, scales to billions
    - pgvector: PostgreSQL extension
    - FAISS: Meta's vector library
    """

    async def initialize(self) -> None:
        """Initialize the store."""
        ...

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        ...

    async def upsert(
        self,
        doc_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert a single vector. No content storage - only ID and vector."""
        ...

    async def upsert_batch(
        self,
        items: List[Tuple[str, List[float], Optional[Dict[str, Any]]]],
    ) -> None:
        """Batch upsert vectors. Each tuple is (doc_id, vector, metadata)."""
        ...

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors. Returns list of (doc_id, score)."""
        ...

    async def delete(self, doc_id: str) -> None:
        """Delete a vector by ID."""
        ...

    async def delete_by_prefix(self, prefix: str) -> int:
        """Delete all vectors whose ID starts with prefix. Returns count."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...


class GraphStoreProtocol(Protocol):
    """Protocol for graph stores (relationships + FTS).

    Implementations:
    - SQLite: Default, embedded, WAL mode
    - PostgreSQL: Cloud-ready, with FTS
    - DuckDB: Analytical queries
    - Neo4j: Native graph (optional)
    """

    async def upsert_nodes(self, nodes: Iterable["GraphNode"]) -> None:
        """Upsert nodes."""
        ...

    async def upsert_edges(self, edges: Iterable["GraphEdge"]) -> None:
        """Upsert edges."""
        ...

    async def get_node(self, node_id: str) -> Optional["GraphNode"]:
        """Get single node by ID."""
        ...

    async def get_nodes_by_file(self, file_path: str) -> List["GraphNode"]:
        """Get all nodes in a file."""
        ...

    async def find_nodes(
        self,
        name: Optional[str] = None,
        type: Optional[str] = None,
        file: Optional[str] = None,
    ) -> List["GraphNode"]:
        """Find nodes by criteria."""
        ...

    async def search_fts(
        self,
        query: str,
        limit: int = 20,
        symbol_types: Optional[List[str]] = None,
    ) -> List["GraphNode"]:
        """Full-text search on names, signatures, docstrings."""
        ...

    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        direction: str = "both",  # in, out, both
        max_depth: int = 1,
    ) -> List["GraphEdge"]:
        """Get edges connected to node."""
        ...

    async def delete_by_file(self, file_path: str) -> None:
        """Delete all nodes/edges for a file."""
        ...

    async def delete_all(self) -> None:
        """Clear entire graph."""
        ...

    async def stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        ...


class UnifiedSymbolStoreProtocol(Protocol):
    """Unified store combining graph + vector for code symbols.

    This is the FACADE that tools should use. It provides:
    - Unified ID management (graph-embedding correlation)
    - Hybrid search (keyword + semantic + graph ranking)
    - Symbol indexing (both graph and embeddings)
    - Relationship queries with semantic ranking

    Substitution Options:
    - SQLite + LanceDB (default, local)
    - PostgreSQL + pgvector (single store, cloud)
    - LanceDB only (vector + metadata)
    - DuckDB + FAISS (analytical)
    """

    # Initialization
    async def initialize(self, repo_root: Path) -> None:
        """Initialize stores for a repository."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...

    # Unified ID helpers
    def make_symbol_id(self, rel_path: str, symbol_name: str) -> str:
        """Create unified ID for symbol."""
        ...

    def make_file_id(self, rel_path: str) -> str:
        """Create unified ID for file."""
        ...

    def parse_id(self, unified_id: str) -> UnifiedId:
        """Parse unified ID string."""
        ...

    # Indexing (writes to both graph AND vector store)
    async def index_symbol(
        self,
        symbol: UnifiedSymbol,
        embedding_text: str,
    ) -> None:
        """Index a single symbol (graph + embedding)."""
        ...

    async def index_symbols_batch(
        self,
        symbols: List[Tuple[UnifiedSymbol, str]],  # (symbol, embedding_text)
        batch_size: int = 500,
    ) -> int:
        """Batch index symbols. Returns count indexed."""
        ...

    async def index_edge(self, edge: UnifiedEdge) -> None:
        """Index an edge."""
        ...

    async def index_edges_batch(self, edges: List[UnifiedEdge]) -> int:
        """Batch index edges. Returns count indexed."""
        ...

    # Unified Search (hybrid by default)
    async def search(self, params: SearchParams) -> List[UnifiedSearchResult]:
        """Unified search combining keyword, semantic, and graph.

        Algorithm:
        1. If mode is HYBRID or SEMANTIC: vector search for similar symbols
        2. If mode is HYBRID or KEYWORD: FTS search in graph
        3. Optionally: weight results by graph importance (PageRank)
        4. Merge and rank results by combined score
        5. If include_neighbors: add caller/callee context
        """
        ...

    async def search_semantic(
        self,
        query: str,
        limit: int = 20,
        threshold: float = 0.25,
    ) -> List[UnifiedSearchResult]:
        """Pure semantic search (vector similarity)."""
        ...

    async def search_keyword(
        self,
        query: str,
        limit: int = 20,
        symbol_types: Optional[List[str]] = None,
    ) -> List[UnifiedSearchResult]:
        """Pure keyword search (FTS)."""
        ...

    # Graph queries
    async def get_symbol(self, unified_id: str) -> Optional[UnifiedSymbol]:
        """Get symbol by unified ID."""
        ...

    async def get_symbols_in_file(self, rel_path: str) -> List[UnifiedSymbol]:
        """Get all symbols in a file."""
        ...

    async def get_callers(
        self,
        unified_id: str,
        max_depth: int = 1,
    ) -> List[UnifiedSymbol]:
        """Get functions that call this symbol."""
        ...

    async def get_callees(
        self,
        unified_id: str,
        max_depth: int = 1,
    ) -> List[UnifiedSymbol]:
        """Get functions called by this symbol."""
        ...

    async def get_related(
        self,
        unified_id: str,
        edge_types: Optional[List[str]] = None,
    ) -> List[Tuple[UnifiedSymbol, str]]:
        """Get related symbols with relationship type."""
        ...

    # Semantic + Graph combined queries
    async def find_similar_symbols(
        self,
        unified_id: str,
        limit: int = 10,
    ) -> List[UnifiedSearchResult]:
        """Find semantically similar symbols to given symbol."""
        ...

    async def semantic_blast_radius(
        self,
        unified_id: str,
        similarity_threshold: float = 0.5,
    ) -> List[UnifiedSearchResult]:
        """Find symbols that might be affected by changes (graph + semantic)."""
        ...

    # Maintenance
    async def delete_file(self, rel_path: str) -> None:
        """Delete all symbols for a file (graph + embeddings)."""
        ...

    async def delete_all(self) -> None:
        """Clear entire store."""
        ...

    async def stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        ...


# =============================================================================
# GRAPH NODE/EDGE (Re-export for convenience)
# =============================================================================

# Import from existing protocol for backward compatibility
from victor.storage.graph.protocol import GraphNode, GraphEdge

__all__ = [
    # Unified ID
    "UnifiedId",
    "SymbolType",
    # Unified data structures
    "UnifiedSymbol",
    "UnifiedEdge",
    # Search
    "SearchMode",
    "SearchParams",
    "UnifiedSearchResult",
    # Protocols
    "VectorStoreProtocol",
    "GraphStoreProtocol",
    "UnifiedSymbolStoreProtocol",
    # Re-exports
    "GraphNode",
    "GraphEdge",
]
