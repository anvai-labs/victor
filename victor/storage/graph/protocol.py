# Graph store protocol for per-repo symbol graphs
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterable, List, Literal, Protocol

GraphTraversalDirection = Literal["out", "in", "both"]


@dataclass
class GraphNode:
    """Represents a code symbol with metadata (body read from file via line numbers).

    Extended for v5 schema to support Code Context Graph (CCG) with:
    - Statement-level granularity for CFG/CDG/DDG edges
    - Hierarchical scope tracking
    - Requirement node linking
    - Visibility annotations
    """

    node_id: str  # stable id, e.g., symbol hash
    type: str  # Extended: function, class, file, module, statement, requirement, chunk
    name: str
    file: str
    line: int | None = None
    end_line: int | None = None  # end line - use with line to read body from file
    lang: str | None = None
    signature: str | None = None  # function/method signature
    docstring: str | None = None  # extracted docstring
    parent_id: str | None = None  # for nested symbols (methods in classes)
    embedding_ref: str | None = None  # key to vector store entry
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ===========================================
    # v5: CCG and enhanced graph fields
    # ===========================================
    ast_kind: str | None = (
        None  # Tree-sitter node kind (e.g., "function_definition", "if_statement")
    )
    scope_id: str | None = None  # Hierarchical scope tracking for nested contexts
    statement_type: str | None = None  # Statement type: assignment, call, return, condition, etc.
    requirement_id: str | None = (
        None  # Link to requirement node (for requirement-graph integration)
    )
    visibility: str | None = None  # Visibility: public, private, protected, package-private


@dataclass
class GraphEdge:
    """Directed edge between nodes.

    Extended edge types (v5):
    - Legacy: CALLS, REFERENCES, CONTAINS, INHERITS, IMPLEMENTS, IMPORTS, INSTANTIATES
    - CFG: CFG_SUCCESSOR, CFG_TRUE, CFG_FALSE, CFG_CASE, CFG_DEFAULT, CFG_LOOP_ENTRY, etc.
    - CDG: CDG, CDG_TRUE, CDG_FALSE, CDG_LOOP, CDG_CASE
    - DDG: DDG_DEF_USE, DDG_RAW, DDG_WAR, DDG_WAW
    - Requirement: SATISFIES, TESTS, DERIVES_FROM, REFINES, CONTRADICTS, COVERS
    - Semantic: SEMANTIC_SIMILAR, STRUCTURAL_SIMILAR, FUNCTIONAL_SIMILAR

    See victor.storage.graph.edge_types for complete definitions.
    """

    src: str
    dst: str
    type: str  # e.g., CALLS, REFERENCES, CONTAINS, INHERITS, CFG_SUCCESSOR, DDG_DEF_USE
    weight: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequirementNode:
    """Represents a requirement node linked to code symbols.

    Used for requirement-to-code traceability (GraphCodeAgent pattern).
    Requirements can be linked to code via SATISFIES, TESTS, and REFINES edges.
    """

    requirement_id: str  # Unique identifier (also used as node_id in graph_node)
    type: str  # requirement, feature, bug, task, user_story
    source: str | None = None  # Source system: github_issue, jira, slack, etc.
    title: str = ""
    description: str | None = None
    priority: float = 0.5  # 0.0 to 1.0
    status: str = "open"  # open, in_progress, resolved, closed
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subgraph:
    """A cached subgraph for efficient multi-hop retrieval.

    Subgraphs are pre-computed neighborhoods around anchor nodes,
    used for fast context retrieval in Graph RAG.
    """

    subgraph_id: str
    anchor_node_id: str
    radius: int  # Hop count for neighborhood
    edge_types: List[str]  # Edge types included in traversal
    node_ids: List[str]  # All nodes in the subgraph
    edges: List[GraphEdge]  # All edges in the subgraph
    node_count: int = 0
    computed_at: str | None = None


@dataclass
class GraphQueryResult:
    """Result from a graph query operation."""

    nodes: List[GraphNode]
    edges: List[GraphEdge]
    subgraphs: List[Subgraph] = field(default_factory=list)
    query: str = ""
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary representation of the result
        """
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "subgraph_count": len(self.subgraphs),
            "query": self.query,
            "execution_time_ms": self.execution_time_ms,
        }


class GraphStoreProtocol(Protocol):
    """Interface for pluggable graph stores."""

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None: ...

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None: ...

    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Iterable[str] | None = None,
        *,
        direction: GraphTraversalDirection = "both",
        max_depth: int = 1,
    ) -> List[GraphEdge]: ...

    async def find_nodes(
        self,
        *,
        name: str | None = None,
        type: str | None = None,
        file: str | None = None,
    ) -> List[GraphNode]: ...

    async def search_symbols(
        self, query: str, *, limit: int = 20, symbol_types: Iterable[str] | None = None
    ) -> List[GraphNode]:
        """Full-text search across symbol names, signatures, bodies, and docstrings."""
        ...

    async def get_node_by_id(self, node_id: str) -> GraphNode | None:
        """Get a single node by its ID."""
        ...

    async def get_all_nodes(self) -> List[GraphNode]:
        """Get all nodes in the graph (bulk retrieval for analytics)."""
        ...

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        """Get all symbols in a specific file."""
        ...

    async def update_file_mtime(
        self, file: str, mtime: float, content_hash: str | None = None
    ) -> None:
        """Record file modification time (and optional content hash) for staleness.

        ``content_hash`` follows the victor-codegraph manifest contract
        (sha-256 of the decoded source text). ``None`` clears any stored hash —
        an unknown hash must never masquerade as a verified one.
        """
        ...

    async def get_stale_files(self, file_mtimes: Dict[str, float]) -> List[str]:
        """Get files that have changed since last index."""
        ...

    async def get_file_hashes(self, files: List[str]) -> Dict[str, str]:
        """Stored content hashes for the given files (missing/None entries omitted)."""
        ...

    async def delete_by_file(self, file: str) -> None:
        """Delete all nodes, edges, and embeddings for a specific file.

        Implementations should also clean up associated embeddings from the
        vector store to prevent orphaned data when files are deleted or
        re-indexed incrementally.

        Args:
            file: File path to delete
        """
        ...

    async def delete_by_repo(self) -> None:
        """Clear current repo graph (per-repo store)."""
        ...

    async def stats(self) -> Dict[str, Any]: ...

    async def get_all_edges(self) -> List[GraphEdge]:
        """Get all edges in the graph (bulk retrieval for loading into memory)."""
        ...

    # ===========================================
    # v5: CCG and Graph RAG methods
    # ===========================================

    async def get_nodes_by_statement_type(
        self, statement_type: str, *, file: str | None = None
    ) -> List[GraphNode]:
        """Get nodes by statement type (assignment, call, return, condition, etc.)."""
        ...

    async def get_nodes_by_requirement(self, requirement_id: str) -> List[GraphNode]:
        """Get all nodes linked to a specific requirement."""
        ...

    async def get_subgraph(
        self,
        anchor_node_id: str,
        radius: int = 2,
        edge_types: Iterable[str] | None = None,
    ) -> Subgraph:
        """Get a cached or compute a new subgraph around an anchor node."""
        ...

    async def cache_subgraph(self, subgraph: Subgraph) -> None:
        """Cache a subgraph for fast retrieval."""
        ...

    async def invalidate_subgraph(self, subgraph_id: str) -> None:
        """Invalidate a cached subgraph (after code changes)."""
        ...

    async def get_nodes_by_scope(self, scope_id: str) -> List[GraphNode]:
        """Get all nodes within a specific scope (hierarchical context)."""
        ...

    async def multi_hop_traverse(
        self,
        start_node_ids: List[str],
        max_hops: int = 2,
        edge_types: Iterable[str] | None = None,
        max_nodes: int = 100,
    ) -> GraphQueryResult:
        """Multi-hop graph traversal for context retrieval."""
        ...

    # ===========================================
    # v5: Lazy loading methods (PH4-006)
    # ===========================================

    async def iter_nodes(
        self,
        *,
        batch_size: int = 100,
        name: str | None = None,
        type: str | None = None,
        file: str | None = None,
    ) -> AsyncIterator[List[GraphNode]]:
        """Iterate over nodes in batches for memory-efficient processing.

        Args:
            batch_size: Number of nodes to yield per batch
            name: Optional filter by name
            type: Optional filter by type
            file: Optional filter by file

        Yields:
            Batches of GraphNode objects
        """
        ...

    async def iter_edges(
        self,
        *,
        batch_size: int = 100,
        edge_types: Iterable[str] | None = None,
    ) -> AsyncIterator[List[GraphEdge]]:
        """Iterate over edges in batches for memory-efficient processing.

        Args:
            batch_size: Number of edges to yield per batch
            edge_types: Optional filter by edge types

        Yields:
            Batches of GraphEdge objects
        """
        ...

    async def iter_neighbors(
        self,
        node_id: str,
        *,
        batch_size: int = 50,
        edge_types: Iterable[str] | None = None,
        direction: GraphTraversalDirection = "out",
    ) -> AsyncIterator[List[GraphEdge]]:
        """Iterate over neighbors in batches for memory-efficient traversal.

        Args:
            node_id: Starting node ID
            batch_size: Number of edges to yield per batch
            edge_types: Optional filter by edge types
            direction: Traversal direction (out, in, both)

        Yields:
            Batches of GraphEdge objects
        """
        ...

    # ===========================================
    # v5: Parallel traversal methods (PH4-007)
    # ===========================================

    async def get_neighbors_batch(
        self,
        node_ids: List[str],
        *,
        edge_types: Iterable[str] | None = None,
        direction: GraphTraversalDirection = "out",
    ) -> Dict[str, List[GraphEdge]]:
        """Get neighbors for multiple nodes in parallel (PH4-007).

        Args:
            node_ids: List of node IDs to get neighbors for
            edge_types: Optional filter by edge types
            direction: Traversal direction

        Returns:
            Dict mapping node_id to list of neighboring edges
        """
        ...

    async def multi_hop_traverse_parallel(
        self,
        start_node_ids: List[str],
        max_hops: int = 2,
        edge_types: Iterable[str] | None = None,
        max_nodes: int = 100,
        max_workers: int = 4,
    ) -> GraphQueryResult:
        """Parallel multi-hop traversal from multiple start nodes (PH4-007).

        Args:
            start_node_ids: Multiple starting node IDs
            max_hops: Maximum hop distance
            edge_types: Optional filter by edge types
            max_nodes: Maximum total nodes to return
            max_workers: Maximum parallel workers

        Returns:
            GraphQueryResult with traversed nodes and edges
        """
        ...
