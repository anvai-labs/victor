"""Codebase search and semantic configuration."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class GraphSettings(BaseModel):
    """Configuration for graph-based code intelligence (Phase 14).

    Attributes:
        enable_ccg: Whether to build Code Context Graph (CFG/CDG/DDG)
        ccg_languages: Languages for which to build CCG
        enable_graph_rag: Whether to enable Graph RAG retrieval
        rag_seed_count: Number of seed nodes for Graph RAG
        rag_max_hops: Maximum hops for multi-hop retrieval
        enable_subgraph_cache: Whether to cache subgraphs for fast retrieval
        subgraph_cache_ttl: TTL for subgraph cache in seconds
        enable_graph_query_tool: Whether to enable graph query tools
        enable_impact_analysis: Whether to enable impact analysis tool
        graph_query_mode: Default graph query mode (semantic, structural, hybrid)
        enable_graph_context_in_init: Whether to add graph context to init.md
        init_max_symbols: Maximum symbols to include in graph context section
        enable_query_cache: Whether to cache graph query results (PH4-005)
        query_cache_max_entries: Maximum number of query results to cache (PH4-005)
        query_cache_ttl: TTL for query cache entries in seconds (PH4-005)
        query_cache_normalize: Whether to normalize queries for better cache hits (PH4-005)
    """

    enable_ccg: bool = True
    ccg_languages: List[str] = Field(
        default_factory=lambda: ["python", "javascript", "typescript", "go", "rust"]
    )
    enable_graph_rag: bool = True
    rag_seed_count: int = 5
    rag_max_hops: int = 2
    enable_subgraph_cache: bool = True
    subgraph_cache_ttl: int = 3600
    enable_graph_query_tool: bool = True
    enable_impact_analysis: bool = True
    graph_query_mode: str = "semantic"
    enable_graph_context_in_init: bool = True
    init_max_symbols: int = 50
    # PH4-005: Graph query cache settings
    enable_query_cache: bool = True
    query_cache_max_entries: int = 100
    query_cache_ttl: int = 3600
    query_cache_normalize: bool = True
    # PH4-006: Lazy loading settings
    enable_lazy_loading: bool = True
    lazy_load_batch_size: int = 100
    lazy_load_neighbor_batch_size: int = 50
    lazy_load_prefetch_enabled: bool = True
    lazy_load_prefetch_count: int = 2
    # PH4-007: Parallel traversal settings
    enable_parallel_traversal: bool = True
    parallel_max_workers: int = 4
    parallel_min_batch_size: int = 3
    parallel_neighbor_threshold: int = 5


class SearchSettings(BaseModel):
    """Codebase search and semantic configuration."""

    unified_embedding_model: str = "BAAI/bge-small-en-v1.5"
    codebase_vector_store: str = "lancedb"
    codebase_embedding_provider: str = "sentence-transformers"
    codebase_embedding_model: str = "BAAI/bge-small-en-v1.5"
    codebase_persist_directory: Optional[str] = None
    codebase_dimension: int = 384
    codebase_batch_size: int = 32
    codebase_structural_indexing_enabled: bool = False
    codebase_chunking_strategy: str = "victor_codegraph"
    codebase_chunk_size: int = 500
    codebase_chunk_overlap: int = 50
    codebase_embedding_extra_config: dict[str, Any] = Field(default_factory=dict)
    codebase_graph_store: str = "sqlite"
    codebase_graph_path: Optional[str] = None
    codebase_graph_writer_mode: str = "off"  # off (default) or compatibility (deprecated)
    core_readonly_tools: Optional[List[str]] = None
    semantic_similarity_threshold: float = 0.25
    semantic_query_expansion_enabled: bool = True
    semantic_max_query_expansions: int = 5
    enable_hybrid_search: bool = False
    hybrid_search_semantic_weight: float = 0.6
    hybrid_search_keyword_weight: float = 0.4
    enable_semantic_threshold_rl_learning: bool = False
    semantic_threshold_overrides: dict = Field(default_factory=dict)
    # RL mode: "full" (write all), "selective" (skip unchanged), "none" (read-only)
    rl_mode: str = "selective"
    # Graph-based code intelligence (Phase 14)
    graph: GraphSettings = Field(default_factory=GraphSettings)
