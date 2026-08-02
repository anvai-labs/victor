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

"""Configuration classes for Graph RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class SubgraphConfig:
    """Configuration for subgraph caching and retrieval.

    Attributes:
        enabled: Whether to enable subgraph caching
        default_radius: Default hop count for subgraph neighborhoods
        max_nodes: Maximum nodes per cached subgraph
        ttl_seconds: Cache TTL in seconds
        edge_types: Default edge types to include in subgraphs
    """

    enabled: bool = True
    default_radius: int = 2
    max_nodes: int = 100
    ttl_seconds: int = 3600
    edge_types: Set[str] = field(
        default_factory=lambda: {
            "CALLS",
            "REFERENCES",
            "CONTAINS",
            "CFG_SUCCESSOR",
            "DDG_DEF_USE",
        }
    )


@dataclass
class GraphIndexConfig:
    """Configuration for graph indexing pipeline.

    Attributes:
        root_path: Root path of the repository to index
        enable_ccg: Whether to build Code Context Graph (CFG/CDG/DDG)
        enable_embeddings: Whether to generate embeddings for nodes
        enable_subgraph_cache: Whether to pre-compute subgraph cache
        incremental: Whether to reindex only changed/deleted files using mtimes
        chunk_size: Number of files to process per batch
        max_file_size_bytes: Maximum file size to process
        exclude_patterns: Glob patterns for files/directories to exclude
        include_patterns: Glob patterns for files to include (if set, excludes others)
        subgraph_config: Configuration for subgraph caching
        embedding_neighborhood_radius: Radius for graph neighborhood in embeddings
        embedding_max_neighbors: Maximum neighbors to include in embedding context
        embedding_batch_size: Batch size for embedding generation
        respect_gitignore: Whether to parse and include .gitignore patterns
        detect_languages: Whether to detect languages and add their build dirs
        enable_module_metrics: Whether to refresh graph_module_metric after graph changes
    """

    root_path: Path
    enable_ccg: bool = True
    enable_embeddings: bool = True
    enable_subgraph_cache: bool = True
    incremental: bool = True
    chunk_size: int = 50  # Will be adjusted in __post_init__ based on mode
    max_file_size_bytes: int = 1_000_000  # 1MB
    exclude_patterns: List[str] = field(
        default_factory=list
    )  # Initialized via get_exclusion_patterns()
    include_patterns: List[str] = field(default_factory=list)
    subgraph_config: SubgraphConfig = field(default_factory=SubgraphConfig)
    embedding_neighborhood_radius: int = 2
    embedding_max_neighbors: int = 50
    embedding_batch_size: int = 100
    # Structural mixing weight for PERSISTED vectors. Keep 0.0 so stored vectors
    # stay comparable to plain-text query embeddings at search time; structural
    # signals belong in ranking (hop-distance decay), not in the stored vector.
    embedding_structural_weight: float = 0.0
    respect_gitignore: bool = True
    detect_languages: bool = True
    enable_module_metrics: bool = True
    parse_workers: int = 0  # Thread-pool workers for parallel AST parsing (0 = cpu_count)
    write_batch_size: int = 20  # Files per mini-batch flush in the streaming pipeline
    queue_maxsize: int = 100  # Bounded queue depth between producers and consumer

    def __post_init__(self):
        """Initialize exclude patterns and adaptive chunk size after instance creation."""
        # Adaptive chunk size: larger batches for force mode, smaller for incremental
        # Force mode benefits from bulk load efficiency (DELETE + INSERT vs UPSERT)
        # Incremental mode uses smaller batches to minimize stale data window
        if self.chunk_size == 50:  # Only adjust if not explicitly set
            self.chunk_size = 200 if not self.incremental else 50

        if not self.exclude_patterns:
            from victor.core.graph_rag.exclude_patterns import get_exclusion_patterns

            self.exclude_patterns = get_exclusion_patterns(
                root_path=self.root_path,
                respect_gitignore=self.respect_gitignore,
                detect_languages=self.detect_languages,
            )


@dataclass
class RetrievalConfig:
    """Configuration for multi-hop graph retrieval.

    Attributes:
        seed_count: Number of seed nodes from dense retrieval
        max_hops: Maximum number of hops for graph traversal
        edge_types: Edge types to traverse (None = all)
        top_k: Maximum number of results to return
        centrality_weight: Weight for node centrality in ranking (0-1)
        size_penalty_weight: Penalty for large subgraphs (0-1)
        semantic_threshold: Minimum semantic similarity for seed nodes
        enable_reranking: Whether to re-rank results after traversal
        max_context_tokens: Maximum tokens in retrieved context
        mode: Seed strategy this retrieval ran under (semantic/structural/
            hybrid) — part of the query-cache key so mode results don't collide
    """

    seed_count: int = 5
    max_hops: int = 2
    edge_types: Optional[Set[str]] = None
    top_k: int = 10
    centrality_weight: float = 0.2
    size_penalty_weight: float = 0.01
    semantic_threshold: float = 0.3
    enable_reranking: bool = True
    max_context_tokens: int = 8000
    mode: str = "semantic"


@dataclass
class PromptConfig:
    """Configuration for graph-aware prompt building.

    Attributes:
        format_style: Prompt format style (hierarchical, flat, compact)
        include_line_numbers: Whether to include line numbers
        include_file_paths: Whether to include file paths
        include_edge_types: Whether to include edge type annotations
        max_symbols_per_section: Maximum symbols per section
        truncate_long_contexts: Whether to truncate long contexts
        context_separator: Separator between context sections
        hierarchy_indicators: Whether to show hierarchy (indentation)
    """

    format_style: str = "hierarchical"  # hierarchical, flat, compact
    include_line_numbers: bool = True
    include_file_paths: bool = True
    include_edge_types: bool = False
    max_symbols_per_section: int = 20
    truncate_long_contexts: bool = True
    context_separator: str = "\n---\n"
    hierarchy_indicators: bool = True

    def get_format_template(self) -> str:
        """Get the prompt format template based on style.

        Returns:
            Format template string
        """
        templates = {
            "hierarchical": """
## Relevant Code Context

### Direct Matches
{direct_matches}

### Impact & Potential Regressions
{impact_analysis}

### Related Symbols
{related_symbols}

### Data Flow
{data_flow}

### Control Flow
{control_flow}
""",
            "flat": """
## Relevant Code

{all_symbols}
""",
            "compact": """
## Code Context

{compact_context}
""",
        }
        return templates.get(self.format_style, templates["hierarchical"])


@dataclass
class GraphRagConfig:
    """Combined configuration for the entire Graph RAG pipeline.

    Attributes:
        index: Graph indexing configuration
        retrieval: Retrieval configuration
        prompt: Prompt building configuration
        subgraph: Subgraph caching configuration
    """

    index: GraphIndexConfig
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    subgraph: SubgraphConfig = field(default_factory=SubgraphConfig)

    @classmethod
    def from_settings(cls, root_path: Path, settings: Any) -> "GraphRagConfig":
        """Create GraphRagConfig from Victor settings.

        Args:
            root_path: Repository root path
            settings: Victor settings object

        Returns:
            GraphRagConfig instance
        """
        # Extract graph settings if available
        enable_ccg = getattr(settings, "enable_ccg", True)
        enable_graph_rag = getattr(settings, "enable_graph_rag", True)

        index_config = GraphIndexConfig(
            root_path=root_path,
            enable_ccg=enable_ccg,
            enable_embeddings=enable_graph_rag,
        )

        return cls(index=index_config)


__all__ = [
    "SubgraphConfig",
    "GraphIndexConfig",
    "RetrievalConfig",
    "PromptConfig",
    "GraphRagConfig",
]
