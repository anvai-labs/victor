"""Embedding and semantic search configuration.

This module contains settings for:
- Unified embedding model configuration
- Tool selection embedding settings
- Codebase semantic search
- Hybrid search configuration
"""

from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class EmbeddingSettings(BaseModel):
    """Embedding and semantic search settings.

    Controls embedding models for tool selection and codebase search,
    semantic similarity thresholds, and hybrid search configuration.
    """

    # ==========================================================================
    # Unified Embedding Model
    # ==========================================================================
    # Shared embedding model for both tool selection and codebase search.
    # Model: BAAI/bge-small-en-v1.5 (130MB, 384-dim, ~6ms)
    # - MTEB score: 62.2 (vs 58.8 for all-MiniLM-L6-v2)
    # - Excellent for code search (trained on code-related tasks)
    # - CPU-optimized, works great on consumer-grade hardware
    # - Native sentence-transformers support (no API needed)
    unified_embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Tool Selection Strategy
    use_semantic_tool_selection: bool = True  # Use embeddings instead of keywords (DEFAULT)
    preload_embeddings: bool = False  # Defer embedding model load to first semantic query

    # ==========================================================================
    # Embedding Provider Configuration
    # ==========================================================================
    # Provider for embedding generation.
    # Options: sentence-transformers (local), ollama, vllm, lmstudio
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = unified_embedding_model  # Shared with codebase search

    # ==========================================================================
    # Codebase Semantic Search (Air-gapped by Default)
    # ==========================================================================
    codebase_vector_store: str = "lancedb"  # lancedb (recommended)
    codebase_embedding_provider: str = "sentence-transformers"  # Local, offline, fast
    codebase_embedding_model: str = unified_embedding_model  # Shared with tool selection
    codebase_persist_directory: Optional[str] = None  # Default: ~/.victor/embeddings/codebase
    codebase_dimension: int = 384  # Embedding dimension
    codebase_batch_size: int = 32  # Batch size for embedding generation
    codebase_structural_indexing_enabled: bool = False
    codebase_chunking_strategy: str = "victor_codegraph"
    codebase_chunk_size: int = 500
    codebase_chunk_overlap: int = 50
    codebase_embedding_extra_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific embedding/vector-store options forwarded to code search backends",
    )
    codebase_graph_store: str = "sqlite"  # Graph backend (sqlite default)
    codebase_graph_path: Optional[str] = None  # Optional explicit graph db path
    codebase_graph_writer_mode: str = "off"  # "off" (default) or "compatibility"
    core_readonly_tools: Optional[List[str]] = None  # Override/extend curated read-only tool set

    # ==========================================================================
    # Semantic Search Quality Improvements
    # ==========================================================================
    # Minimum similarity score for semantic search results [0.1-0.9].
    # Lowered from 0.5 to 0.25 for better recall on technical queries.
    semantic_similarity_threshold: float = 0.25

    # Query expansion with synonyms/related terms for better recall
    semantic_query_expansion_enabled: bool = True
    semantic_max_query_expansions: int = 5  # Max query variations to try (including original)

    # ==========================================================================
    # Hybrid Search (Semantic + Keyword with RRF)
    # ==========================================================================
    # Enable hybrid search combining semantic + keyword with Rank-Based Fusion (RRF).
    # Provides both relevance (semantic) and precision (keyword).
    enable_hybrid_search: bool = False
    hybrid_search_semantic_weight: float = 0.6  # Weight for semantic search (0.0-1.0)
    hybrid_search_keyword_weight: float = 0.4  # Weight for keyword search (0.0-1.0)

    # ==========================================================================
    # RL-Based Threshold Learning
    # ==========================================================================
    # Enable automatic threshold learning per (embedding_model, task_type, tool_context).
    # Uses reinforcement learning to optimize semantic similarity thresholds.
    enable_semantic_threshold_rl_learning: bool = False
    semantic_threshold_overrides: dict = Field(
        default_factory=dict,
        description="Format: {'model:task:tool': threshold}",
    )

    @field_validator("codebase_dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        """Validate embedding dimension is positive.

        Args:
            v: Embedding dimension

        Returns:
            Validated dimension

        Raises:
            ValueError: If dimension is not positive
        """
        if v < 1:
            raise ValueError("codebase_dimension must be >= 1")
        return v

    @field_validator("codebase_batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive.

        Args:
            v: Batch size

        Returns:
            Validated batch size

        Raises:
            ValueError: If batch size is not positive
        """
        if v < 1:
            raise ValueError("codebase_batch_size must be >= 1")
        return v

    @field_validator("codebase_chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Validate chunk size is positive."""
        if v < 1:
            raise ValueError("codebase_chunk_size must be >= 1")
        return v

    @field_validator("codebase_chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int) -> int:
        """Validate chunk overlap is non-negative."""
        if v < 0:
            raise ValueError("codebase_chunk_overlap must be >= 0")
        return v

    @field_validator("semantic_similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        """Validate similarity threshold is in valid range.

        Args:
            v: Threshold value

        Returns:
            Validated threshold

        Raises:
            ValueError: If threshold is out of range
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError("semantic_similarity_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("semantic_max_query_expansions")
    @classmethod
    def validate_max_query_expansions(cls, v: int) -> int:
        """Validate max query expansions is positive.

        Args:
            v: Max expansions

        Returns:
            Validated max expansions

        Raises:
            ValueError: If max expansions is not positive
        """
        if v < 1:
            raise ValueError("semantic_max_query_expansions must be >= 1")
        return v

    @model_validator(mode="after")
    def validate_hybrid_search_weights(self) -> "EmbeddingSettings":
        """Validate that hybrid search weights sum to 1.0."""
        if self.enable_hybrid_search:
            total_weight = self.hybrid_search_semantic_weight + self.hybrid_search_keyword_weight
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(f"Hybrid search weights must sum to 1.0, got {total_weight}")
        return self
