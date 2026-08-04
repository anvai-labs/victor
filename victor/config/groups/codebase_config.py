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

"""Codebase configuration settings.

Extracted from victor/config/settings.py to improve maintainability.
Contains configuration for codebase semantic search and embeddings.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class CodebaseSettings(BaseModel):
    """Codebase semantic search configuration.

    Includes vector store, embedding model, graph store,
    and hybrid search settings.
    """

    # Unified Embedding Model (Optimized for Memory + Cache Efficiency)
    # Using same model for tool selection AND codebase search provides:
    # - 40% memory reduction (130MB vs 200MB)
    # - Better OS page cache utilization (1 model file instead of 2)
    # - Improved CPU L2/L3 cache hit rates
    # - Simpler management (1 model to download/update)
    #
    # Model: BAAI/bge-small-en-v1.5 (130MB, 384-dim, ~6ms)
    # - MTEB score: 62.2 (vs 58.8 for all-MiniLM-L6-v2)
    # - Excellent for code search (trained on code-related tasks)
    # - CPU-optimized, works great on consumer-grade hardware
    # - Native sentence-transformers support (no API needed)
    unified_embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Codebase Semantic Search (Air-gapped by Default)
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
    codebase_graph_writer_mode: str = (
        "off"  # "off" default; "compatibility" is deprecated compatibility mode
    )
    core_readonly_tools: Optional[list[str]] = None  # Override/extend curated read-only tool set

    # Semantic Search Quality Improvements (P4.X - Multi-Provider Excellence)
    semantic_similarity_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Min score [0.1-0.9], lowered from 0.5 for better recall on technical queries",
    )
    semantic_query_expansion_enabled: bool = True  # Expand queries with synonyms/related terms
    semantic_max_query_expansions: int = 5  # Max query variations to try (including original)

    # Hybrid Search (Semantic + Keyword with RRF)
    enable_hybrid_search: bool = False  # Enable hybrid search combining semantic + keyword
    hybrid_search_semantic_weight: float = 0.6  # Weight for semantic search (0.0-1.0)
    hybrid_search_keyword_weight: float = 0.4  # Weight for keyword search (0.0-1.0)

    # RL-based threshold learning per (embedding_model, task_type, tool_context)
    enable_semantic_threshold_rl_learning: bool = False  # Enable automatic threshold learning
    semantic_threshold_overrides: dict = Field(
        default_factory=dict, description="Format: {'model:task:tool': threshold}"
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
        if v <= 0:
            raise ValueError("codebase_dimension must be positive")
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
        if v <= 0:
            raise ValueError("codebase_batch_size must be positive")
        return v

    @field_validator("codebase_chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Validate chunk size is positive."""
        if v <= 0:
            raise ValueError("codebase_chunk_size must be positive")
        return v

    @field_validator("codebase_chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int) -> int:
        """Validate chunk overlap is non-negative."""
        if v < 0:
            raise ValueError("codebase_chunk_overlap must be >= 0")
        return v

    @field_validator("semantic_max_query_expansions")
    @classmethod
    def validate_max_expansions(cls, v: int) -> int:
        """Validate max query expansions is positive.

        Args:
            v: Max query expansions

        Returns:
            Validated max expansions

        Raises:
            ValueError: If max expansions is not positive
        """
        if v <= 0:
            raise ValueError("semantic_max_query_expansions must be positive")
        return v

    @field_validator("hybrid_search_semantic_weight", "hybrid_search_keyword_weight")
    @classmethod
    def validate_hybrid_weights(cls, v: float) -> float:
        """Validate hybrid search weights are in valid range.

        Args:
            v: Weight value

        Returns:
            Validated weight

        Raises:
            ValueError: If weight is not in [0.0, 1.0]
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError("Hybrid search weights must be in [0.0, 1.0]")
        return v
