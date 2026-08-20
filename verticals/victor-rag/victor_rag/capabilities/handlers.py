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

"""RAG capability handlers for configuring and getting RAG settings.

This module provides handler functions for configuring various RAG capabilities
including indexing, retrieval, synthesis, safety, and query enhancement.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor_rag.capabilities.config import (
    _INDEXING_DEFAULTS,
    _RETRIEVAL_DEFAULTS,
    _SYNTHESIS_DEFAULTS,
    _SAFETY_DEFAULTS,
    _load_rag_config,
    _store_rag_section,
)

logger = logging.getLogger(__name__)


def configure_indexing(
    orchestrator: Any,
    *,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    embedding_model: str = "text-embedding-3-small",
    embedding_dimensions: int = 1536,
    store_backend: str = "lancedb",
) -> None:
    """Configure document indexing settings for the orchestrator.

    This capability configures the orchestrator's indexing settings
    for document ingestion and chunking.

    Args:
        orchestrator: Target orchestrator
        chunk_size: Target size for document chunks
        chunk_overlap: Overlap between consecutive chunks
        embedding_model: Model to use for generating embeddings
        embedding_dimensions: Dimension of embedding vectors
        store_backend: Vector store backend (lancedb, chromadb, etc.)
    """
    _store_rag_section(
        orchestrator,
        "indexing",
        {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dimensions,
            "store_backend": store_backend,
        },
    )

    logger.info(f"Configured RAG indexing: chunk_size={chunk_size}, model={embedding_model}")


def get_indexing_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current indexing configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Indexing configuration dict
    """
    return _load_rag_config(orchestrator).get("indexing", dict(_INDEXING_DEFAULTS))


def configure_retrieval(
    orchestrator: Any,
    *,
    top_k: int = 5,
    similarity_threshold: float = 0.7,
    search_type: str = "hybrid",
    rerank_enabled: bool = True,
    max_context_tokens: int = 4000,
) -> None:
    """Configure retrieval settings for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        top_k: Number of top results to retrieve
        similarity_threshold: Minimum similarity score for retrieval
        search_type: Type of search (semantic, keyword, hybrid)
        rerank_enabled: Whether to rerank retrieved results
        max_context_tokens: Maximum tokens in context window
    """
    _store_rag_section(
        orchestrator,
        "retrieval",
        {
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "search_type": search_type,
            "rerank_enabled": rerank_enabled,
            "max_context_tokens": max_context_tokens,
        },
    )

    logger.info(f"Configured RAG retrieval: top_k={top_k}, search_type={search_type}")


def get_retrieval_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current retrieval configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Retrieval configuration dict
    """
    return _load_rag_config(orchestrator).get("retrieval", dict(_RETRIEVAL_DEFAULTS))


def configure_synthesis(
    orchestrator: Any,
    *,
    citation_style: str = "inline",
    include_sources: bool = True,
    max_answer_tokens: int = 2000,
    temperature: float = 0.3,
    require_verification: bool = True,
) -> None:
    """Configure answer synthesis settings for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        citation_style: How to format citations (inline, footnote, endnote)
        include_sources: Whether to include source list in answer
        max_answer_tokens: Maximum tokens in generated answer
        temperature: LLM temperature for synthesis
        require_verification: Whether to verify answer against sources
    """
    _store_rag_section(
        orchestrator,
        "synthesis",
        {
            "citation_style": citation_style,
            "include_sources": include_sources,
            "max_answer_tokens": max_answer_tokens,
            "temperature": temperature,
            "require_verification": require_verification,
        },
    )

    logger.info(f"Configured RAG synthesis: citation_style={citation_style}")


def get_synthesis_config(orchestrator: Any) -> Dict[str, Any]:
    """Get current synthesis configuration.

    Args:
        orchestrator: Target orchestrator

    Returns:
        Synthesis configuration dict
    """
    return _load_rag_config(orchestrator).get("synthesis", dict(_SYNTHESIS_DEFAULTS))


def configure_safety(
    orchestrator: Any,
    *,
    filter_sensitive_data: bool = True,
    max_document_size_mb: int = 50,
    allowed_file_types: Optional[List[str]] = None,
    validate_sources: bool = True,
) -> None:
    """Configure RAG safety settings for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        filter_sensitive_data: Whether to filter sensitive data from documents
        max_document_size_mb: Maximum document size in MB
        allowed_file_types: List of allowed file extensions
        validate_sources: Whether to validate source URLs
    """
    _store_rag_section(
        orchestrator,
        "safety",
        {
            "filter_sensitive_data": filter_sensitive_data,
            "max_document_size_mb": max_document_size_mb,
            "allowed_file_types": allowed_file_types
            or list(_SAFETY_DEFAULTS["allowed_file_types"]),
            "validate_sources": validate_sources,
        },
    )

    logger.info(f"Configured RAG safety: filter_sensitive={filter_sensitive_data}")


def configure_query_enhancement(
    orchestrator: Any,
    *,
    enable_expansion: bool = True,
    enable_decomposition: bool = True,
    max_query_variants: int = 3,
    use_synonyms: bool = True,
) -> None:
    """Configure query enhancement settings for the orchestrator.

    Args:
        orchestrator: Target orchestrator
        enable_expansion: Whether to expand queries with synonyms/related terms
        enable_decomposition: Whether to decompose complex queries
        max_query_variants: Maximum number of query variants to generate
        use_synonyms: Whether to use synonym expansion
    """
    _store_rag_section(
        orchestrator,
        "query_enhancement",
        {
            "enable_expansion": enable_expansion,
            "enable_decomposition": enable_decomposition,
            "max_query_variants": max_query_variants,
            "use_synonyms": use_synonyms,
        },
    )

    logger.info(f"Configured query enhancement: expansion={enable_expansion}")


# Re-export for backward compatibility
__all__ = [
    "configure_indexing",
    "get_indexing_config",
    "configure_retrieval",
    "get_retrieval_config",
    "configure_synthesis",
    "get_synthesis_config",
    "configure_safety",
    "configure_query_enhancement",
]
