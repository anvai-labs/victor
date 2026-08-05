# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Embedding system for Victor codebase intelligence.

This module combines:
1. Generic vector storage infrastructure from victor-core (victor.storage.vector_stores)
2. Code-specific AST-aware chunking (local to victor-coding)

For new code, prefer importing directly from victor.storage.vector_stores for
the generic embedding infrastructure:
    from victor.storage.vector_stores import EmbeddingConfig, EmbeddingRegistry

The code-specific ASTAwareChunker remains in this module:
    from victor_coding.codebase.embeddings.chunker import ASTAwareChunker
"""

# Re-export from victor-core for backward compatibility
from victor.storage.vector_stores import (
    # Base classes
    BaseEmbeddingProvider,
    BaseEmbeddingModel,
    EmbeddingConfig,
    EmbeddingModelConfig,
    EmbeddingSearchResult,
    # Embedding models
    SentenceTransformerModel,
    OllamaEmbeddingModel,
    OpenAIEmbeddingModel,
    CohereEmbeddingModel,
    create_embedding_model,
    # Registry
    EmbeddingRegistry,
)

# Code-specific: AST-aware chunking (stays in victor-coding)
from victor_coding.codebase.embeddings.chunker import ASTAwareChunker

__all__ = [
    # Base classes (from victor-core)
    "BaseEmbeddingProvider",
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "EmbeddingModelConfig",
    "EmbeddingSearchResult",
    # Embedding models (from victor-core)
    "SentenceTransformerModel",
    "OllamaEmbeddingModel",
    "OpenAIEmbeddingModel",
    "CohereEmbeddingModel",
    "create_embedding_model",
    # Registry (from victor-core)
    "EmbeddingRegistry",
    # Code-specific (local)
    "ASTAwareChunker",
]
