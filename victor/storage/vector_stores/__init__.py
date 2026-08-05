# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Generic vector storage module for Victor framework.

This module provides a plugin-based architecture for vector storage with
multiple backend and embedding model options:

**Vector Store Backends:**
- **LanceDBProvider**: Production, fast, disk-based (default)
- **ProximaDBProvider**: Production, fast vector + graph (recommended)

**Embedding Models:**
- **SentenceTransformerModel**: Local, free, CPU/GPU
- **OllamaEmbeddingModel**: Local via Ollama (Qwen3, etc.)
- **OpenAIEmbeddingModel**: Cloud, fast, costs money
- **CohereEmbeddingModel**: Cloud, multilingual

Example:
    ```python
    from victor.storage.vector_stores import EmbeddingConfig, EmbeddingRegistry

    # Production: Local embeddings + LanceDB
    config = EmbeddingConfig(
        vector_store="lancedb",
        embedding_model_type="sentence-transformers",
        embedding_model_name="BAAI/bge-small-en-v1.5",
        persist_directory="~/.victor/embeddings"
    )

    # Get provider
    provider = EmbeddingRegistry.get_provider(config)
    await provider.initialize()

    # Index and search
    await provider.index_documents(documents)
    results = await provider.search("query", k=10)
    ```

This is a core framework module available to all verticals, not just coding.
"""

from victor.storage.vector_stores.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingSearchResult,
)
from victor.storage.vector_stores.models import (
    BaseEmbeddingModel,
    EmbeddingModelConfig,
    SentenceTransformerModel,
    OllamaEmbeddingModel,
    OpenAIEmbeddingModel,
    CohereEmbeddingModel,
    create_embedding_model,
)
from victor.storage.vector_stores.registry import EmbeddingRegistry

__all__ = [
    # Base classes
    "BaseEmbeddingProvider",
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "EmbeddingModelConfig",
    "EmbeddingSearchResult",
    # Embedding models
    "SentenceTransformerModel",
    "OllamaEmbeddingModel",
    "OpenAIEmbeddingModel",
    "CohereEmbeddingModel",
    "create_embedding_model",
    # Registry
    "EmbeddingRegistry",
    "reset_embedding_registry",
    "get_embedding_registry_stats",
]


def reset_embedding_registry() -> None:
    """Reset the embedding provider cache.

    This is a convenience function that calls EmbeddingRegistry.reset().
    Mainly useful for testing to clear cached provider instances.
    """
    EmbeddingRegistry.reset()


def get_embedding_registry_stats() -> dict:
    """Get statistics about the embedding provider cache.

    Returns:
        Dictionary with cache size and provider breakdown
    """
    return EmbeddingRegistry.get_cache_stats()
