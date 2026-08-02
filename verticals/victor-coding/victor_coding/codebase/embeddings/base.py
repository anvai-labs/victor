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

"""Base embedding provider interface.

This module separates concerns:
1. **Embedding Model**: Generates vectors from text (sentence-transformers, OpenAI, etc.)
2. **Vector Store**: Stores and searches vectors (ChromaDB, LanceDB, FAISS, etc.)
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from victor_coding.codebase.embeddings.models import BaseEmbeddingModel


class EmbeddingConfig(BaseModel):
    """Configuration for embedding system.

    This configures BOTH the embedding model (text -> vector) and vector store (storage/search).
    """

    # Vector Store Configuration
    vector_store: str = Field(
        default="lancedb",
        description="Vector store provider (lancedb, chromadb) - LanceDB recommended for performance",
    )
    persist_directory: Optional[str] = Field(
        default=None,
        description="Directory to persist vector store (default: ~/.victor/embeddings/codebase)",
    )
    distance_metric: str = Field(
        default="cosine", description="Distance metric (cosine, euclidean, dot)"
    )

    # Embedding Model Configuration (Air-gapped by Default)
    embedding_model_type: str = Field(
        default="sentence-transformers",
        description="Embedding model type (sentence-transformers=local/offline, ollama, openai, cohere)",
    )
    embedding_model_name: str = Field(
        default="all-MiniLM-L12-v2",
        description="Embedding model name (all-MiniLM-L12-v2 = 384-dim, 120MB, ~8ms, optimal balance)",
    )
    embedding_api_key: Optional[str] = Field(
        default=None,
        description="API key for cloud embedding providers (or Ollama base URL)",
    )

    # Provider-specific configuration
    extra_config: Dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific configuration"
    )


# SDK-compatible embedding search result (no victor-ai dependency)
try:
    from victor_contracts.verticals.protocols.storage import (
        EmbeddingSearchResultData as EmbeddingSearchResult,
    )
except ImportError:
    from victor.storage.vector_stores.base import EmbeddingSearchResult  # type: ignore[no-redef]

# Public search result type for this module
__all_search__ = ["EmbeddingSearchResult"]


class BaseEmbeddingProvider(ABC):
    """Abstract base class for vector store providers.

    This handles STORAGE and SEARCH of embeddings.
    Embedding generation is delegated to a pluggable BaseEmbeddingModel.

    Separation of concerns:
    - Embedding Model: text -> vector (sentence-transformers, OpenAI, etc.)
    - Vector Store: vector storage/search (ChromaDB, LanceDB, FAISS, etc.)

    This allows mixing and matching:
    - OpenAI embeddings + FAISS storage
    - Sentence-transformers + LanceDB storage
    - Cohere embeddings + ChromaDB storage
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        embedding_model: Optional["BaseEmbeddingModel"] = None,
    ):
        """Initialize provider with configuration.

        Args:
            config: Embedding configuration
            embedding_model: Optional pre-configured embedding model.
                           If None, will be created from config.
        """
        self.config = config
        self.embedding_model: Optional["BaseEmbeddingModel"] = embedding_model
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider.

        This should:
        - Load embedding models
        - Connect to databases
        - Set up any required resources

        Should be idempotent (safe to call multiple times).
        """
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (list of floats)
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (optimized batch operation).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    async def index_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """Index a single document.

        Args:
            doc_id: Unique identifier for the document
            content: Content to index
            metadata: Additional metadata (file_path, symbol_name, etc.)
        """
        pass

    @abstractmethod
    async def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Batch index multiple documents (optimized).

        Args:
            documents: List of documents, each with:
                - id: Unique identifier
                - content: Content to index
                - metadata: Additional metadata
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EmbeddingSearchResult]:
        """Search for documents similar to query.

        Args:
            query: Search query
            limit: Maximum number of results
            filter_metadata: Optional metadata filters (e.g., {"file_path": "src/"})

        Returns:
            List of search results, sorted by relevance (highest first)
        """
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """Delete a document from the index.

        Args:
            doc_id: Document identifier to delete
        """
        pass

    @abstractmethod
    async def delete_by_file(self, file_path: str) -> int:
        """Delete all documents from a specific file.

        Used for incremental updates - when a file changes, we delete all
        its chunks and re-index.

        Args:
            file_path: Relative file path to delete documents for

        Returns:
            Number of documents deleted
        """
        pass

    @abstractmethod
    async def clear_index(self) -> None:
        """Clear the entire index (remove all documents)."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index.

        Returns:
            Dictionary with stats like:
            - total_documents: Number of indexed documents
            - total_embeddings: Number of embedding vectors
            - index_size_mb: Size of index in MB
            - model_name: Name of embedding model
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (close connections, etc.).

        Subclasses should override if cleanup is needed.
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(provider={self.config.provider}, model={self.config.model})"
        )
