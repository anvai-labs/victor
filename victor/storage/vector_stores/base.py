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

"""Base embedding provider interface.

This module separates concerns:
1. **Embedding Model**: Generates vectors from text (sentence-transformers, OpenAI, etc.)
2. **Vector Store**: Stores and searches vectors (LanceDB, ProximaDB, etc.)
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from victor.core.search_types import SearchHit

if TYPE_CHECKING:
    from victor.storage.vector_stores.models import BaseEmbeddingModel


class EmbeddingConfig(BaseModel):
    """Configuration for embedding system.

    This configures BOTH the embedding model (text -> vector) and vector store (storage/search).
    """

    # Vector Store Configuration
    vector_store: str = Field(
        default="lancedb",
        description="Vector store provider (lancedb, proximadb) - LanceDB recommended for performance",
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
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model name (BAAI/bge-small-en-v1.5 = 384-dim, 130MB, ~6ms, recommended default)",
    )
    embedding_api_key: Optional[str] = Field(
        default=None,
        description="API key for cloud embedding providers (or Ollama base URL)",
    )

    # Provider-specific configuration
    extra_config: Dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific configuration"
    )


class EmbeddingSearchResult(BaseModel):
    """Result from embedding/vector semantic search.

    For vector store search results based on embedding similarity.
    Renamed from SearchResult to be semantically distinct from other search types.
    """

    file_path: str = Field(description="Path to the file")
    symbol_name: Optional[str] = Field(default=None, description="Symbol name if applicable")
    content: str = Field(description="Content that matched")
    score: float = Field(description="Relevance score (0-1, higher is better)")
    line_number: Optional[int] = Field(default=None, description="Line number in file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_search_hit(self) -> SearchHit:
        """Convert to a generic SearchHit for cross-layer consumers."""
        return SearchHit(
            file_path=self.file_path,
            content=self.content,
            score=self.score,
            line_number=self.line_number,
            metadata=self.metadata,
        )


class BaseEmbeddingProvider(ABC):
    """Abstract base class for vector store providers.

    This handles STORAGE and SEARCH of embeddings.
    Embedding generation is delegated to a pluggable BaseEmbeddingModel.

    Separation of concerns:
    - Embedding Model: text -> vector (sentence-transformers, OpenAI, etc.)
    - Vector Store: vector storage/search (LanceDB, ProximaDB, etc.)

    This allows mixing and matching:
    - OpenAI embeddings + FAISS storage
    - Sentence-transformers + LanceDB storage
    - Cohere embeddings + LanceDB storage
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
            f"{self.__class__.__name__}("
            f"provider={self.config.vector_store}, "
            f"model={self.config.embedding_model_name})"
        )
