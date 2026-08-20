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

"""
Basic in-memory vector store implementation.

This module provides a simple in-memory vector store for graceful
degradation when full vector databases are not available.

SOLID Principles:
- SRP: InMemoryVectorStore only handles in-memory vector storage
- OCP: Extensible through protocol implementation
- LSP: Implements VectorStoreProtocol completely
- ISP: Focused on vector store operations
- DIP: No dependencies on concrete implementations

Usage:
    from victor.contrib.vectorstores import InMemoryVectorStore

    store = InMemoryVectorStore()
    await store.add_documents(texts, embeddings)
    results = await store.search(query_vector, top_k=5)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from victor.framework.vertical_protocols import (
    VectorSearchResult,
    VectorStoreProtocol,
)

logger = logging.getLogger(__name__)


class InMemoryVectorStore(VectorStoreProtocol):
    """
    In-memory vector store for simple similarity search.

    This store provides basic vector similarity search using cosine
    similarity. Data is stored in memory and lost when the process exits.

    For production use with persistent vector databases (FAISS, Chroma, etc.),
    install victor-rag:
        pip install victor-rag

    Example:
        store = InMemoryVectorStore()
        await store.add_documents(
            texts=["doc1", "doc2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )
        results = await store.search(query_vector, top_k=2)
    """

    def __init__(self) -> None:
        """Initialize the in-memory vector store."""
        self._documents: Dict[str, str] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the in-memory store.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata: Optional metadata for each document
            **kwargs: Additional options (ignored)

        Returns:
            List of document IDs
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Number of documents ({len(documents)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        if metadata is None:
            metadata = [{} for _ in documents]

        document_ids: List[str] = []

        for doc_text, embedding, meta in zip(documents, embeddings, metadata):
            doc_id = f"doc_{self._counter}_{uuid.uuid4().hex[:8]}"
            self._counter += 1

            self._documents[doc_id] = doc_text
            self._embeddings[doc_id] = np.array(embedding, dtype=np.float32)
            self._metadata[doc_id] = meta

            document_ids.append(doc_id)

        logger.debug(f"Added {len(document_ids)} documents to in-memory store")
        return document_ids

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[VectorSearchResult]:
        """Search for similar documents using cosine similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            **kwargs: Additional options (ignored)

        Returns:
            List of search results ranked by similarity
        """
        if not self._embeddings:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)

        # Calculate cosine similarity for all documents
        results: List[tuple[str, float]] = []

        for doc_id, doc_embedding in self._embeddings.items():
            # Apply metadata filter if specified
            if filter_metadata:
                doc_meta = self._metadata.get(doc_id, {})
                if not all(doc_meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            # Cosine similarity: dot product of normalized vectors
            similarity = self._cosine_similarity(query_vec, doc_embedding)
            results.append((doc_id, similarity))

        # Sort by similarity (descending) and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        # Convert to VectorSearchResult objects
        search_results: List[VectorSearchResult] = []
        for doc_id, score in results:
            search_results.append(
                VectorSearchResult(
                    document_id=doc_id,
                    text=self._documents[doc_id],
                    score=float(score),
                    metadata=self._metadata.get(doc_id, {}),
                )
            )

        return search_results

    async def delete(
        self,
        document_ids: List[str],
        **kwargs: Any,
    ) -> bool:
        """Delete documents from the store.

        Args:
            document_ids: List of document IDs to delete
            **kwargs: Additional options (ignored)

        Returns:
            True if successful
        """
        for doc_id in document_ids:
            self._documents.pop(doc_id, None)
            self._embeddings.pop(doc_id, None)
            self._metadata.pop(doc_id, None)

        logger.debug(f"Deleted {len(document_ids)} documents from in-memory store")
        return True

    def get_store_info(self) -> Dict[str, Any]:
        """Get store metadata."""
        note = "Data lost on process exit. Install victor-rag for persistent vector stores."
        return {
            "type": "in-memory",
            "document_count": len(self._documents),
            "persistence": False,
            "info": {
                "note": note,
            },
            "note": note,
        }

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))


__all__ = ["InMemoryVectorStore"]
