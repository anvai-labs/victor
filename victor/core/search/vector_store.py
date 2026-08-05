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

"""Base vector store interface and in-memory implementation for core.

This module provides the foundation for semantic search in Victor.
Verticals can extend this or provide their own via VectorStoreProtocol.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from victor.framework.vertical_protocols import VectorStoreProtocol, VectorSearchResult

logger = logging.getLogger(__name__)


class BaseVectorStore(VectorStoreProtocol):
    """Abstract base class for vector stores with common logic."""

    def __init__(self, name: str = "default"):
        self.name = name

    def get_store_info(self) -> Dict[str, Any]:
        """Get store metadata."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
        }


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store for core/testing.

    NOT suitable for large datasets or persistence.
    """

    def __init__(self, name: str = "in_memory"):
        super().__init__(name)
        self._documents: List[str] = []
        self._embeddings: List[List[float]] = []
        self._metadata: List[Dict[str, Any]] = []
        self._ids: List[str] = []

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to memory."""
        import uuid

        new_ids = [str(uuid.uuid4()) for _ in documents]
        self._documents.extend(documents)
        self._embeddings.extend(embeddings)
        self._ids.extend(new_ids)

        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in documents])

        return new_ids

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[VectorSearchResult]:
        """Simple cosine similarity search in memory."""
        import numpy as np

        if not self._embeddings:
            return []

        # Convert to numpy for fast math
        q = np.array(query_embedding)
        db = np.array(self._embeddings)

        # Cosine similarity
        norm_q = np.linalg.norm(q)
        norm_db = np.linalg.norm(db, axis=1)
        similarities = np.dot(db, q) / (norm_db * norm_q)

        # Get top-k indices
        indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i in indices:
            # Apply filters if any (simple exact match)
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if self._metadata[i].get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            results.append(
                VectorSearchResult(
                    document_id=self._ids[i],
                    text=self._documents[i],
                    score=float(similarities[i]),
                    metadata=self._metadata[i],
                )
            )
        return results

    async def delete(self, document_ids: List[str], **kwargs: Any) -> bool:
        """Delete documents from memory."""
        indices_to_remove = []
        for i, doc_id in enumerate(self._ids):
            if doc_id in document_ids:
                indices_to_remove.append(i)

        for i in sorted(indices_to_remove, reverse=True):
            self._documents.pop(i)
            self._embeddings.pop(i)
            self._ids.pop(i)
            self._metadata.pop(i)

        return True
