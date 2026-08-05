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

"""RAG Document Models.

This module provides data models for RAG document storage:
- Document: Source document with metadata
- DocumentChunk: Indexed chunk with embedding
- DocumentSearchResult: Search result with score and context
- DocumentStoreConfig: Configuration for document store
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Document:
    """Source document for RAG.

    Attributes:
        id: Unique document identifier
        content: Full document content
        source: Source path or URL
        doc_type: Document type (pdf, markdown, text, code)
        metadata: Additional metadata
        created_at: Creation timestamp
    """

    id: str
    content: str
    source: str
    doc_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def content_hash(self) -> str:
        """Get content hash for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class DocumentChunk:
    """Indexed chunk of a document.

    Attributes:
        id: Unique chunk identifier
        doc_id: Parent document ID
        content: Chunk content
        embedding: Vector embedding
        chunk_index: Position in document
        start_char: Start character offset
        end_char: End character offset
        metadata: Chunk-specific metadata
    """

    id: str
    doc_id: str
    content: str
    embedding: List[float]
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentSearchResult:
    """RAG document search result with relevance score.

    For document/chunk-based RAG search results.
    Renamed from SearchResult to be semantically distinct from other search types.

    Attributes:
        chunk: The matched chunk
        score: Relevance score (0-1, higher is better)
        highlights: Highlighted text segments
        doc_source: Source document path
    """

    chunk: DocumentChunk
    score: float
    highlights: List[str] = field(default_factory=list)
    doc_source: str = ""

    @property
    def content(self) -> str:
        """Convenience property for chunk content."""
        return self.chunk.content

    @property
    def metadata(self) -> Dict[str, Any]:
        """Convenience property for chunk metadata."""
        return self.chunk.metadata

    @property
    def doc_id(self) -> str:
        """Convenience property for document ID."""
        return self.chunk.doc_id


@dataclass
class DocumentStoreConfig:
    """Configuration for document store.

    Attributes:
        path: Path to store data
        table_name: LanceDB table name
        embedding_dim: Embedding dimension
        use_hybrid_search: Enable hybrid (vector + full-text) search
        rerank_results: Enable reranking
        max_results: Maximum results per search
    """

    path: Path = field(default_factory=lambda: Path(".victor/rag"))
    table_name: str = "documents"
    embedding_dim: int = 384  # Default for sentence-transformers/all-MiniLM-L6-v2
    use_hybrid_search: bool = True
    rerank_results: bool = True
    max_results: int = 20


__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentSearchResult",
    "DocumentStoreConfig",
]
