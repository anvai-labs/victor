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

"""Domain-agnostic models for document ingestion.

These models are designed to be reusable across all verticals:
- RAG: Document chunks for retrieval
- Coding: Code snippets for semantic search
- Research: Paper sections for citation
- DevOps: Config file segments
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DocumentType(Enum):
    """Supported document types for chunking strategies."""

    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    CODE = "code"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"
    AUTO = "auto"  # Auto-detect from content


@dataclass
class ChunkingConfig:
    """Configuration for document chunking.

    This is a domain-agnostic configuration that verticals can extend
    or use with custom defaults.

    Attributes:
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks for context continuity
        min_chunk_size: Minimum chunk size (skip smaller chunks)
        max_chunk_size: Maximum chunk size (hard limit)
        respect_sentence_boundaries: Try to break at sentence ends
        respect_paragraph_boundaries: Try to break at paragraphs
        code_aware: Use code-aware chunking for code files
    """

    chunk_size: int = 1344  # Optimized for BGE embeddings (384 * 3.5)
    chunk_overlap: int = 134  # ~10% overlap
    min_chunk_size: int = 200
    max_chunk_size: int = 2000
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    code_aware: bool = True


@dataclass
class Chunk:
    """A chunk of content from a document.

    This is the framework-level chunk representation. Verticals can
    extend this or convert to their own formats.

    Attributes:
        content: The chunk text content
        start_char: Start position in original document
        end_char: End position in original document
        chunk_index: Index of this chunk in the document
        doc_type: Detected document type
        metadata: Additional metadata (vertical-specific)
    """

    content: str
    start_char: int
    end_char: int
    chunk_index: int = 0
    doc_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Length of chunk content."""
        return len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_index": self.chunk_index,
            "doc_type": self.doc_type,
            "metadata": self.metadata,
        }


@dataclass
class SourceContent:
    """Content extracted from a source (file, URL, etc.).

    Attributes:
        content: The extracted text content
        source: Source path or URL
        doc_type: Detected or specified document type
        metadata: Source-specific metadata
    """

    content: str
    source: str
    doc_type: DocumentType = DocumentType.AUTO
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "DocumentType",
    "ChunkingConfig",
    "Chunk",
    "SourceContent",
]
