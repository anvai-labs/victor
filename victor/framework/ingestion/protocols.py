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

"""Protocols for document ingestion components.

These protocols enable DIP compliance - verticals depend on abstractions
rather than concrete implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from victor.framework.ingestion.models import Chunk, ChunkingConfig, SourceContent


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Protocol for document chunkers.

    Verticals implement this to provide domain-specific chunking strategies.

    Example:
        class ASTAwareChunker:
            '''Coding vertical chunker using tree-sitter.'''

            def chunk(self, content: str, doc_type: str) -> List[Chunk]:
                # Use AST for code-aware chunking
                ...
    """

    @property
    def config(self) -> ChunkingConfig:
        """Get chunking configuration."""
        ...

    def chunk(
        self,
        content: str,
        doc_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Chunk content into smaller pieces.

        Args:
            content: Text content to chunk
            doc_type: Document type for strategy selection
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        ...


@runtime_checkable
class SourceHandlerProtocol(Protocol):
    """Protocol for source content handlers.

    Handles extraction of content from various sources (files, URLs, etc.).

    Example:
        class PDFHandler:
            '''Extracts text from PDF files.'''

            async def extract(self, source: str) -> SourceContent:
                # Use pypdf to extract text
                ...
    """

    async def extract(self, source: str) -> SourceContent:
        """Extract content from a source.

        Args:
            source: Source path or URL

        Returns:
            SourceContent with extracted text and metadata
        """
        ...

    def can_handle(self, source: str) -> bool:
        """Check if this handler can process the source.

        Args:
            source: Source path or URL

        Returns:
            True if this handler can extract from the source
        """
        ...


@runtime_checkable
class IngestionPipelineProtocol(Protocol):
    """Protocol for complete ingestion pipelines.

    Combines source handling and chunking into a unified pipeline.
    """

    async def ingest(
        self,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Ingest content from a source.

        Args:
            source: Source path or URL
            metadata: Optional metadata to attach

        Returns:
            List of chunks from the source
        """
        ...


__all__ = [
    "ChunkerProtocol",
    "SourceHandlerProtocol",
    "IngestionPipelineProtocol",
]
