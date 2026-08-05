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

"""Framework-level document ingestion utilities.

This module provides domain-agnostic ingestion capabilities that can be
used across all verticals. These are reusable building blocks for
document processing pipelines.

Provides:
- ChunkingConfig: Configuration for document chunking
- ChunkerProtocol: Protocol for vertical-specific chunkers
- BaseChunker: Base implementation with common strategies
- DocumentType detection utilities
- Source handlers for files, URLs, PDFs, HTML

Usage:
    from victor.framework.ingestion import (
        ChunkingConfig,
        BaseChunker,
        detect_document_type,
    )

    config = ChunkingConfig(chunk_size=1000, chunk_overlap=100)
    chunker = BaseChunker(config)
    chunks = chunker.chunk_text(content)

Verticals extend this by:
- Providing domain-specific chunking strategies
- Adding metadata extraction relevant to their domain
- Integrating with their storage backends
"""

from victor.framework.ingestion.models import (
    Chunk,
    ChunkingConfig,
    DocumentType,
)
from victor.framework.ingestion.protocols import (
    ChunkerProtocol,
    SourceHandlerProtocol,
)
from victor.framework.ingestion.chunker import (
    BaseChunker,
    detect_document_type,
    EXTENSION_TO_DOCTYPE,
)

__all__ = [
    # Models
    "Chunk",
    "ChunkingConfig",
    "DocumentType",
    # Protocols
    "ChunkerProtocol",
    "SourceHandlerProtocol",
    # Implementations
    "BaseChunker",
    "detect_document_type",
    "EXTENSION_TO_DOCTYPE",
]
