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

"""Document chunking module for Victor core.

Provides intelligent document chunking with multiple strategies:
- Text: Sentence-boundary aware chunking
- HTML: Semantic structure (paragraphs, sections, tables)
- JSON: Object/array boundary preservation
- Markdown: Header-based sections
- Code: Function/class boundary detection

Example:
    from victor.core.chunking import chunk_document, ChunkingConfig

    # Auto-detect document type from source
    chunks = chunk_document(
        content=html_content,
        source="https://sec.gov/filing.htm"
    )

    # Explicit type with custom config
    config = ChunkingConfig(chunk_size=1024, chunk_overlap=100)
    chunks = chunk_document(
        content=json_content,
        doc_type="json",
        config=config
    )

    # Use registry directly for more control
    from victor.core.chunking import ChunkingRegistry

    registry = ChunkingRegistry()
    strategy = registry.get_strategy("html")
    chunks = strategy.chunk(html_content)
"""

from victor.core.chunking.base import (
    Chunk,
    ChunkingConfig,
    ChunkingStrategy,
    ChunkingStrategyProtocol,
)
from victor.core.chunking.detector import (
    detect_document_type,
    detect_from_content,
    detect_from_extension,
)
from victor.core.chunking.registry import (
    ChunkingRegistry,
    chunk_document,
    get_chunking_registry,
)

__all__ = [
    # Core classes
    "Chunk",
    "ChunkingConfig",
    "ChunkingStrategy",
    "ChunkingStrategyProtocol",
    # Registry
    "ChunkingRegistry",
    "get_chunking_registry",
    # Convenience function
    "chunk_document",
    # Detection
    "detect_document_type",
    "detect_from_content",
    "detect_from_extension",
]
