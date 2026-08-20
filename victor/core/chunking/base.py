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

"""Base classes for document chunking.

This module provides the foundational classes for chunking documents:
- ChunkingConfig: Configuration dataclass
- Chunk: Result dataclass for a single chunk
- ChunkingStrategy: Protocol for chunking strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple


@dataclass
class ChunkingConfig:
    """Configuration for document chunking.

    Attributes:
        chunk_size: Target chunk size in characters (default: 512)
        chunk_overlap: Overlap between chunks in characters (default: 50)
        min_chunk_size: Minimum chunk size - avoid tiny chunks (default: 100)
        max_chunk_size: Maximum chunk size - hard limit (default: 2000)
        respect_boundaries: Try to break at natural boundaries (default: True)
    """

    chunk_size: int = 1024
    chunk_overlap: int = 128
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    respect_boundaries: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.min_chunk_size < 0:
            raise ValueError("min_chunk_size must be non-negative")
        if self.max_chunk_size < self.chunk_size:
            raise ValueError("max_chunk_size must be >= chunk_size")


@dataclass
class Chunk:
    """A single chunk of content.

    Attributes:
        content: The chunk text content
        start_char: Starting character position in source
        end_char: Ending character position in source
        chunk_type: Type of chunk (e.g., "paragraph", "section", "header")
        metadata: Additional metadata about the chunk
    """

    content: str
    start_char: int
    end_char: int
    chunk_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Length of chunk content."""
        return len(self.content)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies.

    Each strategy handles a specific document type (HTML, JSON, Markdown, etc.)
    and knows how to chunk it while preserving semantic structure.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize strategy with configuration.

        Args:
            config: Chunking configuration (uses defaults if not provided)
        """
        self.config = config or ChunkingConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name (e.g., 'html', 'json', 'markdown')."""
        ...

    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """Document types this strategy supports."""
        ...

    @abstractmethod
    def chunk(self, content: str) -> List[Chunk]:
        """Chunk content into semantic pieces.

        Args:
            content: Document content to chunk

        Returns:
            List of Chunk objects
        """
        ...

    def can_handle(self, doc_type: str) -> bool:
        """Check if strategy can handle a document type.

        Args:
            doc_type: Document type to check

        Returns:
            True if this strategy can handle the type
        """
        return doc_type.lower() in [t.lower() for t in self.supported_types]


class ChunkingStrategyProtocol(Protocol):
    """Protocol for chunking strategies (for type hints)."""

    @property
    def name(self) -> str: ...

    @property
    def supported_types(self) -> List[str]: ...

    def chunk(self, content: str) -> List[Chunk]: ...

    def can_handle(self, doc_type: str) -> bool: ...
