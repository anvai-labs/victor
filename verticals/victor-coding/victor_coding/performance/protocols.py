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

"""Protocol interfaces for performance-improved backend implementations.

These protocols define the contract that both Python and Rust backends must fulfill.
The PerformanceBackendRegistry uses these protocols to enable transparent switching
between implementations based on availability and priority.

Usage:
    # Register a native backend with high priority
    PerformanceBackendRegistry.register(
        "fast_indexer",
        NativeIndexer,
        priority=80
    )

    # Create highest-priority available backend
    indexer = PerformanceBackendRegistry.create(
        FastIndexerProtocol,
        IndexerConfig()
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple


@dataclass
class IndexedFileData:
    """Result of indexing a single file."""

    file_path: str
    language: str
    content_hash: str
    symbols: List[Dict[str, Any]] = field(default_factory=list)
    call_edges: List[Tuple[str, str]] = field(default_factory=list)
    inherit_edges: List[Tuple[str, str]] = field(default_factory=list)
    implements_edges: List[Tuple[str, str]] = field(default_factory=list)
    compose_edges: List[Tuple[str, str]] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    line_count: int = 0
    error: Optional[str] = None


@dataclass
class ExtractedData:
    """Result of extracting symbols and edges from a file."""

    file_path: str
    language: str
    symbols: List[Dict[str, Any]]
    call_edges: List[Tuple[str, str, int]]
    inherit_edges: List[Tuple[str, str]]
    implements_edges: List[Tuple[str, str]]
    compose_edges: List[Tuple[str, str]]
    references: List[Tuple[str, int, int]]


@dataclass
class ChunkInfo:
    """Information about a code chunk."""

    content: str
    chunk_type: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None
    symbol_type: Optional[str] = None
    file_path: Optional[str] = None
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FastIndexerProtocol(Protocol):
    """Protocol for high-performance file indexing.

    Implementations can be pure Python or Rust-based (via PyO3).
    The PerformanceBackendRegistry selects the highest-priority available implementation.
    """

    priority: int

    async def index_file(
        self, file_path: str | Path, root: str | Path
    ) -> Optional[IndexedFileData]:
        """Index a single file and extract symbols, edges, and metadata.

        Args:
            file_path: Path to the file to index
            root: Root directory of the codebase

        Returns:
            IndexedFileData if successful, None if file cannot be indexed
        """
        ...

    async def index_batch(
        self, file_paths: List[str | Path], root: str | Path
    ) -> List[Optional[IndexedFileData]]:
        """Index multiple files in batch for better throughput.

        Args:
            file_paths: List of file paths to index
            root: Root directory of the codebase

        Returns:
            List of IndexedFileData, one per input file (None for failures)
        """
        ...

    def supports_language(self, language: str) -> bool:
        """Check if this indexer supports the given language."""
        ...

    def get_supported_languages(self) -> List[str]:
        """Get list of languages supported by this indexer."""
        ...


class FastChunkerProtocol(Protocol):
    """Protocol for high-performance code chunking.

    Implementations can be pure Python or Rust-based (via PyO3).
    """

    priority: int

    def chunk_code(
        self, content: str, language: str, file_path: Optional[str] = None
    ) -> List[ChunkInfo]:
        """Chunk code content into semantic units.

        Args:
            content: Source code content
            language: Programming language
            file_path: Optional file path for metadata

        Returns:
            List of chunks with metadata
        """
        ...

    def chunk_file(self, file_path: str | Path, language: Optional[str] = None) -> List[ChunkInfo]:
        """Chunk a file by reading and processing it.

        Args:
            file_path: Path to the file
            language: Optional language (auto-detected if None)

        Returns:
            List of chunks with metadata
        """
        ...

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using configured heuristic.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        ...

    def supports_language(self, language: str) -> bool:
        """Check if this chunker supports the given language."""
        ...


class FastSymbolExtractorProtocol(Protocol):
    """Protocol for high-performance symbol and edge extraction.

    This protocol enables single-pass extraction of symbols and all edge types,
    which is the main optimization target for Rust implementations.
    """

    priority: int

    def extract_all(self, file_path: str | Path, language: str) -> Optional[ExtractedData]:
        """Extract all symbols and edges in a single pass.

        This is the key optimization: instead of 4 separate traversals for
        different edge types, a single traversal extracts everything.

        Args:
            file_path: Path to the file
            language: Programming language

        Returns:
            ExtractedData with symbols and all edge types, or None on error
        """
        ...

    def extract_symbols(self, file_path: str | Path, language: str) -> List[Dict[str, Any]]:
        """Extract only symbols (no edges).

        Args:
            file_path: Path to the file
            language: Programming language

        Returns:
            List of symbol dictionaries
        """
        ...

    def extract_call_edges(
        self, file_path: str | Path, language: str
    ) -> List[Tuple[str, str, int]]:
        """Extract call edges with line numbers.

        Args:
            file_path: Path to the file
            language: Programming language

        Returns:
            List of (caller, callee, line_number) tuples
        """
        ...

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        ...

    def is_native(self) -> bool:
        """Check if this is a native (Rust) implementation."""
        ...


class FastRegexProcessorProtocol(Protocol):
    """Protocol for high-performance regex operations.

    Python's re module is slower than Rust's regex crate for complex patterns.
    This protocol enables optimized regex processing.
    """

    priority: int

    def findall(self, pattern: str, text: str, flags: int = 0) -> List[str]:
        """Find all matches of pattern in text.

        Args:
            pattern: Regex pattern
            text: Text to search
            flags: Optional regex flags

        Returns:
            List of matches
        """
        ...

    def finditer(self, pattern: str, text: str, flags: int = 0) -> Iterable[Any]:
        """Find all matches as an iterator.

        Args:
            pattern: Regex pattern
            text: Text to search
            flags: Optional regex flags

        Returns:
            Iterator of match objects
        """
        ...

    def match(self, pattern: str, text: str, flags: int = 0) -> Optional[Any]:
        """Match pattern at start of text.

        Args:
            pattern: Regex pattern
            text: Text to match
            flags: Optional regex flags

        Returns:
            Match object or None
        """
        ...

    def is_native(self) -> bool:
        """Check if this is a native (Rust) implementation."""
        ...


class BackendCapabilities:
    """Capability flags for backend implementations."""

    supports_parallel: bool = False
    supports_incremental: bool = False
    supports_batching: bool = False
    supports_async: bool = False
    is_native: bool = False
    memory_overhead_mb: int = 0


def get_backend_capabilities(backend: Any) -> BackendCapabilities:
    """Get capabilities for a backend implementation.

    Checks for capability attributes on the backend class and returns
    a BackendCapabilities object with the discovered capabilities.

    Args:
        backend: Backend implementation instance or class

    Returns:
        BackendCapabilities object
    """
    caps = BackendCapabilities()

    if hasattr(backend, "supports_parallel"):
        caps.supports_parallel = backend.supports_parallel
    if hasattr(backend, "supports_incremental"):
        caps.supports_incremental = backend.supports_incremental
    if hasattr(backend, "supports_batching"):
        caps.supports_batching = backend.supports_batching
    if hasattr(backend, "supports_async"):
        caps.supports_async = backend.supports_async
    if hasattr(backend, "is_native"):
        caps.is_native = backend.is_native
    if hasattr(backend, "memory_overhead_mb"):
        caps.memory_overhead_mb = backend.memory_overhead_mb

    return caps


__all__ = [
    "IndexedFileData",
    "ExtractedData",
    "ChunkInfo",
    "FastIndexerProtocol",
    "FastChunkerProtocol",
    "FastSymbolExtractorProtocol",
    "FastRegexProcessorProtocol",
    "BackendCapabilities",
    "get_backend_capabilities",
]
