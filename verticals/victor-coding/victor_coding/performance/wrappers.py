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

"""Wrapper classes for integrating performance backends with existing code.

These wrapper classes adapt the performance backend protocols to work seamlessly
with the existing CodebaseIndex, CodeChunker, and related classes.

Usage:
    # Use wrapped indexer with automatic fallback
    from victor_coding.performance.wrappers import WrappedIndexer

    indexer = WrappedIndexer(config)
    # Will use native if available, otherwise Python
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from victor_coding.performance.factory import BackendFactory
from victor_coding.performance.protocols import (
    ChunkInfo,
    IndexedFileData,
)

logger = logging.getLogger(__name__)


class WrappedIndexer:
    """Wrapper for indexer that uses performance backend when available.

    This class wraps either a native (Rust) or Python indexer, providing
    a unified interface that automatically selects the best available implementation.

    The wrapper maintains compatibility with CodebaseIndex while potentially
    using a high-performance native backend.
    """

    def __init__(self, config: Any, backend: Optional[Any] = None):
        """Initialize the wrapped indexer.

        Args:
            config: Configuration object for the indexer
            backend: Optional specific backend to use (auto-selected if None)
        """
        self._config = config
        self._backend = backend
        self._is_native: bool = False

        if backend is None:
            # Auto-select backend
            self._backend = BackendFactory.create_indexer(config)
        else:
            self._backend = backend

        # Check if native
        if hasattr(self._backend, "is_native"):
            self._is_native = self._backend.is_native
        else:
            self._is_native = False

        if self._is_native:
            logger.info("Using native indexer backend")
        else:
            logger.debug("Using Python indexer backend")

    @property
    def backend(self) -> Any:
        """Get the underlying backend instance."""
        return self._backend

    @property
    def is_native(self) -> bool:
        """Check if using native backend."""
        return self._is_native

    async def index_file(
        self, file_path: Union[str, Path], root: Union[str, Path]
    ) -> Optional[IndexedFileData]:
        """Index a single file.

        Args:
            file_path: Path to the file
            root: Root directory of the codebase

        Returns:
            IndexedFileData or None
        """
        if hasattr(self._backend, "index_file"):
            return await self._backend.index_file(file_path, root)

        # Fallback for Python CodebaseIndex
        return await self._backend._process_file_parallel(
            str(file_path), str(root), self._backend._detect_language(file_path)
        )

    async def index_batch(
        self, file_paths: List[Union[str, Path]], root: Union[str, Path]
    ) -> List[Optional[IndexedFileData]]:
        """Index multiple files.

        Args:
            file_paths: List of file paths
            root: Root directory

        Returns:
            List of IndexedFileData
        """
        if hasattr(self._backend, "index_batch"):
            return await self._backend.index_batch(file_paths, root)

        # Fallback: sequential processing
        results = []
        for file_path in file_paths:
            result = await self.index_file(file_path, root)
            results.append(result)
        return results

    def supports_language(self, language: str) -> bool:
        """Check if language is supported.

        Args:
            language: Language name

        Returns:
            True if supported
        """
        if hasattr(self._backend, "supports_language"):
            return self._backend.supports_language(language)

        # Fallback for Python CodebaseIndex
        return language in self._backend.SYMBOL_QUERIES or language in ["python"]

    def get_supported_languages(self) -> List[str]:
        """Get supported languages.

        Returns:
            List of language names
        """
        if hasattr(self._backend, "get_supported_languages"):
            return self._backend.get_supported_languages()

        # Fallback for Python CodebaseIndex
        return list(self._backend.SYMBOL_QUERIES.keys()) + ["python"]


class WrappedChunker:
    """Wrapper for chunker that uses performance backend when available.

    This class wraps either a native (Rust) or Python chunker, providing
    a unified interface that automatically selects the best available implementation.
    """

    def __init__(self, config: Any, backend: Optional[Any] = None):
        """Initialize the wrapped chunker.

        Args:
            config: Configuration object for the chunker
            backend: Optional specific backend to use (auto-selected if None)
        """
        self._config = config
        self._backend = backend
        self._is_native: bool = False

        if backend is None:
            # Auto-select backend
            self._backend = BackendFactory.create_chunker(config)
        else:
            self._backend = backend

        # Check if native
        if hasattr(self._backend, "is_native"):
            self._is_native = self._backend.is_native
        else:
            self._is_native = False

        if self._is_native:
            logger.info("Using native chunker backend")
        else:
            logger.debug("Using Python chunker backend")

    @property
    def backend(self) -> Any:
        """Get the underlying backend instance."""
        return self._backend

    @property
    def is_native(self) -> bool:
        """Check if using native backend."""
        return self._is_native

    def chunk_code(
        self, content: str, language: str, file_path: Optional[str] = None
    ) -> List[ChunkInfo]:
        """Chunk code content.

        Args:
            content: Source code
            language: Programming language
            file_path: Optional file path

        Returns:
            List of chunks
        """
        if hasattr(self._backend, "chunk_code"):
            return self._backend.chunk_code(content, language, file_path)

        # Fallback for Python CodeChunker
        return self._backend.chunk_text(content, language)

    def chunk_file(
        self, file_path: Union[str, Path], language: Optional[str] = None
    ) -> List[ChunkInfo]:
        """Chunk a file.

        Args:
            file_path: Path to file
            language: Optional language (auto-detected if None)

        Returns:
            List of chunks
        """
        if hasattr(self._backend, "chunk_file"):
            return self._backend.chunk_file(file_path, language)

        # Fallback for Python CodeChunker
        content = Path(file_path).read_text(encoding="utf-8")
        if language is None:
            language = self._backend.detect_language(str(file_path))
        return self.chunk_code(content, language, str(file_path))

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if hasattr(self._backend, "estimate_tokens"):
            return self._backend.estimate_tokens(text)

        # Fallback heuristic
        return len(text) // 4


class WrappedExtractor:
    """Wrapper for symbol extractor that uses performance backend when available.

    This class wraps either a native (Rust) or Python symbol extractor.
    """

    def __init__(self, config: Any = None, backend: Optional[Any] = None):
        """Initialize the wrapped extractor.

        Args:
            config: Optional configuration object
            backend: Optional specific backend to use (auto-selected if None)
        """
        self._config = config
        self._backend = backend
        self._is_native: bool = False

        if backend is None:
            # Auto-select backend
            self._backend = BackendFactory.create_symbol_extractor(config)
        else:
            self._backend = backend

        # Check if native
        if hasattr(self._backend, "is_native"):
            self._is_native = self._backend.is_native
        else:
            self._is_native = False

        if self._is_native:
            logger.info("Using native symbol extractor backend")
        else:
            logger.debug("Using Python symbol extractor backend")

    @property
    def backend(self) -> Any:
        """Get the underlying backend instance."""
        return self._backend

    @property
    def is_native(self) -> bool:
        """Check if using native backend."""
        return self._is_native

    def extract_all(self, file_path: Union[str, Path], language: str) -> Optional[Dict[str, Any]]:
        """Extract all symbols and edges.

        Args:
            file_path: Path to file
            language: Programming language

        Returns:
            Extracted data dictionary or None
        """
        if hasattr(self._backend, "extract_all"):
            result = self._backend.extract_all(file_path, language)
            if result is not None:
                # Convert ExtractedData to dict if needed
                if hasattr(result, "__dict__"):
                    return {
                        "symbols": result.symbols,
                        "call_edges": result.call_edges,
                        "inherit_edges": result.inherit_edges,
                        "implements_edges": result.implements_edges,
                        "compose_edges": result.compose_edges,
                    }
                return result
            return None

        # Fallback: use separate extraction methods
        return self._extract_all_fallback(file_path, language)

    def _extract_all_fallback(
        self, file_path: Union[str, Path], language: str
    ) -> Optional[Dict[str, Any]]:
        """Fallback extraction using separate methods.

        Args:
            file_path: Path to file
            language: Programming language

        Returns:
            Extracted data dictionary
        """
        symbols = self.extract_symbols(file_path, language)
        call_edges = self.extract_call_edges(file_path, language)

        return {
            "symbols": symbols,
            "call_edges": call_edges,
            "inherit_edges": [],
            "implements_edges": [],
            "compose_edges": [],
        }

    def extract_symbols(self, file_path: Union[str, Path], language: str) -> List[Dict[str, Any]]:
        """Extract symbols only.

        Args:
            file_path: Path to file
            language: Programming language

        Returns:
            List of symbol dictionaries
        """
        if hasattr(self._backend, "extract_symbols"):
            return self._backend.extract_symbols(file_path, language)

        # Fallback for Python TreeSitterExtractor

        result = self._backend.extract_symbols(Path(file_path), language)
        return [
            {
                "name": s.name,
                "type": s.type,
                "file_path": s.file_path,
                "line_number": s.line_number,
                "end_line": s.end_line,
                "parent_symbol": s.parent_symbol,
            }
            for s in result
        ]

    def extract_call_edges(self, file_path: Union[str, Path], language: str) -> List[tuple]:
        """Extract call edges.

        Args:
            file_path: Path to file
            language: Programming language

        Returns:
            List of (caller, callee, line_number) tuples
        """
        if hasattr(self._backend, "extract_call_edges"):
            return self._backend.extract_call_edges(file_path, language)

        # Fallback for Python TreeSitterExtractor

        result = self._backend.extract_call_edges(Path(file_path), language)
        return [(e.source, e.target, e.line_number) for e in result]


__all__ = [
    "WrappedIndexer",
    "WrappedChunker",
    "WrappedExtractor",
]
