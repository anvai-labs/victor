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

"""Backend factory for creating performance-improved instances with fallback.

This module provides the BackendFactory class which simplifies creating backend
instances with automatic fallback from native (Rust) to Python implementations.

Usage:
    # Create indexer, trying native first
    indexer = BackendFactory.create_indexer(config)

    # Create chunker with specific preference
    chunker = BackendFactory.create_chunker(
        config,
        prefer_native=True
    )

    # Check if native backends are available
    if BackendFactory.has_native_indexer():
        use_optimized_path()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Type, TypeVar

from victor_coding.performance.protocols import (
    BackendCapabilities,
    FastChunkerProtocol,
    FastIndexerProtocol,
    FastRegexProcessorProtocol,
    FastSymbolExtractorProtocol,
    get_backend_capabilities,
)
from victor_coding.performance.registry import (
    BackendConfig,
    PerformanceBackendRegistry,
    auto_register_native_backends,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BackendFactory:
    """Factory for creating backend instances with automatic fallback.

    This factory encapsulates the logic for:
    1. Trying native (Rust) implementations first
    2. Falling back to pure Python implementations
    3. Logging which implementation was selected
    4. Providing capability information
    """

    _native_checked: bool = False
    _native_available: Dict[Type, bool] = {}

    @classmethod
    def _ensure_native_checked(cls) -> None:
        """Ensure native backends have been checked and registered."""
        if not cls._native_checked:
            auto_register_native_backends()
            cls._native_checked = True

    @classmethod
    def _check_native_availability(cls, protocol: Type) -> bool:
        """Check if native backend is available for a protocol."""
        cls._ensure_native_checked()

        if protocol not in cls._native_available:
            cls._native_available[protocol] = PerformanceBackendRegistry.get_native_available(
                protocol
            )

        return cls._native_available[protocol]

    @classmethod
    def has_native_indexer(cls) -> bool:
        """Check if native tree-sitter indexer is available."""
        return cls._check_native_availability(FastIndexerProtocol)

    @classmethod
    def has_native_chunker(cls) -> bool:
        """Check if native chunker is available."""
        return cls._check_native_availability(FastChunkerProtocol)

    @classmethod
    def has_native_regex(cls) -> bool:
        """Check if native regex processor is available."""
        return cls._check_native_availability(FastRegexProcessorProtocol)

    @classmethod
    def has_native_extractor(cls) -> bool:
        """Check if native symbol extractor is available."""
        return cls._check_native_availability(FastSymbolExtractorProtocol)

    @classmethod
    def create_indexer(
        cls,
        config: Any,
        *,
        prefer_native: bool = True,
        fallback_to_python: bool = True,
    ) -> Any:
        """Create an indexer backend, trying native first.

        Args:
            config: Configuration object for the indexer
            prefer_native: Whether to prefer native implementation
            fallback_to_python: Whether to fall back to Python on error

        Returns:
            Indexer instance (native or Python)
        """
        cls._ensure_native_checked()

        if prefer_native and cls.has_native_indexer():
            try:
                indexer = PerformanceBackendRegistry.create(
                    FastIndexerProtocol,
                    config,
                    backend_config=BackendConfig(use_native=True),
                )
                if indexer is not None:
                    logger.info("Using native tree-sitter indexer")
                    return indexer
            except Exception as e:
                logger.warning(f"Native indexer unavailable: {e}")
                if not fallback_to_python:
                    raise

        # Fallback to Python
        logger.debug("Using Python indexer")
        return cls._create_python_indexer(config)

    @classmethod
    def create_chunker(
        cls,
        config: Any,
        *,
        prefer_native: bool = True,
        fallback_to_python: bool = True,
    ) -> Any:
        """Create a chunker backend, trying native first.

        Args:
            config: Configuration object for the chunker
            prefer_native: Whether to prefer native implementation
            fallback_to_python: Whether to fall back to Python on error

        Returns:
            Chunker instance (native or Python)
        """
        cls._ensure_native_checked()

        if prefer_native and cls.has_native_chunker():
            try:
                chunker = PerformanceBackendRegistry.create(
                    FastChunkerProtocol,
                    config,
                    backend_config=BackendConfig(use_native=True),
                )
                if chunker is not None:
                    logger.info("Using native chunker")
                    return chunker
            except Exception as e:
                logger.warning(f"Native chunker unavailable: {e}")
                if not fallback_to_python:
                    raise

        # Fallback to Python
        logger.debug("Using Python chunker")
        return cls._create_python_chunker(config)

    @classmethod
    def create_symbol_extractor(
        cls,
        config: Any,
        *,
        prefer_native: bool = True,
        fallback_to_python: bool = True,
    ) -> Any:
        """Create a symbol extractor backend, trying native first.

        Args:
            config: Configuration object for the extractor
            prefer_native: Whether to prefer native implementation
            fallback_to_python: Whether to fall back to Python on error

        Returns:
            Extractor instance (native or Python)
        """
        cls._ensure_native_checked()

        if prefer_native and cls.has_native_extractor():
            try:
                extractor = PerformanceBackendRegistry.create(
                    FastSymbolExtractorProtocol,
                    config,
                    backend_config=BackendConfig(use_native=True),
                )
                if extractor is not None:
                    logger.info("Using native symbol extractor")
                    return extractor
            except Exception as e:
                logger.warning(f"Native extractor unavailable: {e}")
                if not fallback_to_python:
                    raise

        # Fallback to Python
        logger.debug("Using Python symbol extractor")
        return cls._create_python_symbol_extractor(config)

    @classmethod
    def create_regex_processor(
        cls,
        config: Any = None,
        *,
        prefer_native: bool = True,
        fallback_to_python: bool = True,
    ) -> Any:
        """Create a regex processor, trying native first.

        Args:
            config: Configuration object (optional)
            prefer_native: Whether to prefer native implementation
            fallback_to_python: Whether to fall back to Python on error

        Returns:
            Regex processor instance (native or Python)
        """
        cls._ensure_native_checked()

        if prefer_native and cls.has_native_regex():
            try:
                processor = PerformanceBackendRegistry.create(
                    FastRegexProcessorProtocol,
                    config,
                    backend_config=BackendConfig(use_native=True),
                )
                if processor is not None:
                    logger.info("Using native regex processor")
                    return processor
            except Exception as e:
                logger.warning(f"Native regex processor unavailable: {e}")
                if not fallback_to_python:
                    raise

        # Fallback to Python re module
        logger.debug("Using Python regex processor")
        return cls._create_python_regex_processor(config)

    @classmethod
    def get_backend_capabilities(cls, backend: Any) -> BackendCapabilities:
        """Get capabilities for a backend instance.

        Args:
            backend: Backend instance

        Returns:
            BackendCapabilities object
        """
        return get_backend_capabilities(backend)

    @classmethod
    def _create_python_indexer(cls, config: Any) -> Any:
        """Create the standard Python indexer.

        Args:
            config: Configuration object

        Returns:
            Python CodebaseIndex instance
        """
        from victor_coding.codebase.indexer import CodebaseIndex

        return CodebaseIndex(config)

    @classmethod
    def _create_python_chunker(cls, config: Any) -> Any:
        """Create the standard Python chunker.

        Args:
            config: Configuration object

        Returns:
            Python CodeChunker instance
        """
        from victor_coding.codebase.chunker import CodeChunker

        return CodeChunker(config)

    @classmethod
    def _create_python_symbol_extractor(cls, config: Any) -> Any:
        """Create the standard Python symbol extractor.

        Args:
            config: Configuration object

        Returns:
            Python TreeSitterExtractor instance
        """
        from victor_coding.codebase.tree_sitter_extractor import TreeSitterExtractor

        return TreeSitterExtractor(config)

    @classmethod
    def _create_python_regex_processor(cls, config: Any = None) -> Any:
        """Create a wrapper around Python's re module.

        Args:
            config: Configuration object (ignored)

        Returns:
            PythonRegexProcessor wrapper
        """
        return PythonRegexProcessor()


class PythonRegexProcessor:
    """Wrapper around Python's re module implementing FastRegexProcessorProtocol.

    This allows the regex processor to be used interchangeably with the
    native Rust implementation.
    """

    priority: int = 50

    def __init__(self) -> None:
        import re

        self._re = re
        self.is_native_impl = False

    def findall(self, pattern: str, text: str, flags: int = 0) -> List[str]:
        """Find all matches of pattern in text."""
        return self._re.findall(pattern, text, flags)

    def finditer(self, pattern: str, text: str, flags: int = 0):
        """Find all matches as an iterator."""
        return self._re.finditer(pattern, text, flags)

    def match(self, pattern: str, text: str, flags: int = 0):
        """Match pattern at start of text."""
        return self._re.match(pattern, text, flags)

    def search(self, pattern: str, text: str, flags: int = 0):
        """Search for pattern anywhere in text."""
        return self._re.search(pattern, text, flags)

    def sub(self, pattern: str, repl: str, text: str, flags: int = 0) -> str:
        """Replace pattern with replacement in text."""
        return self._re.sub(pattern, repl, text, flags)

    def is_native(self) -> bool:
        """Check if this is a native implementation."""
        return self.is_native_impl


__all__ = [
    "BackendFactory",
    "PythonRegexProcessor",
]
