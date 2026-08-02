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

"""Shared fixtures for native acceleration tests.

These fixtures enable TDD by providing:
1. Parameterized backends (python/rust) for equivalence testing
2. Sample data for each accelerator type
3. Metrics reset between tests
"""

from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.native.protocols import (
        SymbolExtractorProtocol,
        ArgumentNormalizerProtocol,
        SimilarityComputerProtocol,
        TextChunkerProtocol,
        AstIndexerProtocol,
    )


# =============================================================================
# Helper to check if Rust is available
# =============================================================================


def is_rust_available() -> bool:
    """Check if Rust native extensions are compiled and available."""
    try:
        from victor.processing.native import is_native_available

        return is_native_available()
    except ImportError:
        return False


# =============================================================================
# Metrics Reset
# =============================================================================


@pytest.fixture(autouse=True)
def reset_native_metrics():
    """Reset native metrics between tests."""
    from victor.native.observability import NativeMetrics

    NativeMetrics.reset_instance()
    yield
    NativeMetrics.reset_instance()


# =============================================================================
# Backend Parametrization
# =============================================================================


def get_backends():
    """Get available backends for parametrization."""
    backends = ["python"]
    if is_rust_available():
        backends.append("rust")
    return backends


@pytest.fixture(params=get_backends())
def backend(request) -> str:
    """Parametrized backend fixture.

    Tests using this fixture run once per available backend.
    """
    return request.param


# =============================================================================
# Symbol Extractor Fixtures
# =============================================================================


@pytest.fixture
def symbol_extractor(backend: str) -> "SymbolExtractorProtocol":
    """Get a symbol extractor for the given backend."""
    if backend == "rust":
        if not is_rust_available():
            pytest.skip("Rust backend not available")
        # TODO: Return Rust implementation when available
        pytest.skip("Rust symbol extractor not yet implemented")
    else:
        from victor.native.python.symbol_extractor import PythonSymbolExtractor

        return PythonSymbolExtractor()


@pytest.fixture
def sample_python_source() -> str:
    """Sample Python source code for testing."""
    return '''
"""Module docstring."""

import os
from typing import List, Optional

CONSTANT = 42

class MyClass:
    """A sample class."""

    def __init__(self, value: int) -> None:
        """Initialize with value."""
        self._value = value

    @property
    def value(self) -> int:
        """Get the value."""
        return self._value

    def _private_method(self) -> None:
        """Private method."""
        pass


def public_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


async def async_function(data: List[str]) -> Optional[str]:
    """Async function example."""
    if not data:
        return None
    return data[0]
'''


# =============================================================================
# Argument Normalizer Fixtures
# =============================================================================


@pytest.fixture
def arg_normalizer(backend: str) -> "ArgumentNormalizerProtocol":
    """Get an argument normalizer for the given backend."""
    if backend == "rust":
        if not is_rust_available():
            pytest.skip("Rust backend not available")
        from victor.native.rust.arg_normalizer import RustArgumentNormalizer

        return RustArgumentNormalizer()
    else:
        from victor.native.python.arg_normalizer import PythonArgumentNormalizer

        return PythonArgumentNormalizer()


@pytest.fixture
def malformed_json_samples() -> list[tuple[str, str]]:
    """Samples of malformed JSON with expected repairs."""
    return [
        # (malformed, expected_valid_json)
        ('{"key": "value"}', '{"key": "value"}'),  # Already valid
        ("{'key': 'value'}", '{"key": "value"}'),  # Single quotes
        ('{"key": "value",}', '{"key": "value"}'),  # Trailing comma
        ('{key: "value"}', '{"key": "value"}'),  # Unquoted key
    ]


# =============================================================================
# Similarity Computer Fixtures
# =============================================================================


@pytest.fixture
def similarity_computer(backend: str) -> "SimilarityComputerProtocol":
    """Get a similarity computer for the given backend."""
    if backend == "rust":
        if not is_rust_available():
            pytest.skip("Rust backend not available")
        from victor.native.rust.similarity import RustSimilarityComputer

        return RustSimilarityComputer()
    else:
        from victor.native.python.similarity import PythonSimilarityComputer

        return PythonSimilarityComputer()


@pytest.fixture
def sample_vectors() -> dict:
    """Sample vectors for similarity testing."""
    return {
        "unit_x": [1.0, 0.0, 0.0],
        "unit_y": [0.0, 1.0, 0.0],
        "unit_z": [0.0, 0.0, 1.0],
        "diagonal": [1.0, 1.0, 1.0],
        "neg_x": [-1.0, 0.0, 0.0],
        "zero": [0.0, 0.0, 0.0],
        "embedding_a": [0.1, 0.2, 0.3, 0.4, 0.5],
        "embedding_b": [0.1, 0.2, 0.3, 0.4, 0.6],  # Similar to a
        "embedding_c": [-0.5, -0.4, -0.3, -0.2, -0.1],  # Opposite to a
    }


# =============================================================================
# Text Chunker Fixtures
# =============================================================================


@pytest.fixture
def text_chunker(backend: str) -> "TextChunkerProtocol":
    """Get a text chunker for the given backend."""
    if backend == "rust":
        if not is_rust_available():
            pytest.skip("Rust backend not available")
        from victor.native.rust.chunker import RustTextChunker

        return RustTextChunker()
    else:
        from victor.native.python.chunker import PythonTextChunker

        return PythonTextChunker()


@pytest.fixture
def sample_multiline_text() -> str:
    """Sample multiline text for chunking tests."""
    return """Line 1: First line of text
Line 2: Second line with more content
Line 3: Third line
Line 4: Fourth line
Line 5: Fifth line
Line 6: Sixth line with even more content here
Line 7: Seventh line
Line 8: Eighth line
Line 9: Ninth line
Line 10: Tenth and final line"""


# =============================================================================
# Metrics Fixtures
# =============================================================================


@pytest.fixture
def native_metrics():
    """Get the NativeMetrics instance for testing."""
    from victor.native.observability import NativeMetrics

    return NativeMetrics.get_instance()


# =============================================================================
# AST Indexer Fixtures (Phase 2)
# =============================================================================


@pytest.fixture
def ast_indexer(backend: str) -> "AstIndexerProtocol":
    """Get an AST indexer for the given backend.

    The AST indexer accelerates hot paths in codebase indexing:
    - is_stdlib_module(): O(1) stdlib lookup
    - extract_identifiers(): SIMD regex extraction
    """
    if backend == "rust":
        if not is_rust_available():
            pytest.skip("Rust backend not available")
        from victor.native.rust.ast_indexer import RustAstIndexer

        return RustAstIndexer()
    else:
        from victor.native.python.ast_indexer import PythonAstIndexer

        return PythonAstIndexer()
