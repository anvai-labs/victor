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

"""SOLID protocol definitions for native accelerations.

This module defines the abstract interfaces (protocols) for all native
acceleration domains. These protocols enable:

- **Single Responsibility**: Each protocol handles one concern
- **Open/Closed**: New accelerators implement existing protocols
- **Liskov Substitution**: Python and Rust implementations are interchangeable
- **Interface Segregation**: Small, focused protocols per domain
- **Dependency Inversion**: High-level code depends on protocols, not implementations

Usage:
    from victor.native.protocols import SymbolExtractorProtocol

    def index_file(extractor: SymbolExtractorProtocol, source: str):
        symbols = extractor.extract_functions(source, "python")
        # Works with both Python and Rust implementations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

# =============================================================================
# Data Types
# =============================================================================


class SymbolType(str, Enum):
    """Type of code symbol."""

    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"


@dataclass(frozen=True)
class NativeSymbol:
    """Code symbol extracted from native (Rust) parser.

    Renamed from Symbol to be semantically distinct:
    - NativeSymbol (here): Rust-extracted symbols (frozen, hashable)
    - IndexedSymbol (victor.coding.codebase.indexer): Pydantic model for index storage
    - RefactorSymbol (victor.coding.refactor.protocol): Refactoring symbol with SourceLocation

    Attributes:
        name: Symbol name (e.g., "my_function")
        type: Symbol type (function, class, method, etc.)
        line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)
        signature: Function/method signature if applicable
        docstring: Documentation string if present
        decorators: List of decorator names
        parent: Parent symbol name (e.g., class name for methods)
        visibility: "public", "private", or "protected"
    """

    name: str
    type: SymbolType
    line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    decorators: Tuple[str, ...] = ()
    parent: Optional[str] = None
    visibility: str = "public"

    def __hash__(self) -> int:
        return hash((self.name, self.type, self.line, self.parent))


# Backward compatibility alias
Symbol = NativeSymbol


@dataclass
class ChunkInfo:
    """Information about a text chunk.

    Attributes:
        text: The chunk text content
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)
        start_offset: Character offset from start of text
        end_offset: Character offset for end of chunk
        overlap_prev: Number of chars overlapping with previous chunk
        metadata: Additional chunk metadata
    """

    text: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    overlap_prev: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoercedType(str, Enum):
    """Type after coercion."""

    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    NULL = "null"
    LIST = "list"
    DICT = "dict"


@dataclass
class CoercedValue:
    """Result of type coercion.

    Attributes:
        value: The coerced value
        original: Original string value
        coerced_type: Type after coercion
        confidence: Confidence in coercion (0.0-1.0)
    """

    value: Any
    original: str
    coerced_type: CoercedType
    confidence: float = 1.0


# =============================================================================
# Base Protocol
# =============================================================================


@runtime_checkable
class NativeAcceleratorProtocol(Protocol):
    """Base protocol for all native accelerations.

    All native accelerator implementations should provide these
    introspection methods for observability and debugging.
    """

    def is_available(self) -> bool:
        """Check if this accelerator is available (compiled/loaded)."""
        ...

    def get_version(self) -> Optional[str]:
        """Get version string, or None if not available."""
        ...

    def get_backend(self) -> str:
        """Get backend identifier ("rust" or "python")."""
        ...

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this accelerator.

        Returns:
            Dictionary with keys like:
            - "calls_total": Total number of calls
            - "duration_ms_total": Total time spent
            - "duration_ms_avg": Average call duration
        """
        ...


# =============================================================================
# Domain-Specific Protocols
# =============================================================================


@runtime_checkable
class SymbolExtractorProtocol(NativeAcceleratorProtocol, Protocol):
    """Protocol for extracting symbols from source code.

    Used by the codebase indexer to build the symbol graph.
    Hot path: Called for every file during indexing.
    """

    def extract_functions(self, source: str, lang: str) -> List[Symbol]:
        """Extract function definitions from source.

        Args:
            source: Source code text
            lang: Language identifier ("python", "javascript", etc.)

        Returns:
            List of Symbol objects for functions
        """
        ...

    def extract_classes(self, source: str, lang: str) -> List[Symbol]:
        """Extract class definitions from source.

        Args:
            source: Source code text
            lang: Language identifier

        Returns:
            List of Symbol objects for classes (with methods as children)
        """
        ...

    def extract_imports(self, source: str, lang: str) -> List[str]:
        """Extract import statements from source.

        Args:
            source: Source code text
            lang: Language identifier

        Returns:
            List of imported module names
        """
        ...

    def extract_references(self, source: str) -> List[str]:
        """Extract all identifier references from source.

        Fast regex-based extraction of all identifiers.

        Args:
            source: Source code text

        Returns:
            List of identifier strings
        """
        ...

    def is_stdlib_module(self, name: str) -> bool:
        """Check if a module name is from the standard library.

        Uses perfect hashing for O(1) lookup.

        Args:
            name: Module name (e.g., "os.path")

        Returns:
            True if module is in stdlib
        """
        ...


@runtime_checkable
class ArgumentNormalizerProtocol(NativeAcceleratorProtocol, Protocol):
    """Protocol for normalizing tool call arguments.

    Hot path: Called on EVERY tool invocation to clean up
    malformed JSON from LLM outputs.
    """

    def normalize_json(self, value: str) -> Tuple[str, bool]:
        """Normalize a potentially malformed JSON string.

        Attempts multiple repair strategies:
        1. Valid JSON (fast path)
        2. AST literal eval
        3. Quote replacement
        4. Streaming repair

        Args:
            value: Potentially malformed JSON string

        Returns:
            Tuple of (normalized JSON string, success boolean)
        """
        ...

    def coerce_type(self, value: str) -> CoercedValue:
        """Coerce a string value to its likely type.

        Detects: int, float, bool, null, or keeps as string.

        Args:
            value: String value to coerce

        Returns:
            CoercedValue with detected type
        """
        ...

    def repair_quotes(self, value: str) -> str:
        """Repair mismatched or incorrect quotes in JSON.

        Handles:
        - Single quotes → double quotes
        - Unescaped quotes
        - Mixed quote styles

        Args:
            value: String with potential quote issues

        Returns:
            String with repaired quotes
        """
        ...


@runtime_checkable
class SimilarityComputerProtocol(NativeAcceleratorProtocol, Protocol):
    """Protocol for computing vector similarity.

    Uses SIMD optimizations when available for 2-5x speedup.
    """

    def cosine(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector (must have same length)

        Returns:
            Cosine similarity in range [-1, 1]

        Raises:
            ValueError: If vectors have different lengths
        """
        ...

    def batch_cosine(self, query: List[float], corpus: List[List[float]]) -> List[float]:
        """Compute cosine similarity of query against corpus.

        Args:
            query: Query vector
            corpus: List of corpus vectors

        Returns:
            List of similarity scores (same order as corpus)
        """
        ...

    def similarity_matrix(
        self,
        queries: List[List[float]],
        corpus: List[List[float]],
        normalize: bool = True,
    ) -> List[List[float]]:
        """Compute pairwise similarity matrix.

        Args:
            queries: List of query vectors
            corpus: List of corpus vectors
            normalize: Whether to L2-normalize vectors first

        Returns:
            Matrix of shape (len(queries), len(corpus))
        """
        ...

    def top_k(
        self, query: List[float], corpus: List[List[float]], k: int
    ) -> List[Tuple[int, float]]:
        """Find top-k most similar vectors.

        Args:
            query: Query vector
            corpus: List of corpus vectors
            k: Number of top results

        Returns:
            List of (index, similarity) tuples, sorted by similarity desc
        """
        ...


@runtime_checkable
class TextChunkerProtocol(NativeAcceleratorProtocol, Protocol):
    """Protocol for chunking text with line awareness.

    Optimized for code chunking where line boundaries matter.
    """

    def chunk_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[ChunkInfo]:
        """Chunk text with overlap, respecting line boundaries.

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in characters
            overlap: Overlap size in characters

        Returns:
            List of ChunkInfo objects
        """
        ...

    def count_lines(self, text: str) -> int:
        """Count lines in text (fast).

        Args:
            text: Text to count lines in

        Returns:
            Number of lines (newline count + 1)
        """
        ...

    def find_line_boundaries(self, text: str) -> List[int]:
        """Find byte offsets of all line starts.

        Args:
            text: Text to analyze

        Returns:
            List of byte offsets where lines start (including 0)
        """
        ...

    def line_at_offset(self, text: str, offset: int) -> int:
        """Get line number for a character offset.

        Args:
            text: Text
            offset: Character offset

        Returns:
            Line number (1-indexed)
        """
        ...


@runtime_checkable
class AstIndexerProtocol(NativeAcceleratorProtocol, Protocol):
    """Protocol for AST-based code indexing acceleration.

    Optimizes the hot paths in codebase indexing:
    - stdlib module detection (called per import)
    - identifier reference extraction (regex fallback)

    These operations are called thousands of times during indexing,
    making them prime candidates for Rust acceleration.
    """

    def is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module name is a standard library module.

        Uses a perfect hash or phf for O(1) lookup of 130+ stdlib modules.

        Args:
            module_name: Full module name (e.g., "os.path", "typing")

        Returns:
            True if the module is in the stdlib or common third-party set
        """
        ...

    def batch_is_stdlib_modules(self, module_names: List[str]) -> List[bool]:
        """Check multiple module names for stdlib membership.

        Batch version for efficiency when processing many imports.

        Args:
            module_names: List of module names to check

        Returns:
            List of booleans, one per module name
        """
        ...

    def extract_identifiers(self, source: str) -> List[str]:
        """Extract all identifier references from source code.

        Uses regex pattern [A-Za-z_][A-Za-z0-9_]* optimized with SIMD.

        Args:
            source: Source code text

        Returns:
            List of unique identifiers found
        """
        ...

    def extract_identifiers_with_positions(self, source: str) -> List[Tuple[str, int, int]]:
        """Extract identifiers with their positions.

        Args:
            source: Source code text

        Returns:
            List of (identifier, start_offset, end_offset) tuples
        """
        ...

    def filter_stdlib_imports(self, imports: List[str]) -> Tuple[List[str], List[str]]:
        """Partition imports into stdlib and non-stdlib.

        Args:
            imports: List of import module names

        Returns:
            Tuple of (stdlib_imports, non_stdlib_imports)
        """
        ...


@runtime_checkable
class TokenCounterProtocol(NativeAcceleratorProtocol, Protocol):
    """Protocol for counting tokens in text.

    Provides exact and approximate token counting using BPE tokenization.
    Hot path: Called for every message during context window management.
    """

    def count_tokens(self, text: str) -> int:
        """Count tokens using exact BPE tokenization.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        ...

    def count_tokens_fast(self, text: str) -> int:
        """Count tokens using fast approximate method.

        Optimized for speed over accuracy.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate number of tokens
        """
        ...

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts in batch.

        More efficient than calling count_tokens() in a loop.

        Args:
            texts: List of texts to count tokens for

        Returns:
            List of token counts, one per input text
        """
        ...


@runtime_checkable
class ContextFitterProtocol(NativeAcceleratorProtocol, Protocol):
    """Protocol for fitting messages into a context window budget.

    Provides message selection and truncation for context window management.
    Hot path: Called on every LLM invocation to manage context.
    """

    def fit_context(
        self,
        messages: List[Dict[str, Any]],
        budget: int,
        strategy: str = "recency",
        preserve_system: bool = True,
    ) -> Any:
        """Fit messages into a token budget.

        Args:
            messages: List of message dicts with 'role', 'content', and
                      optionally 'token_count' and 'priority' fields
            budget: Maximum token budget
            strategy: Fitting strategy ("recency", "priority", "balanced")
            preserve_system: Whether to always preserve system messages

        Returns:
            FitResult with indices of kept messages and statistics
        """
        ...

    def truncate_message(
        self,
        content: str,
        max_tokens: int,
        preserve_lines: bool = True,
    ) -> str:
        """Truncate a message to fit within a token limit.

        Args:
            content: Message content to truncate
            max_tokens: Maximum number of tokens allowed
            preserve_lines: Whether to truncate at line boundaries

        Returns:
            Truncated content string
        """
        ...


@runtime_checkable
class ContentHasherProtocol(NativeAcceleratorProtocol, Protocol):
    """Protocol for content hashing with configurable normalization.

    Provides cross-vertical content hashing for:
    - Text deduplication (OutputDeduplicator)
    - Tool call deduplication (ToolDeduplicationTracker)
    - Cache key generation
    - Change detection

    This protocol abstracts over Python and Rust implementations,
    using native acceleration when available.
    """

    def hash(self, content: str) -> str:
        """Generate hash for content with configured normalization.

        Args:
            content: Content to hash

        Returns:
            Hexadecimal hash string
        """
        ...

    def hash_dict(self, data: Dict[str, Any]) -> str:
        """Hash dictionary with sorted keys for deterministic output.

        Args:
            data: Dictionary to hash (must be JSON-serializable)

        Returns:
            Hexadecimal hash string
        """
        ...

    def hash_list(self, items: List[Any]) -> str:
        """Hash list with sorted items for deterministic output.

        Args:
            items: List to hash

        Returns:
            Hexadecimal hash string
        """
        ...

    def hash_block(self, block: str, min_length: int = 0) -> Optional[str]:
        """Hash content block with optional minimum length check.

        Args:
            block: Content block to hash
            min_length: Minimum length required (0 = no minimum)

        Returns:
            Hash string if block meets minimum length, None otherwise
        """
        ...

    def normalize(self, content: str) -> str:
        """Normalize content without hashing (for inspection/debugging).

        Args:
            content: Content to normalize

        Returns:
            Normalized content string
        """
        ...
