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

"""Robust code chunking strategies for semantic search.

This module provides intelligent chunking of code for embedding generation.
It uses AST-aware parsing to respect code boundaries and creates hierarchical
embeddings for better search accuracy.

Chunking Strategies:
-------------------
1. SYMBOL_ONLY: One embedding per function/class (fast, memory-efficient)
2. BODY_AWARE: Chunks large functions into overlapping segments
3. HIERARCHICAL: Multi-level embeddings (file → class → method → body)

Rationale:
---------
- **Why not fixed-size chunks?** Code has semantic boundaries (functions, classes).
  Splitting mid-function loses context and creates orphan chunks.

- **Why overlapping chunks for large functions?** A 200-line function searching
  for "error handling" might miss matches if the relevant code is in chunk 2
  but the error variable was defined in chunk 1.

- **Why hierarchical?** Searching "authentication" should find:
  1. auth.py (file-level match)
  2. UserAuth class (class-level match)
  3. login() method (method-level match)
  4. Specific code lines (body-level match)

Configuration:
-------------
- MAX_CHUNK_TOKENS: Target chunk size in tokens (~4 chars/token)
- OVERLAP_TOKENS: Overlap between chunks for context preservation
- LARGE_SYMBOL_THRESHOLD: Lines above which to apply body chunking
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.native.protocols import ChunkInfo, TextChunkerProtocol

logger = logging.getLogger(__name__)

# TD-CG / ADR-014: shared code->CPG chunker. Soft import so the legacy in-repo
# tree-sitter path stays the fallback when the package is absent.
try:
    import victor_codegraph as _victor_codegraph
except Exception:  # ImportError, or a partial/native load failure
    _victor_codegraph = None

# Language detection mapping (file extension -> language name)
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".pyx": "python",  # Cython
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".r": "r",
    ".R": "r",
    ".jl": "julia",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".md": "markdown",
    ".rst": "restructuredtext",
}


def detect_language(file_path: str) -> str:
    """Detect programming language from file extension.

    Args:
        file_path: Path to the file

    Returns:
        Language name (e.g., "python", "javascript", "go")
    """
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, "unknown")


class ChunkType(Enum):
    """Types of code chunks for hierarchical embeddings."""

    FILE_SUMMARY = "file_summary"  # High-level file description
    CLASS_SUMMARY = "class_summary"  # Class-level overview
    METHOD_HEADER = "method_header"  # Function signature + docstring
    METHOD_BODY = "method_body"  # Code implementation chunks
    IMPORT_BLOCK = "import_block"  # Import statements
    MODULE_DOCSTRING = "module_docstring"  # Module-level docstring


class ChunkingStrategy(Enum):
    """Available chunking strategies."""

    SYMBOL_ONLY = "symbol_only"  # One chunk per symbol (default)
    BODY_AWARE = "body_aware"  # Chunk large function bodies
    HIERARCHICAL = "hierarchical"  # Full hierarchy (file → class → method → body)


@dataclass
class ChunkConfig:
    """Configuration for chunking behavior.

    Token estimation uses 3.5 chars/token (conservative vs 4x industry norm)
    to prevent truncation with dense code (short identifiers, operators).
    """

    strategy: ChunkingStrategy = ChunkingStrategy.BODY_AWARE
    max_chunk_tokens: int = 512  # Embedding model token limit
    overlap_tokens: int = 64  # Overlap between chunks for context
    large_symbol_threshold: int = 30  # Lines above which to chunk bodies
    include_file_summary: bool = True  # Add file-level embedding
    include_class_summary: bool = True  # Add class-level embeddings
    chars_per_token: float = 3.5  # Conservative (3.5x vs 4x norm) to prevent truncation

    @property
    def max_chunk_chars(self) -> int:
        """Max characters per chunk (512 tokens * 3.5 = 1792 chars)."""
        return int(self.max_chunk_tokens * self.chars_per_token)

    @property
    def overlap_chars(self) -> int:
        """Overlap characters (64 tokens * 3.5 = 224 chars)."""
        return int(self.overlap_tokens * self.chars_per_token)


@dataclass
class CodeChunk:
    """A chunk of code ready for embedding.

    ID Format (hierarchical for disambiguation):
        - File summary: `module/path:__file__`
        - Class: `module/path:ClassName`
        - Method: `module/path:ClassName.method_name`
        - Body chunk: `module/path:ClassName.method_name:chunk_idx`
        - Line range: `module/path:symbol:L10-L50`
        - Window: `module/path:window:chunk_idx`

    Attributes:
        id: Unique hierarchical identifier (docid/symbolid:chunkid)
        content: Text content for embedding (max ~1792 chars @ 3.5x)
        chunk_type: Type of chunk (FILE_SUMMARY, METHOD_HEADER, etc.)
        file_path: Source file path (relative)
        symbol_name: Symbol name if applicable
        symbol_type: function, class, method, etc.
        line_start: Starting line number (1-indexed)
        line_end: Ending line number (1-indexed)
        parent_id: Parent chunk ID for hierarchy
        metadata: Additional metadata for filtering
    """

    id: str
    content: str
    chunk_type: ChunkType
    file_path: str
    symbol_name: Optional[str] = None
    symbol_type: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_document(self) -> Dict[str, Any]:
        """Convert to document format for indexing."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": {
                "chunk_type": self.chunk_type.value,
                "file_path": self.file_path,
                "symbol_name": self.symbol_name,
                "symbol_type": self.symbol_type,
                "line_start": self.line_start,
                "line_end": self.line_end,
                "parent_id": self.parent_id,
                **self.metadata,
            },
        }


class CodeChunker:
    """Intelligent code chunker with AST-aware parsing.

    This chunker respects code boundaries (functions, classes) and creates
    hierarchical embeddings for accurate semantic search.

    Example:
        chunker = CodeChunker(config=ChunkConfig(strategy=ChunkingStrategy.BODY_AWARE))
        chunks = chunker.chunk_file(Path("src/auth.py"), "src/auth.py")

        # Result: List of CodeChunks with different types
        # - FILE_SUMMARY: "File: auth.py - User authentication module..."
        # - CLASS_SUMMARY: "Class UserAuth - Manages user sessions..."
        # - METHOD_HEADER: "def login(username, password): Authenticate user..."
        # - METHOD_BODY: "# Implementation chunk 1 of 3..."
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        """Initialize chunker with configuration.

        Args:
            config: Chunking configuration (defaults to BODY_AWARE strategy)
        """
        self.config = config or ChunkConfig()

    def chunk_file(
        self,
        file_path: Path,
        relative_path: str,
        content: Optional[str] = None,
    ) -> List[CodeChunk]:
        """Chunk a Python file into embeddable chunks.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path for IDs and metadata
            content: Optional file content (reads file if not provided)

        Returns:
            List of CodeChunk objects ready for embedding
        """
        if content is None:
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                return []

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []

        chunks: List[CodeChunk] = []
        lines = content.split("\n")

        # File-level summary (if enabled)
        if self.config.include_file_summary:
            file_chunk = self._create_file_summary(relative_path, tree, lines)
            if file_chunk:
                chunks.append(file_chunk)

        # Process top-level nodes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                chunks.extend(self._chunk_class(node, relative_path, lines))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                chunks.extend(self._chunk_function(node, relative_path, lines, parent_id=None))

        # Import block (useful for dependency analysis)
        import_chunk = self._create_import_block(relative_path, tree, lines)
        if import_chunk:
            chunks.append(import_chunk)

        return chunks

    def _create_file_summary(
        self,
        file_path: str,
        tree: ast.Module,
        lines: List[str],
    ) -> Optional[CodeChunk]:
        """Create a file-level summary chunk.

        This provides high-level context about the file's purpose.
        """
        parts = [f"File: {file_path}"]

        # Extract module docstring
        docstring = ast.get_docstring(tree)
        if docstring:
            # Truncate long docstrings
            if len(docstring) > 500:
                docstring = docstring[:500] + "..."
            parts.append(f"Description: {docstring}")

        # Categorize top-level symbols
        classes = []
        functions = []
        async_functions = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                async_functions.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)

        symbols = [f"class {c}" for c in classes] + [
            f"def {f}" for f in functions + async_functions
        ]

        if symbols:
            parts.append(f"Contains: {', '.join(symbols[:10])}")
            if len(symbols) > 10:
                parts.append(f"  ... and {len(symbols) - 10} more")

        content = "\n".join(parts)

        # Determine file category from path
        is_test = "test" in file_path.lower() or file_path.endswith("_test.py")
        is_init = file_path.endswith("__init__.py")

        return CodeChunk(
            id=f"{file_path}:__file__",
            content=content,
            chunk_type=ChunkType.FILE_SUMMARY,
            file_path=file_path,
            line_start=1,
            line_end=len(lines),
            metadata={
                "symbol_count": len(symbols),
                "class_count": len(classes),
                "function_count": len(functions),
                "async_function_count": len(async_functions),
                "line_count": len(lines),
                "has_docstring": docstring is not None,
                "is_test_file": is_test,
                "is_init_file": is_init,
                "language": detect_language(file_path),
                # Include visibility for schema consistency across all chunks
                "visibility": "public",  # Files are always public
            },
        )

    def _create_import_block(
        self,
        file_path: str,
        tree: ast.Module,
        lines: List[str],
    ) -> Optional[CodeChunk]:
        """Create an import block chunk for dependency analysis."""
        imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        if not imports:
            return None

        content = f"File: {file_path}\nImports: {', '.join(imports)}"

        return CodeChunk(
            id=f"{file_path}:__imports__",
            content=content,
            chunk_type=ChunkType.IMPORT_BLOCK,
            file_path=file_path,
            line_start=1,
            line_end=20,  # Imports are usually at top
            metadata={
                "import_count": len(imports),
                "language": detect_language(file_path),
                "visibility": "public",  # Imports are always public
            },
        )

    def _chunk_class(
        self,
        node: ast.ClassDef,
        file_path: str,
        lines: List[str],
    ) -> List[CodeChunk]:
        """Chunk a class definition."""
        chunks: List[CodeChunk] = []
        class_id = f"{file_path}:{node.name}"

        # Class summary (if enabled)
        if self.config.include_class_summary:
            class_chunk = self._create_class_summary(node, file_path, lines)
            if class_chunk:
                chunks.append(class_chunk)

        # Process methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                chunks.extend(
                    self._chunk_function(
                        item,
                        file_path,
                        lines,
                        parent_id=class_id,
                        class_name=node.name,
                    )
                )

        return chunks

    def _create_class_summary(
        self,
        node: ast.ClassDef,
        file_path: str,
        lines: List[str],
    ) -> Optional[CodeChunk]:
        """Create a class-level summary chunk."""
        parts = [f"Class: {node.name}"]

        # Inheritance
        bases = [self._get_name(base) for base in node.bases]
        if bases:
            parts.append(f"Inherits: {', '.join(bases)}")

        # Docstring
        docstring = ast.get_docstring(node)
        if docstring:
            if len(docstring) > 300:
                docstring = docstring[:300] + "..."
            parts.append(f"Description: {docstring}")

        # Categorize methods
        methods = []
        public_methods = []
        private_methods = []
        dunder_methods = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                methods.append(item.name)
                if item.name.startswith("__") and item.name.endswith("__"):
                    dunder_methods.append(item.name)
                elif item.name.startswith("_"):
                    private_methods.append(item.name)
                else:
                    public_methods.append(item.name)

        if methods:
            parts.append(f"Methods: {', '.join(methods[:10])}")

        content = "\n".join(parts)

        # Calculate line count
        line_count = (node.end_lineno or node.lineno) - node.lineno + 1

        # Determine visibility (private if name starts with _)
        is_private = node.name.startswith("_") and not node.name.startswith("__")
        visibility = "private" if is_private else "public"

        # Count decorators
        decorator_count = len(node.decorator_list)

        return CodeChunk(
            id=f"{file_path}:{node.name}",
            content=content,
            chunk_type=ChunkType.CLASS_SUMMARY,
            file_path=file_path,
            symbol_name=node.name,
            symbol_type="class",
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            metadata={
                "method_count": len(methods),
                "public_method_count": len(public_methods),
                "private_method_count": len(private_methods),
                "dunder_method_count": len(dunder_methods),
                "has_docstring": docstring is not None,
                "line_count": line_count,
                "visibility": visibility,
                "base_count": len(bases),
                "decorator_count": decorator_count,
                "language": detect_language(file_path),
            },
        )

    def _chunk_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        lines: List[str],
        parent_id: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> List[CodeChunk]:
        """Chunk a function/method definition.

        For small functions: Single METHOD_HEADER chunk
        For large functions: METHOD_HEADER + multiple METHOD_BODY chunks
        """
        chunks: List[CodeChunk] = []

        # Build symbol ID
        if class_name:
            symbol_id = f"{file_path}:{class_name}.{node.name}"
            symbol_type = "method"
        else:
            symbol_id = f"{file_path}:{node.name}"
            symbol_type = "function"

        # Function header chunk (signature + docstring)
        header_chunk = self._create_method_header(
            node, file_path, symbol_id, symbol_type, parent_id
        )
        chunks.append(header_chunk)

        # Check if body chunking needed
        func_lines = (node.end_lineno or node.lineno) - node.lineno
        if (
            self.config.strategy in (ChunkingStrategy.BODY_AWARE, ChunkingStrategy.HIERARCHICAL)
            and func_lines > self.config.large_symbol_threshold
        ):
            # Large function - create body chunks
            body_chunks = self._chunk_function_body(node, file_path, lines, symbol_id, class_name)
            chunks.extend(body_chunks)

        return chunks

    def _create_method_header(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        symbol_id: str,
        symbol_type: str,
        parent_id: Optional[str],
    ) -> CodeChunk:
        """Create a method/function header chunk with signature and docstring."""
        parts = []

        # Signature
        is_async = isinstance(node, ast.AsyncFunctionDef)
        prefix = "async def" if is_async else "def"
        signature = self._build_signature(node)
        parts.append(f"{prefix} {node.name}{signature}")

        # Docstring
        docstring = ast.get_docstring(node)
        if docstring:
            if len(docstring) > 400:
                docstring = docstring[:400] + "..."
            parts.append(f"Description: {docstring}")

        # Decorators
        decorators = [self._get_name(d) for d in node.decorator_list]
        if decorators:
            parts.append(f"Decorators: {', '.join(decorators)}")

        content = "\n".join(parts)

        # Calculate line count for this function
        line_count = (node.end_lineno or node.lineno) - node.lineno + 1

        # Determine visibility (private if name starts with _)
        is_private = node.name.startswith("_") and not node.name.startswith("__")
        is_dunder = node.name.startswith("__") and node.name.endswith("__")
        visibility = "dunder" if is_dunder else ("private" if is_private else "public")

        # Count parameters
        param_count = len(node.args.args)
        if node.args.vararg:
            param_count += 1
        if node.args.kwarg:
            param_count += 1

        return CodeChunk(
            id=symbol_id,
            content=content,
            chunk_type=ChunkType.METHOD_HEADER,
            file_path=file_path,
            symbol_name=node.name,
            symbol_type=symbol_type,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            parent_id=parent_id,
            metadata={
                "is_async": is_async,
                "decorator_count": len(decorators),
                "has_docstring": docstring is not None,
                "line_count": line_count,
                "visibility": visibility,
                "param_count": param_count,
                "language": detect_language(file_path),
            },
        )

    def _chunk_function_body(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        lines: List[str],
        symbol_id: str,
        class_name: Optional[str],
    ) -> List[CodeChunk]:
        """Create overlapping body chunks for large functions.

        Uses sliding window with overlap to preserve context.
        """
        chunks: List[CodeChunk] = []

        # Get function body lines (excluding docstring line if present)
        start_line = node.lineno  # 1-indexed
        end_line = node.end_lineno or node.lineno

        # Skip docstring if present
        docstring = ast.get_docstring(node)
        if docstring:
            # Find where docstring ends
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    if stmt.value.value == docstring:
                        start_line = (stmt.end_lineno or stmt.lineno) + 1
                        break

        # Get body lines
        body_lines = lines[start_line - 1 : end_line]
        body_text = "\n".join(body_lines)

        # Calculate chunk parameters
        max_chars = self.config.max_chunk_chars
        overlap_chars = self.config.overlap_chars

        if len(body_text) <= max_chars:
            # Small enough for single chunk
            return []  # Header chunk already has docstring

        # Create overlapping chunks
        chunk_idx = 0
        pos = 0

        while pos < len(body_text):
            # Find chunk end (try to break at line boundary)
            chunk_end = min(pos + max_chars, len(body_text))

            # Try to break at newline
            if chunk_end < len(body_text):
                last_newline = body_text.rfind("\n", pos, chunk_end)
                if last_newline > pos + max_chars // 2:
                    chunk_end = last_newline + 1

            chunk_content = body_text[pos:chunk_end].strip()

            if chunk_content:
                # Calculate line numbers for this chunk
                prefix_lines = body_text[:pos].count("\n")
                chunk_lines = chunk_content.count("\n") + 1
                chunk_start_line = start_line + prefix_lines
                chunk_end_line = chunk_start_line + chunk_lines - 1

                # Determine visibility from function name
                is_private = node.name.startswith("_") and not node.name.startswith("__")
                is_dunder = node.name.startswith("__") and node.name.endswith("__")
                visibility = "dunder" if is_dunder else ("private" if is_private else "public")

                chunks.append(
                    CodeChunk(
                        id=f"{symbol_id}:body:{chunk_idx}",
                        content=f"# Code from {node.name} (part {chunk_idx + 1})\n{chunk_content}",
                        chunk_type=ChunkType.METHOD_BODY,
                        file_path=file_path,
                        symbol_name=node.name,
                        symbol_type="function_body",
                        line_start=chunk_start_line,
                        line_end=chunk_end_line,
                        parent_id=symbol_id,
                        metadata={
                            "chunk_index": chunk_idx,
                            "class_name": class_name,
                            "visibility": visibility,
                            "language": detect_language(file_path),
                        },
                    )
                )

            # Move position forward - ensure we always make progress
            # Minimum step is half of max_chars to avoid infinite loops
            min_step = max(max_chars // 2, 100)
            next_pos = max(chunk_end - overlap_chars, pos + min_step)

            # If we're at the end, stop
            if next_pos >= len(body_text) or chunk_end >= len(body_text):
                break

            pos = next_pos
            chunk_idx += 1

            # Safety limit - max 10 chunks per function body
            if chunk_idx >= 10:
                logger.debug(f"Reached chunk limit for {symbol_id}")
                break

        return chunks

    def _build_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Build function signature string."""
        args = []

        # Regular args
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_name(arg.annotation)}"
            args.append(arg_str)

        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        signature = f"({', '.join(args)})"

        # Return type
        if node.returns:
            signature += f" -> {self._get_name(node.returns)}"

        return signature

    def _get_name(self, node: ast.expr) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Call):
            return f"{self._get_name(node.func)}(...)"
        else:
            return ast.unparse(node) if hasattr(ast, "unparse") else "?"


def chunk_codebase(
    root_path: Path,
    config: Optional[ChunkConfig] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> List[CodeChunk]:
    """Chunk an entire codebase.

    Args:
        root_path: Root directory of the codebase
        config: Chunking configuration
        ignore_patterns: Patterns to ignore (e.g., ["venv/", "__pycache__/"])

    Returns:
        List of all CodeChunks from the codebase
    """
    chunker = CodeChunker(config)
    ignore_patterns = ignore_patterns or [
        "venv/",
        ".venv/",
        "node_modules/",
        ".git/",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
    ]

    all_chunks: List[CodeChunk] = []

    for py_file in root_path.rglob("*.py"):
        # Check ignore patterns
        rel_path = str(py_file.relative_to(root_path))
        if any(pattern in rel_path for pattern in ignore_patterns):
            continue

        chunks = chunker.chunk_file(py_file, rel_path)
        all_chunks.extend(chunks)

    logger.info(f"Chunked {len(all_chunks)} chunks from {root_path}")
    return all_chunks


# =============================================================================
# Tier-Aware Chunking (Phase 4 of Tiered Language Support)
# =============================================================================

# Config file patterns for structured chunking
CONFIG_FILE_PATTERNS: Dict[str, Dict[str, Any]] = {
    "yaml": {
        "block_start": r"^(\w[\w-]*):\s*$",
        "key_value": r"^(\w[\w-]*):\s+\S",
        "list_item": r"^\s*-\s+",
    },
    "json": {
        "object_key": r'"(\w+)"\s*:',
    },
    "toml": {
        "section": r"^\s*\[([^\]]+)\]",
        "key": r"^(\w+)\s*=",
    },
    "xml": {
        "element": r"<(\w+)[^>]*>",
        "attribute": r'(\w+)="[^"]*"',
    },
}


class ChunkingFallback(Enum):
    """Fallback strategies for chunking."""

    PYTHON_AST = "python_ast"  # Full Python AST parsing
    TREE_SITTER = "tree_sitter"  # Tree-sitter grammar
    CONFIG_AWARE = "config_aware"  # Config file structure awareness
    SLIDING_WINDOW = "sliding_window"  # Rust/Python overlapping chunks


class TierAwareChunker:
    """Comprehensive tier-aware chunker with simple, robust fallback chain.

    Provides chunking for ALL languages and file types with a cascading
    fallback strategy:

    1. **Python AST** (Tier 1 Python): Full AST parsing with semantic analysis
    2. **Tree-sitter** (Tier 1/2/3): Grammar-based symbol extraction
    3. **Config-Aware** (Config files): Structure-aware chunking for
       YAML, JSON, TOML, Properties, HOCON, INI, XML
    4. **Sliding Window** (Universal fallback): Rust/Python overlapping
       chunks for any file type not covered above

    Design Rationale:
        - Regex patterns are intentionally NOT used because:
          1. If code is parseable → tree-sitter handles it (grammar-aware)
          2. If code is unparseable → sliding window is more robust than
             fragile regex patterns
        - Sliding window uses Rust native implementation when available
          for 3-5x speedup
        - Simple chain: fewer failure modes, easier to debug

    This ensures NO language or file type is left without proper chunking.

    Example:
        chunker = TierAwareChunker(tree_sitter=TreeSitterExtractor())

        # Python - uses AST
        chunks = chunker.chunk_file(Path("main.py"), "main.py")

        # Go - uses tree-sitter
        chunks = chunker.chunk_file(Path("main.go"), "main.go")

        # YAML config - uses config-aware chunking
        chunks = chunker.chunk_file(Path("config.yaml"), "config.yaml")

        # Unknown/unparseable - uses sliding window (Rust/Python)
        chunks = chunker.chunk_file(Path("data.xyz"), "data.xyz")
    """

    def __init__(
        self,
        tree_sitter: Optional[Any] = None,
        python_chunker: Optional[CodeChunker] = None,
        config: Optional[ChunkConfig] = None,
        text_chunker: Optional["TextChunkerProtocol"] = None,
    ):
        """Initialize tier-aware chunker.

        Args:
            tree_sitter: TreeSitterExtractor instance for tree-sitter languages
            python_chunker: CodeChunker for Python files (created if not provided)
            config: Chunking configuration
            text_chunker: TextChunkerProtocol for sliding window (Rust/Python)
        """
        self._ts = tree_sitter
        self._py_chunker = python_chunker or CodeChunker(config)
        self._config = config or ChunkConfig()

        # Lazy-load text chunker to avoid import cycles
        if text_chunker is not None:
            self._text_chunker = text_chunker
        else:
            self._text_chunker = None  # Lazy-loaded on first use

    def _get_text_chunker(self) -> "TextChunkerProtocol":
        """Get text chunker, lazy-loading if needed.

        Uses Rust implementation when available for 3-5x speedup on
        sliding window chunking.
        """
        if self._text_chunker is None:
            from victor_contracts.processing_runtime import get_default_text_chunker

            self._text_chunker = get_default_text_chunker()
        return self._text_chunker

    def _chunk_info_to_code_chunk(
        self,
        chunk_info: "ChunkInfo",
        relative_path: str,
        language: str,
        chunk_index: int,
    ) -> CodeChunk:
        """Convert protocol ChunkInfo to domain CodeChunk.

        Bridges the generic text chunking protocol (Rust/Python) with
        the domain-specific CodeChunk used for embedding.

        ID Format: `module/path:window:chunk_idx` with line range in metadata.

        Args:
            chunk_info: Protocol ChunkInfo from TextChunkerProtocol
            relative_path: File path for ID and metadata
            language: Detected language
            chunk_index: Sequential chunk number

        Returns:
            CodeChunk ready for embedding
        """
        # Build context header with line range
        header = f"# {relative_path} (lines {chunk_info.start_line}-{chunk_info.end_line})"

        return CodeChunk(
            id=f"{relative_path}:window:{chunk_index}:L{chunk_info.start_line}-L{chunk_info.end_line}",
            content=f"{header}\n{chunk_info.text}",
            chunk_type=ChunkType.METHOD_BODY,
            file_path=relative_path,
            line_start=chunk_info.start_line,
            line_end=chunk_info.end_line,
            metadata={
                "chunk_index": chunk_index,
                "language": language,
                "chunking": "sliding_window",
                "has_overlap": chunk_info.overlap_prev > 0,
                "start_offset": chunk_info.start_offset,
                "end_offset": chunk_info.end_offset,
            },
        )

    def chunk_file(
        self,
        file_path: Path,
        relative_path: str,
        language: Optional[str] = None,
        content: Optional[str] = None,
    ) -> List[CodeChunk]:
        """Chunk a file using the optimal strategy with robust fallback.

        Fallback chain (simple, no regex):
        1. Python AST (for Python files)
        2. Tree-sitter (for supported languages)
        3. Config-aware (for config files)
        4. Sliding window (Rust/Python - universal fallback)

        Regex is intentionally skipped - if tree-sitter can't parse it,
        sliding window is more robust than fragile regex patterns.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path for IDs and metadata
            language: Language name (auto-detected if not provided)
            content: Optional file content (reads file if not provided)

        Returns:
            List of CodeChunk objects ready for embedding
        """
        # Detect language if not provided
        if language is None:
            language = detect_language(str(file_path))

        # Read content if not provided
        if content is None:
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                return []

        # Determine fallback chain based on language
        fallbacks = self._get_fallback_chain(language, file_path)

        # Try each fallback in order
        for fallback in fallbacks:
            try:
                chunks = self._apply_fallback(fallback, file_path, relative_path, language, content)
                if chunks:
                    logger.debug(
                        f"Chunked {relative_path} with {fallback.value}: " f"{len(chunks)} chunks"
                    )
                    return chunks
            except Exception as e:
                logger.debug(f"{fallback.value} failed for {relative_path}: {e}")
                continue

        # Ultimate fallback - should never reach here but just in case
        return self._chunk_with_overlap(file_path, relative_path, language, content)

    def _get_fallback_chain(self, language: str, file_path: Path) -> List[ChunkingFallback]:
        """Determine the fallback chain for a language.

        Args:
            language: Language identifier
            file_path: Path to the file

        Returns:
            Ordered list of fallback strategies to try
        """
        # Python always uses AST first
        if language == "python":
            return [ChunkingFallback.PYTHON_AST, ChunkingFallback.SLIDING_WINDOW]

        # Check if language has tree-sitter support. The shared victor_codegraph
        # parser (TD-CG / ADR-014) stands in for an injected TreeSitterExtractor,
        # so its presence also enables the tree-sitter fallback.
        try:
            from victor_coding.languages.tiers import get_tier

            tier_config = get_tier(language)
            has_tree_sitter = (
                tier_config.has_tree_sitter and self._ts is not None
            ) or _victor_codegraph is not None
        except ImportError:
            has_tree_sitter = self._ts is not None or _victor_codegraph is not None

        # Config files get special treatment
        config_languages = {"yaml", "json", "toml", "xml", "ini", "properties", "hocon"}
        is_config = language in config_languages or self._is_config_file(file_path)

        # Build fallback chain - simple and robust
        # Regex is intentionally NOT used because:
        # 1. If code is parseable → tree-sitter handles it (grammar-aware)
        # 2. If code is unparseable → sliding window is more robust than fragile regex
        chain = []

        if has_tree_sitter:
            chain.append(ChunkingFallback.TREE_SITTER)

        if is_config:
            chain.append(ChunkingFallback.CONFIG_AWARE)

        # Always end with sliding window (Rust/Python native - fast and robust)
        # This handles: unparseable code, unknown languages, tree-sitter failures
        chain.append(ChunkingFallback.SLIDING_WINDOW)

        return chain

    def _is_config_file(self, file_path: Path) -> bool:
        """Check if a file is a config file based on name patterns."""
        name = file_path.name.lower()
        config_patterns = [
            "config",
            "settings",
            "properties",
            ".conf",
            ".cfg",
            ".ini",
            ".env",
            "dockerfile",
            "makefile",
            "gemfile",
            "rakefile",
            "procfile",
            "vagrantfile",
            ".rc",
        ]
        return any(pattern in name for pattern in config_patterns)

    def _apply_fallback(
        self,
        fallback: ChunkingFallback,
        file_path: Path,
        relative_path: str,
        language: str,
        content: str,
    ) -> List[CodeChunk]:
        """Apply a specific fallback strategy."""
        if fallback == ChunkingFallback.PYTHON_AST:
            return self._py_chunker.chunk_file(file_path, relative_path, content)

        elif fallback == ChunkingFallback.TREE_SITTER:
            return self._chunk_with_tree_sitter(file_path, relative_path, language, content)

        elif fallback == ChunkingFallback.CONFIG_AWARE:
            return self._chunk_config_file(file_path, relative_path, language, content)

        elif fallback == ChunkingFallback.SLIDING_WINDOW:
            return self._chunk_with_overlap(file_path, relative_path, language, content)

        return []

    def _chunk_with_tree_sitter(
        self,
        file_path: Path,
        relative_path: str,
        language: str,
        content: str,
    ) -> List[CodeChunk]:
        """Chunk using tree-sitter symbol extraction.

        TD-CG / ADR-014: prefers the shared ``victor_codegraph`` parser when
        installed (removes per-repo tree-sitter duplication, adds JS/TS + size-
        capping); falls back to the injected ``TreeSitterExtractor`` otherwise.
        """
        if _victor_codegraph is not None:
            try:
                vc_chunks = self._chunk_via_victor_codegraph(relative_path, language, content)
                if vc_chunks:
                    return vc_chunks
            except Exception as e:
                logger.debug(
                    f"victor_codegraph tree-sitter delegation failed for "
                    f"{relative_path}: {e}; using legacy extractor"
                )

        if not self._ts:
            return []

        chunks: List[CodeChunk] = []
        lines = content.split("\n")

        symbols = self._ts.extract_symbols(file_path, language)

        if not symbols:
            return []  # Let next fallback handle it

        # File summary
        if self._config.include_file_summary:
            symbol_names = [s.name for s in symbols[:10]]
            summary = (
                f"File: {relative_path}\nLanguage: {language}\nContains: {', '.join(symbol_names)}"
            )
            if len(symbols) > 10:
                summary += f"\n  ... and {len(symbols) - 10} more"

            chunks.append(
                CodeChunk(
                    id=f"{relative_path}:__file__",
                    content=summary,
                    chunk_type=ChunkType.FILE_SUMMARY,
                    file_path=relative_path,
                    line_start=1,
                    line_end=len(lines),
                    metadata={"symbol_count": len(symbols), "language": language},
                )
            )

        # Symbol chunks
        for sym in symbols:
            start = sym.line_number - 1
            end = (sym.end_line or sym.line_number) - 1

            if start < 0 or start >= len(lines):
                continue

            symbol_content = "\n".join(lines[start : end + 1])
            if len(symbol_content) > self._config.max_chunk_chars:
                symbol_content = (
                    symbol_content[: self._config.max_chunk_chars] + "\n# ... truncated"
                )

            chunk_type = (
                ChunkType.CLASS_SUMMARY
                if sym.type in ("class", "struct", "interface", "trait")
                else ChunkType.METHOD_HEADER
            )

            parent = sym.parent_symbol
            chunk_id = (
                f"{relative_path}:{parent}.{sym.name}" if parent else f"{relative_path}:{sym.name}"
            )

            chunks.append(
                CodeChunk(
                    id=chunk_id,
                    content=f"{sym.type.title()}: {sym.name}\n{symbol_content}",
                    chunk_type=chunk_type,
                    file_path=relative_path,
                    symbol_name=sym.name,
                    symbol_type=sym.type,
                    line_start=sym.line_number,
                    line_end=sym.end_line or sym.line_number,
                    parent_id=f"{relative_path}:{parent}" if parent else None,
                    metadata={"language": language, "line_count": end - start + 1},
                )
            )

        return chunks

    def _chunk_via_victor_codegraph(
        self,
        relative_path: str,
        language: str,
        content: str,
    ) -> List[CodeChunk]:
        """Delegate tree-sitter symbol extraction to ``victor_codegraph`` (ADR-014).

        Maps ``victor_codegraph.CodeSymbol`` into this module's hierarchical
        ``CodeChunk`` shape, mirroring the FILE_SUMMARY + per-symbol structure
        emitted by ``_chunk_with_tree_sitter``. Returns ``[]`` when no symbols are
        extracted so the fallback chain can continue.
        """
        parsed = _victor_codegraph.parse(content, language=language, file_path=relative_path)
        symbols = parsed.symbols
        if not symbols:
            return []

        lines = content.split("\n")
        chunks: List[CodeChunk] = []
        class_kinds = {"CLASS", "STRUCT", "INTERFACE", "TRAIT", "ENUM"}

        if self._config.include_file_summary:
            symbol_names = [s.simple_name for s in symbols[:10]]
            summary = (
                f"File: {relative_path}\nLanguage: {language}\n"
                f"Contains: {', '.join(symbol_names)}"
            )
            if len(symbols) > 10:
                summary += f"\n  ... and {len(symbols) - 10} more"
            chunks.append(
                CodeChunk(
                    id=f"{relative_path}:__file__",
                    content=summary,
                    chunk_type=ChunkType.FILE_SUMMARY,
                    file_path=relative_path,
                    line_start=1,
                    line_end=len(lines),
                    metadata={
                        "symbol_count": len(symbols),
                        "language": language,
                        "source": "victor_codegraph",
                    },
                )
            )

        for sym in symbols:
            kind = sym.symbol_type.name
            if kind in {"FILE", "MODULE", "PACKAGE"}:
                continue
            start_line = sym.location.start_line or 1
            end_line = sym.location.end_line or start_line
            body = sym.source_code or ""
            if len(body) > self._config.max_chunk_chars:
                body = body[: self._config.max_chunk_chars] + "\n# ... truncated"

            scope = list(sym.scope_chain or [])
            qualified = ".".join(scope + [sym.simple_name]) if scope else sym.simple_name

            chunks.append(
                CodeChunk(
                    id=f"{relative_path}:{qualified}",
                    content=f"{kind.title()}: {sym.simple_name}\n{body}",
                    chunk_type=(
                        ChunkType.CLASS_SUMMARY if kind in class_kinds else ChunkType.METHOD_HEADER
                    ),
                    file_path=relative_path,
                    symbol_name=sym.simple_name,
                    symbol_type=kind.lower(),
                    line_start=start_line,
                    line_end=end_line,
                    parent_id=(f"{relative_path}:{'.'.join(scope)}" if scope else None),
                    metadata={
                        "language": language,
                        "line_count": end_line - start_line + 1,
                        "source": "victor_codegraph",
                    },
                )
            )

        return chunks

    def _chunk_config_file(
        self,
        file_path: Path,
        relative_path: str,
        language: str,
        content: str,
    ) -> List[CodeChunk]:
        """Chunk config files with structure awareness."""
        chunks: List[CodeChunk] = []
        lines = content.split("\n")

        # File summary
        chunks.append(
            CodeChunk(
                id=f"{relative_path}:__file__",
                content=f"Config File: {relative_path}\nFormat: {language}\nLines: {len(lines)}",
                chunk_type=ChunkType.FILE_SUMMARY,
                file_path=relative_path,
                line_start=1,
                line_end=len(lines),
                metadata={"language": language, "config_type": language},
            )
        )

        # For YAML/JSON/TOML - try to identify top-level sections
        if language in ("yaml", "json", "toml", "hocon"):
            sections = self._extract_config_sections(content, language)

            for section in sections:
                section_content = section.get("content", "")
                if len(section_content) > self._config.max_chunk_chars:
                    section_content = (
                        section_content[: self._config.max_chunk_chars] + "\n# ... truncated"
                    )

                chunks.append(
                    CodeChunk(
                        id=f"{relative_path}:{section['name']}",
                        content=f"Section: {section['name']}\n{section_content}",
                        chunk_type=ChunkType.CLASS_SUMMARY,
                        file_path=relative_path,
                        symbol_name=section["name"],
                        symbol_type="config_section",
                        line_start=section.get("start_line", 1),
                        line_end=section.get("end_line", len(lines)),
                        metadata={"language": language},
                    )
                )

            if chunks:
                return chunks

        # Fallback: chunk by sections/blocks
        return self._chunk_with_overlap(file_path, relative_path, language, content)

    def _extract_config_sections(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Extract top-level sections from config files."""
        sections = []
        lines = content.split("\n")

        if language == "yaml":
            # YAML top-level keys
            current_section = None
            section_start = 0
            section_lines = []

            for i, line in enumerate(lines):
                # Top-level key (no leading whitespace)
                if line and not line[0].isspace() and ":" in line:
                    # Save previous section
                    if current_section:
                        sections.append(
                            {
                                "name": current_section,
                                "content": "\n".join(section_lines),
                                "start_line": section_start,
                                "end_line": i,
                            }
                        )

                    current_section = line.split(":")[0].strip()
                    section_start = i + 1
                    section_lines = [line]
                elif current_section:
                    section_lines.append(line)

            # Save last section
            if current_section:
                sections.append(
                    {
                        "name": current_section,
                        "content": "\n".join(section_lines),
                        "start_line": section_start,
                        "end_line": len(lines),
                    }
                )

        elif language == "toml":
            # TOML sections [section]
            current_section = "root"
            section_start = 1
            section_lines = []

            for i, line in enumerate(lines):
                match = re.match(r"^\s*\[([^\]]+)\]", line)
                if match:
                    if section_lines:
                        sections.append(
                            {
                                "name": current_section,
                                "content": "\n".join(section_lines),
                                "start_line": section_start,
                                "end_line": i,
                            }
                        )
                    current_section = match.group(1)
                    section_start = i + 1
                    section_lines = [line]
                else:
                    section_lines.append(line)

            if section_lines:
                sections.append(
                    {
                        "name": current_section,
                        "content": "\n".join(section_lines),
                        "start_line": section_start,
                        "end_line": len(lines),
                    }
                )

        return sections

    def _chunk_with_overlap(
        self,
        file_path: Path,
        relative_path: str,
        language: str,
        content: str,
    ) -> List[CodeChunk]:
        """Universal fallback: sliding window with overlap.

        Delegates to TextChunkerProtocol (Rust when available) for
        3-5x performance improvement on large files.

        Works for ANY file type. Creates overlapping chunks based on
        line boundaries for context preservation.
        """
        chunks: List[CodeChunk] = []
        lines = content.split("\n")

        # File summary (always created)
        chunks.append(
            CodeChunk(
                id=f"{relative_path}:__file__",
                content=f"File: {relative_path}\nLanguage: {language}\nLines: {len(lines)}",
                chunk_type=ChunkType.FILE_SUMMARY,
                file_path=relative_path,
                line_start=1,
                line_end=len(lines),
                metadata={"language": language, "chunking": "sliding_window"},
            )
        )

        # Skip empty content
        if not content.strip():
            return chunks

        # Delegate to protocol (Rust or Python) for line-aware chunking
        text_chunker = self._get_text_chunker()
        chunk_infos = text_chunker.chunk_with_overlap(
            content,
            self._config.max_chunk_chars,
            self._config.overlap_chars,
        )

        # Convert protocol ChunkInfo to domain CodeChunk
        for idx, chunk_info in enumerate(chunk_infos):
            if chunk_info.text.strip():
                chunks.append(
                    self._chunk_info_to_code_chunk(
                        chunk_info,
                        relative_path,
                        language,
                        chunk_index=idx,
                    )
                )

            # Safety limit (100 chunks per file)
            if idx >= 99:
                logger.debug(f"Reached chunk limit for {relative_path}")
                break

        return chunks
