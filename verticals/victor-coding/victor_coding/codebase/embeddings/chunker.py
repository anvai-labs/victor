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

"""AST-aware code chunking for embeddings.

This module provides chunking strategies that split code at semantic
boundaries (functions, classes, methods) instead of arbitrary text splits.
This produces better embeddings for code search.

Usage:
    >>> from victor_coding.codebase.embeddings.chunker import ASTAwareChunker
    >>> from victor_coding.languages.registry import get_language_registry
    >>>
    >>> registry = get_language_registry()
    >>> registry.discover_plugins()
    >>> chunker = ASTAwareChunker(registry)
    >>> chunks = chunker.chunk_file(Path("main.py"))
    >>> for chunk in chunks:
    ...     print(f"{chunk.chunk_type}: {chunk.symbol_name} ({chunk.start_line}-{chunk.end_line})")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from victor_coding.languages.base import CodeChunk
from victor_coding.languages.registry import LanguageRegistry, get_language_registry

if TYPE_CHECKING:
    from tree_sitter import Node, Parser, Tree

logger = logging.getLogger(__name__)

# TD-CG / ADR-014: shared code->CPG chunker. Soft import so the legacy in-repo
# tree-sitter path stays the fallback when the package is absent.
try:
    import victor_codegraph as _victor_codegraph
except Exception:  # ImportError, or a partial/native load failure
    _victor_codegraph = None


class ASTAwareChunker:
    """Chunks code at AST boundaries for better embeddings.

    Instead of splitting code at arbitrary character/line boundaries,
    this chunker uses tree-sitter to identify semantic boundaries like
    function definitions, class definitions, and module-level code.

    Benefits:
    - Each chunk represents a complete semantic unit
    - Better embedding quality for code search
    - Preserves context (docstrings, decorators, etc.)
    - Works uniformly across all languages via the plugin system
    """

    # Default maximum chunk size in characters
    DEFAULT_MAX_CHUNK_SIZE = 2000

    # Minimum chunk size to avoid too-small chunks
    DEFAULT_MIN_CHUNK_SIZE = 50

    def __init__(
        self,
        registry: Optional[LanguageRegistry] = None,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    ):
        """Initialize the chunker.

        Args:
            registry: Language registry to use. If None, uses global registry.
            max_chunk_size: Maximum chunk size in characters.
                            Larger chunks will be split further.
            min_chunk_size: Minimum chunk size. Smaller chunks will be
                           combined with adjacent chunks.
        """
        self.registry = registry or get_language_registry()
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self._parsers: Dict[str, "Parser"] = {}

    def _get_parser(self, language: str) -> Optional["Parser"]:
        """Get tree-sitter parser for a language."""
        if language in self._parsers:
            return self._parsers[language]

        try:
            from victor_coding.codebase.tree_sitter_manager import get_parser

            parser = get_parser(language)
            self._parsers[language] = parser
            return parser
        except Exception as e:
            logger.debug(f"Could not get parser for {language}: {e}")
            return None

    def chunk_file(self, file_path: Path, language: Optional[str] = None) -> List[CodeChunk]:
        """Split a file into semantic chunks using AST.

        Args:
            file_path: Path to the source file
            language: Language name (auto-detected if None)

        Returns:
            List of semantic code chunks
        """
        if language is None:
            language = self.registry.detect_language(file_path)
            if language is None:
                # Fallback to simple line-based chunking
                return self._chunk_by_lines(file_path)

        # TD-CG / ADR-014: when the shared victor_codegraph chunker is installed it
        # is the source of truth for tree-sitter code chunking (size-capping + a
        # real JS/TS parser). Any failure falls through to the legacy path below.
        if _victor_codegraph is not None:
            try:
                return self._chunk_via_victor_codegraph(file_path, language)
            except Exception as e:
                logger.debug(
                    f"victor_codegraph delegation failed for {file_path}: "
                    f"{e}; using legacy tree-sitter path"
                )

        parser = self._get_parser(language)
        if parser is None:
            return self._chunk_by_lines(file_path)

        try:
            content = file_path.read_bytes()
            tree = parser.parse(content)
            return self._chunk_from_tree(tree, content, str(file_path), language)
        except Exception as e:
            logger.debug(f"Failed to parse {file_path} for chunking: {e}")
            return self._chunk_by_lines(file_path)

    def _chunk_via_victor_codegraph(self, file_path: Path, language: str) -> List[CodeChunk]:
        """Delegate chunking to the shared ``victor_codegraph`` package (ADR-014).

        Adapts each ``victor_codegraph.CodeChunk`` into this module's ``CodeChunk``
        shape (used for embeddings). ``victor_codegraph`` already applies size-capping
        and ships a real JS/TS parser; line range and symbol kind ride on each chunk's
        metadata. Window chunks (no symbol) map to ``chunk_type="text"`` so the
        existing line-fallback contract is preserved.
        """
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        cfg = _victor_codegraph.ChunkConfig(
            max_chunk_tokens=max(1, int(self.max_chunk_size / 3.5)),
            chunk_overlap_tokens=0,
            include_private=True,
        )
        chunks: List[CodeChunk] = []
        for vc in _victor_codegraph.chunk(
            content, language=language, file_path=str(file_path), config=cfg
        ):
            meta = vc.metadata
            start_line = int(meta.get("start_line") or 1)
            end_line = int(meta.get("end_line") or start_line)
            symbol_type = meta.get("symbol_type")
            is_body_split = bool(meta.get("is_body_split"))
            chunks.append(
                CodeChunk(
                    text=vc.text,
                    start_line=start_line,
                    end_line=max(end_line, start_line),
                    chunk_type=symbol_type.lower() if symbol_type else "text",
                    symbol_name=meta.get("simple_name"),
                    parent_symbol=meta.get("simple_name") if is_body_split else None,
                    file_path=str(file_path),
                )
            )
        return chunks

    def _chunk_from_tree(
        self,
        tree: "Tree",
        content: bytes,
        file_path: str,
        language: str,
    ) -> List[CodeChunk]:
        """Extract semantic chunks from a parsed AST.

        Strategy:
        1. Find all top-level definitions (functions, classes)
        2. Create a chunk for each definition
        3. Create chunks for module-level code between definitions
        4. Split large chunks into smaller ones if needed
        """
        chunks: List[CodeChunk] = []
        root = tree.root_node

        # Get plugin queries for this language
        try:
            plugin = self.registry.get(language)
            queries = plugin.tree_sitter_queries
        except KeyError:
            queries = None

        # Find all top-level symbol nodes
        symbol_nodes: List[Tuple["Node", str, str]] = []  # (node, type, name)

        if queries and queries.symbols:
            # Use plugin queries to find symbols
            for pattern in queries.symbols:
                from tree_sitter import Query, QueryCursor

                try:
                    query = Query(tree.language, pattern.query)
                    cursor = QueryCursor(query)
                    captures = cursor.captures(root)

                    for capture_name, nodes in captures.items():
                        if capture_name == "name":
                            for node in nodes:
                                # Find the parent definition node
                                parent = self._find_definition_parent(node, language)
                                if parent:
                                    name = node.text.decode("utf-8", errors="ignore")
                                    symbol_nodes.append((parent, pattern.symbol_type, name))
                except Exception:
                    continue

        # Sort by start position
        symbol_nodes.sort(key=lambda x: x[0].start_byte)

        # Create chunks for each symbol and gaps between them
        prev_end = 0
        content_str = content.decode("utf-8", errors="ignore")
        lines = content_str.split("\n")

        for node, sym_type, sym_name in symbol_nodes:
            # Check for module-level code before this symbol
            if node.start_byte > prev_end:
                gap_text = (
                    content[prev_end : node.start_byte].decode("utf-8", errors="ignore").strip()
                )
                if gap_text and len(gap_text) >= self.min_chunk_size:
                    start_line = content[:prev_end].count(b"\n") + 1
                    end_line = content[: node.start_byte].count(b"\n")
                    chunks.append(
                        CodeChunk(
                            text=gap_text,
                            start_line=start_line,
                            end_line=end_line,
                            chunk_type="module_code",
                            file_path=file_path,
                        )
                    )

            # Create chunk for the symbol
            sym_text = content[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            # Check if we need to split a large chunk
            if len(sym_text) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(
                    sym_text, start_line, sym_type, sym_name, file_path
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    CodeChunk(
                        text=sym_text,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type=sym_type,
                        symbol_name=sym_name,
                        file_path=file_path,
                    )
                )

            prev_end = node.end_byte

        # Check for trailing module code
        if prev_end < len(content):
            trailing = content[prev_end:].decode("utf-8", errors="ignore").strip()
            if trailing and len(trailing) >= self.min_chunk_size:
                start_line = content[:prev_end].count(b"\n") + 1
                end_line = len(lines)
                chunks.append(
                    CodeChunk(
                        text=trailing,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="module_code",
                        file_path=file_path,
                    )
                )

        # If no chunks were created, use line-based fallback
        if not chunks:
            return self._chunk_by_lines_content(content_str, file_path)

        return chunks

    def _find_definition_parent(self, name_node: "Node", language: str) -> Optional["Node"]:
        """Find the definition node that contains a name node."""
        # Definition node types by language
        definition_types = {
            "python": {
                "function_definition",
                "class_definition",
                "decorated_definition",
            },
            "javascript": {
                "function_declaration",
                "class_declaration",
                "method_definition",
                "lexical_declaration",
            },
            "typescript": {
                "function_declaration",
                "class_declaration",
                "method_definition",
                "method_signature",
                "lexical_declaration",
            },
            "go": {"function_declaration", "method_declaration", "type_declaration"},
            "java": {
                "class_declaration",
                "interface_declaration",
                "method_declaration",
            },
            "rust": {
                "function_item",
                "struct_item",
                "enum_item",
                "trait_item",
                "impl_item",
            },
            "cpp": {"function_definition", "class_specifier", "struct_specifier"},
        }

        target_types = definition_types.get(language, set())

        current = name_node.parent
        while current is not None:
            if current.type in target_types:
                return current
            current = current.parent

        return None

    def _split_large_chunk(
        self,
        text: str,
        start_line: int,
        chunk_type: str,
        symbol_name: Optional[str],
        file_path: str,
    ) -> List[CodeChunk]:
        """Split a large chunk into smaller sub-chunks.

        Tries to split at natural boundaries (blank lines, closing braces).
        """
        chunks: List[CodeChunk] = []
        lines = text.split("\n")
        current_chunk: List[str] = []
        current_start = start_line
        current_len = 0

        for i, line in enumerate(lines):
            line_len = len(line) + 1  # +1 for newline

            if current_len + line_len > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n".join(current_chunk)
                chunks.append(
                    CodeChunk(
                        text=chunk_text,
                        start_line=current_start,
                        end_line=current_start + len(current_chunk) - 1,
                        chunk_type=f"{chunk_type}_part",
                        symbol_name=symbol_name,
                        parent_symbol=symbol_name if len(chunks) > 0 else None,
                        file_path=file_path,
                    )
                )
                current_chunk = []
                current_start = start_line + i
                current_len = 0

            current_chunk.append(line)
            current_len += line_len

        # Save remaining
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append(
                CodeChunk(
                    text=chunk_text,
                    start_line=current_start,
                    end_line=current_start + len(current_chunk) - 1,
                    chunk_type=f"{chunk_type}_part" if len(chunks) > 0 else chunk_type,
                    symbol_name=symbol_name,
                    parent_symbol=symbol_name if len(chunks) > 0 else None,
                    file_path=file_path,
                )
            )

        return chunks

    def _chunk_by_lines(self, file_path: Path) -> List[CodeChunk]:
        """Fallback: chunk file by lines when AST is unavailable."""
        try:
            content = file_path.read_text(encoding="utf-8")
            return self._chunk_by_lines_content(content, str(file_path))
        except Exception:
            return []

    def _chunk_by_lines_content(self, content: str, file_path: str) -> List[CodeChunk]:
        """Chunk content by lines."""
        chunks: List[CodeChunk] = []
        lines = content.split("\n")
        current_chunk: List[str] = []
        current_start = 1
        current_len = 0

        for i, line in enumerate(lines, 1):
            line_len = len(line) + 1

            if current_len + line_len > self.max_chunk_size and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(
                    CodeChunk(
                        text=chunk_text,
                        start_line=current_start,
                        end_line=current_start + len(current_chunk) - 1,
                        chunk_type="text",
                        file_path=file_path,
                    )
                )
                current_chunk = []
                current_start = i
                current_len = 0

            current_chunk.append(line)
            current_len += line_len

        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append(
                CodeChunk(
                    text=chunk_text,
                    start_line=current_start,
                    end_line=current_start + len(current_chunk) - 1,
                    chunk_type="text",
                    file_path=file_path,
                )
            )

        return chunks
