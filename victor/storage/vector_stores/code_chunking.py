# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Chunking strategies for code-oriented vector stores.

The provider stack historically used a single symbol-span chunking path inside
``proximadb_multi.py``. This module extracts that logic behind a strategy
interface so more structure-aware chunkers can be added without further
inflating the provider.

Two strategies are implemented:
- ``symbol_span``: current line-span chunking over extracted symbols
- ``tree_sitter_structural``: AST-aware chunking that preserves module-level
  statements and falls back to symbol spans when no parse tree is available
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class CodeChunk:
    """Chunk of code prepared for vector indexing."""

    content: str
    start_line: int
    end_line: int
    chunk_type: str
    symbol_name: Optional[str] = None
    parent_symbol: Optional[str] = None


@runtime_checkable
class CodeSymbolLike(Protocol):
    """Protocol describing the symbol metadata chunkers rely on."""

    name: str
    symbol_type: str
    line_start: int
    line_end: int
    parent_symbol: Optional[str]

    @property
    def qualified_name(self) -> str: ...


@dataclass(frozen=True)
class TreeSitterParseContext:
    """Minimal parse-tree context required by structural chunkers."""

    content: str
    source_bytes: bytes
    root_node: Any
    line_start_offsets: list[int]

    @classmethod
    def from_content(cls, content: str, root_node: Any) -> "TreeSitterParseContext":
        source_bytes = content.encode("utf-8")
        line_start_offsets = [0]
        for index, byte in enumerate(source_bytes):
            if byte == ord("\n"):
                line_start_offsets.append(index + 1)
        return cls(
            content=content,
            source_bytes=source_bytes,
            root_node=root_node,
            line_start_offsets=line_start_offsets,
        )

    def slice(self, start_byte: int, end_byte: int) -> str:
        return self.source_bytes[start_byte:end_byte].decode("utf-8", errors="ignore")

    def byte_offset_to_line(self, offset: int) -> int:
        normalized = max(offset, 0)
        return max(1, bisect_right(self.line_start_offsets, normalized))


@dataclass(frozen=True)
class CodeChunkingContext:
    """Inputs shared across chunking strategies."""

    symbols: Sequence[CodeSymbolLike]
    parse_context: Optional[TreeSitterParseContext] = None


class CodeChunkingStrategy(Protocol):
    """Strategy protocol for code chunk generation."""

    def chunk(
        self,
        file_path: str,
        content: str,
        context: CodeChunkingContext,
    ) -> list[CodeChunk]: ...


def _count_words(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(len(stripped.split()), 1)


def _match_symbol(
    symbols: Sequence[CodeSymbolLike],
    start_line: int,
    end_line: int,
) -> Optional[CodeSymbolLike]:
    containing = [
        symbol
        for symbol in symbols
        if symbol.line_start <= start_line and symbol.line_end >= end_line
    ]
    if containing:
        return min(
            containing,
            key=lambda symbol: (symbol.line_end - symbol.line_start, symbol.line_start),
        )

    best_symbol: Optional[CodeSymbolLike] = None
    best_ratio = 0.0
    span_length = max((end_line - start_line) + 1, 1)
    for symbol in symbols:
        overlap_start = max(start_line, symbol.line_start)
        overlap_end = min(end_line, symbol.line_end)
        if overlap_end < overlap_start:
            continue
        overlap = (overlap_end - overlap_start) + 1
        ratio = overlap / span_length
        if ratio > best_ratio:
            best_symbol = symbol
            best_ratio = ratio
    if best_ratio >= 0.6:
        return best_symbol
    return None


class SymbolSpanCodeChunker:
    """Chunk code by symbol spans with overlap-aware line splitting."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._chunk_size = max(int(chunk_size), 1)
        self._chunk_overlap = max(int(chunk_overlap), 0)

    def chunk(
        self,
        file_path: str,
        content: str,
        context: CodeChunkingContext,
    ) -> list[CodeChunk]:
        del file_path
        lines = content.splitlines()
        if not lines:
            return []

        chunks: list[CodeChunk] = []
        used_ranges: set[tuple[int, int]] = set()

        for symbol in sorted(
            context.symbols,
            key=lambda item: (item.line_start, item.line_end, item.name),
        ):
            start = max(symbol.line_start, 1)
            end = min(symbol.line_end, len(lines))
            if end < start:
                continue
            span = (start, end)
            if span in used_ranges:
                continue
            used_ranges.add(span)
            chunks.extend(
                self._split_line_span(
                    lines=lines,
                    start_line=start,
                    end_line=end,
                    chunk_type=symbol.symbol_type,
                    symbol_name=symbol.qualified_name,
                    parent_symbol=symbol.parent_symbol,
                )
            )

        if not chunks:
            chunks.extend(
                self._split_line_span(
                    lines=lines,
                    start_line=1,
                    end_line=len(lines),
                    chunk_type="module",
                    symbol_name=None,
                    parent_symbol=None,
                )
            )

        return chunks

    def _split_line_span(
        self,
        *,
        lines: list[str],
        start_line: int,
        end_line: int,
        chunk_type: str,
        symbol_name: Optional[str],
        parent_symbol: Optional[str],
    ) -> list[CodeChunk]:
        span_lines = lines[start_line - 1 : end_line]
        if not span_lines:
            return []

        chunks: list[CodeChunk] = []
        current: list[str] = []
        current_words = 0
        current_start = start_line
        overlap = self._chunk_overlap

        for index, line in enumerate(span_lines, start=start_line):
            current.append(line)
            current_words += max(len(line.split()), 1)
            if current_words < self._chunk_size:
                continue

            chunk_lines = list(current)
            chunks.append(
                CodeChunk(
                    content="\n".join(chunk_lines),
                    start_line=current_start,
                    end_line=index,
                    chunk_type=chunk_type,
                    symbol_name=symbol_name,
                    parent_symbol=parent_symbol,
                )
            )

            overlap_lines = chunk_lines[-overlap:] if overlap else []
            current = overlap_lines
            current_words = sum(max(len(item.split()), 1) for item in overlap_lines)
            current_start = max(index - len(overlap_lines) + 1, current_start)

        if current:
            chunks.append(
                CodeChunk(
                    content="\n".join(current),
                    start_line=current_start,
                    end_line=end_line,
                    chunk_type=chunk_type,
                    symbol_name=symbol_name,
                    parent_symbol=parent_symbol,
                )
            )

        return chunks


@dataclass(frozen=True)
class _ByteSpan:
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int


class TreeSitterStructuralCodeChunker:
    """AST-aware chunker that preserves non-symbol module context."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        fallback: Optional[SymbolSpanCodeChunker] = None,
    ) -> None:
        self._chunk_size = max(int(chunk_size), 1)
        self._chunk_overlap = max(int(chunk_overlap), 0)
        self._fallback = fallback or SymbolSpanCodeChunker(chunk_size, chunk_overlap)

    def chunk(
        self,
        file_path: str,
        content: str,
        context: CodeChunkingContext,
    ) -> list[CodeChunk]:
        parse_context = context.parse_context
        if parse_context is None or parse_context.root_node is None:
            return self._fallback.chunk(file_path, content, context)

        spans = self._chunk_node(parse_context.root_node, parse_context, is_root=True)
        chunks: list[CodeChunk] = []
        for span in spans:
            text = parse_context.slice(span.start_byte, span.end_byte)
            if not text.strip():
                continue
            symbol = _match_symbol(context.symbols, span.start_line, span.end_line)
            chunks.append(
                CodeChunk(
                    content=text,
                    start_line=span.start_line,
                    end_line=span.end_line,
                    chunk_type=symbol.symbol_type if symbol else "module",
                    symbol_name=symbol.qualified_name if symbol else None,
                    parent_symbol=symbol.parent_symbol if symbol else None,
                )
            )

        if not chunks:
            return self._fallback.chunk(file_path, parse_context.content, context)
        return chunks

    def _chunk_node(
        self,
        node: Any,
        parse_context: TreeSitterParseContext,
        *,
        is_root: bool,
    ) -> list[_ByteSpan]:
        if (
            not is_root
            and self._span_word_count(node.start_byte, node.end_byte, parse_context)
            <= self._chunk_size
        ):
            return [self._make_span(node.start_byte, node.end_byte, parse_context)]

        children = self._named_children(node)
        if not children:
            return self._split_large_span(node.start_byte, node.end_byte, parse_context)

        segments: list[_ByteSpan] = []
        cursor = node.start_byte
        for child in children:
            if child.start_byte > cursor:
                segments.append(self._make_span(cursor, child.start_byte, parse_context))
            segments.extend(self._chunk_node(child, parse_context, is_root=False))
            cursor = max(cursor, child.end_byte)
        if cursor < node.end_byte:
            segments.append(self._make_span(cursor, node.end_byte, parse_context))
        return self._merge_spans(segments, parse_context)

    def _named_children(self, node: Any) -> list[Any]:
        children = list(getattr(node, "named_children", []) or [])
        filtered = [
            child
            for child in children
            if getattr(child, "end_byte", 0) > getattr(child, "start_byte", 0)
        ]
        return sorted(filtered, key=lambda child: (child.start_byte, child.end_byte))

    def _merge_spans(
        self,
        spans: Sequence[_ByteSpan],
        parse_context: TreeSitterParseContext,
    ) -> list[_ByteSpan]:
        merged: list[_ByteSpan] = []
        current: Optional[_ByteSpan] = None

        for span in spans:
            text = parse_context.slice(span.start_byte, span.end_byte)
            if not text.strip():
                if current is not None:
                    current = self._make_span(current.start_byte, span.end_byte, parse_context)
                continue

            if current is None:
                current = span
                continue

            projected = self._make_span(current.start_byte, span.end_byte, parse_context)
            if (
                self._span_word_count(projected.start_byte, projected.end_byte, parse_context)
                <= self._chunk_size
            ):
                current = projected
                continue

            merged.extend(self._finalize_span(current, parse_context))
            current = span

        if current is not None:
            merged.extend(self._finalize_span(current, parse_context))
        return merged

    def _finalize_span(
        self,
        span: _ByteSpan,
        parse_context: TreeSitterParseContext,
    ) -> list[_ByteSpan]:
        if self._span_word_count(span.start_byte, span.end_byte, parse_context) <= self._chunk_size:
            return [span]
        return self._split_large_span(span.start_byte, span.end_byte, parse_context)

    def _split_large_span(
        self,
        start_byte: int,
        end_byte: int,
        parse_context: TreeSitterParseContext,
    ) -> list[_ByteSpan]:
        start_line = parse_context.byte_offset_to_line(start_byte)
        end_line = parse_context.byte_offset_to_line(max(end_byte - 1, start_byte))
        fallback_chunks = self._fallback._split_line_span(
            lines=parse_context.content.splitlines(),
            start_line=start_line,
            end_line=end_line,
            chunk_type="module",
            symbol_name=None,
            parent_symbol=None,
        )
        return [
            _ByteSpan(
                start_byte=self._line_start_byte(chunk.start_line, parse_context),
                end_byte=self._line_end_byte(chunk.end_line, parse_context),
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
            for chunk in fallback_chunks
        ]

    def _make_span(
        self,
        start_byte: int,
        end_byte: int,
        parse_context: TreeSitterParseContext,
    ) -> _ByteSpan:
        return _ByteSpan(
            start_byte=start_byte,
            end_byte=end_byte,
            start_line=parse_context.byte_offset_to_line(start_byte),
            end_line=parse_context.byte_offset_to_line(max(end_byte - 1, start_byte)),
        )

    def _span_word_count(
        self,
        start_byte: int,
        end_byte: int,
        parse_context: TreeSitterParseContext,
    ) -> int:
        return _count_words(parse_context.slice(start_byte, end_byte))

    def _line_start_byte(self, line_number: int, parse_context: TreeSitterParseContext) -> int:
        index = max(min(line_number - 1, len(parse_context.line_start_offsets) - 1), 0)
        return parse_context.line_start_offsets[index]

    def _line_end_byte(self, line_number: int, parse_context: TreeSitterParseContext) -> int:
        next_index = line_number
        if next_index < len(parse_context.line_start_offsets):
            return parse_context.line_start_offsets[next_index]
        return len(parse_context.source_bytes)


def create_code_chunker(
    strategy: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> CodeChunkingStrategy:
    """Create a code chunking strategy from configuration."""

    normalized = (strategy or "symbol_span").strip().lower()
    if normalized in {"symbol_span", "body_aware", "default"}:
        return SymbolSpanCodeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if normalized in {"tree_sitter_structural", "ast_structural", "cast"}:
        return TreeSitterStructuralCodeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    raise ValueError(f"Unknown code chunking strategy: {strategy}")


__all__ = [
    "CodeChunk",
    "CodeChunkingContext",
    "CodeChunkingStrategy",
    "CodeSymbolLike",
    "SymbolSpanCodeChunker",
    "TreeSitterParseContext",
    "TreeSitterStructuralCodeChunker",
    "create_code_chunker",
]
