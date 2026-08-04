# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""One canonical code chunker for vector indexing.

The former symbol-span and tree-sitter-structural engines duplicated parsing,
budgeting, and boundary logic.  All legacy configuration names now converge on
victor-codegraph v2; package failure uses the adapter's single bounded raw-text
fallback rather than another semantic parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from victor.core.codegraph_adapter import chunk_with_codegraph


@dataclass(frozen=True)
class CodeChunk:
    """Chunk of code prepared for vector indexing."""

    content: str
    start_line: int
    end_line: int
    chunk_type: str
    symbol_name: Optional[str] = None
    parent_symbol: Optional[str] = None
    chunk_id: Optional[str] = None
    symbol_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CodeChunkingStrategy(Protocol):
    """Canonical vector code-chunking contract."""

    def chunk(
        self,
        file_path: str,
        content: str,
        *,
        language: str | None = None,
    ) -> list[CodeChunk]: ...


def _count_words(text: str) -> int:
    stripped = text.strip()
    return max(len(stripped.split()), 1) if stripped else 0


class VictorCodegraphCodeChunker:
    """Canonical v2 chunk planner for all vector-store code paths."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._chunk_size = max(int(chunk_size), 1)
        self._chunk_overlap = max(int(chunk_overlap), 0)

    def chunk(
        self,
        file_path: str,
        content: str,
        *,
        language: str | None = None,
    ) -> list[CodeChunk]:
        return [
            CodeChunk(
                content=item.content,
                start_line=item.start_line,
                end_line=item.end_line,
                chunk_type=item.chunk_type,
                symbol_name=item.symbol_name,
                parent_symbol=item.parent_symbol,
                chunk_id=item.metadata.get("chunk_id"),
                symbol_id=item.metadata.get("symbol_id"),
                metadata=dict(item.metadata),
            )
            for item in chunk_with_codegraph(
                content,
                file_path=file_path,
                language=language,
                max_chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                token_counter=_count_words,
                chars_per_token=4.0,
            )
        ]


_CANONICAL_NAMES = {
    "victor_codegraph",
    "codegraph",
    "default",
    "body_aware",
    # Compatibility aliases for persisted configs; no legacy engine remains.
    "symbol_span",
    "tree_sitter_structural",
    "ast_structural",
    "cast",
}


def normalize_code_chunking_strategy(strategy: str | None) -> str:
    """Normalize every supported legacy name to the one implemented engine."""
    normalized = (strategy or "victor_codegraph").strip().lower()
    if normalized not in _CANONICAL_NAMES:
        raise ValueError(f"Unknown code chunking strategy: {strategy}")
    return "victor_codegraph"


def create_code_chunker(
    strategy: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> CodeChunkingStrategy:
    """Create the canonical chunker while accepting persisted legacy names."""
    normalize_code_chunking_strategy(strategy)
    return VictorCodegraphCodeChunker(chunk_size, chunk_overlap)


__all__ = [
    "CodeChunk",
    "CodeChunkingStrategy",
    "VictorCodegraphCodeChunker",
    "create_code_chunker",
    "normalize_code_chunking_strategy",
]
