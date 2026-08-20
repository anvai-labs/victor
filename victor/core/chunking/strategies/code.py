# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Canonical code chunking strategy.

All semantic code chunking delegates to victor-codegraph v2 through one soft
adapter.  When the package is absent, that adapter provides bounded raw-text
windows; core no longer maintains a second regex parser.
"""

from __future__ import annotations

from typing import List

from victor.core.chunking.base import Chunk, ChunkingStrategy
from victor.core.codegraph_adapter import chunk_with_codegraph


class CodeChunkingStrategy(ChunkingStrategy):
    """AST-aligned, hard-capped code chunking with one shared fallback."""

    @property
    def name(self) -> str:
        return "code"

    @property
    def supported_types(self) -> List[str]:
        return [
            "code",
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "c",
            "cpp",
            "csharp",
        ]

    def chunk(
        self,
        content: str,
        *,
        source: str | None = None,
        language: str | None = None,
    ) -> List[Chunk]:
        if not content or not content.strip():
            return []
        file_path = source or "<unknown>"
        return [
            Chunk(
                content=item.content,
                start_char=item.start_char,
                end_char=item.end_char,
                chunk_type=f"code_{item.chunk_type}",
                metadata={
                    **item.metadata,
                    "symbol_name": item.symbol_name,
                    "parent_symbol": item.parent_symbol,
                    "file_path": file_path,
                },
            )
            for item in chunk_with_codegraph(
                content,
                file_path=file_path,
                language=language,
                max_chunk_size=self.config.max_chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        ]
