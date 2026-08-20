"""Chunking configuration — the size discipline both donor parsers needed.

ProximaDB's ``code.py`` had *no* size-capping (a huge function became one huge chunk);
LlamaIndex ``CodeSplitter`` and Victor's chunker both cap size. This config carries the
cap so the merged parser never emits an over-budget chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ChunkConfig:
    """Size + scope knobs for code chunking.

    The token budget is matched to the embedding model (BGE-small 384-d ~ 512 tokens),
    not an arbitrary char count. ``chars_per_token`` is conservative to avoid truncation.
    """

    # Size cap (the gap fix). A symbol whose body exceeds the budget is split.
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64
    chars_per_token: float = 3.5
    # Symbols below this many lines are never body-split (cheap, keep whole).
    large_symbol_threshold_lines: int = 30

    # Scope filters.
    include_private: bool = True
    include_tests: bool = True
    extract_relations: bool = True
    # Restrict to these languages (None = all detectable).
    languages: list[str] | None = None
    # When supplied, this is the authoritative hard-budget counter.
    token_counter: Callable[[str], int] | None = None

    # Computed budgets (chars), derived in __post_init__.
    max_chunk_chars: int = field(init=False, default=0)
    chunk_overlap_chars: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens cannot be negative")
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.max_chunk_chars = max(1, int(self.max_chunk_tokens * self.chars_per_token))
        self.chunk_overlap_chars = max(
            0,
            min(
                int(self.chunk_overlap_tokens * self.chars_per_token),
                self.max_chunk_chars - 1,
            ),
        )

    def estimate_tokens(self, text: str) -> int:
        """Conservative token estimate for ``text``."""

        if self.token_counter is not None:
            return int(self.token_counter(text))
        return int(len(text) / self.chars_per_token) + 1
