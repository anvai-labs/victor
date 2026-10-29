"""Shared, dependency-free text chunking core.

The framework ingestion chunker (``victor.framework.ingestion``) and the RAG
vertical's ``DocumentChunker`` carried identical copies of the plain-text,
markdown and code chunking strategies. This module is the single home for
that logic so both can delegate to it while keeping their own public APIs,
format-specific strategies (HTML, JSON) and return models.

Only the standard library is used, so the module works in an isolated SDK
install without the Victor host runtime.

Usage::

    from victor_contracts.chunking import ChunkingConfig, TextChunker

    chunker = TextChunker(ChunkingConfig(chunk_size=1000))
    for text, start, end in chunker.chunk_markdown(content):
        ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

__all__ = ["ChunkSpan", "ChunkingConfig", "TextChunker"]


@dataclass
class ChunkingConfig:
    """Configuration for document chunking.

    Defaults are tuned for the BGE embedding model (BAAI/bge-small-en-v1.5):
    384 dimensions and a 512-token context, so ``chunk_size`` is 3.5x the
    dimension (1344 characters, about 336 tokens).

    Attributes:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.
        min_chunk_size: Minimum chunk size; smaller pieces are skipped.
        max_chunk_size: Maximum section size before it is split further.
        respect_sentence_boundaries: Try to break at sentence ends.
        respect_paragraph_boundaries: Try to break at paragraphs.
        code_aware: Use code-aware chunking for code files.
    """

    chunk_size: int = 1344
    chunk_overlap: int = 134
    min_chunk_size: int = 200
    max_chunk_size: int = 2000
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    code_aware: bool = True

    def __post_init__(self) -> None:
        """Reject configurations ``chunk_text`` cannot terminate with.

        Each step advances to ``chunk_end - chunk_overlap``. With
        ``chunk_overlap >= chunk_size`` the window never moves, and with
        ``chunk_overlap > min_chunk_size`` the final window (or a sentence
        break found ``min_chunk_size`` past the start) is re-read forever, so
        both loop indefinitely on ordinary input and are refused up front.

        Raises:
            ValueError: If ``chunk_size`` is not positive, ``min_chunk_size``
                is negative, or ``chunk_overlap`` is negative, not smaller than
                ``chunk_size``, or larger than ``min_chunk_size``.
        """
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.min_chunk_size < 0:
            raise ValueError(f"min_chunk_size must be >= 0, got {self.min_chunk_size}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size, got "
                f"chunk_overlap={self.chunk_overlap}, chunk_size={self.chunk_size}"
            )
        if self.chunk_overlap > self.min_chunk_size:
            raise ValueError(
                "chunk_overlap must not exceed min_chunk_size, got "
                f"chunk_overlap={self.chunk_overlap}, min_chunk_size={self.min_chunk_size}"
            )


class ChunkSpan(NamedTuple):
    """One chunk of a source document.

    A ``NamedTuple``, so it is immutable and unpacks as
    ``(text, start_char, end_char)``.

    Attributes:
        text: Chunk text (whitespace-stripped).
        start_char: Start offset in the source content.
        end_char: End offset in the source content.
    """

    text: str
    start_char: int
    end_char: int


class TextChunker:
    """Plain-text, markdown and code chunking strategies.

    Strategies call each other through ``self`` (markdown and code sections
    that exceed ``max_chunk_size`` are split with :meth:`chunk_text`), so a
    subclass override of one strategy applies everywhere it is used.
    """

    SENTENCE_END = re.compile(r"[.!?]\s+")
    FUNCTION_DEF = re.compile(r"^(def|async def|function|fn|func)\s+\w+", re.MULTILINE)
    CLASS_DEF = re.compile(r"^(class|struct|interface|impl)\s+\w+", re.MULTILINE)
    MARKDOWN_HEADER = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialize the chunker.

        Args:
            config: Chunking configuration (defaults when ``None``).
        """
        self._config = config or ChunkingConfig()

    @property
    def config(self) -> ChunkingConfig:
        """The chunking configuration."""
        return self._config

    def chunk_text(self, content: str) -> list[ChunkSpan]:
        """Chunk plain text, preferring sentence boundaries.

        Args:
            content: Text content.

        Returns:
            Chunk spans in document order.
        """
        chunks: list[ChunkSpan] = []
        current_pos = 0

        while current_pos < len(content):
            chunk_end = min(current_pos + self._config.chunk_size, len(content))

            # Try to find a sentence boundary
            if self._config.respect_sentence_boundaries and chunk_end < len(content):
                search_start = max(current_pos + self._config.min_chunk_size, chunk_end - 100)
                search_text = content[search_start : chunk_end + 50]

                match = None
                for m in self.SENTENCE_END.finditer(search_text):
                    match = m

                if match:
                    chunk_end = search_start + match.end()

            chunk_text = content[current_pos:chunk_end].strip()

            if len(chunk_text) >= self._config.min_chunk_size:
                chunks.append(ChunkSpan(chunk_text, current_pos, chunk_end))

            current_pos = chunk_end - self._config.chunk_overlap
            if current_pos >= len(content) - self._config.min_chunk_size:
                break

        return chunks

    def chunk_markdown(self, content: str) -> list[ChunkSpan]:
        """Chunk markdown into header-delimited sections.

        Content without headers is chunked as plain text.

        Args:
            content: Markdown content.

        Returns:
            Chunk spans in document order.
        """
        chunks: list[ChunkSpan] = []
        headers = list(self.MARKDOWN_HEADER.finditer(content))

        if not headers:
            return self.chunk_text(content)

        for i, header in enumerate(headers):
            start = header.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)

            self._append_section(chunks, content[start:end].strip(), start, end)

        return chunks

    def chunk_code(self, content: str) -> list[ChunkSpan]:
        """Chunk code at function/class definition boundaries.

        Code without recognised definitions is chunked as plain text.

        Args:
            content: Code content.

        Returns:
            Chunk spans in document order.
        """
        chunks: list[ChunkSpan] = []
        definitions: list[tuple[str, int]] = []

        for match in self.FUNCTION_DEF.finditer(content):
            definitions.append(("function", match.start()))
        for match in self.CLASS_DEF.finditer(content):
            definitions.append(("class", match.start()))

        definitions.sort(key=lambda x: x[1])

        if not definitions:
            return self.chunk_text(content)

        # Add content before first definition
        if definitions[0][1] > self._config.min_chunk_size:
            pre_content = content[: definitions[0][1]].strip()
            if len(pre_content) >= self._config.min_chunk_size:
                chunks.append(ChunkSpan(pre_content, 0, definitions[0][1]))

        for i, (_def_type, start) in enumerate(definitions):
            end = definitions[i + 1][1] if i + 1 < len(definitions) else len(content)
            self._append_section(chunks, content[start:end].strip(), start, end)

        return chunks

    def _append_section(self, chunks: list[ChunkSpan], section: str, start: int, end: int) -> None:
        """Append one structural section, splitting it with :meth:`chunk_text`.

        Sections longer than ``max_chunk_size`` are re-chunked (offsets shifted
        by ``start``); sections shorter than ``min_chunk_size`` are dropped.
        """
        if len(section) > self._config.max_chunk_size:
            sub_chunks = self.chunk_text(section)
            for sub_text, sub_start, sub_end in sub_chunks:
                chunks.append(ChunkSpan(sub_text, start + sub_start, start + sub_end))
        elif len(section) >= self._config.min_chunk_size:
            chunks.append(ChunkSpan(section, start, end))

    def estimate_chunks(self, content: str) -> int:
        """Estimate the number of chunks for content.

        Args:
            content: Content to estimate.

        Returns:
            Estimated chunk count (0 for empty content).
        """
        if not content:
            return 0

        effective_size = self._config.chunk_size - self._config.chunk_overlap
        return max(1, len(content) // effective_size)
