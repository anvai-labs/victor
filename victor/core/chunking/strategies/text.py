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

"""Text chunking strategy with sentence boundary awareness."""

import re
from typing import List, Optional

from victor.core.chunking.base import Chunk, ChunkingConfig, ChunkingStrategy


class TextChunkingStrategy(ChunkingStrategy):
    """Chunk plain text with sentence boundary awareness.

    This is the default/fallback strategy that respects sentence endings
    to avoid breaking mid-sentence.

    Example:
        strategy = TextChunkingStrategy()
        chunks = strategy.chunk("This is sentence one. This is sentence two.")
    """

    # Sentence ending patterns
    SENTENCE_END = re.compile(r"[.!?]\s+")

    # Paragraph boundary
    PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

    @property
    def name(self) -> str:
        return "text"

    @property
    def supported_types(self) -> List[str]:
        return ["text", "txt", "log", "plain"]

    def chunk(self, content: str) -> List[Chunk]:
        """Chunk text with sentence boundary awareness.

        Args:
            content: Text content to chunk

        Returns:
            List of Chunk objects
        """
        if not content or not content.strip():
            return []

        chunks = []
        current_pos = 0

        while current_pos < len(content):
            # Determine chunk end position
            chunk_end = min(current_pos + self.config.chunk_size, len(content))

            # Try to find a natural boundary
            if self.config.respect_boundaries and chunk_end < len(content):
                chunk_end = self._find_boundary(content, current_pos, chunk_end)

            # Extract chunk
            chunk_text = content[current_pos:chunk_end].strip()

            # Skip empty or too-small chunks
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        start_char=current_pos,
                        end_char=chunk_end,
                        chunk_type="text",
                    )
                )

            # Move to next chunk with overlap
            current_pos = chunk_end - self.config.chunk_overlap
            if current_pos >= len(content) - self.config.min_chunk_size:
                break

        return chunks

    def _find_boundary(self, content: str, start: int, end: int) -> int:
        """Find the best boundary near the target end position.

        Priority:
        1. Paragraph break
        2. Sentence end
        3. Original position

        Args:
            content: Full content
            start: Chunk start position
            end: Target end position

        Returns:
            Adjusted end position at a natural boundary
        """
        # Search window - look in the last portion of the chunk
        search_start = max(start + self.config.min_chunk_size, end - 200)
        search_text = content[search_start : end + 100]

        # Try paragraph break first
        para_match = None
        for m in self.PARAGRAPH_BREAK.finditer(search_text):
            para_match = m

        if para_match:
            return search_start + para_match.end()

        # Try sentence end
        sent_match = None
        for m in self.SENTENCE_END.finditer(search_text):
            sent_match = m

        if sent_match:
            return search_start + sent_match.end()

        return end
