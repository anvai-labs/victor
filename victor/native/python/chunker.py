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

"""Pure Python text chunker implementation.

Provides line-aware text chunking for code embedding.
"""

from __future__ import annotations

from typing import List, Optional

from victor.native.observability import InstrumentedAccelerator
from victor.native.protocols import ChunkInfo


class PythonTextChunker(InstrumentedAccelerator):
    """Pure Python implementation of TextChunkerProtocol.

    Provides line-aware chunking optimized for code.
    """

    def __init__(self) -> None:
        super().__init__(backend="python")
        self._version = "1.0.0"

    def get_version(self) -> Optional[str]:
        return self._version

    def chunk_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[ChunkInfo]:
        """Chunk text with overlap, respecting line boundaries.

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in characters
            overlap: Overlap size in characters

        Returns:
            List of ChunkInfo objects
        """
        with self._timed_call("text_chunking"):
            if not text:
                return []

            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive")

            if overlap >= chunk_size:
                raise ValueError("overlap must be less than chunk_size")

            # Pre-compute line boundaries for efficient line number lookup
            line_starts = self.find_line_boundaries(text)

            chunks = []
            pos = 0
            chunk_index = 0

            while pos < len(text):
                # Calculate chunk end
                chunk_end = min(pos + chunk_size, len(text))

                # If not at end, try to find a line boundary
                if chunk_end < len(text):
                    # Look for a newline within the chunk
                    last_newline = text.rfind("\n", pos, chunk_end)
                    if last_newline > pos:
                        chunk_end = last_newline + 1  # Include the newline

                # Extract chunk text
                chunk_text = text[pos:chunk_end]

                # Calculate line numbers
                start_line = self._line_at_offset_cached(line_starts, pos)
                end_line = self._line_at_offset_cached(line_starts, chunk_end - 1)

                # Calculate overlap with previous chunk
                overlap_prev = 0
                if chunk_index > 0 and pos > 0:
                    overlap_prev = min(overlap, pos)

                chunks.append(
                    ChunkInfo(
                        text=chunk_text,
                        start_line=start_line,
                        end_line=end_line,
                        start_offset=pos,
                        end_offset=chunk_end,
                        overlap_prev=overlap_prev,
                        metadata={"chunk_index": chunk_index},
                    )
                )

                # Move position forward, accounting for overlap
                step = max(1, chunk_size - overlap)
                pos += step
                chunk_index += 1

                # Adjust position to line boundary if we have overlap
                if pos < len(text) and overlap > 0:
                    # Find the start of the next line after pos
                    next_newline = text.find("\n", max(0, pos - overlap))
                    if next_newline >= 0 and next_newline < pos:
                        pos = next_newline + 1

            return chunks

    def count_lines(self, text: str) -> int:
        """Count lines in text (fast).

        Args:
            text: Text to count lines in

        Returns:
            Number of lines (newline count + 1)
        """
        with self._timed_call("line_counting"):
            if not text:
                return 0
            return text.count("\n") + 1

    def find_line_boundaries(self, text: str) -> List[int]:
        """Find byte offsets of all line starts.

        Args:
            text: Text to analyze

        Returns:
            List of byte offsets where lines start (including 0)
        """
        with self._timed_call("line_boundary_detection"):
            if not text:
                return []

            boundaries = [0]
            pos = 0
            while True:
                pos = text.find("\n", pos)
                if pos == -1:
                    break
                pos += 1
                if pos < len(text):
                    boundaries.append(pos)

            return boundaries

    def line_at_offset(self, text: str, offset: int) -> int:
        """Get line number for a character offset.

        Args:
            text: Text
            offset: Character offset

        Returns:
            Line number (1-indexed)
        """
        with self._timed_call("line_lookup"):
            if offset < 0 or not text:
                return 1
            if offset >= len(text):
                offset = len(text) - 1

            # Count newlines strictly BEFORE offset
            # (newline at offset position belongs to current line, not next)
            line = 1
            for i in range(offset):
                if text[i] == "\n":
                    line += 1
            return line

    def _line_at_offset_cached(self, line_starts: List[int], offset: int) -> int:
        """Get line number using pre-computed line boundaries.

        Uses binary search for O(log n) lookup.

        Args:
            line_starts: Pre-computed line start offsets
            offset: Character offset

        Returns:
            Line number (1-indexed)
        """
        if not line_starts:
            return 1

        # Binary search for the line containing offset
        low, high = 0, len(line_starts) - 1
        while low <= high:
            mid = (low + high) // 2
            if line_starts[mid] <= offset:
                if mid == len(line_starts) - 1 or line_starts[mid + 1] > offset:
                    return mid + 1  # 1-indexed
                low = mid + 1
            else:
                high = mid - 1

        return 1
