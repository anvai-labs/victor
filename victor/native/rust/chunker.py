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

"""Rust text chunker wrapper.

Provides a protocol-compliant wrapper around the Rust line-aware chunking
functions. The wrapper delegates to victor_native functions while maintaining
the TextChunkerProtocol interface.

Performance characteristics:
- count_lines: 5-10x faster (SIMD byte counting)
- find_line_boundaries: 3-5x faster (single pass)
- chunk_with_overlap: 3-5x faster (pre-computed line boundaries)
"""

from __future__ import annotations

from typing import List, Optional

import victor_native

from victor.native.observability import InstrumentedAccelerator
from victor.native.protocols import ChunkInfo


class RustTextChunker(InstrumentedAccelerator):
    """Rust implementation of TextChunkerProtocol.

    Wraps the high-performance Rust line-aware chunking functions
    with protocol-compliant interface.

    Performance characteristics:
    - count_lines: 5-10x faster (SIMD byte counting)
    - find_line_boundaries: 3-5x faster (single-pass iteration)
    - line_at_offset: 2-3x faster (binary search)
    - chunk_with_overlap: 3-5x faster (pre-computed boundaries)
    """

    def __init__(self) -> None:
        super().__init__(backend="rust")
        self._version = victor_native.__version__

    def get_version(self) -> Optional[str]:
        return self._version

    def chunk_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[ChunkInfo]:
        """Chunk text with overlap, respecting line boundaries.

        Delegates to Rust implementation with pre-computed line boundaries.

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

            # Call Rust implementation
            rust_chunks = victor_native.chunk_with_overlap(text, chunk_size, overlap)

            # Convert Rust ChunkInfoRust to protocol ChunkInfo
            return [
                ChunkInfo(
                    text=chunk.text,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    overlap_prev=chunk.overlap_prev,
                    metadata={"chunk_index": chunk.chunk_index},
                )
                for chunk in rust_chunks
            ]

    def count_lines(self, text: str) -> int:
        """Count lines in text (fast).

        Delegates to Rust SIMD-optimized implementation.

        Args:
            text: Text to count lines in

        Returns:
            Number of lines (newline count + 1)
        """
        with self._timed_call("line_counting"):
            return victor_native.count_lines(text)

    def find_line_boundaries(self, text: str) -> List[int]:
        """Find byte offsets of all line starts.

        Delegates to Rust single-pass implementation.

        Args:
            text: Text to analyze

        Returns:
            List of byte offsets where lines start (including 0)
        """
        with self._timed_call("line_boundary_detection"):
            return victor_native.find_line_boundaries(text)

    def line_at_offset(self, text: str, offset: int) -> int:
        """Get line number for a character offset.

        Delegates to Rust binary search implementation.

        Args:
            text: Text
            offset: Character offset

        Returns:
            Line number (1-indexed)
        """
        with self._timed_call("line_lookup"):
            return victor_native.line_at_offset(text, offset)
