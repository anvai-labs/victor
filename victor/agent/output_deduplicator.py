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

"""Output deduplicator for removing repeated content blocks.

This module addresses the issue where some LLM providers (notably Grok/xAI)
may repeat content blocks in their output. The deduplicator uses content
hashing to identify and remove duplicate paragraphs while preserving the
original structure.

Design Pattern: Single Responsibility
====================================
This component has one job: detect and remove duplicate content blocks
from streaming or complete responses.

Usage:
    dedup = OutputDeduplicator()

    # Process complete response
    clean_response = dedup.process(response_with_duplicates)

    # Process streaming chunks
    for chunk in response_stream:
        clean_chunk = dedup.process_chunk(chunk)
        if clean_chunk:
            yield clean_chunk

    # Get statistics
    stats = dedup.get_stats()
    print(f"Removed {stats['duplicates_removed']} duplicate blocks")
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Set

# Import native extensions with fallback
try:
    from victor.processing.native import (
        normalize_block as native_normalize_block,
        rolling_hash_blocks as native_rolling_hash_blocks,
        is_native_available,
    )

    _NATIVE_AVAILABLE = is_native_available()
except ImportError:
    _NATIVE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationStats:
    """Statistics for deduplication operations.

    Attributes:
        total_blocks: Total content blocks processed
        duplicates_removed: Number of duplicate blocks removed
        bytes_saved: Approximate bytes saved by deduplication
        unique_hashes: Set of unique content hashes seen
    """

    total_blocks: int = 0
    duplicates_removed: int = 0
    bytes_saved: int = 0
    unique_hashes: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_blocks": self.total_blocks,
            "duplicates_removed": self.duplicates_removed,
            "bytes_saved": self.bytes_saved,
            "unique_count": len(self.unique_hashes),
            "dedup_ratio": (
                self.duplicates_removed / self.total_blocks if self.total_blocks > 0 else 0.0
            ),
        }


class OutputDeduplicator:
    """Remove duplicate content blocks from LLM responses.

    The deduplicator works by:
    1. Splitting content into logical blocks (paragraphs, list items, code blocks)
    2. Computing a normalized hash for each block
    3. Keeping only the first occurrence of each unique block
    4. Reconstructing the output with duplicates removed

    Attributes:
        min_block_length: Minimum length for a block to be considered for dedup
        normalize_whitespace: Whether to normalize whitespace before hashing
    """

    def __init__(
        self,
        min_block_length: int = 50,
        normalize_whitespace: bool = True,
    ):
        """Initialize the deduplicator.

        Args:
            min_block_length: Minimum characters for a block to be deduplicated.
                             Shorter blocks are always kept to avoid removing
                             important short content like headers or separators.
            normalize_whitespace: If True, normalize whitespace before hashing
                                 to catch near-duplicates with spacing differences.
        """
        from victor.core.utils.content_hasher import ContentHasher

        self._min_block_length = min_block_length
        self._normalize_whitespace = normalize_whitespace
        self._seen_hashes: Set[str] = set()
        self._stats = DeduplicationStats()
        self._partial_block: str = ""
        # Use ContentHasher for consistent hashing across components
        self._hasher = ContentHasher(
            normalize_whitespace=True,
            case_insensitive=True,
            hash_length=12,
        )

    def reset(self) -> None:
        """Reset the deduplicator state for a new response."""
        self._seen_hashes = set()
        self._stats = DeduplicationStats()
        self._partial_block = ""

    def _normalize_block(self, block: str) -> str:
        """Normalize a block for consistent hashing.

        Uses native Rust implementation when available for ~10x speedup.

        Args:
            block: Raw content block

        Returns:
            Normalized version for hashing
        """
        if _NATIVE_AVAILABLE:
            return native_normalize_block(block)

        # Python fallback
        normalized = block.strip()

        if self._normalize_whitespace:
            # Collapse multiple whitespace to single space
            normalized = re.sub(r"\s+", " ", normalized)

        # Remove trailing punctuation differences
        normalized = normalized.rstrip(".,;:")

        return normalized.lower()

    def _hash_block(self, block: str) -> str:
        """Compute a hash for a content block.

        Args:
            block: Content block to hash

        Returns:
            Hex digest of the block's hash
        """
        normalized = self._normalize_block(block)
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def _split_into_blocks(self, content: str) -> list[str]:
        """Split content into logical blocks.

        Blocks are separated by:
        - Double newlines (paragraph breaks)
        - Numbered list items
        - Code block boundaries

        Args:
            content: Full content to split

        Returns:
            List of content blocks
        """
        # First, preserve code blocks as single units
        code_block_pattern = r"```[\s\S]*?```"
        code_blocks = re.findall(code_block_pattern, content)
        content_with_placeholders = re.sub(code_block_pattern, "<<<CODE_BLOCK>>>", content)

        # Split on paragraph breaks (2+ newlines)
        raw_blocks = re.split(r"\n{2,}", content_with_placeholders)

        # Further split numbered lists
        final_blocks = []
        for block in raw_blocks:
            if block.strip() == "<<<CODE_BLOCK>>>":
                # Replace placeholder with actual code block
                if code_blocks:
                    final_blocks.append(code_blocks.pop(0))
            else:
                # Check for numbered list items
                list_pattern = r"(?=\n\s*\d+\.\s)"
                list_items = re.split(list_pattern, block)
                final_blocks.extend([b for b in list_items if b.strip()])

        return final_blocks

    def process(self, content: str) -> str:
        """Process a complete response and remove duplicates.

        Uses native Rust implementation when available for 10-50x speedup.

        Args:
            content: Complete response content

        Returns:
            Response with duplicate blocks removed
        """
        if not content or not content.strip():
            return content

        # Use native implementation when available
        if _NATIVE_AVAILABLE:
            return self._process_native(content)

        # Python fallback
        return self._process_python(content)

    def _process_native(self, content: str) -> str:
        """Process using native Rust implementation."""
        results = native_rolling_hash_blocks(content, self._min_block_length)
        unique_blocks = []

        for block_hash, block, is_duplicate in results:
            self._stats.total_blocks += 1
            duplicate_across_calls = bool(block_hash) and block_hash in self._seen_hashes

            if is_duplicate or duplicate_across_calls:
                self._stats.duplicates_removed += 1
                self._stats.bytes_saved += len(block)
                logger.debug(
                    f"Removed duplicate block (hash={block_hash[:8]}): " f"{block[:50]}..."
                )
            else:
                if block_hash:
                    self._seen_hashes.add(block_hash)
                    self._stats.unique_hashes.add(block_hash)
                unique_blocks.append(block)

        result = "\n\n".join(unique_blocks)

        if self._stats.duplicates_removed > 0:
            logger.info(
                f"Deduplication (native): removed {self._stats.duplicates_removed} "
                f"duplicate blocks, saved ~{self._stats.bytes_saved} bytes"
            )

        return result

    def _process_python(self, content: str) -> str:
        """Process using Python fallback implementation."""
        blocks = self._split_into_blocks(content)
        unique_blocks = []

        for block in blocks:
            self._stats.total_blocks += 1

            # Keep short blocks unconditionally
            if len(block.strip()) < self._min_block_length:
                unique_blocks.append(block)
                continue

            block_hash = self._hash_block(block)

            if block_hash not in self._seen_hashes:
                self._seen_hashes.add(block_hash)
                self._stats.unique_hashes.add(block_hash)
                unique_blocks.append(block)
            else:
                # Duplicate found
                self._stats.duplicates_removed += 1
                self._stats.bytes_saved += len(block)
                logger.debug(
                    f"Removed duplicate block (hash={block_hash[:8]}): " f"{block[:50]}..."
                )

        # Reconstruct with preserved formatting
        result = "\n\n".join(unique_blocks)

        if self._stats.duplicates_removed > 0:
            logger.info(
                f"Deduplication: removed {self._stats.duplicates_removed} "
                f"duplicate blocks, saved ~{self._stats.bytes_saved} bytes"
            )

        return result

    def process_chunk(self, chunk: str) -> str:
        """Process a streaming chunk with deduplication.

        This method accumulates partial blocks and processes
        complete blocks as they become available.

        Args:
            chunk: Streaming chunk of content

        Returns:
            Deduplicated portion of content, may be empty if
            waiting for more content to complete a block
        """
        self._partial_block += chunk

        # Check for complete blocks (paragraph boundaries)
        if "\n\n" not in self._partial_block:
            # No complete block yet, keep accumulating
            return ""

        # Split at the last paragraph break
        parts = self._partial_block.rsplit("\n\n", 1)

        if len(parts) == 2:
            complete, remainder = parts
            self._partial_block = remainder

            # Process the complete portion
            return self.process(complete) + "\n\n"
        else:
            return ""

    def flush(self) -> str:
        """Flush any remaining partial content.

        Call this at the end of streaming to process
        the final block.

        Returns:
            Any remaining deduplicated content
        """
        if self._partial_block:
            result = self.process(self._partial_block)
            self._partial_block = ""
            return result
        return ""

    def get_stats(self) -> dict:
        """Get deduplication statistics.

        Returns:
            Dictionary of deduplication statistics
        """
        return self._stats.to_dict()

    @property
    def duplicates_removed(self) -> int:
        """Get count of duplicates removed."""
        return self._stats.duplicates_removed


class StreamingDeduplicator:
    """Streaming-optimized deduplicator for real-time output.

    This variant uses a sliding window approach suitable for
    streaming responses where content arrives in small chunks.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_block_length: int = 50,
    ):
        """Initialize the streaming deduplicator.

        Args:
            window_size: Number of recent blocks to keep for comparison
            min_block_length: Minimum length for deduplication consideration
        """
        from victor.core.utils.content_hasher import ContentHasher

        self._window_size = window_size
        self._min_block_length = min_block_length
        self._recent_hashes: list[str] = []
        self._buffer: str = ""
        self._stats = DeduplicationStats()
        # Use ContentHasher for consistent hashing across components
        self._hasher = ContentHasher(
            normalize_whitespace=True,
            case_insensitive=True,
            hash_length=12,
        )

    def reset(self) -> None:
        """Reset state for a new response."""
        self._recent_hashes = []
        self._buffer = ""
        self._stats = DeduplicationStats()

    def _hash_block(self, block: str) -> str:
        """Hash a block of content."""
        normalized = re.sub(r"\s+", " ", block.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def add_chunk(self, chunk: str) -> Optional[str]:
        """Add a chunk and return deduplicated output.

        Args:
            chunk: New content chunk

        Returns:
            Deduplicated content to output, or None if buffering
        """
        self._buffer += chunk

        # Look for paragraph boundaries
        if "\n\n" not in self._buffer:
            return None

        # Process complete paragraphs
        parts = self._buffer.rsplit("\n\n", 1)
        complete = parts[0]
        self._buffer = parts[1] if len(parts) > 1 else ""

        # Check each paragraph
        paragraphs = complete.split("\n\n")
        output_paragraphs = []

        for para in paragraphs:
            self._stats.total_blocks += 1

            if len(para.strip()) < self._min_block_length:
                output_paragraphs.append(para)
                continue

            block_hash = self._hash_block(para)

            if block_hash not in self._recent_hashes:
                self._recent_hashes.append(block_hash)
                # Maintain window size
                if len(self._recent_hashes) > self._window_size:
                    self._recent_hashes.pop(0)
                output_paragraphs.append(para)
            else:
                self._stats.duplicates_removed += 1
                self._stats.bytes_saved += len(para)

        if output_paragraphs:
            return "\n\n".join(output_paragraphs) + "\n\n"
        return ""

    def flush(self) -> str:
        """Flush remaining buffer content."""
        if self._buffer:
            result = self._buffer
            self._buffer = ""
            return result
        return ""

    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        return self._stats.to_dict()
