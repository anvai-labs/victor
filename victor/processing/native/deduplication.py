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

"""Deduplication and hashing functions with native acceleration."""

import hashlib
import re
from typing import Any, Dict, List, Tuple

from victor.processing.native._base import _NATIVE_AVAILABLE, _native

# =============================================================================
# DEDUPLICATION FUNCTIONS
# =============================================================================


def normalize_block(block: str) -> str:
    """Normalize a content block for consistent hashing.

    Args:
        block: The content block to normalize

    Returns:
        Normalized string (lowercase, collapsed whitespace, no trailing punctuation)
    """
    if _NATIVE_AVAILABLE:
        return _native.normalize_block(block)

    # Pure Python fallback
    normalized = block.strip()
    if not normalized:
        return ""

    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized)
    # Remove trailing punctuation
    normalized = normalized.rstrip(".,;:")
    return normalized.lower()


def rolling_hash_blocks(content: str, min_block_length: int = 50) -> List[Tuple[str, str, bool]]:
    """Process content and compute hashes for all blocks.

    Args:
        content: The full content to process
        min_block_length: Minimum length for a block to be considered for dedup

    Returns:
        List of tuples: (hash string, original block, is_duplicate boolean)
    """
    if _NATIVE_AVAILABLE:
        return _native.rolling_hash_blocks(content, min_block_length)

    # Pure Python fallback
    blocks = _split_into_blocks(content)
    seen_hashes: set = set()
    results: List[Tuple[str, str, bool]] = []

    for block in blocks:
        if len(block.strip()) < min_block_length:
            results.append(("", block, False))
            continue

        block_hash = _hash_block(block)
        is_duplicate = block_hash in seen_hashes
        if not is_duplicate:
            seen_hashes.add(block_hash)

        results.append((block_hash, block, is_duplicate))

    return results


def find_duplicate_blocks(content: str, min_block_length: int = 50) -> List[Tuple[int, str]]:
    """Find duplicate blocks in content and return their indices.

    Args:
        content: The content to analyze
        min_block_length: Minimum length for dedup consideration

    Returns:
        List of (block_index, hash) for duplicate blocks
    """
    if _NATIVE_AVAILABLE:
        return _native.find_duplicate_blocks(content, min_block_length)

    # Pure Python fallback
    blocks = _split_into_blocks(content)
    seen_hashes: set = set()
    duplicates: List[Tuple[int, str]] = []

    for idx, block in enumerate(blocks):
        if len(block.strip()) < min_block_length:
            continue

        block_hash = _hash_block(block)
        if block_hash in seen_hashes:
            duplicates.append((idx, block_hash))
        else:
            seen_hashes.add(block_hash)

    return duplicates


def _split_into_blocks(content: str) -> List[str]:
    """Split content into logical blocks (pure Python)."""
    if not content:
        return []

    # Preserve code blocks as single units
    code_block_pattern = r"```[\s\S]*?```"
    code_blocks = re.findall(code_block_pattern, content)
    content_with_placeholders = re.sub(code_block_pattern, "<<<CODE_BLOCK>>>", content)

    # Split on paragraph breaks
    raw_blocks = re.split(r"\n{2,}", content_with_placeholders)

    # Replace placeholders
    final_blocks = []
    code_idx = 0
    for block in raw_blocks:
        if "<<<CODE_BLOCK>>>" in block:
            if code_idx < len(code_blocks):
                final_blocks.append(code_blocks[code_idx])
                code_idx += 1
        elif block.strip():
            final_blocks.append(block)

    return final_blocks


def _hash_block(block: str) -> str:
    """Compute hash of a normalized block (pure Python)."""
    normalized = normalize_block(block)
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


# =============================================================================
# HASHING FUNCTIONS
# =============================================================================


def compute_signature(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Compute a signature hash for a tool call.

    Args:
        tool_name: Name of the tool being called
        arguments: Tool arguments dictionary

    Returns:
        16-character hex string signature
    """
    if _NATIVE_AVAILABLE:
        return _native.compute_signature(tool_name, arguments)

    # Pure Python fallback
    # Sort keys for deterministic output
    sorted_args = sorted(arguments.items())
    args_str = ",".join(f"{k}={_value_to_str(v)}" for k, v in sorted_args)
    combined = f"{tool_name}:{args_str}"
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def compute_batch_signatures(tool_calls: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    """Compute signatures for multiple tool calls in batch.

    Args:
        tool_calls: List of (tool_name, arguments_dict) tuples

    Returns:
        List of signature strings, one per tool call
    """
    if _NATIVE_AVAILABLE:
        return _native.compute_batch_signatures(tool_calls)

    # Pure Python fallback
    return [compute_signature(name, args) for name, args in tool_calls]


def signature_similarity(sig1: str, sig2: str) -> float:
    """Compute similarity between two signatures.

    Args:
        sig1: First signature
        sig2: Second signature

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if _NATIVE_AVAILABLE:
        return _native.signature_similarity(sig1, sig2)

    # Pure Python fallback
    if sig1 == sig2:
        return 1.0
    if len(sig1) != len(sig2):
        return 0.0

    matching = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matching / len(sig1)


def _value_to_str(value: Any) -> str:
    """Convert a value to a stable string representation."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)
