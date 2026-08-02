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

"""Streaming filter, thinking token detection, and circular thinking detection.

Native Dispatch Pattern Conventions:
=====================================
This module follows the victor.processing.native dispatch pattern:

1. if _NATIVE_AVAILABLE: - Standard pattern for stable functions
   Use when the native function is guaranteed to exist in all versions.

2. if _NATIVE_AVAILABLE and hasattr(_native, "func"): - Version guard pattern
   Use when the native function may not exist in older/newer versions of
   the native library (e.g., newly added features or renamed functions).

3. from victor.processing.native._base import _NATIVE_AVAILABLE, _native
   Always import both the flag and the module reference from _base.py.
"""

from __future__ import annotations

from victor.processing.native._base import _NATIVE_AVAILABLE, _native

# =============================================================================
# STREAMING FILTER (Thinking Token Detection)
# =============================================================================


class StreamingChunkResultFallback:
    """Fallback result class for streaming content filter."""

    def __init__(
        self,
        content: str,
        is_thinking: bool = False,
        state_changed: bool = False,
        entering_thinking: bool = False,
        exiting_thinking: bool = False,
    ):
        self.content = content
        self.is_thinking = is_thinking
        self.state_changed = state_changed
        self.entering_thinking = entering_thinking
        self.exiting_thinking = exiting_thinking


# Thinking token patterns
_THINKING_START_PATTERNS = [
    "<\uff5cbegin\u2581of\u2581thinking\uff5c>",  # DeepSeek Unicode
    "<|begin_of_thinking|>",  # DeepSeek ASCII
    "<think>",  # Qwen3
]

_THINKING_END_PATTERNS = [
    "<\uff5cend\u2581of\u2581thinking\uff5c>",  # DeepSeek Unicode
    "<|end_of_thinking|>",  # DeepSeek ASCII
    "</think>",  # Qwen3
]

_ALL_THINKING_PATTERNS = _THINKING_START_PATTERNS + _THINKING_END_PATTERNS


def strip_thinking_tokens(content: str) -> str:
    """Strip all thinking tokens from content.

    Args:
        content: Text potentially containing thinking tokens

    Returns:
        Content with thinking tokens removed
    """
    if _NATIVE_AVAILABLE:
        return _native.strip_thinking_tokens(content)

    # Pure Python fallback
    result = content
    for pattern in _ALL_THINKING_PATTERNS:
        result = result.replace(pattern, "")
    return result


def contains_thinking_tokens(content: str) -> bool:
    """Check if content contains any thinking tokens.

    Args:
        content: Text to check

    Returns:
        True if thinking tokens are present
    """
    if _NATIVE_AVAILABLE:
        return _native.contains_thinking_tokens(content)

    # Pure Python fallback
    return any(pattern in content for pattern in _ALL_THINKING_PATTERNS)


def find_thinking_tokens(content: str) -> list[tuple[int, int, int]]:
    """Find all thinking token positions in content.

    Args:
        content: Text to search

    Returns:
        List of (start, end, pattern_index) tuples
    """
    if _NATIVE_AVAILABLE:
        return _native.find_thinking_tokens(content)

    # Pure Python fallback
    results = []
    for idx, pattern in enumerate(_ALL_THINKING_PATTERNS):
        pos = 0
        while True:
            found = content.find(pattern, pos)
            if found == -1:
                break
            results.append((found, found + len(pattern), idx))
            pos = found + 1

    results.sort(key=lambda x: x[0])
    return results


def extract_thinking_content(content: str) -> tuple[str, str]:
    """Extract thinking content from a complete response.

    Args:
        content: Full response text

    Returns:
        Tuple of (main_content, thinking_content)
    """
    if _NATIVE_AVAILABLE:
        return _native.extract_thinking_content(content)

    # Pure Python fallback
    main_content = []
    thinking_content = []
    in_thinking = False
    pos = 0

    while pos < len(content):
        if not in_thinking:
            # Look for start pattern
            earliest_start = len(content)
            start_pattern_len = 0
            for pattern in _THINKING_START_PATTERNS:
                idx = content.find(pattern, pos)
                if idx != -1 and idx < earliest_start:
                    earliest_start = idx
                    start_pattern_len = len(pattern)

            if earliest_start < len(content):
                main_content.append(content[pos:earliest_start])
                in_thinking = True
                pos = earliest_start + start_pattern_len
            else:
                main_content.append(content[pos:])
                break
        else:
            # Look for end pattern
            earliest_end = len(content)
            end_pattern_len = 0
            for pattern in _THINKING_END_PATTERNS:
                idx = content.find(pattern, pos)
                if idx != -1 and idx < earliest_end:
                    earliest_end = idx
                    end_pattern_len = len(pattern)

            if earliest_end < len(content):
                thinking_content.append(content[pos:earliest_end])
                in_thinking = False
                pos = earliest_end + end_pattern_len
            else:
                thinking_content.append(content[pos:])
                break

    return "".join(main_content), "".join(thinking_content)


# Re-export native classes when available
StreamingFilter = None
StreamingChunkResult = StreamingChunkResultFallback

if _NATIVE_AVAILABLE:
    StreamingFilter = _native.StreamingFilter
    StreamingChunkResult = _native.StreamingChunkResult
else:
    # Use Python fallback from response_sanitizer
    try:
        from victor.agent.response_sanitizer import (
            StreamingContentFilter as StreamingFilter,
            StreamingChunkResult,
        )
    except ImportError:
        StreamingFilter = None
        StreamingChunkResult = StreamingChunkResultFallback


# =============================================================================
# THINKING DETECTOR
# =============================================================================

# Circular thinking patterns
_CIRCULAR_PATTERNS = [
    "let me read",
    "let me check",
    "let me look at",
    "let me examine",
    "let me see",
    "i need to read",
    "i need to check",
    "i need to look at",
    "first let me",
    "now let me",
    "let me first",
    "let me start by",
    "i'll need to",
    "i will need to",
    "let me actually use",
    "let me use the",
    "i'll actually read",
    "i'll read",
    "now i'll",
    "now i will",
    "i should read",
    "i should examine",
    "i should check",
    "let me continue",
    "let me proceed",
]


def detect_circular_phrases(text: str) -> bool:
    """Detect if text contains circular thinking phrases.

    Args:
        text: Text to check

    Returns:
        True if circular phrases are detected
    """
    if _NATIVE_AVAILABLE:
        return _native.detect_circular_phrases(text)

    # Pure Python fallback
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in _CIRCULAR_PATTERNS)


def count_circular_patterns(text: str) -> int:
    """Count circular patterns in text.

    Args:
        text: Text to analyze

    Returns:
        Number of circular pattern matches
    """
    if _NATIVE_AVAILABLE:
        return _native.count_circular_patterns(text)

    # Pure Python fallback
    text_lower = text.lower()
    count = 0
    for pattern in _CIRCULAR_PATTERNS:
        pos = 0
        while True:
            found = text_lower.find(pattern, pos)
            if found == -1:
                break
            count += 1
            pos = found + 1
    return count


def find_circular_patterns(text: str) -> list[tuple[int, int, str]]:
    """Find all circular pattern matches.

    Args:
        text: Text to search

    Returns:
        List of (start, end, matched_text) tuples
    """
    if _NATIVE_AVAILABLE:
        return _native.find_circular_patterns(text)

    # Pure Python fallback
    text_lower = text.lower()
    results = []
    for pattern in _CIRCULAR_PATTERNS:
        pos = 0
        while True:
            found = text_lower.find(pattern, pos)
            if found == -1:
                break
            end = found + len(pattern)
            results.append((found, end, text[found:end]))
            pos = found + 1
    results.sort(key=lambda x: x[0])
    return results


# Re-export native classes when available
ThinkingDetector = None
PatternAnalysis = None

if _NATIVE_AVAILABLE:
    ThinkingDetector = _native.ThinkingDetector
    PatternAnalysis = _native.PatternAnalysis
