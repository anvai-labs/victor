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

"""Prompt normalizer middleware for input deduplication.

This module provides normalization and deduplication of user prompts,
reducing token waste from repeated instructions and action verbs.

Design Principles:
- Single Responsibility: Only handles prompt normalization
- Stateful: Tracks recent messages for duplicate detection
- Non-destructive: Original intent is preserved
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set

from victor.framework.prompt_document import (
    PromptBlock,
    PromptDeduplicationProcessor,
    PromptDocument,
)

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """Result of prompt normalization.

    Attributes:
        normalized: The normalized prompt text
        changes: List of changes made (for logging/debugging)
        is_duplicate: True if prompt matches a recent message
        tokens_saved: Estimated tokens saved by normalization
    """

    normalized: str
    changes: List[str] = field(default_factory=list)
    is_duplicate: bool = False
    tokens_saved: int = 0


class PromptNormalizer:
    """Normalizes and deduplicates input prompts.

    The normalizer performs:
    1. Action verb canonicalization (view→read, check→read)
    2. Duplicate detection via content hashing
    3. Continuation message collapsing
    4. Section deduplication in prompt components

    Usage:
        normalizer = PromptNormalizer()
        result = normalizer.normalize("view the auth.py file")
        # result.normalized = "read the auth.py file"
        # result.changes = ["view→read"]

    Note: This normalizer is conservative - it only normalizes obvious
    synonyms and preserves the user's original intent.
    """

    # Action verb canonicalization map
    # Only includes unambiguous synonyms to avoid changing intent
    VERB_SYNONYMS: Dict[str, str] = {
        "view": "read",
        "look at": "read",
        "check": "read",
        "show": "read",
        "display": "read",
        "examine": "analyze",
        "review": "analyze",
        "inspect": "analyze",
    }

    # Patterns for continuation messages that can be collapsed
    CONTINUATION_PATTERNS: List[re.Pattern] = [
        re.compile(r"^continue\.?$", re.IGNORECASE),
        re.compile(r"^go on\.?$", re.IGNORECASE),
        re.compile(r"^proceed\.?$", re.IGNORECASE),
        re.compile(r"^yes\.?$", re.IGNORECASE),
        re.compile(r"^ok\.?$", re.IGNORECASE),
        re.compile(r"^keep going\.?$", re.IGNORECASE),
    ]

    def __init__(self, max_recent: int = 10):
        """Initialize PromptNormalizer.

        Args:
            max_recent: Maximum recent messages to track for deduplication
        """
        self._recent_hashes: Deque[str] = deque(maxlen=max_recent)
        self._recent_messages: Deque[str] = deque(maxlen=max_recent)
        self._continuation_count: int = 0

    def normalize(self, content: str) -> NormalizationResult:
        """Normalize prompt and check for duplicates.

        Args:
            content: Raw user prompt

        Returns:
            NormalizationResult with normalized text and metadata
        """
        changes: List[str] = []
        normalized = content.strip()
        original_length = len(content)

        # 1. Check for continuation messages
        if self._is_continuation(normalized):
            self._continuation_count += 1
            # After 3+ continuations, collapse to simple "continue"
            if self._continuation_count >= 3:
                normalized = "continue"
                changes.append(f"collapsed continuation #{self._continuation_count}")
        else:
            self._continuation_count = 0

        # 2. Normalize action verbs
        for synonym, canonical in self.VERB_SYNONYMS.items():
            # Use word boundaries to avoid partial matches
            pattern = rf"\b{re.escape(synonym)}\b"
            if re.search(pattern, normalized, re.IGNORECASE):
                normalized = re.sub(
                    pattern,
                    canonical,
                    normalized,
                    flags=re.IGNORECASE,
                )
                changes.append(f"{synonym}→{canonical}")

        # 3. Check for exact duplicate
        content_hash = self._hash_content(normalized)
        is_duplicate = content_hash in self._recent_hashes

        if not is_duplicate:
            self._recent_hashes.append(content_hash)
            self._recent_messages.append(normalized)
        else:
            logger.debug(f"Duplicate message detected (hash: {content_hash[:8]})")

        # 4. Calculate tokens saved (rough estimate: 1 token ≈ 4 chars)
        tokens_saved = max(0, (original_length - len(normalized)) // 4)

        if changes:
            logger.debug(f"Prompt normalized: {changes}")

        return NormalizationResult(
            normalized=normalized,
            changes=changes,
            is_duplicate=is_duplicate,
            tokens_saved=tokens_saved,
        )

    def _is_continuation(self, text: str) -> bool:
        """Check if text is a continuation message.

        Args:
            text: Text to check

        Returns:
            True if text matches a continuation pattern
        """
        text_stripped = text.strip()
        for pattern in self.CONTINUATION_PATTERNS:
            if pattern.match(text_stripped):
                return True
        return False

    def _hash_content(self, content: str) -> str:
        """Generate hash for content comparison.

        Args:
            content: Content to hash

        Returns:
            12-character hash string
        """
        # Normalize whitespace before hashing
        normalized = " ".join(content.split())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def deduplicate_sections(self, sections: List[str]) -> List[str]:
        """Remove duplicate sections from prompt components.

        Useful for deduplicating grounding rules or system prompt
        sections from multiple contributors.

        Args:
            sections: List of prompt sections

        Returns:
            List with duplicates removed (preserves order)
        """
        document = PromptDocument(
            PromptBlock(name=f"section_{index}", content=section, header="")
            for index, section in enumerate(sections)
        )
        unique = [
            block.content
            for block in PromptDeduplicationProcessor().process(document).iter_renderable_blocks()
        ]

        if len(unique) < len(sections):
            logger.info(f"Deduplicated prompt sections: {len(sections)} → {len(unique)}")

        return unique

    def is_similar_to_recent(self, content: str, threshold: float = 0.8) -> bool:
        """Check if content is similar to recent messages.

        Uses character-level similarity for fuzzy matching.

        Args:
            content: Content to check
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            True if content is similar to any recent message
        """
        if not self._recent_messages:
            return False

        content_normalized = " ".join(content.lower().split())

        for recent in self._recent_messages:
            recent_normalized = " ".join(recent.lower().split())
            similarity = self._calculate_similarity(content_normalized, recent_normalized)
            if similarity >= threshold:
                logger.debug(f"Similar message detected (similarity: {similarity:.2f})")
                return True

        return False

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate character-level similarity between strings.

        Uses a simple overlap coefficient for efficiency.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not s1 or not s2:
            return 0.0

        # Use character n-grams for comparison (n=3)
        def get_ngrams(s: str, n: int = 3) -> Set[str]:
            return set(s[i : i + n] for i in range(max(0, len(s) - n + 1)))

        ngrams1 = get_ngrams(s1)
        ngrams2 = get_ngrams(s2)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        min_size = min(len(ngrams1), len(ngrams2))

        return intersection / min_size if min_size > 0 else 0.0

    def reset(self) -> None:
        """Reset normalizer state.

        Call this when starting a new conversation.
        """
        self._recent_hashes.clear()
        self._recent_messages.clear()
        self._continuation_count = 0
        logger.debug("Prompt normalizer reset")

    def get_stats(self) -> Dict[str, int]:
        """Get normalizer statistics.

        Returns:
            Dictionary with tracked message count and continuation count
        """
        return {
            "tracked_messages": len(self._recent_messages),
            "continuation_count": self._continuation_count,
        }


# Singleton instance for convenience
_normalizer: Optional[PromptNormalizer] = None


def get_prompt_normalizer() -> PromptNormalizer:
    """Get singleton PromptNormalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = PromptNormalizer()
    return _normalizer


def reset_normalizer() -> None:
    """Reset the singleton normalizer instance."""
    global _normalizer
    if _normalizer is not None:
        _normalizer.reset()
