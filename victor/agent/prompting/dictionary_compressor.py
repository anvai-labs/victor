# Copyright 2026 Vijaykumar Singh
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

"""Lossless dictionary compression for repeated prompt boilerplate."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class PromptCompressionResult:
    """Result of compressing repeated prompt blocks."""

    original_blocks: List[str]
    compressed_blocks: List[str]
    dictionary: Dict[str, str] = field(default_factory=dict)
    dictionary_heading: str = "## Reusable Guidance"
    compressed: bool = False
    saved_chars: int = 0

    @property
    def original_prompt(self) -> str:
        """Return the original prompt text."""
        return "\n\n".join(self.original_blocks)

    @property
    def compressed_prompt(self) -> str:
        """Render the compressed prompt text."""
        if not self.compressed:
            return self.original_prompt

        dictionary_blocks = [self.dictionary_heading]
        for alias, block in self.dictionary.items():
            dictionary_blocks.append(f"{alias}\n{block}")

        return "\n\n".join(dictionary_blocks + self.compressed_blocks)

    def expand(self) -> str:
        """Expand aliased blocks back to the original prompt text."""
        if not self.compressed:
            return self.original_prompt

        expanded_blocks = [self.dictionary.get(block, block) for block in self.compressed_blocks]
        return "\n\n".join(expanded_blocks)


def compress_prompt_blocks(
    blocks: Iterable[str],
    *,
    dictionary_heading: str = "## Reusable Guidance",
    min_block_chars: int = 80,
    min_occurrences: int = 2,
    max_entries: int = 6,
    min_savings_chars: int = 24,
) -> PromptCompressionResult:
    """Compress repeated prompt blocks using short aliases.

    The transformation is lossless via ``PromptCompressionResult.expand()`` and
    falls back automatically when compression would not materially reduce size.
    """
    normalized_blocks = [block.strip() for block in blocks if block and block.strip()]
    original_prompt = "\n\n".join(normalized_blocks)
    if not normalized_blocks:
        return PromptCompressionResult(
            original_blocks=[],
            compressed_blocks=[],
            dictionary_heading=dictionary_heading,
        )

    counts = Counter(normalized_blocks)
    candidates = [
        block
        for block, count in counts.items()
        if count >= min_occurrences and len(block) >= min_block_chars
    ]
    if not candidates:
        return PromptCompressionResult(
            original_blocks=normalized_blocks,
            compressed_blocks=list(normalized_blocks),
            dictionary_heading=dictionary_heading,
        )

    candidates.sort(
        key=lambda block: ((counts[block] - 1) * len(block), counts[block], len(block)),
        reverse=True,
    )
    selected = candidates[:max_entries]

    dictionary: Dict[str, str] = {}
    alias_by_block: Dict[str, str] = {}
    for index, block in enumerate(selected, start=1):
        alias = f"[[R{index}]]"
        dictionary[alias] = block
        alias_by_block[block] = alias

    compressed_blocks = [alias_by_block.get(block, block) for block in normalized_blocks]
    result = PromptCompressionResult(
        original_blocks=normalized_blocks,
        compressed_blocks=compressed_blocks,
        dictionary=dictionary,
        dictionary_heading=dictionary_heading,
        compressed=True,
    )

    saved_chars = len(original_prompt) - len(result.compressed_prompt)
    if saved_chars < min_savings_chars:
        return PromptCompressionResult(
            original_blocks=normalized_blocks,
            compressed_blocks=list(normalized_blocks),
            dictionary_heading=dictionary_heading,
        )

    return PromptCompressionResult(
        original_blocks=normalized_blocks,
        compressed_blocks=compressed_blocks,
        dictionary=dictionary,
        dictionary_heading=dictionary_heading,
        compressed=True,
        saved_chars=saved_chars,
    )


__all__ = ["PromptCompressionResult", "compress_prompt_blocks"]
