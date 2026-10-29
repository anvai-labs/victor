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

"""Canonical prompt document model and processors."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, Protocol, Tuple


@dataclass
class PromptBlock:
    """Canonical block of prompt content."""

    name: str
    content: str
    priority: int = 50
    enabled: bool = True
    header: Optional[str] = None
    kind: str = "section"

    def get_formatted_content(self) -> str:
        """Render the block with its header."""
        if not self.enabled:
            return ""

        if self.header is None:
            header_text = f"## {self.name.replace('_', ' ').title()}\n"
        elif self.header == "":
            header_text = ""
        else:
            header_text = f"{self.header}\n"

        return f"{header_text}{self.content}"


class PromptDocumentProcessor(Protocol):
    """Processor that transforms a prompt document."""

    def process(self, document: "PromptDocument") -> "PromptDocument":
        """Return a transformed prompt document."""


class PromptDocument:
    """Canonical prompt document used by prompt builders and pipelines."""

    def __init__(self, blocks: Optional[Iterable[PromptBlock]] = None) -> None:
        self._blocks: "OrderedDict[str, PromptBlock]" = OrderedDict()
        for block in blocks or []:
            self.upsert(block)

    @property
    def blocks(self) -> "OrderedDict[str, PromptBlock]":
        """Expose blocks for backward-compatible integrations."""
        return self._blocks

    def clone(self) -> "PromptDocument":
        """Return a shallow clone of the document."""
        return PromptDocument(replace(block) for block in self._blocks.values())

    def upsert(self, block: PromptBlock) -> "PromptDocument":
        """Insert or replace a block by name."""
        self._blocks[block.name] = block
        return self

    def remove(self, name: str) -> "PromptDocument":
        """Remove a block if present."""
        self._blocks.pop(name, None)
        return self

    def has_block(self, name: str) -> bool:
        """Check whether a named block exists with non-blank content."""
        block = self._blocks.get(name)
        return block is not None and bool(block.content.strip())

    def get_block(self, name: str) -> Optional[PromptBlock]:
        """Return a named block when present."""
        return self._blocks.get(name)

    def iter_blocks(self) -> List[PromptBlock]:
        """Return all blocks in insertion order."""
        return list(self._blocks.values())

    def iter_named_blocks(self) -> List[Tuple[str, PromptBlock]]:
        """Return all named blocks in insertion order."""
        return list(self._blocks.items())

    def iter_renderable_blocks(self) -> List[PromptBlock]:
        """Return enabled blocks sorted by priority."""
        return sorted(
            (block for block in self._blocks.values() if block.enabled),
            key=lambda block: block.priority,
        )

    def apply(self, processor: PromptDocumentProcessor) -> "PromptDocument":
        """Apply a processor and return the resulting document."""
        return processor.process(self)

    def estimate_length(self) -> int:
        """Estimate rendered character length."""
        return sum(len(block.get_formatted_content()) for block in self.iter_renderable_blocks())

    def render(self) -> str:
        """Render the full prompt document."""
        rendered = [block.get_formatted_content() for block in self.iter_renderable_blocks()]
        return "\n\n".join(part for part in rendered if part)


class PromptDeduplicationProcessor:
    """Remove empty blocks and duplicate enabled content, keeping its first occurrence.

    Nonempty disabled blocks are copied but never claim content in the visible
    prompt. They must not suppress an enabled block or lose their configuration.
    """

    def __init__(self, case_insensitive: bool = True) -> None:
        self._case_insensitive = case_insensitive

    def process(self, document: PromptDocument) -> PromptDocument:
        seen: set[str] = set()
        deduped = PromptDocument()

        for _, block in document.iter_named_blocks():
            normalized = " ".join(block.content.split())
            if not normalized:
                continue

            if not block.enabled:
                deduped.upsert(replace(block))
                continue

            key = normalized.lower() if self._case_insensitive else normalized
            if key in seen:
                continue

            seen.add(key)
            deduped.upsert(replace(block))

        return deduped


class PromptPriorityTrimProcessor:
    """Trim lower-priority blocks until the document fits a character budget."""

    def __init__(
        self,
        max_total_chars: int,
        protected_blocks: Iterable[str] = (),
        min_priority: int = 0,
    ) -> None:
        self._max_total_chars = max_total_chars
        self._protected_blocks = {name.lower() for name in protected_blocks}
        self._min_priority = min_priority

    def process(self, document: PromptDocument) -> PromptDocument:
        if self._max_total_chars <= 0:
            return document.clone()

        trimmed = document.clone()

        def over_budget() -> bool:
            return trimmed.estimate_length() > self._max_total_chars

        candidates = sorted(
            trimmed.iter_named_blocks(),
            key=lambda item: item[1].priority,
            reverse=True,
        )

        for name, block in candidates:
            if not over_budget():
                break
            if block.priority < self._min_priority:
                continue
            if name.lower() in self._protected_blocks:
                continue
            trimmed.remove(name)

        return trimmed


__all__ = [
    "PromptBlock",
    "PromptDocument",
    "PromptDocumentProcessor",
    "PromptDeduplicationProcessor",
    "PromptPriorityTrimProcessor",
]
