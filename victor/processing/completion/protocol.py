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

"""Inline completion protocol types following LSP specification.

This module defines the data types for inline code completion,
based on the Language Server Protocol (LSP) specification with
extensions for AI-powered completions.

This is the CANONICAL location for completion protocol types.

Base LSP types (Position, Range, TextEdit, CompletionItemKind) are
imported from victor.protocols.lsp_types - the canonical source for
all LSP standard types.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional

# Import canonical LSP types - DO NOT redefine these
from victor.protocols.lsp_types import (
    Position,
    Range,
    TextEdit,
    CompletionItemKind,
)


class InsertTextFormat(IntEnum):
    """How the insert text should be interpreted."""

    PLAIN_TEXT = 1
    SNIPPET = 2


class CompletionTriggerKind(IntEnum):
    """How the completion was triggered."""

    INVOKED = 1  # Explicitly invoked (e.g., Ctrl+Space)
    TRIGGER_CHARACTER = 2  # Triggered by a character (e.g., '.')
    TRIGGER_FOR_INCOMPLETE = 3  # Re-triggered for incomplete results


@dataclass
class CompletionContext:
    """Context information for a completion request."""

    trigger_kind: CompletionTriggerKind
    trigger_character: Optional[str] = None


@dataclass
class CompletionParams:
    """Parameters for a completion request."""

    file_path: Path
    position: Position
    context: Optional[CompletionContext] = None

    # Extended parameters for AI completions
    prefix: str = ""  # Text before cursor on current line
    suffix: str = ""  # Text after cursor on current line
    file_content: str = ""  # Full file content
    language: str = ""  # Detected language
    max_results: int = 10


@dataclass
class CompletionItemLabelDetails:
    """Additional details for a completion item label."""

    detail: Optional[str] = None  # Signature or type
    description: Optional[str] = None  # Module or package


@dataclass
class CompletionItem:
    """A completion item represents a completion suggestion.

    Follows LSP CompletionItem specification with extensions.
    """

    label: str  # The label shown in the completion list
    kind: CompletionItemKind = CompletionItemKind.TEXT
    detail: Optional[str] = None  # Type information or short description
    documentation: Optional[str] = None  # Detailed documentation
    deprecated: bool = False
    preselect: bool = False  # Should this item be selected by default?
    sort_text: Optional[str] = None  # Sort key (defaults to label)
    filter_text: Optional[str] = None  # Filter key (defaults to label)
    insert_text: Optional[str] = None  # Text to insert (defaults to label)
    insert_text_format: InsertTextFormat = InsertTextFormat.PLAIN_TEXT
    text_edit: Optional[TextEdit] = None  # Edit to apply
    additional_text_edits: list[TextEdit] = field(default_factory=list)
    commit_characters: list[str] = field(default_factory=list)
    label_details: Optional[CompletionItemLabelDetails] = None

    # Extended fields for AI completions
    provider: str = ""  # Which provider generated this
    confidence: float = 1.0  # Confidence score (0-1)
    tokens_used: int = 0  # Tokens consumed by AI provider
    latency_ms: float = 0.0  # Time to generate


@dataclass
class InlineCompletionItem:
    """An inline completion item for ghost text suggestions.

    This is an extension beyond standard LSP for Copilot-style completions.
    """

    insert_text: str  # The text to insert
    range: Optional[Range] = None  # Range to replace (None = insert at cursor)
    filter_text: Optional[str] = None  # For filtering as user types
    command: Optional[dict[str, Any]] = None  # Command to execute on accept

    # Extended fields
    provider: str = ""
    confidence: float = 1.0
    is_complete: bool = True  # Is this a complete suggestion?
    tokens_used: int = 0
    latency_ms: float = 0.0


@dataclass
class InlineCompletionParams:
    """Parameters for an inline completion request."""

    file_path: Path
    position: Position
    context: Optional[CompletionContext] = None

    # Content context
    prefix: str = ""  # All text before cursor
    suffix: str = ""  # All text after cursor
    file_content: str = ""
    language: str = ""

    # AI-specific parameters
    max_tokens: int = 256  # Max tokens in completion
    temperature: float = 0.0  # Deterministic by default
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class CompletionList:
    """A collection of completion items."""

    is_incomplete: bool  # If true, further typing should trigger re-query
    items: list[CompletionItem] = field(default_factory=list)


@dataclass
class InlineCompletionList:
    """A collection of inline completion items."""

    items: list[InlineCompletionItem] = field(default_factory=list)


@dataclass
class CompletionCapabilities:
    """Capabilities of a completion provider."""

    # Standard LSP capabilities
    supports_completion: bool = True
    supports_inline_completion: bool = False
    supports_resolve: bool = False  # Can resolve additional details
    supports_snippets: bool = False
    trigger_characters: list[str] = field(default_factory=list)

    # Extended capabilities
    supports_multi_line: bool = False  # Multi-line inline completions
    supports_streaming: bool = False  # Can stream completions
    max_context_lines: int = 100  # Max context lines to send
    supported_languages: list[str] = field(default_factory=list)  # Empty = all


@dataclass
class CompletionMetrics:
    """Metrics for completion operations."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    total_tokens_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


__all__ = [
    # Re-exported LSP types
    "Position",
    "Range",
    "TextEdit",
    "CompletionItemKind",
    # Completion-specific types
    "InsertTextFormat",
    "CompletionTriggerKind",
    "CompletionContext",
    "CompletionParams",
    "CompletionItemLabelDetails",
    "CompletionItem",
    "InlineCompletionItem",
    "InlineCompletionParams",
    "CompletionList",
    "InlineCompletionList",
    "CompletionCapabilities",
    "CompletionMetrics",
]
