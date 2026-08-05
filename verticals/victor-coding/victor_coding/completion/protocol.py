# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Completion protocol types - re-exported from canonical location.

This module re-exports all completion protocol types from the canonical
source at victor.processing.completion.protocol for backward compatibility.

All new code should import directly from victor.processing.completion.protocol.
"""

# Re-export everything from canonical location
from victor_contracts.processing_runtime import (
    # LSP base types (originally from victor.protocols.lsp_types)
    Position,
    Range,
    TextEdit,
    CompletionItemKind,
    # Completion-specific types
    InsertTextFormat,
    CompletionTriggerKind,
    CompletionContext,
    CompletionParams,
    CompletionItemLabelDetails,
    CompletionItem,
    InlineCompletionItem,
    InlineCompletionParams,
    CompletionList,
    InlineCompletionList,
    CompletionCapabilities,
    CompletionMetrics,
)

__all__ = [
    "Position",
    "Range",
    "TextEdit",
    "CompletionItemKind",
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
