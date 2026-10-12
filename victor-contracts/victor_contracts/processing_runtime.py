"""SDK host adapters for processing and editing runtime helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.framework.lsp import CompletionItemKind, Position, Range, TextEdit
    from victor.processing.editing import (
        EditOperation,
        EditTransaction,
        FileEditor,
        OperationType,
    )
    from victor.processing.completion.protocol import (
        CompletionCapabilities,
        CompletionContext,
        CompletionItem,
        CompletionItemLabelDetails,
        CompletionList,
        CompletionMetrics,
        CompletionParams,
        CompletionTriggerKind,
        InlineCompletionItem,
        InlineCompletionList,
        InlineCompletionParams,
        InsertTextFormat,
    )
    from victor.native.protocols import (
        ChunkInfo,
        TextChunkerProtocol,
    )
    from victor.processing.native import get_default_text_chunker

__all__ = [
    "OperationType",
    "EditOperation",
    "EditTransaction",
    "FileEditor",
    "get_default_text_chunker",
    "TextChunkerProtocol",
    "ChunkInfo",
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

_LAZY_IMPORTS = {
    "OperationType": "victor.framework.processing",
    "EditOperation": "victor.framework.processing",
    "EditTransaction": "victor.framework.processing",
    "FileEditor": "victor.framework.processing",
    "get_default_text_chunker": "victor.framework.processing",
    "TextChunkerProtocol": "victor.native.protocols",
    "ChunkInfo": "victor.native.protocols",
    "Position": "victor.framework.lsp",
    "Range": "victor.framework.lsp",
    "TextEdit": "victor.framework.lsp",
    "CompletionItemKind": "victor.framework.lsp",
    "InsertTextFormat": "victor.framework.processing",
    "CompletionTriggerKind": "victor.framework.processing",
    "CompletionContext": "victor.framework.processing",
    "CompletionParams": "victor.framework.processing",
    "CompletionItemLabelDetails": "victor.framework.processing",
    "CompletionItem": "victor.framework.processing",
    "InlineCompletionItem": "victor.framework.processing",
    "InlineCompletionParams": "victor.framework.processing",
    "CompletionList": "victor.framework.processing",
    "InlineCompletionList": "victor.framework.processing",
    "CompletionCapabilities": "victor.framework.processing",
    "CompletionMetrics": "victor.framework.processing",
}


def __getattr__(name: str) -> Any:
    """Resolve processing helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(
            f"module 'victor_contracts.processing_runtime' has no attribute {name!r}"
        )

    module = importlib.import_module(module_name)
    return getattr(module, name)
