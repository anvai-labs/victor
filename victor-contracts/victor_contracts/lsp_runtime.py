"""SDK host adapters for LSP runtime types and protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING

from victor_contracts._lazy import make_lazy_getattr

if TYPE_CHECKING:
    from victor.framework.lsp import (
        CompletionItem,
        CompletionItemKind,
        Diagnostic,
        DiagnosticRelatedInformation,
        DiagnosticSeverity,
        DiagnosticTag,
        DocumentSymbol,
        Hover,
        Location,
        LocationLink,
        Position,
        Range,
        SymbolInformation,
        SymbolKind,
        TextDocumentEdit,
        TextDocumentIdentifier,
        TextEdit,
        VersionedTextDocumentIdentifier,
    )
    from victor.framework.lsp_protocols import LSPPoolProtocol, LSPServiceProtocol

__all__ = [
    "DiagnosticSeverity",
    "CompletionItemKind",
    "SymbolKind",
    "DiagnosticTag",
    "Position",
    "Range",
    "Location",
    "LocationLink",
    "DiagnosticRelatedInformation",
    "Diagnostic",
    "CompletionItem",
    "Hover",
    "DocumentSymbol",
    "SymbolInformation",
    "TextEdit",
    "TextDocumentIdentifier",
    "VersionedTextDocumentIdentifier",
    "TextDocumentEdit",
    "LSPServiceProtocol",
    "LSPPoolProtocol",
]

_LAZY_IMPORTS = {
    "DiagnosticSeverity": "victor.framework.lsp",
    "CompletionItemKind": "victor.framework.lsp",
    "SymbolKind": "victor.framework.lsp",
    "DiagnosticTag": "victor.framework.lsp",
    "Position": "victor.framework.lsp",
    "Range": "victor.framework.lsp",
    "Location": "victor.framework.lsp",
    "LocationLink": "victor.framework.lsp",
    "DiagnosticRelatedInformation": "victor.framework.lsp",
    "Diagnostic": "victor.framework.lsp",
    "CompletionItem": "victor.framework.lsp",
    "Hover": "victor.framework.lsp",
    "DocumentSymbol": "victor.framework.lsp",
    "SymbolInformation": "victor.framework.lsp",
    "TextEdit": "victor.framework.lsp",
    "TextDocumentIdentifier": "victor.framework.lsp",
    "VersionedTextDocumentIdentifier": "victor.framework.lsp",
    "TextDocumentEdit": "victor.framework.lsp",
    "LSPServiceProtocol": "victor.framework.lsp_protocols",
    "LSPPoolProtocol": "victor.framework.lsp_protocols",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
