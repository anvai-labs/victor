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

"""Language Server Protocol (LSP) integration for Victor.

This module provides LSP client functionality for code intelligence features
like completion, go-to-definition, find references, and diagnostics.

LSP types are provided by victor.protocols.lsp_types for cross-vertical use.
"""

from victor_coding.lsp.client import LSPClient
from victor_coding.lsp.manager import LSPConnectionPool, get_lsp_manager
from victor_coding.lsp.config import LSPServerConfig, LANGUAGE_SERVERS

# Re-export core LSP types for convenience
from victor_contracts.lsp_runtime import (
    # Enumerations
    DiagnosticSeverity,
    CompletionItemKind,
    SymbolKind,
    DiagnosticTag,
    # Position and Range
    Position,
    Range,
    Location,
    LocationLink,
    # Diagnostics
    DiagnosticRelatedInformation,
    Diagnostic,
    # Completions
    CompletionItem,
    # Hover
    Hover,
    # Symbols
    DocumentSymbol,
    SymbolInformation,
    # Text Edits
    TextEdit,
    TextDocumentIdentifier,
    VersionedTextDocumentIdentifier,
    TextDocumentEdit,
)

__all__ = [
    # Client
    "LSPClient",
    "LSPConnectionPool",
    "get_lsp_manager",
    "LSPServerConfig",
    "LANGUAGE_SERVERS",
    # Core LSP types
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
]
