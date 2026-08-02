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

"""Stable public API for LSP types used by external verticals.

External verticals should import LSP protocol types through this module
rather than from victor.protocols.lsp_types directly.

Usage:
    from victor.framework.lsp import (
        Position,
        Range,
        Diagnostic,
        CompletionItemKind,
    )
"""

from __future__ import annotations

__all__ = [
    "DiagnosticSeverity",  # noqa: F822
    "CompletionItemKind",  # noqa: F822
    "SymbolKind",  # noqa: F822
    "DiagnosticTag",  # noqa: F822
    "Position",  # noqa: F822
    "Range",  # noqa: F822
    "Location",  # noqa: F822
    "LocationLink",  # noqa: F822
    "DiagnosticRelatedInformation",  # noqa: F822
    "Diagnostic",  # noqa: F822
    "CompletionItem",  # noqa: F822
    "Hover",  # noqa: F822
    "DocumentSymbol",  # noqa: F822
    "SymbolInformation",  # noqa: F822
    "TextEdit",  # noqa: F822
    "TextDocumentIdentifier",  # noqa: F822
    "VersionedTextDocumentIdentifier",  # noqa: F822
    "TextDocumentEdit",  # noqa: F822
]


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies and unnecessary loading."""
    _ALL_NAMES = set(__all__)

    if name in _ALL_NAMES:
        import importlib

        module = importlib.import_module("victor.protocols.lsp_types")
        return getattr(module, name)

    raise AttributeError(f"module 'victor.framework.lsp' has no attribute {name!r}")
