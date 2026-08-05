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

"""LSP protocol abstractions for cross-vertical language intelligence.

Provides Protocol interfaces that any vertical can depend on without
coupling to the coding vertical's LSP implementation (SOLID DIP).

Usage:
    # In any vertical that needs language intelligence
    from victor.framework import LSPServiceProtocol

    class MyAnalyzer:
        def __init__(self, lsp_service: LSPServiceProtocol):
            self._lsp = lsp_service

        async def analyze_file(self, path: str) -> dict:
            hover = await self._lsp.get_hover(path, 10, 5)
            # ...

Benefits:
    - RAG vertical: Better code chunking/understanding for indexed code
    - DataAnalysis vertical: Jupyter notebooks with Python LSP
    - DevOps vertical: YAML/Terraform/HCL language support
    - Research vertical: Code repository analysis
    - Future verticals: Any vertical needing language intelligence
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class LSPPosition:
    """Position in a text document."""

    line: int  # 0-indexed
    character: int  # 0-indexed


@dataclass
class LSPRange:
    """Range in a text document."""

    start: LSPPosition
    end: LSPPosition


@dataclass
class LSPLocation:
    """Location in a file."""

    uri: str
    range: LSPRange


@dataclass
class LSPHoverInfo:
    """Hover information from LSP."""

    contents: str
    range: Optional[LSPRange] = None


@dataclass
class LSPCompletionItem:
    """Completion suggestion from LSP."""

    label: str
    kind: Optional[int] = None
    detail: Optional[str] = None
    documentation: Optional[str] = None
    insert_text: Optional[str] = None


@dataclass
class LSPDiagnostic:
    """Diagnostic (error/warning) from LSP."""

    range: LSPRange
    message: str
    severity: int  # 1=error, 2=warning, 3=info, 4=hint
    source: Optional[str] = None
    code: Optional[str] = None


@dataclass
class LSPSymbol:
    """Symbol information from LSP."""

    name: str
    kind: int  # SymbolKind enum value
    range: LSPRange
    selection_range: LSPRange
    children: Optional[List["LSPSymbol"]] = None
    detail: Optional[str] = None
    deprecated: bool = False


@runtime_checkable
class LSPServiceProtocol(Protocol):
    """Protocol for LSP service operations.

    Any vertical can depend on this protocol for language intelligence
    without coupling to the coding vertical's implementation.

    This follows the Dependency Inversion Principle (DIP):
    - High-level modules (verticals) depend on abstractions (this protocol)
    - Low-level modules (coding LSP impl) depend on abstractions (this protocol)
    """

    async def open_document(self, file_path: str, content: Optional[str] = None) -> bool:
        """Open a document for LSP operations.

        Args:
            file_path: Path to the file
            content: Optional file content (reads from disk if not provided)

        Returns:
            True if opened successfully
        """
        ...

    def close_document(self, file_path: str) -> None:
        """Close a document.

        Args:
            file_path: Path to the file
        """
        ...

    async def update_document(self, file_path: str, content: str) -> bool:
        """Update a document's contents.

        Args:
            file_path: Path to the file
            content: New content

        Returns:
            True if updated successfully
        """
        ...

    async def get_hover(self, file_path: str, line: int, character: int) -> Optional[LSPHoverInfo]:
        """Get hover information at position.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character offset (0-indexed)

        Returns:
            Hover information or None
        """
        ...

    async def get_completions(
        self, file_path: str, line: int, character: int
    ) -> List[LSPCompletionItem]:
        """Get completions at position.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character offset (0-indexed)

        Returns:
            List of completion items
        """
        ...

    async def get_definition(self, file_path: str, line: int, character: int) -> List[LSPLocation]:
        """Get definition locations.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character offset (0-indexed)

        Returns:
            List of definition locations
        """
        ...

    async def get_references(self, file_path: str, line: int, character: int) -> List[LSPLocation]:
        """Get reference locations.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character offset (0-indexed)

        Returns:
            List of reference locations
        """
        ...

    def get_diagnostics(self, file_path: str) -> List[LSPDiagnostic]:
        """Get diagnostics for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of diagnostics
        """
        ...

    async def get_document_symbols(self, file_path: str) -> List[LSPSymbol]:
        """Get the document symbol tree (outline) for a file.

        File-level (no cursor needed). Returns signatures/classes/functions with
        their kinds and ranges — the live API surface of the file. Used by the
        FEP-0019 Phase 3 LSP context provider to guide generation.

        Args:
            file_path: Path to the file

        Returns:
            List of top-level symbols (each may carry nested ``children``), or
            an empty list when the server has no symbols / the capability is
            unavailable.
        """
        ...


@runtime_checkable
class LSPPoolProtocol(Protocol):
    """Protocol for LSP connection pool management.

    Provides lifecycle management for language servers.
    Used by components that need to manage multiple language servers.
    """

    async def start_server(self, language: str) -> bool:
        """Start a language server.

        Args:
            language: Language identifier (e.g., "python", "typescript")

        Returns:
            True if started successfully
        """
        ...

    async def stop_server(self, language: str) -> None:
        """Stop a language server.

        Args:
            language: Language identifier
        """
        ...

    async def stop_all(self) -> None:
        """Stop all running language servers."""
        ...

    async def restart_server(self, language: str) -> bool:
        """Restart a language server.

        Args:
            language: Language identifier

        Returns:
            True if restarted successfully
        """
        ...

    def get_available_servers(self) -> List[Dict[str, Any]]:
        """Get list of available language servers.

        Returns:
            List of server info dicts with keys:
            - language: Language identifier
            - name: Server name
            - command: Server command
            - installed: Whether server is installed
            - running: Whether server is running
        """
        ...

    def get_status(self) -> Dict[str, Any]:
        """Get status of all servers.

        Returns:
            Dict of language -> status info
        """
        ...
