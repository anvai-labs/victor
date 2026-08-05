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

"""LSP Manager for managing multiple language servers."""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from victor_coding.lsp.client import (
    LSPClient,
    CompletionItem,
    Hover,
    Position,
    DocumentSymbol,
)
from victor_coding.lsp.config import (
    LANGUAGE_SERVERS,
    get_server_for_file,
)

logger = logging.getLogger(__name__)


@dataclass
class LSPStatus:
    """Status of an LSP server."""

    language: str
    server_name: str
    running: bool
    initialized: bool
    open_documents: int
    capabilities: List[str]


class LSPConnectionPool:
    """Manages multiple language server clients.

    Provides a unified interface for LSP operations across different
    language servers.

    Usage:
        async with LSPConnectionPool() as pool:
            await pool.start_server("python")
            completions = await pool.get_completions("file.py", 10, 5)
        # All servers automatically stopped on exit
    """

    def __init__(self, workspace_root: Optional[str] = None):
        """Initialize the LSP manager.

        Args:
            workspace_root: Root directory of the workspace
        """
        self._workspace_root = workspace_root or str(Path.cwd())
        self._root_uri = self._path_to_uri(self._workspace_root)
        self._clients: Dict[str, LSPClient] = {}  # language_id -> client
        self._auto_start = True

    async def __aenter__(self) -> "LSPConnectionPool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - ensures all servers are stopped."""
        await self.stop_all()

    @staticmethod
    def _path_to_uri(path: str) -> str:
        """Convert a file path to a file:// URI."""
        abs_path = str(Path(path).resolve())
        return f"file://{quote(abs_path)}"

    @staticmethod
    def _uri_to_path(uri: str) -> str:
        """Convert a file:// URI to a path."""
        if uri.startswith("file://"):
            from urllib.parse import unquote

            return unquote(uri[7:])
        return uri

    def set_workspace_root(self, path: str) -> None:
        """Set the workspace root directory.

        Args:
            path: New workspace root
        """
        self._workspace_root = str(Path(path).resolve())
        self._root_uri = self._path_to_uri(self._workspace_root)

    async def start_server(self, language: str) -> bool:
        """Start a language server.

        Args:
            language: Language identifier (e.g., "python", "typescript")

        Returns:
            True if started successfully
        """
        if language in self._clients and self._clients[language].is_running:
            return True

        config = LANGUAGE_SERVERS.get(language)
        if not config:
            logger.warning(f"No server configuration for language: {language}")
            return False

        # Check if server command exists
        if not shutil.which(config.command[0]):
            logger.error(f"Server {config.name} not found: {config.command[0]}")
            logger.info(f"Install with: {config.install_command}")
            return False

        client = LSPClient(config, self._root_uri)
        success = await client.start()

        if success:
            self._clients[language] = client
            logger.info(f"Started LSP server for {language}: {config.name}")

        return success

    async def stop_server(self, language: str) -> None:
        """Stop a language server.

        Args:
            language: Language identifier
        """
        if language in self._clients:
            await self._clients[language].stop()
            del self._clients[language]
            logger.info(f"Stopped LSP server for {language}")

    async def stop_all(self) -> None:
        """Stop all running language servers."""
        for language in list(self._clients.keys()):
            await self.stop_server(language)

    async def restart_server(self, language: str) -> bool:
        """Restart a language server.

        Args:
            language: Language identifier

        Returns:
            True if restarted successfully
        """
        await self.stop_server(language)
        return await self.start_server(language)

    def _get_client_for_file(self, file_path: str) -> Optional[LSPClient]:
        """Get the appropriate client for a file.

        Args:
            file_path: Path to the file

        Returns:
            LSPClient if available
        """
        config = get_server_for_file(file_path)
        if not config:
            return None

        # Find client by language_id
        for _lang, client in self._clients.items():
            if client.config.language_id == config.language_id:
                return client

        return None

    async def _ensure_client_for_file(self, file_path: str) -> Optional[LSPClient]:
        """Ensure a client is running for a file.

        Args:
            file_path: Path to the file

        Returns:
            LSPClient if available
        """
        client = self._get_client_for_file(file_path)
        if client:
            return client

        if not self._auto_start:
            return None

        # Try to start the appropriate server
        config = get_server_for_file(file_path)
        if config:
            for lang, conf in LANGUAGE_SERVERS.items():
                if conf.language_id == config.language_id:
                    if await self.start_server(lang):
                        return self._clients.get(lang)
                    break

        return None

    async def open_document(self, file_path: str, text: Optional[str] = None) -> bool:
        """Open a document in the appropriate server.

        Args:
            file_path: Path to the file
            text: File contents (reads from disk if not provided)

        Returns:
            True if opened successfully
        """
        client = await self._ensure_client_for_file(file_path)
        if not client:
            return False

        uri = self._path_to_uri(file_path)

        if text is None:
            try:
                text = Path(file_path).read_text()
            except Exception as e:
                logger.error(f"Failed to read file: {e}")
                return False

        client.open_document(uri, text)
        return True

    def close_document(self, file_path: str) -> None:
        """Close a document.

        Args:
            file_path: Path to the file
        """
        client = self._get_client_for_file(file_path)
        if client:
            uri = self._path_to_uri(file_path)
            client.close_document(uri)

    async def update_document(self, file_path: str, text: str) -> bool:
        """Update a document's contents.

        Args:
            file_path: Path to the file
            text: New contents

        Returns:
            True if updated successfully
        """
        client = await self._ensure_client_for_file(file_path)
        if not client:
            return False

        uri = self._path_to_uri(file_path)
        client.update_document(uri, text)
        return True

    async def get_completions(
        self, file_path: str, line: int, character: int
    ) -> List[CompletionItem]:
        """Get completions at a position.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character offset (0-indexed)

        Returns:
            List of completion items
        """
        client = await self._ensure_client_for_file(file_path)
        if not client:
            return []

        uri = self._path_to_uri(file_path)
        position = Position(line=line, character=character)
        return await client.get_completions(uri, position)

    async def get_hover(self, file_path: str, line: int, character: int) -> Optional[Hover]:
        """Get hover information at a position.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character offset (0-indexed)

        Returns:
            Hover information or None
        """
        client = await self._ensure_client_for_file(file_path)
        if not client:
            return None

        uri = self._path_to_uri(file_path)
        position = Position(line=line, character=character)
        return await client.get_hover(uri, position)

    async def get_definition(
        self, file_path: str, line: int, character: int
    ) -> List[Dict[str, Any]]:
        """Get definition locations.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character offset (0-indexed)

        Returns:
            List of definition locations as dicts
        """
        client = await self._ensure_client_for_file(file_path)
        if not client:
            return []

        uri = self._path_to_uri(file_path)
        position = Position(line=line, character=character)
        locations = await client.get_definition(uri, position)

        return [
            {
                "file": self._uri_to_path(loc.uri),
                "line": loc.range.start.line + 1,  # 1-indexed for display
                "character": loc.range.start.character,
            }
            for loc in locations
        ]

    async def get_references(
        self, file_path: str, line: int, character: int
    ) -> List[Dict[str, Any]]:
        """Get reference locations.

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character offset (0-indexed)

        Returns:
            List of reference locations as dicts
        """
        client = await self._ensure_client_for_file(file_path)
        if not client:
            return []

        uri = self._path_to_uri(file_path)
        position = Position(line=line, character=character)
        locations = await client.get_references(uri, position)

        return [
            {
                "file": self._uri_to_path(loc.uri),
                "line": loc.range.start.line + 1,
                "character": loc.range.start.character,
            }
            for loc in locations
        ]

    def get_diagnostics(self, file_path: str) -> List[Dict[str, Any]]:
        """Get diagnostics for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of diagnostics as dicts
        """
        client = self._get_client_for_file(file_path)
        if not client:
            return []

        uri = self._path_to_uri(file_path)
        diagnostics = client.get_diagnostics(uri)

        severity_names = {1: "error", 2: "warning", 3: "info", 4: "hint"}

        return [
            {
                "line": d.range.start.line + 1,
                "character": d.range.start.character,
                "message": d.message,
                "severity": severity_names.get(d.severity, "unknown"),
                "source": d.source,
                "code": d.code,
            }
            for d in diagnostics
        ]

    async def get_document_symbols(self, file_path: str) -> List[DocumentSymbol]:
        """Get the document symbol outline for a file (FEP-0019 Phase 3).

        Returns the raw ``DocumentSymbol`` tree (name/kind/range/children) so the
        framework adapter can render the live API surface for generation guidance.

        Args:
            file_path: Path to the file

        Returns:
            List of top-level symbols (each may carry children), or [] if no
            server is available for the file.
        """
        client = await self._ensure_client_for_file(file_path)
        if not client:
            return []
        uri = self._path_to_uri(file_path)
        return await client.get_document_symbols(uri)

    def get_status(self) -> Dict[str, LSPStatus]:
        """Get status of all servers.

        Returns:
            Dict of language -> status
        """
        status = {}
        for language, client in self._clients.items():
            caps = []
            if client._capabilities.get("completionProvider"):
                caps.append("completion")
            if client._capabilities.get("hoverProvider"):
                caps.append("hover")
            if client._capabilities.get("definitionProvider"):
                caps.append("definition")
            if client._capabilities.get("referencesProvider"):
                caps.append("references")

            status[language] = LSPStatus(
                language=language,
                server_name=client.config.name,
                running=client.is_running,
                initialized=client._initialized,
                open_documents=len(client._open_documents),
                capabilities=caps,
            )

        return status

    def get_available_servers(self) -> List[Dict[str, Any]]:
        """Get list of available language servers.

        Returns:
            List of server info dicts
        """
        servers = []
        for lang, config in LANGUAGE_SERVERS.items():
            installed = shutil.which(config.command[0]) is not None
            running = lang in self._clients and self._clients[lang].is_running

            servers.append(
                {
                    "language": lang,
                    "name": config.name,
                    "command": config.command[0],
                    "installed": installed,
                    "running": running,
                    "install_command": config.install_command,
                }
            )

        return servers


# Global instance
_lsp_manager: Optional[LSPConnectionPool] = None


def get_lsp_manager() -> LSPConnectionPool:
    """Get or create the global LSP connection pool.

    Note: Function name kept for backward compatibility.
    """
    global _lsp_manager
    if _lsp_manager is None:
        _lsp_manager = LSPConnectionPool()
    return _lsp_manager


def set_lsp_manager(manager: LSPConnectionPool) -> None:
    """Set the global LSP connection pool.

    Note: Function name kept for backward compatibility.
    """
    global _lsp_manager
    _lsp_manager = manager


def reset_lsp_manager() -> None:
    """Reset the global LSP connection pool (for testing).

    Note: Function name kept for backward compatibility.
    """
    global _lsp_manager
    _lsp_manager = None
