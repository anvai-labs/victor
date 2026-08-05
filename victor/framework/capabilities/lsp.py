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

"""LSP capability implementation for Victor framework.

This module provides the framework-level LSP capability that any vertical
can use for language intelligence. It implements the LSPServiceProtocol
and LSPPoolProtocol defined in victor.framework.lsp_protocols.
"""

from typing import Any, Dict, List, Optional

from victor.framework.capabilities.base import (
    BaseCapabilityProvider,
    CapabilityMetadata,
)
from victor.framework.lsp_protocols import (
    LSPCompletionItem,
    LSPDiagnostic,
    LSPHoverInfo,
    LSPLocation,
    LSPPoolProtocol,
    LSPServiceProtocol,
    LSPSymbol,
)


class LSPCapability(LSPServiceProtocol, LSPPoolProtocol):
    """Framework-level LSP capability.

    This capability provides language intelligence services (hover,
    definitions, completions, etc.) and manages the lifecycle of
    language servers.

    It acts as a bridge between the framework and the actual LSP
    implementation (usually provided by the coding vertical).
    """

    def __init__(self, implementation: Optional[Any] = None):
        """Initialize LSP capability.

        Args:
            implementation: The actual LSP implementation to wrap.
                If None, operations will return empty/default values.
        """
        self._impl = implementation

    # --- LSPServiceProtocol implementation ---

    async def open_document(self, file_path: str, content: Optional[str] = None) -> bool:
        if self._impl and hasattr(self._impl, "open_document"):
            return await self._impl.open_document(file_path, content)
        return False

    def close_document(self, file_path: str) -> None:
        if self._impl and hasattr(self._impl, "close_document"):
            self._impl.close_document(file_path)

    async def update_document(self, file_path: str, content: str) -> bool:
        if self._impl and hasattr(self._impl, "update_document"):
            return await self._impl.update_document(file_path, content)
        return False

    async def get_hover(self, file_path: str, line: int, character: int) -> Optional[LSPHoverInfo]:
        if self._impl and hasattr(self._impl, "get_hover"):
            return await self._impl.get_hover(file_path, line, character)
        return None

    async def get_completions(
        self, file_path: str, line: int, character: int
    ) -> List[LSPCompletionItem]:
        if self._impl and hasattr(self._impl, "get_completions"):
            return await self._impl.get_completions(file_path, line, character)
        return []

    async def get_definition(self, file_path: str, line: int, character: int) -> List[LSPLocation]:
        if self._impl and hasattr(self._impl, "get_definition"):
            return await self._impl.get_definition(file_path, line, character)
        return []

    async def get_references(self, file_path: str, line: int, character: int) -> List[LSPLocation]:
        if self._impl and hasattr(self._impl, "get_references"):
            return await self._impl.get_references(file_path, line, character)
        return []

    def get_diagnostics(self, file_path: str) -> List[LSPDiagnostic]:
        if self._impl and hasattr(self._impl, "get_diagnostics"):
            return self._impl.get_diagnostics(file_path)
        return []

    async def get_document_symbols(self, file_path: str) -> List[LSPSymbol]:
        if self._impl and hasattr(self._impl, "get_document_symbols"):
            return await self._impl.get_document_symbols(file_path)
        return []

    # --- LSPPoolProtocol implementation ---

    async def start_server(self, language: str) -> bool:
        if self._impl and hasattr(self._impl, "start_server"):
            return await self._impl.start_server(language)
        return False

    async def stop_server(self, language: str) -> None:
        if self._impl and hasattr(self._impl, "stop_server"):
            await self._impl.stop_server(language)

    async def stop_all(self) -> None:
        if self._impl and hasattr(self._impl, "stop_all"):
            await self._impl.stop_all()

    async def restart_server(self, language: str) -> bool:
        if self._impl and hasattr(self._impl, "restart_server"):
            return await self._impl.restart_server(language)
        return False

    def get_available_servers(self) -> List[Dict[str, Any]]:
        if self._impl and hasattr(self._impl, "get_available_servers"):
            return self._impl.get_available_servers()
        return []

    def get_status(self) -> Dict[str, Any]:
        if self._impl and hasattr(self._impl, "get_status"):
            return self._impl.get_status()
        return {}


class LSPCapabilityProvider(BaseCapabilityProvider[LSPCapability]):
    """Provider for framework-level LSP capabilities."""

    def __init__(self, lsp_capability: LSPCapability):
        self._lsp = lsp_capability

    def get_capabilities(self) -> Dict[str, LSPCapability]:
        return {"lsp": self._lsp}

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        return {
            "lsp": CapabilityMetadata(
                name="lsp",
                description="Language Server Protocol capability for code intelligence",
                version="1.0",
                tags=["lsp", "intelligence", "code"],
            )
        }
