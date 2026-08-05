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

"""
Basic LSP client implementation.

This module provides a stub LSP client for graceful degradation when
the full victor-coding LSP integration is not available.

SOLID Principles:
- SRP: BasicLSPClient only provides stub implementations
- OCP: Extensible through protocol implementation
- LSP: Implements LanguageServerProtocol completely
- ISP: Focused on LSP operations
- DIP: No dependencies on concrete implementations

Usage:
    from victor.contrib.lsp import BasicLSPClient

    lsp = BasicLSPClient()
    # Returns empty results - full LSP requires victor-coding
    completions = await lsp.get_completions(file_path, line, char)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from victor.framework.vertical_protocols import (
    CompletionItem,
    CompletionItemKind,
    Diagnostic,
    HoverInfo,
    LanguageServerProtocol,
    Location,
)

logger = logging.getLogger(__name__)


class BasicLSPClient(LanguageServerProtocol):
    """
    Basic stub LSP client for graceful degradation.

    This client provides empty results for all LSP operations. It's
    intended as a fallback when the full victor-coding LSP integration
    is not available.

    For full LSP features, install victor-coding:
        pip install victor-coding

    Example:
        lsp = BasicLSPClient()
        completions = await lsp.get_completions(file_path, 10, 5)
        # Returns empty list - no LSP server running
    """

    def __init__(self) -> None:
        """Initialize the basic LSP client."""
        self._servers: dict[str, Any] = {}

    async def start_server(
        self,
        language: str,
        file_path: Path,
        **kwargs: Any,
    ) -> bool:
        """Start language server for a file (stub - returns False).

        Args:
            language: Programming language
            file_path: Path to the file
            **kwargs: Additional options

        Returns:
            False - full LSP requires victor-coding
        """
        logger.debug(f"LSP server not started for {language} - requires victor-coding package")
        return False

    async def stop_server(
        self,
        file_path: Path,
        **kwargs: Any,
    ) -> bool:
        """Stop language server for a file (stub - always succeeds).

        Args:
            file_path: Path to the file
            **kwargs: Additional options

        Returns:
            True - no server to stop
        """
        return True

    async def get_completions(
        self,
        file_path: Path,
        line: int,
        character: int,
        **kwargs: Any,
    ) -> List[CompletionItem]:
        """Get code completions at position (stub - returns empty list).

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character position (0-indexed)
            **kwargs: Additional options

        Returns:
            Empty list - full LSP requires victor-coding
        """
        logger.debug("LSP completions not available - requires victor-coding package")
        return []

    async def get_definition(
        self,
        file_path: Path,
        line: int,
        character: int,
        **kwargs: Any,
    ) -> Optional[Location]:
        """Go to definition at position (stub - returns None).

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character position (0-indexed)
            **kwargs: Additional options

        Returns:
            None - full LSP requires victor-coding
        """
        logger.debug("LSP definition not available - requires victor-coding package")
        return None

    async def get_hover(
        self,
        file_path: Path,
        line: int,
        character: int,
        **kwargs: Any,
    ) -> Optional[HoverInfo]:
        """Get hover information at position (stub - returns None).

        Args:
            file_path: Path to the file
            line: Line number (0-indexed)
            character: Character position (0-indexed)
            **kwargs: Additional options

        Returns:
            None - full LSP requires victor-coding
        """
        logger.debug("LSP hover not available - requires victor-coding package")
        return None

    async def get_diagnostics(
        self,
        file_path: Path,
        **kwargs: Any,
    ) -> List[Diagnostic]:
        """Get diagnostics for a file (stub - returns empty list).

        Args:
            file_path: Path to the file
            **kwargs: Additional options

        Returns:
            Empty list - full LSP requires victor-coding
        """
        logger.debug("LSP diagnostics not available - requires victor-coding package")
        return []

    def get_server_info(self) -> dict[str, Any]:
        """Get server metadata."""
        note = "Full LSP requires victor-coding package. Install with: pip install victor-coding"
        return {
            "name": "BasicLSPClient",
            "version": "1.0.0",
            "capabilities": {
                "stub_only": True,
            },
            "info": {
                "note": note,
            },
            "note": note,
        }


__all__ = ["BasicLSPClient"]
