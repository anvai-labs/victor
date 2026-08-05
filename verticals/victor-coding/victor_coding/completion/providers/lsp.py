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

"""LSP-based completion provider.

Integrates with Language Server Protocol servers to provide
language-aware code completions.
"""

import logging
from typing import Any, Optional

from victor_coding.completion.protocol import (
    CompletionCapabilities,
    CompletionItem,
    CompletionItemKind,
    CompletionItemLabelDetails,
    CompletionList,
    CompletionParams,
    InsertTextFormat,
    InlineCompletionList,
    InlineCompletionParams,
    Position,
    Range,
    TextEdit,
)
from victor_coding.completion.provider import BaseCompletionProvider

logger = logging.getLogger(__name__)


# LSP CompletionItemKind to our CompletionItemKind mapping
LSP_KIND_MAP = {
    1: CompletionItemKind.TEXT,
    2: CompletionItemKind.METHOD,
    3: CompletionItemKind.FUNCTION,
    4: CompletionItemKind.CONSTRUCTOR,
    5: CompletionItemKind.FIELD,
    6: CompletionItemKind.VARIABLE,
    7: CompletionItemKind.CLASS,
    8: CompletionItemKind.INTERFACE,
    9: CompletionItemKind.MODULE,
    10: CompletionItemKind.PROPERTY,
    11: CompletionItemKind.UNIT,
    12: CompletionItemKind.VALUE,
    13: CompletionItemKind.ENUM,
    14: CompletionItemKind.KEYWORD,
    15: CompletionItemKind.SNIPPET,
    16: CompletionItemKind.COLOR,
    17: CompletionItemKind.FILE,
    18: CompletionItemKind.REFERENCE,
    19: CompletionItemKind.FOLDER,
    20: CompletionItemKind.ENUM_MEMBER,
    21: CompletionItemKind.CONSTANT,
    22: CompletionItemKind.STRUCT,
    23: CompletionItemKind.EVENT,
    24: CompletionItemKind.OPERATOR,
    25: CompletionItemKind.TYPE_PARAMETER,
}


class LSPCompletionProvider(BaseCompletionProvider):
    """Completion provider using Language Server Protocol.

    Delegates to LSP servers for language-aware completions.
    Supports any language with an LSP server configured.
    """

    def __init__(
        self,
        priority: int = 80,
        lsp_manager: Optional[Any] = None,
    ):
        """Initialize the LSP completion provider.

        Args:
            priority: Provider priority (default 80 - high)
            lsp_manager: Optional LSPConnectionPool instance
        """
        super().__init__(priority=priority)
        self._lsp_manager = lsp_manager
        self._trigger_characters = [".", ":", "<", '"', "'", "/", "@", "#"]

    @property
    def name(self) -> str:
        return "lsp"

    def get_capabilities(self) -> CompletionCapabilities:
        """Return LSP completion capabilities."""
        return CompletionCapabilities(
            supports_completion=True,
            supports_inline_completion=False,
            supports_resolve=True,
            supports_snippets=True,
            trigger_characters=self._trigger_characters,
            supported_languages=[],  # All languages with LSP support
        )

    def _get_lsp_manager(self) -> Optional[Any]:
        """Get or lazy-load the LSP manager."""
        if self._lsp_manager is not None:
            return self._lsp_manager

        try:
            from victor_coding.lsp.manager import LSPConnectionPool

            self._lsp_manager = LSPConnectionPool()
            return self._lsp_manager
        except ImportError:
            logger.debug("LSP manager not available")
            return None

    async def provide_completions(self, params: CompletionParams) -> CompletionList:
        """Get completions from LSP server.

        Args:
            params: Completion request parameters

        Returns:
            CompletionList with items from LSP
        """
        manager = self._get_lsp_manager()
        if manager is None:
            return CompletionList(is_incomplete=False, items=[])

        try:
            # Get LSP client for this file
            client = await manager.get_client_for_file(params.file_path)
            if client is None:
                return CompletionList(is_incomplete=False, items=[])

            # Request completions from LSP
            lsp_result = await client.request_completion(
                uri=params.file_path.as_uri(),
                line=params.position.line,
                character=params.position.character,
                trigger_kind=params.context.trigger_kind if params.context else 1,
                trigger_character=(params.context.trigger_character if params.context else None),
            )

            if lsp_result is None:
                return CompletionList(is_incomplete=False, items=[])

            # Convert LSP response to our format
            return self._convert_lsp_response(lsp_result)

        except Exception as e:
            logger.warning(f"LSP completion failed: {e}")
            return CompletionList(is_incomplete=False, items=[])

    async def provide_inline_completions(
        self, params: InlineCompletionParams
    ) -> InlineCompletionList:
        """LSP doesn't support inline completions."""
        return InlineCompletionList(items=[])

    async def resolve_completion(self, item: CompletionItem) -> CompletionItem:
        """Resolve additional details for a completion item.

        Args:
            item: The item to resolve

        Returns:
            Item with additional documentation
        """
        manager = self._get_lsp_manager()
        if manager is None:
            return item

        # Item needs to have LSP data attached for resolution
        if not hasattr(item, "_lsp_data"):
            return item

        try:
            # Get any available client for resolution
            client = await manager.get_any_client()
            if client is None:
                return item

            resolved = await client.resolve_completion(item._lsp_data)
            if resolved:
                if "documentation" in resolved:
                    doc = resolved["documentation"]
                    if isinstance(doc, dict):
                        item.documentation = doc.get("value", "")
                    else:
                        item.documentation = str(doc)

            return item

        except Exception as e:
            logger.warning(f"LSP completion resolution failed: {e}")
            return item

    def _convert_lsp_response(self, lsp_result: dict) -> CompletionList:
        """Convert LSP completion response to our format.

        Args:
            lsp_result: Raw LSP response

        Returns:
            Converted CompletionList
        """
        # Handle both list and CompletionList responses
        if isinstance(lsp_result, list):
            items_data = lsp_result
            is_incomplete = False
        else:
            items_data = lsp_result.get("items", [])
            is_incomplete = lsp_result.get("isIncomplete", False)

        items = []
        for lsp_item in items_data:
            item = self._convert_lsp_item(lsp_item)
            if item:
                items.append(item)

        return CompletionList(is_incomplete=is_incomplete, items=items)

    def _convert_lsp_item(self, lsp_item: dict) -> Optional[CompletionItem]:
        """Convert a single LSP completion item.

        Args:
            lsp_item: LSP completion item dict

        Returns:
            Converted CompletionItem or None
        """
        label = lsp_item.get("label")
        if not label:
            return None

        # Map LSP kind to our kind
        lsp_kind = lsp_item.get("kind", 1)
        kind = LSP_KIND_MAP.get(lsp_kind, CompletionItemKind.TEXT)

        # Handle documentation
        doc = lsp_item.get("documentation")
        documentation = None
        if doc:
            if isinstance(doc, dict):
                documentation = doc.get("value", "")
            else:
                documentation = str(doc)

        # Handle insert text format
        insert_format = lsp_item.get("insertTextFormat", 1)
        insert_text_format = (
            InsertTextFormat.SNIPPET if insert_format == 2 else InsertTextFormat.PLAIN_TEXT
        )

        # Handle text edit
        text_edit = None
        lsp_edit = lsp_item.get("textEdit")
        if lsp_edit:
            lsp_range = lsp_edit.get("range", {})
            start = lsp_range.get("start", {})
            end = lsp_range.get("end", {})
            text_edit = TextEdit(
                range=Range(
                    start=Position(
                        line=start.get("line", 0),
                        character=start.get("character", 0),
                    ),
                    end=Position(
                        line=end.get("line", 0),
                        character=end.get("character", 0),
                    ),
                ),
                new_text=lsp_edit.get("newText", ""),
            )

        # Handle label details
        label_details = None
        lsp_details = lsp_item.get("labelDetails")
        if lsp_details:
            label_details = CompletionItemLabelDetails(
                detail=lsp_details.get("detail"),
                description=lsp_details.get("description"),
            )

        item = CompletionItem(
            label=label,
            kind=kind,
            detail=lsp_item.get("detail"),
            documentation=documentation,
            deprecated=lsp_item.get("deprecated", False),
            preselect=lsp_item.get("preselect", False),
            sort_text=lsp_item.get("sortText"),
            filter_text=lsp_item.get("filterText"),
            insert_text=lsp_item.get("insertText"),
            insert_text_format=insert_text_format,
            text_edit=text_edit,
            commit_characters=lsp_item.get("commitCharacters", []),
            label_details=label_details,
            provider=self.name,
        )

        # Attach original LSP data for resolution
        item._lsp_data = lsp_item  # type: ignore

        return item
