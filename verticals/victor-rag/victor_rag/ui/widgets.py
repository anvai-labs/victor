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

"""RAG UI Widgets - Reusable components for RAG interfaces."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

try:
    from textual.app import ComposeResult
    from textual.containers import Container, Vertical
    from textual.message import Message
    from textual.widgets import ListItem, ListView, Markdown, Static
    from textual.widget import Widget

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False

    class Widget:
        pass

    class ComposeResult:
        pass


if TEXTUAL_AVAILABLE:

    class DocumentList(Widget):
        """Widget for displaying and managing documents.

        Shows a list of documents with metadata and supports
        selection and deletion.
        """

        DEFAULT_CSS = """
        DocumentList {
            height: 100%;
        }

        .doc-item {
            height: auto;
            padding: 1;
            margin-bottom: 1;
            background: $surface;
        }

        .doc-item:hover {
            background: $primary-background;
        }

        .doc-item-selected {
            background: $primary;
        }

        .doc-title {
            text-style: bold;
        }

        .doc-meta {
            color: $text-muted;
        }
        """

        class DocumentSelected(Message):
            """Message sent when a document is selected."""

            def __init__(self, doc_id: str) -> None:
                super().__init__()
                self.doc_id = doc_id

        def __init__(self, documents: Optional[List[Any]] = None, *args, **kwargs):
            """Initialize document list.

            Args:
                documents: List of Document objects
            """
            super().__init__(*args, **kwargs)
            self._documents = documents or []
            self._selected_id: Optional[str] = None

        def compose(self) -> ComposeResult:
            """Compose the widget."""
            with Vertical(id="doc-list-container"):
                if not self._documents:
                    yield Static("No documents. Use 'Ingest' to add documents.")
                else:
                    for doc in self._documents:
                        created = datetime.fromtimestamp(doc.created_at).strftime("%Y-%m-%d %H:%M")
                        with Container(
                            classes="doc-item",
                            id=f"doc-item-{doc.id}",
                        ):
                            yield Static(f"{doc.source}", classes="doc-title")
                            yield Static(
                                f"ID: {doc.id} | Type: {doc.doc_type} | "
                                f"Size: {len(doc.content):,} chars | Created: {created}",
                                classes="doc-meta",
                            )

        def update_documents(self, documents: List[Any]) -> None:
            """Update the document list.

            Args:
                documents: New list of documents
            """
            self._documents = documents
            self.refresh(recompose=True)

        async def on_click(self, event) -> None:
            """Handle click events for selection."""
            # Find clicked document item
            for widget in self.query(".doc-item"):
                if widget.id and widget.region.contains_point(event.screen_offset):
                    doc_id = widget.id.replace("doc-item-", "")
                    self._select_document(doc_id)
                    self.post_message(self.DocumentSelected(doc_id))
                    break

        def _select_document(self, doc_id: str) -> None:
            """Select a document visually."""
            # Remove previous selection
            for item in self.query(".doc-item-selected"):
                item.remove_class("doc-item-selected")

            # Add selection to new item
            selected = self.query_one(f"#doc-item-{doc_id}", Container)
            if selected:
                selected.add_class("doc-item-selected")
                self._selected_id = doc_id

        @property
        def selected_id(self) -> Optional[str]:
            """Get the selected document ID."""
            return self._selected_id

    class SearchResults(Widget):
        """Widget for displaying search results.

        Shows search results with scores, content previews,
        and source information.
        """

        DEFAULT_CSS = """
        SearchResults {
            height: 100%;
            overflow-y: auto;
        }

        .result-item {
            height: auto;
            padding: 1;
            margin-bottom: 1;
            background: $surface;
            border: solid $primary;
        }

        .result-header {
            text-style: bold;
        }

        .result-score {
            color: $accent;
        }

        .result-content {
            margin-top: 1;
        }

        .result-source {
            color: $text-muted;
            text-style: italic;
            margin-top: 1;
        }

        .no-results {
            color: $text-muted;
            content-align: center middle;
            height: 100%;
        }
        """

        class ResultSelected(Message):
            """Message sent when a result is selected."""

            def __init__(self, chunk_id: str, doc_id: str) -> None:
                super().__init__()
                self.chunk_id = chunk_id
                self.doc_id = doc_id

        def __init__(
            self,
            results: Optional[List[Any]] = None,
            query: str = "",
            *args,
            **kwargs,
        ):
            """Initialize search results widget.

            Args:
                results: List of DocumentSearchResult objects
                query: The search query
            """
            super().__init__(*args, **kwargs)
            self._results = results or []
            self._query = query

        def compose(self) -> ComposeResult:
            """Compose the widget."""
            if not self._results:
                yield Static(
                    "No results. Enter a search query above.",
                    classes="no-results",
                )
                return

            yield Static(
                f"Found {len(self._results)} results for: '{self._query}'",
                classes="result-header",
            )

            for i, result in enumerate(self._results, 1):
                chunk = result.chunk
                source = result.doc_source or chunk.metadata.get("source", "unknown")
                preview = chunk.content[:300]
                if len(chunk.content) > 300:
                    preview += "..."

                content = f"""**[{i}]** Score: {result.score:.3f}

{preview}

*Source: {source}*
"""
                yield Markdown(
                    content,
                    classes="result-item",
                    id=f"result-{chunk.id}",
                )

        def update_results(self, results: List[Any], query: str = "") -> None:
            """Update search results.

            Args:
                results: New search results
                query: The search query
            """
            self._results = results
            self._query = query
            self.refresh(recompose=True)

        def clear(self) -> None:
            """Clear results."""
            self._results = []
            self._query = ""
            self.refresh(recompose=True)

else:
    # Stubs when Textual is not available
    class DocumentList:
        """Stub when Textual is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError("Textual required for RAG UI")

    class SearchResults:
        """Stub when Textual is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError("Textual required for RAG UI")
