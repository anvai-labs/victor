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

"""RAG Panel - Main TUI interface for RAG operations.

Provides an interactive panel for:
- Document ingestion with drag-and-drop
- Search with real-time results
- Q&A with source citations
- Document management
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional

try:
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical
    from textual.message import Message
    from textual.reactive import reactive
    from textual.widgets import (
        Button,
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        Markdown,
        Static,
        TabbedContent,
        TabPane,
    )
    from textual.widget import Widget

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False

    # Stub classes for when Textual is not available
    class Widget:
        pass

    class ComposeResult:
        pass


if TEXTUAL_AVAILABLE:

    class RAGPanel(Widget):
        """Main RAG interface panel.

        Features:
        - Tabbed interface: Search, Ingest, Documents
        - Real-time search results
        - Document list with delete support
        - Status bar with statistics

        Bindings:
            s: Focus search input
            i: Switch to ingest tab
            d: Switch to documents tab
            r: Refresh document list
        """

        DEFAULT_CSS = """
        RAGPanel {
            height: 100%;
            width: 100%;
        }

        .rag-header {
            dock: top;
            height: 3;
            background: $primary;
            color: $text;
            content-align: center middle;
        }

        .rag-content {
            height: 1fr;
        }

        .search-input {
            dock: top;
            height: 3;
            margin: 1;
        }

        .results-container {
            height: 1fr;
            overflow-y: auto;
            padding: 1;
        }

        .result-item {
            height: auto;
            margin-bottom: 1;
            padding: 1;
            background: $surface;
            border: solid $primary;
        }

        .result-score {
            color: $accent;
        }

        .result-source {
            color: $text-muted;
            text-style: italic;
        }

        .status-bar {
            dock: bottom;
            height: 1;
            background: $surface;
            color: $text-muted;
        }

        .ingest-form {
            padding: 2;
        }

        .form-row {
            height: 3;
            margin-bottom: 1;
        }

        .doc-list {
            height: 1fr;
        }

        .doc-item {
            height: auto;
            padding: 1;
        }
        """

        BINDINGS = [
            Binding("s", "focus_search", "Search"),
            Binding("i", "tab_ingest", "Ingest"),
            Binding("d", "tab_documents", "Documents"),
            Binding("r", "refresh", "Refresh"),
        ]

        # Reactive properties
        doc_count: reactive[int] = reactive(0)
        chunk_count: reactive[int] = reactive(0)
        search_count: reactive[int] = reactive(0)

        def __init__(
            self,
            *args,
            on_search: Optional[Callable[[str], Any]] = None,
            on_ingest: Optional[Callable[[str], Any]] = None,
            on_delete: Optional[Callable[[str], Any]] = None,
            **kwargs,
        ):
            """Initialize RAG panel.

            Args:
                on_search: Callback for search queries
                on_ingest: Callback for document ingestion
                on_delete: Callback for document deletion
            """
            super().__init__(*args, **kwargs)
            self._on_search = on_search
            self._on_ingest = on_ingest
            self._on_delete = on_delete
            self._results: List[Any] = []
            self._documents: List[Any] = []

        def compose(self) -> ComposeResult:
            """Compose the panel layout."""
            yield Static("RAG Knowledge Base", classes="rag-header")

            with TabbedContent(initial="search"):
                # Search Tab
                with TabPane("Search", id="search"):
                    yield Input(
                        placeholder="Enter search query...",
                        id="search-input",
                        classes="search-input",
                    )
                    yield Container(id="results-container", classes="results-container")

                # Ingest Tab
                with TabPane("Ingest", id="ingest"):
                    with Vertical(classes="ingest-form"):
                        yield Label("File Path:")
                        yield Input(
                            placeholder="/path/to/document.md",
                            id="ingest-path",
                            classes="form-row",
                        )
                        yield Label("Document Type:")
                        with Horizontal(classes="form-row"):
                            yield Button("Text", id="type-text", variant="default")
                            yield Button("Markdown", id="type-markdown", variant="primary")
                            yield Button("Code", id="type-code", variant="default")
                            yield Button("PDF", id="type-pdf", variant="default")
                        yield Button("Ingest Document", id="ingest-btn", variant="success")
                        yield Static(id="ingest-status")

                # Documents Tab
                with TabPane("Documents", id="documents"):
                    yield ListView(id="doc-list", classes="doc-list")
                    with Horizontal():
                        yield Button("Refresh", id="refresh-btn")
                        yield Button("Delete Selected", id="delete-btn", variant="error")

            yield Static(
                f"Docs: {self.doc_count} | Chunks: {self.chunk_count} | Searches: {self.search_count}",
                id="status-bar",
                classes="status-bar",
            )

        async def on_mount(self) -> None:
            """Handle mount event."""
            await self._refresh_stats()
            await self._refresh_documents()

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            """Handle input submission."""
            if event.input.id == "search-input":
                await self._do_search(event.value)
            elif event.input.id == "ingest-path":
                await self._do_ingest(event.value)

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button press."""
            button_id = event.button.id

            if button_id == "ingest-btn":
                path_input = self.query_one("#ingest-path", Input)
                await self._do_ingest(path_input.value)
            elif button_id == "refresh-btn":
                await self._refresh_documents()
            elif button_id == "delete-btn":
                await self._delete_selected()
            elif button_id and button_id.startswith("type-"):
                # Document type selection
                doc_type = button_id.replace("type-", "")
                self._selected_doc_type = doc_type
                # Update button states
                for btn in self.query("Button"):
                    if btn.id and btn.id.startswith("type-"):
                        btn.variant = "primary" if btn.id == button_id else "default"

        async def _do_search(self, query: str) -> None:
            """Execute search query."""
            if not query.strip():
                return

            if self._on_search:
                results = await self._on_search(query)
                self._results = results or []
                await self._render_results()
            else:
                # Default search using document store
                from victor_rag.document_store import DocumentStore

                store = DocumentStore()
                await store.initialize()
                self._results = await store.search(query, k=10)
                await self._render_results()

            self.search_count += 1
            self._update_status()

        async def _render_results(self) -> None:
            """Render search results."""
            container = self.query_one("#results-container", Container)
            await container.remove_children()

            if not self._results:
                await container.mount(Static("No results found."))
                return

            for i, result in enumerate(self._results, 1):
                chunk = result.chunk
                source = result.doc_source or chunk.metadata.get("source", "unknown")

                content = f"""**[{i}]** Score: {result.score:.3f}

{chunk.content[:300]}{'...' if len(chunk.content) > 300 else ''}

*Source: {source}*
"""
                await container.mount(Markdown(content, classes="result-item"))

        async def _do_ingest(self, path: str) -> None:
            """Ingest a document."""
            if not path.strip():
                return

            status = self.query_one("#ingest-status", Static)
            status.update("Ingesting...")

            try:
                if self._on_ingest:
                    result = await self._on_ingest(path)
                    status.update(f"Success: {result}")
                else:
                    # Default ingestion
                    from victor_rag.document_store import Document, DocumentStore

                    file_path = Path(path)
                    if not file_path.exists():
                        status.update(f"Error: File not found: {path}")
                        return

                    content = file_path.read_text()
                    doc_type = getattr(self, "_selected_doc_type", "text")

                    doc = Document(
                        id=f"doc_{hash(path) % 10000:04d}",
                        content=content,
                        source=str(file_path),
                        doc_type=doc_type,
                    )

                    store = DocumentStore()
                    await store.initialize()
                    chunks = await store.add_document(doc)

                    status.update(f"Success! Created {len(chunks)} chunks from {file_path.name}")

                await self._refresh_stats()
                await self._refresh_documents()

            except Exception as e:
                status.update(f"Error: {str(e)}")

        async def _refresh_documents(self) -> None:
            """Refresh document list."""
            from victor_rag.document_store import DocumentStore

            store = DocumentStore()
            await store.initialize()
            self._documents = await store.list_documents()

            doc_list = self.query_one("#doc-list", ListView)
            await doc_list.clear()

            for doc in self._documents:
                item = ListItem(
                    Static(f"[{doc.id}] {doc.source} ({doc.doc_type})"),
                    id=f"doc-{doc.id}",
                )
                await doc_list.append(item)

        async def _delete_selected(self) -> None:
            """Delete selected document."""
            doc_list = self.query_one("#doc-list", ListView)
            if doc_list.highlighted_child:
                doc_id = doc_list.highlighted_child.id
                if doc_id and doc_id.startswith("doc-"):
                    doc_id = doc_id[4:]  # Remove "doc-" prefix

                    if self._on_delete:
                        await self._on_delete(doc_id)
                    else:
                        from victor_rag.document_store import DocumentStore

                        store = DocumentStore()
                        await store.initialize()
                        await store.delete_document(doc_id)

                    await self._refresh_documents()
                    await self._refresh_stats()

        async def _refresh_stats(self) -> None:
            """Refresh statistics."""
            from victor_rag.document_store import DocumentStore

            store = DocumentStore()
            await store.initialize()
            stats = store.get_stats()

            self.doc_count = stats.get("total_documents", 0)
            self.chunk_count = stats.get("total_chunks", 0)
            self.search_count = stats.get("total_searches", 0)
            self._update_status()

        def _update_status(self) -> None:
            """Update status bar."""
            status = self.query_one("#status-bar", Static)
            status.update(
                f"Docs: {self.doc_count} | Chunks: {self.chunk_count} | Searches: {self.search_count}"
            )

        def action_focus_search(self) -> None:
            """Focus the search input."""
            self.query_one("#search-input", Input).focus()

        def action_tab_ingest(self) -> None:
            """Switch to ingest tab."""
            tabs = self.query_one(TabbedContent)
            tabs.active = "ingest"

        def action_tab_documents(self) -> None:
            """Switch to documents tab."""
            tabs = self.query_one(TabbedContent)
            tabs.active = "documents"

        async def action_refresh(self) -> None:
            """Refresh documents."""
            await self._refresh_documents()
            await self._refresh_stats()

else:
    # Stub when Textual is not available
    class RAGPanel:
        """Stub RAG panel when Textual is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Textual is required for RAG UI. " "Install with: pip install textual"
            )
