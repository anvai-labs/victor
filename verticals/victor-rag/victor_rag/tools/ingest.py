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

"""RAG Ingest Tool - Ingest documents into the RAG store."""

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import aiohttp

from victor.tools.base import BaseTool, CostTier, ToolResult

logger = logging.getLogger(__name__)


class RAGIngestTool(BaseTool):
    """Ingest documents into the RAG knowledge base.

    Supports multiple sources:
    - Local files (.txt, .md, .py, .pdf, etc.)
    - Web URLs (fetches and extracts text)
    - Direct content input
    - Directory batch ingestion

    Example:
        # Ingest a file
        result = await tool.execute(path="/path/to/document.md")

        # Ingest from URL
        result = await tool.execute(url="https://example.com/docs")

        # Ingest a directory
        result = await tool.execute(path="/path/to/docs", recursive=True)
    """

    name = "rag_ingest"
    description = (
        "Ingest documents into the RAG knowledge base. "
        "Supports local files, URLs, and directories."
    )

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to file or directory to ingest",
            },
            "url": {
                "type": "string",
                "description": "URL to fetch and ingest (alternative to path)",
            },
            "content": {
                "type": "string",
                "description": "Direct content to ingest (alternative to path/url)",
            },
            "doc_type": {
                "type": "string",
                "description": "Document type: text, markdown, code, pdf, html",
                "enum": ["text", "markdown", "code", "pdf", "html"],
                "default": "text",
            },
            "doc_id": {
                "type": "string",
                "description": "Optional custom document ID",
            },
            "recursive": {
                "type": "boolean",
                "description": "Recursively ingest directory contents",
                "default": False,
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern for directory ingestion (e.g., '*.md')",
                "default": "*",
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata to attach to the document",
            },
        },
        "required": [],
    }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.MEDIUM

    async def execute(
        self,
        path: Optional[str] = None,
        url: Optional[str] = None,
        content: Optional[str] = None,
        doc_type: str = "text",
        doc_id: Optional[str] = None,
        recursive: bool = False,
        pattern: str = "*",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute document ingestion.

        Args:
            path: Path to document file or directory
            url: URL to fetch and ingest
            content: Direct content to ingest
            doc_type: Document type
            doc_id: Optional custom ID
            recursive: Recursively ingest directory contents
            pattern: Glob pattern for directory ingestion
            metadata: Optional metadata

        Returns:
            ToolResult with ingestion status
        """
        from victor_rag.document_store import Document

        try:
            # Get or create document store
            store = self._get_document_store()
            await store.initialize()

            # Handle URL ingestion
            if url:
                return await self._ingest_url(store, url, doc_type, doc_id, metadata)

            # Handle directory batch ingestion
            if path:
                file_path = Path(path)
                if file_path.is_dir():
                    return await self._ingest_directory(
                        store, file_path, recursive, pattern, metadata
                    )

            # Handle single file ingestion
            if path and not content:
                file_path = Path(path)
                if not file_path.exists():
                    return ToolResult(
                        success=False,
                        output=f"File not found: {path}",
                    )

                # Detect doc_type from extension if not specified
                if doc_type == "text":
                    ext = file_path.suffix.lower()
                    doc_type = self._detect_doc_type(ext)

                # Read content
                if doc_type == "pdf":
                    content = await self._read_pdf(file_path)
                else:
                    content = file_path.read_text(encoding="utf-8")

                source = str(file_path)
            elif content:
                source = "direct_input"
            else:
                return ToolResult(
                    success=False,
                    output="Either 'path', 'url', or 'content' must be provided",
                )

            # Generate doc_id if not provided
            if not doc_id:
                doc_id = f"doc_{uuid.uuid4().hex[:12]}"

            # Create document
            doc = Document(
                id=doc_id,
                content=content,
                source=source,
                doc_type=doc_type,
                metadata=metadata or {},
            )

            # Ingest document
            chunks = await store.add_document(doc)

            return ToolResult(
                success=True,
                output=(
                    f"Successfully ingested document:\n"
                    f"  ID: {doc_id}\n"
                    f"  Source: {source}\n"
                    f"  Type: {doc_type}\n"
                    f"  Chunks: {len(chunks)}\n"
                    f"  Characters: {len(content):,}"
                ),
            )

        except Exception as e:
            logger.exception(f"Failed to ingest document: {e}")
            return ToolResult(
                success=False,
                output=f"Failed to ingest document: {str(e)}",
            )

    async def _ingest_url(
        self,
        store,
        url: str,
        doc_type: str,
        doc_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> ToolResult:
        """Ingest content from a URL."""
        from victor_rag.document_store import Document

        try:
            # Fetch URL content
            content, detected_type = await self._fetch_url(url)

            # Use detected type if not explicitly specified
            if doc_type == "text":
                doc_type = detected_type

            # Generate doc_id from URL if not provided
            if not doc_id:
                # Create readable ID from URL
                parsed = urlparse(url)
                url_slug = re.sub(r"[^\w\-]", "_", parsed.path.strip("/"))[:30]
                doc_id = f"url_{url_slug}_{uuid.uuid4().hex[:6]}"

            # Create document
            doc = Document(
                id=doc_id,
                content=content,
                source=url,
                doc_type=doc_type,
                metadata={**(metadata or {}), "url": url},
            )

            # Ingest document
            chunks = await store.add_document(doc)

            return ToolResult(
                success=True,
                output=(
                    f"Successfully ingested URL:\n"
                    f"  ID: {doc_id}\n"
                    f"  URL: {url}\n"
                    f"  Type: {doc_type}\n"
                    f"  Chunks: {len(chunks)}\n"
                    f"  Characters: {len(content):,}"
                ),
            )

        except Exception as e:
            logger.exception(f"Failed to fetch URL: {e}")
            return ToolResult(
                success=False,
                output=f"Failed to fetch URL {url}: {str(e)}",
            )

    async def _fetch_url(self, url: str) -> Tuple[str, str]:
        """Fetch content from URL and extract text.

        Returns:
            Tuple of (content, doc_type)
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=60),
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; VictorRAG/1.0)",
                    "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
                },
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                raw_content = await response.text()

                # Determine document type and extract text
                if "html" in content_type:
                    content = self._extract_text_from_html(raw_content)
                    return content, "html"
                elif "json" in content_type:
                    return raw_content, "text"
                elif "pdf" in content_type:
                    # For PDF URLs, we'd need to download and parse
                    # For now, return raw (user should download locally)
                    return raw_content, "text"
                else:
                    return raw_content, "text"

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML content."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up excessive whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n\n".join(lines)

        except ImportError:
            # Fallback: basic regex-based extraction
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

    async def _ingest_directory(
        self,
        store,
        directory: Path,
        recursive: bool,
        pattern: str,
        metadata: Optional[Dict[str, Any]],
    ) -> ToolResult:
        """Ingest all matching files from a directory."""
        from victor_rag.document_store import Document

        # Find matching files
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        # Filter to only files (not directories)
        files = [f for f in files if f.is_file()]

        if not files:
            return ToolResult(
                success=False,
                output=f"No files matching '{pattern}' found in {directory}",
            )

        # Ingest each file
        results = []
        total_chunks = 0
        total_chars = 0
        errors = []

        for file_path in files:
            try:
                # Skip binary files
                if file_path.suffix.lower() in {
                    ".pyc",
                    ".so",
                    ".dll",
                    ".exe",
                    ".bin",
                    ".ico",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".svg",
                }:
                    continue

                # Detect doc_type
                doc_type = self._detect_doc_type(file_path.suffix.lower())

                # Read content
                if doc_type == "pdf":
                    content = await self._read_pdf(file_path)
                else:
                    try:
                        content = file_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        logger.warning(f"Skipping binary file: {file_path}")
                        continue

                # Skip empty files
                if not content.strip():
                    continue

                # Generate doc_id from path
                rel_path = file_path.relative_to(directory)
                sanitized_path = re.sub(r"[^\w\-]", "_", str(rel_path))[:40]
                doc_id = f"dir_{sanitized_path}_{uuid.uuid4().hex[:4]}"

                # Create document
                doc = Document(
                    id=doc_id,
                    content=content,
                    source=str(file_path),
                    doc_type=doc_type,
                    metadata={**(metadata or {}), "directory": str(directory)},
                )

                # Ingest document
                chunks = await store.add_document(doc)
                results.append((file_path.name, len(chunks)))
                total_chunks += len(chunks)
                total_chars += len(content)

            except Exception as e:
                errors.append(f"{file_path.name}: {str(e)}")
                logger.warning(f"Failed to ingest {file_path}: {e}")

        # Build output
        output_lines = [
            f"Directory ingestion complete: {directory}",
            f"  Files processed: {len(results)}",
            f"  Total chunks: {total_chunks}",
            f"  Total characters: {total_chars:,}",
            "",
            "Files ingested:",
        ]

        for name, chunks in results[:20]:  # Show first 20
            output_lines.append(f"  - {name} ({chunks} chunks)")

        if len(results) > 20:
            output_lines.append(f"  ... and {len(results) - 20} more")

        if errors:
            output_lines.append("")
            output_lines.append(f"Errors ({len(errors)}):")
            for err in errors[:5]:
                output_lines.append(f"  - {err}")

        return ToolResult(
            success=len(results) > 0,
            output="\n".join(output_lines),
        )

    def _get_document_store(self):
        """Get or create document store instance."""
        from victor_rag.document_store import DocumentStore

        # Use a singleton pattern for the store
        if not hasattr(self, "_store"):
            self._store = DocumentStore()
        return self._store

    def _detect_doc_type(self, extension: str) -> str:
        """Detect document type from file extension."""
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".rs",
            ".go",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".sh",
        }
        markdown_extensions = {".md", ".markdown", ".rst"}

        if extension in code_extensions:
            return "code"
        elif extension in markdown_extensions:
            return "markdown"
        elif extension == ".pdf":
            return "pdf"
        else:
            return "text"

    async def _read_pdf(self, path: Path) -> str:
        """Read text from PDF file."""
        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    text_parts.append(f"[Page {page_num}]\n{text}")

            return "\n\n".join(text_parts)

        except ImportError:
            raise ImportError(
                "pypdf is required for PDF ingestion. " "Install with: pip install pypdf"
            )
