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

"""Source content handlers for document ingestion.

These handlers extract content from various sources in a domain-agnostic way.
Verticals can use these directly or extend them for domain-specific needs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from victor.framework.ingestion.models import DocumentType, SourceContent
from victor.framework.ingestion.chunker import (
    detect_document_type,
    EXTENSION_TO_DOCTYPE,
)

logger = logging.getLogger(__name__)


# Binary file extensions to skip
BINARY_EXTENSIONS: Set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".mkv",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".pyc",
    ".pyo",
    ".class",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".bin",
    ".dat",
}


def extract_text_from_html(html_content: str) -> str:
    """Extract readable text from HTML content.

    Removes scripts, styles, and extracts text while preserving structure.

    Args:
        html_content: Raw HTML string

    Returns:
        Extracted text content
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "lxml")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Get text with line breaks for block elements
        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        return text.strip()

    except ImportError:
        # Fallback: basic regex-based extraction
        logger.warning("BeautifulSoup not available, using basic HTML extraction")
        # Remove script and style tags
        text = re.sub(
            r"<script[^>]*>.*?</script>",
            "",
            html_content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove all tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Clean up entities and whitespace
        text = re.sub(r"&\w+;", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class FileHandler:
    """Handler for local file content extraction.

    Reads text-based files from the filesystem with encoding detection.

    Example:
        handler = FileHandler()
        if handler.can_handle("/path/to/file.py"):
            content = await handler.extract("/path/to/file.py")
    """

    def __init__(
        self,
        encodings: Optional[List[str]] = None,
        skip_binary: bool = True,
    ):
        """Initialize file handler.

        Args:
            encodings: List of encodings to try (default: utf-8, latin-1)
            skip_binary: Whether to skip binary files
        """
        self._encodings = encodings or ["utf-8", "latin-1", "cp1252"]
        self._skip_binary = skip_binary

    def can_handle(self, source: str) -> bool:
        """Check if this handler can process the source.

        Args:
            source: File path

        Returns:
            True if this is a readable file
        """
        path = Path(source)

        # Skip URLs
        if source.startswith(("http://", "https://")):
            return False

        # Check if file exists
        if not path.exists() or not path.is_file():
            return False

        # Skip binary files
        if self._skip_binary and path.suffix.lower() in BINARY_EXTENSIONS:
            return False

        return True

    async def extract(self, source: str) -> SourceContent:
        """Extract content from a file.

        Args:
            source: File path

        Returns:
            SourceContent with file text
        """
        path = Path(source)

        # Try different encodings
        content = None
        for encoding in self._encodings:
            try:
                content = path.read_text(encoding=encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            raise ValueError(f"Could not decode file with any encoding: {source}")

        # Detect document type
        doc_type_str = detect_document_type(source, content)
        doc_type = (
            DocumentType(doc_type_str)
            if doc_type_str in [e.value for e in DocumentType]
            else DocumentType.TEXT
        )

        return SourceContent(
            content=content,
            source=source,
            doc_type=doc_type,
            metadata={
                "file_size": path.stat().st_size,
                "file_name": path.name,
                "extension": path.suffix,
            },
        )


class PDFHandler:
    """Handler for PDF document extraction.

    Extracts text content from PDF files using pypdf.

    Example:
        handler = PDFHandler()
        if handler.can_handle("/path/to/document.pdf"):
            content = await handler.extract("/path/to/document.pdf")
    """

    def can_handle(self, source: str) -> bool:
        """Check if this is a PDF file.

        Args:
            source: File path

        Returns:
            True if this is a PDF file
        """
        if source.startswith(("http://", "https://")):
            return source.lower().endswith(".pdf")

        path = Path(source)
        return path.suffix.lower() == ".pdf" and path.exists()

    async def extract(self, source: str) -> SourceContent:
        """Extract text from a PDF file.

        Args:
            source: PDF file path

        Returns:
            SourceContent with extracted text
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF extraction: pip install pypdf")

        path = Path(source)
        reader = PdfReader(str(path))

        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(f"--- Page {i + 1} ---\n{text}")

        content = "\n\n".join(pages)

        return SourceContent(
            content=content,
            source=source,
            doc_type=DocumentType.TEXT,
            metadata={
                "page_count": len(reader.pages),
                "file_name": path.name,
            },
        )


class URLHandler:
    """Handler for URL content extraction.

    Fetches and extracts content from web pages.

    Example:
        handler = URLHandler()
        if handler.can_handle("https://example.com/page.html"):
            content = await handler.extract("https://example.com/page.html")
    """

    def __init__(
        self,
        timeout: int = 30,
        max_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        """Initialize URL handler.

        Args:
            timeout: Request timeout in seconds
            max_size: Maximum content size to download
        """
        self._timeout = timeout
        self._max_size = max_size

    def can_handle(self, source: str) -> bool:
        """Check if this is a URL.

        Args:
            source: URL string

        Returns:
            True if this is a valid HTTP(S) URL
        """
        return source.startswith(("http://", "https://"))

    async def extract(self, source: str) -> SourceContent:
        """Fetch and extract content from a URL.

        Args:
            source: URL to fetch

        Returns:
            SourceContent with page content
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for URL fetching: pip install httpx")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(source, follow_redirects=True)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            content = response.text

            # Check content size
            if len(content) > self._max_size:
                content = content[: self._max_size]
                logger.warning(f"Truncated content from {source} to {self._max_size} bytes")

        # Determine document type from content-type header and URL
        if "text/html" in content_type or "application/xhtml" in content_type:
            doc_type = DocumentType.HTML
            # Extract text from HTML
            content = extract_text_from_html(content)
        elif "application/json" in content_type:
            doc_type = DocumentType.JSON
        elif "text/markdown" in content_type:
            doc_type = DocumentType.MARKDOWN
        else:
            # Try to detect from content
            doc_type_str = detect_document_type(source, content)
            doc_type = (
                DocumentType(doc_type_str)
                if doc_type_str in [e.value for e in DocumentType]
                else DocumentType.TEXT
            )

        # Parse URL for metadata
        parsed = urlparse(source)

        return SourceContent(
            content=content,
            source=source,
            doc_type=doc_type,
            metadata={
                "content_type": content_type,
                "domain": parsed.netloc,
                "path": parsed.path,
            },
        )


class DirectoryHandler:
    """Handler for batch processing files in a directory.

    Recursively processes files matching specified patterns.

    Example:
        handler = DirectoryHandler(patterns=["*.py", "*.md"])
        files = await handler.discover("/path/to/directory")
    """

    def __init__(
        self,
        patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ):
        """Initialize directory handler.

        Args:
            patterns: Glob patterns to match (default: all text files)
            exclude_patterns: Patterns to exclude
            recursive: Whether to recurse into subdirectories
        """
        self._patterns = patterns or ["*"]
        self._exclude_patterns = exclude_patterns or [
            ".*",
            "__pycache__",
            "node_modules",
            ".git",
            ".venv",
            "venv",
        ]
        self._recursive = recursive
        self._file_handler = FileHandler()

    def can_handle(self, source: str) -> bool:
        """Check if this is a directory.

        Args:
            source: Directory path

        Returns:
            True if this is an existing directory
        """
        if source.startswith(("http://", "https://")):
            return False

        path = Path(source)
        return path.exists() and path.is_dir()

    async def discover(self, source: str) -> List[str]:
        """Discover files in a directory.

        Args:
            source: Directory path

        Returns:
            List of file paths matching patterns
        """
        path = Path(source)
        files = []

        for pattern in self._patterns:
            if self._recursive:
                matches = path.rglob(pattern)
            else:
                matches = path.glob(pattern)

            for file_path in matches:
                if not file_path.is_file():
                    continue

                # Check exclusions
                should_exclude = False
                for exclude in self._exclude_patterns:
                    if any(part.startswith(exclude.rstrip("*")) for part in file_path.parts):
                        should_exclude = True
                        break

                if should_exclude:
                    continue

                # Check if we can handle this file
                if self._file_handler.can_handle(str(file_path)):
                    files.append(str(file_path))

        return sorted(files)

    async def extract_all(self, source: str) -> List[SourceContent]:
        """Extract content from all files in a directory.

        Args:
            source: Directory path

        Returns:
            List of SourceContent for each file
        """
        files = await self.discover(source)
        contents = []

        for file_path in files:
            try:
                content = await self._file_handler.extract(file_path)
                contents.append(content)
            except Exception as e:
                logger.warning(f"Failed to extract {file_path}: {e}")
                continue

        return contents


__all__ = [
    "FileHandler",
    "PDFHandler",
    "URLHandler",
    "DirectoryHandler",
    "extract_text_from_html",
    "BINARY_EXTENSIONS",
]
