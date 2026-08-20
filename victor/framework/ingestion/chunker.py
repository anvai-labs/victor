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

"""Base document chunker with common chunking strategies.

This module provides domain-agnostic chunking algorithms that can be
reused across all verticals. Verticals can extend BaseChunker or use
it directly with their own configurations.

Strategies:
- Text: Sentence-boundary aware chunking
- Markdown: Header-preserving chunking
- Code: Function/class boundary chunking
- HTML: Semantic element chunking
- JSON: Object/array boundary chunking

Usage:
    from victor.framework.ingestion import BaseChunker, ChunkingConfig

    chunker = BaseChunker(ChunkingConfig(chunk_size=1000))
    chunks = chunker.chunk(content, doc_type="markdown")
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
from json import JSONDecodeError
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from victor.framework.ingestion.models import Chunk, ChunkingConfig

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Document type for content detection and chunking strategy selection."""

    HTML = "html"
    MARKDOWN = "markdown"
    CODE = "code"
    JSON = "json"
    XML = "xml"
    TEXT = "text"


# Document type detection from file extensions
EXTENSION_TO_DOCTYPE: Dict[str, DocumentType] = {
    # HTML/Web
    ".html": DocumentType.HTML,
    ".htm": DocumentType.HTML,
    ".xhtml": DocumentType.HTML,
    # Markdown
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".rst": DocumentType.MARKDOWN,
    # Code (common languages)
    ".py": DocumentType.CODE,
    ".js": DocumentType.CODE,
    ".ts": DocumentType.CODE,
    ".jsx": DocumentType.CODE,
    ".tsx": DocumentType.CODE,
    ".java": DocumentType.CODE,
    ".go": DocumentType.CODE,
    ".rs": DocumentType.CODE,
    ".c": DocumentType.CODE,
    ".cpp": DocumentType.CODE,
    ".h": DocumentType.CODE,
    ".cs": DocumentType.CODE,
    ".rb": DocumentType.CODE,
    ".php": DocumentType.CODE,
    ".swift": DocumentType.CODE,
    ".kt": DocumentType.CODE,
    # Data formats
    ".json": DocumentType.JSON,
    ".yaml": DocumentType.TEXT,
    ".yml": DocumentType.TEXT,
    ".xml": DocumentType.XML,
    ".csv": DocumentType.TEXT,
    # Documents
    ".txt": DocumentType.TEXT,
    ".pdf": DocumentType.TEXT,  # After extraction
}


def detect_document_type(source: str, content: str) -> DocumentType:
    """Detect document type from source URL/path and content.

    Priority:
    1. File extension from source URL/path
    2. Content-based detection (HTML tags, JSON braces, etc.)
    3. Default to "text"

    Args:
        source: Source URL or file path
        content: Document content

    Returns:
        Document type enum value
    """
    source_lower = source.lower()

    # Check for file extension
    for ext, doc_type in EXTENSION_TO_DOCTYPE.items():
        if source_lower.endswith(ext):
            logger.debug(f"Detected doc_type={doc_type} from extension {ext}")
            return doc_type

    # Content-based detection
    content_sample = content[:1000].strip()

    # HTML detection
    if re.search(
        r"<(!DOCTYPE|html|head|body|div|p|table|section)\b",
        content_sample,
        re.IGNORECASE,
    ):
        logger.debug("Detected doc_type=html from content")
        return DocumentType.HTML

    # JSON detection
    if content_sample.startswith(("{", "[")):
        try:
            json_loads(content[:10000])
            logger.debug("Detected doc_type=json from content")
            return DocumentType.JSON
        except JSONDecodeError:
            pass

    # XML detection
    if content_sample.startswith("<?xml") or re.match(r"<\w+[^>]*>", content_sample):
        if not re.search(r"<(html|head|body)\b", content_sample, re.IGNORECASE):
            logger.debug("Detected doc_type=xml from content")
            return DocumentType.XML

    # Markdown detection
    if re.search(r"^#{1,6}\s+\w+", content_sample, re.MULTILINE) or "```" in content_sample:
        logger.debug("Detected doc_type=markdown from content")
        return DocumentType.MARKDOWN

    # Code detection
    if re.search(
        r"^(def |class |function |fn |func |import |from |package )",
        content_sample,
        re.MULTILINE,
    ):
        logger.debug("Detected doc_type=code from content")
        return DocumentType.CODE

    logger.debug("Defaulting to doc_type=text")
    return DocumentType.TEXT


class BaseChunker:
    """Base document chunker with multiple strategies.

    Provides domain-agnostic chunking algorithms. Verticals can:
    - Use this directly with custom ChunkingConfig
    - Extend this class to add domain-specific strategies
    - Wrap this class to add domain-specific metadata

    Example:
        # Direct usage
        chunker = BaseChunker(ChunkingConfig(chunk_size=1000))
        chunks = chunker.chunk(content, "markdown")

        # Vertical extension
        class ASTChunker(BaseChunker):
            def _chunk_code(self, content: str) -> List[Tuple[str, int, int]]:
                # Use tree-sitter for better code chunking
                ...
    """

    # Regex patterns for chunking
    SENTENCE_END = re.compile(r"[.!?]\s+")
    CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    FUNCTION_DEF = re.compile(r"^(def|async def|function|fn|func)\s+\w+", re.MULTILINE)
    CLASS_DEF = re.compile(r"^(class|struct|interface|impl)\s+\w+", re.MULTILINE)
    MARKDOWN_HEADER = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize chunker.

        Args:
            config: Chunking configuration (uses defaults if None)
        """
        self._config = config or ChunkingConfig()

    @property
    def config(self) -> ChunkingConfig:
        """Get chunking configuration."""
        return self._config

    def chunk(
        self,
        content: str,
        doc_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Chunk content into smaller pieces.

        Args:
            content: Text content to chunk
            doc_type: Document type for strategy selection
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        # Select strategy based on document type
        if doc_type == "html":
            raw_chunks = self._chunk_html(content)
        elif doc_type == "code":
            raw_chunks = self._chunk_code(content)
        elif doc_type == "markdown":
            raw_chunks = self._chunk_markdown(content)
        elif doc_type == "json":
            raw_chunks = self._chunk_json(content)
        else:
            raw_chunks = self._chunk_text(content)

        # Convert to Chunk objects
        chunks = []
        for i, (text, start, end) in enumerate(raw_chunks):
            chunk = Chunk(
                content=text,
                start_char=start,
                end_char=end,
                chunk_index=i,
                doc_type=doc_type,
                metadata=metadata.copy() if metadata else {},
            )
            chunks.append(chunk)

        return chunks

    def _chunk_text(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk plain text with sentence boundaries.

        Args:
            content: Text content

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        chunks = []
        current_pos = 0

        while current_pos < len(content):
            chunk_end = min(current_pos + self._config.chunk_size, len(content))

            # Try to find a sentence boundary
            if self._config.respect_sentence_boundaries and chunk_end < len(content):
                search_start = max(current_pos + self._config.min_chunk_size, chunk_end - 100)
                search_text = content[search_start : chunk_end + 50]

                match = None
                for m in self.SENTENCE_END.finditer(search_text):
                    match = m

                if match:
                    chunk_end = search_start + match.end()

            chunk_text = content[current_pos:chunk_end].strip()

            if len(chunk_text) >= self._config.min_chunk_size:
                chunks.append((chunk_text, current_pos, chunk_end))

            current_pos = chunk_end - self._config.chunk_overlap
            if current_pos >= len(content) - self._config.min_chunk_size:
                break

        return chunks

    def _chunk_markdown(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk markdown preserving structure.

        Args:
            content: Markdown content

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        chunks = []
        headers = list(self.MARKDOWN_HEADER.finditer(content))

        if not headers:
            return self._chunk_text(content)

        for i, header in enumerate(headers):
            start = header.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)

            section = content[start:end].strip()

            if len(section) > self._config.max_chunk_size:
                sub_chunks = self._chunk_text(section)
                for sub_text, sub_start, sub_end in sub_chunks:
                    chunks.append((sub_text, start + sub_start, start + sub_end))
            elif len(section) >= self._config.min_chunk_size:
                chunks.append((section, start, end))

        return chunks

    def _chunk_code(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk code preserving function/class boundaries.

        Args:
            content: Code content

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        chunks = []
        definitions = []

        for match in self.FUNCTION_DEF.finditer(content):
            definitions.append(("function", match.start()))
        for match in self.CLASS_DEF.finditer(content):
            definitions.append(("class", match.start()))

        definitions.sort(key=lambda x: x[1])

        if not definitions:
            return self._chunk_text(content)

        # Add content before first definition
        if definitions[0][1] > self._config.min_chunk_size:
            pre_content = content[: definitions[0][1]].strip()
            if len(pre_content) >= self._config.min_chunk_size:
                chunks.append((pre_content, 0, definitions[0][1]))

        for i, (def_type, start) in enumerate(definitions):
            end = definitions[i + 1][1] if i + 1 < len(definitions) else len(content)
            section = content[start:end].strip()

            if len(section) > self._config.max_chunk_size:
                sub_chunks = self._chunk_text(section)
                for sub_text, sub_start, sub_end in sub_chunks:
                    chunks.append((sub_text, start + sub_start, start + sub_end))
            elif len(section) >= self._config.min_chunk_size:
                chunks.append((section, start, end))

        return chunks

    def _chunk_html(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk HTML preserving semantic structure.

        Args:
            content: HTML content

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("BeautifulSoup not available, falling back to text chunking")
            return self._chunk_text(content)

        chunks = []
        soup = BeautifulSoup(content, "lxml")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        semantic_elements = []

        # Try sections first
        for section in soup.find_all(["section", "article"]):
            text = section.get_text(separator=" ", strip=True)
            if len(text) >= self._config.min_chunk_size:
                semantic_elements.append(("section", text))

        # Fall back to headers
        if not semantic_elements:
            for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                header_text = header.get_text(strip=True)
                content_parts = [header_text]

                for sibling in header.find_next_siblings():
                    if sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        break
                    text = sibling.get_text(separator=" ", strip=True)
                    if text:
                        content_parts.append(text)

                combined = "\n".join(content_parts)
                if len(combined) >= self._config.min_chunk_size:
                    semantic_elements.append(("header_section", combined))

        # Fall back to paragraphs
        if not semantic_elements:
            for elem in soup.find_all(["p", "table", "div", "li"]):
                text = elem.get_text(separator=" ", strip=True)
                if len(text) >= self._config.min_chunk_size // 2:
                    semantic_elements.append((elem.name, text))

        if not semantic_elements:
            full_text = soup.get_text(separator="\n", strip=True)
            return self._chunk_text(full_text)

        # Convert to chunks with size limits
        current_chunk = []
        current_size = 0
        pos = 0

        for elem_type, text in semantic_elements:
            if len(text) > self._config.max_chunk_size:
                if current_chunk:
                    combined = "\n\n".join(current_chunk)
                    chunks.append((combined, pos, pos + len(combined)))
                    pos += len(combined)
                    current_chunk = []
                    current_size = 0

                sub_chunks = self._chunk_text(text)
                for sub_text, _, _ in sub_chunks:
                    chunks.append((sub_text, pos, pos + len(sub_text)))
                    pos += len(sub_text)
            elif current_size + len(text) > self._config.chunk_size:
                if current_chunk:
                    combined = "\n\n".join(current_chunk)
                    chunks.append((combined, pos, pos + len(combined)))
                    pos += len(combined)
                current_chunk = [text]
                current_size = len(text)
            else:
                current_chunk.append(text)
                current_size += len(text)

        if current_chunk:
            combined = "\n\n".join(current_chunk)
            if len(combined) >= self._config.min_chunk_size:
                chunks.append((combined, pos, pos + len(combined)))

        return chunks if chunks else self._chunk_text(soup.get_text())

    def _chunk_json(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk JSON preserving object boundaries.

        Args:
            content: JSON content

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        try:
            data = json_loads(content)
        except JSONDecodeError:
            logger.warning("Invalid JSON, falling back to text chunking")
            return self._chunk_text(content)

        chunks = []
        pos = 0

        if isinstance(data, dict):
            for key, value in data.items():
                chunk_data = {key: value}
                chunk_text = json_dumps(chunk_data, indent=2)

                if len(chunk_text) > self._config.max_chunk_size:
                    sub_chunks = self._chunk_text(json_dumps(value, indent=2))
                    for sub_text, _, _ in sub_chunks:
                        header = f"Key: {key}\n"
                        chunks.append((header + sub_text, pos, pos + len(header + sub_text)))
                        pos += len(header + sub_text)
                elif len(chunk_text) >= self._config.min_chunk_size:
                    chunks.append((chunk_text, pos, pos + len(chunk_text)))
                    pos += len(chunk_text)

        elif isinstance(data, list):
            current_batch = []
            current_size = 0

            for item in data:
                item_text = json_dumps(item, indent=2)

                if len(item_text) > self._config.max_chunk_size:
                    if current_batch:
                        batch_text = json_dumps(current_batch, indent=2)
                        chunks.append((batch_text, pos, pos + len(batch_text)))
                        pos += len(batch_text)
                        current_batch = []
                        current_size = 0

                    sub_chunks = self._chunk_text(item_text)
                    for sub_text, _, _ in sub_chunks:
                        chunks.append((sub_text, pos, pos + len(sub_text)))
                        pos += len(sub_text)
                elif current_size + len(item_text) > self._config.chunk_size:
                    if current_batch:
                        batch_text = json_dumps(current_batch, indent=2)
                        chunks.append((batch_text, pos, pos + len(batch_text)))
                        pos += len(batch_text)
                    current_batch = [item]
                    current_size = len(item_text)
                else:
                    current_batch.append(item)
                    current_size += len(item_text)

            if current_batch:
                batch_text = json_dumps(current_batch, indent=2)
                if len(batch_text) >= self._config.min_chunk_size:
                    chunks.append((batch_text, pos, pos + len(batch_text)))

        return chunks if chunks else self._chunk_text(content)

    def estimate_chunks(self, content: str) -> int:
        """Estimate number of chunks for content.

        Args:
            content: Content to estimate

        Returns:
            Estimated chunk count
        """
        if not content:
            return 0

        effective_size = self._config.chunk_size - self._config.chunk_overlap
        return max(1, len(content) // effective_size)


__all__ = [
    "BaseChunker",
    "detect_document_type",
    "EXTENSION_TO_DOCTYPE",
]
