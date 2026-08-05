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

"""Document Chunker - Intelligent document chunking for RAG.

This module provides RAG-specific document chunking that builds on the
framework's BaseChunker. The generic chunking algorithms are in
victor/framework/ingestion/chunker.py.

This vertical-specific module:
- Provides DocumentChunker class with RAG-specific embedding integration
- Wraps framework's BaseChunker for core chunking strategies
- Maintains backward-compatible API (ChunkingConfig, detect_document_type)
- Adds RAG-specific metadata and document type handling

Design:
    - ChunkingConfig: Configuration dataclass (re-exported from framework)
    - DocumentChunker: RAG-specific chunker with embedding support
    - detect_document_type: Document type detection (from framework)

Example:
    chunker = DocumentChunker(ChunkingConfig(chunk_size=1024, overlap=128))

    # Chunk a document with embeddings
    doc = Document(id="1", content="...", source="doc.md", doc_type="markdown")
    chunks = await chunker.chunk_document(doc, embedding_fn)

    # Auto-detect document type from URL
    doc = Document(id="2", content="<html>...", source="https://sec.gov/filing.htm")
    chunks = await chunker.chunk_document(doc, embedding_fn)  # Uses HTML strategy
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List, Optional, Tuple

from victor_rag.document_store import Document, DocumentChunk

logger = logging.getLogger(__name__)

# Document type detection from file extensions and URL patterns
EXTENSION_TO_DOCTYPE = {
    # HTML/Web
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    # Markdown
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "markdown",
    # Code (common)
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".jsx": "code",
    ".tsx": "code",
    ".java": "code",
    ".go": "code",
    ".rs": "code",
    ".c": "code",
    ".cpp": "code",
    ".h": "code",
    ".cs": "code",
    ".rb": "code",
    ".php": "code",
    ".swift": "code",
    ".kt": "code",
    # Data formats
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".csv": "csv",
    # Documents
    ".txt": "text",
    ".pdf": "text",  # After extraction
}


@dataclass
class ChunkingConfig:
    """Configuration for document chunking.

    Optimized for BGE embedding model (BAAI/bge-small-en-v1.5):
    - 384 dimensions, 512 token context limit
    - Default chunk_size = 3.5x dimension = 1344 chars (~336 tokens)
    - Well within 512 token limit while maximizing context per chunk

    Attributes:
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        min_chunk_size: Minimum chunk size (avoid tiny chunks)
        max_chunk_size: Maximum chunk size (hard limit)
        respect_sentence_boundaries: Try to break at sentence ends
        respect_paragraph_boundaries: Try to break at paragraphs
        code_aware: Use code-aware chunking for code files
    """

    # 3.5x embedding dimension (384 * 3.5 = 1344) for optimal context utilization
    # ~336 tokens at 4 chars/token, safely under 512 token limit
    chunk_size: int = 1344
    chunk_overlap: int = 134  # ~10% overlap for context continuity
    min_chunk_size: int = 200  # Scaled proportionally
    max_chunk_size: int = 2000  # Keep max limit for edge cases
    respect_sentence_boundaries: bool = True
    respect_paragraph_boundaries: bool = True
    code_aware: bool = True


# Type alias for embedding function
EmbeddingFn = Callable[[str], Coroutine[Any, Any, List[float]]]


def detect_document_type(source: str, content: str) -> str:
    """Detect document type from source URL/path and content.

    Priority:
    1. File extension from source URL/path
    2. Content-based detection (HTML tags, JSON braces, etc.)
    3. Default to "text"

    Args:
        source: Source URL or file path
        content: Document content

    Returns:
        Document type: "html", "markdown", "code", "json", "xml", "text"
    """
    # Extract extension from source (handles URLs and paths)
    source_lower = source.lower()

    # Check for file extension
    for ext, doc_type in EXTENSION_TO_DOCTYPE.items():
        if source_lower.endswith(ext):
            logger.debug(f"Detected doc_type={doc_type} from extension {ext}")
            return doc_type

    # Content-based detection
    content_sample = content[:1000].strip()

    # HTML detection - look for HTML tags
    if re.search(
        r"<(!DOCTYPE|html|head|body|div|p|table|section)\b",
        content_sample,
        re.IGNORECASE,
    ):
        logger.debug("Detected doc_type=html from content")
        return "html"

    # JSON detection
    if content_sample.startswith(("{", "[")):
        try:
            json.loads(content[:10000])  # Try parsing a sample
            logger.debug("Detected doc_type=json from content")
            return "json"
        except json.JSONDecodeError:
            pass

    # XML detection
    if content_sample.startswith("<?xml") or re.match(r"<\w+[^>]*>", content_sample):
        if not re.search(r"<(html|head|body)\b", content_sample, re.IGNORECASE):
            logger.debug("Detected doc_type=xml from content")
            return "xml"

    # Markdown detection - headers, code blocks
    if re.search(r"^#{1,6}\s+\w+", content_sample, re.MULTILINE) or "```" in content_sample:
        logger.debug("Detected doc_type=markdown from content")
        return "markdown"

    # Code detection - function/class definitions
    if re.search(
        r"^(def |class |function |fn |func |import |from |package )",
        content_sample,
        re.MULTILINE,
    ):
        logger.debug("Detected doc_type=code from content")
        return "code"

    logger.debug("Defaulting to doc_type=text")
    return "text"


class DocumentChunker:
    """Intelligent document chunker with multiple strategies.

    Supports:
    - Plain text chunking with sentence boundaries
    - Markdown-aware chunking preserving structure
    - Code-aware chunking preserving functions/classes
    - Configurable overlap for context continuity

    Example:
        chunker = DocumentChunker()
        chunks = await chunker.chunk_document(doc, embedding_fn)
    """

    # Sentence ending patterns
    SENTENCE_END = re.compile(r"[.!?]\s+")

    # Code block patterns
    CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    FUNCTION_DEF = re.compile(r"^(def|async def|function|fn|func)\s+\w+", re.MULTILINE)
    CLASS_DEF = re.compile(r"^(class|struct|interface|impl)\s+\w+", re.MULTILINE)

    # Markdown patterns
    MARKDOWN_HEADER = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
    MARKDOWN_LIST = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()

    async def chunk_document(
        self,
        doc: Document,
        embedding_fn: EmbeddingFn,
    ) -> List[DocumentChunk]:
        """Chunk a document into indexed chunks.

        Selects the appropriate chunking strategy based on document type.
        Auto-detects type from source URL/path and content if doc_type is
        "auto" or "text".

        Args:
            doc: Document to chunk
            embedding_fn: Async function to generate embeddings

        Returns:
            List of document chunks with embeddings
        """
        # Auto-detect document type if not specified or generic
        doc_type = doc.doc_type
        if doc_type in ("auto", "text", None, ""):
            doc_type = detect_document_type(doc.source or "", doc.content)
            logger.info(f"Auto-detected document type: {doc_type} for source: {doc.source}")

        # Select chunking strategy based on document type
        if doc_type == "html":
            raw_chunks = self._chunk_html(doc.content)
        elif doc_type == "code":
            raw_chunks = self._chunk_code(doc.content)
        elif doc_type == "markdown":
            raw_chunks = self._chunk_markdown(doc.content)
        elif doc_type == "json":
            raw_chunks = self._chunk_json(doc.content)
        else:
            raw_chunks = self._chunk_text(doc.content)

        # Generate embeddings and create chunks
        chunks = []
        for i, (content, start, end) in enumerate(raw_chunks):
            embedding = await embedding_fn(content)

            chunk = DocumentChunk(
                id=f"{doc.id}_{i}_{uuid.uuid4().hex[:8]}",
                doc_id=doc.id,
                content=content,
                embedding=embedding,
                chunk_index=i,
                start_char=start,
                end_char=end,
                metadata={
                    "doc_type": doc.doc_type,
                    "source": doc.source,
                    **doc.metadata,
                },
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
            # Determine chunk end position
            chunk_end = min(current_pos + self.config.chunk_size, len(content))

            # Try to find a sentence boundary
            if self.config.respect_sentence_boundaries and chunk_end < len(content):
                # Look for sentence end in the last portion of the chunk
                search_start = max(current_pos + self.config.min_chunk_size, chunk_end - 100)
                search_text = content[search_start : chunk_end + 50]

                match = None
                for m in self.SENTENCE_END.finditer(search_text):
                    match = m

                if match:
                    chunk_end = search_start + match.end()

            # Extract chunk
            chunk_text = content[current_pos:chunk_end].strip()

            # Skip empty chunks
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append((chunk_text, current_pos, chunk_end))

            # Move to next chunk with overlap
            current_pos = chunk_end - self.config.chunk_overlap
            if current_pos >= len(content) - self.config.min_chunk_size:
                break

        return chunks

    def _chunk_markdown(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk markdown preserving structure.

        Respects headers and code blocks as natural break points.

        Args:
            content: Markdown content

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        chunks = []

        # Find all headers as potential break points
        headers = list(self.MARKDOWN_HEADER.finditer(content))

        if not headers:
            # Fall back to text chunking
            return self._chunk_text(content)

        # Chunk by headers
        for i, header in enumerate(headers):
            start = header.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)

            section = content[start:end].strip()

            # If section is too large, sub-chunk it
            if len(section) > self.config.max_chunk_size:
                sub_chunks = self._chunk_text(section)
                for sub_text, sub_start, sub_end in sub_chunks:
                    chunks.append((sub_text, start + sub_start, start + sub_end))
            elif len(section) >= self.config.min_chunk_size:
                chunks.append((section, start, end))

        return chunks

    def _chunk_code(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk code preserving function/class boundaries.

        Attempts to keep functions and classes intact.

        Args:
            content: Code content

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        chunks = []

        # Find function and class definitions
        definitions = []
        for match in self.FUNCTION_DEF.finditer(content):
            definitions.append(("function", match.start()))
        for match in self.CLASS_DEF.finditer(content):
            definitions.append(("class", match.start()))

        # Sort by position
        definitions.sort(key=lambda x: x[1])

        if not definitions:
            # Fall back to text chunking
            return self._chunk_text(content)

        # Add content before first definition
        if definitions[0][1] > self.config.min_chunk_size:
            pre_content = content[: definitions[0][1]].strip()
            if len(pre_content) >= self.config.min_chunk_size:
                chunks.append((pre_content, 0, definitions[0][1]))

        # Chunk by definitions
        for i, (def_type, start) in enumerate(definitions):
            end = definitions[i + 1][1] if i + 1 < len(definitions) else len(content)

            section = content[start:end].strip()

            # If section is too large, sub-chunk it
            if len(section) > self.config.max_chunk_size:
                sub_chunks = self._chunk_text(section)
                for sub_text, sub_start, sub_end in sub_chunks:
                    chunks.append((sub_text, start + sub_start, start + sub_end))
            elif len(section) >= self.config.min_chunk_size:
                chunks.append((section, start, end))

        return chunks

    def _chunk_html(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk HTML preserving semantic structure.

        Uses paragraph, section, table, and header elements as natural
        break points. Ideal for documents like SEC 10-K filings.

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
        soup = BeautifulSoup(content, "html.parser")

        # Remove script, style, and nav elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Semantic elements to chunk by (in priority order)
        # For SEC filings: sections, articles, divs with content, paragraphs, tables
        semantic_elements = []

        # First try to find major sections (common in SEC filings)
        for section in soup.find_all(["section", "article"]):
            text = section.get_text(separator=" ", strip=True)
            if len(text) >= self.config.min_chunk_size:
                semantic_elements.append(("section", text))

        # If no sections, chunk by headers + following content
        if not semantic_elements:
            for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                # Get header and following siblings until next header
                header_text = header.get_text(strip=True)
                content_parts = [header_text]

                for sibling in header.find_next_siblings():
                    if sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        break
                    text = sibling.get_text(separator=" ", strip=True)
                    if text:
                        content_parts.append(text)

                combined = "\n".join(content_parts)
                if len(combined) >= self.config.min_chunk_size:
                    semantic_elements.append(("header_section", combined))

        # If still empty, chunk by paragraphs and tables
        if not semantic_elements:
            for elem in soup.find_all(["p", "table", "div", "li"]):
                text = elem.get_text(separator=" ", strip=True)
                if len(text) >= self.config.min_chunk_size // 2:
                    semantic_elements.append((elem.name, text))

        # If still empty, use full text
        if not semantic_elements:
            full_text = soup.get_text(separator="\n", strip=True)
            return self._chunk_text(full_text)

        # Convert semantic elements to chunks with size limits
        current_chunk = []
        current_size = 0
        pos = 0

        for elem_type, text in semantic_elements:
            # If this element alone exceeds max size, sub-chunk it
            if len(text) > self.config.max_chunk_size:
                # Flush current chunk
                if current_chunk:
                    combined = "\n\n".join(current_chunk)
                    chunks.append((combined, pos, pos + len(combined)))
                    pos += len(combined)
                    current_chunk = []
                    current_size = 0

                # Sub-chunk the large element
                sub_chunks = self._chunk_text(text)
                for sub_text, _, _ in sub_chunks:
                    chunks.append((sub_text, pos, pos + len(sub_text)))
                    pos += len(sub_text)
            elif current_size + len(text) > self.config.chunk_size:
                # Flush current chunk and start new one
                if current_chunk:
                    combined = "\n\n".join(current_chunk)
                    chunks.append((combined, pos, pos + len(combined)))
                    pos += len(combined)
                current_chunk = [text]
                current_size = len(text)
            else:
                # Add to current chunk
                current_chunk.append(text)
                current_size += len(text)

        # Flush remaining
        if current_chunk:
            combined = "\n\n".join(current_chunk)
            if len(combined) >= self.config.min_chunk_size:
                chunks.append((combined, pos, pos + len(combined)))

        logger.debug(f"HTML chunking produced {len(chunks)} chunks")
        return chunks if chunks else self._chunk_text(soup.get_text())

    def _chunk_json(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk JSON preserving object boundaries.

        Chunks by top-level keys or array items.

        Args:
            content: JSON content

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON, falling back to text chunking")
            return self._chunk_text(content)

        chunks = []
        pos = 0

        if isinstance(data, dict):
            # Chunk by top-level keys
            for key, value in data.items():
                chunk_data = {key: value}
                chunk_text = json.dumps(chunk_data, indent=2)

                if len(chunk_text) > self.config.max_chunk_size:
                    # Sub-chunk large values
                    sub_chunks = self._chunk_text(json.dumps(value, indent=2))
                    for sub_text, _, _ in sub_chunks:
                        header = f"Key: {key}\n"
                        chunks.append((header + sub_text, pos, pos + len(header + sub_text)))
                        pos += len(header + sub_text)
                elif len(chunk_text) >= self.config.min_chunk_size:
                    chunks.append((chunk_text, pos, pos + len(chunk_text)))
                    pos += len(chunk_text)

        elif isinstance(data, list):
            # Chunk by array items (batch small items together)
            current_batch = []
            current_size = 0

            for item in data:
                item_text = json.dumps(item, indent=2)

                if len(item_text) > self.config.max_chunk_size:
                    # Flush batch and sub-chunk large item
                    if current_batch:
                        batch_text = json.dumps(current_batch, indent=2)
                        chunks.append((batch_text, pos, pos + len(batch_text)))
                        pos += len(batch_text)
                        current_batch = []
                        current_size = 0

                    sub_chunks = self._chunk_text(item_text)
                    for sub_text, _, _ in sub_chunks:
                        chunks.append((sub_text, pos, pos + len(sub_text)))
                        pos += len(sub_text)
                elif current_size + len(item_text) > self.config.chunk_size:
                    # Flush batch
                    if current_batch:
                        batch_text = json.dumps(current_batch, indent=2)
                        chunks.append((batch_text, pos, pos + len(batch_text)))
                        pos += len(batch_text)
                    current_batch = [item]
                    current_size = len(item_text)
                else:
                    current_batch.append(item)
                    current_size += len(item_text)

            # Flush remaining
            if current_batch:
                batch_text = json.dumps(current_batch, indent=2)
                if len(batch_text) >= self.config.min_chunk_size:
                    chunks.append((batch_text, pos, pos + len(batch_text)))

        logger.debug(f"JSON chunking produced {len(chunks)} chunks")
        return chunks if chunks else self._chunk_text(content)

    def estimate_chunks(self, content: str) -> int:
        """Estimate number of chunks for content.

        Useful for progress estimation.

        Args:
            content: Content to estimate

        Returns:
            Estimated chunk count
        """
        if not content:
            return 0

        effective_size = self.config.chunk_size - self.config.chunk_overlap
        return max(1, len(content) // effective_size)
