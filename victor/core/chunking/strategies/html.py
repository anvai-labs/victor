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

"""HTML chunking strategy using BeautifulSoup for semantic parsing.

Ideal for documents like SEC 10-K filings, web pages, and HTML documentation.
Preserves semantic structure by chunking at paragraph, section, and table boundaries.
"""

import logging
from typing import List, Optional

from victor.core.chunking.base import Chunk, ChunkingConfig, ChunkingStrategy

logger = logging.getLogger(__name__)


class HTMLChunkingStrategy(ChunkingStrategy):
    """Chunk HTML preserving semantic structure.

    Uses BeautifulSoup to parse HTML and extract semantic elements:
    - Sections and articles (highest priority)
    - Headers with following content
    - Paragraphs and tables
    - List items

    Example:
        strategy = HTMLChunkingStrategy()
        chunks = strategy.chunk("<html><body><p>First para</p><p>Second para</p></body></html>")
    """

    @property
    def name(self) -> str:
        return "html"

    @property
    def supported_types(self) -> List[str]:
        return ["html", "htm", "xhtml"]

    def chunk(self, content: str) -> List[Chunk]:
        """Chunk HTML content preserving semantic structure.

        Args:
            content: HTML content to chunk

        Returns:
            List of Chunk objects
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("BeautifulSoup not available, falling back to text chunking")
            from victor.core.chunking.strategies.text import TextChunkingStrategy

            return TextChunkingStrategy(self.config).chunk(content)

        if not content or not content.strip():
            return []

        soup = BeautifulSoup(content, "lxml")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "meta", "link"]):
            tag.decompose()

        # Extract semantic elements
        semantic_elements = self._extract_semantic_elements(soup)

        if not semantic_elements:
            # Fallback to full text
            full_text = soup.get_text(separator="\n", strip=True)
            from victor.core.chunking.strategies.text import TextChunkingStrategy

            return TextChunkingStrategy(self.config).chunk(full_text)

        # Convert to chunks respecting size limits
        return self._build_chunks(semantic_elements)

    def _extract_semantic_elements(self, soup) -> List[tuple]:
        """Extract semantic elements from parsed HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of (element_type, text) tuples
        """
        elements = []

        # Try sections/articles first (common in modern HTML)
        for section in soup.find_all(["section", "article"]):
            text = section.get_text(separator=" ", strip=True)
            if len(text) >= self.config.min_chunk_size:
                elements.append(("section", text))

        if elements:
            return elements

        # Try headers with following content
        for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            header_text = header.get_text(strip=True)
            content_parts = [header_text]

            # Collect siblings until next header
            for sibling in header.find_next_siblings():
                if sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    break
                text = sibling.get_text(separator=" ", strip=True)
                if text:
                    content_parts.append(text)

            combined = "\n".join(content_parts)
            if len(combined) >= self.config.min_chunk_size:
                elements.append(("header_section", combined))

        if elements:
            return elements

        # Fall back to paragraphs, tables, divs, list items
        for elem in soup.find_all(["p", "table", "div", "li", "blockquote", "pre"]):
            text = elem.get_text(separator=" ", strip=True)
            # Lower threshold for individual elements
            if len(text) >= self.config.min_chunk_size // 2:
                elements.append((elem.name, text))

        return elements

    def _build_chunks(self, elements: List[tuple]) -> List[Chunk]:
        """Build chunks from semantic elements respecting size limits.

        Args:
            elements: List of (element_type, text) tuples

        Returns:
            List of Chunk objects
        """
        from victor.core.chunking.strategies.text import TextChunkingStrategy

        text_strategy = TextChunkingStrategy(self.config)
        chunks = []
        current_parts = []
        current_size = 0
        pos = 0

        for elem_type, text in elements:
            # Large element - sub-chunk it
            if len(text) > self.config.max_chunk_size:
                # Flush current buffer
                if current_parts:
                    combined = "\n\n".join(current_parts)
                    chunks.append(
                        Chunk(
                            content=combined,
                            start_char=pos,
                            end_char=pos + len(combined),
                            chunk_type="html_combined",
                            metadata={"element_count": len(current_parts)},
                        )
                    )
                    pos += len(combined)
                    current_parts = []
                    current_size = 0

                # Sub-chunk the large element
                sub_chunks = text_strategy.chunk(text)
                for sub in sub_chunks:
                    chunks.append(
                        Chunk(
                            content=sub.content,
                            start_char=pos,
                            end_char=pos + len(sub.content),
                            chunk_type=f"html_{elem_type}",
                        )
                    )
                    pos += len(sub.content)

            # Would exceed chunk size - flush and start new
            elif current_size + len(text) > self.config.chunk_size:
                if current_parts:
                    combined = "\n\n".join(current_parts)
                    chunks.append(
                        Chunk(
                            content=combined,
                            start_char=pos,
                            end_char=pos + len(combined),
                            chunk_type="html_combined",
                            metadata={"element_count": len(current_parts)},
                        )
                    )
                    pos += len(combined)
                current_parts = [text]
                current_size = len(text)

            # Add to current buffer
            else:
                current_parts.append(text)
                current_size += len(text)

        # Flush remaining
        if current_parts:
            combined = "\n\n".join(current_parts)
            if len(combined) >= self.config.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=combined,
                        start_char=pos,
                        end_char=pos + len(combined),
                        chunk_type="html_combined",
                        metadata={"element_count": len(current_parts)},
                    )
                )

        logger.debug(f"HTML chunking produced {len(chunks)} chunks")
        return chunks
