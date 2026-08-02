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

"""Markdown chunking strategy preserving document structure."""

import logging
import re
from typing import List, Tuple

from victor.core.chunking.base import Chunk, ChunkingConfig, ChunkingStrategy

logger = logging.getLogger(__name__)


class MarkdownChunkingStrategy(ChunkingStrategy):
    """Chunk Markdown preserving header-based structure.

    Uses headers (# ## ###) as natural section boundaries.
    Preserves code blocks as atomic units.

    Example:
        strategy = MarkdownChunkingStrategy()
        chunks = strategy.chunk("# Title\\n\\nContent...\\n\\n## Section\\n\\nMore content...")
    """

    # Markdown patterns
    HEADER = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    HORIZONTAL_RULE = re.compile(r"^[-*_]{3,}$", re.MULTILINE)

    @property
    def name(self) -> str:
        return "markdown"

    @property
    def supported_types(self) -> List[str]:
        return ["markdown", "md", "rst", "restructuredtext"]

    def chunk(self, content: str) -> List[Chunk]:
        """Chunk Markdown content by headers.

        Args:
            content: Markdown content to chunk

        Returns:
            List of Chunk objects
        """
        if not content or not content.strip():
            return []

        # Find all headers
        headers = list(self.HEADER.finditer(content))

        if not headers:
            # No headers - fall back to text chunking
            from victor.core.chunking.strategies.text import TextChunkingStrategy

            return TextChunkingStrategy(self.config).chunk(content)

        chunks = []

        # Content before first header
        if headers[0].start() > self.config.min_chunk_size:
            pre_content = content[: headers[0].start()].strip()
            if len(pre_content) >= self.config.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=pre_content,
                        start_char=0,
                        end_char=headers[0].start(),
                        chunk_type="markdown_preamble",
                    )
                )

        # Process each header section
        for i, header in enumerate(headers):
            start = header.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)

            section = content[start:end].strip()
            header_level = len(header.group(1))
            header_text = header.group(2)

            # If section is too large, sub-chunk it
            if len(section) > self.config.max_chunk_size:
                sub_chunks = self._chunk_large_section(section, start, header_level, header_text)
                chunks.extend(sub_chunks)
            elif len(section) >= self.config.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=section,
                        start_char=start,
                        end_char=end,
                        chunk_type="markdown_section",
                        metadata={
                            "header_level": header_level,
                            "header_text": header_text,
                        },
                    )
                )

        logger.debug(f"Markdown chunking produced {len(chunks)} chunks")
        return chunks

    def _chunk_large_section(
        self,
        section: str,
        offset: int,
        header_level: int,
        header_text: str,
    ) -> List[Chunk]:
        """Sub-chunk a large section while preserving code blocks.

        Args:
            section: Section content
            offset: Character offset in original document
            header_level: Header level (1-6)
            header_text: Header text

        Returns:
            List of Chunk objects
        """
        from victor.core.chunking.strategies.text import TextChunkingStrategy

        # Preserve code blocks - don't split them
        code_blocks = list(self.CODE_BLOCK.finditer(section))

        if not code_blocks:
            # No code blocks - use text chunking
            text_strategy = TextChunkingStrategy(self.config)
            sub_chunks = text_strategy.chunk(section)
            return [
                Chunk(
                    content=sub.content,
                    start_char=offset + sub.start_char,
                    end_char=offset + sub.end_char,
                    chunk_type="markdown_section_part",
                    metadata={
                        "header_level": header_level,
                        "header_text": header_text,
                    },
                )
                for sub in sub_chunks
            ]

        # Split around code blocks
        chunks = []
        pos = 0

        for code_block in code_blocks:
            # Text before code block
            if code_block.start() > pos:
                text_before = section[pos : code_block.start()].strip()
                if len(text_before) >= self.config.min_chunk_size:
                    text_strategy = TextChunkingStrategy(self.config)
                    for sub in text_strategy.chunk(text_before):
                        chunks.append(
                            Chunk(
                                content=sub.content,
                                start_char=offset + pos + sub.start_char,
                                end_char=offset + pos + sub.end_char,
                                chunk_type="markdown_text",
                                metadata={"header_text": header_text},
                            )
                        )

            # Code block itself
            code_content = code_block.group()
            chunks.append(
                Chunk(
                    content=code_content,
                    start_char=offset + code_block.start(),
                    end_char=offset + code_block.end(),
                    chunk_type="markdown_code_block",
                    metadata={"header_text": header_text},
                )
            )
            pos = code_block.end()

        # Text after last code block
        if pos < len(section):
            text_after = section[pos:].strip()
            if len(text_after) >= self.config.min_chunk_size:
                text_strategy = TextChunkingStrategy(self.config)
                for sub in text_strategy.chunk(text_after):
                    chunks.append(
                        Chunk(
                            content=sub.content,
                            start_char=offset + pos + sub.start_char,
                            end_char=offset + pos + sub.end_char,
                            chunk_type="markdown_text",
                            metadata={"header_text": header_text},
                        )
                    )

        return chunks
