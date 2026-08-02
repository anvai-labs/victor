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

"""Basic code chunking strategy using regex patterns.

This is a lightweight code chunker for core that uses regex to detect
function/class boundaries. For full AST-based chunking with tree-sitter,
use victor_coding.codebase.chunker.CodeChunker instead.
"""

import logging
import re
from typing import List

from victor.core.chunking.base import Chunk, ChunkingConfig, ChunkingStrategy

logger = logging.getLogger(__name__)


class CodeChunkingStrategy(ChunkingStrategy):
    """Chunk code preserving function/class boundaries.

    Uses regex patterns to detect code structure boundaries.
    For more sophisticated AST-based chunking, use victor_coding.

    Example:
        strategy = CodeChunkingStrategy()
        chunks = strategy.chunk("def foo():\\n    pass\\n\\ndef bar():\\n    pass")
    """

    # Function/method definitions across languages
    FUNCTION_PATTERNS = [
        r"^(async\s+)?def\s+\w+",  # Python
        r"^(async\s+)?function\s+\w+",  # JavaScript
        r"^(pub\s+)?(async\s+)?fn\s+\w+",  # Rust
        r"^func\s+\w+",  # Go
        r"^(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(",  # Java/C#
    ]

    # Class/struct definitions
    CLASS_PATTERNS = [
        r"^class\s+\w+",  # Python, JavaScript, Java
        r"^(pub\s+)?struct\s+\w+",  # Rust, Go
        r"^(pub\s+)?enum\s+\w+",  # Rust, Java
        r"^interface\s+\w+",  # TypeScript, Java
        r"^(pub\s+)?impl\s+",  # Rust
    ]

    def __init__(self, config=None):
        super().__init__(config)
        # Compile patterns
        self._function_re = re.compile(
            "|".join(f"({p})" for p in self.FUNCTION_PATTERNS),
            re.MULTILINE,
        )
        self._class_re = re.compile(
            "|".join(f"({p})" for p in self.CLASS_PATTERNS),
            re.MULTILINE,
        )

    @property
    def name(self) -> str:
        return "code"

    @property
    def supported_types(self) -> List[str]:
        return [
            "code",
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "c",
            "cpp",
            "csharp",
        ]

    def chunk(self, content: str) -> List[Chunk]:
        """Chunk code content by function/class boundaries.

        Args:
            content: Code content to chunk

        Returns:
            List of Chunk objects
        """
        if not content or not content.strip():
            return []

        # Find all definition points
        definitions = []

        for match in self._function_re.finditer(content):
            definitions.append(("function", match.start(), match.group()))

        for match in self._class_re.finditer(content):
            definitions.append(("class", match.start(), match.group()))

        # Sort by position
        definitions.sort(key=lambda x: x[1])

        if not definitions:
            # No definitions found - fall back to text chunking
            from victor.core.chunking.strategies.text import TextChunkingStrategy

            return TextChunkingStrategy(self.config).chunk(content)

        chunks = []

        # Content before first definition
        if definitions[0][1] > self.config.min_chunk_size:
            pre_content = content[: definitions[0][1]].strip()
            if len(pre_content) >= self.config.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=pre_content,
                        start_char=0,
                        end_char=definitions[0][1],
                        chunk_type="code_preamble",
                        metadata={"has_imports": "import" in pre_content.lower()},
                    )
                )

        # Chunk by definitions
        for i, (def_type, start, match_text) in enumerate(definitions):
            end = definitions[i + 1][1] if i + 1 < len(definitions) else len(content)
            section = content[start:end].strip()

            # Extract symbol name from match
            symbol_match = re.search(
                r"(?:def|function|fn|func|class|struct|enum|interface|impl)\s+(\w+)",
                match_text,
            )
            symbol_name = symbol_match.group(1) if symbol_match else None

            # If section is too large, sub-chunk it
            if len(section) > self.config.max_chunk_size:
                sub_chunks = self._chunk_large_definition(section, start, def_type, symbol_name)
                chunks.extend(sub_chunks)
            elif len(section) >= self.config.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=section,
                        start_char=start,
                        end_char=end,
                        chunk_type=f"code_{def_type}",
                        metadata={
                            "symbol_type": def_type,
                            "symbol_name": symbol_name,
                        },
                    )
                )

        logger.debug(f"Code chunking produced {len(chunks)} chunks")
        return chunks

    def _chunk_large_definition(
        self,
        section: str,
        offset: int,
        def_type: str,
        symbol_name: str,
    ) -> List[Chunk]:
        """Sub-chunk a large function/class definition.

        Args:
            section: Code section to chunk
            offset: Character offset in original document
            def_type: Definition type (function/class)
            symbol_name: Name of the symbol

        Returns:
            List of Chunk objects
        """
        from victor.core.chunking.strategies.text import TextChunkingStrategy

        text_strategy = TextChunkingStrategy(self.config)
        sub_chunks = text_strategy.chunk(section)

        return [
            Chunk(
                content=sub.content,
                start_char=offset + sub.start_char,
                end_char=offset + sub.end_char,
                chunk_type=f"code_{def_type}_part",
                metadata={
                    "symbol_type": def_type,
                    "symbol_name": symbol_name,
                    "part_index": i,
                },
            )
            for i, sub in enumerate(sub_chunks)
        ]
