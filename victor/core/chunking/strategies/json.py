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

"""JSON chunking strategy preserving object structure."""

import json
import logging
from typing import Any, List

from victor.core.chunking.base import Chunk, ChunkingConfig, ChunkingStrategy

logger = logging.getLogger(__name__)


class JSONChunkingStrategy(ChunkingStrategy):
    """Chunk JSON preserving object/array structure.

    For objects: chunks by top-level keys
    For arrays: batches items together up to chunk_size

    Example:
        strategy = JSONChunkingStrategy()
        chunks = strategy.chunk('{"key1": "value1", "key2": "value2"}')
    """

    @property
    def name(self) -> str:
        return "json"

    @property
    def supported_types(self) -> List[str]:
        return ["json", "jsonl"]

    def chunk(self, content: str) -> List[Chunk]:
        """Chunk JSON content preserving structure.

        Args:
            content: JSON content to chunk

        Returns:
            List of Chunk objects
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON: {e}, falling back to text chunking")
            from victor.core.chunking.strategies.text import TextChunkingStrategy

            return TextChunkingStrategy(self.config).chunk(content)

        if isinstance(data, dict):
            return self._chunk_object(data)
        elif isinstance(data, list):
            return self._chunk_array(data)
        else:
            # Primitive value - return as single chunk
            return [
                Chunk(
                    content=json.dumps(data, indent=2),
                    start_char=0,
                    end_char=len(content),
                    chunk_type="json_primitive",
                )
            ]

    def _chunk_object(self, data: dict) -> List[Chunk]:
        """Chunk a JSON object by top-level keys.

        Args:
            data: Parsed JSON object

        Returns:
            List of Chunk objects
        """
        from victor.core.chunking.strategies.text import TextChunkingStrategy

        text_strategy = TextChunkingStrategy(self.config)
        chunks = []
        pos = 0

        for key, value in data.items():
            chunk_data = {key: value}
            chunk_text = json.dumps(chunk_data, indent=2, ensure_ascii=False)

            # Large value - sub-chunk it
            if len(chunk_text) > self.config.max_chunk_size:
                value_text = json.dumps(value, indent=2, ensure_ascii=False)
                sub_chunks = text_strategy.chunk(value_text)
                for i, sub in enumerate(sub_chunks):
                    header = f"Key: {key} (part {i + 1}/{len(sub_chunks)})\n"
                    full_content = header + sub.content
                    chunks.append(
                        Chunk(
                            content=full_content,
                            start_char=pos,
                            end_char=pos + len(full_content),
                            chunk_type="json_value_part",
                            metadata={
                                "key": key,
                                "part": i + 1,
                                "total_parts": len(sub_chunks),
                            },
                        )
                    )
                    pos += len(full_content)
            elif len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        start_char=pos,
                        end_char=pos + len(chunk_text),
                        chunk_type="json_object",
                        metadata={"key": key},
                    )
                )
                pos += len(chunk_text)

        logger.debug(f"JSON object chunking produced {len(chunks)} chunks")
        return chunks

    def _chunk_array(self, data: list) -> List[Chunk]:
        """Chunk a JSON array by batching items.

        Args:
            data: Parsed JSON array

        Returns:
            List of Chunk objects
        """
        from victor.core.chunking.strategies.text import TextChunkingStrategy

        text_strategy = TextChunkingStrategy(self.config)
        chunks = []
        current_batch = []
        current_size = 0
        pos = 0

        for i, item in enumerate(data):
            item_text = json.dumps(item, indent=2, ensure_ascii=False)

            # Large item - sub-chunk it
            if len(item_text) > self.config.max_chunk_size:
                # Flush current batch
                if current_batch:
                    batch_text = json.dumps(current_batch, indent=2, ensure_ascii=False)
                    chunks.append(
                        Chunk(
                            content=batch_text,
                            start_char=pos,
                            end_char=pos + len(batch_text),
                            chunk_type="json_array_batch",
                            metadata={"item_count": len(current_batch)},
                        )
                    )
                    pos += len(batch_text)
                    current_batch = []
                    current_size = 0

                # Sub-chunk large item
                sub_chunks = text_strategy.chunk(item_text)
                for j, sub in enumerate(sub_chunks):
                    header = f"Array item {i} (part {j + 1}/{len(sub_chunks)})\n"
                    full_content = header + sub.content
                    chunks.append(
                        Chunk(
                            content=full_content,
                            start_char=pos,
                            end_char=pos + len(full_content),
                            chunk_type="json_array_item_part",
                            metadata={"index": i, "part": j + 1},
                        )
                    )
                    pos += len(full_content)

            # Would exceed chunk size - flush batch
            elif current_size + len(item_text) > self.config.chunk_size:
                if current_batch:
                    batch_text = json.dumps(current_batch, indent=2, ensure_ascii=False)
                    chunks.append(
                        Chunk(
                            content=batch_text,
                            start_char=pos,
                            end_char=pos + len(batch_text),
                            chunk_type="json_array_batch",
                            metadata={"item_count": len(current_batch)},
                        )
                    )
                    pos += len(batch_text)
                current_batch = [item]
                current_size = len(item_text)

            # Add to batch
            else:
                current_batch.append(item)
                current_size += len(item_text)

        # Flush remaining
        if current_batch:
            batch_text = json.dumps(current_batch, indent=2, ensure_ascii=False)
            if len(batch_text) >= self.config.min_chunk_size:
                chunks.append(
                    Chunk(
                        content=batch_text,
                        start_char=pos,
                        end_char=pos + len(batch_text),
                        chunk_type="json_array_batch",
                        metadata={"item_count": len(current_batch)},
                    )
                )

        logger.debug(f"JSON array chunking produced {len(chunks)} chunks")
        return chunks
