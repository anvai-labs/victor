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

"""
Basic embeddings provider implementation.

This module provides a simple hash-based embeddings provider for graceful
degradation when full embedding models are not available.

SOLID Principles:
- SRP: BasicEmbeddingsProvider only handles basic embedding generation
- OCP: Extensible through protocol implementation
- LSP: Implements EmbeddingsProtocol completely
- ISP: Focused on embedding operations
- DIP: No dependencies on concrete implementations

Usage:
    from victor.contrib.embeddings import BasicEmbeddingsProvider

    embeddings = BasicEmbeddingsProvider()
    vector = await embeddings.embed_text("Hello world")
    # Note: Returns hash-based vectors, not semantic embeddings
    # For proper embeddings, install victor-rag
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List

from victor.framework.vertical_protocols import (
    EmbeddingsProtocol,
)

logger = logging.getLogger(__name__)


class BasicEmbeddingsProvider(EmbeddingsProtocol):
    """
    Basic hash-based embeddings provider for graceful degradation.

    This provider generates simple hash-based vectors that are deterministic
    but NOT semantically meaningful. It's intended as a fallback when full
    embedding models (from victor-rag or similar) are not available.

    For production use with semantic search, install victor-rag:
        pip install victor-rag

    Example:
        embeddings = BasicEmbeddingsProvider(dimension=384)
        vector = await embeddings.embed_text("Hello world")
        # Returns deterministic hash-based vector
    """

    def __init__(
        self,
        dimension: int = 384,
        model: str = "basic-hash",
    ) -> None:
        """Initialize the embeddings provider.

        Args:
            dimension: Dimension of embedding vectors to generate
            model: Model name identifier
        """
        self._dimension = dimension
        self._model = model

    @property
    def semantic(self) -> bool:
        """Whether this provider produces semantically meaningful vectors."""
        return False

    async def embed_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> List[float]:
        """Embed a single text string using hash-based approach.

        Note: This generates deterministic vectors based on text hash,
        NOT semantic embeddings. For semantic embeddings, use victor-rag.

        Args:
            text: Text to embed
            **kwargs: Additional options (ignored)

        Returns:
            List of floats representing the embedding vector
        """
        logger.warning(
            "Using hash-based embeddings (NOT semantic). "
            "Semantic search, tool selection, and intent classification "
            "will produce incorrect results. Install victor-rag for proper embeddings."
        )

        # Generate deterministic hash-based vector
        text_hash = hashlib.sha256(text.encode()).digest()
        vector: List[float] = []

        # Expand hash to desired dimension
        for i in range(self._dimension):
            byte_index = i % len(text_hash)
            # Convert byte to float in [-1, 1] range
            value = (text_hash[byte_index] / 127.5) - 1.0
            vector.append(value)

        return vector

    async def embed_batch(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed
            **kwargs: Additional options (ignored)

        Returns:
            List of embedding vectors
        """
        embeddings: List[List[float]] = []
        for text in texts:
            vector = await self.embed_text(text)
            embeddings.append(vector)
        return embeddings

    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._dimension

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "name": self._model,
            "dimension": self._dimension,
            "type": "hash-based",
            "semantic": False,
            "info": {
                "note": "Not semantic — install victor-rag for proper embeddings",
            },
        }


__all__ = ["BasicEmbeddingsProvider"]
