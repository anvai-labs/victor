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

"""Embedding and semantic search management.

This module provides EmbeddingManager, which handles embedding
initialization and semantic search. Extracted from ConversationManager
to follow the Single Responsibility Principle (SRP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

import logging
from typing import Any, Dict, List, Optional

from victor.agent.protocols import IEmbeddingManager

logger = logging.getLogger(__name__)


class EmbeddingManager(IEmbeddingManager):
    """Manages embedding store and semantic search.

    This class is responsible for:
    - Initializing the embedding store
    - Performing semantic search over messages
    - Managing embedding service lifecycle

    SRP Compliance: Focuses only on embedding management, delegating
    message storage, context management, and session management to
    specialized components.

    Attributes:
        _embedding_store: Optional ConversationEmbeddingStore
        _enable_embeddings: Whether embeddings are enabled
    """

    def __init__(
        self,
        embedding_store: Optional[Any] = None,
        enable_embeddings: bool = True,
    ):
        """Initialize the embedding manager.

        Args:
            embedding_store: Optional ConversationEmbeddingStore
            enable_embeddings: Whether to enable embeddings
        """
        self._embedding_store = embedding_store
        self._enable_embeddings = enable_embeddings
        self._initialized = False

    async def initialize_embeddings(self) -> None:
        """Initialize embedding store."""
        if not self._enable_embeddings:
            logger.debug("Embeddings disabled, skipping initialization")
            return

        if not self._embedding_store:
            logger.warning("Cannot initialize embeddings: no embedding store")
            return

        if self._initialized:
            logger.debug("Embeddings already initialized")
            return

        try:
            await self._embedding_store.initialize()
            self._initialized = True
            logger.info("Embedding store initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding store: {e}")
            self._initialized = False

    async def semantic_search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Perform semantic search.

        Args:
            query: Search query
            k: Number of results to return
            filters: Optional filters for search

        Returns:
            List of search results
        """
        if not self._enable_embeddings:
            logger.warning("Embeddings disabled, cannot perform semantic search")
            return []

        if not self._embedding_store or not self._initialized:
            logger.warning("Embedding store not initialized, cannot perform search")
            return []

        try:
            results = await self._embedding_store.search_similar_messages(
                query=query,
                k=k,
                filters=filters,
            )
            return results
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def is_initialized(self) -> bool:
        """Check if embedding store is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self._initialized and self._enable_embeddings

    async def add_message_to_embeddings(
        self,
        message_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a message to the embedding store.

        Args:
            message_id: Message ID
            content: Message content
            metadata: Optional metadata

        Returns:
            True if successful, False otherwise
        """
        if not self._enable_embeddings:
            return False

        if not self._embedding_store or not self._initialized:
            return False

        try:
            await self._embedding_store.add_message(
                message_id=message_id,
                content=content,
                metadata=metadata or {},
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to add message to embeddings: {e}")
            return False
