from __future__ import annotations

# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Static embedding collection for small, unchanging datasets.

This module provides a pickle/numpy-backed collection for:
- Tool definitions (~65 items, ~100KB)
- Intent classifications (~20 items, ~30KB)

These are small, static datasets that don't need a full vector database.
Simple pickle + numpy provides fast loading and searching.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.core.pickle_cache import (
    CacheValidation,
    delete_cache_file,
    invalid,
    load_validated_pickle,
    save_pickle_with_metadata,
    valid,
)
from victor.storage.embeddings.service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)

np = None  # type: ignore[assignment]


def _ensure_numpy():
    """Import numpy into module scope on first actual use."""

    global np
    if np is None:
        import numpy

        np = numpy
    return np


@dataclass
class CollectionItem:
    """An item in the static collection."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class StaticEmbeddingCollection:
    """A static embedding collection backed by pickle/numpy.

    Designed for small, unchanging datasets where a full vector database
    would be overkill. Provides:
    - Fast loading from pickle cache
    - Efficient numpy-based similarity search
    - Hash-based cache invalidation
    - Robust validation with auto-rebuild on corruption

    Usage:
        # Create collection
        from victor.config.settings import get_project_paths
        collection = StaticEmbeddingCollection(
            name="intents",
            cache_dir=get_project_paths().global_embeddings_dir,
        )

        # Initialize with items (computes embeddings if cache invalid)
        items = [
            CollectionItem(id="cont_1", text="Let me continue..."),
            CollectionItem(id="done_1", text="Here is the summary..."),
        ]
        await collection.initialize(items)

        # Search
        results = await collection.search("I'll analyze this next", top_k=3)
        for item, score in results:
            print(f"{item.id}: {score:.3f}")
    """

    # Cache version - aligned with Victor version, increment on breaking cache format changes
    # This ensures stale caches are automatically rebuilt
    CACHE_VERSION = "0.2.0"  # Added dimension validation and integrity checks

    def __init__(
        self,
        name: str,
        cache_dir: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """Initialize static embedding collection.

        Args:
            name: Collection name (used for cache file naming)
            cache_dir: Directory for cache files (default: ~/.victor/embeddings/)
            embedding_service: Shared embedding service (uses singleton if not provided)
        """
        _ensure_numpy()
        from victor.config.settings import get_project_paths

        self.name = name
        self.cache_dir = cache_dir or get_project_paths().global_embeddings_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use shared embedding service
        self.embedding_service = embedding_service or get_embedding_service()

        # Cache file path
        self.cache_file = self.cache_dir / f"{name}_collection.pkl"

        # In-memory data
        self._items: Dict[str, CollectionItem] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._item_ids: List[str] = []  # Maps index to item ID
        self._items_hash: Optional[str] = None

    @property
    def is_initialized(self) -> bool:
        """Check if collection is initialized."""
        return self._embeddings is not None and len(self._items) > 0

    @property
    def size(self) -> int:
        """Get number of items in collection."""
        return len(self._items)

    def _calculate_items_hash(self, items: List[CollectionItem]) -> str:
        """Calculate hash of items to detect changes.

        Args:
            items: List of collection items

        Returns:
            SHA256 hash of items
        """
        # Sort by ID for deterministic hash
        sorted_items = sorted(items, key=lambda x: x.id)
        hash_parts = [f"{item.id}:{item.text}" for item in sorted_items]
        combined = "|".join(hash_parts)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _delete_cache(self, reason: str) -> None:
        """Delete corrupted or stale cache file.

        Args:
            reason: Reason for deletion (for logging)
        """
        delete_cache_file(self.cache_file, reason, logger, label=f"Collection '{self.name}'")

    def _load_from_cache(self, items_hash: str) -> bool:
        """Load collection from cache if valid.

        Performs robust validation:
        1. Cache version check (breaking format changes)
        2. Items hash check (content changes)
        3. Model name check (embedding model changes)
        4. Embedding dimension validation (model dimension mismatch)
        5. Data integrity check (corrupt numpy arrays)

        Args:
            items_hash: Current hash of items

        Returns:
            True if loaded successfully, False otherwise
        """

        def _check_version(cache_data: Dict[str, Any]) -> CacheValidation:
            # 1. Check cache version (breaking changes to format)
            cached_version = cache_data.get("cache_version", 1)
            if cached_version != self.CACHE_VERSION:
                logger.debug(
                    f"Collection '{self.name}': cache version mismatch "
                    f"(cached: v{cached_version}, current: v{self.CACHE_VERSION})"
                )
                return invalid(delete=True, reason="version mismatch")
            return valid()

        def _check_items_hash(cache_data: Dict[str, Any]) -> CacheValidation:
            # 2. Verify cache matches current items
            if cache_data.get("items_hash") != items_hash:
                logger.debug(f"Collection '{self.name}': items changed, cache invalidated")
                # Items-hash mismatch intentionally keeps the cache file (no delete).
                return invalid(delete=False)
            return valid()

        def _check_model(cache_data: Dict[str, Any]) -> CacheValidation:
            # 3. Verify embedding model matches
            model_name = self.embedding_service.model_name
            if cache_data.get("model_name") != model_name:
                logger.debug(
                    f"Collection '{self.name}': model changed "
                    f"(cached: {cache_data.get('model_name')}, current: {model_name})"
                )
                return invalid(delete=True, reason="model mismatch")
            return valid()

        def _check_embeddings(cache_data: Dict[str, Any]) -> CacheValidation:
            # 4. Validate embedding dimensions match current model
            embeddings = cache_data.get("embeddings")
            if embeddings is None:
                logger.warning(f"Collection '{self.name}': cache missing embeddings array")
                return invalid(delete=True, reason="missing embeddings")

            expected_dim = self.embedding_service.dimension
            if len(embeddings.shape) != 2:
                logger.warning(
                    f"Collection '{self.name}': embeddings have wrong shape: {embeddings.shape}"
                )
                return invalid(delete=True, reason="invalid shape")

            cached_dim = embeddings.shape[1]
            if cached_dim != expected_dim:
                logger.debug(
                    f"Collection '{self.name}': embedding dimension mismatch "
                    f"(cached: {cached_dim}, expected: {expected_dim})"
                )
                return invalid(delete=True, reason="dimension mismatch")

            # 5. Validate item count matches embedding count
            items = cache_data.get("items", {})
            item_ids = cache_data.get("item_ids", [])
            if embeddings.shape[0] != len(items) or embeddings.shape[0] != len(item_ids):
                logger.warning(
                    f"Collection '{self.name}': data integrity error - "
                    f"embeddings: {embeddings.shape[0]}, items: {len(items)}, ids: {len(item_ids)}"
                )
                return invalid(delete=True, reason="integrity error")

            # 6. Check for NaN or Inf in embeddings (corruption detection)
            if not np.isfinite(embeddings).all():
                logger.warning(f"Collection '{self.name}': embeddings contain NaN or Inf values")
                return invalid(delete=True, reason="corrupted embeddings")
            return valid()

        cache_data = load_validated_pickle(
            self.cache_file,
            validators=[_check_version, _check_items_hash, _check_model, _check_embeddings],
            logger=logger,
            label=f"Collection '{self.name}'",
            missing_message=f"Collection '{self.name}': no cache file found",
        )
        if cache_data is None:
            return False

        # All checks passed - load data
        self._items = cache_data.get("items", {})
        self._embeddings = cache_data.get("embeddings")
        self._item_ids = cache_data.get("item_ids", [])
        self._items_hash = items_hash

        logger.debug(f"Collection '{self.name}': loaded {len(self._items)} items from cache")
        return True

    def _save_to_cache(self, items_hash: str) -> None:
        """Save collection to cache with full metadata for robust validation.

        Args:
            items_hash: Hash of items
        """
        cache_data = {
            "cache_version": self.CACHE_VERSION,
            "items_hash": items_hash,
            "model_name": self.embedding_service.model_name,
            "embedding_dimension": (
                self._embeddings.shape[1] if self._embeddings is not None else 0
            ),
            "item_count": len(self._items),
            "items": self._items,
            "embeddings": self._embeddings,
            "item_ids": self._item_ids,
        }

        if save_pickle_with_metadata(
            self.cache_file, cache_data, logger=logger, label=f"Collection '{self.name}'"
        ):
            cache_size = self.cache_file.stat().st_size / 1024
            logger.debug(
                f"Collection '{self.name}': saved {len(self._items)} items to cache "
                f"({cache_size:.1f} KB)"
            )

    async def initialize(self, items: List[CollectionItem]) -> None:
        """Initialize collection with items.

        Loads from cache if items haven't changed, otherwise computes embeddings.

        Args:
            items: List of collection items
        """
        if not items:
            logger.debug(f"Collection '{self.name}': initialized with empty items")
            self._items = {}
            self._embeddings = np.empty((0, self.embedding_service.dimension), dtype=np.float32)
            self._item_ids = []
            return

        # Calculate hash to check cache validity
        items_hash = self._calculate_items_hash(items)

        # Try to load from cache
        if self._load_from_cache(items_hash):
            return

        # Cache miss - compute embeddings
        logger.debug(f"Collection '{self.name}': computing embeddings for {len(items)} items")

        # Store items
        self._items = {item.id: item for item in items}
        self._item_ids = [item.id for item in items]

        # Generate embeddings in batch
        texts = [item.text for item in items]
        self._embeddings = await self.embedding_service.embed_batch(texts)
        self._items_hash = items_hash

        # Save to cache
        self._save_to_cache(items_hash)

        logger.debug(f"Collection '{self.name}': initialized with {len(items)} items")

    def initialize_sync(self, items: List[CollectionItem]) -> None:
        """Initialize collection with items (sync version).

        Args:
            items: List of collection items
        """
        if not items:
            logger.debug(f"Collection '{self.name}': initialized with empty items")
            self._items = {}
            self._embeddings = np.empty((0, self.embedding_service.dimension), dtype=np.float32)
            self._item_ids = []
            return

        # Calculate hash to check cache validity
        items_hash = self._calculate_items_hash(items)

        # Try to load from cache
        if self._load_from_cache(items_hash):
            return

        # Cache miss - compute embeddings
        logger.debug(f"Collection '{self.name}': computing embeddings for {len(items)} items")

        # Store items
        self._items = {item.id: item for item in items}
        self._item_ids = [item.id for item in items]

        # Generate embeddings in batch
        texts = [item.text for item in items]
        self._embeddings = self.embedding_service.embed_batch_sync(texts)
        self._items_hash = items_hash

        # Save to cache
        self._save_to_cache(items_hash)

        logger.debug(f"Collection '{self.name}': initialized with {len(items)} items")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        use_weighted_similarity: bool = False,
    ) -> List[Tuple[CollectionItem, float]]:
        """Search collection for similar items.

        Args:
            query: Query text
            top_k: Maximum number of results
            threshold: Minimum similarity score (0-1)
            use_weighted_similarity: If True, use weighted cosine similarity with
                key term boosting for better task classification. Default: False
                (maintains backward compatibility).

        Returns:
            List of (item, score) tuples, sorted by score descending
        """
        if not self.is_initialized or self._embeddings is None:
            logger.warning(f"Collection '{self.name}': not initialized, returning empty results")
            return []

        # Get query embedding
        query_embedding = await self.embedding_service.embed_text(query)

        # Calculate similarities
        if use_weighted_similarity:
            # Get corpus texts for weighted similarity
            corpus_texts = [self._items[pid].text for pid in self._item_ids]
            similarities = EmbeddingService.weighted_cosine_similarity(
                query_embedding, query, self._embeddings, corpus_texts
            )
        else:
            # Use standard cosine similarity (backward compatible)
            similarities = EmbeddingService.cosine_similarity_matrix(
                query_embedding, self._embeddings
            )

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                item_id = self._item_ids[idx]
                item = self._items[item_id]
                results.append((item, score))

        return results

    def search_sync(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        use_weighted_similarity: bool = False,
    ) -> List[Tuple[CollectionItem, float]]:
        """Search collection for similar items (sync version).

        Args:
            query: Query text
            top_k: Maximum number of results
            threshold: Minimum similarity score (0-1)
            use_weighted_similarity: If True, use weighted cosine similarity with
                key term boosting for better task classification. Default: False
                (maintains backward compatibility).

        Returns:
            List of (item, score) tuples, sorted by score descending
        """
        if not self.is_initialized or self._embeddings is None:
            logger.warning(f"Collection '{self.name}': not initialized, returning empty results")
            return []

        # Get query embedding
        query_embedding = self.embedding_service.embed_text_sync(query)

        # Calculate similarities
        if use_weighted_similarity:
            # Get corpus texts for weighted similarity
            corpus_texts = [self._items[pid].text for pid in self._item_ids]
            similarities = EmbeddingService.weighted_cosine_similarity(
                query_embedding, query, self._embeddings, corpus_texts
            )
        else:
            # Use standard cosine similarity (backward compatible)
            similarities = EmbeddingService.cosine_similarity_matrix(
                query_embedding, self._embeddings
            )

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                item_id = self._item_ids[idx]
                item = self._items[item_id]
                results.append((item, score))

        return results

    def get_item(self, item_id: str) -> Optional[CollectionItem]:
        """Get item by ID.

        Args:
            item_id: Item ID

        Returns:
            CollectionItem or None if not found
        """
        return self._items.get(item_id)

    def clear_cache(self) -> None:
        """Clear the cache file."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            logger.debug(f"Collection '{self.name}': cache cleared")
