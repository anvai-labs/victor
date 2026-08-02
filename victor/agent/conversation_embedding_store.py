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

"""Lazy LanceDB-based conversation embedding store for semantic retrieval.

Architecture (Lean + FK design):
- SQLite (ConversationStore): Authoritative source for messages, content, metadata
- LanceDB (ConversationEmbeddingStore): Vector index with message_id FK only

Lazy Embedding Strategy:
- Embeddings are NOT computed on message add (eager was wasteful)
- Embeddings are computed on-demand when search_similar() is called
- Un-embedded messages are batch-embedded from SQLite on first search
- Auto-compact after batch operations to reduce file proliferation

Schema (lean - no content duplication):
    - message_id: str (FK -> SQLite messages.id)
    - session_id: str (for filtering)
    - vector: list[float] (384-dim embedding)
    - timestamp: str (ISO format, for pruning)

Storage location: {project}/.victor/embeddings/conversations/

Usage:
    from victor.agent.conversation_embedding_store import ConversationEmbeddingStore
    from victor.storage.embeddings.service import EmbeddingService

    embedding_service = EmbeddingService.get_instance()
    store = ConversationEmbeddingStore(embedding_service, sqlite_db_path)
    await store.initialize()

    # Search triggers lazy embedding of un-embedded messages
    results = await store.search_similar(
        query="auth login implementation",
        session_id="session_xyz",
        limit=10,
    )
    # Returns [(message_id, similarity), ...] - fetch full content from SQLite
"""

import importlib.util
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Availability WITHOUT importing: ``import lancedb`` at module scope eagerly
# loads protobuf, lance_namespace_*, and pyarrow into the CLI cold-start path
# (this module is reached transitively at container bootstrap). find_spec()
# reports whether the packages are installed without executing their top-level
# code; the real import is deferred to ``initialize()``.
LANCEDB_AVAILABLE = (
    importlib.util.find_spec("lancedb") is not None
    and importlib.util.find_spec("pyarrow") is not None
)

if TYPE_CHECKING:
    from victor.storage.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class ConversationEmbeddingSearchResult:
    """Lean result from semantic search - just IDs and scores.

    Full message content should be fetched from SQLite using message_id.
    """

    message_id: str
    session_id: str
    similarity: float
    timestamp: Optional[datetime] = None

    def __repr__(self) -> str:
        return f"ConversationEmbeddingSearchResult(id={self.message_id}, sim={self.similarity:.3f})"


class ConversationEmbeddingStore:
    """Lazy LanceDB vector store for conversation embeddings.

    Key design principles:
    - Lean schema: Only message_id (FK), session_id, vector, timestamp
    - Lazy embedding: Compute on search, not on message add
    - SQLite is source of truth: Content lives there, not duplicated
    - Auto-compact: Reduce file proliferation after batch ops

    Benefits:
    - No write overhead during conversations
    - No content duplication (~50% smaller)
    - Batch embedding is more efficient than one-by-one
    - Compaction keeps LanceDB files manageable
    """

    # LanceDB table name
    TABLE_NAME = "conversation_vectors"

    # Maximum embeddings to keep (prunes oldest when exceeded)
    MAX_EMBEDDINGS = 10_000

    # Prune threshold (batch pruning for efficiency)
    PRUNE_BATCH_SIZE = 500

    # Compact after this many new embeddings
    COMPACT_THRESHOLD = 100

    def __init__(
        self,
        embedding_service: "EmbeddingService",
        sqlite_db_path: Optional[Path] = None,
        lancedb_path: Optional[Path] = None,
    ):
        """Initialize the conversation embedding store.

        Args:
            embedding_service: Shared EmbeddingService for generating embeddings
            sqlite_db_path: Path to SQLite database for lazy embedding
                          (Note: This is a separate embeddings database, NOT project.db or victor.db)
            lancedb_path: Path to LanceDB directory. Defaults to {project}/.victor/embeddings/conversations/
        """
        if not LANCEDB_AVAILABLE:
            raise ImportError("LanceDB not available. Install with: pip install lancedb pyarrow")

        self._embedding_service = embedding_service
        self._sqlite_db_path = sqlite_db_path
        self._lancedb_path = lancedb_path
        self._db: Optional[Any] = None  # LanceDB connection
        self._table: Optional[Any] = None  # LanceDB table
        self._initialized = False
        self._embeddings_added_since_compact = 0

    @property
    def dimension(self) -> int:
        """Get embedding dimension from the embedding service."""
        return self._embedding_service.dimension

    @property
    def is_initialized(self) -> bool:
        """Check if the store is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize LanceDB connection and table."""
        if self._initialized:
            return

        # Determine paths
        if self._lancedb_path is None or self._sqlite_db_path is None:
            from victor.config.settings import get_project_paths

            paths = get_project_paths()
            if self._lancedb_path is None:
                self._lancedb_path = paths.embeddings_dir / "conversations"
            if self._sqlite_db_path is None:
                self._sqlite_db_path = paths.project_db

        # Ensure directory exists
        self._lancedb_path.mkdir(parents=True, exist_ok=True)

        # Connect to LanceDB
        logger.info(f"[ConversationEmbeddingStore] Connecting to LanceDB at {self._lancedb_path}")
        import lancedb
        import pyarrow as pa  # noqa: F401 - required by LanceDB at runtime

        self._db = lancedb.connect(str(self._lancedb_path))

        # Open or create table
        from victor.storage.vector_stores._lancedb_compat import get_table_names

        existing_tables = get_table_names(self._db)
        if self.TABLE_NAME in existing_tables:
            self._table = self._db.open_table(self.TABLE_NAME)
            logger.info(f"[ConversationEmbeddingStore] Opened existing table '{self.TABLE_NAME}'")
        else:
            logger.info(
                f"[ConversationEmbeddingStore] Table '{self.TABLE_NAME}' will be created on first search"
            )

        self._initialized = True
        logger.info("[ConversationEmbeddingStore] Initialized (lazy embedding mode)")

    def _create_table_with_first_record(self, record: Dict[str, Any]) -> None:
        """Create the table with the first record."""
        self._table = self._db.create_table(self.TABLE_NAME, data=[record])
        logger.info(f"[ConversationEmbeddingStore] Created table '{self.TABLE_NAME}'")

    def _get_max_embedded_timestamp(self, session_id: Optional[str] = None) -> Optional[str]:
        """Get the maximum timestamp of embedded messages.

        Uses LanceDB's native Lance dataset to compute MAX(timestamp)
        via PyArrow without materializing the full table into pandas.
        """
        if self._table is None:
            return None

        try:
            import pyarrow.compute as pc

            # Use the underlying Lance dataset for efficient columnar access
            lance_ds = self._table.to_lance()
            scanner = lance_ds.scanner(columns=["timestamp"])
            if session_id:
                scanner = lance_ds.scanner(
                    columns=["timestamp"],
                    filter=f"session_id = '{session_id}'",
                )

            ts_table = scanner.to_table()
            if ts_table.num_rows == 0:
                return None

            max_ts = pc.max(ts_table.column("timestamp")).as_py()
            return str(max_ts) if max_ts else None

        except Exception as e:
            logger.warning(f"[ConversationEmbeddingStore] Failed to get max timestamp: {e}")
            return None

    def _get_unembedded_messages_from_sqlite(
        self,
        session_id: Optional[str] = None,
        after_timestamp: Optional[str] = None,
        min_content_length: int = 20,
        limit: int = 10_000,
    ) -> List[Dict[str, Any]]:
        """Fetch messages from SQLite newer than last embedded timestamp."""
        if self._sqlite_db_path is None or not self._sqlite_db_path.exists():
            return []

        try:
            with sqlite3.connect(self._sqlite_db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Single parameterized query with conditional WHERE clauses
                conditions = ["LENGTH(content) >= ?"]
                params = [min_content_length]

                if session_id:
                    conditions.append("session_id = ?")
                    params.append(session_id)

                if after_timestamp:
                    conditions.append("timestamp > ?")
                    params.append(after_timestamp)

                params.append(limit)

                query = f"""
                    SELECT id, session_id, content, timestamp
                    FROM messages
                    WHERE {' AND '.join(conditions)}
                    ORDER BY timestamp ASC
                    LIMIT ?
                """

                rows = conn.execute(query, params).fetchall()

                return [
                    {
                        "message_id": row["id"],
                        "session_id": row["session_id"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.warning(f"[ConversationEmbeddingStore] Failed to fetch from SQLite: {e}")
            return []

    async def _ensure_embeddings(self, session_id: Optional[str] = None) -> int:
        """Lazily embed any un-embedded messages from SQLite.

        Called automatically on search. Uses timestamp-based incremental sync:
        1. Get MAX(timestamp) from LanceDB (columnar O(1))
        2. Fetch messages from SQLite WHERE timestamp > max (O(log n) indexed)
        3. Batch embed and add to LanceDB

        Args:
            session_id: Optional session filter

        Returns:
            Number of new embeddings created
        """
        if not self._initialized:
            await self.initialize()

        # Get max embedded timestamp (efficient columnar scan)
        # Returns None if: (1) table doesn't exist, (2) table is empty
        max_timestamp = self._get_max_embedded_timestamp(session_id)

        # Edge case: LanceDB empty/uninitialized → full historical backfill
        is_full_backfill = max_timestamp is None
        if is_full_backfill:
            logger.info(
                "[ConversationEmbeddingStore] No existing embeddings found - "
                "will embed all historical messages from SQLite"
            )

        # Get un-embedded messages from SQLite (timestamp-based, not ID-based)
        # When max_timestamp=None, fetches ALL messages (full historical backfill)
        unembedded = self._get_unembedded_messages_from_sqlite(
            session_id=session_id,
            after_timestamp=max_timestamp,
        )

        if not unembedded:
            return 0

        # Log with context about whether this is incremental or full backfill
        if is_full_backfill and len(unembedded) > 100:
            logger.warning(
                f"[ConversationEmbeddingStore] Large historical backfill: "
                f"{len(unembedded)} messages (this may take a moment)"
            )
        else:
            logger.info(
                f"[ConversationEmbeddingStore] Lazy embedding {len(unembedded)} messages..."
            )

        start_time = time.perf_counter()

        # Batch embed content (truncate once, process efficiently)
        contents = [msg["content"][:2000] for msg in unembedded]
        embeddings = await self._embedding_service.embed_batch(contents)

        # Prepare lean records using list comprehension
        records = [
            {
                "message_id": msg["message_id"],
                "session_id": msg["session_id"],
                "vector": embedding.tolist(),
                "timestamp": msg["timestamp"],
            }
            for msg, embedding in zip(unembedded, embeddings, strict=False)
        ]

        # Add to LanceDB
        if self._table is None:
            self._create_table_with_first_record(records[0])
            if len(records) > 1:
                self._table.add(records[1:])
        else:
            self._table.add(records)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"[ConversationEmbeddingStore] Embedded {len(records)} messages "
            f"in {elapsed:.2f}s ({elapsed*1000/len(records):.1f}ms/msg)"
        )

        # Track for auto-compact
        self._embeddings_added_since_compact += len(records)

        # Auto-compact if threshold reached
        if self._embeddings_added_since_compact >= self.COMPACT_THRESHOLD:
            await self._auto_compact()

        # Prune if over limit
        await self._prune_old_embeddings()

        return len(records)

    async def search_similar(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        min_similarity: float = 0.3,
        exclude_message_ids: Optional[List[str]] = None,
    ) -> List[ConversationEmbeddingSearchResult]:
        """Search for semantically similar messages.

        Triggers lazy embedding of any un-embedded messages first.
        Returns message IDs + similarity - fetch full content from SQLite.

        Args:
            query: Query text to search for
            session_id: Optional session to scope search
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            exclude_message_ids: Message IDs to exclude

        Returns:
            List of ConversationEmbeddingSearchResult (message_id + similarity)
        """
        if not self._initialized:
            await self.initialize()

        # Lazy-embed any missing messages
        await self._ensure_embeddings(session_id)

        if self._table is None:
            logger.debug("[ConversationEmbeddingStore] No embeddings yet")
            return []

        start_time = time.perf_counter()

        # Embed query
        query_embedding = await self._embedding_service.embed_text(query[:2000])

        # Search in LanceDB
        search_query = self._table.search(query_embedding.tolist()).limit(limit * 2)

        # Apply session filter
        if session_id:
            search_query = search_query.where(f"session_id = '{session_id}'")

        # Execute search
        results = search_query.to_list()

        # Convert to result objects efficiently
        exclude_set = set(exclude_message_ids or [])
        search_results = []

        for result in results:
            message_id = result.get("message_id", "")
            if message_id in exclude_set:
                continue

            # Convert L2 distance to similarity
            similarity = 1.0 / (1.0 + result.get("_distance", 0.0))
            if similarity < min_similarity:
                continue

            # Parse timestamp only if present
            timestamp = None
            if timestamp_str := result.get("timestamp"):
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except (ValueError, TypeError):
                    pass

            search_results.append(
                ConversationEmbeddingSearchResult(
                    message_id=message_id,
                    session_id=result.get("session_id", ""),
                    similarity=similarity,
                    timestamp=timestamp,
                )
            )

            if len(search_results) >= limit:
                break

        elapsed = time.perf_counter() - start_time
        logger.debug(
            f"[ConversationEmbeddingStore] Search: "
            f"results={len(search_results)}, time={elapsed*1000:.1f}ms"
        )

        return search_results

    async def delete_session(self, session_id: str) -> int:
        """Delete all embeddings for a session.

        Args:
            session_id: Session ID to delete

        Returns:
            Number of embeddings deleted (estimated)
        """
        if not self._initialized:
            await self.initialize()

        if self._table is None:
            return 0

        try:
            count_before = self._table.count_rows()
            self._table.delete(f"session_id = '{session_id}'")
            count_after = self._table.count_rows()
            deleted = count_before - count_after

            logger.info(
                f"[ConversationEmbeddingStore] Deleted {deleted} embeddings for session {session_id}"
            )
            return deleted
        except Exception as e:
            logger.warning(f"[ConversationEmbeddingStore] Delete failed: {e}")
            return 0

    async def _prune_old_embeddings(self) -> int:
        """Prune oldest embeddings if over MAX_EMBEDDINGS limit."""
        if self._table is None:
            return 0

        try:
            count = self._table.count_rows()
            if count <= self.MAX_EMBEDDINGS + self.PRUNE_BATCH_SIZE:
                return 0

            delete_count = count - self.MAX_EMBEDDINGS

            # Get oldest timestamps via Lance scanner (avoids full table materialization)
            import pyarrow.compute as pc

            lance_ds = self._table.to_lance()
            tbl = lance_ds.scanner(columns=["message_id", "timestamp"]).to_table()
            indices = pc.sort_indices(tbl.column("timestamp"))
            oldest_indices = indices[:delete_count]
            oldest_ids = pc.take(tbl.column("message_id"), oldest_indices).to_pylist()

            if not oldest_ids:
                return 0

            # Delete in optimized batches
            deleted = 0
            for i in range(0, len(oldest_ids), 100):
                batch_ids = oldest_ids[i : i + 100]
                id_list = ", ".join(f"'{id}'" for id in batch_ids)
                self._table.delete(f"message_id IN ({id_list})")
                deleted += len(batch_ids)

            logger.info(f"[ConversationEmbeddingStore] Pruned {deleted} old embeddings")
            return deleted

        except Exception as e:
            logger.warning(f"[ConversationEmbeddingStore] Pruning failed: {e}")
            return 0

    async def _auto_compact(self) -> bool:
        """Auto-compact LanceDB to reduce file proliferation."""
        if self._table is None:
            return False

        try:
            logger.info("[ConversationEmbeddingStore] Auto-compacting...")
            self._table.optimize()
            self._embeddings_added_since_compact = 0
            logger.info("[ConversationEmbeddingStore] Compaction complete")
            return True
        except Exception as e:
            logger.warning(f"[ConversationEmbeddingStore] Compaction failed: {e}")
            return False

    async def compact(self) -> bool:
        """Manually trigger compaction."""
        return await self._auto_compact()

    async def rebuild(self, session_id: Optional[str] = None) -> int:
        """Eagerly rebuild embeddings for all un-embedded messages.

        This is used by CLI `victor embeddings --rebuild --conversation` to
        immediately generate embeddings rather than waiting for next search.

        Args:
            session_id: Optional session filter

        Returns:
            Number of embeddings created
        """
        logger.info("[ConversationEmbeddingStore] Starting eager rebuild...")
        count = await self._ensure_embeddings(session_id)
        logger.info(f"[ConversationEmbeddingStore] Rebuild complete: {count} embeddings created")
        return count

    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        if not self._initialized:
            await self.initialize()

        count = 0
        if self._table is not None:
            try:
                count = self._table.count_rows()
            except (AttributeError, RuntimeError, ValueError):
                pass

        return {
            "store": "conversation_embedding_store",
            "backend": "lancedb",
            "mode": "lazy",
            "total_embeddings": count,
            "max_embeddings": self.MAX_EMBEDDINGS,
            "usage_pct": ((count / self.MAX_EMBEDDINGS * 100) if self.MAX_EMBEDDINGS > 0 else 0),
            "embedding_dimension": self.dimension,
            "embedding_model": self._embedding_service.model_name,
            "lancedb_path": str(self._lancedb_path),
            "sqlite_path": str(self._sqlite_db_path),
            "pending_compact": self._embeddings_added_since_compact,
        }

    async def close(self) -> None:
        """Clean up resources."""
        # Compact before closing if we have pending changes
        if self._embeddings_added_since_compact > 0:
            await self._auto_compact()

        self._table = None
        self._db = None
        self._initialized = False
        logger.info("[ConversationEmbeddingStore] Closed")


# Module-level singleton instance
_embedding_store: Optional[ConversationEmbeddingStore] = None


async def get_conversation_embedding_store(
    embedding_service: Optional["EmbeddingService"] = None,
) -> ConversationEmbeddingStore:
    """Get or create the global ConversationEmbeddingStore instance.

    Args:
        embedding_service: EmbeddingService to use. If None, uses the singleton.

    Returns:
        Initialized ConversationEmbeddingStore instance
    """
    global _embedding_store

    if _embedding_store is None:
        if embedding_service is None:
            from victor.storage.embeddings.service import get_embedding_service

            embedding_service = get_embedding_service()

        _embedding_store = ConversationEmbeddingStore(embedding_service)
        await _embedding_store.initialize()

    return _embedding_store
