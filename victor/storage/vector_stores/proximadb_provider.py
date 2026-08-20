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

"""ProximaDB embedding provider for Victor (Code Context Graph backend).

A real :class:`BaseEmbeddingProvider` over ProximaDB's vector engine via the
``proximadb_sdk`` embedded API. For the embedded/local single-repo case it runs
one ``EmbeddedProximaDB`` per repo with an in-process embedding model and
full-precision (fp32) vectors held in RAM — the ``EmbeddingMode::Memory``
equivalent — so semantic seed→expand scores neighbors inline.

Correlation: a document is keyed by its ``oid``. When indexing code symbols the
caller passes ``doc_id = graph/{repo}/node/{symbol_oid}`` — the same id used for
the ORION graph node — so the vector and the graph node are one entity and the
old ``graph_node.embedding_ref`` bridge is unnecessary.

ProximaDB is an optional dependency: if ``proximadb_sdk`` (or the embedded
server binary) is unavailable, ``initialize()`` raises a clear error and callers
fall back to the default LanceDB/SQLite stack.

- Embedded (``embedding_mode="memory"``, default): in-RAM fp32, drop-in local.
- Service (``embedding_mode="cold"`` / ``server_url=``): SQ8 multi-tenant. **WIP**,
  gated on ProximaDB TD-127 (secondary indexes) + TD-130/131 (graph bulk-load +
  REST v2 hybrid).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.storage.proxima_runtime import (
    ProximaEmbeddingMode,
    ProximaRepoConnection,
    ProximaUnavailableError,
    is_proxima_available,
)
from victor.storage.vector_stores.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingSearchResult,
)

logger = logging.getLogger(__name__)


class ProximaDBProvider(BaseEmbeddingProvider):
    """ProximaDB embedding provider for semantic code knowledge.

    Uses ProximaDB's embedded mode (``proximadb_sdk.embedded``) with an
    in-process embedding model so embedding generation, storage, and inline
    semantic scoring all happen in one in-RAM engine (EmbeddingMode::Memory).
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize ProximaDB provider.

        Args:
            config: Embedding configuration. Honored ``extra_config`` keys:
                ``collection_name`` (default ``code_embeddings``),
                ``dimension`` (default 384), ``embedding_mode``
                (``memory``/``cold``), ``server_url`` (service mode, WIP),
                ``binary_path`` (embedded server binary).
        """
        super().__init__(config)
        extra = config.extra_config or {}
        self._collection_name: str = extra.get("collection_name", "code_embeddings")
        self._dimension: int = int(extra.get("dimension", 384))
        self._embedding_mode = ProximaEmbeddingMode.coerce(extra.get("embedding_mode"))
        self._server_url: Optional[str] = extra.get("server_url")
        self._binary_path: Optional[str] = extra.get("binary_path")

        self._conn = None  # shared ProximaRepoConnection (per data_dir)
        self._collection = None  # EmbeddedCollection on the shared instance
        self._model = None  # proximadb_sdk embedding model (in-process)
        self._data_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        if self._initialized:
            return
        if not is_proxima_available():
            raise ImportError(
                "proximadb_sdk is not installed. Install it (pip install proximadb) "
                "or use a different vector_store (lancedb)."
            )
        if self._server_url:
            # Multi-tenant service path is not yet wired for the embedding
            # provider — gated on TD-127/130/131.
            raise ProximaUnavailableError(
                "ProximaDB service mode (server_url) for the embedding provider is "
                "WIP (gated on TD-127/130/131). Use embedded mode for local repos."
            )

        from proximadb_sdk import create_embedding_model

        # Resolve persistent data directory.
        persist_dir = self.config.persist_directory
        if persist_dir:
            self._data_dir = Path(persist_dir).expanduser()
        else:
            from victor.config.settings import get_project_paths

            self._data_dir = get_project_paths().global_embeddings_dir / "proximadb"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # In-process embedding model — ProximaDB embeds inline (Memory mode).
        self._model = create_embedding_model(
            model_type=self.config.embedding_model_type or "sentence-transformers",
            model_name=self.config.embedding_model_name,
        )
        try:
            self._dimension = int(self._model.get_dimension())
        except Exception:  # pragma: no cover - model-specific
            pass

        # Share ONE embedded instance per repo with ProximaGraphStore (keyed by
        # data_dir) so graph nodes and their vectors are co-located (TD-11/12).
        self._conn = await ProximaRepoConnection.acquire(
            self._data_dir, binary_path=self._binary_path
        )
        self._collection = await self._conn.get_or_create_collection(
            self._collection_name,
            dimension=self._dimension,
            distance_metric=self.config.distance_metric or "cosine",
            embedding_model=self._model,
        )

        self._initialized = True
        logger.debug(
            "ProximaDB embedding provider ready (collection=%s, dim=%s, mode=%s)",
            self._collection_name,
            self._dimension,
            self._embedding_mode.value,
        )

    # ------------------------------------------------------------------
    # Embedding generation (delegates to the in-process model)
    # ------------------------------------------------------------------
    async def embed_text(self, text: str) -> List[float]:
        if not self._initialized:
            await self.initialize()
        return await self._model.embed_async(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._initialized:
            await self.initialize()
        return await self._model.embed_batch_async(texts)

    # ------------------------------------------------------------------
    # Indexing — doc_id is the oid (graph/{repo}/node/{symbol_oid})
    # ------------------------------------------------------------------
    async def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self.index_documents([{"id": doc_id, "content": content, "metadata": metadata or {}}])

    async def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not self._initialized:
            await self.initialize()
        if not documents:
            return
        # EmbeddedCollection.insert_with_embedding embeds in-process and stores
        # the text under props["text"]; ProximaDB holds vectors in RAM.
        payload = [
            {
                "id": doc["id"],
                "text": doc["content"],
                "metadata": doc.get("metadata", {}),
            }
            for doc in documents
        ]
        await self._collection.insert_with_embedding(payload)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    async def search_similar(
        self,
        query: str,
        limit: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EmbeddingSearchResult]:
        if not self._initialized:
            await self.initialize()

        raw = await self._collection.search_text(
            query, top_k=limit, filters=filter_metadata or None
        )
        return [self._to_search_result(item) for item in (raw or [])]

    def _to_search_result(self, item: Dict[str, Any]) -> EmbeddingSearchResult:
        # ProximaDB returns id/score (or distance) plus props (incl. "text").
        props = item.get("props") or item.get("metadata") or {}
        if not isinstance(props, dict):
            props = {}
        content = props.get("text", item.get("text", ""))

        if "score" in item and item["score"] is not None:
            score = float(item["score"])
        else:
            distance = float(item.get("distance", 0.0) or 0.0)
            score = 1.0 / (1.0 + distance) if distance >= 0 else 0.0

        meta = {k: v for k, v in props.items() if k != "text"}
        return EmbeddingSearchResult(
            file_path=str(props.get("file_path", props.get("file", ""))),
            symbol_name=props.get("symbol_name") or props.get("name"),
            content=str(content),
            score=score,
            line_number=props.get("line_number") or props.get("line"),
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Deletes / maintenance
    # ------------------------------------------------------------------
    async def delete_document(self, doc_id: str) -> None:
        if not self._initialized:
            await self.initialize()
        try:
            await self._collection.delete([doc_id])
        except Exception as exc:  # pragma: no cover - depends on live server
            logger.debug("ProximaDB delete(%s) failed: %s", doc_id, exc)

    async def delete_by_file(self, file_path: str) -> int:
        if not self._initialized:
            await self.initialize()
        # Bulk delete-by-metadata lands with TD-127 (secondary indexes); until
        # then, stale rows are overwritten on re-index by their stable oid.
        logger.debug(
            "ProximaDB delete_by_file(%s) is a noop until TD-127; rows overwrite "
            "by oid on re-index.",
            file_path,
        )
        return 0

    async def clear_index(self) -> None:
        if not self._initialized:
            await self.initialize()
        try:
            await self._conn.embedded_db.delete_collection(self._collection_name)
            self._conn.forget_collection(self._collection_name)
        except Exception as exc:  # pragma: no cover
            logger.debug("delete_collection failed: %s", exc)
        self._collection = await self._conn.get_or_create_collection(
            self._collection_name,
            dimension=self._dimension,
            distance_metric=self.config.distance_metric or "cosine",
            embedding_model=self._model,
        )

    async def get_stats(self) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        count = 0
        try:
            count = await self._collection.count()
        except Exception:  # pragma: no cover
            pass
        return {
            "provider": "proximadb",
            "engine": "SST",
            "graph_engine": "ORION",
            "embedding_mode": self._embedding_mode.value,
            "total_documents": count,
            "embedding_model_type": self.config.embedding_model_type,
            "embedding_model_name": self.config.embedding_model_name,
            "dimension": self._dimension,
            "distance_metric": self.config.distance_metric,
            "collection_name": self._collection_name,
            "persist_directory": str(self._data_dir) if self._data_dir else None,
        }

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.release()
            self._conn = None
        self._collection = None
        self._model = None
        self._initialized = False


# Convenience function to create ProximaDB provider with sensible defaults
def create_proximadb_provider(
    persist_directory: Optional[str] = None,
    collection_name: str = "code_embeddings",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    server_url: Optional[str] = None,
    embedding_mode: str = "memory",
) -> ProximaDBProvider:
    """Create a ProximaDB provider with optimized defaults for code search.

    Args:
        persist_directory: Where to store data (default: ~/.victor/embeddings/proximadb)
        collection_name: Name of the vector collection
        embedding_model: Sentence-transformer model name (384-d bge-small default)
        server_url: External ProximaDB server URL (service mode, WIP)
        embedding_mode: ``memory`` (embedded, in-RAM fp32) or ``cold`` (service SQ8)

    Returns:
        Configured ProximaDB provider

    Example:
        provider = create_proximadb_provider(persist_directory="~/.myapp/data")
        await provider.initialize()
    """
    config = EmbeddingConfig(
        vector_store="proximadb",
        persist_directory=persist_directory,
        distance_metric="cosine",
        embedding_model_type="sentence-transformers",
        embedding_model_name=embedding_model,
        extra_config={
            "collection_name": collection_name,
            "dimension": 384,  # BAAI/bge-small-en-v1.5 dimension
            "server_url": server_url,
            "embedding_mode": embedding_mode,
        },
    )
    return ProximaDBProvider(config)
