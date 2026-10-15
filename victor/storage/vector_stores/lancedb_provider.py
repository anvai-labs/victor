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

"""LanceDB embedding provider for production deployments.

LanceDB is a modern, fast, embedded vector database perfect for:
- Production deployments (scales to billions of vectors)
- Fast similarity search with disk-based indices
- Serverless and embedded mode
- Zero-copy integration with Apache Arrow

Install: pip install lancedb

For embedding models:
- Ollama: pip install httpx (then: ollama pull qwen3-embedding:8b)
- Sentence-transformers: pip install sentence-transformers
- OpenAI: pip install openai (requires API key)
"""

from dataclasses import asdict
import importlib.util
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Provider registration must not load an optional native backend. Importing
# LanceDB can fail at the machine-code level on unsupported CPUs, even when
# the caller only needs an unrelated provider or the registry's public types.
LANCEDB_AVAILABLE = importlib.util.find_spec("lancedb") is not None

from victor.storage.vector_stores.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingSearchResult,
)
from victor.storage.vector_stores.models import (
    BaseEmbeddingModel,
    EmbeddingModelConfig,
    create_embedding_model,
)
from victor.storage.vector_stores._lancedb_compat import (
    lance_filter_predicate,
    lance_string_predicate,
)

if TYPE_CHECKING:
    from victor_codegraph import EmbeddingProfile

logger = logging.getLogger(__name__)

_PROFILE_METADATA_KEY = b"victor_codegraph.embedding_profile"


_SEARCH_METADATA_EXCLUDE_KEYS = {
    "vector",
    "content",
    "file_path",
    "symbol_name",
    "line_number",
}


class LanceDBProvider(BaseEmbeddingProvider):
    """LanceDB embedding provider.

    Uses LanceDB for vector storage with pluggable embedding models.

    Features:
    - Disk-based storage with mmap for efficiency
    - Scales to billions of vectors
    - Fast ANN search with disk-based indices
    - Automatic embedding generation
    - Metadata filtering
    - Support for multiple embedding models (Ollama, OpenAI, Sentence-transformers, Cohere)
    - Zero-copy reads via Apache Arrow

    Advantages:
    - Better performance for large datasets (>100k docs)
    - Lower memory footprint (disk-based)
    - Faster search with ANN indices
    - Production-ready scalability
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize LanceDB provider.

        Args:
            config: Embedding configuration
        """
        super().__init__(config)

        if not LANCEDB_AVAILABLE:
            raise ImportError("LanceDB not available. Install with: pip install lancedb")

        self.db = None
        self.table = None
        self.embedding_model: Optional[BaseEmbeddingModel] = None

    async def initialize(self) -> None:
        """Initialize LanceDB and load embedding model."""
        if self._initialized:
            return

        # Resolve the backend only when selected, before starting model resources.
        import lancedb

        # Get embedding model configuration from EmbeddingConfig
        model_type = self.config.embedding_model_type
        model_name = self.config.embedding_model_name
        api_key = self.config.embedding_api_key
        dimension = self.config.extra_config.get("dimension", 4096)
        batch_size = self.config.extra_config.get("batch_size", 16)

        # Create embedding model config
        embedding_config = EmbeddingModelConfig(
            model_type=model_type,
            model_name=model_name,
            dimension=dimension,
            api_key=api_key,
            batch_size=batch_size,
        )

        # Initialize embedding model
        self.embedding_model = create_embedding_model(embedding_config)
        await self.embedding_model.initialize()

        # Setup LanceDB
        persist_dir = self.config.persist_directory
        if persist_dir:
            persist_dir = Path(persist_dir).expanduser()
            persist_dir.mkdir(parents=True, exist_ok=True)
        else:
            # LanceDB requires a directory, use centralized path
            from victor.config.settings import get_project_paths

            persist_dir = get_project_paths().global_embeddings_dir / "lancedb"
            persist_dir.mkdir(parents=True, exist_ok=True)

        # Connect to LanceDB
        self.db = lancedb.connect(str(persist_dir))

        # Get or create table
        table_name = self.config.extra_config.get("table_name", "embeddings")

        # Check if table exists
        from victor.storage.vector_stores._lancedb_compat import get_table_names

        existing_tables = get_table_names(self.db)
        if table_name not in existing_tables:
            # Create empty table with schema
            # We'll add data later when indexing
            logger.info(f"Creating new LanceDB table: {table_name}")
        else:
            self.table = self.db.open_table(table_name)
            logger.debug(f"Opened existing LanceDB table: {table_name}")

        self._initialized = True

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not self._initialized:
            await self.initialize()

        return await self.embedding_model.embed_text(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()

        return await self.embedding_model.embed_batch(texts)

    def _collection_profile(self) -> "Optional[EmbeddingProfile]":
        """Validate the persisted table binding, including orphan vector rows.

        A profile is committed with the initial Arrow schema. Legacy collections
        are never relabeled or migrated in place. Storage mutations outside this
        adapter are unsupported; validate their row markers before use anyway.
        """
        if self.table is None:
            return None
        self.table.checkout_latest()
        schema = self.table.schema
        raw = (schema.metadata or {}).get(_PROFILE_METADATA_KEY)
        if raw is None:
            if "embedding_profile_id" in schema.names or "embedding_kind" in schema.names:
                raise ValueError("unbound semantic rows; rebuild in a fresh collection")
            return None
        from victor_codegraph import EmbeddingProfile

        try:
            if not isinstance(raw, bytes) or len(raw) > 8192:
                raise ValueError("invalid profile metadata")
            profile = EmbeddingProfile(**json.loads(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid collection profile; rebuild in a fresh collection") from exc
        if not {"embedding_profile_id", "embedding_kind"}.issubset(schema.names):
            raise ValueError("collection profile columns are missing")
        invalid = (
            f"embedding_profile_id IS NULL OR embedding_profile_id != '{profile.identity}' "
            "OR embedding_kind IS NULL OR embedding_kind != 'semantic'"
        )
        if self.table.count_rows(invalid):
            raise ValueError("mixed collection embedding profiles; rebuild in a fresh collection")
        return profile

    async def validate_embedding_profile(self, profile: "EmbeddingProfile") -> None:
        """Reject incompatible destinations before graph parsing/force deletion."""
        if not self._initialized:
            await self.initialize()
        existing = self._collection_profile()
        if self.table is not None and existing != profile:
            raise ValueError("incompatible collection profile; rebuild in a fresh collection")

    def _reject_unprofiled_write(self, metadata: List[Dict[str, Any]]) -> None:
        if self._collection_profile() is not None or any(
            "embedding_profile_id" in item or "embedding_kind" in item for item in metadata
        ):
            raise ValueError("profiled symbols require index_embedded_documents")

    def _validate_embedded_documents(
        self, documents: List[Dict[str, Any]]
    ) -> "Optional[EmbeddingProfile]":
        """Check the entire incoming batch before changing any persisted row."""
        existing = self._collection_profile()
        metadata = [doc.get("metadata", {}) for doc in documents]
        profiled = any(
            "embedding_profile_id" in item or "embedding_kind" in item for item in metadata
        )
        if not profiled:
            if existing is not None:
                raise ValueError("unprofiled write to a profiled collection")
            return None
        from victor_codegraph import EmbeddingProfile, validate_vectors

        try:
            profile = EmbeddingProfile(**metadata[0]["embedding_profile"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("semantic embedding profile is required") from exc
        for item in metadata:
            if (
                item.get("embedding_kind") != "semantic"
                or item.get("embedding_profile_id") != profile.identity
                or item.get("embedding_profile") != asdict(profile)
            ):
                raise ValueError("mixed or missing incoming embedding profile")
        if self.table is not None and existing != profile:
            raise ValueError("incompatible collection profile; rebuild in a fresh collection")
        ids = [doc["id"] for doc in documents]
        if any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != len(ids):
            raise ValueError("empty or duplicate canonical symbol identity")
        for start in range(0, len(documents), 512):
            batch = documents[start : start + 512]
            validate_vectors(
                (doc["vector"] for doc in batch), count=len(batch), dimension=profile.dimension
            )
        return profile

    async def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index a single document.

        Args:
            doc_id: Document identifier
            content: Document content
            metadata: Optional metadata
        """
        if not self._initialized:
            await self.initialize()

        self._reject_unprofiled_write([metadata or {}])

        # Generate embedding
        embedding = await self.embed_text(content)

        # Prepare document
        document = {
            "id": doc_id,
            "vector": embedding,
            "content": content,
            **(metadata or {}),
        }

        # Add to table (create if doesn't exist)
        table_name = self.config.extra_config.get("table_name", "embeddings")
        if self.table is None:
            self.table = self.db.create_table(table_name, data=[document])
        else:
            self.table.add([document])

    async def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index multiple documents in batch.

        Args:
            documents: List of documents with id, content, metadata
        """
        if not self._initialized:
            await self.initialize()

        if not documents:
            return

        self._reject_unprofiled_write([doc.get("metadata", {}) for doc in documents])

        # Generate embeddings in batch
        contents = [doc["content"] for doc in documents]
        embeddings = await self.embed_batch(contents)

        # Prepare documents for insertion
        lance_docs = []
        for doc, embedding in zip(documents, embeddings, strict=False):
            lance_doc = {
                "id": doc["id"],
                "vector": embedding,
                "content": doc["content"],
                **doc.get("metadata", {}),
            }
            lance_docs.append(lance_doc)

        # Add to table (create if doesn't exist)
        table_name = self.config.extra_config.get("table_name", "embeddings")
        if self.table is None:
            self.table = self.db.create_table(table_name, data=lance_docs)
        else:
            self.table.add(lance_docs)

    async def index_embedded_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents that already carry their vectors (upsert by id).

        Like :meth:`index_documents` but without the embedding step — used by
        the graph indexing pipeline, which batch-embeds through the shared
        EmbeddingService and re-embeds only stale symbols. Existing rows with
        the same ids are replaced so re-indexing never duplicates.

        Args:
            documents: Dicts with ``id``, ``content``, ``vector`` and optional
                ``metadata`` (flattened into columns, e.g. node_id, file_path,
                symbol_name, line_number, content_version).
        """
        if not self._initialized:
            await self.initialize()

        if not documents:
            return

        profile = self._validate_embedded_documents(documents)
        if any(
            {"id", "vector", "content"}.intersection(doc.get("metadata", {})) for doc in documents
        ):
            raise ValueError("metadata cannot override document identity, content or vector")
        lance_docs = []
        for doc in documents:
            lance_docs.append(
                {
                    "id": doc["id"],
                    "vector": list(doc["vector"]),
                    "content": doc.get("content", ""),
                    **doc.get("metadata", {}),
                }
            )

        table_name = self.config.extra_config.get("table_name", "embeddings")
        if profile is not None:
            import pyarrow as pa

            data = pa.Table.from_pylist(lance_docs)
            # Fix width/type and bind the complete model contract in the same
            # initial table commit. Deleting all rows does not erase this binding.
            schema = data.schema.set(
                data.schema.get_field_index("vector"),
                pa.field("vector", pa.list_(pa.float32(), profile.dimension)),
            ).with_metadata({_PROFILE_METADATA_KEY: json.dumps(asdict(profile)).encode()})
            data = data.cast(schema)
            from victor_codegraph import validate_vectors

            # Float32 storage must not turn finite values into infinities or zeros.
            for batch in data.to_batches(max_chunksize=512):
                validate_vectors(
                    batch.column("vector").to_pylist(),
                    count=batch.num_rows,
                    dimension=profile.dimension,
                )
            if self.table is None:
                self.table = self.db.create_table(table_name, data=data)
            else:
                # One commit: a failed upsert must not delete the previous rows.
                self.table.merge_insert(
                    "id"
                ).when_matched_update_all().when_not_matched_insert_all().execute(data)
            return
        if self.table is None:
            self.table = self.db.create_table(table_name, data=lance_docs)
            return
        # Upsert: drop stale rows for these ids, then add the fresh ones.
        ids = ", ".join("'" + str(d["id"]).replace("'", "''") + "'" for d in lance_docs)
        try:
            self.table.delete(f"id IN ({ids})")
        except (RuntimeError, ValueError) as e:
            logger.debug(f"Pre-upsert delete skipped: {e}")
        self.table.add(lance_docs)

    async def search_similar(
        self,
        query: str,
        limit: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EmbeddingSearchResult]:
        """Search for similar documents.

        Args:
            query: Search query
            limit: Maximum number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()

        if self.table is None:
            return []

        predicate = (
            lance_filter_predicate(filter_metadata, allowed_columns=self.table.schema.names)
            if filter_metadata is not None
            else ""
        )
        profile = self._collection_profile()
        if profile is not None:
            actual = self.embedding_model.profile if self.embedding_model is not None else None
            if actual != profile:
                raise ValueError("query encoder profile does not match collection profile")

        # Prefix is part of the shared contract, not a model-name heuristic.
        query_embedding = await self.embed_text((profile.query_prefix if profile else "") + query)
        if profile is not None:
            from victor_codegraph import validate_vectors

            query_embedding = validate_vectors(
                [query_embedding], count=1, dimension=profile.dimension
            )[0]
            import pyarrow as pa

            query_embedding = validate_vectors(
                [pa.array(query_embedding, type=pa.float32()).to_pylist()],
                count=1,
                dimension=profile.dimension,
            )[0]

        # Search in LanceDB
        results = self.table.search(query_embedding).limit(limit)

        # One predicate preserves conjunction on older SDKs. Eligibility is
        # applied before top-k; post-filtering can hide eligible neighbors.
        if predicate:
            results = results.where(predicate, prefilter=True)

        # Execute search
        search_results = []
        for result in results.to_list():
            # LanceDB returns distance (lower is better)
            # Convert to similarity score (0-1, higher is better)
            distance = result.get("_distance", 0.0)
            score = 1.0 / (1.0 + distance)  # Convert distance to similarity

            search_results.append(
                EmbeddingSearchResult(
                    file_path=result.get("file_path", ""),
                    symbol_name=result.get("symbol_name"),
                    content=result.get("content", ""),
                    score=score,
                    line_number=result.get("line_number"),
                    metadata={
                        k: v
                        for k, v in result.items()
                        if not k.startswith("_") and k not in _SEARCH_METADATA_EXCLUDE_KEYS
                    },
                )
            )

        return search_results

    async def delete_document(self, doc_id: str) -> None:
        """Delete a document from index.

        Args:
            doc_id: Document identifier
        """
        predicate = lance_string_predicate("id", doc_id)
        if not self._initialized:
            await self.initialize()

        if self.table is None:
            return

        self.table.delete(predicate)

    async def delete_by_file(self, file_path: str) -> int:
        """Delete all documents from a specific file.

        Used for incremental updates - when a file changes, we delete all
        its chunks and re-index.

        Args:
            file_path: Relative file path to delete documents for

        Returns:
            Number of documents deleted
        """
        predicate = lance_string_predicate("file_path", file_path)
        if not self._initialized:
            await self.initialize()

        if self.table is None:
            return 0

        # Count documents before deletion
        try:
            count_before = self.table.count_rows()
        except (AttributeError, RuntimeError, ValueError):
            count_before = 0

        # Delete documents with matching file_path
        # LanceDB uses SQL-like predicates
        self.table.delete(predicate)

        # Count documents after deletion
        try:
            count_after = self.table.count_rows()
        except (AttributeError, RuntimeError, ValueError):
            count_after = 0

        return count_before - count_after

    async def clear_index(self) -> None:
        """Clear entire index."""
        if not self._initialized:
            await self.initialize()

        # Drop and recreate table
        table_name = self.config.extra_config.get("table_name", "embeddings")
        from victor.storage.vector_stores._lancedb_compat import get_table_names

        if table_name in get_table_names(self.db):
            self.db.drop_table(table_name)

        self.table = None
        print("🗑️  Cleared index")

    async def clear_index_safe(self) -> bool:
        """Clear entire index with error handling.

        Returns:
            True if index was cleared successfully, False otherwise.
            Allows callers to handle failures gracefully rather than being
            left with an empty index when rebuild fails.
        """
        if not self._initialized:
            await self.initialize()

        table_name = self.config.extra_config.get("table_name", "embeddings")
        from victor.storage.vector_stores._lancedb_compat import get_table_names

        # Check if table exists before attempting to drop
        if table_name not in get_table_names(self.db):
            return False  # Nothing to clear

        # Store current count for logging
        old_count = 0
        if self.table is not None:
            try:
                old_count = self.table.count_rows()
            except (AttributeError, RuntimeError, ValueError):
                pass

        try:
            self.db.drop_table(table_name)
            self.table = None
            logger.info(f"[LanceDB] Cleared index ({old_count} documents)")
            return True
        except Exception as exc:
            logger.warning("[LanceDB] Failed to clear index: %s", exc)
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with statistics
        """
        if not self._initialized:
            await self.initialize()

        count = 0
        if self.table is not None:
            try:
                count = self.table.count_rows()
            except (AttributeError, RuntimeError, ValueError):
                count = 0

        return {
            "provider": "lancedb",
            "total_documents": count,
            "embedding_model_type": self.config.embedding_model_type,
            "embedding_model_name": self.config.embedding_model_name,
            "dimension": (self.embedding_model.get_dimension() if self.embedding_model else 4096),
            "distance_metric": self.config.distance_metric,
            "table_name": self.config.extra_config.get("table_name", "embeddings"),
            "persist_directory": self.config.persist_directory,
        }

    async def close(self) -> None:
        """Clean up resources."""
        if self.embedding_model:
            await self.embedding_model.close()
            self.embedding_model = None

        # LanceDB connections are lightweight, no explicit cleanup needed
        self.db = None
        self.table = None
        self._initialized = False
