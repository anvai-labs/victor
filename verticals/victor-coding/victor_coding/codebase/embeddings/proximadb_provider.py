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

"""ProximaDB embedding provider for Victor.

ProximaDB is a high-performance embedded vector + graph database:
- SST engine for fast vector operations (~5ms search latency)
- ORION engine for graph relationships (code symbol relationships)
- Native Rust performance with Python bindings
- Embedded mode (no separate server needed)

Install: pip install proximadb

Features:
- Unified vector + graph storage for semantic code knowledge
- Hardware-accelerated SIMD operations
- Real-time indexing with write-ahead logging
- Hilbert curve locality optimization
- Multiple distance metrics (cosine, euclidean, dot)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor_coding.codebase.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingSearchResult,
)
from victor_coding.codebase.embeddings.models import (
    BaseEmbeddingModel,
    EmbeddingModelConfig,
    create_embedding_model,
)

logger = logging.getLogger(__name__)

# Check if ProximaDB is available
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class ProximaDBProvider(BaseEmbeddingProvider):
    """ProximaDB embedding provider for semantic code knowledge.

    Uses ProximaDB's embedded mode for vector storage with pluggable embedding models.

    Features:
    - SST engine: Write-optimized real-time indexing (~5ms search)
    - ORION graph engine: Code symbol relationships
    - Hardware-accelerated distance computation
    - Persistent storage with WAL durability
    - Low memory footprint with mmap

    Advantages over LanceDB/ChromaDB:
    - Combined vector + graph storage (code relationships)
    - Better real-time write performance (SST engine)
    - Native Rust performance with zero-copy operations
    - Hilbert curve locality optimization (HELIX engine option)
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize ProximaDB provider.

        Args:
            config: Embedding configuration
        """
        super().__init__(config)

        if not HTTPX_AVAILABLE:
            raise ImportError("httpx not available. Install with: pip install httpx")

        self._db = None
        self._collection = None
        self._client: Optional[httpx.AsyncClient] = None
        self.embedding_model: Optional[BaseEmbeddingModel] = None
        self._server_url = "http://localhost:15678"  # Embedded mode port
        self._started = False

    async def initialize(self) -> None:
        """Initialize ProximaDB and load embedding model."""
        if self._initialized:
            return

        # Get embedding model configuration from EmbeddingConfig
        model_type = self.config.embedding_model_type
        model_name = self.config.embedding_model_name
        api_key = self.config.embedding_api_key
        dimension = self.config.extra_config.get("dimension", 384)
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

        logger.debug("Initializing ProximaDB provider")
        logger.debug("Vector Store: ProximaDB (SST engine)")
        logger.debug(f"Embedding Model: {model_name} ({model_type})")

        # Setup data directory
        persist_dir = self.config.persist_directory
        if persist_dir:
            persist_dir = Path(persist_dir).expanduser()
            persist_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Using persistent storage: {persist_dir}")
        else:
            from victor_coding.compat.settings import get_project_paths

            persist_dir = get_project_paths().global_embeddings_dir / "proximadb"
            persist_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Using default storage: {persist_dir}")

        self._data_dir = persist_dir

        # Check for external server or start embedded
        server_url = self.config.extra_config.get("server_url")
        if server_url:
            # Use external server
            self._server_url = server_url
            logger.debug(f"Using external ProximaDB server: {server_url}")
        else:
            # Try to start embedded mode
            await self._start_embedded_server()

        # Create HTTP client
        self._client = httpx.AsyncClient(
            base_url=self._server_url,
            timeout=60.0,
        )

        # Create or get collection
        collection_name = self.config.extra_config.get("collection_name", "code_embeddings")
        await self._ensure_collection(collection_name, dimension)

        self._initialized = True
        logger.debug("ProximaDB provider initialized!")

    async def _start_embedded_server(self) -> None:
        """Start ProximaDB in embedded mode if available."""
        try:
            # Try to import proximadb embedded module
            from proximadb import EmbeddedProximaDB, EmbeddedConfig

            config = EmbeddedConfig(
                data_dir=str(self._data_dir),
                rest_port=15678,
                grpc_port=15679,
                log_level="warn",
                vector_engine="SST",
                graph_engine="ORION",
            )

            self._db = EmbeddedProximaDB(config=config)
            await self._db.start()
            self._started = True
            self._server_url = self._db.rest_url
            logger.debug(f"Started embedded ProximaDB: {self._server_url}")

        except ImportError:
            # Fall back to checking for running server
            await self._check_external_server()

    async def _check_external_server(self) -> None:
        """Check if an external ProximaDB server is running."""
        # Try common ports
        for port in [15678, 5678]:
            url = f"http://localhost:{port}"
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{url}/health", timeout=2.0)
                    if response.status_code == 200:
                        self._server_url = url
                        logger.debug(f"Found running ProximaDB server: {url}")
                        return
            except Exception:
                pass

        raise RuntimeError(
            "ProximaDB not available. Either:\n"
            "1. Install proximadb package: pip install proximadb\n"
            "2. Start ProximaDB server: proximadb-server\n"
            "3. Provide server_url in extra_config"
        )

    async def _ensure_collection(self, name: str, dimension: int) -> None:
        """Ensure collection exists."""
        self._collection_name = name
        self._dimension = dimension

        try:
            response = await self._client.post(
                "/api/v1/collections",
                json={
                    "operation": 1,  # CREATE
                    "collection_id": name,
                    "collection_config": {
                        "name": name,
                        "dimension": dimension,
                        "distance_metric": 1,  # COSINE
                    },
                },
            )

            result = response.json()
            if result.get("success") or "already exists" in str(result):
                logger.debug(f"Collection ready: {name} (dim={dimension})")
            else:
                logger.debug(f"Collection creation result: {result}")

        except Exception as e:
            logger.warning(f"Could not create collection: {e}")

    def _to_sql_value(self, value: Any) -> Dict[str, Any]:
        """Convert Python value to ProximaDB SqlValue format."""
        if value is None:
            return {"null_value": 0}
        elif isinstance(value, bool):
            return {"bool_value": value}
        elif isinstance(value, int):
            return {"int64_value": value}
        elif isinstance(value, float):
            return {"number_value": value}
        elif isinstance(value, str):
            return {"string_value": value}
        elif isinstance(value, (list, tuple)):
            return {"array_value": {"values": [self._to_sql_value(v) for v in value]}}
        else:
            return {"string_value": str(value)}

    def _convert_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata dict to SqlValue format."""
        return {k: self._to_sql_value(v) for k, v in metadata.items()}

    def _extract_sql_value(self, sql_value: Dict[str, Any]) -> Any:
        """Extract Python value from SqlValue format."""
        if "string_value" in sql_value:
            return sql_value["string_value"]
        elif "int64_value" in sql_value:
            return sql_value["int64_value"]
        elif "number_value" in sql_value:
            return sql_value["number_value"]
        elif "bool_value" in sql_value:
            return sql_value["bool_value"]
        elif "null_value" in sql_value:
            return None
        elif "array_value" in sql_value:
            return [self._extract_sql_value(v) for v in sql_value["array_value"].get("values", [])]
        return sql_value

    def _extract_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Python values from SqlValue metadata."""
        return {k: self._extract_sql_value(v) for k, v in metadata.items()}

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

        # Generate embedding
        embedding = await self.embed_text(content)

        # Prepare vector with metadata
        vector_data = {
            "id": doc_id,
            "vector": embedding,
        }

        # Add content to metadata for retrieval
        full_metadata = {"content": content, **(metadata or {})}
        vector_data["metadata"] = self._convert_metadata(full_metadata)

        # Insert into collection
        response = await self._client.post(
            "/api/v1/vectors/batch",
            json={
                "collection_id": self._collection_name,
                "vectors": [vector_data],
            },
        )

        result = response.json()
        if not result.get("success"):
            logger.warning(f"Insert may have failed: {result}")

    async def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index multiple documents in batch.

        Args:
            documents: List of documents with id, content, metadata
        """
        if not self._initialized:
            await self.initialize()

        if not documents:
            return

        # Generate embeddings in batch
        contents = [doc["content"] for doc in documents]
        embeddings = await self.embed_batch(contents)

        # Prepare vectors for insertion
        vectors = []
        for doc, embedding in zip(documents, embeddings, strict=False):
            # Merge content into metadata for retrieval
            full_metadata = {"content": doc["content"], **doc.get("metadata", {})}

            vectors.append(
                {
                    "id": doc["id"],
                    "vector": embedding,
                    "metadata": self._convert_metadata(full_metadata),
                }
            )

        # Batch insert
        response = await self._client.post(
            "/api/v1/vectors/batch",
            json={
                "collection_id": self._collection_name,
                "vectors": vectors,
            },
        )

        result = response.json()
        if not result.get("success"):
            logger.warning(f"Batch insert may have failed: {result}")

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

        # Generate query embedding
        query_embedding = await self.embed_text(query)

        # Prepare search request
        search_query = {"vector": query_embedding}
        if filter_metadata:
            search_query["filters"] = self._convert_metadata(filter_metadata)

        # Execute search
        response = await self._client.post(
            "/api/v1/search",
            json={
                "collection_id": self._collection_name,
                "queries": [search_query],
                "top_k": limit,
            },
        )

        data = response.json()

        # Parse results (handle nested structure)
        results = []
        if data.get("success") and data.get("results"):
            inner = data["results"]
            if isinstance(inner, dict) and "results" in inner:
                raw_results = inner["results"] or []
            elif isinstance(inner, list):
                raw_results = inner
            else:
                raw_results = []

            for result in raw_results:
                # Extract metadata
                metadata = {}
                if "metadata" in result:
                    metadata = self._extract_metadata(result["metadata"])

                # Calculate similarity score (ProximaDB returns distance)
                distance = result.get("distance", result.get("score", 0.0))
                # Convert distance to similarity (0-1, higher is better)
                score = 1.0 / (1.0 + distance) if distance >= 0 else 0.0

                results.append(
                    EmbeddingSearchResult(
                        file_path=metadata.get("file_path", ""),
                        symbol_name=metadata.get("symbol_name"),
                        content=metadata.get("content", ""),
                        score=score,
                        line_number=metadata.get("line_number"),
                        metadata={k: v for k, v in metadata.items() if k not in ["content"]},
                    )
                )

        return results

    async def delete_document(self, doc_id: str) -> None:
        """Delete a document from index.

        Args:
            doc_id: Document identifier
        """
        if not self._initialized:
            await self.initialize()

        # ProximaDB delete endpoint (if available)
        try:
            await self._client.delete(
                f"/api/v1/vectors/{self._collection_name}/{doc_id}",
            )
        except Exception:
            # Delete may not be fully implemented yet
            pass

    async def delete_by_file(self, file_path: str) -> int:
        """Delete all documents from a specific file.

        Args:
            file_path: Relative file path to delete documents for

        Returns:
            Number of documents deleted
        """
        if not self._initialized:
            await self.initialize()

        # ProximaDB doesn't have a bulk delete by metadata yet
        # This would require listing documents with that file_path and deleting individually
        # For now, return 0 (documents remain but will be overwritten on re-index)
        return 0

    async def clear_index(self) -> None:
        """Clear entire index."""
        if not self._initialized:
            await self.initialize()

        # Delete and recreate collection
        try:
            await self._client.delete(
                f"/api/v1/collections/{self._collection_name}",
            )
        except Exception:
            pass

        # Recreate collection
        await self._ensure_collection(self._collection_name, self._dimension)
        logger.debug("Cleared index")

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with statistics
        """
        if not self._initialized:
            await self.initialize()

        count = 0
        try:
            response = await self._client.get(
                f"/api/v1/collections/{self._collection_name}",
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("collection"):
                    stats = data["collection"].get("stats", {})
                    count = stats.get("vector_count", 0)
        except Exception:
            pass

        return {
            "provider": "proximadb",
            "engine": "SST",
            "graph_engine": "ORION",
            "total_documents": count,
            "embedding_model_type": self.config.embedding_model_type,
            "embedding_model_name": self.config.embedding_model_name,
            "dimension": (self.embedding_model.get_dimension() if self.embedding_model else 384),
            "distance_metric": self.config.distance_metric,
            "collection_name": self._collection_name,
            "persist_directory": str(self._data_dir),
            "server_url": self._server_url,
        }

    async def close(self) -> None:
        """Clean up resources."""
        # Close embedding model
        if self.embedding_model:
            await self.embedding_model.close()
            self.embedding_model = None

        # Close HTTP client
        if self._client:
            await self._client.aclose()
            self._client = None

        # Stop embedded server if we started it
        if self._db and self._started:
            try:
                await self._db.stop()
            except Exception:
                pass
            self._db = None
            self._started = False

        self._initialized = False


# Convenience function to create ProximaDB provider with sensible defaults
def create_proximadb_provider(
    persist_directory: Optional[str] = None,
    collection_name: str = "code_embeddings",
    embedding_model: str = "all-MiniLM-L12-v2",
    server_url: Optional[str] = None,
) -> ProximaDBProvider:
    """Create a ProximaDB provider with optimized defaults for code search.

    Args:
        persist_directory: Where to store data (default: ~/.victor/embeddings/proximadb)
        collection_name: Name of the vector collection
        embedding_model: Sentence-transformer model name
        server_url: External ProximaDB server URL (optional)

    Returns:
        Configured ProximaDB provider

    Example:
        provider = create_proximadb_provider(
            persist_directory="~/.myapp/data",
            collection_name="my_codebase"
        )
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
            "dimension": 384,  # all-MiniLM-L12-v2 dimension
            "server_url": server_url,
        },
    )
    return ProximaDBProvider(config)
