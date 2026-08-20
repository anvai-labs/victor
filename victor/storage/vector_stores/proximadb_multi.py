# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""ProximaDB multi-model provider for Victor code intelligence.

This provider keeps Victor's existing ``BaseEmbeddingProvider`` contract for
vector search, and adds multi-model code indexing on top of ProximaDB:

- Vector collection: semantic code chunks
- Document collection: full-file snapshots with metadata
- Graph collection: modules, classes, functions, and relationships
- Metrics collection: time-series style code metrics snapshots

The current ProximaDB SDK exposes first-class vector and graph primitives, but
not the same level of document/time-series helpers yet. Document and metrics
records are therefore stored in dedicated collections as typed records with
metadata, which keeps the layout forward-compatible while staying usable today.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from victor.storage.vector_stores.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingSearchResult,
)
from victor.core.utils.capability_loader import load_tree_sitter_get_parser
from victor.storage.vector_stores.code_chunking import (
    CodeChunk,
    create_code_chunker,
    normalize_code_chunking_strategy,
)
from victor.storage.vector_stores.change_impact import ChangeImpactAccumulator
from victor.storage.vector_stores.issue_localization import IssueLocalizationAccumulator
from victor.storage.vector_stores.models import (
    BaseEmbeddingModel,
    EmbeddingModelConfig,
    create_embedding_model,
)

# Language registry and tree-sitter queries are provided by external vertical
# packages (e.g., victor-coding) via CapabilityRegistry. We use Any for type
# annotations and _NullLanguageRegistry as fallback when no provider is installed.

try:
    from tree_sitter import Query, QueryCursor

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

try:
    from proximadb_sdk.graph import ProximaDBGraph
    from proximadb_sdk.models import (
        CollectionConfig,
        DistanceMetric,
        IndexingAlgorithm,
        StorageEngine,
        VectorRecord,
    )
    from proximadb_sdk.unified_client import ProximaDBClient

    PROXIMADB_SDK_AVAILABLE = True
except Exception:
    ProximaDBGraph = None  # type: ignore[assignment]
    CollectionConfig = None  # type: ignore[assignment]
    DistanceMetric = None  # type: ignore[assignment]
    IndexingAlgorithm = None  # type: ignore[assignment]
    StorageEngine = None  # type: ignore[assignment]
    VectorRecord = None  # type: ignore[assignment]
    ProximaDBClient = None  # type: ignore[assignment]
    PROXIMADB_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"[^a-zA-Z0-9_]+")
_CONTROL_FLOW_KEYWORDS = (
    "if",
    "elif",
    "else",
    "for",
    "while",
    "case",
    "except",
    "catch",
    "&&",
    "||",
    "?",
)
_IMPORT_QUERIES: Dict[str, str] = {
    "javascript": """
        (import_statement source: (string) @source)
        (call_expression
            function: (identifier) @_fn
            arguments: (arguments (string) @source)
            (#eq? @_fn "require"))
    """,
    "typescript": """
        (import_statement source: (string) @source)
        (call_expression
            function: (identifier) @_fn
            arguments: (arguments (string) @source)
            (#eq? @_fn "require"))
    """,
    "rust": """
        (use_declaration argument: (_) @source)
    """,
    "go": """
        (import_spec path: (interpreted_string_literal) @source)
    """,
    "java": """
        (import_declaration (scoped_identifier) @source)
    """,
}


@dataclass
class _FallbackVectorRecord:
    """Minimal record shape used when the SDK model import is unavailable."""

    id: str
    vector: List[float]
    source: str
    metadata: Dict[str, Any]


@dataclass
class _GraphSymbol:
    """Intermediate symbol representation used for graph/document indexing."""

    name: str
    symbol_type: str
    line_start: int
    line_end: int
    parent_symbol: Optional[str] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        if self.parent_symbol:
            return f"{self.parent_symbol}.{self.name}"
        return self.name


@dataclass
class _GraphEdge:
    """Intermediate graph edge."""

    source: str
    target: str
    edge_type: str
    line_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _GraphSnapshot:
    """All structural information extracted from a file."""

    symbols: List[_GraphSymbol]
    call_edges: List[_GraphEdge]
    inheritance_edges: List[_GraphEdge]
    implements_edges: List[_GraphEdge]
    composition_edges: List[_GraphEdge]
    imports: List[str]


@dataclass
class _TreeSitterAnalysis:
    """Parse-once tree-sitter artifacts used by graph extraction."""

    parser: Any
    tree: Any
    queries: Any


class _NullLanguageRegistry:
    """Fallback registry used when victor-coding is not installed."""

    def __init__(self) -> None:
        self._plugins: Dict[str, Any] = {}

    def discover_plugins(self) -> None:
        return None

    def get(self, language: str) -> Any:
        del language
        return None

    def detect_from_content(self, content: str, filename: Optional[str] = None) -> Optional[str]:
        del content, filename
        return None

    def detect_language(self, path: Path) -> Optional[str]:
        del path
        return None


class ProximaDBMultiModelProvider(BaseEmbeddingProvider):
    """Victor provider backed by ProximaDB's vector and graph primitives."""

    def __init__(
        self,
        config: EmbeddingConfig,
        client: Optional[Any] = None,
        language_registry: Optional[Any] = None,
    ) -> None:
        super().__init__(config)

        if not PROXIMADB_SDK_AVAILABLE and client is None:
            raise ImportError(
                "proximadb_sdk not available. Install with: pip install proximadb-python"
            )

        self.embedding_model: Optional[BaseEmbeddingModel] = None
        self._client = client
        self._graph_api: Optional[Any] = None
        if language_registry is not None:
            self._language_registry = language_registry
        else:
            # Discover language registry via CapabilityRegistry (provided by
            # external verticals like victor-coding at bootstrap time)
            discovered = self._discover_language_registry()
            self._language_registry = (
                discovered if discovered is not None else _NullLanguageRegistry()
            )

        if not getattr(self._language_registry, "_plugins", {}):
            self._language_registry.discover_plugins()

        self._dimension = int(config.extra_config.get("dimension", 384))
        self._batch_size = int(config.extra_config.get("batch_size", 16))
        self._chunk_size = int(config.extra_config.get("chunk_size", 500))
        self._chunk_overlap = int(config.extra_config.get("chunk_overlap", 50))
        self._code_chunking_strategy = normalize_code_chunking_strategy(
            str(config.extra_config.get("code_chunking_strategy", "victor_codegraph"))
        )
        self._code_chunker = create_code_chunker(
            self._code_chunking_strategy,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        self._graph_enabled = bool(config.extra_config.get("graph_enabled", True))
        self._document_enabled = bool(config.extra_config.get("document_enabled", True))
        self._metrics_enabled = bool(config.extra_config.get("metrics_enabled", True))
        self._query_scan_limit = int(config.extra_config.get("query_scan_limit", 500))

        workspace = config.extra_config.get("workspace")
        if not workspace:
            workspace = Path.cwd().name or "victor"
        self._workspace = self._normalize_collection_name(str(workspace))
        self._vector_collection = self._collection_name("vector_collection", "vectors")
        self._document_collection = self._collection_name("document_collection", "documents")
        self._metrics_collection = self._collection_name("metrics_collection", "metrics")
        self._graph_collection = self._collection_name("graph_collection", "graph")

    @staticmethod
    def _discover_language_registry() -> Optional[Any]:
        """Discover a language registry from CapabilityRegistry.

        External vertical packages (e.g., victor-coding) register their
        language registry as a capability at bootstrap. Returns None if
        no provider is installed.
        """
        try:
            from victor.core.capability_registry import CapabilityRegistry

            registry = CapabilityRegistry.get_instance()
            # Look for any registered provider with a get/detect_language interface
            for proto_type, (provider, _status) in registry._providers.items():
                if hasattr(provider, "get") and hasattr(provider, "detect_language"):
                    return provider
        except Exception:
            pass
        return None

    async def initialize(self) -> None:
        """Initialize the embedding model and required ProximaDB collections."""
        if self._initialized:
            return

        model_config = EmbeddingModelConfig(
            model_type=self.config.embedding_model_type,
            model_name=self.config.embedding_model_name,
            dimension=self._dimension,
            api_key=self.config.embedding_api_key,
            batch_size=self._batch_size,
        )
        self.embedding_model = create_embedding_model(model_config)
        await self.embedding_model.initialize()

        if self._client is None:
            server_url = self.config.extra_config.get("server_url", "http://localhost:5678")
            client_kwargs = {
                "url": server_url,
                "api_key": self.config.extra_config.get("api_key"),
                "pool_size": int(self.config.extra_config.get("pool_size", 10)),
                "pool_maxsize": int(self.config.extra_config.get("pool_maxsize", 20)),
                "verify_ssl": bool(self.config.extra_config.get("verify_ssl", True)),
                "enable_http2": bool(self.config.extra_config.get("enable_http2", True)),
            }
            self._client = ProximaDBClient(**client_kwargs)

        self._ensure_collection(self._vector_collection)
        if self._document_enabled:
            self._ensure_collection(self._document_collection)
        if self._metrics_enabled:
            self._ensure_collection(self._metrics_collection)
        if self._graph_enabled:
            self._ensure_graph()
            self._graph_api = self._build_graph_api()

        self._initialized = True

    async def embed_text(self, text: str) -> List[float]:
        """Generate an embedding vector for a single text."""
        if not self._initialized:
            await self.initialize()
        if self.embedding_model is None:
            raise RuntimeError("Embedding model is not initialized")
        return await self.embedding_model.embed_text(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for multiple texts."""
        if not self._initialized:
            await self.initialize()
        if self.embedding_model is None:
            raise RuntimeError("Embedding model is not initialized")
        return await self.embedding_model.embed_batch(texts)

    async def index_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """Index a single vector-search document in the vector collection."""
        if not self._initialized:
            await self.initialize()

        vector = await self.embed_text(content)
        record = self._vector_record(
            record_id=doc_id,
            vector=vector,
            source=content,
            metadata=metadata,
        )
        self._client.insert_vectors(self._vector_collection, records=[record])

    async def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index multiple vector-search documents in batch."""
        if not self._initialized:
            await self.initialize()
        if not documents:
            return

        vectors = await self.embed_batch([doc["content"] for doc in documents])
        records = [
            self._vector_record(
                record_id=doc["id"],
                vector=vector,
                source=doc["content"],
                metadata=doc.get("metadata", {}),
            )
            for doc, vector in zip(documents, vectors, strict=False)
        ]
        self._client.insert_vectors(self._vector_collection, records=records)

    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EmbeddingSearchResult]:
        """Search semantic code chunks stored in ProximaDB's vector engine."""
        if not self._initialized:
            await self.initialize()

        query_vector = await self.embed_text(query)
        hits = self._client.search(
            self._vector_collection,
            vector=query_vector,
            top_k=limit,
            metadata_filter=filter_metadata,
        )

        results: List[EmbeddingSearchResult] = []
        for hit in hits:
            metadata = dict(getattr(hit, "metadata", {}) or {})
            results.append(
                EmbeddingSearchResult(
                    file_path=metadata.get("file_path", ""),
                    symbol_name=metadata.get("symbol_name"),
                    content=getattr(hit, "source", None) or metadata.get("content", ""),
                    score=float(getattr(hit, "score", 0.0)),
                    line_number=metadata.get("start_line") or metadata.get("line_number"),
                    metadata={
                        key: value
                        for key, value in metadata.items()
                        if key not in {"content", "file_path", "symbol_name", "line_number"}
                    },
                )
            )
        return results

    async def delete_document(self, doc_id: str) -> None:
        """Delete a single vector document by ID."""
        if not self._initialized:
            await self.initialize()
        self._client.delete_vectors(self._vector_collection, [doc_id])

    async def delete_by_file(self, file_path: str) -> int:
        """Delete vector, document, and metric records that belong to a file."""
        if not self._initialized:
            await self.initialize()

        deleted = 0
        for collection in self._iter_record_collections():
            ids = self._search_ids_for_file(collection, file_path)
            if ids:
                self._client.delete_vectors(collection, ids)
                deleted += len(ids)

        # Graph APIs currently expose graph-wide delete, but not file-scoped delete.
        # Re-indexing with stable node IDs updates existing nodes/edges; removals remain
        # a known limitation until file-scoped graph delete lands in the SDK/server.
        return deleted

    async def clear_index(self) -> None:
        """Drop and recreate all ProximaDB collections used by this provider."""
        if not self._initialized:
            await self.initialize()

        for collection in self._iter_record_collections():
            try:
                self._client.delete_collection(collection)
            except Exception:
                logger.debug("Failed to delete collection %s during clear_index", collection)
            self._ensure_collection(collection)

        if self._graph_enabled:
            try:
                self._client.delete_graph(self._graph_collection)
            except Exception:
                logger.debug(
                    "Failed to delete graph %s during clear_index",
                    self._graph_collection,
                )
            self._ensure_graph()
            self._graph_api = self._build_graph_api()

    async def get_stats(self) -> Dict[str, Any]:
        """Return provider configuration and best-effort collection statistics."""
        if not self._initialized:
            await self.initialize()

        stats: Dict[str, Any] = {
            "provider": "proximadb_multi",
            "workspace": self._workspace,
            "vector_collection": self._vector_collection,
            "document_collection": self._document_collection,
            "graph_collection": self._graph_collection,
            "metrics_collection": self._metrics_collection,
            "dimension": self._dimension,
            "embedding_model_type": self.config.embedding_model_type,
            "embedding_model_name": self.config.embedding_model_name,
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "graph_enabled": self._graph_enabled,
            "document_enabled": self._document_enabled,
            "metrics_enabled": self._metrics_enabled,
        }

        try:
            collection_stats = {}
            for collection in self._iter_record_collections():
                info = self._client.get_collection(collection)
                collection_stats[collection] = info
            stats["collections"] = collection_stats
        except Exception:
            logger.debug("Collection stats unavailable for ProximaDB multi-model provider")

        try:
            stats["graph"] = self._client.get_graph(self._graph_collection)
        except Exception:
            logger.debug("Graph stats unavailable for ProximaDB multi-model provider")

        return stats

    async def close(self) -> None:
        """Release local resources."""
        if self.embedding_model is not None:
            await self.embedding_model.close()
            self.embedding_model = None
        self._initialized = False

    async def index_code_file(
        self,
        file_path: str,
        content: str,
        language: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Index a code file across vector, document, graph, and metrics models."""
        if not self._initialized:
            await self.initialize()

        language_name = self._resolve_language(file_path, content, language)
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        indexed_at = datetime.now(timezone.utc)
        base_metadata: Dict[str, Any] = {
            "workspace": self._workspace,
            "file_path": file_path,
            "language": language_name,
            "file_hash": file_hash,
            "indexed_at": indexed_at.isoformat(),
            "chunking_strategy": self._code_chunking_strategy,
        }
        if metadata:
            base_metadata.update(metadata)

        await self.delete_by_file(file_path)
        tree_analysis = self._build_tree_sitter_analysis(file_path, content, language_name)
        graph_snapshot = self._extract_graph_snapshot(
            file_path,
            content,
            language_name,
            tree_analysis=tree_analysis,
        )

        chunks = self._chunk_code(
            file_path,
            content,
            language_name,
        )
        if chunks:
            documents = []
            for index, chunk in enumerate(chunks):
                chunk_metadata = {
                    **base_metadata,
                    **chunk.metadata,
                    "record_type": "code_chunk",
                    "chunk_index": index,
                    "chunk_type": chunk.chunk_type,
                    "symbol_name": chunk.symbol_name,
                    "parent_symbol": chunk.parent_symbol,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_id": chunk.chunk_id,
                    "symbol_id": chunk.symbol_id,
                }
                documents.append(
                    {
                        "id": chunk.chunk_id or self._chunk_record_id(file_path, file_hash, index),
                        "content": chunk.content,
                        "metadata": chunk_metadata,
                    }
                )
            await self.index_documents(documents)

        if self._document_enabled:
            self._store_document_snapshot(file_path, content, language_name, base_metadata)

        graph_counts = {
            "functions": 0,
            "classes": 0,
            "calls": 0,
            "imports": len(graph_snapshot.imports),
            "inherits": len(graph_snapshot.inheritance_edges),
            "implements": len(graph_snapshot.implements_edges),
            "composition": len(graph_snapshot.composition_edges),
        }
        if self._graph_enabled:
            graph_counts = self._store_graph_snapshot(
                file_path=file_path,
                language=language_name,
                base_metadata=base_metadata,
                snapshot=graph_snapshot,
            )

        metric_count = 0
        if self._metrics_enabled:
            metric_count = self._store_metrics_snapshot(
                file_path=file_path,
                content=content,
                language=language_name,
                symbols=graph_snapshot.symbols,
                recorded_at=indexed_at,
                base_metadata=base_metadata,
            )

        return {
            "workspace": self._workspace,
            "language": language_name,
            "chunking_strategy": self._code_chunking_strategy,
            "vectors": len(chunks),
            "document": self._document_enabled,
            "graph": graph_counts,
            "timeseries": metric_count,
        }

    async def hybrid_search(
        self,
        query: str,
        graph_query: Optional[str] = None,
        document_filter: Optional[Dict[str, Any]] = None,
        time_range: Optional[tuple[datetime, datetime]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Run a best-effort hybrid query across vector, graph, documents, and metrics."""
        if not self._initialized:
            await self.initialize()

        combined: Dict[str, Dict[str, Any]] = {}

        for result in await self.search_similar(query, limit=max(top_k * 2, top_k)):
            key = self._hybrid_key(
                result.metadata.get("file_path") or result.file_path,
                result.metadata.get("symbol_name") or result.symbol_name,
                result.line_number,
            )
            combined[key] = {
                "id": key,
                "file_path": result.file_path,
                "symbol_name": result.symbol_name,
                "content": result.content,
                "score": result.score,
                "sources": ["vector"],
                "metadata": {**result.metadata, "file_path": result.file_path},
            }

        if graph_query:
            for row in self._search_graph(graph_query, top_k=max(top_k * 2, top_k)):
                self._merge_hybrid_result(combined, row, source="graph", weight=0.35)

        if self._document_enabled:
            for row in self._search_records(
                collection=self._document_collection,
                filter_metadata=document_filter,
                limit=max(top_k * 2, top_k),
            ):
                self._merge_hybrid_result(combined, row, source="document", weight=0.15)

        if self._metrics_enabled and time_range is not None:
            metric_rows = self._search_records(
                collection=self._metrics_collection,
                filter_metadata=document_filter,
                limit=self._query_scan_limit,
            )
            for row in metric_rows:
                metadata = dict(getattr(row, "metadata", {}) or {})
                recorded_at = metadata.get("recorded_at")
                if not self._within_time_range(recorded_at, time_range):
                    continue
                self._merge_hybrid_result(combined, row, source="metrics", weight=0.1)

        ranked = list(combined.values())
        if document_filter:
            ranked = [
                row
                for row in ranked
                if self._metadata_matches(row.get("metadata", {}), document_filter)
            ]
        ranked.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        return ranked[:top_k]

    async def get_code_metrics(self, file_path: str, days: int = 30) -> List[Dict[str, Any]]:
        """Return metric snapshots for a file from the compatibility metrics collection."""
        if not self._initialized:
            await self.initialize()

        now = datetime.now(timezone.utc)
        earliest = now - timedelta(days=days)
        results = []
        for row in self._search_records(
            collection=self._metrics_collection,
            filter_metadata={"file_path": file_path, "record_type": "metric"},
            limit=self._query_scan_limit,
        ):
            metadata = dict(getattr(row, "metadata", {}) or {})
            recorded_at = metadata.get("recorded_at")
            if not self._within_time_range(recorded_at, (earliest, now)):
                continue
            results.append(
                {
                    "metric_name": metadata.get("metric_name"),
                    "metric_value": metadata.get("metric_value"),
                    "recorded_at": recorded_at,
                    "language": metadata.get("language"),
                    "file_path": metadata.get("file_path"),
                }
            )
        results.sort(key=lambda row: row["recorded_at"])
        return results

    async def find_callers(
        self,
        function_name: str,
        file_path: Optional[str] = None,
        edge_type: str = "CALLS",
        max_depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """Find graph callers for a function using ProximaDB's graph helper."""
        if not self._initialized:
            await self.initialize()

        graph_api = self._graph_api or self._build_graph_api()
        if graph_api is None or not hasattr(graph_api, "find_callers"):
            return []

        properties: Dict[str, Any] = {"name": function_name}
        if file_path:
            properties["file_path"] = file_path

        nodes = self._query_graph_nodes(labels=["Function"], properties=properties)
        callers: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for node in nodes:
            try:
                matches = graph_api.find_callers(
                    node["id"],
                    edge_type=edge_type,
                    max_depth=max_depth,
                )
            except Exception:
                logger.debug("Graph caller lookup failed for %s", node.get("id"))
                continue

            for caller in matches:
                metadata = dict(getattr(caller, "properties", {}) or {})
                caller_id = getattr(caller, "id", None) or metadata.get("id")
                if not caller_id or caller_id in seen_ids:
                    continue
                seen_ids.add(caller_id)
                callers.append(
                    {
                        "id": caller_id,
                        "name": metadata.get("qualified_name") or metadata.get("name"),
                        "file_path": metadata.get("file_path"),
                        "line_start": metadata.get("line_start"),
                        "line_end": metadata.get("line_end"),
                        "labels": list(getattr(caller, "labels", []) or []),
                        "metadata": metadata,
                    }
                )

        callers.sort(
            key=lambda row: (
                row.get("file_path") or "",
                row.get("line_start") or 0,
                row.get("name") or "",
            )
        )
        return callers

    async def find_callees(
        self,
        function_name: str,
        file_path: Optional[str] = None,
        edge_type: str = "CALLS",
        max_depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """Find functions reached by a graph traversal from a starting function."""
        if not self._initialized:
            await self.initialize()

        properties: Dict[str, Any] = {"name": function_name}
        if file_path:
            properties["file_path"] = file_path

        nodes = self._query_graph_nodes(labels=["Function"], properties=properties)
        callees: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for node in nodes:
            start_node_id = node.get("id")
            if not start_node_id:
                continue

            traversal = self._traverse_graph(
                start_node_id=start_node_id,
                edge_types=[edge_type] if edge_type else None,
                max_depth=max_depth,
            )
            for visited in traversal.get("nodes", []):
                callee = self._graph_node_to_result(visited)
                callee_id = callee.get("id")
                if not callee_id or callee_id == start_node_id or callee_id in seen_ids:
                    continue
                seen_ids.add(callee_id)
                callees.append(callee)

        callees.sort(
            key=lambda row: (
                row.get("file_path") or "",
                row.get("line_start") or 0,
                row.get("name") or "",
            )
        )
        return callees

    async def trace_execution_path(
        self,
        entry_function: str,
        file_path: Optional[str] = None,
        max_depth: int = 3,
        edge_type: str = "CALLS",
    ) -> Dict[str, Any]:
        """Trace a bounded execution path from an entry function through the call graph."""
        if not self._initialized:
            await self.initialize()

        properties: Dict[str, Any] = {"name": entry_function}
        if file_path:
            properties["file_path"] = file_path

        nodes = self._query_graph_nodes(labels=["Function"], properties=properties)
        if not nodes:
            return {
                "entry_point": None,
                "nodes": [],
                "edges": [],
                "edge_type": edge_type,
                "max_depth": max_depth,
            }

        entry_node = nodes[0]
        entry_node_id = entry_node.get("id")
        if not entry_node_id:
            return {
                "entry_point": None,
                "nodes": [],
                "edges": [],
                "edge_type": edge_type,
                "max_depth": max_depth,
            }

        traversal = self._traverse_graph(
            start_node_id=entry_node_id,
            edge_types=[edge_type] if edge_type else None,
            max_depth=max_depth,
        )
        node_map: Dict[str, Dict[str, Any]] = {
            entry_node_id: self._graph_node_to_result(entry_node)
        }
        for node in traversal.get("nodes", []):
            normalized = self._graph_node_to_result(node)
            node_id = normalized.get("id")
            if node_id:
                node_map[node_id] = normalized

        edges = []
        for edge in traversal.get("edges", []):
            normalized = self._graph_edge_to_result(edge)
            if edge_type and normalized.get("edge_type") != edge_type:
                continue
            edges.append(normalized)

        ordered_nodes = list(node_map.values())
        ordered_nodes.sort(
            key=lambda row: (
                row.get("file_path") or "",
                row.get("line_start") or 0,
                row.get("name") or "",
            )
        )

        return {
            "entry_point": node_map[entry_node_id],
            "nodes": ordered_nodes,
            "edges": edges,
            "edge_type": edge_type,
            "max_depth": max_depth,
        }

    async def find_similar_bugs(
        self,
        bug_description: str,
        language: Optional[str] = None,
        top_k: int = 10,
        include_graph_context: bool = True,
        context_limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find semantically similar bug sites and enrich them with local graph context."""
        if not self._initialized:
            await self.initialize()

        document_filter = {"language": language} if language else None
        ranked = await self.hybrid_search(
            query=bug_description,
            document_filter=document_filter,
            top_k=top_k,
        )

        enriched: List[Dict[str, Any]] = []
        for row in ranked:
            result = dict(row)
            metadata = dict(result.get("metadata", {}) or {})
            symbol_name = (
                result.get("symbol_name") or metadata.get("qualified_name") or metadata.get("name")
            )
            file_path = result.get("file_path") or metadata.get("file_path")

            if include_graph_context and symbol_name and file_path:
                callers = await self.find_callers(symbol_name, file_path=file_path)
                callees = await self.find_callees(symbol_name, file_path=file_path)
                result["graph_context"] = {
                    "callers": callers[:context_limit],
                    "callees": callees[:context_limit],
                    "related_files": sorted(
                        {
                            candidate.get("file_path")
                            for candidate in [
                                *callers[:context_limit],
                                *callees[:context_limit],
                            ]
                            if candidate.get("file_path")
                        }
                    ),
                }
            else:
                result["graph_context"] = {
                    "callers": [],
                    "callees": [],
                    "related_files": [],
                }
            enriched.append(result)

        return enriched

    async def localize_issue(
        self,
        issue_description: str,
        language: Optional[str] = None,
        top_k: int = 10,
        include_graph_context: bool = True,
        context_limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return file-level candidates for an issue using semantic seeds plus graph expansion."""
        if not self._initialized:
            await self.initialize()

        document_filter = {"language": language} if language else None
        ranked = await self.hybrid_search(
            query=issue_description,
            document_filter=document_filter,
            top_k=max(top_k * 2, top_k),
        )

        accumulator = IssueLocalizationAccumulator(
            issue_description,
            context_limit=context_limit,
        )
        for row in ranked:
            accumulator.add_seed(row)
            metadata = dict(row.get("metadata", {}) or {})
            symbol_name = (
                row.get("symbol_name") or metadata.get("qualified_name") or metadata.get("name")
            )
            file_path = row.get("file_path") or metadata.get("file_path")
            score = float(row.get("score", 0.0) or 0.0)

            if not include_graph_context or not symbol_name or not file_path:
                continue

            callers = await self.find_callers(
                symbol_name,
                file_path=file_path,
                max_depth=1,
            )
            callees = await self.find_callees(
                symbol_name,
                file_path=file_path,
                max_depth=1,
            )
            accumulator.attach_graph_context(
                file_path,
                callers=callers[:context_limit],
                callees=callees[:context_limit],
            )
            accumulator.add_graph_neighbors(
                seed_file_path=file_path,
                seed_symbol=symbol_name,
                seed_score=score,
                relation="callers",
                neighbors=callers[:context_limit],
            )
            accumulator.add_graph_neighbors(
                seed_file_path=file_path,
                seed_symbol=symbol_name,
                seed_score=score,
                relation="callees",
                neighbors=callees[:context_limit],
            )

        return accumulator.finalize(top_k=top_k)

    async def analyze_change_impact(
        self,
        change_description: str,
        language: Optional[str] = None,
        top_k: int = 10,
        include_graph_context: bool = True,
        context_limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return ranked blast-radius candidates for a proposed change."""
        if not self._initialized:
            await self.initialize()

        document_filter = {"language": language} if language else None
        ranked = await self.hybrid_search(
            query=change_description,
            document_filter=document_filter,
            top_k=max(top_k * 2, top_k),
        )

        accumulator = ChangeImpactAccumulator(
            change_description,
            context_limit=context_limit,
        )
        for row in ranked:
            accumulator.add_seed(row)
            metadata = dict(row.get("metadata", {}) or {})
            symbol_name = (
                row.get("symbol_name") or metadata.get("qualified_name") or metadata.get("name")
            )
            file_path = row.get("file_path") or metadata.get("file_path")
            score = float(row.get("score", 0.0) or 0.0)

            if not include_graph_context or not symbol_name or not file_path:
                continue

            callers = await self.find_callers(
                symbol_name,
                file_path=file_path,
                max_depth=2,
            )
            callees = await self.find_callees(
                symbol_name,
                file_path=file_path,
                max_depth=1,
            )
            accumulator.attach_graph_context(
                file_path,
                callers=callers[:context_limit],
                callees=callees[:context_limit],
            )
            accumulator.add_graph_neighbors(
                seed_file_path=file_path,
                seed_symbol=symbol_name,
                seed_score=score,
                relation="callers",
                neighbors=callers[:context_limit],
            )
            accumulator.add_graph_neighbors(
                seed_file_path=file_path,
                seed_symbol=symbol_name,
                seed_score=score,
                relation="callees",
                neighbors=callees[:context_limit],
            )

        return accumulator.finalize(top_k=top_k)

    def _ensure_collection(self, collection_name: str) -> None:
        """Create a collection if it does not already exist."""
        try:
            if CollectionConfig is not None:
                config = CollectionConfig(
                    name=collection_name,
                    dimension=self._dimension,
                    distance_metric=self._distance_metric(),
                    storage_engine=self._storage_engine(),
                    primary_indexing_algorithm=self._indexing_algorithm(),
                )
                self._client.create_collection(collection_name, config=config)
            else:
                self._client.create_collection(
                    collection_name,
                    dimension=self._dimension,
                )
        except Exception:
            logger.debug("Collection %s already exists or could not be created", collection_name)

    def _ensure_graph(self) -> None:
        """Create the graph namespace if supported by the client."""
        try:
            self._client.create_graph(
                graph_id=self._graph_collection,
                name=self._graph_collection,
                description=f"Victor code graph for workspace {self._workspace}",
            )
        except Exception:
            logger.debug(
                "Graph %s already exists or could not be created",
                self._graph_collection,
            )

    def _build_graph_api(self) -> Optional[Any]:
        """Create the optional SDK graph helper when available."""
        if self._client is None or ProximaDBGraph is None:
            return None
        try:
            return ProximaDBGraph(self._client, self._graph_collection)
        except Exception:
            logger.debug("Failed to initialize ProximaDBGraph for %s", self._graph_collection)
            return None

    def _query_graph_nodes(
        self,
        labels: List[str],
        properties: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        try:
            result = self._client.query_nodes(
                graph_id=self._graph_collection,
                labels=labels,
                properties=properties,
            )
        except TypeError:
            result = self._client.query_nodes(labels=labels, properties=properties)
        except Exception:
            logger.debug(
                "Graph node lookup failed for labels=%s properties=%s",
                labels,
                properties,
            )
            return []

        if isinstance(result, dict):
            return list(result.get("nodes", []) or [])
        return []

    def _traverse_graph(
        self,
        start_node_id: str,
        edge_types: Optional[List[str]],
        max_depth: int,
    ) -> Dict[str, Any]:
        try:
            result = self._client.traverse_graph(
                graph_id=self._graph_collection,
                start_node_id=start_node_id,
                max_depth=max_depth,
                edge_types=edge_types,
            )
        except TypeError:
            result = self._client.traverse_graph(
                start_node_id=start_node_id,
                max_depth=max_depth,
                edge_types=edge_types,
            )
        except Exception:
            logger.debug("Graph traversal failed for %s", start_node_id)
            return {"nodes": [], "edges": []}

        if isinstance(result, dict):
            return result
        return {"nodes": [], "edges": []}

    def _graph_node_to_result(self, node: Any) -> Dict[str, Any]:
        if isinstance(node, dict):
            node_id = node.get("id")
            labels = list(node.get("labels", []) or [])
            metadata = dict(node.get("properties", {}) or {})
        else:
            node_id = getattr(node, "id", None)
            labels = list(getattr(node, "labels", []) or [])
            metadata = dict(getattr(node, "properties", {}) or {})

        return {
            "id": node_id,
            "name": metadata.get("qualified_name") or metadata.get("name"),
            "file_path": metadata.get("file_path"),
            "line_start": metadata.get("line_start"),
            "line_end": metadata.get("line_end"),
            "labels": labels,
            "metadata": metadata,
        }

    def _graph_edge_to_result(self, edge: Any) -> Dict[str, Any]:
        if isinstance(edge, dict):
            edge_id = edge.get("id")
            from_node_id = edge.get("from_node_id") or edge.get("from_node")
            to_node_id = edge.get("to_node_id") or edge.get("to_node")
            edge_type = edge.get("edge_type")
            metadata = dict(edge.get("properties", {}) or {})
        else:
            edge_id = getattr(edge, "id", None)
            from_node_id = getattr(edge, "from_node_id", None) or getattr(edge, "from_node", None)
            to_node_id = getattr(edge, "to_node_id", None) or getattr(edge, "to_node", None)
            edge_type = getattr(edge, "edge_type", None)
            metadata = dict(getattr(edge, "properties", {}) or {})

        return {
            "id": edge_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "edge_type": edge_type,
            "metadata": metadata,
        }

    def _collection_name(self, key: str, suffix: str) -> str:
        override = self.config.extra_config.get(key)
        if override:
            return self._normalize_collection_name(str(override))
        return self._normalize_collection_name(f"{self._workspace}_{suffix}")

    def _normalize_collection_name(self, name: str) -> str:
        normalized = _IDENTIFIER_RE.sub("_", name.strip().lower()).strip("_") or "victor_store"
        if len(normalized) < 8:
            normalized = f"{normalized}_store"
        return normalized

    def _distance_metric(self) -> Any:
        mapping = {
            "cosine": "cosine",
            "euclidean": "euclidean",
            "dot": "dot_product",
            "dot_product": "dot_product",
        }
        metric = mapping.get(self.config.distance_metric.lower(), "cosine")
        if DistanceMetric is None:
            return metric
        return DistanceMetric(metric)

    def _storage_engine(self) -> Any:
        engine = str(self.config.extra_config.get("vector_storage_engine", "sst")).lower()
        if StorageEngine is None:
            return engine
        try:
            return StorageEngine(engine)
        except ValueError:
            return StorageEngine.SST

    def _indexing_algorithm(self) -> Any:
        algorithm = str(self.config.extra_config.get("vector_index", "hnsw")).lower()
        if IndexingAlgorithm is None:
            return algorithm
        try:
            return IndexingAlgorithm(algorithm)
        except ValueError:
            return IndexingAlgorithm.HNSW

    def _vector_record(
        self,
        record_id: str,
        vector: Sequence[float],
        source: str,
        metadata: Dict[str, Any],
    ) -> Any:
        if VectorRecord is None:
            return _FallbackVectorRecord(
                id=record_id,
                vector=list(vector),
                source=source,
                metadata=self._sanitize_metadata(metadata),
            )
        return VectorRecord(
            id=record_id,
            vector=list(vector),
            source=source,
            metadata=self._sanitize_metadata(metadata),
        )

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
                continue
            if isinstance(value, list):
                sanitized[key] = [item for item in value if isinstance(item, (str, int, float))]
                continue
            sanitized[key] = json.dumps(value, sort_keys=True, default=str)
        return sanitized

    def _iter_record_collections(self) -> Iterable[str]:
        yield self._vector_collection
        if self._document_enabled:
            yield self._document_collection
        if self._metrics_enabled:
            yield self._metrics_collection

    def _search_ids_for_file(self, collection: str, file_path: str) -> List[str]:
        hits = self._search_records(
            collection=collection,
            filter_metadata={"file_path": file_path},
            limit=self._query_scan_limit,
        )
        return [getattr(hit, "id", "") for hit in hits if getattr(hit, "id", "")]

    def _search_records(
        self,
        collection: str,
        filter_metadata: Optional[Dict[str, Any]],
        limit: int,
    ) -> List[Any]:
        try:
            return self._client.search(
                collection,
                vector=self._zero_vector(),
                top_k=limit,
                metadata_filter=filter_metadata or None,
            )
        except Exception:
            logger.debug("Search against collection %s failed", collection)
            return []

    def _store_document_snapshot(
        self,
        file_path: str,
        content: str,
        language: str,
        base_metadata: Dict[str, Any],
    ) -> None:
        record_id = self._document_record_id(file_path, base_metadata["file_hash"])
        metadata = {
            **base_metadata,
            "record_type": "document",
            "title": Path(file_path).name,
            "file_extension": Path(file_path).suffix,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": content.count("\n") + 1,
            "language": language,
        }
        record = self._vector_record(
            record_id=record_id,
            vector=self._zero_vector(),
            source=content,
            metadata=metadata,
        )
        self._client.insert_vectors(self._document_collection, records=[record])

    def _store_metrics_snapshot(
        self,
        file_path: str,
        content: str,
        language: str,
        symbols: List[_GraphSymbol],
        recorded_at: datetime,
        base_metadata: Dict[str, Any],
    ) -> int:
        metrics = self._extract_code_metrics(content, language, symbols)
        records = []
        for metric in metrics:
            record_id = self._metric_record_id(file_path, metric["metric_name"], recorded_at)
            metadata = {
                **base_metadata,
                "record_type": "metric",
                "recorded_at": recorded_at.isoformat(),
                "recorded_at_epoch_ms": int(recorded_at.timestamp() * 1000),
                **metric,
            }
            records.append(
                self._vector_record(
                    record_id=record_id,
                    vector=self._zero_vector(),
                    source=json.dumps(metric, sort_keys=True),
                    metadata=metadata,
                )
            )

        if records:
            self._client.insert_vectors(self._metrics_collection, records=records)
        return len(records)

    def _chunk_code(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> List[CodeChunk]:
        return self._code_chunker.chunk(file_path, content, language=language)

    def _extract_graph_snapshot(
        self,
        file_path: str,
        content: str,
        language: str,
        tree_analysis: Optional[_TreeSitterAnalysis] = None,
    ) -> _GraphSnapshot:
        if tree_analysis is not None:
            return self._extract_tree_sitter_snapshot(file_path, content, language, tree_analysis)

        if TREE_SITTER_AVAILABLE:
            parsed = self._build_tree_sitter_analysis(file_path, content, language)
            if parsed is not None:
                return self._extract_tree_sitter_snapshot(file_path, content, language, parsed)

        if language == "python":
            return self._extract_python_ast_snapshot(file_path, content)

        return _GraphSnapshot([], [], [], [], [], [])

    def _build_tree_sitter_analysis(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> Optional[_TreeSitterAnalysis]:
        if not TREE_SITTER_AVAILABLE:
            return None

        try:
            plugin = self._language_registry.get(language)
            queries = plugin.tree_sitter_queries
            if not queries or not queries.symbols:
                return None
            get_parser = load_tree_sitter_get_parser()
            parser = get_parser(language)
            tree = parser.parse(content.encode("utf-8"))
        except Exception as exc:
            logger.debug("Tree-sitter parse failed for %s: %s", file_path, exc)
            return None

        return _TreeSitterAnalysis(
            parser=parser,
            tree=tree,
            queries=queries,
        )

    def _extract_tree_sitter_snapshot(
        self,
        file_path: str,
        content: str,
        language: str,
        analysis: _TreeSitterAnalysis,
    ) -> _GraphSnapshot:
        symbols = self._extract_symbols_from_tree(
            analysis.tree,
            analysis.parser,
            analysis.queries,
            content,
        )
        calls = self._extract_call_edges_from_tree(
            analysis.tree,
            analysis.parser,
            analysis.queries,
        )
        inheritance = self._extract_pair_edges_from_tree(
            analysis.tree,
            analysis.parser,
            analysis.queries.inheritance,
            "child",
            "base",
            "INHERITS",
        )
        implements = self._extract_pair_edges_from_tree(
            analysis.tree,
            analysis.parser,
            analysis.queries.implements,
            "child",
            "interface",
            "IMPLEMENTS",
        )
        if analysis.queries.implements and not implements:
            implements = self._extract_pair_edges_from_tree(
                analysis.tree,
                analysis.parser,
                analysis.queries.implements,
                "child",
                "base",
                "IMPLEMENTS",
            )
        composition = self._extract_pair_edges_from_tree(
            analysis.tree,
            analysis.parser,
            analysis.queries.composition,
            "owner",
            "type",
            "COMPOSITION",
        )
        imports = self._extract_imports(content, language)
        self._enrich_python_symbols(symbols, content)
        return _GraphSnapshot(symbols, calls, inheritance, implements, composition, imports)

    def _extract_symbols_from_tree(
        self,
        tree: Any,
        parser: Any,
        queries: Any,
        content: str,
    ) -> List[_GraphSymbol]:
        symbols: List[_GraphSymbol] = []
        for pattern in queries.symbols:
            captures = self._run_query(tree, parser, pattern.query)
            name_nodes = captures.get("name", [])
            def_nodes = captures.get("def", [])
            def_by_start = {node.start_point[0]: node for node in def_nodes}
            for name_node in name_nodes:
                name = name_node.text.decode("utf-8", errors="ignore")
                if not name:
                    continue
                end_line = name_node.end_point[0] + 1
                name_line = name_node.start_point[0]
                for def_line, def_node in def_by_start.items():
                    if def_line <= name_line <= def_node.end_point[0]:
                        end_line = def_node.end_point[0] + 1
                        break

                symbols.append(
                    _GraphSymbol(
                        name=name,
                        symbol_type=pattern.symbol_type,
                        line_start=name_node.start_point[0] + 1,
                        line_end=end_line,
                        parent_symbol=self._find_enclosing_symbol(
                            name_node, queries.enclosing_scopes
                        ),
                    )
                )
        return self._dedupe_symbols(symbols)

    def _extract_call_edges_from_tree(
        self,
        tree: Any,
        parser: Any,
        queries: Any,
    ) -> List[_GraphEdge]:
        if not queries.calls:
            return []

        edges: List[_GraphEdge] = []
        captures = self._run_query(tree, parser, queries.calls)
        for node in captures.get("callee", []):
            callee = node.text.decode("utf-8", errors="ignore")
            caller = self._find_enclosing_symbol(node, queries.enclosing_scopes)
            if not callee or not caller:
                continue
            edges.append(
                _GraphEdge(
                    source=caller,
                    target=callee,
                    edge_type="CALLS",
                    line_number=node.start_point[0] + 1,
                )
            )
        return self._dedupe_edges(edges)

    def _extract_pair_edges_from_tree(
        self,
        tree: Any,
        parser: Any,
        query_src: Optional[str],
        left_key: str,
        right_key: str,
        edge_type: str,
    ) -> List[_GraphEdge]:
        if not query_src:
            return []

        captures = self._run_query(tree, parser, query_src)
        left_nodes = captures.get(left_key, [])
        right_nodes = captures.get(right_key, [])
        edges = []
        for left_node, right_node in zip(left_nodes, right_nodes, strict=False):
            left_name = left_node.text.decode("utf-8", errors="ignore")
            right_name = right_node.text.decode("utf-8", errors="ignore")
            if not left_name or not right_name:
                continue
            edges.append(
                _GraphEdge(
                    source=left_name,
                    target=right_name,
                    edge_type=edge_type,
                    line_number=left_node.start_point[0] + 1,
                )
            )
        return self._dedupe_edges(edges)

    def _run_query(self, tree: Any, parser: Any, query_src: str) -> Dict[str, List[Any]]:
        try:
            query = Query(parser.language, query_src)
            cursor = QueryCursor(query)
            return cursor.captures(tree.root_node)
        except Exception as exc:
            logger.debug("Tree-sitter query failed: %s", exc)
            return {}

    def _find_enclosing_symbol(
        self,
        node: Any,
        enclosing_scopes: List[tuple[str, str]],
    ) -> Optional[str]:
        if not enclosing_scopes:
            return None

        current = node.parent
        method_name: Optional[str] = None
        class_name: Optional[str] = None
        while current is not None:
            for node_type, field_name in enclosing_scopes:
                if current.type != node_type:
                    continue
                field = current.child_by_field_name(field_name)
                if field is None:
                    continue
                text = field.text.decode("utf-8", errors="ignore")
                if node_type in {
                    "class_definition",
                    "class_declaration",
                    "interface_declaration",
                    "struct_item",
                    "enum_item",
                    "trait_item",
                }:
                    class_name = class_name or text
                else:
                    method_name = method_name or text
            current = current.parent
        if method_name:
            if class_name:
                return f"{class_name}.{method_name}"
            return method_name
        return class_name

    def _extract_python_ast_snapshot(self, file_path: str, content: str) -> _GraphSnapshot:
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError as exc:
            logger.debug("Python AST parse failed for %s: %s", file_path, exc)
            return _GraphSnapshot([], [], [], [], [], [])

        symbols: List[_GraphSymbol] = []
        call_edges: List[_GraphEdge] = []
        inheritance_edges: List[_GraphEdge] = []
        imports: List[str] = []

        class PythonVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.scope: List[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                parent = self.scope[-1] if self.scope else None
                symbols.append(
                    _GraphSymbol(
                        name=node.name,
                        symbol_type="class",
                        line_start=node.lineno,
                        line_end=getattr(node, "end_lineno", node.lineno),
                        parent_symbol=parent,
                        docstring=ast.get_docstring(node),
                    )
                )
                for base in node.bases:
                    base_name = self._name_for_expr(base)
                    if base_name:
                        inheritance_edges.append(
                            _GraphEdge(
                                source=node.name,
                                target=base_name,
                                edge_type="INHERITS",
                                line_number=node.lineno,
                            )
                        )
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                self._visit_function(node, is_async=False)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
                self._visit_function(node, is_async=True)

            def _visit_function(
                self,
                node: ast.FunctionDef | ast.AsyncFunctionDef,
                is_async: bool,
            ) -> None:
                parent = self.scope[-1] if self.scope else None
                symbols.append(
                    _GraphSymbol(
                        name=node.name,
                        symbol_type="function",
                        line_start=node.lineno,
                        line_end=getattr(node, "end_lineno", node.lineno),
                        parent_symbol=parent,
                        signature=self._signature_for_function(node, is_async=is_async),
                        docstring=ast.get_docstring(node),
                        metadata={"is_async": is_async},
                    )
                )
                self.scope.append(node.name if not parent else f"{parent}.{node.name}")
                self.generic_visit(node)
                self.scope.pop()

            def visit_Call(self, node: ast.Call) -> Any:
                caller = self.scope[-1] if self.scope else None
                callee = self._name_for_expr(node.func)
                if caller and callee:
                    call_edges.append(
                        _GraphEdge(
                            source=caller,
                            target=callee,
                            edge_type="CALLS",
                            line_number=getattr(node, "lineno", 0),
                        )
                    )
                self.generic_visit(node)

            def visit_Import(self, node: ast.Import) -> Any:
                for alias in node.names:
                    imports.append(alias.name)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
                if node.module:
                    imports.append(node.module)

            def _name_for_expr(self, expr: ast.AST) -> Optional[str]:
                if isinstance(expr, ast.Name):
                    return expr.id
                if isinstance(expr, ast.Attribute):
                    return expr.attr
                if isinstance(expr, ast.Call):
                    return self._name_for_expr(expr.func)
                return None

            def _signature_for_function(
                self,
                node: ast.FunctionDef | ast.AsyncFunctionDef,
                is_async: bool,
            ) -> str:
                params = [arg.arg for arg in node.args.args]
                if node.args.vararg:
                    params.append(f"*{node.args.vararg.arg}")
                params.extend(arg.arg for arg in node.args.kwonlyargs)
                if node.args.kwarg:
                    params.append(f"**{node.args.kwarg.arg}")
                prefix = "async " if is_async else ""
                return f"{prefix}{node.name}({', '.join(params)})"

        visitor = PythonVisitor()
        visitor.visit(tree)
        return _GraphSnapshot(
            symbols=self._dedupe_symbols(symbols),
            call_edges=self._dedupe_edges(call_edges),
            inheritance_edges=self._dedupe_edges(inheritance_edges),
            implements_edges=[],
            composition_edges=[],
            imports=sorted(set(imports)),
        )

    def _enrich_python_symbols(self, symbols: List[_GraphSymbol], content: str) -> None:
        if not symbols:
            return
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        by_line = {(symbol.name, symbol.line_start): symbol for symbol in symbols}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = (node.name, node.lineno)
                symbol = by_line.get(key)
                if symbol is None:
                    continue
                params = [arg.arg for arg in node.args.args]
                if node.args.vararg:
                    params.append(f"*{node.args.vararg.arg}")
                if node.args.kwarg:
                    params.append(f"**{node.args.kwarg.arg}")
                prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                symbol.signature = f"{prefix}{node.name}({', '.join(params)})"
                symbol.docstring = ast.get_docstring(node)
                symbol.metadata["is_async"] = isinstance(node, ast.AsyncFunctionDef)
            elif isinstance(node, ast.ClassDef):
                key = (node.name, node.lineno)
                symbol = by_line.get(key)
                if symbol is not None:
                    symbol.docstring = ast.get_docstring(node)

    def _extract_imports(self, content: str, language: str) -> List[str]:
        if language == "python":
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return []
            imports: List[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            return sorted(set(imports))

        query_src = _IMPORT_QUERIES.get(language)
        if not query_src or not TREE_SITTER_AVAILABLE:
            return []

        try:
            get_parser = load_tree_sitter_get_parser()
            parser = get_parser(language)
            tree = parser.parse(content.encode("utf-8"))
            captures = self._run_query(tree, parser, query_src)
        except Exception as exc:
            logger.debug("Import extraction failed for %s: %s", language, exc)
            return []

        imports = []
        for node in captures.get("source", []):
            value = node.text.decode("utf-8", errors="ignore").strip("'\"")
            if value:
                imports.append(value)
        return sorted(set(imports))

    def _dedupe_symbols(self, symbols: List[_GraphSymbol]) -> List[_GraphSymbol]:
        deduped: Dict[tuple[str, str, int, Optional[str]], _GraphSymbol] = {}
        for symbol in symbols:
            key = (
                symbol.name,
                symbol.symbol_type,
                symbol.line_start,
                symbol.parent_symbol,
            )
            deduped[key] = symbol
        return list(deduped.values())

    def _dedupe_edges(self, edges: List[_GraphEdge]) -> List[_GraphEdge]:
        deduped: Dict[tuple[str, str, str, int], _GraphEdge] = {}
        for edge in edges:
            key = (edge.source, edge.target, edge.edge_type, edge.line_number)
            deduped[key] = edge
        return list(deduped.values())

    def _store_graph_snapshot(
        self,
        file_path: str,
        language: str,
        base_metadata: Dict[str, Any],
        snapshot: _GraphSnapshot,
    ) -> Dict[str, int]:
        queued_nodes: Dict[str, Dict[str, Any]] = {}
        queued_edges: Dict[str, Dict[str, Any]] = {}
        module_id = self._module_node_id(file_path)
        self._queue_graph_node(
            queued_nodes,
            node_id=module_id,
            labels=["Module"],
            properties={
                **base_metadata,
                "name": Path(file_path).name,
                "node_type": "module",
            },
        )

        symbol_nodes = {}
        function_count = 0
        class_count = 0

        for symbol in snapshot.symbols:
            labels = ["Class"] if symbol.symbol_type == "class" else ["Function"]
            node_id = self._symbol_node_id(file_path, symbol)
            properties = {
                **base_metadata,
                "name": symbol.name,
                "qualified_name": symbol.qualified_name,
                "symbol_type": symbol.symbol_type,
                "file_path": file_path,
                "language": language,
                "line_start": symbol.line_start,
                "line_end": symbol.line_end,
                "parent_symbol": symbol.parent_symbol,
                "signature": symbol.signature,
                "docstring": symbol.docstring,
                **symbol.metadata,
            }
            self._queue_graph_node(
                queued_nodes,
                node_id=node_id,
                labels=labels,
                properties=properties,
            )
            symbol_nodes[symbol.qualified_name] = node_id
            symbol_nodes[symbol.name] = node_id

            if symbol.symbol_type == "class":
                class_count += 1
            else:
                function_count += 1

            if symbol.parent_symbol and symbol.parent_symbol in symbol_nodes:
                self._queue_graph_edge(
                    queued_edges,
                    edge_id=self._edge_id(
                        "defines",
                        symbol.parent_symbol,
                        symbol.qualified_name,
                        file_path,
                    ),
                    from_node_id=symbol_nodes[symbol.parent_symbol],
                    to_node_id=node_id,
                    edge_type="DEFINES",
                    properties={"file_path": file_path},
                )
            else:
                self._queue_graph_edge(
                    queued_edges,
                    edge_id=self._edge_id("defines", file_path, symbol.qualified_name, file_path),
                    from_node_id=module_id,
                    to_node_id=node_id,
                    edge_type="DEFINES",
                    properties={"file_path": file_path},
                )

        call_count = 0
        for edge in snapshot.call_edges:
            source_id = symbol_nodes.get(edge.source)
            if not source_id:
                continue
            target_id = symbol_nodes.get(edge.target)
            is_internal_target = target_id is not None
            if target_id is None:
                target_id = self._external_symbol_node_id(edge.target, "Function")
                self._queue_graph_node(
                    queued_nodes,
                    node_id=target_id,
                    labels=["Function", "External"],
                    properties={
                        "workspace": self._workspace,
                        "name": edge.target,
                        "qualified_name": edge.target,
                        "symbol_type": "function",
                        "node_type": "external_function",
                    },
                )
            self._queue_graph_edge(
                queued_edges,
                edge_id=self._edge_id(
                    "calls", edge.source, edge.target, file_path, edge.line_number
                ),
                from_node_id=source_id,
                to_node_id=target_id,
                edge_type="CALLS",
                properties={
                    "file_path": file_path,
                    "line_number": edge.line_number,
                    **edge.metadata,
                },
            )
            if is_internal_target:
                call_count += 1

        for edge in snapshot.inheritance_edges:
            self._store_symbol_relationship(
                edge,
                file_path,
                symbol_nodes,
                queued_nodes,
                queued_edges,
                class_label="Class",
            )
        for edge in snapshot.implements_edges:
            self._store_symbol_relationship(
                edge,
                file_path,
                symbol_nodes,
                queued_nodes,
                queued_edges,
                class_label="Interface",
            )
        for edge in snapshot.composition_edges:
            self._store_symbol_relationship(
                edge,
                file_path,
                symbol_nodes,
                queued_nodes,
                queued_edges,
                class_label="Class",
            )

        for imported in snapshot.imports:
            import_id = self._module_import_node_id(imported)
            self._queue_graph_node(
                queued_nodes,
                node_id=import_id,
                labels=["Module", "Imported"],
                properties={
                    "workspace": self._workspace,
                    "name": imported,
                    "file_path": imported,
                },
            )
            self._queue_graph_edge(
                queued_edges,
                edge_id=self._edge_id("imports", file_path, imported, file_path),
                from_node_id=module_id,
                to_node_id=import_id,
                edge_type="IMPORTS",
                properties={"file_path": file_path},
            )

        self._flush_graph_records(
            nodes=list(queued_nodes.values()),
            edges=list(queued_edges.values()),
        )

        return {
            "functions": function_count,
            "classes": class_count,
            "calls": call_count,
            "imports": len(snapshot.imports),
            "inherits": len(snapshot.inheritance_edges),
            "implements": len(snapshot.implements_edges),
            "composition": len(snapshot.composition_edges),
        }

    def _store_symbol_relationship(
        self,
        edge: _GraphEdge,
        file_path: str,
        symbol_nodes: Dict[str, str],
        queued_nodes: Dict[str, Dict[str, Any]],
        queued_edges: Dict[str, Dict[str, Any]],
        class_label: str,
    ) -> None:
        source_id = symbol_nodes.get(edge.source)
        if source_id is None:
            source_id = self._external_symbol_node_id(edge.source, class_label)
            self._queue_graph_node(
                queued_nodes,
                node_id=source_id,
                labels=[class_label, "External"],
                properties={"workspace": self._workspace, "name": edge.source},
            )

        target_id = symbol_nodes.get(edge.target)
        if target_id is None:
            target_id = self._external_symbol_node_id(edge.target, class_label)
            self._queue_graph_node(
                queued_nodes,
                node_id=target_id,
                labels=[class_label, "External"],
                properties={"workspace": self._workspace, "name": edge.target},
            )

        self._queue_graph_edge(
            queued_edges,
            edge_id=self._edge_id(edge.edge_type.lower(), edge.source, edge.target, file_path),
            from_node_id=source_id,
            to_node_id=target_id,
            edge_type=edge.edge_type,
            properties={
                "file_path": file_path,
                "line_number": edge.line_number,
                **edge.metadata,
            },
        )

    def _queue_graph_node(
        self,
        queued_nodes: Dict[str, Dict[str, Any]],
        node_id: str,
        labels: List[str],
        properties: Dict[str, Any],
    ) -> None:
        existing = queued_nodes.get(node_id)
        if existing is None:
            queued_nodes[node_id] = {
                "id": node_id,
                "labels": list(dict.fromkeys(labels)),
                "properties": dict(properties),
            }
            return

        existing["labels"] = list(dict.fromkeys([*existing.get("labels", []), *labels]))
        existing["properties"].update(properties)

    def _queue_graph_edge(
        self,
        queued_edges: Dict[str, Dict[str, Any]],
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
        properties: Dict[str, Any],
    ) -> None:
        queued_edges[edge_id] = {
            "id": edge_id,
            "from_node": from_node_id,
            "to_node": to_node_id,
            "edge_type": edge_type,
            "properties": dict(properties),
        }

    def _flush_graph_records(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> None:
        graph_api = self._graph_api or self._build_graph_api()
        if graph_api is not None:
            try:
                graph_api.batch_create_nodes(nodes, batch_size=max(self._batch_size, 1))
                graph_api.batch_create_edges(edges, batch_size=max(self._batch_size, 1))
                return
            except Exception:
                logger.debug("Graph batch write failed for %s", self._graph_collection)

        for node in nodes:
            self._create_graph_node(
                node_id=node["id"],
                labels=node.get("labels", []),
                properties=node.get("properties", {}),
            )
        for edge in edges:
            self._create_graph_edge(
                edge_id=edge["id"],
                from_node_id=edge["from_node"],
                to_node_id=edge["to_node"],
                edge_type=edge["edge_type"],
                properties=edge.get("properties", {}),
            )

    def _create_graph_node(
        self,
        node_id: str,
        labels: List[str],
        properties: Dict[str, Any],
    ) -> None:
        raw_client = getattr(self._client, "_client", None)
        payload = {
            "node_id": node_id,
            "labels": labels,
            "properties": properties,
        }
        try:
            if raw_client and hasattr(raw_client, "create_node"):
                raw_client.create_node(graph_id=self._graph_collection, **payload)
                return
        except TypeError:
            pass
        except Exception:
            logger.debug("Raw graph node creation failed for %s", node_id)

        try:
            self._client.create_node(**payload)
        except Exception:
            logger.debug("Graph node creation failed for %s", node_id)

    def _create_graph_edge(
        self,
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
        properties: Dict[str, Any],
    ) -> None:
        raw_client = getattr(self._client, "_client", None)
        payload = {
            "edge_id": edge_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "edge_type": edge_type,
            "properties": properties,
        }
        try:
            if raw_client and hasattr(raw_client, "create_edge"):
                raw_client.create_edge(graph_id=self._graph_collection, **payload)
                return
        except TypeError:
            pass
        except Exception:
            logger.debug("Raw graph edge creation failed for %s", edge_id)

        try:
            self._client.create_edge(**payload)
        except Exception:
            logger.debug("Graph edge creation failed for %s", edge_id)

    def _extract_code_metrics(
        self,
        content: str,
        language: str,
        symbols: List[_GraphSymbol],
    ) -> List[Dict[str, Any]]:
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        non_comment_lines = [
            line for line in non_empty_lines if not self._is_comment(line, language)
        ]
        function_count = sum(1 for symbol in symbols if symbol.symbol_type == "function")
        class_count = sum(1 for symbol in symbols if symbol.symbol_type == "class")

        metrics = [
            {"metric_name": "lines_of_code", "metric_value": len(non_comment_lines)},
            {"metric_name": "raw_line_count", "metric_value": len(lines)},
            {"metric_name": "function_count", "metric_value": function_count},
            {"metric_name": "class_count", "metric_value": class_count},
            {
                "metric_name": "cyclomatic_complexity",
                "metric_value": self._estimate_complexity(content, language),
            },
            {
                "metric_name": "max_nesting_depth",
                "metric_value": self._estimate_nesting_depth(content, language),
            },
        ]
        for metric in metrics:
            metric["language"] = language
        return metrics

    def _estimate_complexity(self, content: str, language: str) -> int:
        if language == "python":
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return 1
            count = 1
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (
                        ast.If,
                        ast.For,
                        ast.AsyncFor,
                        ast.While,
                        ast.Try,
                        ast.ExceptHandler,
                        ast.With,
                        ast.AsyncWith,
                        ast.BoolOp,
                        ast.IfExp,
                        ast.Match,
                    ),
                ):
                    count += 1
            return count

        lowered = content.lower()
        count = 1
        for keyword in _CONTROL_FLOW_KEYWORDS:
            count += lowered.count(keyword)
        return count

    def _estimate_nesting_depth(self, content: str, language: str) -> int:
        if language == "python":
            max_depth = 0
            for line in content.splitlines():
                if not line.strip():
                    continue
                indent = len(line) - len(line.lstrip(" "))
                max_depth = max(max_depth, indent // 4)
            return max_depth

        depth = 0
        max_depth = 0
        for char in content:
            if char == "{":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == "}":
                depth = max(depth - 1, 0)
        return max_depth

    def _is_comment(self, line: str, language: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if language == "python":
            return stripped.startswith("#")
        return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")

    def _search_graph(self, graph_query: str, top_k: int) -> List[Dict[str, Any]]:
        graph_api = self._graph_api or self._build_graph_api()
        if graph_api is not None and graph_query.lstrip().upper().startswith("MATCH"):
            try:
                result = graph_api.query_cypher(graph_query)
                rows: List[Dict[str, Any]] = []
                for node in getattr(result, "nodes", [])[:top_k]:
                    metadata = dict(getattr(node, "properties", {}) or {})
                    rows.append(
                        {
                            "id": getattr(node, "id", None),
                            "file_path": metadata.get("file_path"),
                            "symbol_name": metadata.get("qualified_name") or metadata.get("name"),
                            "content": metadata.get("source", ""),
                            "score": 0.5,
                            "metadata": metadata,
                        }
                    )
                if rows:
                    return rows
            except Exception:
                logger.debug("Graph helper query failed: %s", graph_query)

        try:
            result = self._client.execute_sql(graph_query, collection=self._graph_collection)
        except TypeError:
            result = self._client.execute_sql(graph_query)
        except Exception:
            logger.debug("Graph query failed: %s", graph_query)
            return []

        rows = result.get("rows", []) if isinstance(result, dict) else []
        parsed_rows = []
        for row in rows[:top_k]:
            metadata = dict(row.get("metadata", {}) or {})
            if "properties" in row and isinstance(row["properties"], dict):
                metadata.update(row["properties"])
            parsed_rows.append(
                {
                    "id": row.get("id") or row.get("node_id") or row.get("edge_id"),
                    "file_path": metadata.get("file_path"),
                    "symbol_name": metadata.get("qualified_name") or metadata.get("name"),
                    "content": row.get("content") or row.get("source") or "",
                    "score": float(row.get("score", 0.5)),
                    "metadata": metadata,
                }
            )
        return parsed_rows

    def _merge_hybrid_result(
        self,
        combined: Dict[str, Dict[str, Any]],
        row: Any,
        source: str,
        weight: float,
    ) -> None:
        if isinstance(row, dict):
            metadata = dict(row.get("metadata", {}) or {})
            file_path = row.get("file_path") or metadata.get("file_path", "")
            symbol_name = row.get("symbol_name") or metadata.get("symbol_name")
            content = row.get("content") or row.get("source") or ""
            score = float(row.get("score", 0.0))
            line_number = metadata.get("start_line") or metadata.get("line_number")
        else:
            metadata = dict(getattr(row, "metadata", {}) or {})
            file_path = metadata.get("file_path", "")
            symbol_name = metadata.get("symbol_name")
            content = getattr(row, "source", None) or metadata.get("content", "")
            score = float(getattr(row, "score", 0.0))
            line_number = metadata.get("start_line") or metadata.get("line_number")

        key = self._hybrid_key(file_path, symbol_name, line_number)
        if key not in combined:
            combined[key] = {
                "id": key,
                "file_path": file_path,
                "symbol_name": symbol_name,
                "content": content,
                "score": score * weight,
                "sources": [source],
                "metadata": {**metadata, "file_path": file_path},
            }
            return

        combined[key]["score"] += score * weight
        if source not in combined[key]["sources"]:
            combined[key]["sources"].append(source)
        if content and not combined[key]["content"]:
            combined[key]["content"] = content
        combined[key]["metadata"].update(metadata)

    def _metadata_matches(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    def _within_time_range(
        self,
        timestamp_value: Optional[str],
        time_range: tuple[datetime, datetime],
    ) -> bool:
        if not timestamp_value:
            return False
        try:
            timestamp = datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
        except ValueError:
            return False
        start, end = time_range
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return start <= timestamp <= end

    def _resolve_language(
        self,
        file_path: str,
        content: str,
        language: Optional[str],
    ) -> str:
        if language:
            return language.lower()
        detected = self._language_registry.detect_from_content(content, filename=file_path)
        if detected:
            return detected
        detected = self._language_registry.detect_language(Path(file_path))
        return detected or "python"

    def _zero_vector(self) -> List[float]:
        return [0.0] * self._dimension

    def _chunk_record_id(self, file_path: str, file_hash: str, index: int) -> str:
        return f"{self._workspace}:chunk:{file_path}:{file_hash}:{index}"

    def _document_record_id(self, file_path: str, file_hash: str) -> str:
        return f"{self._workspace}:document:{file_path}:{file_hash}"

    def _metric_record_id(self, file_path: str, metric_name: str, recorded_at: datetime) -> str:
        return (
            f"{self._workspace}:metric:{file_path}:{metric_name}:"
            f"{int(recorded_at.timestamp() * 1000)}"
        )

    def _module_node_id(self, file_path: str) -> str:
        return f"{self._workspace}:module:{file_path}"

    def _module_import_node_id(self, module_name: str) -> str:
        return f"{self._workspace}:imported_module:{module_name}"

    def _symbol_node_id(self, file_path: str, symbol: _GraphSymbol) -> str:
        return (
            f"{self._workspace}:{symbol.symbol_type}:{file_path}:"
            f"{symbol.qualified_name}:{symbol.line_start}"
        )

    def _external_symbol_node_id(self, symbol_name: str, symbol_label: str) -> str:
        return f"{self._workspace}:external:{symbol_label.lower()}:{symbol_name}"

    def _edge_id(
        self,
        prefix: str,
        source: str,
        target: str,
        file_path: str,
        line_number: Optional[int] = None,
    ) -> str:
        material = f"{prefix}|{source}|{target}|{file_path}|{line_number or 0}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        return f"{self._workspace}:{prefix}:{digest}"

    def _hybrid_key(
        self,
        file_path: Optional[str],
        symbol_name: Optional[str],
        line_number: Optional[int],
    ) -> str:
        if symbol_name:
            return f"{file_path or ''}:{symbol_name}"
        return f"{file_path or ''}::{line_number or 0}"


__all__ = ["ProximaDBMultiModelProvider"]
