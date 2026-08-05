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

"""ProximaDB-backed graph store (Code Context Graph backend, TD-11/12/13).

``ProximaGraphStore`` implements :class:`GraphStoreProtocol` over ProximaDB's
ORION graph engine via ``proximadb_sdk.graph.ProximaDBGraph``. Tier-A symbols and
semantic edges live in the correlated hot graph, while Tier-B statement nodes
and CFG/CDG/DDG edges are routed to durable, indexed CPG fragments and fetched
only for explicit dataflow drilldowns. A code symbol is **one** durable
``ProximaRecord`` containing both semantic properties and its vector, addressed
by the same ``oid`` used by its derived ORION traversal projection. A vector hit
therefore maps to its graph node by identity and the always-empty
``graph_node.embedding_ref`` bridge is retired.

SQLite stays the default. This backend is selected per-repo (see
``victor.storage.graph.registry``). The **embedded** path (one local
``EmbeddedProximaDB`` per repo, in-RAM vectors — the ``EmbeddingMode::Memory``
equivalent) is the build/verification target. The multi-tenant **service** path
(``server_url=``) is **WIP**, gated on ProximaDB TD-127 (secondary indexes) and
TD-130/131 (graph bulk-load + REST v2 hybrid).

``ProximaDBGraph`` is synchronous; all calls are dispatched via
``asyncio.to_thread`` to keep the protocol async end-to-end.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

from victor.storage.graph.cpg_fragments import CpgFragmentStore, CpgFragmentStoreProtocol
from victor.storage.graph.edge_types import EdgeType
from victor.storage.graph.protocol import (
    GraphEdge,
    GraphNode,
    GraphQueryResult,
    GraphStoreProtocol,
    GraphTraversalDirection,
    Subgraph,
)
from victor.storage.proxima_runtime import (
    ProximaEmbeddingMode,
    ProximaRepoConnection,
    ProximaUnavailableError,
    graph_id_for_repo,
    is_proxima_available,
    repo_id_from_path,
)

logger = logging.getLogger(__name__)

# Node properties that are first-class GraphNode fields rather than free-form
# metadata. Kept flat in ProximaDB props so ProximaDBGraph.search_symbols /
# find_nodes (which read props["name"], ["signature"], ["file"], ...) work.
_NODE_FIELD_KEYS = (
    "name",
    "file",
    "line",
    "end_line",
    "lang",
    "signature",
    "docstring",
    "parent_id",
    "ast_kind",
    "scope_id",
    "statement_type",
    "requirement_id",
    "visibility",
)


class ProximaGraphStore(GraphStoreProtocol):
    """GraphStoreProtocol implementation backed by ProximaDB (ORION + HNSW)."""

    def __init__(
        self,
        project_path: Optional[Path] = None,
        *,
        repo: Optional[str] = None,
        server_url: Optional[str] = None,
        embedding_mode: ProximaEmbeddingMode | str = ProximaEmbeddingMode.MEMORY,
        data_dir: Optional[Path] = None,
        binary_path: Optional[str] = None,
        graph: Any = None,
        client: Any = None,
        cpg_store: CpgFragmentStoreProtocol | None = None,
        record_collection: Any = None,
    ) -> None:
        """Create a ProximaGraphStore.

        Args:
            project_path: Project root; the repo id is derived from its name.
            repo: Explicit repo id (overrides the one derived from project_path).
            server_url: If set, connect to a running ProximaDB service instead of
                starting an embedded instance. This is the multi-tenant path and
                is **WIP** (gated on TD-127/130/131).
            embedding_mode: ``memory`` (embedded, in-RAM fp32) or ``cold`` (service,
                SQ8). Only ``memory`` runs against the embedded path today.
            data_dir: Override the embedded data directory (default: per-repo
                ``.victor`` dir under the project).
            binary_path: Explicit ``proximadb-server`` binary path (embedded only).
            graph: Pre-built ``ProximaDBGraph`` (or compatible). Injecting this
                bypasses all bootstrap — used by tests to drive the real adapter
                against an in-memory fake client without a server binary.
            client: Pre-built ProximaDB client (used with an injected graph for
                node/edge deletion which ``ProximaDBGraph`` does not expose).
            cpg_store: Optional Tier-B fragment store. By default, statement
                nodes and CFG/CDG/DDG edges are persisted under ``data_dir``.
            record_collection: Injected ProximaRecord collection used by tests.
                Production creates the collection on the shared connection.
        """
        self._project_path = Path(project_path).resolve() if project_path else None
        self._repo = repo or repo_id_from_path(self._project_path)
        self._graph_id = graph_id_for_repo(self._repo)
        self._server_url = server_url
        self._embedding_mode = ProximaEmbeddingMode.coerce(embedding_mode)
        self._binary_path = binary_path

        if data_dir is not None:
            self._data_dir = Path(data_dir)
        elif self._project_path is not None:
            self._data_dir = self._project_path / ".victor" / "proximadb"
        else:
            self._data_dir = Path.cwd() / ".victor" / "proximadb"

        self._graph = graph
        self._client = client
        self._conn = None  # shared ProximaRepoConnection (embedded path)
        self._initialized = graph is not None

        # Durable Tier-A authority: graph properties and the embedding occupy one
        # ProximaRecord. ORION nodes are a rebuildable traversal projection applied
        # only after the record commit succeeds (TD-12).
        self._vector_dim = 384
        self._symbol_collection_name = f"{self._repo}_codegraph_records"
        self._symbol_collection: Any = record_collection
        self._record_nodes: Dict[str, GraphNode] = {}
        self._record_vectors: Dict[str, List[float]] = {}
        self._record_node_ids: set[str] = set()

        # In-memory sidecars for facts ProximaDBGraph does not yet persist
        # natively (relational Tier-C work lands with TD-127). Idempotent and
        # safe to rebuild.
        self._file_mtimes: Dict[str, float] = {}
        self._file_hashes: Dict[str, str] = {}
        self._subgraph_cache: Dict[str, Subgraph] = {}

        # Tier A is the ORION graph above. Tier B is deliberately a separate,
        # durable fragment index so statements and intra-procedural edges never
        # inflate default graph scans/traversals. It is initialized lazily only
        # when a CPG write or explicit dataflow drilldown occurs.
        self._cpg_store = cpg_store or CpgFragmentStore(self._data_dir / "cpg_fragments.sqlite3")

    @property
    def repo_root(self) -> Optional[Path]:
        """Resolved project root this store indexes (used for cache scoping)."""
        return self._project_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        if self._initialized and self._graph is not None:
            return
        if not is_proxima_available():
            raise ProximaUnavailableError(
                "proximadb_sdk is not installed; cannot use the 'proxima' graph "
                "backend. Keep codebase_graph_store=sqlite or install proximadb."
            )

        # Graph node/edge operations are served over REST (the gRPC client has no
        # graph RPCs), so the client must use the REST protocol explicitly —
        # otherwise it auto-selects gRPC and 404s against the REST port.
        if self._server_url:
            # Multi-tenant service path — WIP until TD-127/130/131 merge.
            from proximadb_sdk.graph import ProximaDBGraph
            from proximadb_sdk.unified_client import ProximaDBClient

            logger.warning(
                "ProximaGraphStore service mode (server_url=%s) is WIP and gated on "
                "ProximaDB TD-127/130/131; embedded mode is the supported path.",
                self._server_url,
            )
            self._client = ProximaDBClient(url=self._server_url, protocol="rest")
            try:
                await asyncio.to_thread(self._client.create_graph, self._graph_id)
            except Exception as exc:  # pragma: no cover - depends on live server
                logger.debug("create_graph(%s) noop/failed: %s", self._graph_id, exc)
            self._graph = ProximaDBGraph(self._client, self._graph_id)
        else:
            # Embedded path: share ONE instance per repo with the embedding
            # provider so the graph and its vectors are co-located (TD-11/12).
            self._conn = await ProximaRepoConnection.acquire(
                self._data_dir, binary_path=self._binary_path
            )
            self._client = self._conn.client
            self._graph = await self._conn.graph(self._graph_id)

        self._initialized = True

    async def close(self) -> None:
        await self._cpg_store.close()
        if self._conn is not None:
            await self._conn.release()
            self._conn = None
        self._client = None
        self._graph = None
        self._symbol_collection = None
        self._record_nodes.clear()
        self._record_vectors.clear()
        self._record_node_ids.clear()
        self._initialized = False

    async def _ensure(self) -> Any:
        if self._graph is None:
            await self.initialize()
        return self._graph

    # ------------------------------------------------------------------
    # Conversion: victor <-> proximadb_sdk graph types
    # ------------------------------------------------------------------
    def _to_proxima_node(self, node: GraphNode) -> Any:
        from proximadb_sdk.graph import GraphNode as PGNode

        props: Dict[str, Any] = {}
        for key in _NODE_FIELD_KEYS:
            value = getattr(node, key, None)
            if value is not None:
                props[key] = value
        if node.metadata:
            props["metadata"] = dict(node.metadata)
        # Note: embedding_ref is intentionally dropped — correlation is by oid
        # (node_id == vector record id), so the bridge column is retired here.
        return PGNode(id=node.node_id, labels=[node.type], properties=props)

    def _from_proxima_node(self, pnode: Any) -> GraphNode:
        props = dict(getattr(pnode, "properties", {}) or {})
        labels = list(getattr(pnode, "labels", []) or [])
        node_type = labels[0] if labels else str(props.get("type", "symbol"))
        metadata = props.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        line = props.get("line")
        end_line = props.get("end_line")
        return GraphNode(
            node_id=getattr(pnode, "id", ""),
            type=node_type,
            name=str(props.get("name", "")),
            file=str(props.get("file", props.get("file_path", ""))),
            line=int(line) if isinstance(line, (int, float)) else None,
            end_line=int(end_line) if isinstance(end_line, (int, float)) else None,
            lang=props.get("lang"),
            signature=props.get("signature"),
            docstring=props.get("docstring"),
            parent_id=props.get("parent_id"),
            metadata=metadata,
            ast_kind=props.get("ast_kind"),
            scope_id=props.get("scope_id"),
            statement_type=props.get("statement_type"),
            requirement_id=props.get("requirement_id"),
            visibility=props.get("visibility"),
        )

    def _to_proxima_edge(self, edge: GraphEdge) -> Any:
        from proximadb_sdk.graph import GraphEdge as PGEdge

        return PGEdge(
            id=f"{edge.src}|{edge.type}|{edge.dst}",
            from_node=edge.src,
            to_node=edge.dst,
            edge_type=edge.type,
            properties=dict(edge.metadata or {}),
            weight=edge.weight,
        )

    def _from_proxima_edge(self, pedge: Any) -> GraphEdge:
        return GraphEdge(
            src=getattr(pedge, "from_node", ""),
            dst=getattr(pedge, "to_node", ""),
            type=getattr(pedge, "edge_type", ""),
            weight=getattr(pedge, "weight", None),
            metadata=dict(getattr(pedge, "properties", {}) or {}),
        )

    @staticmethod
    def _sorted_edges(edges: List[GraphEdge]) -> List[GraphEdge]:
        # Drop malformed/empty edges (the REST graph API can return a single
        # blank edge for a node with no neighbors) and match the
        # SqliteGraphStore/MemoryGraphStore ordering for parity.
        valid = [e for e in edges if e.src and e.dst and e.type]
        return sorted(valid, key=lambda e: (e.src, e.dst, e.type))

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        node_list = list(nodes)
        hot_nodes = [node for node in node_list if node.type != "statement"]
        cold_nodes = [node for node in node_list if node.type == "statement"]
        if hot_nodes:
            graph = await self._ensure()
            pending_nodes = [self._node_for_record(node, has_embedding=False) for node in hot_nodes]
            await self._write_node_records(pending_nodes)
            await self._apply_node_projection(graph, pending_nodes)
        if cold_nodes:
            await self._cpg_store.upsert_nodes(cold_nodes)

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        edge_list = list(edges)
        hot_edges = [edge for edge in edge_list if not EdgeType.is_ccg_edge(edge.type)]
        cold_edges = [edge for edge in edge_list if EdgeType.is_ccg_edge(edge.type)]
        if hot_edges:
            graph = await self._ensure()
            payload = [self._to_proxima_edge(edge) for edge in hot_edges]
            await asyncio.to_thread(graph.batch_create_edges, payload)
        if cold_edges:
            await self._cpg_store.upsert_edges(cold_edges)

    async def update_node_metadata(self, node_id: str, metadata: Dict[str, Any]) -> None:
        """Atomically merge metadata while preserving the record's vector.

        Used by the indexing pipeline; no-op if the node is unknown.
        """
        node = await self.get_node_by_id(node_id)
        if node is None:
            return
        merged = dict(node.metadata or {})
        merged.update(metadata or {})
        has_embedding = bool(merged.get("has_embedding"))
        vector = self._record_vectors.get(node_id)
        if has_embedding and vector is None:
            raise RuntimeError(
                f"Cannot preserve vector for {node_id}: committed ProximaRecord is not cached"
            )
        committed = self._node_for_record(
            node,
            has_embedding=has_embedding,
            metadata=merged,
        )
        await self._write_node_records([committed], {node_id: vector} if vector else None)
        graph = await self._ensure()
        await self._apply_node_projection(graph, [committed])

    async def _symbol_records(self) -> Any:
        """Get/create the correlated ProximaRecord collection on the shared instance.

        Each record owns the node properties and vector under one ``oid``. ORION
        consumes a projection of those properties; it is not a second authority.
        """
        if self._symbol_collection is not None:
            return self._symbol_collection
        if self._conn is None:
            return None
        self._symbol_collection = await self._conn.get_or_create_collection(
            self._symbol_collection_name, dimension=self._vector_dim
        )
        return self._symbol_collection

    def _node_for_record(
        self,
        node: GraphNode,
        *,
        has_embedding: bool,
        metadata: Dict[str, Any] | None = None,
    ) -> GraphNode:
        merged = dict(node.metadata or {})
        if not has_embedding:
            merged.pop("has_embedding", None)
            merged.pop("content_version", None)
            merged.pop("embedding_ref", None)
        if metadata:
            merged.update(metadata)
        merged["has_embedding"] = has_embedding
        return replace(node, embedding_ref=None, metadata=merged)

    def _to_proxima_record(
        self, node: GraphNode, vector: List[float] | None = None
    ) -> Dict[str, Any]:
        """Build the one durable graph-property + vector envelope for a symbol."""
        values = list(vector) if vector is not None else [0.0] * self._vector_dim
        if len(values) != self._vector_dim:
            raise ValueError(
                f"Embedding for {node.node_id} has {len(values)} dimensions; "
                f"expected {self._vector_dim}"
            )
        props: Dict[str, Any] = {
            "record_kind": "graph_node",
            "graph_id": self._graph_id,
            "type": node.type,
            "metadata": dict(node.metadata or {}),
            "has_embedding": bool((node.metadata or {}).get("has_embedding")),
        }
        for key in _NODE_FIELD_KEYS:
            value = getattr(node, key, None)
            if value is not None:
                props[key] = value
        return {"id": node.node_id, "vector": values, "props": props}

    @staticmethod
    def _validate_record_write(result: Any, expected: int) -> None:
        """Fail closed unless Proxima confirms every requested record write."""
        if isinstance(result, dict):
            failed = int(result.get("failed_count", result.get("failed", 0)) or 0)
            succeeded = int(
                result.get("inserted_count", result.get("success", expected if not failed else 0))
                or 0
            )
            errors = result.get("errors") or []
        else:
            failed = int(getattr(result, "failed", 0) or 0)
            succeeded = int(getattr(result, "success", 0) or 0)
            errors = getattr(result, "errors", None) or []
        if failed or succeeded != expected:
            detail = "; ".join(str(error) for error in errors) or (
                f"confirmed {succeeded} of {expected} records"
            )
            raise RuntimeError(f"Atomic ProximaRecord write failed: {detail}")

    async def _write_node_records(
        self,
        nodes: List[GraphNode],
        vectors: Dict[str, List[float]] | None = None,
    ) -> None:
        collection = await self._symbol_records()
        if collection is None:
            raise RuntimeError("ProximaDB atomic record collection is unavailable")
        payload = [
            self._to_proxima_record(node, (vectors or {}).get(node.node_id)) for node in nodes
        ]
        result = await collection.insert_records(payload)
        self._validate_record_write(result, len(payload))
        for node, record in zip(nodes, payload):
            self._record_nodes[node.node_id] = node
            self._record_vectors[node.node_id] = list(record["vector"])
            self._record_node_ids.add(node.node_id)

    async def _apply_node_projection(self, graph: Any, nodes: List[GraphNode]) -> None:
        """Apply the rebuildable ORION projection after the record commit."""
        payload = [self._to_proxima_node(node) for node in nodes]
        result = await asyncio.to_thread(graph.batch_create_nodes, payload)
        if isinstance(result, dict) and not result.get("success", False):
            raise RuntimeError(f"ORION node projection failed: {result.get('errors') or result}")

    async def upsert_node_record(
        self,
        node_id: str,
        embedding: List[float],
        *,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Atomically replace a symbol's graph properties and vector.

        The ProximaRecord write is the durable commit. ORION is updated afterward
        as a rebuildable projection; projection failure propagates so retry repairs
        it without ever producing a graph-only or vector-only authoritative state.
        """
        node = self._record_nodes.get(node_id) or await self.get_node_by_id(node_id)
        if node is None:
            raise KeyError(f"Unknown graph node: {node_id}")
        vector = list(embedding)
        committed = self._node_for_record(node, has_embedding=True, metadata=metadata)
        await self._write_node_records([committed], {node_id: vector})
        graph = await self._ensure()
        await self._apply_node_projection(graph, [committed])

    async def semantic_search(
        self,
        query_vector: List[float],
        *,
        top_k: int = 10,
        filters: Dict[str, Any] | None = None,
    ) -> List[GraphNode]:
        """Vector seed → graph node: search co-indexed symbol vectors, resolve oids.

        Returns the graph nodes whose vectors are nearest to ``query_vector``,
        ordered by similarity. This is the semantic-seed half of hybrid
        seed→expand, served from the one correlated collection (vector hit → node
        is identity, not a join). Empty when no vectors are co-indexed.
        """
        collection = await self._symbol_records()
        if collection is None:
            return []
        search_filters = dict(filters or {})
        search_filters["has_embedding"] = True
        hits = await collection.search(query_vector, top_k=top_k, filters=search_filters)
        nodes: List[GraphNode] = []
        for hit in hits or []:
            oid = hit.get("id") if isinstance(hit, dict) else None
            if not oid:
                continue
            node = await self.get_node_by_id(oid)
            if node is not None:
                nodes.append(node)
        return nodes

    # ------------------------------------------------------------------
    # Reads / traversal
    # ------------------------------------------------------------------
    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[Iterable[str]] = None,
        *,
        direction: GraphTraversalDirection = "both",
        max_depth: int = 1,
    ) -> List[GraphEdge]:
        if direction not in {"out", "in", "both"}:
            raise ValueError(f"Unsupported graph traversal direction: {direction}")
        if max_depth < 1:
            return []

        requested_types = list(edge_types) if edge_types is not None else []
        if requested_types:
            cold_types = [
                edge_type for edge_type in requested_types if EdgeType.is_ccg_edge(edge_type)
            ]
            hot_types = [
                edge_type for edge_type in requested_types if not EdgeType.is_ccg_edge(edge_type)
            ]
            if cold_types and not hot_types:
                return await self._cpg_store.get_neighbors(
                    node_id,
                    cold_types,
                    direction=direction,
                    max_depth=max_depth,
                )
            if cold_types:
                hot_result, cold_result = await asyncio.gather(
                    self._get_hot_neighbors(
                        node_id,
                        hot_types,
                        direction=direction,
                        max_depth=max_depth,
                    ),
                    self._cpg_store.get_neighbors(
                        node_id,
                        cold_types,
                        direction=direction,
                        max_depth=max_depth,
                    ),
                )
                return self._sorted_edges([*hot_result, *cold_result])

        # An unfiltered traversal is intentionally Tier A only. Cold CPG data
        # must be requested through explicit CFG/CDG/DDG edge filters.
        return await self._get_hot_neighbors(
            node_id,
            requested_types or None,
            direction=direction,
            max_depth=max_depth,
        )

    async def _get_hot_neighbors(
        self,
        node_id: str,
        edge_types: Optional[Iterable[str]] = None,
        *,
        direction: GraphTraversalDirection = "both",
        max_depth: int = 1,
    ) -> List[GraphEdge]:
        graph = await self._ensure()
        allowed = list(edge_types) if edge_types else None
        raw = await asyncio.to_thread(
            graph.get_neighbors,
            node_id,
            allowed,
            direction=direction,
            max_depth=max_depth,
        )
        return self._sorted_edges([self._from_proxima_edge(e) for e in raw])

    async def find_nodes(
        self,
        *,
        name: str | None = None,
        type: str | None = None,
        file: str | None = None,
    ) -> List[GraphNode]:
        if type == "statement":
            return await self._cpg_store.find_nodes(name=name, file=file)
        graph = await self._ensure()
        raw = await asyncio.to_thread(lambda: graph.find_nodes(name=name, type=type, file=file))
        return [self._from_proxima_node(n) for n in raw]

    async def search_symbols(
        self,
        query: str,
        *,
        limit: int = 20,
        symbol_types: Optional[Iterable[str]] = None,
    ) -> List[GraphNode]:
        graph = await self._ensure()
        types = list(symbol_types) if symbol_types else None
        raw = await asyncio.to_thread(graph.search_symbols, query, limit, types)
        return [self._from_proxima_node(n) for n in raw]

    async def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        if node_id in self._record_nodes:
            return self._record_nodes[node_id]
        graph = await self._ensure()
        raw = await asyncio.to_thread(graph.get_node_by_id, node_id)
        if raw is not None:
            return self._from_proxima_node(raw)
        return await self._cpg_store.get_node_by_id(node_id)

    async def get_all_nodes(self) -> List[GraphNode]:
        graph = await self._ensure()
        raw = await asyncio.to_thread(lambda: graph.get_all_nodes(include_internal=True))
        by_id = {node.node_id: node for node in (self._from_proxima_node(n) for n in raw)}
        by_id.update(self._record_nodes)
        nodes = list(by_id.values())
        nodes.sort(key=lambda n: (n.file, n.line or 0, n.name))
        return nodes

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        graph = await self._ensure()
        raw = await asyncio.to_thread(graph.get_nodes_by_file, file)
        by_id = {node.node_id: node for node in (self._from_proxima_node(n) for n in raw)}
        by_id.update(
            (node_id, node) for node_id, node in self._record_nodes.items() if node.file == file
        )
        nodes = list(by_id.values())
        nodes.sort(key=lambda n: (n.line or 0, n.name))
        return nodes

    async def get_all_edges(self) -> List[GraphEdge]:
        graph = await self._ensure()
        raw = await asyncio.to_thread(graph.get_all_edges)
        return self._sorted_edges([self._from_proxima_edge(e) for e in raw])

    # ------------------------------------------------------------------
    # File mtime / staleness (in-memory sidecar until Tier-C relational facts)
    # ------------------------------------------------------------------
    async def update_file_mtime(
        self, file: str, mtime: float, content_hash: str | None = None
    ) -> None:
        self._file_mtimes[str(file)] = mtime
        if content_hash is None:
            self._file_hashes.pop(str(file), None)
        else:
            self._file_hashes[str(file)] = content_hash

    async def get_file_hashes(self, files: List[str]) -> Dict[str, str]:
        return {f: self._file_hashes[str(f)] for f in files if str(f) in self._file_hashes}

    async def get_stale_files(self, file_mtimes: Dict[str, float]) -> List[str]:
        return [
            file
            for file, current in file_mtimes.items()
            if self._file_mtimes.get(str(file), float("-inf")) < current
        ]

    async def get_indexed_files(self) -> List[str]:
        return sorted(self._file_mtimes.keys())

    # ------------------------------------------------------------------
    # Deletes
    # ------------------------------------------------------------------
    async def delete_by_file(self, file: str) -> None:
        await self._ensure()
        nodes = await self.get_nodes_by_file(file)
        cold_nodes = await self._cpg_store.get_nodes_by_file(file)
        for node in nodes:
            await self._delete_node(node.node_id)
        await self._cpg_store.delete_by_file(file)
        self._file_mtimes.pop(str(file), None)
        self._file_hashes.pop(str(file), None)
        # Invalidate any cached subgraphs anchored on deleted nodes.
        deleted_ids = {n.node_id for n in [*nodes, *cold_nodes]}
        for sg_id in [
            sg_id for sg_id, sg in self._subgraph_cache.items() if sg.anchor_node_id in deleted_ids
        ]:
            self._subgraph_cache.pop(sg_id, None)

    async def _delete_node(self, node_id: str) -> None:
        # Delete the authoritative record as one entity, then its ORION projection.
        collection = await self._symbol_records()
        if collection is None:
            raise RuntimeError("ProximaDB atomic record collection is unavailable")
        deleted = await collection.delete([node_id])
        if deleted is not None and int(deleted) < 1:
            logger.debug("record %s was already absent", node_id)
        self._record_nodes.pop(node_id, None)
        self._record_vectors.pop(node_id, None)
        self._record_node_ids.discard(node_id)
        if self._client is not None:
            delete_node = getattr(self._client, "delete_node", None)
            if delete_node is not None:
                try:
                    await asyncio.to_thread(delete_node, node_id=node_id, graph_id=self._graph_id)
                except Exception as exc:  # pragma: no cover - depends on live server
                    logger.debug("delete_node(%s) failed: %s", node_id, exc)

    async def _clear_symbol_records(self) -> None:
        """Delete the unified symbol-record authority and invalidate projections."""
        if self._conn is not None:
            deleted = await self._conn.embedded_db.delete_collection(self._symbol_collection_name)
            if not deleted:
                raise RuntimeError(
                    f"Failed to delete ProximaDB collection {self._symbol_collection_name}"
                )
            self._conn.forget_collection(self._symbol_collection_name)
        elif self._symbol_collection is not None:
            clear = getattr(self._symbol_collection, "clear", None)
            if clear is not None:
                await clear()
            elif self._record_node_ids:
                await self._symbol_collection.delete(sorted(self._record_node_ids))
        self._symbol_collection = None
        self._record_nodes.clear()
        self._record_vectors.clear()
        self._record_node_ids.clear()

    async def delete_by_repo(self, clear_embeddings: bool = False) -> None:
        # Graph properties and embeddings are one record now; clearing only one
        # modality is structurally impossible. The compatibility flag is ignored.
        del clear_embeddings
        await self._clear_symbol_records()
        if self._client is None:
            await self._cpg_store.clear()
            self._file_mtimes.clear()
            self._file_hashes.clear()
            self._subgraph_cache.clear()
            return
        delete_graph = getattr(self._client, "delete_graph", None)
        if delete_graph is not None:
            try:
                await asyncio.to_thread(delete_graph, self._graph_id)
            except Exception as exc:  # pragma: no cover
                logger.debug("delete_graph(%s) failed: %s", self._graph_id, exc)
        create_graph = getattr(self._client, "create_graph", None)
        if create_graph is not None:
            try:
                await asyncio.to_thread(create_graph, self._graph_id)
            except Exception:  # pragma: no cover
                pass
        await self._cpg_store.clear()
        self._file_mtimes.clear()
        self._file_hashes.clear()
        self._subgraph_cache.clear()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    async def stats(self) -> Dict[str, Any]:
        graph = await self._ensure()
        try:
            raw = await asyncio.to_thread(graph.get_stats)
        except Exception as exc:  # pragma: no cover
            logger.debug("get_stats failed: %s", exc)
            raw = {}
        raw_stats = raw if isinstance(raw, dict) else {}
        cold_stats = await self._cpg_store.stats()
        return {
            "backend": "proxima",
            "repo": self._repo,
            "graph_id": self._graph_id,
            "embedding_mode": self._embedding_mode.value,
            "service_mode_wip": bool(self._server_url),
            **raw_stats,
            "tier_a_nodes": int(raw_stats.get("node_count", raw_stats.get("nodes", 0))),
            "tier_a_edges": int(raw_stats.get("edge_count", raw_stats.get("edges", 0))),
            "tier_b_nodes": cold_stats["nodes"],
            "tier_b_edges": cold_stats["edges"],
            "tier_b_files": cold_stats["files"],
        }

    # ------------------------------------------------------------------
    # v5 CCG helpers (filter over node properties)
    # ------------------------------------------------------------------
    async def get_nodes_by_statement_type(
        self, statement_type: str, *, file: str | None = None
    ) -> List[GraphNode]:
        return await self._cpg_store.get_nodes_by_statement_type(statement_type, file=file)

    async def get_nodes_by_requirement(self, requirement_id: str) -> List[GraphNode]:
        hot = [n for n in await self.get_all_nodes() if n.requirement_id == requirement_id]
        cold = await self._cpg_store.get_nodes_by_requirement(requirement_id)
        return [*hot, *cold]

    async def get_nodes_by_scope(self, scope_id: str) -> List[GraphNode]:
        return await self._cpg_store.get_nodes_by_scope(scope_id)

    # ------------------------------------------------------------------
    # Subgraph cache (in-memory; computed via multi-hop traversal)
    # ------------------------------------------------------------------
    async def get_subgraph(
        self,
        anchor_node_id: str,
        radius: int = 2,
        edge_types: Iterable[str] | None = None,
    ) -> Subgraph:
        edge_type_list = list(edge_types) if edge_types else []
        cache_key = f"{anchor_node_id}:{radius}:{','.join(sorted(edge_type_list))}"
        cached = self._subgraph_cache.get(cache_key)
        if cached is not None:
            return cached
        result = await self.multi_hop_traverse(
            [anchor_node_id], max_hops=radius, edge_types=edge_types or None
        )
        subgraph = Subgraph(
            subgraph_id=cache_key,
            anchor_node_id=anchor_node_id,
            radius=radius,
            edge_types=edge_type_list,
            node_ids=[n.node_id for n in result.nodes],
            edges=result.edges,
            node_count=len(result.nodes),
        )
        self._subgraph_cache[cache_key] = subgraph
        return subgraph

    async def cache_subgraph(self, subgraph: Subgraph) -> None:
        self._subgraph_cache[subgraph.subgraph_id] = subgraph

    async def invalidate_subgraph(self, subgraph_id: str) -> None:
        self._subgraph_cache.pop(subgraph_id, None)

    # ------------------------------------------------------------------
    # Multi-hop traversal
    # ------------------------------------------------------------------
    async def multi_hop_traverse(
        self,
        start_node_ids: List[str],
        max_hops: int = 2,
        edge_types: Iterable[str] | None = None,
        max_nodes: int = 100,
    ) -> GraphQueryResult:
        return await self.multi_hop_traverse_parallel(
            start_node_ids,
            max_hops=max_hops,
            edge_types=edge_types,
            max_nodes=max_nodes,
            max_workers=1,
        )

    async def get_neighbors_batch(
        self,
        node_ids: List[str],
        *,
        edge_types: Iterable[str] | None = None,
        direction: GraphTraversalDirection = "out",
    ) -> Dict[str, List[GraphEdge]]:
        if not node_ids:
            return {}
        results = await asyncio.gather(
            *[
                self.get_neighbors(node_id, edge_types=edge_types, direction=direction, max_depth=1)
                for node_id in node_ids
            ],
            return_exceptions=True,
        )
        out: Dict[str, List[GraphEdge]] = {}
        for node_id, result in zip(node_ids, results):
            if isinstance(result, Exception):
                logger.warning("get_neighbors(%s) failed: %s", node_id, result)
                out[node_id] = []
            else:
                out[node_id] = result
        return out

    async def multi_hop_traverse_parallel(
        self,
        start_node_ids: List[str],
        max_hops: int = 2,
        edge_types: Iterable[str] | None = None,
        max_nodes: int = 100,
        max_workers: int = 4,
    ) -> GraphQueryResult:
        start_time = time.time()
        if not start_node_ids:
            return GraphQueryResult(nodes=[], edges=[], query="parallel_traversal")

        visited: set[str] = set(start_node_ids)
        frontier: List[str] = list(start_node_ids)
        all_edges: Dict[tuple[str, str, str], GraphEdge] = {}
        all_nodes: Dict[str, GraphNode] = {}

        for node_id in start_node_ids:
            node = await self.get_node_by_id(node_id)
            if node:
                all_nodes[node_id] = node

        for _hop in range(max_hops):
            if not frontier or len(all_nodes) >= max_nodes:
                break
            batch_size = max(1, max_workers)
            neighbor_map: Dict[str, List[GraphEdge]] = {}
            for i in range(0, len(frontier), batch_size):
                batch = frontier[i : i + batch_size]
                neighbor_map.update(
                    await self.get_neighbors_batch(batch, edge_types=edge_types, direction="out")
                )

            next_frontier: set[str] = set()
            for neighbors in neighbor_map.values():
                for edge in neighbors:
                    all_edges.setdefault((edge.src, edge.dst, edge.type), edge)
                    neighbor_id = edge.dst
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.add(neighbor_id)
                        if len(all_nodes) < max_nodes:
                            node = await self.get_node_by_id(neighbor_id)
                            if node:
                                all_nodes[neighbor_id] = node
            frontier = list(next_frontier)

        return GraphQueryResult(
            nodes=list(all_nodes.values()),
            edges=self._sorted_edges(list(all_edges.values())),
            query="parallel_traversal",
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    # ------------------------------------------------------------------
    # Lazy iteration
    # ------------------------------------------------------------------
    async def iter_nodes(
        self,
        *,
        batch_size: int = 100,
        name: str | None = None,
        type: str | None = None,
        file: str | None = None,
    ) -> AsyncIterator[List[GraphNode]]:
        if name is not None or type is not None or file is not None:
            nodes = await self.find_nodes(name=name, type=type, file=file)
        else:
            nodes = await self.get_all_nodes()
        for i in range(0, len(nodes), batch_size):
            yield nodes[i : i + batch_size]

    async def iter_edges(
        self,
        *,
        batch_size: int = 100,
        edge_types: Iterable[str] | None = None,
    ) -> AsyncIterator[List[GraphEdge]]:
        requested = list(edge_types) if edge_types is not None else []
        cold_types = [edge_type for edge_type in requested if EdgeType.is_ccg_edge(edge_type)]
        hot_types = [edge_type for edge_type in requested if not EdgeType.is_ccg_edge(edge_type)]

        # Unfiltered iteration is Tier A only; Tier B must always be explicit.
        if not requested or hot_types:
            allowed = set(hot_types) if requested else None
            edges = await self.get_all_edges()
            if allowed is not None:
                edges = [edge for edge in edges if edge.type in allowed]
            for i in range(0, len(edges), batch_size):
                yield edges[i : i + batch_size]

        if cold_types:
            async for batch in self._cpg_store.iter_edges(
                batch_size=batch_size, edge_types=cold_types
            ):
                yield batch

    async def iter_neighbors(
        self,
        node_id: str,
        *,
        batch_size: int = 50,
        edge_types: Iterable[str] | None = None,
        direction: GraphTraversalDirection = "out",
    ) -> AsyncIterator[List[GraphEdge]]:
        edges = await self.get_neighbors(
            node_id, edge_types=edge_types, direction=direction, max_depth=1
        )
        for i in range(0, len(edges), batch_size):
            yield edges[i : i + batch_size]
