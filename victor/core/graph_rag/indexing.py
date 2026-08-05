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

"""Graph indexing pipeline - G-Indexing stage of Graph RAG.

This module implements the first stage of the Graph RAG pipeline:
building a rich code graph with embeddings for semantic retrieval.

Based on GraphRAG methodology:
1. Build symbol graph from code structure
2. Build CCG (optional) for statement-level granularity
3. Generate embeddings for nodes
4. Cache subgraphs for efficient retrieval

ARCHITECTURAL SEAM (2026-04-29):
Language-specific edge detection should be handled by per-language
handlers implementing the LanguageEdgeHandler protocol. See
language_handlers.py for the protocol and handlers/ directory
for implementations.

MIGRATION STATUS:
- Core indexing pipeline: Uses language handlers via get_edge_handler()
- Fallback: Legacy _build_calls_edges() for unsupported languages
- Target: All edge detection moved to victor-coding language plugins

Future sessions: Add handlers for new languages in handlers/ directory,
then move them to victor-coding package as part of language plugins.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import fnmatch
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
)

from victor.core.graph_rag.exclude_patterns import is_path_excluded

if TYPE_CHECKING:
    from victor.storage.graph.protocol import GraphStoreProtocol

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

# Fallback only — preferred path is TreeSitterAnalysisProtocol from the
# capability registry (see GraphIndexingPipeline._analysis_provider). The maps
# below are consulted when no enhanced provider is registered or when an
# enhanced provider's call raises.
_TREE_SITTER_LANGUAGE_MODULES = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
}

_TREE_SITTER_DEFINITION_TYPES = {
    "python": {
        "function_definition",
        "class_definition",
        "async_function_definition",
        "decorated_definition",
    },
    "javascript": {
        "function_declaration",
        "function_expression",
        "class_declaration",
        "class_expression",
        "method_definition",
    },
    "typescript": {
        "function_declaration",
        "function_expression",
        "class_declaration",
        "class_expression",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
    },
    "go": {
        "function_declaration",
        "type_declaration",
        "method_declaration",
    },
    "rust": {
        "function_item",
        "struct_item",
        "impl_item",
        "trait_item",
    },
    "java": {
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
    },
}

_TREE_SITTER_NODE_TYPE_MAP = {
    "function_definition": "function",
    "async_function_definition": "function",
    "class_definition": "class",
    "decorated_definition": "function",
    "function_declaration": "function",
    "function_expression": "function",
    "arrow_function": "function",
    "class_declaration": "class",
    "class_expression": "class",
    "method_definition": "method",
    "interface_declaration": "interface",
    "type_alias_declaration": "type_alias",
    "go_function_declaration": "function",
    "go_method_declaration": "method",
    "go_type_declaration": "type",
    "function_item": "function",
    "struct_item": "struct",
    "impl_item": "impl",
    "trait_item": "trait",
    "java_method_declaration": "method",
    "java_class_declaration": "class",
    "java_interface_declaration": "interface",
    "enum_declaration": "enum",
}

_TREE_SITTER_IDENTIFIER_TYPES = {
    "identifier",
    "property_identifier",
    "field_identifier",
    "type_identifier",
}


# Node types produced by _TREE_SITTER_NODE_TYPE_MAP that represent a
# free-standing or member function. When a provider symbol's parent is
# class-like, _provider_symbols_to_graph_nodes promotes any node currently
# tagged with one of these types to ``method`` so the graph distinguishes
# methods from top-level functions.
_FUNCTION_LIKE_NODE_TYPES = frozenset({"function", "function_item"})


def _module_node_id(file_path: str) -> str:
    """Stable node_id for the synthetic module node of a source file.

    The prefix collision-proofs the id against symbol nodes that happen to be
    named ``"module"`` and hashed at the same (file, line) coordinate. Kept as
    a module-level helper so both the indexer and the import resolver agree on
    the encoding without sharing state.
    """
    try:
        from victor_codegraph import CodeSymbolType, stable_symbol_key

        return stable_symbol_key("repository", CodeSymbolType.FILE, file_path, None, file_path)
    except Exception:
        # Mixed-install fallback during the v2 migration.
        import hashlib

        return hashlib.sha256(f"module:{file_path}".encode()).hexdigest()[:16]


# Import at runtime for use in non-type-checked contexts
def _get_graph_types() -> tuple[type, type]:
    """Lazy import of graph types."""
    from victor.storage.graph.protocol import GraphEdge, GraphNode

    return GraphNode, GraphEdge


@dataclass
class GraphIndexStats:
    """Statistics from graph indexing operation.

    Attributes:
        files_processed: Number of files successfully processed
        files_skipped: Number of files skipped (too large, excluded, etc.)
        nodes_created: Number of graph nodes created
        edges_created: Number of graph edges created
        ccg_nodes_created: Number of CCG (statement-level) nodes
        ccg_edges_created: Number of CCG edges
        embeddings_generated: Number of embeddings generated
        subgraphs_cached: Number of subgraphs cached
        module_metrics_computed: Number of module metric rows computed
        processing_time_seconds: Total processing time
        error_count: Number of errors encountered
        errors: List of error messages
    """

    files_processed: int = 0
    files_skipped: int = 0
    files_unchanged: int = 0
    files_deleted: int = 0
    nodes_created: int = 0
    edges_created: int = 0
    ccg_nodes_created: int = 0
    ccg_edges_created: int = 0
    embeddings_generated: int = 0
    subgraphs_cached: int = 0
    module_metrics_computed: int = 0
    cross_file_calls_resolved: int = 0
    cross_file_relationships_resolved: int = 0
    imports_resolved: int = 0
    # Files that tried the enhanced TreeSitterAnalysisProtocol provider but
    # had to fall back to the hardcoded extraction path (because the provider
    # raised or returned None for a language it claimed to support).
    provider_fallbacks: int = 0
    processing_time_seconds: float = 0.0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary.

        Returns:
            Dictionary representation of stats
        """
        return {
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "files_unchanged": self.files_unchanged,
            "files_deleted": self.files_deleted,
            "nodes_created": self.nodes_created,
            "edges_created": self.edges_created,
            "ccg_nodes_created": self.ccg_nodes_created,
            "ccg_edges_created": self.ccg_edges_created,
            "embeddings_generated": self.embeddings_generated,
            "subgraphs_cached": self.subgraphs_cached,
            "module_metrics_computed": self.module_metrics_computed,
            "cross_file_calls_resolved": self.cross_file_calls_resolved,
            "cross_file_relationships_resolved": self.cross_file_relationships_resolved,
            "imports_resolved": self.imports_resolved,
            "provider_fallbacks": self.provider_fallbacks,
            "processing_time_seconds": self.processing_time_seconds,
            "error_count": self.error_count,
            "errors": self.errors[:10],  # Limit errors in output
        }


@dataclass
class ParseResult:
    """Parsed output for one source file produced by a worker thread.

    Designed to cross the thread→async boundary via asyncio.Queue.  All fields
    are plain Python values (no locks, no event-loop objects) so they are safe
    to hand off from a ThreadPoolExecutor worker to the async consumer.

    Attributes:
        file_path:    Absolute path to the source file.
        language:     Detected language, or None when the file should be skipped.
        symbol_nodes: Symbol-level graph nodes extracted from the AST.
        ccg_nodes:    Code Context Graph nodes (CFG/CDG/DDG) if CCG is enabled.
        ccg_edges:    Code Context Graph edges corresponding to ccg_nodes.
        vanished:     True when the file disappeared between discovery and parse.
        error:        Set when an unexpected exception occurred during parsing.
    """

    file_path: Path
    language: Optional[str]
    symbol_nodes: List[Any] = field(default_factory=list)
    ccg_nodes: List[Any] = field(default_factory=list)
    ccg_edges: List[Any] = field(default_factory=list)
    vanished: bool = False
    error: Optional[Exception] = None
    # True when this file tried the enhanced TreeSitterAnalysisProtocol
    # provider but had to fall back to the hardcoded extraction path.
    provider_fallback: bool = False
    # Raw import strings extracted by the analysis provider (e.g. "import os",
    # "from typing import List"). Resolved to IMPORTS edges between module
    # nodes by _resolve_imports() after all files have been persisted, so
    # cross-file targets bind correctly. Empty when the provider is the null
    # stub or extract_imports raises.
    raw_imports: List[str] = field(default_factory=list)


async def run_indexing_with_lock(
    root_path: Path,
    operation: Callable[[], Awaitable[_T]],
    timeout_seconds: float = 300.0,
) -> _T:
    """Serialize direct graph-indexing operations across processes.

    Deep init, manual graph rebuilds, and watcher refreshes all write to the
    shared project graph tables. This helper makes ad hoc indexing callers use
    the same cross-process lock as the background refresh path.
    """
    from victor.core.indexing.index_lock import IndexLockRegistry

    lock_registry = IndexLockRegistry.get_instance()
    path_lock = await lock_registry.acquire_lock(
        root_path.resolve(),
        timeout_seconds=timeout_seconds,
    )
    async with path_lock:
        return await operation()


class GraphIndexingPipeline:
    """Pipeline for building and indexing code graphs.

    This class orchestrates the graph indexing process:
    1. Discover source files
    2. Extract symbols and build graph
    3. Build CCG if enabled
    4. Generate embeddings if enabled
    5. Cache subgraphs for fast retrieval

    Attributes:
        graph_store: Graph store for persisting nodes and edges
        config: Indexing configuration
    """

    def __init__(
        self,
        graph_store: GraphStoreProtocol,
        config: Any,  # GraphIndexConfig
    ) -> None:
        """Initialize the indexing pipeline.

        Args:
            graph_store: Graph store for persistence
            config: GraphIndexConfig instance
        """
        self.graph_store = graph_store
        self.config = config
        self._ccg_builder = None
        self._files_to_process: Set[str] = set()
        # Content hashes computed during incremental planning (manifest
        # contract), persisted alongside mtimes after each file is written so
        # the next run can verify mtime-only churn as not-stale.
        self._pending_file_hashes: Dict[str, str] = {}
        # Raw call records captured during per-file parsing, resolved against
        # a project-wide name index in _resolve_cross_file_calls() after all
        # nodes have been persisted.
        #   (caller_node_id, callee_name, receiver_type, is_method_call)
        # receiver_type is set for method calls obj.method() when the language
        # plugin could infer obj's static type (Rust today). is_method_call
        # is True for dot-dispatch syntax even when receiver_type couldn't be
        # inferred -- the resolver uses this to drop method calls with no
        # inferable type (name-only fallback would bind them to unrelated
        # user-defined methods with the same leaf name).
        self._pending_call_records: List[Tuple[str, str, Optional[str], bool]] = []
        # Buffered non-CALLS structural relationships (INHERITS / IMPLEMENTS /
        # COMPOSITION) emitted by language handlers. Source is always a
        # class-like symbol in the current file so it resolves immediately
        # against name_to_ids; target is a leaf name that may live in another
        # file and is resolved in _resolve_cross_file_relationships() against
        # a project-wide class/interface index after persistence.
        #   (source_node_id, target_name, edge_type)
        self._pending_relationship_records: List[Tuple[str, str, str]] = []
        # Buffered (source_file, raw_import_string, language) records. Resolved
        # to IMPORTS edges between synthesized module nodes after persistence
        # by _resolve_imports() — keeping source unresolved here lets a single
        # project-wide module index handle both same-package and cross-package
        # targets uniformly.
        self._pending_import_records: List[Tuple[str, str, str]] = []
        # Repository-wide victor-codegraph snapshot prepared once per indexing
        # run.  Per-file edge projection reads from this snapshot instead of
        # reparsing each file, and the final relation pass retains import-aware
        # cross-file resolution rather than degrading to the legacy name index.
        self._codegraph_repository: Any = None
        self._codegraph_owned_symbol_ids: Set[str] = set()
        self._codegraph_owned_files: Set[str] = set()
        # Parser cache is thread-local so each ThreadPoolExecutor worker gets
        # its own parser instance (tree-sitter parsers are not thread-safe to share).
        self._thread_local = threading.local()
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # Lazily-resolved TreeSitterAnalysisProtocol provider. Resolved once
        # per pipeline; if the registry's enhanced provider is replaced after
        # construction the pipeline keeps its original reference.
        self._analysis_resolved = False
        self._analysis_provider: Any = None
        self._analysis_enhanced: bool = False

        if config.enable_ccg:
            try:
                from victor.core.indexing import CodeContextGraphBuilder

                self._ccg_builder = CodeContextGraphBuilder(graph_store)
            except ImportError:
                logger.warning("CCG builder not available, skipping CCG construction")

    async def index_repository(
        self,
        root_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> GraphIndexStats:
        """Index a repository into the graph store.

        Args:
            root_path: Optional root path override (uses config.root_path if None)
            progress_callback: Called after each file — (done, total, filename).
            status_callback: Called at phase boundaries — (human-readable message).

        Returns:
            GraphIndexStats with indexing results
        """

        def _status(msg: str) -> None:
            if status_callback:
                status_callback(msg)

        root = (root_path or self.config.root_path).resolve()
        stats = GraphIndexStats()
        start_time = time.time()

        logger.info(f"Starting graph indexing for {root}")

        # Initialize graph store
        await self.graph_store.initialize()

        # Self-heal: a concurrent process (the graph-watch background
        # refresher, a crashed prior run, or manual DROP) can leave
        # project.db without the graph_node / graph_edge / graph_file_mtime
        # tables that subsequent calls depend on. Force one more schema
        # creation pass right before we touch the DB so the user doesn't
        # have to remember --force to recover. SQLite's CREATE TABLE IF
        # NOT EXISTS is cheap when tables are already present.
        ensure_schema = getattr(self.graph_store, "_ensure_schema", None)
        if callable(ensure_schema):
            try:
                ensure_schema()
            except Exception as exc:
                logger.debug("Defensive schema re-ensure skipped: %s", exc)

        # Fresh run: discard any call records buffered by a prior invocation.
        self._pending_call_records.clear()
        self._pending_relationship_records.clear()
        self._pending_import_records.clear()
        self._codegraph_repository = None
        self._codegraph_owned_symbol_ids.clear()
        self._codegraph_owned_files.clear()

        # Resolve the analysis provider — and let it finish language-plugin
        # discovery — BEFORE the parallel parse burst. Resolution is lazy and
        # discovery imports ~30 grammar wheels; when the first resolution
        # happens inside a worker thread, every file parsed while discovery
        # is still importing sees supports_language() == False and silently
        # degrades to the legacy extraction path (no raw imports, poorer
        # symbols). Observed: 51/92 vscode-victor files lost their IMPORTS
        # edges to this race.
        _status("Resolving analysis provider…")
        self._get_analysis_provider()

        # Force mode: clear all existing data before rebuilding
        if not self.config.incremental:
            logger.info("Force rebuild: clearing all existing graph data")
            _status("Clearing existing graph data…")
            clear_embeddings = not self.config.enable_embeddings
            await self.graph_store.delete_by_repo(clear_embeddings=clear_embeddings)

        # Discover source files
        _status("Discovering source files…")
        files = await self._discover_files(root)
        logger.info("Discovered %d indexable source files", len(files))
        discovered_files = files

        _status(f"Planning: {len(files)} files found — checking mtimes…")
        planning_stats = await self._prepare_incremental_work(files, root)
        self._merge_stats(stats, planning_stats)

        if self.config.incremental:
            files = [
                file_path
                for file_path in files
                if self._graph_file_key(file_path, root) in self._files_to_process
            ]
            logger.info(
                "Incremental graph indexing plan for %s: %d changed, %d unchanged, %d deleted",
                root,
                len(files),
                planning_stats.files_unchanged,
                planning_stats.files_deleted,
            )

        # Parse one consistent repository snapshot only when at least one file
        # will be indexed.  The snapshot still includes unchanged discovered
        # files: a changed caller may import an unchanged target, and an
        # unchanged caller may acquire a newly-added target.
        if files:
            await asyncio.to_thread(
                self._prepare_codegraph_repository,
                root,
                discovered_files,
            )

        # Process files in batches — parse in parallel, write serially.
        # Emit progress_callback(0, total, "") immediately so the UI can set
        # the progress bar total before the first file is written.
        total_files = len(files)
        _status(f"Indexing {total_files} files…")
        if progress_callback and total_files:
            progress_callback(0, total_files, "")

        files_done = 0
        for i in range(0, total_files, self.config.chunk_size):
            batch = files[i : i + self.config.chunk_size]
            batch_stats = await self._process_batch(
                batch,
                total_files=total_files,
                done_offset=files_done,
                progress_callback=progress_callback,
            )
            self._merge_stats(stats, batch_stats)
            files_done += (
                batch_stats.files_processed + batch_stats.files_skipped + batch_stats.files_deleted
            )

        graph_changed = bool(
            stats.files_processed
            or stats.files_deleted
            or stats.nodes_created
            or stats.edges_created
            or stats.ccg_nodes_created
            or stats.ccg_edges_created
        )

        # Generate embeddings if enabled
        if self.config.enable_embeddings and graph_changed:
            embedding_stats = await self._generate_embeddings()
            self._merge_stats(stats, embedding_stats)

        # Cache subgraphs if enabled
        if self.config.enable_subgraph_cache and graph_changed:
            subgraph_stats = await self._cache_subgraphs()
            self._merge_stats(stats, subgraph_stats)

        # Resolve cross-file CALLS edges before module metrics — the module
        # analyzer derives module adjacency from cross-file graph_edge rows, so
        # CALLS resolution must run first or graph_module_metric stays empty.
        if graph_changed:
            (
                cg_calls,
                cg_relationships,
                cg_imports,
            ) = await self._resolve_codegraph_repository_relations()

            resolved = await self._resolve_cross_file_calls(root)
            stats.cross_file_calls_resolved = cg_calls + resolved
            stats.edges_created += cg_calls + resolved

            # Same pass for INHERITS/IMPLEMENTS/COMPOSITION — base classes,
            # interfaces, and composed types often live in another file, so
            # per-file resolution alone leaves these edges unresolved.
            rel_resolved = await self._resolve_cross_file_relationships(root)
            stats.cross_file_relationships_resolved = cg_relationships + rel_resolved
            stats.edges_created += cg_relationships + rel_resolved

            # IMPORTS edges between module nodes. The synthetic module nodes
            # were already persisted by _inject_module_node; we just emit
            # the cross-module edges here.
            imports_resolved = await self._resolve_imports(root)
            stats.imports_resolved = cg_imports + imports_resolved
            stats.edges_created += cg_imports + imports_resolved

        if getattr(self.config, "enable_module_metrics", True) and graph_changed:
            stats.module_metrics_computed = self._refresh_module_metrics(root)

        stats.processing_time_seconds = time.time() - start_time

        logger.info(
            f"Indexing complete: {stats.files_processed} files, "
            f"{stats.nodes_created} nodes, {stats.edges_created} edges"
        )

        return stats

    def _prepare_codegraph_repository(self, root_path: Path, files: List[Path]) -> None:
        """Prepare one filtered semantic snapshot for this indexing run.

        ``victor-codegraph`` remains a soft dependency.  Any import or parse
        failure leaves the snapshot unset and all existing provider/handler
        resolvers continue unchanged.
        """
        from victor.core.graph_rag.codegraph_repository import prepare_repository_snapshot

        allowed_files = (self._canonical_file_str(path) for path in files)
        self._codegraph_repository = prepare_repository_snapshot(root_path, allowed_files)

    async def _resolve_codegraph_repository_relations(self) -> Tuple[int, int, int]:
        """Project import-aware v2 repository relations into Graph-RAG edges.

        Returns ``(calls, structural_relationships, imports)``.  Only relations
        resolved by the repository pass are projected here; in-file relations
        remain the responsibility of ``_build_calls_edges``.  Legacy buffers
        are removed only for sources whose v2 identities matched the Graph-RAG
        nodes, preserving the soft fallback for mixed or absent installations.
        """
        repository = self._codegraph_repository
        if repository is None or not self._codegraph_owned_files:
            return 0, 0, 0

        from victor.core.graph_rag.codegraph_repository import project_resolved_relations
        from victor.storage.graph.edge_types import EdgeType

        projection = project_resolved_relations(repository)
        _, GraphEdge = _get_graph_types()
        edges = [
            GraphEdge(src=src, dst=dst, type=EdgeType(edge_type))
            for src, dst, edge_type in projection.edges
        ]

        if edges:
            async with self._graph_store_write_batch():
                await self.graph_store.upsert_edges(edges)

        owned_ids = self._codegraph_owned_symbol_ids
        self._pending_call_records = [
            record for record in self._pending_call_records if record[0] not in owned_ids
        ]
        self._pending_relationship_records = [
            record for record in self._pending_relationship_records if record[0] not in owned_ids
        ]
        owned_files = self._codegraph_owned_files
        self._pending_import_records = [
            record
            for record in self._pending_import_records
            if self._canonical_file_str(record[0]) not in owned_files
        ]
        logger.info(
            "victor-codegraph repository resolution: %d CALLS, %d structural, %d IMPORTS edges",
            projection.calls,
            projection.relationships,
            projection.imports,
        )
        return projection.calls, projection.relationships, projection.imports

    async def _resolve_cross_file_calls(self, root_path: Path) -> int:
        """Resolve buffered CALLS records against project-wide name indices.

        Per-file edge building can only resolve callees defined in the same
        file. This pass runs after all nodes have been persisted, builds two
        project-wide indices, and emits CALLS edges via UPSERT (so same-file
        matches don't duplicate anything written earlier):

        1. *Impl-type index* — ``(impl_type, method_name) -> [node_id]``
           derived from methods inside ``impl T`` blocks (parent_id linkage).
           Used when a record carries ``receiver_type``; a hit here is
           precise enough to bypass the fanout cap.
        2. *Leaf-name index* — ``name -> [node_id]`` over callable symbol
           types (``function``, ``method``, ``impl``). Used as the fallback
           when no receiver type is known or the receiver-typed lookup
           found no candidates. Subject to
           ``config.cross_file_call_max_fanout`` (default 25) so common
           leaf names (``new``, ``default``, ``from``) don't inflate the
           graph with noise.
        """
        if not self._pending_call_records:
            return 0

        max_fanout = int(getattr(self.config, "cross_file_call_max_fanout", 25))

        from victor.core.database import ProjectDatabaseManager

        db = ProjectDatabaseManager(root_path)
        conn = db._get_raw_connection()

        # Leaf-name index (function/method/impl) plus per-node file map so
        # name-only resolution can prefer same-file candidates. Rust function
        # names are module-scoped, so a local helper `fn inputs(t)` called
        # inside a test module should bind to the same-file definition rather
        # than fanning out to every other module's `fn inputs` (audit finding:
        # 11 `inputs` and 21 `parse` definitions in proximaDB, all binding
        # under the cap).
        #
        # Exclude functions whose parent is a *trait impl* (`impl <Trait> for
        # <Type>`). Trait method names like `drop`, `fmt`, `eq`, `hash`,
        # `clone`, `from` are invoked by the compiler or by typed method
        # dispatch, never by plain function calls. Without this filter,
        # `drop(x)` (the stdlib std::mem::drop) was fanning out to all 16
        # user `impl Drop for T { fn drop }` impls in proximaDB. Detected
        # by the impl-node signature: trait impls always contain ` for `
        # (e.g. "impl Drop for SearchState {"), inherent impls don't
        # (e.g. "impl Foo {").
        name_index: Dict[str, List[str]] = {}
        node_file_index: Dict[str, str] = {}
        for row in conn.execute(
            "SELECT m.name, m.node_id, m.file FROM graph_node m "
            "LEFT JOIN graph_node impl ON m.parent_id = impl.node_id "
            "AND impl.type = 'impl' "
            "WHERE m.name IS NOT NULL "
            "AND m.type IN ('function','method','impl') "
            "AND (impl.signature IS NULL OR impl.signature NOT LIKE '% for %')"
        ):
            name, node_id, file = row[0], row[1], (row[2] if len(row) > 2 else None)
            name_index.setdefault(name, []).append(node_id)
            if file is not None:
                node_file_index[node_id] = file

        # Impl-type index. The schema doesn't carry an explicit impl_type
        # column, but methods inside `impl T` have parent_id pointing at the
        # impl_item node (type='impl', name='T'). One join is enough.
        impl_method_index: Dict[Tuple[str, str], List[str]] = {}
        for row in conn.execute(
            "SELECT impl.name AS impl_type, m.name AS method_name, m.node_id "
            "FROM graph_node m "
            "JOIN graph_node impl ON m.parent_id = impl.node_id "
            "WHERE impl.type = 'impl' "
            "AND m.name IS NOT NULL "
            "AND m.type IN ('function','method')"
        ):
            impl_method_index.setdefault((row[0], row[1]), []).append(row[2])

        _, GraphEdge = _get_graph_types()
        from victor.storage.graph.edge_types import EdgeType

        edges: List[Any] = []
        skipped_fanout = 0
        unresolved = 0
        receiver_typed_hits = 0
        receiver_typed_unresolved = 0
        method_calls_dropped = 0
        for record in self._pending_call_records:
            # Tolerate legacy tuple shapes for callers that haven't migrated.
            if len(record) == 4:
                caller_id, callee_name, receiver_type, is_method_call = record
            elif len(record) == 3:
                caller_id, callee_name, receiver_type = record
                is_method_call = receiver_type is not None
            else:
                caller_id, callee_name = record  # type: ignore[misc]
                receiver_type = None
                is_method_call = False

            # Receiver-typed lookup. When receiver_type is set, the plugin has
            # told us the call targets a specific impl T::method. A hit binds
            # exactly there and bypasses the fanout cap (the binding is
            # unambiguous). A miss means T is not in our graph -- almost
            # always a stdlib type like Vec/HashMap or an external crate.
            if receiver_type is not None:
                typed_candidates = impl_method_index.get((receiver_type, callee_name))
                if typed_candidates:
                    receiver_typed_hits += 1
                    for callee_id in typed_candidates:
                        if callee_id == caller_id:
                            continue
                        edges.append(GraphEdge(src=caller_id, dst=callee_id, type=EdgeType.CALLS))
                else:
                    receiver_typed_unresolved += 1
                continue

            # Method-syntax dot-dispatch with no inferable receiver type.
            # Name-only fallback would bind to unrelated user-defined methods
            # with the same leaf name (observed: `collect`, `iter`, `clone`
            # each had ~10-20 user impls below the fanout cap, fanning out
            # to all of them). Drop instead. Plain function calls and path
            # calls keep the name-only fallback below.
            if is_method_call:
                method_calls_dropped += 1
                continue

            # Name-only path (plain function call or path call `Mod::func()`).
            candidates = name_index.get(callee_name)
            if not candidates:
                unresolved += 1
                continue

            # Module scoping: if any candidate lives in the caller's file,
            # restrict to those. Plain function calls in Rust resolve to
            # module-scoped names; cross-file binding only makes sense when
            # there's no local match. Same-file preference dramatically cuts
            # fanout for common helper names like `inputs`, `parse`, `new`
            # that appear in many test modules.
            caller_file = node_file_index.get(caller_id)
            if caller_file is not None:
                same_file = [c for c in candidates if node_file_index.get(c) == caller_file]
                if same_file:
                    candidates = same_file

            if len(candidates) > max_fanout:
                skipped_fanout += 1
                continue
            for callee_id in candidates:
                if callee_id == caller_id:
                    continue
                edges.append(GraphEdge(src=caller_id, dst=callee_id, type=EdgeType.CALLS))

        if edges:
            async with self._graph_store_write_batch():
                await self.graph_store.upsert_edges(edges)

        n = len(edges)
        records_count = len(self._pending_call_records)
        self._pending_call_records.clear()
        logger.info(
            "Cross-file CALLS resolution: %d edges emitted from %d call records "
            "(%d receiver-typed hits, %d receiver-typed unresolved [external/stdlib], "
            "%d method calls dropped [no inferable receiver], "
            "%d name-only unresolved, %d skipped due to fanout>%d)",
            n,
            records_count,
            receiver_typed_hits,
            receiver_typed_unresolved,
            method_calls_dropped,
            unresolved,
            skipped_fanout,
            max_fanout,
        )
        return n

    async def _resolve_cross_file_relationships(self, root_path: Path) -> int:
        """Resolve buffered INHERITS/IMPLEMENTS/COMPOSITION records.

        For each (source_id, target_name, edge_type) record, look up the
        target by leaf name in a project-wide index restricted to class-like
        node types. Most relationship targets (base classes, interfaces,
        composed types) are class/interface/struct symbols, so we deliberately
        exclude callable types from the index — that prevents an
        ``INHERITS`` from accidentally binding to a same-named function.

        Subject to the same ``cross_file_call_max_fanout`` cap as CALLS, so
        a target name with many class definitions (rare but possible) does
        not explode the edge count.
        """
        if not self._pending_relationship_records:
            return 0

        max_fanout = int(getattr(self.config, "cross_file_call_max_fanout", 25))

        from victor.core.database import ProjectDatabaseManager

        db = ProjectDatabaseManager(root_path)
        conn = db._get_raw_connection()

        # Class-like name index — restricts target binding to types that
        # actually make sense as INHERITS/IMPLEMENTS/COMPOSITION targets.
        class_name_index: Dict[str, List[str]] = {}
        for row in conn.execute(
            "SELECT name, node_id FROM graph_node "
            "WHERE name IS NOT NULL "
            "AND type IN ('class','interface','struct','impl','trait','type','type_alias','enum')"
        ):
            class_name_index.setdefault(row[0], []).append(row[1])

        _, GraphEdge = _get_graph_types()

        edges: List[Any] = []
        unresolved = 0
        skipped_fanout = 0
        for src_id, target_name, edge_type in self._pending_relationship_records:
            candidates = class_name_index.get(target_name)
            if not candidates:
                unresolved += 1
                continue
            if len(candidates) > max_fanout:
                skipped_fanout += 1
                continue
            for dst_id in candidates:
                if dst_id == src_id:
                    continue
                edges.append(GraphEdge(src=src_id, dst=dst_id, type=edge_type))

        if edges:
            async with self._graph_store_write_batch():
                await self.graph_store.upsert_edges(edges)

        n = len(edges)
        records_count = len(self._pending_relationship_records)
        self._pending_relationship_records.clear()
        logger.info(
            "Cross-file relationship resolution: %d edges emitted from %d records "
            "(%d unresolved targets, %d skipped due to fanout>%d)",
            n,
            records_count,
            unresolved,
            skipped_fanout,
            max_fanout,
        )
        return n

    async def _resolve_imports(self, root_path: Path) -> int:
        """Resolve buffered import records into IMPORTS edges between modules.

        Pipeline:
          1. Parse each raw import string into one or more module names via
             the language's :class:`LanguageImportResolver` strategy
             (``import_resolvers.py``; languages without a registered
             strategy log and skip).
          2. Resolve each module name to a project file through the same
             strategy (e.g. Python ``foo.bar.baz`` → ``foo/bar/baz.py``,
             Rust ``<src-dir>::foo::bar`` → ``foo/bar.rs``). Names that
             don't resolve to a project file are stdlib/third-party and get
             skipped — we only graph intra-project edges.
          3. Compute synthetic module node ids on both sides and emit
             IMPORTS edges. Same-file self-imports are dropped.

        Returns the number of IMPORTS edges emitted. Behaves as a no-op when
        the buffer is empty (e.g. the analysis provider was the null stub
        for every file).
        """
        if not self._pending_import_records:
            return 0

        from victor.core.graph_rag.import_resolvers import (
            ImportResolverRegistry,
            LanguageImportResolver,
        )

        # One strategy instance per language per run — strategies may cache
        # project-layout state (e.g. the Rust workspace crate map) for the
        # duration of the run.
        resolvers: Dict[str, Optional[LanguageImportResolver]] = {}

        # Cache resolutions across the batch so the filesystem isn't hit
        # repeatedly for popular imports (``victor.core.database`` shows up
        # in hundreds of files).
        resolution_cache: Dict[Tuple[str, str], Optional[Path]] = {}

        # Filter targets to files that actually got indexed in this run. A
        # file can resolve on disk but still lack a module node — common
        # cases: excluded by pattern, unsupported language, parse failure.
        # Without this filter we'd emit IMPORTS edges pointing at module
        # nodes that don't exist, which dangles graph_edge rows even though
        # they don't trigger the parent_id FK directly.
        #
        # Path semantics: _make_module_node stores the canonical
        # repo-relative form in both the ``file`` column and the node_id
        # hash input. We do the same here so set membership and hash
        # derivation both agree.
        from victor.core.database import ProjectDatabaseManager

        db = ProjectDatabaseManager(root_path)
        conn = db._get_raw_connection()
        indexed_module_files: Set[str] = {
            row[0]
            for row in conn.execute(
                "SELECT file FROM graph_node WHERE type = 'module' AND file IS NOT NULL"
            )
        }

        _, GraphEdge = _get_graph_types()
        from victor.storage.graph.edge_types import EdgeType

        edges: List[Any] = []
        emitted_pairs: Set[Tuple[str, str]] = set()
        unresolved = 0
        unsupported = 0
        unindexed_target = 0
        for src_file, raw, language in self._pending_import_records:
            src_canonical = self._canonical_file_str(src_file)
            if src_canonical not in indexed_module_files:
                # Source vanished between extraction and resolution — skip
                # to keep the IMPORTS endpoints anchored at real nodes.
                continue
            if language not in resolvers:
                resolvers[language] = ImportResolverRegistry.create(language)
            resolver = resolvers[language]
            if resolver is None:
                unsupported += 1
                continue
            module_names = resolver.parse(raw, src_file, root_path)
            for mod in module_names:
                key = (mod, language)
                if key in resolution_cache:
                    target_path = resolution_cache[key]
                else:
                    target_path = resolver.resolve(mod, root_path)
                    resolution_cache[key] = target_path
                if target_path is None:
                    unresolved += 1
                    continue
                target_canonical = self._canonical_file_str(target_path)
                if target_canonical == src_canonical:
                    continue  # self-import (``from . import x`` from __init__)
                if target_canonical not in indexed_module_files:
                    unindexed_target += 1
                    continue
                pair = (src_canonical, target_canonical)
                if pair in emitted_pairs:
                    continue
                emitted_pairs.add(pair)
                edges.append(
                    GraphEdge(
                        src=_module_node_id(src_canonical),
                        dst=_module_node_id(target_canonical),
                        type=EdgeType.IMPORTS,
                    )
                )

        if edges:
            async with self._graph_store_write_batch():
                await self.graph_store.upsert_edges(edges)

        n = len(edges)
        records_count = len(self._pending_import_records)
        self._pending_import_records.clear()
        logger.info(
            "Import resolution: %d IMPORTS edges emitted from %d import records "
            "(%d unresolved external/stdlib, %d unsupported-language, "
            "%d targets not indexed)",
            n,
            records_count,
            unresolved,
            unsupported,
            unindexed_target,
        )
        return n

    def _refresh_module_metrics(self, root_path: Path) -> int:
        """Refresh module-level graph metrics after graph writes."""
        try:
            from victor.core.analysis.module_analyzer import ModuleAnalyzer

            analyzer = ModuleAnalyzer(project_path=root_path)
            metrics = analyzer.compute_all()
            if metrics:
                analyzer.persist(metrics)
            logger.info("Refreshed %d graph module metric rows", len(metrics))
            return len(metrics)
        except Exception as exc:
            logger.warning("Failed to refresh graph module metrics: %s", exc)
            return 0

    async def _discover_files(self, root_path: Path) -> List[Path]:
        """Discover source files to index.

        Args:
            root_path: Root directory to search

        Returns:
            List of file paths to index
        """
        files: List[Tuple[str, Path]] = []
        exclude_patterns = self.config.exclude_patterns
        include_patterns = self.config.include_patterns

        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Check file size
            try:
                if file_path.stat().st_size > self.config.max_file_size_bytes:
                    continue
            except OSError:
                continue

            rel_path = file_path.relative_to(root_path)
            path_str = str(rel_path)

            # Check exclude patterns
            if is_path_excluded(file_path, root_path, exclude_patterns):
                continue

            # Check include patterns (if specified)
            if include_patterns:
                included = any(
                    fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path_str, f"**/{pattern}")
                    for pattern in include_patterns
                )
                if not included:
                    continue

            language = self._detect_language(file_path)
            if not self._is_indexable_language(language):
                continue

            files.append((language, file_path))

        return [
            file_path
            for _, file_path in sorted(
                files,
                key=lambda item: (
                    item[0],
                    item[1].relative_to(root_path).as_posix(),
                ),
            )
        ]

    @staticmethod
    def _compute_file_hash(file_path: Path) -> Optional[str]:
        """Content hash of a source file per the victor-codegraph manifest contract.

        Prefers the shared implementation (identical decode + hash across every
        consumer of the code graph); falls back to sha-256 of the raw bytes,
        which equals the contract hash for valid UTF-8 files (decode→encode is
        the identity there).
        """
        try:
            from victor_codegraph import content_hash, read_source_text

            text = read_source_text(file_path)
            return content_hash(text) if text is not None else None
        except ImportError:
            pass
        except Exception:
            return None
        try:
            import hashlib

            return hashlib.sha256(file_path.read_bytes()).hexdigest()
        except OSError:
            return None

    async def _verify_stale_by_content(
        self,
        stale_files: Set[str],
        current_files: Dict[str, Path],
        file_mtimes: Dict[str, float],
    ) -> Set[str]:
        """Drop mtime-stale files whose content hash is unchanged (manifest contract).

        touch, branch flip-flops, and checkouts bump mtimes without changing
        content; reparsing those is pure waste. Files verified unchanged get
        their stored mtime refreshed (so the fast path skips them next run) and
        are removed from the stale set. Hashes for genuinely changed files are
        buffered in ``_pending_file_hashes`` for persistence after processing.
        """
        if not stale_files:
            return stale_files

        get_hashes = getattr(self.graph_store, "get_file_hashes", None)
        stored_hashes: Dict[str, str] = {}
        if callable(get_hashes):
            try:
                stored_hashes = await get_hashes(sorted(stale_files))
            except Exception as exc:
                logger.debug("get_file_hashes unavailable, mtime-only staleness: %s", exc)

        verified_unchanged: Set[str] = set()
        for file_key in sorted(stale_files):
            file_path = current_files.get(file_key)
            if file_path is None:
                continue
            digest = self._compute_file_hash(file_path)
            if digest is None:
                continue
            self._pending_file_hashes[file_key] = digest
            if stored_hashes.get(file_key) == digest:
                verified_unchanged.add(file_key)
                await self._update_store_mtime(file_key, file_mtimes.get(file_key, 0.0), digest)

        if verified_unchanged:
            logger.info(
                "Content-hash verification: %d mtime-changed files are content-identical, skipped",
                len(verified_unchanged),
            )
        return stale_files - verified_unchanged

    async def _update_store_mtime(
        self, file: str, mtime: float, content_hash: Optional[str]
    ) -> None:
        """Persist mtime + content hash, tolerating stores without the hash kwarg."""
        try:
            await self.graph_store.update_file_mtime(file, mtime, content_hash=content_hash)
        except TypeError:
            await self.graph_store.update_file_mtime(file, mtime)

    def _file_hash_for(self, file_path: Path) -> Optional[str]:
        """Hash to persist with a processed file's mtime.

        Incremental runs computed it during planning; force/full runs compute
        it here so the NEXT incremental run has hashes to verify against.
        """
        key = self._graph_file_key(file_path, self.config.root_path.resolve())
        digest = self._pending_file_hashes.get(key)
        if digest is None:
            digest = self._compute_file_hash(file_path)
        return digest

    async def _prepare_incremental_work(
        self,
        files: List[Path],
        root_path: Optional[Path] = None,
    ) -> GraphIndexStats:
        """Prepare an incremental indexing plan based on file mtimes and deletions.

        Two-layer staleness: the mtime fast path (no reads for unchanged
        mtimes), then content-hash verification of mtime-changed files via the
        victor-codegraph manifest contract, so mtime-only churn never reparses.
        """
        root = (root_path or self.config.root_path).resolve()
        self._files_to_process = {self._graph_file_key(file_path, root) for file_path in files}
        self._pending_file_hashes = {}
        if not self.config.incremental:
            return GraphIndexStats()

        stats = GraphIndexStats()
        current_files = {self._graph_file_key(file_path, root): file_path for file_path in files}
        file_mtimes: Dict[str, float] = {}
        for path_str, file_path in list(current_files.items()):
            try:
                file_mtimes[path_str] = file_path.stat().st_mtime
            except FileNotFoundError:
                current_files.pop(path_str, None)
                await self._handle_vanished_file(file_path)
            except OSError:
                current_files.pop(path_str, None)
                await self._handle_vanished_file(file_path)

        stale_files = set(await self.graph_store.get_stale_files(file_mtimes))
        stale_files = await self._verify_stale_by_content(stale_files, current_files, file_mtimes)
        indexed_files = await self._get_indexed_files()
        deleted_files = sorted(indexed_files - set(current_files))

        for file_path in deleted_files:
            await self.graph_store.delete_by_file(file_path)
            stats.files_deleted += 1

        files_to_process: Set[str] = set()
        for file_path in sorted(stale_files):
            await self.graph_store.delete_by_file(file_path)
            files_to_process.add(file_path)

        stats.files_unchanged = max(0, len(current_files) - len(files_to_process))
        self._files_to_process = files_to_process
        return stats

    def _graph_file_key(self, file_path: Path, root_path: Path) -> str:
        """Return the storage key used for project-local graph file paths."""
        try:
            resolved = file_path.resolve(strict=False)
        except OSError:
            resolved = file_path.absolute()
        try:
            resolved_root = root_path.resolve(strict=False)
        except OSError:
            resolved_root = root_path.absolute()
        try:
            return resolved.relative_to(resolved_root).as_posix()
        except ValueError:
            return str(resolved)

    async def _get_indexed_files(self) -> Set[str]:
        """Return currently indexed files, preferring store-native mtime metadata."""
        get_indexed_files = getattr(self.graph_store, "get_indexed_files", None)
        if callable(get_indexed_files):
            try:
                indexed_files = await get_indexed_files()
                return {str(file_path) for file_path in indexed_files}
            except Exception as exc:
                logger.debug("Graph store get_indexed_files fallback: %s", exc)

        indexed_nodes = await self.graph_store.get_all_nodes()
        return {node.file for node in indexed_nodes if node.file}

    def _is_indexable_language(self, language: str) -> bool:
        """Return whether the built-in indexer can extract useful graph data for a language."""
        symbol_languages = {
            "python",
            "javascript",
            "typescript",
            "go",
            "rust",
            "java",
            "c",
            "cpp",
        }
        if language in symbol_languages:
            return True

        if not self.config.enable_ccg:
            return False

        try:
            from victor.core.indexing.ccg_builder import SUPPORTED_CCG_LANGUAGES

            return language in SUPPORTED_CCG_LANGUAGES
        except Exception:
            return False

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return (creating lazily) a thread pool for CPU-bound AST parsing."""
        if self._executor is None:
            workers = getattr(self.config, "parse_workers", 0) or None
            if workers is None:
                workers = min(os.cpu_count() or 4, 8)
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="ccg-parse"
            )
        return self._executor

    def _create_ts_parser(self, language: str) -> Optional[Any]:
        """Create a fresh tree-sitter parser for *language* (caller caches it)."""
        if language not in _TREE_SITTER_LANGUAGE_MODULES:
            return None
        try:
            import tree_sitter as ts

            module_name, func_name = _TREE_SITTER_LANGUAGE_MODULES[language]
            lang_module = __import__(module_name)
            lang_func = getattr(lang_module, func_name)
            lang_obj = lang_func()
            ts_language = (
                ts.Language(lang_obj) if not isinstance(lang_obj, ts.Language) else lang_obj
            )
            return ts.Parser(ts_language)
        except Exception:
            return None

    def _get_thread_loop(self) -> asyncio.AbstractEventLoop:
        """Return (creating once per thread) a persistent event loop for the caller thread.

        CCG build methods are declared `async def` but have no internal await points
        — they are CPU-only work.  Rather than spawning a new event loop per call
        (which orphans connections), each worker thread owns one loop that persists
        for the thread's lifetime.
        """
        if not hasattr(self._thread_local, "event_loop"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._thread_local.event_loop = loop
        return self._thread_local.event_loop

    def _get_thread_ccg_builder(self) -> Optional[Any]:
        """Return (creating once per thread) a CCG builder for the caller thread.

        CodeContextGraphBuilder stores _tree_sitter_parser / _tree_sitter_language
        as instance attributes and is therefore not safe to share across threads.
        Each worker thread keeps its own instance.  graph_store is passed for API
        compatibility but is never written during build_ccg_for_file().
        """
        if not hasattr(self._thread_local, "ccg_builder"):
            try:
                from victor.core.indexing import CodeContextGraphBuilder

                self._thread_local.ccg_builder = CodeContextGraphBuilder(self.graph_store)
            except ImportError:
                self._thread_local.ccg_builder = None
        return self._thread_local.ccg_builder

    def _build_ccg_inline(
        self, file_path: Path, language: Optional[str]
    ) -> Tuple[List[Any], List[Any]]:
        """Run the CCG builder for *file_path* on the current worker thread.

        Returns ``([], [])`` when CCG is disabled, the language is unknown,
        or any per-file exception occurs — the indexing pipeline tolerates
        missing CCG output silently because it's an enrichment layer, not a
        correctness requirement. Both extraction paths (TSA-fast and the
        legacy tree-sitter fallback) call this so CCG runs uniformly; the
        regression where CCG only ran on legacy-fallback files is fixed
        here.
        """
        if not (self.config.enable_ccg and language):
            return [], []
        builder = self._get_thread_ccg_builder()
        if builder is None:
            return [], []
        try:
            loop = self._get_thread_loop()
            return loop.run_until_complete(builder.build_ccg_for_file(file_path, language))
        except Exception as exc:
            logger.debug("Thread CCG failed for %s: %s", file_path, exc)
            return [], []

    def _get_analysis_provider(self) -> Tuple[Any, bool]:
        """Resolve the TreeSitterAnalysisProtocol provider once.

        Returns a (provider, enhanced) tuple. Both are cached after first
        call. Provider can be None when nothing is registered; enhanced is
        False when only the null stub is registered.
        """
        if self._analysis_resolved:
            return self._analysis_provider, self._analysis_enhanced
        try:
            from victor.core.capability_registry import CapabilityRegistry
            from victor.framework.vertical_protocols import TreeSitterAnalysisProtocol

            registry = CapabilityRegistry.get_instance()
            self._analysis_provider = registry.get(TreeSitterAnalysisProtocol)
            self._analysis_enhanced = registry.is_enhanced(TreeSitterAnalysisProtocol)
        except Exception:
            self._analysis_provider = None
            self._analysis_enhanced = False
        self._analysis_resolved = True
        return self._analysis_provider, self._analysis_enhanced

    def _make_module_node(self, file_path: Path, language: str) -> Any:
        """Synthesize a module-level GraphNode for *file_path*.

        Module nodes anchor file-scoped edges (CONTAINS for top-level
        symbols, IMPORTS for cross-file dependencies). The id is stable
        across runs via :func:`_module_node_id` so incremental re-indexing
        upserts cleanly instead of duplicating.

        Path normalization is critical: SqliteGraphStore canonicalizes the
        ``file`` column to repo-relative form but does *not* recompute
        node_id. If the id were derived from the absolute path and the
        resolver later looked up by the canonical relative path, the two
        wouldn't match and IMPORTS edges would silently dangle. So we
        canonicalize here and hash the canonical form — the resolver does
        the same.
        """
        GraphNode, _ = _get_graph_types()
        canonical_file = self._canonical_file_str(file_path)
        return GraphNode(
            node_id=_module_node_id(canonical_file),
            type="module",
            name=file_path.stem,
            file=canonical_file,
            line=0,
            lang=language,
            ast_kind="module",
        )

    def _canonical_file_str(self, file_path: Path | str) -> str:
        """Repo-relative canonical form mirroring SqliteGraphStore.

        Kept inline rather than importing the store's helper because the
        store may not have been initialized yet at parse time, and the
        rule is simple enough to duplicate without drift risk: resolve
        symlinks, take ``relative_to(project_root)`` when possible,
        otherwise fall back to the absolute string.
        """
        root = self.config.root_path.resolve()
        p = Path(file_path).expanduser()
        if not p.is_absolute():
            p = root / p
        try:
            resolved = p.resolve(strict=False)
        except OSError:
            resolved = p.absolute()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            return str(resolved)

    def _inject_module_node(
        self,
        symbol_nodes: List[Any],
        file_path: Path,
        language: str,
    ) -> List[Any]:
        """Prepend a module node and parent every top-level symbol to it.

        Top-level symbols (those that landed with ``parent_id is None``
        after extraction) inherit the module node as their parent so the
        existing CONTAINS emission in ``_build_symbol_edges`` produces
        module→symbol edges with no new code path. Existing parent_id
        values from nested-symbol resolution are preserved.

        Returns the augmented list. Safe on empty inputs — a file with
        only imports still gets a module node so cross-file IMPORTS
        targets can resolve to it.
        """
        module_node = self._make_module_node(file_path, language)
        for node in symbol_nodes:
            if node.parent_id is None:
                node.parent_id = module_node.node_id
        return [module_node, *symbol_nodes]

    def _provider_symbols_to_graph_nodes(
        self,
        symbols: List[Dict[str, Any]],
        file_path: Path,
        language: str,
    ) -> List[Any]:
        """Convert provider symbol dicts to GraphNode objects.

        Two-pass so parent_id can be resolved against the per-file (name,
        line) index built from this same batch:

        1. Build a GraphNode for every symbol dict, recording its
           ``(name, line_start)`` key alongside the dict's optional
           ``parent_symbol``/``parent_line`` hints from the provider.
        2. For each symbol that declared a parent, look up the parent's
           ``node_id`` via the index and stamp it on the child. When the
           parent claims to be class-like (``parent_is_class``), promote
           a function/def-typed child to ``method`` so the graph node
           taxonomy matches the legacy recursive extractor.

        Falls back to ``_TREE_SITTER_NODE_TYPE_MAP`` for normalizing
        ast_kind into the legacy node ``type`` taxonomy when the dict's
        own symbol_type can't be mapped directly.
        """
        import hashlib

        GraphNode, _ = _get_graph_types()
        file_str = str(file_path)

        # Pass 1: materialize nodes and index them by (name, line) so the
        # parent_symbol/parent_line hints we got from the provider can be
        # resolved to a stable node_id in pass 2.
        nodes: List[Any] = []
        index: Dict[Tuple[str, int], str] = {}
        # Keep symbol dict alongside its node so pass 2 doesn't re-walk
        # the input list looking up by index.
        annotated: List[Tuple[Any, Dict[str, Any]]] = []
        for sym in symbols:
            name = sym.get("name")
            line_start = sym.get("line_start")
            if not name or line_start is None:
                continue
            line_int = int(line_start)
            ast_kind = sym.get("ast_kind") or sym.get("symbol_type")
            node_type = _TREE_SITTER_NODE_TYPE_MAP.get(
                ast_kind or sym.get("symbol_type") or "unknown",
                sym.get("symbol_type") or "unknown",
            )
            node_id = (
                sym.get("symbol_id")
                or hashlib.sha256(f"{file_str}:{name}:{line_int}".encode()).hexdigest()[:16]
            )
            node = GraphNode(
                node_id=node_id,
                type=node_type,
                name=name,
                file=sym.get("file_path") or file_str,
                line=line_int,
                end_line=int(sym.get("line_end") or line_int),
                lang=language,
                signature=sym.get("signature"),
                docstring=sym.get("docstring"),
                visibility=sym.get("visibility"),
                ast_kind=ast_kind,
            )
            nodes.append(node)
            index[(name, line_int)] = node_id
            annotated.append((node, sym))

        # Pass 2: resolve parent_id from provider hints. The provider
        # already walked the AST to find the nearest enclosing scope, so
        # we just need to translate (parent_symbol, parent_line) into the
        # parent's node_id. Methods on class-like parents get promoted to
        # the ``method`` node type so the taxonomy distinguishes free
        # functions from class methods.
        for node, sym in annotated:
            direct_parent_id = sym.get("parent_id")
            if direct_parent_id and direct_parent_id != node.node_id:
                node.parent_id = direct_parent_id
                if sym.get("parent_is_class") and node.type in _FUNCTION_LIKE_NODE_TYPES:
                    node.type = "method"
                continue
            parent_name = sym.get("parent_symbol")
            parent_line = sym.get("parent_line")
            if not parent_name or parent_line is None:
                continue
            parent_id = index.get((parent_name, int(parent_line)))
            if parent_id is None or parent_id == node.node_id:
                continue
            node.parent_id = parent_id
            if sym.get("parent_is_class") and node.type in _FUNCTION_LIKE_NODE_TYPES:
                node.type = "method"

        # Sort parent-before-child so the SQL bulk insert satisfies the
        # parent_id FK constraint (the project DB enables PRAGMA
        # foreign_keys=ON). The provider emits classes before functions
        # regardless of source nesting, which would otherwise FK-fail for
        # patterns like ``def outer(): class Inner: pass`` where Inner's
        # parent (outer) lives in the function pattern's later-iterated
        # output. Ordering by line is sufficient because parents always
        # appear on a smaller line than their children in source.
        nodes.sort(key=lambda n: (n.line if n.line is not None else 0, n.name or ""))
        return nodes

    def _parse_file_sync(self, file_path: Path) -> ParseResult:
        """Read + tree-sitter parse + symbol extraction + CCG building.

        Runs inside a ThreadPoolExecutor.  All work is CPU-bound — no async store
        calls are made.  Always returns a ParseResult (never raises).

        Thread safety: each worker thread has its own parser cache, event loop,
        and CCG builder (all stored in self._thread_local).
        """
        if not file_path.is_file():
            return ParseResult(file_path=file_path, language=None, vanished=True)

        language = self._detect_language(file_path)
        if language == "unknown":
            return ParseResult(file_path=file_path, language=None)  # skip

        try:
            source_code = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ParseResult(file_path=file_path, language=language)

        # ── Symbol extraction ──────────────────────────────────────────────
        nodes: List[Any] = []
        provider_fallback = False

        # Prefer the enhanced TreeSitterAnalysisProtocol provider when one is
        # registered and claims to support this language. The hardcoded path
        # below is kept as a fallback for: no provider registered (null stub
        # only), provider rejects the language, or provider raises mid-extract.
        provider, enhanced = self._get_analysis_provider()
        if enhanced and provider is not None:
            try:
                if provider.supports_language(language):
                    canonical_file = self._canonical_file_str(file_path)
                    symbol_dicts = provider.extract_symbols(
                        bytes(source_code, "utf-8"),
                        language,
                        file_path=canonical_file,
                    )
                    nodes.extend(
                        self._provider_symbols_to_graph_nodes(symbol_dicts, file_path, language)
                    )
                    # Best-effort import capture — failure here must not poison
                    # symbol extraction, so missing extract_imports or a
                    # provider exception both degrade silently to no imports.
                    raw_imports: List[str] = []
                    try:
                        raw_imports = list(
                            provider.extract_imports(
                                bytes(source_code, "utf-8"),
                                language,
                                file_path=canonical_file,
                            )
                            or []
                        )
                    except Exception:
                        logger.debug(
                            "extract_imports failed for %s — skipping IMPORTS edges from this file",
                            file_path,
                        )
                    # CCG (statement-level CFG/CDG/DDG) is independent of the
                    # symbol-extraction path — it runs against the same source
                    # and shouldn't be skipped just because TSA handled the
                    # symbols. Before the TSA refactor this path didn't exist
                    # and CCG always ran via the legacy block below; after,
                    # the early return left CCG silently empty for every file
                    # the provider handled (i.e. almost all files).
                    ccg_nodes, ccg_edges = self._build_ccg_inline(file_path, language)
                    nodes = self._inject_module_node(nodes, file_path, language)
                    return ParseResult(
                        file_path=file_path,
                        language=language,
                        symbol_nodes=nodes,
                        ccg_nodes=ccg_nodes,
                        ccg_edges=ccg_edges,
                        raw_imports=raw_imports,
                    )
            except Exception as exc:
                logger.debug(
                    "Analysis provider failed for %s (%s): %s — falling back",
                    file_path,
                    language,
                    exc,
                )
                provider_fallback = True
                nodes = []

        if not hasattr(self._thread_local, "parser_cache"):
            self._thread_local.parser_cache = {}

        try:
            cache = self._thread_local.parser_cache
            parser = cache.get(language)
            if parser is None:
                # LRU=1: evict any other language's parser before loading a new
                # one (see _get_tree_sitter_parser for the full rationale).
                if cache:
                    cache.clear()
                parser = self._create_ts_parser(language)
                if parser is not None:
                    cache[language] = parser

            if parser is not None:
                tree = parser.parse(bytes(source_code, "utf-8"))
                nodes.extend(
                    self._extract_definitions(tree.root_node, file_path, language, source_code)
                )
            else:
                nodes.extend(self._extract_symbols_fallback(source_code, file_path, language))
        except Exception as e:
            logger.debug(
                "Thread parse failed for %s (%s): %s — using fallback",
                file_path,
                language,
                e,
            )
            try:
                nodes.extend(self._extract_symbols_fallback(source_code, file_path, language))
            except Exception:
                pass

        # ── CCG (CFG / CDG / DDG) ─────────────────────────────────────────
        ccg_nodes, ccg_edges = self._build_ccg_inline(file_path, language)

        # Legacy extraction path doesn't go through the provider, so it has
        # no raw_imports to surface — IMPORTS edges only flow when the TSA
        # provider is registered. Still inject the module node so CONTAINS
        # edges exist for top-level symbols even on the fallback path.
        nodes = self._inject_module_node(nodes, file_path, language)
        return ParseResult(
            file_path=file_path,
            language=language,
            symbol_nodes=nodes,
            ccg_nodes=ccg_nodes,
            ccg_edges=ccg_edges,
            provider_fallback=provider_fallback,
        )

    async def _write_parsed_result_legacy(self, result: ParseResult) -> GraphIndexStats:
        """Fallback: write one file's parsed result individually (used when bulk write fails)."""
        stats = GraphIndexStats()

        if result.language is None:
            stats.files_skipped += 1
            return stats

        symbol_edges = await self._build_symbol_edges(result.symbol_nodes, result.file_path)

        async with self._graph_store_write_batch():
            await self.graph_store.upsert_nodes(result.symbol_nodes)
            await self.graph_store.upsert_edges(symbol_edges)
            if result.ccg_nodes:
                await self.graph_store.upsert_nodes(result.ccg_nodes)
            if result.ccg_edges:
                await self.graph_store.upsert_edges(result.ccg_edges)
            if result.file_path.is_file():
                await self._update_store_mtime(
                    str(result.file_path),
                    result.file_path.stat().st_mtime,
                    self._file_hash_for(result.file_path),
                )

        stats.files_processed += 1
        stats.nodes_created += len(result.symbol_nodes)
        stats.edges_created += len(symbol_edges)
        stats.ccg_nodes_created += len(result.ccg_nodes)
        stats.ccg_edges_created += len(result.ccg_edges)
        if result.provider_fallback:
            stats.provider_fallbacks += 1
        # Mirror the streaming path's import buffering so the legacy per-file
        # write path produces IMPORTS edges too.
        if result.raw_imports and result.language:
            for raw in result.raw_imports:
                self._pending_import_records.append((str(result.file_path), raw, result.language))
        return stats

    async def _process_batch(
        self,
        files: List[Path],
        total_files: int = 0,
        done_offset: int = 0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> GraphIndexStats:
        """Delegate to _IndexingStreamPipeline for streaming producer-consumer processing.

        The streaming pipeline:
          - Parses all files in parallel via the thread pool (CPU-bound AST + CCG)
          - Feeds results through a bounded asyncio.Queue for back-pressure
          - Flushes mini-batches (write_batch_size files) with ONE bulk store transaction
        """
        write_batch_size = getattr(self.config, "write_batch_size", 20)
        queue_maxsize = getattr(self.config, "queue_maxsize", 100)
        streaming = _IndexingStreamPipeline(
            pipeline=self,
            write_batch_size=write_batch_size,
            queue_maxsize=queue_maxsize,
        )
        return await streaming.run(
            files=files,
            total_files=total_files,
            done_offset=done_offset,
            progress_callback=progress_callback,
        )

    async def _process_file(self, file_path: Path) -> GraphIndexStats:
        """Process a single file.

        Args:
            file_path: Path to file

        Returns:
            File processing stats
        """
        stats = GraphIndexStats()

        if not file_path.is_file():
            await self._handle_vanished_file(file_path)
            stats.files_deleted += 1
            return stats

        GraphNode, GraphEdge = _get_graph_types()
        from victor.storage.graph.edge_types import EdgeType

        try:
            # Detect language
            language = self._detect_language(file_path)
            if language == "unknown":
                stats.files_skipped += 1
                return stats

            # Extract symbols using tree-sitter
            symbol_nodes = await self._extract_symbols(file_path, language)

            # Build symbol edges (CALLS, REFERENCES, CONTAINS)
            symbol_edges = await self._build_symbol_edges(symbol_nodes, file_path)

            # Build CCG if enabled
            ccg_nodes: List[Any] = []
            ccg_edges: List[Any] = []
            if self.config.enable_ccg and self._ccg_builder:
                ccg_nodes, ccg_edges = await self._ccg_builder.build_ccg_for_file(
                    file_path, language
                )

            async with self._graph_store_write_batch():
                # Store symbols
                await self.graph_store.upsert_nodes(symbol_nodes)
                await self.graph_store.upsert_edges(symbol_edges)

                if ccg_nodes:
                    await self.graph_store.upsert_nodes(ccg_nodes)
                if ccg_edges:
                    await self.graph_store.upsert_edges(ccg_edges)

                # Update file mtime + content hash for staleness tracking
                mtime = file_path.stat().st_mtime
                await self._update_store_mtime(
                    str(file_path), mtime, self._file_hash_for(file_path)
                )
        except FileNotFoundError:
            await self._handle_vanished_file(file_path)
            stats.files_deleted += 1
            return stats

        stats.files_processed += 1
        stats.nodes_created += len(symbol_nodes)
        stats.edges_created += len(symbol_edges)
        stats.ccg_nodes_created += len(ccg_nodes)
        stats.ccg_edges_created += len(ccg_edges)

        return stats

    async def _handle_vanished_file(self, file_path: Path) -> None:
        """Drop graph state for a file that disappeared during indexing."""
        logger.info("Skipping vanished file during graph indexing: %s", file_path)
        try:
            await self.graph_store.delete_by_file(str(file_path))
        except Exception as exc:
            logger.debug("Failed to clean vanished file %s from graph store: %s", file_path, exc)

    @asynccontextmanager
    async def _graph_store_write_batch(self):
        """Use store-native batched writes when available."""
        write_batch = getattr(self.graph_store, "write_batch", None)
        if callable(write_batch):
            async with write_batch():
                yield
            return
        yield

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension.

        Args:
            file_path: Path to file

        Returns:
            Language identifier
        """
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "c_sharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".md": "markdown",
        }
        return ext_map.get(file_path.suffix.lower(), "unknown")

    async def _extract_symbols(
        self,
        file_path: Path,
        language: str,
    ) -> List[Any]:
        """Extract symbols from a source file.

        Args:
            file_path: Path to source file
            language: Programming language

        Returns:
            List of symbol nodes
        """
        GraphNode, _ = _get_graph_types()

        nodes: List[Any] = []

        try:
            source_code = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return nodes

        # Use tree-sitter to extract symbols
        try:
            parser = self._get_tree_sitter_parser(language)
            tree = parser.parse(bytes(source_code, "utf-8"))

            # Extract function/class definitions
            nodes.extend(
                self._extract_definitions(tree.root_node, file_path, language, source_code)
            )

        except (ImportError, AttributeError, ValueError, Exception) as e:
            # Fall back to simple regex-based extraction
            logger.debug(f"Tree-sitter extraction failed for {language}: {e}, using fallback")
            nodes.extend(self._extract_symbols_fallback(source_code, file_path, language))

        return nodes

    def _extract_definitions(
        self,
        root_node: Any,
        file_path: Path,
        language: str,
        source_code: str,
    ) -> List[Any]:
        """Extract function/class definitions using tree-sitter.

        Args:
            root_node: Tree-sitter root node
            file_path: Path to source file
            language: Programming language
            source_code: Source code text

        Returns:
            List of symbol nodes
        """
        GraphNode, _ = _get_graph_types()

        nodes: List[Any] = []
        source_lines = source_code.splitlines()
        definition_types = _TREE_SITTER_DEFINITION_TYPES.get(language, set())
        file_str = str(file_path)
        map_node_type = _TREE_SITTER_NODE_TYPE_MAP.get

        def extract(node: Any, parent_id: str | None = None) -> None:
            node_type = node.type

            # Check if this is a definition we care about
            if node_type in definition_types:
                # Get line numbers
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Extract name
                name = self._extract_name(node)

                # Get signature
                signature = self._extract_signature(node, source_lines, start_line, end_line)

                # Get docstring
                docstring = self._extract_docstring(
                    node, source_lines, start_line, end_line, language
                )

                # Generate node ID
                import hashlib

                node_id = hashlib.sha256(f"{file_str}:{name}:{start_line}".encode()).hexdigest()[
                    :16
                ]

                # Determine visibility
                visibility = self._determine_visibility(node, name, language)

                # Create node
                graph_node = GraphNode(
                    node_id=node_id,
                    type=map_node_type(node_type, "unknown"),
                    name=name,
                    file=file_str,
                    line=start_line,
                    end_line=end_line,
                    lang=language,
                    signature=signature,
                    docstring=docstring,
                    parent_id=parent_id,
                    visibility=visibility,
                    ast_kind=node_type,
                )
                nodes.append(graph_node)

                # DEBUG: Log node creation
                logger.debug(
                    f"Created node: {name} (type={node_type}, parent_id={parent_id[:8] + '...' if parent_id else None})"
                )

                # Recurse into children (for nested classes/functions)
                for child in self._iter_tree_children(node):
                    extract(child, graph_node.node_id)
            else:
                # Continue traversing
                for child in self._iter_tree_children(node):
                    extract(child, parent_id)

        extract(root_node)
        return nodes

    def _is_definition_type(self, node_type: str, language: str) -> bool:
        """Check if node type is a definition we should extract.

        Args:
            node_type: Tree-sitter node type
            language: Programming language

        Returns:
            True if this is a definition type
        """
        return node_type in _TREE_SITTER_DEFINITION_TYPES.get(language, set())

    def _map_node_type(self, tree_sitter_type: str) -> str:
        """Map tree-sitter node type to our graph node type.

        Args:
            tree_sitter_type: Tree-sitter node type

        Returns:
            Graph node type string
        """
        return _TREE_SITTER_NODE_TYPE_MAP.get(tree_sitter_type, "unknown")

    def _extract_name(self, node: Any) -> str:
        """Extract name from a definition node.

        Args:
            node: Tree-sitter node

        Returns:
            Name string
        """

        name = self._extract_name_from_node(node)
        return name or "<unnamed>"

    def _extract_signature(
        self,
        node: Any,
        source_lines: List[str],
        start_line: int,
        end_line: int,
    ) -> str | None:
        """Extract function/method signature.

        Args:
            node: Tree-sitter node
            source_lines: Source code lines
            start_line: Start line number
            end_line: End line number

        Returns:
            Signature string or None
        """
        if 0 < start_line <= len(source_lines):
            # Get the first line as signature preview
            return source_lines[start_line - 1].strip()[:200]
        return None

    def _extract_docstring(
        self,
        node: Any,
        source_lines: List[str],
        start_line: int,
        end_line: int,
        language: str,
    ) -> str | None:
        """Extract docstring from a definition node.

        Args:
            node: Tree-sitter node
            source_lines: Source code lines
            start_line: Start line number
            end_line: End line number
            language: Programming language

        Returns:
            Docstring or None
        """
        # Look for string literals in the function body
        # This is a simplified implementation
        return None

    def _determine_visibility(
        self,
        node: Any,
        name: str,
        language: str,
    ) -> str | None:
        """Determine visibility of a definition.

        Args:
            node: Tree-sitter node
            name: Definition name
            language: Programming language

        Returns:
            Visibility string or None
        """
        # Python: underscore prefix = private
        if language == "python":
            if name.startswith("_"):
                return "private"
            return "public"

        # JavaScript/TypeScript: check modifiers
        if language in {"javascript", "typescript"}:
            for child in self._iter_tree_children(node):
                if child.type == "property_identifier":
                    text = self._node_text(child)
                    if text == "private":
                        return "private"
                    if text == "protected":
                        return "protected"
            return "public"

        # Java: check modifiers
        if language == "java":
            for child in self._iter_tree_children(node):
                if child.type in {"modifiers", "modifier", "annotation"}:
                    text = self._node_text(child)
                    if "private" in text:
                        return "private"
                    if "protected" in text:
                        return "protected"
            return "public"

        return None

    def _get_tree_sitter_parser(self, language: str) -> Any:
        """Get or create a thread-local cached tree-sitter parser for a language.

        Cache is LRU=1: at most one parser per worker thread. Combined with
        language-grouped file discovery (see ``_discover_files``), the hot
        parser stays pinned across an entire language run and is evicted only
        when the language changes — bounding peak parser memory regardless of
        how many languages the repo contains.
        """
        if not hasattr(self._thread_local, "parser_cache"):
            self._thread_local.parser_cache = {}
        cache = self._thread_local.parser_cache

        parser = cache.get(language)
        if parser is not None:
            return parser

        # LRU=1: evict any other language's parser before loading a new one.
        if cache:
            cache.clear()

        parser = self._create_ts_parser(language)
        if parser is None:
            raise ValueError(f"Unsupported language: {language}")
        cache[language] = parser
        return parser

    def _iter_tree_children(self, node: Any) -> Any:
        """Return named children when available to avoid punctuation-heavy traversal."""
        try:
            children = node.named_children
        except AttributeError:
            children = node.children
        return children

    def _node_text(self, node: Any) -> str:
        """Return node text decoded as UTF-8."""
        text = node.text
        if isinstance(text, str):
            return text
        if isinstance(text, memoryview):
            text = text.tobytes()
        if isinstance(text, bytearray):
            text = bytes(text)
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return str(text)

    def _extract_identifier_text(self, node: Any) -> str | None:
        """Extract identifier-like text from a node or its immediate named children."""
        if node.type in _TREE_SITTER_IDENTIFIER_TYPES:
            return self._node_text(node)

        for child in self._iter_tree_children(node):
            if child.type in _TREE_SITTER_IDENTIFIER_TYPES:
                return self._node_text(child)

        return None

    def _extract_symbols_fallback(
        self,
        source_code: str,
        file_path: Path,
        language: str,
    ) -> List[Any]:
        """Fallback symbol extraction using simple regex patterns.

        Args:
            source_code: Source code text
            file_path: Path to source file
            language: Programming language

        Returns:
            List of symbol nodes
        """
        GraphNode, _ = _get_graph_types()
        import re
        import hashlib

        # ADR-015 (Phase 1): prefer the shared victor-codegraph parser when importable —
        # real AST (functions/classes/methods, multi-language) instead of the Python-only
        # def/class regex below. Canonical v2 ids are preserved so downstream
        # consumers share one identity scheme; falls back to the regex on any
        # parser failure.
        try:
            import victor_codegraph as _vcg
        except Exception:
            _vcg = None
        if _vcg is not None:
            canonical_file = Path(file_path).as_posix()
            try:
                canonical_file = self._canonical_file_str(file_path)
            except (AttributeError, TypeError):
                # This extraction seam is deliberately usable as a stateless
                # fallback in isolation (including by compatibility callers).
                pass
            try:
                parsed = _vcg.parse(
                    source_code,
                    language=language,
                    file_path=canonical_file,
                )
            except Exception:
                parsed = None
            if parsed is not None and parsed.symbols:
                _AST_KIND = {
                    "class": "class_definition",
                    "struct": "class_definition",
                    "interface": "class_definition",
                    "trait": "class_definition",
                    "enum": "enum_definition",
                }
                delegated: List[Any] = []
                for s in parsed.symbols:
                    name = s.simple_name
                    line = s.location.start_line
                    node_id = s.id
                    stype = s.symbol_type.name.lower()
                    delegated.append(
                        GraphNode(
                            node_id=node_id,
                            type=stype,
                            name=name,
                            file=canonical_file,
                            line=line,
                            lang=language,
                            ast_kind=_AST_KIND.get(stype, "function_definition"),
                        )
                    )
                return delegated

        nodes: List[Any] = []

        if language == "python":
            # Simple regex for Python functions and classes
            func_pattern = r"^(def|async def)\s+(\w+)"
            class_pattern = r"^class\s+(\w+)"

            for i, line in enumerate(source_code.splitlines(), 1):
                func_match = re.match(func_pattern, line.strip())
                if func_match:
                    name = func_match.group(2)
                    node_id = hashlib.sha256(f"{file_path}:{name}:{i}".encode()).hexdigest()[:16]
                    nodes.append(
                        GraphNode(
                            node_id=node_id,
                            type="function",
                            name=name,
                            file=str(file_path),
                            line=i,
                            lang=language,
                            ast_kind="function_definition",
                        )
                    )

                class_match = re.match(class_pattern, line.strip())
                if class_match:
                    name = class_match.group(1)
                    node_id = hashlib.sha256(f"{file_path}:{name}:{i}".encode()).hexdigest()[:16]
                    nodes.append(
                        GraphNode(
                            node_id=node_id,
                            type="class",
                            name=name,
                            file=str(file_path),
                            line=i,
                            lang=language,
                            ast_kind="class_definition",
                        )
                    )

        return nodes

    async def _build_symbol_edges(
        self,
        nodes: List[Any],
        file_path: Path,
    ) -> List[Any]:
        """Build edges between symbol nodes.

        Args:
            nodes: Symbol nodes from the file
            file_path: Path to source file

        Returns:
            List of graph edges
        """
        _, GraphEdge = _get_graph_types()
        from victor.storage.graph.edge_types import EdgeType

        edges: List[Any] = []

        # Build CONTAINS edges for parent-child relationships
        for node in nodes:
            if node.parent_id:
                edges.append(
                    GraphEdge(
                        src=node.parent_id,
                        dst=node.node_id,
                        type=EdgeType.CONTAINS,
                    )
                )

        # Build CALLS edges by analyzing source code for cross-references
        calls_edges = await self._build_calls_edges(nodes, file_path)
        edges.extend(calls_edges)

        return edges

    async def _build_calls_edges(
        self,
        nodes: List[Any],
        file_path: Path,
    ) -> List[Any]:
        """Build CALLS edges by analyzing function calls in source code.

        ARCHITECTURAL SEAM (2026-04-29):
        This method now uses the LanguageEdgeHandler protocol for
        per-language edge detection. Handlers are registered in
        handlers/ directory and discovered via get_edge_handler().

        MIGRATION PATH:
        1. New languages: Add handler in handlers/{lang}_edges.py
        2. Register handler in handlers/__init__.py
        3. Handler is auto-discovered here via get_edge_handler()
        4. Eventually move handlers to victor-coding language plugins

        Fallback: Legacy _build_calls_edges_legacy() used when no
        handler is registered for the language. To be removed once
        all supported languages have handlers.

        Args:
            nodes: Symbol nodes from the file
            file_path: Path to source file

        Returns:
            List of CALLS edges
        """
        _, GraphEdge = _get_graph_types()
        from victor.storage.graph.edge_types import EdgeType

        edges: List[Any] = []

        # Build name -> node_id mapping for lookups
        name_to_ids: Dict[str, List[str]] = {}
        for node in nodes:
            if node.name:
                name_to_ids.setdefault(node.name, []).append(node.node_id)

        # Detect language
        language = self._detect_language(file_path)
        if language == "unknown":
            return edges

        # ADR-015/v2 canonical path: use the same relations and stable ids that
        # produced ``nodes``. Resolved in-file edges can be emitted immediately;
        # structured unresolved references feed the existing project-wide resolver.
        try:
            import victor_codegraph as _vcg

            canonical_file = self._canonical_file_str(file_path)
            repository = self._codegraph_repository
            parsed = repository.files.get(canonical_file) if repository is not None else None
            if parsed is None:
                source_code = file_path.read_text(encoding="utf-8")
                parsed = _vcg.parse(
                    source_code,
                    language=language,
                    file_path=canonical_file,
                )
            node_ids = {node.node_id for node in nodes}
            if {symbol.id for symbol in parsed.symbols} <= node_ids:
                self._codegraph_owned_files.add(canonical_file)
                self._codegraph_owned_symbol_ids.update(symbol.id for symbol in parsed.symbols)
                structural = {
                    "EXTENDS": EdgeType.INHERITS,
                    "IMPLEMENTS": EdgeType.IMPLEMENTS,
                }
                for relation in parsed.relations:
                    if relation.relation_type.name == "CALLS":
                        if relation.target_ref is None and relation.to_symbol_id in node_ids:
                            edges.append(
                                GraphEdge(
                                    src=relation.from_symbol_id,
                                    dst=relation.to_symbol_id,
                                    type=EdgeType.CALLS,
                                )
                            )
                        elif relation.target_ref is not None:
                            self._pending_call_records.append(
                                (
                                    relation.from_symbol_id,
                                    relation.target_ref.name,
                                    None,
                                    bool(relation.target_ref.qualifier),
                                )
                            )
                    elif relation.relation_type.name in structural:
                        edge_type = structural[relation.relation_type.name]
                        if relation.target_ref is None and relation.to_symbol_id in node_ids:
                            edges.append(
                                GraphEdge(
                                    src=relation.from_symbol_id,
                                    dst=relation.to_symbol_id,
                                    type=edge_type,
                                )
                            )
                        elif relation.target_ref is not None:
                            self._pending_relationship_records.append(
                                (
                                    relation.from_symbol_id,
                                    relation.target_ref.name,
                                    edge_type.value,
                                )
                            )
                return edges
        except Exception as exc:
            logger.debug("victor-codegraph edge delegation failed for %s: %s", file_path, exc)

        # NEW PATH: Use language-specific edge handler (2026-04-29)
        try:
            from victor.core.graph_rag.language_handlers import get_edge_handler

            handler = get_edge_handler(language)
            if handler is not None:
                logger.debug(f"Using edge handler for language: {language}")
                edges.extend(
                    await self._build_edges_with_handler(handler, nodes, file_path, name_to_ids)
                )
                return edges
        except ImportError:
            logger.debug("Language handler system not available, using legacy")

        # FALLBACK: load-bearing, not legacy debt. Edge handlers are supplied by
        # the victor-coding plugin via CapabilityRegistry/TreeSitterAnalysisProtocol;
        # a core-only install has NO handlers for any language, so this path is the
        # only CALLS detection there. Removal requires core bundling a default
        # handler provider first.
        logger.debug(f"Using legacy CALLS edge detection for: {language}")
        return await self._build_calls_edges_legacy(nodes, file_path, name_to_ids)

    async def _build_edges_with_handler(
        self,
        handler: Any,
        nodes: List[Any],
        file_path: Path,
        name_to_ids: Dict[str, List[str]],
    ) -> List[Any]:
        """Build edges using a language-specific edge handler.

        Args:
            handler: LanguageEdgeHandler instance
            nodes: Symbol nodes from the file
            file_path: Path to source file
            name_to_ids: Name to node_id mapping

        Returns:
            List of CALLS edges
        """
        _, GraphEdge = _get_graph_types()
        from victor.storage.graph.edge_types import EdgeType

        edges: List[Any] = []

        try:
            # Get language module
            language = self._detect_language(file_path)
            if language not in _TREE_SITTER_LANGUAGE_MODULES:
                return edges

            # Parse source
            source_code = file_path.read_text(encoding="utf-8")
            parser = self._get_tree_sitter_parser(language)
            tree = parser.parse(bytes(source_code, "utf-8"))

            # Use handler to detect calls
            from victor.core.graph_rag.language_handlers import CallEdge

            result = await handler.detect_calls_edges(tree, source_code, file_path)

            # Buffer raw (caller_id, callee_name, receiver_type, is_method_call)
            # records. CALLS edges are emitted later by
            # _resolve_cross_file_calls() using a project-wide name index, so
            # that callees defined in another file resolve too. Caller is
            # always in this file, so per-file name_to_ids is fine.
            # receiver_type and is_method_call are plugin-supplied.
            buffered = 0
            for call in result.calls:
                receiver_type = getattr(call, "receiver_type", None)
                is_method_call = getattr(call, "is_method_call", False)
                for caller_id in name_to_ids.get(call.caller_name, []):
                    self._pending_call_records.append(
                        (caller_id, call.callee_name, receiver_type, is_method_call)
                    )
                    buffered += 1

            # Buffer (src_id, target_name, edge_type) for non-CALLS
            # relationships (INHERITS / IMPLEMENTS / COMPOSITION). The source
            # is always declared in this file so we can resolve it via
            # name_to_ids immediately; the target may live in another file,
            # so leaf-name resolution against the project-wide class index
            # happens later in _resolve_cross_file_relationships().
            rel_buffered = 0
            for rel in getattr(result, "relationships", None) or []:
                for src_id in name_to_ids.get(rel.source_name, []):
                    self._pending_relationship_records.append(
                        (src_id, rel.target_name, rel.edge_type)
                    )
                    rel_buffered += 1

            logger.debug(
                f"Handler detected {len(result.calls)} calls + "
                f"{len(getattr(result, 'relationships', None) or [])} relationships, "
                f"buffered {buffered} call records and {rel_buffered} relationship "
                f"records for cross-file resolution"
            )

        except (ImportError, Exception) as e:
            logger.warning(f"Handler-based edge detection failed: {e}")

        return edges

    async def _build_calls_edges_legacy(
        self,
        nodes: List[Any],
        file_path: Path,
        name_to_ids: Dict[str, List[str]],
    ) -> List[Any]:
        """Fallback CALLS edge detection for installs without edge handlers.

        NOT removable on a schedule: edge handlers come from the victor-coding
        plugin (CapabilityRegistry / TreeSitterAnalysisProtocol), so a core-only
        install has no handler for ANY language and this is its only CALLS
        detection. The former "remove by 2026-Q2 once all languages migrate"
        TODO was wrong — removal is conditional on core shipping a default
        handler provider, not on language coverage.

        Args:
            nodes: Symbol nodes from the file
            file_path: Path to source file
            name_to_ids: Name to node_id mapping

        Returns:
            List of CALLS edges
        """
        _, GraphEdge = _get_graph_types()
        from victor.storage.graph.edge_types import EdgeType

        edges: List[Any] = []

        # Try tree-sitter for call analysis
        try:
            # Detect language
            language = self._detect_language(file_path)
            if language == "unknown":
                return edges

            # Get language module
            if language not in _TREE_SITTER_LANGUAGE_MODULES:
                return edges

            # Parse source
            source_code = file_path.read_text(encoding="utf-8")
            parser = self._get_tree_sitter_parser(language)
            tree = parser.parse(bytes(source_code, "utf-8"))

            # Find function calls
            calls = self._extract_function_calls(tree.root_node, language)
            logger.debug(f"Legacy: Found {len(calls)} function calls in {file_path.name}")

            # Buffer raw records for project-wide resolution in
            # _resolve_cross_file_calls(); see the handler path for the
            # rationale. The legacy regex-based extractor doesn't model
            # receiver types or method-call syntax, so it pushes None /
            # False — the resolver falls back to name-only with fanout cap.
            buffered = 0
            for caller_name, callee_name in calls:
                for caller_id in name_to_ids.get(caller_name, []):
                    self._pending_call_records.append((caller_id, callee_name, None, False))
                    buffered += 1
            logger.debug(f"Legacy: Buffered {buffered} call records for cross-file resolution")

        except (ImportError, Exception) as e:
            logger.debug(f"Legacy CALLS edge extraction failed: {e}")

        return edges

    def _extract_function_calls(
        self,
        root_node: Any,
        language: str,
    ) -> List[tuple[str, str]]:
        """Extract function calls from tree-sitter AST.

        Args:
            root_node: Tree-sitter root node
            language: Programming language

        Returns:
            List of (caller_name, callee_name) tuples
        """
        calls: List[tuple[str, str]] = []
        current_function: str | None = None

        def extract(node: Any, parent_function: str | None = None) -> None:
            nonlocal current_function

            node_type = node.type

            # Track current function scope
            if node_type in ("function_definition", "class_definition"):
                # Extract function/class name
                name = self._extract_name_from_node(node)
                if name:
                    # For methods inside classes, track parent
                    current_function = name
                    parent_function = name

            # Find call nodes (language-specific)
            if language == "python" and node_type == "call":
                callee_name = self._extract_callee_name(node, language)
                if callee_name and current_function:
                    calls.append((current_function, callee_name))

            elif language in ("javascript", "typescript") and node_type == "call_expression":
                callee_name = self._extract_callee_name(node, language)
                if callee_name and current_function:
                    calls.append((current_function, callee_name))

            # Recurse
            for child in self._iter_tree_children(node):
                extract(child, parent_function)

        extract(root_node)
        return calls

    def _extract_name_from_node(self, node: Any) -> str | None:
        """Extract name from a tree-sitter definition node.

        For most node types (function, struct, class, etc.) the ``name`` field
        gives the symbol's identifier. Rust ``impl_item`` is the exception: it
        has no ``name`` field, only ``type`` (the implementing type) and an
        optional ``trait`` (for ``impl Trait for Type``). Without the ``type``
        fallback, the first-identifier scan returns the trait name -- so
        ``impl Default for Foo`` would be recorded as "Default" and all
        Default::default() calls would fan out to every user-defined `fn
        default` method (observed: 800k+ fanout on proximaDB).

        Args:
            node: Tree-sitter node

        Returns:
            Name string or None
        """
        try:
            name_node = node.child_by_field_name("name")
        except AttributeError:
            name_node = None

        if name_node is not None:
            name = self._extract_identifier_text(name_node)
            if name:
                return name

        # Rust impl_item: prefer `type` (implementing type) over the
        # first-identifier scan (which would return the trait name).
        try:
            type_node = node.child_by_field_name("type")
        except AttributeError:
            type_node = None

        if type_node is not None:
            name = self._extract_identifier_text(type_node)
            if name:
                return name

        for child in self._iter_tree_children(node):
            name = self._extract_identifier_text(child)
            if name:
                return name
        return None

    def _extract_callee_name(self, call_node: Any, language: str) -> str | None:
        """Extract callee name from a call node.

        Args:
            call_node: Tree-sitter call node
            language: Programming language

        Returns:
            Callee name or None
        """
        # For Python: call -> function (identifier)
        # For JS/TS: call_expression -> member_expression or identifier

        def find_identifier(node: Any) -> str | None:
            identifier = self._extract_identifier_text(node)
            if identifier:
                return identifier
            for child in self._iter_tree_children(node):
                result = find_identifier(child)
                if result:
                    return result
            return None

        # For Python, look at first child of call
        children = self._iter_tree_children(call_node)
        if language == "python" and children:
            func_node = children[0]
            if func_node.type == "identifier":
                return self._node_text(func_node)
            elif func_node.type == "attribute":
                # For method calls like obj.method(), get the method name
                for child in self._iter_tree_children(func_node):
                    if child.type in _TREE_SITTER_IDENTIFIER_TYPES:
                        return self._node_text(child)

        # For JS/TS, find first identifier
        elif language in ("javascript", "typescript"):
            return find_identifier(call_node)

        return None

    def _get_vector_provider(self) -> Any:
        """Vector store for persisting node embeddings, or None.

        Proxima-style stores co-locate graph properties and vectors via an atomic
        ``upsert_node_record`` — they need no external provider. For the
        default sqlite store, build a provider from the store's settings-driven
        config (falls back to defaults).
        """
        if hasattr(self.graph_store, "upsert_node_record"):
            return None
        try:
            from victor.storage.vector_stores.registry import EmbeddingRegistry

            build = getattr(self.graph_store, "_build_embedding_config", None)
            if build is not None:
                config = build()
            else:
                from victor.storage.vector_stores.base import EmbeddingConfig

                config = EmbeddingConfig()
            return EmbeddingRegistry.create(config)
        except Exception as e:
            logger.debug("Vector store unavailable for graph embeddings: %s", e)
            return None

    async def _generate_embeddings(self) -> GraphIndexStats:
        """Generate embeddings for graph nodes — batched and staleness-aware.

        - Incremental runs consider only nodes from files parsed this run;
          force/full runs consider every node.
        - A node is skipped when its stored ``content_version`` (blake2b-8 of
          the embedded text: name + signature + docstring) matches and it
          already has an embedding — bodies aren't embedded, so body-only edits
          don't force re-embeds.
        - Text vectors come from ONE ``EmbeddingService.embed_batch`` call per
          batch (see ``GraphAwareEmbedder.embed_batch``).
        - Persistence: ``index_embedded_documents`` on the vector store keyed
          by node_id (default sqlite path), or the graph store's own
          ``upsert_node_record`` (one ProximaRecord containing graph props + vector).

        Returns:
            Stats for embedding generation
        """
        stats = GraphIndexStats()
        start_time = time.time()

        try:
            import hashlib

            from victor.processing.graph_embeddings import (
                GraphAwareEmbedder,
                GraphEmbeddingConfig,
            )
            from victor.storage.embeddings.service import get_embedding_service

            config = GraphEmbeddingConfig(
                neighborhood_radius=self.config.embedding_neighborhood_radius or 2,
                include_edge_types=True,
                structural_weight=self.config.embedding_structural_weight,
                semantic_weight=1.0 - self.config.embedding_structural_weight,
                max_neighbors=self.config.embedding_max_neighbors or 50,
            )
            embedder = GraphAwareEmbedder(config=config)

            embedding_service = get_embedding_service()
            if embedding_service is None:
                logger.warning("EmbeddingService not available, skipping embedding generation")
                return stats

            # Candidate nodes: changed files only on incremental runs.
            if self.config.incremental and self._files_to_process:
                nodes = []
                for file_key in sorted(self._files_to_process):
                    nodes.extend(await self.graph_store.get_nodes_by_file(file_key))
            else:
                nodes = await self.graph_store.get_all_nodes()

            # Staleness filter: fingerprint of the text that would be embedded.
            candidates = []
            versions: Dict[str, str] = {}
            for node in nodes:
                text = GraphAwareEmbedder.node_text(node)
                if not text:
                    continue
                cv = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
                meta = node.metadata or {}
                if meta.get("content_version") == cv and meta.get("has_embedding"):
                    continue
                versions[node.node_id] = cv
                candidates.append(node)

            logger.info(
                "Generating embeddings for %d nodes (%d unchanged skipped)",
                len(candidates),
                len(nodes) - len(candidates),
            )
            if not candidates:
                return stats

            vector_provider = self._get_vector_provider()
            node_by_id = {n.node_id: n for n in candidates}
            batch_size = self.config.embedding_batch_size or 100
            embedded_total = 0

            for i in range(0, len(candidates), batch_size):
                batch = candidates[i : i + batch_size]
                embeddings = await embedder.embed_batch(batch, self.graph_store)
                if not embeddings:
                    continue

                persisted_ids: set[str] = set()
                atomic_record_store = getattr(self.graph_store, "upsert_node_record", None)
                try:
                    if vector_provider is not None and hasattr(
                        vector_provider, "index_embedded_documents"
                    ):
                        docs = []
                        for node_id, vector in embeddings.items():
                            node = node_by_id[node_id]
                            docs.append(
                                {
                                    "id": node_id,
                                    "content": GraphAwareEmbedder.node_text(node),
                                    "vector": vector,
                                    "metadata": {
                                        "node_id": node_id,
                                        "file_path": node.file,
                                        "symbol_name": node.name,
                                        "line_number": node.line or 0,
                                        "content_version": versions[node_id],
                                    },
                                }
                            )
                        await vector_provider.index_embedded_documents(docs)
                        persisted_ids.update(embeddings)
                    elif atomic_record_store is not None:
                        for node_id, vector in embeddings.items():
                            try:
                                await atomic_record_store(
                                    node_id,
                                    vector,
                                    metadata={
                                        "has_embedding": True,
                                        "content_version": versions[node_id],
                                    },
                                )
                                persisted_ids.add(node_id)
                            except Exception as e:
                                logger.warning(
                                    "Atomic record persistence failed for %s: %s", node_id, e
                                )
                                stats.error_count += 1
                                stats.errors.append(f"Atomic ProximaRecord {node_id}: {e}")
                except Exception as e:
                    logger.warning(f"Vector persistence failed for batch: {e}")
                    stats.error_count += 1
                    stats.errors.append(f"Vector persistence: {e}")
                    continue  # don't mark nodes embedded if vectors didn't persist

                if atomic_record_store is None:
                    for node_id in persisted_ids:
                        try:
                            await self.graph_store.update_node_metadata(
                                node_id,
                                {
                                    "embedding_ref": f"emb:{node_id}",
                                    "has_embedding": True,
                                    "content_version": versions[node_id],
                                },
                            )
                        except Exception as e:
                            logger.warning(f"Failed to mark embedding for {node_id}: {e}")
                            stats.error_count += 1
                            stats.errors.append(f"Embedding metadata failed for {node_id}: {e}")

                embedded_total += len(persisted_ids)
                logger.debug(f"Embedded batch {i // batch_size + 1}")

            stats.embeddings_generated = embedded_total
            logger.info(f"Generated {embedded_total} embeddings in {time.time() - start_time:.2f}s")

        except ImportError as e:
            logger.warning(f"Graph embedding components not available: {e}")
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            stats.error_count += 1
            stats.errors.append(f"Embedding generation: {e}")

        return stats

    async def _cache_subgraphs(self) -> GraphIndexStats:
        """Cache subgraphs for important nodes.

        Returns:
            Stats for subgraph caching
        """
        stats = GraphIndexStats()

        # TODO: Implement subgraph caching
        # This would:
        # 1. Identify important nodes (high centrality, recent changes, etc.)
        # 2. Compute subgraphs around these nodes
        # 3. Cache them in the graph store

        logger.info("Subgraph caching not yet implemented")
        return stats

    def _merge_stats(self, target: GraphIndexStats, source: GraphIndexStats) -> None:
        """Merge source stats into target stats.

        Args:
            target: Target stats to merge into
            source: Source stats to merge from
        """
        target.files_processed += source.files_processed
        target.files_skipped += source.files_skipped
        target.files_unchanged += source.files_unchanged
        target.files_deleted += source.files_deleted
        target.nodes_created += source.nodes_created
        target.edges_created += source.edges_created
        target.ccg_nodes_created += source.ccg_nodes_created
        target.ccg_edges_created += source.ccg_edges_created
        target.embeddings_generated += source.embeddings_generated
        target.subgraphs_cached += source.subgraphs_cached
        target.module_metrics_computed += source.module_metrics_computed
        target.provider_fallbacks += source.provider_fallbacks
        target.cross_file_calls_resolved += source.cross_file_calls_resolved
        target.cross_file_relationships_resolved += source.cross_file_relationships_resolved
        target.imports_resolved += source.imports_resolved
        target.error_count += source.error_count
        target.errors.extend(source.errors)


class _IndexingStreamPipeline:
    """Parallel-parse → bounded-queue → batched-write streaming pipeline.

    Architecture
    ============
    Producers (N threads):  parse files in thread pool → enqueue ParseResult
    Consumer  (1 coroutine): drain queue → accumulate mini-batch → bulk write

    Back-pressure
    =============
    asyncio.Queue(maxsize=queue_maxsize) throttles producers when the consumer
    is slower than the thread pool: each producer coroutine blocks on queue.put()
    until the consumer frees space.  The thread pool itself caps CPU concurrency
    to max_workers (min(cpu_count, 8) by default).

    Memory footprint
    ================
    At most queue_maxsize ParseResults are held in RAM simultaneously, regardless
    of total file count.  Mini-batches of write_batch_size files are flushed in
    one SQLite transaction, so peak write memory is bounded as well.
    """

    # Class-level sentinel — shared across all instances, safe for `is` comparisons
    _STREAM_DONE: object = object()

    def __init__(
        self,
        pipeline: GraphIndexingPipeline,
        write_batch_size: int = 20,
        queue_maxsize: int = 100,
    ) -> None:
        self._pipeline = pipeline
        self._write_batch_size = write_batch_size
        self._queue_maxsize = queue_maxsize

    # ── Public entry point ─────────────────────────────────────────────────

    async def run(
        self,
        files: List[Path],
        total_files: int = 0,
        done_offset: int = 0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> GraphIndexStats:
        """Run producer and consumer concurrently; return merged stats."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        stats, _ = await asyncio.gather(
            self._consume(queue, total_files, done_offset, progress_callback),
            self._produce(files, queue),
        )
        return stats

    # ── Producer ───────────────────────────────────────────────────────────

    async def _produce(self, files: List[Path], queue: asyncio.Queue) -> None:
        """Submit every file to the thread pool; put each ParseResult into the queue.

        asyncio.gather runs all coroutines concurrently.  Actual CPU work is
        bounded by the thread pool's max_workers; the queue's maxsize provides
        additional back-pressure so the thread pool slows down when the consumer
        (store write) falls behind.
        """
        loop = asyncio.get_event_loop()
        executor = self._pipeline._get_executor()

        async def _parse_and_put(fp: Path) -> None:
            try:
                result = await loop.run_in_executor(executor, self._pipeline._parse_file_sync, fp)
            except Exception as exc:
                result = ParseResult(file_path=fp, language=None, error=exc)
            await queue.put(result)  # blocks when queue full → natural back-pressure

        await asyncio.gather(*(_parse_and_put(fp) for fp in files), return_exceptions=True)
        await queue.put(self._STREAM_DONE)

    # ── Consumer ───────────────────────────────────────────────────────────

    async def _consume(
        self,
        queue: asyncio.Queue,
        total_files: int,
        done_offset: int,
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> GraphIndexStats:
        """Drain queue, sort results into pending list, flush mini-batches to store."""
        stats = GraphIndexStats()
        pending: List[ParseResult] = []
        files_done = done_offset

        while True:
            item = await queue.get()

            if item is self._STREAM_DONE:
                if pending:
                    flush_stats = await self._flush(
                        pending, files_done, total_files, progress_callback
                    )
                    self._pipeline._merge_stats(stats, flush_stats)
                    files_done += len(pending)
                break

            result: ParseResult = item

            # Handle terminal states immediately — no buffering needed
            if result.vanished:
                await self._pipeline._handle_vanished_file(result.file_path)
                stats.files_deleted += 1
                files_done += 1
                continue

            if result.error is not None:
                stats.error_count += 1
                stats.errors.append(f"{result.file_path}: {result.error}")
                files_done += 1
                continue

            if result.language is None:
                stats.files_skipped += 1
                files_done += 1
                continue

            pending.append(result)

            if len(pending) >= self._write_batch_size:
                flush_stats = await self._flush(pending, files_done, total_files, progress_callback)
                self._pipeline._merge_stats(stats, flush_stats)
                files_done += len(pending)
                pending.clear()
                # Cooperative yield between mini-batches: _flush runs
                # synchronous edge-extraction and bulk-write CPU work that
                # can block the event loop for large batches. A bare
                # asyncio.sleep(0) gives other coroutines (heartbeats,
                # progress reporters, status pollers) a chance to run
                # without slowing the pipeline materially.
                await asyncio.sleep(0)

        return stats

    # ── Flush ──────────────────────────────────────────────────────────────

    async def _flush(
        self,
        batch: List[ParseResult],
        done_offset: int,
        total_files: int,
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> GraphIndexStats:
        """Build symbol edges then commit one bulk store transaction for *batch*.

        Falls back to per-file writes if the bulk transaction fails (e.g.
        transient SQLite lock contention in incremental mode).
        """
        stats = GraphIndexStats()
        all_sym_nodes: List[Any] = []
        all_sym_edges: List[Any] = []
        all_ccg_nodes: List[Any] = []
        all_ccg_edges: List[Any] = []

        # Phase A: build symbol edges (pure graph traversal, no I/O)
        for result in batch:
            sym_edges = await self._pipeline._build_symbol_edges(
                result.symbol_nodes, result.file_path
            )
            all_sym_nodes.extend(result.symbol_nodes)
            all_sym_edges.extend(sym_edges)
            all_ccg_nodes.extend(result.ccg_nodes)
            all_ccg_edges.extend(result.ccg_edges)
            stats.nodes_created += len(result.symbol_nodes)
            stats.edges_created += len(sym_edges)
            stats.ccg_nodes_created += len(result.ccg_nodes)
            stats.ccg_edges_created += len(result.ccg_edges)
            # Buffer raw imports per file for post-persistence resolution to
            # IMPORTS edges. We stash (source_file, raw_string, language) so
            # the resolver can pick the right per-language parser.
            if result.raw_imports and result.language:
                for raw in result.raw_imports:
                    self._pipeline._pending_import_records.append(
                        (str(result.file_path), raw, result.language)
                    )

        # Phase B: ONE bulk write transaction for the entire mini-batch
        try:
            async with self._pipeline._graph_store_write_batch():
                if all_sym_nodes:
                    await self._pipeline.graph_store.upsert_nodes(all_sym_nodes)
                if all_sym_edges:
                    await self._pipeline.graph_store.upsert_edges(all_sym_edges)
                if all_ccg_nodes:
                    await self._pipeline.graph_store.upsert_nodes(all_ccg_nodes)
                if all_ccg_edges:
                    await self._pipeline.graph_store.upsert_edges(all_ccg_edges)
                for result in batch:
                    if result.file_path.is_file():
                        await self._pipeline._update_store_mtime(
                            str(result.file_path),
                            result.file_path.stat().st_mtime,
                            self._pipeline._file_hash_for(result.file_path),
                        )
            stats.files_processed += len(batch)
            stats.provider_fallbacks += sum(
                1 for r in batch if getattr(r, "provider_fallback", False)
            )
        except Exception as exc:
            logger.warning(
                "Bulk mini-batch write failed (%d files); retrying per-file: %s",
                len(batch),
                exc,
            )
            stats = GraphIndexStats()
            for result in batch:
                try:
                    fs = await self._pipeline._write_parsed_result_legacy(result)
                    self._pipeline._merge_stats(stats, fs)
                except Exception as e:
                    stats.error_count += 1
                    stats.errors.append(f"{result.file_path}: {e}")

        if progress_callback and batch:
            progress_callback(done_offset + len(batch), total_files, str(batch[-1].file_path))

        return stats


__all__ = ["GraphIndexingPipeline", "GraphIndexStats", "ParseResult"]
