from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.core.graph_rag.config import GraphIndexConfig
from victor.core.graph_rag.indexing import (
    GraphIndexingPipeline,
    GraphIndexStats,
    ParseResult,
    run_indexing_with_lock,
)
from victor.storage.graph.protocol import GraphEdge, GraphNode


class _RecordingGraphStore:
    def __init__(self) -> None:
        self.in_write_batch = False
        self.write_batch_entries = 0
        self._write_batch_depth = 0
        self.calls: list[tuple[str, bool, int | None]] = []
        self.nodes: list = []

    @asynccontextmanager
    async def write_batch(self):
        is_outermost = self._write_batch_depth == 0
        if is_outermost:
            self.in_write_batch = True
            self.write_batch_entries += 1
        self._write_batch_depth += 1
        try:
            yield
        finally:
            self._write_batch_depth -= 1
            if self._write_batch_depth == 0:
                self.in_write_batch = False

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert_nodes(self, nodes):
        rows = list(nodes)
        self.nodes.extend(rows)
        self.calls.append(("nodes", self.in_write_batch, len(rows)))

    async def get_all_nodes(self):
        """The cross-file resolution passes build their name indices from here.

        They used to read the SQLite ``graph_node`` table with raw SQL, which
        returned nothing on any other backend. Reading through the store is what
        makes those passes backend-agnostic, so the fake has to hold nodes.
        """
        return list(self.nodes)

    async def upsert_edges(self, edges):
        rows = list(edges)
        self.calls.append(("edges", self.in_write_batch, len(rows)))

    async def update_file_mtime(self, file: str, mtime: float) -> None:
        self.calls.append(("mtime", self.in_write_batch, None))

    async def delete_by_file(self, file: str) -> None:
        self.calls.append(("delete", self.in_write_batch, None))

    async def get_stale_files(self, file_mtimes):
        return []


class _FakeNode:
    def __init__(
        self,
        node_type: str,
        *,
        text: bytes | str = b"",
        named_children: list["_FakeNode"] | None = None,
        field_children: dict[str, "_FakeNode"] | None = None,
    ) -> None:
        self.type = node_type
        self.text = text
        self.named_children = named_children or []
        self.children = self.named_children
        self._field_children = field_children or {}

    def child_by_field_name(self, name: str):
        return self._field_children.get(name)


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_uses_store_write_batch_for_file_writes(
    monkeypatch, tmp_path: Path
):
    file_path = tmp_path / "sample.py"
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")

    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    async def _fake_extract_symbols(_file_path: Path, _language: str):
        return [
            GraphNode(
                node_id="symbol:sample.py:foo",
                type="function",
                name="foo",
                file=str(_file_path),
                line=1,
                lang="python",
            )
        ]

    async def _fake_build_symbol_edges(_nodes, _file_path: Path):
        return [
            GraphEdge(
                src="file:sample.py",
                dst="symbol:sample.py:foo",
                type="CONTAINS",
            )
        ]

    monkeypatch.setattr(pipeline, "_extract_symbols", _fake_extract_symbols)
    monkeypatch.setattr(pipeline, "_build_symbol_edges", _fake_build_symbol_edges)

    stats = await pipeline._process_file(file_path)

    assert stats.files_processed == 1
    assert graph_store.write_batch_entries == 1
    assert graph_store.calls == [
        ("nodes", True, 1),
        ("edges", True, 1),
        ("mtime", True, None),
    ]


def test_extract_name_from_node_uses_type_field_for_impl_blocks():
    """`impl Default for Foo` must extract as "Foo" (the implementing type),
    not "Default" (the trait). Tree-sitter exposes the implementing type via
    the ``type`` field; the trait via ``trait``. Without this, every
    Default::default() call binds to all 800k user-defined `fn default`
    methods because they're all keyed by impl_name="Default".

    Verifies the extractor consults child_by_field_name("type") for impl_item
    nodes before falling back to the first-identifier scan (which would
    return the trait name).
    """
    pytest.importorskip("tree_sitter_rust")
    import tree_sitter
    import tree_sitter_rust

    lang = tree_sitter.Language(tree_sitter_rust.language())
    parser = tree_sitter.Parser(lang)

    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=Path("."),
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    cases = [
        (b"impl Foo { fn a() {} }", "Foo"),
        (b"impl Default for Foo { fn default() {} }", "Foo"),
        (b"impl<T> Display for Bar<T> { fn fmt() {} }", "Bar"),
    ]
    for src, expected in cases:
        tree = parser.parse(src)
        impl_node = next(c for c in tree.root_node.children if c.type == "impl_item")
        assert (
            pipeline._extract_name_from_node(impl_node) == expected
        ), f"src={src!r} expected={expected!r}"


def test_extract_name_from_node_prefers_direct_name_field(tmp_path: Path):
    pipeline = GraphIndexingPipeline(
        _RecordingGraphStore(),
        GraphIndexConfig(
            root_path=tmp_path,
            enable_ccg=False,
            enable_embeddings=False,
            enable_subgraph_cache=False,
        ),
    )

    name_node = _FakeNode("identifier", text=b"fast_name")
    method_node = _FakeNode(
        "function_definition",
        named_children=[name_node],
        field_children={"name": name_node},
    )

    assert pipeline._extract_name_from_node(method_node) == "fast_name"


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_batches_multi_file_writes_with_single_store_transaction(
    monkeypatch, tmp_path: Path
):
    first_file = tmp_path / "first.py"
    second_file = tmp_path / "second.py"
    first_file.write_text("def first():\n    return 1\n", encoding="utf-8")
    second_file.write_text("def second():\n    return 2\n", encoding="utf-8")

    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    # Patch _parse_file_sync (runs in thread pool) — return pre-built symbol nodes.
    # CCG is disabled in config so ccg_nodes/ccg_edges are empty.
    def _fake_parse_file_sync(file_path: Path):
        stem = file_path.stem
        sym_node = GraphNode(
            node_id=f"symbol:{stem}.py:{stem}",
            type="function",
            name=stem,
            file=str(file_path),
            line=1,
            lang="python",
        )
        return ParseResult(
            file_path=file_path,
            language="python",
            symbol_nodes=[sym_node],
        )

    async def _fake_build_symbol_edges(_nodes, file_path: Path):
        stem = file_path.stem
        return [
            GraphEdge(
                src=f"file:{stem}.py",
                dst=f"symbol:{stem}.py:{stem}",
                type="CONTAINS",
            )
        ]

    monkeypatch.setattr(pipeline, "_parse_file_sync", _fake_parse_file_sync)
    monkeypatch.setattr(pipeline, "_build_symbol_edges", _fake_build_symbol_edges)

    stats = await pipeline._process_batch([first_file, second_file])

    assert stats.files_processed == 2
    # ONE write_batch transaction for the entire two-file batch
    assert graph_store.write_batch_entries == 1
    # Bulk write: all symbol nodes in one call, all edges in one call, then per-file mtimes
    assert graph_store.calls == [
        ("nodes", True, 2),  # both files' nodes merged into one upsert_nodes call
        ("edges", True, 2),  # both files' edges merged into one upsert_edges call
        ("mtime", True, None),  # first_file mtime
        ("mtime", True, None),  # second_file mtime
    ]


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_cleans_vanished_file_without_error(
    tmp_path: Path,
):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    missing_file = tmp_path / "missing.py"
    stats = await pipeline._process_file(missing_file)

    assert stats.files_processed == 0
    assert stats.files_deleted == 1
    assert stats.error_count == 0
    assert graph_store.calls == [("delete", False, None)]


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_excludes_root_level_coverage_temp_files(
    tmp_path: Path,
):
    (tmp_path / "module.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / ".coverage.Vijays-MacBook-Pro.local.69421.XNBbPQdx.c").write_text(
        "not really C source", encoding="utf-8"
    )

    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
        respect_gitignore=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    files = await pipeline._discover_files(tmp_path)

    assert files == [tmp_path / "module.py"]


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_discovers_files_by_language_then_path(
    tmp_path: Path,
):
    (tmp_path / "z_python.py").write_text("def zed():\n    return 1\n", encoding="utf-8")
    (tmp_path / "a_typescript.ts").write_text("export const value = 1\n", encoding="utf-8")
    (tmp_path / "b_python.py").write_text("def bee():\n    return 2\n", encoding="utf-8")
    (tmp_path / "a_javascript.js").write_text("export const value = 1\n", encoding="utf-8")

    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
        respect_gitignore=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    files = await pipeline._discover_files(tmp_path)

    assert files == [
        tmp_path / "a_javascript.js",
        tmp_path / "b_python.py",
        tmp_path / "z_python.py",
        tmp_path / "a_typescript.ts",
    ]


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_refreshes_module_metrics_after_changed_graph(
    monkeypatch,
    tmp_path: Path,
):
    store = MagicMock()
    store.initialize = AsyncMock()
    store.delete_by_repo = AsyncMock()
    events: list[tuple[str, object]] = []

    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
        incremental=False,
    )
    pipeline = GraphIndexingPipeline(store, config)
    monkeypatch.setattr(
        pipeline,
        "_discover_files",
        AsyncMock(return_value=[tmp_path / "sample.py"]),
    )
    monkeypatch.setattr(
        pipeline,
        "_process_batch",
        AsyncMock(return_value=GraphIndexStats(files_processed=1, nodes_created=1)),
    )

    class _FakeModuleAnalyzer:
        def __init__(self, *, project_path):
            events.append(("init", project_path))

        def compute_all(self):
            events.append(("compute", None))
            return ["metric"]

        def persist(self, metrics):
            events.append(("persist", metrics))

    monkeypatch.setattr(
        "victor.core.analysis.module_analyzer.ModuleAnalyzer",
        _FakeModuleAnalyzer,
    )

    stats = await pipeline.index_repository()

    assert stats.module_metrics_computed == 1
    assert events == [
        ("init", tmp_path.resolve()),
        ("compute", None),
        ("persist", ["metric"]),
    ]


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_ignores_files_that_vanish_before_incremental_planning(
    tmp_path: Path,
):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    missing_file = tmp_path / ".coverage.Vijays-MacBook-Pro.local.69421.XNBbPQdx.c"

    stats = await pipeline._prepare_incremental_work([missing_file])

    assert stats.files_deleted == 0
    assert stats.files_unchanged == 0
    assert pipeline._files_to_process == set()
    assert graph_store.calls == [("delete", False, None)]


@pytest.mark.asyncio
async def test_run_indexing_with_lock_uses_project_index_lock(monkeypatch, tmp_path: Path):
    lock_events: list[str] = []

    class _FakePathLock:
        async def __aenter__(self):
            lock_events.append("enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            lock_events.append("exit")
            return False

    class _FakeRegistry:
        async def acquire_lock(self, path, use_file_lock=True, timeout_seconds=300.0):
            assert use_file_lock is True
            assert timeout_seconds == 300.0
            lock_events.append(str(path))
            return _FakePathLock()

    async def _operation():
        lock_events.append("operation")
        return "indexed"

    monkeypatch.setattr(
        "victor.core.indexing.index_lock.IndexLockRegistry.get_instance",
        lambda: _FakeRegistry(),
    )

    result = await run_indexing_with_lock(tmp_path, _operation)

    assert result == "indexed"
    assert lock_events == [str(tmp_path.resolve()), "enter", "operation", "exit"]


async def _seed_symbols(
    pipeline, rows=None, *, name_rows=None, impl_rows=None, node_type="function"
) -> None:
    """Seed the graph store with the symbols a resolution pass binds against.

    Mirrors what ``_FakeProjectDatabaseManager`` used to feed the raw SQL:

    - ``name_rows`` — ``(name, node_id[, file])`` callable symbols, the
      project-wide leaf-name index.
    - ``impl_rows`` — ``(impl_type, method_name, node_id)``. Recreated as a
      method node whose ``parent_id`` points at an ``impl`` node named
      ``impl_type``, which is the real shape the resolver reads.

    The passes now build these indices via ``get_all_nodes()`` rather than
    querying SQLite directly, so they work on any backend — a ProximaDB-backed
    repo previously resolved zero cross-file edges because ``graph_node`` was
    empty there.
    """
    from victor.storage.graph import GraphNode

    # Keyed by node_id: a test may list the same symbol in both name_rows and
    # impl_rows (the leaf-name index and the receiver-typed index overlap).
    # Emitting it twice would double its fanout and change what the resolver
    # decides, so the impl_rows pass augments the existing node rather than
    # adding a second one.
    by_id: dict = {}
    for row in list(rows or name_rows or []):
        name, node_id = row[0], row[1]
        file = row[2] if len(row) >= 3 and row[2] is not None else ""
        by_id[node_id] = GraphNode(node_id=node_id, type=node_type, name=name, file=file)

    for impl_type, method_name, node_id in list(impl_rows or []):
        impl_id = f"impl:{impl_type}"
        if impl_id not in by_id:
            by_id[impl_id] = GraphNode(node_id=impl_id, type="impl", name=impl_type, file="")
        existing = by_id.get(node_id)
        by_id[node_id] = GraphNode(
            node_id=node_id,
            type="method",
            name=method_name,
            file=existing.file if existing else "",
            parent_id=impl_id,
        )

    await pipeline.graph_store.upsert_nodes(list(by_id.values()))


class _FakeProjectDatabaseManager:
    """Stand-in for ProjectDatabaseManager.

    Returns ``name_rows`` for the project-wide leaf-name SELECT and
    ``impl_rows`` for the impl-type JOIN that powers receiver-typed
    resolution. Distinguishes the two by looking for ``JOIN`` in the query.

    ``name_rows`` entries may be 2-tuples ``(name, node_id)`` (legacy) or
    3-tuples ``(name, node_id, file)``. The resolver uses ``file`` for
    same-file preference in name-only resolution; 2-tuples are padded with
    None to keep older tests compatible.
    """

    def __init__(
        self,
        rows: list[tuple[str, str]] | None = None,
        *,
        name_rows: list[tuple] | None = None,
        impl_rows: list[tuple[str, str, str]] | None = None,
    ):
        raw = list(rows or name_rows or [])
        self._name_rows = [row if len(row) >= 3 else (*row, None) for row in raw]
        self._impl_rows = list(impl_rows or [])

    def __call__(self, _project_path):
        return self

    def _get_raw_connection(self):
        outer = self

        class _Conn:
            def execute(self, query):
                # Discriminator: the impl-type lookup selects "AS impl_type";
                # the leaf-name lookup never does. (Both queries now use JOIN
                # since the name query also LEFT-joins to filter trait
                # impls.)
                if "impl_type" in query:
                    return iter(outer._impl_rows)
                return iter(outer._name_rows)

        return _Conn()


@pytest.mark.asyncio
async def test_resolve_cross_file_calls_returns_zero_when_buffer_empty(tmp_path: Path):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 0
    assert graph_store.calls == []


@pytest.mark.asyncio
async def test_resolve_cross_file_calls_emits_edges_for_cross_file_callees(
    monkeypatch, tmp_path: Path
):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    # Two raw call records: caller_a calls "shared_fn", caller_b calls "shared_fn".
    pipeline._pending_call_records = [
        ("caller_a", "shared_fn", None, False),
        ("caller_b", "shared_fn", None, False),
    ]

    # Project-wide name index has one callee, in a different file (cross-file).
    await _seed_symbols(pipeline, [("shared_fn", "callee_x")])

    captured_edges: list[GraphEdge] = []

    async def _capture(edges):
        captured_edges.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 2
    assert {(e.src, e.dst, e.type) for e in captured_edges} == {
        ("caller_a", "callee_x", "CALLS"),
        ("caller_b", "callee_x", "CALLS"),
    }
    # Buffer is drained after the pass so a subsequent index_repository starts clean.
    assert pipeline._pending_call_records == []


@pytest.mark.asyncio
async def test_resolve_cross_file_calls_skips_self_loops(monkeypatch, tmp_path: Path):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("node_self", "recurse", None, False)]

    await _seed_symbols(pipeline, [("recurse", "node_self")])

    captured: list[GraphEdge] = []
    monkeypatch.setattr(graph_store, "upsert_edges", lambda edges: captured.extend(edges))

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 0
    assert captured == []


@pytest.mark.asyncio
async def test_resolve_cross_file_calls_respects_fanout_cap(monkeypatch, tmp_path: Path):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    # Cap at 2 candidates per leaf name; ambiguous names beyond that skip resolution.
    config.cross_file_call_max_fanout = 2  # type: ignore[attr-defined]
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [
        ("caller", "popular", None, False),  # 3 candidates -> skipped (above cap)
        ("caller", "rare", None, False),  # 1 candidate  -> emitted
    ]

    await _seed_symbols(
        pipeline,
        [
            ("popular", "p1"),
            ("popular", "p2"),
            ("popular", "p3"),
            ("rare", "r1"),
        ],
    )

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 1
    assert {(e.src, e.dst) for e in captured} == {("caller", "r1")}


def test_graph_index_stats_to_dict_includes_cross_file_calls_resolved():
    stats = GraphIndexStats(cross_file_calls_resolved=42)
    assert stats.to_dict()["cross_file_calls_resolved"] == 42


def test_tree_sitter_parser_cache_is_lru_one_per_language(monkeypatch, tmp_path: Path):
    """Switching languages evicts the prior language's parser (bounded memory)."""
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    constructed: list[str] = []

    def _fake_create(language: str):
        constructed.append(language)
        return MagicMock(name=f"parser-{language}")

    monkeypatch.setattr(pipeline, "_create_ts_parser", _fake_create)

    py = pipeline._get_tree_sitter_parser("python")
    # Same language -> cache hit, no new construction
    py_again = pipeline._get_tree_sitter_parser("python")
    assert py is py_again
    assert constructed == ["python"]
    assert list(pipeline._thread_local.parser_cache.keys()) == ["python"]

    # Switch language -> evict python, load rust
    rs = pipeline._get_tree_sitter_parser("rust")
    assert constructed == ["python", "rust"]
    assert list(pipeline._thread_local.parser_cache.keys()) == ["rust"]
    assert pipeline._thread_local.parser_cache["rust"] is rs

    # Switch back -> python must be reconstructed (it was evicted)
    pipeline._get_tree_sitter_parser("python")
    assert constructed == ["python", "rust", "python"]
    assert list(pipeline._thread_local.parser_cache.keys()) == ["python"]


# -----------------------------------------------------------------------------
# Receiver-typed resolution: a method call obj.method() with a known receiver
# type binds only to methods inside `impl T`, instead of fanning out to every
# `method` defined anywhere in the project.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_filters_by_impl_type_when_receiver_known(monkeypatch, tmp_path: Path):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("caller", "render", "Foo", True)]

    # Project-wide name index says "render" exists 4 times (above default fanout=25 we'd
    # still emit; below we keep it small to show impl-type filter wins anyway).
    name_rows = [
        ("render", "render_foo_id"),
        ("render", "render_bar_id"),
        ("render", "render_baz_id"),
        ("render", "render_qux_id"),
    ]
    # Impl-type join: only render_foo_id lives inside impl Foo.
    impl_rows = [
        ("Foo", "render", "render_foo_id"),
        ("Bar", "render", "render_bar_id"),
        ("Baz", "render", "render_baz_id"),
        ("Qux", "render", "render_qux_id"),
    ]
    await _seed_symbols(pipeline, name_rows=name_rows, impl_rows=impl_rows)

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 1
    assert {(e.src, e.dst) for e in captured} == {("caller", "render_foo_id")}


@pytest.mark.asyncio
async def test_resolve_does_not_fall_back_when_receiver_type_unmatched(monkeypatch, tmp_path: Path):
    """If the receiver type is set but no impl T::method matches, drop the call.

    The receiver type tells us the call targets a specific T::method. If T isn't
    in our graph (typically a stdlib type like Vec or HashMap, or an external
    crate), name-only fallback would fan out to every user-defined method with
    the same leaf name -- almost certainly the wrong type. That observed
    behavior produced large inflation on names like `iter`, `format`, `clone`
    where 10-20 user impls share the leaf name. Strict receiver-typed
    semantics: typed lookup is exact, and a miss means unresolved, not
    name-only.
    """
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("caller", "render", "MysteryType", True)]

    name_rows = [("render", "render_foo_id")]
    impl_rows = [("Foo", "render", "render_foo_id")]  # no impl MysteryType
    await _seed_symbols(pipeline, name_rows=name_rows, impl_rows=impl_rows)

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 0
    assert captured == []


@pytest.mark.asyncio
async def test_resolve_drops_method_calls_with_no_inferable_receiver(monkeypatch, tmp_path: Path):
    """Method-syntax call (`x.method()`) with receiver_type=None must NOT fall
    back to name-only. Reasoning: the user wrote dot-dispatch, so they wanted
    a specific impl; if we couldn't infer the type, binding to user-defined
    methods of unrelated types with the same leaf name is almost always wrong.

    Plain function calls (is_method_call=False) keep the name-only fallback
    because `func()` is globally unambiguous in well-named codebases.
    """
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    # 4-tuple buffer entries: (caller_id, callee_name, receiver_type, is_method_call)
    pipeline._pending_call_records = [
        ("caller", "collect", None, True),  # method call, no type -> drop
        ("caller", "free_fn", None, False),  # plain call -> name-only fallback works
    ]

    # 10 user-defined `collect` methods (below fanout cap, so old behavior
    # would emit edges to all 10). `free_fn` has exactly one definition.
    impl_rows = [(f"T{i}", "collect", f"collect_{i}") for i in range(10)] + [
        ("Helper", "free_fn", "free_fn_id"),
    ]
    name_rows = [("collect", f"collect_{i}") for i in range(10)] + [
        ("free_fn", "free_fn_id"),
    ]
    await _seed_symbols(pipeline, name_rows=name_rows, impl_rows=impl_rows)

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 1
    assert {(e.src, e.dst) for e in captured} == {("caller", "free_fn_id")}


@pytest.mark.asyncio
async def test_resolve_does_not_fanout_when_external_stdlib_receiver(monkeypatch, tmp_path: Path):
    """Regression: `vec.iter()` where vec: Vec (stdlib) must not fan out.

    Reproduces the inflation observed on proximaDB: one call site with
    receiver_type='Vec' was emitting edges to all 12 user-defined `iter`
    methods because name-only fallback found 12 candidates below the cap of
    25. With strict receiver-typed semantics, this call is unresolved (Vec
    isn't in user code) and zero edges are emitted.
    """
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("caller", "iter", "Vec", True)]

    # 12 user-defined iter methods on various unrelated types, all under fanout cap.
    impl_rows = [
        (t, "iter", f"iter_{t}")
        for t in (
            "BTree",
            "SkipList",
            "ResultSet",
            "ZeroOverhead",
            "Ultra",
            "QuantBatch",
            "PartSet",
            "Cache",
            "Lru",
            "TypedMeta",
            "LabelSet",
            "CapSet",
        )
    ]
    name_rows = [(row[1], row[2]) for row in impl_rows]
    await _seed_symbols(pipeline, name_rows=name_rows, impl_rows=impl_rows)

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    # With strict semantics: 0 edges. (Old behavior would emit 12.)
    assert emitted == 0
    assert captured == []


@pytest.mark.asyncio
async def test_resolve_receiver_typed_match_bypasses_fanout_cap(monkeypatch, tmp_path: Path):
    """A receiver-typed match is precise enough that fanout cap shouldn't apply.

    If 30 impls of Foo all define `method`, the receiver-typed lookup should still
    emit edges to all of them (the user wrote `obj: Foo`; that's an unambiguous
    intent compared to a bare leaf-name match).
    """
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    config.cross_file_call_max_fanout = 2  # type: ignore[attr-defined]
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("caller", "method", "Foo", True)]

    # Pathological: 5 different `method` definitions inside impl Foo (e.g., from
    # multiple `impl Foo` blocks scattered across files).
    impl_rows = [("Foo", "method", f"foo_method_{i}") for i in range(5)] + [
        ("Other", "method", "other_method_id"),
    ]
    name_rows = [(row[1], row[2]) for row in impl_rows]
    await _seed_symbols(pipeline, name_rows=name_rows, impl_rows=impl_rows)

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    # All 5 Foo::method nodes are bound; Other::method is not.
    assert emitted == 5
    dsts = {e.dst for e in captured}
    assert dsts == {f"foo_method_{i}" for i in range(5)}


# -----------------------------------------------------------------------------
# Same-file preference for plain function calls: when name-only resolution
# finds candidates in the caller's file, restrict to those. Rust function
# name resolution is module-scoped; a local `fn inputs(t)` helper called
# inside a test module shouldn't fan out to every other module's `fn inputs`
# with the same leaf name. (Method calls don't need this because typed
# dispatch already routes them precisely.)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_name_only_resolution_prefers_same_file_candidate(monkeypatch, tmp_path: Path):
    """A plain function call inside file A binds only to the file-A candidate,
    not every same-leaf-name function elsewhere.

    Reproduces the audit finding: `inputs(&t)` (a test helper) had 11
    project-wide candidates, all bound to a single call site under the
    fanout cap. Same-file preference restricts to the local helper.
    """
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    # Plain function call: receiver_type=None, is_method_call=False.
    pipeline._pending_call_records = [("caller_a", "inputs", None, False)]

    # name_rows: (name, node_id, file). caller_a lives in file_a; there are
    # `inputs` definitions in file_a, file_b, and file_c.
    name_rows = [
        ("caller_a_self", "caller_a", "src/file_a.rs"),  # so caller's file is known
        ("inputs", "inputs_a", "src/file_a.rs"),
        ("inputs", "inputs_b", "src/file_b.rs"),
        ("inputs", "inputs_c", "src/file_c.rs"),
    ]
    await _seed_symbols(pipeline, name_rows=name_rows)

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 1
    assert {(e.src, e.dst) for e in captured} == {("caller_a", "inputs_a")}


@pytest.mark.asyncio
async def test_name_index_excludes_trait_impl_methods(tmp_path: Path):
    """Trait method names must not appear in the name-only candidate list.

    Names like `drop`, `fmt`, `eq`, `hash` are invoked by the compiler or by
    typed method dispatch, never by a plain call. Without the filter, `drop(x)`
    (std::mem::drop) fanned out across every user `impl Drop for T { fn drop }`.

    Asserts the *behaviour* rather than the SQL text: this index is now built
    from the graph store so it works on any backend, so there is no query string
    left to inspect.
    """
    from victor.storage.graph import GraphNode

    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    # `impl Drop for Widget { fn drop }` — a trait impl, so `drop` is excluded.
    # `impl Widget { fn drop }` — inherent, so its `drop` stays a candidate.
    await graph_store.upsert_nodes(
        [
            GraphNode(
                node_id="trait_impl",
                type="impl",
                name="Drop",
                file="src/w.rs",
                signature="impl Drop for Widget {",
            ),
            GraphNode(
                node_id="trait_drop",
                type="method",
                name="drop",
                file="src/w.rs",
                parent_id="trait_impl",
            ),
            GraphNode(
                node_id="inherent_impl",
                type="impl",
                name="Widget",
                file="src/w.rs",
                signature="impl Widget {",
            ),
            GraphNode(
                node_id="inherent_drop",
                type="method",
                name="drop",
                file="src/w.rs",
                parent_id="inherent_impl",
            ),
        ]
    )

    pipeline._pending_call_records = [("caller", "drop", None, False)]

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    graph_store.upsert_edges = _capture  # type: ignore[assignment]
    await pipeline._resolve_cross_file_calls(tmp_path)

    targets = {e.dst for e in captured}
    assert "trait_drop" not in targets, "trait-impl method must not be a call candidate"
    assert "inherent_drop" in targets, "inherent method must remain a candidate"


@pytest.mark.asyncio
async def test_name_only_falls_through_when_no_same_file_candidate(monkeypatch, tmp_path: Path):
    """If no candidate is in the caller's file, keep the original
    cross-file candidate set (subject to fanout cap). Conservative: we
    only restrict when same-file candidates exist."""
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("caller_a", "helper", None, False)]

    # caller in file_a, but helper only defined in file_b and file_c.
    name_rows = [
        ("caller_a_self", "caller_a", "src/file_a.rs"),
        ("helper", "helper_b", "src/file_b.rs"),
        ("helper", "helper_c", "src/file_c.rs"),
    ]
    await _seed_symbols(pipeline, name_rows=name_rows)

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    # No same-file match -> fall through; both cross-file candidates emitted
    # (2 < fanout cap of 25, so they pass).
    assert emitted == 2
    assert {e.dst for e in captured} == {"helper_b", "helper_c"}


# ────────────────────────────────────────────────────────────────────────
# TSA-4: TreeSitterAnalysisProtocol provider integration in _parse_file_sync
# ────────────────────────────────────────────────────────────────────────


class _FakeAnalysisProvider:
    """Test double for TreeSitterAnalysisProtocol."""

    def __init__(
        self,
        *,
        supported: set[str] | None = None,
        symbols: list[dict] | None = None,
        raise_on_extract: bool = False,
    ) -> None:
        self.supported = supported or {"python"}
        self.symbols = symbols or []
        self.raise_on_extract = raise_on_extract
        self.calls: list[str] = []

    def supports_language(self, language: str) -> bool:
        return language in self.supported

    def extract_symbols(self, content, language, *, file_path):
        self.calls.append(file_path)
        if self.raise_on_extract:
            raise RuntimeError("provider boom")
        return list(self.symbols)


def _make_pipeline(tmp_path: Path) -> GraphIndexingPipeline:
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    return GraphIndexingPipeline(_RecordingGraphStore(), config)


def test_parse_file_sync_uses_enhanced_provider_when_available(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "a.py"
    file_path.write_text("def whatever(): pass\n", encoding="utf-8")

    pipeline = _make_pipeline(tmp_path)
    provider = _FakeAnalysisProvider(
        symbols=[
            {
                "name": "whatever",
                "symbol_type": "function",
                "file_path": str(file_path),
                "line_start": 1,
                "line_end": 1,
                "ast_kind": "function_definition",
            }
        ],
    )
    monkeypatch.setattr(pipeline, "_get_analysis_provider", lambda: (provider, True))

    # Guarantee a failure if the hardcoded path runs instead of the provider.
    def _should_not_run(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("_extract_definitions should not run when provider succeeds")

    monkeypatch.setattr(pipeline, "_extract_definitions", _should_not_run)

    result = pipeline._parse_file_sync(file_path)

    assert provider.calls == ["a.py"]
    # _parse_file_sync prepends a synthetic module node so IMPORTS edges and
    # top-level CONTAINS edges have something to attach to. The extracted
    # symbol follows it.
    assert [n.name for n in result.symbol_nodes] == ["a", "whatever"]
    assert result.symbol_nodes[0].type == "module"
    whatever = result.symbol_nodes[1]
    assert whatever.lang == "python"
    assert whatever.ast_kind == "function_definition"
    # Top-level symbols inherit the module as parent so the existing CONTAINS
    # emission produces module → symbol edges.
    assert whatever.parent_id == result.symbol_nodes[0].node_id
    assert result.provider_fallback is False


def test_parse_file_sync_falls_back_when_provider_raises(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "b.py"
    file_path.write_text("def whatever(): pass\n", encoding="utf-8")

    pipeline = _make_pipeline(tmp_path)
    provider = _FakeAnalysisProvider(raise_on_extract=True)
    monkeypatch.setattr(pipeline, "_get_analysis_provider", lambda: (provider, True))

    # Replace the hardcoded path with a deterministic stub so the test does
    # not depend on tree-sitter wheel installation.
    fallback_node = GraphNode(
        node_id="fallback",
        type="function",
        name="whatever",
        file=str(file_path),
        line=1,
        lang="python",
    )
    monkeypatch.setattr(
        pipeline,
        "_extract_symbols_fallback",
        lambda *args, **kwargs: [fallback_node],
    )
    monkeypatch.setattr(pipeline, "_create_ts_parser", lambda lang: None)

    result = pipeline._parse_file_sync(file_path)

    assert result.provider_fallback is True
    # Module node is injected by both extraction paths so CONTAINS edges
    # work uniformly; fallback symbols follow it.
    assert [n.name for n in result.symbol_nodes] == ["b", "whatever"]
    assert result.symbol_nodes[0].type == "module"


def test_parse_file_sync_skips_provider_when_only_stub_registered(
    monkeypatch, tmp_path: Path
) -> None:
    file_path = tmp_path / "c.py"
    file_path.write_text("def whatever(): pass\n", encoding="utf-8")

    pipeline = _make_pipeline(tmp_path)
    provider = _FakeAnalysisProvider()
    # enhanced=False mirrors the null stub state at root bootstrap.
    monkeypatch.setattr(pipeline, "_get_analysis_provider", lambda: (provider, False))

    def _stub_extract(*args, **kwargs):
        return [
            GraphNode(
                node_id="legacy",
                type="function",
                name="whatever",
                file=str(file_path),
                line=1,
                lang="python",
            )
        ]

    monkeypatch.setattr(pipeline, "_extract_symbols_fallback", _stub_extract)
    monkeypatch.setattr(pipeline, "_create_ts_parser", lambda lang: None)

    result = pipeline._parse_file_sync(file_path)

    assert provider.calls == [], "provider must not be called when enhanced is False"
    # Legacy path also synthesizes a module node — see above.
    assert [n.name for n in result.symbol_nodes] == ["c", "whatever"]
    assert result.symbol_nodes[0].type == "module"
    assert result.provider_fallback is False


def test_provider_symbols_to_graph_nodes_maps_dict_fields() -> None:
    pipeline = _make_pipeline(Path("/tmp"))
    symbols = [
        {
            "name": "Foo",
            "symbol_type": "class",
            "file_path": "/x.py",
            "line_start": 10,
            "line_end": 20,
            "ast_kind": "class_definition",
            "signature": "class Foo(Base):",
            "visibility": "public",
        }
    ]
    nodes = pipeline._provider_symbols_to_graph_nodes(symbols, Path("/x.py"), "python")
    assert len(nodes) == 1
    node = nodes[0]
    assert node.name == "Foo"
    assert node.file == "/x.py"
    assert node.line == 10
    assert node.end_line == 20
    assert node.signature == "class Foo(Base):"
    assert node.visibility == "public"
    assert node.ast_kind == "class_definition"


def test_provider_symbols_to_graph_nodes_preserves_codegraph_v2_identity() -> None:
    pipeline = _make_pipeline(Path("/tmp"))
    symbols = [
        {
            "name": "f",
            "symbol_type": "function",
            "file_path": "pkg/m.py",
            "line_start": 9,
            "line_end": 10,
            "ast_kind": "function_definition",
            "symbol_id": "stable-v2-id",
        }
    ]
    node = pipeline._provider_symbols_to_graph_nodes(symbols, Path("pkg/m.py"), "python")[0]
    assert node.node_id == "stable-v2-id"


def test_module_node_id_matches_codegraph_file_identity() -> None:
    codegraph = pytest.importorskip("victor_codegraph")
    from victor.core.graph_rag.indexing import _module_node_id

    expected = codegraph.stable_symbol_key(
        "repository", codegraph.CodeSymbolType.FILE, "pkg/m.py", None, "pkg/m.py"
    )
    assert _module_node_id("pkg/m.py") == expected


@pytest.mark.asyncio
async def test_core_call_edges_preserve_codegraph_v2_ids(tmp_path: Path) -> None:
    codegraph = pytest.importorskip("victor_codegraph")
    from victor.core.graph_rag.indexing import _get_graph_types

    source = "def caller(): return target()\n\ndef target(): return 1\n"
    path = tmp_path / "m.py"
    path.write_text(source)
    pipeline = _make_pipeline(tmp_path)
    parsed = codegraph.parse(source, file_path="m.py")
    GraphNode, _ = _get_graph_types()
    nodes = [
        GraphNode(
            node_id=s.id,
            type=s.symbol_type.name.lower(),
            name=s.simple_name,
            file="m.py",
            line=s.location.start_line,
            lang="python",
        )
        for s in parsed.symbols
    ]

    edges = await pipeline._build_calls_edges(nodes, path)
    call = next(edge for edge in edges if edge.type.value == "CALLS")
    expected = next(r for r in parsed.relations if r.relation_type.name == "CALLS")
    assert (call.src, call.dst) == (expected.from_symbol_id, expected.to_symbol_id)


@pytest.mark.asyncio
async def test_codegraph_repository_relations_use_import_context_and_suppress_name_fanout(
    tmp_path: Path,
) -> None:
    """Repository resolution must bind a call only to the imported definition.

    The legacy resolver fans a leaf-name call out to every matching symbol.  The
    v2 repository resolver instead uses the caller's imports and preserves the
    canonical symbol ids.  Once that semantic result is available, the lossy
    legacy records for the same source must be discarded.
    """
    codegraph = pytest.importorskip("victor_codegraph")
    caller_path = tmp_path / "caller.py"
    caller_path.write_text(
        "from preferred import Base, target\n\n"
        "class Child(Base):\n    pass\n\n"
        "def caller():\n    return target()\n",
        encoding="utf-8",
    )
    (tmp_path / "preferred.py").write_text(
        "class Base:\n    pass\n\ndef target():\n    return 'preferred'\n",
        encoding="utf-8",
    )
    (tmp_path / "other.py").write_text(
        "class Base:\n    pass\n\ndef target():\n    return 'other'\n",
        encoding="utf-8",
    )

    graph_store = _RecordingGraphStore()
    pipeline = GraphIndexingPipeline(
        graph_store,
        GraphIndexConfig(
            root_path=tmp_path,
            enable_ccg=False,
            enable_embeddings=False,
            enable_subgraph_cache=False,
        ),
    )
    repository = codegraph.parse_repo(tmp_path)
    pipeline._codegraph_repository = repository
    caller = next(symbol for symbol in repository.symbols if symbol.simple_name == "caller")
    preferred = next(
        symbol
        for symbol in repository.symbols
        if symbol.simple_name == "target" and symbol.location.file_path == "preferred.py"
    )
    other = next(
        symbol
        for symbol in repository.symbols
        if symbol.simple_name == "target" and symbol.location.file_path == "other.py"
    )
    child = next(symbol for symbol in repository.symbols if symbol.simple_name == "Child")
    preferred_base = next(
        symbol
        for symbol in repository.symbols
        if symbol.simple_name == "Base" and symbol.location.file_path == "preferred.py"
    )
    other_base = next(
        symbol
        for symbol in repository.symbols
        if symbol.simple_name == "Base" and symbol.location.file_path == "other.py"
    )
    pipeline._codegraph_owned_symbol_ids = {caller.id, child.id}
    pipeline._codegraph_owned_files = {"caller.py"}
    pipeline._pending_call_records = [(caller.id, "target", None, False)]
    pipeline._pending_relationship_records = [(child.id, "Base", "INHERITS")]
    pipeline._pending_import_records = [
        (str(caller_path), "from preferred import target", "python")
    ]

    captured: list[GraphEdge] = []

    async def _capture_edges(edges):
        captured.extend(edges)

    graph_store.upsert_edges = _capture_edges

    calls, relationships, imports = await pipeline._resolve_codegraph_repository_relations()

    assert (calls, relationships, imports) == (1, 1, 1)
    call_edges = [edge for edge in captured if edge.type.value == "CALLS"]
    assert [(edge.src, edge.dst) for edge in call_edges] == [(caller.id, preferred.id)]
    assert all(edge.dst != other.id for edge in call_edges)
    relationship_edges = [edge for edge in captured if edge.type.value == "INHERITS"]
    assert [(edge.src, edge.dst) for edge in relationship_edges] == [(child.id, preferred_base.id)]
    assert all(edge.dst != other_base.id for edge in relationship_edges)
    assert pipeline._pending_call_records == []
    assert pipeline._pending_relationship_records == []
    assert pipeline._pending_import_records == []


@pytest.mark.asyncio
async def test_codegraph_repository_relation_fallback_preserves_legacy_buffers(
    tmp_path: Path,
) -> None:
    pipeline = GraphIndexingPipeline(
        _RecordingGraphStore(),
        GraphIndexConfig(
            root_path=tmp_path,
            enable_ccg=False,
            enable_embeddings=False,
            enable_subgraph_cache=False,
        ),
    )
    pending_call = ("caller", "target", None, False)
    pending_relationship = ("child", "Base", "INHERITS")
    pending_import = (str(tmp_path / "caller.py"), "import target", "python")
    pipeline._pending_call_records = [pending_call]
    pipeline._pending_relationship_records = [pending_relationship]
    pipeline._pending_import_records = [pending_import]

    assert await pipeline._resolve_codegraph_repository_relations() == (0, 0, 0)
    assert pipeline._pending_call_records == [pending_call]
    assert pipeline._pending_relationship_records == [pending_relationship]
    assert pipeline._pending_import_records == [pending_import]


@pytest.mark.asyncio
async def test_core_call_edges_reuse_prepared_repository_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    codegraph = pytest.importorskip("victor_codegraph")
    from victor.core.graph_rag.indexing import _get_graph_types

    source = "def caller(): return target()\n\ndef target(): return 1\n"
    path = tmp_path / "m.py"
    path.write_text(source, encoding="utf-8")
    repository = codegraph.parse_repo(tmp_path)
    pipeline = _make_pipeline(tmp_path)
    pipeline._codegraph_repository = repository
    GraphNode, _ = _get_graph_types()
    nodes = [
        GraphNode(
            node_id=symbol.id,
            type=symbol.symbol_type.name.lower(),
            name=symbol.simple_name,
            file="m.py",
            line=symbol.location.start_line,
            lang="python",
        )
        for symbol in repository.files["m.py"].symbols
    ]
    monkeypatch.setattr(
        codegraph,
        "parse",
        lambda *args, **kwargs: pytest.fail("prepared snapshot must avoid a per-file reparse"),
    )

    edges = await pipeline._build_calls_edges(nodes, path)

    assert len([edge for edge in edges if edge.type.value == "CALLS"]) == 1
    assert pipeline._codegraph_owned_files == {"m.py"}


@pytest.mark.asyncio
async def test_codegraph_repository_relations_cover_import_only_modules(tmp_path: Path) -> None:
    codegraph = pytest.importorskip("victor_codegraph")
    importer = tmp_path / "importer.py"
    importer.write_text("import target\n", encoding="utf-8")
    (tmp_path / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    graph_store = _RecordingGraphStore()
    pipeline = GraphIndexingPipeline(
        graph_store,
        GraphIndexConfig(
            root_path=tmp_path,
            enable_ccg=False,
            enable_embeddings=False,
            enable_subgraph_cache=False,
        ),
    )
    pipeline._codegraph_repository = codegraph.parse_repo(tmp_path)
    module_node = pipeline._make_module_node(importer, "python")

    await pipeline._build_calls_edges([module_node], importer)
    calls, relationships, imports = await pipeline._resolve_codegraph_repository_relations()

    assert (calls, relationships, imports) == (0, 0, 1)
    assert pipeline._codegraph_owned_files == {"importer.py"}


def test_provider_fallback_increments_stat_in_merge(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    target = GraphIndexStats()
    source = GraphIndexStats(provider_fallbacks=3)
    pipeline._merge_stats(target, source)
    assert target.provider_fallbacks == 3


@pytest.mark.asyncio
async def test_resolve_cross_file_relationships_returns_zero_when_buffer_empty(
    tmp_path: Path,
) -> None:
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    emitted = await pipeline._resolve_cross_file_relationships(tmp_path)
    assert emitted == 0


@pytest.mark.asyncio
async def test_resolve_cross_file_relationships_emits_inherits_edges(
    monkeypatch, tmp_path: Path
) -> None:
    """INHERITS/IMPLEMENTS/COMPOSITION records should resolve against the
    project-wide class index and produce edges, including cross-file ones.
    Regression guard: the TSA path used to drop all non-CALLS edges
    silently in the adapter.
    """
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    pipeline._pending_relationship_records = [
        ("child_id", "Parent", "INHERITS"),
        ("child_id", "Iface", "IMPLEMENTS"),
        ("child_id", "Helper", "COMPOSITION"),
    ]

    # INHERITS/IMPLEMENTS/COMPOSITION bind only to class-like types.
    await _seed_symbols(
        pipeline,
        [("Parent", "parent_id"), ("Iface", "iface_id"), ("Helper", "helper_id")],
        node_type="class",
    )

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_relationships(tmp_path)

    assert emitted == 3
    assert {(e.src, e.dst, e.type) for e in captured} == {
        ("child_id", "parent_id", "INHERITS"),
        ("child_id", "iface_id", "IMPLEMENTS"),
        ("child_id", "helper_id", "COMPOSITION"),
    }
    # Buffer drained after pass — next run starts clean.
    assert pipeline._pending_relationship_records == []


@pytest.mark.asyncio
async def test_resolve_cross_file_relationships_drops_self_loops(
    monkeypatch, tmp_path: Path
) -> None:
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    # Pathological case: a class named "Self" that "inherits" from itself.
    pipeline._pending_relationship_records = [("node_self", "Self", "INHERITS")]
    await _seed_symbols(pipeline, [("Self", "node_self")])

    captured: list[GraphEdge] = []
    monkeypatch.setattr(graph_store, "upsert_edges", lambda edges: captured.extend(edges))

    emitted = await pipeline._resolve_cross_file_relationships(tmp_path)
    assert emitted == 0
    assert captured == []


def test_parse_python_imports_handles_all_forms(tmp_path: Path) -> None:
    from victor.core.graph_rag.import_resolvers import PythonImportResolver

    src = tmp_path / "pkg" / "sub" / "mod.py"
    src.parent.mkdir(parents=True)
    src.write_text("")
    parse = PythonImportResolver().parse
    # Plain imports
    assert parse("import os", str(src), tmp_path) == ["os"]
    assert parse("import os.path", str(src), tmp_path) == ["os.path"]
    assert parse("import os as o", str(src), tmp_path) == ["os"]
    # Comma-separated imports
    assert parse("import a, b, c", str(src), tmp_path) == ["a", "b", "c"]
    # from-import variants emit base + base.name candidates so the resolver
    # can bind whichever shape exists on disk.
    assert parse("from typing import List", str(src), tmp_path) == [
        "typing",
        "typing.List",
    ]
    assert parse("from x.y import z", str(src), tmp_path) == ["x.y", "x.y.z"]
    # Relative imports resolve against the source file's package
    assert parse("from . import sibling", str(src), tmp_path) == [
        "pkg.sub",
        "pkg.sub.sibling",
    ]
    assert parse("from .sibling import thing", str(src), tmp_path) == [
        "pkg.sub.sibling",
        "pkg.sub.sibling.thing",
    ]
    # Two dots walks one package up
    assert parse("from ..other import x", str(src), tmp_path) == [
        "pkg.other",
        "pkg.other.x",
    ]
    # Aliases get stripped; wildcards drop out
    assert parse("from a import b as c, d", str(src), tmp_path) == ["a", "a.b", "a.d"]
    assert parse("from a import *", str(src), tmp_path) == ["a"]
    # Malformed / unrelated text returns empty list (no exception)
    assert parse("# just a comment", str(src), tmp_path) == []
    assert parse("from", str(src), tmp_path) == []


def test_resolve_module_to_path_prefers_module_over_package(tmp_path: Path) -> None:
    from victor.core.graph_rag.import_resolvers import (
        ImportResolverRegistry,
        PythonImportResolver,
    )

    resolver = PythonImportResolver()
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "child.py").write_text("")
    (tmp_path / "pkg" / "child_pkg").mkdir()
    (tmp_path / "pkg" / "child_pkg" / "__init__.py").write_text("")

    # Module file wins when both shapes could match.
    assert resolver.resolve("pkg.child", tmp_path) == (tmp_path / "pkg" / "child.py")
    # Falls back to __init__.py when only the package directory exists.
    assert resolver.resolve("pkg.child_pkg", tmp_path) == (
        tmp_path / "pkg" / "child_pkg" / "__init__.py"
    )
    # Unknown module: stdlib / third-party / typo — must return None so the
    # resolver can count it as "external" and skip rather than crash.
    assert resolver.resolve("definitely.not.here", tmp_path) is None
    # Languages without a registered strategy have no resolver (documented
    # limitation — the pipeline counts their records as unsupported).
    assert ImportResolverRegistry.create("go") is None


async def _seed_indexed_modules(pipeline, files) -> None:
    """Persist a module node per file through the pipeline's graph store.

    ``_resolve_imports`` filters IMPORTS targets to files that actually produced
    a module node, so a target with no node would dangle. That set used to come
    from raw SQL against SQLite's ``graph_node`` table — which returned nothing
    on any other backend — and is now read through ``get_all_nodes()``. Seeding
    the store is therefore how a test declares "these files were indexed".
    """
    from victor.storage.graph import GraphNode

    paths = [files] if isinstance(files, (str, Path)) else list(files)
    await pipeline.graph_store.upsert_nodes(
        [
            GraphNode(
                node_id=f"mod:{i}",
                type="module",
                name=Path(f).stem,
                file=f if isinstance(f, str) else pipeline._canonical_file_str(f),
            )
            for i, f in enumerate(paths)
        ]
    )


class _FakeIndexedFilesDb:
    """Stand-in for ProjectDatabaseManager that pretends a fixed set of
    files were indexed in this run. The resolver filters IMPORTS targets
    through this set to avoid dangling edges when a file resolves on disk
    but was skipped during indexing (exclude pattern, parse failure, …).
    """

    def __init__(self, indexed_files: list[str]):
        self._files = list(indexed_files)

    def __call__(self, _project_path):
        return self

    def _get_raw_connection(self):
        outer = self

        class _Conn:
            def execute(self, query):
                # Only one SELECT in _resolve_imports — return module file rows.
                return iter((f,) for f in outer._files)

        return _Conn()


@pytest.mark.asyncio
async def test_resolve_imports_emits_edges_between_module_nodes(
    monkeypatch, tmp_path: Path
) -> None:
    pipeline = _make_pipeline(tmp_path)
    # Create a minimal project tree so module resolution finds real files.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    src = tmp_path / "pkg" / "a.py"
    target = tmp_path / "pkg" / "b.py"
    src.write_text("from pkg.b import foo\n")
    target.write_text("def foo(): pass\n")

    # Match SqliteGraphStore's canonical storage form: repo-relative posix
    # paths. The resolver compares its computed canonical paths against this
    # set, so the test fixture must mirror that contract.
    await _seed_indexed_modules(
        pipeline, [pipeline._canonical_file_str(src), pipeline._canonical_file_str(target)]
    )

    pipeline._pending_import_records = [
        (str(src), "from pkg.b import foo", "python"),
    ]

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    pipeline.graph_store.upsert_edges = _capture  # type: ignore[assignment]

    emitted = await pipeline._resolve_imports(tmp_path)
    assert emitted == 1
    assert len(captured) == 1
    edge = captured[0]
    assert edge.type == "IMPORTS"
    # Edge endpoints must hash the canonical relative path — same form
    # _make_module_node uses, so the IMPORTS edge resolves to the real
    # module node row.
    from victor.core.graph_rag.indexing import _module_node_id

    assert edge.src == _module_node_id(pipeline._canonical_file_str(src))
    assert edge.dst == _module_node_id(pipeline._canonical_file_str(target))
    # Buffer drained after resolution so a subsequent index run starts clean.
    assert pipeline._pending_import_records == []


@pytest.mark.asyncio
async def test_resolve_imports_skips_stdlib_and_self(monkeypatch, tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    src = tmp_path / "x.py"
    src.write_text("")
    await _seed_indexed_modules(pipeline, [pipeline._canonical_file_str(src)])
    pipeline._pending_import_records = [
        # External — won't resolve to a project file.
        (str(src), "import os", "python"),
        # Self import that resolves back to the same file.
        (str(src), "import x", "python"),
    ]

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    pipeline.graph_store.upsert_edges = _capture  # type: ignore[assignment]

    emitted = await pipeline._resolve_imports(tmp_path)
    assert emitted == 0
    assert captured == []


@pytest.mark.asyncio
async def test_resolve_imports_deduplicates_repeated_pairs(monkeypatch, tmp_path: Path) -> None:
    """The same ``import x`` appearing twice in one file should only emit
    one IMPORTS edge — graph_edge is keyed by (src, dst, type)."""
    pipeline = _make_pipeline(tmp_path)
    src = tmp_path / "a.py"
    target = tmp_path / "b.py"
    src.write_text("import b\nimport b\n")
    target.write_text("")
    await _seed_indexed_modules(
        pipeline, [pipeline._canonical_file_str(src), pipeline._canonical_file_str(target)]
    )
    pipeline._pending_import_records = [
        (str(src), "import b", "python"),
        (str(src), "import b", "python"),
    ]

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    pipeline.graph_store.upsert_edges = _capture  # type: ignore[assignment]
    emitted = await pipeline._resolve_imports(tmp_path)
    assert emitted == 1
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_resolve_imports_unsupported_languages_are_skipped(
    monkeypatch, tmp_path: Path
) -> None:
    """Languages without a registered resolver strategy are counted as
    ``unsupported-language`` and produce no edges (instead of crashing)."""
    pipeline = _make_pipeline(tmp_path)
    src = tmp_path / "a.go"
    src.write_text("")
    await _seed_indexed_modules(pipeline, [pipeline._canonical_file_str(src)])
    pipeline._pending_import_records = [(str(src), 'import "fmt"', "go")]

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    pipeline.graph_store.upsert_edges = _capture  # type: ignore[assignment]
    emitted = await pipeline._resolve_imports(tmp_path)
    assert emitted == 0
    assert captured == []


@pytest.mark.asyncio
async def test_resolve_imports_rust_use_declarations(monkeypatch, tmp_path: Path) -> None:
    """Rust ``use`` declarations must produce module-level IMPORTS edges —
    both intra-crate (``crate::``) and across workspace crates. These edges
    are what Martin Ca/Ce are computed from, so without them Rust coupling
    metrics fall back to noisy CALLS fan-in.
    """
    pipeline = _make_pipeline(tmp_path)
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "mainapp"\nversion = "0.1.0"\n\n'
        '[workspace]\nmembers = ["crates/catalog"]\n'
    )
    (tmp_path / "src" / "services").mkdir(parents=True)
    (tmp_path / "src" / "storage").mkdir()
    src = tmp_path / "src" / "services" / "observer.rs"
    src.write_text("")
    storage_mod = tmp_path / "src" / "storage" / "mod.rs"
    storage_mod.write_text("")
    catalog = tmp_path / "crates" / "catalog"
    (catalog / "src").mkdir(parents=True)
    (catalog / "Cargo.toml").write_text('[package]\nname = "proxi-catalog"\nversion = "0.1.0"\n')
    catalog_lib = catalog / "src" / "lib.rs"
    catalog_lib.write_text("")

    await _seed_indexed_modules(
        pipeline,
        [
            pipeline._canonical_file_str(src),
            pipeline._canonical_file_str(storage_mod),
            pipeline._canonical_file_str(catalog_lib),
        ],
    )
    pipeline._pending_import_records = [
        (str(src), "use crate::storage::StorageEngine;", "rust"),
        (str(src), "use proxi_catalog::CatalogManager;", "rust"),
    ]

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    pipeline.graph_store.upsert_edges = _capture  # type: ignore[assignment]

    emitted = await pipeline._resolve_imports(tmp_path)
    assert emitted == 2
    from victor.core.graph_rag.indexing import _module_node_id

    src_id = _module_node_id(pipeline._canonical_file_str(src))
    assert {(e.src, e.dst, e.type) for e in captured} == {
        (src_id, _module_node_id(pipeline._canonical_file_str(storage_mod)), "IMPORTS"),
        (src_id, _module_node_id(pipeline._canonical_file_str(catalog_lib)), "IMPORTS"),
    }


def test_index_repository_resolves_provider_before_parse_burst(
    tmp_path: Path,
) -> None:
    """The analysis provider must be resolved on the event-loop thread
    BEFORE any file is parsed. Lazy resolution inside worker threads races
    plugin discovery: files parsed while discovery is importing grammars
    see supports_language() == False and silently lose their raw imports
    (observed: 51/92 vscode-victor files had no IMPORTS edges).

    Sync test with an explicit asyncio.run so the ordering assertion is
    not affected by the shared pytest-asyncio loop fixtures.
    """
    import asyncio

    (tmp_path / "a.py").write_text("import os\n")
    pipeline = _make_pipeline(tmp_path)

    events: list[str] = []
    orig_resolve = pipeline._get_analysis_provider
    orig_discover = pipeline._discover_files

    def spy_resolve():
        events.append("resolve")
        return orig_resolve()

    async def spy_discover(root):
        events.append("discover")
        return await orig_discover(root)

    pipeline._get_analysis_provider = spy_resolve  # type: ignore[method-assign]
    pipeline._discover_files = spy_discover  # type: ignore[method-assign]

    asyncio.run(pipeline.index_repository(root_path=tmp_path))

    # Discovery precedes all parsing, so resolve-before-discover guarantees
    # no worker thread can observe a half-initialized provider.
    assert "resolve" in events and "discover" in events
    assert events.index("resolve") < events.index("discover")


@pytest.mark.asyncio
async def test_resolve_imports_cpp_includes(monkeypatch, tmp_path: Path) -> None:
    """C/C++ #include directives (quoted-relative and root-relative) must
    produce module-level IMPORTS edges; system headers must not."""
    pipeline = _make_pipeline(tmp_path)
    (tmp_path / "server" / "logging").mkdir(parents=True)
    (tmp_path / "runtime").mkdir()
    src = tmp_path / "runtime" / "engine.cpp"
    src.write_text("")
    header = tmp_path / "runtime" / "engine.h"
    header.write_text("")
    logger_h = tmp_path / "server" / "logging" / "logger.h"
    logger_h.write_text("")

    await _seed_indexed_modules(
        pipeline, [pipeline._canonical_file_str(p) for p in (src, header, logger_h)]
    )
    pipeline._pending_import_records = [
        (str(src), '#include "engine.h"', "cpp"),
        (str(src), '#include "server/logging/logger.h"', "cpp"),
        (str(src), "#include <vector>", "cpp"),
    ]

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    pipeline.graph_store.upsert_edges = _capture  # type: ignore[assignment]

    emitted = await pipeline._resolve_imports(tmp_path)
    assert emitted == 2
    from victor.core.graph_rag.indexing import _module_node_id

    src_id = _module_node_id(pipeline._canonical_file_str(src))
    assert {(e.src, e.dst) for e in captured} == {
        (src_id, _module_node_id(pipeline._canonical_file_str(header))),
        (src_id, _module_node_id(pipeline._canonical_file_str(logger_h))),
    }


@pytest.mark.asyncio
async def test_resolve_imports_typescript_imports(monkeypatch, tmp_path: Path) -> None:
    """TypeScript imports (relative, tsconfig alias, barrel re-export) must
    produce module-level IMPORTS edges for Martin Ca/Ce."""
    pipeline = _make_pipeline(tmp_path)
    (tmp_path / "tsconfig.json").write_text(
        '{"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["src/*"]}}}'
    )
    (tmp_path / "src" / "utils").mkdir(parents=True)
    (tmp_path / "src" / "components").mkdir()
    src = tmp_path / "src" / "main.ts"
    src.write_text("")
    graph_util = tmp_path / "src" / "utils" / "graph.ts"
    graph_util.write_text("")
    barrel = tmp_path / "src" / "components" / "index.ts"
    barrel.write_text("")
    button = tmp_path / "src" / "components" / "Button.tsx"
    button.write_text("")

    await _seed_indexed_modules(
        pipeline, [pipeline._canonical_file_str(p) for p in (src, graph_util, barrel, button)]
    )
    pipeline._pending_import_records = [
        (str(src), "import { graph } from './utils/graph'", "typescript"),
        (str(src), "import { Button } from './components'", "typescript"),
        (str(barrel), "export * from './Button'", "typescript"),
        (str(src), "import Button from '@/components/Button'", "typescript"),
        (str(src), "import React from 'react'", "typescript"),
    ]

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    pipeline.graph_store.upsert_edges = _capture  # type: ignore[assignment]

    emitted = await pipeline._resolve_imports(tmp_path)
    assert emitted == 4
    from victor.core.graph_rag.indexing import _module_node_id

    src_id = _module_node_id(pipeline._canonical_file_str(src))
    barrel_id = _module_node_id(pipeline._canonical_file_str(barrel))
    assert {(e.src, e.dst) for e in captured} == {
        (src_id, _module_node_id(pipeline._canonical_file_str(graph_util))),
        (src_id, barrel_id),
        (barrel_id, _module_node_id(pipeline._canonical_file_str(button))),
        (src_id, _module_node_id(pipeline._canonical_file_str(button))),
    }


@pytest.mark.asyncio
async def test_resolve_imports_drops_dangling_targets(monkeypatch, tmp_path: Path) -> None:
    """A target file that exists on disk but wasn't indexed (excluded by
    pattern, parse-failed, etc.) must NOT produce an IMPORTS edge — the
    edge would point at a non-existent module node.
    """
    pipeline = _make_pipeline(tmp_path)
    src = tmp_path / "a.py"
    target = tmp_path / "b.py"
    src.write_text("import b\n")
    target.write_text("def foo(): pass\n")
    # Only `src` made it into the indexed-files set; `target` is on disk
    # but was filtered out of indexing.
    await _seed_indexed_modules(pipeline, [pipeline._canonical_file_str(src)])
    pipeline._pending_import_records = [(str(src), "import b", "python")]

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    pipeline.graph_store.upsert_edges = _capture  # type: ignore[assignment]

    emitted = await pipeline._resolve_imports(tmp_path)
    assert emitted == 0
    assert captured == []


@pytest.mark.asyncio
async def test_inject_module_node_links_top_level_symbols_via_parent_id(
    tmp_path: Path,
) -> None:
    """Without explicit module nodes, top-level symbols would have no
    parent_id and the existing CONTAINS pass would emit no edges for them.
    """
    pipeline = _make_pipeline(tmp_path)
    GraphNode, _ = _get_graph_types_for_test()
    top1 = GraphNode(node_id="t1", type="function", name="foo", file="x.py", line=1)
    top2 = GraphNode(node_id="t2", type="class", name="Bar", file="x.py", line=5)
    nested = GraphNode(node_id="n1", type="method", name="m", file="x.py", line=6, parent_id="t2")

    augmented = pipeline._inject_module_node([top1, top2, nested], Path("x.py"), "python")

    assert augmented[0].type == "module"
    module_id = augmented[0].node_id
    # Top-level nodes get parented to the module so the existing CONTAINS
    # emission in _build_symbol_edges produces module→symbol edges.
    assert top1.parent_id == module_id
    assert top2.parent_id == module_id
    # Nested-symbol parent_id is preserved — we only inject for top-level.
    assert nested.parent_id == "t2"


def _get_graph_types_for_test():
    from victor.storage.graph.protocol import GraphEdge, GraphNode

    return GraphNode, GraphEdge


@pytest.mark.asyncio
async def test_resolve_cross_file_relationships_unresolved_target_skipped(
    monkeypatch, tmp_path: Path
) -> None:
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_relationship_records = [("child_id", "Vendored", "INHERITS")]

    # Project class index has no "Vendored" — common for third-party bases.
    await _seed_symbols(pipeline, [])

    captured: list[GraphEdge] = []
    monkeypatch.setattr(graph_store, "upsert_edges", lambda edges: captured.extend(edges))

    emitted = await pipeline._resolve_cross_file_relationships(tmp_path)
    assert emitted == 0
    assert captured == []


def test_provider_symbols_to_graph_nodes_resolves_parent_id_from_hints() -> None:
    """Provider-supplied parent_symbol/parent_line must materialize as
    parent_id on the child GraphNode so _build_symbol_edges emits CONTAINS
    edges. Without this, the new TSA path silently drops the parent/child
    relationship the legacy recursive extractor used to thread through.
    """
    pipeline = _make_pipeline(Path("/tmp"))
    symbols = [
        {
            "name": "Foo",
            "symbol_type": "class",
            "file_path": "/x.py",
            "line_start": 1,
            "line_end": 5,
            "ast_kind": "class_definition",
        },
        {
            "name": "bar",
            "symbol_type": "function",
            "file_path": "/x.py",
            "line_start": 2,
            "line_end": 3,
            "ast_kind": "function_definition",
            "parent_symbol": "Foo",
            "parent_line": 1,
            "parent_kind": "class_definition",
            "parent_is_class": True,
        },
    ]
    nodes = pipeline._provider_symbols_to_graph_nodes(symbols, Path("/x.py"), "python")
    by_name = {n.name: n for n in nodes}
    foo = by_name["Foo"]
    bar = by_name["bar"]
    assert foo.parent_id is None
    assert bar.parent_id == foo.node_id
    # parent_is_class=True promotes the function-typed child to method.
    assert bar.type == "method"


def test_provider_symbols_to_graph_nodes_does_not_promote_when_parent_not_class() -> None:
    """A nested function inside another function must stay typed as
    ``function`` — promotion only fires for class-like parents.
    """
    pipeline = _make_pipeline(Path("/tmp"))
    symbols = [
        {
            "name": "outer",
            "symbol_type": "function",
            "file_path": "/x.py",
            "line_start": 1,
            "line_end": 4,
            "ast_kind": "function_definition",
        },
        {
            "name": "inner",
            "symbol_type": "function",
            "file_path": "/x.py",
            "line_start": 2,
            "line_end": 3,
            "ast_kind": "function_definition",
            "parent_symbol": "outer",
            "parent_line": 1,
            "parent_kind": "function_definition",
            "parent_is_class": False,
        },
    ]
    nodes = pipeline._provider_symbols_to_graph_nodes(symbols, Path("/x.py"), "python")
    by_name = {n.name: n for n in nodes}
    assert by_name["inner"].parent_id == by_name["outer"].node_id
    assert by_name["inner"].type == "function"


def test_provider_symbols_to_graph_nodes_skips_unresolved_parent_hint() -> None:
    """If parent_symbol points at a symbol not in the same batch (cross-file
    inheritance, partial extraction), parent_id stays None — silently
    skipping is correct because CONTAINS edges only make sense within the
    same file.
    """
    pipeline = _make_pipeline(Path("/tmp"))
    symbols = [
        {
            "name": "Orphan",
            "symbol_type": "function",
            "file_path": "/x.py",
            "line_start": 5,
            "line_end": 6,
            "ast_kind": "function_definition",
            "parent_symbol": "DoesNotExist",
            "parent_line": 1,
            "parent_is_class": True,
        }
    ]
    nodes = pipeline._provider_symbols_to_graph_nodes(symbols, Path("/x.py"), "python")
    assert len(nodes) == 1
    assert nodes[0].parent_id is None
    # No promotion either, since parent_id wasn't resolved.
    assert nodes[0].type == "function"


# ────────────────────────────────────────────────────────────────────────
# TSA-5: cooperative yield between mini-batches
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consume_yields_between_flushes(monkeypatch, tmp_path: Path) -> None:
    """``_consume`` should ``await asyncio.sleep(0)`` after each flush so
    long-running indexing runs do not starve other coroutines.
    """
    import asyncio

    from victor.core.graph_rag.indexing import _IndexingStreamPipeline

    pipeline = _make_pipeline(tmp_path)
    streaming = _IndexingStreamPipeline(pipeline, write_batch_size=1, queue_maxsize=8)

    # Replace _flush with a no-op so we can count cooperative yields
    # independent of the bulk-write machinery.
    flush_calls: list[int] = []

    async def _fake_flush(batch, done_offset, total_files, cb):
        flush_calls.append(len(batch))
        return GraphIndexStats(files_processed=len(batch))

    monkeypatch.setattr(streaming, "_flush", _fake_flush)

    # Count cooperative sleeps from inside the consume loop.
    sleep_count = {"n": 0}
    original_sleep = asyncio.sleep

    async def _counting_sleep(seconds):
        if seconds == 0:
            sleep_count["n"] += 1
        await original_sleep(seconds)

    monkeypatch.setattr(asyncio, "sleep", _counting_sleep)

    queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    # Enqueue three real ParseResults plus the sentinel.
    for i in range(3):
        await queue.put(
            ParseResult(
                file_path=tmp_path / f"x{i}.py",
                language="python",
                symbol_nodes=[],
            )
        )
    await queue.put(streaming._STREAM_DONE)

    stats = await streaming._consume(queue, total_files=3, done_offset=0, progress_callback=None)

    # write_batch_size=1 -> flush runs per file. The 3rd flush happens on
    # the STREAM_DONE branch and intentionally does NOT yield (we're done),
    # so we expect exactly 2 cooperative yields from the in-loop branch.
    assert flush_calls == [1, 1, 1]
    assert sleep_count["n"] >= 2
    assert stats.files_processed == 3
