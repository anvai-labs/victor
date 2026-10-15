"""Tests for SqliteLanceDBStore initialization and helpers."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.storage.unified.sqlite_lancedb import SqliteLanceDBStore
from victor.storage.unified.protocol import (
    UnifiedId,
    UnifiedSymbol,
    SearchParams,
    SearchMode,
)


@pytest.fixture
def store(tmp_path):
    return SqliteLanceDBStore(
        repo_root=tmp_path,
        persist_directory=tmp_path / ".victor",
    )


@pytest.mark.parametrize("path", ["target' OR '1'='1", "missing' OR '1'='1"])
async def test_delete_file_uses_exact_vector_predicate(store, tmp_path, path):
    lancedb = pytest.importorskip("lancedb")
    store._vector_table = lancedb.connect(str(tmp_path / "vectors")).create_table(
        "symbols",
        data=[
            {"file_path": "target' OR '1'='1", "vector": [1.0, 0.0]},
            {"file_path": "other.py", "vector": [0.0, 1.0]},
        ],
    )
    store._initialized = True
    store._graph_store = AsyncMock()
    await store.delete_file(path)
    store._graph_store.delete_by_file.assert_awaited_once_with(path)
    remaining = store._vector_table.to_arrow()["file_path"].to_pylist()
    assert sorted(remaining) == (
        ["other.py"] if path.startswith("target") else ["other.py", "target' OR '1'='1"]
    )


@pytest.mark.parametrize("path", [None, 12, "bad\0path"])
async def test_delete_file_rejects_invalid_identity_before_graph_mutation(store, path):
    store._initialized = True
    store._graph_store = AsyncMock()
    store._vector_table = MagicMock()
    with pytest.raises(ValueError):
        await store.delete_file(path)
    store._graph_store.delete_by_file.assert_not_awaited()
    store._vector_table.delete.assert_not_called()


class TestInit:
    def test_defaults(self, store, tmp_path):
        assert store.repo_root == tmp_path.resolve()
        assert store._initialized is False
        assert store._graph_store is None
        assert store._vector_store is None

    def test_custom_persist_directory(self, tmp_path):
        custom = tmp_path / "custom_dir"
        s = SqliteLanceDBStore(repo_root=tmp_path, persist_directory=custom)
        assert s.persist_directory == custom

    @pytest.mark.asyncio
    async def test_initialize_uses_custom_project_db_path(self, tmp_path):
        custom = tmp_path / "custom_dir"
        store = SqliteLanceDBStore(repo_root=tmp_path, persist_directory=custom)

        fake_graph_store = MagicMock()
        fake_graph_store.initialize = AsyncMock()

        class _FakeEmbeddingModel:
            async def initialize(self):
                return None

        class _FakeFactory:
            def create_model(self, *_args, **_kwargs):
                return _FakeEmbeddingModel()

        class _FakeRegistry:
            def get(self, *_args, **_kwargs):
                return _FakeFactory()

        with (
            patch(
                "victor.storage.graph.sqlite_store.SqliteGraphStore",
                return_value=fake_graph_store,
            ) as mock_graph_store,
            patch(
                "victor.core.capability_registry.CapabilityRegistry.get_instance",
                return_value=_FakeRegistry(),
            ),
            patch.object(store, "_init_vector_store", AsyncMock()),
        ):
            await store.initialize()

        graph_path_arg = mock_graph_store.call_args.args[0]
        assert Path(graph_path_arg) == custom / "project.db"
        fake_graph_store.initialize.assert_awaited_once()


class TestUnifiedIdHelpers:
    def test_make_symbol_id(self, store):
        sid = store.make_symbol_id("src/foo.py", "MyClass")
        assert "src/foo.py" in sid
        assert "MyClass" in sid

    def test_make_symbol_id_preserves_qualified_symbol_names(self, store):
        sid = store.make_symbol_id("src/foo.py", "MyClass.run")
        parsed = store.parse_id(sid)
        assert parsed.path == "src/foo.py"
        assert parsed.name == "MyClass.run"

    def test_make_file_id(self, store):
        fid = store.make_file_id("src/foo.py")
        assert "src/foo.py" in fid

    def test_parse_id_roundtrip(self, store):
        sid = store.make_symbol_id("src/foo.py", "bar")
        parsed = store.parse_id(sid)
        assert isinstance(parsed, UnifiedId)


class TestClose:
    async def test_close_resets_state(self, store):
        store._initialized = True
        store._graph_store = MagicMock()
        store._graph_store.close = AsyncMock()
        store._embedding_model = MagicMock()
        store._embedding_model.close = AsyncMock()
        store._vector_store = MagicMock()
        store._vector_table = MagicMock()

        await store.close()

        assert store._initialized is False
        assert store._graph_store is None
        assert store._embedding_model is None
        assert store._vector_store is None
        assert store._vector_table is None


class TestCombineScores:
    def test_semantic_only(self, store):
        score = store._combine_scores(
            semantic=0.8,
            keyword=None,
            graph=None,
            semantic_weight=0.6,
            graph_weight=0.2,
        )
        assert 0.7 < score < 0.9

    def test_all_scores(self, store):
        score = store._combine_scores(
            semantic=1.0,
            keyword=1.0,
            graph=1.0,
            semantic_weight=0.5,
            graph_weight=0.3,
        )
        assert 0.9 < score <= 1.0

    def test_no_scores(self, store):
        score = store._combine_scores(
            semantic=None,
            keyword=None,
            graph=None,
            semantic_weight=0.5,
            graph_weight=0.3,
        )
        assert score == 0.0


def _symbol(uid: str = "sym:src/foo.py:Foo", name: str = "Foo") -> UnifiedSymbol:
    return UnifiedSymbol(unified_id=uid, name=name, type="class", file_path="src/foo.py")


# ---------------------------------------------------------------------------
# Item 6 - Hot-path coverage: index + search
# ---------------------------------------------------------------------------


class TestIndexSymbol:
    async def test_index_symbol_upserts_to_graph(self, store):
        store._initialized = True
        store._graph_store = MagicMock()
        store._graph_store.upsert_nodes = AsyncMock()
        store._vector_store = None

        await store.index_symbol(_symbol(), "class Foo")

        store._graph_store.upsert_nodes.assert_awaited_once()
        nodes = store._graph_store.upsert_nodes.call_args.args[0]
        assert nodes[0].node_id == "sym:src/foo.py:Foo"

    async def test_index_symbol_skips_vector_when_no_store(self, store):
        store._initialized = True
        store._graph_store = MagicMock()
        store._graph_store.upsert_nodes = AsyncMock()
        store._vector_store = None
        store._embedding_model = AsyncMock()

        await store.index_symbol(_symbol(), "class Foo")

        store._embedding_model.embed_text.assert_not_called()


class TestIndexSymbolsBatch:
    async def test_batch_indexes_all_symbols(self, store):
        store._initialized = True
        store._graph_store = MagicMock()
        store._graph_store.upsert_nodes = AsyncMock()
        store._vector_store = None

        symbols = [(_symbol(f"sym:a.py:C{i}", f"C{i}"), f"text{i}") for i in range(3)]
        await store.index_symbols_batch(symbols)

        store._graph_store.upsert_nodes.assert_awaited_once()
        nodes = store._graph_store.upsert_nodes.call_args.args[0]
        assert len(nodes) == 3

    async def test_batch_returns_correct_count(self, store):
        store._initialized = True
        store._graph_store = MagicMock()
        store._graph_store.upsert_nodes = AsyncMock()
        store._vector_store = None

        symbols = [(_symbol(f"sym:a.py:S{i}", f"S{i}"), "text") for i in range(5)]
        count = await store.index_symbols_batch(symbols)

        assert count == 5

    async def test_empty_batch_returns_zero(self, store):
        store._initialized = True
        count = await store.index_symbols_batch([])
        assert count == 0


class TestHybridSearch:
    async def test_hybrid_merges_semantic_and_keyword_scores(self, store):
        store._initialized = True
        semantic_result = MagicMock()
        semantic_result.symbol.unified_id = "s1"
        semantic_result.score = 0.9
        semantic_result.semantic_score = 0.9
        semantic_result.keyword_score = None
        semantic_result.graph_score = None

        keyword_result = MagicMock()
        keyword_result.symbol.unified_id = "s1"
        keyword_result.score = 1.0
        keyword_result.keyword_score = 1.0

        semantic_search = AsyncMock(return_value=[semantic_result])
        keyword_search = AsyncMock(return_value=[keyword_result])
        with (
            patch.object(store, "_semantic_search", semantic_search),
            patch.object(store, "_keyword_search", keyword_search),
        ):
            params = SearchParams(query="foo", limit=10, mode=SearchMode.HYBRID)
            results = await store.search(params)

        semantic_search.assert_awaited_once()
        keyword_search.assert_awaited_once()
        assert len(results) == 1

    async def test_semantic_only_mode_skips_fts(self, store):
        store._initialized = True
        semantic_search = AsyncMock(return_value=[])
        keyword_search = AsyncMock(return_value=[])
        with (
            patch.object(store, "_semantic_search", semantic_search),
            patch.object(store, "_keyword_search", keyword_search),
        ):
            params = SearchParams(query="foo", limit=5, mode=SearchMode.SEMANTIC)
            await store.search(params)

        semantic_search.assert_awaited_once()
        keyword_search.assert_not_awaited()

    async def test_keyword_only_mode_skips_vector(self, store):
        store._initialized = True
        semantic_search = AsyncMock(return_value=[])
        keyword_search = AsyncMock(return_value=[])
        with (
            patch.object(store, "_semantic_search", semantic_search),
            patch.object(store, "_keyword_search", keyword_search),
        ):
            params = SearchParams(query="foo", limit=5, mode=SearchMode.KEYWORD)
            await store.search(params)

        semantic_search.assert_not_awaited()
        keyword_search.assert_awaited_once()

    async def test_results_limited_to_params_limit(self, store):
        store._initialized = True
        results_mock = [
            MagicMock(symbol=MagicMock(unified_id=f"s{i}"), score=float(i)) for i in range(10)
        ]

        with (
            patch.object(store, "_semantic_search", AsyncMock(return_value=results_mock)),
            patch.object(store, "_keyword_search", AsyncMock(return_value=[])),
        ):
            params = SearchParams(query="foo", limit=3, mode=SearchMode.SEMANTIC)
            results = await store.search(params)

        assert len(results) <= 3


class TestGraphQueries:
    @pytest.mark.asyncio
    async def test_get_callers_uses_incoming_multi_hop_traversal(self, store):
        store._initialized = True
        store.get_symbol = AsyncMock(
            side_effect=[
                MagicMock(unified_id="b"),
                MagicMock(unified_id="c"),
            ]
        )
        store._graph_store = MagicMock()
        store._graph_store.get_neighbors = AsyncMock(
            return_value=[
                MagicMock(src="b", dst="target", type="CALLS"),
                MagicMock(src="c", dst="b", type="CALLS"),
            ]
        )

        callers = await store.get_callers("target", max_depth=2)

        store._graph_store.get_neighbors.assert_awaited_once_with(
            "target",
            edge_types=["CALLS"],
            direction="in",
            max_depth=2,
        )
        assert [caller.unified_id for caller in callers] == ["b", "c"]

    @pytest.mark.asyncio
    async def test_get_callees_uses_outgoing_multi_hop_traversal(self, store):
        store._initialized = True
        store.get_symbol = AsyncMock(
            side_effect=[
                MagicMock(unified_id="b"),
                MagicMock(unified_id="c"),
            ]
        )
        store._graph_store = MagicMock()
        store._graph_store.get_neighbors = AsyncMock(
            return_value=[
                MagicMock(src="target", dst="b", type="CALLS"),
                MagicMock(src="b", dst="c", type="CALLS"),
            ]
        )

        callees = await store.get_callees("target", max_depth=2)

        store._graph_store.get_neighbors.assert_awaited_once_with(
            "target",
            edge_types=["CALLS"],
            direction="out",
            max_depth=2,
        )
        assert [callee.unified_id for callee in callees] == ["b", "c"]
