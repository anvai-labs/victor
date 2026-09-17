"""Code-search contracts without embedding downloads or a native vector database.

Temporary files exercise literal retrieval; controllable async indexes exercise
integrity/recovery and semantic dispatch at the provider boundary.
"""

import asyncio
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from victor_coding.tools import code_search_tool as search


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(search, "_INDEX_CACHE", {})
    monkeypatch.setattr(search, "_INFLIGHT_INDEX_REBUILDS", {})


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, search.IntegrityProbeOutcome(rebuilt=True)),
        (False, search.IntegrityProbeOutcome()),
        (None, search.IntegrityProbeOutcome()),
        (1, search.IntegrityProbeOutcome()),
    ],
)
def test_legacy_probe_outcomes(value, expected):
    assert search._coerce_integrity_probe_outcome(value) == expected


def test_probe_outcome_is_immutable_and_preserves_stale():
    outcome = search.IntegrityProbeOutcome(stale=True)
    assert search._coerce_integrity_probe_outcome(outcome) is outcome
    with pytest.raises(FrozenInstanceError):
        outcome.stale = False


@pytest.mark.parametrize(
    "raw,expected",
    [
        (" tests ", search.SearchFilters(test_only=True)),
        ("Python", search.SearchFilters(language="python")),
        (" src/*.py ", search.SearchFilters(file_pattern="src/*.py")),
        (
            {"symbol": " parse ", "extensions": [" py ", "", "  "]},
            search.SearchFilters(symbol="parse", extensions=["py"]),
        ),
        ("", None),
        ("unsupported", None),
        (42, None),
        (None, None),
    ],
)
def test_filter_inputs(raw, expected):
    assert search._normalize_search_filters(raw) == expected


def test_filter_normalization_does_not_mutate_caller():
    filters = search.SearchFilters(symbol=" parse ", language=" ", extensions=[" py "])
    normalized = search._normalize_search_filters(filters)
    assert normalized == search.SearchFilters(symbol="parse", extensions=["py"])
    assert filters.symbol == " parse "
    assert filters.extensions == [" py "]
    assert search._normalize_search_filters(normalized) is normalized


@pytest.mark.parametrize(
    "path,pattern,expected",
    [
        ("src/main.py", "*.py", True),
        ("src/main.py", "main.py", True),
        ("src/main.py", "src/*.py", True),
        ("src/main.py", "other/main.py", False),
        ("src/main.py", "*.js", False),
        ("src/main.py", "./src/main.py", True),
        (r"src\main.py", r"src\*.py", True),
    ],
)
def test_file_patterns(tmp_path, path, pattern, expected):
    assert (
        search._matches_literal_file_pattern(path, pattern, search_root=str(tmp_path)) is expected
    )


@pytest.mark.parametrize(
    "result,language,expected",
    [
        ({"file_path": "src/app.PY"}, "python3", True),
        ({"path": "ui/app.tsx"}, "ts", True),
        ({"path": "src/app.go"}, "golang", True),
        ({"metadata": {"language": "js"}}, "javascript", True),
        ({"file_path": "app.py"}, "rust", False),
        ({"path": "app.custom"}, "custom", True),
    ],
)
def test_language_filter(result, language, expected):
    assert search._result_matches_language_filter(result, language) is expected


@pytest.mark.parametrize(
    "result,symbol,expected",
    [
        ({"qualified_name": "Parser.parse"}, "parse", True),
        ({"metadata": {"symbol_name": "Parser:parse"}}, "parse", True),
        ({"snippet": "parse(value)"}, "parse", True),
        ({"content": "parse_value(value)"}, "parse", False),
        ({"content": "a+b(value)"}, "a+b", True),
    ],
)
def test_symbol_filter(result, symbol, expected):
    assert search._result_matches_symbol_filter(result, symbol) is expected


@pytest.mark.parametrize("test_only,expected", [(True, [0, 2]), (False, [1])])
def test_test_filter_metadata_overrides_filename(test_only, expected):
    hits = [
        {"path": "tests/test_a.py"},
        {"path": "tests/test_b.py", "metadata": {"is_test_file": False}},
        {"path": "src/a.py", "metadata": {"is_test_file": True}},
    ]
    assert search._filter_search_results_by_test_only(hits, test_only) == [
        hits[i] for i in expected
    ]


def test_combined_postfilters_then_limit_preserves_input():
    hits = [
        {"path": "app.py", "content": "def parse(): pass"},
        {"path": "tests/test_app.js", "content": "parse()"},
        {"path": "tests/test_app.py", "content": "parse_other()"},
        {"path": "tests/test_app.py", "content": "parse()"},
        {"path": "tests/test_more.py", "content": "parse()"},
    ]
    original = {"results": hits, "count": 5, "success": True}
    filters = search.SearchFilters(
        language="python", symbol="parse", test_only=True, extensions=["py"]
    )
    result = search._apply_literal_result_filters(original, filters, max_results=1)
    assert result["results"] == [hits[3]]
    assert result["count"] == 1
    assert original["count"] == 5
    assert search._literal_search_fetch_limit(4, filters) == 8


def test_cache_invalidation_respects_root_boundaries(tmp_path, monkeypatch):
    root = tmp_path / "project"
    nested = root / "src"
    sibling = tmp_path / "project-other"
    cache = {str(root): {}, str(nested): {}, str(sibling): {}, "invalid": None}
    monkeypatch.setattr(search, "_INDEX_CACHE", cache)
    monkeypatch.setattr(search.time, "time", lambda: 123)
    assert search.mark_index_cache_stale_for_path(str(nested / "main.py")) == 2
    for path in (root, nested):
        assert cache[str(path)]["stale"] is True
        assert cache[str(path)]["stale_reason"] == "modified_file"
        assert cache[str(path)]["invalidated_at"] == 123
    assert cache[str(sibling)] == {}


def test_cache_invalidation_uses_injected_cache(tmp_path):
    injected = {str(tmp_path): {"stale": False}}
    context = {"cache_manager": SimpleNamespace(index_cache=injected)}
    assert search.mark_index_cache_stale_for_path(str(tmp_path / "app.py"), exec_ctx=context) == 1
    assert injected[str(tmp_path)]["stale"] is True
    assert search._INDEX_CACHE == {}


def test_clear_cache_preserves_reference():
    cache = search._INDEX_CACHE
    cache["root"] = {"stale": True}
    search.clear_index_cache()
    assert search._get_index_cache() is cache
    assert cache == {}


def test_failure_cache_suppresses_build_until_expiry(monkeypatch):
    cache = {}
    search._cache_index_build_failure(cache, "key", "provider missing", ttl=60)
    with pytest.raises(ImportError, match="reindex=True"):
        search._raise_if_cached_index_build_failure(cache, "key")
    monkeypatch.setattr(cache["key"], "is_expired", lambda: True)
    search._raise_if_cached_index_build_failure(cache, "key")


@pytest.mark.asyncio
async def test_integrity_skips_active_indexing(monkeypatch):
    index = SimpleNamespace(_is_indexing=True, semantic_search=AsyncMock())
    rebuild = AsyncMock()
    monkeypatch.setattr(search, "_background_index_rebuild", rebuild)
    assert await search._probe_index_integrity(index) == search.IntegrityProbeOutcome()
    index.semantic_search.assert_not_called()
    rebuild.assert_not_called()


@pytest.mark.asyncio
async def test_integrity_healthy_persisted_rows_skip_embedding(monkeypatch):
    index = SimpleNamespace(
        _vector_store=SimpleNamespace(_table=SimpleNamespace(count_rows=lambda: 3)),
        semantic_search=AsyncMock(),
    )
    assert await search._probe_index_integrity(index) == search.IntegrityProbeOutcome()
    index.semantic_search.assert_not_called()


@pytest.mark.asyncio
async def test_integrity_search_probe_success():
    index = SimpleNamespace(semantic_search=AsyncMock(return_value=[]))
    assert await search._probe_index_integrity(index) == search.IntegrityProbeOutcome()
    index.semantic_search.assert_awaited_once_with(query="test", max_results=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", ["database locked", "provider initializing", "has no attribute x"]
)
async def test_integrity_transient_errors_do_not_rebuild(monkeypatch, error):
    index = SimpleNamespace(semantic_search=AsyncMock(side_effect=RuntimeError(error)))
    rebuild = AsyncMock()
    monkeypatch.setattr(search, "_background_index_rebuild", rebuild)
    assert await search._probe_index_integrity(index) == search.IntegrityProbeOutcome()
    rebuild.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("rebuilt", [True, False])
async def test_integrity_corruption_rebuild_outcome(monkeypatch, rebuilt):
    index = SimpleNamespace(
        _is_indexed=True, semantic_search=AsyncMock(side_effect=ValueError("corrupt index"))
    )
    rebuild = AsyncMock(return_value=rebuilt)
    monkeypatch.setattr(search, "_background_index_rebuild", rebuild)
    assert await search._probe_index_integrity(index) == search.IntegrityProbeOutcome(
        rebuilt=rebuilt, stale=not rebuilt
    )
    assert index._is_indexed is False
    rebuild.assert_awaited_once_with(index)


@pytest.mark.asyncio
async def test_integrity_timeout_cancels_probe_before_rebuild(monkeypatch):
    cancelled = asyncio.Event()

    async def blocked(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def rebuild(index):
        assert cancelled.is_set()
        return False

    monkeypatch.setattr(search, "_background_index_rebuild", rebuild)
    outcome = await search._probe_index_integrity(
        SimpleNamespace(semantic_search=blocked), timeout=0
    )
    assert outcome.stale is True


@pytest.mark.asyncio
async def test_integrity_parent_cancellation_cleans_probe(monkeypatch):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def blocked(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    rebuild = AsyncMock()
    monkeypatch.setattr(search, "_background_index_rebuild", rebuild)
    task = asyncio.create_task(
        search._probe_index_integrity(SimpleNamespace(semantic_search=blocked))
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    rebuild.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,expected",
    [(None, True), (RuntimeError("broken"), False), (asyncio.TimeoutError(), False)],
)
async def test_background_rebuild_outcomes(monkeypatch, error, expected):
    index = SimpleNamespace(index_codebase=AsyncMock(side_effect=error))
    finalize = AsyncMock()
    monkeypatch.setattr(search, "_finalize_index_storage", finalize)
    assert await search._background_index_rebuild(index) is expected
    assert finalize.await_count == int(expected)


@pytest.mark.asyncio
async def test_rebuild_single_flight_and_cleanup():
    release = asyncio.Event()
    factory = Mock(side_effect=lambda: release.wait())
    first = search._spawn_index_rebuild_once("root", factory)
    assert search._spawn_index_rebuild_once("root", factory) is first
    assert factory.call_count == 1
    release.set()
    await first
    await asyncio.sleep(0)
    assert "root" not in search._INFLIGHT_INDEX_REBUILDS
    second = search._spawn_index_rebuild_once("root", factory)
    assert second is not first
    await second


@pytest.mark.asyncio
async def test_incremental_refresh_clears_staleness(monkeypatch, tmp_path):
    index = SimpleNamespace(incremental_reindex=AsyncMock())
    entry = {"index": index, "stale": True}
    monkeypatch.setattr(search, "_latest_mtime", lambda root: 42)
    monkeypatch.setattr(search, "_finalize_index_storage", AsyncMock())
    watcher = AsyncMock()
    monkeypatch.setattr(search, "_ensure_file_watcher_subscription", watcher)
    assert await search._refresh_cached_index_incrementally(entry, tmp_path, ensure_watcher=True)
    assert entry["stale"] is False
    assert entry["latest_mtime"] == 42
    index.incremental_reindex.assert_awaited_once()
    watcher.assert_awaited_once_with(entry, tmp_path, None)


@pytest.mark.asyncio
async def test_incremental_refresh_unsupported_marks_stale(tmp_path):
    entry = {"index": object()}
    assert not await search._refresh_cached_index_incrementally(entry, tmp_path)
    assert entry["stale"] is True


@pytest.mark.asyncio
async def test_empty_query_rejected_without_index(monkeypatch):
    build = AsyncMock()
    monkeypatch.setattr(search, "_get_or_build_index", build)
    assert (await search.code_search("  "))["success"] is False
    build.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,query",
    [("text", "unique_marker"), ("literal", "unique_marker"), ("filename", "sample.py")],
)
async def test_literal_public_api_real_files(tmp_path, mode, query):
    (tmp_path / "sample.py").write_text("def unique_marker():\n    pass\n")
    result = await search.code_search(query, path=str(tmp_path), mode=mode)
    assert result["success"] is True
    assert result["mode"] == mode
    assert result["count"] == 1
    assert "sample.py" in str(result["results"])


@pytest.mark.asyncio
async def test_file_pattern_supplies_empty_query(tmp_path):
    (tmp_path / "sample.py").write_text("pass\n")
    result = await search.code_search("", path=str(tmp_path), filters={"file_pattern": "*.py"})
    assert result["mode"] == "filename"
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_missing_root_reports_actionable_error(tmp_path):
    result = await search.code_search("find handlers", path=str(tmp_path / "absent" / "nested"))
    assert result["success"] is False
    assert "valid directory" in result["error"]


@pytest.mark.asyncio
async def test_semantic_requires_settings(tmp_path):
    result = await search.code_search("find handlers", path=str(tmp_path))
    assert result["success"] is False
    assert "Settings" in result["error"]


@pytest.mark.asyncio
async def test_disabled_embeddings_fallback_retains_filters(tmp_path, monkeypatch):
    (tmp_path / "sample.py").write_text("unique_marker()\n")
    (tmp_path / "other.js").write_text("unique_marker()\n")
    build = AsyncMock()
    monkeypatch.setattr(search, "_get_or_build_index", build)
    result = await search.code_search(
        "unique_marker",
        path=str(tmp_path),
        filters={"language": "python"},
        _exec_ctx={"settings": SimpleNamespace(), "disable_embeddings": True},
    )
    assert result["success"] is True
    assert result["count"] == 1
    assert "semantic_disabled" in str(result)
    build.assert_not_called()


@pytest.fixture
def semantic_backend(tmp_path, monkeypatch):
    index = SimpleNamespace(
        semantic_search=AsyncMock(
            return_value=[
                {
                    "file_path": str(tmp_path / "app.py"),
                    "content": "def parse(): pass",
                    "score": 0.8,
                    "line_number": 1,
                    "metadata": {"language": "python", "is_test_file": False},
                }
            ]
        )
    )
    build = AsyncMock(return_value=(index, True))
    monkeypatch.setattr(search, "_get_or_build_index", build)
    monkeypatch.setattr(
        search, "_collect_code_search_backend_metadata", lambda *args: {"backend": "fake"}
    )
    monkeypatch.setattr(search, "enrich_code_search_results", lambda hits, **kwargs: (hits, {}))
    monkeypatch.setattr(search, "rerank_code_search_results", lambda hits, **kwargs: (hits, {}))
    return index, build, {"settings": SimpleNamespace()}


@pytest.mark.asyncio
async def test_semantic_public_api_filters_and_projects(tmp_path, semantic_backend):
    index, build, context = semantic_backend
    result = await search.code_search(
        "find parser",
        path=str(tmp_path),
        k=2,
        reindex=True,
        filters={
            "file_pattern": "*.py",
            "language": "python",
            "symbol": "parse",
            "test_only": False,
            "extensions": ["py"],
        },
        _exec_ctx=context,
    )
    assert result["success"] is True
    assert result["count"] == 1
    assert result["mode"] == "semantic"
    assert result["metadata"]["rebuilt"] is True
    assert result["results"][0]["combined_score"] > 0
    assert "app.py" in result["formatted_results"]
    build.assert_awaited_once_with(
        tmp_path, context["settings"], force_reindex=True, exec_ctx=context
    )
    assert index.semantic_search.await_args.kwargs["filter_metadata"] == {"symbol_name": "parse"}
    assert index.semantic_search.await_args.kwargs["max_results"] == 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,reason",
    [
        (RuntimeError("bad data"), "semantic_search_error"),
        (asyncio.TimeoutError(), "semantic_search_timeout"),
    ],
)
async def test_semantic_search_failure_falls_back(tmp_path, semantic_backend, error, reason):
    index, _, context = semantic_backend
    index.semantic_search.side_effect = error
    (tmp_path / "app.py").write_text("parse()\n")
    result = await search.code_search("parse", path=str(tmp_path), _exec_ctx=context)
    assert result["success"] is True
    assert result["fallback"] == reason
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_stale_index_never_serves_semantic_hits(tmp_path, semantic_backend):
    index, _, context = semantic_backend
    search._INDEX_CACHE[str(tmp_path)] = {"stale": True}
    result = await search.code_search("parse", path=str(tmp_path), _exec_ctx=context)
    assert result["fallback"] == "semantic_index_stale"
    index.semantic_search.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,method",
    [
        ("bugs", "find_similar_bugs"),
        ("localize", "localize_issue"),
        ("impact", "analyze_change_impact"),
    ],
)
async def test_optional_provider_capability_dispatch(tmp_path, semantic_backend, mode, method):
    index, _, context = semantic_backend
    capability = AsyncMock(
        return_value=[{"file_path": "app.py", "content": "parse()", "score": 0.8}]
    )
    setattr(index, method, capability)
    result = await search.code_search(
        "find parser", path=str(tmp_path), mode=mode, _exec_ctx=context
    )
    assert result["success"] is True
    assert result["mode"] == mode
    assert result["metadata"]["provider_capability"] == method
    capability.assert_awaited_once()
    index.semantic_search.assert_not_called()


@pytest.fixture
def index_factory(tmp_path, monkeypatch):
    import victor_contracts.capability_runtime as capabilities
    import victor_contracts.indexing_runtime as indexing

    index = SimpleNamespace(
        _is_indexed=False, index_codebase=AsyncMock(), incremental_reindex=AsyncMock()
    )
    factory = SimpleNamespace(create=Mock(return_value=index))
    lock = SimpleNamespace(__aenter__=AsyncMock(), __aexit__=AsyncMock())
    registry = SimpleNamespace(acquire_lock=AsyncMock(return_value=lock), mark_lock_used=Mock())
    monkeypatch.setattr(capabilities, "get_capability_provider", lambda protocol: factory)
    monkeypatch.setattr(capabilities, "is_capability_enhanced", lambda protocol: True)
    monkeypatch.setattr(indexing.IndexLockRegistry, "get_instance", lambda: registry)
    monkeypatch.setattr(
        search,
        "_build_codebase_embedding_config",
        lambda *args: {"persist_directory": str(tmp_path / "index")},
    )
    monkeypatch.setattr(search, "build_codebase_index_manifest", lambda config: {"version": 1})
    monkeypatch.setattr(search, "_get_index_build_failure_cache", lambda ctx: {})
    monkeypatch.setattr(search, "_latest_mtime", lambda root: 10)
    monkeypatch.setattr(search, "has_persisted_codebase_index_data", lambda path: False)
    monkeypatch.setattr(search, "has_compatible_codebase_index_manifest", lambda *args: True)
    monkeypatch.setattr(search, "write_codebase_index_manifest", Mock())
    monkeypatch.setattr(search, "_finalize_index_storage", AsyncMock())
    monkeypatch.setattr(search, "_ensure_file_watcher_subscription", AsyncMock())
    return index, factory, lock


@pytest.mark.asyncio
async def test_index_build_then_memory_cache_reuse(tmp_path, index_factory):
    index, factory, lock = index_factory
    assert await search._get_or_build_index(tmp_path, SimpleNamespace()) == (index, True)
    assert await search._get_or_build_index(tmp_path, SimpleNamespace()) == (index, False)
    index.index_codebase.assert_awaited_once()
    factory.create.assert_called_once()
    lock.__aexit__.assert_awaited_once()
    assert search._INDEX_CACHE[str(tmp_path)]["stale"] is False


@pytest.mark.asyncio
async def test_changed_files_incrementally_refresh_cached_index(
    tmp_path, index_factory, monkeypatch
):
    index, factory, lock = index_factory
    await search._get_or_build_index(tmp_path, SimpleNamespace())
    monkeypatch.setattr(search, "_latest_mtime", lambda root: 20)
    assert await search._get_or_build_index(tmp_path, SimpleNamespace()) == (index, False)
    index.incremental_reindex.assert_awaited_once()
    assert search._INDEX_CACHE[str(tmp_path)]["latest_mtime"] == 20
    assert factory.create.call_count == 1
    assert lock.__aexit__.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidate", ["stale", "manifest", "force"])
async def test_invalid_cache_rebuilds(tmp_path, index_factory, invalidate):
    index, factory, _ = index_factory
    await search._get_or_build_index(tmp_path, SimpleNamespace())
    entry = search._INDEX_CACHE[str(tmp_path)]
    if invalidate == "stale":
        entry["stale"] = True
    elif invalidate == "manifest":
        entry["index_manifest"] = {"version": 0}
    assert await search._get_or_build_index(
        tmp_path, SimpleNamespace(), force_reindex=invalidate == "force"
    ) == (index, True)
    assert factory.create.call_count == 2
    assert index.index_codebase.await_count == 2


@pytest.mark.asyncio
async def test_persisted_probe_staleness_is_cached(tmp_path, index_factory, monkeypatch):
    index, _, lock = index_factory
    monkeypatch.setattr(search, "has_persisted_codebase_index_data", lambda path: True)
    monkeypatch.setattr(
        search,
        "_probe_index_integrity",
        AsyncMock(return_value=search.IntegrityProbeOutcome(stale=True)),
    )
    assert await search._get_or_build_index(tmp_path, SimpleNamespace()) == (index, False)
    assert search._INDEX_CACHE[str(tmp_path)]["stale"] is True
    index.index_codebase.assert_not_called()
    lock.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_build_releases_lock_without_caching(tmp_path, index_factory):
    index, _, lock = index_factory
    index.index_codebase.side_effect = RuntimeError("disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        await search._get_or_build_index(tmp_path, SimpleNamespace())
    lock.__aexit__.assert_awaited_once()
    assert str(tmp_path) not in search._INDEX_CACHE
