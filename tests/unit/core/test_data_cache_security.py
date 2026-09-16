"""Data cache persistence must never construct classes supplied by disk data."""

import base64
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from victor.core.data_cache import read_cache_data, write_cache_data


class HostileValue:
    def __reduce__(self):
        return (os.system, ("touch cache-payload-executed",))


@pytest.mark.parametrize("dtype", ["float32", ">f8", "int64", "complex128", "bool"])
def test_numeric_arrays_preserve_dtype_shape_and_mutability(tmp_path, dtype):
    path = tmp_path / "cache.json"
    value = np.arange(12).reshape(3, 4).astype(dtype)[:, ::2]
    write_cache_data(path, {"array": value, "tuple": (b"bytes", 1)})
    loaded = read_cache_data(path)
    assert loaded["array"].dtype == value.dtype
    np.testing.assert_array_equal(loaded["array"], value)
    loaded["array"][0, 0] = 1
    assert loaded["tuple"] == (b"bytes", 1)


def test_legacy_pickle_and_object_arrays_cannot_execute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "cache.json"
    payload = pickle.dumps(HostileValue())
    path.write_bytes(payload)
    with pytest.raises((ValueError, UnicodeDecodeError)):
        read_cache_data(path)
    with pytest.raises(TypeError, match="Unsupported"):
        write_cache_data(path, np.array([HostileValue()], dtype=object))
    path.write_text(
        json.dumps(
            {
                "format": "victor-data-cache-v1",
                "data": [
                    "ndarray",
                    {"dtype": "O", "shape": [1], "bytes": base64.b64encode(payload).decode()},
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="numeric"):
        read_cache_data(path)
    assert not Path("cache-payload-executed").exists()


@pytest.mark.parametrize("shape", [[10**18], [-1], [True], "not a shape"])
def test_forged_array_shape_is_rejected(tmp_path, shape):
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps(
            {
                "format": "victor-data-cache-v1",
                "data": ["ndarray", {"dtype": "float64", "shape": shape, "bytes": ""}],
            }
        )
    )
    with pytest.raises((ValueError, OverflowError)):
        read_cache_data(path)


def test_unsupported_write_preserves_existing_file(tmp_path):
    path = tmp_path / "cache.json"
    write_cache_data(path, {"keep": True})
    before = path.read_bytes()
    entries_before = set(tmp_path.iterdir())
    with pytest.raises(TypeError):
        write_cache_data(path, HostileValue())
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == entries_before


@pytest.mark.skipif(os.name != "posix", reason="POSIX link and mode checks")
@pytest.mark.parametrize("link_kind", ["hardlink", "symlink"])
def test_atomic_write_does_not_follow_preseeded_links(tmp_path, link_kind):
    external = tmp_path / "external"
    external.write_text("preserve")
    external.chmod(0o644)
    path = tmp_path / "cache.json"
    if link_kind == "hardlink":
        os.link(external, path)
    else:
        path.symlink_to(external)
    write_cache_data(path, {"safe": True})
    assert external.read_text() == "preserve"
    assert external.stat().st_mode & 0o777 == 0o644
    assert path.stat().st_mode & 0o777 == 0o600
    assert read_cache_data(path) == {"safe": True}


def test_readers_never_observe_partial_concurrent_writes(tmp_path):
    path = tmp_path / "cache.json"
    write_cache_data(path, {"value": 0, "padding": "x" * 10000})

    def work(i):
        write_cache_data(path, {"value": i, "padding": "x" * 10000})
        data = read_cache_data(path)
        assert isinstance(data["value"], int)
        assert data["padding"] == "x" * 10000

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(work, range(60)))


def test_bad_analytics_section_does_not_partially_replace_state(tmp_path):
    from victor.agent.usage_analytics import AnalyticsConfig, UsageAnalytics

    analytics = UsageAnalytics(AnalyticsConfig(cache_dir=tmp_path))
    analytics.record_tool_execution("keep", True, 10.0)
    before = analytics._tool_records
    write_cache_data(
        analytics._cache_file,
        {
            "tool_records": {
                "new": [{"timestamp": 1.0, "success": True, "execution_time_ms": 2.0}]
            },
            "provider_records": {"invalid": [{}]},
        },
    )
    analytics._load_from_cache()
    assert analytics._tool_records is before
    assert set(analytics._tool_records) == {"keep"}


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {
            "keep": {
                "usage_count": 10**400,
                "success_count": 1,
                "last_used": 0,
                "recent_contexts": [],
            }
        },
    ],
)
def test_bad_usage_cache_cannot_poison_later_recording(tmp_path, payload):
    from victor.tools.semantic_selector import SemanticToolSelector

    selector = SemanticToolSelector(cache_dir=tmp_path)
    selector._record_tool_usage("keep", "query")
    before = selector._tool_usage_cache
    write_cache_data(selector._usage_cache_file, payload)
    selector._load_usage_cache()
    assert selector._tool_usage_cache is before
    selector._record_tool_usage("keep", "another query")
    assert selector._tool_usage_cache["keep"]["usage_count"] == 2


def test_bad_collection_records_rebuild_without_mutating_state(tmp_path):
    from types import SimpleNamespace
    from victor.storage.embeddings.collections import StaticEmbeddingCollection

    collection = StaticEmbeddingCollection(
        "test",
        cache_dir=tmp_path,
        embedding_service=SimpleNamespace(model_name="test", dimension=2),
    )
    write_cache_data(
        collection.cache_file,
        {
            "cache_version": collection.CACHE_VERSION,
            "items_hash": "hash",
            "model_name": "test",
            "embeddings": np.zeros((1, 2)),
            "items": {"id": {}},
            "item_ids": ["id"],
        },
    )
    assert collection._load_from_cache("hash") is False
    assert collection._items == {}
    assert collection._embeddings is None


@pytest.mark.parametrize("embeddings", [{"wrong": np.zeros(2)}, np.zeros((1, 0))])
def test_bad_corpus_array_does_not_publish_failed_snapshot(tmp_path, embeddings):
    from types import SimpleNamespace
    from victor.agent.prompt_corpus_registry import PromptCorpusRegistry

    entry = SimpleNamespace(embedding=None)
    registry = PromptCorpusRegistry(corpus=[entry], cache_dir=tmp_path)
    write_cache_data(
        registry._cache_file,
        {
            "corpus_hash": "hash",
            "embeddings": embeddings,
        },
    )
    assert registry._load_from_cache("hash") is False
    assert registry._corpus_embeddings is None
    assert registry._corpus_hash is None
    assert entry.embedding is None


@pytest.mark.parametrize(
    "data",
    [
        {"tool_records": {"bad": [{"timestamp": "nan", "success": True, "execution_time_ms": 1}]}},
        {"condensation_stats": {"bad": {}}},
        {
            "condensation_stats": {
                "bad": {"count": 1, "original_chars": 10**400, "condensed_chars": 0}
            }
        },
        {
            "tool_records": {
                "bad": [
                    {
                        "timestamp": 1,
                        "success": True,
                        "execution_time_ms": 1,
                        "context_tokens": 10**400,
                    }
                ]
            }
        },
    ],
)
def test_invalid_metric_snapshot_is_not_installed(tmp_path, data):
    from victor.agent.usage_analytics import AnalyticsConfig, UsageAnalytics

    analytics = UsageAnalytics(AnalyticsConfig(cache_dir=tmp_path))
    before = analytics._tool_records
    write_cache_data(analytics._cache_file, data)
    analytics._load_from_cache()
    assert analytics._tool_records is before
    analytics.record_output_condensation("bad", 100, 20)
    assert analytics._condensation_stats["bad"]["count"] == 1


@pytest.mark.parametrize("item_ids", [{"id": 0}, ["id", "id"], ["missing"]])
def test_collection_requires_complete_id_list(tmp_path, item_ids):
    from types import SimpleNamespace
    from victor.storage.embeddings.collections import StaticEmbeddingCollection

    collection = StaticEmbeddingCollection(
        "test",
        cache_dir=tmp_path,
        embedding_service=SimpleNamespace(model_name="test", dimension=2),
    )
    write_cache_data(
        collection.cache_file,
        {
            "cache_version": collection.CACHE_VERSION,
            "items_hash": "hash",
            "model_name": "test",
            "embeddings": np.zeros((1, 2)),
            "items": {"id": {"id": "id", "text": "text", "metadata": {}}},
            "item_ids": item_ids,
        },
    )
    assert collection._load_from_cache("hash") is False
    assert collection._items == {}
