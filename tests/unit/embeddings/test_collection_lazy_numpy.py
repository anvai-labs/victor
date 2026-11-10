"""Lazy numpy import in StaticEmbeddingCollection must not weaken cache loading.

Construction no longer imports numpy (bare installs), so every path that touches
``np`` must import it itself. ``_load_from_cache`` is called directly (tests,
future callers) and must validate a cache from a fresh module state instead of
failing on ``np is None`` and deleting a valid file as a "load error".
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from victor.framework.agent_factory import _log_skill_matcher_init_outcome
from victor.storage.embeddings import collections as collections_module
from victor.storage.embeddings.collections import CollectionItem, StaticEmbeddingCollection


class _StubEmbeddings:
    model_name = "stub"
    dimension = 2

    def embed_batch_sync(self, texts):
        return np.array([[float(i), 1.0] for i, _ in enumerate(texts)], dtype=np.float32)


def _items():
    return [CollectionItem(id="a", text="alpha"), CollectionItem(id="b", text="beta")]


def test_valid_cache_loads_from_fresh_module_state(tmp_path, monkeypatch):
    writer = StaticEmbeddingCollection(
        "lazy", cache_dir=tmp_path, embedding_service=_StubEmbeddings()
    )
    writer.initialize_sync(_items())
    items_hash = writer._calculate_items_hash(_items())
    assert writer.cache_file.exists()

    # Simulate a process where no vector math has run yet.
    monkeypatch.setattr(collections_module, "np", None)
    reader = StaticEmbeddingCollection(
        "lazy", cache_dir=tmp_path, embedding_service=_StubEmbeddings()
    )

    assert reader._load_from_cache(items_hash) is True
    assert reader.cache_file.exists(), "valid cache was deleted as a load error"
    assert reader._item_ids == ["a", "b"]
    np.testing.assert_array_equal(reader._embeddings, writer._embeddings)


def test_corrupt_embeddings_still_rejected_from_fresh_module_state(tmp_path, monkeypatch):
    from victor.core.data_cache import write_cache_data

    collection = StaticEmbeddingCollection(
        "lazy", cache_dir=tmp_path, embedding_service=_StubEmbeddings()
    )
    write_cache_data(
        collection.cache_file,
        {
            "cache_version": collection.CACHE_VERSION,
            "items_hash": "hash",
            "model_name": "stub",
            "embeddings": np.array([[np.nan, 1.0]]),
            "items": {"a": {"id": "a", "text": "alpha", "metadata": {}}},
            "item_ids": ["a"],
        },
    )
    monkeypatch.setattr(collections_module, "np", None)

    assert collection._load_from_cache("hash") is False
    assert collection._embeddings is None


def test_skill_matcher_init_failure_is_consumed_and_logged(caplog):
    loop = asyncio.new_event_loop()
    try:
        future = loop.create_future()
        future.set_exception(ModuleNotFoundError("No module named 'numpy'"))
        with caplog.at_level(logging.INFO, logger="victor.framework.agent_factory"):
            _log_skill_matcher_init_outcome(future)
        # Retrieved: asyncio will not report "exception was never retrieved".
        assert future._log_traceback is False
        assert "Skill auto-selection unavailable" in caplog.text
    finally:
        loop.close()


@pytest.mark.parametrize("outcome", ["ok", "cancelled"])
def test_skill_matcher_init_success_or_cancel_logs_nothing(caplog, outcome):
    loop = asyncio.new_event_loop()
    try:
        future = loop.create_future()
        if outcome == "ok":
            future.set_result(None)
        else:
            future.cancel()
        with caplog.at_level(logging.INFO, logger="victor.framework.agent_factory"):
            _log_skill_matcher_init_outcome(future)
        assert "Skill auto-selection unavailable" not in caplog.text
    finally:
        loop.close()
