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

"""Tests for EmbeddingCacheManager async/sync boundaries."""

from unittest.mock import AsyncMock

import pytest

from victor.storage.cache.embedding_cache_manager import EmbeddingCacheManager


class TestEmbeddingCacheManagerSyncBridge:
    """Tests for sync wrappers around async rebuild operations."""

    def test_rebuild_task_classifiers_sync_runs_from_sync_context(self):
        """Sync wrapper should bridge to the async rebuild implementation."""
        manager = object.__new__(EmbeddingCacheManager)
        manager.rebuild_task_classifiers = AsyncMock(return_value=17)

        result = manager.rebuild_task_classifiers_sync()

        assert result == 17
        manager.rebuild_task_classifiers.assert_awaited_once_with(None)

    @pytest.mark.asyncio
    async def test_rebuild_task_classifiers_sync_rejects_async_context(self):
        """Sync wrapper should fail fast when called from an async context."""
        manager = object.__new__(EmbeddingCacheManager)
        manager.rebuild_task_classifiers = AsyncMock(return_value=17)

        with pytest.raises(RuntimeError, match="Cannot call rebuild_task_classifiers_sync"):
            manager.rebuild_task_classifiers_sync()

        manager.rebuild_task_classifiers.assert_not_called()


def test_status_and_clear_cover_data_and_legacy_embedding_caches(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from victor.storage.cache.embedding_cache_manager import CacheType

    monkeypatch.setattr(
        "victor.config.settings.get_project_paths",
        lambda: SimpleNamespace(
            global_embeddings_dir=tmp_path,
            global_victor_dir=tmp_path / "global",
            project_victor_dir=tmp_path / "project",
        ),
    )
    manager = EmbeddingCacheManager()
    files = [
        "tool_embeddings_model_hash.v2.json",
        "tool_embeddings_model_hash.pkl",
        "task_classifier_collection.v2.json",
        "task_classifier_collection.pkl",
    ]
    for name in files:
        (tmp_path / name).write_text("derived cache")
    unrelated = tmp_path / "keep.json"
    unrelated.write_text("keep")
    assert len(manager.get_cache_info(CacheType.TOOL).files) == 2
    assert len(manager.get_cache_info(CacheType.INTENT).files) == 2
    result = manager.clear([CacheType.TOOL, CacheType.INTENT])
    assert result.cleared_files == 4
    assert all(not (tmp_path / name).exists() for name in files)
    assert unrelated.read_text() == "keep"
