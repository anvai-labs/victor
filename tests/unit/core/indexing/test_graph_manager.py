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

"""Tests for GraphManager."""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from victor.core.indexing.graph_manager import GraphManager
from victor.core.indexing.file_watcher import FileChangeEvent, FileChangeType


@pytest.fixture(autouse=True)
async def reset_graph_manager():
    """Reset GraphManager singleton before each test."""
    manager = GraphManager.get_instance()
    await manager.clear_cache()
    manager._watcher_subscribed.clear()
    manager._watcher_callbacks.clear()
    manager._refresh_failures.clear()
    manager._last_refresh_completed_at.clear()
    manager._last_refresh_source_mtime.clear()
    yield
    await manager.clear_cache()
    manager._watcher_subscribed.clear()
    manager._watcher_callbacks.clear()
    manager._refresh_failures.clear()
    manager._last_refresh_completed_at.clear()
    manager._last_refresh_source_mtime.clear()


@pytest.fixture
def temp_codebase():
    """Create a temporary codebase directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create some test files
        (root / "test.py").write_text("print('hello')")
        (root / "src").mkdir()
        (root / "src" / "module.py").write_text("def foo(): pass")
        yield root


class TestGraphManager:
    """Test GraphManager functionality."""

    @pytest.mark.asyncio
    async def test_singleton_instance(self):
        """Verify singleton pattern works."""
        manager1 = GraphManager.get_instance()
        manager2 = GraphManager.get_instance()

        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_get_stats_empty(self):
        """Verify stats are correct for empty manager."""
        manager = GraphManager.get_instance()

        stats = manager.get_stats()
        assert stats["total_graphs"] == 0
        assert stats["fresh_graphs"] == 0
        assert stats["stale_graphs"] == 0
        assert stats["watched_roots"] == 0

    @pytest.mark.asyncio
    async def test_cache_graph(self, temp_codebase):
        """Verify graph caching works."""
        manager = GraphManager.get_instance()

        # Mock graph result
        mock_graph = {"nodes": 5, "edges": 10}
        cache_key = f"{temp_codebase}:pagerank"

        manager._graph_cache[cache_key] = {
            "graph": mock_graph,
            "built_at": 1234567890.0,
            "stale": False,
        }

        stats = manager.get_stats()
        assert stats["total_graphs"] == 1
        assert stats["fresh_graphs"] == 1
        assert stats["stale_graphs"] == 0

    @pytest.mark.asyncio
    async def test_mark_stale_on_file_change(self, temp_codebase):
        """Verify file changes mark graphs as stale."""
        manager = GraphManager.get_instance()
        root_str = str(temp_codebase.resolve())

        # Cache some graphs
        for mode in ["pagerank", "centrality", "trace"]:
            cache_key = f"{root_str}:{mode}"
            manager._graph_cache[cache_key] = {
                "graph": {"mode": mode},
                "built_at": 1234567890.0,
                "stale": False,
            }

        # Simulate file change event
        event = FileChangeEvent(
            path=temp_codebase / "test.py",
            change_type=FileChangeType.MODIFIED,
            timestamp=None,
        )

        await manager._on_file_change(event, temp_codebase, None)

        # All graphs should be marked stale
        stats = manager.get_stats()
        assert stats["total_graphs"] == 3
        assert stats["fresh_graphs"] == 0
        assert stats["stale_graphs"] == 3

    @pytest.mark.asyncio
    async def test_invalidate_root(self, temp_codebase):
        """Verify invalidate_root marks all graphs for root as stale."""
        manager = GraphManager.get_instance()
        root_str = str(temp_codebase.resolve())

        # Cache some graphs
        for mode in ["pagerank", "centrality"]:
            cache_key = f"{root_str}:{mode}"
            manager._graph_cache[cache_key] = {
                "graph": {"mode": mode},
                "built_at": 1234567890.0,
                "stale": False,
            }

        # Invalidate root
        invalidated = await manager.invalidate_root(temp_codebase)

        assert invalidated == 2

        stats = manager.get_stats()
        assert stats["stale_graphs"] == 2

    @pytest.mark.asyncio
    async def test_clear_cache_all(self, temp_codebase):
        """Verify clear_cache removes all graphs."""
        manager = GraphManager.get_instance()

        # Cache some graphs
        for mode in ["pagerank", "centrality"]:
            cache_key = f"{temp_codebase}:{mode}"
            manager._graph_cache[cache_key] = {
                "graph": {"mode": mode},
                "built_at": 1234567890.0,
                "stale": False,
            }

        # Clear all
        cleared = await manager.clear_cache()

        assert cleared == 2

        stats = manager.get_stats()
        assert stats["total_graphs"] == 0

    @pytest.mark.asyncio
    async def test_clear_cache_specific_root(self, temp_codebase):
        """Verify clear_cache removes graphs for specific root."""
        manager = GraphManager.get_instance()
        root_str = str(temp_codebase.resolve())

        # Cache graphs for two different roots
        with tempfile.TemporaryDirectory() as tmpdir2:
            root2 = Path(tmpdir2)
            root2_str = str(root2.resolve())

            for i, root in enumerate([temp_codebase, root2]):
                root_str_current = str(root.resolve())
                for mode in ["pagerank", "centrality"]:
                    cache_key = f"{root_str_current}:{mode}"
                    manager._graph_cache[cache_key] = {
                        "graph": {"root": i},
                        "built_at": 1234567890.0,
                        "stale": False,
                    }

            # Clear only first root
            cleared = await manager.clear_cache(temp_codebase)

            assert cleared == 2

            stats = manager.get_stats()
            assert stats["total_graphs"] == 2  # Still 2 from root2

    @pytest.mark.asyncio
    async def test_watcher_subscription_tracking(self, temp_codebase):
        """Verify file watcher subscription is tracked."""
        manager = GraphManager.get_instance()

        # Initially not subscribed
        root_str = str(temp_codebase.resolve())
        assert root_str not in manager._watcher_subscribed

        # Ensure file watcher (subscribes)
        await manager._ensure_file_watcher(temp_codebase, None)

        # Should be subscribed now
        assert root_str in manager._watcher_subscribed

        # Call again - should not duplicate subscription
        await manager._ensure_file_watcher(temp_codebase, None)

        # Still only one subscription
        assert len(manager._watcher_subscribed) == 1

    @pytest.mark.asyncio
    async def test_stop_background_refresh_unsubscribes_file_watcher(
        self, temp_codebase, monkeypatch
    ):
        """Stopping refresh should detach watcher callbacks and stop idle watchers."""
        manager = GraphManager.get_instance()
        root_str = str(temp_codebase.resolve())
        stop_calls = []

        class FakeWatcher:
            def __init__(self) -> None:
                self.subscribers = set()
                self.unsubscribed = []

            def subscribe(self, callback):
                self.subscribers.add(callback)

            def unsubscribe(self, callback):
                self.unsubscribed.append(callback)
                self.subscribers.discard(callback)

            def get_stats(self):
                return {"subscribers": len(self.subscribers)}

        fake_watcher = FakeWatcher()

        class FakeWatcherRegistry:
            def __init__(self) -> None:
                self._watchers = {root_str: fake_watcher}

            async def get_watcher(self, root, **kwargs):
                return fake_watcher

            async def stop_watcher(self, root):
                stop_calls.append(root.resolve())
                self._watchers.pop(str(root.resolve()), None)
                return True

        monkeypatch.setattr(
            "victor.core.indexing.graph_manager.FileWatcherRegistry.get_instance",
            staticmethod(lambda: FakeWatcherRegistry()),
        )

        await manager.ensure_background_refresh(temp_codebase)
        assert root_str in manager._watcher_subscribed
        assert root_str in manager._watcher_callbacks

        await manager.stop_background_refresh(temp_codebase)

        assert root_str not in manager._watcher_subscribed
        assert root_str not in manager._watcher_callbacks
        assert len(fake_watcher.unsubscribed) == 1
        assert stop_calls == [temp_codebase.resolve()]

    @pytest.mark.asyncio
    async def test_background_refresh_cooldown_delays_recent_success(self, temp_codebase):
        """Recent successful refreshes should suppress immediate rescan churn."""
        manager = GraphManager.get_instance()
        root_str = str(temp_codebase.resolve())
        manager._background_refresh[root_str] = {"min_refresh_interval_seconds": 30.0}
        manager._last_refresh_completed_at[root_str] = time.time() - 5.0

        delay = manager._refresh_cooldown_delay(root_str)

        assert 20.0 <= delay <= 30.0

    @pytest.mark.asyncio
    async def test_background_refresh_skips_indexing_when_source_mtime_unchanged(
        self, temp_codebase, monkeypatch
    ):
        """Repeated refreshes without source changes should not invoke the indexer."""
        manager = GraphManager.get_instance()
        root = temp_codebase.resolve()
        root_str = str(root)
        manager._background_refresh[root_str] = {
            "pending": False,
            "enable_ccg": True,
            "on_refresh_complete": None,
        }
        manager._last_refresh_source_mtime[root_str] = 123.0

        monkeypatch.setattr("victor.tools.common.latest_mtime", lambda _root: 123.0)

        class FailingPipeline:
            def __init__(self, *args, **kwargs):
                raise AssertionError("indexer should not be constructed")

        monkeypatch.setattr(
            "victor.core.graph_rag.GraphIndexingPipeline",
            FailingPipeline,
        )

        stats = await manager._refresh_graph_index(root)

        assert stats.files_processed == 0
        assert stats.files_deleted == 0
        assert manager._last_refresh_source_mtime[root_str] == 123.0

    @pytest.mark.asyncio
    async def test_file_deleted_marks_stale(self, temp_codebase):
        """Verify file deletion marks graphs as stale."""
        manager = GraphManager.get_instance()
        root_str = str(temp_codebase.resolve())

        # Cache a graph
        cache_key = f"{root_str}:pagerank"
        manager._graph_cache[cache_key] = {
            "graph": {"nodes": 5},
            "built_at": 1234567890.0,
            "stale": False,
        }

        # Simulate file deletion
        event = FileChangeEvent(
            path=temp_codebase / "test.py",
            change_type=FileChangeType.DELETED,
            timestamp=None,
        )

        await manager._on_file_change(event, temp_codebase, None)

        # Graph should be marked stale
        cache_entry = manager._graph_cache.get(cache_key)
        assert cache_entry is not None
        assert cache_entry["stale"] is True

    @pytest.mark.asyncio
    async def test_file_created_marks_stale(self, temp_codebase):
        """Verify file creation marks graphs as stale."""
        manager = GraphManager.get_instance()
        root_str = str(temp_codebase.resolve())

        # Cache a graph
        cache_key = f"{root_str}:pagerank"
        manager._graph_cache[cache_key] = {
            "graph": {"nodes": 5},
            "built_at": 1234567890.0,
            "stale": False,
        }

        # Simulate file creation
        event = FileChangeEvent(
            path=temp_codebase / "new.py",
            change_type=FileChangeType.CREATED,
            timestamp=None,
        )

        await manager._on_file_change(event, temp_codebase, None)

        # Graph should be marked stale
        cache_entry = manager._graph_cache.get(cache_key)
        assert cache_entry is not None
        assert cache_entry["stale"] is True

    @pytest.mark.asyncio
    async def test_different_roots_separate_caches(self, temp_codebase):
        """Verify different roots have separate graph caches."""
        manager = GraphManager.get_instance()

        with tempfile.TemporaryDirectory() as tmpdir2:
            root2 = Path(tmpdir2)

            # Cache graphs for both roots
            for i, root in enumerate([temp_codebase, root2]):
                root_str = str(root.resolve())
                cache_key = f"{root_str}:pagerank"
                manager._graph_cache[cache_key] = {
                    "graph": {"root": i},
                    "built_at": 1234567890.0,
                    "stale": False,
                }

            stats = manager.get_stats()
            assert stats["total_graphs"] == 2

            # Invalidate first root - should not affect second
            await manager.invalidate_root(temp_codebase)

            stats = manager.get_stats()
            assert stats["stale_graphs"] == 1  # Only first root

    @pytest.mark.asyncio
    async def test_different_modes_separate_caches(self, temp_codebase):
        """Verify different modes have separate caches."""
        manager = GraphManager.get_instance()

        # Cache graphs for different modes
        for mode in ["pagerank", "centrality", "trace"]:
            cache_key = f"{temp_codebase}:{mode}"
            manager._graph_cache[cache_key] = {
                "graph": {"mode": mode},
                "built_at": 1234567890.0,
                "stale": False,
            }

        stats = manager.get_stats()
        assert stats["total_graphs"] == 3

        # All should be fresh
        assert stats["fresh_graphs"] == 3

    @pytest.mark.asyncio
    async def test_background_refresh_failure_is_tracked_and_reported(
        self, temp_codebase, monkeypatch
    ):
        """Recoverable background refresh failures should be tracked without raising."""
        manager = GraphManager.get_instance()
        root = temp_codebase.resolve()
        root_str = str(root)
        notified = []

        manager._background_refresh[root_str] = {
            "pending": False,
            "on_refresh_error": lambda refreshed_root, exc: notified.append((refreshed_root, exc)),
        }

        async def _boom(_root):
            raise OSError(24, "Too many open files")

        monkeypatch.setattr(manager, "_refresh_graph_index", _boom)

        await manager._run_refresh_loop(root)

        assert len(notified) == 1
        assert notified[0][0] == root
        assert "Too many open files" in str(notified[0][1])

        failure = manager.get_stats()["refresh_failures"][root_str]
        assert failure["count"] == 1
        assert failure["error_type"] == "OSError"
        assert failure["category"] == "resource_exhaustion"
        assert failure["recoverable"] is True
        assert failure["severity"] == "warning"
        assert failure["retry_delay_seconds"] == 5.0
        assert failure["error_code"] == 24
        assert isinstance(failure["next_retry_at"], float)
        assert "Too many open files" in failure["last_error"]

    @pytest.mark.asyncio
    async def test_background_refresh_retries_transient_missing_file_once(
        self, temp_codebase, monkeypatch
    ):
        """Transient file races should be retried once before surfacing as failures."""
        manager = GraphManager.get_instance()
        root = temp_codebase.resolve()
        root_str = str(root)
        calls = 0

        manager._background_refresh[root_str] = {
            "pending": False,
            "on_refresh_error": None,
        }

        async def _flaky_refresh(_root):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise FileNotFoundError(2, "No such file or directory")
            return None

        monkeypatch.setattr(manager, "_refresh_graph_index", _flaky_refresh)

        await manager._run_refresh_loop(root)

        assert calls == 2
        assert root_str not in manager.get_stats()["refresh_failures"]


class TestQueryCacheInvalidation:
    """Tests for graph query cache invalidation on refresh."""

    def test_invalidate_query_cache_calls_invalidate_repo(self):
        """_invalidate_query_cache should route to the query cache's invalidate_repo."""
        with mock.patch(
            "victor.core.graph_rag.query_cache.get_graph_query_cache"
        ) as mock_get_cache:
            mock_cache = mock.MagicMock()
            mock_get_cache.return_value = mock_cache

            GraphManager._invalidate_query_cache("/some/repo")

            mock_cache.invalidate_repo.assert_called_once_with("/some/repo")

    def test_invalidate_query_cache_swallows_errors(self):
        """Cache invalidation failures must never break a refresh."""
        with mock.patch(
            "victor.core.graph_rag.query_cache.get_graph_query_cache",
            side_effect=RuntimeError("cache exploded"),
        ):
            GraphManager._invalidate_query_cache("/some/repo")  # must not raise
