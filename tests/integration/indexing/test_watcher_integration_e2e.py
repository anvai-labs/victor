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

"""Integration tests for file watcher initialization and cleanup."""

import asyncio
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

from victor.core.graph_rag import GraphIndexConfig, GraphIndexingPipeline
from victor.core.indexing.file_watcher import (
    FileChangeEvent,
    FileChangeType,
    FileWatcherRegistry,
)
from victor.core.indexing.watcher_initializer import (
    initialize_file_watchers,
    stop_file_watchers,
    get_project_paths_from_context,
    initialize_from_context,
    cleanup_session,
)
from victor.core.indexing.graph_manager import GraphManager
from victor.storage.graph import create_graph_store


@pytest.fixture(autouse=True)
async def reset_registries():
    """Reset all registries before each test."""
    # Reset file watcher registry
    fw_registry = FileWatcherRegistry.get_instance()
    await fw_registry.stop_all()
    fw_registry._watchers.clear()

    # Reset graph manager
    gm_manager = GraphManager.get_instance()
    await gm_manager.clear_cache()
    gm_manager._watcher_subscribed.clear()
    await gm_manager.stop_background_refresh()

    yield

    # Cleanup after test
    await fw_registry.stop_all()
    fw_registry._watchers.clear()
    await gm_manager.clear_cache()
    gm_manager._watcher_subscribed.clear()
    await gm_manager.stop_background_refresh()


@pytest.fixture
def temp_project():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "test.py").write_text("print('hello')")
        (root / "src").mkdir()
        (root / "src" / "module.py").write_text("def foo(): pass")
        yield root


class TestFileWatcherInitializer:
    """Test file watcher initialization utilities."""

    async def _build_initial_graph(self, temp_project: Path):
        """Build an initial graph and background refresh manager for a temp project."""
        graph_store = create_graph_store("sqlite", project_path=temp_project)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        pipeline = GraphIndexingPipeline(
            graph_store,
            GraphIndexConfig(
                root_path=temp_project,
                enable_ccg=False,
                enable_embeddings=False,
                enable_subgraph_cache=False,
            ),
        )
        await pipeline.index_repository()

        manager = GraphManager.get_instance()
        await manager.ensure_background_refresh(temp_project, enable_ccg=False)
        return graph_store, manager

    @pytest.mark.asyncio
    async def test_initialize_single_project(self, temp_project):
        """Verify single project initialization."""
        await initialize_file_watchers([temp_project])

        registry = FileWatcherRegistry.get_instance()
        stats = registry.get_stats()

        assert stats["total_watchers"] == 1
        assert str(temp_project.resolve()) in stats["watcher_details"]

    @pytest.mark.asyncio
    async def test_initialize_multiple_projects(self):
        """Verify multiple project initialization."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            root1 = Path(tmpdir1)
            (root1 / "test.py").write_text("print('1')")

            with tempfile.TemporaryDirectory() as tmpdir2:
                root2 = Path(tmpdir2)
                (root2 / "test.py").write_text("print('2')")

                await initialize_file_watchers([root1, root2])

                registry = FileWatcherRegistry.get_instance()
                stats = registry.get_stats()

                assert stats["total_watchers"] == 2

    @pytest.mark.asyncio
    async def test_initialize_empty_list(self):
        """Verify empty list doesn't create watchers."""
        await initialize_file_watchers([])

        registry = FileWatcherRegistry.get_instance()
        stats = registry.get_stats()

        assert stats["total_watchers"] == 0

    @pytest.mark.asyncio
    async def test_stop_specific_watchers(self, temp_project):
        """Verify stopping specific watchers."""
        await initialize_file_watchers([temp_project])

        # Verify watcher exists
        registry = FileWatcherRegistry.get_instance()
        assert registry.get_stats()["total_watchers"] == 1

        # Stop it
        await stop_file_watchers([temp_project])

        # Verify stopped
        assert registry.get_stats()["total_watchers"] == 0

    @pytest.mark.asyncio
    async def test_stop_all_watchers(self):
        """Verify stopping all watchers."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            root1 = Path(tmpdir1)
            (root1 / "test.py").write_text("print('1')")

            with tempfile.TemporaryDirectory() as tmpdir2:
                root2 = Path(tmpdir2)
                (root2 / "test.py").write_text("print('2')")

                await initialize_file_watchers([root1, root2])
                assert FileWatcherRegistry.get_instance().get_stats()["total_watchers"] == 2

                # Stop all
                await stop_file_watchers(None)

                # Verify all stopped
                assert FileWatcherRegistry.get_instance().get_stats()["total_watchers"] == 0

    @pytest.mark.asyncio
    async def test_get_project_paths_from_context(self, temp_project):
        """Verify project path extraction from context."""
        exec_ctx = {
            "cwd": str(temp_project),
            "project_paths": [str(temp_project)],
        }

        paths = get_project_paths_from_context(exec_ctx)

        # Should deduplicate
        assert len(paths) == 1
        assert paths[0].resolve() == temp_project.resolve()

    @pytest.mark.asyncio
    async def test_initialize_from_context(self, temp_project):
        """Verify initialization from context."""
        exec_ctx = {
            "cwd": str(temp_project),
        }

        await initialize_from_context(exec_ctx)

        registry = FileWatcherRegistry.get_instance()
        stats = registry.get_stats()

        assert stats["total_watchers"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_session(self, temp_project):
        """Verify session cleanup stops all watchers."""
        await initialize_file_watchers([temp_project])

        # Verify watcher exists
        assert FileWatcherRegistry.get_instance().get_stats()["total_watchers"] == 1

        # Cleanup
        await cleanup_session()

        # Verify cleaned up
        assert FileWatcherRegistry.get_instance().get_stats()["total_watchers"] == 0

    @pytest.mark.asyncio
    async def test_initialize_reuses_cached_watchers(self, temp_project):
        """Verify initializing same path twice reuses watcher."""
        # First initialization
        await initialize_file_watchers([temp_project])
        registry1 = FileWatcherRegistry.get_instance()
        watcher1 = registry1._watchers.get(str(temp_project.resolve()))

        # Second initialization
        await initialize_file_watchers([temp_project])
        registry2 = FileWatcherRegistry.get_instance()
        watcher2 = registry2._watchers.get(str(temp_project.resolve()))

        # Should be same watcher instance
        assert watcher1 is watcher2

    @pytest.mark.asyncio
    async def test_concurrent_initialization(self):
        """Verify concurrent initialization of different paths."""
        import asyncio

        async def init_and_check(path: Path):
            await initialize_file_watchers([path])
            registry = FileWatcherRegistry.get_instance()
            return registry.get_stats()["total_watchers"]

        with tempfile.TemporaryDirectory() as tmpdir1:
            root1 = Path(tmpdir1)
            (root1 / "test.py").write_text("print('1')")

            with tempfile.TemporaryDirectory() as tmpdir2:
                root2 = Path(tmpdir2)
                (root2 / "test.py").write_text("print('2')")

                # Initialize concurrently
                results = await asyncio.gather(
                    init_and_check(root1),
                    init_and_check(root2),
                )

                # Both should succeed
                # Results may be 1 or 2 depending on timing
                assert all(r >= 1 for r in results)

                # Final state should have 2 watchers
                stats = FileWatcherRegistry.get_instance().get_stats()
                assert stats["total_watchers"] == 2

    @pytest.mark.asyncio
    async def test_graph_manager_background_refresh_updates_project_graph(self, temp_project):
        """GraphManager should incrementally refresh the stored graph after file changes."""
        graph_store, manager = await self._build_initial_graph(temp_project)

        target_file = temp_project / "src" / "module.py"
        time.sleep(0.5)
        target_file.write_text("def bar():\n    return 2\n")

        await manager._on_file_change(
            FileChangeEvent(
                path=target_file,
                change_type=FileChangeType.MODIFIED,
                timestamp=datetime.now(),
            ),
            temp_project,
            None,
        )
        await manager.wait_for_refresh(temp_project)

        refreshed_nodes = await graph_store.get_nodes_by_file(str(target_file))
        # A synthetic module node (named after the file stem) is prepended per file.
        assert {node.name for node in refreshed_nodes} == {"bar", "module"}
        assert str(temp_project.resolve()) not in manager._refresh_tasks

    @pytest.mark.asyncio
    async def test_graph_manager_background_refresh_invokes_refresh_callback(self, temp_project):
        """Background refresh should report incremental stats through the completion callback."""
        graph_store = create_graph_store("sqlite", project_path=temp_project)
        await graph_store.initialize()
        await graph_store.delete_by_repo()

        pipeline = GraphIndexingPipeline(
            graph_store,
            GraphIndexConfig(
                root_path=temp_project,
                enable_ccg=False,
                enable_embeddings=False,
                enable_subgraph_cache=False,
            ),
        )
        await pipeline.index_repository()

        callback_events: list[tuple[Path, int, int, int]] = []

        def _on_refresh_complete(root: Path, stats) -> None:
            callback_events.append(
                (
                    root,
                    stats.files_processed,
                    stats.files_deleted,
                    stats.files_unchanged,
                )
            )

        manager = GraphManager.get_instance()
        await manager.ensure_background_refresh(
            temp_project,
            enable_ccg=False,
            on_refresh_complete=_on_refresh_complete,
        )

        target_file = temp_project / "src" / "module.py"
        time.sleep(0.5)
        target_file.write_text("def bar():\n    return 2\n")

        await manager._on_file_change(
            FileChangeEvent(
                path=target_file,
                change_type=FileChangeType.MODIFIED,
                timestamp=datetime.now(),
            ),
            temp_project,
            None,
        )
        await manager.wait_for_refresh(temp_project)

        assert callback_events
        refreshed_root, changed, deleted, unchanged = callback_events[-1]
        assert refreshed_root == temp_project.resolve()
        assert changed == 1
        assert deleted == 0
        assert unchanged == 1

    @pytest.mark.asyncio
    async def test_graph_manager_background_refresh_invokes_async_error_callback(
        self, temp_project, monkeypatch
    ):
        """Recoverable refresh failures should flow through async error callbacks."""
        manager = GraphManager.get_instance()
        callback_events: list[tuple[Path, str, str]] = []

        async def _on_refresh_error(root: Path, exc: Exception) -> None:
            callback_events.append((root, type(exc).__name__, str(exc)))

        await manager.ensure_background_refresh(
            temp_project,
            enable_ccg=False,
            on_refresh_error=_on_refresh_error,
        )

        async def _boom(_root: Path):
            raise OSError(24, "Too many open files")

        monkeypatch.setattr(manager, "_refresh_graph_index", _boom)

        await manager._run_refresh_loop(temp_project.resolve())

        assert callback_events == [
            (temp_project.resolve(), "OSError", "[Errno 24] Too many open files")
        ]
        failure = manager.get_stats()["refresh_failures"][str(temp_project.resolve())]
        assert failure["count"] == 1
        assert failure["error_type"] == "OSError"
        assert "Too many open files" in failure["last_error"]

    @pytest.mark.asyncio
    async def test_graph_manager_background_refresh_indexes_new_file(self, temp_project):
        """Background refresh should index newly created files."""
        graph_store, manager = await self._build_initial_graph(temp_project)

        new_file = temp_project / "src" / "new_module.py"
        time.sleep(0.5)
        new_file.write_text("def created_later():\n    return 3\n")

        await manager._on_file_change(
            FileChangeEvent(
                path=new_file,
                change_type=FileChangeType.CREATED,
                timestamp=datetime.now(),
            ),
            temp_project,
            None,
        )
        await manager.wait_for_refresh(temp_project)

        new_nodes = await graph_store.get_nodes_by_file(str(new_file))
        assert {node.name for node in new_nodes} == {"created_later", "new_module"}

    @pytest.mark.asyncio
    async def test_graph_manager_background_refresh_deletes_removed_file(self, temp_project):
        """Background refresh should remove graph state for deleted files."""
        graph_store, manager = await self._build_initial_graph(temp_project)

        deleted_file = temp_project / "src" / "module.py"
        assert {node.name for node in await graph_store.get_nodes_by_file(str(deleted_file))} == {
            "foo",
            "module",
        }

        time.sleep(0.5)
        deleted_file.unlink()

        await manager._on_file_change(
            FileChangeEvent(
                path=deleted_file,
                change_type=FileChangeType.DELETED,
                timestamp=datetime.now(),
            ),
            temp_project,
            None,
        )
        await manager.wait_for_refresh(temp_project)

        assert await graph_store.get_nodes_by_file(str(deleted_file)) == []

    @pytest.mark.asyncio
    async def test_graph_manager_background_refresh_handles_renamed_file(self, temp_project):
        """Background refresh should remove old graph state and index the renamed file."""
        graph_store, manager = await self._build_initial_graph(temp_project)

        old_file = temp_project / "src" / "module.py"
        renamed_file = temp_project / "src" / "renamed_module.py"
        time.sleep(0.5)
        old_file.rename(renamed_file)

        await manager._on_file_change(
            FileChangeEvent(
                path=renamed_file,
                change_type=FileChangeType.RENAMED,
                timestamp=datetime.now(),
                old_path=old_file,
            ),
            temp_project,
            None,
        )
        await manager.wait_for_refresh(temp_project)

        assert await graph_store.get_nodes_by_file(str(old_file)) == []
        assert {node.name for node in await graph_store.get_nodes_by_file(str(renamed_file))} == {
            "foo",
            "renamed_module",
        }


class TestCrossSessionBehavior:
    """Test cross-session persistence and invalidation."""

    @pytest.mark.asyncio
    async def test_new_session_detects_changes(self, temp_project):
        """Verify new session detects file changes between sessions."""
        import time

        # Session 1: Initialize and cache
        await initialize_from_context({"cwd": str(temp_project)})

        # Simulate file change
        time.sleep(0.5)  # Ensure different mtime
        (temp_project / "new.py").write_text("print('new file')")

        # Session 2: Reinitialize (should detect change)
        await cleanup_session()  # Simulate session end
        await initialize_from_context({"cwd": str(temp_project)})

        # Should still work (no errors)
        registry = FileWatcherRegistry.get_instance()
        stats = registry.get_stats()
        assert stats["total_watchers"] == 1

    @pytest.mark.asyncio
    async def test_graph_invalidation_across_sessions(self, temp_project):
        """Verify graph cache invalidation works across sessions."""
        from victor.core.indexing.graph_manager import GraphManager

        # Session 1: Build and cache graph
        manager = GraphManager.get_instance()
        cache_key = f"{temp_project.resolve()}:pagerank"
        manager._graph_cache[cache_key] = {
            "graph": {"test": "data"},
            "built_at": 1234567890.0,
            "stale": False,
        }

        # Simulate file change
        (temp_project / "test.py").write_text("print('modified')")

        # Session 2: Initialize file watcher (should mark stale)
        await initialize_from_context({"cwd": str(temp_project)})

        # Trigger file change event
        from victor.core.indexing.file_watcher import FileChangeEvent, FileChangeType

        event = FileChangeEvent(
            path=temp_project / "test.py",
            change_type=FileChangeType.MODIFIED,
            timestamp=None,
        )

        await manager._on_file_change(event, temp_project, None)

        # Graph should be marked stale
        cache_entry = manager._graph_cache.get(cache_key)
        assert cache_entry is not None
        assert cache_entry["stale"] is True
