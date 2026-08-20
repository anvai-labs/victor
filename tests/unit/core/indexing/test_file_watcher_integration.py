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

"""Tests for FileWatcherService and FileWatcherRegistry."""

import asyncio
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from victor.core.indexing.file_watcher import (
    FileChangeType,
    FileChangeEvent,
    FileWatcherService,
    FileWatcherRegistry,
)


@pytest.fixture(autouse=True)
async def reset_file_watcher_registry():
    """Reset FileWatcherRegistry singleton before each test."""
    registry = FileWatcherRegistry.get_instance()
    await registry.stop_all()
    registry._watchers.clear()
    yield
    await registry.stop_all()
    registry._watchers.clear()


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


class TestFileWatcherService:
    """Test FileWatcherService functionality."""

    @pytest.mark.asyncio
    async def test_initial_scan(self, temp_codebase):
        """Verify initial scan discovers all files."""
        watcher = FileWatcherService(temp_codebase, poll_interval_seconds=0.1)

        await watcher.start()

        # Should have discovered 2 files
        assert len(watcher._file_mtimes) == 2

        await watcher.stop()

    @pytest.mark.asyncio
    async def test_detect_new_file(self, temp_codebase):
        """Verify file watcher detects new files."""
        watcher = FileWatcherService(temp_codebase, poll_interval_seconds=0.1, debounce_seconds=0.2)

        changes_received = []

        def on_change(event):
            changes_received.append(event)

        watcher.subscribe(on_change)
        await watcher.start()

        # Create new file
        (temp_codebase / "new.py").write_text("print('new')")
        await asyncio.sleep(0.5)  # Wait for poll + debounce

        # Should detect CREATED event
        assert len(changes_received) >= 1
        assert any(
            c.change_type == FileChangeType.CREATED and c.path.name == "new.py"
            for c in changes_received
        )

        await watcher.stop()

    @pytest.mark.asyncio
    async def test_detect_modified_file(self, temp_codebase):
        """Verify file watcher detects file modifications."""
        watcher = FileWatcherService(temp_codebase, poll_interval_seconds=0.1, debounce_seconds=0.2)

        changes_received = []

        def on_change(event):
            changes_received.append(event)

        watcher.subscribe(on_change)
        await watcher.start()

        # Modify existing file
        test_file = temp_codebase / "test.py"
        test_file.write_text("print('modified')")
        await asyncio.sleep(0.5)  # Wait for poll + debounce

        # Should detect MODIFIED event
        assert len(changes_received) >= 1
        assert any(
            c.change_type == FileChangeType.MODIFIED and c.path.name == "test.py"
            for c in changes_received
        )

        await watcher.stop()

    @pytest.mark.asyncio
    async def test_detect_deleted_file(self, temp_codebase):
        """Verify file watcher detects file deletions."""
        watcher = FileWatcherService(temp_codebase, poll_interval_seconds=0.1, debounce_seconds=0.2)

        changes_received = []

        def on_change(event):
            changes_received.append(event)

        watcher.subscribe(on_change)
        await watcher.start()

        # Delete existing file
        test_file = temp_codebase / "test.py"
        test_file.unlink()
        await asyncio.sleep(0.5)  # Wait for poll + debounce

        # Should detect DELETED event
        assert len(changes_received) >= 1
        assert any(
            c.change_type == FileChangeType.DELETED and c.path.name == "test.py"
            for c in changes_received
        )

        await watcher.stop()

    @pytest.mark.asyncio
    async def test_debouncing_rapid_changes(self, temp_codebase):
        """Verify rapid changes are debounced."""
        watcher = FileWatcherService(temp_codebase, poll_interval_seconds=0.1, debounce_seconds=0.5)

        changes_received = []

        def on_change(event):
            changes_received.append(event)

        watcher.subscribe(on_change)
        await watcher.start()

        # Make rapid changes
        test_file = temp_codebase / "test.py"
        for i in range(5):
            test_file.write_text(f"v{i}")
            await asyncio.sleep(0.05)  # Rapid changes

        await asyncio.sleep(1.0)  # Wait for debounce

        # Should receive fewer events than changes (debounced)
        assert len(changes_received) < 5

        await watcher.stop()

    @pytest.mark.asyncio
    async def test_exclude_patterns(self, temp_codebase):
        """Verify exclude patterns work correctly."""
        watcher = FileWatcherService(
            temp_codebase, poll_interval_seconds=0.1, exclude_patterns={"src"}
        )

        await watcher.start()

        # Should not discover files in src/
        assert "src" not in str(watcher._file_mtimes)
        assert len(watcher._file_mtimes) == 1  # Only test.py

        await watcher.stop()

    @pytest.mark.asyncio
    async def test_excluded_directories_are_pruned_from_scan(self, temp_codebase):
        """Runtime directories should not be traversed during watcher scans."""
        victor_dir = temp_codebase / ".victor"
        nested_dir = victor_dir / "backups"
        nested_dir.mkdir(parents=True)
        (nested_dir / "large-runtime-state.db").write_text("runtime state")

        watcher = FileWatcherService(temp_codebase, poll_interval_seconds=0.1)

        await watcher.start()

        assert str(victor_dir) not in str(watcher._file_mtimes)
        assert len(watcher._file_mtimes) == 2

        await watcher.stop()

    @pytest.mark.asyncio
    async def test_default_excludes_coverage_temp_files(self, temp_codebase):
        """Coverage temp artifacts should not enter the watched file set."""
        coverage_file = temp_codebase / ".coverage.Vijays-MacBook-Pro.local.69421.XNBbPQdx.c"
        coverage_file.write_text("temporary coverage data")

        watcher = FileWatcherService(temp_codebase, poll_interval_seconds=0.1)

        await watcher.start()

        assert str(coverage_file) not in watcher._file_mtimes
        assert len(watcher._file_mtimes) == 2

        await watcher.stop()

    @pytest.mark.asyncio
    async def test_default_excludes_coverage_temp_file_events(self, temp_codebase):
        """Creating ignored coverage temp files should not publish change events."""
        watcher = FileWatcherService(
            temp_codebase,
            poll_interval_seconds=0.1,
            debounce_seconds=0.2,
        )
        changes_received = []

        def on_change(event):
            changes_received.append(event)

        watcher.subscribe(on_change)
        await watcher.start()

        coverage_file = temp_codebase / ".coverage.Vijays-MacBook-Pro.local.69421.XNBbPQdx.c"
        coverage_file.write_text("temporary coverage data")
        await asyncio.sleep(0.5)

        assert not any(change.path == coverage_file for change in changes_received)

        await watcher.stop()

    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe(self, temp_codebase):
        """Verify subscribe/unsubscribe mechanics."""
        watcher = FileWatcherService(temp_codebase, poll_interval_seconds=0.1)

        callback1 = mock.Mock()
        callback2 = mock.Mock()

        watcher.subscribe(callback1)
        assert len(watcher._subscribers) == 1

        watcher.subscribe(callback2)
        assert len(watcher._subscribers) == 2

        watcher.unsubscribe(callback1)
        assert len(watcher._subscribers) == 1

        watcher.unsubscribe(callback2)
        assert len(watcher._subscribers) == 0

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, temp_codebase):
        """Verify statistics are tracked correctly."""
        watcher = FileWatcherService(temp_codebase, poll_interval_seconds=0.1)

        await watcher.start()

        stats = watcher.get_stats()
        assert stats["files_watched"] == 2
        assert stats["watching"] == 1

        await watcher.stop()

        stats = watcher.get_stats()
        assert stats["watching"] == 0


class TestFileWatcherRegistry:
    """Test FileWatcherRegistry singleton functionality."""

    @pytest.mark.asyncio
    async def test_singleton_instance(self):
        """Verify singleton pattern works."""
        registry1 = FileWatcherRegistry.get_instance()
        registry2 = FileWatcherRegistry.get_instance()

        assert registry1 is registry2

    @pytest.mark.asyncio
    async def test_get_watcher_caches(self, temp_codebase):
        """Verify get_watcher returns cached instance."""
        registry = FileWatcherRegistry.get_instance()

        watcher1 = await registry.get_watcher(temp_codebase)
        watcher2 = await registry.get_watcher(temp_codebase)

        assert watcher1 is watcher2

    @pytest.mark.asyncio
    async def test_different_paths_different_watchers(self, temp_codebase):
        """Verify different paths get different watchers."""
        registry = FileWatcherRegistry.get_instance()

        with tempfile.TemporaryDirectory() as tmpdir2:
            root2 = Path(tmpdir2)
            # Add a file so the directory isn't skipped by empty directory optimization
            (root2 / "file.txt").write_text("content")

            watcher1 = await registry.get_watcher(temp_codebase)
            watcher2 = await registry.get_watcher(root2)

            assert watcher1 is not watcher2
            assert len(registry._watchers) == 2

    @pytest.mark.asyncio
    async def test_stop_watcher(self, temp_codebase):
        """Verify stop_watcher removes watcher from registry."""
        registry = FileWatcherRegistry.get_instance()

        watcher = await registry.get_watcher(temp_codebase)
        # FileWatcherRegistry uses resolved paths
        resolved_path = str(temp_codebase.resolve())
        assert resolved_path in registry._watchers

        stopped = await registry.stop_watcher(temp_codebase)
        assert stopped is True
        assert resolved_path not in registry._watchers

    @pytest.mark.asyncio
    async def test_stop_all(self, temp_codebase):
        """Verify stop_all removes all watchers."""
        registry = FileWatcherRegistry.get_instance()

        with tempfile.TemporaryDirectory() as tmpdir2:
            root2 = Path(tmpdir2)
            # Add a file so the directory isn't skipped by empty directory optimization
            (root2 / "file.txt").write_text("content")

            await registry.get_watcher(temp_codebase)
            await registry.get_watcher(root2)

            assert len(registry._watchers) == 2

            await registry.stop_all()

            assert len(registry._watchers) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, temp_codebase):
        """Verify get_stats returns watcher statistics."""
        registry = FileWatcherRegistry.get_instance()

        await registry.get_watcher(temp_codebase)

        stats = registry.get_stats()
        assert stats["total_watchers"] == 1
        assert "watcher_details" in stats
        # FileWatcherRegistry uses resolved paths
        resolved_path = str(temp_codebase.resolve())
        assert resolved_path in stats["watcher_details"]


class TestFileChangeEvent:
    """Test FileChangeEvent dataclass."""

    def test_event_creation(self):
        """Verify event creation works."""
        event = FileChangeEvent(
            path=Path("/test/file.py"),
            change_type=FileChangeType.CREATED,
            timestamp=None,
        )

        assert event.path == Path("/test/file.py")
        assert event.change_type == FileChangeType.CREATED
        assert event.old_path is None

    def test_event_string_representation(self):
        """Verify event string representation."""
        event = FileChangeEvent(
            path=Path("/test/file.py"),
            change_type=FileChangeType.MODIFIED,
            timestamp=None,
        )

        assert "modified" in str(event).lower()
        assert "file.py" in str(event)

    def test_rename_event(self):
        """Verify rename event includes old_path."""
        event = FileChangeEvent(
            path=Path("/test/new_file.py"),
            change_type=FileChangeType.RENAMED,
            timestamp=None,
            old_path=Path("/test/old_file.py"),
        )

        assert event.path == Path("/test/new_file.py")
        assert event.old_path == Path("/test/old_file.py")
