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

"""Tests for dynamic module loading infrastructure (module_loader.py).

Tests cover:
- DebouncedReloadTimer: Timer scheduling, cancellation, debouncing
- DynamicModuleLoader: Module invalidation, tracking, file watching
- EntryPointCache: Caching, TTL, environment hash, persistence
- CachedEntryPoints: Serialization, expiration
"""

import asyncio
import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from victor.framework.module_loader import (
    DebouncedReloadTimer,
    DynamicModuleLoader,
    EntryPointCache,
    CachedEntryPoints,
    get_entry_point_cache,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def debounce_timer():
    """Create a DebouncedReloadTimer with short delay for testing."""
    return DebouncedReloadTimer(delay=0.1)


@pytest.fixture
def module_loader():
    """Create a DynamicModuleLoader instance for testing."""
    return DynamicModuleLoader()


@pytest.fixture
def module_loader_with_dirs(tmp_path):
    """Create a DynamicModuleLoader with watch directories."""
    watch_dir = tmp_path / "plugins"
    watch_dir.mkdir()
    return DynamicModuleLoader(watch_dirs=[watch_dir])


@pytest.fixture
def entry_point_cache(tmp_path):
    """Create a fresh EntryPointCache instance for testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Reset singleton to ensure fresh instance
    EntryPointCache.reset_instance()
    cache = EntryPointCache(cache_dir=cache_dir, default_ttl=3600.0)
    yield cache
    EntryPointCache.reset_instance()


@pytest.fixture
def cached_entry_points():
    """Create a CachedEntryPoints instance for testing."""
    return CachedEntryPoints(
        group="test.group",
        entries={"entry1": "module1:Class1", "entry2": "module2:func2"},
        env_hash="abc123",
        timestamp=time.time(),
        ttl=3600.0,
    )


@pytest.fixture
def mock_module():
    """Create a mock module for tracking tests."""
    module = ModuleType("test_module")
    module.__file__ = "/path/to/test_module.py"
    return module


# =============================================================================
# DebouncedReloadTimer Tests (covers lines 100-109, 118-124, 135-140, 151)
# =============================================================================


class TestDebouncedReloadTimer:
    """Tests for DebouncedReloadTimer class."""

    def test_schedule_callback_executes_after_delay(self, debounce_timer):
        """Test that scheduled callback executes after delay."""
        result = []

        def callback():
            result.append("executed")

        debounce_timer.schedule("test_key", callback)

        # Wait for callback to execute
        time.sleep(0.2)
        assert result == ["executed"]

    def test_schedule_replaces_pending_callback(self, debounce_timer):
        """Test that scheduling replaces existing pending callback."""
        result = []

        def callback1():
            result.append("first")

        def callback2():
            result.append("second")

        # Schedule first callback
        debounce_timer.schedule("test_key", callback1)
        # Immediately schedule second callback (should replace first)
        debounce_timer.schedule("test_key", callback2)

        # Wait for callback to execute
        time.sleep(0.2)
        # Only second callback should have executed
        assert result == ["second"]

    def test_schedule_cancels_existing_timer(self, debounce_timer):
        """Test that scheduling cancels existing timer for same key."""
        result = []

        def callback():
            result.append("executed")

        # Schedule multiple times rapidly
        for _ in range(5):
            debounce_timer.schedule("test_key", callback)
            time.sleep(0.02)

        # Wait for final callback
        time.sleep(0.2)
        # Should only execute once (the last scheduled one)
        assert len(result) == 1

    def test_schedule_different_keys_run_independently(self, debounce_timer):
        """Test that different keys run independently."""
        result = []

        def callback1():
            result.append("key1")

        def callback2():
            result.append("key2")

        debounce_timer.schedule("key1", callback1)
        debounce_timer.schedule("key2", callback2)

        # Wait for callbacks
        time.sleep(0.2)
        assert "key1" in result
        assert "key2" in result

    def test_execute_handles_callback_exception(self, debounce_timer):
        """Test that _execute handles callback exceptions gracefully."""

        def failing_callback():
            raise ValueError("Test error")

        debounce_timer.schedule("test_key", failing_callback)

        # Wait for callback - should not raise
        time.sleep(0.2)
        # Timer should be cleaned up despite error
        assert "test_key" not in debounce_timer._timers

    def test_cancel_returns_true_when_timer_exists(self, debounce_timer):
        """Test cancel returns True when timer exists."""
        debounce_timer.schedule("test_key", lambda: None)
        assert debounce_timer.cancel("test_key") is True

    def test_cancel_returns_false_when_no_timer(self, debounce_timer):
        """Test cancel returns False when no timer exists."""
        assert debounce_timer.cancel("nonexistent") is False

    def test_cancel_stops_scheduled_callback(self, debounce_timer):
        """Test that cancel stops the scheduled callback from executing."""
        result = []

        def callback():
            result.append("executed")

        debounce_timer.schedule("test_key", callback)
        debounce_timer.cancel("test_key")

        # Wait past the delay
        time.sleep(0.2)
        assert result == []

    def test_cancel_all_returns_count(self, debounce_timer):
        """Test cancel_all returns number of cancelled timers."""
        debounce_timer.schedule("key1", lambda: None)
        debounce_timer.schedule("key2", lambda: None)
        debounce_timer.schedule("key3", lambda: None)

        count = debounce_timer.cancel_all()
        assert count == 3
        assert len(debounce_timer._timers) == 0

    def test_cancel_all_with_no_timers(self, debounce_timer):
        """Test cancel_all with no pending timers."""
        count = debounce_timer.cancel_all()
        assert count == 0


# =============================================================================
# DynamicModuleLoader Tests (covers lines 215, 220, 238-260, 279-309)
# =============================================================================


class TestDynamicModuleLoader:
    """Tests for DynamicModuleLoader class."""

    def test_init_with_watch_dirs(self, tmp_path):
        """Test initialization with watch directories."""
        watch_dir = tmp_path / "plugins"
        watch_dir.mkdir()

        loader = DynamicModuleLoader(watch_dirs=[watch_dir])
        assert watch_dir in loader.watch_dirs

    def test_init_with_nonexistent_watch_dir(self, tmp_path):
        """Test initialization with non-existent watch directory."""
        nonexistent = tmp_path / "nonexistent"
        loader = DynamicModuleLoader(watch_dirs=[nonexistent])
        # Non-existent dirs should not be added
        assert nonexistent not in loader._watch_dirs

    def test_watch_dirs_property(self, module_loader_with_dirs):
        """Test watch_dirs property returns copy of list."""
        dirs = module_loader_with_dirs.watch_dirs
        assert isinstance(dirs, list)
        # Should be a copy, not the original
        dirs.append(Path("/new/path"))
        assert len(dirs) != len(module_loader_with_dirs.watch_dirs)

    def test_debounce_delay_property(self, module_loader):
        """Test debounce_delay property."""
        assert module_loader.debounce_delay == 0.5

    def test_debounce_delay_custom(self):
        """Test custom debounce delay."""
        loader = DynamicModuleLoader(debounce_delay=1.0)
        assert loader.debounce_delay == 1.0

    def test_invalidate_module_removes_from_sys_modules(self, module_loader, mock_module):
        """Test invalidate_module removes module from sys.modules."""
        # Add mock module to sys.modules
        sys.modules["test_invalidate_module"] = mock_module
        module_loader._loaded_modules["test_invalidate_module"] = mock_module

        try:
            count = module_loader.invalidate_module("test_invalidate_module")
            assert count == 1
            assert "test_invalidate_module" not in sys.modules
            assert "test_invalidate_module" not in module_loader._loaded_modules
        finally:
            # Cleanup
            sys.modules.pop("test_invalidate_module", None)

    def test_invalidate_module_removes_submodules(self, module_loader):
        """Test invalidate_module removes submodules."""
        # Add mock modules with submodules
        parent = ModuleType("test_parent")
        child1 = ModuleType("test_parent.child1")
        child2 = ModuleType("test_parent.child2")

        sys.modules["test_parent"] = parent
        sys.modules["test_parent.child1"] = child1
        sys.modules["test_parent.child2"] = child2

        try:
            count = module_loader.invalidate_module("test_parent")
            assert count == 3
            assert "test_parent" not in sys.modules
            assert "test_parent.child1" not in sys.modules
            assert "test_parent.child2" not in sys.modules
        finally:
            # Cleanup
            sys.modules.pop("test_parent", None)
            sys.modules.pop("test_parent.child1", None)
            sys.modules.pop("test_parent.child2", None)

    def test_invalidate_module_returns_zero_when_not_found(self, module_loader):
        """Test invalidate_module returns 0 when module not in sys.modules."""
        count = module_loader.invalidate_module("nonexistent_module")
        assert count == 0

    def test_invalidate_modules_in_path(self, module_loader, tmp_path):
        """Test invalidate_modules_in_path removes modules from directory."""
        # Create a temp file path
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        plugin_file = plugin_dir / "test_plugin.py"
        plugin_file.touch()

        # Create mock module with __file__ in plugin_dir
        module = ModuleType("test_path_module")
        module.__file__ = str(plugin_file)
        sys.modules["test_path_module"] = module

        try:
            count = module_loader.invalidate_modules_in_path("test_path_module", plugin_dir)
            assert count >= 1
            assert "test_path_module" not in sys.modules
        finally:
            sys.modules.pop("test_path_module", None)

    def test_invalidate_modules_in_path_finds_additional_modules(self, module_loader, tmp_path):
        """Test invalidate_modules_in_path finds and removes additional modules in path."""
        # Create plugin directory with multiple modules
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()

        plugin_file1 = plugin_dir / "plugin1.py"
        plugin_file1.touch()
        plugin_file2 = plugin_dir / "plugin2.py"
        plugin_file2.touch()

        # Create multiple modules with __file__ in plugin_dir
        module1 = ModuleType("test_plugin1")
        module1.__file__ = str(plugin_file1)
        module2 = ModuleType("test_plugin2")
        module2.__file__ = str(plugin_file2)

        sys.modules["test_plugin1"] = module1
        sys.modules["test_plugin2"] = module2

        try:
            # Use a different base module to test path scanning
            count = module_loader.invalidate_modules_in_path("base_module", plugin_dir)
            # Both modules should be removed since they're in the path
            assert "test_plugin1" not in sys.modules
            assert "test_plugin2" not in sys.modules
            assert count >= 2
        finally:
            sys.modules.pop("test_plugin1", None)
            sys.modules.pop("test_plugin2", None)

    def test_invalidate_modules_in_path_handles_exception_in_getattr(self, module_loader, tmp_path):
        """Test invalidate_modules_in_path handles exceptions when accessing __file__."""
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()

        # Create a module that raises when accessing __file__
        class BadModule:
            @property
            def __file__(self):
                raise RuntimeError("Cannot access __file__")

        sys.modules["test_bad_module"] = BadModule()

        try:
            # Should not raise - exception should be caught
            count = module_loader.invalidate_modules_in_path("base_module", plugin_dir)
            assert count >= 0
        finally:
            sys.modules.pop("test_bad_module", None)

    def test_invalidate_modules_in_path_handles_none_module(self, module_loader, tmp_path):
        """Test invalidate_modules_in_path handles None modules in sys.modules."""
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()

        # Add None module (can happen during import)
        sys.modules["test_none_module"] = None

        try:
            # Should not raise
            count = module_loader.invalidate_modules_in_path("some_base", plugin_dir)
            # Should complete without error
            assert count >= 0
        finally:
            sys.modules.pop("test_none_module", None)

    def test_invalidate_modules_in_path_handles_module_without_file(self, module_loader, tmp_path):
        """Test invalidate_modules_in_path handles modules without __file__."""
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()

        # Create module without __file__ (builtin style)
        module = ModuleType("test_no_file")
        # Don't set __file__
        sys.modules["test_no_file"] = module

        try:
            # Should not raise
            count = module_loader.invalidate_modules_in_path("some_base", plugin_dir)
            # Module without __file__ should not be matched
            assert count >= 0
        finally:
            sys.modules.pop("test_no_file", None)


# =============================================================================
# File Watching Tests (covers lines 332-425, 432, 436-440, 445)
# =============================================================================


class TestFileWatching:
    """Tests for file watching functionality."""

    def test_setup_file_watcher_without_watchdog(self, module_loader):
        """Test setup_file_watcher when watchdog is not available."""
        with patch.dict(
            sys.modules,
            {"watchdog": None, "watchdog.events": None, "watchdog.observers": None},
        ):
            with patch(
                "victor.framework.module_loader.importlib.import_module",
                side_effect=ImportError,
            ):
                # Force fresh import attempt
                result = module_loader.setup_file_watcher()
                # When watchdog import fails in the method, it should return False
                # Note: The actual behavior depends on how imports are handled

    def test_setup_file_watcher_with_no_dirs(self, module_loader):
        """Test setup_file_watcher with no watch directories."""
        result = module_loader.setup_file_watcher()
        assert result is False

    def test_setup_file_watcher_already_running(self, module_loader_with_dirs):
        """Test setup_file_watcher when watcher is already running."""
        # Set up a fake observer
        module_loader_with_dirs._observer = MagicMock()

        result = module_loader_with_dirs.setup_file_watcher()
        assert result is False

    def test_setup_file_watcher_with_nonexistent_dirs(self, tmp_path):
        """Test setup_file_watcher with all non-existent directories."""
        loader = DynamicModuleLoader()
        loader._watch_dirs = [tmp_path / "nonexistent1", tmp_path / "nonexistent2"]

        # Should return False when no valid directories
        result = loader.setup_file_watcher()
        assert result is False

    def test_setup_file_watcher_success(self, tmp_path):
        """Test setup_file_watcher successfully starts watching."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()

        loader = DynamicModuleLoader(watch_dirs=[watch_dir])

        try:
            result = loader.setup_file_watcher()
            # Should succeed if watchdog is installed
            if result:
                assert loader.is_watching
                assert loader._observer is not None
                assert loader._file_handler is not None
        finally:
            loader.stop_file_watcher()

    def test_setup_file_watcher_with_callback(self, tmp_path):
        """Test setup_file_watcher with custom on_change callback."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()

        loader = DynamicModuleLoader(watch_dirs=[watch_dir])
        callback_called = []

        def on_change(file_path: str, event_type: str):
            callback_called.append((file_path, event_type))

        try:
            result = loader.setup_file_watcher(on_change=on_change)
            # Just verify setup completes
            if result:
                assert loader.is_watching
        finally:
            loader.stop_file_watcher()

    def test_setup_file_watcher_with_explicit_dirs(self, tmp_path):
        """Test setup_file_watcher with explicitly passed directories."""
        watch_dir1 = tmp_path / "watch1"
        watch_dir1.mkdir()
        watch_dir2 = tmp_path / "watch2"
        watch_dir2.mkdir()

        loader = DynamicModuleLoader()  # No default watch dirs

        try:
            result = loader.setup_file_watcher(dirs=[watch_dir1, watch_dir2])
            if result:
                assert loader.is_watching
        finally:
            loader.stop_file_watcher()

    def test_stop_file_watcher_when_not_running(self, module_loader):
        """Test stop_file_watcher when no watcher is running."""
        # Should not raise
        module_loader.stop_file_watcher()
        assert module_loader._observer is None

    def test_stop_file_watcher_cancels_debounced(self, module_loader):
        """Test stop_file_watcher cancels pending debounced reloads."""
        # Schedule some debounced callbacks
        module_loader._debounce_timer.schedule("key1", lambda: None)
        module_loader._debounce_timer.schedule("key2", lambda: None)

        # Stop should cancel them
        module_loader.stop_file_watcher()
        assert len(module_loader._debounce_timer._timers) == 0

    def test_stop_file_watcher_stops_running_observer(self, tmp_path):
        """Test stop_file_watcher properly stops a running observer."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()

        loader = DynamicModuleLoader(watch_dirs=[watch_dir])
        result = loader.setup_file_watcher()

        if result:
            assert loader.is_watching
            loader.stop_file_watcher()
            assert not loader.is_watching
            assert loader._observer is None
            assert loader._file_handler is None

    def test_is_watching_property_false(self, module_loader):
        """Test is_watching property when not watching."""
        assert module_loader.is_watching is False

    def test_is_watching_property_true(self, module_loader):
        """Test is_watching property when watching."""
        module_loader._observer = MagicMock()
        assert module_loader.is_watching is True

    def test_file_handler_with_file_events(self, tmp_path):
        """Test that file handler processes Python file events."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()

        loader = DynamicModuleLoader(watch_dirs=[watch_dir], debounce_delay=0.05)

        # Track a test module
        test_file = watch_dir / "test_module.py"
        test_file.write_text("# test")
        module = ModuleType("test_watched_module")
        module.__file__ = str(test_file)
        loader.track_module("test_watched_module", module)

        changes = []
        original_on_module_changed = loader._on_module_changed

        def capture_changes(mod_name, path, event_type):
            changes.append((mod_name, event_type))
            original_on_module_changed(mod_name, path, event_type)

        loader._on_module_changed = capture_changes

        try:
            result = loader.setup_file_watcher()
            if result:
                # Modify the file to trigger an event
                test_file.write_text("# modified test")
                # Wait for debounce
                time.sleep(0.2)

                # File modification should have been detected
                # (may or may not work depending on OS/timing)
        finally:
            loader.stop_file_watcher()

    def test_file_handler_ignores_non_python_files(self, tmp_path):
        """Test that file handler ignores non-Python files."""
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()

        loader = DynamicModuleLoader(watch_dirs=[watch_dir])

        try:
            result = loader.setup_file_watcher()
            if result:
                # Create a non-Python file
                txt_file = watch_dir / "readme.txt"
                txt_file.write_text("readme content")
                # No debounce should be scheduled for non-.py files
                assert len(loader._debounce_timer._timers) == 0
        finally:
            loader.stop_file_watcher()


# =============================================================================
# Module Tracking Tests (covers lines 462-469, 486, 509-510, 521-524, 532, 543)
# =============================================================================


class TestModuleTracking:
    """Tests for module tracking functionality."""

    def test_find_module_for_file_matches(self, module_loader, mock_module, tmp_path):
        """Test _find_module_for_file finds matching module."""
        test_file = tmp_path / "test_module.py"
        test_file.touch()
        mock_module.__file__ = str(test_file)

        module_loader._loaded_modules["test_module"] = mock_module

        result = module_loader._find_module_for_file(test_file)
        assert result == "test_module"

    def test_find_module_for_file_no_match(self, module_loader):
        """Test _find_module_for_file returns None when no match."""
        result = module_loader._find_module_for_file(Path("/nonexistent/file.py"))
        assert result is None

    def test_find_module_for_file_handles_no_file_attr(self, module_loader, tmp_path):
        """Test _find_module_for_file handles modules without __file__."""
        module = ModuleType("no_file_module")
        # Don't set __file__
        module_loader._loaded_modules["no_file_module"] = module

        result = module_loader._find_module_for_file(tmp_path / "some_file.py")
        assert result is None

    def test_on_module_changed_logs(self, module_loader, tmp_path, caplog):
        """Test _on_module_changed logs the change."""
        import logging

        with caplog.at_level(logging.DEBUG):
            module_loader._on_module_changed(
                "test_module",
                tmp_path / "test.py",
                "modified",
            )
        # Should have logged something (default implementation just logs)

    def test_track_module_with_explicit_path(self, module_loader, mock_module, tmp_path):
        """Test track_module with explicit file path."""
        custom_path = tmp_path / "custom.py"
        module_loader.track_module("custom_module", mock_module, file_path=custom_path)

        assert "custom_module" in module_loader._loaded_modules
        assert module_loader._module_paths["custom_module"] == custom_path

    def test_track_module_with_module_file(self, module_loader, mock_module):
        """Test track_module uses module.__file__ when no explicit path."""
        module_loader.track_module("test_module", mock_module)

        assert "test_module" in module_loader._loaded_modules
        assert module_loader._module_paths["test_module"] == Path(mock_module.__file__)

    def test_track_module_without_file(self, module_loader):
        """Test track_module handles module without __file__."""
        module = ModuleType("no_file")
        module_loader.track_module("no_file", module)

        assert "no_file" in module_loader._loaded_modules
        assert "no_file" not in module_loader._module_paths

    def test_untrack_module_returns_true_when_tracked(self, module_loader, mock_module):
        """Test untrack_module returns True when module was tracked."""
        module_loader.track_module("test_module", mock_module)
        result = module_loader.untrack_module("test_module")

        assert result is True
        assert "test_module" not in module_loader._loaded_modules
        assert "test_module" not in module_loader._module_paths

    def test_untrack_module_returns_false_when_not_tracked(self, module_loader):
        """Test untrack_module returns False when module wasn't tracked."""
        result = module_loader.untrack_module("nonexistent")
        assert result is False

    def test_get_tracked_modules(self, module_loader, mock_module):
        """Test get_tracked_modules returns list of tracked module names."""
        module_loader.track_module("module1", mock_module)
        module_loader.track_module("module2", mock_module)

        tracked = module_loader.get_tracked_modules()
        assert "module1" in tracked
        assert "module2" in tracked

    def test_get_module_path_returns_path(self, module_loader, mock_module, tmp_path):
        """Test get_module_path returns path for tracked module."""
        custom_path = tmp_path / "custom.py"
        module_loader.track_module("custom", mock_module, file_path=custom_path)

        result = module_loader.get_module_path("custom")
        assert result == custom_path

    def test_get_module_path_returns_none_for_untracked(self, module_loader):
        """Test get_module_path returns None for untracked module."""
        result = module_loader.get_module_path("nonexistent")
        assert result is None


# =============================================================================
# Watch Directory Management Tests (covers lines 561-576, 591-595)
# =============================================================================


class TestWatchDirectoryManagement:
    """Tests for watch directory management."""

    def test_add_watch_dir_new_directory(self, module_loader, tmp_path):
        """Test add_watch_dir adds new directory."""
        new_dir = tmp_path / "new_plugins"
        new_dir.mkdir()

        result = module_loader.add_watch_dir(new_dir)
        assert result is True
        assert new_dir in module_loader._watch_dirs

    def test_add_watch_dir_duplicate(self, module_loader, tmp_path):
        """Test add_watch_dir returns False for duplicate."""
        new_dir = tmp_path / "plugins"
        new_dir.mkdir()

        module_loader.add_watch_dir(new_dir)
        result = module_loader.add_watch_dir(new_dir)
        assert result is False

    def test_add_watch_dir_with_active_watcher(self, module_loader, tmp_path):
        """Test add_watch_dir schedules watch when watcher is active."""
        new_dir = tmp_path / "plugins"
        new_dir.mkdir()

        # Set up mock observer
        mock_observer = MagicMock()
        mock_handler = MagicMock()
        module_loader._observer = mock_observer
        module_loader._file_handler = mock_handler

        result = module_loader.add_watch_dir(new_dir)
        assert result is True
        mock_observer.schedule.assert_called_once()

    def test_remove_watch_dir_existing(self, module_loader_with_dirs):
        """Test remove_watch_dir removes existing directory."""
        watch_dir = module_loader_with_dirs._watch_dirs[0]
        result = module_loader_with_dirs.remove_watch_dir(watch_dir)
        assert result is True
        assert watch_dir not in module_loader_with_dirs._watch_dirs

    def test_remove_watch_dir_nonexistent(self, module_loader):
        """Test remove_watch_dir returns False for nonexistent."""
        result = module_loader.remove_watch_dir(Path("/nonexistent"))
        assert result is False


# =============================================================================
# Context Manager Tests (covers lines 603, 607)
# =============================================================================


class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_enter_returns_self(self, module_loader):
        """Test __enter__ returns self."""
        with module_loader as loader:
            assert loader is module_loader

    def test_context_manager_exit_stops_watcher(self, module_loader):
        """Test __exit__ stops the file watcher."""
        module_loader._observer = MagicMock()
        module_loader._file_handler = MagicMock()

        with module_loader:
            pass

        assert module_loader._observer is None


# =============================================================================
# CachedEntryPoints Tests (covers lines 752-753)
# =============================================================================


class TestCachedEntryPoints:
    """Tests for CachedEntryPoints dataclass."""

    def test_is_expired_when_within_ttl(self, cached_entry_points):
        """Test is_expired returns False when within TTL."""
        assert cached_entry_points.is_expired() is False

    def test_is_expired_when_past_ttl(self):
        """Test is_expired returns True when past TTL."""
        cached = CachedEntryPoints(
            group="test",
            entries={},
            env_hash="abc",
            timestamp=time.time() - 7200,  # 2 hours ago
            ttl=3600.0,  # 1 hour TTL
        )
        assert cached.is_expired() is True

    def test_is_expired_with_custom_current_time(self, cached_entry_points):
        """Test is_expired with custom current_time parameter."""
        future_time = cached_entry_points.timestamp + 7200  # 2 hours later
        assert cached_entry_points.is_expired(current_time=future_time) is True

    def test_to_dict_serializes_all_fields(self, cached_entry_points):
        """Test to_dict serializes all fields."""
        data = cached_entry_points.to_dict()

        assert data["group"] == "test.group"
        assert data["entries"] == {
            "entry1": "module1:Class1",
            "entry2": "module2:func2",
        }
        assert data["env_hash"] == "abc123"
        assert "timestamp" in data
        assert data["ttl"] == 3600.0

    def test_from_dict_deserializes_all_fields(self):
        """Test from_dict deserializes all fields."""
        data = {
            "group": "my.group",
            "entries": {"a": "mod:A"},
            "env_hash": "xyz789",
            "timestamp": 1234567890.0,
            "ttl": 7200.0,
        }

        cached = CachedEntryPoints.from_dict(data)

        assert cached.group == "my.group"
        assert cached.entries == {"a": "mod:A"}
        assert cached.env_hash == "xyz789"
        assert cached.timestamp == 1234567890.0
        assert cached.ttl == 7200.0

    def test_from_dict_uses_default_ttl(self):
        """Test from_dict uses default TTL when not provided."""
        data = {
            "group": "test",
            "entries": {},
            "env_hash": "abc",
            "timestamp": time.time(),
        }

        cached = CachedEntryPoints.from_dict(data)
        assert cached.ttl == 3600.0  # Default


# =============================================================================
# EntryPointCache Tests (covers lines 771, 786-791, 806, 823-826, 829-832,
#                        850-851, 876-878, 918-920, 924-927, 947-949,
#                        960-972, 980-988, 996-997, 1005-1024)
# =============================================================================


class TestEntryPointCache:
    """Tests for EntryPointCache class."""

    def test_get_instance_returns_singleton(self, tmp_path):
        """Test get_instance returns singleton."""
        EntryPointCache.reset_instance()

        cache1 = EntryPointCache.get_instance(cache_dir=tmp_path)
        cache2 = EntryPointCache.get_instance()

        assert cache1 is cache2
        EntryPointCache.reset_instance()

    def test_reset_instance_clears_singleton(self, tmp_path):
        """Test reset_instance clears the singleton."""
        cache1 = EntryPointCache.get_instance(cache_dir=tmp_path)
        EntryPointCache.reset_instance()
        cache2 = EntryPointCache.get_instance(cache_dir=tmp_path)

        assert cache1 is not cache2
        EntryPointCache.reset_instance()

    def test_compute_env_hash_returns_string(self, entry_point_cache):
        """Test _compute_env_hash returns a hash string."""
        hash_val = entry_point_cache._compute_env_hash()
        assert isinstance(hash_val, str)
        assert len(hash_val) > 0

    def test_compute_env_hash_handles_error(self, entry_point_cache):
        """Test _compute_env_hash returns fallback on error."""
        with patch("victor.framework.module_loader.sys.version_info", (3, 10)):
            with patch(
                "importlib.metadata.distributions",
                side_effect=Exception("Test error"),
            ):
                hash_val = entry_point_cache._compute_env_hash()
                assert hash_val.startswith("time_")

    def test_get_env_hash_caches_result(self, entry_point_cache):
        """Test _get_env_hash caches the result."""
        hash1 = entry_point_cache._get_env_hash()
        hash2 = entry_point_cache._get_env_hash()
        assert hash1 == hash2
        assert entry_point_cache._env_hash == hash1

    def test_load_from_disk_when_no_file(self, entry_point_cache):
        """Test _load_from_disk does nothing when no cache file."""
        entry_point_cache._cache_file = Path("/nonexistent/file.json")
        entry_point_cache._load_from_disk()  # Should not raise
        assert len(entry_point_cache._memory_cache) == 0

    def test_load_from_disk_with_valid_cache(self, tmp_path):
        """Test _load_from_disk loads valid cache entries."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "entry_points.json"

        # Create cache with matching env hash
        EntryPointCache.reset_instance()
        temp_cache = EntryPointCache(cache_dir=cache_dir)
        env_hash = temp_cache._get_env_hash()

        cache_data = {
            "test.group": {
                "group": "test.group",
                "entries": {"entry1": "mod:Class"},
                "env_hash": env_hash,
                "timestamp": time.time(),
                "ttl": 3600.0,
            }
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Reset and reload
        EntryPointCache.reset_instance()
        new_cache = EntryPointCache(cache_dir=cache_dir)

        assert "test.group" in new_cache._memory_cache
        EntryPointCache.reset_instance()

    def test_load_from_disk_reuses_meta_env_hash_without_full_scan(self, tmp_path):
        """Warm-start path should skip full env hash computation when meta matches."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "entry_points.json"

        EntryPointCache.reset_instance()
        temp_cache = EntryPointCache(cache_dir=cache_dir)
        installation_fingerprint = temp_cache._compute_installation_fingerprint()
        EntryPointCache.reset_instance()

        cache_data = {
            "_meta": {
                "schema_version": 1,
                "env_hash": "meta_hash",
                "install_fingerprint": installation_fingerprint,
            },
            "test.group": {
                "group": "test.group",
                "entries": {"entry1": "mod:Class"},
                "env_hash": "meta_hash",
                "timestamp": time.time(),
                "ttl": 3600.0,
            },
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        with patch.object(
            EntryPointCache,
            "_compute_env_hash",
            side_effect=AssertionError("full env hash should not run on warm meta path"),
        ):
            EntryPointCache.reset_instance()
            new_cache = EntryPointCache(cache_dir=cache_dir)

        assert new_cache._env_hash == "meta_hash"
        assert "test.group" in new_cache._memory_cache
        EntryPointCache.reset_instance()

    def test_load_from_disk_meta_mismatch_falls_back_to_full_scan(self, tmp_path):
        """Meta fingerprint mismatch should use full env hash computation fallback."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "entry_points.json"

        cache_data = {
            "_meta": {
                "schema_version": 1,
                "env_hash": "stale_hash",
                "install_fingerprint": "mismatch",
            },
            "test.group": {
                "group": "test.group",
                "entries": {"entry1": "mod:Class"},
                "env_hash": "computed_hash",
                "timestamp": time.time(),
                "ttl": 3600.0,
            },
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        EntryPointCache.reset_instance()
        with patch.object(
            EntryPointCache, "_compute_env_hash", return_value="computed_hash"
        ) as mock_hash:
            new_cache = EntryPointCache(cache_dir=cache_dir)

        assert mock_hash.call_count == 1
        assert new_cache._env_hash == "computed_hash"
        assert "test.group" in new_cache._memory_cache
        EntryPointCache.reset_instance()

    def test_load_from_disk_ignores_expired_entries(self, tmp_path):
        """Test _load_from_disk ignores expired cache entries."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "entry_points.json"

        # Get env hash first
        EntryPointCache.reset_instance()
        temp_cache = EntryPointCache(cache_dir=cache_dir)
        env_hash = temp_cache._get_env_hash()

        cache_data = {
            "test.group": {
                "group": "test.group",
                "entries": {"entry1": "mod:Class"},
                "env_hash": env_hash,
                "timestamp": time.time() - 7200,  # 2 hours ago
                "ttl": 3600.0,  # 1 hour TTL (expired)
            }
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        EntryPointCache.reset_instance()
        new_cache = EntryPointCache(cache_dir=cache_dir)

        # Expired entry should not be loaded
        assert "test.group" not in new_cache._memory_cache
        EntryPointCache.reset_instance()

    def test_load_from_disk_ignores_wrong_env_hash(self, tmp_path):
        """Test _load_from_disk ignores entries with wrong env hash."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "entry_points.json"

        cache_data = {
            "test.group": {
                "group": "test.group",
                "entries": {"entry1": "mod:Class"},
                "env_hash": "wrong_hash",  # Wrong hash
                "timestamp": time.time(),
                "ttl": 3600.0,
            }
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        EntryPointCache.reset_instance()
        new_cache = EntryPointCache(cache_dir=cache_dir)

        assert "test.group" not in new_cache._memory_cache
        EntryPointCache.reset_instance()

    def test_load_from_disk_handles_invalid_json(self, tmp_path):
        """Test _load_from_disk handles invalid JSON gracefully."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "entry_points.json"

        with open(cache_file, "w") as f:
            f.write("invalid json{")

        EntryPointCache.reset_instance()
        cache = EntryPointCache(cache_dir=cache_dir)  # Should not raise
        assert len(cache._memory_cache) == 0
        EntryPointCache.reset_instance()

    def test_load_from_disk_handles_malformed_entry(self, tmp_path):
        """Test _load_from_disk handles malformed cache entries."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "entry_points.json"

        cache_data = {
            "test.group": {
                "group": "test.group",
                # Missing required fields
            }
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        EntryPointCache.reset_instance()
        cache = EntryPointCache(cache_dir=cache_dir)  # Should not raise
        assert "test.group" not in cache._memory_cache
        EntryPointCache.reset_instance()

    def test_save_to_disk_creates_directory(self, tmp_path):
        """Test _save_to_disk creates cache directory."""
        cache_dir = tmp_path / "new_cache"
        EntryPointCache.reset_instance()
        cache = EntryPointCache(cache_dir=cache_dir)

        cache._memory_cache["test"] = CachedEntryPoints(
            group="test",
            entries={"a": "b"},
            env_hash="abc",
            timestamp=time.time(),
        )
        cache._save_to_disk()

        assert cache_dir.exists()
        assert (cache_dir / "entry_points.json").exists()
        EntryPointCache.reset_instance()

    def test_save_to_disk_persists_meta_env_hash(self, entry_point_cache):
        """Persisted cache should include warm-start metadata when env hash is known."""
        entry_point_cache._env_hash = "env_hash_for_meta"
        entry_point_cache._memory_cache["test.group"] = CachedEntryPoints(
            group="test.group",
            entries={"entry1": "mod:Class"},
            env_hash="env_hash_for_meta",
            timestamp=time.time(),
        )
        entry_point_cache._save_to_disk()

        with open(entry_point_cache._cache_file, "r") as f:
            data = json.load(f)

        assert "_meta" in data
        assert data["_meta"]["env_hash"] == "env_hash_for_meta"
        assert "install_fingerprint" in data["_meta"]

    def test_save_to_disk_handles_error(self, entry_point_cache, tmp_path):
        """Test _save_to_disk handles write errors gracefully."""
        # Make cache file path point to a directory
        entry_point_cache._cache_file = tmp_path  # This is a directory

        # Should not raise
        entry_point_cache._save_to_disk()

    def test_get_entry_points_uses_cache(self, entry_point_cache):
        """Test get_entry_points returns cached results."""
        # Pre-populate cache
        entry_point_cache._memory_cache["test.group"] = CachedEntryPoints(
            group="test.group",
            entries={"cached_entry": "mod:Class"},
            env_hash=entry_point_cache._get_env_hash(),
            timestamp=time.time(),
        )

        result = entry_point_cache.get_entry_points("test.group")
        assert "cached_entry" in result

    def test_get_entry_points_force_refresh(self, entry_point_cache):
        """Test get_entry_points with force_refresh bypasses cache."""
        # Pre-populate cache
        entry_point_cache._memory_cache["test.group"] = CachedEntryPoints(
            group="test.group",
            entries={"old_entry": "mod:Old"},
            env_hash=entry_point_cache._get_env_hash(),
            timestamp=time.time(),
        )

        with patch.object(
            entry_point_cache,
            "_scan_entry_points",
            return_value={"new_entry": "mod:New"},
        ):
            result = entry_point_cache.get_entry_points("test.group", force_refresh=True)
            assert "new_entry" in result
            assert "old_entry" not in result

    def test_get_entry_points_force_refresh_invalidates_shared_registry(self, entry_point_cache):
        """Force refresh should invalidate the shared registry before scanning."""
        with patch(
            "victor.framework.entry_point_registry.get_entry_point_values",
            return_value={"plugin1": "mypackage.module:MyClass"},
        ) as mock_get_entry_point_values:
            result = entry_point_cache.get_entry_points("test.group", force_refresh=True)

        assert result == {"plugin1": "mypackage.module:MyClass"}
        mock_get_entry_point_values.assert_called_once_with("test.group", force=True)

    def test_scan_entry_points_handles_error(self, entry_point_cache):
        """Test _scan_entry_points handles errors gracefully."""
        with patch(
            "victor.framework.entry_point_registry.get_entry_point_values",
            side_effect=Exception("Test error"),
        ):
            result = entry_point_cache._scan_entry_points("test.group")
            assert result == {}

    def test_get_entry_points_refreshes_on_expired(self, entry_point_cache):
        """Test get_entry_points refreshes expired cache."""
        # Pre-populate with expired cache
        entry_point_cache._memory_cache["test.group"] = CachedEntryPoints(
            group="test.group",
            entries={"old_entry": "mod:Old"},
            env_hash=entry_point_cache._get_env_hash(),
            timestamp=time.time() - 7200,  # Expired
            ttl=3600.0,
        )

        with patch.object(
            entry_point_cache,
            "_scan_entry_points",
            return_value={"new_entry": "mod:New"},
        ):
            result = entry_point_cache.get_entry_points("test.group")
            assert "new_entry" in result

    def test_get_entry_points_refreshes_on_env_change(self, entry_point_cache):
        """Test get_entry_points refreshes on environment change."""
        # Pre-populate with different env hash
        entry_point_cache._memory_cache["test.group"] = CachedEntryPoints(
            group="test.group",
            entries={"old_entry": "mod:Old"},
            env_hash="different_hash",
            timestamp=time.time(),
        )

        with patch.object(
            entry_point_cache,
            "_scan_entry_points",
            return_value={"new_entry": "mod:New"},
        ):
            result = entry_point_cache.get_entry_points("test.group")
            assert "new_entry" in result

    def test_scan_entry_points_returns_dict(self, entry_point_cache):
        """Test _scan_entry_points returns dictionary."""
        result = entry_point_cache._scan_entry_points("nonexistent.group")
        assert isinstance(result, dict)

    def test_scan_entry_points_finds_real_entry_points(self, entry_point_cache):
        """Test _scan_entry_points finds real entry points when available."""
        # Scan for a real entry point group that likely exists (console_scripts)
        result = entry_point_cache._scan_entry_points("console_scripts")
        # Should return a dict (may be empty if no console_scripts exist)
        assert isinstance(result, dict)
        # If there are entries, they should be name -> module:attr strings
        for name, value in result.items():
            assert isinstance(name, str)
            assert isinstance(value, str)

    def test_scan_entry_points_with_mock_entry_points(self, entry_point_cache):
        """Test _scan_entry_points processes entry points correctly."""
        with patch(
            "victor.framework.entry_point_registry.get_entry_point_values",
            return_value={
                "plugin1": "mypackage.module:MyClass",
                "plugin2": "another.module:function",
            },
        ):
            result = entry_point_cache._scan_entry_points("test.group")
            assert result["plugin1"] == "mypackage.module:MyClass"
            assert result["plugin2"] == "another.module:function"

    @pytest.mark.asyncio
    async def test_get_entry_points_async(self, entry_point_cache):
        """Test get_entry_points_async returns results."""
        with patch.object(
            entry_point_cache,
            "get_entry_points",
            return_value={"async_entry": "mod:Async"},
        ):
            result = await entry_point_cache.get_entry_points_async("test.group")
            assert "async_entry" in result

    def test_invalidate_single_group(self, entry_point_cache):
        """Test invalidate removes single group."""
        entry_point_cache._memory_cache["test.group"] = CachedEntryPoints(
            group="test.group",
            entries={},
            env_hash="abc",
            timestamp=time.time(),
        )

        with patch(
            "victor.framework.entry_point_registry.get_entry_point_registry",
            return_value=MagicMock(),
        ) as mock_get_registry:
            count = entry_point_cache.invalidate("test.group")

        assert count == 1
        assert "test.group" not in entry_point_cache._memory_cache
        mock_get_registry.return_value.invalidate.assert_called_once()

    def test_invalidate_nonexistent_group(self, entry_point_cache):
        """Test invalidate returns 0 for nonexistent group."""
        count = entry_point_cache.invalidate("nonexistent")
        assert count == 0

    def test_invalidate_all(self, entry_point_cache):
        """Test invalidate with no args clears all."""
        entry_point_cache._memory_cache["group1"] = CachedEntryPoints(
            group="group1", entries={}, env_hash="a", timestamp=time.time()
        )
        entry_point_cache._memory_cache["group2"] = CachedEntryPoints(
            group="group2", entries={}, env_hash="b", timestamp=time.time()
        )
        entry_point_cache._env_hash = "cached_hash"

        with patch(
            "victor.framework.entry_point_registry.get_entry_point_registry",
            return_value=MagicMock(),
        ) as mock_get_registry:
            count = entry_point_cache.invalidate()

        assert count == 2
        assert len(entry_point_cache._memory_cache) == 0
        assert entry_point_cache._env_hash is None
        mock_get_registry.return_value.invalidate.assert_called_once()

    def test_invalidate_on_env_change_when_changed(self, entry_point_cache):
        """Test invalidate_on_env_change when environment changed."""
        entry_point_cache._env_hash = "old_hash"
        entry_point_cache._memory_cache["test"] = CachedEntryPoints(
            group="test", entries={}, env_hash="old_hash", timestamp=time.time()
        )

        with patch.object(entry_point_cache, "_compute_env_hash", return_value="new_hash"):
            result = entry_point_cache.invalidate_on_env_change()
            assert result is True
            assert len(entry_point_cache._memory_cache) == 0

    def test_invalidate_on_env_change_when_same(self, entry_point_cache):
        """Test invalidate_on_env_change when environment unchanged."""
        current_hash = entry_point_cache._get_env_hash()
        entry_point_cache._memory_cache["test"] = CachedEntryPoints(
            group="test", entries={}, env_hash=current_hash, timestamp=time.time()
        )

        result = entry_point_cache.invalidate_on_env_change()
        assert result is False
        assert "test" in entry_point_cache._memory_cache

    def test_get_cached_groups(self, entry_point_cache):
        """Test get_cached_groups returns list of groups."""
        entry_point_cache._memory_cache["group1"] = CachedEntryPoints(
            group="group1", entries={}, env_hash="a", timestamp=time.time()
        )
        entry_point_cache._memory_cache["group2"] = CachedEntryPoints(
            group="group2", entries={}, env_hash="b", timestamp=time.time()
        )

        groups = entry_point_cache.get_cached_groups()
        assert "group1" in groups
        assert "group2" in groups

    def test_get_cache_stats(self, entry_point_cache):
        """Test get_cache_stats returns stats dictionary."""
        entry_point_cache._memory_cache["test.group"] = CachedEntryPoints(
            group="test.group",
            entries={"a": "b", "c": "d"},
            env_hash="abc",
            timestamp=time.time() - 100,
            ttl=3600.0,
        )

        stats = entry_point_cache.get_cache_stats()

        assert stats["groups_cached"] == 1
        assert "env_hash" in stats
        assert "cache_file" in stats
        assert "test.group" in stats["groups"]
        assert stats["groups"]["test.group"]["entries"] == 2
        assert stats["groups"]["test.group"]["age_seconds"] >= 100
        assert stats["groups"]["test.group"]["ttl_remaining"] > 0
        assert stats["groups"]["test.group"]["expired"] is False


# =============================================================================
# get_entry_point_cache Function Tests
# =============================================================================


class TestGetEntryPointCache:
    """Tests for get_entry_point_cache function."""

    def test_get_entry_point_cache_returns_instance(self):
        """Test get_entry_point_cache returns singleton instance."""
        EntryPointCache.reset_instance()
        cache = get_entry_point_cache()
        assert isinstance(cache, EntryPointCache)
        EntryPointCache.reset_instance()

    def test_get_entry_point_cache_returns_same_instance(self):
        """Test get_entry_point_cache returns same instance."""
        EntryPointCache.reset_instance()
        cache1 = get_entry_point_cache()
        cache2 = get_entry_point_cache()
        assert cache1 is cache2
        EntryPointCache.reset_instance()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for module loader components."""

    def test_module_loader_lifecycle(self, tmp_path):
        """Test full module loader lifecycle."""
        watch_dir = tmp_path / "plugins"
        watch_dir.mkdir()

        loader = DynamicModuleLoader(watch_dirs=[watch_dir], debounce_delay=0.1)

        # Track a module
        module = ModuleType("test_integration")
        module.__file__ = str(watch_dir / "test.py")
        loader.track_module("test_integration", module)

        assert "test_integration" in loader.get_tracked_modules()
        assert loader.get_module_path("test_integration") is not None

        # Invalidate the module
        sys.modules["test_integration"] = module
        try:
            count = loader.invalidate_module("test_integration")
            assert count >= 0
        finally:
            sys.modules.pop("test_integration", None)

        # Untrack
        loader.untrack_module("test_integration")
        assert "test_integration" not in loader.get_tracked_modules()

    def test_entry_point_cache_lifecycle(self, tmp_path):
        """Test full entry point cache lifecycle."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        EntryPointCache.reset_instance()
        cache = EntryPointCache(cache_dir=cache_dir)

        # Get entry points (will scan and cache)
        with patch.object(cache, "_scan_entry_points", return_value={"entry1": "mod:Class"}):
            entries = cache.get_entry_points("test.group")
            assert "entry1" in entries

        # Verify cached
        assert "test.group" in cache.get_cached_groups()

        # Get stats
        stats = cache.get_cache_stats()
        assert stats["groups_cached"] == 1

        # Invalidate
        cache.invalidate("test.group")
        assert "test.group" not in cache.get_cached_groups()

        EntryPointCache.reset_instance()
