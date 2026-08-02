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
# See the the License for the specific language governing permissions and
# limitations under the License.

"""Tests for shell command cache."""

import time
from unittest.mock import patch

import pytest

from victor.tools.shell_command_cache import (
    ShellCommandCache,
    get_shell_cache,
    execute_with_cache,
    clear_shell_cache,
    get_shell_cache_stats,
)


class TestShellCommandCache:
    """Test shell command cache."""

    def test_cache_initialization(self):
        """Test cache initializes with correct defaults."""
        cache = ShellCommandCache(default_ttl_minutes=5)
        assert cache._default_ttl.total_seconds() == 300
        assert len(cache._cache) == 0

    def test_cache_get_set(self):
        """Test cache set and get operations."""
        cache = ShellCommandCache(default_ttl_minutes=5)

        # Set a result
        result = (0, "output", "error")
        cache.set("echo hello", result)

        # Get it back
        cached = cache.get("echo hello")
        assert cached == result

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ShellCommandCache(default_ttl_minutes=5)
        result = cache.get("nonexistent command")
        assert result is None

    def test_cache_key_includes_cwd(self):
        """Test cache keys include working directory."""
        cache = ShellCommandCache(default_ttl_minutes=5)

        result = (0, "output", "error")
        cache.set("ls", result, cwd="/tmp")

        # Same cwd should hit
        assert cache.get("ls", cwd="/tmp") == result

        # Different cwd should miss
        assert cache.get("ls", cwd="/home") is None

    def test_cache_expiration(self):
        """Test cache entries expire based on TTL."""
        cache = ShellCommandCache(default_ttl_minutes=0)  # 0 TTL = immediate expiration

        result = (0, "output", "error")
        cache.set("test command", result)

        # Should be expired immediately
        assert cache.get("test command") is None

    def test_cache_clear(self):
        """Test cache can be cleared."""
        cache = ShellCommandCache(default_ttl_minutes=5)

        cache.set("cmd1", (0, "out1", "err1"))
        cache.set("cmd2", (0, "out2", "err2"))
        assert len(cache._cache) == 2

        cache.clear()
        assert len(cache._cache) == 0

    def test_cleanup_expired(self):
        """Test expired entries are cleaned up."""
        cache = ShellCommandCache(default_ttl_minutes=0)

        cache.set("cmd1", (0, "out1", "err1"))
        cache.set("cmd2", (0, "out2", "err2"))

        removed = cache.cleanup_expired()
        assert removed == 2
        assert len(cache._cache) == 0

    def test_command_specific_ttl(self):
        """Test different commands have different TTLs."""
        cache = ShellCommandCache(default_ttl_minutes=5)

        # gh command should have 10 minute TTL
        gh_ttl = cache._get_ttl_for_command("gh run view 123")
        assert gh_ttl.total_seconds() == 600  # 10 minutes

        # ls command should have 1 minute TTL
        ls_ttl = cache._get_ttl_for_command("ls -la")
        assert ls_ttl.total_seconds() == 60  # 1 minute

        # Unknown command should use default
        unknown_ttl = cache._get_ttl_for_command("unknown command")
        assert unknown_ttl.total_seconds() == 300  # 5 minutes

    def test_global_cache_singleton(self):
        """Test global cache instance."""
        cache1 = get_shell_cache()
        cache2 = get_shell_cache()
        assert cache1 is cache2

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ShellCommandCache(default_ttl_minutes=5)

        cache.set("cmd1", (0, "out1", "err1"))
        stats = cache.get_stats()

        assert stats["entries"] == 1
        assert "default_ttl_minutes" in stats
        assert "command_ttls" in stats


class TestExecuteWithCache:
    """Test execute_with_cache function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_shell_cache()

    @patch("victor.tools.shell_command_cache.subprocess.run")
    def test_execute_with_cache_miss(self, mock_run):
        """Test execution when cache miss."""
        # Create a proper mock result
        mock_result = type(
            "CompletedProcess", (), {"returncode": 0, "stdout": "output", "stderr": ""}
        )()
        mock_run.return_value = mock_result

        returncode, stdout, stderr = execute_with_cache("echo test")

        assert returncode == 0
        assert stdout == "output"
        assert mock_run.call_count == 1

    @patch("victor.tools.shell_command_cache.subprocess.run")
    def test_execute_with_cache_hit(self, mock_run):
        """Test execution when cache hit."""
        # Create a proper mock result
        mock_result = type(
            "CompletedProcess", (), {"returncode": 0, "stdout": "output", "stderr": ""}
        )()
        mock_run.return_value = mock_result

        # First call - cache miss
        returncode, stdout, stderr = execute_with_cache("echo test")

        # Second call - cache hit (should not call subprocess again)
        returncode, stdout, stderr = execute_with_cache("echo test")

        assert mock_run.call_count == 1  # Called only once

    @patch("victor.tools.shell_command_cache.subprocess.run")
    def test_execute_with_cache_disabled(self, mock_run):
        """Test execution with caching disabled."""
        # Create a proper mock result
        mock_result = type(
            "CompletedProcess", (), {"returncode": 0, "stdout": "output", "stderr": ""}
        )()
        mock_run.return_value = mock_result

        # Execute with cache disabled
        execute_with_cache("echo test", use_cache=False)
        execute_with_cache("echo test", use_cache=False)

        # Should call subprocess both times
        assert mock_run.call_count == 2


class TestGlobalCacheFunctions:
    """Test global cache utility functions."""

    def test_clear_cache(self):
        """Test clearing global cache."""
        cache = get_shell_cache()
        cache.set("test", (0, "", ""))

        clear_shell_cache()

        assert len(cache._cache) == 0

    def test_get_cache_stats(self):
        """Test getting cache stats."""
        cache = get_shell_cache()
        cache.set("test", (0, "", ""))

        stats = get_shell_cache_stats()

        assert stats["entries"] >= 1
        assert "default_ttl_minutes" in stats
