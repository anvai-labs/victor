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

"""Tests for unified cache tool module."""

import pytest
from unittest.mock import MagicMock

from victor.tools.cache_tool import cache, _get_cache_manager


class TestGetCacheManager:
    """Tests for _get_cache_manager function."""

    def test_get_cache_manager_from_context(self):
        """Test getting cache manager from context."""
        mock_manager = MagicMock()
        context = {"cache_manager": mock_manager}
        result = _get_cache_manager(context)
        assert result == mock_manager

    def test_get_cache_manager_without_context(self):
        """Test getting cache manager without context returns None."""
        result = _get_cache_manager(None)
        assert result is None

    def test_get_cache_manager_empty_context(self):
        """Test getting cache manager with empty context returns None."""
        result = _get_cache_manager({})
        assert result is None


class TestCacheStats:
    """Tests for cache stats action."""

    @pytest.mark.asyncio
    async def test_cache_stats_no_manager(self):
        """Test cache stats when no manager is set."""
        result = await cache(action="stats", context=None)
        assert result["success"] is False
        assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_cache_stats_with_manager(self):
        """Test getting cache statistics."""
        mock_manager = MagicMock()
        mock_manager.get_stats.return_value = {
            "memory_hit_rate": 0.75,
            "disk_hit_rate": 0.5,
            "memory_hits": 100,
            "memory_misses": 25,
            "disk_hits": 50,
            "disk_misses": 50,
            "sets": 200,
        }
        context = {"cache_manager": mock_manager}
        result = await cache(action="stats", context=context)
        assert result["success"] is True
        assert "stats" in result
        assert "formatted_report" in result


class TestCacheClear:
    """Tests for cache clear action."""

    @pytest.mark.asyncio
    async def test_cache_clear_no_manager(self):
        """Test cache clear when no manager is set."""
        result = await cache(action="clear", context=None)
        assert result["success"] is False
        assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_cache_clear_all(self):
        """Test clearing all cache."""
        mock_manager = MagicMock()
        mock_manager.clear.return_value = 10
        context = {"cache_manager": mock_manager}
        result = await cache(action="clear", context=context)
        assert result["success"] is True
        mock_manager.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_clear_namespace(self):
        """Test clearing specific namespace."""
        mock_manager = MagicMock()
        mock_manager.clear.return_value = 5
        context = {"cache_manager": mock_manager}
        result = await cache(action="clear", namespace="responses", context=context)
        assert result["success"] is True
        mock_manager.clear.assert_called_once_with("responses")


class TestCacheInfo:
    """Tests for cache info action."""

    @pytest.mark.asyncio
    async def test_cache_info_no_manager(self):
        """Test cache info when no manager is set."""
        result = await cache(action="info", context=None)
        assert result["success"] is False
        assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_cache_info_with_manager(self):
        """Test getting cache configuration info."""
        mock_config = MagicMock()
        mock_config.enable_memory = True
        mock_config.memory_max_size = 1000
        mock_config.memory_ttl = 300
        mock_config.enable_disk = True
        mock_config.disk_max_size = 100 * 1024 * 1024
        mock_config.disk_ttl = 86400 * 7
        mock_config.disk_path = "/tmp/cache"

        mock_manager = MagicMock()
        mock_manager.config = mock_config
        context = {"cache_manager": mock_manager}

        result = await cache(action="info", context=context)
        assert result["success"] is True
        assert "config" in result
        assert "formatted_report" in result


class TestCacheUnknownAction:
    """Tests for unknown action handling."""

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        """Test handling of unknown action."""
        mock_manager = MagicMock()
        context = {"cache_manager": mock_manager}
        result = await cache(action="invalid", context=context)
        assert result["success"] is False
        assert "Unknown action" in result["error"]
