"""Tests for ProviderPool deduplication and runtime wiring."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.providers.factory import ProviderPool


class TestProviderPoolDeduplication:
    """Verify only one ProviderPool class exists after dedup."""

    def test_single_provider_pool_class(self):
        """There should be exactly one ProviderPool in factory module."""
        import victor.providers.factory as factory_module

        pool_classes = [
            name
            for name, obj in vars(factory_module).items()
            if isinstance(obj, type) and obj.__name__ == "ProviderPool"
        ]
        assert len(pool_classes) == 1

    def test_provider_pool_has_typed_signatures(self):
        """The remaining ProviderPool should have proper type hints."""
        import inspect

        sig = inspect.signature(ProviderPool.acquire)
        params = sig.parameters
        assert "provider_name" in params
        assert "model" in params
        assert "factory" in params


class TestProviderPoolBasic:
    @pytest.fixture
    def pool(self):
        return ProviderPool(max_size_per_key=2)

    @pytest.mark.asyncio
    async def test_acquire_creates_new(self, pool):
        mock_provider = MagicMock()
        factory = AsyncMock(return_value=mock_provider)
        result = await pool.acquire("anthropic", "claude-3", factory)
        assert result is mock_provider
        factory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_acquire_reuses_released(self, pool):
        mock_provider = MagicMock()
        factory = AsyncMock(return_value=mock_provider)
        provider = await pool.acquire("anthropic", "claude-3", factory)
        await pool.release("anthropic", "claude-3", provider)
        reused = await pool.acquire("anthropic", "claude-3", factory)
        assert reused is mock_provider
        assert factory.await_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_release_shuts_down_when_pool_full(self, pool):
        factory = AsyncMock(side_effect=[MagicMock(), MagicMock(), MagicMock()])
        p1 = await pool.acquire("a", "b", factory)
        p2 = await pool.acquire("a", "b", factory)
        p3 = await pool.acquire("a", "b", factory)
        p3.shutdown = AsyncMock()
        await pool.release("a", "b", p1)
        await pool.release("a", "b", p2)
        # Pool is full (max 2), p3 should be shut down
        await pool.release("a", "b", p3)
        p3.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_stats(self, pool):
        stats = pool.get_stats()
        assert "total_available" in stats
        assert "total_in_use" in stats

    @pytest.mark.asyncio
    async def test_shutdown_clears_pool(self, pool):
        mock_provider = MagicMock()
        mock_provider.shutdown = AsyncMock()
        factory = AsyncMock(return_value=mock_provider)
        await pool.acquire("a", "b", factory)
        await pool.release("a", "b", mock_provider)
        await pool.shutdown()
        stats = pool.get_stats()
        assert stats["total_available"] == 0
