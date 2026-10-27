"""Extracted tool pipeline utilities (from tool_pipeline.py)."""

from __future__ import annotations

import time

import pytest

from victor.agent.tool_pipeline_utils import LRUToolCache, ToolRateLimiter


class TestToolRateLimiter:
    def test_allows_burst_then_rate_limits(self):
        limiter = ToolRateLimiter(rate=10.0, burst=5)
        for _ in range(5):
            assert limiter.acquire() is True
        assert limiter.acquire() is False  # exhausted

    def test_refills_over_time(self):
        limiter = ToolRateLimiter(rate=1000.0, burst=1)
        assert limiter.acquire() is True
        assert limiter.acquire() is False
        time.sleep(0.01)  # 10ms at 1000/s = ~10 tokens
        assert limiter.acquire() is True


class TestLRUToolCache:
    def test_evicts_least_recently_used(self):
        cache = LRUToolCache(max_size=2, ttl_seconds=60.0)
        cache.set("a", {"result": 1})
        cache.set("b", {"result": 2})
        cache.get("a")  # touch a
        cache.set("c", {"result": 3})  # should evict b
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_ttl_expiry(self):
        cache = LRUToolCache(max_size=10, ttl_seconds=0.01)
        cache.set("x", {"result": 1})
        time.sleep(0.02)
        assert cache.get("x") is None
