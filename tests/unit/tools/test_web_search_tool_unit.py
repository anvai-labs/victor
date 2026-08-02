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

"""Tests for web_search_tool module."""

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.tools.base import ToolConfig
from victor.tools import web_search_tool as web_search_tool_module
from victor.tools.web_search_tool import (
    _parse_ddg_results,
    _get_web_config,
    web_fetch,
    web_search,
    RateLimiter,
    _get_rate_limiter,
)


class TestGetWebConfig:
    """Tests for _get_web_config function."""

    def test_config_from_tool_config(self):
        """Test getting config from ToolConfig context."""
        mock_provider = MagicMock()
        tool_config = ToolConfig(
            provider=mock_provider,
            model="test-model",
            web_fetch_top=5,
            web_fetch_pool=10,
            max_content_length=8000,
            generic_result_cache_enabled=True,
            generic_result_cache_ttl=120,
            http_connection_pool_enabled=True,
            http_connection_pool_max_connections=64,
        )
        context = {"tool_config": tool_config}

        config = _get_web_config(context)

        assert config["provider"] == mock_provider
        assert config["model"] == "test-model"
        assert config["fetch_top"] == 5
        assert config["fetch_pool"] == 10
        assert config["max_content_length"] == 8000
        assert config["generic_result_cache_enabled"] is True
        assert config["generic_result_cache_ttl"] == 120
        assert config["http_connection_pool_enabled"] is True
        assert config["http_connection_pool_max_connections"] == 64

    def test_config_without_context(self):
        """Test getting config without context returns defaults."""
        config = _get_web_config(None)

        assert config["provider"] is None
        assert config["model"] is None
        assert config["fetch_top"] is None
        assert config["fetch_pool"] is None
        assert config["max_content_length"] == 5000
        assert config["generic_result_cache_enabled"] is False
        assert (
            config["http_connection_pool_enabled"] is True
        )  # HTTP pooling enabled by default for performance

    def test_config_with_empty_context(self):
        """Test getting config with empty context returns defaults."""
        config = _get_web_config({})

        assert config["provider"] is None
        assert config["model"] is None


class TestParseDDGResults:
    """Tests for _parse_ddg_results function."""

    def test_parse_empty_html(self):
        """Test parsing empty HTML."""
        results = _parse_ddg_results("<html></html>", max_results=10)
        assert results == []

    def test_parse_valid_results(self):
        """Test parsing valid DuckDuckGo results."""
        html = """
        <html>
        <body>
        <div class="result">
            <a class="result__a" href="https://example.com">Example Title</a>
            <a class="result__snippet">This is a test snippet</a>
        </div>
        <div class="result">
            <a class="result__a" href="https://test.com">Test Title</a>
            <a class="result__snippet">Another snippet</a>
        </div>
        </body>
        </html>
        """
        results = _parse_ddg_results(html, max_results=10)
        assert len(results) == 2
        assert results[0]["title"] == "Example Title"
        assert results[0]["url"] == "https://example.com"

    def test_parse_respects_max_results(self):
        """Test that max_results is respected."""
        html = """
        <html>
        <body>
        <div class="result">
            <a class="result__a" href="https://a.com">A</a>
        </div>
        <div class="result">
            <a class="result__a" href="https://b.com">B</a>
        </div>
        <div class="result">
            <a class="result__a" href="https://c.com">C</a>
        </div>
        </body>
        </html>
        """
        results = _parse_ddg_results(html, max_results=2)
        assert len(results) <= 2

    def test_parse_result_missing_link(self):
        """Test parsing result without link."""
        html = """
        <html>
        <body>
        <div class="result">
            <span>No link here</span>
        </div>
        </body>
        </html>
        """
        results = _parse_ddg_results(html, max_results=10)
        assert results == []


class TestRuntimeInfraIntegration:
    """Tests for runtime cache/pool integrations in web tools."""

    @pytest.mark.asyncio
    async def test_web_search_passes_pool_enabled_config_to_request_layer(self):
        """web_search should forward pool-enabled config to _request_text."""
        html = """
        <html><body>
        <div class="result">
            <a class="result__a" href="https://example.com">Example</a>
            <a class="result__snippet">Snippet</a>
        </div>
        </body></html>
        """
        tool_config = ToolConfig(http_connection_pool_enabled=True)
        context = {"tool_config": tool_config}

        with patch(
            "victor.tools.web_search_tool._request_text",
            new=AsyncMock(return_value=(200, html)),
        ) as mock_request:
            result = await web_search(query="test query", _exec_ctx=context)

        assert result["success"] is True
        assert mock_request.await_count == 1
        web_config = mock_request.await_args.kwargs["web_config"]
        assert web_config["http_connection_pool_enabled"] is True

    @pytest.mark.asyncio
    async def test_web_fetch_uses_generic_cache_on_repeated_calls(self, monkeypatch):
        """web_fetch should serve repeated identical requests from GenericResultCache."""
        monkeypatch.setattr(web_search_tool_module, "_GENERIC_WEB_CACHE", None)
        html = "<html><body><main>" + ("content " * 30) + "</main></body></html>"
        url = f"https://example.com/{uuid.uuid4().hex}"
        tool_config = ToolConfig(generic_result_cache_enabled=True, generic_result_cache_ttl=120)
        context = {"tool_config": tool_config}

        with patch(
            "victor.tools.web_search_tool._request_text",
            new=AsyncMock(return_value=(200, html)),
        ) as mock_request:
            first = await web_fetch(url, _exec_ctx=context)

        assert first["success"] is True
        assert mock_request.await_count == 1

        with patch(
            "victor.tools.web_search_tool._request_text",
            new=AsyncMock(side_effect=AssertionError("network should not be called")),
        ):
            second = await web_fetch(url, _exec_ctx=context)

        assert second["success"] is True
        assert second["cached"] is True

    @pytest.mark.asyncio
    async def test_web_fetch_browser_render_skips_http_request(self):
        with (
            patch(
                "victor.tools.web_search_tool._request_text",
                new=AsyncMock(side_effect=AssertionError("http fetch should not be called")),
            ),
            patch(
                "victor.tools.web_search_tool._render_page_text",
                new=AsyncMock(
                    return_value={
                        "success": True,
                        "content": "Rendered page text",
                        "html_length": 1234,
                    }
                ),
            ) as mock_render,
        ):
            result = await web_fetch("https://example.com/app", render="browser")

        assert result["success"] is True
        assert result["content"] == "Rendered page text"
        assert result["render"] == "browser"
        assert result["html_length"] == 1234
        mock_render.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_web_fetch_auto_falls_back_to_browser_for_low_text_pages(self):
        short_html = "<html><body><div id='root'></div><script>app()</script></body></html>"
        with (
            patch(
                "victor.tools.web_search_tool._request_text",
                new=AsyncMock(return_value=(200, short_html)),
            ),
            patch(
                "victor.tools.web_search_tool._render_page_text",
                new=AsyncMock(
                    return_value={
                        "success": True,
                        "content": "Rendered SPA content",
                        "html_length": 4096,
                    }
                ),
            ) as mock_render,
        ):
            result = await web_fetch("https://example.com/spa", render="auto")

        assert result["success"] is True
        assert result["content"] == "Rendered SPA content"
        assert result["render"] == "browser"
        assert result["http_status_code"] == 200
        mock_render.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_web_fetch_http_mode_does_not_browser_fallback(self):
        short_html = "<html><body><div id='root'></div><script>app()</script></body></html>"
        with (
            patch(
                "victor.tools.web_search_tool._request_text",
                new=AsyncMock(return_value=(200, short_html)),
            ),
            patch(
                "victor.tools.web_search_tool._render_page_text",
                new=AsyncMock(side_effect=AssertionError("browser should not be called")),
            ),
        ):
            result = await web_fetch("https://example.com/spa", render="http")

        assert result["success"] is False
        assert "No content" in result["error"]


class TestWebSearchExecContext:
    """Tests that web_search and web_fetch use _exec_ctx properly."""

    def test_web_search_schema_excludes_exec_ctx(self):
        """After rename, _exec_ctx should not appear in web_search schema."""
        props = web_search.Tool.parameters.get("properties", {})
        assert "_exec_ctx" not in props
        # Old 'context' param should also not be there anymore
        assert "context" not in props

    def test_web_fetch_schema_excludes_exec_ctx(self):
        """After rename, _exec_ctx should not appear in web_fetch schema."""
        props = web_fetch.Tool.parameters.get("properties", {})
        assert "_exec_ctx" not in props
        assert "context" not in props
        assert "render" in props

    @pytest.mark.asyncio
    async def test_web_search_receives_exec_context(self):
        """web_search should receive execution context via _exec_ctx."""
        html = "<html><body><div class='result'><a class='result__a' href='https://x.com'>X</a></div></body></html>"

        with patch(
            "victor.tools.web_search_tool._request_text",
            new=AsyncMock(return_value=(200, html)),
        ):
            with patch("victor.tools.web_search_tool._get_web_config") as mock_config:
                mock_config.return_value = {
                    "provider": None,
                    "model": None,
                    "fetch_top": None,
                    "fetch_pool": None,
                    "max_content_length": 5000,
                    "generic_result_cache_enabled": False,
                    "generic_result_cache_ttl": 300,
                    "http_connection_pool_enabled": False,
                    "http_connection_pool_max_connections": 100,
                    "http_connection_pool_max_connections_per_host": 10,
                    "http_connection_pool_connection_timeout": 30,
                    "http_connection_pool_total_timeout": 60,
                }
                ctx = {"tool_config": MagicMock()}
                await web_search(query="test", _exec_ctx=ctx)
                mock_config.assert_called_once_with(ctx)


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_acquire_enforces_rate_limit(self):
        """Test that acquire enforces rate limit."""
        # Create rate limiter: 600 requests per minute = 0.1 second interval
        limiter = RateLimiter(requests_per_minute=600)

        url = "https://example.com"
        start = time.time()

        # Make 2 requests
        await limiter.acquire(url)
        await limiter.acquire(url)

        elapsed = time.time() - start

        # Should have waited at least min_interval (0.1 seconds)
        assert elapsed >= 0.1, f"Expected >= 0.1s wait, got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_acquire_different_hosts(self):
        """Test that different hosts have independent rate limits."""
        limiter = RateLimiter(requests_per_minute=60)  # 1 per second

        start = time.time()

        # Make requests to different hosts - should not wait
        await limiter.acquire("https://host1.com")
        await limiter.acquire("https://host2.com")
        await limiter.acquire("https://host3.com")

        elapsed = time.time() - start

        # Should complete quickly (no waiting between different hosts)
        assert elapsed < 1.0, f"Expected < 1.0s for different hosts, got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_on_rate_limit_error_calculates_backoff(self):
        """Test exponential backoff calculation."""
        limiter = RateLimiter(
            initial_delay=1.0,
            backoff_multiplier=2.0,
            max_delay=10.0,
        )

        url = "https://example.com"

        # First attempt: 1.0s delay
        delay1 = limiter.on_rate_limit_error(url, 0)
        assert delay1 == 1.0

        # Second attempt: 2.0s delay (1.0 * 2)
        delay2 = limiter.on_rate_limit_error(url, 1)
        assert delay2 == 2.0

        # Third attempt: 4.0s delay (2.0 * 2)
        delay3 = limiter.on_rate_limit_error(url, 2)
        assert delay3 == 4.0

        # Fourth attempt: 8.0s delay (4.0 * 2)
        delay4 = limiter.on_rate_limit_error(url, 3)
        assert delay4 == 8.0

        # Fifth attempt: 10.0s delay (capped at max_delay)
        delay5 = limiter.on_rate_limit_error(url, 4)
        assert delay5 == 10.0

    def test_should_retry_within_max_retries(self):
        """Test should_retry returns True within max retries."""
        limiter = RateLimiter(max_retries=5)

        for attempt in range(5):
            assert limiter.should_retry("https://example.com", attempt) is True

        # Max retries exceeded
        assert limiter.should_retry("https://example.com", 5) is False

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """Test reset clears request tracking."""
        limiter = RateLimiter(requests_per_minute=60)

        # Simulate some requests via acquire (proper flow)
        url1 = "https://example.com"
        url2 = "https://other.com"

        await limiter.acquire(url1)
        await limiter.acquire(url2)

        # Verify hosts are tracked
        host1 = limiter._get_host(url1)
        host2 = limiter._get_host(url2)
        assert host1 in limiter._request_times
        assert host2 in limiter._request_times

        # Reset specific host
        limiter.reset(url1)
        assert host1 not in limiter._request_times
        assert host2 in limiter._request_times

        # Reset all
        limiter.reset()
        assert len(limiter._request_times) == 0


class TestGetRateLimiter:
    """Tests for _get_rate_limiter function."""

    def test_returns_singleton(self):
        """Test that _get_rate_limiter returns singleton instance."""
        limiter1 = _get_rate_limiter()
        limiter2 = _get_rate_limiter()

        assert limiter1 is limiter2

    def test_uses_default_parameters(self):
        """Test that default parameters are used."""
        # Reset global rate limiter
        from victor.tools.web_search_tool import _RATE_LIMITER

        web_search_tool_module._RATE_LIMITER = None

        limiter = _get_rate_limiter()

        assert limiter.requests_per_minute == 10
        assert limiter.max_retries == 5
        assert limiter.initial_delay == 1.0
        assert limiter.max_delay == 60.0
        assert limiter.backoff_multiplier == 2.0

    def test_uses_custom_parameters(self):
        """Test that custom parameters are applied."""
        # Reset global rate limiter
        from victor.tools.web_search_tool import _RATE_LIMITER

        web_search_tool_module._RATE_LIMITER = None

        limiter = _get_rate_limiter(
            requests_per_minute=30,
            max_retries=3,
            initial_delay=2.0,
            max_delay=30.0,
            backoff_multiplier=3.0,
        )

        assert limiter.requests_per_minute == 30
        assert limiter.max_retries == 3
        assert limiter.initial_delay == 2.0
        assert limiter.max_delay == 30.0
        assert limiter.backoff_multiplier == 3.0
