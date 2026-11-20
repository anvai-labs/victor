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

"""Tests for web_search provider dispatch over the key-gated adapters (FOU-A5).

No real network: respx mocks every endpoint, including the DuckDuckGo
fallback path.
"""

import httpx
import pytest
import respx

from victor.tools.web_search_tool import web_search

TAVILY_URL = "https://api.tavily.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
SERPER_URL = "https://google.serper.dev/search"
DDG_HTML_URL = "https://html.duckduckgo.com/html/"

_EMPTY_DDG_HTML = "<html><body><div class='results'></div></body></html>"


@pytest.fixture(autouse=True)
def _clear_web_caches(monkeypatch):
    """Isolate tests from the generic result cache and rate limiter, and
    force the httpx request path.

    ``_request_text`` prefers the aiohttp connection pool when enabled (the
    shipped default config enables it), and respx cannot intercept aiohttp —
    a pool-enabled run would leak REAL network requests from these tests and
    return live DuckDuckGo results instead of the mocks (seen on CI:
    result_count 5 from the live endpoint where the mock provides 1). Forcing
    the pool off keeps every request on the respx-intercepted httpx path.
    """
    from victor.tools import web_search_tool as wst

    monkeypatch.setattr(
        wst,
        "_DEFAULT_WEB_CONFIG",
        {**wst._DEFAULT_WEB_CONFIG, "http_connection_pool_enabled": False},
    )
    wst._GENERIC_WEB_CACHE = None
    wst._RATE_LIMITER = None
    yield
    wst._GENERIC_WEB_CACHE = None
    wst._RATE_LIMITER = None


class TestExplicitProviderDispatch:
    @respx.mock
    async def test_tavily_provider_returns_structured_results(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-k")
        respx.post(TAVILY_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Tavily hit",
                            "url": "https://example.com/t",
                            "content": "Tavily snippet text.",
                        }
                    ]
                },
            )
        )

        result = await web_search(query="test", provider="tavily")

        assert result["success"] is True
        assert result["provider"] == "tavily"
        assert result["result_count"] == 1
        assert "https://example.com/t" in result["results"]
        assert result["structured_results"][0]["url"] == "https://example.com/t"

    @respx.mock
    async def test_brave_provider_dispatch(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-k")
        respx.get(BRAVE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "web": {
                        "results": [{"title": "B", "url": "https://b.example", "description": "d"}]
                    }
                },
            )
        )

        result = await web_search(query="test", provider="brave")

        assert result["success"] is True
        assert result["provider"] == "brave"
        assert result["result_count"] == 1

    @respx.mock
    async def test_missing_key_returns_not_configured_result(self, monkeypatch):
        for var in ("TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY", "SERPER_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("victor.config.api_keys.get_service_key", lambda service: None)
        route = respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": []}))

        result = await web_search(query="test", provider="tavily")

        assert result["success"] is False
        assert result["provider_configured"] is False
        assert result["provider"] == "tavily"
        assert "TAVILY_API_KEY" in result["error"]
        assert not route.called  # never fires an unauthenticated request

    @respx.mock
    async def test_provider_api_error_returns_structured_error(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "serper-k")
        respx.post(SERPER_URL).mock(return_value=httpx.Response(500, text="boom"))

        result = await web_search(query="test", provider="serper")

        assert result["success"] is False
        assert result["provider"] == "serper"
        assert "error" in result
        assert "500" in result["error"] or "failed" in result["error"].lower()

    async def test_unknown_provider_rejected(self):
        result = await web_search(query="test", provider="not-a-provider")

        assert result["success"] is False
        assert "provider" in result["error"].lower()


class TestAutoFallback:
    @respx.mock
    async def test_auto_uses_first_configured_provider(self, monkeypatch):
        for var in ("TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY", "SERPER_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-k")
        monkeypatch.setattr("victor.config.api_keys.get_service_key", lambda service: None)
        respx.get(BRAVE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "web": {
                        "results": [{"title": "B", "url": "https://b.example", "description": "d"}]
                    }
                },
            )
        )

        result = await web_search(query="test", provider="auto")

        assert result["success"] is True
        assert result["provider"] == "brave"

    @respx.mock
    async def test_auto_without_keys_falls_back_to_duckduckgo(self, monkeypatch):
        for var in ("TAVILY_API_KEY", "BRAVE_SEARCH_API_KEY", "SERPER_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("victor.config.api_keys.get_service_key", lambda service: None)
        respx.post(DDG_HTML_URL).mock(
            return_value=httpx.Response(
                200,
                html="""
                <html><body>
                <div class="result">
                  <a class="result__a" href="https://ddg.example">DDG Title</a>
                  <a class="result__snippet">DDG snippet.</a>
                </div>
                </body></html>
                """,
            )
        )

        result = await web_search(query="test", provider="auto")

        assert result["success"] is True
        assert result["provider"] == "duckduckgo"
        assert result["result_count"] == 1

    @respx.mock
    async def test_explicit_duckduckgo_alias_ddg(self, monkeypatch):
        monkeypatch.setattr("victor.config.api_keys.get_service_key", lambda service: None)
        respx.post(DDG_HTML_URL).mock(return_value=httpx.Response(200, html=_EMPTY_DDG_HTML))

        result = await web_search(query="test", provider="ddg")

        assert result["success"] is True
        assert result["provider"] == "duckduckgo"

    @respx.mock
    async def test_ddg_search_never_routes_through_the_aiohttp_pool(self, monkeypatch):
        """Negative guard for the CI regression (result_count 5 == live DDG).

        respx intercepts httpx only; if ``_request_text`` ever takes the
        aiohttp pool branch during these tests, the request bypasses every
        mock and hits the REAL DuckDuckGo. The pool must stay disabled here —
        the fixture forces it off, and this test fails loudly if that regresses
        (instead of silently leaking network and flipping on CI). The recorder
        asserts AFTER the call because the pool branch swallows exceptions
        into its httpx fallback.
        """
        from victor.tools import http_pool

        pool_calls: list = []

        async def _recording_forbidden_pool(*args, **kwargs):
            pool_calls.append(args)
            raise AssertionError("pool must not be used under respx-mocked tests")

        monkeypatch.setattr(http_pool, "get_http_pool", _recording_forbidden_pool)
        monkeypatch.setattr("victor.config.api_keys.get_service_key", lambda service: None)
        respx.post(DDG_HTML_URL).mock(
            return_value=httpx.Response(
                200,
                html="""
                <html><body>
                <div class="result">
                  <a class="result__a" href="https://ddg.example">DDG Title</a>
                  <a class="result__snippet">DDG snippet.</a>
                </div>
                </body></html>
                """,
            )
        )

        result = await web_search(query="test", provider="auto")

        assert result["success"] is True
        assert result["provider"] == "duckduckgo"
        assert result["result_count"] == 1
        assert pool_calls == [], "request routed through the aiohttp pool (real-network leak)"


class TestRegistration:
    async def test_web_search_exposes_provider_parameter(self):
        from victor.tools.web_search_tool import web_search as ws

        assert "provider" in ws.__annotations__

    async def test_web_tools_discoverable_via_shared_registry(self):
        from victor.agent.shared_tool_registry import SharedToolRegistry

        registry = SharedToolRegistry.get_instance()
        try:
            tools = registry.get_tool_classes()
            assert "web_search" in tools
            assert "web_fetch" in tools
        finally:
            SharedToolRegistry.reset_instance()
