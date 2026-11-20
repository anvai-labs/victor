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

"""Tests for the key-gated search API adapters (FOU-A5 phase 1).

All HTTP traffic is mocked with respx — no real network anywhere.
"""

import httpx
import pytest
import respx

from victor.tools.search_providers import (
    ProviderNotConfigured,
    SearchProviderError,
    SearchResult,
    get_provider,
    list_configured_providers,
    list_providers,
)
from victor.tools.search_providers.base import resolve_provider_key
from victor.tools.search_providers.brave import BraveSearchProvider
from victor.tools.search_providers.serper import SerperProvider
from victor.tools.search_providers.tavily import TavilyProvider

TAVILY_URL = "https://api.tavily.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
SERPER_URL = "https://google.serper.dev/search"


class TestTavilyAdapter:
    @respx.mock
    @pytest.mark.asyncio
    async def test_success_maps_structured_results(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        route = respx.post(TAVILY_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "AsyncIO docs",
                            "url": "https://docs.python.org/3/library/asyncio.html",
                            "content": "Event loop primitives.",
                        },
                        {"title": "No URL should be skipped", "url": "", "content": "x"},
                    ]
                },
            )
        )

        results = await TavilyProvider().search("python asyncio", max_results=5)

        assert route.called
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].title == "AsyncIO docs"
        assert results[0].url == "https://docs.python.org/3/library/asyncio.html"
        assert results[0].snippet == "Event loop primitives."

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_results(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": []}))

        results = await TavilyProvider().search("nothing matches", max_results=5)

        assert results == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_api_error_raises_structured_error(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        respx.post(TAVILY_URL).mock(return_value=httpx.Response(500, text="boom"))

        with pytest.raises(SearchProviderError):
            await TavilyProvider().search("query", max_results=5)

    @pytest.mark.asyncio
    async def test_missing_key_raises_not_configured(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setattr(
            "victor.tools.search_providers.base.resolve_provider_key",
            lambda env_var, service: None,
        )

        with pytest.raises(ProviderNotConfigured):
            await TavilyProvider().search("query", max_results=5)

    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_header_carries_key(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")
        route = respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": []}))

        await TavilyProvider().search("q", max_results=3)

        assert route.calls[0].request.headers["authorization"] == "Bearer tvly-secret"


class TestBraveAdapter:
    @respx.mock
    @pytest.mark.asyncio
    async def test_success_maps_structured_results(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")
        route = respx.get(BRAVE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {
                                "title": "Brave result",
                                "url": "https://example.com/a",
                                "description": "A page about things.",
                            }
                        ]
                    }
                },
            )
        )

        results = await BraveSearchProvider().search("things", max_results=5)

        assert route.called
        assert len(results) == 1
        assert results[0].title == "Brave result"
        assert results[0].url == "https://example.com/a"
        assert results[0].snippet == "A page about things."
        assert route.calls[0].request.headers["x-subscription-token"] == "brave-key"

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_web_section(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")
        respx.get(BRAVE_URL).mock(return_value=httpx.Response(200, json={}))

        assert await BraveSearchProvider().search("q", max_results=5) == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_api_error_raises_structured_error(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-key")
        respx.get(BRAVE_URL).mock(return_value=httpx.Response(429, text="rate limited"))

        with pytest.raises(SearchProviderError):
            await BraveSearchProvider().search("q", max_results=5)

    @pytest.mark.asyncio
    async def test_missing_key_raises_not_configured(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.setattr(
            "victor.tools.search_providers.base.resolve_provider_key",
            lambda env_var, service: None,
        )

        with pytest.raises(ProviderNotConfigured):
            await BraveSearchProvider().search("q", max_results=5)


class TestSerperAdapter:
    @respx.mock
    @pytest.mark.asyncio
    async def test_success_maps_structured_results(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "serper-key")
        route = respx.post(SERPER_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "organic": [
                        {
                            "title": "Serper result",
                            "link": "https://example.com/s",
                            "snippet": "Organic hit.",
                        }
                    ]
                },
            )
        )

        results = await SerperProvider().search("q", max_results=5)

        assert route.called
        assert len(results) == 1
        assert results[0].title == "Serper result"
        assert results[0].url == "https://example.com/s"
        assert results[0].snippet == "Organic hit."
        assert route.calls[0].request.headers["x-api-key"] == "serper-key"

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_results(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "serper-key")
        respx.post(SERPER_URL).mock(return_value=httpx.Response(200, json={"organic": []}))

        assert await SerperProvider().search("q", max_results=5) == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_api_error_raises_structured_error(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "serper-key")
        respx.post(SERPER_URL).mock(return_value=httpx.Response(403, json={"message": "no"}))

        with pytest.raises(SearchProviderError):
            await SerperProvider().search("q", max_results=5)

    @pytest.mark.asyncio
    async def test_missing_key_raises_not_configured(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        monkeypatch.setattr(
            "victor.tools.search_providers.base.resolve_provider_key",
            lambda env_var, service: None,
        )

        with pytest.raises(ProviderNotConfigured):
            await SerperProvider().search("q", max_results=5)


class TestKeyResolution:
    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "env-key")
        assert resolve_provider_key("TAVILY_API_KEY", "tavily") == "env-key"

    def test_service_key_fallback(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        monkeypatch.setattr("victor.config.api_keys.get_service_key", lambda service: "keyring-key")
        assert resolve_provider_key("SERPER_API_KEY", "serper") == "keyring-key"

    def test_absent_everywhere_returns_none(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.setattr("victor.config.api_keys.get_service_key", lambda service: None)
        assert resolve_provider_key("BRAVE_SEARCH_API_KEY", "brave") is None


class TestProviderRegistry:
    def test_get_provider_known_names(self):
        assert isinstance(get_provider("tavily"), TavilyProvider)
        assert isinstance(get_provider("brave"), BraveSearchProvider)
        assert isinstance(get_provider("brave_search"), BraveSearchProvider)
        assert isinstance(get_provider("serper"), SerperProvider)

    def test_get_provider_unknown_returns_none(self):
        assert get_provider("nope") is None

    def test_list_providers_covers_all_three(self):
        names = set(list_providers())
        assert {"tavily", "brave", "serper"} <= names

    def test_list_configured_providers_respects_keys(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        monkeypatch.setattr("victor.config.api_keys.get_service_key", lambda service: None)
        configured = list_configured_providers()
        assert "tavily" in configured
        assert "brave" not in configured
        assert "serper" not in configured

    def test_service_env_vars_map_contains_search_services(self):
        from victor.config.api_keys import SERVICE_ENV_VARS

        assert SERVICE_ENV_VARS.get("tavily") == "TAVILY_API_KEY"
        assert SERVICE_ENV_VARS.get("brave") == "BRAVE_SEARCH_API_KEY"
        assert SERVICE_ENV_VARS.get("serper") == "SERPER_API_KEY"
