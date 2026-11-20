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

"""Tavily search API adapter (thin httpx client over the SearchProvider protocol).

API reference: https://docs.tavily.com/
"""

import logging
from typing import Optional

import httpx

from victor.tools.search_providers.base import (
    DEFAULT_SEARCH_TIMEOUT_SECONDS,
    ProviderNotConfigured,
    SearchProviderError,
    SearchResult,
    resolve_provider_key,
)

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilyProvider:
    """Key-gated Tavily search adapter."""

    name = "tavily"
    env_var = "TAVILY_API_KEY"

    def __init__(self, api_key: Optional[str] = None):
        # Explicit key overrides resolution (used by tests and embedders).
        self._api_key = api_key

    def is_configured(self) -> bool:
        return bool(self._api_key) or bool(resolve_provider_key(self.env_var, self.name))

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        timeout_seconds: float = DEFAULT_SEARCH_TIMEOUT_SECONDS,
    ) -> list[SearchResult]:
        key = self._api_key or resolve_provider_key(self.env_var, self.name)
        if not key:
            raise ProviderNotConfigured(self.name, self.env_var)

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                TAVILY_SEARCH_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "query": query,
                    "max_results": max(1, int(max_results)),
                    "include_answer": False,
                },
            )

        if response.status_code != 200:
            raise SearchProviderError(
                self.name,
                f"search request failed (status {response.status_code})",
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise SearchProviderError(self.name, f"invalid JSON response: {exc}") from exc

        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise SearchProviderError(self.name, "response missing 'results' list")

        results: list[SearchResult] = []
        for item in raw_results[: max(0, int(max_results))]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            if not url:
                continue
            results.append(
                SearchResult(
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=str(item.get("content") or ""),
                )
            )
        logger.debug("[tavily] query=%r results=%d", query, len(results))
        return results
