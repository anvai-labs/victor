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

"""Brave Search API adapter (thin httpx client over the SearchProvider protocol).

API reference: https://api-dashboard.search.brave.com/app/documentation/web-search
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

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchProvider:
    """Key-gated Brave Search API adapter."""

    name = "brave"
    env_var = "BRAVE_SEARCH_API_KEY"

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
            response = await client.get(
                BRAVE_SEARCH_URL,
                headers={
                    "X-Subscription-Token": key,
                    "Accept": "application/json",
                },
                params={
                    "q": query,
                    "count": max(1, int(max_results)),
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

        web_section = payload.get("web") or {}
        raw_results = web_section.get("results") if isinstance(web_section, dict) else None
        if not isinstance(raw_results, list):
            raw_results = []

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
                    snippet=str(item.get("description") or ""),
                )
            )
        logger.debug("[brave] query=%r results=%d", query, len(results))
        return results
