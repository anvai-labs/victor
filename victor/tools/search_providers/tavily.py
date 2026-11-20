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


from victor.tools.search_providers.base import (
    DEFAULT_SEARCH_TIMEOUT_SECONDS,
    KeyGatedSearchProvider,
    SearchProviderError,
    SearchResult,
)

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilyProvider(KeyGatedSearchProvider):
    """Key-gated Tavily search adapter."""

    name = "tavily"
    env_var = "TAVILY_API_KEY"

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        timeout_seconds: float = DEFAULT_SEARCH_TIMEOUT_SECONDS,
    ) -> list[SearchResult]:
        key = self.require_key()
        payload = await self.fetch_json(
            lambda client: client.post(
                TAVILY_SEARCH_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "query": query,
                    "max_results": max(1, int(max_results)),
                    "include_answer": False,
                },
            ),
            timeout_seconds=timeout_seconds,
        )

        if not isinstance(payload.get("results"), list):
            # Tavily's wire contract: the results list is always present.
            raise SearchProviderError(self.name, "response missing 'results' list")

        results = self.normalize_results(
            self.results_list(payload, "results"),
            url_key="url",
            title_key="title",
            snippet_key="content",
            max_results=max_results,
        )
        logger.debug("[tavily] query=%r results=%d", query, len(results))
        return results
