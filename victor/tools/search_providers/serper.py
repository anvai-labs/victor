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

"""Serper Google Search adapter (thin httpx client over the SearchProvider protocol).

API reference: https://serper.dev/ (Google SERP API)
"""

import logging

from victor.tools.search_providers.base import (
    DEFAULT_SEARCH_TIMEOUT_SECONDS,
    KeyGatedSearchProvider,
    SearchResult,
)

logger = logging.getLogger(__name__)

SERPER_SEARCH_URL = "https://google.serper.dev/search"


class SerperProvider(KeyGatedSearchProvider):
    """Key-gated Serper (Google SERP) adapter."""

    name = "serper"
    env_var = "SERPER_API_KEY"

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
                SERPER_SEARCH_URL,
                headers={"X-API-KEY": key},
                json={
                    "q": query,
                    "num": max(1, int(max_results)),
                },
            ),
            timeout_seconds=timeout_seconds,
        )

        results = self.normalize_results(
            self.results_list(payload, "organic"),
            url_key="link",
            title_key="title",
            snippet_key="snippet",
            max_results=max_results,
        )
        logger.debug("[serper] query=%r results=%d", query, len(results))
        return results
