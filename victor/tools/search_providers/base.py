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

"""Shared contract for key-gated search API adapters (FOU-A5 phase 1)."""

import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)

# Default per-request timeout for search API calls.
DEFAULT_SEARCH_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class SearchResult:
    """One normalized search hit (title/url/snippet contract)."""

    title: str
    url: str
    snippet: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class SearchProviderError(RuntimeError):
    """A search API returned an error or unusable response."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"{provider}: {message}")


class ProviderNotConfigured(SearchProviderError):
    """The provider's API key is absent — the provider is not offered."""

    def __init__(self, provider: str, env_var: str):
        self.env_var = env_var
        super().__init__(
            provider,
            f"not configured — set {env_var} in the environment or run "
            f"`victor keys --set-service {provider}`",
        )


@runtime_checkable
class SearchProvider(Protocol):
    """Protocol every search API adapter implements."""

    name: str
    env_var: str

    def is_configured(self) -> bool:
        """True when the provider's key resolves right now."""
        ...

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        timeout_seconds: float = DEFAULT_SEARCH_TIMEOUT_SECONDS,
    ) -> list[SearchResult]:
        """Run one web search and return structured results.

        Raises:
            ProviderNotConfigured: when the API key is absent.
            SearchProviderError: when the API responds with an error or
                an unusable payload. Network exceptions from httpx
                propagate to the caller (the tool converts them into
                structured error results).
        """
        ...


def resolve_provider_key(env_var: str, service: str) -> Optional[str]:
    """Resolve a search provider key: env var first, then Victor's service chain.

    The service chain (keyring / ``~/.victor/api_keys.yaml``) is consulted via
    ``victor.config.api_keys.get_service_key`` so ``victor keys --set-service
    <name>`` works for search providers the same way it does for market-data
    services. Import is deferred to keep this module import-cheap and to avoid
    a hard dependency on the config layer at tool-import time.
    """
    env_key = os.environ.get(env_var)
    if env_key and env_key.strip():
        return env_key.strip()
    try:
        from victor.config.api_keys import get_service_key

        return get_service_key(service)
    except Exception as exc:  # pragma: no cover - defensive: config layer absent
        logger.debug("Service key lookup for %s failed: %s", service, exc)
        return None


def build_provider_configured_error(provider: str, env_var: str) -> dict:
    """Structured "provider not configured" result for the web_search tool."""
    return {
        "success": False,
        "provider": provider,
        "provider_configured": False,
        "results": "",
        "result_count": 0,
        "error": (
            f"Search provider '{provider}' is not configured — set {env_var} in "
            f"the environment or run `victor keys --set-service {provider}`"
        ),
    }


class KeyGatedSearchProvider:
    """Shared plumbing for key-gated search API adapters.

    Subclasses set ``name``/``env_var`` and implement :meth:`search` in terms
    of :meth:`require_key` + :meth:`fetch_json` + ``normalize_results``; this
    base owns key resolution, the HTTP exchange and status/JSON error mapping
    so the wire-specific code in each adapter stays minimal (duplication
    ratchet: the per-adapter copy-paste of this plumbing is a ratchet
    failure).
    """

    name: str = ""
    env_var: str = ""

    def __init__(self, api_key: Optional[str] = None):
        # Explicit key overrides resolution (used by tests and embedders).
        self._api_key = api_key

    def is_configured(self) -> bool:
        return bool(self._api_key) or bool(resolve_provider_key(self.env_var, self.name))

    def require_key(self) -> str:
        key = self._api_key or resolve_provider_key(self.env_var, self.name)
        if not key:
            raise ProviderNotConfigured(self.name, self.env_var)
        return key

    async def fetch_json(
        self,
        request: Callable[[httpx.AsyncClient], Awaitable[httpx.Response]],
        *,
        timeout_seconds: float = DEFAULT_SEARCH_TIMEOUT_SECONDS,
    ) -> dict:
        """Run one httpx request and return the decoded JSON payload.

        ``request`` builds the awaited call against a fresh client, e.g.
        ``lambda client: client.post(url, headers=..., json=...)``. Non-200
        statuses and undecodable JSON raise :class:`SearchProviderError`.
        """
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await request(client)

        if response.status_code != 200:
            raise SearchProviderError(
                self.name,
                f"search request failed (status {response.status_code})",
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise SearchProviderError(self.name, f"invalid JSON response: {exc}") from exc
        return payload

    @staticmethod
    def normalize_results(
        raw_results: object,
        *,
        url_key: str,
        title_key: str = "title",
        snippet_key: str = "snippet",
        max_results: int = 5,
    ) -> "list[SearchResult]":
        """Normalize a provider result list into the SearchResult contract.

        Non-dict entries and entries without a URL are skipped; the list is
        truncated to ``max_results``.
        """
        if not isinstance(raw_results, list):
            raw_results = []
        results: list[SearchResult] = []
        for item in raw_results[: max(0, int(max_results))]:
            if not isinstance(item, dict):
                continue
            url = str(item.get(url_key) or "")
            if not url:
                continue
            results.append(
                SearchResult(
                    title=str(item.get(title_key) or ""),
                    url=url,
                    snippet=str(item.get(snippet_key) or ""),
                )
            )
        return results

    @staticmethod
    def results_list(payload: dict, *path: str) -> list:
        """Extract the result list at a (nested) payload path.

        Returns ``[]`` when any hop is missing or mistyped, so adapters never
        need their own defensive type checks.
        """
        node: object = payload
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        return node if isinstance(node, list) else []
