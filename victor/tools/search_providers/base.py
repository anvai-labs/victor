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
from typing import Optional, Protocol, runtime_checkable

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
