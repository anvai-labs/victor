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

"""Key-gated search API adapters (FOU-A5 phase 1).

Each adapter is a thin async ``httpx`` client implementing the single
:class:`SearchProvider` protocol. Adapters are OPTIONAL and degrade
gracefully: an absent key means the provider is simply not offered —
``web_search`` falls back to the keyless DuckDuckGo path that already
lives in :mod:`victor.tools.web_search_tool` (no new scraping here).

Key resolution per provider (highest priority first):
  1. Environment variable (``TAVILY_API_KEY``, ``BRAVE_SEARCH_API_KEY``,
     ``SERPER_API_KEY``)
  2. Victor service-key chain (keyring / ``~/.victor/api_keys.yaml``) via
     ``victor keys --set-service <name>``
"""

from victor.tools.search_providers.base import (
    ProviderNotConfigured,
    SearchProvider,
    SearchProviderError,
    SearchResult,
    build_provider_configured_error,
    resolve_provider_key,
)
from victor.tools.search_providers.brave import BraveSearchProvider
from victor.tools.search_providers.serper import SerperProvider
from victor.tools.search_providers.tavily import TavilyProvider

# Canonical name -> provider class. Aliases map onto canonical names.
PROVIDER_CLASSES: dict[str, type[SearchProvider]] = {
    "tavily": TavilyProvider,
    "brave": BraveSearchProvider,
    "serper": SerperProvider,
}

PROVIDER_ALIASES: dict[str, str] = {
    "brave_search": "brave",
}

# "auto" preference order: cheapest/generous free tiers first.
AUTO_PROVIDER_ORDER: tuple[str, ...] = ("tavily", "brave", "serper")

# Search providers plus the keyless DuckDuckGo fallback accepted by web_search.
KNOWN_PROVIDER_NAMES: frozenset[str] = frozenset(
    PROVIDER_CLASSES.keys() | PROVIDER_ALIASES.keys() | {"auto", "duckduckgo", "ddg"}
)


def canonical_provider_name(name: str) -> str | None:
    """Map a provider name/alias to its canonical name (None if unknown)."""
    normalized = (name or "").strip().lower()
    if not normalized:
        return None
    return PROVIDER_ALIASES.get(normalized, normalized)


def get_provider(name: str) -> SearchProvider | None:
    """Instantiate a provider by canonical name or alias (None if unknown)."""
    canonical = canonical_provider_name(name)
    if canonical is None or canonical not in PROVIDER_CLASSES:
        return None
    return PROVIDER_CLASSES[canonical]()


def list_providers() -> list[str]:
    """List canonical provider names."""
    return sorted(PROVIDER_CLASSES)


def list_configured_providers() -> list[str]:
    """List providers whose key resolves right now (env or service chain)."""
    return [name for name, cls in sorted(PROVIDER_CLASSES.items()) if cls().is_configured()]


__all__ = [
    "AUTO_PROVIDER_ORDER",
    "KNOWN_PROVIDER_NAMES",
    "PROVIDER_ALIASES",
    "PROVIDER_CLASSES",
    "BraveSearchProvider",
    "ProviderNotConfigured",
    "SearchProvider",
    "SearchProviderError",
    "SearchResult",
    "SerperProvider",
    "TavilyProvider",
    "canonical_provider_name",
    "get_provider",
    "list_configured_providers",
    "list_providers",
    "resolve_provider_key",
]
