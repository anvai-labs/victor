# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Registry-version-gated cache for tool schema/token estimates.

Extracted from ToolService.estimate_tool_tokens (co-design review U4-F1)
to keep tool_service.py under its hotspot size-ratchet cap. The cache
lives here, beside the one consumer that owns it, rather than growing
the hotspot further.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

EstimateKey = Tuple[str, Optional[str], Optional[str]]


class ToolEstimateCache:
    """(tool_name, tier, provider_category) -> token-estimate cache.

    Wholesale-invalidated whenever the owning registry's
    ``schema_cache_version`` changes — every register/unregister/enable/
    disable/batch_update call bumps it (co-design review U4-F1), so a
    single mutation drops the whole cache instead of leaking stale
    per-tool estimates across turns.
    """

    def __init__(self) -> None:
        self._entries: Dict[EstimateKey, int] = {}
        self._version: Optional[int] = None

    def get(self, key: EstimateKey, registry_version: Optional[int]) -> Optional[int]:
        """Return a cached estimate, refreshing wholesale on a version change."""
        if registry_version is None:
            return None
        self._sync(registry_version)
        return self._entries.get(key)

    def put(self, key: EstimateKey, estimate: int, registry_version: Optional[int]) -> None:
        if registry_version is None:
            return
        self._sync(registry_version)
        self._entries[key] = estimate

    def _sync(self, registry_version: int) -> None:
        if registry_version != self._version:
            self._entries = {}
            self._version = registry_version

    @staticmethod
    def resolve_registry(owner: Any) -> Any:
        """Prefer ``owner._registrar``; fall back to ``owner._registry``.

        Explicit ``is None`` check: a falsy-but-real (empty) registry must
        not be skipped in favor of the fallback attribute (adversarial-
        review finding — `or` treated an empty ToolRegistry as absent).
        """
        registrar = getattr(owner, "_registrar", None)
        return registrar if registrar is not None else getattr(owner, "_registry", None)

    @staticmethod
    def resolve_version(registry: Any) -> Optional[int]:
        """Extract an int ``schema_cache_version`` from *registry*, or None.

        Adversarial-review findings folded in: a Mock/stub registrar
        exposing a non-int version must degrade to uncached rather than
        thrash the cache once per call.
        """
        version = getattr(registry, "schema_cache_version", None)
        return version if isinstance(version, int) else None


def estimate_tool_tokens(
    service: Any,
    tool: Any,
    *,
    provider_category: Optional[str] = None,
    use_cache: bool = True,
) -> int:
    """Estimate token cost for *tool* at its current schema level, cached.

    Extracted from ToolService.estimate_tool_tokens (co-design review
    U4-F1) to keep tool_service.py under its hotspot size-ratchet cap.
    Falls back to a name-length heuristic when the tool's schema cannot be
    generated. The cache is (tool_name, tier, provider_category) ->
    estimate, wholesale-invalidated on the owning registry's
    schema_cache_version. ``use_cache=False`` bypasses it entirely — the
    _demote_tools_to_fit_budget STUB probe temporarily patches
    ``tool._schema_level``, making its estimate unrepresentative of the
    real tier (adversarial-review finding).
    """
    from victor.config.tool_tiers import get_provider_tool_tier, get_tool_tier

    cache: ToolEstimateCache = service.__dict__.setdefault(
        "_tool_estimate_cache", ToolEstimateCache()
    )
    registry_version = (
        ToolEstimateCache.resolve_version(ToolEstimateCache.resolve_registry(service))
        if use_cache
        else None
    )
    try:
        tier = (
            get_provider_tool_tier(tool.name, provider_category)
            if provider_category
            else get_tool_tier(tool.name)
        )
        cache_key = (tool.name, tier, provider_category)
        cached = cache.get(cache_key, registry_version) if use_cache else None
        if cached is not None:
            return cached
        estimate = len(str(tool.to_schema(tier))) // 4
        if use_cache:
            cache.put(cache_key, estimate, registry_version)
        return estimate
    except Exception:
        return len(tool.name) + 50
