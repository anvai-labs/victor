"""Runtime capability discovery structures for providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProviderRuntimeCapabilities:
    """Runtime capabilities and limits for a provider/model pair."""

    provider: str
    model: str
    context_window: int
    supports_tools: bool
    supports_streaming: bool
    source: str = "config"  # e.g., "discovered", "config"
    raw: Optional[Dict[str, Any]] = None


def cached_config_context_window(
    cache: Dict[str, int], *, provider: str, base_url: str, model: str
) -> int:
    """Return a model's context window from a per-endpoint cache, else from config limits.

    Local-server providers (Ollama, LMStudio) populate ``cache`` from runtime discovery; on a
    miss the configured ``provider_context_limits`` value is resolved and memoized.

    Args:
        cache: Provider-owned cache keyed by ``"<base_url>:<model>"``.
        provider: Provider name used for the config lookup (e.g. ``"ollama"``).
        base_url: Server endpoint the cache entry belongs to.
        model: Model identifier.

    Returns:
        Context window size in tokens.
    """
    cache_key = f"{base_url}:{model}"
    if cache_key in cache:
        return cache[cache_key]

    from victor.config.config_loaders import get_provider_limits

    context_window = get_provider_limits(provider, model).context_window
    cache[cache_key] = context_window
    return context_window
