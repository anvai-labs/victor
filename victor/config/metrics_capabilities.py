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

"""Provider metrics capabilities loader.

This module provides config-driven metrics capabilities for different providers.
Capabilities are loaded from YAML configuration with caching for performance.

Resolution Order:
    1. User overrides (~/.victor/profiles.yaml provider_metrics)
    2. Model-specific settings (models section with fnmatch)
    3. Provider defaults (providers section)
    4. Global defaults (defaults section)

Example usage:
    caps = get_metrics_capabilities("anthropic", "claude-3-5-sonnet-20241022")
    if caps.cost_enabled:
        cost = (tokens / 1_000_000) * caps.input_cost_per_mtok
"""

from __future__ import annotations

import fnmatch
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Cache for loaded config
_config_cache: Optional[Dict[str, Any]] = None
_cache_timestamp: float = 0
_cache_ttl: float = 300.0  # 5 minutes


@dataclass
class ProviderMetricsCapabilities:
    """Metrics capabilities for a provider/model combination.

    This dataclass encapsulates what metrics a provider supports and
    the pricing configuration for cost calculation.
    """

    provider: str
    model: str

    # Token metrics capabilities
    supports_prompt_tokens: bool = False
    supports_completion_tokens: bool = False
    supports_total_tokens: bool = False
    supports_cache_tokens: bool = False

    # Cost metrics
    cost_enabled: bool = False
    pricing_source: Optional[str] = None  # "config" | None
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    cache_read_cost_per_mtok: float = 0.0
    cache_write_cost_per_mtok: float = 0.0

    # Estimation settings (fallback when no actual tokens)
    fallback_enabled: bool = True
    chars_per_token: int = 4

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> Dict[str, float]:
        """Calculate cost breakdown for token usage.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            cache_read_tokens: Number of cache read tokens (Anthropic)
            cache_write_tokens: Number of cache write tokens (Anthropic)

        Returns:
            Dict with input_cost, output_cost, cache_cost, total_cost
        """
        if not self.cost_enabled:
            return {
                "input_cost": 0.0,
                "output_cost": 0.0,
                "cache_cost": 0.0,
                "total_cost": 0.0,
            }

        input_cost = (prompt_tokens / 1_000_000) * self.input_cost_per_mtok
        output_cost = (completion_tokens / 1_000_000) * self.output_cost_per_mtok
        cache_cost = (cache_read_tokens / 1_000_000) * self.cache_read_cost_per_mtok + (
            cache_write_tokens / 1_000_000
        ) * self.cache_write_cost_per_mtok
        total_cost = input_cost + output_cost + cache_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cache_cost": cache_cost,
            "total_cost": total_cost,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider,
            "model": self.model,
            "token_metrics": {
                "supports_prompt_tokens": self.supports_prompt_tokens,
                "supports_completion_tokens": self.supports_completion_tokens,
                "supports_total_tokens": self.supports_total_tokens,
                "supports_cache_tokens": self.supports_cache_tokens,
            },
            "cost_metrics": {
                "enabled": self.cost_enabled,
                "pricing_source": self.pricing_source,
                "input_cost_per_mtok": self.input_cost_per_mtok,
                "output_cost_per_mtok": self.output_cost_per_mtok,
                "cache_read_cost_per_mtok": self.cache_read_cost_per_mtok,
                "cache_write_cost_per_mtok": self.cache_write_cost_per_mtok,
            },
            "estimation": {
                "fallback_enabled": self.fallback_enabled,
                "chars_per_token": self.chars_per_token,
            },
        }


def _load_config() -> Dict[str, Any]:
    """Load metrics capabilities config with caching.

    Returns:
        Merged config from bundled defaults and user overrides
    """
    global _config_cache, _cache_timestamp

    now = time.time()
    if _config_cache is not None and (now - _cache_timestamp) < _cache_ttl:
        return _config_cache

    config: Dict[str, Any] = {}

    # Phase 1: Load bundled config
    bundled_path = Path(__file__).parent / "provider_metrics.yaml"
    if bundled_path.exists():
        try:
            with open(bundled_path, "r") as f:
                config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded bundled metrics config from {bundled_path}")
        except Exception as e:
            logger.warning(f"Failed to load bundled metrics config: {e}")

    # Phase 2: Load user overrides
    try:
        from victor.config.settings import Settings

        user_profiles_path = Settings.get_config_dir() / "profiles.yaml"
        if user_profiles_path.exists():
            with open(user_profiles_path, "r") as f:
                user_data = yaml.safe_load(f) or {}

            if "provider_metrics" in user_data:
                _merge_config(config, user_data["provider_metrics"])
                logger.debug(f"Merged user metrics overrides from {user_profiles_path}")
    except Exception as e:
        logger.debug(f"No user metrics overrides: {e}")

    _config_cache = config
    _cache_timestamp = now
    return config


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Deep merge override config into base config.

    Args:
        base: Base configuration (modified in place)
        override: Override configuration to merge
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge_config(base[key], value)
        else:
            base[key] = value


def _get_provider_config(config: Dict[str, Any], provider: str) -> Dict[str, Any]:
    """Get provider-specific config with defaults applied.

    Args:
        config: Full configuration
        provider: Provider name

    Returns:
        Provider config with defaults merged
    """
    defaults = config.get("defaults", {})
    providers = config.get("providers", {})

    # Start with defaults
    result = _deep_copy(defaults)

    # Merge provider-specific
    if provider in providers:
        _merge_config(result, providers[provider])

    return result


def _get_model_pricing(
    config: Dict[str, Any], provider: str, model: str
) -> Optional[Dict[str, float]]:
    """Get model-specific pricing config.

    Args:
        config: Full configuration
        provider: Provider name
        model: Model name (supports fnmatch patterns)

    Returns:
        Pricing dict or None
    """
    # Check models section for overrides
    models = config.get("models") or {}
    for pattern, model_config in models.items():
        if fnmatch.fnmatch(model, pattern):
            pricing = model_config.get("cost_metrics", {}).get("pricing")
            if pricing:
                return pricing

    # Check provider pricing
    providers = config.get("providers", {})
    provider_config = providers.get(provider, {})
    pricing_config = provider_config.get("cost_metrics", {}).get("pricing", {})

    # Try exact match first
    if model in pricing_config:
        return pricing_config[model]

    # Try pattern matching
    for pattern, pricing in pricing_config.items():
        if fnmatch.fnmatch(model, pattern):
            return pricing

    return None


def _deep_copy(d: Dict[str, Any]) -> Dict[str, Any]:
    """Create a deep copy of a dict (simple implementation)."""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _deep_copy(value)
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def get_metrics_capabilities(provider: str, model: str) -> ProviderMetricsCapabilities:
    """Get metrics capabilities for a provider/model combination.

    Args:
        provider: Provider name (e.g., "anthropic", "openai", "ollama")
        model: Model name (e.g., "claude-3-5-sonnet-20241022", "gpt-4o")

    Returns:
        ProviderMetricsCapabilities with resolved settings
    """
    config = _load_config()
    provider_config = _get_provider_config(config, provider)

    # Extract token metrics
    token_metrics = provider_config.get("token_metrics", {})
    supports_prompt = token_metrics.get("prompt_tokens", False)
    supports_completion = token_metrics.get("completion_tokens", False)
    supports_total = token_metrics.get("total_tokens", False)
    supports_cache = token_metrics.get("cache_read_tokens", False) or token_metrics.get(
        "cache_write_tokens", False
    )

    # Extract cost metrics
    cost_metrics = provider_config.get("cost_metrics", {})
    cost_enabled = cost_metrics.get("enabled", False)
    pricing_source = cost_metrics.get("pricing_source")

    # Get model-specific pricing
    input_cost = 0.0
    output_cost = 0.0
    cache_read_cost = 0.0
    cache_write_cost = 0.0

    if cost_enabled and pricing_source == "config":
        pricing = _get_model_pricing(config, provider, model)
        if pricing:
            input_cost = pricing.get("input_per_mtok", 0.0)
            output_cost = pricing.get("output_per_mtok", 0.0)
            cache_read_cost = pricing.get("cache_read_per_mtok", 0.0)
            cache_write_cost = pricing.get("cache_write_per_mtok", 0.0)

    # Extract estimation settings
    estimation = provider_config.get("estimation", {})
    fallback_enabled = estimation.get("fallback_enabled", True)
    chars_per_token = estimation.get("chars_per_token", 4)

    return ProviderMetricsCapabilities(
        provider=provider,
        model=model,
        supports_prompt_tokens=supports_prompt,
        supports_completion_tokens=supports_completion,
        supports_total_tokens=supports_total,
        supports_cache_tokens=supports_cache,
        cost_enabled=cost_enabled,
        pricing_source=pricing_source,
        input_cost_per_mtok=input_cost,
        output_cost_per_mtok=output_cost,
        cache_read_cost_per_mtok=cache_read_cost,
        cache_write_cost_per_mtok=cache_write_cost,
        fallback_enabled=fallback_enabled,
        chars_per_token=chars_per_token,
    )


def clear_cache() -> None:
    """Clear the config cache (useful for testing)."""
    global _config_cache, _cache_timestamp
    _config_cache = None
    _cache_timestamp = 0


def get_supported_providers() -> list[str]:
    """Get list of providers with configured capabilities.

    Returns:
        List of provider names
    """
    config = _load_config()
    return list(config.get("providers", {}).keys())


def get_provider_models_with_pricing(provider: str) -> list[str]:
    """Get list of models with configured pricing for a provider.

    Args:
        provider: Provider name

    Returns:
        List of model names with pricing
    """
    config = _load_config()
    providers = config.get("providers", {})
    provider_config = providers.get(provider, {})
    pricing = provider_config.get("cost_metrics", {}).get("pricing", {})
    return list(pricing.keys())
