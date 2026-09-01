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

"""Provider configuration registry - OCP-compliant provider settings.

This module implements the Strategy pattern for provider-specific configuration,
eliminating the OCP violation in Settings.get_provider_settings().

Design Goals:
- Open for extension: Add new providers by registering a strategy
- Closed for modification: No changes to core logic needed
- Single Responsibility: Each strategy handles one provider
- Simplified: DefaultProviderConfig handles most simple API key providers

Migration Notes:
- Simple API key providers now use DefaultProviderConfig
- Complex providers (OAuth, multiple endpoints) use dedicated strategies
- Legacy strategy classes are deprecated but kept for backward compatibility

Usage:
    from victor.config.provider_config_registry import (
        get_provider_config_registry,
        register_provider_config,
    )

    # Get settings for a provider
    registry = get_provider_config_registry()
    settings = registry.get_settings("anthropic", settings_instance)

    # Register a custom provider
    @register_provider_config("my_provider")
    class MyProviderConfig(ProviderConfigStrategy):
        def get_settings(self, settings: "Settings") -> Dict[str, Any]:
            return {"api_key": get_api_key("my_provider"), "base_url": "..."}
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

from pydantic import SecretStr

from victor.config.secrets import unwrap_secrets

if TYPE_CHECKING:
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class ProviderConfigStrategy(ABC):
    """Abstract base for provider-specific configuration strategies.

    Each provider implements this interface to define how its
    settings are resolved (API keys, base URLs, etc.).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """The provider name this strategy handles."""
        ...

    @property
    def aliases(self) -> List[str]:
        """Alternative names that map to this provider."""
        return []

    @abstractmethod
    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get provider-specific settings.

        Args:
            settings: The Settings instance for accessing config
            base_settings: Settings already loaded from profiles.yaml

        Returns:
            Dictionary of provider settings (api_key, base_url, etc.)
        """
        ...


# =============================================================================
# Built-in Provider Strategies
# =============================================================================

# Default endpoints for simple API key providers
# These providers use API key auth with a single endpoint
DEFAULT_PROVIDER_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "xai": "https://api.x.ai/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groqcloud": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "together": "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "huggingface": "https://api-inference.huggingface.co",
    "replicate": "https://api.replicate.com/v1",
}


class DefaultProviderConfig(ProviderConfigStrategy):
    """Default configuration strategy for simple API key providers.

    This strategy handles providers that:
    - Use API key authentication
    - Have a single endpoint
    - Don't require special configuration

    This eliminates the need for individual strategy classes for each provider.
    """

    def __init__(
        self,
        provider_name: str,
        base_url: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ):
        """Initialize default provider config.

        Args:
            provider_name: The provider name
            base_url: Optional base URL (uses default if not specified)
            aliases: Optional alternative names that map to this provider
        """
        self._provider_name = provider_name
        self._base_url = base_url or DEFAULT_PROVIDER_ENDPOINTS.get(provider_name)
        self._aliases = aliases or []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def aliases(self) -> List[str]:
        return self._aliases

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get provider settings using default resolution."""
        from victor.config.api_keys import get_api_key

        result = dict(base_settings)

        # Try to get API key from nested provider group first (Phase 5 layout),
        # then flat attribute (backward compat), then env/secrets resolver.
        settings_attr = f"{self._provider_name}_api_key"
        provider_group = getattr(settings, "provider", None)
        raw_key = getattr(provider_group, settings_attr, None) if provider_group else None
        if raw_key is None:
            raw_key = getattr(settings, settings_attr, None)
        if raw_key and hasattr(raw_key, "get_secret_value"):
            api_key = raw_key.get_secret_value()
        else:
            api_key = raw_key or get_api_key(self._provider_name)

        if api_key:
            result["api_key"] = api_key

        # Set base URL if available
        if self._base_url:
            result.setdefault("base_url", self._base_url)

        return result


# =============================================================================
# Complex Provider Strategies (OAuth, Multiple Endpoints, URL Detection)
# =============================================================================

# NOTE: Simple provider strategies below are deprecated.
# Use DefaultProviderConfig instead for new providers.
# Existing strategies kept for backward compatibility.


class OpenAIConfig(ProviderConfigStrategy):
    """Configuration strategy for OpenAI."""

    @property
    def provider_name(self) -> str:
        return "openai"

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        from victor.config.api_keys import get_api_key

        result = dict(base_settings)
        auth_mode = base_settings.get("auth_mode", "api_key")

        if auth_mode == "oauth":
            # OAuth mode: use ChatGPT subscription
            result["auth_mode"] = "oauth"
            result.setdefault("base_url", "https://chatgpt.com/backend-api/codex")
        else:
            # API key mode
            raw_key = settings.provider.openai_api_key
            api_key = (raw_key.get_secret_value() if raw_key else None) or get_api_key("openai")
            if api_key:
                result["api_key"] = api_key
            result.setdefault("base_url", "https://api.openai.com/v1")

        return result


class OllamaConfig(ProviderConfigStrategy):
    """Configuration strategy for Ollama (local)."""

    @property
    def provider_name(self) -> str:
        return "ollama"

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(base_settings)
        result.setdefault("base_url", settings.provider.ollama_base_url)
        return result


class LMStudioConfig(ProviderConfigStrategy):
    """Configuration strategy for LM Studio (local)."""

    @property
    def provider_name(self) -> str:
        return "lmstudio"

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(base_settings)

        # lmstudio_base_urls lives under settings.provider after Phase 5 LEGACY mapping
        provider_group = getattr(settings, "provider", None)
        urls = (
            getattr(provider_group, "lmstudio_base_urls", None)
            if provider_group
            else getattr(settings, "lmstudio_base_urls", None)
        ) or []
        # If provider config supplied a list, merge/override
        if "base_url" in result:
            cfg_url = result["base_url"]
            if isinstance(cfg_url, list):
                urls = cfg_url
            elif isinstance(cfg_url, str):
                urls = [cfg_url]

        chosen = None
        try:
            import httpx

            for url in urls:
                try:
                    resp = httpx.get(f"{url}/v1/models", timeout=1.5)
                    if resp.status_code == 200:
                        chosen = url
                        break
                except Exception as e:
                    logger.debug("LM Studio probe failed for %s: %s", url, e)
                    continue
        except ImportError:
            logger.debug("httpx not available for LM Studio URL probing")

        if urls:
            result["base_url"] = f"{(chosen or urls[0]).rstrip('/')}/v1"
        return result


class VLLMConfig(ProviderConfigStrategy):
    """Configuration strategy for vLLM (local)."""

    @property
    def provider_name(self) -> str:
        return "vllm"

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(base_settings)
        result.setdefault("base_url", settings.provider.vllm_base_url)
        return result


class QwenConfig(ProviderConfigStrategy):
    """Configuration strategy for Qwen (Alibaba Cloud)."""

    @property
    def provider_name(self) -> str:
        return "qwen"

    @property
    def aliases(self) -> List[str]:
        from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec

        return list(get_openai_compat_provider_spec("qwen").aliases)

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        from victor.config.api_keys import get_api_key
        from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec

        result = dict(base_settings)
        spec = get_openai_compat_provider_spec("qwen")
        auth_mode = base_settings.get("auth_mode", "api_key")
        if auth_mode == "oauth":
            result["auth_mode"] = "oauth"
            result.setdefault("base_url", spec.endpoint_options["portal"])
        else:
            api_key = get_api_key("qwen")
            if api_key:
                result["api_key"] = api_key
            result.setdefault("base_url", spec.endpoint_options["standard"])
        return result


class ZAIConfig(ProviderConfigStrategy):
    """Configuration strategy for Z.AI (Zhipu GLM).

    Handles coding plan endpoint switching: the Coding Plan requires
    a different base URL (/api/coding/paas/v4/) than the standard API.
    """

    @property
    def provider_name(self) -> str:
        return "zai"

    @property
    def aliases(self) -> List[str]:
        from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec

        return list(get_openai_compat_provider_spec("zai").aliases)

    def get_settings(
        self,
        settings: "Settings",
        base_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        from victor.config.api_keys import get_api_key
        from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec

        result = dict(base_settings)
        spec = get_openai_compat_provider_spec("zai")

        # Resolve API key
        api_key = get_api_key("zai")
        if api_key:
            result["api_key"] = api_key

        # Coding plan uses a dedicated endpoint
        coding_plan = result.pop("coding_plan", False)
        if coding_plan:
            result.setdefault("base_url", spec.endpoint_options["coding"])
            result["coding_plan"] = True
        else:
            result.setdefault("base_url", spec.endpoint_options["standard"])

        return result


# =============================================================================
# Registry
# =============================================================================


def _normalize_gateway_root(url: str) -> str:
    """Canonicalize a configured gateway URL to its ROOT (no /v1, no /v1beta).

    The proxy serves the AI routes under its own path prefixes
    (``/v1/chat/completions``, ``/v1/messages``, ``/v1beta/models/...``), and the
    sandhi adapters append those prefixes themselves — so the base the transport
    holds must be the bare root. Config authors naturally paste either the root
    or the ``/v1`` form (both appear in victor's own docs and tests); every
    accepted spelling collapses to one canonical root here, and the per-family
    derivation happens at the one place that knows the family
    (``sandhi_transport._typed_provider``). Mirrors the rule
    ``GatewayRoute.openai_kwargs`` already encodes for the OpenAI case.
    """
    root = url.rstrip("/")
    for suffix in ("/v1", "/v1beta"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return root.rstrip("/")


def resolve_provider_gateway(base_settings: Dict[str, Any], provider: str) -> None:
    """Normalize and env-resolve a per-provider Sandhi gateway block in place.

    The ``gateway`` block (loaded from ``profiles.yaml`` providers section as a
    ``ProviderGatewayConfig``) is normalized to a plain dict ``{"url", "virtual_key"}``
    so it can flow through ``**kwargs`` into the provider's ``extra_config`` and be
    read by the Sandhi transport. The URL falls back to ``SANDHI_GATEWAY_URL`` when
    the block omits it, and is canonicalized to the proxy ROOT (a trailing ``/v1``
    or ``/v1beta`` is accepted and stripped — the transport derives the per-family
    prefix). The virtual key is resolved from the block, else from a per-provider
    env var (``SANDHI_GATEWAY_VIRTUAL_KEY_<PROVIDER>``), else the global
    ``SANDHI_GATEWAY_VIRTUAL_KEY``. A block with no URL anywhere is dropped
    (gateway mode stays off); a URL with no resolvable key is kept as an empty
    string so the transport can fail closed with a clear error.
    """
    gateway = base_settings.get("gateway")
    if gateway is None:
        # No block at all: gateway mode is off unless the env supplies a URL.
        # Materialize a block from SANDHI_GATEWAY_URL so env-only configs
        # (SANDHI_GATEWAY_URL + SANDHI_GATEWAY_VIRTUAL_KEY) work without any
        # per-provider YAML.
        if not os.environ.get("SANDHI_GATEWAY_URL", "").strip():
            return
        gateway = {}
    if isinstance(gateway, dict):
        url = str(gateway.get("url") or "").strip()
    else:
        url = str(getattr(gateway, "url", "") or "").strip()
    if not url:
        # Required-field validation aside, an explicit empty url also means
        # "take the env".
        url = os.environ.get("SANDHI_GATEWAY_URL", "").strip()
    if not url:
        if isinstance(gateway, dict) and gateway:
            base_settings.pop("gateway", None)
        else:
            base_settings.pop("gateway", None)
        return
    url = _normalize_gateway_root(url)
    if isinstance(gateway, dict):
        virtual_key = gateway.get("virtual_key")
    else:
        virtual_key = getattr(gateway, "virtual_key_value", getattr(gateway, "virtual_key", None))
    if isinstance(virtual_key, SecretStr):
        virtual_key = virtual_key.get_secret_value()
    virtual_key = str(virtual_key or "").strip()
    if not virtual_key:
        virtual_key = (
            os.environ.get(f"SANDHI_GATEWAY_VIRTUAL_KEY_{provider.upper()}")
            or os.environ.get("SANDHI_GATEWAY_VIRTUAL_KEY")
            or ""
        ).strip()
    base_settings["gateway"] = {"url": url, "virtual_key": virtual_key}


@dataclass
class ProviderConfigRegistry:
    """Registry for provider configuration strategies.

    Implements the Registry pattern for OCP compliance.
    Thread-safe for concurrent access.
    """

    _strategies: Dict[str, ProviderConfigStrategy] = field(default_factory=dict)
    _aliases: Dict[str, str] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def register(self, strategy: ProviderConfigStrategy) -> None:
        """Register a provider configuration strategy.

        Args:
            strategy: The strategy to register
        """
        with self._lock:
            name = strategy.provider_name
            self._strategies[name] = strategy

            # Register aliases
            for alias in strategy.aliases:
                self._aliases[alias] = name

            logger.debug(f"Registered provider config: {name}")

    def get_settings(
        self,
        provider: str,
        settings: "Settings",
        profile_overrides: Optional[Dict[str, Any]] = None,
        account_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get settings for a provider.

        Merge precedence (lowest to highest):
        1. profiles.yaml providers section (base_settings)
        2. account_data from AccountManager config.yaml (credentials)
        3. profile_overrides from CLI flags (--coding-plan, --auth-mode)
        4. Strategy logic (provider-specific endpoint switching, URL probing)

        Args:
            provider: Provider name (or alias)
            settings: Settings instance
            profile_overrides: Runtime overrides (e.g., coding_plan, auth_mode from CLI)
            account_data: Credential data from AccountManager (api_key, auth_mode, base_url)

        Returns:
            Provider settings dictionary
        """
        with self._lock:
            # Resolve alias
            resolved = self._aliases.get(provider, provider)

            # 1. Load base settings from profiles.yaml providers section
            base_settings = {}
            provider_config = settings.load_provider_config(resolved)
            if provider_config:
                base_settings.update(unwrap_secrets(provider_config.model_dump(exclude_none=True)))

            # 2. Merge account data (credentials from config.yaml)
            if account_data:
                # API key from AccountManager always wins (it's the credential source)
                if "api_key" in account_data:
                    base_settings["api_key"] = account_data["api_key"]
                # Other account data fills gaps (don't override existing settings)
                for key in ("auth_mode", "base_url"):
                    if key in account_data and key not in base_settings:
                        base_settings[key] = account_data[key]

            # 3. Apply profile overrides (CLI flags like --coding-plan, --auth-mode)
            if profile_overrides:
                base_settings.update(profile_overrides)

            # 3.5 Normalize + env-resolve the per-provider Sandhi gateway block so it
            # flows as a plain dict into the provider kwargs regardless of strategy.
            resolve_provider_gateway(base_settings, resolved)

            # Get strategy
            strategy = self._strategies.get(resolved)
            if strategy:
                return strategy.get_settings(settings, base_settings)

            # Fallback to DefaultProviderConfig for simple API key providers
            if provider in DEFAULT_PROVIDER_ENDPOINTS:
                logger.debug(f"Using default config strategy for provider '{provider}'")
                return DefaultProviderConfig(provider).get_settings(settings, base_settings)

            # Last resort: return base settings if no strategy
            logger.debug(f"No config strategy for provider '{provider}', using base settings")
            return base_settings

    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        with self._lock:
            return list(self._strategies.keys())


# =============================================================================
# Singleton and Decorators
# =============================================================================

_registry_instance: Optional[ProviderConfigRegistry] = None
_registry_lock = threading.Lock()


def get_provider_config_registry() -> ProviderConfigRegistry:
    """Get the global provider config registry.

    Returns:
        Global ProviderConfigRegistry instance
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = ProviderConfigRegistry()
                _register_builtin_providers(_registry_instance)

    return _registry_instance


def _register_builtin_providers(registry: ProviderConfigRegistry) -> None:
    """Register all built-in provider configurations."""
    builtin_strategies: List[ProviderConfigStrategy] = [
        DefaultProviderConfig("anthropic"),
        OpenAIConfig(),
        DefaultProviderConfig("google", aliases=["gemini"]),
        DefaultProviderConfig("xai", aliases=["grok"]),
        OllamaConfig(),
        LMStudioConfig(),
        VLLMConfig(),
        DefaultProviderConfig("moonshot", aliases=["kimi"]),
        DefaultProviderConfig("deepseek"),
        DefaultProviderConfig("groqcloud", aliases=["groq"]),
        ZAIConfig(),
        QwenConfig(),
    ]

    for strategy in builtin_strategies:
        registry.register(strategy)


def register_provider_config(
    provider_name: str,
) -> Callable[[Type[ProviderConfigStrategy]], Type[ProviderConfigStrategy]]:
    """Decorator to register a provider config strategy.

    Usage:
        @register_provider_config("my_provider")
        class MyProviderConfig(ProviderConfigStrategy):
            ...
    """

    def decorator(
        cls: Type[ProviderConfigStrategy],
    ) -> Type[ProviderConfigStrategy]:
        instance = cls()
        get_provider_config_registry().register(instance)
        return cls

    return decorator


__all__ = [
    "ProviderConfigStrategy",
    "ProviderConfigRegistry",
    "get_provider_config_registry",
    "register_provider_config",
    "resolve_provider_gateway",
    # Default strategy (handles most simple API key providers)
    "DefaultProviderConfig",
    "DEFAULT_PROVIDER_ENDPOINTS",
    # Built-in strategies (for testing/extension)
    "OpenAIConfig",
    "OllamaConfig",
    "LMStudioConfig",
    "VLLMConfig",
    "QwenConfig",
]
