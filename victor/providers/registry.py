from __future__ import annotations

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

"""Provider registry for managing and discovering LLM providers."""

import importlib
import os
import platform
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from victor.core.registry import BaseRegistry
from victor.providers.base import BaseProvider, ProviderNotFoundError

__all__ = ["ProviderRegistry", "ProviderNotFoundError", "TIER_1_PROVIDERS"]

#: Tier-1 support-tier providers (primary registry keys) per ADR-029
#: (docs/architecture/adr/029-provider-support-tiers.md): integration-tested,
#: issues triaged, tracked for upstream API drift. All other registered
#: providers are Community support tier (unit-tested, best-effort). This is
#: informational metadata only — it MUST NOT be used for behavioral gating.
TIER_1_PROVIDERS: frozenset[str] = frozenset(
    {"anthropic", "openai", "google", "ollama", "vllm", "bedrock"}
)


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class _LazyProviderSpec:
    """Deferred provider import specification."""

    module_path: str
    class_name: str
    primary_name: str
    preflight: Optional[Callable[[str], Optional[str]]] = None


_lazy_provider_specs: Dict[str, _LazyProviderSpec] = {}
_lazy_provider_lock = threading.Lock()
_mlx_preflight_result: Optional[Tuple[bool, str]] = None
_provider_aliases: Dict[str, str] = {}


def _env_flag(name: str) -> Optional[bool]:
    """Parse a boolean-like environment variable."""
    value = os.getenv(name)
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return None


def _register_lazy_provider(
    aliases: List[str],
    *,
    module_path: str,
    class_name: str,
    preflight: Optional[Callable[[str], Optional[str]]] = None,
) -> None:
    """Register aliases that should be materialized on first use."""
    primary_name = aliases[0]
    spec = _LazyProviderSpec(
        module_path=module_path,
        class_name=class_name,
        primary_name=primary_name,
        preflight=preflight,
    )
    for alias in aliases:
        _lazy_provider_specs[alias] = spec
        _provider_aliases[alias] = primary_name


def _run_mlx_preflight() -> Tuple[bool, str]:
    """Verify MLX runtime readiness in a subprocess to avoid crashing main process."""
    global _mlx_preflight_result
    if _mlx_preflight_result is not None:
        return _mlx_preflight_result

    if _env_flag("VICTOR_ENABLE_MLX_PROVIDER") is False:
        _mlx_preflight_result = (
            False,
            "disabled by VICTOR_ENABLE_MLX_PROVIDER=0.",
        )
        return _mlx_preflight_result

    if sys.platform != "darwin":
        _mlx_preflight_result = (
            False,
            "requires macOS/Metal runtime.",
        )
        return _mlx_preflight_result

    if platform.machine().lower() != "arm64":
        _mlx_preflight_result = (
            False,
            "requires Apple Silicon (arm64).",
        )
        return _mlx_preflight_result

    if _env_flag("VICTOR_MLX_SKIP_PREFLIGHT") is True:
        _mlx_preflight_result = (True, "")
        return _mlx_preflight_result

    check_cmd = [
        sys.executable,
        "-c",
        "import mlx_lm\nimport mlx.core as mx\nprint(mx.default_device())",
    ]
    try:
        result = subprocess.run(
            check_cmd,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        _mlx_preflight_result = (
            False,
            f"failed to run MLX runtime preflight ({exc}).",
        )
        return _mlx_preflight_result

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip().splitlines()
        tail = details[-1] if details else "unknown MLX runtime error"
        _mlx_preflight_result = (
            False,
            f"MLX runtime preflight failed ({tail}).",
        )
        return _mlx_preflight_result

    _mlx_preflight_result = (True, "")
    return _mlx_preflight_result


def _mlx_preflight(provider_name: str) -> Optional[str]:
    """Preflight gate for MLX provider aliases."""
    ok, reason = _run_mlx_preflight()
    if ok:
        return None
    return (
        f"Provider '{provider_name}' is unavailable: {reason} "
        "Set VICTOR_MLX_SKIP_PREFLIGHT=1 to bypass this safety check."
    )


def _materialize_lazy_provider(name: str) -> Optional[Type[BaseProvider]]:
    """Import and register a lazily-declared provider.

    Uses double-checked locking to ensure thread-safe materialization:
    only one thread imports the module while others wait and reuse the result.
    """
    # First check (no lock) — fast path when already materialized
    spec = _lazy_provider_specs.get(name)
    if spec is None:
        return _registry_instance.get(name)

    with _lazy_provider_lock:
        # Second check (under lock) — another thread may have materialized
        spec = _lazy_provider_specs.get(name)
        if spec is None:
            return _registry_instance.get(name)

        if spec.preflight:
            preflight_error = spec.preflight(name)
            if preflight_error:
                raise ProviderNotFoundError(message=preflight_error, provider=name)

        try:
            module = importlib.import_module(spec.module_path)
            provider_class = getattr(module, spec.class_name)
        except Exception as exc:
            raise ProviderNotFoundError(
                message=(
                    f"Provider '{name}' failed to load from "
                    f"{spec.module_path}.{spec.class_name}: {exc}"
                ),
                provider=name,
            ) from exc

        for alias, alias_spec in list(_lazy_provider_specs.items()):
            if alias_spec == spec:
                _registry_instance.register(alias, provider_class)
                del _lazy_provider_specs[alias]

        return provider_class


class _ProviderRegistryImpl(BaseRegistry[str, Type[BaseProvider]]):
    """Internal registry implementation for provider management.

    Extends BaseRegistry to provide provider-specific functionality including:
    - Factory method for instantiating providers
    - Raises ProviderNotFoundError on missing providers
    """

    def get_or_raise(self, name: str) -> Type[BaseProvider]:
        """Get a provider class by name, raising if not found.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            ProviderNotFoundError: If provider not found
        """
        provider = self.get(name)
        if provider is None:
            provider = _materialize_lazy_provider(name)
        if provider is None:
            available = sorted(set(self.list_all()) | set(_lazy_provider_specs.keys()))
            raise ProviderNotFoundError(
                message=f"Provider '{name}' not found. Available: {', '.join(available)}",
                provider=name,
            )
        return provider

    def create(self, name: str, **kwargs: Any) -> BaseProvider:
        """Create a provider instance.

        Args:
            name: Provider name
            **kwargs: Provider initialization arguments

        Returns:
            Provider instance

        Raises:
            ProviderNotFoundError: If provider not found
        """
        provider_class = self.get_or_raise(name)
        # Sandhi is the typed wire/runtime boundary for every admitted provider family.
        from victor.providers.sandhi_transport import resolve_transport_class

        provider_class = resolve_transport_class(name, provider_class, kwargs)
        return provider_class(**kwargs)


# Singleton instance for backward compatibility
_registry_instance = _ProviderRegistryImpl()


class ProviderRegistry:
    """Registry for LLM provider management.

    This is a static class facade that maintains backward compatibility
    with existing code while delegating to a BaseRegistry-based implementation.

    For new code, consider using the instance-based approach via
    `get_provider_registry()` for better testability.
    """

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """Register a provider.

        Args:
            name: Provider name (e.g., "ollama", "anthropic")
            provider_class: Provider class
        """
        _lazy_provider_specs.pop(name, None)
        _provider_aliases.setdefault(name, name)
        _registry_instance.register(name, provider_class)

    @classmethod
    def get(cls, name: str) -> Type[BaseProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            ProviderNotFoundError: If provider not found
        """
        return _registry_instance.get_or_raise(name)

    @classmethod
    def get_optional(cls, name: str) -> Optional[Type[BaseProvider]]:
        """Get a provider class by name, returning None if not found.

        Args:
            name: Provider name

        Returns:
            Provider class or None if not found
        """
        provider = _registry_instance.get(name)
        if provider is not None:
            return provider
        try:
            return _materialize_lazy_provider(name)
        except ProviderNotFoundError:
            return None

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseProvider:
        """Create a provider instance.

        Args:
            name: Provider name
            **kwargs: Provider initialization arguments

        Returns:
            Provider instance

        Raises:
            ProviderNotFoundError: If provider not found
        """
        return _registry_instance.create(name, **kwargs)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """
        providers = set(_registry_instance.list_all())
        providers.update(_lazy_provider_specs.keys())
        return sorted(providers)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider name

        Returns:
            True if registered, False otherwise
        """
        return (name in _registry_instance) or (name in _lazy_provider_specs)

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a provider.

        Args:
            name: Provider name

        Returns:
            True if the provider was found and removed, False otherwise
        """
        removed_lazy = _lazy_provider_specs.pop(name, None) is not None
        removed_registered = _registry_instance.unregister(name)
        primary_name = _provider_aliases.get(name)
        if primary_name == name:
            for alias, primary in list(_provider_aliases.items()):
                if primary == name:
                    _provider_aliases.pop(alias, None)
        else:
            _provider_aliases.pop(name, None)
        return removed_lazy or removed_registered

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers."""
        _registry_instance.clear()
        _lazy_provider_specs.clear()
        _provider_aliases.clear()
        global _mlx_preflight_result
        _mlx_preflight_result = None

    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """Get mapping of provider aliases to their primary names.

        Returns a dictionary where keys are all known aliases (including primary
        names) and values are the canonical/primary provider name. This provides
        a single source of truth for provider alias resolution.

        Returns:
            Dictionary mapping alias -> primary provider name

        Example:
            >>> aliases = ProviderRegistry.get_aliases()
            >>> aliases["claude"]
            "anthropic"
            >>> aliases["anthropic"]
            "anthropic"
        """
        aliases = dict(_provider_aliases)
        for name in _registry_instance.list_all():
            aliases.setdefault(name, name)

        return aliases


def get_provider_registry() -> _ProviderRegistryImpl:
    """Get the provider registry instance.

    This function provides access to the underlying BaseRegistry-based
    implementation for testing or advanced use cases.

    Returns:
        The singleton provider registry instance
    """
    return _registry_instance


# Auto-register all providers
def _register_default_providers() -> None:
    """Register all default providers."""
    # Core providers
    _register_lazy_provider(
        ["ollama"],
        module_path="victor.providers.ollama_provider",
        class_name="OllamaProvider",
    )
    _register_lazy_provider(
        ["anthropic"],
        module_path="victor.providers.anthropic_provider",
        class_name="AnthropicProvider",
    )
    _register_lazy_provider(
        ["openai"],
        module_path="victor.providers.openai_provider",
        class_name="OpenAIProvider",
    )
    _register_lazy_provider(
        ["google"],
        module_path="victor.providers.google_provider",
        class_name="GoogleProvider",
    )
    _register_lazy_provider(
        ["xai", "grok"],
        module_path="victor.providers.xai_provider",
        class_name="XAIProvider",
    )
    _register_lazy_provider(
        ["zai", "zhipuai", "zhipu", "zai-coding-plan", "zai-coding", "glm-coding"],
        module_path="victor.providers.zai_provider",
        class_name="ZAIProvider",
    )
    _register_lazy_provider(
        ["qwen", "alibaba", "dashscope"],
        module_path="victor.providers.qwen_provider",
        class_name="QwenProvider",
    )
    _register_lazy_provider(
        ["lmstudio"],
        module_path="victor.providers.lmstudio_provider",
        class_name="LMStudioProvider",
    )
    _register_lazy_provider(
        ["moonshot", "kimi"],
        module_path="victor.providers.moonshot_provider",
        class_name="MoonshotProvider",
    )
    _register_lazy_provider(
        ["deepseek"],
        module_path="victor.providers.deepseek_provider",
        class_name="DeepSeekProvider",
    )
    _register_lazy_provider(
        ["groq", "groqcloud"],
        module_path="victor.providers.groq_provider",
        class_name="GroqProvider",
    )
    _register_lazy_provider(
        ["mistral"],
        module_path="victor.providers.mistral_provider",
        class_name="MistralProvider",
    )
    _register_lazy_provider(
        ["together"],
        module_path="victor.providers.together_provider",
        class_name="TogetherProvider",
    )
    _register_lazy_provider(
        ["openrouter"],
        module_path="victor.providers.openrouter_provider",
        class_name="OpenRouterProvider",
    )
    _register_lazy_provider(
        ["fireworks"],
        module_path="victor.providers.fireworks_provider",
        class_name="FireworksProvider",
    )
    _register_lazy_provider(
        ["cerebras"],
        module_path="victor.providers.cerebras_provider",
        class_name="CerebrasProvider",
    )

    # Local backends (hardware/runtime dependent: CUDA/ROCm/CPU/MPS).
    # Keep these lazy-only without hardware preflight because they are
    # OpenAI-compatible server adapters (local or remote endpoints), not
    # in-process runtime initializers.
    _register_lazy_provider(
        ["vllm"],
        module_path="victor.providers.vllm_provider",
        class_name="VLLMProvider",
    )
    _register_lazy_provider(
        ["llamacpp", "llama-cpp", "llama.cpp"],
        module_path="victor.providers.llamacpp_provider",
        class_name="LlamaCppProvider",
    )
    # Self-hosted InferFlux inference server (sister project; OpenAI-compatible
    # dialect). Unlike the local trio above it has an own-slug Sandhi catalog
    # descriptor (ADR-0008), so it routes by slug instead of the default openai
    # dispatch with an explicit base_url.
    _register_lazy_provider(
        ["inferflux"],
        module_path="victor.providers.inferflux_provider",
        class_name="InferfluxProvider",
    )
    # MLX LM (Apple Silicon-optimized inference)
    # NOTE: Must remain lazy. Importing mlx_lm can hard-abort on unsupported
    # Metal runtime environments (e.g., missing visible MPS device).
    _register_lazy_provider(
        ["mlx", "mlx-lm", "applesilicon"],
        module_path="victor.providers.mlx_provider",
        class_name="MLXProvider",
        preflight=_mlx_preflight,
    )

    # Enterprise cloud providers
    _register_lazy_provider(
        ["vertex", "vertexai"],
        module_path="victor.providers.vertex_provider",
        class_name="VertexAIProvider",
    )
    _register_lazy_provider(
        ["azure", "azure-openai"],
        module_path="victor.providers.azure_openai_provider",
        class_name="AzureOpenAIProvider",
    )
    _register_lazy_provider(
        ["bedrock", "aws"],
        module_path="victor.providers.bedrock_provider",
        class_name="BedrockProvider",
    )
    _register_lazy_provider(
        ["huggingface", "hf"],
        module_path="victor.providers.huggingface_provider",
        class_name="HuggingFaceProvider",
    )
    _register_lazy_provider(
        ["replicate"],
        module_path="victor.providers.replicate_provider",
        class_name="ReplicateProvider",
    )


# Register providers on module import
_register_default_providers()
