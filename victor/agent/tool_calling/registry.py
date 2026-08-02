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

"""
Registry for tool calling adapters.

Provides factory methods for creating the appropriate adapter based on
provider name and model.

Architecture:
    Most LLM providers use OpenAI-compatible tool calling format. Only three
    providers have genuinely different API formats requiring dedicated adapters:
    - Anthropic: content blocks with input_schema
    - Google/Vertex: function_call with dict args
    - Bedrock: Converse API with toolSpec/toolUse

    All other providers (DeepSeek, Azure, Z.AI, xAI, Groq, etc.) use the
    OpenAI format and share OpenAIToolCallingAdapter by default. Provider-
    specific adapters are retained and can be enabled via settings for cases
    that need model-specific handling (e.g., DeepSeek reasoner, Azure o1).

    Setting: ``settings.tool_calling.use_provider_specific_adapters`` (default False)
    controls whether provider-specific adapters are used. When False (default),
    OpenAIToolCallingAdapter handles all OpenAI-compatible providers uniformly.
"""

import logging
import os
from typing import Any, Dict, Optional, Type

from victor.agent.tool_calling.base import BaseToolCallingAdapter

logger = logging.getLogger(__name__)

# Providers that have genuinely different API formats (NOT OpenAI-compatible)
_NON_OPENAI_PROVIDERS = {"anthropic", "google", "bedrock"}

# Providers that need text-based fallback parsing (local models)
_LOCAL_FALLBACK_PROVIDERS = {"ollama", "lmstudio"}

# OpenAI-compatible providers that have dedicated adapter classes
# (retained for opt-in via use_provider_specific_adapters setting)
_OPENAI_COMPAT_WITH_SPECIFIC_ADAPTER = {"deepseek", "azure", "vllm"}


class ToolCallingAdapterRegistry:
    """Registry for tool calling adapters.

    Maintains a mapping of provider names to adapter classes and provides
    factory methods for creating adapters.

    By default, all OpenAI-compatible providers use OpenAIToolCallingAdapter.
    Provider-specific adapters (DeepSeek, Azure, vLLM) are available via:
    - Setting: ``tool_calling.use_provider_specific_adapters: true``
    - Env var: ``VICTOR_USE_PROVIDER_SPECIFIC_ADAPTERS=true``

    Usage:
        # Register a custom adapter
        ToolCallingAdapterRegistry.register("custom", CustomAdapter)

        # Get an adapter for a provider
        adapter = ToolCallingAdapterRegistry.get_adapter("ollama", model="llama3.1")
    """

    _adapters: Dict[str, Type[BaseToolCallingAdapter]] = {}

    @classmethod
    def register(cls, provider_name: str, adapter_class: Type[BaseToolCallingAdapter]) -> None:
        """Register an adapter class for a provider.

        Args:
            provider_name: Name of the provider (lowercase)
            adapter_class: Adapter class to register
        """
        cls._adapters[provider_name.lower()] = adapter_class
        logger.debug(
            f"Registered tool calling adapter: {provider_name} -> {adapter_class.__name__}"
        )

    @classmethod
    def _use_provider_specific(cls, config: Optional[Dict[str, Any]] = None) -> bool:
        """Check if provider-specific adapters should be used.

        Checks (in order):
        1. Settings passed via config dict
        2. Environment variable VICTOR_USE_PROVIDER_SPECIFIC_ADAPTERS
        """
        # Check settings
        if config:
            settings = config.get("settings")
            if settings:
                tc_settings = getattr(settings, "tool_calling", None)
                if tc_settings:
                    val = getattr(tc_settings, "use_provider_specific_adapters", None)
                    if val is not None:
                        return bool(val)

        # Check env var
        env_val = os.environ.get("VICTOR_USE_PROVIDER_SPECIFIC_ADAPTERS", "").lower()
        return env_val in ("true", "1", "yes")

    @classmethod
    def get_adapter(
        cls,
        provider_name: str,
        model: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseToolCallingAdapter:
        """Get an adapter for a provider.

        Args:
            provider_name: Name of the provider
            model: Model name/identifier
            config: Optional configuration

        Returns:
            Appropriate adapter instance

        Raises:
            ValueError: If no adapter found for provider
        """
        provider_key = provider_name.lower()

        # Import adapters here to avoid circular imports
        from victor.agent.tool_calling.adapters import (
            AnthropicToolCallingAdapter,
            AzureOpenAIToolCallingAdapter,
            BedrockToolCallingAdapter,
            DeepSeekToolCallingAdapter,
            GoogleToolCallingAdapter,
            LMStudioToolCallingAdapter,
            OllamaToolCallingAdapter,
            OpenAICompatToolCallingAdapter,
            OpenAIToolCallingAdapter,
        )

        use_specific = cls._use_provider_specific(config)

        # Register default adapters if not already registered
        if not cls._adapters:
            # Non-OpenAI providers always use their dedicated adapters
            cls._adapters = {
                "anthropic": AnthropicToolCallingAdapter,
                "google": GoogleToolCallingAdapter,
                "bedrock": BedrockToolCallingAdapter,
                # Local models need text-based fallback parsing
                "ollama": OllamaToolCallingAdapter,
                "lmstudio": LMStudioToolCallingAdapter,
            }

        # Non-OpenAI and local providers always use dedicated adapters
        if provider_key in cls._adapters and provider_key in (
            _NON_OPENAI_PROVIDERS | _LOCAL_FALLBACK_PROVIDERS
        ):
            adapter_class = cls._adapters[provider_key]
            return adapter_class(model=model, config=config)

        # Provider-specific adapters (opt-in via setting)
        if use_specific and provider_key in _OPENAI_COMPAT_WITH_SPECIFIC_ADAPTER:
            specific_adapters = {
                "deepseek": DeepSeekToolCallingAdapter,
                "azure": AzureOpenAIToolCallingAdapter,
                "vllm": OpenAICompatToolCallingAdapter,
            }
            adapter_class = specific_adapters[provider_key]
            if adapter_class == OpenAICompatToolCallingAdapter:
                return adapter_class(  # type: ignore[call-arg]
                    model=model,
                    config=config,
                    provider_variant=provider_key,
                )
            logger.debug(
                f"Using provider-specific adapter for '{provider_key}': "
                f"{adapter_class.__name__}"
            )
            return adapter_class(model=model, config=config)

        # Check custom registrations
        if provider_key in cls._adapters:
            adapter_class = cls._adapters[provider_key]
            if adapter_class == OpenAICompatToolCallingAdapter:
                return adapter_class(  # type: ignore[call-arg]
                    model=model,
                    config=config,
                    provider_variant=provider_key,
                )
            return adapter_class(model=model, config=config)

        # Default: all OpenAI-compatible providers use OpenAIToolCallingAdapter
        return OpenAIToolCallingAdapter(model=model, config=config)

    @classmethod
    def list_providers(cls) -> list:
        """List registered provider names."""
        # Ensure defaults are loaded
        cls.get_adapter("ollama")  # Triggers default registration
        return list(cls._adapters.keys())

    @classmethod
    def is_registered(cls, provider_name: str) -> bool:
        """Check if a provider has a registered adapter."""
        # Ensure defaults are loaded
        if not cls._adapters:
            cls.get_adapter("ollama")  # Triggers default registration
        return provider_name.lower() in cls._adapters
