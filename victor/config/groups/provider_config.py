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

"""Provider configuration settings.

Extracted from victor/config/settings.py to improve maintainability.
Contains configuration for LLM providers, API keys, and model settings.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

try:
    from pydantic import SecretStr
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    from pydantic_settings import SettingsConfigDict


def reveal_secret(secret: Optional[SecretStr]) -> Optional[str]:
    """Reveal a SecretStr value.

    Args:
        secret: SecretStr to reveal

    Returns:
        Plain string value or None
    """
    if secret is None:
        return None
    return secret.get_secret_value() if isinstance(secret, SecretStr) else str(secret)


class ProviderConfig(BaseSettings):
    """Configuration for a specific provider."""

    api_key: Optional[SecretStr] = None
    base_url: Optional[Union[str, List[str]]] = None
    timeout: int = 300  # 5 minutes - increased for CPU-only inference
    max_retries: int = 3
    organization: Optional[str] = None  # For OpenAI

    @property
    def api_key_value(self) -> Optional[str]:
        """Return the plain API key for provider construction."""
        return reveal_secret(self.api_key)

    def to_runtime_dict(self) -> Dict[str, Any]:
        """Serialize provider config with secrets safely unwrapped.

        Excludes SecretStr fields from model_dump to prevent intermediate
        plaintext in stack traces, then adds them back explicitly.
        """
        secret_fields = {
            name
            for name, info in self.model_fields.items()
            if info.annotation is SecretStr
            or (
                hasattr(info.annotation, "__args__")
                and SecretStr in getattr(info.annotation, "__args__", ())
            )
        }
        result = self.model_dump(exclude_none=True, exclude=secret_fields)
        for name in secret_fields:
            val = getattr(self, name, None)
            if val is not None:
                result[name] = val.get_secret_value() if isinstance(val, SecretStr) else val
        return result


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    name: str
    provider: str
    temperature: float = 0.7
    max_tokens: int = 4096
    supports_tool_calls: bool = True
    supports_prompt_caching: bool = False
    supports_kv_prefix_caching: bool = False


class ProviderSettings(BaseModel):
    """Provider-related settings extracted from main Settings class.

    Groups all provider configuration including API keys, models,
    default generation parameters, and provider-specific settings.
    """

    # Sandhi typed transport: ask for the full native provider body under
    # ChatResponseV1.extensions. Off by default — Victor consumes only the
    # neutral contract (F1, victor#665) and the body is debug-only weight.
    # Flip on for wire-level debugging of a provider integration.
    sandhi_include_native_response: bool = False

    # Default provider and model
    default_provider: str = "ollama"
    default_model: str = "qwen3.5:27b-q4_K_M"  # MoE model: fast + knowledgeable

    # Default generation parameters
    default_temperature: float = Field(
        default=0.6, ge=0.0, le=2.0, description="Default temperature for generation"
    )
    default_max_tokens: int = Field(
        default=4096, gt=0, description="Default maximum tokens for generation"
    )

    # Provider-specific API keys (stored as SecretStr for security)
    # Note: Only providers actively used in main Settings are included here
    anthropic_api_key: Optional[SecretStr] = None
    openai_api_key: Optional[SecretStr] = None
    google_api_key: Optional[SecretStr] = None
    moonshot_api_key: Optional[SecretStr] = None  # Moonshot AI for Kimi K2 models
    deepseek_api_key: Optional[SecretStr] = None  # DeepSeek for DeepSeek-V3 models

    # Local server URLs
    # Can be overridden via environment variables:
    #   OLLAMA_BASE_URL, LMSTUDIO_BASE_URLS (comma-separated), VLLM_BASE_URL
    # For LAN servers, set: LMSTUDIO_BASE_URLS="http://<your-server>:1234,http://localhost:1234"
    ollama_base_url: str = "http://localhost:11434"
    # LMStudio tiered endpoints (try in order) - defaults to localhost only
    # Set LMSTUDIO_BASE_URLS env var to add LAN servers
    lmstudio_base_urls: List[str] = Field(default_factory=lambda: ["http://127.0.0.1:1234"])
    vllm_base_url: str = "http://localhost:8000"

    # LMStudio resource guard
    lmstudio_max_vram_gb: Optional[float] = Field(
        default=48.0,
        description="Cap model selection to this budget (GB); override via env/config",
    )

    # Provider configurations (dict keyed by provider name)
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)

    # Model capabilities
    tool_calling_models: Dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("default_provider")
    @classmethod
    def validate_default_provider(cls, v: str) -> str:
        """Validate that default_provider is a known provider.

        Args:
            v: Provider name

        Returns:
            Validated provider name

        Raises:
            ValueError: If provider is unknown
        """
        # All registered providers from victor.providers.registry
        known_providers = {
            # Core providers
            "ollama",
            "anthropic",
            "openai",
            "google",
            # Cloud providers
            "xai",
            "grok",
            "zai",
            "zhipuai",
            "zhipu",
            "qwen",
            "alibaba",
            "dashscope",
            "moonshot",
            "kimi",
            "deepseek",
            "groqcloud",
            "mistral",
            "together",
            "openrouter",
            "fireworks",
            "cerebras",
            "vertex",
            "vertexai",
            "azure",
            "azure-openai",
            "bedrock",
            "aws",
            "huggingface",
            "hf",
            "replicate",
            # Local backends
            "lmstudio",
            "vllm",
            "llamacpp",
            "llama-cpp",
            "llama.cpp",
            "mlx",
            "mlx-lm",
            "applesilicon",
        }

        if v not in known_providers:
            raise ValueError(
                f"Unknown provider '{v}'. " f"Known providers: {', '.join(sorted(known_providers))}"
            )

        return v

    @field_validator("default_model")
    @classmethod
    def validate_default_model(cls, v: str, info) -> str:
        """Validate that default_model is not empty.

        Args:
            v: Model name
            info: Field validation info

        Returns:
            Validated model name

        Raises:
            ValueError: If model is empty
        """
        if not v or not v.strip():
            raise ValueError("default_model cannot be empty")

        # Warn if model looks like a default placeholder
        if v.lower() in ("llama2", "model", "default"):
            # We'll allow this but warn
            pass

        return v

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider.

        Args:
            provider: Provider name (e.g., 'anthropic', 'openai')

        Returns:
            API key as string if available, None otherwise
        """
        key_field = f"{provider}_api_key"
        key = getattr(self, key_field, None)
        if key is not None:
            return reveal_secret(key)
        return None

    def get_provider_config(self, provider: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider.

        Args:
            provider: Provider name

        Returns:
            ProviderConfig if available, None otherwise
        """
        return self.providers.get(provider)
