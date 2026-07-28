# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Unified account configuration for Victor providers.

This module provides a simplified configuration model that unifies provider,
model, authentication, and endpoint configuration into a single ProviderAccount
abstraction. This replaces the complex multi-layer configuration system.

Key Benefits:
- Single source of truth for provider configuration
- Unified ~/.victor/config.yaml file
- Simpler authentication (API key, OAuth, or none)
- Model suffix support for endpoint variants (e.g., glm-4.6:coding)
- Built-in migration from old configuration format

Example Configuration (~/.victor/config.yaml):
    accounts:
      default:
        provider: anthropic
        model: claude-sonnet-4-5
        auth:
          method: api_key
          source: keyring
        tags: [chat, coding]

      glm-coding:
        provider: zai
        model: glm-4.6:coding  # Model suffix for coding endpoint
        auth:
          method: api_key
          source: keyring
        tags: [coding, premium]

    defaults:
      account: default
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Enums for Account Configuration
# =============================================================================


class AuthenticationMethod(str, Enum):
    """Authentication method for provider access.

    Determines how to authenticate with the provider:
    - API_KEY: API key authentication
    - OAUTH: OAuth 2.0 flow
    - NONE: No authentication required
    """

    API_KEY = "api_key"  # API key authentication
    OAUTH = "oauth"  # OAuth 2.0 flow
    NONE = "none"  # No authentication


class CredentialSource(str, Enum):
    """Source for credentials.

    Determines where credentials are stored/retrieved:
    - KEYRING: System keyring (most secure)
    - ENV: Environment variable (for CI/CD)
    - FILE: Config file (least secure, not recommended for API keys)
    """

    KEYRING = "keyring"  # System keyring
    ENV = "env"  # Environment variable
    FILE = "file"  # Config file
    SENTINELPASS = "sentinelpass"  # SentinelPass local vault lookup
    CODEX = "codex"  # Read OpenAI OAuth tokens from Codex CLI auth.json
    CLAUDE_CODE = "claude-code"  # Read Anthropic OAuth tokens from Claude Code


# =============================================================================
# Account Configuration Models
# =============================================================================


@dataclass
class AuthConfig:
    """Authentication configuration for a provider account.

    Supports three authentication methods:
    - api_key: Traditional API key authentication
    - oauth: OAuth flow (for OpenAI, Qwen)
    - none: No authentication required (local providers like Ollama)

    The source determines where credentials are stored:
    - keyring: System keyring (most secure)
    - env: Environment variable (for CI/CD)
    - file: ~/.victor/config.yaml (least secure, not recommended for API keys)
    """

    method: AuthenticationMethod = AuthenticationMethod.API_KEY
    source: CredentialSource = CredentialSource.KEYRING
    value: Optional[str] = None  # For client_id or explicit keys

    def is_secure(self) -> bool:
        """Check if this auth config uses secure storage."""
        return self.source == "keyring" or self.method == "none"

    def requires_api_key(self) -> bool:
        """Check if this auth requires an API key."""
        return self.method == "api_key"

    def requires_oauth(self) -> bool:
        """Check if this auth requires OAuth flow."""
        return self.method == "oauth"


@dataclass
class ProviderAccount:
    """A complete provider account with authentication and configuration.

    This unifies the previous separate concepts of:
    - Provider selection
    - Model selection
    - API key configuration
    - Endpoint configuration
    - Profile settings

    Attributes:
        provider: Provider name (anthropic, openai, zai, ollama, etc.)
        model: Model identifier (claude-sonnet-4-5, gpt-4o, glm-4.6:coding)
        auth: Authentication configuration
        endpoint: Custom endpoint URL (optional, overrides default)
        name: Account name for identification (default: "default")
        tags: Tags for categorization (chat, coding, premium, local, etc.)
        temperature: Default temperature for generation
        max_tokens: Default max tokens for generation
    """

    provider: str
    model: str
    auth: AuthConfig
    endpoint: Optional[str] = None
    name: str = "default"
    tags: List[str] = field(default_factory=list)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    # Additional provider-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def is_local(self) -> bool:
        """Check if this is a local provider (no API key required)."""
        return self.provider in {"ollama", "lmstudio", "vllm"}

    def is_oauth_enabled(self) -> bool:
        """Check if this account uses OAuth authentication."""
        return self.auth.method == "oauth"

    def get_endpoint_variant(self) -> Optional[str]:
        """Extract endpoint variant from model suffix (e.g., 'coding' from 'glm-4.6:coding')."""
        if ":" in self.model:
            parts = self.model.rsplit(":", 1)
            if len(parts) == 2:
                return parts[1]
        return None

    def get_base_model(self) -> str:
        """Get base model name without endpoint suffix."""
        if ":" in self.model:
            return self.model.rsplit(":", 1)[0]
        return self.model

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result = {
            "provider": self.provider,
            "model": self.model,
            "name": self.name,
        }

        # Add auth config
        if self.auth.method != "api_key" or self.auth.source != "keyring" or self.auth.value:
            result["auth"] = {
                "method": self.auth.method,
                "source": self.auth.source,
            }
            if self.auth.value:
                result["auth"]["value"] = self.auth.value

        # Add optional fields
        if self.endpoint:
            result["endpoint"] = self.endpoint
        if self.tags:
            result["tags"] = self.tags
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.extra_params:
            result["extra_params"] = self.extra_params

        return result

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ProviderAccount":
        """Create ProviderAccount from dictionary (YAML parsing)."""
        # Parse auth config
        auth_data = data.get("auth", {})
        if isinstance(auth_data, str):
            # Simple string format: "api_key", "oauth", or "none"
            auth = AuthConfig(method=auth_data, source="keyring")
        elif isinstance(auth_data, dict):
            auth = AuthConfig(
                method=auth_data.get("method", "api_key"),
                source=auth_data.get("source", "keyring"),
                value=auth_data.get("value"),
            )
        else:
            # Default auth
            auth = AuthConfig(method="api_key", source="keyring")

        return cls(
            name=name,
            provider=data["provider"],
            model=data["model"],
            auth=auth,
            endpoint=data.get("endpoint"),
            tags=data.get("tags", []),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            extra_params=data.get("extra_params", {}),
        )


@dataclass
class ConfigDefaults:
    """Default configuration settings."""

    account: str = "default"  # Default account to use
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result = {}
        if self.account != "default":
            result["account"] = self.account
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        return result


@dataclass
class VictorConfig:
    """Complete Victor configuration file.

    This represents the entire ~/.victor/config.yaml file structure.
    """

    accounts: Dict[str, ProviderAccount] = field(default_factory=dict)
    defaults: ConfigDefaults = field(default_factory=ConfigDefaults)

    def get_account(self, name: str) -> Optional[ProviderAccount]:
        """Get account by name."""
        return self.accounts.get(name)

    def add_account(self, account: ProviderAccount) -> None:
        """Add or update an account."""
        self.accounts[account.name] = account

    def remove_account(self, name: str) -> bool:
        """Remove account by name. Returns True if removed."""
        if name in self.accounts:
            del self.accounts[name]
            return True
        return False

    def list_accounts(self) -> List[ProviderAccount]:
        """Get all accounts as a list."""
        return list(self.accounts.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: Dict[str, Any] = {}

        if self.accounts:
            result["accounts"] = {
                name: account.to_dict() for name, account in self.accounts.items()
            }

        defaults_dict = self.defaults.to_dict()
        if defaults_dict:
            result["defaults"] = defaults_dict

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VictorConfig":
        """Create VictorConfig from dictionary (YAML parsing)."""
        # Parse accounts
        accounts_data = data.get("accounts", {})
        accounts = {
            name: ProviderAccount.from_dict(name, acc_data)
            for name, acc_data in accounts_data.items()
        }

        # Parse defaults
        defaults_data = data.get("defaults", {})
        defaults = ConfigDefaults(
            account=defaults_data.get("account", "default"),
            temperature=defaults_data.get("temperature"),
            max_tokens=defaults_data.get("max_tokens"),
        )

        return cls(accounts=accounts, defaults=defaults)


# =============================================================================
# Account Manager
# =============================================================================


class AccountManager:
    """Unified account and configuration management.

    This replaces the complex ProviderConfigRegistry + APIKeyManager + Settings
    combination with a simpler, more intuitive interface.

    Configuration Resolution Order:
    1. CLI flags (--provider, --model, --account, --endpoint)
    2. ~/.victor/config.yaml (accounts section)
    3. Environment variables (for CI/CD)
    4. System keyring

    Usage:
        # Get default account
        manager = AccountManager()
        account = manager.get_account()

        # Resolve provider config for use with providers
        config = manager.resolve_provider_config(account)

        # Save a new account
        account = ProviderAccount(
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key", source="keyring"),
            name="my-claude"
        )
        manager.save_account(account)
    """

    # Default endpoints for providers
    DEFAULT_ENDPOINTS: Dict[str, str] = {
        "anthropic": "https://api.anthropic.com",
        "openai": "https://api.openai.com/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta",
        "xai": "https://api.x.ai/v1",
        "moonshot": "https://api.moonshot.cn/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "zai": "https://api.z.ai/api/paas/v4/",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "groqcloud": "https://api.groq.com/openai/v1",
        "cerebras": "https://api.cerebras.ai/v1",
        "mistral": "https://api.mistral.ai/v1",
        "together": "https://api.together.xyz/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "fireworks": "https://api.fireworks.ai/inference/v1",
    }

    # OAuth-enabled providers
    OAUTH_PROVIDERS: Set[str] = {"openai", "qwen"}

    # Local providers (no API key required)
    LOCAL_PROVIDERS: Set[str] = {"ollama", "lmstudio", "vllm"}

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize AccountManager.

        Args:
            config_path: Path to config.yaml (default: ~/.victor/config.yaml)
        """
        self._config_path = config_path or self._get_default_config_path()
        self._config: Optional[VictorConfig] = None
        self._env_vars: Optional[Dict[str, str]] = None

    @staticmethod
    def _get_default_config_path() -> Path:
        """Get default config path with security."""
        try:
            from victor.config.secure_paths import get_victor_dir

            return get_victor_dir() / "config.yaml"
        except ImportError:
            return Path.home() / ".victor" / "config.yaml"

    @property
    def config_path(self) -> Path:
        """Get the config file path."""
        return self._config_path

    def load_config(self) -> VictorConfig:
        """Load configuration from file.

        Returns:
            Loaded VictorConfig, or empty config if file doesn't exist.
        """
        if self._config is not None:
            return self._config

        if not self._config_path.exists():
            logger.debug(f"Config file not found: {self._config_path}")
            self._config = VictorConfig()
            return self._config

        try:
            with open(self._config_path, "r") as f:
                data = yaml.safe_load(f) or {}
                self._config = VictorConfig.from_dict(data)
                logger.debug(f"Loaded config from {self._config_path}")
                return self._config
        except Exception as e:
            logger.warning(f"Failed to load config from {self._config_path}: {e}")
            self._config = VictorConfig()
            return self._config

    def save_config(self, config: Optional[VictorConfig] = None) -> None:
        """Save configuration to file.

        Args:
            config: Config to save (default: current config)
        """
        config_to_save = config or self._config
        if config_to_save is None:
            config_to_save = self.load_config()

        # Ensure parent directory exists
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write config with secure permissions
        with open(self._config_path, "w") as f:
            yaml.dump(config_to_save.to_dict(), f, default_flow_style=False, sort_keys=False)

        # Set secure permissions (0600)
        self._config_path.chmod(0o600)

        self._config = config_to_save
        logger.info(f"Saved config to {self._config_path}")

    def get_account(
        self,
        name: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[ProviderAccount]:
        """Get an account by name, or find matching account.

        Resolution order:
        1. Account name (if provided)
        2. Provider + model match
        3. Default account from config

        Args:
            name: Account name to lookup
            provider: Provider to match
            model: Model to match

        Returns:
            ProviderAccount if found, None otherwise
        """
        config = self.load_config()

        # 1. Lookup by name
        if name:
            account = config.get_account(name)
            if account:
                return account
            logger.debug(f"Account not found: {name}")
            return None

        # 2. Match by provider + model
        if provider and model:
            for account in config.list_accounts():
                if account.provider == provider and account.model == model:
                    return account
            logger.debug(f"No account found for {provider}/{model}")

        # 3. Return default account
        default_name = config.defaults.account
        return config.get_account(default_name)

    def resolve_provider_config(
        self,
        account: Optional[ProviderAccount] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Resolve provider configuration for use with provider adapters.

        This produces a configuration dict compatible with the existing
        provider system, containing api_key, base_url, model, etc.

        Args:
            account: Account to resolve (default: get default account)
            **kwargs: Override values (provider, model, endpoint, api_key, etc.)

        Returns:
            Provider configuration dict
        """
        # Get account
        if account is None:
            account = self.get_account(
                name=kwargs.get("account_name"),
                provider=kwargs.get("provider"),
                model=kwargs.get("model"),
            )

        if account is None:
            raise ValueError("No account found. Please run 'victor auth setup' to configure.")

        # Build config dict
        config = {
            "provider": kwargs.get("provider", account.provider),
            "model": kwargs.get("model", account.model),
            "base_model": account.get_base_model(),
        }

        # Resolve endpoint
        endpoint = kwargs.get("endpoint")
        if endpoint:
            config["base_url"] = endpoint
        elif account.endpoint:
            config["base_url"] = account.endpoint
        elif account.provider in self.DEFAULT_ENDPOINTS:
            config["base_url"] = self.DEFAULT_ENDPOINTS[account.provider]

        # Handle authentication
        auth_method = kwargs.get("auth_method", account.auth.method)

        if auth_method == "none":
            # Local provider, no auth needed
            pass
        elif auth_method == "oauth":
            # OAuth authentication
            config["auth_mode"] = "oauth"
            if account.auth.source in {"codex", "claude-code"}:
                config["oauth_source"] = account.auth.source
            # Get OAuth client_id from keyring
            client_id = self._get_oauth_client_id(account.provider)
            if client_id:
                config["oauth_client_id"] = client_id
        else:  # api_key
            # Try to get API key from multiple sources
            api_key = kwargs.get("api_key")

            if not api_key:
                # Account-scoped key FIRST: an explicitly-configured per-account key
                # (auth.value) is authoritative for THAT account. Provider-scoped env
                # vars and keyring are shared across every account of a provider, so
                # for multiple accounts under one provider (e.g. several anthropic-
                # dialect upstreams: kimi/deepseek/zai via /anthropic endpoints) the
                # shared slot would otherwise shadow — or be clobbered by — the
                # account's own key. auth.value also resolves non-interactively
                # (config file, not keyring), so `victor run` resolves it too.
                #
                # sentinelpass stores a lookup DOMAIN (not a key) in auth.value, so it
                # is excluded here and resolved via its own source path.
                if account.auth.value and account.auth.source != "sentinelpass":
                    api_key = account.auth.value

                # Provider-scoped environment variable.
                if not api_key:
                    api_key = self._get_api_key_from_env(account.provider)

                # Provider-scoped keyring.
                if not api_key and account.auth.source == "keyring":
                    api_key = self._get_api_key_from_keyring(account.provider)

            if api_key:
                config["api_key"] = api_key
            elif not account.is_local():
                logger.warning(f"No API key found for {account.provider}")

        # Add generation parameters (kwargs override account values)
        if "temperature" in kwargs:
            config["temperature"] = kwargs["temperature"]
        elif account.temperature is not None:
            config["temperature"] = account.temperature

        if "max_tokens" in kwargs:
            config["max_tokens"] = kwargs["max_tokens"]
        elif account.max_tokens is not None:
            config["max_tokens"] = account.max_tokens

        # Add extra params
        config.update(account.extra_params)

        return config

    def save_account(self, account: ProviderAccount) -> None:
        """Save or update an account.

        Args:
            account: Account to save
        """
        config = self.load_config()
        config.add_account(account)
        self.save_config(config)
        logger.info(f"Saved account: {account.name}")

    def remove_account(self, name: str) -> bool:
        """Remove an account by name.

        Args:
            name: Account name to remove

        Returns:
            True if removed, False if not found
        """
        config = self.load_config()
        removed = config.remove_account(name)
        if removed:
            self.save_config(config)
            logger.info(f"Removed account: {name}")
        return removed

    def list_accounts(self) -> List[ProviderAccount]:
        """List all configured accounts.

        Returns:
            List of all accounts
        """
        config = self.load_config()
        return config.list_accounts()

    def list_providers(self) -> List[str]:
        """Get list of unique configured providers.

        Returns:
            List of provider names
        """
        accounts = self.list_accounts()
        return sorted(set(acc.provider for acc in accounts))

    # ========================================================================
    # Private helper methods
    # ========================================================================

    def _get_api_key_from_env(self, provider: str) -> Optional[str]:
        """Get API key from environment variable."""
        import os

        env_var = self._get_provider_env_var(provider)
        if env_var:
            return os.environ.get(env_var)
        return None

    def _get_api_key_from_keyring(self, provider: str) -> Optional[str]:
        """Get API key from system keyring."""
        try:
            from victor.config.api_keys import _get_key_from_keyring

            return _get_key_from_keyring(provider)
        except ImportError:
            return None

    def _get_oauth_client_id(self, provider: str) -> Optional[str]:
        """Get OAuth client_id from keyring."""
        try:
            from victor.config.api_keys import _get_key_from_keyring

            return _get_key_from_keyring(f"{provider}_oauth_client_id")
        except ImportError:
            return None

    def _get_provider_env_var(self, provider: str) -> Optional[str]:
        """Get environment variable name for provider."""
        # Provider to env var mapping
        env_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "xai": "XAI_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "zai": "ZAI_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "groqcloud": "GROQCLOUD_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "together": "TOGETHER_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "fireworks": "FIREWORKS_API_KEY",
        }
        return env_vars.get(provider)

    def _check_migration_needed(self) -> bool:
        """Check if old configuration exists and migration is needed."""
        old_config_exists = (Path.home() / ".victor" / "profiles.yaml").exists() or (
            Path.home() / ".victor" / "api_keys.yaml"
        ).exists()
        new_config_exists = self._config_path.exists()
        return old_config_exists and not new_config_exists


# =============================================================================
# Singleton instance
# =============================================================================


_default_account_manager: Optional[AccountManager] = None


def get_account_manager() -> AccountManager:
    """Get the default AccountManager instance."""
    global _default_account_manager
    if _default_account_manager is None:
        _default_account_manager = AccountManager()
    return _default_account_manager


def reset_account_manager() -> None:
    """Reset the default AccountManager instance (for testing)."""
    global _default_account_manager
    _default_account_manager = None
