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

"""Simplified provider configuration resolution.

This module replaces the complex ProviderConfigRegistry with a simpler
resolution system that respects the following priority order:

1. CLI flags (--provider, --model, --account, --endpoint, --api-key)
2. ~/.victor/config.yaml (accounts section)
3. Environment variables (for CI/CD)
4. System keyring (secure storage)

The resolver integrates with the new AccountManager to provide a clean
interface for getting provider configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from victor.config.accounts import (
    AccountManager,
    AuthConfig,
    ProviderAccount,
    get_account_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Resolution Context
# =============================================================================


@dataclass
class ResolutionContext:
    """Context for provider configuration resolution.

    This captures all the inputs that influence resolution:
    - CLI flags and arguments
    - Environment variables
    - Config file settings
    - Keyring values
    """

    # CLI-provided values (highest priority)
    cli_provider: Optional[str] = None
    cli_model: Optional[str] = None
    cli_account: Optional[str] = None
    cli_endpoint: Optional[str] = None
    cli_api_key: Optional[str] = None
    cli_auth_mode: Optional[str] = None
    cli_temperature: Optional[float] = None
    cli_max_tokens: Optional[int] = None

    # Additional CLI parameters
    cli_extra_params: Dict[str, Any] = None

    # Working directory (for project-local config)
    working_dir: Optional[Path] = None

    def __post_init__(self):
        if self.cli_extra_params is None:
            self.cli_extra_params = {}

    def has_cli_overrides(self) -> bool:
        """Check if any CLI overrides are set."""
        return any(
            [
                self.cli_provider,
                self.cli_model,
                self.cli_account,
                self.cli_endpoint,
                self.cli_api_key,
                self.cli_auth_mode,
            ]
        )

    def get_overrides(self) -> Dict[str, Any]:
        """Get all CLI overrides as a dict."""
        overrides = {}

        if self.cli_provider:
            overrides["provider"] = self.cli_provider
        if self.cli_model:
            overrides["model"] = self.cli_model
        if self.cli_endpoint:
            overrides["endpoint"] = self.cli_endpoint
        if self.cli_api_key:
            overrides["api_key"] = self.cli_api_key
        if self.cli_auth_mode:
            overrides["auth_method"] = self.cli_auth_mode
        if self.cli_temperature is not None:
            overrides["temperature"] = self.cli_temperature
        if self.cli_max_tokens is not None:
            overrides["max_tokens"] = self.cli_max_tokens

        overrides.update(self.cli_extra_params)
        return overrides


# =============================================================================
# Provider Resolver
# =============================================================================


class ProviderResolver:
    """Simplified provider configuration resolver.

    This replaces the complex ProviderConfigRegistry with a straightforward
    resolution process that respects the priority order.

    Usage:
        resolver = ProviderResolver()

        # Resolve from context
        config = resolver.resolve(context)

        # Quick resolve with minimal params
        config = resolver.resolve_quick(provider="anthropic", model="claude-sonnet-4-5")
    """

    def __init__(self, account_manager: Optional[AccountManager] = None):
        """Initialize resolver.

        Args:
            account_manager: AccountManager instance (default: singleton)
        """
        self._account_manager = account_manager or get_account_manager()

    @property
    def account_manager(self) -> AccountManager:
        """Get the account manager."""
        return self._account_manager

    def resolve(self, context: Optional[ResolutionContext] = None) -> Dict[str, Any]:
        """Resolve provider configuration from context.

        Resolution order:
        1. CLI flags (from context)
        2. Config file (via AccountManager)
        3. Environment variables
        4. System keyring

        Args:
            context: Resolution context with CLI overrides

        Returns:
            Provider configuration dict

        Raises:
            ValueError: If no provider/account can be resolved
        """
        if context is None:
            context = ResolutionContext()

        # Get CLI overrides
        overrides = context.get_overrides()

        # Resolve account
        account_name = context.cli_account
        account = self._account_manager.get_account(
            name=account_name,
            provider=context.cli_provider,
            model=context.cli_model,
        )

        # If no account found and no CLI provider, try to detect
        if account is None and not context.cli_provider:
            account = self._detect_local_provider()

        # Use account if found, otherwise create from CLI overrides
        if account is None:
            if not context.cli_provider:
                raise ValueError(
                    "No provider configured. Run 'victor auth setup' to configure "
                    "or specify --provider flag."
                )
            # Create ad-hoc account from CLI overrides
            account = self._create_account_from_cli(context)

        # Resolve provider config
        config = self._account_manager.resolve_provider_config(
            account=account,
            account_name=context.cli_account,
            **overrides,
        )

        return config

    def resolve_quick(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Quick resolution with common parameters.

        .. deprecated::
            Use Settings.get_provider_settings() instead, which routes through
            ProviderConfigRegistry for correct provider-specific logic.

        Args:
            provider: Provider name
            model: Model name
            api_key: API key (optional)
            endpoint: Custom endpoint (optional)
            **kwargs: Additional parameters

        Returns:
            Provider configuration dict
        """
        context = ResolutionContext(
            cli_provider=provider,
            cli_model=model,
            cli_api_key=api_key,
            cli_endpoint=endpoint,
            cli_extra_params=kwargs,
        )
        return self.resolve(context)

    def resolve_for_orchestrator(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        account: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Resolve configuration for orchestrator use.

        This is the main entry point for the orchestrator to get
        provider configuration.

        Args:
            provider: Provider name (optional)
            model: Model name (optional)
            account: Account name (optional)
            **kwargs: Additional parameters

        Returns:
            Provider configuration dict
        """
        context = ResolutionContext(
            cli_provider=provider,
            cli_model=model,
            cli_account=account,
            cli_extra_params=kwargs,
        )
        return self.resolve(context)

    # ========================================================================
    # Private helper methods
    # ========================================================================

    def _detect_local_provider(self) -> Optional[ProviderAccount]:
        """Detect if a local provider is available.

        Checks for Ollama, LM Studio, or vLLM in order.

        Returns:
            ProviderAccount if detected, None otherwise
        """
        # Try Ollama
        if self._check_ollama_available():
            return ProviderAccount(
                name="ollama-detected",
                provider="ollama",
                model="llama3.2",  # Default model
                auth=AuthConfig(method="none"),
                tags=["local", "detected"],
            )

        # Try LM Studio
        if self._check_lmstudio_available():
            return ProviderAccount(
                name="lmstudio-detected",
                provider="lmstudio",
                model="local-model",  # Will be detected by provider
                auth=AuthConfig(method="none"),
                tags=["local", "detected"],
            )

        return None

    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available."""
        import subprocess
        import socket

        # Check if ollama command exists
        try:
            subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

        # Check if API is accessible
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", 11434))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _check_lmstudio_available(self) -> bool:
        """Check if LM Studio is available."""
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", 1234))
            sock.close()
            return result == 0
        except Exception:
            return False

    def _create_account_from_cli(self, context: ResolutionContext) -> ProviderAccount:
        """Create ad-hoc account from CLI context."""
        if not context.cli_provider:
            raise ValueError("Provider is required")

        # Determine auth method
        auth_method = "api_key"
        if context.cli_provider in AccountManager.LOCAL_PROVIDERS:
            auth_method = "none"
        elif context.cli_auth_mode == "oauth":
            auth_method = "oauth"

        # Determine auth source
        auth_source = "env"  # Default to env for CLI-specified providers
        if context.cli_api_key:
            auth_source = "file"  # Explicit key provided

        return ProviderAccount(
            name="cli-ad-hoc",
            provider=context.cli_provider,
            model=context.cli_model or "default",
            auth=AuthConfig(method=auth_method, source=auth_source, value=context.cli_api_key),
            endpoint=context.cli_endpoint,
            tags=["cli", "ad-hoc"],
        )


# =============================================================================
# Convenience Functions
# =============================================================================


_default_resolver: Optional[ProviderResolver] = None


def get_provider_resolver() -> ProviderResolver:
    """Get the default ProviderResolver instance."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = ProviderResolver()
    return _default_resolver


def resolve_provider_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    account: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Resolve provider configuration (convenience function).

    This is the main entry point for getting provider configuration.

    Args:
        provider: Provider name
        model: Model name
        account: Account name
        **kwargs: Additional parameters

    Returns:
        Provider configuration dict
    """
    resolver = get_provider_resolver()
    return resolver.resolve_for_orchestrator(
        provider=provider,
        model=model,
        account=account,
        **kwargs,
    )


def reset_provider_resolver() -> None:
    """Reset the default resolver instance (for testing)."""
    global _default_resolver
    _default_resolver = None
