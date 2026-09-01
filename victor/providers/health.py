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
Provider Health Check API.

This module provides pre-flight health checks for provider configuration:
- Check if provider is registered
- Check if API key is available
- Validate API key format
- Optional connectivity test (without consuming tokens)

Usage:
    from victor.providers.health import ProviderHealthChecker

    checker = ProviderHealthChecker()
    result = await checker.check_provider(
        provider="deepseek",
        model="deepseek-chat",
        check_connectivity=False,
    )

    if not result.healthy:
        print(result.error_message)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.core.health import HealthStatus
from victor.providers.registry import ProviderRegistry
from victor.providers.resolution import (
    UnifiedApiKeyResolver,
    APIKeyNotFoundError,
    APIKeyResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ProviderHealthResult:
    """Result of provider health check."""

    healthy: bool
    provider: str
    model: str
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)
    status: str = HealthStatus.UNKNOWN

    def __post_init__(self):
        """Set status based on healthy flag."""
        if self.status == HealthStatus.UNKNOWN:
            self.status = HealthStatus.HEALTHY if self.healthy else HealthStatus.UNHEALTHY

    @property
    def error_message(self) -> str:
        """Get formatted error message."""
        if self.healthy:
            return "Provider is healthy"
        return "\n".join(self.issues)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "healthy": self.healthy,
            "provider": self.provider,
            "model": self.model,
            "issues": self.issues,
            "warnings": self.warnings,
            "info": self.info,
            "status": self.status.upper(),  # Convert to uppercase for API compatibility
        }


class ProviderHealthChecker:
    """
    Pre-flight health checks for provider configuration.

    Checks:
    1. Provider is registered
    2. API key is available (or not required)
    3. API key format is valid
    4. Optional: Connectivity test (actual API call)

    Usage:
        checker = ProviderHealthChecker()

        # Fast check (no API calls)
        result = await checker.check_provider("deepseek", "deepseek-chat")

        # Thorough check (with API call)
        result = await checker.check_provider(
            "deepseek",
            "deepseek-chat",
            check_connectivity=True,
        )
    """

    # API key format patterns (relaxed for flexibility)
    KEY_PATTERNS = {
        "anthropic": r"^sk-ant-[a-zA-Z0-9_-]{40,}$",  # More flexible length
        "openai": r"^sk-[a-zA-Z0-9]{20,}$",  # More flexible
        "deepseek": r"^sk-[a-zA-Z0-9]{10,}$",  # More flexible
        "google": r"^.{10,}$",  # Google keys vary
        "xai": r"^xai-[a-zA-Z0-9]{20,}$",  # More flexible
    }

    # Providers that don't need API keys
    LOCAL_PROVIDERS = {"ollama", "lmstudio", "vllm", "inferflux"}

    def __init__(self, non_interactive: Optional[bool] = None):
        """Initialize health checker.

        Args:
            non_interactive: Force non-interactive mode for resolver
        """
        self.resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
        self._registered_providers: Dict[str, Any] = {}
        self._provider_health: Dict[str, ProviderHealthResult] = {}

    async def check_provider(
        self,
        provider: str,
        model: str,
        check_connectivity: bool = False,
        timeout: float = 5.0,
        **kwargs: Any,
    ) -> ProviderHealthResult:
        """
        Check if provider is properly configured.

        Args:
            provider: Provider name
            model: Model to check
            check_connectivity: Make actual API call (slower but thorough)
            timeout: Timeout for connectivity check
            **kwargs: Additional provider arguments

        Returns:
            ProviderHealthResult with status and actionable issues
        """
        provider = provider.lower()
        issues: List[str] = []
        warnings: List[str] = []
        info: Dict[str, Any] = {}

        # Check 1: Provider registration
        if not self._check_provider_registered(provider):
            issues.append(
                f"Provider '{provider}' is not registered. "
                f"Available providers: {self._list_available_providers()}"
            )
            return ProviderHealthResult(
                healthy=False,
                provider=provider,
                model=model,
                issues=issues,
                warnings=warnings,
                info=info,
            )

        info["registered"] = True
        logger.debug(f"Provider '{provider}' is registered")

        # Check 2: API key availability
        key_result = self._check_api_key(provider, kwargs.get("api_key"))

        if key_result.key is None and provider not in self.LOCAL_PROVIDERS:
            # Build detailed error with sources attempted
            error = APIKeyNotFoundError(
                provider=provider,
                sources_attempted=key_result.sources_attempted,
                non_interactive=key_result.non_interactive,
                model=model,
            )
            issues.append(str(error))
            info["key_sources_attempted"] = [
                {
                    "source": s.source,
                    "description": s.description,
                    "found": s.found,
                }
                for s in key_result.sources_attempted
            ]
        else:
            info["key_source"] = key_result.source_detail
            logger.debug(f"Provider '{provider}' API key found: {key_result.source_detail}")

        # Check 3: API key format (if key exists)
        if key_result.key and provider in self.KEY_PATTERNS:
            if not self._validate_key_format(provider, key_result.key):
                issues.append(
                    f"API key format validation failed for '{provider}'. "
                    f"Expected format: {self._get_format_description(provider)}"
                )
            else:
                info["key_format_valid"] = True
                logger.debug(f"Provider '{provider}' API key format valid")

        # Check 4: Connectivity (optional)
        if check_connectivity and provider not in self.LOCAL_PROVIDERS:
            connectivity_result = await self._check_connectivity(
                provider, model, key_result.key, timeout, **kwargs
            )
            if not connectivity_result["success"]:
                issues.append(connectivity_result.get("error", "Connectivity check failed"))
            else:
                info["connectivity"] = "OK"
                logger.debug(f"Provider '{provider}' connectivity check passed")

        # Warnings
        if key_result.key and key_result.source == "keyring":
            if key_result.non_interactive:
                warnings.append(
                    f"Using keychain for API key in non-interactive mode. "
                    f"This may block background jobs. "
                    f"Set {self._get_env_var(provider)} environment variable instead."
                )

        result = ProviderHealthResult(
            healthy=len(issues) == 0,
            provider=provider,
            model=model,
            issues=issues,
            warnings=warnings,
            info=info,
        )

        # Cache the result
        self._provider_health[provider] = result

        return result

    def _check_provider_registered(self, provider: str) -> bool:
        """Check if provider is registered."""
        try:
            ProviderRegistry.get(provider)
            return True
        except Exception:
            return False

    def _list_available_providers(self) -> List[str]:
        """Get list of available providers."""
        try:
            return ProviderRegistry.list_providers()
        except Exception:
            return []

    def _check_api_key(
        self,
        provider: str,
        explicit_key: Optional[str],
    ) -> APIKeyResult:
        """Check if API key is available."""
        return self.resolver.get_api_key(provider, explicit_key=explicit_key)

    def _validate_key_format(self, provider: str, key: str) -> bool:
        """Validate API key format."""
        pattern = self.KEY_PATTERNS.get(provider)
        if not pattern:
            return True  # No pattern defined, skip validation
        return bool(re.match(pattern, key))

    def _get_format_description(self, provider: str) -> str:
        """Get human-readable format description."""
        descriptions = {
            "anthropic": "sk-ant- followed by 95+ characters",
            "openai": "sk- followed by 48+ characters",
            "deepseek": "sk- followed by 20+ characters",
            "xai": "xai- followed by 40+ characters",
        }
        return descriptions.get(provider, "valid API key format")

    def _get_env_var(self, provider: str) -> Optional[str]:
        """Get environment variable name for provider."""
        from victor.providers.resolution import _get_provider_env_var

        return _get_provider_env_var(provider)

    async def _check_connectivity(
        self,
        provider: str,
        model: str,
        api_key: str,
        timeout: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Check provider connectivity with minimal API call.

        Makes a minimal API call to verify the provider is accessible
        and the API key is valid.

        Args:
            provider: Provider name
            model: Model to test
            api_key: API key to use
            timeout: Timeout for check
            **kwargs: Additional arguments

        Returns:
            Dict with 'success' and optional 'error' key
        """
        try:
            # Create provider instance
            provider_instance = ProviderRegistry.create(
                provider,
                api_key=api_key,
                timeout=timeout,
                **kwargs,
            )

            # Make minimal test call
            from victor.providers.base import Message

            test_message = Message(role="user", content="Hi")

            # Try to make a simple call (most providers will fail or succeed quickly)
            try:
                # asyncio.wait_for (not asyncio.timeout, which is 3.11+) for
                # Python 3.10 compatibility.
                await asyncio.wait_for(
                    provider_instance.chat(
                        messages=[test_message],
                        model=model,
                        max_tokens=1,
                    ),
                    timeout=timeout,
                )
                return {"success": True, "response": "Connectivity OK"}
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error": f"Connectivity check timed out after {timeout}s",
                }
            except Exception as e:
                # Some providers may return auth errors which is also useful info
                error_str = str(e).lower()
                if any(term in error_str for term in ["auth", "unauthorized", "invalid key"]):
                    return {"success": False, "error": f"Authentication failed: {e}"}
                # Other errors still indicate connectivity (just not a successful call)
                return {
                    "success": True,
                    "warning": f"Provider responded with error (but is reachable): {e}",
                }

        except Exception as e:
            return {"success": False, "error": f"Failed to create provider: {e}"}

    def register_provider(self, provider_name: str, provider_instance: Any) -> None:
        """Register a provider instance for health tracking.

        Args:
            provider_name: Name of the provider
            provider_instance: Provider instance to track
        """
        self._registered_providers[provider_name] = provider_instance
        logger.debug(f"Registered provider '{provider_name}' for health monitoring")

    def unregister_provider(self, provider_name: str) -> None:
        """Unregister a provider from health tracking.

        Args:
            provider_name: Name of the provider to unregister
        """
        self._registered_providers.pop(provider_name, None)
        self._provider_health.pop(provider_name, None)
        logger.debug(f"Unregistered provider '{provider_name}' from health monitoring")

    def get_healthy_providers(self) -> List[str]:
        """Get list of healthy providers.

        Returns:
            List of provider names that are currently healthy
        """
        healthy = [
            name
            for name, result in self._provider_health.items()
            if result.healthy or result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        ]
        return healthy

    def get_provider_health(self, provider_name: str) -> Optional[ProviderHealthResult]:
        """Get cached health result for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            ProviderHealthResult if available, None otherwise
        """
        return self._provider_health.get(provider_name)


async def check_provider_health(
    provider: str,
    model: str,
    check_connectivity: bool = False,
    timeout: float = 5.0,
    **kwargs: Any,
) -> ProviderHealthResult:
    """
    Convenience function to check provider health.

    Args:
        provider: Provider name
        model: Model to check
        check_connectivity: Make actual API call
        timeout: Timeout for connectivity check
        **kwargs: Additional provider arguments

    Returns:
        ProviderHealthResult
    """
    checker = ProviderHealthChecker()
    return await checker.check_provider(
        provider=provider,
        model=model,
        check_connectivity=check_connectivity,
        timeout=timeout,
        **kwargs,
    )
