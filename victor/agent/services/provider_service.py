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

"""Provider service implementation.

Extracts provider management from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Provider initialization and configuration
- Provider switching with validation
- Provider health checks
- Provider capability discovery
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ProviderInfoImpl:
    """Implementation of provider information."""

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        api_key_configured: bool,
        base_url: Optional[str],
        supports_streaming: bool,
        supports_tool_calling: bool,
        max_tokens: int,
    ):
        self.provider_name = provider_name
        self.model_name = model_name
        self.api_key_configured = api_key_configured
        self.base_url = base_url
        self.supports_streaming = supports_streaming
        self.supports_tool_calling = supports_tool_calling
        self.max_tokens = max_tokens


class ProviderService:
    """[CANONICAL] Service for provider management.

    The target implementation for provider operations following the
    state-passed architectural pattern. Supersedes ProviderCoordinator.

    This service follows SOLID principles:
    - SRP: Only handles provider operations
    - OCP: Extensible through registry
    - LSP: Implements ProviderServiceProtocol
    - ISP: Focused interface
    - DIP: Depends on abstractions

    Example:
        service = ProviderService(registry=registry)
        await service.switch_provider('anthropic', 'claude-sonnet-4-5')
    """

    def __init__(
        self,
        registry: Any,
    ):
        """Initialize the provider service.

        Args:
            registry: Provider registry
        """
        self._registry = registry
        self._provider_manager: Optional[Any] = None
        self._current_provider: Optional["BaseProvider"] = None
        self._current_info: Optional[ProviderInfoImpl] = None
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

        # Track provider switches and rate limits
        self._switch_count: int = 0
        self._rate_limit_stats: Dict[str, Any] = {
            "rate_limits_hit": 0,
            "total_wait_time": 0.0,
            "last_rate_limit_time": None,
            "provider_rate_limits": {},
        }
        self._post_switch_hooks: List[callable] = []

    def bind_runtime_components(
        self,
        *,
        provider_manager: Optional[Any] = None,
    ) -> None:
        """Bind live runtime collaborators after bootstrap."""
        if provider_manager is not None:
            self._provider_manager = provider_manager
            self._sync_from_provider_manager()

    async def switch_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        validate: bool = True,
    ) -> None:
        """Switch to a different provider.

        Args:
            provider: Provider name
            model: Optional model name
            validate: If True, validate provider before switching
        """
        if self._provider_manager is not None and hasattr(
            self._provider_manager, "switch_provider"
        ):
            switched = await self._provider_manager.switch_provider(provider, model)
            if not switched:
                raise ValueError(f"Provider switch failed: {provider}")
            self._sync_from_provider_manager()
            self._notify_post_switch_hooks(provider, model)
            return

        self._logger.info(f"Switching to provider: {provider}")

        # Get provider from registry
        new_provider = self._registry.get_provider(provider)
        if new_provider is None:
            raise ValueError(f"Provider not found: {provider}")

        # Set model if specified
        if model:
            new_provider.set_model(model)

        # Update current provider
        self._current_provider = new_provider
        self._current_info = self._create_provider_info(new_provider)

        # Increment switch count
        self._switch_count += 1

        # Call post-switch hooks
        self._notify_post_switch_hooks(provider, model)

        self._logger.info(f"Switched to provider: {provider} (switch #{self._switch_count})")

    async def switch_model(self, model: str) -> None:
        """Switch models on the current provider."""
        if self._provider_manager is not None and hasattr(self._provider_manager, "switch_model"):
            switched = await self._provider_manager.switch_model(model)
            if not switched:
                raise ValueError(f"Model switch failed: {model}")
            self._sync_from_provider_manager()
            self._notify_post_switch_hooks(self.provider_name, model)
            return

        if self.provider_name in {"", "none"}:
            raise ValueError("No provider is currently configured")

        await self.switch_provider(self.provider_name, model)

    def get_current_provider_info(self) -> ProviderInfoImpl:
        """Get current provider information.

        Returns:
            ProviderInfo with current provider details
        """
        if self._provider_manager is not None:
            self._sync_from_provider_manager()

        if self._current_info is None:
            # Create info for current provider
            if self._current_provider:
                self._current_info = self._create_provider_info(self._current_provider)
            else:
                # Return default info
                self._current_info = ProviderInfoImpl(
                    provider_name="none",
                    model_name="none",
                    api_key_configured=False,
                    base_url=None,
                    supports_streaming=False,
                    supports_tool_calling=False,
                    max_tokens=0,
                )

        return self._current_info

    async def check_provider_health(
        self,
        provider: Optional[str] = None,
    ) -> bool:
        """Check if a provider is healthy.

        Args:
            provider: Provider name to check, or None for current

        Returns:
            True if provider is healthy, False otherwise
        """
        if provider:
            target_provider = self._registry.get_provider(provider)
        else:
            target_provider = self._current_provider

        if target_provider is None:
            return False

        # Default health check: try to get model
        try:
            return target_provider.get_model() is not None
        except Exception:
            return False

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._provider_manager is not None and hasattr(
            self._provider_manager, "start_health_monitoring"
        ):
            await self._provider_manager.start_health_monitoring()
        elif hasattr(self._registry, "start_health_monitoring"):
            await self._registry.start_health_monitoring()

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        if self._provider_manager is not None and hasattr(
            self._provider_manager, "stop_health_monitoring"
        ):
            await self._provider_manager.stop_health_monitoring()
        elif hasattr(self._registry, "stop_health_monitoring"):
            await self._registry.stop_health_monitoring()

    def get_available_providers(self) -> List[str]:
        """Get list of available providers.

        Returns:
            List of provider names
        """
        return self._registry.list_providers()

    async def get_provider_capabilities(
        self,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get provider capabilities.

        Args:
            provider: Provider name, or None for current

        Returns:
            Dictionary with capability information
        """
        if provider:
            target_provider = self._registry.get_provider(provider)
        else:
            target_provider = self._current_provider

        if target_provider is None:
            return {}

        # Get capabilities from provider
        return {
            "streaming": getattr(target_provider, "supports_streaming", True),
            "tools": getattr(target_provider, "supports_tools", True),
            "max_tokens": getattr(target_provider, "max_tokens", 100000),
        }

    def get_current_provider(self) -> "BaseProvider":
        """Get the current provider instance.

        Returns:
            Current provider instance

        Raises:
            ValueError: If no provider is configured
        """
        if self._provider_manager is not None:
            self._sync_from_provider_manager()

        if self._current_provider is None:
            raise ValueError("No provider is currently configured")

        return self._current_provider

    async def test_provider(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> bool:
        """Test a provider with a simple request.

        Args:
            provider: Provider name to test
            model: Optional model name

        Returns:
            True if test succeeded, False otherwise
        """
        try:
            target_provider = self._registry.get_provider(provider)
            if target_provider is None:
                return False

            if model:
                target_provider.set_model(model)

            # Simple health check
            return await self.check_provider_health(provider)

        except Exception:
            return False

    def get_rate_limit_wait_time(self, error: Exception) -> float:
        """Calculate wait time from a rate limit error.

        Parses common rate limit error formats to extract wait time.
        Falls back to default wait time (60s) if parsing fails.

        Args:
            error: The rate limit exception

        Returns:
            Wait time in seconds
        """
        import re

        # Check for retry_after attribute (e.g., ProviderRateLimitError)
        if hasattr(error, "retry_after") and error.retry_after is not None:
            try:
                return float(error.retry_after)
            except (ValueError, TypeError):
                pass

        error_str = str(error)

        # Try to extract retry-after from common patterns
        patterns = [
            r"try\s+again\s+in\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)?",
            r"retry\s+after\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)?",
            r"wait\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)?",
            r"(\d+(?:\.\d+)?)\s*(?:seconds?|s)\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        # Check for Retry-After header in error message
        if hasattr(error, "response"):
            response = getattr(error, "response", None)
            if response and hasattr(response, "headers"):
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        return float(retry_after)
                    except ValueError:
                        pass

        return 60.0

    def is_healthy(self) -> bool:
        """Check if the provider service is healthy.

        Returns:
            True if the service is healthy
        """
        if self._provider_manager is not None:
            return getattr(self._provider_manager, "provider", None) is not None
        return self._current_provider is not None

    # ==========================================================================
    # Provider Properties
    # ==========================================================================

    @property
    def provider(self) -> Optional["BaseProvider"]:
        """Get the current provider instance.

        Returns the current provider instance, or None if no provider is set.

        Returns:
            Current provider instance or None

        Example:
            provider = service.provider
            if provider:
                print(f"Using: {provider.name}")
        """
        if self._provider_manager is not None:
            return getattr(self._provider_manager, "provider", None)
        return self._current_provider

    @property
    def model(self) -> str:
        """Get the current model name.

        Returns the model name of the current provider, or "none" if no provider is set.

        Returns:
            Model name string

        Example:
            model = service.model
            print(f"Using model: {model}")
        """
        if self._provider_manager is not None:
            return getattr(self._provider_manager, "model", "none")
        info = self.get_current_provider_info()
        return info.model_name if info else "none"

    @property
    def provider_name(self) -> str:
        """Get the current provider name.

        Returns the provider name, or "none" if no provider is set.

        Returns:
            Provider name string

        Example:
            name = service.provider_name
            print(f"Using provider: {name}")
        """
        if self._provider_manager is not None:
            return getattr(self._provider_manager, "provider_name", "none")
        info = self.get_current_provider_info()
        return info.provider_name if info else "none"

    @property
    def tool_adapter(self) -> Optional[Any]:
        """Get the tool adapter for the current provider.

        Returns the tool adapter instance if available, or None.

        Returns:
            Tool adapter instance or None

        Example:
            adapter = service.tool_adapter
            if adapter:
                tool_calls = adapter.parse_tool_calls(response)
        """
        if self._provider_manager is not None:
            return getattr(self._provider_manager, "tool_adapter", None)
        if self._current_provider is None:
            return None

        # Try to get tool adapter from provider
        return getattr(self._current_provider, "tool_adapter", None)

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Get the current provider's capabilities.

        Returns a dictionary with provider capabilities including
        streaming support, tool support, max tokens, etc.

        Returns:
            Dictionary with capability information

        Example:
            caps = service.capabilities
            print(f"Streaming: {caps['streaming']}")
            print(f"Max tokens: {caps['max_tokens']}")
        """
        if self._provider_manager is not None:
            capabilities = getattr(self._provider_manager, "capabilities", None)
            if capabilities is None:
                return {
                    "streaming": False,
                    "tools": False,
                    "max_tokens": 0,
                }
            return {
                "streaming": getattr(capabilities, "streaming_tool_calls", False),
                "tools": getattr(capabilities, "native_tool_calls", False),
                "max_tokens": (getattr(self.provider, "max_tokens", 0) if self.provider else 0),
            }

        # Use cached info if available, otherwise get from provider
        if self._current_info:
            info = self._current_info
            return {
                "streaming": info.supports_streaming,
                "tools": info.supports_tool_calling,
                "max_tokens": info.max_tokens,
            }

        # Fallback to async method (returns empty dict if no provider)
        return {
            "streaming": False,
            "tools": False,
            "max_tokens": 0,
        }

    # ==========================================================================
    # Switch Tracking and Rate Limiting
    # ==========================================================================

    @property
    def switch_count(self) -> int:
        """Get the number of provider switches.

        Returns:
            Number of times the provider has been switched

        Example:
            count = service.switch_count
            print(f"Provider switched {count} times")
        """
        if self._provider_manager is not None:
            return getattr(self._provider_manager, "switch_count", 0)
        return self._switch_count

    def track_rate_limit(self, provider_name: str, wait_time: float) -> None:
        """Track a rate limit event for a provider.

        Args:
            provider_name: Name of the provider that hit rate limit
            wait_time: Wait time in seconds

        Example:
            service.track_rate_limit("anthropic", 60.0)
        """
        from datetime import datetime

        self._rate_limit_stats["rate_limits_hit"] += 1
        self._rate_limit_stats["total_wait_time"] += wait_time
        self._rate_limit_stats["last_rate_limit_time"] = datetime.now().isoformat()

        # Track per-provider stats
        if provider_name not in self._rate_limit_stats["provider_rate_limits"]:
            self._rate_limit_stats["provider_rate_limits"][provider_name] = {
                "count": 0,
                "total_wait_time": 0.0,
            }

        provider_stats = self._rate_limit_stats["provider_rate_limits"][provider_name]
        provider_stats["count"] += 1
        provider_stats["total_wait_time"] += wait_time

        self._logger.debug(
            f"Tracked rate limit for {provider_name}: {wait_time}s "
            f"(total: {self._rate_limit_stats['rate_limits_hit']} events)"
        )

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limit statistics.

        Returns:
            Dictionary with rate limit statistics:
            - rate_limits_hit: Total number of rate limit events
            - total_wait_time: Total wait time across all rate limits
            - last_rate_limit_time: ISO timestamp of last rate limit
            - provider_rate_limits: Per-provider breakdown

        Example:
            stats = service.get_rate_limit_stats()
            print(f"Rate limits hit: {stats['rate_limits_hit']}")
        """
        return dict(self._rate_limit_stats)

    def register_post_switch_hook(self, hook: callable) -> None:
        """Register a callback to be called after provider switches.

        The callback will be called with (provider_name, model_name) arguments
        after each successful provider switch.

        Args:
            hook: Callable that accepts (provider_name, model_name)

        Example:
            def on_switch(provider, model):
                print(f"Switched to {provider} / {model}")

            service.register_post_switch_hook(on_switch)
        """
        if hook not in self._post_switch_hooks:
            self._post_switch_hooks.append(hook)
            self._logger.debug(f"Registered post-switch hook: {hook}")

    def _notify_post_switch_hooks(self, provider_name: str, model: Optional[str]) -> None:
        """Notify all registered post-switch hooks.

        Args:
            provider_name: Name of the provider switched to
            model: Model name (if any)
        """
        for hook in self._post_switch_hooks:
            try:
                hook(provider_name, model)
            except Exception as e:
                self._logger.warning(f"Post-switch hook failed: {e}")

    def _sync_from_provider_manager(self) -> None:
        """Refresh cached provider info from the live provider manager."""
        manager = self._provider_manager
        if manager is None:
            return

        provider = getattr(manager, "provider", None)
        self._current_provider = provider
        if provider is None:
            self._current_info = None
            return

        info = self._create_provider_info(provider)
        info.provider_name = getattr(manager, "provider_name", info.provider_name)
        info.model_name = getattr(manager, "model", info.model_name)
        self._current_info = info

    def _create_provider_info(self, provider: "BaseProvider") -> ProviderInfoImpl:
        """Create provider info from provider instance.

        Args:
            provider: Provider instance

        Returns:
            ProviderInfo with provider details
        """
        return ProviderInfoImpl(
            provider_name=getattr(provider, "name", "unknown"),
            model_name=getattr(provider, "model", "unknown"),
            api_key_configured=getattr(provider, "api_key", None) is not None,
            base_url=getattr(provider, "base_url", None),
            supports_streaming=getattr(provider, "supports_streaming", True),
            supports_tool_calling=getattr(provider, "supports_tools", True),
            max_tokens=getattr(provider, "max_tokens", 100000),
        )
