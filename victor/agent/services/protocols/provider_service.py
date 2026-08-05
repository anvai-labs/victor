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

"""Provider service protocol.

Defines the interface for provider management operations.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.providers.base import (
        BaseProvider,
        CompletionResponse,
        StreamChunk,
    )


@runtime_checkable
class ProviderInfo(Protocol):
    """Information about a provider configuration.

    Provides metadata about the current provider state.
    """

    @property
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'anthropic', 'openai')."""
        ...

    @property
    def model_name(self) -> str:
        """Name of the model."""
        ...

    @property
    def api_key_configured(self) -> bool:
        """Whether API key is configured."""
        ...

    @property
    def base_url(self) -> Optional[str]:
        """Base URL for API endpoints."""
        ...

    @property
    def supports_streaming(self) -> bool:
        """Whether provider supports streaming."""
        ...

    @property
    def supports_tool_calling(self) -> bool:
        """Whether provider supports function calling."""
        ...

    @property
    def max_tokens(self) -> int:
        """Maximum tokens supported by model."""
        ...


@runtime_checkable
class ProviderServiceProtocol(Protocol):
    """[CANONICAL] Protocol for provider management service.

    This protocol represents the target architecture for provider operations,
    replacing the facade-driven Coordinator pattern with a state-passed
    Service pattern.

    This protocol follows the Interface Segregation Principle (ISP)
    by focusing only on provider-related operations.

    Methods:
        switch_provider: Switch to a different provider
        get_current_provider_info: Get current provider information
        check_provider_health: Check if current provider is healthy
        get_available_providers: Get list of available providers
        get_provider_capabilities: Get provider capabilities

    Example:
        class MyProviderService(ProviderServiceProtocol):
            def __init__(self, registry, health_checker):
                self._registry = registry
                self._health = health_checker
                self._current = None

            async def switch_provider(self, provider, model):
                # Validate and switch
                new_provider = self._registry.get(provider)
                await self._health.check(new_provider)
                self._current = new_provider
    """

    async def switch_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        validate: bool = True,
    ) -> None:
        """Switch to a different provider.

        Performs validation and health checks before switching
        to ensure the new provider is available and functional.

        Args:
            provider: Provider name (e.g., 'anthropic', 'openai')
            model: Optional model name. If None, uses default
            validate: If True, validate provider before switching

        Raises:
            ProviderNotFoundError: If provider is not registered
            ProviderValidationError: If provider validation fails
            ProviderHealthCheckError: If provider health check fails

        Example:
            await provider_service.switch_provider('openai', 'gpt-4')
            # Now using OpenAI with GPT-4
        """
        ...

    async def switch_model(self, model: str) -> None:
        """Switch models on the current provider.

        Args:
            model: Target model name
        """
        ...

    def get_current_provider_info(self) -> "ProviderInfo":
        """Get current provider information.

        Returns metadata about the currently configured provider
        including name, model, capabilities, and configuration.

        Returns:
            ProviderInfo with current provider details

        Example:
            info = provider_service.get_current_provider_info()
            print(f"Using {info.provider_name}/{info.model_name}")
            print(f"Tool calling: {info.supports_tool_calling}")
        """
        ...

    async def check_provider_health(
        self,
        provider: Optional[str] = None,
    ) -> bool:
        """Check if a provider is healthy.

        Performs a health check on the specified provider
        (or current provider if None) to verify it's functional.

        Args:
            provider: Provider name to check, or None for current

        Returns:
            True if provider is healthy, False otherwise

        Example:
            if not await provider_service.check_provider_health():
                logger.warning("Current provider is unhealthy")
                await provider_service.switch_provider('fallback')
        """
        ...

    def get_available_providers(self) -> List[str]:
        """Get list of available providers.

        Returns all registered provider names that can be
        used with switch_provider().

        Returns:
            List of provider names

        Example:
            providers = provider_service.get_available_providers()
            print(f"Available providers: {', '.join(providers)}")
        """
        ...

    async def get_provider_capabilities(
        self,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get provider capabilities.

        Returns detailed capability information for the provider
        including supported features, limits, and optimizations.

        Args:
            provider: Provider name, or None for current provider

        Returns:
            Dictionary with capability information

        Example:
            caps = await provider_service.get_provider_capabilities()
            print(f"Max tokens: {caps['max_tokens']}")
            print(f"Streaming: {caps['supports_streaming']}")
        """
        ...

    async def start_health_monitoring(self) -> None:
        """Start background provider health monitoring."""
        ...

    async def stop_health_monitoring(self) -> None:
        """Stop background provider health monitoring."""
        ...

    def get_current_provider(self) -> "BaseProvider":
        """Get the current provider instance.

        Returns the actual provider object for direct use
        if needed (prefer using service methods when possible).

        Returns:
            Current provider instance

        Raises:
            ProviderNotConfiguredError: If no provider is configured
        """
        ...

    async def test_provider(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> bool:
        """Test a provider with a simple request.

        Useful for validating configuration before switching.

        Args:
            provider: Provider name to test
            model: Optional model name

        Returns:
            True if test succeeded, False otherwise

        Example:
            if await provider_service.test_provider('openai'):
                await provider_service.switch_provider('openai')
        """
        ...

    def get_rate_limit_wait_time(self, error: Exception) -> float:
        """Extract retry wait time from a provider rate-limit error."""
        ...

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Return accumulated provider rate-limit statistics."""
        ...

    def is_healthy(self) -> bool:
        """Check if the provider service is healthy.

        A healthy provider service should:
        - Have a provider configured
        - Have valid provider credentials
        - Be able to reach the provider API

        Returns:
            True if the service is healthy, False otherwise
        """
        ...


@runtime_checkable
class ProviderResilienceProtocol(Protocol):
    """Extended protocol for provider resilience features.

    Provides circuit breaker, retry, and fallback functionality
    for robust provider management.
    """

    async def get_circuit_breaker_state(
        self,
        provider: str,
    ) -> Dict[str, Any]:
        """Get circuit breaker state for a provider.

        Returns the current state of the circuit breaker including
        open/closed state, failure counts, and last failure time.

        Args:
            provider: Provider name

        Returns:
            Circuit breaker state information
        """
        ...

    async def reset_circuit_breaker(self, provider: str) -> None:
        """Reset circuit breaker for a provider.

        Manually resets the circuit breaker to closed state,
        allowing requests to the provider again.

        Args:
            provider: Provider name
        """
        ...

    async def get_fallback_provider(
        self,
        failed_provider: str,
    ) -> Optional[str]:
        """Get fallback provider for a failed provider.

        Returns the best fallback provider based on configuration
        and health status.

        Args:
            failed_provider: Provider that failed

        Returns:
            Fallback provider name, or None if no fallback available
        """
        ...

    async def switch_with_fallback(
        self,
        provider: str,
        model: Optional[str] = None,
        max_attempts: int = 3,
    ) -> bool:
        """Switch to provider with automatic fallback.

        Attempts to switch to the specified provider, automatically
        falling back to alternatives if it fails.

        Args:
            provider: Primary provider to switch to
            model: Optional model name
            max_attempts: Maximum provider attempts before giving up

        Returns:
            True if successfully switched to any provider, False otherwise
        """
        ...


@runtime_checkable
class ProviderOptimizationProtocol(Protocol):
    """Protocol for provider optimization features.

    Provides methods for optimizing provider usage and performance.
    """

    async def optimize_request(
        self,
        messages: List[Any],
        tools: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Optimize request for the current provider.

        Applies provider-specific optimizations like token counting,
        batching, and request formatting.

        Args:
            messages: Messages to send
            tools: Optional tool definitions

        Returns:
            Optimized request parameters
        """
        ...

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: Optional[str] = None,
    ) -> float:
        """Estimate request cost in USD.

        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            provider: Provider name, or None for current

        Returns:
            Estimated cost in USD

        Example:
            cost = provider_service.estimate_cost(1000, 500)
            print(f"Estimated cost: ${cost:.4f}")
        """
        ...

    async def get_rate_limit_status(
        self,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get rate limit status for a provider.

        Returns current rate limit information including remaining
        requests, reset time, and limit values.

        Args:
            provider: Provider name, or None for current

        Returns:
            Rate limit status information
        """
        ...
