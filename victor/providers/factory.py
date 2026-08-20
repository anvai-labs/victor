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
Provider Factory with Shared Infrastructure Integration.

This module provides a unified factory for creating providers with all
shared infrastructure components automatically integrated:
- ResilientProvider (circuit breaker, retry, fallback)
- RequestQueue (rate limiting, priority queue)
- StreamingMetricsCollector (performance monitoring)
- ConversationMemoryManager (multi-turn context)

Usage:
    from victor.providers.factory import ManagedProviderFactory

    # Create enhanced provider with all improvements
    provider = await ManagedProviderFactory.create(
        provider_name="anthropic",
        model="claude-3-5-haiku-20241022",
        api_key=api_key,
        enable_resilience=True,
        enable_rate_limiting=True,
        enable_metrics=True,
    )

    # Use like any other provider
    response = await provider.chat(messages, model=model)

    # Access metrics
    metrics = provider.get_metrics()

    # Cleanup
    await provider.shutdown()
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from victor.providers.base import BaseProvider, CompletionResponse, Message, StreamChunk
from victor.providers.registry import ProviderRegistry
from victor.providers.resilience import (
    CircuitBreakerConfig,
    ProviderRetryConfig,
    ResilientProvider,
)
from victor.providers.concurrency import (
    RequestQueue,
    ConcurrencyConfig,
    RequestPriority,
    get_provider_config,
)
from victor.observability.analytics.streaming_metrics import (
    StreamingMetricsCollector,
    MetricsStreamWrapper,
)

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for enhanced provider creation.

    Attributes:
        provider_name: Name of the provider (anthropic, openai, ollama, etc.)
        model: Model identifier
        api_key: API key for cloud providers
        base_url: Base URL for API endpoints
        timeout: Request timeout in seconds

        # Resilience settings
        enable_resilience: Enable circuit breaker and retry
        circuit_breaker_config: Custom circuit breaker configuration
        retry_config: Custom retry configuration
        fallback_providers: List of fallback provider configs

        # Rate limiting settings
        enable_rate_limiting: Enable rate limiting and request queuing
        concurrency_config: Custom concurrency configuration
        num_workers: Number of worker tasks for request processing

        # Metrics settings
        enable_metrics: Enable streaming metrics collection
        metrics_history_size: Number of metrics samples to retain

        # Provider-specific settings
        extra_kwargs: Additional provider-specific arguments
    """

    provider_name: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 120.0

    # Resilience settings
    enable_resilience: bool = True
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    retry_config: Optional[ProviderRetryConfig] = None
    fallback_providers: List["ProviderConfig"] = field(default_factory=list)

    # Rate limiting settings
    enable_rate_limiting: bool = True
    concurrency_config: Optional[ConcurrencyConfig] = None
    num_workers: int = 3

    # Metrics settings
    enable_metrics: bool = True
    metrics_history_size: int = 1000

    # Provider-specific settings
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


class ManagedProvider:
    """
    A provider wrapper that integrates all shared infrastructure.

    Combines:
    - Base provider (Anthropic, OpenAI, Ollama, etc.)
    - ResilientProvider (circuit breaker, retry, fallback)
    - RequestQueue (rate limiting, priority queue)
    - StreamingMetricsCollector (performance monitoring)

    Provides a unified interface that works with any provider while
    adding enterprise-grade features transparently.

    Note: Previously named 'ManagedProvider'. Renamed for clarity.
    """

    def __init__(
        self,
        base_provider: BaseProvider,
        resilient_provider: Optional[ResilientProvider] = None,
        request_manager: Optional[RequestQueue] = None,
        metrics_collector: Optional[StreamingMetricsCollector] = None,
        config: Optional[ProviderConfig] = None,
    ):
        """Initialize managed provider.

        Args:
            base_provider: The underlying provider instance
            resilient_provider: Optional resilient wrapper
            request_manager: Optional request manager for rate limiting
            metrics_collector: Optional metrics collector
            config: Provider configuration
        """
        self._base_provider = base_provider
        self._resilient_provider = resilient_provider
        self._request_manager = request_manager
        self._metrics_collector = metrics_collector
        self._config = config

        # Use resilient provider if available, otherwise base
        self._active_provider = resilient_provider or base_provider

        logger.info(
            f"ManagedProvider created for {self.name}. "
            f"Resilience: {resilient_provider is not None}, "
            f"RateLimiting: {request_manager is not None}, "
            f"Metrics: {metrics_collector is not None}"
        )

    @property
    def name(self) -> str:
        """Get provider name."""
        return getattr(self._base_provider, "name", "unknown")

    @property
    def model(self) -> str:
        """Get model name."""
        return self._config.model if self._config else "unknown"

    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""
        return self._base_provider.supports_tools()

    async def chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat request with all enhancements.

        Args:
            messages: Conversation messages
            model: Model override
            tools: Tool definitions
            priority: Request priority for queuing
            **kwargs: Additional arguments

        Returns:
            CompletionResponse from the model
        """
        model = model or (self._config.model if self._config else None)

        async def _do_chat():
            return await self._active_provider.chat(
                messages=messages,
                model=model,
                tools=tools,
                **kwargs,
            )

        # Use request manager if available for rate limiting
        if self._request_manager:
            return await self._request_manager.submit(
                _do_chat(),
                priority=priority,
            )
        else:
            return await _do_chat()

    async def stream_chat(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat response with metrics collection.

        Args:
            messages: Conversation messages
            model: Model override
            tools: Tool definitions
            priority: Request priority for queuing
            **kwargs: Additional arguments

        Yields:
            StreamChunk objects with content and metadata
        """
        model = model or (self._config.model if self._config else None)

        # Get base stream
        base_stream = self._active_provider.stream_chat(
            messages=messages,
            model=model,
            tools=tools,
            **kwargs,
        )

        # Wrap with metrics collection if enabled. aclosing both paths finalizes the
        # underlying provider/httpx generator on-task instead of leaving it to GC.
        if self._metrics_collector:
            wrapped = MetricsStreamWrapper(
                stream=base_stream,
                collector=self._metrics_collector,
                model=model or "unknown",
                provider=self.name,
            )
            async with contextlib.aclosing(wrapped) as inner_stream:
                async for chunk in inner_stream:
                    yield chunk
        else:
            async with contextlib.aclosing(base_stream) as inner_stream:
                async for chunk in inner_stream:
                    yield chunk

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get collected metrics.

        Returns:
            Dictionary with metrics data or None if metrics disabled
        """
        if not self._metrics_collector:
            return None

        return {
            "summary": self._metrics_collector.get_summary(),
            "recent": self._metrics_collector.get_recent_metrics(),
            "history_size": len(self._metrics_collector._metrics_history),
        }

    def get_resilience_stats(self) -> Optional[Dict[str, Any]]:
        """Get resilience statistics.

        Returns:
            Dictionary with circuit breaker and retry stats
        """
        if not self._resilient_provider:
            return None

        return self._resilient_provider.get_stats()

    def get_rate_limit_stats(self) -> Optional[Dict[str, Any]]:
        """Get rate limiting statistics.

        Returns:
            Dictionary with rate limit and queue stats
        """
        if not self._request_manager:
            return None

        return self._request_manager.get_stats()

    async def shutdown(self) -> None:
        """Shutdown all components gracefully."""
        import asyncio
        import inspect

        logger.info(f"Shutting down ManagedProvider for {self.name}...")

        if self._request_manager:
            await self._request_manager.shutdown()

        # Close base provider if it has a close method
        if hasattr(self._base_provider, "close"):
            close_method = self._base_provider.close
            if asyncio.iscoroutinefunction(close_method) or inspect.iscoroutinefunction(
                close_method
            ):
                await close_method()
            elif callable(close_method):
                result = close_method()
                # Handle case where close returns a coroutine
                if asyncio.iscoroutine(result):
                    await result

        logger.info(f"ManagedProvider for {self.name} shutdown complete")

    def is_healthy(self) -> bool:
        """Check if the provider is still operational.

        Verifies that the underlying provider and resilience layer
        are in a usable state. Used by ProviderPool to avoid
        returning stale providers.
        """
        # Check circuit breaker state if resilient provider is active
        if self._resilient_provider is not None:
            cb = getattr(self._resilient_provider, "_circuit_breaker", None)
            if cb is not None and hasattr(cb, "state"):
                if cb.state == "open":
                    return False
        # Check if base provider client is still alive
        client = getattr(self._base_provider, "_client", None)
        if client is not None and hasattr(client, "is_closed"):
            if client.is_closed:
                return False
        return True


class ProviderPool:
    """Connection pool for ManagedProvider instances.

    Reuses provider instances keyed by (provider_name, model) to avoid
    repeated initialization overhead. Thread-safe via asyncio.Lock.

    Usage:
        pool = ProviderPool(max_size_per_key=3)
        provider = await pool.acquire("anthropic", "claude-3-5-haiku", factory_fn)
        # ... use provider ...
        await pool.release("anthropic", "claude-3-5-haiku", provider)
    """

    def __init__(self, max_size_per_key: int = 3) -> None:
        self._max_size = max_size_per_key
        self._pools: Dict[tuple, List[ManagedProvider]] = {}
        self._in_use: Dict[tuple, int] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        provider_name: str,
        model: str,
        factory: Callable[[], Awaitable["ManagedProvider"]],
    ) -> "ManagedProvider":
        """Get a provider from pool or create a new one."""
        key = (provider_name, model)
        async with self._lock:
            pool = self._pools.setdefault(key, [])
            if pool:
                provider = pool.pop()
                self._in_use[key] = self._in_use.get(key, 0) + 1
                logger.debug("Reusing pooled provider for %s/%s", provider_name, model)
                return provider

        # Create outside lock to avoid blocking other acquires
        provider = await factory()
        async with self._lock:
            self._in_use[key] = self._in_use.get(key, 0) + 1
        logger.debug("Created new provider for %s/%s", provider_name, model)
        return provider

    async def release(self, provider_name: str, model: str, provider: "ManagedProvider") -> None:
        """Return a provider to the pool.

        Checks provider health before re-pooling. Unhealthy providers
        are shut down instead of being returned to the pool.
        """
        key = (provider_name, model)
        async with self._lock:
            self._in_use[key] = max(0, self._in_use.get(key, 0) - 1)
            pool = self._pools.setdefault(key, [])
            if len(pool) < self._max_size and provider.is_healthy():
                pool.append(provider)
            else:
                if not provider.is_healthy():
                    logger.debug(
                        "Discarding unhealthy provider for %s/%s",
                        provider_name,
                        model,
                    )
                await provider.shutdown()

    async def shutdown(self) -> None:
        """Shutdown all pooled providers."""
        async with self._lock:
            for key, pool in self._pools.items():
                for provider in pool:
                    try:
                        await provider.shutdown()
                    except Exception as e:
                        logger.warning("Error shutting down pooled provider %s: %s", key, e)
            self._pools.clear()
            self._in_use.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pools": {
                f"{k[0]}/{k[1]}": {
                    "available": len(v),
                    "in_use": self._in_use.get(k, 0),
                }
                for k, v in self._pools.items()
            },
            "total_available": sum(len(v) for v in self._pools.values()),
            "total_in_use": sum(self._in_use.values()),
        }


class ManagedProviderFactory:
    """
    Factory for creating providers with shared infrastructure.

    Automatically integrates:
    - Circuit breaker and retry logic
    - Rate limiting with priority queues
    - Streaming metrics collection
    - Fallback provider chains
    - Connection pooling via ProviderPool

    Usage:
        # Simple creation (no pooling)
        provider = await ManagedProviderFactory.create(
            provider_name="anthropic",
            model="claude-3-5-haiku-20241022",
            api_key=api_key,
        )

        # With pooling (reuses providers)
        provider = await ManagedProviderFactory.acquire(
            provider_name="anthropic",
            model="claude-3-5-haiku-20241022",
            api_key=api_key,
        )
        # ... use provider ...
        await ManagedProviderFactory.release(provider)
    """

    _default_pool: Optional[ProviderPool] = None

    _default_pool: Optional["ProviderPool"] = None

    @classmethod
    def get_pool(cls, max_size_per_key: int = 3) -> "ProviderPool":
        """Get or create the default provider pool."""
        if cls._default_pool is None:
            cls._default_pool = ProviderPool(max_size_per_key=max_size_per_key)
        return cls._default_pool

    @classmethod
    async def acquire(
        cls,
        provider_name: str,
        model: str,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> ManagedProvider:
        """Acquire a provider from the pool (or create new)."""
        pool = cls.get_pool()

        async def _factory():
            return await cls.create(
                provider_name=provider_name,
                model=model,
                api_key=api_key,
                **kwargs,
            )

        return await pool.acquire(provider_name, model, _factory)

    @classmethod
    async def release(cls, provider: ManagedProvider) -> None:
        """Release a provider back to the pool."""
        if cls._default_pool is not None:
            name = provider.name
            model = provider.model
            await cls._default_pool.release(name, model, provider)

    @classmethod
    async def shutdown_pool(cls) -> None:
        """Shutdown the default provider pool."""
        if cls._default_pool is not None:
            await cls._default_pool.shutdown()
            cls._default_pool = None

    @classmethod
    async def create(
        cls,
        provider_name: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        enable_resilience: bool = True,
        enable_rate_limiting: bool = True,
        enable_metrics: bool = True,
        fallback_configs: Optional[List[ProviderConfig]] = None,
        **kwargs: Any,
    ) -> ManagedProvider:
        """Create an enhanced provider with shared infrastructure.

        Args:
            provider_name: Provider name (anthropic, openai, ollama, etc.)
            model: Model identifier
            api_key: API key for cloud providers
            base_url: Base URL override
            timeout: Request timeout
            enable_resilience: Enable circuit breaker and retry
            enable_rate_limiting: Enable rate limiting
            enable_metrics: Enable streaming metrics
            fallback_configs: Fallback provider configurations
            **kwargs: Additional provider-specific arguments

        Returns:
            ManagedProvider with all integrations
        """
        config = ProviderConfig(
            provider_name=provider_name,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            enable_resilience=enable_resilience,
            enable_rate_limiting=enable_rate_limiting,
            enable_metrics=enable_metrics,
            fallback_providers=fallback_configs or [],
            extra_kwargs=kwargs,
        )

        return await cls.create_from_config(config)

    @classmethod
    async def create_from_config(cls, config: ProviderConfig) -> ManagedProvider:
        """Create an enhanced provider from configuration.

        Args:
            config: ProviderConfig with all settings

        Returns:
            ManagedProvider with all integrations
        """
        # Build provider kwargs
        provider_kwargs: Dict[str, Any] = {}
        if config.api_key:
            provider_kwargs["api_key"] = config.api_key
        if config.base_url:
            provider_kwargs["base_url"] = config.base_url
        if config.timeout:
            provider_kwargs["timeout"] = config.timeout
        provider_kwargs.update(config.extra_kwargs)

        # Create base provider
        base_provider = ProviderRegistry.create(
            config.provider_name,
            **provider_kwargs,
        )

        # Create resilient provider if enabled
        resilient_provider: Optional[ResilientProvider] = None
        if config.enable_resilience:
            # Create fallback providers
            fallback_providers = []
            for fb_config in config.fallback_providers:
                fb_kwargs: Dict[str, Any] = {}
                if fb_config.api_key:
                    fb_kwargs["api_key"] = fb_config.api_key
                if fb_config.base_url:
                    fb_kwargs["base_url"] = fb_config.base_url
                fb_kwargs.update(fb_config.extra_kwargs)

                fb_provider = ProviderRegistry.create(
                    fb_config.provider_name,
                    **fb_kwargs,
                )
                fallback_providers.append(fb_provider)

            resilient_provider = ResilientProvider(
                provider=base_provider,
                circuit_config=config.circuit_breaker_config,
                retry_config=config.retry_config,
                fallback_providers=fallback_providers,
                request_timeout=config.timeout,
            )

        # Create request manager if rate limiting enabled
        request_manager: Optional[RequestQueue] = None
        if config.enable_rate_limiting:
            # Get provider-specific config or use custom
            concurrency_config = config.concurrency_config or get_provider_config(
                config.provider_name,
                config.model,
            )
            request_manager = RequestQueue(
                config=concurrency_config,
                num_workers=config.num_workers,
            )

        # Create metrics collector if enabled
        metrics_collector: Optional[StreamingMetricsCollector] = None
        if config.enable_metrics:
            metrics_collector = StreamingMetricsCollector(
                max_history=config.metrics_history_size,
            )

        return ManagedProvider(
            base_provider=base_provider,
            resilient_provider=resilient_provider,
            request_manager=request_manager,
            metrics_collector=metrics_collector,
            config=config,
        )

    @classmethod
    async def create_with_fallbacks(
        cls,
        primary_config: ProviderConfig,
        fallback_configs: List[ProviderConfig],
    ) -> ManagedProvider:
        """Create provider with explicit fallback chain.

        Args:
            primary_config: Primary provider configuration
            fallback_configs: Ordered list of fallback configurations

        Returns:
            ManagedProvider with fallback chain
        """
        primary_config.fallback_providers = fallback_configs
        return await cls.create_from_config(primary_config)

    @classmethod
    async def create_with_smart_routing(
        cls,
        provider_name: Optional[str],  # Suggestion only
        model: str,
        routing_profile: str = "balanced",
        custom_fallback_chain: Optional[List[str]] = None,
        performance_window_size: int = 100,
        learning_enabled: bool = True,
        resource_awareness_enabled: bool = True,
        **kwargs: Any,
    ) -> Any:  # Returns SmartRoutingProvider
        """Create provider with smart routing enabled.

        Creates multiple providers from the routing profile and wraps them
        in a SmartRoutingProvider that intelligently selects the best provider
        for each request based on health, resources, cost, latency, and performance.

        Args:
            provider_name: Suggested primary provider (can be overridden by routing)
            model: Model identifier
            routing_profile: Profile name from routing_profiles.yaml
                (balanced, cost-optimized, performance, local-first)
            custom_fallback_chain: Custom fallback chain (overrides profile)
            performance_window_size: Number of requests for learning
            learning_enabled: Enable adaptive learning from performance
            resource_awareness_enabled: Enable GPU/API quota detection
            **kwargs: Additional provider arguments

        Returns:
            SmartRoutingProvider instance

        Example:
            # Create smart routing provider
            provider = await ManagedProviderFactory.create_with_smart_routing(
                provider_name="ollama",  # Suggestion
                model="qwen3-coder:30b",
                routing_profile="balanced",
            )

            # Use like any other provider
            response = await provider.chat(messages, model=model)
        """
        from victor.providers.smart_router import SmartRoutingProvider
        from victor.providers.routing_config import (
            SmartRoutingConfig,
            load_routing_profiles,
        )

        # Load routing profile
        profiles = load_routing_profiles()
        profile = profiles.get(routing_profile)
        if not profile:
            logger.warning(f"Profile '{routing_profile}' not found, using 'balanced'")
            profile = profiles.get("balanced")

        # Get fallback chain (custom or from profile)
        if custom_fallback_chain:
            provider_names = custom_fallback_chain
        else:
            provider_names = profile.get_fallback_chain("default")

        # If provider_name suggested, prepend it if not already in chain
        if provider_name and provider_name.lower() not in [p.lower() for p in provider_names]:
            provider_names.insert(0, provider_name)

        # Create all providers in the chain
        providers = []
        for provider_name_str in provider_names:
            try:
                provider = await cls.create(
                    provider_name=provider_name_str,
                    model=model,
                    **kwargs,
                )
                providers.append(provider._active_provider)  # Use ResilientProvider
            except Exception as e:
                logger.warning(f"Failed to create provider '{provider_name_str}': {e}")
                continue

        if not providers:
            raise Exception(f"No providers could be created from chain: {provider_names}")

        # Create smart routing config
        config = SmartRoutingConfig(
            enabled=True,
            profile_name=routing_profile,
            custom_fallback_chain=custom_fallback_chain,
            performance_window_size=performance_window_size,
            learning_enabled=learning_enabled,
            resource_awareness_enabled=resource_awareness_enabled,
        )

        # Wrap in SmartRoutingProvider
        smart_provider = SmartRoutingProvider(
            providers=[p._base_provider for p in providers],  # Pass base providers
            config=config,
        )

        logger.info(
            f"Created SmartRoutingProvider with {len(providers)} providers, "
            f"profile='{routing_profile}'"
        )

        return smart_provider


# Convenience function for quick provider creation
async def create_enhanced_provider(
    provider_name: str,
    model: str,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ManagedProvider:
    """Quick helper to create an enhanced provider.

    Args:
        provider_name: Provider name
        model: Model identifier
        api_key: API key
        **kwargs: Additional arguments

    Returns:
        ManagedProvider with default settings
    """
    return await ManagedProviderFactory.create(
        provider_name=provider_name,
        model=model,
        api_key=api_key,
        **kwargs,
    )
