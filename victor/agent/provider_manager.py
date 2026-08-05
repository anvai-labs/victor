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

"""Unified provider and model management.

This module provides centralized management for:
- Provider initialization and switching
- Model switching and hot-swap
- Provider health monitoring
- Fallback chain management
- Tool calling adapter coordination

Design Pattern: Facade + Composition
===================================
ProviderManager acts as a facade coordinating specialized components:
- ProviderSwitcher: Provider and model switching logic
- ProviderHealthMonitor: Health monitoring
- ToolAdapterCoordinator: Tool capability detection
- DefaultProviderClassificationStrategy: Provider classification

SRP Compliance:
ProviderManager now delegates to focused components instead of
implementing all functionality directly. This reduces the class from
~680 lines to ~200 lines while maintaining the same public API.

Usage:
    from victor.agent.provider_manager import ProviderManager

    manager = ProviderManager(
        settings=settings,
        initial_provider=provider,
        initial_model=model,
    )

    # Switch providers with health check
    await manager.switch_provider("anthropic", "claude-sonnet-4-20250514")

    # Get healthy providers
    healthy = await manager.get_healthy_providers()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from victor.agent.model_switcher import ModelSwitcher, SwitchReason, ModelSwitchEvent
from victor.agent.tool_calling import (
    ToolCallingAdapterRegistry,
    ToolCallingCapabilities,
)
from victor.agent.provider import (
    ProviderSwitcher,
    ProviderHealthMonitor,
    ToolAdapterCoordinator,
    ProviderSwitcherState,
)
from victor.agent.strategies import DefaultProviderClassificationStrategy
from victor.core.errors import ProviderNotFoundError
from victor.providers.base import BaseProvider
from victor.providers.registry import ProviderRegistry
from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities
from victor.providers.capability_contract import ProviderCapabilityContract

logger = logging.getLogger(__name__)


@dataclass
class ProviderManagerConfig:
    """Configuration for ProviderManager.

    Attributes:
        enable_health_checks: Enable provider health monitoring
        health_check_interval: Interval between health checks (seconds)
        auto_fallback: Automatically fallback to healthy provider on failure
        fallback_providers: Ordered list of fallback providers
        max_fallback_attempts: Maximum fallback attempts before giving up
    """

    enable_health_checks: bool = True
    health_check_interval: float = 60.0
    auto_fallback: bool = True
    fallback_providers: List[str] = field(default_factory=list)
    max_fallback_attempts: int = 3


@dataclass
class ProviderState:
    """Current state of a provider.

    Attributes:
        provider: Provider instance
        provider_name: Provider name
        model: Current model
        tool_adapter: Tool calling adapter
        capabilities: Tool calling capabilities
        is_healthy: Whether provider is healthy
        last_error: Last error message (if any)
        switch_count: Number of times provider/model has been switched
    """

    provider: BaseProvider
    provider_name: str
    model: str
    tool_adapter: Any = None
    capabilities: Optional[ToolCallingCapabilities] = None
    is_healthy: bool = True
    last_error: Optional[str] = None
    runtime_capabilities: Optional[ProviderRuntimeCapabilities] = None
    switch_count: int = 0
    contract: Optional[ProviderCapabilityContract] = None


class ProviderManager:
    """Unified management of providers and models.

    Coordinates provider switching, health monitoring, and fallback
    handling for robust LLM interactions.

    SRP Compliance:
    This class now delegates to specialized components:
    - ProviderSwitcher handles switching logic
    - ProviderHealthMonitor handles health checks
    - ToolAdapterCoordinator handles tool adapters
    - ClassificationStrategy handles provider classification

    Features:
    - Hot-swap providers without losing context
    - Automatic health monitoring
    - Fallback chain support
    - Switch history tracking
    - Tool capability detection
    """

    def __init__(
        self,
        settings: Any,
        initial_provider: Optional[BaseProvider] = None,
        initial_model: Optional[str] = None,
        provider_name: Optional[str] = None,
        config: Optional[ProviderManagerConfig] = None,
    ):
        """Initialize the provider manager using composition.

        Args:
            settings: Application settings
            initial_provider: Initial provider instance
            initial_model: Initial model name
            provider_name: Provider name (e.g., 'anthropic', 'ollama')
            config: Optional configuration
        """
        self.settings = settings
        self.config = config or ProviderManagerConfig()

        # DIP Compliance: Depend on abstractions via composition
        self._classification_strategy = DefaultProviderClassificationStrategy()

        # Initialize health monitor
        self._health_monitor = ProviderHealthMonitor(
            settings=settings,
            enable_health_checks=config.enable_health_checks if config else True,
            health_check_interval=config.health_check_interval if config else 60.0,
        )

        # Initialize tool adapter coordinator
        self._adapter_coordinator = ToolAdapterCoordinator(
            provider_switcher=None,  # Will set after creating switcher
            settings=settings,
        )

        # Initialize provider switcher (depends on coordinator)
        self._provider_switcher = ProviderSwitcher(
            classification_strategy=self._classification_strategy,
            event_emitter=self,  # ProviderManager implements event emission
            health_monitor=self._health_monitor,
            adapter_coordinator=self._adapter_coordinator,
        )

        # Update coordinator with switcher reference
        self._adapter_coordinator._provider_switcher = self._provider_switcher

        # Legacy state for backward compatibility
        self._current_state: Optional[ProviderState] = None

        # Initialize with provided values
        if initial_provider and initial_model:
            name = provider_name or getattr(initial_provider, "name", "unknown")
            provider_name_lower = name.lower()

            # Set state in switcher
            self._provider_switcher.set_initial_state(
                provider=initial_provider,
                provider_name=provider_name_lower,
                model=initial_model,
            )

            # Create legacy state object
            self._current_state = ProviderState(
                provider=initial_provider,
                provider_name=provider_name_lower,
                model=initial_model,
            )

        # Model switcher for tracking (kept for backward compatibility)
        self._model_switcher = ModelSwitcher()
        if self._current_state:
            self._model_switcher.set_current(
                self._current_state.provider_name,
                self._current_state.model,
            )

        # Callbacks for provider changes
        self._on_switch_callbacks: List[Callable[[ProviderState], None]] = []

        # Runtime capability cache (per provider/model)
        self._capability_cache: Dict[tuple[str, str], ProviderRuntimeCapabilities] = {}

        logger.debug(
            f"ProviderManager initialized: {self._current_state.provider_name if self._current_state else 'none'}"
        )

    @property
    def provider(self) -> Optional[BaseProvider]:
        """Get current provider instance."""
        return self._provider_switcher.get_current_provider()

    @property
    def provider_name(self) -> str:
        """Get current provider name."""
        state = self._provider_switcher.get_current_state()
        return state.provider_name if state else ""

    @property
    def model(self) -> str:
        """Get current model name."""
        return self._provider_switcher.get_current_model()

    @property
    def capabilities(self) -> Optional[ToolCallingCapabilities]:
        """Get current tool calling capabilities."""
        return self._current_state.capabilities if self._current_state else None

    @property
    def tool_adapter(self) -> Optional[Any]:
        """Get current tool calling adapter."""
        return self._current_state.tool_adapter if self._current_state else None

    @property
    def switch_count(self) -> int:
        """Get the number of provider/model switches."""
        return self._current_state.switch_count if self._current_state else 0

    def get_current_state(self) -> Optional[ProviderState]:
        """Get the current provider state.

        Returns:
            Current ProviderState or None if not configured
        """
        return self._current_state

    def is_cloud_provider(self, provider_name: Optional[str] = None) -> bool:
        """Check if provider is cloud-based.

        Args:
            provider_name: Optional provider name (uses current if not provided)

        Returns:
            True if cloud provider, False otherwise
        """
        name = provider_name or self.provider_name
        return self._classification_strategy.is_cloud_provider(name)

    def is_local_provider(self, provider_name: Optional[str] = None) -> bool:
        """Check if provider is local.

        Args:
            provider_name: Optional provider name (uses current if not provided)

        Returns:
            True if local provider, False otherwise
        """
        name = provider_name or self.provider_name
        return self._classification_strategy.is_local_provider(name)

    def get_context_window(self) -> int:
        """Get context window size for current provider/model.

        Tries to get the context window from the provider first, then falls
        back to the centralized YAML configuration.

        Returns:
            Context window in tokens
        """
        # Try to get from the provider's own implementation first
        cache_key = (self.provider_name, self.model)
        if cache_key in self._capability_cache:
            return self._capability_cache[cache_key].context_window

        if self.provider and hasattr(self.provider, "get_context_window"):
            try:
                # This allows providers like Ollama to dynamically determine the window
                return self.provider.get_context_window(self.model)
            except Exception:
                logger.warning(
                    f"Provider {self.provider_name} failed to dynamically get context window."
                )

        # Fall back to the centralized YAML configuration
        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits(self.provider_name, self.model)
        logger.debug(
            f"Using context window from config for {self.provider_name}: {limits.context_window}"
        )
        return limits.context_window

    def initialize_tool_adapter(self) -> ToolCallingCapabilities:
        """Initialize tool calling adapter for current provider/model.

        Delegates to ToolAdapterCoordinator.

        Returns:
            Tool calling capabilities
        """
        if not self._current_state:
            raise ValueError("No provider configured")

        # Delegate to coordinator (SRP)
        capabilities = self._adapter_coordinator.initialize_adapter()

        # Update legacy state for backward compatibility
        self._current_state.tool_adapter = self._adapter_coordinator.get_adapter()
        self._current_state.capabilities = capabilities

        logger.info(
            f"Tool adapter initialized: native={capabilities.native_tool_calls}, "
            f"format={capabilities.tool_call_format.value}"
        )

        return capabilities

    async def _discover_and_cache_capabilities(
        self,
    ) -> Optional[ProviderRuntimeCapabilities]:
        """Discover provider capabilities asynchronously and cache result."""
        if not self._current_state:
            return None

        cache_key = (self.provider_name, self.model)
        if cache_key in self._capability_cache:
            return self._capability_cache[cache_key]

        try:
            discovery = await self.provider.discover_capabilities(self.model)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(
                f"Capability discovery failed for {self.provider_name}:{self.model} ({exc}); "
                "falling back to config."
            )
            from victor.config.config_loaders import get_provider_limits

            limits = get_provider_limits(self.provider_name, self.model)
            discovery = ProviderRuntimeCapabilities(
                provider=self.provider_name,
                model=self.model,
                context_window=limits.context_window,
                supports_tools=self.provider.supports_tools() if self.provider else False,  # type: ignore[union-attr]
                supports_streaming=self.provider.supports_streaming() if self.provider else False,  # type: ignore[union-attr]
                source="config",
            )

        self._capability_cache[cache_key] = discovery
        self._current_state.runtime_capabilities = discovery
        return discovery

    async def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None,
        reason: SwitchReason = SwitchReason.USER_REQUEST,
        **provider_kwargs: Any,
    ) -> bool:
        """Switch to a different provider.

        Delegates to ProviderSwitcher for core switching logic while
        maintaining backward compatibility with legacy features like
        fallback chains and capability discovery.

        Args:
            provider_name: Name of the provider
            model: Optional model name (uses current if not provided)
            reason: Reason for the switch
            **provider_kwargs: Additional provider arguments

        Returns:
            True if switch was successful
        """
        try:
            # Determine model to use
            new_model = model or self.model

            # Convert SwitchReason enum to string for ProviderSwitcher
            reason_str = "manual" if reason == SwitchReason.USER_REQUEST else "auto"

            # Delegate to ProviderSwitcher (SRP)
            result = await self._provider_switcher.switch_provider(
                provider_name=provider_name,
                model=new_model,
                reason=reason_str,
                settings=self.settings,
                **provider_kwargs,
            )

            if not result:
                # Attempt fallback if configured
                if self.config.auto_fallback:
                    logger.warning(
                        f"Provider switch to {provider_name} failed, attempting fallback"
                    )
                    return await self._attempt_fallback(reason)
                return False

            # Sync legacy state with switcher state
            switcher_state = self._provider_switcher.get_current_state()
            if switcher_state:
                _old_switch_count = self._current_state.switch_count if self._current_state else 0
                self._current_state = ProviderState(
                    provider=switcher_state.provider,
                    provider_name=switcher_state.provider_name,
                    model=switcher_state.model,
                    switch_count=switcher_state.switch_count,
                    last_error=switcher_state.last_error,
                )

                # Re-initialize tool adapter and discover capabilities
                self.initialize_tool_adapter()
                await self._discover_and_cache_capabilities()

                # Update model switcher for backward compatibility
                self._model_switcher.switch(
                    provider=provider_name,
                    model=new_model,
                    reason=reason,
                    metadata={
                        "from_provider": self.provider_name,
                        "from_model": self.model,
                    },
                )

            return True

        except ProviderNotFoundError:
            # Let ProviderNotFoundError propagate to caller
            raise
        except Exception as e:
            logger.error(f"Failed to switch provider to {provider_name}: {e}")
            if self._current_state:
                self._current_state.last_error = str(e)
            return False

    async def switch_model(
        self,
        model: str,
        reason: SwitchReason = SwitchReason.USER_REQUEST,
    ) -> bool:
        """Switch to a different model on the current provider.

        Delegates to ProviderSwitcher for core switching logic.

        Args:
            model: New model name
            reason: Reason for the switch

        Returns:
            True if switch was successful
        """
        if not self._current_state:
            logger.error("No provider configured, cannot switch model")
            return False

        try:
            old_model = self.model

            # Convert SwitchReason enum to string for ProviderSwitcher
            reason_str = "manual" if reason == SwitchReason.USER_REQUEST else "auto"

            # Delegate to ProviderSwitcher (SRP)
            result = await self._provider_switcher.switch_model(
                model=model,
                reason=reason_str,
            )

            if not result:
                return False

            # Sync legacy state with switcher state
            switcher_state = self._provider_switcher.get_current_state()
            if switcher_state:
                self._current_state.model = switcher_state.model
                self._current_state.switch_count = switcher_state.switch_count
                self._current_state.last_error = switcher_state.last_error

                # Re-initialize tool adapter
                self.initialize_tool_adapter()

                # Discover runtime capabilities (async, cached)
                await self._discover_and_cache_capabilities()

                # Update model switcher for backward compatibility
                self._model_switcher.switch(
                    provider=self.provider_name,
                    model=model,
                    reason=reason,
                    metadata={"from_model": old_model},
                )

            logger.info(
                f"Switched model: {old_model} -> {model} "
                f"(native_tools={self.capabilities.native_tool_calls if self.capabilities else 'unknown'})"
            )

            # Emit RL event for model selection
            self._emit_model_selected_event(
                old_model=old_model,
                new_model=model,
                reason=reason,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to switch model to {model}: {e}")
            if self._current_state:
                self._current_state.last_error = str(e)
            return False

    def _emit_model_selected_event(
        self,
        old_model: str,
        new_model: str,
        reason: "SwitchReason",
    ) -> None:
        """Emit RL event for model selection.

        Args:
            old_model: Previous model name
            new_model: New model name
            reason: Reason for the switch
        """
        try:
            from victor.framework.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            event = RLEvent(
                type=RLEventType.MODEL_SELECTED,
                provider=self.provider_name,
                model=new_model,
                success=True,
                quality_score=0.5,  # Will be updated by outcome
                metadata={
                    "old_model": old_model,
                    "reason": reason.value if hasattr(reason, "value") else str(reason),
                },
            )

            hooks.emit(event)

        except Exception as e:
            logger.debug(f"Model selected event emission failed: {e}")

    def get_info(self) -> Dict[str, Any]:
        """Get information about current provider and model.

        Returns:
            Dictionary with provider/model info and capabilities
        """
        if not self._current_state:
            return {"provider": None, "model": None}

        caps = self.capabilities
        runtime_caps = self._capability_cache.get((self.provider_name, self.model))
        return {
            "provider": self.provider_name,
            "model": self.model,
            "supports_tools": (self.provider.supports_tools() if self.provider else False),
            "native_tool_calls": caps.native_tool_calls if caps else False,
            "streaming_tool_calls": caps.streaming_tool_calls if caps else False,
            "parallel_tool_calls": caps.parallel_tool_calls if caps else False,
            "thinking_mode": caps.thinking_mode if caps else False,
            "context_window": (
                runtime_caps.context_window if runtime_caps else self.get_context_window()
            ),
            "is_cloud": self.is_cloud_provider(),
            "is_local": self.is_local_provider(),
            "is_healthy": self._current_state.is_healthy,
        }

    def get_switch_history(self) -> List[ModelSwitchEvent]:
        """Get history of provider/model switches.

        Returns:
            List of switch events
        """
        return self._model_switcher.get_switch_history()

    def add_switch_callback(self, callback: Callable[[ProviderState], None]) -> None:
        """Add callback for provider/model switches.

        Args:
            callback: Function called with new ProviderState after switch
        """
        # Register with ProviderSwitcher for automatic notification
        self._provider_switcher.on_switch(callback)

    # IProviderEventEmitter implementation
    def emit_switch_event(self, event: Dict[str, Any]) -> None:
        """Emit provider switch event.

        Implementation of IProviderEventEmitter protocol.

        Args:
            event: Event dictionary with switch details
        """
        # This is called by ProviderSwitcher
        # For now, we just log - can be extended to emit to RL tracking, etc.
        logger.debug(f"Switch event emitted: {event}")

    async def _check_provider_health(self, provider: BaseProvider, provider_name: str) -> bool:
        """Check health of a provider.

        Delegates to ProviderHealthMonitor.

        Args:
            provider: Provider instance
            provider_name: Provider name

        Returns:
            True if provider is healthy
        """
        # Delegate to health monitor (SRP)
        return await self._health_monitor.check_health(provider)

    async def _attempt_fallback(self, original_reason: SwitchReason) -> bool:
        """Attempt to fallback to a healthy provider.

        Args:
            original_reason: Original reason for switch attempt

        Returns:
            True if fallback was successful
        """
        fallback_providers = self.config.fallback_providers

        if not fallback_providers:
            logger.warning("No fallback providers configured")
            return False

        for attempt, provider_name in enumerate(fallback_providers):
            if attempt >= self.config.max_fallback_attempts:
                break

            logger.info(f"Attempting fallback to {provider_name} (attempt {attempt + 1})")

            try:
                # Disable auto_fallback to prevent infinite loop
                old_auto_fallback = self.config.auto_fallback
                self.config.auto_fallback = False

                result = await self.switch_provider(
                    provider_name,
                    reason=SwitchReason.FALLBACK,
                )

                self.config.auto_fallback = old_auto_fallback

                if result:
                    logger.info(f"Fallback to {provider_name} successful")
                    return True

            except Exception as e:
                logger.warning(f"Fallback to {provider_name} failed: {e}")

        logger.error("All fallback attempts exhausted")
        return False

    def _notify_switch(self, new_state: ProviderState) -> None:
        """Notify callbacks of provider switch."""
        for callback in self._on_switch_callbacks:
            try:
                callback(new_state)
            except Exception as e:
                logger.warning(f"Switch callback error: {e}")

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring.

        Delegates to ProviderHealthMonitor.
        """
        # Delegate to health monitor (SRP)
        await self._health_monitor.start_health_checks(
            interval=self.config.health_check_interval,
            provider=self.provider,
            provider_name=self.provider_name,
        )

        # Legacy task tracking
        self._health_check_task = self._health_monitor._health_check_task
        logger.info("Started provider health monitoring")

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring.

        Delegates to ProviderHealthMonitor.
        """
        # Delegate to health monitor (SRP)
        await self._health_monitor.stop_health_checks()

        # Clear legacy task tracking
        self._health_check_task = None
        logger.debug("Stopped provider health monitoring")

    async def get_healthy_providers(self) -> List[str]:
        """Get list of healthy providers.

        Delegates to ProviderHealthMonitor.

        Returns:
            List of healthy provider names sorted by latency
        """
        # Delegate to health monitor (SRP)
        return await self._health_monitor.get_healthy_providers()

    async def close(self) -> None:
        """Close provider and cleanup."""
        await self.stop_health_monitoring()

        if self.provider:
            try:
                await self.provider.close()
            except Exception as e:
                logger.debug(f"Error closing provider: {e}")

        logger.debug("ProviderManager closed")
