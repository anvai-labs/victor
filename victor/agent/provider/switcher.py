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

"""Provider switching operations.

This module provides ProviderSwitcher, which handles provider and model
switching logic. Extracted from ProviderManager to follow the Single
Responsibility Principle (SRP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

import logging
import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from victor.agent.model_switcher import ModelSwitcher, SwitchReason
from victor.agent.protocols import (
    IProviderSwitcher,
    IProviderEventEmitter,
    IProviderClassificationStrategy,
    IToolAdapterCoordinator,
    IProviderHealthMonitor,
)
from victor.core.errors import ProviderNotFoundError
from victor.providers.base import BaseProvider
from victor.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


@dataclass
class ProviderSwitcherState:
    """Current state of provider switching.

    Attributes:
        provider: Provider instance
        provider_name: Provider name
        model: Current model
        switch_count: Number of switches performed
        last_error: Last error message (if any)
    """

    provider: BaseProvider
    provider_name: str
    model: str
    switch_count: int = 0
    last_error: Optional[str] = None


class ProviderSwitcher(IProviderSwitcher):
    """Handles provider and model switching operations.

    This class is responsible for:
    - Switching between providers
    - Switching between models on the same provider
    - Maintaining switch history
    - Coordinating with health monitor
    - Coordinating with tool adapter coordinator
    - Emitting switch events

    SRP Compliance: Focuses only on switching logic, delegating
    health monitoring, tool coordination, and event emission to
    specialized components.

    Attributes:
        _classification_strategy: Strategy for provider classification
        _event_emitter: Component for emitting switch events
        _health_monitor: Component for health checking
        _adapter_coordinator: Component for tool adapter coordination
        _current_state: Current provider/model state
        _model_switcher: ModelSwitcher for tracking switches
        _switch_history: History of all switches
        _on_switch_callbacks: Callbacks invoked on switch
    """

    def __init__(
        self,
        classification_strategy: IProviderClassificationStrategy,
        event_emitter: IProviderEventEmitter,
        health_monitor: Optional[IProviderHealthMonitor] = None,
        adapter_coordinator: Optional[IToolAdapterCoordinator] = None,
    ):
        """Initialize the provider switcher.

        Args:
            classification_strategy: Strategy for classifying providers
            event_emitter: Component for emitting switch events
            health_monitor: Optional health monitor for pre-switch checks
            adapter_coordinator: Optional tool adapter coordinator
        """
        self._classification_strategy = classification_strategy
        self._event_emitter = event_emitter
        self._health_monitor = health_monitor
        self._adapter_coordinator = adapter_coordinator

        self._current_state: Optional[ProviderSwitcherState] = None
        self._model_switcher = ModelSwitcher()
        self._switch_history: List[Dict[str, Any]] = []
        self._on_switch_callbacks: List[Callable[[ProviderSwitcherState], None]] = []

    def get_current_provider(self) -> Optional[BaseProvider]:
        """Get current provider instance.

        Returns:
            Current provider or None if not configured
        """
        return self._current_state.provider if self._current_state else None

    def get_current_model(self) -> str:
        """Get current model name.

        Returns:
            Current model name or empty string if not configured
        """
        return self._current_state.model if self._current_state else ""

    def get_current_state(self) -> Optional[ProviderSwitcherState]:
        """Get current switcher state.

        Returns:
            Current state or None if not configured
        """
        return self._current_state

    def set_initial_state(
        self,
        provider: BaseProvider,
        provider_name: str,
        model: str,
    ) -> None:
        """Set initial provider state (used during initialization).

        Args:
            provider: Provider instance
            provider_name: Provider name
            model: Model name
        """
        self._current_state = ProviderSwitcherState(
            provider=provider,
            provider_name=provider_name.lower(),
            model=model,
        )

    async def switch_provider(
        self,
        provider_name: str,
        model: str,
        reason: str = "manual",
        settings: Optional[Any] = None,
        **provider_kwargs: Any,
    ) -> bool:
        """Switch to a different provider.

        Args:
            provider_name: Name of the provider to switch to
            model: Model identifier
            reason: Reason for the switch
            settings: Optional settings for provider configuration
            **provider_kwargs: Additional provider arguments

        Returns:
            True if switch succeeded, False otherwise
        """
        try:
            # Get provider settings if not provided
            if not provider_kwargs and settings and hasattr(settings, "get_provider_settings"):
                provider_kwargs = settings.get_provider_settings(provider_name)

            # Create new provider
            new_provider = ProviderRegistry.create(provider_name, **provider_kwargs)

            # Check health if health monitor is available
            if self._health_monitor:
                is_healthy = await self._health_monitor.check_health(new_provider)
                if not is_healthy:
                    logger.warning(f"Provider {provider_name} failed health check, aborting switch")
                    return False

            # Store old state for switch tracking
            old_state = self._current_state
            old_provider = old_state.provider_name if old_state else "none"
            old_model = old_state.model if old_state else "none"
            old_switch_count = old_state.switch_count if old_state else 0

            # Create new state
            self._current_state = ProviderSwitcherState(
                provider=new_provider,
                provider_name=provider_name.lower(),
                model=model,
                switch_count=old_switch_count + 1,
            )

            # Reinitialize tool adapter if coordinator is available
            if self._adapter_coordinator:
                self._adapter_coordinator.initialize_adapter()

            # Record switch in ModelSwitcher
            self._model_switcher.switch(
                provider=provider_name,
                model=model,
                reason=(SwitchReason.USER_REQUEST if reason == "manual" else SwitchReason.FALLBACK),
                metadata={
                    "from_provider": old_provider,
                    "from_model": old_model,
                },
            )

            # Add to switch history
            self._switch_history.append(
                {
                    "from_provider": old_provider,
                    "from_model": old_model,
                    "to_provider": provider_name,
                    "to_model": model,
                    "reason": reason,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

            # Emit switch event
            self._event_emitter.emit_switch_event(
                {
                    "from_provider": old_provider,
                    "from_model": old_model,
                    "to_provider": provider_name,
                    "to_model": model,
                    "reason": reason,
                }
            )

            # Notify callbacks
            self._notify_switch(self._current_state)

            logger.info(
                f"Switched provider: {old_provider}:{old_model} -> "
                f"{provider_name}:{model} ({reason})"
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

    async def switch_model(self, model: str, reason: str = "manual") -> bool:
        """Switch to a different model on current provider.

        Args:
            model: New model name
            reason: Reason for the switch

        Returns:
            True if switch succeeded, False otherwise
        """
        if not self._current_state:
            logger.error("No provider configured, cannot switch model")
            return False

        try:
            old_model = self._current_state.model

            # Update model in state
            self._current_state.model = model
            self._current_state.switch_count += 1

            # Reinitialize tool adapter if coordinator is available
            if self._adapter_coordinator:
                self._adapter_coordinator.initialize_adapter()

            # Record switch in ModelSwitcher
            self._model_switcher.switch(
                provider=self._current_state.provider_name,
                model=model,
                reason=(SwitchReason.USER_REQUEST if reason == "manual" else SwitchReason.FALLBACK),
                metadata={"from_model": old_model},
            )

            # Add to switch history
            self._switch_history.append(
                {
                    "from_provider": self._current_state.provider_name,
                    "from_model": old_model,
                    "to_provider": self._current_state.provider_name,
                    "to_model": model,
                    "reason": reason,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

            # Emit switch event
            self._event_emitter.emit_switch_event(
                {
                    "from_provider": self._current_state.provider_name,
                    "from_model": old_model,
                    "to_provider": self._current_state.provider_name,
                    "to_model": model,
                    "reason": reason,
                }
            )

            # Notify callbacks
            self._notify_switch(self._current_state)

            logger.info(f"Switched model: {old_model} -> {model} ({reason})")

            return True

        except Exception as e:
            logger.error(f"Failed to switch model to {model}: {e}")
            if self._current_state:
                self._current_state.last_error = str(e)
            return False

    def get_switch_history(self) -> List[Dict[str, Any]]:
        """Get history of provider/model switches.

        Returns:
            List of switch event dictionaries with keys:
            - from_provider, from_model, to_provider, to_model
            - reason, timestamp
        """
        return self._switch_history.copy()

    def on_switch(self, callback: Callable[[ProviderSwitcherState], None]) -> None:
        """Register callback for provider/model switches.

        Args:
            callback: Function to call when switch occurs
        """
        self._on_switch_callbacks.append(callback)

    def _notify_switch(self, state: ProviderSwitcherState) -> None:
        """Notify all registered switch callbacks.

        Args:
            state: Current switcher state after switch
        """
        for callback in self._on_switch_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Switch callback error: {e}")
