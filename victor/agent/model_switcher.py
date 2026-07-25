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

"""Model hot-swap capability for switching models mid-conversation.

This module provides functionality to switch between LLM models during
an active conversation while preserving context and history.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SwitchReason(Enum):
    """Reasons for switching models."""

    USER_REQUEST = "user_request"  # User explicitly requested switch
    PERFORMANCE = "performance"  # Auto-switch for performance
    COST = "cost"  # Auto-switch for cost optimization
    CAPABILITY = "capability"  # Auto-switch for specific capability
    FALLBACK = "fallback"  # Fallback due to error/unavailability
    LOAD_BALANCING = "load_balancing"  # Distribute load across models


@dataclass
class ModelSwitchEvent:
    """Records a model switch event."""

    from_provider: str
    from_model: str
    to_provider: str
    to_model: str
    reason: SwitchReason
    timestamp: datetime = field(default_factory=datetime.now)
    context_preserved: bool = True
    message_count: int = 0  # Messages at time of switch
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Information about an available model."""

    provider: str
    model_id: str
    display_name: str
    context_window: int = 8192
    supports_tools: bool = True
    supports_streaming: bool = True
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    capabilities: List[str] = field(default_factory=list)
    is_local: bool = False

    @property
    def full_id(self) -> str:
        """Get full model identifier (provider:model)."""
        return f"{self.provider}:{self.model_id}"


class ModelSwitcher:
    """Manages model switching during conversations.

    Features:
    - Hot-swap models without losing conversation context
    - Track switch history
    - Auto-switch based on task requirements
    - Support for fallback chains
    """

    def __init__(self):
        """Initialize the model switcher."""
        self._current_provider: Optional[str] = None
        self._current_model: Optional[str] = None
        self._switch_history: List[ModelSwitchEvent] = []
        self._callbacks: List[Callable[[ModelSwitchEvent], None]] = []
        self._available_models: Dict[str, ModelInfo] = {}
        self._fallback_chain: List[str] = []
        self._message_count: int = 0

        # Register default models
        self._register_default_models()

    def _register_default_models(self) -> None:
        """Register commonly available models."""
        # Anthropic models
        self.register_model(
            ModelInfo(
                provider="anthropic",
                model_id="claude-sonnet-4-20250514",
                display_name="Claude Sonnet 4",
                context_window=200000,
                supports_tools=True,
                capabilities=["code", "analysis", "reasoning"],
            )
        )
        self.register_model(
            ModelInfo(
                provider="anthropic",
                model_id="claude-opus-4-5-20251101",
                display_name="Claude Opus 4.5",
                context_window=200000,
                supports_tools=True,
                capabilities=["code", "analysis", "reasoning", "complex"],
            )
        )

        # OpenAI models
        self.register_model(
            ModelInfo(
                provider="openai",
                model_id="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                context_window=128000,
                supports_tools=True,
                capabilities=["code", "analysis"],
            )
        )
        self.register_model(
            ModelInfo(
                provider="openai",
                model_id="gpt-4o",
                display_name="GPT-4o",
                context_window=128000,
                supports_tools=True,
                capabilities=["code", "analysis", "vision"],
            )
        )

        # Google models
        self.register_model(
            ModelInfo(
                provider="google",
                model_id="gemini-2.0-flash",
                display_name="Gemini 2.0 Flash",
                context_window=1000000,
                supports_tools=True,
                capabilities=["code", "analysis", "fast"],
            )
        )

        # Local models (Ollama)
        self.register_model(
            ModelInfo(
                provider="ollama",
                model_id="qwen2.5-coder:14b",
                display_name="Qwen 2.5 Coder 14B",
                context_window=32768,
                supports_tools=True,
                is_local=True,
                capabilities=["code", "local"],
            )
        )
        self.register_model(
            ModelInfo(
                provider="ollama",
                model_id="llama3.1:8b",
                display_name="Llama 3.1 8B",
                context_window=32768,
                supports_tools=True,
                is_local=True,
                capabilities=["general", "local", "fast"],
            )
        )

    def register_model(self, model: ModelInfo) -> None:
        """Register a model as available.

        Cost fields left at zero are hydrated from the canonical pricing
        config (config/provider_metrics.yaml) so cheapest-model selection
        sorts on real prices instead of a hand-maintained copy. Callers that
        pass explicit costs keep them.

        Args:
            model: Model information
        """
        if model.cost_per_1k_input == 0.0 and model.cost_per_1k_output == 0.0:
            try:
                from victor.config.metrics_capabilities import get_metrics_capabilities

                capabilities = get_metrics_capabilities(model.provider, model.model_id)
                if capabilities.cost_enabled:
                    model.cost_per_1k_input = capabilities.input_cost_per_mtok / 1000
                    model.cost_per_1k_output = capabilities.output_cost_per_mtok / 1000
            except Exception:
                logger.debug("Canonical pricing lookup failed for %s", model.full_id, exc_info=True)
        self._available_models[model.full_id] = model
        logger.debug(f"Registered model: {model.full_id}")

    def set_current(self, provider: str, model: str) -> None:
        """Set the current model without recording a switch.

        Used for initial setup.

        Args:
            provider: Provider name
            model: Model ID
        """
        self._current_provider = provider
        self._current_model = model
        logger.info(f"Current model set to {provider}:{model}")

    @property
    def current_provider(self) -> Optional[str]:
        """Get current provider."""
        return self._current_provider

    @property
    def current_model(self) -> Optional[str]:
        """Get current model."""
        return self._current_model

    @property
    def current_full_id(self) -> Optional[str]:
        """Get current full model ID."""
        if self._current_provider and self._current_model:
            return f"{self._current_provider}:{self._current_model}"
        return None

    def switch(
        self,
        provider: str,
        model: str,
        reason: SwitchReason = SwitchReason.USER_REQUEST,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Switch to a different model.

        Args:
            provider: Target provider
            model: Target model ID
            reason: Reason for the switch
            metadata: Additional metadata

        Returns:
            True if switch was successful
        """
        # Record the switch
        event = ModelSwitchEvent(
            from_provider=self._current_provider or "",
            from_model=self._current_model or "",
            to_provider=provider,
            to_model=model,
            reason=reason,
            context_preserved=True,
            message_count=self._message_count,
            metadata=metadata or {},
        )

        old_provider = self._current_provider
        old_model = self._current_model

        # Update current
        self._current_provider = provider
        self._current_model = model

        # Add to history
        self._switch_history.append(event)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Switch callback error: {e}")

        logger.info(
            f"Switched from {old_provider}:{old_model} to {provider}:{model} "
            f"(reason: {reason.value})"
        )
        return True

    def switch_by_id(
        self,
        full_id: str,
        reason: SwitchReason = SwitchReason.USER_REQUEST,
    ) -> bool:
        """Switch to a model by full ID.

        Args:
            full_id: Full model ID (provider:model)
            reason: Reason for switch

        Returns:
            True if successful
        """
        if ":" in full_id:
            provider, model = full_id.split(":", 1)
        else:
            # Assume it's just a model name, try to find it
            model_info = self.find_model(full_id)
            if model_info:
                provider = model_info.provider
                model = model_info.model_id
            else:
                logger.warning(f"Model not found: {full_id}")
                return False

        return self.switch(provider, model, reason)

    def find_model(self, query: str) -> Optional[ModelInfo]:
        """Find a model by name or partial ID.

        Args:
            query: Model name or partial ID

        Returns:
            ModelInfo if found
        """
        query_lower = query.lower()

        # Exact match
        if query in self._available_models:
            return self._available_models[query]

        # Search by display name or model ID
        for _model_id, info in self._available_models.items():
            if query_lower in info.display_name.lower() or query_lower in info.model_id.lower():
                return info

        return None

    def get_switch_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent switch history.

        Args:
            limit: Maximum entries to return

        Returns:
            List of switch events as dicts
        """
        return [
            {
                "from": f"{e.from_provider}:{e.from_model}",
                "to": f"{e.to_provider}:{e.to_model}",
                "reason": e.reason.value,
                "timestamp": e.timestamp.isoformat(),
                "messages_at_switch": e.message_count,
            }
            for e in self._switch_history[-limit:]
        ]

    def get_available_models(
        self,
        provider: Optional[str] = None,
        local_only: bool = False,
        capability: Optional[str] = None,
    ) -> List[ModelInfo]:
        """Get list of available models.

        Args:
            provider: Filter by provider
            local_only: Only return local models
            capability: Filter by capability

        Returns:
            List of matching models
        """
        models = list(self._available_models.values())

        if provider:
            models = [m for m in models if m.provider == provider]

        if local_only:
            models = [m for m in models if m.is_local]

        if capability:
            models = [m for m in models if capability in m.capabilities]

        return models

    def register_callback(self, callback: Callable[[ModelSwitchEvent], None]) -> None:
        """Register a callback for model switches.

        Args:
            callback: Function to call on switch
        """
        self._callbacks.append(callback)

    def set_fallback_chain(self, models: List[str]) -> None:
        """Set fallback chain for error recovery.

        Args:
            models: List of model IDs in fallback order
        """
        self._fallback_chain = models
        logger.debug(f"Fallback chain set: {models}")

    def get_fallback(self) -> Optional[str]:
        """Get next model in fallback chain.

        Returns:
            Next model ID or None
        """
        current = self.current_full_id
        if not current or not self._fallback_chain:
            return None

        try:
            idx = self._fallback_chain.index(current)
            if idx + 1 < len(self._fallback_chain):
                return self._fallback_chain[idx + 1]
        except ValueError:
            # Current not in chain, return first
            if self._fallback_chain:
                return self._fallback_chain[0]

        return None

    def switch_to_fallback(self) -> bool:
        """Switch to next fallback model.

        Returns:
            True if switched successfully
        """
        fallback = self.get_fallback()
        if fallback:
            return self.switch_by_id(fallback, SwitchReason.FALLBACK)
        return False

    def update_message_count(self, count: int) -> None:
        """Update the current message count.

        Args:
            count: Current message count
        """
        self._message_count = count

    def get_model_for_task(self, task_type: str) -> Optional[ModelInfo]:
        """Get recommended model for a task type.

        Args:
            task_type: Type of task (e.g., "code", "analysis", "fast")

        Returns:
            Recommended model or None
        """
        # Find models with matching capability
        matching = [m for m in self._available_models.values() if task_type in m.capabilities]

        if not matching:
            return None

        # Sort by cost (cheapest first for equivalent capability)
        matching.sort(key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)
        return matching[0]

    def suggest_model(
        self,
        context_size: int = 0,
        needs_tools: bool = True,
        prefer_local: bool = False,
        max_cost_per_1k: float = 0.0,
    ) -> Optional[ModelInfo]:
        """Suggest a model based on requirements.

        Args:
            context_size: Required context window size
            needs_tools: Whether tool calling is required
            prefer_local: Prefer local models
            max_cost_per_1k: Maximum cost per 1k tokens (0 = no limit)

        Returns:
            Suggested model or None
        """
        candidates = list(self._available_models.values())

        # Filter by requirements
        if context_size > 0:
            candidates = [m for m in candidates if m.context_window >= context_size]

        if needs_tools:
            candidates = [m for m in candidates if m.supports_tools]

        if max_cost_per_1k > 0:
            candidates = [
                m
                for m in candidates
                if (m.cost_per_1k_input + m.cost_per_1k_output) <= max_cost_per_1k
            ]

        if not candidates:
            return None

        # Sort by preference
        if prefer_local:
            # Local first, then by cost
            candidates.sort(key=lambda m: (not m.is_local, m.cost_per_1k_input))
        else:
            # By cost
            candidates.sort(key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)

        return candidates[0]

    def get_status(self) -> Dict[str, Any]:
        """Get current switcher status.

        Returns:
            Status dictionary
        """
        model_info = None
        if self.current_full_id:
            model_info = self._available_models.get(self.current_full_id)

        return {
            "current_provider": self._current_provider,
            "current_model": self._current_model,
            "current_display_name": model_info.display_name if model_info else None,
            "switch_count": len(self._switch_history),
            "message_count": self._message_count,
            "available_models": len(self._available_models),
        }


# Global instance
_model_switcher: Optional[ModelSwitcher] = None


def get_model_switcher() -> ModelSwitcher:
    """Get or create the global model switcher."""
    global _model_switcher
    if _model_switcher is None:
        _model_switcher = ModelSwitcher()
    return _model_switcher


def set_model_switcher(switcher: ModelSwitcher) -> None:
    """Set the global model switcher instance."""
    global _model_switcher
    _model_switcher = switcher


def reset_model_switcher() -> None:
    """Reset the global model switcher (for testing)."""
    global _model_switcher
    _model_switcher = None
