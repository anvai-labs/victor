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

"""Model event emitter for tracking LLM interactions.

SOLID Principles:
- SRP: Focused solely on LLM model events
- OCP: Extensible via inheritance (can add custom model tracking)
- LSP: Substitutable with IModelEventEmitter
- ISP: Implements focused interface
- DIP: Depends on ObservabilityBus abstraction, not concrete implementation

Migration Notes:
- Migrated from legacy EventBus to canonical core/events system
- Uses topic-based routing ("model.request", "model.response") instead of category-based
- Fully async API with sync wrappers for gradual migration
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from victor.observability.emitters.base import IModelEventEmitter
from victor.core.events import ObservabilityBus, SyncEventWrapper

logger = logging.getLogger(__name__)


class ModelEventEmitter(IModelEventEmitter):
    """Emits LLM model interaction events to ObservabilityBus.

    Thread-safe, performant model interaction tracking.
    Handles errors gracefully to avoid impacting model calls.

    Example:
        >>> emitter = ModelEventEmitter()
        >>> emitter.model_request("anthropic", "claude-3-5-sonnet", 1000)
        >>> # ... LLM call ...
        >>> emitter.model_response("anthropic", "claude-3-5-sonnet", 1000, 500, 1500.0)

    Migration:
        This emitter now uses the canonical core/events system instead of
        the legacy observability/event_bus.py. Events use topic-based routing
        ("model.request", "model.response") instead of category-based routing.
    """

    def __init__(self, bus: Optional[ObservabilityBus] = None):
        """Initialize the model event emitter.

        Args:
            bus: Optional ObservabilityBus instance. If None, uses DI container.
        """
        self._bus = bus
        self._sync_wrapper: Optional[SyncEventWrapper] = None
        self._enabled = True

    def _get_bus(self) -> Optional[ObservabilityBus]:
        """Get ObservabilityBus instance.

        Returns:
            ObservabilityBus instance or None if unavailable
        """
        if self._bus:
            return self._bus

        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    def _get_sync_wrapper(self) -> Optional[SyncEventWrapper]:
        """Get sync wrapper for gradual migration.

        Returns:
            SyncEventWrapper instance or None if unavailable
        """
        if self._sync_wrapper:
            return self._sync_wrapper

        bus = self._get_bus()
        if bus:
            self._sync_wrapper = SyncEventWrapper(bus.backend)
            return self._sync_wrapper

        return None

    async def emit_async(
        self,
        topic: str,
        data: Dict[str, Any],
    ) -> bool:
        """Emit a model event asynchronously.

        Args:
            topic: Event topic (e.g., "model.request", "model.response")
            data: Event payload

        Returns:
            True if emission succeeded, False otherwise
        """
        if not self._enabled:
            return False

        bus = self._get_bus()
        if bus:
            try:
                # Add category metadata for observability features
                data_with_category = {
                    **data,
                    "category": "model",
                }
                return await bus.emit(topic, data_with_category)
            except Exception as e:
                logger.debug(f"Failed to emit model event: {e}")
                return False

        return False

    def emit(
        self,
        topic: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit a model event synchronously (for gradual migration).

        This method wraps the async emit_async() method using emit_event_sync()
        so sync callers can reuse the shared async/sync bridge behavior.

        Args:
            topic: Event topic (e.g., "model.request", "model.response")
            data: Event payload
        """
        try:
            from victor.core.events.emit_helper import emit_event_sync

            bus = self._get_bus()
            if bus:
                emit_event_sync(
                    bus,
                    topic=topic,
                    data=data,
                    source="ModelEventEmitter",
                )
        except Exception as e:
            logger.debug(f"Failed to emit model event: {e}")

    async def model_request_async(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        **metadata: Any,
    ) -> bool:
        """Emit LLM request event asynchronously.

        Args:
            provider: Model provider (anthropic, openai, etc.)
            model: Model name
            prompt_tokens: Number of tokens in prompt
            **metadata: Additional metadata

        Returns:
            True if emission succeeded
        """
        return await self.emit_async(
            topic="model.request",
            data={
                "provider": provider,
                "model": model,
                "prompt_tokens": prompt_tokens,
                **metadata,
            },
        )

    def model_request(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        **metadata: Any,
    ) -> None:
        """Emit LLM request event (sync wrapper).

        Args:
            provider: Model provider (anthropic, openai, etc.)
            model: Model name
            prompt_tokens: Number of tokens in prompt
            **metadata: Additional metadata
        """
        self.emit(
            topic="model.request",
            data={
                "provider": provider,
                "model": model,
                "prompt_tokens": prompt_tokens,
                **metadata,
            },
        )

    async def model_response_async(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        **metadata: Any,
    ) -> bool:
        """Emit LLM response event asynchronously.

        Args:
            provider: Model provider
            model: Model name
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            latency_ms: Request latency in milliseconds
            **metadata: Additional metadata

        Returns:
            True if emission succeeded
        """
        total_tokens = prompt_tokens + completion_tokens

        return await self.emit_async(
            topic="model.response",
            data={
                "provider": provider,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_ms": latency_ms,
                **metadata,
            },
        )

    def model_response(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        **metadata: Any,
    ) -> None:
        """Emit LLM response event (sync wrapper).

        Args:
            provider: Model provider
            model: Model name
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            latency_ms: Request latency in milliseconds
            **metadata: Additional metadata
        """
        total_tokens = prompt_tokens + completion_tokens

        self.emit(
            topic="model.response",
            data={
                "provider": provider,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_ms": latency_ms,
                **metadata,
            },
        )

    async def model_streaming_delta_async(
        self,
        provider: str,
        model: str,
        delta: str,
        **metadata: Any,
    ) -> bool:
        """Emit streaming delta event asynchronously.

        Note: This is high-volume. Consider event sampling for production.

        Args:
            provider: Model provider
            model: Model name
            delta: Text delta from streaming response
            **metadata: Additional metadata

        Returns:
            True if emission succeeded
        """
        return await self.emit_async(
            topic="model.streaming_delta",
            data={
                "provider": provider,
                "model": model,
                "delta": delta[:500],  # Truncate for event
                "delta_length": len(delta),
                **metadata,
            },
        )

    def model_streaming_delta(
        self,
        provider: str,
        model: str,
        delta: str,
        **metadata: Any,
    ) -> None:
        """Emit streaming delta event (sync wrapper).

        Note: This is high-volume. Consider event sampling for production.

        Args:
            provider: Model provider
            model: Model name
            delta: Text delta from streaming response
            **metadata: Additional metadata
        """
        self.emit(
            topic="model.streaming_delta",
            data={
                "provider": provider,
                "model": model,
                "delta": delta[:500],  # Truncate for event
                "delta_length": len(delta),
                **metadata,
            },
        )

    async def model_error_async(
        self,
        provider: str,
        model: str,
        error: Exception,
        **metadata: Any,
    ) -> bool:
        """Emit LLM error event asynchronously.

        Args:
            provider: Model provider
            model: Model name
            error: The exception that occurred
            **metadata: Additional metadata

        Returns:
            True if emission succeeded
        """
        return await self.emit_async(
            topic="model.error",
            data={
                "provider": provider,
                "model": model,
                "error": str(error),
                "error_type": type(error).__name__,
                **metadata,
            },
        )

    def model_error(
        self,
        provider: str,
        model: str,
        error: Exception,
        **metadata: Any,
    ) -> None:
        """Emit LLM error event (sync wrapper).

        Args:
            provider: Model provider
            model: Model name
            error: The exception that occurred
            **metadata: Additional metadata
        """
        self.emit(
            topic="model.error",
            data={
                "provider": provider,
                "model": model,
                "error": str(error),
                "error_type": type(error).__name__,
                **metadata,
            },
        )

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if event emission is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self._enabled
