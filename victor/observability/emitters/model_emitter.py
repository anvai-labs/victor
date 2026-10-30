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

from typing import Any

from victor.observability.emitters.base import BaseEventEmitter, IModelEventEmitter


class ModelEventEmitter(BaseEventEmitter, IModelEventEmitter):
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

    _CATEGORY = "model"

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
